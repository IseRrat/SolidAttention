#include "solid/attention.hpp"
#include "solid/session.hpp"
#include "solid/ranking_math.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

namespace {
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F operation) {
    bool failed = false;
    try { operation(); } catch (const std::exception &) { failed = true; }
    require(failed, "ranker accepted invalid operation");
}
// Independent host implementation of the device contract. Retains the published
// payload, not a reference to RepresentativeIndex, to expose stale publications.
class OracleRanker final : public solid::BlockRanker {
public:
    OracleRanker(solid::KvGeometry g, std::uint32_t q, std::uint32_t r) : geometry(g), query_heads(q), representatives(r) {}
    void publish(std::uint32_t layer, std::uint64_t first, std::uint64_t count, const std::vector<solid::Half> & keys) override {
        const auto block_elements = static_cast<std::size_t>(geometry.heads) * representatives * geometry.dimension;
        require(first == published[layer] && keys.size() == count * block_elements, "publication gap or payload shape");
        for (std::uint64_t i = 0; i < count; ++i) {
            banks[{layer, first + i}] = {keys.begin() + i * block_elements, keys.begin() + (i + 1) * block_elements};
        }
        published[layer] += count;
        publications += count;
    }
    std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                const solid::BlockCandidates & candidates) override {
        ++calls;
        require(candidates.remaining && !candidates.blocks.empty(), "empty work reached device");
        std::vector<std::vector<std::uint64_t>> result(geometry.heads);
        const auto group = query_heads / geometry.heads;
        for (std::uint32_t head = 0; head < geometry.heads; ++head) {
            std::vector<std::pair<long double, std::size_t>> scored;
            for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
                const auto & block = banks.at({layer, candidates.blocks[i]});
                long double score = 0;
                for (std::uint32_t r = 0; r < representatives; ++r) {
                    for (std::uint32_t q = 0; q < group; ++q) for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                        score += static_cast<long double>(query[(head * group + q) * geometry.dimension + d]) *
                            solid::from_half(block[(head * representatives + r) * geometry.dimension + d]) / group;
                    }
                }
                scored.emplace_back(score, i);
            }
            std::sort(scored.begin(), scored.end(), [&](auto a, auto b) {
                return a.first == b.first ? candidates.blocks[a.second] < candidates.blocks[b.second] : a.first > b.first;
            });
            std::uint32_t covered = 0;
            for (const auto & item : scored) {
                result[head].push_back(candidates.blocks[item.second]);
                covered += candidates.tokens[item.second];
                if (covered >= candidates.remaining) break;
            }
        }
        if (corrupt == 1) result.pop_back();
        if (corrupt == 2) result[0].push_back(result[0].front());
        if (corrupt == 3) result[0][0] = UINT64_MAX;
        if (corrupt == 4) result[0].clear();
        if (corrupt == 5) throw std::runtime_error("injected ranking failure");
        return result;
    }
    solid::RankerStats stats() const override { return {123, 45, calls, 0, publications}; }
    const char * name() const override { return "host_contract_oracle"; }
    solid::KvGeometry geometry;
    std::uint32_t query_heads, representatives;
    std::map<std::pair<std::uint32_t, std::uint64_t>, std::vector<solid::Half>> banks;
    std::map<std::uint32_t, std::uint64_t> published;
    std::uint64_t calls = 0, publications = 0;
    int corrupt = 0;
};
void identical(const std::vector<solid::HeadSelection> & actual, const std::vector<solid::HeadSelection> & expected) {
    require(actual.size() == expected.size(), "head count");
    for (std::size_t h = 0; h < actual.size(); ++h) {
        require(actual[h].positions == expected[h].positions && actual[h].blocks == expected[h].blocks,
                "offloaded selection differs from independent CPU path");
    }
}
void incremental(bool late_attach, bool ties, std::uint32_t count) {
    const solid::KvGeometry g{2, 3, 9, 4, 65, 64, solid::KvLayout::interleaved};
    const solid::SparseBudget budget{21, 3, 5, count};
    solid::RepresentativeIndex reference(g, 6, budget), offloaded(g, 6, budget);
    auto oracle = std::make_unique<OracleRanker>(g, 6, count);
    auto * inspect = oracle.get();
    if (!late_attach) offloaded.set_ranker(std::move(oracle));
    std::mt19937 random(1917);
    std::vector<float> query(54);
    std::vector<solid::Half> keys(27);
    for (std::uint32_t p = 0; p < g.capacity_tokens; ++p) {
        if (late_attach && p == 25) offloaded.set_ranker(std::move(oracle));
        for (std::uint32_t layer = 0; layer < g.layers; ++layer) {
            for (auto & q : query) q = ties ? 0.0f : (static_cast<int>(random() % 41) - 20) / 16.0f;
            for (auto & k : keys) k = solid::to_half((static_cast<int>(random() % 61) - 30) / 8.0f);
            // The early-Q frontier finalizes old blocks before current K arrives.
            reference.observe_query(layer, p, query);
            offloaded.observe_query(layer, p, query);
            identical(solid::select_blocks(g, budget, offloaded, layer, p, query),
                      solid::select_blocks(g, budget, reference, layer, p, query));
            reference.append_keys(layer, p, keys);
            offloaded.append_keys(layer, p, keys);
        }
    }
    require(inspect->calls > 0 && inspect->publications == 30, "missing device calls or incremental finalized blocks");
    require(offloaded.ranker_stats().device_bytes == 123, "device memory accounting");
    rejects([&] { offloaded.set_ranker(std::make_unique<OracleRanker>(g, 6, count)); });
    // Neither the last partial block nor tokens still in the local window publish.
    require(inspect->published[0] == 15 && inspect->published[1] == 15, "tail publication boundary");
}
void corrupt_outputs() {
    const solid::KvGeometry g{1, 1, 1, 2, 32, 64};
    const solid::SparseBudget budget{7, 1, 1, 1};
    solid::RepresentativeIndex index(g, 2, budget);
    for (int p = 0; p < 15; ++p) index.observe(0, p, {1, 1}, {solid::to_half(p)});
    auto oracle = std::make_unique<OracleRanker>(g, 2, 1);
    auto * inspect = oracle.get();
    index.set_ranker(std::move(oracle));
    for (int corruption = 1; corruption <= 5; ++corruption) {
        inspect->corrupt = corruption;
        rejects([&] { (void) solid::select_blocks(g, budget, index, 0, 14, {1, 1}); });
    }
    inspect->corrupt = 0;
    rejects([&] { (void) index.rank_blocks(0, {1, 1}, {{0,0},{1,1},1}); });
    rejects([&] { (void) index.rank_blocks(0, {1, 1}, {{0},{0},1}); });
    rejects([&] { (void) index.rank_blocks(0, {1, 1}, {{15},{1},1}); });
    rejects([&] { (void) index.rank_blocks(0, {1, 1}, {{0},{1},8}); });
}
void session_failure() {
    const solid::KvGeometry g{1, 1, 2, 2, 32, 64};
    const solid::SparseBudget budget{4, 1, 1, 1};
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g), 2, budget, true, 10);
    for (std::uint32_t p = 0; p < 10; ++p) (void) session.process(0, p, 1, {1,1,1,1}, {1,2}, {3,4}, 0.5f);
    auto oracle = std::make_unique<OracleRanker>(g, 2, 1);
    oracle->corrupt = 5;
    session.set_ranker(std::move(oracle));
    require(session.stats().selection_backend == "host_contract_oracle", "actual backend identity");
    rejects([&] { (void) session.gather(0, 10, {1,1,1,1}, {1,2}, {3,4}); });
    rejects([&] { (void) session.gather(0, 11, {1,1,1,1}, {1,2}, {3,4}); });
}
void reduction_policy() {
    using solid::RankChoice;
    const RankChoice empty{0, UINT64_MAX, UINT64_MAX};
    require(solid::better_choice(empty, {-1e300,7,1}).block == 7, "negative finite winner");
    require(solid::better_choice({0,8,0}, {0,3,1}).block == 3, "logical-ID tie break");
    require(solid::better_choice({1,8,0}, {0,3,1}).block == 8, "higher relevance winner");
    // Emulate the CUDA CTA tournament on the CPU, including unused lanes.
    std::mt19937 random(731);
    for (const std::size_t count : {1, 31, 256, 257, 4097}) {
        std::vector<RankChoice> choices(256, empty), input;
        for (std::size_t i = 0; i < count; ++i) input.push_back({static_cast<double>(random()%19)-9, i*2+1, i});
        auto sorted = input;
        std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a.score == b.score ? a.block < b.block : a.score > b.score; });
        for (const auto & expected : sorted) {
            std::fill(choices.begin(), choices.end(), empty);
            for (std::size_t lane = 0; lane < 256; ++lane) for (auto i = lane; i < input.size(); i += 256) {
                choices[lane] = solid::better_choice(choices[lane], input[i]);
            }
            for (std::size_t offset = 128; offset; offset /= 2) for (std::size_t lane = 0; lane < offset; ++lane) {
                choices[lane] = solid::better_choice(choices[lane], choices[lane + offset]);
            }
            require(choices[0].block == expected.block, "CTA tournament differs from full-sort oracle");
            input[choices[0].candidate] = empty;
        }
    }
}
} // namespace
int main() {
    try {
        reduction_policy();
        for (bool late : {false,true}) for (bool ties : {false,true}) for (std::uint32_t r : {1,2,4}) incremental(late,ties,r);
        corrupt_outputs(); session_failure();
        std::cout << "block ranking contracts passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
