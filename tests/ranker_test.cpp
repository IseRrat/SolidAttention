// CPU 稀疏预算、代表评分、固定窗口与历史块选择检查。
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
    bool owns_representative_keys() const override { return true; }
    std::vector<solid::Half> read_representatives(std::uint32_t layer,std::uint64_t block) override {
        return banks.at({layer,block});
    }
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
    const auto finalized_keys=static_cast<std::uint64_t>(inspect->publications)*g.heads*count*g.dimension*sizeof(solid::Half);
    require(offloaded.resident_bytes()+finalized_keys<=reference.resident_bytes(),"published K mirror survived device ownership transfer");
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
void shared_history_blocks() {
    const solid::KvGeometry g{1, 2, 1, 4, 32, 64};
    const solid::SparseBudget budget{11, 3, 3, 1, true};
    solid::RepresentativeIndex index(g, 4, budget);
    const std::vector<float> query(4, 1);
    for (std::uint64_t p = 0; p < 26; ++p) {
        const auto left = p >= 3 && p < 7 ? 10.f : 1.f;
        const auto right = p >= 11 && p < 15 ? 20.f : 1.f;
        index.observe(0, p, query, {solid::to_half(left), solid::to_half(right)});
    }
    require(index.representative_positions(0, 0, 0) == std::vector<std::uint64_t>{3},
            "first historical unit starts after init, even without storage alignment");
    const auto result = solid::select_blocks(g, budget, index, 0, 22, query);
    const std::vector<std::uint64_t> tokens{0, 1, 2, 11, 12, 13, 14, 19, 20, 21, 22};
    require(result[0].positions == tokens && result[1].positions == tokens, "all heads must select the same complete unit");
    require(result[0].blocks == std::vector<std::uint64_t>{0, 2, 3, 4, 5},
            "logical window needs the corresponding physical storage blocks");
    const auto remainder = solid::select_blocks(g, budget, index, 0, 25, query);
    require(remainder[0].positions == std::vector<std::uint64_t>{0, 1, 2, 19, 20, 21, 22, 23, 24, 25},
            "the block completed during this query is still consumed as a remainder");

    auto multi_key_budget = budget;
    multi_key_budget.representatives = 2;
    solid::RepresentativeIndex per_query(g, 4, multi_key_budget);
    for (std::uint64_t p = 0; p < 12; ++p) {
        per_query.observe(0, p, {1, -1, 1, 1}, {solid::to_half(static_cast<float>(p)), solid::to_half(0)});
    }
    require(per_query.representative_positions(0, 0, 0) == std::vector<std::uint64_t>{6, 5} &&
            per_query.representative_positions(0, 1, 0) == std::vector<std::uint64_t>{3, 4},
            "query heads sharing a KV head can choose different representative keys");
    require(per_query.score(0, 0, 0, query) == 5.5 && per_query.score(0, 1, 0, query) == 3.5,
            "each query head ranks against the mean of its own representative keys");
}
void bounded_prefill_chunk() {
    const solid::KvGeometry g{1, 1, 1, 4, 40, 64};
    const solid::SparseBudget budget{16, 2, 5, 1, true, true};
    solid::RepresentativeIndex index(g, 1, budget);
    for (std::uint64_t p = 0; p < 16; ++p) {
        index.observe(0, p, {1}, {solid::to_half(static_cast<float>(p))});
        const auto visible = std::min<std::uint64_t>(p + 1, budget.local);
        index.observe_probabilities(0, p, std::vector<float>(visible, 1.f / visible));
    }
    const auto positions = solid::select_prefill_context(g, budget, index, 0, 16, 3, {1});
    require(positions.positions == std::vector<std::uint64_t>{0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18} &&
            positions.local_only_from == 8, "prefill plan separates fixed history from sliding local rows");
    rejects([&] { (void) solid::select_prefill_context(g, budget, index, 0, 16, 6, {1}); });
}
void long_prefill_chunk() {
    const solid::KvGeometry g{1, 1, 2, 32, 1024, 64};
    const solid::SparseBudget budget{128, 32, 32, 1, true, true};
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g), 1, budget, true, 1024,
                                    {}, {}, {}, 0, {}, solid::RelativeRope{2, 10000.f, false});
    const std::vector<float> zero(1024);
    for (int first : {0, 512}) {
        std::vector<float> values;
        for (int p = first; p < first + 512; ++p) values.insert(values.end(), {static_cast<float>(p), 0.f});
        const auto output = session.process(0, first, 512, zero, zero, values, 1.f);
        if (first == 512) {
            // Keep initial tokens, the first selected block, the chunk-entry
            // remainder, and only the final query's own recent window.
            require(std::abs(output[1022] - 35216.f / 97.f) < 1e-3f,
                    "Prefill did not refresh its selection at the local-window group boundary");
        }
    }
    require(session.stats().selected_tokens <= 1024 * budget.total, "chunk exceeded the per-query budget");
    require(session.stats().infllm_prefill_group_token_limit == 32 && session.stats().infllm_prefill_groups == 32 &&
            session.stats().infllm_prefill_rank_groups == 28, "Prefill retrieval cadence differs from its local-window group size");
}
void prefill_budget_boundary() {
    const solid::KvGeometry g{1, 1, 2, 4, 40, 64};
    const solid::SparseBudget budget{16, 2, 5, 1, true, true};
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g), 1, budget, true, 18,
                                    {}, {}, {}, 0, {}, solid::RelativeRope{2, 10000.f, false});
    std::vector<float> query(28), keys(28), values;
    for (int p = 0; p < 14; ++p) values.insert(values.end(), {static_cast<float>(p), static_cast<float>(p)});
    (void) session.process(0, 0, 14, query, keys, values, 1.f);

    const auto output = session.process(0, 14, 4, std::vector<float>(8), std::vector<float>(8),
                                        {14, 14, 15, 15, 16, 16, 17, 17}, 1.f);
    require(std::abs(output[0] - 7.f) < 1e-6f && std::abs(output[2] - 7.5f) < 1e-6f,
            "prefill lost visible history before crossing its token budget");
}
void prefill_representative_scores() {
    const solid::KvGeometry geometry{1, 1, 2, 4, 24, 64};
    const solid::SparseBudget budget{12, 1, 4, 2, true, true};
    solid::RepresentativeIndex scalar(geometry, 2, budget), batched(geometry, 2, budget);
    for (std::uint64_t first : {0, 3, 6, 9, 12, 15}) {
        const auto origin = first + 1 - std::min<std::uint64_t>(first + 1, budget.local);
        const auto end = first + 3;
        std::vector<float> scores((end - origin) * 2);
        std::vector<solid::Half> rows;
        for (auto position = first; position < end; ++position) {
            const auto count = std::min<std::uint64_t>(position + 1, budget.local);
            const auto key = solid::to_half(static_cast<float>(position));
            rows.insert(rows.end(), {key, 0, 0, 0});
            scalar.observe(0, position, {1, 0, 1, 0}, {key, 0});
            std::vector<float> probabilities(count * 2);
            for (std::size_t head = 0; head < 2; ++head) for (std::uint64_t i = 0; i < count; ++i) {
                const auto score = static_cast<float>((head ? count - i : i + 1)) / 16.f;
                probabilities[head * count + i] = score;
                scores[(position + 1 - count + i - origin) * 2 + head] += score;
            }
            scalar.observe_probabilities(0, position, probabilities);
        }
        batched.observe_prefill_scores(0, first, 3, rows, scores);
        for (std::uint64_t block = 0; block < geometry.blocks_per_head(); ++block) {
            require(scalar.ready(0, block) == batched.ready(0, block), "chunk published wrong representative blocks");
            if (scalar.ready(0, block)) for (int head = 0; head < 2; ++head) {
                require(scalar.representative_positions(0, head, block) == batched.representative_positions(0, head, block),
                        "per-key chunk score changed the representatives");
            }
        }
    }
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
        corrupt_outputs(); shared_history_blocks(); bounded_prefill_chunk(); long_prefill_chunk(); prefill_budget_boundary();
        prefill_representative_scores(); session_failure();
        std::cout << "block ranking contracts passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
