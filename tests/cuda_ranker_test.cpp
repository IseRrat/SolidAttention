#include "solid/block_ranker.hpp"
#include "representative_builder_cases.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F operation) {
    bool failed = false;
    try { operation(); } catch (const std::exception &) { failed = true; }
    require(failed, "invalid CUDA ranking request was accepted");
}
void run(int device,std::uint32_t dimension) {
    const solid::KvGeometry g{2, 2, dimension, 32, 32768, 64};
    constexpr std::uint32_t queries = 6, representatives = 3, budget = 1024, blocks = 513;
    auto ranker = solid::make_cuda_block_ranker(g, queries, representatives, budget, device);
    const auto initial_bytes = ranker->stats().device_bytes;
    const auto stride = static_cast<std::size_t>(g.heads) * representatives * g.dimension;
    std::mt19937 random(2171);
    std::vector<solid::Half> keys(blocks * stride);
    for (auto & k : keys) k = solid::to_half((static_cast<int>(random()%101)-50)/16.0f);
    ranker->publish(0, 0, 255, {keys.begin(), keys.begin() + 255*stride});
    ranker->publish(0, 255, blocks-255, {keys.begin() + 255*stride, keys.end()});
    ranker->publish(1, 0, blocks, keys);
    const auto steady_bytes = ranker->stats().device_bytes;
    const auto bank_bytes = 2 * g.blocks_per_head() * stride * sizeof(solid::Half);
    require(steady_bytes == initial_bytes + bank_bytes, "CUDA bank allocation accounting");
    std::uint64_t calls = 0, scored = 0;
    for (const std::size_t count : {0,1,31,255,256,257,513}) for (const std::uint32_t left : {0,1,37,512,1024}) {
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> q(queries*g.dimension);
            for (auto & value : q) value = mode == 0 ? 0 : (static_cast<int>(random()%61)-30) * (mode == 1 ? 0.125f : 1e25f);
            solid::BlockCandidates candidates;
            candidates.remaining = left;
            for (std::uint64_t block = 0; block < count; ++block) {
                candidates.blocks.push_back(block);
                candidates.tokens.push_back(mode == 0 ? g.block_tokens : static_cast<std::uint32_t>(block%32+1));
            }
            const auto actual = ranker->rank(mode%2, q, candidates);
            require(actual.size() == g.heads, "CUDA rank head count");
            if (count && left) { ++calls; scored += count * g.heads; }
            for (std::uint32_t head = 0; head < g.heads; ++head) {
                std::vector<std::pair<double,std::uint64_t>> scores;
                for (const auto block : candidates.blocks) {
                    double sum = 0;
                    for (std::uint32_t r = 0; r < representatives; ++r) {
                        for (std::uint32_t query = head*3; query < (head+1)*3; ++query) {
                            for (std::uint32_t d = 0; d < g.dimension; ++d) {
                                sum += static_cast<double>(q[query*g.dimension+d]) *
                                    solid::from_half(keys[block*stride+(head*representatives+r)*g.dimension+d])/3;
                            }
                        }
                    }
                    scores.emplace_back(sum, block);
                }
                std::sort(scores.begin(), scores.end(), [](auto a, auto b) {
                    return a.first == b.first ? a.second < b.second : a.first > b.first;
                });
                std::vector<std::uint64_t> expected;
                std::uint32_t covered = 0;
                for (const auto & score : scores) {
                    if (covered >= left) break;
                    expected.push_back(score.second);
                    covered += candidates.tokens[score.second];
                }
                require(actual[head] == expected, "CUDA score/top-k differs from independent full-sort oracle");
            }
            require(ranker->stats().device_bytes == steady_bytes, "CUDA rank allocations grew with calls");
        }
    }
    require(ranker->stats().calls == calls && ranker->stats().scored_blocks == scored &&
            ranker->stats().published_blocks == blocks*2, "CUDA ranking counters");
    std::vector<float> query(queries*g.dimension,1);
    rejects([&] { (void) ranker->rank(0, query, {{513},{1},1}); });
    rejects([&] { (void) ranker->rank(0, query, {{0,0},{1,1},1}); });
    rejects([&] { (void) ranker->rank(0, query, {{0},{0},1}); });
    rejects([&] { (void) ranker->rank(0, query, {{0},{33},1}); });
    rejects([&] { (void) ranker->rank(0, query, {{0},{1},1025}); });
    rejects([&] { ranker->publish(0,0,1,{keys.begin(),keys.begin()+stride}); });
    auto invalid = std::vector<solid::Half>(stride,0x7c00);
    rejects([&] { ranker->publish(0,513,1,invalid); });
    query[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void) ranker->rank(0, query, {{0},{1},1}); });
}

void shared_topk(int device) {
    const solid::KvGeometry g{1, 4, 32, 32, 262144, 64};
    constexpr std::uint32_t query_heads = 28, budget = 1024;
    auto ranker = solid::make_cuda_block_ranker(g, query_heads, 1, budget, device);
    const auto blocks = g.blocks_per_head();
    std::vector<solid::Half> keys(blocks * g.heads * g.dimension, 0);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        keys[b * g.heads * g.dimension] = solid::to_half(static_cast<float>(static_cast<int>((b * 37) % 4096) - 2048));
    }
    ranker->publish(0, 0, blocks, keys);
    const auto bytes = ranker->stats().device_bytes;
    std::vector<float> query(query_heads * g.dimension, 0);
    for (const auto count : {31u, 513u, 4096u, 8192u}) for (const auto left : {1u, 37u, 512u, 1024u}) {
        for (const auto sign : {-1.f, 0.f, 1.f}) for (const bool uniform : {false, true}) {
            query[0] = sign;
            solid::BlockCandidates candidates;
            candidates.remaining = left;
            candidates.shared_heads = true;
            std::vector<std::pair<float,std::uint64_t>> scores;
            for (std::uint64_t b = 0; b < count; ++b) {
                candidates.blocks.push_back(b);
                candidates.tokens.push_back(uniform ? g.block_tokens : b % g.block_tokens + 1);
                scores.emplace_back(sign * solid::from_half(keys[b * g.heads * g.dimension]), b);
            }
            std::sort(scores.begin(), scores.end(), [](auto a, auto b) {
                return a.first == b.first ? a.second < b.second : a.first > b.first;
            });
            std::vector<std::uint64_t> expected;
            std::uint32_t covered = 0;
            for (const auto & item : scores) {
                if (covered >= left) break;
                expected.push_back(item.second);
                covered += candidates.tokens[item.second];
            }
            const auto selected = ranker->rank(0, query, candidates);
            for (const auto & head : selected) require(head == expected, "shared CUB/fallback Top-K differs from stable full-sort oracle");
            require(ranker->stats().device_bytes == bytes, "shared Top-K workspace grew during retrieval");
        }
    }
}
} // namespace
int main(int argc, char ** argv) {
    // Device checks are opt-in and excluded from local CTest/check.sh runs.
    if (argc < 2 || std::string(argv[1]) != "--run-device-check") {
        std::cout << "Usage: solid-cuda-ranker-test --run-device-check [device ordinal]\n";
        return argc == 1 ? 0 : 2;
    }
    try {
        if (argc > 3) throw std::invalid_argument("unexpected device-check argument");
        const auto device = argc == 3 ? std::stoi(argv[2]) : 0;
        for (const auto dimension:{7u,32u,33u,128u}) run(device,dimension);
        shared_topk(device);
        test_builder::run_cases([&](auto geometry,auto query_heads,auto budget) {
            return solid::make_cuda_block_ranker(geometry,query_heads,budget.representatives,budget.total,device,budget.local);
        });
        std::cout << "CUDA block ranking device checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
