// 读取原版 InfLLM 对照数据，核对代表、窗口与选块结果的通用样例。
#pragma once
#include "solid/attention.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <set>
#include <string>

namespace test_infllm {
template<class T> T read(std::istream & input) {
    T value{};
    if (!input.read(reinterpret_cast<char *>(&value), sizeof(value))) throw std::runtime_error("truncated InfLLM fixture");
    return value;
}
template<class T> std::vector<T> read(std::istream & input, std::size_t count) {
    std::vector<T> result(count);
    if (count && !input.read(reinterpret_cast<char *>(result.data()), count * sizeof(T))) throw std::runtime_error("truncated InfLLM array");
    return result;
}
inline void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// The binary oracle is produced by the unmodified upstream append_global and
// get_block_k methods. submit optionally exercises the real CUDA builder.
template<class Configure, class Submit, class Mean>
void run(const std::string & path, Configure configure, Submit submit, Mean mean_query) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open InfLLM reference fixture");
    const auto magic = read<char>(input, 8);
    require(std::string(magic.begin(), magic.end()) == "IFLMREF3", "InfLLM fixture format");
    const auto cases = read<std::uint32_t>(input);
    std::uint64_t checked_blocks = 0, frames = 0, tied_sets = 0, checked_rankings = 0;
    for (std::uint32_t c = 0; c < cases; ++c) {
        const auto kv_heads = read<std::uint32_t>(input), heads = read<std::uint32_t>(input);
        const auto width = read<std::uint32_t>(input), block = read<std::uint32_t>(input);
        const auto initial = read<std::uint32_t>(input), local = read<std::uint32_t>(input);
        const auto total = read<std::uint32_t>(input), capacity = read<std::uint32_t>(input);
        const auto representatives = read<std::uint32_t>(input), frame_count = read<std::uint32_t>(input);
        const solid::KvGeometry geometry{1, kv_heads, width, block, capacity, 64};
        const solid::SparseBudget budget{total, initial, local, representatives, true, true};
        solid::RepresentativeIndex index(geometry, heads, budget);
        configure(index, geometry, heads, budget);
        std::vector<solid::Half> all_kv;
        std::uint64_t published = 0;
        for (std::uint32_t f = 0; f < frame_count; ++f) {
            const auto first = read<std::uint64_t>(input);
            const auto count = read<std::uint32_t>(input);
            const auto origin = first + 1 - std::min<std::uint64_t>(first + 1, local);
            auto kv = read<solid::Half>(input, static_cast<std::size_t>(count) * kv_heads * width * 2);
            auto scores = read<float>(input, static_cast<std::size_t>(first + count - origin) * heads);
            all_kv.insert(all_kv.end(), kv.begin(), kv.end());
            submit(index, first, count, kv, scores);
            const auto added = read<std::uint32_t>(input);
            for (std::uint32_t b = 0; b < added; ++b) {
                const auto id = read<std::uint64_t>(input);
                require(id == published++, "upstream block publication order");
                auto block_scores = read<solid::Half>(input, heads * block);
                auto upstream_positions = read<std::uint64_t>(input, heads * representatives);
                auto upstream_mean = read<solid::Half>(input, heads * width);
                require(index.ready(0, id), "block was not published with upstream");
                for (std::uint32_t h = 0; h < heads; ++h) {
                    auto selected = index.representative_positions(0, h, id);
                    require(selected.size() == representatives, "representative count differs from upstream");
                    const auto begin = initial + id * block;
                    std::vector<float> ordered(block);
                    for (std::uint32_t t = 0; t < block; ++t) ordered[t] = solid::from_half(block_scores[h * block + t]);
                    std::sort(ordered.begin(), ordered.end(), std::greater<float>());
                    std::vector<float> actual_scores;
                    for (const auto p : selected) {
                        require(p >= begin && p < begin + block, "representative escaped its complete block");
                        actual_scores.push_back(solid::from_half(block_scores[h * block + p - begin]));
                    }
                    require(std::set<std::uint64_t>(selected.begin(), selected.end()).size() == representatives,
                        "representative position repeated");
                    std::sort(actual_scores.begin(), actual_scores.end(), std::greater<float>());
                    require(std::equal(actual_scores.begin(), actual_scores.end(), ordered.begin()),
                        "representative Top-K differs from upstream scores");
                    auto expected = std::vector<std::uint64_t>(upstream_positions.begin() + h * representatives,
                        upstream_positions.begin() + (h + 1) * representatives);
                    auto chosen = selected;
                    std::sort(chosen.begin(), chosen.end());
                    std::sort(expected.begin(), expected.end());
                    const bool identical = chosen == expected;
                    if (!identical) {
                        // torch.topk does not promise stable indices for tied
                        // boundary scores. Require the same optimal score set.
                        require(representatives < block && ordered[representatives - 1] == ordered[representatives],
                            "non-tied representative set differs from upstream");
                        ++tied_sets;
                    }
                    for (std::uint32_t d = 0; d < width; ++d) {
                        float sum = 0;
                        for (const auto p : selected) {
                            const auto offset = (p * kv_heads * 2 + (h / (heads / kv_heads)) * 2) * width + d;
                            sum += solid::from_half(all_kv.at(offset));
                        }
                        const auto mean = solid::to_half(sum / representatives);
                        if (identical) require(mean == upstream_mean[h * width + d], "gather/mean differs from upstream");
                        std::vector<float> unit_query(heads * width, 0);
                        unit_query[h * width + d] = 1;
                        require(index.score(0, h, id, unit_query) == solid::from_half(mean),
                            "published GPU/CPU representative is not the gathered FP16 mean");
                    }
                }
                ++checked_blocks;
            }
            require(published == 0 || index.ready(0, published - 1), "published prefix has a hole");
            require(published == geometry.blocks_per_head() || !index.ready(0, published), "partial block published too early");
            ++frames;
        }
        const auto ranking_cases = read<std::uint32_t>(input);
        for (std::uint32_t r = 0; r < ranking_cases; ++r) {
            const auto count = read<std::uint32_t>(input), candidates = read<std::uint32_t>(input), topk = read<std::uint32_t>(input);
            auto ids = read<std::uint64_t>(input, candidates);
            auto queries = read<float>(input, static_cast<std::size_t>(count) * heads * width);
            auto upstream_mean = read<float>(input, heads * width);
            auto upstream_scores = read<float>(input, candidates);
            auto upstream_topk = read<std::uint64_t>(input, topk);
            const auto mean = mean_query(queries, count, heads * width);
            require(mean.size() == upstream_mean.size(), "retrieval mean shape differs from upstream");
            for (std::size_t i = 0; i < mean.size(); ++i) {
                // Bound differences between sequential and upstream parallel
                // F32 reductions by their sum-of-magnitudes rounding error.
                double magnitude = 0;
                for (std::uint32_t q = 0; q < count; ++q) magnitude += std::abs(queries[q * mean.size() + i]);
                const auto tolerance = 2 * std::numeric_limits<float>::epsilon() * magnitude;
                if (!std::isfinite(mean[i]) || std::abs(mean[i] - upstream_mean[i]) > tolerance) {
                    throw std::runtime_error("retrieval mean differs from upstream: case=" + std::to_string(c) +
                        " tokens=" + std::to_string(count) + " element=" + std::to_string(i) +
                        " expected=" + std::to_string(upstream_mean[i]) +
                        " actual=" + std::to_string(mean[i]));
                }
            }
            const solid::BlockCandidates request{ids, std::vector<std::uint32_t>(candidates, block), topk * block, true};
            const auto ranked = index.rank_blocks(0, mean, request);
            require(ranked.size() == kv_heads, "retrieval KV-head mapping differs from upstream");
            std::vector<float> optimal;
            for (const auto value : upstream_scores) optimal.push_back(value);
            std::sort(optimal.begin(), optimal.end(), std::greater<float>());
            for (const auto & selected : ranked) {
                require(selected.size() == upstream_topk.size(), "retrieval Top-K count differs from upstream");
                std::set<std::uint64_t> unique;
                std::vector<float> selected_scores;
                for (const auto id : selected) {
                    const auto found = std::lower_bound(ids.begin(), ids.end(), id);
                    require(found != ids.end() && *found == id && unique.insert(id).second, "retrieval candidate invalid/repeated");
                    selected_scores.push_back(upstream_scores[found - ids.begin()]);
                }
                std::sort(selected_scores.begin(), selected_scores.end(), std::greater<float>());
                require(std::equal(selected_scores.begin(), selected_scores.end(), optimal.begin()),
                    "retrieval selected inferior blocks compared with upstream calc_block_topk");
            }
            ++checked_rankings;
        }
    }
    require(input.peek() == std::char_traits<char>::eof(), "trailing InfLLM fixture data");
    std::cout << "InfLLM upstream comparison: " << cases << " cases, " << frames << " frames, "
              << checked_blocks << " complete blocks, " << checked_rankings << " retrievals; valid tied Top-K sets=" << tied_sets << '\n';
}
} // namespace test_infllm
