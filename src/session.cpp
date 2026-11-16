#include "solid/session.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace solid {
AttentionSession::AttentionSession(std::unique_ptr<KvStore> store, std::uint32_t query_heads, SparseBudget budget,
                                   bool sparse, std::uint64_t prefill_tokens, const std::filesystem::path & selection_trace) :
    store_(std::move(store)), query_heads_(query_heads), budget_(budget), sparse_(sparse), prefill_tokens_(prefill_tokens) {
    if (!store_ || query_heads == 0 || query_heads % store_->geometry().heads != 0 ||
        prefill_tokens == 0 || prefill_tokens > store_->geometry().capacity_tokens) throw std::invalid_argument("invalid attention session");
    if (sparse_) representatives_ = std::make_unique<RepresentativeIndex>(store_->geometry(), query_heads, budget);
    if (!selection_trace.empty()) {
        if (std::filesystem::exists(selection_trace)) throw std::runtime_error("selection trace already exists");
        trace_.open(selection_trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create selection trace");
    }
}
std::vector<float> AttentionSession::process(std::uint32_t layer, std::uint64_t first_position, std::size_t tokens,
                                            const std::vector<float> & queries, const std::vector<float> & keys,
                                            const std::vector<float> & values, float scale) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    const auto query_elements = static_cast<std::size_t>(query_heads_) * g.dimension;
    const auto kv_elements = static_cast<std::size_t>(g.heads) * g.dimension;
    if (tokens == 0 || tokens > g.capacity_tokens || first_position > g.capacity_tokens - tokens ||
        queries.size() / tokens != query_elements || queries.size() % tokens != 0 ||
        keys.size() / tokens != kv_elements || keys.size() % tokens != 0 || values.size() != keys.size()) {
        throw std::invalid_argument("attention batch shape or capacity mismatch");
    }
    std::vector<float> output(queries.size());
    std::vector<float> q(query_elements);
    try {
        for (std::size_t token = 0; token < tokens; ++token) {
            const auto position = first_position + token;
            std::copy_n(queries.data() + token * query_elements, query_elements, q.data());
            append_token(layer, position, q, keys.data() + token * kv_elements, values.data() + token * kv_elements);
            std::vector<HeadSelection> selection;
            const bool use_sparse = sparse_ && position >= prefill_tokens_;
            if (use_sparse) {
                selection = select_blocks(g, budget_, *representatives_, layer, position, q);
                record(layer, position, selection);
                for (const auto & head : selection) stats_.selected_tokens += head.positions.size();
            }
            auto attended = streamed_attention(*store_, layer, position, query_heads_, q, scale, use_sparse ? &selection : nullptr);
            std::copy(attended.begin(), attended.end(), output.begin() + token * query_elements);
            ++stats_.attention_queries;
            std::uint64_t workspace = (output.capacity() + q.capacity() + attended.capacity()) * sizeof(float) +
                (2 * kv_elements + static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension) * sizeof(Half) +
                static_cast<std::size_t>(query_heads_ / g.heads) * (g.dimension + 2) * sizeof(double);
            for (const auto & head : selection) workspace += (head.positions.capacity() + head.blocks.capacity()) * sizeof(std::uint64_t);
            stats_.workspace_peak_bytes = std::max(stats_.workspace_peak_bytes, workspace);
        }
    } catch (...) { failed_ = true; throw; }
    return output;
}
void AttentionSession::append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                    const float * keys, const float * values) {
    const auto elements = static_cast<std::size_t>(geometry().heads) * geometry().dimension;
    for (const auto value : query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite query");
    std::vector<Half> k(elements), v(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        k[i] = to_half(keys[i]);
        v[i] = to_half(values[i]);
        if (!std::isfinite(from_half(k[i])) || !std::isfinite(from_half(v[i]))) throw std::invalid_argument("K/V cannot be represented as finite FP16");
    }
    store_->append(layer, position, k, v);
    if (representatives_) representatives_->observe(layer, position, query, k);
}
std::uint32_t AttentionSession::gather_slots() const {
    const auto tokens = sparse_ ? budget_.total : geometry().capacity_tokens;
    const auto padded = (static_cast<std::uint64_t>(tokens) + 31) / 32 * 32;
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::overflow_error("gather length exceeds backend limits");
    return static_cast<std::uint32_t>(padded);
}
std::uint64_t AttentionSession::gather_elements() const {
    return static_cast<std::uint64_t>(gather_slots()) * (2ull * geometry().heads * geometry().dimension + query_heads_);
}
std::vector<Half> AttentionSession::gather(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                          const std::vector<float> & keys, const std::vector<float> & values) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    if (position < prefill_tokens_ || query.size() != static_cast<std::size_t>(query_heads_) * g.dimension ||
        keys.size() != static_cast<std::size_t>(g.heads) * g.dimension || keys.size() != values.size()) {
        throw std::invalid_argument("gather requires one decode query");
    }
    try {
        append_token(layer, position, query, keys.data(), values.data());
        std::vector<HeadSelection> selection;
        if (sparse_) selection = select_blocks(g, budget_, *representatives_, layer, position, query);
        else {
            selection.resize(g.heads);
            for (auto & head : selection) {
                head.positions.resize(static_cast<std::size_t>(position + 1));
                std::iota(head.positions.begin(), head.positions.end(), std::uint64_t{0});
            }
        }
        record(layer, position, selection);
        const auto slots = gather_slots();
        const auto mask_offset = static_cast<std::size_t>(slots) * 2 * g.heads * g.dimension;
        std::vector<Half> packed(static_cast<std::size_t>(gather_elements()), 0);
        std::fill(packed.begin() + mask_offset, packed.end(), Half{0xfc00});
        KvBlock block;
        const auto group = query_heads_ / g.heads;
        for (std::uint32_t head = 0; head < g.heads; ++head) {
            std::uint64_t loaded = std::numeric_limits<std::uint64_t>::max();
            const auto & positions = selection[head].positions;
            if (positions.empty() || positions.size() > slots) throw std::logic_error("invalid gathered selection size");
            for (std::size_t i = 0; i < positions.size(); ++i) {
                const auto id = positions[i] / g.block_tokens;
                if (loaded != id) { store_->read(layer, head, id, block); loaded = id; }
                const auto row = positions[i] % g.block_tokens;
                const auto offset = (static_cast<std::size_t>(head) * slots + i) * 2 * g.dimension;
                for (std::uint32_t d = 0; d < g.dimension; ++d) {
                    packed[offset + d] = block.keys[row * g.dimension + d];
                    packed[offset + g.dimension + d] = block.values[row * g.dimension + d];
                }
            }
            for (std::uint32_t q = head * group; q < (head + 1) * group; ++q) {
                std::fill_n(packed.data() + mask_offset + q * slots, positions.size(), Half{0});
            }
            stats_.selected_tokens += positions.size();
        }
        ++stats_.attention_queries;
        stats_.workspace_peak_bytes = std::max<std::uint64_t>(stats_.workspace_peak_bytes,
            packed.capacity() * sizeof(Half) + static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension * sizeof(Half));
        return packed;
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::record(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & selection) {
    if (!trace_.is_open()) return;
    for (std::size_t head = 0; head < selection.size(); ++head) {
        trace_ << "{\"layer\":" << layer << ",\"position\":" << position << ",\"kv_head\":" << head << ",\"ranges\":[";
        const auto & positions = selection[head].positions;
        for (std::size_t i = 0; i < positions.size();) {
            const auto first = i++;
            while (i < positions.size() && positions[i] == positions[i - 1] + 1) ++i;
            if (first > 0) trace_ << ',';
            trace_ << '[' << positions[first] << ',' << i - first << ']';
        }
        trace_ << "]}\n";
    }
    if (!trace_) throw std::runtime_error("selection trace write failed");
}
void AttentionSession::flush() {
    if (failed_) throw std::runtime_error("attention session failed");
    try {
        store_->flush();
        if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("selection trace flush failed"); }
    } catch (...) { failed_ = true; throw; }
}
SessionStats AttentionSession::stats() const {
    auto result = stats_;
    result.kv_resident_bytes = store_->resident_bytes();
    result.representative_bytes = representatives_ ? representatives_->resident_bytes() : 0;
    result.io = store_->io_stats();
    return result;
}
} // namespace solid
