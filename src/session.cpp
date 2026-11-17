#include "solid/session.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace solid {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
}
AttentionSession::AttentionSession(std::unique_ptr<KvStore> store, std::uint32_t query_heads, SparseBudget budget,
                                   bool sparse, std::uint64_t prefill_tokens, const std::filesystem::path & selection_trace,
                                   const std::filesystem::path & selection_replay, std::unique_ptr<AsyncReader> reader,
                                   std::uint32_t prefetch_slots_per_head, const std::filesystem::path & prefetch_trace) :
    store_(std::move(store)), query_heads_(query_heads), budget_(budget), sparse_(sparse), prefill_tokens_(prefill_tokens) {
    if (!store_ || query_heads == 0 || query_heads % store_->geometry().heads != 0 ||
        prefill_tokens == 0 || prefill_tokens > store_->geometry().capacity_tokens) throw std::invalid_argument("invalid attention session");
    if (sparse_) representatives_ = std::make_unique<RepresentativeIndex>(store_->geometry(), query_heads, budget);
    if (reader) {
        if (!sparse_ || geometry().layout != KvLayout::interleaved) throw std::invalid_argument("prefetch requires sparse interleaved KV");
        if (!prefetch_slots_per_head) prefetch_slots_per_head = static_cast<std::uint32_t>(std::min<std::uint64_t>(geometry().blocks_per_head(),
            (static_cast<std::uint64_t>(budget.total) + geometry().block_tokens - 1) / geometry().block_tokens + 3));
        prefetch_ = std::make_unique<Prefetcher>(*store_, budget, prefetch_slots_per_head, std::move(reader), prefetch_trace);
    }
    if (!selection_replay.empty()) {
        if (!sparse_) throw std::invalid_argument("selection replay requires sparse attention");
        if (!selection_trace.empty() && std::filesystem::weakly_canonical(selection_trace) == std::filesystem::weakly_canonical(selection_replay)) {
            throw std::invalid_argument("selection replay and output paths must differ");
        }
        replay_.open(selection_replay, std::ios::binary);
        if (!replay_) throw std::runtime_error("cannot open selection replay");
    }
    if (!selection_trace.empty()) {
        if (std::filesystem::exists(selection_trace)) throw std::runtime_error("selection trace already exists");
        trace_.open(selection_trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create selection trace");
    }
}
std::vector<float> AttentionSession::process(std::uint32_t layer, std::uint64_t first_position, std::size_t tokens,
                                            const std::vector<float> & queries, const std::vector<float> & keys,
                                            const std::vector<float> & values, float scale, bool interleaved_inputs) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    const auto query_elements = static_cast<std::size_t>(query_heads_) * g.dimension;
    const auto kv_elements = static_cast<std::size_t>(g.heads) * g.dimension;
    if (tokens == 0 || tokens > g.capacity_tokens || first_position > g.capacity_tokens - tokens ||
        queries.size() / tokens != query_elements || queries.size() % tokens != 0 ||
        keys.size() / tokens != kv_elements * (interleaved_inputs ? 2 : 1) || keys.size() % tokens != 0 ||
        (interleaved_inputs ? !values.empty() : values.size() != keys.size())) {
        throw std::invalid_argument("attention batch shape or capacity mismatch");
    }
    std::vector<float> output(queries.size());
    std::vector<float> q(query_elements);
    try {
        for (std::size_t token = 0; token < tokens; ++token) {
            const auto position = first_position + token;
            std::copy_n(queries.data() + token * query_elements, query_elements, q.data());
            append_token(layer, position, q, keys.data() + token * kv_elements * (interleaved_inputs ? 2 : 1),
                         interleaved_inputs ? nullptr : values.data() + token * kv_elements);
            std::vector<HeadSelection> selection;
            const bool use_sparse = sparse_ && position >= prefill_tokens_;
            if (use_sparse) {
                const auto start = Clock::now();
                selection = select(layer, position, q);
                stats_.selection_ns += elapsed(start);
                record(layer, position, selection);
                for (const auto & head : selection) stats_.selected_tokens += head.positions.size();
            }
            const auto attention_start = Clock::now();
            auto attended = streamed_attention(*store_, layer, position, query_heads_, q, scale, use_sparse ? &selection : nullptr);
            stats_.streaming_attention_ns += elapsed(attention_start);
            std::copy(attended.begin(), attended.end(), output.begin() + token * query_elements);
            ++stats_.attention_queries;
            std::uint64_t workspace = (output.capacity() + q.capacity() + attended.capacity()) * sizeof(float) +
                (3 * kv_elements + static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension) * sizeof(Half) +
                static_cast<std::size_t>(query_heads_ / g.heads) * (g.dimension + 2) * sizeof(double);
            for (const auto & head : selection) workspace += (head.positions.capacity() + head.blocks.capacity()) * sizeof(std::uint64_t);
            stats_.workspace_peak_bytes = std::max(stats_.workspace_peak_bytes, workspace);
        }
    } catch (...) { failed_ = true; throw; }
    return output;
}
void AttentionSession::append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                    const float * keys, const float * values, bool query_observed) {
    const auto elements = static_cast<std::size_t>(geometry().heads) * geometry().dimension;
    const auto start = Clock::now();
    for (const auto value : query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite query");
    std::vector<Half> kv(elements * 2);
    const auto dimension = geometry().dimension;
    for (std::size_t i = 0; i < kv.size(); ++i) {
        const auto head = i / (2 * dimension);
        const auto within = i % (2 * dimension);
        const float value = values ? (within < dimension ? keys[head * dimension + within] :
            values[head * dimension + within - dimension]) : keys[i];
        kv[i] = to_half(value);
        if (!std::isfinite(from_half(kv[i]))) throw std::invalid_argument("K/V cannot be represented as finite FP16");
    }
    store_->append_interleaved(layer, position, kv);
    if (representatives_) {
        std::vector<Half> k(elements);
        for (std::uint32_t head = 0; head < geometry().heads; ++head) {
            std::copy_n(kv.data() + head * 2 * dimension, dimension, k.data() + head * dimension);
        }
        if (query_observed) representatives_->append_keys(layer, position, k);
        else representatives_->observe(layer, position, query, k);
    }
    stats_.append_ns += elapsed(start);
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
                                          const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    if (position < prefill_tokens_ || query.size() != static_cast<std::size_t>(query_heads_) * g.dimension ||
        keys.size() != static_cast<std::size_t>(g.heads) * g.dimension * (interleaved_inputs ? 2 : 1) ||
        (interleaved_inputs ? !values.empty() : keys.size() != values.size())) {
        throw std::invalid_argument("gather requires one decode query");
    }
    try {
        append_token(layer, position, query, keys.data(), interleaved_inputs ? nullptr : values.data());
        std::vector<HeadSelection> selection;
        const auto selection_start = Clock::now();
        if (sparse_) selection = select(layer, position, query);
        else {
            selection.resize(g.heads);
            for (auto & head : selection) {
                head.positions.resize(static_cast<std::size_t>(position + 1));
                std::iota(head.positions.begin(), head.positions.end(), std::uint64_t{0});
            }
        }
        stats_.selection_ns += elapsed(selection_start);
        record(layer, position, selection);
        const auto gather_start = Clock::now();
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
                if (g.layout == KvLayout::interleaved) {
                    const auto head_offset = static_cast<std::size_t>(head) * slots * 2 * g.dimension;
                    std::copy_n(block.key(row), 2 * g.dimension, packed.data() + head_offset + i * 2 * g.dimension);
                } else {
                    const auto head_offset = static_cast<std::size_t>(head) * slots * g.dimension;
                    std::copy_n(block.key(row), g.dimension, packed.data() + head_offset + i * g.dimension);
                    std::copy_n(block.value(row), g.dimension, packed.data() + head_offset + static_cast<std::size_t>(g.heads) * slots * g.dimension + i * g.dimension);
                }
                stats_.gather_copy_bytes += 2 * g.dimension * sizeof(Half);
            }
            for (std::uint32_t q = head * group; q < (head + 1) * group; ++q) {
                std::fill_n(packed.data() + mask_offset + q * slots, positions.size(), Half{0});
            }
            stats_.selected_tokens += positions.size();
        }
        stats_.gather_ns += elapsed(gather_start);
        ++stats_.attention_queries;
        stats_.workspace_peak_bytes = std::max<std::uint64_t>(stats_.workspace_peak_bytes,
            packed.capacity() * sizeof(Half) + static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension * sizeof(Half));
        return packed;
    } catch (...) { failed_ = true; throw; }
}
Prefetcher & AttentionSession::prefetcher() {
    if (!prefetch_) throw std::logic_error("prefetch is disabled");
    return *prefetch_;
}
void AttentionSession::begin_prefetch(std::uint32_t layer, std::uint64_t position) {
    if (failed_) throw std::runtime_error("attention session failed");
    if (prefetch_ && position >= prefill_tokens_ && position < geometry().capacity_tokens) {
        try { prefetch_->begin(layer, position); } catch (...) { failed_ = true; throw; }
    }
}
std::vector<HeadSelection> AttentionSession::select_decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                                          const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs) {
    if (failed_) throw std::runtime_error("attention session failed");
    if (!prefetch_ || position < prefill_tokens_ || query.size() != static_cast<std::size_t>(query_heads_) * geometry().dimension ||
        keys.size() != static_cast<std::size_t>(geometry().heads) * geometry().dimension * (interleaved_inputs ? 2 : 1) ||
        (interleaved_inputs ? !values.empty() : values.size() != keys.size())) throw std::invalid_argument("invalid prefetched decode input");
    try {
        append_token(layer, position, query, keys.data(), interleaved_inputs ? nullptr : values.data());
        const auto start = Clock::now();
        auto selected = select(layer, position, query);
        stats_.selection_ns += elapsed(start);
        record(layer, position, selected);
        for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
        ++stats_.attention_queries;
        return selected;
    } catch (...) { failed_ = true; throw; }
}
std::vector<HeadSelection> AttentionSession::select_before_kv(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) {
    if (failed_ || early_ || !prefetch_ || position < prefill_tokens_ || store_->length(layer) != position) {
        throw std::logic_error("early selection requires an uncommitted decode position");
    }
    try {
        const auto start = Clock::now();
        representatives_->observe_query(layer, position, query);
        auto selected = select(layer, position, query);
        stats_.selection_ns += elapsed(start);
        early_ = std::make_pair(layer, position);
        record(layer, position, selected);
        for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
        return selected;
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::append_after_selection(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                             const std::vector<float> & keys, const std::vector<float> & values, bool packed) {
    if (failed_ || !early_ || *early_ != std::make_pair(layer, position) ||
        query.size() != static_cast<std::size_t>(query_heads_) * geometry().dimension ||
        keys.size() != static_cast<std::size_t>(geometry().heads) * geometry().dimension * (packed ? 2 : 1) ||
        (packed ? !values.empty() : values.size() != keys.size())) throw std::invalid_argument("late KV does not match early selection");
    try {
        append_token(layer, position, query, keys.data(), packed ? nullptr : values.data(), true);
        early_.reset();
        ++stats_.attention_queries;
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
std::vector<HeadSelection> AttentionSession::select(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) {
    if (!replay_.is_open()) return select_blocks(geometry(), budget_, *representatives_, layer, position, query);
    std::vector<HeadSelection> selection(geometry().heads);
    for (std::uint32_t head = 0; head < geometry().heads; ++head) {
        std::string line;
        // Each range consumes at least one token. Bound input before allocation.
        const auto limit = static_cast<std::uint64_t>(budget_.total) * 48 + 256;
        for (char ch; replay_.get(ch) && ch != '\n';) {
            if (line.size() >= limit) throw std::invalid_argument("selection replay line exceeds budget");
            line.push_back(ch);
        }
        if (line.empty()) throw std::invalid_argument("selection replay ended before requested query");
        const auto item = nlohmann::json::parse(line);
        const auto integer = [](const nlohmann::json & value) -> std::uint64_t {
            if (!value.is_number_integer() || value.get<std::int64_t>() < 0) throw std::invalid_argument("invalid replay integer");
            return value.get<std::uint64_t>();
        };
        if (!item.is_object() || item.size() != 4 || integer(item.at("layer")) != layer ||
            integer(item.at("position")) != position || integer(item.at("kv_head")) != head || !item.at("ranges").is_array()) {
            throw std::invalid_argument("selection replay query identity mismatch");
        }
        auto & result = selection[head];
        for (const auto & range : item.at("ranges")) {
            if (!range.is_array() || range.size() != 2) throw std::invalid_argument("invalid replay range");
            const auto first = integer(range[0]), count = integer(range[1]);
            if (count == 0 || first > position || count > position - first + 1 ||
                count > budget_.total - result.positions.size() || (!result.positions.empty() && first <= result.positions.back())) {
                throw std::invalid_argument("replay range exceeds history or token budget");
            }
            for (std::uint64_t i = 0; i < count; ++i) {
                const auto p = first + i;
                result.positions.push_back(p);
                const auto block = p / geometry().block_tokens;
                if (result.blocks.empty() || result.blocks.back() != block) result.blocks.push_back(block);
            }
        }
        if (result.positions.empty()) throw std::invalid_argument("empty selection replay");
        const auto contains = [&](std::uint64_t p) { return std::binary_search(result.positions.begin(), result.positions.end(), p); };
        for (std::uint64_t p = 0; p < std::min<std::uint64_t>(budget_.initial, position + 1); ++p) {
            if (!contains(p)) throw std::invalid_argument("replay omits initial tokens");
        }
        for (auto p = position + 1 - std::min<std::uint64_t>(budget_.local, position + 1); p <= position; ++p) {
            if (!contains(p)) throw std::invalid_argument("replay omits local tokens");
        }
    }
    return selection;
}
void AttentionSession::flush() {
    if (failed_) throw std::runtime_error("attention session failed");
    try {
        store_->flush();
        if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("selection trace flush failed"); }
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::finish() {
    if (prefetch_) prefetch_->finish();
    flush();
    if (replay_.is_open()) {
        replay_ >> std::ws;
        if (replay_.peek() != std::char_traits<char>::eof()) {
            failed_ = true;
            throw std::invalid_argument("selection replay has unused queries");
        }
    }
}
SessionStats AttentionSession::stats() const {
    auto result = stats_;
    result.kv_resident_bytes = store_->resident_bytes();
    result.representative_bytes = representatives_ ? representatives_->resident_bytes() : 0;
    result.io = store_->io_stats();
    if (prefetch_) result.prefetch = prefetch_->stats();
    return result;
}
} // namespace solid
