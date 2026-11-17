#pragma once

#include "solid/attention.hpp"
#include <fstream>
#include <memory>

namespace solid {
struct SessionStats {
    std::uint64_t attention_queries = 0;
    std::uint64_t selected_tokens = 0;
    std::uint64_t workspace_peak_bytes = 0;
    std::uint64_t kv_resident_bytes = 0;
    std::uint64_t representative_bytes = 0;
    IoStats io;
    std::uint64_t append_ns = 0;
    std::uint64_t selection_ns = 0;
    std::uint64_t gather_ns = 0;
    std::uint64_t gather_copy_bytes = 0;
    std::uint64_t streaming_attention_ns = 0;
};

class AttentionSession {
public:
    AttentionSession(std::unique_ptr<KvStore> store, std::uint32_t query_heads, SparseBudget budget,
                     bool sparse, std::uint64_t prefill_tokens, const std::filesystem::path & selection_trace = {},
                     const std::filesystem::path & selection_replay = {});
    std::vector<float> process(std::uint32_t layer, std::uint64_t first_position, std::size_t tokens,
                               const std::vector<float> & queries, const std::vector<float> & keys,
                               const std::vector<float> & values, float scale, bool interleaved_inputs = false);
    std::vector<Half> gather(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                              const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs = false);
    std::uint32_t gather_slots() const;
    std::uint64_t gather_elements() const;
    void flush();
    void finish();
    SessionStats stats() const;
    const KvGeometry & geometry() const { return store_->geometry(); }
    std::uint32_t query_heads() const { return query_heads_; }
    SparseBudget budget() const { return budget_; }
private:
    std::unique_ptr<KvStore> store_;
    std::unique_ptr<RepresentativeIndex> representatives_;
    std::uint32_t query_heads_;
    SparseBudget budget_;
    bool sparse_;
    std::uint64_t prefill_tokens_;
    SessionStats stats_;
    std::ofstream trace_;
    std::ifstream replay_;
    bool failed_ = false;
    void append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                      const float * keys, const float * values);
    void record(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & selection);
    std::vector<HeadSelection> select(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
};
} // namespace solid
