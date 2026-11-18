#pragma once

#include "solid/attention.hpp"
#include "solid/prefetch.hpp"
#include "solid/tiled_prefill.hpp"
#include <fstream>
#include <memory>

namespace solid {
struct SessionStats {
    std::uint64_t attention_queries = 0;
    std::uint64_t selected_tokens = 0;
    std::uint64_t workspace_peak_bytes = 0;
    std::uint64_t kv_resident_bytes = 0;
    std::uint64_t representative_bytes = 0;
    RankerStats ranking;
    std::string selection_backend = "cpu";
    std::string representative_backend = "cpu";
    std::string prefill_backend = "cpu_streaming";
    PrefillStats prefill;
    std::uint64_t prefill_host_bytes = 0, prefill_host_peak_bytes = 0;
    IoStats io;
    std::uint64_t append_ns = 0;
    std::uint64_t selection_ns = 0;
    std::uint64_t gather_ns = 0;
    std::uint64_t gather_copy_bytes = 0;
    std::uint64_t streaming_attention_ns = 0;
    std::uint64_t tiled_prefill_ns = 0;
    std::uint64_t direct_decode_queries = 0, decode_kv_download_bytes = 0;
    std::uint64_t direct_prefill_queries = 0, prefill_kv_download_bytes = 0;
    std::uint64_t prefill_graph_device_bytes = 0, prefill_graph_device_peak_bytes = 0;
    PrefetchStats prefetch;
};

class AttentionSession {
public:
    AttentionSession(std::unique_ptr<KvStore> store, std::uint32_t query_heads, SparseBudget budget,
                     bool sparse, std::uint64_t prefill_tokens, const std::filesystem::path & selection_trace = {},
                     const std::filesystem::path & selection_replay = {}, std::unique_ptr<AsyncReader> reader = {},
                     std::uint32_t prefetch_slots_per_head = 0, const std::filesystem::path & prefetch_trace = {});
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
    void set_ranker(std::unique_ptr<BlockRanker> ranker);
    void set_prefill(std::unique_ptr<TiledPrefill> prefill);
    void release_prefill_if_complete();
    std::optional<DeviceMemory> prefill_direct_memory() const;
    void process_prefill_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,DeviceQuery queries,
                                DeviceKvBatch device_kv,const std::vector<Half> & host_kv,float scale,DeviceOutput output);
    void retain_prefill_graph_output(std::uint64_t bytes);
    void release_prefill_graph_output(std::uint64_t bytes) noexcept;
    bool prefetch_enabled() const { return prefetch_ != nullptr; }
    Prefetcher & prefetcher();
    void begin_prefetch(std::uint32_t layer, std::uint64_t position, PredictionCosts costs = {});
    std::vector<HeadSelection> select_decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                            const std::vector<float> & keys, const std::vector<float> & values,
                                            bool interleaved_inputs = false);
    std::vector<HeadSelection> select_before_kv(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
    void append_after_selection(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs);
    std::optional<DeviceMemory> direct_memory() const { return representatives_ && !replay_.is_open() ? representatives_->direct_memory() : std::optional<DeviceMemory>{}; }
    std::vector<HeadSelection> select_before_kv_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query);
    void append_after_selection_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys);
private:
    std::unique_ptr<KvStore> store_;
    std::unique_ptr<RepresentativeIndex> representatives_;
    std::unique_ptr<Prefetcher> prefetch_;
    std::unique_ptr<TiledPrefill> prefill_;
    std::vector<Half> prefill_tile_;
    std::uint32_t query_heads_;
    SparseBudget budget_;
    bool sparse_;
    std::uint64_t prefill_tokens_;
    SessionStats stats_;
    std::ofstream trace_;
    std::ifstream replay_;
    bool failed_ = false;
    std::optional<std::pair<std::uint32_t, std::uint64_t>> early_;
    bool direct_early_ = false;
    void append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                      const float * keys, const float * values, bool query_observed = false);
    void record(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & selection);
    std::vector<HeadSelection> select(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
    std::vector<float> tiled_prefill(std::uint32_t layer, std::uint64_t first, std::size_t tokens,
                                    const std::vector<float> & query, const std::vector<float> & keys,
                                    const std::vector<float> & values, float scale, bool packed);
};
} // namespace solid
