#pragma once

#include "solid/attention.hpp"
#include "solid/prefetch.hpp"
#include "solid/tiled_prefill.hpp"
#include <fstream>
#include <functional>
#include <memory>

namespace solid {
struct SessionStats {
    std::uint64_t attention_queries = 0;
    std::uint64_t selected_tokens = 0;
    std::uint64_t workspace_peak_bytes = 0;
    std::uint64_t kv_resident_bytes = 0;
    std::uint64_t kv_write_buffer_bytes = 0;
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
    std::uint64_t dense_prefetch_layers = 0, dense_prefetch_wait_ns = 0;
    std::uint64_t sparse_prefetch_layers = 0, sparse_prefetch_wait_ns = 0;
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
                     std::uint32_t prefetch_slots_per_head = 0, const std::filesystem::path & prefetch_trace = {},
                     std::optional<RelativeRope> rope = {});
    std::vector<float> process(std::uint32_t layer, std::uint64_t first_position, std::size_t tokens,
                               const std::vector<float> & queries, const std::vector<float> & keys,
                               const std::vector<float> & values, float scale, bool interleaved_inputs = false);
    std::vector<Half> gather(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                              const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs = false);
    void gather_into(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                     const std::vector<float> & keys, const std::vector<float> & values,
                     Half * packed, std::size_t elements, bool interleaved_inputs = false);
    std::uint32_t gather_slots() const;
    std::uint64_t gather_elements() const;
    void read_decode_rows(std::uint32_t layer, const std::vector<KvRowRange> & ranges, AsyncReader * reader, std::size_t depth);
    void account_cached_gather(std::uint64_t ns) { stats_.gather_ns += ns; }
    void read_dense_history(std::uint32_t layer, std::uint32_t position, Half * destination, AsyncReader * reader);
    void append_dense_decode(std::uint32_t layer, std::uint64_t position, const float * keys, const float * values);
    bool layer_write_ready(std::uint32_t layer) const;
    void flush_ready_layer(std::uint32_t layer) { store_->flush_full_layer(layer); }
    void account_dense_prefetch(std::uint64_t wait_ns) {
        ++stats_.dense_prefetch_layers;
        stats_.dense_prefetch_wait_ns += wait_ns;
    }
    void account_sparse_prefetch(std::uint64_t wait_ns) {
        ++stats_.sparse_prefetch_layers;
        stats_.sparse_prefetch_wait_ns += wait_ns;
    }
    void flush();
    void enable_direct_reads();
    void finish();
    SessionStats stats() const;
    const KvGeometry & geometry() const { return store_->geometry(); }
    const KvGeometry & representative_geometry() const { return representatives_->representative_geometry(); }
    std::uint32_t query_heads() const { return query_heads_; }
    SparseBudget budget() const { return budget_; }
    const std::optional<RelativeRope> & relative_rope() const { return rope_; }
    void observe_probabilities(std::uint32_t layer, std::uint64_t position, const std::vector<float> & probabilities);
    bool device_probabilities() const { return representatives_ && representatives_->device_probabilities(); }
    void observe_device_probabilities(std::uint32_t layer, std::uint64_t position, DeviceKvBatch kv, DeviceQuery scores,
        void * compute_stream = nullptr);
    void finish_device_probabilities();
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
                                            bool interleaved_inputs = false, const std::function<void()> & after_append = {});
    std::vector<HeadSelection> select_before_kv(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
    std::vector<HeadSelection> select_decode_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query,
        const std::vector<float> & keys, const std::vector<float> & values, const std::function<void()> & after_append = {});
    void append_after_selection(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs);
    std::optional<DeviceMemory> direct_memory() const;
    std::vector<HeadSelection> select_before_kv_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query);
    void append_after_selection_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys);
    void stage_device_keys(std::uint32_t layer, std::uint64_t position, DeviceKeys keys,void * compute_stream=nullptr);
    void archive_device_kv(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv);
private:
    std::unique_ptr<KvStore> store_;
    std::unique_ptr<RepresentativeIndex> representatives_;
    std::unique_ptr<Prefetcher> prefetch_;
    std::unique_ptr<TiledPrefill> prefill_;
    std::vector<Half> prefill_tile_;
    std::uint32_t query_heads_;
    SparseBudget budget_;
    std::optional<RelativeRope> rope_;
    bool sparse_;
    std::uint64_t prefill_tokens_;
    SessionStats stats_;
    std::ofstream trace_;
    std::ifstream replay_;
    bool failed_ = false;
    std::optional<std::pair<std::uint32_t, std::uint64_t>> early_;
    bool direct_early_ = false;
    bool device_keys_staged_ = false;
    void append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                      const float * keys, const float * values, bool query_observed = false);
    void record(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & selection);
    std::vector<HeadSelection> select(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
    std::vector<float> tiled_prefill(std::uint32_t layer, std::uint64_t first, std::size_t tokens,
                                    const std::vector<float> & query, const std::vector<float> & keys,
                                    const std::vector<float> & values, float scale, bool packed);
    std::vector<float> infllm_prefill(std::uint32_t layer, std::uint64_t first, std::size_t tokens,
                                    const std::vector<float> & query, const std::vector<float> & keys,
                                    const std::vector<float> & values, float scale, bool packed,
                                    DeviceQuery device_query = {}, DeviceOutput device_output = {},
                                    DeviceKvBatch device_kv = {}, const std::vector<Half> * archived = nullptr);
};
} // namespace solid
