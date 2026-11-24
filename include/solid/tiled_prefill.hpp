#pragma once

#include "solid/kv_store.hpp"
#include "solid/device_input.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace solid {
struct RelativeRope;
struct SparsePrefillFrame {
    std::uint64_t first_query;
    std::uint32_t queries;
    float scale;
    std::uint32_t local;
    const RelativeRope & rope;
    const std::vector<std::uint64_t> & positions;
    const std::vector<float> & query;
    // [KV head, selected position, K/V, D].
    const std::vector<Half> & kv;
    // The local union can include rows that age out before the last query.
    std::size_t local_only_from = static_cast<std::size_t>(-1);
    DeviceQuery device_query;
    DeviceOutput device_output;
    DeviceKvBatch device_kv;
    bool keep_scores_on_device = false;
};
struct SparsePrefillOutput {
    std::vector<float> attention;
    // [key, query head], starting at the first query's local window.
    std::vector<float> key_scores;
    DeviceQuery device_scores = {};
};
struct PrefillStats {
    std::uint32_t query_capacity=0, tile_capacity=0;
    std::uint64_t kv_device_bytes = 0, workspace_device_bytes = 0;
    std::uint64_t kv_device_peak_bytes = 0, workspace_device_peak_bytes = 0;
    std::uint64_t batches = 0, query_tokens = 0, tiles = 0, kv_tokens = 0, upload_bytes = 0;
    std::uint64_t query_upload_bytes = 0, output_download_bytes = 0, output_device_copy_bytes = 0;
    std::uint64_t probability_download_bytes = 0;
    std::uint64_t device_query_tokens = 0, device_kv_tokens = 0;
    std::uint64_t retrieval_mean_events = 0, retrieval_mean_host_waits = 0;
};
class TiledPrefill {
public:
    virtual ~TiledPrefill() = default;
    virtual std::uint32_t query_capacity() const = 0;
    virtual std::uint32_t tile_capacity() const = 0;
    virtual void begin(std::uint64_t first_query, std::uint32_t queries, float scale, const std::vector<float> & query) = 0;
    virtual std::optional<DeviceMemory> direct_memory() const { return {}; }
    // Borrowed device Q/KV remain live and producer-complete through finish/abort.
    virtual void begin_device(std::uint64_t,std::uint32_t,float,DeviceQuery) { throw std::logic_error("device prefill query is disabled"); }
    virtual void consume_device(std::uint64_t,std::uint32_t,DeviceKvBatch) { throw std::logic_error("device prefill KV is disabled"); }
    // Complete all readers and the device output copy before returning.
    virtual void finish_to(DeviceOutput) { throw std::logic_error("device prefill output is disabled"); }
    // Interleaved [kv_head, tile_capacity, 2, D]. Return after the source can
    // be reused; computation may still be in flight on the owned stream.
    virtual void consume(std::uint64_t first_key, std::uint32_t valid_tokens, const std::vector<Half> & kv) = 0;
    virtual std::vector<float> finish() = 0;
    virtual void abort() = 0;
    virtual PrefillStats stats() const = 0;
    virtual const char * name() const = 0;
    virtual bool supports_sparse() const { return false; }
    virtual DeviceQuery mean_sparse_query(DeviceQuery, std::uint32_t) { throw std::logic_error("device sparse query is disabled"); }
    // Retain the input Q until the ranker has consumed the returned event, or
    // this backend has completed its following sparse()/abort() call.
    virtual DeviceQueryWithEvent mean_sparse_query_async(DeviceQuery query, std::uint32_t count) {
        return {mean_sparse_query(query, count), nullptr};
    }
    virtual SparsePrefillOutput sparse(const SparsePrefillFrame &) { throw std::logic_error("sparse prefill is disabled"); }
};
std::unique_ptr<TiledPrefill> make_cuda_tiled_prefill(const KvGeometry & geometry, std::uint32_t query_heads,
                                                    std::uint32_t queries, std::uint32_t tile_tokens, int device,
                                                    std::uint32_t sparse_local = 0);
std::unique_ptr<TiledPrefill> make_sycl_tiled_prefill(const KvGeometry & geometry, std::uint32_t query_heads,
                                                    std::uint32_t queries, std::uint32_t tile_tokens, void * backend_queue);
} // namespace solid
