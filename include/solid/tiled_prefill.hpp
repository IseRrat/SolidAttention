#pragma once

#include "solid/kv_store.hpp"
#include <memory>
#include <vector>

namespace solid {
struct PrefillStats {
    std::uint32_t query_capacity=0, tile_capacity=0;
    std::uint64_t kv_device_bytes = 0, workspace_device_bytes = 0;
    std::uint64_t kv_device_peak_bytes = 0, workspace_device_peak_bytes = 0;
    std::uint64_t batches = 0, query_tokens = 0, tiles = 0, kv_tokens = 0, upload_bytes = 0;
};
class TiledPrefill {
public:
    virtual ~TiledPrefill() = default;
    virtual std::uint32_t query_capacity() const = 0;
    virtual std::uint32_t tile_capacity() const = 0;
    virtual void begin(std::uint64_t first_query, std::uint32_t queries, float scale, const std::vector<float> & query) = 0;
    // Interleaved [kv_head, tile_capacity, 2, D]. Return after the source can
    // be reused; computation may still be in flight on the owned stream.
    virtual void consume(std::uint64_t first_key, std::uint32_t valid_tokens, const std::vector<Half> & kv) = 0;
    virtual std::vector<float> finish() = 0;
    virtual void abort() = 0;
    virtual PrefillStats stats() const = 0;
    virtual const char * name() const = 0;
};
std::unique_ptr<TiledPrefill> make_cuda_tiled_prefill(const KvGeometry & geometry, std::uint32_t query_heads,
                                                    std::uint32_t queries, std::uint32_t tile_tokens, int device);
} // namespace solid
