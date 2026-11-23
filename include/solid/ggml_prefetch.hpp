#pragma once
#include "solid/device_cache.hpp"
#include "solid/session.hpp"
#include "solid/scheduler.hpp"
#include "solid/attention_events.hpp"

struct ggml_backend;
struct ggml_tensor;
namespace solid {
class GgmlPrefetch {
public:
    GgmlPrefetch(AttentionSession & session, ggml_backend * backend, const std::filesystem::path & trace = {},
                 ScheduleMode mode = ScheduleMode::naive, const std::filesystem::path & schedule_trace = {},int coordinator_cpu=-1,
                 std::unique_ptr<AttentionEvents> events = {},bool share_host_cache = false);
    ~GgmlPrefetch();
    ggml_tensor * tensor() const;
    std::uint32_t tokens_per_head() const;
    std::uint64_t attention_epoch() const;
    std::vector<Half> decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                              const std::vector<float> & keys, const std::vector<float> & values, bool packed_inputs);
    void completed(std::uint64_t epoch);
    void process_complete();
    void * probability_compute_stream() const;
    bool early_enabled() const;
    bool reuses_sync() const;
    StreamCompletionToken * completion_token(std::uint32_t layer) const;
    bool completion_host_marker() const;
    void select_early(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, bool query_covers_attention);
    std::optional<DeviceMemory> direct_memory() const;
    void select_early_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query, bool query_covers_attention,
                             std::optional<std::uint64_t> projection_ns = {},bool deferred_archive = true);
    // The graph retains Q until the KV frontier or quiesce() has joined the
    // selection worker. The synchronous entry above consumes Q before return.
    void select_early_device_async(std::uint32_t layer,std::uint64_t position,DeviceQuery query,bool query_covers_attention,
                                   std::optional<std::uint64_t> projection_ns = {},bool deferred_archive = true);
    // Use deferred_archive=false when the later frontier is decode_half().
    // The device-only decode_device() frontier retains the default DAG.
    std::vector<Half> decode_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys,
                                  std::optional<DeviceQuery> serial_query = {},ggml_tensor * device_source = nullptr);
    std::vector<Half> decode_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys, ggml_tensor * source,
                                   std::optional<std::uint64_t> projection_ns = {});
    void begin_prefetch(std::uint32_t layer, std::uint64_t position);
    ScheduleStats schedule_stats() const;
    // The caller keeps the context/backend alive until this has returned.
    void quiesce();
    DeviceCacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void select_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, const DeviceQuery * device,
                     bool covers,std::optional<std::uint64_t> projection_ns = {},bool asynchronous = true,bool deferred_archive = true);
    std::vector<Half> decode_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                  const std::vector<float> & keys, const std::vector<float> & values, bool packed,
                                  const std::vector<Half> * half, const DeviceKeys * device, const DeviceQuery * serial_query,ggml_tensor * device_source,
                                  std::optional<std::uint64_t> projection_ns = {});
};
} // namespace solid
