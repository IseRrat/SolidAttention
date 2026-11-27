// 连接 llama.cpp 计算图与 SSD 调度：提前选块、补读上传、当前 KV 归档和完成回执。
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
                 std::unique_ptr<AttentionEvents> events = {},bool share_host_cache = false,bool coalesce_uploads = true,bool per_completion_uploads = false);
    ~GgmlPrefetch();
    ggml_tensor * tensor() const;
    std::uint32_t tokens_per_head() const;
    std::uint64_t attention_epoch() const;
    std::vector<Half> decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                              const std::vector<float> & keys, const std::vector<float> & values, bool packed_inputs);
    // 确认本层 attention 完成，释放槽位并推进下一层预取与本层写回。
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
                             bool deferred_archive = true);
    // The graph retains Q until the KV frontier or quiesce() has joined the
    // selection worker. The synchronous entry above consumes Q before return.
    // Q 到达后把正式选块和补读交给工作线程；调用方保留 Q 到后续汇合点。
    void select_early_device_async(std::uint32_t layer,std::uint64_t position,DeviceQuery query,bool query_covers_attention,
                                   bool deferred_archive = true);
    // Use deferred_archive=false when the later frontier is decode_half().
    // The device-only decode_device() frontier retains the default DAG.
    std::vector<Half> decode_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys,
                                  std::optional<DeviceQuery> serial_query = {},ggml_tensor * device_source = nullptr);
    // 当前设备 KV 到达后的汇合入口，安装本层 attention 所需数据并安排归档。
    std::vector<Half> decode_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys, ggml_tensor * source);
    void begin_prefetch(std::uint32_t layer, std::uint64_t position);
    // 图 11(b) 的阶段边界：整批预取上传完成后，图才回传当前 KV。
    void wait_naive_prefetch();
    ScheduleStats schedule_stats() const;
    // The caller keeps the context/backend alive until this has returned.
    // 结束会话前收齐读写、搬运和工作线程，之后调用方才能释放后端与图。
    void quiesce();
    DeviceCacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void select_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, const DeviceQuery * device,
                     bool covers,bool asynchronous = true,bool deferred_archive = true);
    std::vector<Half> decode_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                  const std::vector<float> & keys, const std::vector<float> & values, bool packed,
                                  const std::vector<Half> * half, const DeviceKeys * device, const DeviceQuery * serial_query,ggml_tensor * device_source);
};
} // namespace solid
