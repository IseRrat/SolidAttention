#pragma once
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include "solid/trace.hpp"

namespace solid {
enum class ScheduleMode { naive, overlap, sync_reuse };
ScheduleMode parse_schedule_mode(const std::string & name);
const char * schedule_mode_name(ScheduleMode mode);

// CPU is at the KV consumer frontier. Exclude selection and any loading
// already hidden before that frontier; GPU upload waits are timed separately.
inline std::uint64_t cache_load_host_block_ns(std::uint64_t consumer_ready,
        std::uint64_t selection_ready,std::uint64_t uploads_submitted) {
    const auto start=consumer_ready>selection_ready?consumer_ready:selection_ready;
    return uploads_submitted>start?uploads_submitted-start:0;
}
enum class TaskResource { host, ssd, h2d, d2h, gpu };
enum class TaskState { pending, running, complete, failed };
struct TaskSpec {
    std::string name;
    TaskResource resource = TaskResource::host;
    std::vector<std::uint32_t> predecessors;
    std::uint64_t estimate_ns = 1;
};
struct TaskRecord {
    TaskSpec spec;
    TaskState state = TaskState::pending;
    std::uint64_t start_ns = 0, end_ns = 0, latest_start = 0;
    bool critical = false;
    std::optional<std::uint64_t> device_ns{};
    std::uint64_t duration_ns() const { return device_ns.value_or(end_ns-start_ns); }
};
class ScheduleTrace {
public:
    explicit ScheduleTrace(const std::filesystem::path & path = {});
    ~ScheduleTrace();
    std::uint64_t now() const;
    void define(std::uint64_t frame, std::uint32_t layer, std::uint64_t position, ScheduleMode mode,
                std::uint32_t task, const TaskSpec & spec);
    void event(std::uint64_t frame, std::uint32_t task, const char * name, std::uint64_t timestamp,
               const std::string & fence = {},std::optional<std::uint64_t> device_ns = {});
    void flush();
private:
    std::uint64_t origin_;
    std::mutex mutex_;
    std::ofstream stream_;
};
class TaskDag {
public:
    TaskDag(std::vector<TaskSpec> tasks, ScheduleTrace * trace = nullptr, std::uint64_t frame = 0,
            std::uint32_t layer = 0, std::uint64_t position = 0, ScheduleMode mode = ScheduleMode::naive);
    std::vector<std::uint32_t> ready(const std::vector<std::uint32_t> & candidates = {}) const;
    void start(std::uint32_t task, std::uint64_t now,const std::string & proof = {});
    void complete(std::uint32_t task, std::uint64_t now, const std::string & fence = {},std::optional<std::uint64_t> device_ns = {});
    void fence(const std::vector<std::uint32_t> & covered, std::uint64_t now, const std::string & proof);
    void abort(std::uint64_t now);
    std::uint64_t critical_path_ns() const;
    std::uint64_t overlap_ns(std::uint32_t a, std::uint32_t b) const;
    bool finished() const;
    TaskRecord task(std::uint32_t id) const;
    std::size_t size() const { return tasks_.size(); }
private:
    std::vector<TaskRecord> tasks_;
    std::vector<std::uint32_t> order_;
    ScheduleTrace * trace_;
    std::uint64_t frame_;
    std::uint32_t layer_;
    std::uint64_t position_;
    std::vector<trace::Range> ranges_;
    mutable std::recursive_mutex mutex_;
    bool dependencies_complete(std::uint32_t task) const;
};
// One persistent worker, one bounded job. Exceptions are delivered by join().
class TaskWorker {
public:
    explicit TaskWorker(int cpu=-1, const char * name="sa.worker");
    ~TaskWorker();
    void submit(std::function<void()> work);
    void join();
    void wait_until(const std::function<bool()> & ready);
    void notify_progress();
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> work_;
    std::exception_ptr error_;
    bool stopping_ = false, busy_ = false, done_ = false;
    std::thread thread_;
    bool initialized_=false;
    std::exception_ptr initialization_error_;
    void run();
};
struct ScheduleStats {
    bool coalesce_uploads = true;
    bool per_completion_uploads = false;
    std::string completion_backend="graph_boundaries";
    int coordinator_cpu=-1;
    std::uint64_t frames = 0, tasks = 0, worker_jobs = 0;
    std::uint64_t block_frames=0,block_tasks=0,block_critical_path_ns=0;
    std::uint64_t device_event_completions=0,device_attention_ns=0,device_boundary_fallbacks=0,uploads_before_boundary=0,copy_event_queries=0;
    std::uint64_t critical_path_ns = 0, miss_projection_overlap_ns = 0;
    std::uint64_t query_reused_fences = 0, process_reused_fences = 0, attention_fences = 0;
    std::uint64_t attention_receipt_reuses = 0;
    std::uint64_t fallback_fences = 0, event_waits = 0, batch_waits = 0, stream_waits = 0;
    std::uint64_t staging_bytes = 0;
    std::uint64_t backend_drains = 0, backend_compute_waits = 0, batch_groups = 0, events_coalesced = 0;
    std::uint64_t historical_upload_batches = 0, historical_upload_blocks = 0, historical_upload_ranges = 0;
    std::uint64_t archive_attention_reuses = 0, archive_separate_completions = 0;
    std::uint64_t history_consumer_wait_ns = 0, history_consumer_wait_calls = 0;
    std::uint64_t cache_load_host_block_ns = 0, cache_load_host_block_layers = 0;
    std::uint64_t cache_load_device_block_ns = 0, cache_load_device_block_layers = 0;
    // Host-observed Q frontier to attention completion, including exposed I/O.
    std::uint64_t attention_layer_span_ns = 0, attention_layer_span_layers = 0;
    std::uint64_t attention_consumer_span_ns = 0, attention_consumer_span_layers = 0;
};
} // namespace solid
