// 异步 SSD 读写接口、请求优先级和完成回执；实现分别位于 async_reader.cpp 与 io_worker.cpp。
#pragma once
#include "solid/kv_store.hpp"
#include <memory>
#include <functional>
#include <stdexcept>

namespace solid {
// 原始 I/O 操作的完成回执，包含请求编号与读写结果。
struct ReadCompletion { std::uint64_t id; std::int64_t result; };
enum class ReadPriority : std::uint8_t { correction_selected, correction_deterministic, selected, deterministic, writeback };
struct ReadDispatch {
    ReadPriority priority=ReadPriority::deterministic;
    bool critical=false;
    std::uint64_t latest_start=UINT64_MAX;
    bool before(const ReadDispatch & other) const {
        // 同属 I/O 的就绪任务按算法 1 的关键路径、LST 排序。
        // 请求类别只用于两者均相同时的稳定选择。
        if (critical!=other.critical) return critical;
        if (latest_start!=other.latest_start) return latest_start<other.latest_start;
        return priority<other.priority;
    }
};
struct AsyncIoStats {
    std::uint64_t read_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t completed_reads = 0;
    std::uint64_t write_requests = 0, write_bytes = 0;
    std::uint64_t cancel_requests = 0;
    std::uint64_t cancel_completions = 0;
    std::uint64_t cancel_successes = 0;
    std::uint64_t cancel_not_found = 0;
    std::uint64_t cancel_busy = 0;
    std::uint64_t cancel_errors = 0;
    std::uint64_t queued_cancellations = 0;
    std::uint64_t worker_submissions = 0;
    int worker_cpu = -1;
    bool dedicated_worker = false;
    std::uint32_t worker_capacity=0;
};
// 异步读写约定：submit 只接纳请求，poll 返回完成回执，drain 等待已接纳操作结束。
class AsyncReader {
public:
    virtual ~AsyncReader() = default;
    virtual bool submit(std::uint64_t id, int fd, void * destination, std::size_t bytes, std::uint64_t offset) = 0;
    virtual bool submit_prioritized(std::uint64_t id, int fd, void * destination, std::size_t bytes,
                                    std::uint64_t offset, ReadPriority) { return submit(id,fd,destination,bytes,offset); }
    // Promote admitted jobs that have not entered the kernel yet.
    virtual void prioritize(std::uint64_t, ReadPriority) {}
    virtual bool submit_scheduled(std::uint64_t id,int fd,void * destination,std::size_t bytes,
                                  std::uint64_t offset,ReadDispatch dispatch) {
        return submit_prioritized(id,fd,destination,bytes,offset,dispatch.priority);
    }
    virtual void schedule(std::uint64_t id,ReadDispatch dispatch) { prioritize(id,dispatch.priority); }
    virtual bool supports_write() const { return false; }
    virtual bool submit_write(std::uint64_t, int, const void *, std::size_t, std::uint64_t) {
        throw std::logic_error("I/O engine does not support writes");
    }
    virtual void cancel(std::uint64_t id) = 0;
    virtual void flush() = 0;
    // Only the original operation completion releases the kernel's buffer access.
    // A queued job can complete as canceled before a kernel writer exists.
    virtual bool poll(ReadCompletion & completion, bool wait) = 0;
    virtual void drain() = 0;
    virtual AsyncIoStats stats() const = 0;
    // A readiness descriptor is consumed by the sole I/O owner, before it
    // checks the completion queue, to avoid a lost-wakeup race.
    virtual int completion_handle() const { return -1; }
    virtual void acknowledge_notifications() {}
    virtual void rethrow_failure() const {}
};
bool uring_available();
std::unique_ptr<AsyncReader> make_uring_reader(std::uint32_t queue_depth);
// 给底层引擎配一个独立 CPU 线程，统一持有提交队列和完成队列。
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()> factory,
                                          std::uint32_t queue_depth,int cpu=-1,bool notify_each_completion=false);
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t queue_depth,int cpu=-1,std::uint32_t admission_capacity=0,
                                                       bool notify_each_completion=false);
} // namespace solid
