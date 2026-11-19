#pragma once
#include "solid/kv_store.hpp"
#include <memory>
#include <functional>
#include <stdexcept>

namespace solid {
struct ReadCompletion { std::uint64_t id; std::int64_t result; };
enum class ReadPriority : std::uint8_t { correction_selected, correction_deterministic, selected, deterministic, writeback };
struct ReadDispatch {
    ReadPriority priority=ReadPriority::deterministic;
    bool critical=false;
    std::uint64_t latest_start=UINT64_MAX;
    bool before(const ReadDispatch & other) const {
        const bool correction = priority <= ReadPriority::correction_deterministic;
        const bool other_correction = other.priority <= ReadPriority::correction_deterministic;
        if (correction != other_correction) return correction;
        // Within the same frontier, use Algorithm 1's critical/LST order.
        // Block category only breaks ties; it cannot hide a critical load.
        if (critical!=other.critical) return critical;
        if (latest_start!=other.latest_start) return latest_start<other.latest_start;
        return priority<other.priority;
    }
};
struct AsyncIoStats {
    std::uint64_t read_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t completed_reads = 0;
    std::uint64_t write_requests = 0, write_bytes = 0, write_ns = 0;
    std::uint64_t cancel_requests = 0;
    std::uint64_t cancel_completions = 0;
    std::uint64_t cancel_successes = 0;
    std::uint64_t cancel_not_found = 0;
    std::uint64_t cancel_busy = 0;
    std::uint64_t cancel_errors = 0;
    std::uint64_t completion_ns = 0;
    std::uint64_t queued_cancellations = 0;
    std::uint64_t worker_submissions = 0;
    int worker_cpu = -1;
    bool dedicated_worker = false;
    std::uint32_t worker_capacity=0;
};
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
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()> factory,
                                          std::uint32_t queue_depth,int cpu=-1);
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t queue_depth,int cpu=-1,std::uint32_t admission_capacity=0);
} // namespace solid
