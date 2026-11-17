#pragma once
#include "solid/kv_store.hpp"
#include <memory>
#include <functional>

namespace solid {
struct ReadCompletion { std::uint64_t id; std::int64_t result; };
enum class ReadPriority : std::uint8_t { correction_selected, correction_deterministic, selected, deterministic };
struct AsyncIoStats {
    std::uint64_t read_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t completed_reads = 0;
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
    virtual void cancel(std::uint64_t id) = 0;
    virtual void flush() = 0;
    // Only original READ completion ends a buffer's kernel-write lifetime.
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
