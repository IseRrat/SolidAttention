#pragma once
#include "solid/kv_store.hpp"
#include <memory>

namespace solid {
struct ReadCompletion { std::uint64_t id; std::int64_t result; };
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
};
class AsyncReader {
public:
    virtual ~AsyncReader() = default;
    virtual bool submit(std::uint64_t id, int fd, void * destination, std::size_t bytes, std::uint64_t offset) = 0;
    virtual void cancel(std::uint64_t id) = 0;
    virtual void flush() = 0;
    // Only original READ completion ends a buffer's kernel-write lifetime.
    virtual bool poll(ReadCompletion & completion, bool wait) = 0;
    virtual void drain() = 0;
    virtual AsyncIoStats stats() const = 0;
};
bool uring_available();
std::unique_ptr<AsyncReader> make_uring_reader(std::uint32_t queue_depth);
} // namespace solid
