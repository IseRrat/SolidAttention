#pragma once

#include "ggml-backend.h"
#include <cstdint>
#include <memory>
#include <optional>

// Returns null for a device without shared DRAM. An allocation failure on a
// supported device is reported as an exception.
extern "C" ggml_backend_buffer_t solid_sycl_alloc_shared_kv(ggml_backend_t backend, size_t bytes);
extern "C" void * solid_sycl_queue(ggml_backend_t backend);

namespace solid {
class SyclCopyEvent {
public:
    virtual ~SyclCopyEvent() = default;
    virtual bool ready() const = 0;
    virtual void wait() = 0;
    virtual void wait_on_compute() = 0;
    virtual std::optional<std::uint64_t> duration() const = 0;
};
class SyclCopyQueue {
public:
    virtual ~SyclCopyQueue() = default;
    virtual void copy(void * target, const void * source, std::size_t bytes) = 0;
    virtual std::unique_ptr<SyclCopyEvent> fence() = 0;
    virtual void drain() = 0;
};
std::unique_ptr<SyclCopyQueue> make_sycl_copy_queue(ggml_backend_t backend);
} // namespace solid
