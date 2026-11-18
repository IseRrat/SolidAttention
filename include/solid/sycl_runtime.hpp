#pragma once

#include <sycl/sycl.hpp>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

namespace solid::sycl_runtime {

inline sycl::queue in_order_queue(const sycl::queue & parent) {
    return sycl::queue(parent.get_context(), parent.get_device(),
        [](sycl::exception_list errors) { for (const auto & error : errors) std::rethrow_exception(error); },
        sycl::property::queue::in_order{});
}

inline std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("SYCL buffer size");
    return a * b;
}

struct UsmDelete {
    sycl::queue * queue = nullptr;
    void operator()(void * pointer) const { sycl::free(pointer, *queue); }
};
template<class T> using Buffer = std::unique_ptr<T, UsmDelete>;

template<class T> Buffer<T> allocate(std::size_t count, sycl::queue & queue) {
    (void) product(count, sizeof(T));
    Buffer<T> buffer(sycl::malloc_device<T>(count, queue), UsmDelete{&queue});
    if (!buffer) throw std::bad_alloc();
    return buffer;
}

inline void validate_pointer(const void * pointer, const sycl::queue & queue) {
    const auto context = queue.get_context();
    const auto kind = sycl::get_pointer_type(pointer, context);
    const auto device = queue.get_device();
    if (kind == sycl::usm::alloc::host || (kind == sycl::usm::alloc::unknown && device.has(sycl::aspect::usm_system_allocations))) return;
    if (kind == sycl::usm::alloc::unknown || sycl::get_pointer_device(pointer, context) != device) {
        throw std::invalid_argument("borrowed SYCL input belongs to another device or context");
    }
}

inline void flag_invalid(int * status) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*status).store(1);
}
} // namespace solid::sycl_runtime
