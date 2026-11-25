#pragma once

#include "solid/kv_store.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace solid {
enum class DeviceMemory { cuda, host_emulator, sycl };

// Borrowed, producer-complete inputs. Retain the allocation through the
// consuming call, or through receipt collection for an asynchronous API.
struct DeviceQuery {
    const float * data = nullptr;
    std::size_t elements = 0;
    DeviceMemory memory = DeviceMemory::cuda;
};
// CUDA producer event accompanying a query that is still being computed.
// The producer owns the query and event through rank completion/receipt collection.
// Plain DeviceQuery retains its producer-complete contract.
struct DeviceQueryWithEvent {
    DeviceQuery query;
    void * cuda_ready_event = nullptr;
};
struct DeviceKeys {
    const Half * data = nullptr;
    std::size_t elements = 0;
    std::size_t head_stride = 0; // Half elements; permits interleaved K/V.
    DeviceMemory memory = DeviceMemory::cuda;
};
// Contiguous token-major interleaved FP16 [token, head, 2, D].
struct DeviceKvBatch {
    const Half * data = nullptr;
    std::size_t elements = 0;
    DeviceMemory memory = DeviceMemory::cuda;
};
struct DeviceOutput {
    float * data = nullptr;
    std::size_t elements = 0;
    DeviceMemory memory = DeviceMemory::cuda;
};
inline std::size_t device_elements(std::uint32_t tokens,std::uint32_t heads,std::uint32_t width,std::uint32_t planes=1) {
    std::size_t count=1;
    for (const auto value:{tokens,heads,width,planes}) {
        if (!value || count>std::numeric_limits<std::size_t>::max()/value) throw std::invalid_argument("device input element count overflow or zero");
        count*=value;
    }
    return count;
}
inline void validate_device_query(DeviceQuery query, DeviceMemory memory, std::uint32_t heads, std::uint32_t width,std::uint32_t tokens=1) {
    if (!query.data || query.memory != memory || heads == 0 || width == 0 ||
        query.elements != device_elements(tokens,heads,width) ||
        reinterpret_cast<std::uintptr_t>(query.data)%alignof(float)) throw std::invalid_argument("device query shape, memory or alignment");
}
inline void validate_device_kv(DeviceKvBatch kv,DeviceMemory memory,std::uint32_t heads,std::uint32_t width,std::uint32_t tokens) {
    if (!kv.data || kv.memory!=memory || kv.elements!=device_elements(tokens,heads,width,2) ||
        reinterpret_cast<std::uintptr_t>(kv.data)%alignof(Half)) throw std::invalid_argument("device KV batch shape, memory or alignment");
}
inline void validate_device_output(DeviceOutput output,DeviceMemory memory,std::uint32_t heads,std::uint32_t width,std::uint32_t tokens) {
    validate_device_query({output.data,output.elements,output.memory},memory,heads,width,tokens);
}
inline bool device_ranges_overlap(const void * a,std::size_t count_a,std::size_t size_a,
                                   const void * b,std::size_t count_b,std::size_t size_b) {
    const auto first_a=reinterpret_cast<std::uintptr_t>(a),first_b=reinterpret_cast<std::uintptr_t>(b);
    if (!size_a || !size_b || count_a>SIZE_MAX/size_a || count_b>SIZE_MAX/size_b || count_a*size_a>UINTPTR_MAX-first_a || count_b*size_b>UINTPTR_MAX-first_b) {
        throw std::invalid_argument("device input address range overflow");
    }
    return first_a<first_b+count_b*size_b && first_b<first_a+count_a*size_a;
}
inline void validate_device_keys(DeviceKeys keys, DeviceMemory memory, std::uint32_t heads, std::uint32_t width) {
    if (!keys.data || keys.memory != memory || heads == 0 || width == 0 || keys.head_stride < width ||
        (heads > 1 && keys.head_stride > (std::numeric_limits<std::size_t>::max()-width)/(heads-1)) ||
        keys.elements < static_cast<std::size_t>(heads-1)*keys.head_stride+width ||
        reinterpret_cast<std::uintptr_t>(keys.data)%alignof(Half)) throw std::invalid_argument("device key shape, stride, memory or alignment");
}
} // namespace solid
