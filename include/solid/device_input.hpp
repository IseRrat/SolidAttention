#pragma once

#include "solid/kv_store.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace solid {
enum class DeviceMemory { cuda, host_emulator };

// Borrowed, producer-complete inputs. The owner retains the allocation until
// the consuming call returns, including its private-stream synchronization.
struct DeviceQuery {
    const float * data = nullptr;
    std::size_t elements = 0;
    DeviceMemory memory = DeviceMemory::cuda;
};
struct DeviceKeys {
    const Half * data = nullptr;
    std::size_t elements = 0;
    std::size_t head_stride = 0; // Half elements; permits interleaved K/V.
    DeviceMemory memory = DeviceMemory::cuda;
};
inline void validate_device_query(DeviceQuery query, DeviceMemory memory, std::uint32_t heads, std::uint32_t width) {
    if (!query.data || query.memory != memory || heads == 0 || width == 0 ||
        heads > std::numeric_limits<std::size_t>::max()/width || query.elements != static_cast<std::size_t>(heads)*width ||
        reinterpret_cast<std::uintptr_t>(query.data)%alignof(float)) throw std::invalid_argument("device query shape, memory or alignment");
}
inline void validate_device_keys(DeviceKeys keys, DeviceMemory memory, std::uint32_t heads, std::uint32_t width) {
    if (!keys.data || keys.memory != memory || heads == 0 || width == 0 || keys.head_stride < width ||
        (heads > 1 && keys.head_stride > (std::numeric_limits<std::size_t>::max()-width)/(heads-1)) ||
        keys.elements < static_cast<std::size_t>(heads-1)*keys.head_stride+width ||
        reinterpret_cast<std::uintptr_t>(keys.data)%alignof(Half)) throw std::invalid_argument("device key shape, stride, memory or alignment");
}
} // namespace solid
