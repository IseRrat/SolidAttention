// 为投影节点绑定设备计时事件，向调度器提供 GPU 实际执行时长。
#pragma once

#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

struct ggml_backend;

namespace solid {
class DeviceInterval {
public:
    virtual ~DeviceInterval() = default;
    virtual void record(void * queue, bool end) = 0;
    virtual std::optional<std::uint64_t> duration() = 0;
};

struct ProjectionIntervals {
    std::unique_ptr<DeviceInterval> query, kv;
};

struct DeviceTimingBinding {
    std::uint64_t magic = 0x534f4c54494d4531ull;
    DeviceInterval * interval = nullptr;
};
inline constexpr std::size_t device_timing_offset = 48;
// 把投影计时对象挂到张量参数，后端执行时在设备队列记录起止事件。
inline void bind_device_timing(ggml_tensor * tensor, DeviceInterval * interval) {
    static_assert(device_timing_offset + sizeof(DeviceTimingBinding) <= GGML_MAX_OP_PARAMS);
    if (!interval) return;
    const DeviceTimingBinding binding{DeviceTimingBinding{}.magic, interval};
    std::memcpy(reinterpret_cast<unsigned char *>(tensor->op_params) + device_timing_offset, &binding, sizeof(binding));
}
inline DeviceInterval * device_timing(const ggml_tensor * tensor) {
    DeviceTimingBinding binding;
    std::memcpy(&binding, reinterpret_cast<const unsigned char *>(tensor->op_params) + device_timing_offset, sizeof(binding));
    return binding.magic == DeviceTimingBinding{}.magic ? binding.interval : nullptr;
}

std::unique_ptr<DeviceInterval> make_cuda_device_interval(int device);
std::unique_ptr<DeviceInterval> make_sycl_device_interval(void * backend_queue);
} // namespace solid

extern "C" void * solid_cuda_queue(ggml_backend * backend);
