// CUDA 局部 Key 聚集：从 raw-K 环与本组新 K 生成近期 attention 所需的连续行。
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace solid::device_ops_detail {
// Gather the existing InfLLM raw-K window and this group's new keys. Negative
// prefix positions are padding. No SSD read or persistent KV copy is added.
static __global__ void gather_local_keys(const std::uint16_t * ring, const std::uint16_t * current,
    std::uint16_t * output, const std::int32_t * position, std::uint64_t first_query,
    std::uint32_t queries, std::uint32_t tokens, std::uint32_t capacity,
    std::uint32_t heads, std::uint32_t width) {
    const auto first = position ? static_cast<std::int64_t>(position[0]) : static_cast<std::int64_t>(first_query);
    const auto origin = first + queries - tokens;
    const auto size = static_cast<std::uint64_t>(tokens) * heads * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto column = i % width, token = (i / width) % tokens, head = i / width / tokens;
        const auto absolute = origin + static_cast<std::int64_t>(token);
        std::uint16_t value = 0;
        if (absolute >= first) value = current[((absolute - first) * heads + head) * 2 * width + column];
        else if (absolute >= 0) value = ring[(static_cast<std::uint64_t>(absolute) % capacity * heads + head) * width + column];
        output[i] = value;
    }
}
}
