// GGML 自定义操作的 SYCL 实现，处理 KV 整理、局部 Key 和完成通知。
#pragma once

#include "solid/device_ops.hpp"
#include <sycl/sycl.hpp>

namespace solid {
inline bool sycl_device_op(sycl::queue & queue, ggml_tensor * output) {
    if (!device_op_supported(output)) return false;
    switch (device_op(output)) {
        case DeviceOp::local_keys: {
            const auto binding = local_keys_binding(output);
            const auto * ring = reinterpret_cast<const std::uint16_t *>(binding.ring);
            const auto * current = static_cast<const std::uint16_t *>(output->src[0]->data);
            const auto * position = static_cast<const std::int32_t *>(output->src[1]->data);
            auto * target = static_cast<std::uint16_t *>(output->data);
            const auto width = output->ne[0], tokens = output->ne[1], heads = output->ne[2], queries = output->src[0]->ne[2];
            queue.parallel_for(sycl::range<1>(static_cast<std::size_t>(ggml_nelements(output))), [=](sycl::id<1> index) {
                const auto i = static_cast<std::int64_t>(index[0]);
                const auto first = static_cast<std::int64_t>(position[0]);
                const auto p = first + queries - tokens + i / width % tokens;
                const auto head = i / width / tokens, d = i % width;
                target[i] = p < 0 ? std::uint16_t{0} : p >= first ? current[((p - first) * heads + head) * 2 * width + d] :
                    ring[(p % binding.capacity * heads + head) * width + d];
            });
            return true;
        }
        case DeviceOp::query_token:
        case DeviceOp::kv_token:
            if (auto * interval=device_timing(output)) interval->record(&queue,true);
            queue.fill(static_cast<float *>(output->data), 0.0f, 1);
            return true;
        case DeviceOp::release_mask:
        case DeviceOp::prefill_result:
        case DeviceOp::attention_result:
            if (output->data != output->src[0]->data) queue.memcpy(output->data, output->src[0]->data, ggml_nbytes(output));
            return true;
        case DeviceOp::consolidate_kv: {
            const auto * key = output->src[0];
            const auto * value = output->src[1];
            const auto * k = static_cast<const char *>(key->data);
            const auto * v = static_cast<const char *>(value->data);
            const auto kh = key->nb[1], kt = key->nb[2];
            const auto vh = value->nb[1], vt = value->nb[2];
            const auto width = static_cast<std::size_t>(key->ne[0]);
            const auto heads = static_cast<std::size_t>(key->ne[1]);
            auto * packed = static_cast<sycl::half *>(output->data);
            queue.parallel_for(sycl::range<1>(static_cast<std::size_t>(ggml_nelements(output))), [=](sycl::id<1> i) {
                const auto token = i[0] / (2 * width * heads);
                const auto head = (i[0] / (2 * width)) % heads;
                const auto * row = (i[0] / width) % 2 ? v + token*vt + head*vh : k + token*kt + head*kh;
                packed[i[0]] = sycl::half(reinterpret_cast<const float *>(row)[i[0] % width]);
            });
            return true;
        }
        default:
            return false;
    }
}
} // namespace solid
