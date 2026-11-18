#pragma once

#include "solid/device_ops.hpp"
#include <sycl/sycl.hpp>

namespace solid {
inline bool sycl_device_op(sycl::queue & queue, ggml_tensor * output) {
    if (!device_op_supported(output)) return false;
    switch (device_op(output)) {
        case DeviceOp::query_token:
        case DeviceOp::kv_token:
            queue.fill(static_cast<float *>(output->data), 0.0f, 1);
            return true;
        case DeviceOp::release_mask:
        case DeviceOp::prefill_result:
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
