#pragma once

#include "ggml.h"
#include "solid/device_timing.hpp"
#include "solid/stream_completion.hpp"
#include <cstdint>
#include <cstring>

namespace solid {
enum class DeviceOp : std::uint32_t { none = 0, query_token = 1, consolidate_kv = 2, release_mask = 3, prefill_result = 4, kv_token = 5, attention_result = 6 };
struct DeviceOpBinding {
    std::uint64_t magic = 0x534f4c49444f5031ull;
    std::uint32_t version = 1;
    DeviceOp kind = DeviceOp::none;
};
inline constexpr std::size_t device_op_offset = 32;
inline DeviceOp device_op(const ggml_tensor * tensor) {
    static_assert(device_op_offset+sizeof(DeviceOpBinding) <= GGML_MAX_OP_PARAMS, "device operation tag exceeds op params");
    if (!tensor || tensor->op != GGML_OP_CUSTOM) return DeviceOp::none;
    DeviceOpBinding binding{};
    std::memcpy(&binding, reinterpret_cast<const unsigned char *>(tensor->op_params)+device_op_offset, sizeof(binding));
    return binding.magic == DeviceOpBinding{}.magic && binding.version == 1 ? binding.kind : DeviceOp::none;
}
inline bool device_op_supported(const ggml_tensor * tensor) {
    if (!tensor || !ggml_is_contiguous(tensor)) return false;
    const auto * a = tensor->src[0];
    const auto vector = [](const ggml_tensor * source) {
        return source && source->type == GGML_TYPE_F32 && source->ne[0] > 0 && source->ne[1] > 0 &&
            source->ne[2] > 0 && source->ne[3] == 1 && source->nb[0] == sizeof(float) &&
            source->nb[1] >= static_cast<std::size_t>(source->ne[0])*sizeof(float) &&
            source->nb[2] >= static_cast<std::size_t>(source->ne[1])*source->nb[1];
    };
    switch (device_op(tensor)) {
        case DeviceOp::attention_result:
            return a && a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
                tensor->type == GGML_TYPE_F32 && ggml_are_same_shape(a, tensor) &&
                tensor->src[1] && tensor->src[1]->type == GGML_TYPE_F32 && ggml_nelements(tensor->src[1]) == 1;
        case DeviceOp::query_token:
            return vector(a) && ggml_is_contiguous(a) && tensor->type == GGML_TYPE_F32 && ggml_nelements(tensor) == 1;
        case DeviceOp::kv_token:
            return device_op(a)==DeviceOp::consolidate_kv && ggml_is_contiguous(a) && tensor->type==GGML_TYPE_F32 && ggml_nelements(tensor)==1;
        case DeviceOp::consolidate_kv: {
            const auto * b = tensor->src[1];
            const auto * q = tensor->src[2];
            const auto * frontier = tensor->src[3];
            return vector(a) && vector(b) && ggml_are_same_shape(a,b) && vector(q) && ggml_is_contiguous(q) &&
                q->ne[0] == a->ne[0] && q->ne[1]%a->ne[1] == 0 && q->ne[2] == a->ne[2] && frontier && frontier->type == GGML_TYPE_F32 &&
                ggml_nelements(frontier) == 1 && tensor->type == GGML_TYPE_F16 && tensor->ne[0] == 2*a->ne[0] &&
                tensor->ne[1] == a->ne[1] && tensor->ne[2] == a->ne[2] && tensor->ne[3] == 1;
        }
        case DeviceOp::release_mask:
            return a && a->type == GGML_TYPE_F16 && tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(a) &&
                ggml_are_same_shape(a,tensor) && device_op(tensor->src[1]) == DeviceOp::consolidate_kv &&
                (!tensor->src[2] || (vector(tensor->src[2]) && ggml_is_contiguous(tensor->src[2])));
        case DeviceOp::prefill_result: {
            const auto * frontier=tensor->src[1],* q=tensor->src[2],* kv=tensor->src[3];
            return a && a->type==GGML_TYPE_F32 && ggml_is_contiguous(a) && tensor->type==GGML_TYPE_F32 &&
                ggml_are_same_shape(a,tensor) && frontier && frontier->type==GGML_TYPE_F32 && ggml_nelements(frontier)==1 &&
                vector(q) && ggml_is_contiguous(q) && device_op(kv)==DeviceOp::consolidate_kv &&
                tensor->ne[0]==q->ne[0]*q->ne[1] && tensor->ne[1]==q->ne[2] && tensor->ne[2]==1 && tensor->ne[3]==1 &&
                kv->ne[0]==2*q->ne[0] && kv->ne[2]==q->ne[2];
        }
        default: return false;
    }
}
// Pure graph operations: no CPU coordination side effects in captured kernels.
ggml_tensor * device_query_token(ggml_context * context, ggml_tensor * query, ggml_tensor * dependency = nullptr);
ggml_tensor * device_attention_result(ggml_context * context, ggml_tensor * output, ggml_tensor * receipt,
    ggml_tensor * kv = nullptr, ggml_tensor * scores = nullptr);
ggml_tensor * device_kv_token(ggml_context * context, ggml_tensor * kv);
ggml_tensor * device_consolidate_kv(ggml_context * context, ggml_tensor * key, ggml_tensor * value,
                                   ggml_tensor * query, ggml_tensor * early_selection);
ggml_tensor * device_release_mask(ggml_context * context, ggml_tensor * mask, ggml_tensor * kv, ggml_tensor * query = nullptr);
ggml_tensor * device_prefill_result(ggml_context * context,ggml_tensor * output,ggml_tensor * control,ggml_tensor * query,ggml_tensor * kv);
} // namespace solid
