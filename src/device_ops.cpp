#include "solid/device_ops.hpp"
#include "solid/kv_store.hpp"
#include <stdexcept>

namespace solid {
namespace {
void forward(ggml_tensor * output, int ith, int, void *) {
    if (ith) return;
    GGML_ASSERT(device_op_supported(output));
    switch (device_op(output)) {
        case DeviceOp::query_token:
        case DeviceOp::kv_token: *static_cast<float *>(output->data) = 0; break;
        case DeviceOp::release_mask:
        case DeviceOp::prefill_result:
            if (output->data!=output->src[0]->data) std::memcpy(output->data, output->src[0]->data, ggml_nbytes(output));
            break;
        case DeviceOp::consolidate_kv: {
            auto * target = static_cast<Half *>(output->data);
            const auto width = output->ne[0]/2;
            for (std::int64_t p=0;p<output->ne[2];++p) for (std::int64_t h = 0; h < output->ne[1]; ++h) for (int plane = 0; plane < 2; ++plane) {
                const auto * source = static_cast<const unsigned char *>(output->src[plane]->data)+p*output->src[plane]->nb[2]+h*output->src[plane]->nb[1];
                for (std::int64_t d = 0; d < width; ++d) {
                    float value; std::memcpy(&value, source+d*sizeof(float), sizeof(value));
                    target[((p*output->ne[1]+h)*2+plane)*width+d] = to_half(value);
                }
            }
            break;
        }
        default: GGML_ABORT("unknown SolidAttention device operation");
    }
}
ggml_tensor * make(ggml_context * context, ggml_type type, const std::int64_t * shape, ggml_tensor ** sources,
                   int count, DeviceOp kind, const char * name) {
    auto * output = kind==DeviceOp::prefill_result?
        ggml_custom_inplace(context,sources[0],sources+1,count-1,forward,1,nullptr):
        ggml_custom_4d(context,type,shape[0],shape[1],shape[2],shape[3],sources,count,forward,1,nullptr);
    const DeviceOpBinding binding{DeviceOpBinding{}.magic,1,kind};
    std::memcpy(reinterpret_cast<unsigned char *>(output->op_params)+device_op_offset,&binding,sizeof(binding));
    if (!device_op_supported(output)) throw std::invalid_argument("unsupported device operation shape or dependencies");
    ggml_set_name(output,name);
    return output;
}
}
ggml_tensor * device_query_token(ggml_context * context, ggml_tensor * query) {
    const std::int64_t shape[]{1,1,1,1};
    return make(context,GGML_TYPE_F32,shape,&query,1,DeviceOp::query_token,"solid_device_query_token");
}
ggml_tensor * device_kv_token(ggml_context * context, ggml_tensor * kv) {
    const std::int64_t shape[]{1,1,1,1};
    return make(context,GGML_TYPE_F32,shape,&kv,1,DeviceOp::kv_token,"solid_device_kv_ready");
}
ggml_tensor * device_consolidate_kv(ggml_context * context, ggml_tensor * key, ggml_tensor * value,
                                   ggml_tensor * query, ggml_tensor * early) {
    const std::int64_t shape[]{2*key->ne[0],key->ne[1],key->ne[2],1};
    ggml_tensor * sources[]{key,value,query,early};
    return make(context,GGML_TYPE_F16,shape,sources,4,DeviceOp::consolidate_kv,"solid_device_kv_fp16");
}
ggml_tensor * device_release_mask(ggml_context * context, ggml_tensor * mask, ggml_tensor * kv, ggml_tensor * query) {
    ggml_tensor * sources[]{mask,kv,query};
    return make(context,GGML_TYPE_F16,mask->ne,sources,query?3:2,DeviceOp::release_mask,"solid_device_release_mask");
}
ggml_tensor * device_prefill_result(ggml_context * context,ggml_tensor * output,ggml_tensor * control,ggml_tensor * query,ggml_tensor * kv) {
    ggml_tensor * sources[]{output,control,query,kv};
    return make(context,GGML_TYPE_F32,output->ne,sources,4,DeviceOp::prefill_result,"solid_device_prefill_result");
}
} // namespace solid
