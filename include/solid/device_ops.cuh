#pragma once

#include "solid/device_ops.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace solid {
namespace device_ops_detail {
static __global__ void query_token(float * output) { *output = 0; }
static __global__ void consolidate(const char * key, const char * value, __half * output,
                                    std::size_t key_stride, std::size_t value_stride,std::size_t key_token_stride,std::size_t value_token_stride,
                                    std::uint64_t width, std::uint64_t heads,std::uint64_t tokens) {
    for (auto i = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < tokens*heads*2*width; i += static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto token=i/(heads*2*width),head=(i/(2*width))%heads,d=i%width;
        const auto * row = (i/width)%2 ? value+token*value_token_stride+head*value_stride : key+token*key_token_stride+head*key_stride;
        output[i] = __float2half_rn(reinterpret_cast<const float *>(row)[d]);
    }
}
static __global__ void copy_result(const std::uint32_t * source,std::uint32_t * output,std::uint64_t count) {
    for (auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
         i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) output[i]=source[i];
}
static __global__ void copy_mask(const std::uint16_t * source, std::uint16_t * output, std::uint64_t count) {
    for (auto i = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(gridDim.x)*blockDim.x) output[i] = source[i];
}
}
inline cudaError_t cuda_device_op(ggml_tensor * output, cudaStream_t stream) {
    if (!device_op_supported(output)) return cudaErrorInvalidValue;
    const auto count = static_cast<std::uint64_t>(ggml_nelements(output));
    const auto grid = static_cast<unsigned>((count+255)/256 > 65535 ? 65535 : (count+255)/256);
    switch (device_op(output)) {
        case DeviceOp::query_token:
        case DeviceOp::kv_token:
            if (auto * interval=device_timing(output)) interval->record(stream,true);
            device_ops_detail::query_token<<<1,1,0,stream>>>(static_cast<float *>(output->data));
            break;
        case DeviceOp::consolidate_kv:
            device_ops_detail::consolidate<<<grid,256,0,stream>>>(static_cast<const char *>(output->src[0]->data),
                static_cast<const char *>(output->src[1]->data),static_cast<__half *>(output->data),output->src[0]->nb[1],
                output->src[1]->nb[1],output->src[0]->nb[2],output->src[1]->nb[2],output->ne[0]/2,output->ne[1],output->ne[2]); break;
        case DeviceOp::release_mask:
            device_ops_detail::copy_mask<<<grid,256,0,stream>>>(static_cast<const std::uint16_t *>(output->src[0]->data),
                static_cast<std::uint16_t *>(output->data),count); break;
        case DeviceOp::prefill_result:
            if (output->data==output->src[0]->data) return cudaSuccess;
            device_ops_detail::copy_result<<<grid,256,0,stream>>>(static_cast<const std::uint32_t *>(output->src[0]->data),
                static_cast<std::uint32_t *>(output->data),count); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
} // namespace solid
