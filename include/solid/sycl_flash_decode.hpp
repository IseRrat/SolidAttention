#pragma once

#include "ggml.h"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace solid {
inline constexpr std::size_t sycl_decode_tile = 64;
void sycl_attention_decode(sycl::queue & queue,ggml_tensor * output);

inline bool sycl_masked_decode_supported(const ggml_tensor * output) {
    constexpr char name[] = "solid_prefetched_flash_attention";
    if (!output || output->op != GGML_OP_FLASH_ATTN_EXT ||
        (std::strncmp(output->name, name, sizeof(name)-1) != 0 && std::strcmp(output->name,"solid_gathered_flash_attention") != 0) ||
        output->src[4] || output->src[5]) return false;
    const auto * q=output->src[0], * k=output->src[1], * v=output->src[2], * mask=output->src[3];
    if (!q || !k || !v || !mask || q->type!=GGML_TYPE_F32 || k->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 ||
        mask->type!=GGML_TYPE_F16 || output->type!=GGML_TYPE_F32 || q->ne[1]!=1 || q->ne[3]!=1 ||
        q->ne[0]<1 || q->ne[0]>512 || k->ne[0]!=q->ne[0] || k->ne[1]<1 || k->ne[2]<1 || k->ne[3]!=1 ||
        !ggml_are_same_shape(k,v) || q->ne[2]<1 || q->ne[2]%k->ne[2] ||
        mask->ne[0]!=k->ne[1] || mask->ne[1]!=1 || mask->ne[2]!=q->ne[2] || mask->ne[3]!=1 ||
        output->ne[0]!=q->ne[0] || output->ne[1]!=q->ne[2] || output->ne[2]!=1 || output->ne[3]!=1 ||
        q->nb[0]!=sizeof(float) || q->nb[2]%sizeof(float) || k->nb[0]!=sizeof(sycl::half) ||
        v->nb[0]!=sizeof(sycl::half) || k->nb[1]%sizeof(sycl::half) || v->nb[1]%sizeof(sycl::half) ||
        k->nb[2]%sizeof(sycl::half) || v->nb[2]%sizeof(sycl::half) ||
        !ggml_is_contiguous(mask) || !ggml_is_contiguous(output)) return false;
    float params[3];
    std::memcpy(params,output->op_params,sizeof(params));
    return std::isfinite(params[0]) && params[0]>0 && params[1]==0 && params[2]==0;
}

inline std::size_t sycl_masked_decode_alloc(const ggml_tensor * output) {
    const auto tokens=static_cast<std::size_t>(output->src[1]->ne[1]);
    const auto tiles=(tokens+sycl_decode_tile-1)/sycl_decode_tile;
    const auto row_bytes=static_cast<std::size_t>(output->ne[0]+2)*sizeof(float);
    const auto bytes=ggml_nbytes(output);
    if (static_cast<std::size_t>(output->ne[1])>(std::numeric_limits<std::size_t>::max()-bytes)/row_bytes/tiles) {
        throw std::overflow_error("SYCL decode workspace exceeds address space");
    }
    return bytes+static_cast<std::size_t>(output->ne[1])*tiles*row_bytes;
}

inline sycl::event sycl_masked_decode(sycl::queue & queue, ggml_tensor * output) {
    const auto * q=output->src[0], * k=output->src[1], * v=output->src[2], * mask=output->src[3];
    const auto width=static_cast<std::size_t>(q->ne[0]), heads=static_cast<std::size_t>(q->ne[2]);
    const auto gqa=heads/static_cast<std::size_t>(k->ne[2]), tokens=static_cast<std::size_t>(k->ne[1]);
    const auto tiles=(tokens+sycl_decode_tile-1)/sycl_decode_tile;
    const auto q_stride=q->nb[2]/sizeof(float), k_head=k->nb[2]/sizeof(sycl::half), v_head=v->nb[2]/sizeof(sycl::half);
    const auto k_token=k->nb[1]/sizeof(sycl::half), v_token=v->nb[1]/sizeof(sycl::half);
    const auto * queries=static_cast<const float *>(q->data);
    const auto * keys=static_cast<const sycl::half *>(k->data), * values=static_cast<const sycl::half *>(v->data);
    const auto * masks=static_cast<const sycl::half *>(mask->data);
    auto * result=static_cast<float *>(output->data), * scratch=result+width*heads;
    float scale;
    std::memcpy(&scale,output->op_params,sizeof(scale));
    constexpr std::size_t lanes=32;
    constexpr float negative_inf=-std::numeric_limits<float>::infinity();

    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(heads*tiles*lanes),sycl::range<1>(lanes)), [=](sycl::nd_item<1> item) {
        const auto group=item.get_group_linear_id(), lane=item.get_local_linear_id();
        const auto head=group/tiles, kv_head=head/gqa, first=(group%tiles)*sycl_decode_tile;
        const auto * query=queries+head*q_stride;
        auto * partial=scratch+group*(width+2);
        float cached_query[512/lanes], numerator[512/lanes]{};
        for (std::size_t d=lane;d<width;d+=lanes) cached_query[d/lanes]=query[d];
        float maximum=negative_inf, denominator=0.f;
        for (std::size_t token=first;token<tokens && token<first+sycl_decode_tile;++token) {
            const float bias=static_cast<float>(masks[head*tokens+token]);
            if (bias==negative_inf) continue;
            const auto * key=keys+kv_head*k_head+token*k_token;
            const auto * value=values+kv_head*v_head+token*v_token;
            float dot=0.f;
            for (std::size_t d=lane;d<width;d+=lanes) dot+=cached_query[d/lanes]*static_cast<float>(key[d]);
            const float score=scale*sycl::reduce_over_group(item.get_group(),dot,sycl::plus<float>())+bias;
            const float next_max=sycl::fmax(maximum,score);
            const float keep=denominator>0.f?sycl::exp(maximum-next_max):0.f;
            const float weight=sycl::exp(score-next_max);
            for (std::size_t d=lane;d<width;d+=lanes) numerator[d/lanes]=keep*numerator[d/lanes]+weight*static_cast<float>(value[d]);
            denominator=keep*denominator+weight;
            maximum=next_max;
        }
        for (std::size_t d=lane;d<width;d+=lanes) partial[d]=numerator[d/lanes];
        if (lane==0) { partial[width]=maximum; partial[width+1]=denominator; }
    });

    return queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(heads*lanes),sycl::range<1>(lanes)), [=](sycl::nd_item<1> item) {
        const auto head=item.get_group_linear_id(), lane=item.get_local_linear_id();
        const auto * head_tiles=scratch+head*tiles*(width+2);
        float maximum=negative_inf;
        for (std::size_t t=0;t<tiles;++t) maximum=sycl::fmax(maximum,head_tiles[t*(width+2)+width]);
        float numerator[512/lanes]{}, denominator=0.f;
        for (std::size_t t=0;t<tiles;++t) {
            const auto * partial=head_tiles+t*(width+2);
            if (partial[width+1]==0.f) continue;
            const float weight=sycl::exp(partial[width]-maximum);
            denominator+=weight*partial[width+1];
            for (std::size_t d=lane;d<width;d+=lanes) numerator[d/lanes]+=weight*partial[d];
        }
        for (std::size_t d=lane;d<width;d+=lanes) result[head*width+d]=denominator>0.f?numerator[d/lanes]/denominator:0.f;
    });
}
} // namespace solid
