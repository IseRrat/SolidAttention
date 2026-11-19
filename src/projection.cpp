#include "solid/projection.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace solid {
namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
void matrix(const ggml_tensor * tensor, std::int64_t rows) {
    if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) || tensor->ne[0] <= 0 ||
        tensor->ne[1] != rows || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        throw std::invalid_argument("KV projection requires contiguous, allocated matrices");
    }
    const auto * traits = ggml_get_type_traits(tensor->type);
    if ((tensor->type != GGML_TYPE_F32 && !traits->to_float) || traits->blck_size_interleave > 1 ||
        tensor->ne[0] % traits->blck_size != 0) throw std::invalid_argument("unsupported KV projection row encoding");
}
void decode(const ggml_tensor * tensor, std::int64_t row, std::vector<unsigned char> & raw, std::vector<float> & values) {
    raw.resize(ggml_row_size(tensor->type, tensor->ne[0]));
    values.resize(static_cast<std::size_t>(tensor->ne[0]));
    ggml_backend_tensor_get(tensor, raw.data(), row * tensor->nb[1], raw.size());
    if (tensor->type == GGML_TYPE_F32) std::memcpy(values.data(), raw.data(), raw.size());
    else ggml_get_type_traits(tensor->type)->to_float(raw.data(), values.data(), tensor->ne[0]);
}
}
struct KvProjection::Impl {
    Context context{nullptr, ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    std::uint32_t heads;
    std::uint32_t dimension;
    ProjectionStats stats;
};
KvProjection::KvProjection(const ggml_tensor * key, const ggml_tensor * value,
                           const ggml_tensor * key_bias, const ggml_tensor * value_bias,
                           std::uint32_t heads, std::uint32_t dimension, ggml_backend_buffer_type * buffer_type) :
    impl_(std::make_unique<Impl>()) {
    const auto start = std::chrono::steady_clock::now();
    const auto rows = static_cast<std::uint64_t>(heads) * dimension;
    if (!buffer_type || !heads || !dimension || rows > std::numeric_limits<std::int32_t>::max() / 2) {
        throw std::invalid_argument("invalid fused KV projection geometry");
    }
    matrix(key, rows);
    matrix(value, rows);
    if (key->ne[0] != value->ne[0]) throw std::invalid_argument("K/V projection input dimensions differ");
    for (const auto * bias : {key_bias, value_bias}) {
        if (bias) matrix(bias, 1);
        if (bias && bias->ne[0] != static_cast<std::int64_t>(rows)) throw std::invalid_argument("KV bias shape");
    }
    impl_->heads = heads;
    impl_->dimension = dimension;
    const auto type = key->type == value->type ? key->type : GGML_TYPE_F32;
    impl_->context.reset(ggml_init({ggml_tensor_overhead() * 3 + 1024, nullptr, true}));
    if (!impl_->context) throw std::runtime_error("fused projection metadata allocation failed");
    impl_->weight = ggml_new_tensor_2d(impl_->context.get(), type, key->ne[0], rows * 2);
    ggml_set_name(impl_->weight, "solid_fused_kv_weight");
    if (key_bias || value_bias) {
        impl_->bias = ggml_new_tensor_1d(impl_->context.get(), GGML_TYPE_F32, rows * 2);
        ggml_set_name(impl_->bias, "solid_fused_kv_bias");
    }
    impl_->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(impl_->context.get(), buffer_type));
    if (!impl_->buffer) throw std::runtime_error("fused projection weight allocation failed");
    ggml_backend_buffer_set_usage(impl_->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<unsigned char> raw;
    std::vector<float> decoded;
    for (std::uint32_t head = 0; head < heads; ++head) for (int kind = 0; kind < 2; ++kind) {
        const auto * source = kind == 0 ? key : value;
        const auto first_source_row=static_cast<std::int64_t>(head)*dimension;
        const auto first_target_row=(static_cast<std::int64_t>(head)*2+kind)*dimension;
        if (type==source->type) {
            // A whole head retains its native row encodings. Bound the host
            // workspace to one head while batching the backend transfers.
            const auto row_bytes=ggml_row_size(source->type,source->ne[0]);
            if (row_bytes>SIZE_MAX/dimension) throw std::overflow_error("fused KV head size overflow");
            const auto bytes=row_bytes*dimension;
            raw.resize(bytes);
            ggml_backend_tensor_get(source,raw.data(),first_source_row*source->nb[1],bytes);
            ggml_backend_tensor_set(impl_->weight,raw.data(),first_target_row*impl_->weight->nb[1],bytes);
        } else {
            for (std::uint32_t d=0;d<dimension;++d) {
                const auto source_row=first_source_row+d;
                const auto target_row=first_target_row+d;
                decode(source, source_row, raw, decoded);
                ggml_backend_tensor_set(impl_->weight, decoded.data(), target_row * impl_->weight->nb[1], decoded.size() * sizeof(float));
            }
        }
    }
    if (impl_->bias) {
        std::vector<float> biases[2];
        for (int kind = 0; kind < 2; ++kind) {
            const auto * source = kind == 0 ? key_bias : value_bias;
            if (source) decode(source, 0, raw, biases[kind]);
            else biases[kind].assign(static_cast<std::size_t>(rows), 0);
        }
        for (std::uint32_t head = 0; head < heads; ++head) for (int kind = 0; kind < 2; ++kind) {
            ggml_backend_tensor_set(impl_->bias, biases[kind].data() + head * dimension,
                                   (static_cast<std::size_t>(head) * 2 + kind) * dimension * sizeof(float), dimension * sizeof(float));
        }
    }
    impl_->stats.buffer_bytes = ggml_backend_buffer_get_size(impl_->buffer.get());
    impl_->stats.source_bytes = ggml_nbytes(key) + ggml_nbytes(value) + (key_bias ? ggml_nbytes(key_bias) : 0) + (value_bias ? ggml_nbytes(value_bias) : 0);
    for (const auto * source : {key, value, key_bias, value_bias}) {
        if (source && ggml_backend_buffer_is_host(source->buffer)) impl_->stats.source_host_bytes += ggml_nbytes(source);
    }
    impl_->stats.weight_type = ggml_type_name(type);
    impl_->stats.initialization_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}
KvProjection::~KvProjection() = default;
ggml_tensor * KvProjection::project(ggml_context * context, ggml_tensor * input) const {
    if (!input || input->ne[0] != impl_->weight->ne[0] || input->ne[2] != 1 || input->ne[3] != 1) {
        throw std::invalid_argument("fused KV projection input shape");
    }
    auto * result = ggml_mul_mat(context, impl_->weight, input);
    ggml_set_name(result, "solid_kv_projection");
    if (impl_->bias) result = ggml_add(context, result, impl_->bias);
    return ggml_reshape_3d(context, result, impl_->dimension * 2, impl_->heads, input->ne[1]);
}
const ProjectionStats & KvProjection::stats() const { return impl_->stats; }
const ggml_tensor * KvProjection::weight() const { return impl_->weight; }
ggml_tensor * KvProjection::key_view(ggml_context * context, ggml_tensor * packed) {
    return ggml_view_3d(context, packed, packed->ne[0] / 2, packed->ne[1], packed->ne[2], packed->nb[1], packed->nb[2], 0);
}
ggml_tensor * KvProjection::value_view(ggml_context * context, ggml_tensor * packed) {
    return ggml_view_3d(context, packed, packed->ne[0] / 2, packed->ne[1], packed->ne[2], packed->nb[1], packed->nb[2], packed->ne[0] / 2 * sizeof(float));
}
ggml_tensor * KvProjection::update_key(ggml_context * context, ggml_tensor * packed, ggml_tensor * key) {
    auto * written = ggml_cpy(context, key, key_view(context, packed));
    ggml_set_name(written, "solid_rope_key_writeback");
    // The view depends on CPY, while view_src keeps the full original allocation.
    return ggml_view_3d(context, written, packed->ne[0], packed->ne[1], packed->ne[2], packed->nb[1], packed->nb[2], 0);
}
} // namespace solid
