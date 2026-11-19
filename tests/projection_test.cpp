#include "solid/projection.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
void run(ggml_type kt, ggml_type vt, bool biases, bool neox = false) {
    constexpr int width = 256, dimension = 4, heads = 2, tokens = 3, rows = dimension * heads;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    auto * key = ggml_new_tensor_2d(context.get(), kt, width, rows);
    auto * value = ggml_new_tensor_2d(context.get(), vt, width, rows);
    auto * kb = biases ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, rows) : nullptr;
    auto * vb = biases ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, rows) : nullptr;
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, tokens);
    auto * positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, tokens);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weights(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(weights != nullptr, "source allocation");
    std::vector<float> key_values(width * rows), value_values(width * rows), x(width * tokens), key_bias(rows), value_bias(rows);
    for (int i = 0; i < width * rows; ++i) {
        key_values[i] = std::sin(i * 0.17f) * 0.1f;
        value_values[i] = std::cos(i * 0.13f) * 0.2f;
    }
    for (int i = 0; i < width * tokens; ++i) x[i] = std::cos(i * 0.11f);
    for (int i = 0; i < rows; ++i) { key_bias[i] = i * 0.125f; value_bias[i] = -i * 0.25f; }
    std::vector<unsigned char> encodings[2];
    for (int kind = 0; kind < 2; ++kind) {
        auto * tensor = kind == 0 ? key : value;
        auto & data = kind == 0 ? key_values : value_values;
        auto & encoded = encodings[kind];
        encoded.resize(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, data.data(), encoded.data(), 0, rows, width, nullptr);
        ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
        if (tensor->type != GGML_TYPE_F32) {
            for (int row = 0; row < rows; ++row) ggml_get_type_traits(tensor->type)->to_float(encoded.data() + row * tensor->nb[1], data.data() + row * width, width);
        }
    }
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    const int pos[] = {0, 3, 7};
    ggml_backend_tensor_set(positions, pos, 0, sizeof(pos));
    if (biases) {
        ggml_backend_tensor_set(kb, key_bias.data(), 0, key_bias.size() * sizeof(float));
        ggml_backend_tensor_set(vb, value_bias.data(), 0, value_bias.size() * sizeof(float));
    }
    solid::KvProjection projection(key, value, kb, vb, heads, dimension, ggml_backend_get_default_buffer_type(backend.get()));
    require(projection.weight()->type == (kt == vt ? kt : GGML_TYPE_F32), "fused weight type policy");
    if (kt == vt) {
        std::vector<unsigned char> row(key->nb[1]);
        for (int head = 0; head < heads; ++head) for (int kind = 0; kind < 2; ++kind) for (int d = 0; d < dimension; ++d) {
            const auto target = (head * 2 + kind) * dimension + d;
            ggml_backend_tensor_get(projection.weight(), row.data(), target * row.size(), row.size());
            require(std::memcmp(row.data(), encodings[kind].data() + (head * dimension + d) * row.size(), row.size()) == 0,
                    "quantized rows changed during consolidation");
        }
    }
    auto * packed = projection.project(context.get(), input);
    auto * keys = solid::KvProjection::key_view(context.get(), packed);
    auto * roped = ggml_rope_ext(context.get(), keys, positions, nullptr, dimension,
                               neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL, 16, 10000, 1, 0, 1, 32, 1);
    auto * updated = solid::KvProjection::update_key(context.get(), packed, roped);
    auto * graph = ggml_new_graph_custom(context.get(), 128, false);
    ggml_build_forward_expand(graph, updated);
    // Only graph intermediates were created after the source buffer allocation.
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> compute(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(compute != nullptr, "compute allocation");
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "fused graph execution");
    std::vector<float> actual(rows * 2 * tokens);
    ggml_backend_tensor_get(updated, actual.data(), 0, actual.size() * sizeof(float));
    for (int token = 0; token < tokens; ++token) for (int head = 0; head < heads; ++head) {
        double expected_k[dimension], expected_v[dimension];
        for (int d = 0; d < dimension; ++d) {
            const auto row = head * dimension + d;
            expected_k[d] = biases ? key_bias[row] : 0;
            expected_v[d] = biases ? value_bias[row] : 0;
            for (int col = 0; col < width; ++col) {
                expected_k[d] += key_values[row * width + col] * static_cast<double>(x[token * width + col]);
                expected_v[d] += value_values[row * width + col] * static_cast<double>(x[token * width + col]);
            }
        }
        for (int pair = 0; pair < dimension / 2; ++pair) {
            const int a_index = neox ? pair : pair * 2, b_index = neox ? pair + dimension / 2 : pair * 2 + 1;
            const double angle = pos[token] * std::pow(10000.0, -2.0 * pair / dimension);
            const auto a = expected_k[a_index], b = expected_k[b_index];
            expected_k[a_index] = a * std::cos(angle) - b * std::sin(angle);
            expected_k[b_index] = a * std::sin(angle) + b * std::cos(angle);
        }
        // Quantized matmul also quantizes activations; the row encoding test above
        // checks exact preservation, this bound covers the backend's Q8 dot product.
        const double tolerance = ggml_is_quantized(kt) && kt == vt ? 0.02 : 2e-4;
        for (int d = 0; d < dimension; ++d) {
            const auto offset = ((token * heads + head) * 2) * dimension + d;
            require(std::abs(actual[offset] - expected_k[d]) < tolerance, "fused K projection/RoPE differs from oracle");
            require(std::abs(actual[offset + dimension] - expected_v[d]) < tolerance, "fused V rows changed by K RoPE");
        }
    }
    require(projection.stats().buffer_bytes >= ggml_nbytes(projection.weight()), "projection memory accounting");
    const auto source_bytes = ggml_nbytes(key) + ggml_nbytes(value) + (kb ? ggml_nbytes(kb) + ggml_nbytes(vb) : 0);
    require(projection.stats().source_bytes == source_bytes && projection.stats().source_host_bytes == source_bytes,
            "host projection source memory accounting");
    bool rejected = false;
    try { solid::KvProjection invalid(key, value, kb, vb, heads + 1, dimension, ggml_backend_get_default_buffer_type(backend.get())); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid fused projection shape accepted");
}
}
int main() {
    try {
        run(GGML_TYPE_F32, GGML_TYPE_F32, true);
        run(GGML_TYPE_F16, GGML_TYPE_F16, false);
        run(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, true);
        run(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, true, true);
        run(GGML_TYPE_F32, GGML_TYPE_F32, true, true);
        std::cout << "fused projection, quantized rows, mixed types, GQA strides and RoPE checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
