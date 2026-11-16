#include "solid/llama_adapter.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
        solid::KvGeometry geometry{1, 1, 2, 2, 16, 64};
        solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(geometry), 2, {}, false, 3);
        auto frame = std::make_shared<solid::AttentionFrame>();
        frame->positions = {0, 1, 2};
        solid::AttentionNode operation(session, frame, 0, 1 / std::sqrt(2.0f));
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!context) throw std::runtime_error("ggml context");
        auto * packed = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 2, 4, 3);
        ggml_set_input(packed);
        auto * query = ggml_view_3d(context.get(), packed, 2, 2, 3, 2 * sizeof(float), 8 * sizeof(float), 0);
        auto * key = ggml_view_3d(context.get(), packed, 2, 1, 3, 2 * sizeof(float), 8 * sizeof(float), 4 * sizeof(float));
        auto * value = ggml_view_3d(context.get(), packed, 2, 1, 3, 2 * sizeof(float), 8 * sizeof(float), 6 * sizeof(float));
        auto * result = operation.build(context.get(), query, key, value);
        auto * graph = ggml_new_graph_custom(context.get(), 128, false);
        ggml_build_forward_expand(graph, result);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) throw std::runtime_error("CPU backend");
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("graph allocation");
        std::vector<float> data;
        for (int p = 0; p < 3; ++p) {
            const float token[] = {1, 0.5f, -0.5f, 1, static_cast<float>(p + 1), p * 0.5f,
                                   static_cast<float>(p), static_cast<float>(p + 2)};
            data.insert(data.end(), std::begin(token), std::end(token));
        }
        ggml_backend_tensor_set(packed, data.data(), 0, data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph execution");
        operation.check();
        std::vector<float> actual(12);
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        for (int token = 0; token < 3; ++token) for (int head = 0; head < 2; ++head) for (int d = 0; d < 2; ++d) {
            double weights = 0, sum = 0;
            for (int p = 0; p <= token; ++p) {
                const double score = ((head == 0 ? 1 : -0.5) * (p + 1) + (head == 0 ? 0.5 : 1) * p * 0.5) / std::sqrt(2.0);
                const double weight = std::exp(score);
                weights += weight;
                sum += weight * (d == 0 ? p : p + 2);
            }
            if (std::abs(actual[token * 4 + head * 2 + d] - sum / weights) > 1e-5) {
                throw std::runtime_error("strided QKV graph differs from independent attention oracle");
            }
        }
        frame->positions = {4, 5, 6};
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("failure graph execution");
        bool propagated = false;
        try { operation.check(); } catch (const std::exception &) { propagated = true; }
        if (!propagated) throw std::runtime_error("graph callback failure did not propagate");
        std::cout << "strided attention graph and failure propagation checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
