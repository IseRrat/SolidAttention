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
        constexpr int dimension = 128, heads = 2, queries = 4;
        solid::KvGeometry geometry{1, heads, dimension, 2, 16, 64};
        solid::SparseBudget budget{4, 1, 1, 1};
        solid::AttentionSession gathered(std::make_unique<solid::MemoryKvStore>(geometry), queries, budget, true, 6);
        solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(geometry), queries, budget, true, 6);
        std::vector<float> q(queries * dimension), keys(heads * dimension), values(heads * dimension);
        for (int head = 0; head < queries; ++head) q[head * dimension] = head < 2 ? 1.0f : -1.0f;
        const float scale = 1 / std::sqrt(static_cast<float>(dimension));
        for (int p = 0; p < 6; ++p) {
            for (int head = 0; head < heads; ++head) for (int d = 0; d < dimension; ++d) {
                keys[head * dimension + d] = d == 0 ? static_cast<float>(p + 1) : 0;
                values[head * dimension + d] = (p + head + d) * 0.125f;
            }
            (void) gathered.process(0, p, 1, q, keys, values, scale);
            (void) reference.process(0, p, 1, q, keys, values, scale);
        }
        for (int head = 0; head < heads; ++head) for (int d = 0; d < dimension; ++d) {
            keys[head * dimension + d] = d == 0 ? 7.0f : 0;
            values[head * dimension + d] = (6 + head + d) * 0.125f;
        }
        const auto expected = reference.process(0, 6, 1, q, keys, values, scale);
        auto frame = std::make_shared<solid::AttentionFrame>();
        frame->positions = {6};
        solid::AttentionNode operation(gathered, frame, 0, scale, {}, true);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!context) throw std::runtime_error("ggml context");
        auto * qt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, queries, 1);
        auto * kt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, heads, 1);
        auto * vt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, heads, 1);
        auto * result = operation.build(context.get(), qt, kt, vt);
        if (operation.host_node()->type != GGML_TYPE_F16) throw std::runtime_error("gather must retain FP16 precision");
        auto * graph = ggml_new_graph_custom(context.get(), 128, false);
        ggml_build_forward_expand(graph, result);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("graph allocation");
        ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
        ggml_backend_tensor_set(kt, keys.data(), 0, keys.size() * sizeof(float));
        ggml_backend_tensor_set(vt, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("gathered graph execution");
        operation.check();
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-4f) {
                throw std::runtime_error("gathered Flash Attention differs from streaming reference at " + std::to_string(i));
            }
        }
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("failure graph execution");
        bool failed = false;
        try { operation.check(); } catch (const std::exception &) { failed = true; }
        if (!failed) throw std::runtime_error("gathered callback failure did not propagate");
        std::cout << "FP16 gather, GQA masks, Flash Attention and failure checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
