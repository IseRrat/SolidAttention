// 选中 KV 的聚集、掩码和 attention 输入布局检查。
#include "solid/llama_adapter.hpp"
#include "solid/dense_offload.hpp"
#include "solid/device_ops.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

void run(solid::KvLayout layout, bool on_disk, bool packed_inputs, bool sparse, bool infllm = false, bool cached = false,
         bool dense_offload = false) {
        constexpr int dimension = 128, heads = 2, queries = 4;
        solid::KvGeometry geometry{1, heads, dimension, 2, 16, 64, layout};
        geometry.contiguous_planes = cached || dense_offload;
        solid::SparseBudget budget{4, 1, 1, 1};
        std::optional<solid::RelativeRope> rope;
        if (infllm) {
            budget = {6, 1, 3, 2, true, true};
            rope = solid::RelativeRope{dimension, 10000.f, true};
        }
        std::unique_ptr<solid::KvStore> store;
        if (on_disk) {
            const auto path = std::filesystem::temp_directory_path() / ("solid-gather-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            store = std::make_unique<solid::FileKvStore>(geometry, solid::create_session_file(path, true), 2048);
        } else store = std::make_unique<solid::MemoryKvStore>(geometry);
        solid::AttentionSession gathered(std::move(store), queries, budget, sparse, 6, {}, {}, {}, 0, {}, rope);
        solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(geometry), queries, budget, sparse, 6, {}, {}, {}, 0, {}, rope);
        std::vector<float> q(queries * dimension), keys(heads * dimension), values(heads * dimension);
        for (int head = 0; head < queries; ++head) q[head * dimension] = head < 2 ? 1.0f : -1.0f;
        if (rope) for (int head = 0; head < queries; ++head) rope->rotate(q.data() + head * dimension, budget.local - 1);
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
        gathered.flush();
        auto frame = std::make_shared<solid::AttentionFrame>();
        frame->positions = {6};
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        const auto cache = cached ? solid::make_infllm_selected_cache(gathered, backend.get()) : nullptr;
        auto lookahead = cached ? std::make_unique<solid::SparseBaselineLookahead>(gathered, *cache, false) : nullptr;
        auto dense = dense_offload ? std::make_unique<solid::DenseOffload>(gathered, backend.get()) : nullptr;
        solid::AttentionNode operation(gathered, frame, 0, scale, {}, true, nullptr, nullptr, cache.get(), dense.get(), lookahead.get());
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!context) throw std::runtime_error("ggml context");
        auto * qt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, queries, 1);
        auto * packed = packed_inputs ? ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension * 2, heads, 1) : nullptr;
        auto * kt = packed_inputs ? ggml_view_3d(context.get(), packed, dimension, heads, 1, dimension * 2 * sizeof(float), heads * dimension * 2 * sizeof(float), 0) :
            ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, heads, 1);
        auto * vt = packed_inputs ? ggml_view_3d(context.get(), packed, dimension, heads, 1, dimension * 2 * sizeof(float), heads * dimension * 2 * sizeof(float), dimension * sizeof(float)) :
            ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, heads, 1);
        auto * graph = ggml_new_graph_custom(context.get(), 128, false);
        auto * result = operation.build(context.get(), qt, kt, vt, packed, nullptr, backend.get(), graph);
        if (operation.host_node()->type != GGML_TYPE_F16) throw std::runtime_error("gather must retain FP16 precision");
        if (operation.metadata_node()) ggml_build_forward_expand(graph, operation.metadata_node());
        ggml_build_forward_expand(graph, result);
        bool in_attention = false;
        for (int n = 0; n < ggml_graph_n_nodes(graph); ++n) {
            const auto * node = ggml_graph_node(graph, n);
            if (std::string_view(ggml_get_name(node)).find("infllm.QK.") == 0) in_attention = true;
            if (std::string_view(ggml_get_name(node)) == "solid_infllm_recent_scores") in_attention = false;
            if (in_attention && node->op == GGML_OP_CUSTOM && solid::device_op(node) == solid::DeviceOp::none) {
                throw std::runtime_error("CPU metadata interrupted attention before recent probability gather");
            }
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("graph allocation");
        ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
        if (packed) {
            std::vector<float> kv(heads * dimension * 2);
            for (int h = 0; h < heads; ++h) for (int d = 0; d < dimension; ++d) {
                kv[h * dimension * 2 + d] = keys[h * dimension + d];
                kv[h * dimension * 2 + dimension + d] = values[h * dimension + d];
            }
            ggml_backend_tensor_set(packed, kv.data(), 0, kv.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(kt, keys.data(), 0, keys.size() * sizeof(float));
            ggml_backend_tensor_set(vt, values.data(), 0, values.size() * sizeof(float));
        }
        const auto check_output = [&](const std::vector<float> & expected_output) {
            if (dense) dense->begin(frame->positions.front());
            if (lookahead) lookahead->begin(0, frame->positions.front());
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("gathered graph execution");
            operation.check();
            if (dense) { dense->attention_complete(); dense->finish_decode(); }
            if (lookahead) lookahead->finish_decode();
            std::vector<float> actual(expected_output.size());
            ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected_output[i]) > 2e-4f) {
                    throw std::runtime_error("gathered attention differs from streaming reference at " + std::to_string(i));
                }
            }
        };
        const auto before = gathered.stats().io.read_requests;
        check_output(expected);
        if (cached) {
            const auto cold = gathered.stats().io.read_requests;
            frame->positions = {7};
            check_output(reference.process(0, 7, 1, q, keys, values, scale));
            if (cold <= before || gathered.stats().io.read_requests - cold < cold - before) {
                throw std::runtime_error("InfLLM baseline carried historical KV into the next decode step");
            }
            if (gathered.stats().sparse_prefetch_layers != 2) throw std::runtime_error("sparse decode missed fixed window upload");
            return;
        }
        if (dense) {
            frame->positions = {7};
            check_output(reference.process(0, 7, 1, q, keys, values, scale));
            if (gathered.stats().dense_prefetch_layers != 2 || dense->stats().uploads != 2) {
                throw std::runtime_error("dense offload skipped history upload");
            }
            if (dense->stats().upload_bytes != 13ull * heads * dimension * 2 * sizeof(solid::Half)) {
                throw std::runtime_error("dense offload uploaded unused context capacity");
            }
            return;
        }
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("failure graph execution");
        bool failed = false;
        try { operation.check(); } catch (const std::exception &) { failed = true; }
        if (!failed) throw std::runtime_error("gathered callback failure did not propagate");
}
int main() {
    try {
        for (auto layout : {solid::KvLayout::separate, solid::KvLayout::interleaved}) {
            run(layout, false, false, true);
            run(layout, true, true, true);
            run(layout, true, false, false);
            run(layout, true, layout == solid::KvLayout::interleaved, true, true);
        }
        run(solid::KvLayout::separate, true, false, true, true, true);
        run(solid::KvLayout::separate, true, false, false, false, false, true);
        std::cout << "SSD gather, dense layer lookahead, cached InfLLM, FP16 layouts and Flash Attention passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
