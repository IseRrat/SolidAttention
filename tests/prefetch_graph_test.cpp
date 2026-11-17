#include "prefetch_fakes.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/llama_adapter.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace test_prefetch;
namespace {
struct Observer {
    solid::GgmlPrefetch * device;
    std::uint32_t completions = 0;
    static bool evaluate(ggml_tensor * tensor, bool ask, void * userdata) {
        auto & self = *static_cast<Observer *>(userdata);
        const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor), "solid_prefetched_flash_attention") == 0;
        if (ask) return attention;
        if (attention) { self.device->completed(self.device->attention_epoch()); ++self.completions; }
        return true;
    }
};
void run(bool disk) {
    constexpr int dimension = 128, kv_heads = 2, query_heads = 4;
    const solid::KvGeometry geometry{2, kv_heads, dimension, 2, 32, 64, solid::KvLayout::interleaved};
    const solid::SparseBudget budget{6, 1, 1, 1};
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<solid::KvStore> store;
    std::unique_ptr<solid::AsyncReader> reader;
    if (disk) {
        const auto path = std::filesystem::temp_directory_path() / ("solid-prefetch-graph-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        store = std::make_unique<solid::FileKvStore>(geometry, solid::create_session_file(path, true), 2048);
        reader = solid::make_uring_reader(2);
    } else { store = std::make_unique<solid::MemoryKvStore>(geometry); reader = std::make_unique<Reader>(); }
    solid::AttentionSession session(std::move(store), query_heads, budget, true, 6, {}, {}, std::move(reader), 6);
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(geometry), query_heads, budget, true, 6);
    const float scale = 1 / std::sqrt(static_cast<float>(dimension));
    std::vector<float> query(query_heads * dimension), key(kv_heads * dimension), value(key.size());
    const auto inputs = [&](int layer, int p) {
        for (int q = 0; q < query_heads; ++q) for (int d = 0; d < dimension; ++d) {
            query[q * dimension + d] = d == 0 ? (q < 2 ? 1.0f : -1.0f) * ((p % 3) ? 1 : -1) : 0;
        }
        for (int h = 0; h < kv_heads; ++h) for (int d = 0; d < dimension; ++d) {
            key[h * dimension + d] = d == 0 ? static_cast<float>(p + layer) : 0;
            value[h * dimension + d] = (p + layer + h + d) * 0.125f;
        }
    };
    for (int p = 0; p < 6; ++p) for (int layer = 0; layer < 2; ++layer) {
        inputs(layer, p);
        (void) session.process(layer, p, 1, query, key, value, scale);
        (void) reference.process(layer, p, 1, query, key, value, scale);
    }
    session.flush();
    solid::GgmlPrefetch device(session, backend.get());
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    auto * q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, query_heads, 1);
    auto * k = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, kv_heads, 1);
    auto * v = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, kv_heads, 1);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    std::shared_ptr<solid::AttentionFrame> frames[2];
    std::unique_ptr<solid::AttentionNode> operations[2];
    ggml_tensor * outputs[2];
    ggml_cgraph * graphs[2];
    for (int layer = 0; layer < 2; ++layer) {
        frames[layer] = std::make_shared<solid::AttentionFrame>();
        operations[layer] = std::make_unique<solid::AttentionNode>(session, frames[layer], layer, scale, nullptr, true, &device);
        outputs[layer] = operations[layer]->build(context.get(), q, k, v);
        require(solid::to_half(-INFINITY) == 0xfc00, "half mask representation");
        graphs[layer] = ggml_new_graph_custom(context.get(), 128, false);
        ggml_build_forward_expand(graphs[layer], outputs[layer]);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(buffer != nullptr, "prefetch graph allocation");
    ggml_backend_t backends[] = {backend.get()};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, nullptr, 1, 256, false, true), ggml_backend_sched_free);
    require(scheduler != nullptr, "prefetch graph scheduler");
    Observer observer{&device};
    ggml_backend_sched_set_eval_callback(scheduler.get(), Observer::evaluate, &observer);
    for (int p = 6; p < 12; ++p) {
        session.begin_prefetch(0, p); // Before current Q/K/V are available.
        for (int layer = 0; layer < 2; ++layer) {
            inputs(layer, p);
            const auto expected = reference.process(layer, p, 1, query, key, value, scale);
            frames[layer]->positions = {static_cast<std::uint64_t>(p)};
            ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
            ggml_backend_tensor_set(k, key.data(), 0, key.size() * sizeof(float));
            ggml_backend_tensor_set(v, value.data(), 0, value.size() * sizeof(float));
            require(ggml_backend_sched_graph_compute(scheduler.get(), graphs[layer]) == GGML_STATUS_SUCCESS, "prefetch graph execution");
            operations[layer]->check();
            std::vector<float> actual(expected.size());
            ggml_backend_tensor_get(outputs[layer], actual.data(), 0, actual.size() * sizeof(float));
            for (std::size_t i = 0; i < actual.size(); ++i) require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-4f,
                "prefetched arbitrary-slot attention differs from synchronous sparse reference");
            ggml_backend_sched_reset(scheduler.get());
        }
    }
    require(device.stats().uploads > 0 && session.stats().prefetch.predicted_blocks > 0, "prefetch path was not exercised");
    require(observer.completions == 12, "scheduler failed to confirm each layer's attention completion");
    if (disk) require(session.stats().prefetch.io.completed_reads > 0, "GGML path did not complete liburing reads");
    device.quiesce(); session.finish();
}
}
int main() {
    try {
        run(false);
        if (solid::uring_available()) run(true);
        std::cout << "multi-layer prefetch, liburing, GQA slot masks and CPU Flash Attention parity passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
