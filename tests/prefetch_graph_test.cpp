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
#include <condition_variable>
#include <fstream>
#include "nlohmann/json.hpp"

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
struct OverlapBarrier {
    std::mutex mutex;
    std::condition_variable cv;
    bool waiting = false, released = false;
    void io_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (released) return;
        waiting = true; cv.notify_all(); cv.wait(lock, [&] { return released; });
    }
    void projection() {
        std::unique_lock<std::mutex> lock(mutex);
        if (released) return;
        cv.wait(lock, [&] { return waiting; }); released = true; cv.notify_all();
    }
    static void compute(ggml_tensor * dst, int ith, int, void * userdata) {
        if (ith) return;
        static_cast<OverlapBarrier *>(userdata)->projection();
        std::memcpy(dst->data, dst->src[0]->data, ggml_nbytes(dst));
    }
};
void run(bool disk, solid::ScheduleMode mode, const std::filesystem::path & traces) {
    constexpr int dimension = 128, kv_heads = 2, query_heads = 4;
    const solid::KvGeometry geometry{2, kv_heads, dimension, 2, 32, 64, solid::KvLayout::interleaved};
    const solid::SparseBudget budget{6, 1, 1, 1};
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<solid::KvStore> store;
    std::unique_ptr<solid::AsyncReader> reader;
    OverlapBarrier barrier;
    const bool controlled = !disk && mode != solid::ScheduleMode::naive;
    if (disk) {
        const auto path = std::filesystem::temp_directory_path() / ("solid-prefetch-graph-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        store = std::make_unique<solid::FileKvStore>(geometry, solid::create_session_file(path, true), 2048);
        reader = solid::make_uring_reader(2);
    } else {
        auto file = std::make_unique<File>();
        auto fake = std::make_unique<Reader>(file.get());
        if (controlled) fake->before_wait = [&] { barrier.io_wait(); };
        store = std::make_unique<solid::FileKvStore>(geometry, std::move(file), 2048);
        reader = std::move(fake);
    }
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
    const auto directory = traces.empty() ? std::filesystem::path{} : traces / (disk ? "uring" : "memory");
    if (!directory.empty()) std::filesystem::create_directory(directory);
    solid::GgmlPrefetch device(session, backend.get(), directory.empty() ? std::filesystem::path{} : directory / "residency.jsonl",
        mode, directory.empty() ? std::filesystem::path{} : directory / "schedule.jsonl");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    auto * q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, query_heads, 1);
    ggml_set_input(q);
    ggml_tensor * keys[2]; ggml_tensor * values[2];
    std::shared_ptr<solid::AttentionFrame> frames[2];
    std::unique_ptr<solid::AttentionNode> operations[2];
    ggml_tensor * outputs[2];
    auto * graph = ggml_new_graph_custom(context.get(), 512, false);
    for (int layer = 0; layer < 2; ++layer) {
        frames[layer] = std::make_shared<solid::AttentionFrame>();
        keys[layer] = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, kv_heads, 1);
        values[layer] = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, kv_heads, 1);
        ggml_set_input(keys[layer]); ggml_set_input(values[layer]);
        auto * query_tensor = q;
        if (layer) query_tensor = ggml_add(context.get(), q, ggml_scale(context.get(),
            ggml_reshape_3d(context.get(), outputs[layer-1], dimension, query_heads, 1), 0));
        auto * projected_key = ggml_scale(context.get(), keys[layer], 1);
        if (controlled && layer == 0) projected_key = ggml_custom_4d(context.get(), GGML_TYPE_F32, dimension, kv_heads, 1, 1,
            &keys[layer], 1, OverlapBarrier::compute, 1, &barrier);
        operations[layer] = std::make_unique<solid::AttentionNode>(session, frames[layer], layer, scale, nullptr, true, &device);
        outputs[layer] = operations[layer]->build(context.get(), query_tensor, projected_key, values[layer]);
        if (operations[layer]->selection_node()) {
            if (layer) {
                const bool covers = solid::graph_depends_on(query_tensor, operations[layer-1]->attention_node());
                require(covers && !solid::graph_depends_on(q, operations[layer-1]->attention_node()), "query completion proof");
                operations[layer]->set_query_proof(covers); operations[layer-1]->defer_completion();
            }
            ggml_build_forward_expand(graph, operations[layer]->selection_node());
            if (layer == 1) operations[layer]->defer_completion();
        }
        require(solid::to_half(-INFINITY) == 0xfc00, "half mask representation");
    }
    ggml_build_forward_expand(graph, outputs[1]);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(buffer != nullptr, "prefetch graph allocation");
    ggml_backend_t backends[] = {backend.get()};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, nullptr, 1, 1024, false, true), ggml_backend_sched_free);
    require(scheduler != nullptr, "prefetch graph scheduler");
    Observer observer{&device};
    ggml_backend_sched_set_eval_callback(scheduler.get(), Observer::evaluate, &observer);
    for (int p = 6; p < 12; ++p) {
        device.begin_prefetch(0, p); // Before current Q/K/V are available.
        std::vector<float> expected[2];
        for (int layer = 0; layer < 2; ++layer) {
            inputs(layer, p);
            expected[layer] = reference.process(layer, p, 1, query, key, value, scale);
            frames[layer]->positions = {static_cast<std::uint64_t>(p)};
            ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
            ggml_backend_tensor_set(keys[layer], key.data(), 0, key.size() * sizeof(float));
            ggml_backend_tensor_set(values[layer], value.data(), 0, value.size() * sizeof(float));
        }
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS, "prefetch graph execution");
        device.process_complete();
        for (int layer = 0; layer < 2; ++layer) {
            operations[layer]->check();
            std::vector<float> actual(expected[layer].size());
            ggml_backend_tensor_get(outputs[layer], actual.data(), 0, actual.size() * sizeof(float));
            for (std::size_t i = 0; i < actual.size(); ++i) require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[layer][i]) < 3e-4f,
                "prefetched arbitrary-slot attention differs from synchronous sparse reference");
        }
        ggml_backend_sched_reset(scheduler.get());
    }
    require(device.stats().uploads > 0 && session.stats().prefetch.predicted_blocks > 0, "prefetch path was not exercised");
    const auto stats = device.schedule_stats();
    require(stats.frames == 12 && stats.tasks == 144, "microtask frames not fully completed");
    if (mode == solid::ScheduleMode::sync_reuse) require(stats.query_reused_fences == 6 && stats.process_reused_fences == 6 && stats.attention_fences == 0,
        "sync reuse did not remove per-attention host fences");
    else require(observer.completions == 12, "scheduler failed to confirm each layer's attention completion");
    if (controlled) require(stats.miss_projection_overlap_ns > 0, "missing I/O and KV projection did not overlap");
    if (disk) require(session.stats().prefetch.io.completed_reads > 0, "GGML path did not complete liburing reads");
    device.quiesce(); session.finish();
    if (!directory.empty()) {
        std::ofstream output(directory / "stats.json");
        output << nlohmann::json({{"frames", stats.frames}, {"tasks", stats.tasks}, {"critical_path_ns", stats.critical_path_ns},
            {"miss_projection_overlap_ns", stats.miss_projection_overlap_ns}, {"query_reused_fences", stats.query_reused_fences},
            {"process_reused_fences", stats.process_reused_fences}, {"attention_fences", stats.attention_fences},
            {"fallback_fences", stats.fallback_fences}}).dump();
        require(output.good(), "graph trace stats write failed");
    }
}
}
int main(int argc, char ** argv) {
    try {
        const auto mode = solid::parse_schedule_mode(argc > 1 ? argv[1] : "naive");
        const std::filesystem::path traces = argc > 2 ? argv[2] : "";
        run(false, mode, traces);
        if (solid::uring_available()) run(true, mode, traces);
        std::cout << "multi-layer prefetch, liburing, GQA slot masks and CPU Flash Attention parity passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
