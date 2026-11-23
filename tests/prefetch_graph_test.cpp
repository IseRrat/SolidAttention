#include "prefetch_fakes.hpp"
#include "representative_builder_cases.hpp"
#include "tiled_prefill_cases.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/llama_adapter.hpp"
#include "solid/device_ops.hpp"
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
void early_half_frontier(solid::ScheduleMode mode,bool asynchronous) {
    const solid::KvGeometry geometry{1,1,8,2,8,64,solid::KvLayout::interleaved};
    const solid::SparseBudget budget{4,1,1,1};
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    auto file=std::make_unique<File>();
    auto reader=std::make_unique<Reader>(file.get());
    auto store=std::make_unique<solid::FileKvStore>(geometry,std::move(file),2048);
    auto * archive=store.get();
    solid::AttentionSession session(std::move(store),1,budget,true,2,{}, {},std::move(reader),4);
    session.set_ranker(std::make_unique<test_builder::Emulator>(geometry,1,budget,true));
    std::vector<float> query(8,0.25f),key(8,0.5f),value(8,0.75f);
    for (int p=0;p<2;++p) (void) session.process(0,p,1,query,key,value,1);
    session.flush();
    solid::GgmlPrefetch prefetch(session,backend.get(),{},mode);
    for (std::uint64_t p=2;p<4;++p) {
        prefetch.begin_prefetch(0,p);
        const solid::DeviceQuery q{query.data(),query.size(),solid::DeviceMemory::host_emulator};
        if (asynchronous) prefetch.select_early_device_async(0,p,q,true,{},false);
        else prefetch.select_early_device(0,p,q,true,{},false);
        std::vector<solid::Half> kv(16,solid::to_half(0.5f));
        std::fill(kv.begin()+8,kv.end(),solid::to_half(0.75f));
        const auto mask=prefetch.decode_half(0,p,kv,{kv.data(),kv.size(),16,solid::DeviceMemory::host_emulator});
        require(archive->length(0)==p+1,"early FP16 frontier did not append its KV");
        solid::KvBlock stored;
        archive->read(0,0,p/geometry.block_tokens,stored);
        require(std::equal(kv.begin(),kv.end(),stored.storage()+(p%geometry.block_tokens)*16),
            "early FP16 frontier archived different KV values");
        require(std::count(mask.begin(),mask.end(),solid::Half{0})==static_cast<std::ptrdiff_t>(p+1),
            "early FP16 frontier lost selected rows");
        // CPU uploads are complete; no attention graph is submitted in this
        // entry-point regression, so its completion boundary is already safe.
        prefetch.process_complete();
    }
    prefetch.quiesce();
    require(prefetch.schedule_stats().frames==2,"early FP16 frontier left unfinished microtasks");
    require(session.stats().decode_kv_download_bytes==64,"early FP16 frontier did not archive both input rows");
}
class Events final:public solid::AttentionEvents {
public:
    explicit Events(std::uint32_t layers):AttentionEvents(layers) {}
    bool copy_ready(ggml_backend_event *) const override { return true; }
    bool host_marker() const override { return true; }
    const char * name() const override { return "cpu_protocol"; }
    std::function<void()> first_receipt;
protected:
    std::uint64_t elapsed_ns(std::uint32_t) const override {
        if (!checked_) { checked_=true; if (first_receipt) first_receipt(); }
        return 7;
    }
private:
    mutable bool checked_=false;
};
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
void run(bool disk, solid::ScheduleMode mode, const std::filesystem::path & traces, bool builder,bool native,bool direct,bool packed,
         bool shared=false,bool infllm=false) {
    constexpr int dimension = 128, kv_heads = 2, query_heads = 4;
    const solid::KvGeometry geometry{2, kv_heads, dimension, 2, 12, 64, solid::KvLayout::interleaved};
    const solid::SparseBudget budget = infllm ? solid::SparseBudget{6, 1, 2, 2, true, true} : solid::SparseBudget{6, 1, 1, 1};
    const auto rope = infllm ? std::optional<solid::RelativeRope>{{dimension, 10000.f, true}} : std::nullopt;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<solid::KvStore> store;
    std::unique_ptr<solid::AsyncReader> reader;
    Reader * submitted_reads=nullptr;
    OverlapBarrier barrier;
    const bool controlled = !disk && mode != solid::ScheduleMode::naive && !shared && !infllm;
    if (disk) {
        const auto path = std::filesystem::temp_directory_path() / ("solid-prefetch-graph-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        store = std::make_unique<solid::FileKvStore>(geometry, solid::create_session_file(path, true), 2048);
        reader = solid::make_threaded_uring_reader(2);
    } else {
        auto file = std::make_unique<File>();
        auto fake = std::make_unique<Reader>(file.get());
        submitted_reads=fake.get();
        if (controlled) fake->before_wait = [&] { barrier.io_wait(); };
        store = std::make_unique<solid::FileKvStore>(geometry, std::move(file), 2048);
        reader = std::move(fake);
    }
    solid::AttentionSession session(std::move(store), query_heads, budget, true, 6, {}, {}, std::move(reader), 6, {}, rope);
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(geometry), query_heads, budget, true, 6, {}, {}, {}, 0, {}, rope);
    if (builder || direct) {
        auto rank_budget = budget;
        if (infllm) {
            rank_budget.local = 0;
            rank_budget.representatives = 1;
        }
        session.set_ranker(std::make_unique<test_builder::Emulator>(session.representative_geometry(), query_heads, rank_budget, direct));
    }
    const float scale = 1 / std::sqrt(static_cast<float>(dimension));
    std::vector<float> query(query_heads * dimension), key(kv_heads * dimension), value(key.size());
    const auto inputs = [&](int layer, int p) {
        for (int q = 0; q < query_heads; ++q) for (int d = 0; d < dimension; ++d) {
            query[q * dimension + d] = d == 0 ? (q < 2 ? 1.0f : -1.0f) * ((p % 3) ? 1 : -1) : 0;
        }
        if (rope) for (int q = 0; q < query_heads; ++q) rope->rotate(query.data() + q * dimension, budget.local - 1);
        for (int h = 0; h < kv_heads; ++h) for (int d = 0; d < dimension; ++d) {
            key[h * dimension + d] = d == 0 ? static_cast<float>(p + layer) : 0;
            value[h * dimension + d] = (p + layer + h + d) * 0.125f;
        }
    };
    if (direct && !infllm) {
        session.set_prefill(std::make_unique<test_tiled::Emulator>(geometry,query_heads,3,4,true));
        for (int layer=0;layer<2;++layer) {
            std::vector<float> q,k,v;
            for (int p=0;p<6;++p) {
                inputs(layer,p); q.insert(q.end(),query.begin(),query.end()); k.insert(k.end(),key.begin(),key.end()); v.insert(v.end(),value.begin(),value.end());
            }
            std::vector<solid::Half> kv(k.size()*2);
            for (int p=0;p<6;++p) for (int h=0;h<kv_heads;++h) for (int d=0;d<dimension;++d) {
                const auto source=(p*kv_heads+h)*dimension+d,target=(p*kv_heads+h)*2*dimension+d;
                kv[target]=solid::to_half(k[source]); kv[target+dimension]=solid::to_half(v[source]);
            }
            std::vector<float> output(q.size(),NAN);
            session.process_prefill_device(layer,0,6,{q.data(),q.size(),solid::DeviceMemory::host_emulator},
                {kv.data(),kv.size(),solid::DeviceMemory::host_emulator},kv,scale,{output.data(),output.size(),solid::DeviceMemory::host_emulator});
            test_tiled::close(output,reference.process(layer,0,6,q,k,v,scale));
            session.release_prefill_if_complete();
        }
    } else for (int p = 0; p < 6; ++p) for (int layer = 0; layer < 2; ++layer) {
        inputs(layer, p); (void) session.process(layer, p, 1, query, key, value, scale);
        (void) reference.process(layer, p, 1, query, key, value, scale);
    }
    session.flush();
    const auto directory = traces.empty() ? std::filesystem::path{} : traces / (disk ? "uring" : "memory");
    if (!directory.empty()) std::filesystem::create_directory(directory);
    auto events=native?std::make_unique<Events>(geometry.layers):nullptr;
    if (shared && events && submitted_reads) events->first_receipt=[&] {
        const auto next_layer=geometry.block_offset(1,0,0);
        require(std::any_of(submitted_reads->submitted.begin(),submitted_reads->submitted.end(),[&](const auto & read) {
            return read.offset>=next_layer;
        }),"shared next-layer READ waited for attention completion");
    };
    solid::GgmlPrefetch device(session, backend.get(), directory.empty() ? std::filesystem::path{} : directory / "residency.jsonl",
        mode, directory.empty() ? std::filesystem::path{} : directory / "schedule.jsonl",-1,
        std::move(events),shared);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    auto * q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimension, query_heads, 1);
    ggml_set_input(q);
    ggml_tensor * keys[2]; ggml_tensor * values[2]; ggml_tensor * packed_kv[2]{};
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
        require(!operations[layer]->can_reuse(6), "unbuilt attention graph must not be reused");
        if (packed) {
            packed_kv[layer] = ggml_concat(context.get(),projected_key,values[layer],0);
            projected_key = ggml_view_3d(context.get(),packed_kv[layer],dimension,kv_heads,1,
                dimension*2*sizeof(float),dimension*kv_heads*2*sizeof(float),0);
            values[layer] = ggml_view_3d(context.get(),packed_kv[layer],dimension,kv_heads,1,
                dimension*2*sizeof(float),dimension*kv_heads*2*sizeof(float),dimension*sizeof(float));
        }
        outputs[layer] = operations[layer]->build(context.get(), query_tensor, projected_key, values[layer],packed_kv[layer]);
        ggml_set_output(outputs[layer]);
        if (infllm) {
            require(solid::device_op(outputs[layer]) == solid::DeviceOp::attention_result,
                    "InfLLM output must remain on its attention backend");
            const auto * receipt = outputs[layer]->src[1];
            require(ggml_nelements(receipt) == 1 && ggml_nelements(receipt->src[1]) == 1 &&
                    ggml_nelements(receipt->src[0]) == budget.local * query_heads,
                    "InfLLM receipt copied full attention or historical probabilities");
        }
        if (direct) {
            auto * frontier=operations[layer]->host_node()->src[0];
            auto * kv=frontier;
            if (mode!=solid::ScheduleMode::naive) {
                require(solid::device_op(frontier)==solid::DeviceOp::kv_token && ggml_nbytes(frontier)==sizeof(float),
                    "overlap CPU frontier must receive only a device KV ready token");
                kv=frontier->src[0];
            }
            require(solid::device_op(kv)==solid::DeviceOp::consolidate_kv,"device KV consolidator missing");
            if (mode!=solid::ScheduleMode::naive) {
                require(solid::graph_depends_on(kv,operations[layer]->selection_node()),"device KV must depend on early selection");
                require(kv->src[2]==operations[layer]->selection_node()->src[0]->src[0],"Q lifetime dependency is missing");
            } else require(kv->src[2]==kv->src[3]->src[0] && solid::device_op(kv->src[3])==solid::DeviceOp::query_token,
                "serial KV lacks its GPU query frontier");
            if (!infllm) {
                auto * mask=operations[layer]->attention_node()->src[3]->src[0];
                require(solid::device_op(mask)==solid::DeviceOp::release_mask && mask->src[1]==kv &&
                    mask->src[2]==kv->src[2] && solid::graph_depends_on(mask,operations[layer]->host_node()),"Q/KV release must follow the mask frontier");
            }
        }
        if (operations[layer]->selection_node()) {
            if (layer && !infllm) {
                const bool covers = solid::graph_depends_on(query_tensor, operations[layer-1]->attention_node());
                require(covers && !solid::graph_depends_on(q, operations[layer-1]->attention_node()), "query completion proof");
                operations[layer]->set_query_proof(covers); operations[layer-1]->defer_completion();
            }
            ggml_build_forward_expand(graph, operations[layer]->selection_node());
            if (layer == 1) operations[layer]->defer_completion();
        }
        if (infllm) ggml_build_forward_expand(graph, operations[layer]->host_node());
        require(solid::to_half(-INFINITY) == 0xfc00, "half mask representation");
    }
    ggml_build_forward_expand(graph, outputs[1]);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        direct ? nullptr : ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(direct || buffer != nullptr, "prefetch graph allocation");
    ggml_backend_t backends[] = {backend.get()};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, nullptr, 1, 1024, false, true), ggml_backend_sched_free);
    require(scheduler != nullptr, "prefetch graph scheduler");
    Observer observer{&device};
    ggml_backend_sched_set_eval_callback(scheduler.get(), Observer::evaluate, &observer);
    for (int p = 6; p < 12; ++p) {
        for (const auto & operation : operations) {
            require(operation->can_reuse(p) == (p > 6), "decode graph reuse requires an initialized previous position");
        }
        device.begin_prefetch(0, p); // Before current Q/K/V are available.
        if (direct) require(ggml_backend_sched_alloc_graph(scheduler.get(),graph),"reusing direct graph allocation");
        std::vector<float> expected[2];
        for (int layer = 0; layer < 2; ++layer) {
            inputs(layer, p);
            expected[layer] = reference.process(layer, p, 1, query, key, value, scale);
            frames[layer]->positions = {static_cast<std::uint64_t>(p)};
            require(!operations[layer]->can_reuse(p) && !operations[layer]->can_reuse(p + 2) &&
                    operations[layer]->can_reuse(p + 1) == (p + 1 < geometry.capacity_tokens),
                    "decode reuse accepted a repeated, skipped or out-of-capacity position");
            ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
            ggml_backend_tensor_set(keys[layer], key.data(), 0, key.size() * sizeof(float));
            // Packed views are produced from the separate source leaf.
            auto * value_input=packed?packed_kv[layer]->src[1]:values[layer];
            ggml_backend_tensor_set(value_input, value.data(), 0, value.size() * sizeof(float));
        }
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS, "prefetch graph execution");
        device.process_complete();
        for (int layer = 0; layer < 2; ++layer) {
            operations[layer]->check();
            std::vector<float> actual(expected[layer].size());
            ggml_backend_tensor_get(outputs[layer], actual.data(), 0, actual.size() * sizeof(float));
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[layer][i]) >= 3e-4f) {
                    throw std::runtime_error("prefetched attention differs at token " + std::to_string(p) + " layer " +
                        std::to_string(layer) + " element " + std::to_string(i) + " (" + std::to_string(actual[i]) +
                        " vs " + std::to_string(expected[layer][i]) + ")");
                }
            }
        }
        ggml_backend_sched_reset(scheduler.get());
    }
    device.quiesce(); session.finish();
    require(device.stats().uploads > 0 && session.stats().prefetch.predicted_blocks > 0, "prefetch path was not exercised");
    if (shared) {
        require(session.stats().prefetch.host_bytes==0 && device.stats().upload_bytes==0,
                "shared cache retained host payloads or uploaded historical KV");
        if (!direct) require(device.stats().shared_host_copy_bytes==0,"shared historical bank was copied");
        require(device.stats().shared_resident_bytes==device.stats().resident_bytes,"shared buffer allocation accounting");
    }
    const auto stats = device.schedule_stats();
    require(stats.frames == 12 && stats.tasks == 144, "microtask frames not fully completed");
    require(mode==solid::ScheduleMode::naive ? stats.block_frames==0 : stats.block_frames==12 && stats.block_tasks>12,
            "historical block DAG not completed through actual copy receipts");
    if (infllm && !native) require(stats.attention_fences == 12 && observer.completions == 0, "InfLLM probability receipts did not complete attention");
    else if (native) require(stats.device_event_completions==12 && stats.device_attention_ns==84 && stats.device_boundary_fallbacks==0 &&
        stats.query_reused_fences==0 && stats.process_reused_fences==0 && observer.completions==0,"native completion did not replace graph boundary fences");
    else if (mode == solid::ScheduleMode::sync_reuse) require(stats.query_reused_fences == 6 && stats.process_reused_fences == 6 && stats.attention_fences == 0,
        "sync reuse did not remove per-attention host fences");
    else require(observer.completions == 12, "scheduler failed to confirm each layer's attention completion");
    if (controlled) require(stats.miss_projection_overlap_ns > 0, "missing I/O and KV projection did not overlap");
    if (disk) require(session.stats().prefetch.io.completed_reads > 0, "GGML path did not complete liburing reads");
    if (disk && mode != solid::ScheduleMode::naive) {
        require(session.stats().prefetch.io.write_bytes == 12ull*kv_heads*dimension*2*sizeof(solid::Half),
                "decode did not commit its full write buffers through liburing");
    }
    if ((builder || direct) && !infllm) require(session.stats().ranking.builder_queries == 24 && session.stats().ranking.builder_appends == 24 &&
        session.stats().ranking.built_blocks == 10, "graph did not execute representative builder across prefill and decode");
    if (direct && !infllm) require(session.stats().direct_prefill_queries==12 && session.stats().ranking.device_builder_batches==4 &&
        session.stats().direct_decode_queries==12 && session.stats().ranking.device_builder_queries==24 &&
        session.stats().ranking.device_builder_appends==24 && session.stats().ranking.device_rank_queries>0 &&
        session.stats().decode_kv_download_bytes==12ull*kv_heads*dimension*2*sizeof(solid::Half),"device decode counters or FP16 transfer size");
    if (direct && infllm) require(session.stats().direct_decode_queries==12 && session.stats().ranking.device_rank_queries>0 &&
        session.stats().ranking.device_builder_queries==0 &&
        session.stats().decode_kv_download_bytes==12ull*kv_heads*dimension*2*sizeof(solid::Half),"InfLLM direct ranking or KV archive missing");
    if (direct) require(device.stats().current_device_rows==12ull*kv_heads &&
        device.stats().current_device_bytes==12ull*kv_heads*dimension*2*sizeof(solid::Half),"current KV did not use device row transfers");
    if (!directory.empty()) {
        std::ofstream output(directory / "stats.json");
        output << nlohmann::json({{"frames", stats.frames}, {"tasks", stats.tasks}, {"critical_path_ns", stats.critical_path_ns},
            {"miss_projection_overlap_ns", stats.miss_projection_overlap_ns}, {"query_reused_fences", stats.query_reused_fences},
            {"process_reused_fences", stats.process_reused_fences}, {"attention_fences", stats.attention_fences},
            {"attention_receipt_reuses",stats.attention_receipt_reuses},
            {"fallback_fences", stats.fallback_fences},{"device_event_completions",stats.device_event_completions},
            {"device_attention_ns",stats.device_attention_ns}}).dump();
        require(output.good(), "graph trace stats write failed");
        std::ofstream blocks(directory/"block-stats.json");
        blocks<<nlohmann::json({{"block_frames",stats.block_frames},{"block_tasks",stats.block_tasks},
            {"block_critical_path_ns",stats.block_critical_path_ns}}).dump();
        require(blocks.good(),"block trace stats write failed");
        const auto copies=device.stats();
        std::ofstream transfers(directory/"device-stats.json");
        transfers<<nlohmann::json({{"current_device_rows",copies.current_device_rows},{"current_device_bytes",copies.current_device_bytes},
            {"current_host_bytes",copies.current_host_bytes},{"refreshed_device_tails",copies.refreshed_tails},
            {"retained_device_prefix_bytes",copies.retained_prefix_bytes}}).dump();
        require(transfers.good(),"device copy stats write failed");
    }
}
}
int main(int argc, char ** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="infllm") {
            for (auto mode : {solid::ScheduleMode::naive, solid::ScheduleMode::overlap, solid::ScheduleMode::sync_reuse}) {
                run(false, mode, {}, false, false, false, true, false, true);
                run(false, mode, {}, false, mode==solid::ScheduleMode::sync_reuse, true, true, false, true);
            }
            std::cout << "InfLLM attention through prefetched slots and device frontiers passed\n";
            return 0;
        }
        if (argc==2 && std::string(argv[1])=="early-half") {
            for (auto mode:{solid::ScheduleMode::overlap,solid::ScheduleMode::sync_reuse})
                for (bool asynchronous:{false,true}) early_half_frontier(mode,asynchronous);
            std::cout<<"early FP16 frontier, append and completion passed\n"; return 0;
        }
        if (argc==2 && std::string(argv[1])=="shared") {
            for (auto mode:{solid::ScheduleMode::naive,solid::ScheduleMode::overlap,solid::ScheduleMode::sync_reuse}) {
                for (bool direct:{false,true}) {
                    run(false,mode,{},false,false,direct,direct,true);
                    if (solid::uring_available()) run(true,mode,{},false,false,direct,direct,true);
                }
            }
            std::cout<<"shared SSD destinations, attention leases and private current rows passed\n"; return 0;
        }
        const auto mode = solid::parse_schedule_mode(argc > 1 ? argv[1] : "naive");
        bool builder=false,native=false,direct=false,packed=false,shared=false;
        int index=2;
        for (;index<argc && std::string(argv[index]).rfind("--",0)==0;++index) {
            const std::string option=argv[index];
            if (option=="--builder") builder=true;
            else if (option=="--native-events") native=true;
            else if (option=="--direct") direct=true;
            else if (option=="--packed") packed=true;
            else if (option=="--shared") shared=true;
            else throw std::invalid_argument("unknown graph check option");
        }
        const std::filesystem::path traces = argc > index ? argv[index] : "";
        run(false, mode, traces, builder,native,direct,packed,shared);
        if (solid::uring_available()) run(true, mode, traces, builder,native,direct,packed,shared);
        std::cout << "multi-layer prefetch, liburing, GQA slot masks and CPU Flash Attention parity passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
