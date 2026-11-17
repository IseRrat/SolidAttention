#include "solid/ggml_prefetch.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <limits>
#include <map>
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace solid {
namespace {
class GgmlTransport final : public DeviceTransport {
public:
    GgmlTransport(ggml_backend_t backend, KvGeometry geometry, std::uint32_t per_head, ScheduleMode mode) :
        backend_(backend), geometry_(geometry), per_head_(per_head) {
        if (!backend || geometry.layout != KvLayout::interleaved) throw std::invalid_argument("invalid GGML prefetch transport");
        const auto tokens = (static_cast<std::uint64_t>(per_head) * geometry.block_tokens + 31) / 32 * 32;
        if (tokens > INT32_MAX || tokens * geometry.heads > static_cast<std::uint64_t>(INT64_MAX) / (2ull * geometry.dimension * sizeof(Half))) {
            throw std::overflow_error("GGML prefetch transport allocation overflow");
        }
        context_.reset(ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("device cache metadata allocation");
        tensor_ = ggml_new_tensor_3d(context_.get(), GGML_TYPE_F16, geometry.dimension * 2, tokens, geometry.heads);
        ggml_set_name(tensor_, "solid_prefetch_device_slots");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend));
        if (!buffer_) throw std::runtime_error("device cache allocation");
        ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer_.get(), 0);
        copy_backend_ = backend_;
        if (mode != ScheduleMode::naive) {
            auto * device = ggml_backend_get_device(backend_);
            if (!device) throw std::runtime_error("copy backend requires a registered device");
            copy_owner_.reset(ggml_backend_dev_init(device, nullptr));
            if (!copy_owner_) throw std::runtime_error("independent copy backend initialization failed");
            copy_backend_ = copy_owner_.get();
            auto * host_type = ggml_backend_dev_host_buffer_type(device);
            if (!host_type) host_type = ggml_backend_get_default_buffer_type(copy_backend_);
            if (!ggml_backend_buft_is_host(host_type)) throw std::runtime_error("copy staging requires host memory");
            const auto bytes = static_cast<std::size_t>(geometry.heads) * per_head * geometry.block_tokens * 2 * geometry.dimension * sizeof(Half);
            staging_.reset(ggml_backend_buft_alloc_buffer(host_type, bytes));
            if (!staging_) throw std::runtime_error("pinned copy staging allocation failed");
            stats_.staging_bytes = ggml_backend_buffer_get_size(staging_.get());
        }
    }
    ~GgmlTransport() override {
        if (!events_.empty()) {
            try { drain(); } catch (...) { std::terminate(); }
            for (const auto & event : events_) if (event.second) ggml_backend_event_free(event.second);
        }
    }
    std::uint64_t upload(std::uint32_t slot, const KvBlock & block) override {
        if (slot >= geometry_.heads * static_cast<std::uint64_t>(per_head_) || block.layout != KvLayout::interleaved ||
            block.dimension != geometry_.dimension || block.capacity != geometry_.block_tokens ||
            block.data.size() != static_cast<std::size_t>(geometry_.block_tokens) * 2 * geometry_.dimension) {
            throw std::invalid_argument("device upload shape");
        }
        if (next_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("copy ticket exhausted");
        const auto id = next_++;
        auto * device = ggml_backend_get_device(copy_backend_);
        auto * event = device ? ggml_backend_event_new(device) : nullptr;
        try {
            events_.emplace(id, event);
            const auto head = slot / per_head_, local = slot % per_head_;
            const auto offset = head * tensor_->nb[2] + local * geometry_.block_tokens * tensor_->nb[1];
            const auto bytes = block.data.size() * sizeof(Half);
            const void * source = block.data.data();
            if (staging_) {
                auto * destination = static_cast<unsigned char *>(ggml_backend_buffer_get_base(staging_.get())) + static_cast<std::size_t>(slot) * bytes;
                std::memcpy(destination, source, bytes); source = destination;
            }
            ggml_backend_tensor_set_async(copy_backend_, tensor_, source, offset, bytes);
            if (event) ggml_backend_event_record(event, copy_backend_);
            else ggml_backend_synchronize(copy_backend_);
        } catch (...) {
            ggml_backend_synchronize(copy_backend_);
            events_.erase(id);
            if (event) ggml_backend_event_free(event);
            throw;
        }
        return id;
    }
    bool complete(std::uint64_t ticket, bool wait) override {
        const auto found = events_.find(ticket);
        if (found == events_.end()) throw std::logic_error("unknown upload ticket");
        if (found->second && ticket > completed_through_ && !wait) return false;
        if (found->second) {
            if (ticket > completed_through_) { ggml_backend_event_synchronize(found->second); ++stats_.event_waits; }
            ggml_backend_event_free(found->second);
        }
        events_.erase(found);
        return true;
    }
    void drain() override { ggml_backend_synchronize(copy_backend_); ++stats_.backend_drains; completed_through_ = next_ - 1; }
    void wait_batch() override {
        if (events_.empty()) return;
        ++stats_.batch_groups; stats_.events_coalesced += events_.size();
        const auto latest = events_.rbegin();
        if (latest->first > completed_through_ && latest->second) {
            ggml_backend_event_synchronize(latest->second); ++stats_.batch_waits;
        }
        completed_through_ = next_ - 1;
    }
    void wait_on_compute() {
        if (!events_.empty() && events_.rbegin()->second) {
            ggml_backend_event_wait(backend_, events_.rbegin()->second); ++stats_.stream_waits;
        }
    }
    void synchronize_compute() { ggml_backend_synchronize(backend_); ++stats_.backend_compute_waits; }
    ScheduleStats stats() const { return stats_; }
    std::uint64_t resident_bytes() const override { return ggml_backend_buffer_get_size(buffer_.get()); }
    ggml_tensor * tensor() const { return tensor_; }
private:
    ggml_backend_t backend_;
    KvGeometry geometry_;
    std::uint32_t per_head_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> copy_owner_{nullptr, ggml_backend_free};
    ggml_backend_t copy_backend_ = nullptr;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> staging_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor * tensor_ = nullptr;
    std::map<std::uint64_t, ggml_backend_event_t> events_;
    std::uint64_t next_ = 1;
    std::uint64_t completed_through_ = 0;
    ScheduleStats stats_;
};
}
namespace {
enum Task : std::uint32_t { q_ready, q_d2h, select_task, kv_projection, kv_d2h, miss_read, history_h2d,
                           append, current_h2d, attention, store_task, next_prefetch, task_count };
struct LayerWork {
    std::uint32_t layer;
    std::uint64_t position;
    std::vector<float> query;
    std::vector<HeadSelection> selection;
    TaskDag dag;
    std::shared_ptr<BlockStage> blocks;
    bool consumed = false;
    LayerWork(std::uint32_t layer, std::uint64_t position, ScheduleMode mode, ScheduleTrace & trace,
               std::uint64_t id, const std::array<std::uint64_t, task_count> & estimates) :
        layer(layer), position(position), dag([&] {
            const bool serial = mode == ScheduleMode::naive;
            std::vector<TaskSpec> specs = {
                {"q_gpu_ready", TaskResource::gpu, {}, 1},
                {"q_d2h_ready", TaskResource::d2h, {q_ready}, 1},
                {serial ? "selection_ready" : "selection", TaskResource::host, serial ? std::vector<std::uint32_t>{append,q_d2h} : std::vector<std::uint32_t>{q_d2h}, 1},
                {"kv_projection_to_host", TaskResource::gpu, serial ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{select_task}, 1},
                {"kv_d2h_ready", TaskResource::d2h, {kv_projection}, 1},
                {"miss_read_and_reconcile", TaskResource::ssd, {select_task}, 1},
                {"history_h2d", TaskResource::h2d, serial ? std::vector<std::uint32_t>{miss_read} : std::vector<std::uint32_t>{select_task}, 1},
                {serial ? "append_and_selection" : "append_current_kv", TaskResource::host,
                    serial ? std::vector<std::uint32_t>{kv_d2h} : std::vector<std::uint32_t>{kv_d2h,miss_read,history_h2d}, 1},
                {"current_h2d", TaskResource::h2d, serial ? std::vector<std::uint32_t>{append,history_h2d} : std::vector<std::uint32_t>{append}, 1},
                {"attention_until_boundary", TaskResource::gpu, {current_h2d}, 1},
                {"buffered_ssd_store", TaskResource::ssd, {append}, 1},
                {"next_layer_prefetch", TaskResource::ssd, {current_h2d}, 1},
            };
            for (std::size_t i = 0; i < specs.size(); ++i) specs[i].estimate_ns = estimates[i];
            return specs;
        }(), &trace, id, layer, position, mode) {}
};
}
struct GgmlPrefetch::Impl {
    AttentionSession & session;
    ScheduleMode mode;
    TaskWorker worker;
    mutable std::mutex mutex;
    ScheduleTrace trace;
    ScheduleTrace block_trace;
    GgmlTransport transport;
    DeviceCache device;
    ScheduleStats totals;
    std::array<std::uint64_t, task_count> estimates;
    std::uint64_t block_read_estimate=1,block_copy_estimate=1;
    std::shared_ptr<LayerWork> work;
    std::uint64_t sequence = 0;
    bool quiesced = true;
    Impl(AttentionSession & s, ggml_backend_t backend, const std::filesystem::path & residency, ScheduleMode mode,
          const std::filesystem::path & schedule,int cpu) :
        session(s), mode(mode), worker(cpu), trace(schedule), block_trace(schedule.empty()?std::filesystem::path{}:
            schedule.parent_path()/(schedule.stem().string()+".blocks"+schedule.extension().string())),
        transport(backend, s.geometry(), s.prefetcher().slots_per_head(), mode),
        device(s.prefetcher(), s.geometry(), s.query_heads(), transport, residency) { estimates.fill(1); totals.coordinator_cpu=cpu; }
    std::shared_ptr<LayerWork> new_work(std::uint32_t layer, std::uint64_t position) {
        if (work && !work->dag.finished()) throw std::logic_error("new layer started before previous completion proof");
        if (sequence == UINT64_MAX) throw std::overflow_error("schedule frame identity exhausted");
        work = std::make_shared<LayerWork>(layer, position, mode, trace, ++sequence, estimates);
        work->dag.fence({q_ready,q_d2h}, trace.now(), "query_download_complete");
        return work;
    }
    void finish_frame(const std::string & proof) {
        if (!work || work->dag.task(attention).state != TaskState::running) throw std::logic_error("attention lacks a scheduled start");
        work->dag.complete(attention, trace.now(), proof);
        if (!work->dag.finished()) throw std::logic_error("layer completed with unfinished microtasks");
        ++totals.frames; totals.tasks += work->dag.size();
        totals.critical_path_ns += work->dag.critical_path_ns();
        totals.miss_projection_overlap_ns += work->dag.overlap_ns(miss_read, kv_projection);
        if (work->blocks) {
            const auto stats=work->blocks->stats();
            ++totals.block_frames; totals.block_tasks+=stats.tasks; totals.block_critical_path_ns+=stats.critical_path_ns;
            const auto update=[](std::uint64_t & estimate,std::uint64_t total,std::uint64_t count) {
                if (!count) return;
                const auto measured=std::max<std::uint64_t>(1,total/count);
                estimate=measured>=estimate?estimate+(measured-estimate)/8:estimate-(estimate-measured)/8;
            };
            update(block_read_estimate,stats.read_ns,stats.read_samples);
            update(block_copy_estimate,stats.copy_ns,stats.copy_samples);
        }
        for (std::uint32_t id = 0; id < task_count; ++id) {
            const auto task = work->dag.task(id);
            const auto duration = task.end_ns - task.start_ns;
            if (duration) estimates[id] = duration >= estimates[id] ? estimates[id] + (duration - estimates[id]) / 8 : estimates[id] - (estimates[id] - duration) / 8;
        }
    }
    void next_layer(const std::shared_ptr<LayerWork> & current) {
        const auto next = current->layer + 1 < session.geometry().layers ? current->layer + 1 : 0;
        const auto position = next ? current->position : current->position + 1;
        current->dag.start(next_prefetch, trace.now());
        session.begin_prefetch(next, position);
        current->dag.complete(next_prefetch, trace.now());
    }
    void release_attention(const char * proof) {
        if (!device.attention_pending()) return;
        if (device.attention_complete(device.attention_epoch())) {
            finish_frame(proof);
            session.prefetcher().poll(); device.prefetch_ready();
        }
    }
};
GgmlPrefetch::GgmlPrefetch(AttentionSession & session, ggml_backend_t backend, const std::filesystem::path & trace,
                          ScheduleMode mode, const std::filesystem::path & schedule_trace,int cpu) :
    impl_(std::make_unique<Impl>(session, backend, trace, mode, schedule_trace,cpu)) {}
GgmlPrefetch::~GgmlPrefetch() { if (!impl_->quiesced) { try { quiesce(); } catch (...) {} } }
ggml_tensor * GgmlPrefetch::tensor() const { return impl_->transport.tensor(); }
std::uint32_t GgmlPrefetch::tokens_per_head() const { return impl_->device.tokens_per_head(); }
std::uint64_t GgmlPrefetch::attention_epoch() const { return impl_->device.attention_epoch(); }
bool GgmlPrefetch::early_enabled() const { return impl_->mode != ScheduleMode::naive; }
bool GgmlPrefetch::reuses_sync() const { return impl_->mode == ScheduleMode::sync_reuse; }
void GgmlPrefetch::begin_prefetch(std::uint32_t layer, std::uint64_t position) {
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->session.begin_prefetch(layer, position);
}
void GgmlPrefetch::select_early(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, bool covers) {
    if (!early_enabled()) throw std::logic_error("early selection is disabled");
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->quiesced = false;
    if (impl_->device.attention_pending()) {
        if (reuses_sync() && covers) { ++impl_->totals.query_reused_fences; impl_->release_attention("query_dependency_reuse"); }
        else { impl_->transport.synchronize_compute(); ++impl_->totals.fallback_fences; impl_->release_attention("explicit_fallback"); }
    }
    impl_->session.begin_prefetch(layer, position);
    const auto work = impl_->new_work(layer, position);
    work->query = query;
    work->dag.start(select_task, impl_->trace.now());
    work->selection = impl_->session.select_before_kv(layer, position, query);
    work->dag.complete(select_task, impl_->trace.now());
    work->dag.start(kv_projection, impl_->trace.now());
    ++impl_->totals.worker_jobs;
    impl_->worker.submit([this, work] {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        work->dag.start(miss_read, impl_->trace.now());
        impl_->device.wait_transfers();
        auto & host = impl_->session.prefetcher();
        bool copying=false;
        const auto leases = host.resolve_history(work->layer, work->position, work->selection,
            [&](const std::vector<BlockIdentity> & plan) { impl_->device.begin_stage(plan); },
            [&](const SlotLease & lease) {
                if (!copying) { work->dag.start(history_h2d,impl_->trace.now()); copying=true; }
                impl_->device.stage_ready(lease,work->blocks);
            },[&](const std::vector<BlockWork> & plan) {
                auto blocks=plan;
                const auto ready=host.ready();
                for (auto & block:blocks) {
                    block.read_ns=std::any_of(ready.begin(),ready.end(),[&](const SlotLease & lease) { return lease.identity==block.identity; })?1:impl_->block_read_estimate;
                    block.copy_ns=impl_->device.find(block.identity)?1:impl_->block_copy_estimate;
                }
                work->blocks=std::make_shared<BlockStage>(std::move(blocks),&impl_->block_trace,impl_->sequence,work->layer,work->position,impl_->mode);
                return work->blocks;
            });
        work->dag.complete(miss_read, impl_->trace.now());
        if (!copying) work->dag.start(history_h2d, impl_->trace.now());
        for (const auto & lease : leases) host.unpin(lease);
    });
}
std::vector<Half> GgmlPrefetch::decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                      const std::vector<float> & keys, const std::vector<float> & values, bool packed) {
    const auto current = impl_->work;
    if (early_enabled()) {
        if (!current || current->layer != layer || current->position != position || current->consumed) throw std::logic_error("KV arrived without its early query");
        current->dag.fence({kv_projection,kv_d2h}, impl_->trace.now(), "kv_download_complete");
    }
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->quiesced = false;
    auto work = current;
    if (!early_enabled()) {
        impl_->device.wait_transfers(); impl_->session.begin_prefetch(layer, position);
        work = impl_->new_work(layer, position);
        work->dag.fence({kv_projection,kv_d2h}, impl_->trace.now(), "serial_kv_download_complete");
        work->dag.start(append, impl_->trace.now());
        work->selection = impl_->session.select_decode(layer, position, query, keys, values, packed);
        work->dag.complete(append, impl_->trace.now());
        work->dag.fence({select_task}, impl_->trace.now(), "serial_selection_complete");
        work->dag.start(miss_read, impl_->trace.now());
    } else {
        if (reuses_sync()) impl_->device.wait_transfers_coalesced(); else impl_->device.wait_transfers();
        work->dag.complete(history_h2d, impl_->trace.now(), "historical_copy_complete");
        work->dag.start(append, impl_->trace.now());
        impl_->session.append_after_selection(layer, position, work->query, keys, values, packed);
        work->dag.complete(append, impl_->trace.now());
    }
    auto & host = impl_->session.prefetcher();
    const auto leases = host.resolve(layer, position, work->selection);
    if (!early_enabled()) {
        work->dag.complete(miss_read, impl_->trace.now());
        work->dag.fence({history_h2d}, impl_->trace.now(), "serial_no_early_upload");
    }
    work->dag.start(current_h2d, impl_->trace.now());
    std::vector<Half> mask;
    try { mask = impl_->device.prepare(leases, work->selection, reuses_sync()); }
    catch (...) { for (const auto & lease : leases) host.unpin(lease); throw; }
    for (const auto & lease : leases) host.unpin(lease);
    work->consumed = true;
    if (!early_enabled()) {
        work->dag.complete(current_h2d, impl_->trace.now(), "serial_upload_complete");
        work->dag.start(attention, impl_->trace.now());
        work->dag.fence({store_task}, impl_->trace.now(), "serial_buffered_store");
        impl_->next_layer(work);
    } else {
        if (reuses_sync()) impl_->transport.wait_on_compute();
        else {
            work->dag.complete(current_h2d, impl_->trace.now(), "upload_complete");
            work->dag.start(attention, impl_->trace.now());
        }
        ++impl_->totals.worker_jobs;
        impl_->worker.submit([this, work] {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (reuses_sync()) {
                impl_->device.wait_transfers_coalesced();
                work->dag.complete(current_h2d, impl_->trace.now(), "shared_upload_batch");
                work->dag.start(attention, impl_->trace.now());
            }
            // At the shared frontier both I/O tasks are ready. Choose by the
            // DAG's critical-path/LST priority using the previous frame costs.
            for (int remaining = 2; remaining; --remaining) {
                const auto io = work->dag.ready({store_task,next_prefetch});
                if (io.empty()) throw std::logic_error("post-attention I/O dependencies missing");
                if (io.front() == next_prefetch) impl_->next_layer(work);
                else {
                    work->dag.start(store_task, impl_->trace.now());
                    impl_->session.flush_full_layer(work->layer);
                    work->dag.complete(store_task, impl_->trace.now());
                }
            }
        });
    }
    return mask;
}
void GgmlPrefetch::completed(std::uint64_t epoch) {
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->device.attention_pending() && epoch == impl_->device.attention_epoch()) {
        ++impl_->totals.attention_fences; impl_->release_attention("attention_backend_complete");
    }
}
void GgmlPrefetch::process_complete() {
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->device.attention_pending()) { ++impl_->totals.process_reused_fences; impl_->release_attention("process_synchronize_reuse"); }
}
void GgmlPrefetch::quiesce() {
    if (impl_->quiesced) return;
    std::exception_ptr error;
    try { impl_->worker.join(); } catch (...) { error = std::current_exception(); }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->transport.synchronize_compute(); impl_->transport.drain();
    try { impl_->device.wait_transfers(); } catch (...) { error = std::current_exception(); }
    try {
        if (impl_->device.attention_pending()) (void) impl_->device.attention_complete(impl_->device.attention_epoch());
    } catch (...) { if (!error) error = std::current_exception(); }
    try { impl_->device.finish(); } catch (...) { if (!error) error = std::current_exception(); }
    try {
        if (impl_->work && !impl_->work->dag.finished()) impl_->work->dag.abort(impl_->trace.now());
    } catch (...) { if (!error) error=std::current_exception(); }
    try {
        if (impl_->work && impl_->work->blocks && !impl_->work->blocks->finished()) impl_->work->blocks->abort();
    } catch (...) { if (!error) error=std::current_exception(); }
    try { impl_->trace.flush(); } catch (...) { if (!error) error = std::current_exception(); }
    try { impl_->block_trace.flush(); } catch (...) { if (!error) error = std::current_exception(); }
    impl_->quiesced = true;
    if (error) std::rethrow_exception(error);
}
DeviceCacheStats GgmlPrefetch::stats() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->device.stats(); }
ScheduleStats GgmlPrefetch::schedule_stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto result = impl_->totals;
    const auto transport = impl_->transport.stats();
    result.event_waits = transport.event_waits; result.batch_waits = transport.batch_waits;
    result.stream_waits = transport.stream_waits; result.staging_bytes = transport.staging_bytes;
    result.backend_drains = transport.backend_drains; result.backend_compute_waits = transport.backend_compute_waits;
    result.batch_groups = transport.batch_groups; result.events_coalesced = transport.events_coalesced;
    return result;
}
} // namespace solid
