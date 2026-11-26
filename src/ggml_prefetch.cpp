#include "solid/ggml_prefetch.hpp"
#include "solid/device_timing.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef SOLID_HAVE_CUDA_SELECTION
#include "ggml-cuda.h"
#endif
#ifdef SOLID_HAVE_SYCL
#include "ggml-sycl.h"
#include "solid/sycl_backend.hpp"
#endif
#include <limits>
#include <chrono>
#include <map>
#include <array>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef __linux__
#include <poll.h>
#include <cerrno>
#include <system_error>
#endif

namespace solid {
namespace {
std::unique_ptr<AttentionEvents> attention_events(AttentionSession & session,ggml_backend_t backend,ScheduleMode mode,
                                                  std::unique_ptr<AttentionEvents> supplied) {
    if (supplied) {
        if (mode!=ScheduleMode::sync_reuse) throw std::invalid_argument("device completion requires sync_reuse scheduling");
        return supplied;
    }
#ifdef SOLID_HAVE_CUDA_SELECTION
    if (mode==ScheduleMode::sync_reuse && ggml_backend_is_cuda(backend)) {
        auto * registry=ggml_backend_cuda_reg();
        for (std::size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(registry);++ordinal)
            if (ggml_backend_reg_dev_get(registry,ordinal)==ggml_backend_get_device(backend))
                return make_cuda_attention_events(session.geometry().layers,static_cast<int>(ordinal));
        throw std::runtime_error("CUDA attention completion device identity mismatch");
    }
#endif
#ifdef SOLID_HAVE_SYCL
    if (mode==ScheduleMode::sync_reuse && ggml_backend_is_sycl(backend)) {
        return make_sycl_attention_events(session.geometry().layers,solid_sycl_queue(backend));
    }
#endif
    (void) session; (void) backend;
    return {};
}
class GgmlTransport final : public DeviceTransport {
    struct RowCopy { ggml_tensor source{},target{}; };
    struct CopyRecord {
        ggml_backend_event_t event=nullptr;
        std::unique_ptr<std::array<RowCopy,2>> row;
        DeviceInterval * interval=nullptr;
#ifdef SOLID_HAVE_SYCL
        std::unique_ptr<SyclCopyEvent> sycl;
#endif
    };
public:
    GgmlTransport(ggml_backend_t backend, KvGeometry geometry, std::uint32_t per_head, ScheduleMode mode,AttentionEvents * native,bool share_host_cache,bool coalesce_uploads) :
        backend_(backend), geometry_(geometry), per_head_(per_head),native_(native) {
        if (!backend) throw std::invalid_argument("invalid GGML prefetch transport");
        stats_.coalesce_uploads=coalesce_uploads;
        if (share_host_cache) {
            if (!ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(backend))) throw std::invalid_argument("explicit shared cache requires a host-addressable backend");
            shared_=true;
        }
#ifdef SOLID_HAVE_SYCL
        if (ggml_backend_is_sycl(backend) && solid_sycl_shared_kv_supported(backend)) shared_=true;
#endif
        if (shared_ && geometry.layout != KvLayout::interleaved) throw std::invalid_argument("shared KV slots require interleaved storage");
        if (shared_) {
            if (per_head_==UINT32_MAX) throw std::overflow_error("shared cache slot count");
            ++per_head_; // One private current-token block per head.
        }
        const auto tokens = (static_cast<std::uint64_t>(per_head_) * geometry.block_tokens + 31) / 32 * 32;
        if (tokens > INT32_MAX || tokens * geometry.heads > static_cast<std::uint64_t>(INT64_MAX) / (2ull * geometry.dimension * sizeof(Half))) {
            throw std::overflow_error("GGML prefetch transport allocation overflow");
        }
        context_.reset(ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("device cache metadata allocation");
        tensor_ = geometry.layout == KvLayout::interleaved ?
            ggml_new_tensor_3d(context_.get(), GGML_TYPE_F16, geometry.dimension * 2, tokens, geometry.heads) :
            ggml_new_tensor_4d(context_.get(), GGML_TYPE_F16, geometry.dimension, tokens, geometry.heads, 2);
        ggml_set_name(tensor_, "solid_prefetch_device_slots");
#ifdef SOLID_HAVE_SYCL
        if (ggml_backend_is_sycl(backend)) {
            sycl_copy_=make_sycl_copy_queue(backend);
            const auto bytes=ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), tensor_);
            if (shared_) buffer_.reset(solid_sycl_alloc_shared_kv(backend, bytes),ggml_backend_buffer_free);
            if (shared_ && !buffer_) throw std::runtime_error("shared KV allocation failed");
            if (buffer_) {
                auto * memory=ggml_backend_buffer_get_base(buffer_.get());
                ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (ggml_backend_tensor_alloc(buffer_.get(),tensor_,memory)!=GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("SYCL shared KV tensor initialization failed");
                }
                std::memset(memory,0,bytes);
                shared_=true;
                copy_backend_=backend_;
                return;
            }
        }
#endif
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend),ggml_backend_buffer_free);
        if (!buffer_) throw std::runtime_error("device cache allocation");
        ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer_.get(), 0);
        copy_backend_ = backend_;
        if (shared_) return;
        bool pinned_staging=mode != ScheduleMode::naive;
#ifdef SOLID_HAVE_CUDA_SELECTION
        pinned_staging=pinned_staging || ggml_backend_is_cuda(backend_);
#endif
        if (pinned_staging) {
            auto * device = ggml_backend_get_device(backend_);
            if (!device) throw std::runtime_error("copy backend requires a registered device");
#ifdef SOLID_HAVE_SYCL
            if (mode != ScheduleMode::naive && !sycl_copy_)
#else
            if (mode != ScheduleMode::naive)
#endif
            {
                copy_owner_.reset(ggml_backend_dev_init(device, nullptr));
                if (!copy_owner_) throw std::runtime_error("independent copy backend initialization failed");
                copy_backend_ = copy_owner_.get();
            }
            auto * host_type = ggml_backend_dev_host_buffer_type(device);
            if (!host_type) host_type = ggml_backend_get_default_buffer_type(copy_backend_);
            if (!ggml_backend_buft_is_host(host_type)) throw std::runtime_error("copy staging requires host memory");
            const auto bytes = static_cast<std::size_t>(geometry.heads) * per_head * geometry.block_tokens * 2 * geometry.dimension * sizeof(Half);
            staging_.reset(ggml_backend_buft_alloc_buffer(host_type, bytes),ggml_backend_buffer_free);
            if (!staging_) throw std::runtime_error("pinned copy staging allocation failed");
            stats_.staging_bytes = ggml_backend_buffer_get_size(staging_.get());
        }
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (ggml_backend_is_cuda(backend_)) {
            auto * registry=ggml_backend_cuda_reg();
            for (std::size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(registry);++ordinal) {
                if (ggml_backend_reg_dev_get(registry,ordinal)!=ggml_backend_get_device(backend_)) continue;
                cuda_device_=static_cast<int>(ordinal);
                solid::trace::name_cuda_stream(solid_cuda_queue(backend_), "model.compute");
                if (copy_backend_ != backend_) solid::trace::name_cuda_stream(solid_cuda_queue(copy_backend_), "kv.H2D");
                copy_intervals_.resize(static_cast<std::size_t>(geometry_.heads)*per_head_);
                break;
            }
        }
#endif
    }
    ~GgmlTransport() override {
        if (!events_.empty() || archive_pending_) {
            try {
                if (archive_on_compute_) synchronize_compute();
                drain();
            } catch (...) { std::terminate(); }
            for (const auto & event : events_) if (event.second.event) ggml_backend_event_free(event.second.event);
            clear_archive();
        }
    }
    std::uint64_t upload(std::uint32_t slot, const KvBlock & block) override {
        if (slot >= geometry_.heads * static_cast<std::uint64_t>(per_head_) || block.layout != geometry_.layout ||
            block.dimension != geometry_.dimension || block.capacity != geometry_.block_tokens ||
            block.storage_size() != static_cast<std::size_t>(geometry_.block_tokens) * 2 * geometry_.dimension) {
            throw std::invalid_argument("device upload shape");
        }
        if (next_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("copy ticket exhausted");
        const auto id = next_++;
        const auto bytes = block.storage_size() * sizeof(Half);
        if (shared_) {
            if (block.storage()!=shared_slot(slot)) throw std::logic_error("shared upload changed the physical slot");
            events_.emplace(id, CopyRecord{});
            return id;
        }
#ifdef SOLID_HAVE_SYCL
        if (sycl_copy_) {
            auto & record=events_.try_emplace(id).first->second;
            try {
                const auto * source=stage_host(slot,0,block.storage(),bytes);
                for (std::size_t plane=0;plane<planes();++plane)
                    sycl_copy_->copy(static_cast<unsigned char *>(tensor_->data)+slot_offset(slot,plane),
                        static_cast<const unsigned char *>(source)+plane*bytes/planes(),bytes/planes());
                record.sycl=sycl_copy_->fence();
            } catch (...) {
                try { drain(); } catch (...) {}
                events_.erase(id);
                throw;
            }
            return id;
        }
#endif
        auto * device = ggml_backend_get_device(copy_backend_);
        auto * event = device ? ggml_backend_event_new(device) : nullptr;
        try {
            auto & record=events_.try_emplace(id).first->second;
            record.event=event;
            const auto * source=stage_host(slot,0,block.storage(),bytes);
            begin_copy_interval(record,slot);
            for (std::size_t plane=0;plane<planes();++plane)
                ggml_backend_tensor_set_async(copy_backend_, tensor_, static_cast<const unsigned char *>(source)+plane*bytes/planes(),
                    slot_offset(slot,plane), bytes/planes());
            record_copy_interval(record.interval,true);
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
    bool batches_uploads() const override {
#ifdef SOLID_HAVE_CUDA_SELECTION
        return cuda_device_>=0 && staging_ && !shared_;
#else
        return false;
#endif
    }
    std::uint64_t upload_batch(const std::vector<DeviceUpload> & blocks) override {
        if (!batches_uploads() || blocks.empty()) throw std::logic_error("invalid CUDA KV upload batch");
        if (next_==UINT64_MAX) throw std::overflow_error("copy ticket exhausted");
        struct Range { std::size_t offset; std::uintptr_t source; std::size_t bytes; };
        const auto bytes=static_cast<std::size_t>(geometry_.block_tokens)*2*geometry_.dimension*sizeof(Half);
        std::vector<Range> ranges;
        ranges.reserve(blocks.size()*planes());
        for (const auto & copy : blocks) {
            const auto & block=*copy.block;
            if (copy.slot>=static_cast<std::uint64_t>(geometry_.heads)*per_head_ || block.layout!=geometry_.layout ||
                block.dimension!=geometry_.dimension || block.capacity!=geometry_.block_tokens ||
                block.storage_size()!=bytes/sizeof(Half)) throw std::invalid_argument("batched device upload shape");
            const auto * source=stage_host(copy.slot,0,block.storage(),bytes);
            for (std::size_t plane=0;plane<planes();++plane)
                ranges.push_back({slot_offset(copy.slot,plane),reinterpret_cast<std::uintptr_t>(source)+plane*bytes/planes(),bytes/planes()});
        }
        std::sort(ranges.begin(),ranges.end(),[](const Range & a,const Range & b) { return a.offset<b.offset; });
        const auto id=next_++;
        auto & record=events_.try_emplace(id).first->second;
        std::uint64_t submitted=0;
        try {
            auto * device=ggml_backend_get_device(copy_backend_);
            record.event=ggml_backend_event_new(device);
            if (!record.event) throw std::runtime_error("CUDA KV batch completion event allocation failed");
            if (blocks.size()==1) begin_copy_interval(record,blocks.front().slot);
            for (std::size_t i=0;i<ranges.size();) {
                auto range=ranges[i++];
                while (stats_.coalesce_uploads && i<ranges.size() && range.offset+range.bytes==ranges[i].offset &&
                       range.source+range.bytes==ranges[i].source) {
                    range.bytes+=ranges[i++].bytes;
                }
                ggml_backend_tensor_set_async(copy_backend_,tensor_,reinterpret_cast<const void *>(range.source),range.offset,range.bytes);
                ++submitted;
            }
            record_copy_interval(record.interval,true);
            ggml_backend_event_record(record.event,copy_backend_);
        } catch (...) {
            ggml_backend_synchronize(copy_backend_);
            if (record.event) ggml_backend_event_free(record.event);
            events_.erase(id);
            throw;
        }
        ++stats_.historical_upload_batches;
        stats_.historical_upload_blocks+=blocks.size();
        stats_.historical_upload_ranges+=submitted;
        return id;
    }
    std::optional<DeviceMemory> current_memory() const override {
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (ggml_backend_is_cuda(backend_)) return DeviceMemory::cuda;
#endif
#ifdef SOLID_HAVE_SYCL
        if (sycl_copy_) return DeviceMemory::sycl;
#endif
        auto * device=ggml_backend_get_device(backend_);
        if (device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU) return DeviceMemory::host_emulator;
        return {};
    }
    void bind_current(ggml_tensor * source,const CurrentDeviceKv & current) {
        const auto memory=current_memory();
        if (!memory) throw std::invalid_argument("current device transfer backend");
        validate_device_kv(current.kv,*memory,geometry_.heads,geometry_.dimension,1);
        if (!source || source->type!=GGML_TYPE_F16 || !ggml_is_contiguous(source) || source->ne[0]!=2ll*geometry_.dimension ||
            source->ne[1]!=geometry_.heads || source->ne[2]!=1 || source->ne[3]!=1 || source->data!=current.kv.data) {
            throw std::invalid_argument("current device tensor shape or source identity");
        }
        auto * buffer=source->view_src?source->view_src->buffer:source->buffer;
        if (!buffer || (*memory==DeviceMemory::host_emulator)!=ggml_backend_buffer_is_host(buffer) ||
            device_ranges_overlap(source->data,current.kv.elements,sizeof(Half),tensor_->data,
                static_cast<std::size_t>(ggml_nelements(tensor_)),sizeof(Half))) throw std::invalid_argument("current device source buffer or alias");
        if (*memory!=DeviceMemory::host_emulator && ggml_backend_buffer_get_type(buffer)!=ggml_backend_buffer_get_type(tensor_->buffer)) {
            throw std::invalid_argument("current KV and bank use different device buffer types");
        }
        current_source_=source;
    }
    void clear_current() noexcept { current_source_=nullptr; }
    std::uint64_t upload_current(std::uint32_t slot,const KvBlock * block,const CurrentDeviceKv & current,std::uint32_t prefix) override {
        const auto row=static_cast<std::uint32_t>(current.position%geometry_.block_tokens);
        const auto packed_row_bytes=static_cast<std::size_t>(2)*geometry_.dimension*sizeof(Half);
        const auto row_bytes=packed_row_bytes/planes();
        const auto version=row+(current.unarchived?0:1);
        if (!current_source_ || current_source_->data!=current.kv.data || slot>=static_cast<std::uint64_t>(geometry_.heads)*per_head_ || prefix>row ||
            (version!=0 && !block) || (block && (block->layout!=geometry_.layout || block->dimension!=geometry_.dimension ||
            block->capacity!=geometry_.block_tokens || block->storage_size()!=static_cast<std::size_t>(geometry_.block_tokens)*2*geometry_.dimension ||
            block->valid_tokens!=version || block->version!=version || block->first_position!=current.position-row))) {
            throw std::invalid_argument("current device transfer snapshot");
        }
        if (next_==UINT64_MAX) throw std::overflow_error("copy ticket exhausted");
        const auto id=next_++;
        const auto head=slot/per_head_;
        const auto prefix_offset=static_cast<std::size_t>(prefix)*row_bytes;
        const auto prefix_bytes=static_cast<std::size_t>(row-prefix)*row_bytes;
        const auto plane_bytes=static_cast<std::size_t>(geometry_.block_tokens)*row_bytes;
        const auto * current_row=static_cast<const unsigned char *>(current_source_->data)+head*packed_row_bytes;
        const auto prefix_source = [&](std::size_t plane) -> const void * {
            if (!prefix_bytes) return nullptr;
            const auto source_offset=plane*plane_bytes+prefix_offset;
            return stage_host(slot,source_offset,reinterpret_cast<const unsigned char *>(block->storage())+source_offset,prefix_bytes);
        };
#ifdef SOLID_HAVE_SYCL
        if (sycl_copy_) {
            auto & record=events_.try_emplace(id).first->second;
            try {
                for (std::size_t plane=0;plane<planes();++plane) {
                    auto * target=static_cast<unsigned char *>(tensor_->data)+slot_offset(slot,plane);
                    if (prefix_bytes) sycl_copy_->copy(target+prefix_offset,prefix_source(plane),prefix_bytes);
                    sycl_copy_->copy(target+row*row_bytes,current_row+plane*row_bytes,row_bytes);
                }
                record.sycl=sycl_copy_->fence();
            } catch (...) {
                try { drain(); } catch (...) {}
                events_.erase(id);
                throw;
            }
            return id;
        }
#endif
        auto * device=ggml_backend_get_device(copy_backend_);
        auto * event=device?ggml_backend_event_new(device):nullptr;
        try {
            auto rows=std::make_unique<std::array<RowCopy,2>>();
            for (std::size_t plane=0;plane<planes();++plane) {
                (*rows)[plane].source=row_tensor(*current_source_,const_cast<unsigned char *>(current_row)+plane*row_bytes);
                (*rows)[plane].target=row_tensor(*tensor_,static_cast<unsigned char *>(tensor_->data)+slot_offset(slot,plane)+row*row_bytes);
            }
            auto & record=events_.try_emplace(id).first->second;
            record.event=event;
            record.row=std::move(rows);
            begin_copy_interval(record,slot);
            for (std::size_t plane=0;plane<planes();++plane) {
                if (prefix_bytes)
                    ggml_backend_tensor_set_async(copy_backend_,tensor_,prefix_source(plane),slot_offset(slot,plane)+prefix_offset,prefix_bytes);
                // Separate K/V use two views covered by the same completion.
                ggml_backend_tensor_copy_async(copy_backend_,copy_backend_,&(*record.row)[plane].source,&(*record.row)[plane].target);
            }
            record_copy_interval(record.interval,true);
            if (event) ggml_backend_event_record(event,copy_backend_);
            else ggml_backend_synchronize(copy_backend_);
        } catch (...) {
            ggml_backend_synchronize(copy_backend_);
            events_.erase(id); if (event) ggml_backend_event_free(event);
            throw;
        }
        return id;
    }
    bool complete(std::uint64_t ticket, bool wait) override {
        const auto found = events_.find(ticket);
        if (found == events_.end()) throw std::logic_error("unknown upload ticket");
        copy_interval_.reset();
#ifdef SOLID_HAVE_SYCL
        if (found->second.sycl && ticket>completed_through_) {
            if (!wait) {
                ++stats_.copy_event_queries;
                if (!found->second.sycl->ready()) return false;
            }
            found->second.sycl->wait();
            completed_through_=ticket;
            stats_.event_waits+=wait;
        }
#endif
        if (found->second.event && ticket > completed_through_ && !wait) {
            if (!native_) return false;
            ++stats_.copy_event_queries;
            if (!native_->copy_ready(found->second.event)) return false;
            completed_through_=ticket;
        }
        if (found->second.event) {
            if (ticket > completed_through_) { ggml_backend_event_synchronize(found->second.event); ++stats_.event_waits; }
        }
        try {
            if (found->second.interval) copy_interval_=found->second.interval->duration();
#ifdef SOLID_HAVE_SYCL
            if (found->second.sycl) copy_interval_=found->second.sycl->duration();
#endif
        } catch (...) {
            // Transfer completion is established independently of profiling.
            copy_interval_.reset();
        }
        if (found->second.event) ggml_backend_event_free(found->second.event);
        events_.erase(found);
        return true;
    }
    std::optional<std::uint64_t> copy_interval() const override { return copy_interval_; }
    void drain() override {
#ifdef SOLID_HAVE_SYCL
        if (sycl_copy_) sycl_copy_->drain();
        else
#endif
            ggml_backend_synchronize(copy_backend_);
        ++stats_.backend_drains;
        completed_through_=next_-1;
    }
    void wait_batch() override {
        if (events_.empty()) return;
        ++stats_.batch_groups; stats_.events_coalesced += events_.size();
        const auto latest = events_.rbegin();
        if (latest->first > completed_through_) {
#ifdef SOLID_HAVE_SYCL
            if (sycl_copy_) {
                for (auto i=events_.rbegin(); i!=events_.rend() && i->first>completed_through_; ++i) {
                    if (!i->second.sycl) continue;
                    i->second.sycl->wait();
                    ++stats_.batch_waits;
                    break;
                }
            } else
#endif
            if (latest->second.event) { ggml_backend_event_synchronize(latest->second.event); ++stats_.batch_waits; }
        }
        completed_through_ = next_ - 1;
    }
    void wait_on_compute() {
#ifdef SOLID_HAVE_SYCL
        if (sycl_copy_) {
            for (auto i=events_.rbegin(); i!=events_.rend(); ++i) {
                if (!i->second.sycl) continue;
                i->second.sycl->wait_on_compute();
                ++stats_.stream_waits;
                break;
            }
            return;
        }
#endif
        if (!events_.empty() && events_.rbegin()->second.event) {
            const auto & last=*events_.rbegin();
#ifdef SOLID_HAVE_CUDA_SELECTION
            if (cuda_device_>=0) {
                if (load_wait_pending_) throw std::logic_error("previous KV wait measurement has not been collected");
                if (!load_wait_) load_wait_=make_cuda_device_interval(cuda_device_);
                record_copy_interval(load_wait_.get(),false,backend_);
            }
#endif
            ggml_backend_event_wait(backend_,last.second.event);
#ifdef SOLID_HAVE_CUDA_SELECTION
            if (cuda_device_>=0) {
                record_copy_interval(load_wait_.get(),true,backend_);
                load_wait_pending_=true;
            }
#endif
            compute_dependency_ticket_=last.first;
            ++stats_.stream_waits;
        }
    }
    void acknowledge_compute_dependency() {
        completed_through_=std::max(completed_through_,compute_dependency_ticket_);
        compute_dependency_ticket_=0;
    }
    std::optional<std::uint64_t> collect_cache_load_wait() {
        if (!load_wait_pending_) return {};
        // The existing attention receipt proves both timestamps are complete.
        const auto elapsed=load_wait_->duration();
        if (!elapsed) throw std::logic_error("missing CUDA KV wait timestamps");
        load_wait_pending_=false;
        return elapsed;
    }
    bool can_archive_on_attention_stream() const {
#ifdef SOLID_HAVE_CUDA_SELECTION
        return cuda_device_>=0 && native_;
#else
        return false;
#endif
    }
    void * cuda_compute_stream() const {
#ifdef SOLID_HAVE_CUDA_SELECTION
        return solid_cuda_queue(backend_);
#else
        return nullptr;
#endif
    }
    void start_archive(const std::vector<std::uint32_t> & slots,std::uint64_t position,ggml_tensor * source=nullptr) {
        if (archive_pending_ || (!source && slots.size()!=geometry_.heads)) throw std::logic_error("invalid KV archive frontier");
        auto * archive_backend=source?backend_:copy_backend_;
        auto * device=ggml_backend_get_device(archive_backend);
        const auto row_bytes=static_cast<std::size_t>(2)*geometry_.dimension*sizeof(Half);
        const auto bytes=row_bytes*geometry_.heads;
        if (!archive_buffer_) {
            auto * type=device?ggml_backend_dev_host_buffer_type(device):nullptr;
            if (!type) type=ggml_backend_get_default_buffer_type(copy_backend_);
            if (!ggml_backend_buft_is_host(type)) throw std::runtime_error("KV archive needs host-accessible storage");
            archive_buffer_.reset(ggml_backend_buft_alloc_buffer(type,bytes));
            if (!archive_buffer_) throw std::runtime_error("KV archive allocation failed");
            stats_.staging_bytes+=ggml_backend_buffer_get_size(archive_buffer_.get());
        }
#ifdef SOLID_HAVE_SYCL
        if (!sycl_copy_)
#endif
            if (!source) archive_event_=device?ggml_backend_event_new(device):nullptr;
        archive_pending_=true;
        archive_on_compute_=source!=nullptr;
        try {
#ifdef SOLID_HAVE_CUDA_SELECTION
            if (cuda_device_>=0 && !archive_interval_) archive_interval_=make_cuda_device_interval(cuda_device_);
#endif
            record_copy_interval(archive_interval_.get(),false,archive_backend);
            auto * output=static_cast<unsigned char *>(ggml_backend_buffer_get_base(archive_buffer_.get()));
            if (source) {
                // The packed projection is already ready and contiguous across heads.
                // Queue one download before the compute stream waits for the KV bank.
                ggml_backend_tensor_get_async(backend_,source,output,0,bytes);
                record_copy_interval(archive_interval_.get(),true,archive_backend);
                return;
            }
            for (std::size_t h=0;h<slots.size();++h) {
                if (slots[h]/per_head_!=h) throw std::logic_error("KV archive slot belongs to the wrong head");
                const auto token=(slots[h]%per_head_)*geometry_.block_tokens+position%geometry_.block_tokens;
                for (std::size_t plane=0;plane<planes();++plane) {
                    const auto offset=h*tensor_->nb[2]+token*tensor_->nb[1]+plane*tensor_->nb[3];
                    const auto bytes=row_bytes/planes();
#ifdef SOLID_HAVE_SYCL
                    if (sycl_copy_) sycl_copy_->copy(output+h*row_bytes+plane*bytes,static_cast<const unsigned char *>(tensor_->data)+offset,bytes);
                    else
#endif
                        ggml_backend_tensor_get_async(copy_backend_,tensor_,output+h*row_bytes+plane*bytes,offset,bytes);
                }
            }
            record_copy_interval(archive_interval_.get(),true);
#ifdef SOLID_HAVE_SYCL
            if (sycl_copy_) archive_sycl_=sycl_copy_->fence();
            else
#endif
            if (archive_event_) ggml_backend_event_record(archive_event_,copy_backend_);
            else ggml_backend_synchronize(copy_backend_);
        } catch (...) {
            if (archive_on_compute_) synchronize_compute();
            else drain();
            clear_archive();
            throw;
        }
    }
    std::vector<Half> finish_archive(std::optional<std::uint64_t> & device_ns,bool compute_complete) {
        if (!archive_pending_ || archive_on_compute_!=compute_complete) throw std::logic_error("missing KV archive completion proof");
        if (compute_complete) ++stats_.archive_attention_reuses;
        else ++stats_.archive_separate_completions;
#ifdef SOLID_HAVE_SYCL
        if (archive_sycl_) archive_sycl_->wait();
#endif
        if (archive_event_) ggml_backend_event_synchronize(archive_event_);
        try {
            if (archive_interval_) device_ns=archive_interval_->duration();
#ifdef SOLID_HAVE_SYCL
            if (archive_sycl_) device_ns=archive_sycl_->duration();
#endif
        } catch (...) { device_ns.reset(); }
        const auto * data=static_cast<const Half *>(ggml_backend_buffer_get_base(archive_buffer_.get()));
        clear_archive();
        return {data,data+static_cast<std::size_t>(geometry_.heads)*2*geometry_.dimension};
    }
    // The caller must already have drained the stream carrying the archive.
    void clear_archive() {
#ifdef SOLID_HAVE_SYCL
        archive_sycl_.reset();
#endif
        if (archive_event_) ggml_backend_event_free(archive_event_);
        archive_event_=nullptr;
        archive_pending_=false;
        archive_on_compute_=false;
    }
    void synchronize_compute() { ggml_backend_synchronize(backend_); ++stats_.backend_compute_waits; }
    ScheduleStats stats() const { return stats_; }
    std::uint64_t resident_bytes() const override { return ggml_backend_buffer_get_size(buffer_.get()); }
    bool shared_memory() const override { return shared_; }
    bool aliases_host_slots() const override { return shared_; }
    bool binds_host_slots() const override { return shared_ || staging_!=nullptr; }
    bool defer_prefetch_slots() const override { return shared_ || can_archive_on_attention_stream(); }
    std::shared_ptr<void> host_storage_owner() const override { return shared_?buffer_:staging_; }
    Half * host_slot_storage(std::uint32_t slot) const override {
        if (shared_) return shared_slot(slot);
        if (!staging_ || slot>=static_cast<std::uint64_t>(geometry_.heads)*per_head_) throw std::out_of_range("pinned KV slot");
        const auto elements=static_cast<std::size_t>(geometry_.block_tokens)*2*geometry_.dimension;
        return static_cast<Half *>(ggml_backend_buffer_get_base(staging_.get()))+slot*elements;
    }
    Half * shared_slot(std::uint32_t slot) const {
        if (!shared_ || slot>=static_cast<std::uint64_t>(geometry_.heads)*per_head_) throw std::out_of_range("shared KV slot");
        const auto offset=(slot/per_head_)*tensor_->nb[2]+(slot%per_head_)*geometry_.block_tokens*tensor_->nb[1];
        return reinterpret_cast<Half *>(static_cast<unsigned char *>(tensor_->data)+offset);
    }
    ggml_tensor * tensor() const { return tensor_; }
private:
    std::size_t planes() const { return geometry_.layout==KvLayout::interleaved?1:2; }
    std::size_t slot_offset(std::uint32_t slot,std::size_t plane) const {
        return (slot/per_head_)*tensor_->nb[2]+(slot%per_head_)*geometry_.block_tokens*tensor_->nb[1]+plane*tensor_->nb[3];
    }
    void begin_copy_interval(CopyRecord & record,std::uint32_t slot) {
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (cuda_device_<0) return;
        auto & interval=copy_intervals_[slot];
        if (!interval) interval=make_cuda_device_interval(cuda_device_);
        record.interval=interval.get();
        record_copy_interval(record.interval,false);
#else
        (void) record; (void) slot;
#endif
    }
    void record_copy_interval(DeviceInterval * interval,bool end,ggml_backend_t backend=nullptr) {
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (interval) interval->record(solid_cuda_queue(backend?backend:copy_backend_),end);
#else
        (void) interval; (void) end; (void) backend;
#endif
    }
    const void * stage_host(std::uint32_t,std::size_t,const void * source,std::size_t bytes) {
        if (!staging_) return source;
        const auto begin=reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(staging_.get()));
        const auto address=reinterpret_cast<std::uintptr_t>(source);
        const auto size=ggml_backend_buffer_get_size(staging_.get());
        if (address<begin || address-begin>size || bytes>size-(address-begin)) throw std::logic_error("KV source is outside its pinned I/O bank");
        return source;
    }
    ggml_tensor row_tensor(const ggml_tensor & source,void * data) const {
        auto row=source; row.type=GGML_TYPE_F16; row.op=GGML_OP_NONE; row.flags=0;
        row.buffer=source.view_src?source.view_src->buffer:source.buffer;
        row.view_src=nullptr; row.view_offs=0; row.data=data; row.extra=nullptr;
        for (auto & input:row.src) input=nullptr;
        row.ne[0]=2ll*geometry_.dimension/planes();
        for (int i=1;i<GGML_MAX_DIMS;++i) row.ne[i]=1;
        row.nb[0]=sizeof(Half);
        for (int i=1;i<GGML_MAX_DIMS;++i) row.nb[i]=static_cast<std::size_t>(row.ne[0])*sizeof(Half);
        return row;
    }
    ggml_backend_t backend_;
    KvGeometry geometry_;
    std::uint32_t per_head_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> copy_owner_{nullptr, ggml_backend_free};
    ggml_backend_t copy_backend_ = nullptr;
#ifdef SOLID_HAVE_CUDA_SELECTION
    int cuda_device_=-1;
    std::vector<std::unique_ptr<DeviceInterval>> copy_intervals_;
#endif
    std::unique_ptr<DeviceInterval> archive_interval_;
    std::optional<std::uint64_t> copy_interval_;
    std::unique_ptr<DeviceInterval> load_wait_;
    bool load_wait_pending_=false;
#ifdef SOLID_HAVE_SYCL
    std::unique_ptr<SyclCopyQueue> sycl_copy_;
    std::unique_ptr<SyclCopyEvent> archive_sycl_;
#endif
    std::shared_ptr<ggml_backend_buffer> staging_;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> archive_buffer_{nullptr, ggml_backend_buffer_free};
    ggml_backend_event_t archive_event_=nullptr;
    bool archive_pending_=false;
    bool archive_on_compute_=false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::shared_ptr<ggml_backend_buffer> buffer_;
    ggml_tensor * tensor_ = nullptr;
    bool shared_ = false;
    std::map<std::uint64_t, CopyRecord> events_;
    ggml_tensor * current_source_=nullptr;
    std::uint64_t next_ = 1;
    std::uint64_t completed_through_ = 0;
    std::uint64_t compute_dependency_ticket_ = 0;
    ScheduleStats stats_;
    AttentionEvents * native_;
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
    std::vector<SlotLease> history;
    TaskDag dag;
    std::shared_ptr<BlockStage> blocks;
    std::shared_ptr<const StoreReceipt> store;
    bool consumed = false;
    bool direct = false;
    bool deferred_archive;
    bool archive_at_attention;
    std::optional<std::uint64_t> consumer_entry_ns;
    LayerWork(std::uint32_t layer, std::uint64_t position, ScheduleMode mode, ScheduleTrace & trace,
                std::uint64_t id, const std::array<std::uint64_t, task_count> & estimates, bool archive, bool single_layer,
                bool shared_archive) :
        layer(layer), position(position), dag([&] {
            const bool serial = mode == ScheduleMode::naive;
            std::vector<TaskSpec> specs = {
                {"q_gpu_ready", TaskResource::gpu, {}, 1},
                {"q_d2h_ready", TaskResource::d2h, {q_ready}, 1},
                {serial ? "selection_ready" : "selection", TaskResource::host, serial ? std::vector<std::uint32_t>{append,q_d2h} : std::vector<std::uint32_t>{q_d2h}, 1},
                {"kv_projection_to_host", TaskResource::gpu, serial ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{q_ready}, 1},
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
            if (archive) {
                if (!shared_archive) specs[kv_d2h].predecessors={current_h2d};
                specs[append].predecessors={kv_d2h};
                specs[current_h2d].predecessors={kv_projection,miss_read,history_h2d};
                if (shared_archive) {
                    specs[current_h2d].predecessors.pop_back();
                    specs[attention].predecessors.push_back(history_h2d);
                    specs[next_prefetch].predecessors={miss_read};
                }
                if (single_layer) specs[next_prefetch].predecessors.push_back(append);
            }
            for (std::size_t i = 0; i < specs.size(); ++i) specs[i].estimate_ns = estimates[i];
            return specs;
        }(), &trace, id, layer, position, mode), deferred_archive(archive), archive_at_attention(shared_archive) {}
};
}
struct GgmlPrefetch::Impl {
    AttentionSession & session;
    ggml_backend_t backend;
    ScheduleMode mode;
    const bool per_completion_uploads;
    TaskWorker worker;
    mutable std::mutex mutex;
    ScheduleTrace trace;
    ScheduleTrace block_trace;
    std::unique_ptr<AttentionEvents> events;
    GgmlTransport transport;
    DeviceCache device;
    ScheduleStats totals;
    std::array<std::uint64_t, task_count> estimates;
    std::uint64_t block_read_estimate=1,block_copy_estimate=1;
    std::shared_ptr<LayerWork> work;
    std::deque<std::shared_ptr<LayerWork>> storing;
    std::uint64_t sequence = 0;
    bool quiesced = true;
    std::atomic<std::uint64_t> released_epoch{0};
    Impl(AttentionSession & s, ggml_backend_t backend, const std::filesystem::path & residency, ScheduleMode mode,
        const std::filesystem::path & schedule,int cpu,std::unique_ptr<AttentionEvents> supplied,bool share_host_cache,bool coalesce_uploads,bool per_completion) :
        session(s), backend(backend), mode(mode), per_completion_uploads(per_completion), worker(cpu,"sa.coordinator"), trace(schedule), block_trace(schedule.empty()?std::filesystem::path{}:
            schedule.parent_path()/(schedule.stem().string()+".blocks"+schedule.extension().string())),
        events(attention_events(s,backend,mode,std::move(supplied))),
        transport(backend, s.geometry(), s.prefetcher().slots_per_head(), mode,events.get(),share_host_cache,coalesce_uploads),
        device(s.prefetcher(), s.geometry(), s.query_heads(), transport, residency) {
        if (per_completion && (coalesce_uploads || mode==ScheduleMode::naive)) throw std::invalid_argument("invalid per-completion upload schedule");
        totals.per_completion_uploads=per_completion;
        estimates.fill(1); totals.coordinator_cpu=cpu; totals.completion_backend=events?events->name():"graph_boundaries";
    }
    std::shared_ptr<LayerWork> new_work(std::uint32_t layer, std::uint64_t position,bool direct=false,
                                        std::optional<std::uint64_t> projection_ns={},bool deferred_archive=true) {
        if (work && work->dag.task(attention).state != TaskState::complete) throw std::logic_error("new layer started before previous attention completion");
        if (sequence == UINT64_MAX) throw std::overflow_error("schedule frame identity exhausted");
        const bool archive=direct && deferred_archive && mode!=ScheduleMode::naive;
        work = std::make_shared<LayerWork>(layer, position, mode, trace, ++sequence, estimates, archive,
            session.geometry().layers==1, archive && mode==ScheduleMode::sync_reuse && transport.can_archive_on_attention_stream());
        const auto time=trace.now();
        if (projection_ns) {
            work->dag.start(q_ready,time);
            work->dag.complete(q_ready,time,"query_device_projection",projection_ns);
            work->dag.fence({q_d2h},time,"query_device_input_ready");
        } else work->dag.fence({q_ready,q_d2h},time,direct?"query_device_input_ready":"query_download_complete");
        return work;
    }
    void finish_frame(const std::shared_ptr<LayerWork> & finished) {
        const auto & work = finished;
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
            const auto duration = task.duration_ns();
            if (duration) estimates[id] = duration >= estimates[id] ? estimates[id] + (duration - estimates[id]) / 8 : estimates[id] - (estimates[id] - duration) / 8;
        }
    }
    void observe_stores() {
        for (auto it = storing.begin(); it != storing.end();) {
            auto & frame = *it;
            if (!frame->store->completed_ns) { ++it; continue; }
            frame->dag.complete(store_task, *frame->store->completed_ns, "original_write_complete");
            if (frame->dag.finished()) finish_frame(frame);
            it = storing.erase(it);
        }
    }
    void start_store(const std::shared_ptr<LayerWork> & frame) {
        frame->dag.start(store_task, trace.now());
        frame->store = session.prefetcher().store_full_layer(frame->layer);
        if (frame->store) storing.push_back(frame);
        else frame->dag.complete(store_task, trace.now(), "buffered_or_synchronous_store");
    }
    void next_layer(const std::shared_ptr<LayerWork> & current) {
        const auto next = current->layer + 1 < session.geometry().layers ? current->layer + 1 : 0;
        const auto position = next ? current->position : current->position + 1;
        current->dag.start(next_prefetch, trace.now());
        if (position<session.geometry().capacity_tokens) {
            begin_prediction(next, position);
            current->dag.complete(next_prefetch, trace.now(),transport.aliases_host_slots()?"shared_prefetch_planned":"");
        } else current->dag.complete(next_prefetch,trace.now(),"context_capacity_reached");
    }
    void begin_prediction(std::uint32_t layer, std::uint64_t position) {
        session.begin_prefetch(layer, position, {block_read_estimate, block_copy_estimate});
        observe_stores();
    }
    void finish_archive(const std::shared_ptr<LayerWork> & frame) {
        std::optional<std::uint64_t> archive_ns;
        const auto kv=transport.finish_archive(archive_ns,frame->archive_at_attention);
        frame->dag.complete(kv_d2h,trace.now(),
            frame->archive_at_attention?"kv_archive_attention_complete":"kv_archive_copy_complete",archive_ns);
        frame->dag.start(append,trace.now());
        session.archive_device_kv(frame->layer,frame->position,kv);
        frame->dag.complete(append,trace.now());
    }
    void release_attention(const char * proof,std::optional<std::uint64_t> device_ns={},const std::function<void()> & before_upload={}) {
        if (!device.attention_pending()) return;
        if (work->archive_at_attention) {
            // No successor upload may be submitted until the attention reader retires.
            // This receipt also covers the last upload event waited on by compute.
            transport.acknowledge_compute_dependency();
            if (const auto wait=transport.collect_cache_load_wait()) {
                totals.cache_load_device_block_ns+=*wait;
                ++totals.cache_load_device_block_layers;
            }
            device.reap_transfers();
            const auto now=trace.now();
            work->dag.complete(history_h2d,now,"attention_upload_dependency");
            work->dag.complete(current_h2d,now,"attention_upload_dependency");
            work->dag.start(attention,now);
        }
        if (device.attention_complete(device.attention_epoch())) {
            const auto completed=trace.now();
            work->dag.complete(attention, completed, proof, device_ns);
            totals.attention_layer_span_ns+=completed-work->dag.task(q_ready).end_ns;
            ++totals.attention_layer_span_layers;
            if (!work->consumer_entry_ns) throw std::logic_error("attention completed without its KV adapter entry");
            totals.attention_consumer_span_ns+=completed-*work->consumer_entry_ns;
            ++totals.attention_consumer_span_layers;
            if (work->archive_at_attention) {
                finish_archive(work);
                if (work->dag.task(next_prefetch).state==TaskState::pending) next_layer(work);
            }
            if (work->dag.finished()) finish_frame(work);
            if (per_completion_uploads) {
                device.prefetch_ready(before_upload,true);
                session.prefetcher().poll(false,[&](const SlotLease & lease) {
                    device.stage_ready_batch({lease},{},before_upload);
                });
                observe_stores();
            } else {
                session.prefetcher().poll();
                observe_stores();
                device.prefetch_ready(before_upload);
            }
            if (work->archive_at_attention) {
                start_store(work);
                if (work->dag.finished()) finish_frame(work);
            }
            released_epoch.store(device.attention_epoch(), std::memory_order_release);
            worker.notify_progress();
        }
    }
    void progress_after_attention(const std::shared_ptr<LayerWork> & current,std::uint64_t epoch) {
        if (!events) return;
        bool released=false;
        auto & host=session.prefetcher();
        const auto before_upload=[this,epoch] {
            if (!events->handoff_seen(epoch) && !events->stopped()) ++totals.uploads_before_boundary;
        };
        while (!events->stopped()) {
            // Clear notifications before inspecting queues/state. A producer
            // arriving after these checks leaves a counter for the next wait.
            events->acknowledge();
            if (current->archive_at_attention && !released) device.reap_transfers();
            const Prefetcher::ReadyBlock upload_ready=per_completion_uploads && released ?
                Prefetcher::ReadyBlock{[&](const SlotLease & lease) { device.stage_ready_batch({lease},{},before_upload); }} : Prefetcher::ReadyBlock{};
            host.poll(false,upload_ready); observe_stores();
            if (!released) if (const auto receipt=events->poll(current->layer,epoch)) {
                const char * proof;
                if (receipt->proof==AttentionProof::device_signal) {
                    ++totals.device_event_completions; totals.device_attention_ns+=receipt->device_ns;
                    proof=receipt->device_ns?"attention_device_event":"attention_device_event_untimed";
                } else {
                    ++totals.device_boundary_fallbacks;
                    if (receipt->proof==AttentionProof::query_boundary) { ++totals.query_reused_fences; proof="query_dependency_reuse"; }
                    else if (receipt->proof==AttentionProof::process_boundary) { ++totals.process_reused_fences; proof="process_synchronize_reuse"; }
                    else { ++totals.attention_receipt_reuses; proof="attention_dependency_reuse"; }
                }
                release_attention(proof, receipt->device_ns ? std::optional<std::uint64_t>{receipt->device_ns} : std::nullopt,before_upload);
                device.reap_transfers();
                released=true;
            }
            if (released) {
                // Give formal correction the coordinator as soon as its query
                // reaches the CPU. Remaining predicted SSD jobs stay queued.
                if (events->handoff_seen(epoch)) return;
                if ((host.pending_reads() || host.pending_stores()) && host.completion_handle()<0) {
                    host.poll(true,upload_ready);
                    observe_stores();
                }
                if (!per_completion_uploads) device.prefetch_ready(before_upload);
                if (!host.pending_reads() && !host.pending_stores()) {
                    device.reap_transfers();
                    return;
                }
            }
#ifdef __linux__
            pollfd descriptors[2]={{events->completion_handle(),POLLIN,0},{host.completion_handle(),POLLIN,0}};
            int status;
            do { status=::poll(descriptors,2,10); } while (status<0 && errno==EINTR);
            if (status<0) throw std::system_error(errno,std::generic_category(),"wait for attention or prefetch completion");
            for (const auto & descriptor:descriptors) if (descriptor.revents&(POLLERR|POLLHUP|POLLNVAL)) throw std::runtime_error("device completion notification failed");
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        }
    }
};
GgmlPrefetch::GgmlPrefetch(AttentionSession & session, ggml_backend_t backend, const std::filesystem::path & trace,
                          ScheduleMode mode, const std::filesystem::path & schedule_trace,int cpu,std::unique_ptr<AttentionEvents> events,bool share_host_cache,bool coalesce_uploads,bool per_completion_uploads) :
    impl_(std::make_unique<Impl>(session, backend, trace, mode, schedule_trace,cpu,std::move(events),share_host_cache,coalesce_uploads,per_completion_uploads)) {}
GgmlPrefetch::~GgmlPrefetch() { if (!impl_->quiesced) { try { quiesce(); } catch (...) {} } }
ggml_tensor * GgmlPrefetch::tensor() const { return impl_->transport.tensor(); }
std::uint32_t GgmlPrefetch::tokens_per_head() const { return impl_->device.tokens_per_head(); }
std::uint64_t GgmlPrefetch::attention_epoch() const { return impl_->device.attention_epoch(); }
bool GgmlPrefetch::early_enabled() const { return impl_->mode != ScheduleMode::naive; }
bool GgmlPrefetch::reuses_sync() const { return impl_->mode == ScheduleMode::sync_reuse; }
StreamCompletionToken * GgmlPrefetch::completion_token(std::uint32_t layer) const { return impl_->events?impl_->events->token(layer):nullptr; }
bool GgmlPrefetch::completion_host_marker() const { return impl_->events && impl_->events->host_marker(); }
void GgmlPrefetch::begin_prefetch(std::uint32_t layer, std::uint64_t position) {
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->begin_prediction(layer, position);
}
void GgmlPrefetch::select_early(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, bool covers) {
    select_impl(layer, position, query, nullptr, covers);
}
std::optional<DeviceMemory> GgmlPrefetch::direct_memory() const {
    const auto memory = impl_->session.direct_memory();
    if (!memory) return {};
    if (*memory == DeviceMemory::host_emulator && ggml_backend_dev_type(ggml_backend_get_device(impl_->backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) return memory;
#ifdef SOLID_HAVE_CUDA_SELECTION
    if (*memory == DeviceMemory::cuda && ggml_backend_is_cuda(impl_->backend)) return memory;
#endif
#ifdef SOLID_HAVE_SYCL
    if (*memory == DeviceMemory::sycl && ggml_backend_is_sycl(impl_->backend)) return memory;
#endif
    return {};
}
void GgmlPrefetch::select_early_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query, bool covers,
                                       std::optional<std::uint64_t> projection_ns,bool deferred_archive) {
    select_impl(layer,position,{},&query,covers,projection_ns,false,deferred_archive);
}
void GgmlPrefetch::select_early_device_async(std::uint32_t layer,std::uint64_t position,DeviceQuery query,bool covers,
                                            std::optional<std::uint64_t> projection_ns,bool deferred_archive) {
    select_impl(layer, position, {}, &query, covers,projection_ns,true,deferred_archive);
}
void GgmlPrefetch::select_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query, const DeviceQuery * device,
                                bool covers,std::optional<std::uint64_t> projection_ns,bool asynchronous,bool deferred_archive) {
    if (!early_enabled()) throw std::logic_error("early selection is disabled");
    if (device) {
        const auto memory=direct_memory();
        if (!memory || device->memory!=*memory) throw std::invalid_argument("device selection backend mismatch");
    }
    if (impl_->events) {
        if (covers) impl_->events->boundary(AttentionProof::query_boundary);
        else impl_->events->stop();
    }
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->quiesced = false;
    if (impl_->device.attention_pending()) {
        if (reuses_sync() && covers) { ++impl_->totals.query_reused_fences; impl_->release_attention("query_dependency_reuse"); }
        else {
            impl_->transport.synchronize_compute();
            if (impl_->events) impl_->events->retire_after_sync();
            ++impl_->totals.fallback_fences; impl_->release_attention("explicit_fallback");
        }
    }
    impl_->begin_prediction(layer, position);
    const auto work = impl_->new_work(layer, position,device!=nullptr,projection_ns,deferred_archive);
    work->query = query;
    work->direct = device != nullptr;
    // Queue score/top-k while the graph is at its Q frontier. Waking an I/O
    // worker before submission can miss the entire KV-projection window.
    const bool submitted = asynchronous && device && device->memory == DeviceMemory::cuda;
    if (submitted) {
        work->dag.start(select_task, impl_->trace.now());
        impl_->session.submit_before_kv_device(layer, position, *device);
    }
    if (!asynchronous) {
        work->dag.start(select_task,impl_->trace.now());
        work->selection=device?impl_->session.select_before_kv_device(layer,position,*device):
            impl_->session.select_before_kv(layer,position,query);
        work->dag.complete(select_task,impl_->trace.now());
    }
    work->dag.start(kv_projection, impl_->trace.now());
    const auto device_query=device && asynchronous?std::optional<DeviceQuery>{*device}:std::nullopt;
    ++impl_->totals.worker_jobs;
    impl_->worker.submit([this, work, device_query, asynchronous, submitted] {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        // Q is producer-complete. The graph retains its device allocation
        // through the KV and mask nodes while selection uses its own queue.
        if (asynchronous) {
            if (!submitted) work->dag.start(select_task, impl_->trace.now());
            work->selection = submitted ? impl_->session.finish_before_kv_device() : device_query ?
                impl_->session.select_before_kv_device(work->layer,work->position,*device_query) :
                impl_->session.select_before_kv(work->layer,work->position,work->query);
            work->dag.complete(select_task, impl_->trace.now());
        }
        work->dag.start(miss_read, impl_->trace.now());
        const bool admit_before_stage = work->archive_at_attention && !impl_->transport.aliases_host_slots();
        if (!admit_before_stage) impl_->device.wait_transfers();
        auto & host = impl_->session.prefetcher();
        bool copying=false;
        auto leases = host.resolve_history(work->layer, work->position, work->selection,
            [&](const std::vector<BlockIdentity> & plan) { impl_->device.begin_stage(plan,
                work->direct?std::optional<CurrentBlock>{CurrentBlock{work->layer,work->position}}:std::optional<CurrentBlock>{}); },
            impl_->per_completion_uploads ? Prefetcher::ReadyBlock{[&](const SlotLease & lease) {
                if (!copying) { work->dag.start(history_h2d,impl_->trace.now()); copying=true; }
                impl_->device.stage_ready_batch({lease},work->blocks);
            }} : Prefetcher::ReadyBlock{},[&](const std::vector<BlockWork> & plan) {
                auto blocks=plan;
                const auto ready=host.ready();
                for (auto & block:blocks) {
                    block.read_ns=std::any_of(ready.begin(),ready.end(),[&](const SlotLease & lease) { return lease.identity==block.identity; })?1:impl_->block_read_estimate;
                    block.copy_ns=impl_->device.find(block.identity)?1:impl_->block_copy_estimate;
                }
                work->blocks=std::make_shared<BlockStage>(std::move(blocks),&impl_->block_trace,impl_->sequence,work->layer,work->position,impl_->mode);
                return work->blocks;
            }, work->deferred_archive,impl_->per_completion_uploads ? Prefetcher::ReadyBatch{} : Prefetcher::ReadyBatch{[&](const std::vector<SlotLease> & ready) {
                if (!copying) { work->dag.start(history_h2d,impl_->trace.now()); copying=true; }
                impl_->device.stage_ready_batch(ready,work->blocks);
            }}, admit_before_stage,impl_->per_completion_uploads);
        if (work->deferred_archive) work->history = std::move(leases);
        else for (const auto & lease : leases) host.unpin(lease);
        work->dag.complete(miss_read, impl_->trace.now());
        if (!copying) work->dag.start(history_h2d, impl_->trace.now());
    });
}
std::vector<Half> GgmlPrefetch::decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                      const std::vector<float> & keys, const std::vector<float> & values, bool packed) {
    return decode_impl(layer, position, query, keys, values, packed, nullptr, nullptr, nullptr, nullptr);
}
std::vector<Half> GgmlPrefetch::decode_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys,
                                          std::optional<DeviceQuery> serial_query,ggml_tensor * source) {
    return decode_impl(layer, position, {}, {}, {}, true, &kv, &keys, serial_query?&*serial_query:nullptr,source);
}
std::vector<Half> GgmlPrefetch::decode_device(std::uint32_t layer,std::uint64_t position,DeviceKeys keys,ggml_tensor * source,
                                            std::optional<std::uint64_t> projection_ns) {
    if (!source || !early_enabled()) throw std::invalid_argument("deferred KV requires a device source and an early query");
    return decode_impl(layer, position, {}, {}, {}, true, nullptr, &keys, nullptr, source,projection_ns);
}
std::vector<Half> GgmlPrefetch::decode_impl(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                          const std::vector<float> & keys, const std::vector<float> & values, bool packed,
                                          const std::vector<Half> * half, const DeviceKeys * device, const DeviceQuery * serial_query,ggml_tensor * source,
                                          std::optional<std::uint64_t> projection_ns) {
    const auto consumer_entry=impl_->trace.now();
    const auto current = impl_->work;
    const bool deferred = device && !half;
    if ((half && !device) || (device && (early_enabled() == (serial_query!=nullptr)))) {
        throw std::invalid_argument("device decode query frontier mismatch");
    }
    if (early_enabled()) {
        if (!current || current->layer != layer || current->position != position || current->consumed) throw std::logic_error("KV arrived without its early query");
        if (current->direct != (device != nullptr) || current->deferred_archive!=deferred) throw std::logic_error("KV memory does not match early query");
        if (deferred) current->dag.complete(kv_projection, impl_->trace.now(), "kv_device_ready",projection_ns);
        else current->dag.fence({kv_projection,kv_d2h}, impl_->trace.now(), "kv_download_complete");
    }
    const auto consumer_ready=impl_->trace.now();
    const auto wait_begin = std::chrono::steady_clock::now();
    {
        trace::Scope wait("CPU Wait KV", layer, position);
        impl_->worker.join();
    }
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-wait_begin).count();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->totals.history_consumer_wait_ns += static_cast<std::uint64_t>(waited);
    ++impl_->totals.history_consumer_wait_calls;
    if (early_enabled()) {
        const auto selection=current->dag.task(select_task);
        const auto loading=current->dag.task(miss_read);
        if (selection.state!=TaskState::complete || loading.state!=TaskState::complete)
            throw std::logic_error("KV load blocking measurement requires completed selection and loading");
        impl_->totals.cache_load_host_block_ns+=cache_load_host_block_ns(consumer_ready,selection.end_ns,loading.end_ns);
        ++impl_->totals.cache_load_host_block_layers;
    }
    impl_->quiesced = false;
    auto work = current;
    if (!early_enabled()) {
        impl_->device.wait_transfers(); impl_->begin_prediction(layer, position);
        work = impl_->new_work(layer, position,half!=nullptr);
        work->dag.fence({kv_projection,kv_d2h}, impl_->trace.now(), half?"serial_kv_fp16_download_complete":"serial_kv_download_complete");
        work->dag.start(append, impl_->trace.now());
        if (half) {
            work->selection = impl_->session.select_before_kv_device(layer,position,*serial_query);
            impl_->session.append_after_selection_half(layer,position,*half,*device);
        } else work->selection = impl_->session.select_decode(layer, position, query, keys, values, packed);
        work->dag.complete(append, impl_->trace.now());
        work->dag.fence({select_task}, impl_->trace.now(), "serial_selection_complete");
        work->dag.start(miss_read, impl_->trace.now());
    } else {
        if (!work->archive_at_attention) {
            if (reuses_sync()) impl_->device.wait_transfers_coalesced(); else impl_->device.wait_transfers();
            work->dag.complete(history_h2d, impl_->trace.now(), "historical_copy_complete");
        }
        if (deferred) impl_->session.stage_device_keys(layer,position,*device,
            work->archive_at_attention?impl_->transport.cuda_compute_stream():nullptr);
        else {
            work->dag.start(append, impl_->trace.now());
            if (half) impl_->session.append_after_selection_half(layer, position, *half, *device);
            else impl_->session.append_after_selection(layer, position, work->query, keys, values, packed);
            work->dag.complete(append, impl_->trace.now());
        }
    }
    work->consumer_entry_ns=consumer_entry;
    auto & host = impl_->session.prefetcher();
    if (!deferred) work->history = host.resolve(layer, position, work->selection);
    if (!early_enabled()) {
        work->dag.complete(miss_read, impl_->trace.now());
        work->dag.fence({history_h2d}, impl_->trace.now(), "serial_no_early_upload");
    }
    work->dag.start(current_h2d, impl_->trace.now());
    std::vector<Half> mask;
    const auto elements=deferred?static_cast<std::size_t>(ggml_nelements(source)):half?half->size():0;
    const CurrentDeviceKv current_kv{layer,position,{device?device->data:nullptr,elements,device?device->memory:DeviceMemory::cuda},deferred};
    try {
        if (source) impl_->transport.bind_current(source,current_kv);
        if (deferred) host.remember_selection(layer, position, work->selection);
        mask = impl_->device.prepare(work->history, work->selection, reuses_sync(),source?&current_kv:nullptr,
                                     work->archive_at_attention,
                                     impl_->session.budget().infllm_attention ? impl_->session.budget().local : 0);
        impl_->transport.clear_current();
    } catch (...) { impl_->transport.clear_current(); throw; }
    for (const auto & lease : work->history) host.unpin(lease);
    work->history.clear();
    work->consumed = true;
    const auto epoch=impl_->device.attention_epoch();
    if (impl_->events) impl_->events->arm(layer,epoch);
    if (!early_enabled()) {
        work->dag.complete(current_h2d, impl_->trace.now(), "serial_upload_complete");
        work->dag.start(attention, impl_->trace.now());
        work->dag.fence({store_task}, impl_->trace.now(), "serial_buffered_store");
        impl_->next_layer(work);
    } else {
        if (work->archive_at_attention) {
            const auto archive_start=impl_->trace.now();
            impl_->transport.start_archive({},position,source);
            work->dag.start(kv_d2h,archive_start);
        }
        if (reuses_sync()) impl_->transport.wait_on_compute();
        else {
            work->dag.complete(current_h2d, impl_->trace.now(), "upload_complete");
            work->dag.start(attention, impl_->trace.now());
        }
        if (deferred && !work->archive_at_attention) impl_->transport.start_archive(impl_->device.current_slots({layer,position}), position);
        ++impl_->totals.worker_jobs;
        impl_->worker.submit([this, work,epoch] {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (work->archive_at_attention) {
                if (impl_->session.geometry().layers>1) impl_->next_layer(work);
                impl_->progress_after_attention(work,epoch);
                return;
            }
            bool store_started = false;
            if (reuses_sync()) {
                auto & host = impl_->session.prefetcher();
                if (!work->deferred_archive && host.asynchronous_stores()) {
                    // The append is complete and the write buffers do not
                    // borrow the host slots that the copy stream still owns.
                    impl_->start_store(work);
                    store_started = true;
                }
                impl_->device.wait_transfers_coalesced();
                work->dag.complete(current_h2d, impl_->trace.now(), "shared_upload_batch");
                work->dag.start(attention, impl_->trace.now());
            }
            bool next_started=false;
            if (work->deferred_archive) {
                work->dag.start(kv_d2h,impl_->trace.now());
                if (impl_->session.geometry().layers>1) { impl_->next_layer(work); next_started=true; }
                impl_->finish_archive(work);
            }
            // Dispatch the remaining I/O tasks at their shared frontier.
            for (int remaining = 2-store_started-next_started; remaining; --remaining) {
                const auto io = work->dag.ready({store_task,next_prefetch});
                if (io.empty()) throw std::logic_error("post-attention I/O dependencies missing");
                if (io.front() == next_prefetch) impl_->next_layer(work);
                else impl_->start_store(work);
            }
            impl_->progress_after_attention(work,epoch);
        });
    }
    return mask;
}
void GgmlPrefetch::completed(std::uint64_t epoch) {
    if (impl_->events && epoch!=impl_->device.attention_epoch()) return;
    if (impl_->events) {
        impl_->events->boundary(AttentionProof::attention_boundary,epoch);
        impl_->worker.wait_until([&] { return impl_->released_epoch.load(std::memory_order_acquire)>=epoch; });
        if (impl_->released_epoch.load(std::memory_order_acquire)>=epoch) return;
    }
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->device.attention_pending() && epoch == impl_->device.attention_epoch()) {
        ++impl_->totals.attention_fences; impl_->release_attention("attention_backend_complete");
    }
}
void GgmlPrefetch::process_complete() {
    if (impl_->events) impl_->events->boundary(AttentionProof::process_boundary);
    impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->device.attention_pending()) { ++impl_->totals.process_reused_fences; impl_->release_attention("process_synchronize_reuse"); }
}
void * GgmlPrefetch::probability_compute_stream() const {
    return impl_->mode == ScheduleMode::sync_reuse ? impl_->transport.cuda_compute_stream() : nullptr;
}
void GgmlPrefetch::quiesce() {
    if (impl_->quiesced) return;
    if (impl_->events) impl_->events->stop();
    std::exception_ptr error;
    try { impl_->worker.join(); } catch (...) { error = std::current_exception(); }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->transport.synchronize_compute(); impl_->transport.drain();
    if (!error && impl_->device.attention_pending() && impl_->work && impl_->work->archive_at_attention &&
        impl_->work->dag.task(kv_d2h).state==TaskState::running) {
        try {
            impl_->finish_archive(impl_->work);
            impl_->start_store(impl_->work);
        } catch (...) { error=std::current_exception(); }
    }
    impl_->transport.clear_archive();
    if (impl_->events) impl_->events->retire_after_sync();
    try { impl_->device.wait_transfers(); } catch (...) { if (!error) error = std::current_exception(); }
    if (impl_->work) {
        for (const auto & lease : impl_->work->history) {
            try { impl_->session.prefetcher().unpin(lease); } catch (...) { if (!error) error=std::current_exception(); }
        }
        impl_->work->history.clear();
    }
    try {
        if (impl_->device.attention_pending()) (void) impl_->device.attention_complete(impl_->device.attention_epoch());
    } catch (...) { if (!error) error = std::current_exception(); }
    try { impl_->device.finish(); } catch (...) { if (!error) error = std::current_exception(); }
    try {
        impl_->session.prefetcher().finish_stores();
        impl_->observe_stores();
    } catch (...) { if (!error) error = std::current_exception(); }
    for (const auto & frame : impl_->storing) {
        try { if (!frame->dag.finished()) frame->dag.abort(impl_->trace.now()); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    impl_->storing.clear();
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
    result.coalesce_uploads=transport.coalesce_uploads;
    result.event_waits = transport.event_waits; result.batch_waits = transport.batch_waits;
    result.stream_waits = transport.stream_waits; result.staging_bytes = transport.staging_bytes;
    result.backend_drains = transport.backend_drains; result.backend_compute_waits = transport.backend_compute_waits;
    result.batch_groups = transport.batch_groups; result.events_coalesced = transport.events_coalesced;
    result.copy_event_queries=transport.copy_event_queries;
    result.historical_upload_batches=transport.historical_upload_batches;
    result.historical_upload_blocks=transport.historical_upload_blocks;
    result.historical_upload_ranges=transport.historical_upload_ranges;
    result.archive_attention_reuses=transport.archive_attention_reuses;
    result.archive_separate_completions=transport.archive_separate_completions;
    return result;
}
} // namespace solid
