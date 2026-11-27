// 把主机已就绪的 KV 块送入显存，构造 attention 槽位映射，在计算完成后释放占用。
#include "solid/device_cache.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <stdexcept>

namespace solid {
DeviceCache::DeviceCache(Prefetcher & host, KvGeometry geometry, std::uint32_t query_heads, DeviceTransport & transport,
                         const std::filesystem::path & trace) :
    host_(host), geometry_(geometry), query_heads_(query_heads), transport_(transport),
    slots_(geometry.heads, host.slots_per_head()+(transport.aliases_host_slots()?1:0)) {
    geometry.validate();
    if (!query_heads || query_heads % geometry.heads != 0) throw std::invalid_argument("invalid device cache geometry");
    (void) tokens_per_head();
    if (transport_.binds_host_slots()) {
        std::vector<Half *> addresses;
        for (std::uint32_t head=0;head<geometry.heads;++head) for (std::uint32_t slot=0;slot<host.slots_per_head();++slot) {
            addresses.push_back(transport_.host_slot_storage(head*slots_.slots_per_head()+slot));
        }
        host_.bind_shared_blocks(addresses,transport_.host_storage_owner(),transport_.defer_prefetch_slots());
    }
    if (!trace.empty()) {
        if (std::filesystem::exists(trace)) throw std::runtime_error("device trace already exists");
        trace_.open(trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create device trace");
    }
}
std::optional<SlotLease> DeviceCache::acquire_shared(const BlockIdentity & identity,std::uint32_t index) {
    const auto & entry=slots_.entry(index);
    if (entry.state!=SlotState::free) {
        if (entry.readers || entry.producer) throw std::logic_error("shared slot still has a device reader or writer");
        const auto old=slots_.lease(index);
        event("overwrite_available",old); slots_.retire(old); ++stats_.overwritten_blocks;
    }
    return slots_.acquire_at(identity,index);
}
DeviceCache::~DeviceCache() {
    try { wait_transfers(); }
    catch (...) { try { transport_.drain(); } catch (...) { std::terminate(); } }
}
std::uint32_t DeviceCache::tokens_per_head() const {
    const auto tokens = (static_cast<std::uint64_t>(slots_.slots_per_head()) * geometry_.block_tokens + 31) / 32 * 32;
    if (tokens > static_cast<std::uint64_t>(INT32_MAX)) throw std::overflow_error("device cache length exceeds backend limits");
    return static_cast<std::uint32_t>(tokens);
}
void DeviceCache::event(const char * name, const SlotLease & lease,const CurrentDeviceKv * current,std::uint32_t retained,std::uint64_t source_generation) {
    if (!trace_.is_open()) return;
    trace_ << "{\"event\":\"" << name << "\",\"timestamp_ns\":"
           << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
           << ",\"epoch\":" << epoch_ << ",\"layer\":" << lease.identity.layer
           << ",\"kv_head\":" << lease.identity.head << ",\"block\":" << lease.identity.block << ",\"version\":" << lease.identity.version
           << ",\"slot\":" << lease.slot << ",\"generation\":" << lease.generation;
    if (current) {
        const auto row=current->position%geometry_.block_tokens;
        const auto row_bytes=static_cast<std::uint64_t>(2)*geometry_.dimension*sizeof(Half);
        trace_ << ",\"copy_kind\":\"current_device\",\"current_position\":" << current->position
               << ",\"retained_prefix_tokens\":" << retained << ",\"prefix_generation\":" << source_generation
               << ",\"row_bytes\":" << row_bytes << ",\"host_copy_bytes\":" << (row-retained)*row_bytes
               << ",\"device_copy_bytes\":" << row_bytes;
    }
    trace_ << "}\n";
    if (!trace_) throw std::runtime_error("device residency trace write failed");
}
std::shared_ptr<trace::Batch> DeviceCache::upload_phase(std::uint32_t layer, bool prefetch) {
    if (!trace::enabled() || transport_.aliases_host_slots()) return {};
    auto phase = upload_phase_.lock();
    if (!phase || layer != upload_layer_ || prefetch != upload_prefetch_) {
        phase = std::make_shared<trace::Batch>(prefetch ? "CPU->GPU Prefetch" : "CPU->GPU Load", layer, UINT64_MAX);
        upload_phase_ = phase; upload_layer_ = layer; upload_prefetch_ = prefetch;
    }
    phase->submitted();
    return phase;
}
void DeviceCache::poll_transfers(bool wait,std::optional<std::uint64_t> through) {
    std::exception_ptr error;
    for (auto it = transfers_.begin(); it != transfers_.end();) {
        if (!transport_.complete(it->ticket,wait && (!through || it->ticket<=*through))) { ++it; continue; }
        for (const auto & slot : it->slots) {
            if (slot.host) host_.unpin(*slot.host);
            const bool current = slots_.complete(slot.device);
            if (!current) ++stats_.stale_events;
            try {
                const auto & stage = slot.stage ? slot.stage : it->stage;
                if (stage) stage->copy_complete(slot.device.identity,"copy_ticket_complete");
                if (current) event("upload_complete",slot.device);
            } catch (...) { if (!error) error=std::current_exception(); }
        }
        it = transfers_.erase(it);
    }
    if (error) std::rethrow_exception(error);
}
void DeviceCache::reap_transfers() { poll_transfers(false); }
void DeviceCache::wait_transfers() { poll_transfers(true); }
void DeviceCache::wait_transfers_coalesced() { if (!transfers_.empty()) transport_.wait_batch(); poll_transfers(true); }
void DeviceCache::retain(const std::vector<BlockIdentity> & identities) {
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto & entry = slots_.entry(i);
        if (entry.state == SlotState::free || std::find(identities.begin(), identities.end(), entry.identity) != identities.end()) continue;
        if (entry.readers || entry.producer) throw std::logic_error("device slot reuse before transfer/attention completion");
        const auto lease = slots_.lease(i);
        event("overwrite_available", lease);
        slots_.retire(lease);
        ++stats_.overwritten_blocks;
    }
}
SlotLease DeviceCache::upload(const SlotLease & host,const std::shared_ptr<BlockStage> & stage,const std::function<void()> & before_upload) {
    auto existing = slots_.find(host.identity);
    const auto shared_index=host.identity.head*slots_.slots_per_head()+host.slot%host_.slots_per_head();
    if (existing && transport_.aliases_host_slots() && existing->slot!=shared_index) {
        const auto & old=slots_.entry(existing->slot);
        if (old.readers || old.producer) throw std::logic_error("shared identity moved while still in use");
        event("overwrite_available",*existing); slots_.retire(*existing); ++stats_.overwritten_blocks; existing.reset();
    }
    if (existing) {
        if (stage) {
            stage->copy_started(host.identity);
            if (slots_.entry(existing->slot).producer) {
                bool attached = false;
                for (auto & transfer : transfers_) for (auto & slot : transfer.slots) {
                    if (slot.device == *existing) {
                        if (slot.stage) throw std::logic_error("device upload already has a formal consumer");
                        slot.stage = stage;
                        attached = true;
                    }
                }
                if (!attached) throw std::logic_error("device upload lost its completion receipt");
            } else stage->copy_complete(host.identity,"device_resident");
        }
        return *existing;
    }
    const auto device = transport_.aliases_host_slots() ? acquire_shared(host.identity,
        shared_index) : slots_.acquire(host.identity);
    if (!device) throw std::runtime_error("device slot capacity exceeded");
    host_.pin(host);
    std::uint64_t ticket = 0;
    try {
        const auto & block = host_.block(host);
        if (stage) stage->copy_started(host.identity);
        if (before_upload) before_upload();
        auto phase = upload_phase(host.identity.layer, !stage);
        ticket = transport_.upload(device->slot, block);
        transfers_.push_back({ticket, {{host,*device}},stage,std::move(phase)});
        ++stats_.uploads;
        const auto bytes = block.storage_size() * sizeof(Half);
        if (!transport_.aliases_host_slots()) {
            if (transport_.shared_memory()) stats_.shared_host_copy_bytes += bytes;
            else stats_.upload_bytes += bytes;
        }
        event("upload", *device);
    } catch (...) {
        // Even allocation/trace errors after upload must retain the host lease
        // until the transport has released every source buffer.
        transport_.drain();
        if (ticket && !transfers_.empty() && transfers_.back().ticket == ticket) transfers_.pop_back();
        if (ticket) (void) transport_.complete(ticket, true);
        host_.unpin(host);
        slots_.retire(*device);
        slots_.complete(*device);
        if (stage) { try { stage->abort(); } catch (...) {} }
        throw;
    }
    return *device;
}
// 同批提交已就绪主机块，并保留源槽位直到共享传输回执完成。
void DeviceCache::upload_batch(const std::vector<SlotLease> & leases,const std::shared_ptr<BlockStage> & stage,
                               const std::function<void()> & before_upload) {
    Transfer transfer{0,{},stage};
    transfer.slots.reserve(leases.size());
    std::vector<DeviceUpload> copies;
    copies.reserve(leases.size());
    std::uint64_t bytes=0;
    try {
        for (const auto & host : leases) {
            if (slots_.find(host.identity)) {
                if (stage) (void) upload(host,stage);
                continue;
            }
            const auto & block=host_.block(host);
            // Prefer matching offsets in the pinned and device banks so adjacent
            // blocks can be sent directly with one contiguous transfer.
            const auto preferred=host.identity.head*slots_.slots_per_head()+host.slot%host_.slots_per_head();
            auto device=slots_.acquire_at(host.identity,preferred);
            if (!device) device=slots_.acquire(host.identity);
            if (!device) throw std::runtime_error("device slot capacity exceeded");
            transfer.slots.push_back({{},*device});
            host_.pin(host);
            transfer.slots.back().host=host;
            if (stage) stage->copy_started(host.identity);
            copies.push_back({device->slot,&block});
            bytes+=block.storage_size()*sizeof(Half);
        }
        if (copies.empty()) return;
        if (before_upload) before_upload();
        transfer.phase = upload_phase(transfer.slots.front().device.identity.layer, !stage);
        transfer.ticket=transport_.upload_batch(copies);
        for (const auto & slot : transfer.slots) event("upload",slot.device);
        transfers_.push_back(std::move(transfer));
        stats_.uploads+=copies.size();
        stats_.upload_bytes+=bytes;
    } catch (...) {
        if (transfer.ticket) {
            transport_.drain();
            (void) transport_.complete(transfer.ticket,true);
        }
        for (const auto & slot : transfer.slots) {
            if (slot.host) host_.unpin(*slot.host);
            slots_.retire(slot.device);
            slots_.complete(slot.device);
        }
        if (stage) { try { stage->abort(); } catch (...) {} }
        throw;
    }
}
std::optional<SlotLease> DeviceCache::prefix(const BlockIdentity & identity) const {
    std::optional<SlotLease> best;
    for (std::uint32_t i=0;i<slots_.size();++i) {
        if (transport_.aliases_host_slots() && i%slots_.slots_per_head()!=slots_.slots_per_head()-1) continue;
        const auto & entry=slots_.entry(i);
        if (entry.state!=SlotState::ready || entry.producer || entry.readers) continue;
        const auto & old=entry.identity;
        if (old.layer==identity.layer && old.head==identity.head && old.block==identity.block && old.version<identity.version &&
            (!best || old.version>best->identity.version)) best=slots_.lease(i);
    }
    return best;
}
SlotLease DeviceCache::upload_current(const BlockIdentity & identity,const std::optional<SlotLease> & host,const CurrentDeviceKv & current) {
    const auto row=static_cast<std::uint32_t>(current.position%geometry_.block_tokens);
    const auto previous=prefix(identity);
    const auto retained=previous?static_cast<std::uint32_t>(previous->identity.version):0;
    if (retained>row) throw std::logic_error("current device prefix extends beyond the historical rows");
    const auto device=previous?std::optional<SlotLease>{slots_.refresh(*previous,identity)}:
        transport_.aliases_host_slots()?acquire_shared(identity,(identity.head+1)*slots_.slots_per_head()-1):slots_.acquire(identity);
    if (!device) throw std::runtime_error("current device slot capacity exceeded");
    std::uint64_t ticket=0;
    bool pinned=false;
    try {
        const KvBlock * block=nullptr;
        if (host) { host_.pin(*host); pinned=true; block=&host_.block(*host); }
        if (previous) event("overwrite_available",*previous);
        ticket=transport_.upload_current(device->slot,block,current,retained);
        transfers_.push_back({ticket,{{host,*device}},{}});
        const auto row_bytes=static_cast<std::uint64_t>(2)*geometry_.dimension*sizeof(Half);
        ++stats_.uploads; ++stats_.current_device_rows; stats_.current_device_bytes+=row_bytes;
        if (transport_.shared_memory()) stats_.shared_host_copy_bytes+=(row-retained)*row_bytes;
        else stats_.upload_bytes+=(row-retained)*row_bytes;
        stats_.current_host_bytes+=(row-retained)*row_bytes;
        stats_.retained_prefix_bytes+=retained*row_bytes;
        if (previous) { ++stats_.refreshed_tails; ++stats_.overwritten_blocks; }
        event("upload",*device,&current,retained,previous?previous->generation:0);
    } catch (...) {
        transport_.drain();
        if (ticket && !transfers_.empty() && transfers_.back().ticket==ticket) transfers_.pop_back();
        if (ticket) (void) transport_.complete(ticket,true);
        if (pinned) host_.unpin(*host);
        slots_.retire(*device); slots_.complete(*device);
        throw;
    }
    return *device;
}
void DeviceCache::prefetch_ready(const std::function<void()> & before_upload,bool individual) {
    if (!attention_.empty()) throw std::logic_error("prefetch upload while device attention still owns slots");
    poll_transfers(false);
    const auto ready = host_.ready();
    std::vector<BlockIdentity> identities;
    for (const auto & lease : ready) identities.push_back(lease.identity);
    // Transfers from an earlier prefetch pass still own both endpoints.
    for (const auto & transfer : transfers_) for (const auto & slot : transfer.slots) identities.push_back(slot.device.identity);
    retain(identities);
    if (transport_.batches_uploads()) {
        if (individual) {
            for (const auto & lease:ready) upload_batch({lease},{},before_upload);
        } else upload_batch(ready,{},before_upload);
        return;
    }
    for (const auto & lease : ready) {
        const auto existing=slots_.find(lease.identity);
        if (transport_.aliases_host_slots() && existing && existing->slot%slots_.slots_per_head()==slots_.slots_per_head()-1) continue;
        (void) upload(lease,{},before_upload);
    }
}
void DeviceCache::stage(const std::vector<SlotLease> & host_leases) {
    if (!attention_.empty()) throw std::logic_error("stage before previous attention completed");
    wait_transfers();
    std::vector<BlockIdentity> actual;
    for (const auto & lease : host_leases) actual.push_back(lease.identity);
    retain(actual);
    for (const auto & lease : host_leases) (void) upload(lease);
}
void DeviceCache::begin_stage(const std::vector<BlockIdentity> & identities,std::optional<CurrentBlock> current) {
    if (!attention_.empty()) throw std::logic_error("streamed stage before previous attention completed");
    if (current && (current->layer>=geometry_.layers || current->position>=geometry_.capacity_tokens)) throw std::invalid_argument("current prefix stage position");
    auto keep=identities;
    if (current) for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto & entry = slots_.entry(i);
        if (entry.state != SlotState::free && entry.identity.layer == current->layer &&
            entry.identity.block == current->position / geometry_.block_tokens) keep.push_back(entry.identity);
    }
    if (transport_.defer_prefetch_slots() && !transport_.aliases_host_slots()) {
        reap_transfers();
        std::optional<std::uint64_t> reuse_ticket;
        for (const auto & transfer : transfers_) {
            if (std::any_of(transfer.slots.begin(), transfer.slots.end(), [&](const TransferSlot & slot) {
                return std::find(keep.begin(), keep.end(), slot.device.identity) == keep.end();
            })) reuse_ticket = transfer.ticket;
        }
        if (reuse_ticket) poll_transfers(true, reuse_ticket);
    } else {
        wait_transfers();
    }
    retain(keep);
}
void DeviceCache::stage_ready(const SlotLease & lease,const std::shared_ptr<BlockStage> & stage,const std::function<void()> & before_upload) {
    if (!attention_.empty()) throw std::logic_error("streamed upload while attention owns slots");
    poll_transfers(false);
    const auto existing=slots_.find(lease.identity);
    if (transport_.aliases_host_slots() && existing && existing->slot%slots_.slots_per_head()==slots_.slots_per_head()-1) {
        if (slots_.entry(existing->slot).producer) throw std::logic_error("current prefix is still being installed");
        if (stage) { stage->copy_started(lease.identity); stage->copy_complete(lease.identity,"device_resident"); }
        return;
    }
    (void) upload(lease,stage,before_upload);
}
void DeviceCache::stage_ready_completion(const SlotLease & lease,const std::shared_ptr<BlockStage> & stage,const std::function<void()> & before_upload) {
    if (!transport_.batches_uploads()) { stage_ready(lease,stage,before_upload); return; }
    if (!attention_.empty()) throw std::logic_error("completion upload while attention owns slots");
    upload_batch({lease},stage,before_upload);
}
void DeviceCache::stage_ready_batch(const std::vector<SlotLease> & leases,const std::shared_ptr<BlockStage> & stage,const std::function<void()> & before_upload) {
    if (!transport_.batches_uploads()) {
        for (const auto & lease : leases) stage_ready(lease,stage,before_upload);
        return;
    }
    if (!attention_.empty()) throw std::logic_error("batched upload while attention owns slots");
    poll_transfers(false);
    upload_batch(leases,stage,before_upload);
}
// 把逻辑 token 位置映射为显存槽位和掩码，保留当前 attention 的读租约。
std::vector<Half> DeviceCache::prepare(const std::vector<SlotLease> & host_leases, const std::vector<HeadSelection> & selection,
                                       bool deferred_upload,const CurrentDeviceKv * current,bool keep_history_in_flight,
                                       std::uint32_t local_window) {
    if (!attention_.empty()) throw std::logic_error("previous device attention has not completed");
    if (keep_history_in_flight && (!deferred_upload || !current || !transport_.defer_prefetch_slots() || transport_.aliases_host_slots())) {
        throw std::logic_error("in-flight history requires a fenced device copy queue");
    }
    if (selection.size() != geometry_.heads) throw std::invalid_argument("device selection head count");
    std::vector<std::optional<SlotLease>> current_hosts;
    std::vector<BlockIdentity> actual;
    if (current) {
        const auto memory=transport_.current_memory();
        if (!memory || current->layer>=geometry_.layers || current->position>=geometry_.capacity_tokens) throw std::invalid_argument("current device KV backend or position");
        validate_device_kv(current->kv,*memory,geometry_.heads,geometry_.dimension,1);
        current_hosts.resize(geometry_.heads);
        const auto block=current->position/geometry_.block_tokens;
        const auto version=current->position%geometry_.block_tokens+(current->unarchived?0:1);
        std::set<std::pair<std::uint32_t,std::uint64_t>> unique;
        for (const auto & lease:host_leases) {
            const auto & id=lease.identity;
            if (id.layer!=current->layer || id.head>=geometry_.heads || !id.version || id.version>geometry_.block_tokens ||
                id.block>current->position/geometry_.block_tokens || !unique.emplace(id.head,id.block).second) {
                throw std::invalid_argument("current device working set identities");
            }
            if (id.block==block) {
                if (id.version!=version) throw std::invalid_argument("current device working set has an invalid host prefix");
                current_hosts[id.head]=lease;
            } else actual.push_back(id);
        }
        for (std::uint32_t head=0;head<geometry_.heads;++head) {
            if (selection[head].positions.empty() || selection[head].positions.back()!=current->position ||
                (version && !current_hosts[head])) {
                throw std::invalid_argument("current device KV is absent from the formal working set");
            }
        }
    } else for (const auto & lease:host_leases) actual.push_back(lease.identity);
    if (keep_history_in_flight) {
        reap_transfers();
        // Refreshing a partially uploaded tail changes its slot generation.
        // Retire only through the last pending tail before refreshing it.
        auto last_tail=transfers_.end();
        for (auto it=transfers_.begin();it!=transfers_.end();++it) {
            if (std::any_of(it->slots.begin(),it->slots.end(),[&](const TransferSlot & slot) {
                    return slot.device.identity.layer==current->layer &&
                        slot.device.identity.block==current->position/geometry_.block_tokens;
                })) last_tail=it;
        }
        if (last_tail!=transfers_.end()) poll_transfers(true,last_tail->ticket);
    } else wait_transfers();
    std::vector<BlockIdentity> current_ids;
    if (current) for (std::uint32_t h=0;h<geometry_.heads;++h) {
        const BlockIdentity id{current->layer,h,current->position/geometry_.block_tokens,current->position%geometry_.block_tokens+1};
        current_ids.push_back(id);
        if (slots_.find(id)) actual.push_back(id);
        else if (const auto old=prefix(id)) actual.push_back(old->identity);
    }
    retain(actual);
    std::vector<SlotLease> used;
    std::vector<SlotLease> batched;
    for (const auto & host:host_leases) {
        if (current && host.identity.block==current_ids.front().block) continue;
        auto device = slots_.find(host.identity);
        if (transport_.aliases_host_slots()) {
            if (device && device->slot==host.identity.head*slots_.slots_per_head()+host.slot%host_.slots_per_head()) ++stats_.reused_blocks;
            device=upload(host);
        } else if (device) ++stats_.reused_blocks;
        else if (transport_.batches_uploads()) { batched.push_back(host); continue; }
        else device = upload(host);
        used.push_back(*device);
    }
    if (!batched.empty()) {
        upload_batch(batched,{});
        for (const auto & host:batched) used.push_back(*slots_.find(host.identity));
    }
    for (const auto & id:current_ids) {
        if (const auto ready=slots_.find(id)) { ++stats_.reused_blocks; used.push_back(*ready); }
        else used.push_back(upload_current(id,current_hosts[id.head],*current));
    }
    if (!deferred_upload) wait_transfers();
    const auto tokens = tokens_per_head();
    std::vector<Half> mask(static_cast<std::size_t>(tokens) * query_heads_, Half{0xfc00});
    const auto marker_offset = mask.size();
    if (local_window) mask.resize(marker_offset + static_cast<std::size_t>(tokens) * geometry_.heads, Half{0});
    const auto block_key = [](const SlotLease & lease) {
        return std::make_pair(lease.identity.head, lease.identity.block);
    };
    std::sort(used.begin(), used.end(), [&](const auto & a, const auto & b) { return block_key(a) < block_key(b); });
    const auto group = query_heads_ / geometry_.heads;
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        if (selection[head].positions.empty()) throw std::invalid_argument("empty device attention selection");
        auto found = used.end();
        for (const auto position : selection[head].positions) {
            const auto key = std::make_pair(head, position / geometry_.block_tokens);
            if (found == used.end() || block_key(*found) != key) {
                found = std::lower_bound(used.begin(), used.end(), key,
                    [&](const SlotLease & lease, const auto & wanted) { return block_key(lease) < wanted; });
            }
            if (found == used.end() || block_key(*found) != key || position % geometry_.block_tokens >= found->identity.version) {
                throw std::invalid_argument("device selection has no matching versioned slot");
            }
            const auto row = (found->slot % slots_.slots_per_head()) * geometry_.block_tokens + position % geometry_.block_tokens;
            for (auto q = head * group; q < (head + 1) * group; ++q) mask[static_cast<std::size_t>(q) * tokens + row] = 0;
            const auto distance = selection[head].positions.back() - position;
            if (distance < local_window) {
                mask[marker_offset + static_cast<std::size_t>(head) * tokens + row] =
                    to_half(static_cast<float>(local_window - distance));
            }
        }
    }
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("attention epoch exhausted");
    ++epoch_;
    std::size_t pinned = 0;
    std::vector<SlotLease> host_readers;
    host_readers.reserve(host_leases.size());
    try {
        for (const auto & lease : used) {
            const bool pending = slots_.entry(lease.slot).producer;
            slots_.pin(lease, deferred_upload); ++pinned;
            event(pending ? "attention_pin_waiting" : "attention_pin", lease);
        }
        if (transport_.aliases_host_slots()) for (const auto & host:host_leases) {
            if (std::any_of(used.begin(),used.end(),[&](const auto & device) { return device.identity==host.identity; })) {
                host_.pin(host); host_readers.push_back(host);
            }
        }
        attention_ = std::move(used);
        attention_hosts_=std::move(host_readers);
    } catch (...) {
        for (const auto & host:host_readers) host_.unpin(host);
        for (std::size_t i = 0; i < pinned; ++i) slots_.unpin(used[i]);
        throw;
    }
    return mask;
}
std::vector<std::uint32_t> DeviceCache::current_slots(CurrentBlock current) const {
    std::vector<std::uint32_t> result;
    for (std::uint32_t head=0;head<geometry_.heads;++head) {
        const BlockIdentity id{current.layer, head, current.position/geometry_.block_tokens, current.position%geometry_.block_tokens+1};
        const auto found=std::find_if(attention_.begin(),attention_.end(),[&](const auto & lease) { return lease.identity==id; });
        if (found==attention_.end()) throw std::logic_error("KV archive requested without an attention lease");
        result.push_back(found->slot);
    }
    return result;
}
bool DeviceCache::attention_complete(std::uint64_t epoch) {
    if (epoch != epoch_ || attention_.empty()) { ++stats_.stale_events; return false; }
    for (const auto & lease : attention_) if (slots_.entry(lease.slot).producer) throw std::logic_error("attention completion precedes upload acknowledgment");
    auto completed = std::move(attention_);
    attention_.clear();
    std::exception_ptr error;
    for (const auto & lease : completed) {
        slots_.unpin(lease);
        try { event("attention_complete", lease); } catch (...) { if (!error) error = std::current_exception(); }
    }
    auto host_readers=std::move(attention_hosts_); attention_hosts_.clear();
    for (const auto & lease:host_readers) {
        try { host_.unpin(lease); } catch (...) { if (!error) error=std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
    return true;
}
void DeviceCache::finish() {
    if (!attention_.empty()) throw std::logic_error("finish before device attention completion");
    wait_transfers();
    transport_.drain();
    if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("device trace flush failed"); }
}
DeviceCacheStats DeviceCache::stats() const {
    auto result = stats_;
    result.resident_bytes = transport_.resident_bytes();
    if (transport_.shared_memory()) result.shared_resident_bytes = result.resident_bytes;
    return result;
}
} // namespace solid
