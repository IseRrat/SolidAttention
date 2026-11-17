#include "solid/device_cache.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace solid {
DeviceCache::DeviceCache(Prefetcher & host, KvGeometry geometry, std::uint32_t query_heads, DeviceTransport & transport,
                         const std::filesystem::path & trace) :
    host_(host), geometry_(geometry), query_heads_(query_heads), transport_(transport), slots_(geometry.heads, host.slots_per_head()) {
    geometry.validate();
    if (geometry.layout != KvLayout::interleaved || !query_heads || query_heads % geometry.heads != 0) throw std::invalid_argument("invalid device cache geometry");
    (void) tokens_per_head();
    if (!trace.empty()) {
        if (std::filesystem::exists(trace)) throw std::runtime_error("device trace already exists");
        trace_.open(trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create device trace");
    }
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
void DeviceCache::event(const char * name, const SlotLease & lease) {
    if (!trace_.is_open()) return;
    trace_ << "{\"event\":\"" << name << "\",\"timestamp_ns\":"
           << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
           << ",\"epoch\":" << epoch_ << ",\"layer\":" << lease.identity.layer
           << ",\"kv_head\":" << lease.identity.head << ",\"block\":" << lease.identity.block << ",\"version\":" << lease.identity.version
           << ",\"slot\":" << lease.slot << ",\"generation\":" << lease.generation << "}\n";
    if (!trace_) throw std::runtime_error("device residency trace write failed");
}
void DeviceCache::poll_transfers(bool wait) {
    std::exception_ptr error;
    for (auto it = transfers_.begin(); it != transfers_.end();) {
        if (!transport_.complete(it->ticket, wait)) { ++it; continue; }
        host_.unpin(it->host);
        const auto lease = it->device;
        const bool current = slots_.complete(lease);
        if (!current) ++stats_.stale_events;
        it = transfers_.erase(it);
        if (current) { try { event("upload_complete", lease); } catch (...) { if (!error) error = std::current_exception(); } }
    }
    if (error) std::rethrow_exception(error);
}
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
SlotLease DeviceCache::upload(const SlotLease & host) {
    if (const auto existing = slots_.find(host.identity)) return *existing;
    const auto device = slots_.acquire(host.identity);
    if (!device) throw std::runtime_error("device slot capacity exceeded");
    host_.pin(host);
    std::uint64_t ticket = 0;
    try {
        const auto & block = host_.block(host);
        ticket = transport_.upload(device->slot, block);
        transfers_.push_back({ticket, host, *device});
        ++stats_.uploads;
        stats_.upload_bytes += block.data.size() * sizeof(Half);
        event("upload", *device);
    } catch (...) {
        // Even allocation/trace errors after upload must retain the host lease
        // until the transport has released every source buffer.
        transport_.drain();
        if (!transfers_.empty() && transfers_.back().device == *device) transfers_.pop_back();
        if (ticket) (void) transport_.complete(ticket, true);
        host_.unpin(host);
        slots_.retire(*device);
        slots_.complete(*device);
        throw;
    }
    return *device;
}
void DeviceCache::prefetch_ready() {
    if (!attention_.empty()) throw std::logic_error("prefetch upload while device attention still owns slots");
    poll_transfers(false);
    const auto ready = host_.ready();
    std::vector<BlockIdentity> identities;
    for (const auto & lease : ready) identities.push_back(lease.identity);
    // Transfers from an earlier prefetch pass still own both endpoints.
    for (const auto & transfer : transfers_) identities.push_back(transfer.device.identity);
    retain(identities);
    for (const auto & lease : ready) (void) upload(lease);
}
void DeviceCache::stage(const std::vector<SlotLease> & host_leases) {
    if (!attention_.empty()) throw std::logic_error("stage before previous attention completed");
    wait_transfers();
    std::vector<BlockIdentity> actual;
    for (const auto & lease : host_leases) actual.push_back(lease.identity);
    retain(actual);
    for (const auto & lease : host_leases) (void) upload(lease);
}
std::vector<Half> DeviceCache::prepare(const std::vector<SlotLease> & host_leases, const std::vector<HeadSelection> & selection, bool deferred_upload) {
    if (!attention_.empty()) throw std::logic_error("previous device attention has not completed");
    if (selection.size() != geometry_.heads) throw std::invalid_argument("device selection head count");
    wait_transfers();
    std::vector<BlockIdentity> actual;
    for (const auto & lease : host_leases) actual.push_back(lease.identity);
    retain(actual);
    std::vector<SlotLease> used;
    for (const auto & host : host_leases) {
        auto device = slots_.find(host.identity);
        if (device) ++stats_.reused_blocks;
        else device = upload(host);
        used.push_back(*device);
    }
    if (!deferred_upload) wait_transfers();
    const auto tokens = tokens_per_head();
    std::vector<Half> mask(static_cast<std::size_t>(tokens) * query_heads_, Half{0xfc00});
    const auto group = query_heads_ / geometry_.heads;
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        if (selection[head].positions.empty()) throw std::invalid_argument("empty device attention selection");
        for (const auto position : selection[head].positions) {
            const auto found = std::find_if(used.begin(), used.end(), [&](const SlotLease & lease) {
                return lease.identity.head == head && lease.identity.block == position / geometry_.block_tokens;
            });
            if (found == used.end() || position % geometry_.block_tokens >= found->identity.version) throw std::invalid_argument("device selection has no matching versioned slot");
            const auto row = (found->slot % slots_.slots_per_head()) * geometry_.block_tokens + position % geometry_.block_tokens;
            for (auto q = head * group; q < (head + 1) * group; ++q) mask[static_cast<std::size_t>(q) * tokens + row] = 0;
        }
    }
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("attention epoch exhausted");
    ++epoch_;
    std::size_t pinned = 0;
    try {
        for (const auto & lease : used) {
            const bool pending = slots_.entry(lease.slot).producer;
            slots_.pin(lease, deferred_upload); ++pinned;
            event(pending ? "attention_pin_waiting" : "attention_pin", lease);
        }
        attention_ = std::move(used);
    } catch (...) { for (std::size_t i = 0; i < pinned; ++i) slots_.unpin(used[i]); throw; }
    return mask;
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
    if (error) std::rethrow_exception(error);
    return true;
}
void DeviceCache::finish() {
    if (!attention_.empty()) throw std::logic_error("finish before device attention completion");
    wait_transfers();
    transport_.drain();
    if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("device trace flush failed"); }
}
DeviceCacheStats DeviceCache::stats() const { auto result = stats_; result.resident_bytes = transport_.resident_bytes(); return result; }
} // namespace solid
