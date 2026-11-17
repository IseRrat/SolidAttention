#pragma once
#include "solid/prefetch.hpp"

namespace solid {
class DeviceTransport {
public:
    virtual ~DeviceTransport() = default;
    // Upload must preserve the source until completion. An exception must drain
    // any copy started by this call before returning control to the caller.
    virtual std::uint64_t upload(std::uint32_t slot, const KvBlock & block) = 0;
    virtual bool complete(std::uint64_t ticket, bool wait) = 0;
    // Wait for issued copies; ticket receipts remain valid for complete().
    virtual void drain() = 0;
    virtual void wait_batch() { drain(); }
    virtual std::uint64_t resident_bytes() const = 0;
};
struct DeviceCacheStats {
    std::uint64_t uploads = 0;
    std::uint64_t upload_bytes = 0;
    std::uint64_t reused_blocks = 0;
    std::uint64_t overwritten_blocks = 0;
    std::uint64_t stale_events = 0;
    std::uint64_t resident_bytes = 0;
};
class DeviceCache {
public:
    DeviceCache(Prefetcher & host, KvGeometry geometry, std::uint32_t query_heads, DeviceTransport & transport,
                const std::filesystem::path & trace = {});
    ~DeviceCache();
    void prefetch_ready();
    void wait_transfers();
    void wait_transfers_coalesced();
    void stage(const std::vector<SlotLease> & host_leases);
    void begin_stage(const std::vector<BlockIdentity> & identities);
    void stage_ready(const SlotLease & host_lease,const std::shared_ptr<BlockStage> & stage = {});
    std::vector<Half> prepare(const std::vector<SlotLease> & host_leases, const std::vector<HeadSelection> & selection,
                              bool deferred_upload = false);
    // Called only after the backend confirms attention completion.
    bool attention_complete(std::uint64_t epoch);
    std::uint64_t attention_epoch() const { return epoch_; }
    bool attention_pending() const { return !attention_.empty(); }
    std::uint32_t tokens_per_head() const;
    void finish();
    DeviceCacheStats stats() const;
    std::optional<SlotLease> find(const BlockIdentity & id) const { return slots_.find(id); }
private:
    struct Transfer { std::uint64_t ticket; SlotLease host; SlotLease device; std::shared_ptr<BlockStage> stage; };
    Prefetcher & host_;
    KvGeometry geometry_;
    std::uint32_t query_heads_;
    DeviceTransport & transport_;
    SlotTable slots_;
    std::vector<Transfer> transfers_;
    std::vector<SlotLease> attention_;
    std::ofstream trace_;
    DeviceCacheStats stats_;
    std::uint64_t epoch_ = 0;
    void poll_transfers(bool wait);
    void retain(const std::vector<BlockIdentity> & identities);
    SlotLease upload(const SlotLease & host,const std::shared_ptr<BlockStage> & stage = {});
    void event(const char * name, const SlotLease & lease);
};
} // namespace solid
