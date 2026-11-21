#pragma once
#include "solid/prefetch.hpp"

namespace solid {
struct CurrentDeviceKv {
    std::uint32_t layer=0;
    std::uint64_t position=0;
    DeviceKvBatch kv;
    bool unarchived = false;
};
struct CurrentBlock {
    std::uint32_t layer=0;
    std::uint64_t position=0;
};
struct DeviceUpload {
    std::uint32_t slot;
    const KvBlock * block;
};
class DeviceTransport {
public:
    virtual ~DeviceTransport() = default;
    // Upload must preserve the source until completion. An exception must drain
    // any copy started by this call before returning control to the caller.
    virtual std::uint64_t upload(std::uint32_t slot, const KvBlock & block) = 0;
    virtual bool batches_uploads() const { return false; }
    // All source blocks remain borrowed until the shared ticket completes.
    virtual std::uint64_t upload_batch(const std::vector<DeviceUpload> &) {
        throw std::logic_error("batched device KV transfer is disabled");
    }
    virtual std::optional<DeviceMemory> current_memory() const { return {}; }
    // Retain prefix rows [0,prefix), upload the missing historical rows, and
    // copy the current device row. The borrowed device source stays live until
    // the ticket completes; exceptions drain any started writers/readers.
    virtual std::uint64_t upload_current(std::uint32_t,const KvBlock *,const CurrentDeviceKv &,std::uint32_t) {
        throw std::logic_error("current device KV transfer is disabled");
    }
    virtual bool complete(std::uint64_t ticket, bool wait) = 0;
    virtual std::optional<std::uint64_t> copy_interval() const { return {}; }
    // Wait for issued copies; ticket receipts remain valid for complete().
    virtual void drain() = 0;
    virtual void wait_batch() { drain(); }
    virtual std::uint64_t resident_bytes() const = 0;
    virtual bool shared_memory() const { return false; }
    virtual bool aliases_host_slots() const { return false; }
    virtual bool binds_host_slots() const { return aliases_host_slots(); }
    virtual Half * host_slot_storage(std::uint32_t) const { return nullptr; }
    virtual std::shared_ptr<void> host_storage_owner() const { return {}; }
};
struct DeviceCacheStats {
    std::uint64_t uploads = 0;
    std::uint64_t upload_bytes = 0;
    std::uint64_t reused_blocks = 0;
    std::uint64_t overwritten_blocks = 0;
    std::uint64_t stale_events = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t shared_resident_bytes = 0, shared_host_copy_bytes = 0;
    std::uint64_t current_device_rows=0,current_device_bytes=0,current_host_bytes=0;
    std::uint64_t refreshed_tails=0,retained_prefix_bytes=0;
};
class DeviceCache {
public:
    DeviceCache(Prefetcher & host, KvGeometry geometry, std::uint32_t query_heads, DeviceTransport & transport,
                const std::filesystem::path & trace = {});
    ~DeviceCache();
    // before_upload runs immediately before a new transfer starts, not after its
    // enqueue returns; the graph boundary may arrive during that enqueue.
    void prefetch_ready(const std::function<void()> & before_upload = {});
    void reap_transfers();
    void wait_transfers();
    void wait_transfers_coalesced();
    void stage(const std::vector<SlotLease> & host_leases);
    void begin_stage(const std::vector<BlockIdentity> & identities,std::optional<CurrentBlock> current = {});
    void stage_ready(const SlotLease & host_lease,const std::shared_ptr<BlockStage> & stage = {});
    void stage_ready_batch(const std::vector<SlotLease> & leases,const std::shared_ptr<BlockStage> & stage);
    std::vector<Half> prepare(const std::vector<SlotLease> & host_leases, const std::vector<HeadSelection> & selection,
                              bool deferred_upload = false,const CurrentDeviceKv * current = nullptr);
    // Called only after the backend confirms attention completion.
    bool attention_complete(std::uint64_t epoch);
    std::uint64_t attention_epoch() const { return epoch_; }
    std::vector<std::uint32_t> current_slots(CurrentBlock current) const;
    bool attention_pending() const { return !attention_.empty(); }
    std::uint32_t tokens_per_head() const;
    void finish();
    DeviceCacheStats stats() const;
    std::optional<SlotLease> find(const BlockIdentity & id) const { return slots_.find(id); }
private:
    struct TransferSlot { std::optional<SlotLease> host; SlotLease device; };
    struct Transfer { std::uint64_t ticket; std::vector<TransferSlot> slots; std::shared_ptr<BlockStage> stage; };
    Prefetcher & host_;
    KvGeometry geometry_;
    std::uint32_t query_heads_;
    DeviceTransport & transport_;
    SlotTable slots_;
    std::vector<Transfer> transfers_;
    std::vector<SlotLease> attention_;
    std::vector<SlotLease> attention_hosts_;
    std::ofstream trace_;
    DeviceCacheStats stats_;
    std::uint64_t epoch_ = 0;
    void poll_transfers(bool wait);
    void retain(const std::vector<BlockIdentity> & identities);
    SlotLease upload(const SlotLease & host,const std::shared_ptr<BlockStage> & stage = {},
                     const std::function<void()> & before_upload = {});
    void upload_batch(const std::vector<SlotLease> & leases,const std::shared_ptr<BlockStage> & stage,
                      const std::function<void()> & before_upload = {});
    SlotLease upload_current(const BlockIdentity & identity,const std::optional<SlotLease> & host,const CurrentDeviceKv & current);
    std::optional<SlotLease> acquire_shared(const BlockIdentity & identity,std::uint32_t slot);
    std::optional<SlotLease> prefix(const BlockIdentity & identity) const;
    void event(const char * name, const SlotLease & lease,const CurrentDeviceKv * current=nullptr,
               std::uint32_t retained=0,std::uint64_t source_generation=0);
};
} // namespace solid
