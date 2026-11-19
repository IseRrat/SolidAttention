#pragma once
#include "solid/async_reader.hpp"
#include "solid/attention.hpp"
#include "solid/block_identity.hpp"
#include "solid/block_stage.hpp"
#include <fstream>
#include <deque>
#include <map>
#include <optional>
#include <tuple>

namespace solid {
enum class SlotState { free, loading, ready, pinned, retiring };
class SlotTable {
public:
    struct Entry {
        SlotState state = SlotState::free;
        std::uint64_t generation = 0;
        BlockIdentity identity;
        std::uint32_t readers = 0;
        bool producer = false;
    };
    SlotTable(std::uint32_t heads, std::uint32_t slots_per_head);
    std::optional<SlotLease> find(const BlockIdentity & id) const;
    std::optional<SlotLease> acquire(const BlockIdentity & id);
    std::optional<SlotLease> acquire_at(const BlockIdentity & id, std::uint32_t slot);
    SlotLease refresh(const SlotLease & previous, const BlockIdentity & identity);
    bool complete(const SlotLease & lease);
    bool retire(const SlotLease & lease);
    void pin(const SlotLease & lease, bool allow_pending = false);
    void unpin(const SlotLease & lease);
    bool matches(const SlotLease & lease) const;
    SlotLease lease(std::uint32_t slot) const;
    const Entry & entry(std::uint32_t slot) const { return slots_.at(slot); }
    std::uint32_t size() const { return static_cast<std::uint32_t>(slots_.size()); }
    std::uint32_t slots_per_head() const { return slots_per_head_; }
private:
    std::vector<Entry> slots_;
    std::uint32_t slots_per_head_;
};
struct PrefetchStats {
    std::uint64_t history_resolutions = 0;
    std::uint64_t formal_resolutions = 0;
    std::uint64_t predicted_blocks = 0;
    std::uint64_t ready_hits = 0;
    std::uint64_t inflight_hits = 0;
    std::uint64_t missing_blocks = 0;
    std::uint64_t unused_blocks = 0;
    std::uint64_t refreshed_blocks = 0;
    std::uint64_t stale_completions = 0;
    std::uint64_t host_bytes = 0;
    std::uint64_t wait_ns = 0;
    std::uint64_t resolve_ns = 0;
    std::uint32_t slots_per_head = 0;
    AsyncIoStats io;
};
struct PredictionCosts {
    std::uint64_t full_read_ns = 1;
    std::uint64_t full_upload_ns = 1;
};
struct StoreReceipt {
    std::optional<std::uint64_t> completed_ns;
};
class Prefetcher {
public:
    using ReadyBlock = std::function<void(const SlotLease &)>;
    using StagePlan = std::function<void(const std::vector<BlockIdentity> &)>;
    using StageFactory = std::function<std::shared_ptr<BlockStage>(const std::vector<BlockWork> &)>;
    Prefetcher(KvStore & store, SparseBudget budget, std::uint32_t slots_per_head,
               std::unique_ptr<AsyncReader> reader, const std::filesystem::path & trace = {});
    ~Prefetcher();
    void bind_shared_blocks(const std::vector<Half *> & addresses, std::shared_ptr<void> owner);
    void begin(std::uint32_t layer, std::uint64_t position, PredictionCosts costs = {});
    std::vector<SlotLease> resolve(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual);
    std::vector<SlotLease> resolve_history(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual,
                                         const StagePlan & plan = {}, const ReadyBlock & ready = {}, const StageFactory & factory = {}, bool keep_tail = false);
    void remember_selection(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual);
    void poll(bool wait = false);
    const KvBlock & block(const SlotLease & lease) const;
    void pin(const SlotLease & lease) { slots_.pin(lease); }
    void unpin(const SlotLease & lease) { slots_.unpin(lease); }
    std::vector<SlotLease> ready() const;
    std::shared_ptr<const StoreReceipt> store_full_layer(std::uint32_t layer);
    bool asynchronous_stores() const { return disk_ != nullptr; }
    void wait_store(std::uint32_t layer);
    void finish_stores();
    void finish();
    PrefetchStats stats() const;
    std::uint32_t slots_per_head() const { return slots_.slots_per_head(); }
    std::optional<std::uint32_t> active_layer() const { return layer_; }
    int completion_handle() const { return reader_->completion_handle(); }
    bool pending_reads() const { return !requests_.empty() || !queued_.empty(); }
    bool pending_stores() const { return !write_queue_.empty(); }
private:
    struct Job { SlotLease lease; KvReadSegment segment; ReadDispatch dispatch; };
    struct ReadStage { SlotLease lease; std::shared_ptr<BlockStage> stage; };
    struct WriteBatch {
        std::vector<KvWriteSegment> segments;
        std::size_t next = 0;
        std::map<std::uint64_t, std::size_t> inflight;
        bool failed = false;
    };
    KvStore & store_;
    SparseBudget budget_;
    SlotTable slots_;
    std::vector<KvBlock> blocks_;
    std::vector<std::size_t> pending_;
    std::vector<ReadStage> stages_;
    std::vector<ReadDispatch> upload_order_;
    std::vector<std::vector<std::vector<std::uint64_t>>> history_;
    std::map<std::uint64_t, Job> requests_;
    std::vector<Job> queued_;
    std::unique_ptr<AsyncReader> reader_;
    FileKvStore * disk_ = nullptr;
    std::vector<std::shared_ptr<StoreReceipt>> store_receipts_;
    std::deque<std::uint32_t> write_queue_;
    std::optional<WriteBatch> write_batch_;
    std::ofstream trace_;
    std::optional<std::uint32_t> layer_;
    std::uint64_t position_ = 0;
    std::uint64_t next_id_ = 1;
    PrefetchStats stats_;
    bool failed_ = false;
    void check() const;
    void issue(const SlotLease & lease, std::uint32_t prefix, ReadDispatch dispatch);
    SlotLease load(const BlockIdentity & identity, ReadDispatch dispatch,const std::shared_ptr<BlockStage> & stage = {});
    void promote(const SlotLease & lease, ReadDispatch dispatch);
    std::optional<SlotLease> prefix(const BlockIdentity & identity) const;
    void read_complete(const SlotLease & lease,const char * proof);
    bool selected(std::uint64_t block) const;
    void retire(const SlotLease & lease);
    void pump();
    void pump_store();
    bool complete_store(const ReadCompletion & completion);
    void event(const char * name, const SlotLease & lease);
    std::vector<SlotLease> resolve_impl(std::uint32_t layer, std::uint64_t position,
                                       const std::vector<HeadSelection> & actual, bool historical,
                                       const StagePlan & plan = {}, const ReadyBlock & ready = {}, const StageFactory & factory = {});
};
} // namespace solid
