// 根据选块历史发起 SSD 预取；正式选择到来后复用命中、补读缺失，并管理主机槽位及写回。
#include "solid/prefetch.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <stdexcept>

namespace solid {
SlotTable::SlotTable(std::uint32_t heads, std::uint32_t per_head) : slots_per_head_(per_head) {
    const auto count = static_cast<std::uint64_t>(heads) * per_head;
    if (!heads || !per_head || count > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("invalid slot capacity");
    slots_.resize(static_cast<std::size_t>(count));
}
bool SlotTable::matches(const SlotLease & lease) const {
    if (lease.slot >= slots_.size()) return false;
    const auto & slot = slots_[lease.slot];
    return slot.state != SlotState::free && slot.generation == lease.generation && slot.identity == lease.identity;
}
SlotLease SlotTable::lease(std::uint32_t index) const {
    const auto & slot = slots_.at(index);
    return {index, slot.generation, slot.identity};
}
std::optional<SlotLease> SlotTable::find(const BlockIdentity & id) const {
    if (id.head >= slots_.size() / slots_per_head_) throw std::out_of_range("slot head");
    for (auto i = id.head * slots_per_head_; i < (id.head + 1) * slots_per_head_; ++i) {
        const auto & slot = slots_[i];
        if (slot.state != SlotState::free && slot.state != SlotState::retiring && slot.identity == id) return lease(i);
    }
    return {};
}
std::optional<SlotLease> SlotTable::acquire(const BlockIdentity & id) {
    if (find(id)) throw std::logic_error("duplicate slot identity");
    for (int pass = 0; pass < 2; ++pass) for (auto i = id.head * slots_per_head_; i < (id.head + 1) * slots_per_head_; ++i) {
        auto & slot = slots_[i];
        if (slot.state == SlotState::free && (pass || slot.generation > 0)) {
            if (slot.generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("slot generation exhausted");
            slot.identity = id;
            ++slot.generation;
            slot.producer = true;
            slot.readers = 0;
            slot.state = SlotState::loading;
            return lease(i);
        }
    }
    return {};
}
std::optional<SlotLease> SlotTable::acquire_at(const BlockIdentity & id,std::uint32_t index) {
    if (index>=slots_.size() || index/slots_per_head_!=id.head || find(id)) throw std::invalid_argument("invalid shared slot identity");
    auto & slot=slots_[index];
    if (slot.state!=SlotState::free) return {};
    if (slot.generation==UINT64_MAX) throw std::overflow_error("slot generation exhausted");
    slot.identity=id; ++slot.generation; slot.producer=true; slot.readers=0; slot.state=SlotState::loading;
    return lease(index);
}
SlotLease SlotTable::refresh(const SlotLease & old, const BlockIdentity & id) {
    if (!matches(old) || slots_[old.slot].state != SlotState::ready || id.layer != old.identity.layer ||
        id.head != old.identity.head || id.block != old.identity.block || id.version <= old.identity.version || find(id)) {
        throw std::logic_error("slot refresh requires a completed, unpinned immutable prefix");
    }
    auto & slot = slots_[old.slot];
    if (slot.generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("slot generation exhausted");
    ++slot.generation;
    slot.identity = id;
    slot.producer = true;
    slot.state = SlotState::loading;
    return lease(old.slot);
}
bool SlotTable::complete(const SlotLease & lease) {
    if (!matches(lease)) return false;
    auto & slot = slots_[lease.slot];
    if (!slot.producer) return false;
    slot.producer = false;
    if (slot.state == SlotState::retiring) {
        if (slot.readers == 0) slot.state = SlotState::free;
    } else slot.state = slot.readers ? SlotState::pinned : SlotState::ready;
    return true;
}
bool SlotTable::retire(const SlotLease & lease) {
    if (!matches(lease)) return false;
    auto & slot = slots_[lease.slot];
    slot.state = (!slot.producer && slot.readers == 0) ? SlotState::free : SlotState::retiring;
    return slot.state == SlotState::free;
}
void SlotTable::pin(const SlotLease & lease, bool allow_pending) {
    if (!matches(lease)) throw std::logic_error("pin of stale slot lease");
    auto & slot = slots_[lease.slot];
    if ((slot.state != SlotState::ready && slot.state != SlotState::pinned && !(allow_pending && slot.state == SlotState::loading)) ||
        slot.readers == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("slot is not ready for a reader");
    }
    ++slot.readers;
    slot.state = slot.producer ? SlotState::loading : SlotState::pinned;
}
void SlotTable::unpin(const SlotLease & lease) {
    if (!matches(lease) || slots_[lease.slot].readers == 0) throw std::logic_error("unpin of stale or unowned slot");
    auto & slot = slots_[lease.slot];
    if (--slot.readers == 0) {
        if (slot.state == SlotState::retiring) { if (!slot.producer) slot.state = SlotState::free; }
        else slot.state = slot.producer ? SlotState::loading : SlotState::ready;
    }
}
Prefetcher::Prefetcher(KvStore & store, SparseBudget budget, std::uint32_t per_head,
                       std::unique_ptr<AsyncReader> reader, const std::filesystem::path & trace, bool prefetch_ahead) :
    store_(store), budget_(budget), slots_(store.geometry().heads, per_head), blocks_(slots_.size()), pending_(slots_.size()),stages_(slots_.size()),
    upload_order_(slots_.size()),
    history_(store.geometry().layers, std::vector<std::vector<std::uint64_t>>(store.geometry().heads)), reader_(std::move(reader)),
    store_receipts_(store.geometry().layers), prefetch_ahead_(prefetch_ahead) {
    stats_.ahead_enabled = prefetch_ahead_;
    budget.validate(store.geometry());
    if (!reader_) throw std::invalid_argument("prefetch reader is required");
    if (reader_->supports_write() && store_.native_handle() >= 0) disk_ = dynamic_cast<FileKvStore *>(&store_);
    const auto & g = store.geometry();
    for (auto & block : blocks_) block.data.reserve(static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension);
    for (const auto & block : blocks_) stats_.host_bytes += block.data.capacity() * sizeof(Half);
    stats_.slots_per_head = per_head;
    if (!trace.empty()) {
        if (std::filesystem::exists(trace)) throw std::runtime_error("prefetch trace already exists");
        trace_.open(trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create prefetch trace");
    }
}
Prefetcher::~Prefetcher() {
    try { finish(); }
    catch (...) {
        try {
            reader_->drain();
            if (write_batch_) disk_->complete_write(write_queue_.front(), false);
        } catch (...) { std::terminate(); }
    }
}
void Prefetcher::bind_shared_blocks(const std::vector<Half *> & addresses,std::shared_ptr<void> owner,bool defer_prediction_slots) {
    check();
    if (layer_ || pending_reads() || addresses.size()!=blocks_.size() || !owner ||
        std::any_of(addresses.begin(),addresses.end(),[](const auto * p) { return !p; })) {
        throw std::invalid_argument("shared KV binding requires an unused prefetch cache");
    }
    const auto elements=static_cast<std::size_t>(store_.geometry().block_tokens)*2*store_.geometry().dimension;
    for (std::size_t i=0;i<blocks_.size();++i) blocks_[i].bind_shared(addresses[i],elements,owner);
    defer_prediction_slots_=defer_prediction_slots;
    stats_.host_bytes=0; // The shared bank is counted by the device transport.
}
void Prefetcher::check() const { if (failed_) throw std::runtime_error("prefetch session failed"); }
std::shared_ptr<const StoreReceipt> Prefetcher::store_full_layer(std::uint32_t layer) {
    check();
    if (!disk_) { store_.flush_full_layer(layer); return {}; }
    if (auto receipt = store_receipts_.at(layer)) return receipt;
    if (!disk_->write_buffer_full(layer)) return {};
    auto receipt = std::make_shared<StoreReceipt>();
    write_queue_.push_back(layer);
    store_receipts_[layer] = receipt;
    try { pump(); }
    catch (...) { failed_ = true; throw; }
    return receipt;
}
void Prefetcher::wait_store(std::uint32_t layer) {
    while (store_receipts_.at(layer)) { check(); poll(true); }
}
void Prefetcher::finish_stores() {
    while (pending_stores()) { check(); poll(true); }
}
void Prefetcher::pump_store() {
    if (failed_ || write_queue_.empty()) return;
    if (!write_batch_) write_batch_.emplace(WriteBatch{disk_->begin_write(write_queue_.front()), 0, {}});
    auto & batch = *write_batch_;
    if (trace::enabled() && !batch.phase)
        batch.phase = std::make_shared<trace::Batch>("CPU->SSD Store", write_queue_.front(), disk_->length(write_queue_.front()) - 1);
    while (batch.next < batch.segments.size()) {
        if (next_id_ >= (std::uint64_t{1} << 63)) throw std::overflow_error("write request identity exhausted");
        const auto & segment = batch.segments[batch.next];
        const auto it = batch.inflight.emplace(next_id_, segment.bytes).first;
        if (!reader_->submit_write(next_id_, disk_->native_handle(), segment.source, segment.bytes, segment.file_offset)) {
            batch.inflight.erase(it);
            break;
        }
        if (batch.phase) batch.phase->submitted();
        ++next_id_;
        ++batch.next;
    }
}
bool Prefetcher::complete_store(const ReadCompletion & completion) {
    if (!write_batch_) return false;
    auto & batch = *write_batch_;
    const auto found = batch.inflight.find(completion.id);
    if (found == batch.inflight.end()) return false;
    const auto layer = write_queue_.front();
    const auto bytes = found->second;
    batch.inflight.erase(found);
    if (batch.failed) return true;
    if (completion.result < 0 || static_cast<std::uint64_t>(completion.result) != bytes) {
        failed_ = true;
        batch.failed = true;
        throw std::runtime_error("async KV write failed: layer=" + std::to_string(layer) + " result=" + std::to_string(completion.result));
    }
    if (!failed_ && batch.next == batch.segments.size() && batch.inflight.empty()) {
        disk_->complete_write(layer, true);
        store_receipts_[layer]->complete = true;
        store_receipts_[layer].reset();
        write_queue_.pop_front();
        write_batch_.reset();
    }
    return true;
}
void Prefetcher::event(const char * name, const SlotLease & lease) {
    if (!trace_.is_open()) return;
    trace_ << "{\"event\":\"" << name << "\",\"timestamp_ns\":"
           << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
           << ",\"position\":" << position_ << ",\"layer\":" << lease.identity.layer
           << ",\"kv_head\":" << lease.identity.head << ",\"block\":" << lease.identity.block
           << ",\"version\":" << lease.identity.version << ",\"slot\":" << lease.slot << ",\"generation\":" << lease.generation << "}\n";
    if (!trace_) { failed_ = true; throw std::runtime_error("prefetch trace write failed"); }
}
void Prefetcher::pump() {
    pump_predictions();
    while (!queued_.empty()) {
        const auto next=std::min_element(queued_.begin(),queued_.end(),[](const Job & a,const Job & b) { return a.dispatch.before(b.dispatch); });
        const auto job = *next;
        if (next_id_ >= (std::uint64_t{1} << 63)) throw std::overflow_error("read request identity exhausted");
        auto * destination=reinterpret_cast<unsigned char *>(blocks_[job.lease.slot].storage())+job.segment.buffer_offset;
        const auto request=requests_.emplace(next_id_,ReadRequest{job,
            store_.prepare_read(destination,job.segment.bytes,job.segment.file_offset)}).first;
        try {
            const auto & io=request->second.read;
            if (!reader_->submit_scheduled(next_id_,io.fd,io.buffer,io.bytes,io.offset,job.dispatch)) {
                requests_.erase(request);
                break;
            }
            if (job.phase) job.phase->submitted();
        } catch (...) {
            requests_.erase(request);
            throw;
        }
        ++next_id_;
        queued_.erase(next);
        const auto & observer=stages_[job.lease.slot];
        if (observer.stage && observer.lease==job.lease) observer.stage->read_started(job.lease.identity);
        event("read_submit",job.lease);
    }
    try { pump_store(); }
    catch (...) { failed_ = true; throw; }
    reader_->flush();
}
void Prefetcher::issue(const SlotLease & lease, std::uint32_t prefix, ReadDispatch dispatch,bool dispatch_now,
                      const std::shared_ptr<trace::Batch> & phase) {
    const auto & id = lease.identity;
    const auto segments = store_.plan_read(id.layer, id.head, id.block, blocks_[lease.slot], prefix);
    if (blocks_[lease.slot].version != id.version) throw std::logic_error("read plan version changed before submission");
    pending_[lease.slot] = segments.size();
    for (const auto & segment : segments) queued_.push_back({lease, segment, dispatch, phase ? phase : read_phase_});
    if (segments.empty()) { slots_.complete(lease); read_complete(lease,"host_forwarded"); event("ready", lease); }
    if (dispatch_now) pump();
}
void Prefetcher::pump_predictions() {
    if (failed_) return;
    for (auto it=predictions_.begin();it!=predictions_.end();) {
        auto lease=slots_.find(it->identity);
        if (!lease) {
            lease=slots_.acquire(it->identity);
            if (!lease) { ++it; continue; }
            stages_[lease->slot]={*lease,{}};
            upload_order_[lease->slot]=it->copy;
            issue(*lease,0,it->read,false,it->phase);
            event("predict",*lease);
        } else upload_order_[lease->slot]=it->copy;
        ++stats_.predicted_blocks;
        it=predictions_.erase(it);
    }
}
SlotLease Prefetcher::load(const BlockIdentity & id, ReadDispatch dispatch,const std::shared_ptr<BlockStage> & stage) {
    auto lease = slots_.acquire(id);
    while (!lease) {
        if (requests_.empty() && queued_.empty()) throw std::runtime_error("host slot capacity held by readers or required blocks");
        poll(true);
        lease = slots_.acquire(id);
    }
    issue_missing(*lease, dispatch, stage, true);
    return *lease;
}
void Prefetcher::issue_missing(const SlotLease & lease, ReadDispatch dispatch, const std::shared_ptr<BlockStage> & stage, bool dispatch_now) {
    stages_[lease.slot] = {lease, stage};
    upload_order_[lease.slot] = dispatch;
    issue(lease, 0, dispatch, dispatch_now);
    ++stats_.missing_blocks;
    event("missing", lease);
}
void Prefetcher::promote(const SlotLease & lease, ReadDispatch dispatch) {
    for (auto & job:queued_) if (job.lease==lease && dispatch.before(job.dispatch)) job.dispatch=dispatch;
    for (auto & item:requests_) if (item.second.lease==lease && dispatch.before(item.second.dispatch)) {
        reader_->schedule(item.first,dispatch); item.second.dispatch=dispatch;
    }
}
std::optional<SlotLease> Prefetcher::prefix(const BlockIdentity & id) const {
    std::optional<SlotLease> result;
    const auto end = (id.head+1) * slots_.slots_per_head();
    for (auto i = id.head * slots_.slots_per_head(); i < end; ++i) {
        const auto & entry = slots_.entry(i);
        const auto & old = entry.identity;
        if (entry.state != SlotState::free && entry.state != SlotState::retiring &&
            old.layer == id.layer && old.block == id.block && old.version < id.version &&
            (!result || old.version > result->identity.version)) result = slots_.lease(i);
    }
    return result;
}
void Prefetcher::read_complete(const SlotLease & lease,const char * proof) {
    auto & observer=stages_[lease.slot];
    if (observer.stage && observer.lease==lease) {
        auto stage=std::move(observer.stage);
        stage->read_complete(lease.identity,proof);
    }
}
bool Prefetcher::selected(std::uint64_t block) const {
    const auto & g=store_.geometry();
    const auto first=block*g.block_tokens;
    const auto local_first = budget_.recent_start(g, position_);
    return position_+1>budget_.total && first<local_first && first+g.block_tokens>budget_.initial;
}
void Prefetcher::retire(const SlotLease & lease) {
    if (!slots_.matches(lease)) return;
    if (slots_.entry(lease.slot).state != SlotState::retiring) event("unused", lease);
    if (slots_.retire(lease)) return;
    for (auto it = queued_.begin(); it != queued_.end();) {
        if (it->lease == lease) { --pending_[lease.slot]; it = queued_.erase(it); }
        else ++it;
    }
    for (const auto & pair : requests_) if (pair.second.lease == lease) reader_->cancel(pair.first);
    if (pending_[lease.slot] == 0) slots_.complete(lease);
    reader_->flush();
}
void Prefetcher::poll(bool wait,const ReadyBlock & ready) {
    reader_->acknowledge_notifications();
    pump();
    ReadCompletion completion{};
    bool first = true;
    while (reader_->poll(completion, wait && first)) {
        first = false;
        if (complete_store(completion)) { pump(); continue; }
        const auto found = requests_.find(completion.id);
        if (found == requests_.end()) { ++stats_.stale_completions; continue; }
        const auto job = std::move(found->second);
        requests_.erase(found);
        if (!slots_.matches(job.lease)) { ++stats_.stale_completions; continue; }
        const bool retiring = slots_.entry(job.lease.slot).state == SlotState::retiring;
        if (!retiring) completion.result=job.read.finish(completion.result);
        if (pending_[job.lease.slot] == 0) throw std::logic_error("duplicate read completion");
        --pending_[job.lease.slot];
        if (!retiring && (completion.result < 0 || static_cast<std::uint64_t>(completion.result) != job.segment.bytes)) {
            slots_.retire(job.lease);
            if (pending_[job.lease.slot] == 0) slots_.complete(job.lease);
            failed_ = true;
            throw std::runtime_error("async KV read failed: layer=" + std::to_string(job.lease.identity.layer) +
                " head=" + std::to_string(job.lease.identity.head) + " block=" + std::to_string(job.lease.identity.block) +
                " slot=" + std::to_string(job.lease.slot) + " generation=" + std::to_string(job.lease.generation) +
                " result=" + std::to_string(completion.result));
        }
        if (pending_[job.lease.slot] == 0) {
            slots_.complete(job.lease);
            if (!retiring) read_complete(job.lease,"original_read_complete");
            event(retiring ? "retired" : "ready", job.lease);
            // A complete KV block can start H2D before another queued CQE is
            // consumed. A split K/V block is delivered only after both segments.
            if (!retiring && ready) {
                slots_.pin(job.lease);
                try { ready(job.lease); } catch (...) { slots_.unpin(job.lease); throw; }
                slots_.unpin(job.lease);
            }
        }
        pump();
    }
}
void Prefetcher::wait_reads() {
    check();
    while (pending_reads()) poll(true);
}
// 根据该层历史选择准备预测请求；主机槽位与请求数量受工作区容量限制。
void Prefetcher::begin(std::uint32_t layer, std::uint64_t position) {
    check();
    const auto & g = store_.geometry();
    if (layer >= g.layers || position >= g.capacity_tokens || store_.length(layer) != position) throw std::invalid_argument("prefetch begin position mismatch");
    if (layer_ == layer && position_ == position) { poll(); return; }
    trace::BatchContext phase(read_phase_, "SSD->CPU Prefetch", layer, position);
    if (!defer_prediction_slots_) for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_.entry(i).readers) throw std::logic_error("host slots still held by a transfer");
    }
    predictions_.clear();
    layer_ = layer;
    position_ = position;
    std::vector<BlockIdentity> predicted;
    if (prefetch_ahead_) for (std::uint32_t head = 0; head < g.heads; ++head) {
        std::vector<std::uint64_t> ids;
        const auto add = [&](std::uint64_t id) {
            if (id * g.block_tokens < position && std::find(ids.begin(), ids.end(), id) == ids.end() && ids.size() < slots_.slots_per_head()) ids.push_back(id);
        };
        if (position + 1 <= budget_.total) {
            for (std::uint64_t b = 0; b * g.block_tokens < position; ++b) add(b);
        } else {
            for (std::uint64_t p = 0; p < std::min<std::uint64_t>(budget_.initial, position); ++p) add(p / g.block_tokens);
            const auto first = budget_.recent_start(g, position);
            for (auto p = first; p < position; ++p) add(p / g.block_tokens);
            for (const auto id : history_[layer][head]) add(id);
        }
        for (const auto id : ids) predicted.push_back({layer, head, id, std::min<std::uint64_t>(g.block_tokens, position - id * g.block_tokens)});
    }
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto lease = slots_.lease(i);
        if (slots_.entry(i).state != SlotState::free && std::find(predicted.begin(), predicted.end(), lease.identity) == predicted.end()) retire(lease);
    }
    const auto priority=[&](const BlockIdentity & id) {
        const auto & history=history_[layer][id.head];
        return selected(id.block) && std::find(history.begin(),history.end(),id.block)!=history.end() ? ReadPriority::selected : ReadPriority::deterministic;
    };
    std::vector<BlockWork> tasks;
    tasks.reserve(predicted.size());
    for (const auto & id : predicted) tasks.push_back({id, priority(id)});
    const auto plan=plan_block_dispatch(tasks);
    std::vector<Prediction> ordered;
    ordered.reserve(predicted.size());
    for (std::size_t i=0;i<predicted.size();++i)
        ordered.push_back({predicted[i],plan[i].read,plan[i].copy,read_phase_});
    std::stable_sort(ordered.begin(), ordered.end(), [](const Prediction & a, const Prediction & b) {
        return a.read.before(b.read);
    });
    if (defer_prediction_slots_) predictions_=std::move(ordered);
    else for (const auto & job : ordered) {
        auto lease = slots_.find(job.identity);
        if (!lease) {
            lease = load(job.identity, job.read);
            event("predict", *lease);
        }
        upload_order_[lease->slot] = job.copy;
        ++stats_.predicted_blocks;
    }
    pump();
}
std::vector<SlotLease> Prefetcher::resolve(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual) {
    return resolve_impl(layer, position, actual, false);
}
std::vector<SlotLease> Prefetcher::resolve_history(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual,
                                                const StagePlan & plan, const ReadyBlock & ready, const StageFactory & factory,
                                                bool keep_tail, const ReadyBatch & batch, bool admit_before_stage, bool per_completion) {
    auto past = actual;
    for (auto & head : past) {
        if (!head.positions.empty() && head.positions.back() == position) head.positions.pop_back();
        if (keep_tail && position%store_.geometry().block_tokens && (head.positions.empty() || head.positions.back()<position-1)) {
            head.positions.push_back(position-1);
        }
    }
    return resolve_impl(layer, position, past, true, plan, ready, factory, batch, admit_before_stage, per_completion);
}
// 把正式需求和预测结果对齐：保留命中、处理版本变化、补读缺失并通知上传方。
std::vector<SlotLease> Prefetcher::resolve_impl(std::uint32_t layer, std::uint64_t position,
                                            const std::vector<HeadSelection> & actual, bool historical,
                                            const StagePlan & plan, const ReadyBlock & ready,const StageFactory & factory,
                                            const ReadyBatch & batch, bool admit_before_stage, bool per_completion) {
    check();
    const auto & g = store_.geometry();
    if (per_completion && (!ready || batch)) throw std::invalid_argument("per-completion resolve requires an individual ready callback");
    const auto available = position + (historical ? 0 : 1);
    if (layer_ != layer || position_ != position || actual.size() != g.heads || store_.length(layer) != available) {
        throw std::invalid_argument("prefetch resolve identity mismatch");
    }
    trace::BatchContext phase(read_phase_, "SSD->CPU Load", layer, position);
    std::vector<BlockWork> by_identity;
    const auto local_first=position+1-std::min<std::uint64_t>(budget_.local,position+1);
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        const auto & positions = actual[head].positions;
        if ((!historical && positions.empty()) || positions.size() > budget_.total || (!positions.empty() && positions.back() >= available) ||
            !std::is_sorted(positions.begin(), positions.end()) || std::adjacent_find(positions.begin(), positions.end()) != positions.end()) {
            throw std::invalid_argument("invalid actual prefetch selection");
        }
        const auto first=by_identity.size();
        for (const auto p : positions) {
            const auto block=p/g.block_tokens;
            if (by_identity.size()==first || by_identity.back().identity.block!=block) {
                by_identity.push_back({{layer,head,block,std::min<std::uint64_t>(g.block_tokens,available-block*g.block_tokens)}});
            }
            if (position+1>budget_.total && p>=budget_.initial && p<local_first) {
                by_identity.back().priority=ReadPriority::correction_selected;
            }
        }
        if (by_identity.size()-first>slots_.slots_per_head()) throw std::invalid_argument("actual selection exceeds slots_per_head");
    }
    const auto find_block=[&](const BlockIdentity & id) {
        const BlockIdentity first{id.layer,id.head,id.block,0};
        return std::lower_bound(by_identity.begin(),by_identity.end(),first,
            [](const BlockWork & block,const BlockIdentity & identity) { return block.identity<identity; });
    };
    auto work=by_identity;
    std::stable_partition(work.begin(),work.end(),[](const BlockWork & item) {
        return item.priority==ReadPriority::correction_selected;
    });
    std::vector<BlockIdentity> needed;
    needed.reserve(work.size());
    for (const auto & item:work) needed.push_back(item.identity);
    std::shared_ptr<BlockStage> stage;
    if (factory) {
        stage=factory(work);
        if (!stage) throw std::logic_error("empty block stage from factory");
        const auto ordered=stage->read_order();
        auto check_order=ordered,check_needed=needed;
        std::sort(check_order.begin(),check_order.end()); std::sort(check_needed.begin(),check_needed.end());
        if (check_order!=check_needed) throw std::logic_error("block stage changed required membership");
        needed=ordered;
    }
    const auto dispatch=[&](const BlockIdentity & id) {
        return stage?stage->dispatch(id):ReadDispatch{find_block(id)->priority};
    };
    // Formal selection owns admission now. It will issue the required blocks
    // with correction keys, rather than filling freed slots with stale guesses.
    predictions_.clear();
    // Promote all existing required jobs before waiting for any old producer.
    for (const auto & id:needed) {
        const auto lease=slots_.find(id);
        if (!lease) {
            if (const auto old=prefix(id)) promote(*old,dispatch(id));
            continue;
        }
        stages_[lease->slot]={*lease,stage};
        upload_order_[lease->slot] = stage ? stage->copy_dispatch(id) : dispatch(id);
        if (stage) {
            if (slots_.entry(lease->slot).state!=SlotState::loading) read_complete(*lease,"host_resident");
            else if (std::any_of(requests_.begin(),requests_.end(),[&](const auto & item) { return item.second.lease==*lease; })) stage->read_started(id,"inflight_at_plan");
        }
        promote(*lease,dispatch(id));
    }
    if (plan && !admit_before_stage) plan(needed);
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto lease = slots_.lease(i);
        if (slots_.entry(i).state==SlotState::free || slots_.entry(i).state==SlotState::retiring) continue;
        const auto & id=lease.identity;
        const auto match=find_block(id);
        if (match==by_identity.end() || match->identity.layer!=id.layer || match->identity.head!=id.head || match->identity.block!=id.block) {
            ++stats_.unused_blocks;
            retire(lease);
        }
    }
    std::map<BlockIdentity, SlotLease> admitted;
    if (admit_before_stage) {
        // Only start reads into free host slots while earlier uploads may still
        // be using other slots. Refreshing a prefix can wait for those copies.
        for (const auto & id : needed) {
            if (slots_.find(id) || prefix(id)) continue;
            const auto lease = slots_.acquire(id);
            if (!lease) continue;
            admitted.emplace(id, *lease);
            issue_missing(*lease, dispatch(id), stage, false);
        }
        if (!admitted.empty()) pump();
        if (plan) plan(needed);
    }
    std::vector<SlotLease> result;
    for (const auto & id : needed) {
        if (const auto it = admitted.find(id); it != admitted.end()) {
            result.push_back(it->second);
            continue;
        }
        auto lease = slots_.find(id);
        if (lease) {
            if (slots_.entry(lease->slot).state == SlotState::loading) { ++stats_.inflight_hits; event("inflight_hit", *lease); }
            else { ++stats_.ready_hits; event("hit", *lease); }
        } else {
            if (const auto old = prefix(id)) {
                while (slots_.entry(old->slot).state == SlotState::loading) poll(true);
                const auto count = blocks_[old->slot].valid_tokens;
                lease = slots_.refresh(*old, id);
                stages_[lease->slot]={*lease,stage};
                upload_order_[lease->slot] = dispatch(id);
                issue(*lease, count,dispatch(id));
                ++stats_.refreshed_blocks;
                event("refresh", *lease);
            } else {
                lease = load(id,dispatch(id),stage);
            }
        }
        result.push_back(*lease);
    }
    std::vector<bool> delivered(result.size(),false);
    std::map<BlockIdentity,std::size_t> result_indices;
    if (stage || per_completion) for (std::size_t i=0;i<result.size();++i) result_indices.emplace(result[i].identity,i);
    std::size_t remaining=result.size();
    std::vector<SlotLease> ready_wave;
    if (batch) ready_wave.reserve(remaining);
    const auto deliver=[&](std::size_t i) {
        if (delivered[i]) return;
        if (batch) ready_wave.push_back(result[i]);
        if (ready) {
            slots_.pin(result[i]);
            try { ready(result[i]); } catch (...) { slots_.unpin(result[i]); throw; }
            slots_.unpin(result[i]);
        }
        delivered[i]=true; --remaining;
    };
    const ReadyBlock on_completion=per_completion ? ReadyBlock{[&](const SlotLease & lease) {
        const auto found=result_indices.find(lease.identity);
        if (found!=result_indices.end()) {
            if (!(result[found->second]==lease)) throw std::logic_error("completed block lease changed during resolve");
            deliver(found->second);
        }
    }} : ReadyBlock{};
    while (remaining) {
        std::vector<std::size_t> candidates;
        if (stage) {
            for (const auto & id:stage->copy_order()) {
                const auto found=result_indices.find(id);
                if (found==result_indices.end()) throw std::logic_error("ready block outside required stage");
                candidates.push_back(found->second);
            }
        } else for (std::size_t i=0;i<result.size();++i) candidates.push_back(i);
        for (const auto i:candidates) if (!delivered[i] && slots_.entry(result[i].slot).state!=SlotState::loading) {
            deliver(i);
        }
        if (!ready_wave.empty()) {
            batch(ready_wave);
            ready_wave.clear();
        }
        if (remaining) poll(true,on_completion);
    }
    if (!historical) remember_selection(layer,position,actual);
    for (const auto & lease : result) slots_.pin(lease);
    if (historical) ++stats_.history_resolutions;
    return result;
}
void Prefetcher::remember_selection(std::uint32_t layer,std::uint64_t position,const std::vector<HeadSelection> & actual) {
    if (layer_!=layer || position_!=position || actual.size()!=store_.geometry().heads) throw std::invalid_argument("selection history frame mismatch");
    const auto local_first=position+1-std::min<std::uint64_t>(budget_.local,position+1);
    for (std::size_t head=0;head<actual.size();++head) {
        auto & history=history_[layer][head];
        history.clear();
        for (const auto p:actual[head].positions) if (p>=budget_.initial && p<local_first) {
            const auto id=p/store_.geometry().block_tokens;
            if (history.empty() || history.back()!=id) history.push_back(id);
        }
    }
    ++stats_.formal_resolutions;
}
const KvBlock & Prefetcher::block(const SlotLease & lease) const {
    if (!slots_.matches(lease) || (slots_.entry(lease.slot).state != SlotState::ready && slots_.entry(lease.slot).state != SlotState::pinned)) {
        throw std::logic_error("read of unavailable or stale host slot");
    }
    return blocks_[lease.slot];
}
std::vector<SlotLease> Prefetcher::ready() const {
    std::vector<SlotLease> result;
    for (std::uint32_t i = 0; i < slots_.size(); ++i) if (slots_.entry(i).state == SlotState::ready) result.push_back(slots_.lease(i));
    std::stable_sort(result.begin(), result.end(), [&](const SlotLease & a, const SlotLease & b) {
        return upload_order_[a.slot].before(upload_order_[b.slot]);
    });
    return result;
}
void Prefetcher::finish() {
    predictions_.clear();
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_.entry(i).readers) throw std::logic_error("finish with pinned host slot");
        if (slots_.entry(i).state != SlotState::free) retire(slots_.lease(i));
    }
    while (!requests_.empty()) poll(true);
    finish_stores();
    reader_->drain();
    reader_->rethrow_failure();
    if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("prefetch trace flush failed"); }
}
PrefetchStats Prefetcher::stats() const { auto result = stats_; result.io = reader_->stats(); return result; }
} // namespace solid
