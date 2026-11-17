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
void SlotTable::pin(const SlotLease & lease) {
    if (!matches(lease)) throw std::logic_error("pin of stale slot lease");
    auto & slot = slots_[lease.slot];
    if ((slot.state != SlotState::ready && slot.state != SlotState::pinned) || slot.readers == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("slot is not ready for a reader");
    }
    ++slot.readers;
    slot.state = SlotState::pinned;
}
void SlotTable::unpin(const SlotLease & lease) {
    if (!matches(lease) || slots_[lease.slot].readers == 0) throw std::logic_error("unpin of stale or unowned slot");
    auto & slot = slots_[lease.slot];
    if (--slot.readers == 0) {
        if (slot.state == SlotState::retiring) { if (!slot.producer) slot.state = SlotState::free; }
        else slot.state = SlotState::ready;
    }
}
Prefetcher::Prefetcher(KvStore & store, SparseBudget budget, std::uint32_t per_head,
                       std::unique_ptr<AsyncReader> reader, const std::filesystem::path & trace) :
    store_(store), budget_(budget), slots_(store.geometry().heads, per_head), blocks_(slots_.size()), pending_(slots_.size()),
    history_(store.geometry().layers, std::vector<std::vector<std::uint64_t>>(store.geometry().heads)), reader_(std::move(reader)) {
    budget.validate(store.geometry());
    if (!reader_) throw std::invalid_argument("prefetch reader is required");
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
    catch (...) { try { reader_->drain(); } catch (...) { std::terminate(); } }
}
void Prefetcher::check() const { if (failed_) throw std::runtime_error("prefetch session failed"); }
void Prefetcher::event(const char * name, const SlotLease & lease) {
    if (!trace_.is_open()) return;
    trace_ << "{\"event\":\"" << name << "\",\"position\":" << position_ << ",\"layer\":" << lease.identity.layer
           << ",\"kv_head\":" << lease.identity.head << ",\"block\":" << lease.identity.block
           << ",\"version\":" << lease.identity.version << ",\"slot\":" << lease.slot << ",\"generation\":" << lease.generation << "}\n";
    if (!trace_) { failed_ = true; throw std::runtime_error("prefetch trace write failed"); }
}
void Prefetcher::pump() {
    while (!queued_.empty()) {
        const auto job = queued_.front();
        if (next_id_ >= (std::uint64_t{1} << 63)) throw std::overflow_error("read request identity exhausted");
        if (!reader_->submit(next_id_, store_.native_handle(), reinterpret_cast<unsigned char *>(blocks_[job.lease.slot].data.data()) +
                             job.segment.buffer_offset, job.segment.bytes, job.segment.file_offset)) break;
        requests_.emplace(next_id_++, job);
        queued_.erase(queued_.begin());
    }
    reader_->flush();
}
void Prefetcher::issue(const SlotLease & lease, std::uint32_t prefix) {
    const auto & id = lease.identity;
    const auto segments = store_.plan_read(id.layer, id.head, id.block, blocks_[lease.slot], prefix);
    if (blocks_[lease.slot].version != id.version) throw std::logic_error("read plan version changed before submission");
    pending_[lease.slot] = segments.size();
    for (const auto & segment : segments) queued_.push_back({lease, segment});
    if (segments.empty()) { slots_.complete(lease); event("ready", lease); }
    pump();
}
SlotLease Prefetcher::load(const BlockIdentity & id) {
    auto lease = slots_.acquire(id);
    while (!lease) {
        if (requests_.empty() && queued_.empty()) throw std::runtime_error("host slot capacity held by readers or required blocks");
        poll(true);
        lease = slots_.acquire(id);
    }
    issue(*lease);
    return *lease;
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
void Prefetcher::poll(bool wait) {
    const auto start = std::chrono::steady_clock::now();
    pump();
    ReadCompletion completion{};
    bool first = true;
    while (reader_->poll(completion, wait && first)) {
        first = false;
        const auto found = requests_.find(completion.id);
        if (found == requests_.end()) { ++stats_.stale_completions; continue; }
        const auto job = found->second;
        requests_.erase(found);
        if (!slots_.matches(job.lease)) { ++stats_.stale_completions; continue; }
        const bool retiring = slots_.entry(job.lease.slot).state == SlotState::retiring;
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
            event(retiring ? "retired" : "ready", job.lease);
        }
        pump();
    }
    if (wait) stats_.wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}
void Prefetcher::begin(std::uint32_t layer, std::uint64_t position) {
    check();
    const auto & g = store_.geometry();
    if (layer >= g.layers || position >= g.capacity_tokens || store_.length(layer) != position) throw std::invalid_argument("prefetch begin position mismatch");
    if (layer_ == layer && position_ == position) { poll(); return; }
    for (std::uint32_t i = 0; i < slots_.size(); ++i) if (slots_.entry(i).readers) throw std::logic_error("host slots still held by a transfer");
    layer_ = layer;
    position_ = position;
    std::vector<BlockIdentity> predicted;
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        std::vector<std::uint64_t> ids;
        const auto add = [&](std::uint64_t id) {
            if (id * g.block_tokens < position && std::find(ids.begin(), ids.end(), id) == ids.end() && ids.size() < slots_.slots_per_head()) ids.push_back(id);
        };
        if (position + 1 <= budget_.total) {
            for (std::uint64_t b = 0; b * g.block_tokens < position; ++b) add(b);
        } else {
            for (std::uint64_t p = 0; p < std::min<std::uint64_t>(budget_.initial, position); ++p) add(p / g.block_tokens);
            const auto first = position + 1 - std::min<std::uint64_t>(budget_.local, position + 1);
            for (auto p = first; p < position; ++p) add(p / g.block_tokens);
            for (const auto id : history_[layer][head]) add(id);
        }
        for (const auto id : ids) predicted.push_back({layer, head, id, std::min<std::uint64_t>(g.block_tokens, position - id * g.block_tokens)});
    }
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto lease = slots_.lease(i);
        if (slots_.entry(i).state != SlotState::free && std::find(predicted.begin(), predicted.end(), lease.identity) == predicted.end()) retire(lease);
    }
    for (const auto & id : predicted) {
        if (!slots_.find(id)) { const auto lease = load(id); event("predict", lease); }
        ++stats_.predicted_blocks;
    }
    pump();
}
std::vector<SlotLease> Prefetcher::resolve(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & actual) {
    const auto start = std::chrono::steady_clock::now();
    check();
    const auto & g = store_.geometry();
    if (layer_ != layer || position_ != position || actual.size() != g.heads || store_.length(layer) != position + 1) {
        throw std::invalid_argument("prefetch resolve identity mismatch");
    }
    std::vector<BlockIdentity> needed;
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        const auto & positions = actual[head].positions;
        if (positions.empty() || positions.size() > budget_.total || positions.back() > position ||
            !std::is_sorted(positions.begin(), positions.end()) || std::adjacent_find(positions.begin(), positions.end()) != positions.end()) {
            throw std::invalid_argument("invalid actual prefetch selection");
        }
        std::set<std::uint64_t> ids;
        for (const auto p : positions) ids.insert(p / g.block_tokens);
        if (ids.size() > slots_.slots_per_head()) throw std::invalid_argument("actual selection exceeds slots_per_head");
        for (const auto id : ids) needed.push_back({layer, head, id, std::min<std::uint64_t>(g.block_tokens, position + 1 - id * g.block_tokens)});
    }
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        const auto lease = slots_.lease(i);
        const auto matching_block = [&](const BlockIdentity & id) { return id.layer == lease.identity.layer && id.head == lease.identity.head && id.block == lease.identity.block; };
        if (slots_.entry(i).state != SlotState::free && slots_.entry(i).state != SlotState::retiring &&
            std::none_of(needed.begin(), needed.end(), matching_block)) { ++stats_.unused_blocks; retire(lease); }
    }
    std::vector<SlotLease> result;
    for (const auto & id : needed) {
        auto lease = slots_.find(id);
        if (lease) {
            if (slots_.entry(lease->slot).state == SlotState::loading) { ++stats_.inflight_hits; event("inflight_hit", *lease); }
            else { ++stats_.ready_hits; event("hit", *lease); }
        } else {
            std::optional<SlotLease> prefix;
            for (std::uint32_t i = 0; i < slots_.size(); ++i) {
                const auto old = slots_.lease(i);
                const auto state = slots_.entry(i).state;
                if (state != SlotState::free && state != SlotState::retiring && old.identity.layer == layer && old.identity.head == id.head &&
                    old.identity.block == id.block && old.identity.version < id.version) { prefix = old; break; }
            }
            if (prefix) {
                while (slots_.entry(prefix->slot).state == SlotState::loading) poll(true);
                const auto count = blocks_[prefix->slot].valid_tokens;
                lease = slots_.refresh(*prefix, id);
                issue(*lease, count);
                ++stats_.refreshed_blocks;
                event("refresh", *lease);
            } else {
                lease = load(id);
                ++stats_.missing_blocks;
                event("missing", *lease);
            }
        }
        result.push_back(*lease);
    }
    for (const auto & lease : result) while (slots_.entry(lease.slot).state == SlotState::loading) poll(true);
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        auto & history = history_[layer][head];
        history.clear();
        const auto local_first = position + 1 - std::min<std::uint64_t>(budget_.local, position + 1);
        for (const auto p : actual[head].positions) if (p >= budget_.initial && p < local_first) {
            const auto id = p / g.block_tokens;
            if (history.empty() || history.back() != id) history.push_back(id);
        }
    }
    for (const auto & lease : result) slots_.pin(lease);
    stats_.resolve_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    return result;
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
    return result;
}
void Prefetcher::finish() {
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_.entry(i).readers) throw std::logic_error("finish with pinned host slot");
        if (slots_.entry(i).state != SlotState::free) retire(slots_.lease(i));
    }
    while (!requests_.empty()) poll(true);
    reader_->drain();
    if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("prefetch trace flush failed"); }
}
PrefetchStats Prefetcher::stats() const { auto result = stats_; result.io = reader_->stats(); return result; }
} // namespace solid
