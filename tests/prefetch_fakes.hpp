// 预取检查共用的模拟 SSD 读取器与可控制完成时机的辅助对象。
#pragma once
#include "solid/device_cache.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>

namespace test_prefetch {
inline void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F operation) {
    bool caught = false;
    try { operation(); } catch (const std::exception &) { caught = true; }
    require(caught, "invalid lifecycle was accepted");
}
class File final : public solid::RandomAccessFile {
public:
    std::vector<unsigned char> bytes;
    std::size_t read_at(void * dst, std::size_t size, std::uint64_t offset) override {
        if (offset + size > bytes.size()) return 0;
        std::memcpy(dst, bytes.data() + offset, size); return size;
    }
    std::size_t write_at(const void * src, std::size_t size, std::uint64_t offset) override {
        if (offset + size > bytes.size()) throw std::runtime_error("file boundary");
        std::memcpy(bytes.data() + offset, src, size); return size;
    }
    void resize(std::uint64_t size) override { bytes.resize(size); }
};
class Reader final : public solid::AsyncReader {
public:
    explicit Reader(File * file = nullptr, std::size_t capacity = 2) : file(file), capacity(capacity) {}
    struct Pending { std::uint64_t id; unsigned char * dst; std::size_t bytes; std::uint64_t offset; bool cancelled; };
    File * file;
    std::size_t capacity;
    bool auto_complete = false;
    bool reverse = true;
    bool fail_next = false;
    std::optional<solid::ReadCompletion> duplicate;
    std::function<void()> before_wait;
    solid::AsyncIoStats counters;
    std::vector<Pending> pending;
    std::vector<Pending> submitted;
    bool submit(std::uint64_t id, int, void * dst, std::size_t bytes, std::uint64_t offset) override {
        if (pending.size() >= capacity) return false;
        require(file != nullptr, "unexpected file read");
        const auto address = reinterpret_cast<std::uintptr_t>(dst);
        for (const auto & old : pending) {
            const auto prior = reinterpret_cast<std::uintptr_t>(old.dst);
            require(address + bytes <= prior || prior + old.bytes <= address, "reused destination before original completion");
        }
        pending.push_back({id, static_cast<unsigned char *>(dst), bytes, offset, false});
        submitted.push_back(pending.back());
        ++counters.read_requests;
        return true;
    }
    void cancel(std::uint64_t id) override {
        for (auto & op : pending) if (op.id == id && !op.cancelled) {
            op.cancelled = true;
            ++counters.cancel_requests;
            ++counters.cancel_completions; // Ack arrives before the original read.
            ++counters.cancel_busy;
        }
    }
    void flush() override {}
    bool poll(solid::ReadCompletion & result, bool wait) override {
        if (duplicate) { result = *duplicate; duplicate.reset(); return true; }
        if (pending.empty() || (!wait && !auto_complete)) return false;
        if (wait && before_wait) before_wait();
        const auto index = reverse ? pending.size() - 1 : 0;
        const auto op = pending[index];
        pending.erase(pending.begin() + index);
        if (fail_next) { fail_next = false; result = {op.id, -5}; }
        else {
            require(op.offset + op.bytes <= file->bytes.size(), "async file boundary");
            // An EALREADY cancellation acknowledgment can precede the original
            // write. It does not end the destination buffer's lifetime.
            std::memcpy(op.dst, file->bytes.data() + op.offset, op.bytes);
            counters.read_bytes += op.bytes;
            result = {op.id, static_cast<std::int64_t>(op.bytes)};
        }
        ++counters.completed_reads;
        return true;
    }
    void drain() override { solid::ReadCompletion value{}; while (poll(value, true)) {} }
    solid::AsyncIoStats stats() const override { return counters; }
};
class Transport final : public solid::DeviceTransport {
public:
    struct Copy { std::uint32_t slot; const std::vector<solid::Half> * source; bool done = false; };
    std::map<std::uint64_t, std::vector<Copy>> copies;
    std::map<std::uint32_t, std::vector<solid::Half>> memory;
    std::uint64_t next = 1;
    bool immediate = false;
    bool fail_upload = false;
    bool batch_uploads = false;
    std::uint64_t completion_polls=0;
    std::uint64_t upload(std::uint32_t slot, const solid::KvBlock & block) override {
        if (fail_upload) throw std::runtime_error("injected transfer failure");
        const auto id = next++;
        copies.emplace(id, std::vector<Copy>{{slot,&block.data,false}});
        return id;
    }
    bool batches_uploads() const override { return batch_uploads; }
    std::uint64_t upload_batch(const std::vector<solid::DeviceUpload> & blocks) override {
        if (fail_upload) throw std::runtime_error("injected transfer failure");
        const auto id=next++;
        auto & batch=copies[id];
        for (const auto & block:blocks) batch.push_back({block.slot,&block.block->data,false});
        return id;
    }
    void write(Copy & copy) { if (!copy.done) { memory[copy.slot] = *copy.source; copy.done = true; } }
    bool complete(std::uint64_t id, bool wait) override {
        ++completion_polls;
        auto found = copies.find(id);
        require(found != copies.end(), "unknown fake transfer");
        if (!found->second.front().done && !wait && !immediate) return false;
        for (auto & copy:found->second) write(copy);
        copies.erase(found);
        return true;
    }
    void drain() override { for (auto & pair : copies) for (auto & copy:pair.second) write(copy); }
    std::uint64_t resident_bytes() const override { return 4096; }
};
inline std::vector<solid::Half> token(const solid::KvGeometry & g, std::uint32_t layer, std::uint64_t position) {
    std::vector<solid::Half> kv(static_cast<std::size_t>(g.heads) * 2 * g.dimension);
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = solid::to_half(static_cast<float>(layer * 100 + position + i * 0.125));
    return kv;
}
inline void fill(solid::KvStore & store, std::uint64_t count) {
    for (std::uint64_t p = 0; p < count; ++p) for (std::uint32_t layer = 0; layer < store.geometry().layers; ++layer) {
        store.append_interleaved(layer, p, token(store.geometry(), layer, p));
    }
    store.flush();
}
inline void verify(solid::KvStore & store, solid::Prefetcher & prefetch, const std::vector<solid::SlotLease> & leases) {
    for (const auto & lease : leases) {
        solid::KvBlock expected;
        store.read(lease.identity.layer, lease.identity.head, lease.identity.block, expected);
        const auto & actual = prefetch.block(lease);
        require(actual.data == expected.data && actual.version == expected.version, "prefetch differs from synchronous latest KV");
    }
}
} // namespace test_prefetch
