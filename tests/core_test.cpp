#include "solid/attention.hpp"
#include "solid/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "expected failure was accepted");
}
class TestFile final : public solid::RandomAccessFile {
public:
    std::vector<unsigned char> bytes;
    std::size_t max_transfer = 7;
    bool fail_reads = false;
    bool fail_writes = false;
    bool eof = false;
    std::size_t read_at(void * destination, std::size_t count, std::uint64_t offset) override {
        if (fail_reads) throw std::runtime_error("injected read failure");
        if (eof || offset >= bytes.size()) return 0;
        count = std::min({count, max_transfer, bytes.size() - static_cast<std::size_t>(offset)});
        std::memcpy(destination, bytes.data() + offset, count);
        return count;
    }
    std::size_t write_at(const void * source, std::size_t count, std::uint64_t offset) override {
        if (fail_writes) throw std::runtime_error("injected write failure");
        count = std::min(count, max_transfer);
        if (offset + count > bytes.size()) throw std::runtime_error("test file boundary");
        std::memcpy(bytes.data() + offset, source, count);
        return count;
    }
    void resize(std::uint64_t size) override { bytes.resize(static_cast<std::size_t>(size)); }
};
void halves() {
    for (std::uint32_t value = 0; value < 65536; ++value) {
        if ((value & 0x7c00) == 0x7c00 && (value & 1023)) continue;
        require(solid::to_half(solid::from_half(static_cast<solid::Half>(value))) == value, "half round trip");
    }
    require(solid::to_half(std::ldexp(1.0f, -25)) == 0, "subnormal midpoint must round to even");
    require(solid::to_half(std::nextafter(std::ldexp(1.0f, -25), 1.0f)) == 1, "subnormal rounding above midpoint");
    require(solid::to_half(1.00048828125f) == 0x3c00, "normal midpoint rounding");
    require(solid::to_half(1.00146484375f) == 0x3c02, "normal even rounding");
}
void store_layout(solid::KvLayout layout) {
    solid::KvGeometry geometry{2, 2, 3, 4, 13, 64, layout};
    auto raw = std::make_unique<TestFile>();
    auto * inspect = raw.get();
    solid::FileKvStore disk(geometry, std::move(raw), 72);
    solid::MemoryKvStore memory(geometry);
    const auto allocation = disk.resident_bytes();
    for (std::uint32_t p = 0; p < 13; ++p) {
        for (std::uint32_t layer = 0; layer < 2; ++layer) {
            std::vector<solid::Half> keys(6), values(6);
            for (std::size_t d = 0; d < 6; ++d) {
                keys[d] = solid::to_half(static_cast<float>(layer * 100 + p * 6 + d));
                values[d] = solid::to_half(-solid::from_half(keys[d]) - 1);
            }
            disk.append(layer, p, keys, values);
            memory.append(layer, p, keys, values);
            for (std::uint32_t head = 0; head < 2; ++head) for (std::uint32_t block = 0; block <= p / 4; ++block) {
                solid::KvBlock actual, expected;
                disk.read(layer, head, block, actual);
                memory.read(layer, head, block, expected);
                require(actual.keys == expected.keys && actual.values == expected.values &&
                        actual.valid_tokens == expected.valid_tokens && actual.version == expected.version,
                        "dirty or committed block differs from memory oracle");
            }
        }
        require(disk.resident_bytes() == allocation, "SSD resident allocation grew with context length");
    }
    disk.flush();
    solid::KvBlock tail;
    disk.read(1, 1, 3, tail);
    require(tail.valid_tokens == 1 && tail.keys[3] == 0, "invalid tail padding");
    require(disk.io_stats().read_requests > 0 && disk.io_stats().write_requests > 0, "missing I/O counters");
    require(std::memcmp(inspect->bytes.data(), "SOLIDKV1", 8) == 0, "file format header");
    rejects([&] { disk.append(0, 13, std::vector<solid::Half>(6), std::vector<solid::Half>(6)); });
    inspect->eof = true;
    rejects([&] { disk.read(0, 0, 0, tail); });
    require(disk.failed(), "short read must poison the session");
    rejects([&] { disk.read(1, 0, 1, tail); });
}
void write_failure() {
    auto raw = std::make_unique<TestFile>();
    auto * inspect = raw.get();
    solid::FileKvStore disk({1, 1, 1, 2, 4, 64}, std::move(raw), 4);
    disk.append(0, 0, {solid::to_half(1)}, {solid::to_half(2)});
    inspect->fail_writes = true;
    rejects([&] { disk.flush(); });
    rejects([&] { disk.append(0, 1, {0}, {0}); });
}
void native_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("solid-kv-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        solid::FileKvStore disk({1, 1, 1, 2, 4, 64}, solid::create_session_file(path, true), 4);
        disk.append(0, 0, {solid::to_half(3)}, {solid::to_half(7)});
        disk.append(0, 1, {solid::to_half(4)}, {solid::to_half(8)});
        disk.flush();
        solid::KvBlock block;
        disk.read(0, 0, 0, block);
        require(solid::from_half(block.values[1]) == 8, "native file round trip");
        rejects([&] { (void) solid::create_session_file(path, true); });
        require(std::filesystem::exists(path), "existing session file was removed");
    }
    require(!std::filesystem::exists(path), "session cleanup");
}
void attention_and_selection() {
    solid::KvGeometry geometry{1, 2, 1, 2, 16, 64};
    solid::SparseBudget budget{4, 1, 1, 1};
    solid::RepresentativeIndex representatives(geometry, 4, budget);
    solid::MemoryKvStore memory(geometry);
    solid::FileKvStore disk(geometry, std::make_unique<TestFile>(), 16);
    for (std::uint64_t p = 0; p < 10; ++p) {
        const std::vector<solid::Half> keys(2, solid::to_half(static_cast<float>(p + 1)));
        const std::vector<solid::Half> values{solid::to_half(static_cast<float>(p)), solid::to_half(static_cast<float>(p * p))};
        representatives.observe(0, p, {1, 1, 1, 1}, keys);
        memory.append(0, p, keys, values);
        disk.append(0, p, keys, values);
        if (p == 1) require(!representatives.ready(0, 0), "representative finalized too early");
        if (p == 2) require(representatives.ready(0, 0), "representative not finalized after full window");
    }
    const std::vector<float> query{1, 1, -1, -1};
    auto selection = solid::select_blocks(geometry, budget, representatives, 0, 9, query);
    require(selection[0].positions == std::vector<std::uint64_t>({0, 6, 7, 9}), "positive head ranking");
    require(selection[1].positions == std::vector<std::uint64_t>({0, 1, 2, 9}), "negative head ranking or token budget");
    require(representatives.representative_positions(0, 0, 0) == std::vector<std::uint64_t>{1}, "representative score ordering");
    const auto sparse_memory = solid::streamed_attention(memory, 0, 9, 4, query, 1, &selection);
    const auto sparse_disk = solid::streamed_attention(disk, 0, 9, 4, query, 1, &selection);
    require(sparse_memory == sparse_disk, "memory and SSD sparse attention differ");
    const auto dense = solid::streamed_attention(disk, 0, 5, 4, query, 1);
    for (std::size_t h = 0; h < 4; ++h) {
        long double sum = 0, weighted = 0;
        for (int p = 0; p <= 5; ++p) {
            const auto weight = std::exp(static_cast<long double>(query[h]) * (p + 1));
            sum += weight;
            weighted += weight * (h < 2 ? p : p * p);
        }
        require(std::abs(dense[h] - static_cast<float>(weighted / sum)) < 1e-5f, "causal online softmax differs from independent dense oracle");
    }
    selection[0].positions.push_back(10);
    rejects([&] { (void) solid::streamed_attention(memory, 0, 9, 4, query, 1, &selection); });
    rejects([&] { (void) solid::select_blocks(geometry, budget, representatives, 0, 9,
                 {std::numeric_limits<float>::quiet_NaN(), 1, 1, 1}); });
}
void session_batching() {
    solid::KvGeometry geometry{1, 1, 2, 2, 16, 64};
    solid::SparseBudget budget{4, 1, 1, 1};
    solid::AttentionSession whole(std::make_unique<solid::MemoryKvStore>(geometry), 2, budget, true, 3);
    solid::AttentionSession split(std::make_unique<solid::FileKvStore>(geometry, std::make_unique<TestFile>(), 16), 2, budget, true, 3);
    std::vector<float> q, k, v;
    for (int p = 0; p < 8; ++p) {
        q.insert(q.end(), {1, 0.5f, -0.5f, 1});
        k.insert(k.end(), {p * 0.25f, 1});
        v.insert(v.end(), {static_cast<float>(p), -static_cast<float>(p)});
    }
    const auto expected = whole.process(0, 0, 8, q, k, v, 0.5f);
    std::vector<float> actual;
    for (int first = 0; first < 8; first += 2) {
        const auto batch = split.process(0, first, 2, {q.begin() + first * 4, q.begin() + (first + 2) * 4},
            {k.begin() + first * 2, k.begin() + (first + 2) * 2}, {v.begin() + first * 2, v.begin() + (first + 2) * 2}, 0.5f);
        actual.insert(actual.end(), batch.begin(), batch.end());
    }
    split.flush();
    require(actual == expected, "microbatch partition changes attention or selection");
    require(split.stats().attention_queries == 8 && split.stats().io.write_bytes > 0, "session counters");
}
}
int main() {
    try {
        halves();
        store_layout(solid::KvLayout::separate);
        store_layout(solid::KvLayout::interleaved);
        write_failure();
        native_file();
        attention_and_selection();
        session_batching();
        auto overflow = solid::KvGeometry{0xffffffff, 0xffffffff, 0xffffffff, 32, 0xffffffff, 4096};
        rejects([&] { overflow.validate(); });
        std::cout << "KV, I/O, representative and attention checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
