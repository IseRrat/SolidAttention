#include "solid/kv_store.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
class File final : public solid::RandomAccessFile {
public:
    std::vector<unsigned char> bytes;
    std::size_t read_at(void * data, std::size_t size, std::uint64_t offset) override {
        if (offset + size > bytes.size()) throw std::runtime_error("read boundary");
        std::memcpy(data, bytes.data() + offset, size);
        return size;
    }
    std::size_t write_at(const void * data, std::size_t size, std::uint64_t offset) override {
        if (offset + size > bytes.size()) throw std::runtime_error("write boundary");
        std::memcpy(bytes.data() + offset, data, size);
        return size;
    }
    void resize(std::uint64_t size) override { bytes.resize(size); }
};
solid::IoStats run(solid::KvLayout layout) {
    solid::KvGeometry geometry{2, 2, 128, 3, 11, 4096, layout};
    auto file = std::make_unique<File>();
    auto * raw = file.get();
    solid::FileKvStore store(geometry, std::move(file), 2048); // two whole tokens
    for (std::uint64_t p = 0; p < 10; ++p) for (std::uint32_t layer = 0; layer < 2; ++layer) {
        std::vector<solid::Half> kv(512);
        for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = solid::to_half(static_cast<float>(p * 0.5 + layer + i * 0.01));
        store.append_interleaved(layer, p, kv);
    }
    store.flush();
    const auto before = store.io_stats();
    for (std::uint32_t layer = 0; layer < 2; ++layer) for (std::uint32_t head = 0; head < 2; ++head) for (std::uint64_t id = 0; id < 4; ++id) {
        solid::KvBlock block;
        store.read(layer, head, id, block);
        require(block.layout == layout, "reader changed physical block layout");
        const auto offset = geometry.block_offset(layer, head, id);
        for (std::uint32_t row = 0; row < block.valid_tokens; ++row) for (std::uint32_t d = 0; d < 128; ++d) {
            const auto p = id * 3 + row;
            require(block.key(row)[d] == solid::to_half(static_cast<float>(p * 0.5 + layer + (head * 256 + d) * 0.01)), "key row identity");
            require(block.value(row)[d] == solid::to_half(static_cast<float>(p * 0.5 + layer + (head * 256 + 128 + d) * 0.01)), "value row identity");
            if (layout == solid::KvLayout::interleaved) {
                require(std::memcmp(block.key(row), raw->bytes.data() + offset + row * 512, 512) == 0, "interleaved payload was reordered on read");
            }
        }
        for (auto row = block.valid_tokens; row < 3; ++row) for (int d = 0; d < 128; ++d) {
            require(block.key(row)[d] == 0 && block.value(row)[d] == 0, "tail padding");
        }
    }
    auto delta = store.io_stats();
    delta.read_requests -= before.read_requests;
    delta.read_bytes -= before.read_bytes;
    return delta;
}
}
int main() {
    try {
        const auto separate = run(solid::KvLayout::separate), interleaved = run(solid::KvLayout::interleaved);
        require(separate.read_requests == 32 && interleaved.read_requests == 16, "committed block request count");
        require(separate.read_bytes == interleaved.read_bytes && interleaved.read_bytes == 20480, "logical payload byte count");
        require(separate.write_requests == 2 * interleaved.write_requests && separate.write_bytes == interleaved.write_bytes, "write request consolidation");
        std::cout << "interleaved payload, aligned offsets, tail and request accounting checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
