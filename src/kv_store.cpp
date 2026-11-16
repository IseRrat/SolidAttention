#include "solid/kv_store.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

namespace solid {
namespace {
constexpr std::uint64_t header_bytes = 4096;
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / a) {
        throw std::overflow_error("KV size exceeds signed 64-bit offsets");
    }
    return a * b;
}
std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const auto remainder = value % alignment;
    if (remainder == 0) return value;
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - alignment + remainder) {
        throw std::overflow_error("KV alignment overflow");
    }
    return value + alignment - remainder;
}
void put_u64(unsigned char * p, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(value >> (8 * i));
}
class NativeFile final : public RandomAccessFile {
public:
    NativeFile(std::filesystem::path path, bool remove_on_close) : path_(std::move(path)), remove_(remove_on_close) {
#ifdef _WIN32
        const auto status = _wsopen_s(&fd_, path_.c_str(), _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                                     _SH_DENYRW, _S_IREAD | _S_IWRITE);
        if (status != 0) throw std::system_error(status, std::generic_category(), "create KV file");
#else
        fd_ = ::open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "create KV file");
#endif
    }
    ~NativeFile() override {
#ifdef _WIN32
        if (fd_ >= 0) _close(fd_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
        if (remove_) { std::error_code error; std::filesystem::remove(path_, error); }
    }
    std::size_t read_at(void * destination, std::size_t bytes, std::uint64_t offset) override {
        check_offset(offset);
#ifdef _WIN32
        if (_lseeki64(fd_, static_cast<__int64>(offset), SEEK_SET) < 0) fail("seek KV read");
        int n;
        do { n = _read(fd_, destination, static_cast<unsigned int>(std::min<std::size_t>(bytes, INT_MAX))); } while (n < 0 && errno == EINTR);
#else
        ssize_t n;
        do { n = ::pread(fd_, destination, std::min<std::size_t>(bytes, INT_MAX), static_cast<off_t>(offset)); } while (n < 0 && errno == EINTR);
#endif
        if (n < 0) fail("read KV file");
        return static_cast<std::size_t>(n);
    }
    std::size_t write_at(const void * source, std::size_t bytes, std::uint64_t offset) override {
        check_offset(offset);
#ifdef _WIN32
        if (_lseeki64(fd_, static_cast<__int64>(offset), SEEK_SET) < 0) fail("seek KV write");
        int n;
        do { n = _write(fd_, source, static_cast<unsigned int>(std::min<std::size_t>(bytes, INT_MAX))); } while (n < 0 && errno == EINTR);
#else
        ssize_t n;
        do { n = ::pwrite(fd_, source, std::min<std::size_t>(bytes, INT_MAX), static_cast<off_t>(offset)); } while (n < 0 && errno == EINTR);
#endif
        if (n < 0) fail("write KV file");
        return static_cast<std::size_t>(n);
    }
    void resize(std::uint64_t bytes) override {
        check_offset(bytes);
#ifdef _WIN32
        const auto status = _chsize_s(fd_, bytes);
        if (status != 0) throw std::system_error(status, std::generic_category(), "resize KV file");
#else
        if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) fail("resize KV file");
#endif
    }
private:
    int fd_ = -1;
    std::filesystem::path path_;
    bool remove_;
    static void check_offset(std::uint64_t value) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("file offset overflow");
        }
    }
    [[noreturn]] static void fail(const char * operation) {
        throw std::system_error(errno, std::generic_category(), operation);
    }
};
} // namespace

Half to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
    const auto exponent = (bits >> 23) & 255;
    const auto fraction = bits & 0x7fffff;
    if (exponent == 255) return static_cast<Half>(sign | 0x7c00 | (fraction ? 0x200 : 0));
    const int e = static_cast<int>(exponent) - 127;
    if (e > 15) return static_cast<Half>(sign | 0x7c00);
    if (e < -25) return sign;
    if (e < -14) {
        const auto mantissa = fraction | 0x800000;
        const int shift = -e - 1;
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((std::uint32_t{1} << shift) - 1);
        const auto midpoint = std::uint32_t{1} << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1))) ++rounded;
        return static_cast<Half>(sign | rounded);
    }
    auto rounded = (static_cast<std::uint32_t>(e + 15) << 10) | (fraction >> 13);
    const auto remainder = fraction & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) ++rounded;
    return static_cast<Half>(sign | rounded);
}

float from_half(Half value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
    const auto exponent = (value >> 10) & 31;
    auto fraction = static_cast<std::uint32_t>(value & 1023);
    std::uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            int e = -14;
            while ((fraction & 0x400) == 0) { fraction <<= 1; --e; }
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | ((fraction & 1023) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000 | (fraction << 13);
    else bits = sign | (static_cast<std::uint32_t>(exponent + 112) << 23) | (fraction << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void KvGeometry::validate() const {
    if (layers == 0 || heads == 0 || dimension == 0 || block_tokens == 0 || capacity_tokens == 0 ||
        alignment == 0 || alignment > header_bytes || (alignment & (alignment - 1)) != 0 ||
        (layout != KvLayout::separate && layout != KvLayout::interleaved)) {
        throw std::invalid_argument("invalid KV geometry");
    }
    (void) file_bytes();
}
std::uint64_t KvGeometry::blocks_per_head() const {
    if (block_tokens == 0) throw std::invalid_argument("block_tokens must be positive");
    return (static_cast<std::uint64_t>(capacity_tokens) + block_tokens - 1) / block_tokens;
}
std::uint64_t KvGeometry::plane_bytes() const {
    return product(product(block_tokens, dimension), sizeof(Half));
}
std::uint64_t KvGeometry::slot_bytes() const {
    if (alignment == 0) throw std::invalid_argument("alignment must be positive");
    return layout == KvLayout::separate ? product(2, align_up(plane_bytes(), alignment)) : align_up(product(2, plane_bytes()), alignment);
}
std::uint64_t KvGeometry::file_bytes() const {
    const auto data = product(product(product(layers, heads), blocks_per_head()), slot_bytes());
    if (data > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - header_bytes) {
        throw std::overflow_error("KV file size overflow");
    }
    return data + header_bytes;
}
std::uint64_t KvGeometry::block_offset(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const {
    if (layer >= layers || head >= heads || block >= blocks_per_head()) throw std::out_of_range("KV block identity");
    const auto index = (static_cast<std::uint64_t>(layer) * heads + head) * blocks_per_head() + block;
    return header_bytes + product(index, slot_bytes());
}
std::unique_ptr<RandomAccessFile> create_session_file(const std::filesystem::path & path, bool remove_on_close) {
    return std::make_unique<NativeFile>(path, remove_on_close);
}
KvStore::KvStore(KvGeometry geometry) : geometry_(geometry) { geometry_.validate(); }
void KvStore::validate_append(std::uint32_t layer, std::uint64_t position, std::uint64_t current,
                             const std::vector<Half> & keys, const std::vector<Half> & values) const {
    if (layer >= geometry_.layers || position != current || position >= geometry_.capacity_tokens ||
        keys.size() != static_cast<std::uint64_t>(geometry_.heads) * geometry_.dimension || keys.size() != values.size()) {
        throw std::invalid_argument("KV append requires consecutive positions and complete heads");
    }
}
void KvStore::prepare_block(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                           std::uint64_t current, KvBlock & result) const {
    (void) geometry_.block_offset(layer, head, block);
    const auto first = block * geometry_.block_tokens;
    if (first >= current) throw std::out_of_range("KV block has no valid tokens");
    result.first_position = first;
    result.valid_tokens = static_cast<std::uint32_t>(std::min<std::uint64_t>(geometry_.block_tokens, current - first));
    result.version = result.valid_tokens;
    result.keys.assign(static_cast<std::size_t>(geometry_.block_tokens) * geometry_.dimension, 0);
    result.values.assign(result.keys.size(), 0);
}
MemoryKvStore::MemoryKvStore(KvGeometry geometry) : KvStore(geometry), data_(geometry.layers), lengths_(geometry.layers, 0) {
    const auto elements = static_cast<std::size_t>(geometry.capacity_tokens) * geometry.heads * 2 * geometry.dimension;
    for (auto & data : data_) data.reserve(elements);
}
void MemoryKvStore::append(std::uint32_t layer, std::uint64_t position,
                           const std::vector<Half> & keys, const std::vector<Half> & values) {
    const auto current = length(layer);
    validate_append(layer, position, current, keys, values);
    auto & data = data_[layer];
    const auto previous = data.size();
    data.resize(previous + keys.size() * 2);
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        std::copy_n(keys.data() + head * geometry_.dimension, geometry_.dimension, data.data() + previous + head * 2 * geometry_.dimension);
        std::copy_n(values.data() + head * geometry_.dimension, geometry_.dimension, data.data() + previous + (head * 2 + 1) * geometry_.dimension);
    }
    ++lengths_[layer];
}
void MemoryKvStore::read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & result) {
    prepare_block(layer, head, block, length(layer), result);
    for (std::uint32_t i = 0; i < result.valid_tokens; ++i) {
        const auto source = ((result.first_position + i) * geometry_.heads + head) * 2 * geometry_.dimension;
        std::copy_n(data_[layer].data() + source, geometry_.dimension, result.keys.data() + i * geometry_.dimension);
        std::copy_n(data_[layer].data() + source + geometry_.dimension, geometry_.dimension, result.values.data() + i * geometry_.dimension);
    }
}
std::uint64_t MemoryKvStore::length(std::uint32_t layer) const { return lengths_.at(layer); }
std::uint64_t MemoryKvStore::resident_bytes() const {
    std::uint64_t total = lengths_.capacity() * sizeof(std::uint64_t) + data_.capacity() * sizeof(std::vector<Half>);
    for (const auto & data : data_) total += data.capacity() * sizeof(Half);
    return total;
}
FileKvStore::FileKvStore(KvGeometry geometry, std::unique_ptr<RandomAccessFile> file, std::size_t buffer_bytes) :
    KvStore(geometry), file_(std::move(file)), pending_(geometry.layers), lengths_(geometry.layers, 0),
    committed_(geometry.layers, 0), token_elements_(static_cast<std::size_t>(geometry.heads) * 2 * geometry.dimension),
    pending_tokens_(buffer_bytes / sizeof(Half) / token_elements_) {
    if (!file_ || pending_tokens_ == 0) throw std::invalid_argument("write buffer must fit one token across all KV heads");
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::invalid_argument("KV v1 requires little-endian storage");
    for (auto & pending : pending_) pending.values.reserve(pending_tokens_ * token_elements_);
    scratch_.resize(static_cast<std::size_t>(geometry.block_tokens) * 2 * geometry.dimension);
    file_->resize(geometry.file_bytes());
    std::array<unsigned char, header_bytes> header{};
    std::memcpy(header.data(), "SOLIDKV1", 8);
    const std::uint64_t fields[] = {1, header_bytes, geometry.layers, geometry.heads, geometry.dimension,
        geometry.block_tokens, geometry.capacity_tokens, geometry.alignment, static_cast<std::uint32_t>(geometry.layout), sizeof(Half)};
    for (std::size_t i = 0; i < std::size(fields); ++i) put_u64(header.data() + 8 + i * 8, fields[i]);
    write_exact(header.data(), header.size(), 0);
    stats_ = {};
}
void FileKvStore::check_ready() const { if (failed_) throw std::runtime_error("KV session failed during I/O"); }
void FileKvStore::read_exact(void * destination, std::size_t bytes, std::uint64_t offset) {
    auto * data = static_cast<unsigned char *>(destination);
    std::size_t done = 0;
    try {
        while (done < bytes) {
            const auto n = file_->read_at(data + done, bytes - done, offset + done);
            ++stats_.read_requests;
            if (n == 0 || n > bytes - done) throw std::runtime_error("incomplete KV read at " + std::to_string(offset + done));
            stats_.read_bytes += n;
            done += n;
        }
    } catch (...) { failed_ = true; throw; }
}
void FileKvStore::write_exact(const void * source, std::size_t bytes, std::uint64_t offset) {
    const auto * data = static_cast<const unsigned char *>(source);
    std::size_t done = 0;
    try {
        while (done < bytes) {
            const auto n = file_->write_at(data + done, bytes - done, offset + done);
            ++stats_.write_requests;
            if (n == 0 || n > bytes - done) throw std::runtime_error("incomplete KV write at " + std::to_string(offset + done));
            stats_.write_bytes += n;
            done += n;
        }
    } catch (...) { failed_ = true; throw; }
}
void FileKvStore::append(std::uint32_t layer, std::uint64_t position,
                         const std::vector<Half> & keys, const std::vector<Half> & values) {
    check_ready();
    validate_append(layer, position, length(layer), keys, values);
    auto & pending = pending_.at(layer);
    if (pending.values.size() / token_elements_ == pending_tokens_) flush_layer(layer);
    if (pending.values.empty()) pending.first = position;
    const auto previous = pending.values.size();
    pending.values.resize(previous + token_elements_);
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        std::copy_n(keys.data() + head * geometry_.dimension, geometry_.dimension,
                    pending.values.data() + previous + head * 2 * geometry_.dimension);
        std::copy_n(values.data() + head * geometry_.dimension, geometry_.dimension,
                    pending.values.data() + previous + (head * 2 + 1) * geometry_.dimension);
    }
    ++lengths_[layer];
}
void FileKvStore::flush_layer(std::uint32_t layer) {
    check_ready();
    auto & pending = pending_.at(layer);
    const auto count = pending.values.size() / token_elements_;
    const auto dimension = geometry_.dimension;
    for (std::size_t first = 0; first < count;) {
        const auto position = pending.first + first;
        const auto in_block = position % geometry_.block_tokens;
        const auto take = std::min<std::size_t>(count - first, geometry_.block_tokens - in_block);
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            const auto offset = geometry_.block_offset(layer, head, position / geometry_.block_tokens);
            if (geometry_.layout == KvLayout::separate) {
                for (std::size_t i = 0; i < take; ++i) {
                    const auto source = (first + i) * token_elements_ + head * 2 * dimension;
                    std::copy_n(pending.values.data() + source, dimension, scratch_.data() + i * dimension);
                    std::copy_n(pending.values.data() + source + dimension, dimension, scratch_.data() + (take + i) * dimension);
                }
                write_exact(scratch_.data(), take * dimension * sizeof(Half), offset + in_block * dimension * sizeof(Half));
                write_exact(scratch_.data() + take * dimension, take * dimension * sizeof(Half),
                            offset + align_up(geometry_.plane_bytes(), geometry_.alignment) + in_block * dimension * sizeof(Half));
            } else {
                for (std::size_t i = 0; i < take; ++i) {
                    std::copy_n(pending.values.data() + (first + i) * token_elements_ + head * 2 * dimension,
                                2 * dimension, scratch_.data() + i * 2 * dimension);
                }
                write_exact(scratch_.data(), take * 2 * dimension * sizeof(Half), offset + in_block * 2 * dimension * sizeof(Half));
            }
        }
        first += take;
    }
    committed_[layer] = lengths_[layer];
    pending.values.clear();
    pending.first = committed_[layer];
}
void FileKvStore::read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & result) {
    check_ready();
    prepare_block(layer, head, block, length(layer), result);
    const auto dimension = geometry_.dimension;
    const auto offset = geometry_.block_offset(layer, head, block);
    const auto committed = committed_[layer] <= result.first_position ? 0 :
        std::min<std::uint64_t>(result.valid_tokens, committed_[layer] - result.first_position);
    if (committed > 0) {
        if (geometry_.layout == KvLayout::separate) {
            read_exact(result.keys.data(), committed * dimension * sizeof(Half), offset);
            read_exact(result.values.data(), committed * dimension * sizeof(Half), offset + align_up(geometry_.plane_bytes(), geometry_.alignment));
        } else {
            read_exact(scratch_.data(), committed * 2 * dimension * sizeof(Half), offset);
            for (std::size_t i = 0; i < committed; ++i) {
                std::copy_n(scratch_.data() + i * 2 * dimension, dimension, result.keys.data() + i * dimension);
                std::copy_n(scratch_.data() + (i * 2 + 1) * dimension, dimension, result.values.data() + i * dimension);
            }
        }
    }
    const auto & pending = pending_[layer];
    for (std::uint32_t i = static_cast<std::uint32_t>(committed); i < result.valid_tokens; ++i) {
        const auto source = (result.first_position + i - pending.first) * token_elements_ + head * 2 * dimension;
        if (source + 2 * dimension > pending.values.size()) throw std::logic_error("KV dirty range mismatch");
        std::copy_n(pending.values.data() + source, dimension, result.keys.data() + i * dimension);
        std::copy_n(pending.values.data() + source + dimension, dimension, result.values.data() + i * dimension);
    }
}
void FileKvStore::flush() { for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) flush_layer(layer); }
std::uint64_t FileKvStore::length(std::uint32_t layer) const { check_ready(); return lengths_.at(layer); }
std::uint64_t FileKvStore::resident_bytes() const {
    std::uint64_t total = (lengths_.capacity() + committed_.capacity()) * sizeof(std::uint64_t) + scratch_.capacity() * sizeof(Half);
    for (const auto & pending : pending_) total += sizeof(Pending) + pending.values.capacity() * sizeof(Half);
    return total;
}
} // namespace solid
