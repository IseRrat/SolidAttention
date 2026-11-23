#include "solid/kv_store.hpp"
#include "solid/trace.hpp"
#include "solid/async_reader.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
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
    int native_handle() const override { return fd_; }
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
#ifdef __linux__
        if (direct_fd_ >= 0) ::close(direct_fd_);
#endif
#endif
        if (remove_) { std::error_code error; std::filesystem::remove(path_, error); }
    }
    void enable_direct_reads() override {
#ifdef __linux__
        if (direct_fd_>=0) return;
        if (::fdatasync(fd_)!=0) fail("flush KV before enabling direct reads");
        direct_fd_=::open(path_.c_str(),O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_<0) fail("open KV file for direct reads");
#ifdef STATX_DIOALIGN
        struct statx attributes{};
        if (::statx(direct_fd_,"",AT_EMPTY_PATH,STATX_DIOALIGN,&attributes)==0 && (attributes.stx_mask&STATX_DIOALIGN)) {
            if (!attributes.stx_dio_mem_align || !attributes.stx_dio_offset_align) {
                throw std::runtime_error("KV filesystem does not support direct reads");
            }
            address_alignment_=attributes.stx_dio_mem_align;
            offset_alignment_=attributes.stx_dio_offset_align;
        }
#endif
#else
        throw std::runtime_error("direct KV reads require Linux");
#endif
    }
    KvReadRequest prepare_read(void * destination,std::size_t bytes,std::uint64_t offset) const override {
        KvReadRequest result{fd_,destination,bytes,offset,{}};
#ifdef __linux__
        if (direct_fd_>=0) {
            check_offset(offset);
            if (bytes>static_cast<std::uint64_t>(INT64_MAX)-offset) throw std::overflow_error("direct KV read size");
            result.fd=direct_fd_;
            if (offset%offset_alignment_ || bytes%offset_alignment_ ||
                reinterpret_cast<std::uintptr_t>(destination)%address_alignment_) {
                result.skip=offset%offset_alignment_;
                result.offset=offset-result.skip;
                result.bytes=align_up(bytes+result.skip,offset_alignment_);
                if (result.bytes>INT_MAX) throw std::overflow_error("aligned KV read size");
                result.destination=destination;
                result.logical_bytes=bytes;
                std::size_t space=result.bytes+address_alignment_-1;
                result.scratch.reset(new unsigned char[space]);
                result.buffer=result.scratch.get();
                if (!std::align(address_alignment_,result.bytes,result.buffer,space)) throw std::logic_error("KV read alignment");
            }
        }
#endif
        return result;
    }
    std::size_t read_at(void * destination, std::size_t bytes, std::uint64_t offset) override {
        check_offset(offset);
#ifdef _WIN32
        if (_lseeki64(fd_, static_cast<__int64>(offset), SEEK_SET) < 0) fail("seek KV read");
        int n;
        do { n = _read(fd_, destination, static_cast<unsigned int>(std::min<std::size_t>(bytes, INT_MAX))); } while (n < 0 && errno == EINTR);
#else
        auto request=prepare_read(destination,std::min<std::size_t>(bytes,INT_MAX),offset);
        ssize_t n;
        do { n = ::pread(request.fd,request.buffer,request.bytes,static_cast<off_t>(request.offset)); } while (n < 0 && errno == EINTR);
        n=request.finish(n);
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
#ifdef __linux__
    int direct_fd_ = -1;
    std::size_t address_alignment_ = 4096, offset_alignment_ = 4096;
#endif
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

std::int64_t KvReadRequest::finish(std::int64_t transferred) const {
    if (!scratch || transferred<0) return transferred;
    const auto read=static_cast<std::uint64_t>(transferred);
    const auto copied=read>skip?std::min<std::uint64_t>(logical_bytes,read-skip):0;
    if (copied) std::memcpy(destination,static_cast<const unsigned char *>(buffer)+skip,copied);
    return static_cast<std::int64_t>(copied);
}

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
        (layout != KvLayout::separate && layout != KvLayout::interleaved) ||
        (contiguous_planes && layout != KvLayout::separate)) {
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
std::uint64_t KvGeometry::value_offset() const {
    return align_up(contiguous_planes ? product(product(capacity_tokens, dimension), sizeof(Half)) : plane_bytes(), alignment);
}
std::uint64_t KvGeometry::file_bytes() const {
    const auto head_bytes = contiguous_planes ? product(2, value_offset()) : product(blocks_per_head(), slot_bytes());
    const auto data = product(product(layers, heads), head_bytes);
    if (data > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - header_bytes) {
        throw std::overflow_error("KV file size overflow");
    }
    return data + header_bytes;
}
std::uint64_t KvGeometry::block_offset(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const {
    if (layer >= layers || head >= heads || block >= blocks_per_head()) throw std::out_of_range("KV block identity");
    if (contiguous_planes) {
        return header_bytes + product(static_cast<std::uint64_t>(layer) * heads + head, 2 * value_offset()) +
            product(block, plane_bytes());
    }
    const auto index = (static_cast<std::uint64_t>(layer) * heads + head) * blocks_per_head() + block;
    return header_bytes + product(index, slot_bytes());
}
std::unique_ptr<RandomAccessFile> create_session_file(const std::filesystem::path & path, bool remove_on_close) {
    return std::make_unique<NativeFile>(path, remove_on_close);
}
KvStore::KvStore(KvGeometry geometry) : geometry_(geometry) { geometry_.validate(); }
void KvStore::read_rows(std::uint32_t layer, std::uint32_t head, std::uint64_t first, std::uint32_t count,
                        Half * keys, Half * values, std::vector<Half> &) {
    const auto & g = geometry_;
    if (!keys || head >= g.heads || first > length(layer) || count > length(layer) - first) {
        throw std::invalid_argument("KV row range");
    }
    KvBlock block;
    for (std::uint32_t copied = 0; copied < count;) {
        const auto position = first + copied;
        read(layer, head, position / g.block_tokens, block);
        const auto row = position % g.block_tokens;
        if (row >= block.valid_tokens) throw std::logic_error("incomplete KV row range");
        const auto take = std::min<std::uint32_t>(count - copied, block.valid_tokens - row);
        if (!values && g.layout == KvLayout::interleaved) {
            std::copy_n(block.key(row), static_cast<std::size_t>(take) * 2 * g.dimension,
                        keys + static_cast<std::size_t>(copied) * 2 * g.dimension);
        } else {
            for (std::uint32_t i = 0; i < take; ++i) {
                const auto offset = static_cast<std::size_t>(copied + i) * g.dimension;
                std::copy_n(block.key(row + i), g.dimension, keys + (values ? offset : 2 * offset));
                std::copy_n(block.value(row + i), g.dimension, values ? values + offset : keys + 2 * offset + g.dimension);
            }
        }
        copied += take;
    }
}
std::vector<KvReadSegment> KvStore::plan_read(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                                            KvBlock & destination, std::uint32_t) {
    read(layer, head, block, destination);
    return {};
}
Half * KvBlock::key(std::size_t row) {
    return storage() + row * dimension * (layout == KvLayout::interleaved ? 2 : 1);
}
void KvBlock::bind_shared(Half * address, std::size_t elements, std::shared_ptr<void> owner) {
    if (!address || !elements || !owner) throw std::invalid_argument("shared KV storage requires an address and owner");
    shared=std::make_shared<SharedStorage>(SharedStorage{address,elements,std::move(owner)});
    std::vector<Half>().swap(data);
}
void KvBlock::clear_storage(std::size_t elements) {
    if (shared) {
        if (elements!=shared->elements) throw std::invalid_argument("shared KV storage shape mismatch");
        std::fill_n(shared->address,elements,Half{0});
    } else data.assign(elements,0);
}
Half * KvBlock::value(std::size_t row) {
    return key(row) + (layout == KvLayout::interleaved ? dimension : static_cast<std::size_t>(capacity) * dimension);
}
const Half * KvBlock::key(std::size_t row) const { return const_cast<KvBlock *>(this)->key(row); }
const Half * KvBlock::value(std::size_t row) const { return const_cast<KvBlock *>(this)->value(row); }
void KvStore::append(std::uint32_t layer, std::uint64_t position,
                     const std::vector<Half> & keys, const std::vector<Half> & values) {
    if (keys.size() != static_cast<std::size_t>(geometry_.heads) * geometry_.dimension || values.size() != keys.size()) {
        throw std::invalid_argument("KV append requires complete heads");
    }
    std::vector<Half> kv(keys.size() * 2);
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        std::copy_n(keys.data() + head * geometry_.dimension, geometry_.dimension, kv.data() + head * 2 * geometry_.dimension);
        std::copy_n(values.data() + head * geometry_.dimension, geometry_.dimension, kv.data() + (head * 2 + 1) * geometry_.dimension);
    }
    append_interleaved(layer, position, kv);
}
void KvStore::validate_append(std::uint32_t layer, std::uint64_t position, std::uint64_t current,
                             const std::vector<Half> & kv) const {
    if (layer >= geometry_.layers || position != current || position >= geometry_.capacity_tokens ||
        kv.size() != static_cast<std::uint64_t>(geometry_.heads) * 2 * geometry_.dimension) {
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
    result.layout = geometry_.layout;
    result.dimension = geometry_.dimension;
    result.capacity = geometry_.block_tokens;
    result.clear_storage(static_cast<std::size_t>(geometry_.block_tokens) * 2 * geometry_.dimension);
}
MemoryKvStore::MemoryKvStore(KvGeometry geometry) : KvStore(geometry), data_(geometry.layers), lengths_(geometry.layers, 0) {
    const auto elements = static_cast<std::size_t>(geometry.capacity_tokens) * geometry.heads * 2 * geometry.dimension;
    for (auto & data : data_) data.reserve(elements);
}
void MemoryKvStore::append_interleaved(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv) {
    const auto current = length(layer);
    validate_append(layer, position, current, kv);
    auto & data = data_[layer];
    data.insert(data.end(), kv.begin(), kv.end());
    ++lengths_[layer];
}
void MemoryKvStore::read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & result) {
    prepare_block(layer, head, block, length(layer), result);
    for (std::uint32_t i = 0; i < result.valid_tokens; ++i) {
        const auto source = ((result.first_position + i) * geometry_.heads + head) * 2 * geometry_.dimension;
        std::copy_n(data_[layer].data() + source, geometry_.dimension, result.key(i));
        std::copy_n(data_[layer].data() + source + geometry_.dimension, geometry_.dimension, result.value(i));
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
    scratch_.reserve(pending_tokens_ * token_elements_);
    file_->resize(geometry.file_bytes());
    std::array<unsigned char, header_bytes> header{};
    std::memcpy(header.data(), "SOLIDKV1", 8);
    const std::uint64_t fields[] = {1, header_bytes, geometry.layers, geometry.heads, geometry.dimension,
        geometry.block_tokens, geometry.capacity_tokens, geometry.alignment, static_cast<std::uint32_t>(geometry.layout),
        sizeof(Half), geometry.contiguous_planes};
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
            const auto start = std::chrono::steady_clock::now();
            const auto n = file_->read_at(data + done, bytes - done, offset + done);
            stats_.read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
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
            const auto start = std::chrono::steady_clock::now();
            const auto n = file_->write_at(data + done, bytes - done, offset + done);
            stats_.write_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            ++stats_.write_requests;
            if (n == 0 || n > bytes - done) throw std::runtime_error("incomplete KV write at " + std::to_string(offset + done));
            stats_.write_bytes += n;
            done += n;
        }
    } catch (...) { failed_ = true; throw; }
}
void FileKvStore::append_interleaved(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv) {
    check_ready();
    validate_append(layer, position, length(layer), kv);
    if (writing_layer_ == layer) throw std::logic_error("KV append before the layer's write completed");
    auto & pending = pending_.at(layer);
    if (pending.values.size() / token_elements_ == pending_tokens_) flush_layer(layer);
    if (pending.values.empty()) pending.first = position;
    pending.values.insert(pending.values.end(), kv.begin(), kv.end());
    ++lengths_[layer];
}
void FileKvStore::flush_layer(std::uint32_t layer) {
    if (pending_.at(layer).values.empty()) return;
    trace::Scope phase("CPU->SSD Store", layer);
    const auto segments = begin_write(layer);
    try {
        for (const auto & segment : segments) write_exact(segment.source, segment.bytes, segment.file_offset);
    } catch (...) { complete_write(layer, false); throw; }
    complete_write(layer, true);
}
bool FileKvStore::write_buffer_full(std::uint32_t layer) const {
    check_ready();
    return pending_.at(layer).values.size() / token_elements_ == pending_tokens_;
}
std::vector<KvWriteSegment> FileKvStore::begin_write(std::uint32_t layer) {
    check_ready();
    if (writing_layer_) throw std::logic_error("KV write workspace is in use");
    const auto & pending = pending_.at(layer);
    if (pending.values.empty()) throw std::logic_error("cannot start an empty KV write");
    const auto count = pending.values.size() / token_elements_;
    const auto dimension = geometry_.dimension;
    scratch_.resize(pending.values.size());
    std::vector<KvWriteSegment> segments;
    auto * packed = scratch_.data();
    for (std::size_t first = 0; first < count;) {
        const auto position = pending.first + first;
        const auto in_block = position % geometry_.block_tokens;
        const auto take = geometry_.contiguous_planes ? count - first :
            std::min<std::size_t>(count - first, geometry_.block_tokens - in_block);
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            const auto offset = geometry_.block_offset(layer, head, position / geometry_.block_tokens);
            if (geometry_.layout == KvLayout::separate) {
                for (std::size_t i = 0; i < take; ++i) {
                    const auto source = (first + i) * token_elements_ + head * 2 * dimension;
                    std::copy_n(pending.values.data() + source, dimension, packed + i * dimension);
                    std::copy_n(pending.values.data() + source + dimension, dimension, packed + (take + i) * dimension);
                }
                const auto bytes = take * dimension * sizeof(Half);
                segments.push_back({offset + in_block * dimension * sizeof(Half), packed, bytes});
                segments.push_back({offset + geometry_.value_offset() + in_block * dimension * sizeof(Half),
                                    packed + take * dimension, bytes});
            } else {
                for (std::size_t i = 0; i < take; ++i) {
                    std::copy_n(pending.values.data() + (first + i) * token_elements_ + head * 2 * dimension,
                                2 * dimension, packed + i * 2 * dimension);
                }
                segments.push_back({offset + in_block * 2 * dimension * sizeof(Half), packed, take * 2 * dimension * sizeof(Half)});
            }
            packed += take * 2 * dimension;
        }
        first += take;
    }
    writing_layer_ = layer;
    return segments;
}
void FileKvStore::complete_write(std::uint32_t layer, bool success) {
    if (writing_layer_ != layer) throw std::logic_error("KV write receipt has a different layer");
    writing_layer_.reset();
    if (!success) { failed_ = true; return; }
    auto & pending = pending_.at(layer);
    committed_[layer] = pending.first + pending.values.size() / token_elements_;
    pending.values.clear();
    pending.first = committed_[layer];
}
void FileKvStore::read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & result) {
    const auto segments = plan_read(layer, head, block, result);
    for (const auto & segment : segments) {
        read_exact(reinterpret_cast<unsigned char *>(result.storage()) + segment.buffer_offset, segment.bytes, segment.file_offset);
    }
}
void FileKvStore::read_rows(std::uint32_t layer, std::uint32_t head, std::uint64_t first, std::uint32_t count,
                            Half * keys, Half * values, std::vector<Half> & scratch) {
    const auto & g = geometry_;
    const bool packed = g.layout == KvLayout::interleaved && g.slot_bytes() == 2 * g.plane_bytes() && !values;
    if (!g.contiguous_planes && !packed) {
        KvStore::read_rows(layer, head, first, count, keys, values, scratch);
        return;
    }
    const auto size = length(layer);
    if (!keys || head >= g.heads || first > size || count > size - first) throw std::invalid_argument("KV row range");
    if (!count) return;
    const auto committed = committed_[layer] > first ? std::min<std::uint64_t>(count, committed_[layer] - first) : 0;
    if (committed) {
        const auto offset = g.block_offset(layer, head, first / g.block_tokens) +
            (first % g.block_tokens) * g.dimension * sizeof(Half) * (packed ? 2 : 1);
        const auto elements = static_cast<std::size_t>(committed) * g.dimension;
        const auto bytes = elements * sizeof(Half);
        if (packed) read_exact(keys, 2 * bytes, offset);
        else if (values) {
            read_exact(keys, bytes, offset);
            read_exact(values, bytes, offset + g.value_offset());
        } else {
            scratch.resize(2 * elements);
            read_exact(scratch.data(), bytes, offset);
            read_exact(scratch.data() + elements, bytes, offset + g.value_offset());
            for (std::size_t i = 0; i < committed; ++i) {
                std::copy_n(scratch.data() + i * g.dimension, g.dimension, keys + i * 2 * g.dimension);
                std::copy_n(scratch.data() + elements + i * g.dimension, g.dimension, keys + (i * 2 + 1) * g.dimension);
            }
        }
    }
    const auto & pending = pending_[layer];
    for (auto i = committed; i < count; ++i) {
        const auto source = (first + i - pending.first) * token_elements_ + head * 2 * g.dimension;
        if (source + 2 * g.dimension > pending.values.size()) throw std::logic_error("KV dirty range mismatch");
        const auto offset = static_cast<std::size_t>(i) * g.dimension;
        if (values) {
            std::copy_n(pending.values.data() + source, g.dimension, keys + offset);
            std::copy_n(pending.values.data() + source + g.dimension, g.dimension, values + offset);
        } else std::copy_n(pending.values.data() + source, 2 * g.dimension, keys + 2 * offset);
    }
}
void FileKvStore::read_dense_layer(std::uint32_t layer, std::uint32_t count, std::uint32_t slots, Half * destination,
                                    AsyncReader * reader) {
    check_ready();
    const auto & g = geometry_;
    if (!g.contiguous_planes || !destination || count > slots || count != length(layer)) {
        throw std::invalid_argument("invalid dense KV layer read");
    }
    const auto stride = static_cast<std::size_t>(slots) * g.dimension;
    const auto plane = stride * g.heads;
    std::vector<KvRowRange> ranges;
    ranges.reserve(g.heads);
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        ranges.push_back({head, 0, count, destination + head * stride, destination + plane + head * stride});
    }
    read_separate_rows(layer, ranges, reader, 2 * static_cast<std::size_t>(g.heads));
}
void FileKvStore::read_separate_rows(std::uint32_t layer, const std::vector<KvRowRange> & ranges, AsyncReader * reader,
                                     std::size_t depth) {
    if (ranges.empty()) return;
    const auto & g = geometry_;
    const auto size = length(layer);
    if (!g.contiguous_planes || !depth) throw std::invalid_argument("invalid separate KV read batch");
    if (!reader || native_handle() < 0) {
        std::vector<Half> scratch;
        for (const auto & range : ranges) read_rows(layer, range.head, range.first, range.count, range.keys, range.values, scratch);
        return;
    }
    const auto committed = committed_[layer];
    const auto start = std::chrono::steady_clock::now();
    std::vector<KvReadRequest> requests;
    requests.reserve(depth);
    const auto finish_batch = [&] {
        if (requests.empty()) return;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto & read = requests[i];
            if (!reader->submit(i + 1, read.fd, read.buffer, read.bytes, read.offset)) throw std::runtime_error("KV read queue capacity");
        }
        reader->flush();
        for (std::size_t i = 0; i < requests.size(); ++i) {
            ReadCompletion receipt{};
            if (!reader->poll(receipt, true) || !receipt.id || receipt.id > requests.size()) throw std::runtime_error("missing KV read completion");
            const auto & read = requests[receipt.id - 1];
            const auto bytes = read.scratch ? read.logical_bytes : read.bytes;
            if (read.finish(receipt.result) != static_cast<std::int64_t>(bytes)) throw std::runtime_error("incomplete KV read");
            ++stats_.read_requests;
            stats_.read_bytes += bytes;
        }
        reader->acknowledge_notifications();
        requests.clear();
    };
    try {
        for (const auto & range : ranges) {
            const auto [head, first, count, keys, values] = range;
            if (!keys || !values || head >= g.heads || first > size || count > size - first) throw std::invalid_argument("KV row range");
            const auto disk = committed > first ? std::min<std::uint64_t>(count, committed - first) : 0;
            if (disk) {
                const auto offset = g.block_offset(layer, head, first / g.block_tokens) +
                    (first % g.block_tokens) * g.dimension * sizeof(Half);
                const auto bytes = disk * g.dimension * sizeof(Half);
                requests.push_back(prepare_read(keys, bytes, offset));
                if (requests.size() == depth) finish_batch();
                requests.push_back(prepare_read(values, bytes, offset + g.value_offset()));
                if (requests.size() == depth) finish_batch();
            }
            const auto & pending = pending_[layer];
            for (auto i = disk; i < count; ++i) {
                const auto source = (first + i - pending.first) * token_elements_ + head * 2 * g.dimension;
                if (source + 2 * g.dimension > pending.values.size()) throw std::logic_error("KV dirty range mismatch");
                std::copy_n(pending.values.data() + source, g.dimension, keys + i * g.dimension);
                std::copy_n(pending.values.data() + source + g.dimension, g.dimension, values + i * g.dimension);
            }
        }
        finish_batch();
        stats_.read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    } catch (...) {
        failed_ = true;
        // Keep the aligned buffers alive until the original read completions retire.
        try { reader->drain(); } catch (...) { std::terminate(); }
        throw;
    }

}
std::vector<KvReadSegment> FileKvStore::plan_read(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                                                KvBlock & result, std::uint32_t prefix) {
    check_ready();
    if (prefix == 0) prepare_block(layer, head, block, length(layer), result);
    else {
        const auto first = block * geometry_.block_tokens;
        (void) geometry_.block_offset(layer, head, block);
        if (result.first_position != first || result.dimension != geometry_.dimension || result.layout != geometry_.layout ||
            result.capacity != geometry_.block_tokens || result.storage_size() != static_cast<std::size_t>(geometry_.block_tokens) * 2 * geometry_.dimension ||
            prefix > result.valid_tokens || first >= length(layer)) {
            throw std::invalid_argument("invalid KV read prefix");
        }
        result.valid_tokens = static_cast<std::uint32_t>(std::min<std::uint64_t>(geometry_.block_tokens, length(layer) - first));
        result.version = result.valid_tokens;
    }
    std::vector<KvReadSegment> segments;
    const auto dimension = geometry_.dimension;
    const auto offset = geometry_.block_offset(layer, head, block);
    const auto committed = committed_[layer] <= result.first_position ? 0 :
        std::min<std::uint64_t>(result.valid_tokens, committed_[layer] - result.first_position);
    if (committed > prefix) {
        const auto bytes = (committed - prefix) * dimension * sizeof(Half);
        if (geometry_.layout == KvLayout::separate) {
            segments.push_back({offset + prefix * dimension * sizeof(Half), prefix * dimension * sizeof(Half), static_cast<std::size_t>(bytes)});
            segments.push_back({offset + geometry_.value_offset() + prefix * dimension * sizeof(Half),
                                (static_cast<std::size_t>(geometry_.block_tokens) + prefix) * dimension * sizeof(Half), static_cast<std::size_t>(bytes)});
        } else {
            segments.push_back({offset + prefix * 2 * dimension * sizeof(Half), prefix * 2 * dimension * sizeof(Half), static_cast<std::size_t>(bytes * 2)});
        }
    }
    const auto & pending = pending_[layer];
    for (auto i = static_cast<std::uint32_t>(std::max<std::uint64_t>(committed, prefix)); i < result.valid_tokens; ++i) {
        const auto source = (result.first_position + i - pending.first) * token_elements_ + head * 2 * dimension;
        if (source + 2 * dimension > pending.values.size()) throw std::logic_error("KV dirty range mismatch");
        std::copy_n(pending.values.data() + source, dimension, result.key(i));
        std::copy_n(pending.values.data() + source + dimension, dimension, result.value(i));
    }
    return segments;
}
void FileKvStore::flush() {
    check_ready();
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) flush_layer(layer);
}
void FileKvStore::flush_full_layer(std::uint32_t layer) {
    check_ready();
    if (pending_.at(layer).values.size() / token_elements_ == pending_tokens_) flush_layer(layer);
}
std::uint64_t FileKvStore::length(std::uint32_t layer) const { check_ready(); return lengths_.at(layer); }
std::uint64_t FileKvStore::resident_bytes() const {
    return (lengths_.capacity() + committed_.capacity()) * sizeof(std::uint64_t) +
        pending_.capacity() * sizeof(Pending) + scratch_.capacity() * sizeof(Half) + write_buffer_bytes();
}
std::uint64_t FileKvStore::write_buffer_bytes() const {
    std::uint64_t total = 0;
    for (const auto & pending : pending_) total += pending.values.capacity() * sizeof(Half);
    return total;
}
} // namespace solid
