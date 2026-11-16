#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace solid {
using Half = std::uint16_t;
Half to_half(float value);
float from_half(Half value);

enum class KvLayout : std::uint32_t { separate = 0, interleaved = 1 };

struct KvGeometry {
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t dimension = 0;
    std::uint32_t block_tokens = 32;
    std::uint32_t capacity_tokens = 0;
    std::uint32_t alignment = 4096;
    KvLayout layout = KvLayout::separate;

    void validate() const;
    std::uint64_t blocks_per_head() const;
    std::uint64_t plane_bytes() const;
    std::uint64_t slot_bytes() const;
    std::uint64_t file_bytes() const;
    std::uint64_t block_offset(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const;
};

struct KvBlock {
    std::uint64_t first_position = 0;
    std::uint32_t valid_tokens = 0;
    std::uint64_t version = 0;
    std::vector<Half> keys;
    std::vector<Half> values;
};

struct IoStats {
    std::uint64_t read_requests = 0;
    std::uint64_t write_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
};

class RandomAccessFile {
public:
    virtual ~RandomAccessFile() = default;
    virtual std::size_t read_at(void * destination, std::size_t bytes, std::uint64_t offset) = 0;
    virtual std::size_t write_at(const void * source, std::size_t bytes, std::uint64_t offset) = 0;
    virtual void resize(std::uint64_t bytes) = 0;
};

std::unique_ptr<RandomAccessFile> create_session_file(const std::filesystem::path & path, bool remove_on_close);

class KvStore {
public:
    explicit KvStore(KvGeometry geometry);
    virtual ~KvStore() = default;
    const KvGeometry & geometry() const { return geometry_; }
    virtual void append(std::uint32_t layer, std::uint64_t position,
                        const std::vector<Half> & keys, const std::vector<Half> & values) = 0;
    virtual void read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & destination) = 0;
    virtual void flush() = 0;
    virtual std::uint64_t length(std::uint32_t layer) const = 0;
    virtual std::uint64_t resident_bytes() const = 0;
    virtual IoStats io_stats() const { return {}; }
protected:
    KvGeometry geometry_;
    void validate_append(std::uint32_t layer, std::uint64_t position, std::uint64_t length,
                         const std::vector<Half> & keys, const std::vector<Half> & values) const;
    void prepare_block(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                       std::uint64_t length, KvBlock & destination) const;
};

class MemoryKvStore final : public KvStore {
public:
    explicit MemoryKvStore(KvGeometry geometry);
    void append(std::uint32_t layer, std::uint64_t position,
                const std::vector<Half> & keys, const std::vector<Half> & values) override;
    void read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & destination) override;
    void flush() override {}
    std::uint64_t length(std::uint32_t layer) const override;
    std::uint64_t resident_bytes() const override;
private:
    std::vector<std::vector<Half>> data_;
    std::vector<std::uint64_t> lengths_;
};

class FileKvStore final : public KvStore {
public:
    FileKvStore(KvGeometry geometry, std::unique_ptr<RandomAccessFile> file,
                std::size_t write_buffer_bytes_per_layer = 32768);
    void append(std::uint32_t layer, std::uint64_t position,
                const std::vector<Half> & keys, const std::vector<Half> & values) override;
    void read(std::uint32_t layer, std::uint32_t head, std::uint64_t block, KvBlock & destination) override;
    void flush() override;
    std::uint64_t length(std::uint32_t layer) const override;
    std::uint64_t resident_bytes() const override;
    IoStats io_stats() const override { return stats_; }
    bool failed() const { return failed_; }
private:
    struct Pending {
        std::uint64_t first = 0;
        std::vector<Half> values;
    };
    std::unique_ptr<RandomAccessFile> file_;
    std::vector<Pending> pending_;
    std::vector<std::uint64_t> lengths_;
    std::vector<std::uint64_t> committed_;
    std::vector<Half> scratch_;
    std::size_t token_elements_;
    std::size_t pending_tokens_;
    IoStats stats_;
    bool failed_ = false;

    void check_ready() const;
    void flush_layer(std::uint32_t layer);
    void read_exact(void * destination, std::size_t bytes, std::uint64_t offset);
    void write_exact(const void * source, std::size_t bytes, std::uint64_t offset);
};
} // namespace solid
