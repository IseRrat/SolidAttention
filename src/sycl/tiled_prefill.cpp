// SYCL 分片 Prefill：逐片累积 attention，管理设备输入、输出与工作区。
#include "solid/tiled_prefill.hpp"
#include "solid/sycl_runtime.hpp"

#include <algorithm>
#include <cmath>

namespace solid {
namespace {
using sycl_runtime::Buffer;
using sycl_runtime::flag_invalid;
using sycl_runtime::product;

constexpr std::size_t lanes = 32;
constexpr float negative_inf = -std::numeric_limits<float>::infinity();

struct PrefillTile {
    const float * query;
    const sycl::half * kv;
    float * numerator, * maximum, * denominator;
    int * status;
    std::uint64_t first_query, first_key;
    std::size_t token_stride, head_stride;
    std::uint32_t valid, query_heads, group, dimension;
    float scale;
};

template<std::size_t Width> void accumulate(sycl::queue & queue, std::size_t rows, PrefillTile tile) {
    constexpr auto fragments = (Width + lanes - 1) / lanes;
    queue.parallel_for(sycl::nd_range<1>(product(rows, lanes), lanes), [=](sycl::nd_item<1> item) {
        const auto row = item.get_group_linear_id();
        const auto lane = item.get_local_linear_id();
        const auto position = tile.first_query + row / tile.query_heads;
        if (tile.first_key > position) return;
        const auto kv_head = (row % tile.query_heads) / tile.group;
        const auto width = tile.dimension;
        auto * numerator = tile.numerator + row * width;
        auto * query = tile.query + row * width;
        float cached_query[fragments], values[fragments];
        for (std::size_t d = lane; d < width; d += lanes) {
            cached_query[d / lanes] = query[d];
            values[d / lanes] = numerator[d];
        }
        auto maximum = tile.maximum[row];
        auto denominator = tile.denominator[row];
        const auto count = sycl::min<std::uint64_t>(tile.valid, position - tile.first_key + 1);
        for (std::size_t token = 0; token < count; ++token) {
            const auto * kv = tile.kv + kv_head * tile.head_stride + token * tile.token_stride;
            float dot = 0;
            for (std::size_t d = lane; d < width; d += lanes) dot = sycl::fma(cached_query[d / lanes], static_cast<float>(kv[d]), dot);
            const float score = sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>()) * tile.scale;
            if (!sycl::isfinite(score)) {
                if (!lane) flag_invalid(tile.status);
                return;
            }
            const auto next = sycl::fmax(maximum, score);
            const auto keep = sycl::exp(maximum - next);
            const auto weight = sycl::exp(score - next);
            for (std::size_t d = lane; d < width; d += lanes) {
                values[d / lanes] = keep * values[d / lanes] + weight * static_cast<float>(kv[width + d]);
            }
            denominator = keep * denominator + weight;
            maximum = next;
        }
        for (std::size_t d = lane; d < width; d += lanes) numerator[d] = values[d / lanes];
        if (!lane) {
            tile.maximum[row] = maximum;
            tile.denominator[row] = denominator;
        }
    });
}

class SyclTiledPrefill final : public TiledPrefill {
public:
    SyclTiledPrefill(KvGeometry geometry, std::uint32_t query_heads, std::uint32_t queries, std::uint32_t tile_tokens,
                     const sycl::queue & backend) :
        queue_(sycl_runtime::in_order_queue(backend)), geometry_(geometry), query_heads_(query_heads), queries_(queries) {
        geometry_.validate();
        if (!query_heads || query_heads % geometry_.heads || !queries || queries > geometry_.capacity_tokens ||
            !tile_tokens || geometry_.dimension > 1024) throw std::invalid_argument("invalid SYCL tiled prefill geometry");
        const auto device = queue_.get_device();
        if (!device.has(sycl::aspect::fp16) || !device.has(sycl::aspect::usm_device_allocations) ||
            device.get_info<sycl::info::device::max_work_group_size>() < lanes) throw std::invalid_argument("SYCL prefill device capabilities");
        const auto block = static_cast<std::uint64_t>(geometry_.block_tokens);
        const auto tile = (std::min(tile_tokens, geometry_.capacity_tokens) + block - 1) / block * block;
        if (tile > UINT32_MAX) throw std::overflow_error("SYCL prefill tile capacity");
        tile_ = static_cast<std::uint32_t>(tile);
        tile_elements_ = device_elements(tile_, geometry_.heads, geometry_.dimension, 2);
        const auto rows = product(queries_, query_heads_);
        const auto elements = product(rows, geometry_.dimension);
        kv_ = allocate<Half>(tile_elements_, stats_.kv_device_bytes);
        query_ = allocate<float>(elements, stats_.workspace_device_bytes);
        numerator_ = allocate<float>(elements, stats_.workspace_device_bytes);
        maximum_ = allocate<float>(rows, stats_.workspace_device_bytes);
        denominator_ = allocate<float>(rows, stats_.workspace_device_bytes);
        status_ = allocate<int>(1, stats_.workspace_device_bytes);
        stats_.tile_capacity = tile_;
        stats_.query_capacity = queries_;
        stats_.kv_device_peak_bytes = stats_.kv_device_bytes;
        stats_.workspace_device_peak_bytes = stats_.workspace_device_bytes;
    }

    ~SyclTiledPrefill() override { queue_.wait(); }
    std::uint32_t query_capacity() const override { return queries_; }
    std::uint32_t tile_capacity() const override { return tile_; }
    std::optional<DeviceMemory> direct_memory() const override { return DeviceMemory::sycl; }
    const char * name() const override { return "sycl_online_tiled"; }
    PrefillStats stats() const override { return stats_; }

    void begin(std::uint64_t first, std::uint32_t queries, float scale, const std::vector<float> & query) override {
        if (query.size() != device_elements(queries, query_heads_, geometry_.dimension) ||
            !std::all_of(query.begin(), query.end(), [](float x) { return std::isfinite(x); })) {
            throw std::invalid_argument("invalid SYCL prefill query");
        }
        begin_impl(first, queries, scale, query.data(), false);
    }
    void begin_device(std::uint64_t first, std::uint32_t queries, float scale, DeviceQuery query) override {
        validate_device_query(query, DeviceMemory::sycl, query_heads_, geometry_.dimension, queries);
        sycl_runtime::validate_pointer(query.data, queue_);
        begin_impl(first, queries, scale, query.data, true);
    }
    void consume(std::uint64_t first, std::uint32_t valid, const std::vector<Half> & kv) override {
        if (kv.size() != tile_elements_) throw std::invalid_argument("SYCL prefill host tile shape");
        consume_impl(first, valid, kv.data(), false);
    }
    void consume_device(std::uint64_t first, std::uint32_t valid, DeviceKvBatch kv) override {
        validate_device_kv(kv, DeviceMemory::sycl, geometry_.heads, geometry_.dimension, valid);
        sycl_runtime::validate_pointer(kv.data, queue_);
        consume_impl(first, valid, kv.data, true);
    }
    std::vector<float> finish() override {
        check_complete();
        std::vector<float> result(product(product(count_, query_heads_), geometry_.dimension));
        finish_impl(result.data(), result.size(), false);
        return result;
    }
    void finish_to(DeviceOutput output) override {
        check_complete();
        validate_device_output(output, DeviceMemory::sycl, query_heads_, geometry_.dimension, count_);
        sycl_runtime::validate_pointer(output.data, queue_);
        finish_impl(output.data, output.elements, true);
    }
    void abort() override {
        fail();
        queue_.throw_asynchronous();
    }

private:
    void begin_impl(std::uint64_t first, std::uint32_t count, float scale, const float * query, bool direct) {
        check();
        if (open_ || !count || count > queries_ || first > geometry_.capacity_tokens - count || !std::isfinite(scale) || scale <= 0) {
            throw std::invalid_argument("SYCL prefill frame range or scale");
        }
        const auto rows = product(count, query_heads_);
        const auto elements = product(rows, geometry_.dimension);
        const auto bytes = product(elements, sizeof(float));
        submit([&] {
            sycl::event upload;
            if (!direct) upload = queue_.memcpy(query_.get(), query, bytes);
            queue_.fill(status_.get(), 0, 1);
            queue_.fill(numerator_.get(), 0.0f, elements);
            queue_.fill(maximum_.get(), negative_inf, rows);
            queue_.fill(denominator_.get(), 0.0f, rows);
            if (!direct) upload.wait_and_throw();
        });
        current_query_ = direct ? query : query_.get();
        first_ = first;
        count_ = count;
        scale_ = scale;
        next_key_ = 0;
        open_ = true;
        if (direct) stats_.device_query_tokens += count;
        else stats_.query_upload_bytes += bytes;
    }
    void consume_impl(std::uint64_t first, std::uint32_t valid, const Half * kv, bool direct) {
        check();
        const auto end = first_ + count_;
        if (!open_ || first != next_key_ || first >= end || !valid || valid > std::min<std::uint64_t>(tile_, end - first)) {
            throw std::invalid_argument("SYCL prefill tile order or shape");
        }
        const auto bytes = product(tile_elements_, sizeof(Half));
        const auto rows = product(count_, query_heads_);
        const auto width = geometry_.dimension;
        const PrefillTile tile{current_query_, reinterpret_cast<const sycl::half *>(direct ? kv : kv_.get()), numerator_.get(),
            maximum_.get(), denominator_.get(), status_.get(), first_, first,
            product(product(2, width), direct ? geometry_.heads : 1), product(product(2, width), direct ? 1 : tile_),
            valid, query_heads_, query_heads_ / geometry_.heads, width, scale_};
        submit([&] {
            sycl::event upload;
            if (!direct) upload = queue_.memcpy(kv_.get(), kv, bytes);
            if (width <= 32) accumulate<32>(queue_, rows, tile);
            else if (width <= 64) accumulate<64>(queue_, rows, tile);
            else if (width <= 128) accumulate<128>(queue_, rows, tile);
            else if (width <= 256) accumulate<256>(queue_, rows, tile);
            else if (width <= 512) accumulate<512>(queue_, rows, tile);
            else accumulate<1024>(queue_, rows, tile);
            // Return the host tile only after its data has reached the queue's
            // device buffer. The causal kernel can still be in flight.
            if (!direct) upload.wait_and_throw();
        });
        next_key_ += valid;
        ++stats_.tiles;
        stats_.kv_tokens += valid;
        if (direct) stats_.device_kv_tokens += valid;
        else stats_.upload_bytes += bytes;
    }
    void finish_impl(float * output, std::size_t elements, bool direct) {
        auto * numerator = numerator_.get();
        const auto * denominator = denominator_.get();
        auto * status = status_.get();
        const auto width = geometry_.dimension;
        const auto bytes = product(elements, sizeof(float));
        int error = 0;
        submit([&] {
            queue_.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> i) {
                const auto mass = denominator[i[0] / width];
                const auto value = numerator[i[0]] / mass;
                if (!(mass > 0) || !sycl::isfinite(value)) {
                    flag_invalid(status);
                    numerator[i[0]] = 0.0f;
                } else numerator[i[0]] = value;
            });
            queue_.memcpy(output, numerator, bytes);
            queue_.memcpy(&error, status, sizeof(error));
            queue_.wait_and_throw();
            if (error) throw std::runtime_error("non-finite or empty SYCL tiled attention");
        });
        open_ = false;
        current_query_ = nullptr;
        ++stats_.batches;
        stats_.query_tokens += count_;
        if (direct) stats_.output_device_copy_bytes += bytes;
        else stats_.output_download_bytes += bytes;
    }
    void check() const { if (failed_) throw std::runtime_error("SYCL tiled prefill failed"); }
    void check_complete() const {
        check();
        if (!open_ || next_key_ != first_ + count_) throw std::logic_error("SYCL prefill requires the complete causal prefix");
    }
    void fail() noexcept {
        failed_ = true;
        open_ = false;
        try { queue_.wait(); } catch (...) {}
        current_query_ = nullptr;
    }
    template<class Submit> void submit(Submit command) {
        try { command(); } catch (...) { fail(); throw; }
    }
    template<class T> Buffer<T> allocate(std::size_t count, std::uint64_t & bytes) {
        auto data = sycl_runtime::allocate<T>(count, queue_);
        bytes += product(count, sizeof(T));
        return data;
    }

    sycl::queue queue_;
    KvGeometry geometry_;
    std::uint32_t query_heads_, queries_, tile_ = 0, count_ = 0;
    std::size_t tile_elements_ = 0;
    std::uint64_t first_ = 0, next_key_ = 0;
    float scale_ = 1;
    bool open_ = false, failed_ = false;
    Buffer<Half> kv_;
    Buffer<float> query_, numerator_, maximum_, denominator_;
    Buffer<int> status_;
    const float * current_query_ = nullptr;
    PrefillStats stats_;
};
} // namespace

std::unique_ptr<TiledPrefill> make_sycl_tiled_prefill(const KvGeometry & geometry, std::uint32_t query_heads,
                                                    std::uint32_t queries, std::uint32_t tile_tokens, void * backend_queue) {
    if (!backend_queue) throw std::invalid_argument("SYCL prefill requires an attention queue");
    return std::make_unique<SyclTiledPrefill>(geometry, query_heads, queries, tile_tokens, *static_cast<sycl::queue *>(backend_queue));
}
} // namespace solid
