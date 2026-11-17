#include "solid/block_ranker.hpp"
#include "solid/ranking_math.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace solid {
namespace {
void checked(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > SIZE_MAX / b) throw std::overflow_error("CUDA ranker allocation overflow");
    return a * b;
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        checked(cudaGetDevice(&previous_), "get CUDA ranker device");
        if (previous_ != device) { checked(cudaSetDevice(device), "select CUDA ranker device"); changed_ = true; }
    }
    ~DeviceScope() { if (changed_) (void) cudaSetDevice(previous_); }
private:
    int previous_ = 0;
    bool changed_ = false;
};

// Each thread scores one (head, candidate), with the same FP64 accumulation
// order as RepresentativeIndex::score. Q is [query_head, D], K is FP16
// [logical_block, kv_head, representative, D]. No shared reduction changes
// the within-score order; FMA contraction is disabled for this target.
__global__ void score_blocks(const __half * keys, const float * query, const std::uint64_t * ids,
                             double * scores, int * error, std::uint64_t count, std::uint32_t heads,
                             std::uint32_t group, std::uint32_t representatives, std::uint32_t dimension) {
    const auto total = count * heads;
    for (auto item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < total; item += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto head = item / count;
        const auto candidate = item % count;
        const auto base = (ids[candidate] * heads + head) * representatives * dimension;
        double value = 0;
        for (std::uint32_t r = 0; r < representatives; ++r) {
            for (std::uint32_t q = 0; q < group; ++q) {
                for (std::uint32_t d = 0; d < dimension; ++d) {
                    value += static_cast<double>(query[(head * group + q) * dimension + d]) *
                             __half2float(keys[base + r * dimension + d]) / group;
                }
            }
        }
        if (!isfinite(value)) { atomicExch(error, 1); value = -CUDART_INF; }
        scores[item] = value;
    }
}

// One CTA per head. Select only enough blocks for the remaining token budget;
// costs exclude deterministic init/local positions. Every winning block keeps
// its logical ID, independent of cache slot order. At most budget iterations,
// normally ceil(remaining / block_tokens), with one reduction per iteration.
__global__ void select_top_blocks(double * scores, const std::uint64_t * ids, const std::uint32_t * costs,
                                  std::uint64_t * output, std::uint64_t count, std::uint32_t remaining,
                                  std::uint32_t stride) {
    __shared__ RankChoice choices[256];
    __shared__ std::uint32_t left;
    const auto head = blockIdx.x;
    const auto thread = threadIdx.x;
    if (!thread) left = remaining;
    __syncthreads();
    for (std::uint32_t rank = 0; rank < stride && left; ++rank) {
        RankChoice best{0, UINT64_MAX, UINT64_MAX};
        for (std::uint64_t i = thread; i < count; i += blockDim.x) {
            const auto score = scores[static_cast<std::uint64_t>(head) * count + i];
            if (isfinite(score)) best = better_choice(best, {score, ids[i], i});
        }
        choices[thread] = best;
        __syncthreads();
        for (unsigned offset = blockDim.x / 2; offset; offset /= 2) {
            if (thread < offset) choices[thread] = better_choice(choices[thread], choices[thread + offset]);
            __syncthreads();
        }
        if (!thread) {
            const auto winner = choices[0];
            if (winner.candidate == UINT64_MAX) left = 0;
            else {
                output[static_cast<std::uint64_t>(head) * stride + rank] = winner.block;
                scores[static_cast<std::uint64_t>(head) * count + winner.candidate] = -CUDART_INF;
                left -= min(left, costs[winner.candidate]);
            }
        }
        __syncthreads();
    }
}

class CudaBlockRanker final : public BlockRanker {
public:
    CudaBlockRanker(KvGeometry geometry, std::uint32_t query_heads, std::uint32_t representatives,
                    std::uint32_t budget, int device) :
        geometry_(geometry), query_heads_(query_heads), representatives_(representatives), budget_(budget), device_(device) {
        geometry_.validate();
        if (!query_heads || query_heads % geometry.heads || !representatives || representatives > geometry.block_tokens ||
            !budget || budget > geometry.capacity_tokens || device < 0) throw std::invalid_argument("invalid CUDA ranker geometry");
        block_elements_ = product(product(geometry.heads, representatives), geometry.dimension);
        layer_bytes_ = product(product(geometry.blocks_per_head(), block_elements_), sizeof(Half));
        banks_.resize(geometry.layers, nullptr);
        published_.resize(geometry.layers, 0);
        output_.resize(product(geometry.heads, budget));
        stats_.host_bytes = banks_.capacity() * sizeof(void *) + published_.capacity() * sizeof(std::uint64_t) +
                            output_.capacity() * sizeof(std::uint64_t) + sizeof(error_);
        DeviceScope scope(device_);
        try {
            cudaDeviceProp properties{};
            checked(cudaGetDeviceProperties(&properties, device_), "CUDA ranker properties");
            if (geometry.heads > static_cast<std::uint32_t>(properties.maxGridSize[0]) || properties.maxThreadsPerBlock < 256) {
                throw std::invalid_argument("CUDA ranker launch geometry exceeds device limits");
            }
            checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create CUDA ranker stream");
            allocate(query_, product(product(query_heads, geometry.dimension), sizeof(float)));
            allocate(ids_, product(geometry.blocks_per_head(), sizeof(std::uint64_t)));
            allocate(costs_, product(geometry.blocks_per_head(), sizeof(std::uint32_t)));
            allocate(scores_, product(product(geometry.heads, geometry.blocks_per_head()), sizeof(double)));
            allocate(output_device_, product(output_.size(), sizeof(std::uint64_t)));
            allocate(error_device_, sizeof(int));
        } catch (...) { release(); throw; }
    }
    ~CudaBlockRanker() override { release(); }
    void publish(std::uint32_t layer, std::uint64_t first, std::uint64_t count, const std::vector<Half> & keys) override {
        check();
        if (layer >= geometry_.layers || first != published_.at(layer) || !count ||
            count > geometry_.blocks_per_head() - first || keys.size() != product(count, block_elements_)) {
            throw std::invalid_argument("CUDA representative publication must append complete immutable blocks");
        }
        for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite CUDA representative key");
        DeviceScope scope(device_);
        try {
            if (!banks_[layer]) allocate(banks_[layer], layer_bytes_);
            const auto offset = product(first, block_elements_);
            checked(cudaMemcpyAsync(static_cast<Half *>(banks_[layer]) + offset, keys.data(), product(keys.size(), sizeof(Half)),
                                    cudaMemcpyHostToDevice, stream_), "upload CUDA representatives");
            checked(cudaStreamSynchronize(stream_), "complete CUDA representative upload");
            published_[layer] += count;
            stats_.published_blocks += count;
        } catch (...) { fail(); throw; }
    }
    std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                const BlockCandidates & candidates) override {
        check();
        if (layer >= geometry_.layers || query.size() != product(query_heads_, geometry_.dimension) ||
            candidates.blocks.size() != candidates.tokens.size() || candidates.blocks.size() > geometry_.blocks_per_head() ||
            candidates.remaining > budget_) throw std::invalid_argument("CUDA ranking shape or budget");
        for (const auto q : query) if (!std::isfinite(q)) throw std::invalid_argument("non-finite CUDA ranking query");
        for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
            if (candidates.blocks[i] >= published_[layer] || (i && candidates.blocks[i] <= candidates.blocks[i-1]) ||
                !candidates.tokens[i] || candidates.tokens[i] > geometry_.block_tokens) throw std::invalid_argument("invalid CUDA rank candidate");
        }
        std::vector<std::vector<std::uint64_t>> result(geometry_.heads);
        if (!candidates.remaining || candidates.blocks.empty()) return result;
        const auto count = candidates.blocks.size();
        const auto stride = static_cast<std::uint32_t>(std::min<std::size_t>(candidates.remaining, count));
        const auto bytes = product(product(geometry_.heads, stride), sizeof(std::uint64_t));
        DeviceScope scope(device_);
        try {
            checked(cudaMemcpyAsync(query_, query.data(), product(query.size(), sizeof(float)), cudaMemcpyHostToDevice, stream_), "upload rank query");
            checked(cudaMemcpyAsync(ids_, candidates.blocks.data(), product(count, sizeof(std::uint64_t)), cudaMemcpyHostToDevice, stream_), "upload rank candidates");
            checked(cudaMemcpyAsync(costs_, candidates.tokens.data(), product(count, sizeof(std::uint32_t)), cudaMemcpyHostToDevice, stream_), "upload rank costs");
            checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear rank status");
            checked(cudaMemsetAsync(output_device_, 0xff, bytes, stream_), "clear rank output");
            const auto grid = static_cast<unsigned>(std::min<std::uint64_t>((count * geometry_.heads + 255) / 256, 65535));
            score_blocks<<<grid, 256, 0, stream_>>>(static_cast<const __half *>(banks_[layer]), query_, ids_, scores_, error_device_,
                count, geometry_.heads, query_heads_ / geometry_.heads, representatives_, geometry_.dimension);
            checked(cudaGetLastError(), "launch block relevance kernel");
            select_top_blocks<<<geometry_.heads, 256, 0, stream_>>>(scores_, ids_, costs_, output_device_, count, candidates.remaining, stride);
            checked(cudaGetLastError(), "launch block top-k kernel");
            checked(cudaMemcpyAsync(output_.data(), output_device_, bytes, cudaMemcpyDeviceToHost, stream_), "download selected block IDs");
            checked(cudaMemcpyAsync(&error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download rank status");
            checked(cudaStreamSynchronize(stream_), "complete CUDA block ranking");
            if (error_) throw std::runtime_error("non-finite CUDA block relevance");
        } catch (...) { fail(); throw; }
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            for (std::uint32_t i = 0; i < stride; ++i) {
                const auto block = output_[static_cast<std::size_t>(head) * stride + i];
                if (block == UINT64_MAX) break;
                result[head].push_back(block);
            }
        }
        ++stats_.calls;
        stats_.scored_blocks += count * geometry_.heads;
        return result;
    }
    RankerStats stats() const override { return stats_; }
    const char * name() const override { return "cuda_fp64_topk"; }
private:
    template<class T> void allocate(T *& pointer, std::size_t bytes) {
        checked(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes), "allocate CUDA ranker buffer");
        stats_.device_bytes += bytes;
    }
    void check() const { if (failed_) throw std::runtime_error("CUDA block ranker failed"); }
    void fail() noexcept { failed_ = true; if (stream_) (void) cudaStreamSynchronize(stream_); }
    void release() noexcept {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device_) != cudaSuccess) return;
        if (stream_) (void) cudaStreamSynchronize(stream_);
        for (auto * bank : banks_) if (bank) (void) cudaFree(bank);
        if (query_) (void) cudaFree(query_);
        if (ids_) (void) cudaFree(ids_);
        if (costs_) (void) cudaFree(costs_);
        if (scores_) (void) cudaFree(scores_);
        if (output_device_) (void) cudaFree(output_device_);
        if (error_device_) (void) cudaFree(error_device_);
        if (stream_) (void) cudaStreamDestroy(stream_);
        if (restore && previous != device_) (void) cudaSetDevice(previous);
    }
    KvGeometry geometry_;
    std::uint32_t query_heads_, representatives_, budget_;
    int device_;
    std::size_t block_elements_ = 0, layer_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    std::vector<void *> banks_;
    std::vector<std::uint64_t> published_, output_;
    float * query_ = nullptr;
    std::uint64_t * ids_ = nullptr;
    std::uint32_t * costs_ = nullptr;
    double * scores_ = nullptr;
    std::uint64_t * output_device_ = nullptr;
    int * error_device_ = nullptr;
    int error_ = 0;
    bool failed_ = false;
    RankerStats stats_;
};
} // namespace
std::unique_ptr<BlockRanker> make_cuda_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, int device) {
    return std::make_unique<CudaBlockRanker>(geometry, query_heads, representatives, budget, device);
}
} // namespace solid
