#include "solid/block_ranker.hpp"
#include "solid/trace.hpp"
#include "solid/ranking_math.hpp"
#include "solid/representative_math.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
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

// All lanes supply one product. Lane zero accumulates in dimension order,
// matching the reference even for cancellation-heavy scores and exact ties.
__device__ double append_ordered_chunk(double value,double product,unsigned valid) {
    for (unsigned source=0;source<valid;++source) {
        const auto term=__shfl_sync(0xffffffff,product,source);
        if ((threadIdx.x&31)==0) value+=term;
    }
    return value;
}

// One warp scores a (head, candidate). Adjacent lanes load adjacent Q/K
// dimensions. The representative, GQA and dimension accumulation order is
// unchanged; FMA contraction is disabled for this target.
__global__ void score_blocks(const __half * keys, const float * query, const std::uint64_t * ids,
                             double * scores, int * error, std::uint64_t count, std::uint32_t heads,
                             std::uint32_t group, std::uint32_t representatives, std::uint32_t dimension) {
    const auto total = count * heads;
    const auto lane=threadIdx.x&31;
    const auto warp=(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto warps=static_cast<std::uint64_t>(gridDim.x)*blockDim.x/32;
    for (auto item=warp;item<total;item+=warps) {
        const auto head = item / count;
        const auto candidate = item % count;
        const auto base = (ids[candidate] * heads + head) * representatives * dimension;
        double value = 0;
        for (std::uint32_t r = 0; r < representatives; ++r) {
            for (std::uint32_t q = 0; q < group; ++q) {
                for (std::uint32_t first=0;first<dimension;first+=32) {
                    const auto d=first+lane;
                    double product=0;
                    if (d<dimension) product=static_cast<double>(query[(head*group+q)*dimension+d]) *
                        __half2float(keys[base+r*dimension+d])/group;
                    value=append_ordered_chunk(value,product,min(32u,dimension-first));
                }
            }
        }
        if (!lane) {
            if (!isfinite(value)) { atomicExch(error,1); value=-CUDART_INF; }
            scores[item]=value;
        }
    }
}

// Flatten the query heads into one score per historical block, then run one
// Top-K over the shared list. GQA heads reference their existing KV rows.
__global__ void score_shared_blocks(const __half * keys, const float * query, std::uint64_t * ids,
                                    double * scores, int * error, std::uint64_t count, std::uint32_t heads,
                                    std::uint32_t group, std::uint32_t representatives, std::uint32_t dimension, bool prefix) {
    __shared__ float warps[8];
    const auto lane = threadIdx.x % 32;
    const auto warp = threadIdx.x / 32;
    const auto width = static_cast<std::uint64_t>(heads) * group * dimension;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x); i < count; i += gridDim.x) {
        const auto * block = keys + (prefix ? i : ids[i]) * heads * representatives * dimension;
        float total = 0;
        for (auto j = static_cast<std::uint64_t>(threadIdx.x); j < width; j += blockDim.x) {
            const auto head = j / dimension / group;
            const auto d = j % dimension;
            float key = 0;
            for (std::uint32_t r = 0; r < representatives; ++r) {
                key += __half2float(block[(head * representatives + r) * dimension + d]);
            }
            if (representatives > 1) key = __half2float(__float2half(key / representatives));
            total += query[j] * key;
        }
        for (unsigned offset = 16; offset; offset /= 2) total += __shfl_down_sync(0xffffffff, total, offset);
        if (!lane) warps[warp] = total;
        __syncthreads();
        if (warp == 0) {
            total = lane < 8 ? warps[lane] : 0;
            for (unsigned offset = 16; offset; offset /= 2) total += __shfl_down_sync(0xffffffff, total, offset);
            if (!lane) {
                if (!isfinite(total)) atomicExch(error, 1);
                scores[i] = static_cast<double>(total);
                if (prefix) ids[i] = i;
            }
        }
        __syncthreads();
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

__global__ void builder_mean_query(const float * query, double * mean, int * error, std::uint32_t heads,
                                    std::uint32_t width, std::uint32_t group) {
    const auto total = static_cast<std::uint64_t>(heads)*width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < total; i += static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        mean[i] = representative_mean(query, static_cast<std::uint32_t>(i/width), static_cast<std::uint32_t>(i%width), width, group);
        if (!isfinite(mean[i])) atomicExch(error, 1);
    }
}
__global__ void builder_append_keys(const Half * source, Half * target, int * error,
                                    std::size_t head_stride, std::uint32_t heads, std::uint32_t width) {
    const auto total = static_cast<std::uint64_t>(heads)*width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < total; i += static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto key = source[(i / width) * head_stride + i % width];
        if (error && !isfinite(__half2float(__ushort_as_half(key)))) atomicExch(error, 1);
        target[i] = key;
    }
}
__global__ void validate_query_values(const float * query, std::uint64_t count, int * error) {
    for (auto i = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(gridDim.x)*blockDim.x) if (!isfinite(query[i])) atomicExch(error,1);
}
struct DeviceHalf {
    __device__ float operator()(Half bits) const { return __half2float(__ushort_as_half(bits)); }
};
__global__ void builder_update_scores(const Half * keys, double * scores, const double * mean, int * error,
                                       std::uint64_t first, std::uint64_t count, std::uint32_t local,
                                       std::uint32_t heads, std::uint32_t input_heads, std::uint32_t width) {
    const auto total = count*heads;
    const auto lane=threadIdx.x&31;
    const auto warp=(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto warps=static_cast<std::uint64_t>(gridDim.x)*blockDim.x/32;
    for (auto i=warp;i<total;i+=warps) {
        const auto head = i%heads;
        const auto row = (first + i / heads) % local;
        const auto slot = row * heads + head;
        const auto key_slot = row * input_heads + head / (heads / input_heads);
        auto value=lane?0.:scores[slot];
        for (std::uint32_t first_dimension=0;first_dimension<width;first_dimension+=32) {
            const auto d=first_dimension+lane;
            double product=0;
            if (d<width) product=mean[head*width+d]*DeviceHalf{}(keys[key_slot*width+d]);
            value=append_ordered_chunk(value,product,min(32u,width-first_dimension));
        }
        if (!lane) {
            if (!isfinite(value)) atomicExch(error,1);
            scores[slot]=value;
        }
    }
}
__global__ void builder_finalize(const Half * ring, const double * token_scores, Half * top_keys,
                                  std::uint64_t * top_positions, double * top_scores, Half * bank,
                                  std::uint64_t * built_positions, double * built_scores,
                                  std::uint64_t position, std::uint32_t local, std::uint32_t heads,
                                  std::uint32_t input_heads, std::uint32_t width, std::uint32_t representatives,
                                  std::uint32_t block_tokens) {
    const auto head = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (head >= heads) return;
    const auto old = position-local;
    const auto slot = (old%local)*heads+head;
    const auto key_slot = (old % local) * input_heads + head / (heads / input_heads);
    const auto offset = head*representatives;
    insert_representative(old, token_scores[slot]/local, ring+key_slot*width, representatives, width,
                          top_positions+offset, top_scores+offset, top_keys+offset*width);
    if ((old+1)%block_tokens) return;
    const auto target = ((old/block_tokens)*heads+head)*representatives;
    for (std::uint32_t r = 0; r < representatives; ++r) {
        built_positions[offset+r] = top_positions[offset+r];
        built_scores[offset+r] = top_scores[offset+r];
        for (std::uint32_t d = 0; d < width; ++d) bank[(target+r)*width+d] = top_keys[(offset+r)*width+d];
        top_positions[offset+r] = UINT64_MAX;
        top_scores[offset+r] = 0;
    }
}

__global__ void probability_append_keys(const Half * input, Half * ring, Half * scores, int * error,
    std::uint64_t first, std::uint32_t count, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t width) {
    const auto elements = static_cast<std::uint64_t>(count) * kv_heads * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto token = i / (kv_heads * width), column = i % (kv_heads * width);
        const auto key = input[token * kv_heads * 2 * width + (column / width) * 2 * width + column % width];
        if (!isfinite(DeviceHalf{}(key))) atomicExch(error, 1);
        ring[((first + token) % capacity) * kv_heads * width + column] = key;
    }
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < static_cast<std::uint64_t>(count) * heads; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        scores[((first + i / heads) % capacity) * heads + i % heads] = 0;
    }
}
__global__ void probability_add_scores(const float * input, Half * scores, int * error,
    std::uint64_t first, std::uint32_t count, std::uint32_t heads, std::uint32_t capacity, bool key_major) {
    const auto elements = static_cast<std::uint64_t>(count) * heads;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto key = i / heads, head = i % heads;
        const auto value = input[key_major ? i : head * count + key];
        if (!isfinite(value) || value < 0) atomicExch(error, 1);
        const auto slot = ((first + key) % capacity) * heads + head;
        // Upstream get_score returns K's dtype and append_global uses add_
        // into a buffer of that same dtype. Both FP16 roundings matter.
        scores[slot] = __half_as_ushort(__hadd(__ushort_as_half(scores[slot]), __float2half(value)));
    }
}
struct ReduceRankChoice {
    __device__ RankChoice operator()(RankChoice a, RankChoice b) const { return better_choice(a, b); }
};

// InfLLM ContextManager.get_block_k: Top-K over each complete block, then
// gather its raw keys. One CTA owns one (block, query head); grouped query
// heads keep separate scores while referring to the same KV-head keys.
// CUB's stable radix sort retains the index order for equal scores.
__global__ void probability_block_topk(const Half * scores, std::uint64_t * positions,
    double * result_scores, std::uint64_t first_block, std::uint32_t initial,
    std::uint32_t capacity, std::uint32_t heads, std::uint32_t representatives, std::uint32_t block_tokens) {
    using Sort = cub::BlockRadixSort<double, 128, 1, std::uint32_t>;
    __shared__ typename Sort::TempStorage storage;
    const auto head = blockIdx.x % heads;
    const auto block = first_block + blockIdx.x / heads;
    const auto first = initial + block * block_tokens;
    double value[1] = {threadIdx.x < block_tokens ? static_cast<double>(DeviceHalf{}(scores[((first + threadIdx.x) % capacity) * heads + head])) : -CUDART_INF};
    std::uint32_t index[1] = {threadIdx.x};
    Sort(storage).SortDescending(value, index);
    if (threadIdx.x < representatives) {
        const auto result = static_cast<std::uint64_t>(blockIdx.x) * representatives + threadIdx.x;
        positions[result] = first + index[0];
        if (result_scores) result_scores[result] = value[0];
    }
}

// Keep arbitrary configured block sizes supported. Large blocks use CUB's
// parallel reduction for each winner, without serially copying key vectors.
__global__ void probability_large_block_topk(const Half * scores, std::uint64_t * positions,
    double * result_scores, std::uint64_t first_block, std::uint32_t initial,
    std::uint32_t capacity, std::uint32_t heads, std::uint32_t representatives, std::uint32_t block_tokens) {
    using Reduce = cub::BlockReduce<RankChoice, 128>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ RankChoice previous;
    const auto head = blockIdx.x % heads;
    const auto first = initial + (first_block + blockIdx.x / heads) * block_tokens;
    if (!threadIdx.x) previous = {CUDART_INF, 0, UINT64_MAX};
    __syncthreads();
    for (std::uint32_t rank = 0; rank < representatives; ++rank) {
        RankChoice best{0, UINT64_MAX, UINT64_MAX};
        for (std::uint64_t i = threadIdx.x; i < block_tokens; i += blockDim.x) {
            const auto position = first + i;
            const auto score = static_cast<double>(DeviceHalf{}(scores[(position % capacity) * heads + head]));
            if (score < previous.score || (score == previous.score && position > previous.block)) {
                best = better_choice(best, {score, position, i});
            }
        }
        const auto winner = Reduce(storage).Reduce(best, ReduceRankChoice{});
        if (!threadIdx.x) {
            previous = winner;
            const auto result = static_cast<std::uint64_t>(blockIdx.x) * representatives + rank;
            positions[result] = winner.block;
            if (result_scores) result_scores[result] = winner.score;
        }
        __syncthreads();
    }
}

// The upstream gather(...).mean(dim=-2) is independent for every output
// dimension. No partial-block top-key payload or CPU key copy is needed.
__global__ void probability_gather_mean(const Half * ring, const std::uint64_t * positions, Half * bank,
    std::uint64_t first_block, std::uint32_t capacity, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t width, std::uint32_t representatives) {
    const auto head = blockIdx.x % heads;
    const auto block = first_block + blockIdx.x / heads;
    const auto kv_head = head / (heads / kv_heads);
    const auto result = static_cast<std::uint64_t>(blockIdx.x) * representatives;
    for (auto d = threadIdx.x; d < width; d += blockDim.x) {
        float mean = 0;
        for (std::uint32_t r = 0; r < representatives; ++r) {
            const auto row = positions[result + r] % capacity;
            mean += DeviceHalf{}(ring[(row * kv_heads + kv_head) * width + d]);
        }
        bank[(block * heads + head) * width + d] = __half_as_ushort(__float2half(mean / representatives));
    }
}
class CudaBlockRanker final : public BlockRanker {
    struct BuilderState {
        bool initialized = false, query_pending = false;
        std::uint64_t length = 0;
        Half * keys = nullptr;
        double * scores = nullptr;
        Half * top_keys = nullptr;
        std::uint64_t * top_positions = nullptr;
        double * top_scores = nullptr;
    };
    struct ProbabilityReceipt {
        std::uint32_t layer, count;
        std::uint64_t end, mature_end, first_block, block_end;
    };
    struct ProbabilityState {
        Half * keys = nullptr;
        Half * scores = nullptr;
    };
public:
    CudaBlockRanker(KvGeometry geometry, std::uint32_t query_heads, std::uint32_t representatives,
                    std::uint32_t budget, int device, std::uint32_t local, std::uint32_t input_heads, bool representative_diagnostics) :
        geometry_(geometry), query_heads_(query_heads), representatives_(representatives), budget_(budget), local_(local),
        input_heads_(input_heads ? input_heads : geometry.heads), device_(device), representative_diagnostics_(representative_diagnostics) {
        geometry_.validate();
        if (!query_heads || query_heads % geometry.heads || !representatives || representatives > geometry.block_tokens ||
            geometry.heads % input_heads_ || !budget || budget > geometry.capacity_tokens || local > budget || device < 0)
            throw std::invalid_argument("invalid CUDA ranker geometry");
        block_elements_ = product(product(geometry.heads, representatives), geometry.dimension);
        layer_bytes_ = product(product(geometry.blocks_per_head(), block_elements_), sizeof(Half));
        banks_.resize(geometry.layers, nullptr);
        published_.resize(geometry.layers, 0);
        output_capacity_ = product(geometry.heads, budget);
        if (local_) builders_.resize(geometry.layers);
        stats_.host_bytes = banks_.capacity() * sizeof(void *) + published_.capacity() * sizeof(std::uint64_t) +
                            sizeof(error_) + builders_.capacity()*sizeof(BuilderState);
        DeviceScope scope(device_);
        try {
            cudaDeviceProp properties{};
            checked(cudaGetDeviceProperties(&properties, device_), "CUDA ranker properties");
            if (geometry.heads > static_cast<std::uint32_t>(properties.maxGridSize[0]) || properties.maxThreadsPerBlock < 256) {
                throw std::invalid_argument("CUDA ranker launch geometry exceeds device limits");
            }
            checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create CUDA ranker stream");
            trace::name_cuda_stream(stream_, "sparse.score+representatives");
            allocate(query_, product(product(query_heads, geometry.dimension), sizeof(float)));
            allocate(ids_, product(geometry.blocks_per_head(), sizeof(std::uint64_t)));
            allocate(costs_, product(geometry.blocks_per_head(), sizeof(std::uint32_t)));
            allocate(scores_, product(product(geometry.heads, geometry.blocks_per_head()), sizeof(double)));
            allocate(sorted_scores_, product(geometry.blocks_per_head(), sizeof(double)));
            allocate(sorted_ids_, product(geometry.blocks_per_head(), sizeof(std::uint64_t)));
            // The pinned llama.cpp CUDA argsort uses this CUB primitive for
            // one row. Stable sorting also preserves our ascending-ID tie rule.
            checked(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_temp_bytes_, scores_, sorted_scores_,
                ids_, sorted_ids_, geometry.blocks_per_head(), 0, sizeof(double) * 8, stream_), "size shared block sort workspace");
            allocate(sort_temp_, sort_temp_bytes_);
            allocate(output_device_, product(output_capacity_, sizeof(std::uint64_t)));
            const auto candidates = geometry.blocks_per_head();
            const auto host_bytes = product(output_capacity_ + candidates, sizeof(std::uint64_t)) +
                product(candidates, sizeof(std::uint32_t)) + sizeof(int);
            checked(cudaHostAlloc(&rank_host_, host_bytes, cudaHostAllocDefault), "allocate pinned rank metadata and receipt");
            stats_.host_bytes += host_bytes;
            output_host_ = static_cast<std::uint64_t *>(rank_host_);
            ids_host_ = output_host_ + output_capacity_;
            costs_host_ = reinterpret_cast<std::uint32_t *>(ids_host_ + candidates);
            rank_error_ = reinterpret_cast<int *>(costs_host_ + candidates);
            allocate(error_device_, sizeof(int));
            if (local_) {
                const auto before = stats_.device_bytes;
                allocate(mean_query_, product(product(geometry.heads, geometry.dimension), sizeof(double)));
                allocate(built_positions_, product(product(geometry.heads, representatives), sizeof(std::uint64_t)));
                allocate(built_scores_, product(product(geometry.heads, representatives), sizeof(double)));
                stats_.builder_device_bytes += stats_.device_bytes-before;
            }
        } catch (...) { release(); throw; }
    }
    ~CudaBlockRanker() override { release(); }
    void publish(std::uint32_t layer, std::uint64_t first, std::uint64_t count, const std::vector<Half> & keys) override {
        check();
        if (layer >= geometry_.layers || (local_ && builders_[layer].initialized) || first != published_.at(layer) || !count ||
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
    std::uint32_t builder_window() const override { return local_; }
    const char * builder_name() const override { return probability_local_ ? "cuda_attention_probability" : local_ ? "cuda_window_fp64" : "cpu"; }
    bool configure_probability_builder(std::uint32_t initial, std::uint32_t local, std::uint32_t representatives) override {
        check();
        if (local_ || probability_local_ || representatives_ != 1 || !local || !representatives ||
            representatives > geometry_.block_tokens || query_heads_ != geometry_.heads) return false;
        probability_initial_ = initial;
        probability_local_ = local;
        probability_top_ = representatives;
        const auto group_tokens = std::min(512u, local);
        probability_capacity_ = local + group_tokens + geometry_.block_tokens;
        probability_states_.resize(geometry_.layers);
        probability_lengths_.resize(geometry_.layers);
        probability_matured_.assign(geometry_.layers, initial);
        DeviceScope scope(device_);
        const auto before = stats_.device_bytes;
        const auto blocks = (group_tokens + geometry_.block_tokens - 1) / geometry_.block_tokens + 1;
        try {
            checked(cudaEventCreateWithFlags(&probability_done_, cudaEventDisableTiming), "create probability completion event");
            allocate(built_positions_, product(product(blocks, geometry_.heads), representatives * sizeof(std::uint64_t)));
            if (representative_diagnostics_) allocate(built_scores_, product(product(blocks, geometry_.heads), representatives * sizeof(double)));
            const auto slots = representative_diagnostics_ ? product(product(blocks, geometry_.heads), probability_top_) : 0;
            const auto bytes = slots * (sizeof(std::uint64_t) + sizeof(double)) + sizeof(int);
            checked(cudaHostAlloc(&probability_host_, bytes, cudaHostAllocDefault), "allocate pinned probability receipt");
            stats_.host_bytes += bytes;
            probability_host_positions_ = static_cast<std::uint64_t *>(probability_host_);
            probability_host_scores_ = reinterpret_cast<double *>(probability_host_positions_ + slots);
            probability_host_error_ = reinterpret_cast<int *>(probability_host_scores_ + slots);
        } catch (...) { fail(); throw; }
        stats_.builder_device_bytes += stats_.device_bytes - before;
        return true;
    }
    bool device_probabilities() const override { return probability_local_ != 0; }
    bool host_representative_metadata() const override { return !probability_local_ || representative_diagnostics_; }
    std::vector<BuiltRepresentatives> probability_update(const ProbabilityFrame & frame) override {
        return probability_update_impl(frame, nullptr);
    }
    void submit_probability_update(const ProbabilityFrame & frame, void * compute_stream, bool on_compute) override {
        if (!compute_stream || frame.count != 1) throw std::invalid_argument("asynchronous probability update requires one decode token and its compute queue");
        (void) probability_update_impl(frame, compute_stream, on_compute);
    }
    std::vector<BuiltRepresentatives> finish_probability_update() override {
        check();
        if (!probability_pending_) throw std::logic_error("no asynchronous probability receipt");
        DeviceScope scope(device_);
        try {
            checked(cudaEventSynchronize(probability_done_), "collect probability receipt");
            if (*probability_host_error_) throw std::runtime_error("non-finite probability keys or scores");
            auto result = commit_probability(*probability_pending_, probability_host_positions_, probability_host_scores_);
            probability_pending_.reset();
            return result;
        } catch (...) { fail(); throw; }
    }
    std::vector<BuiltRepresentatives> probability_update_impl(ProbabilityFrame frame, void * compute_stream, bool on_compute = false) {
        check();
        if (probability_pending_) throw std::logic_error("previous probability receipt must be collected");
        if (!probability_local_ || frame.layer >= geometry_.layers || !frame.count || frame.count > std::min(512u, probability_local_) ||
            probability_lengths_.at(frame.layer) != frame.first || frame.first >= geometry_.capacity_tokens ||
            frame.count > geometry_.capacity_tokens - frame.first) throw std::invalid_argument("CUDA probability frame");
        const auto end = frame.first + frame.count;
        const auto origin = frame.first + 1 - std::min<std::uint64_t>(frame.first + 1, probability_local_);
        validate_device_kv(frame.kv, DeviceMemory::cuda, input_heads_, geometry_.dimension, frame.count);
        if (frame.score_first != origin || frame.scores.memory != DeviceMemory::cuda ||
            frame.scores.elements != (end - origin) * geometry_.heads) throw std::invalid_argument("CUDA probability score shape");
        validate_pointer(frame.kv.data);
        validate_pointer(frame.scores.data);
        DeviceScope scope(device_);
        const auto queue = on_compute ? static_cast<cudaStream_t>(compute_stream) : stream_;
        auto & state = probability_states_[frame.layer];
        const auto mature_end = std::max<std::uint64_t>(probability_initial_, end - std::min<std::uint64_t>(end, probability_local_));
        const auto first_block = published_[frame.layer];
        const auto block_end = (mature_end - probability_initial_) / geometry_.block_tokens;
        const auto metadata = (block_end - first_block) * geometry_.heads * probability_top_;
        const ProbabilityReceipt receipt{frame.layer, frame.count, end, mature_end, first_block, block_end};
        try {
            if (!state.keys) {
                const auto before = stats_.device_bytes;
                allocate(state.keys, product(product(probability_capacity_, input_heads_), geometry_.dimension * sizeof(Half)));
                allocate(state.scores, product(product(probability_capacity_, geometry_.heads), sizeof(Half)));
                checked(cudaMemsetAsync(state.scores, 0, product(product(probability_capacity_, geometry_.heads), sizeof(Half)), queue), "initialize probability scores");
                stats_.builder_device_bytes += stats_.device_bytes - before;
            }
            if (!banks_[frame.layer]) allocate(banks_[frame.layer], layer_bytes_);
            if (compute_stream && !on_compute) capture_probability_inputs(frame, static_cast<cudaStream_t>(compute_stream));
            checked(cudaMemsetAsync(error_device_, 0, sizeof(int), queue), "clear probability status");
            const auto key_items = product(product(frame.count, input_heads_), geometry_.dimension);
            const auto score_items = product(end - origin, geometry_.heads);
            const auto key_grid = static_cast<unsigned>(std::min<std::size_t>((key_items + 255) / 256, 256));
            const auto score_grid = static_cast<unsigned>(std::min<std::size_t>((score_items + 255) / 256, 256));
            probability_append_keys<<<key_grid, 256, 0, queue>>>(frame.kv.data, state.keys, state.scores, error_device_,
                frame.first, frame.count, probability_capacity_, input_heads_, geometry_.heads, geometry_.dimension);
            checked(cudaGetLastError(), "append probability keys");
            probability_add_scores<<<score_grid, 256, 0, queue>>>(frame.scores.data, state.scores, error_device_,
                origin, static_cast<std::uint32_t>(end - origin), geometry_.heads, probability_capacity_, frame.key_major);
            checked(cudaGetLastError(), "accumulate probability scores");
            if (metadata) {
                const auto grid = static_cast<unsigned>((block_end - first_block) * geometry_.heads);
                if (geometry_.block_tokens <= 128) {
                    probability_block_topk<<<grid, 128, 0, queue>>>(state.scores, built_positions_, built_scores_,
                        first_block, probability_initial_, probability_capacity_, geometry_.heads, probability_top_, geometry_.block_tokens);
                } else {
                    probability_large_block_topk<<<grid, 128, 0, queue>>>(state.scores, built_positions_, built_scores_,
                        first_block, probability_initial_, probability_capacity_, geometry_.heads, probability_top_, geometry_.block_tokens);
                }
                checked(cudaGetLastError(), "select probability block representatives");
                probability_gather_mean<<<grid, 128, 0, queue>>>(state.keys, built_positions_, static_cast<Half *>(banks_[frame.layer]),
                    first_block, probability_capacity_, geometry_.heads, input_heads_, geometry_.dimension, probability_top_);
                checked(cudaGetLastError(), "gather and average probability representatives");
                if (representative_diagnostics_) {
                    checked(cudaMemcpyAsync(probability_host_positions_, built_positions_, metadata * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, queue), "download representative positions");
                    checked(cudaMemcpyAsync(probability_host_scores_, built_scores_, metadata * sizeof(double), cudaMemcpyDeviceToHost, queue), "download representative scores");
                }
            }
            checked(cudaMemcpyAsync(probability_host_error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, queue), "download probability status");
            if (compute_stream) {
                checked(cudaEventRecord(probability_done_, queue), "record probability completion");
                probability_pending_ = receipt;
                return {};
            }
            checked(cudaStreamSynchronize(queue), "complete probability receipt");
            if (*probability_host_error_) throw std::runtime_error("non-finite probability keys or scores");
        } catch (...) {
            // A failed capture must drain borrowed sources as well as the
            // ranker queue before graph memory or snapshot storage is reused.
            if (compute_stream) (void) cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream));
            fail(); throw;
        }
        return commit_probability(receipt, probability_host_positions_, probability_host_scores_);
    }
    std::vector<BuiltRepresentatives> commit_probability(const ProbabilityReceipt & receipt,
        const std::uint64_t * positions, const double * scores) {
        std::vector<BuiltRepresentatives> result(receipt.block_end - receipt.first_block);
        const auto stride = static_cast<std::size_t>(geometry_.heads) * probability_top_;
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i].block = receipt.first_block + i;
            if (representative_diagnostics_) {
                result[i].positions.assign(positions + i * stride, positions + (i + 1) * stride);
                result[i].scores.assign(scores + i * stride, scores + (i + 1) * stride);
            }
        }
        published_[receipt.layer] = receipt.block_end;
        probability_lengths_[receipt.layer] = receipt.end;
        probability_matured_[receipt.layer] = receipt.mature_end;
        stats_.published_blocks += result.size();
        stats_.built_blocks += result.size();
        stats_.device_builder_appends += receipt.count;
        return result;
    }
    std::optional<DeviceMemory> direct_memory() const override { return DeviceMemory::cuda; }
    void seed_builder(std::uint32_t layer, const RepresentativeSeed & seed) override {
        check();
        if (!local_ || layer >= geometry_.layers || builders_[layer].initialized) throw std::invalid_argument("CUDA builder initialization state");
        validate_representative_seed(geometry_, representatives_, local_, seed, input_heads_);
        const auto pending = std::min<std::uint64_t>(seed.length, local_);
        const auto first = seed.length-pending;
        if (published_[layer] != first/geometry_.block_tokens) throw std::invalid_argument("CUDA builder published prefix differs from seed");
        auto & state = builders_[layer];
        DeviceScope scope(device_);
        try {
            if (!banks_[layer]) allocate(banks_[layer], layer_bytes_);
            const auto before = stats_.device_bytes;
            const auto keys_per_token = product(input_heads_, geometry_.dimension);
            allocate(state.keys, product(product(local_, keys_per_token), sizeof(Half)));
            allocate(state.scores, product(product(local_, geometry_.heads), sizeof(double)));
            allocate(state.top_keys, product(block_elements_, sizeof(Half)));
            allocate(state.top_positions, product(product(geometry_.heads, representatives_), sizeof(std::uint64_t)));
            allocate(state.top_scores, product(product(geometry_.heads, representatives_), sizeof(double)));
            stats_.builder_device_bytes += stats_.device_bytes-before;
            const auto ring_start = first%local_;
            const auto first_count = std::min<std::uint64_t>(pending, local_-ring_start);
            const auto copy_part = [&](std::uint64_t ring, std::uint64_t source, std::uint64_t count) {
                if (!count) return;
                checked(cudaMemcpyAsync(state.keys+ring*keys_per_token, seed.pending_keys.data()+source*keys_per_token,
                    product(product(count, keys_per_token), sizeof(Half)), cudaMemcpyHostToDevice, stream_), "seed pending keys");
                checked(cudaMemcpyAsync(state.scores+ring*geometry_.heads, seed.pending_scores.data()+source*geometry_.heads,
                    product(product(count, geometry_.heads), sizeof(double)), cudaMemcpyHostToDevice, stream_), "seed pending scores");
            };
            copy_part(ring_start, 0, first_count);
            copy_part(0, first_count, pending-first_count);
            checked(cudaMemcpyAsync(state.top_keys, seed.partial_keys.data(), product(seed.partial_keys.size(), sizeof(Half)), cudaMemcpyHostToDevice, stream_), "seed partial keys");
            checked(cudaMemcpyAsync(state.top_positions, seed.partial_positions.data(), product(seed.partial_positions.size(), sizeof(std::uint64_t)), cudaMemcpyHostToDevice, stream_), "seed partial positions");
            checked(cudaMemcpyAsync(state.top_scores, seed.partial_scores.data(), product(seed.partial_scores.size(), sizeof(double)), cudaMemcpyHostToDevice, stream_), "seed partial scores");
            checked(cudaStreamSynchronize(stream_), "complete CUDA builder initialization");
            state.length = seed.length; state.initialized = true;
        } catch (...) { fail(); throw; }
    }
    std::optional<BuiltRepresentatives> build_query(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) override {
        if (query.size() != product(query_heads_, geometry_.dimension)) throw std::invalid_argument("CUDA builder query shape");
        for (const auto q : query) if (!std::isfinite(q)) throw std::invalid_argument("non-finite CUDA builder query");
        return build_query_impl(layer, position, query.data(), false);
    }
    std::optional<BuiltRepresentatives> build_query_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query) override {
        validate_device_query(query, DeviceMemory::cuda, query_heads_, geometry_.dimension);
        validate_pointer(query.data);
        auto result = build_query_impl(layer, position, query.data, true);
        ++stats_.device_builder_queries;
        return result;
    }
    std::optional<BuiltRepresentatives> build_query_for_rank_device(std::uint32_t layer,std::uint64_t position,DeviceQuery query) override {
        validate_device_query(query,DeviceMemory::cuda,query_heads_,geometry_.dimension);
        validate_pointer(query.data);
        auto result=build_query_impl(layer,position,query.data,true,true);
        ++stats_.device_builder_queries;
        return result;
    }
    void finish_query_for_rank_device() override {
        if (!deferred_query_) return;
        DeviceScope scope(device_);
        try {
            checked(cudaMemcpyAsync(&error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream_),"download deferred builder status");
            checked(cudaStreamSynchronize(stream_),"complete deferred representative query");
            deferred_query_.reset();
            if (error_) throw std::runtime_error("non-finite representative window score");
        } catch (...) { fail(); throw; }
    }
    std::optional<BuiltRepresentatives> build_query_impl(std::uint32_t layer, std::uint64_t position, const float * query, bool device,
                                                         bool defer_for_rank=false) {
        check();
        if (deferred_query_ || !local_ || layer >= geometry_.layers || !builders_[layer].initialized || builders_[layer].query_pending ||
            builders_[layer].length != position || position >= geometry_.capacity_tokens) throw std::invalid_argument("CUDA builder query state or shape");
        auto & state = builders_[layer];
        const auto pending = std::min<std::uint64_t>(position, local_);
        const bool complete = position >= local_ && (position-local_+1)%geometry_.block_tokens == 0;
        std::optional<BuiltRepresentatives> result;
        if (complete) {
            result.emplace(); result->block = (position-local_)/geometry_.block_tokens;
            result->keys.resize(block_elements_);
            result->positions.resize(product(geometry_.heads, representatives_));
            result->scores.resize(result->positions.size());
        }
        if (pending || device) {
            DeviceScope scope(device_);
            try {
                if (!device) {
                    const auto bytes = product(product(query_heads_, geometry_.dimension), sizeof(float));
                    checked(cudaMemcpyAsync(query_, query, bytes, cudaMemcpyHostToDevice, stream_), "upload builder query");
                    stats_.query_upload_bytes += bytes;
                }
                checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear builder status");
                const auto mean_grid = static_cast<unsigned>(std::min<std::uint64_t>((static_cast<std::uint64_t>(geometry_.heads)*geometry_.dimension+255)/256,65535));
                builder_mean_query<<<mean_grid,256,0,stream_>>>(device ? query : query_, mean_query_, error_device_, geometry_.heads, geometry_.dimension, query_heads_/geometry_.heads);
                checked(cudaGetLastError(), "launch builder GQA mean");
                const auto score_grid = static_cast<unsigned>(std::min<std::uint64_t>((pending*geometry_.heads+7)/8,65535));
                if (pending) {
                    builder_update_scores<<<score_grid,256,0,stream_>>>(state.keys, state.scores, mean_query_, error_device_, position-pending,
                        pending, local_, geometry_.heads, input_heads_, geometry_.dimension);
                    checked(cudaGetLastError(), "launch representative window update");
                }
                if (position >= local_) {
                    builder_finalize<<<(geometry_.heads+127)/128,128,0,stream_>>>(state.keys, state.scores, state.top_keys, state.top_positions,
                        state.top_scores, static_cast<Half *>(banks_[layer]), built_positions_, built_scores_, position, local_, geometry_.heads,
                        input_heads_, geometry_.dimension, representatives_, geometry_.block_tokens);
                    checked(cudaGetLastError(), "launch representative finalize");
                }
                if (result) {
                    const auto offset = product(result->block, block_elements_);
                    checked(cudaMemcpyAsync(result->keys.data(), static_cast<Half *>(banks_[layer])+offset, product(block_elements_, sizeof(Half)), cudaMemcpyDeviceToHost, stream_), "download built keys");
                    checked(cudaMemcpyAsync(result->positions.data(), built_positions_, product(result->positions.size(), sizeof(std::uint64_t)), cudaMemcpyDeviceToHost, stream_), "download built positions");
                    checked(cudaMemcpyAsync(result->scores.data(), built_scores_, product(result->scores.size(), sizeof(double)), cudaMemcpyDeviceToHost, stream_), "download built scores");
                }
                if (defer_for_rank && !result) deferred_query_=layer;
                else {
                    checked(cudaMemcpyAsync(&error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download builder status");
                    checked(cudaStreamSynchronize(stream_), "complete representative window update");
                    if (error_) throw std::runtime_error("non-finite representative window score");
                }
            } catch (...) { fail(); throw; }
        }
        if (complete) { ++published_[layer]; ++stats_.published_blocks; ++stats_.built_blocks; }
        state.query_pending = true; ++stats_.builder_queries;
        return result;
    }
    void build_append(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & keys) override {
        check();
        if (!local_ || layer >= geometry_.layers || !builders_[layer].initialized || !builders_[layer].query_pending ||
            builders_[layer].length != position || position >= geometry_.capacity_tokens ||
            keys.size() != product(input_heads_, geometry_.dimension)) throw std::invalid_argument("CUDA builder append state or shape");
        for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite CUDA builder key");
        auto & state = builders_[layer];
        DeviceScope scope(device_);
        try {
            const auto slot = position%local_;
            checked(cudaMemcpyAsync(state.keys+slot*keys.size(), keys.data(), product(keys.size(), sizeof(Half)), cudaMemcpyHostToDevice, stream_), "append builder key");
            stats_.builder_key_upload_bytes += product(keys.size(), sizeof(Half));
            checked(cudaMemsetAsync(state.scores+slot*geometry_.heads, 0, product(geometry_.heads, sizeof(double)), stream_), "reset new representative score");
            checked(cudaStreamSynchronize(stream_), "complete builder append");
            ++state.length; state.query_pending = false; ++stats_.builder_appends;
        } catch (...) { fail(); throw; }
    }
    void build_append_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys) override {
        append_device_impl(layer,position,keys,nullptr);
    }
    void build_append_device_on_compute(std::uint32_t layer,std::uint64_t position,DeviceKeys keys,void * compute_stream) override {
        if (!compute_stream) throw std::invalid_argument("missing CUDA compute stream for representative append");
        append_device_impl(layer,position,keys,static_cast<cudaStream_t>(compute_stream));
    }
    void append_device_impl(std::uint32_t layer,std::uint64_t position,DeviceKeys keys,cudaStream_t compute_stream) {
        check();
        validate_device_keys(keys, DeviceMemory::cuda, input_heads_, geometry_.dimension);
        if (!local_ || layer >= geometry_.layers || !builders_[layer].initialized || !builders_[layer].query_pending ||
            builders_[layer].length != position || position >= geometry_.capacity_tokens) throw std::invalid_argument("CUDA device append state");
        validate_pointer(keys.data);
        auto & state = builders_[layer];
        DeviceScope scope(device_);
        try {
            const auto stream=compute_stream?compute_stream:stream_;
            if (!compute_stream) checked(cudaMemsetAsync(error_device_,0,sizeof(int),stream),"clear device append status");
            const auto count = product(input_heads_, geometry_.dimension);
            const auto grid = static_cast<unsigned>(std::min<std::size_t>((count+255)/256,65535));
            // The attention receipt protects the borrowed K on the compute path.
            // The host archive validates the same FP16 row before the next query.
            builder_append_keys<<<grid,256,0,stream>>>(keys.data,state.keys+(position%local_)*count,
                compute_stream?nullptr:error_device_,keys.head_stride,input_heads_,geometry_.dimension);
            checked(cudaGetLastError(), "launch device representative append");
            checked(cudaMemsetAsync(state.scores+(position%local_)*geometry_.heads,0,product(geometry_.heads,sizeof(double)),stream),"reset device representative score");
            if (compute_stream) ++stats_.builder_compute_appends;
            else {
                checked(cudaMemcpyAsync(&error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream),"download device append status");
                checked(cudaStreamSynchronize(stream),"complete device representative append");
                if (error_) throw std::runtime_error("non-finite device representative key");
            }
            ++state.length; state.query_pending = false; ++stats_.builder_appends; ++stats_.device_builder_appends;
        } catch (...) { fail(); throw; }
    }
    std::vector<BuiltRepresentatives> build_batch_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,
                                                         DeviceQuery query,DeviceKvBatch kv) override {
        check();
        validate_device_query(query,DeviceMemory::cuda,query_heads_,geometry_.dimension,tokens);
        validate_device_kv(kv,DeviceMemory::cuda,input_heads_,geometry_.dimension,tokens);
        if (!local_ || layer>=geometry_.layers || !builders_[layer].initialized || builders_[layer].query_pending ||
            tokens>geometry_.capacity_tokens || first>geometry_.capacity_tokens-tokens || builders_[layer].length!=first) {
            throw std::invalid_argument("CUDA builder batch state or position");
        }
        validate_pointer(query.data); validate_pointer(kv.data);
        auto & state=builders_[layer];
        std::vector<BuiltRepresentatives> result;
        result.reserve((static_cast<std::uint64_t>(tokens)+geometry_.block_tokens-1)/geometry_.block_tokens);
        const auto qwidth=product(query_heads_,geometry_.dimension),kwidth=product(geometry_.heads,geometry_.dimension);
        const auto input_width = product(input_heads_, geometry_.dimension);
        const auto kv_stride = product(input_width, 2);
        const auto append_grid = static_cast<unsigned>(std::min<std::size_t>((input_width + 255) / 256, 65535));
        const auto mean_grid=static_cast<unsigned>(std::min<std::size_t>((kwidth+255)/256,65535));
        DeviceScope scope(device_);
        try {
            checked(cudaMemsetAsync(error_device_,0,sizeof(int),stream_),"clear representative batch status");
            for (std::uint32_t i=0;i<tokens;++i) {
                const auto position=first+i,pending=std::min<std::uint64_t>(position,local_);
                builder_mean_query<<<mean_grid,256,0,stream_>>>(query.data+i*qwidth,mean_query_,error_device_,
                    geometry_.heads,geometry_.dimension,query_heads_/geometry_.heads);
                checked(cudaGetLastError(),"launch batch GQA mean");
                if (pending) {
                    const auto grid=static_cast<unsigned>(std::min<std::uint64_t>((pending*geometry_.heads+7)/8,65535));
                    builder_update_scores<<<grid,256,0,stream_>>>(state.keys,state.scores,mean_query_,error_device_,
                        position-pending,pending,local_,geometry_.heads,input_heads_,geometry_.dimension);
                    checked(cudaGetLastError(),"launch batch representative score update");
                }
                if (position>=local_) {
                    builder_finalize<<<(geometry_.heads+127)/128,128,0,stream_>>>(state.keys,state.scores,state.top_keys,state.top_positions,
                        state.top_scores,static_cast<Half *>(banks_[layer]),built_positions_,built_scores_,position,local_,geometry_.heads,
                        input_heads_,geometry_.dimension,representatives_,geometry_.block_tokens);
                    checked(cudaGetLastError(),"launch batch representative finalize");
                }
                if (position>=local_ && (position-local_+1)%geometry_.block_tokens==0) {
                    result.emplace_back(); auto & built=result.back(); built.block=(position-local_)/geometry_.block_tokens;
                    built.keys.resize(block_elements_); built.positions.resize(product(geometry_.heads,representatives_)); built.scores.resize(built.positions.size());
                    checked(cudaMemcpyAsync(built.keys.data(),static_cast<Half *>(banks_[layer])+product(built.block,block_elements_),
                        product(block_elements_,sizeof(Half)),cudaMemcpyDeviceToHost,stream_),"download batch representative keys");
                    checked(cudaMemcpyAsync(built.positions.data(),built_positions_,product(built.positions.size(),sizeof(std::uint64_t)),
                        cudaMemcpyDeviceToHost,stream_),"download batch representative positions");
                    checked(cudaMemcpyAsync(built.scores.data(),built_scores_,product(built.scores.size(),sizeof(double)),
                        cudaMemcpyDeviceToHost,stream_),"download batch representative scores");
                }
                builder_append_keys<<<append_grid,256,0,stream_>>>(kv.data+i*kv_stride,state.keys+(position%local_)*input_width,error_device_,
                    2*geometry_.dimension,input_heads_,geometry_.dimension);
                checked(cudaGetLastError(),"launch batch representative append");
                checked(cudaMemsetAsync(state.scores+(position%local_)*geometry_.heads,0,product(geometry_.heads,sizeof(double)),stream_),"reset batch representative score");
            }
            checked(cudaMemcpyAsync(&error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream_),"download representative batch status");
            checked(cudaStreamSynchronize(stream_),"complete device representative batch");
            if (error_) throw std::runtime_error("non-finite device representative batch");
            state.length+=tokens; published_[layer]+=result.size();
            stats_.published_blocks+=result.size(); stats_.built_blocks+=result.size();
            stats_.builder_queries+=tokens; stats_.builder_appends+=tokens;
            stats_.device_builder_queries+=tokens; stats_.device_builder_appends+=tokens; ++stats_.device_builder_batches;
        } catch (...) { fail(); throw; }
        return result;
    }
    std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                const BlockCandidates & candidates) override {
        if (query.size() != product(query_heads_, geometry_.dimension)) throw std::invalid_argument("CUDA ranking query shape");
        for (const auto q : query) if (!std::isfinite(q)) throw std::invalid_argument("non-finite CUDA ranking query");
        return rank_impl(layer, query.data(), candidates, false);
    }
    std::vector<std::vector<std::uint64_t>> rank_device(std::uint32_t layer, DeviceQuery query,
                                                       const BlockCandidates & candidates) override {
        return rank_device_after_event(layer, {query, nullptr}, candidates);
    }
    std::vector<std::vector<std::uint64_t>> rank_device_after_event(std::uint32_t layer, DeviceQueryWithEvent input,
                                                                  const BlockCandidates & candidates) override {
        const auto query = input.query;
        validate_device_query(query, DeviceMemory::cuda, query_heads_, geometry_.dimension);
        validate_pointer(query.data);
        auto result = rank_impl(layer, query.data, candidates, true, input.cuda_ready_event);
        ++stats_.device_rank_queries;
        return result;
    }
    RankCompletion submit_rank_device(std::uint32_t layer, DeviceQueryWithEvent input,
                                     const BlockCandidates & candidates) override {
        validate_device_query(input.query, DeviceMemory::cuda, query_heads_, geometry_.dimension);
        validate_pointer(input.query.data);
        (void) rank_impl(layer, input.query.data, candidates, true, input.cuda_ready_event, true);
        ++stats_.device_rank_queries;
        const auto ticket = rank_ticket_;
        return [this, ticket] {
            if (!pending_rank_ || ticket != rank_ticket_) throw std::logic_error("expired CUDA rank receipt");
            return finish_rank();
        };
    }
    std::vector<std::vector<std::uint64_t>> rank_impl(std::uint32_t layer, const float * query,
                                                    const BlockCandidates & candidates, bool device, void * cuda_ready_event = nullptr,
                                                    bool deferred = false) {
        check();
        candidates.validate_shape();
        if (pending_rank_) throw std::logic_error("previous rank receipt must be collected");
        if (probability_pending_) throw std::logic_error("block ranking requires its preceding probability receipt");
        if (deferred_query_ && (!device || *deferred_query_!=layer)) throw std::logic_error("ranking does not match deferred representative query");
        if (layer >= geometry_.layers ||
            candidates.size() > geometry_.blocks_per_head() ||
            candidates.remaining > budget_) throw std::invalid_argument("CUDA ranking shape or budget");
        trace::Scope phase("GPU Score/select", layer,
            probability_local_ ? probability_lengths_[layer] : UINT64_MAX);
        if (candidates.is_prefix() && (candidates.prefix_tokens > geometry_.block_tokens || candidates.size() > published_[layer])) {
            throw std::invalid_argument("invalid CUDA rank prefix");
        }
        for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
            if (candidates.blocks[i] >= published_[layer] || (i && candidates.blocks[i] <= candidates.blocks[i-1]) ||
                !candidates.tokens[i] || candidates.tokens[i] > geometry_.block_tokens) throw std::invalid_argument("invalid CUDA rank candidate");
        }
        if (!candidates.remaining || !candidates.size()) {
            finish_query_for_rank_device();
            if (device) {
                DeviceScope scope(device_);
                try {
                    if (cuda_ready_event) {
                        checked(cudaStreamWaitEvent(stream_, static_cast<cudaEvent_t>(cuda_ready_event), 0), "order empty ranking after query mean");
                        ++stats_.query_ready_event_waits;
                    }
                    checked(cudaMemsetAsync(error_device_,0,sizeof(int),stream_),"clear empty rank status");
                    const auto count=product(query_heads_,geometry_.dimension);
                    const auto grid=static_cast<unsigned>(std::min<std::size_t>((count+255)/256,65535));
                    validate_query_values<<<grid,256,0,stream_>>>(query,count,error_device_);
                    checked(cudaGetLastError(),"validate empty rank query");
                    checked(cudaMemcpyAsync(rank_error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream_),"download empty rank status");
                } catch (...) { fail(); throw; }
            }
            if (!device) *rank_error_ = 0;
            pending_rank_ = PendingRank{0, geometry_.heads, 0, candidates.shared_heads};
            ++rank_ticket_;
            return deferred ? RankedBlocks{} : finish_rank();
        }
        const auto count = candidates.size();
        const auto smallest = candidates.is_prefix() ? candidates.prefix_tokens : *std::min_element(candidates.tokens.begin(), candidates.tokens.end());
        const auto max_selected = (static_cast<std::uint64_t>(candidates.remaining) + smallest - 1) / smallest;
        const auto stride = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_selected, count));
        const auto lists = candidates.shared_heads ? 1u : geometry_.heads;
        const auto bytes = product(product(lists, stride), sizeof(std::uint64_t));
        const bool uniform_shared = candidates.shared_heads && (candidates.is_prefix() ||
            std::all_of(candidates.tokens.begin(), candidates.tokens.end(),
                [smallest](std::uint32_t cost) { return cost == smallest; }));
        DeviceScope scope(device_);
        try {
            if (cuda_ready_event) {
                checked(cudaStreamWaitEvent(stream_, static_cast<cudaEvent_t>(cuda_ready_event), 0), "order ranking after query mean");
                ++stats_.query_ready_event_waits;
            }
            if (!device) {
                const auto bytes = product(product(query_heads_, geometry_.dimension), sizeof(float));
                checked(cudaMemcpyAsync(query_, query, bytes, cudaMemcpyHostToDevice, stream_), "upload rank query");
                stats_.query_upload_bytes += bytes;
            }
            if (!candidates.is_prefix()) {
                std::copy(candidates.blocks.begin(), candidates.blocks.end(), ids_host_);
                std::copy(candidates.tokens.begin(), candidates.tokens.end(), costs_host_);
                checked(cudaMemcpyAsync(ids_, ids_host_, product(count, sizeof(std::uint64_t)), cudaMemcpyHostToDevice, stream_), "upload rank candidates");
                checked(cudaMemcpyAsync(costs_, costs_host_, product(count, sizeof(std::uint32_t)), cudaMemcpyHostToDevice, stream_), "upload rank costs");
            }
            if (!deferred_query_) checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear rank status");
            if (!uniform_shared) checked(cudaMemsetAsync(output_device_, 0xff, bytes, stream_), "clear rank output");
            if (candidates.shared_heads) {
                const auto grid = static_cast<unsigned>(std::min<std::uint64_t>(count, 65535));
                score_shared_blocks<<<grid, 256, 0, stream_>>>(static_cast<const __half *>(banks_[layer]), device ? query : query_,
                    ids_, scores_, error_device_, count, geometry_.heads, query_heads_ / geometry_.heads, representatives_, geometry_.dimension,
                    candidates.is_prefix());
            } else {
                const auto grid = static_cast<unsigned>(std::min<std::uint64_t>((count * geometry_.heads + 7) / 8, 65535));
                score_blocks<<<grid, 256, 0, stream_>>>(static_cast<const __half *>(banks_[layer]), device ? query : query_, ids_, scores_, error_device_,
                    count, geometry_.heads, query_heads_ / geometry_.heads, representatives_, geometry_.dimension);
            }
            checked(cudaGetLastError(), "launch block relevance kernel");
            if (uniform_shared) {
                auto workspace_bytes = sort_temp_bytes_;
                checked(cub::DeviceRadixSort::SortPairsDescending(sort_temp_, workspace_bytes, scores_, sorted_scores_,
                    ids_, sorted_ids_, static_cast<std::uint64_t>(count), 0, sizeof(double) * 8, stream_), "sort shared block relevance with CUB");
            } else {
                select_top_blocks<<<lists, 256, 0, stream_>>>(scores_, ids_, costs_, output_device_, count, candidates.remaining, stride);
                checked(cudaGetLastError(), "launch block top-k kernel");
            }
            checked(cudaMemcpyAsync(output_host_, uniform_shared ? sorted_ids_ : output_device_, bytes, cudaMemcpyDeviceToHost, stream_), "download selected block IDs");
            checked(cudaMemcpyAsync(rank_error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download rank status");
        } catch (...) { fail(); throw; }
        pending_rank_ = PendingRank{count, lists, stride, candidates.shared_heads};
        ++rank_ticket_;
        return deferred ? RankedBlocks{} : finish_rank();
    }
    RankedBlocks finish_rank() {
        check(true);
        if (!pending_rank_) throw std::logic_error("no pending CUDA rank receipt");
        DeviceScope scope(device_);
        try {
            checked(cudaStreamSynchronize(stream_), "collect CUDA block ranking");
            if (deferred_query_) ++stats_.builder_rank_shared_waits;
            deferred_query_.reset();
            if (*rank_error_) throw std::runtime_error("non-finite CUDA representative score or block relevance");
        } catch (...) { fail(); throw; }
        const auto pending = *pending_rank_;
        pending_rank_.reset();
        RankedBlocks result(geometry_.heads);
        for (std::uint32_t head = 0; head < pending.lists; ++head) {
            for (std::uint32_t i = 0; i < pending.stride; ++i) {
                const auto block = output_host_[static_cast<std::size_t>(head) * pending.stride + i];
                if (block == UINT64_MAX) break;
                result[head].push_back(block);
            }
        }
        if (pending.shared) std::fill(result.begin() + 1, result.end(), result.front());
        if (pending.count) {
            ++stats_.calls;
            stats_.scored_blocks += pending.count * pending.lists;
            shared_ranking_ = pending.shared;
        }
        return result;
    }
    RankerStats stats() const override { return stats_; }
    bool prefix_candidates() const override { return true; }
    const char * name() const override { return probability_local_ ? "cuda_infllm_fp32_topk" : shared_ranking_ ? "cuda_shared_fp32_topk" : "cuda_fp64_topk"; }
    bool owns_representative_keys() const override { return true; }
    std::vector<Half> read_representatives(std::uint32_t layer, std::uint64_t block) override {
        check();
        if (layer >= geometry_.layers || block >= published_[layer]) throw std::out_of_range("CUDA representative readback block");
        std::vector<Half> keys(block_elements_);
        DeviceScope scope(device_);
        try {
            checked(cudaMemcpyAsync(keys.data(), static_cast<const Half *>(banks_[layer]) + block * block_elements_,
                product(block_elements_, sizeof(Half)), cudaMemcpyDeviceToHost, stream_), "read back representative block");
            checked(cudaStreamSynchronize(stream_), "complete representative readback");
        } catch (...) { fail(); throw; }
        return keys;
    }
private:
    void capture_probability_inputs(ProbabilityFrame & frame, cudaStream_t compute) {
        if (!probability_snapshot_keys_) {
            const auto before = stats_.device_bytes;
            allocate(probability_snapshot_keys_, product(product(input_heads_, geometry_.dimension), 2 * sizeof(Half)));
            allocate(probability_snapshot_scores_, product(product(probability_local_, geometry_.heads), sizeof(float)));
            stats_.builder_device_bytes += stats_.device_bytes - before;
            checked(cudaEventCreateWithFlags(&probability_captured_, cudaEventDisableTiming), "create probability input capture event");
        }
        // These snapshots precede all subsequent graph work on compute. Once
        // they finish, the graph may overwrite its K/V and probability tensors.
        checked(cudaMemcpyAsync(probability_snapshot_keys_, frame.kv.data, frame.kv.elements * sizeof(Half),
            cudaMemcpyDeviceToDevice, compute), "capture probability KV");
        checked(cudaMemcpyAsync(probability_snapshot_scores_, frame.scores.data, frame.scores.elements * sizeof(float),
            cudaMemcpyDeviceToDevice, compute), "capture probability scores");
        checked(cudaEventRecord(probability_captured_, compute), "record probability input capture");
        checked(cudaStreamWaitEvent(stream_, probability_captured_, 0), "order probability update after capture");
        frame.kv.data = probability_snapshot_keys_;
        frame.scores.data = probability_snapshot_scores_;
    }
    void validate_pointer(const void * pointer) const {
        check();
        DeviceScope scope(device_);
        cudaPointerAttributes attributes{};
        checked(cudaPointerGetAttributes(&attributes, pointer), "inspect borrowed CUDA input");
        if ((attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) || attributes.device != device_) {
            throw std::invalid_argument("borrowed CUDA input belongs to another memory space or device");
        }
    }
    template<class T> void allocate(T *& pointer, std::size_t bytes) {
        checked(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes), "allocate CUDA ranker buffer");
        stats_.device_bytes += bytes;
    }
    void check(bool collecting_rank = false) const {
        if (failed_) throw std::runtime_error("CUDA block ranker failed");
        if (pending_rank_ && !collecting_rank) throw std::logic_error("previous rank receipt must be collected");
    }
    void fail() noexcept {
        failed_ = true;
        if (probability_pending_ && probability_done_) (void) cudaEventSynchronize(probability_done_);
        if (stream_) (void) cudaStreamSynchronize(stream_);
        deferred_query_.reset();
        pending_rank_.reset();
    }
    void release() noexcept {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device_) != cudaSuccess) return;
        if (probability_pending_ && probability_done_) (void) cudaEventSynchronize(probability_done_);
        if (stream_) (void) cudaStreamSynchronize(stream_);
        for (auto * bank : banks_) if (bank) (void) cudaFree(bank);
        for (const auto & state : builders_) {
            if (state.keys) (void) cudaFree(state.keys);
            if (state.scores) (void) cudaFree(state.scores);
            if (state.top_keys) (void) cudaFree(state.top_keys);
            if (state.top_positions) (void) cudaFree(state.top_positions);
            if (state.top_scores) (void) cudaFree(state.top_scores);
        }
        for (const auto & state : probability_states_) {
            if (state.keys) (void) cudaFree(state.keys);
            if (state.scores) (void) cudaFree(state.scores);
        }
        if (mean_query_) (void) cudaFree(mean_query_);
        if (built_positions_) (void) cudaFree(built_positions_);
        if (built_scores_) (void) cudaFree(built_scores_);
        if (query_) (void) cudaFree(query_);
        if (ids_) (void) cudaFree(ids_);
        if (costs_) (void) cudaFree(costs_);
        if (scores_) (void) cudaFree(scores_);
        if (sorted_scores_) (void) cudaFree(sorted_scores_);
        if (sorted_ids_) (void) cudaFree(sorted_ids_);
        if (sort_temp_) (void) cudaFree(sort_temp_);
        if (output_device_) (void) cudaFree(output_device_);
        if (error_device_) (void) cudaFree(error_device_);
        if (probability_snapshot_keys_) (void) cudaFree(probability_snapshot_keys_);
        if (probability_snapshot_scores_) (void) cudaFree(probability_snapshot_scores_);
        if (probability_captured_) (void) cudaEventDestroy(probability_captured_);
        if (probability_done_) (void) cudaEventDestroy(probability_done_);
        if (probability_host_) (void) cudaFreeHost(probability_host_);
        if (rank_host_) (void) cudaFreeHost(rank_host_);
        if (stream_) (void) cudaStreamDestroy(stream_);
        if (restore && previous != device_) (void) cudaSetDevice(previous);
    }
    KvGeometry geometry_;
    std::uint32_t query_heads_, representatives_, budget_, local_, input_heads_;
    int device_;
    std::size_t block_elements_ = 0, layer_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    std::vector<void *> banks_;
    std::vector<std::uint64_t> published_;
    struct PendingRank { std::size_t count; std::uint32_t lists, stride; bool shared; };
    std::optional<PendingRank> pending_rank_;
    std::uint64_t rank_ticket_ = 0;
    std::size_t output_capacity_ = 0;
    void * rank_host_ = nullptr;
    std::uint64_t * output_host_ = nullptr, * ids_host_ = nullptr;
    std::uint32_t * costs_host_ = nullptr;
    int * rank_error_ = nullptr;
    std::vector<BuilderState> builders_;
    std::vector<ProbabilityState> probability_states_;
    std::vector<std::uint64_t> probability_lengths_, probability_matured_;
    std::uint32_t probability_initial_ = 0, probability_local_ = 0, probability_top_ = 0, probability_capacity_ = 0;
    std::optional<ProbabilityReceipt> probability_pending_;
    Half * probability_snapshot_keys_ = nullptr;
    float * probability_snapshot_scores_ = nullptr;
    void * probability_host_ = nullptr;
    std::uint64_t * probability_host_positions_ = nullptr;
    double * probability_host_scores_ = nullptr;
    int * probability_host_error_ = nullptr;
    cudaEvent_t probability_captured_ = nullptr, probability_done_ = nullptr;
    std::optional<std::uint32_t> deferred_query_;
    double * mean_query_ = nullptr;
    std::uint64_t * built_positions_ = nullptr;
    double * built_scores_ = nullptr;
    float * query_ = nullptr;
    std::uint64_t * ids_ = nullptr;
    std::uint32_t * costs_ = nullptr;
    double * scores_ = nullptr;
    double * sorted_scores_ = nullptr;
    std::uint64_t * sorted_ids_ = nullptr;
    unsigned char * sort_temp_ = nullptr;
    std::size_t sort_temp_bytes_ = 0;
    bool shared_ranking_ = false;
    bool representative_diagnostics_ = false;
    std::uint64_t * output_device_ = nullptr;
    int * error_device_ = nullptr;
    int error_ = 0;
    bool failed_ = false;
    RankerStats stats_;
};
} // namespace
std::unique_ptr<BlockRanker> make_cuda_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, int device, std::uint32_t local,
                                                   std::uint32_t input_heads, bool representative_diagnostics) {
    return std::make_unique<CudaBlockRanker>(geometry, query_heads, representatives, budget, device, local, input_heads, representative_diagnostics);
}
} // namespace solid
