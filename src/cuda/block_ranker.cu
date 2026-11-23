#include "solid/block_ranker.hpp"
#include "solid/ranking_math.hpp"
#include "solid/representative_math.hpp"
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
__global__ void score_shared_blocks(const __half * keys, const float * query, const std::uint64_t * ids,
                                    double * scores, int * error, std::uint64_t count, std::uint32_t heads,
                                    std::uint32_t group, std::uint32_t representatives, std::uint32_t dimension) {
    __shared__ float warps[8];
    const auto lane = threadIdx.x % 32;
    const auto warp = threadIdx.x / 32;
    const auto width = static_cast<std::uint64_t>(heads) * group * dimension;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x); i < count; i += gridDim.x) {
        const auto * block = keys + ids[i] * heads * representatives * dimension;
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

__global__ void probability_append_keys(const Half * input, Half * ring, double * scores, int * error,
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
__global__ void probability_add_scores(const float * input, double * scores, int * error,
    std::uint64_t first, std::uint32_t count, std::uint32_t heads, std::uint32_t capacity, bool key_major) {
    const auto elements = static_cast<std::uint64_t>(count) * heads;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto key = i / heads, head = i % heads;
        const auto value = input[key_major ? i : head * count + key];
        if (!isfinite(value) || value < 0) atomicExch(error, 1);
        scores[((first + key) % capacity) * heads + head] += value;
    }
}
__global__ void probability_finalize(const Half * ring, const double * scores, Half * top_keys,
    std::uint64_t * top_positions, double * top_scores, Half * bank, std::uint64_t * positions, double * result_scores,
    std::uint64_t first, std::uint64_t end, std::uint64_t first_block, std::uint32_t initial,
    std::uint32_t capacity, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t width,
    std::uint32_t representatives, std::uint32_t block_tokens) {
    const auto head = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= heads) return;
    const auto offset = head * representatives;
    for (auto position = first; position < end; ++position) {
        const auto row = position % capacity;
        const auto * key = ring + (row * kv_heads + head / (heads / kv_heads)) * width;
        insert_representative(position, scores[row * heads + head], key, representatives, width,
            top_positions + offset, top_scores + offset, top_keys + offset * width);
        if ((position + 1 - initial) % block_tokens) continue;
        const auto block = (position - initial) / block_tokens;
        const auto result = ((block - first_block) * heads + head) * representatives;
        for (std::uint32_t d = 0; d < width; ++d) {
            float mean = 0;
            for (std::uint32_t r = 0; r < representatives; ++r) mean += DeviceHalf{}(top_keys[(offset + r) * width + d]);
            bank[(block * heads + head) * width + d] = __half_as_ushort(__float2half(mean / representatives));
        }
        for (std::uint32_t r = 0; r < representatives; ++r) {
            positions[result + r] = top_positions[offset + r];
            result_scores[result + r] = top_scores[offset + r];
            top_positions[offset + r] = UINT64_MAX;
            top_scores[offset + r] = 0;
        }
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
public:
    CudaBlockRanker(KvGeometry geometry, std::uint32_t query_heads, std::uint32_t representatives,
                    std::uint32_t budget, int device, std::uint32_t local, std::uint32_t input_heads) :
        geometry_(geometry), query_heads_(query_heads), representatives_(representatives), budget_(budget), local_(local),
        input_heads_(input_heads ? input_heads : geometry.heads), device_(device) {
        geometry_.validate();
        if (!query_heads || query_heads % geometry.heads || !representatives || representatives > geometry.block_tokens ||
            geometry.heads % input_heads_ || !budget || budget > geometry.capacity_tokens || local > budget || device < 0)
            throw std::invalid_argument("invalid CUDA ranker geometry");
        block_elements_ = product(product(geometry.heads, representatives), geometry.dimension);
        layer_bytes_ = product(product(geometry.blocks_per_head(), block_elements_), sizeof(Half));
        banks_.resize(geometry.layers, nullptr);
        published_.resize(geometry.layers, 0);
        output_.resize(product(geometry.heads, budget));
        if (local_) builders_.resize(geometry.layers);
        stats_.host_bytes = banks_.capacity() * sizeof(void *) + published_.capacity() * sizeof(std::uint64_t) +
                            output_.capacity() * sizeof(std::uint64_t) + sizeof(error_) + builders_.capacity()*sizeof(BuilderState);
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
        probability_capacity_ = local + 512 + geometry_.block_tokens;
        probability_states_.resize(geometry_.layers);
        probability_lengths_.resize(geometry_.layers);
        probability_matured_.assign(geometry_.layers, initial);
        DeviceScope scope(device_);
        const auto before = stats_.device_bytes;
        const auto blocks = (512 + geometry_.block_tokens - 1) / geometry_.block_tokens + 1;
        try {
            allocate(built_positions_, product(product(blocks, geometry_.heads), representatives * sizeof(std::uint64_t)));
            allocate(built_scores_, product(product(blocks, geometry_.heads), representatives * sizeof(double)));
        } catch (...) { fail(); throw; }
        stats_.builder_device_bytes += stats_.device_bytes - before;
        return true;
    }
    bool device_probabilities() const override { return probability_local_ != 0; }
    std::vector<BuiltRepresentatives> probability_update(const ProbabilityFrame & frame) override {
        check();
        if (!probability_local_ || frame.layer >= geometry_.layers || !frame.count || frame.count > 512 ||
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
        auto & state = probability_states_[frame.layer];
        const auto mature_end = std::max<std::uint64_t>(probability_initial_, end - std::min<std::uint64_t>(end, probability_local_));
        const auto first_block = published_[frame.layer];
        const auto block_end = (mature_end - probability_initial_) / geometry_.block_tokens;
        std::vector<BuiltRepresentatives> result(block_end - first_block);
        const auto metadata = result.size() * geometry_.heads * probability_top_;
        std::vector<std::uint64_t> positions(metadata);
        std::vector<double> scores(metadata);
        try {
            if (!state.keys) {
                const auto before = stats_.device_bytes;
                allocate(state.keys, product(product(probability_capacity_, input_heads_), geometry_.dimension * sizeof(Half)));
                allocate(state.scores, product(product(probability_capacity_, geometry_.heads), sizeof(double)));
                allocate(state.top_keys, product(product(geometry_.heads, probability_top_), geometry_.dimension * sizeof(Half)));
                allocate(state.top_positions, product(product(geometry_.heads, probability_top_), sizeof(std::uint64_t)));
                allocate(state.top_scores, product(product(geometry_.heads, probability_top_), sizeof(double)));
                checked(cudaMemsetAsync(state.scores, 0, product(product(probability_capacity_, geometry_.heads), sizeof(double)), stream_), "initialize probability scores");
                checked(cudaMemsetAsync(state.top_positions, 0xff, product(product(geometry_.heads, probability_top_), sizeof(std::uint64_t)), stream_), "initialize probability representatives");
                checked(cudaMemsetAsync(state.top_scores, 0, product(product(geometry_.heads, probability_top_), sizeof(double)), stream_), "initialize representative scores");
                stats_.builder_device_bytes += stats_.device_bytes - before;
            }
            if (!banks_[frame.layer]) allocate(banks_[frame.layer], layer_bytes_);
            checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear probability status");
            const auto key_items = product(product(frame.count, input_heads_), geometry_.dimension);
            const auto score_items = product(end - origin, geometry_.heads);
            const auto key_grid = static_cast<unsigned>(std::min<std::size_t>((key_items + 255) / 256, 256));
            const auto score_grid = static_cast<unsigned>(std::min<std::size_t>((score_items + 255) / 256, 256));
            probability_append_keys<<<key_grid, 256, 0, stream_>>>(frame.kv.data, state.keys, state.scores, error_device_,
                frame.first, frame.count, probability_capacity_, input_heads_, geometry_.heads, geometry_.dimension);
            checked(cudaGetLastError(), "append probability keys");
            probability_add_scores<<<score_grid, 256, 0, stream_>>>(frame.scores.data, state.scores, error_device_,
                origin, static_cast<std::uint32_t>(end - origin), geometry_.heads, probability_capacity_, frame.key_major);
            checked(cudaGetLastError(), "accumulate probability scores");
            probability_finalize<<<static_cast<unsigned>((geometry_.heads + 127) / 128), 128, 0, stream_>>>(
                state.keys, state.scores, state.top_keys, state.top_positions, state.top_scores,
                static_cast<Half *>(banks_[frame.layer]), built_positions_, built_scores_,
                probability_matured_[frame.layer], mature_end, first_block, probability_initial_, probability_capacity_,
                geometry_.heads, input_heads_, geometry_.dimension, probability_top_, geometry_.block_tokens);
            checked(cudaGetLastError(), "finalize probability representatives");
            if (metadata) {
                checked(cudaMemcpyAsync(positions.data(), built_positions_, metadata * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream_), "download representative positions");
                checked(cudaMemcpyAsync(scores.data(), built_scores_, metadata * sizeof(double), cudaMemcpyDeviceToHost, stream_), "download representative scores");
            }
            checked(cudaMemcpyAsync(&error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download probability status");
            checked(cudaStreamSynchronize(stream_), "complete probability receipt");
            if (error_) throw std::runtime_error("non-finite probability keys or scores");
        } catch (...) { fail(); throw; }
        const auto stride = static_cast<std::size_t>(geometry_.heads) * probability_top_;
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i].block = first_block + i;
            result[i].positions.assign(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
            result[i].scores.assign(scores.begin() + i * stride, scores.begin() + (i + 1) * stride);
        }
        published_[frame.layer] = block_end;
        probability_lengths_[frame.layer] = end;
        probability_matured_[frame.layer] = mature_end;
        stats_.published_blocks += result.size();
        stats_.built_blocks += result.size();
        stats_.device_builder_appends += frame.count;
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
        validate_device_query(query, DeviceMemory::cuda, query_heads_, geometry_.dimension);
        validate_pointer(query.data);
        auto result = rank_impl(layer, query.data, candidates, true);
        ++stats_.device_rank_queries;
        return result;
    }
    std::vector<std::vector<std::uint64_t>> rank_impl(std::uint32_t layer, const float * query,
                                                    const BlockCandidates & candidates, bool device) {
        check();
        if (deferred_query_ && (!device || *deferred_query_!=layer)) throw std::logic_error("ranking does not match deferred representative query");
        if (layer >= geometry_.layers ||
            candidates.blocks.size() != candidates.tokens.size() || candidates.blocks.size() > geometry_.blocks_per_head() ||
            candidates.remaining > budget_) throw std::invalid_argument("CUDA ranking shape or budget");
        for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
            if (candidates.blocks[i] >= published_[layer] || (i && candidates.blocks[i] <= candidates.blocks[i-1]) ||
                !candidates.tokens[i] || candidates.tokens[i] > geometry_.block_tokens) throw std::invalid_argument("invalid CUDA rank candidate");
        }
        std::vector<std::vector<std::uint64_t>> result(geometry_.heads);
        if (!candidates.remaining || candidates.blocks.empty()) {
            finish_query_for_rank_device();
            if (device) {
                DeviceScope scope(device_);
                try {
                    checked(cudaMemsetAsync(error_device_,0,sizeof(int),stream_),"clear empty rank status");
                    const auto count=product(query_heads_,geometry_.dimension);
                    const auto grid=static_cast<unsigned>(std::min<std::size_t>((count+255)/256,65535));
                    validate_query_values<<<grid,256,0,stream_>>>(query,count,error_device_);
                    checked(cudaGetLastError(),"validate empty rank query");
                    checked(cudaMemcpyAsync(&error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream_),"download empty rank status");
                    checked(cudaStreamSynchronize(stream_),"complete empty rank validation");
                    if (error_) throw std::runtime_error("non-finite device ranking query");
                } catch (...) { fail(); throw; }
            }
            return result;
        }
        const auto count = candidates.blocks.size();
        const auto smallest = *std::min_element(candidates.tokens.begin(), candidates.tokens.end());
        const auto max_selected = (static_cast<std::uint64_t>(candidates.remaining) + smallest - 1) / smallest;
        const auto stride = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_selected, count));
        const auto lists = candidates.shared_heads ? 1u : geometry_.heads;
        const auto bytes = product(product(lists, stride), sizeof(std::uint64_t));
        DeviceScope scope(device_);
        try {
            if (!device) {
                const auto bytes = product(product(query_heads_, geometry_.dimension), sizeof(float));
                checked(cudaMemcpyAsync(query_, query, bytes, cudaMemcpyHostToDevice, stream_), "upload rank query");
                stats_.query_upload_bytes += bytes;
            }
            checked(cudaMemcpyAsync(ids_, candidates.blocks.data(), product(count, sizeof(std::uint64_t)), cudaMemcpyHostToDevice, stream_), "upload rank candidates");
            checked(cudaMemcpyAsync(costs_, candidates.tokens.data(), product(count, sizeof(std::uint32_t)), cudaMemcpyHostToDevice, stream_), "upload rank costs");
            if (!deferred_query_) checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear rank status");
            checked(cudaMemsetAsync(output_device_, 0xff, bytes, stream_), "clear rank output");
            if (candidates.shared_heads) {
                const auto grid = static_cast<unsigned>(std::min<std::uint64_t>(count, 65535));
                score_shared_blocks<<<grid, 256, 0, stream_>>>(static_cast<const __half *>(banks_[layer]), device ? query : query_,
                    ids_, scores_, error_device_, count, geometry_.heads, query_heads_ / geometry_.heads, representatives_, geometry_.dimension);
            } else {
                const auto grid = static_cast<unsigned>(std::min<std::uint64_t>((count * geometry_.heads + 7) / 8, 65535));
                score_blocks<<<grid, 256, 0, stream_>>>(static_cast<const __half *>(banks_[layer]), device ? query : query_, ids_, scores_, error_device_,
                    count, geometry_.heads, query_heads_ / geometry_.heads, representatives_, geometry_.dimension);
            }
            checked(cudaGetLastError(), "launch block relevance kernel");
            select_top_blocks<<<lists, 256, 0, stream_>>>(scores_, ids_, costs_, output_device_, count, candidates.remaining, stride);
            checked(cudaGetLastError(), "launch block top-k kernel");
            checked(cudaMemcpyAsync(output_.data(), output_device_, bytes, cudaMemcpyDeviceToHost, stream_), "download selected block IDs");
            checked(cudaMemcpyAsync(&error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download rank status");
            checked(cudaStreamSynchronize(stream_), "complete CUDA block ranking");
            if (deferred_query_) ++stats_.builder_rank_shared_waits;
            deferred_query_.reset();
            if (error_) throw std::runtime_error("non-finite CUDA representative score or block relevance");
        } catch (...) { fail(); throw; }
        for (std::uint32_t head = 0; head < lists; ++head) {
            for (std::uint32_t i = 0; i < stride; ++i) {
                const auto block = output_[static_cast<std::size_t>(head) * stride + i];
                if (block == UINT64_MAX) break;
                result[head].push_back(block);
            }
        }
        if (candidates.shared_heads) std::fill(result.begin() + 1, result.end(), result.front());
        ++stats_.calls;
        stats_.scored_blocks += count * lists;
        shared_ranking_ = candidates.shared_heads;
        return result;
    }
    RankerStats stats() const override { return stats_; }
    const char * name() const override { return shared_ranking_ ? "cuda_shared_fp32_topk" : "cuda_fp64_topk"; }
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
    void check() const { if (failed_) throw std::runtime_error("CUDA block ranker failed"); }
    void fail() noexcept { failed_ = true; if (stream_) (void) cudaStreamSynchronize(stream_); deferred_query_.reset(); }
    void release() noexcept {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device_) != cudaSuccess) return;
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
            if (state.top_keys) (void) cudaFree(state.top_keys);
            if (state.top_positions) (void) cudaFree(state.top_positions);
            if (state.top_scores) (void) cudaFree(state.top_scores);
        }
        if (mean_query_) (void) cudaFree(mean_query_);
        if (built_positions_) (void) cudaFree(built_positions_);
        if (built_scores_) (void) cudaFree(built_scores_);
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
    std::uint32_t query_heads_, representatives_, budget_, local_, input_heads_;
    int device_;
    std::size_t block_elements_ = 0, layer_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    std::vector<void *> banks_;
    std::vector<std::uint64_t> published_, output_;
    std::vector<BuilderState> builders_;
    std::vector<BuilderState> probability_states_;
    std::vector<std::uint64_t> probability_lengths_, probability_matured_;
    std::uint32_t probability_initial_ = 0, probability_local_ = 0, probability_top_ = 0, probability_capacity_ = 0;
    std::optional<std::uint32_t> deferred_query_;
    double * mean_query_ = nullptr;
    std::uint64_t * built_positions_ = nullptr;
    double * built_scores_ = nullptr;
    float * query_ = nullptr;
    std::uint64_t * ids_ = nullptr;
    std::uint32_t * costs_ = nullptr;
    double * scores_ = nullptr;
    bool shared_ranking_ = false;
    std::uint64_t * output_device_ = nullptr;
    int * error_device_ = nullptr;
    int error_ = 0;
    bool failed_ = false;
    RankerStats stats_;
};
} // namespace
std::unique_ptr<BlockRanker> make_cuda_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, int device, std::uint32_t local,
                                                   std::uint32_t input_heads) {
    return std::make_unique<CudaBlockRanker>(geometry, query_heads, representatives, budget, device, local, input_heads);
}
} // namespace solid
