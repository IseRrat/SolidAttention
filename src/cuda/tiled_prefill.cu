#include "solid/tiled_prefill.hpp"
#include "solid/trace.hpp"
#include "solid/attention.hpp"
#include "solid/online_attention.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace solid {
namespace {
void checked(cudaError_t status,const char * operation) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void checked(cublasStatus_t status, const char * operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
    }
}
std::size_t product(std::size_t a,std::size_t b) {
    if (b && a>SIZE_MAX/b) throw std::overflow_error("CUDA prefill allocation overflow");
    return a*b;
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        checked(cudaGetDevice(&previous_),"get prefill device");
        if (previous_!=device) { checked(cudaSetDevice(device),"select prefill device"); changed_=true; }
    }
    ~DeviceScope() { if (changed_) (void) cudaSetDevice(previous_); }
private:
    int previous_=0; bool changed_=false;
};
__global__ void initialize(float * weights,float * maximum,float * sum,std::uint64_t rows,std::uint32_t width) {
    for (auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*width;
         i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        weights[i]=0;
        if (i<rows) { maximum[i]=-CUDART_INF_F; sum[i]=0; }
    }
}
// Four warps process four consecutive queries for one query head. Each CTA
// loads an eight-token K/V panel once, then all four queries reuse that panel.
// The maximum shared footprint is 32 KiB at dimension 1024.
template<int Width>
__global__ void accumulate(const float * query,const __half * kv,float * weights,float * maximum,float * sum,int * error,
    std::uint64_t first_query,std::uint64_t first_key,std::uint32_t valid,std::uint32_t queries,
    std::uint32_t query_heads,std::uint32_t group,std::uint32_t width,std::size_t token_stride,std::size_t head_stride,float scale) {
    constexpr unsigned queries_per_cta=4;
    constexpr unsigned panel_tokens=8;
    constexpr int fragments=(Width+31)/32;
    __shared__ __half panel[panel_tokens*2*Width];
    const auto lane=threadIdx.x%32;
    const auto warp=threadIdx.x/32;
    const auto query_groups=(static_cast<std::uint64_t>(queries)+queries_per_cta-1)/queries_per_cta;
    const auto jobs=query_groups*query_heads;
    for (auto job=static_cast<std::uint64_t>(blockIdx.x);job<jobs;job+=gridDim.x) {
        const auto query_head=job%query_heads;
        const auto query_start=(job/query_heads)*queries_per_cta;
        const auto query_index=query_start+warp;
        const bool active=query_index<queries;
        const auto row=query_index*query_heads+query_head;
        const auto position=first_query+query_index;
        const auto last_query=min(query_start+queries_per_cta,static_cast<std::uint64_t>(queries))-1;
        const auto last_position=first_query+last_query;
        // This branch is uniform across the whole CTA, including inactive warps.
        if (first_key>last_position) continue;
        const auto count=static_cast<std::uint32_t>(min(static_cast<std::uint64_t>(valid),last_position-first_key+1));
        const auto head=query_head/group;
        float q[fragments]{}, output[fragments]{};
        for (int r=0;r<fragments;++r) {
            const auto d=lane+32*r;
            if (active && d<width) {
                q[r]=query[row*width+d];
                output[r]=weights[row*width+d];
            }
        }
        auto m=active?maximum[row]:-CUDART_INF_F;
        auto denominator=active?sum[row]:0.f;
        bool finite_row=true;
        for (std::uint32_t first=0;first<count;first+=panel_tokens) {
            const auto tokens=min(panel_tokens,count-first);
            const auto elements=tokens*2*width;
            for (auto i=threadIdx.x;i<elements;i+=blockDim.x) {
                const auto token=i/(2*width);
                const auto offset=i%(2*width);
                panel[i]=kv[head*head_stride+(first+token)*token_stride+offset];
            }
            __syncthreads();
            for (std::uint32_t token=0;token<tokens;++token) {
                if (!active || !finite_row || first_key+first+token>position) continue;
                const auto base=token*2*width;
                float score=0;
                for (int r=0;r<fragments;++r) {
                    const auto d=lane+32*r;
                    if (d<width) score=fmaf(q[r],__half2float(panel[base+d]),score);
                }
                for (int delta=16;delta;delta/=2) score+=__shfl_down_sync(0xffffffff,score,delta);
                score=__shfl_sync(0xffffffff,score,0)*scale;
                if (!isfinite(score)) {
                    if (!lane) atomicExch(error,1);
                    finite_row=false;
                    continue;
                }
                const auto update=online_attention_update(m,denominator,score);
                m=update.maximum;
                denominator=update.sum;
                for (int r=0;r<fragments;++r) {
                    const auto d=lane+32*r;
                    if (d<width) output[r]=output[r]*update.previous_scale+__half2float(panel[base+width+d])*update.weight;
                }
            }
            // Failed or masked query warps still participate in this barrier.
            __syncthreads();
        }
        if (active) {
            if (!lane) { maximum[row]=m; sum[row]=denominator; }
            for (int r=0;r<fragments;++r) {
                const auto d=lane+32*r;
                if (d<width) weights[row*width+d]=output[r];
            }
        }
    }
}
__global__ void normalize(float * values,const float * sum,int * error,std::uint64_t elements,std::uint32_t width) {
    for (auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;
         i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto denominator=sum[i/width];
        const auto value=values[i]/denominator;
        if (!(denominator>0) || !isfinite(value)) { atomicExch(error,1); values[i]=0; }
        else values[i]=value;
    }
}

__global__ void pack_query(const float * input, __half * packed, std::uint32_t heads,
                           std::uint32_t queries, std::uint32_t width) {
    const auto size = static_cast<std::uint64_t>(heads) * queries * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto q = i / width % queries;
        const auto h = i / width / queries;
        packed[i] = __float2half(input[(q * heads + h) * width + i % width]);
    }
}

__global__ void pack_kv(const __half * input, __half * keys, __half * values, std::uint32_t heads,
                        std::uint32_t tokens, std::uint32_t width, std::size_t token_stride, std::size_t head_stride,
                        const __half * current, std::uint32_t historical) {
    const auto size = static_cast<std::uint64_t>(heads) * tokens * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto head = i / width / tokens, token = i / width % tokens, column = i % width;
        const bool on_device = current && token >= historical;
        const auto * source = on_device ?
            current + ((token - historical) * heads + head) * 2 * width + column :
            input + head * head_stride + token * token_stride + column;
        keys[i] = *source;
        values[i] = source[width];
    }
}
__global__ void mean_sparse_queries(const float * queries, float * mean, std::size_t width, std::uint32_t count) {
    for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < width; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        float value = 0.f;
        for (std::uint32_t q = 0; q < count; ++q) value += queries[q * width + i] / count;
        mean[i] = value;
    }
}

// Each block merges one tile into the running FP32 softmax. Probabilities
// are packed for the tensor-core V product. Masked columns contribute zero.
__global__ void merge_scores(const float * scores, __half * probabilities, float * maximum, float * sum,
                             float * previous_scale, int * error, std::uint64_t first_query, std::uint64_t first_key,
                             std::uint32_t queries, std::uint32_t heads, std::uint32_t tokens) {
    __shared__ float reduce[256];
    __shared__ float next_maximum;
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    const auto query = row % queries;
    const auto output_row = query * heads + row / queries;
    const auto causal = first_query + query;
    const auto valid = first_key > causal ? 0u :
        static_cast<std::uint32_t>(min(static_cast<std::uint64_t>(tokens), causal - first_key + 1));
    if (!valid) {
        if (!threadIdx.x) previous_scale[row] = 1;
        for (auto i = threadIdx.x; i < tokens; i += blockDim.x) probabilities[row * tokens + i] = __float2half(0);
        return;
    }
    float local_maximum = -CUDART_INF_F;
    for (auto i = threadIdx.x; i < valid; i += blockDim.x) {
        const auto value = scores[row * tokens + i];
        if (isfinite(value)) local_maximum = fmaxf(local_maximum, value);
        else atomicExch(error, 1);
    }
    reduce[threadIdx.x] = local_maximum;
    __syncthreads();
    for (auto span = blockDim.x / 2; span; span /= 2) {
        if (threadIdx.x < span) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + span]);
        __syncthreads();
    }
    if (!threadIdx.x) {
        const auto old_maximum = maximum[output_row];
        next_maximum = fmaxf(old_maximum, reduce[0]);
        previous_scale[row] = sum[output_row] > 0 ? expf(old_maximum - next_maximum) : 0;
        maximum[output_row] = next_maximum;
    }
    __syncthreads();
    float local_sum = 0;
    for (auto i = threadIdx.x; i < tokens; i += blockDim.x) {
        const auto weight = i < valid ? expf(scores[row * tokens + i] - next_maximum) : 0;
        probabilities[row * tokens + i] = __float2half(weight);
        local_sum += weight;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (auto span = blockDim.x / 2; span; span /= 2) {
        if (threadIdx.x < span) reduce[threadIdx.x] += reduce[threadIdx.x + span];
        __syncthreads();
    }
    if (!threadIdx.x) sum[output_row] = previous_scale[row] * sum[output_row] + reduce[0];
}

__global__ void merge_values(const float * tile, const float * previous_scale, float * output,
                             std::uint32_t queries, std::uint32_t heads, std::uint32_t width) {
    const auto elements = static_cast<std::uint64_t>(queries) * heads * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto row = i / width;
        const auto destination = (row % queries * heads + row / queries) * width + i % width;
        output[destination] = output[destination] * previous_scale[row] + tile[i];
    }
}

__global__ void rotary_frequencies(float * frequencies, std::uint32_t dimensions, float base) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < dimensions / 2; i += gridDim.x * blockDim.x) {
        frequencies[i] = powf(base, -2.f * i / dimensions);
    }
}

__global__ void pack_local_queries(const float * query, __half * packed, const float * frequencies,
    std::uint64_t first, std::uint64_t origin, std::uint32_t local, std::uint32_t heads, std::uint32_t count,
    std::uint32_t width, std::uint32_t rotary, bool split_halves) {
    const auto size = static_cast<std::uint64_t>(heads) * count * width;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto q = i / width % count, head = i / width / count, d = i % width;
        const auto * row = query + (q * heads + head) * width;
        float value = row[d];
        if (d < rotary) {
            const auto pair = split_halves ? d % (rotary / 2) : d / 2;
            const bool second = split_halves ? d >= rotary / 2 : d % 2;
            const auto partner = split_halves ? (second ? pair : pair + rotary / 2) : d ^ 1;
            const auto delta = static_cast<std::int64_t>(first + q - origin) - (local - 1);
            float sine, cosine;
            sincosf(static_cast<float>(delta) * frequencies[pair], &sine, &cosine);
            value = value * cosine + (second ? row[partner] : -row[partner]) * sine;
        }
        packed[i] = __float2half(value);
    }
}

__global__ void rotate_local_keys(__half * keys, const std::uint64_t * positions, const float * frequencies,
    std::uint64_t origin, std::uint32_t tokens, std::uint32_t heads, std::uint32_t width, std::uint32_t rotary, bool split_halves) {
    const auto size = static_cast<std::uint64_t>(heads) * tokens * (rotary / 2);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto pair = i % (rotary / 2), row = i / (rotary / 2);
        const auto position = positions[row % tokens];
        if (position < origin) continue;
        const auto x = row * width + (split_halves ? pair : 2 * pair);
        const auto y = x + (split_halves ? rotary / 2 : 1);
        const auto a = __half2float(keys[x]), b = __half2float(keys[y]);
        float sine, cosine;
        sincosf(static_cast<float>(position - origin) * frequencies[pair], &sine, &cosine);
        keys[x] = __float2half(a * cosine - b * sine);
        keys[y] = __float2half(a * sine + b * cosine);
    }
}

__global__ void sparse_softmax(const float * global_scores, const float * local_scores, __half * probabilities,
    const std::uint64_t * positions, int * error, std::uint64_t first,
    std::uint32_t count, std::uint32_t local, std::uint32_t tokens,
    std::uint32_t local_only_from) {
    __shared__ float reduction[256];
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    const auto p = first + row % count;
    const auto * global = global_scores + row * tokens;
    const auto * nearby = local_scores + row * tokens;
    auto * weights = probabilities + row * tokens;
    float maximum = -CUDART_INF_F;
    for (auto i = threadIdx.x; i < tokens; i += blockDim.x) {
        if (positions[i] > p) break;
        if (i >= local_only_from && p - positions[i] >= local) continue;
        const auto score = p - positions[i] < local ? nearby[i] : global[i];
        if (isfinite(score)) maximum = fmaxf(maximum, score);
        else atomicExch(error, 1);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (auto span = blockDim.x / 2; span; span /= 2) {
        if (threadIdx.x < span) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + span]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0;
    for (auto i = threadIdx.x; i < tokens; i += blockDim.x) {
        if (positions[i] > p) break;
        if (i >= local_only_from && p - positions[i] >= local) continue;
        const auto score = p - positions[i] < local ? nearby[i] : global[i];
        sum += expf(score - maximum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (auto span = blockDim.x / 2; span; span /= 2) {
        if (threadIdx.x < span) reduction[threadIdx.x] += reduction[threadIdx.x + span];
        __syncthreads();
    }
    sum = reduction[0];
    if (!(sum > 0) || !isfinite(sum)) {
        if (!threadIdx.x) atomicExch(error, 1);
        for (auto i = threadIdx.x; i < tokens; i += blockDim.x) weights[i] = __float2half(0);
        return;
    }
    for (auto i = threadIdx.x; i < tokens; i += blockDim.x) {
        const bool causal = positions[i] <= p;
        const bool is_local = causal && p - positions[i] < local;
        const bool valid = causal && (i < local_only_from || is_local);
        weights[i] = __float2half(valid ? expf((is_local ? nearby[i] : global[i]) - maximum) / sum : 0);
    }
}

__global__ void sum_sparse_local_scores(const __half * probabilities, float * scores, std::uint64_t first,
    std::uint64_t origin, std::uint32_t queries, std::uint32_t heads, std::uint32_t local,
    std::uint32_t tokens, std::uint32_t local_begin) {
    const auto key_count = tokens - local_begin;
    const auto size = static_cast<std::uint64_t>(heads) * key_count;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto head = i / key_count, key = i % key_count;
        const auto position = origin + key;
        const auto begin = position > first ? position - first : 0;
        const auto end = min(static_cast<std::uint64_t>(queries), position + local - first);
        const auto * weights = probabilities + (head * queries + begin) * tokens + local_begin + key;
        float sum = 0;
        for (auto query = begin; query < end; ++query) sum += __half2float(weights[(query - begin) * tokens]);
        scores[key * heads + head] = sum;
    }
}

__global__ void scatter_sparse_output(const float * input, float * output, int * error, std::uint32_t width,
                                      std::uint32_t heads, std::uint32_t count) {
    const auto size = static_cast<std::uint64_t>(width) * heads * count;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const auto row = i / width;
        const auto value = input[i];
        if (!isfinite(value)) atomicExch(error, 1);
        output[(row % count * heads + row / count) * width + i % width] = value;
    }
}

class CudaTiledPrefill final:public TiledPrefill {
public:
    CudaTiledPrefill(KvGeometry geometry,std::uint32_t query_heads,std::uint32_t queries,std::uint32_t tile_tokens,int device,
                     std::uint32_t sparse_local):
        g_(geometry),qh_(query_heads),queries_(queries),sparse_local_(sparse_local),device_(device) {
        g_.validate();
        if (!qh_ || qh_%g_.heads || !queries || queries>g_.capacity_tokens || !tile_tokens || device<0 || g_.dimension>1024) {
            throw std::invalid_argument("invalid CUDA tiled prefill geometry");
        }
        const auto tile=(std::min<std::uint64_t>(tile_tokens,g_.capacity_tokens)+g_.block_tokens-1)/g_.block_tokens*g_.block_tokens;
        if (tile>UINT32_MAX) throw std::overflow_error("prefill tile capacity");
        tile_=static_cast<std::uint32_t>(tile);
        stats_.query_capacity=queries_; stats_.tile_capacity=tile_;
        tile_elements_=product(product(product(g_.heads,tile_),2),g_.dimension);
        const auto rows=product(queries_,qh_);
        DeviceScope scope(device_);
        try {
            checked(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create prefill stream");
            trace::name_cuda_stream(stream_, "prefill.attention");
            checked(cudaEventCreateWithFlags(&copied_,cudaEventDisableTiming),"create prefill upload event");
            allocate(kv_,product(tile_elements_,sizeof(Half)),stats_.kv_device_bytes);
            allocate(query_,product(product(rows,g_.dimension),sizeof(float)),stats_.workspace_device_bytes);
            allocate(weights_,product(product(rows,g_.dimension),sizeof(float)),stats_.workspace_device_bytes);
            allocate(maximum_,product(rows,sizeof(float)),stats_.workspace_device_bytes);
            allocate(sum_,product(rows,sizeof(float)),stats_.workspace_device_bytes);
            allocate(error_device_,sizeof(int),stats_.workspace_device_bytes);
            const auto cells = product(rows, tile_);
            // Sparse chunks include the union of their sliding windows as
            // well as the selected historical rows.
            const std::size_t max_cells = (sparse_local_ ? 64u : 16u) << 20;
            const bool gemm = g_.dimension % 16 == 0 && cells <= max_cells;
            if (sparse_local_ && (!gemm || sparse_local_ > tile_)) throw std::invalid_argument("invalid CUDA sparse prefill geometry");
            if ((queries_ >= 32 || sparse_local_) && gemm) {
                checked(cublasCreate(&blas_), "create prefill cuBLAS handle");
                checked(cublasSetStream(blas_, stream_), "bind prefill cuBLAS stream");
                allocate(packed_query_, product(product(rows, g_.dimension), sizeof(Half)), stats_.workspace_device_bytes);
                allocate(packed_kv_, product(tile_elements_, sizeof(Half)), stats_.kv_device_bytes);
                allocate(scores_, product(cells, sizeof(float)), stats_.workspace_device_bytes);
                allocate(probabilities_, product(cells, sizeof(Half)), stats_.workspace_device_bytes);
                allocate(tile_output_, product(product(rows, g_.dimension), sizeof(float)), stats_.workspace_device_bytes);
                allocate(previous_scale_, product(rows, sizeof(float)), stats_.workspace_device_bytes);
            }
            if (sparse_local_) {
                allocate(local_scores_, product(cells, sizeof(float)), stats_.workspace_device_bytes);
                allocate(local_feedback_, product(product(qh_, tile_), sizeof(float)), stats_.workspace_device_bytes);
                allocate(key_positions_, product(tile_, sizeof(std::uint64_t)), stats_.workspace_device_bytes);
                allocate(frequencies_, product(g_.dimension / 2, sizeof(float)), stats_.workspace_device_bytes);
            }
            stats_.kv_device_peak_bytes=stats_.kv_device_bytes;
            stats_.workspace_device_peak_bytes=stats_.workspace_device_bytes;
        } catch (...) { release(); throw; }
    }
    ~CudaTiledPrefill() override { release(); }
    std::uint32_t query_capacity() const override { return queries_; }
    std::uint32_t tile_capacity() const override { return tile_; }
    const char * name() const override { return sparse_local_ ? "cuda_infllm_batched" : blas_ ? "cuda_cublas_tiled" : "cuda_online_tiled"; }
    bool supports_sparse() const override { return sparse_local_ != 0; }
    std::optional<DeviceMemory> direct_memory() const override { return DeviceMemory::cuda; }
    PrefillStats stats() const override { return stats_; }
    void begin(std::uint64_t first,std::uint32_t queries,float scale,const std::vector<float> & query) override {
        if (query.size()!=product(product(queries,qh_),g_.dimension)) throw std::invalid_argument("prefill query shape");
        for (const auto value:query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite prefill query");
        begin_impl(first,queries,scale,query.data(),false);
    }
    void begin_device(std::uint64_t first,std::uint32_t queries,float scale,DeviceQuery query) override {
        validate_device_query(query,DeviceMemory::cuda,qh_,g_.dimension,queries); validate_pointer(query.data);
        begin_impl(first,queries,scale,query.data,true); stats_.device_query_tokens+=queries;
    }
    void begin_impl(std::uint64_t first,std::uint32_t queries,float scale,const float * query,bool device) {
        check();
        if (open_ || !queries || queries>queries_ || first>g_.capacity_tokens-queries ||
            !std::isfinite(scale) || scale<=0) throw std::invalid_argument("prefill frame shape, position or scale");
        DeviceScope scope(device_);
        try {
            if (!device) {
                const auto bytes=product(product(product(queries,qh_),g_.dimension),sizeof(float));
                checked(cudaMemcpyAsync(query_,query,bytes,cudaMemcpyHostToDevice,stream_),"upload prefill Q");
                checked(cudaEventRecord(copied_,stream_),"record prefill Q completion"); stats_.query_upload_bytes+=bytes;
            }
            checked(cudaMemsetAsync(error_device_,0,sizeof(int),stream_),"clear prefill status");
            const auto rows=static_cast<std::uint64_t>(queries)*qh_;
            initialize<<<grid(rows*g_.dimension),256,0,stream_>>>(weights_,maximum_,sum_,rows,g_.dimension);
            checked(cudaGetLastError(),"initialize online prefill state");
            if (!device) checked(cudaEventSynchronize(copied_),"complete prefill Q upload");
            current_query_=device?query:query_;
            first_=first; count_=queries; scale_=scale; next_key_=0; open_=true;
            if (blas_ && count_ >= 32) {
                pack_query<<<grid(rows * g_.dimension), 256, 0, stream_>>>(current_query_, packed_query_, qh_, count_, g_.dimension);
                checked(cudaGetLastError(), "pack prefill queries for GEMM");
            }
        } catch (...) { fail(); throw; }
    }
    void consume(std::uint64_t first,std::uint32_t valid,const std::vector<Half> & kv) override {
        if (kv.size()!=tile_elements_) throw std::invalid_argument("prefill host tile shape");
        consume_impl(first,valid,kv.data(),false);
    }
    void consume_device(std::uint64_t first,std::uint32_t valid,DeviceKvBatch kv) override {
        validate_device_kv(kv,DeviceMemory::cuda,g_.heads,g_.dimension,valid); validate_pointer(kv.data);
        consume_impl(first,valid,kv.data,true);
    }
    void consume_impl(std::uint64_t first,std::uint32_t valid,const Half * kv,bool device) {
        check();
        const auto end=first_+count_;
        if (!open_ || first!=next_key_ || first>=end || !valid || valid>std::min<std::uint64_t>(tile_,end-first)) {
            throw std::invalid_argument("prefill tile order, shape or valid rows");
        }
        DeviceScope scope(device_);
        try {
            const auto bytes=product(tile_elements_,sizeof(Half));
            if (!device) {
                checked(cudaMemcpyAsync(kv_,kv,bytes,cudaMemcpyHostToDevice,stream_),"upload prefill KV tile");
                checked(cudaEventRecord(copied_,stream_),"record prefill tile upload");
            }
            const auto jobs=((static_cast<std::uint64_t>(count_)+3)/4)*qh_;
            const auto blocks=static_cast<unsigned>(std::min<std::uint64_t>(jobs,65535));
            const auto token_stride=product(2,g_.dimension)*(device?g_.heads:1);
            const auto head_stride=product(2,g_.dimension)*(device?1:tile_);
            if (blas_ && count_ >= 32) {
                consume_gemm(first, valid, device ? kv : kv_, token_stride, head_stride);
            } else {
#define SOLID_PREFILL_LAUNCH(W) accumulate<W><<<blocks,128,0,stream_>>>(current_query_,reinterpret_cast<const __half *>(device?kv:kv_),weights_,maximum_,sum_,error_device_,first_,first,valid,count_,qh_,qh_/g_.heads,g_.dimension,token_stride,head_stride,scale_)
            if (g_.dimension<=32) { SOLID_PREFILL_LAUNCH(32); }
            else if (g_.dimension<=64) { SOLID_PREFILL_LAUNCH(64); }
            else if (g_.dimension<=128) { SOLID_PREFILL_LAUNCH(128); }
            else if (g_.dimension<=256) { SOLID_PREFILL_LAUNCH(256); }
            else if (g_.dimension<=512) { SOLID_PREFILL_LAUNCH(512); }
            else { SOLID_PREFILL_LAUNCH(1024); }
#undef SOLID_PREFILL_LAUNCH
                checked(cudaGetLastError(),"launch causal tiled prefill");
            }
            // The source lease ends after H2D. The kernel may overlap the next
            // host read; the same stream orders the next overwrite of kv_.
            if (!device) { checked(cudaEventSynchronize(copied_),"complete prefill tile upload"); stats_.upload_bytes+=bytes; }
            else stats_.device_kv_tokens+=valid;
            next_key_+=valid; ++stats_.tiles; stats_.kv_tokens+=valid;
        } catch (...) { fail(); throw; }
    }
    void multiply_scores(std::uint32_t valid, std::uint32_t queries, float scale, float * output,
                         std::uint32_t first_key, std::uint32_t key_count) {
        if (!key_count) return;
        const auto width = static_cast<int>(g_.dimension);
        const auto tokens = static_cast<int>(valid);
        const auto queries_per_head = static_cast<int>(queries * (qh_ / g_.heads));
        const auto kv_stride = static_cast<long long>(width) * tokens;
        const auto q_stride = static_cast<long long>(width) * queries_per_head;
        const auto score_stride = static_cast<long long>(tokens) * queries_per_head;
        const float zero = 0;
        checked(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, key_count, queries_per_head, width,
            &scale, packed_kv_ + static_cast<std::size_t>(first_key) * width, CUDA_R_16F, width, kv_stride,
            packed_query_, CUDA_R_16F, width, q_stride,
            &zero, output + first_key, CUDA_R_32F, tokens, score_stride, g_.heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
            "multiply prefill keys and queries");
    }
    void multiply_values(std::uint32_t valid, std::uint32_t queries) {
        const auto width = static_cast<int>(g_.dimension);
        const auto tokens = static_cast<int>(valid);
        const auto queries_per_head = static_cast<int>(queries * (qh_ / g_.heads));
        const auto kv_stride = static_cast<long long>(width) * tokens;
        const auto q_stride = static_cast<long long>(width) * queries_per_head;
        const auto score_stride = static_cast<long long>(tokens) * queries_per_head;
        const auto * values = packed_kv_ + kv_stride * g_.heads;
        const float zero = 0, one = 1;
        checked(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, width, queries_per_head, tokens,
            &one, values, CUDA_R_16F, width, kv_stride, probabilities_, CUDA_R_16F, tokens, score_stride,
            &zero, tile_output_, CUDA_R_32F, width, q_stride, g_.heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
            "multiply prefill values and probabilities");
    }
    void pack_gemm_kv(const Half * kv, std::uint32_t valid, std::size_t token_stride, std::size_t head_stride,
                      const Half * current = nullptr, std::uint32_t historical = 0) {
        const auto elements = product(product(valid, g_.heads), g_.dimension);
        pack_kv<<<grid(elements), 256, 0, stream_>>>(reinterpret_cast<const __half *>(kv), packed_kv_, packed_kv_ + elements,
            g_.heads, valid, g_.dimension, token_stride, head_stride, reinterpret_cast<const __half *>(current), historical);
        checked(cudaGetLastError(), "pack prefill keys and values for GEMM");
    }
    void consume_gemm(std::uint64_t first, std::uint32_t valid, const Half * kv, std::size_t token_stride,
                      std::size_t head_stride) {
        const auto rows = static_cast<std::uint64_t>(count_) * qh_;
        pack_gemm_kv(kv, valid, token_stride, head_stride);
        multiply_scores(valid, count_, scale_, scores_, 0, valid);
        merge_scores<<<static_cast<unsigned>(rows), 256, 0, stream_>>>(scores_, probabilities_, maximum_, sum_,
            previous_scale_, error_device_, first_, first, count_, qh_, valid);
        checked(cudaGetLastError(), "merge causal prefill softmax");
        multiply_values(valid, count_);
        merge_values<<<grid(rows * g_.dimension), 256, 0, stream_>>>(tile_output_, previous_scale_, weights_,
            count_, qh_, g_.dimension);
        checked(cudaGetLastError(), "merge prefill value tiles");
    }
    DeviceQuery mean_sparse_query(DeviceQuery query, std::uint32_t count) override {
        if (!supports_sparse() || open_ || count > queries_) throw std::invalid_argument("sparse prefill mean frame");
        validate_device_query(query, DeviceMemory::cuda, qh_, g_.dimension, count);
        validate_pointer(query.data);
        DeviceScope scope(device_);
        const auto width = query.elements / count;
        try {
            mean_sparse_queries<<<grid(width), 256, 0, stream_>>>(query.data, query_, width, count);
            checked(cudaGetLastError(), "average sparse prefill queries");
            checked(cudaStreamSynchronize(stream_), "make sparse retrieval query available");
        } catch (...) { fail(); throw; }
        return {query_, width, DeviceMemory::cuda};
    }
    SparsePrefillOutput sparse(const SparsePrefillFrame & frame) override {
        check();
        const bool direct = frame.device_query.data != nullptr;
        const auto rows = product(frame.queries, qh_);
        const auto elements = product(rows, g_.dimension);
        const auto tokens = frame.positions.size();
        const auto kv_elements = product(product(tokens, g_.heads), 2 * g_.dimension);
        if (!supports_sparse() || open_ || frame.local != sparse_local_ || !frame.queries || frame.queries > queries_ ||
            frame.first_query > g_.capacity_tokens - frame.queries || !tokens || tokens > tile_ ||
            frame.query.size() != (direct ? 0 : elements) || frame.kv.size() != kv_elements ||
            !std::isfinite(frame.scale) || frame.scale <= 0 ||
            !frame.rope.dimensions || frame.rope.dimensions > g_.dimension || frame.rope.dimensions % 2 ||
            !std::isfinite(frame.rope.base) || frame.rope.base <= 0) throw std::invalid_argument("sparse CUDA prefill frame");
        if (direct) {
            validate_device_query(frame.device_query, DeviceMemory::cuda, qh_, g_.dimension, frame.queries);
            validate_device_output(frame.device_output, DeviceMemory::cuda, qh_, g_.dimension, frame.queries);
            validate_device_kv(frame.device_kv, DeviceMemory::cuda, g_.heads, g_.dimension, frame.queries);
            validate_pointer(frame.device_query.data);
            validate_pointer(frame.device_output.data);
            validate_pointer(frame.device_kv.data);
        } else if (frame.device_output.data || frame.device_kv.data) throw std::invalid_argument("incomplete sparse prefill device frame");
        const auto end = frame.first_query + frame.queries;
        const auto origin = frame.first_query + 1 - std::min<std::uint64_t>(frame.first_query + 1, frame.local);
        if (frame.positions.back() != end - 1 || !std::is_sorted(frame.positions.begin(), frame.positions.end()) ||
            std::adjacent_find(frame.positions.begin(), frame.positions.end()) != frame.positions.end() ||
            frame.positions.end() - std::lower_bound(frame.positions.begin(), frame.positions.end(), origin) !=
                static_cast<std::ptrdiff_t>(end - origin)) {
            throw std::invalid_argument("sparse CUDA prefill positions must include all local tokens");
        }
        const auto valid = static_cast<std::uint32_t>(tokens);
        const auto local_begin = static_cast<std::uint32_t>(tokens - (end - origin));
        const auto local_only_from = static_cast<std::uint32_t>(std::min(tokens, frame.local_only_from));
        const auto global_end = frame.positions.begin() + local_only_from;
        const auto global_keys = end > frame.local ? static_cast<std::uint32_t>(
            std::upper_bound(frame.positions.begin(), global_end, end - 1 - frame.local) - frame.positions.begin()) : 0;
        const auto score_elements = product(qh_, end - origin);
        SparsePrefillOutput result{std::vector<float>(direct ? 0 : elements),
            std::vector<float>(frame.keep_scores_on_device ? 0 : score_elements)};
        const auto q_bytes = result.attention.size() * sizeof(float);
        const auto feedback_bytes = score_elements * sizeof(float);
        if (frame.keep_scores_on_device && !direct) throw std::invalid_argument("device scores require device prefill inputs");
        const auto row_bytes = 2 * g_.dimension * sizeof(Half);
        const auto history = direct ? valid - frame.queries : valid;
        const auto kv_bytes = static_cast<std::size_t>(history) * g_.heads * row_bytes;
        const auto * queries = direct ? frame.device_query.data : query_;
        auto * output = direct ? frame.device_output.data : weights_;
        DeviceScope scope(device_);
        try {
            if (!direct) checked(cudaMemcpyAsync(query_, frame.query.data(), q_bytes, cudaMemcpyHostToDevice, stream_), "upload sparse prefill queries");
            if (history) checked(cudaMemcpy2DAsync(kv_, valid * row_bytes, frame.kv.data(), valid * row_bytes,
                history * row_bytes, g_.heads, cudaMemcpyHostToDevice, stream_), "upload sparse prefill historical KV");
            checked(cudaMemcpyAsync(key_positions_, frame.positions.data(), tokens * sizeof(std::uint64_t),
                cudaMemcpyHostToDevice, stream_), "upload sparse prefill positions");
            checked(cudaMemsetAsync(error_device_, 0, sizeof(int), stream_), "clear sparse prefill status");
            pack_gemm_kv(kv_, valid, 2 * g_.dimension, 2 * tokens * g_.dimension, frame.device_kv.data, history);
            if (global_keys) {
                pack_query<<<grid(elements), 256, 0, stream_>>>(queries, packed_query_, qh_, frame.queries, g_.dimension);
                checked(cudaGetLastError(), "pack global sparse prefill queries");
                multiply_scores(valid, frame.queries, frame.scale, scores_, 0, global_keys);
            }
            rotary_frequencies<<<grid(frame.rope.dimensions / 2), 256, 0, stream_>>>(frequencies_, frame.rope.dimensions, frame.rope.base);
            checked(cudaGetLastError(), "prepare sparse rotary frequencies");
            pack_local_queries<<<grid(elements), 256, 0, stream_>>>(queries, packed_query_, frequencies_,
                frame.first_query, origin, frame.local, qh_, frame.queries, g_.dimension, frame.rope.dimensions, frame.rope.split_halves);
            checked(cudaGetLastError(), "rotate local sparse prefill queries");
            rotate_local_keys<<<grid(tokens * g_.heads * (frame.rope.dimensions / 2)), 256, 0, stream_>>>(packed_kv_, key_positions_,
                frequencies_, origin, valid, g_.heads, g_.dimension, frame.rope.dimensions, frame.rope.split_halves);
            checked(cudaGetLastError(), "rotate local sparse prefill keys");
            multiply_scores(valid, frame.queries, frame.scale, local_scores_, local_begin, valid - local_begin);
            sparse_softmax<<<static_cast<unsigned>(rows), 256, 0, stream_>>>(scores_, local_scores_, probabilities_,
                key_positions_, error_device_, frame.first_query, frame.queries, frame.local, valid, local_only_from);
            checked(cudaGetLastError(), "apply joint sparse prefill softmax");
            sum_sparse_local_scores<<<grid(score_elements), 256, 0, stream_>>>(probabilities_, local_feedback_,
                frame.first_query, origin, frame.queries, qh_, frame.local, valid, local_begin);
            checked(cudaGetLastError(), "sum recent attention scores by key");
            multiply_values(valid, frame.queries);
            scatter_sparse_output<<<grid(elements), 256, 0, stream_>>>(tile_output_, output, error_device_,
                g_.dimension, qh_, frame.queries);
            checked(cudaGetLastError(), "reorder sparse prefill output");
            if (!direct) checked(cudaMemcpyAsync(result.attention.data(), weights_, q_bytes, cudaMemcpyDeviceToHost, stream_),
                "download sparse prefill output");
            if (!frame.keep_scores_on_device) checked(cudaMemcpyAsync(result.key_scores.data(), local_feedback_, feedback_bytes,
                cudaMemcpyDeviceToHost, stream_), "download summed recent prefill scores");
            checked(cudaMemcpyAsync(&error_, error_device_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "download sparse prefill status");
            checked(cudaStreamSynchronize(stream_), "complete sparse CUDA prefill");
            if (error_) throw std::runtime_error("non-finite or empty sparse CUDA prefill");
        } catch (...) { fail(); throw; }
        ++stats_.batches;
        ++stats_.tiles;
        stats_.query_tokens += frame.queries;
        stats_.kv_tokens += tokens;
        stats_.query_upload_bytes += q_bytes;
        stats_.upload_bytes += kv_bytes;
        stats_.output_download_bytes += q_bytes;
        if (frame.keep_scores_on_device) {
            result.device_scores = {local_feedback_, score_elements, DeviceMemory::cuda};
        } else stats_.probability_download_bytes += feedback_bytes;
        if (direct) { stats_.device_query_tokens += frame.queries; stats_.device_kv_tokens += frame.queries; }
        return result;
    }
    std::vector<float> finish() override {
        std::vector<float> output(product(product(count_,qh_),g_.dimension));
        finish_impl(output.data(),output.size(),false);
        return output;
    }
    void finish_to(DeviceOutput output) override {
        validate_device_output(output,DeviceMemory::cuda,qh_,g_.dimension,count_); validate_pointer(output.data);
        finish_impl(output.data,output.elements,true);
    }
    void finish_impl(float * output,std::size_t elements,bool device) {
        check();
        if (!open_ || next_key_!=first_+count_) throw std::logic_error("prefill finish requires the complete causal prefix");
        DeviceScope scope(device_);
        try {
            normalize<<<grid(elements),256,0,stream_>>>(weights_,sum_,error_device_,elements,g_.dimension);
            checked(cudaGetLastError(),"normalize tiled prefill");
            const auto bytes=product(elements,sizeof(float));
            checked(cudaMemcpyAsync(output,weights_,bytes,device?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost,stream_),"copy prefill output");
            checked(cudaMemcpyAsync(&error_,error_device_,sizeof(int),cudaMemcpyDeviceToHost,stream_),"download prefill status");
            checked(cudaStreamSynchronize(stream_),"complete tiled prefill frame");
            if (error_) throw std::runtime_error("non-finite or empty tiled attention");
            open_=false; ++stats_.batches; stats_.query_tokens+=count_;
            current_query_=nullptr;
            if (device) stats_.output_device_copy_bytes+=bytes; else stats_.output_download_bytes+=bytes;
        } catch (...) { fail(); throw; }
    }
    void abort() override {
        DeviceScope scope(device_);
        const auto status=cudaStreamSynchronize(stream_);
        open_=false; failed_=true;
        current_query_=nullptr;
        checked(status,"abort tiled prefill");
    }
private:
    void validate_pointer(const void * pointer) const {
        check(); DeviceScope scope(device_); cudaPointerAttributes attributes{};
        checked(cudaPointerGetAttributes(&attributes,pointer),"inspect borrowed prefill input");
        if ((attributes.type!=cudaMemoryTypeDevice && attributes.type!=cudaMemoryTypeManaged) || attributes.device!=device_) {
            throw std::invalid_argument("prefill input belongs to another memory space or device");
        }
    }
    static unsigned grid(std::uint64_t elements) { return static_cast<unsigned>(std::min<std::uint64_t>((elements+255)/256,65535)); }
    template<class T> void allocate(T *& target,std::size_t bytes,std::uint64_t & category) {
        checked(cudaMalloc(reinterpret_cast<void **>(&target),bytes),"allocate tiled prefill"); category+=bytes;
    }
    void check() const { if (failed_) throw std::runtime_error("tiled prefill backend failed"); }
    void fail() noexcept { failed_=true; open_=false; if (stream_) (void) cudaStreamSynchronize(stream_); current_query_=nullptr; }
    void release() noexcept {
        int previous=0; const bool restore=cudaGetDevice(&previous)==cudaSuccess;
        if (cudaSetDevice(device_)!=cudaSuccess) return;
        if (stream_) (void) cudaStreamSynchronize(stream_);
        if (blas_) (void) cublasDestroy(blas_);
        if (query_) (void) cudaFree(query_);
        if (kv_) (void) cudaFree(kv_);
        if (weights_) (void) cudaFree(weights_);
        if (maximum_) (void) cudaFree(maximum_);
        if (sum_) (void) cudaFree(sum_);
        if (error_device_) (void) cudaFree(error_device_);
        if (packed_query_) (void) cudaFree(packed_query_);
        if (packed_kv_) (void) cudaFree(packed_kv_);
        if (scores_) (void) cudaFree(scores_);
        if (probabilities_) (void) cudaFree(probabilities_);
        if (tile_output_) (void) cudaFree(tile_output_);
        if (previous_scale_) (void) cudaFree(previous_scale_);
        if (local_scores_) (void) cudaFree(local_scores_);
        if (local_feedback_) (void) cudaFree(local_feedback_);
        if (key_positions_) (void) cudaFree(key_positions_);
        if (frequencies_) (void) cudaFree(frequencies_);
        if (copied_) (void) cudaEventDestroy(copied_);
        if (stream_) (void) cudaStreamDestroy(stream_);
        if (restore && previous!=device_) (void) cudaSetDevice(previous);
    }
    KvGeometry g_;
    std::uint32_t qh_,queries_,sparse_local_,tile_=0,count_=0;
    int device_,error_=0;
    std::uint64_t first_=0,next_key_=0;
    std::size_t tile_elements_=0;
    float scale_=1;
    bool open_=false,failed_=false;
    cudaStream_t stream_=nullptr;
    cudaEvent_t copied_=nullptr;
    cublasHandle_t blas_ = nullptr;
    __half * packed_query_ = nullptr, * packed_kv_ = nullptr, * probabilities_ = nullptr;
    float * scores_ = nullptr, * tile_output_ = nullptr, * previous_scale_ = nullptr;
    float * local_scores_ = nullptr, * local_feedback_ = nullptr, * frequencies_ = nullptr;
    std::uint64_t * key_positions_ = nullptr;
    float * query_=nullptr,* weights_=nullptr,* maximum_=nullptr,* sum_=nullptr;
    const float * current_query_=nullptr;
    Half * kv_=nullptr;
    int * error_device_=nullptr;
    PrefillStats stats_;
};
} // namespace
std::unique_ptr<TiledPrefill> make_cuda_tiled_prefill(const KvGeometry & geometry,std::uint32_t query_heads,
                                                    std::uint32_t queries,std::uint32_t tile_tokens,int device,
                                                    std::uint32_t sparse_local) {
    return std::make_unique<CudaTiledPrefill>(geometry,query_heads,queries,tile_tokens,device,sparse_local);
}
} // namespace solid
