#include "solid/tiled_prefill.hpp"
#include "solid/online_attention.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
template<int Width>
__global__ void accumulate(const float * query,const __half * kv,float * weights,float * maximum,float * sum,int * error,
    std::uint64_t first_query,std::uint64_t first_key,std::uint32_t valid,std::uint64_t rows,
    std::uint32_t query_heads,std::uint32_t group,std::uint32_t width,std::size_t token_stride,std::size_t head_stride,float scale) {
    constexpr int fragments=(Width+31)/32;
    const auto lane=threadIdx.x%32;
    const auto warp=(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto warps=static_cast<std::uint64_t>(gridDim.x)*blockDim.x/32;
    for (auto row=warp;row<rows;row+=warps) {
        const auto position=first_query+row/query_heads;
        if (first_key>position) continue;
        const auto head=(row%query_heads)/group;
        float q[fragments], output[fragments];
        for (int r=0;r<fragments;++r) {
            const auto d=lane+32*r;
            q[r]=d<width?query[row*width+d]:0;
            output[r]=d<width?weights[row*width+d]:0;
        }
        auto m=maximum[row], denominator=sum[row];
        const auto count=static_cast<std::uint32_t>(min(static_cast<std::uint64_t>(valid),position-first_key+1));
        for (std::uint32_t token=0;token<count;++token) {
            const auto base=head*head_stride+token*token_stride;
            float score=0;
            for (int r=0;r<fragments;++r) {
                const auto d=lane+32*r;
                if (d<width) score=fmaf(q[r],__half2float(kv[base+d]),score);
            }
            for (int delta=16;delta;delta/=2) score+=__shfl_down_sync(0xffffffff,score,delta);
            score=__shfl_sync(0xffffffff,score,0)*scale;
            if (!isfinite(score)) { if (!lane) atomicExch(error,1); return; }
            const auto update=online_attention_update(m,denominator,score);
            m=update.maximum; denominator=update.sum;
            for (int r=0;r<fragments;++r) {
                const auto d=lane+32*r;
                if (d<width) output[r]=output[r]*update.previous_scale+__half2float(kv[base+width+d])*update.weight;
            }
        }
        if (!lane) { maximum[row]=m; sum[row]=denominator; }
        for (int r=0;r<fragments;++r) {
            const auto d=lane+32*r;
            if (d<width) weights[row*width+d]=output[r];
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
class CudaTiledPrefill final:public TiledPrefill {
public:
    CudaTiledPrefill(KvGeometry geometry,std::uint32_t query_heads,std::uint32_t queries,std::uint32_t tile_tokens,int device):
        g_(geometry),qh_(query_heads),queries_(queries),device_(device) {
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
            checked(cudaEventCreateWithFlags(&copied_,cudaEventDisableTiming),"create prefill upload event");
            allocate(kv_,product(tile_elements_,sizeof(Half)),stats_.kv_device_bytes);
            allocate(query_,product(product(rows,g_.dimension),sizeof(float)),stats_.workspace_device_bytes);
            allocate(weights_,product(product(rows,g_.dimension),sizeof(float)),stats_.workspace_device_bytes);
            allocate(maximum_,product(rows,sizeof(float)),stats_.workspace_device_bytes);
            allocate(sum_,product(rows,sizeof(float)),stats_.workspace_device_bytes);
            allocate(error_device_,sizeof(int),stats_.workspace_device_bytes);
            stats_.kv_device_peak_bytes=stats_.kv_device_bytes;
            stats_.workspace_device_peak_bytes=stats_.workspace_device_bytes;
        } catch (...) { release(); throw; }
    }
    ~CudaTiledPrefill() override { release(); }
    std::uint32_t query_capacity() const override { return queries_; }
    std::uint32_t tile_capacity() const override { return tile_; }
    const char * name() const override { return "cuda_online_tiled"; }
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
            const auto rows=static_cast<std::uint64_t>(count_)*qh_;
            const auto blocks=static_cast<unsigned>(std::min<std::uint64_t>((rows+3)/4,65535));
            const auto token_stride=product(2,g_.dimension)*(device?g_.heads:1);
            const auto head_stride=product(2,g_.dimension)*(device?1:tile_);
#define SOLID_PREFILL_LAUNCH(W) accumulate<W><<<blocks,128,0,stream_>>>(current_query_,reinterpret_cast<const __half *>(device?kv:kv_),weights_,maximum_,sum_,error_device_,first_,first,valid,rows,qh_,qh_/g_.heads,g_.dimension,token_stride,head_stride,scale_)
            if (g_.dimension<=32) { SOLID_PREFILL_LAUNCH(32); }
            else if (g_.dimension<=64) { SOLID_PREFILL_LAUNCH(64); }
            else if (g_.dimension<=128) { SOLID_PREFILL_LAUNCH(128); }
            else if (g_.dimension<=256) { SOLID_PREFILL_LAUNCH(256); }
            else if (g_.dimension<=512) { SOLID_PREFILL_LAUNCH(512); }
            else { SOLID_PREFILL_LAUNCH(1024); }
#undef SOLID_PREFILL_LAUNCH
            checked(cudaGetLastError(),"launch causal tiled prefill");
            // The source lease ends after H2D. The kernel may overlap the next
            // host read; the same stream orders the next overwrite of kv_.
            if (!device) { checked(cudaEventSynchronize(copied_),"complete prefill tile upload"); stats_.upload_bytes+=bytes; }
            else stats_.device_kv_tokens+=valid;
            next_key_+=valid; ++stats_.tiles; stats_.kv_tokens+=valid;
        } catch (...) { fail(); throw; }
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
        if (query_) (void) cudaFree(query_);
        if (kv_) (void) cudaFree(kv_);
        if (weights_) (void) cudaFree(weights_);
        if (maximum_) (void) cudaFree(maximum_);
        if (sum_) (void) cudaFree(sum_);
        if (error_device_) (void) cudaFree(error_device_);
        if (copied_) (void) cudaEventDestroy(copied_);
        if (stream_) (void) cudaStreamDestroy(stream_);
        if (restore && previous!=device_) (void) cudaSetDevice(previous);
    }
    KvGeometry g_;
    std::uint32_t qh_,queries_,tile_=0,count_=0;
    int device_,error_=0;
    std::uint64_t first_=0,next_key_=0;
    std::size_t tile_elements_=0;
    float scale_=1;
    bool open_=false,failed_=false;
    cudaStream_t stream_=nullptr;
    cudaEvent_t copied_=nullptr;
    float * query_=nullptr,* weights_=nullptr,* maximum_=nullptr,* sum_=nullptr;
    const float * current_query_=nullptr;
    Half * kv_=nullptr;
    int * error_device_=nullptr;
    PrefillStats stats_;
};
} // namespace
std::unique_ptr<TiledPrefill> make_cuda_tiled_prefill(const KvGeometry & geometry,std::uint32_t query_heads,
                                                    std::uint32_t queries,std::uint32_t tile_tokens,int device) {
    return std::make_unique<CudaTiledPrefill>(geometry,query_heads,queries,tile_tokens,device);
}
} // namespace solid
