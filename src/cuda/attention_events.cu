// CUDA attention 完成事件与投影计时；通过通知描述符唤醒 CPU 协调线程。
#include "solid/attention_events.hpp"
#include "solid/cuda_completion.cuh"
#include "solid/device_timing.hpp"
#include "ggml-backend-impl.h"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace solid {
namespace {
void check(cudaError_t status,const char * operation) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct Device {
    int original;
    explicit Device(int device) { check(cudaGetDevice(&original),"get attention event device"); check(cudaSetDevice(device),"set attention event device"); }
    ~Device() { (void) cudaSetDevice(original); }
};
class CudaDeviceInterval final : public DeviceInterval {
public:
    explicit CudaDeviceInterval(int device) : device_(device) {
        Device scope(device_);
        check(cudaEventCreate(&begin_),"create CUDA projection begin event");
        try { check(cudaEventCreate(&end_),"create CUDA projection end event"); }
        catch (...) { (void) cudaEventDestroy(begin_); throw; }
    }
    ~CudaDeviceInterval() override {
        int previous=-1;
        (void) cudaGetDevice(&previous);
        if (cudaSetDevice(device_)==cudaSuccess) {
            (void) cudaEventDestroy(begin_);
            (void) cudaEventDestroy(end_);
        }
        if (previous>=0) (void) cudaSetDevice(previous);
    }
    void record(void * stream,bool end) override {
        check(cuda_record_completion_event(end?end_:begin_,static_cast<cudaStream_t>(stream)),"record CUDA projection event");
        recorded_[end]=true;
    }
    std::optional<std::uint64_t> duration() override {
        if (!recorded_[0] || !recorded_[1]) return {};
        Device scope(device_);
        float ms=0;
        check(cudaEventElapsedTime(&ms,begin_,end_),"measure CUDA projection interval");
        const auto ns=static_cast<double>(ms)*1e6;
        if (!std::isfinite(ns) || ns<0 || ns>=static_cast<double>(UINT64_MAX)) throw std::runtime_error("invalid CUDA projection interval");
        return static_cast<std::uint64_t>(ns);
    }
private:
    int device_;
    cudaEvent_t begin_=nullptr,end_=nullptr;
    bool recorded_[2]{};
};
class CudaAttentionEvents final:public AttentionEvents {
public:
    CudaAttentionEvents(std::uint32_t layers,int device):AttentionEvents(layers),device_(device),layers_(layers) {
        Device scope(device_);
        try {
            for (std::uint32_t i=0;i<layers_;++i) {
                auto * target=token(i);
                cudaEvent_t begin=nullptr,end=nullptr;
                check(cudaEventCreate(&begin),"create attention begin event"); target->begin_event=begin;
                check(cudaEventCreate(&end),"create attention end event"); target->end_event=end;
            }
        } catch (...) { destroy(); throw; }
    }
    ~CudaAttentionEvents() override {
        int previous=-1; (void) cudaGetDevice(&previous);
        if (cudaSetDevice(device_)==cudaSuccess) destroy();
        if (previous>=0) (void) cudaSetDevice(previous);
    }
    bool copy_ready(ggml_backend_event * event) const override {
        Device scope(device_);
        const auto status=cudaEventQuery(static_cast<cudaEvent_t>(event->context));
        if (status==cudaErrorNotReady) return false;
        check(status,"query KV upload event"); return true;
    }
    const char * name() const override { return "cuda_stream_events"; }
protected:
    std::uint64_t elapsed_ns(std::uint32_t layer) const override {
        Device scope(device_); const auto * target=const_cast<CudaAttentionEvents *>(this)->token(layer);
        const auto end=static_cast<cudaEvent_t>(target->end_event);
        check(cudaEventQuery(end),"confirm attention end event");
        float ms=0; check(cudaEventElapsedTime(&ms,static_cast<cudaEvent_t>(target->begin_event),end),"measure attention event interval");
        const auto ns=static_cast<double>(ms)*1e6;
        if (!std::isfinite(ns) || ns<0 || ns>=static_cast<double>(UINT64_MAX)) throw std::runtime_error("invalid attention device interval");
        return static_cast<std::uint64_t>(ns);
    }
private:
    void destroy() noexcept {
        for (std::uint32_t i=0;i<layers_;++i) {
            auto * target=token(i);
            if (target->begin_event) (void) cudaEventDestroy(static_cast<cudaEvent_t>(target->begin_event));
            if (target->end_event) (void) cudaEventDestroy(static_cast<cudaEvent_t>(target->end_event));
            target->begin_event=target->end_event=nullptr;
        }
    }
    int device_;
    std::uint32_t layers_;
};
}
std::unique_ptr<AttentionEvents> make_cuda_attention_events(std::uint32_t layers,int device) { return std::make_unique<CudaAttentionEvents>(layers,device); }
std::unique_ptr<DeviceInterval> make_cuda_device_interval(int device) { return std::make_unique<CudaDeviceInterval>(device); }
} // namespace solid
