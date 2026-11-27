// CUDA attention 完成事件；通过通知描述符唤醒 CPU 协调线程。
#include "solid/attention_events.hpp"
#include "solid/cuda_completion.cuh"
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
class CudaAttentionEvents final:public AttentionEvents {
public:
    CudaAttentionEvents(std::uint32_t layers,int device):AttentionEvents(layers),device_(device),layers_(layers) {
        Device scope(device_);
        try {
            for (std::uint32_t i=0;i<layers_;++i) {
                auto * target=token(i);
                cudaEvent_t end=nullptr;
                check(cudaEventCreateWithFlags(&end,cudaEventDisableTiming),"create attention end event"); target->end_event=end;
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
} // namespace solid
