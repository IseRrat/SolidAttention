#include "solid/trace.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef SOLID_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
#ifdef SOLID_HAVE_NVTX_CUDA
#include <nvtx3/nvToolsExtCudaRt.h>
#endif
#ifdef __linux__
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace solid::trace {
bool enabled() noexcept {
#ifdef SOLID_HAVE_NVTX
    static const bool value = [] {
        const auto * flag = std::getenv("SOLID_NVTX_TRACE");
        return flag && std::strcmp(flag, "1") == 0;
    }();
    return value;
#else
    return false;
#endif
}
#ifdef SOLID_HAVE_NVTX
namespace {
nvtxDomainHandle_t domain() noexcept {
    static auto value = nvtxDomainCreateA("SolidAttention");
    return value;
}
nvtxEventAttributes_t attributes(const char * stage, std::uint32_t layer, std::uint64_t position,
                               const char * detail, char (&label)[256]) noexcept {
    char location[80] = {};
    if (layer != UINT32_MAX && position != UINT64_MAX)
        std::snprintf(location, sizeof(location), " L%u token=%llu", layer, static_cast<unsigned long long>(position));
    else if (layer != UINT32_MAX) std::snprintf(location, sizeof(location), " L%u", layer);
    else if (position != UINT64_MAX) std::snprintf(location, sizeof(location), " token=%llu", static_cast<unsigned long long>(position));
    std::snprintf(label, sizeof(label), "%s%s%s%s", stage, location, detail ? " / " : "", detail ? detail : "");
    nvtxEventAttributes_t result{};
    result.version = NVTX_VERSION; result.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    result.messageType = NVTX_MESSAGE_TYPE_ASCII; result.message.ascii = label;
    result.colorType = NVTX_COLOR_ARGB;
    result.color = std::strstr(stage, "SSD") ? 0xff806493u :
        std::strstr(stage, "CPU->GPU") ? 0xff657c90u :
        std::strstr(stage, "GPU") ? 0xff708b65u : 0xffaa7d45u;
    return result;
}
}
#endif
void name_thread(const char * name) noexcept {
#ifdef __linux__
    char short_name[16]; std::snprintf(short_name, sizeof(short_name), "%.15s", name);
    (void) pthread_setname_np(pthread_self(), short_name);
#ifdef SOLID_HAVE_NVTX
    if (enabled()) nvtxNameOsThreadA(static_cast<std::uint32_t>(syscall(SYS_gettid)), name);
#endif
#else
    (void) name;
#endif
}
void name_cuda_stream(void * stream, const char * name) noexcept {
#ifdef SOLID_HAVE_NVTX_CUDA
    if (enabled()) nvtxNameCudaStreamA(static_cast<cudaStream_t>(stream), name);
#else
    (void) stream; (void) name;
#endif
}
Range::Range(const char * stage, std::uint32_t layer, std::uint64_t position, const char * detail) noexcept {
#ifdef SOLID_HAVE_NVTX
    if (!enabled()) return;
    char label[256];
    const auto event = attributes(stage, layer, position, detail, label);
    id_ = nvtxDomainRangeStartEx(domain(), &event);
#else
    (void) stage; (void) layer; (void) position; (void) detail;
#endif
}
void Range::finish() noexcept {
#ifdef SOLID_HAVE_NVTX
    if (id_) nvtxDomainRangeEnd(domain(), id_);
#endif
    id_ = 0;
}
void Scope::start(const char * stage, std::uint32_t layer, std::uint64_t position, const char * detail) noexcept {
    finish();
#ifdef SOLID_HAVE_NVTX
    if (!enabled()) return;
    char label[256];
    const auto event = attributes(stage, layer, position, detail, label);
    nvtxDomainRangePushEx(domain(), &event);
    active_ = true;
#else
    (void) stage; (void) layer; (void) position; (void) detail;
#endif
}
void Scope::finish() noexcept {
#ifdef SOLID_HAVE_NVTX
    if (active_) nvtxDomainRangePop(domain());
#endif
    active_ = false;
}
const char * operation_stage(const char * op, const char * input, const char * output) noexcept {
    if (std::strstr(output, "infllm.QK.")) return "GPU/attention QK enqueue";
    if (std::strstr(output, "infllm.PV.")) return "GPU/attention PV enqueue";
    if (!std::strcmp(op, "MUL_MAT")) {
        if (std::strstr(input, "attn_q.weight")) return "GPU/Q projection enqueue";
        if (std::strstr(input, "solid_fused_kv_weight")) return "GPU/KV projection enqueue";
        if (std::strstr(input, "attn_k.weight")) return "GPU/K projection enqueue";
        if (std::strstr(input, "attn_v.weight")) return "GPU/V projection enqueue";
        if (std::strstr(input, "attn_output.weight")) return "GPU/attention output projection enqueue";
        if (std::strstr(input, "ffn_")) return "GPU/FFN projection enqueue";
    }
    if (!std::strcmp(op, "FLASH_ATTN_EXT")) return "GPU/attention enqueue";
    if (!std::strcmp(op, "SOFT_MAX")) return "GPU/attention softmax enqueue";
    if (!std::strcmp(op, "ROPE")) return "GPU/RoPE enqueue";
    if (!std::strcmp(op, "GET_ROWS")) return "GPU/gather rows enqueue";
    if (!std::strcmp(op, "SET_ROWS")) return "GPU/store rows enqueue";
    return "GPU/operator enqueue";
}
} // namespace solid::trace

extern "C" void solid_trace_name_cpu_worker(int index) noexcept {
    char label[32];
    std::snprintf(label, sizeof(label), "ggml.cpu.%d", index);
    solid::trace::name_thread(label);
}
