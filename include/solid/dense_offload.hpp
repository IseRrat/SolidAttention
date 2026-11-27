// 纯 Offload 基线的整层 KV 工作区、预取与写回接口。
#pragma once

#include "solid/device_cache.hpp"
#include <memory>

struct ggml_backend;
struct ggml_context;
struct ggml_tensor;

namespace solid {
class AttentionSession;

class DenseOffload {
public:
    DenseOffload(AttentionSession & session, ggml_backend * backend);
    ~DenseOffload();
    // 开始本步第 0 层历史 KV 的后台读取。
    void begin(std::uint64_t position);
    // 等待该层读盘完成，将历史 KV 上传到 GPU 并确认上传完成。
    void upload_history(std::uint32_t layer, std::uint64_t position);
    // 把本步新 K/V 放入该层写缓冲，供本次及后续读取使用。
    void archive(std::uint32_t layer, std::uint64_t position, const float * keys, const float * values);
    // 本层 attention 完成后，收齐已提交写回，再启动下一层历史读取。
    void attention_complete();
    void join();
    // 检查本轮 Decode 的所有层均已完成，状态已回到空闲。
    void finish_decode();
    ggml_tensor * plane(ggml_context * context, bool values) const;
    DeviceCacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace solid
