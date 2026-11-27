// 按 KV head 合并 K/V 投影权重，并为交织投影结果建立 K、V 视图。
#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer_type;

namespace solid {
struct ProjectionStats {
    std::uint64_t buffer_bytes = 0;
    std::uint64_t source_bytes = 0;
    std::uint64_t source_host_bytes = 0;
    std::string weight_type;
};

// Rows are [head, K/V, dimension]; quantized rows are copied whole.
class KvProjection {
public:
    KvProjection(const ggml_tensor * key, const ggml_tensor * value,
                 const ggml_tensor * key_bias, const ggml_tensor * value_bias,
                 std::uint32_t heads, std::uint32_t dimension, ggml_backend_buffer_type * buffer_type);
    ~KvProjection();
    KvProjection(const KvProjection &) = delete;
    KvProjection & operator=(const KvProjection &) = delete;
    // 用融合权重执行一次矩阵乘法，输出按 head 交织的 K/V。
    ggml_tensor * project(ggml_context * context, ggml_tensor * input) const;
    const ProjectionStats & stats() const;
    const ggml_tensor * weight() const;
    // 建立指向交织结果的 K 视图，读取时使用对应 stride。
    static ggml_tensor * key_view(ggml_context * context, ggml_tensor * packed);
    // 建立指向交织结果的 V 视图，跳过同 head 的 K 行。
    static ggml_tensor * value_view(ggml_context * context, ggml_tensor * packed);
    // Write post-RoPE K back to its original rows; V keeps its projected rows.
    static ggml_tensor * update_key(ggml_context * context, ggml_tensor * packed, ggml_tensor * key);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace solid
