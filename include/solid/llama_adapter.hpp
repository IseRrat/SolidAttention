#pragma once

#include "solid/config.hpp"
#include "solid/session.hpp"
#include "solid/device_cache.hpp"
#include "solid/scheduler.hpp"
#include <exception>
#include <memory>
#include <mutex>

struct llama_model;
struct llm_graph_context;
class llm_graph_input_attn_kv;
struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_sched;
struct llama_layer;
struct llm_graph_qkv;

namespace solid {
struct AttentionFrame { std::vector<std::uint64_t> positions; };

class AttentionFailure {
public:
    void capture(std::exception_ptr error, std::uint32_t layer = 0, std::uint64_t position = 0);
    void check() const;
private:
    mutable std::mutex mutex_;
    std::exception_ptr error_;
    std::uint32_t layer_ = 0;
    std::uint64_t position_ = 0;
};

class AttentionNode {
public:
    AttentionNode(AttentionSession & session, std::shared_ptr<AttentionFrame> frame, std::uint32_t layer,
                  float scale, std::shared_ptr<AttentionFailure> failure = {}, bool gathered = false, class GgmlPrefetch * prefetch = nullptr);
    ~AttentionNode();
    ggml_tensor * build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                       ggml_tensor * packed_kv = nullptr, ggml_backend_sched * scheduler = nullptr,
                       ggml_backend * decode_backend = nullptr);
    void check() const;
    ggml_tensor * host_node() const;
    ggml_tensor * selection_node() const;
    ggml_tensor * attention_node() const;
    void set_query_proof(bool covers_previous_attention);
    void defer_completion();
    void complete_attention();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AttentionAdapter {
public:
    AttentionAdapter(const llama_model * model, const AttentionConfig & options, std::uint32_t capacity,
                     std::uint64_t prefill_tokens, const std::filesystem::path & trace);
    ~AttentionAdapter();
    AttentionAdapter(const AttentionAdapter &) = delete;
    AttentionAdapter & operator=(const AttentionAdapter &) = delete;
    void check() const;
    void after_process();
    void flush();
    void finish();
    SessionStats stats() const;
    SparseBudget budget() const;
    std::unique_ptr<AttentionNode> node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale,
                                      bool gathered = false, ggml_backend * backend = nullptr);
    bool gathered_decode(std::uint64_t position) const;
    bool fused_projection() const;
    const class KvProjection & projection(std::uint32_t layer) const;
    std::uint64_t projection_buffer_bytes() const;
    std::uint64_t projection_initialization_ns() const;
    std::vector<std::string> projection_weight_types() const;
    bool prefetch_enabled() const;
    void begin_prefetch(std::uint32_t layer, std::uint64_t position);
    bool observe(ggml_tensor * tensor, bool ask) noexcept;
    void quiesce();
    DeviceCacheStats device_stats() const;
    ScheduleStats schedule_stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool adapter_active(const llama_model * model);
bool graph_depends_on(const ggml_tensor * tensor, const ggml_tensor * ancestor);
bool adapter_fused_projection(const llama_model * model);
llm_graph_qkv adapter_qkv(const llm_graph_context & graph, const llama_layer & layer, ggml_tensor * input,
                         std::int64_t dimension_q, std::int64_t heads_q, std::int64_t dimension_k,
                         std::int64_t heads_k, std::int64_t dimension_v, std::int64_t heads_v, int layer_id, bool reshape);
llm_graph_input_attn_kv * adapter_input(const llm_graph_context & graph);
ggml_tensor * adapter_attention(const llm_graph_context & graph, llm_graph_input_attn_kv * input,
                                ggml_tensor * query, ggml_tensor * key, ggml_tensor * value, float scale, int layer);
} // namespace solid
