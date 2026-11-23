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
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_sched;
struct llama_layer;
struct llm_graph_qkv;

namespace solid {
struct ProjectionIntervals;
class InfllmSelectedCache;
class DenseOffload;
std::shared_ptr<InfllmSelectedCache> make_infllm_selected_cache(AttentionSession & session, ggml_backend * backend);
class SparseBaselineLookahead {
public:
    static constexpr std::size_t queue_depth = 64;
    SparseBaselineLookahead(AttentionSession & session, InfllmSelectedCache & workspace, bool use_uring);
    void begin(std::uint32_t layer, std::uint64_t position);
    void upload(std::uint32_t layer, std::uint64_t position);
    void start_writeback(std::uint32_t layer);
    void completed(std::uint32_t layer, std::uint64_t position);
    AsyncReader * reader() const { return reader_.get(); }
    void finish_decode() const;
    void join() { worker_.join(); }
private:
    AttentionSession & session_;
    InfllmSelectedCache & workspace_;
    std::unique_ptr<AsyncReader> reader_;
    std::optional<std::pair<std::uint32_t, std::uint64_t>> frame_;
    TaskWorker worker_;
};
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
                  float scale, std::shared_ptr<AttentionFailure> failure = {}, bool gathered = false, class GgmlPrefetch * prefetch = nullptr,
                  ProjectionIntervals * timing = nullptr, InfllmSelectedCache * selected_cache = nullptr,
                  DenseOffload * dense = nullptr, SparseBaselineLookahead * sparse_lookahead = nullptr);
    ~AttentionNode();
    ggml_tensor * build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                       ggml_tensor * packed_kv = nullptr, ggml_backend_sched * scheduler = nullptr,
                       ggml_backend * decode_backend = nullptr, ggml_cgraph * graph = nullptr);
    void check() const;
    bool can_reuse(std::uint64_t next_position) const;
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
                     std::uint64_t prefill_tokens, const std::filesystem::path & trace, std::uint32_t prefill_queries=128);
    ~AttentionAdapter();
    AttentionAdapter(const AttentionAdapter &) = delete;
    AttentionAdapter & operator=(const AttentionAdapter &) = delete;
    void check() const;
    void after_process();
    void flush();
    void enable_direct_reads();
    void finish();
    SessionStats stats() const;
    SparseBudget budget() const;
    const std::optional<RelativeRope> & relative_rope() const;
    std::unique_ptr<AttentionNode> node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale,
                                      bool gathered = false, ggml_backend * backend = nullptr);
    bool gathered_decode(std::uint64_t position) const;
    bool fused_projection() const;
    const class KvProjection & projection(std::uint32_t layer) const;
    ProjectionIntervals * projection_intervals(std::uint32_t layer, ggml_backend * backend);
    std::uint64_t projection_buffer_bytes() const;
    std::uint64_t projection_source_bytes() const;
    std::uint64_t projection_source_host_bytes() const;
    std::uint64_t projection_initialization_ns() const;
    std::vector<std::string> projection_weight_types() const;
    bool prefetch_enabled() const;
    bool dense_offload_enabled() const;
    void begin_prefetch(std::uint32_t layer, std::uint64_t position);
    bool observe(ggml_tensor * tensor, bool ask) noexcept;
    void quiesce();
    DeviceCacheStats device_stats() const;
    ScheduleStats schedule_stats() const;
    std::vector<int> inference_cpus() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool adapter_active(const llama_model * model);
bool graph_depends_on(const ggml_tensor * tensor, const ggml_tensor * ancestor);
bool adapter_fused_projection(const llama_model * model);
void adapter_mark_projection(const llm_graph_context & graph, ggml_tensor * projection, int layer, bool query);
llm_graph_qkv adapter_qkv(const llm_graph_context & graph, const llama_layer & layer, ggml_tensor * input,
                         std::int64_t dimension_q, std::int64_t heads_q, std::int64_t dimension_k,
                         std::int64_t heads_k, std::int64_t dimension_v, std::int64_t heads_v, int layer_id, bool reshape);
llm_graph_input_attn_kv * adapter_input(const llm_graph_context & graph);
ggml_tensor * adapter_attention(const llm_graph_context & graph, llm_graph_input_attn_kv * input,
                                ggml_tensor * query, ggml_tensor * key, ggml_tensor * value, float scale, int layer);
} // namespace solid
