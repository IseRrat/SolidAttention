#pragma once

#include "solid/config.hpp"
#include "solid/session.hpp"
#include <exception>
#include <memory>
#include <mutex>

struct llama_model;
struct llm_graph_context;
class llm_graph_input_attn_kv;
struct ggml_context;
struct ggml_tensor;

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
                  float scale, std::shared_ptr<AttentionFailure> failure = {}, bool gathered = false);
    ~AttentionNode();
    ggml_tensor * build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value);
    void check() const;
    ggml_tensor * host_node() const;
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
    void flush();
    SessionStats stats() const;
    SparseBudget budget() const;
    std::unique_ptr<AttentionNode> node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale, bool gathered = false);
    bool gathered_decode(std::uint64_t position) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool adapter_active(const llama_model * model);
llm_graph_input_attn_kv * adapter_input(const llm_graph_context & graph);
ggml_tensor * adapter_attention(const llm_graph_context & graph, llm_graph_input_attn_kv * input,
                                ggml_tensor * query, ggml_tensor * key, ggml_tensor * value, float scale, int layer);
} // namespace solid
