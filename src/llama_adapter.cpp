#include "solid/llama_adapter.hpp"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace solid {
namespace {
std::mutex registry_mutex;
std::unordered_map<const llama_model *, AttentionAdapter *> registry;
AttentionAdapter * lookup(const llama_model * model) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find(model);
    return it == registry.end() ? nullptr : it->second;
}
}
void AttentionFailure::capture(std::exception_ptr error, std::uint32_t layer, std::uint64_t position) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) { error_ = std::move(error); layer_ = layer; position_ = position; }
}
void AttentionFailure::check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) {
        const auto context = "attention layer=" + std::to_string(layer_) + " position=" + std::to_string(position_) + ": ";
        try { std::rethrow_exception(error_); }
        catch (const std::exception & error) { throw std::runtime_error(context + error.what()); }
        catch (...) { throw std::runtime_error(context + "unknown failure"); }
    }
}
struct AttentionNode::Impl {
    AttentionSession & session;
    std::shared_ptr<AttentionFrame> frame;
    std::uint32_t layer;
    float scale;
    std::shared_ptr<AttentionFailure> failure;
    bool gathered;
    ggml_tensor * host = nullptr;

    static void compute(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith != 0) return;
        auto & self = *static_cast<Impl *>(userdata);
        try {
            self.failure->check();
            const auto count = static_cast<std::size_t>(destination->ne[1]);
            if (self.frame->positions.size() != count || count == 0) throw std::invalid_argument("attention graph positions are not initialized");
            for (std::size_t i = 1; i < count; ++i) {
                if (self.frame->positions[i] != self.frame->positions[0] + i) throw std::invalid_argument("attention requires consecutive positions");
            }
            std::vector<float> inputs[3];
            for (int source = 0; source < 3; ++source) {
                const auto * tensor = destination->src[source];
                if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) throw std::logic_error("attention inputs must be contiguous F32");
                inputs[source].resize(static_cast<std::size_t>(ggml_nelements(tensor)));
                std::memcpy(inputs[source].data(), tensor->data, inputs[source].size() * sizeof(float));
            }
            if (self.gathered) {
                const auto result = self.session.gather(self.layer, self.frame->positions.front(), inputs[0], inputs[1], inputs[2]);
                if (result.size() != static_cast<std::size_t>(ggml_nelements(destination))) throw std::logic_error("gather output shape");
                std::memcpy(destination->data, result.data(), result.size() * sizeof(Half));
            } else {
                const auto result = self.session.process(self.layer, self.frame->positions.front(), count, inputs[0], inputs[1], inputs[2], self.scale);
                if (result.size() != static_cast<std::size_t>(ggml_nelements(destination))) throw std::logic_error("attention output shape");
                std::memcpy(destination->data, result.data(), result.size() * sizeof(float));
            }
        } catch (...) {
            self.failure->capture(std::current_exception(), self.layer, self.frame->positions.empty() ? 0 : self.frame->positions.front());
            if (self.gathered) {
                auto * data = static_cast<Half *>(destination->data);
                std::fill_n(data, ggml_nelements(destination), Half{0});
                const auto slots = self.session.gather_slots();
                const auto mask = static_cast<std::size_t>(slots) * 2 * self.session.geometry().heads * self.session.geometry().dimension;
                std::fill(data + mask, data + ggml_nelements(destination), Half{0xfc00});
                for (std::uint32_t head = 0; head < self.session.query_heads(); ++head) data[mask + head * slots] = 0;
            } else std::fill_n(static_cast<float *>(destination->data), ggml_nelements(destination), 0.0f);
        }
    }
};
AttentionNode::AttentionNode(AttentionSession & session, std::shared_ptr<AttentionFrame> frame,
                            std::uint32_t layer, float scale, std::shared_ptr<AttentionFailure> failure, bool gathered) :
    impl_(std::make_unique<Impl>(Impl{session, std::move(frame), layer, scale,
                                    failure ? std::move(failure) : std::make_shared<AttentionFailure>(), gathered})) {}
AttentionNode::~AttentionNode() = default;
ggml_tensor * AttentionNode::build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value) {
    const auto & g = impl_->session.geometry();
    if (!std::isfinite(impl_->scale) || impl_->scale <= 0) throw std::invalid_argument("invalid attention scale");
    if (impl_->gathered && impl_->session.gather_elements() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / sizeof(Half)) {
        throw std::overflow_error("gather allocation exceeds backend limits");
    }
    if (query->ne[0] != g.dimension || key->ne[0] != g.dimension || value->ne[0] != g.dimension ||
        query->ne[1] != impl_->session.query_heads() || key->ne[1] != g.heads || value->ne[1] != g.heads ||
        query->ne[2] != key->ne[2] || key->ne[2] != value->ne[2] || query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1) {
        throw std::invalid_argument("unsupported attention tensor shape");
    }
    ggml_tensor * inputs[] = {query, key, value};
    for (auto & input : inputs) {
        if (input->type != GGML_TYPE_F32) input = ggml_cast(context, input, GGML_TYPE_F32);
        input = ggml_cont(context, input);
    }
    if (impl_->gathered && query->ne[2] != 1) throw std::invalid_argument("gathered attention requires one decode token");
    auto * result = ggml_custom_4d(context, impl_->gathered ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   impl_->gathered ? static_cast<int64_t>(impl_->session.gather_elements()) : query->ne[0] * query->ne[1],
                                   impl_->gathered ? 1 : query->ne[2], 1, 1,
                                   inputs, 3, Impl::compute, 1, impl_.get());
    ggml_set_name(result, "solid_sync_attention");
    impl_->host = result;
    if (impl_->gathered) {
        const auto slots = impl_->session.gather_slots();
        const auto stride = static_cast<std::size_t>(g.dimension) * 2 * sizeof(Half);
        auto * keys = ggml_view_3d(context, result, g.dimension, slots, g.heads, stride, stride * slots, 0);
        auto * values = ggml_view_3d(context, result, g.dimension, slots, g.heads, stride, stride * slots, g.dimension * sizeof(Half));
        const auto mask_offset = static_cast<std::size_t>(slots) * stride * g.heads;
        auto * mask = ggml_view_3d(context, result, slots, 1, impl_->session.query_heads(), slots * sizeof(Half), slots * sizeof(Half), mask_offset);
        auto * q = query->type == GGML_TYPE_F32 ? query : ggml_cast(context, query, GGML_TYPE_F32);
        q = ggml_permute(context, q, 0, 2, 1, 3);
        result = ggml_flash_attn_ext(context, q, keys, values, mask, impl_->scale, 0, 0);
        ggml_prec_set_acc(result, GGML_PREC_F32);
        result = ggml_reshape_2d(context, result, g.dimension * impl_->session.query_heads(), 1);
        ggml_set_name(result, "solid_gathered_attention");
    }
    return result;
}
void AttentionNode::check() const { impl_->failure->check(); }
ggml_tensor * AttentionNode::host_node() const { return impl_->host; }

struct AttentionAdapter::Impl {
    const llama_model * model;
    std::unique_ptr<AttentionSession> session;
    std::shared_ptr<AttentionFailure> failure = std::make_shared<AttentionFailure>();
    bool gpu_decode;
    std::uint64_t prefill_tokens;
};
AttentionAdapter::AttentionAdapter(const llama_model * model, const AttentionConfig & options,
                                  std::uint32_t capacity, std::uint64_t prefill_tokens, const std::filesystem::path & trace) :
    impl_(std::make_unique<Impl>()) {
    if (!model || !options.enabled || (model->arch != LLM_ARCH_LLAMA && model->arch != LLM_ARCH_QWEN2)) {
        throw std::invalid_argument("adapter requires a Llama or Qwen2 decoder");
    }
    const auto & hp = model->hparams;
    if (hp.n_expert != 0 || hp.n_swa != 0 || hp.attn_soft_cap || hp.f_max_alibi_bias != 0 ||
        hp.n_embd_head_k() != hp.n_embd_head_v()) {
        throw std::invalid_argument("adapter requires dense uniform layers with equal K/V dimensions");
    }
    KvGeometry geometry{hp.n_layer(), hp.n_head_kv(), hp.n_embd_head_k(), options.block_tokens, capacity, 4096, KvLayout::separate};
    for (std::uint32_t layer = 0; layer < hp.n_layer(); ++layer) {
        if (hp.n_head(layer) != hp.n_head() || hp.n_head_kv(layer) != hp.n_head_kv() ||
            hp.n_embd_head_k(layer) != geometry.dimension || hp.n_embd_head_v(layer) != geometry.dimension) {
            throw std::invalid_argument("adapter requires uniform layer geometry");
        }
    }
    SparseBudget budget;
    budget.total = options.context_budget_tokens ? options.context_budget_tokens :
        static_cast<std::uint32_t>(prefill_tokens < 4096 ? std::max<std::uint64_t>(1, prefill_tokens / 4) : 1024);
    if (options.context_budget_tokens) { budget.initial = options.init_tokens; budget.local = options.local_tokens; }
    else {
        const auto deterministic = std::max<std::uint32_t>(1, budget.total / 2);
        budget.initial = std::min(options.block_tokens, deterministic / 2);
        budget.local = deterministic - budget.initial;
    }
    budget.representatives = options.representative_count;
    if (!options.sparse) { budget.total = capacity; budget.initial = 0; budget.local = capacity; }
    std::unique_ptr<KvStore> store;
    if (options.in_memory) store = std::make_unique<MemoryKvStore>(geometry);
    else {
        if (!options.session_file.parent_path().empty()) std::filesystem::create_directories(options.session_file.parent_path());
        store = std::make_unique<FileKvStore>(geometry, create_session_file(options.session_file, !options.keep_session_file),
                                              options.write_buffer_bytes_per_layer);
    }
    impl_->model = model;
    impl_->gpu_decode = options.gpu_decode;
    impl_->prefill_tokens = prefill_tokens;
    impl_->session = std::make_unique<AttentionSession>(std::move(store), hp.n_head(), budget, options.sparse, prefill_tokens, trace);
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (!registry.emplace(model, this).second) throw std::invalid_argument("one attention session per model is supported");
}
AttentionAdapter::~AttentionAdapter() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.erase(impl_->model);
}
void AttentionAdapter::check() const { impl_->failure->check(); }
void AttentionAdapter::flush() { check(); impl_->session->flush(); }
SessionStats AttentionAdapter::stats() const { check(); return impl_->session->stats(); }
SparseBudget AttentionAdapter::budget() const { return impl_->session->budget(); }
std::unique_ptr<AttentionNode> AttentionAdapter::node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale, bool gathered) {
    return std::make_unique<AttentionNode>(*impl_->session, std::move(frame), layer, scale, impl_->failure, gathered);
}
bool AttentionAdapter::gathered_decode(std::uint64_t position) const { return impl_->gpu_decode && position >= impl_->prefill_tokens; }

class AdapterInput final : public llm_graph_input_attn_kv {
public:
    AdapterInput(const llm_graph_context & graph, AttentionAdapter & adapter) :
        llm_graph_input_attn_kv(graph.hparams, graph.cparams, nullptr), adapter(adapter), frame(std::make_shared<AttentionFrame>()) {}
    void set_input(const llama_ubatch * batch) override {
        if (!batch || batch->n_pos != 1 || !batch->pos || !batch->n_seq_id || !batch->seq_id) {
            throw std::invalid_argument("adapter requires explicit text positions and sequence IDs");
        }
        frame->positions.resize(batch->n_tokens);
        for (std::uint32_t i = 0; i < batch->n_tokens; ++i) {
            if (batch->pos[i] < 0 || batch->n_seq_id[i] != 1 || batch->seq_id[i][0] != 0) {
                throw std::invalid_argument("adapter supports one sequence with nonnegative positions");
            }
            frame->positions[i] = static_cast<std::uint64_t>(batch->pos[i]);
        }
    }
    bool can_reuse(const llm_graph_params &) override { return false; }
    AttentionAdapter & adapter;
    std::shared_ptr<AttentionFrame> frame;
    std::vector<std::unique_ptr<AttentionNode>> nodes;
};
bool adapter_active(const llama_model * model) { return lookup(model) != nullptr; }
llm_graph_input_attn_kv * adapter_input(const llm_graph_context & graph) {
    auto * adapter = lookup(&graph.model);
    if (!adapter) throw std::logic_error("attention adapter is not registered");
    return static_cast<AdapterInput *>(graph.res->add_input(std::make_unique<AdapterInput>(graph, *adapter)));
}
ggml_tensor * adapter_attention(const llm_graph_context & graph, llm_graph_input_attn_kv * input,
                                ggml_tensor * query, ggml_tensor * key, ggml_tensor * value, float scale, int layer) {
    auto * custom = dynamic_cast<AdapterInput *>(input);
    if (!custom || layer < 0) throw std::logic_error("invalid attention graph input");
    const bool gathered = query->ne[2] == 1 && graph.ubatch.pos && graph.ubatch.pos[0] >= 0 &&
        custom->adapter.gathered_decode(static_cast<std::uint64_t>(graph.ubatch.pos[0]));
    auto node = custom->adapter.node(custom->frame, static_cast<std::uint32_t>(layer), scale, gathered);
    auto * result = node->build(graph.ctx0, query, key, value);
    ggml_backend_sched_set_tensor_backend(graph.sched, node->host_node(), graph.backend_cpu);
    custom->nodes.push_back(std::move(node));
    return result;
}
} // namespace solid
