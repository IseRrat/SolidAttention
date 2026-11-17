#include "solid/llama_adapter.hpp"
#include "solid/projection.hpp"
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
    bool packed_inputs = false;
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
            for (int source = 0; source < (self.packed_inputs ? 2 : 3); ++source) {
                const auto * tensor = destination->src[source];
                if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) throw std::logic_error("attention inputs must be contiguous F32");
                inputs[source].resize(static_cast<std::size_t>(ggml_nelements(tensor)));
                std::memcpy(inputs[source].data(), tensor->data, inputs[source].size() * sizeof(float));
            }
            if (self.gathered) {
                const auto result = self.session.gather(self.layer, self.frame->positions.front(), inputs[0], inputs[1], inputs[2], self.packed_inputs);
                if (result.size() != static_cast<std::size_t>(ggml_nelements(destination))) throw std::logic_error("gather output shape");
                std::memcpy(destination->data, result.data(), result.size() * sizeof(Half));
            } else {
                const auto result = self.session.process(self.layer, self.frame->positions.front(), count, inputs[0], inputs[1], inputs[2], self.scale, self.packed_inputs);
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
ggml_tensor * AttentionNode::build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                                  ggml_tensor * packed_kv, ggml_backend_sched * scheduler, ggml_backend * decode_backend) {
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
    if (packed_kv && (packed_kv->type != GGML_TYPE_F32 || !ggml_is_contiguous(packed_kv) ||
        packed_kv->ne[0] != g.dimension * 2 || packed_kv->ne[1] != g.heads || packed_kv->ne[2] != query->ne[2] || packed_kv->ne[3] != 1)) {
        throw std::invalid_argument("packed KV graph input shape");
    }
    impl_->packed_inputs = packed_kv != nullptr;
    ggml_tensor * inputs[] = {query, packed_kv ? packed_kv : key, packed_kv ? nullptr : value};
    for (int i = 0; i < (packed_kv ? 2 : 3); ++i) {
        auto & input = inputs[i];
        if (input->type != GGML_TYPE_F32) input = ggml_cast(context, input, GGML_TYPE_F32);
        input = ggml_cont(context, input);
    }
    if (impl_->gathered && query->ne[2] != 1) throw std::invalid_argument("gathered attention requires one decode token");
    auto * result = ggml_custom_4d(context, impl_->gathered ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   impl_->gathered ? static_cast<int64_t>(impl_->session.gather_elements()) : query->ne[0] * query->ne[1],
                                   impl_->gathered ? 1 : query->ne[2], 1, 1,
                                   inputs, packed_kv ? 2 : 3, Impl::compute, 1, impl_.get());
    ggml_set_name(result, "solid_sync_attention");
    impl_->host = result;
    if (impl_->gathered) {
        if (g.layout == KvLayout::interleaved && scheduler && decode_backend) {
            result = ggml_dup(context, result);
            ggml_set_name(result, "solid_kv_upload");
            ggml_backend_sched_set_tensor_backend(scheduler, result, decode_backend);
        }
        const auto slots = impl_->session.gather_slots();
        const auto row = static_cast<std::size_t>(g.dimension) * sizeof(Half);
        const auto head_stride = row * slots * (g.layout == KvLayout::interleaved ? 2 : 1);
        const auto token_stride = g.layout == KvLayout::interleaved ? row * 2 : row;
        const auto value_offset = g.layout == KvLayout::interleaved ? row : row * slots * g.heads;
        auto * keys = ggml_view_3d(context, result, g.dimension, slots, g.heads, token_stride, head_stride, 0);
        auto * values = ggml_view_3d(context, result, g.dimension, slots, g.heads, token_stride, head_stride, value_offset);
        const auto mask_offset = row * slots * 2 * g.heads;
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
    std::vector<std::unique_ptr<KvProjection>> projections;
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
    KvGeometry geometry{hp.n_layer(), hp.n_head_kv(), hp.n_embd_head_k(), options.block_tokens, capacity, 4096,
                        options.interleaved ? KvLayout::interleaved : KvLayout::separate};
    for (std::uint32_t layer = 0; layer < hp.n_layer(); ++layer) {
        if (hp.n_head(layer) != hp.n_head() || hp.n_head_kv(layer) != hp.n_head_kv() ||
            hp.n_embd_head_k(layer) != geometry.dimension || hp.n_embd_head_v(layer) != geometry.dimension) {
            throw std::invalid_argument("adapter requires uniform layer geometry");
        }
    }
    if (options.fuse_kv_projection) {
        for (std::uint32_t id = 0; id < hp.n_layer(); ++id) {
            const auto & layer = model->layers[id];
            if (layer.wqkv || !layer.wq || layer.wk_s || layer.wv_s) {
                throw std::invalid_argument("fused KV projection requires separate Q/K/V weights with native row encodings and no weight scales");
            }
            auto * buft = ggml_backend_dev_buffer_type(model->dev_layer(id));
            impl_->projections.push_back(std::make_unique<KvProjection>(layer.wk, layer.wv, layer.wk_b, layer.wv_b,
                                                                      geometry.heads, geometry.dimension, buft));
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
    impl_->session = std::make_unique<AttentionSession>(std::move(store), hp.n_head(), budget, options.sparse, prefill_tokens, trace, options.selection_replay);
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (!registry.emplace(model, this).second) throw std::invalid_argument("one attention session per model is supported");
}
AttentionAdapter::~AttentionAdapter() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.erase(impl_->model);
}
void AttentionAdapter::check() const { impl_->failure->check(); }
void AttentionAdapter::flush() { check(); impl_->session->flush(); }
void AttentionAdapter::finish() { check(); impl_->session->finish(); }
SessionStats AttentionAdapter::stats() const { check(); return impl_->session->stats(); }
SparseBudget AttentionAdapter::budget() const { return impl_->session->budget(); }
std::unique_ptr<AttentionNode> AttentionAdapter::node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale, bool gathered) {
    return std::make_unique<AttentionNode>(*impl_->session, std::move(frame), layer, scale, impl_->failure, gathered);
}
bool AttentionAdapter::gathered_decode(std::uint64_t position) const { return impl_->gpu_decode && position >= impl_->prefill_tokens; }
bool AttentionAdapter::fused_projection() const { return !impl_->projections.empty(); }
const KvProjection & AttentionAdapter::projection(std::uint32_t layer) const { return *impl_->projections.at(layer); }
std::uint64_t AttentionAdapter::projection_buffer_bytes() const {
    std::uint64_t bytes = 0;
    for (const auto & p : impl_->projections) bytes += p->stats().buffer_bytes;
    return bytes;
}
std::uint64_t AttentionAdapter::projection_initialization_ns() const {
    std::uint64_t ns = 0;
    for (const auto & p : impl_->projections) ns += p->stats().initialization_ns;
    return ns;
}
std::vector<std::string> AttentionAdapter::projection_weight_types() const {
    std::vector<std::string> types;
    for (const auto & p : impl_->projections) types.push_back(p->stats().weight_type);
    return types;
}

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
    std::unordered_map<int, ggml_tensor *> projected_kv;
};
bool adapter_active(const llama_model * model) { return lookup(model) != nullptr; }
bool adapter_fused_projection(const llama_model * model) {
    const auto * adapter = lookup(model);
    return adapter && adapter->fused_projection();
}
llm_graph_qkv adapter_qkv(const llm_graph_context & graph, const llama_layer & layer, ggml_tensor * input,
                         std::int64_t dimension_q, std::int64_t heads_q, std::int64_t dimension_k,
                         std::int64_t heads_k, std::int64_t dimension_v, std::int64_t heads_v, int id, bool reshape) {
    auto * adapter = lookup(&graph.model);
    if (!adapter || !reshape || id < 0 || dimension_k != dimension_v || heads_k != heads_v ||
        dimension_k != graph.hparams.n_embd_head_k() || heads_k != graph.hparams.n_head_kv() ||
        !graph.loras->empty() || graph.hdmd) throw std::invalid_argument("unsupported fused KV projection graph");
    auto * q = graph.build_lora_mm(layer.wq, input, layer.wq_s);
    if (layer.wq_b) q = ggml_add(graph.ctx0, q, layer.wq_b);
    auto * packed = adapter->projection(id).project(graph.ctx0, input);
    if (graph.hparams.f_clamp_kqv > 0) {
        const auto bound = graph.hparams.f_clamp_kqv;
        q = ggml_clamp(graph.ctx0, q, -bound, bound);
        packed = ggml_clamp(graph.ctx0, packed, -bound, bound);
    }
    AdapterInput * custom = nullptr;
    for (const auto & candidate : graph.res->inputs) {
        if (auto * found = dynamic_cast<AdapterInput *>(candidate.get())) { custom = found; break; }
    }
    if (!custom || !custom->projected_kv.emplace(id, packed).second) throw std::logic_error("fused KV graph input missing or repeated layer");
    return {ggml_reshape_3d(graph.ctx0, q, dimension_q, heads_q, input->ne[1]),
            KvProjection::key_view(graph.ctx0, packed), KvProjection::value_view(graph.ctx0, packed)};
}
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
    ggml_tensor * packed = nullptr;
    if (custom->adapter.fused_projection()) {
        packed = KvProjection::update_key(graph.ctx0, custom->projected_kv.at(layer), key);
    }
    auto * device = graph.backend_cpu;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(graph.sched); ++i) {
        auto * candidate = ggml_backend_sched_get_backend(graph.sched, i);
        if (ggml_backend_get_device(candidate) == graph.model.dev_layer(layer)) { device = candidate; break; }
    }
    auto * result = node->build(graph.ctx0, query, key, value, packed, graph.sched, device);
    ggml_backend_sched_set_tensor_backend(graph.sched, node->host_node(), graph.backend_cpu);
    custom->nodes.push_back(std::move(node));
    return result;
}
} // namespace solid
