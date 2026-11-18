#include "solid/llama_adapter.hpp"
#include "solid/projection.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/device_ops.hpp"
#include "solid/thread_affinity.hpp"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "ggml-backend.h"
#ifdef SOLID_HAVE_CUDA_SELECTION
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace solid {
namespace {
std::mutex registry_mutex;
std::unordered_map<const llama_model *, AttentionAdapter *> registry;
AttentionAdapter * lookup(const llama_model * model) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find(model);
    return it == registry.end() ? nullptr : it->second;
}
struct PrefillOutput {
    AttentionSession & session;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context{nullptr,ggml_free};
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer{nullptr,ggml_backend_buffer_free};
    ggml_tensor * tensor=nullptr;
    std::uint64_t bytes=0;
    PrefillOutput(AttentionSession & session,ggml_backend_t backend,std::int64_t width,std::int64_t tokens):session(session) {
        context.reset(ggml_init({ggml_tensor_overhead()+1024,nullptr,true}));
        if (!context) throw std::runtime_error("prefill result metadata allocation");
        tensor=ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,width,tokens);
        ggml_set_name(tensor,"solid_prefill_device_result_source");
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(),backend));
        if (!buffer) throw std::runtime_error("prefill result device allocation");
        ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer.get(),0);
        const auto allocated=ggml_backend_buffer_get_size(buffer.get());
        session.retain_prefill_graph_output(allocated); bytes=allocated;
    }
    ~PrefillOutput() { session.release_prefill_graph_output(bytes); }
};
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
    GgmlPrefetch * prefetch;
    bool packed_inputs = false;
    std::optional<DeviceMemory> direct = {};
    ggml_tensor * device_query = nullptr;
    ggml_tensor * device_kv = nullptr;
    std::unique_ptr<PrefillOutput> prefill_output = {};
    ggml_tensor * host = nullptr;
    ggml_tensor * early = nullptr;
    ggml_tensor * flash = nullptr;
    bool query_proof = false;
    std::uint64_t epoch = 0;
    static void publish_completion(ggml_tensor * destination,int ith,int,void * userdata) {
        if (ith) return;
        std::memcpy(destination->data,destination->src[0]->data,ggml_nbytes(destination));
        auto * token=static_cast<StreamCompletionToken *>(userdata); token->notify(token->userdata);
    }
    static void select_query(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith != 0) return;
        auto & self = *static_cast<Impl *>(userdata);
        *static_cast<float *>(destination->data) = 0;
        try {
            self.failure->check();
            if (self.frame->positions.size() != 1) throw std::invalid_argument("early query requires one initialized position");
            const auto * source = destination->src[0];
            if (source->type != GGML_TYPE_F32 || !ggml_is_contiguous(source)) throw std::logic_error("early query input must be contiguous F32");
            if (self.direct) {
                if (ggml_nelements(source) != 1 || !self.device_query || !self.device_query->data) throw std::logic_error("device query token or allocation missing");
                self.prefetch->select_early_device(self.layer, self.frame->positions.front(),
                    {static_cast<const float *>(self.device_query->data),static_cast<std::size_t>(ggml_nelements(self.device_query)),*self.direct},self.query_proof);
            } else {
                std::vector<float> query(static_cast<std::size_t>(ggml_nelements(source)));
                std::memcpy(query.data(), source->data, query.size() * sizeof(float));
                self.prefetch->select_early(self.layer, self.frame->positions.front(), query, self.query_proof);
            }
        } catch (...) { self.failure->capture(std::current_exception(), self.layer, self.frame->positions.empty() ? 0 : self.frame->positions.front()); }
    }

    static void compute(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith != 0) return;
        auto & self = *static_cast<Impl *>(userdata);
        try {
            self.failure->check();
            const auto count = static_cast<std::size_t>(self.prefill_output?self.device_query->ne[2]:destination->ne[1]);
            if (self.frame->positions.size() != count || count == 0) throw std::invalid_argument("attention graph positions are not initialized");
            for (std::size_t i = 1; i < count; ++i) {
                if (self.frame->positions[i] != self.frame->positions[0] + i) throw std::invalid_argument("attention requires consecutive positions");
            }
            std::vector<float> inputs[3];
            const bool early = self.early != nullptr;
            for (int source = 0; !self.direct && source < (self.packed_inputs ? 2 : 3) - (early ? 1 : 0); ++source) {
                const auto * tensor = destination->src[source];
                if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) throw std::logic_error("attention inputs must be contiguous F32");
                inputs[source].resize(static_cast<std::size_t>(ggml_nelements(tensor)));
                std::memcpy(inputs[source].data(), tensor->data, inputs[source].size() * sizeof(float));
            }
            if (self.prefill_output) {
                const auto * source=destination->src[0];
                if (source->type!=GGML_TYPE_F16 || !ggml_is_contiguous(source)) throw std::logic_error("prefill offload requires contiguous FP16 KV");
                std::vector<Half> kv(static_cast<std::size_t>(ggml_nelements(source)));
                std::memcpy(kv.data(),source->data,kv.size()*sizeof(Half));
                self.session.process_prefill_device(self.layer,self.frame->positions.front(),static_cast<std::uint32_t>(count),
                    {static_cast<const float *>(self.device_query->data),static_cast<std::size_t>(ggml_nelements(self.device_query)),*self.direct},
                    {static_cast<const Half *>(self.device_kv->data),kv.size(),*self.direct},kv,self.scale,
                    {static_cast<float *>(self.prefill_output->tensor->data),static_cast<std::size_t>(ggml_nelements(self.prefill_output->tensor)),*self.direct});
                *static_cast<float *>(destination->data)=0;
            } else if (self.prefetch) {
                std::vector<Half> result;
                if (self.direct) {
                    const auto * source = destination->src[0];
                    if (source->type != GGML_TYPE_F16 || !ggml_is_contiguous(source) || !self.device_kv || !self.device_kv->data) throw std::logic_error("device KV offload or allocation missing");
                    std::vector<Half> kv(static_cast<std::size_t>(ggml_nelements(source)));
                    std::memcpy(kv.data(),source->data,kv.size()*sizeof(Half));
                    std::optional<DeviceQuery> serial_query;
                    if (!early) serial_query = DeviceQuery{static_cast<const float *>(self.device_query->data),
                        static_cast<std::size_t>(ggml_nelements(self.device_query)),*self.direct};
                    result = self.prefetch->decode_half(self.layer,self.frame->positions.front(),kv,
                        {static_cast<const Half *>(self.device_kv->data),kv.size(),static_cast<std::size_t>(self.device_kv->ne[0]),*self.direct},serial_query,self.device_kv);
                } else result = early ? self.prefetch->decode(self.layer, self.frame->positions.front(), {}, inputs[0], self.packed_inputs ? std::vector<float>{} : inputs[1], self.packed_inputs) :
                    self.prefetch->decode(self.layer, self.frame->positions.front(), inputs[0], inputs[1], inputs[2], self.packed_inputs);
                self.epoch = self.prefetch->attention_epoch();
                if (result.size() != static_cast<std::size_t>(ggml_nelements(destination))) throw std::logic_error("prefetch mask shape");
                std::memcpy(destination->data, result.data(), result.size() * sizeof(Half));
            } else if (self.gathered) {
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
            if (self.prefetch) {
                // The remaining graph uses a finite fallback row. Finish every
                // producer before that graph can read the shared device bank.
                try { self.prefetch->quiesce(); }
                catch (...) { self.failure->capture(std::current_exception(), self.layer, self.frame->positions.empty() ? 0 : self.frame->positions.front()); }
                self.epoch = self.prefetch->attention_epoch();
                auto * data = static_cast<Half *>(destination->data);
                std::fill_n(data, ggml_nelements(destination), Half{0xfc00});
                for (std::uint32_t h = 0; h < self.session.query_heads(); ++h) data[h * self.prefetch->tokens_per_head()] = 0;
            } else if (self.gathered) {
                auto * data = static_cast<Half *>(destination->data);
                std::fill_n(data, ggml_nelements(destination), Half{0});
                const auto slots = self.session.gather_slots();
                const auto mask = static_cast<std::size_t>(slots) * 2 * self.session.geometry().heads * self.session.geometry().dimension;
                std::fill(data + mask, data + ggml_nelements(destination), Half{0xfc00});
                for (std::uint32_t head = 0; head < self.session.query_heads(); ++head) data[mask + head * slots] = 0;
            } else {
                if (self.prefill_output) ggml_backend_buffer_clear(self.prefill_output->buffer.get(),0);
                std::fill_n(static_cast<float *>(destination->data), ggml_nelements(destination), 0.0f);
            }
        }
    }
};
AttentionNode::AttentionNode(AttentionSession & session, std::shared_ptr<AttentionFrame> frame,
                            std::uint32_t layer, float scale, std::shared_ptr<AttentionFailure> failure, bool gathered, GgmlPrefetch * prefetch) :
    impl_(std::make_unique<Impl>(Impl{session, std::move(frame), layer, scale,
                                    failure ? std::move(failure) : std::make_shared<AttentionFailure>(), gathered, prefetch})) {
    if (prefetch && !gathered) throw std::invalid_argument("prefetch node requires decode attention");
}
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
    impl_->direct = impl_->prefetch ? impl_->prefetch->direct_memory() : (!impl_->gathered?impl_->session.prefill_direct_memory():std::optional<DeviceMemory>{});
    if (impl_->direct && *impl_->direct == DeviceMemory::cuda && (!scheduler || !decode_backend)) throw std::invalid_argument("device Q/K graph requires scheduler and decode backend");
    if (impl_->direct && !impl_->prefetch) {
        if (!decode_backend) throw std::invalid_argument("device prefill requires a result backend");
        const auto type=ggml_backend_dev_type(ggml_backend_get_device(decode_backend));
        if ((*impl_->direct==DeviceMemory::host_emulator) != (type==GGML_BACKEND_DEVICE_TYPE_CPU)) throw std::invalid_argument("prefill result memory/backend mismatch");
        impl_->prefill_output=std::make_unique<PrefillOutput>(impl_->session,decode_backend,query->ne[0]*query->ne[1],query->ne[2]);
    }
    const auto on_device = [&](ggml_tensor * tensor) {
        if (scheduler && decode_backend) ggml_backend_sched_set_tensor_backend(scheduler,tensor,decode_backend);
        return tensor;
    };
    ggml_tensor * inputs[] = {query, packed_kv ? packed_kv : key, packed_kv ? nullptr : value};
    for (int i = 0; i < (impl_->direct ? 1 : (packed_kv ? 2 : 3)); ++i) {
        auto & input = inputs[i];
        if (input->type != GGML_TYPE_F32) input = ggml_cast(context, input, GGML_TYPE_F32);
        input = ggml_cont(context, input);
    }
    int input_count = packed_kv ? 2 : 3;
    if (impl_->direct) {
        impl_->device_query = on_device(inputs[0]);
        auto * token = on_device(device_query_token(context,impl_->device_query));
        if (impl_->prefetch && impl_->prefetch->early_enabled()) {
            impl_->early = ggml_custom_4d(context,GGML_TYPE_F32,1,1,1,1,&token,1,Impl::select_query,1,impl_.get());
            ggml_set_name(impl_->early,"solid_early_selection_device");
        }
        const auto vectors=[&](ggml_tensor * source) {
            if (source->type!=GGML_TYPE_F32) source=ggml_cast(context,source,GGML_TYPE_F32);
            if (source->nb[0]!=sizeof(float) || source->nb[1]<static_cast<std::size_t>(source->ne[0])*sizeof(float) ||
                source->nb[2]<static_cast<std::size_t>(source->ne[1])*source->nb[1]) source=ggml_cont(context,source);
            return source;
        };
        key=vectors(key); value=vectors(value);
        // The query and early frontier extend Q's allocation through selection.
        impl_->device_kv = on_device(device_consolidate_kv(context,key,value,impl_->device_query,impl_->early?impl_->early:token));
        inputs[0] = impl_->device_kv; input_count = 1;
    } else if (impl_->prefetch && impl_->prefetch->early_enabled()) {
        auto * q = inputs[0];
        impl_->early = ggml_custom_4d(context, GGML_TYPE_F32, 1, 1, 1, 1, &q, 1, Impl::select_query, 1, impl_.get());
        ggml_set_name(impl_->early, "solid_early_selection");
        if (packed_kv) { inputs[0] = inputs[1]; inputs[1] = impl_->early; input_count = 2; }
        else { inputs[0] = inputs[1]; inputs[1] = inputs[2]; inputs[2] = impl_->early; input_count = 3; }
    }
    if (impl_->gathered && query->ne[2] != 1) throw std::invalid_argument("gathered attention requires one decode token");
    const auto elements = impl_->prefetch ? static_cast<std::uint64_t>(impl_->prefetch->tokens_per_head()) * impl_->session.query_heads() :
                                          impl_->session.gather_elements();
    auto * result = ggml_custom_4d(context, impl_->gathered ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   impl_->prefill_output?1:(impl_->gathered ? static_cast<int64_t>(elements) : query->ne[0] * query->ne[1]),
                                   impl_->prefill_output || impl_->gathered ? 1 : query->ne[2], 1, 1,
                                   inputs, input_count, Impl::compute, 1, impl_.get());
    ggml_set_name(result, "solid_sync_attention");
    impl_->host = result;
    if (impl_->prefill_output) {
        result=on_device(device_prefill_result(context,impl_->prefill_output->tensor,result,impl_->device_query,impl_->device_kv));
    } else if (impl_->prefetch) {
        // The future GPU consumer retains KV until the host append has used K.
        if (impl_->direct) result = on_device(device_release_mask(context,result,impl_->device_kv,impl_->device_query));
        const auto slots = impl_->prefetch->tokens_per_head();
        auto * bank = impl_->prefetch->tensor();
        const auto stride = static_cast<std::size_t>(g.dimension) * 2 * sizeof(Half);
        auto * keys = ggml_view_3d(context, bank, g.dimension, slots, g.heads, stride, bank->nb[2], 0);
        auto * values = ggml_view_3d(context, bank, g.dimension, slots, g.heads, stride, bank->nb[2], g.dimension * sizeof(Half));
        auto * mask = ggml_view_3d(context, result, slots, 1, impl_->session.query_heads(), slots * sizeof(Half), slots * sizeof(Half), 0);
        auto * q = query->type == GGML_TYPE_F32 ? query : ggml_cast(context, query, GGML_TYPE_F32);
        q = ggml_permute(context, q, 0, 2, 1, 3);
        result = ggml_flash_attn_ext(context, q, keys, values, mask, impl_->scale, 0, 0);
        ggml_prec_set_acc(result, GGML_PREC_F32);
        ggml_set_name(result, "solid_prefetched_flash_attention");
        impl_->flash = result;
        if (scheduler && decode_backend) ggml_backend_sched_set_tensor_backend(scheduler, result, decode_backend);
        if (auto * token=impl_->prefetch->completion_token(impl_->layer)) {
            static_assert(stream_completion_offset+sizeof(StreamCompletionBinding)<=GGML_MAX_OP_PARAMS,"completion binding exceeds GGML op params");
            bind_stream_completion(result->op_params,sizeof(result->op_params),token);
            ggml_set_name(result,"solid_prefetched_flash_attention_native");
            if (impl_->prefetch->completion_host_marker()) {
                auto * source=result;
                result=ggml_custom_4d(context,GGML_TYPE_F32,source->ne[0],source->ne[1],source->ne[2],source->ne[3],
                    &source,1,Impl::publish_completion,1,token);
                ggml_set_name(result,"solid_attention_completion_emulator");
            }
        }
        result = ggml_reshape_2d(context, result, g.dimension * impl_->session.query_heads(), 1);
    } else if (impl_->gathered) {
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
ggml_tensor * AttentionNode::selection_node() const { return impl_->early; }
ggml_tensor * AttentionNode::attention_node() const { return impl_->flash; }
void AttentionNode::set_query_proof(bool covers) { impl_->query_proof = covers; }
void AttentionNode::defer_completion() {
    if (impl_->prefetch && impl_->prefetch->reuses_sync() && impl_->flash && !impl_->prefetch->completion_token(impl_->layer))
        ggml_set_name(impl_->flash, "solid_prefetched_flash_attention_reuse");
}
void AttentionNode::complete_attention() { if (impl_->prefetch) impl_->prefetch->completed(impl_->epoch); }

struct AttentionAdapter::Impl {
    const llama_model * model;
    std::unique_ptr<WorkerPlacement> placement;
    std::unique_ptr<AttentionSession> session;
    std::shared_ptr<AttentionFailure> failure = std::make_shared<AttentionFailure>();
    bool gpu_decode;
    std::uint64_t prefill_tokens;
    std::vector<std::unique_ptr<KvProjection>> projections;
    std::filesystem::path residency_trace;
    std::filesystem::path schedule_trace;
    ScheduleMode schedule = ScheduleMode::naive;
    ggml_backend_t prefetch_backend = nullptr;
    std::unique_ptr<GgmlPrefetch> device;
    std::string selection_backend;
    bool configure_ranker = false;
    std::string prefill_backend;
    bool configure_prefill = false;
    std::uint32_t prefill_queries=128, prefill_tile_tokens=512;
};
AttentionAdapter::AttentionAdapter(const llama_model * model, const AttentionConfig & options,
                                  std::uint32_t capacity, std::uint64_t prefill_tokens, const std::filesystem::path & trace, std::uint32_t prefill_queries) :
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
        if (options.prefetch && model->dev_layer(layer) != model->dev_layer(0)) throw std::invalid_argument("prefetch requires all attention layers on one backend");
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
    impl_->selection_backend = options.selection_backend;
    impl_->configure_ranker = options.sparse && options.gpu_decode && options.selection_replay.empty() && options.selection_backend != "cpu";
    impl_->prefill_backend=options.prefill_backend;
    impl_->configure_prefill=options.gpu_decode && options.prefill_backend!="cpu";
    impl_->prefill_queries=std::min(prefill_queries,capacity);
    impl_->prefill_tile_tokens=options.prefill_tile_tokens;
    if (options.prefetch) impl_->placement=std::make_unique<WorkerPlacement>(options.io_cpu,options.coordinator_cpu);
    const auto slots=options.prefetch_slots_per_head ? options.prefetch_slots_per_head : std::min<std::uint64_t>(geometry.blocks_per_head(),
        (static_cast<std::uint64_t>(budget.total)+geometry.block_tokens-1)/geometry.block_tokens+3);
    const auto admission=std::max<std::uint64_t>(options.prefetch_queue_depth,static_cast<std::uint64_t>(geometry.heads)*slots*2);
    if (options.prefetch && admission>16384) throw std::invalid_argument("prefetch worker admission exceeds 16384 requests");
    impl_->session = std::make_unique<AttentionSession>(std::move(store), hp.n_head(), budget, options.sparse, prefill_tokens,
        options.selection_trace ? trace : std::filesystem::path{},
        options.selection_replay, options.prefetch ? make_threaded_uring_reader(options.prefetch_queue_depth,impl_->placement->io_cpu(),static_cast<std::uint32_t>(admission)) : nullptr,
        options.prefetch_slots_per_head, options.prefetch_trace ? trace.parent_path() / "prefetch.jsonl" : std::filesystem::path{});
    if (options.prefetch_trace) impl_->residency_trace = trace.parent_path() / "residency.jsonl";
    impl_->schedule = parse_schedule_mode(options.schedule);
    if (options.schedule_trace) impl_->schedule_trace = trace.parent_path() / "schedule.jsonl";
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (!registry.emplace(model, this).second) throw std::invalid_argument("one attention session per model is supported");
}
AttentionAdapter::~AttentionAdapter() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.erase(impl_->model);
}
void AttentionAdapter::check() const { impl_->failure->check(); }
void AttentionAdapter::after_process() {
    try { if (impl_->device) impl_->device->process_complete(); } catch (...) { impl_->failure->capture(std::current_exception()); }
    check();
    impl_->session->release_prefill_if_complete();
}
void AttentionAdapter::flush() { check(); impl_->session->flush(); }
void AttentionAdapter::finish() { check(); quiesce(); impl_->session->finish(); }
SessionStats AttentionAdapter::stats() const { check(); return impl_->session->stats(); }
SparseBudget AttentionAdapter::budget() const { return impl_->session->budget(); }
std::unique_ptr<AttentionNode> AttentionAdapter::node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale,
                                                    bool gathered, ggml_backend_t backend) {
    if (impl_->configure_prefill) {
        bool configured=false;
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (backend && ggml_backend_is_cuda(backend)) {
            auto * registry=ggml_backend_cuda_reg();
            for (std::size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(registry);++ordinal) {
                if (ggml_backend_reg_dev_get(registry,ordinal)!=ggml_backend_get_device(backend)) continue;
                impl_->session->set_prefill(make_cuda_tiled_prefill(impl_->session->geometry(),impl_->session->query_heads(),
                    impl_->prefill_queries,impl_->prefill_tile_tokens,static_cast<int>(ordinal)));
                configured=true; break;
            }
        }
#endif
        if (!configured && impl_->prefill_backend=="cuda") throw std::runtime_error("CUDA prefill requires a CUDA build and CUDA attention backend");
        impl_->configure_prefill=false;
    }
    if (impl_->configure_ranker) {
        bool configured = false;
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (backend && ggml_backend_is_cuda(backend)) {
            auto * registry = ggml_backend_cuda_reg();
            for (std::size_t ordinal = 0; ordinal < ggml_backend_reg_dev_count(registry); ++ordinal) {
                if (ggml_backend_reg_dev_get(registry, ordinal) != ggml_backend_get_device(backend)) continue;
                impl_->session->set_ranker(make_cuda_block_ranker(impl_->session->geometry(), impl_->session->query_heads(),
                    impl_->session->budget().representatives, impl_->session->budget().total, static_cast<int>(ordinal), impl_->session->budget().local));
                configured = true;
                break;
            }
        }
#endif
        if (!configured && impl_->selection_backend == "cuda") throw std::runtime_error("CUDA selection requires a CUDA build and CUDA attention backend");
        impl_->configure_ranker = false;
    }
    if (gathered && prefetch_enabled()) {
        if (!backend) throw std::logic_error("prefetch decode backend is required");
        if (!impl_->device) {
            impl_->device = std::make_unique<GgmlPrefetch>(*impl_->session, backend, impl_->residency_trace, impl_->schedule, impl_->schedule_trace,
                impl_->placement->coordinator_cpu());
            impl_->prefetch_backend = backend;
        } else if (impl_->prefetch_backend != backend) throw std::logic_error("prefetch decode backend changed");
    }
    return std::make_unique<AttentionNode>(*impl_->session, std::move(frame), layer, scale, impl_->failure, gathered,
                                          gathered ? impl_->device.get() : nullptr);
}
bool AttentionAdapter::prefetch_enabled() const { return impl_->session->prefetch_enabled(); }
void AttentionAdapter::begin_prefetch(std::uint32_t layer, std::uint64_t position) {
    check();
    if (impl_->device) impl_->device->begin_prefetch(layer, position);
    else impl_->session->begin_prefetch(layer, position);
}
bool AttentionAdapter::observe(ggml_tensor * tensor, bool ask) noexcept {
    const bool normal = tensor->op == GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor), "solid_prefetched_flash_attention") == 0;
    const bool reused = tensor->op == GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor), "solid_prefetched_flash_attention_reuse") == 0;
    const bool ours = normal || reused;
    if (ask) return normal;
    if (!ours) return true;
    try { if (impl_->device) impl_->device->completed(impl_->device->attention_epoch()); }
    catch (...) { impl_->failure->capture(std::current_exception()); return false; }
    return true;
}
void AttentionAdapter::quiesce() { if (impl_->device) impl_->device->quiesce(); }
DeviceCacheStats AttentionAdapter::device_stats() const { return impl_->device ? impl_->device->stats() : DeviceCacheStats{}; }
ScheduleStats AttentionAdapter::schedule_stats() const { return impl_->device ? impl_->device->schedule_stats() : ScheduleStats{}; }
std::vector<int> AttentionAdapter::inference_cpus() const { return impl_->placement ? impl_->placement->inference_cpus() : std::vector<int>{}; }
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
        if (batch->n_tokens == 1) adapter.begin_prefetch(0, frame->positions.front());
    }
    bool can_reuse(const llm_graph_params &) override { return false; }
    AttentionAdapter & adapter;
    std::shared_ptr<AttentionFrame> frame;
    std::vector<std::unique_ptr<AttentionNode>> nodes;
    std::unordered_map<int, ggml_tensor *> projected_kv;
};
bool adapter_active(const llama_model * model) { return lookup(model) != nullptr; }
bool graph_depends_on(const ggml_tensor * tensor, const ggml_tensor * ancestor) {
    if (!tensor || !ancestor) return false;
    std::vector<const ggml_tensor *> pending{tensor};
    std::unordered_set<const ggml_tensor *> visited;
    while (!pending.empty()) {
        const auto * current = pending.back(); pending.pop_back();
        if (!current || !visited.insert(current).second) continue;
        if (current == ancestor) return true;
        for (auto * source : current->src) if (source) pending.push_back(source);
        if (current->view_src) pending.push_back(current->view_src);
    }
    return false;
}
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
    ggml_tensor * packed = nullptr;
    if (custom->adapter.fused_projection()) {
        packed = KvProjection::update_key(graph.ctx0, custom->projected_kv.at(layer), key);
    }
    auto * device = graph.backend_cpu;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(graph.sched); ++i) {
        auto * candidate = ggml_backend_sched_get_backend(graph.sched, i);
        if (ggml_backend_get_device(candidate) == graph.model.dev_layer(layer)) { device = candidate; break; }
    }
    auto node = custom->adapter.node(custom->frame, static_cast<std::uint32_t>(layer), scale, gathered, device);
    auto * result = node->build(graph.ctx0, query, key, value, packed, graph.sched, device);
    if (node->selection_node()) {
        if (!custom->nodes.empty() && custom->nodes.back()->attention_node()) {
            const auto * previous = custom->nodes.back()->attention_node();
            const bool covers = graph_depends_on(query, previous);
            node->set_query_proof(covers);
            if (covers) custom->nodes.back()->defer_completion();
        }
        // Query selection is an explicit graph frontier before traversing KV.
        ggml_backend_sched_set_tensor_backend(graph.sched, node->selection_node(), graph.backend_cpu);
        ggml_build_forward_expand(graph.gf, node->selection_node());
        if (layer + 1 == static_cast<int>(graph.hparams.n_layer())) node->defer_completion();
    }
    ggml_backend_sched_set_tensor_backend(graph.sched, node->host_node(), graph.backend_cpu);
    custom->nodes.push_back(std::move(node));
    return result;
}
} // namespace solid
