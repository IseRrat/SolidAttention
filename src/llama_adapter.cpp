#include "solid/llama_adapter.hpp"
#include "solid/projection.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/device_ops.hpp"
#include "solid/device_timing.hpp"
#include "solid/trace.hpp"
#include "solid/dense_offload.hpp"
#include "solid/thread_affinity.hpp"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef SOLID_HAVE_CUDA_SELECTION
#include "ggml-cuda.h"
#endif
#ifdef SOLID_HAVE_SYCL
#include "ggml-sycl.h"
#include "solid/sycl_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
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
ggml_backend_t layer_backend(const llm_graph_context & graph,int layer) {
    const auto * device=graph.model.dev_layer(layer);
    for (int i=0;i<ggml_backend_sched_get_n_backends(graph.sched);++i) {
        auto * backend=ggml_backend_sched_get_backend(graph.sched,i);
        if (ggml_backend_get_device(backend)==device) return backend;
    }
    return graph.backend_cpu;
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
class InfllmSelectedCache {
public:
    InfllmSelectedCache(AttentionSession & session, ggml_backend_t backend) : session_(session), backend_(backend), copy_backend_(backend) {
        const auto & geometry = session.geometry();
        const auto budget = session.budget();
        const auto block = geometry.block_tokens;
        const auto block_count = [block](std::uint64_t tokens) { return (tokens + block - 1) / block; };
        const auto history = budget.total - budget.initial - budget.local;
        const auto groups = block_count(history) + block_count(budget.initial) +
            block_count(budget.local) + 2;
        const auto blocks = std::min(groups, geometry.blocks_per_head() + 1);
        if (blocks * block > INT32_MAX) throw std::overflow_error("InfLLM cache row index");
        rows_ = static_cast<std::uint32_t>(blocks * block);
        width_ = static_cast<std::size_t>(geometry.heads) * geometry.dimension;
        entries_.resize(blocks);
        context_.reset(ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("InfLLM cache metadata allocation failed");
        bank_ = ggml_new_tensor_4d(context_.get(), GGML_TYPE_F16, geometry.dimension, rows_, geometry.heads, 2);
        ggml_set_name(bank_, "infllm_selected_kv_cache");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend));
        if (!buffer_) throw std::runtime_error("InfLLM cache allocation failed");
        ggml_backend_buffer_clear(buffer_.get(), 0);
        stats_.resident_bytes = ggml_backend_buffer_get_size(buffer_.get());
        auto * device = ggml_backend_get_device(backend);
        auto * host_type = device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (!host_type || !ggml_backend_buft_is_host(host_type)) host_type = ggml_backend_cpu_buffer_type();
        staging_buffer_.reset(ggml_backend_buft_alloc_buffer(host_type, ggml_nbytes(bank_)));
        if (!staging_buffer_) throw std::runtime_error("InfLLM read buffer allocation failed");
        staging_ = static_cast<Half *>(ggml_backend_buffer_get_base(staging_buffer_.get()));
        stats_.staging_bytes = ggml_backend_buffer_get_size(staging_buffer_.get());
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (ggml_backend_is_cuda(backend_)) {
            // InfLLM's GLOBAL_STREAM submits copies independently of model
            // compute. A separate GGML backend owns that stream/context;
            // the model backend remains confined to the inference thread.
            copy_owner_.reset(ggml_backend_dev_init(device, nullptr));
            if (!copy_owner_) throw std::runtime_error("InfLLM copy backend initialization failed");
            copy_backend_ = copy_owner_.get();
            trace::name_cuda_stream(solid_cuda_queue(copy_backend_), "sparse.kv.H2D");
        }
#endif
    }
    ~InfllmSelectedCache() { synchronize_copy(); }
    ggml_backend_t backend() const { return backend_; }
    DeviceCacheStats stats() const { return stats_; }
    ggml_tensor * plane(ggml_context * context, bool values) const {
        return ggml_view_3d(context, bank_, bank_->ne[0], rows_, bank_->ne[2], bank_->nb[1], bank_->nb[2],
                            values ? bank_->nb[3] : 0);
    }
    void prefetch_fixed(std::uint32_t layer, std::uint64_t position, AsyncReader * reader, std::size_t depth) {
        begin_frame(layer, position);
        const auto budget = session_.budget();
        std::vector<std::uint64_t> positions;
        const auto initial = position + 1 <= budget.total ? position : budget.initial;
        const auto recent = position + 1 <= budget.total ? position : budget.recent_start(session_.geometry(), position);
        for (std::uint64_t p = 0; p < initial; ++p) positions.push_back(p);
        for (auto p = recent; p < position; ++p) positions.push_back(p);
        prepared_ = plan(position, positions, false);
        read(layer, prepared_, reader, depth, 0, nullptr, nullptr, "SSD->CPU Prefetch");
    }
    void upload_fixed() {
        upload(prepared_, "CPU->GPU Prefetch");
        prepared_.clear();
    }
    void load(std::uint32_t layer, std::uint64_t position, const std::vector<std::uint64_t> & selected,
              Half * metadata, std::uint32_t slots, const float * key, const float * value,
              AsyncReader * reader = nullptr, std::size_t depth = 64) {
        if (selected.empty() || selected.size() > slots) throw std::logic_error("InfLLM cache selection size");
        begin_frame(layer, position);
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        const auto & geometry = session_.geometry();
        const auto budget = session_.budget();
        indices_.resize(slots);
        const auto loads = plan(position, selected, true);
        read(layer, loads, reader, depth, position, key, value);
        upload(loads);
        std::fill(indices_.begin() + selected.size(), indices_.end(), indices_.front());
        std::memcpy(metadata, indices_.data(), static_cast<std::size_t>(slots) * sizeof(std::int32_t));
        auto * mask = metadata + 2 * slots;
        for (std::uint32_t head = 0; head < session_.query_heads(); ++head) {
            std::fill_n(mask + head * slots, selected.size(), Half{0});
            std::fill_n(mask + head * slots + selected.size(), slots - selected.size(), Half{0xfc00});
        }
        auto * markers = mask + static_cast<std::size_t>(slots) * session_.query_heads();
        for (std::uint32_t head = 0; head < geometry.heads; ++head) {
            auto * row = markers + head * slots;
            std::fill_n(row, slots, Half{0});
            for (std::size_t i = 0; i < selected.size(); ++i) {
                const auto distance = position - selected[i];
                if (distance < budget.local) row[i] = to_half(static_cast<float>(budget.local - distance));
            }
        }
        session_.account_cached_gather(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }
private:
    struct Entry { std::uint64_t first = 0, last = 0; std::uint32_t valid = 0; bool used = false; };
    struct Load { std::uint64_t first; std::uint32_t row, count; };
    std::optional<std::pair<std::uint32_t, std::uint64_t>> frame_;
    void begin_frame(std::uint32_t layer, std::uint64_t position) {
        const auto frame = std::make_pair(layer, position);
        if (frame_ == frame) return;
        if (!prepared_.empty()) throw std::logic_error("sparse workspace still holds a pending upload");
        std::fill(entries_.begin(), entries_.end(), Entry{});
        frame_ = frame;
    }
    std::vector<Load> plan(std::uint64_t position, const std::vector<std::uint64_t> & selected, bool map_indices) {
        const auto & geometry = session_.geometry();
        const auto budget = session_.budget();
        const auto block = geometry.block_tokens;
        const auto block_start = [&](std::uint64_t token) {
            const std::uint64_t origin = token < budget.initial ? 0 : budget.initial;
            return origin + (token - origin) / block * block;
        };
        for (auto & entry : entries_) {
            const auto it = std::lower_bound(selected.begin(), selected.end(), entry.first);
            if (entry.used && it != selected.end() && block_start(*it) == entry.first) entry.last = position;
        }
        std::vector<Load> loads;
        for (std::size_t i = 0; i < selected.size();) {
            const auto first = block_start(selected[i]);
            auto end = i + 1;
            while (end < selected.size() && block_start(selected[end]) == first) ++end;
            auto entry = std::find_if(entries_.begin(), entries_.end(), [&](const auto & e) { return e.used && e.first == first; });
            if (entry == entries_.end()) {
                entry = std::min_element(entries_.begin(), entries_.end(), [](const auto & a, const auto & b) {
                    return a.used != b.used ? !a.used : a.last < b.last;
                });
                if (entry->used && entry->last == position) throw std::logic_error("InfLLM cache cannot fit selected blocks");
                if (entry->used) ++stats_.overwritten_blocks;
                *entry = {first, position, 0, true};
            }
            const auto row = static_cast<std::uint32_t>(entry - entries_.begin()) * block;
            const auto required = static_cast<std::uint32_t>(selected[end - 1] - first + 1);
            if (entry->valid >= required) {
                if (map_indices) ++stats_.reused_blocks;
            }
            else {
                const Load missing{first + entry->valid, row + entry->valid, required - entry->valid};
                if (!loads.empty() && loads.back().first + loads.back().count == missing.first &&
                    loads.back().row + loads.back().count == missing.row) loads.back().count += missing.count;
                else loads.push_back(missing);
                entry->valid = required;
            }
            if (map_indices) for (; i < end; ++i) indices_[i] = static_cast<std::int32_t>(row + selected[i] - first);
            i = end;
        }
        return loads;
    }
    void read(std::uint32_t layer, const std::vector<Load> & loads, AsyncReader * reader, std::size_t depth,
              std::uint64_t position = 0, const float * key = nullptr, const float * value = nullptr, const char * phase_name = "SSD->CPU Load") {
        trace::Scope phase;
        if (!loads.empty()) phase.start(phase_name, layer, frame_->second);
        const auto & geometry = session_.geometry();
        const auto dim = geometry.dimension;
        std::vector<KvRowRange> ranges;
        for (const auto & load : loads) {
            const bool has_current = key && load.first + load.count - 1 == position;
            const auto history = load.count - static_cast<std::uint32_t>(has_current);
            for (std::uint32_t h = 0; h < geometry.heads; ++h) {
                auto * keys = staging_ + static_cast<std::size_t>(h) * rows_ * dim + load.row * dim;
                auto * values = keys + static_cast<std::size_t>(geometry.heads) * rows_ * dim;
                if (history) ranges.push_back({h, load.first, history, keys, values});
                if (has_current) for (std::uint32_t d = 0; d < dim; ++d) {
                    const auto source = static_cast<std::size_t>(h) * dim + d;
                    const auto target = static_cast<std::size_t>(history) * dim + d;
                    keys[target] = to_half(key[source]);
                    values[target] = to_half(value[source]);
                }
            }
        }
        session_.read_decode_rows(layer, ranges, reader, depth);
    }
    void upload(const std::vector<Load> & loads, const char * phase_name = "CPU->GPU Load") {
        const auto & geometry = session_.geometry();
        const auto dim = geometry.dimension;
        if (loads.empty()) return;
        trace::Scope phase(phase_name, frame_->first, frame_->second);
        auto ranges = loads;
        std::sort(ranges.begin(), ranges.end(), [](const auto & a, const auto & b) { return a.row < b.row; });
        std::vector<Load> transfers;
        for (const auto & range : ranges) {
            if (!transfers.empty() && transfers.back().row + transfers.back().count == range.row) {
                transfers.back().count += range.count;
            } else transfers.push_back(range);
        }
        try {
#ifdef SOLID_HAVE_CUDA_SELECTION
            // The queue accessor also selects this backend's CUDA device on
            // the calling worker thread before submitting memory operations.
            if (copy_owner_) (void) solid_cuda_queue(copy_backend_);
#endif
            for (const auto & range : transfers) {
                const auto bytes = static_cast<std::size_t>(range.count) * dim * sizeof(Half);
                for (int plane = 0; plane < 2; ++plane) for (std::uint32_t h = 0; h < geometry.heads; ++h) {
                    const auto offset = plane * bank_->nb[3] + h * bank_->nb[2] + range.row * bank_->nb[1];
                    ggml_backend_tensor_set_async(copy_backend_, bank_, reinterpret_cast<const char *>(staging_) + offset, offset, bytes);
                    ++stats_.uploads;
                    stats_.upload_bytes += bytes;
                }
            }
            // One receipt protects the read buffer for reuse by the next SSD job.
            synchronize_copy();
        } catch (...) {
            synchronize_copy();
            throw;
        }
    }
    void synchronize_copy() {
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (copy_owner_) (void) solid_cuda_queue(copy_backend_);
#endif
        ggml_backend_synchronize(copy_backend_);
    }
    AttentionSession & session_;
    ggml_backend_t backend_;
    ggml_backend_t copy_backend_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> copy_owner_{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor * bank_ = nullptr;
    std::size_t width_ = 0;
    std::uint32_t rows_ = 0;
    std::vector<Entry> entries_;
    std::vector<std::int32_t> indices_;
    std::vector<Load> prepared_;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> staging_buffer_{nullptr, ggml_backend_buffer_free};
    Half * staging_ = nullptr;
    DeviceCacheStats stats_;
};
std::shared_ptr<InfllmSelectedCache> make_infllm_selected_cache(AttentionSession & session, ggml_backend * backend) {
    if (!backend || !session.relative_rope() || session.prefetch_enabled() || session.geometry().layout != KvLayout::separate) {
        throw std::invalid_argument("InfLLM selected cache requires separate K/V and a decode backend");
    }
    return std::make_shared<InfllmSelectedCache>(session,backend);
}
SparseBaselineLookahead::SparseBaselineLookahead(AttentionSession & session, InfllmSelectedCache & workspace, bool use_uring) :
    session_(session), workspace_(workspace), reader_(use_uring ? make_uring_reader(queue_depth) : nullptr) {}
void SparseBaselineLookahead::begin(std::uint32_t layer, std::uint64_t position) {
    if (frame_ || layer >= session_.geometry().layers) throw std::logic_error("invalid sparse prefetch frontier");
    worker_.submit([this, layer, position] {
        workspace_.prefetch_fixed(layer, position, reader_.get(), queue_depth);
        workspace_.upload_fixed();
    });
    frame_ = std::make_pair(layer, position);
}
void SparseBaselineLookahead::upload(std::uint32_t layer, std::uint64_t position) {
    if (frame_ != std::make_optional(std::make_pair(layer, position))) throw std::logic_error("sparse prefetch layer or position mismatch");
    const auto start = std::chrono::steady_clock::now();
    worker_.join();
    session_.account_sparse_prefetch(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    frame_.reset();
}
void SparseBaselineLookahead::completed(std::uint32_t layer, std::uint64_t position) {
    if (layer + 1 < session_.geometry().layers) begin(layer + 1, position);
}
void SparseBaselineLookahead::start_writeback(std::uint32_t layer) {
    if (session_.layer_write_ready(layer)) worker_.submit([this, layer] { session_.flush_ready_layer(layer); });
}
void SparseBaselineLookahead::finish_decode() const {
    if (frame_) throw std::logic_error("sparse decode left a layer in flight");
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
    ProjectionIntervals * timing = nullptr;
    InfllmSelectedCache * selected_cache = nullptr;
    DenseOffload * dense = nullptr;
    SparseBaselineLookahead * sparse_lookahead = nullptr;
    trace::Range current_store;
    bool packed_inputs = false;
    std::optional<DeviceMemory> direct = {};
    std::optional<DeviceMemory> selected_device = {};
    void * probability_stream = nullptr;
    ggml_tensor * device_query = nullptr;
    ggml_tensor * device_kv = nullptr;
    ggml_tensor * device_scores = nullptr;
    std::unique_ptr<PrefillOutput> prefill_output = {};
    ggml_tensor * host = nullptr;
    ggml_tensor * metadata = nullptr;
    ggml_tensor * local_logits = nullptr;
    std::int64_t local_source_rows = 0;
    ggml_tensor * early = nullptr;
    ggml_tensor * flash = nullptr;
    bool query_proof = false;
    std::uint64_t epoch = 0;
    static void publish_completion(ggml_tensor * destination,int ith,int,void * userdata) {
        if (ith) return;
        std::memcpy(destination->data,destination->src[0]->data,ggml_nbytes(destination));
        auto * token=static_cast<StreamCompletionToken *>(userdata); token->notify(token->userdata);
    }
    static void cache_indices(ggml_tensor * destination, int ith, int, void *) {
        if (ith) return;
        const auto bytes = static_cast<std::size_t>(destination->ne[0]) * sizeof(std::int32_t);
        for (std::int64_t h = 0; h < destination->ne[1]; ++h) {
            std::memcpy(static_cast<char *>(destination->data) + h * bytes, destination->src[0]->data, bytes);
        }
    }
    static void dense_history(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        auto & index = *static_cast<std::int32_t *>(destination->data);
        index = 0;
        try {
            self.failure->check();
            const auto position = self.frame->positions.at(0);
            self.dense->upload_history(self.layer, position);
            self.current_store = trace::Range("GPU->CPU Store", self.layer, position);
            index = static_cast<std::int32_t>(position);
        } catch (...) { self.failure->capture(std::current_exception(), self.layer); }
    }
    static void sparse_history(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        *static_cast<std::int32_t *>(destination->data) = 0;
        try {
            self.failure->check();
            self.sparse_lookahead->upload(self.layer, self.frame->positions.at(0));
            self.current_store = trace::Range("GPU->CPU Store", self.layer, self.frame->positions.at(0));
        } catch (...) { self.failure->capture(std::current_exception(), self.layer); }
    }
    static void dense_archive(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        self.current_store.finish();
        auto * mask = static_cast<Half *>(destination->data);
        const auto slots = static_cast<std::size_t>(ggml_nelements(destination));
        std::fill_n(mask, slots, Half{0xfc00});
        mask[0] = 0;
        try {
            self.failure->check();
            const auto position = self.frame->positions.at(0);
            self.dense->archive(self.layer, position, static_cast<const float *>(destination->src[1]->data),
                               static_cast<const float *>(destination->src[2]->data));
            std::fill_n(mask, position + 1, Half{0});
        } catch (...) { self.failure->capture(std::current_exception(), self.layer); }
    }
    static void receive_attention(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        try {
            self.failure->check();
            const auto position = self.frame->positions.at(0);
            if (self.session.device_probabilities()) {
                if (!self.device_kv || !self.device_scores) throw std::logic_error("device probability inputs missing");
                if (self.prefetch) self.prefetch->completed(self.epoch);
                if (self.sparse_lookahead) self.sparse_lookahead->completed(self.layer, position);
                auto * overlap_stream = self.prefetch ? self.prefetch->probability_compute_stream() : nullptr;
                auto * probability_stream = overlap_stream ? overlap_stream : self.probability_stream;
                self.session.observe_device_probabilities(self.layer, position,
                    {static_cast<const Half *>(self.device_kv->data), static_cast<std::size_t>(ggml_nelements(self.device_kv)), DeviceMemory::cuda},
                    {static_cast<const float *>(self.device_scores->data), static_cast<std::size_t>(ggml_nelements(self.device_scores)), DeviceMemory::cuda},
                    probability_stream, !overlap_stream && probability_stream);
                *static_cast<float *>(destination->data) = 0;
                return;
            }
            const auto local = self.session.budget().local;
            const auto recent = std::min<std::uint64_t>(position + 1, local);
            const auto * probability = destination->src[0];
            const auto * p = static_cast<const float *>(probability->data);
            const auto heads = self.session.query_heads();
            std::vector<float> scores(static_cast<std::size_t>(heads) * recent);
            for (std::uint32_t q = 0; q < heads; ++q) {
                std::copy_n(p + q * local + local - recent, recent, scores.data() + q * recent);
            }
            if (self.prefetch) self.prefetch->completed(self.epoch);
            if (self.sparse_lookahead) self.sparse_lookahead->completed(self.layer, position);
            self.session.observe_probabilities(self.layer, position, scores);
            *static_cast<float *>(destination->data) = 0;
        } catch (...) {
            self.failure->capture(std::current_exception(), self.layer, self.frame->positions.empty() ? 0 : self.frame->positions.front());
            std::fill_n(static_cast<float *>(destination->data), ggml_nelements(destination), 0.f);
        }
    }
    static void staged_metadata(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        auto * physical = static_cast<std::int32_t *>(destination->data);
        std::fill_n(physical, ggml_nelements(destination), 0);
        try {
            self.failure->check();
            const auto budget = self.session.budget();
            const auto total = budget.total, local = budget.local;
            const auto heads = self.session.geometry().heads;
            const auto group = self.session.query_heads() / heads;
            const auto recent = std::min<std::uint64_t>(self.frame->positions.at(0) + 1, local);
            const auto slots = ggml_nelements(destination->src[0]) / heads;
            const auto * markers = static_cast<const Half *>(destination->src[0]->data);
            const auto * host = static_cast<const Half *>(destination->src[1]->data);
            const auto * source_indices = self.selected_cache ? reinterpret_cast<const std::int32_t *>(host) : nullptr;
            const auto * mask = host + (self.selected_cache ? 2 * slots : 0);
            auto * logical = physical + total * heads;
            for (std::uint32_t h = 0; h < heads; ++h) {
                auto * p = physical + h * total;
                auto * l = logical + h * total;
                std::fill_n(p, total, -1);
                std::int32_t padding = -1;
                std::uint32_t globals = 0;
                for (std::int64_t i = 0; i < slots; ++i) {
                    if (from_half(mask[h * group * slots + i]) != 0.f) { if (padding < 0) padding = i; continue; }
                    const auto marker = static_cast<std::uint32_t>(from_half(markers[h * slots + i]));
                    std::uint32_t target;
                    if (marker) {
                        if (marker <= local - recent || marker > local || p[marker - 1] != -1) throw std::logic_error("invalid or duplicate local slot");
                        target = marker - 1;
                    } else {
                        if (globals == total - local) throw std::logic_error("historical selection exceeds its budget");
                        target = local + globals++;
                    }
                    const auto row = source_indices ? source_indices[i] : static_cast<std::int32_t>(i);
                    if (row < 0 || row >= self.local_source_rows) throw std::logic_error("invalid staged KV row");
                    p[target] = row;
                    l[target] = i;
                }
                for (std::uint32_t i = local - recent; i < local; ++i) if (p[i] < 0) throw std::logic_error("missing local KV row");
                for (std::uint32_t i = 0; i < total; ++i) if (p[i] < 0) {
                    if (padding < 0) throw std::logic_error("missing invalid slot for staged padding");
                    p[i] = source_indices ? source_indices[padding] : padding;
                    l[i] = padding;
                }
            }
        } catch (...) {
            self.failure->capture(std::current_exception(), self.layer);
            std::fill_n(physical, ggml_nelements(destination), 0);
        }
    }
    static void decode_metadata(ggml_tensor * destination, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Impl *>(userdata);
        auto * positions = static_cast<std::int32_t *>(destination->data);
        std::fill_n(positions, ggml_nelements(destination), 0);
        try {
            self.failure->check();
            const auto local = self.session.budget().local;
            const auto recent = std::min<std::uint64_t>(self.frame->positions.at(0) + 1, local);
            const auto heads = self.session.query_heads();
            const auto group = heads / self.session.geometry().heads;
            const auto slots = ggml_nelements(destination->src[0]) / self.session.geometry().heads;
            const auto * markers = static_cast<const Half *>(destination->src[0]->data);
            const auto position_count = slots * self.session.geometry().heads;
            if (slots * heads > INT32_MAX || ggml_nelements(destination) != position_count + local * heads) {
                throw std::logic_error("InfLLM decode metadata shape");
            }
            for (std::int64_t i = 0; i < position_count; ++i) {
                positions[i] = std::max(0, static_cast<int>(from_half(markers[i])) - 1);
            }
            auto * indices = positions + position_count;
            for (std::uint32_t kv_head = 0; kv_head < self.session.geometry().heads; ++kv_head) {
                const auto q = kv_head * group;
                auto * row = indices + q * local;
                std::fill_n(row, local, -1);
                for (std::int64_t i = 0; i < slots; ++i) {
                    const auto marker = static_cast<std::uint32_t>(from_half(markers[kv_head * slots + i]));
                    if (!marker) continue;
                    if (marker <= local - recent || marker > local || row[marker - 1] != -1) {
                        throw std::logic_error("invalid or duplicate InfLLM recent slot");
                    }
                    row[marker - 1] = static_cast<std::int32_t>(q * slots + i);
                }
                for (std::uint32_t i = local - recent; i < local; ++i) {
                    if (row[i] < 0) throw std::logic_error("missing InfLLM recent slot");
                }
                std::fill_n(row, local - recent, 0);
                for (std::uint32_t head = q + 1; head < q + group; ++head) {
                    auto * target = indices + head * local;
                    std::fill_n(target, local - recent, 0);
                    for (std::uint32_t i = local - recent; i < local; ++i) {
                        target[i] = row[i] + static_cast<std::int32_t>((head - q) * slots);
                    }
                }
            }
        } catch (...) {
            self.failure->capture(std::current_exception(), self.layer);
            std::fill_n(positions, ggml_nelements(destination), 0);
        }
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
                self.prefetch->select_early_device_async(self.layer, self.frame->positions.front(),
                    {static_cast<const float *>(self.device_query->data),static_cast<std::size_t>(ggml_nelements(self.device_query)),*self.direct},
                    self.query_proof,self.timing && self.timing->query ? self.timing->query->duration() : std::nullopt);
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
        self.current_store.finish();
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
                if (self.selected_device && source == 0) continue;
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
                    if (!self.device_kv || !self.device_kv->data) throw std::logic_error("device KV allocation missing");
                    const DeviceKeys keys{static_cast<const Half *>(self.device_kv->data),static_cast<std::size_t>(ggml_nelements(self.device_kv)),
                        static_cast<std::size_t>(self.device_kv->ne[0]),*self.direct};
                    if (early) {
                        if (source->type!=GGML_TYPE_F32 || ggml_nelements(source)!=1) throw std::logic_error("device KV ready token missing");
                        result=self.prefetch->decode_device(self.layer,self.frame->positions.front(),keys,self.device_kv,
                            self.timing && self.timing->kv ? self.timing->kv->duration() : std::nullopt);
                    } else {
                        if (source->type!=GGML_TYPE_F16 || !ggml_is_contiguous(source) || ggml_nelements(source)!=ggml_nelements(self.device_kv)) {
                            throw std::logic_error("serial KV offload shape differs from device KV");
                        }
                        std::vector<Half> kv(static_cast<std::size_t>(ggml_nelements(source)));
                        std::memcpy(kv.data(),source->data,kv.size()*sizeof(Half));
                        const DeviceQuery query{static_cast<const float *>(self.device_query->data),
                            static_cast<std::size_t>(ggml_nelements(self.device_query)),*self.direct};
                        result=self.prefetch->decode_half(self.layer,self.frame->positions.front(),kv,keys,query,self.device_kv);
                    }
                } else result = early ? self.prefetch->decode(self.layer, self.frame->positions.front(), {}, inputs[0], self.packed_inputs ? std::vector<float>{} : inputs[1], self.packed_inputs) :
                    self.prefetch->decode(self.layer, self.frame->positions.front(), inputs[0], inputs[1], inputs[2], self.packed_inputs);
                self.epoch = self.prefetch->attention_epoch();
                if (result.size() != static_cast<std::size_t>(ggml_nelements(destination))) throw std::logic_error("prefetch mask shape");
                std::memcpy(destination->data, result.data(), result.size() * sizeof(Half));
            } else if (self.selected_cache) {
                const auto position = self.frame->positions.front();
                const auto after_append = [&] {
                    if (self.sparse_lookahead) self.sparse_lookahead->start_writeback(self.layer);
                };
                const auto selected = self.selected_device ?
                    self.session.select_decode_device(self.layer, position,
                        {static_cast<const float *>(self.device_query->data), static_cast<std::size_t>(ggml_nelements(self.device_query)),
                         *self.selected_device}, inputs[1], inputs[2], after_append) :
                    self.session.select_decode(self.layer, position, inputs[0], inputs[1], inputs[2], false, after_append);
                if (self.sparse_lookahead) self.sparse_lookahead->join();
                self.selected_cache->load(self.layer, position, selected.front().positions,
                                          static_cast<Half *>(destination->data), self.session.gather_slots(),
                                          inputs[1].data(), inputs[2].data(),
                                          self.sparse_lookahead ? self.sparse_lookahead->reader() : nullptr);
            } else if (self.gathered) {
                self.session.gather_into(self.layer,self.frame->positions.front(),inputs[0],inputs[1],inputs[2],
                    static_cast<Half *>(destination->data),static_cast<std::size_t>(ggml_nelements(destination)),self.packed_inputs);
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
                if (self.session.relative_rope()) {
                    std::fill_n(data + static_cast<std::size_t>(self.prefetch->tokens_per_head()) * self.session.query_heads(),
                                static_cast<std::size_t>(self.prefetch->tokens_per_head()) * self.session.geometry().heads, Half{0});
                }
            } else if (self.selected_cache) {
                auto * data = static_cast<Half *>(destination->data);
                const auto slots = self.session.gather_slots();
                std::fill_n(data, ggml_nelements(destination), Half{0});
                for (std::uint32_t head = 0; head < self.session.query_heads(); ++head) {
                    std::fill_n(data + (2 + head) * slots + 1, slots - 1, Half{0xfc00});
                }
            } else if (self.gathered) {
                auto * data = static_cast<Half *>(destination->data);
                std::fill_n(data, ggml_nelements(destination), Half{0});
                const auto slots = self.session.gather_slots();
                const auto mask = static_cast<std::size_t>(slots) * 2 * self.session.geometry().heads * self.session.geometry().dimension;
                std::fill_n(data + mask, static_cast<std::size_t>(slots) * self.session.query_heads(), Half{0xfc00});
                for (std::uint32_t head = 0; head < self.session.query_heads(); ++head) data[mask + head * slots] = 0;
            } else {
                if (self.prefill_output) ggml_backend_buffer_clear(self.prefill_output->buffer.get(),0);
                std::fill_n(static_cast<float *>(destination->data), ggml_nelements(destination), 0.0f);
            }
        }
    }
};
AttentionNode::AttentionNode(AttentionSession & session, std::shared_ptr<AttentionFrame> frame,
                            std::uint32_t layer, float scale, std::shared_ptr<AttentionFailure> failure, bool gathered, GgmlPrefetch * prefetch,
                            ProjectionIntervals * timing, InfllmSelectedCache * selected_cache, DenseOffload * dense,
                            SparseBaselineLookahead * sparse_lookahead) :
    impl_(std::make_unique<Impl>(Impl{session, std::move(frame), layer, scale,
                                    failure ? std::move(failure) : std::make_shared<AttentionFailure>(), gathered, prefetch, timing,
                                    selected_cache, dense, sparse_lookahead})) {
    if (prefetch && !gathered) throw std::invalid_argument("prefetch node requires decode attention");
}
AttentionNode::~AttentionNode() = default;
ggml_tensor * AttentionNode::build(ggml_context * context, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                                  ggml_tensor * packed_kv, ggml_backend_sched * scheduler, ggml_backend * decode_backend,
                                  ggml_cgraph * graph, ggml_tensor * absolute_positions) {
    const auto & g = impl_->session.geometry();
#ifdef SOLID_HAVE_CUDA_SELECTION
    if (impl_->session.device_probabilities() && decode_backend && ggml_backend_is_cuda(decode_backend)) {
        impl_->probability_stream = solid_cuda_queue(decode_backend);
    }
#endif
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
    if (impl_->direct && *impl_->direct != DeviceMemory::host_emulator && (!scheduler || !decode_backend)) {
        throw std::invalid_argument("device Q/K graph requires scheduler and decode backend");
    }
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
    if (impl_->dense) {
        if (!impl_->gathered || query->ne[2] != 1 || packed_kv || !graph) throw std::logic_error("invalid dense offload graph");
        const auto contiguous_f32 = [&](ggml_tensor * input) {
            if (input->type != GGML_TYPE_F32) input = on_device(ggml_cast(context, input, GGML_TYPE_F32));
            return ggml_is_contiguous(input) ? input : on_device(ggml_cont(context, input));
        };
        auto * archived_key = contiguous_f32(key);
        auto * archived_value = contiguous_f32(value);
        auto * current_key = ggml_permute(context, archived_key, 0, 2, 1, 3);
        auto * current_value = ggml_permute(context, archived_value, 0, 2, 1, 3);
        auto * history = ggml_custom_4d(context, GGML_TYPE_I32, 1, 1, 1, 1, nullptr, 0, Impl::dense_history, 1, impl_.get());
        ggml_set_name(history, "offload_dense_history_upload");
        if (scheduler) ggml_backend_sched_set_tensor_backend(scheduler, history,
            ggml_backend_sched_get_backend(scheduler, ggml_backend_sched_get_n_backends(scheduler) - 1));
        auto * index = on_device(ggml_dup(context, history));
        ggml_tensor * archive_inputs[]{index, archived_key, archived_value};
        const auto slots = impl_->session.gather_slots();
        auto * mask = ggml_custom_4d(context, GGML_TYPE_F16, slots, 1, 1, 1, archive_inputs, 3, Impl::dense_archive, 1, impl_.get());
        ggml_set_name(mask, "offload_dense_archive");
        impl_->host = mask;
        // The worker is reading while projections run. The uploaded index
        // splits history upload from the current K/V download to the archive.
        ggml_build_forward_expand(graph, query);
        ggml_build_forward_expand(graph, archived_key);
        ggml_build_forward_expand(graph, archived_value);
        ggml_build_forward_expand(graph, mask);
        auto * q = ggml_permute(context, contiguous_f32(query), 0, 2, 1, 3);
        auto * keys = on_device(ggml_set_rows(context, impl_->dense->plane(context, false), current_key, index));
        auto * values = on_device(ggml_set_rows(context, impl_->dense->plane(context, true), current_value, index));
        auto * flash = on_device(ggml_flash_attn_ext(context, q, keys, values, mask, impl_->scale, 0, 0));
        ggml_prec_set_acc(flash, GGML_PREC_F32);
        ggml_set_name(flash, "offload_dense_flash_attention");
        return ggml_reshape_2d(context, flash, g.dimension * impl_->session.query_heads(), 1);
    }
    ggml_tensor * inputs[] = {query, packed_kv ? packed_kv : key, packed_kv ? nullptr : value, nullptr};
    for (int i = 0; i < (impl_->direct ? 1 : (packed_kv ? 2 : 3)); ++i) {
        auto & input = inputs[i];
        if (input->type != GGML_TYPE_F32) input = ggml_cast(context, input, GGML_TYPE_F32);
        input = ggml_cont(context, input);
    }
    int input_count = packed_kv ? 2 : 3;
    if (impl_->selected_cache && impl_->session.direct_memory()) {
        const auto memory = impl_->session.direct_memory();
        bool matches = false;
#ifdef SOLID_HAVE_CUDA_SELECTION
        matches = *memory == DeviceMemory::cuda && decode_backend && ggml_backend_is_cuda(decode_backend);
#endif
#ifdef SOLID_HAVE_SYCL
        matches = matches || (*memory == DeviceMemory::sycl && decode_backend && ggml_backend_is_sycl(decode_backend));
#endif
        if (matches) {
            impl_->selected_device = memory;
            impl_->device_query = on_device(inputs[0]);
            inputs[0] = on_device(device_query_token(context, impl_->device_query));
            if (impl_->session.device_probabilities()) {
                impl_->device_kv = on_device(device_consolidate_kv(context, inputs[1], inputs[2], impl_->device_query, inputs[0]));
            }
        }
    }
    if (impl_->direct) {
        impl_->device_query = on_device(inputs[0]);
        auto * token = on_device(device_query_token(context,impl_->device_query));
        if (impl_->timing) bind_device_timing(token,impl_->timing->query.get());
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
        inputs[0] = impl_->prefetch && impl_->early ? on_device(device_kv_token(context,impl_->device_kv)) : impl_->device_kv;
        if (impl_->timing && impl_->early) bind_device_timing(inputs[0],impl_->timing->kv.get());
        input_count = 1;
    } else if (impl_->prefetch && impl_->prefetch->early_enabled()) {
        auto * q = inputs[0];
        impl_->early = ggml_custom_4d(context, GGML_TYPE_F32, 1, 1, 1, 1, &q, 1, Impl::select_query, 1, impl_.get());
        ggml_set_name(impl_->early, "solid_early_selection");
        if (packed_kv) { inputs[0] = inputs[1]; inputs[1] = impl_->early; input_count = 2; }
        else { inputs[0] = inputs[1]; inputs[1] = inputs[2]; inputs[2] = impl_->early; input_count = 3; }
    }
    if (impl_->sparse_lookahead) {
        if (!graph || !impl_->selected_cache || packed_kv) throw std::logic_error("invalid sparse baseline graph");
        auto * history = ggml_custom_4d(context, GGML_TYPE_I32, 1, 1, 1, 1, nullptr, 0, Impl::sparse_history, 1, impl_.get());
        ggml_set_name(history, "offload_sparse_fixed_upload");
        if (scheduler) ggml_backend_sched_set_tensor_backend(scheduler, history,
            ggml_backend_sched_get_backend(scheduler, ggml_backend_sched_get_n_backends(scheduler) - 1));
        inputs[input_count++] = on_device(ggml_dup(context, history));
        for (int i = 0; i < input_count; ++i) ggml_build_forward_expand(graph, inputs[i]);
    }
    if (impl_->gathered && absolute_positions && impl_->session.device_probabilities()) {
        const auto window = impl_->session.probability_key_window(impl_->layer);
        if (window.data && window.memory == DeviceMemory::cuda) {
            const auto local = impl_->session.budget().local;
            const auto & rope = *impl_->session.relative_rope();
            auto * local_keys = on_device(device_local_keys(context, impl_->device_kv, absolute_positions, window, local));
            local_keys = on_device(ggml_cont(context, ggml_permute(context, local_keys, 0, 2, 1, 3)));
            auto * relative = on_device(ggml_cast(context, on_device(ggml_arange(context, 0, local, 1)), GGML_TYPE_I32));
            local_keys = on_device(ggml_rope_ext(context, local_keys, relative, nullptr, rope.dimensions,
                rope.split_halves ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL, 0, rope.base, 1, 0, 1, 0, 0));
            local_keys = ggml_permute(context, local_keys, 0, 2, 1, 3);
            impl_->local_logits = on_device(ggml_mul_mat(context, local_keys, ggml_permute(context, impl_->device_query, 0, 2, 1, 3)));
            ggml_prec_set_acc(impl_->local_logits, GGML_PREC_F32);
            ggml_format_name(impl_->local_logits, "infllm.local.QK.L%u", impl_->layer);
            if (graph && !impl_->prefetch) ggml_build_forward_expand(graph, impl_->local_logits);
        }
    }
    if (impl_->gathered && query->ne[2] != 1) throw std::invalid_argument("gathered attention requires one decode token");
    const auto elements = impl_->selected_cache ? static_cast<std::uint64_t>(impl_->session.gather_slots()) *
        (2 + impl_->session.query_heads() + g.heads) : impl_->prefetch ? static_cast<std::uint64_t>(impl_->prefetch->tokens_per_head()) *
        (impl_->session.query_heads() + (impl_->session.relative_rope() ? g.heads : 0)) :
                                          impl_->session.gather_elements();
    auto * result = ggml_custom_4d(context, impl_->gathered ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   impl_->prefill_output?1:(impl_->gathered ? static_cast<int64_t>(elements) : query->ne[0] * query->ne[1]),
                                   impl_->prefill_output || impl_->gathered ? 1 : query->ne[2], 1, 1,
                                   inputs, input_count, Impl::compute, 1, impl_.get());
    ggml_set_name(result, "solid_sync_attention");
    impl_->host = result;
    const auto finish_infllm = [&](ggml_tensor * logits, ggml_tensor * output, ggml_tensor * compact, bool staged) {
        if (impl_->prefetch) {
            if (auto * token = impl_->prefetch->completion_token(impl_->layer)) {
                if (impl_->prefetch->completion_host_marker()) {
                    auto * source = output;
                    output = ggml_custom_4d(context, GGML_TYPE_F32, source->ne[0], source->ne[1], source->ne[2], source->ne[3],
                        &source, 1, Impl::publish_completion, 1, token);
                    ggml_set_name(output, "solid_infllm_completion_emulator");
                }
#ifdef SOLID_HAVE_CUDA_SELECTION
                else if (decode_backend && ggml_backend_is_cuda(decode_backend)) {
                    static_assert(stream_completion_offset + sizeof(StreamCompletionBinding) <= GGML_MAX_OP_PARAMS);
                    bind_stream_completion(logits->op_params, sizeof(logits->op_params), token, CompletionPhase::begin);
                    if (!staged) bind_stream_completion(output->op_params, sizeof(output->op_params), token, CompletionPhase::end);
                }
#endif
            }
        }
        if (impl_->session.device_probabilities()) {
            impl_->device_scores = compact;
            compact = on_device(device_query_token(context, compact));
        }
        output = on_device(ggml_cont(context, ggml_reshape_2d(context, output, g.dimension * impl_->session.query_heads(), 1)));
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (staged && impl_->prefetch && decode_backend && ggml_backend_is_cuda(decode_backend)) {
            if (auto * token = impl_->prefetch->completion_token(impl_->layer))
                bind_stream_completion(output->op_params, sizeof(output->op_params), token, CompletionPhase::end);
        }
#endif
        auto * ready = on_device(device_query_token(context, output,
            impl_->session.device_probabilities() ? impl_->device_kv : nullptr));
        ggml_tensor * receipt_inputs[]{compact, ready};
        auto * received = ggml_custom_4d(context, GGML_TYPE_F32, 1, 1, 1, 1,
            receipt_inputs, 2, Impl::receive_attention, 1, impl_.get());
        ggml_set_name(received, "solid_infllm_attention_receipt");
        return on_device(device_attention_result(context, output, received,
            impl_->session.device_probabilities() ? impl_->device_kv : nullptr, impl_->device_scores));
    };
    const auto infllm_attention = [&](ggml_tensor * q, ggml_tensor * keys, ggml_tensor * values,
                                      ggml_tensor * mask, std::size_t marker_offset) {
        const auto slots = mask->ne[0];
        const auto & rope = *impl_->session.relative_rope();
        auto * markers = ggml_view_1d(context, impl_->host, slots * g.heads, marker_offset);
        if (impl_->local_logits) {
            const auto local = impl_->session.budget().local;
            const auto total = impl_->session.budget().total;
            const auto global = total - local;
            const auto query_heads = impl_->session.query_heads();
            impl_->local_source_rows = keys->ne[1];
            ggml_tensor * sources[]{markers, impl_->host};
            impl_->metadata = ggml_custom_4d(context, GGML_TYPE_I32, total, g.heads, 2, 1,
                                            sources, 2, Impl::staged_metadata, 1, impl_.get());
            ggml_set_name(impl_->metadata, "infllm_staged_indices");
            auto * logical = ggml_view_3d(context, impl_->metadata, total, 1, g.heads,
                total * sizeof(std::int32_t), total * sizeof(std::int32_t), total * g.heads * sizeof(std::int32_t));
            logical = ggml_reshape_2d(context, logical, total, g.heads);
            auto * head_mask = ggml_view_3d(context, mask, 1, slots, g.heads,
                sizeof(Half), slots * (query_heads / g.heads) * sizeof(Half), 0);
            auto * joint_mask = on_device(ggml_get_rows(context, head_mask, logical));
            joint_mask = ggml_reshape_3d(context, joint_mask, total, 1, g.heads);
            joint_mask = on_device(ggml_repeat_4d(context, joint_mask, total, query_heads / g.heads, g.heads, 1));
            joint_mask = ggml_reshape_3d(context, joint_mask, total, 1, query_heads);
            const auto indices = [&](std::uint32_t count, std::uint32_t first) {
                return on_device(ggml_cont(context, ggml_view_2d(context, impl_->metadata, count, g.heads,
                    total * sizeof(std::int32_t), first * sizeof(std::int32_t))));
            };
            auto * local_indices = indices(local, 0);
            auto * global_indices = global ? indices(global, local) : nullptr;
            const auto gather = [&](ggml_tensor * rows, ggml_tensor * ids) {
                return on_device(ggml_cast(context, on_device(ggml_get_rows(context, rows, ids)), GGML_TYPE_F16));
            };
            auto * local_values = gather(values, local_indices);
            ggml_tensor * global_logits = nullptr, * global_values = nullptr;
            if (global) {
                global_logits = on_device(ggml_mul_mat(context, gather(keys, global_indices), q));
                ggml_prec_set_acc(global_logits, GGML_PREC_F32);
                ggml_format_name(global_logits, "infllm.QK.L%u", impl_->layer);
                global_values = gather(values, global_indices);
            }
            auto * logits = global ? on_device(ggml_concat(context, impl_->local_logits, global_logits, 0)) : impl_->local_logits;
            auto * probabilities = on_device(ggml_soft_max_ext(context, logits, joint_mask, impl_->scale, 0));
            ggml_format_name(probabilities, "infllm.softmax.L%u", impl_->layer);
            const auto probability_view = [&](std::uint32_t count, std::uint32_t first) {
                return on_device(ggml_cont(context, ggml_view_3d(context, probabilities, count, 1, query_heads,
                    probabilities->nb[1], probabilities->nb[2], first * sizeof(float))));
            };
            auto * local_probabilities = probability_view(local, 0);
            const auto multiply = [&](ggml_tensor * v, ggml_tensor * p) {
                auto * output = on_device(ggml_mul_mat(context, on_device(ggml_cont(context, ggml_transpose(context, v))), p));
                ggml_prec_set_acc(output, GGML_PREC_F32);
                return output;
            };
            auto * output = multiply(local_values, local_probabilities);
            if (global) output = on_device(ggml_add(context, output, multiply(global_values, probability_view(global, local))));
            ggml_format_name(output, "infllm.PV.L%u", impl_->layer);
            return finish_infllm(global_logits ? global_logits : probabilities, output, local_probabilities, true);
        }
        const auto position_count = slots * g.heads;
        const auto index_count = static_cast<std::int64_t>(impl_->session.budget().local) * impl_->session.query_heads();
        impl_->metadata = ggml_custom_4d(context, GGML_TYPE_I32, position_count + index_count, 1, 1, 1,
                                        &markers, 1, Impl::decode_metadata, 1, impl_.get());
        ggml_set_name(impl_->metadata, "infllm_decode_metadata");
        auto * positions = ggml_view_1d(context, impl_->metadata, position_count, 0);
        ggml_set_name(positions, "solid_infllm_key_positions");
        auto * indices = ggml_view_1d(context, impl_->metadata, index_count, position_count * sizeof(std::int32_t));
        ggml_set_name(indices, "solid_infllm_recent_indices");
        auto * rows = on_device(ggml_cont(context, keys));
        rows = ggml_reshape_3d(context, rows, g.dimension, 1, slots * g.heads);
        rows = on_device(ggml_rope_ext(context, rows, positions, nullptr, rope.dimensions,
            rope.split_halves ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL, 0, rope.base, 1, 0, 1, 0, 0));
        rows = ggml_reshape_3d(context, rows, g.dimension, slots, g.heads);
        auto * logits = on_device(ggml_mul_mat(context, rows, q));
        ggml_format_name(logits, "infllm.QK.L%u", impl_->layer);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        auto * probabilities = on_device(ggml_soft_max_ext(context, logits, mask, impl_->scale, 0));
        ggml_format_name(probabilities, "infllm.softmax.L%u", impl_->layer);
        auto * transposed_values = on_device(ggml_cont(context, ggml_transpose(context, values)));
        auto * output = on_device(ggml_mul_mat(context, transposed_values, probabilities));
        ggml_format_name(output, "infllm.PV.L%u", impl_->layer);
        ggml_prec_set_acc(output, GGML_PREC_F32);
        auto * compact = on_device(ggml_get_rows(context, ggml_reshape_2d(context, probabilities, 1, slots * impl_->session.query_heads()), indices));
        ggml_set_name(compact, "solid_infllm_recent_scores");
        return finish_infllm(logits, output, compact, false);
    };
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
        if (impl_->session.relative_rope()) {
            return infllm_attention(q, keys, values, mask, static_cast<std::size_t>(slots) * impl_->session.query_heads() * sizeof(Half));
        }
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
    } else if (impl_->selected_cache) {
        const auto slots = impl_->session.gather_slots();
        if (impl_->local_logits) {
            const auto offset = static_cast<std::size_t>(slots) * sizeof(std::int32_t);
            auto * mask = ggml_view_3d(context, result, slots, 1, impl_->session.query_heads(),
                slots * sizeof(Half), slots * sizeof(Half), offset);
            return infllm_attention(ggml_permute(context, impl_->device_query, 0, 2, 1, 3),
                impl_->selected_cache->plane(context, false), impl_->selected_cache->plane(context, true), mask,
                offset + static_cast<std::size_t>(slots) * impl_->session.query_heads() * sizeof(Half));
        }
        auto * indices = ggml_custom_4d(context, GGML_TYPE_I32, slots, g.heads, 1, 1, &result, 1, Impl::cache_indices, 1, nullptr);
        ggml_set_name(indices, "infllm_selected_kv_indices");
        const auto gather = [&](bool values) {
            auto * rows = on_device(ggml_get_rows(context, impl_->selected_cache->plane(context, values), indices));
            rows = on_device(ggml_cast(context, rows, GGML_TYPE_F16));
            return rows;
        };
        const auto mask_offset = static_cast<std::size_t>(slots) * sizeof(std::int32_t);
        auto * mask = ggml_view_3d(context, result, slots, 1, impl_->session.query_heads(), slots * sizeof(Half), slots * sizeof(Half), mask_offset);
        auto * q = query->type == GGML_TYPE_F32 ? query : ggml_cast(context, query, GGML_TYPE_F32);
        return infllm_attention(ggml_permute(context, q, 0, 2, 1, 3), gather(false), gather(true), mask,
            mask_offset + static_cast<std::size_t>(slots) * impl_->session.query_heads() * sizeof(Half));
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
        if (impl_->session.relative_rope()) {
            return infllm_attention(q, keys, values, mask,
                mask_offset + static_cast<std::size_t>(slots) * impl_->session.query_heads() * sizeof(Half));
        }
        result = ggml_flash_attn_ext(context, q, keys, values, mask, impl_->scale, 0, 0);
        ggml_prec_set_acc(result, GGML_PREC_F32);
#ifdef SOLID_HAVE_SYCL
        if (scheduler && decode_backend && ggml_backend_is_sycl(decode_backend)) {
            ggml_set_name(result, "solid_gathered_flash_attention");
            on_device(result);
        }
#endif
        result = ggml_reshape_2d(context, result, g.dimension * impl_->session.query_heads(), 1);
        ggml_set_name(result, "solid_gathered_attention");
    }
    return result;
}
void AttentionNode::check() const { impl_->failure->check(); }
bool AttentionNode::can_reuse(std::uint64_t next_position) const {
    // Decode uses a fixed bank and mask shape. Prefill owns batch-sized output
    // storage and must be rebuilt, including a one-token final prefill batch.
    return impl_->host && impl_->gathered && impl_->frame->positions.size() == 1 &&
        next_position < impl_->session.geometry().capacity_tokens &&
        impl_->frame->positions.front() < next_position &&
        next_position - impl_->frame->positions.front() == 1;
}
ggml_tensor * AttentionNode::host_node() const { return impl_->host; }
ggml_tensor * AttentionNode::metadata_node() const { return impl_->metadata; }
ggml_tensor * AttentionNode::local_attention_node() const { return impl_->local_logits; }
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
    std::shared_ptr<InfllmSelectedCache> infllm_workspace;
    std::unique_ptr<SparseBaselineLookahead> sparse_lookahead;
    bool sparse_lookahead_enabled = false;
    std::unique_ptr<DenseOffload> dense;
    bool dense_enabled = false;
    std::vector<std::optional<ProjectionIntervals>> timing;
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
    geometry.contiguous_planes = !options.in_memory && !options.interleaved;
    impl_->dense_enabled = options.gpu_decode && !options.sparse && !options.fuse_kv_projection && geometry.contiguous_planes;
    impl_->timing.resize(geometry.layers);
    if (options.infllm_attention && !options.in_memory && !options.prefetch && !options.interleaved &&
        !options.fuse_kv_projection && options.selection_replay.empty()) {
        impl_->sparse_lookahead_enabled = true;
    }
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
    budget.shared_heads = options.sparse;
    budget.infllm_attention = options.infllm_attention;
    std::optional<RelativeRope> rope;
    if (budget.infllm_attention) {
        const auto type = llama_model_rope_type(model);
        if ((type != LLAMA_ROPE_TYPE_NORM && type != LLAMA_ROPE_TYPE_NEOX) || hp.use_kq_norm) {
            throw std::invalid_argument("InfLLM requires standard text RoPE and raw Q/K projections");
        }
        rope = RelativeRope{hp.n_rot(), hp.rope_freq_base_train, type == LLAMA_ROPE_TYPE_NEOX};
        for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
        if (model->dev_layer(layer) != model->dev_layer(0)) impl_->dense_enabled = false;
        if (model->dev_layer(layer) != model->dev_layer(0)) impl_->sparse_lookahead_enabled = false;
            if (hp.n_rot(layer) != rope->dimensions) throw std::invalid_argument("InfLLM requires uniform rotary dimensions");
        }
    }
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
    if (budget.infllm_attention && options.gpu_decode && impl_->prefill_backend == "auto") impl_->prefill_backend = "cuda";
    impl_->configure_prefill = options.gpu_decode && impl_->prefill_backend != "cpu";
    impl_->prefill_queries=std::min(prefill_queries,capacity);
    impl_->prefill_tile_tokens=options.prefill_tile_tokens;
    if (options.prefetch) impl_->placement=std::make_unique<WorkerPlacement>(options.io_cpu,options.coordinator_cpu);
    budget.validate(geometry);
    const auto slots = options.prefetch_slots_per_head ? options.prefetch_slots_per_head : budget.cache_slots(geometry);
    const auto admission=std::max<std::uint64_t>(options.prefetch_queue_depth,static_cast<std::uint64_t>(geometry.heads)*slots*2);
    if (options.prefetch && admission>16384) throw std::invalid_argument("prefetch worker admission exceeds 16384 requests");
    impl_->session = std::make_unique<AttentionSession>(std::move(store), hp.n_head(), budget, options.sparse, prefill_tokens,
        options.selection_trace ? trace : std::filesystem::path{},
        options.selection_replay, options.prefetch ? make_threaded_uring_reader(options.prefetch_queue_depth,impl_->placement->io_cpu(),static_cast<std::uint32_t>(admission)) : nullptr,
        options.prefetch_slots_per_head, options.prefetch_trace ? trace.parent_path() / "prefetch.jsonl" : std::filesystem::path{}, rope,
        options.prefetch_ahead);
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
    try {
        if (impl_->device) impl_->device->process_complete();
        if (impl_->dense) impl_->dense->finish_decode();
        if (impl_->sparse_lookahead) impl_->sparse_lookahead->finish_decode();
        impl_->session->finish_device_probabilities();
    } catch (...) { impl_->failure->capture(std::current_exception()); }
    check();
    impl_->session->release_prefill_if_complete();
}
void AttentionAdapter::flush() { check(); impl_->session->flush(); }
void AttentionAdapter::enable_direct_reads() { check(); impl_->session->enable_direct_reads(); }
void AttentionAdapter::finish() { check(); quiesce(); impl_->session->finish(); }
SessionStats AttentionAdapter::stats() const { check(); return impl_->session->stats(); }
SparseBudget AttentionAdapter::budget() const { return impl_->session->budget(); }
const std::optional<RelativeRope> & AttentionAdapter::relative_rope() const { return impl_->session->relative_rope(); }
std::unique_ptr<AttentionNode> AttentionAdapter::node(std::shared_ptr<AttentionFrame> frame, std::uint32_t layer, float scale,
                                                    bool gathered, ggml_backend_t backend) {
    if (impl_->configure_prefill) {
        bool configured=false;
        const auto & prefill=impl_->prefill_backend;
#ifdef SOLID_HAVE_CUDA_SELECTION
        if ((prefill=="auto" || prefill=="cuda") && backend && ggml_backend_is_cuda(backend)) {
            auto * registry=ggml_backend_cuda_reg();
            for (std::size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(registry);++ordinal) {
                if (ggml_backend_reg_dev_get(registry,ordinal)!=ggml_backend_get_device(backend)) continue;
                const auto budget = impl_->session->budget();
                const auto queries = budget.infllm_attention ? std::min(impl_->prefill_queries, budget.prefill_group_tokens())
                                                             : impl_->prefill_queries;
                const auto sparse_tile = std::min<std::uint64_t>(impl_->session->geometry().capacity_tokens,
                    static_cast<std::uint64_t>(budget.total) + queries - 1);
                impl_->session->set_prefill(make_cuda_tiled_prefill(impl_->session->geometry(),impl_->session->query_heads(),
                    queries, budget.infllm_attention ? static_cast<std::uint32_t>(sparse_tile) : impl_->prefill_tile_tokens, static_cast<int>(ordinal),
                    budget.infllm_attention ? budget.local : 0));
                configured=true; break;
            }
        }
#endif
#ifdef SOLID_HAVE_SYCL
        if ((prefill=="auto" || prefill=="sycl") && backend && ggml_backend_is_sycl(backend)) {
            impl_->session->set_prefill(make_sycl_tiled_prefill(impl_->session->geometry(),impl_->session->query_heads(),
                impl_->prefill_queries,impl_->prefill_tile_tokens,solid_sycl_queue(backend)));
            configured=true;
        }
#endif
        if (!configured && prefill=="cuda") throw std::runtime_error("CUDA prefill requires a CUDA build and CUDA attention backend");
        if (!configured && prefill=="sycl") throw std::runtime_error("SYCL prefill requires a SYCL build and SYCL attention backend");
        impl_->configure_prefill=false;
    }
    if (impl_->configure_ranker) {
        bool configured = false;
        const auto & selector = impl_->selection_backend;
#ifdef SOLID_HAVE_CUDA_SELECTION
        if ((selector == "auto" || selector == "cuda") && backend && ggml_backend_is_cuda(backend)) {
            auto * registry = ggml_backend_cuda_reg();
            for (std::size_t ordinal = 0; ordinal < ggml_backend_reg_dev_count(registry); ++ordinal) {
                if (ggml_backend_reg_dev_get(registry, ordinal) != ggml_backend_get_device(backend)) continue;
                const auto budget = impl_->session->budget();
                auto ranker = make_cuda_block_ranker(impl_->session->representative_geometry(), impl_->session->query_heads(),
                    budget.infllm_attention ? 1 : budget.representatives, budget.total, static_cast<int>(ordinal),
                    budget.infllm_attention ? 0 : budget.local, impl_->session->geometry().heads);
                if (budget.infllm_attention && (impl_->prefill_backend == "cuda" || impl_->prefill_backend == "auto")) {
                    if (!ranker->configure_probability_builder(budget.initial, budget.local, budget.representatives)) {
                        throw std::runtime_error("CUDA probability representative builder unavailable");
                    }
                }
                impl_->session->set_ranker(std::move(ranker));
                configured = true;
                break;
            }
        }
#endif
#ifdef SOLID_HAVE_SYCL
        if ((selector == "auto" || selector == "sycl") && backend && ggml_backend_is_sycl(backend)) {
            const auto budget = impl_->session->budget();
            impl_->session->set_ranker(make_sycl_block_ranker(impl_->session->representative_geometry(), impl_->session->query_heads(),
                budget.infllm_attention ? 1 : budget.representatives, budget.total, solid_sycl_queue(backend),
                budget.infllm_attention ? 0 : budget.local, impl_->session->geometry().heads));
            configured = true;
        }
#endif
        if (!configured && selector == "cuda") throw std::runtime_error("CUDA selection requires a CUDA build and CUDA attention backend");
        if (!configured && selector == "sycl") throw std::runtime_error("SYCL selection requires a SYCL build and SYCL attention backend");
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
    InfllmSelectedCache * selected_cache = nullptr;
    if (gathered && backend && impl_->sparse_lookahead_enabled &&
        ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        auto & cache = impl_->infllm_workspace;
        if (!cache) cache = make_infllm_selected_cache(*impl_->session, backend);
        if (cache->backend() != backend) throw std::logic_error("InfLLM cache backend changed");
        selected_cache = cache.get();
        if (!impl_->sparse_lookahead) {
            impl_->sparse_lookahead = std::make_unique<SparseBaselineLookahead>(*impl_->session, *cache, uring_available());
        }
    }
    if (gathered && impl_->dense_enabled && !impl_->dense) {
        impl_->dense = std::make_unique<DenseOffload>(*impl_->session, backend);
    }
    return std::make_unique<AttentionNode>(*impl_->session, std::move(frame), layer, scale, impl_->failure, gathered,
                                          gathered ? impl_->device.get() : nullptr,
                                          gathered ? projection_intervals(layer,backend) : nullptr, selected_cache,
                                          gathered ? impl_->dense.get() : nullptr,
                                          gathered ? impl_->sparse_lookahead.get() : nullptr);
}
bool AttentionAdapter::prefetch_enabled() const { return impl_->session->prefetch_enabled(); }
bool AttentionAdapter::dense_offload_enabled() const { return impl_->dense_enabled; }
void AttentionAdapter::begin_prefetch(std::uint32_t layer, std::uint64_t position) {
    check();
    if (impl_->dense && gathered_decode(position)) impl_->dense->begin(position);
    else if (impl_->sparse_lookahead && gathered_decode(position)) impl_->sparse_lookahead->begin(layer, position);
    else if (impl_->device) impl_->device->begin_prefetch(layer, position);
    else impl_->session->begin_prefetch(layer, position);
}
bool AttentionAdapter::observe(ggml_tensor * tensor, bool ask) noexcept {
    if (impl_->dense && tensor->op == GGML_OP_FLASH_ATTN_EXT &&
        std::strcmp(ggml_get_name(tensor), "offload_dense_flash_attention") == 0) {
        if (ask) return true;
        try { check(); impl_->dense->attention_complete(); }
        catch (...) { impl_->failure->capture(std::current_exception()); return false; }
        return true;
    }
    const bool normal = tensor->op == GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor), "solid_prefetched_flash_attention") == 0;
    const bool reused = tensor->op == GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor), "solid_prefetched_flash_attention_reuse") == 0;
    const bool ours = normal || reused;
    if (ask) return normal;
    if (!ours) return true;
    try { if (impl_->device) impl_->device->completed(impl_->device->attention_epoch()); }
    catch (...) { impl_->failure->capture(std::current_exception()); return false; }
    return true;
}
void AttentionAdapter::quiesce() {
    if (impl_->device) impl_->device->quiesce();
    if (impl_->dense) impl_->dense->join();
    if (impl_->sparse_lookahead) impl_->sparse_lookahead->join();
}
DeviceCacheStats AttentionAdapter::device_stats() const {
    if (impl_->device) return impl_->device->stats();
    if (impl_->dense) return impl_->dense->stats();
    return impl_->infllm_workspace ? impl_->infllm_workspace->stats() : DeviceCacheStats{};
}
ScheduleStats AttentionAdapter::schedule_stats() const {
    if (impl_->device) return impl_->device->schedule_stats();
    ScheduleStats result;
    result.staging_bytes = device_stats().staging_bytes;
    return result;
}
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
std::uint64_t AttentionAdapter::projection_source_bytes() const {
    std::uint64_t bytes = 0;
    for (const auto & p : impl_->projections) bytes += p->stats().source_bytes;
    return bytes;
}
std::uint64_t AttentionAdapter::projection_source_host_bytes() const {
    std::uint64_t bytes = 0;
    for (const auto & p : impl_->projections) bytes += p->stats().source_host_bytes;
    return bytes;
}
std::vector<std::string> AttentionAdapter::projection_weight_types() const {
    std::vector<std::string> types;
    for (const auto & p : impl_->projections) types.push_back(p->stats().weight_type);
    return types;
}

ProjectionIntervals * AttentionAdapter::projection_intervals(std::uint32_t layer,ggml_backend_t backend) {
    if (!backend || !prefetch_enabled() || impl_->schedule==ScheduleMode::naive || !impl_->session->direct_memory()) return nullptr;
    auto & slot=impl_->timing.at(layer);
    if (!slot) {
        ProjectionIntervals intervals;
#ifdef SOLID_HAVE_CUDA_SELECTION
        if (ggml_backend_is_cuda(backend)) {
            auto * registry=ggml_backend_cuda_reg();
            for (std::size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(registry);++ordinal) {
                if (ggml_backend_reg_dev_get(registry,ordinal)!=ggml_backend_get_device(backend)) continue;
                intervals.query=make_cuda_device_interval(static_cast<int>(ordinal));
                intervals.kv=make_cuda_device_interval(static_cast<int>(ordinal));
                break;
            }
        }
#endif
#ifdef SOLID_HAVE_SYCL
        if (ggml_backend_is_sycl(backend)) {
            const auto queue=solid_sycl_queue(backend);
            intervals.query=make_sycl_device_interval(queue);
            intervals.kv=make_sycl_device_interval(queue);
        }
#endif
        slot.emplace(std::move(intervals));
    }
    return slot->query || slot->kv ? &*slot : nullptr;
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
            if (batch->pos[i] < 0 || batch->n_seq_id[i] != 1 || !batch->seq_id[i] || batch->seq_id[i][0] != 0) {
                throw std::invalid_argument("adapter supports one sequence with nonnegative positions");
            }
            frame->positions[i] = static_cast<std::uint64_t>(batch->pos[i]);
        }
        if (retrieval_positions) {
            const std::vector<std::int32_t> angles(batch->n_tokens, static_cast<std::int32_t>(adapter.budget().local - 1));
            ggml_backend_tensor_set(retrieval_positions, angles.data(), 0, angles.size() * sizeof(std::int32_t));
        }
        if (batch->n_tokens == 1) adapter.begin_prefetch(0, frame->positions.front());
    }
    bool can_reuse(const llm_graph_params & params) override {
        // llm_graph_result checks model, topology, outputs and the other inputs.
        // This input owns the mutable positions and the custom attention nodes.
        const auto & batch = params.ubatch;
        if (batch.n_tokens != 1 || batch.n_pos != 1 || !batch.pos || batch.pos[0] < 0 ||
            !batch.n_seq_id || batch.n_seq_id[0] != 1 || !batch.seq_id || !batch.seq_id[0] ||
            batch.seq_id[0][0] != 0 || nodes.size() != params.hparams.n_layer()) return false;
        return !nodes.empty() && std::all_of(nodes.begin(), nodes.end(), [&](const auto & node) {
            return node->can_reuse(static_cast<std::uint64_t>(batch.pos[0]));
        });
    }
    AttentionAdapter & adapter;
    std::shared_ptr<AttentionFrame> frame;
    std::vector<std::unique_ptr<AttentionNode>> nodes;
    std::unordered_map<int, ggml_tensor *> projected_kv;
    ggml_tensor * retrieval_positions = nullptr;
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
void adapter_mark_projection(const llm_graph_context & graph,ggml_tensor * projection,int layer,bool query) {
    auto * adapter=lookup(&graph.model);
    if (!adapter || layer<0 || graph.ubatch.n_tokens!=1 || !graph.ubatch.pos || graph.ubatch.pos[0]<0 ||
        !adapter->gathered_decode(static_cast<std::uint64_t>(graph.ubatch.pos[0])) || !graph.loras->empty()) return;
    auto * timing=adapter->projection_intervals(static_cast<std::uint32_t>(layer),layer_backend(graph,layer));
    if (!timing) return;
    while (projection && (projection->op==GGML_OP_RESHAPE || projection->op==GGML_OP_MUL || projection->op==GGML_OP_ADD)) {
        projection=projection->src[0];
    }
    if (projection && projection->op==GGML_OP_MUL_MAT) bind_device_timing(projection,query ? timing->query.get() : timing->kv.get());
}
llm_graph_qkv adapter_qkv(const llm_graph_context & graph, const llama_layer & layer, ggml_tensor * input,
                         std::int64_t dimension_q, std::int64_t heads_q, std::int64_t dimension_k,
                         std::int64_t heads_k, std::int64_t dimension_v, std::int64_t heads_v, int id, bool reshape) {
    auto * adapter = lookup(&graph.model);
    if (!adapter || !reshape || id < 0 || dimension_k != dimension_v || heads_k != heads_v ||
        dimension_k != graph.hparams.n_embd_head_k() || heads_k != graph.hparams.n_head_kv() ||
        !graph.loras->empty() || graph.hdmd) throw std::invalid_argument("unsupported fused KV projection graph");
    auto * q = graph.build_lora_mm(layer.wq, input, layer.wq_s);
    adapter_mark_projection(graph,q,id,true);
    if (layer.wq_b) q = ggml_add(graph.ctx0, q, layer.wq_b);
    auto * packed = adapter->projection(id).project(graph.ctx0, input);
    ggml_format_name(packed, "kv_projection.L%d", id);
    adapter_mark_projection(graph,packed,id,false);
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
    ggml_tensor * absolute_positions = nullptr;
    const auto & rope = custom->adapter.relative_rope();
    if (rope) {
        if (query->op != GGML_OP_ROPE || key->op != GGML_OP_ROPE) {
            throw std::logic_error("InfLLM requires raw Q/K before the model RoPE operation");
        }
        absolute_positions = query->src[1];
        if (!custom->retrieval_positions) {
            custom->retrieval_positions = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, query->ne[2]);
            ggml_set_input(custom->retrieval_positions);
        }
        query = ggml_rope_ext(graph.ctx0, query->src[0], custom->retrieval_positions, nullptr, rope->dimensions,
            rope->split_halves ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL, 0, rope->base, 1, 0, 1, 0, 0);
        key = key->src[0];
    }
    if (custom->adapter.fused_projection()) {
        packed = rope ? custom->projected_kv.at(layer) : KvProjection::update_key(graph.ctx0, custom->projected_kv.at(layer), key);
    }
    auto * device=layer_backend(graph,layer);
    auto node = custom->adapter.node(custom->frame, static_cast<std::uint32_t>(layer), scale, gathered, device);
    auto * result = node->build(graph.ctx0, query, key, value, packed, graph.sched, device, graph.gf, absolute_positions);
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
    if (node->local_attention_node()) ggml_build_forward_expand(graph.gf, node->local_attention_node());
    if (auto * metadata = node->metadata_node()) {
        // Both tables depend only on the completed host frontier. Schedule
        // them together before K copies/rotation/QK/PV; no CPU index node
        // should interrupt the GPU attention and probability gather.
        ggml_backend_sched_set_tensor_backend(graph.sched, metadata, graph.backend_cpu);
        ggml_build_forward_expand(graph.gf, metadata);
    }
    // The prefetched bank is populated by the host frontier. Expand it before
    // visiting matrix inputs that copy K/V out of the bank.
    if (rope && gathered && custom->adapter.prefetch_enabled()) ggml_build_forward_expand(graph.gf, node->host_node());
    custom->nodes.push_back(std::move(node));
    return result;
}
} // namespace solid
