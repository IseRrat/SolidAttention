#include "solid/ggml_prefetch.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <limits>
#include <map>
#include <stdexcept>

namespace solid {
namespace {
class GgmlTransport final : public DeviceTransport {
public:
    GgmlTransport(ggml_backend_t backend, KvGeometry geometry, std::uint32_t per_head) :
        backend_(backend), geometry_(geometry), per_head_(per_head) {
        if (!backend || geometry.layout != KvLayout::interleaved) throw std::invalid_argument("invalid GGML prefetch transport");
        const auto tokens = (static_cast<std::uint64_t>(per_head) * geometry.block_tokens + 31) / 32 * 32;
        if (tokens > INT32_MAX || tokens * geometry.heads > static_cast<std::uint64_t>(INT64_MAX) / (2ull * geometry.dimension * sizeof(Half))) {
            throw std::overflow_error("GGML prefetch transport allocation overflow");
        }
        context_.reset(ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("device cache metadata allocation");
        tensor_ = ggml_new_tensor_3d(context_.get(), GGML_TYPE_F16, geometry.dimension * 2, tokens, geometry.heads);
        ggml_set_name(tensor_, "solid_prefetch_device_slots");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend));
        if (!buffer_) throw std::runtime_error("device cache allocation");
        ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer_.get(), 0);
    }
    ~GgmlTransport() override {
        if (!events_.empty()) {
            try { drain(); } catch (...) { std::terminate(); }
            for (const auto & event : events_) if (event.second) ggml_backend_event_free(event.second);
        }
    }
    std::uint64_t upload(std::uint32_t slot, const KvBlock & block) override {
        if (slot >= geometry_.heads * static_cast<std::uint64_t>(per_head_) || block.layout != KvLayout::interleaved ||
            block.dimension != geometry_.dimension || block.capacity != geometry_.block_tokens ||
            block.data.size() != static_cast<std::size_t>(geometry_.block_tokens) * 2 * geometry_.dimension) {
            throw std::invalid_argument("device upload shape");
        }
        if (next_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("copy ticket exhausted");
        const auto id = next_++;
        auto * device = ggml_backend_get_device(backend_);
        auto * event = device ? ggml_backend_event_new(device) : nullptr;
        try {
            events_.emplace(id, event);
            const auto head = slot / per_head_, local = slot % per_head_;
            const auto offset = head * tensor_->nb[2] + local * geometry_.block_tokens * tensor_->nb[1];
            ggml_backend_tensor_set_async(backend_, tensor_, block.data.data(), offset, block.data.size() * sizeof(Half));
            if (event) ggml_backend_event_record(event, backend_);
            else ggml_backend_synchronize(backend_);
        } catch (...) {
            ggml_backend_synchronize(backend_);
            events_.erase(id);
            if (event) ggml_backend_event_free(event);
            throw;
        }
        return id;
    }
    bool complete(std::uint64_t ticket, bool wait) override {
        const auto found = events_.find(ticket);
        if (found == events_.end()) throw std::logic_error("unknown upload ticket");
        if (found->second && !wait) return false;
        if (found->second) { ggml_backend_event_synchronize(found->second); ggml_backend_event_free(found->second); }
        events_.erase(found);
        return true;
    }
    void drain() override { ggml_backend_synchronize(backend_); }
    std::uint64_t resident_bytes() const override { return ggml_backend_buffer_get_size(buffer_.get()); }
    ggml_tensor * tensor() const { return tensor_; }
private:
    ggml_backend_t backend_;
    KvGeometry geometry_;
    std::uint32_t per_head_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor * tensor_ = nullptr;
    std::map<std::uint64_t, ggml_backend_event_t> events_;
    std::uint64_t next_ = 1;
};
}
struct GgmlPrefetch::Impl {
    AttentionSession & session;
    GgmlTransport transport;
    DeviceCache device;
    bool quiesced = false;
    Impl(AttentionSession & s, ggml_backend_t backend, const std::filesystem::path & trace) :
        session(s), transport(backend, s.geometry(), s.prefetcher().slots_per_head()),
        device(s.prefetcher(), s.geometry(), s.query_heads(), transport, trace) {}
};
GgmlPrefetch::GgmlPrefetch(AttentionSession & session, ggml_backend_t backend, const std::filesystem::path & trace) :
    impl_(std::make_unique<Impl>(session, backend, trace)) {}
GgmlPrefetch::~GgmlPrefetch() = default;
ggml_tensor * GgmlPrefetch::tensor() const { return impl_->transport.tensor(); }
std::uint32_t GgmlPrefetch::tokens_per_head() const { return impl_->device.tokens_per_head(); }
std::uint64_t GgmlPrefetch::attention_epoch() const { return impl_->device.attention_epoch(); }
std::vector<Half> GgmlPrefetch::decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                      const std::vector<float> & keys, const std::vector<float> & values, bool packed) {
    impl_->quiesced = false;
    impl_->device.wait_transfers();
    impl_->session.begin_prefetch(layer, position);
    const auto selection = impl_->session.select_decode(layer, position, query, keys, values, packed);
    auto & host = impl_->session.prefetcher();
    const auto leases = host.resolve(layer, position, selection);
    std::vector<Half> mask;
    try { mask = impl_->device.prepare(leases, selection); }
    catch (...) { for (const auto & lease : leases) host.unpin(lease); throw; }
    for (const auto & lease : leases) host.unpin(lease);
    const auto next_layer = layer + 1 < impl_->session.geometry().layers ? layer + 1 : 0;
    const auto next_position = next_layer ? position : position + 1;
    impl_->session.begin_prefetch(next_layer, next_position);
    return mask;
}
void GgmlPrefetch::completed(std::uint64_t epoch) {
    if (impl_->device.attention_complete(epoch)) {
        impl_->session.prefetcher().poll();
        impl_->device.prefetch_ready();
    }
}
void GgmlPrefetch::quiesce() {
    if (impl_->quiesced) return;
    impl_->transport.drain();
    std::exception_ptr error;
    try { impl_->device.wait_transfers(); } catch (...) { error = std::current_exception(); }
    try {
        if (impl_->device.attention_pending()) (void) impl_->device.attention_complete(impl_->device.attention_epoch());
    } catch (...) { if (!error) error = std::current_exception(); }
    try { impl_->device.finish(); } catch (...) { if (!error) error = std::current_exception(); }
    impl_->quiesced = true;
    if (error) std::rethrow_exception(error);
}
DeviceCacheStats GgmlPrefetch::stats() const { return impl_->device.stats(); }
} // namespace solid
