#include "solid/dense_offload.hpp"
#include "solid/session.hpp"
#include "solid/scheduler.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <limits>

namespace solid {
struct DenseOffload::Impl {
    using Clock = std::chrono::steady_clock;
    AttentionSession & session;
    ggml_backend_t copy_backend;
    const std::uint32_t slots;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * bank = nullptr;
    std::vector<Half> staging;
    Half * aligned = nullptr;
    std::unique_ptr<AsyncReader> reader;
    DeviceCacheStats stats;
    enum class Phase { idle, reading, uploaded, archived };
    Phase phase = Phase::idle;
    std::uint64_t position = 0;
    std::uint32_t layer = 0;
    // Destroy the worker first so the ring and its staging stay alive while it drains.
    TaskWorker worker;

    Impl(AttentionSession & owner, ggml_backend * backend) : session(owner), copy_backend(backend), slots(owner.gather_slots()) {
        const auto & geometry = session.geometry();
        if (!backend || !geometry.contiguous_planes || geometry.layout != KvLayout::separate) {
            throw std::invalid_argument("dense offload requires separate contiguous K/V planes");
        }
        context.reset(ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context) throw std::runtime_error("dense offload metadata allocation failed");
        bank = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, geometry.dimension, slots, geometry.heads, 2);
        ggml_set_name(bank, "offload_dense_kv_bank");
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw std::runtime_error("dense offload layer buffer allocation failed");
        ggml_backend_buffer_clear(buffer.get(), 0);
        stats.resident_bytes = ggml_backend_buffer_get_size(buffer.get());
        constexpr std::size_t alignment = 4096;
        const auto bytes = ggml_nbytes(bank);
        if (bytes > SIZE_MAX - alignment) throw std::overflow_error("dense KV staging size");
        staging.resize((bytes + alignment) / sizeof(Half));
        void * pointer = staging.data();
        auto space = staging.size() * sizeof(Half);
        aligned = static_cast<Half *>(std::align(alignment, bytes, pointer, space));
        if (!aligned) throw std::logic_error("dense KV staging alignment");
        stats.staging_bytes = staging.capacity() * sizeof(Half);
        if (uring_available() && ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            reader = make_uring_reader(2 * geometry.heads);
        }
    }
    void submit(std::uint32_t target) {
        layer = target;
        phase = Phase::reading;
        worker.submit([this, target, p = position] {
            session.read_dense_history(target, static_cast<std::uint32_t>(p), aligned, reader.get());
        });
    }
    void expect(std::uint32_t target, std::uint64_t p, Phase expected) const {
        if (phase != expected || target != layer || p != position) throw std::logic_error("dense KV decode frontier mismatch");
    }
};

DenseOffload::DenseOffload(AttentionSession & session, ggml_backend * backend) : impl_(std::make_unique<Impl>(session, backend)) {}
DenseOffload::~DenseOffload() = default;
void DenseOffload::begin(std::uint64_t position) {
    if (impl_->phase != Impl::Phase::idle || position >= impl_->session.geometry().capacity_tokens) {
        throw std::logic_error("invalid starting point for dense KV read");
    }
    impl_->position = position;
    impl_->submit(0);
}
void DenseOffload::upload_history(std::uint32_t layer, std::uint64_t position) {
    impl_->expect(layer, position, Impl::Phase::reading);
    const auto start = Impl::Clock::now();
    impl_->worker.join();
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(Impl::Clock::now() - start).count();
    impl_->session.account_dense_prefetch(waited);
    const auto & geometry = impl_->session.geometry();
    const auto bytes = static_cast<std::size_t>(position) * geometry.dimension * sizeof(Half);
    // The bank retains its allocation, but only initialized historical rows
    // participate in this upload. The current row is installed by the graph.
    try {
        for (int plane = 0; plane < 2; ++plane) for (std::uint32_t head = 0; head < geometry.heads; ++head) {
            const auto offset = plane * impl_->bank->nb[3] + head * impl_->bank->nb[2];
            ggml_backend_tensor_set_async(impl_->copy_backend, impl_->bank,
                reinterpret_cast<const char *>(impl_->aligned) + offset, offset, bytes);
        }
        ggml_backend_synchronize(impl_->copy_backend);
    } catch (...) { ggml_backend_synchronize(impl_->copy_backend); throw; }
    ++impl_->stats.uploads;
    impl_->stats.upload_bytes += bytes * 2 * geometry.heads;
    impl_->phase = Impl::Phase::uploaded;
}
void DenseOffload::archive(std::uint32_t layer, std::uint64_t position, const float * keys, const float * values) {
    impl_->expect(layer, position, Impl::Phase::uploaded);
    impl_->session.append_dense_decode(layer, position, keys, values);
    if (impl_->session.layer_write_ready(layer)) {
        impl_->worker.submit([this, layer] { impl_->session.flush_ready_layer(layer); });
    }
    impl_->phase = Impl::Phase::archived;
}
void DenseOffload::attention_complete() {
    if (impl_->phase != Impl::Phase::archived) throw std::logic_error("dense attention completed before KV archive");
    impl_->worker.join();
    const auto next = impl_->layer + 1;
    if (next < impl_->session.geometry().layers) impl_->submit(next);
    else impl_->phase = Impl::Phase::idle;
}
void DenseOffload::join() { impl_->worker.join(); }
void DenseOffload::finish_decode() {
    if (impl_->phase != Impl::Phase::idle) throw std::logic_error("dense decode finished with an incomplete layer");
}
ggml_tensor * DenseOffload::plane(ggml_context * context, bool values) const {
    auto * bank = impl_->bank;
    return ggml_view_3d(context, bank, bank->ne[0], bank->ne[1], bank->ne[2], bank->nb[1], bank->nb[2],
        values ? bank->nb[3] : 0);
}
DeviceCacheStats DenseOffload::stats() const { return impl_->stats; }
} // namespace solid
