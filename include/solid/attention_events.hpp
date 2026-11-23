#pragma once
#include "solid/stream_completion.hpp"
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

struct ggml_backend_event;
namespace solid {
enum class AttentionProof { device_signal, query_boundary, process_boundary, attention_boundary };
struct AttentionReceipt { AttentionProof proof; std::uint64_t device_ns; };
class AttentionEvents {
public:
    explicit AttentionEvents(std::uint32_t layers);
    virtual ~AttentionEvents();
    AttentionEvents(const AttentionEvents &)=delete;
    AttentionEvents & operator=(const AttentionEvents &)=delete;
    StreamCompletionToken * token(std::uint32_t layer);
    void arm(std::uint32_t layer,std::uint64_t epoch);
    std::optional<AttentionReceipt> poll(std::uint32_t layer,std::uint64_t epoch);
    void boundary(AttentionProof proof,std::uint64_t epoch=0);
    void stop() noexcept;
    bool stopped() const { return stopping_.load(std::memory_order_acquire); }
    bool boundary_seen(std::uint64_t epoch) const;
    bool handoff_seen(std::uint64_t epoch) const;
    // Called after an existing full backend fence, including canceled frames.
    void retire_after_sync();
    int completion_handle() const { return notification_; }
    void acknowledge();
    virtual bool copy_ready(ggml_backend_event * event) const=0;
    virtual bool host_marker() const { return false; }
    virtual const char * name() const=0;
protected:
    virtual std::uint64_t elapsed_ns(std::uint32_t layer) const=0;
private:
    struct Slot;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::atomic<std::uint64_t> active_{0},consumed_{0},boundary_{0},handoff_{0};
    std::atomic<int> boundary_proof_{0},notification_error_{0};
    std::atomic<bool> stopping_{false};
    std::uint64_t last_epoch_=0;
    int notification_=-1;
    static void signal(void * userdata) noexcept;
    static std::uint64_t armed_epoch(void * userdata) noexcept;
    static void signal_epoch(void * userdata,std::uint64_t epoch) noexcept;
    void wake() noexcept;
};
std::unique_ptr<AttentionEvents> make_cuda_attention_events(std::uint32_t layers,int device);
std::unique_ptr<AttentionEvents> make_sycl_attention_events(std::uint32_t layers,void * backend_queue);
} // namespace solid
