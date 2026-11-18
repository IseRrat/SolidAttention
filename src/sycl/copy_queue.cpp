#include "solid/sycl_backend.hpp"
#include "solid/sycl_runtime.hpp"

namespace solid {
namespace {
class CopyEvent final : public SyclCopyEvent {
public:
    CopyEvent(sycl::event event, sycl::queue & compute) : event_(std::move(event)), compute_(compute) {}

    bool ready() const override {
        return event_.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
    }
    void wait() override { event_.wait_and_throw(); }
    void wait_on_compute() override { compute_.ext_oneapi_submit_barrier({event_}); }

private:
    sycl::event event_;
    sycl::queue & compute_;
};

class CopyQueue final : public SyclCopyQueue {
public:
    explicit CopyQueue(sycl::queue & compute) : compute_(compute), queue_(sycl_runtime::in_order_queue(compute)) {}

    void copy(void * target, const void * source, std::size_t bytes) override { queue_.memcpy(target, source, bytes); }
    std::unique_ptr<SyclCopyEvent> fence() override {
        return std::make_unique<CopyEvent>(queue_.ext_oneapi_submit_barrier(), compute_);
    }
    void drain() override { queue_.wait_and_throw(); }

private:
    sycl::queue & compute_;
    sycl::queue queue_;
};
} // namespace

std::unique_ptr<SyclCopyQueue> make_sycl_copy_queue(ggml_backend_t backend) {
    return std::make_unique<CopyQueue>(*static_cast<sycl::queue *>(solid_sycl_queue(backend)));
}
} // namespace solid
