#include "solid/sycl_backend.hpp"
#include "solid/sycl_runtime.hpp"

namespace solid {
namespace {
class CopyEvent final : public SyclCopyEvent {
public:
    CopyEvent(sycl::event event, std::optional<sycl::event> start, sycl::queue & compute) :
        event_(std::move(event)), start_(std::move(start)), compute_(compute) {}

    bool ready() const override {
        return event_.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
    }
    void wait() override { event_.wait_and_throw(); }
    void wait_on_compute() override { compute_.ext_oneapi_submit_barrier({event_}); }
    std::optional<std::uint64_t> duration() const override {
        if (!start_) return {};
        const auto start=start_->get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event_.get_profiling_info<sycl::info::event_profiling::command_end>();
        if (end<start) throw std::runtime_error("invalid SYCL copy interval");
        return end-start;
    }

private:
    sycl::event event_;
    std::optional<sycl::event> start_;
    sycl::queue & compute_;
};

class CopyQueue final : public SyclCopyQueue {
public:
    explicit CopyQueue(sycl::queue & compute) : compute_(compute), profiling_(compute.get_device().has(sycl::aspect::queue_profiling)),
        queue_(sycl_runtime::in_order_queue(compute,profiling_)) {}

    void copy(void * target, const void * source, std::size_t bytes) override {
        last_=queue_.memcpy(target,source,bytes);
        if (profiling_ && !first_) first_=last_;
    }
    std::unique_ptr<SyclCopyEvent> fence() override {
        if (!last_) throw std::logic_error("SYCL copy event requires a submitted copy");
        auto event=std::make_unique<CopyEvent>(*last_,first_,compute_);
        first_.reset(); last_.reset();
        return event;
    }
    void drain() override {
        first_.reset(); last_.reset();
        queue_.wait_and_throw();
    }

private:
    sycl::queue & compute_;
    bool profiling_;
    sycl::queue queue_;
    std::optional<sycl::event> first_, last_;
};
} // namespace

std::unique_ptr<SyclCopyQueue> make_sycl_copy_queue(ggml_backend_t backend) {
    return std::make_unique<CopyQueue>(*static_cast<sycl::queue *>(solid_sycl_queue(backend)));
}
} // namespace solid
