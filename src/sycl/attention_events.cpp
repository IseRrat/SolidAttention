// SYCL attention 完成通知，把队列事件交给主机调度器。
#include "solid/attention_events.hpp"
#include "solid/sycl_flash_decode.hpp"
#include "ggml-backend-impl.h"

namespace solid {
namespace {

struct Frontier {
    sycl::queue * notifications = nullptr;
    sycl::event end;
};

class SyclAttentionEvents final : public AttentionEvents {
public:
    SyclAttentionEvents(std::uint32_t layers,const sycl::queue & compute) : AttentionEvents(layers),
        notifications_(compute.get_context(),compute.get_device()), frontiers_(layers) {
        for (std::uint32_t i=0;i<layers;++i) {
            frontiers_[i].notifications=&notifications_;
            token(i)->end_event=&frontiers_[i];
        }
    }
    ~SyclAttentionEvents() override { notifications_.wait(); }

    bool copy_ready(ggml_backend_event * event) const override {
        return static_cast<sycl::event *>(event->context)->get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete;
    }
    const char * name() const override { return "sycl_queue_events"; }

private:
    sycl::queue notifications_;
    std::vector<Frontier> frontiers_;
};
} // namespace

void sycl_attention_decode(sycl::queue & queue,ggml_tensor * output) {
    auto * token=stream_completion(output->op_params,sizeof(output->op_params));
    if (!token) {
        sycl_masked_decode(queue,output);
        return;
    }
    auto & frontier=*static_cast<Frontier *>(token->end_event);
    const auto epoch=token->armed_epoch(token->userdata);
    frontier.end=sycl_masked_decode(queue,output);
    // An older notification can arrive after GGML has proved completion at a
    // graph boundary. Carry the submitted epoch across the notification queue.
    frontier.notifications->submit([&](sycl::handler & cgh) {
        cgh.depends_on(frontier.end);
        cgh.host_task([token,epoch] { token->notify_epoch(token->userdata,epoch); });
    });
}

std::unique_ptr<AttentionEvents> make_sycl_attention_events(std::uint32_t layers,void * backend_queue) {
    if (!backend_queue) throw std::invalid_argument("SYCL completion requires an attention queue");
    return std::make_unique<SyclAttentionEvents>(layers,*static_cast<sycl::queue *>(backend_queue));
}
} // namespace solid
