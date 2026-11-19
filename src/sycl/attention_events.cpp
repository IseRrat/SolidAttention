#include "solid/attention_events.hpp"
#include "solid/device_timing.hpp"
#include "solid/sycl_flash_decode.hpp"
#include "ggml-backend-impl.h"

#if __has_include(<sycl/ext/oneapi/experimental/profiling_tag.hpp>)
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#define SOLID_SYCL_PROFILING_TAG 1
#endif

namespace solid {
namespace {

struct Frontier {
    sycl::queue * notifications = nullptr;
    sycl::event begin, end;
    bool timed = false;
};

std::uint64_t elapsed_tags(const sycl::event & begin, const sycl::event & end) {
    const auto start_ns=begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto finish_ns=end.get_profiling_info<sycl::info::event_profiling::command_start>();
    if (finish_ns<start_ns) throw std::runtime_error("invalid SYCL device interval");
    return finish_ns-start_ns;
}

#ifdef SOLID_SYCL_PROFILING_TAG
class SyclDeviceInterval final : public DeviceInterval {
public:
    void record(void * queue,bool end) override {
        events_[end]=sycl::ext::oneapi::experimental::submit_profiling_tag(*static_cast<sycl::queue *>(queue));
        recorded_[end]=true;
    }
    std::optional<std::uint64_t> duration() override {
        if (!recorded_[0] || !recorded_[1]) return {};
        return elapsed_tags(events_[0],events_[1]);
    }
private:
    sycl::event events_[2];
    bool recorded_[2]{};
};
#endif

class SyclAttentionEvents final : public AttentionEvents {
public:
    SyclAttentionEvents(std::uint32_t layers,const sycl::queue & compute) : AttentionEvents(layers),
        notifications_(compute.get_context(),compute.get_device()), frontiers_(layers) {
        for (std::uint32_t i=0;i<layers;++i) {
            frontiers_[i].notifications=&notifications_;
#ifdef SOLID_SYCL_PROFILING_TAG
            frontiers_[i].timed=compute.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag);
#endif
            token(i)->end_event=&frontiers_[i];
        }
    }
    ~SyclAttentionEvents() override { notifications_.wait(); }

    bool copy_ready(ggml_backend_event * event) const override {
        return static_cast<sycl::event *>(event->context)->get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete;
    }
    const char * name() const override { return "sycl_queue_events"; }

protected:
    std::uint64_t elapsed_ns(std::uint32_t layer) const override {
        const auto & frontier=frontiers_.at(layer);
        auto end_event=frontier.end;
        end_event.wait_and_throw();
        if (!frontier.timed) return 0;
        return elapsed_tags(frontier.begin,frontier.end);
    }

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
#ifdef SOLID_SYCL_PROFILING_TAG
    if (frontier.timed) frontier.begin=sycl::ext::oneapi::experimental::submit_profiling_tag(queue);
#endif
    frontier.end=sycl_masked_decode(queue,output);
#ifdef SOLID_SYCL_PROFILING_TAG
    if (frontier.timed) frontier.end=sycl::ext::oneapi::experimental::submit_profiling_tag(queue);
#endif
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
std::unique_ptr<DeviceInterval> make_sycl_device_interval(void * backend_queue) {
#ifdef SOLID_SYCL_PROFILING_TAG
    const auto & queue=*static_cast<sycl::queue *>(backend_queue);
    if (queue.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag)) return std::make_unique<SyclDeviceInterval>();
#else
    (void) backend_queue;
#endif
    return {};
}
} // namespace solid
