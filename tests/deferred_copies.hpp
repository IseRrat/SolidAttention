#pragma once
#include "prefetch_fakes.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <condition_variable>
#include <mutex>

namespace test_prefetch {
// A CPU allocation with a separate stream whose writes remain pending until
// an event or drain explicitly completes them. No GPU executes in this test.
struct DeferredCopies {
    struct Copy { ggml_tensor * tensor; const void * source; std::size_t offset, bytes; std::uint64_t sequence; };
    struct Event { DeferredCopies * owner; std::uint64_t through = 0; };
    ggml_backend_device device{};
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Copy> pending;
    std::uint64_t submitted = 0, completed = 0;
    bool observed_pending = false;
    bool fail_device_copy=false;
    std::function<void()> on_upload;
    DeferredCopies() {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "deferred-cpu-test"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); };
        device.iface.get_host_buffer_type = device.iface.get_buffer_type;
        device.iface.init_backend = [](ggml_backend_dev_t dev, const char *) {
            auto * stream = new ggml_backend{};
            stream->device = dev; stream->context = dev->context;
            stream->iface.get_name = [](ggml_backend_t) { return "deferred-copy-stream"; };
            stream->iface.free = [](ggml_backend_t backend) { delete backend; };
            stream->iface.set_tensor_async = [](ggml_backend_t backend, ggml_tensor * tensor, const void * source, std::size_t offset, std::size_t bytes) {
                auto & self = *static_cast<DeferredCopies *>(backend->context);
                { std::lock_guard<std::mutex> lock(self.mutex); self.pending.push_back({tensor,source,offset,bytes,++self.submitted}); }
                self.cv.notify_all();
                if (self.on_upload) self.on_upload();
            };
            stream->iface.cpy_tensor_async=[](ggml_backend_t source_backend,ggml_backend_t target_backend,const ggml_tensor * source,ggml_tensor * target) {
                auto & self=*static_cast<DeferredCopies *>(target_backend->context);
                require(source_backend==target_backend,"device row copy used the compute stream");
                { std::lock_guard<std::mutex> lock(self.mutex); self.pending.push_back({target,source->data,0,ggml_nbytes(target),++self.submitted}); }
                self.cv.notify_all();
                if (self.fail_device_copy) throw std::runtime_error("injected current device copy failure");
                if (self.on_upload) self.on_upload();
                return true;
            };
            stream->iface.synchronize = [](ggml_backend_t backend) { static_cast<DeferredCopies *>(backend->context)->flush(UINT64_MAX); };
            stream->iface.event_record = [](ggml_backend_t backend, ggml_backend_event_t event) {
                auto & self = *static_cast<DeferredCopies *>(backend->context);
                std::lock_guard<std::mutex> lock(self.mutex);
                static_cast<Event *>(event->context)->through = self.submitted;
            };
            return stream;
        };
        device.iface.event_new = [](ggml_backend_dev_t dev) {
            return new ggml_backend_event{dev,new Event{static_cast<DeferredCopies *>(dev->context)}};
        };
        device.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t event) { delete static_cast<Event *>(event->context); delete event; };
        device.iface.event_synchronize = [](ggml_backend_dev_t, ggml_backend_event_t event) {
            auto & record = *static_cast<Event *>(event->context); record.owner->flush(record.through);
        };
    }
    void flush(std::uint64_t through) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = pending.begin(); it != pending.end();) {
            if (it->sequence > through) { ++it; continue; }
            ggml_backend_tensor_set(it->tensor,it->source,it->offset,it->bytes);
            ++completed; it = pending.erase(it);
        }
    }
    void wait_for_upload() {
        std::unique_lock<std::mutex> lock(mutex); cv.wait(lock,[&] { return !pending.empty(); });
    }
    void inspect() { std::lock_guard<std::mutex> lock(mutex); observed_pending = !pending.empty(); }
};
} // namespace test_prefetch
