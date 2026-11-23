#include "solid/attention_events.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#ifdef __linux__
#include <poll.h>
#endif

namespace {
void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F work) { bool failed=false; try { work(); } catch (const std::exception &) { failed=true; } require(failed,"invalid device receipt accepted"); }
class Events final:public solid::AttentionEvents {
public:
    Events():AttentionEvents(2) {}
    bool error=false;
    bool copy_ready(ggml_backend_event *) const override { return true; }
    const char * name() const override { return "cpu_protocol"; }
protected:
    std::uint64_t elapsed_ns(std::uint32_t) const override { if (error) throw std::runtime_error("event timing failure"); return 42; }
};
void publish(solid::StreamCompletionToken * token) { token->notify(token->userdata); }
void binding() {
    Events a,b;
    std::array<unsigned char,64> params{};
    for (std::size_t i=0;i<32;++i) params[i]=static_cast<unsigned char>(i);
    const auto prefix=params;
    solid::bind_stream_completion(params.data(),params.size(),a.token(0));
    require(solid::stream_completion(params.data(),params.size())==a.token(0),"completion binding lost its owner");
    require(std::memcmp(params.data(),prefix.data(),32)==0,"completion binding changed Flash Attention parameters");
    const auto before=params;
    solid::bind_stream_completion(params.data(),params.size(),b.token(0));
    require(params!=before && a.token(0)->identity!=b.token(0)->identity,"graph properties cannot distinguish registration replacement");
    const auto identity=b.token(0)->identity;
    ++b.token(0)->identity;
    require(!solid::stream_completion(params.data(),params.size()),"stale graph registration was accepted");
    b.token(0)->identity=identity;
    solid::bind_stream_completion(params.data(),params.size(),nullptr);
    require(!solid::stream_completion(params.data(),params.size()) && !solid::stream_completion(params.data(),32),"empty binding accepted");
}
void receipts() {
    Events events;
    events.arm(0,1); require(!events.poll(0,1),"unrecorded event became ready");
    events.boundary(solid::AttentionProof::process_boundary,999);
    require(!events.poll(0,1),"stale observer epoch released a current frame");
    rejects([&] { events.arm(1,2); }); rejects([&] { events.poll(1,1); });
    events.acknowledge();
    std::thread completion([&] { publish(events.token(0)); });
#ifdef __linux__
    pollfd fd{events.completion_handle(),POLLIN,0}; const auto status=::poll(&fd,1,5000);
    completion.join(); require(status==1 && (fd.revents&POLLIN),"completion notification lost a wakeup");
#else
    completion.join();
#endif
    events.acknowledge();
    const auto first=events.poll(0,1);
    require(first && first->proof==solid::AttentionProof::device_signal && first->device_ns==42,"device event proof or timing lost");
    events.boundary(solid::AttentionProof::attention_boundary,1);
    require(events.boundary_seen(1) && !events.handoff_seen(1),"attention receipt stopped next-layer prefetch");
    events.boundary(solid::AttentionProof::query_boundary,1);
    require(events.handoff_seen(1),"next query did not hand off prefetch coordination");
    publish(events.token(0)); require(!events.poll(0,1),"duplicate completion counted twice");
    rejects([&] { events.arm(0,1); });
    events.arm(1,2); events.boundary(solid::AttentionProof::query_boundary);
    const auto query=events.poll(1,2);
    require(query && query->proof==solid::AttentionProof::query_boundary && query->device_ns==0,"query boundary forged device timing");
    events.arm(0,3); publish(events.token(1)); require(!events.poll(0,3),"other layer's old callback ended current attention");
    events.boundary(solid::AttentionProof::process_boundary);
    require(events.poll(0,3)->proof==solid::AttentionProof::process_boundary,"process boundary proof");
    events.arm(1,4); events.stop(); require(events.stopped(),"wait cancellation lost");
    rejects([&] { events.arm(0,5); });
    events.retire_after_sync(); events.arm(0,5); require(!events.stopped() && !events.boundary_seen(5) && !events.handoff_seen(5),"new frame inherited a stop or old boundary");
    publish(events.token(0)); events.error=true;
    rejects([&] { events.poll(0,5); }); events.error=false;
    require(events.poll(0,5).has_value(),"event error silently consumed its receipt");
}
}
int main() {
    try { binding(); receipts(); std::cout<<"stream completion bindings, notifications, epochs and boundary proofs passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
