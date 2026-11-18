#include "deferred_copies.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/llama_adapter.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <future>
#include <atomic>
#ifdef __linux__
#include <sys/eventfd.h>
#include <unistd.h>
#include <cerrno>
#endif

using namespace test_prefetch;
namespace {
struct Flow {
    std::mutex mutex;
    std::condition_variable cv;
    bool receipt=false,upload=false;
    void received() { std::lock_guard<std::mutex> lock(mutex); receipt=true; }
    void uploaded() {
        { std::lock_guard<std::mutex> lock(mutex); if (receipt) upload=true; }
        cv.notify_all();
    }
    static void ffn(ggml_tensor * destination,int ith,int,void * userdata) {
        if (ith) return;
        auto & self=*static_cast<Flow *>(userdata);
        std::unique_lock<std::mutex> lock(self.mutex);
        const bool progress=self.cv.wait_for(lock,std::chrono::seconds(5),[&] { return self.upload; });
        require(progress,"next-layer H2D did not advance while FFN waited");
        std::memcpy(destination->data,destination->src[0]->data,ggml_nbytes(destination));
    }
};
class Events final:public solid::AttentionEvents {
public:
    Events(std::uint32_t layers,Flow * flow=nullptr):AttentionEvents(layers),flow(flow) {}
    bool timing_error=false;
    bool copy_ready(ggml_backend_event * event) const override {
        auto & record=*static_cast<DeferredCopies::Event *>(event->context); record.owner->flush(record.through); return true;
    }
    bool host_marker() const override { return true; }
    const char * name() const override { return "cpu_protocol"; }
protected:
    std::uint64_t elapsed_ns(std::uint32_t) const override {
        if (timing_error) throw std::runtime_error("injected native timing failure");
        if (flow) flow->received(); return 13;
    }
private:
    Flow * flow;
};
struct BackendDevice {
    ggml_backend_t backend;
    ggml_backend_dev_t original;
    void (*original_wait)(ggml_backend_t,ggml_backend_event_t);
    BackendDevice(ggml_backend_t b,ggml_backend_dev_t d):backend(b),original(b->device),original_wait(b->iface.event_wait) {
        b->device=d;
        b->iface.event_wait=[](ggml_backend_t,ggml_backend_event_t event) {
            auto & record=*static_cast<DeferredCopies::Event *>(event->context); record.owner->flush(record.through);
        };
    }
    ~BackendDevice() { backend->device=original; backend->iface.event_wait=original_wait; }
};
const solid::KvGeometry geometry{2,1,64,1,16,64,solid::KvLayout::interleaved};
const solid::SparseBudget budget{2,1,1,1};
void seed(solid::AttentionSession & session) {
    std::vector<float> q(64,.125f),k(64,.25f),v(64,.5f);
    for (std::uint64_t p=0;p<4;++p) for (std::uint32_t layer=0;layer<2;++layer) (void) session.process(layer,p,1,q,k,v,.125f);
}
void ffn_overlap() {
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(geometry),1,budget,true,4,{}, {},std::make_unique<Reader>(),4);
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(geometry),1,budget,true,4); seed(session); seed(reference);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    DeferredCopies copies; BackendDevice restore(backend.get(),&copies.device);
    Flow flow; copies.on_upload=[&] { flow.uploaded(); };
    auto events=std::make_unique<Events>(geometry.layers,&flow);
    solid::GgmlPrefetch cache(session,backend.get(),{},solid::ScheduleMode::sync_reuse,{},-1,std::move(events));
    auto frame=std::make_shared<solid::AttentionFrame>(); frame->positions={4};
    solid::AttentionNode node(session,frame,0,.125f,{},true,&cache);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({2*1024*1024,nullptr,true}),ggml_free);
    auto * q=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,64,1,1);
    auto * k=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,64,1,1);
    auto * v=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,64,1,1);
    auto * output=node.build(context.get(),q,k,v);
    auto * ffn=ggml_custom_4d(context.get(),GGML_TYPE_F32,64,1,1,1,&output,1,Flow::ffn,1,&flow);
    auto * graph=ggml_new_graph_custom(context.get(),128,false);
    ggml_build_forward_expand(graph,node.selection_node()); ggml_build_forward_expand(graph,ffn);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(context.get(),backend.get()),ggml_backend_buffer_free);
    require(buffer!=nullptr,"native graph allocation");
    std::vector<float> query(64,.125f),key(64,.25f),value(64,.75f);
    ggml_backend_tensor_set(q,query.data(),0,256); ggml_backend_tensor_set(k,key.data(),0,256); ggml_backend_tensor_set(v,value.data(),0,256);
    cache.begin_prefetch(0,4);
    require(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"native synthetic graph failed");
    cache.process_complete(); node.check();
    const auto expected=reference.process(0,4,1,query,key,value,.125f);
    std::vector<float> actual(64); ggml_backend_tensor_get(ffn,actual.data(),0,256);
    for (std::size_t i=0;i<64;++i) require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<3e-4f,"native completion changed attention math");
    const auto stats=cache.schedule_stats();
    require(flow.upload && stats.device_event_completions==1 && stats.uploads_before_boundary>0 && stats.attention_fences==0,
            "completion failed to release the bank and start H2D before the query/process boundary");
    require(stats.copy_event_queries>0,"background H2D receipts were never queried");
    cache.quiesce(); session.finish();
    require(copies.pending.empty() && copies.completed==copies.submitted,"native graph left live source copies");
}
void interrupted(bool timing_error) {
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(geometry),1,budget,true,4,{}, {},std::make_unique<Reader>(),4); seed(session);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    DeferredCopies copies; BackendDevice restore(backend.get(),&copies.device);
    auto events=std::make_unique<Events>(geometry.layers); auto * observe=events.get(); observe->timing_error=timing_error;
    solid::GgmlPrefetch cache(session,backend.get(),{},solid::ScheduleMode::sync_reuse,{},-1,std::move(events));
    std::vector<float> query(64,.125f),key(64,.25f),value(64,.75f);
    cache.select_early(0,4,query,false); (void) cache.decode(0,4,{},key,value,false);
    if (timing_error) {
        auto * token=observe->token(0); token->notify(token->userdata);
        bool preserved=false;
        try { cache.process_complete(); } catch (const std::exception & error) { preserved=std::string(error.what()).find("injected native timing failure")!=std::string::npos; }
        require(preserved,"native worker error was lost");
    }
    // The stream marker was never executed in the canceled case. stop() must
    // wake the coordinator before a full backend fence drains its sources.
    cache.quiesce(); session.finish();
    require(copies.pending.empty() && copies.completed==copies.submitted,"native cancellation left live writers");
}
#ifdef __linux__
struct ReadGate {
    std::atomic<bool> opened{true};
    int fd=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
    ReadGate() { require(fd>=0,"read gate notification"); }
    ~ReadGate() { close(fd); }
    void open(bool value) noexcept {
        opened.store(value,std::memory_order_release); const std::uint64_t count=1;
        while (write(fd,&count,sizeof(count))<0 && errno==EINTR) {}
    }
};
class GatedReads final:public solid::AsyncReader {
public:
    explicit GatedReads(std::shared_ptr<ReadGate> gate):gate(std::move(gate)),reader(solid::make_uring_reader(2)) {}
    bool submit(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset) override { return reader->submit(id,fd,destination,bytes,offset); }
    void cancel(std::uint64_t id) override { reader->cancel(id); }
    void flush() override { reader->flush(); }
    bool poll(solid::ReadCompletion & result,bool wait) override {
        require(!wait,"gated engine must use readiness");
        return gate->opened.load(std::memory_order_acquire) && reader->poll(result,false);
    }
    void drain() override { reader->drain(); }
    solid::AsyncIoStats stats() const override { return reader->stats(); }
    int completion_handle() const override { return gate->opened.load(std::memory_order_acquire)?reader->completion_handle():gate->fd; }
    void acknowledge_notifications() override {
        std::uint64_t count;
        while (read(gate->fd,&count,sizeof(count))==sizeof(count)) {}
        reader->acknowledge_notifications();
    }
private:
    std::shared_ptr<ReadGate> gate;
    std::unique_ptr<solid::AsyncReader> reader;
};
void query_handoff() {
    if (!solid::uring_available()) return;
    auto gate=std::make_shared<ReadGate>();
    const auto path=std::filesystem::temp_directory_path()/("solid-native-handoff-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto store=std::make_unique<solid::FileKvStore>(geometry,solid::create_session_file(path,true),2048);
    auto reads=solid::make_io_worker([gate] { return std::make_unique<GatedReads>(gate); },8);
    solid::AttentionSession session(std::move(store),1,budget,true,4,{}, {},std::move(reads),4); seed(session); session.flush();
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    DeferredCopies copies; BackendDevice restore(backend.get(),&copies.device);
    copies.on_upload=[&] { std::lock_guard<std::mutex> lock(copies.mutex); if (copies.submitted==2) gate->open(false); };
    auto events=std::make_unique<Events>(geometry.layers); auto * observe=events.get();
    solid::GgmlPrefetch cache(session,backend.get(),{},solid::ScheduleMode::sync_reuse,{},-1,std::move(events));
    struct OpenOnExit { std::shared_ptr<ReadGate> gate; ~OpenOnExit() { gate->open(true); } } release{gate};
    std::vector<float> query(64,.125f),key(64,.25f),value(64,.75f);
    cache.select_early(0,4,query,false); (void) cache.decode(0,4,{},key,value,false);
    require(!gate->opened.load(std::memory_order_acquire),"next-layer READ completion gate was not engaged");
    auto * token=observe->token(0); token->notify(token->userdata);
    auto selection=std::async(std::launch::async,[&] { cache.select_early(1,4,query,true); });
    const bool prompt=selection.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    gate->open(true); selection.get();
    require(prompt,"formal selection waited for every predicted SSD completion");
    cache.quiesce(); session.finish();
    require(copies.pending.empty(),"query handoff lost copy source ownership");
}
#endif
}
int main() {
    try { ffn_overlap(); interrupted(false); interrupted(true);
#ifdef __linux__
        query_handoff();
#endif
        std::cout<<"native attention notification, FFN/H2D overlap, query handoff, cancellation and worker errors passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
