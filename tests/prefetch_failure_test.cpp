#include "deferred_copies.hpp"
#include "prefetch_fakes.hpp"
#include "solid/ggml_prefetch.hpp"
#include "solid/llama_adapter.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <condition_variable>
#include <iostream>
#include <mutex>

using namespace test_prefetch;
namespace {
struct Fault {
    DeferredCopies & copies;
    std::shared_ptr<solid::AttentionFrame> frame;
    static void change_position(ggml_tensor * dst, int ith, int, void * userdata) {
        if (ith) return;
        auto & self = *static_cast<Fault *>(userdata);
        self.copies.wait_for_upload();
        ++self.frame->positions.front(); // Reject late KV after history uploads were submitted.
        std::memcpy(dst->data,dst->src[0]->data,ggml_nbytes(dst));
    }
    static void inspect_mask(ggml_tensor * dst, int ith, int, void * userdata) {
        if (ith) return;
        static_cast<Fault *>(userdata)->copies.inspect();
        std::memcpy(dst->data,dst->src[0]->data,ggml_nbytes(dst));
    }
};
void failed_graph(solid::ScheduleMode mode) {
    DeferredCopies copies;
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    backend->device = &copies.device;
    ggml_backend_cpu_set_n_threads(backend.get(),1);
    const solid::KvGeometry g{1,1,128,1,16,64,solid::KvLayout::interleaved};
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),1,{4,1,1,1},true,4,{}, {},std::make_unique<Reader>(),6);
    std::vector<float> q(128,0.1f),k(128,0.2f),v(128,0.3f);
    for (std::uint64_t position = 0; position < 4; ++position) (void) session.process(0,position,1,q,k,v,0.125f);
    solid::GgmlPrefetch cache(session,backend.get(),{},mode);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({1024*1024,nullptr,true}),ggml_free);
    auto * query = ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,128,1,1);
    auto * key = ggml_dup_tensor(context.get(),query), * value = ggml_dup_tensor(context.get(),query);
    auto frame = std::make_shared<solid::AttentionFrame>(); frame->positions = {4};
    Fault fault{copies,frame};
    auto * late_key = ggml_custom_4d(context.get(),GGML_TYPE_F32,128,1,1,1,&key,1,Fault::change_position,1,&fault);
    solid::AttentionNode node(session,frame,0,0.125f,nullptr,true,&cache);
    auto * result = node.build(context.get(),query,late_key,value);
    auto * mask = node.host_node();
    auto * inspected = ggml_custom_4d(context.get(),GGML_TYPE_F16,mask->ne[0],1,1,1,&mask,1,Fault::inspect_mask,1,&fault);
    node.attention_node()->src[3] = ggml_view_3d(context.get(),inspected,cache.tokens_per_head(),1,1,
        cache.tokens_per_head()*sizeof(solid::Half),cache.tokens_per_head()*sizeof(solid::Half),0);
    auto * graph = ggml_new_graph_custom(context.get(),128,false);
    ggml_build_forward_expand(graph,node.selection_node());
    ggml_build_forward_expand(graph,result);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(),backend.get()),ggml_backend_buffer_free);
    require(buffer != nullptr,"failure graph allocation");
    ggml_backend_tensor_set(query,q.data(),0,q.size()*sizeof(float));
    ggml_backend_tensor_set(key,k.data(),0,k.size()*sizeof(float));
    ggml_backend_tensor_set(value,v.data(),0,v.size()*sizeof(float));
    cache.begin_prefetch(0,4);
    require(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS,"failure graph could not finish its cleanup nodes");
    bool preserved = false;
    try { node.check(); } catch (const std::exception & error) { preserved = std::string(error.what()).find("KV arrived without its early query") != std::string::npos; }
    cache.quiesce(); session.finish();
    require(preserved,"cleanup replaced the original graph error");
    require(copies.submitted > 0 && copies.completed == copies.submitted,"test did not retain and drain delayed uploads");
    require(!copies.observed_pending,"attention received an error mask while a copy stream still owned the KV bank");
}
}
int main() {
    try {
        failed_graph(solid::ScheduleMode::overlap);
        failed_graph(solid::ScheduleMode::sync_reuse);
        std::cout << "failed attention drains independent uploads and preserves its first error\n"; return 0;
    }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
