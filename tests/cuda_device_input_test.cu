#include "prefetch_fakes.hpp"
#include "solid/device_ops.cuh"
#include "solid/ggml_prefetch.hpp"
#include "solid/llama_adapter.hpp"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <string>

namespace {
using test_prefetch::require;
void checked(cudaError_t status,const char * operation) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct DeviceBuffer {
    void * data=nullptr;
    explicit DeviceBuffer(std::size_t bytes) { checked(cudaMalloc(&data,bytes),"allocate device input check"); }
    ~DeviceBuffer() { if (data) (void) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &)=delete;
    DeviceBuffer & operator=(const DeviceBuffer &)=delete;
};
struct Capture {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    Capture() { checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create device input stream"); }
    ~Capture() {
        if (stream) (void) cudaStreamSynchronize(stream);
        if (executable) (void) cudaGraphExecDestroy(executable);
        if (graph) (void) cudaGraphDestroy(graph);
        if (stream) (void) cudaStreamDestroy(stream);
    }
};
struct Observer {
    solid::GgmlPrefetch & cache;
    static bool evaluate(ggml_tensor * tensor,bool ask,void * userdata) {
        const bool observed=tensor->op==GGML_OP_FLASH_ATTN_EXT && std::strcmp(ggml_get_name(tensor),"solid_prefetched_flash_attention")==0;
        if (ask) return observed;
        if (observed) {
            auto & self=*static_cast<Observer *>(userdata);
            self.cache.completed(self.cache.attention_epoch());
        }
        return true;
    }
};
// Pure kernels replay with fresh inputs. Odd D, padded head strides, half-way
// rounding, subnormals, signed zero and overflow are checked bit for bit.
void operations() {
    constexpr int width=257,heads=2,tokens=3;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({1024*1024,nullptr,true}),ggml_free);
    auto * packed=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,width*2+3,heads,tokens);
    auto * k=ggml_view_3d(context.get(),packed,width,heads,tokens,packed->nb[1],packed->nb[2],0);
    auto * v=ggml_view_3d(context.get(),packed,width,heads,tokens,packed->nb[1],packed->nb[2],width*sizeof(float));
    auto * q=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,width,heads*3,tokens);
    auto * early=ggml_new_tensor_1d(context.get(),GGML_TYPE_F32,1);
    auto * token=solid::device_query_token(context.get(),q);
    auto * kv=solid::device_consolidate_kv(context.get(),k,v,q,early);
    auto * mask=ggml_new_tensor_1d(context.get(),GGML_TYPE_F16,79);
    auto * release=solid::device_release_mask(context.get(),mask,kv);
    constexpr int ring_capacity=7, local_tokens=8;
    DeviceBuffer ring(ring_capacity*heads*width*sizeof(solid::Half));
    auto * positions=ggml_new_tensor_1d(context.get(),GGML_TYPE_I32,tokens);
    auto * local=solid::device_local_keys(context.get(),kv,positions,
        {static_cast<solid::Half *>(ring.data),ring_capacity,solid::DeviceMemory::cuda},local_tokens);
    DeviceBuffer position_data(ggml_nbytes(positions)),local_data(ggml_nbytes(local));
    positions->data=position_data.data; local->data=local_data.data;
    DeviceBuffer input(ggml_nbytes(packed)),query(ggml_nbytes(q)),frontier(4),marker(4),output(ggml_nbytes(kv)),
        masks(ggml_nbytes(mask)),released(ggml_nbytes(release));
    packed->data=input.data; k->data=input.data; v->data=static_cast<float *>(input.data)+width;
    q->data=query.data; early->data=frontier.data; token->data=marker.data;
    kv->data=output.data; mask->data=masks.data; release->data=released.data;
    std::vector<float> values(ggml_nelements(packed),-91),queries(ggml_nelements(q),0.5f);
    std::vector<solid::Half> expected_mask(79),actual_mask(79),actual(ggml_nelements(kv));
    for (std::size_t i=0;i<expected_mask.size();++i) expected_mask[i]=static_cast<solid::Half>(i*811);
    checked(cudaMemcpy(query.data,queries.data(),ggml_nbytes(q),cudaMemcpyHostToDevice),"initialize device query");
    checked(cudaMemcpy(masks.data,expected_mask.data(),ggml_nbytes(mask),cudaMemcpyHostToDevice),"initialize mask bits");
    Capture capture;
    checked(cudaStreamBeginCapture(capture.stream,cudaStreamCaptureModeRelaxed),"capture pure input operations");
    checked(solid::cuda_device_op(token,capture.stream),"capture query frontier");
    checked(solid::cuda_device_op(kv,capture.stream),"capture FP16 KV");
    checked(solid::cuda_device_op(release,capture.stream),"capture KV release");
    checked(solid::cuda_device_op(local,capture.stream),"capture local Key window");
    checked(cudaStreamEndCapture(capture.stream,&capture.graph),"finish input capture");
    checked(cudaGraphInstantiate(&capture.executable,capture.graph,nullptr,nullptr,0),"instantiate input graph");
    const float edge[]{0.0f,-0.0f,0x1p-25f,0x1.8p-24f,0x1.ffcp-15f,1.00048828125f,1.00146484375f,65504.0f,65520.0f,-65520.0f};
    for (int epoch=0;epoch<3;++epoch) {
        const int first=epoch==0?0:epoch==1?2:12;
        const std::int32_t pos[]{first,first+1,first+2};
        std::vector<solid::Half> ring_keys(ring_capacity*heads*width);
        for (int p=std::max(0,first-ring_capacity);p<first;++p)
            for (int h=0;h<heads;++h) for (int d=0;d<width;++d)
                ring_keys[(p%ring_capacity*heads+h)*width+d]=static_cast<solid::Half>(1+p*131+h*19+d);
        checked(cudaMemcpy(ring.data,ring_keys.data(),ring_keys.size()*sizeof(solid::Half),cudaMemcpyHostToDevice),"initialize Key ring");
        checked(cudaMemcpy(position_data.data,pos,sizeof(pos),cudaMemcpyHostToDevice),"replace local positions");
        for (int p=0;p<tokens;++p) for (int h=0;h<heads;++h) for (int plane=0;plane<2;++plane) for (int d=0;d<width;++d)
            values[(p*heads+h)*(2*width+3)+plane*width+d]=edge[(p+d+h+plane+epoch)%10];
        checked(cudaMemcpy(input.data,values.data(),ggml_nbytes(packed),cudaMemcpyHostToDevice),"replace captured input");
        checked(cudaGraphLaunch(capture.executable,capture.stream),"replay input operations");
        checked(cudaStreamSynchronize(capture.stream),"complete input operations");
        checked(cudaMemcpy(actual.data(),output.data,ggml_nbytes(kv),cudaMemcpyDeviceToHost),"read FP16 bits");
        checked(cudaMemcpy(actual_mask.data(),released.data,ggml_nbytes(release),cudaMemcpyDeviceToHost),"read released mask bits");
        std::vector<solid::Half> local_result(ggml_nelements(local));
        checked(cudaMemcpy(local_result.data(),local_data.data,ggml_nbytes(local),cudaMemcpyDeviceToHost),"read local Key window");
        for (int h=0;h<heads;++h) for (int t=0;t<local_tokens;++t) for (int d=0;d<width;++d) {
            const int p=first+tokens-local_tokens+t;
            const auto expected=p<0?solid::Half{0}:p<first?ring_keys[(p%ring_capacity*heads+h)*width+d]:
                actual[((p-first)*heads+h)*2*width+d];
            require(local_result[(h*local_tokens+t)*width+d]==expected,"captured local Key prefix, ring wrap or current input");
        }
        float zero=1; checked(cudaMemcpy(&zero,marker.data,4,cudaMemcpyDeviceToHost),"read scalar token");
        require(zero==0 && actual_mask==expected_mask,"captured token or mask changed");
        for (int p=0;p<tokens;++p) for (int h=0;h<heads;++h) for (int plane=0;plane<2;++plane) for (int d=0;d<width;++d)
            require(actual[((p*heads+h)*2+plane)*width+d]==solid::to_half(values[(p*heads+h)*(2*width+3)+plane*width+d]),"CUDA FP16 conversion differs from SSD reference bits");
    }
}
// Model-free integration on the experiment machine: two actual CUDA/CPU split
// layers, private ranker stream, native completion and one allocated graph reused
// across decode positions. CPU scalar sparse attention is the numerical oracle.
void decode(int device,bool packed,solid::ScheduleMode mode) {
    const solid::KvGeometry g{2,2,128,2,32,64,solid::KvLayout::interleaved};
    const solid::SparseBudget budget{6,1,1,1};
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(device),ggml_backend_free),
        cpu(ggml_backend_cpu_init(),ggml_backend_free);
    require(gpu && cpu,"decode check backends");
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),4,budget,true,6,{}, {},
        std::make_unique<test_prefetch::Reader>(),6);
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(g),4,budget,true,6);
    session.set_ranker(solid::make_cuda_block_ranker(g,4,1,6,device,1));
    std::vector<float> q(512),k(256),v(256);
    const float scale=1/std::sqrt(128.0f);
    const auto fill=[&](int layer,int p) {
        for (std::size_t i=0;i<q.size();++i) q[i]=static_cast<float>(static_cast<int>((i*13+p*17)%41)-20)/32;
        for (std::size_t i=0;i<k.size();++i) {
            k[i]=static_cast<float>(static_cast<int>((i*7+p*11+layer*19)%29)-14)/16;
            v[i]=static_cast<float>(static_cast<int>((i*3+p*5+layer*17)%37)-18)/16;
        }
    };
    for (int p=0;p<6;++p) for (int layer=0;layer<2;++layer) {
        fill(layer,p); (void) session.process(layer,p,1,q,k,v,scale); (void) reference.process(layer,p,1,q,k,v,scale);
    }
    const auto before=session.stats();
    solid::GgmlPrefetch cache(session,gpu.get(),{},mode);
    require(cache.direct_memory()==solid::DeviceMemory::cuda,"actual CUDA direct mode missing");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({4*1024*1024,nullptr,true}),ggml_free);
    ggml_backend_t backends[]{gpu.get(),cpu.get()};
    std::unique_ptr<ggml_backend_sched,decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends,nullptr,2,1024,false,true),ggml_backend_sched_free);
    Observer observer{cache}; ggml_backend_sched_set_eval_callback(scheduler.get(),Observer::evaluate,&observer);
    auto * query=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,128,4,1); ggml_set_input(query);
    ggml_tensor * keys[2],* values[2],* outputs[2];
    std::shared_ptr<solid::AttentionFrame> frames[2];
    std::unique_ptr<solid::AttentionNode> nodes[2];
    auto * graph=ggml_new_graph_custom(context.get(),512,false);
    for (int layer=0;layer<2;++layer) {
        keys[layer]=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,128,2,1); values[layer]=ggml_dup_tensor(context.get(),keys[layer]);
        ggml_set_input(keys[layer]); ggml_set_input(values[layer]);
        auto * current_q=layer?ggml_add(context.get(),query,ggml_scale(context.get(),ggml_reshape_3d(context.get(),outputs[0],128,4,1),0)):query;
        auto * key=ggml_scale(context.get(),keys[layer],1),* value=values[layer];
        ggml_tensor * kv=nullptr;
        if (packed) {
            kv=ggml_concat(context.get(),key,value,0);
            key=ggml_view_3d(context.get(),kv,128,2,1,256*sizeof(float),512*sizeof(float),0);
            value=ggml_view_3d(context.get(),kv,128,2,1,256*sizeof(float),512*sizeof(float),128*sizeof(float));
        }
        frames[layer]=std::make_shared<solid::AttentionFrame>();
        nodes[layer]=std::make_unique<solid::AttentionNode>(session,frames[layer],layer,scale,nullptr,true,&cache);
        outputs[layer]=nodes[layer]->build(context.get(),current_q,key,value,kv,scheduler.get(),gpu.get());
        ggml_set_output(outputs[layer]);
        nodes[layer]->set_query_proof(layer!=0);
        if (layer) nodes[layer-1]->defer_completion();
        ggml_backend_sched_set_tensor_backend(scheduler.get(),nodes[layer]->host_node(),cpu.get());
        if (nodes[layer]->selection_node()) {
            ggml_backend_sched_set_tensor_backend(scheduler.get(),nodes[layer]->selection_node(),cpu.get());
            ggml_build_forward_expand(graph,nodes[layer]->selection_node());
        }
    }
    nodes[1]->defer_completion();
    ggml_build_forward_expand(graph,outputs[1]);
    require(ggml_backend_sched_alloc_graph(scheduler.get(),graph),"allocate CUDA split graph");
    for (int p=6;p<14;++p) {
        cache.begin_prefetch(0,p);
        std::vector<float> expected[2];
        for (int layer=0;layer<2;++layer) {
            fill(layer,p); frames[layer]->positions={static_cast<std::uint64_t>(p)};
            expected[layer]=reference.process(layer,p,1,q,k,v,scale);
            ggml_backend_tensor_set(query,q.data(),0,q.size()*sizeof(float));
            ggml_backend_tensor_set(keys[layer],k.data(),0,k.size()*sizeof(float));
            ggml_backend_tensor_set(values[layer],v.data(),0,v.size()*sizeof(float));
        }
        require(ggml_backend_sched_graph_compute(scheduler.get(),graph)==GGML_STATUS_SUCCESS,"CUDA direct decode graph");
        cache.process_complete();
        for (int layer=0;layer<2;++layer) {
            nodes[layer]->check(); std::vector<float> actual(512);
            ggml_backend_tensor_get(outputs[layer],actual.data(),0,actual.size()*sizeof(float));
            for (std::size_t i=0;i<actual.size();++i) require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[layer][i])<2e-3f,"CUDA direct attention differs from scalar sparse reference");
        }
    }
    cache.quiesce(); session.finish();
    const auto after=session.stats();
    require(cache.stats().current_device_rows==32 && cache.stats().current_device_bytes==32ull*g.dimension*2*sizeof(solid::Half),
        "CUDA decode did not copy current KV rows on device");
    require(after.direct_decode_queries==16 && after.ranking.device_builder_queries==16 && after.ranking.device_builder_appends==16 &&
        after.ranking.device_rank_queries>0 && after.ranking.query_upload_bytes==before.ranking.query_upload_bytes &&
        after.ranking.builder_key_upload_bytes==before.ranking.builder_key_upload_bytes,"direct decode re-uploaded Q/K or skipped device inputs");
    require(after.ranking.builder_rank_shared_waits>0,"CUDA decode did not reuse the block ranking wait");
    const auto schedule=cache.schedule_stats();
    if (mode==solid::ScheduleMode::sync_reuse) {
        require(schedule.device_event_completions+schedule.device_boundary_fallbacks==16,
            "CUDA graph replay lost attention completion receipts");
        require(schedule.archive_attention_reuses==16 && schedule.archive_separate_completions==0,
            "CUDA KV archive did not share the attention completion");
        require(schedule.stream_waits==16 && schedule.batch_waits==0,
            "CUDA decode blocked the host on an attention upload batch");
        require(after.ranking.builder_compute_appends==16,"CUDA decode waited for representative key append on the host");
    } else if (mode==solid::ScheduleMode::overlap) {
        require(schedule.archive_attention_reuses==0 && schedule.archive_separate_completions==16,
            "CUDA overlap did not retain independent KV archive completions");
    }
    ggml_backend_sched_reset(scheduler.get());
}
}
int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") { std::cout<<"Use --run-device-check [device] on the experiment machine\n"; return 0; }
    try {
        if (argc>3) throw std::invalid_argument("unexpected device check argument");
        const int device=argc==3?std::stoi(argv[2]):0; checked(cudaSetDevice(device),"select device");
        operations();
        for (bool packed:{false,true}) for (auto mode:{solid::ScheduleMode::naive,solid::ScheduleMode::overlap,solid::ScheduleMode::sync_reuse}) decode(device,packed,mode);
        std::cout<<"CUDA borrowed Q/K, FP16 kernels, split graphs and replay checks passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
