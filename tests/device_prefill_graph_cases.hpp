// 设备 Prefill 图复用与输入输出共享检查的通用样例。
#pragma once
#include "solid/device_ops.hpp"
#include "solid/llama_adapter.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace test_prefill_graph {
inline void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
template<class Prefill,class Ranker> void run(ggml_backend_t gpu,ggml_backend_t cpu,solid::DeviceMemory memory,
                                             Prefill prefill,Ranker ranker,bool sparse,bool packed,bool disk,bool failure=false,bool strided=false) {
    const solid::KvGeometry g{2,2,7,4,32,64,solid::KvLayout::interleaved};
    const solid::SparseBudget budget{12,2,3,2}; constexpr int count=5,qw=42,kw=14;
    std::unique_ptr<solid::KvStore> store;
    if (disk) {
        const auto path=std::filesystem::temp_directory_path()/("solid-prefill-graph-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        store=std::make_unique<solid::FileKvStore>(g,solid::create_session_file(path,true),2048);
    } else store=std::make_unique<solid::MemoryKvStore>(g);
    solid::AttentionSession session(std::move(store),6,budget,sparse,15),reference(std::make_unique<solid::MemoryKvStore>(g),6,budget,sparse,15);
    session.set_prefill(prefill(g,6,3,7));
    if (sparse) session.set_ranker(ranker(g,6,budget));
    require(session.prefill_direct_memory()==memory,"prefill device capability");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({4*1024*1024,nullptr,true}),ggml_free);
    ggml_backend_t backends[]{gpu,cpu}; const auto backends_count=gpu==cpu?1:2;
    std::unique_ptr<ggml_backend_sched,decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends,nullptr,backends_count,1024,false,true),ggml_backend_sched_free);
    require(scheduler!=nullptr,"prefill graph scheduler");
    auto * q=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,7,6,count); ggml_set_input(q);
    ggml_tensor * keys[2],* values[2],* outputs[2];
    std::shared_ptr<solid::AttentionFrame> frames[2]; std::unique_ptr<solid::AttentionNode> nodes[2];
    auto * graph=ggml_new_graph_custom(context.get(),512,false);
    for (int layer=0;layer<2;++layer) {
        keys[layer]=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,strided?2:7,strided?7:2,count); values[layer]=ggml_dup_tensor(context.get(),keys[layer]);
        ggml_set_input(keys[layer]); ggml_set_input(values[layer]);
        auto * query=layer?ggml_add(context.get(),q,ggml_scale(context.get(),ggml_reshape_3d(context.get(),outputs[0],7,6,count),0)):q;
        auto * k=strided?ggml_permute(context.get(),keys[layer],1,0,2,3):ggml_scale(context.get(),keys[layer],1);
        auto * v=strided?ggml_permute(context.get(),values[layer],1,0,2,3):values[layer]; ggml_tensor * kv=nullptr;
        if (packed) {
            kv=ggml_concat(context.get(),k,v,0);
            k=ggml_view_3d(context.get(),kv,7,2,count,14*sizeof(float),28*sizeof(float),0);
            v=ggml_view_3d(context.get(),kv,7,2,count,14*sizeof(float),28*sizeof(float),7*sizeof(float));
        }
        frames[layer]=std::make_shared<solid::AttentionFrame>();
        nodes[layer]=std::make_unique<solid::AttentionNode>(session,frames[layer],layer,0.25f);
        outputs[layer]=nodes[layer]->build(context.get(),query,k,v,kv,scheduler.get(),gpu);
        ggml_set_output(outputs[layer]); ggml_backend_sched_set_tensor_backend(scheduler.get(),nodes[layer]->host_node(),cpu);
        require(solid::device_op(outputs[layer])==solid::DeviceOp::prefill_result &&
            outputs[layer]->view_src==outputs[layer]->src[0] && solid::graph_depends_on(outputs[layer],nodes[layer]->host_node()),"prefill output view/control edge");
        require(outputs[layer]->src[2]==nodes[layer]->host_node()->src[0]->src[2] &&
            outputs[layer]->src[3]==nodes[layer]->host_node()->src[0],"prefill borrowed Q/KV lifetime edges");
    }
    ggml_build_forward_expand(graph,outputs[1]);
    require(ggml_backend_sched_alloc_graph(scheduler.get(),graph),"allocate prefill split graph");
    for (auto * output:outputs) require(output->data==output->src[0]->data,"prefill result was copied instead of forwarded");
    std::vector<float> query(count*qw),key(count*kw),value(key.size());
    const auto fill=[&](int layer,int first) {
        for (std::size_t i=0;i<query.size();++i) query[i]=(static_cast<int>((i*7+first*11)%41)-20)/31.0f;
        for (std::size_t i=0;i<key.size();++i) {
            key[i]=(static_cast<int>((i*3+layer*5+first*13)%29)-14)/19.0f;
            value[i]=(static_cast<int>((i*5+layer*7+first*17)%37)-18)/13.0f;
        }
    };
    for (int first=0;first<15;first+=count) {
        std::vector<float> expected[2];
        for (int layer=0;layer<2;++layer) {
            fill(layer,first); expected[layer]=reference.process(layer,first,count,query,key,value,0.25f);
            frames[layer]->positions.clear(); for (int i=0;i<count;++i) frames[layer]->positions.push_back(first+i);
            if (failure && !layer) key[0]=NAN;
            ggml_backend_tensor_set(q,query.data(),0,query.size()*sizeof(float));
            if (strided) {
                std::vector<float> key_input(key.size()),value_input(value.size());
                for (int p=0;p<count;++p) for (int h=0;h<2;++h) for (int d=0;d<7;++d) {
                    key_input[(p*7+d)*2+h]=key[(p*2+h)*7+d]; value_input[(p*7+d)*2+h]=value[(p*2+h)*7+d];
                }
                ggml_backend_tensor_set(keys[layer],key_input.data(),0,key_input.size()*sizeof(float));
                ggml_backend_tensor_set(values[layer],value_input.data(),0,value_input.size()*sizeof(float));
            } else {
                ggml_backend_tensor_set(keys[layer],key.data(),0,key.size()*sizeof(float));
                ggml_backend_tensor_set(values[layer],value.data(),0,value.size()*sizeof(float));
            }
        }
        require(ggml_backend_sched_graph_compute(scheduler.get(),graph)==GGML_STATUS_SUCCESS,"execute prefill split graph");
        for (int layer=0;layer<2;++layer) {
            std::vector<float> actual(count*qw); ggml_backend_tensor_get(outputs[layer],actual.data(),0,actual.size()*sizeof(float));
            if (failure) {
                bool rejected=false; try { nodes[layer]->check(); } catch (const std::exception &) { rejected=true; }
                require(rejected,"prefill graph failure was lost");
                for (auto x:actual) require(x==0,"failed prefill exposed an unwritten device output");
            } else {
                nodes[layer]->check();
                for (std::size_t i=0;i<actual.size();++i) require(std::isfinite(actual[i]) &&
                    std::abs(actual[i]-expected[layer][i])<3e-4f,"prefill device graph differs from scalar causal reference");
            }
        }
        if (failure) break;
        session.release_prefill_if_complete();
    }
    const auto stats=session.stats();
    if (!failure) {
        require(stats.direct_prefill_queries==30 && stats.prefill.device_query_tokens==30 && stats.prefill.device_kv_tokens==48 &&
            stats.prefill.query_upload_bytes==0 && stats.prefill.output_download_bytes==0 && stats.prefill.output_device_copy_bytes==30*qw*4,
            "prefill graph transferred Q or output through host");
        require(stats.prefill.workspace_device_bytes==0 && stats.prefill_host_bytes==0 && stats.prefill_graph_device_bytes>=2*count*qw*4,
            "graph output lifetime or pool release accounting");
        if (sparse) require(stats.ranking.device_builder_batches==12 && stats.ranking.device_builder_queries==30 &&
            stats.ranking.query_upload_bytes==0 && stats.ranking.builder_key_upload_bytes==0,"prefill graph representative batches or Q/K uploads");
        std::vector<float> qrow(qw,0.5f),krow(kw,0.25f),vrow(kw,0.125f);
        for (std::uint32_t layer=0;layer<2;++layer) {
            const auto actual=session.process(layer,15,1,qrow,krow,vrow,0.25f),expected=reference.process(layer,15,1,qrow,krow,vrow,0.25f);
            for (std::size_t i=0;i<actual.size();++i) require(std::abs(actual[i]-expected[i])<3e-4f,"prefill/decode representative state differs");
        }
        session.finish();
    }
    ggml_backend_sched_reset(scheduler.get()); nodes[0].reset(); nodes[1].reset();
    require(session.stats().prefill_graph_device_bytes==0 && session.stats().prefill_graph_device_peak_bytes>0,"prefill graph buffers leaked");
}
} // namespace test_prefill_graph
