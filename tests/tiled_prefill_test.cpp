#include "tiled_prefill_cases.hpp"
#include <iostream>
#ifdef SOLID_PREFILL_GRAPH_TEST
#include "solid/llama_adapter.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#endif

using namespace test_tiled;
namespace {
class FaultStore final:public solid::KvStore {
public:
    explicit FaultStore(solid::KvGeometry g):KvStore(g),memory(g) {}
    void append_interleaved(std::uint32_t layer,std::uint64_t p,const std::vector<solid::Half> & kv) override { memory.append_interleaved(layer,p,kv); }
    void read(std::uint32_t layer,std::uint32_t head,std::uint64_t block,solid::KvBlock & out) override {
        if (fail && ++reads>2) throw std::runtime_error("injected storage read");
        memory.read(layer,head,block,out);
    }
    void flush() override {}
    std::uint64_t length(std::uint32_t layer) const override { return memory.length(layer); }
    std::uint64_t resident_bytes() const override { return memory.resident_bytes(); }
    solid::MemoryKvStore memory;
    bool fail=false;
    int reads=0;
};
void error_paths() {
    const solid::KvGeometry g{1,1,2,2,32,64};
    const solid::SparseBudget budget{8,1,3,1};
    for (int failure:{1,2,3,4,5}) {
        solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),2,budget,false,12);
        int aborts=0;
        auto backend=std::make_unique<Emulator>(g,2,3,4); backend->failure=failure; backend->abort_count=&aborts;
        session.set_prefill(std::move(backend));
        rejects([&] { (void) session.process(0,0,2,std::vector<float>(8,1),std::vector<float>(4,1),std::vector<float>(4,2),0.5f); });
        require(aborts==1,"prefill failure did not drain backend");
        rejects([&] { (void) session.process(0,2,1,{1,1,1,1},{1,1},{2,2},0.5f); });
    }
    auto store=std::make_unique<FaultStore>(g); auto * inspect=store.get();
    solid::AttentionSession session(std::move(store),2,budget,false,12);
    (void) session.process(0,0,8,std::vector<float>(32,1),std::vector<float>(16,1),std::vector<float>(16,2),0.5f);
    int aborts=0;
    auto backend=std::make_unique<Emulator>(g,2,3,4); backend->abort_count=&aborts;
    session.set_prefill(std::move(backend)); inspect->fail=true;
    bool original=false;
    try { (void) session.process(0,8,3,std::vector<float>(12,1),std::vector<float>(6,1),std::vector<float>(6,2),0.5f); }
    catch (const std::exception & error) { original=std::string(error.what())=="injected storage read"; }
    require(original && aborts==1,"storage error lost or owned device frame leaked");
}
void protocol() {
    const solid::KvGeometry g{1,1,2,2,16,64};
    Emulator backend(g,2,3,4);
    rejects([&] { (void) backend.finish(); });
    backend.begin(0,1,0.5f,{1,1,1,1});
    rejects([&] { backend.begin(0,1,0.5f,{1,1,1,1}); });
    rejects([&] { backend.consume(1,1,std::vector<solid::Half>(16)); });
    backend.consume(0,1,std::vector<solid::Half>(16));
    (void) backend.finish();
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),2,{8,1,3,1},false,12);
    auto invalid=std::make_unique<Emulator>(g,2,3,4); invalid->capacity=3;
    rejects([&] { session.set_prefill(std::move(invalid)); });
    session.set_prefill(std::make_unique<Emulator>(g,2,3,4));
    rejects([&] { session.set_prefill(std::make_unique<Emulator>(g,2,3,4)); });
}
#ifdef SOLID_PREFILL_GRAPH_TEST
void graph() {
    constexpr int width=7,heads=2,qh=6,count=5;
    const solid::KvGeometry g{1,heads,width,2,16,64};
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),qh,{8,1,3,1},false,count);
    session.set_prefill(std::make_unique<Emulator>(g,qh,3,4));
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(g),qh,{8,1,3,1},false,count);
    std::vector<float> q(count*qh*width),k(count*heads*width),v(k.size());
    for (std::size_t i=0;i<q.size();++i) q[i]=(static_cast<int>(i%13)-6)/8.0f;
    for (std::size_t i=0;i<k.size();++i) { k[i]=(static_cast<int>(i%17)-8)/8.0f; v[i]=(i%19)/16.0f; }
    const auto expected=reference.process(0,0,count,q,k,v,0.25f);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    auto * qt=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,width,qh,count);
    auto * kt=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,width,heads,count);
    auto * vt=ggml_dup_tensor(ctx.get(),kt);
    auto frame=std::make_shared<solid::AttentionFrame>(); frame->positions={0,1,2,3,4};
    solid::AttentionNode operation(session,frame,0,0.25f);
    auto * out=operation.build(ctx.get(),qt,kt,vt);
    auto * gf=ggml_new_graph_custom(ctx.get(),128,false); ggml_build_forward_expand(gf,out);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()),ggml_backend_buffer_free);
    require(buffer!=nullptr,"prefill graph allocation");
    ggml_backend_tensor_set(qt,q.data(),0,q.size()*sizeof(float));
    ggml_backend_tensor_set(kt,k.data(),0,k.size()*sizeof(float));
    ggml_backend_tensor_set(vt,v.data(),0,v.size()*sizeof(float));
    require(ggml_backend_graph_compute(cpu.get(),gf)==GGML_STATUS_SUCCESS,"tiled prefill graph execution");
    operation.check();
    std::vector<float> actual(expected.size()); ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
    close(actual,expected);
}
#endif
} // namespace
int main() {
    try {
        run_cases([](auto g,auto qh,auto queries,auto tile) { return std::make_unique<Emulator>(g,qh,queries,tile); });
        error_paths(); protocol();
#ifdef SOLID_PREFILL_GRAPH_TEST
        graph();
#endif
        std::cout<<"tiled prefill contracts passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
