// 主机模拟设备 Q/KV 的 Prefill 数值与数据流检查。
#include "representative_builder_cases.hpp"
#include "tiled_prefill_cases.hpp"
#include <iostream>
#include <chrono>

namespace {
using test_tiled::require;
using test_tiled::rejects;
constexpr auto memory=solid::DeviceMemory::host_emulator;
class ReadFaultStore final:public solid::KvStore {
public:
    explicit ReadFaultStore(solid::KvGeometry g):KvStore(g),store(g) {}
    void append_interleaved(std::uint32_t layer,std::uint64_t p,const std::vector<solid::Half> & kv) override { store.append_interleaved(layer,p,kv); }
    void read(std::uint32_t layer,std::uint32_t head,std::uint64_t block,solid::KvBlock & output) override {
        if (fault && ++reads==2) throw std::runtime_error("injected device prefill history error");
        store.read(layer,head,block,output);
    }
    void flush() override {}
    std::uint64_t length(std::uint32_t layer) const override { return store.length(layer); }
    std::uint64_t resident_bytes() const override { return store.resident_bytes(); }
    solid::MemoryKvStore store;
    bool fault=false;
    int reads=0;
};
void representative_batches(bool ties,std::uint32_t attach) {
    const solid::KvGeometry g{2,2,7,4,77,64}; const solid::SparseBudget b{19,2,5,2};
    solid::RepresentativeIndex actual(g,6,b),reference(g,6,b);
    std::mt19937 random(17731);
    std::uint32_t first=0,batches=0;
    while (first<g.capacity_tokens) {
        if (first==attach) actual.set_ranker(std::make_unique<test_builder::Emulator>(g,6,b,true));
        auto count=std::min<std::uint32_t>(g.capacity_tokens-first,first<attach?attach-first:13);
        for (std::uint32_t layer=0;layer<g.layers;++layer) {
            std::vector<float> q(count*42); std::vector<solid::Half> kv(count*28),keys(14);
            for (auto & value:q) value=ties?0:(static_cast<int>(random()%43)-21)/16.0f;
            for (auto & value:kv) value=solid::to_half((static_cast<int>(random()%31)-15)/8.0f);
            for (std::uint32_t p=0;p<count;++p) {
                for (std::uint32_t h=0;h<2;++h) std::copy_n(kv.data()+p*28+h*14,7,keys.data()+h*7);
                const std::vector<float> query(q.begin()+p*42,q.begin()+(p+1)*42);
                reference.observe(layer,first+p,query,keys);
                if (first<attach) actual.observe(layer,first+p,query,keys);
            }
            if (first>=attach) { actual.observe_batch_device(layer,first,count,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory}); ++batches; }
            const std::vector<float> query(q.end()-42,q.end());
            for (std::uint64_t block=0;block<g.blocks_per_head();++block) {
                require(actual.ready(layer,block)==reference.ready(layer,block),"batch representative maturity");
                if (actual.ready(layer,block)) for (std::uint32_t h=0;h<2;++h) require(
                    actual.representative_positions(layer,h,block)==reference.representative_positions(layer,h,block) &&
                    actual.score(layer,h,block,query)==reference.score(layer,h,block,query),"batch representative keys or stable top-R differ");
            }
            const auto selected=solid::select_blocks(g,b,actual,layer,first+count-1,query);
            const auto expected=solid::select_blocks(g,b,reference,layer,first+count-1,query);
            for (std::uint32_t h=0;h<2;++h) require(selected[h].positions==expected[h].positions,"batch representative handoff changes selection");
        }
        first+=count;
    }
    require(actual.ranker_stats().device_builder_batches==batches,"batch submission count");
}
void session_case(bool sparse,bool disk,solid::KvLayout layout) {
    const solid::KvGeometry g{2,2,7,4,37,64,layout}; const solid::SparseBudget b{16,2,5,2};
    std::unique_ptr<solid::KvStore> store;
    if (disk) {
        const auto path=std::filesystem::temp_directory_path()/("solid-device-prefill-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        store=std::make_unique<solid::FileKvStore>(g,solid::create_session_file(path,true),2048);
    } else store=std::make_unique<solid::MemoryKvStore>(g);
    solid::AttentionSession actual(std::move(store),6,b,sparse,29),reference(std::make_unique<solid::MemoryKvStore>(g),6,b,sparse,29);
    actual.set_prefill(std::make_unique<test_tiled::Emulator>(g,6,3,7,true));
    if (sparse) actual.set_ranker(std::make_unique<test_builder::Emulator>(g,6,b,true));
    std::mt19937 random(4711); std::uint32_t first=0;
    for (const std::uint32_t count:{5,13,11}) {
        for (std::uint32_t layer=0;layer<g.layers;++layer) {
            std::vector<float> q(count*42),k(count*14),v(k.size()),output(q.size(),NAN);
            for (auto & x:q) x=(static_cast<int>(random()%43)-21)/17.0f;
            for (auto & x:k) x=(static_cast<int>(random()%37)-18)/19.0f;
            for (auto & x:v) x=(static_cast<int>(random()%47)-23)/13.0f;
            std::vector<solid::Half> kv(count*28);
            for (std::uint32_t p=0;p<count;++p) for (std::uint32_t h=0;h<2;++h) for (std::uint32_t d=0;d<7;++d) {
                kv[p*28+h*14+d]=solid::to_half(k[p*14+h*7+d]); kv[p*28+h*14+7+d]=solid::to_half(v[p*14+h*7+d]);
            }
            const auto expected=reference.process(layer,first,count,q,k,v,0.25f);
            actual.process_prefill_device(layer,first,count,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory},kv,0.25f,{output.data(),output.size(),memory});
            test_tiled::close(output,expected);
            actual.release_prefill_if_complete();
            if (first==18 && !layer) require(actual.stats().prefill_host_bytes>0,"prefill pools released before last layer");
        }
        first+=count;
    }
    const auto stats=actual.stats();
    require(stats.direct_prefill_queries==58 && stats.prefill.device_query_tokens==58 && stats.prefill.device_kv_tokens==160 &&
        stats.prefill.query_upload_bytes==0 && stats.prefill.output_download_bytes==0 && stats.prefill.output_device_copy_bytes==58*42*4 &&
        stats.prefill.upload_bytes==34*2*8*2*7*2 && stats.prefill_kv_download_bytes==58*28*2,"device prefill transfers or frame partitions");
    require(!actual.prefill_direct_memory() && stats.prefill_host_bytes==0 && stats.prefill.workspace_device_bytes==0,"prefill pool release");
    if (sparse) require(stats.ranking.device_builder_batches==22 && stats.ranking.device_builder_queries==58 && stats.ranking.device_builder_appends==58,
        "representative batch counts");
    for (std::uint32_t p=29;p<35;++p) for (std::uint32_t layer=0;layer<2;++layer) {
        std::vector<float> q(42,0.5f),k(14,0.25f),v(14,p/8.0f);
        test_tiled::close(actual.process(layer,p,1,q,k,v,0.25f),reference.process(layer,p,1,q,k,v,0.25f));
    }
    actual.finish();
}
void errors() {
    const solid::KvGeometry g{1,1,2,2,16,64}; const solid::SparseBudget b{8,1,3,1};
    std::vector<float> q(16,1),out(16,0); std::vector<solid::Half> kv(16,solid::to_half(0.5f));
    for (int failure:{1,2,3,4,5}) {
        solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),2,b,false,8);
        int aborts=0; auto backend=std::make_unique<test_tiled::Emulator>(g,2,3,4,true); backend->failure=failure; backend->abort_count=&aborts;
        session.set_prefill(std::move(backend));
        rejects([&] { session.process_prefill_device(0,0,4,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory},kv,0.5f,{out.data(),out.size(),memory}); });
        require(aborts==1,"device prefill failed without aborting readers");
        rejects([&] { (void) session.process(0,0,1,{1,1,1,1},{1,1},{1,1},0.5f); });
    }
    solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),2,b,false,8);
    session.set_prefill(std::make_unique<test_tiled::Emulator>(g,2,3,4,true));
    rejects([&] { session.process_prefill_device(0,0,4,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory},kv,0.5f,{q.data(),q.size(),memory}); });
    rejects([&] { solid::device_elements(UINT32_MAX,UINT32_MAX,UINT32_MAX,2); });
    rejects([&] { solid::device_ranges_overlap(q.data(),1,0,out.data(),1,sizeof(float)); });
    session.process_prefill_device(0,0,4,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory},kv,0.5f,{out.data(),out.size(),memory});
    for (int corrupt:{13,14}) {
        solid::RepresentativeIndex index(g,2,b); auto ranker=std::make_unique<test_builder::Emulator>(g,2,b,true); ranker->corrupt=corrupt;
        index.set_ranker(std::move(ranker));
        std::vector<float> batch(32,1); std::vector<solid::Half> keys(32,0);
        rejects([&] { index.observe_batch_device(0,0,8,{batch.data(),batch.size(),memory},{keys.data(),keys.size(),memory}); });
    }
    auto store=std::make_unique<ReadFaultStore>(g); auto * inspect=store.get();
    solid::AttentionSession history(std::move(store),2,b,false,16);
    (void) history.process(0,0,8,std::vector<float>(32,1),std::vector<float>(16,0.5f),std::vector<float>(16,0.25f),0.5f);
    int aborts=0; auto backend=std::make_unique<test_tiled::Emulator>(g,2,3,4,true);
    backend->failure=6; backend->abort_count=&aborts; history.set_prefill(std::move(backend)); inspect->fault=true;
    bool original=false;
    try { history.process_prefill_device(0,8,4,{q.data(),q.size(),memory},{kv.data(),kv.size(),memory},kv,0.5f,{out.data(),out.size(),memory}); }
    catch (const std::exception & error) { original=std::string(error.what())=="injected device prefill history error"; }
    require(original && aborts==1,"history failure did not drain readers or preserve its first error");
}
}
int main() {
    try {
        for (bool ties:{false,true}) for (std::uint32_t attach:{0,1,5,10,26}) representative_batches(ties,attach);
        for (bool sparse:{false,true}) for (bool disk:{false,true}) for (auto layout:{solid::KvLayout::separate,solid::KvLayout::interleaved}) session_case(sparse,disk,layout);
        errors(); std::cout<<"device prefill inputs, representative batches, output and handoff contracts passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
