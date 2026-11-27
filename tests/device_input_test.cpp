// 借用设备指针的输入形状、内存归属和生命周期检查。
#include "representative_builder_cases.hpp"
#include <iostream>
#include <limits>

using namespace test_builder;
namespace {
void views() {
    float q[6]{};
    solid::Half k[12]{};
    const auto memory=solid::DeviceMemory::host_emulator;
    solid::validate_device_query({q,6,memory},memory,2,3);
    solid::validate_device_keys({k,9,6,memory},memory,2,3);
    rejects([&] { solid::validate_device_query({q,6,solid::DeviceMemory::cuda},memory,2,3); });
    rejects([&] { solid::validate_device_query({nullptr,6,memory},memory,2,3); });
    rejects([&] { solid::validate_device_query({q,5,memory},memory,2,3); });
    rejects([&] { solid::validate_device_keys({k,8,6,memory},memory,2,3); });
    rejects([&] { solid::validate_device_keys({k,12,2,memory},memory,2,3); });
    rejects([&] { solid::validate_device_keys({k,SIZE_MAX,SIZE_MAX,memory},memory,2,3); });
    rejects([&] { solid::validate_device_keys({k,12,6,memory},memory,0,3); });
    rejects([&] { solid::validate_device_query({reinterpret_cast<const float *>(reinterpret_cast<const char *>(q)+1),6,memory},memory,2,3); });
}
void parity(bool ties, std::uint32_t attach) {
    const solid::KvGeometry g{2,2,7,4,77,64};
    const solid::SparseBudget b{19,2,5,2};
    solid::RepresentativeIndex reference(g,6,b),actual(g,6,b);
    std::mt19937 random(32341);
    std::vector<float> q(6*g.dimension);
    std::vector<solid::Half> keys(g.heads*g.dimension),kv(keys.size()*2,0x7c00);
    for (std::uint32_t p=0;p<g.capacity_tokens;++p) {
        if (p==attach) actual.set_ranker(std::make_unique<Emulator>(g,6,b,true));
        for (std::uint32_t layer=0;layer<g.layers;++layer) {
            for (auto & value:q) value=ties?0:(static_cast<int>(random()%37)-18)/8.0f;
            for (std::uint32_t h=0;h<g.heads;++h) for (std::uint32_t d=0;d<g.dimension;++d) {
                const auto value=solid::to_half((static_cast<int>(random()%33)-16)/8.0f);
                keys[h*g.dimension+d]=kv[h*2*g.dimension+d]=value;
            }
            const solid::DeviceQuery view{q.data(),q.size(),solid::DeviceMemory::host_emulator};
            reference.observe_query(layer,p,q);
            if (p>=attach) actual.observe_query_device(layer,p,view); else actual.observe_query(layer,p,q);
            const auto expected=solid::select_blocks(g,b,reference,layer,p,q);
            const auto selected=p>=attach?solid::select_blocks(g,b,actual,layer,p,view):solid::select_blocks(g,b,actual,layer,p,q);
            for (std::uint32_t h=0;h<g.heads;++h) {
                require(selected[h].positions==expected[h].positions,"borrowed Q changed selected tokens");
                for (std::uint64_t block=0;block<g.blocks_per_head();++block) {
                    require(actual.ready(layer,block)==reference.ready(layer,block),"direct builder maturity differs");
                    if (actual.ready(layer,block)) require(actual.representative_positions(layer,h,block)==reference.representative_positions(layer,h,block) &&
                        actual.score(layer,h,block,q)==reference.score(layer,h,block,q),"strided device K changed representative bits or ordering");
                }
            }
            reference.append_keys(layer,p,keys);
            if (p>=attach) actual.append_keys_device(layer,p,{kv.data(),kv.size(),2*g.dimension,solid::DeviceMemory::host_emulator});
            else actual.append_keys(layer,p,keys);
        }
    }
    const auto stats=actual.ranker_stats();
    require(stats.device_builder_queries==2ull*(g.capacity_tokens-attach) && stats.device_builder_appends==stats.device_builder_queries,
        "borrowed query and append counts");
}
void errors() {
    const solid::KvGeometry g{1,2,3,2,16,64};
    const solid::SparseBudget b{6,1,1,1};
    solid::RepresentativeIndex index(g,4,b);
    std::vector<float> q(12,1);
    std::vector<solid::Half> kv(12,0);
    solid::DeviceQuery view{q.data(),q.size(),solid::DeviceMemory::host_emulator};
    rejects([&] { index.observe_query_device(0,0,view); });
    index.set_ranker(std::make_unique<Emulator>(g,4,b,true));
    rejects([&] { index.append_keys_device(0,0,{kv.data(),kv.size(),6,solid::DeviceMemory::host_emulator}); });
    auto bad=view; bad.elements=1; rejects([&] { index.observe_query_device(0,0,bad); });
    q[3]=NAN; rejects([&] { index.observe_query_device(0,0,view); }); q[3]=1;
    index.observe_query_device(0,0,view);
    rejects([&] { index.observe_query_device(0,0,view); });
    kv[6]=0x7c00; rejects([&] { index.append_keys_device(0,0,{kv.data(),kv.size(),6,solid::DeviceMemory::host_emulator}); }); kv[6]=0;
    index.append_keys_device(0,0,{kv.data(),kv.size(),6,solid::DeviceMemory::host_emulator});
    rejects([&] { index.append_keys_device(0,0,{kv.data(),kv.size(),6,solid::DeviceMemory::host_emulator}); });
}
}
int main() {
    try {
        views(); errors();
        for (bool ties:{false,true}) for (std::uint32_t attach:{0,1,5,8,10,26,75}) parity(ties,attach);
        std::cout<<"borrowed device inputs, strided K and CPU representative parity passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
