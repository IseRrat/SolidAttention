#pragma once
#include "solid/session.hpp"
#include "solid/online_attention.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace test_tiled {
inline void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F work) { bool failed=false; try { work(); } catch (const std::exception &) { failed=true; } require(failed,"invalid tiled prefill operation accepted"); }
class Emulator final:public solid::TiledPrefill {
public:
    Emulator(solid::KvGeometry g,std::uint32_t qh,std::uint32_t queries,std::uint32_t tile,bool direct=false):g(g),qh(qh),queries(queries),direct_(direct) {
        capacity=static_cast<std::uint32_t>((std::min<std::uint64_t>(tile,g.capacity_tokens)+g.block_tokens-1)/g.block_tokens*g.block_tokens);
        stats_.kv_device_bytes=static_cast<std::uint64_t>(g.heads)*capacity*2*g.dimension*sizeof(solid::Half);
        stats_.workspace_device_bytes=static_cast<std::uint64_t>(queries)*qh*(2*g.dimension+2)*sizeof(float)+sizeof(int);
        stats_.kv_device_peak_bytes=stats_.kv_device_bytes; stats_.workspace_device_peak_bytes=stats_.workspace_device_bytes;
        stats_.query_capacity=queries; stats_.tile_capacity=capacity;
    }
    std::uint32_t query_capacity() const override { return queries; }
    std::uint32_t tile_capacity() const override { return capacity; }
    const char * name() const override { return "host_tiled_emulator"; }
    solid::PrefillStats stats() const override { return stats_; }
    std::optional<solid::DeviceMemory> direct_memory() const override {
        return direct_?std::optional<solid::DeviceMemory>{solid::DeviceMemory::host_emulator}:std::optional<solid::DeviceMemory>{};
    }
    void begin_device(std::uint64_t first,std::uint32_t count,float scale,solid::DeviceQuery query) override {
        require(direct_,"direct tiled emulator is disabled");
        solid::validate_device_query(query,solid::DeviceMemory::host_emulator,qh,g.dimension,count);
        begin(first,count,scale,{query.data,query.data+query.elements}); stats_.device_query_tokens+=count;
        stats_.query_upload_bytes-=query.elements*sizeof(float);
    }
    void consume_device(std::uint64_t first_key,std::uint32_t valid,solid::DeviceKvBatch kv) override {
        require(direct_,"direct tiled emulator is disabled");
        solid::validate_device_kv(kv,solid::DeviceMemory::host_emulator,g.heads,g.dimension,valid);
        std::vector<solid::Half> tile(static_cast<std::size_t>(g.heads)*capacity*2*g.dimension,0x7e00);
        for (std::uint32_t p=0;p<valid;++p) for (std::uint32_t h=0;h<g.heads;++h) {
            std::copy_n(kv.data+(static_cast<std::size_t>(p)*g.heads+h)*2*g.dimension,2*g.dimension,
                tile.data()+(static_cast<std::size_t>(h)*capacity+p)*2*g.dimension);
        }
        consume(first_key,valid,tile); stats_.upload_bytes-=tile.size()*sizeof(solid::Half); stats_.device_kv_tokens+=valid;
    }
    void finish_to(solid::DeviceOutput output) override {
        require(direct_,"direct tiled emulator is disabled");
        solid::validate_device_output(output,solid::DeviceMemory::host_emulator,qh,g.dimension,count);
        const auto result=finish(); require(result.size()==output.elements,"direct emulator output shape");
        for (auto value:result) require(std::isfinite(value),"non-finite direct emulator output");
        std::copy(result.begin(),result.end(),output.data); stats_.output_download_bytes-=result.size()*sizeof(float);
        stats_.output_device_copy_bytes+=result.size()*sizeof(float);
    }
    void begin(std::uint64_t first,std::uint32_t count,float scale,const std::vector<float> & query) override {
        if (failure==1) throw std::runtime_error("injected begin failure");
        require(!open && count && count<=queries,"emulator frame state");
        require(query.size()==static_cast<std::size_t>(count)*qh*g.dimension,"emulator Q shape");
        for (const auto q:query) require(std::isfinite(q),"emulator query finite");
        require(scale>0 && std::isfinite(scale),"emulator scale");
        this->first=first; this->count=count; this->scale=scale; q=query;
        maximum.assign(static_cast<std::size_t>(count)*qh,-INFINITY); sums.assign(maximum.size(),0);
        output.assign(query.size(),0); next=0; open=true;
        stats_.query_upload_bytes+=query.size()*sizeof(float);
    }
    void consume(std::uint64_t first_key,std::uint32_t valid,const std::vector<solid::Half> & kv) override {
        if (failure==2) throw std::runtime_error("injected consume failure");
        require(open && first_key==next && valid && valid<=std::min<std::uint64_t>(capacity,first+count-next),"emulator tile order");
        require(kv.size()==static_cast<std::size_t>(g.heads)*capacity*2*g.dimension,"emulator KV shape");
        for (std::size_t row=0;row<maximum.size();++row) {
            const auto position=first+row/qh;
            const auto head=(row%qh)/(qh/g.heads);
            for (std::uint32_t t=0;t<valid && first_key+t<=position;++t) {
                const auto base=(head*capacity+t)*2*g.dimension;
                float lanes[32]={};
                for (std::uint32_t lane=0;lane<32;++lane) for (auto d=lane;d<g.dimension;d+=32) {
                    lanes[lane]=std::fma(q[row*g.dimension+d],solid::from_half(kv[base+d]),lanes[lane]);
                }
                for (unsigned delta=16;delta;delta/=2) for (unsigned lane=0;lane<32-delta;++lane) lanes[lane]+=lanes[lane+delta];
                const auto update=solid::online_attention_update(maximum[row],sums[row],lanes[0]*scale);
                maximum[row]=update.maximum; sums[row]=update.sum;
                for (std::uint32_t d=0;d<g.dimension;++d) output[row*g.dimension+d]=output[row*g.dimension+d]*update.previous_scale+
                    solid::from_half(kv[base+g.dimension+d])*update.weight;
            }
        }
        next+=valid; ++stats_.tiles; stats_.kv_tokens+=valid; stats_.upload_bytes+=kv.size()*sizeof(solid::Half);
    }
    std::vector<float> finish() override {
        if (failure==3) throw std::runtime_error("injected finish failure");
        require(open && next==first+count,"emulator complete prefix");
        for (std::size_t i=0;i<output.size();++i) output[i]/=sums[i/g.dimension];
        open=false; ++stats_.batches; stats_.query_tokens+=count;
        auto result=output;
        stats_.output_download_bytes+=result.size()*sizeof(float);
        if (failure==4) result.pop_back();
        if (failure==5) result[0]=NAN;
        return result;
    }
    void abort() override { open=false; if (abort_count) ++*abort_count; if (failure==6) throw std::runtime_error("injected abort failure"); }
    solid::KvGeometry g;
    std::uint32_t qh,queries,capacity,count=0;
    std::uint64_t first=0,next=0;
    float scale=1;
    bool open=false;
    int failure=0;
    int * abort_count=nullptr;
    std::vector<float> q,maximum,sums,output;
    solid::PrefillStats stats_;
    bool direct_=false;
};
inline void close(const std::vector<float> & actual,const std::vector<float> & expected,float tolerance=3e-4f) {
    require(actual.size()==expected.size(),"tiled output shape");
    for (std::size_t i=0;i<actual.size();++i) require(std::isfinite(actual[i]) &&
        std::abs(actual[i]-expected[i])<tolerance*std::max(1.0f,std::abs(expected[i])),"tiled attention differs from FP64 dense reference");
}
template<class Factory> void frames(Factory factory,std::uint32_t width,bool constant=false,std::uint32_t queries=5,
                                     float tolerance=3e-4f) {
    const solid::KvGeometry g{1,2,width,4,49,64,solid::KvLayout::interleaved};
    constexpr std::uint32_t qh=6;
    solid::MemoryKvStore store(g);
    std::mt19937 random(761);
    const auto sample=[&] { return (static_cast<int>(random()%31)-15)/8.0f; };
    std::vector<std::vector<solid::Half>> keys,values;
    for (std::uint32_t p=0;p<g.capacity_tokens;++p) {
        keys.emplace_back(static_cast<std::size_t>(g.heads)*width);
        values.emplace_back(keys.back().size());
        for (auto & k:keys.back()) k=solid::to_half(sample());
        for (auto & v:values.back()) v=solid::to_half(constant?7.25f:sample());
        store.append(0,p,keys.back(),values.back());
    }
    auto backend=factory(g,qh,queries,7);
    const auto bytes=backend->stats().kv_device_bytes+backend->stats().workspace_device_bytes;
    std::uint64_t query_tokens=0;
    for (const auto first: {0u,3u,7u,24u,44u}) {
        const auto count=std::min(queries,g.capacity_tokens-first);
        query_tokens += count;
        const auto scale=1/std::sqrt(static_cast<float>(width));
        std::vector<float> query(static_cast<std::size_t>(count)*qh*width);
        for (auto & q:query) q=sample();
        const auto saved=query;
        backend->begin(first,count,scale,query);
        std::fill(query.begin(),query.end(),NAN); // begin has returned the Q source lease.
        const auto cap=backend->tile_capacity();
        std::vector<solid::Half> tile(static_cast<std::size_t>(g.heads)*cap*2*width,0x7e00);
        for (std::uint64_t start=0;start<first+count;start+=cap) {
            const auto valid=static_cast<std::uint32_t>(std::min<std::uint64_t>(cap,first+count-start));
            for (std::uint32_t h=0;h<g.heads;++h) for (std::uint32_t p=0;p<valid;++p) {
                auto * target=tile.data()+(static_cast<std::size_t>(h)*cap+p)*2*width;
                std::copy_n(keys[start+p].data()+h*width,width,target);
                std::copy_n(values[start+p].data()+h*width,width,target+width);
            }
            backend->consume(start,valid,tile);
            std::fill(tile.begin(),tile.end(),solid::Half{0x7e00}); // H2D has returned the source lease.
        }
        auto actual=backend->finish();
        std::vector<float> expected;
        for (std::uint32_t p=0;p<count;++p) {
            const auto begin=saved.begin()+static_cast<std::size_t>(p)*qh*width;
            auto row=solid::streamed_attention(store,0,first+p,qh,{begin,begin+qh*width},scale);
            expected.insert(expected.end(),row.begin(),row.end());
        }
        close(actual,expected,tolerance);
        require(backend->stats().kv_device_bytes+backend->stats().workspace_device_bytes==bytes,"prefill allocations grew with history length");
    }
    require(backend->stats().batches==5 && backend->stats().query_tokens==query_tokens,"prefill frame counters");
}
template<class Factory> void session_case(Factory factory,solid::KvLayout layout,bool packed) {
    const solid::KvGeometry g{2,2,7,4,37,64,layout};
    const solid::SparseBudget budget{16,2,5,2};
    solid::AttentionSession actual(std::make_unique<solid::MemoryKvStore>(g),6,budget,true,29);
    solid::AttentionSession reference(std::make_unique<solid::MemoryKvStore>(g),6,budget,true,29);
    actual.set_prefill(factory(g,6,5,7));
    std::mt19937 random(181);
    std::uint32_t start=0;
    for (const auto size:{13u,13u,7u}) {
        for (std::uint32_t layer=0;layer<2;++layer) {
            std::vector<float> q(size*42),k(size*14),v(k.size());
            for (auto & x:q) x=(static_cast<int>(random()%19)-9)/8.0f;
            for (auto & x:k) x=(static_cast<int>(random()%17)-8)/8.0f;
            for (auto & x:v) x=(static_cast<int>(random()%29)-14)/8.0f;
            if (packed) {
                std::vector<float> interleaved(k.size()*2);
                for (std::uint32_t p=0;p<size;++p) for (std::uint32_t h=0;h<g.heads;++h) for (std::uint32_t d=0;d<g.dimension;++d) {
                    const auto i=(p*g.heads+h)*g.dimension+d;
                    const auto offset=(p*g.heads+h)*2*g.dimension+d;
                    interleaved[offset]=k[i]; interleaved[offset+g.dimension]=v[i];
                }
                k=std::move(interleaved); v.clear();
            }
            close(actual.process(layer,start,size,q,k,v,0.25f,packed),reference.process(layer,start,size,q,k,v,0.25f,packed));
            actual.release_prefill_if_complete();
            if (start==26 && layer==0) require(actual.stats().prefill_host_bytes>0,"prefill pool released before all layers completed");
        }
        start+=size;
    }
    const auto stats=actual.stats();
    require(stats.prefill.query_tokens==58 && stats.prefill_host_bytes==0 && stats.prefill.kv_device_bytes==0 &&
        stats.prefill.workspace_device_bytes==0 && stats.prefill_host_peak_bytes>0 && stats.prefill.kv_device_peak_bytes>0,"prefill release or retained peak counters");
}
template<class Factory> void run_cases(Factory factory) {
    for (const std::uint32_t width:{1,7,33,64,80,128,256,512,1024}) frames(factory,width);
    frames(factory,7,true);
    for (bool packed:{false,true}) {
        session_case(factory,solid::KvLayout::separate,packed); session_case(factory,solid::KvLayout::interleaved,packed);
    }
}
} // namespace test_tiled
