#include "prefetch_fakes.hpp"
#include <iostream>

using namespace test_prefetch;
namespace {
class CurrentTransport final:public solid::DeviceTransport {
public:
    struct Copy {
        std::uint32_t slot;
        const solid::KvBlock * host;
        const solid::Half * current=nullptr;
        std::uint32_t prefix=0,row=0,head=0;
        bool done=false;
    };
    explicit CurrentTransport(solid::KvGeometry g):g(g) {}
    std::optional<solid::DeviceMemory> current_memory() const override { return solid::DeviceMemory::host_emulator; }
    std::uint64_t upload(std::uint32_t slot,const solid::KvBlock & block) override {
        const auto id=next++; copies.emplace(id,Copy{slot,&block}); return id;
    }
    std::uint64_t upload_current(std::uint32_t slot,const solid::KvBlock & block,const solid::CurrentDeviceKv & source,std::uint32_t prefix) override {
        const auto id=next++;
        copies.emplace(id,Copy{slot,&block,source.kv.data,prefix,static_cast<std::uint32_t>(source.position%g.block_tokens),block_head(slot)});
        if (fail_on && ++current_calls==fail_on) { drain(); copies.erase(id); throw std::runtime_error("injected queued device row copy"); }
        return id;
    }
    void write(Copy & copy) {
        if (copy.done) return;
        const auto width=static_cast<std::size_t>(2)*g.dimension;
        auto & output=memory[copy.slot]; output.resize(g.block_tokens*width,0);
        if (!copy.current) output=copy.host->data;
        else {
            for (auto row=copy.prefix;row<copy.row;++row) std::copy_n(copy.host->data.data()+row*width,width,output.data()+row*width);
            std::copy_n(copy.current+copy.head*width,width,output.data()+copy.row*width);
        }
        copy.done=true;
    }
    bool complete(std::uint64_t ticket,bool wait) override {
        auto it=copies.find(ticket); require(it!=copies.end(),"unknown current copy receipt");
        if (!wait && !it->second.done) return false;
        write(it->second); copies.erase(it); return true;
    }
    void drain() override { for (auto & item:copies) write(item.second); }
    std::uint64_t resident_bytes() const override { return 4096; }
    std::uint32_t block_head(std::uint32_t slot) const { return slot/slots; }
    solid::KvGeometry g;
    std::uint32_t slots=4,fail_on=0,current_calls=0;
    std::uint64_t next=1;
    std::map<std::uint64_t,Copy> copies;
    std::map<std::uint32_t,std::vector<solid::Half>> memory;
};
std::vector<solid::HeadSelection> all(const solid::KvGeometry & g,std::uint64_t p) {
    std::vector<solid::HeadSelection> result(g.heads);
    for (auto & head:result) for (std::uint64_t i=0;i<=p;++i) { head.positions.push_back(i); if (i%g.block_tokens==0) head.blocks.push_back(i/g.block_tokens); }
    return result;
}
template<class F> void rejects(F work) { bool failed=false; try { work(); } catch (const std::exception &) { failed=true; } require(failed,"invalid current device operation accepted"); }
void verify_current(solid::KvStore & store,solid::DeviceCache & cache,CurrentTransport & transport,std::uint64_t p,std::uint32_t layer=0) {
    for (std::uint32_t h=0;h<store.geometry().heads;++h) for (std::uint64_t b=0;b<=p/store.geometry().block_tokens;++b) {
        solid::KvBlock expected; store.read(layer,h,b,expected);
        const auto lease=cache.find({layer,h,b,expected.version}); require(lease.has_value(),"formal bank block missing");
        const auto & actual=transport.memory.at(lease->slot);
        require(std::equal(expected.data.begin(),expected.data.begin()+expected.valid_tokens*2*expected.dimension,actual.begin()),"device row/prefix payload differs from store");
    }
}
void prefix_and_gap() {
    const solid::KvGeometry g{1,2,3,4,24,64,solid::KvLayout::interleaved};
    const solid::SparseBudget budget{12,1,3,1};
    solid::MemoryKvStore store(g); fill(store,3);
    solid::Prefetcher host(store,budget,4,std::make_unique<Reader>());
    CurrentTransport transport(g); solid::DeviceCache cache(host,g,6,transport);
    host.begin(0,3); cache.prefetch_ready(); cache.wait_transfers();
    const auto old=cache.find({0,0,0,3}); require(old.has_value(),"predicted tail prefix missing");
    cache.begin_stage({},solid::CurrentBlock{0,3}); require(cache.find({0,0,0,3}).has_value(),"early history stage retired the current prefix");
    auto kv=token(g,0,3); store.append_interleaved(0,3,kv);
    auto selection=all(g,3); auto leases=host.resolve(0,3,selection);
    const solid::CurrentDeviceKv source{0,3,{kv.data(),kv.size(),solid::DeviceMemory::host_emulator}};
    auto bad=source; bad.kv.memory=solid::DeviceMemory::cuda;
    rejects([&] { (void) cache.prepare(leases,selection,true,&bad); });
    const auto mask=cache.prepare(leases,selection,true,&source);
    const auto latest=cache.find({0,0,0,4});
    require(latest && latest->slot==old->slot && latest->generation==old->generation+1 && !cache.find(old->identity),"tail refresh did not advance its existing slot generation");
    require(!transport.copies.empty(),"device source lease was completed too early");
    for (const auto & lease:leases) host.unpin(lease);
    rejects([&] { cache.attention_complete(cache.attention_epoch()); });
    rejects([&] { cache.begin_stage({}); });
    cache.wait_transfers(); verify_current(store,cache,transport,3);
    std::fill(kv.begin(),kv.end(),solid::Half{0x7e00});
    require(mask.size()==cache.tokens_per_head()*6ull,"current device mask shape");
    require(cache.stats().refreshed_tails==2 && cache.stats().current_host_bytes==0 && cache.stats().retained_prefix_bytes==2*3*12,
        "retained tail prefix transferred historical bytes again");
    cache.attention_complete(cache.attention_epoch());
    kv=token(g,0,4); host.begin(0,4); store.append_interleaved(0,4,kv);
    selection=all(g,4); leases=host.resolve(0,4,selection);
    const solid::CurrentDeviceKv fresh{0,4,{kv.data(),kv.size(),solid::DeviceMemory::host_emulator}};
    (void) cache.prepare(leases,selection,false,&fresh); for (const auto & lease:leases) host.unpin(lease);
    verify_current(store,cache,transport,4); cache.attention_complete(cache.attention_epoch());
    // Leave one older device row, then skip an update. Only the gap is H2D.
    store.append_interleaved(0,5,token(g,0,5)); host.begin(0,6);
    cache.begin_stage({{0,0,0,4},{0,1,0,4}},solid::CurrentBlock{0,6});
    kv=token(g,0,6); store.append_interleaved(0,6,kv); selection=all(g,6); leases=host.resolve(0,6,selection);
    const solid::CurrentDeviceKv gap{0,6,{kv.data(),kv.size(),solid::DeviceMemory::host_emulator}};
    (void) cache.prepare(leases,selection,true,&gap); for (const auto & lease:leases) host.unpin(lease);
    cache.wait_transfers(); verify_current(store,cache,transport,6); cache.attention_complete(cache.attention_epoch());
    const auto stats=cache.stats();
    require(stats.current_device_rows==6 && stats.current_device_bytes==6*12 && stats.current_host_bytes==2*12 &&
        stats.refreshed_tails==4 && stats.retained_prefix_bytes==2*4*12,"current/gap transfer counters");
    cache.finish(); host.finish(); require(transport.copies.empty(),"current row source remained live");
}
void cold_and_failure(bool fail) {
    const solid::KvGeometry g{1,2,3,4,24,64,solid::KvLayout::interleaved}; const solid::SparseBudget budget{12,1,3,1};
    solid::MemoryKvStore store(g); fill(store,7); solid::Prefetcher host(store,budget,4,std::make_unique<Reader>());
    CurrentTransport transport(g); transport.fail_on=fail?2:0; solid::DeviceCache cache(host,g,6,transport);
    host.begin(0,7); auto kv=token(g,0,7); store.append_interleaved(0,7,kv);
    auto selection=all(g,7); auto leases=host.resolve(0,7,selection);
    const solid::CurrentDeviceKv current{0,7,{kv.data(),kv.size(),solid::DeviceMemory::host_emulator}};
    if (fail) {
        rejects([&] { (void) cache.prepare(leases,selection,true,&current); });
        for (const auto & lease:leases) host.unpin(lease); cache.wait_transfers();
        require(!cache.attention_pending(),"failed row transfer armed attention");
    } else {
        (void) cache.prepare(leases,selection,true,&current); for (const auto & lease:leases) host.unpin(lease);
        cache.wait_transfers(); verify_current(store,cache,transport,7); cache.attention_complete(cache.attention_epoch());
        require(cache.stats().current_host_bytes==2*3*12 && cache.stats().retained_prefix_bytes==0,"cold tail copied the current row from host");
    }
    cache.finish(); host.finish(); require(transport.copies.empty(),"copy failure leaked source leases");
}
void cross_layer() {
    const solid::KvGeometry g{2,2,3,4,24,64,solid::KvLayout::interleaved}; const solid::SparseBudget budget{12,1,3,1};
    solid::MemoryKvStore store(g); fill(store,3); solid::Prefetcher host(store,budget,4,std::make_unique<Reader>());
    CurrentTransport transport(g); solid::DeviceCache cache(host,g,6,transport);
    host.begin(0,3); cache.prefetch_ready(); cache.wait_transfers();
    require(cache.find({0,0,0,3}).has_value(),"cross-layer prefix was not seeded");
    host.begin(1,3); cache.begin_stage({},solid::CurrentBlock{1,3});
    auto kv=token(g,1,3); store.append_interleaved(1,3,kv); const auto selected=all(g,3); auto leases=host.resolve(1,3,selected);
    const solid::CurrentDeviceKv current{1,3,{kv.data(),kv.size(),solid::DeviceMemory::host_emulator}};
    auto wrong=current; wrong.layer=0;
    rejects([&] { (void) cache.prepare(leases,selected,true,&wrong); });
    (void) cache.prepare(leases,selected,true,&current); for (const auto & lease:leases) host.unpin(lease);
    cache.wait_transfers(); verify_current(store,cache,transport,3,1); cache.attention_complete(cache.attention_epoch());
    require(cache.stats().refreshed_tails==0 && cache.stats().retained_prefix_bytes==0 && cache.stats().current_host_bytes==2*3*12,
        "a different layer supplied the current block prefix");
    cache.finish(); host.finish();
}
}
int main() {
    try { prefix_and_gap(); cold_and_failure(false); cold_and_failure(true); cross_layer();
        std::cout<<"current device rows, prefix generations, partial gaps and delayed copy ownership passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
