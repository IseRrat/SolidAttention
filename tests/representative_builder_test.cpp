// 运行 CPU 代表构造器的通用数值与状态检查。
#include "representative_builder_cases.hpp"
#include "solid/session.hpp"
#include <iostream>

using namespace test_builder;
namespace {
void invalid_seeds() {
    const solid::KvGeometry g{1,1,1,4,32,64};
    solid::RepresentativeSeed seed;
    seed.length = 5; seed.pending_keys = {0,0}; seed.pending_scores = {0,0};
    seed.partial_keys = {0,0}; seed.partial_positions = {0,1}; seed.partial_scores = {2,1};
    solid::validate_representative_seed(g,2,2,seed);
    auto bad = seed; bad.pending_keys.pop_back(); rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.partial_positions[1]=0; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.partial_positions[1]=3; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.partial_scores[1]=3; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.pending_scores[0]=NAN; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.pending_keys[0]=0x7c00; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    bad=seed; bad.length=33; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
    rejects([&] { solid::validate_representative_seed(g,2,0,seed); });
    rejects([&] { solid::validate_representative_seed(g,5,2,seed); });
    // One matured token, followed by explicit unused padding.
    seed.length=3; seed.partial_positions={0,UINT64_MAX};
    solid::validate_representative_seed(g,2,2,seed);
    bad=seed; bad.partial_positions[1]=1; rejects([&] { solid::validate_representative_seed(g,2,2,bad); });
}
void bad_builder_outputs() {
    const solid::KvGeometry g{1,1,2,4,32,64};
    const solid::SparseBudget budget{10,1,3,2};
    for (int corruption = 1; corruption <= 8; ++corruption) {
        solid::RepresentativeIndex index(g,2,budget);
        auto backend = std::make_unique<Emulator>(g,2,budget); auto * inspect = backend.get();
        index.set_ranker(std::move(backend));
        for (std::uint32_t p = 0; p < 6; ++p) index.observe(0,p,{1,2,3,4},{solid::to_half(p),0});
        inspect->corrupt=corruption;
        rejects([&] { index.observe_query(0,6,{1,2,3,4}); });
        require(!index.ready(0,0),"malformed block became visible");
    }
    solid::RepresentativeIndex index(g,2,budget);
    index.observe_query(0,0,{1,2,3,4});
    rejects([&] { index.set_ranker(std::make_unique<Emulator>(g,2,budget)); });
    index.append_keys(0,0,{0,0});
    auto other = budget; ++other.local;
    rejects([&] { index.set_ranker(std::make_unique<Emulator>(g,2,other)); });
}
void query_and_append_failures() {
    const solid::KvGeometry g{1,1,2,4,32,64};
    const solid::SparseBudget budget{10,1,3,2};
    for (int failure : {10,11}) {
        solid::AttentionSession session(std::make_unique<solid::MemoryKvStore>(g),2,budget,true,12);
        for (int p = 0; p < 5; ++p) (void) session.process(0,p,1,{1,2,3,4},{1,2},{2,3},0.5f);
        auto backend=std::make_unique<Emulator>(g,2,budget); auto * inspect=backend.get();
        session.set_ranker(std::move(backend)); inspect->corrupt=failure;
        require(session.stats().representative_backend == "host_window_emulator","builder identity");
        rejects([&] { (void) session.process(0,5,1,{1,2,3,4},{1,2},{2,3},0.5f); });
        inspect->corrupt=0;
        rejects([&] { (void) session.process(0,6,1,{1,2,3,4},{1,2},{2,3},0.5f); });
    }
}
void uneven_layers() {
    const solid::KvGeometry g{2,1,1,4,32,64};
    const solid::SparseBudget budget{10,1,3,2};
    solid::RepresentativeIndex index(g,2,budget), reference(g,2,budget);
    for (int layer = 0; layer < 2; ++layer) for (int p = 0; p < 5+layer*8; ++p) {
        index.observe(layer,p,{1,2},{solid::to_half(p)}); reference.observe(layer,p,{1,2},{solid::to_half(p)});
    }
    auto failed = std::make_unique<Emulator>(g,2,budget); failed->corrupt=12;
    rejects([&] { index.set_ranker(std::move(failed)); });
    require(std::string(index.builder_name()) == "cpu","failed initialization changed CPU ownership");
    index.set_ranker(std::make_unique<Emulator>(g,2,budget));
    for (int layer = 0; layer < 2; ++layer) for (int p = 5+layer*8; p < 31; ++p) {
        index.observe(layer,p,{1,2},{solid::to_half(p)}); reference.observe(layer,p,{1,2},{solid::to_half(p)});
        for (std::uint64_t block = 0; block < g.blocks_per_head(); ++block) if (reference.ready(layer,block)) {
            require(index.representative_positions(layer,0,block) == reference.representative_positions(layer,0,block),"per-layer seed state leaked");
        }
    }
}
} // namespace
int main() {
    try {
        run_cases([](auto g,auto qh,auto b) { return std::make_unique<Emulator>(g,qh,b); });
        invalid_seeds(); bad_builder_outputs(); query_and_append_failures(); uneven_layers();
        std::cout << "representative builder contracts passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
