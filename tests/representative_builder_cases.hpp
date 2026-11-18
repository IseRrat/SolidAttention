#pragma once

#include "solid/attention.hpp"
#include "solid/representative_math.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace test_builder {
inline void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F operation) {
    bool failed = false;
    try { operation(); } catch (const std::exception &) { failed = true; }
    require(failed, "invalid representative builder operation accepted");
}

// CPU execution of the CUDA per-item math. The reference in run_case uses
// RepresentativeIndex's existing deque, scalar accumulation and std::sort.
class Emulator final : public solid::BlockRanker {
    struct Layer {
        bool initialized = false, pending = false;
        std::uint64_t length = 0;
        std::vector<solid::Half> ring;
        std::vector<double> scores;
        solid::RepresentativeSeed top;
    };
public:
    Emulator(solid::KvGeometry geometry, std::uint32_t query_heads, solid::SparseBudget budget, bool direct = false) :
        g(geometry), qh(query_heads), b(budget), layers(geometry.layers), prefixes(geometry.layers,0), direct_(direct) {}
    std::optional<solid::DeviceMemory> direct_memory() const override {
        return direct_ ? std::optional<solid::DeviceMemory>{solid::DeviceMemory::host_emulator} : std::optional<solid::DeviceMemory>{};
    }
    std::vector<float> device_query(solid::DeviceQuery query) const {
        require(direct_,"direct emulator is disabled");
        solid::validate_device_query(query,solid::DeviceMemory::host_emulator,qh,g.dimension);
        std::vector<float> values(query.data,query.data+query.elements);
        for (auto q:values) require(std::isfinite(q),"non-finite device query");
        return values;
    }
    std::optional<solid::BuiltRepresentatives> build_query_device(std::uint32_t layer,std::uint64_t p,solid::DeviceQuery query) override {
        auto result=build_query(layer,p,device_query(query)); ++stats_.device_builder_queries; return result;
    }
    void build_append_device(std::uint32_t layer,std::uint64_t p,solid::DeviceKeys keys) override {
        require(direct_,"direct emulator is disabled");
        solid::validate_device_keys(keys,solid::DeviceMemory::host_emulator,g.heads,g.dimension);
        std::vector<solid::Half> values(static_cast<std::size_t>(g.heads)*g.dimension);
        for (std::uint32_t h=0;h<g.heads;++h) for (std::uint32_t d=0;d<g.dimension;++d) {
            const auto k=keys.data[h*keys.head_stride+d];
            require(std::isfinite(solid::from_half(k)),"non-finite device key"); values[h*g.dimension+d]=k;
        }
        build_append(layer,p,values); ++stats_.device_builder_appends;
    }
    std::vector<std::vector<std::uint64_t>> rank_device(std::uint32_t layer,solid::DeviceQuery query,const solid::BlockCandidates & candidates) override {
        auto result=rank(layer,device_query(query),candidates); ++stats_.device_rank_queries; return result;
    }
    void publish(std::uint32_t layer, std::uint64_t first, std::uint64_t count, const std::vector<solid::Half> & keys) override {
        const auto size = static_cast<std::size_t>(g.heads)*b.representatives*g.dimension;
        require(first == prefixes.at(layer) && keys.size() == count*size, "emulator publication prefix");
        for (std::uint64_t i = 0; i < count; ++i) bank[{layer,first+i}] = {keys.begin()+i*size,keys.begin()+(i+1)*size};
        prefixes[layer] += count; stats_.published_blocks += count;
    }
    std::uint32_t builder_window() const override { return b.local; }
    const char * builder_name() const override { return "host_window_emulator"; }
    void seed_builder(std::uint32_t layer, const solid::RepresentativeSeed & seed) override {
        solid::validate_representative_seed(g,b.representatives,b.local,seed);
        if (corrupt == 12 && layer == 1) throw std::runtime_error("injected second-layer seed failure");
        auto & state = layers.at(layer);
        require(!state.initialized, "duplicate emulator seed");
        const auto pending = std::min<std::uint64_t>(seed.length,b.local);
        const auto first = seed.length-pending;
        require(prefixes[layer] == first/g.block_tokens,"seed prefix mismatch");
        const auto width = static_cast<std::size_t>(g.heads)*g.dimension;
        state.ring.assign(b.local*width,0); state.scores.assign(static_cast<std::size_t>(b.local)*g.heads,0);
        for (std::uint64_t i = 0; i < pending; ++i) {
            const auto slot = (first+i)%b.local;
            std::copy_n(seed.pending_keys.begin()+i*width,width,state.ring.begin()+slot*width);
            std::copy_n(seed.pending_scores.begin()+i*g.heads,g.heads,state.scores.begin()+slot*g.heads);
        }
        state.top.partial_keys = seed.partial_keys;
        state.top.partial_scores = seed.partial_scores;
        state.top.partial_positions = seed.partial_positions;
        state.length = seed.length; state.initialized = true;
    }
    std::optional<solid::BuiltRepresentatives> build_query(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) override {
        auto & state = layers.at(layer);
        require(state.initialized && !state.pending && state.length == position,"emulator query order");
        if (corrupt == 10) throw std::runtime_error("injected builder query failure");
        std::vector<double> mean(static_cast<std::size_t>(g.heads)*g.dimension);
        for (std::uint32_t h = 0; h < g.heads; ++h) for (std::uint32_t d = 0; d < g.dimension; ++d) {
            mean[h*g.dimension+d] = solid::representative_mean(query.data(),h,d,g.dimension,qh/g.heads);
        }
        const auto count = std::min<std::uint64_t>(position,b.local);
        for (auto p = position-count; p < position; ++p) for (std::uint32_t h = 0; h < g.heads; ++h) {
            const auto slot = (p%b.local)*g.heads+h;
            state.scores[slot] = solid::update_representative_score(state.scores[slot],mean.data()+h*g.dimension,
                state.ring.data()+slot*g.dimension,g.dimension,[](solid::Half k) { return solid::from_half(k); });
        }
        std::optional<solid::BuiltRepresentatives> result;
        if (position >= b.local) {
            const auto old = position-b.local;
            for (std::uint32_t h = 0; h < g.heads; ++h) {
                const auto slot = (old%b.local)*g.heads+h;
                const auto offset = static_cast<std::size_t>(h)*b.representatives;
                solid::insert_representative(old,state.scores[slot]/b.local,state.ring.data()+slot*g.dimension,
                    b.representatives,g.dimension,state.top.partial_positions.data()+offset,
                    state.top.partial_scores.data()+offset,state.top.partial_keys.data()+offset*g.dimension);
            }
            if ((old+1)%g.block_tokens == 0) {
                result = solid::BuiltRepresentatives{old/g.block_tokens,state.top.partial_keys,state.top.partial_positions,state.top.partial_scores};
                publish(layer,result->block,1,result->keys);
                std::fill(state.top.partial_positions.begin(),state.top.partial_positions.end(),UINT64_MAX);
                std::fill(state.top.partial_scores.begin(),state.top.partial_scores.end(),0);
                ++stats_.built_blocks;
            }
        }
        state.pending = true; ++stats_.builder_queries;
        if (corrupt == 1) { if (result) result.reset(); else result.emplace(); }
        if (result) {
            if (corrupt == 2) ++result->block;
            if (corrupt == 3) result->keys.pop_back();
            if (corrupt == 4) result->positions[0] = UINT64_MAX;
            if (corrupt == 5) result->positions[1] = result->positions[0];
            if (corrupt == 6) result->scores[0] = NAN;
            if (corrupt == 7) result->keys[0] = 0x7c00;
            if (corrupt == 8) result->scores[1] = result->scores[0]+100;
        }
        return result;
    }
    void build_append(std::uint32_t layer, std::uint64_t position, const std::vector<solid::Half> & keys) override {
        auto & state = layers.at(layer);
        require(state.initialized && state.pending && state.length == position,"emulator append order");
        if (corrupt == 11) throw std::runtime_error("injected builder append failure");
        const auto slot = position%b.local;
        std::copy(keys.begin(),keys.end(),state.ring.begin()+slot*keys.size());
        std::fill_n(state.scores.begin()+slot*g.heads,g.heads,0);
        state.pending = false; ++state.length; ++stats_.builder_appends;
    }
    std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer,const std::vector<float> & query,const solid::BlockCandidates & candidates) override {
        std::vector<std::vector<std::uint64_t>> result(g.heads);
        for (std::uint32_t h = 0; h < g.heads; ++h) {
            std::vector<std::pair<double,std::size_t>> scores;
            for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
                const auto & keys = bank.at({layer,candidates.blocks[i]});
                double value = 0;
                for (std::uint32_t r = 0; r < b.representatives; ++r) for (auto q = h*(qh/g.heads); q < (h+1)*(qh/g.heads); ++q) {
                    for (std::uint32_t d = 0; d < g.dimension; ++d) value += static_cast<double>(query[q*g.dimension+d])*
                        solid::from_half(keys[(h*b.representatives+r)*g.dimension+d])/(qh/g.heads);
                }
                scores.emplace_back(value,i);
            }
            std::sort(scores.begin(),scores.end(),[&](auto a,auto c) {
                return a.first == c.first ? candidates.blocks[a.second] < candidates.blocks[c.second] : a.first > c.first;
            });
            std::uint64_t covered = 0;
            for (const auto & score : scores) {
                result[h].push_back(candidates.blocks[score.second]); covered += candidates.tokens[score.second];
                if (covered >= candidates.remaining) break;
            }
        }
        ++stats_.calls; return result;
    }
    solid::RankerStats stats() const override { return stats_; }
    const char * name() const override { return "host_builder_and_ranker"; }
    solid::KvGeometry g;
    std::uint32_t qh;
    solid::SparseBudget b;
    std::vector<Layer> layers;
    std::vector<std::uint64_t> prefixes;
    std::map<std::pair<std::uint32_t,std::uint64_t>,std::vector<solid::Half>> bank;
    solid::RankerStats stats_;
    int corrupt = 0;
    bool direct_ = false;
};

template<class Factory> void run_case(Factory factory, solid::KvGeometry g, std::uint32_t qh, solid::SparseBudget budget,
                                       std::uint32_t attach, bool ties) {
    solid::RepresentativeIndex reference(g,qh,budget), actual(g,qh,budget);
    std::vector<float> query(static_cast<std::size_t>(qh)*g.dimension);
    std::vector<solid::Half> keys(static_cast<std::size_t>(g.heads)*g.dimension);
    std::mt19937 random(1871);
    std::uint64_t device_bytes = 0;
    const auto configure = [&] {
        actual.set_ranker(factory(g,qh,budget)); device_bytes = actual.ranker_stats().device_bytes;
    };
    if (!attach) configure();
    for (std::uint32_t p = 0; p < g.capacity_tokens; ++p) {
        if (attach && p == attach) configure();
        for (std::uint32_t layer = 0; layer < g.layers; ++layer) {
            for (auto & q : query) q = ties ? 0 : (static_cast<int>(random()%43)-21)/16.0f;
            for (auto & k : keys) k = solid::to_half((static_cast<int>(random()%31)-15)/8.0f);
            reference.observe_query(layer,p,query); actual.observe_query(layer,p,query);
            for (std::uint64_t block = 0; block < g.blocks_per_head(); ++block) {
                require(actual.ready(layer,block) == reference.ready(layer,block),"representative maturity differs");
                if (actual.ready(layer,block)) for (std::uint32_t head = 0; head < g.heads; ++head) {
                    require(actual.representative_positions(layer,head,block) == reference.representative_positions(layer,head,block),"representative token selection differs");
                    require(actual.score(layer,head,block,query) == reference.score(layer,head,block,query),"built K payload differs");
                }
            }
            const auto selected = solid::select_blocks(g,budget,actual,layer,p,query);
            const auto expected = solid::select_blocks(g,budget,reference,layer,p,query);
            for (std::uint32_t head = 0; head < g.heads; ++head) require(selected[head].positions == expected[head].positions,"builder/ranker selection differs");
            reference.append_keys(layer,p,keys); actual.append_keys(layer,p,keys);
            if (p >= attach) require(actual.ranker_stats().device_bytes == device_bytes,"builder device allocation grew after initialization");
        }
    }
    const auto stats = actual.ranker_stats();
    require(stats.builder_queries == static_cast<std::uint64_t>(g.capacity_tokens-attach)*g.layers &&
            stats.builder_appends == stats.builder_queries,"builder query/append counters");
    require(stats.published_blocks == static_cast<std::uint64_t>(g.capacity_tokens-budget.local)/g.block_tokens*g.layers,"builder published prefix");
}

template<class Factory> void run_cases(Factory factory) {
    const solid::KvGeometry g{2,2,7,4,77,64};
    for (bool ties : {false,true}) for (std::uint32_t attach : {0,1,4,5,6,8,9,10,26,75}) {
        run_case(factory,g,6,{19,2,5,2},attach,ties);
    }
    run_case(factory,{1,1,1,1,31,64},2,{5,1,1,1},0,false);
    run_case(factory,{1,2,128,4,43,64},4,{16,2,1,4},10,false);
    run_case(factory,{2,1,9,4,13,64},3,{13,0,13,1},0,false);
    run_case(factory,{1,1,7,32,577,64},3,{512,16,480,4},501,false);
}
} // namespace test_builder
