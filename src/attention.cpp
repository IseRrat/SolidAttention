#include "solid/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace solid {
namespace {
void finite_query(const std::vector<float> & query) {
    for (float value : query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite query");
}
}
void SparseBudget::validate(const KvGeometry & geometry) const {
    if (total == 0 || total > geometry.capacity_tokens || local == 0 || static_cast<std::uint64_t>(initial) + local > total ||
        representatives == 0 || representatives > geometry.block_tokens) {
        throw std::invalid_argument("invalid sparse token budget or representative count");
    }
}
RepresentativeIndex::RepresentativeIndex(KvGeometry geometry, std::uint32_t query_heads, SparseBudget budget) :
    geometry_(geometry), query_heads_(query_heads), budget_(budget), pending_(geometry.layers),
    blocks_(static_cast<std::size_t>(geometry.layers) * geometry.heads), lengths_(geometry.layers, 0), query_pending_(geometry.layers, false) {
    geometry_.validate();
    budget_.validate(geometry_);
    if (query_heads == 0 || query_heads % geometry.heads != 0) throw std::invalid_argument("invalid GQA head mapping");
    for (auto & blocks : blocks_) blocks.resize(geometry.blocks_per_head());
}
std::size_t RepresentativeIndex::index(std::uint32_t layer, std::uint32_t head) const {
    if (layer >= geometry_.layers || head >= geometry_.heads) throw std::out_of_range("representative identity");
    return static_cast<std::size_t>(layer) * geometry_.heads + head;
}
void RepresentativeIndex::observe(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                  const std::vector<Half> & keys) {
    if (keys.size() != static_cast<std::size_t>(geometry_.heads) * geometry_.dimension) throw std::invalid_argument("representative key shape");
    for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite key");
    observe_query(layer, position, query);
    append_keys(layer, position, keys);
}
void RepresentativeIndex::observe_query(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) {
    if (position != lengths_.at(layer) || position >= geometry_.capacity_tokens ||
        query_pending_.at(layer) || query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension) {
        throw std::invalid_argument("representative observation shape or position mismatch");
    }
    finite_query(query);
    if (ranker_ && ranker_->builder_window()) {
        accept_built(layer, position, ranker_->build_query(layer, position, query));
        query_pending_[layer] = true;
        return;
    }
    const auto group = query_heads_ / geometry_.heads;
    std::vector<double> mean_query(static_cast<std::size_t>(geometry_.heads) * geometry_.dimension, 0);
    for (std::uint32_t head = 0; head < query_heads_; ++head) {
        for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
            mean_query[(head / group) * geometry_.dimension + d] += static_cast<double>(query[head * geometry_.dimension + d]) / group;
        }
    }
    auto & pending = pending_[layer];
    for (auto & token : pending) {
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
                const auto offset = head * geometry_.dimension + d;
                token.scores[head] += mean_query[offset] * from_half(token.keys[offset]);
            }
        }
    }
    while (!pending.empty() && position - pending.front().position >= budget_.local) {
        auto & token = pending.front();
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            auto & block = blocks_[index(layer, head)].at(token.position / geometry_.block_tokens);
            Representative representative{token.position, token.scores[head] / budget_.local, {}};
            const auto first = token.keys.begin() + head * geometry_.dimension;
            representative.key.assign(first, first + geometry_.dimension);
            block.tokens.push_back(std::move(representative));
            std::sort(block.tokens.begin(), block.tokens.end(), [](const auto & a, const auto & b) {
                return a.score == b.score ? a.position < b.position : a.score > b.score;
            });
            if (block.tokens.size() > budget_.representatives) block.tokens.resize(budget_.representatives);
            ++block.finalized;
        }
        const auto block = token.position / geometry_.block_tokens;
        if (ranker_ && ready(layer, block)) {
            publish(*ranker_, layer, block, 1);
            release_published_keys(layer, block);
        }
        pending.pop_front();
    }
    query_pending_[layer] = true;
}
void RepresentativeIndex::append_keys(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & keys) {
    if (position != lengths_.at(layer) || !query_pending_.at(layer) || keys.size() != static_cast<std::size_t>(geometry_.heads) * geometry_.dimension) {
        throw std::invalid_argument("representative append requires its query first");
    }
    for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite key");
    if (ranker_ && ranker_->builder_window()) ranker_->build_append(layer, position, keys);
    else pending_[layer].push_back({position, keys, std::vector<double>(geometry_.heads, 0)});
    query_pending_[layer] = false;
    ++lengths_[layer];
}
void RepresentativeIndex::observe_query_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query) {
    const auto memory = direct_memory();
    if (!memory || position != lengths_.at(layer) || position >= geometry_.capacity_tokens || query_pending_.at(layer)) {
        throw std::invalid_argument("device representative query state");
    }
    validate_device_query(query, *memory, query_heads_, geometry_.dimension);
    accept_built(layer, position, ranker_->build_query_device(layer, position, query));
    query_pending_[layer] = true;
}
void RepresentativeIndex::append_keys_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys) {
    const auto memory = direct_memory();
    if (!memory || position != lengths_.at(layer) || !query_pending_.at(layer)) throw std::invalid_argument("device representative append state");
    validate_device_keys(keys, *memory, geometry_.heads, geometry_.dimension);
    ranker_->build_append_device(layer, position, keys);
    query_pending_[layer] = false;
    ++lengths_[layer];
}
void RepresentativeIndex::observe_batch_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,DeviceQuery query,DeviceKvBatch kv) {
    const auto memory=direct_memory();
    if (!memory || !tokens || tokens>geometry_.capacity_tokens || first>geometry_.capacity_tokens-tokens ||
        lengths_.at(layer)!=first || query_pending_.at(layer)) throw std::invalid_argument("device representative batch state");
    validate_device_query(query,*memory,query_heads_,geometry_.dimension,tokens);
    validate_device_kv(kv,*memory,geometry_.heads,geometry_.dimension,tokens);
    auto built=ranker_->build_batch_device(layer,first,tokens,query,kv);
    std::size_t index=0;
    for (std::uint32_t i=0;i<tokens;++i) {
        const auto p=first+i;
        const bool complete=p>=budget_.local && (p-budget_.local+1)%geometry_.block_tokens==0;
        if (complete) {
            if (index==built.size()) throw std::runtime_error("device representative batch omitted a complete block");
            accept_built(layer,p,std::optional<BuiltRepresentatives>{std::move(built[index++])});
        }
    }
    if (index!=built.size()) throw std::runtime_error("device representative batch returned excess blocks");
    lengths_[layer]+=tokens;
}
bool RepresentativeIndex::ready(std::uint32_t layer, std::uint64_t block) const {
    return blocks_.at(index(layer, 0)).at(block).finalized == geometry_.block_tokens;
}
double RepresentativeIndex::score(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                                  const std::vector<float> & query) const {
    if (query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension || !ready(layer, block)) {
        throw std::invalid_argument("block representative is not ready or query has wrong shape");
    }
    const auto & tokens = blocks_.at(index(layer, head)).at(block).tokens;
    std::vector<Half> device_keys;
    if (std::any_of(tokens.begin(), tokens.end(), [](const auto & token) { return token.key.empty(); })) {
        if (!ranker_ || !ranker_->owns_representative_keys()) throw std::logic_error("representative key owner is missing");
        device_keys = ranker_->read_representatives(layer, block);
        const auto elements = static_cast<std::size_t>(geometry_.heads) * budget_.representatives * geometry_.dimension;
        if (device_keys.size() != elements) throw std::runtime_error("representative readback shape mismatch");
        for (const auto key : device_keys) if (!std::isfinite(from_half(key))) throw std::runtime_error("non-finite representative readback");
    }
    const auto group = query_heads_ / geometry_.heads;
    double value = 0;
    for (std::size_t r = 0; r < tokens.size(); ++r) {
        const auto * key = device_keys.empty() ? tokens[r].key.data() :
            device_keys.data() + (static_cast<std::size_t>(head) * budget_.representatives + r) * geometry_.dimension;
        for (std::uint32_t q = head * group; q < (head + 1) * group; ++q) {
            for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
                value += static_cast<double>(query[q * geometry_.dimension + d]) * from_half(key[d]) / group;
            }
        }
    }
    if (!std::isfinite(value)) throw std::runtime_error("non-finite block relevance");
    return value;
}
void RepresentativeIndex::publish(BlockRanker & ranker, std::uint32_t layer, std::uint64_t first, std::uint64_t count) const {
    std::vector<Half> keys;
    const auto stride = static_cast<std::uint64_t>(geometry_.heads) * budget_.representatives * geometry_.dimension;
    if (count && stride > SIZE_MAX / count) throw std::overflow_error("representative publication size");
    keys.reserve(static_cast<std::size_t>(count * stride));
    for (auto block = first; block < first + count; ++block) {
        if (!ready(layer, block)) throw std::logic_error("publishing an unfinished representative block");
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            const auto & tokens = blocks_[index(layer, head)].at(block).tokens;
            if (tokens.size() != budget_.representatives) throw std::logic_error("finalized representative count");
            for (const auto & token : tokens) keys.insert(keys.end(), token.key.begin(), token.key.end());
        }
    }
    if (count) ranker.publish(layer, first, count, keys);
}
void RepresentativeIndex::set_ranker(std::unique_ptr<BlockRanker> ranker) {
    if (!ranker || ranker_) throw std::logic_error("representative ranker already configured or empty");
    if (std::any_of(query_pending_.begin(), query_pending_.end(), [](bool value) { return value; })) {
        throw std::logic_error("representative ranker requires a query boundary");
    }
    if (ranker->builder_window() && ranker->builder_window() != budget_.local) throw std::invalid_argument("representative builder window differs");
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        std::uint64_t count = 0;
        while (count < geometry_.blocks_per_head() && ready(layer, count)) ++count;
        publish(*ranker, layer, 0, count);
        if (ranker->builder_window()) ranker->seed_builder(layer, builder_seed(layer));
    }
    ranker_ = std::move(ranker);
    if (ranker_->builder_window()) for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        pending_[layer].clear();
        const auto matured = lengths_[layer] - std::min<std::uint64_t>(lengths_[layer], budget_.local);
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            auto & partial = blocks_[index(layer, head)].at(matured/geometry_.block_tokens);
            partial.tokens.clear(); partial.finalized = 0;
        }
    }
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        for (std::uint64_t block = 0; block < geometry_.blocks_per_head() && ready(layer, block); ++block) {
            release_published_keys(layer, block);
        }
    }
}
void RepresentativeIndex::release_published_keys(std::uint32_t layer, std::uint64_t block) {
    if (!ranker_ || !ranker_->owns_representative_keys()) return;
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        for (auto & token : blocks_[index(layer, head)].at(block).tokens) std::vector<Half>().swap(token.key);
    }
}
RepresentativeSeed RepresentativeIndex::builder_seed(std::uint32_t layer) const {
    RepresentativeSeed seed;
    seed.length = lengths_.at(layer);
    for (const auto & token : pending_[layer]) {
        seed.pending_keys.insert(seed.pending_keys.end(), token.keys.begin(), token.keys.end());
        seed.pending_scores.insert(seed.pending_scores.end(), token.scores.begin(), token.scores.end());
    }
    const auto count = static_cast<std::size_t>(geometry_.heads)*budget_.representatives;
    seed.partial_positions.assign(count, UINT64_MAX);
    seed.partial_scores.assign(count, 0);
    seed.partial_keys.assign(count*geometry_.dimension, 0);
    const auto matured = seed.length - std::min<std::uint64_t>(seed.length, budget_.local);
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        const auto & block = blocks_[index(layer, head)].at(matured/geometry_.block_tokens);
        for (std::size_t r = 0; r < block.tokens.size(); ++r) {
            const auto offset = static_cast<std::size_t>(head)*budget_.representatives+r;
            seed.partial_positions[offset] = block.tokens[r].position;
            seed.partial_scores[offset] = block.tokens[r].score;
            std::copy(block.tokens[r].key.begin(), block.tokens[r].key.end(), seed.partial_keys.begin()+offset*geometry_.dimension);
        }
    }
    validate_representative_seed(geometry_, budget_.representatives, budget_.local, seed);
    return seed;
}
void RepresentativeIndex::accept_built(std::uint32_t layer, std::uint64_t position, const std::optional<BuiltRepresentatives> & built) {
    const bool complete = position >= budget_.local && (position-budget_.local+1)%geometry_.block_tokens == 0;
    if (built.has_value() != complete) throw std::runtime_error("representative builder completion boundary");
    if (!built) return;
    const auto block = (position-budget_.local)/geometry_.block_tokens;
    const auto count = static_cast<std::size_t>(geometry_.heads)*budget_.representatives;
    if (built->block != block || built->keys.size() != count*geometry_.dimension || built->positions.size() != count || built->scores.size() != count) {
        throw std::runtime_error("representative builder output shape or block");
    }
    for (const auto key : built->keys) if (!std::isfinite(from_half(key))) throw std::runtime_error("non-finite built representative key");
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        std::set<std::uint64_t> unique;
        for (std::uint32_t r = 0; r < budget_.representatives; ++r) {
            const auto i = static_cast<std::size_t>(head)*budget_.representatives+r;
            const auto p = built->positions[i]; const auto score = built->scores[i];
            if (p/geometry_.block_tokens != block || !unique.insert(p).second || !std::isfinite(score) || (r &&
                (score > built->scores[i-1] || (score == built->scores[i-1] && p < built->positions[i-1])))) {
                throw std::runtime_error("built representative position, score or order");
            }
        }
    }
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        auto & target = blocks_[index(layer, head)].at(block);
        if (target.finalized == geometry_.block_tokens) throw std::runtime_error("representative builder repeated a complete block");
        target.tokens.clear();
        for (std::uint32_t r = 0; r < budget_.representatives; ++r) {
            const auto i = static_cast<std::size_t>(head)*budget_.representatives+r;
            const auto begin = built->keys.begin()+i*geometry_.dimension;
            std::vector<Half> keys;
            if (!ranker_->owns_representative_keys()) keys.assign(begin, begin+geometry_.dimension);
            target.tokens.push_back({built->positions[i], built->scores[i], std::move(keys)});
        }
        target.finalized = geometry_.block_tokens;
    }
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_blocks(
    std::uint32_t layer, const std::vector<float> & query, const BlockCandidates & candidates) const {
    if (query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension) throw std::invalid_argument("block query shape");
    finite_query(query);
    return rank_impl(layer, &query, nullptr, candidates);
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_blocks(
    std::uint32_t layer, DeviceQuery query, const BlockCandidates & candidates) const {
    const auto memory = direct_memory();
    if (!memory) throw std::logic_error("direct device ranking is disabled");
    validate_device_query(query, *memory, query_heads_, geometry_.dimension);
    return rank_impl(layer, nullptr, &query, candidates);
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_impl(
    std::uint32_t layer, const std::vector<float> * query, const DeviceQuery * device, const BlockCandidates & candidates) const {
    if (layer >= geometry_.layers || candidates.blocks.size() != candidates.tokens.size() || candidates.remaining > budget_.total) {
        throw std::invalid_argument("block ranking shape or budget");
    }
    std::uint64_t available = 0;
    for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
        if ((i && candidates.blocks[i] <= candidates.blocks[i-1]) || candidates.blocks[i] >= geometry_.blocks_per_head() ||
            !ready(layer, candidates.blocks[i]) || candidates.tokens[i] == 0 || candidates.tokens[i] > geometry_.block_tokens) {
            throw std::invalid_argument("invalid block candidate");
        }
        available += candidates.tokens[i];
    }
    const auto needed = std::min<std::uint64_t>(available, candidates.remaining);
    std::vector<std::vector<std::uint64_t>> result(geometry_.heads);
    if (!needed) return result;
    if (device) result = ranker_->rank_device(layer, *device, candidates);
    else if (ranker_) result = ranker_->rank(layer, *query, candidates);
    else for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        std::vector<std::pair<double, std::uint64_t>> best;
        const auto better = [](const auto & a, const auto & b) { return a.first == b.first ? a.second < b.second : a.first > b.first; };
        for (const auto block : candidates.blocks) {
            best.emplace_back(score(layer, head, block, *query), block);
            std::push_heap(best.begin(), best.end(), better);
            if (best.size() > candidates.remaining) { std::pop_heap(best.begin(), best.end(), better); best.pop_back(); }
        }
        std::sort(best.begin(), best.end(), better);
        std::uint64_t covered = 0;
        for (const auto & item : best) {
            result[head].push_back(item.second);
            const auto offset = std::lower_bound(candidates.blocks.begin(), candidates.blocks.end(), item.second) - candidates.blocks.begin();
            covered += candidates.tokens[offset];
            if (covered >= needed) break;
        }
    }
    if (result.size() != geometry_.heads) throw std::runtime_error("ranker returned wrong head count");
    for (const auto & head : result) {
        std::set<std::uint64_t> unique;
        std::uint64_t covered = 0;
        for (const auto block : head) {
            const auto found = std::lower_bound(candidates.blocks.begin(), candidates.blocks.end(), block);
            if (found == candidates.blocks.end() || *found != block || !unique.insert(block).second || covered >= needed) {
                throw std::runtime_error("ranker returned invalid, duplicate or excess block");
            }
            covered += candidates.tokens[found - candidates.blocks.begin()];
        }
        if (covered < needed) throw std::runtime_error("ranker returned insufficient token coverage");
    }
    return result;
}
std::vector<std::uint64_t> RepresentativeIndex::representative_positions(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const {
    std::vector<std::uint64_t> result;
    for (const auto & token : blocks_.at(index(layer, head)).at(block).tokens) result.push_back(token.position);
    return result;
}
std::uint64_t RepresentativeIndex::resident_bytes() const {
    std::uint64_t total = lengths_.capacity() * sizeof(std::uint64_t) + (query_pending_.capacity() + 7) / 8;
    for (const auto & blocks : blocks_) {
        total += blocks.capacity() * sizeof(Block);
        for (const auto & block : blocks) {
            total += block.tokens.capacity() * sizeof(Representative);
            for (const auto & token : block.tokens) total += token.key.capacity() * sizeof(Half);
        }
    }
    for (const auto & pending : pending_) for (const auto & token : pending) {
        total += sizeof(Pending) + token.keys.capacity() * sizeof(Half) + token.scores.capacity() * sizeof(double);
    }
    return total;
}
template<class Query> static std::vector<HeadSelection> select_blocks_impl(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const Query & query) {
    budget.validate(geometry);
    if (position >= geometry.capacity_tokens) throw std::out_of_range("selection position");
    const auto count = position + 1;
    std::vector<HeadSelection> result(geometry.heads);
    std::set<std::uint64_t> deterministic;
    if (count <= budget.total) {
        for (std::uint64_t i = 0; i < count; ++i) deterministic.insert(i);
    } else {
        for (std::uint64_t i = 0; i < std::min<std::uint64_t>(budget.initial, count); ++i) deterministic.insert(i);
        for (auto i = count - std::min<std::uint64_t>(budget.local, count); i < count; ++i) deterministic.insert(i);
    }
    BlockCandidates candidates;
    candidates.remaining = budget.total - static_cast<std::uint32_t>(deterministic.size());
    if (count > budget.total && candidates.remaining) {
        for (std::uint64_t block = 0; block <= position / geometry.block_tokens; ++block) {
            if (!representatives.ready(layer, block)) continue;
            const auto first = block * geometry.block_tokens;
            const auto last = std::min<std::uint64_t>(count, first + geometry.block_tokens);
            std::uint32_t cost = 0;
            for (auto p = first; p < last; ++p) if (deterministic.count(p) == 0) ++cost;
            if (cost) { candidates.blocks.push_back(block); candidates.tokens.push_back(cost); }
        }
    }
    const auto ranked = representatives.rank_blocks(layer, query, candidates);
    for (std::uint32_t head = 0; head < geometry.heads; ++head) {
        auto chosen = deterministic;
        for (const auto block : ranked[head]) {
            const auto first = block * geometry.block_tokens;
            for (auto p = first; p < std::min<std::uint64_t>(count, first + geometry.block_tokens) && chosen.size() < budget.total; ++p) {
                chosen.insert(p);
            }
            if (chosen.size() >= budget.total) break;
        }
        result[head].positions.assign(chosen.begin(), chosen.end());
        for (const auto p : chosen) {
            const auto block = p / geometry.block_tokens;
            if (result[head].blocks.empty() || result[head].blocks.back() != block) result[head].blocks.push_back(block);
        }
    }
    return result;
}
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const std::vector<float> & query) {
    if (query.size() != static_cast<std::size_t>(representatives.query_heads())*geometry.dimension) throw std::invalid_argument("selection query shape");
    finite_query(query);
    return select_blocks_impl(geometry, budget, representatives, layer, position, query);
}
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, DeviceQuery query) {
    const auto memory = representatives.direct_memory();
    if (!memory) throw std::logic_error("direct device selection is disabled");
    validate_device_query(query, *memory, representatives.query_heads(), geometry.dimension);
    return select_blocks_impl(geometry, budget, representatives, layer, position, query);
}
std::vector<float> streamed_attention(KvStore & store, std::uint32_t layer, std::uint64_t position,
                                      std::uint32_t query_heads, const std::vector<float> & query, float scale,
                                      const std::vector<HeadSelection> * selected) {
    const auto & geometry = store.geometry();
    if (query_heads == 0 || query_heads % geometry.heads != 0 ||
        query.size() != static_cast<std::size_t>(query_heads) * geometry.dimension ||
        position >= store.length(layer) || !std::isfinite(scale) || scale <= 0 ||
        (selected && selected->size() != geometry.heads)) throw std::invalid_argument("attention shape or position mismatch");
    finite_query(query);
    const auto group = query_heads / geometry.heads;
    std::vector<float> output(query.size(), 0);
    KvBlock block;
    for (std::uint32_t head = 0; head < geometry.heads; ++head) {
        std::vector<double> maxima(group, -std::numeric_limits<double>::infinity()), sums(group, 0);
        std::vector<double> weighted(static_cast<std::size_t>(group) * geometry.dimension, 0);
        const auto consume = [&](std::uint64_t p, std::uint32_t in_block) {
            if (p > position || in_block >= block.valid_tokens) throw std::invalid_argument("future or invalid attention position");
            for (std::uint32_t q = 0; q < group; ++q) {
                double score = 0;
                for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                    score += static_cast<double>(query[(head * group + q) * geometry.dimension + d]) *
                             from_half(block.key(in_block)[d]);
                }
                score *= scale;
                if (!std::isfinite(score)) throw std::runtime_error("non-finite attention score");
                const double maximum = std::max(maxima[q], score);
                const double old_scale = std::exp(maxima[q] - maximum);
                const double weight = std::exp(score - maximum);
                for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                    const auto value = from_half(block.value(in_block)[d]);
                    if (!std::isfinite(value)) throw std::runtime_error("non-finite attention value");
                    weighted[q * geometry.dimension + d] = weighted[q * geometry.dimension + d] * old_scale + weight * value;
                }
                sums[q] = sums[q] * old_scale + weight;
                maxima[q] = maximum;
            }
        };
        if (selected) {
            std::uint64_t loaded = std::numeric_limits<std::uint64_t>::max();
            std::uint64_t previous = 0;
            bool first = true;
            for (const auto p : selected->at(head).positions) {
                if (!first && p <= previous) throw std::invalid_argument("selected positions must be sorted and unique");
                first = false;
                previous = p;
                const auto id = p / geometry.block_tokens;
                if (id != loaded) { store.read(layer, head, id, block); loaded = id; }
                consume(p, static_cast<std::uint32_t>(p % geometry.block_tokens));
            }
        } else {
            for (std::uint64_t id = 0; id <= position / geometry.block_tokens; ++id) {
                store.read(layer, head, id, block);
                for (std::uint32_t i = 0; i < block.valid_tokens && block.first_position + i <= position; ++i) consume(block.first_position + i, i);
            }
        }
        for (std::uint32_t q = 0; q < group; ++q) {
            if (!(sums[q] > 0)) throw std::invalid_argument("attention selection is empty");
            for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                output[(head * group + q) * geometry.dimension + d] = static_cast<float>(weighted[q * geometry.dimension + d] / sums[q]);
            }
        }
    }
    return output;
}
} // namespace solid
