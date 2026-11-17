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
    blocks_(static_cast<std::size_t>(geometry.layers) * geometry.heads), lengths_(geometry.layers, 0) {
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
    if (position != lengths_.at(layer) || position >= geometry_.capacity_tokens ||
        query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension ||
        keys.size() != static_cast<std::size_t>(geometry_.heads) * geometry_.dimension) {
        throw std::invalid_argument("representative observation shape or position mismatch");
    }
    finite_query(query);
    for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite key");
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
        pending.pop_front();
    }
    pending.push_back({position, keys, std::vector<double>(geometry_.heads, 0)});
    ++lengths_[layer];
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
    const auto group = query_heads_ / geometry_.heads;
    double value = 0;
    for (const auto & token : tokens) {
        for (std::uint32_t q = head * group; q < (head + 1) * group; ++q) {
            for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
                value += static_cast<double>(query[q * geometry_.dimension + d]) * from_half(token.key[d]) / group;
            }
        }
    }
    if (!std::isfinite(value)) throw std::runtime_error("non-finite block relevance");
    return value;
}
std::vector<std::uint64_t> RepresentativeIndex::representative_positions(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const {
    std::vector<std::uint64_t> result;
    for (const auto & token : blocks_.at(index(layer, head)).at(block).tokens) result.push_back(token.position);
    return result;
}
std::uint64_t RepresentativeIndex::resident_bytes() const {
    std::uint64_t total = lengths_.capacity() * sizeof(std::uint64_t);
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
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const std::vector<float> & query) {
    budget.validate(geometry);
    if (position >= geometry.capacity_tokens) throw std::out_of_range("selection position");
    if (query.size() != static_cast<std::size_t>(representatives.query_heads()) * geometry.dimension) {
        throw std::invalid_argument("selection query shape");
    }
    finite_query(query);
    const auto count = position + 1;
    std::vector<HeadSelection> result(geometry.heads);
    for (std::uint32_t head = 0; head < geometry.heads; ++head) {
        std::set<std::uint64_t> chosen;
        if (count <= budget.total) {
            for (std::uint64_t i = 0; i < count; ++i) chosen.insert(i);
        } else {
            for (std::uint64_t i = 0; i < std::min<std::uint64_t>(budget.initial, count); ++i) chosen.insert(i);
            for (auto i = count - std::min<std::uint64_t>(budget.local, count); i < count; ++i) chosen.insert(i);
            std::vector<std::pair<double, std::uint64_t>> candidates;
            const auto limit = budget.total - chosen.size();
            const auto better = [](const auto & a, const auto & b) {
                return a.first == b.first ? a.second < b.second : a.first > b.first;
            };
            candidates.reserve(limit + 1);
            for (std::uint64_t block = 0; block <= position / geometry.block_tokens; ++block) {
                if (!representatives.ready(layer, block)) continue;
                const auto first = block * geometry.block_tokens;
                const auto last = std::min<std::uint64_t>(count, first + geometry.block_tokens);
                bool needed = false;
                for (auto p = first; p < last; ++p) if (chosen.count(p) == 0) { needed = true; break; }
                if (needed && limit > 0) {
                    candidates.emplace_back(representatives.score(layer, head, block, query), block);
                    std::push_heap(candidates.begin(), candidates.end(), better);
                    if (candidates.size() > limit) {
                        std::pop_heap(candidates.begin(), candidates.end(), better);
                        candidates.pop_back();
                    }
                }
            }
            std::sort(candidates.begin(), candidates.end(), better);
            for (const auto & candidate : candidates) {
                const auto first = candidate.second * geometry.block_tokens;
                for (auto p = first; p < std::min<std::uint64_t>(count, first + geometry.block_tokens) && chosen.size() < budget.total; ++p) {
                    chosen.insert(p);
                }
                if (chosen.size() >= budget.total) break;
            }
        }
        result[head].positions.assign(chosen.begin(), chosen.end());
        for (const auto p : chosen) {
            const auto block = p / geometry.block_tokens;
            if (result[head].blocks.empty() || result[head].blocks.back() != block) result[head].blocks.push_back(block);
        }
    }
    return result;
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
