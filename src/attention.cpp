// 组织初始、近期与历史上下文，调用评分器选块，并实现 CPU 流式 attention 参考。
#include "solid/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace solid {
namespace {
void finite_query(const std::vector<float> & query) {
    for (float value : query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite query");
}
// InfLLM's local_score and global_remainder_local_score use the FP16 K
// dtype: round each chunk's score first, then round the in-place sum.
double accumulate_infllm_score(double previous, float chunk) {
    return from_half(to_half(static_cast<float>(previous) + from_half(to_half(chunk))));
}
}
void SparseBudget::validate(const KvGeometry & geometry) const {
    if (total == 0 || total > geometry.capacity_tokens || local == 0 || static_cast<std::uint64_t>(initial) + local > total ||
        representatives == 0 || representatives > geometry.block_tokens ||
        (infllm_attention && (!shared_heads || local > 2048))) {
        throw std::invalid_argument("invalid sparse token budget or representative count");
    }
}
std::uint32_t SparseBudget::prefill_group_tokens() const {
    return std::min(infllm_prefill_chunk_tokens, local);
}
void RelativeRope::rotate(float * row, std::uint32_t position) const {
    const auto half = dimensions / 2;
    const auto step = std::pow(base, -2.f / dimensions);
    float angle = static_cast<float>(position);
    for (std::uint32_t i = 0; i < half; ++i) {
        const auto x = split_halves ? i : 2 * i;
        const auto y = split_halves ? i + half : 2 * i + 1;
        const auto a = row[x], b = row[y];
        const auto cosine = std::cos(angle), sine = std::sin(angle);
        row[x] = a * cosine - b * sine;
        row[y] = a * sine + b * cosine;
        angle *= step;
    }
}
void mean_infllm_query(const float * queries, float * mean, std::size_t width, std::uint32_t count) {
    if (!queries || !mean || !width || !count || count > 512) throw std::invalid_argument("InfLLM mean query shape");
    for (std::size_t d = 0; d < width; ++d) {
        float sum = 0;
        for (std::uint32_t i = 0; i < count; ++i) sum += queries[i * width + d];
        mean[d] = sum / count;
    }
}
std::uint32_t SparseBudget::cache_slots(const KvGeometry & geometry) const {
    const auto block = geometry.block_tokens;
    auto slots = (static_cast<std::uint64_t>(total) + block - 1) / block + 3;
    if (shared_heads && initial % block) slots += (total - initial - local) / block;
    return static_cast<std::uint32_t>(std::min(slots, geometry.blocks_per_head()));
}
std::uint64_t SparseBudget::complete_history_blocks(const KvGeometry & geometry, std::uint64_t position) const {
    const auto fixed = static_cast<std::uint64_t>(initial) + local;
    // The current query still consumes the remainder before publishing new blocks.
    return position > fixed ? (position - fixed) / geometry.block_tokens : 0;
}
std::uint64_t SparseBudget::recent_start(const KvGeometry & geometry, std::uint64_t position) const {
    const auto count = position + 1;
    const auto window = count - std::min<std::uint64_t>(count, local);
    if (!shared_heads || count <= total) return window;
    const auto remainder = initial + complete_history_blocks(geometry, position) * geometry.block_tokens;
    return std::max(remainder, count - (total - initial));
}
RepresentativeIndex::RepresentativeIndex(KvGeometry geometry, std::uint32_t query_heads, SparseBudget budget) :
    geometry_(geometry), query_heads_(query_heads), kv_heads_(geometry.heads), budget_(budget), pending_(geometry.layers),
    lengths_(geometry.layers, 0), complete_blocks_(geometry.layers, 0), query_pending_(geometry.layers, false), probability_lengths_(geometry.layers, 0) {
    geometry_.validate();
    budget_.validate(geometry_);
    if (query_heads == 0 || query_heads % geometry.heads != 0) throw std::invalid_argument("invalid GQA head mapping");
    if (budget_.shared_heads) geometry_.heads = query_heads_;
    blocks_.resize(static_cast<std::size_t>(geometry_.layers) * geometry_.heads);
}
std::size_t RepresentativeIndex::index(std::uint32_t layer, std::uint32_t head) const {
    if (layer >= geometry_.layers || head >= geometry_.heads) throw std::out_of_range("representative identity");
    return static_cast<std::size_t>(layer) * geometry_.heads + head;
}
RepresentativeIndex::Block & RepresentativeIndex::host_block(std::uint32_t layer, std::uint32_t head, std::uint64_t block) {
    if (device_only_representatives_ || block >= geometry_.blocks_per_head()) throw std::out_of_range("host representative block");
    auto & blocks = blocks_.at(index(layer, head));
    if (blocks.size() <= block) blocks.resize(block + 1);
    return blocks[block];
}
void RepresentativeIndex::observe(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                  const std::vector<Half> & keys) {
    if (keys.size() != static_cast<std::size_t>(kv_heads_) * geometry_.dimension) throw std::invalid_argument("representative key shape");
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
    if (budget_.infllm_attention) {
        begin_probability_query(layer, position);
        return;
    }
    const auto origin = history_start();
    if (position < origin) {
        query_pending_[layer] = true;
        return;
    }
    if (ranker_ && ranker_->builder_window()) {
        accept_built(layer, position - origin, ranker_->build_query(layer, position - origin, query));
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
    const auto key_group = geometry_.heads / kv_heads_;
    for (auto & token : pending) {
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
                const auto offset = head * geometry_.dimension + d;
                const auto key_offset = (head / key_group) * geometry_.dimension + d;
                token.scores[head] += mean_query[offset] * from_half(token.keys[key_offset]);
            }
        }
    }
    if (position >= budget_.local) finalize_before(layer, position - budget_.local + 1, budget_.local);
    query_pending_[layer] = true;
}
void RepresentativeIndex::finalize_before(std::uint32_t layer, std::uint64_t first_position, double divisor) {
    const auto origin = history_start();
    const auto key_group = geometry_.heads / kv_heads_;
    auto & pending = pending_[layer];
    while (!pending.empty() && pending.front().position < first_position) {
        auto & token = pending.front();
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            auto & block = host_block(layer, head, (token.position - origin) / geometry_.block_tokens);
            Representative representative{token.position, token.scores[head] / divisor, {}};
            const auto first = token.keys.begin() + (head / key_group) * geometry_.dimension;
            representative.key.assign(first, first + geometry_.dimension);
            block.tokens.push_back(std::move(representative));
            std::sort(block.tokens.begin(), block.tokens.end(), [](const auto & a, const auto & b) {
                return a.score == b.score ? a.position < b.position : a.score > b.score;
            });
            if (block.tokens.size() > budget_.representatives) block.tokens.resize(budget_.representatives);
            ++block.finalized;
        }
        const auto block = (token.position - origin) / geometry_.block_tokens;
        if (blocks_[index(layer, 0)][block].finalized == geometry_.block_tokens) {
            if (block != complete_blocks_[layer]) throw std::logic_error("representative completion has a gap");
            ++complete_blocks_[layer];
            if (ranker_) {
                publish(*ranker_, layer, block, 1);
                release_published_keys(layer, block);
            }
        }
        pending.pop_front();
    }
}
void RepresentativeIndex::observe_probabilities(std::uint32_t layer, std::uint64_t position,
                                                  const std::vector<float> & probabilities) {
    const auto count = std::min<std::uint64_t>(position + 1, budget_.local);
    const auto first = position + 1 - count;
    if (!budget_.infllm_attention || device_probabilities_ || lengths_.at(layer) != position + 1 || probability_lengths_[layer] != position ||
        probabilities.size() != count * query_heads_) throw std::invalid_argument("representative probability frame");
    for (const auto probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0 || probability > 1) {
            throw std::invalid_argument("non-finite or out-of-range attention probability");
        }
    }
    for (auto & token : pending_[layer]) {
        if (token.position < first) continue;
        for (std::uint32_t head = 0; head < query_heads_; ++head) {
            token.scores[head] = accumulate_infllm_score(token.scores[head], probabilities[head * count + token.position - first]);
        }
    }
    finalize_before(layer, first, 1);
    ++probability_lengths_[layer];
}
void RepresentativeIndex::append_keys(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & keys) {
    if (position != lengths_.at(layer) || !query_pending_.at(layer) || keys.size() != static_cast<std::size_t>(kv_heads_) * geometry_.dimension) {
        throw std::invalid_argument("representative append requires its query first");
    }
    for (const auto key : keys) if (!std::isfinite(from_half(key))) throw std::invalid_argument("non-finite key");
    if (position >= history_start() && !device_probabilities_) {
        if (ranker_ && ranker_->builder_window()) ranker_->build_append(layer, position - history_start(), keys);
        else pending_[layer].push_back({position, keys, std::vector<double>(geometry_.heads, 0)});
    }
    query_pending_[layer] = false;
    ++lengths_[layer];
}
void RepresentativeIndex::observe_prefill_scores(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
                                                const std::vector<Half> & kv, const std::vector<float> & key_scores) {
    const auto width = static_cast<std::size_t>(kv_heads_) * geometry_.dimension;
    if (!budget_.infllm_attention || device_probabilities_ || !count || count > budget_.prefill_group_tokens() || first != lengths_.at(layer) ||
        probability_lengths_[layer] != first || query_pending_[layer] ||
        first >= geometry_.capacity_tokens || count > geometry_.capacity_tokens - first ||
        kv.size() != count * 2 * width) {
        throw std::invalid_argument("InfLLM prefill representative frame");
    }
    const auto end = first + count;
    const auto origin = first + 1 - std::min<std::uint64_t>(first + 1, budget_.local);
    if (key_scores.size() != (end - origin) * query_heads_) throw std::invalid_argument("InfLLM prefill key scores shape");
    for (const auto score : key_scores) {
        if (!std::isfinite(score) || score < 0 || score > count) throw std::invalid_argument("invalid InfLLM prefill key score");
    }
    std::vector<Half> keys(width);
    for (std::uint32_t i = 0; i < count; ++i) {
        for (std::uint32_t h = 0; h < kv_heads_; ++h) {
            const auto offset = static_cast<std::size_t>(h) * geometry_.dimension;
            std::copy_n(kv.data() + (i * width + offset) * 2, geometry_.dimension, keys.data() + offset);
        }
        query_pending_[layer] = true;
        append_keys(layer, first + i, keys);
    }
    for (auto & token : pending_[layer]) {
        if (token.position < origin) continue;
        for (std::uint32_t head = 0; head < query_heads_; ++head) {
            token.scores[head] = accumulate_infllm_score(token.scores[head], key_scores[(token.position - origin) * query_heads_ + head]);
        }
    }
    finalize_before(layer, end - std::min<std::uint64_t>(end, budget_.local), 1);
    probability_lengths_[layer] = end;
}
void RepresentativeIndex::observe_query_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query,bool will_rank) {
    const auto memory = direct_memory();
    if (!memory || position != lengths_.at(layer) || position >= geometry_.capacity_tokens || query_pending_.at(layer)) {
        throw std::invalid_argument("device representative query state");
    }
    validate_device_query(query, *memory, query_heads_, geometry_.dimension);
    if (position >= history_start()) {
        const auto relative = position - history_start();
        accept_built(layer, relative, will_rank ? ranker_->build_query_for_rank_device(layer, relative, query) :
                                                  ranker_->build_query_device(layer, relative, query));
    }
    query_pending_[layer] = true;
}
void RepresentativeIndex::begin_probability_query(std::uint32_t layer, std::uint64_t position) {
    finish_device_probabilities();
    if (!budget_.infllm_attention || position != lengths_.at(layer) || position >= geometry_.capacity_tokens ||
        query_pending_.at(layer) || probability_lengths_[layer] != position) {
        throw std::logic_error("InfLLM query requires all preceding keys and attention probabilities");
    }
    query_pending_[layer] = true;
}
void RepresentativeIndex::finish_query_for_rank_device() {
    if (ranker_) ranker_->finish_query_for_rank_device();
}
void RepresentativeIndex::append_keys_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys,void * compute_stream) {
    const auto memory = direct_memory();
    if (!memory || position != lengths_.at(layer) || !query_pending_.at(layer)) throw std::invalid_argument("device representative append state");
    validate_device_keys(keys, *memory, kv_heads_, geometry_.dimension);
    if (position >= history_start()) {
        const auto relative = position - history_start();
        if (compute_stream) ranker_->build_append_device_on_compute(layer, relative, keys, compute_stream);
        else ranker_->build_append_device(layer, relative, keys);
    }
    query_pending_[layer] = false;
    ++lengths_[layer];
}
void RepresentativeIndex::observe_batch_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,DeviceQuery query,DeviceKvBatch kv) {
    const auto memory=direct_memory();
    if (!memory || !tokens || tokens>geometry_.capacity_tokens || first>geometry_.capacity_tokens-tokens ||
        lengths_.at(layer)!=first || query_pending_.at(layer)) throw std::invalid_argument("device representative batch state");
    validate_device_query(query,*memory,query_heads_,geometry_.dimension,tokens);
    validate_device_kv(kv,*memory,kv_heads_,geometry_.dimension,tokens);
    const auto end = first + tokens;
    const auto active_first = std::max(first, history_start());
    if (active_first >= end) {
        lengths_[layer] = end;
        return;
    }
    const auto skipped = active_first - first;
    query.data += skipped * (query.elements / tokens);
    kv.data += skipped * (kv.elements / tokens);
    query.elements -= skipped * (query.elements / tokens);
    kv.elements -= skipped * (kv.elements / tokens);
    const auto relative = active_first - history_start();
    const auto count = static_cast<std::uint32_t>(end - active_first);
    auto built = ranker_->build_batch_device(layer, relative, count, query, kv);
    std::size_t index=0;
    for (std::uint32_t i=0;i<count;++i) {
        const auto p=relative+i;
        const bool complete=p>=budget_.local && (p-budget_.local+1)%geometry_.block_tokens==0;
        if (complete) {
            if (index==built.size()) throw std::runtime_error("device representative batch omitted a complete block");
            accept_built(layer,p,std::optional<BuiltRepresentatives>{std::move(built[index++])});
        }
    }
    if (index!=built.size()) throw std::runtime_error("device representative batch returned excess blocks");
    lengths_[layer] = end;
}
bool RepresentativeIndex::ready(std::uint32_t layer, std::uint64_t block) const {
    if (block >= geometry_.blocks_per_head()) throw std::out_of_range("representative block");
    return block < complete_blocks_.at(layer);
}
double RepresentativeIndex::score(std::uint32_t layer, std::uint32_t head, std::uint64_t block,
                                  const std::vector<float> & query) const {
    if (query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension || !ready(layer, block)) {
        throw std::invalid_argument("block representative is not ready or query has wrong shape");
    }
    const auto slot = index(layer, head);
    static const std::vector<Representative> empty;
    const auto & tokens = device_only_representatives_ ? empty : blocks_.at(slot).at(block).tokens;
    std::vector<Half> device_keys;
    if (device_only_representatives_ || std::any_of(tokens.begin(), tokens.end(), [](const auto & token) { return token.key.empty(); })) {
        if (!ranker_ || !ranker_->owns_representative_keys()) throw std::logic_error("representative key owner is missing");
        device_keys = ranker_->read_representatives(layer, block);
        const auto elements = static_cast<std::size_t>(geometry_.heads) *
            (budget_.infllm_attention ? 1 : budget_.representatives) * geometry_.dimension;
        if (device_keys.size() != elements) throw std::runtime_error("representative readback shape mismatch");
        for (const auto key : device_keys) if (!std::isfinite(from_half(key))) throw std::runtime_error("non-finite representative readback");
    }
    const auto group = query_heads_ / geometry_.heads;
    if (budget_.infllm_attention && !device_keys.empty()) {
        double dot = 0;
        for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
            const auto i = static_cast<std::size_t>(head) * geometry_.dimension + d;
            dot += static_cast<double>(query[i]) * from_half(device_keys[i]);
        }
        return dot;
    }
    double value = 0;
    if (budget_.shared_heads && tokens.size() > 1) {
        for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
            float mean = 0;
            for (std::size_t r = 0; r < tokens.size(); ++r) {
                const auto offset = (static_cast<std::size_t>(head) * budget_.representatives + r) * geometry_.dimension + d;
                mean += from_half(device_keys.empty() ? tokens[r].key[d] : device_keys[offset]);
            }
            const auto key = from_half(to_half(mean / tokens.size()));
            value += static_cast<double>(query[static_cast<std::size_t>(head) * geometry_.dimension + d]) * key;
        }
        if (!std::isfinite(value)) throw std::runtime_error("non-finite block relevance");
        return value;
    }
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
    const auto stride = static_cast<std::uint64_t>(geometry_.heads) *
        (budget_.infllm_attention ? 1 : budget_.representatives) * geometry_.dimension;
    if (count && stride > SIZE_MAX / count) throw std::overflow_error("representative publication size");
    keys.reserve(static_cast<std::size_t>(count * stride));
    for (auto block = first; block < first + count; ++block) {
        if (!ready(layer, block)) throw std::logic_error("publishing an unfinished representative block");
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            const auto & tokens = blocks_[index(layer, head)].at(block).tokens;
            if (tokens.size() != budget_.representatives) throw std::logic_error("finalized representative count");
            if (budget_.infllm_attention) {
                for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
                    float mean = 0;
                    for (const auto & token : tokens) mean += from_half(token.key[d]);
                    keys.push_back(to_half(mean / tokens.size()));
                }
            } else {
                for (const auto & token : tokens) keys.insert(keys.end(), token.key.begin(), token.key.end());
            }
        }
    }
    if (count) ranker.publish(layer, first, count, keys);
}
void RepresentativeIndex::set_ranker(std::unique_ptr<BlockRanker> ranker) {
    if (!ranker || ranker_) throw std::logic_error("representative ranker already configured or empty");
    if (std::any_of(query_pending_.begin(), query_pending_.end(), [](bool value) { return value; })) {
        throw std::logic_error("representative ranker requires a query boundary");
    }
    if (ranker->builder_window() && (budget_.infllm_attention || ranker->builder_window() != budget_.local)) {
        throw std::invalid_argument("representative builder window differs");
    }
    device_probabilities_ = ranker->device_probabilities();
    if (device_probabilities_ && (!budget_.infllm_attention ||
        !std::all_of(lengths_.begin(), lengths_.end(), [](auto length) { return length == 0; }))) {
        throw std::invalid_argument("device probability builder requires an empty InfLLM index");
    }
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        std::uint64_t count = 0;
        while (count < geometry_.blocks_per_head() && ready(layer, count)) ++count;
        publish(*ranker, layer, 0, count);
        if (ranker->builder_window()) ranker->seed_builder(layer, builder_seed(layer));
    }
    ranker_ = std::move(ranker);
    if (ranker_->builder_window()) for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        pending_[layer].clear();
        const auto length = lengths_[layer] - std::min(lengths_[layer], history_start());
        const auto matured = length - std::min<std::uint64_t>(length, budget_.local);
        for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
            auto & partial = host_block(layer, head, matured/geometry_.block_tokens);
            partial.tokens.clear(); partial.finalized = 0;
        }
    }
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        for (std::uint64_t block = 0; block < geometry_.blocks_per_head() && ready(layer, block); ++block) {
            release_published_keys(layer, block);
        }
    }
    device_only_representatives_ = device_probabilities_ && !ranker_->host_representative_metadata();
    if (device_only_representatives_) std::vector<std::vector<Block>>().swap(blocks_);
    if (device_probabilities_) std::vector<std::deque<Pending>>().swap(pending_);
}
void RepresentativeIndex::observe_device_probabilities(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceKvBatch kv, DeviceQuery scores, bool key_major, void * compute_stream, bool on_compute) {
    if (pending_probability_) throw std::logic_error("previous device probability receipt is pending");
    if (on_compute && !compute_stream) throw std::invalid_argument("probability compute queue is missing");
    if (!device_probabilities_ || !count || probability_lengths_.at(layer) != first ||
        query_pending_.at(layer) || lengths_.at(layer) != first + count) {
        throw std::logic_error("device probability receipt has no matching appended keys");
    }
    const auto origin = first + 1 - std::min<std::uint64_t>(first + 1, budget_.local);
    const ProbabilityFrame frame{layer, first, count, origin, kv, scores, key_major};
    if (compute_stream) {
        ranker_->submit_probability_update(frame, compute_stream, on_compute);
        pending_probability_ = ProbabilityReceipt{layer, first + count};
        return;
    }
    accept_probability_blocks(layer, ranker_->probability_update(frame));
    probability_lengths_[layer] = first + count;
}
void RepresentativeIndex::finish_device_probabilities() {
    if (!pending_probability_) return;
    const auto receipt = *pending_probability_;
    accept_probability_blocks(receipt.layer, ranker_->finish_probability_update());
    probability_lengths_[receipt.layer] = receipt.end;
    pending_probability_.reset();
}
void RepresentativeIndex::accept_probability_blocks(std::uint32_t layer, const std::vector<BuiltRepresentatives> & built) {
    for (const auto & item : built) {
        if (item.block != complete_blocks_.at(layer)) throw std::runtime_error("device representative completion has a gap");
        if (device_only_representatives_) {
            if (!item.keys.empty() || !item.positions.empty() || !item.scores.empty()) {
                throw std::runtime_error("unexpected host representative payload");
            }
            ++complete_blocks_[layer];
            continue;
        }
        if (item.positions.size() != static_cast<std::size_t>(geometry_.heads) * budget_.representatives ||
            item.scores.size() != item.positions.size()) throw std::runtime_error("device representative metadata shape");
        for (std::uint32_t h = 0; h < geometry_.heads; ++h) {
            auto & block = host_block(layer, h, item.block);
            if (block.finalized) throw std::runtime_error("device representative block repeated");
            for (std::uint32_t r = 0; r < budget_.representatives; ++r) {
                const auto i = static_cast<std::size_t>(h) * budget_.representatives + r;
                if ((item.positions[i] - history_start()) / geometry_.block_tokens != item.block ||
                    !std::isfinite(item.scores[i])) throw std::runtime_error("invalid device representative metadata");
                block.tokens.push_back({item.positions[i], item.scores[i], {}});
            }
            block.finalized = geometry_.block_tokens;
        }
        ++complete_blocks_[layer];
    }
}
void RepresentativeIndex::observe_prefill_device_scores(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceKvBatch kv, DeviceQuery scores) {
    const auto memory = rank_memory();
    if (!device_probabilities_ || !memory || !count || count > budget_.prefill_group_tokens() || first != lengths_.at(layer) || query_pending_[layer] ||
        probability_lengths_[layer] != first || first >= geometry_.capacity_tokens || count > geometry_.capacity_tokens - first) {
        throw std::invalid_argument("device prefill representative frame");
    }
    validate_device_kv(kv, *memory, kv_heads_, geometry_.dimension, count);
    // The device probability builder validates and retains the keys in its ring.
    // Only the append position belongs to the host index in this mode.
    lengths_[layer] += count;
    observe_device_probabilities(layer, first, count, kv, scores, true);
}
void RepresentativeIndex::release_published_keys(std::uint32_t layer, std::uint64_t block) {
    if (!ranker_ || !ranker_->owns_representative_keys()) return;
    for (std::uint32_t head = 0; head < geometry_.heads; ++head) {
        for (auto & token : blocks_[index(layer, head)].at(block).tokens) std::vector<Half>().swap(token.key);
    }
}
RepresentativeSeed RepresentativeIndex::builder_seed(std::uint32_t layer) const {
    RepresentativeSeed seed;
    seed.length = lengths_.at(layer) - std::min(lengths_.at(layer), history_start());
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
        const auto & blocks = blocks_[index(layer, head)];
        if (matured/geometry_.block_tokens >= blocks.size()) continue;
        const auto & block = blocks[matured/geometry_.block_tokens];
        for (std::size_t r = 0; r < block.tokens.size(); ++r) {
            const auto offset = static_cast<std::size_t>(head)*budget_.representatives+r;
            seed.partial_positions[offset] = block.tokens[r].position - history_start();
            seed.partial_scores[offset] = block.tokens[r].score;
            std::copy(block.tokens[r].key.begin(), block.tokens[r].key.end(), seed.partial_keys.begin()+offset*geometry_.dimension);
        }
    }
    validate_representative_seed(geometry_, budget_.representatives, budget_.local, seed, kv_heads_);
    return seed;
}
void RepresentativeIndex::accept_built(std::uint32_t layer, std::uint64_t position, const std::optional<BuiltRepresentatives> & built) {
    const bool complete = position >= budget_.local && (position-budget_.local+1)%geometry_.block_tokens == 0;
    if (built.has_value() != complete) throw std::runtime_error("representative builder completion boundary");
    if (!built) return;
    const auto block = (position-budget_.local)/geometry_.block_tokens;
    if (block != complete_blocks_.at(layer)) throw std::runtime_error("device representative completion has a gap");
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
        auto & target = host_block(layer, head, block);
        if (target.finalized == geometry_.block_tokens) throw std::runtime_error("representative builder repeated a complete block");
        target.tokens.clear();
        for (std::uint32_t r = 0; r < budget_.representatives; ++r) {
            const auto i = static_cast<std::size_t>(head)*budget_.representatives+r;
            const auto begin = built->keys.begin()+i*geometry_.dimension;
            std::vector<Half> keys;
            if (!ranker_->owns_representative_keys()) keys.assign(begin, begin+geometry_.dimension);
            target.tokens.push_back({history_start() + built->positions[i], built->scores[i], std::move(keys)});
        }
        target.finalized = geometry_.block_tokens;
    }
    ++complete_blocks_[layer];
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_blocks(
    std::uint32_t layer, const std::vector<float> & query, const BlockCandidates & candidates) const {
    if (query.size() != static_cast<std::size_t>(query_heads_) * geometry_.dimension) throw std::invalid_argument("block query shape");
    finite_query(query);
    return rank_impl(layer, &query, nullptr, candidates);
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_blocks(
    std::uint32_t layer, DeviceQuery query, const BlockCandidates & candidates) const {
    const auto memory = rank_memory();
    if (!memory) throw std::logic_error("direct device ranking is disabled");
    validate_device_query(query, *memory, query_heads_, geometry_.dimension);
    return rank_impl(layer, nullptr, &query, candidates);
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_impl(
    std::uint32_t layer, const std::vector<float> * query, const DeviceQuery * device, const BlockCandidates & input,
    void * cuda_ready_event, RankCompletion * completion) const {
    input.validate_shape();
    if (layer >= geometry_.layers || input.size() > geometry_.blocks_per_head() || input.remaining > budget_.total ||
        (input.is_prefix() && (input.prefix_tokens > geometry_.block_tokens || input.size() > complete_blocks_[layer]))) {
        throw std::invalid_argument("block ranking shape or budget");
    }
    // Existing external rankers keep their explicit-list contract. CUDA consumes
    // InfLLM's contiguous bank prefix directly, with no host enumeration.
    std::optional<BlockCandidates> expanded;
    if (input.is_prefix() && ranker_ && !ranker_->prefix_candidates()) expanded = input.expanded();
    const auto & candidates = expanded ? *expanded : input;
    std::uint64_t available = 0;
    if (candidates.is_prefix()) available = candidates.size() * candidates.prefix_tokens;
    else for (std::size_t i = 0; i < candidates.blocks.size(); ++i) {
        if ((i && candidates.blocks[i] <= candidates.blocks[i-1]) || candidates.blocks[i] >= geometry_.blocks_per_head() ||
            !ready(layer, candidates.blocks[i]) || candidates.tokens[i] == 0 || candidates.tokens[i] > geometry_.block_tokens) {
            throw std::invalid_argument("invalid block candidate");
        }
        available += candidates.tokens[i];
    }
    const auto needed = std::min<std::uint64_t>(available, candidates.remaining);
    std::vector<std::vector<std::uint64_t>> result(geometry_.heads);
    const auto ready = [completion](RankedBlocks value) {
        if (!completion) return value;
        *completion = [value = std::move(value)]() mutable { return std::move(value); };
        return RankedBlocks{};
    };
    if (!needed) return ready(RankedBlocks(kv_heads_));
    if (candidates.shared_heads && available <= candidates.remaining) {
        auto blocks = candidates.blocks;
        if (candidates.is_prefix()) { blocks.resize(candidates.size()); std::iota(blocks.begin(), blocks.end(), std::uint64_t{0}); }
        return ready(RankedBlocks(kv_heads_, blocks));
    }
    const auto validate = [](const BlockCandidates & candidates, std::uint64_t needed,
                             std::uint32_t heads, std::uint32_t kv_heads, RankedBlocks result) {
        if (result.size() != heads) throw std::runtime_error("ranker returned wrong head count");
        for (const auto & head : result) {
            std::set<std::uint64_t> unique;
            std::uint64_t covered = 0;
            for (const auto block : head) {
                const auto offset = candidates.is_prefix() ? block : static_cast<std::uint64_t>(
                    std::lower_bound(candidates.blocks.begin(), candidates.blocks.end(), block) - candidates.blocks.begin());
                if (offset >= candidates.size() || candidates.block(offset) != block || !unique.insert(block).second || covered >= needed) {
                    throw std::runtime_error("ranker returned invalid, duplicate or excess block");
                }
                covered += candidates.cost(offset);
            }
            if (covered < needed) throw std::runtime_error("ranker returned insufficient token coverage");
        }
        result.resize(kv_heads);
        return result;
    };
    if (device && completion) {
        auto pending = ranker_->submit_rank_device(layer, {*device, cuda_ready_event}, candidates);
        *completion = [pending = std::move(pending), validate, candidates, needed,
                       heads = geometry_.heads, kv_heads = kv_heads_]() mutable {
            return validate(candidates, needed, heads, kv_heads, pending());
        };
        return {};
    }
    if (device) result = ranker_->rank_device_after_event(layer, {*device, cuda_ready_event}, candidates);
    else if (ranker_) result = ranker_->rank(layer, *query, candidates);
    else for (std::uint32_t head = 0; head < (candidates.shared_heads ? 1u : geometry_.heads); ++head) {
        std::vector<std::pair<double, std::uint64_t>> best;
        const auto better = [](const auto & a, const auto & b) { return a.first == b.first ? a.second < b.second : a.first > b.first; };
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const auto block = candidates.block(i);
            auto value = budget_.infllm_attention && candidates.shared_heads ? infllm_shared_score(layer, block, *query) :
                score(layer, head, block, *query);
            if (candidates.shared_heads && !budget_.infllm_attention) {
                for (std::uint32_t h = 1; h < geometry_.heads; ++h) value += score(layer, h, block, *query);
            }
            best.emplace_back(value, block);
            std::push_heap(best.begin(), best.end(), better);
            if (best.size() > candidates.remaining) { std::pop_heap(best.begin(), best.end(), better); best.pop_back(); }
        }
        std::sort(best.begin(), best.end(), better);
        std::uint64_t covered = 0;
        for (const auto & item : best) {
            result[head].push_back(item.second);
            const auto offset = candidates.is_prefix() ? item.second : static_cast<std::uint64_t>(
                std::lower_bound(candidates.blocks.begin(), candidates.blocks.end(), item.second) - candidates.blocks.begin());
            covered += candidates.cost(offset);
            if (covered >= needed) break;
        }
        if (candidates.shared_heads) std::fill(result.begin() + 1, result.end(), result.front());
    }
    return validate(candidates, needed, geometry_.heads, kv_heads_, std::move(result));
}
RankCompletion RepresentativeIndex::submit_rank_blocks(
    std::uint32_t layer, DeviceQuery query, const BlockCandidates & candidates) const {
    const auto memory = rank_memory();
    if (!memory) throw std::logic_error("direct device ranking is disabled");
    validate_device_query(query, *memory, query_heads_, geometry_.dimension);
    RankCompletion completion;
    (void) rank_impl(layer, nullptr, &query, candidates, nullptr, &completion);
    return completion;
}
std::vector<std::vector<std::uint64_t>> RepresentativeIndex::rank_blocks(
    std::uint32_t layer, DeviceQueryWithEvent query, const BlockCandidates & candidates) const {
    const auto memory = rank_memory();
    if (!memory || (query.cuda_ready_event && *memory != DeviceMemory::cuda)) {
        throw std::logic_error("event-dependent query requires its CUDA ranker");
    }
    validate_device_query(query.query, *memory, query_heads_, geometry_.dimension);
    return rank_impl(layer, nullptr, &query.query, candidates, query.cuda_ready_event);
}
double RepresentativeIndex::infllm_shared_score(std::uint32_t layer, std::uint64_t block, const std::vector<float> & query) const {
    // ContextManager.calc_block_topk flattens all query heads into one
    // matrix-vector product. Queries retain llama.cpp's F32 compute dtype;
    // cached representative values are promoted from their F16 storage.
    float value = 0;
    for (std::uint32_t h = 0; h < geometry_.heads; ++h) {
        const auto & tokens = blocks_.at(index(layer, h)).at(block).tokens;
        for (std::uint32_t d = 0; d < geometry_.dimension; ++d) {
            float mean = 0;
            for (const auto & token : tokens) mean += from_half(token.key.at(d));
            const auto key = from_half(to_half(mean / tokens.size()));
            value += query[static_cast<std::size_t>(h) * geometry_.dimension + d] * key;
        }
    }
    if (!std::isfinite(value)) throw std::runtime_error("non-finite InfLLM block score");
    return value;
}
std::vector<std::uint64_t> RepresentativeIndex::representative_positions(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const {
    if (device_only_representatives_) throw std::logic_error("representative position diagnostics are disabled");
    if (block >= geometry_.blocks_per_head()) throw std::out_of_range("representative block");
    std::vector<std::uint64_t> result;
    const auto & blocks = blocks_.at(index(layer, head));
    if (block < blocks.size()) for (const auto & token : blocks[block].tokens) result.push_back(token.position);
    return result;
}
std::uint64_t RepresentativeIndex::resident_bytes() const {
    std::uint64_t total = blocks_.capacity() * sizeof(std::vector<Block>) +
        pending_.capacity() * sizeof(std::deque<Pending>) +
        (lengths_.capacity() + complete_blocks_.capacity() + probability_lengths_.capacity()) * sizeof(std::uint64_t) +
        (query_pending_.capacity() + 7) / 8;
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
                                        std::uint64_t position, const Query & query, SelectionCompletion * completion = nullptr) {
    budget.validate(geometry);
    if (position >= geometry.capacity_tokens) throw std::out_of_range("selection position");
    const auto count = position + 1;
    const auto initial_end = count <= budget.total ? count : budget.initial;
    const auto local_first = count <= budget.total ? count : budget.recent_start(geometry, position);
    const auto origin = representatives.history_start();
    std::vector<std::uint64_t> deterministic(initial_end + count - local_first);
    std::iota(deterministic.begin(), deterministic.begin() + initial_end, std::uint64_t{0});
    std::iota(deterministic.begin() + initial_end, deterministic.end(), local_first);
    BlockCandidates candidates;
    candidates.remaining = static_cast<std::uint32_t>(std::min<std::uint64_t>(budget.total - deterministic.size(),
                                                                           local_first - initial_end));
    candidates.shared_heads = budget.shared_heads;
    if (budget.shared_heads) candidates.remaining -= candidates.remaining % geometry.block_tokens;
    if (candidates.remaining && budget.infllm_attention) {
        candidates.prefix_blocks = std::min(budget.complete_history_blocks(geometry, position), representatives.complete_blocks(layer));
        candidates.prefix_tokens = geometry.block_tokens;
    } else if (candidates.remaining) {
        const auto first = budget.shared_heads ? 0 : initial_end / geometry.block_tokens;
        const auto limit = budget.shared_heads ? budget.complete_history_blocks(geometry, position) :
            (local_first + geometry.block_tokens - 1) / geometry.block_tokens;
        for (auto block = first; block < limit; ++block) {
            if (!representatives.ready(layer, block)) break;
            const auto begin = std::max(initial_end, origin + block * geometry.block_tokens);
            const auto end = std::min(local_first, origin + (block + 1) * geometry.block_tokens);
            candidates.blocks.push_back(block);
            candidates.tokens.push_back(static_cast<std::uint32_t>(end - begin));
        }
    }
    auto assemble = [geometry, budget, initial_end, local_first, origin,
                           deterministic = std::move(deterministic)](const RankedBlocks & ranked) {
        std::vector<HeadSelection> result(geometry.heads);
        for (std::uint32_t head = 0; head < geometry.heads; ++head) {
            auto & chosen = result[head].positions;
            chosen = deterministic;
            for (const auto block : ranked[head]) {
                const auto begin = std::max(initial_end, origin + block * geometry.block_tokens);
                const auto end = std::min(local_first, origin + (block + 1) * geometry.block_tokens);
                for (auto p = begin; p < end && chosen.size() < budget.total; ++p) chosen.push_back(p);
                if (chosen.size() >= budget.total) break;
            }
            std::sort(chosen.begin() + deterministic.size(), chosen.end());
            std::rotate(chosen.begin() + initial_end, chosen.begin() + deterministic.size(), chosen.end());
            for (const auto p : chosen) {
                const auto block = p / geometry.block_tokens;
                if (result[head].blocks.empty() || result[head].blocks.back() != block) result[head].blocks.push_back(block);
            }
        }
        return result;
    };
    if constexpr (std::is_same_v<Query, DeviceQuery>) {
        if (completion) {
            auto pending = representatives.submit_rank_blocks(layer, query, candidates);
            *completion = [pending = std::move(pending), assemble = std::move(assemble)]() mutable { return assemble(pending()); };
            return {};
        }
    }
    return assemble(representatives.rank_blocks(layer, query, candidates));
}
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const std::vector<float> & query) {
    if (query.size() != static_cast<std::size_t>(representatives.query_heads())*geometry.dimension) throw std::invalid_argument("selection query shape");
    finite_query(query);
    return select_blocks_impl(geometry, budget, representatives, layer, position, query);
}
SelectionCompletion submit_select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t position, DeviceQuery query) {
    const auto memory = representatives.rank_memory();
    if (!memory) throw std::logic_error("direct device selection is disabled");
    validate_device_query(query, *memory, representatives.query_heads(), geometry.dimension);
    SelectionCompletion completion;
    (void) select_blocks_impl(geometry, budget, representatives, layer, position, query, &completion);
    return completion;
}
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, DeviceQuery query) {
    const auto memory = representatives.rank_memory();
    if (!memory) throw std::logic_error("direct device selection is disabled");
    validate_device_query(query, *memory, representatives.query_heads(), geometry.dimension);
    return select_blocks_impl(geometry, budget, representatives, layer, position, query);
}
namespace {
template<class Query>
PrefillContext select_prefill_context_impl(const KvGeometry & g, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    const Query & mean_query) {
    budget.validate(g);
    if (!budget.infllm_attention || !count || first >= g.capacity_tokens || count > g.capacity_tokens - first ||
        count > budget.prefill_group_tokens()) throw std::invalid_argument("InfLLM prefill range");
    const auto end = first + count;
    if (first < budget.total && end > budget.total) {
        throw std::invalid_argument("InfLLM prefill chunk crosses the sparse budget boundary");
    }
    if (end <= budget.total) {
        std::vector<std::uint64_t> positions(end);
        std::iota(positions.begin(), positions.end(), std::uint64_t{0});
        return {std::move(positions), static_cast<std::size_t>(end)};
    }

    // The chunk shares its historical selection. Each query applies its own
    // causal sliding window to the union of local positions.
    const auto complete = budget.complete_history_blocks(g, first);
    const auto complete_end = budget.initial + complete * g.block_tokens;
    const auto local_first = first + 1 - budget.local;
    const auto historical_tokens = budget.total - budget.initial - budget.local;
    const auto remainder = std::min<std::uint64_t>(local_first - complete_end, historical_tokens);
    const auto recent = local_first - remainder;
    BlockCandidates candidates;
    candidates.shared_heads = true;
    candidates.remaining = (historical_tokens - remainder) / g.block_tokens * g.block_tokens;
    if (candidates.remaining) {
        candidates.prefix_blocks = complete;
        candidates.prefix_tokens = g.block_tokens;
    }
    auto blocks = representatives.rank_blocks(layer, mean_query, candidates).front();
    std::sort(blocks.begin(), blocks.end());
    std::vector<std::uint64_t> positions;
    positions.reserve(static_cast<std::size_t>(budget.total) + count - 1);
    for (std::uint64_t p = 0; p < budget.initial; ++p) positions.push_back(p);
    for (auto block : blocks) {
        const auto start = budget.initial + block * g.block_tokens;
        for (auto p = start; p < start + g.block_tokens; ++p) positions.push_back(p);
    }
    for (auto p = recent; p < local_first; ++p) positions.push_back(p);
    const auto local_only_from = positions.size();
    for (auto p = local_first; p < end; ++p) positions.push_back(p);
    return {std::move(positions), local_only_from};
}
} // namespace
PrefillContext select_prefill_context(const KvGeometry & g, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    const std::vector<float> & mean_query) {
    return select_prefill_context_impl(g, budget, representatives, layer, first, count, mean_query);
}
PrefillContext select_prefill_context(const KvGeometry & g, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceQuery mean_query) {
    return select_prefill_context_impl(g, budget, representatives, layer, first, count, mean_query);
}
PrefillContext select_prefill_context(const KvGeometry & g, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceQueryWithEvent mean_query) {
    return select_prefill_context_impl(g, budget, representatives, layer, first, count, mean_query);
}
std::vector<float> streamed_attention(KvStore & store, std::uint32_t layer, std::uint64_t position,
                                      std::uint32_t query_heads, const std::vector<float> & query, float scale,
                                      const std::vector<HeadSelection> * selected, const RelativeRope * rope,
                                      std::uint32_t local, std::vector<float> * probabilities) {
    const auto & geometry = store.geometry();
    if (query_heads == 0 || query_heads % geometry.heads != 0 ||
        query.size() != static_cast<std::size_t>(query_heads) * geometry.dimension ||
        position >= store.length(layer) || !std::isfinite(scale) || scale <= 0 ||
        (selected && selected->size() != geometry.heads)) throw std::invalid_argument("attention shape or position mismatch");
    if ((rope || probabilities) && !local) throw std::invalid_argument("relative attention requires a local window");
    finite_query(query);
    const auto group = query_heads / geometry.heads;
    const auto recent = std::min<std::uint64_t>(position + 1, local);
    const auto first_recent = position + 1 - recent;
    if (probabilities) probabilities->assign(static_cast<std::size_t>(query_heads) * recent, 0);
    std::vector<float> output(query.size(), 0);
    std::vector<float> rotated_key(rope ? geometry.dimension : 0);
    KvBlock block;
    for (std::uint32_t head = 0; head < geometry.heads; ++head) {
        std::vector<double> maxima(group, -std::numeric_limits<double>::infinity()), sums(group, 0);
        std::vector<double> weighted(static_cast<std::size_t>(group) * geometry.dimension, 0);
        std::vector<double> recent_logits(probabilities ? group * recent : 0, -std::numeric_limits<double>::infinity());
        const auto consume = [&](std::uint64_t p, std::uint32_t in_block) {
            if (p > position || in_block >= block.valid_tokens) throw std::invalid_argument("future or invalid attention position");
            const auto * key = block.key(in_block);
            if (rope && p >= first_recent) {
                std::transform(key, key + geometry.dimension, rotated_key.begin(), from_half);
                rope->rotate(rotated_key.data(), static_cast<std::uint32_t>(local - 1 - (position - p)));
            }
            for (std::uint32_t q = 0; q < group; ++q) {
                double score = 0;
                for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                    score += static_cast<double>(query[(head * group + q) * geometry.dimension + d]) *
                             (rope && p >= first_recent ? from_half(to_half(rotated_key[d])) : from_half(key[d]));
                }
                score *= scale;
                if (!std::isfinite(score)) throw std::runtime_error("non-finite attention score");
                if (probabilities && p >= first_recent) recent_logits[q * recent + p - first_recent] = score;
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
            if (probabilities) {
                for (std::size_t i = 0; i < recent; ++i) {
                    const auto logit = recent_logits[q * recent + i];
                    if (!std::isfinite(logit)) throw std::logic_error("attention selection omitted a local token");
                    (*probabilities)[(head * group + q) * recent + i] = static_cast<float>(std::exp(logit - maxima[q]) / sums[q]);
                }
            }
            for (std::uint32_t d = 0; d < geometry.dimension; ++d) {
                output[(head * group + q) * geometry.dimension + d] = static_cast<float>(weighted[q * geometry.dimension + d] / sums[q]);
            }
        }
    }
    return output;
}
} // namespace solid
