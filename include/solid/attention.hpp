#pragma once

#include "solid/kv_store.hpp"
#include "solid/block_ranker.hpp"
#include <deque>
#include <vector>

namespace solid {

struct SparseBudget {
    std::uint32_t total = 1024;
    std::uint32_t initial = 32;
    std::uint32_t local = 480;
    std::uint32_t representatives = 1;
    bool shared_heads = false;
    bool infllm_attention = false;
    void validate(const KvGeometry & geometry) const;
    std::uint32_t prefill_group_tokens() const;
    std::uint32_t cache_slots(const KvGeometry & geometry) const;
    std::uint64_t complete_history_blocks(const KvGeometry & geometry, std::uint64_t position) const;
    std::uint64_t recent_start(const KvGeometry & geometry, std::uint64_t position) const;
};
struct RelativeRope {
    std::uint32_t dimensions;
    float base;
    bool split_halves;
    void rotate(float * row, std::uint32_t position) const;
};
// InfLLM retrieval averages global Q in its compute dtype (llama.cpp F32).
// The caller owns width output elements; KV storage precision is independent.
void mean_infllm_query(const float * queries, float * mean, std::size_t width, std::uint32_t count);

class RepresentativeIndex {
public:
    RepresentativeIndex(KvGeometry geometry, std::uint32_t query_heads, SparseBudget budget);
    void observe(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                 const std::vector<Half> & keys);
    void observe_query(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query);
    void begin_probability_query(std::uint32_t layer, std::uint64_t position);
    void observe_prefill_scores(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
                                const std::vector<Half> & kv, const std::vector<float> & key_scores);
    bool device_probabilities() const { return device_probabilities_; }
    void observe_device_probabilities(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
                                     DeviceKvBatch kv, DeviceQuery scores, bool key_major, void * compute_stream = nullptr);
    void finish_device_probabilities();
    void observe_prefill_device_scores(std::uint32_t layer, std::uint64_t first, std::uint32_t count,
        DeviceKvBatch kv, DeviceQuery scores);
    void append_keys(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & keys);
    void observe_probabilities(std::uint32_t layer, std::uint64_t position, const std::vector<float> & probabilities);
    void observe_query_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query,bool will_rank=false);
    void finish_query_for_rank_device();
    void append_keys_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys,void * compute_stream=nullptr);
    void observe_batch_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,DeviceQuery query,DeviceKvBatch kv);
    std::optional<DeviceMemory> direct_memory() const { return ranker_ && ranker_->builder_window() ? ranker_->direct_memory() : std::optional<DeviceMemory>{}; }
    std::optional<DeviceMemory> rank_memory() const { return ranker_ ? ranker_->direct_memory() : std::optional<DeviceMemory>{}; }
    bool ready(std::uint32_t layer, std::uint64_t block) const;
    std::uint64_t complete_blocks(std::uint32_t layer) const { return complete_blocks_.at(layer); }
    double score(std::uint32_t layer, std::uint32_t head, std::uint64_t block, const std::vector<float> & query) const;
    std::vector<std::uint64_t> representative_positions(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const;
    std::uint64_t resident_bytes() const;
    std::uint32_t query_heads() const { return query_heads_; }
    const KvGeometry & representative_geometry() const { return geometry_; }
    std::uint64_t history_start() const { return budget_.shared_heads ? budget_.initial : 0; }
    void set_ranker(std::unique_ptr<BlockRanker> ranker);
    RankerStats ranker_stats() const { return ranker_ ? ranker_->stats() : RankerStats{}; }
    const char * ranker_name() const { return ranker_ ? ranker_->name() : "cpu"; }
    const char * builder_name() const { return ranker_ ? ranker_->builder_name() : "cpu"; }
    std::vector<std::vector<std::uint64_t>> rank_blocks(std::uint32_t layer, const std::vector<float> & query,
                                                       const BlockCandidates & candidates) const;
    std::vector<std::vector<std::uint64_t>> rank_blocks(std::uint32_t layer, DeviceQuery query,
                                                       const BlockCandidates & candidates) const;
    std::vector<std::vector<std::uint64_t>> rank_blocks(std::uint32_t layer, DeviceQueryWithEvent query,
                                                       const BlockCandidates & candidates) const;
    RankCompletion submit_rank_blocks(std::uint32_t layer, DeviceQuery query, const BlockCandidates & candidates) const;
private:
    struct Representative { std::uint64_t position; double score; std::vector<Half> key; };
    struct Block { std::uint32_t finalized = 0; std::vector<Representative> tokens; };
    struct Pending { std::uint64_t position; std::vector<Half> keys; std::vector<double> scores; };
    KvGeometry geometry_;
    std::uint32_t query_heads_;
    std::uint32_t kv_heads_;
    SparseBudget budget_;
    std::vector<std::deque<Pending>> pending_;
    std::vector<std::vector<Block>> blocks_;
    std::vector<std::uint64_t> lengths_;
    std::vector<std::uint64_t> complete_blocks_;
    std::vector<bool> query_pending_;
    std::vector<std::uint64_t> probability_lengths_;
    std::unique_ptr<BlockRanker> ranker_;
    bool device_probabilities_ = false;
    struct ProbabilityReceipt { std::uint32_t layer; std::uint64_t end; };
    std::optional<ProbabilityReceipt> pending_probability_;
    void accept_probability_blocks(std::uint32_t layer, const std::vector<BuiltRepresentatives> & built);
    void publish(BlockRanker & ranker, std::uint32_t layer, std::uint64_t first, std::uint64_t count) const;
    RepresentativeSeed builder_seed(std::uint32_t layer) const;
    void accept_built(std::uint32_t layer, std::uint64_t position, const std::optional<BuiltRepresentatives> & built);
    void release_published_keys(std::uint32_t layer, std::uint64_t block);
    void finalize_before(std::uint32_t layer, std::uint64_t first, double divisor);
    std::size_t index(std::uint32_t layer, std::uint32_t head) const;
    double infllm_shared_score(std::uint32_t layer, std::uint64_t block, const std::vector<float> & query) const;
    std::vector<std::vector<std::uint64_t>> rank_impl(std::uint32_t layer, const std::vector<float> * query,
                                                     const DeviceQuery * device, const BlockCandidates & candidates,
                                                     void * cuda_ready_event = nullptr, RankCompletion * completion = nullptr) const;
};

struct HeadSelection {
    std::vector<std::uint64_t> positions;
    std::vector<std::uint64_t> blocks;
};
using SelectionCompletion = std::function<std::vector<HeadSelection>()>;
SelectionCompletion submit_select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t position, DeviceQuery query);

std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const std::vector<float> & query);
std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, DeviceQuery query);
inline constexpr std::uint32_t infllm_prefill_chunk_tokens = 512;
struct PrefillContext {
    std::vector<std::uint64_t> positions;
    std::size_t local_only_from;
};
PrefillContext select_prefill_context(const KvGeometry & geometry, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    const std::vector<float> & mean_query);
PrefillContext select_prefill_context(const KvGeometry & geometry, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceQuery mean_query);
PrefillContext select_prefill_context(const KvGeometry & geometry, const SparseBudget & budget,
    const RepresentativeIndex & representatives, std::uint32_t layer, std::uint64_t first, std::uint32_t count,
    DeviceQueryWithEvent mean_query);

// Q contains query-head-major vectors. K/V retain absolute positions and FP16 precision.
std::vector<float> streamed_attention(KvStore & store, std::uint32_t layer, std::uint64_t position,
                                      std::uint32_t query_heads, const std::vector<float> & query, float scale,
                                      const std::vector<HeadSelection> * selected = nullptr, const RelativeRope * rope = nullptr,
                                      std::uint32_t local = 0, std::vector<float> * probabilities = nullptr);
} // namespace solid
