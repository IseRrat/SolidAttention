// 代表向量构造与块评分接口；CPU/CUDA/SYCL 通过同一输入和预算约定完成选块。
#pragma once

#include "solid/kv_store.hpp"
#include "solid/device_input.hpp"
#include <memory>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace solid {
using RankedBlocks = std::vector<std::vector<std::uint64_t>>;
using RankCompletion = std::function<RankedBlocks()>;
// Candidates have the same eligibility and token cost for every KV head.
// 候选逻辑块与每块 token 成本；连续候选可仅用块数量表达，省去逐块 CPU 数组。
struct BlockCandidates {
    std::vector<std::uint64_t> blocks;
    std::vector<std::uint32_t> tokens;
    std::uint32_t remaining = 0;
    bool shared_heads = false;
    // InfLLM's VectorTensor[:length]: all complete blocks, one uniform cost.
    // The prefix form carries no per-block CPU arrays.
    std::uint64_t prefix_blocks = 0;
    std::uint32_t prefix_tokens = 0;
    bool is_prefix() const { return prefix_tokens != 0; }
    std::uint64_t size() const { return is_prefix() ? prefix_blocks : blocks.size(); }
    std::uint64_t block(std::size_t i) const { return is_prefix() ? i : blocks[i]; }
    std::uint32_t cost(std::size_t i) const { return is_prefix() ? prefix_tokens : tokens[i]; }
    void validate_shape() const {
        if (is_prefix() ? (!shared_heads || !blocks.empty() || !tokens.empty()) :
            (prefix_blocks || blocks.size() != tokens.size())) throw std::invalid_argument("invalid block candidate representation");
    }
    BlockCandidates expanded() const {
        validate_shape();
        if (!is_prefix()) return *this;
        BlockCandidates result;
        result.shared_heads = shared_heads;
        result.remaining = remaining;
        result.blocks.resize(prefix_blocks);
        result.tokens.assign(prefix_blocks, prefix_tokens);
        for (std::size_t i = 0; i < result.blocks.size(); ++i) result.blocks[i] = i;
        return result;
    }
};
struct RankerStats {
    std::uint64_t device_bytes = 0;
    std::uint64_t host_bytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t scored_blocks = 0;
    std::uint64_t published_blocks = 0;
    std::uint64_t builder_device_bytes = 0;
    std::uint64_t builder_queries = 0;
    std::uint64_t builder_appends = 0;
    std::uint64_t built_blocks = 0;
    std::uint64_t device_rank_queries = 0, device_builder_queries = 0, device_builder_appends = 0;
    std::uint64_t query_upload_bytes = 0, builder_key_upload_bytes = 0;
    std::uint64_t device_builder_batches = 0;
    std::uint64_t builder_rank_shared_waits = 0;
    std::uint64_t builder_compute_appends = 0;
    std::uint64_t query_ready_event_waits = 0;
};
struct RepresentativeSeed {
    std::uint64_t length = 0;
    // Oldest to newest: [pending, KV head, D] and [pending, scoring head].
    std::vector<Half> pending_keys;
    std::vector<double> pending_scores;
    // Current partially finalized block: [head, representative, D] and
    // [head, representative]. UINT64_MAX marks trailing unused entries.
    std::vector<Half> partial_keys;
    std::vector<std::uint64_t> partial_positions;
    std::vector<double> partial_scores;
};
struct BuiltRepresentatives {
    std::uint64_t block = 0;
    std::vector<Half> keys;
    std::vector<std::uint64_t> positions;
    std::vector<double> scores;
};
struct ProbabilityFrame {
    std::uint32_t layer;
    std::uint64_t first;
    std::uint32_t count;
    std::uint64_t score_first;
    DeviceKvBatch kv;
    DeviceQuery scores;
    bool key_major = true;
};
void validate_representative_seed(const KvGeometry & geometry, std::uint32_t representatives, std::uint32_t local,
                                  const RepresentativeSeed & seed, std::uint32_t input_heads = 0);
class BlockRanker {
public:
    virtual ~BlockRanker() = default;
    // Immutable finalized representatives: [block, kv_head, representative, D].
    // Return only after the caller may reuse keys.
    // 发布已定稿的块代表到评分器，后续检索直接读取这份代表表。
    virtual void publish(std::uint32_t layer, std::uint64_t first_block, std::uint64_t count,
                         const std::vector<Half> & keys) = 0;
    // Per head, descending relevance; equal scores use ascending logical block ID.
    // Return enough blocks to cover remaining tokens, or all candidates if fewer.
    // 按相关性选足剩余历史预算；shared_heads 模式让各 head 使用同一组逻辑块。
    virtual std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                        const BlockCandidates & candidates) = 0;
    virtual RankerStats stats() const = 0;
    virtual const char * name() const = 0;
    virtual bool prefix_candidates() const { return false; }
    virtual bool configure_probability_builder(std::uint32_t, std::uint32_t, std::uint32_t) { return false; }
    virtual bool device_probabilities() const { return false; }
    virtual bool host_representative_metadata() const { return true; }
    virtual DeviceKeyWindow probability_key_window(std::uint32_t) const { return {}; }
    virtual std::vector<BuiltRepresentatives> probability_update(const ProbabilityFrame &) {
        throw std::logic_error("device probability builder is disabled");
    }
    // One outstanding receipt. A separate queue snapshots borrowed inputs;
    // on_compute preserves upstream's update-before-FFN GPU queue order.
    // 提交设备近期概率更新；on_compute 指定沿计算队列执行以保持更新顺序。
    virtual void submit_probability_update(const ProbabilityFrame &, void *, bool = false) {
        throw std::logic_error("asynchronous probability builder is disabled");
    }
    virtual std::vector<BuiltRepresentatives> finish_probability_update() {
        throw std::logic_error("asynchronous probability builder is disabled");
    }
    // An owning backend keeps every published immutable K block. The index may
    // release its host payload; diagnostic scoring reads back one block only.
    virtual bool owns_representative_keys() const { return false; }
    virtual std::vector<Half> read_representatives(std::uint32_t, std::uint64_t) {
        throw std::logic_error("representative readback is disabled");
    }
    virtual std::uint32_t builder_window() const { return 0; }
    virtual const char * builder_name() const { return "cpu"; }
    virtual std::optional<DeviceMemory> direct_memory() const { return {}; }
    virtual std::vector<std::vector<std::uint64_t>> rank_device(std::uint32_t, DeviceQuery, const BlockCandidates &) {
        throw std::logic_error("direct device ranking is disabled");
    }
    virtual std::vector<std::vector<std::uint64_t>> rank_device_after_event(std::uint32_t layer, DeviceQueryWithEvent query,
                                                                          const BlockCandidates & candidates) {
        if (query.cuda_ready_event) throw std::logic_error("event-dependent device ranking is disabled");
        return rank_device(layer, query.query, candidates);
    }
    // Submit before returning to model computation; collect on the I/O worker.
    // One outstanding receipt owns the ranker's workspace. Keep the query and
    // ranker alive until collection; candidate metadata is copied on submission.
    // 提交设备 Q 的异步评分/Top-K；返回的收取函数负责取得最终块编号。
    virtual RankCompletion submit_rank_device(std::uint32_t layer, DeviceQueryWithEvent query,
                                              const BlockCandidates & candidates) {
        auto result = rank_device_after_event(layer, query, candidates);
        return [result = std::move(result)]() mutable { return std::move(result); };
    }
    virtual std::optional<BuiltRepresentatives> build_query_device(std::uint32_t, std::uint64_t, DeviceQuery) {
        throw std::logic_error("direct device representative query is disabled");
    }
    virtual std::optional<BuiltRepresentatives> build_query_for_rank_device(std::uint32_t layer,std::uint64_t position,DeviceQuery query) {
        return build_query_device(layer,position,query);
    }
    virtual void finish_query_for_rank_device() {}
    virtual void build_append_device(std::uint32_t, std::uint64_t, DeviceKeys) {
        throw std::logic_error("direct device representative append is disabled");
    }
    virtual void build_append_device_on_compute(std::uint32_t,std::uint64_t,DeviceKeys,void *) {
        throw std::logic_error("representative append on compute queue is disabled");
    }
    virtual std::vector<BuiltRepresentatives> build_batch_device(std::uint32_t,std::uint64_t,std::uint32_t,DeviceQuery,DeviceKvBatch) {
        throw std::logic_error("device representative batch is disabled");
    }
    virtual void seed_builder(std::uint32_t, const RepresentativeSeed &) { throw std::logic_error("representative builder is disabled"); }
    // A complete block is committed to the ranker's bank before it is returned.
    virtual std::optional<BuiltRepresentatives> build_query(std::uint32_t, std::uint64_t, const std::vector<float> &) {
        throw std::logic_error("representative builder is disabled");
    }
    virtual void build_append(std::uint32_t, std::uint64_t, const std::vector<Half> &) {
        throw std::logic_error("representative builder is disabled");
    }
};

// CUDA factory is linked by SOLID_CUDA builds. Construction selects one device;
// operations restore the calling thread's previous device on exit.
std::unique_ptr<BlockRanker> make_cuda_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, int device,
                                                   std::uint32_t local = 0, std::uint32_t input_heads = 0,
                                                   bool representative_diagnostics = false);
std::unique_ptr<BlockRanker> make_sycl_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, void * backend_queue,
                                                   std::uint32_t local, std::uint32_t input_heads = 0);
} // namespace solid
