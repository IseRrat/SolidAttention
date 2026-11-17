#pragma once

#include "solid/kv_store.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace solid {
// Candidates have the same eligibility and token cost for every KV head.
struct BlockCandidates {
    std::vector<std::uint64_t> blocks;
    std::vector<std::uint32_t> tokens;
    std::uint32_t remaining = 0;
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
};
struct RepresentativeSeed {
    std::uint64_t length = 0;
    // Oldest to newest: [min(length, local), head, D] and [pending, head].
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
void validate_representative_seed(const KvGeometry & geometry, std::uint32_t representatives, std::uint32_t local,
                                  const RepresentativeSeed & seed);
class BlockRanker {
public:
    virtual ~BlockRanker() = default;
    // Immutable finalized representatives: [block, kv_head, representative, D].
    // Return only after the caller may reuse keys.
    virtual void publish(std::uint32_t layer, std::uint64_t first_block, std::uint64_t count,
                         const std::vector<Half> & keys) = 0;
    // Per head, descending relevance; equal scores use ascending logical block ID.
    // Return enough blocks to cover remaining tokens, or all candidates if fewer.
    virtual std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                        const BlockCandidates & candidates) = 0;
    virtual RankerStats stats() const = 0;
    virtual const char * name() const = 0;
    virtual std::uint32_t builder_window() const { return 0; }
    virtual const char * builder_name() const { return "cpu"; }
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
                                                   std::uint32_t local = 0);
} // namespace solid
