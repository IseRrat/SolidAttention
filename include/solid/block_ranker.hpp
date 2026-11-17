#pragma once

#include "solid/kv_store.hpp"
#include <memory>
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
};
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
};

// CUDA factory is linked by SOLID_CUDA builds. Construction selects one device;
// operations restore the calling thread's previous device on exit.
std::unique_ptr<BlockRanker> make_cuda_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, int device);
} // namespace solid
