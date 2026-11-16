#pragma once

#include "solid/kv_store.hpp"
#include <deque>
#include <vector>

namespace solid {

struct SparseBudget {
    std::uint32_t total = 1024;
    std::uint32_t initial = 32;
    std::uint32_t local = 480;
    std::uint32_t representatives = 1;
    void validate(const KvGeometry & geometry) const;
};

class RepresentativeIndex {
public:
    RepresentativeIndex(KvGeometry geometry, std::uint32_t query_heads, SparseBudget budget);
    void observe(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                 const std::vector<Half> & keys);
    bool ready(std::uint32_t layer, std::uint64_t block) const;
    double score(std::uint32_t layer, std::uint32_t head, std::uint64_t block, const std::vector<float> & query) const;
    std::vector<std::uint64_t> representative_positions(std::uint32_t layer, std::uint32_t head, std::uint64_t block) const;
    std::uint64_t resident_bytes() const;
    std::uint32_t query_heads() const { return query_heads_; }
private:
    struct Representative { std::uint64_t position; double score; std::vector<Half> key; };
    struct Block { std::uint32_t finalized = 0; std::vector<Representative> tokens; };
    struct Pending { std::uint64_t position; std::vector<Half> keys; std::vector<double> scores; };
    KvGeometry geometry_;
    std::uint32_t query_heads_;
    SparseBudget budget_;
    std::vector<std::deque<Pending>> pending_;
    std::vector<std::vector<Block>> blocks_;
    std::vector<std::uint64_t> lengths_;
    std::size_t index(std::uint32_t layer, std::uint32_t head) const;
};

struct HeadSelection {
    std::vector<std::uint64_t> positions;
    std::vector<std::uint64_t> blocks;
};

std::vector<HeadSelection> select_blocks(const KvGeometry & geometry, const SparseBudget & budget,
                                        const RepresentativeIndex & representatives, std::uint32_t layer,
                                        std::uint64_t position, const std::vector<float> & query);

// Q contains query-head-major vectors. K/V retain absolute positions and FP16 precision.
std::vector<float> streamed_attention(KvStore & store, std::uint32_t layer, std::uint64_t position,
                                      std::uint32_t query_heads, const std::vector<float> & query, float scale,
                                      const std::vector<HeadSelection> * selected = nullptr);
} // namespace solid
