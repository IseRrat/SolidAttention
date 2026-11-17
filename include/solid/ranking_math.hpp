#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define SOLID_RANK_HD __host__ __device__
#else
#define SOLID_RANK_HD
#endif

namespace solid {
struct RankChoice {
    double score;
    std::uint64_t block;
    std::uint64_t candidate;
};
// A sentinel loses even to a finite score at the negative end of FP64.
SOLID_RANK_HD inline RankChoice better_choice(RankChoice a, RankChoice b) {
    if (a.candidate == UINT64_MAX) return b;
    if (b.candidate == UINT64_MAX) return a;
    if (a.score != b.score) return a.score > b.score ? a : b;
    return a.block < b.block ? a : b;
}
} // namespace solid

#undef SOLID_RANK_HD
