// CPU/CUDA 共用的排名比较规则：分数优先，同分时较小逻辑块编号优先。
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
// 返回较优候选；分数相同时保留较小块编号，以固定选择顺序。
SOLID_RANK_HD inline RankChoice better_choice(RankChoice a, RankChoice b) {
    if (a.candidate == UINT64_MAX) return b;
    if (b.candidate == UINT64_MAX) return a;
    if (a.score != b.score) return a.score > b.score ? a : b;
    return a.block < b.block ? a : b;
}
} // namespace solid

#undef SOLID_RANK_HD
