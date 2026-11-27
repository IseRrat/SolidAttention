// 代表向量参考路径的均值、分数累加与稳定插入公式，供 CPU/CUDA 共用。
#pragma once

#include "solid/kv_store.hpp"
#include <cstdint>

#ifdef __CUDACC__
#define SOLID_REP_HD __host__ __device__
#else
#define SOLID_REP_HD
#endif

namespace solid {
SOLID_REP_HD inline double representative_mean(const float * query, std::uint32_t head, std::uint32_t dimension,
                                               std::uint32_t width, std::uint32_t group) {
    double value = 0;
    for (std::uint32_t q = 0; q < group; ++q) value += static_cast<double>(query[(static_cast<std::uint64_t>(head)*group+q)*width+dimension])/group;
    return value;
}
template<class DecodeHalf>
SOLID_REP_HD inline double update_representative_score(double value, const double * mean, const Half * key,
                                                        std::uint32_t width, DecodeHalf decode) {
    for (std::uint32_t d = 0; d < width; ++d) value += mean[d] * decode(key[d]);
    return value;
}
// Arrays refer to one KV head. Stable insertion preserves the CPU score/position
// ordering; each slot owns a contiguous FP16 K row.
template<class Score>
SOLID_REP_HD inline void insert_representative(std::uint64_t position, Score score, const Half * key,
                                               std::uint32_t count, std::uint32_t width,
                                               std::uint64_t * positions, Score * scores, Half * keys) {
    std::uint32_t target = count;
    for (std::uint32_t r = 0; r < count; ++r) {
        if (positions[r] == UINT64_MAX || score > scores[r] || (score == scores[r] && position < positions[r])) { target = r; break; }
    }
    if (target == count) return;
    for (auto r = count - 1; r > target; --r) {
        positions[r] = positions[r-1]; scores[r] = scores[r-1];
        for (std::uint32_t d = 0; d < width; ++d) keys[static_cast<std::uint64_t>(r)*width+d] = keys[static_cast<std::uint64_t>(r-1)*width+d];
    }
    positions[target] = position; scores[target] = score;
    for (std::uint32_t d = 0; d < width; ++d) keys[static_cast<std::uint64_t>(target)*width+d] = key[d];
}
} // namespace solid

#undef SOLID_REP_HD
