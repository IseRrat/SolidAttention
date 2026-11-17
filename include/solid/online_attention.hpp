#pragma once
#include <cmath>

#ifdef __CUDACC__
#define SOLID_ATTN_HD __host__ __device__
#else
#define SOLID_ATTN_HD
#endif
namespace solid {
struct OnlineAttentionUpdate { float maximum, sum, previous_scale, weight; };
SOLID_ATTN_HD inline OnlineAttentionUpdate online_attention_update(float maximum, float sum, float score) {
    const auto next = maximum > score ? maximum : score;
    const auto previous = expf(maximum-next);
    const auto weight = expf(score-next);
    return {next, sum*previous+weight, previous, weight};
}
} // namespace solid
#undef SOLID_ATTN_HD
