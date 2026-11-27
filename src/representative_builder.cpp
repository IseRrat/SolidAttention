// CPU 代表向量构造器：维护待定 Key 与累计分数，在完整块退出窗口后生成代表。
#include "solid/block_ranker.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace solid {
void validate_representative_seed(const KvGeometry & geometry, std::uint32_t representatives, std::uint32_t local,
                                  const RepresentativeSeed & seed, std::uint32_t input_heads) {
    geometry.validate();
    if (!input_heads) input_heads = geometry.heads;
    if (!local || local > geometry.capacity_tokens || !representatives || representatives > geometry.block_tokens ||
        geometry.heads % input_heads || seed.length > geometry.capacity_tokens) {
        throw std::invalid_argument("invalid representative builder geometry or length");
    }
    const auto pending = std::min<std::uint64_t>(seed.length, local);
    const auto matured = seed.length - pending;
    const auto partial = matured % geometry.block_tokens;
    const auto slots = static_cast<std::uint64_t>(geometry.heads) * representatives;
    if (seed.pending_keys.size() != pending * input_heads * geometry.dimension ||
        seed.pending_scores.size() != pending * geometry.heads || seed.partial_positions.size() != slots ||
        seed.partial_scores.size() != slots || seed.partial_keys.size() != slots * geometry.dimension) {
        throw std::invalid_argument("representative builder seed shape");
    }
    for (const auto k : seed.pending_keys) if (!std::isfinite(from_half(k))) throw std::invalid_argument("non-finite pending representative key");
    for (const auto k : seed.partial_keys) if (!std::isfinite(from_half(k))) throw std::invalid_argument("non-finite partial representative key");
    for (const auto value : seed.pending_scores) if (!std::isfinite(value)) throw std::invalid_argument("non-finite pending representative score");
    const auto valid = std::min<std::uint64_t>(partial, representatives);
    for (std::uint32_t head = 0; head < geometry.heads; ++head) {
        std::set<std::uint64_t> positions;
        for (std::uint32_t r = 0; r < representatives; ++r) {
            const auto i = static_cast<std::size_t>(head)*representatives+r;
            const auto p = seed.partial_positions[i];
            const auto score = seed.partial_scores[i];
            if (!std::isfinite(score)) throw std::invalid_argument("non-finite partial representative score");
            if (r >= valid) {
                if (p != UINT64_MAX) throw std::invalid_argument("representative seed padding");
            } else if (p < matured-partial || p >= matured || !positions.insert(p).second || (r &&
                (score > seed.partial_scores[i-1] || (score == seed.partial_scores[i-1] && p < seed.partial_positions[i-1])))) {
                throw std::invalid_argument("representative seed position or ordering");
            }
        }
    }
}
} // namespace solid
