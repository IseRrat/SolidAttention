#pragma once
#include <cstdint>
#include <tuple>

namespace solid {
struct BlockIdentity {
    std::uint32_t layer=0,head=0;
    std::uint64_t block=0,version=0;
    auto key() const { return std::tie(layer,head,block,version); }
    bool operator==(const BlockIdentity & other) const { return key()==other.key(); }
    bool operator<(const BlockIdentity & other) const { return key()<other.key(); }
};
struct SlotLease {
    std::uint32_t slot=0;
    std::uint64_t generation=0;
    BlockIdentity identity;
    bool operator==(const SlotLease & b) const { return slot==b.slot && generation==b.generation && identity==b.identity; }
};
} // namespace solid
