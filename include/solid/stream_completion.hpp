#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace solid {
// The launch hook only records events and enqueues notify. notify must never
// invoke a device API or wait for work running on another device stream.
struct StreamCompletionToken {
    void (*notify)(void *) noexcept=nullptr;
    void * userdata=nullptr;
    void * begin_event=nullptr;
    void * end_event=nullptr;
    std::uint64_t identity=0;
    std::uint64_t (*armed_epoch)(void *) noexcept=nullptr;
    void (*notify_epoch)(void *,std::uint64_t) noexcept=nullptr;
};
struct StreamCompletionBinding {
    std::uint64_t magic=0;
    StreamCompletionToken * token=nullptr;
    std::uint64_t identity=0;
};
constexpr std::size_t stream_completion_offset=32;
constexpr std::uint64_t stream_completion_magic=0x534f4c4944434d50ull;
inline void bind_stream_completion(void * params,std::size_t bytes,StreamCompletionToken * token) {
    if (bytes<stream_completion_offset+sizeof(StreamCompletionBinding)) return;
    const StreamCompletionBinding binding{token?stream_completion_magic:0,token,token?token->identity:0};
    std::memcpy(static_cast<unsigned char *>(params)+stream_completion_offset,&binding,sizeof(binding));
}
inline StreamCompletionToken * stream_completion(const void * params,std::size_t bytes) {
    if (bytes<stream_completion_offset+sizeof(StreamCompletionBinding)) return nullptr;
    StreamCompletionBinding binding;
    std::memcpy(&binding,static_cast<const unsigned char *>(params)+stream_completion_offset,sizeof(binding));
    if (binding.magic!=stream_completion_magic || !binding.token) return nullptr;
    return binding.identity==binding.token->identity?binding.token:nullptr;
}
} // namespace solid
