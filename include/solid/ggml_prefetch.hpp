#pragma once
#include "solid/device_cache.hpp"
#include "solid/session.hpp"

struct ggml_backend;
struct ggml_tensor;
namespace solid {
class GgmlPrefetch {
public:
    GgmlPrefetch(AttentionSession & session, ggml_backend * backend, const std::filesystem::path & trace = {});
    ~GgmlPrefetch();
    ggml_tensor * tensor() const;
    std::uint32_t tokens_per_head() const;
    std::uint64_t attention_epoch() const;
    std::vector<Half> decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                              const std::vector<float> & keys, const std::vector<float> & values, bool packed_inputs);
    void completed(std::uint64_t epoch);
    // The caller keeps the context/backend alive until this has returned.
    void quiesce();
    DeviceCacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace solid
