#pragma once

#include "solid/device_cache.hpp"
#include <memory>

struct ggml_backend;
struct ggml_context;
struct ggml_tensor;

namespace solid {
class AttentionSession;

class DenseOffload {
public:
    DenseOffload(AttentionSession & session, ggml_backend * backend);
    ~DenseOffload();
    void begin(std::uint64_t position);
    void upload_history(std::uint32_t layer, std::uint64_t position);
    void archive(std::uint32_t layer, std::uint64_t position, const float * keys, const float * values);
    void attention_complete();
    void join();
    void finish_decode();
    ggml_tensor * plane(ggml_context * context, bool values) const;
    DeviceCacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace solid
