#pragma once
#include <cstdint>

namespace solid::trace {
bool enabled() noexcept;
void name_thread(const char * name) noexcept;
void name_cuda_stream(void * stream, const char * name) noexcept;
const char * operation_stage(const char * op, const char * input, const char * output) noexcept;

// Lexical ranges stay on their creating thread and form the NVTX range stack.
class Scope {
public:
    Scope() noexcept = default;
    explicit Scope(const char * stage, std::uint32_t layer = UINT32_MAX,
                   std::uint64_t position = UINT64_MAX, const char * detail = nullptr) noexcept {
        start(stage, layer, position, detail);
    }
    ~Scope() { finish(); }
    Scope(const Scope &) = delete;
    Scope & operator=(const Scope &) = delete;
    void start(const char * stage, std::uint32_t layer = UINT32_MAX,
               std::uint64_t position = UINT64_MAX, const char * detail = nullptr) noexcept;
    void finish() noexcept;
private:
    bool active_ = false;
};

// Start/end ranges can cross CPU threads. They describe host-observed stage
// intervals; actual GPU execution remains on Nsight's kernel/memory rows.
class Range {
public:
    Range() noexcept = default;
    explicit Range(const char * stage, std::uint32_t layer = UINT32_MAX,
                   std::uint64_t position = UINT64_MAX, const char * detail = nullptr) noexcept;
    ~Range() { finish(); }
    Range(const Range &) = delete;
    Range & operator=(const Range &) = delete;
    Range(Range && other) noexcept : id_(other.id_) { other.id_ = 0; }
    Range & operator=(Range && other) noexcept {
        if (this != &other) { finish(); id_ = other.id_; other.id_ = 0; }
        return *this;
    }
    void finish() noexcept;
private:
    std::uint64_t id_ = 0;
};
} // namespace solid::trace

// C entry used by the generated GGML CPU worker implementation.
extern "C" void solid_trace_name_cpu_worker(int index) noexcept;
