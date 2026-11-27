// Nsight 阶段标记及线程/stream 命名接口；标记计算、搬运、读写和等待。
#pragma once
#include <cstdint>
#include <memory>

namespace solid::trace {
bool enabled() noexcept;
void name_thread(const char * name) noexcept;
void name_cuda_stream(void * stream, const char * name) noexcept;

// Lexical ranges stay on their creating thread and form the NVTX range stack.
// 同一 CPU 线程内的阶段范围，析构时结束，用于 Prefill/Decode 等嵌套标记。
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
// 允许跨 CPU 线程结束的阶段范围，显示主机观察的生命周期。
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

// One visible I/O phase shared by all requests in a layer batch. The final
// request receipt releases the range, including cancellation receipts.
// 让同一批读盘或上传请求共用一个阶段条，最后一个持有者释放后结束。
class Batch {
public:
    Batch(const char * stage, std::uint32_t layer, std::uint64_t position) noexcept
        : stage_(stage), layer_(layer), position_(position) {}
    void submitted() noexcept {
        if (!started_) { range_ = Range(stage_, layer_, position_); started_ = true; }
    }
private:
    const char * stage_;
    std::uint32_t layer_;
    std::uint64_t position_;
    bool started_ = false;
    Range range_;
};
class BatchContext {
public:
    BatchContext(std::shared_ptr<Batch> & current, const char * stage, std::uint32_t layer, std::uint64_t position)
        : current_(current), previous_(current) {
        if (enabled()) current_ = std::make_shared<Batch>(stage, layer, position);
    }
    ~BatchContext() { current_ = std::move(previous_); }
    BatchContext(const BatchContext &) = delete;
    BatchContext & operator=(const BatchContext &) = delete;
private:
    std::shared_ptr<Batch> & current_;
    std::shared_ptr<Batch> previous_;
};

// Aggregate model computation into projections and a complete attention phase.
class ComputeScope {
public:
    ComputeScope(const char * op, const char * input, const char * output, void * stream) noexcept;
    ~ComputeScope();
    ComputeScope(const ComputeScope &) = delete;
    ComputeScope & operator=(const ComputeScope &) = delete;
private:
    Scope projection_;
    bool attention_end_ = false;
    bool kv_end_ = false;
};
} // namespace solid::trace

// C entry used by the generated GGML CPU worker implementation.
extern "C" void solid_trace_name_cpu_worker(int index) noexcept;
