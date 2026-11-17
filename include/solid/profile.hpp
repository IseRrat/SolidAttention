#pragma once

#include "ggml-backend.h"
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <string>

using solid_transfer_callback = void (*)(const ggml_tensor *, ggml_backend_t, ggml_backend_t, std::uint64_t, void *);
extern "C" void solid_set_transfer_callback(solid_transfer_callback callback, void * userdata);

namespace solid {
struct ProfileTotal {
    std::uint64_t calls = 0;
    std::uint64_t bytes = 0;
    std::uint64_t ns = 0;
};
class ExecutionProfile {
public:
    explicit ExecutionProfile(bool enabled);
    ~ExecutionProfile();
    ExecutionProfile(const ExecutionProfile &) = delete;
    ExecutionProfile & operator=(const ExecutionProfile &) = delete;
    void set_decode() { phase_ = "decode"; }
    void check() const;
    static bool evaluate(ggml_tensor * tensor, bool ask, void * userdata);
    const std::map<std::string, ProfileTotal> & nodes() const { return nodes_; }
    const std::map<std::string, ProfileTotal> & transfers() const { return transfers_; }
private:
    static void transferred(const ggml_tensor * tensor, ggml_backend_t source, ggml_backend_t target,
                            std::uint64_t ns, void * userdata);
    bool enabled_;
    std::string phase_ = "prefill";
    std::chrono::steady_clock::time_point start_;
    std::exception_ptr error_;
    std::map<std::string, ProfileTotal> nodes_;
    std::map<std::string, ProfileTotal> transfers_;
};
} // namespace solid
