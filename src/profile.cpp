// 通过 GGML 回调收集逐算子和传输耗时；启用时包含诊断所需的后端同步。
#include "solid/profile.hpp"
#include <stdexcept>

namespace solid {
ExecutionProfile::ExecutionProfile(bool enabled) : enabled_(enabled) {
    if (enabled_) solid_set_transfer_callback(transferred, this);
}
ExecutionProfile::~ExecutionProfile() { if (enabled_) solid_set_transfer_callback(nullptr, nullptr); }
void ExecutionProfile::check() const { if (error_) std::rethrow_exception(error_); }
bool ExecutionProfile::evaluate(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & self = *static_cast<ExecutionProfile *>(userdata);
    if (ask) { self.start_ = std::chrono::steady_clock::now(); return true; }
    try {
        auto & total = self.nodes_[self.phase_ + ":" + ggml_op_name(tensor->op) + ":" + ggml_get_name(tensor)];
        ++total.calls;
        total.bytes += ggml_nbytes(tensor);
        total.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - self.start_).count();
    } catch (...) { self.error_ = std::current_exception(); return false; }
    return true;
}
void ExecutionProfile::transferred(const ggml_tensor * tensor, ggml_backend_t source, ggml_backend_t target,
                                    std::uint64_t ns, void * userdata) {
    auto & self = *static_cast<ExecutionProfile *>(userdata);
    try {
        const auto is_host = [](ggml_backend_t backend) {
            auto * device = ggml_backend_get_device(backend);
            return device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
        };
        auto * buffer = tensor->buffer;
        if (!buffer && tensor->view_src) buffer = tensor->view_src->buffer;
        const bool source_host = buffer ? ggml_backend_buffer_is_host(buffer) : is_host(source);
        const std::string direction = source_host ? (is_host(target) ? "H2H" : "H2D") : (is_host(target) ? "D2H" : "D2D");
        auto & total = self.transfers_[self.phase_ + ":" + direction + ":" + ggml_get_name(tensor)];
        ++total.calls;
        total.bytes += ggml_nbytes(tensor);
        total.ns += ns;
    } catch (...) { self.error_ = std::current_exception(); }
}
} // namespace solid
