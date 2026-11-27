// 逐算子计算与传输统计的启停、结果和回调检查。
#include "solid/profile.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
        solid::ExecutionProfile profile(true);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        ggml_set_input(input);
        auto * output = ggml_scale(context.get(), input, 2);
        ggml_set_name(output, "profile_test");
        auto * graph = ggml_new_graph_custom(context.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_t backends[] = {backend.get()};
        std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
            ggml_backend_sched_new(backends, nullptr, 1, 32, false, true), ggml_backend_sched_free);
        if (!scheduler || !ggml_backend_sched_alloc_graph(scheduler.get(), graph)) throw std::runtime_error("profile graph allocation");
        ggml_backend_sched_set_eval_callback(scheduler.get(), solid::ExecutionProfile::evaluate, &profile);
        const float data[] = {1, -2, 3, -4};
        for (int phase = 0; phase < 2; ++phase) {
            if (phase == 1) profile.set_decode();
            ggml_backend_tensor_set(input, data, 0, sizeof(data));
            if (ggml_backend_sched_graph_compute(scheduler.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("profile graph execution");
            profile.check();
            float actual[4];
            ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
            for (int i = 0; i < 4; ++i) if (actual[i] != data[i] * 2) throw std::runtime_error("profiling changed graph output");
        }
        if (profile.nodes().size() != 2) throw std::runtime_error("profile phases are not separated");
        for (const auto & pair : profile.nodes()) if (pair.second.calls != 1 || pair.second.bytes != 16 || pair.second.ns == 0) {
            throw std::runtime_error("profile counts or timing");
        }
        std::cout << "serialized node profiling and phase accounting checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
