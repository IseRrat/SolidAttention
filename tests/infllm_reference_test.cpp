// 运行原版 InfLLM 与本仓库 CPU 路径的对照检查。
#include "infllm_reference_cases.hpp"

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: solid-infllm-reference-test upstream-fixture.bin");
        test_infllm::run(argv[1], [](auto &, auto, auto, auto) {},
            [](auto & index, auto first, auto count, const auto & kv, const auto & scores) {
                index.observe_prefill_scores(0, first, count, kv, scores);
            }, [](const auto & queries, auto count, auto width) {
                std::vector<float> mean(width);
                solid::mean_infllm_query(queries.data(), mean.data(), width, count);
                return mean;
            });
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
