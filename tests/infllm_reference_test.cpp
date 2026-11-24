#include "infllm_reference_cases.hpp"

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: solid-infllm-reference-test upstream-fixture.bin");
        test_infllm::run(argv[1], [](auto &, auto, auto, auto) {},
            [](auto & index, auto first, auto count, const auto & kv, const auto & scores) {
                index.observe_prefill_scores(0, first, count, kv, scores);
            });
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
