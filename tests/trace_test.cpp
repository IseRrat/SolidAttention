#include "solid/trace.hpp"
#include "solid/scheduler.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <pthread.h>
#endif

namespace {
void check_name(const char * expected) {
#ifdef __linux__
    char actual[16]{};
    if (pthread_getname_np(pthread_self(), actual, sizeof(actual)) || std::string(actual) != expected)
        throw std::runtime_error("thread role was not applied to the OS thread");
#else
    (void) expected;
#endif
}
}
int main() {
    try {
        solid::trace::name_thread("sa.trace.main");
        check_name("sa.trace.main");
        { solid::trace::Scope phase("Prefill"); }
        solid::trace::Scope decode("Decode", UINT32_MAX, 1024);
        { solid::trace::ComputeScope q("MUL_MAT", "blk.2.attn_q.weight", "Qcur-2", nullptr); }
        { solid::trace::ComputeScope kv("MUL_MAT", "solid_fused_kv_weight", "kv_projection.L2", nullptr); }
        { solid::trace::ComputeScope k("MUL_MAT", "blk.2.attn_k.weight", "Kcur-2", nullptr); }
        { solid::trace::ComputeScope rope("ROPE", "Kcur-2", "Kcur-2", nullptr); }
        { solid::trace::ComputeScope v("MUL_MAT", "blk.2.attn_v.weight", "Vcur-2", nullptr); }
        { solid::trace::ComputeScope qk("MUL_MAT", "keys", "infllm.QK.L2", nullptr); }
        { solid::trace::ComputeScope softmax("SOFT_MAX", "logits", "infllm.softmax.L2", nullptr); }
        { solid::trace::ComputeScope pv("MUL_MAT", "values", "infllm.PV.L2", nullptr); }
        std::shared_ptr<solid::trace::Batch> current, receipt;
        {
            solid::trace::BatchContext batch(current, "SSD->CPU Prefetch", 2, 1024);
            receipt = current;
            if (receipt) { receipt->submitted(); receipt->submitted(); }
        }
        if (current) throw std::runtime_error("I/O context leaked past request planning");
        solid::trace::Range asynchronous("GPU->CPU Store", 2, 1024);
        auto transferred=std::make_shared<solid::trace::Range>(std::move(asynchronous));
        solid::TaskWorker coordinator(-1,"sa.coordinator");
        coordinator.submit([transferred, receipt=std::move(receipt)]() mutable {
            check_name("sa.coordinator");
            solid::trace::Scope layer("CPU->GPU Load",2,1024);
            receipt.reset();
            transferred->finish();
        });
        coordinator.join();
        std::cout << "thread role names, cross-thread ranges and stage annotations passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
