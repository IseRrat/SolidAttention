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
        solid::trace::Scope request("Run", UINT32_MAX, UINT64_MAX, "annotation-check");
        { solid::trace::Scope phase("Prefill"); }
        solid::trace::Scope decode("Decode", UINT32_MAX, 1024);
        solid::trace::Range asynchronous("SSD->CPU submit..receipt", 2, 1024);
        auto transferred=std::make_shared<solid::trace::Range>(std::move(asynchronous));
        solid::TaskWorker coordinator(-1,"sa.coordinator");
        coordinator.submit([transferred] {
            check_name("sa.coordinator");
            solid::trace::Scope layer("CPU->GPU stage",2,1024,"annotation-check");
            transferred->finish();
        });
        coordinator.join();
        std::cout << "thread role names, cross-thread ranges and stage annotations passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
