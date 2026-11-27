// 读取 CPU 拓扑和线程亲和性，为 I/O、协调线程预留核心，并在结束时恢复原设置。
#include "solid/thread_affinity.hpp"
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace solid {
#ifdef __linux__
namespace {
cpu_set_t mask() {
    cpu_set_t result; CPU_ZERO(&result);
    const auto status=pthread_getaffinity_np(pthread_self(),sizeof(result),&result);
    if (status) throw std::system_error(status,std::generic_category(),"read thread CPU affinity");
    return result;
}
void apply(const cpu_set_t & cpus) {
    const auto status=pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus);
    if (status) throw std::system_error(status,std::generic_category(),"set thread CPU affinity");
}
std::string core(int cpu) {
    const auto base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
    std::string package,id;
    std::ifstream a(base+"physical_package_id"),b(base+"core_id");
    if (!(a>>package) || !(b>>id)) return "cpu:"+std::to_string(cpu);
    return package+":"+id;
}
}
std::vector<int> thread_cpus() {
    const auto current=mask(); std::vector<int> result;
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&current)) result.push_back(cpu);
    return result;
}
void bind_thread_cpu(int cpu) {
    if (cpu<0) { if (cpu!=-1) throw std::invalid_argument("CPU identifier must be -1 or nonnegative"); return; }
    if (cpu>=CPU_SETSIZE) throw std::invalid_argument("CPU identifier exceeds affinity mask capacity");
    cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(cpu,&selected); apply(selected);
}
struct WorkerPlacement::Impl {
    cpu_set_t original;
    pthread_t owner;
    int io=-1,coordinator=-1;
    std::vector<int> inference;
    bool applied=false;
};
WorkerPlacement::WorkerPlacement(int io,int coordinator):impl_(std::make_unique<Impl>()) {
    if (io < -1 || coordinator < -1) throw std::invalid_argument("invalid worker CPU identifier");
    impl_->original=mask(); impl_->owner=pthread_self();
    const auto allowed=thread_cpus();
    std::map<int,std::string> cores;
    for (const auto cpu:allowed) cores[cpu]=core(cpu);
    const auto valid=[&](int cpu) { return std::find(allowed.begin(),allowed.end(),cpu)!=allowed.end(); };
    if ((io>=0 && !valid(io)) || (coordinator>=0 && !valid(coordinator))) throw std::invalid_argument("worker CPU is outside the inference affinity");
    for (auto a=allowed.rbegin();a!=allowed.rend();++a) {
        if (io>=0 && *a!=io) continue;
        for (auto b=allowed.rbegin();b!=allowed.rend();++b) {
            if ((coordinator>=0 && *b!=coordinator) || cores[*a]==cores[*b]) continue;
            std::vector<int> remaining;
            for (const auto cpu:allowed) if (cores[cpu]!=cores[*a] && cores[cpu]!=cores[*b]) remaining.push_back(cpu);
            if (remaining.empty()) continue;
            impl_->io=*a; impl_->coordinator=*b; impl_->inference=std::move(remaining);
            cpu_set_t selected; CPU_ZERO(&selected);
            for (const auto cpu:impl_->inference) CPU_SET(cpu,&selected);
            apply(selected); impl_->applied=true; return;
        }
    }
    throw std::invalid_argument("prefetch requires distinct I/O and coordination cores plus an inference core");
}
WorkerPlacement::~WorkerPlacement() {
    if (impl_->applied) (void) pthread_setaffinity_np(impl_->owner,sizeof(impl_->original),&impl_->original);
}
#else
std::vector<int> thread_cpus() { return {}; }
void bind_thread_cpu(int cpu) { if (cpu!=-1) throw std::runtime_error("CPU binding requires Linux"); }
struct WorkerPlacement::Impl {};
WorkerPlacement::WorkerPlacement(int,int) { throw std::runtime_error("I/O worker placement requires Linux"); }
WorkerPlacement::~WorkerPlacement()=default;
#endif
int WorkerPlacement::io_cpu() const {
#ifdef __linux__
    return impl_->io;
#else
    return -1;
#endif
}
int WorkerPlacement::coordinator_cpu() const {
#ifdef __linux__
    return impl_->coordinator;
#else
    return -1;
#endif
}
const std::vector<int> & WorkerPlacement::inference_cpus() const {
#ifdef __linux__
    return impl_->inference;
#else
    static const std::vector<int> empty; return empty;
#endif
}
} // namespace solid
