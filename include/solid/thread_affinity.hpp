#pragma once
#include <memory>
#include <vector>

namespace solid {
std::vector<int> thread_cpus();
void bind_thread_cpu(int cpu);
// Reserve distinct physical cores for the I/O and coordination workers. The
// creating inference thread excludes their siblings until this scope ends.
class WorkerPlacement {
public:
    WorkerPlacement(int io_cpu=-1,int coordinator_cpu=-1);
    ~WorkerPlacement();
    int io_cpu() const;
    int coordinator_cpu() const;
    const std::vector<int> & inference_cpus() const;
    WorkerPlacement(const WorkerPlacement &)=delete;
    WorkerPlacement & operator=(const WorkerPlacement &)=delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace solid
