#include "solid/scheduler.hpp"
#include "solid/thread_affinity.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <stdexcept>

namespace solid {
namespace {
std::uint64_t timestamp() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
const char * resource_name(TaskResource resource) {
    switch (resource) {
        case TaskResource::host: return "host";
        case TaskResource::ssd: return "ssd";
        case TaskResource::h2d: return "h2d";
        case TaskResource::d2h: return "d2h";
        case TaskResource::gpu: return "gpu";
    }
    throw std::logic_error("unknown task resource");
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("DAG duration overflow");
    return a + b;
}
}
ScheduleMode parse_schedule_mode(const std::string & name) {
    if (name == "naive") return ScheduleMode::naive;
    if (name == "overlap") return ScheduleMode::overlap;
    if (name == "overlap+sync_reuse") return ScheduleMode::sync_reuse;
    throw std::invalid_argument("unknown schedule mode");
}
const char * schedule_mode_name(ScheduleMode mode) {
    switch (mode) {
        case ScheduleMode::naive: return "naive";
        case ScheduleMode::overlap: return "overlap";
        case ScheduleMode::sync_reuse: return "overlap+sync_reuse";
    }
    throw std::invalid_argument("unknown schedule mode");
}
ScheduleTrace::ScheduleTrace(const std::filesystem::path & path) : origin_(timestamp()) {
    if (!path.empty()) {
        if (std::filesystem::exists(path)) throw std::runtime_error("schedule trace already exists");
        stream_.open(path, std::ios::binary);
        if (!stream_) throw std::runtime_error("cannot create schedule trace");
    }
}
ScheduleTrace::~ScheduleTrace() = default;
std::uint64_t ScheduleTrace::now() const { return timestamp(); }
void ScheduleTrace::define(std::uint64_t frame, std::uint32_t layer, std::uint64_t position, ScheduleMode mode,
                           std::uint32_t task, const TaskSpec & spec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return;
    stream_ << nlohmann::json({{"event", "define"}, {"frame", frame}, {"layer", layer}, {"position", position},
        {"mode", schedule_mode_name(mode)}, {"task", task}, {"name", spec.name}, {"resource", resource_name(spec.resource)},
        {"predecessors", spec.predecessors}, {"estimate_ns", spec.estimate_ns}, {"origin_ns", origin_}, {"clock", "steady_clock_ns"}}).dump() << '\n';
    if (!stream_) throw std::runtime_error("schedule trace write failed");
}
void ScheduleTrace::event(std::uint64_t frame, std::uint32_t task, const char * name, std::uint64_t time, const std::string & fence,std::optional<std::uint64_t> device_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return;
    nlohmann::json record={{"event", name}, {"frame", frame}, {"task", task}, {"timestamp_ns", time}, {"fence", fence}};
    if (device_ns) record["device_ns"]=*device_ns;
    stream_ << record.dump() << '\n';
    if (!stream_) throw std::runtime_error("schedule trace write failed");
}
void ScheduleTrace::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_.is_open()) { stream_.flush(); if (!stream_) throw std::runtime_error("schedule trace flush failed"); }
}
TaskDag::TaskDag(std::vector<TaskSpec> specs, ScheduleTrace * trace, std::uint64_t frame, std::uint32_t layer,
                 std::uint64_t position, ScheduleMode mode) : trace_(trace), frame_(frame) {
    if (specs.empty() || specs.size() > UINT32_MAX) throw std::invalid_argument("invalid DAG size");
    for (auto & spec : specs) tasks_.push_back({std::move(spec)});
    std::vector<std::size_t> indegree(tasks_.size());
    std::vector<std::vector<std::uint32_t>> successors(tasks_.size());
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) {
        std::set<std::uint32_t> unique;
        const auto & spec = tasks_[id].spec;
        if (spec.name.empty() || !spec.estimate_ns) throw std::invalid_argument("invalid task specification");
        for (const auto before : spec.predecessors) {
            if (before >= tasks_.size() || before == id || !unique.insert(before).second) throw std::invalid_argument("invalid DAG edge");
            successors[before].push_back(id); ++indegree[id];
        }
    }
    std::vector<std::uint32_t> queue;
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) if (!indegree[id]) queue.push_back(id);
    for (std::size_t i = 0; i < queue.size(); ++i) {
        const auto id = queue[i]; order_.push_back(id);
        for (const auto after : successors[id]) if (--indegree[after] == 0) queue.push_back(after);
    }
    if (order_.size() != tasks_.size()) throw std::invalid_argument("DAG contains a cycle");
    std::vector<std::uint64_t> earliest(tasks_.size()), tail(tasks_.size());
    std::uint64_t makespan = 0;
    for (const auto id : order_) {
        for (const auto before : tasks_[id].spec.predecessors) earliest[id] = std::max(earliest[id], add(earliest[before], tasks_[before].spec.estimate_ns));
        makespan = std::max(makespan, add(earliest[id], tasks_[id].spec.estimate_ns));
    }
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        const auto id = *it;
        for (const auto after : successors[id]) tail[id] = std::max(tail[id], add(tasks_[after].spec.estimate_ns, tail[after]));
        tasks_[id].latest_start = makespan - add(tasks_[id].spec.estimate_ns, tail[id]);
        tasks_[id].critical = tasks_[id].latest_start == earliest[id];
    }
    if (trace_) for (std::uint32_t id = 0; id < tasks_.size(); ++id) trace_->define(frame, layer, position, mode, id, tasks_[id].spec);
}
bool TaskDag::dependencies_complete(std::uint32_t task) const {
    for (const auto before : tasks_.at(task).spec.predecessors) if (tasks_[before].state != TaskState::complete) return false;
    return true;
}
std::vector<std::uint32_t> TaskDag::ready(const std::vector<std::uint32_t> & candidates) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<std::uint32_t> ids;
    std::vector<bool> seen(tasks_.size(),false);
    const auto consider=[&](std::uint32_t id) {
        if (id<tasks_.size() && !seen[id]) {
            seen[id]=true;
            if (tasks_[id].state==TaskState::pending && dependencies_complete(id)) ids.push_back(id);
        }
    };
    if (candidates.empty()) for (std::uint32_t id=0;id<tasks_.size();++id) consider(id);
    else for (const auto id:candidates) consider(id);
    std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
        const auto & x = tasks_[a]; const auto & y = tasks_[b];
        if (x.critical != y.critical) return x.critical;
        if (x.critical && x.latest_start != y.latest_start) return x.latest_start < y.latest_start;
        const auto io = [](TaskResource r) { return r == TaskResource::ssd || r == TaskResource::h2d || r == TaskResource::d2h; };
        if (io(x.spec.resource) != io(y.spec.resource)) return io(x.spec.resource);
        return x.latest_start == y.latest_start ? a < b : x.latest_start < y.latest_start;
    });
    return ids;
}
void TaskDag::start(std::uint32_t id, std::uint64_t now,const std::string & proof) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto & record = tasks_.at(id);
    if (record.state != TaskState::pending || !dependencies_complete(id)) throw std::logic_error("task started before dependencies completed");
    for (const auto before : record.spec.predecessors) if (now < tasks_[before].end_ns) throw std::logic_error("nonmonotonic task start");
    record.start_ns = now; record.state = TaskState::running;
    if (trace_) trace_->event(frame_, id, "start", now,proof);
}
void TaskDag::complete(std::uint32_t id, std::uint64_t now, const std::string & proof,std::optional<std::uint64_t> device_ns) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto & record = tasks_.at(id);
    if (record.state != TaskState::running || now < record.start_ns) throw std::logic_error("invalid task completion");
    record.end_ns = now; record.state = TaskState::complete; record.device_ns=device_ns;
    if (trace_) trace_->event(frame_, id, "complete", now, proof,device_ns);
}
void TaskDag::fence(const std::vector<std::uint32_t> & covered, std::uint64_t now, const std::string & proof) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (proof.empty() || covered.empty()) throw std::invalid_argument("fence requires a completion proof");
    std::set<std::uint32_t> remaining(covered.begin(), covered.end());
    if (remaining.size() != covered.size()) throw std::invalid_argument("duplicate task in fence");
    // Validate all edges before mutating any task.
    for (const auto id : remaining) {
        if (tasks_.at(id).state == TaskState::complete || tasks_[id].state == TaskState::failed) throw std::logic_error("stale fence");
        if (tasks_[id].state == TaskState::running && now < tasks_[id].start_ns) throw std::logic_error("nonmonotonic fence");
        for (const auto before : tasks_[id].spec.predecessors) if (tasks_[before].state != TaskState::complete && !remaining.count(before)) {
            throw std::logic_error("fence does not cover a prerequisite");
        }
        for (const auto before : tasks_[id].spec.predecessors) if (tasks_[before].state == TaskState::complete && now < tasks_[before].end_ns) {
            throw std::logic_error("fence precedes its prerequisite");
        }
    }
    for (const auto id : order_) if (remaining.count(id)) {
        if (tasks_[id].state == TaskState::pending) start(id, now);
        complete(id, now, proof);
    }
}
void TaskDag::abort(std::uint64_t now) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) if (tasks_[id].state != TaskState::complete && tasks_[id].state != TaskState::failed) {
        tasks_[id].state = TaskState::failed;
        if (trace_) trace_->event(frame_, id, "failed", now);
    }
}
TaskRecord TaskDag::task(std::uint32_t id) const { std::lock_guard<std::recursive_mutex> lock(mutex_); return tasks_.at(id); }
bool TaskDag::finished() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return std::all_of(tasks_.begin(), tasks_.end(), [](const auto & task) { return task.state == TaskState::complete; });
}
std::uint64_t TaskDag::critical_path_ns() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!finished()) throw std::logic_error("critical path requires completed tasks");
    std::vector<std::uint64_t> costs(tasks_.size());
    std::uint64_t maximum = 0;
    for (const auto id : order_) {
        for (const auto before : tasks_[id].spec.predecessors) costs[id] = std::max(costs[id], costs[before]);
        costs[id] = add(costs[id], tasks_[id].end_ns - tasks_[id].start_ns); maximum = std::max(maximum, costs[id]);
    }
    return maximum;
}
std::uint64_t TaskDag::overlap_ns(std::uint32_t a, std::uint32_t b) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto & x = tasks_.at(a); const auto & y = tasks_.at(b);
    if (x.state != TaskState::complete || y.state != TaskState::complete) throw std::logic_error("overlap requires completed tasks");
    const auto begin = std::max(x.start_ns, y.start_ns), end = std::min(x.end_ns, y.end_ns);
    return end > begin ? end - begin : 0;
}
TaskWorker::TaskWorker(int cpu) {
    thread_=std::thread([this,cpu] {
        std::exception_ptr error;
        try { bind_thread_cpu(cpu); } catch (...) { error=std::current_exception(); }
        { std::lock_guard<std::mutex> lock(mutex_); initialization_error_=error; initialized_=true; }
        cv_.notify_all();
        if (!error) run();
    });
    std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[&] { return initialized_; });
    if (initialization_error_) { const auto error=initialization_error_; lock.unlock(); thread_.join(); std::rethrow_exception(error); }
}
TaskWorker::~TaskWorker() {
    try { join(); } catch (...) {}
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    cv_.notify_all(); thread_.join();
}
void TaskWorker::submit(std::function<void()> work) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_ || !work || stopping_) throw std::logic_error("worker capacity exceeded");
    work_ = std::move(work); busy_ = true; done_ = false; error_ = nullptr;
    cv_.notify_all();
}
void TaskWorker::join() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!busy_) return;
    cv_.wait(lock, [&] { return done_; });
    auto error = error_; busy_ = false; done_ = false; error_ = nullptr;
    lock.unlock(); if (error) std::rethrow_exception(error);
}
void TaskWorker::run() {
    for (;;) {
        std::function<void()> work;
        { std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock, [&] { return stopping_ || (busy_ && !done_ && work_); });
          if (stopping_) return;
          work = std::move(work_); }
        std::exception_ptr error;
        try { work(); } catch (...) { error = std::current_exception(); }
        { std::lock_guard<std::mutex> lock(mutex_); error_ = error; done_ = true; }
        cv_.notify_all();
    }
}
} // namespace solid
