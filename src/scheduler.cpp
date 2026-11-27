// 实现任务依赖推进和关键路径优先选择，并管理持久工作线程与调度统计。
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
// 取单调时钟的纳秒时间戳，供任务起止计时使用。
std::uint64_t timestamp() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// 把任务使用的资源转成日志名称：CPU、SSD、H2D、D2H 或 GPU。
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
// 累加两段纳秒时长，并检查总时长是否超过整数可表示范围。
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("DAG duration overflow");
    return a + b;
}
}
// 将配置中的模式名称解析为 Naive、计算搬运重叠（OL）或重叠加同步复用（OL+SR）。
ScheduleMode parse_schedule_mode(const std::string & name) {
    if (name == "naive") return ScheduleMode::naive;
    if (name == "overlap") return ScheduleMode::overlap;
    if (name == "overlap+sync_reuse") return ScheduleMode::sync_reuse;
    throw std::invalid_argument("unknown schedule mode");
}
// 把调度模式转回配置名称，供结果和调度日志记录。
const char * schedule_mode_name(ScheduleMode mode) {
    switch (mode) {
        case ScheduleMode::naive: return "naive";
        case ScheduleMode::overlap: return "overlap";
        case ScheduleMode::sync_reuse: return "overlap+sync_reuse";
    }
    throw std::invalid_argument("unknown schedule mode");
}
// 初始化日志时间原点；传入路径时创建逐行 JSON 日志，空路径表示关闭记录。
ScheduleTrace::ScheduleTrace(const std::filesystem::path & path) : origin_(path.empty()?0:timestamp()) {
    if (!path.empty()) {
        if (std::filesystem::exists(path)) throw std::runtime_error("schedule trace already exists");
        stream_.open(path, std::ios::binary);
        if (!stream_) throw std::runtime_error("cannot create schedule trace");
    }
}
// 销毁记录器，由文件流成员自动关闭日志文件。
ScheduleTrace::~ScheduleTrace() = default;
// 详细调度日志开启时读取单调时钟；关闭时返回 0，省去推理中的计时查询。
std::uint64_t ScheduleTrace::now() const { return enabled()?timestamp():0; }
// 写入任务定义：所属层和 token、使用资源、前驱任务及固定排序单位，供离线还原 DAG。
void ScheduleTrace::define(std::uint64_t frame, std::uint32_t layer, std::uint64_t position, ScheduleMode mode,
                           std::uint32_t task, const TaskSpec & spec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return;
    stream_ << nlohmann::json({{"event", "define"}, {"frame", frame}, {"layer", layer}, {"position", position},
        {"mode", schedule_mode_name(mode)}, {"task", task}, {"name", spec.name}, {"resource", resource_name(spec.resource)},
        {"predecessors", spec.predecessors}, {"cost_units", spec.cost_units}, {"origin_ns", origin_}, {"clock", "steady_clock_ns"}}).dump() << '\n';
    if (!stream_) throw std::runtime_error("schedule trace write failed");
}
// 写入任务开始、完成或失败事件；可附完成证明和设备实际耗时。
void ScheduleTrace::event(std::uint64_t frame, std::uint32_t task, const char * name, std::uint64_t time, const std::string & fence,std::optional<std::uint64_t> device_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return;
    nlohmann::json record={{"event", name}, {"frame", frame}, {"task", task}, {"timestamp_ns", time}, {"fence", fence}};
    if (device_ns) record["device_ns"]=*device_ns;
    stream_ << record.dump() << '\n';
    if (!stream_) throw std::runtime_error("schedule trace write failed");
}
// 把日志流中尚未输出的内容交给文件系统，并检查写入状态。
void ScheduleTrace::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_.is_open()) { stream_.flush(); if (!stream_) throw std::runtime_error("schedule trace flush failed"); }
}
// 建立任务依赖图并生成拓扑顺序；按固定任务单位计算关键路径及最晚开始序位（LST）。
// 推理路径每个微任务使用一个单位；与最早开始序位相等的任务属于关键路径。
TaskDag::TaskDag(std::vector<TaskSpec> specs, ScheduleTrace * trace, std::uint64_t frame, std::uint32_t layer,
                 std::uint64_t position, ScheduleMode mode) : trace_(trace && trace->enabled() ? trace : nullptr), frame_(frame), layer_(layer), position_(position) {
    if (specs.empty() || specs.size() > UINT32_MAX) throw std::invalid_argument("invalid DAG size");
    tasks_.reserve(specs.size());
    order_.reserve(specs.size());
    for (auto & spec : specs) tasks_.push_back({std::move(spec)});
    if (solid::trace::enabled() && std::any_of(tasks_.begin(), tasks_.end(), [](const auto & t) {
        return t.spec.name == "kv_d2h_ready";
    })) ranges_.resize(tasks_.size());
    // 建立后继表和入度；拓扑遍历时，每完成一个前驱就减少后继的入度。
    std::vector<std::size_t> indegree(tasks_.size());
    std::vector<std::vector<std::uint32_t>> successors(tasks_.size());
    std::vector<std::uint32_t> seen(tasks_.size(), UINT32_MAX);
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) {
        const auto & spec = tasks_[id].spec;
        if (spec.name.empty() || !spec.cost_units) throw std::invalid_argument("invalid task specification");
        for (const auto before : spec.predecessors) {
            if (before >= tasks_.size() || before == id || seen[before] == id) throw std::invalid_argument("invalid DAG edge");
            seen[before] = id;
            successors[before].push_back(id); ++indegree[id];
        }
    }
    std::vector<std::uint32_t> queue;
    queue.reserve(tasks_.size());
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) if (!indegree[id]) queue.push_back(id);
    for (std::size_t i = 0; i < queue.size(); ++i) {
        const auto id = queue[i]; order_.push_back(id);
        for (const auto after : successors[id]) if (--indegree[after] == 0) queue.push_back(after);
    }
    if (order_.size() != tasks_.size()) throw std::invalid_argument("DAG contains a cycle");
    // 正向求每个任务最早何时能开始，以及整张图的预计完成时间。
    std::vector<std::uint64_t> earliest(tasks_.size()), tail(tasks_.size());
    std::uint64_t makespan = 0;
    for (const auto id : order_) {
        for (const auto before : tasks_[id].spec.predecessors) earliest[id] = std::max(earliest[id], add(earliest[before], tasks_[before].spec.cost_units));
        makespan = std::max(makespan, add(earliest[id], tasks_[id].spec.cost_units));
    }
    // 反向求后续最长耗时，倒推出各任务最晚何时开始才不推迟整张图。
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        const auto id = *it;
        for (const auto after : successors[id]) tail[id] = std::max(tail[id], add(tasks_[after].spec.cost_units, tail[after]));
        tasks_[id].latest_start = makespan - add(tasks_[id].spec.cost_units, tail[id]);
        tasks_[id].critical = tasks_[id].latest_start == earliest[id];
    }
    if (trace_) for (std::uint32_t id = 0; id < tasks_.size(); ++id) trace_->define(frame, layer, position, mode, id, tasks_[id].spec);
}
// 检查指定任务的全部直接前驱是否已经完成。
bool TaskDag::dependencies_complete(std::uint32_t task) const {
    for (const auto before : tasks_.at(task).spec.predecessors) if (tasks_[before].state != TaskState::complete) return false;
    return true;
}
// 找出尚未开始且前驱已完成的任务，并按关键路径、I/O、LST 排序。
// candidates 为空时检查全图，否则只检查调用方给出的候选；返回顺序由调用方用于派发。
std::vector<std::uint32_t> TaskDag::ready(const std::vector<std::uint32_t> & candidates) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<std::uint32_t> ids;
    std::vector<bool> seen(tasks_.size(),false);
    // 同一候选只检查一次，符合启动条件时才加入就绪列表。
    const auto consider=[&](std::uint32_t id) {
        if (id<tasks_.size() && !seen[id]) {
            seen[id]=true;
            if (tasks_[id].state==TaskState::pending && dependencies_complete(id)) ids.push_back(id);
        }
    };
    if (candidates.empty()) for (std::uint32_t id=0;id<tasks_.size();++id) consider(id);
    else for (const auto id:candidates) consider(id);
    // 比较就绪任务的派发优先级；同优先级与同 LST 时用任务编号固定顺序。
    std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
        const auto & x = tasks_[a]; const auto & y = tasks_[b];
        if (x.critical != y.critical) return x.critical;
        if (x.critical && x.latest_start != y.latest_start) return x.latest_start < y.latest_start;
        // SSD 读写、主机上传与设备回传都归入 I/O 任务。
        const auto io = [](TaskResource r) { return r == TaskResource::ssd || r == TaskResource::h2d || r == TaskResource::d2h; };
        if (io(x.spec.resource) != io(y.spec.resource)) return io(x.spec.resource);
        return x.latest_start == y.latest_start ? a < b : x.latest_start < y.latest_start;
    });
    return ids;
}
// 确认依赖已满足后，将任务标为运行中，保存开始时间并开启对应阶段记录。
void TaskDag::start(std::uint32_t id, std::uint64_t now,const std::string & proof) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto & record = tasks_.at(id);
    if (record.state != TaskState::pending || !dependencies_complete(id)) throw std::logic_error("task started before dependencies completed");
    for (const auto before : record.spec.predecessors) if (now < tasks_[before].end_ns) throw std::logic_error("nonmonotonic task start");
    record.start_ns = now; record.state = TaskState::running;
    if (!ranges_.empty()) {
        const auto * phase = record.spec.name == "kv_d2h_ready" ? "GPU->CPU Store" : nullptr;
        if (phase) ranges_[id] = trace::Range(phase, layer_, position_);
    }
    if (trace_) trace_->event(frame_, id, "start", now,proof);
}
// 将运行中的任务标为完成，保存主机结束时间和可选设备耗时，结束阶段记录。
void TaskDag::complete(std::uint32_t id, std::uint64_t now, const std::string & proof,std::optional<std::uint64_t> device_ns) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto & record = tasks_.at(id);
    if (record.state != TaskState::running || now < record.start_ns) throw std::logic_error("invalid task completion");
    record.end_ns = now; record.state = TaskState::complete; record.device_ns=device_ns;
    if (!ranges_.empty()) ranges_[id].finish();
    if (trace_) trace_->event(frame_, id, "complete", now, proof,device_ns);
}
// 接收调用方提供的一份实际完成证明，按依赖顺序确认它覆盖的多个任务。
// 本函数更新 DAG 状态；实际设备等待由调用方执行，复用同一证明可减少重复等待。
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
// 中断图：将尚未结束的任务标为失败，并关闭其阶段记录，保留已经完成的任务状态。
void TaskDag::abort(std::uint64_t now) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (std::uint32_t id = 0; id < tasks_.size(); ++id) if (tasks_[id].state != TaskState::complete && tasks_[id].state != TaskState::failed) {
        tasks_[id].state = TaskState::failed;
        if (!ranges_.empty()) ranges_[id].finish();
        if (trace_) trace_->event(frame_, id, "failed", now);
    }
}
// 返回完整任务副本，包含名称、依赖和运行状态，供显式检查或诊断使用。
TaskRecord TaskDag::task(std::uint32_t id) const { std::lock_guard<std::recursive_mutex> lock(mutex_); return tasks_.at(id); }
// 仅复制状态、时间和优先级字段，供调度热路径查询，省去复制名称与依赖数组。
TaskSnapshot TaskDag::snapshot(std::uint32_t id) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto & task = tasks_.at(id);
    return {task.state,task.start_ns,task.end_ns,task.latest_start,task.critical,task.device_ns};
}
// 判断图中所有任务是否均已成功完成。
bool TaskDag::finished() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return std::all_of(tasks_.begin(), tasks_.end(), [](const auto & task) { return task.state == TaskState::complete; });
}
// 创建并命名常驻工作线程，按 cpu 参数设置亲和性，等线程初始化完成后返回。
TaskWorker::TaskWorker(int cpu, const char * name) {
    thread_=std::thread([this,cpu,name=std::string(name)] {
        trace::name_thread(name.c_str());
        std::exception_ptr error;
        try { bind_thread_cpu(cpu); } catch (...) { error=std::current_exception(); }
        { std::lock_guard<std::mutex> lock(mutex_); initialization_error_=error; initialized_=true; }
        cv_.notify_all();
        if (!error) run();
    });
    std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[&] { return initialized_; });
    if (initialization_error_) { const auto error=initialization_error_; lock.unlock(); thread_.join(); std::rethrow_exception(error); }
}
// 等当前任务结束，通知工作线程退出，再回收线程。
TaskWorker::~TaskWorker() {
    try { join(); } catch (...) {}
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    cv_.notify_all(); thread_.join();
}
// 投递一个后台任务并唤醒工作线程；当前任务收取完成后才能投递下一项。
void TaskWorker::submit(std::function<void()> work) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_ || !work || stopping_) throw std::logic_error("worker capacity exceeded");
    work_ = std::move(work); busy_ = true; done_ = false; error_ = nullptr;
    cv_.notify_all();
}
// 等待当前任务结束、收取结果状态；任务抛出的异常在调用线程重新抛出，线程保留供下次使用。
void TaskWorker::join() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!busy_) return;
    cv_.wait(lock, [&] { return done_; });
    auto error = error_; busy_ = false; done_ = false; error_ = nullptr;
    lock.unlock(); if (error) std::rethrow_exception(error);
}
// 工作线程主循环：等待任务、执行函数、保存异常或完成状态，再通知等待方。
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
// 等待调用方关心的中间进度；ready 成立、整个任务结束或线程空闲时返回。
// 整个任务尚未结束时，调用方仍需后续 join 收尾。
void TaskWorker::wait_until(const std::function<bool()> & ready) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !busy_ || done_ || ready(); });
    const auto error = error_;
    if (error) { busy_ = false; done_ = false; error_ = nullptr; }
    lock.unlock();
    if (error) std::rethrow_exception(error);
}
// 通知任务有新进度，让 wait_until 的等待方重新检查自己的 ready 条件。
void TaskWorker::notify_progress() {
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
}
} // namespace solid
