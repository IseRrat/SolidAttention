// DAG 依赖、关键路径、LST 优先级与工作线程检查。
#include "solid/scheduler.hpp"
#include "prefetch_fakes.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>

using namespace test_prefetch;
using solid::TaskResource;
namespace {
// 对照算法 1 的三层选择规则，并检查实测起止时间不会改写静态排序。
void algorithm_one_order() {
    const std::vector<solid::TaskSpec> specs{
        {"start",TaskResource::host,{}}, {"critical",TaskResource::gpu,{0}},
        {"critical-next",TaskResource::gpu,{1}}, {"critical-last",TaskResource::gpu,{2}},
        {"short-io",TaskResource::ssd,{0}}, {"host",TaskResource::host,{0}},
        {"host-next",TaskResource::host,{5}}, {"earlier-io",TaskResource::h2d,{0}},
        {"io-next",TaskResource::host,{7}}, {"join",TaskResource::host,{3,4,6,8}}};
    solid::TaskDag a(specs),b(specs);
    a.start(0,1); a.complete(0,2);
    b.start(0,100); b.complete(0,1000000);
    require(a.ready()==std::vector<std::uint32_t>{1,7,4,5},
        "Algorithm 1 must choose critical tasks, then I/O by LST, then other ready tasks");
    require(a.ready()==b.ready(),"observed task duration changed static DAG priorities");
    a.start(1,2);
    require(a.ready()==std::vector<std::uint32_t>{7,4,5},"a running predecessor unlocked its successor");
    require(a.ready({4,7}).front()==7,"I/O task with later LST outranked an earlier one");
    const solid::ReadDispatch critical{solid::ReadPriority::selected,true,10};
    const solid::ReadDispatch correction{solid::ReadPriority::correction_selected,false,0};
    require(critical.before(correction),"request category overrode critical-path priority");
    require(solid::ReadDispatch{solid::ReadPriority::deterministic,true,1}.before(critical),
        "request category overrode earlier LST");
}
void dependency_and_priority() {
    solid::TaskDag dag({{"query", TaskResource::d2h, {}, 1}, {"miss", TaskResource::ssd, {0}, 8},
        {"kv", TaskResource::gpu, {0}, 5}, {"attention", TaskResource::gpu, {1,2}, 4}, {"store", TaskResource::ssd, {2}, 1}});
    rejects([&] { dag.start(3, 0); });
    dag.start(0, 0); dag.complete(0, 1);
    require(dag.ready().front() == 1, "critical correction did not outrank noncritical work");
    dag.start(1, 1); dag.start(2, 1);
    rejects([&] { dag.fence({3}, 2, "incomplete coverage"); });
    require(dag.task(3).state == solid::TaskState::pending, "invalid shared fence partially advanced task state");
    dag.complete(2, 6); dag.start(4, 6); dag.complete(4, 7); dag.complete(1, 9);
    dag.start(3, 9); dag.complete(3, 13);
    for (std::uint32_t id=0;id<dag.size();++id) {
        const auto full=dag.task(id);
        const auto status=dag.snapshot(id);
        require(status.state==full.state && status.start_ns==full.start_ns && status.end_ns==full.end_ns &&
                status.latest_start==full.latest_start && status.critical==full.critical &&
                status.device_ns==full.device_ns && status.duration_ns()==full.duration_ns(),"lightweight task state lost scheduling or timing data");
    }
    require(dag.finished(), "longest dependency chain");
    rejects([&] { dag.complete(1, 14); });
    rejects([&] { solid::TaskDag cycle({{"a",TaskResource::host,{1},1},{"b",TaskResource::host,{0},1}}); });
    rejects([&] { solid::TaskDag duplicate({{"a",TaskResource::host,{},1},{"b",TaskResource::host,{0,0},1}}); });
    solid::TaskDag fence({{"gpu",TaskResource::gpu,{},1},{"d2h",TaskResource::d2h,{0},1}});
    fence.start(0, 5);
    rejects([&] { fence.fence({0,1}, 4, "too early"); });
    require(fence.task(0).state == solid::TaskState::running && fence.task(1).state == solid::TaskState::pending, "failed fence mutated state");
    fence.fence({0,1}, 8, "one boundary covers both");
    require(fence.finished(), "shared fence did not complete covered tasks");
    solid::TaskDag device({{"gpu",TaskResource::gpu,{},1},{"upload",TaskResource::h2d,{0},1},{"consumer",TaskResource::host,{0,1},1}});
    device.start(0,10); device.complete(0,20,"gpu_event",3);
    require(device.snapshot(0).device_ns==3 && device.snapshot(0).duration_ns()==3,"snapshot replaced GPU service time with CPU receipt time");
    rejects([&] { (void) device.snapshot(3); });
}
void worker_overlap_and_error() {
    solid::TaskDag dag({{"q",TaskResource::host,{},1},{"ssd",TaskResource::ssd,{0},1},
                       {"kv",TaskResource::gpu,{0},1},{"attention",TaskResource::gpu,{1,2},1}});
    dag.start(0, 0); dag.complete(0, 1);
    solid::TaskWorker worker;
    std::mutex mutex; std::condition_variable cv;
    bool waiting = false, release = false;
    worker.submit([&] {
        dag.start(1, 2);
        std::unique_lock<std::mutex> lock(mutex); waiting = true; cv.notify_all();
        cv.wait(lock, [&] { return release; }); lock.unlock(); dag.complete(1, 5);
    });
    { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return waiting; }); }
    rejects([&] { worker.submit([] {}); });
    dag.start(2, 3); // GPU work progresses while the I/O worker is deliberately pending.
    { std::lock_guard<std::mutex> lock(mutex); release = true; } cv.notify_all();
    worker.join(); dag.complete(2, 6); dag.start(3, 6); dag.complete(3, 8);
    require(dag.finished(), "bounded worker serialized independent tasks");
    worker.submit([] { throw std::runtime_error("injected worker failure"); });
    rejects([&] { worker.join(); });
    std::atomic<int> value{0};
    worker.submit([&] { value = 7; }); worker.join();
    require(value == 7, "worker did not recover after delivered failure");
}
void early_representatives() {
    const solid::KvGeometry g{1, 2, 1, 2, 24, 64, solid::KvLayout::interleaved};
    const solid::SparseBudget budget{6, 1, 2, 1};
    solid::RepresentativeIndex serial(g,4,budget), early(g,4,budget);
    for (std::uint64_t p = 0; p < 20; ++p) {
        const std::vector<float> q{p % 3 ? 1.0f : -1.0f, 0.5f, -1.0f, 0.25f};
        const std::vector<solid::Half> k{solid::to_half(static_cast<float>(p)), solid::to_half(-static_cast<float>(p))};
        serial.observe(0,p,q,k); early.observe_query(0,p,q);
        rejects([&] { early.observe_query(0,p,q); });
        const auto a = solid::select_blocks(g,budget,serial,0,p,q), b = solid::select_blocks(g,budget,early,0,p,q);
        for (std::size_t h = 0; h < a.size(); ++h) require(a[h].positions == b[h].positions, "query-only selection changed representative timing or chosen tokens");
        early.append_keys(0,p,k);
    }
}
void pending_device_reader() {
    solid::SlotTable table(1,1);
    const auto lease = *table.acquire({0,0,0,1});
    table.pin(lease,true);
    require(table.entry(0).producer && table.entry(0).readers == 1, "deferred attention did not retain producer and reader");
    table.unpin(lease);
    require(table.entry(0).state == solid::SlotState::loading, "reader rollback published unfinished upload");
    table.pin(lease,true); table.retire(lease);
    require(!table.acquire({0,0,1,1}), "device slot reused before upload completion");
    table.complete(lease);
    require(!table.acquire({0,0,1,1}), "device slot reused before attention release");
    table.unpin(lease);
    require(table.acquire({0,0,1,1}).has_value(), "released device slot remained unavailable");
}
}
int main() {
    try {
        algorithm_one_order(); dependency_and_priority(); worker_overlap_and_error(); early_representatives(); pending_device_reader();
        std::cout << "DAG priority, shared fences, worker overlap, early representatives and deferred leases passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
