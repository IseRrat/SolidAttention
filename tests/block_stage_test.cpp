// 块级读盘/上传状态和优先级检查，包含固定形状实现与通用 DAG 的对照。
#include "prefetch_fakes.hpp"
#include "solid/block_stage.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>

using namespace test_prefetch;
namespace {
void fixed_graph_matches_generic_dag() {
    std::mt19937 random(2026);
    for (const auto count:{0u,1u,2u,8u,32u,256u}) for (int trial=0;trial<8;++trial) {
        std::vector<solid::BlockWork> blocks;
        std::vector<solid::TaskSpec> specs;
        std::vector<std::uint32_t> reads,copies;
        for (std::uint32_t i=0;i<count;++i) {
            blocks.push_back({{0,i%4,i/4,1},i%2?solid::ReadPriority::selected:solid::ReadPriority::deterministic,
                1+random()%9,1+random()%9});
            specs.push_back({"read",solid::TaskResource::ssd,{},blocks.back().read_ns});
            specs.push_back({"copy",solid::TaskResource::h2d,{2*i},blocks.back().copy_ns});
            reads.push_back(2*i); copies.push_back(2*i+1);
        }
        specs.push_back({"join",solid::TaskResource::host,copies,1});
        solid::TaskDag reference(specs);
        solid::BlockStage stage(blocks);
        const auto plan=solid::plan_block_dispatch(blocks);
        if (!count) { require(stage.finished() && stage.stats().tasks==1,"empty fixed DAG"); continue; }
        for (std::uint32_t i=0;i<count;++i) {
            const auto read=reference.snapshot(2*i),copy=reference.snapshot(2*i+1);
            require(plan[i].read.critical==read.critical && plan[i].read.latest_start==read.latest_start &&
                    plan[i].copy.critical==copy.critical && plan[i].copy.latest_start==copy.latest_start &&
                    plan[i].read.priority==blocks[i].priority,"fixed DAG priority differs from Algorithm 1 traversal");
        }
        const auto identities=[&](const auto & tasks) {
            std::vector<solid::BlockIdentity> result;
            for (const auto task:tasks) result.push_back(blocks[task/2].identity);
            return result;
        };
        require(stage.read_order()==identities(reference.ready(reads)),"fixed read order changed critical/LST/ties");
        std::vector<std::uint32_t> arrival(count); std::iota(arrival.begin(),arrival.end(),0);
        std::shuffle(arrival.begin(),arrival.end(),random);
        std::uint64_t stamp=1;
        for (const auto i:arrival) {
            stage.read_complete(blocks[i].identity);
            reference.start(2*i,stamp++); reference.complete(2*i,stamp++);
            auto ready=reference.ready(copies);
            require(stage.copy_order()==identities(ready),"out-of-order reads changed ready copy order");
            if (i%3==0) for (const auto task:ready) {
                stage.copy_started(blocks[task/2].identity); stage.copy_complete(blocks[task/2].identity,"copy_ticket_complete",2);
                reference.start(task,stamp++); reference.complete(task,stamp++,"copy_ticket_complete",2);
            }
        }
        for (const auto task:reference.ready(copies)) {
            stage.copy_started(blocks[task/2].identity); stage.copy_complete(blocks[task/2].identity,"copy_ticket_complete",3);
            reference.start(task,stamp++); reference.complete(task,stamp++,"copy_ticket_complete",3);
        }
        require(stage.finished() && stage.copy_order().empty(),"fixed DAG join did not wait for all copies");
        solid::TaskDag measured(specs); std::uint64_t last=0;
        for (const auto & block:blocks) {
            const auto read=stage.read_task(block.identity),copy=stage.copy_task(block.identity);
            const auto i=static_cast<std::uint32_t>(&block-blocks.data());
            measured.start(2*i,read.start_ns); measured.complete(2*i,read.end_ns);
            measured.start(2*i+1,copy.start_ns); measured.complete(2*i+1,copy.end_ns,"copy_ticket_complete",copy.device_ns);
            last=std::max(last,copy.end_ns);
        }
        measured.fence({2*count},last,"complete");
        require(stage.stats().critical_path_ns==measured.critical_path_ns() && stage.stats().read_samples==count &&
                stage.stats().copy_samples==count,"fixed DAG timing differs from general dependency traversal");
    }
}
void trace_independent_priorities() {
    const solid::BlockIdentity a{0,0,1,32}, b{0,1,2,32};
    const std::vector<solid::BlockWork> blocks{{a,solid::ReadPriority::selected,10,2},{b,solid::ReadPriority::selected,2,20}};
    const auto path=std::filesystem::temp_directory_path()/("solid-block-trace-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".jsonl");
    {
        solid::ScheduleTrace enabled(path), disabled;
        require(enabled.enabled() && !disabled.enabled(),"trace open state");
        solid::BlockStage logged(blocks,&enabled), normal(blocks,&disabled);
        require(logged.read_order()==normal.read_order(),"disabling block labels changed scheduling order");
        for (const auto & id:{a,b}) {
            const auto x=logged.dispatch(id),y=normal.dispatch(id);
            require(x.critical==y.critical && x.latest_start==y.latest_start,"trace switch changed dispatch priority");
        }
        require(logged.read_task(a).spec.name=="block_read_h0_b1_v32" &&
                normal.read_task(a).spec.name=="block_read","optional detailed block labels");
        enabled.flush();
    }
    std::ifstream input(path); std::string contents((std::istreambuf_iterator<char>(input)),{}); input.close();
    std::filesystem::remove(path);
    require(contents.find("block_read_h0_b1_v32")!=std::string::npos,"enabled trace lost detailed block identity");
}
void dependencies(bool prediction) {
    const solid::BlockIdentity a{0,0,1,1},b{0,0,2,1},c{0,1,3,1};
    const auto selected=prediction?solid::ReadPriority::selected:solid::ReadPriority::correction_selected;
    const auto deterministic=prediction?solid::ReadPriority::deterministic:solid::ReadPriority::correction_deterministic;
    solid::BlockStage stage({{a,selected,10,2}, {b,deterministic,2,20}, {c,selected,10,1}});
    require(stage.read_order()==std::vector<solid::BlockIdentity>{b,a,c},"block reads ignored critical path/LST");
    require(stage.dispatch(b).critical && stage.dispatch(b).latest_start==0 &&
            stage.dispatch(b).before(stage.dispatch(a)) && stage.dispatch(a).before(stage.dispatch(c)),"DAG dispatch keys changed order");
    require(stage.copy_order().empty(),"copy ready before original READ");
    rejects([&] { stage.copy_started(a); }); rejects([&] { stage.read_started({0,0,99,1}); });
    stage.read_started(a); stage.read_complete(a);
    require(stage.copy_order()==std::vector<solid::BlockIdentity>{a},"independent block failed to unlock H2D");
    stage.copy_started(a); stage.copy_complete(a);
    require(!stage.finished(),"join released with incomplete blocks");
    stage.read_complete(c,"host_resident"); stage.read_complete(b,"inflight_at_plan");
    require(stage.copy_order()==std::vector<solid::BlockIdentity>{b,c},"H2D ready queue ignored critical path/LST");
    require(stage.copy_dispatch(b).before(stage.copy_dispatch(c)),"H2D dispatch category hid a critical copy");
    stage.copy_started(b); stage.copy_started(c);
    stage.copy_complete(c,"device_resident"); require(!stage.finished(),"copy submission satisfied the join");
    stage.copy_complete(b,"copy_ticket_complete");
    require(stage.finished() && stage.stats().tasks==7 && stage.stats().read_samples==1 && stage.stats().copy_samples==2,"block receipts or observed-boundary costs");
    rejects([&] { stage.copy_complete(a); }); rejects([&] { stage.read_complete(a); });
    solid::BlockStage empty({}); require(empty.finished() && empty.stats().tasks==1,"empty historical gate");
    rejects([&] { solid::BlockStage duplicate({{a},{a}}); });
}
void real_queue(bool fail, bool admit_before_stage = false) {
    const solid::KvGeometry g{1,1,2,1,16,64,solid::KvLayout::interleaved};
    auto file=std::make_unique<File>(); auto reader=std::make_unique<Reader>(file.get(),admit_before_stage ? 2 : 1); auto * observe=reader.get();
    solid::FileKvStore store(g,std::move(file),8); fill(store,6);
    solid::Prefetcher host(store,{4,1,1,1},4,std::move(reader));
    Transport transport; solid::DeviceCache cache(host,g,2,transport); host.begin(0,6);
    std::shared_ptr<solid::BlockStage> stage;
    std::vector<solid::SlotLease> leases;
    std::size_t callbacks=0;
    const auto queued_before_selection = observe->submitted.size();
    const auto resolve=[&] {
        return host.resolve_history(0,6,{{{0,1,3,6},{}}},
            [&](const auto & plan) {
                if (admit_before_stage) require(observe->submitted.size() > queued_before_selection,
                    "correction read waited for the device stage");
                cache.begin_stage(plan);
            },
            [&](const solid::SlotLease & lease) { transport.fail_upload=fail && ++callbacks==2; cache.stage_ready(lease,stage); },
            [&](std::vector<solid::BlockWork> plan) {
                for (auto & block:plan) { block.read_ns=block.identity.block==3?100:2; block.copy_ns=2; }
                stage=std::make_shared<solid::BlockStage>(std::move(plan)); return stage;
            }, false, {}, admit_before_stage);
    };
    if (fail) rejects([&] { (void) resolve(); });
    else {
        leases=resolve(); require(!stage->finished(),"H2D task completed at submission");
        solid::KvBlock block; const auto expected=store.plan_read(0,0,3,block).front().file_offset;
        require(observe->submitted.size()==3 && observe->submitted[1].offset==expected,"actual constrained SSD submission ignored block DAG");
        for (const auto & lease:leases) host.unpin(lease);
        cache.wait_transfers(); require(stage->finished(),"copy receipt failed to unlock block join");
        require(stage->stats().tasks==7,"actual read/H2D graph task count");
    }
    cache.finish(); host.finish();
    if (fail) require(!stage->finished(),"failed block DAG reported successful completion");
}
void split_segments() {
    const solid::KvGeometry g{1,1,2,1,16,64,solid::KvLayout::separate};
    auto file=std::make_unique<File>(); auto reader=std::make_unique<Reader>(file.get(),1); auto * observe=reader.get();
    solid::FileKvStore store(g,std::move(file),8); fill(store,6);
    solid::Prefetcher host(store,{4,1,1,1},4,std::move(reader)); host.begin(0,6);
    std::shared_ptr<solid::BlockStage> stage;
    const auto leases=host.resolve_history(0,6,{{{0,1,3,6},{}}},{},[&](const solid::SlotLease & lease) {
        require(stage->read_task(lease.identity).state==solid::TaskState::complete,"partial K/V read unlocked H2D");
        verify(store,host,{lease}); stage->copy_started(lease.identity); stage->copy_complete(lease.identity,"cpu_copy_complete");
    },[&](const auto & plan) { stage=std::make_shared<solid::BlockStage>(plan); return stage; });
    require(stage->finished() && observe->stats().completed_reads==6,"two-segment block completion lost");
    for (const auto & lease:leases) host.unpin(lease);
    host.finish();
}
void failed_read_after_upload() {
    const solid::KvGeometry g{1,1,2,1,16,64,solid::KvLayout::interleaved};
    auto file=std::make_unique<File>(); auto reader=std::make_unique<Reader>(file.get(),1); auto * observe=reader.get();
    solid::FileKvStore store(g,std::move(file),8); fill(store,6);
    solid::Prefetcher host(store,{4,1,1,1},4,std::move(reader));
    Transport transport; solid::DeviceCache cache(host,g,2,transport); host.begin(0,6);
    observe->before_wait=[&] { if (!transport.copies.empty()) observe->fail_next=true; };
    std::shared_ptr<solid::BlockStage> stage;
    rejects([&] { (void) host.resolve_history(0,6,{{{0,1,3,6},{}}},
        [&](const auto & plan) { cache.begin_stage(plan); },
        [&](const solid::SlotLease & lease) { cache.stage_ready(lease,stage); },
        [&](const auto & plan) { stage=std::make_shared<solid::BlockStage>(plan); return stage; }); });
    require(stage && !transport.copies.empty(),"read failure did not overlap an owned upload source");
    stage->abort(); cache.finish(); host.finish();
    require(transport.copies.empty() && observe->pending.empty() && !stage->finished(),"read failure left live writers or successful block gate");
}
}
int main() {
    try { fixed_graph_matches_generic_dag(); trace_independent_priorities(); dependencies(false); dependencies(true); real_queue(false); real_queue(true); real_queue(false, true); split_segments(); failed_read_after_upload(); std::cout<<"block event DAG, critical/LST dispatch, real queue order and receipts passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
