// 计算块级关键路径与最晚开始时间，跟踪 SSD 读取和 H2D 完成状态。
#include "solid/block_stage.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

namespace solid {
namespace {
std::uint64_t sum(std::uint64_t a,std::uint64_t b) {
    if (b>UINT64_MAX-a) throw std::overflow_error("block DAG duration overflow");
    return a+b;
}
}
std::vector<BlockDispatch> plan_block_dispatch(const std::vector<BlockWork> & blocks) {
    if (blocks.size()>(UINT32_MAX-1)/2) throw std::invalid_argument("block stage exceeds task capacity");
    std::uint64_t longest=0;
    for (const auto & block:blocks) {
        if (!block.read_ns || !block.copy_ns) throw std::invalid_argument("zero block task estimate");
        longest=std::max(longest,sum(block.read_ns,block.copy_ns));
    }
    (void) sum(longest,1); // Shared join: same unit estimate as TaskDag.
    std::vector<BlockDispatch> result;
    result.reserve(blocks.size());
    for (const auto & block:blocks) {
        const auto length=sum(block.read_ns,block.copy_ns);
        const bool critical=length==longest;
        result.push_back({{block.priority,critical,longest-length},
                          {block.priority,critical,longest-block.copy_ns}});
    }
    return result;
}
BlockStage::BlockStage(std::vector<BlockWork> blocks,ScheduleTrace * trace,std::uint64_t frame,
                       std::uint32_t layer,std::uint64_t position,ScheduleMode mode):
    blocks_(std::move(blocks)),trace_(trace && trace->enabled()?trace:nullptr),
    dispatch_(plan_block_dispatch(blocks_)),progress_(blocks_.size()),copy_order_(blocks_.size()),
    copies_remaining_(blocks_.size()) {
    indices_.reserve(blocks_.size());
    for (std::uint32_t i=0;i<blocks_.size();++i) {
        const auto & block=blocks_[i];
        if (block.identity.layer!=layer || !block.identity.version) throw std::invalid_argument("invalid block stage identity");
        indices_.emplace_back(block.identity,i);
    }
    std::sort(indices_.begin(),indices_.end(),[](const auto & a,const auto & b) { return a.first<b.first; });
    for (std::size_t i=1;i<indices_.size();++i)
        if (indices_[i-1].first==indices_[i].first) throw std::invalid_argument("duplicate block stage identity");
    std::iota(copy_order_.begin(),copy_order_.end(),0);
    std::sort(copy_order_.begin(),copy_order_.end(),[&](auto a,auto b) {
        const auto & x=dispatch_[a].copy; const auto & y=dispatch_[b].copy;
        if (x.critical!=y.critical) return x.critical;
        return x.latest_start==y.latest_start?a<b:x.latest_start<y.latest_start;
    });
    if (trace_) {
        std::vector<TaskSpec> tasks;
        std::vector<std::uint32_t> copies;
        tasks.reserve(2*blocks_.size()+1); copies.reserve(blocks_.size());
        for (std::uint32_t i=0;i<blocks_.size();++i) {
            const auto & block=blocks_[i];
            const auto suffix="_h"+std::to_string(block.identity.head)+"_b"+std::to_string(block.identity.block)+"_v"+std::to_string(block.identity.version);
            tasks.push_back({"block_read"+suffix,TaskResource::ssd,{},block.read_ns});
            tasks.push_back({"block_h2d"+suffix,TaskResource::h2d,{2*i},block.copy_ns});
            copies.push_back(2*i+1);
        }
        tasks.push_back({"historical_blocks_ready",TaskResource::host,std::move(copies),1});
        diagnostic_dag_=std::make_unique<TaskDag>(std::move(tasks),trace_,frame,layer,position,mode);
        if (blocks_.empty()) diagnostic_dag_->fence({0},now(),"empty_history");
    }
}
std::uint64_t BlockStage::now() const {
    return trace_?trace_->now():std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::uint32_t BlockStage::index(const BlockIdentity & identity) const {
    const auto found=std::lower_bound(indices_.begin(),indices_.end(),identity,
        [](const auto & item,const auto & id) { return item.first<id; });
    if (found==indices_.end() || !(found->first==identity)) throw std::logic_error("block event outside planned stage");
    return found->second;
}
std::vector<BlockIdentity> BlockStage::read_order() const {
    std::vector<std::uint32_t> order;
    order.reserve(blocks_.size());
    for (std::uint32_t i=0;i<blocks_.size();++i) if (progress_[i].read_state==TaskState::pending) order.push_back(i);
    std::sort(order.begin(),order.end(),[&](auto a,auto b) {
        const auto & x=dispatch_[a].read; const auto & y=dispatch_[b].read;
        if (x.critical!=y.critical) return x.critical;
        return x.latest_start==y.latest_start?a<b:x.latest_start<y.latest_start;
    });
    std::vector<BlockIdentity> result;
    result.reserve(order.size());
    for (const auto i:order) result.push_back(blocks_[i].identity);
    return result;
}
std::vector<BlockIdentity> BlockStage::copy_order() const {
    std::vector<BlockIdentity> result;
    result.reserve(ready_count_);
    for (const auto i:copy_order_) if (progress_[i].copy_ready) result.push_back(blocks_[i].identity);
    return result;
}
ReadDispatch BlockStage::dispatch(const BlockIdentity & id) const { return dispatch_[index(id)].read; }
ReadDispatch BlockStage::copy_dispatch(const BlockIdentity & id) const { return dispatch_[index(id)].copy; }
void BlockStage::start_read(std::uint32_t i,std::uint64_t time,const std::string & proof) {
    auto & p=progress_[i];
    if (p.read_state==TaskState::pending) {
        if (diagnostic_dag_) diagnostic_dag_->start(2*i,time,proof);
        p.observed_read=!proof.empty(); p.read_start=time; p.read_state=TaskState::running;
    } else if (p.read_state!=TaskState::running) throw std::logic_error("read restarted after stage completion");
}
void BlockStage::read_started(const BlockIdentity & id,const std::string & proof) {
    if (!aborted_) start_read(index(id),now(),proof);
}
void BlockStage::read_complete(const BlockIdentity & id,const std::string & proof) {
    if (aborted_) return;
    const auto i=index(id); const auto time=now(); start_read(i,time,proof);
    if (diagnostic_dag_) diagnostic_dag_->complete(2*i,time,proof);
    progress_[i].read_end=time; progress_[i].read_state=TaskState::complete;
    progress_[i].copy_ready=true; ++ready_count_;
}
void BlockStage::copy_started(const BlockIdentity & id) {
    if (aborted_) throw std::logic_error("new upload after block stage abort");
    const auto i=index(id); auto & p=progress_[i];
    if (!p.copy_ready || p.copy_state!=TaskState::pending) throw std::logic_error("copy started before read or after submission");
    const auto time=now();
    if (diagnostic_dag_) diagnostic_dag_->start(2*i+1,time);
    p.copy_start=time; p.copy_state=TaskState::running; p.copy_ready=false; --ready_count_;
}
void BlockStage::copy_complete(const BlockIdentity & id,const std::string & proof,std::optional<std::uint64_t> device_ns) {
    if (aborted_) return;
    const auto i=index(id); auto & p=progress_[i];
    if (p.copy_state!=TaskState::running) throw std::logic_error("copy completed before submission or more than once");
    const auto time=now();
    if (diagnostic_dag_) diagnostic_dag_->complete(2*i+1,time,proof,device_ns);
    p.resident_copy=proof=="device_resident"; p.copy_end=time; p.copy_device_ns=device_ns; p.copy_state=TaskState::complete;
    if (--copies_remaining_==0 && diagnostic_dag_)
        diagnostic_dag_->fence({static_cast<std::uint32_t>(2*blocks_.size())},time,"all_block_copies_complete");
}
bool BlockStage::finished() const { return !copies_remaining_; }
void BlockStage::abort() {
    if (aborted_) return;
    aborted_=true; ready_count_=0;
    for (auto & p:progress_) {
        p.copy_ready=false;
        if (p.read_state!=TaskState::complete) p.read_state=TaskState::failed;
        if (p.copy_state!=TaskState::complete) p.copy_state=TaskState::failed;
    }
    if (diagnostic_dag_) diagnostic_dag_->abort(now());
}
TaskRecord BlockStage::record(std::uint32_t i,bool copy) const {
    if (diagnostic_dag_) return diagnostic_dag_->task(2*i+copy);
    const auto & p=progress_[i]; const auto & order=copy?dispatch_[i].copy:dispatch_[i].read;
    const auto & b=blocks_[i];
    return {{copy?"block_h2d":"block_read",copy?TaskResource::h2d:TaskResource::ssd,
        copy?std::vector<std::uint32_t>{2*i}:std::vector<std::uint32_t>{},copy?b.copy_ns:b.read_ns},
        copy?p.copy_state:p.read_state,copy?p.copy_start:p.read_start,copy?p.copy_end:p.read_end,
        order.latest_start,order.critical,copy?p.copy_device_ns:std::optional<std::uint64_t>{}};
}
TaskRecord BlockStage::read_task(const BlockIdentity & id) const { return record(index(id),false); }
TaskRecord BlockStage::copy_task(const BlockIdentity & id) const { return record(index(id),true); }
BlockStageStats BlockStage::stats() const {
    if (!finished()) throw std::logic_error("block stage metrics requested before all copies complete");
    BlockStageStats result; result.tasks=2*blocks_.size()+1;
    for (const auto & p:progress_) {
        const auto read=p.read_end-p.read_start,copy=p.copy_device_ns.value_or(p.copy_end-p.copy_start);
        result.critical_path_ns=std::max(result.critical_path_ns,sum(read,copy));
        if (!p.observed_read) { result.read_ns+=read; ++result.read_samples; }
        if (!p.resident_copy) { result.copy_ns+=copy; ++result.copy_samples; }
    }
    return result;
}
} // namespace solid
