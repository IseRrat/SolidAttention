#include "solid/block_stage.hpp"
#include <chrono>
#include <stdexcept>

namespace solid {
BlockStage::BlockStage(std::vector<BlockWork> blocks,ScheduleTrace * trace,std::uint64_t frame,
                       std::uint32_t layer,std::uint64_t position,ScheduleMode mode):
    blocks_(std::move(blocks)),trace_(trace),observed_reads_(blocks_.size()),resident_copies_(blocks_.size()) {
    if (blocks_.size()>(UINT32_MAX-1)/2) throw std::invalid_argument("block stage exceeds task capacity");
    std::vector<TaskSpec> tasks;
    std::vector<std::uint32_t> copies;
    tasks.reserve(2*blocks_.size()+1);
    copies.reserve(blocks_.size());
    const bool names=trace && trace->enabled();
    for (std::uint32_t i=0;i<blocks_.size();++i) {
        const auto & block=blocks_[i];
        if (block.identity.layer!=layer || !block.identity.version || !indices_.emplace(block.identity,i).second) throw std::invalid_argument("invalid block stage identity");
        const auto suffix=names?"_h"+std::to_string(block.identity.head)+"_b"+std::to_string(block.identity.block)+"_v"+std::to_string(block.identity.version):std::string{};
        tasks.push_back({"block_read"+suffix,TaskResource::ssd,{},block.read_ns});
        tasks.push_back({"block_h2d"+suffix,TaskResource::h2d,{2*i},block.copy_ns});
        copies.push_back(2*i+1);
    }
    tasks.push_back({"historical_blocks_ready",TaskResource::host,std::move(copies),1});
    dag_=std::make_unique<TaskDag>(std::move(tasks),trace,frame,layer,position,mode);
    copy_priorities_.reserve(blocks_.size());
    for (std::uint32_t i=0;i<blocks_.size();++i) {
        const auto task=dag_->snapshot(2*i+1);
        // All copy tasks share a resource. Use the DAG's critical/LST order
        // with the task index as the final tie-break.
        copy_priorities_.emplace_back(!task.critical,task.latest_start,i);
    }
    copies_remaining_=blocks_.size();
    if (blocks_.empty()) dag_->fence({0},now(),"empty_history");
}
std::uint64_t BlockStage::now() const {
    return trace_?trace_->now():std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::uint32_t BlockStage::index(const BlockIdentity & identity) const {
    const auto found=indices_.find(identity);
    if (found==indices_.end()) throw std::logic_error("block event outside planned stage");
    return found->second;
}
std::vector<BlockIdentity> BlockStage::read_order() const {
    std::vector<std::uint32_t> candidates;
    for (std::uint32_t i=0;i<blocks_.size();++i) candidates.push_back(2*i);
    std::vector<BlockIdentity> result;
    for (const auto task:dag_->ready(candidates)) result.push_back(blocks_[task/2].identity);
    return result;
}
std::vector<BlockIdentity> BlockStage::copy_order() const {
    std::vector<BlockIdentity> result;
    result.reserve(ready_copies_.size());
    for (const auto & item:ready_copies_) result.push_back(blocks_[std::get<2>(item)].identity);
    return result;
}
ReadDispatch BlockStage::dispatch(const BlockIdentity & id) const {
    const auto i=index(id); const auto task=dag_->snapshot(2*i);
    return {blocks_[i].priority,task.critical,task.latest_start};
}
ReadDispatch BlockStage::copy_dispatch(const BlockIdentity & id) const {
    const auto i = index(id);
    const auto task = dag_->snapshot(2*i+1);
    return {blocks_[i].priority,task.critical,task.latest_start};
}
void BlockStage::read_started(const BlockIdentity & id,const std::string & proof) {
    const auto i=index(id);
    if (aborted_) return;
    const auto state=dag_->snapshot(2*i).state;
    if (state==TaskState::pending) { observed_reads_[i]=!proof.empty(); dag_->start(2*i,now(),proof); }
    else if (state!=TaskState::running) throw std::logic_error("read restarted after stage completion");
}
void BlockStage::read_complete(const BlockIdentity & id,const std::string & proof) {
    const auto i=index(id);
    if (aborted_) return;
    read_started(id,proof); dag_->complete(2*i,now(),proof);
    ready_copies_.insert(copy_priorities_[i]);
}
void BlockStage::copy_started(const BlockIdentity & id) {
    if (aborted_) throw std::logic_error("new upload after block stage abort");
    const auto i=index(id);
    dag_->start(2*i+1,now());
    ready_copies_.erase(copy_priorities_[i]);
}
void BlockStage::copy_complete(const BlockIdentity & id,const std::string & proof,std::optional<std::uint64_t> device_ns) {
    const auto i=index(id);
    if (aborted_) return;
    resident_copies_[i]=proof=="device_resident";
    dag_->complete(2*i+1,now(),proof,device_ns);
    const auto join=static_cast<std::uint32_t>(2*blocks_.size());
    if (--copies_remaining_==0) dag_->fence({join},now(),"all_block_copies_complete");
}
bool BlockStage::finished() const { return dag_->finished(); }
void BlockStage::abort() { if (!aborted_) { aborted_=true; ready_copies_.clear(); dag_->abort(now()); } }
TaskRecord BlockStage::read_task(const BlockIdentity & id) const { return dag_->task(2*index(id)); }
TaskRecord BlockStage::copy_task(const BlockIdentity & id) const { return dag_->task(2*index(id)+1); }
BlockStageStats BlockStage::stats() const {
    if (!finished()) throw std::logic_error("block stage metrics requested before all copies complete");
    BlockStageStats result; result.tasks=dag_->size(); result.critical_path_ns=dag_->critical_path_ns();
    for (std::uint32_t i=0;i<blocks_.size();++i) {
        const auto read=dag_->snapshot(2*i),copy=dag_->snapshot(2*i+1);
        if (!observed_reads_[i]) { result.read_ns+=read.duration_ns(); ++result.read_samples; }
        if (!resident_copies_[i]) { result.copy_ns+=copy.duration_ns(); ++result.copy_samples; }
    }
    return result;
}
} // namespace solid
