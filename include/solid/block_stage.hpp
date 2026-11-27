// 单层历史块的读盘→上传状态与优先级；固定依赖形状使用紧凑状态表。
#pragma once
#include "solid/async_reader.hpp"
#include "solid/block_identity.hpp"
#include "solid/scheduler.hpp"
#include <utility>

namespace solid {
struct BlockWork {
    BlockIdentity identity;
    ReadPriority priority=ReadPriority::correction_deterministic;
    std::uint64_t read_ns=1,copy_ns=1;
};
struct BlockStageStats {
    std::uint64_t tasks=0,read_ns=0,copy_ns=0,read_samples=0,copy_samples=0,critical_path_ns=0;
};
struct BlockDispatch { ReadDispatch read,copy; };
// Algorithm 1 for the fixed parallel READ -> H2D -> join graph.
// 为 READ→H2D→汇合的固定形状计算关键路径与 LST，返回读盘和上传优先级。
std::vector<BlockDispatch> plan_block_dispatch(const std::vector<BlockWork> & blocks);
// One bounded historical working set. The coordinator owns all transitions;
// read dispatch keys can be passed to the independent I/O owner by value.
class BlockStage {
public:
    BlockStage(std::vector<BlockWork> blocks,ScheduleTrace * trace=nullptr,std::uint64_t frame=0,
               std::uint32_t layer=0,std::uint64_t position=0,ScheduleMode mode=ScheduleMode::overlap);
    // 返回按关键路径和 LST 排序的待读块。
    std::vector<BlockIdentity> read_order() const;
    // 仅返回读盘已完成、上传尚未开始的块，按上传优先级排列。
    std::vector<BlockIdentity> copy_order() const;
    ReadDispatch dispatch(const BlockIdentity & identity) const;
    ReadDispatch copy_dispatch(const BlockIdentity & identity) const;
    void read_started(const BlockIdentity & identity,const std::string & proof={});
    void read_complete(const BlockIdentity & identity,const std::string & proof={});
    void copy_started(const BlockIdentity & identity);
    // 确认上传完成；有设备计时时用 device_ns 统计实际搬运时长。
    void copy_complete(const BlockIdentity & identity,const std::string & proof={},std::optional<std::uint64_t> device_ns={});
    bool finished() const;
    void abort();
    BlockStageStats stats() const;
    TaskRecord read_task(const BlockIdentity & identity) const;
    TaskRecord copy_task(const BlockIdentity & identity) const;
private:
    struct Progress {
        TaskState read_state=TaskState::pending,copy_state=TaskState::pending;
        std::uint64_t read_start=0,read_end=0,copy_start=0,copy_end=0;
        std::optional<std::uint64_t> copy_device_ns;
        bool observed_read=false,resident_copy=false,copy_ready=false;
    };
    void start_read(std::uint32_t index,std::uint64_t time,const std::string & proof);
    TaskRecord record(std::uint32_t index,bool copy) const;
    std::uint64_t now() const;
    std::uint32_t index(const BlockIdentity & identity) const;
    std::vector<BlockWork> blocks_;
    std::vector<std::pair<BlockIdentity,std::uint32_t>> indices_;
    std::unique_ptr<TaskDag> diagnostic_dag_;
    ScheduleTrace * trace_;
    std::vector<BlockDispatch> dispatch_;
    std::vector<Progress> progress_;
    std::vector<std::uint32_t> copy_order_;
    bool aborted_=false;
    std::size_t copies_remaining_=0,ready_count_=0;
};
} // namespace solid
