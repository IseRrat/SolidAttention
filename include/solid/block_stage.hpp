#pragma once
#include "solid/async_reader.hpp"
#include "solid/block_identity.hpp"
#include "solid/scheduler.hpp"
#include <map>

namespace solid {
struct BlockWork {
    BlockIdentity identity;
    ReadPriority priority=ReadPriority::correction_deterministic;
    std::uint64_t read_ns=1,copy_ns=1;
};
struct BlockStageStats {
    std::uint64_t tasks=0,read_ns=0,copy_ns=0,read_samples=0,copy_samples=0,critical_path_ns=0;
};
// One bounded historical working set. The coordinator owns all transitions;
// read dispatch keys can be passed to the independent I/O owner by value.
class BlockStage {
public:
    BlockStage(std::vector<BlockWork> blocks,ScheduleTrace * trace=nullptr,std::uint64_t frame=0,
               std::uint32_t layer=0,std::uint64_t position=0,ScheduleMode mode=ScheduleMode::overlap);
    std::vector<BlockIdentity> read_order() const;
    std::vector<BlockIdentity> copy_order() const;
    ReadDispatch dispatch(const BlockIdentity & identity) const;
    void read_started(const BlockIdentity & identity,const std::string & proof={});
    void read_complete(const BlockIdentity & identity,const std::string & proof={});
    void copy_started(const BlockIdentity & identity);
    void copy_complete(const BlockIdentity & identity,const std::string & proof={});
    bool finished() const;
    void abort();
    BlockStageStats stats() const;
    TaskRecord read_task(const BlockIdentity & identity) const;
    TaskRecord copy_task(const BlockIdentity & identity) const;
private:
    std::uint64_t now() const;
    std::uint32_t index(const BlockIdentity & identity) const;
    std::vector<BlockWork> blocks_;
    std::map<BlockIdentity,std::uint32_t> indices_;
    std::unique_ptr<TaskDag> dag_;
    ScheduleTrace * trace_;
    std::vector<bool> observed_reads_,resident_copies_;
    bool aborted_=false;
    std::size_t copies_remaining_=0;
};
} // namespace solid
