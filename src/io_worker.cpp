#include "solid/async_reader.hpp"
#include "solid/thread_affinity.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#ifdef __linux__
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#endif

namespace solid {
#ifdef __linux__
namespace {
class IoWorker final:public AsyncReader {
    enum class Phase { queued, active, complete };
    struct Request {
        int fd; void * destination; std::size_t bytes; std::uint64_t offset,sequence;
        Phase phase=Phase::queued;
        bool cancelled=false,cancel_sent=false;
        ReadPriority priority=ReadPriority::deterministic;
    };
public:
    IoWorker(std::function<std::unique_ptr<AsyncReader>()> factory,std::uint32_t depth,int cpu):
        factory_(std::move(factory)),depth_(depth),cpu_(cpu) {
        if (!factory_ || depth<2 || depth>16384 || cpu < -1) throw std::invalid_argument("invalid I/O worker settings");
        wake_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
        if (wake_<0) throw std::system_error(errno,std::generic_category(),"create I/O worker wakeup");
        try {
            thread_=std::thread(&IoWorker::run,this);
            std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[&] { return ready_; });
            if (error_) { const auto error=error_; lock.unlock(); thread_.join(); std::rethrow_exception(error); }
        } catch (...) { if (thread_.joinable()) thread_.join(); close(wake_); throw; }
    }
    ~IoWorker() override {
        try { drain(); } catch (...) {}
        { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
        notify(); thread_.join(); close(wake_);
    }
    bool submit(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset) override {
        return submit_prioritized(id,fd,destination,bytes,offset,ReadPriority::deterministic);
    }
    bool submit_prioritized(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset,ReadPriority priority) override {
        {
            std::lock_guard<std::mutex> lock(mutex_); check();
            if (!id || id>=(std::uint64_t{1}<<63) || fd<0 || !destination || !bytes || bytes>INT_MAX ||
                offset>static_cast<std::uint64_t>(INT64_MAX)-bytes || requests_.count(id)) throw std::invalid_argument("invalid queued I/O request");
            if (draining_) throw std::logic_error("read submitted during I/O drain");
            if (requests_.size()>=depth_) return false;
            if (sequence_==UINT64_MAX) throw std::overflow_error("I/O submission sequence exhausted");
            auto request=Request{fd,destination,bytes,offset,++sequence_}; request.priority=priority;
            requests_.emplace(id,request);
        }
        notify(); return true;
    }
    void prioritize(std::uint64_t id,ReadPriority priority) override {
        { std::lock_guard<std::mutex> lock(mutex_); check(); const auto found=requests_.find(id);
          if (found!=requests_.end() && found->second.phase!=Phase::complete && priority<found->second.priority) found->second.priority=priority; }
        notify();
    }
    void cancel(std::uint64_t id) override {
        { std::lock_guard<std::mutex> lock(mutex_); check(); const auto found=requests_.find(id);
          if (found!=requests_.end() && found->second.phase!=Phase::complete) found->second.cancelled=true; }
        notify();
    }
    void flush() override { { std::lock_guard<std::mutex> lock(mutex_); check(); } notify(); }
    bool poll(ReadCompletion & result,bool wait) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (wait) cv_.wait(lock,[&] { return error_ || !completed_.empty() || requests_.empty(); });
        check();
        if (completed_.empty()) return false;
        result=completed_.front(); completed_.pop_front();
        if (requests_.erase(result.id)!=1) throw std::logic_error("I/O worker completion identity");
        return true;
    }
    void drain() override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!ended_) {
            draining_=true; const auto epoch=++drain_requested_;
            for (auto & item:requests_) item.second.cancelled=true;
            lock.unlock(); notify(); lock.lock();
            cv_.wait(lock,[&] { return ended_ || drain_completed_>=epoch; });
        }
        if (ended_ && !safe_) throw std::logic_error("I/O worker exited before buffer drain");
        requests_.clear(); completed_.clear(); draining_=false;
    }
    AsyncIoStats stats() const override { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
    void rethrow_failure() const override { std::lock_guard<std::mutex> lock(mutex_); if (error_) std::rethrow_exception(error_); }
private:
    void check() const { if (error_) std::rethrow_exception(error_); if (ended_ || stopping_) throw std::runtime_error("I/O worker stopped"); }
    void notify() noexcept {
        std::uint64_t value=1;
        for (;;) {
            const auto written=write(wake_,&value,sizeof(value));
            if (written==static_cast<ssize_t>(sizeof(value)) || (written<0 && errno==EAGAIN)) return;
            if (written<0 && errno==EINTR) continue;
            std::terminate();
        }
    }
    void clear_wakeup() {
        std::uint64_t value;
        for (;;) {
            const auto read_bytes=read(wake_,&value,sizeof(value));
            if (read_bytes==static_cast<ssize_t>(sizeof(value))) continue;
            if (read_bytes<0 && errno==EINTR) continue;
            if (read_bytes<0 && errno==EAGAIN) return;
            throw std::system_error(read_bytes<0?errno:EIO,std::generic_category(),"consume I/O command wakeup");
        }
    }
    void snapshot(AsyncReader & reader) {
        auto value=reader.stats();
        std::lock_guard<std::mutex> lock(mutex_);
        value.queued_cancellations=queued_cancellations_; value.worker_submissions=submissions_;
        value.worker_cpu=cpu_; value.dedicated_worker=true; stats_=value;
        stats_.worker_capacity=depth_;
    }
    void run() noexcept {
        std::unique_ptr<AsyncReader> reader;
        try {
            bind_thread_cpu(cpu_); reader=factory_();
            if (!reader || reader->stats().dedicated_worker) throw std::invalid_argument("I/O worker requires a direct reader");
            snapshot(*reader);
            { std::lock_guard<std::mutex> lock(mutex_); ready_=true; }
            cv_.notify_all();
            for (;;) {
                // Consume notification counters BEFORE inspecting their queues.
                // A completion/command arriving after the queue check keeps a
                // pending counter and wakes poll(), including cancellation CQEs.
                clear_wakeup(); reader->acknowledge_notifications();
                std::vector<std::uint64_t> cancel;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (auto & item:requests_) {
                        auto & request=item.second;
                        if (request.phase==Phase::queued && request.cancelled) {
                            request.phase=Phase::complete; ++queued_cancellations_;
                            completed_.push_back({item.first,-ECANCELED});
                        } else if (request.phase==Phase::active && request.cancelled && !request.cancel_sent) {
                            request.cancel_sent=true; cancel.push_back(item.first);
                        }
                    }
                }
                for (const auto id:cancel) reader->cancel(id);
                for (;;) {
                    std::uint64_t id=0; Request request{};
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        auto best=requests_.end();
                        for (auto it=requests_.begin();it!=requests_.end();++it) if (it->second.phase==Phase::queued && !it->second.cancelled &&
                            (best==requests_.end() || std::tie(it->second.priority,it->second.sequence)<
                                                     std::tie(best->second.priority,best->second.sequence))) best=it;
                        if (best==requests_.end()) break;
                        id=best->first; request=best->second; best->second.phase=Phase::active;
                    }
                    if (!reader->submit(id,request.fd,request.destination,request.bytes,request.offset)) {
                        std::lock_guard<std::mutex> lock(mutex_); requests_.at(id).phase=Phase::queued; break;
                    }
                    ++submissions_;
                }
                reader->flush();
                ReadCompletion completion{};
                bool delivered=false;
                while (reader->poll(completion,false)) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    const auto found=requests_.find(completion.id);
                    if (found==requests_.end() || found->second.phase!=Phase::active) throw std::logic_error("unknown I/O worker READ completion");
                    found->second.phase=Phase::complete; completed_.push_back(completion);
                    delivered=true;
                }
                snapshot(*reader); cv_.notify_all();
                bool quiesce=false,stop=false,queued=false;
                std::uint64_t epoch=0;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    quiesce=drain_requested_>drain_completed_;
                    for (const auto & item:requests_) {
                        if (item.second.phase!=Phase::complete) quiesce=false;
                        if (item.second.phase==Phase::queued) queued=true;
                    }
                    epoch=drain_requested_; stop=stopping_;
                }
                if (quiesce) {
                    reader->drain(); snapshot(*reader);
                    { std::lock_guard<std::mutex> lock(mutex_); drain_completed_=epoch; }
                    cv_.notify_all();
                }
                if (stop) { reader->drain(); break; }
                if (delivered && queued) continue;
                pollfd descriptors[2]={{wake_,POLLIN,0},{reader->completion_handle(),POLLIN,0}};
                int status;
                do { status=::poll(descriptors,descriptors[1].fd<0?1:2,descriptors[1].fd<0?1:-1); } while (status<0 && errno==EINTR);
                if (status<0) throw std::system_error(errno,std::generic_category(),"wait for I/O completion or command");
                for (const auto & item:descriptors) if (item.revents&(POLLERR|POLLHUP|POLLNVAL)) throw std::runtime_error("I/O readiness descriptor failed");
            }
            { std::lock_guard<std::mutex> lock(mutex_); ended_=true; safe_=true; }
            cv_.notify_all();
        } catch (...) {
            const auto error=std::current_exception();
            // An engine exception never grants permission to recycle a buffer.
            // First confirm all kernel writers have stopped, then report it.
            try { if (reader) reader->drain(); } catch (...) { std::terminate(); }
            { std::lock_guard<std::mutex> lock(mutex_); error_=error; ready_=true; ended_=true; safe_=true; }
            cv_.notify_all();
        }
    }
    std::function<std::unique_ptr<AsyncReader>()> factory_;
    std::uint32_t depth_;
    int cpu_,wake_=-1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::map<std::uint64_t,Request> requests_;
    std::deque<ReadCompletion> completed_;
    std::exception_ptr error_;
    bool ready_=false,ended_=false,safe_=false,stopping_=false,draining_=false;
    std::uint64_t sequence_=0,drain_requested_=0,drain_completed_=0,queued_cancellations_=0,submissions_=0;
    AsyncIoStats stats_;
};
} // namespace
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()> factory,std::uint32_t depth,int cpu) {
    return std::make_unique<IoWorker>(std::move(factory),depth,cpu);
}
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t depth,int cpu,std::uint32_t capacity) {
    return make_io_worker([depth] { return make_uring_reader(depth); },std::max(depth,capacity),cpu);
}
#else
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()>,std::uint32_t,int) { throw std::runtime_error("I/O worker requires Linux"); }
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t,int,std::uint32_t) { throw std::runtime_error("I/O worker requires Linux"); }
#endif
} // namespace solid
