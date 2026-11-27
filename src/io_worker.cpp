#include "solid/async_reader.hpp"
#include "solid/trace.hpp"
#include "solid/thread_affinity.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <set>
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
        int fd; const void * data; std::size_t bytes; std::uint64_t offset,sequence;
        Phase phase=Phase::queued;
        bool cancelled=false,cancel_sent=false,write=false;
        ReadDispatch dispatch{};
    };
    struct ReadyKey { ReadDispatch dispatch; std::uint64_t sequence,id; };
    struct ReadyBefore {
        bool operator()(const ReadyKey & a,const ReadyKey & b) const {
            if (a.dispatch.before(b.dispatch)) return true;
            if (b.dispatch.before(a.dispatch)) return false;
            return std::tie(a.sequence,a.id)<std::tie(b.sequence,b.id);
        }
    };
    static ReadyKey ready_key(std::uint64_t id,const Request & request) { return {request.dispatch,request.sequence,id}; }
    void queue_cancel(std::uint64_t id,Request & request) {
        if (request.phase!=Phase::complete && !request.cancelled) {
            pending_cancels_.push_back(id); request.cancelled=true;
            if (request.phase==Phase::queued) ready_requests_.erase(ready_key(id,request));
        }
    }
public:
    IoWorker(std::function<std::unique_ptr<AsyncReader>()> factory,std::uint32_t depth,int cpu,bool notify_each,bool measure_ready):
        factory_(std::move(factory)),depth_(depth),cpu_(cpu),notify_each_(notify_each),measure_ready_(measure_ready) {
        if (!factory_ || depth<2 || depth>16384 || cpu < -1) throw std::invalid_argument("invalid I/O worker settings");
        wake_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
        if (wake_<0) throw std::system_error(errno,std::generic_category(),"create I/O worker wakeup");
        results_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
        if (results_<0) { const auto error=errno; close(wake_); throw std::system_error(error,std::generic_category(),"create I/O result wakeup"); }
        try {
            thread_=std::thread(&IoWorker::run,this);
            std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[&] { return ready_; });
            if (error_) { const auto error=error_; lock.unlock(); thread_.join(); std::rethrow_exception(error); }
        } catch (...) { if (thread_.joinable()) thread_.join(); close(wake_); close(results_); throw; }
    }
    ~IoWorker() override {
        try { drain(); } catch (...) {}
        { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
        notify(); thread_.join(); close(wake_); close(results_);
    }
    bool submit(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset) override {
        return submit_prioritized(id,fd,destination,bytes,offset,ReadPriority::deterministic);
    }
    bool submit_prioritized(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset,ReadPriority priority) override {
        return submit_scheduled(id,fd,destination,bytes,offset,{priority});
    }
    bool submit_scheduled(std::uint64_t id,int fd,void * destination,std::size_t bytes,std::uint64_t offset,ReadDispatch dispatch) override {
        return admit(id, fd, destination, bytes, offset, dispatch, false);
    }
    bool supports_write() const override { return writes_supported_; }
    bool submit_write(std::uint64_t id,int fd,const void * source,std::size_t bytes,std::uint64_t offset) override {
        if (!supports_write()) throw std::logic_error("I/O worker write support is disabled");
        return admit(id, fd, source, bytes, offset, {ReadPriority::writeback}, true);
    }
    bool admit(std::uint64_t id,int fd,const void * data,std::size_t bytes,std::uint64_t offset,ReadDispatch dispatch,bool write) {
        {
            std::lock_guard<std::mutex> lock(mutex_); check();
            if (!id || id>=(std::uint64_t{1}<<63) || fd<0 || !data || !bytes || bytes>INT_MAX ||
                offset>static_cast<std::uint64_t>(INT64_MAX)-bytes || requests_.count(id)) throw std::invalid_argument("invalid queued I/O request");
            if (draining_) throw std::logic_error("read submitted during I/O drain");
            if (requests_.size() >= depth_ - (write ? 1 : 0)) return false;
            if (sequence_==UINT64_MAX) throw std::overflow_error("I/O submission sequence exhausted");
            auto request=Request{fd,data,bytes,offset,++sequence_}; request.dispatch=dispatch; request.write=write;
            requests_.emplace(id,request);
            try { ready_requests_.insert(ready_key(id,request)); }
            catch (...) { requests_.erase(id); throw; }
        }
        notify(); return true;
    }
    void prioritize(std::uint64_t id,ReadPriority priority) override {
        schedule(id,{priority});
    }
    void schedule(std::uint64_t id,ReadDispatch dispatch) override {
        { std::lock_guard<std::mutex> lock(mutex_); check(); const auto found=requests_.find(id);
          if (found!=requests_.end() && found->second.phase!=Phase::complete && dispatch.before(found->second.dispatch)) {
              auto & request=found->second;
              if (request.phase==Phase::queued && !request.cancelled) {
                  auto changed=ready_key(id,request); changed.dispatch=dispatch;
                  ready_requests_.insert(changed);
                  ready_requests_.erase(ready_key(id,request));
              }
              request.dispatch=dispatch;
          } }
        notify();
    }
    void cancel(std::uint64_t id) override {
        { std::lock_guard<std::mutex> lock(mutex_); check(); const auto found=requests_.find(id);
          if (found!=requests_.end()) queue_cancel(id,found->second); }
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
            for (auto & item:requests_) queue_cancel(item.first,item.second);
            lock.unlock(); notify(); lock.lock();
            cv_.wait(lock,[&] { return ended_ || drain_completed_>=epoch; });
        }
        if (ended_ && !safe_) throw std::logic_error("I/O worker exited before buffer drain");
        requests_.clear(); completed_.clear(); ready_requests_.clear(); pending_cancels_.clear(); draining_=false;
    }
    AsyncIoStats stats() const override { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
    void rethrow_failure() const override { std::lock_guard<std::mutex> lock(mutex_); if (error_) std::rethrow_exception(error_); }
    int completion_handle() const override { return results_; }
    void acknowledge_notifications() override { consume(results_); }
private:
    void check() const { if (error_) std::rethrow_exception(error_); if (ended_ || stopping_) throw std::runtime_error("I/O worker stopped"); }
    void notify() noexcept { signal(wake_); }
    void signal(int fd) noexcept {
        std::uint64_t value=1;
        for (;;) {
            const auto written=write(fd,&value,sizeof(value));
            if (written==static_cast<ssize_t>(sizeof(value)) || (written<0 && errno==EAGAIN)) return;
            if (written<0 && errno==EINTR) continue;
            std::terminate();
        }
    }
    void consume(int fd) {
        std::uint64_t value;
        for (;;) {
            const auto read_bytes=read(fd,&value,sizeof(value));
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
        stats_.ready_time_measured=measure_ready_;
    }
    void run() noexcept {
        trace::name_thread("sa.ssd-io");
        std::unique_ptr<AsyncReader> reader;
        try {
            bind_thread_cpu(cpu_); reader=factory_();
            if (!reader || reader->stats().dedicated_worker) throw std::invalid_argument("I/O worker requires a direct reader");
            snapshot(*reader);
            { std::lock_guard<std::mutex> lock(mutex_); writes_supported_=reader->supports_write(); ready_=true; }
            cv_.notify_all();
            for (;;) {
                // Consume notification counters BEFORE inspecting their queues.
                // A completion/command arriving after the queue check keeps a
                // pending counter and wakes poll(), including cancellation CQEs.
                consume(wake_); reader->acknowledge_notifications();
                bool delivered=false;
                std::vector<std::uint64_t> cancel;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (const auto id:pending_cancels_) {
                        const auto found=requests_.find(id);
                        if (found==requests_.end()) continue;
                        auto & request=found->second;
                        if (request.phase==Phase::queued) {
                            ready_requests_.erase(ready_key(id,request));
                            request.phase=Phase::complete; ++queued_cancellations_;
                            completed_.push_back({id,-ECANCELED});
                            delivered=true;
                        } else if (request.phase==Phase::active && !request.cancel_sent) {
                            request.cancel_sent=true; cancel.push_back(id);
                        }
                    }
                    pending_cancels_.clear();
                }
                for (const auto id:cancel) reader->cancel(id);
                for (;;) {
                    std::uint64_t id=0; Request request{};
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (ready_requests_.empty()) break;
                        const auto best=ready_requests_.begin();
                        id=best->id; auto & selected=requests_.at(id);
                        if (selected.phase!=Phase::queued || selected.cancelled) throw std::logic_error("stale I/O ready request");
                        request=selected; selected.phase=Phase::active;
                        ready_requests_.erase(best);
                    }
                    const bool submitted = request.write ? reader->submit_write(id,request.fd,request.data,request.bytes,request.offset) :
                        reader->submit(id,request.fd,const_cast<void *>(request.data),request.bytes,request.offset);
                    if (!submitted) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        auto & pending=requests_.at(id); pending.phase=Phase::queued;
                        if (!pending.cancelled) ready_requests_.insert(ready_key(id,pending));
                        break;
                    }
                    ++submissions_;
                }
                reader->flush();
                ReadCompletion completion{};
                while (reader->poll(completion,false)) {
                    if (measure_ready_) completion.observed_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        const auto found=requests_.find(completion.id);
                        if (found==requests_.end() || found->second.phase!=Phase::active) throw std::logic_error("unknown I/O worker completion");
                        found->second.phase=Phase::complete; completed_.push_back(completion);
                    }
                    // Notify before consuming the next CQE, so its processing
                    // cannot delay the coordinator's first upload submission.
                    if (notify_each_) { signal(results_); cv_.notify_all(); }
                    delivered=true;
                }
                snapshot(*reader); if (delivered && !notify_each_) signal(results_); cv_.notify_all();
                bool quiesce=false,stop=false,queued=false;
                std::uint64_t epoch=0;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    // Complete requests remain paired with unconsumed receipts.
                    quiesce=drain_requested_>drain_completed_ && requests_.size()==completed_.size();
                    queued=!ready_requests_.empty();
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
            signal(results_);
            cv_.notify_all();
        }
    }
    std::function<std::unique_ptr<AsyncReader>()> factory_;
    std::uint32_t depth_;
    int cpu_,wake_=-1,results_=-1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::map<std::uint64_t,Request> requests_;
    std::set<ReadyKey,ReadyBefore> ready_requests_;
    std::vector<std::uint64_t> pending_cancels_;
    std::deque<ReadCompletion> completed_;
    std::exception_ptr error_;
    bool ready_=false,ended_=false,safe_=false,stopping_=false,draining_=false;
    bool writes_supported_=false;
    const bool notify_each_,measure_ready_;
    std::uint64_t sequence_=0,drain_requested_=0,drain_completed_=0,queued_cancellations_=0,submissions_=0;
    AsyncIoStats stats_;
};
} // namespace
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()> factory,std::uint32_t depth,int cpu,bool notify_each,bool measure_ready) {
    return std::make_unique<IoWorker>(std::move(factory),depth,cpu,notify_each,measure_ready);
}
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t depth,int cpu,std::uint32_t capacity,bool notify_each,bool measure_ready) {
    return make_io_worker([depth] { return make_uring_reader(depth); },std::max(depth,capacity),cpu,notify_each,measure_ready);
}
#else
std::unique_ptr<AsyncReader> make_io_worker(std::function<std::unique_ptr<AsyncReader>()>,std::uint32_t,int,bool,bool) { throw std::runtime_error("I/O worker requires Linux"); }
std::unique_ptr<AsyncReader> make_threaded_uring_reader(std::uint32_t,int,std::uint32_t,bool,bool) { throw std::runtime_error("I/O worker requires Linux"); }
#endif
} // namespace solid
