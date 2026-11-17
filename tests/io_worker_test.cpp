#include "solid/async_reader.hpp"
#include "solid/thread_affinity.hpp"
#include "solid/scheduler.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#ifdef __linux__
#include <sys/eventfd.h>
#include <unistd.h>
#endif

namespace {
void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F work) { bool failed=false; try { work(); } catch (const std::exception &) { failed=true; } require(failed,"invalid worker operation accepted"); }
#ifdef __linux__
struct Control {
    std::mutex mutex;
    std::condition_variable cv;
    int event=-1;
    bool block_submit=false,entered=false,allow_submit=false,allow_completion=false,poll_error=false;
    int cancellations=0;
    std::size_t engine_capacity=2;
    std::thread::id owner;
    std::vector<int> cpus;
    std::vector<std::uint64_t> submissions;
    void signal() {
        std::lock_guard<std::mutex> lock(mutex);
        if (event>=0) { std::uint64_t n=1; require(write(event,&n,sizeof(n))==static_cast<ssize_t>(sizeof(n)),"test completion signal"); }
    }
    void release() {
        { std::lock_guard<std::mutex> lock(mutex); allow_submit=true; allow_completion=true; }
        cv.notify_all(); signal();
    }
    bool wait(const std::function<bool()> & predicate) {
        std::unique_lock<std::mutex> lock(mutex); return cv.wait_for(lock,std::chrono::seconds(5),predicate);
    }
};
struct Release {
    std::shared_ptr<Control> control;
    ~Release() { control->release(); }
};
class Reader final:public solid::AsyncReader {
    struct Request { void * buffer; std::size_t bytes; };
public:
    explicit Reader(std::shared_ptr<Control> control):control(std::move(control)),owner(std::this_thread::get_id()) {
        event=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC); require(event>=0,"test eventfd");
        std::lock_guard<std::mutex> lock(this->control->mutex);
        this->control->event=event; this->control->owner=owner; this->control->cpus=solid::thread_cpus();
    }
    ~Reader() override {
        require(requests.empty(),"reader destroyed with live kernel writers");
        { std::lock_guard<std::mutex> lock(control->mutex); control->event=-1; }
        close(event);
    }
    void owned() const { require(std::this_thread::get_id()==owner,"I/O engine escaped its owner thread"); }
    bool submit(std::uint64_t id,int,void * destination,std::size_t bytes,std::uint64_t) override {
        owned();
        if (requests.size()>=control->engine_capacity) return false;
        {
            std::unique_lock<std::mutex> lock(control->mutex);
            if (control->block_submit && !control->entered) {
                control->entered=true; control->cv.notify_all();
                control->cv.wait(lock,[&] { return control->allow_submit; });
            }
        }
        requests.emplace(id,Request{destination,bytes}); ++counters.read_requests;
        { std::lock_guard<std::mutex> lock(control->mutex); control->submissions.push_back(id); }
        control->cv.notify_all(); return true;
    }
    void cancel(std::uint64_t) override {
        owned(); ++counters.cancel_requests; ++counters.cancel_completions; ++counters.cancel_busy;
        { std::lock_guard<std::mutex> lock(control->mutex); ++control->cancellations; }
        control->cv.notify_all(); // Cancellation acknowledgment precedes original buffer writes.
    }
    void flush() override { owned(); }
    bool poll(solid::ReadCompletion & result,bool wait) override {
        owned(); require(!wait,"I/O worker blocked inside engine poll instead of readiness wait");
        { std::lock_guard<std::mutex> lock(control->mutex);
          if (control->poll_error) throw std::runtime_error("synthetic poll failure");
          if (!control->allow_completion) return false; }
        if (requests.empty()) return false;
        const auto first=requests.begin();
        std::memset(first->second.buffer,static_cast<int>(first->first),first->second.bytes);
        result={first->first,static_cast<std::int64_t>(first->second.bytes)};
        counters.read_bytes+=first->second.bytes; ++counters.completed_reads; requests.erase(first); return true;
    }
    void drain() override {
        owned();
        for (const auto & request:requests) std::memset(request.second.buffer,static_cast<int>(request.first),request.second.bytes);
        requests.clear();
    }
    int completion_handle() const override { owned(); return event; }
    void acknowledge_notifications() override { owned(); std::uint64_t n; while (read(event,&n,sizeof(n))==static_cast<ssize_t>(sizeof(n))) {} }
    solid::AsyncIoStats stats() const override { owned(); return counters; }
    std::shared_ptr<Control> control;
    std::thread::id owner;
    int event=-1;
    std::map<std::uint64_t,Request> requests;
    solid::AsyncIoStats counters;
};
std::unique_ptr<solid::AsyncReader> worker(const std::shared_ptr<Control> & control,int cpu=-1) {
    return solid::make_io_worker([control] { return std::make_unique<Reader>(control); },2,cpu);
}
void cancellation_and_drain() {
    auto control=std::make_shared<Control>(); control->block_submit=true;
    auto io=worker(control); Release release{control};
    require(control->owner!=std::this_thread::get_id(),"engine constructed on the coordinator");
    std::array<unsigned char,8> a{},b{};
    require(io->submit(1,0,a.data(),a.size(),0),"first worker read");
    require(control->wait([&] { return control->entered; }),"worker did not enter submit");
    require(io->submit(2,0,b.data(),b.size(),0) && !io->submit(3,0,b.data(),b.size(),0),"bounded queue admission");
    io->cancel(2);
    solid::ReadCompletion result{};
    require(!io->poll(result,false) && b[0]==0,"queued cancellation completed before its owner observed it");
    { std::lock_guard<std::mutex> lock(control->mutex); control->allow_submit=true; }
    control->cv.notify_all();
    require(io->poll(result,true) && result.id==2 && result.result<0 && b[0]==0,"queued cancellation wrote to its source");
    io->cancel(1);
    require(control->wait([&] { return control->cancellations==1; }),"active cancellation was not submitted");
    require(!io->poll(result,false) && a[0]==0,"cancel ACK ended original READ lifetime");
    std::atomic<bool> drained=false;
    std::exception_ptr drain_error;
    std::thread drain([&] { try { io->drain(); drained=true; } catch (...) { drain_error=std::current_exception(); } });
    const bool still_live=!drained.load();
    control->release(); drain.join();
    if (drain_error) std::rethrow_exception(drain_error);
    require(still_live && drained && a[0]==1 && b[0]==0,"drain returned before original buffer writes ended");
    require(io->stats().queued_cancellations==1 && io->stats().worker_submissions==1 && io->stats().dedicated_worker,"worker accounting");
    require(io->submit(4,0,b.data(),b.size(),0),"drained worker was not reusable"); io->flush();
    require(io->poll(result,true) && result.id==4 && b[0]==4,"worker reuse after drain");
}
void failures() {
    rejects([] { (void) solid::make_io_worker([]()->std::unique_ptr<solid::AsyncReader> { throw std::runtime_error("factory failure"); },2); });
    auto control=std::make_shared<Control>(); auto io=worker(control); Release release{control};
    std::array<unsigned char,8> bytes{};
    require(io->submit(7,0,bytes.data(),bytes.size(),0),"failure test submit");
    { std::lock_guard<std::mutex> lock(control->mutex); control->poll_error=true; }
    control->signal();
    rejects([&] { solid::ReadCompletion result{}; (void) io->poll(result,true); });
    io->drain();
    rejects([&] { io->rethrow_failure(); });
    rejects([&] { (void) io->submit(8,0,bytes.data(),bytes.size(),0); });
}
void backlog_progress() {
    auto control=std::make_shared<Control>(); control->engine_capacity=1; control->allow_completion=true;
    auto io=worker(control); Release release{control};
    std::array<unsigned char,8> a{},b{};
    require(io->submit(1,0,a.data(),a.size(),0) && io->submit(2,0,b.data(),b.size(),0),"worker backlog admission");
    solid::ReadCompletion result{};
    require(io->poll(result,true) && io->poll(result,true) && a[0]==1 && b[0]==2,"worker did not refill engine after a completion");
    require(io->stats().worker_capacity==2,"worker capacity metadata");
}
void priority_progress() {
    auto control=std::make_shared<Control>(); control->engine_capacity=1;
    auto io=solid::make_io_worker([control] { return std::make_unique<Reader>(control); },5); Release release{control};
    std::array<unsigned char,5> bytes{};
    require(io->submit(1,0,&bytes[0],1,0),"priority gate submit");
    require(control->wait([&] { return control->submissions.size()==1; }),"priority gate did not enter kernel");
    require(io->submit(2,0,&bytes[1],1,0) && io->submit(3,0,&bytes[2],1,0) &&
            io->submit_prioritized(4,0,&bytes[3],1,0,solid::ReadPriority::selected) &&
            io->submit_prioritized(5,0,&bytes[4],1,0,solid::ReadPriority::selected),"priority admission");
    io->prioritize(3,solid::ReadPriority::correction_selected);
    control->release();
    solid::ReadCompletion result{};
    for (int n=0;n<5;++n) require(io->poll(result,true),"priority completion lost");
    { std::lock_guard<std::mutex> lock(control->mutex);
      require(control->submissions==std::vector<std::uint64_t>{1,3,4,5,2},"pending promotion or equal-priority FIFO failed"); }
    for (int n=0;n<5;++n) require(bytes[n]==n+1,"priority changed destination identity");
}
void placement() {
    const auto original=solid::thread_cpus(); require(!original.empty(),"missing test CPU affinity");
    {
        solid::TaskWorker coordinator(original.front());
        bool pinned=false; coordinator.submit([&] { pinned=solid::thread_cpus()==std::vector<int>{original.front()}; }); coordinator.join();
        require(pinned && solid::thread_cpus()==original,"coordinator CPU binding changed the inference thread");
    }
    rejects([] { solid::TaskWorker invalid(-2); });
    rejects([] { solid::WorkerPlacement invalid(2147483647,-1); });
    try {
        solid::WorkerPlacement plan;
        require(plan.io_cpu()!=plan.coordinator_cpu() && solid::thread_cpus()==plan.inference_cpus(),"worker core reservation");
        require(std::find(plan.inference_cpus().begin(),plan.inference_cpus().end(),plan.io_cpu())==plan.inference_cpus().end(),"I/O core left in inference mask");
        auto control=std::make_shared<Control>(); auto io=worker(control,plan.io_cpu()); Release release{control};
        require(control->cpus==std::vector<int>{plan.io_cpu()},"I/O thread did not bind its reserved CPU");
        solid::TaskWorker coordinator(plan.coordinator_cpu());
        bool pinned=false; coordinator.submit([&] { pinned=solid::thread_cpus()==std::vector<int>{plan.coordinator_cpu()}; }); coordinator.join(); require(pinned,"reserved coordinator core");
    } catch (const std::invalid_argument & error) {
        require(std::string(error.what()).find("plus an inference core")!=std::string::npos,"unexpected placement failure");
        std::cout<<"worker reservation needs three available physical cores\n";
    }
    require(solid::thread_cpus()==original,"worker reservation did not restore original affinity");
}
#endif
} // namespace
int main() {
#ifdef __linux__
    try { cancellation_and_drain(); backlog_progress(); priority_progress(); failures(); placement(); std::cout<<"I/O ownership, cancellation, drain and placement passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
#else
    std::cout<<"Linux I/O worker checks\n"; return 77;
#endif
}
