#include "solid/async_reader.hpp"
#include <cerrno>
#include <chrono>
#include <climits>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#ifdef SOLID_HAVE_URING
#include <liburing.h>
#include <sys/eventfd.h>
#include <unistd.h>
#endif

namespace solid {
#ifdef SOLID_HAVE_URING
namespace {
class UringReader final : public AsyncReader {
public:
    explicit UringReader(std::uint32_t depth) : limit_(depth) {
        if (depth < 2 || depth > 16384) throw std::invalid_argument("uring queue depth must be in [2,16384]");
        const auto status = io_uring_queue_init(depth * 2, &ring_, 0);
        if (status < 0) throw std::system_error(-status, std::generic_category(), "io_uring_queue_init");
        notification_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
        if (notification_<0) { const auto error=errno; io_uring_queue_exit(&ring_); throw std::system_error(error,std::generic_category(),"create uring completion event"); }
        const auto registered=io_uring_register_eventfd(&ring_,notification_);
        if (registered<0) { close(notification_); io_uring_queue_exit(&ring_); throw std::system_error(-registered,std::generic_category(),"register uring completion event"); }
    }
    ~UringReader() override {
        try { drain(); } catch (...) { std::terminate(); }
        io_uring_queue_exit(&ring_);
        close(notification_);
    }
    bool submit(std::uint64_t id, int fd, void * destination, std::size_t bytes, std::uint64_t offset) override {
        if (!id || id >= cancel_bit || !destination || bytes == 0 || bytes > INT_MAX || fd < 0 ||
            offset > static_cast<std::uint64_t>(INT64_MAX) - bytes || reads_.count(id) || cancellations_.count(id)) {
            throw std::invalid_argument("invalid async read request");
        }
        if (reads_.size() >= limit_) return false;
        Request request{fd, static_cast<unsigned char *>(destination), bytes, 0, offset, false, 0, std::chrono::steady_clock::now()};
        auto inserted = reads_.emplace(id, request);
        try { stage(id, inserted.first->second); }
        catch (...) { reads_.erase(inserted.first); throw; }
        return true;
    }
    void cancel(std::uint64_t id) override {
        const auto found = reads_.find(id);
        if (found == reads_.end() || found->second.cancelled) return;
        found->second.cancelled = true;
        auto * sqe = get_sqe();
        io_uring_prep_cancel64(sqe, id, 0);
        io_uring_sqe_set_data64(sqe, id | cancel_bit);
        cancellations_.insert(id);
        ++stats_.cancel_requests;
    }
    void flush() override {
        while (io_uring_sq_ready(&ring_) != 0) {
            int count;
            do { count = io_uring_submit(&ring_); } while (count == -EINTR);
            if (count < 0) throw std::system_error(-count, std::generic_category(), "io_uring_submit");
            if (count == 0) throw std::runtime_error("io_uring_submit made no progress");
        }
    }
    bool poll(ReadCompletion & completion, bool wait) override {
        flush();
        for (;;) {
            io_uring_cqe * cqe = nullptr;
            int status = io_uring_peek_cqe(&ring_, &cqe);
            if (status == -EAGAIN && wait && (!reads_.empty() || !cancellations_.empty())) {
                do { status = io_uring_wait_cqe(&ring_, &cqe); } while (status == -EINTR);
            }
            if (status == -EAGAIN || (!cqe && reads_.empty() && cancellations_.empty())) return false;
            if (status < 0) throw std::system_error(-status, std::generic_category(), "io_uring completion");
            if (!cqe) return false;
            const auto tag = io_uring_cqe_get_data64(cqe);
            const auto result = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            if (tag & cancel_bit) {
                if (cancellations_.erase(tag & ~cancel_bit) != 1) throw std::runtime_error("unknown cancellation completion");
                ++stats_.cancel_completions;
                if (result == 0) ++stats_.cancel_successes;
                else if (result == -ENOENT) ++stats_.cancel_not_found;
                else if (result == -EALREADY) ++stats_.cancel_busy;
                else ++stats_.cancel_errors;
                // Cancellation CQEs never release the original read buffer.
                continue;
            }
            const auto found = reads_.find(tag);
            if (found == reads_.end()) throw std::runtime_error("unknown read completion");
            auto & request = found->second;
            if ((result == -EINTR || result == -EAGAIN) && !request.cancelled && ++request.retries <= 64) {
                stage(tag, request); flush(); continue;
            }
            std::int64_t final_result = result;
            if (result > 0 && static_cast<std::size_t>(result) <= request.bytes - request.done) {
                request.done += result;
                request.retries = 0;
                stats_.read_bytes += result;
                if (request.done < request.bytes && !request.cancelled) { stage(tag, request); flush(); continue; }
                final_result = request.done == request.bytes ? static_cast<std::int64_t>(request.done) : -ECANCELED;
            } else if (result == 0) final_result = -ENODATA;
            else if (result > 0) final_result = -EOVERFLOW;
            stats_.completion_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - request.start).count();
            ++stats_.completed_reads;
            completion = {tag, final_result};
            reads_.erase(found);
            return true;
        }
    }
    void drain() override {
        std::vector<std::uint64_t> ids;
        for (const auto & pair : reads_) ids.push_back(pair.first);
        for (const auto id : ids) cancel(id);
        flush();
        ReadCompletion completion{};
        while (!reads_.empty() || !cancellations_.empty()) (void) poll(completion, true);
    }
    AsyncIoStats stats() const override { return stats_; }
    int completion_handle() const override { return notification_; }
    void acknowledge_notifications() override {
        std::uint64_t count;
        for (;;) {
            const auto result=::read(notification_,&count,sizeof(count));
            if (result==static_cast<ssize_t>(sizeof(count))) continue;
            if (result<0 && errno==EINTR) continue;
            if (result<0 && errno==EAGAIN) return;
            throw std::system_error(result<0?errno:EIO,std::generic_category(),"consume uring notification");
        }
    }
private:
    struct Request {
        int fd;
        unsigned char * destination;
        std::size_t bytes;
        std::size_t done;
        std::uint64_t offset;
        bool cancelled;
        std::uint32_t retries;
        std::chrono::steady_clock::time_point start;
    };
    static constexpr std::uint64_t cancel_bit = std::uint64_t{1} << 63;
    io_uring ring_{};
    int notification_=-1;
    std::uint32_t limit_;
    std::map<std::uint64_t, Request> reads_;
    std::set<std::uint64_t> cancellations_;
    AsyncIoStats stats_;
    io_uring_sqe * get_sqe() {
        auto * sqe = io_uring_get_sqe(&ring_);
        if (!sqe) { flush(); sqe = io_uring_get_sqe(&ring_); }
        if (!sqe) throw std::runtime_error("io_uring submission queue exhausted");
        return sqe;
    }
    void stage(std::uint64_t id, Request & request) {
        auto * sqe = get_sqe();
        io_uring_prep_read(sqe, request.fd, request.destination + request.done,
                           static_cast<unsigned int>(request.bytes - request.done), request.offset + request.done);
        io_uring_sqe_set_data64(sqe, id);
        ++stats_.read_requests;
    }
};
}
bool uring_available() { return true; }
std::unique_ptr<AsyncReader> make_uring_reader(std::uint32_t depth) { return std::make_unique<UringReader>(depth); }
#else
bool uring_available() { return false; }
std::unique_ptr<AsyncReader> make_uring_reader(std::uint32_t) { throw std::runtime_error("build with SOLID_URING=ON for async SSD prefetch"); }
#endif
} // namespace solid
