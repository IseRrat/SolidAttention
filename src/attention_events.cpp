// 主机模拟完成通知与后端工厂入口；CUDA、SYCL 实现位于各自子目录。
#include "solid/attention_events.hpp"
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>
#ifdef __linux__
#include <sys/eventfd.h>
#include <unistd.h>
#endif

namespace solid {
struct AttentionEvents::Slot {
    AttentionEvents * owner;
    StreamCompletionToken token;
    std::atomic<std::uint64_t> armed{0},done{0};
};
namespace {
std::atomic<std::uint64_t> identities{1};
std::uint64_t identity() {
    auto current=identities.load();
    for (;;) {
        if (current==UINT64_MAX) throw std::overflow_error("completion registration identity exhausted");
        if (identities.compare_exchange_weak(current,current+1)) return current;
    }
}
}
AttentionEvents::AttentionEvents(std::uint32_t layers) {
    if (!layers) throw std::invalid_argument("attention completion requires layers");
    for (std::uint32_t i=0;i<layers;++i) {
        auto slot=std::make_unique<Slot>(); slot->owner=this;
        slot->token.notify=signal; slot->token.userdata=slot.get(); slot->token.identity=identity();
        slot->token.armed_epoch=armed_epoch; slot->token.notify_epoch=signal_epoch;
        slots_.push_back(std::move(slot));
    }
#ifdef __linux__
    notification_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
    if (notification_<0) throw std::system_error(errno,std::generic_category(),"create attention completion wakeup");
#endif
}
AttentionEvents::~AttentionEvents() {
#ifdef __linux__
    if (notification_>=0) close(notification_);
#endif
}
StreamCompletionToken * AttentionEvents::token(std::uint32_t layer) { return &slots_.at(layer)->token; }
void AttentionEvents::arm(std::uint32_t layer,std::uint64_t epoch) {
    auto & slot=*slots_.at(layer);
    const auto previous=active_.load(std::memory_order_acquire);
    if (!epoch || epoch<=last_epoch_ || (previous && consumed_.load(std::memory_order_acquire)!=previous)) throw std::logic_error("completion armed before previous receipt was retired");
    stopping_.store(false,std::memory_order_release); boundary_.store(0,std::memory_order_release);
    handoff_.store(0,std::memory_order_release);
    slot.done.store(0,std::memory_order_release); slot.armed.store(epoch,std::memory_order_release);
    last_epoch_=epoch; active_.store(epoch,std::memory_order_release);
}
void AttentionEvents::signal(void * userdata) noexcept {
    signal_epoch(userdata,armed_epoch(userdata));
}
std::uint64_t AttentionEvents::armed_epoch(void * userdata) noexcept {
    return static_cast<Slot *>(userdata)->armed.load(std::memory_order_acquire);
}
void AttentionEvents::signal_epoch(void * userdata,std::uint64_t epoch) noexcept {
    auto & slot=*static_cast<Slot *>(userdata);
    if (!epoch || armed_epoch(userdata)!=epoch) return;
    auto done=slot.done.load(std::memory_order_relaxed);
    while (done<epoch) {
        if (slot.done.compare_exchange_weak(done,epoch,std::memory_order_release,std::memory_order_relaxed)) {
            slot.owner->wake();
            return;
        }
    }
}
void AttentionEvents::wake() noexcept {
#ifdef __linux__
    const std::uint64_t value=1;
    for (;;) {
        const auto result=write(notification_,&value,sizeof(value));
        if (result==static_cast<ssize_t>(sizeof(value)) || (result<0 && errno==EAGAIN)) return;
        if (result<0 && errno==EINTR) continue;
        notification_error_.store(result<0?errno:EIO,std::memory_order_release); return;
    }
#endif
}
void AttentionEvents::acknowledge() {
    const auto error=notification_error_.load(std::memory_order_acquire);
    if (error) throw std::system_error(error,std::generic_category(),"attention notification failed");
#ifdef __linux__
    std::uint64_t value;
    for (;;) {
        const auto result=read(notification_,&value,sizeof(value));
        if (result==static_cast<ssize_t>(sizeof(value))) continue;
        if (result<0 && errno==EINTR) continue;
        if (result<0 && errno==EAGAIN) return;
        throw std::system_error(result<0?errno:EIO,std::generic_category(),"consume attention completion wakeup");
    }
#endif
}
std::optional<AttentionReceipt> AttentionEvents::poll(std::uint32_t layer,std::uint64_t epoch) {
    auto & slot=*slots_.at(layer);
    if (!epoch || slot.armed.load(std::memory_order_acquire)!=epoch || active_.load(std::memory_order_acquire)!=epoch) throw std::logic_error("attention receipt epoch mismatch");
    if (consumed_.load(std::memory_order_acquire)==epoch) return {};
    if (slot.done.load(std::memory_order_acquire)==epoch) {
        const auto duration=elapsed_ns(layer); consumed_.store(epoch,std::memory_order_release);
        return AttentionReceipt{AttentionProof::device_signal,duration};
    }
    if (boundary_seen(epoch)) {
        const auto proof=static_cast<AttentionProof>(boundary_proof_.load(std::memory_order_acquire));
        consumed_.store(epoch,std::memory_order_release); return AttentionReceipt{proof,0};
    }
    return {};
}
void AttentionEvents::boundary(AttentionProof proof,std::uint64_t epoch) {
    if (proof==AttentionProof::device_signal) throw std::invalid_argument("host boundary requires a dependency proof");
    const auto active=active_.load(std::memory_order_acquire);
    if (!active || (epoch && epoch!=active)) return;
    boundary_proof_.store(static_cast<int>(proof),std::memory_order_release);
    if (proof!=AttentionProof::attention_boundary) handoff_.store(active,std::memory_order_release);
    boundary_.store(active,std::memory_order_release); wake();
}
bool AttentionEvents::boundary_seen(std::uint64_t epoch) const { return epoch && boundary_.load(std::memory_order_acquire)==epoch; }
bool AttentionEvents::handoff_seen(std::uint64_t epoch) const { return epoch && handoff_.load(std::memory_order_acquire)==epoch; }
void AttentionEvents::stop() noexcept { stopping_.store(true,std::memory_order_release); wake(); }
void AttentionEvents::retire_after_sync() { consumed_.store(active_.load(std::memory_order_acquire),std::memory_order_release); }
} // namespace solid
