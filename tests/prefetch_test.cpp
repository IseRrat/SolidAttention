#include "prefetch_fakes.hpp"
#include <iostream>
#include <numeric>

using namespace test_prefetch;
namespace {
void slots() {
    solid::SlotTable table(1, 1);
    const solid::BlockIdentity a{0, 0, 0, 1}, b{0, 0, 1, 1};
    const auto first = *table.acquire(a);
    require(!table.retire(first) && !table.acquire(b), "loading slot became reusable after cancel request");
    require(table.complete(first), "read completion lost");
    const auto second = *table.acquire(b);
    require(first.slot == second.slot && first.generation < second.generation, "slot generation did not advance");
    require(!table.complete(first), "stale completion accepted");
    table.complete(second);
    table.pin(second);
    table.pin(second);
    require(!table.retire(second) && !table.acquire(a), "pinned slot became reusable");
    table.unpin(second);
    require(!table.acquire(a), "first of two readers released the slot");
    table.unpin(second);
    const auto third = *table.acquire(a);
    rejects([&] { table.unpin(second); });
    require(!table.complete(second) && table.complete(third), "old generation affected new slot");
}
void prefix() {
    const solid::KvGeometry g{1, 2, 2, 4, 16, 64, solid::KvLayout::interleaved};
    auto file = std::make_unique<File>();
    auto reader = std::make_unique<Reader>(file.get());
    auto * observe = reader.get();
    solid::FileKvStore store(g, std::move(file), 32);
    fill(store, 6);
    solid::Prefetcher prefetch(store, {8, 1, 2, 1}, 2, std::move(reader));
    prefetch.begin(0, 6);
    observe->auto_complete = true;
    prefetch.poll();
    const auto reads = observe->stats().read_requests;
    store.append_interleaved(0, 6, token(g, 0, 6));
    std::vector<solid::HeadSelection> selected(2);
    for (auto & head : selected) { head.positions.resize(7); std::iota(head.positions.begin(), head.positions.end(), 0); }
    const auto leases = prefetch.resolve(0, 6, selected);
    verify(store, prefetch, leases);
    require(observe->stats().read_requests == reads, "dirty tail refresh reread an immutable committed prefix");
    require(prefetch.stats().refreshed_blocks == 2, "tail version not refreshed");
    rejects([&] { prefetch.begin(0, 7); });
    for (const auto & lease : leases) prefetch.unpin(lease);
    observe->duplicate = solid::ReadCompletion{999999, 8};
    prefetch.poll();
    require(prefetch.stats().stale_completions == 1, "stale notification was not ignored");
    prefetch.finish();
}
void wrong_predictions() {
    const solid::KvGeometry g{2, 1, 2, 1, 32, 64, solid::KvLayout::interleaved};
    auto file = std::make_unique<File>();
    auto reader = std::make_unique<Reader>(file.get());
    auto * observe = reader.get();
    solid::FileKvStore store(g, std::move(file), 8);
    fill(store, 8);
    solid::Prefetcher prefetch(store, {6, 1, 1, 1}, 6, std::move(reader));
    const auto run = [&](std::uint32_t layer, std::uint64_t p, const std::vector<std::uint64_t> & positions) {
        prefetch.begin(layer, p);
        store.append_interleaved(layer, p, token(g, layer, p));
        const auto leases = prefetch.resolve(layer, p, {{positions, {}}});
        verify(store, prefetch, leases);
        for (const auto & lease : leases) prefetch.unpin(lease);
    };
    run(0, 8, {0, 1, 2, 3, 4, 8});
    run(1, 8, {0, 1, 2, 3, 4, 8});
    run(0, 9, {0, 5, 6, 7, 8, 9});
    require(prefetch.stats().unused_blocks >= 4 && observe->stats().cancel_requests > 0, "incorrect predictions were not cancelled and overwritten");
    require(prefetch.stats().inflight_hits > 0 && prefetch.stats().missing_blocks > 0, "inflight hit or missing correction missing");
    prefetch.finish();
    require(observe->pending.empty(), "finish left original reads alive");
}
void read_error(solid::KvLayout layout) {
    const solid::KvGeometry g{1, 1, 2, 1, 16, 64, layout};
    auto file = std::make_unique<File>();
    auto reader = std::make_unique<Reader>(file.get());
    reader->fail_next = true;
    solid::FileKvStore store(g, std::move(file), 8);
    fill(store, 6);
    solid::Prefetcher prefetch(store, {4, 1, 1, 1}, 4, std::move(reader));
    prefetch.begin(0, 6);
    store.append_interleaved(0, 6, token(g, 0, 6));
    rejects([&] { (void) prefetch.resolve(0, 6, {{{0, 1, 2, 6}, {}}}); });
    rejects([&] { prefetch.begin(0, 7); });
}
void capacity() {
    const solid::KvGeometry g{1, 1, 2, 1, 16, 64, solid::KvLayout::interleaved};
    solid::MemoryKvStore store(g);
    fill(store, 6);
    solid::Prefetcher host(store, {4, 1, 1, 1}, 2, std::make_unique<Reader>());
    host.begin(0, 6);
    store.append_interleaved(0, 6, token(g, 0, 6));
    rejects([&] { (void) host.resolve(0, 6, {{{0, 1, 2, 6}, {}}}); });
    host.finish();
}
void device() {
    const solid::KvGeometry g{1, 1, 2, 1, 16, 64, solid::KvLayout::interleaved};
    solid::MemoryKvStore store(g);
    fill(store, 6);
    solid::Prefetcher host(store, {4, 1, 1, 1}, 4, std::make_unique<Reader>());
    Transport transport;
    solid::DeviceCache cache(host, g, 2, transport);
    host.begin(0, 6);
    cache.prefetch_ready();
    // The host source of the queued H2D cannot be recycled even though SSD I/O is complete.
    store.append_interleaved(0, 6, token(g, 0, 6));
    rejects([&] { host.begin(0, 7); });
    cache.wait_transfers();
    const std::vector<solid::HeadSelection> first{{{0, 1, 2, 6}, {}}};
    auto leases = host.resolve(0, 6, first);
    const auto mask = cache.prepare(leases, first);
    for (const auto & lease : leases) host.unpin(lease);
    const auto zero = *cache.find({0, 0, 0, 1}), one = *cache.find({0, 0, 1, 1});
    const auto zero_bytes = transport.memory.at(zero.slot), one_bytes = transport.memory.at(one.slot);
    require(std::count(mask.begin(), mask.end(), solid::Half{0}) == 8, "GQA masks lost selected token identities");
    require(!cache.attention_complete(cache.attention_epoch() + 1), "stale attention event released slots");
    rejects([&] { cache.prefetch_ready(); });
    require(cache.attention_complete(cache.attention_epoch()), "attention did not release readers");
    host.begin(0, 7);
    store.append_interleaved(0, 7, token(g, 0, 7));
    const std::vector<solid::HeadSelection> second{{{0, 1, 3, 7}, {}}};
    leases = host.resolve(0, 7, second);
    (void) cache.prepare(leases, second);
    for (const auto & lease : leases) host.unpin(lease);
    require(*cache.find({0, 0, 0, 1}) == zero && *cache.find({0, 0, 1, 1}) == one, "valid device blocks moved during correction");
    require(transport.memory.at(zero.slot) == zero_bytes && transport.memory.at(one.slot) == one_bytes, "valid device payload was rewritten");
    cache.attention_complete(cache.attention_epoch());
    cache.finish(); host.finish();
    require(cache.stats().uploads == 6 && cache.stats().reused_blocks >= 3, "device upload/hit accounting");
}
}
int main() {
    try {
        slots(); prefix(); wrong_predictions(); read_error(solid::KvLayout::interleaved);
        read_error(solid::KvLayout::separate); capacity(); device();
        std::cout << "prefetch, correction, cancellation races, versions, host/device leases and stable slots passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
