// CUDA Prefill 与异步均值/选块衔接检查。
#include "tiled_prefill_cases.hpp"
#include "infllm_reference_cases.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <future>
#include <chrono>
#include <string>
#include <type_traits>

namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T>
void upload_ready(T * destination, const T * source, std::size_t elements) {
    cuda_check(cudaMemcpy(destination, source, elements * sizeof(T), cudaMemcpyHostToDevice));
    // A pageable H2D cudaMemcpy may return after staging, before DMA ends.
    // The borrowed-input APIs consume on nonblocking streams and require a
    // producer-complete allocation, so drain the upload's default stream.
    cuda_check(cudaStreamSynchronize(nullptr));
}
template<class T>
auto device_buffer(std::size_t count) {
    void * data = nullptr;
    cuda_check(cudaMalloc(&data, count * sizeof(T)));
    return std::unique_ptr<T, decltype(&cudaFree)>(static_cast<T *>(data), cudaFree);
}
void sparse_prefill(int device) {
    cuda_check(cudaSetDevice(device));
    constexpr std::uint32_t width = 64, heads = 2, query_heads = 4, local = 5, queries = 4;
    constexpr std::uint64_t first = 12;
    const solid::KvGeometry geometry{1, heads, width, 4, 32, 64, solid::KvLayout::interleaved};
    auto backend = solid::make_cuda_tiled_prefill(geometry, query_heads, queries, 16, device, local);
    solid::MemoryKvStore store(geometry);
    std::vector<std::vector<solid::Half>> raw;
    for (std::uint64_t p = 0; p < first + queries; ++p) {
        auto & row = raw.emplace_back(2 * width * heads);
        for (std::size_t d = 0; d < row.size(); ++d) row[d] = solid::to_half((static_cast<int>((d + 13 * p) % 29) - 14) / 16.f);
        store.append_interleaved(0, p, row);
    }
    const std::vector<std::uint64_t> positions{0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<solid::Half> packed;
    for (std::uint32_t h = 0; h < heads; ++h) for (const auto p : positions) {
        const auto * row = raw[p].data() + 2 * h * width;
        packed.insert(packed.end(), row, row + 2 * width);
    }
    for (const auto split : {false, true}) {
        const solid::RelativeRope rope{width, 10000.f, split};
        std::vector<float> q(queries * query_heads * width);
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = (static_cast<int>(i % 19) - 9) / 16.f;
        for (std::uint32_t i = 0; i < queries * query_heads; ++i) rope.rotate(q.data() + i * width, local - 1);
        constexpr std::size_t fixed = 4;
        const auto actual = backend->sparse({first, queries, 0.125f, local, rope, positions, q, packed, fixed});
        constexpr auto origin = first + 1 - local;
        std::vector<float> expected, feedback((first + queries - origin) * query_heads);
        for (std::uint32_t i = 0; i < queries; ++i) {
            const auto visible = std::upper_bound(positions.begin(), positions.end(), first + i);
            const auto recent = std::lower_bound(positions.begin() + fixed, visible, first + i + 1 - local);
            solid::HeadSelection selected{{positions.begin(), positions.begin() + fixed}, {}};
            selected.positions.insert(selected.positions.end(), recent, visible);
            const std::vector<solid::HeadSelection> selection(heads, selected);
            const auto query = q.begin() + i * query_heads * width;
            std::vector<float> probabilities;
            const auto output = solid::streamed_attention(store, 0, first + i, query_heads,
                {query, query + query_heads * width}, 0.125f, &selection, &rope, local, &probabilities);
            expected.insert(expected.end(), output.begin(), output.end());
            for (std::uint32_t head = 0; head < query_heads; ++head) {
                for (std::uint32_t key = 0; key < local; ++key) {
                    feedback[(i + key) * query_heads + head] += probabilities[head * local + key];
                }
            }
        }
        test_tiled::close(actual.attention, expected, 2e-3f);
        test_tiled::close(actual.key_scores, feedback, 2e-3f);

        std::vector<solid::Half> current;
        for (auto p = first; p < first + queries; ++p) current.insert(current.end(), raw[p].begin(), raw[p].end());
        auto device_q = device_buffer<float>(q.size());
        auto device_kv = device_buffer<solid::Half>(current.size());
        auto device_result = device_buffer<float>(q.size());
        upload_ready(device_q.get(), q.data(), q.size());
        upload_ready(device_kv.get(), current.data(), current.size());
        const solid::DeviceQuery query{device_q.get(), q.size(), solid::DeviceMemory::cuda};
        const auto mean = backend->mean_sparse_query(query, queries);
        std::vector<float> host_mean(mean.elements), expected_mean(mean.elements);
        cuda_check(cudaMemcpy(host_mean.data(), mean.data, mean.elements * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::uint32_t i = 0; i < queries; ++i) for (std::size_t d = 0; d < mean.elements; ++d) {
            expected_mean[d] += q[i * mean.elements + d];
        }
        for (auto & value : expected_mean) value /= queries;
        test_tiled::close(host_mean, expected_mean, 1e-6f);

        auto event_ranker = solid::make_cuda_block_ranker(geometry, query_heads, 1, 16, device);
        std::vector<solid::Half> representatives(2 * heads * width);
        for (std::size_t i = 0; i < representatives.size(); ++i) {
            representatives[i] = solid::to_half((static_cast<int>(i % 13) - 6) / 8.f);
        }
        event_ranker->publish(0, 0, 2, representatives);
        const solid::BlockCandidates request{{0, 1}, {4, 4}, 4, true};
        const auto expected_ranking = event_ranker->rank_device(0, mean, request);
        const auto prior_waits = backend->stats().retrieval_mean_host_waits;
        const auto queued = backend->mean_sparse_query_async(query, queries);
        const auto queued_ranking = event_ranker->rank_device_after_event(0, queued, request);
        test_tiled::require(queued_ranking == expected_ranking && event_ranker->stats().query_ready_event_waits == 1 &&
            backend->stats().retrieval_mean_host_waits == prior_waits,
            "event-dependent retrieval changed ranking or waited on the host for its mean");
        const auto empty_query = backend->mean_sparse_query_async(query, queries);
        const auto empty_ranking = event_ranker->rank_device_after_event(0, empty_query, {{}, {}, 0, true});
        test_tiled::require(empty_ranking == std::vector<std::vector<std::uint64_t>>(heads) &&
            event_ranker->stats().query_ready_event_waits == 2 && backend->stats().retrieval_mean_host_waits == prior_waits,
            "empty event-dependent ranking ignored query readiness");
        // Hold the producer queue until submission returns. A hidden pageable
        // copy or premature host wait would keep submit blocked at this gate.
        cudaStream_t producer = nullptr;
        cudaEvent_t ready = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
        cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        std::promise<void> release;
        auto gate = release.get_future().share();
        cuda_check(cudaLaunchHostFunc(producer, [](void * value) {
            static_cast<std::shared_future<void> *>(value)->wait();
        }, &gate));
        cuda_check(cudaEventRecord(ready, producer));
        auto candidates = request;
        const auto submitted_calls = event_ranker->stats().calls;
        auto submission = std::async(std::launch::async, [&] {
            return event_ranker->submit_rank_device(0, {mean, ready}, candidates);
        });
        const bool returned = submission.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        release.set_value();
        auto receipt = submission.get();
        // Submission must own this metadata until its asynchronous H2D ends.
        candidates.blocks.assign(2, UINT64_MAX);
        candidates.tokens.assign(2, 0);
        bool rejects_reuse = false;
        try { (void) event_ranker->submit_rank_device(0, {mean, nullptr}, request); }
        catch (const std::logic_error &) { rejects_reuse = true; }
        const bool uncollected = event_ranker->stats().calls == submitted_calls;
        const auto deferred_ranking = receipt();
        bool rejects_receipt = false;
        try { (void) receipt(); }
        catch (const std::logic_error &) { rejects_receipt = true; }
        cuda_check(cudaStreamSynchronize(producer));
        cuda_check(cudaEventDestroy(ready));
        cuda_check(cudaStreamDestroy(producer));
        test_tiled::require(returned && uncollected && rejects_reuse && rejects_receipt && deferred_ranking == expected_ranking,
            "deferred rank waited before the consumer, reused live metadata, or changed selected IDs");
        auto prefix_receipt = event_ranker->submit_rank_device(0, {mean, nullptr}, {{}, {}, 4, true, 2, 4});
        bool rejects_old_receipt = false;
        try { (void) receipt(); }
        catch (const std::logic_error &) { rejects_old_receipt = true; }
        test_tiled::require(rejects_old_receipt && prefix_receipt() == expected_ranking,
            "bank-prefix submission changed retrieval or let an old receipt consume a new result");
        const auto before = backend->stats();
        const auto direct = backend->sparse({first, queries, 0.125f, local, rope, positions, {}, packed, fixed, query,
            {device_result.get(), q.size(), solid::DeviceMemory::cuda},
            {device_kv.get(), current.size(), solid::DeviceMemory::cuda}});
        std::vector<float> output(q.size());
        cuda_check(cudaMemcpy(output.data(), device_result.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        test_tiled::close(output, expected, 2e-3f);
        test_tiled::close(direct.key_scores, feedback, 2e-3f);
        const auto after = backend->stats();
        test_tiled::require(direct.attention.empty() && after.query_upload_bytes == before.query_upload_bytes &&
            after.output_download_bytes == before.output_download_bytes &&
            after.upload_bytes - before.upload_bytes == (packed.size() - current.size()) * sizeof(solid::Half),
            "direct sparse prefill transferred GPU input or output through host");
        constexpr std::uint32_t ring_capacity = 7;
        std::vector<solid::Half> ring(ring_capacity * heads * width);
        for (auto p = first - ring_capacity; p < first; ++p) for (std::uint32_t h = 0; h < heads; ++h)
            std::copy_n(raw[p].data() + 2 * h * width, width, ring.data() + (p % ring_capacity * heads + h) * width);
        auto device_ring = device_buffer<solid::Half>(ring.size());
        upload_ready(device_ring.get(), ring.data(), ring.size());
        backend->begin_sparse_local({first, queries, .125f, local, rope, query,
            {device_kv.get(), current.size(), solid::DeviceMemory::cuda},
            {device_ring.get(), ring_capacity, solid::DeviceMemory::cuda}});
        const auto after_local = backend->mean_sparse_query_async(query, queries);
        test_tiled::require(event_ranker->rank_device_after_event(0, after_local, request) == expected_ranking,
            "local-first prefill changed retrieval");
        const auto staged = backend->sparse({first, queries, .125f, local, rope, positions, {}, packed, fixed, query,
            {device_result.get(), q.size(), solid::DeviceMemory::cuda},
            {device_kv.get(), current.size(), solid::DeviceMemory::cuda}});
        cuda_check(cudaMemcpy(output.data(), device_result.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        test_tiled::close(output, expected, 2e-3f);
        test_tiled::close(staged.key_scores, feedback, 2e-3f);
    }

    packed[width] = solid::to_half(std::numeric_limits<float>::quiet_NaN());
    const std::vector<float> zero_query(queries * query_heads * width);
    const solid::RelativeRope rope{width, 10000.f, false};
    try {
        (void) backend->sparse({first, queries, 0.125f, local, rope, positions, zero_query, packed});
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()) == "non-finite or empty sparse CUDA prefill") return;
        throw;
    }
    throw std::runtime_error("sparse CUDA prefill accepted a non-finite value result");
}
void probability_representatives(int device) {
    cuda_check(cudaSetDevice(device));
    const solid::KvGeometry geometry{1, 1, 64, 32, 1024, 64};
    const solid::SparseBudget budget{256, 32, 96, 4, true, true};
    solid::RepresentativeIndex reference(geometry, 2, budget);
    auto scoring_geometry = geometry;
    scoring_geometry.heads = 2;
    auto ranker = solid::make_cuda_block_ranker(scoring_geometry, 2, 1, budget.total, device, 0, 1, true);
    test_tiled::require(ranker->configure_probability_builder(budget.initial, budget.local, budget.representatives),
                        "probability builder configuration");
    solid::RepresentativeIndex observed(geometry, 2, budget);
    observed.set_ranker(std::move(ranker));
    auto production_ranker = solid::make_cuda_block_ranker(scoring_geometry, 2, 1, budget.total, device, 0, 1);
    test_tiled::require(production_ranker->configure_probability_builder(budget.initial, budget.local, budget.representatives) &&
        !production_ranker->host_representative_metadata(), "production representative diagnostics were enabled");
    solid::RepresentativeIndex production(geometry, 2, budget);
    production.set_ranker(std::move(production_ranker));
    const auto fixed_host_bytes = production.resident_bytes();
    std::uint64_t checked_means = 0;
    cudaStream_t queue = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&queue, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> compute(queue, cudaStreamDestroy);
    for (std::uint64_t first = 0; first < 800;) {
        const std::uint32_t count = first < 768 ? 96 : 1;
        const auto end = first + count;
        const auto origin = first + 1 - std::min<std::uint64_t>(first + 1, budget.local);
        std::vector<solid::Half> kv(count * 128);
        std::vector<float> scores((end - origin) * 2);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto position = first + i;
            std::vector<solid::Half> keys(64);
            for (std::size_t d = 0; d < 64; ++d) {
                keys[d] = solid::to_half(static_cast<float>((position + d) % 19) / 16.f);
                kv[i * 128 + d] = keys[d];
            }
            const auto recent = std::min<std::uint64_t>(position + 1, budget.local);
            std::vector<float> probabilities(recent * 2);
            for (std::size_t h = 0; h < 2; ++h) for (std::uint64_t j = 0; j < recent; ++j) {
                const auto value = static_cast<float>((position + j * (h + 1)) % 7 + 1) / 1024.f;
                probabilities[h * recent + j] = value;
                const auto key = position + 1 - recent + j - origin;
                scores[count == 1 ? h * recent + j : key * 2 + h] += value;
            }
        }
        if (count != 1) reference.observe_prefill_scores(0, first, count, kv, scores);
        else {
            reference.observe(0, first, std::vector<float>(128, 0.f), {kv.begin(), kv.begin() + 64});
            reference.observe_probabilities(0, first, scores);
        }
        auto device_kv = device_buffer<solid::Half>(kv.size());
        auto device_scores = device_buffer<float>(scores.size());
        upload_ready(device_kv.get(), kv.data(), kv.size());
        upload_ready(device_scores.get(), scores.data(), scores.size());
        const solid::DeviceKvBatch current{device_kv.get(), kv.size(), solid::DeviceMemory::cuda};
        const solid::DeviceQuery probabilities{device_scores.get(), scores.size(), solid::DeviceMemory::cuda};
        if (count != 1) {
            observed.observe_prefill_device_scores(0, first, count, current, probabilities);
            production.observe_prefill_device_scores(0, first, count, current, probabilities);
        }
        else {
            observed.begin_probability_query(0, first);
            production.begin_probability_query(0, first);
            observed.append_keys(0, first, {kv.begin(), kv.begin() + 64});
            production.append_keys(0, first, {kv.begin(), kv.begin() + 64});
            observed.observe_device_probabilities(0, first, count, current, probabilities, false, compute.get());
            if (first == 768) {
                std::promise<void> release;
                auto gate = release.get_future().share();
                cuda_check(cudaLaunchHostFunc(compute.get(), [](void * value) {
                    static_cast<std::shared_future<void> *>(value)->wait();
                }, &gate));
                auto submission = std::async(std::launch::async, [&] {
                    production.observe_device_probabilities(0, first, count, current, probabilities, false, compute.get(), true);
                });
                const bool returned = submission.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
                release.set_value();
                submission.get();
                test_tiled::require(returned, "upstream compute-queue probability update waited on the host");
            } else production.observe_device_probabilities(0, first, count, current, probabilities, false, compute.get(), true);
            // The next graph nodes may immediately reuse both borrowed inputs.
            cuda_check(cudaMemsetAsync(device_kv.get(), 0xff, kv.size() * sizeof(solid::Half), compute.get()));
            cuda_check(cudaMemsetAsync(device_scores.get(), 0xff, scores.size() * sizeof(float), compute.get()));
            cuda_check(cudaStreamSynchronize(compute.get()));
            observed.finish_device_probabilities();
            production.finish_device_probabilities();
        }
        for (std::uint64_t block = 0; block < geometry.blocks_per_head(); ++block) {
            test_tiled::require(observed.ready(0, block) == reference.ready(0, block), "GPU representative publication boundary");
            test_tiled::require(production.ready(0, block) == reference.ready(0, block), "production representative publication boundary");
            if (!reference.ready(0, block)) continue;
            for (std::uint32_t h = 0; h < 2; ++h) {
                test_tiled::require(reference.representative_positions(0, h, block) == observed.representative_positions(0, h, block),
                                    "GPU probability representatives differ from InfLLM reference");
                if (block >= checked_means) for (const auto d : {0u, 31u, 63u}) {
                    std::vector<float> unit(128, 0);
                    unit[h * 64 + d] = 1;
                    test_tiled::require(production.score(0, h, block, unit) == observed.score(0, h, block, unit),
                        "production bank mean differs from the audited representative");
                }
            }
        }
        checked_means = production.complete_blocks(0);
        const solid::BlockCandidates request{{}, {}, 64, true, checked_means, geometry.block_tokens};
        const std::vector<float> query(128, .125f);
        test_tiled::require(production.rank_blocks(0, query, request) == observed.rank_blocks(0, query, request),
            "production-only representative storage changed retrieval");
        test_tiled::require(production.resident_bytes() == fixed_host_bytes,
            "production accumulated per-block host representative metadata");
        first = end;
    }
    bool rejects_positions = false;
    try { (void) production.representative_positions(0, 0, 0); }
    catch (const std::logic_error &) { rejects_positions = true; }
    test_tiled::require(rejects_positions, "production exposed disabled representative diagnostics");
}
}

int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") {
        std::cout<<"Usage: solid-cuda-prefill-test --run-device-check [device ordinal] [upstream-fixture.bin]\n"; return argc==1?0:2;
    }
    try {
        if (argc>4) throw std::invalid_argument("unexpected device-check argument");
        const auto device=argc>=3?std::stoi(argv[2]):0;
        const auto factory = [&](auto g, auto qh, auto queries, auto tile) {
            return solid::make_cuda_tiled_prefill(g, qh, queries, tile, device);
        };
        test_tiled::run_cases(factory);
        test_tiled::frames(factory, 128, false, 32, 1e-3f);
        sparse_prefill(device);
        probability_representatives(device);
        std::unique_ptr<solid::TiledPrefill> fixture_mean;
        if (argc == 4) test_infllm::run(argv[3], [&](auto & index, auto geometry, auto heads, auto budget) {
            const auto kv_heads = geometry.heads;
            fixture_mean = solid::make_cuda_tiled_prefill(geometry, heads, budget.prefill_group_tokens(),
                geometry.capacity_tokens, device, budget.local);
            geometry.heads = heads;
            auto ranker = solid::make_cuda_block_ranker(geometry, heads, 1, budget.total, device, 0, kv_heads, true);
            test_infllm::require(ranker->configure_probability_builder(budget.initial, budget.local, budget.representatives),
                "upstream fixture probability builder configuration");
            index.set_ranker(std::move(ranker));
        }, [](auto & index, auto first, auto count, const auto & kv, const auto & scores) {
            auto device_kv = device_buffer<solid::Half>(kv.size());
            auto device_scores = device_buffer<float>(scores.size());
            upload_ready(device_kv.get(), kv.data(), kv.size());
            upload_ready(device_scores.get(), scores.data(), scores.size());
            index.observe_prefill_device_scores(0, first, count,
                {device_kv.get(), kv.size(), solid::DeviceMemory::cuda},
                {device_scores.get(), scores.size(), solid::DeviceMemory::cuda});
        }, [&](const auto & queries, auto count, auto width) {
            auto device_queries = device_buffer<float>(queries.size());
            upload_ready(device_queries.get(), queries.data(), queries.size());
            const auto mean = fixture_mean->mean_sparse_query({device_queries.get(), queries.size(), solid::DeviceMemory::cuda}, count);
            test_tiled::require(mean.elements == width, "CUDA reference retrieval query shape");
            std::vector<float> result(width);
            cuda_check(cudaMemcpy(result.data(), mean.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return result;
        });
        std::cout<<"CUDA tiled prefill device checks passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
