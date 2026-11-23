#include "tiled_prefill_cases.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <string>

namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
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
        cuda_check(cudaMemcpy(device_q.get(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device_kv.get(), current.data(), current.size() * sizeof(solid::Half), cudaMemcpyHostToDevice));
        const solid::DeviceQuery query{device_q.get(), q.size(), solid::DeviceMemory::cuda};
        const auto mean = backend->mean_sparse_query(query, queries);
        std::vector<float> host_mean(mean.elements), expected_mean(mean.elements);
        cuda_check(cudaMemcpy(host_mean.data(), mean.data, mean.elements * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::uint32_t i = 0; i < queries; ++i) for (std::size_t d = 0; d < mean.elements; ++d) {
            expected_mean[d] += q[i * mean.elements + d] / queries;
        }
        test_tiled::close(host_mean, expected_mean, 1e-6f);
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
    auto ranker = solid::make_cuda_block_ranker(scoring_geometry, 2, 1, budget.total, device, 0, 1);
    test_tiled::require(ranker->configure_probability_builder(budget.initial, budget.local, budget.representatives),
                        "probability builder configuration");
    for (std::uint64_t first = 0; first < 800;) {
        const std::uint32_t count = first < 768 ? 128 : 1;
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
            reference.observe(0, position, std::vector<float>(128, 0.f), keys);
            const auto recent = std::min<std::uint64_t>(position + 1, budget.local);
            std::vector<float> probabilities(recent * 2);
            for (std::size_t h = 0; h < 2; ++h) for (std::uint64_t j = 0; j < recent; ++j) {
                const auto value = static_cast<float>((position + j * (h + 1)) % 7 + 1) / 1024.f;
                probabilities[h * recent + j] = value;
                const auto key = position + 1 - recent + j - origin;
                scores[count == 1 ? h * recent + j : key * 2 + h] += value;
            }
            reference.observe_probabilities(0, position, probabilities);
        }
        auto device_kv = device_buffer<solid::Half>(kv.size());
        auto device_scores = device_buffer<float>(scores.size());
        cuda_check(cudaMemcpy(device_kv.get(), kv.data(), kv.size() * sizeof(solid::Half), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device_scores.get(), scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
        const auto built = ranker->probability_update({0, first, count, origin,
            {device_kv.get(), kv.size(), solid::DeviceMemory::cuda},
            {device_scores.get(), scores.size(), solid::DeviceMemory::cuda}, count != 1});
        for (const auto & block : built) for (std::uint32_t h = 0; h < 2; ++h) {
            const auto expected = reference.representative_positions(0, h, block.block);
            test_tiled::require(std::equal(expected.begin(), expected.end(), block.positions.begin() + h * budget.representatives),
                                "GPU probability representatives differ from InfLLM reference");
        }
        first = end;
    }
}
}

int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") {
        std::cout<<"Usage: solid-cuda-prefill-test --run-device-check [device ordinal]\n"; return argc==1?0:2;
    }
    try {
        if (argc>3) throw std::invalid_argument("unexpected device-check argument");
        const auto device=argc==3?std::stoi(argv[2]):0;
        const auto factory = [&](auto g, auto qh, auto queries, auto tile) {
            return solid::make_cuda_tiled_prefill(g, qh, queries, tile, device);
        };
        test_tiled::run_cases(factory);
        test_tiled::frames(factory, 128, false, 32, 1e-3f);
        sparse_prefill(device);
        probability_representatives(device);
        std::cout<<"CUDA tiled prefill device checks passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
