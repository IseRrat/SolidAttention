#include "solid/async_reader.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        if (!solid::uring_available()) { std::cout << "liburing disabled for this build\n"; return 77; }
        const auto path = std::filesystem::temp_directory_path() / ("solid-uring-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        auto file = solid::create_session_file(path, true);
        file->resize(64);
        std::array<unsigned char, 64> expected{};
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<unsigned char>(i);
        if (file->write_at(expected.data(), expected.size(), 0) != expected.size()) throw std::runtime_error("test setup write");
        for (bool threaded:{false,true}) {
        auto reader = threaded ? solid::make_threaded_uring_reader(2) : solid::make_uring_reader(2);
        std::array<unsigned char, 32> a{}, b{};
        if (!reader->submit(1, file->native_handle(), a.data(), a.size(), 0) ||
            !reader->submit(2, file->native_handle(), b.data(), b.size(), 32) ||
            reader->submit(3, file->native_handle(), b.data(), b.size(), 0)) throw std::runtime_error("uring queue capacity");
        reader->flush();
        solid::ReadCompletion c{};
        for (int i = 0; i < 2; ++i) if (!reader->poll(c, true) || c.result != 32) throw std::runtime_error("uring read completion");
        for (std::size_t i = 0; i < 32; ++i) if (a[i] != expected[i] || b[i] != expected[i + 32]) throw std::runtime_error("uring payload");
        // The file ends halfway through this logical read: exercise short-read retry and EOF.
        reader->submit(4, file->native_handle(), a.data(), a.size(), 48);
        reader->flush();
        if (!reader->poll(c, true) || c.id != 4 || c.result >= 0) throw std::runtime_error("uring EOF accepted");
        reader->submit(5, file->native_handle(), a.data(), a.size(), 0);
        reader->flush(); reader->cancel(5); reader->flush();
        if (!reader->poll(c, true) || c.id != 5) throw std::runtime_error("cancel lost original completion");
        reader->drain();
        if (reader->stats().cancel_requests != reader->stats().cancel_completions ||
            reader->stats().completed_reads+reader->stats().queued_cancellations != 4) {
            throw std::runtime_error("uring cancellation drain accounting");
        }
        if (reader->stats().dedicated_worker!=threaded) throw std::runtime_error("uring owner thread identity");
        std::cout<<"threaded="<<threaded<<" READ completions="<<reader->stats().completed_reads
                 <<" queued cancellations="<<reader->stats().queued_cancellations<<'\n';
        }
#ifdef __linux__
        file->resize(4096);
        file->enable_direct_reads();
        std::array<unsigned char, 19> direct{};
        if (file->read_at(direct.data(),direct.size(),7)!=direct.size() ||
            !std::equal(direct.begin(),direct.end(),expected.begin()+7)) {
            throw std::runtime_error("unaligned synchronous direct read payload");
        }
        direct.fill(0);
        auto request=file->prepare_read(direct.data(),direct.size(),11);
        auto reader=solid::make_threaded_uring_reader(2);
        if (request.fd==file->native_handle() || !reader->submit(1,request.fd,request.buffer,request.bytes,request.offset)) {
            throw std::runtime_error("async direct read submission");
        }
        reader->flush();
        solid::ReadCompletion completion{};
        if (!reader->poll(completion,true) || completion.id!=1 ||
            request.finish(completion.result)!=static_cast<std::int64_t>(direct.size()) ||
            !std::equal(direct.begin(),direct.end(),expected.begin()+11)) {
            throw std::runtime_error("unaligned asynchronous direct read payload");
        }
        const solid::KvGeometry geometry{1, 2, 128, 32, 32, 4096, solid::KvLayout::separate, true};
        auto layer_path = path;
        layer_path += "-dense";
        solid::FileKvStore dense(geometry, solid::create_session_file(layer_path, true), 2048);
        for (std::uint64_t p = 0; p < 3; ++p) {
            std::vector<solid::Half> token(512);
            for (std::size_t i = 0; i < token.size(); ++i) token[i] = static_cast<solid::Half>(p * 512 + i + 1);
            dense.append_interleaved(0, p, token);
        }
        dense.enable_direct_reads();
        auto batch = solid::make_uring_reader(4);
        std::vector<solid::Half> planes(2 * 2 * 128 * 32);
        dense.read_dense_layer(0, 3, 32, planes.data(), batch.get());
        for (std::uint32_t kind = 0; kind < 2; ++kind) for (std::uint32_t h = 0; h < 2; ++h) {
            for (std::uint32_t p = 0; p < 3; ++p) for (std::uint32_t d = 0; d < 128; ++d) {
                if (planes[((kind * 2 + h) * 32 + p) * 128 + d] != p * 512 + h * 256 + kind * 128 + d + 1) {
                    throw std::runtime_error("batched layer read changed K/V order or pending rows");
                }
            }
        }
        if (dense.io_stats().read_requests != 4 || batch->stats().completed_reads != 4) {
            throw std::runtime_error("batched layer issued a read for pending K/V rows");
        }
        auto sparse_reads = solid::make_uring_reader(2);
        std::array<solid::Half, 128> key_0{}, value_0{}, key_1{}, value_1{}, pending_key{}, pending_value{};
        dense.read_separate_rows(0, {{1, 0, 1, key_0.data(), value_0.data()},
            {0, 1, 1, key_1.data(), value_1.data()}, {0, 2, 1, pending_key.data(), pending_value.data()}}, sparse_reads.get(), 2);
        if (sparse_reads->stats().completed_reads != 4 || key_0.front() != 257 || value_0.front() != 385 ||
            key_1.front() != 513 || value_1.front() != 641 || pending_key.front() != 1025 || pending_value.front() != 1153) {
            throw std::runtime_error("multi-batch sparse read changed K/V ranges or issued disk reads for pending rows");
        }
#endif
        std::cout << "liburing temporary-file reads, short reads, EOF, cancellation and drain passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
