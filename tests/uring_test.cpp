#include "solid/async_reader.hpp"
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
        std::cout << "liburing temporary-file reads, short reads, EOF, cancellation and drain passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
