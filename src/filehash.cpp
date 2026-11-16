#include "solid/filehash.hpp"
extern "C" {
#include "sha256.h"
}

#include <array>
#include <fstream>
#include <stdexcept>

namespace solid {
std::string sha256_file(const std::filesystem::path & path) {
    const auto size = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash file: " + path.string());
    sha256_t context;
    sha256_init(&context);
    std::array<unsigned char, 65536> buffer{};
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        sha256_update(&context, buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("file hash read failed: " + path.string());
    if (size != std::filesystem::file_size(path) || modified != std::filesystem::last_write_time(path)) {
        throw std::runtime_error("file changed during hashing: " + path.string());
    }
    unsigned char bytes[SHA256_DIGEST_SIZE];
    sha256_final(&context, bytes);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto byte : bytes) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}
}
