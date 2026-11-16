#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace solid {

struct ModelFile {
    std::filesystem::path path;
    std::string sha256;
};

struct RunConfig {
    std::filesystem::path model;
    std::string model_sha256;
    std::vector<ModelFile> additional_model_files;
    std::filesystem::path prompt;
    std::filesystem::path output;
    std::uint32_t context_tokens = 4096;
    std::uint32_t batch_tokens = 512;
    std::uint32_t microbatch_tokens = 128;
    std::uint32_t max_new_tokens = 512;
    int threads = 4;
    int gpu_layers = 99;
    bool flash_attention = true;
    bool kv_on_gpu = true;
    bool ignore_eos = false;

    static RunConfig load(const std::filesystem::path & path);
};

} // namespace solid
