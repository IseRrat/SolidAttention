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

struct AttentionConfig {
    bool enabled = false;
    bool in_memory = false;
    bool sparse = false;
    bool gpu_decode = true;
    std::uint32_t block_tokens = 32;
    std::uint32_t context_budget_tokens = 0;
    std::uint32_t init_tokens = 0;
    std::uint32_t local_tokens = 0;
    std::uint32_t representative_count = 1;
    std::uint32_t write_buffer_bytes_per_layer = 32768;
    std::filesystem::path session_file;
    bool keep_session_file = false;
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
    AttentionConfig attention;
    bool dump_logits = false;
    bool replay = false;
    std::vector<std::int32_t> replay_token_ids;

    static RunConfig load(const std::filesystem::path & path);
};

} // namespace solid
