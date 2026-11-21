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
    bool interleaved = false;
    bool fuse_kv_projection = false;
    bool profile = false;
    bool prefetch = false;
    bool prefetch_trace = false;
    std::string schedule = "naive";
    bool schedule_trace = false;
    bool selection_trace = true;
    std::string selection_backend = "auto";
    std::string prefill_backend = "auto";
    std::uint32_t prefill_tile_tokens = 512;
    std::uint32_t prefetch_queue_depth = 64;
    int io_cpu=-1,coordinator_cpu=-1;
    std::uint32_t prefetch_slots_per_head = 0;
    std::uint32_t block_tokens = 32;
    std::uint32_t context_budget_tokens = 0;
    std::uint32_t init_tokens = 0;
    std::uint32_t local_tokens = 0;
    std::uint32_t representative_count = 1;
    std::uint32_t write_buffer_bytes_per_layer = 32768;
    std::filesystem::path session_file;
    std::filesystem::path selection_replay;
    bool keep_session_file = false;
    bool direct_decode_reads = false;
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
    std::string kv_dtype = "f16";
    std::vector<std::int32_t> prompt_token_ids;
    AttentionConfig attention;
    bool dump_logits = false;
    bool replay = false;
    std::vector<std::int32_t> replay_token_ids;

    static RunConfig load(const std::filesystem::path & path);
};

} // namespace solid
