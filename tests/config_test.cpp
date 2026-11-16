#include "solid/config.hpp"
#include "solid/filehash.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("solid-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    const auto config = root / "config.json";
    const std::string prefix = "{\"schema_version\":1,\"sampling\":\"greedy\",\"kv_dtype\":\"f16\","
        "\"model\":\"weights.gguf\",\"model_sha256\":\"" + std::string(64, 'a') + "\","
        "\"prompt\":\"prompt.txt\",\"output\":\"out.json\",\"context_tokens\":1024,"
        "\"batch_tokens\":128,\"microbatch_tokens\":32,\"max_new_tokens\":16,"
        "\"threads\":4,\"gpu_layers\":0,\"flash_attention\":false,\"kv_on_gpu\":false,\"ignore_eos\":false";
    auto write = [&](const std::string & content) { std::ofstream(config) << content; };
    auto reject = [&](const std::string & content) {
        write(content);
        bool failed = false;
        try { (void) solid::RunConfig::load(config); } catch (const std::exception &) { failed = true; }
        if (!failed) throw std::runtime_error("invalid configuration accepted");
    };
    try {
        write("abc");
        if (solid::sha256_file(config) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
            throw std::runtime_error("file SHA256 differs from known vector");
        }
        write(std::string(1000000, 'a'));
        if (solid::sha256_file(config) != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") {
            throw std::runtime_error("streaming SHA256 differs from known vector");
        }
        write(prefix + "}");
        const auto parsed = solid::RunConfig::load(config);
        if (parsed.model != root / "weights.gguf" || parsed.batch_tokens != 128) {
            throw std::runtime_error("configuration path or numeric parsing failed");
        }
        write(prefix + ",\"attention\":{\"storage\":\"ssd\",\"sparse\":true,\"context_budget_tokens\":4,\"init_tokens\":1,\"local_tokens\":1}}");
        const auto sparse = solid::RunConfig::load(config);
        if (!sparse.attention.enabled || sparse.attention.context_budget_tokens != 4 || sparse.attention.session_file != root / "kv.cache") {
            throw std::runtime_error("attention configuration parsing failed");
        }
        reject(prefix + ",\"unrecognized\":true}");
        reject(prefix + ",\"threads\":8}");
        reject(prefix + ",\"replay_token_ids\":[0]}");
        reject(prefix + ",\"attention\":{\"storage\":\"ssd\",\"sparse\":true,\"context_budget_tokens\":4,\"init_tokens\":4,\"local_tokens\":1}}");
        reject(prefix + ",\"attention\":{\"storage\":\"ssd\",\"sparse\":true,\"unexpected\":1}}");
        auto change = [&](const std::string & from, const std::string & to) {
            auto content = prefix + "}";
            content.replace(content.find(from), from.size(), to);
            reject(content);
        };
        change("\"threads\":4", "\"threads\":0");
        change("\"threads\":4", "\"threads\":1.5");
        change("\"gpu_layers\":0", "\"gpu_layers\":-1");
        change("\"context_tokens\":1024", "\"context_tokens\":16");
        change("\"microbatch_tokens\":32", "\"microbatch_tokens\":256");
        change(std::string(64, 'a'), "invalid");
        change("\"kv_dtype\":\"f16\"", "\"kv_dtype\":\"q4_0\"");
        change("weights.gguf", "weights-00001-of-00002.gguf");
        reject("[]");
        std::filesystem::remove(config);
        std::filesystem::remove(root);
        std::cout << "config checks passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::filesystem::remove(config);
        std::filesystem::remove(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
