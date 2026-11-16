#include "solid/config.hpp"
#include "solid/filehash.hpp"
#include "solid/llama_adapter.hpp"
#include "solid_build_info.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "logits require IEEE float32");
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
double milliseconds(Clock::time_point from) {
    return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
}
std::string read_file(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int>(buffer.size()), 0, true);
    if (size < 0) {
        buffer.resize(static_cast<std::size_t>(-size));
        size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int>(buffer.size()), 0, true);
    }
    if (size < 0) throw std::runtime_error("token_to_piece failed");
    return {buffer.data(), static_cast<std::size_t>(size)};
}
llama_token greedy(llama_context * ctx, int vocabulary) {
    const auto * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("logits unavailable");
    int best = -1;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocabulary; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logits");
        if (best < 0 || logits[i] > maximum) { best = i; maximum = logits[i]; }
    }
    if (best < 0) throw std::runtime_error("empty vocabulary");
    return static_cast<llama_token>(best);
}
void submit(llama_context * ctx, llama_batch_ext * batch, const llama_token * tokens,
            std::size_t count, llama_pos first, bool output, solid::AttentionAdapter * adapter = nullptr) {
    llama_batch_ext_clear(batch);
    for (std::size_t i = 0; i < count; ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos position = first + static_cast<llama_pos>(i);
        if (index < 0 || !llama_batch_ext_set_pos(batch, index, &position) ||
            !llama_batch_ext_set_output_logits(batch, index, output && i + 1 == count)) {
            throw std::runtime_error("batch construction failed");
        }
    }
    const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (status != 0) throw std::runtime_error("llama_process failed: " + std::to_string(status));
    llama_synchronize(ctx);
    if (adapter) adapter->check();
}
Json metadata(const llama_model * model) {
    Json result = Json::object();
    for (int i = 0; i < llama_model_meta_count(model); ++i) {
        std::vector<char> key(1024), value(4096);
        int n = llama_model_meta_key_by_index(model, i, key.data(), key.size());
        if (n < 0) throw std::runtime_error("metadata key read failed");
        if (static_cast<std::size_t>(n) >= key.size()) {
            key.resize(static_cast<std::size_t>(n) + 1);
            llama_model_meta_key_by_index(model, i, key.data(), key.size());
        }
        n = llama_model_meta_val_str_by_index(model, i, value.data(), value.size());
        if (n < 0) throw std::runtime_error("metadata value read failed");
        if (static_cast<std::size_t>(n) >= value.size()) {
            value.resize(static_cast<std::size_t>(n) + 1);
            llama_model_meta_val_str_by_index(model, i, value.data(), value.size());
        }
        result[key.data()] = value.data();
    }
    return result;
}
struct BackendLifetime {
    BackendLifetime() { ggml_backend_load_all(); llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "solid-baseline --config FILE\nsolid-baseline --validate-config FILE\nsolid-baseline --build-info\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << solid_build_info << '\n';
            return 0;
        }
        if (argc != 3 || (std::string(argv[1]) != "--config" && std::string(argv[1]) != "--validate-config")) {
            throw std::invalid_argument("use --config FILE or --validate-config FILE");
        }
        const auto config = solid::RunConfig::load(argv[2]);
        if (std::string(argv[1]) == "--validate-config") return 0;
        if (solid::sha256_file(config.model) != config.model_sha256) throw std::runtime_error("GGUF SHA256 mismatch");
        for (const auto & part : config.additional_model_files) {
            if (solid::sha256_file(part.path) != part.sha256) throw std::runtime_error("GGUF shard SHA256 mismatch: " + part.path.string());
        }
        if (std::filesystem::exists(config.output)) throw std::runtime_error("output already exists");
        if (std::filesystem::equivalent(config.model, config.prompt)) throw std::runtime_error("model and prompt must differ");
        const auto prompt = read_file(config.prompt);
        if (prompt.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("prompt exceeds tokenizer input limit");
        }
        const auto request_start = Clock::now();
        BackendLifetime backend;
        if (config.gpu_layers > 0 && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            throw std::runtime_error("GPU layers requested but no GPU backend is available");
        }
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = config.gpu_layers;
        const auto load_start = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(config.model.string().c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const double load_ms = milliseconds(load_start);
        if (llama_model_has_encoder(model.get())) throw std::runtime_error("runner requires a decoder-only model");
        const auto * vocab = llama_model_get_vocab(model.get());
        for (const auto token : config.replay_token_ids) {
            if (token >= llama_vocab_n_tokens(vocab)) throw std::invalid_argument("replay token ID exceeds model vocabulary");
        }
        int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
        if (n >= 0 || n == std::numeric_limits<int>::min()) throw std::runtime_error("tokenizer sizing failed");
        std::vector<llama_token> tokens(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                           static_cast<int32_t>(tokens.size()), true, true);
        if (n <= 0) throw std::runtime_error("tokenization failed or prompt is empty");
        tokens.resize(static_cast<std::size_t>(n));
        if (tokens.size() + static_cast<std::uint64_t>(config.max_new_tokens) - 1 > config.context_tokens) {
            throw std::runtime_error("prompt plus generated inputs exceed context_tokens");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = config.context_tokens;
        cp.n_batch = config.batch_tokens;
        cp.n_ubatch = config.microbatch_tokens;
        cp.n_seq_max = 1;
        cp.n_outputs_max = 1;
        cp.n_outputs_max_per_seq = 1;
        cp.n_threads = config.threads;
        cp.n_threads_batch = config.threads;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = config.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = config.kv_on_gpu;
        cp.no_perf = false;
        std::unique_ptr<solid::AttentionAdapter> adapter;
        if (config.attention.enabled) {
            std::filesystem::create_directories(config.output.parent_path());
            adapter = std::make_unique<solid::AttentionAdapter>(model.get(), config.attention, config.context_tokens,
                       tokens.size(), config.output.parent_path() / "selection.jsonl");
        }
        const auto context_start = Clock::now();
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context initialization failed");
        const double context_ms = milliseconds(context_start);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("batch allocation failed");
        const auto prefill_start = Clock::now();
        for (std::size_t first = 0; first < tokens.size();) {
            const auto count = std::min<std::size_t>(config.batch_tokens, tokens.size() - first);
            submit(ctx.get(), batch.get(), tokens.data() + first, count, static_cast<llama_pos>(first), first + count == tokens.size(), adapter.get());
            first += count;
        }
        if (adapter) adapter->flush();
        const double prefill_ms = milliseconds(prefill_start);
        std::ofstream logits_file;
        const auto logits_path = config.output.parent_path() / "logits.f32";
        if (config.dump_logits) {
            std::filesystem::create_directories(config.output.parent_path());
            if (std::filesystem::exists(logits_path)) throw std::runtime_error("logits output already exists");
            logits_file.open(logits_path, std::ios::binary);
            if (!logits_file) throw std::runtime_error("cannot create logits output");
        }
        Json trace = Json::array();
        std::vector<llama_token> generated;
        std::vector<llama_token> decode_inputs;
        std::string text;
        std::string stop = "length";
        double decode_ms = 0;
        for (std::uint32_t i = 0; i < config.max_new_tokens; ++i) {
            const auto start = Clock::now();
            double process_ms = 0;
            if (i > 0) {
                const auto processing = Clock::now();
                const llama_token input = config.replay ? config.replay_token_ids[i - 1] : generated.back();
                decode_inputs.push_back(input);
                submit(ctx.get(), batch.get(), &input, 1,
                       static_cast<llama_pos>(tokens.size() + i - 1), true, adapter.get());
                process_ms = milliseconds(processing);
            }
            const auto sample_start = Clock::now();
            const auto token = greedy(ctx.get(), llama_vocab_n_tokens(vocab));
            const double sample_ms = milliseconds(sample_start);
            if (config.dump_logits) {
                const auto * logits = llama_get_logits_ith(ctx.get(), -1);
                logits_file.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(llama_vocab_n_tokens(vocab)) * sizeof(float));
                if (!logits_file) throw std::runtime_error("logits write failed");
            }
            const bool eos = llama_vocab_is_eog(vocab, token);
            const auto bytes = piece(vocab, token);
            generated.push_back(token);
            text += bytes;
            const double total_ms = milliseconds(start);
            if (i > 0) decode_ms += total_ms;
            trace.push_back({{"index", i}, {"token_id", token}, {"source", i == 0 ? "prefill" : "decode"},
                             {"process_ms", process_ms}, {"sample_ms", sample_ms}, {"total_ms", total_ms}, {"eos", eos}});
            if (eos && !config.ignore_eos && !config.replay) { stop = "eos"; break; }
        }
        if (config.replay) stop = "replay_length";
        if (config.dump_logits) { logits_file.close(); if (!logits_file) throw std::runtime_error("logits close failed"); }
        const auto final_flush_start = Clock::now();
        if (adapter) adapter->flush();
        const double final_flush_ms = milliseconds(final_flush_start);
        const auto perf = llama_perf_context(ctx.get());
        Json result = {
            {"schema_version", 1}, {"upstream_commit", SOLID_UPSTREAM_COMMIT},
            {"build_info", Json::parse(solid_build_info)},
            {"model", config.model.string()}, {"model_sha256", config.model_sha256},
            {"model_metadata", metadata(model.get())}, {"prompt_token_ids", tokens}, {"generated_token_ids", generated},
            {"decode_input_token_ids", decode_inputs}, {"replay", config.replay},
            {"text", text}, {"stop_reason", stop}, {"trace", trace},
            {"model_shape", {{"layers", llama_model_n_layer(model.get())}, {"query_heads", llama_model_n_head(model.get())},
                             {"kv_heads", llama_model_n_head_kv(model.get())}, {"embedding", llama_model_n_embd(model.get())}}},
            {"effective_context_tokens", llama_n_ctx(ctx.get())},
            {"timing", {{"model_load_ms", load_ms}, {"context_init_ms", context_ms}, {"prefill_ms", prefill_ms},
                        {"decode_loop_ms", decode_ms}, {"decode_steps", generated.size() - 1},
                        {"final_flush_ms", final_flush_ms},
                        {"first_token_ms", prefill_ms + trace.front().at("total_ms").get<double>()},
                        {"request_ms", milliseconds(request_start)}}},
            {"llama_counters", {{"prefill_tokens", perf.n_p_eval}, {"decode_tokens", perf.n_eval},
                                {"prefill_ms", perf.t_p_eval_ms}, {"decode_ms", perf.t_eval_ms}}}
        };
        result["model_parts_sha256"] = Json::array({config.model_sha256});
        for (const auto & part : config.additional_model_files) result["model_parts_sha256"].push_back(part.sha256);
        if (config.dump_logits) {
            result["logits"] = {{"file", "logits.f32"}, {"dtype", "float32-le"}, {"rows", generated.size()},
                                {"columns", llama_vocab_n_tokens(vocab)}, {"sha256", solid::sha256_file(logits_path)}};
        }
        if (adapter) {
            const auto stats = adapter->stats();
            const auto budget = adapter->budget();
            result["attention"] = {
                {"storage", config.attention.in_memory ? "memory" : "ssd"}, {"sparse", config.attention.sparse},
                {"block_tokens", config.attention.block_tokens}, {"context_budget_tokens", budget.total},
                {"init_tokens", budget.initial}, {"local_tokens", budget.local}, {"representative_count", budget.representatives},
                {"head_aggregation", "mean"}, {"position_policy", "absolute_native_rope"},
                {"attention_backend", config.attention.gpu_decode ? "ggml_flash_decode+cpu_streaming_prefill" : "cpu_streaming_reference"},
                {"io_mode", "buffered"},
                {"attention_queries", stats.attention_queries}, {"selected_tokens", stats.selected_tokens},
                {"workspace_tracked_peak_bytes", stats.workspace_peak_bytes}, {"kv_buffer_bytes", stats.kv_resident_bytes},
                {"representative_buffer_bytes", stats.representative_bytes},
                {"read_requests", stats.io.read_requests}, {"read_bytes", stats.io.read_bytes},
                {"write_requests", stats.io.write_requests}, {"write_bytes", stats.io.write_bytes}
            };
        }
        std::filesystem::create_directories(config.output.parent_path());
        const auto temporary = config.output.string() + ".tmp";
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file << result.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
        file.close();
        if (!file) throw std::runtime_error("result write failed");
        std::filesystem::rename(temporary, config.output);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "solid-baseline: " << error.what() << '\n';
        return 1;
    }
}
