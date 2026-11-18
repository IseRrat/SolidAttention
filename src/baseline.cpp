#include "solid/config.hpp"
#include "solid/filehash.hpp"
#include "solid/llama_adapter.hpp"
#include "solid/profile.hpp"
#include "solid_build_info.hpp"
#include "llama.h"
#include "llama-memory.h"
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
    if (adapter) adapter->after_process();
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
Json counters(const solid::SessionStats & s, const solid::DeviceCacheStats & d) {
    return {{"read_requests", s.io.read_requests + s.prefetch.io.read_requests},
        {"read_bytes", s.io.read_bytes + s.prefetch.io.read_bytes}, {"write_requests", s.io.write_requests},
        {"write_bytes", s.io.write_bytes}, {"io_read_ms", s.io.read_ns / 1e6}, {"io_write_ms", s.io.write_ns / 1e6},
        {"attention_queries", s.attention_queries}, {"selected_tokens", s.selected_tokens},
        {"append_ms", s.append_ns / 1e6}, {"selection_ms", s.selection_ns / 1e6},
        {"selection_rank_calls", s.ranking.calls}, {"selection_scored_blocks", s.ranking.scored_blocks},
        {"selection_published_blocks", s.ranking.published_blocks},
        {"representative_builder_queries", s.ranking.builder_queries}, {"representative_builder_appends", s.ranking.builder_appends},
        {"representative_builder_blocks", s.ranking.built_blocks},
        {"device_rank_queries",s.ranking.device_rank_queries},{"device_builder_queries",s.ranking.device_builder_queries},
        {"device_builder_appends",s.ranking.device_builder_appends},{"query_upload_bytes",s.ranking.query_upload_bytes},
        {"builder_key_upload_bytes",s.ranking.builder_key_upload_bytes},{"direct_decode_queries",s.direct_decode_queries},
        {"decode_kv_download_bytes",s.decode_kv_download_bytes},
        {"device_builder_batches",s.ranking.device_builder_batches},{"direct_prefill_queries",s.direct_prefill_queries},
        {"prefill_kv_download_bytes",s.prefill_kv_download_bytes},{"prefill_query_upload_bytes",s.prefill.query_upload_bytes},
        {"prefill_device_query_tokens",s.prefill.device_query_tokens},{"prefill_device_kv_tokens",s.prefill.device_kv_tokens},
        {"prefill_output_download_bytes",s.prefill.output_download_bytes},{"prefill_output_device_copy_bytes",s.prefill.output_device_copy_bytes},
        {"prefill_tiled_batches", s.prefill.batches}, {"prefill_tiled_queries", s.prefill.query_tokens},
        {"prefill_tiles", s.prefill.tiles}, {"prefill_kv_tokens", s.prefill.kv_tokens},
        {"prefill_upload_bytes", s.prefill.upload_bytes}, {"tiled_prefill_inclusive_ms", s.tiled_prefill_ns/1e6},
        {"gather_inclusive_ms", s.gather_ns / 1e6}, {"gather_copy_bytes", s.gather_copy_bytes},
        {"streaming_attention_inclusive_ms", s.streaming_attention_ns / 1e6},
        {"prefetch_wait_ms", s.prefetch.wait_ns / 1e6}, {"prefetch_resolve_ms", s.prefetch.resolve_ns / 1e6},
        {"device_uploads", d.uploads}, {"device_upload_bytes", d.upload_bytes},
        {"device_shared_host_copy_bytes", d.shared_host_copy_bytes},
        {"current_device_rows",d.current_device_rows},{"current_device_bytes",d.current_device_bytes},
        {"current_host_bytes",d.current_host_bytes},{"refreshed_device_tails",d.refreshed_tails},{"retained_device_prefix_bytes",d.retained_prefix_bytes}};
}
Json difference(const Json & after, const Json & before) {
    Json out = Json::object();
    for (const auto & entry : after.items()) {
        if (entry.value().is_number_unsigned()) out[entry.key()] = entry.value().get<std::uint64_t>() - before.at(entry.key()).get<std::uint64_t>();
        else out[entry.key()] = entry.value().get<double>() - before.at(entry.key()).get<double>();
    }
    return out;
}
struct BackendLifetime {
    BackendLifetime() { ggml_backend_load_all(); llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};
struct EvaluationBridge {
    solid::AttentionAdapter * adapter;
    solid::ExecutionProfile * profile;
    static bool evaluate(ggml_tensor * tensor, bool ask, void * userdata) {
        auto & self = *static_cast<EvaluationBridge *>(userdata);
        const bool a = self.adapter ? self.adapter->observe(tensor, ask) : !ask;
        const bool p = self.profile ? solid::ExecutionProfile::evaluate(tensor, ask, self.profile) : !ask;
        return ask ? a || p : a && p;
    }
};
struct AdapterContextLifetime {
    llama_context * context;
    solid::AttentionAdapter * adapter;
    ~AdapterContextLifetime() {
        if (adapter) {
            llama_synchronize(context);
            // quiesce drains device writes before reporting trace errors; retain
            // the original run error while the context is being unwound.
            try { adapter->quiesce(); } catch (...) {}
        }
    }
};
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "solid-baseline --config FILE\nsolid-baseline --validate-config FILE\nsolid-baseline --tokenize FILE\nsolid-baseline --build-info\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << solid_build_info << '\n';
            return 0;
        }
        if (argc != 3 || (std::string(argv[1]) != "--config" && std::string(argv[1]) != "--validate-config" && std::string(argv[1]) != "--tokenize")) {
            throw std::invalid_argument("use --config FILE, --validate-config FILE or --tokenize FILE");
        }
        const auto config = solid::RunConfig::load(argv[2]);
        if (std::string(argv[1]) == "--validate-config") return 0;
        const bool tokenize_only = std::string(argv[1]) == "--tokenize";
        if (solid::sha256_file(config.model) != config.model_sha256) throw std::runtime_error("GGUF SHA256 mismatch");
        for (const auto & part : config.additional_model_files) {
            if (solid::sha256_file(part.path) != part.sha256) throw std::runtime_error("GGUF shard SHA256 mismatch: " + part.path.string());
        }
        if (!tokenize_only && std::filesystem::exists(config.output)) throw std::runtime_error("output already exists");
        if (std::filesystem::equivalent(config.model, config.prompt)) throw std::runtime_error("model and prompt must differ");
        const auto prompt = read_file(config.prompt);
        if (prompt.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("prompt exceeds tokenizer input limit");
        }
        const auto request_start = Clock::now();
        BackendLifetime backend;
        solid::ExecutionProfile profile(config.attention.profile);
        if (!tokenize_only && config.gpu_layers > 0 && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            throw std::runtime_error("GPU layers requested but no GPU backend is available");
        }
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = tokenize_only ? 0 : config.gpu_layers;
        mp.vocab_only = tokenize_only;
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
        std::vector<llama_token> tokens(config.prompt_token_ids.begin(),config.prompt_token_ids.end());
        if (tokenize_only || tokens.empty()) {
            int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
            if (n >= 0 || n == std::numeric_limits<int>::min()) throw std::runtime_error("tokenizer sizing failed");
            tokens.resize(static_cast<std::size_t>(-n));
            n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), true, true);
            if (n <= 0) throw std::runtime_error("tokenization failed or prompt is empty");
            tokens.resize(static_cast<std::size_t>(n));
        }
        if (tokenize_only) {
            Json parts = Json::array({config.model_sha256});
            for (const auto & part : config.additional_model_files) parts.push_back(part.sha256);
            std::cout << Json({{"schema_version", 1}, {"model_parts_sha256", parts}, {"source_sha256", solid::sha256_file(config.prompt)},
                {"add_special", true}, {"parse_special", true}, {"token_ids", tokens}, {"build_info", Json::parse(solid_build_info)}}).dump() << '\n';
            return 0;
        }
        for (const auto token : tokens) if (token >= llama_vocab_n_tokens(vocab)) throw std::invalid_argument("prompt token ID exceeds model vocabulary");
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
        cp.type_k = cp.type_v = config.kv_dtype == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
        cp.flash_attn_type = config.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = config.kv_on_gpu;
        cp.no_perf = false;
        std::unique_ptr<solid::AttentionAdapter> adapter;
        const auto adapter_start = Clock::now();
        if (config.attention.enabled) {
            std::filesystem::create_directories(config.output.parent_path());
            adapter = std::make_unique<solid::AttentionAdapter>(model.get(), config.attention, config.context_tokens,
                       tokens.size(), config.output.parent_path() / "selection.jsonl",config.microbatch_tokens);
        }
        const double adapter_ms = milliseconds(adapter_start);
        EvaluationBridge evaluation{adapter.get(), config.attention.profile ? &profile : nullptr};
        if (config.attention.profile || config.attention.prefetch) {
            cp.cb_eval = EvaluationBridge::evaluate;
            cp.cb_eval_user_data = &evaluation;
        }
        const auto context_start = Clock::now();
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context initialization failed");
        AdapterContextLifetime adapter_context{ctx.get(), adapter.get()};
        const double context_ms = milliseconds(context_start);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("batch allocation failed");
        const auto prefill_start = Clock::now();
        for (std::size_t first = 0; first < tokens.size();) {
            const auto count = std::min<std::size_t>(config.batch_tokens, tokens.size() - first);
            submit(ctx.get(), batch.get(), tokens.data() + first, count, static_cast<llama_pos>(first), first + count == tokens.size(), adapter.get());
            profile.check();
            first += count;
        }
        if (adapter) adapter->flush();
        const double prefill_ms = milliseconds(prefill_start);
        const auto prefill_counters = adapter ? counters(adapter->stats(), adapter->device_stats()) : Json::object();
        profile.set_decode();
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
        std::string raw_text;
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
                profile.check();
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
            raw_text += bytes;
            if (!eos && !llama_vocab_is_control(vocab, token)) text += bytes;
            const double total_ms = milliseconds(start);
            if (i > 0) decode_ms += total_ms;
            trace.push_back({{"index", i}, {"token_id", token}, {"source", i == 0 ? "prefill" : "decode"},
                             {"process_ms", process_ms}, {"sample_ms", sample_ms}, {"total_ms", total_ms}, {"eos", eos}});
            if (eos && !config.ignore_eos && !config.replay) { stop = "eos"; break; }
        }
        if (config.replay) stop = "replay_length";
        if (config.dump_logits) { logits_file.close(); if (!logits_file) throw std::runtime_error("logits close failed"); }
        const auto decode_counters = adapter ? counters(adapter->stats(), adapter->device_stats()) : Json::object();
        const auto final_flush_start = Clock::now();
        if (adapter) adapter->finish();
        const double final_flush_ms = milliseconds(final_flush_start);
        const auto perf = llama_perf_context(ctx.get());
        Json result = {
            {"schema_version", 1}, {"upstream_commit", SOLID_UPSTREAM_COMMIT},
            {"build_info", Json::parse(solid_build_info)},
            {"model", config.model.string()}, {"model_sha256", config.model_sha256},
            {"model_metadata", metadata(model.get())}, {"prompt_token_ids", tokens}, {"generated_token_ids", generated},
            {"decode_input_token_ids", decode_inputs}, {"replay", config.replay},
            {"text", text}, {"raw_text", raw_text}, {"stop_reason", stop}, {"trace", trace}, {"kv_dtype", config.kv_dtype},
            {"prompt_source", config.prompt_token_ids.empty() ? "tokenized_text" : "token_ids"},
            {"model_shape", {{"layers", llama_model_n_layer(model.get())}, {"query_heads", llama_model_n_head(model.get())},
                             {"kv_heads", llama_model_n_head_kv(model.get())}, {"embedding", llama_model_n_embd(model.get())}}},
            {"effective_context_tokens", llama_n_ctx(ctx.get())},
            {"timing", {{"model_load_ms", load_ms}, {"context_init_ms", context_ms}, {"prefill_ms", prefill_ms},
                        {"adapter_init_ms", adapter_ms},
                        {"decode_loop_ms", decode_ms}, {"decode_steps", generated.size() - 1},
                        {"final_flush_ms", final_flush_ms},
                        {"first_token_ms", prefill_ms + trace.front().at("total_ms").get<double>()},
                        {"request_ms", milliseconds(request_start)}}},
            {"llama_counters", {{"prefill_tokens", perf.n_p_eval}, {"decode_tokens", perf.n_eval},
                                {"prefill_ms", perf.t_p_eval_ms}, {"decode_ms", perf.t_eval_ms}}}
        };
        result["model_parts_sha256"] = Json::array({config.model_sha256});
        result["memory"] = {{"native_memory_buffers", Json::array()}, {"kv_system_tracked_bytes", 0}};
        if (!adapter) {
            std::uint64_t bytes = 0;
            if (const auto * memory = llama_get_memory(ctx.get())) for (const auto & entry : memory->memory_breakdown()) {
                bytes += entry.second;
                result["memory"]["native_memory_buffers"].push_back({{"type", ggml_backend_buft_name(entry.first)},
                    {"bytes", entry.second}, {"host", ggml_backend_buft_is_host(entry.first)}});
            }
            result["memory"]["kv_system_tracked_bytes"] = bytes;
        }
        for (const auto & part : config.additional_model_files) result["model_parts_sha256"].push_back(part.sha256);
        if (config.dump_logits) {
            result["logits"] = {{"file", "logits.f32"}, {"dtype", "float32-le"}, {"rows", generated.size()},
                                {"columns", llama_vocab_n_tokens(vocab)}, {"sha256", solid::sha256_file(logits_path)}};
        }
        if (adapter) {
            const auto stats = adapter->stats();
            const auto budget = adapter->budget();
            const auto device = adapter->device_stats();
            const auto staging = adapter->schedule_stats().staging_bytes;
            result["memory"]["kv_system_tracked_bytes"] = stats.kv_resident_bytes + stats.representative_bytes +
                stats.prefetch.host_bytes + device.resident_bytes + staging + stats.ranking.device_bytes + stats.ranking.host_bytes +
                stats.prefill.kv_device_bytes + stats.prefill_host_bytes;
            result["memory"]["custom_buffers"] = {{"store_bytes", stats.kv_resident_bytes}, {"representatives_bytes", stats.representative_bytes},
                {"prefetch_host_bytes", stats.prefetch.host_bytes}, {"device_kv_bytes", device.resident_bytes}, {"copy_staging_bytes", staging},
                {"selection_device_bytes", stats.ranking.device_bytes}, {"selection_host_bytes", stats.ranking.host_bytes},
                {"prefill_kv_device_bytes", stats.prefill.kv_device_bytes}, {"prefill_kv_host_bytes", stats.prefill_host_bytes},
                {"workspace_tracked_peak_bytes", stats.workspace_peak_bytes}, {"extra_projection_weight_bytes", adapter->projection_buffer_bytes()}};
            result["memory"]["selection_device_breakdown"] = {{"representative_builder_bytes", stats.ranking.builder_device_bytes},
                {"ranking_and_bank_bytes", stats.ranking.device_bytes-stats.ranking.builder_device_bytes}};
            result["memory"]["prefill_workspace"]={{"device_bytes",stats.prefill.workspace_device_bytes},
                {"device_peak_bytes",stats.prefill.workspace_device_peak_bytes},{"kv_device_peak_bytes",stats.prefill.kv_device_peak_bytes},
                {"kv_host_peak_bytes",stats.prefill_host_peak_bytes},{"graph_output_device_bytes",stats.prefill_graph_device_bytes},
                {"graph_output_device_peak_bytes",stats.prefill_graph_device_peak_bytes}};
            result["attention_phases"] = {{"prefill", prefill_counters}, {"decode", difference(decode_counters, prefill_counters)},
                {"final_flush", difference(counters(stats, device), decode_counters)}};
            result["attention"] = {
                {"storage", config.attention.in_memory ? "memory" : "ssd"}, {"sparse", config.attention.sparse},
                {"layout", config.attention.interleaved ? "interleaved" : "separate"},
                {"fuse_kv_projection", config.attention.fuse_kv_projection},
                {"prefetch", config.attention.prefetch},
                {"selection_trace_enabled", config.attention.selection_trace},
                {"schedule", config.attention.schedule},
                {"projection_extra_weight_bytes", adapter->projection_buffer_bytes()},
                {"projection_initialization_ms", adapter->projection_initialization_ns() / 1e6},
                {"projection_weight_types", adapter->projection_weight_types()},
                {"selection_source", config.attention.selection_replay.empty() ? "dynamic" : "replay"},
                {"selection_backend", stats.selection_backend}, {"selection_backend_requested", config.attention.selection_backend},
                {"selection_rank_calls", stats.ranking.calls}, {"selection_scored_blocks", stats.ranking.scored_blocks},
                {"selection_published_blocks", stats.ranking.published_blocks},
                {"representative_backend", stats.representative_backend}, {"representative_builder_queries", stats.ranking.builder_queries},
                {"representative_builder_appends", stats.ranking.builder_appends}, {"representative_builder_blocks", stats.ranking.built_blocks},
                {"device_rank_queries",stats.ranking.device_rank_queries},{"device_builder_queries",stats.ranking.device_builder_queries},
                {"device_builder_appends",stats.ranking.device_builder_appends},{"query_upload_bytes",stats.ranking.query_upload_bytes},
                {"builder_key_upload_bytes",stats.ranking.builder_key_upload_bytes},{"direct_decode_queries",stats.direct_decode_queries},
                {"decode_kv_download_bytes",stats.decode_kv_download_bytes},
                {"device_builder_batches",stats.ranking.device_builder_batches},{"direct_prefill_queries",stats.direct_prefill_queries},
                {"prefill_kv_download_bytes",stats.prefill_kv_download_bytes},{"prefill_query_upload_bytes",stats.prefill.query_upload_bytes},
                {"prefill_device_query_tokens",stats.prefill.device_query_tokens},{"prefill_device_kv_tokens",stats.prefill.device_kv_tokens},
                {"prefill_output_download_bytes",stats.prefill.output_download_bytes},{"prefill_output_device_copy_bytes",stats.prefill.output_device_copy_bytes},
                {"block_tokens", config.attention.block_tokens}, {"context_budget_tokens", budget.total},
                {"init_tokens", budget.initial}, {"local_tokens", budget.local}, {"representative_count", budget.representatives},
                {"head_aggregation", "mean"}, {"position_policy", "absolute_native_rope"},
                {"attention_backend", config.attention.gpu_decode ? "ggml_flash_decode+"+stats.prefill_backend : "cpu_streaming_reference"},
                {"prefill_backend",stats.prefill_backend},{"prefill_backend_requested",config.attention.prefill_backend},
                {"prefill_query_capacity",stats.prefill.query_capacity},{"prefill_tile_capacity",stats.prefill.tile_capacity},
                {"io_mode", "buffered"},
                {"attention_queries", stats.attention_queries}, {"selected_tokens", stats.selected_tokens},
                {"workspace_tracked_peak_bytes", stats.workspace_peak_bytes}, {"kv_buffer_bytes", stats.kv_resident_bytes},
                {"representative_buffer_bytes", stats.representative_bytes},
                {"read_requests", stats.io.read_requests + stats.prefetch.io.read_requests},
                {"read_bytes", stats.io.read_bytes + stats.prefetch.io.read_bytes},
                {"write_requests", stats.io.write_requests}, {"write_bytes", stats.io.write_bytes},
                {"io_read_ms", stats.io.read_ns / 1e6}, {"io_write_ms", stats.io.write_ns / 1e6},
                {"append_ms", stats.append_ns / 1e6}, {"selection_ms", stats.selection_ns / 1e6},
                {"gather_inclusive_ms", stats.gather_ns / 1e6}, {"gather_copy_bytes", stats.gather_copy_bytes},
                {"streaming_attention_inclusive_ms", stats.streaming_attention_ns / 1e6}
            };
            if (config.attention.prefetch) {
                result["prefetch"] = {
                    {"io_engine", "liburing"}, {"queue_depth", config.attention.prefetch_queue_depth},
                    {"io_execution",stats.prefetch.io.dedicated_worker?"dedicated_thread":"caller_thread"},
                    {"io_cpu",stats.prefetch.io.worker_cpu},{"inference_cpus",adapter->inference_cpus()},
                    {"io_admission_capacity",stats.prefetch.io.worker_capacity},
                    {"worker_submissions",stats.prefetch.io.worker_submissions},{"queued_cancellations",stats.prefetch.io.queued_cancellations},
                    {"slots_per_head", stats.prefetch.slots_per_head},
                    {"trace_enabled", config.attention.prefetch_trace},
                    {"history_resolutions", stats.prefetch.history_resolutions}, {"formal_resolutions", stats.prefetch.formal_resolutions},
                    {"predicted_blocks", stats.prefetch.predicted_blocks}, {"ready_hits", stats.prefetch.ready_hits},
                    {"inflight_hits", stats.prefetch.inflight_hits}, {"missing_blocks", stats.prefetch.missing_blocks},
                    {"unused_blocks", stats.prefetch.unused_blocks}, {"refreshed_blocks", stats.prefetch.refreshed_blocks},
                    {"stale_completions", stats.prefetch.stale_completions}, {"host_buffer_bytes", stats.prefetch.host_bytes},
                    {"wait_ms", stats.prefetch.wait_ns / 1e6}, {"resolve_inclusive_ms", stats.prefetch.resolve_ns / 1e6},
                    {"async_read_requests", stats.prefetch.io.read_requests}, {"async_read_bytes", stats.prefetch.io.read_bytes},
                    {"async_write_requests", stats.prefetch.io.write_requests}, {"async_write_bytes", stats.prefetch.io.write_bytes},
                    {"completed_reads", stats.prefetch.io.completed_reads}, {"cancel_requests", stats.prefetch.io.cancel_requests},
                    {"cancel_completions", stats.prefetch.io.cancel_completions}, {"read_completion_latency_sum_ms", stats.prefetch.io.completion_ns / 1e6},
                    {"cancel_successes", stats.prefetch.io.cancel_successes}, {"cancel_not_found", stats.prefetch.io.cancel_not_found},
                    {"cancel_busy", stats.prefetch.io.cancel_busy}, {"cancel_errors", stats.prefetch.io.cancel_errors},
                    {"device_uploads", device.uploads}, {"device_upload_bytes", device.upload_bytes},
                    {"device_shared_host_copy_bytes", device.shared_host_copy_bytes}, {"device_shared_buffer_bytes", device.shared_resident_bytes},
                    {"current_device_rows",device.current_device_rows},{"current_device_bytes",device.current_device_bytes},
                    {"current_host_bytes",device.current_host_bytes},{"refreshed_device_tails",device.refreshed_tails},
                    {"retained_device_prefix_bytes",device.retained_prefix_bytes},
                    {"device_reused_blocks", device.reused_blocks}, {"device_overwritten_blocks", device.overwritten_blocks},
                    {"device_stale_events", device.stale_events}, {"device_buffer_bytes", device.resident_bytes}
                };
                const auto prefetch_trace = config.output.parent_path() / "prefetch.jsonl";
                if (config.attention.prefetch_trace) result["prefetch_trace_sha256"] = solid::sha256_file(prefetch_trace);
                // A one-output request completes in prefill and creates the device
                // cache only when a decode graph is built.
                const auto residency = config.output.parent_path() / "residency.jsonl";
                if (std::filesystem::exists(residency)) result["residency_trace_sha256"] = solid::sha256_file(residency);
                const auto schedule = adapter->schedule_stats();
                result["schedule"] = {
                    {"mode", config.attention.schedule}, {"trace_enabled", config.attention.schedule_trace},
                    {"frames", schedule.frames}, {"tasks", schedule.tasks}, {"worker_jobs", schedule.worker_jobs},
                    {"block_frames",schedule.block_frames},{"block_tasks",schedule.block_tasks},
                    {"block_observed_critical_path_sum_ms",schedule.block_critical_path_ns/1e6},
                    {"completion_backend",schedule.completion_backend},{"device_event_completions",schedule.device_event_completions},
                    {"device_attention_event_ms",schedule.device_attention_ns/1e6},{"device_boundary_fallbacks",schedule.device_boundary_fallbacks},
                    {"uploads_before_boundary",schedule.uploads_before_boundary},{"copy_event_queries",schedule.copy_event_queries},
                    {"coordinator_cpu",schedule.coordinator_cpu},
                    {"observed_critical_path_sum_ms", schedule.critical_path_ns / 1e6},
                    {"miss_projection_overlap_sum_ms", schedule.miss_projection_overlap_ns / 1e6},
                    {"query_reused_fences", schedule.query_reused_fences}, {"process_reused_fences", schedule.process_reused_fences},
                    {"attention_fences", schedule.attention_fences}, {"fallback_fences", schedule.fallback_fences},
                    {"copy_event_waits", schedule.event_waits}, {"copy_batch_waits", schedule.batch_waits},
                    {"copy_stream_waits", schedule.stream_waits}, {"copy_batch_groups", schedule.batch_groups},
                    {"copy_events_coalesced", schedule.events_coalesced}, {"backend_drains", schedule.backend_drains},
                    {"backend_compute_waits", schedule.backend_compute_waits}, {"staging_buffer_bytes", schedule.staging_bytes}
                };
                const auto schedule_trace = config.output.parent_path() / "schedule.jsonl";
                if (std::filesystem::exists(schedule_trace)) result["schedule_trace_sha256"] = solid::sha256_file(schedule_trace);
                const auto block_trace=config.output.parent_path()/"schedule.blocks.jsonl";
                if (std::filesystem::exists(block_trace)) result["block_schedule_trace_sha256"]=solid::sha256_file(block_trace);
            }
            if (config.attention.selection_trace) result["selection_trace_sha256"] = solid::sha256_file(config.output.parent_path() / "selection.jsonl");
            if (!config.attention.selection_replay.empty()) result["selection_replay_sha256"] = solid::sha256_file(config.attention.selection_replay);
        }
        if (config.attention.profile) {
            const auto serialize = [](const std::map<std::string, solid::ProfileTotal> & totals) {
                Json entries = Json::object();
                for (const auto & pair : totals) entries[pair.first] = {{"calls", pair.second.calls}, {"bytes", pair.second.bytes}, {"ms", pair.second.ns / 1e6}};
                return entries;
            };
            result["profile"] = {{"mode", "serialized_nodes_and_transfers"}, {"nodes", serialize(profile.nodes())},
                                 {"transfers", serialize(profile.transfers())}};
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
