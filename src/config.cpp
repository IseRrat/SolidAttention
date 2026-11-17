#include "solid/config.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <vector>

namespace solid {
namespace {
using nlohmann::json;

std::uint32_t positive(const json & object, const char * key) {
    const auto & v = object.at(key);
    if (!v.is_number_integer() || v.get<std::int64_t>() <= 0 ||
        v.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(key) + " must be an integer in [1, INT32_MAX]");
    }
    return v.get<std::uint32_t>();
}

std::filesystem::path resolve(const std::filesystem::path & base, const std::string & value) {
    if (value.empty()) {
        throw std::invalid_argument("model, prompt and output paths must be set");
    }
    return std::filesystem::absolute(base / value).lexically_normal();
}
} // namespace

RunConfig RunConfig::load(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open config: " + path.string());
    }
    std::vector<std::set<std::string>> object_keys;
    const auto j = json::parse(input, [&](int, json::parse_event_t event, json & parsed) {
        if (event == json::parse_event_t::object_start) object_keys.emplace_back();
        if (event == json::parse_event_t::key && !object_keys.back().insert(parsed.get<std::string>()).second) {
            throw std::invalid_argument("duplicate config key: " + parsed.get<std::string>());
        }
        if (event == json::parse_event_t::object_end) object_keys.pop_back();
        return true;
    });
    const std::set<std::string> keys = {
        "schema_version", "model", "model_sha256", "additional_model_files", "prompt", "output", "context_tokens",
        "batch_tokens", "microbatch_tokens", "max_new_tokens", "threads", "gpu_layers",
        "flash_attention", "kv_on_gpu", "ignore_eos", "sampling", "kv_dtype", "attention", "dump_logits", "replay_token_ids"
    };
    if (!j.is_object()) {
        throw std::invalid_argument("config must be an object");
    }
    for (const auto & item : j.items()) {
        if (keys.count(item.key()) == 0) {
            throw std::invalid_argument("unknown config key: " + item.key());
        }
    }
    if (j.at("schema_version") != 1 || j.at("sampling") != "greedy" || j.at("kv_dtype") != "f16") {
        throw std::invalid_argument("expected schema_version=1, sampling=greedy and kv_dtype=f16");
    }
    RunConfig result;
    const auto base = std::filesystem::absolute(path).parent_path();
    result.model = resolve(base, j.at("model").get<std::string>());
    result.prompt = resolve(base, j.at("prompt").get<std::string>());
    result.output = resolve(base, j.at("output").get<std::string>());
    result.model_sha256 = j.at("model_sha256").get<std::string>();
    const auto validate_sha = [](const std::string & value) {
        if (value.size() != 64 || !std::all_of(value.begin(), value.end(),
            [](unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })) {
            throw std::invalid_argument("SHA256 must contain 64 lowercase hexadecimal characters");
        }
    };
    validate_sha(result.model_sha256);
    std::set<std::filesystem::path> model_paths{result.model};
    if (j.contains("additional_model_files")) {
        if (!j.at("additional_model_files").is_array()) throw std::invalid_argument("additional_model_files must be an array");
        for (const auto & part : j.at("additional_model_files")) {
            if (!part.is_object() || part.size() != 2 || !part.contains("path") || !part.contains("sha256")) {
                throw std::invalid_argument("each model part requires path and sha256");
            }
            ModelFile file{resolve(base, part.at("path").get<std::string>()), part.at("sha256").get<std::string>()};
            validate_sha(file.sha256);
            if (!model_paths.insert(file.path).second) throw std::invalid_argument("duplicate model part path");
            result.additional_model_files.push_back(std::move(file));
        }
    }
    std::smatch match;
    const auto name = result.model.filename().string();
    if (std::regex_match(name, match, std::regex("(.+)-([0-9]{5})-of-([0-9]{5})\\.gguf"))) {
        const int count = std::stoi(match[3].str());
        if (std::stoi(match[2].str()) != 1 || count < 1 || model_paths.size() != static_cast<std::size_t>(count)) {
            throw std::invalid_argument("declare every GGUF shard and select the first shard as model");
        }
        for (int i = 1; i <= count; ++i) {
            const auto digits = std::to_string(i);
            const auto shard = match[1].str() + "-" + std::string(5 - digits.size(), '0') + digits + "-of-" + match[3].str() + ".gguf";
            if (model_paths.count(result.model.parent_path() / shard) == 0) {
                throw std::invalid_argument("missing declared GGUF shard: " + shard);
            }
        }
    } else if (!result.additional_model_files.empty()) {
        throw std::invalid_argument("additional_model_files requires standard GGUF shard names");
    }
    result.context_tokens = positive(j, "context_tokens");
    result.batch_tokens = positive(j, "batch_tokens");
    result.microbatch_tokens = positive(j, "microbatch_tokens");
    result.max_new_tokens = positive(j, "max_new_tokens");
    result.threads = static_cast<int>(positive(j, "threads"));
    if (!j.at("gpu_layers").is_number_integer() || j.at("gpu_layers").get<std::int64_t>() < 0 ||
        j.at("gpu_layers").get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("gpu_layers must be in [0, INT_MAX]");
    }
    result.gpu_layers = j.at("gpu_layers").get<int>();
    result.flash_attention = j.at("flash_attention").get<bool>();
    result.kv_on_gpu = j.at("kv_on_gpu").get<bool>();
    result.ignore_eos = j.at("ignore_eos").get<bool>();
    if (j.contains("dump_logits")) result.dump_logits = j.at("dump_logits").get<bool>();
    if (j.contains("replay_token_ids")) {
        const auto & ids = j.at("replay_token_ids");
        if (!ids.is_array() || ids.size() != result.max_new_tokens - 1) throw std::invalid_argument("replay_token_ids requires max_new_tokens - 1 entries");
        result.replay = true;
        for (const auto & id : ids) {
            if (!id.is_number_integer() || id.get<std::int64_t>() < 0 ||
                id.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("replay token ID must be a nonnegative int32");
            }
            result.replay_token_ids.push_back(id.get<std::int32_t>());
        }
    }
    if (j.contains("attention")) {
        const auto & a = j.at("attention");
        const std::set<std::string> allowed = {"storage", "sparse", "gpu_decode", "layout", "fuse_kv_projection", "profile", "block_tokens", "context_budget_tokens",
            "init_tokens", "local_tokens", "representative_count", "write_buffer_bytes_per_layer", "session_file", "keep_session_file", "selection_replay",
            "prefetch", "prefetch_trace", "prefetch_queue_depth", "prefetch_slots_per_head"};
        if (!a.is_object()) throw std::invalid_argument("attention must be an object");
        for (const auto & item : a.items()) if (allowed.count(item.key()) == 0) throw std::invalid_argument("unknown attention key: " + item.key());
        auto & target = result.attention;
        target.enabled = true;
        const auto storage = a.at("storage").get<std::string>();
        if (storage != "memory" && storage != "ssd") throw std::invalid_argument("attention storage must be memory or ssd");
        target.in_memory = storage == "memory";
        target.sparse = a.at("sparse").get<bool>();
        if (a.contains("layout")) {
            const auto layout = a.at("layout").get<std::string>();
            if (layout != "separate" && layout != "interleaved") throw std::invalid_argument("layout must be separate or interleaved");
            target.interleaved = layout == "interleaved";
        }
        if (a.contains("fuse_kv_projection")) target.fuse_kv_projection = a.at("fuse_kv_projection").get<bool>();
        if (a.contains("profile")) target.profile = a.at("profile").get<bool>();
        if (a.contains("selection_replay")) {
            if (!target.sparse) throw std::invalid_argument("selection replay requires sparse attention");
            target.selection_replay = resolve(base, a.at("selection_replay").get<std::string>());
        }
        if (a.contains("gpu_decode")) target.gpu_decode = a.at("gpu_decode").get<bool>();
        if (a.contains("prefetch")) target.prefetch = a.at("prefetch").get<bool>();
        if (a.contains("prefetch_trace")) target.prefetch_trace = a.at("prefetch_trace").get<bool>();
        if (a.contains("prefetch_queue_depth")) {
            target.prefetch_queue_depth = positive(a, "prefetch_queue_depth");
            if (target.prefetch_queue_depth < 2 || target.prefetch_queue_depth > 16384) throw std::invalid_argument("prefetch_queue_depth must be in [2,16384]");
        }
        if (a.contains("prefetch_slots_per_head")) target.prefetch_slots_per_head = positive(a, "prefetch_slots_per_head");
        if (target.prefetch && (target.in_memory || !target.sparse || !target.interleaved || !target.gpu_decode)) {
            throw std::invalid_argument("prefetch requires SSD sparse interleaved storage and GGML decode");
        }
        if (!target.prefetch && (a.contains("prefetch_queue_depth") || a.contains("prefetch_slots_per_head") || a.contains("prefetch_trace"))) throw std::invalid_argument("prefetch options require prefetch=true");
        if (a.contains("block_tokens")) target.block_tokens = positive(a, "block_tokens");
        if (a.contains("representative_count")) target.representative_count = positive(a, "representative_count");
        if (target.representative_count > target.block_tokens) throw std::invalid_argument("representative_count exceeds block_tokens");
        if (a.contains("write_buffer_bytes_per_layer")) target.write_buffer_bytes_per_layer = positive(a, "write_buffer_bytes_per_layer");
        if (a.contains("context_budget_tokens")) {
            target.context_budget_tokens = positive(a, "context_budget_tokens");
            if (target.context_budget_tokens > result.context_tokens) throw std::invalid_argument("attention budget exceeds context_tokens");
            if (!a.at("init_tokens").is_number_integer() || a.at("init_tokens").get<std::int64_t>() < 0 ||
                a.at("init_tokens").get<std::uint64_t>() > target.context_budget_tokens) throw std::invalid_argument("invalid init_tokens");
            target.init_tokens = a.at("init_tokens").get<std::uint32_t>();
            target.local_tokens = positive(a, "local_tokens");
            if (static_cast<std::uint64_t>(target.init_tokens) + target.local_tokens > target.context_budget_tokens) {
                throw std::invalid_argument("init/local exceed context budget");
            }
        } else if (a.contains("init_tokens") || a.contains("local_tokens")) {
            throw std::invalid_argument("explicit init/local require context_budget_tokens");
        }
        if (target.prefetch_slots_per_head > (static_cast<std::uint64_t>(result.context_tokens) + target.block_tokens - 1) / target.block_tokens) {
            throw std::invalid_argument("prefetch_slots_per_head exceeds context block count");
        }
        target.session_file = a.contains("session_file") ? resolve(base, a.at("session_file").get<std::string>()) :
            result.output.parent_path() / "kv.cache";
        if (a.contains("keep_session_file")) target.keep_session_file = a.at("keep_session_file").get<bool>();
    }
    if (result.microbatch_tokens > result.batch_tokens || result.batch_tokens > result.context_tokens ||
        result.max_new_tokens > result.context_tokens) {
        throw std::invalid_argument("require microbatch <= batch <= context and max_new_tokens <= context");
    }
    return result;
}
} // namespace solid
