"""Shared validation and result provenance for offline experiment tools."""
import copy
import json
import math
from pathlib import Path
import statistics

from run_baseline import digest, atomic_json


def load(path):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON: {value}")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,parse_constant=nonfinite)


def number(value, minimum=0, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"expected finite {'integer' if integer else 'number'} >= {minimum}")
    return value


def token_ids(values):
    if not isinstance(values, list) or not values: raise ValueError("token list must be nonempty")
    for value in values:
        number(value, integer=True)
        if value > 2147483647: raise ValueError("token ID exceeds int32")
    return values


def absolute_config(path):
    path = Path(path).resolve(strict=True)
    config = load(path)
    for key in ("model", "prompt", "output"):
        config[key] = str((path.parent / config[key]).resolve())
    for part in config.get("additional_model_files", []):
        part["path"] = str((path.parent / part["path"]).resolve())
    for key in ("session_file", "selection_replay"):
        if key in config.get("attention", {}):
            config["attention"][key] = str((path.parent / config["attention"][key]).resolve())
    return config


def model_hashes(config):
    values = [config["model_sha256"]] + [part["sha256"] for part in config.get("additional_model_files", [])]
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values):
        raise ValueError("model hashes must be lowercase SHA256")
    return values


def check_hash(path, expected):
    if digest(Path(path)) != expected: raise ValueError(f"checksum differs: {path}")


def audit_run(result_path, source_config=None, source_sha=None):
    path = Path(result_path)
    run = load(path.parent / "manifest.json")
    if run["status"] != "complete" or run["returncode"] != 0: raise ValueError("run did not complete")
    check_hash(path, run["result_sha256"])
    check_hash(path.parent / "config.json", run["effective_config_sha256"])
    check_hash(path.parent / "prompt.txt", run["prompt_sha256"])
    for name,sha in run.get("artifacts_sha256",{}).items(): check_hash(path.parent/name,sha)
    effective = load(path.parent / "config.json")
    if source_config is not None:
        check_hash(source_config, source_sha)
        if run["source_config_sha256"] != source_sha: raise ValueError("run source configuration differs")
        expected = absolute_config(source_config)
        check_hash(Path(expected["prompt"]), run["prompt_sha256"])
        expected["prompt"] = str((path.parent / "prompt.txt").resolve())
        if "selection_replay" in expected.get("attention", {}):
            replay = path.parent / "selection.replay.jsonl"
            check_hash(replay, run["selection_replay_sha256"])
            check_hash(expected["attention"]["selection_replay"], run["selection_replay_sha256"])
            expected["attention"]["selection_replay"] = str(replay.resolve())
        if effective != expected: raise ValueError("effective configuration differs from prepared inputs")
    result = load(path)
    if result["model_parts_sha256"] != model_hashes(effective): raise ValueError("result model identity differs")
    if "prompt_token_ids" in effective and result["prompt_token_ids"] != effective["prompt_token_ids"]:
        raise ValueError("result prompt token IDs differ")
    if result.get("kv_dtype", "f16") != effective["kv_dtype"]: raise ValueError("result KV precision differs")
    weight_dtype = effective.get("kv_projection_weight_dtype")
    if weight_dtype and result.get("kv_projection_weight_dtype") != weight_dtype:
        raise ValueError("K/V projection weight precision differs")
    configured = effective.get("attention")
    if bool(configured) != ("attention" in result): raise ValueError("result attention mode differs")
    if configured:
        actual = result["attention"]
        defaults = {"layout":"separate","fuse_kv_projection":False,"prefetch":False,"schedule":"naive","block_tokens":32,
                    "representative_count":1,"infllm_attention":False}
        for key in ("storage","sparse",*defaults):
            if actual.get(key,defaults.get(key)) != configured.get(key,defaults.get(key)): raise ValueError(f"result attention setting differs: {key}")
        for key in ("context_budget_tokens","init_tokens","local_tokens"):
            if key in configured and actual[key] != configured[key]: raise ValueError("result context budget differs")
        expected_io = "direct_decode_reads" if configured.get("direct_decode_reads", False) else "buffered"
        if actual.get("io_mode", "buffered") != expected_io: raise ValueError("result KV I/O mode differs")
        if configured.get("infllm_attention") and configured.get("prefill_backend") == "cuda":
            if actual.get("prefill_backend") != "cuda_infllm_batched":
                raise ValueError("InfLLM result did not execute CUDA sparse prefill")
        if weight_dtype and actual.get("fuse_kv_projection"):
            weight_types = actual.get("projection_weight_types", [])
            if not weight_types or set(weight_types) != {weight_dtype}:
                raise ValueError("fused K/V projection weight precision differs")
    if result.get("build_info") != run.get("build_info") or result["upstream_commit"] != run["upstream_commit"]:
        raise ValueError("result build provenance differs")
    return result, run, effective


def distribution(values):
    if not values: return None
    values = sorted(number(value) for value in values)
    def percentile(q):
        position = (len(values)-1)*q
        left = int(position)
        return values[left] + (values[min(left+1,len(values)-1)]-values[left])*(position-left)
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": values[0], "max": values[-1], "p50": percentile(.5), "p90": percentile(.9),
            "p99": percentile(.99), "p999": percentile(.999)}


def run_metrics(result):
    trace = result["trace"]
    tokens = token_ids(result["generated_token_ids"])
    inputs = result["decode_input_token_ids"]
    if len(trace) != len(tokens) or len(inputs)+1 != len(tokens): raise ValueError("token trace length differs")
    for index, row in enumerate(trace):
        if row["index"] != index or row["token_id"] != tokens[index] or row["source"] != ("decode" if index else "prefill"):
            raise ValueError("invalid token trace identity")
        for key in ("total_ms", "sample_ms", "process_ms"): number(row[key])
    timing = result["timing"]
    elapsed = number(timing["decode_loop_ms"])
    if timing["decode_steps"] != len(inputs) or not math.isclose(elapsed, sum(row["total_ms"] for row in trace[1:]), rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("decode timing/count differs from trace")
    out = {"decode_steps": len(inputs), "output_tokens": len(tokens), "decode_loop_ms": elapsed,
           "decode_tokens_per_second": len(inputs)*1000/elapsed if elapsed else None,
           "prefill_ms": number(timing["prefill_ms"]), "first_token_ms": number(timing["first_token_ms"]),
           "request_ms": number(timing["request_ms"]), "decode_latency_ms": distribution([row["total_ms"] for row in trace[1:]]),
           "decode_process_ms": distribution([row["process_ms"] for row in trace[1:]])}
    out["kv_system_tracked_bytes"] = result.get("memory",{}).get("kv_system_tracked_bytes")
    for name in ("attention", "attention_phases", "prefetch", "schedule"):
        if name in result: out[name] = copy.deepcopy(result[name])
    return out
