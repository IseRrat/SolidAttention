#!/usr/bin/env python3
# 准备并汇总 separate/interleaved KV 布局对照，核对两组输入和数值结果。
"""Prepare or compare a fixed-token, fixed-selection 2x2 layout/projection ablation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from compare_logits import compare, load_result
from prepare_replay import prepare

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

# 从同一参考运行生成两种 KV 布局配置及对照清单。
def create(reference, destination, results):
    reference = Path(reference).resolve(strict=True)
    value, _ = load_result(reference)
    attention = value["attention"]
    if not attention["sparse"] or attention["storage"] != "ssd":
        raise ValueError("layout ablation requires an SSD sparse reference")
    selection = reference.parent / "selection.jsonl"
    selection_hash = digest(selection)
    if "selection_trace_sha256" in value and value["selection_trace_sha256"] != selection_hash:
        raise ValueError("reference selection checksum differs")
    destination = Path(destination).resolve()
    results = Path(results).resolve()
    # Validate the reference and all settings before creating any output.
    config = prepare(reference, reference.parent / "config.json", results / "unused/result.json")
    base = config["attention"]
    base.pop("session_file", None)
    for key in ("prefetch_queue_depth", "prefetch_slots_per_head", "prefetch_trace", "schedule", "schedule_trace",
                "io_cpu", "coordinator_cpu"):
        base.pop(key, None)
    base["prefetch"] = False
    base["keep_session_file"] = False
    base["selection_replay"] = str(selection)
    base["selection_backend"] = "cpu"
    base["selection_trace"] = True
    base.pop("profile", None)
    for key in ("block_tokens", "context_budget_tokens", "init_tokens", "local_tokens", "representative_count"):
        base[key] = attention[key]
    configs = []
    for layout in ("separate", "interleaved"):
        for fused in (False, True):
            name = layout + ("-fused" if fused else "-separate")
            target = copy.deepcopy(config)
            target["attention"]["layout"] = layout
            target["attention"]["fuse_kv_projection"] = fused
            target["output"] = str(results / name / "result.json")
            configs.append((name, target))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "reference": str(reference), "reference_sha256": digest(reference),
                "selection_sha256": selection_hash, "variants": []}
    for name, target in configs:
        path = destination / (name + ".json")
        path.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
        manifest["variants"].append({"name": name, "config": str(path), "config_sha256": digest(path),
                                     "result": target["output"], "layout": target["attention"]["layout"],
                                     "fused": target["attention"]["fuse_kv_projection"]})
    path = destination / "ablation.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path

def selection_difference(left, right):
    with Path(left).open(encoding="utf-8") as a, Path(right).open(encoding="utf-8") as b:
        line = 0
        while True:
            x, y = a.readline(), b.readline()
            if not x and not y:
                return None
            line += 1
            if not x or not y or json.loads(x) != json.loads(y):
                return line

def metrics(result):
    attention = result["attention"]
    keys = ("layout", "fuse_kv_projection", "read_requests", "read_bytes", "write_requests", "write_bytes",
            "io_read_ms", "io_write_ms", "append_ms", "selection_ms", "gather_inclusive_ms", "gather_copy_bytes",
            "streaming_attention_inclusive_ms", "projection_extra_weight_bytes", "projection_initialization_ms")
    out = {key: attention[key] for key in keys if key in attention}
    out["decode_loop_ms"] = result["timing"]["decode_loop_ms"]
    out["decode_steps"] = result["timing"]["decode_steps"]
    if "profile" in result:
        # Preserve phase, direction, tensor/op identity and counts for audit.
        out["profile"] = result["profile"]
    return out

# 核对布局对照的输入、选块和数值结果，再汇总搬运及耗时指标。
def report(manifest_path, atol=1e-3, rtol=1e-4):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or len(manifest["variants"]) != 4 or
            {(v["layout"], v["fused"]) for v in manifest["variants"]} !=
            {(layout, fused) for layout in ("separate", "interleaved") for fused in (False, True)} or
            len({v["result"] for v in manifest["variants"]}) != 4):
        raise ValueError("ablation manifest must contain all four independent variants")
    reference_path = Path(manifest["reference"])
    if digest(reference_path) != manifest["reference_sha256"]:
        raise ValueError("reference result changed after preparation")
    source_selection = reference_path.parent / "selection.jsonl"
    if digest(source_selection) != manifest["selection_sha256"]:
        raise ValueError("reference selection changed after preparation")
    results = []
    provenance = None
    for variant in manifest["variants"]:
        if digest(variant["config"]) != variant["config_sha256"]:
            raise ValueError(f"ablation configuration changed: {variant['name']}")
        path = Path(variant["result"])
        run = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if (run["status"] != "complete" or run["returncode"] != 0 or run["result_sha256"] != digest(path) or
                run["source_config_sha256"] != variant["config_sha256"] or
                run["effective_config_sha256"] != digest(path.parent / "config.json")):
            raise ValueError("ablation run provenance differs from recorded files")
        identity = (run["code_commit"], run["runner_sha256"])
        if provenance is None:
            provenance = identity
        elif provenance != identity:
            raise ValueError("ablation runs use different code or runner binaries")
        value = json.loads(path.read_text(encoding="utf-8"))
        attention = value["attention"]
        expected_attention = json.loads(Path(variant["config"]).read_text(encoding="utf-8"))["attention"]
        for key in ("storage", "sparse", "block_tokens", "context_budget_tokens", "init_tokens", "local_tokens", "representative_count"):
            if attention[key] != expected_attention[key]:
                raise ValueError(f"result sparse budget differs: {key}")
        if (attention["layout"], attention["fuse_kv_projection"], attention["selection_source"]) != (variant["layout"], variant["fused"], "replay"):
            raise ValueError(f"result mode differs from ablation: {variant['name']}")
        candidate_selection = path.parent / "selection.jsonl"
        if digest(candidate_selection) != value["selection_trace_sha256"]:
            raise ValueError("candidate selection checksum differs")
        if value["selection_replay_sha256"] != manifest["selection_sha256"]:
            raise ValueError("candidate replay source differs")
        first_difference = selection_difference(source_selection, candidate_selection)
        numerical = compare(reference_path, path, atol, rtol)
        results.append({"name": variant["name"], "passed": numerical["passed"] and first_difference is None,
                        "first_selection_difference_line": first_difference, "logits": numerical, "metrics": metrics(value)})
    return {"passed": all(value["passed"] for value in results), "variants": results}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("reference", type=Path)
    prepare_parser.add_argument("destination", type=Path)
    prepare_parser.add_argument("--results", type=Path, required=True)
    compare_parser = actions.add_parser("compare")
    compare_parser.add_argument("manifest", type=Path)
    compare_parser.add_argument("--atol", type=float, default=1e-3)
    compare_parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.action == "prepare":
        print(create(args.reference, args.destination, args.results))
        return 0
    result = report(args.manifest, args.atol, args.rtol)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"layout_ablation: {error}", file=sys.stderr)
        sys.exit(2)
