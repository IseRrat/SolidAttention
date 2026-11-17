#!/usr/bin/env python3
"""Prepare three fixed-selection scheduling runs and audit their completed traces."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

from compare_prefetch import compare_prefetch
from layout_ablation import digest
from prepare_prefetch import prepare_prefetch

MODES = ("naive", "overlap", "overlap+sync_reuse")
NAMES = ("naive", "overlap", "sync-reuse")
TASKS = {"q_gpu_ready", "q_d2h_ready", "selection", "selection_ready", "kv_projection_to_host",
         "kv_d2h_ready", "miss_read_and_reconcile", "history_h2d", "append_current_kv",
         "append_and_selection", "current_h2d", "attention_until_boundary", "buffered_ssd_store", "next_layer_prefetch"}


def integer(value, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("schedule integers must be nonnegative; identifiers and estimates must be positive")
    return value


def audit_schedule(path, mode):
    if mode not in MODES:
        raise ValueError("unknown scheduling mode")
    frames = {}
    origins = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        frame_id, task_id = integer(item["frame"], True), integer(item["task"])
        frame = frames.setdefault(frame_id, {"tasks": {}, "identity": None})
        tasks = frame["tasks"]
        event = item["event"]
        if event == "define":
            identity = integer(item["layer"]), integer(item["position"])
            if frame["identity"] is not None and frame["identity"] != identity:
                raise ValueError("frame identity changed")
            frame["identity"] = identity
            if task_id in tasks or item["mode"] != mode or item["name"] not in TASKS:
                raise ValueError("duplicate task or unexpected task/mode")
            if item["clock"] != "steady_clock_ns" or item["resource"] not in ("host", "ssd", "h2d", "d2h", "gpu"):
                raise ValueError("unknown schedule clock/resource")
            origins.add(integer(item["origin_ns"]))
            integer(item["estimate_ns"], True)
            deps = item["predecessors"]
            if not isinstance(deps, list) or len(deps) != len(set(deps)):
                raise ValueError("duplicate or invalid dependencies")
            for dep in deps: integer(dep)
            tasks[task_id] = {"name": item["name"], "deps": deps, "state": "pending", "origin": item["origin_ns"]}
            continue
        if task_id not in tasks:
            raise ValueError("schedule event before task definition")
        task = tasks[task_id]
        now = integer(item["timestamp_ns"])
        if now < task["origin"]:
            raise ValueError("schedule timestamp precedes clock origin")
        if event == "start":
            if task["state"] != "pending":
                raise ValueError("task started more than once")
            for dep in task["deps"]:
                if dep not in tasks or tasks[dep]["state"] != "complete" or tasks[dep]["end"] > now:
                    raise ValueError("task started before dependency completion")
            task["start"] = now
            task["state"] = "running"
        elif event == "complete":
            if task["state"] != "running" or now < task["start"]:
                raise ValueError("invalid schedule completion")
            task.update(state="complete", end=now, fence=item["fence"])
        else:
            raise ValueError("failed or unknown schedule event")
    if len(origins) > 1:
        raise ValueError("schedule trace mixes clock origins")
    totals = {"frames": len(frames), "tasks": 0, "critical_path_ns": 0, "miss_projection_overlap_ns": 0,
              "query_reused_fences": 0, "process_reused_fences": 0, "attention_fences": 0, "fallback_fences": 0}
    proofs = {"query_dependency_reuse": "query_reused_fences", "process_synchronize_reuse": "process_reused_fences",
              "attention_backend_complete": "attention_fences", "explicit_fallback": "fallback_fences"}
    identities = set()
    for frame in frames.values():
        tasks = frame["tasks"]
        names = {task["name"]: task for task in tasks.values()}
        expected = TASKS - ({"selection", "append_current_kv"} if mode == "naive" else {"selection_ready", "append_and_selection"})
        if set(names) != expected or len(tasks) != 12 or frame["identity"] in identities:
            raise ValueError("schedule frame has missing/duplicate tasks or layer positions")
        identities.add(frame["identity"])
        if any(task["state"] != "complete" for task in tasks.values()):
            raise ValueError("schedule trace ends with unfinished tasks")
        serial = mode == "naive"
        selection = "selection_ready" if serial else "selection"
        append = "append_and_selection" if serial else "append_current_kv"
        edges = {"q_gpu_ready": [], "q_d2h_ready": ["q_gpu_ready"],
                 selection: [append, "q_d2h_ready"] if serial else ["q_d2h_ready"],
                 "kv_projection_to_host": [] if serial else [selection],
                 "kv_d2h_ready": ["kv_projection_to_host"], "miss_read_and_reconcile": [selection],
                 "history_h2d": ["miss_read_and_reconcile"] if serial else [selection],
                 append: ["kv_d2h_ready"] if serial else ["kv_d2h_ready", "miss_read_and_reconcile", "history_h2d"],
                 "current_h2d": [append, "history_h2d"] if serial else [append],
                 "attention_until_boundary": ["current_h2d"], "buffered_ssd_store": [append],
                 "next_layer_prefetch": ["current_h2d"]}
        for name, task in names.items():
            if {tasks[dep]["name"] for dep in task["deps"]} != set(edges[name]):
                raise ValueError("recorded schedule dependencies differ from the execution pipeline")
        # Starts already validate acyclicity and predecessor completion. Resolve
        # in topological order so the audit does not depend on numeric task IDs.
        costs = {}
        while len(costs) != len(tasks):
            ready = [key for key, task in tasks.items() if key not in costs and all(dep in costs for dep in task["deps"])]
            if not ready: raise ValueError("schedule dependency cycle")
            for key in ready:
                task = tasks[key]
                costs[key] = max((costs[dep] for dep in task["deps"]), default=0) + task["end"] - task["start"]
        totals["tasks"] += len(tasks)
        totals["critical_path_ns"] += max(costs.values())
        miss, kv = names["miss_read_and_reconcile"], names["kv_projection_to_host"]
        totals["miss_projection_overlap_ns"] += max(0, min(miss["end"], kv["end"]) - max(miss["start"], kv["start"]))
        proof = names["attention_until_boundary"]["fence"]
        if proof not in proofs: raise ValueError("attention completion lacks a known proof")
        if mode != "overlap+sync_reuse" and proof == "query_dependency_reuse":
            raise ValueError("query reuse proof in a mode that requires attention completion")
        totals[proofs[proof]] += 1
    return totals


def create(reference, destination, results, queue_depth=64, slots_per_head=None):
    reference = Path(reference).resolve(strict=True)
    source = json.loads(reference.read_text(encoding="utf-8"))
    if "profile" in source:
        raise ValueError("prepare scheduling comparisons from a reference with profile=false")
    if not source["decode_input_token_ids"]:
        raise ValueError("scheduling comparisons require at least one decode step")
    destination, results = Path(destination).resolve(), Path(results).resolve()
    base = prepare_prefetch(reference, results / "unused/result.json", True, queue_depth, slots_per_head)
    base["attention"].update(profile=False, schedule_trace=True)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "reference": str(reference), "reference_sha256": digest(reference),
                "selection_sha256": digest(reference.parent / "selection.jsonl"), "variants": []}
    for name, mode in zip(NAMES, MODES):
        config = copy.deepcopy(base)
        config["attention"]["schedule"] = mode
        config["output"] = str(results / name / "result.json")
        path = destination / (name + ".json")
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest["variants"].append({"name": name, "mode": mode, "config": str(path),
                                     "config_sha256": digest(path), "result": config["output"]})
    path = destination / "schedule-ablation.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def report(manifest_path, atol=1e-3, rtol=1e-4):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    variants = manifest["variants"]
    if (manifest.get("schema_version") != 1 or len(variants) != 3 or
            {(v["name"], v["mode"]) for v in variants} != set(zip(NAMES, MODES)) or
            len({v["result"] for v in variants}) != 3 or len({v["config"] for v in variants}) != 3):
        raise ValueError("scheduling manifest must contain three independent modes")
    reference = Path(manifest["reference"])
    if digest(reference) != manifest["reference_sha256"] or digest(reference.parent / "selection.jsonl") != manifest["selection_sha256"]:
        raise ValueError("scheduling reference changed after preparation")
    rows = []
    for variant in variants:
        config_path, path = Path(variant["config"]), Path(variant["result"])
        if digest(config_path) != variant["config_sha256"]:
            raise ValueError("scheduling configuration changed after preparation")
        run = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if run["source_config_sha256"] != variant["config_sha256"]:
            raise ValueError("scheduling run used another source configuration")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        effective = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        # The launcher snapshots the prompt and replay into each run directory.
        expected = copy.deepcopy(config)
        expected["prompt"] = str(path.parent / "prompt.txt")
        expected["attention"]["selection_replay"] = str(path.parent / "selection.replay.jsonl")
        if (digest(expected["prompt"]) != digest(config["prompt"]) or
                digest(expected["attention"]["selection_replay"]) != manifest["selection_sha256"] or expected != effective):
            raise ValueError("effective scheduling configuration differs from prepared configuration")
        value = json.loads(path.read_text(encoding="utf-8"))
        mode = variant["mode"]
        if (value["attention"]["schedule"] != mode or value["attention"]["selection_source"] != "replay" or
                value["selection_replay_sha256"] != manifest["selection_sha256"] or "profile" in value):
            raise ValueError("scheduling mode, replay source or profiling setting differs")
        comparison = compare_prefetch(reference, path, atol, rtol)
        trace_path = path.parent / "schedule.jsonl"
        if digest(trace_path) != value["schedule_trace_sha256"]:
            raise ValueError("schedule trace checksum differs")
        audit = audit_schedule(trace_path, mode)
        recorded = value["schedule"]
        if recorded["mode"] != mode or recorded["trace_enabled"] is not True:
            raise ValueError("scheduling metrics mode/trace setting differs")
        for key, actual in audit.items():
            metric = {"critical_path_ns": "observed_critical_path_sum_ms",
                      "miss_projection_overlap_ns": "miss_projection_overlap_sum_ms"}.get(key, key)
            expected = actual / 1e6 if key.endswith("_ns") else actual
            observed = recorded[metric]
            if type(observed) not in (int, float) or not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"schedule metric differs from trace: {metric}")
        if value["timing"]["decode_steps"] and not audit["frames"]:
            raise ValueError("decode run has no schedule frames")
        rows.append({"name": variant["name"], "passed": comparison["passed"], "comparison": comparison,
                     "schedule_audit": audit, "schedule": recorded, "timing": value["timing"]})
    return {"passed": all(row["passed"] for row in rows), "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    p = actions.add_parser("prepare")
    p.add_argument("reference", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--queue-depth", default=64, type=int)
    p.add_argument("--slots-per-head", type=int)
    p = actions.add_parser("compare")
    p.add_argument("manifest", type=Path)
    p.add_argument("--atol", default=1e-3, type=float)
    p.add_argument("--rtol", default=1e-4, type=float)
    args = parser.parse_args()
    if args.action == "prepare":
        print(create(args.reference, args.destination, args.results, args.queue_depth, args.slots_per_head))
        return 0
    result = report(args.manifest, args.atol, args.rtol)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"schedule_ablation: {error}", file=sys.stderr)
        sys.exit(2)
