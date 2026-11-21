#!/usr/bin/env python3
"""Prepare three fixed-selection scheduling runs and audit their completed traces."""
import argparse
import copy
import json
import math
import re
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
SHARED_ARCHIVE_PROOF = "kv_archive_attention_complete"
DEVICE_PROOFS = {"attention_until_boundary": ("attention_device_event",), "q_gpu_ready": ("query_device_projection",),
                 "kv_projection_to_host": ("kv_device_ready",),
                 "kv_d2h_ready": ("kv_archive_copy_complete", SHARED_ARCHIVE_PROOF)}


def integer(value, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("schedule integers must be nonnegative; identifiers and estimates must be positive")
    return value


def audit_schedule(path, mode, *, blocks=False):
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
            valid_name = (item["name"] == "historical_blocks_ready" or re.fullmatch(r"block_(read|h2d)_h\d+_b\d+_v[1-9]\d*", item["name"])) if blocks else item["name"] in TASKS
            if task_id in tasks or item["mode"] != mode or not valid_name:
                raise ValueError("duplicate task or unexpected task/mode")
            if item["clock"] != "steady_clock_ns" or item["resource"] not in ("host", "ssd", "h2d", "d2h", "gpu"):
                raise ValueError("unknown schedule clock/resource")
            origins.add(integer(item["origin_ns"]))
            integer(item["estimate_ns"], True)
            deps = item["predecessors"]
            if not isinstance(deps, list) or len(deps) != len(set(deps)):
                raise ValueError("duplicate or invalid dependencies")
            for dep in deps: integer(dep)
            tasks[task_id] = {"name": item["name"], "deps": deps, "state": "pending", "origin": item["origin_ns"], "resource": item["resource"]}
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
            if "device_ns" in item:
                valid = ("copy_ticket_complete",) if blocks and task["name"].startswith("block_h2d_") else DEVICE_PROOFS.get(task["name"], ())
                if item["fence"] not in valid:
                    raise ValueError("device interval lacks a matching completion proof")
                task["device_ns"]=integer(item["device_ns"])
        else:
            raise ValueError("failed or unknown schedule event")
    if len(origins) > 1:
        raise ValueError("schedule trace mixes clock origins")
    if blocks:
        return audit_block_frames(frames)
    totals = {"frames": len(frames), "tasks": 0, "critical_path_ns": 0, "miss_projection_overlap_ns": 0,
              "query_reused_fences": 0, "process_reused_fences": 0, "attention_fences": 0, "fallback_fences": 0,"device_event_completions":0,"device_attention_ns":0}
    proofs = {"query_dependency_reuse": "query_reused_fences", "process_synchronize_reuse": "process_reused_fences",
              "attention_backend_complete": "attention_fences", "explicit_fallback": "fallback_fences",
              "attention_device_event": "device_event_completions", "attention_device_event_untimed": "device_event_completions"}
    identities = set()
    single_layer = len({frame["identity"][0] for frame in frames.values() if frame["identity"] is not None}) == 1
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
                 "kv_projection_to_host": [] if serial else ["q_gpu_ready"],
                 "kv_d2h_ready": ["kv_projection_to_host"], "miss_read_and_reconcile": [selection],
                 "history_h2d": ["miss_read_and_reconcile"] if serial else [selection],
                 append: ["kv_d2h_ready"] if serial else ["kv_d2h_ready", "miss_read_and_reconcile", "history_h2d"],
                 "current_h2d": [append, "history_h2d"] if serial else [append],
                 "attention_until_boundary": ["current_h2d"], "buffered_ssd_store": [append],
                 "next_layer_prefetch": ["current_h2d"]}
        archive_proof = names["kv_d2h_ready"]["fence"]
        if archive_proof in ("kv_archive_copy_complete", SHARED_ARCHIVE_PROOF):
            if serial or names["kv_projection_to_host"]["fence"] != "kv_device_ready":
                raise ValueError("device archive lacks its KV production proof")
            if archive_proof == "kv_archive_copy_complete":
                edges["kv_d2h_ready"] = ["current_h2d"]
            elif mode != "overlap+sync_reuse":
                raise ValueError("shared archive proof outside sync reuse mode")
            elif names["kv_d2h_ready"]["end"] < names["attention_until_boundary"]["end"]:
                raise ValueError("shared archive completed before attention")
            edges[append] = ["kv_d2h_ready"]
            edges["current_h2d"] = ["kv_projection_to_host", "miss_read_and_reconcile", "history_h2d"]
            if archive_proof == SHARED_ARCHIVE_PROOF:
                edges["current_h2d"].remove("history_h2d")
                edges["attention_until_boundary"].append("history_h2d")
                edges["next_layer_prefetch"] = ["miss_read_and_reconcile"]
                if any(names[name]["fence"] != "attention_upload_dependency" for name in ("history_h2d", "current_h2d")):
                    raise ValueError("CUDA uploads lack a shared attention completion proof")
            if single_layer:
                edges["next_layer_prefetch"].append(append)
        if names["next_layer_prefetch"]["fence"] in ("shared_attention_buffer_released", "shared_context_capacity_reached"):
            edges["next_layer_prefetch"].append("attention_until_boundary")
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
                duration = task.get("device_ns", task["end"] - task["start"])
                costs[key] = max((costs[dep] for dep in task["deps"]), default=0) + duration
        totals["tasks"] += len(tasks)
        totals["critical_path_ns"] += max(costs.values())
        miss, kv = names["miss_read_and_reconcile"], names["kv_projection_to_host"]
        totals["miss_projection_overlap_ns"] += max(0, min(miss["end"], kv["end"]) - max(miss["start"], kv["start"]))
        proof = names["attention_until_boundary"]["fence"]
        if proof not in proofs: raise ValueError("attention completion lacks a known proof")
        if mode != "overlap+sync_reuse" and proof in ("query_dependency_reuse", "attention_device_event", "attention_device_event_untimed"):
            raise ValueError("query reuse proof in a mode that requires attention completion")
        if proof=="attention_device_event":
            if "device_ns" not in names["attention_until_boundary"]: raise ValueError("native attention completion lacks device timing")
            totals["device_attention_ns"]+=names["attention_until_boundary"]["device_ns"]
        totals[proofs[proof]] += 1
    return totals


def audit_block_frames(frames):
    totals = {"block_frames": len(frames), "block_tasks": 0, "block_critical_path_ns": 0}
    identities = set()
    for frame in frames.values():
        if frame["identity"] in identities:
            raise ValueError("duplicate block frame layer/position")
        identities.add(frame["identity"])
        tasks = frame["tasks"]
        if any(task["state"] != "complete" for task in tasks.values()):
            raise ValueError("block trace ends with unfinished tasks")
        names = {task["name"]: key for key, task in tasks.items()}
        if len(names) != len(tasks) or "historical_blocks_ready" not in names:
            raise ValueError("missing or duplicate block tasks")
        join = tasks[names["historical_blocks_ready"]]
        copies = []
        costs = []
        for name, key in names.items():
            if not name.startswith("block_read_"):
                continue
            read = tasks[key]
            copy_key = names.get(name.replace("block_read_", "block_h2d_", 1))
            if copy_key is None:
                raise ValueError("block READ lacks a matching H2D task")
            copy = tasks[copy_key]
            if read["deps"] or read["resource"] != "ssd" or copy["deps"] != [key] or copy["resource"] != "h2d":
                raise ValueError("block read/H2D dependencies or resources differ")
            if read["fence"] not in ("original_read_complete", "host_resident", "host_forwarded") or copy["fence"] not in ("copy_ticket_complete", "device_resident"):
                raise ValueError("block completion lacks an ownership proof")
            copies.append(copy_key)
            costs.append(read["end"] - read["start"] + copy.get("device_ns", copy["end"] - copy["start"]))
        if (len(tasks) != 2*len(copies)+1 or set(join["deps"]) != set(copies) or join["resource"] != "host" or
                join["fence"] != ("all_block_copies_complete" if copies else "empty_history")):
            raise ValueError("block join does not cover every H2D completion")
        totals["block_tasks"] += len(tasks)
        totals["block_critical_path_ns"] += max(costs, default=0) + join["end"] - join["start"]
    return totals


def audit_block_schedule(path, mode, parent=None):
    result = audit_schedule(path, mode, blocks=True)
    if parent is not None:
        def identities(source):
            rows = (json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines())
            return {(row["frame"], row["layer"], row["position"]) for row in rows if row["event"] == "define"}
        expected = set() if mode == "naive" else identities(parent)
        if identities(path) != expected:
            raise ValueError("block frames differ from the parent layer schedule")
    return result


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
        block_path = path.parent / "schedule.blocks.jsonl"
        if digest(block_path) != value["block_schedule_trace_sha256"]:
            raise ValueError("block schedule trace checksum differs")
        audit.update(audit_block_schedule(block_path, mode, trace_path))
        recorded = value["schedule"]
        if recorded["mode"] != mode or recorded["trace_enabled"] is not True:
            raise ValueError("scheduling metrics mode/trace setting differs")
        for key, actual in audit.items():
            metric = {"critical_path_ns": "observed_critical_path_sum_ms",
                      "miss_projection_overlap_ns": "miss_projection_overlap_sum_ms",
                      "block_critical_path_ns": "block_observed_critical_path_sum_ms","device_attention_ns":"device_attention_event_ms"}.get(key, key)
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
