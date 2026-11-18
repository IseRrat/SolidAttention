#!/usr/bin/env python3
"""Compare prefetch selection/logits and audit recorded device-slot lifetimes."""
import argparse
import json
from pathlib import Path
import sys

from compare_logits import compare
from layout_ablation import digest, selection_difference


def audit_residency(path, block_tokens, slots_per_head):
    if type(block_tokens) is not int or block_tokens <= 0 or type(slots_per_head) is not int or slots_per_head <= 0:
        raise ValueError("invalid residency geometry")
    states = {}
    pins = 0
    uploads = 0
    current = dict(current_device_rows=0,current_device_bytes=0,current_host_bytes=0,
                   refreshed_device_tails=0,retained_device_prefix_bytes=0)
    row_size = None
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            for key in ("slot", "generation", "layer", "kv_head", "block", "version", "epoch"):
                if type(item[key]) is not int or item[key] < 0:
                    raise ValueError(f"invalid residency integer at line {line_number}")
            slot, generation, event = item["slot"], item["generation"], item["event"]
            if (not generation or not 1 <= item["version"] <= block_tokens or slot // slots_per_head != item["kv_head"]):
                raise ValueError(f"residency identity outside configured bounds at line {line_number}")
            identity = tuple(item[key] for key in ("layer", "kv_head", "block", "version"))
            previous = states.get(slot)
            if event == "upload":
                if previous and (previous["state"] != "free" or generation <= previous["generation"]):
                    raise ValueError(f"device overwrite before release at line {line_number}")
                fields=("current_position","retained_prefix_tokens","prefix_generation","row_bytes","host_copy_bytes","device_copy_bytes")
                if "copy_kind" in item or any(key in item for key in fields):
                    if item.get("copy_kind")!="current_device" or any(type(item.get(key)) is not int or item[key]<0 for key in fields):
                        raise ValueError(f"invalid current-device copy metadata at line {line_number}")
                    prefix, row_bytes = item["retained_prefix_tokens"],item["row_bytes"]
                    position=item["current_position"]
                    if (position//block_tokens!=item["block"] or position%block_tokens+1!=item["version"] or
                            prefix>=item["version"] or row_bytes<4 or row_bytes%4 or
                            item["host_copy_bytes"]!=(item["version"]-1-prefix)*row_bytes or item["device_copy_bytes"]!=row_bytes or
                            (row_size is not None and row_size!=row_bytes)):
                        raise ValueError(f"current-device row or byte coverage differs at line {line_number}")
                    if prefix:
                        if (not previous or previous["identity"][:3]!=identity[:3] or previous["identity"][3]!=prefix or
                                item["prefix_generation"]!=previous["generation"] or generation!=previous["generation"]+1):
                            raise ValueError(f"retained prefix lacks its previous generation at line {line_number}")
                        current["refreshed_device_tails"]+=1
                    elif item["prefix_generation"]:
                        raise ValueError(f"cold current row claims an earlier generation at line {line_number}")
                    row_size=row_bytes
                    current["current_device_rows"]+=1; current["current_device_bytes"]+=row_bytes
                    current["current_host_bytes"]+=item["host_copy_bytes"]; current["retained_device_prefix_bytes"]+=prefix*row_bytes
                states[slot] = {"state": "loading", "generation": generation, "identity": identity, "epoch": None}
                uploads += 1
                continue
            if not previous or previous["generation"] != generation or previous["identity"] != identity:
                raise ValueError(f"stale residency event at line {line_number}")
            expected, target = {
                "upload_complete": (("loading_pinned", "pinned") if previous["state"] == "loading_pinned" else ("loading", "ready")),
                "attention_pin": ("ready", "pinned"),
                "attention_pin_waiting": ("loading", "loading_pinned"),
                "attention_complete": ("pinned", "ready"),
                "overwrite_available": ("ready", "free"),
            }.get(event, (None, None))
            if previous["state"] != expected:
                raise ValueError(f"invalid device transition at line {line_number}: {event}")
            if event in ("attention_pin", "attention_pin_waiting"):
                if not item["epoch"]: raise ValueError("attention epoch must be positive")
                previous["epoch"] = item["epoch"]; pins += 1
            if event == "attention_complete" and previous["epoch"] != item["epoch"]:
                raise ValueError(f"attention completion epoch differs at line {line_number}")
            previous["state"] = target
    if any(item["state"] in ("pinned", "loading", "loading_pinned") for item in states.values()):
        raise ValueError("device trace ends with active transfers or attention readers")
    result = {"uploads": uploads, "attention_pins": pins, "slots_used": len(states), "balanced": True}
    if current["current_device_rows"]: result.update(current)
    return result


def compare_prefetch(reference_path, candidate_path, atol=1e-3, rtol=1e-4):
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    provenance = None
    for path in (reference_path, candidate_path):
        run = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if (run["status"] != "complete" or run["returncode"] != 0 or run["result_sha256"] != digest(path) or
                run["effective_config_sha256"] != digest(path.parent / "config.json")):
            raise ValueError("prefetch run provenance differs from recorded files")
        identity = run["code_commit"], run["runner_sha256"]
        if provenance is None: provenance = identity
        elif identity != provenance: raise ValueError("reference and candidate use different code or binaries")
    a, b = reference["attention"], candidate["attention"]
    if a.get("prefetch", False) or not b["prefetch"]:
        raise ValueError("compare a synchronous reference with a prefetch candidate")
    for key in ("storage", "sparse", "layout", "fuse_kv_projection", "block_tokens", "context_budget_tokens", "init_tokens", "local_tokens", "representative_count"):
        if a[key] != b[key]: raise ValueError(f"attention setting differs: {key}")
    for path, value in ((reference_path, reference), (candidate_path, candidate)):
        if digest(path.parent / "selection.jsonl") != value["selection_trace_sha256"]:
            raise ValueError("selection trace checksum differs")
    if ("profile" in reference) != ("profile" in candidate):
        raise ValueError("reference and candidate must use the same profiling setting")
    if digest(candidate_path.parent / "prefetch.jsonl") != candidate["prefetch_trace_sha256"]:
        raise ValueError("prefetch trace checksum differs")
    counters = candidate["prefetch"]
    if counters["cancel_requests"] != counters["cancel_completions"]:
        raise ValueError("prefetch run has undrained cancellation completions")
    audit = None
    if "residency_trace_sha256" in candidate:
        path = candidate_path.parent / "residency.jsonl"
        if digest(path) != candidate["residency_trace_sha256"]:
            raise ValueError("residency checksum differs")
        audit = audit_residency(path, b["block_tokens"], counters["slots_per_head"])
        for key in ("current_device_rows","current_device_bytes","current_host_bytes","refreshed_device_tails","retained_device_prefix_bytes"):
            if key in counters and (type(counters[key]) is not int or counters[key]<0 or counters[key]!=audit.get(key,0)):
                raise ValueError(f"current-device counter differs from residency trace: {key}")
    if candidate["timing"]["decode_steps"] and (not audit or not audit["attention_pins"]):
        raise ValueError("decode run has no audited attention slot leases")
    numerical = compare(reference_path, candidate_path, atol, rtol)
    difference = selection_difference(reference_path.parent / "selection.jsonl", candidate_path.parent / "selection.jsonl")
    return {"passed": numerical["passed"] and difference is None, "logits": numerical,
            "first_selection_difference_line": difference, "residency": audit, "prefetch": counters}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        result = compare_prefetch(args.reference, args.candidate, args.atol, args.rtol)
        print(json.dumps(result, indent=2, allow_nan=False))
        sys.exit(0 if result["passed"] else 1)
    except (OSError, ValueError, KeyError) as error:
        print(f"compare_prefetch: {error}", file=sys.stderr)
        sys.exit(2)
