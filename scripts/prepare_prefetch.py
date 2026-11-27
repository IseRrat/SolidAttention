#!/usr/bin/env python3
# 从参考运行生成预取对照配置与清单，沿用相同的模型、输入和上下文。
"""Prepare fixed-token prefetch verification from a synchronous sparse reference."""
import argparse
import json
from pathlib import Path

from compare_logits import load_result
from layout_ablation import digest
from prepare_replay import prepare


# 沿用参考模型和输入，生成预取对照配置及结果位置清单。
def prepare_prefetch(reference, output, fixed_selection=False, queue_depth=64, slots_per_head=None):
    if type(queue_depth) is not int or not 2 <= queue_depth <= 16384:
        raise ValueError("queue depth must be in [2,16384]")
    if slots_per_head is not None and (type(slots_per_head) is not int or slots_per_head <= 0):
        raise ValueError("slots per head must be positive")
    reference = Path(reference).resolve(strict=True)
    value, _ = load_result(reference)
    attention = value["attention"]
    if (attention["storage"], attention["sparse"], attention["layout"], attention.get("prefetch", False)) != ("ssd", True, "interleaved", False):
        raise ValueError("use a synchronous SSD sparse interleaved reference")
    selection = reference.parent / "selection.jsonl"
    if digest(selection) != value["selection_trace_sha256"]:
        raise ValueError("reference selection checksum differs")
    config = prepare(reference, reference.parent / "config.json", output)
    mode = config["attention"]
    if not mode.get("gpu_decode", True):
        raise ValueError("prefetch verification requires a GGML decode reference")
    if slots_per_head is not None and slots_per_head > (config["context_tokens"] + attention["block_tokens"] - 1) // attention["block_tokens"]:
        raise ValueError("slots per head exceed context block count")
    for key in ("session_file", "selection_replay", "prefetch_slots_per_head", "schedule", "schedule_trace"):
        mode.pop(key, None)
    mode.update({"prefetch": True, "prefetch_trace": True, "selection_trace": True, "prefetch_queue_depth": queue_depth, "keep_session_file": False})
    for key in ("block_tokens", "context_budget_tokens", "init_tokens", "local_tokens", "representative_count"):
        mode[key] = attention[key]
    if fixed_selection:
        mode["selection_replay"] = str(selection)
        mode["selection_backend"] = "cpu"
    if slots_per_head is not None: mode["prefetch_slots_per_head"] = slots_per_head
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-selection", action="store_true")
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--slots-per-head", type=int)
    args = parser.parse_args()
    config = prepare_prefetch(args.reference, args.output, args.fixed_selection, args.queue_depth, args.slots_per_head)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
