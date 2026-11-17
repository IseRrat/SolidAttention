#!/usr/bin/env python3
"""Measure consecutive formal selected-block overlap by layer and KV head."""
import argparse
import json
from pathlib import Path

from experiment_common import atomic_json, audit_run, check_hash, digest, distribution, number


def analyze(result_path):
    path = Path(result_path)
    result,run,_ = audit_run(path)
    attention = result["attention"]
    trace = path.parent/"selection.jsonl"
    check_hash(trace,result["selection_trace_sha256"])
    block = number(attention["block_tokens"],1,True)
    previous,by_head,records = {},{},[]
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            layer,head,position = (number(row[key],0,True) for key in ("layer","kv_head","position"))
            positions = set(); last = -1
            for first,count in row["ranges"]:
                number(first,0,True); number(count,1,True)
                if first <= last or first+count-1 > position or len(positions)+count > attention["context_budget_tokens"]:
                    raise ValueError("invalid selection ranges/order/budget")
                positions.update(range(first,first+count)); last = first+count-1
            local_first = position+1-min(attention["local_tokens"],position+1)
            selected = {p//block for p in positions if p >= attention["init_tokens"] and p < local_first}
            key = layer,head
            before = previous.get(key)
            if before and position <= before[0]: raise ValueError("duplicate or nonmonotonic selection identity")
            ratio = len(selected & before[1])/len(selected) if before and before[0]+1 == position and selected else None
            record = {"layer":layer,"kv_head":head,"position":position,"selected_blocks":len(selected),
                      "selected_tokens":sum(p >= attention["init_tokens"] and p < local_first for p in positions),
                      "init_tokens":sum(p < attention["init_tokens"] for p in positions),
                      "local_tokens":sum(p >= local_first for p in positions),"consecutive_selected_overlap":ratio}
            records.append(record)
            if ratio is not None: by_head.setdefault(f"{layer}/{head}",[]).append(ratio)
            previous[key] = position,selected
    return {"definition":"intersection of consecutive selected block IDs divided by current selected block count; init/local token regions excluded",
            "result_sha256":digest(path),"selection_sha256":digest(trace),"code_commit":run["code_commit"],
            "overlap":distribution([row["consecutive_selected_overlap"] for row in records if row["consecutive_selected_overlap"] is not None]),
            "by_layer_head":{key:distribution(values) for key,values in by_head.items()},"records":records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("result",type=Path); parser.add_argument("output",type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists(): raise ValueError("selection analysis output already exists")
        atomic_json(args.output,analyze(args.result))
    except (OSError,ValueError,KeyError,TypeError) as error: parser.exit(2,f"selection_analysis: {error}\n")
