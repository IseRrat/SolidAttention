#!/usr/bin/env python3
"""Prepare, execute and collect reproducible experiment matrices."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

from experiment_common import (absolute_config, audit_run, check_hash, digest, distribution, load,
                               model_hashes, number, run_metrics, token_ids, atomic_json)
from experiment_runtime import Background, condition, fio_command, fio_observations, isolated_command, swap_limit
from process_lifetime import stop_process, termination_signals

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"origin", "quant", "indram", "indram-sparse", "offload", "offload-sparse", "interleaved", "prefetch", "overlap", "solid"}


def name(value):
    if not isinstance(value,str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*",value):
        raise ValueError("experiment names must be path-safe identifiers")
    return value


def variant_config(template, variant, block, budget):
    if variant not in VARIANTS: raise ValueError("unknown experiment variant")
    config = copy.deepcopy(template)
    for key in ("attention", "replay_token_ids", "prompt_token_ids"): config.pop(key,None)
    config.update(kv_dtype="q4_0" if variant == "quant" else "f16", dump_logits=False)
    if variant in ("origin", "quant", "indram"):
        if variant == "indram": config["kv_on_gpu"] = False
        return config
    sparse = variant != "offload"
    interleaved = variant in ("interleaved", "prefetch", "overlap", "solid")
    prefetch = variant in ("prefetch", "overlap", "solid")
    attention = {"storage": "memory" if variant == "indram-sparse" else "ssd", "sparse": sparse,
                 "layout": "interleaved" if interleaved else "separate", "fuse_kv_projection": interleaved,
                 "block_tokens": block, "representative_count": 1, "write_buffer_bytes_per_layer": 32768,
                 "selection_trace": False, "profile": False, "keep_session_file": False, "prefetch": prefetch}
    if sparse:
        selector = template.get("attention", {}).get("selection_backend", "auto")
        if selector not in ("auto", "cpu", "cuda"): raise ValueError("unknown selection backend")
        attention["selection_backend"] = selector
    if prefetch:
        attention.update(prefetch_trace=False, prefetch_queue_depth=64, schedule_trace=False,
                         schedule={"prefetch":"naive","overlap":"overlap","solid":"overlap+sync_reuse"}[variant])
    if sparse and budget:
        deterministic = max(1,budget//2)
        initial = min(block,deterministic//2)
        attention.update(context_budget_tokens=budget, init_tokens=initial, local_tokens=deterministic-initial)
    config["attention"] = attention
    return config


def create(spec_path, destination):
    spec_path, destination = Path(spec_path).resolve(strict=True), Path(destination).resolve()
    spec = load(spec_path)
    if set(spec) != {"schema_version","models","conditions","suites","repeats","warmups","seed","max_new_tokens"} or spec["schema_version"] != 1:
        raise ValueError("unsupported experiment specification fields/version")
    for key, minimum in (("repeats",1),("warmups",0),("seed",0),("max_new_tokens",2)): number(spec[key],minimum,True)
    conditions = {}
    for item in spec["conditions"]:
        item = copy.deepcopy(condition(item))
        key = name(item["name"])
        if key in conditions: raise ValueError("duplicate condition")
        item["session_root"] = str((spec_path.parent/item["session_root"]).resolve())
        if item.get("background"):
            item["background"]["file"] = str((spec_path.parent/item["background"]["file"]).resolve())
        conditions[key] = item
    if not conditions or not spec["models"] or not spec["suites"]: raise ValueError("empty experiment matrix")
    models = {}
    sources = [{"path":str(spec_path),"sha256":digest(spec_path)}]
    for model in spec["models"]:
        if set(model) != {"name","config","tokens","corpus"}: raise ValueError("model needs name, config, corpus and tokens")
        key = name(model["name"])
        if key in models: raise ValueError("duplicate model")
        config_path = (spec_path.parent/model["config"]).resolve(strict=True)
        bundle_path = (spec_path.parent/model["tokens"]).resolve(strict=True)
        config, bundle = absolute_config(config_path), load(bundle_path)
        config["prompt"] = str((spec_path.parent/model["corpus"]).resolve(strict=True))
        if bundle["schema_version"] != 1 or bundle["model_parts_sha256"] != model_hashes(config): raise ValueError("token corpus/model identity differs")
        if bundle["add_special"] is not True or bundle["parse_special"] is not True: raise ValueError("tokenizer settings differ")
        check_hash(Path(config["prompt"]), bundle["source_sha256"])
        token_ids(bundle["token_ids"])
        models[key] = config,bundle
        for path in (config_path,bundle_path,Path(config["prompt"])): sources.append({"path":str(path),"sha256":digest(path)})
    rows, suites = [], set()
    for suite in spec["suites"]:
        if set(suite) != {"name","lengths","variants","blocks","budgets","conditions"}: raise ValueError("invalid suite fields")
        suite_name = name(suite["name"])
        if suite_name in suites: raise ValueError("duplicate suite")
        suites.add(suite_name)
        for key in ("lengths","variants","blocks","budgets","conditions"):
            if not suite[key] or len(suite[key]) != len(set(suite[key])): raise ValueError("suite axes must be nonempty and unique")
        for length in suite["lengths"]: number(length,1,True)
        for block in suite["blocks"]: number(block,1,True)
        for budget in suite["budgets"]: number(budget,0,True)
        for model,(template,bundle) in models.items():
            for length in suite["lengths"]:
                if len(bundle["token_ids"]) < length: raise ValueError(f"corpus too short for {model}/{length}")
                for block in suite["blocks"]:
                    for budget in suite["budgets"]:
                        for setting in suite["conditions"]:
                            if setting not in conditions: raise ValueError("unknown suite condition")
                            for variant in suite["variants"]:
                                config = variant_config(template,variant,block,budget)
                                config.update(context_tokens=length+spec["max_new_tokens"]-1, max_new_tokens=spec["max_new_tokens"],
                                              ignore_eos=True, prompt_token_ids=bundle["token_ids"][:length])
                                if budget > config["context_tokens"]: raise ValueError("budget exceeds context capacity")
                                if config["batch_tokens"] > config["context_tokens"]: raise ValueError("batch exceeds experiment context")
                                group = dict(suite=suite_name,model=model,input_tokens=length,block_tokens=block,budget_tokens=budget,condition=setting,variant=variant)
                                for warmup,count in ((True,spec["warmups"]),(False,spec["repeats"])):
                                    for repeat in range(count):
                                        rows.append({"group":group,"warmup":warmup,"repeat":repeat,"config_value":copy.deepcopy(config)})
    # Keep warmups separate; randomize each block with a recorded seed.
    rng = random.Random(spec["seed"])
    warmups = [row for row in rows if row["warmup"]]; measured = [row for row in rows if not row["warmup"]]
    rng.shuffle(warmups); rng.shuffle(measured)
    rows = warmups+measured
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"configs").mkdir()
    for index,row in enumerate(rows):
        row["id"] = f"job-{index:06d}"
        config = row.pop("config_value")
        config["output"] = str(destination/"runs"/row["id"]/"result.json")
        if "attention" in config and config["attention"]["storage"] == "ssd":
            config["attention"]["session_file"] = str(Path(conditions[row["group"]["condition"]]["session_root"])/destination.name/(row["id"]+".cache"))
        path = destination/"configs"/(row["id"]+".json")
        atomic_json(path,config)
        row.update(config=str(path),config_sha256=digest(path),result=config["output"])
    path = destination/"plan.json"
    atomic_json(path,{"schema_version":1,"kind":"performance","seed":spec["seed"],"sources":sources,
                      "conditions":conditions,"jobs":rows,"input_policy":"exact-prefix-free-generation-fixed-output-length"})
    return path


def read_plan(path):
    path = Path(path).resolve(strict=True)
    plan = load(path)
    if plan["schema_version"] != 1 or not plan["jobs"]: raise ValueError("empty/unsupported experiment plan")
    ids = set(); outputs = set()
    for source in plan["sources"]: check_hash(source["path"],source["sha256"])
    for row in plan["jobs"]:
        name(row["id"])
        if row["id"] in ids or row["result"] in outputs: raise ValueError("duplicate job ID/output")
        ids.add(row["id"]); outputs.add(row["result"])
        check_hash(row["config"],row["config_sha256"])
        if str((path.parent/"runs"/row["id"]/"result.json").resolve()) != row["result"]: raise ValueError("run output escapes the plan directory")
        config = load(row["config"])
        if config["output"] != row["result"]: raise ValueError("configuration output differs from job")
        condition(plan["conditions"][row["group"]["condition"]])
    return path,plan


def audited_job(row, settings, attempt, runner_sha=None):
    result,run,config = audit_run(Path(row["result"]),row["config"],row["config_sha256"])
    if runner_sha is not None and run["runner_sha256"] != runner_sha: raise ValueError("runner differs from completed run")
    if len(result["generated_token_ids"]) != config["max_new_tokens"] or result["stop_reason"] != "length":
        raise ValueError("performance run did not produce the fixed output length")
    metrics = run_metrics(result)
    group = run.get("cgroup_end",run.get("cgroup_last",{}))
    if settings.get("memory_max_bytes") is not None and (
            group.get("memory.max") != str(settings["memory_max_bytes"]) or group.get("memory.swap.max") != str(swap_limit(settings))):
        raise ValueError("observed cgroup limits differ from the experiment condition")
    reads = None
    if settings.get("background"):
        observations = fio_observations(Path(attempt)/"fio.json")
        jobs = observations[-1].get("jobs",[]) if observations else []
        if not jobs or any(job["error"] or not job["read"]["total_ios"] for job in jobs):
            raise ValueError("fio report is missing, contains errors or has no completed reads")
        reads = [job["read"] for job in jobs]
    return result,run,config,metrics,group,reads


def execute(plan_path, runner, limit=None):
    if platform.system() != "Linux": raise ValueError("execute experiment plans in the Linux experiment environment")
    plan_path,plan = read_plan(plan_path)
    runner = Path(runner).resolve(strict=True)
    ledger = plan_path.parent/"execution.json"
    identity = {"plan_sha256":digest(plan_path),"runner_sha256":digest(runner)}
    state = load(ledger) if ledger.exists() else dict(identity,jobs={})
    if any(state[key] != value for key,value in identity.items()): raise ValueError("execution inputs changed; prepare a fresh plan")
    lock_path = plan_path.parent/"execution.lock"
    # POSIX locks are released by the kernel after interruption; resume keeps all attempt files.
    import fcntl
    with termination_signals(), lock_path.open("a+") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("another executor owns this experiment plan")
        if ledger.exists():
            state = load(ledger)
            if any(state[key] != value for key,value in identity.items()): raise ValueError("execution inputs changed")
        completed = 0
        for row in plan["jobs"]:
            settings = plan["conditions"][row["group"]["condition"]]
            attempt = plan_path.parent/"attempts"/row["id"]
            old = state["jobs"].get(row["id"])
            if old:
                if old["status"] == "running":
                    old["status"] = "interrupted"
                    atomic_json(ledger,state)
                if old["status"] == "complete": audited_job(row,settings,attempt,identity["runner_sha256"])
                # Terminal failures remain visible; reruns use a new plan.
                continue
            if limit is not None and completed >= limit: break
            output = Path(row["result"])
            attempt.mkdir(parents=True,exist_ok=False)
            config = load(row["config"])
            protected = [config["model"],config["prompt"],output,config.get("attention",{}).get("session_file",output)]
            protected.extend(part["path"] for part in config.get("additional_model_files",[]))
            command = isolated_command([sys.executable,str(ROOT/"scripts/run_baseline.py"),row["config"],"--runner",str(runner)],settings)
            record = {"status":"running","command":command,"started_utc":datetime.now(timezone.utc).isoformat()}
            state["jobs"][row["id"]] = record
            atomic_json(ledger,state)
            start = time.monotonic()
            try:
                background = fio_command(settings,attempt,protected)
                record["background_command"] = background
                atomic_json(ledger,state)
                with Background(background,attempt) as load_process, (attempt/"stdout.txt").open("wb") as stdout, (attempt/"stderr.txt").open("wb") as stderr:
                    process = subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
                    try:
                        while process.poll() is None:
                            load_process.check(); time.sleep(.1)
                    except BaseException as primary:
                        try: stop_process(process,group=True)
                        except BaseException as cleanup: raise primary from cleanup
                        raise
                    load_process.check()
                record["returncode"] = process.returncode
                if process.returncode: record["status"] = "failed"
                else:
                    audited_job(row,settings,attempt,identity["runner_sha256"])
                    record["status"] = "complete"
            except KeyboardInterrupt:
                record["status"] = "interrupted"; raise
            except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
                record.update(status="failed",error=str(error))
            finally:
                record["elapsed_seconds"] = time.monotonic()-start
                record["artifacts"] = {path.name:digest(path) for path in attempt.iterdir() if path.is_file()}
                manifest_path = output.parent/"manifest.json"
                if manifest_path.exists(): record["manifest_sha256"] = digest(manifest_path)
                atomic_json(ledger,state)
            completed += 1
    return state


def collect(plan_path, destination):
    plan_path,plan = read_plan(plan_path)
    ledger_path = plan_path.parent/"execution.json"
    state = load(ledger_path) if ledger_path.exists() else {"plan_sha256":digest(plan_path),"jobs":{}}
    if state["plan_sha256"] != digest(plan_path): raise ValueError("execution ledger uses another plan")
    runs, groups, identity = [],{},None
    for row in plan["jobs"]:
        record = state["jobs"].get(row["id"],{"status":"pending"})
        result_path = Path(row["result"])
        item = {key:copy.deepcopy(row[key]) for key in ("id","group","repeat","warmup","result")}
        item["status"] = record["status"]
        if "manifest_sha256" in record: check_hash(result_path.parent/"manifest.json",record["manifest_sha256"])
        for artifact,sha in record.get("artifacts",{}).items():
            if Path(artifact).name != artifact: raise ValueError("invalid attempt artifact name")
            check_hash(plan_path.parent/"attempts"/row["id"]/artifact,sha)
        if record["status"] == "complete":
            settings = plan["conditions"][row["group"]["condition"]]
            result,run,config,metrics,group,reads = audited_job(row,settings,plan_path.parent/"attempts"/row["id"],state.get("runner_sha256"))
            key = run["code_commit"],run["runner_sha256"],run["upstream_commit"]
            if identity is None: identity = key
            elif identity != key: raise ValueError("experiment runs mix code or runner identities")
            if state.get("runner_sha256") != run["runner_sha256"]: raise ValueError("ledger runner differs from completed runs")
            item["metrics"] = metrics
            item["manifest_sha256"] = digest(result_path.parent/"manifest.json")
            item["result_sha256"] = digest(result_path)
            item["sampled_peak_rss_bytes"] = run.get("sampled_peak_rss_bytes")
            item["cgroup_end"] = group
            if reads is not None: item["fio_observed_reads"] = reads
        elif "manifest_sha256" in record:
            failed = load(result_path.parent/"manifest.json")
            item["failure"] = {key:failed.get(key) for key in ("status","returncode","cgroup_start","cgroup_end")}
            item["failure"]["executor_error"] = record.get("error")
        if not row["warmup"]:
            group_key = json.dumps(row["group"],sort_keys=True)
            groups.setdefault(group_key,[]).append(item)
        runs.append(item)
    summaries = []
    for items in groups.values():
        good = [row for row in items if row["status"] == "complete"]
        summaries.append({"group":items[0]["group"],"expected":len(items),"complete":len(good),
                          "status_counts":{status:sum(row["status"] == status for row in items) for status in sorted({row["status"] for row in items})},
                          "metrics":{key:distribution([row["metrics"][key] for row in good if row["metrics"][key] is not None])
                                     for key in ("decode_tokens_per_second","decode_loop_ms","prefill_ms","first_token_ms","request_ms","kv_system_tracked_bytes")},
                          "run_ids":[row["id"] for row in items]})
    output = {"schema_version":1,"plan":str(plan_path),"plan_sha256":digest(plan_path),"execution_sha256":digest(ledger_path) if ledger_path.exists() else None,
              "provenance":list(identity) if identity else None,"percentile":"linear interpolation at (n-1)*q; per-run samples and repeat distributions kept separately",
              "groups":summaries,"runs":runs}
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/"summary.json",output)
    with (destination/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        fields = ["suite","model","input_tokens","block_tokens","budget_tokens","condition","variant","expected","complete","median_tokens_per_second","min_tokens_per_second","max_tokens_per_second"]
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for group in summaries:
            metric = group["metrics"]["decode_tokens_per_second"]
            writer.writerow(dict(group["group"],expected=group["expected"],complete=group["complete"],
                                 **{key+"_tokens_per_second":metric[key] if metric else None for key in ("median","min","max")}))
    lines = ["# 实验记录", "",f"计划：`{plan_path}`；SHA256：`{output['plan_sha256']}`。", "",
             "| 实验组 | 完成 / 计划 | decode token/s 中位数 |", "| --- | --- | --- |"]
    for group in summaries:
        metric = group["metrics"]["decode_tokens_per_second"]
        lines.append(f"| {' / '.join(str(v) for v in group['group'].values())} | {group['complete']} / {group['expected']} | {metric['median'] if metric else '待采集'} |")
    (destination/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action",required=True)
    p = actions.add_parser("prepare"); p.add_argument("spec",type=Path); p.add_argument("destination",type=Path)
    p = actions.add_parser("run"); p.add_argument("plan",type=Path); p.add_argument("--runner",type=Path,default=ROOT/"build/release/bin/solid-baseline"); p.add_argument("--limit",type=int)
    p = actions.add_parser("collect"); p.add_argument("plan",type=Path); p.add_argument("destination",type=Path)
    args = parser.parse_args()
    if args.action == "prepare": print(create(args.spec,args.destination))
    elif args.action == "run":
        if args.limit is not None: number(args.limit,1,True)
        state = execute(args.plan,args.runner,args.limit)
        return 1 if any(job["status"] != "complete" for job in state["jobs"].values()) else 0
    else: collect(args.plan,args.destination); print(args.destination)
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f"experiments: {error}",file=sys.stderr); sys.exit(2)
