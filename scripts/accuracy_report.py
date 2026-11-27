#!/usr/bin/env python3
# 整理官方准确率汇总表，保留模型、数据集与评分器版本信息。
"""Import an official OpenCompass summary column with audited backend provenance."""
import argparse
import csv
from importlib import metadata
from pathlib import Path
import subprocess

from experiment_common import atomic_json, audit_run, digest, load, number

TASKS = {"winogrande","arc-challenge","mmlu","gsm8k","longbench"}
LONG_EIGHT = {"2wikimqa","triviaqa","hotpotqa","multifieldqa_en","musique","narrativeqa","qasper","gov_report"}


def import_summary(summary, model_column, mapping, repository, revision, config, backend, destination, longbench=None):
    repository = Path(repository).resolve(strict=True)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision): raise ValueError("pin a full OpenCompass commit SHA")
    def git(*args): return subprocess.run(["git","-C",str(repository),*args],capture_output=True,text=True,check=True).stdout.strip()
    if git("rev-parse","HEAD") != revision or git("status","--porcelain"): raise ValueError("OpenCompass checkout differs from the pinned clean revision")
    summary,mapping,config,backend = map(lambda path:Path(path).resolve(strict=True),(summary,mapping,config,backend))
    task_map = load(mapping)
    if not task_map or not set(task_map.values()) <= TASKS or len(task_map.values()) != len(set(task_map.values())):
        raise ValueError("map one official summary row to each requested task")
    scores = {}
    with summary.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "dataset" not in reader.fieldnames or model_column not in reader.fieldnames:
            raise ValueError("OpenCompass CSV needs dataset and the requested model column")
        for row in reader:
            if row["dataset"] not in task_map: continue
            task = task_map[row["dataset"]]
            if task in scores: raise ValueError("duplicate selected OpenCompass score row")
            score = number(float(row[model_column]))
            if score > 100: raise ValueError("accuracy score exceeds 100 percent")
            scores[task] = score
    identity = None; runs = []
    for path in sorted(backend.glob("calls/*/run/result.json")):
        result,manifest,_ = audit_run(path)
        key = (manifest["code_commit"],manifest["runner_sha256"],result["kv_dtype"],result.get("attention",{}).get("sparse",False),
               tuple(result["model_parts_sha256"]),result.get("attention",{}).get("schedule","naive"))
        if identity is None: identity = key
        elif key != identity: raise ValueError("backend records mix model approach, code or binaries")
        runs.append({"result":str(path),"sha256":digest(path),"manifest_sha256":digest(path.parent/"manifest.json")})
    if not runs: raise ValueError("accuracy summary needs audited backend model calls")
    provenance = {"summary_sha256":digest(summary),"mapping_sha256":digest(mapping),"config_sha256":digest(config),
                  "framework_revision":revision,"model_column":model_column,"backend_provenance":list(identity),"runs":runs}
    if longbench is not None:
        value = load(longbench)
        if "longbench" in scores or set(value["scores_percent"]) != LONG_EIGHT: raise ValueError("LongBench needs the eight requested dataset scores")
        if value["provenance"] != list(identity[:2]): raise ValueError("LongBench and OpenCompass use different code/binaries")
        expected_model = {"model_parts_sha256":list(identity[4]),"kv_dtype":identity[2],"sparse":identity[3],"schedule":identity[5]}
        if value["model_identity"] != expected_model: raise ValueError("LongBench and OpenCompass use different models/approaches")
        for score in value["scores_percent"].values():
            number(score)
            if score > 100: raise ValueError("LongBench score exceeds 100 percent")
        scores["longbench"] = sum(value["scores_percent"].values())/8
        provenance["longbench_sha256"] = digest(Path(longbench))
    if set(scores) != TASKS: raise ValueError("accuracy table requires all five task columns")
    result = {"scores_percent":scores,"average_percent":sum(scores.values())/5,"aggregation":"arithmetic mean across five task columns; LongBench=mean of eight datasets",
              "provenance":provenance,"packages":{key:package_version(key) for key in ("opencompass","numpy","torch","rouge","jieba","fuzzywuzzy")}}
    destination = Path(destination)
    if destination.exists(): raise ValueError("accuracy report already exists")
    atomic_json(destination,result)
    return result


def package_version(package):
    try: return metadata.version(package)
    except metadata.PackageNotFoundError: return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("summary","mapping","repository","config","backend","destination"): parser.add_argument(key,type=Path)
    parser.add_argument("--model-column",required=True); parser.add_argument("--revision",required=True); parser.add_argument("--longbench",type=Path)
    args = parser.parse_args()
    try: import_summary(args.summary,args.model_column,args.mapping,args.repository,args.revision,args.config,args.backend,args.destination,args.longbench)
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error: parser.exit(2,f"accuracy_report: {error}\n")
