#!/usr/bin/env python3
"""Prepare rendered evaluation samples, generate predictions and use official LongBench scoring."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys

from accuracy_client import RunnerClient
from experiment_common import atomic_json, audit_run, check_hash, digest, load, number
from experiments import name

LONGBENCH = {"2wikimqa","triviaqa","hotpotqa","multifieldqa_en","multifieldqa_zh","musique","narrativeqa","qasper","gov_report",
             "passage_retrieval_en","passage_retrieval_zh"}


def prepare(source, template, destination, dataset, revision, max_new_tokens=512):
    name(dataset)
    if not revision: raise ValueError("dataset revision must be recorded")
    number(max_new_tokens,1,True)
    source,template,destination = Path(source).resolve(strict=True),Path(template).resolve(strict=True),Path(destination).resolve()
    text = template.read_text(encoding="utf-8")
    samples,ids = [],set()
    with source.open(encoding="utf-8") as stream:
        for line_number,line in enumerate(stream,1):
            row = json.loads(line)
            prompt = text.format_map(row)
            sample_id = str(row.get("_id",row.get("id",hashlib.sha256(line.encode("utf-8")).hexdigest())))
            if not sample_id or sample_id in ids: raise ValueError("duplicate/empty sample ID")
            ids.add(sample_id)
            if not isinstance(row["answers"],list) or not row["answers"] or any(not isinstance(a,str) for a in row["answers"]):
                raise ValueError("evaluation samples require nonempty string answers")
            samples.append({"id":sample_id,"dataset":dataset,"prompt":prompt,"answers":row["answers"],
                            "all_classes":row.get("all_classes",[]),"length":row.get("length",0),"max_new_tokens":max_new_tokens})
    if not samples: raise ValueError("empty evaluation dataset")
    destination.mkdir(parents=True,exist_ok=False)
    with (destination/"samples.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for row in samples: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    manifest = {"schema_version":1,"dataset":dataset,"revision":revision,"sample_count":len(samples),
                "source":str(source),"source_sha256":digest(source),"template":str(template),"template_sha256":digest(template),
                "samples_sha256":digest(destination/"samples.jsonl")}
    atomic_json(destination/"dataset.json",manifest)
    return destination/"dataset.json"


def read_samples(manifest):
    manifest = Path(manifest).resolve(strict=True)
    metadata = load(manifest)
    if metadata["schema_version"] != 1: raise ValueError("unsupported dataset schema")
    for key in ("source","template"): check_hash(metadata[key],metadata[key+"_sha256"])
    check_hash(manifest.parent/"samples.jsonl",metadata["samples_sha256"])
    rows = [json.loads(line) for line in (manifest.parent/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != metadata["sample_count"] or len({row["id"] for row in rows}) != len(rows): raise ValueError("dataset sample coverage differs")
    return manifest,rows


def generate(manifest, template, runner, output, variant="solid", max_seq_len=131583, truncation="error", block_tokens=32, budget_tokens=0):
    manifest,rows = read_samples(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/"generation.lock").open("a+") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("another generator owns this output directory")
        identity = {"dataset_sha256":digest(manifest),"template_sha256":digest(Path(template)),"runner_sha256":digest(Path(runner)),
                    "variant":variant,"max_seq_len":max_seq_len,"truncation":truncation,"block_tokens":block_tokens,"budget_tokens":budget_tokens}
        state_path = output/"predictions.json"
        state = load(state_path) if state_path.exists() else dict(identity,predictions={})
        if any(state[key] != value for key,value in identity.items()): raise ValueError("generation inputs changed; use another output directory")
        client = RunnerClient(template,runner,output/"backend",variant,max_seq_len,truncation,block_tokens,budget_tokens)
        for row in rows:
            if row["id"] in state["predictions"]:
                record = state["predictions"][row["id"]]["run"]
                check_hash(record["result"],record["result_sha256"])
                check_hash(Path(record["result"]).parent/"manifest.json",record["manifest_sha256"])
                audit_run(record["result"])
                continue
            result,record = client.generate(row["prompt"],row["max_new_tokens"])
            state["predictions"][row["id"]] = {"pred":result["text"],"run":record}
            atomic_json(state_path,state)
        return state_path


def export(manifest, predictions, destination):
    manifest,rows = read_samples(manifest)
    predictions = Path(predictions).resolve(strict=True)
    state = load(predictions)
    if state["dataset_sha256"] != digest(manifest) or set(state["predictions"]) != {row["id"] for row in rows}:
        raise ValueError("prediction identity/coverage differs from dataset")
    output = Path(destination); output.mkdir(parents=True,exist_ok=False)
    groups = {}; identity = None; model_identity = None
    for row in rows:
        record = state["predictions"][row["id"]]
        run = record["run"]
        check_hash(run["result"],run["result_sha256"])
        check_hash(Path(run["result"]).parent/"manifest.json",run["manifest_sha256"])
        result,metadata,_ = audit_run(run["result"])
        if result["text"] != record["pred"]: raise ValueError("prediction differs from recorded model result")
        key = metadata["code_commit"],metadata["runner_sha256"]
        if identity is None: identity = key
        elif identity != key: raise ValueError("accuracy predictions mix code/binary identities")
        model = {"model_parts_sha256":result["model_parts_sha256"],"kv_dtype":result["kv_dtype"],
                 "sparse":result.get("attention",{}).get("sparse",False),"schedule":result.get("attention",{}).get("schedule","naive")}
        if model_identity is None: model_identity = model
        elif model_identity != model: raise ValueError("accuracy predictions mix models or approaches")
        groups.setdefault(row["dataset"],[]).append({"id":row["id"],"pred":record["pred"],"answers":row["answers"],
                                                    "all_classes":row["all_classes"],"length":row["length"]})
    for dataset,items in groups.items():
        name(dataset)
        with (output/(dataset+".jsonl")).open("w",encoding="utf-8") as stream:
            for row in items: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    atomic_json(output/"export.json",{"schema_version":1,"dataset_sha256":digest(manifest),"predictions_sha256":digest(predictions),
                "provenance":identity,"model_identity":model_identity,"files":{path.name:digest(path) for path in output.glob("*.jsonl")}})
    return output


@contextmanager
def official_longbench(root):
    """Load both official modules from their source bytes for this invocation."""
    keys = ("metrics","solid_official_longbench")
    missing = object()
    previous = {key:sys.modules.get(key,missing) for key in keys}
    search_path = sys.path[:]
    bytecode = sys.dont_write_bytecode
    hashes = {}
    try:
        sys.path.insert(0,str(root))
        sys.dont_write_bytecode = True
        modules = []
        for key,filename in zip(keys,("metrics.py","eval.py")):
            path = root/filename
            source = path.read_bytes()
            hashes[filename] = hashlib.sha256(source).hexdigest()
            spec = importlib.util.spec_from_file_location(key,path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            exec(compile(source,str(path),"exec"),module.__dict__)
            modules.append(module)
        yield modules[-1],hashes
        for filename,sha in hashes.items(): check_hash(root/filename,sha)
    finally:
        for key,value in previous.items():
            if value is missing: sys.modules.pop(key,None)
            else: sys.modules[key] = value
        sys.path[:] = search_path
        sys.dont_write_bytecode = bytecode


def score_longbench(directory, repository, revision, destination):
    directory,repository = Path(directory).resolve(strict=True),Path(repository).resolve(strict=True)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision): raise ValueError("pin a full official LongBench commit SHA")
    def git(*args): return subprocess.run(["git","-C",str(repository),*args],capture_output=True,text=True,check=True).stdout.strip()
    if git("rev-parse","HEAD") != revision or git("status","--porcelain"): raise ValueError("LongBench checkout differs from the pinned clean revision")
    metadata = load(directory/"export.json")
    for source in metadata.get("sources",[]): check_hash(source["path"],source["sha256"])
    root = repository/"LongBench"
    if not (root/"eval.py").is_file(): raise ValueError("expected LongBench/eval.py in the official repository")
    scores,counts = {},{}
    if not metadata["files"]: raise ValueError("empty prediction export")
    with official_longbench(root) as (official,source_hashes):
        for filename,sha in metadata["files"].items():
            dataset = Path(filename).stem
            if Path(filename).name != filename or dataset not in LONGBENCH: raise ValueError("unsupported LongBench dataset")
            check_hash(directory/filename,sha)
            rows = [json.loads(line) for line in (directory/filename).read_text(encoding="utf-8").splitlines()]
            if not rows: raise ValueError("empty LongBench prediction file")
            if any(row["all_classes"] != rows[0]["all_classes"] for row in rows): raise ValueError("inconsistent class lists")
            score = official.scorer(dataset,[row["pred"] for row in rows],[row["answers"] for row in rows],rows[0]["all_classes"])
            number(score)
            scores[dataset] = score; counts[dataset] = len(rows)
    if git("rev-parse","HEAD") != revision or git("status","--porcelain"): raise ValueError("LongBench checkout changed while scoring")
    result = {"scores_percent":scores,"sample_counts":counts,"official_revision":revision,
              "official_files_sha256":source_hashes,
              "export_sha256":digest(directory/"export.json"),"provenance":metadata["provenance"],"model_identity":metadata["model_identity"]}
    destination = Path(destination)
    if destination.exists(): raise ValueError("score output already exists")
    atomic_json(destination,result)
    return result


def merge_exports(sources, destination):
    files = {}; identity = None; receipts = []
    if not sources: raise ValueError("merge requires prediction exports")
    for source in sources:
        source = Path(source).resolve(strict=True)
        metadata = load(source/"export.json")
        key = metadata["provenance"],metadata["model_identity"]
        if identity is None: identity = key
        elif identity != key: raise ValueError("prediction exports use different models, approaches or code")
        receipts.append({"path":str(source/"export.json"),"sha256":digest(source/"export.json")})
        for filename,sha in metadata["files"].items():
            if filename in files or Path(filename).name != filename: raise ValueError("duplicate or invalid prediction filename")
            check_hash(source/filename,sha)
            files[filename] = source/filename,sha
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for filename,(source,sha) in files.items():
        shutil.copyfile(source,destination/filename)
        check_hash(destination/filename,sha)
    atomic_json(destination/"export.json",{"schema_version":1,"provenance":identity[0],"model_identity":identity[1],
                "sources":receipts,"files":{filename:sha for filename,(_,sha) in files.items()}})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_subparsers(dest="action",required=True)
    p = actions.add_parser("prepare"); p.add_argument("source",type=Path); p.add_argument("template",type=Path); p.add_argument("destination",type=Path)
    p.add_argument("--dataset",required=True); p.add_argument("--revision",required=True); p.add_argument("--max-new-tokens",type=int,default=512)
    p = actions.add_parser("run"); p.add_argument("manifest",type=Path); p.add_argument("template",type=Path); p.add_argument("output",type=Path)
    p.add_argument("--runner",required=True,type=Path); p.add_argument("--variant",default="solid"); p.add_argument("--max-seq-len",type=int,default=131583); p.add_argument("--truncation",choices=("error","middle"),default="error")
    p.add_argument("--block-tokens",type=int,default=32); p.add_argument("--budget-tokens",type=int,default=0)
    p = actions.add_parser("export"); p.add_argument("manifest",type=Path); p.add_argument("predictions",type=Path); p.add_argument("destination",type=Path)
    p = actions.add_parser("score-longbench"); p.add_argument("directory",type=Path); p.add_argument("repository",type=Path); p.add_argument("destination",type=Path); p.add_argument("--revision",required=True)
    p = actions.add_parser("merge"); p.add_argument("destination",type=Path); p.add_argument("sources",nargs="+",type=Path)
    args = parser.parse_args()
    if args.action == "prepare": print(prepare(args.source,args.template,args.destination,args.dataset,args.revision,args.max_new_tokens))
    elif args.action == "run": print(generate(args.manifest,args.template,args.runner,args.output,args.variant,args.max_seq_len,args.truncation,args.block_tokens,args.budget_tokens))
    elif args.action == "export": print(export(args.manifest,args.predictions,args.destination))
    elif args.action == "merge": print(merge_exports(args.sources,args.destination))
    else: print(json.dumps(score_longbench(args.directory,args.repository,args.revision,args.destination),indent=2))


if __name__ == "__main__":
    try: main()
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error:
        print(f"accuracy: {error}",file=sys.stderr); sys.exit(2)
