"""Serialized model client shared by LongBench and OpenCompass adapters."""
from array import array
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

from compare_logits import load_result
from experiment_common import absolute_config, atomic_json, audit_run, digest, model_hashes, number, token_ids
from experiments import variant_config

ROOT = Path(__file__).resolve().parents[1]


def truncate(tokens, capacity, policy):
    number(capacity,1,True)
    if len(tokens) <= capacity: return list(tokens)
    if policy != "middle": raise ValueError("prompt exceeds the available context; select middle truncation explicitly")
    first = (capacity+1)//2
    return tokens[:first] + (tokens[-(capacity-first):] if capacity > first else [])


def mean_nll(result_path, targets, denominator=None):
    value,binary = load_result(Path(result_path))
    token_ids(targets)
    if value["logits"]["rows"] != len(targets) or value["decode_input_token_ids"] != targets[:-1]:
        raise ValueError("teacher-forced targets differ from recorded logits/input rows")
    columns = value["logits"]["columns"]
    total = 0.0
    with binary.open("rb") as stream:
        for target in targets:
            row = array("f"); row.frombytes(stream.read(columns*4))
            if sys.byteorder != "little": row.byteswap()
            if target >= columns or any(not math.isfinite(x) for x in row): raise ValueError("invalid target or nonfinite logits")
            maximum = max(row)
            total += math.log(math.fsum(math.exp(x-maximum) for x in row)) + maximum-row[target]
    denominator = len(targets) if denominator is None else number(denominator,1,True)
    return total/denominator


class RunnerClient:
    def __init__(self, template, runner, output, variant="solid", max_seq_len=131583, truncation="error", block_tokens=32, budget_tokens=0):
        self.template = absolute_config(template)
        self.runner = Path(runner).resolve(strict=True)
        self.output = Path(output).resolve()
        self.variant = variant
        self.max_seq_len = number(max_seq_len,2,True)
        if truncation not in ("error","middle"): raise ValueError("unknown truncation policy")
        self.truncation = truncation
        self.block_tokens = number(block_tokens,1,True)
        self.budget_tokens = number(budget_tokens,0,True)
        self.cache = OrderedDict()

    def tokenize(self, text):
        if not isinstance(text,str): raise ValueError("model input must be a rendered string")
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key); return list(self.cache[key])
        folder = self.output/"tokenization"/uuid.uuid4().hex
        folder.mkdir(parents=True)
        prompt = folder/"prompt.txt"; prompt.write_text(text,encoding="utf-8",newline="")
        config = copy.deepcopy(self.template)
        for name in ("prompt_token_ids","replay_token_ids"): config.pop(name,None)
        config.update(prompt=str(prompt),output=str(folder/"unused.json"))
        atomic_json(folder/"config.json",config)
        with (folder/"stderr.txt").open("wb") as errors:
            result = subprocess.run([str(self.runner),"--tokenize",str(folder/"config.json")],stdout=subprocess.PIPE,stderr=errors,check=True)
        bundle = json.loads(result.stdout)
        if bundle["model_parts_sha256"] != model_hashes(config) or bundle["source_sha256"] != digest(prompt): raise ValueError("tokenizer identity differs")
        tokens = token_ids(bundle["token_ids"])
        atomic_json(folder/"tokens.json",bundle)
        self.cache[key] = tokens
        if len(self.cache) > 128: self.cache.popitem(last=False)
        return list(tokens)

    def _run(self, text, prompt_tokens, outputs, replay=None, scoring_input_tokens=None):
        folder = self.output/"calls"/uuid.uuid4().hex
        folder.mkdir(parents=True)
        prompt = folder/"prompt.txt"; prompt.write_text(text,encoding="utf-8",newline="")
        config = variant_config(self.template,self.variant,self.block_tokens,self.budget_tokens)
        config.update(prompt=str(prompt),output=str(folder/"run/result.json"),context_tokens=self.max_seq_len,
                      prompt_token_ids=prompt_tokens,max_new_tokens=outputs,ignore_eos=replay is not None,dump_logits=replay is not None)
        if replay is not None:
            config["replay_token_ids"] = replay
            if config.get("attention",{}).get("sparse") and not self.budget_tokens:
                # Whole-sequence scoring uses the visible evaluation input to
                # choose the same fixed request budget across candidate labels.
                length = scoring_input_tokens if scoring_input_tokens is not None else len(prompt_tokens)
                total = max(1,length//4) if length < 4096 else 1024
                deterministic = max(1,total//2)
                initial = min(self.block_tokens,deterministic//2)
                config["attention"].update(context_budget_tokens=total,init_tokens=initial,local_tokens=deterministic-initial)
        path = folder/"source.json"; atomic_json(path,config)
        with (folder/"stdout.txt").open("wb") as stdout,(folder/"stderr.txt").open("wb") as stderr:
            subprocess.run([sys.executable,str(ROOT/"scripts/run_baseline.py"),str(path),"--runner",str(self.runner)],stdout=stdout,stderr=stderr,check=True)
        value,manifest,_ = audit_run(Path(config["output"]),path,digest(path))
        return value,{"result":config["output"],"result_sha256":digest(Path(config["output"])),
                      "manifest_sha256":digest(Path(config["output"]).parent/"manifest.json"),
                      "code_commit":manifest["code_commit"],"runner_sha256":manifest["runner_sha256"],
                      "truncation":self.truncation,"prompt_tokens":len(prompt_tokens)}

    def generate(self, text, max_new_tokens=512):
        number(max_new_tokens,1,True)
        tokens = truncate(self.tokenize(text),self.max_seq_len-max_new_tokens+1,self.truncation)
        return self._run(text,tokens,max_new_tokens)

    def score(self, text, mask_length=None):
        tokens = self.tokenize(text)
        if len(tokens) > self.max_seq_len: raise ValueError("PPL scoring requires inputs within max_seq_len")
        prefix = 1 if mask_length is None else number(mask_length,1,True)
        if prefix >= len(tokens): raise ValueError("PPL mask leaves no target tokens")
        targets = tokens[prefix:]
        value,record = self._run(text,tokens[:prefix],len(targets),targets[:-1],len(tokens) if mask_length is None else prefix)
        # OpenCompass's HF interface returns average cross entropy. Its
        # unmasked denominator includes the initial prompt token.
        score = mean_nll(record["result"],targets,len(tokens) if mask_length is None else len(targets))
        record.update(mask_length=mask_length,scored_tokens=len(targets),mean_nll=score)
        atomic_json(Path(record["result"]).parent.parent/"score.json",record)
        return score,record
