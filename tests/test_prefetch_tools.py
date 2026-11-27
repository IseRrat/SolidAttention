# 预取对照工具和槽位使用记录审计检查。
from array import array
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_prefetch
import compare_prefetch


def write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def residency():
    identity = {"slot": 0, "generation": 1, "layer": 0, "kv_head": 0, "block": 0, "version": 2, "epoch": 1}
    return [dict(identity, event=name) for name in ("upload", "upload_complete", "attention_pin", "attention_complete")]


class PrefetchToolsTests(unittest.TestCase):
    def fixture(self, root):
        source, target = root / "reference", root / "candidate"
        source.mkdir(); target.mkdir()
        attention = {"storage": "ssd", "sparse": True, "layout": "interleaved", "fuse_kv_projection": True,
                     "prefetch": False, "block_tokens": 2, "context_budget_tokens": 4, "init_tokens": 1,
                     "local_tokens": 1, "representative_count": 1}
        result = {"attention": attention, "model_parts_sha256": ["a" * 64], "prompt_token_ids": [7],
                  "generated_token_ids": [1, 2], "decode_input_token_ids": [8], "timing": {"decode_steps": 1}}
        binary = array("f", [1, 2, 3, 4])
        if sys.byteorder != "little": binary.byteswap()
        selection = {"layer": 0, "position": 3, "kv_head": 0, "ranges": [[0, 1], [3, 1]]}
        for folder in (source, target):
            (folder / "logits.f32").write_bytes(binary.tobytes())
            write(folder / "selection.jsonl", selection)
            (folder / "prompt.txt").write_text("fixed prompt")
        result["selection_trace_sha256"] = compare_prefetch.digest(source / "selection.jsonl")
        result["logits"] = {"file": "logits.f32", "dtype": "float32-le", "rows": 2, "columns": 2,
                            "sha256": compare_prefetch.digest(source / "logits.f32")}
        config = {"model": "/models/fixed.gguf", "context_tokens": 32, "threads": 4,
                  "attention": dict(attention, session_file="old.cache", selection_replay="old-selection.jsonl")}
        write(source / "config.json", config)
        write(source / "result.json", result)
        candidate = copy.deepcopy(result)
        candidate["attention"]["prefetch"] = True
        candidate["prefetch"] = {"slots_per_head": 2, "cancel_requests": 1, "cancel_completions": 1}
        (target / "prefetch.jsonl").write_text("")
        (target / "residency.jsonl").write_text("".join(json.dumps(row) + "\n" for row in residency()))
        candidate["prefetch_trace_sha256"] = compare_prefetch.digest(target / "prefetch.jsonl")
        candidate["residency_trace_sha256"] = compare_prefetch.digest(target / "residency.jsonl")
        write(target / "config.json", config)
        write(target / "result.json", candidate)
        for folder in (source, target): self.manifest(folder)
        return source, target

    def manifest(self, folder):
        write(folder / "manifest.json", {"status": "complete", "returncode": 0, "code_commit": "code", "runner_sha256": "runner",
              "result_sha256": compare_prefetch.digest(folder / "result.json"),
              "effective_config_sha256": compare_prefetch.digest(folder / "config.json")})

    def test_preparation_preserves_inputs_and_resets_session_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = self.fixture(root)
            config = prepare_prefetch.prepare_prefetch(source / "result.json", target / "other.json")
            self.assertEqual(config["replay_token_ids"], [8])
            self.assertEqual(config["context_tokens"], 32)
            self.assertTrue(config["attention"]["prefetch_trace"])
            self.assertNotIn("selection_replay", config["attention"])
            self.assertNotIn("session_file", config["attention"])
            fixed = prepare_prefetch.prepare_prefetch(source / "result.json", target / "other.json", True, 2, 4)
            self.assertEqual(fixed["attention"]["selection_replay"], str(source / "selection.jsonl"))
            self.assertEqual(fixed["attention"]["prefetch_slots_per_head"], 4)
            with self.assertRaises(ValueError): prepare_prefetch.prepare_prefetch(source / "result.json", target / "other.json", False, 1)
            with self.assertRaises(ValueError): prepare_prefetch.prepare_prefetch(source / "result.json", target / "other.json", False, 2, 17)
            saved = json.loads((source / "config.json").read_text())
            saved["attention"]["gpu_decode"] = False
            write(source / "config.json", saved)
            with self.assertRaises(ValueError): prepare_prefetch.prepare_prefetch(source / "result.json", target / "other.json")

    def test_comparison_checks_selection_and_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = self.fixture(Path(temporary))
            report = compare_prefetch.compare_prefetch(source / "result.json", target / "result.json")
            self.assertTrue(report["passed"])
            write(target / "selection.jsonl", {"layer": 0, "position": 3, "kv_head": 0, "ranges": [[0, 2], [3, 1]]})
            candidate = json.loads((target / "result.json").read_text())
            candidate["selection_trace_sha256"] = compare_prefetch.digest(target / "selection.jsonl")
            write(target / "result.json", candidate); self.manifest(target)
            self.assertFalse(compare_prefetch.compare_prefetch(source / "result.json", target / "result.json")["passed"])
            manifest = json.loads((target / "manifest.json").read_text())
            manifest["runner_sha256"] = "other runner"
            write(target / "manifest.json", manifest)
            with self.assertRaises(ValueError): compare_prefetch.compare_prefetch(source / "result.json", target / "result.json")

    def test_audit_rejects_early_overwrite_and_stale_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            good = residency()
            for rows in (good[:3] + [dict(good[0], generation=2)],
                         good[:3] + [dict(good[3], generation=0)],
                         good[:3] + [dict(good[3], epoch=2)]):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError): compare_prefetch.audit_residency(path, 2, 2)

    def test_audit_requires_transfer_and_attention_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            for rows in (residency()[:1], residency()[:3]):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError): compare_prefetch.audit_residency(path, 2, 2)
            path.write_text("".join(json.dumps(row) + "\n" for row in residency()))
            self.assertTrue(compare_prefetch.audit_residency(path, 2, 2)["balanced"])

    def test_current_device_prefix_requires_generation_and_complete_coverage(self):
        previous=residency()
        source=previous[0]
        upload=dict(source,generation=2,version=4,epoch=2,event="upload",copy_kind="current_device",current_position=3,
                    retained_prefix_tokens=2,prefix_generation=1,row_bytes=16,host_copy_bytes=16,device_copy_bytes=16)
        def rows(value):
            following={key:value[key] for key in source if key!="event"}
            return previous+[dict(source,event="overwrite_available"),value]+[dict(following,event=name) for name in
                ("upload_complete","attention_pin","attention_complete")]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"residency.jsonl"
            def audit(value):
                path.write_text("".join(json.dumps(row)+"\n" for row in rows(value)))
                return compare_prefetch.audit_residency(path,4,2)
            self.assertEqual(audit(upload)["retained_device_prefix_bytes"],32)
            self.assertEqual(audit(upload)["current_host_bytes"],16)
            cold=dict(upload,retained_prefix_tokens=0,prefix_generation=0,host_copy_bytes=48)
            self.assertEqual(audit(cold)["refreshed_device_tails"],0)
            for change in (dict(prefix_generation=0),dict(generation=3),dict(current_position=4),dict(device_copy_bytes=32),
                           dict(retained_prefix_tokens=1,host_copy_bytes=32),dict(host_copy_bytes=0),dict(row_bytes=True),dict(copy_kind="host")):
                with self.assertRaises(ValueError): audit(dict(upload,**change))
            missing=dict(upload); missing.pop("copy_kind")
            with self.assertRaises(ValueError): audit(missing)

    def test_current_device_result_counters_match_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target=self.fixture(Path(temporary))
            candidate=json.loads((target/"result.json").read_text()); candidate["prefetch"]["current_device_rows"]=1
            write(target/"result.json",candidate); self.manifest(target)
            with self.assertRaises(ValueError): compare_prefetch.compare_prefetch(source/"result.json",target/"result.json")


if __name__ == "__main__": unittest.main()
