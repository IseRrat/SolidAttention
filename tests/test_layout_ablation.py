import copy
from array import array
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ablation", SCRIPTS / "layout_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class LayoutAblationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        (source / "prompt.txt").write_text("prompt")
        (source / "config.json").write_text(json.dumps({"model": "/models/fixed.gguf", "context_tokens": 32,
            "gpu_layers": 4, "threads": 2, "attention": {"storage": "ssd", "sparse": True}}))
        selection = {"layer": 0, "position": 3, "kv_head": 0, "ranges": [[0, 1], [3, 1]]}
        (source / "selection.jsonl").write_text(json.dumps(selection) + "\n")
        logits = array("f", [1, 2, 3, 4])
        if sys.byteorder != "little": logits.byteswap()
        (source / "logits.f32").write_bytes(logits.tobytes())
        attention = {"storage": "ssd", "sparse": True, "layout": "separate", "fuse_kv_projection": False,
                     "block_tokens": 2, "context_budget_tokens": 4, "init_tokens": 1, "local_tokens": 1,
                     "representative_count": 1, "selection_source": "dynamic"}
        for key in ("read_requests", "read_bytes", "write_requests", "write_bytes", "io_read_ms", "io_write_ms",
                    "append_ms", "selection_ms", "gather_inclusive_ms", "gather_copy_bytes", "streaming_attention_inclusive_ms",
                    "projection_extra_weight_bytes", "projection_initialization_ms"):
            attention[key] = 0
        result = {"attention": attention, "model_parts_sha256": ["a" * 64], "prompt_token_ids": [7],
                  "generated_token_ids": [1, 2], "decode_input_token_ids": [8],
                  "timing": {"decode_loop_ms": 0, "decode_steps": 1},
                  "selection_trace_sha256": ablation.digest(source / "selection.jsonl"),
                  "logits": {"file": "logits.f32", "dtype": "float32-le", "rows": 2, "columns": 2,
                             "sha256": ablation.digest(source / "logits.f32")}}
        (source / "result.json").write_text(json.dumps(result))
        return source, result

    def test_preparation_locks_inputs_and_produces_four_independent_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            manifest_path = ablation.create(source / "result.json", root / "configs", root / "runs", True)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["variants"]), 4)
            self.assertFalse((root / "runs").exists())
            modes = set()
            for variant in manifest["variants"]:
                config = json.loads(Path(variant["config"]).read_text())
                self.assertEqual(config["replay_token_ids"], [8])
                self.assertEqual(config["context_tokens"], 32)
                self.assertEqual(config["attention"]["context_budget_tokens"], 4)
                self.assertEqual(config["attention"]["selection_replay"], str(source / "selection.jsonl"))
                self.assertTrue(config["attention"]["profile"])
                modes.add((config["attention"]["layout"], config["attention"]["fuse_kv_projection"]))
            self.assertEqual(len(modes), 4)

    def test_comparison_checks_selection_logits_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, reference = self.fixture(root)
            manifest_path = ablation.create(source / "result.json", root / "configs", root / "runs")
            manifest = json.loads(manifest_path.read_text())
            for variant in manifest["variants"]:
                path = Path(variant["result"])
                path.parent.mkdir(parents=True)
                config_file = path.parent / "config.json"
                config_file.write_bytes(Path(variant["config"]).read_bytes())
                for name in ("selection.jsonl", "logits.f32"):
                    (path.parent / name).write_bytes((source / name).read_bytes())
                candidate = copy.deepcopy(reference)
                candidate["attention"].update({"layout": variant["layout"], "fuse_kv_projection": variant["fused"], "selection_source": "replay"})
                candidate["selection_replay_sha256"] = manifest["selection_sha256"]
                path.write_text(json.dumps(candidate))
                run = {"status": "complete", "returncode": 0, "code_commit": "code", "runner_sha256": "runner",
                       "source_config_sha256": variant["config_sha256"], "effective_config_sha256": ablation.digest(config_file),
                       "result_sha256": ablation.digest(path)}
                (path.parent / "manifest.json").write_text(json.dumps(run))
            self.assertTrue(ablation.report(manifest_path)["passed"])
            path = Path(manifest["variants"][0]["result"])
            candidate = json.loads(path.read_text())
            changed = path.parent / "selection.jsonl"
            changed.write_text(json.dumps({"layer": 0, "position": 3, "kv_head": 0, "ranges": [[0, 2], [3, 1]]}) + "\n")
            candidate["selection_trace_sha256"] = ablation.digest(changed)
            path.write_text(json.dumps(candidate))
            run_path = path.parent / "manifest.json"
            run = json.loads(run_path.read_text())
            run["result_sha256"] = ablation.digest(path)
            run_path.write_text(json.dumps(run))
            report = ablation.report(manifest_path)
            self.assertFalse(report["passed"])
            self.assertEqual(report["variants"][0]["first_selection_difference_line"], 1)
            config_file = path.parent / "config.json"
            config_file.write_text("changed")
            with self.assertRaises(ValueError): ablation.report(manifest_path)

    def test_changed_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            manifest_path = ablation.create(source / "result.json", root / "configs", root / "runs")
            (source / "selection.jsonl").write_text("changed")
            with self.assertRaises(ValueError): ablation.report(manifest_path)

    def test_missing_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            manifest_path = ablation.create(source / "result.json", root / "configs", root / "runs")
            manifest = json.loads(manifest_path.read_text())
            manifest["variants"].pop()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): ablation.report(manifest_path)


if __name__ == "__main__":
    unittest.main()
