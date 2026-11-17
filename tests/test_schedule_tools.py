import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import schedule_ablation as scheduling
from compare_prefetch import audit_residency, digest
import test_prefetch_tools as fixtures
from test_prefetch_tools import residency, write


def trace(mode):
    serial = mode == "naive"
    names = ["q_gpu_ready", "q_d2h_ready", "selection_ready" if serial else "selection", "kv_projection_to_host",
             "kv_d2h_ready", "miss_read_and_reconcile", "history_h2d", "append_and_selection" if serial else "append_current_kv",
             "current_h2d", "attention_until_boundary", "buffered_ssd_store", "next_layer_prefetch"]
    deps = [[], [0], [7, 1] if serial else [1], [] if serial else [2], [3], [2], [5] if serial else [2],
            [4] if serial else [4, 5, 6], [7, 6] if serial else [7], [8], [7], [8]]
    rows = [dict(event="define", frame=1, task=i, layer=0, position=3, mode=mode, name=name, resource="host",
                 predecessors=deps[i], estimate_ns=1, origin_ns=0, clock="steady_clock_ns") for i, name in enumerate(names)]
    # Use a topological serial trace for deterministic audit arithmetic.
    order = [0, 1, 3, 4, 7, 2, 5, 6, 8, 10, 11, 9] if serial else [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 9]
    for time, task in enumerate(order):
        rows.append(dict(event="start", frame=1, task=task, timestamp_ns=2*time, fence=""))
        proof = ("query_dependency_reuse" if mode == "overlap+sync_reuse" else "attention_backend_complete") if task == 9 else ""
        rows.append(dict(event="complete", frame=1, task=task, timestamp_ns=2*time+1, fence=proof))
    return rows


def jsonlines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ScheduleToolsTests(unittest.TestCase):
    def fixture(self, root):
        helper = fixtures.PrefetchToolsTests()
        source, candidate = helper.fixture(root)
        manifest_path = scheduling.create(source / "result.json", root / "configs", root / "runs")
        manifest = json.loads(manifest_path.read_text())
        for variant in manifest["variants"]:
            folder = Path(variant["result"]).parent
            folder.mkdir(parents=True)
            for name in ("logits.f32", "selection.jsonl", "prefetch.jsonl", "residency.jsonl", "prompt.txt"):
                (folder / name).write_bytes((candidate / name).read_bytes())
            (folder / "selection.replay.jsonl").write_bytes((source / "selection.jsonl").read_bytes())
            config = json.loads(Path(variant["config"]).read_text())
            config["prompt"] = str(folder / "prompt.txt")
            config["attention"]["selection_replay"] = str(folder / "selection.replay.jsonl")
            write(folder / "config.json", config)
            value = json.loads((candidate / "result.json").read_text())
            value["attention"].update(schedule=variant["mode"], selection_source="replay")
            value["selection_replay_sha256"] = manifest["selection_sha256"]
            jsonlines(folder / "schedule.jsonl", trace(variant["mode"]))
            audit = scheduling.audit_schedule(folder / "schedule.jsonl", variant["mode"])
            metrics = {({"critical_path_ns": "observed_critical_path_sum_ms", "miss_projection_overlap_ns": "miss_projection_overlap_sum_ms"}.get(key, key)):
                       value / 1e6 if key.endswith("_ns") else value for key, value in audit.items()}
            value["schedule"] = dict(metrics, mode=variant["mode"], trace_enabled=True)
            value["schedule_trace_sha256"] = digest(folder / "schedule.jsonl")
            write(folder / "result.json", value)
            helper.manifest(folder)
            run = json.loads((folder / "manifest.json").read_text())
            run["source_config_sha256"] = variant["config_sha256"]
            write(folder / "manifest.json", run)
        return manifest_path, manifest

    def test_preparation_and_full_report_preserve_fixed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = self.fixture(Path(temporary))
            for variant in manifest["variants"]:
                config = json.loads(Path(variant["config"]).read_text())
                self.assertEqual(config["replay_token_ids"], [8])
                self.assertEqual(config["attention"]["schedule"], variant["mode"])
                self.assertTrue(config["attention"]["schedule_trace"])
            self.assertTrue(scheduling.report(path)["passed"])
            reference = Path(manifest["reference"])
            result = json.loads(reference.read_text())
            result["decode_input_token_ids"] = []
            write(reference, result)
            with self.assertRaisesRegex(ValueError, "one decode step"):
                scheduling.create(reference, Path(temporary) / "empty-configs", Path(temporary) / "empty-runs")
            self.assertFalse((Path(temporary) / "empty-configs").exists())

    def test_report_rejects_metric_and_provenance_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = self.fixture(Path(temporary))
            folder = Path(manifest["variants"][0]["result"]).parent
            result = json.loads((folder / "result.json").read_text())
            result["schedule"]["tasks"] += 1
            write(folder / "result.json", result)
            # Preserve valid result hashing while changing the recorded metric.
            run = json.loads((folder / "manifest.json").read_text())
            run["result_sha256"] = digest(folder / "result.json")
            write(folder / "manifest.json", run)
            with self.assertRaisesRegex(ValueError, "metric differs"): scheduling.report(path)
            manifest["variants"].pop()
            write(path, manifest)
            with self.assertRaisesRegex(ValueError, "three independent"): scheduling.report(path)

    def test_trace_rejects_dependency_violation_failed_and_incomplete_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schedule.jsonl"
            good = trace("overlap")
            self.assertEqual(len(good), 36)
            wrong = copy.deepcopy(good)
            wrong[14]["task"] = 9  # attention starts before current H2D.
            for rows in (wrong, good[:-1], good + [dict(event="failed", frame=1, task=0, timestamp_ns=99)]):
                jsonlines(path, rows)
                with self.assertRaises(ValueError): scheduling.audit_schedule(path, "overlap")
            jsonlines(path, good)
            self.assertEqual(scheduling.audit_schedule(path, "overlap")["tasks"], 12)

    def test_streamed_h2d_can_start_before_the_read_batch_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schedule.jsonl"
            rows = trace("overlap")
            for row in rows:
                if row["event"] == "start" and row["task"] == 6:
                    row["timestamp_ns"] = 10
            jsonlines(path, rows)
            self.assertEqual(scheduling.audit_schedule(path, "overlap")["frames"], 1)
            wrong = copy.deepcopy(rows)
            wrong[7]["predecessors"] = [4, 6]
            jsonlines(path, wrong)
            with self.assertRaises(ValueError): scheduling.audit_schedule(path, "overlap")

    def test_deferred_pin_keeps_copy_and_attention_lifetimes_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "residency.jsonl"
            rows = residency()
            waiting = dict(rows[2], event="attention_pin_waiting")
            jsonlines(path, [rows[0], waiting, rows[1], rows[3]])
            self.assertEqual(audit_residency(path, 2, 2)["attention_pins"], 1)
            for bad in ([rows[0], waiting, rows[3], rows[1]], [rows[0], waiting],
                        [rows[0], waiting, rows[1], dict(rows[3], epoch=2)]):
                jsonlines(path, bad)
                with self.assertRaises(ValueError): audit_residency(path, 2, 2)


if __name__ == "__main__": unittest.main()
