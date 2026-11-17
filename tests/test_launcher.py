import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import signal
import sys
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import process_lifetime

spec = importlib.util.spec_from_file_location("launcher", Path(__file__).resolve().parents[1] / "scripts/run_baseline.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_failure_recording_running_manifest_stops_the_spawned_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            process=mock.Mock(pid=99999999,returncode=None)
            def wait(**kwargs): process.returncode=-15; return -15
            process.wait.side_effect=wait
            real_write=launcher.atomic_json
            def write(path,value):
                if value["status"]=="running": raise OSError("injected manifest disk failure")
                real_write(path,value)
            with mock.patch.object(launcher.subprocess,"Popen",return_value=process), \
                    mock.patch.object(launcher,"cgroup_memory",return_value={}), \
                    mock.patch.object(launcher,"process_io",return_value={}), \
                    mock.patch.object(launcher,"atomic_json",side_effect=write), \
                    mock.patch.object(process_lifetime.os,"killpg",create=True) as stop:
                with self.assertRaisesRegex(OSError,"manifest disk failure"):
                    launcher.run_monitored({"command":["synthetic"]},root,root/"result.json")
            process.wait.assert_called_once()
            self.assertIn(mock.call(process.pid,signal.SIGTERM),stop.call_args_list)
            state=json.loads((root/"manifest.json").read_text())
            self.assertEqual(state["status"],"failed")
            self.assertEqual(state["returncode"],-15)

    def test_secondary_record_failure_preserves_original_startup_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); process=mock.Mock(pid=99999999,returncode=-15)
            with mock.patch.object(launcher.subprocess,"Popen",return_value=process), \
                    mock.patch.object(launcher,"cgroup_memory",side_effect=RuntimeError("original startup failure")), \
                    mock.patch.object(launcher,"atomic_json",side_effect=OSError("secondary record failure")), \
                    mock.patch.object(process_lifetime.os,"killpg",create=True):
                with self.assertRaisesRegex(RuntimeError,"original startup failure"):
                    launcher.run_monitored({"command":["synthetic"]},root,root/"result.json")
            process.wait.assert_called_once()

    def test_successful_short_runner_records_flushed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); output=root/"result.json"; output.write_text('{}')
            process=mock.Mock(pid=99999999,returncode=0); process.poll.return_value=0
            with mock.patch.object(launcher.subprocess,"Popen",return_value=process), \
                    mock.patch.object(launcher,"cgroup_memory",return_value={}), \
                    mock.patch.object(launcher,"process_io",return_value={}):
                state=launcher.run_monitored({"command":["synthetic"]},root,output)
            self.assertEqual(state["status"],"complete")
            self.assertIsNone(state["sampled_peak_rss_bytes"])
            self.assertEqual(state["rss_samples"],0)
            for name,sha in state["artifacts_sha256"].items(): self.assertEqual(launcher.digest(root/name),sha)

    def test_prefetch_requires_uring_build_feature(self):
        launcher.verify_features({}, {})
        launcher.verify_features({"attention": {"prefetch": True}}, {"features": {"uring": True}})
        with self.assertRaises(ValueError):
            launcher.verify_features({"attention": {"prefetch": True}}, {"features": {"uring": False}})
    def test_sha256_matches_known_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input"
            path.write_bytes(b"abc")
            self.assertEqual(launcher.digest(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_atomic_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            launcher.atomic_json(path, {"value": 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                launcher.atomic_json(path, {"value": float("nan")})
            self.assertEqual(original, path.read_bytes())

    def test_missing_process_is_observable(self):
        self.assertEqual(launcher.process_memory(999999999), {})
        self.assertEqual(launcher.cgroup_memory(999999999), {})

    def test_every_model_shard_is_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            config = {"model": str(first), "model_sha256": launcher.digest(first)}
            with self.assertRaisesRegex(ValueError, "all GGUF shards"):
                launcher.validate_model_files(config, root)
            config["additional_model_files"] = [{"path": second.name, "sha256": launcher.digest(second)}]
            self.assertEqual(len(launcher.validate_model_files(config, root)), 2)
            second.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                launcher.validate_model_files(config, root)

    def test_stale_binary_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "CMakeLists.txt"
            source.write_text("original")
            (root / "scripts").mkdir()
            generator = root / "scripts/generate_adapter.py"
            generator.write_text("generator")
            info = {"upstream_commit": launcher.UPSTREAM, "source_files": {
                "CMakeLists.txt": launcher.digest(source), "scripts/generate_adapter.py": launcher.digest(generator)}}
            launcher.verify_build_info(info, root)
            (root / "src/cuda").mkdir(parents=True)
            kernel = root / "src/cuda/block_ranker.cu"
            kernel.write_text("kernel")
            with self.assertRaisesRegex(ValueError, "build inputs differ"):
                launcher.verify_build_info(info, root)
            info["source_files"]["src/cuda/block_ranker.cu"] = launcher.digest(kernel)
            launcher.verify_build_info(info, root)
            kernel.write_text("changed kernel")
            with self.assertRaisesRegex(ValueError, "different source"):
                launcher.verify_build_info(info, root)
            kernel.write_text("kernel")
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "different source"):
                launcher.verify_build_info(info, root)

    def test_cuda_selection_requires_cuda_build(self):
        config = {"attention": {"selection_backend": "cuda"}}
        with self.assertRaisesRegex(ValueError, "SOLID_CUDA"):
            launcher.verify_features(config, {"features": {"cuda": False}})
        launcher.verify_features(config, {"features": {"cuda": True}})
        launcher.verify_features({"attention": {"selection_backend": "cpu"}}, {"features": {}})
        launcher.verify_features({"attention": {"selection_backend": "auto"}}, {"features": {}})
        with self.assertRaisesRegex(ValueError,"CUDA prefill"):
            launcher.verify_features({"attention":{"prefill_backend":"cuda"}},{"features":{"cuda":False}})
        launcher.verify_features({"attention":{"prefill_backend":"cuda"}},{"features":{"cuda":True}})


if __name__ == "__main__":
    unittest.main()
