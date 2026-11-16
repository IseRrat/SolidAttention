import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("launcher", Path(__file__).resolve().parents[1] / "scripts/run_baseline.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
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
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "different source"):
                launcher.verify_build_info(info, root)


if __name__ == "__main__":
    unittest.main()
