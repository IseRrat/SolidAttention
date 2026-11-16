import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adapter_generator", ROOT / "scripts/generate_adapter.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class GeneratorTests(unittest.TestCase):
    def test_overlay_preserves_upstream_and_is_repeatable(self):
        source = ROOT / "third_party/llama.cpp/src"
        before = {name: (source / name).read_bytes() for name in ("llama-graph.cpp", "llama-context.cpp")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generator.generate(source, output)
            first = {name: (output / name).read_bytes() for name in before}
            generator.generate(source, output)
            self.assertEqual(first, {name: (output / name).read_bytes() for name in before})
            self.assertIn(b"solid::adapter_attention", first["llama-graph.cpp"])
            self.assertIn(b"model.no_alloc() || solid::adapter_active", first["llama-context.cpp"])
        self.assertEqual(before, {name: (source / name).read_bytes() for name in before})

    def test_changed_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            generator.replace_once("different upstream text", "required anchor", "replacement")


if __name__ == "__main__":
    unittest.main()
