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
        backend = source.parent / "ggml/src/ggml-backend.cpp"
        backend_before = backend.read_bytes()
        cuda=source.parent/"ggml/src/ggml-cuda/ggml-cuda.cu"
        cuda_before=cuda.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generator.generate(source, output)
            first = {name: (output / name).read_bytes() for name in before}
            first_backend = (output / "ggml-backend.cpp").read_bytes()
            first_cuda=(output/"ggml-cuda.cu").read_bytes()
            generator.generate(source, output)
            self.assertEqual(first, {name: (output / name).read_bytes() for name in before})
            self.assertEqual(first_backend, (output / "ggml-backend.cpp").read_bytes())
            self.assertEqual(first_cuda,(output/"ggml-cuda.cu").read_bytes())
            self.assertEqual(first_cuda.count(b"CUDA_CHECK(solid::cuda_attention_begin("),1)
            self.assertEqual(first_cuda.count(b"CUDA_CHECK(solid::cuda_attention_end("),1)
            self.assertLess(first_cuda.index(b"CUDA_CHECK(solid::cuda_attention_begin("),
                            first_cuda.index(b"CUDA_CHECK(solid::cuda_attention_end("))
            self.assertIn(b"solid::adapter_attention", first["llama-graph.cpp"])
            self.assertIn(b"model.no_alloc() || solid::adapter_active", first["llama-context.cpp"])
            self.assertIn(b"solid::adapter_qkv", first["llama-graph.cpp"])
            self.assertEqual((output / "ggml-backend.cpp").read_bytes().count(b"solid_copy_scope solid_copy("), 2)
        self.assertEqual(before, {name: (source / name).read_bytes() for name in before})
        self.assertEqual(backend_before, backend.read_bytes())
        self.assertEqual(cuda_before,cuda.read_bytes())

    def test_changed_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            generator.replace_once("different upstream text", "required anchor", "replacement")


if __name__ == "__main__":
    unittest.main()
