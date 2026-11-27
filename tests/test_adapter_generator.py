# llama.cpp 适配覆盖生成、重复执行与上游锚点检查。
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
        cpu=source.parent/"ggml/src/ggml-cpu/ggml-cpu.c"
        cpu_before=cpu.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generator.generate(source, output)
            first = {name: (output / name).read_bytes() for name in before}
            first_backend = (output / "ggml-backend.cpp").read_bytes()
            first_cuda=(output/"ggml-cuda.cu").read_bytes()
            first_cpu=(output/"ggml-cpu.c").read_bytes()
            generator.generate(source, output)
            self.assertEqual(first, {name: (output / name).read_bytes() for name in before})
            self.assertEqual(first_backend, (output / "ggml-backend.cpp").read_bytes())
            self.assertEqual(first_cuda,(output/"ggml-cuda.cu").read_bytes())
            self.assertEqual(first_cpu,(output/"ggml-cpu.c").read_bytes())
            self.assertEqual(first_cpu.count(b"solid_trace_name_cpu_worker(state->ith)"),1)
            self.assertEqual(first_cuda.count(b"CUDA_CHECK(solid::cuda_attention_begin(completion,"),4)
            self.assertEqual(first_cuda.count(b"CUDA_CHECK(solid::cuda_attention_end(completion,"),4)
            self.assertEqual(first_cuda.count(b"case GGML_OP_CUSTOM:"),2)
            self.assertEqual(first_cuda.count(b"CUDA_CHECK(solid::cuda_device_op(dst, ctx.stream()))"),1)
            self.assertIn(b"return solid::device_op_supported(op);",first_cuda)
            self.assertIn(b"if (!solid::device_op_supported(dst)) return false;",first_cuda)
            self.assertLess(first_cuda.index(b"CUDA_CHECK(solid::cuda_attention_begin("),
                            first_cuda.index(b"CUDA_CHECK(solid::cuda_attention_end("))
            for operation, launch in (
                (b"FLASH_ATTN_EXT", b"ggml_cuda_flash_attn_ext(ctx, dst);"),
                (b"MUL_MAT", b"ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);"),
                (b"CONT", b"ggml_cuda_dup(ctx, dst);"),
                (b"SOFT_MAX", b"ggml_cuda_op_soft_max(ctx, dst);"),
            ):
                start = first_cuda.index(b"case GGML_OP_" + operation + b": {")
                end = first_cuda.index(b"\n        }", start)
                boundary = first_cuda[start:end]
                self.assertLess(boundary.index(b"cuda_attention_begin("), boundary.index(launch))
                self.assertLess(boundary.index(launch), boundary.index(b"cuda_attention_end("))
                self.assertIn(b"phase != solid::CompletionPhase::end", boundary)
                self.assertIn(b"phase != solid::CompletionPhase::begin", boundary)
            self.assertIn(b"solid::adapter_attention", first["llama-graph.cpp"])
            self.assertIn(b"model.no_alloc() || solid::adapter_active", first["llama-context.cpp"])
            self.assertIn(b"solid::adapter_qkv", first["llama-graph.cpp"])
            self.assertNotIn(b"solid_copy_scope", (output / "ggml-backend.cpp").read_bytes())
            self.assertNotIn(b"device_timing", (output / "ggml-cuda.cu").read_bytes())
        self.assertEqual(before, {name: (source / name).read_bytes() for name in before})
        self.assertEqual(backend_before, backend.read_bytes())
        self.assertEqual(cuda_before,cuda.read_bytes())
        self.assertEqual(cpu_before,cpu.read_bytes())

    def test_changed_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            generator.replace_once("different upstream text", "required anchor", "replacement")

if __name__ == "__main__":
    unittest.main()
