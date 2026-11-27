# logits 对照、误差容限与首次分歧报告检查。
import importlib.util
from array import array
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).resolve().parents[1] / "scripts/compare_logits.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def write_result(self, root, name, values, inputs=None):
        data = array("f", values)
        if sys.byteorder != "little": data.byteswap()
        binary = root / (name + ".f32")
        binary.write_bytes(data.tobytes())
        result = {"model_parts_sha256": ["a" * 64], "prompt_token_ids": [1],
                  "decode_input_token_ids": [2] if inputs is None else inputs,
                  "logits": {"file": binary.name, "dtype": "float32-le", "rows": 2, "columns": 3,
                             "sha256": comparison.file_sha256(binary)}}
        path = root / (name + ".json")
        path.write_text(json.dumps(result))
        return path

    def test_tolerance_and_first_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.write_result(root, "reference", [1, 2, 3, 4, 5, 6])
            near = self.write_result(root, "near", [1, 2, 3.00001, 4, 5, 6])
            self.assertTrue(comparison.compare(reference, near)["passed"])
            changed = self.write_result(root, "changed", [1, 2, 1, 4, 5, 6])
            result = comparison.compare(reference, changed)
            self.assertFalse(result["passed"])
            self.assertEqual(result["first_violation"]["column"], 2)
            self.assertEqual(result["argmax_differences"][0]["candidate"], 1)

    def test_inputs_checksums_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.write_result(root, "reference", [1, 2, 3, 4, 5, 6])
            mismatch = self.write_result(root, "mismatch", [1, 2, 3, 4, 5, 6], [3])
            with self.assertRaisesRegex(ValueError, "decode_input_token_ids"):
                comparison.compare(reference, mismatch)
            nan = self.write_result(root, "nan", [1, 2, float("nan"), 4, 5, 6])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                comparison.compare(reference, nan)
            (root / "nan.f32").write_bytes(b"x" * 24)
            with self.assertRaisesRegex(ValueError, "checksum"):
                comparison.compare(reference, nan)


if __name__ == "__main__":
    unittest.main()
