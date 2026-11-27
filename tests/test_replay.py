# 固定 Decode 输入准备和参考运行上下文保留检查。
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).resolve().parents[1] / "scripts/prepare_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_reference_inputs_and_context_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.json").write_text(json.dumps({"generated_token_ids": [1, 2, 3], "decode_input_token_ids": [8, 9]}))
            (root / "prompt.txt").write_text("reference prompt")
            (root / "config.json").write_text(json.dumps({"model": "/models/reference.gguf", "context_tokens": 131072,
                                                           "prompt": "/old/prompt", "attention": {"storage": "memory"}}))
            (root / "mode.json").write_text(json.dumps({"model": "different.gguf", "context_tokens": 4096,
                                                        "attention": {"storage": "ssd", "sparse": False}}))
            result = replay.prepare(root / "result.json", root / "mode.json", root / "other/result.json")
            self.assertEqual(result["model"], "/models/reference.gguf")
            self.assertEqual(result["context_tokens"], 131072)
            self.assertEqual(result["prompt"], str((root / "prompt.txt").resolve()))
            self.assertEqual(result["replay_token_ids"], [8, 9])
            self.assertEqual(result["max_new_tokens"], 3)
            self.assertTrue(result["dump_logits"])
            self.assertEqual(result["attention"]["storage"], "ssd")


if __name__ == "__main__":
    unittest.main()
