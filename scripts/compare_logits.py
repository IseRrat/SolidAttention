#!/usr/bin/env python3
# 比较两次运行的输入、模型身份与 logits，报告数值误差和首次分歧位置。
"""Compare fixed-input logits from two recorded runs without loading a model."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_result(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    metadata = value["logits"]
    if metadata["dtype"] != "float32-le":
        raise ValueError("unsupported logits dtype")
    for key in ("rows", "columns"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError("logits dimensions must be positive integers")
    binary = path.parent / metadata["file"]
    if binary.stat().st_size != metadata["rows"] * metadata["columns"] * 4:
        raise ValueError("logits byte count differs from metadata")
    if file_sha256(binary) != metadata["sha256"]:
        raise ValueError("logits checksum mismatch")
    if metadata["rows"] != len(value["decode_input_token_ids"]) + 1:
        raise ValueError("logits rows differ from fixed input count")
    return value, binary


# 先核对两次运行身份和输入，再逐项比较 logits 并定位首个超差位置。
def compare(reference_path, candidate_path, atol=1e-3, rtol=1e-4):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    reference, reference_binary = load_result(Path(reference_path))
    candidate, candidate_binary = load_result(Path(candidate_path))
    for key in ("model_parts_sha256", "prompt_token_ids", "decode_input_token_ids"):
        if reference[key] != candidate[key]:
            raise ValueError(f"runs differ in {key}")
    rows, columns = reference["logits"]["rows"], reference["logits"]["columns"]
    if (rows, columns) != (candidate["logits"]["rows"], candidate["logits"]["columns"]):
        raise ValueError("runs have different logits dimensions")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    violations = 0
    first_violation = None
    argmax_differences = []
    with reference_binary.open("rb") as left, candidate_binary.open("rb") as right:
        for row in range(rows):
            a, b = array("f"), array("f")
            a.frombytes(left.read(columns * 4))
            b.frombytes(right.read(columns * 4))
            if len(a) != columns or len(b) != columns:
                raise ValueError("short logits read")
            if sys.byteorder != "little":
                a.byteswap()
                b.byteswap()
            best_a, best_b = 0, 0
            for column, (x, y) in enumerate(zip(a, b)):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"non-finite logits at row {row}, column {column}")
                if x > a[best_a]: best_a = column
                if y > b[best_b]: best_b = column
                difference = abs(x - y)
                maximum_absolute = max(maximum_absolute, difference)
                maximum_relative = max(maximum_relative, difference / max(abs(x), abs(y), 1e-6))
                if difference > atol + rtol * abs(x):
                    violations += 1
                    if first_violation is None:
                        first_violation = {"row": row, "column": column, "reference": x, "candidate": y}
            if best_a != best_b:
                argmax_differences.append({"row": row, "reference": best_a, "candidate": best_b})
    return {"passed": violations == 0, "rows": rows, "columns": columns, "atol": atol, "rtol": rtol,
            "max_absolute_error": maximum_absolute, "max_relative_error": maximum_relative,
            "violations": violations, "first_violation": first_violation, "argmax_differences": argmax_differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.atol, args.rtol)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"compare_logits: {error}", file=sys.stderr)
        sys.exit(2)
