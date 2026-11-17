"""Cross-check real synthetic GGML traces against the offline auditors."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_prefetch import audit_residency
from schedule_ablation import audit_schedule, MODES, NAMES

with tempfile.TemporaryDirectory(prefix="solid-schedule-audit-") as temporary:
    for name, mode in zip(NAMES, MODES):
        directory = Path(temporary) / name
        directory.mkdir()
        subprocess.run([sys.argv[1], mode, str(directory)], check=True, timeout=45)
        for folder in directory.iterdir():
            observed = audit_schedule(folder / "schedule.jsonl", mode)
            expected = json.loads((folder / "stats.json").read_text())
            if observed != expected:
                raise ValueError(f"schedule trace differs from C++ metrics: {name}/{folder.name}: {observed} != {expected}")
            audit_residency(folder / "residency.jsonl", 2, 6)
print("all three modes: native GGML schedule/residency traces and metrics audited")
