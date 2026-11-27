# 读取调度记录，检查任务依赖、完成顺序与同步证明。
"""Cross-check real synthetic GGML traces against the offline auditors."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_prefetch import audit_residency
from schedule_ablation import audit_schedule, audit_block_schedule, MODES, NAMES

with tempfile.TemporaryDirectory(prefix="solid-schedule-audit-") as temporary:
    for name, mode in list(zip(NAMES,MODES))+[("native-events","overlap+sync_reuse"),("device-input","overlap+sync_reuse"),("device-serial","naive"),("shared-buffer","overlap+sync_reuse")]:
        directory = Path(temporary) / name
        directory.mkdir()
        flags=["--native-events"] if name=="native-events" else []
        if name=="device-input": flags=["--direct","--packed","--native-events"]
        if name=="device-serial": flags=["--direct","--packed"]
        if name=="shared-buffer": flags=["--shared","--direct","--packed","--native-events"]
        subprocess.run([sys.argv[1], mode,*flags,str(directory)], check=True, timeout=45)
        for folder in directory.iterdir():
            schedule=folder / "schedule.jsonl"
            observed = audit_schedule(schedule, mode)
            expected = json.loads((folder / "stats.json").read_text())
            if observed != expected:
                raise ValueError(f"schedule trace differs from C++ metrics: {name}/{folder.name}: {observed} != {expected}")
            archives=sum(json.loads(line).get("fence")=="kv_archive_copy_complete" for line in schedule.read_text().splitlines())
            if archives != (observed["frames"] if name in ("device-input","shared-buffer") else 0):
                raise ValueError(f"unexpected deferred KV archives: {name}/{folder.name}")
            residency=audit_residency(folder / "residency.jsonl", 2, 7 if name=="shared-buffer" else 6)
            for key,value in json.loads((folder/"device-stats.json").read_text()).items():
                if residency.get(key,0)!=value: raise ValueError(f"device row transfer metrics differ: {name}/{folder.name}/{key}")
            blocks=audit_block_schedule(folder/"schedule.blocks.jsonl",mode,folder/"schedule.jsonl")
            if blocks != json.loads((folder/"block-stats.json").read_text()):
                raise ValueError(f"block schedule trace differs from C++ metrics: {name}/{folder.name}")
print("three modes and completion protocol: CPU GGML traces and metrics audited")
