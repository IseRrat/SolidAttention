# 批量检查仓库实验配置的模型字段、预算和方法开关。
"""Check every prepared experiment variant with the actual runner config parser."""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_experiments as fixtures
import experiment_common as common
import experiments

with tempfile.TemporaryDirectory(prefix="solid-matrix-config-") as temporary:
    root=Path(temporary)
    spec,_=fixtures.fixture(root)
    value=common.load(spec)
    value.update(warmups=0,repeats=1)
    value["suites"][0].update(variants=sorted(experiments.VARIANTS),budgets=[4])
    fixtures.write(spec,value)
    plan=common.load(experiments.create(spec,root/"plan"))
    for row in plan["jobs"]:
        result=subprocess.run([sys.argv[1],"--validate-config",row["config"]],capture_output=True,text=True,timeout=15)
        if result.returncode: raise ValueError(f"{row['group']['variant']}: {result.stderr}")
print("all ten generated experiment variants validated; no model loaded")
