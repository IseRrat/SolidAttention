#!/usr/bin/env bash
# 按给定矩阵调用性能采集流程，归档各次运行的日志和结果。
set -euo pipefail
plan=$1
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
reports="$(dirname -- "$plan")/nsys"
mkdir -p "$reports"
python - "$plan" "$reports" <<'PY'
import csv,json,sys
from pathlib import Path
plan=Path(sys.argv[1]); reports=Path(sys.argv[2])
jobs=json.loads(plan.read_text(encoding='utf-8'))['jobs']
ledger=plan.parent/'execution.json'
recorded=json.loads(ledger.read_text(encoding='utf-8'))['jobs'] if ledger.exists() else {}
(reports/'job-order.txt').write_text(''.join(j['id']+'\n' for j in jobs if j['id'] not in recorded))
with (reports/'INDEX.tsv').open('w',newline='') as output:
    writer=csv.writer(output,delimiter='\t')
    writer.writerow(['report','method','input_tokens','model','job','config'])
    for j in jobs:
        g=j['group']
        writer.writerow([j['id']+'.nsys-rep',g['variant'],g['input_tokens'],g['model'],j['id'],j['config']])
PY
while read -r job; do
    printf 'Profiling %s\n' "$job"
    bash "$repo/scripts/solid-nsys.sh" "$reports/$job" \
        python -u "$repo/scripts/experiments.py" run "$plan" \
        --runner "$repo/build/server5090/bin/solid-baseline" --limit 1
    python - "$plan" "$job" <<'PY'
import json,sys
from pathlib import Path
state=json.loads((Path(sys.argv[1]).parent/'execution.json').read_text(encoding='utf-8'))
if state['jobs'].get(sys.argv[2],{}).get('status')!='complete':
    raise SystemExit('profiled report does not match a completed job: '+sys.argv[2])
PY
    test -s "$reports/$job.nsys-rep"
    if ! test -f "$reports/cuda-kernel-check.txt"; then
        nsys stats --report=cuda_gpu_kern_sum "$reports/$job.nsys-rep" > "$reports/cuda-kernel-check.tmp"
        grep -q 'Time (%)' "$reports/cuda-kernel-check.tmp"
        mv "$reports/cuda-kernel-check.tmp" "$reports/cuda-kernel-check.txt"
    fi
done < "$reports/job-order.txt"
