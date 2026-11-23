#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    printf 'Usage: %s REPORT_PREFIX COMMAND [ARG ...]\n' "$0" >&2
    exit 2
fi
report=$1
shift
export SOLID_NVTX_TRACE=1
exec nsys profile --trace=cuda-sw,nvtx,osrt --cuda-trace-scope=process-tree \
    --sample=none --cpuctxsw=none --osrt-threshold=50000 --cuda-graph-trace=node \
    --output="$report" "$@"
