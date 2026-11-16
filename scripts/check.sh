#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
case "${1:---unit-only}" in
    --unit-only)
        cmake --preset checks
        cmake --build --preset checks
        ctest --preset checks
        ;;
    --cuda)
        export PATH="/usr/local/cuda/bin:$PATH"
        cmake --preset cuda
        cmake --build --preset cuda
        ctest --preset cuda
        python3 scripts/run_baseline.py --check-build
        ;;
    *)
        echo "Usage: scripts/check.sh [--unit-only|--cuda]" >&2
        exit 2
        ;;
esac
python3 -m unittest discover -s tests -p 'test_*.py' -v
