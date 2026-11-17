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
    --sanitize)
        cmake -S . -B build/sanitizers -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSOLID_BUILD_RUNNER=OFF \
            -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
            -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
        cmake --build build/sanitizers --parallel 4
        ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build/sanitizers --output-on-failure
        ;;
    *)
        echo "Usage: scripts/check.sh [--unit-only|--cuda|--sanitize]" >&2
        exit 2
        ;;
esac
python3 -m unittest discover -s tests -p 'test_*.py' -v
