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
    --sanitize-adapter)
        cmake -S . -B build/sanitizers-adapter -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSOLID_BUILD_RUNNER=ON -DSOLID_CUDA=OFF \
            -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
            -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
        cmake --build build/sanitizers-adapter --parallel 4 --target solid-current-device-test solid-prefetch-failure-test \
            solid-prefetch-graph-test solid-native-completion-test
        ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build/sanitizers-adapter --output-on-failure \
            -R '^(current-device-kv|prefetch-failure-drain|native-completion-progress|device-input.*graph)$'
        ;;
    *)
        echo "Usage: scripts/check.sh [--unit-only|--cuda|--sanitize|--sanitize-adapter]" >&2
        exit 2
        ;;
esac
python3 -m unittest discover -s tests -p 'test_*.py' -v
