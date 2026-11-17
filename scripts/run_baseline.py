#!/usr/bin/env python3
"""Run the pinned baseline and save its inputs, provenance and process memory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from process_lifetime import stop_process, termination_signals

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "ef6bc6f367082338006c8379bc7bcab3e1584766"


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during hashing: {path}")
    return value.hexdigest()


def validate_model_files(config, base):
    model = Path(config["model"])
    declared = {model: config["model_sha256"]}
    for part in config.get("additional_model_files", []):
        path = (base / part["path"]).resolve()
        if path in declared:
            raise ValueError("duplicate model file")
        declared[path] = part["sha256"]
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    if match:
        prefix, index, count = match.groups()
        count = int(count)
        if int(index) != 1 or count < 1:
            raise ValueError("model must name the first GGUF shard")
        expected = {model.parent / f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)}
        if set(declared) != expected:
            raise ValueError("all GGUF shards must have declared hashes")
    actual = []
    for path, expected_sha in declared.items():
        value = digest(path)
        if value != expected_sha:
            raise ValueError(f"GGUF SHA256 differs for {path.name}")
        actual.append({"path": str(path), "sha256": value, "bytes": path.stat().st_size})
    return actual


def command_output(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def verify_build_info(info, root=ROOT):
    source_files = {"CMakeLists.txt", "scripts/generate_adapter.py"}
    for folder, pattern in (("src", "*.cpp"), ("include", "*.hpp"), ("cmake", "*")):
        source_files.update(path.relative_to(root).as_posix() for path in (root / folder).rglob(pattern) if path.is_file())
    if info["upstream_commit"] != UPSTREAM or set(info["source_files"]) != source_files:
        raise ValueError("runner build inputs differ from the current project; rebuild it")
    for name, expected in info["source_files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"runner was built from different source: {name}")


def verify_features(config, info):
    if config.get("attention", {}).get("prefetch") and not info.get("features", {}).get("uring"):
        raise ValueError("prefetch requires a runner built with SOLID_URING=ON")


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def process_memory(pid):
    values = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, text = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmLck", "VmSwap"}:
                values[key + "_bytes"] = int(text.split()[0]) * 1024
    except FileNotFoundError:
        pass
    return values


def process_io(pid):
    try:
        return {key: int(value) for key,value in (line.split(":",1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())}
    except (FileNotFoundError, PermissionError):
        return {}


def cgroup_memory(pid=None, group=None):
    try:
        if group is None:
            relative = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                            if line.startswith("0::"))
            group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        group = Path(group)
        values = {"path": str(group)}
        for name in ("memory.max", "memory.peak", "memory.current", "memory.swap.max", "memory.stat", "memory.events", "io.stat", "cpuset.cpus.effective"):
            path = group / name
            if path.exists():
                values[name] = path.read_text().strip()
        return values
    except (FileNotFoundError, StopIteration, PermissionError):
        return {}


def run_monitored(manifest, run_dir, output):
    """Own the runner from creation through monitoring, cleanup and recording."""
    start = time.monotonic()
    process = None
    primary = None
    peak_rss = 0
    rss_samples = 0
    with termination_signals(), (run_dir / "stdout.txt").open("wb") as stdout, \
            (run_dir / "stderr.txt").open("wb") as stderr, (run_dir / "memory.jsonl").open("w",encoding="utf-8") as memory:
        try:
            process = subprocess.Popen(manifest["command"],stdout=stdout,stderr=stderr,start_new_session=True)
            manifest["status"] = "running"
            manifest["cgroup_start"] = cgroup_memory(process.pid)
            manifest["proc_io_start"] = process_io(process.pid)
            atomic_json(run_dir / "manifest.json",manifest)
            while process.poll() is None:
                sample = process_memory(process.pid)
                peak_rss = max(peak_rss,sample.get("VmHWM_bytes",sample.get("VmRSS_bytes",0)))
                if "VmHWM_bytes" in sample or "VmRSS_bytes" in sample: rss_samples += 1
                memory.write(json.dumps({"seconds":time.monotonic()-start,**sample,"proc_io":process_io(process.pid)})+"\n")
                manifest["cgroup_last"] = cgroup_memory(process.pid)
                time.sleep(.1)
        except BaseException as error:
            primary = error
            if process is not None:
                try: stop_process(process,group=True,timeout=5)
                except BaseException as cleanup_error: manifest["cleanup_error"] = str(cleanup_error)
        manifest["returncode"] = process.returncode if process is not None else None
        manifest["elapsed_seconds"] = time.monotonic()-start
        manifest["sampled_peak_rss_bytes"] = peak_rss if rss_samples else None
        manifest["rss_samples"] = rss_samples
        manifest["status"] = ("interrupted" if isinstance(primary,KeyboardInterrupt) else "failed") if primary else (
            "complete" if process.returncode == 0 and output.is_file() else "failed")
        try:
            group = manifest.get("cgroup_start",{}).get("path")
            manifest["cgroup_end"] = cgroup_memory(group=group) if group else {}
            if manifest["status"] == "complete": manifest["result_sha256"] = digest(output)
            stdout.flush(); stderr.flush(); memory.flush()
            manifest["artifacts_sha256"] = {name:digest(run_dir/name) for name in ("stdout.txt","stderr.txt","memory.jsonl")}
        except BaseException as error:
            if primary is None: primary = error; manifest["status"] = "failed"
            else: manifest["recording_error"] = str(error)
        if primary is not None: manifest["error"] = str(primary)
        try: atomic_json(run_dir / "manifest.json",manifest)
        except BaseException:
            if primary is None: raise
        if primary is not None: raise primary
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/release/bin/solid-baseline")
    parser.add_argument("--prepare-only", action="store_true", help="verify files and write inputs without inference")
    parser.add_argument("--check-build", action="store_true", help="verify runner build inputs without a model")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("run this launcher in the Linux experiment environment")
    runner = args.runner.resolve(strict=True)
    build_info = json.loads(command_output([str(runner), "--build-info"]))
    verify_build_info(build_info)
    if args.check_build:
        print("runner source fingerprint verified")
        return 0
    if args.config is None:
        parser.error("a configuration file is required")
    config_path = args.config.resolve(strict=True)
    subprocess.run([str(runner), "--validate-config", str(config_path)], check=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_features(config, build_info)
    for key in ("model", "prompt", "output"):
        config[key] = str((config_path.parent / config[key]).resolve())
    if "session_file" in config.get("attention", {}):
        config["attention"]["session_file"] = str((config_path.parent / config["attention"]["session_file"]).resolve())
    replay_path = config.get("attention", {}).get("selection_replay")
    replay_hash = None
    if replay_path:
        replay_path = (config_path.parent / replay_path).resolve(strict=True)
        replay_hash = digest(replay_path)
    model, prompt, output = (Path(config[key]) for key in ("model", "prompt", "output"))
    model_files = validate_model_files(config, config_path.parent)
    for part in config.get("additional_model_files", []):
        part["path"] = str((config_path.parent / part["path"]).resolve())
    upstream = command_output(["git", "rev-parse", "HEAD"], ROOT / "third_party/llama.cpp")
    if upstream != UPSTREAM:
        raise ValueError("upstream commit differs from the pinned version")
    if command_output(["git", "status", "--porcelain"], ROOT / "third_party/llama.cpp"):
        raise ValueError("upstream contains local changes; record an adapter commit before running")
    if command_output(["git", "status", "--porcelain"]):
        raise ValueError("commit the project source before recording a baseline")
    run_dir = output.parent
    run_dir.mkdir(parents=True, exist_ok=False)
    prompt_copy = run_dir / "prompt.txt"
    prompt_copy.write_bytes(prompt.read_bytes())
    config["prompt"] = str(prompt_copy)
    if replay_path:
        replay_copy = run_dir / "selection.replay.jsonl"
        shutil.copyfile(replay_path, replay_copy)
        if digest(replay_copy) != replay_hash:
            raise ValueError("selection replay changed while preparing the run")
        config["attention"]["selection_replay"] = str(replay_copy)
    effective_config = run_dir / "config.json"
    atomic_json(effective_config, config)
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "code_commit": command_output(["git", "rev-parse", "HEAD"]),
        "code_tree_clean": True,
        "upstream_commit": upstream,
        "runner_sha256": digest(runner),
        "build_info": build_info,
        "model_files": model_files,
        "prompt_sha256": digest(prompt_copy),
        "source_config_sha256": digest(config_path),
        "effective_config_sha256": digest(effective_config),
        "selection_replay_sha256": replay_hash,
        "command": [str(runner), "--config", str(effective_config)],
        "system": {"uname": list(platform.uname()), "python": sys.version,
                   "os_release": Path("/etc/os-release").read_text()},
        "environment": {key: os.environ.get(key) for key in
                        ("CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS", "OMP_NUM_THREADS")},
        "memory_sample_interval_seconds": 0.1,
        "launcher_sha256": digest(Path(__file__)),
    }
    try:
        manifest["gpu"] = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest["gpu"] = None
    atomic_json(run_dir / "manifest.json", manifest)
    if args.prepare_only:
        print(run_dir)
        return 0
    run_monitored(manifest,run_dir,output)
    print(run_dir)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_baseline: {error}", file=sys.stderr)
        sys.exit(1)
