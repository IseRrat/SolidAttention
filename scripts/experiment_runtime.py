"""External-environment process isolation and read-only fio interference."""
import os
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid

from experiment_common import number


def condition(value):
    allowed = {"name", "session_root", "cache_state", "memory_max_bytes", "swap_max_bytes", "cpus", "background"}
    if set(value)-allowed: raise ValueError("unknown experiment condition field")
    if value["cache_state"] not in ("os-managed", "warm", "externally-prepared-cold"):
        raise ValueError("cache_state must describe the externally controlled page cache")
    for key in ("memory_max_bytes", "swap_max_bytes"):
        if value.get(key) is not None: number(value[key], integer=True, minimum=1 if key == "memory_max_bytes" else 0)
    cpus = value.get("cpus", [])
    if not isinstance(cpus, list) or len(cpus) != len(set(cpus)): raise ValueError("CPU affinity must contain unique CPU IDs")
    for cpu in cpus: number(cpu, integer=True)
    background = value.get("background")
    if background:
        if set(background) != {"kind", "file", "rate", "iodepth"}: raise ValueError("invalid fio condition fields")
        if background["kind"] not in ("sequential-read", "random-read"): raise ValueError("unsupported fio read pattern")
        number(background["rate"], minimum=1, integer=True)
        number(background["iodepth"], minimum=1, integer=True)
    return value


def isolated_command(command, settings):
    condition(settings)
    if settings.get("cpus"): command = ["taskset", "--cpu-list", ",".join(map(str,settings["cpus"])), *command]
    if settings.get("memory_max_bytes") is not None:
        command = ["systemd-run", "--user", "--scope", "--quiet", "--unit=solid-"+uuid.uuid4().hex,
                   "--property=MemoryMax="+str(settings["memory_max_bytes"]),
                   "--property=MemorySwapMax="+str(settings.get("swap_max_bytes",0)), *command]
    return command


def fio_command(settings, directory, protected=()):
    background = condition(settings).get("background")
    if not background: return None
    path = Path(background["file"]).resolve(strict=True)
    if not path.is_file() or path.stat().st_size < 4096: raise ValueError("fio requires an existing regular test file")
    if any(Path(other).exists() and os.path.samefile(path,other) for other in protected):
        raise ValueError("fio test file overlaps experiment input, model or session")
    random = background["kind"] == "random-read"
    return ["fio", "--name=solid-background", "--readonly", "--allow_file_create=0", "--filename="+str(path),
            "--rw="+("randread" if random else "read"), "--bs="+("4k" if random else "1m"),
            "--ioengine=io_uring", "--direct=1", "--thread=1", "--numjobs=1", "--iodepth="+str(background["iodepth"]),
            "--size="+str(path.stat().st_size//4096*4096), "--time_based=1", "--runtime=86400",
            ("--rate_iops=" if random else "--rate=")+str(background["rate"]),
            "--write_bw_log="+str(Path(directory)/"fio"), "--log_avg_msec=1000", "--per_job_logs=0",
            "--output-format=json", "--status-interval=1", "--eta=never"]


def fio_observations(path):
    """fio status-interval emits concatenated JSON objects, including a final report."""
    data = Path(path).read_text(encoding="utf-8",errors="replace")
    decoder = json.JSONDecoder(); position = 0; objects = []
    while True:
        start = data.find("{",position)
        if start < 0: break
        try: value,position = decoder.raw_decode(data,start)
        except json.JSONDecodeError: break  # The writer can be midway through a status update.
        if isinstance(value,dict) and "jobs" in value: objects.append(value)
    return objects


class Background:
    def __init__(self, command, directory):
        self.command, self.directory = command, Path(directory)
        self.process = self.output = self.error = None

    def __enter__(self):
        if not self.command: return self
        self.output = (self.directory/"fio.json").open("wb")
        self.error = (self.directory/"fio.stderr.txt").open("wb")
        try:
            self.process = subprocess.Popen(self.command, stdout=self.output, stderr=self.error, start_new_session=True)
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                if self.process.poll() is not None: raise RuntimeError("fio exited before workload readiness")
                observations = fio_observations(self.directory/"fio.json")
                if observations and sum(job.get("read",{}).get("total_ios",0) for job in observations[-1]["jobs"]) > 0: return self
                time.sleep(.1)
            raise RuntimeError("fio did not report a completed read within 30 seconds")
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def check(self):
        if self.process and self.process.poll() is not None: raise RuntimeError("fio stopped during inference")

    def __exit__(self, *unused):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.output: self.output.close()
        if self.error: self.error.close()
