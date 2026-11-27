# 为实验进程设置内存、显存和交换空间条件，并管理可选的后台 I/O 负载。
"""External-environment process isolation and read-only fio interference."""
import os
import json
from pathlib import Path
import subprocess
import time
import uuid

from experiment_common import number
from run_baseline import cgroup_memory
from process_lifetime import stop_process


def condition(value):
    allowed = {"name", "session_root", "cache_state", "memory_max_bytes", "swap_max_bytes",
               "cuda_memory_max_mib", "cpus", "background"}
    if set(value)-allowed: raise ValueError("unknown experiment condition field")
    if value["cache_state"] not in ("os-managed", "warm", "externally-prepared-cold"):
        raise ValueError("cache_state must describe the externally controlled page cache")
    for key in ("memory_max_bytes", "swap_max_bytes"):
        if value.get(key) is not None: number(value[key], integer=True, minimum=1 if key == "memory_max_bytes" else 0)
    if value.get("swap_max_bytes") is not None and value.get("memory_max_bytes") is None:
        raise ValueError("swap_max_bytes requires memory_max_bytes")
    if value.get("cuda_memory_max_mib") is not None:
        number(value["cuda_memory_max_mib"], minimum=1, integer=True)
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


# 把资源限制包装到子进程启动命令中。
def isolated_command(command, settings):
    condition(settings)
    limit = cuda_memory_limit(settings)
    if limit:
        pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        if not pipe:
            raise ValueError("CUDA memory limit requires a private NVIDIA MPS server")
        mps = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n",
                             text=True, capture_output=True, timeout=5, check=True)
        if len(mps.stdout.split()) != 1 or not mps.stdout.strip().isdigit():
            raise ValueError("expected exactly one active NVIDIA MPS server")
        command = ["env", f"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT={limit}", *command]
    if settings.get("cpus"): command = ["taskset", "--cpu-list", ",".join(map(str,settings["cpus"])), *command]
    if settings.get("memory_max_bytes") is not None:
        # A container can already supply the exact limits. Its child inherits
        # them, and audited_job independently checks the runner's cgroup.
        current = cgroup_memory(pid=os.getpid())
        if (current.get("memory.max") == str(settings["memory_max_bytes"]) and
                current.get("memory.swap.max") == str(swap_limit(settings))):
            return command
        command = ["systemd-run", "--user", "--scope", "--quiet", "--unit=solid-"+uuid.uuid4().hex,
                   "--property=MemoryMax="+str(settings["memory_max_bytes"]),
                   "--property=MemorySwapMax="+str(swap_limit(settings)), *command]
    return command


# 读取实验约定的 CUDA 显存上限，交给启动环境使用。
def cuda_memory_limit(settings):
    mib = settings.get("cuda_memory_max_mib")
    if mib is None:
        return None
    return f"0={mib // 1024}G" if mib % 1024 == 0 else f"0={mib}M"


def swap_limit(settings):
    return 0 if settings.get("swap_max_bytes") is None else settings["swap_max_bytes"]


# 生成后台磁盘负载命令，避开模型及本轮受保护的输入文件。
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
            # fio's SIGINT handler writes its final statistics before exiting.
            import signal
            try: os.killpg(self.process.pid,signal.SIGINT)
            except ProcessLookupError: pass
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: stop_process(self.process,group=True,timeout=5)
        if self.output: self.output.close()
        if self.error: self.error.close()
