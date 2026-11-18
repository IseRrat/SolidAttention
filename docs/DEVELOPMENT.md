# 构建与检查

已执行的检查记录见 [CHECKS](CHECKS.md)。

同步 SSD、内存参考和稀疏模式的配置及数值对照见 [SSD_PATH](SSD_PATH.md)。

## 源码

```bash
git submodule update --init --recursive
```

上游基点为 `ef6bc6f367082338006c8379bc7bcab3e1584766`。配置入口为 `cmake/Upstream.cmake`，接口位置见 [UPSTREAM_MAP](UPSTREAM_MAP.md)。

## 检查代码

构建配置与核心库并运行单元检查：

```bash
bash scripts/check.sh --unit-only
```

构建 CUDA 运行器并运行配置、命令行与脚本检查：

```bash
bash scripts/check.sh --cuda
```

构建依赖 Linux、CMake 3.22+、Ninja、C++17、Git 和 Python 3。CUDA 构建使用目标环境的 Toolkit，可通过 `CUDACXX` 指定编译器。

Linux 默认启用 `SOLID_URING`，准备 `liburing-dev` 开发包。纯可移植核心构建可设置 `-DSOLID_URING=OFF`；预取入口检查对应构建特性。`bash scripts/check.sh --sanitize` 对核心库与配置运行地址、泄漏和未定义行为检查。

`bash scripts/check.sh --sanitize-adapter` 在独立 CPU 构建目录检查当前 KV 复制、三种设备输入图、融合输入、复制失败排空与原生完成协议。7 项检查启用地址、泄漏和未定义行为检查，覆盖 GGML 适配器及异步张量描述符。

`solid-baseline --build-info` 输出编译时的源码哈希。`scripts/run_baseline.py --check-build` 校验二进制与当前源码是否一致；启动实验前自动执行同一检查。

## 在实验环境运行原版基线

首个适配模型使用 [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f) 的 Q4_K_M 两分片。仓库 revision、文件哈希和大小保存在 `configs/models.lock.json`。

复制 `configs/qwen2.5-7b-q4_k_m.json` 为 `configs/baseline.local.json`，填写实验环境路径；路径相对于配置文件所在目录。配置采用 greedy 采样、F16 KV，显式指定 Flash Attention 和 GPU 层数。每轮使用独立的结果目录。

```bash
sha256sum models/*.gguf
python3 scripts/run_baseline.py configs/baseline.local.json --prepare-only
```

`--prepare-only` 校验模型哈希、固定提交和配置，保存运行输入。正式运行使用另一个全新的结果目录：

```bash
python3 scripts/run_baseline.py configs/baseline.local.json
```

结果目录包含：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 实际运行配置与绝对路径 |
| `prompt.txt` | 本轮输入副本 |
| `manifest.json` | 代码、二进制、模型和输入哈希，系统配置、进程退出状态 |
| `result.json` | 模型元数据、输入/输出 token、逐 token 计时、停止原因 |
| `memory.jsonl` | 进程 RSS/HWM、锁页、swap 与 `/proc/PID/io` 采样 |
| `stdout.txt`、`stderr.txt` | 完整标准输出和后端日志 |
| `selection.jsonl` | `selection_trace=true` 时每层、每轮、每 KV head 的实际选择范围 |
| `selection.replay.jsonl` | 固定选择回放时的输入轨迹副本 |
| `prefetch.jsonl`、`residency.jsonl` | `prefetch_trace=true` 时的主机预取与设备槽位生命周期 |
| `schedule.blocks.jsonl` | `schedule_trace=true` 时历史块 READ/H2D/汇合图 |
| `schedule.jsonl` | `schedule_trace=true` 时的逐层 DAG、任务依赖、主机观测区间及完成依据 |

启动脚本要求主仓库和上游源码已提交；本地配置与结果目录由 `.gitignore` 排除。内存采样间隔为 100 ms，manifest 标注采样峰值及 cgroup 范围；VRAM 和 KV 分配信息保留在后端日志中。

交错布局、融合投影、固定集合消融与分项观测入口见 [阶段 2](phases/PHASE_2.md)。各阶段完成后整理的实现与检查证据见 [阶段记录](phases/README.md)。

liburing 历史预取、同步参考对照与轨迹核查见 [阶段 3](phases/PHASE_3.md)。微任务 DAG、独立复制后端、同步复用和三组固定集合对照见 [阶段 4](phases/PHASE_4.md)。性能配置关闭 `selection_trace`、`schedule_trace`、`prefetch_trace`、`profile` 和 `dump_logits`，保持同一输入与正式选择规则。

精确 token、实验准备/执行/续跑/汇总、cgroup 与只读 fio、OpenCompass/LongBench 接口、准确率表和绘图命令见 [阶段 5](phases/PHASE_5.md)。`--tokenize CONFIG` 读取词表并输出 token JSON；原生 `kv_dtype=q4_0` 供准确率对照，自定义 attention 使用 F16。

异常路径、独立复制排空、SIGTERM/子进程清理和实验完成状态的检查见 [工程审查记录](ENGINEERING_REVIEW.md)。当前完整入口覆盖 51 项 CTest 和 64 项 Python 检查；合成进程组检查在 Linux 执行。

CUDA 评分与 top-k 的内核接入、`selection_backend`、分配计数及实验环境设备检查见 [GPU 选择完成记录](implementation/GPU_SELECTION.md)。本机完整检查编译设备检查程序，数学执行使用 CPU 合成数据。

CUDA 窗口构造、完整状态接管和预取调度图衔接见 [GPU 代表构造完成记录](implementation/GPU_REPRESENTATIVES.md)。核心库 8 项 CTest 覆盖地址、泄漏和未定义行为检查。

CUDA 分块 prefill、配置、内存峰值与池释放见 [GPU prefill 完成记录](implementation/GPU_PREFILL.md)。加入该路径后，核心库 9 项 CTest 通过地址、泄漏和未定义行为检查。

独立 I/O 与协调线程、`io_cpu/coordinator_cpu`、物理核预留、有界队列及取消协议见 [I/O 线程完成记录](implementation/IO_WORKERS.md)。当前核心库 15 项 CTest 通过地址、泄漏和未定义行为检查。

selected 优先、正式请求提升、SSD/H2D 流水和依赖汇合见 [块读取流水完成记录](implementation/BLOCK_READ_PIPELINE.md)。

历史块 READ/H2D 图、关键路径/LST、`schedule.blocks.jsonl`、实际队列顺序及完成依据核查见 [块事件调度完成记录](implementation/BLOCK_EVENT_SCHEDULER.md)。

CUDA attention 原生事件、图重放通知、后台 H2D、正式选择接管和设备区间见 [原生完成事件记录](implementation/NATIVE_COMPLETION.md)。设备检查程序完成编译，运行入口保留在实验环境；本机使用 CPU 合成协议。

Decode 设备 Q/K、FP16 consolidator、三种调度共用输入及计算图保活检查见 [设备输入完成记录](implementation/DEVICE_INPUTS.md)。`solid-cuda-device-input-test --run-device-check 0` 在实验机核对真实 CUDA split 图和同图多轮复用；本机完整入口只编译该程序。

Prefill 设备 Q/当前 KV、批量代表、设备输出视图和图结果内存见 [Prefill 设备接入](implementation/PREFILL_DEVICE_INPUTS.md)。本机运行 CPU split 图、临时文件、读取失败和预取 decode 交接；`solid-cuda-device-prefill-test --run-device-check 0` 在实验机执行设备核查。原生通知用例可通过 `solid-native-completion-test --repeat 30` 重复核对唤醒与回收。

当前 KV 行 D2D、尾块前缀原槽复用和历史缺口 H2D 见 [当前设备 KV](implementation/CURRENT_DEVICE_KV.md)。三种模式及延迟复制失败纳入 CPU 图核查；设备检查程序同时核对当前行复制计数。
