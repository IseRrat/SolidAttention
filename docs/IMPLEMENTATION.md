# 实现记录

更新日期为 2026-11-19。本文件汇总仓库已有代码、实际调用链、验证状态和已定位问题。后续代码修改同步更新对应条目。每条记录以源码入口和执行行为为依据，核查状态随验证结果更新。

## 当前版本与验证状态

GitHub 最近确认的提交为 `d3be8d0`，包含 SYCL 分块 prefill 和 attention 完成路径。该提交之前依次接入共享 KV、设备选块、图内 Q/KV 和独立复制队列。

本地另外保存投影及传输设备计时、调度成本更新，以及中断前修改的 CUDA 多 query 共享 KV prefill。计时改动通过双后端编译、源码指纹检查和相关既有 CPU 图检查。CUDA prefill 最新修改的构建结果待核实。最近一次提交与推送因自动审批服务额度限制受阻。

本机已进行编译、CPU 合成张量和临时文件检查。模型推理、真实 GPU 数值、性能及论文实验数据待实验环境核查。各模块已有检查通过情况对应当时源码，最新改动需要匹配最新二进制指纹。

## 上游和模型接入

- llama.cpp 通过 submodule 固定到 `ef6bc6f367082338006c8379bc7bcab3e1584766`，CMake 检查版本。 [版本约束](../cmake/Upstream.cmake)
- 生成器读取固定上游，在构建目录生成图、context、通用 backend 和 CUDA/SYCL backend 的接入文件，检查替换锚点。 [生成器](../scripts/generate_adapter.py) [构建接入](../cmake/Adapter.cmake)
- 模型图将 Q/K/V 接入 `AttentionNode`，输出继续进入上游 attention 输出投影、残差和 FFN。原生 KV 分配由适配器接管。入口支持单请求、连续绝对位置、统一层形状的 Llama 和 Qwen2。 [适配器](../src/llama_adapter.cpp)
- 配置检查类型、未知及重复字段、路径、预算、GGUF 分片与特性组合。运行器提供 `--config`、`--build-info` 和配置解析入口，保存结果、来源哈希和运行状态。 [配置](../src/config.cpp) [运行器](../src/baseline.cpp) [启动工具](../scripts/run_baseline.py)

## KV 存储和交错投影

内存和 SSD 存储共用 `KvGeometry`、`KvBlock` 与会话接口。KV 为 FP16，每块默认 32 token。SSD 文件按 layer、KV head、逻辑块定位，文件头为 4096 字节，标记 `SOLIDKV1`。separate 布局分离 K/V plane，interleaved 布局为 `[block token, K/V, D]`。 [存储接口](../include/solid/kv_store.hpp) [存储实现](../src/kv_store.cpp)

每层默认 32 KiB 聚合写缓冲。读取合并磁盘内容与待写 token，短传输补齐剩余字节。异步回写打包区持有源数据，全部原 WRITE 完成后推进提交版本。dirty 数据可在写入完成前转发给读取者。 [存储实现](../src/kv_store.cpp) [异步读写](../src/async_reader.cpp) [预取器](../src/prefetch.cpp)

融合投影在初始化时按 KV head 拼接 K/V 权重行，并保留相同量化格式的原始行编码。权重类型不同会转为 F32。bias 按同一排列拼接。单次矩阵乘法输出 `[2D,Hkv,T]`，K/V 使用 stride 视图，RoPE 后把 K 写回交错视图。当前模型仍持有原始 K/V 权重，融合权重额外占用容量并记录分配。 [融合投影](../src/projection.cpp)

Decode 设备操作把当前 K/V 转为 FP16 `[T,Hkv,2,D]`，Q 与就绪标量建立图依赖。CUDA 和 SYCL 各有设备实现。 [图操作](../src/device_ops.cpp) [CUDA 操作](../include/solid/device_ops.cuh) [SYCL 操作](../include/solid/sycl_device_ops.hpp)

## 稀疏选择和代表构造

自动预算在 prompt 小于 4096 token 时取长度的四分之一，达到 4096 时取 1024。预算一半分配给 init/local，长输入默认 init 32、local 480，其余最多 512。显式配置可指定预算和窗口。候选按 token 去重，最终块按剩余 token 成本截取。位置沿用上游原生 RoPE 的绝对位置。 [预算与模型入口](../src/llama_adapter.cpp) [选择逻辑](../src/attention.cpp)

代表构造保留后续 local 窗口，对对应 GQA 组的 Q 求均值，累计其与历史 K 的点积。token 收齐后续窗口后参加所属块的 top-R，完整块定稿后发布代表。代表同分按 token 位置排序，块同分按逻辑块号排序。 [CPU 索引](../src/attention.cpp) [共同插入规则](../include/solid/representative_math.hpp) [种子校验](../src/representative_builder.cpp)

CUDA 保存每层代表 bank、pending K 环、窗口分数和部分块 top-R。设备 kernel 完成均值、窗口更新、定稿、块评分与预算 top-k。支持 CPU 状态接管、设备 Q/K 借用和 prefill 批量构造。当前块评分由每线程串行遍历一个候选块的维度、GQA 和代表，top-k 按 head 做归约。 [CUDA 选择器](../src/cuda/block_ranker.cu)

SYCL 在同一会话协议上维护设备 bank 和环形状态，设备支持 FP64 时使用 FP64，其余使用两个 FP32 分量。支持图内输入和批量构造，实际身份及上传字节写入运行结果。 [SYCL 选择器](../src/sycl/block_ranker.cpp)

`selection_backend=auto/cpu/cuda/sycl` 控制后端，`auto` 按 attention 设备接入。固定选择回放和动态选择使用同一预算、身份与工作集入口。

## Prefill 和 Attention

CPU 参考路径逐块读取因果前缀，使用在线 softmax。CUDA、SYCL 的设备路径按 query microbatch 和有界 KV tile 累计最大值、分母及带权 V。历史 KV 由存储读取并上传，当前批次使用设备交错 KV，输出留在 GGML 设备缓冲。全部 prompt 层完成后释放 prefill 池并保留峰值计数。 [会话](../src/session.cpp) [CUDA prefill](../src/cuda/tiled_prefill.cu) [SYCL prefill](../src/sycl/tiled_prefill.cpp)

CUDA 最近核查过的 prefill 使用每 query head 一个 warp，按 token 顺序累加。本地新增修改将四个 query 放入一个 CTA，分批共享八行 K/V，维度 1024 时共享区为 32 KiB，因果遮罩、尾 query、跨 tile 累加和失败分支保持全 CTA 的 barrier 一致。该修改的编译和数值结果待核实。

Decode 使用物理槽位对应的逐 query head mask。CUDA 调用上游 strided Flash Attention。SYCL 的自写内核以 64 行切片执行 FP32 在线 softmax，再合并切片，支持交错 K/V stride 和逐 head mask。 [Decode 接入](../src/llama_adapter.cpp) [SYCL decode](../include/solid/sycl_flash_decode.hpp)

`prefill_backend=auto/cpu/cuda/sycl` 与 `prefill_tile_tokens` 控制分块后端和容量，设备池、主机工作区、上传和结果字节写入统计。

## 历史预测、纠错和缓冲所有权

预测来自同层上一 decode 轮 selected 历史，同时确定性预取 init/local。正式 Q 到达后按完整块身份核对集合，提升已有请求，补读缺块，回收错误预测占据的槽位。命中的物理槽位保留，mask 对应实际槽位排列。 [预取](../src/prefetch.cpp) [设备工作集](../src/device_cache.cpp)

| 资源 | 身份和持有者 | 释放依据 |
| --- | --- | --- |
| SSD 块 | layer、head、block 和有效版本 | 原读写请求的完成回执 |
| DRAM 槽位 | 身份、generation、producer 和 readers | 原 READ 完成，所有上传源读者归还 |
| VRAM 槽位 | 身份、generation、复制 producer 和 attention 读者 | 复制完成，原 attention 读取结束 |
| 当前图 Q/KV | GGML 张量及显式依赖 | 选择、代表追加、当前行复制的消费者完成 |
| 回写打包区 | 单次聚合写的所有 WRITE 片段 | 全部 WRITE 完成 |

异步读取基于 liburing，取消后仍等待原请求完成以释放缓冲。正式纠错优先于预测，selected 优先于 init/local，同类任务使用 critical/LST 和稳定顺序。 [I/O 接口](../include/solid/async_reader.hpp) [liburing 实现](../src/async_reader.cpp) [槽位接口](../include/solid/block_identity.hpp)

独立 I/O 所有者和 SSD/GPU 协调线程使用有界队列，支持完成通知、取消排空与异常传播。可配置物理 CPU 绑定，推理线程排除预留核心。 [I/O worker](../src/io_worker.cpp) [线程安排](../src/thread_affinity.cpp)

SYCL 在支持统一内存和共享 USM 的设备上分配共享 KV bank，主机把历史块写入已释放槽位。其他设备使用复制队列。独立 SYCL 复制队列与 GGML 使用相同设备和 context。 [共享分配接入](../scripts/generate_adapter.py) [复制队列](../src/sycl/copy_queue.cpp)

## 调度和当前 KV 归档

`schedule=naive/overlap/overlap+sync_reuse` 共用工作集。层图记录 Q 前沿、选块、KV 生成、补读、历史上传、当前行安装、attention、归档、回写和下一层预取。正式块图按块记录 READ→H2D→汇合，预测图计算 read/copy 调度键。 [层调度](../src/ggml_prefetch.cpp) [DAG](../src/scheduler.cpp) [块调度](../src/block_stage.cpp)

当前行先复制到 bank，尾块已有前缀保留，缺口从主机补齐。当前行安装完成后，attention 与主机归档分别推进。协调线程收割归档后追加会话并发起聚合回写，多层请求可提前派发下一层 SSD 读取，单层请求等待本层追加。 [当前行与归档](../src/ggml_prefetch.cpp)

CUDA attention 用原生事件和图重放通知确认结束，SYCL 用合并 kernel 完成事件和带 epoch 的通知确认结束。协调线程归还旧工作集读者，再上传后继就绪块。请求和后继 Q 的图边界也提供完成依据。 [完成协议](../src/attention_events.cpp) [CUDA 事件](../src/cuda/attention_events.cu) [SYCL 事件](../src/sycl/attention_events.cpp)

本地设备计时改动在投影矩阵乘法前记录起点，在设备 Q/KV 就绪前沿记录终点，并收集历史上传、当前 KV 归档区间。任务成本优先使用设备回执，历史成本平滑更新后用于后继派发。主机时间用于提交与依赖顺序。 [计时绑定](../include/solid/device_timing.hpp) [回执接入](../src/ggml_prefetch.cpp)

## 已有运行和结果工具

| 入口 | 已写的功能 |
| --- | --- |
| `scripts/run_baseline.py` | 构建来源、模型和配置核对，运行监控，内存采样，日志与原子结果 |
| `scripts/process_lifetime.py` | 子进程组所有权、中断和超时清理 |
| `scripts/prepare_replay.py`、`layout_ablation.py` | 固定选择回放与布局对照 |
| `scripts/prepare_prefetch.py`、`compare_prefetch.py` | 预取配置、数值和槽位轨迹核查 |
| `scripts/schedule_ablation.py` | 三种调度入口，依赖和关键路径记录核查 |
| `scripts/experiments.py` | 矩阵生成、执行、续跑、完成核对和汇总 |
| `scripts/accuracy.py`、`accuracy_client.py`、`opencompass_model.py` | 准确率生成与 OpenCompass 接口 |
| `scripts/accuracy_report.py`、`plot_experiments.py` | 结果整理、评分来源和可追溯绘图 |
| `scripts/compare_logits.py`、`selection_analysis.py` | logits 与选择集合分析 |

已有配置覆盖原版、内存/SSD、稀疏、交错、融合投影、预取及三种调度。模型与外部数据来源锁定在 `configs`，实验矩阵代码保存在仓库，实测结果待实验环境生成。

## 构建和核查入口

依赖包括 Linux、CMake、Ninja、C++17、Python，以及所选设备的 CUDA 或 oneAPI 工具链。liburing 用于 SSD 异步路径。

```bash
git submodule update --init --recursive
cmake --preset cuda
cmake --build --preset cuda
cmake --preset sycl
cmake --build --preset sycl --target solid-baseline
python3 scripts/run_baseline.py --check-build
python3 scripts/run_baseline.py --runner build/sycl/bin/solid-baseline --check-build
```

现有检查入口为 `scripts/check.sh`，支持核心、CUDA 构建和 CPU 检查、sanitizer 及 adapter sanitizer。真实 GPU 检查程序在实验环境执行。

最近计时改动的验证通过了双后端运行器构建与指纹核对，10 项已有 CPU 调度/设备输入检查，以及 7 项调度工具检查。检查中修复了复制后端身份读取导致的合成用例崩溃。最新 CUDA prefill 修改发生在此后。

## 论文核对中已定位的问题

1. 论文 §4、图 8 描述预合并权重直接产生交错 KV，并通过 stride 访问 attention。现有路径还包含 RoPE K 写回与 F32→FP16 consolidator。需要逐项核对写回、转换和额外权重容量，确定可以融合的操作及其数值要求。
2. 论文 §6.1、图 11 的选择与补读可和 KV 投影重叠。当前 `adapter_attention` 显式展开 CPU 选择前沿，设备 consolidator 又依赖该前沿。当前读取与 KV 生成有重叠，选择阶段与 KV 投影的实际并行程度仍需逐节点核对。
3. 层任务图、块任务图和 I/O 队列已分别接入成本和回执。跨层统一资源竞争、deadline 传递以及 critical/LST 的生效范围仍需核对，特别是提前派发的 READ、回写和后继上传。
4. CUDA 块评分和窗口更新含线程内串行点积。并行化需要明确累加精度、并列排序和候选稳定性。论文 §7 给出 GPU kernels 总量，正文 §4 同时明确允许复用框架的 strided attention，具体作者内核拆分待源码证据确认。
5. CPU 和 CUDA 将代表窗口分数除以 local 后回填。当前 SYCL 定稿路径保存累计分数，回填归一化需要修正并核对状态接管。
6. 中断前的 CUDA 多 query prefill 修改需先核实构建结果和 barrier、因果边界、共享容量。其算法属于当前工程选择，论文正文对 prefill kernel 的具体拆分给出信息有限。

论文依据为本地 SolidAttention PDF 的 §3.1、§4–§7、图 3、5、7、8、10、11 与算法 1。图 3 对应三层缓冲和三个系统模块，图 8 对应权重合并，图 10 对应预测纠错，图 11 对应微任务重叠。 [论文](https://www.usenix.org/system/files/fast26-zheng.pdf)
