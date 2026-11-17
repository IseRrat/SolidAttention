# 阶段 4：微任务调度

首版代码与合成数据检查完成于 2026-11-17。依据为 SolidAttention §6、图 11 和算法 1。当前由 GGML 固定图推进计算，worker 重叠执行历史补读，并在回写和下一层预取之间应用关键路径/LST 优先级。后续扩展为块级就绪队列，覆盖预测读取、纠错及传输任务，补齐路径见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。

## 1. 交付与代码入口

| 模块 | 代码入口 | 完成内容 |
| --- | --- | --- |
| 调度核心 | [scheduler.hpp](../../include/solid/scheduler.hpp)、[scheduler.cpp](../../src/scheduler.cpp) | DAG、拓扑检查、关键路径、LST、就绪优先级、共享完成确认、常驻有界 worker |
| 提前选择 | [attention.cpp](../../src/attention.cpp)、[session.cpp](../../src/session.cpp) | 当前 Q 评分与当前 K 追加分开，保持代表定稿及正式选择语义 |
| 分段读取 | [prefetch.cpp](../../src/prefetch.cpp) | 先纠错历史 KV，再追加并刷新当前尾块 |
| 复制与设备租约 | [device_cache.cpp](../../src/device_cache.cpp) | 历史块提前上传、批量完成确认、上传中的未来 attention 读者 |
| 调度执行 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 独立复制后端、固定主机暂存、读盘/投影重叠、后台回写与下一层预取 |
| 图与同步接入 | [llama_adapter.cpp](../../src/llama_adapter.cpp)、[baseline.cpp](../../src/baseline.cpp) | Q 选择前沿、跨层祖先检查、attention 完成边界复用、错误传播与排空 |
| 固定集合对照 | [schedule_ablation.py](../../scripts/schedule_ablation.py) | 三组配置、输入与来源锁定、logits/选择/租约/DAG/指标核查 |

上游 submodule 继续固定在 `ef6bc6f367082338006c8379bc7bcab3e1584766`。接入代码由主仓库维护，构建目录生成上游适配文件。

## 2. 三种执行模式

`attention.schedule` 默认取 `naive`，`attention.schedule_trace` 默认取 `false`。这两个字段均要求 `prefetch=true`，沿用阶段 3 的 SSD、稀疏、交错布局、GGML decode 和统一层后端约束。

| 模式 | 当前层 | 跨层与完成确认 |
| --- | --- | --- |
| `naive` | Q/K/V 下载后追加、选择、补读、上传、attention | 逐 attention 后端同步确认，沿用阶段 3 的预取流程 |
| `overlap` | Q 就绪即选择，worker 补读历史，计算后端继续 KV 投影 | 独立复制后端；后台满缓冲回写、下一层读取；逐 attention 确认 |
| `overlap+sync_reuse` | 保留 overlap 数据流，上传按批次确认，计算流等待已记录复制事件 | 下一层 Q 的依赖完成证明及请求原有同步边界释放设备读者 |

配置模板为 [offload-prefetch.json](../../configs/offload-prefetch.json)、[offload-overlap.json](../../configs/offload-overlap.json)、[offload-sync-reuse.json](../../configs/offload-sync-reuse.json)。后两组显式打开调度轨迹，便于核查。

```json
"attention": {
  "storage": "ssd",
  "sparse": true,
  "layout": "interleaved",
  "fuse_kv_projection": true,
  "prefetch": true,
  "prefetch_trace": true,
  "schedule": "overlap+sync_reuse",
  "schedule_trace": true
}
```

## 3. 提前选择与计算/I/O 重叠

构图时显式展开仅依赖 Q 的 CPU 选择节点，再遍历当前层 KV 分支。当前 Q 下载完成后，选择器先用 Q 更新过去 token 的代表评分，定稿满足窗口条件的历史块，并形成正式逻辑选择。当前 K 随后追加；当前 token 通过 init/local 和尾块版本进入工作集。

`observe_query` 与 `append_keys` 分别校验位置和调用顺序。改变 Q 的逐 token 测试覆盖提前选择与原串行选择的一致性。固定集合模式在相同前沿读取回放轨迹。

```text
Q 下载完成 → 代表评分/正式选择 ┬→ worker: 历史纠错/补读 → 历史 H2D
                             └→ GGML: KV 投影/RoPE → KV 下载
                                                ↓ 两路完成
                                 追加当前 KV → 尾块刷新/当前 H2D
                                                ↓
                                           attention/FFN
                                                ║
                       worker: 满写缓冲回写、下一层历史读取
```

历史读取只访问位置 `< current_position`，先用旧尾块的有效前缀。KV 下载完成后，主线程先记录其完成边界，再等待 worker，保留两路重叠的实际观测区间。追加当前 KV 后，正式纠错更新尾块 generation 和新增行，其余有效块保持原位。

协调 worker 为一个常驻线程，最多持有一个任务；独立 I/O 线程拥有 liburing，通过有界队列接收读请求并返回完成。两个线程分别预留物理核，详见 [I/O 线程完成记录](../implementation/IO_WORKERS.md)。主线程在层切换、输入更新、完成确认及清理时 join；会话、读写缓冲、主机槽位和复制后端由同一状态锁保护。DAG 自身独立加锁，KV 完成记录可与阻塞读盘并行。

## 4. DAG 与优先级

每个 decode 层形成一帧，共 12 项任务：

| 任务 | 完成依据 |
| --- | --- |
| `q_gpu_ready`、`q_d2h_ready` | 选择节点已获得可读 Q，共享下载完成证明 |
| `selection` | 代表评分及逻辑选择完成；naive 记为 `selection_ready` |
| `kv_projection_to_host`、`kv_d2h_ready` | late CPU 节点已获得当前 KV |
| `miss_read_and_reconcile` | 历史实际块的读取、取消纠错和版本核对完成 |
| `history_h2d` | 历史上传事件完成 |
| `append_current_kv` | 当前 KV 写入缓冲及代表 K 追加；naive 记为 `append_and_selection` |
| `current_h2d` | 当前版本上传事件完成 |
| `attention_until_boundary` | attention 后端回调、后继 Q 依赖或请求同步完成 |
| `buffered_ssd_store` | 满载层写缓冲完成回写；其余轮次保留聚合缓冲 |
| `next_layer_prefetch` | 下一层预测读取已启动 |

构造 DAG 时拒绝环、重复边和无效前驱。任务启动要求全部前驱完成；完成确认要求任务正在运行且时间单调。共享 fence 在修改状态前核对完整前置覆盖和时间，随后按拓扑顺序确认。

初始估算成本为 1 ns，后续使用完成帧观测成本的 1/8 指数平滑。关键路径与 LST 通过前向、反向拓扑计算。就绪优先级依次为关键任务、关键任务较小 LST、其余 I/O、较小 LST 和任务编号。GGML 推进计算图，liburing 推进在途读取；worker 在写回/下一层预取的共同就绪前沿使用该优先级安排两个 I/O 任务。

## 5. 复制与同步复用

overlap 模式在同一注册设备上创建独立 GGML 后端，承载复制流。固定主机暂存区按设备槽位寻址，CUDA 使用后端提供的 pinned host buffer，CPU 使用主机 buffer。暂存区与设备 KV bank 均按单层容量复用。

每次上传先复制到固定暂存位置，再提交异步设置并记录事件。所有票据来自同一复制流；等待最新已记录事件后，批量确认该批票据。计算后端在事件记录之后提交跨流等待。设备与主机端的 producer/reader 租约分别释放，沿用 generation 校验。

`overlap+sync_reuse` 允许设备上传期间登记未来 attention 读者。轨迹为 `upload → attention_pin_waiting → upload_complete → attention_complete`；producer 与 reader 同时存活，槽位持续占用。attention 完成确认还要求上传已被主机确认。

跨层复用先遍历实际计算图，检查下一层 Q 是否以先前原始 Flash Attention 节点为祖先。证明成立时，Q 下载完成边界同时证明前层 attention/FFN 已结束，释放前层设备读者。最后一层使用运行器已有的 `llama_synchronize`，由 `after_process()` 释放读者。保留的逐 attention 回调与显式 fallback 均独立计数。

事件记录、流等待及完成语义依据锁定版本 GGML 接口与 [NVIDIA 事件接口](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)、[异步执行说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)。

## 6. 内存与错误清理

设每个 KV head 为 `S` 个块槽位，块大小 `B`，head 数 `Hkv`，维度 `D`：

```text
staging_payload_bytes = Hkv * S * B * 2 * D * sizeof(FP16)
device_payload_tokens_per_head = align_up(S*B, 32)
```

`schedule.staging_buffer_bytes` 记录后端实际分配容量。阶段 3 的主机 payload、设备 bank、代表向量与各层写缓冲继续分项记录。额外常驻数据为一个 worker、一帧 DAG、当前 Q/选择、12 项成本估计及复制票据；缓冲容量随层预算确定。

后台写入使用现有 buffered 文件接口，将满缓冲回写移到 worker；前台追加仍检查缓冲容量。读计划与回写在同一状态锁下创建，已提交前缀保持不可变，新增脏行从写缓冲转发。

worker 的异常在 join 时传回主线程，adapter 保存首个错误并在采样前抛出。清理先 join，再同步计算与复制、确认上传、释放设备读者、结束主机预取，最后销毁缓冲和文件。中断帧的剩余任务写为 `failed`；离线核查器拒绝失败或未闭合的帧。

计算图中的错误分支先完成 `GgmlPrefetch::quiesce()`，再向剩余 attention 节点提供收尾 mask。独立复制流已释放 KV bank 后，后续节点才读取设备数据。延迟复制回归与错误保留证据见 [工程审查记录](../ENGINEERING_REVIEW.md)。

## 7. 轨迹与计时口径

`schedule_trace=true` 输出 `schedule.jsonl`；每条定义记录 frame/layer/position、模式、task、资源、前驱和成本估计。事件记录 start/complete/failed、完成依据及绝对 `steady_clock_ns` 时间。预取和驻留轨迹增加同一主机时钟的 `timestamp_ns`，供时间轴关联。

任务时间为主机观测到的完成区间。Q GPU/D2H 与 KV D2H 使用零时长证明标记；`kv_projection_to_host` 覆盖选择返回至 KV 可读之间的发射、计算及下载；`attention_until_boundary` 覆盖上传确认至相应完成边界，可包含后续 FFN、Q 和主机等待。naive 的 KV 使用到达时的零时长标记。逐 kernel 分析另用 `profile.nodes` 的串行观测模式；该模式显式增加节点同步，固定集合调度对照使用 `profile=false`。

| `result.json` 字段 | 口径 |
| --- | --- |
| `schedule.frames/tasks/worker_jobs` | 完成层帧、DAG 任务及后台作业数 |
| `observed_critical_path_sum_ms` | 每帧以任务区间为权重的最长依赖链之和，任务间调度空隙另由时间轴观察 |
| `miss_projection_overlap_sum_ms` | 每帧历史纠错/读取与 KV 到主机区间的交集之和 |
| `query_reused_fences/process_reused_fences` | 通过后继 Q、请求同步确认 attention 的次数 |
| `attention_fences/fallback_fences` | 逐 attention 回调、显式安全同步的次数 |
| `copy_event_waits/copy_batch_waits/copy_stream_waits` | 单事件主机等待、批量主机等待、计算流等待调用数 |
| `copy_batch_groups/copy_events_coalesced` | 批次与批次覆盖票据数 |
| `backend_drains/backend_compute_waits` | 复制后端排空与额外计算后端同步，包括清理 |
| `prefetch.history_resolutions/formal_resolutions` | 提前历史纠错与追加后正式纠错次数 |

`ready_hits/inflight_hits/missing_blocks/refreshed_blocks` 统计两次纠错中发生的块决策。对照读取成本使用实际 `async_read_requests/async_read_bytes`、复制字节及任务区间。端到端配置关闭 `schedule_trace`、`prefetch_trace`、`profile` 和 `dump_logits`，每轮独立保存结果。

## 8. 实验环境的三组对照

在当前提交先保存同步 SSD 稀疏交错参考，使用 `profile=false`、`dump_logits=true`。同步参考准备方法见 [阶段 3](PHASE_3.md#6-实验环境的对照步骤)。随后执行：

```bash
python3 scripts/schedule_ablation.py prepare results/sync/result.json results/schedule-configs --results results/schedule-check
for mode in naive overlap sync-reuse; do
  python3 scripts/run_baseline.py "results/schedule-configs/${mode}.json"
done
python3 scripts/schedule_ablation.py compare results/schedule-configs/schedule-ablation.json --atol 0.001 --rtol 0.0001
```

准备工具锁定模型、实际 decode 输入、上下文、预算、布局、投影、队列和正式选择轨迹，三组共用选择回放。可通过 `--queue-depth` 和 `--slots-per-head` 设置共同容量。配置与实验目录置于已忽略的 `results/`，创建新目录保存每次对照。

比较器校验准备配置、实际配置、输入副本、结果与轨迹哈希，以及三组和参考的代码/二进制身份；逐步比较 logits 和选择。驻留核查覆盖延迟 pin 的上传与读者生命周期；DAG 核查覆盖全部 12 项任务、执行依赖、完成顺序和闭合状态，并独立重算临界路径、重叠和完成依据计数。

动态选择的模型级核查另用阶段 3 工具准备候选，再设置调度模式；固定集合对照用于逐项定位调度机制。

## 9. 完成检查

`bash scripts/check.sh --cuda` 通过 CUDA 编译与链接、27 项 CTest、23 项 Python 检查及运行器源码指纹核对。`bash scripts/check.sh --sanitize` 通过 6 项核心 CTest，以及地址、泄漏和未定义行为检查。

| 检查 | 证据 |
| --- | --- |
| DAG 与优先级 | 环/重复边拒绝、前驱完成要求、关键路径/LST、共享 fence 原子校验、精确重叠计算 |
| worker | 条件变量控制的并发进展、单任务容量、异常传递与复用 |
| 选择时序 | 改变 Q、GQA、多轮代表定稿，提前与串行选择一致 |
| 计算图 | 三种模式、两层 GQA、任意物理槽位 mask，CPU Flash Attention 对同步参考 |
| 真实重叠 | 合成 K 投影解除 worker 的缺块读取屏障，验证两路可并发推进 |
| 文件路径 | liburing 临时文件与内存假后端均通过两层图核查 |
| 同步复用 | 每例 12 帧/144 任务，6 次 Q 边界和 6 次请求边界确认 |
| 跨语言轨迹 | C++ 实际计算图导出三种模式轨迹，Python 重算指标并核查设备租约 |
| 离线工具 | 固定输入、三组配置、来源、指标变更、依赖违规、失败/未闭合及提前释放拒绝 |

本机输入为合成张量与临时文件，数学计算使用 CPU 后端。模型数值、CUDA 复制与计算并发、设备延迟、显存峰值及论文实验在实验环境按上述入口核查。阶段索引见 [完成记录](README.md)，后续实验见 [阶段 5](../ROADMAP.md#阶段-5论文实验)。
