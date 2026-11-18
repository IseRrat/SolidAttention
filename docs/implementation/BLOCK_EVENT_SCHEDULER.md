# 历史块事件 DAG 与资源分派

代码与合成检查完成于 2026-11-18。接入论文 §6.1、算法 1 的历史工作集调度：每块读取、每块 H2D 与整体就绪通过事件连接，关键路径和 LST 决定实际分派。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 块身份 | [block_identity.hpp](../../include/solid/block_identity.hpp) | layer/head/block/version 与槽位 generation |
| 块任务图 | [block_stage.cpp](../../src/block_stage.cpp) | `2*N+1` 项任务、读取/H2D 就绪次序、完成汇合、错误终止及计数 |
| 资源顺序 | [scheduler.cpp](../../src/scheduler.cpp)、[async_reader.hpp](../../include/solid/async_reader.hpp) | 关键路径、LST、类别与 FIFO 调度键；候选集合查询 |
| SSD 执行 | [prefetch.cpp](../../src/prefetch.cpp)、[io_worker.cpp](../../src/io_worker.cpp) | 原 READ 完成连接块任务；调度键传入独立 I/O 队列 |
| H2D 执行 | [device_cache.cpp](../../src/device_cache.cpp) | 按就绪图启动上传，复制票据完成后确认任务及源释放 |
| 图接入与成本 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 历史计划、常驻状态、帧间估算更新、异常排空 |
| 结果与核查 | [baseline.cpp](../../src/baseline.cpp)、[schedule_ablation.py](../../scripts/schedule_ablation.py) | 块级文件哈希、计数、独立重算及父层帧一致性 |

## 2. 任务及资源就绪

```text
block_read_A → block_h2d_A ─┐
block_read_B → block_h2d_B ─┼→ historical_blocks_ready
block_read_C → block_h2d_C ─┘
```

N 为正式历史工作集的块数，每块包含完整逻辑身份与版本。工厂返回的集合必须与正式选择完全匹配。读取可先进入有界主机准入队列；I/O 所有者按关键任务优先、较小 LST、类别及 FIFO 次序填充内核 SQ。相同成本下保留 selected 优先次序。

读取成本或后继 H2D 成本改变时，最长依赖链与 LST 会改变分派顺序。资源反压只限制提交，已进入内核的请求继续沿原有 READ/取消协议结束。H2D 候选由已完成的块读取解锁；复制流接受就绪任务，票据保留两端租约。

一个块的全部 READ segment 完成后确认 `block_read`。独立 K/V 文件的两段读取共同结束后才可上传。H2D 提交只启动任务；完成票据确认后结束任务。最后一项 H2D 完成后，汇合节点才完成。层级追加前沿继续等待历史读取、历史复制和当前 KV 下载。

## 3. 常驻数据、观测与错误

计划建立时已有的主机块标注 `host_resident`；写缓冲转发标注 `host_forwarded`。已在途读取从计划观察边界开始记录，start 带 `inflight_at_plan`；其结束仍由原 READ 完成确认。已在设备的块通过 `device_resident` 确认复制节点。

读取/H2D 初始估算为 1 ns，后续采用已完成帧的主机观测成本作 1/8 指数平滑。主机常驻、转发及计划前已在途读取从读取成本样本中排除；设备常驻命中从复制成本样本中排除。观测区间包含队列或完成收割等待。设备执行时间随设备事件接入继续单列。

任务与映射容量受当前单层历史块数约束。上传票据持有共享任务图，直到复制结束。中止将剩余任务标记失败，已发出的读和复制继续排空并释放源；后到的完成保留物理回收能力。离线核查拒绝失败或缺少完成的图。

## 4. 文件与结果字段

`schedule_trace=true` 时同时保存：

- `schedule.jsonl`：原有每层 12 项任务。
- `schedule.blocks.jsonl`：历史块的 READ/H2D/汇合图，带 head、block 和 version。

块帧与父层帧的 frame/layer/position 一致；naive 保留空块轨迹，overlap 与同步复用模式覆盖全部已完成层帧。结果记录 `block_schedule_trace_sha256`、`schedule.block_frames`、`block_tasks` 和 `block_observed_critical_path_sum_ms`。

核查器验证每个 READ 对应同身份 H2D、真实完成依据、完整汇合、资源类型、时间单调及父层身份，并独立重算块级临界路径和计数。缺失块帧、位置变更、取消 ACK 充当 READ 完成、提交充当复制完成均被拒绝。

## 5. 检查证据与后续接入

CUDA 编译与链接、38 项 CTest、61 项 Linux Python 检查及源码指纹通过。核心库 11 项 CTest 的地址、泄漏及未定义行为检查通过。

[block_stage_test.cpp](../../tests/block_stage_test.cpp) 核对关键路径/LST、未知身份、错误启动、原 READ 依赖、复制票据和空历史；容量 1 的读取队列以实际文件偏移证明成本改变会改变提交次序。另覆盖两段 K/V、第二次上传失败、上传进行时读取失败及完整源回收。

[io_worker_test.cpp](../../tests/io_worker_test.cpp) 在内核容量 1 下核对关键任务、LST、类别和同级 FIFO。[audit_schedule_graph.py](../../tests/audit_schedule_graph.py) 对三组实际 CPU GGML 图的内存与线程化 liburing 路径重算两级轨迹及指标。[test_schedule_tools.py](../../tests/test_schedule_tools.py) 覆盖错误依赖、资源、身份、完成依据和父层帧缺失。

[CUDA attention 完成事件](NATIVE_COMPLETION.md) 已接入后继上传。后续把预测期读取、当前 KV 的 D2H/回写和跨层槽位释放接入统一资源图。范围见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。模型、GPU 执行与性能评测在实验环境进行。
