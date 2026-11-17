# 阶段完成记录

每个阶段完成代码与检查后，新增一篇核查记录。记录包括目标、代码入口、数据流、配置、检查证据及实验环境的运行步骤。实现变更时同步修订对应记录。

下表记录首版交付。论文对应的完整执行路径继续按 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md) 补齐，其中阶段 1 的 prefill 使用 CPU 流式参考，阶段 4 的 LST 选择作用于回写与下一层预取的共同就绪前沿。

| 阶段 | 核查记录 | 代码完成情况 |
| --- | --- | --- |
| 0 | [上游与原版基线](PHASE_0.md) | 固定依赖、原版运行器、配置与来源校验 |
| 1 | [同步 SSD 与稀疏参考](PHASE_1.md) | KV 文件、流式 prefill、动态选块、内存/SSD 对照入口 |
| 2 | [交错布局与融合投影](PHASE_2.md) | 交错读写、单次 KV 投影、stride attention、固定集合消融工具 |
| 3 | [历史预取与纠错](PHASE_3.md) | liburing、历史预测、补读、DRAM/VRAM 槽位生命周期与轨迹核查 |
| 4 | [微任务调度](PHASE_4.md) | DAG/LST、读盘与投影重叠、独立复制后端、同步复用、三组对照与轨迹核查 |
| 5 | [论文实验代码与评测入口](PHASE_5.md) | 精确输入、CUDA 矩阵、资源与干扰、重复与续跑、准确率接口、官方评分及绘图 |

开发检查在本机使用编译、临时文件和合成张量。模型输出、任务准确率、设备延迟和内存峰值在实验环境核查。阶段顺序见 [实施计划](../ROADMAP.md)，逐次检查结果见 [CHECKS](../CHECKS.md)。

## 核心代码补齐记录

| 完成项 | 核查记录 | 所属路径 |
| --- | --- | --- |
| CUDA 块评分与预算 top-k | [GPU 选择](../implementation/GPU_SELECTION.md) | 阶段 1 的动态选择、阶段 4 的 Q-only 选择前沿 |
| CUDA 后续窗口代表构造 | [GPU 代表构造](../implementation/GPU_REPRESENTATIVES.md) | 阶段 1 的代表构造、prefill/decode 交接、阶段 4 图接入 |
| 有界 CUDA 分块 prefill | [GPU prefill](../implementation/GPU_PREFILL.md) | 阶段 1 的长输入 attention、存储读取及工作区上限 |
| 独立 I/O 与协调线程 | [I/O 线程](../implementation/IO_WORKERS.md) | 阶段 3 的请求/完成及取消；阶段 4 的线程分工和 CPU 核预留 |
| selected 优先与逐块 SSD/H2D 流水 | [块读取流水](../implementation/BLOCK_READ_PIPELINE.md) | 阶段 3 的预测/纠错、阶段 4 的读复制重叠和依赖汇合 |
| 历史块事件 DAG 与资源分派 | [块事件调度](../implementation/BLOCK_EVENT_SCHEDULER.md) | 阶段 3 的原 READ；阶段 4 的块级关键路径/LST、复制票据及两级核查 |
