# 系统设计

## 1. 执行路线

Linux + llama.cpp + CUDA，单请求 decode。先获取并锁定上游，提供全内存参考运行器，定位 KV 管理和计算图接口，再依次实现同步 SSD、动态选块、交错布局、预取和调度。操作步骤见 [开始复现](START_HERE.md)，阶段验收见 [实施计划](ROADMAP.md)，锁定提交的调用路径见 [接口定位](UPSTREAM_MAP.md)。

模型权重使用 INT4，KV 使用 FP16。实验固定权重、输入、生成参数和可见历史；稀疏路径额外固定选择器与上下文预算。

## 2. 逻辑块与磁盘布局

KV 对由 `(layer_id, kv_head_id, token_pos)` 标识；`block_id = floor(token_pos / B)`，`B` 为每块 token 数。每个 head 的块独立选择，GQA 的 query head 到 KV head 映射由模型配置确定。

物理 I/O 单元采用单层、单 KV head、单逻辑块。同步基线使用独立 K/V plane；交错布局的 payload 形状为 `[B, 2, D]`，顺序为 `K0 V0 K1 V1 ...`。`D` 为 head 维度，K/V 维度相同；每个元素占 `s` 字节。实现与文件头见 [同步 SSD 路径](SSD_PATH.md)。

```text
payload_bytes = B * 2 * D * s
separate_slot_bytes = 2 * align_up(B * D * s, io_alignment)
interleaved_slot_bytes = align_up(payload_bytes, io_alignment)
slot_bytes = selected_layout_slot_bytes
blocks_per_head = ceil(context_capacity / B)
slot_index = (layer_id * kv_heads + kv_head_id) * blocks_per_head + block_id
byte_offset = data_offset + slot_index * slot_bytes
```

会话文件头记录布局版本、模型 KV 形状、dtype、容量和对齐；固定槽位只存 payload 与 padding。内存块表记录有效 token 数、绝对位置、版本与读写状态。尾块按完整槽位存储，attention 按有效长度访问。

示例：`B=32, D=128, s=2` 时，独立 K/V 各为 8 KiB，交错请求为 16 KiB。多个 head 的聚合读取作为布局扫描项，同时测量请求数量、读放大和吞吐。跨 head 聚合需要连续地址与相应选择集合。

同步版本使用普通文件 I/O。性能版本比较 buffered I/O 和 direct I/O；direct I/O 对齐参数从目标设备与文件系统获取，应用到文件偏移、长度和缓冲区地址。所有结果记录请求字节与设备实际读取字节。

## 3. 内存预算与缓冲区

decode 按单层工作集复用 DRAM 和 VRAM 缓冲区。设 `C` 为该层预算内 token 数，`Hkv` 为 KV head 数：

```text
layer_kv_bytes = C * Hkv * 2 * D * s
write_buffer_bytes = layers * 32 KiB
representative_bytes = layers * Hkv * blocks_per_head * representative_count * D * s
```

物理分配还包含整块取整和 I/O padding。各 head 的选择预算独立计算，显存容量按全部 head 的槽位总和检查。

容量示例：32 层、8 个 KV head、head 维度 128、FP16、131072 token、32 token/块、每块 1 个代表时，代表向量占 256 MiB；1024 token 的单层 KV 工作集占 4 MiB。块表和对齐开销另计。

| 区域 | 容量与用途 | 释放或复用条件 |
| --- | --- | --- |
| DRAM 读缓冲区 | 一层选中 KV，承接 SSD 读取和 H2D | 对应 H2D 完成且主机读者释放 |
| VRAM attention 缓冲区 | 一层选中 KV 和当前 token KV | 当前 attention 完成 |
| DRAM 写缓冲区 | 每层 32 KiB，聚合新增 KV | SSD 写入完成 |
| 传输暂存区 | 固定容量，容纳 D2H 及尾块更新 | 消费该数据的写入完成 |
| overlap H2D 暂存 | 一层设备槽位对应的 pinned host buffer，实际容量写入结果 | 对应复制事件完成，设备槽位可复用 |
| 元数据和代表向量 | 按层、head、块常驻 | 会话结束 |
| 模型与计算工作区 | 由后端分配 | 按后端生命周期管理 |

DRAM 与 VRAM 缓冲区分别管理生命周期。当前层 attention 期间，DRAM 可在 H2D 完成后装入下一层 KV；下一层 H2D 等待 VRAM attention 缓冲区释放。首版将 init/local 计入同一层预算，随层切换装载。

实验分别记录以上分项、进程内存、页缓存和设备显存峰值。额外暂存区以配置限制容量，容量耗尽时暂停提交并等待消费完成。

## 4. 动态稀疏选择

每层维护 init、local 和 selected 集合。init 覆盖初始窗口，local 覆盖最新窗口，其余历史块按当前 Q 与代表 K 的相关度选 top-k。合并后按逻辑身份去重，保留绝对位置和有效长度。

论文预算：输入长度短于 4k 时取 25%，其余取 1k；init/local 合占一半，selected 占一半，块大小 32。输入长度在请求开始时确定，整个 decode 使用该预算。init/local 的拆分、整块取整和当前 token 的预算归属在阶段 1 锁定，纳入边界测试。

`RepresentativeBuilder` 按 InfLLM §3.2 构造代表 K，记录代表 token 的位置和构造窗口。选择器配置包括代表数量、分数聚合、query head 到 KV head 的聚合、并列排序和 NaN/Inf 处理。构造器与查询端共用同一套 RoPE 和位置规则。

代表向量在块离开 local 窗口并具备评分所需 Q 后定稿；定稿前保留所需 K/Q 工作集，计入内存预算。prefill 尾部和 decode 新块使用同一状态转换。历史块完整保存在 SSD，供后续轮次选择。

CUDA 选择后端接收定稿代表的增量发布，执行块相关度和预算 top-k；候选与 init/local 去重使用同一入口。`selection_backend=auto/cpu/cuda` 控制后端，结果保存实际身份、发布/评分计数和设备分配。算子布局、执行顺序及核查见 [GPU 选择](implementation/GPU_SELECTION.md)。

CUDA 后端同时通过固定 local 窗口计算代表分数，保留每块 R 个候选并直接提交完整代表块。CPU 接管状态包含 pending K/分数及半完成块的候选；全部层初始化后释放 CPU pending 工作集。边界、分配公式和图检查见 [GPU 代表构造](implementation/GPU_REPRESENTATIVES.md)。

## 5. Prefill 与新增 KV

prefill 使用上游投影和有界 microbatch，在 CPU 上逐块读取历史 KV 并在线累计 softmax。decode 收集预算内的 FP16 K/V，接入 GGML Flash Attention。`gpu_decode=false` 提供 CPU 流式参考。固定 token 的 logits 对照入口见 [同步 SSD 路径](SSD_PATH.md)。

每个 microbatch 保留历史位置和 causal mask。代表构造器消费所需 Q/K，满足窗口条件后提交块元数据。decode 在 prefill 写入、元数据和尾块版本全部完成后开始。首次 decode 使用 init/local 预取及正式 selected 选择。

decode 当前 token 的 KV 先进入 attention 工作集，再进入每层写缓冲区。写缓冲区按 32 KiB 聚合，满载或会话结束时写出；部分尾块按版本更新。读取通过块表定位最新版本：已提交版本从 SSD 读取，待提交版本从写缓冲区转发。缓冲区持有至写入完成；写失败使会话进入错误状态。

## 6. KV consolidator

`KvBlock` 保持独立或交错排列，CPU attention 通过行指针读取，decode 工作集通过 stride/view 提供 K/V。模型加载后按 `[head,K/V,D]` 行顺序合并投影权重，用一次矩阵乘法生成交错输出；K 沿用原生 RoPE 并写回对应行。

同类型权重逐行保留量化编码，混合类型逐行转换成 FP32 融合权重。融合 buffer 容量、初始化时间及实际权重类型写入结果。布局和投影独立开关；消融固定 token 块大小、输入、选择集合及 KV 精度。文件调用、工作集构造、后端复制和 attention 分项记录，完成说明见 [阶段 2](phases/PHASE_2.md)。

## 7. 历史预取与槽位管理

每层保存上一 decode 轮的正式 selected 集合。下一轮在当前层 Q 就绪前发出 init/local 和历史 selected 读取。正式选择完成后更新历史。

```text
missing = actual - ready_resident
unused = predicted - actual
```

`ready_resident` 以 layer、head、block、版本和传输完成状态匹配。在途的实际所需块继续等待；其余 missing 块发起补读，优先使用已释放的 unused 槽位。有效槽位保持原位，attention 使用逻辑块到物理槽位映射。

槽位状态为 `FREE → LOADING → READY → PINNED → READY/FREE`。复用条件是旧 I/O 已完成或取消已确认、GPU/CPU 读者均已释放。每次租用递增 generation，完成通知核对 `(slot, generation, block_id, version)`；过期通知直接丢弃。

取消请求提交后，槽位保持占用直至原操作完成或确认取消。generation 用于通知归属校验；物理写入完成状态控制缓冲区释放。预测数量按剩余容量裁剪，实际集合容量在提交前检查。

liburing 路径以原 READ CQE 结束内核写入期。主机槽位和设备槽位分别管理 producer/readers；H2D 完成事件释放主机源租约，attention 完成证明释放设备读者。尾块等待旧读取完成后保留不可变前缀并补齐新行，设备 mask 按实际物理槽位构建。代码与容量公式见 [阶段 3](phases/PHASE_3.md)，同步复用见 [阶段 4](phases/PHASE_4.md)。

## 8. SSD-aware 调度

微任务包括 Q 投影、KV 投影、选择、SSD 读取、H2D、attention、D2H、SSD 回写和 FFN。每条任务依赖使用完成事件表达：

```text
q_proj → select → reconcile → missing_ssd_read → missing_h2d → attention
prefetch_ssd_read → prefetch_h2d → attention
kv_proj → rope_and_current_kv → attention
kv_proj → d2h → write_buffer → ssd_store
attention → ffn → next_layer_q_proj
```

H2D 还依赖目标 VRAM 槽释放；SSD 读取依赖目标 DRAM 槽释放；回写与后续读取通过块版本连接。就绪任务按关键路径优先，再按最晚开始时间排序；非关键 I/O 尽早提交。重点实现 missing load 与 KV projection 重叠、下一层读取与当前层 attention 重叠。

同步复用分别比较 `naive`、`overlap`、`overlap+sync_reuse`。共享同步点覆盖所合并任务的全部前置事件。轨迹分别记录 SSD 完成、主机复制完成、GPU 事件完成及槽位释放。

阶段 4 通过 Q-only CPU 节点建立选择前沿，先更新过去 token 的代表评分并选择历史块，worker 纠错读盘时 GGML 继续 KV 投影。late CPU 节点接收当前 KV 后追加并刷新尾块。独立复制后端与固定 pinned 暂存承载 H2D，worker 在共同就绪前沿按 DAG 的关键路径/LST 安排满缓冲回写和后继预取。

sync_reuse 通过实际图祖先核查，以后继 Q 下载完成证明释放前层 attention 读者；末层使用请求已有同步。上传中的设备槽位允许登记未来读者，producer 和 reader 均完成后复用。详细轨迹采用主机观测区间，逐 kernel 观测使用单独的串行 profile 配置。实现、指标和三组固定集合核查见 [阶段 4 完成记录](phases/PHASE_4.md)。

## 9. 模块接口

```text
SelectionResult select(Query q, LayerHead lh, Position p, Budget b)
PrefetchPlan predict(LayerHead lh, Position p, History h, Budget b)
IoTicket read_block(BlockId id, Version v, SlotLease lease)
IoTicket write_block(BlockId id, Version v, BufferLease source)
CorrectionPlan reconcile(PrefetchPlan predicted, SelectionResult actual, SlotTable slots)
Event schedule(TaskDag dag, Resources resources)
```

选择器输出逻辑块、位置和有效长度，I/O 层负责版本化读写，调度器持有 lease 并管理事件。错误包含会话、轮次、layer、head、block、slot、generation 和操作类型。

## 10. 配置

| 配置组 | 字段 |
| --- | --- |
| 上游与模型 | 主仓库提交、上游基点、适配提交、模型 revision、GGUF 哈希、量化格式 |
| 后端 | CUDA/驱动、GPU 层数、Flash Attention、KV dtype、RoPE 配置 |
| 选择 | block_tokens、context_budget_tokens、init/local/selected_tokens、representative_count、head 聚合、取整规则 |
| 内存 | DRAM 限额、VRAM 预算、各缓冲区容量、prefill microbatch/tile 大小 |
| I/O | SSD 目录、模式、对齐、队列深度、线程绑定、写缓冲区大小 |
| 测量 | 输入哈希、种子、采样、输出上限、缓存状态、重复次数 |

启动时校验模型形状、预算、文件容量、偏移和缓冲区容量。每次实验保存完整配置、命令、日志和原始逐 token 轨迹。
