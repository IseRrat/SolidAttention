# SolidAttention-Lab 详细设计（v0.2）

## 1. 目标、边界与基线

本项目先复现 FAST '26 SolidAttention 的关键因果链：**完整 KV 留在 SSD → 动态选块 → K/V 交错提高 I/O 粒度 → 历史选择预取 → 依赖感知调度减少阻塞**。目标是检验这些机制在受限内存 PC 上的延迟、内存与准确率权衡；现阶段不是论文代码的移植或性能承诺。论文依据见 [REFERENCES](REFERENCES.md)。

第一实现目标为 Linux、llama.cpp、CUDA、单请求 decode。基线包括全内存推理、朴素全量 SSD offload、同步稀疏 SSD 路径。所有比较必须固定模型权重、量化、输入、生成参数和可见上下文。模型与 llama.cpp 版本在阶段 0 锁定，避免先写适配层后才发现 KV 形状或执行接口变化。

## 2. 术语与数据模型

- `layer_id`、`kv_head_id`、`token_pos` 唯一标识一个 KV 对；使用 KV head 而非 query head，兼容 GQA/MQA。
- `block_id = floor(token_pos / block_tokens)`，块身份固定，不以缓存槽号充当身份。最后一个块可不满；有效 token 数与每个 token 的绝对位置单独记录。
- 每个 token 的 K 和 V 维度、数据类型、量化比例、RoPE 阶段以及 head 映射必须与所选推理后端一致。K/V 在 SSD 和内存中始终成对移动。
- `BlockMeta` 至少含 `(layer, kv_head, block_id, first_pos, valid_tokens, byte_offset, bytes, format_version, state)`；偏移和长度做溢出与边界检查。元数据驻留内存，完整 KV 内容驻留 SSD；init/local 的热块副本允许进入有限缓冲区。
- `BlockSelection` 是本轮**实际**要参与注意力的逻辑块集合；`Prediction` 仅是预取提示，不得改变实际选择结果。

磁盘布局拟按 layer/head/block 连续分配，每块有版本化头部与对齐的 payload。真正的 I/O 对齐由 SSD、文件系统与 liburing 设置确定，不能把论文实验中的 32 token 误作字节对齐要求。阶段 1 先实现普通文件 I/O 与正确性，再评估 direct I/O、队列深度和写缓冲。断电恢复、跨进程共享和持久化模型缓存不列入首版目标；每次会话从已知状态初始化 KV 文件。

## 3. 动态稀疏注意力

对每个 layer 和 KV head，维护固定的 init 区、最新的 local 窗口，以及从其余历史块中选出的 selected 集合。由代表向量与当前 Q 的相似度选 top-k，合并三类集合后去重，且只允许位置早于或等于当前 decode 位置的有效 token 参与注意力。预算按**token 数**核算，边界块实际有效长度也计入；窗口交叠不得双算。长上下文论文配置：32 token/块、≥4k 时上下文预算 1k token，init/local 合占一半，selected 占一半；短于 4k 时预算为输入长度的 25%。这是评测配置，实际 `k`、窗口及不足整块时的取整策略须在实验配置中显式记录。

代表向量的计算需要与论文所引用的 InfLLM 方案及适配模型保持一致。实现前先做独立选择器与全内存参考比较，检查打分方向、top-k 并列、NaN/Inf、head 分组、RoPE、位置掩码和块更新。论文中“动态稀疏”不等于永久丢弃未选块：未选中的历史 KV 继续保存在 SSD，以供未来重新选择。

### 3.1 代表向量的可实施路线

InfLLM 原文 §3.2 的做法是在每个 memory unit 中，根据 token 的 K 与后续 local 窗口 Q 的平均点积计算代表分数，选分数最高的若干 token；查询时以当前 Q 与这些代表 K 的点积汇总得到 unit 相关度。SolidAttention §3.1 说沿用 InfLLM 提取代表向量，但并未规定适用于本项目的代表 token 数、头间共享或所有位置编码边界。因此先写一个可替换的 `RepresentativeBuilder`，以 InfLLM 公式为第一候选；保存每块代表 K 与构造时的绝对位置。`rank_blocks` 按每个 query head 映射到 KV head 后打分，合并 head 选择的规则须作为明确配置，不可默默任选一种。先用全内存实现与逐 token 全注意力对照，再与论文所述选块召回实验对照，冻结参数后才接入 SSD。若模型的 RoPE 处理与 InfLLM 的位置重映射不相容，必须记录适配方案及准确率差异，不宣称是逐字复现。

### 3.2 Prefill 到 decode

prefill 不应套用“上一轮 decode 选择”逻辑。prefill 产生每层每个 KV head 的全部 K/V，按固定块号连续写入会话 SSD 文件，并同步生成代表元数据；只有确认文件、元数据和最后尾块一致后，decode 才可开始。首个 decode token 没有选择历史，使用 init/local 预取加正式 selected 选择。decode 新 token 的 KV 先对当前注意力可见，再按块追加或更新到 SSD；部分块的更新不能覆盖仍被读取的旧版本。会话结束时清理临时文件，任何写失败使后续读取该块失败而不是回退到旧内容。

## 4. KV consolidator 与数据布局

基本 payload 顺序定义为每个 token 的 `K, V` 交错：`K0 V0 K1 V1 ...`，而非先存整块 K 再存整块 V。这样同一 token 的 K/V 一次顺序读取即可获取。论文指出交错可增大传输单元并减少操作数，**不靠扩大 token 块来换带宽**；增大块内 token 数可能损害检索准确率。

首版在 CPU 侧以明确的 encode/decode 实现布局，验证每种 dtype 的字节顺序、padding、对齐和 round-trip。之后在模型初始化时研究合并 K/V 投影权重，使输出直接形成交错布局；attention kernel 通过 stride/view 读出 K 与 V，避免每步重排。融合投影须与原投影逐元素对照；如后端不支持等价视图，先保留可靠的显式转换并单独测量其成本。

## 5. 预取与纠错

每层保存上一 decode 轮的 selected 块 ID。下一轮在正式 Q 可用前，确定性预取 init/local 块，并按该层历史列表预取 selected 块。冷启动时没有历史，只预取确定性块。历史记录应在**正式选择完成后**更新，而不是在预测发出时更新。

正式选择后求 `missing = actual − resident` 与 `unused = predicted − actual`。为 missing 发读请求；可优先占用 unused 的槽位，但必须等该槽位的异步写入结束且没有任何 GPU/CPU 读者，再将新块写入。有效预取槽位不搬迁；用 `logical_block_id → physical_slot` 映射供 attention 访问。论文的“无需重排”依赖 K/V 成对且位置语义正确，不能据此省略位置、mask 或槽位映射。仅当全部实际块就绪、对应事件完成后才启动 attention。预取失败可退回同步 miss load；真正的 SSD 读取失败应显式报错，不能静默使用旧数据。

建议槽位状态：`FREE → LOADING → READY → PINNED → READY/FREE`，回写可另设 `DIRTY/WRITING`。每次复用槽位递增 generation；完成事件同时核对 `(slot, generation, block_id)`，丢弃迟到的旧 I/O 完成通知。`PINNED` 或 `LOADING` 槽不得覆盖。容量不足时先停止额外预测，保留实际选择所需空间；如果实际集合仍无法容纳，配置报错或使用分批注意力的明确路径，不能越界。

## 6. SSD-aware 调度

以任务 DAG 而非固定函数调用顺序定义每层 decode。主要节点：`q_proj`、`kv_proj`、`select`、`prefetch`、`miss_load`、`attention`、`store_new_kv`、`ffn`。约束包括：`q_proj → select → miss_load → attention`；`kv_proj → attention` 且新 KV 必须可见；`attention` 完成才可释放其 pin；新 KV 写入 SSD 前须完成生产者事件；下一层预取可在当前层计算期间提前发起。具体边在 llama.cpp 集成后以事件轨迹验证。

先实现串行参考调度，再加异步 I/O。就绪任务按关键路径优先；相同优先级可参考最晚开始时间；非关键 I/O 尽早发出以覆盖计算窗口。同步点复用只在合并的事件确实覆盖所有先决条件时成立，不能为了减少同步而提前释放 buffer。必须区分 GPU 事件完成、主机复制完成、SSD I/O 完成和数据对 attention 可见。回写与预取可共享合适的同步点，但回写失败必须保留错误状态并阻止后续依赖读取。

## 7. 接口草案与模块边界

```text
SelectionResult select(Query q, LayerHead lh, Position p, Budget b)
PrefetchPlan predict(LayerHead lh, Position p, History h, Budget b)
IoTicket read_block(BlockId id, SlotLease lease)
IoTicket write_block(BlockId id, BufferLease source)
CorrectionPlan reconcile(PrefetchPlan predicted, SelectionResult actual, SlotTable slots)
Event schedule(TaskDag dag, Resources resources)
```

`SelectionResult` 保留每个块的 token 位置和有效长度；`SlotLease` 和 `BufferLease` 持有到事件完成。I/O 层不决定注意力选择，选择器不直接操作 SSD，调度器负责跨层事件依赖。所有异步接口返回可观察的完成或失败，日志记录 layer、轮次、块、槽位、generation 和阶段耗时，便于重放错误轨迹。

## 8. 正确性不变量与失败处理

1. 任何参与 attention 的逻辑块都有匹配的、已完成加载的物理槽位；K/V、位置与 mask 对应同一 token。
2. 在途 I/O 不得修改仍被 GPU 或主机读取的缓冲区；槽位复用必须通过 generation 校验。
3. 预取结果只影响等待时间和槽位安排，不改变实际选择集合或输出数值。
4. 新生成的 KV 在标记 SSD 版本可读前已完成写入；读写错误可定位、可传播，不使用部分块。
5. 预算、文件偏移、块长度、头数和 dtype 都经过边界检查；最后一块 padding 不进入注意力。

故障注入覆盖短读、读失败、写失败、取消、乱序完成、预取全错、槽位不足和模型中断。实验结果记录失败与退化行为，不把异常路径混入正常性能统计。

## 9. 评测与创新决策

测量 token/s、端到端逐 token 平均/P50/P99 延迟、I/O 阻塞、读取字节、请求数、命中率、SSD 带宽、回写量、峰值 DRAM/VRAM、以及长上下文任务准确率。先依次打开交错、预取和调度，保持其他变量一致；再扫描块大小、预算、上下文长度和 I/O 干扰。论文最高 3.1×/98% 是参照数字，不作为首版验收硬指标。创新方向由复现后的瓶颈数据选择，明确区别于论文原机制，见 [ROADMAP](ROADMAP.md) 与 [TEST_PLAN](TEST_PLAN.md)。

## 10. 配置与第一条实施路径

配置文件至少需要 `model_path`、模型/权重哈希、llama.cpp commit、backend、`block_tokens`、`context_budget_tokens`、`init_tokens`、`local_tokens`、`selected_tokens`、`representative_count`、选择规则、SSD 文件目录、I/O 模式和队列深度。启动时检查预算之和、块大小、模型 KV 形状和缓冲区容量。所有结果输出同一份完整配置，避免参数写在源代码里无法复查。

实际开发顺序：

1. 锁定 llama.cpp 版本，画出 prefill/decode 的 KV 生产、写入和 attention 消费点；记录所选模型的层数、query/KV head、head 维度、dtype、RoPE 与 mask 规则。
2. 写纯内存的布局编码器、选择器和代表构造器；通过确定性小张量测试和全注意力参考。
3. 接入单会话 SSD 文件，先同步读写与无稀疏全量加载，再启用稀疏选块；在每步记录逻辑块、物理槽及实际参与 token。
4. 验证交错布局与原布局数值一致，单独测 I/O 粒度、请求数和转换成本。
5. 启用历史预取和纠错，先在受控延迟的模拟 I/O 下覆盖全错、乱序与容量边界，再接 liburing。
6. 最后接异步调度；逐条解除串行屏障，每解除一条就跑数据依赖与故障注入测试。

每一步的失败回退路径和验收项见 [测试计划](TEST_PLAN.md)。代码目录与具体 CMake 目标在选定 llama.cpp 版本后固定，避免此文档虚构上游接口。所有未决选择集中记录在 [DECISIONS](DECISIONS.md)，作出决定时附证据和影响范围。
