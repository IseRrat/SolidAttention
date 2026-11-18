# 实施计划

开发检查使用编译和合成数据单元测试。模型推理、性能与准确率评测在实验环境执行。

每个阶段完成代码与检查后，在 [阶段完成记录](phases/README.md) 新增该阶段的实现与核查说明。

当前阶段记录描述首版代码及其检查结果。论文核心路径的后续开发见 [论文实现核查](PAPER_IMPLEMENTATION_AUDIT.md)：完整资源图、投影/传输设备观测及 SYCL。先完成这些代码，再推进正式评测与阶段 6。

已完成的核心补齐：[CUDA 块评分与动态选块](implementation/GPU_SELECTION.md)、[CUDA 窗口代表构造](implementation/GPU_REPRESENTATIVES.md)、[有界 GPU 分块 prefill](implementation/GPU_PREFILL.md)。[独立 I/O 与协调线程](implementation/IO_WORKERS.md) 也已完成。[历史块事件调度](implementation/BLOCK_EVENT_SCHEDULER.md) 已接入正式工作集。[CUDA attention 完成事件](implementation/NATIVE_COMPLETION.md) 已驱动后继上传。[Decode 设备 Q/K](implementation/DEVICE_INPUTS.md) 已接入三种调度和 FP16 KV 追加。[Prefill 设备输入与输出](implementation/PREFILL_DEVICE_INPUTS.md) 已接入批量代表、历史分区和图结果。[当前设备 KV](implementation/CURRENT_DEVICE_KV.md) 已接入 D2D、尾块原槽保留及缺口 H2D。后续推进预测期任务图及其他算子/传输事件。

## 阶段 0：上游与原版基线

获取 llama.cpp，固定提交和模型，完成 Linux/CUDA 构建与短输入推理。记录环境、GGUF 哈希、生成配置、耗时及内存。沿 KV 生成、位置处理、attention 和后端执行路径填写接口表，确定长输入 prefill 的分块接入点。

代码产物：固定的上游依赖、统一构建入口、原版运行器、配置与哈希校验、结果记录脚本、接口定位表。

实验产物：环境记录、固定模型的原版结果与逐 token 轨迹。

验收：相同配置可重复运行；KV 形状、dtype、位置规则和计时区间可查。

## 阶段 1：同步 SSD 与稀疏参考

先实现 KV 文件布局和同步全量读写，建立 Offload 基线。短输入与 dense 全内存参考对照；长输入实现分批 prefill、历史 KV tile 读取及有界工作区。接着实现代表构造和动态选块，建立 InDRAM+Sparse 与 Offload+Sparse。

产物：文件格式、同步 SSD 路径、选择器、长上下文 prefill、内存分项记录。

验收：全量路径通过数值对照；稀疏内存与 SSD 路径选择集合一致、输出在容差内一致；通过尾块、写缓冲转发和 I/O 错误测试；prefill 与 decode 的内存峰值符合配置。

## 阶段 2：交错布局

实现 token 级 K/V 交错，明确单次请求字节数；适配 stride/view 和融合投影。固定选择集合，比较独立 K/V 与交错布局的读请求数、传输、转换成本和 attention 延迟。

代码产物：交错读写、融合投影、stride attention、固定选择回放、分项观测与四组布局消融工具。

代码验收：布局 round-trip、量化行保留、Normal/NeoX RoPE、GQA、两种布局的 attention 数值、固定选择和错误处理检查通过。

实验产物：模型级数值对照、布局消融结果、设备传输与内存记录。实验验收：固定输入和选择集合，模型输出符合数值容差，I/O、投影、转换与 attention 成本可分项测量。

完成记录见 [阶段 2](phases/PHASE_2.md)。

## 阶段 3：历史预取与纠错

接入 liburing。按同层上一轮选择预取 selected，确定性预取 init/local；正式选择后补读并复用已释放槽位。分别实现 DRAM/VRAM 生命周期、取消完成确认和 generation 校验。

代码产物：liburing 读取、历史预取器、版本化 DRAM/VRAM 槽位、完成与取消协议、物理槽位 mask、命中与补读轨迹和对照工具。

代码验收：冷启动、命中、全错 selected、乱序通知、取消竞态和容量边界通过；合成数据的实际选择与同步参考一致，attention 输出符合容差，缓冲区通过内存检查。

实验验收：动态/固定选择核查、模型级 logits、CUDA 复制与执行、任务准确率和资源峰值。完成记录见 [阶段 3](phases/PHASE_3.md)。

## 阶段 4：微任务调度

建立任务 DAG，拆分 SSD、H2D、D2H 和 GPU 完成事件。依次打开 miss load/KV projection 重叠、下一层读取和同步点复用，复用单层工作集。

代码产物：DAG/LST 调度、有界协调与独立 I/O 线程、独立复制后端、Q 依赖与请求边界复用、事件轨迹和 `naive/overlap/overlap+sync_reuse` 对照工具。

代码验收：数据依赖与槽位释放时序正确；三种模式的合成图输出通过串行对照；实际轨迹的同步次数、临界路径和重叠可独立重算。完成记录见 [阶段 4](phases/PHASE_4.md)。

实验验收：固定模型输入与选择，核查 logits、CUDA 复制/计算并发、同步次数、端到端延迟和资源峰值。

## 阶段 5：论文实验

按 [实验矩阵](TEST_PLAN.md) 运行上下文长度、准确率、消融、块大小、预算和 SSD 干扰实验。先完成 CUDA 主路线，再扩展 SYCL、FlexGen 适配和能耗测量。

代码产物：精确 prompt token、CUDA 实验矩阵、独立重复与续跑、cgroup/fio 条件、阶段计数、准确率接口、官方评分、统计与绘图。完成记录见 [阶段 5](phases/PHASE_5.md)。

代码验收：合成输入、假进程续跑、缺失/失败记录、计时口径、评分对齐、来源核查与绘图导出检查通过。

实验产物：CUDA 原始数据、准确率表、图表和复现报告。实验验收：每张图可追溯到配置、代码与输入，报告重复运行波动、内存分项及硬件设置。

## 阶段 6：策略研究

根据阶段 5 的瓶颈选择自适应预取、重要性/读成本预算分配或缓冲区调度，保留固定历史预取作为对照。每项策略独立消融，报告延迟、准确率、内存和写放大。
