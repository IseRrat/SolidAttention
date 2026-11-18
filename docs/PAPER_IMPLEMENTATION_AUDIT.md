# 论文实现核查

核查日期：2026-11-18。代码基点：`afe4773bfcf6c92b2a80210e391f2356b0a961e6`。依据：SolidAttention §3.1、§4、§6、§7 和图 5、图 11、算法 1；[论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)，印刷页 70–74。

## 1. 实现规模与统计范围

论文 §7 报告约 25k 行代码，其中 GPU kernels 约 12k、llama.cpp adapters 约 1k。§1 说明实现覆盖 CUDA 与 SYCL。论文按模块给出的规模可作为实现范围的核查线索。

基点 `afe4773` 的统计使用已跟踪文件的物理行数，包含注释与空行，排除上游 submodule、生成文件和构建产物。后续新增代码见各项补齐记录。

| 基点代码类别 | 行数 |
| --- | ---: |
| 运行时 C++ 源码与头文件 | 4,423 |
| Python/shell 工具与评测入口 | 2,361 |
| 测试 | 2,724 |
| 构建支持 | 245 |
| 合计 | 9,753 |
| 新增 CUDA/SYCL GPU kernel 源码 | 0 |

基点的 GPU 运算通过上游 GGML 投影、RoPE 和 Flash Attention 接口执行。§4 明确允许通过 stride 访问复用 attention kernel。论文给出 GPU kernel 总量，具体算子和双后端的代码分配需要逐项查证。后续逐算子登记复用、适配和新增代码，以执行路径和数据流作为交付依据。

## 2. 已接入的执行路径

- KV 文件与每层聚合写缓冲；CPU 流式 attention 参考；代表构造与稀疏选块。
- 合并 K/V 投影权重、交错 KV 文件与设备 bank、strided GGML Flash Attention decode。
- liburing 异步读、历史预测与纠错、独立 DRAM/VRAM 租约和 generation 校验。
- Q-only CPU 选择前沿、历史补读与 KV 投影重叠、独立 H2D 后端、跨层完成证明复用。
- 实验矩阵、固定 token/选择回放、准确率接口、来源核查与绘图工具。
- CUDA 块相关度评分与预算 top-k、定稿代表增量上传、后端身份与设备内存记录，完成记录见 [GPU 选择](implementation/GPU_SELECTION.md)。
- CUDA 后续窗口代表构造、环形评分工作区、CPU 状态接管与定稿块直接发布，完成记录见 [GPU 代表构造](implementation/GPU_REPRESENTATIVES.md)。
- 有界 CUDA 分块 prefill、在线 softmax、源租约及池释放，完成记录见 [GPU prefill](implementation/GPU_PREFILL.md)。

- 独立 I/O 与协调线程、有界队列、物理核预留与取消排空，完成记录见 [I/O 线程](implementation/IO_WORKERS.md)。

- selected 跨 head 优先、正式请求提升与逐块 SSD/H2D 流水，完成记录见 [块读取流水](implementation/BLOCK_READ_PIPELINE.md)。

- 历史工作集的块级 READ/H2D DAG、关键路径/LST 与完成票据，完成记录见 [块事件调度](implementation/BLOCK_EVENT_SCHEDULER.md)。

- CUDA attention 设备事件、图重放通知、后继 H2D、Q 边界交接及设备计时，完成记录见 [原生完成事件](implementation/NATIVE_COMPLETION.md)。

## 3. 论文路径与后续核心代码

| 核查项 | 当前执行路径 | 后续代码与核查要求 |
| --- | --- | --- |
| GPU 算子覆盖，§3.1、§4、§7 | Q/KV 经主机前沿接入 [CUDA 代表构造](implementation/GPU_REPRESENTATIVES.md) 和 [评分/top-k](implementation/GPU_SELECTION.md)；CPU 维护候选与核查镜像；decode attention 复用 GGML GPU 算子 | 补齐直接消费设备 Q 与交错 KV 更新；逐算子记录输入输出、dtype、stride、设备和依赖 |
| 长输入 prefill | [CUDA 分块 prefill](implementation/GPU_PREFILL.md) 在固定 Q/KV 池上累计因果 attention；代表使用 GPU 窗口构造；CPU 流式路径用于参考 | 继续接入设备 Q/K，核查模型级误差、峰值与性能 |
| 算法 1 的调度范围，§6.1 | [块事件调度](implementation/BLOCK_EVENT_SCHEDULER.md) 在历史工作集建立逐块 READ→H2D→汇合；关键路径/LST 传入独立 I/O 队列并决定就绪复制次序；外层保留 12 项任务 | 继续将预测期读取、当前 KV 的 D2H/回写及跨层槽位释放连接统一资源图；CUDA attention 事件已解锁后继上传 |
| selected 优先，§6.1 | [块读取流水](implementation/BLOCK_READ_PIPELINE.md) 先确定集合，再跨 head 按 selected 优先派发；正式所需请求提升，完成块逐个进入 H2D；受限队列核对实际次序及最终汇合 | 历史正式工作集已连接块级成本、关键路径/LST 和资源分派；继续把预测期成本与跨层依赖纳入图 |
| I/O 与协调分工，§7 | [独立 I/O 所有者](implementation/IO_WORKERS.md) 执行 liburing 操作并自动续填内核队列，协调线程执行纠错、H2D、回写和后继预取；独立物理核与推理核分工 | 设备事件已解锁后继传输，受控完成通知验证正式选择及时接管；继续连接预测期任务图与回写 |
| 同步与时间观测，§6.2 | [CUDA 原生完成事件](implementation/NATIVE_COMPLETION.md) 记录 Flash Attention 设备区间，图重放通知解锁槽位与后继上传；复制完成查询归还源；主机边界继续作为依赖依据 | 继续把 Q/KV 投影与其他传输的设备区间关联具体任务，核查完整执行路径的同步与成本 |
| SYCL，§1、§8.1 | 主工程提供 CUDA 构建入口，GPU 路径依赖 GGML 通用接口 | 提供 SYCL 构建、具体算子与事件适配、统一内存路径和对应合成检查 |

prefill 的 GPU 分块方案属于长上下文工程补齐；论文 §7 对实现的 GPU kernel 分配仅给出合计。选择器的设备归属、算子拆分和后端差异需结合论文及可获取的作者实现进一步锁定。

## 4. 后续开发顺序与交付

1. 在 CUDA 代表构造/评分/top-k、分块 prefill 基础上补齐直接消费设备 Q 和交错 KV 更新的数据流。
2. 在历史 READ/H2D 块图基础上纳入预测期读取、D2H/回写与跨层槽位事件。
3. 在 CUDA attention 事件基础上完善 Q/KV 投影及其他传输的设备观测。
4. 复核 CUDA 全路径，补齐 SYCL 与统一内存接入。
5. 每项完成代码与检查后更新所属阶段记录，再进入模型及设备评测。

本机核查使用编译、合成张量、临时文件和错误注入。模型推理、性能、准确率与真实设备实验在实验环境执行。
