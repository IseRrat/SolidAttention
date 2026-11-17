# 论文实现核查

核查日期：2026-11-17。代码基点：`afe4773bfcf6c92b2a80210e391f2356b0a961e6`。依据：SolidAttention §3.1、§4、§6、§7 和图 5、图 11、算法 1；[论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)，印刷页 70–74。

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

## 3. 论文路径与后续核心代码

| 核查项 | 当前执行路径 | 后续代码与核查要求 |
| --- | --- | --- |
| GPU 算子覆盖，§3.1、§4、§7 | Q/KV 经主机前沿接入 [CUDA 代表构造](implementation/GPU_REPRESENTATIVES.md) 和 [评分/top-k](implementation/GPU_SELECTION.md)；CPU 维护候选与核查镜像；decode attention 复用 GGML GPU 算子 | 补齐直接消费设备 Q 与交错 KV 更新；逐算子记录输入输出、dtype、stride、设备和依赖 |
| 长输入 prefill | [CUDA 分块 prefill](implementation/GPU_PREFILL.md) 在固定 Q/KV 池上累计因果 attention；代表使用 GPU 窗口构造；CPU 流式路径用于参考 | 继续接入设备 Q/K，核查模型级误差、峰值与性能 |
| 算法 1 的调度范围，§6.1 | [scheduler.cpp](../src/scheduler.cpp) 提供关键路径/LST；[ggml_prefetch.cpp](../src/ggml_prefetch.cpp) 沿固定图与回调推进，每层聚合为 12 项任务；`ready()` 在回写/下一层预取之间选择 | 建立块级预测读、纠错读、H2D、D2H、回写的资源就绪队列，以真实完成事件解锁后继，按关键路径/LST 分派；合成执行轨迹证明优先级影响提交次序 |
| selected 优先，§6.1 | [prefetch.cpp](../src/prefetch.cpp) 的 `begin()` 按 head 加入 init、local、历史 selected，再按插入次序启动读取 | 给块请求保留类别、依赖和估算成本，让关键纠错依赖的 selected 请求获得优先级；用受限队列检查实际发出顺序和饥饿边界 |
| I/O 与协调分工，§7 | 一个常驻 `TaskWorker` 执行纠错读、H2D 提交、满缓冲回写和下一层预取，liburing 由调用方推进 | 拆分专用 I/O 与 SSD-GPU 协调线程，配置独立 CPU 绑定；用有界队列传递请求及完成事件，核查关停、反压和租约释放 |
| 同步与时间观测，§6.2 | GGML 事件支持独立 H2D、批量确认和同步复用；DAG 的 GPU 区间采用主机观测边界，投影包含下载阶段 | 补齐设备事件记录和任务级完成关联，分别记录主机区间、设备执行与传输；由事件关系核查共享完成点和槽位复用 |
| SYCL，§1、§8.1 | 主工程提供 CUDA 构建入口，GPU 路径依赖 GGML 通用接口 | 提供 SYCL 构建、具体算子与事件适配、统一内存路径和对应合成检查 |

prefill 的 GPU 分块方案属于长上下文工程补齐；论文 §7 对实现的 GPU kernel 分配仅给出合计。选择器的设备归属、算子拆分和后端差异需结合论文及可获取的作者实现进一步锁定。

## 4. 后续开发顺序与交付

1. 在 CUDA 代表构造/评分/top-k、分块 prefill 基础上补齐直接消费设备 Q 和交错 KV 更新的数据流。
2. 把聚合任务扩展为块级事件任务，连接资源就绪队列与关键路径/LST 分派，落实 selected 优先。
3. 拆分 I/O 与协调线程、配置 CPU 绑定，完善异步传输与设备完成观测。
4. 复核 CUDA 全路径，补齐 SYCL 与统一内存接入。
5. 每项完成代码与检查后更新所属阶段记录，再进入模型及设备评测。

本机核查使用编译、合成张量、临时文件和错误注入。模型推理、性能、准确率与真实设备实验在实验环境执行。
