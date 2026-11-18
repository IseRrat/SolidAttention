# SolidAttention-Lab

复现 FAST 2026 论文 **SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs**，在 llama.cpp 上实现 SSD KV cache、动态稀疏注意力、历史预取和计算/I/O 重叠。

系统采用单请求 decode、INT4 模型权重与 FP16 KV cache。CUDA 路径覆盖 GPU 代表构造、动态选块、有界 prefill、历史预取及当前 KV 安装与异步归档。SYCL 路径已接入共享 KV、GPU 代表构造、动态选块和逐 head 稀疏 decode。后续设备算子及跨层资源核查见 [论文实现核查](docs/PAPER_IMPLEMENTATION_AUDIT.md)。

## 开发入口

1. [开始复现](docs/START_HERE.md)：获取上游、编译、记录基线和定位接口。
2. [实施计划](docs/ROADMAP.md)：开发顺序、阶段产物和验收条件。
3. [系统设计](docs/DESIGN.md)：块布局、内存预算、prefill、预取和调度。
4. [实施决策](docs/DECISIONS.md)：已选方案及参数锁定任务。
5. [测试与评测](docs/TEST_PLAN.md)：正确性测试、实验矩阵和统计口径。
6. [论文依据](docs/REFERENCES.md)：论文参数、图表与实验对应关系。
7. [构建与检查](docs/DEVELOPMENT.md)：代码构建、单元检查和实验环境运行入口。
8. [同步 SSD 路径](docs/SSD_PATH.md)：内存/SSD 模式、选块规则、文件格式和固定 token 对照。
9. [阶段完成记录](docs/phases/README.md)：各阶段完成后的代码入口、检查证据与核查步骤。
10. [历史预取与纠错](docs/phases/PHASE_3.md)：liburing、DRAM/VRAM 租约、纠错轨迹和同步参考核查。
11. [微任务调度](docs/phases/PHASE_4.md)：计算/I/O 重叠、同步复用、三组固定集合对照与任务轨迹核查。
12. [论文实验代码](docs/phases/PHASE_5.md)：精确输入、实验矩阵、续跑与汇总、准确率接口、官方评分和可追溯绘图。
13. [工程审查记录](docs/ENGINEERING_REVIEW.md)：异步错误清理、子进程管理、实验完成核查和 scorer 来源修复。
14. [论文实现核查](docs/PAPER_IMPLEMENTATION_AUDIT.md)：论文代码规模、当前执行路径、核心实现差距与补齐顺序。
15. [CUDA 块评分与动态选块](docs/implementation/GPU_SELECTION.md)：新增 GPU 内核、代表发布、预算 top-k、实际后端与内存记录、检查证据。
16. [CUDA 窗口代表构造](docs/implementation/GPU_REPRESENTATIVES.md)：后续窗口评分、环形状态、块内代表、CPU 状态接管及图核查。
17. [有界 CUDA 分块 prefill](docs/implementation/GPU_PREFILL.md)：因果 attention、在线 softmax、分块读取、源生命周期、池释放和数值核查。

18. [独立 I/O 与协调线程](docs/implementation/IO_WORKERS.md)：有界提交、完成通知、取消排空、物理核预留和实际线程记录。

19. [selected 优先读取与逐块上传](docs/implementation/BLOCK_READ_PIPELINE.md)：跨 head 提交顺序、正式请求提升、SSD/H2D 流水和完成汇合。

20. [历史块事件 DAG 与资源分派](docs/implementation/BLOCK_EVENT_SCHEDULER.md)：块级读取/H2D 依赖、关键路径/LST、实际队列顺序与两级轨迹核查。

21. [CUDA attention 完成事件](docs/implementation/NATIVE_COMPLETION.md)：图捕获与重放通知、槽位释放、后继上传、正式选择接管和设备计时。

22. [Decode 设备 Q/K 与 FP16 KV](docs/implementation/DEVICE_INPUTS.md)：设备输入借用、图保活依赖、三种调度共用路径、FP16 consolidator 和核查入口。

23. [Prefill 设备输入与输出](docs/implementation/PREFILL_DEVICE_INPUTS.md)：批量代表、历史/当前 KV 分区、结果视图、缓冲区生命周期和端到端交接。

24. [当前 KV 设备复制](docs/implementation/CURRENT_DEVICE_KV.md)：当前行 D2D、尾块前缀保留、版本更新、复制描述符生命周期和字节核查。

25. [当前 KV 归档](docs/implementation/KV_ARCHIVE.md)：attention 与主机归档分流、跨层读取与槽位释放。

26. [SYCL 共享 KV](docs/implementation/SYCL_SHARED_KV.md)：集成 GPU 的共享 bank、逐 head mask decode 和构建入口。
27. [SYCL 动态选块](docs/implementation/SYCL_SELECTION.md)：设备代表构造、逐 head 相关度和 token 预算归约。

## 系统机制

- KV consolidator：按 token 交错存储 K/V，以一次传输读取成对数据。
- Speculative prefetcher：确定性预取 init/local 块，按同层上一 decode 轮的选择历史预取 selected 块，正式选择后补读遗漏块。
- SSD-aware scheduler：将投影、选择、传输、attention 和回写拆成微任务，按依赖重叠执行并复用同步点。

## 复现目标

在固定模型、输入和稀疏选择策略下，测量各项机制对 decode 延迟、I/O 阻塞、KV 驻留内存和任务准确率的影响。完成 CUDA 路线后扩展 SYCL，并根据瓶颈数据开展自适应预取与预算分配实验。

论文：Xinrui Zheng 等，FAST '26，pp. 67-82。[USENIX 页面](https://www.usenix.org/conference/fast26/presentation/zheng) · [论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。
