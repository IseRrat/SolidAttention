# SolidAttention-Lab

复现 FAST 2026 论文 **SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs**，在 llama.cpp 上实现 SSD KV cache、动态稀疏注意力、历史预取和计算/I/O 重叠。

首条路线采用 Linux + CUDA，单请求 decode，INT4 模型权重和 FP16 KV cache。开发从获取并固定 llama.cpp 版本、运行原版基线开始。

## 开发入口

1. [开始复现](docs/START_HERE.md)：获取上游、编译、记录基线和定位接口。
2. [实施计划](docs/ROADMAP.md)：开发顺序、阶段产物和验收条件。
3. [系统设计](docs/DESIGN.md)：块布局、内存预算、prefill、预取和调度。
4. [实施决策](docs/DECISIONS.md)：已选方案及参数锁定任务。
5. [测试与评测](docs/TEST_PLAN.md)：正确性测试、实验矩阵和统计口径。
6. [论文依据](docs/REFERENCES.md)：论文参数、图表与实验对应关系。
7. [构建与检查](docs/DEVELOPMENT.md)：代码构建、单元检查和实验环境运行入口。

## 系统机制

- KV consolidator：按 token 交错存储 K/V，以一次传输读取成对数据。
- Speculative prefetcher：确定性预取 init/local 块，按同层上一 decode 轮的选择历史预取 selected 块，正式选择后补读遗漏块。
- SSD-aware scheduler：将投影、选择、传输、attention 和回写拆成微任务，按依赖重叠执行并复用同步点。

## 复现目标

在固定模型、输入和稀疏选择策略下，测量各项机制对 decode 延迟、I/O 阻塞、KV 驻留内存和任务准确率的影响。完成 CUDA 路线后扩展 SYCL，并根据瓶颈数据开展自适应预取与预算分配实验。

论文：Xinrui Zheng 等，FAST '26，pp. 67-82。[USENIX 页面](https://www.usenix.org/conference/fast26/presentation/zheng) · [论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。
