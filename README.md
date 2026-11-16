# SolidAttention-Lab

复现 FAST 2026 论文 **SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs**，建立可验证的 SSD KV cache 推理原型，并在复现基线之上研究新的策略。本仓库是独立的研究复现项目；目前处于设计与实验准备阶段，**尚无可运行的推理实现，也不声称达到论文性能**。

## 核心问题

长上下文推理的完整 KV cache 难以放进内存。动态稀疏注意力可以只计算部分历史块，但如果把每个被选中的小块从 SSD 临时读取，随机 I/O 会拖慢低并发场景。论文通过三项协同设计应对这一矛盾：

1. **KV consolidator**：保持注意力选择的 token 块大小不变，在每个 token 内交错存放 K 和 V，使一次传输同时带回两者。
2. **Speculative prefetcher**：init/local 块确定性预取；selected 块参考同层上一 decode 轮的选择历史预取。正式选择后加载遗漏块，直接复用错误预取的槽位。
3. **SSD-aware scheduler**：将投影、选择、加载、注意力、回写拆成有依赖的微任务，让 I/O 与计算重叠，并复用同步点。

这些描述来自论文；本项目的具体适配、参数和创新方案会以实验结果为准。

## 从哪里开始

- [阶段计划](docs/ROADMAP.md)：每阶段的产物、验证门槛与创新入口。
- [开发入口](docs/START_HERE.md)：开始复现时的阅读与操作顺序。
- [详细设计](docs/DESIGN.md)：模块边界、数据布局、执行时序、正确性约束和接口草案。
- [决策记录](docs/DECISIONS.md)：论文未展开的实现细节和待锁定配置。
- [测试与评测计划](docs/TEST_PLAN.md)：功能、准确率、性能及消融实验。
- [论文与复现边界](docs/REFERENCES.md)：原文链接、关键参数和需要再次确认的细节。

## 复现边界

论文的实现基于 llama.cpp、liburing，并有 CUDA 和 SYCL 后端。第一条实施路线拟选择 **Linux + llama.cpp + CUDA**，先实现可对照的单请求 decode 路径；SYCL、跨平台支持与新的研究策略列为后续阶段。这个选择是项目计划，不代表论文只支持 CUDA。

## 当前状态

| 内容 | 状态 |
| --- | --- |
| 论文机制梳理和实施计划 | 已建立初稿 |
| 代码、模型适配、真实 SSD I/O | 未开始 |
| 正确性、长上下文和性能数据 | 未测量 |

论文：Xinrui Zheng 等，FAST '26，页 67–82，[USENIX 页面](https://www.usenix.org/conference/fast26/presentation/zheng) / [PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。论文报告的最高 3.1× 推理加速和最高 98% KV 内存占用下降是**论文结果**，不是本仓库结果。
