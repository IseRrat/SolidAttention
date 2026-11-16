# 论文依据

Xinrui Zheng, Dongliang Wei, Jianxiang Gao, Yixin Song, Zeyu Mi, Haibo Chen. *SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs*. FAST '26, pp. 67-82。[会议页面](https://www.usenix.org/conference/fast26/presentation/zheng) · [论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。

## 机制与实现参数

| 论文位置 | 内容 | 项目对应 |
| --- | --- | --- |
| §3.1，图 3-4 | 完整 KV 存于 SSD；init/local/selected；Q 与块代表相关度选 top-k | 逻辑块、代表构造、选择器 |
| §3.2，图 5 | 单层 SSD/DRAM/VRAM 数据流及跨层预取 | 缓冲区生命周期与事件轨迹 |
| §4，图 7-8 | token 级 K/V 交错、投影权重合并、strided attention 访问 | 布局及融合投影 |
| §5，图 9-10 | 同层上一轮选择预测 selected，补读遗漏并覆盖误预取槽位 | 历史预取和纠错 |
| §6，图 11、算法 1 | 微任务 DAG、关键路径和 LST 优先级、同步复用 | 调度器 |
| §7 | llama.cpp + liburing；I/O 与协调各使用一个专用 CPU 核；每层 32 KiB 写缓冲 | 上游适配、线程绑定、聚合回写 |
| §7 | 块大小 32；输入短于 4k 的预算为 25%，其余 1k；init/local 合占一半 | 预算配置 |
| §8.1-8.2 | INT4 权重、FP16 KV、batch=1、最大输出 512；端到端实验 DRAM 限额 16 GB | 性能配置 |
| §8.2，图 14 | 单层预算内 KV 缓冲，常驻代表向量；约 62× KV 内存下降 | 内存分项 |
| §8.3，表 1 | OpenCompass，通用任务与 LongBench 准确率 | 质量评测 |
| §8.4-8.6 | 预取、布局、调度、预算、块大小与干扰实验 | TEST_PLAN 实验矩阵 |

CUDA 参照环境：Ultra 9 185H、RTX 4070 Laptop 8 GB、Linux 6.8、CUDA 12.8。SYCL 参照环境：Ultra 7 255H、Arc 140T、Linux 6.11、oneAPI 2025.0。两台机器均为 64 GB DDR5、Samsung 990 PRO 1 TB。

## 适配时锁定的参数

上游提交、GGUF 的具体 INT4 格式、模型 revision、代表数量、init/local 拆分、head 聚合、RoPE、I/O 对齐和队列深度，集中记录在 [实施决策](DECISIONS.md)。

## 辅助来源

- Chaojun Xiao 等，*InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory*，[论文](https://arxiv.org/pdf/2402.04617)。§3.2 用于代表评分、代表 token 和块相关度计算。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：模型执行、KV 管理、计算图及 CUDA/SYCL 后端；[构建文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。
- [liburing](https://github.com/axboe/liburing)：异步 SSD I/O。
- [fio](https://github.com/axboe/fio)：SSD 微基准和干扰负载。
