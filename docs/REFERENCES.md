# 论文依据与复现边界

主要来源：Xinrui Zheng, Dongliang Wei, Jianxiang Gao, Yixin Song, Zeyu Mi, Haibo Chen, *SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs*, 24th USENIX Conference on File and Storage Technologies (FAST '26), pp. 67–82. [会议页面](https://www.usenix.org/conference/fast26/presentation/zheng) · [论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。本设计根据用户提供的 17 页论文核对。

| 论文位置 | 已确认的内容 | 在本项目中的处理 |
| --- | --- | --- |
| §3.1、图 3–4 | SSD 保存完整 KV；init/local/selected 三类块；代表向量与 Q 相似度选 top-k | 先实现确定性选择及块身份映射 |
| §4、图 7–8 | token 级 K/V 交错使传输单元增大，不增加块内 token 数；K/V 投影权重可离线合并 | 先做布局与访问，再适配融合投影 |
| §5、图 9–10 | 每层上一轮选择历史预测 selected 块；正式选择后补读遗漏块并覆盖无用预取槽 | 纠错是正确性条件，预取只影响性能 |
| §6、图 11、算法 1 | 微任务 DAG、关键路径优先、同步点复用，需防止并行 I/O 覆盖仍在使用的数据 | 以缓冲区状态机和事件依赖实现 |
| §7 | llama.cpp + liburing；实验设块大小 32 token；短于 4k 的预算为输入长度 25%，否则 1k；init/local 合占预算一半 | 参数作为论文复现配置，不写死为通用最优值 |
| §8 | CUDA 与 SYCL、多个模型、速度/内存/准确率/消融评测 | 先做 CUDA，随后评估其他后端 |

**待核实的实现细节**：论文 §3.1 指出代表向量沿用 InfLLM，但并未给出足以直接重写全部边界条件的伪代码；模型适配时应核对其引用实现、RoPE 与分组注意力语义。论文的具体模型版本、量化配置、I/O 队列设置和设备性能也应在实验前锁定。设计文档中的接口和状态机是我们的工程方案，不能当作论文作者的原始代码。

辅助一手来源：Chaojun Xiao 等，*InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory*，[论文 PDF](https://arxiv.org/pdf/2402.04617)，§3.2 给出 representative score、每块多个代表 token 与查询相关度的计算。SolidAttention 引用其代表选择思想，但没有在正文中规定本项目所需的全部参数；不能直接把 InfLLM 的默认参数当作 SolidAttention 的实验配置。
