# 阶段 1：同步 SSD 与稀疏参考

代码完成于 2026-11-17，阶段提交 `7fbff91`。

## 目标与产物

在固定 llama.cpp 上接入独立 KV 存储和 attention，实现 Offload、InDRAM+Sparse 与 Offload+Sparse。完整接口、文件格式及选块规则见 [SSD_PATH](../SSD_PATH.md)。

| 模块 | 代码入口 | 完成内容 |
| --- | --- | --- |
| KV 存储 | `src/kv_store.cpp` | FP16、固定块与对齐、独立 K/V plane、每层写缓冲、最新数据转发 |
| 同步文件 | `FileKvStore`、`RandomAccessFile` | 短传输重试、文件独占创建、I/O 失败传播、临时文件清理 |
| 稀疏参考 | `src/attention.cpp` | 后续窗口构造代表、GQA 组内平均 Q、动态选块、init/local 去重与预算 |
| 会话 | `src/session.cpp` | 有界分批 prefill、历史块在线 softmax、FP16 decode 工作集与 mask |
| 计算图接入 | `src/llama_adapter.cpp` | 原生 RoPE、输出投影及 FFN；CPU 流式参考与 GGML Flash Attention decode |
| 上游生成 | `scripts/generate_adapter.py`、`cmake/Adapter.cmake` | 在构建目录生成接入文件，固定 submodule 源码 |
| 对照工具 | `scripts/prepare_replay.py`、`scripts/compare_logits.py` | 固定实际输入 token，保存与比较 FP32 logits |

## 数据流

prefill 按上游 microbatch 生产 Q/K/V，逐 token 追加 FP16 KV；attention 按历史块读取并在线累计 softmax。代表构造器在收齐后续 local 窗口后定稿。

decode 先追加当前 KV，再按当前 Q 选块；从 SSD 已提交数据和写缓冲读取最新内容，构造预算内的 FP16 工作集及各 query head 的 mask。GGML Flash Attention 消费该工作集，计算完成后向运行器传递错误状态。

配置入口为 `indram-reference.json`、`offload.json`、`indram-sparse.json` 和 `offload-sparse.json`。稀疏配置默认块大小 32；预算和 GQA/位置规则记录在结果文件中。

后续补齐的 CUDA 块评分、预算 top-k、代表增量上传和后端选择见 [GPU 选择完成记录](../implementation/GPU_SELECTION.md)。CPU 代表构造与参考选择保留，`selection_backend` 控制评分后端。

CUDA 后续窗口构造、半完成状态接管、完整块提交和三种调度图对照见 [GPU 代表构造完成记录](../implementation/GPU_REPRESENTATIVES.md)。CUDA 后端从首个 attention 图初始化，代表更新覆盖 prefill 与 decode。

## 检查

阶段完成时通过 CUDA 编译、11 项 CTest、10 项 Python 检查及源码指纹核对。证据覆盖 FP16 舍入、布局与尾块、写缓冲转发、I/O 错误、独立 softmax 公式、GQA、microbatch 一致性和 CPU Flash Attention。详细记录见 [CHECKS](../CHECKS.md)。

## 实验环境核查

按 [SSD_PATH 的固定 token 对照步骤](../SSD_PATH.md#固定-token-数值对照)，依次核查原版与内存 dense、内存 dense 与 SSD dense、内存 sparse 与 SSD sparse。保存 logits 容差、选择集合、内存分项与设备资源峰值。
