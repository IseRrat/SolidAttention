# CUDA 块评分与动态选块

代码与本机检查完成于 2026-11-17。本轮交付为 CUDA 块相关度评分、预算约束 top-k、代表增量上传及运行器接入。论文依据为 §3.1 图 4 的 Q/代表向量相关度和 top-k；GQA 聚合与多个代表的处理沿用项目的 CPU 参考。

2026-11-18 接入 [CUDA 窗口代表构造](GPU_REPRESENTATIVES.md)，在首个 attention 图初始化后端，覆盖 prefill 与 decode 的代表更新。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 后端协议 | [block_ranker.hpp](../../include/solid/block_ranker.hpp) | 定稿代表发布、候选块与有效 token 数、分 head 排名、内存和调用统计 |
| CUDA 内核 | [block_ranker.cu](../../src/cuda/block_ranker.cu) | 相关度评分、并行归约 top-k、固定容量设备工作区、独立 stream 与设备恢复 |
| 排序规则 | [ranking_math.hpp](../../include/solid/ranking_math.hpp) | 分数降序、逻辑块编号升序、空候选处理 |
| 代表与选择 | [attention.cpp](../../src/attention.cpp) | 初始化发布、增量发布、候选去重、剩余 token 预算、后端输出核查 |
| 会话与图接入 | [session.cpp](../../src/session.cpp)、[llama_adapter.cpp](../../src/llama_adapter.cpp) | 首个 attention 图配置实际后端，串行和 Q-only 选择调用共用入口，错误使会话结束 |
| 结果与构建 | [baseline.cpp](../../src/baseline.cpp)、[Adapter.cmake](../../cmake/Adapter.cmake) | 后端身份、内存和阶段计数；CUDA 库编译与链接 |

## 2. 计算规则

定稿代表以 FP16 存储，设备 bank 布局为 `[logical_block, kv_head, representative, D]`。Q 使用 FP32，布局为 `[query_head, D]`。对于 KV head `h`，相关度为该组 query heads 与所有代表的点积之和，再按组大小平均。

评分内核的累加顺序为代表、query head、维度，与 CPU `RepresentativeIndex::score()` 一致，使用 FP64 累加并关闭 FMA 收缩。该精度方案用于建立选择一致性基础。

top-k 每个 KV head 使用一个 256 线程 CTA，按相关度归约出下一个块。同分时取较小的逻辑块编号。每次选出后扣除该块中落在 init/local 之外的 token 数，达到剩余预算时结束；最后一块由主机选择器保留预算所需的位置。候选不足时返回全部可用块。

候选块只包含完整定稿且提供额外历史 token 的块。各 head 共享候选身份与 token 成本，相关度和排名分别计算。物理缓存槽位由原有预取器和设备租约管理。

## 3. 代表发布与执行顺序

```text
CPU 参考/ CUDA 窗口构造 → 完整块定稿 → 提交该块全部 head 的代表
                                       ↓
当前 Q 就绪 → 更新窗口代表、形成候选集合
          → CUDA 评分 → CUDA top-k → 返回逻辑块 ID
          → 主机核对候选、唯一性与覆盖量 → 预取纠错/attention
```

首次附加后端时，按层上传已有的定稿前缀。后续每个完整块发布一次；设备端要求发布范围紧接已有前缀，代表发布后保持固定。CUDA 窗口构造直接写入选块 bank；CPU 构造路径通过发布接口上传。

CUDA 后端使用独立 non-blocking stream。代表上传完成后归还主机 payload；一次 rank 在输入上传、两个内核和 ID 下载完成后返回。每个入口保存并恢复调用线程原有 CUDA device。CUDA 失败会完成流清理，再沿既有 attention 错误路径传播。

CUDA 动态选块的预取 decode 直接消费设备 Q，三种调度与保活依赖见 [设备输入](DEVICE_INPUTS.md)。[Prefill 设备接入](PREFILL_DEVICE_INPUTS.md) 使用批量 Q/K 更新代表及设备结果；主机接口继续支持参考和独立算子核查。完整执行范围见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。

## 4. 配置与结果

在已有稀疏 attention 配置中设置：

```json
"selection_backend": "auto"
```

| 值 | 执行策略 |
| --- | --- |
| `auto`，默认 | 首个 attention 后端为 CUDA 时启用 CUDA 代表构造与评分；其余使用 CPU |
| `cpu` | CPU 评分与排名，用于参考对照 |
| `cuda` | 要求 CUDA 构建、CUDA decode 后端和动态稀疏选择，条件在配置和接入时核查 |

固定选择回放直接消费已记录的集合。布局和调度消融工具生成回放配置时设置 `selection_backend=cpu`；动态预取对照保留配置中的选块后端。实验矩阵及准确率入口也保留模板的后端选择。

结果字段：

| 字段 | 内容 |
| --- | --- |
| `attention.selection_backend_requested` | 请求的配置值 |
| `attention.selection_backend` | 实际 `cpu`、`cuda_fp64_topk` 或 `replay` |
| `attention.selection_rank_calls` | 有候选且预算大于零的 CUDA rank 调用数；每次包含两个内核 |
| `attention.selection_scored_blocks` | 各 KV head 实际评分的候选块数量之和 |
| `attention.selection_published_blocks` | 按层计数的已发布逻辑块数量 |
| `memory.custom_buffers.selection_device_bytes` | 自有 CUDA bank、Q、候选、分数、ID 输出及状态分配 |
| `memory.custom_buffers.selection_host_bytes` | 常驻主机 ID 输出和层元数据 |

调用与发布计数同时进入 `attention_phases`。设备与常驻主机分配进入 `kv_system_tracked_bytes`。bank 按已使用层懒分配，每层容量为 `blocks_per_head * kv_heads * representatives * D * 2` 字节；scratch 固定容量并跨调用复用。首次发布按层构造临时 payload，其上限为一层 bank 的字节数。

源码指纹包含 `.cu` 文件，启动时核对内核源码哈希。CUDA 架构设置继承锁定上游的 `ggml-cuda` 目标。

## 5. 已执行的检查

- CUDA 12.6、`sm_89` 编译与运行器链接；31 项 CTest、56 项 Linux Python 检查、运行器源码指纹通过。
- 核心库 7 项 CTest 通过 AddressSanitizer、UndefinedBehaviorSanitizer 和泄漏检查。
- [ranker_test.cpp](../../tests/ranker_test.cpp) 使用独立主机后端，保存实际发布 payload，用完整排序计算期望结果。覆盖两层、GQA、1/2/4 个代表、零分并列、早期与延迟附加、增量定稿、尾块和局部预算。
- 与 CUDA 共享的归约比较规则在 CPU 上通过 CTA tournament 对照，候选数量覆盖 1、31、256、257、4097。
- 错误注入覆盖 head 数量、重复/未知/额外块、覆盖不足和后端异常；会话拒绝继续使用失败状态。
- 配置解析、CUDA 构建要求、`.cu` 来源变更拒绝、实验后端设置保留及既有回放工具检查通过。

以上数学执行使用 CPU 合成数据。本机执行范围为编译和单元检查。

## 6. 实验环境设备检查入口

[cuda_ranker_test.cpp](../../tests/cuda_ranker_test.cpp) 已编译、链接，设备执行通过显式参数启动。该程序单独保留在实验环境入口，`check.sh` 和 CTest 执行 CPU 检查。

```bash
build/release/bin/solid-cuda-ranker-test --run-device-check 0
```

程序用合成 FP16 代表和 FP32 Q 对照独立完整排序，覆盖 0/1/31/255/256/257/513 个候选、GQA 组大小 3、多代表、非 warp 整数维度、同分、大小幅度、不同预算、分段上传、两层及设备分配稳定性。模型级选块/logits 对照随后使用既有固定 token 工具执行，保存配置、后端身份和来源哈希。
