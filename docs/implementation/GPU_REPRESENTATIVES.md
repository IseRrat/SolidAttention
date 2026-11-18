# CUDA 窗口代表构造

代码与本机检查完成于 2026-11-18。交付为 GQA 均值、后续窗口评分、块内代表选择三个 CUDA 内核，以及 CPU 状态接管、定稿发布和运行器接入。

## 1. 计算规则与依据

SolidAttention §3.1 沿用 InfLLM 的代表提取。[InfLLM §3.2](https://arxiv.org/html/2402.04617v2) 用 token `m` 后续 `L` 个 Q 与其 K 的点积平均值作为代表分数：`r(m) = sum(Q(m+j) · K(m), j=1..L) / L`。项目的 GQA 扩展先按 query head 组求均值；原生 RoPE、FP16 K 和并列按位置升序的规则继续与 CPU 参考一致。

位置 `t` 的 Q 先更新全部待定 token 的分数，再定稿 `t-L`；当前 K 在 Q 更新后进入环形槽位。块 `b` 的完整代表在 `t=b*B+B-1+L` 定稿。每个 head 保留分数最高的 `R` 个 K。

## 2. 代码入口与执行路径

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 协议与快照 | [block_ranker.hpp](../../include/solid/block_ranker.hpp) | pending K/分数、半完成块代表、完整块结果、Q/K 状态交接 |
| 共用数学 | [representative_math.hpp](../../include/solid/representative_math.hpp) | GQA 均值、按维度累计、按分数与位置插入代表；CPU/CUDA 共用 |
| 快照核查 | [representative_builder.cpp](../../src/representative_builder.cpp) | 长度、窗口、形状、位置、排序、padding 和有限数值 |
| CUDA 执行 | [block_ranker.cu](../../src/cuda/block_ranker.cu) | 三个构造内核、固定窗口设备状态、完整块直接写入选块 bank |
| 索引接入 | [attention.cpp](../../src/attention.cpp) | 完整初始化后移交 CPU pending 状态，校验完成边界及返回代表 |
| 图接入与结果 | [llama_adapter.cpp](../../src/llama_adapter.cpp)、[baseline.cpp](../../src/baseline.cpp) | 首个 attention 图选择后端，记录构造身份、阶段计数和分配 |

```text
Q 经主机选择前沿到达
  → CUDA GQA 均值 → 更新环形窗口分数 → 成熟 token 插入块内代表
                                      ↓ 完整块定稿
                         直接提交 GPU 选块 bank，并返回主机镜像
                                      ↓
                         CUDA 块评分/top-k → 历史预取纠错

当前 K 到达 → 写入 position % L 槽位，清零新 token 分数
```

CUDA 数学使用 FP64 并关闭 FMA 收缩，保持 CPU 的 query、维度累计顺序。构造、选块和元数据传输共用自有 stream，每次调用在输入源和结果归还前完成同步。GPU bank 内的定稿代表直接供评分内核使用；主机镜像供索引核查与 CPU 数值对照。

## 3. 窗口、接管与生命周期

每层保存 `L` 个 FP16 K 及对应分数，按 `position % L` 寻址；块内保存每个 head 的 `R` 个候选。Q 更新完成后才能复用当前 K 的槽位。

长度为 `N` 的 CPU 状态在 query 边界接管。令 `M=max(N-L,0)`：

- 上传已有 `floor(M/B)` 个完整块的代表。
- 移交位置 `M..N-1` 的 pending K 和累计分数。
- 移交块 `floor(M/B)` 已定稿部分的代表、分数和位置。

各层独立初始化，全部层初始化成功后释放 CPU pending 工作集。初始化中途异常保留 CPU 索引所有权。构造后端返回完整块时，索引检查期望块、完成时刻、有限值、唯一位置和排序，再发布 ready 状态。

`selection_backend=auto/cpu/cuda` 继续控制选择路径。CUDA 路径在首个 attention 图初始化，代表构造覆盖 prefill 和 decode；`cpu` 使用原有参考。固定选择回放沿用 CPU 路径。工厂参数 `local=0` 提供仅评分后端，用于独立算子核查。

CUDA 预取 decode 已接入设备 Q 和交错 FP16 K，三种调度共用借用接口。主机 prefill 接口继续提供 Q/K；输入类型、图保活和上传计数见 [设备输入](DEVICE_INPUTS.md)。

## 4. 分配与记录

每层构造工作区字节数为：

```text
L * Hkv * (2*D + 8) + Hkv * R * (2*D + 16)
```

共享工作区为 `Hkv*D*8 + Hkv*R*16`，包含均值与完整块输出元数据。代表 bank 沿用 [GPU 选择](GPU_SELECTION.md) 的分配。初始化后，设备分配跨 token 复用。

结果保存 `attention.representative_backend=cuda_window_fp64`、`representative_builder_queries`、`representative_builder_appends`、`representative_builder_blocks`；计数同时进入 `attention_phases`。

`memory.custom_buffers.selection_device_bytes` 包含全部选择/构造分配。`memory.selection_device_breakdown` 将总量分为 `representative_builder_bytes` 和 `ranking_and_bank_bytes`，`kv_system_tracked_bytes` 计入一次总量。

## 5. 检查证据

CUDA 编译与链接、35 项 CTest、56 项 Linux Python 检查和源码指纹通过；核心库 8 项 CTest 的地址、泄漏和未定义行为检查通过。

[representative_builder_cases.hpp](../../tests/representative_builder_cases.hpp) 在 CPU 上执行 CUDA 共用的逐项数学，和原有 deque、标量累计及完整排序路径对照。覆盖起点/中途接管、窗口绕回、块边界、GQA、多代表、并列、非 warp 整数维度、尾块、完整 local 窗口和固定设备容量。

[representative_builder_test.cpp](../../tests/representative_builder_test.cpp) 另覆盖非法快照、半次 query 接管拒绝、第二层初始化失败后重试、错误完成时刻、块号/形状/位置/分数错误，以及构造失败后的会话终止。

三种调度模式的实际 CPU GGML 图分别接入构造后端模拟器，核对 prefill/decode、预取、任意槽位 Flash Attention 数值及同步复用。每例检查 24 次 Q 更新、24 次 K 追加和 10 个完整代表块。

设备数值检查已加入 [cuda_ranker_test.cpp](../../tests/cuda_ranker_test.cpp) 并完成编译，实验环境通过 `solid-cuda-ranker-test --run-device-check 0` 执行。本机数学检查使用 CPU 合成数据。

## 6. 后续核心路径

有界 attention 已接入 [GPU 分块 prefill](GPU_PREFILL.md)。窗口构造按 token 调用，后续继续接入设备 Q/K 和批处理更新。
