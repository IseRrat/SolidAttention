# Prefill 设备输入、批量代表与设备输出

代码与合成检查完成于 2026-11-18。在 [有界分块 prefill](GPU_PREFILL.md) 上接入设备 Q、当前批次交错 KV、批量代表更新和设备输出。论文 §2.1 描述并行 prefill，§3.1 给出代表与选择，§4 给出交错 KV；固定 tile 和图输出生命周期沿用本工程的长上下文计算方案。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 设备借用 | [device_input.hpp](../../include/solid/device_input.hpp)、[tiled_prefill.hpp](../../include/solid/tiled_prefill.hpp) | 批量形状、地址范围、输入/输出重叠、借用期和设备输出接口 |
| 批量代表 | [block_ranker.cu](../../src/cuda/block_ranker.cu)、[attention.cpp](../../src/attention.cpp) | 依序排入窗口评分、定稿和 K 追加；完成块镜像校验；批次末完成私有流 |
| 分块 attention | [tiled_prefill.cu](../../src/cuda/tiled_prefill.cu) | 借用设备 Q/KV、两种 tile stride、FP32 在线 softmax、设备结果写入和中止排空 |
| 存储与分区 | [session.cpp](../../src/session.cpp) | FP16 追加、历史/当前 KV 分区、非块对齐读取、query 分批及 prefill/decode 交接 |
| 图算子 | [device_ops.cpp](../../src/device_ops.cpp)、[device_ops.cuh](../../include/solid/device_ops.cuh) | 多 token KV consolidator 与结果视图；CPU 数学及 CUDA 分派 |
| 图接入 | [llama_adapter.cpp](../../src/llama_adapter.cpp) | Q/KV 保活、CPU 标量控制、独立结果 buffer、设备视图及释放记账 |
| 结果记录 | [baseline.cpp](../../src/baseline.cpp) | 传输、设备消费、代表批次与图结果内存计数 |

CUDA tiled prefill 在 dense 会话中直接接入；稀疏会话同时使用 CUDA 代表构造。CPU 代表构造及固定集合回放沿用主机输入。接入根据实际后端能力自动选择。

## 2. 数据流与 causal 分区

```text
设备 Q [T,Hq,D] ──────────────────────→ 批量代表更新 / tiled attention
       │                                          ↑
设备 K/V → FP16 [T,Hkv,2,D] ─────────→ 当前批次 KV ─┘
                    │
                    └→ 主机 FP16 追加 → SSD / 写缓冲
历史 KV → 按块读取 → 固定主机 tile → 固定设备 tile → tiled attention
                                                          │
                                                          ↓
CPU 控制完成标量 ← 私有流完成 ← 设备结果 buffer ← 归一化输出
       │                               │
       └────────→ 带控制依赖的结果视图 ←┘ → 后续 GPU 计算
```

输入布局为连续 FP32 Q `[T,Hq,D]` 和 FP16 KV `[T,Hkv,2,D]`。consolidator 支持多 token、独立 K/V 和融合投影视图；维度方向为其他 stride 时先整理，再转换为 FP16。

每个 query frame 最多处理 `query_capacity` 个 token。其 KV 前缀按输入批次起点分区：起点前的历史从存储读取，当前批次直接引用设备 KV。历史末端和当前起点可以位于块内，读取按绝对位置与有效行数拷贝。每个 query 仅累计自身及此前的位置。

例如当前输入起点为 5、长度为 13、query capacity 为 3：依次计算位置 5–7、8–10、11–13、14–16、17。每帧的历史区间为 0–4，当前区间为 5 至该帧末端；当前区间的设备借用覆盖对应 frame。

主机接收 FP16 KV 用于追加。attention 的 Q、当前 KV 和输出沿设备接口传递。历史 tile 的主机源在 H2D 完成事件后归还，设备 tile 复用由同一私有流排序。

## 3. 批量代表和借用期

代表批次按位置顺序排入均值、窗口评分、定稿和 K 追加内核，保持原 FP64 累计次序与位置优先规则。所有 token 共用既有环和 scratch；完整代表块直接进入评分 bank。每个完成块的 K、位置和分数镜像进入 CPU 核查，批次末执行一次显式 stream 同步。索引核对完成边界、完整块数量和 payload 后推进长度。

`begin_device` 借用 Q，`consume_device` 借用 token-major KV。生产者先完成，源分配持续至 `finish_to` 或 `abort` 返回。计算图的结果节点引用 Q/KV 与 CPU 完成标量，分配器按图依赖保留输入。

每个存活 prefill 图节点分配自己的结果 buffer，大小随该图的 microbatch 变化。`finish_to` 在私有流上把结果写入这个 buffer，完成全部设备读者后返回。GGML `CUSTOM` inplace 视图携带控制、Q 和 KV 依赖，后续计算直接引用结果 buffer。图节点释放时归还其内存记账；全部层完成 prompt 后释放原 tiled prefill 计算池，图结果继续保持自身生命周期。

输出与借用输入的地址范围分别校验，重叠时拒绝执行。存储、代表或 attention 失败时保存首个错误并中止私有计算；图提供有限的收尾输出。清理错误继续保留首错。

## 4. 计数与内存

| 字段 | 内容 |
| --- | --- |
| `direct_prefill_queries` | 完成设备输入 prefill 的 layer/token 数 |
| `device_builder_batches` | 成功提交完成的设备代表批次数 |
| `prefill_kv_download_bytes` | 主机追加接收的 FP16 KV 字节数 |
| `prefill_query_upload_bytes` | 主机 Q 接口的上传字节数 |
| `prefill_device_query_tokens` | 设备 Q frame 的 token 数 |
| `prefill_device_kv_tokens` | 各 frame 直接消费的当前 KV token 数，可包含跨 frame 的复用 |
| `prefill_output_download_bytes` | 主机输出接口的下载字节数 |
| `prefill_output_device_copy_bytes` | 私有流写入设备结果 buffer 的字节数 |

字段同时进入分阶段计数。原 `prefill_upload_bytes` 记录历史主机 tile 上传。`memory.prefill_workspace` 新增 `graph_output_device_bytes` 和 `graph_output_device_peak_bytes`，按实际分配记录存活图结果及峰值；原 KV 池、计算 scratch、主机 tile 分别保留原计数。

## 5. 检查证据

CUDA 算子、运行器和设备检查程序完成编译与链接。50 项 CTest、62 项 Linux Python 检查和源码指纹通过；核心库 14 项 CTest 的地址、泄漏和未定义行为检查通过。

- [device_prefill_test.cpp](../../tests/device_prefill_test.cpp)：两层、多 head、GQA、多代表、窗口绕回、中途接管、dense/sparse、内存/临时文件、两种 KV 布局、未对齐起点、输入输出重叠、历史读取失败、中止失败和 decode 状态交接。
- [device_prefill_graph_cases.hpp](../../tests/device_prefill_graph_cases.hpp)、[device_prefill_graph_test.cpp](../../tests/device_prefill_graph_test.cpp)：三次 microbatch、两层实际 CPU split 图、独立/融合/非连续 K/V、设备结果别名、保活边、数值、错误收尾和 buffer 回收。
- [prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp)：设备 prefill 接入三种调度的预取 decode，核对内存与线程化 liburing、代表状态、任意设备槽位及原生完成协议；轨迹由独立工具重算。
- [device_ops_test.cpp](../../tests/device_ops_test.cpp)：多 token、奇数维度、padding、FP16 rounding 和逐位数据核对。

[cuda_device_prefill_test.cpp](../../tests/cuda_device_prefill_test.cpp) 已编译和链接，实验机运行实际 CUDA/CPU split 图、批量代表、历史 tile、结果视图和 prefill/decode 状态核查：

```bash
build/release/bin/solid-cuda-device-prefill-test --run-device-check 0
```

本机执行 CPU 合成数据、微小临时文件及错误注入。设备执行在实验环境进行。

并发回归中通过调试器定位并修复模拟读取门的通知切换；稳定通知通道连续通过 30 轮。具体记录见 [工程审查](../ENGINEERING_REVIEW.md#6-2026-11-18模拟读取门与稳定通知)。后续工程范围见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。
