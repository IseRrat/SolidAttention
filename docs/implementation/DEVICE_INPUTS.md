# Decode 设备 Q/K 与 FP16 KV 接入

代码与合成检查完成于 2026-11-18。对应论文 §3.1 的代表构造和块选择、§4 的交错 KV，以及 §6 的计算/I/O 依赖。CUDA decode 直接使用投影后的设备 Q 和 FP16 K，主机追加接收交错 FP16 KV。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 借用输入 | [device_input.hpp](../../include/solid/device_input.hpp) | 内存类型、元素数、head stride、对齐和溢出校验 |
| CUDA 消费 | [block_ranker.cu](../../src/cuda/block_ranker.cu) | 设备 Q 更新窗口与评分、跨 stride 的 K 追加、有限值及设备身份核查 |
| 图算子 | [device_ops.hpp](../../include/solid/device_ops.hpp)、[device_ops.cuh](../../include/solid/device_ops.cuh)、[device_ops.cpp](../../src/device_ops.cpp) | Q 标量令牌、FP16 KV consolidator、释放 mask；CUDA 实现和 CPU 数学实现 |
| 图接入 | [llama_adapter.cpp](../../src/llama_adapter.cpp) | 选择前沿、设备张量保活、FP16 主机追加和 Flash Attention |
| 调度与索引 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp)、[attention.cpp](../../src/attention.cpp)、[session.cpp](../../src/session.cpp) | 三种调度模式共用输入、候选与结果核查、状态提交和错误传播 |
| 构建与记录 | [generate_adapter.py](../../scripts/generate_adapter.py)、[baseline.cpp](../../src/baseline.cpp) | 构建目录生成 CUDA CUSTOM 分派、输入计数与上传字节记录 |

## 2. 数据流

`selection_backend=auto/cuda` 在 CUDA 动态选块、GPU decode 和预取组合下自动接入。`naive`、`overlap`、`overlap+sync_reuse` 使用同一份设备输入接口和 FP16 KV 转换。CPU 选择和固定集合回放沿用主机输入路径。

```text
overlap / overlap+sync_reuse:
设备 Q → GPU 标量令牌 → 4 字节主机前沿 → 设备代表更新/选块 → 历史读盘/H2D
                         ↓ CPU 选择完成标量
设备 K/V + 设备 Q ──────→ GPU FP16 [Hkv,2,D] → 主机追加/设备 K 更新 → mask
                                    │                              │
                                    └──────── 释放 mask 算子 ←──────┘
                                                    ↓
                                             Flash Attention
```

串行模式先在 GPU 完成 Q 令牌和 KV consolidator，主机收到 FP16 KV 后依次执行设备代表更新、选择、追加、历史读取和 H2D。两种重叠模式在 Q 标量到达时开始选块和历史读取，随后继续 KV 投影与转换。

Q 布局为连续 FP32 `[Hq,D]`；交错 KV 为 FP16 `[Hkv,2,D]`。K 借用视图指向每个 head 的第一段，`head_stride=2*D`，CUDA 追加内核只读 K。consolidator 接受独立 K/V 或融合投影产生的 strided K/V，使用 round-to-nearest-even 转换，与 SSD 的 half 位模式一致。

每层每个 decode token 交给主机追加的 KV 为 `Hkv*2*D*sizeof(Half)` 字节。代表构造和评分直接读取设备 Q，代表环直接读取设备 K。[Prefill 设备接入](PREFILL_DEVICE_INPUTS.md) 覆盖批量 Q/K、历史 tile 分区和设备输出；主机接口继续用于参考。

[当前设备 KV](CURRENT_DEVICE_KV.md) 将 consolidator 的当前行直接 D2D 到 attention bank，保留同层/head/block 的可用设备前缀，只从主机补齐历史缺口。释放 mask 的 Q/KV 图引用覆盖复制期，票据持有独立张量描述符；对应传输字节及版本进入驻留核查。

## 3. 生命周期与图重放

借用接口要求生产者先完成，所有者保留输入分配至调用返回。CUDA 消费函数核对内存类型与 device ordinal，使用自有 non-blocking stream，在归还借用源前完成该 stream。空候选和初始空窗口也核查设备 Q 的有限值。

GGML 图明确保存以下依赖：

1. KV consolidator 引用 Q 和选择完成标量，Q 的设备分配覆盖提前选择。
2. 释放 mask 算子引用主机 mask、原始设备 KV 和设备 Q，Q/KV 分配覆盖主机追加及串行设备选择。
3. Flash Attention 消费释放后的 mask，再按既有事件与租约协议使用设备 bank。

三个设备算子均执行纯 GPU 数学或复制；CPU 协调在现有主机节点执行。CUDA 图捕获与重放复用稳定算子参数。CUSTOM 标记放在 `op_params` 的 32 字节偏移后，保留 CPU 回调参数；设备分派核对标记和形状后执行。生成器保持固定上游源文件。

输入形状、stride 或设备不符时拒绝调用。CUDA 错误排空自有 stream 并结束后端；主机追加或协调失败沿用首错保存、复制排空和收尾 mask 协议。

## 4. 结果字段与轨迹

以下字段进入总 attention 记录及分阶段计数：

| 字段 | 内容 |
| --- | --- |
| `direct_decode_queries` | 完成设备输入追加的 layer/token 数 |
| `device_builder_queries` | 成功消费设备 Q 的代表更新次数 |
| `device_rank_queries` | 成功消费设备 Q 的排名请求数 |
| `device_builder_appends` | 成功消费设备 K 的追加次数 |
| `query_upload_bytes` | 主机 Q 接口提交的上传字节数 |
| `builder_key_upload_bytes` | 主机 K 接口提交的代表环上传字节数 |
| `decode_kv_download_bytes` | 设备输入路径交给主机追加的 FP16 KV 字节数 |

阶段差值用于区分 prefill 与 decode。层图保留原任务身份；设备 Q 就绪记录使用 `query_device_input_ready`，FP16 KV 前沿使用 `kv_fp16_download_complete` 或 `serial_kv_fp16_download_complete`。attention 完成事件、历史块图及原 READ/复制票据保持独立核查。

## 5. 检查证据

CUDA 运行器、算子与设备检查程序完成编译和链接；48 项 CTest、62 项 Linux Python 检查、源码指纹通过。核心库 13 项 CTest 的地址、泄漏和未定义行为检查通过。

- [device_input_test.cpp](../../tests/device_input_test.cpp)：多层、GQA、多代表、窗口绕回、中途接管、并列分数、stride、类型、对齐、溢出、NaN 与调用顺序；与 CPU 标量索引核对代表和选中 token。
- [device_ops_test.cpp](../../tests/device_ops_test.cpp)：CPU 合成图核对奇数宽度、padding、signed zero、subnormal、half-way rounding、溢出以及 mask 位模式；拒绝未知 CUSTOM 标记和不支持的 stride。
- [prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp)：三种调度模式、两层 GQA、独立与融合 KV、重复计算图分配、设备输入保活边、CPU Flash Attention 数值及线程化 liburing 临时文件。
- [prefetch_failure_test.cpp](../../tests/prefetch_failure_test.cpp)：设备输入和主机输入均在历史复制尚未结束时注入错误，核对排空和首错传播。
- [audit_schedule_graph.py](../../tests/audit_schedule_graph.py)：独立重算串行、重叠、原生完成及设备输入路径的层图、块图和租约。

[cuda_device_input_test.cu](../../tests/cuda_device_input_test.cu) 已编译和链接。实验机入口覆盖纯算子三轮 CUDA 图重放、FP16 位模式、两层实际 CUDA/CPU split 图、三种调度、独立/融合 KV、同图多位置复用，以及 decode 的 Q/K 上传计数：

```bash
build/release/bin/solid-cuda-device-input-test --run-device-check 0
```

本机执行 CPU 合成检查；设备程序在实验环境运行。后续工程范围见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。
