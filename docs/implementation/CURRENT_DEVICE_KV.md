# 当前 KV 设备复制与尾块前缀复用

代码与合成检查完成于 2026-11-18。在 [Decode 设备输入](DEVICE_INPUTS.md) 上，将 FP16 consolidator 的当前 KV 行直接复制到 attention 的设备 bank。论文 §4 对应交错 KV，§6.2 对应计算、传输与槽位完成依赖；尾块保留和复制票据沿用本工程的版本化缓存协议。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 当前行接口 | [device_cache.hpp](../../include/solid/device_cache.hpp) | 当前 layer/position、借用设备 KV、传输票据与计数 |
| 尾块与租约 | [device_cache.cpp](../../src/device_cache.cpp) | 保留同层/head/block 前缀、原槽更新 generation、补齐缺口、读写完成核查 |
| GGML 复制 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 设备 buffer 身份与地址校验、逐行 D2D、历史缺口 H2D、复制描述符保活及错误排空 |
| 计算图 | [llama_adapter.cpp](../../src/llama_adapter.cpp) | 将设备 KV 张量传入调度器，以释放 mask 的图引用保持源分配 |
| 结果与离线核查 | [baseline.cpp](../../src/baseline.cpp)、[compare_prefetch.py](../../scripts/compare_prefetch.py) | 实际字节计数、前缀 generation 与覆盖范围重算 |

## 2. 数据流与尾块

```text
设备 K/V → FP16 [Hkv,2,D] ──D2D──→ 当前块的当前行 → Flash Attention
                    │                         ↑
                    └→ 主机 FP16 追加 → SSD   │
历史主机快照 ─────────── 缺失前缀 H2D ─────────┘
已有同身份设备前缀 ───────── 原槽保留 ─────────┘
```

当前绝对位置为 `p`，块大小为 `B`，块内行号 `r=p%B`，最新块版本为 `r+1`。每个 KV head 的有效前缀按 `(layer, head, block)` 匹配，选择已经完成复制、读者已释放的最大旧版本 `v`。保留 `[0,v)`，主机仅上传 `[v,r)`，设备复制当前行 `r`。缺少旧前缀时取 `v=0`。

例如 `B=4`、`p=6`：旧版本为 1 时保留第 0 行，从主机补第 1 行，再从设备复制第 2 行。旧版本为 2 时直接保留两行并复制当前行。未选中的 padding 由既有物理槽位 mask 屏蔽。

Q 前沿的历史整理提前保留当前块的可用前缀。当前 KV 到达后，正式工作集在同一个物理槽更新身份和 generation。每次更新要求旧 producer/readers 都已结束；不同层、head 或逻辑块分别校验。沿用单层有界 bank 及固定槽位容量。

## 3. 复制流与源生命周期

借用输入为连续 FP16 `[Hkv,2,D]`。GGML 张量的指针、形状、内存类型和设备 buffer 类型必须与借用视图及 bank 一致，源与 bank 的地址范围分离。主机收到 FP16 KV 的前沿已确认 consolidator 生产完成。

`naive` 在计算后端执行复制；两种 overlap 模式在独立复制后端执行缺口 H2D 与当前行 D2D。两者按同一流排序并由末端事件确认完成。复制票据持有源/目标行的 GGML 张量描述符，直至完成确认或错误排空。图中的释放 mask 引用当前 KV 和 Q，覆盖异步设备读取期。

普通模式等待复制完成后交给 attention；同步复用模式先在计算流等待最新复制事件，再执行释放 mask 和 attention。主机快照 pin、设备 producer 和 attention reader 按各自的完成依据释放。复制提交后发生错误时，先排空该流，再销毁描述符、回收源租约并进入既有首错收尾。

层图保留 `current_h2d` 聚合任务名，其传输内容包含历史缺口 H2D 和当前行 D2D。两类传输共用既有复制完成前沿；历史逐块图和原 READ 票据继续独立核查。

## 4. 计数与轨迹

| 字段 | 口径 |
| --- | --- |
| `current_device_rows` | 成功提交到 bank 的当前设备行数，逐 KV head 累计 |
| `current_device_bytes` | 当前行 D2D 字节数 |
| `current_host_bytes` | 当前块缺失历史行的 H2D 字节数 |
| `refreshed_device_tails` | 在原设备槽更新版本的尾块次数 |
| `retained_device_prefix_bytes` | 各次更新保留的有效前缀字节数 |

`device_upload_bytes` 汇总实际 H2D，包括历史整块和当前块缺口。D2D 单独计数。尾块原槽更新计入 `overwritten_blocks`，表示旧逻辑身份被新版本替换。

`residency.jsonl` 的当前复制事件增加 `copy_kind=current_device`、`current_position`、`retained_prefix_tokens`、`prefix_generation`、`row_bytes`、`host_copy_bytes`、`device_copy_bytes`。核查器要求旧身份释放、原槽 generation 递增、前缀身份/版本一致，独立重算传输字节并与结果计数比较。缺少字段、错误 generation、错误块位置和覆盖缺口均拒绝。

## 5. 检查证据

CUDA 运行器及设备检查程序编译、链接通过；51 项 CTest、64 项 Linux Python 检查、源码指纹通过。核心库 15 项 CTest 的地址、泄漏与未定义行为检查通过。CPU 适配器另有 7 项内存检查覆盖实际 GGML 图和异步复制描述符。

- [current_device_test.cpp](../../tests/current_device_test.cpp)：原槽更新、延迟完成、跨块、跳过更新后的缺口、冷尾块、跨层隔离、错误内存类型和第二个 head 提交失败。
- [prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp)：三种调度、两层 GQA、设备 prefill/decode 交接、独立/融合 KV、线程化 liburing 和逐行复制计数。
- [deferred_copies.hpp](../../tests/deferred_copies.hpp)、[prefetch_failure_test.cpp](../../tests/prefetch_failure_test.cpp)：延迟使用复制描述符，在实际复制排队后抛错，核查描述符、源张量和收尾 mask 的生命周期。
- [test_prefetch_tools.py](../../tests/test_prefetch_tools.py)、[audit_schedule_graph.py](../../tests/audit_schedule_graph.py)：前缀轨迹、generation、完整覆盖及结果字节计数独立核查。

```bash
bash scripts/check.sh --cuda
bash scripts/check.sh --sanitize
bash scripts/check.sh --sanitize-adapter
```

本机运行 CPU 合成张量、微小临时文件及错误注入。已编译的 [cuda_device_input_test.cu](../../tests/cuda_device_input_test.cu) 在实验机核对三种模式、独立/融合输入、多轮图复用及当前 D2D 计数：

```bash
build/release/bin/solid-cuda-device-input-test --run-device-check 0
```

后续代码按 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md) 推进预测期读取、回写与跨层事件的完整资源图。
