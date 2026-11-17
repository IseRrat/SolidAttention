# 有界 CUDA 分块 prefill

代码与本机检查完成于 2026-11-18。交付为有界 Q/KV 工作区、因果分块 attention、在线 softmax、H2D 源生命周期和 prefill 池释放。此路径将原有全量因果 prefill 的数学计算放到 GPU。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 分块协议 | [tiled_prefill.hpp](../../include/solid/tiled_prefill.hpp) | begin/consume/finish/abort、容量、分配与阶段统计 |
| 数学更新 | [online_attention.hpp](../../include/solid/online_attention.hpp) | 最大值、分母与历史累加量的在线重标定 |
| CUDA 执行 | [tiled_prefill.cu](../../src/cuda/tiled_prefill.cu) | 初始化、warp attention、输出归一化；独立 stream 与 H2D 完成事件 |
| 存储与批处理 | [session.cpp](../../src/session.cpp) | 追加当前 microbatch、分层读取历史块、布局转换、query 分批和失败排空 |
| 图接入 | [llama_adapter.cpp](../../src/llama_adapter.cpp) | 按实际设备选择后端，请求边界释放完整 prefill 池 |
| 配置与结果 | [config.cpp](../../src/config.cpp)、[baseline.cpp](../../src/baseline.cpp) | 后端、tile、分配峰值及阶段计数 |

## 2. 执行与因果规则

```text
microbatch Q/K/V → 追加 FP16 KV、更新代表构造
              → 分批上传 Q，初始化在线 softmax
              → 读取历史 KV tile → H2D → warp attention
                  ↑ 上一 tile 的计算与下一次主机读取重叠
              → 归一化、返回 FP32 attention 输出
```

每个 warp 负责一个 query head，GQA 映射到对应 KV head。Q 为 FP32，KV tile 为 FP16 `[kv_head,tile_tokens,2,D]`，维度读取在 warp 内合并。内核只消费位置 `key_position <= query_position` 的行。

跨 tile 保留最大值、softmax 分母和加权 V。新最大值出现时重标定已有状态，再加入当前行，使用 FP32 累计。容量覆盖 1–1024 的 head 维度，按 32/64/128/256/512/1024 寄存器分片编译。

一批 Q 的历史 KV 从位置 0 读到本批最后一个 query，批内的后续 K 由因果条件屏蔽。独立 K/V 与交错文件均打包成同一 tile。Session 对大于 query 容量的输入继续分批，并拆开跨 prefill/decode 边界的输入。

## 3. 缓冲区与完成证明

设备侧复用一个 KV tile 和有界 Q/加权输出/最大值/分母工作区，容量随配置固定。主机侧复用一个 tile；块读取通过已有存储转发规则获得最新 KV。

Q 和 tile 的 H2D 后记录完成事件。consume 等待该事件后归还主机源，GPU attention 可以继续执行；下一次设备 tile 覆写由同一 stream 排在前次计算之后。finish 等待归一化、输出与状态下载结束。

存储、后端或输出检查异常时，Session 先 abort 自有 prefill 帧再传播首个错误。所有层达到 prompt 长度后，在已有请求同步边界释放设备池与主机 tile，保留累计计数和分配峰值。

## 4. 配置与内存口径

在 attention 配置中设置：

```json
"prefill_backend": "auto",
"prefill_tile_tokens": 512
```

`auto` 在 CUDA attention 后端及 `gpu_decode=true` 时启用该路径；`cpu` 保留流式参考；`cuda` 要求相应构建和设备。默认 query 容量为运行器的 microbatch 大小。KV tile 按逻辑块大小向上取整，并以请求容量限制。实际容量记录在 `attention.prefill_query_capacity`、`prefill_tile_capacity`。

设 Q 容量为 `Q`，KV tile 为 `T`，head 维度为 `D`：

```text
KV tile device bytes = Hkv * T * 2 * D * 2
Q/online workspace bytes = Q * Hq * (2*D+2) * 4 + 4
host KV tile bytes = KV tile device bytes
```

`memory.custom_buffers` 保存当前 prefill KV 分配；`memory.prefill_workspace` 保存设备工作区当前值、峰值、KV 设备峰值及主机峰值。KV 分配进入 `kv_system_tracked_bytes`，Q/计算工作区单列。池释放后当前值归零，峰值保留。

`attention_phases` 保存 tiled batches、queries、tiles、KV token 数、上传字节和主机观测的 prefill 包含区间。运行结果保存实际后端 `cuda_online_tiled`。实验/准确率入口保留模板里的 prefill 后端与 tile 设置。

## 5. 检查证据与实验入口

CUDA 编译与链接、36 项 CTest、57 项 Linux Python 检查、源码指纹通过；核心库 9 项 CTest 的地址、泄漏和未定义行为检查通过。

[tiled_prefill_cases.hpp](../../tests/tiled_prefill_cases.hpp) 在 CPU 上执行 warp 点积和在线重标定，对照原有 FP64 因果 attention。覆盖 GQA、1/7/33/64/80/128/256/512/1024 维、常量 V、NaN padding、主机 Q/KV 源立即复用、完整历史、尾块和固定设备容量。

[tiled_prefill_test.cpp](../../tests/tiled_prefill_test.cpp) 覆盖两层、两种文件布局、融合投影输入、多 query 分批、跨 prefill/decode 边界、池释放及峰值保留；实际 CPU GGML 图消费同一后端协议。故障注入覆盖 begin/consume/finish、输出形状/数值和存储读取，核对 abort 及会话失败状态。

[cuda_prefill_test.cpp](../../tests/cuda_prefill_test.cpp) 完成编译与链接，实验机器使用：

```bash
build/release/bin/solid-cuda-prefill-test --run-device-check 0
```

本机数学执行使用 CPU 合成数据。模型 logits、设备执行与性能对照在实验环境执行。
