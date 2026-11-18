# CUDA attention 完成事件与后继上传

代码与合成检查完成于 2026-11-18。对应论文 §6.2 的共享完成点和受限工作集复用：CUDA attention 完成后交还设备槽位，协调线程在后继 Q 到达前推进下一层 H2D。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| 完成绑定 | [stream_completion.hpp](../../include/solid/stream_completion.hpp) | 稳定 token、注册身份及 Flash Attention 参数尾部绑定 |
| 通知协议 | [attention_events.cpp](../../src/attention_events.cpp) | 单活动 epoch、完成票据、主机边界、停止唤醒及 eventfd |
| CUDA 事件 | [attention_events.cu](../../src/cuda/attention_events.cu)、[cuda_completion.cuh](../../include/solid/cuda_completion.cuh) | 设备事件、图捕获、轻量通知、设备区间和复制完成查询 |
| 上游接入 | [generate_adapter.py](../../scripts/generate_adapter.py)、[Adapter.cmake](../../cmake/Adapter.cmake) | 构建目录生成 CUDA 挂钩，固定原版源码 |
| 协调与上传 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 完成后释放 bank、继续预测读取及 H2D、正式选择接管 |
| I/O 通知 | [io_worker.cpp](../../src/io_worker.cpp)、[prefetch.cpp](../../src/prefetch.cpp) | 独立完成描述符、消费顺序与原 READ 生命周期 |
| 计时与核查 | [scheduler.cpp](../../src/scheduler.cpp)、[schedule_ablation.py](../../scripts/schedule_ablation.py) | 逐帧设备区间、完成依据、累计值独立重算 |

## 2. 执行及图重放

```text
当前 KV H2D → 计算流等待复制 → begin event → Flash Attention → end event → 通知 → FFN
                                                                        ↓
协调线程：                                                     释放设备读者 → 下一层 H2D
```

CUDA 后端的 `overlap+sync_reuse` 自动使用此路径。每层保留一对设备事件及稳定完成 token。挂钩在实际 Flash Attention 调用前后记录事件，并将通知排在同一计算流上。通知函数只发布原子状态并写入 eventfd；协调线程读取事件区间、释放设备读者和发出后续复制。NVIDIA 对回调的顺序及接口约束见 [Execution Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)。

捕获期间使用 external event record 节点和 host function 节点，图每次重放均产生通知。事件句柄及通知 token 的生命周期覆盖缓存图。相关语义见 [CUDA 图约束](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html)。

完成绑定放在 `op_params` 的 32 字节偏移后，保留现有 scale、bias、softcap、precision 和稀疏参数。注册身份与 token 地址一起进入固定上游的完整 tensor 属性比较；更换注册会改变图属性。上游 scheduler 为新 split 分配新 uid，同一有效图重放继续复用稳定绑定。

## 3. epoch、源租约与正式选择

每次 attention 使用递增的全局 epoch。前帧完成票据消费或已有完整后端同步结束后进入新帧。协调线程校验当前 layer/epoch，只累计一次通知；过期观察回调携带的 epoch 在发布边界前核对。

I/O 所有者为协调线程提供独立的完成描述符。协调线程先消费通知计数，再检查状态和完成队列，同时等待 attention、读取或停止事件。原 READ 确认结束后，主机块才进入 H2D；复制完成查询归还源 pin。attention 通知结束当前设备读者，后继上传沿既有 generation 和版本规则复用槽位。

后继 Q 到达时发布依赖边界并唤醒协调线程；协调线程及时交回正式选择前沿。在途预测读取继续由 I/O 所有者管理，正式选择可提升所需请求并取消多余预测。请求结束沿用已有同步边界；停止先唤醒并 join 协调任务，再排空计算和复制、归还租约。

## 4. 结果与计时口径

| 字段 | 内容 |
| --- | --- |
| `schedule.completion_backend` | CUDA 为 `cuda_stream_events`；原有边界路径为 `graph_boundaries` |
| `device_event_completions` | 设备通知确认的 attention 帧数 |
| `device_attention_event_ms` | begin/end 设备事件区间之和 |
| `device_boundary_fallbacks` | 协调线程通过已有 Q/请求边界完成的帧数 |
| `uploads_before_boundary` | 后继 Q/请求边界到达前发出的块数 |
| `copy_event_queries` | 复制完成状态查询次数 |

`schedule.jsonl` 的 `attention_device_event` 完成记录保存 `device_ns`。核查器要求完成依据与设备区间成对出现，并独立重算累计值；Q/请求边界保持各自依据。主机观测区间、设备区间和端到端计时分别记录。设备事件区间的计时语义见 [Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)。

CPU 合成检查通过 `cpu_protocol` 模拟通知与复制协议。CUDA 设备执行、计时和并发效果在实验环境核查。

## 5. 检查证据

CUDA 挂钩、运行器及设备检查程序完成编译与链接。42 项 CTest、62 项 Linux Python 检查、源码指纹通过；核心库 12 项 CTest 的地址、泄漏和未定义行为检查通过。源码指纹同时覆盖 `.cuh`。

- [attention_events_test.cpp](../../tests/attention_events_test.cpp)：参数前缀保留、注册更换、通知唤醒、layer/epoch、重复完成、边界依据、取消和计时错误。
- [native_completion_test.cpp](../../tests/native_completion_test.cpp)：实际 CPU Flash Attention 数值对照；FFN 等待期间后继 H2D 进展；完成通知尚未执行时关停；worker 计时错误传播。受控 liburing 保持预测读取完成通知在途，验证正式选择及时接管及后续回收。
- [prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp)：两层 GQA、代表构造、动态选择、内存与线程化 liburing；原生完成协议替换逐 attention 主机边界。
- [audit_schedule_graph.py](../../tests/audit_schedule_graph.py)：独立重算层级图、块图、设备区间及全部计数。Python 用例拒绝缺少依据、负值或缺失的设备区间。
- [test_adapter_generator.py](../../tests/test_adapter_generator.py)：生成结果可重复、上下游调用位置、原版源码保持固定；源码变化按指纹拒绝旧构建。

[cuda_attention_events_test.cu](../../tests/cuda_attention_events_test.cu) 已编译与链接，实验机器运行同一 CUDA 图的三轮 epoch 重放：

```bash
build/release/bin/solid-cuda-attention-events-test --run-device-check 0
```

本机核查使用 CPU 合成张量、受控通知与微小临时文件。后续代码继续接入设备 Q/K、预测期与 D2H/回写任务图及 SYCL，范围见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。
