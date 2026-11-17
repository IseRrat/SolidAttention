# selected 优先读取与逐块上传

代码与合成检查完成于 2026-11-18。依据论文 §6.1：selected 预取优先于 init/local 传输，I/O 按 KV 块划分；读取与上传通过各块的完成状态衔接。

## 1. 代码入口

| 入口 | 行为 |
| --- | --- |
| [async_reader.hpp](../../include/solid/async_reader.hpp) | 四类读取优先级、带优先级提交及待提交请求提升 |
| [io_worker.cpp](../../src/io_worker.cpp) | 按优先级及 FIFO 次序填充内核队列；保留被拒绝准入期间发生的提升 |
| [prefetch.cpp](../../src/prefetch.cpp) | 固定预测集合后跨 head 排序、正式所需请求提升、完成回调与源 pin |
| [device_cache.cpp](../../src/device_cache.cpp) | 按完整计划保留设备槽位、就绪块逐个上传、在途源租约 |
| [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 历史纠错读与 H2D 流水，聚合 DAG 的前驱及完成确认 |
| [schedule_ablation.py](../../scripts/schedule_ablation.py) | 读取/上传重叠的依赖核查及汇合要求 |

## 2. 预算、类别与实际提交

预测集合仍先为每个 head 纳入 init/local，再按剩余槽位纳入历史 selected。集合确定后，跨 head 将历史 selected 排在确定性块之前发出；预算及身份保持原有规则。

正式选择确定后，对已有所需请求提升优先级，取消多余预测，并为遗漏块分配槽位。待提交次序为：

1. 正式所需 selected。
2. 正式所需 init/local。
3. 历史预测 selected。
4. 预测 init/local。

同优先级按提交序号执行。主机预取队列和独立 I/O 所有者均保留优先级；在内核容量不足时，后续补充继续使用最新顺序。已进入内核的读取沿用原完成/取消协议。一个层帧的有限工作集完成后才能切换，确定性块最终纳入 attention 的依赖汇合。

## 3. 逐块读取与上传

```text
正式选择 → 完整块身份计划 → 回收多余设备槽位
                          → 发起/保留所需读取
                              块 A ready → H2D A ─┐
                              块 B read  → H2D B ─┤→ 历史完成 → 追加当前 KV
                              块 C read  → H2D C ─┘
```

设备先按完整历史计划保留匹配槽位，再消费逐块就绪回调。一个块的全部 READ segment 完成后即可上传；另一个块可以继续读盘。回调临时 pin 源，DeviceCache 再持有独立复制租约，H2D 完成后释放。已有有效设备块保持原位。

回调异常先释放临时 pin。已启动的复制继续持有各自源租约，由正常 finish 或异常 quiesce 排空。尾块版本更新仍先确认旧 producer，再保留不可变前缀并补齐新行。

overlap 聚合 DAG 中，`history_h2d` 依赖正式选择，可与 `miss_read_and_reconcile` 重叠；`append_current_kv` 等待 KV 下载、纠错读取与历史 H2D 三路完成。naive 保留串行依赖。当前 DAG 继续记录每层 12 项聚合任务；块级成本、资源就绪队列及 LST 分派按 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md) 继续扩展。

## 4. 检查证据

CUDA 编译与链接、37 项 CTest、60 项 Linux Python 检查及源码指纹通过。核心库 10 项 CTest 的地址、泄漏与未定义行为检查通过。

- [prefetch_test.cpp](../../tests/prefetch_test.cpp)：两种布局、两个 head、容量 1/64，核对真实提交文件偏移，selected 跨 head 优先及预测集合；受限队列验证已就绪块上传时其余读取仍在途；第二次上传失败后核对完整回收。
- [io_worker_test.cpp](../../tests/io_worker_test.cpp)：内核容量 1，确认正式 selected 的提升、预测 selected 优先、同级 FIFO 及全部目标身份。
- [prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp)：三种实际 CPU GGML 图、线程化 liburing、代表构造协议模拟与同步数值参考；轨迹经过独立依赖核查。
- [test_schedule_tools.py](../../tests/test_schedule_tools.py)：接受读取与历史 H2D 重叠，拒绝追加缺少读取汇合依赖的轨迹。

本机检查使用 CPU 合成张量与微小临时文件。模型、GPU 执行与性能评测在实验环境进行。
