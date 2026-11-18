# SYCL Decode 设备输入与传输队列

论文图 8、图 11 和 §6.2 对应交错 KV、投影后的选块前沿和当前行的传输依赖。本阶段把 SYCL decode 接进预取调度的设备输入入口。

- [图算子](../../include/solid/sycl_device_ops.hpp) 在 attention 设备上完成 F32 到 FP16 的交错写入，布局为 `[token, KV head, K/V, D]`。`overlap` 与 `overlap+sync_reuse` 以标量依赖通知协调线程。选择器直接读取图上的 Q，代表构造直接读取交错 KV 的 K。`naive` 的图边界仍带 FP16 KV，按串行顺序归档。
- [复制队列](../../src/sycl/copy_queue.cpp) 与 GGML 使用相同设备和上下文，拥有独立的队列。离散设备的历史块从固定主机缓冲进入 bank。集成设备的历史块直接写入共享 bank。当前 KV 行按 head 从图张量装入版本化槽位，保留命中的尾块前缀，仅补齐缺口。
- [调度接入](../../src/ggml_prefetch.cpp) 用复制事件确认源与槽位。`sync_reuse` 将最后的复制事件作为 attention 队列的依赖。归档操作排在当前行安装之后，协调线程收割完成事件再追加和提交 SSD 回写。多层时，后继 SSD 读取可在归档完成前派发。

运行记录中的 `device_rank_queries`、`device_builder_queries`、`device_builder_appends` 和 `direct_decode_queries` 用于核查设备输入。SYCL 与 CUDA 运行器在本机编译、链接并核查源码指纹。设备推理和吞吐在实验环境核查。
