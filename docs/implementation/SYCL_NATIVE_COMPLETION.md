# SYCL Attention 完成通知

论文图 11 与 §6.2 的同步点连接了工作集槽位回收、当前 KV 归档和后继预取。SYCL 在 `overlap+sync_reuse` 中使用独立通知队列，把当前 attention 的合并内核完成事件交给协调线程。

- [完成入口](../../src/sycl/attention_events.cpp) 捕获此次提交的 epoch。依赖满足后通知线程唤醒协调线程，协调线程用 epoch 核对事件归属，归还工作集槽位并提交已就绪的后继上传。
- 支持设备时间戳的 SYCL 设备在两段 attention kernel 外记录起止点。`completion_backend=sycl_queue_events` 标识实际通知路径，设备时长写入调度记录。GGML 图边界仍可提供完成依据。
- 逐 head mask 的设备内核同时服务预取和普通 gather。上游原版 attention 继续使用上游 dispatch。SolidAttention 的相关 SYCL 子图逐次提交，确保每轮都绑定当前 epoch。

本机核查 SYCL 编译、链接和源码指纹，以及共享完成逻辑的既有 CPU 图检查。真实设备上的完成顺序与重叠在实验环境核查。
