# SYCL 长输入 Prefill 与 Decode 交接

SYCL 的 prefill 按 microbatch 和固定 KV tile 容量执行因果 attention。每个 work-group 处理一个 query head 的一行，逐 tile 更新 FP32 最大值、分母和带权 V，读完因果前缀后归一化。

- [分块内核](../../src/sycl/tiled_prefill.cpp) 从会话读取历史 KV tile，并直接使用本批次的交错 FP16 图张量。历史 tile 上传结束即可归还主机源，后续覆盖由队列顺序保护。当前批次的 Q 与结果保留在 GGML 设备缓冲。
- [批量代表](../../src/sycl/block_ranker.cpp) 在同一窗口上按 token 顺序消费本批次 Q/K。每个块在窗口成熟后发布到设备 bank，每批次一次完成等待。进入 decode 时继续使用这份代表 bank、环形窗口和会话 KV 版本。
- `attention.prefill_backend=auto` 按设备选择实现，`sycl` 显式选择 SYCL。落盘继续使用 FP16 KV；`prefill_device_query_tokens`、`prefill_device_kv_tokens`、`device_builder_batches` 与峰值容量可随运行结果核查。

本机核查 SYCL、CUDA 运行器的编译与源码指纹，以及已有的 prefill 图检查。模型和真实设备数据在实验环境核查。
