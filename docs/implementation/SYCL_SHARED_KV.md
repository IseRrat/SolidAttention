# SYCL 共享 KV 与逐 head decode

论文 §6.2 在统一内存平台上让主机和设备共享 KV 工作集。SYCL 路径为支持共享分配的集成 GPU 使用 USM bank，协调线程填充版本化设备槽位，GPU 直接消费 bank 的交错视图。

## 数据与完成依赖

- [构建接入](../../cmake/Adapter.cmake) 和 [生成器](../../scripts/generate_adapter.py) 在构建目录产生 SYCL 后端接入文件。保持固定 llama.cpp 基点，投影、RoPE、FFN 与原生基线使用上游后端。
- [共享缓冲](../../include/solid/sycl_backend.hpp) 在 GGML SYCL 设备上下文分配和释放 USM。设备识别同时检查共享物理内存与共享分配能力。其他设备沿用现有复制后端和独立 KV bank。
- [传输协调](../../src/ggml_prefetch.cpp) 在 CPU 端将就绪块写入 USM 槽位。SSD 读取保留独立的 DRAM 槽，可在当前层 attention 期间推进。attention 完成并释放读者后才允许改写共享 bank。共享写入不提交 H2D 或分配 pinned 上传暂存。
- 固定上游的 SYCL Flash Attention 需要广播 mask。SolidAttention 每个 query head 使用自己的工作集 mask，[SYCL decode 内核](../../include/solid/sycl_flash_decode.hpp) 以各 head 的 mask 直接访问 K/V stride。每个工作组处理一个 head 的 64 行分片，FP32 在线 softmax 累计最大值、分母与带权 V。第二个内核按全局最大值合并，跳过全遮罩分片。GGML 为分片结果预留图工作区。
- `prefetch.device_shared_buffer_bytes` 记录共享 bank 容量，`prefetch.device_shared_host_copy_bytes` 记录写入它的主机字节数。`device_upload_bytes` 记录显式上传。当前 SYCL 选择和 prefill 沿 CPU 路径运行。

## 构建入口

固定上游引用了较新的 SYCL 架构枚举。本阶段使用 oneAPI DPC++ 2025.3.3 和 oneMKL 2025.3 系列构建。论文 §8.1 所列 oneAPI 2025.0 对应作者的实验环境。此处保留同一 llama.cpp 基点的 CUDA 与 SYCL 对照。 [Intel DPC++ 发行说明](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl
cmake --build --preset sycl --target solid-baseline
python3 scripts/run_baseline.py --runner build/sycl/bin/solid-baseline --check-build
```

`sycl` 预设启用 oneMKL 并使用普通 SYCL 队列。实际设备核查用既有稀疏预取配置，确认共享 bank 字节数、选块与逐步 logits。设备执行留在实验环境。

共享缓冲接入、逐 head 内核和完整 SYCL 运行器通过编译与链接。运行器包含 `sycl-spir64` 设备代码，源码指纹一致。CUDA 路径的现有 CPU 合成图、liburing 临时文件与构建检查通过。
