# SYCL 动态选块与代表构造

论文 §3.1、§4 和 §5 的选择路径按 KV head 管理代表、候选和 token 预算。SYCL 运行器将评分、预算归约和代表构造接入同一份会话索引。调度器收到每个 head 的逻辑块号后，继续按选块、历史预测及 SSD 版本推进。

- [代表构造](../../src/sycl/block_ranker.cpp) 在设备上保留每层最近 `local` 个 token 的 K 和分数。每步计算 GQA 查询均值，累计后续查询对窗口内 K 的贡献。最老一行出窗时进入所属块的局部 top-R，块完整时发布 FP16 代表到设备 bank。完整块的 K、位置和分数回填会话索引。
- [块选择](../../src/sycl/block_ranker.cpp) 为每个 KV head、每个候选块计算相关度。每个 head 的工作组按得分降序、逻辑块号升序选出块，按候选贡献的 token 数扣减预算。init/local 去重与候选生成沿用会话规则。
- 设备支持 FP64 时沿用 CPU、CUDA 路径的累加顺序。其余设备使用两个 FP32 分量保留舍入残差。运行记录中的 `selection_backend` 和 `representative_backend` 标记计算路径，`query_upload_bytes`、`builder_key_upload_bytes` 记录此阶段经主机桥接的输入字节。
- `attention.selection_backend=auto` 按 attention 设备选择实现。`sycl` 显式要求 SYCL attention 后端和动态稀疏配置。模型入口与固定 llama.cpp 基点沿用[共享 KV 阶段](SYCL_SHARED_KV.md)。

预取 decode 的图张量和复制事件见 [SYCL 设备输入](SYCL_DEVICE_INPUTS.md)。

本阶段核查 SYCL、CUDA 运行器的编译、链接及源码指纹，并运行既有的配置、选块和代表构造 CPU 契约检查。真实设备推理在实验环境核查。
