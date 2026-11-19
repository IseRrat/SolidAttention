# SolidAttention-Lab

复现 FAST 2026 论文 SolidAttention，在固定版本 llama.cpp 上实现 SSD KV cache、交错 KV、动态稀疏选择、历史预取及计算与 I/O 重叠。

代码入口、已有功能、当前验证状态和待处理问题统一记录在 [实现记录](docs/IMPLEMENTATION.md)。后续代码修改同步更新这份记录。

配置位于 configs，运行入口为 solid-baseline --config FILE，构建使用仓库的 CUDA 或 SYCL CMake preset。完整构建说明见实现记录。

[论文页面](https://www.usenix.org/conference/fast26/presentation/zheng) · [论文 PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)
