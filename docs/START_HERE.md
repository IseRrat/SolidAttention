# 开始复现

本文提供获取依赖、构建、记录原版参考和进入 SSD 接入的顺序。各阶段完成后的代码与检查说明见 [阶段完成记录](phases/README.md)。

## 1. 获取 llama.cpp 并固定版本

上游已通过 Git submodule 固定到 `ef6bc6f367082338006c8379bc7bcab3e1584766`。获取项目后恢复依赖：

```bash
git submodule update --init --recursive
git -C third_party/llama.cpp rev-parse HEAD
```

提交号由 `cmake/Upstream.cmake` 校验，接口记录见 [UPSTREAM_MAP](UPSTREAM_MAP.md)。运行原版基线使用 [构建与检查](DEVELOPMENT.md) 中的统一入口；每次实验同时记录主仓库和上游提交。

## 2. 在 Linux 上构建 CUDA 基线

准备 C++ 工具链、CMake、CUDA Toolkit 和模型 GGUF 文件。构建方式参考 [上游构建文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)，具体选项以锁定提交的 `docs/build.md` 为准。

```bash
cmake --preset cuda
cmake --build --preset cuda
ctest --preset cuda
```

先查看生成程序的 `--help`，选定一个模型运行短输入。保存实际命令、完整日志、生成结果、prefill/decode 耗时以及 DRAM/VRAM 峰值。固定 GPU 层数、KV dtype、Flash Attention 开关、上下文容量和生成参数。

首次运行使用模型可完整容纳的短上下文；长上下文实验随后扩展到 1k、2k、4k、8k、16k、32k、64k、128k。模型原生上下文、扩展配置及 prompt 截断规则写入实验配置。

## 3. 记录环境与模型

| 项目 | 记录内容 |
| --- | --- |
| 代码 | 主仓库提交、上游基点、适配提交、构建命令 |
| 软件 | Linux 内核、编译器、CMake、CUDA、驱动、liburing 版本 |
| 硬件 | CPU、GPU、DRAM/VRAM 容量、SSD 型号、固件、文件系统、挂载参数 |
| 模型 | 模型仓库及 revision、GGUF SHA256、具体 INT4 格式、tokenizer、KV dtype |
| 运行 | prompt 文件及哈希、实际 token 数、输出上限、种子、采样、线程数、GPU 层数 |
| 内存 | DRAM 限额、swap 配置、页缓存策略、KV/工作区/传输缓冲区容量 |

## 4. 定位上游接口

沿首次运行路径填写接口表，记录锁定提交中的文件、符号、张量形状和调用顺序。

| 接入点 | 需要确认的内容 |
| --- | --- |
| 模型加载 | 层数、query/KV head 数、head 维度、量化格式、RoPE 参数 |
| KV 管理 | 分配、追加、位置索引、有效长度、cache 清理 |
| 计算图 | Q/K/V 投影、RoPE、KV 写入、attention、FFN 的依赖 |
| Attention | K/V stride、GQA 映射、位置与 mask、稀疏块寻址 |
| 后端 | 主机与 GPU 复制、事件、执行流、计算图拆分边界 |
| Prefill | microbatch 切分、历史 KV 访问、分块写出和内存释放 |

已定位的符号和调用关系见 [UPSTREAM_MAP](UPSTREAM_MAP.md)，适配时在该表记录改动。

## 5. 进入同步 SSD 路径

按 [实施计划](ROADMAP.md) 先完成全量同步读写，再加入稀疏选择、交错布局、历史预取和调度。每个版本保留独立开关，使用 [测试与评测](TEST_PLAN.md) 中的同组配置对照。
