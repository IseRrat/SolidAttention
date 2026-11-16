# 开始复现

## 1. 获取 llama.cpp 并固定版本

在项目根目录将上游加入 `third_party/llama.cpp`，使用 Git submodule 记录依赖提交：

```bash
git submodule add https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp rev-parse HEAD
```

完成首次构建和短输入推理后，将通过验证的完整提交号写入 [实施决策 D01](DECISIONS.md)。后续实验使用同一提交。现有检出的依赖通过以下命令恢复：

```bash
git submodule update --init --recursive
```

对 llama.cpp 的适配保存在项目 fork 的 `codex/solidattention` 分支，主仓库 submodule 指向对应提交；实验同时记录上游基点和适配提交。

## 2. 在 Linux 上构建 CUDA 基线

准备 C++ 工具链、CMake、CUDA Toolkit 和模型 GGUF 文件。构建方式参考 [上游构建文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)，具体选项以锁定提交的 `docs/build.md` 为准。

```bash
cmake -S third_party/llama.cpp -B build/llama-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build/llama-cuda --config Release -j 8
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

第一轮接口验证完成后，在 [系统设计](DESIGN.md) 中补充实际符号及调用位置。

## 5. 进入同步 SSD 路径

按 [实施计划](ROADMAP.md) 先完成全量同步读写，再加入稀疏选择、交错布局、历史预取和调度。每个版本保留独立开关，使用 [测试与评测](TEST_PLAN.md) 中的同组配置对照。
