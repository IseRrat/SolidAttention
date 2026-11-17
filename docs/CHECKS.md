# 开发检查记录

## 2026-11-17：上游接入与原版运行器

| 项目 | 检查结果 |
| --- | --- |
| 上游依赖 | submodule 固定到 `ef6bc6f367082338006c8379bc7bcab3e1584766`，Windows/Linux 读取一致 |
| 构建 | Ubuntu 24.04 / GCC 13.3.0 / CUDA Toolkit 12.6.85，CUDA 版本编译与链接通过 |
| 配置 | 类型、数值范围、预算关系、未知字段、重复字段、路径和 GGUF 分片声明检查通过 |
| 哈希 | SHA256 标准向量与跨读取分块的百万字节向量通过 |
| 命令行 | `--help`、`--build-info`、锁定模型配置解析通过 |
| 构建来源 | 编译时源码哈希与当前文件逐项一致，陈旧二进制识别测试通过 |
| 结果文件 | 原子 JSON 写入、非有限值拒绝和已有结果保留测试通过 |
| 模型分片 | 分片遗漏与内容变更检测通过，官方 Q4_K_M 两分片信息已锁定 |
| 文档 | 本地链接、代码块和差异格式检查通过 |

最终检查入口：

```bash
bash scripts/check.sh --cuda
```

结果为 4 项 CTest 和 5 项 Python 单元检查通过。检查使用配置文件、临时文本和合成数据。模型推理、性能及准确率使用 [DEVELOPMENT](DEVELOPMENT.md) 中的实验环境入口执行。

接口定位见 [UPSTREAM_MAP](UPSTREAM_MAP.md)。

## 2026-11-17：同步 SSD 与稀疏路径

| 检查范围 | 证据 |
| --- | --- |
| FP16 转换 | 全部有限 half 位模式 round-trip、subnormal 和 ties-to-even 边界 |
| 同步文件 | 独立与交错布局、多层多头、尾块、短读写重试、脏数据转发 |
| 错误处理 | EOF、写失败、在用文件独占创建、临时文件清理、计算图错误传递 |
| 内存 | SSD 写缓存容量随上下文保持固定；内存参考按上下文容量预留空间 |
| 代表与选择 | 后续窗口定稿、GQA、正负相关度、init/local 去重及 token 预算 |
| Prefill | 分批与连续处理的一致性、causal mask、独立 dense softmax 公式对照 |
| 图接入 | 非连续 Q/K/V 视图、FP16 收集、多头 mask、padding、CPU Flash Attention 数值对照 |
| 回放工具 | 固定实际输入 token、参考上下文继承、checksum、非有限值及首次超差定位 |
| 上游适配 | 构建目录生成文件，原始源文件字节保持一致，固定提交检查通过 |

完整检查入口为 `bash scripts/check.sh --cuda`，包含 CUDA 编译、11 项 CTest、10 项 Python 单元检查和源码指纹核对。计算使用合成张量与临时文件。实验环境的原版/内存/SSD 数值对照命令见 [SSD_PATH](SSD_PATH.md)。

## 2026-11-17：交错布局与融合投影

| 检查范围 | 证据 |
| --- | --- |
| 交错文件 | 两层两头、对齐、尾块与原排列；完整块读取请求 32 对 16，字节数相同 |
| 连续 KV 输入 | 内存与 SSD 接受 `[Hkv,2,D]`；固定选择回放的输出与原路径一致 |
| 融合投影 | F32/F16/Q4_K、混合 Q4_K/Q6_K、bias、量化行逐字节保留、多头多 token |
| 位置与 attention | Normal/NeoX RoPE、K 回写和 V 保留；两种布局的 CPU Flash Attention、GQA mask 与 padding |
| 观测 | CPU 计算图输出与分阶段统计、次数、输出字节；CUDA 接入编译与链接 |
| 固定集合消融 | 四组配置、实际输入 token、预算、选择轨迹、logits 与来源检查；缺失组与文件变更拒绝 |
| 回归与来源 | 原有配置、哈希、I/O 失败和数值检查通过；三份生成文件保持上游源码一致 |

最终入口 `bash scripts/check.sh --cuda` 通过 CUDA 编译、17 项 CTest、14 项 Python 检查及源码指纹核对。数学检查使用 CPU 后端与合成张量，文件检查使用临时文件。模型数值、CUDA 运行与设备性能的实验环境入口见 [阶段 2 完成记录](phases/PHASE_2.md)。

`bash scripts/check.sh --unit-only` 的 Debug 核心构建、3 项 CTest 和 14 项 Python 检查通过。15 篇 Markdown 的本地链接、标题锚点、代码块与文风检查通过，配置 JSON 解析通过。

## 2026-11-17：历史预取与纠错

| 检查范围 | 证据 |
| --- | --- |
| liburing 2.5 | 临时文件、有限队列、短读/EOF、原 READ 与取消 CQE、取消排空 |
| 历史预测与纠错 | 按层/头的历史、冷启动、命中、全错 selected、在途命中、缺块补读与容量拒绝 |
| 版本与顺序 | 尾块不可变前缀、脏行转发、逆序完成、取消确认先到、旧 generation 与重复通知 |
| DRAM/设备租约 | H2D pin、attention pin、完成后释放、有效块原位与 payload 保留 |
| 计算图 | 两层 GQA、真实 liburing/内存读取、任意物理槽位 mask、CPU Flash Attention 对同步稀疏参考 |
| 完成与清理 | GGML scheduler 完成回调、独立/交错多片段 I/O 失败、缓冲区排空 |
| 配置与工具 | 预取选项与构建特性、固定输入、选择差异、覆盖/epoch 审计及运行来源 |

`bash scripts/check.sh --cuda` 通过 CUDA 编译与链接、21 项 CTest、19 项 Python 检查和源码指纹核对。`bash scripts/check.sh --sanitize` 的 5 项核心 CTest 与地址、泄漏、未定义行为检查通过。输入为合成张量与临时文件，数学计算使用 CPU 后端。模型与设备实验入口见 [阶段 3 完成记录](phases/PHASE_3.md)。

## 2026-11-17：微任务调度

| 检查范围 | 证据 |
| --- | --- |
| DAG/LST | 环、重复边、前驱状态、关键路径优先、最晚开始时间、共享 fence 覆盖与单调性 |
| 有界 worker | 条件变量控制的并发进展、容量拒绝、异常传递及后续复用 |
| 提前选择 | 多轮变化 Q、GQA、代表定稿；Q-only 选择与原串行选择一致 |
| 调度计算图 | naive/overlap/sync_reuse，两层、任意物理槽位 mask，CPU Flash Attention 对同步稀疏参考 |
| 缺块/投影重叠 | 缺块读取屏障由合成 K 投影解除，两路任务均推进且观测区间相交 |
| 同步与租约 | 每例 12 帧/144 任务；reuse 为 6 次 Q 边界与 6 次请求边界；延迟 pin 保留上传 producer |
| 真实文件 | 三种模式均在临时文件和真实 liburing 上通过多层计算图检查 |
| 跨语言核查 | C++ 实际图导出调度/驻留轨迹，Python 独立核对任务依赖、最长路径、重叠和同步计数 |
| 工具与配置 | 三组固定输入及选择、来源与副本校验、模式约束、指标变更、失败/未闭合和提前释放拒绝 |

`bash scripts/check.sh --cuda` 通过 CUDA 编译与链接、27 项 CTest、23 项 Python 检查和源码指纹核对。`bash scripts/check.sh --sanitize` 通过 6 项核心 CTest 与地址、泄漏和未定义行为检查。输入为合成张量与临时文件，数学计算使用 CPU 后端。任务指标采用主机观测完成区间；模型、CUDA 实际并发和设备性能的实验环境入口见 [阶段 4 完成记录](phases/PHASE_4.md)。

17 篇 Markdown 的本地链接、标题锚点、代码块与文风检查通过，配置 JSON 解析及差异格式检查通过。

## 2026-11-17：论文实验代码与评测入口

| 检查范围 | 证据 |
| --- | --- |
| 词表与固定输入 | 零张量、纯元数据的合成 GGUF；两次 token 化一致、源/模型哈希、ID 类型和容量 |
| 模式配置 | 十种矩阵 variant 全部通过真实运行器配置解析；原生 Q4_0 与自定义 F16 约束 |
| 实验计划 | 精确 token 前缀、生成容量、预热与重复、关闭详细轨迹、源文件变更与非法轴拒绝 |
| 执行与恢复 | 假进程推进、独占锁、已完成项核查与跳过；失败和 pending 重复单独保留 |
| 统计与来源 | decode 步数、首 token 归属、逐 token 分位数、重复分布、代码/二进制混用及哈希变更拒绝 |
| 干扰接口 | 只读 fio 参数、已有文件限制、输入身份检查、连续 JSON 与未完成状态处理 |
| 准确率 | teacher-forced NLL、稳定 log-sum-exp、目标对齐、显式中间截断、预测覆盖及模型身份 |
| 官方接口 | OpenCompass 生成/PPL 契约、LongBench scorer revision 与字段、八任务导出合并、官方 CSV 导入 |
| 选择稳定性 | init/local 区域排除、按层/head 比较、相邻 selected 块重合率 |
| 绘图 | 合成文件夹具、summary 独立重算、完整重复要求、PNG/PDF 导出、误差条和来源清单 |

`bash scripts/check.sh --cuda` 通过 CUDA 编译与链接、29 项 CTest、42 项 Python 检查和最终源码指纹核对。`--sanitize` 的 6 项核心 CTest、地址、泄漏与未定义行为检查通过。18 篇 Markdown 的链接、标题锚点、代码块、文风及配置 JSON 检查通过。

新增验证使用零张量 GGUF、合成 token/logits、假进程与临时文件。绘图依赖安装在已忽略的构建目录，合成 PNG/PDF 完成可视核查。模型、设备与 fio 实验在实验环境按 [阶段 5](phases/PHASE_5.md) 执行。

## 2026-11-17：工程审查与收尾

| 检查范围 | 证据 |
| --- | --- |
| attention 错误路径 | CPU 延迟复制后端复现提前读取；修复后两种 overlap 模式先排空上传并保留原始错误 |
| 启动与监控 | manifest/cgroup 故障后停止子进程；二次记录失败保留首个错误；零 RSS 样本记录 null |
| 中断与进程组 | SIGTERM 处理器恢复、退出竞态、超时强制清理、真实合成父子进程组结束 |
| 完成核查 | 首次执行、续跑及 collector 共用校验；短输出、未应用限额、空/错误干扰记录拒绝 |
| scorer 来源 | 绕过同名模块缓存、源字节哈希与评分后校验、导入状态恢复 |

CUDA 编译与链接、30 项 CTest、Linux 54 项 Python 检查及源码指纹通过。核心库 6 项 CTest 的地址、泄漏和未定义行为检查通过。详细触发、修复与入口见 [工程审查记录](ENGINEERING_REVIEW.md)。输入为 CPU 合成张量、临时文件及合成子进程。
