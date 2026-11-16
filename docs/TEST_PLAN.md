# 测试与评测

## 1. 参考路径与数值对照

| 路径 | KV 位置与选择 | 用途 |
| --- | --- | --- |
| InDRAM | 全量内存 KV，dense attention | 数值参考与内存对照 |
| Offload | 全量 SSD KV，dense attention | SSD 全量基线 |
| InDRAM+Sparse | 内存 KV，动态选块 | 隔离稀疏算法影响 |
| Offload+Sparse | 同步 SSD，动态选块，独立 K/V | 主性能基线 |
| +Interleave | 交错 K/V | 布局收益 |
| +Prefetch | 加历史预取 | 阻塞时间收益 |
| +Overlap | 加计算/I/O 重叠 | 调度收益 |
| +SyncReuse | 加同步点复用 | 完整系统 |

dense 内存与 SSD 路径比较 K/V、attention 输出和 logits。稀疏内存与 SSD 路径使用相同选择器，比较实际块集合及数值输出。布局、预取和调度对照固定正式选择集合；自由生成之外增加固定 token 输入重放，逐层定位差异。

数值指标记录最大绝对误差、相对误差和首个偏差位置。每个后端/dtype 在阶段 1 根据参考路径重复性固定容差，复用到后续阶段。准确率验收阈值在正式评测前按数据集确定并写入配置。

## 2. 功能验收

检查入口为 `bash scripts/check.sh --cuda`；测试覆盖与结果记录见 [CHECKS](CHECKS.md)，模型固定 token 对照见 [SSD_PATH](SSD_PATH.md)。

| 场景 | 构造 | 观察结果 |
| --- | --- | --- |
| 交错布局 | 多 head、递增 K/V、满块和尾块 | encode/decode 逐元素一致，attention 只访问有效位置 |
| 预算边界 | init/local 交叠、重复候选、非整块预算、当前 token | 去重后的 token 数符合配置，位置与预算记录完整 |
| 块重排 | 同一 K/V 和 mask 按不同槽位排列 | attention 输出符合容差 |
| 代表定稿 | 块离开 local 窗口、prefill 尾部、新 decode 块 | 构造窗口、代表位置和块版本一致 |
| 长 prefill | 小规模 dense 对照及超过内存容量的长输入，不同 microbatch/tile | 小规模输出符合数值容差；长输入从 SSD 分块访问历史 KV，内存峰值符合限额 |
| 预取 | 冷启动、全命中、部分命中、全错 | 正式选择与同步参考一致，补读后输出符合容差 |
| 在途 I/O | 延迟实际写入目标缓冲区，申请复用槽位 | 槽位保持占用，旧操作完成后允许复用 |
| 取消 I/O | 取消请求与完成通知乱序 | 取消确认或原操作完成后释放目标缓冲区 |
| 迟到通知 | 数据写入和 lease 释放已完成，复用后注入重复旧通知 | generation 校验拒绝旧通知，当前块状态与内容保持一致 |
| 在用槽位 | attention 或 H2D 持有 lease，同时提交下一层预取 | 目标缓冲区在最后读者完成后释放 |
| 尾块读写 | 写缓冲未满、SSD 版本较旧、立即读取新增 token | 从最新待提交版本转发，flush 后读回一致 |
| 容量压力 | 预测集合超过容量、实际集合超过容量 | 预测按容量裁剪；实际超额返回配置错误 |
| I/O 故障 | 短读、短写、失败、会话中断 | 短传输按剩余区间处理；失败传播到会话，块版本保持可查 |

每条测试保存输入、预期块集合、事件轨迹和输出摘要。

## 3. 论文配置

- 模型：Llama-3.2-3B、Llama-3.1-8B、Qwen-2.5-7B；权重 INT4，KV FP16。GGUF 的具体量化格式和 revision 记录在配置中。
- 请求：batch=1，最大输出 512 token。固定采样、种子、EOS 策略，记录实际生成长度。
- 稀疏：块大小 32；输入短于 4k 时预算为输入长度的 25%，其余为 1k；init/local 合占一半。
- CUDA 硬件参照：RTX 4070 Laptop 8 GB、Ultra 9 185H、Samsung 990 PRO 1 TB、Linux 6.8、CUDA 12.8。论文机器装有 64 GB DRAM，端到端性能实验限制 DRAM 使用为 16 GB。
- 精度参照：Origin 为原版 llama.cpp；Quant 使用 INT4 KV；Ours 为 FP16 KV 的稀疏系统，三者权重均为 INT4。

性能配置固定相同输入与精度；稀疏消融固定相同预算和选块策略。全内存对照记录实际所需容量。上下文范围覆盖 prompt 和生成 token，模型扩展及截断规则随配置保存。

## 4. 实验矩阵

| 实验 | 输入与扫描 | 对应论文 | 记录指标 |
| --- | --- | --- | --- |
| CUDA 端到端 | WikiText-2，1k/2k/4k/8k/16k/32k/64k/128k | 图 12 | token/s、逐 token 延迟、失败配置 |
| KV 内存 | 16k/32k/64k/128k | 图 14 | KV、代表向量、缓冲区及总内存 |
| 长上下文检索 | PassageRetrieval-en/zh，块 32/64/128/256 | 图 6 | 任务检索得分 |
| 选择稳定性 | LongBench 八任务，1k 预算 | 图 9 | 相邻轮 selected 重合率、预取命中率 |
| 准确率 | Winogrande、ARC-Challenge、MMLU、GSM8K、LongBench | 表 1 | 各任务得分、均值及样本差异 |
| 预取消融 | WikiText-2，16k 输入，开/关预取 | 图 16 | 每轮 KV 加载阻塞时间 |
| 布局消融 | 固定预算和选块，独立/交错 K/V | 图 17a | attention 延迟、请求数、传输开销 |
| 调度消融 | naive/overlap/overlap+sync_reuse | 图 17b | attention 延迟、同步次数、事件轨迹 |
| 敏感性 | 块 16/32/64/128，预算 1k/2k/3k/4k | 图 18 | attention 计算及 I/O 阻塞 |
| SSD 差异 | 不同 SSD、预算 1k/4k、长度 1k/4k/16k/64k | 图 19 | 平均、P90、P99 延迟 |
| I/O 干扰 | fio 顺序读和 4 KiB 随机读，独立测试文件 | 图 20-21 | token/s、平均及尾延迟、背景负载 |

LongBench 八任务使用 2WikiMQA、TriviaQA、HotpotQA、MultiFieldQA、MuSiQue、NarrativeQA、Qasper、GovReport，记录数据版本、样本 ID、prompt 模板、截断和评分脚本。选择重合率定义为 `|S(t) ∩ S(t-1)| / |S(t)|`，按层/head 记录；init/local 单独统计。图 9 的论文相似度另列原文值及其定义核对项。

图 13、15、17b、18、19 的原实验平台为 SYCL；CUDA 路线采集对应机制数据，SYCL 扩展后补齐平台对照。FlexGen 的 Llama/Qwen 适配、Qwen2.5-14B 和 GovReport 能耗作为扩展实验。

## 5. 计时、内存与 SSD 口径

分别记录模型加载、prefill、首 token、decode 和整请求耗时。decode token/s 为实际生成 token 数除以对应 decode 时间；同时保存计数与起止事件。遇到 prefill 直接产生首 token 的接口，单独记录该 token 的归属。

逐 token 延迟给出平均、P50、P90、P99；重复运行报告中位数和范围。初始使用 5 次独立测量，预热次数和样本量写入配置。I/O 阻塞取 attention 等待 KV 就绪的临界路径区间，使用事件时间轴计算重叠区间。

使用 cgroup v2 设置 DRAM 限额并记录 `memory.max`、`memory.peak`、`memory.stat` 和 swap 设置；CUDA 显存单独统计。页缓存、pinned memory、代表向量、读写缓冲区、模型与工作区分别计量。

buffered I/O 分冷缓存与热缓存两组，记录缓存准备步骤。direct I/O 记录对齐、队列深度和提交批量。设备计数与进程 I/O 计数并列，后台负载单独记录。prefill 新写入数据在 decode 时的页缓存驻留纳入测量。

加速比使用同组基线耗时除以系统耗时。论文 128k 下最高 3.1× 对应 Offload+Sparse，最高 98% 对应 KV 内存下降；报告同时保留进程总内存和显存数值。

## 6. 结果保存

每次运行输出完整配置、代码与模型哈希、命令、标准日志、逐 token 记录、I/O 事件、内存采样及任务得分。功能检查通过后采集该版本性能。结果图表标注模型、平台、比较对象和误差统计。
