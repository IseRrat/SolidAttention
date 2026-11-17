# 阶段 5：论文实验代码与评测入口

代码完成于 2026-11-17。依据为 SolidAttention §7、§8.1 至 §8.6、表 1 和图 6、9、12、14 至 21。交付包括实验矩阵、执行与续跑、指标汇总、准确率接口、评分导入和可追溯绘图。模型与设备数据由实验环境采集。

## 1. 代码入口

| 模块 | 入口 | 完成内容 |
| --- | --- | --- |
| 运行器 | [baseline.cpp](../../src/baseline.cpp)、[config.cpp](../../src/config.cpp) | 词表读取、固定 prompt token、原生 Q4_0 KV、阶段计数和内存分项 |
| 实验矩阵 | [experiments.py](../../scripts/experiments.py) | 准备配置、预热与重复、随机顺序、有界执行、断点续跑、失败记录、汇总 |
| 来源与指标 | [experiment_common.py](../../scripts/experiment_common.py) | 配置/副本/结果哈希、模型与二进制身份、逐 token 计时和分位数 |
| 资源与干扰 | [experiment_runtime.py](../../scripts/experiment_runtime.py) | systemd scope、内存与 swap 限额、CPU 亲和性、只读 fio、就绪与清理 |
| 准确率后端 | [accuracy_client.py](../../scripts/accuracy_client.py)、[opencompass_model.py](../../scripts/opencompass_model.py) | 单请求生成、词表长度、teacher-forced 平均 NLL、OpenCompass 注册 |
| 样本与 LongBench | [accuracy.py](../../scripts/accuracy.py) | 模板渲染、样本身份、生成续跑、标准预测导出、官方 scorer 调用 |
| 准确率表 | [accuracy_report.py](../../scripts/accuracy_report.py) | 官方 CSV 列导入、五任务平均、八数据集 LongBench 平均及后端来源 |
| 选择稳定性 | [selection_analysis.py](../../scripts/selection_analysis.py) | 按层/head 的相邻正式 selected 块重合率 |
| 绘图 | [plot_experiments.py](../../scripts/plot_experiments.py) | 独立重算汇总、完整重复检查、PNG/PDF、误差条与来源清单 |

阶段 2、4 的固定输入和固定选择消融工具继续用于机制对照。性能矩阵使用相同 prompt token、固定输出长度和动态选择，自由生成的每步输入保存在各自结果中。

## 2. 精确输入与运行器字段

`solid-baseline --tokenize CONFIG` 校验模型分片哈希，以 `vocab_only=true` 读取词表，将 CONFIG 中的 prompt 文本编码为 JSON。输出包括模型分片哈希、源文本哈希、`add_special/parse_special`、token 列表及构建来源。

```bash
build/release/bin/solid-baseline --tokenize configs/corpus.local.json > data/wikitext2-qwen.tokens.json
```

`corpus.local.json` 从对应模型模板复制，prompt 指向实验环境中固定版本的 WikiText-2 拼接文本。token 化在完整文本上进行；矩阵从该模型 token 序列取精确前缀。每个模型使用自己的词表文件与输入哈希。

| 字段 | 行为 |
| --- | --- |
| `prompt_token_ids` | 可选的非空 int32 列表，直接作为实际 prompt；逐项检查词表边界与上下文容量 |
| `kv_dtype` | `f16` 为所有自定义 attention 路径；原生路径另支持 `q4_0`，要求 Flash Attention |
| `attention.selection_trace` | 默认 `true`；性能配置设为 `false`，省去正式选择轨迹写入 |
| `prompt_source` | 结果记录 `tokenized_text` 或 `token_ids` |
| `text/raw_text` | 前者过滤 EOG/control token；后者保留全部生成 token 的文本表示 |

prompt 长度包含 tokenizer 插入的 special token。配置容量采用 `prompt_tokens + max_new_tokens - 1`，最后一个输出来自已有 logits；上游实际容量另记为 `effective_context_tokens`。固定 token 回放同时继承参考 prompt ID，保持模型输入一致。

## 3. CUDA 论文矩阵

主模板为 [cuda-paper.json](../../configs/experiments/cuda-paper.json)：

| suite | 扫描与默认组 | 对应依据 |
| --- | --- | --- |
| `end-to-end` | 1k 至 128k；Offload、Offload+Sparse、SolidAttention | 图 12 |
| `memory` | 16k/32k/64k/128k；原生、SolidAttention | 图 14 |
| `in-memory` | InDRAM、InDRAM+Sparse、SolidAttention | 图 15 的机制对照，CUDA 路线 |
| `prefetch` | 16k、1k 预算；同步交错、历史预取 | 图 16 |
| `sensitivity` | 16/32/64/128 token 块；1k/2k/3k/4k 预算 | 图 18 的机制对照，CUDA 路线 |

默认使用 1 次预热、5 次测量、种子 2026、最多 512 个输出。性能矩阵固定生成 512 个 token，`ignore_eos=true`；自动预算用 `budgets=[0]` 表示。显式预算按确定性区域占一半拆分 init/local。模型权重、输入和生成规则随配置保存。

模板先接入已锁定的 Qwen2.5-7B，`models` 中分别指定模型 `config`、原始文本 `corpus` 和对应词表的 `tokens` 文件。扩充模型时，为 Llama-3.2-3B、Llama-3.1-8B 提供已校验的配置和各自 token 文件。每个模型的上下文扩展设置与原生位置规则从 GGUF/模板继承，实验记录保留完整模型元数据。

```bash
python3 scripts/experiments.py prepare configs/experiments/cuda-paper.json results/paper-plan
python3 scripts/experiments.py run results/paper-plan/plan.json --runner build/release/bin/solid-baseline --limit 1
python3 scripts/experiments.py run results/paper-plan/plan.json --runner build/release/bin/solid-baseline
python3 scripts/experiments.py collect results/paper-plan/plan.json results/paper-report
```

准备步骤校验全部轴、模型/词表身份、文本哈希及容量后创建新计划。生成的配置为绝对路径，每次运行使用独立结果目录和 SSD 会话文件。计划内的预热先执行，预热与测量各按保存的种子打乱顺序。

执行器使用 POSIX 独占文件锁，`--limit` 控制本次推进的任务数。重启后核查已完成运行并跳过，继续 pending 任务。失败与中断作为终态保留；重新测量使用新计划。`execution.json` 保存命令、时间、退出码、错误、日志和干扰文件哈希。

## 4. 内存限制、SSD 与干扰

默认 condition 采用 `17179869184` 字节（16 GiB）的内存上限和零 swap。执行器通过 `systemd-run --user --scope` 在启动 launcher 前设置 `MemoryMax/MemorySwapMax`，所需 memory controller 由实验环境提供。CPU 亲和性通过 condition 的 `cpus` 配置，空列表使用环境调度。

launcher 保存 cgroup 起止的 `memory.max/peak/stat/events`、`memory.swap.max`、`io.stat` 和有效 CPU 集合；每 100 ms 保存进程内存及 `/proc/PID/io`。汇总核对观测到的内存/swap 上限与计划。cgroup 范围包含 launcher 和模型进程。

SSD 模板为 [cuda-io.json](../../configs/experiments/cuda-io.json)。先在实验环境配置两个 `session_root` 挂载路径及已存在的 fio 测试文件，再准备计划：

```bash
python3 scripts/experiments.py prepare configs/experiments/cuda-io.json results/io-plan
python3 scripts/experiments.py run results/io-plan/plan.json --runner build/release/bin/solid-baseline
python3 scripts/experiments.py collect results/io-plan/plan.json results/io-report
```

SSD 对照覆盖 1k/4k/16k/64k 输入和 1k/4k 预算。干扰覆盖 1 至 4 GB/s 顺序读、200k 至 800k IOPS 的 4 KiB 随机读；GB/s 使用十进制字节。请求速率与 fio 实测 `read` 数据并列保存。

fio 仅使用已有普通文件，显式启用 `readonly`、`allow_file_create=0` 和 direct read，核对测试文件与模型、prompt、会话文件的身份。启动后等待状态 JSON 中出现完成的 read，再运行推理；后台进程意外退出使当前测量失败。结束与中断均停止该后台进程组。`fio.json` 作为连续 JSON 状态流读取，最终报告及带宽日志保留在任务 attempt 目录。

`cache_state` 保存实验环境声明，默认 `os-managed`。冷热缓存由实验机的准备流程控制，并与文件系统、SSD 型号和驱动信息一并记录。buffered KV 的页缓存纳入 cgroup 测量。

资源接口依据 [systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html)、[systemd 资源控制](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)、[cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html) 和 [fio 文档](https://fio.readthedocs.io/en/latest/fio_doc.html)。

## 5. 计时、阶段计数与内存

| 指标 | 计算或边界 |
| --- | --- |
| `decode_tokens_per_second` | `decode_steps * 1000 / decode_loop_ms`，首个 prefill 输出单独计数 |
| `decode_latency_ms` | `trace[1:]` 的 total_ms；每次运行给出均值、P50/P90/P99/P99.9 |
| 重复测量统计 | 各次运行指标的中位数、最小/最大、均值及分位数 |
| `first_token_ms` | prefill 加首次采样区间；模型/上下文加载分别记录 |
| `request_ms` | 运行器 request 起点到结果构造，包括模型和上下文初始化、计算及末尾排空 |
| `attention_phases.prefill/decode/final_flush` | 三个边界的计数快照及差量，包括读写、上传、选择、收集和等待 |
| `memory.native_memory_buffers` | 原生 memory 接口报告的各后端实际 buffer 字节与 host 属性 |
| `memory.custom_buffers` | KV store、代表、预取主机块、设备 KV、复制暂存、跟踪工作区和额外投影权重 |
| `kv_system_tracked_bytes` | 原生 memory buffer 之和；自定义路径为 store、代表、主机预取、设备 KV 和复制暂存之和 |

KV 系统统计、工作区、额外投影权重、RSS、页缓存及后端显存日志分别保留。原生 memory 的 rotation buffer 包含在其接口报告中。自定义路径的模拟原生 KV 分配从物理容量统计中排除。

同步读取时间、后台请求等待和主机完成区间沿用各自字段；图 16、18 的阻塞分析结合阶段 3、4 的详细轨迹与实验环境的设备观测。分位数采用 `(n-1)*q` 线性插值。每次运行和重复测量各自保存分布。

## 6. 结果核查与绘图

collector 验证源配置、实际配置、输入副本、模型、构建、结果和日志哈希，核对 token 轨迹及 decode 总时长。跨运行要求同一代码、运行器和上游提交。报告列出每组计划数、完成数与各状态计数，预热保留在原始记录中。

输出为 `summary.json`、`summary.csv` 和 `REPORT.md`。绘图从原始运行重新汇总，与 summary 逐项核对，并要求所选 suite 的全部重复完成：

```bash
python3 -m pip install -r requirements-experiments.txt
python3 scripts/plot_experiments.py results/paper-report/summary.json results/paper-speed-figure --suite end-to-end
python3 scripts/plot_experiments.py results/paper-report/summary.json results/paper-memory-figure --suite memory --metric kv_system_tracked_bytes
python3 scripts/plot_experiments.py results/paper-report/summary.json results/paper-budget-figure --suite sensitivity --x budget_tokens
```

每组中心为重复测量中位数，误差条为最小至最大值。PNG/PDF 共用数据，`figure.json` 保存 summary/计划哈希、代码身份、原始组与 run ID、绘图库版本、显示单位及图文件哈希。

## 7. 准确率样本与官方 LongBench

样本准备接受本地 JSONL：`id/_id`、模板需要的文本字段、`answers`，以及可选 `all_classes/length`。模板为显式的纯文本格式串，包含模型指令格式。数据 revision、源文件和模板哈希随 manifest 保存。

```bash
python3 scripts/accuracy.py prepare data/qasper.jsonl data/qasper-template.txt results/qasper-data --dataset qasper --revision DATA_REVISION
python3 scripts/accuracy.py run results/qasper-data/dataset.json configs/qwen2.5-7b-q4_k_m.json results/qasper-solid --runner build/release/bin/solid-baseline --variant solid
python3 scripts/accuracy.py export results/qasper-data/dataset.json results/qasper-solid/predictions.json results/qasper-pred
python3 scripts/accuracy.py score-longbench results/qasper-pred /path/to/LongBench results/qasper-score.json --revision FULL_40_CHARACTER_SHA
```

每个样本使用独立会话与结果文件；完成样本按 ID 续跑。默认超长输入返回容量错误；显式 `--truncation middle` 保留 token 前后两段，并记录原始文本和实际 ID。`--block-tokens/--budget-tokens` 支持检索与预算扫描，预算 0 使用自动规则。

预测导出按 ID 检查覆盖、结果文本、模型、精度、调度和运行来源，生成官方 `pred/answers/all_classes/length` 格式。scorer 从完整提交 SHA 对应的干净 LongBench checkout 加载 `LongBench/eval.py`，调用官方 scorer，并保存源码哈希。八任务集合为 `2wikimqa/triviaqa/hotpotqa/multifieldqa_en/musique/narrativeqa/qasper/gov_report`，检索另有 `passage_retrieval_en/zh`。

各数据集导出后，用 `accuracy.py merge DESTINATION EXPORT_DIRECTORY...` 合并成八任务预测目录。合并核对模型、精度、调度、代码身份和每个输入文件的哈希，拒绝重复数据集；随后对合并目录执行 `score-longbench`，生成供表 1 汇总使用的八任务分数。

评分接口依据 [LongBench eval.py](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py) 及 [metrics.py](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)。数据、prompt 模板、模型指令包装和 scorer revision 在正式评测前一起锁定。

## 8. OpenCompass 与表 1 汇总

[opencompass_model.py](../../configs/experiments/opencompass_model.py) 提供 `origin/quant/solid` 三个模型配置。把项目根目录加入 `PYTHONPATH`，设置 `SOLID_MODEL_CONFIG`、`SOLID_RUNNER`、`SOLID_ACCURACY_OUTPUT`，在固定提交的 OpenCompass 实验配置中导入该 `models` 列表，并选择 Winogrande、ARC-Challenge、MMLU、GSM8K 和 LongBench 的正式数据集配置。

生成接口串行执行 batch=1，采用 greedy；`get_ppl` 对目标 token 进行 teacher forcing，用稳定 log-sum-exp 读取逐行 FP32 logits，返回平均交叉熵。带 mask 时，prefix 整段 prefill，剩余目标逐 token 评分；无 mask 时从首 token 开始，分母按 OpenCompass HF 接口采用输入总 token 数。稀疏预算按带 mask 的 prefix 或无 mask 的完整评测输入长度确定。每次评分保存目标数、mask、结果及 score 记录。

接口依据 [OpenCompass BaseModel](https://github.com/open-compass/opencompass/blob/main/opencompass/models/base.py)、[HF 平均 NLL](https://github.com/open-compass/opencompass/blob/main/opencompass/models/huggingface.py) 和 [PPLInferencer](https://github.com/open-compass/opencompass/blob/main/opencompass/openicl/icl_inferencer/icl_ppl_inferencer.py)。

官方 summary CSV 通过显式行映射导入。映射 JSON 的 key 为 CSV `dataset` 值，value 为 `winogrande/arc-challenge/mmlu/gsm8k/longbench` 之一，每列对应一行：

```bash
python3 scripts/accuracy_report.py opencompass-summary.csv data/task-mapping.json /path/to/opencompass results/eval.py results/opencompass/solid results/solid-table.json --model-column solid-solid --revision FULL_40_CHARACTER_SHA
```

LongBench 单独评分时，映射保留四个通用任务，另用 `--longbench` 传入八任务官方分数。汇总核对代码、二进制、模型、KV 精度及调度身份；LongBench 为八数据集均值，总平均为五个任务列的均值。CSV、映射、评测配置、框架 revision、包版本与实际后端运行记录进入结果来源清单。

## 9. 选择稳定性与固定集合消融

给独立核查运行打开 `attention.selection_trace=true`，保存动态选择后执行：

```bash
python3 scripts/selection_analysis.py results/selection-check/result.json results/selected-overlap.json
```

定义为 `|S(t) ∩ S(t-1)| / |S(t)|`，S 为排除 init/local token 区域后的 selected 块 ID。逐层/head 记录，空集合与位置间断保留为无比较样本；正式 token 范围检查顺序、位置和预算。

布局四组固定集合工具见 [阶段 2](PHASE_2.md)，调度三组固定集合工具见 [阶段 4](PHASE_4.md)。性能矩阵关闭详细轨迹，机制核查配置单独保存。

## 10. 完成检查与实验交付

CUDA 编译、29 项 CTest、42 项 Python 检查和最终源码指纹核对通过。核心库 6 项 CTest 通过地址、泄漏与未定义行为检查。新增检查覆盖零张量 GGUF 词表、全部十种实验配置的实际解析、输入 ID/精度配置、合成矩阵及假进程续跑、失败与缺失重复、来源变更、NLL 对齐、官方接口契约、预测合并与评分覆盖、选择重合率和 fio JSON 状态流。

绘图用合成文件夹具完成 PNG/PDF 导出与可视检查，依赖安装在已忽略的构建目录。检查数据为合成 token、logits、元数据和临时文件。本机保持代码与合成数据检查流程。

实验环境交付包括模型/数据/评测版本锁定、CUDA 模型数值与并发核查、论文矩阵原始记录、设备与文件系统记录、重复测量、准确率表和图表。SYCL、FlexGen 的 Llama/Qwen 扩展与能耗测量按 [实验计划](../TEST_PLAN.md) 推进。阶段索引见 [完成记录](README.md)。
