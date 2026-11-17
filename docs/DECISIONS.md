# 实施决策

## 已选方案

| 编号 | 方案 | 落实位置 |
| --- | --- | --- |
| D01 | Linux + llama.cpp + CUDA；上游 submodule 固定到 `ef6bc6f367082338006c8379bc7bcab3e1584766`，由 CMake 校验 | 阶段 0，START_HERE、UPSTREAM_MAP |
| D02 | SSD 保存完整 KV，decode 复用单层 DRAM/VRAM 工作集 | DESIGN §3 |
| D03 | 32 token 块；短输入预算 25%，其余 1k；init/local 和 selected 各占一半 | DESIGN §4，论文 §7 |
| D04 | 代表构造按 InfLLM §3.2，参数与模型位置规则一起锁定 | DESIGN §4 |
| D05 | 逻辑身份为 layer/head/block，首版物理单元为单 head 块 | DESIGN §2 |
| D06 | 同步全量 → 同步稀疏 → 交错 → 预取 → 调度 | ROADMAP |
| D07 | 论文评测配置采用 INT4 权重、FP16 KV、batch=1、最大输出 512 | TEST_PLAN |
| D08 | 每层 32 KiB 写缓冲，最新待提交版本由内存转发 | DESIGN §5，论文 §7 |
| D09 | I/O 完成或取消确认后释放槽位，generation 校验完成通知 | DESIGN §7 |
| D10 | 上游适配生成到构建目录，主仓库保存生成器和接入代码 | cmake/Adapter.cmake，UPSTREAM_MAP |
| D11 | 阶段 1 使用流式 prefill 与 FP16 收集后的 GGML Flash Attention decode | SSD_PATH，strided-attention-graph、gathered-flash-attention |
| D12 | 默认 1 个代表，GQA 组内平均 Q，使用原生 RoPE；并列按逻辑位置排序 | SSD_PATH，kv-and-attention |
| D13 | 文件块和 attention 工作集共用布局开关；交错读取保留 `[B,2,D]`，attention 使用 stride 视图 | 阶段 2，interleaved-layout、gathered-flash-attention |
| D14 | 同类型融合权重保留完整量化行；混合类型逐行转为 FP32；额外 buffer 单独记录 | 阶段 2，fused-kv-projection |
| D15 | 消融独立控制布局和投影，回放相同输入与选择集合；分项观测采用统一同步模式 | 阶段 2，layout_ablation.py、execution-profile |
| D16 | liburing 原 READ CQE 结束主机槽位写入期；取消 CQE 单独统计，短读按剩余部分继续 | 阶段 3，uring-io、prefetch-and-residency |
| D17 | DRAM 与设备各用固定槽位表；复制完成释放主机 pin，attention 完成释放设备 pin，纠错保持有效块原位 | 阶段 3，DeviceCache、prefetched-flash-attention |
| D18 | 预取尾块先完成原读取，再以新 generation 保留不可变前缀、补齐最新行；详细轨迹由独立开关控制 | 阶段 3，plan_read、prefetch_trace |
| D19 | Q 单独到达时完成代表评分与选择，历史补读在有界 worker 执行，GGML 继续当前 KV 投影 | 阶段 4，select_early、scheduled-flash-overlap |
| D20 | 独立 GGML 复制后端与单层 pinned 暂存；同流最新已记录事件覆盖批量复制 | 阶段 4，GgmlTransport、DeviceCache |
| D21 | 检查后继 Q 的实际图祖先后复用完成证明；末层复用已有请求同步，显式 fallback 单独计数 | 阶段 4，graph_depends_on、after_process |
| D22 | DAG 按关键路径/LST 安排就绪 I/O；主机完成区间、逐 kernel 观测和端到端计时分别记录 | 阶段 4，TaskDag、schedule_ablation.py |
| D23 | 性能输入从固定文本对应的模型 token 序列取精确前缀；容量覆盖 prompt 与生成输入 | 阶段 5，--tokenize、prompt_token_ids、experiments.py |
| D24 | 预热与测量分开，独立结果目录、保存随机种子；已完成项核查后跳过，失败与中断保留终态 | 阶段 5，execution.json、POSIX 锁、collect |
| D25 | decode 吞吐率使用 decode_steps；阶段计数、单轮延迟与跨重复统计分别保存 | 阶段 5，attention_phases、run_metrics |
| D26 | 干扰使用已有文件的只读 fio；观测完成 read 后开始推理，记录请求速率和实测报告 | 阶段 5，experiment_runtime.py、cuda-io.json |
| D27 | 准确率连接固定框架版本；原生 Q4_0 KV 对照，自定义 F16 KV；生成与 teacher-forced NLL 均保存来源 | 阶段 5，RunnerClient、OpenCompass、官方 LongBench scorer |
| D28 | 图表独立重算来源，要求完整测量；中位数与最小/最大误差条共用 PNG/PDF 数据 | 阶段 5，plot_experiments.py、figure.json |
| D29 | CUDA 代表块使用 FP16；相关度按 CPU 顺序 FP64 累加，关闭 FMA 收缩；top-k 按剩余 token 成本结束，同分按逻辑 ID 升序 | [GPU 选择](implementation/GPU_SELECTION.md)，block_ranker.cu、block-ranking-contract |
| D30 | CUDA 代表构造使用 local 大小的 K/分数环和固定 R 候选；query 边界移交完整 CPU 状态，定稿块直接写入 GPU bank；构造工作区在选择总分配中计入一次 | [GPU 代表构造](implementation/GPU_REPRESENTATIVES.md)，representative-builder-contract |
| D31 | CUDA prefill 按有界 Q/KV tile 累计 FP32 因果 attention，H2D 完成归还主机源，同流保护设备复用；prompt 完成后释放池，KV 与计算工作区分别记录峰值 | [GPU prefill](implementation/GPU_PREFILL.md)，tiled-prefill-contract |

## 参数锁定任务

| 阶段 | 项目 | 决策所需证据 |
| --- | --- | --- |
| 0 | 首个模型的实验路径与上下文配置 | `configs/models.lock.json`、模型元数据、显存占用 |
| 0 | Linux/CUDA/liburing、硬件与文件系统 | 环境记录、SSD 随机读微基准 |
| 实验 | microbatch 大小、代表数量和窗口预算扫描 | 固定 token 对照、选块轨迹及任务得分 |
| 实验 | 数值容差和准确率阈值 | 流式参考、原版 logits 和任务样本分布 |
| 2 | direct I/O 对齐、请求聚合、队列深度 | 实际设备字节数、吞吐和尾延迟 |
| 3 | 各缓冲区容量、线程绑定 | 重叠轨迹、内存与容量压力测试 |
| 5 | 复跑次数、准确率阈值 | 运行波动与样本得分分布 |

## 决策记录格式

每项锁定后追加日期、配置值、代码提交、证据路径和影响实验。参数变更同步更新配置与对应测试。
