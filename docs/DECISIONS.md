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
