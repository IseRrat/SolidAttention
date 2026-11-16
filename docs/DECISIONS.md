# 实施决策

## 已选方案

| 编号 | 方案 | 落实位置 |
| --- | --- | --- |
| D01 | Linux + llama.cpp + CUDA；上游以 submodule 固定，适配提交保存在项目 fork | 阶段 0，START_HERE |
| D02 | SSD 保存完整 KV，decode 复用单层 DRAM/VRAM 工作集 | DESIGN §3 |
| D03 | 32 token 块；短输入预算 25%，其余 1k；init/local 和 selected 各占一半 | DESIGN §4，论文 §7 |
| D04 | 代表构造按 InfLLM §3.2，参数与模型位置规则一起锁定 | DESIGN §4 |
| D05 | 逻辑身份为 layer/head/block，首版物理单元为单 head 块 | DESIGN §2 |
| D06 | 同步全量 → 同步稀疏 → 交错 → 预取 → 调度 | ROADMAP |
| D07 | 论文评测配置采用 INT4 权重、FP16 KV、batch=1、最大输出 512 | TEST_PLAN |
| D08 | 每层 32 KiB 写缓冲，最新待提交版本由内存转发 | DESIGN §5，论文 §7 |
| D09 | I/O 完成或取消确认后释放槽位，generation 校验完成通知 | DESIGN §7 |

## 参数锁定任务

| 阶段 | 项目 | 决策所需证据 |
| --- | --- | --- |
| 0 | 上游完整提交号、fork 地址及适配分支 | 编译日志、短输入结果、接口定位表 |
| 0 | 首个模型、GGUF revision/哈希及具体 INT4 格式 | 模型 KV 形状、上下文配置、显存占用 |
| 0 | Linux/CUDA/liburing、硬件与文件系统 | 环境记录、SSD 随机读微基准 |
| 1 | prefill microbatch/tile、attention 接入点 | dense 输出对照、受限内存峰值 |
| 1 | 代表 token 数、评分窗口、head 聚合 | 选块轨迹、长文本检索与任务得分 |
| 1 | init/local 拆分、整块取整、当前 token 计数 | 边界输入、实际参与 token 数 |
| 1 | RoPE、位置与 mask | 原版张量对照、块重排实验 |
| 2 | direct I/O 对齐、请求聚合、队列深度 | 实际设备字节数、吞吐和尾延迟 |
| 3 | 各缓冲区容量、线程绑定 | 重叠轨迹、内存与容量压力测试 |
| 1 | 数值容差 | 参考路径重复性与后端误差分布 |
| 5 | 复跑次数、准确率阈值 | 运行波动与样本得分分布 |

## 决策记录格式

每项锁定后追加日期、配置值、代码提交、证据路径和影响实验。参数变更同步更新配置与对应测试。
