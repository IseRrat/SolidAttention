# 预测期块调度

论文 §6.1 和算法 1 将读取与上传拆成块级任务。预测阶段在 Q 到达前为候选块建立 `SSD READ → H2D` 计划，同时把调度键交给 SSD 队列和就绪上传队列。

## 代码路径

1. `Prefetcher::begin` 按 init/local 与同层上一轮 selected 历史确定有界集合。`BlockStage` 根据 `READ → H2D` 依赖计算关键路径与最迟开始时间。READ 成本沿用历史块采样的平滑估算，再按有效行数缩放，主机已驻留块取最小成本。H2D 成本使用完整块复制估算。
2. SSD 派发先比较正式纠错与预测。预测内部先派发 selected，再派发 init/local。同类任务按关键路径、较小 LST 和稳定入队顺序派发。协调线程与独立 I/O 所有者使用同一比较规则。已提交到内核的请求按原 READ 协议收尾。
3. 主机槽随请求保存 H2D 调度键。原 READ 的全部片段完成后，块进入就绪集合。上一层 attention 完成并释放显存读者后，`DeviceCache::prefetch_ready` 按计划顺序上传就绪块。复制票据持有主机源直到完成。
4. 正式选块使用完整身份匹配并提升已有请求的优先级。所需版本尚未生成时，先提升同层、同 head、同块的旧前缀读取，再等原 READ 完成并补齐差量。正式图继续用真实读和复制票据确认汇合。

预测计划计算调度键，执行状态由原有槽位、读请求和复制票据持有。未命中的块按原取消和回收协议处理。正式工作集继续使用 `schedule.blocks.jsonl` 记录完成依据。

## 核查入口

计划、提升与就绪排序在 [prefetch.cpp](../../src/prefetch.cpp)，共用队列比较规则在 [async_reader.hpp](../../include/solid/async_reader.hpp)，跨层读取和上传在 [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp)。已有 I/O 队列检查覆盖正式纠错对预测的提升、selected 优先及同序 FIFO。CUDA 编译与链接、现有 51 项 CTest、64 项 Python 检查及源码指纹通过。当前层回写与下一层资源依赖见 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md)。
