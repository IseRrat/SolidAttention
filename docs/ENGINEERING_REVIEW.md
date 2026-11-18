# 工程审查与收尾记录

完成于 2026-11-17。本轮检查覆盖阶段 0 至 5 的计算图接入、异步复制、槽位生命周期、异常传播、运行器与实验进程管理，以及评测来源核查。

## 1. 错误路径中的复制与 attention

`AttentionNode` 在处理错误后仍需向计算图提供收尾 mask。独立复制流持有 KV bank 时，后续 attention 必须等 producer 完成再读取。

修复位于 [llama_adapter.cpp](../src/llama_adapter.cpp)：先保存首个错误，调用 `GgmlPrefetch::quiesce()`，join worker、同步计算与复制并释放上传租约，随后提供收尾 mask。清理异常保留为后续信息，采样前抛出最初的 attention 错误。

[prefetch_failure_test.cpp](../tests/prefetch_failure_test.cpp) 使用 CPU buffer 和独立延迟复制流。测试先提交历史 H2D，再注入晚到 KV 的位置错误；mask 检查节点在 attention 前确认待完成上传数为零。修复前此检查失败，修复后 `overlap` 与 `overlap+sync_reuse` 均通过，且保存原始错误。

正常路径继续通过后继 Q 的依赖或请求边界确认完成，见 [阶段 4](phases/PHASE_4.md)。

## 2. 子进程所有权与异常清理

运行器从 `Popen` 创建到 cgroup 读取、running manifest 写入和采样，统一由 [run_monitored](../scripts/run_baseline.py) 管理。任一步失败都会停止所拥有的进程组并等待退出，再记录 failed/interrupted。记录文件再次失败时继续抛出首个错误。

[process_lifetime.py](../scripts/process_lifetime.py) 提供共享接口：

| 接口 | 行为 |
| --- | --- |
| `termination_signals` | 把 SIGTERM 转为受控中断；清理期间忽略重复 SIGTERM，退出后恢复原信号处理器 |
| `stop_process` | 发送终止信号、等待、超时强制结束；处理信号到达前进程已经退出的竞态；清理所属进程组的剩余成员 |
| `run_owned` | 为后端调用创建独立 session，异常与中断时完成清理后再传递错误 |

实验执行器、准确率 tokenizer/生成调用和运行器均使用这些接口。fio 保留 SIGINT 写出最终统计的退出流程，超时进入进程组清理。

检查包括：启动后 manifest 写失败、cgroup 初始化失败、二次记录失败、短进程日志校验、信号处理器恢复、超时与退出竞态，以及真实合成父子进程组停止。

RSS 记录增加 `rss_samples`。有效采样数为零时，`sampled_peak_rss_bytes` 保存 `null`；有效样本的峰值与 cgroup 峰值分别保存。

## 3. 实验完成与续跑核查

[experiments.py](../scripts/experiments.py) 的 `audited_job` 统一供首次执行、续跑跳过和 collector 使用。标记 complete 前核查：

1. 配置、输入、结果、代码和运行器身份。
2. 固定输出长度、停止原因、逐 token 轨迹和 decode 总时长。
3. 计划内存/swap 限额与 cgroup 实际值。
4. 干扰报告存在非空 jobs、完成 read，且各 job 的错误码为零。

缺字段和类型错误作为任务失败记录。续跑先进行相同核查，再跳过已完成项。`swap_max_bytes=null` 在启用内存上限时采用零 swap；单独指定 swap 限额要求同时配置内存上限。

合成测试覆盖短输出提前拒绝、无效 complete 记录的续跑拒绝、限额未生效、空 fio jobs、零 read、I/O 错误及有效对照。实验操作入口见 [阶段 5](phases/PHASE_5.md)。

## 4. 官方 scorer 的加载来源

[accuracy.py](../scripts/accuracy.py) 从锁定 checkout 的 `metrics.py` 和 `eval.py` 源字节编译并加载模块。此次评分使用独立的临时模块绑定；结束后恢复原 `sys.modules`、`sys.path` 和 bytecode 设置。

实际加载的源字节用于生成记录哈希，评分后再次校验文件与 Git revision/干净状态。官方 checkout 保持源码状态。缓存污染测试预先放入一个同名 `metrics` 模块，再验证本次加载使用固定目录的 scorer，并恢复调用前的模块状态。

## 5. 检查证据

| 检查 | 结果 |
| --- | --- |
| CUDA 编译与链接 | 通过，包含修复后的 adapter |
| CTest | 30 项通过，新增 `prefetch-failure-drain` |
| Python | Linux 54 项通过，包含真实合成进程组检查 |
| 核心内存检查 | 6 项 CTest 通过地址、泄漏和未定义行为检查 |
| 来源 | 运行器源码指纹与当前 C++ 源码一致，上游 submodule 保持固定提交 |
| 文档 | 本地链接、标题锚点、代码块、文风与配置 JSON 检查 |

本机检查使用 CPU 合成张量、延迟复制后端、临时文件及合成子进程。模型与设备实验继续使用阶段 5 的实验环境入口。

## 6. 2026-11-18：模拟读取门与稳定通知

Prefill 设备接入回归中，`native-completion-progress` 出现一次等待超时。调试器重现时，主线程在 `query_handoff` 的 quiesce 等待，协调 worker 等待读取完成，I/O 所有者阻塞在 poll。

[native_completion_test.cpp](../tests/native_completion_test.cpp) 的模拟门在开关时返回不同通知描述符：旧通道已进入等待，新通道的通知可能已被消费。修复为复制原 READ eventfd 的描述符，门变化与内核完成始终写入同一通知对象；I/O 所有者沿既有顺序消费通知并检查完成队列。

修复后连续 30 轮 CPU 合成检查通过，覆盖通知、FFN/H2D 进展、Q 交接、取消、错误与源租约回收。完整 50 项 CTest、62 项 Python 和源码指纹通过。重复入口为 `solid-native-completion-test --repeat 30`。Prefill 数据流与生命周期见 [设备接入记录](implementation/PREFILL_DEVICE_INPUTS.md)。
