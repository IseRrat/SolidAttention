# 独立 I/O 与协调线程

代码与合成检查完成于 2026-11-18。对应论文 §7 的 I/O、SSD-GPU 协调分工和 CPU 核预留，接入阶段 3 的预取读取及阶段 4 的计算重叠。

## 1. 代码入口

| 模块 | 入口 | 行为 |
| --- | --- | --- |
| I/O 所有者 | [io_worker.cpp](../../src/io_worker.cpp) | 专用线程创建、提交、取消、收割及销毁 liburing reader |
| 完成通知 | [async_reader.cpp](../../src/async_reader.cpp) | CQE eventfd、原 READ 生命周期、短读续传及取消确认 |
| CPU 分工 | [thread_affinity.cpp](../../src/thread_affinity.cpp) | 物理核拓扑、工作线程绑定、推理核预留及恢复 |
| 协调线程 | [scheduler.cpp](../../src/scheduler.cpp)、[ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | TaskWorker 启动绑定及错误传递；纠错、H2D、回写、后继预取 |
| 运行器 | [llama_adapter.cpp](../../src/llama_adapter.cpp)、[baseline.cpp](../../src/baseline.cpp) | reader 工厂、容量、线程配置及实际分工记录 |

## 2. 队列、完成与反压

协调线程将读请求放入有界队列，I/O 线程按队列次序填充内核 SQ。原 READ 完成后，I/O 线程自动继续提交积压请求；GPU 计算期间读取可独立推进。完成队列由协调线程消费，消费前保留请求身份及容量占用。

命令和 CQE 各使用一个 eventfd。I/O 线程先消费通知计数，再检查命令及 CQ，最后等待任一描述符就绪。真实 liburing 路径在空闲时阻塞等待；受控测试 reader 可采用毫秒轮询回退。

内核深度使用 `prefetch_queue_depth`。主机准入容量为 `max(queue_depth, 2 * Hkv * slots_per_head)`，上限 16384；独立/交错布局的每槽最多两次读均可纳入准入队列。主机 payload 继续由固定槽位池约束。

## 3. 取消、错误与关停

提交前取消直接产生逻辑取消完成，目标缓冲区始终无内核写者。已提交请求转发取消命令，原 READ 完成后归还目标；取消 ACK 单独计数。`completed_reads + queued_cancellations` 核对逻辑请求结束数。

drain 标记所有请求取消，等待在途原 READ 完成，再排空取消确认并清理记录。正常 drain 后可重新使用 reader。底层异常先排空写者，再保存首个错误；显式 finish 将该错误传回调用者。销毁按 drain、停止、join、释放描述符执行，槽位缓冲在 reader 之后释放。

## 4. CPU 配置及记录

预取配置支持 `io_cpu` 和 `coordinator_cpu`，默认 `-1` 自动选择。自动选择从进程允许的 CPU 中选取两个不同物理核，并给推理保留至少一个物理核。两个工作核的全部允许 SMT 同胞从推理线程亲和性中移除；随后创建的推理线程继承剩余集合。会话关闭后恢复原有亲和性。

显式 CPU 必须位于原允许集合、属于不同物理核，并留有推理核。配置解析拒绝小于 -1、越界整数及相同显式编号。运行结果记录 `prefetch.io_execution`、`io_cpu`、`io_admission_capacity`、`worker_submissions`、`queued_cancellations`、`inference_cpus` 和 `schedule.coordinator_cpu`。

实验变体保留预取配置中的线程编号；生成同步参考或布局消融配置时去掉相应字段。

## 5. 检查证据

CUDA 编译与链接、37 项 CTest、59 项 Linux Python 检查及源码指纹通过。核心库 10 项 CTest 的地址、泄漏及未定义行为检查通过。

[io_worker_test.cpp](../../tests/io_worker_test.cpp) 使用条件变量与受控完成描述符，核对 reader 全生命周期的线程归属、构造失败、提交前取消、取消 ACK 与原 READ、drain 后复用、故障排空、内核容量小于准入容量时的自动补充，以及线程绑定和推理亲和性恢复。

[uring_test.cpp](../../tests/uring_test.cpp) 在微小临时文件上分别检查直接与线程化 liburing 的短读、EOF、取消及 drain。[prefetch_graph_test.cpp](../../tests/prefetch_graph_test.cpp) 的磁盘路径使用独立 I/O 线程，三种调度模式与 GPU 代表协议模拟路径均与同步 CPU 数值参考核对。配置工具覆盖线程设置保留及消融转换。

本机检查使用 CPU 合成张量和微小临时文件。GPU 执行、模型及性能评测在实验环境进行。块级资源队列和设备 attention 完成事件继续按 [论文实现核查](../PAPER_IMPLEMENTATION_AUDIT.md) 推进。
