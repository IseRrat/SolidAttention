# 实现记录

更新日期为 2026-11-19。条目覆盖本仓库现存的新增与修改文件，包含当前本地改动。后续修改同步更新对应文件条目。

## 一、上游接入与构建

1. **行尾规则**
   - 文件 [.gitattributes](../.gitattributes)。
   - 内容 源码和文档统一使用 LF，Windows 批处理使用 CRLF。

2. **本地产物排除**
   - 文件 [.gitignore](../.gitignore)。
   - 内容 排除构建目录、模型、数据、结果、缓存、日志及本地配置。

3. **上游地址**
   - 文件 [.gitmodules](../.gitmodules)。
   - 内容 登记 llama.cpp 子模块路径及官方仓库地址。

4. **上游版本引用**
   - 文件 [third_party/llama.cpp](../third_party/llama.cpp)。
   - 内容 固定提交 ef6bc6f367082338006c8379bc7bcab3e1584766，供模型图、量化矩阵乘法、RoPE、FFN 和设备后端使用。

5. **工程目标与依赖**
   - 文件 [CMakeLists.txt](../CMakeLists.txt)。
   - 内容 建立核心库、配置库、运行器和既有检查目标，接入线程库、liburing、llama.cpp 与可选 GPU 后端。

6. **构建配置**
   - 文件 [CMakePresets.json](../CMakePresets.json)。
   - 内容 定义 checks、cuda、sycl 的编译器、输出目录、后端开关和并行构建参数。

7. **依赖版本约束**
   - 文件 [cmake/Upstream.cmake](../cmake/Upstream.cmake)。
   - 内容 核对 llama.cpp 子模块版本与源码路径，检查构建所需上游文件。

8. **上游适配编译**
   - 文件 [cmake/Adapter.cmake](../cmake/Adapter.cmake)。
   - 内容 把生成的图与后端文件替换进编译目标，加入适配器、预取和设备源码，关联 CUDA/SYCL 库与编译选项。

9. **源码指纹**
   - 文件 [cmake/BuildInfo.cmake](../cmake/BuildInfo.cmake)。
   - 内容 收集 C++、CUDA、头文件、构建文件及适配生成器的 SHA256，生成后端特性与上游版本信息。

10. **构建信息模板**
   - 文件 [cmake/build_info.hpp.in](../cmake/build_info.hpp.in)。
   - 内容 承载编译时源码哈希、上游提交及 CUDA、SYCL、liburing 特性 JSON。

11. **构建目录适配生成**
   - 文件 [scripts/generate_adapter.py](../scripts/generate_adapter.py)。
   - 内容 为 llama.cpp 图、context、通用 backend、CUDA backend 和 SYCL backend 生成接入文件。加入自定义 attention、KV 分配接管、设备操作、共享 KV、完成事件和本地计时标记，校验替换锚点。

12. **仓库入口**
   - 文件 [README.md](../README.md)。
   - 内容 保留项目用途、论文链接及本实现记录的唯一文档入口。

13. **实现记录**
   - 文件 [docs/IMPLEMENTATION.md](../docs/IMPLEMENTATION.md)。
   - 内容 按模块逐文件记录已有修改，随代码变更更新相应条目。

## 二、配置、运行器与会话

1. **配置结构**
   - 文件 [include/solid/config.hpp](../include/solid/config.hpp)。
   - 内容 定义模型分片、上下文、批量、生成、KV 布局、稀疏预算、预取、调度及后端选择字段。

2. **配置解析**
   - 文件 [src/config.cpp](../src/config.cpp)。
   - 内容 校验字段类型、重复及未知字段、数值范围、预算关系、路径和后端组合，解析模型分片及选择回放配置。

3. **统一运行器**
   - 文件 [src/baseline.cpp](../src/baseline.cpp)。
   - 内容 接入模型加载、分片校验、词表分词、固定 token 输入、greedy 生成及 logits 输出，记录 prefill、decode、最终回写、内存分项、设备身份与 I/O 计数。

4. **文件哈希接口**
   - 文件 [include/solid/filehash.hpp](../include/solid/filehash.hpp)。
   - 内容 声明文件 SHA256 计算接口。

5. **文件哈希实现**
   - 文件 [src/filehash.cpp](../src/filehash.cpp)。
   - 内容 分块读取文件并计算 SHA256，供模型与结果来源核对。

6. **会话接口与统计**
   - 文件 [include/solid/session.hpp](../include/solid/session.hpp)。
   - 内容 定义 KV 存储、代表索引、选择回放、prefill、decode、设备输入、归档及工作区统计接口。

7. **会话处理**
   - 文件 [src/session.cpp](../src/session.cpp)。
   - 内容 组织当前 KV 追加、稀疏选择、流式参考 attention、历史分块读取、设备 prefill 和 decode 工作集，保存选择轨迹及实际缓冲容量。接入设备 K 追加与主机 KV 归档的分步处理。

## 三、KV 存储与交错投影

1. **KV 数据与存储接口**
   - 文件 [include/solid/kv_store.hpp](../include/solid/kv_store.hpp)。
   - 内容 定义层数、head、维度、块大小、容量、布局和对齐参数，提供 FP16 块、内存存储、文件存储、读写片段和计数结构。

2. **KV 文件与缓冲**
   - 文件 [src/kv_store.cpp](../src/kv_store.cpp)。
   - 内容 实现 FP16 转换、内存 KV、SOLIDKV1 文件头、按层/head/块定位、separate 和 interleaved 布局、短读写补齐、每层默认 32 KiB 聚合写缓冲、dirty 数据读取与异步 WRITE 提交版本管理。

3. **融合投影接口**
   - 文件 [include/solid/projection.hpp](../include/solid/projection.hpp)。
   - 内容 定义融合权重、bias、K/V 视图、RoPE K 写回及额外权重容量统计接口。

4. **融合 K/V 权重**
   - 文件 [src/projection.cpp](../src/projection.cpp)。
   - 内容 按 KV head 交错拼接 K/V 权重行与 bias，相同类型保留量化行编码，混合类型转 F32。生成单次 KV 矩阵乘法、stride 视图和 RoPE 后 K 写回节点，统计额外权重分配与初始化耗时。

## 四、稀疏选择与代表向量

1. **预算与代表索引**
   - 文件 [include/solid/attention.hpp](../include/solid/attention.hpp)。
   - 内容 定义 init/local/selected 预算、逐 head 选择结果、代表构造、设备状态接管、批量观察与参考 attention 接口。

2. **代表与选择规则**
   - 文件 [src/attention.cpp](../src/attention.cpp)。
   - 内容 累计后续 local 窗口内的 GQA 平均查询对历史 K 的分数，按得分及位置选出块内 top-R，完整块定稿并发布。计算块相关度，合并 init/local/selected，按 token 去重和剩余预算裁剪，处理并列排序。

3. **设备选择器协议**
   - 文件 [include/solid/block_ranker.hpp](../include/solid/block_ranker.hpp)。
   - 内容 定义候选块及 token 成本、代表种子、定稿结果、设备内存与上传计数，提供发布、评分、窗口更新、设备输入和批量代表接口。

4. **代表数学规则**
   - 文件 [include/solid/representative_math.hpp](../include/solid/representative_math.hpp)。
   - 内容 提供 GQA 查询均值、窗口分数更新和稳定 top-R 插入，保持分数并列时按绝对位置升序。

5. **代表种子检查**
   - 文件 [src/representative_builder.cpp](../src/representative_builder.cpp)。
   - 内容 校验 pending K、窗口分数、部分块代表的形状、有限值、位置范围、唯一性及排序，检查尾部空槽。

6. **块排名归约规则**
   - 文件 [include/solid/ranking_math.hpp](../include/solid/ranking_math.hpp)。
   - 内容 定义候选比较与归约结构，按有效候选、得分和逻辑块号确定顺序。

7. **CUDA 代表与选块**
   - 文件 [src/cuda/block_ranker.cu](../src/cuda/block_ranker.cu)。
   - 内容 实现 FP64 块评分、逐 head token 预算 top-k、定稿代表增量发布、GQA 均值、环形窗口更新及块内 top-R。支持设备 Q/K 借用、CPU 种子接管、prefill 批量代表与设备分配统计。

8. **SYCL 代表与选块**
   - 文件 [src/sycl/block_ranker.cpp](../src/sycl/block_ranker.cpp)。
   - 内容 实现设备代表 bank、pending 环、窗口更新、块定稿、逐 head 评分和预算归约。按设备能力选 FP64 或双 FP32 分量，接入设备 Q/K、批量代表和主机回填。

## 五、Prefill、Decode 与设备图操作

1. **分块 Prefill 协议**
   - 文件 [include/solid/tiled_prefill.hpp](../include/solid/tiled_prefill.hpp)。
   - 内容 定义 query/KV tile 容量、主机及设备输入、在线累加、设备结果、终止与池容量统计接口。

2. **在线 Softmax 数学**
   - 文件 [include/solid/online_attention.hpp](../include/solid/online_attention.hpp)。
   - 内容 计算新最大值、历史分母缩放和新 token 权重，供分块 attention 累加。

3. **CUDA 分块 Prefill**
   - 文件 [src/cuda/tiled_prefill.cu](../src/cuda/tiled_prefill.cu)。
   - 内容 实现有界 Q/KV 工作区、因果 attention、FP32 在线 softmax、归一化、上传源释放和设备结果复制。本地后续修改增加四个 query 共用八行 K/V 的 CTA 内共享面板，包含尾 query、因果边界和统一 barrier 处理。

4. **SYCL 分块 Prefill**
   - 文件 [src/sycl/tiled_prefill.cpp](../src/sycl/tiled_prefill.cpp)。
   - 内容 实现历史 tile 上传、当前设备 KV 读取、逐 query head 在线 attention、最终归一化及设备输出，记录 query、KV、工作区和复制容量。

5. **借用设备数据结构**
   - 文件 [include/solid/device_input.hpp](../include/solid/device_input.hpp)。
   - 内容 定义 CUDA、SYCL 和主机模拟内存类别，定义 Q、K、交错 KV 批次与结果视图，检查形状、stride、对齐、地址范围和别名。

6. **设备图操作标记**
   - 文件 [include/solid/device_ops.hpp](../include/solid/device_ops.hpp)。
   - 内容 定义 Q 就绪、KV 就绪、交错 FP16 KV、mask 释放和 prefill 结果操作，校验张量形状及依赖。

7. **图节点构造**
   - 文件 [src/device_ops.cpp](../src/device_ops.cpp)。
   - 内容 创建自定义 GGML 节点及 CPU 对应实现，绑定操作标记，生成交错 FP16 KV 与保活依赖，接入原位 prefill 结果。

8. **CUDA 图操作**
   - 文件 [include/solid/device_ops.cuh](../include/solid/device_ops.cuh)。
   - 内容 实现就绪标量、F32 K/V 到 FP16 交错写入、mask 和结果复制，本地接入 Q/KV 终点计时标记。

9. **SYCL 图操作**
   - 文件 [include/solid/sycl_device_ops.hpp](../include/solid/sycl_device_ops.hpp)。
   - 内容 实现设备就绪标量、交错 FP16 写入、mask 与结果复制，本地接入 Q/KV 终点计时标记。

10. **SYCL 稀疏 Decode**
   - 文件 [include/solid/sycl_flash_decode.hpp](../include/solid/sycl_flash_decode.hpp)。
   - 内容 支持逐 query head mask 与交错 K/V stride，以 64 行切片执行在线 softmax，再合并分片结果，为 GGML 预留输出附属工作区。

11. **模型适配接口**
   - 文件 [include/solid/llama_adapter.hpp](../include/solid/llama_adapter.hpp)。
   - 内容 声明 AttentionAdapter、AttentionNode、错误容器、模型图接入、融合投影及本地投影计时接口。

12. **模型图适配**
   - 文件 [src/llama_adapter.cpp](../src/llama_adapter.cpp)。
   - 内容 接入 Llama/Qwen2 单序列 Q/K/V、设备选择器、prefill、稀疏 decode 和预取工作集，建立设备张量保活与 CPU 选择节点，传递首个错误，处理请求收尾。本地增加投影与就绪前沿计时绑定。

## 六、SSD 预取、读写与设备缓存

1. **块与租约身份**
   - 文件 [include/solid/block_identity.hpp](../include/solid/block_identity.hpp)。
   - 内容 定义 layer/head/block/version 身份和 slot/generation 租约，用于识别块版本及槽位复用。

2. **异步 I/O 协议**
   - 文件 [include/solid/async_reader.hpp](../include/solid/async_reader.hpp)。
   - 内容 定义读写提交、完成回执、取消、优先级、调度键、完成通知和统计，提供 liburing 与 I/O worker 工厂。

3. **liburing 读写**
   - 文件 [src/async_reader.cpp](../src/async_reader.cpp)。
   - 内容 实现异步 READ/WRITE 提交和 CQE 收割、原请求与取消处理、短传输结果核对、事件通知、提交深度和 I/O 统计。

4. **独立 I/O 所有者**
   - 文件 [src/io_worker.cpp](../src/io_worker.cpp)。
   - 内容 实现持久 I/O 线程、有界命令队列、请求提升、完成转发、取消排空和错误传递，隔离协调线程与底层 liburing 所有权。

5. **主机预取状态**
   - 文件 [include/solid/prefetch.hpp](../include/solid/prefetch.hpp)。
   - 内容 定义槽位状态和租约、预测历史、在途读取、版本修正、dirty 转发、回写回执与预取统计接口。

6. **历史预测与纠错**
   - 文件 [src/prefetch.cpp](../src/prefetch.cpp)。
   - 内容 按同层上一轮 selected 和当前 init/local 建立预测集合，正式选块后保留命中块、提升请求和补读缺块。处理尾块旧前缀、generation、原 READ 完成、源租约和聚合异步回写，生成 READ/H2D 派发顺序。

7. **设备工作集接口**
   - 文件 [include/solid/device_cache.hpp](../include/solid/device_cache.hpp)。
   - 内容 定义设备传输票据、当前行输入、槽位与 attention 读者接口，记录上传、命中、原槽前缀保留及共享内存字节。本地增加复制区间回执和就绪收割接口。

8. **设备槽位管理**
   - 文件 [src/device_cache.cpp](../src/device_cache.cpp)。
   - 内容 实现预测块上传、正式工作集核对、错误预测槽位回收和逐 query head mask。当前行支持 D2D、尾块原槽更新与缺口补齐，维护复制源及 attention 读者租约，本地把复制设备时长写入块任务。

9. **预取图协调接口**
   - 文件 [include/solid/ggml_prefetch.hpp](../include/solid/ggml_prefetch.hpp)。
   - 内容 提供提前选择、设备 decode、完成通知、预测开始、资源排空和统计接口，本地增加投影设备时长参数。

10. **GGML 传输与协调**
   - 文件 [src/ggml_prefetch.cpp](../src/ggml_prefetch.cpp)。
   - 内容 实现单层设备 bank、历史块上传、当前行安装、D2H 归档、后继层读取和异步回写的衔接，管理独立复制流及票据。SYCL 支持共享 bank 和独立队列，本地加入上传/归档计时、设备成本回填及完成票据收割。

11. **SYCL 后端桥接**
   - 文件 [include/solid/sycl_backend.hpp](../include/solid/sycl_backend.hpp)。
   - 内容 声明共享 KV 分配、GGML 队列访问、独立复制队列和事件接口，本地扩展复制区间读取。

12. **SYCL 独立复制队列**
   - 文件 [src/sycl/copy_queue.cpp](../src/sycl/copy_queue.cpp)。
   - 内容 在相同设备和 context 建立独立有序队列，封装复制、事件查询、等待与计算队列依赖，本地收集复制起止时间戳。

13. **SYCL 公共资源**
   - 文件 [include/solid/sycl_runtime.hpp](../include/solid/sycl_runtime.hpp)。
   - 内容 封装有序队列、设备 USM 分配、释放、尺寸溢出和借用指针归属检查，提供设备错误标志，本地加入复制队列 profiling 属性。

## 七、任务调度、完成通知与计时

1. **微任务数据与接口**
   - 文件 [include/solid/scheduler.hpp](../include/solid/scheduler.hpp)。
   - 内容 定义 host/SSD/H2D/D2H/GPU 资源、依赖、状态、时间、critical/LST、调度轨迹和持久 worker。本地增加设备时长优先的任务成本读取。

2. **DAG 与协调 Worker**
   - 文件 [src/scheduler.cpp](../src/scheduler.cpp)。
   - 内容 检查依赖图，计算关键路径和最迟开始时间，按 critical、I/O 和 LST 选择就绪任务，记录开始、完成、同步覆盖与失败，提供单任务容量 worker。本地关键路径使用设备回执成本。

3. **块任务接口**
   - 文件 [include/solid/block_stage.hpp](../include/solid/block_stage.hpp)。
   - 内容 定义逐块 READ/H2D 任务、派发顺序、就绪及汇合状态与统计，本地增加复制设备时长参数。

4. **块级派发与汇合**
   - 文件 [src/block_stage.cpp](../src/block_stage.cpp)。
   - 内容 为每个块建立 READ→H2D 依赖，生成 SSD 和复制调度键，记录原读、驻留、上传及全部块完成，本地使用设备复制时长更新成本。

5. **图完成绑定**
   - 文件 [include/solid/stream_completion.hpp](../include/solid/stream_completion.hpp)。
   - 内容 定义完成 token、参数区绑定和身份校验，增加捕获 epoch 与指定 epoch 通知接口。

6. **Attention 完成协议**
   - 文件 [include/solid/attention_events.hpp](../include/solid/attention_events.hpp)。
   - 内容 声明 arm、poll、边界证明、停止和通知接口，定义 CUDA/SYCL 完成事件工厂。

7. **完成状态管理**
   - 文件 [src/attention_events.cpp](../src/attention_events.cpp)。
   - 内容 维护每层 armed/done epoch 和全局消费状态，使用 eventfd 唤醒协调线程，处理设备通知、后继 Q/请求边界和停止。本地协议拒绝过期 epoch 通知。

8. **CUDA 完成发射**
   - 文件 [include/solid/cuda_completion.cuh](../include/solid/cuda_completion.cuh)。
   - 内容 封装普通及图捕获下的外部事件记录、attention 起止事件和流主机回调通知。

9. **CUDA 事件与投影计时**
   - 文件 [src/cuda/attention_events.cu](../src/cuda/attention_events.cu)。
   - 内容 实现 attention 起止事件、复制事件查询和设备耗时读取。本地增加可绑定矩阵乘法及就绪前沿的投影区间对象。

10. **SYCL 事件与投影计时**
   - 文件 [src/sycl/attention_events.cpp](../src/sycl/attention_events.cpp)。
   - 内容 使用 decode 结束事件和独立通知队列发布带 epoch 的完成消息，支持设备时间戳。本地增加投影区间对象和区间读取。

11. **本地设备区间协议**
   - 文件 [include/solid/device_timing.hpp](../include/solid/device_timing.hpp)。
   - 内容 新增设备区间接口、Q/KV 投影区间容器、GGML 参数区绑定及 CUDA 队列桥接声明。

12. **执行剖析接口**
   - 文件 [include/solid/profile.hpp](../include/solid/profile.hpp)。
   - 内容 定义算子和传输调用次数、字节数、耗时、prefill/decode 阶段及回调接口。

13. **执行剖析实现**
   - 文件 [src/profile.cpp](../src/profile.cpp)。
   - 内容 累计 GGML 算子与跨后端传输的分项结果，管理剖析回调和错误，输出对应阶段统计。

14. **线程布局接口**
   - 文件 [include/solid/thread_affinity.hpp](../include/solid/thread_affinity.hpp)。
   - 内容 声明物理 CPU 识别、线程绑定、I/O/协调核心预留和推理可用 CPU 集合。

15. **物理核心安排**
   - 文件 [src/thread_affinity.cpp](../src/thread_affinity.cpp)。
   - 内容 读取可用 CPU 与拓扑，校验核心选择，绑定 I/O 和协调线程，构建推理线程使用的 CPU 集合。

## 八、运行与结果工具

1. **运行监控与来源核对**
   - 文件 [scripts/run_baseline.py](../scripts/run_baseline.py)。
   - 内容 校验模型哈希、源码指纹及所需后端，管理运行状态、日志、RSS/cgroup 采样与原子结果，收尾所拥有进程。

2. **子进程管理**
   - 文件 [scripts/process_lifetime.py](../scripts/process_lifetime.py)。
   - 内容 统一进程组所有权、终止信号、超时、异常与中断清理，并等待子进程退出。

3. **数值结果对照**
   - 文件 [scripts/compare_logits.py](../scripts/compare_logits.py)。
   - 内容 比较固定输入和生成结果中的 logits，报告偏差和 token 一致性。

4. **选择回放配置**
   - 文件 [scripts/prepare_replay.py](../scripts/prepare_replay.py)。
   - 内容 整理固定选择轨迹及配套运行配置，保留模型、输入和轨迹来源。

5. **布局对照工具**
   - 文件 [scripts/layout_ablation.py](../scripts/layout_ablation.py)。
   - 内容 准备 separate/interleaved 与融合投影对照，比较来源、选择集合、数值和传输记录。

6. **预取对照配置**
   - 文件 [scripts/prepare_prefetch.py](../scripts/prepare_prefetch.py)。
   - 内容 生成同步和预取配置，固定输入、模型、选择及输出目录。

7. **预取结果核查**
   - 文件 [scripts/compare_prefetch.py](../scripts/compare_prefetch.py)。
   - 内容 对照数值和选择集合，检查主机/设备驻留轨迹中的身份、generation、复制与 attention 租约。

8. **调度结果核查**
   - 文件 [scripts/schedule_ablation.py](../scripts/schedule_ablation.py)。
   - 内容 准备三组调度配置，检查任务依赖、完成证明、块汇合和驻留结果。本地支持 Q/KV、上传及归档的设备成本。

9. **选择集合分析**
   - 文件 [scripts/selection_analysis.py](../scripts/selection_analysis.py)。
   - 内容 分析逐层选择历史的相似性、交集和集合特征。

10. **结果公共处理**
   - 文件 [scripts/experiment_common.py](../scripts/experiment_common.py)。
   - 内容 实现严格 JSON、数值字段校验、哈希来源、原子写入与统计汇总函数。

11. **实验进程隔离**
   - 文件 [scripts/experiment_runtime.py](../scripts/experiment_runtime.py)。
   - 内容 实现 CPU/内存条件、cgroup 限制和只读 fio 干扰进程管理，记录条件与收尾状态。

12. **实验矩阵工具**
   - 文件 [scripts/experiments.py](../scripts/experiments.py)。
   - 内容 展开模型、长度、预算、块大小和机制组合，管理 warmup、重复、续跑、完成核对、采集及报告。

13. **生成后端客户端**
   - 文件 [scripts/accuracy_client.py](../scripts/accuracy_client.py)。
   - 内容 封装 tokenizer 和固定输入生成调用，管理后端结果、超时与子进程生命周期。

14. **准确率任务入口**
   - 文件 [scripts/accuracy.py](../scripts/accuracy.py)。
   - 内容 准备和执行任务样本生成，核对模型、数据、token 和评分来源。

15. **OpenCompass 模型接口**
   - 文件 [scripts/opencompass_model.py](../scripts/opencompass_model.py)。
   - 内容 把仓库 tokenizer 和生成客户端接入 OpenCompass 的模型接口。

16. **准确率报告**
   - 文件 [scripts/accuracy_report.py](../scripts/accuracy_report.py)。
   - 内容 核对样本和评分来源、聚合任务指标并生成准确率对照结果。

17. **实验绘图**
   - 文件 [scripts/plot_experiments.py](../scripts/plot_experiments.py)。
   - 内容 从核对过的结果生成论文机制对照图，保存输入来源与图表对应信息。

18. **既有检查入口**
   - 文件 [scripts/check.sh](../scripts/check.sh)。
   - 内容 封装核心、CUDA 构建和 CPU 用例、核心 sanitizer 及 adapter sanitizer 目标。

19. **绘图依赖**
   - 文件 [requirements-experiments.txt](../requirements-experiments.txt)。
   - 内容 声明 matplotlib 的版本范围。

## 九、配置与输入文件

1. **模型来源锁定**
   - 文件 [configs/models.lock.json](../configs/models.lock.json)。
   - 内容 登记模型仓库版本、GGUF 分片名称、大小与 SHA256。

2. **原版模型配置**
   - 文件 [configs/qwen2.5-7b-q4_k_m.json](../configs/qwen2.5-7b-q4_k_m.json)。
   - 内容 指定 Qwen2.5-7B Q4_K_M 模型分片、F16 KV、生成和上下文参数。

3. **内存参考配置**
   - 文件 [configs/indram-reference.json](../configs/indram-reference.json)。
   - 内容 启用自定义内存 KV 与全量 attention 参考路径。

4. **内存稀疏配置**
   - 文件 [configs/indram-sparse.json](../configs/indram-sparse.json)。
   - 内容 启用内存 KV、动态代表选择及稀疏 decode。

5. **内存交错稀疏配置**
   - 文件 [configs/indram-sparse-interleaved.json](../configs/indram-sparse-interleaved.json)。
   - 内容 组合内存 KV、稀疏选择、交错布局与融合 K/V 投影。

6. **SSD 全量配置**
   - 文件 [configs/offload.json](../configs/offload.json)。
   - 内容 启用 SSD KV 与全量历史 attention。

7. **SSD 稀疏配置**
   - 文件 [configs/offload-sparse.json](../configs/offload-sparse.json)。
   - 内容 组合 SSD KV 和动态稀疏选择。

8. **SSD 交错配置**
   - 文件 [configs/offload-interleaved.json](../configs/offload-interleaved.json)。
   - 内容 组合 SSD、交错 KV 与融合投影。

9. **SSD 交错稀疏配置**
   - 文件 [configs/offload-sparse-interleaved.json](../configs/offload-sparse-interleaved.json)。
   - 内容 组合稀疏选择、SSD 交错 KV 与融合投影。

10. **历史预取配置**
   - 文件 [configs/offload-prefetch.json](../configs/offload-prefetch.json)。
   - 内容 启用 liburing、版本化槽位和历史选择预取。

11. **重叠调度配置**
   - 文件 [configs/offload-overlap.json](../configs/offload-overlap.json)。
   - 内容 启用提前 Q 选择、历史补读与 KV 生成重叠。

12. **同步复用配置**
   - 文件 [configs/offload-sync-reuse.json](../configs/offload-sync-reuse.json)。
   - 内容 在重叠路径启用完成证明复用和设备工作集交接。

13. **论文实验矩阵**
   - 文件 [configs/experiments/cuda-paper.json](../configs/experiments/cuda-paper.json)。
   - 内容 定义 CUDA 上下文长度、内存、机制消融、块大小和预算对照组合。

14. **SSD 干扰矩阵**
   - 文件 [configs/experiments/cuda-io.json](../configs/experiments/cuda-io.json)。
   - 内容 定义 CUDA 场景中的 SSD I/O 干扰条件及对照组合。

15. **OpenCompass 配置**
   - 文件 [configs/experiments/opencompass_model.py](../configs/experiments/opencompass_model.py)。
   - 内容 登记仓库模型适配类和生成配置。

16. **短输入样本**
   - 文件 [prompts/smoke.txt](../prompts/smoke.txt)。
   - 内容 提供运行入口使用的固定短文本。

## 十、已有检查文件

1. **配置检查**
   - 文件 [tests/config_test.cpp](../tests/config_test.cpp)。
   - 内容 覆盖字段、预算、路径、模型声明及非法配置。

2. **存储与参考计算检查**
   - 文件 [tests/core_test.cpp](../tests/core_test.cpp)。
   - 内容 覆盖 FP16、内存/文件 KV、短传输、追加读取和参考 attention。

3. **交错布局检查**
   - 文件 [tests/layout_test.cpp](../tests/layout_test.cpp)。
   - 内容 覆盖两种 KV 布局、文件偏移、读写内容和相关传输计数。

4. **融合投影检查**
   - 文件 [tests/projection_test.cpp](../tests/projection_test.cpp)。
   - 内容 对照独立与融合 K/V 的权重行、bias、输出和视图。

5. **选块协议检查**
   - 文件 [tests/ranker_test.cpp](../tests/ranker_test.cpp)。
   - 内容 覆盖多 head、GQA、代表发布、候选成本、并列及异常结果。

6. **代表共享用例**
   - 文件 [tests/representative_builder_cases.hpp](../tests/representative_builder_cases.hpp)。
   - 内容 提供窗口成熟、块定稿、种子接管、边界和排序的公共数据及断言。

7. **代表构造检查**
   - 文件 [tests/representative_builder_test.cpp](../tests/representative_builder_test.cpp)。
   - 内容 调用共享用例检查设备构造协议与参考结果。

8. **设备输入协议检查**
   - 文件 [tests/device_input_test.cpp](../tests/device_input_test.cpp)。
   - 内容 覆盖借用 Q/K 的形状、stride、类型、地址和会话顺序。

9. **图操作检查**
   - 文件 [tests/device_ops_test.cpp](../tests/device_ops_test.cpp)。
   - 内容 覆盖就绪标量、交错转换、mask/结果复制及图依赖。

10. **Prefill 共享用例**
   - 文件 [tests/tiled_prefill_cases.hpp](../tests/tiled_prefill_cases.hpp)。
   - 内容 提供因果前缀、分块、维度、GQA、边界和数值对照数据。

11. **分块 Prefill 检查**
   - 文件 [tests/tiled_prefill_test.cpp](../tests/tiled_prefill_test.cpp)。
   - 内容 核查有界容量、在线 attention、源生命周期和图结果。

12. **设备 Prefill 协议检查**
   - 文件 [tests/device_prefill_test.cpp](../tests/device_prefill_test.cpp)。
   - 内容 核查借用设备 Q/KV、结果、历史分区和输入约束。

13. **设备 Prefill 图用例**
   - 文件 [tests/device_prefill_graph_cases.hpp](../tests/device_prefill_graph_cases.hpp)。
   - 内容 提供批次、图结果、代表状态和 decode 交接公共用例。

14. **设备 Prefill 图检查**
   - 文件 [tests/device_prefill_graph_test.cpp](../tests/device_prefill_graph_test.cpp)。
   - 内容 调用公共用例核查设备输入和输出图的接入。

15. **Stride 图检查**
   - 文件 [tests/graph_test.cpp](../tests/graph_test.cpp)。
   - 内容 对照 GGML strided attention 与参考计算。

16. **Gather 图检查**
   - 文件 [tests/gather_test.cpp](../tests/gather_test.cpp)。
   - 内容 核查槽位 K/V、mask 和稀疏 Flash Attention 输出。

17. **预取替身与公共数据**
   - 文件 [tests/prefetch_fakes.hpp](../tests/prefetch_fakes.hpp)。
   - 内容 提供受控读取、块数据、会话和预取协议检查辅助。

18. **延迟复制替身**
   - 文件 [tests/deferred_copies.hpp](../tests/deferred_copies.hpp)。
   - 内容 提供待事件或 drain 后写入的 CPU 复制队列，记录源持有、提交与完成。

19. **预测与驻留检查**
   - 文件 [tests/prefetch_test.cpp](../tests/prefetch_test.cpp)。
   - 内容 覆盖历史预测、纠错、版本、取消和槽位租约。

20. **当前设备行检查**
   - 文件 [tests/current_device_test.cpp](../tests/current_device_test.cpp)。
   - 内容 覆盖 D2D 当前行、尾块前缀复用、缺口复制与失败排空。

21. **预取计算图检查**
   - 文件 [tests/prefetch_graph_test.cpp](../tests/prefetch_graph_test.cpp)。
   - 内容 覆盖多层、GQA、三种调度、设备输入、融合输入、完成事件和参考数值。

22. **错误清理检查**
   - 文件 [tests/prefetch_failure_test.cpp](../tests/prefetch_failure_test.cpp)。
   - 内容 注入图与复制错误，检查原错误保留及后续 attention 前的复制排空。

23. **微任务检查**
   - 文件 [tests/scheduler_test.cpp](../tests/scheduler_test.cpp)。
   - 内容 覆盖依赖、关键路径、LST、任务优先级、状态和 worker 错误。

24. **块任务检查**
   - 文件 [tests/block_stage_test.cpp](../tests/block_stage_test.cpp)。
   - 内容 覆盖逐块 READ/H2D 排序、驻留、完成汇合和状态错误。

25. **完成协议检查**
   - 文件 [tests/attention_events_test.cpp](../tests/attention_events_test.cpp)。
   - 内容 覆盖 epoch、设备通知、主机边界、停止和消费规则。

26. **原生完成协议检查**
   - 文件 [tests/native_completion_test.cpp](../tests/native_completion_test.cpp)。
   - 内容 使用受控通知核查 FFN 期间后继上传、Q 边界接管、取消与 worker 错误。

27. **I/O 所有权检查**
   - 文件 [tests/io_worker_test.cpp](../tests/io_worker_test.cpp)。
   - 内容 覆盖有界命令、提升、读写完成、取消和独立线程收尾。

28. **liburing 检查**
   - 文件 [tests/uring_test.cpp](../tests/uring_test.cpp)。
   - 内容 使用临时文件覆盖读写、取消、通知和排空。

29. **剖析检查**
   - 文件 [tests/profile_test.cpp](../tests/profile_test.cpp)。
   - 内容 核查节点及传输回调、阶段与统计记录。

30. **CUDA 选块检查程序**
   - 文件 [tests/cuda_ranker_test.cpp](../tests/cuda_ranker_test.cpp)。
   - 内容 调用设备选择器对照代表构造、评分、预算和参考结果。

31. **CUDA Prefill 检查程序**
   - 文件 [tests/cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)。
   - 内容 调用 CUDA 后端执行分块 attention 公共用例。

32. **CUDA Prefill 图检查程序**
   - 文件 [tests/cuda_device_prefill_test.cpp](../tests/cuda_device_prefill_test.cpp)。
   - 内容 覆盖真实设备借用输入、结果图和 decode 交接。

33. **CUDA 输入检查程序**
   - 文件 [tests/cuda_device_input_test.cu](../tests/cuda_device_input_test.cu)。
   - 内容 覆盖真实设备 Q/K 输入和设备图操作。

34. **CUDA 完成检查程序**
   - 文件 [tests/cuda_attention_events_test.cu](../tests/cuda_attention_events_test.cu)。
   - 内容 覆盖设备事件、通知和 CUDA 图重放。

35. **调度轨迹对照**
   - 文件 [tests/audit_schedule_graph.py](../tests/audit_schedule_graph.py)。
   - 内容 核查合成 GGML 轨迹、C++ 指标、依赖、归档和设备驻留的一致性。

36. **准确率工具检查**
   - 文件 [tests/test_accuracy.py](../tests/test_accuracy.py)。
   - 内容 覆盖任务准备、生成输入、客户端结果和来源检查。

37. **评分报告检查**
   - 文件 [tests/test_accuracy_report.py](../tests/test_accuracy_report.py)。
   - 内容 覆盖评分来源、样本完整性和指标汇总。

38. **生成器检查**
   - 文件 [tests/test_adapter_generator.py](../tests/test_adapter_generator.py)。
   - 内容 覆盖固定上游锚点和生成的图、后端接入内容。

39. **数值对照工具检查**
   - 文件 [tests/test_comparison.py](../tests/test_comparison.py)。
   - 内容 覆盖 logits、token 与输入来源差异。

40. **矩阵与结果检查**
   - 文件 [tests/test_experiments.py](../tests/test_experiments.py)。
   - 内容 覆盖矩阵展开、运行条件、续跑、完成核对、汇总和绘图输入。

41. **启动器检查**
   - 文件 [tests/test_launcher.py](../tests/test_launcher.py)。
   - 内容 覆盖配置来源、特性、模型哈希、运行记录和监控异常。

42. **布局工具检查**
   - 文件 [tests/test_layout_ablation.py](../tests/test_layout_ablation.py)。
   - 内容 覆盖对照配置、固定集合和布局结果核查。

43. **预取工具检查**
   - 文件 [tests/test_prefetch_tools.py](../tests/test_prefetch_tools.py)。
   - 内容 覆盖预取配置及驻留轨迹的身份、generation 和租约。

44. **子进程清理检查**
   - 文件 [tests/test_process_lifetime.py](../tests/test_process_lifetime.py)。
   - 内容 覆盖超时、中断、错误和进程组收尾。

45. **回放工具检查**
   - 文件 [tests/test_replay.py](../tests/test_replay.py)。
   - 内容 覆盖选择轨迹、配置与固定输入来源。

46. **调度工具检查**
   - 文件 [tests/test_schedule_tools.py](../tests/test_schedule_tools.py)。
   - 内容 覆盖任务依赖、完成凭据、设备区间与关键路径成本。

47. **选择分析检查**
   - 文件 [tests/test_selection_analysis.py](../tests/test_selection_analysis.py)。
   - 内容 覆盖集合相似度、历史和输入有效性。

48. **分词检查数据**
   - 文件 [tests/tokenize_fixture.py](../tests/tokenize_fixture.py)。
   - 内容 生成小型词表夹具并核对词表模式分词结果。

49. **矩阵配置检查**
   - 文件 [tests/validate_experiment_configs.py](../tests/validate_experiment_configs.py)。
   - 内容 展开实验配置并调用运行器解析，核对组合与输出。

## 十一、文档合并

以下文件的实现内容已归入上面的模块条目，仓库中删除对应旧文件。

- `docs/CHECKS.md`
- `docs/DECISIONS.md`
- `docs/DESIGN.md`
- `docs/DEVELOPMENT.md`
- `docs/ENGINEERING_REVIEW.md`
- `docs/PAPER_IMPLEMENTATION_AUDIT.md`
- `docs/REFERENCES.md`
- `docs/ROADMAP.md`
- `docs/SSD_PATH.md`
- `docs/START_HERE.md`
- `docs/TEST_PLAN.md`
- `docs/UPSTREAM_MAP.md`
- `docs/implementation/ASYNC_WRITEBACK.md`
- `docs/implementation/BLOCK_EVENT_SCHEDULER.md`
- `docs/implementation/BLOCK_READ_PIPELINE.md`
- `docs/implementation/CURRENT_DEVICE_KV.md`
- `docs/implementation/DEVICE_INPUTS.md`
- `docs/implementation/GPU_PREFILL.md`
- `docs/implementation/GPU_REPRESENTATIVES.md`
- `docs/implementation/GPU_SELECTION.md`
- `docs/implementation/IO_WORKERS.md`
- `docs/implementation/KV_ARCHIVE.md`
- `docs/implementation/NATIVE_COMPLETION.md`
- `docs/implementation/PREDICTION_DISPATCH.md`
- `docs/implementation/PREFILL_DEVICE_INPUTS.md`
- `docs/implementation/SYCL_DEVICE_INPUTS.md`
- `docs/implementation/SYCL_NATIVE_COMPLETION.md`
- `docs/implementation/SYCL_PREFILL.md`
- `docs/implementation/SYCL_SELECTION.md`
- `docs/implementation/SYCL_SHARED_KV.md`
- `docs/phases/PHASE_0.md`
- `docs/phases/PHASE_1.md`
- `docs/phases/PHASE_2.md`
- `docs/phases/PHASE_3.md`
- `docs/phases/PHASE_4.md`
- `docs/phases/PHASE_5.md`
- `docs/phases/README.md`
