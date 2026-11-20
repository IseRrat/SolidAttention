# 实现记录

更新日期为 2026-11-20。按论文模块记录已写入代码的功能，后续修改同步更新对应条目。

## 一、KV Consolidator

对应论文 §4、图 7、图 8。

1. **合并 K/V 投影权重**

   初始化时按 KV head 交错拼接 K/V 权重和 bias，用一次矩阵乘法生成交错结果。runner 的融合配置将原始 K/V 权重和 bias 加载到 CPU 侧，按层设备分配融合权重，避免在 GPU 同时保存原始与融合的两份权重。加载时保留 GGUF 原始行编码。相同量化类型每个 head 的连续权重行整段读取、写入，主机中转区只持有一份 head 数据。混合类型逐行解码为 F32，再写入融合权重。已记录融合分配、源权重总字节、其中的主机字节及初始化耗时。

   [projection.cpp](../src/projection.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[baseline.cpp](../src/baseline.cpp)。

2. **交错 KV 与 Attention 视图**

   设备批次采用 [token, KV head, K/V, D] 的 FP16 排列，SSD 单 head 块采用 [block token, K/V, D] 排列。已生成 K/V stride 视图，接入 attention，并保留 separate 布局供机制对照。

   [kv_store.cpp](../src/kv_store.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)。

3. **RoPE 写回与 FP16 转换**

   已实现 RoPE 后 K 写回交错投影结果，以及 CUDA、SYCL 上的 F32 K/V 到 FP16 交错写入。设备图操作包含当前 Q/KV 的依赖与保活标记。

   [projection.cpp](../src/projection.cpp)、[device_ops.cpp](../src/device_ops.cpp)、[device_ops.cuh](../include/solid/device_ops.cuh)、[sycl_device_ops.hpp](../include/solid/sycl_device_ops.hpp)。

4. **SSD 块存储**

   已实现按 layer、KV head、逻辑块定位的文件格式、4096 字节文件头、对齐偏移、有效行数及块版本。支持连续追加、历史读取、短读写补齐和磁盘/待写数据合并读取。

   [kv_store.hpp](../include/solid/kv_store.hpp)、[kv_store.cpp](../src/kv_store.cpp)。

## 二、动态稀疏 Attention

对应论文 §3.1、图 4。

1. **三类上下文与预算**

   已实现 init、local、selected 集合，按 token 去重，末块按剩余预算截取。短输入自动预算为长度的四分之一，达到 4096 token 后取 1024。长输入默认 init=32、local=480、selected 最多 512。

   [attention.cpp](../src/attention.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)。

2. **代表向量构造**

   已实现后续 local 窗口评分、GQA 查询均值、块内 top-R 和完整块定稿。代表同分按绝对位置排序。支持 CPU 状态种子和定稿代表增量发布。CUDA/SYCL 接管后，已发布代表的 K 数据由设备 bank 保存，主机释放对应载荷并保留位置、分数和定稿状态；后续定稿也采用这一归属。逐块评分核查按需回读单块，保持原累加顺序，正常选块直接使用设备 bank。

   [attention.cpp](../src/attention.cpp)、[representative_builder.cpp](../src/representative_builder.cpp)、[representative_math.hpp](../include/solid/representative_math.hpp)。

3. **CUDA 代表构造和选块**

   已实现设备代表 bank、pending K 环、窗口分数与部分块 top-R。设备 kernel 覆盖均值、窗口更新、定稿、FP64 块评分和逐 head 预算 top-k。块评分与窗口更新按 warp 分配任务，相邻 lane 读取相邻 Q/K 维度并计算乘积，lane 0 按原维度顺序累计，保留代表、GQA 和维度的累加次序。支持不足 32 维的尾段及 grid-stride 任务分配。设备 Q/K、CPU 状态接管和 prefill 批量代表均使用此路径。块同分按逻辑 ID 排序，选择数量按 token 成本控制。

   [block_ranker.cu](../src/cuda/block_ranker.cu)、[block_ranker.hpp](../include/solid/block_ranker.hpp)。

4. **SYCL 评分和选块**

   已写设备代表 bank、窗口更新、定稿、评分和预算归约，接入图内 Q/K 及批量构造。按设备能力使用 FP64 或双 FP32 分量。块评分与窗口评分更新由 64 个同组线程按连续维度读取 Q/K、计算乘积，通过组内暂存交给线程 0 按原维度顺序累加；覆盖不足 64 维的尾段，保留代表、GQA 和维度的累加次序。定稿前将累计分数除以 local，与 CPU、CUDA 的代表种子和分数回填使用相同口径。

   [block_ranker.cpp](../src/sycl/block_ranker.cpp)。

5. **稀疏 Decode**

   已实现物理槽位对应的逐 query head mask。CUDA 使用上游 strided Flash Attention；SYCL 内核以 64 行切片执行 FP32 在线 softmax，再合并结果，支持交错 K/V stride。

   [llama_adapter.cpp](../src/llama_adapter.cpp)、[sycl_flash_decode.hpp](../include/solid/sycl_flash_decode.hpp)。

## 三、Speculative Prefetcher

对应论文 §5.1、§5.2、图 10。

1. **历史选择预测**

   已按同层上一 decode 轮 selected 历史，加上当前 init/local 建立预测工作集。每个预测请求带块身份、所需版本和读取/上传派发键。

   [prefetch.cpp](../src/prefetch.cpp)、[prefetch.hpp](../include/solid/prefetch.hpp)。

2. **正式选择纠错**

   已实现命中块保留、在途请求提升、缺块补读和错误预测回收。尾块版本变化会处理在途旧前缀并补齐缺口。正式纠错优先于预测；同一前沿内先比较 critical 和 LST，相同时 selected 优先于 init/local。SSD 队列与上传派发使用同一组排序键。

   [prefetch.cpp](../src/prefetch.cpp)、[async_reader.hpp](../include/solid/async_reader.hpp)。

3. **物理槽位原地覆盖**

   已实现逻辑块到设备槽位映射、命中原槽保留、错误预测槽位回收和对应 mask。当前设备行支持 D2D、尾块前缀复用及主机缺口补齐。

   [device_cache.cpp](../src/device_cache.cpp)、[device_cache.hpp](../include/solid/device_cache.hpp)。

4. **主机/设备缓冲保护**

   已维护 layer/head/block/version 身份及 slot/generation 租约。槽位记录写入者和读者数量，读取中、可用、被读取及待回收状态分别处理。取消 READ 后，槽位保持待回收，收到原请求完成回执并释放全部读者后才允许覆盖；取消操作自身的回执只更新取消结果。GPU 复制持有主机源和设备目标，attention 持有设备工作集；共享内存路径同时持有对应主机槽位。尾块更新递增 generation，保留已完成的旧前缀，只填入新增行。

   请求结束时先停止设备完成通知的后续推进，等待协调任务、计算与复制完成，再释放 attention 和历史块读者、完成已提交 WRITE。会话结束排空预测 READ 和取消请求，写出剩余聚合缓冲。

   [block_identity.hpp](../include/solid/block_identity.hpp)、[prefetch.cpp](../src/prefetch.cpp)、[device_cache.cpp](../src/device_cache.cpp)、[async_reader.cpp](../src/async_reader.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[session.cpp](../src/session.cpp)。

## 四、SSD-aware Scheduler

对应论文 §6.1、§6.2、图 11、算法 1。

1. **层与块微任务依赖**

   已建立 Q 前沿、选择、KV 生成、补读、历史上传、当前行安装、attention、归档、回写和下一层预取的层图。逐块建立 READ→H2D→汇合依赖，计算 critical/LST，记录开始、完成与同步证明。

   [scheduler.cpp](../src/scheduler.cpp)、[block_stage.cpp](../src/block_stage.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)。

2. **Q 前沿与提前补读**

   已将设备 Q 就绪标量和 CPU 前沿接入模型图。overlap 与 overlap+sync_reuse 的前沿通过异步设备入口把选择交给持久协调线程后返回，GPU 可继续 KV 投影与交错转换。协调线程完成代表更新、选择和历史补读，KV 就绪处收割其结果。异步入口借用的 Q 由图依赖保留至 KV 汇合或排空结束，原同步设备入口在返回前消费 Q。选择异常在 KV 前沿传递。任务图把选择和 KV 生成作为 Q 就绪后的两个分支。naive 保留串行路径。

   [llama_adapter.cpp](../src/llama_adapter.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)。

3. **当前行安装与主机归档**

   已将当前设备行装入 bank，保留原尾块前缀，只复制缺口。安装后 attention 和 D2H 归档分别推进，归档追加后提交聚合回写，多层时可提前派发下一层 SSD 读取。最后一层到达上下文容量末尾时，完成后继预取任务的收尾记录，停止向容量外的位置发起预测，当前 attention、归档及回写照常完成。

   设备 Q 提前选择入口支持同步、异步调用，并显式指定后续 KV 的归档方式。已下载到主机的 FP16 KV 采用“下载完成→追加→当前行安装”的依赖，进入 FP16 追加函数；设备 KV 采用“当前行安装→下载归档→追加”的依赖，默认模型图使用这一方式。两种入口均保留选择位置、输入类型和归档方式的一致性检查。

   [ggml_prefetch.hpp](../include/solid/ggml_prefetch.hpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[session.cpp](../src/session.cpp)、[device_cache.cpp](../src/device_cache.cpp)。

4. **Attention 完成通知**

   CUDA 已接入原生事件和图捕获/重放通知；SYCL 已接入结束事件和带 epoch 的独立队列通知。协调线程据此释放旧工作集读者并推进后继上传，后继 Q 与请求边界也具有完成证明。早于后继边界的上传次数，以上传开始派发时的边界状态为准。

   [attention_events.cpp](../src/attention_events.cpp)、[attention_events.cu](../src/cuda/attention_events.cu)、[cuda_completion.cuh](../include/solid/cuda_completion.cuh)、[attention_events.cpp](../src/sycl/attention_events.cpp)。

5. **统一内存和复制队列**

   SYCL 已为支持共享 USM 的集成 GPU 分配共享 KV bank，SSD READ 与 dirty 数据转发直接填入该 bank 的历史槽位，attention 使用同一地址。主机释放独立块载荷，历史块安装只登记槽位和完成票据。共享内存的所有权随块视图保留，覆盖原 I/O 完成及排空过程。

   共享槽位在 attention 期间同时持有主机和设备读者。下一层预测先使用空闲槽位发起 SSD 读取；没有空位的块保留在有界预测计划中，原读者释放后继续派发。正式选择接管时清除尚未派发的预测，按纠错顺序加载所需块。每个 KV head 另保留一个当前块，写入当前设备行并复用尾块前缀。独立复制队列处理当前行及其历史缺口，具有事件查询和计算队列依赖。

   CUDA 及独立显存的异步复制路径将主机块直接绑定到上传所用的固定内存 bank，SSD READ 和 dirty 转发直接填充上传源。历史块与尾块前缀上传复用该地址，原复制票据持有源租约，省去普通内存到固定内存的中间复制。固定内存计入复制缓冲占用。

   [ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[copy_queue.cpp](../src/sycl/copy_queue.cpp)、[sycl_backend.hpp](../include/solid/sycl_backend.hpp)。

6. **设备任务计时**

   已写 Q/KV 投影区间、历史块上传及 KV 归档区间回执，DAG 使用设备时长更新成本，调度轨迹核查采用同一口径。CUDA 用原生事件记录，SYCL 使用设备支持的时间戳或复制队列 profiling。复制票据完成时收集区间，后继层使用更新后的成本。

   [device_timing.hpp](../include/solid/device_timing.hpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[scheduler.cpp](../src/scheduler.cpp)、[schedule_ablation.py](../scripts/schedule_ablation.py)。

## 五、论文系统的工程接入

对应论文 §7。

1. **llama.cpp 模型图接入**

   已在固定上游接入 Llama/Qwen2 单序列的自定义 KV、attention、选择和预取。模型输出投影、残差、FFN 及量化矩阵乘法继续使用上游实现。构建目录生成适配图与设备后端文件。连续单 token decode 接入上游计算图复用：各层保持固定 bank、mask 和节点，输入阶段更新位置并启动预取；复用检查覆盖逐层连续位置、单序列与容量边界。prefill 切换到 decode 时重新构图。

   [llama_adapter.hpp](../include/solid/llama_adapter.hpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[generate_adapter.py](../scripts/generate_adapter.py)、[Adapter.cmake](../cmake/Adapter.cmake)。

2. **liburing 与独立线程**

   已实现 READ/WRITE、CQE、完成通知、请求提升、取消排空及提交深度控制。I/O 所有者与协调线程持久化，支持有界队列、错误传播和物理核心预留。

   [async_reader.cpp](../src/async_reader.cpp)、[io_worker.cpp](../src/io_worker.cpp)、[scheduler.cpp](../src/scheduler.cpp)、[thread_affinity.cpp](../src/thread_affinity.cpp)。

3. **每层聚合写**

   已实现默认 32 KiB 每层写缓冲、dirty 数据转发、异步 WRITE 打包区和提交版本更新。源数据持有至全部原 WRITE 完成。

   [kv_store.cpp](../src/kv_store.cpp)、[prefetch.cpp](../src/prefetch.cpp)。

4. **长输入 Prefill**

   已实现有界 Q/KV tile、因果在线 softmax、历史分块读取、当前设备 KV、设备结果和 decode 代表状态交接。CUDA 每个 CTA 的四个 warp 处理同一 head 的四个连续 query，共用八行 K/V 共享面板。最大维度 1024 时面板占 32 KiB，包含尾 query、因果边界及非有限分数的统一 barrier 处理。SYCL 使用逐 query head 的分块在线累加。

   [tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[tiled_prefill.cpp](../src/sycl/tiled_prefill.cpp)、[session.cpp](../src/session.cpp)。

5. **运行与结果记录**

   已写模型和源码哈希、配置检查、固定 token/选择回放、运行监控、数值输出和分项内存/I/O 统计。已写实验矩阵、准确率接口、结果核查及绘图工具。

   [baseline.cpp](../src/baseline.cpp)、[config.cpp](../src/config.cpp)、[run_baseline.py](../scripts/run_baseline.py)、[experiments.py](../scripts/experiments.py)、[accuracy.py](../scripts/accuracy.py)、[plot_experiments.py](../scripts/plot_experiments.py)。
