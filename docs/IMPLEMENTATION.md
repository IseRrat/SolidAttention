# 实现记录

更新日期为 2026-11-25。

## 一、KV Consolidator

对应论文 §4、图 7、图 8。

1. **合并 K/V 投影权重**

   初始化时按 KV head 交错拼接 K/V 权重和 bias，用一次矩阵乘法生成交错结果。原始 K/V 权重和 bias 加载在 CPU 侧，融合权重按层放到计算设备。相同量化类型按 head 整段复制原始 GGUF 编码，兼容配置的混合类型逐行解码为 F32。CUDA 对照配置锁定 Qwen 官方 Q4_0 的两个文件哈希，模型加载后检查每层 K/V 投影权重均为 Q4_0。结果记录矩阵权重的类型分布，汇总核对融合后仍为 Q4_0。KV 数据保持 FP16。

   [projection.cpp](../src/projection.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[baseline.cpp](../src/baseline.cpp)、[config.hpp](../include/solid/config.hpp)、[config.cpp](../src/config.cpp)、[qwen2.5-7b-q4_0.json](../configs/qwen2.5-7b-q4_0.json)、[experiment_common.py](../scripts/experiment_common.py)。

2. **交错 KV 与 Attention 视图**

   设备批次采用 [token, KV head, K/V, D] 的 FP16 排列，SSD 单 head 块采用 [block token, K/V, D] 排列。已生成 K/V stride 视图，接入 attention，并保留 separate 布局供机制对照。

   [kv_store.cpp](../src/kv_store.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)。

3. **RoPE 写回与 FP16 转换**

   已实现 RoPE 后 K 写回交错投影结果，以及 CUDA、SYCL 上的 F32 K/V 到 FP16 交错写入。设备图操作包含当前 Q/KV 的依赖与保活标记。

   [projection.cpp](../src/projection.cpp)、[device_ops.cpp](../src/device_ops.cpp)、[device_ops.cuh](../include/solid/device_ops.cuh)、[sycl_device_ops.hpp](../include/solid/sycl_device_ops.hpp)。

4. **SSD 块存储**

   SolidAttention 以 layer、KV head、交织物理块定位。两个 SSD 基线分别为每个 layer、KV head 保存连续 K 平面和 V 平面，文件头记录该布局。支持按行区间读取以及已提交数据和待写尾部的合并。纯卸载可用每个 head 两笔逻辑读取回已提交历史，稀疏卸载可以合并相邻命中行。交织布局沿用块版本和按块请求。

   [kv_store.hpp](../include/solid/kv_store.hpp)、[kv_store.cpp](../src/kv_store.cpp)。

## 二、动态稀疏 Attention

对应论文 §3.1、图 4。

1. **三类上下文与预算**

   两条稀疏实验路径共用 init、local 和 selected。init 之后的连续 token 构成逻辑历史块。当前步退出近期窗口且尚未进入候选块的尾部仍参与本步 attention，完整块从下一步进入候选。选中的整块按绝对位置映射到文件物理块。开头、近期及尚未成块的尾部先占用预算，其余向下取整为可选整块。短输入预算为输入长度的四分之一，达到 4096 token 后取 1024，长输入默认 init=32、local=480。两条路径都用 32 token 的检索块，并拒绝为同一对照套件里的稀疏基线单独改预算。

   [attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[session.cpp](../src/session.cpp)、[prefetch.cpp](../src/prefetch.cpp)、[experiments.py](../scripts/experiments.py)、[ranker_test.cpp](../tests/ranker_test.cpp)、[test_experiments.py](../tests/test_experiments.py)。

2. **代表向量构造**

   CUDA 对照配置下，两条稀疏路径按 query head 累加联合 softmax 中属于近期窗口的实际概率。同一组 query heads 映射到同一个 KV head 的待定 K 行。每批近期概率先转成 FP16，再加到 FP16 累计分数上，CPU 参考与 CUDA 设备更新遵循相同的两次舍入，沿用原版 `append_global` 的分数类型。历史块定稿时，每个 query head 取 4 个代表 Key 并求均值，设备排名只保存均值向量。多头分数汇总成一个块分数，块和代表同分时分别按逻辑块 ID 和绝对位置排序。原配置的局部点积建索引由低层兼容接口继续提供。运行结果记录代表评分与位置编码方式。

   [attention.cpp](../src/attention.cpp)、[representative_builder.cpp](../src/representative_builder.cpp)、[representative_math.hpp](../include/solid/representative_math.hpp)、[block_ranker.hpp](../include/solid/block_ranker.hpp)、[session.hpp](../include/solid/session.hpp)、[baseline.cpp](../src/baseline.cpp)。

3. **CUDA 代表构造和选块**

   设备维护代表 bank、pending K 环和 FP16 窗口分数。完整块退出近期窗口后，每个“历史块、query head”对应一个 CUDA 线程块，独立执行块内 Top-K。32-token 实验块及不超过 128-token 的兼容块调用 CUB 稳定基数排序；更大的兼容块调用 CUB 并行归约选出代表。后续 gather/mean 内核并行读取代表 Key、计算各维均值，生成 FP16 块代表。待定环保留尚未完整退出窗口的块，代表构造按完整块提交，省去逐 token 插入及搬移代表 Key。位置、分数和状态通过一次分配的页锁定回执缓冲返回主机，同步、异步入口复用该缓冲。

   两条稀疏实验路径使用相同的 CUDA 评分与 Top-K。检索 Q 保留 llama.cpp 的 F32 计算类型，组内均值、跨头块点积和排序分数使用 FP32；FP16 代表 bank 在计算时提升为 FP32。每个线程块并行归约一个历史块的跨头点积，共享 Top-K 只回传一份块编号。已就绪的全部候选放得进预算时直接使用。原有逐头 FP64 评分及增量代表构造接口继续提供给既有调用。各选块接口同分时按逻辑块 ID 排序。

   CPU 的 `mean_infllm_query` 和共享块评分使用相同的 F32 输入与累加规则；`session.cpp` 的主机输入 Prefill 与 `tiled_prefill.cu` 的设备输入 Prefill 分别接入对应均值入口。InfLLM CUDA 结果中的 ranker 名称为 `cuda_infllm_fp32_topk`。原版 `calc_block_topk` 跟随模型张量类型，上游 `benchmark/pred.py` 加载 BF16 模型。真实模型核验在第 0 层、位置 1024 确认过：有限的 FP32 块分数压成 FP16 后溢出。评分入口已去掉这项额外转换，KV、代表 bank 和近期概率累计仍为 FP16。

   共享选块且候选块 token 成本相同时，`block_ranker.cu` 复用上游 llama.cpp CUDA argsort 使用的 CUB `DeviceRadixSort::SortPairsDescending`，对完整跨头块分数排序后取足以覆盖预算的前缀。候选编号按递增顺序提交，稳定排序保留同分时较小编号优先的规则。排序输出和临时工作区按最大候选数一次分配，持续调用复用，释放时统一归还；显存统计包括这部分分配。逐头或不同 token 成本的兼容入口沿用既有预算归约。两条稀疏实验路径使用相同的 CUB 入口。

   `cuda_ranker_test.cpp` 用 31、513、4,096、8,192 个候选块，覆盖同分、正反评分、不同预算及等成本和变成本，与独立的稳定全排序结果核对，并检查持续选块无新增显存分配。服务器 CUDA 评分选块检查及 15 组原版 InfLLM 检索对照通过。[block_ranker.cu](../src/cuda/block_ranker.cu)、[cuda_ranker_test.cpp](../tests/cuda_ranker_test.cpp)。

   [attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[session.cpp](../src/session.cpp)、[tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[block_ranker.cu](../src/cuda/block_ranker.cu)、[block_ranker.hpp](../include/solid/block_ranker.hpp)。

   CUDA Prefill 的均值与排名之间采用设备事件衔接。`mean_sparse_query_async` 在 attention 队列生成 FP32 均值，记录完成事件，返回均值指针和事件；选块器在评分队列插入该事件依赖，随后执行块点积和 Top-K，在所选编号回传处收取结果。模型推理调用这条异步均值入口，省去均值阶段单独的 CPU 等待。输入 Q、均值缓冲和事件由生产者持有，覆盖排名使用期；候选直接全部采用时，后续同队列 attention 完成输入读取。`DeviceQueryWithEvent` 单独表达这项依赖，普通 `DeviceQuery` 继续要求输入已完成生产。同步均值接口按均值事件收取结果。

   结果中的 `prefill_mean_events`、`prefill_mean_host_waits` 分别记录均值事件和同步均值收取次数，`query_ready_event_waits` 记录插入评分队列的 GPU 事件依赖次数。CUDA 对照用例分别调用同步、异步均值排名，核对选块一致，并检查普通候选、空候选两条路径的事件依赖及 CPU 均值等待计数。代码和 CUDA 对照用例已编译，原版 CPU 对照及 54 项 CPU 回归通过。

   [device_input.hpp](../include/solid/device_input.hpp)、[block_ranker.hpp](../include/solid/block_ranker.hpp)、[tiled_prefill.hpp](../include/solid/tiled_prefill.hpp)、[attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[session.cpp](../src/session.cpp)、[block_ranker.cu](../src/cuda/block_ranker.cu)、[tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[baseline.cpp](../src/baseline.cpp)、[cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)。

4. **SYCL 评分和选块**

   设备维护代表 bank、窗口更新、定稿和排名，接入图内 Q/K 及批量构造。按设备能力使用 FP64 或双 FP32 分量。共享选择模式先评分、聚合所有头的分数，再执行一份 Top-K。定稿前将累计分数除以 local，与 CPU、CUDA 的代表种子和分数回填使用同一规则。

   [block_ranker.cpp](../src/sycl/block_ranker.cpp)。

5. **稀疏 Decode**

   CUDA 对照配置下，两条稀疏路径保存 RoPE 前的 K，将 Query 固定旋转到窗口末尾。近期 Key 按距当前 token 的位置旋转，历史、初始和过期尾部的 Key 使用原值。GGML 矩阵乘法算 logits，联合 softmax 后通过矩阵乘法得到 attention 输出。设备槽位同时记录近期位置，输出完成后把近期 token 的概率交给代表索引。SolidAttention 的物理槽位和 SSD 预取仍由自身调度器管理。兼容配置继续使用原有 strided Flash Attention 接口。`generate_adapter.py` 生成的位置输入回调只向仍在图中、已经分配缓冲的绝对位置张量写入数据。

   两条稀疏路径的 K/V 在矩阵乘法中使用 FP16，累加精度设为 FP32。近期槽位索引按 query head 和时间顺序生成，设备上的 GET_ROWS 从概率表中只取近期分数。CPU 回执接收近期分数和一个 attention 完成标量，确认本层归档后更新代表索引。attention 输出通过带回执依赖的原地设备节点直接交给后续层。CPU、CUDA、SYCL 均实现该节点，数值回归覆盖解码图复用、近期索引和原生完成通知。

   初始、历史和近期位置合并为唯一的 KV 位置列表。Prefill 按每个 query 的距离选择近期或历史 logits，同一位置在联合 softmax 和 V 累积中只计入一次；分组内已过期的临时近期行按掩码处理。

   CUDA 选块器启用设备概率累计时，近期分数和当前 FP16 Key 直接交给 GPU 的环形缓冲。每个 query head 独立累加分数，整块退出近期窗口时在 GPU 生成均值代表。CPU 接收新完成块的代表位置、分数和状态。Prefill 的按 key 汇总分数与 Decode 的单 token 分数接入同一条更新路径。回归用例覆盖 96-token 输入组、逐 token 解码和环形缓冲周转；CPU 参考按同样的 Prefill 分组、Decode 单步边界累计 FP16 分数。[attention.cpp](../src/attention.cpp)、[attention.hpp](../include/solid/attention.hpp)、[block_ranker.hpp](../include/solid/block_ranker.hpp)、[block_ranker.cu](../src/cuda/block_ranker.cu)、[session.cpp](../src/session.cpp)、[session.hpp](../include/solid/session.hpp)、[tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[tiled_prefill.hpp](../include/solid/tiled_prefill.hpp)、[cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)。

   稀疏基线用设备 Q 直接评分，仅把所选块编号交给主机安排 SSD 读取。历史行直接读入与设备 bank 对应的页锁定主机位置。相邻目标行合并上传，每批上传共享一次完成等待，源缓冲在完成后复用。纯 Offload 的整层上传也使用页锁定缓冲。 [llama_adapter.cpp](../src/llama_adapter.cpp)、[dense_offload.cpp](../src/dense_offload.cpp)、[gather_test.cpp](../tests/gather_test.cpp)。

   近期位置表按 KV head 扫描一次，组内 Query head 复用这份位置映射，各自加上概率表的 head 偏移。CUDA Top-K 按候选块的最小有效 token 数计算结果个数上界，限制输出清零及回传的范围，兼容完整块、部分块和不同 token 成本。设备评分回归使用完整块和不同成本的候选，与 CPU 全量排序结果核对。 [llama_adapter.cpp](../src/llama_adapter.cpp)、[block_ranker.cu](../src/cuda/block_ranker.cu)、[cuda_ranker_test.cpp](../tests/cuda_ranker_test.cpp)。

   GPU 代表更新按本步 Key 和分数元素数确定线程块数量。单 token Decode 只派发所需的线程块，多 token Prefill 保留网格步进处理。设备分数模式仅返回设备指针和元素数，省去同尺寸的 CPU 分数向量分配及清零。稀疏基线收到 attention 完成回执后先提交下一层 SSD 固定窗口读取，再更新本层代表分数。下一层读取和本层代表更新使用独立的数据。 [block_ranker.cu](../src/cuda/block_ranker.cu)、[tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[llama_adapter.cpp](../src/llama_adapter.cpp)。

   设备概率回执的就绪节点同时依赖 attention 输出和当前设备 KV，保证 CPU 使用设备指针前两者已完成。输出依赖节点持有设备 KV 和近期分数，覆盖代表更新的输入捕获过程。近期分数直接作为就绪节点输入，省去额外的连续副本。 [device_ops.hpp](../include/solid/device_ops.hpp)、[device_ops.cpp](../src/device_ops.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)。

   `baseline.cpp` 在 InfLLM 路径通过图内 attention 回执确认完成，关闭旧 FlashAttention 路径使用的额外 eval 观察回调。旧 FlashAttention 预取、纯卸载的完成通知及显式张量诊断保留所需回调。普通 InfLLM 推理避免触发 GGML 观察模式对每个 GPU 子图的强制同步。

   `9b3b4b0` 完成两条稀疏路径的 8K、8 步 Decode 短核验。配置中的 `selection_trace`、`prefetch_trace`、`schedule_trace`、`profile`、`dump_logits` 均关闭，保留 Nsight 关键阶段采集。稀疏基线 Prefill 为 1.376 秒、Decode 为 504.0 ms；SolidAttention 为 1.301 秒、349.9 ms。两条路径生成的 9 个 token 相同。SolidAttention 的流同步调用由上一提交的 6,122 次降为 5,442 次，224 层次调用全部使用 attention 原生完成事件与归档同步复用，独立归档完成点、显式 attention fence 和上传主机批次等待均为零。原始结果及 `.nsys-rep` 保存于 `results/server5090/nsys-infllm-8k-9b3b4b0.2qgc3e62/`。

   Decode 的 RoPE 位置表和近期概率索引合并为一个 I32 元数据张量，两段视图分别供旋转与概率 GET_ROWS 使用。`decode_metadata` 在选块及 KV 准备完成的主机边界统一生成两表，保留近期位置的范围、唯一性和完整性检查。模型图通过 `AttentionNode::metadata_node` 将该节点排在 K 复制、旋转、QK、softmax、PV 之前；概率提取复用已经生成的索引。`gather_test.cpp` 检查从 QK 到近期概率提取之间无 CPU 自定义节点插入，并对照流式 attention 输出；`prefetch_graph_test.cpp` 用相同顺序核对图复用与代表更新。服务器 54 项 CPU 回归通过。8K、8 步 Decode 的 Nsight 核验中，SolidAttention 的流同步调用由 7,018 次降为 6,122 次，稀疏基线由 7,010 次降为 6,562 次。[llama_adapter.hpp](../include/solid/llama_adapter.hpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[gather_test.cpp](../tests/gather_test.cpp)、[prefetch_graph_test.cpp](../tests/prefetch_graph_test.cpp)。

   同一提交 `138a486` 的短对照保留 Nsight 与两条路径的选块记录，分别核对逐块调度 JSON 开启和关闭的情况。关闭额外逐块 JSON 后，稀疏基线 8 步 Decode 为 516.2 ms，SolidAttention 为 356.5 ms；逐块 JSON 开启的 SolidAttention 为 640.2 ms。两组原始结果及 `.nsys-rep` 分别保存于 `results/server5090/nsys-infllm-8k-138a486.iokc6m03/` 和 `results/server5090/nsys-infllm-8k-138a486-nvtx.c8svfapi/`，配置文件记录各自日志开关。仓库实验生成器沿用默认关闭 `schedule_trace` 的配置。

   [device_ops.hpp](../include/solid/device_ops.hpp)、[device_ops.cuh](../include/solid/device_ops.cuh)、[sycl_device_ops.hpp](../include/solid/sycl_device_ops.hpp)、[device_ops.cpp](../src/device_ops.cpp)。

   [llama_adapter.cpp](../src/llama_adapter.cpp)、[llama_adapter.hpp](../include/solid/llama_adapter.hpp)、[device_cache.cpp](../src/device_cache.cpp)、[device_cache.hpp](../include/solid/device_cache.hpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[sycl_flash_decode.hpp](../include/solid/sycl_flash_decode.hpp)、[gather_test.cpp](../tests/gather_test.cpp)、[prefetch_graph_test.cpp](../tests/prefetch_graph_test.cpp)。

6. **原版 InfLLM 代表构造对照**

   `third_party/infllm` 保存 thunlp/InfLLM 提交 `12b70798f56e56ebb23c53c7018091a3f540a028` 的原始 Python 包及 MIT 许可证，`ORIGIN` 记录提交、来源和源码归档 SHA256。`infllm_reference_fixture.py` 直接调用这份源码的 `append_global`、`get_block_k`、GQA 展开及 `VectorTensor` 写入，生成 CPU 张量参考数据；初始化时只将存储放到 CPU，保留原函数。参考包含 32-token 检索块、28 个 query heads 对 4 个 KV heads、96/480/512-token Prefill、逐 token Decode、环形缓冲周转、未对齐 init 及 7/256-token 兼容块。每批输入均检查原版“组长不超过近期窗口”的前提。

   `infllm_reference_cases.hpp` 读取原版输出，逐批核对完整块发布边界、代表 Top-K 得分集合、选中的原始 Key 和 FP16 均值。分数唯一时核对原版代表集合；Top-K 边界同分时核对最优得分集合及实际所选 Key 的均值。随后直接调用原版 `calc_block_topk` 和 `VectorTensor.get_topk`，分别用 1、37 和最多 512 个 Q 生成检索参考；另加 Q 与块分数超过 FP16 有限范围的检索用例。检索候选使用代表边界无歧义的完整块，确保双方拥有相同的代表 bank。测试将 FP16 bank 的值无损提升为 FP32，与 llama.cpp 的 F32 Query 配对交给原版 matmul。参考格式 `IFLMREF3` 保存 F32 Query 均值和块分数，核对 GQA 返回列表及 Top-K 最优得分集合。

   `infllm_reference_test.cpp` 接入 CPU 索引及均值函数，`cuda_prefill_test.cpp` 接入真实 CUDA 代表 bank、检索内核和 Prefill Q 均值内核。`CMakeLists.txt` 添加独立的 CPU 对照程序。CPU 和 CUDA 均已通过 5 组、172 批、101 个完整块及 15 组检索的原版对照；服务器 54 项 CPU 回归通过，CUDA tiled Prefill 检查通过。同分代表的构造样例单独核对合法集合；检索比较使用拥有相同代表 bank 的样例。

   CUDA 对照的输入上传由 `upload_ready` 收取默认上传队列完成后，再交给独立计算队列，保证借用输入已就绪。`infllm_reference_cases.hpp` 的 F32 均值检查采用由输入绝对值和浮点精度计算的归约误差界，失败时报告样例、组长、元素及双方数值；选块继续严格核对原版最优得分集合。

   attention 另做过一次直接原版对照：在 CPU 分配频率表后，执行原版 `RotaryEmbeddingESM` 的旋转函数及 `TorchMultiStageDotProductionAttention.append/finalize`。按已约定的单 token 单次计入规则，为每个 Query 传入互不重复的近期与历史集合；缓存 FP16 K/V 提升为 F32 参与参考矩阵乘法。5 组样例覆盖全量因果前缀、组内滑出窗口、480-token Prefill、逐 token Decode、28 个 Query heads 对 4 个 KV heads，以及未成块尾部。CPU 输出与近期概率累计最大绝对差分别为 0.000068307、0.000035763；CUDA 为 0.000246644、0.000299931。参考使用 F32，CUDA GEMM 输入及概率含 FP16 舍入。结果和 Nsight 报告保存于 `results/server5090/infllm-original-attention-7fd2521/`。

   `7820c8d` 的服务器核验完成 Qwen2.5-7B Q4_0、8K 输入、8 步 Decode 的三方法短推理。三组均启用 Nsight、8 GiB 显存限制、16 GiB 主机内存限制及 Decode O_DIRECT。Prefill 分别为 Offload 1.575 秒、Offload＋Sparse 1.405 秒、SolidAttention 1.299 秒；两条稀疏路径均记录 784 个均值事件、0 次均值主机等待。结果、选块、逐阶段时间线及数值比较保存在 `results/server5090/nsys-infllm-8k-7820c8d/`，报告编号与方法由其中的 `capture/nsys/INDEX.tsv` 对应。

   [原版源码及许可](../third_party/infllm/)、[来源记录](../third_party/infllm/ORIGIN)、[infllm_reference_fixture.py](../scripts/infllm_reference_fixture.py)、[infllm_reference_cases.hpp](../tests/infllm_reference_cases.hpp)、[infllm_reference_test.cpp](../tests/infllm_reference_test.cpp)、[cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)、[CMakeLists.txt](../CMakeLists.txt)。

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

   生成 attention mask 前，将本层使用的显存槽位按 KV head 和逻辑块号排序。每次切换到另一个选中块时二分查找槽位，块内各 token 复用查询结果，并逐行检查版本范围。后续固定、释放槽位使用同一份列表。

   [device_cache.cpp](../src/device_cache.cpp)、[device_cache.hpp](../include/solid/device_cache.hpp)。

4. **主机/设备缓冲保护**

   已维护 layer/head/block/version 身份及 slot/generation 租约。槽位记录写入者和读者数量，读取中、可用、被读取及待回收状态分别处理。取消 READ 后，槽位保持待回收，收到原请求完成回执并释放全部读者后才允许覆盖；取消操作自身的回执只更新取消结果。GPU 复制持有主机源和设备目标，attention 持有设备工作集；共享内存路径同时持有对应主机槽位。尾块更新递增 generation，保留已完成的旧前缀，只填入新增行。

   请求结束时先停止设备完成通知的后续推进，等待协调任务、计算与复制完成，再释放 attention 和历史块读者、完成已提交 WRITE。会话结束排空预测 READ 和取消请求，写出剩余聚合缓冲。

   [block_identity.hpp](../include/solid/block_identity.hpp)、[prefetch.cpp](../src/prefetch.cpp)、[device_cache.cpp](../src/device_cache.cpp)、[async_reader.cpp](../src/async_reader.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[session.cpp](../src/session.cpp)。

## 四、SSD-aware Scheduler

对应论文 §6.1、§6.2、图 11、算法 1。

1. **层与块微任务依赖**

   已建立 Q 前沿、选择、KV 生成、补读、历史上传、当前行安装、attention、归档、回写和下一层预取的层图。逐块建立 READ→H2D→汇合依赖，计算 critical/LST，记录开始、完成与同步证明。

   正式选择确认时顺序遍历已选 token，一次生成每块的读取优先级，并用有序块表查询在途及驻留块是否仍被选中。块调度器在读取完成时把该块加入就绪队列，上传开始时移出该块。就绪队列沿用 critical、LST、任务编号顺序。

   [scheduler.cpp](../src/scheduler.cpp)、[block_stage.cpp](../src/block_stage.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)。

2. **Q 前沿与提前补读**

   已将设备 Q 就绪标量和 CPU 前沿接入模型图。overlap 与 overlap+sync_reuse 的前沿通过异步设备入口把选择交给持久协调线程后返回，GPU 可继续 KV 投影与交错转换。协调线程完成代表更新、选择和历史补读，KV 就绪处收割其结果。异步入口借用的 Q 由图依赖保留至 KV 汇合或排空结束，原同步设备入口在返回前消费 Q。选择异常在 KV 前沿传递。任务图把选择和 KV 生成作为 Q 就绪后的两个分支。naive 保留串行路径。

   SolidAttention 的 InfLLM 路径直接用设备上的 Q 做评分和 Top-K，代表向量继续由实际的近期 attention 概率生成。当前 token 的 K 从归档的 FP16 K/V 中提取，归档完成后接收本层近期概率。Offload＋Sparse 和 SolidAttention 沿用同一套候选块、评分规则和 25%／1K 总预算。

   CUDA 的同步复用路径在主机有空槽时先批量提交缺块读取，再等待前一批显存上传归还槽位。暂时被上传占用的源槽保留租约。需要刷新已有块尾或等待槽位的请求随后处理。

   [llama_adapter.cpp](../src/llama_adapter.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[session.hpp](../include/solid/session.hpp)、[session.cpp](../src/session.cpp)、[prefetch.hpp](../include/solid/prefetch.hpp)、[prefetch.cpp](../src/prefetch.cpp)、[block_stage_test.cpp](../tests/block_stage_test.cpp)。

3. **当前行安装与主机归档**

   当前设备行装入显存 KV 缓冲区时保留尾块前缀，只复制缺口。CUDA 的 `overlap+sync_reuse` 模式从连续的 FP16 投影一次下载所有 KV 头。下载在 attention 的计算流上排在等待历史 KV 上传之前，可与另一条流上的上传重叠。协调线程收到 attention 完成通知后读取已下载的当前行，将其追加到主机缓冲区，释放上一层显存槽位，推进下一层上传和当前层聚合回写。多层模型的下一层 SSD 读取可提前派发。普通 `overlap` 保留独立的归档完成点。上下文容量末尾照常完成当前 attention、归档和回写。

   实验结果记录共享 attention 完成点和单独确认归档的次数。调度轨迹核对脚本校验共享归档对应的投影依赖。CUDA 回归用例核对两种调度模式的完成次数及 attention 输出。

   设备 Q 提前选择入口支持同步、异步调用，并显式指定后续 KV 的归档方式。已下载到主机的 FP16 KV 采用“下载完成→追加→当前行安装”的依赖，进入 FP16 追加函数；设备 KV 采用“当前行安装→下载归档→追加”的依赖，默认模型图使用这一方式。两种入口均保留选择位置、输入类型和归档方式的一致性检查。

   [ggml_prefetch.hpp](../include/solid/ggml_prefetch.hpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[session.cpp](../src/session.cpp)、[device_cache.cpp](../src/device_cache.cpp)、[scheduler.hpp](../include/solid/scheduler.hpp)、[baseline.cpp](../src/baseline.cpp)、[schedule_ablation.py](../scripts/schedule_ablation.py)、[cuda_device_input_test.cu](../tests/cuda_device_input_test.cu)。

4. **Attention 完成通知**

   attention 回执确认本层完成，后继 Q 就绪和整次计算结束分别触发协调任务交接。`AttentionEvents` 分别记录完成证明与交接标记。收到 attention 回执后，模型线程通过 `TaskWorker` 的进度通知等待本层归档和槽位释放；协调线程继续接收下一层 SSD 完成回执并上传已就绪的块。下一层 Q 到达时交接给正式选块，请求结束或取消时排空任务。`native_completion_test.cpp` 延迟下一层读取，检查 attention 回执先返回，随后解锁读取时仍能继续上传。

   [attention_events.hpp](../include/solid/attention_events.hpp)、[attention_events.cpp](../src/attention_events.cpp)、[scheduler.hpp](../include/solid/scheduler.hpp)、[scheduler.cpp](../src/scheduler.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[attention_events_test.cpp](../tests/attention_events_test.cpp)、[native_completion_test.cpp](../tests/native_completion_test.cpp)。

   `baseline.cpp` 将 attention 回执复用单列为 `attention_receipt_reuses`，调度轨迹使用 `attention_dependency_reuse` 完成证明。`prefetch_graph_test.cpp` 写出同口径的图运行计数，`schedule_ablation.py` 和 `test_schedule_tools.py` 核对这类证明及次数。

   CUDA 已接入原生事件和图捕获/重放通知；SYCL 已接入结束事件和带 epoch 的独立队列通知。协调线程据此释放旧工作集读者并推进后继上传，后继 Q 与请求边界也具有完成证明。早于后继边界的上传次数，以上传开始派发时的边界状态为准。

   SolidAttention 的 CUDA InfLLM attention 在 QK 矩阵乘法前记录开始，在概率乘 V 后通知协调线程。下一层 SSD 预取可先发起，完成通知用于确认当前层上传依赖和当前 KV 回传，再推进当前层聚合写回及下一层显存上传。CPU 回执在归档完成后用近期概率更新代表索引。回归用例以相同输入核对重用解码图的输出、设备选块调用和 FP16 归档字节。 [llama_adapter.cpp](../src/llama_adapter.cpp)、[stream_completion.hpp](../include/solid/stream_completion.hpp)、[generate_adapter.py](../scripts/generate_adapter.py)、[prefetch_graph_test.cpp](../tests/prefetch_graph_test.cpp)、[representative_builder_cases.hpp](../tests/representative_builder_cases.hpp)。

   CUDA `overlap+sync_reuse` 的设备 KV 路径允许历史上传在构建掩码和提交当前行时继续执行。计算流统一等待最后一个上传事件，协调线程用 attention 完成回执确认整个上传批次并归还槽位。下一层 SSD 请求可以先排队，暂时被上传占用的页锁定槽位在写入方归还后才接收新数据。调度轨迹核对共同上传依赖，CUDA 回归用例检查计算流等待和主机批次等待的次数。 [device_cache.hpp](../include/solid/device_cache.hpp)、[device_cache.cpp](../src/device_cache.cpp)、[prefetch.hpp](../include/solid/prefetch.hpp)、[prefetch.cpp](../src/prefetch.cpp)、[ggml_prefetch.cpp](../src/ggml_prefetch.cpp)、[schedule_ablation.py](../scripts/schedule_ablation.py)、[cuda_device_input_test.cu](../tests/cuda_device_input_test.cu)。

   CUDA 在已有代表块无需发布的解码步，将代表分数更新与动态选块排入同一条流，等 Top-K 回传时一起检查更新和评分状态。需要发布新代表块或无需动态选块的步，在各自消费点确认更新完成。同步复用路径将当前 key 的追加放进 attention 计算流，共用 attention 回执，归档时验证相同 FP16 数据。结果记录选块等待复用及计算流追加次数。 [block_ranker.hpp](../include/solid/block_ranker.hpp)、[block_ranker.cu](../src/cuda/block_ranker.cu)、[attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[session.hpp](../include/solid/session.hpp)、[session.cpp](../src/session.cpp)、[baseline.cpp](../src/baseline.cpp)。

   [attention_events.cpp](../src/attention_events.cpp)、[attention_events.cu](../src/cuda/attention_events.cu)、[cuda_completion.cuh](../include/solid/cuda_completion.cuh)、[attention_events.cpp](../src/sycl/attention_events.cpp)。

   SolidAttention 的 CUDA `overlap+sync_reuse` 路径异步更新 InfLLM 代表。本层 attention 回执在模型计算流上捕获当前 FP16 K/V 和近期概率，随后代表队列等待捕获事件，执行原有 Key 追加、分数累加和代表生成内核。后续 FFN 只需排在输入捕获之后，CPU 在下一次 Q 前沿收取代表完成事件和元数据，每次模型计算结束及会话排空时收齐剩余结果。异步任务最多一份，快照大小由单 token 的 K/V 和近期概率窗口决定；位置、分数和错误状态通过固定页锁定缓冲区回传。Offload＋Sparse 保持原同步更新入口，两条路径使用相同的代表算法。

   `block_ranker.hpp`、`block_ranker.cu` 提供提交、收取、快照及失败排空；`attention.hpp`、`attention.cpp` 管理待收取的层和位置；`session.hpp`、`session.cpp` 传播错误并完成会话收尾；`llama_adapter.cpp`、`ggml_prefetch.hpp`、`ggml_prefetch.cpp` 将异步入口限定到 SolidAttention 的同步复用调度。`cuda_prefill_test.cpp` 在提交后立即覆盖原 K/V 和分数缓冲，再与 CPU 参考核对代表位置和发布边界。

5. **统一内存和复制队列**

   SYCL 已为支持共享 USM 的集成 GPU 分配共享 KV bank，SSD READ 与 dirty 数据转发直接填入该 bank 的历史槽位，attention 使用同一地址。主机释放独立块载荷，历史块安装只登记槽位和完成票据。共享内存的所有权随块视图保留，覆盖原 I/O 完成及排空过程。

   共享槽位在 attention 期间同时持有主机和设备读者。下一层预测先使用空闲槽位发起 SSD 读取；没有空位的块保留在有界预测计划中，原读者释放后继续派发。正式选择接管时清除尚未派发的预测，按纠错顺序加载所需块。每个 KV head 另保留一个当前块，写入当前设备行并复用尾块前缀。独立复制队列处理当前行及其历史缺口，具有事件查询和计算队列依赖。

   CUDA 及独立显存的异步复制路径将主机块直接绑定到上传所用的固定内存 bank，SSD READ 和 dirty 转发直接填充上传源。历史块与尾块前缀上传复用该地址，原复制票据持有源租约，省去普通内存到固定内存的中间复制。固定内存计入复制缓冲占用。

   CUDA 独立复制队列按当前就绪的一批历史块提交上传。新块优先占用与主机缓冲偏移相同的空闲显存槽。源、目标都连续的块通过一次 ggml 异步上传传输，离散区间也共用同批完成事件。命中块保留原槽，同批源和目标租约在完成事件到达后释放。正式选择先提交已就绪的一批，再等待尚未返回的补读。结果记录历史上传的块数、实际连续传输次数和完成事件批次数。

   [device_cache.hpp](../include/solid/device_cache.hpp)、[prefetch.hpp](../include/solid/prefetch.hpp)、[scheduler.hpp](../include/solid/scheduler.hpp)、[device_cache.cpp](../src/device_cache.cpp)、[prefetch.cpp](../src/prefetch.cpp)、[baseline.cpp](../src/baseline.cpp)。批次回归检查覆盖共享完成票据、失败后重试、源缓冲保护和命中槽位复用。[prefetch_test.cpp](../tests/prefetch_test.cpp)、[prefetch_fakes.hpp](../tests/prefetch_fakes.hpp)。

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

   密集路径按一段历史行读入有界 tile，分离布局分别读 K、V，交织布局中相邻且无填充的物理行一次读取。当前批直接使用设备侧 K/V。CUDA 大批次使用 cuBLAS 批量矩阵乘法及跨 tile FP32 在线 softmax，纯卸载配置采用 512 token 微批和 1024 token 历史 tile。

   Offload＋Sparse 和 SolidAttention 共用 InfLLM 稀疏预填充。每组长度上限取 `min(512, local)`，同时受模型微批剩余长度和 attention 工作区容量限制；跨越稀疏预算边界时先结束当前组。默认 1K/2K/至少 4K 输入分别采用 96/224/480-token 上限，25%／1K 总预算及 32-token 检索块沿用既定配置。超出预算后按组内 Q 的均值检索一次历史 KV 块。预算内的起始前缀使用全量因果 attention。每组读取并上传一次历史工作集，组内各 token 共用初始位置和检索结果，各自应用因果掩码和滑动近期窗口。CUDA softmax 屏蔽已滑出窗口的临时位置，每个 token 可见的 KV 数按总预算限制。 [attention.hpp](../include/solid/attention.hpp)、[attention.cpp](../src/attention.cpp)、[session.cpp](../src/session.cpp)、[tiled_prefill.hpp](../include/solid/tiled_prefill.hpp)、[tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[ranker_test.cpp](../tests/ranker_test.cpp)、[cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)。

   CUDA attention、Q 均值、CPU 参考及设备代表更新入口均检查组长约束。设备环形缓冲按 `local + 分组上限 + block_tokens` 分配，代表回执工作区按分组上限估算本次最多完成的块数。`llama_adapter.cpp` 用相同上限创建 CUDA attention 工作区。`session.hpp`、`session.cpp` 记录实际组长上限、总组数和进入稀疏阶段的组数；`baseline.cpp` 将其写入 `prefill_group_token_limit`、`prefill_groups`、`prefill_rank_groups`。CPU 回归用均匀 attention 独立核对分组边界变化后的输出及检索组数。

   cuBLAS 分别对可能参与历史 attention 的行和近期窗口的行计算 QK，CUDA 依据每个 token 的因果和窗口位置做联合 softmax 与 V 累积。GPU 上按 key 和 query head 汇总本组近期分数。设备代表模式在 GPU 环形缓冲中追加 Key、累加分数并生成代表，CPU 记录追加位置和已发布块的元数据。CPU 代表模式接收近期分数并累加，发布离开近期窗口的代表块。

   设备代表模式的 Prefill 直接使用设备 K/V 和分数，主机归档 KV 写入按层写缓冲后，省去当前检索组的额外 KV 副本及逐 token 提取 Key。设备 Key 的有限值检查由 GPU 代表更新完成，归档前检查主机 KV 的有限值。CPU Query、均值和 Key 临时数组按实际计算路径分配。`attention.hpp`、`attention.cpp`、`session.cpp` 接入此路径；`cuda_prefill_test.cpp` 对照 CPU 参考，检查多组 Prefill 转单 token Decode 后的代表位置和发布边界。

   CUDA 设备选块路径直接从模型图接收设备 Q，均值向量在 GPU 生成后交给设备评分器，attention 输出直接写入下一层可用的设备缓冲。新 K/V 就地参与 attention，并回传一次 FP16 副本供 SSD 写入。工作集上传仅拷贝历史行。统计分别记录设备 Q/KV 数量、SSD 归档字节和近期分数回传字节。`scatter_sparse_output` 检查 V 累积结果。`baseline.cpp` 的 `EvaluationBridge` 支持采集首个多 token 批次的前两层投影和 attention 输出。

   [tiled_prefill.cu](../src/cuda/tiled_prefill.cu)、[tiled_prefill.hpp](../include/solid/tiled_prefill.hpp)、[tiled_prefill.cpp](../src/sycl/tiled_prefill.cpp)、[session.cpp](../src/session.cpp)、[session.hpp](../include/solid/session.hpp)、[attention.cpp](../src/attention.cpp)、[attention.hpp](../include/solid/attention.hpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[config.cpp](../src/config.cpp)、[baseline.cpp](../src/baseline.cpp)、[experiments.py](../scripts/experiments.py)、[experiment_common.py](../scripts/experiment_common.py)、[test_experiments.py](../tests/test_experiments.py)、[ranker_test.cpp](../tests/ranker_test.cpp)、[cuda_prefill_test.cpp](../tests/cuda_prefill_test.cpp)。

   CUDA 稀疏预填充的分数工作区支持最多 64 Mi 个分数元素，容纳 512-token 检索组的历史块与滑动窗口并集。密集预填充沿用原工作区上限。

   `baseline.cpp` 的可选 attention 张量诊断增加 InfLLM Decode 捕获，`SOLID_DIAGNOSTIC_PHASE` 区分 Prefill 和 Decode。Decode 在 PV 节点完成时记录每层 Q、旋转后 K、V、mask、联合概率及输出，连同类型、形状和 stride 保存；捕获使用同步调度保护被读取的 KV bank。8K 单步诊断用拆开的 K/V 投影对照两条稀疏路径：前 26 层 Q 逐值相同，实际有效 KV 集合相同，attention 输出最大差异约为 5.96×10⁻⁷。后两层的 Q 和当前 KV 随后产生差异。原始 8 步短核验中，两条稀疏路径生成 token 相同，896 条选块记录有 348 条不同，logits 最大绝对差为 2.216；这些数据与逐层诊断分别保存。

5. **Offload 全量 Decode**

   `dense_offload.cpp` 用一份对齐的主机缓冲和一份显存缓冲逐层交换分离的 FP16 K/V。解码输入就绪时启动第 0 层读取，当前层 FlashAttention 完成时启动下一层读取。`kv_store.cpp` 一批提交该层各头的连续 K/V 读请求，直接读对齐区间并从待写缓冲补齐尚未落盘的行。

   `llama_adapter.cpp` 将历史上传与当前 K/V 回传拆成两个图节点。Q 留在计算设备，K/V 投影完成后上传整层历史、回传本步 K/V，再用 GGML `SET_ROWS` 填入当前行，由 llama.cpp 的 FlashAttention 计算。当前层写缓冲达到容量时，后台线程在 attention 执行期间提交写回。`session.cpp` 记录历史等待与归档，`baseline.cpp` 记录逐阶段读取、等待和上传量。`gather_test.cpp` 检查两个连续解码步的输出，`uring_test.cpp` 核对批读后的 K/V 和待写尾部。

6. **运行与结果记录**

   `llama_adapter.cpp` 的 `SparseBaselineLookahead` 在解码输入就绪时准备首层初始和近期窗口，在当前层 attention 回执后准备下一层同类窗口。读取线程在 SSD 读取完成后立即提交固定窗口上传，等待这批上传完成后归还主机源缓冲；GGML 固定窗口消费节点收取该后台任务。CUDA 上传使用同一设备上的独立 GGML backend，队列命名为 `sparse.kv.H2D`，与模型计算队列分开。后台线程通过既有队列接口选择正确的 CUDA 设备；上传出错及工作区销毁时排空复制队列。

   所有层共用一份显存工作缓冲，每次切换层或解码步时重新从 SSD 读取所需的历史 KV。正式选块后再批量补读、上传剩余历史 KV。本步 K/V 直接取自回传的投影结果，已满的写缓冲在选块期间后台落盘。GGML 按逻辑位置聚集分离的 K、V 并计算原有 InfLLM attention。解码回归检查连续两步均读取历史 KV，并与参考 attention 对照。 [llama_adapter.hpp](../include/solid/llama_adapter.hpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[gather_test.cpp](../tests/gather_test.cpp)。

   `kv_store.hpp`、`kv_store.cpp` 的分离 KV 区间批读按队列深度提交 `io_uring`，从待写缓冲补齐尚未落盘的行。`session.hpp`、`session.cpp` 提供读取及等待计时，`llama_adapter.hpp` 将预取及缓存绑定到解码节点，`baseline.cpp` 记录固定窗口等待、上传、命中和淘汰。`gather_test.cpp` 对照连续解码两步的输出和读请求计数。

   Offload 和 Offload＋Sparse 同步读入 ggml 的目标缓冲区，纯卸载取当前层完整历史，稀疏卸载取相邻选中行组成的区间。两者分别填充每个 query head 的有效 mask。SolidAttention 的交织、推测性预取及 SSD 调度由对应模块执行。[session.hpp](../include/solid/session.hpp)、[session.cpp](../src/session.cpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[gather_test.cpp](../tests/gather_test.cpp)。

   SSD 三组支持在解码开始前启用 Linux `O_DIRECT` 读取，预填充结束后先提交现有 KV 数据。对齐的请求直接写入目标缓冲区，未对齐的请求扩展到文件系统要求的边界，并在同步读取或异步回执后复制有效区间。异步取消保留临时缓冲直到原始请求完成。结果记录并核对 I/O 模式。 [kv_store.hpp](../include/solid/kv_store.hpp)、[kv_store.cpp](../src/kv_store.cpp)、[prefetch.hpp](../include/solid/prefetch.hpp)、[prefetch.cpp](../src/prefetch.cpp)、[session.hpp](../include/solid/session.hpp)、[session.cpp](../src/session.cpp)、[llama_adapter.hpp](../include/solid/llama_adapter.hpp)、[llama_adapter.cpp](../src/llama_adapter.cpp)、[config.hpp](../include/solid/config.hpp)、[config.cpp](../src/config.cpp)、[baseline.cpp](../src/baseline.cpp)、[experiments.py](../scripts/experiments.py)、[experiment_common.py](../scripts/experiment_common.py)、[uring_test.cpp](../tests/uring_test.cpp)。

   CUDA 对照配置启用解码 `O_DIRECT`，cgroup 主机内存上限 16 GiB、swap 上限为零。批量运行器确认当前环境有单个 MPS 服务，为 CUDA 客户端设置 8 GiB 分配上限并记录实际传给 runner 的环境值。内存结果分列 KV 工作区、按层写缓冲以及 cgroup 文件缓存采样峰值。[cuda-paper.json](../configs/experiments/cuda-paper.json)、[experiment_runtime.py](../scripts/experiment_runtime.py)、[experiments.py](../scripts/experiments.py)、[run_baseline.py](../scripts/run_baseline.py)。

   已写模型和源码哈希、配置检查、固定 token/选择回放、运行监控、数值输出和分项内存/I/O 统计。可通过 `SOLID_DIAGNOSTIC_ATTENTION` 在同步路径和 naive 预取路径导出首个 decode 的逐层 Q、K、V、mask 和 attention 输出，同时记录类型、形状、步幅和所在缓冲；导出发生在完成观察器释放缓存之前，入口限制调度模式以保留导出期间的缓存读者。已写实验矩阵、准确率接口、结果核查及绘图工具。

   [baseline.cpp](../src/baseline.cpp)、[config.cpp](../src/config.cpp)、[run_baseline.py](../scripts/run_baseline.py)、[experiments.py](../scripts/experiments.py)、[accuracy.py](../scripts/accuracy.py)、[plot_experiments.py](../scripts/plot_experiments.py)。

7. **Nsight 论文阶段标记**

   `trace.hpp`、`trace.cpp` 接入 NVIDIA NVTX v3.3.0，运行时由 `SOLID_NVTX_TRACE=1` 开启。`third_party/nvtx` 保存官方头文件、许可和来源校验值，`CMakeLists.txt`、`Adapter.cmake` 接入构建。统一放在 `Stages` 分组，外层仅记录 Prefill 和逐步 Decode；层内记录 Q 投影、KV 投影、评分选块、完整 Attention、SSD→CPU 预取与补读、CPU→GPU 预取与装入、GPU→CPU 当前 KV 回传、CPU→SSD 写回。标签保留层号和适用的 token 位置。

   `generate_adapter.py`、`trace.cpp` 在模型计算入口只标记 Q 投影、合并的 K/V 投影或融合 KV 投影，将 InfLLM 的 QK、softmax、PV 合为一个 Attention 区间。`llama_adapter.cpp` 为该区间边界和融合 KV 张量设置层号名称。`block_ranker.cu` 每次检索记录一个评分选块阶段，`session.cpp` 每层 Prefill attention 记录一个阶段。`baseline.cpp` 记录整体 Prefill 和逐步 Decode。逐算子、逐函数、逐块的 NVTX 条均已移除；`scheduler.cpp` 仅保留当前层 KV 回传的跨线程阶段。

   `prefetch.hpp`、`prefetch.cpp` 将同层、同用途的异步 SSD 请求绑定到一个共享批次标记。预取和正式补读分别记录，从第一笔请求接受提交开始，到最后一笔回执及规划持有者释放时结束；取消中的原请求继续持有标记至回执。异步写回从实际提交开始，到该层写批次完成时结束。`device_cache.hpp`、`device_cache.cpp` 将同层、同用途的并发上传票据合为一个阶段，按预取和正式装入区分，最后一个票据收割后结束。标记持有者仅保存追踪元数据。

   `dense_offload.cpp`、`llama_adapter.cpp` 分别记录两条基线的整层或固定窗口预取，以及正式选中 KV 的补读、上传。`kv_store.cpp` 在写缓冲有数据且实际执行同步写回时记录 Store。三组保留原有 32 KiB 每层聚合写规则，缓冲攒满和请求结束排空时产生实际写盘阶段。

   CPU 线程职责名为 `sa.infer`、`sa.coordinator`、`sa.ssd-io`、`off.ssd-read`、`sp.window-io`；后者执行稀疏基线固定窗口的 SSD 读取与上传。`sa.infer` 只用于实际推理进程，配置检查和构建信息查询保留原进程名。Python 调度与监控分别为 `sa.matrix`、`sa.monitor`，GGML 持久 CPU worker 按编号命名。CUDA stream 保留模型计算、KV 上传、评分与代表更新、Prefill attention 的职责名。对应实现位于 `scheduler.hpp`、`scheduler.cpp`、`llama_adapter.hpp`、`llama_adapter.cpp`、`ggml_prefetch.cpp`、`io_worker.cpp`、`process_lifetime.py`、`experiments.py`、`run_baseline.py`、`tiled_prefill.cu`。

   `scripts/solid-nsys.sh` 开启 CUDA、NVTX、系统库调用与 CUDA Graph 节点采集。`scripts/profile_matrix.sh` 为每个任务生成独立 `.nsys-rep` 和 `INDEX.tsv`，记录方法、输入长度、模型、任务 ID、配置文件；续跑按执行清单跳过已有任务，并核对采集对应任务已完成。`run_baseline.py` 在运行清单记录标记开关。CPU 阶段条记录主机观察到的区间，GPU kernel 和 memory 行记录实际计算与传输。

   `tests/trace_test.cpp` 检查线程职责名、完整 Attention 的合并、批次标记共享、跨线程结束及上下文释放。`tests/test_adapter_generator.py` 检查生成结果可重复、上游源码保持原样，以及 FlashAttention 和 InfLLM 两个完成通知入口。
