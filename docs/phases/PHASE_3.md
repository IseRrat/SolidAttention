# 阶段 3：历史预取与纠错

代码完成于 2026-11-17。依据为 SolidAttention §5.1–5.2、图 10：按同层上一轮的 selected 历史预取，确定性加载 init/local；正式选择后补读遗漏块，复用释放的槽位，保持有效块原位。

## 1. 交付与代码入口

| 模块 | 代码入口 | 完成内容 |
| --- | --- | --- |
| 异步读取 | [async_reader.cpp](../../src/async_reader.cpp) | liburing、有限队列、短读重试、EOF、取消 CQE 与原 READ CQE、排空 |
| 读取计划 | [kv_store.cpp](../../src/kv_store.cpp)：`plan_read` | 已提交文件片段、写缓冲转发、已有前缀保留、尾块版本刷新 |
| 历史预取与纠错 | [prefetch.cpp](../../src/prefetch.cpp)：`Prefetcher` | 每层每 head 的历史、确定性预测、在途命中、补读、取消与主机槽位 |
| 槽位状态 | [prefetch.hpp](../../include/solid/prefetch.hpp)：`SlotTable` | layer/head/block/version 身份、generation、producer 与 reader 租约 |
| 设备驻留 | [device_cache.cpp](../../src/device_cache.cpp)：`DeviceCache` | 复制票据、设备槽位复用、有效块保持原位、物理槽位 mask |
| GGML 复制与事件 | [ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) | 固定设备 KV buffer、异步设置与完成事件、多层预取衔接 |
| 图接入 | [llama_adapter.cpp](../../src/llama_adapter.cpp) | decode 前启动预取、mask 节点、Flash Attention 完成确认 |
| 运行与清理 | [baseline.cpp](../../src/baseline.cpp) | 观测回调衔接、错误传播、context 销毁前排空设备工作 |
| 对照与轨迹核查 | [prepare_prefetch.py](../../scripts/prepare_prefetch.py)、[compare_prefetch.py](../../scripts/compare_prefetch.py) | 固定输入、动态/固定选择、logits、槽位生命周期与来源校验 |

写入继续使用每层聚合缓冲和同步回写。异步读取通过独立 liburing 队列进行，文件采用 buffered I/O。

## 2. 构建与配置

Linux 构建启用 `SOLID_URING`，链接系统 liburing。开发检查使用 Ubuntu 24.04、GCC 13.3、CUDA Toolkit 12.6 与 liburing 2.5。依赖准备及检查：

```bash
sudo apt-get install liburing-dev
bash scripts/check.sh --cuda
bash scripts/check.sh --sanitize
```

`solid-baseline --build-info` 的 `features` 记录 CUDA 和 liburing 开关。运行脚本在读取模型前检查预取所需的构建特性。

预取模式入口为 [offload-prefetch.json](../../configs/offload-prefetch.json)：

```json
"attention": {
  "storage": "ssd",
  "sparse": true,
  "layout": "interleaved",
  "fuse_kv_projection": true,
  "prefetch": true,
  "prefetch_queue_depth": 64,
  "prefetch_trace": true,
  "block_tokens": 32,
  "representative_count": 1,
  "write_buffer_bytes_per_layer": 32768
}
```

| 字段 | 默认值 | 行为 |
| --- | --- | --- |
| `prefetch` | `false` | 启用历史预测、异步补读与固定设备槽位 |
| `prefetch_queue_depth` | 64 | 逻辑在途读取上限，范围 2–16384；提交队列为取消操作保留空间 |
| `prefetch_slots_per_head` | 自动 | 每个 KV head 的固定块槽位数；实际集合在分配前核查容量 |
| `prefetch_trace` | `false` | 输出主机预取与设备驻留 JSONL；核查配置显式打开 |

入口要求 SSD、稀疏、交错布局、GGML decode，以及所有 attention 层使用同一后端。融合投影沿用阶段 2 的独立开关。配置检查覆盖队列深度、槽位上限和选项组合。

设 token 预算为 `C`，块大小为 `B`，容量为 `N`，KV head 数为 `Hkv`：

```text
S = min(ceil(N/B), ceil(C/B) + 3)     # 自动块槽位数 / KV head
P = align_up(S*B, 32)                # attention 的物理 token 槽位数 / KV head
host_payload_bytes = Hkv * S * B * 2 * D * 2
device_kv_bytes = Hkv * P * 2 * D * 2
mask_bytes = query_heads * P * 2
```

`+3` 为 init/local 边界与块取整留出空间。显式槽位数适用于额外碎片化的固定轨迹。实际集合超出配置容量时，错误在补读和 attention 前传递。内存记录分别列出主机 payload、设备 buffer、代表向量、写缓冲及跟踪工作区。

## 3. 预测与正式纠错

首轮 decode 在当前 Q/K/V 就绪前启动第 0 层读取。预测集合先加入 init/local 已存在的块，再加入该层上一轮实际 selected 块；逻辑块去重后按 head 容量裁剪。首次进入一层时，历史为空；可见历史小于预算时预取全部已存在块。

当前层正式处理依次执行：

1. 等待先前 H2D 源缓冲的复制确认，追加当前 FP16 KV，并执行原有选择器或固定轨迹回放。
2. 以完整身份核对主机槽位。版本相同的 READY 块直接使用，LOADING 块等待原读取，遗漏块发起补读。
3. 将错误预测置为 retiring，移除待提交操作，对已提交读取请求取消；原 READ 完成后释放槽位。
4. 优先复用已使用并释放的槽位。有效主机块和设备块保持原位，更新实际 selected 历史。
5. 完成必要的 H2D，生成按物理槽位排列的 mask，并为本次 attention 固定设备读者租约。
6. 主机复制完成后启动下一层读取；最后一层提前预测下一轮第 0 层。请求结束时排空剩余预测。

每层、每 head 独立保存 selected 历史，init/local 按本轮位置重新确定。预测阶段只访问已生成的 KV，当前 token 在正式追加时进入工作集。

## 4. DRAM/VRAM 生命周期

主机与设备各使用一张 `SlotTable`，槽位身份包含 `(layer,head,block,version)`，租约再包含 `(slot,generation)`：

```text
FREE -> LOADING -> READY -> PINNED -> READY
          |                    |
          +----> RETIRING <-----+
                    |
          producer 完成且 readers=0
                    |
                   FREE
```

主机槽位的 producer 为原 READ；其 reader 可以是 H2D。设备槽位的 producer 为上传操作；其 reader 为 attention。每次分配或尾块刷新递增 generation，完成通知逐项核对租约；旧通知计入 stale 统计并跳过状态更新。

liburing 用单独的标签区分取消 CQE 与 READ CQE。取消状态分别记录成功、找不到、正在执行与其他错误，原读取的完成通知负责结束内核写入期。读取的全部片段完成后才发布 READY。取消与完成的顺序依据 [liburing 取消接口文档](https://manpages.ubuntu.com/manpages/noble/man3/io_uring_prep_cancel.3.html)。

H2D 提交后，主机源缓冲保持 pin，直到设备事件确认复制完成。CPU 后端通过同步复制确认。Flash Attention 的 GGML 观测回调在后端同步后释放设备读者租约，并尝试上传下一层已读取的预测块。

`schedule=naive` 逐层确认 attention 完成。[阶段 4](PHASE_4.md) 在这些租约和完成接口上接入微任务 DAG、计算/I/O 重叠与同步复用。

## 5. 尾块版本与错误清理

预取时版本为块内已生成 token 数。正式追加当前 token 后，旧尾块先等待原读取完成，再以新 generation 保留不可变前缀。`plan_read(prefix)` 只读取新增的已提交行，并从写缓冲转发新增脏行。当前 K/V 的新版本与历史位置共同进入正式 mask。

设备将新版本上传到已释放槽位；其他有效块的物理位置与 payload 保持不变。mask 仅开放实际选择的 token 行，完整块中的其余行和 padding 使用负无穷。Q/K 沿用原生 RoPE，GQA 仍按原组映射。

短读取按剩余偏移和长度继续提交，EOF 和 I/O 错误使当前会话失败。运行器在采样前检查错误。context 销毁前的清理 guard 同步设备、排空复制并释放租约；liburing 原读取和取消通知在主机缓冲区与文件句柄释放前排空。

## 6. 实验环境的对照步骤

在当前提交先保存同步参考。复制 `offload-sparse-interleaved.json` 为 `configs/sync.local.json`，设置模型/输入路径、`dump_logits=true` 和新结果目录 `results/sync/`，然后运行：

```bash
python3 scripts/run_baseline.py configs/sync.local.json
python3 scripts/prepare_prefetch.py results/sync/result.json configs/prefetch-check.local.json --output results/prefetch-check/result.json
python3 scripts/run_baseline.py configs/prefetch-check.local.json
python3 scripts/compare_prefetch.py results/sync/result.json results/prefetch-check/result.json --atol 0.001 --rtol 0.0001
```

准备工具继承模型、实际输入 token、上下文、后端、投影与生成设置，锁定参考结果中的块大小和预算，使用新会话文件并打开核查轨迹。动态模式同时核对正式选择集合与逐步 logits。

固定集合核查在准备命令添加 `--fixed-selection`，并使用新的配置和结果路径。此模式回放参考 `selection.jsonl`，核查相同选择下的预取、纠错、复制和 attention。

比较器核对同一代码与二进制、配置/结果哈希、输入、矩阵形状、选择集合和轨迹校验和。设备轨迹核查器逐槽验证 generation、上传完成、attention pin/release 与覆盖顺序，并检查请求结束时的租约平衡。

端到端性能配置使用 `prefetch_trace=false`、`profile=false`、`dump_logits=false`，每轮写入独立目录。设备带宽、任务准确率、延迟波动和资源峰值在实验环境记录。

## 7. 轨迹与指标

| 文件或字段 | 内容 |
| --- | --- |
| `selection.jsonl` | 正式选择的逻辑 token 范围，延续同步参考格式 |
| `prefetch.jsonl` | predict、hit、inflight_hit、missing、unused、refresh、ready、retired 及版本化槽位身份 |
| `residency.jsonl` | upload、upload_complete、attention_pin、attention_complete、overwrite_available 和 epoch |
| `predicted_blocks` | 预测决策的块数，包含请求末尾提前预测的下一轮 |
| `ready_hits/inflight_hits/missing_blocks/refreshed_blocks` | 正式核对时的完整版本命中、在途命中、补读和尾块刷新 |
| `unused_blocks` | 正式纠错中淘汰的错误预测块 |
| `async_read_requests/async_read_bytes/completed_reads` | liburing 实际读提交次数、读取字节与逻辑完成数 |
| `cancel_*` | 取消请求、完成与结果分类 |
| `wait_ms/resolve_inclusive_ms` | 阻塞等待与正式纠错总耗时 |
| `read_completion_latency_sum_ms` | 单请求完成延迟的累加值，保留并发请求的重叠区间 |
| `device_uploads/device_upload_bytes` | 显式设备上传调用与完整块字节 |
| `device_reused_blocks/device_overwritten_blocks` | 正式设备命中与释放后复用的块数 |
| `host_buffer_bytes/device_buffer_bytes` | 主机块 payload 容量与设备后端实际 buffer 容量 |

统计对象为系统与后端 API 调用。阶段 2 的 `profile.transfers` 记录 GGML 调度器复制；本阶段显式槽位上传记录在 `device_upload*`，两类指标按数据流分别分析。

## 8. 完成检查

`bash scripts/check.sh --cuda` 通过 CUDA 编译与链接、21 项 CTest、19 项 Python 检查及源码指纹核对。`bash scripts/check.sh --sanitize` 通过核心库的 5 项 CTest 和地址、泄漏、未定义行为检查。

| 检查 | 证据 |
| --- | --- |
| 真实 liburing | 临时文件读取、队列容量、短读与 EOF、取消竞态、原请求与取消通知排空 |
| 主机槽位 | 冷启动、命中、全错 selected 历史、在途命中、补读、尾块版本与容量边界 |
| 乱序与旧通知 | 逆序完成、取消确认先于原写入、旧 generation 和重复通知 |
| 复制与设备槽位 | H2D 期间主机 pin、attention 期间设备 pin、释放后覆盖、有效槽位与 payload 保持原位 |
| 多层 attention | 两层、GQA、动态 Q；真实 liburing/内存读取、物理槽位 mask 与 CPU Flash Attention 对同步稀疏参考 |
| 完成回调 | GGML scheduler 对每层 attention 确认完成，再释放设备租约 |
| 失败与清理 | I/O 失败、独立/交错布局多片段读失败、取消、排空和内存检查 |
| 配置与工具 | 组合和范围拒绝、构建特性、实际输入继承、选择差异、覆盖时序、epoch 和来源检查 |

本机检查使用合成张量与临时文件。模型输出、CUDA 实际执行、任务准确率与设备性能按本节实验入口核查。全部阶段入口见 [阶段索引](README.md)，检查摘要见 [CHECKS](../CHECKS.md)，后续任务见 [阶段 4](../ROADMAP.md#阶段-4微任务调度)。

2026-11-18 补齐了独立 I/O 所有者、有界请求/完成队列、物理核预留和取消排空，详见 [I/O 线程完成记录](../implementation/IO_WORKERS.md)。

2026-11-18 接入跨 head selected 优先提交与正式请求提升，完成块逐个进入设备复制。顺序、租约与受限队列证据见 [块读取流水](../implementation/BLOCK_READ_PIPELINE.md)。
