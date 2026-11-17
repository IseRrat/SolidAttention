# 阶段 2：交错布局与融合投影

代码完成于 2026-11-17。论文依据为 SolidAttention §4、图 7–8：初始化时合并 K/V 投影权重，生成 token 级交错输出；attention 用 stride 视图访问交错 K/V。

## 1. 交付范围

本阶段打通交错文件、投影、主机工作集和 GGML attention，并提供固定输入、固定选择集合的四组消融配置生成器与结果比较器。

| 产物 | 入口 | 核查内容 |
| --- | --- | --- |
| 布局选择 | `AttentionConfig`、`src/config.cpp` | `layout`、`fuse_kv_projection`、`profile`、`selection_replay` |
| 原布局块读取 | `KvBlock`、`FileKvStore::read` | `data` 保持文件布局，`key(row)` / `value(row)` 提供逻辑行 |
| 连续 KV 追加 | `KvStore::append_interleaved` | `[Hkv,2,D]` 直接进入每层写缓冲 |
| 初始化权重合并 | `src/projection.cpp`：`KvProjection` | 完整量化行拷贝、混合类型转换、bias、缓冲区与耗时记录 |
| 计算图接入 | `adapter_qkv`、`KvProjection::update_key` | 独立 Q 投影、单次 KV 投影、原生 K RoPE 回写 |
| Attention 视图 | `AttentionNode::build` | FP16 K/V stride、GQA mask、交错工作集上传节点 |
| 固定选择回放 | `AttentionSession::select` / `finish` | 身份、历史范围、预算、init/local、文件完整消费校验 |
| 分项观测 | `src/profile.cpp`、生成的 `ggml-backend.cpp` | 分阶段的节点时间、跨后端复制次数/字节/时间 |
| 消融工具 | `scripts/layout_ablation.py` | 四组配置、来源校验、选择集合与 logits 对照、分项结果 |

## 2. 配置与模式

```json
"attention": {
  "storage": "ssd",
  "sparse": true,
  "layout": "interleaved",
  "fuse_kv_projection": true,
  "profile": false,
  "block_tokens": 32,
  "representative_count": 1,
  "write_buffer_bytes_per_layer": 32768
}
```

| 字段 | 默认值 | 行为 |
| --- | --- | --- |
| `layout` | `separate` | 选择文件块和 decode 工作集的 K/V 排列 |
| `fuse_kv_projection` | `false` | 初始化融合权重，以单次矩阵乘法生成 KV |
| `profile` | `false` | 开启逐节点、逐复制的同步计时 |
| `selection_replay` | 空 | 从 JSONL 读取每层、每轮、每 KV head 的固定选择 |

布局和融合投影可独立组合，形成 `separate-separate`、`separate-fused`、`interleaved-separate`、`interleaved-fused` 四组。第一个名称表示缓存布局，第二个名称表示 KV 投影。

直接使用的交错配置为 `configs/offload-interleaved.json`、`configs/offload-sparse-interleaved.json` 和 `configs/indram-sparse-interleaved.json`。它们均开启融合投影，沿用锁定模型与公共生成设置。

## 3. 文件与 attention 工作集

文件块按 `(layer, kv_head, block)` 定位，文件头保留 `SOLIDKV1`，布局字段取 0 或 1。设块大小为 `B`、head 维度为 `D`、FP16 元素字节数为 `s=2`：

```text
separate:    [K0 ... K(B-1)] [alignment padding] [V0 ... V(B-1)]
interleaved: [K0 V0 K1 V1 ... K(B-1) V(B-1)]
separate_slot_bytes    = 2 * align_up(B * D * s, alignment)
interleaved_slot_bytes = align_up(B * 2 * D * s, alignment)
```

对于完整已提交块，独立布局发出两次读取，交错布局发出一次读取。`B=32,D=128` 时，payload 分别为两次 8 KiB 和一次 16 KiB。尾块按有效 token 数读取，待写入行从写缓冲转发。短传输按剩余字节重试，计数随实际 API 调用增加。

`KvBlock::data` 保存原布局。交错块直接读入该缓冲区；prefill 在线 softmax 通过行指针取 K/V，decode 将选中行成对复制到交错工作集。读取过程保持 K/V 交错顺序。

decode 工作集包含预算内 KV 和 FP16 mask，槽位数 `S=align_up(budget,32)`：

| 布局 | KV 排列 | token stride | head stride | V 起始偏移 |
| --- | --- | --- | --- | --- |
| 独立 | `[2,Hkv,S,D]` | `D*s` | `S*D*s` | `Hkv*S*D*s` |
| 交错 | `[Hkv,S,2,D]` | `2*D*s` | `S*2*D*s` | `D*s` |

两种布局共用选择集合、GQA 映射与 mask。独立布局提供连续 K/V 视图；交错布局通过完整工作集的 `solid_kv_upload` 节点接入目标层后端，再建立 K/V 视图。该上传节点包含后端暂存及 DUP 计算，分项计时记录其开销。跨后端实际请求由复制观测器记录。

## 4. 融合投影与位置

模型加载完成、context 创建前，`AttentionAdapter` 为每层建立 `KvProjection`。权重输出行按 `[kv_head,K/V,D]` 排列，生成张量形状为 `[2D,Hkv,tokens]`。Q 继续使用上游投影；融合 KV 的 bias 与权重行采用相同次序。

相同 GGML 类型的 K/V 权重按完整行原样复制，量化块边界沿输入维度保留。K/V 类型不同的层逐行反量化为 FP32，再合并为 FP32 权重。初始化暂存区按单行分配。结果文件记录每层实际融合类型、额外权重 buffer 字节和初始化时间。

原 GGUF 权重继续由模型持有，融合 buffer 由 adapter 持有并在会话结束释放。因此融合权重容量单列为 `projection_extra_weight_bytes`，并参与实验环境的资源核查。

核查 INT4 投影性能时使用 K/V 类型一致的模型，并在配置中重新锁定模型分片与哈希。现有 Q4_K_M 路线逐层记录实际融合类型，供资源与算子成本分析。

融合入口接受独立的 Q/K/V 权重、统一 K/V head 形状、原生行编码和可选 bias；层形状、权重 scale、LoRA 与 Hadamard 图变换在入口检查。Q/K 的位置计算沿用模型原生 RoPE。K 的位置结果经 `solid_rope_key_writeback` 写回交错张量中的 K 行，V 保留投影结果；后续主机节点通过一个连续 KV 输入读取该张量。

## 5. 固定选择集合与消融

在实验环境先用当前版本的 `offload-sparse` 配置保存参考结果，设置 `dump_logits=true`。参考目录含 `config.json`、`prompt.txt`、`result.json`、`logits.f32` 和 `selection.jsonl`。

```bash
python3 scripts/layout_ablation.py prepare results/reference/result.json results/layout-configs --results results/layout-check
for mode in separate-separate separate-fused interleaved-separate interleaved-fused; do
  python3 scripts/run_baseline.py "results/layout-configs/${mode}.json"
done
python3 scripts/layout_ablation.py compare results/layout-configs/ablation.json --atol 0.001 --rtol 0.0001
```

准备工具继承参考的模型、输入、上下文、后端与生成设置，锁定解析后的块大小和各项预算，固定每步实际输入 token。四组配置均回放同一选择轨迹。`run_baseline.py` 将轨迹复制到每轮目录的 `selection.replay.jsonl`，记录快照哈希。

回放器按请求顺序逐行读取 `(layer,position,kv_head,ranges)`，核对范围有序、唯一、历史有效、预算及 init/local。文件过早结束、身份错位和请求末尾剩余行均触发错误。

比较器检查四组完整性、配置与结果哈希、执行成功状态、相同代码与二进制、预算、选择轨迹及逐步 logits。输出每组最大误差、首次超差、argmax 差异、首次选择差异与分项指标。

分项计时使用另一组独立目录，在 `prepare` 命令添加 `--profile`；该组四轮统一启用同步计时。端到端性能测量使用 `profile=false`、`dump_logits=false` 的独立配置。

## 6. 计时与内存口径

| 字段或节点 | 记录范围 |
| --- | --- |
| `read_requests/read_bytes`、`write_requests/write_bytes` | buffered 文件 API 的调用与 payload 字节 |
| `io_read_ms/io_write_ms` | 文件 API 调用耗时 |
| `append_ms` | FP16 转换、追加、代表更新及期间触发的回写 |
| `selection_ms` | 动态选择或固定轨迹解析 |
| `gather_inclusive_ms` | 选块后的读取、KV 复制、padding 与 mask 构造 |
| `gather_copy_bytes` | 从块缓冲复制到 attention 工作集的有效 KV 字节 |
| `streaming_attention_inclusive_ms` | CPU 流式 attention，包含历史块读取 |
| `projection_initialization_ms` | 融合权重与 bias 的分配、读取、转换及设置 |
| `projection_extra_weight_bytes` | 融合权重与 bias 的后端 buffer 容量 |
| `profile.nodes` | 按 prefill/decode、GGML op 与张量名聚合的调用、输出字节及时间 |
| `profile.transfers` | 按 prefill/decode、H2D/D2H/H2H/D2D 与张量名聚合的复制调用、字节及时间 |

观测模式逐节点同步，并在跨后端复制前后同步，结果标记为 `serialized_nodes_and_transfers`。它记录含后端调用与同步的分项耗时。关注 `solid_kv_projection`、`solid_rope_key_writeback`、`solid_kv_upload` 和 `FLASH_ATTN_EXT`。

表中 inclusive 项含其内部文件耗时；分析时按调用层级读取。设备实际流量、页缓存、进程 RSS 与设备显存峰值在实验环境额外记录。

## 7. 完成检查

环境沿用阶段 1 的 Ubuntu 24.04、GCC 13.3、CUDA Toolkit 12.6。执行：

```bash
bash scripts/check.sh --cuda
```

结果：CUDA 编译与链接、17 项 CTest、14 项 Python 检查和源码指纹核对通过。单元计算使用 CPU 后端、临时文件与合成张量。

| 检查 | 证据 |
| --- | --- |
| 文件布局 | 两层两头、对齐、尾块、payload 原排列；完整块请求数 32 对 16，读取字节相同 |
| 旧路径回归 | FP16、短读写、写缓冲、I/O 错误、microbatch、代表与选择全部通过 |
| 融合权重 | F32、F16、Q4_K、Q4_K/Q6_K；相同类型量化行逐字节保留 |
| 位置与 bias | 多头、多 token、独立投影公式，Normal/NeoX RoPE，K 回写与 V 保留 |
| Attention | 两种布局、内存/SSD、连续融合 KV 输入、GQA mask、padding、CPU Flash Attention 与流式参考 |
| 选择回放 | 集合与输出一致；错头、未来位置、重叠、遗漏 local、文件过早结束及剩余行拒绝 |
| 观测 | CPU 图输出、prefill/decode 分项、次数与输出字节 |
| 消融工具 | 四组完整性、实际输入继承、预算锁定、选择差异、文件变更与来源校验 |
| 上游来源 | 三个构建目录生成文件可重复生成；submodule 源文件保持原样 |

模型级数值、CUDA 实际复制与执行、任务准确率、端到端延迟和资源峰值按上面的实验入口核查。开发检查记录见 [CHECKS](../CHECKS.md)，下一阶段任务见 [ROADMAP](../ROADMAP.md#阶段-3历史预取与纠错)。
