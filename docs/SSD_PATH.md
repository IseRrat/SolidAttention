# 同步 SSD 路径

## 运行模式

统一入口为 `solid-baseline --config FILE`。实验启动使用 `scripts/run_baseline.py`，配置与日志保存在每轮独立目录。

| 配置 | 存储与 attention |
| --- | --- |
| `qwen2.5-7b-q4_k_m.json` | 原版 llama.cpp，全内存 KV |
| `indram-reference.json` | 内存 KV，流式 dense prefill，GGML Flash Attention decode |
| `offload.json` | SSD KV，全量历史读取，GGML Flash Attention decode |
| `indram-sparse.json` | 内存 KV，动态选块，GGML Flash Attention decode |
| `offload-sparse.json` | SSD KV，动态选块，GGML Flash Attention decode |
| `offload-interleaved.json` | SSD dense，交错布局与融合 KV 投影 |
| `offload-sparse-interleaved.json` | SSD sparse，交错布局与融合 KV 投影 |
| `indram-sparse-interleaved.json` | 内存 sparse，交错工作集与融合 KV 投影 |

`attention.gpu_decode=false` 将 decode 切换到 CPU 流式数值参考。Flash Attention 的计算后端随模型层的分配，由 GGML 调度；选块和同步文件读取在 CPU 执行。

## llama.cpp 接入

`scripts/generate_adapter.py` 从固定上游生成构建目录中的 `llama-context.cpp`、`llama-graph.cpp` 和 `ggml-backend.cpp`。生成器校验替换位置，CMake 编译生成的文件；上游 submodule 保留原样。

- context 为原生 KV 管理器保留位置元数据，以零字节 dummy buffer 代替完整 KV 分配。
- graph 将已完成投影与 RoPE 的 Q/K/V 接入 `AttentionNode`，沿用原版输出投影、残差和 FFN。
- prefill 按配置 microbatch 执行，CPU attention 逐块读取历史，在线累计 softmax。
- decode 在 CPU 收集 FP16 K/V 和各 query head 的 mask，通过视图接入 GGML Flash Attention。
- `AttentionAdapter::check()` 在每次计算完成后传递 I/O、位置和数值错误；`flush()` 完成 prefill 与请求末尾的回写。

阶段 2 的 `layout`、`fuse_kv_projection`、`profile` 和 `selection_replay` 字段控制交错布局、单次 KV 投影、分项计时及固定集合回放。实现、数据形状和四组消融入口见 [阶段 2 完成记录](phases/PHASE_2.md)。

入口支持 Llama、Qwen2 的单序列文本 decode、统一层形状和 FP16 KV。模型形状在 adapter 构造时检查。每个模型绑定一个会话，位置从 0 连续追加。

## KV 文件格式

`FileKvStore` 使用独占创建的会话文件、buffered I/O 和每层 32 KiB 写缓冲。读取合并已提交内容与待写入 token；每次 flush 完成全部相关写入后推进已提交长度。短传输按剩余字节重试，I/O 失败使会话进入错误状态。

文件头为 4096 字节，起始 8 字节为 `SOLIDKV1`。随后依次为 little-endian uint64：版本 1、头部长度、层数、KV head 数、head 维度、块 token 数、上下文容量、对齐、布局、元素字节数 2。其余头部填零，payload 为 IEEE FP16 little-endian。

```text
blocks_per_head = ceil(capacity_tokens / block_tokens)
plane_bytes = block_tokens * head_dimension * 2
plane_stride = align_up(plane_bytes, alignment)
slot_bytes = layout == separate ? 2 * plane_stride : align_up(2 * plane_bytes, alignment)
offset = 4096 + ((layer * kv_heads + head) * blocks_per_head + block) * slot_bytes
K_offset = offset
V_offset = offset + plane_stride
```

`layout=separate` 使用独立 K/V plane，上面的 K/V 偏移对应此模式。`layout=interleaved` 使用 `[B,2,D]` payload，token `i` 的 K 偏移为 `offset+i*2*D*2`，V 紧随 K。读取保留原排列；完整已提交块分别发出两次和一次读取。会话的有效长度和块版本由进程维护；`keep_session_file=true` 保留文件供检查。

## 选块规则

自动预算取 `max(1, floor(prompt_tokens / 4))`，输入达到 4096 时取 1024。确定性预算为 `max(1, floor(total / 2))`，init 取 `min(block_tokens, floor(deterministic / 2))`，local 使用确定性预算的剩余部分。长输入默认 init=32、local=480、selected 最多 512。

显式配置 `context_budget_tokens` 时，同时填写 `init_tokens` 和 `local_tokens`。实际集合按 token 去重并限制在总预算内，selected 的最后一块按剩余 token 数裁剪。候选不足时记录实际集合长度；padding 的 mask 为负无穷。

代表评分按 [InfLLM §3.2](https://arxiv.org/html/2402.04617) 的后续 local 窗口计算。GQA 的 query head 在对应 KV head 组内取均值。每个 token 收齐后续窗口后参与块内排名，完整块的全部 token 完成评分后定稿；块代表默认取 1 个，数量由 `representative_count` 指定。

块相关度为当前组内平均 Q 与代表 K 的点积之和。分数并列按块号排序，代表 token 并列按绝对位置排序。Q/K 使用上游已完成的原生 RoPE，块内 KV 按绝对位置访问。

`selection.jsonl` 按 layer、position、KV head 记录实际 token 的连续区间 `[first, count]`。结果文件记录最终预算、head 聚合、位置规则、I/O 请求与字节数，以及 KV、代表向量和跟踪工作区的 buffer 容量。进程 RSS 与后端显存分配由实验日志单独记录。

## 固定 token 数值对照

在实验环境给参考配置设置 `dump_logits=true`，先保存参考结果。`prepare_replay.py` 复用参考目录的模型、输入、上下文及生成设置，模式模板选择 attention 路径：

```bash
python3 scripts/prepare_replay.py results/reference/result.json configs/offload.json configs/offload.local.json --output results/offload-check/result.json
python3 scripts/run_baseline.py configs/offload.local.json
python3 scripts/compare_logits.py results/reference/result.json results/offload-check/result.json --atol 0.001 --rtol 0.0001
```

回放固定每步实际送入模型的 token，保存 FP32 logits、文件校验和、输入 token 与停止原因。比较器核对模型分片、输入和矩阵形状，输出最大误差、首次超差位置和 argmax 差异。内存稀疏与 SSD 稀疏使用相同预算及选块规则进行对照。性能配置关闭 `dump_logits`。
