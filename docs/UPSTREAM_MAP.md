# llama.cpp 接口定位

基点：`ef6bc6f367082338006c8379bc7bcab3e1584766`，2026-11-16。入口由 Git submodule 与 `cmake/Upstream.cmake` 同时固定。CMake 配置时检查提交号。

## 调用路径

```text
solid-baseline
  llama_model_load_from_file
  llama_init_from_model
    llama_model::create_memory
  llama_batch_ext_add_token / set_pos / set_output_logits
  llama_process(DECODE)
    llama_context::decode
      llama_memory_i::init_batch
      llama_context::process_ubatch
        llama_context::graph_params
        llama_model_llama::graph / llama_model_qwen2::graph
        llm_graph_context::build_qkv / build_rope / build_attn
      ggml_backend_sched_graph_compute_async
  llama_synchronize
  llama_get_logits_ith
```

## 接入点

以下路径相对于 `third_party/llama.cpp`，符号对应锁定提交。

| 接入点 | 文件与符号 | 形状与行为 |
| --- | --- | --- |
| 模型参数 | `include/llama.h`：`llama_model_n_layer/n_head/n_head_kv`、`llama_model_meta_*` | 元数据含 architecture、head 维度、RoPE、具体 GGUF 量化格式 |
| 内存类型选择 | `src/llama-model.cpp`：`llama_model::create_memory` | 按模型结构创建 KV、SWA 或 hybrid memory |
| KV 生命周期 | `src/llama-memory.h`：`llama_memory_i`、`llama_memory_context_i` | init_batch、apply、next 与状态传播 |
| KV 分配 | `src/llama-kv-cache.cpp`：`llama_kv_cache` 构造器 | 当前实现按层分配历史容量；SSD 路径需替换存储与寻址 |
| KV 追加 | `src/llama-kv-cache.cpp`：`apply_ubatch`、`cpy_k`、`cpy_v` | 逻辑 cell 与张量行索引配对 |
| K 视图 | `src/llama-kv-cache.cpp`：`get_k` | `[Dk, Hkv, Nkv, Nstream]`，nb[1] 为 head stride，nb[2] 为 token stride |
| V 视图 | `src/llama-kv-cache.cpp`：`get_v` | 普通视图 `[Dv,Hkv,Nkv,Nstream]`；转置布局为 `[Nkv,Hkv,Dv,Nstream]` |
| 投影与位置 | `src/models/llama.cpp`：`llama_model_llama::graph` | `build_qkv` 后对 Q/K 执行 `build_rope`，再进入 attention |
| Qwen2.5 | `src/models/qwen2.cpp`：`llama_model_qwen2::graph` | 检查 Q/K/V bias、RoPE 类型和 GQA 映射 |
| 选块接入 | `src/llama-graph.cpp`：`llm_graph_context::build_attn(llm_graph_input_attn_kv *)` | Q/K/V 已生产；当前路径先 cpy_k/v，再 get_k/v，最后 build_attn_mha |
| Attention | `src/llama-graph.cpp`：`build_attn_mha` | 先 permute，Flash Attention 使用 F16 K/V 与 F32 累加；mask 与 kv_rows 为独立输入 |
| 位置与 mask | `src/llama-kv-cache.cpp`：`set_input_attn_mask` | mask 使用逻辑位置与序列归属，块映射需保留此语义 |
| microbatch | `src/llama-batch.cpp`：`llama_batch_allocr::split_simple/split_equal/split_seq` | 划分计算批次，历史 KV 的容量由 memory 实现管理 |
| 后端调度 | `ggml/include/ggml-backend.h`：`ggml_backend_sched_*` | tensor backend、图执行、复制与同步 |
| 观测回调 | `ggml/include/ggml-backend.h`：`ggml_backend_sched_eval_callback` | ask 阶段选择观测节点，执行阶段读取结果；观测模式会引入同步 |

## SSD 路径的改动范围

接入由 `cmake/Adapter.cmake` 与 `scripts/generate_adapter.py` 生成两个构建目录编译单元，调用主仓库的 `src/llama_adapter.cpp`。具体数据流与模式见 [SSD_PATH](SSD_PATH.md)。

1. 由独立存储层管理完整历史 KV，向计算图提供预算内的每层工作集。
2. 在 Q 就绪后评分，按 KV head 收集逻辑块，生成对应位置与 mask。
3. 将 K/V 写入、SSD 读入、H2D、attention 和释放槽位表达为明确依赖。
4. prefill 的 microbatch 继续复用原版投影；历史 attention 改为有界 KV tile 累加。
5. 用全内存/同步 SSD 对照检查 logits、KV、位置及最终选择集合，再接预取和异步调度。

## 计时边界

运行器在每次 `llama_process` 后调用 `llama_synchronize`。prefill 时间覆盖 prompt 全部分批处理；首个生成 token 使用 prefill logits。逐 token 记录将该 token 标记为 `prefill`，后续 token 标记为 `decode`。`decode_steps` 为后续实际处理步数，EOS token ID 与停止原因进入结果文件。
