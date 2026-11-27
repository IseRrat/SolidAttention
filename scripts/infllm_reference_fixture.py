#!/usr/bin/env python3
# 调用仓库固定版本的原版 InfLLM，生成代表和选块对照样例。
"""Export CPU fixtures by executing the pinned, unmodified InfLLM methods."""
import argparse
import importlib
import json
from pathlib import Path
import struct
import sys
import types

import torch


def upstream(root):
    # Load the real attention modules without the unrelated HF patcher/chat
    # entry point. No algorithm or torch operation is replaced.
    for name, path in (("inf_llm", root / "inf_llm"),
                       ("inf_llm.attention", root / "inf_llm/attention")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module("inf_llm.attention.context_manager")


# 在 CPU 上建立原版 InfLLM 上下文，供小规模数值对照调用。
def cpu_context(module, kv_heads, heads, width, block, initial, local, representatives, capacity):
    # Bypass CUDA allocation in __init__, retain all actual append_global,
    # MemoryUnit, get_block_k, grouped-head expansion and VectorTensor code.
    context = module.ContextManager.__new__(module.ContextManager)
    for name, value in dict(num_units=1, num_heads_kv=kv_heads, num_heads=heads,
                            unit_size=heads, unit_size_kv=kv_heads, dim_head=width,
                            block_size=block, n_init=initial, n_local=local,
                            repr_topk=representatives, init_exc=False,
                            num_global_block=0, _global_remainder_st=0,
                            _global_remainder_ed=0, cuda_cache=None,
                            pin_memory=False).items():
        setattr(context, name, value)
    context.global_remainder = tuple(torch.empty(1, kv_heads, 0, width, dtype=torch.float16) for _ in range(2))
    context.global_remainder_local_score = torch.empty(1, heads, 0, dtype=torch.float16)
    context.init_k = torch.empty(1, kv_heads, 0, width, dtype=torch.float16)
    context.init_v = torch.empty_like(context.init_k)
    context.global_blocks = [[]]
    bank = module.VectorTensor.__new__(module.VectorTensor)
    bank.hidden_size = heads * width
    bank.length = 0
    bank.cache_size = capacity
    bank.data = torch.empty(capacity, bank.hidden_size, dtype=torch.float16)
    context.block_k = [bank]
    return context


# 运行原版代表构造和检索，将输入与期望结果写入对照文件。
def write_fixture(output, module):
    cases = [
        # KV heads, query heads, D, block, init, local, total budget, capacity, chunks
        (1, 2, 32, 32, 32, 96, 256, 1024, [96] * 8 + [1] * 32),
        (4, 28, 32, 32, 32, 480, 1024, 2048, [480] * 3 + [96] + [1] * 32),
        (2, 4, 32, 256, 3, 257, 1024, 2048, [257] * 5 + [251] + [1] * 32),
        (1, 3, 32, 7, 3, 11, 64, 256, [11] * 6 + [7] + [1] * 16),
        (1, 2, 32, 32, 32, 512, 2048, 4096, [512] * 3 + [1] * 32),
    ]
    summary = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        file.write(b"IFLMREF3")
        file.write(struct.pack("<I", len(cases)))
        for kv_heads, heads, width, block, initial, local, total, capacity, chunks in cases:
            representatives = 4
            context = cpu_context(module, kv_heads, heads, width, block, initial, local, representatives, capacity)
            file.write(struct.pack("<10I", kv_heads, heads, width, block, initial, local,
                                   total, capacity, representatives, len(chunks)))
            first = 0
            unique_blocks = []
            for frame, count in enumerate(chunks):
                assert count <= min(512, local)
                end = first + count
                origin = max(0, first + 1 - local)
                p = torch.arange(first, end)[:, None, None, None]
                h = torch.arange(kv_heads)[None, :, None, None]
                component = torch.arange(2)[None, None, :, None]
                d = torch.arange(width)[None, None, None, :]
                kv = (((p * 11 + h * 7 + component * 5 + d * 3) % 127 - 63) / 32).half()
                scores = ((torch.arange(origin, end)[:, None] * 13 + torch.arange(heads)[None, :] * 7 + frame * 3) % 97 + 1).float() / 256
                # Deliberately exercise rounding at both the chunk conversion
                # and accumulation boundaries, including values below one ULP.
                scores += (frame % 3) * 0.000013
                keys = kv[:, :, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
                values = kv[:, :, 1].permute(1, 0, 2).unsqueeze(0).contiguous()
                context.global_remainder = tuple(torch.cat((old, new), dim=-2)
                                                 for old, new in zip(context.global_remainder, (keys, values)))
                context.global_remainder_local_score = torch.cat((context.global_remainder_local_score,
                    torch.zeros(1, heads, count, dtype=torch.float16)), dim=-1)
                upstream_origin = max(0, first - local)
                local_score = torch.zeros(1, heads, end - upstream_origin, dtype=torch.float16)
                local_score[:, :, origin - upstream_origin:] = scores.T.half()
                before = context.num_global_block
                context.append_global(count, end - upstream_origin, local_score)
                file.write(struct.pack("<QI", first, count))
                file.write(kv.numpy().tobytes())
                file.write(scores.contiguous().numpy().tobytes())
                file.write(struct.pack("<I", context.num_global_block - before))
                for block_id in range(before, context.num_global_block):
                    start = initial + block_id * block
                    block_scores = context.global_remainder_local_score[0, :, start:start + block]
                    ordered = block_scores.sort(dim=-1, descending=True).values
                    if torch.all(ordered[:, representatives - 1] > ordered[:, representatives]):
                        unique_blocks.append(block_id)
                    picked = block_scores.topk(representatives, dim=-1).indices + start
                    mean = context.block_k[0].get_data()[block_id]
                    # get_block_k is executed once more independently of the
                    # stored bank to catch any fixture layout mistake.
                    direct = context.get_block_k(context.global_remainder[0][:, :, start:start + block],
                        block_scores.unsqueeze(0)).mean(dim=-2).reshape(-1)
                    assert torch.equal(mean, direct)
                    file.write(struct.pack("<Q", block_id))
                    file.write(block_scores.contiguous().numpy().tobytes())
                    file.write(picked.to(torch.int64).contiguous().numpy().tobytes())
                    file.write(mean.contiguous().numpy().tobytes())
                first = end
            ranking_cases = write_rankings(file, module, context, unique_blocks, heads, width, capacity)
            summary.append(dict(kv_heads=kv_heads, query_heads=heads, block_tokens=block,
                                frames=len(chunks), tokens=first, published_blocks=context.num_global_block,
                                retrieval_cases=ranking_cases, unique_representative_blocks=len(unique_blocks)))
    return summary


def write_rankings(file, module, context, blocks, heads, width, capacity):
    # Exclude blocks with a tied representative boundary: torch.topk permits
    # a different legal representative subset, hence a different mean vector.
    # Every retained candidate has exactly the same bank in both runtimes.
    if not blocks:
        file.write(struct.pack("<I", 0))
        return 0
    # The llama.cpp adapter produces F32 queries and stores KV/representatives
    # in F16. Promote stored values for the original matmul, preserving every
    # stored value. Upstream inherits model dtype, not a fixed F16 score type.
    bank = context.block_k[0].get_data()[blocks].float().contiguous()
    retrieval = module.ContextManager.__new__(module.ContextManager)
    retrieval.num_units, retrieval.unit_size, retrieval.dim_head = 1, heads, width
    retrieval.num_global_block = len(blocks)
    retrieval.global_blocks = [[None] * len(blocks)]
    retrieval._use_chunk_topk = False
    vectors = module.VectorTensor.__new__(module.VectorTensor)
    vectors.data, vectors.length, vectors.hidden_size = bank, len(blocks), heads * width
    retrieval.block_k = [vectors]
    limit = min(512, context.n_local, capacity)
    queries = [(count, 1) for count in dict.fromkeys((1, min(37, limit), limit))]
    queries.append((1, 1048576))
    file.write(struct.pack("<I", len(queries)))
    for query_case, (count, scale) in enumerate(queries):
        raw = ((torch.arange(count * heads * width).reshape(count, heads, width) * 19 + query_case * 7) % 211 - 105).float() / 73
        raw += 0.000013
        raw *= scale
        global_q = raw.permute(1, 0, 2).unsqueeze(0).contiguous()
        mean = global_q.mean(dim=2).reshape(-1)
        logits = torch.matmul(bank, mean[:, None]).reshape(-1)
        assert torch.isfinite(logits).all()
        if scale > 1:
            assert raw.abs().max() > torch.finfo(torch.float16).max
            assert logits.abs().max() > torch.finfo(torch.float16).max
        retrieval.topk = min(2, len(blocks))
        picks = retrieval.calc_block_topk(global_q)[0]
        file.write(struct.pack("<III", count, len(blocks), retrieval.topk))
        file.write(torch.tensor(blocks, dtype=torch.int64).numpy().tobytes())
        file.write(raw.contiguous().numpy().tobytes())
        file.write(mean.contiguous().numpy().tobytes())
        file.write(logits.contiguous().numpy().tobytes())
        file.write(torch.tensor([blocks[i] for i in picks], dtype=torch.int64).numpy().tobytes())
    return len(queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    module = upstream(root / "third_party/infllm")
    summary = write_fixture(args.output, module)
    print(json.dumps(dict(upstream_commit="12b70798f56e56ebb23c53c7018091a3f540a028",
                          torch=torch.__version__, device="cpu", cases=summary), indent=2))


if __name__ == "__main__":
    main()
