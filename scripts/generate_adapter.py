#!/usr/bin/env python3
"""Generate build-local integration files from the pinned upstream sources."""
import argparse
from pathlib import Path


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"upstream adapter anchor changed: {before[:90]!r}")
    return text.replace(before, after, 1)


def generate(source, destination):
    graph = (source / "llama-graph.cpp").read_text(encoding="utf-8")
    graph = '#include "solid/llama_adapter.hpp"\n' + graph
    anchor = "llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {\n"
    graph = replace_once(graph, anchor, anchor + "    if (solid::adapter_active(&model)) return solid::adapter_input(*this);\n")
    signature = "ggml_tensor * llm_graph_context::build_attn(\n        llm_graph_input_attn_kv * inp,"
    start = graph.index(signature)
    body = graph.index(") const {", start) + len(") const {")
    graph = graph[:body] + '''
    if (solid::adapter_active(&model)) {
        auto * result = solid::adapter_attention(*this, inp, q_cur, k_cur, v_cur, kq_scale, il);
        if (wo) result = build_lora_mm(wo, result, wo_s);
        if (wo_b) result = ggml_add(ctx0, result, wo_b);
        return result;
    }
''' + graph[body:]
    context = (source / "llama-context.cpp").read_text(encoding="utf-8")
    context = '#include "solid/llama_adapter.hpp"\n' + context
    context = replace_once(context, "/*.no_alloc  =*/ model.no_alloc(),", "/*.no_alloc  =*/ model.no_alloc() || solid::adapter_active(&model),")
    destination.mkdir(parents=True, exist_ok=True)
    for name, text in (("llama-graph.cpp", graph), ("llama-context.cpp", context)):
        target = destination / name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.source, args.destination)
