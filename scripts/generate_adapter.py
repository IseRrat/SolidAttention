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
    anchor = "    const int64_t n_embd_q = n_embd_head_q * n_head_q;\n"
    graph = replace_once(graph, anchor, '''    if (solid::adapter_fused_projection(&model)) {
        return solid::adapter_qkv(*this, layer, cur, n_embd_head_q, n_head_q, n_embd_head_k, n_head_k,
                                  n_embd_head_v, n_head_v, il, reshape);
    }
''' + anchor)
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
    backend = (source.parent / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    backend = '#include "solid/profile.hpp"\n' + backend
    # Copy scopes synchronize only during profiling. Prefetched reads have their own
    # scope; the consuming split returns before opening a second scope.
    helper = '''
namespace {
solid_transfer_callback solid_transfer_observer = nullptr;
void * solid_transfer_userdata = nullptr;
struct solid_copy_scope {
    const ggml_tensor * tensor;
    ggml_backend_t source;
    ggml_backend_t target;
    std::chrono::steady_clock::time_point start;
    solid_copy_scope(const ggml_tensor * t, ggml_backend_t s, ggml_backend_t d) : tensor(t), source(s), target(d) {
        if (solid_transfer_observer) {
            ggml_backend_synchronize(source);
            ggml_backend_synchronize(target);
            start = std::chrono::steady_clock::now();
        }
    }
    ~solid_copy_scope() {
        if (solid_transfer_observer) {
            ggml_backend_synchronize(target);
            ggml_backend_synchronize(source);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            solid_transfer_observer(tensor, source, target, ns, solid_transfer_userdata);
        }
    }
};
}
extern "C" void solid_set_transfer_callback(solid_transfer_callback callback, void * userdata) {
    solid_transfer_observer = callback;
    solid_transfer_userdata = userdata;
}
'''
    anchor = "static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched,"
    backend = replace_once(backend, anchor, helper + "\n" + anchor)
    anchor = "    if (input->flags & GGML_TENSOR_FLAG_INPUT) {\n"
    backend = replace_once(backend, anchor, "    solid_copy_scope solid_copy(input, input_backend, split_backend);\n\n" + anchor)
    anchor = "        ggml_backend_tensor_get_async(backend, input, input_cpy->data, 0, ggml_nbytes(input));\n"
    backend = replace_once(backend, anchor, "        solid_copy_scope solid_copy(input, backend, sched->backends[next->backend_id]);\n" + anchor)
    destination.mkdir(parents=True, exist_ok=True)
    for name, text in (("llama-graph.cpp", graph), ("llama-context.cpp", context), ("ggml-backend.cpp", backend)):
        target = destination / name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.source, args.destination)
