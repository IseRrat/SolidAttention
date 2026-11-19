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
    for label, matrix, is_query in (("Qcur", "wq", "true"), ("Kcur", "wk", "false")):
        anchor = f"        {label} = build_lora_mm(layer.{matrix}, cur, layer.{matrix}_s);\n"
        graph = replace_once(graph, anchor, anchor + f"        solid::adapter_mark_projection(*this, {label}, il, {is_query});\n")
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
    cuda = (source.parent / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    cuda = '#include "solid/cuda_completion.cuh"\n#include "solid/device_ops.cuh"\n' + cuda
    anchor = '''        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;'''
    cuda = replace_once(cuda, anchor, '''        case GGML_OP_CUSTOM:
            if (!solid::device_op_supported(dst)) return false;
            CUDA_CHECK(solid::cuda_device_op(dst, ctx.stream()));
            break;
''' + anchor)
    anchor = '''        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);'''
    cuda = replace_once(cuda, anchor, '''        case GGML_OP_CUSTOM:
            return solid::device_op_supported(op);
''' + anchor)
    anchor = '''        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;'''
    cuda = replace_once(cuda, anchor, '''        case GGML_OP_FLASH_ATTN_EXT: {
            auto * solid_completion = solid::stream_completion(dst->op_params, sizeof(dst->op_params));
            if (solid_completion) CUDA_CHECK(solid::cuda_attention_begin(solid_completion, ctx.stream()));
            ggml_cuda_flash_attn_ext(ctx, dst);
            if (solid_completion) CUDA_CHECK(solid::cuda_attention_end(solid_completion, ctx.stream()));
            break;
        }''')
    anchor = '                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);'
    cuda = replace_once(cuda, anchor, '''                if (node->op == GGML_OP_MUL_MAT) {
                    if (auto * interval = solid::device_timing(node)) interval->record(cuda_ctx->stream(), false);
                }
''' + anchor)
    anchor = 'int ggml_backend_cuda_get_device_count() {\n'
    cuda = replace_once(cuda, anchor, '''extern "C" void * solid_cuda_queue(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    return ctx->stream();
}

''' + anchor)
    sycl = (source.parent / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")
    sycl = ('#include "solid/sycl_backend.hpp"\n#include "solid/sycl_flash_decode.hpp"\n'
            '#include "solid/sycl_device_ops.hpp"\n#include <memory>\n#include <stdexcept>\n') + sycl
    sycl = replace_once(sycl, '    bool is_usm_system;\n', '    bool is_usm_system;\n    bool solid_shared = false;\n')
    anchor = '''            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);'''
    sycl = replace_once(sycl, anchor, '''            if (solid_shared)
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(dev_ptr, *stream)));
            else if (is_usm_system)
                free_aligned_mem_host(dev_ptr);''')
    anchor = 'static bool is_bmg_g31_arch(int device) {\n'
    sycl = replace_once(sycl, anchor, '''extern "C" bool solid_sycl_shared_kv_supported(ggml_backend_t backend) {
    if (!ggml_backend_is_sycl(backend)) return false;
    auto * type = static_cast<ggml_backend_sycl_buffer_type_context *>(ggml_backend_get_default_buffer_type(backend)->context);
    const auto device=type->stream->get_device();
    return device.get_info<sycl::info::device::host_unified_memory>() && device.has(sycl::aspect::usm_shared_allocations);
}
extern "C" ggml_backend_buffer_t solid_sycl_alloc_shared_kv(ggml_backend_t backend, size_t bytes) {
    if (!ggml_backend_is_sycl(backend) || !bytes) throw std::invalid_argument("invalid SYCL KV buffer request");
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    auto * type = static_cast<ggml_backend_sycl_buffer_type_context *>(buft->context);
    auto & queue = *type->stream;
    const auto device = queue.get_device();
    if (!device.get_info<sycl::info::device::host_unified_memory>() || !device.has(sycl::aspect::usm_shared_allocations)) return nullptr;
    auto * ptr = sycl::aligned_alloc_shared(SYCL_BUFFER_ALIGNMENT, bytes, queue);
    if (!ptr) throw std::bad_alloc();
    std::unique_ptr<ggml_backend_sycl_buffer_context> context;
    try { context = std::make_unique<ggml_backend_sycl_buffer_context>(type->device, ptr, type->stream, false); }
    catch (...) { sycl::free(ptr, queue); throw; }
    context->solid_shared = true;
    auto * buffer = ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, context.get(), bytes);
    if (!buffer) throw std::bad_alloc();
    context.release();
    return buffer;
}

''' + anchor)
    anchor = 'static bool is_bmg_g31_arch(int device) {\n'
    sycl = replace_once(sycl, anchor, '''extern "C" void * solid_sycl_queue(ggml_backend_t backend) {
    if (!ggml_backend_is_sycl(backend)) throw std::invalid_argument("expected a SYCL backend");
    return static_cast<ggml_backend_sycl_context *>(backend->context)->stream();
}

''' + anchor)
    anchor = '''    // Reserve the additional scratch so it's visible to the graph allocator
'''
    sycl = replace_once(sycl, anchor, '''    if (solid::sycl_masked_decode_supported(tensor)) return solid::sycl_masked_decode_alloc(tensor);
''' + anchor)
    anchor = '''        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;'''
    sycl = replace_once(sycl, anchor, '''        case GGML_OP_CUSTOM:
            if (!solid::sycl_device_op(*ctx.stream(), dst)) return false;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (solid::sycl_masked_decode_supported(dst)) solid::sycl_attention_decode(*ctx.stream(), dst);
            else ggml_sycl_flash_attn_ext(ctx, dst);
            break;''')
    anchor = '''            return op->src[5] == nullptr && ggml_sycl_flash_attn_ext_supported(device, op);'''
    sycl = replace_once(sycl, anchor, '''            return solid::sycl_masked_decode_supported(op) ||
                   (op->src[5] == nullptr && ggml_sycl_flash_attn_ext_supported(device, op));''')
    anchor = '''        case GGML_OP_FLASH_ATTN_EXT:
            // K/V row lists (ggml_flash_attn_ext_rows) are not supported'''
    sycl = replace_once(sycl, anchor, '''        case GGML_OP_CUSTOM:
            return solid::device_op_supported(op);
''' + anchor)
    anchor = '''        const ggml_op node_op = node->op;

        if (!graph_node_is_in_local_device(device_buft, node))'''
    sycl = replace_once(sycl, anchor, '''        const ggml_op node_op = node->op;

        // SolidAttention publishes a fresh host notification for every launch.
        if (solid::sycl_masked_decode_supported(node) || solid::device_op(node) != solid::DeviceOp::none ||
            (node_op == GGML_OP_MUL_MAT && solid::device_timing(node))) return false;

        if (!graph_node_is_in_local_device(device_buft, node))''')
    anchor = '        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);'
    sycl = replace_once(sycl, anchor, '''        if (node->op == GGML_OP_MUL_MAT) {
            if (auto * interval = solid::device_timing(node)) interval->record(sycl_ctx->stream(), false);
        }
''' + anchor)
    destination.mkdir(parents=True, exist_ok=True)
    for name, text in (("llama-graph.cpp", graph), ("llama-context.cpp", context), ("ggml-backend.cpp", backend),
                       ("ggml-cuda.cu", cuda), ("ggml-sycl.cpp", sycl)):
        target = destination / name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.source, args.destination)
