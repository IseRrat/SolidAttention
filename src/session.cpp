#include "solid/session.hpp"
#include "solid/trace.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace solid {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
void pack_token(const KvGeometry & g, const float * keys, const float * values, Half * kv) {
    const auto width = g.dimension;
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        for (std::uint32_t d = 0; d < 2 * width; ++d) {
            const auto offset = static_cast<std::size_t>(head) * 2 * width + d;
            const auto value = !values ? keys[offset] : d < width ? keys[head * width + d] : values[head * width + d - width];
            kv[offset] = to_half(value);
            if (!std::isfinite(from_half(kv[offset]))) throw std::invalid_argument("K/V cannot be represented as finite FP16");
        }
    }
}
void copy_keys(const KvGeometry & g, const Half * kv, std::vector<Half> & keys) {
    for (std::uint32_t head = 0; head < g.heads; ++head) {
        const auto offset = static_cast<std::size_t>(head) * g.dimension;
        std::copy_n(kv + offset * 2, g.dimension, keys.data() + offset);
    }
}
}
AttentionSession::AttentionSession(std::unique_ptr<KvStore> store, std::uint32_t query_heads, SparseBudget budget,
                                   bool sparse, std::uint64_t prefill_tokens, const std::filesystem::path & selection_trace,
                                   const std::filesystem::path & selection_replay, std::unique_ptr<AsyncReader> reader,
                                   std::uint32_t prefetch_slots_per_head, const std::filesystem::path & prefetch_trace,
                                   std::optional<RelativeRope> rope) :
    store_(std::move(store)), query_heads_(query_heads), budget_(budget), rope_(rope), sparse_(sparse), prefill_tokens_(prefill_tokens) {
    if (!store_ || query_heads == 0 || query_heads % store_->geometry().heads != 0 ||
        prefill_tokens == 0 || prefill_tokens > store_->geometry().capacity_tokens) throw std::invalid_argument("invalid attention session");
    if (budget_.infllm_attention != rope_.has_value() ||
        (rope_ && (!sparse_ || !rope_->dimensions || rope_->dimensions % 2 || rope_->dimensions > geometry().dimension ||
                   !std::isfinite(rope_->base) || rope_->base <= 0))) {
        throw std::invalid_argument("invalid InfLLM rotary attention parameters");
    }
    if (rope_) stats_.prefill_backend = "cpu_infllm_sparse";
    if (sparse_) representatives_ = std::make_unique<RepresentativeIndex>(store_->geometry(), query_heads, budget);
    if (reader) {
        if (!sparse_ || geometry().layout != KvLayout::interleaved) throw std::invalid_argument("prefetch requires sparse interleaved KV");
        if (!prefetch_slots_per_head) prefetch_slots_per_head = budget.cache_slots(geometry());
        prefetch_ = std::make_unique<Prefetcher>(*store_, budget, prefetch_slots_per_head, std::move(reader), prefetch_trace);
    }
    if (!selection_replay.empty()) {
        if (!sparse_) throw std::invalid_argument("selection replay requires sparse attention");
        if (!selection_trace.empty() && std::filesystem::weakly_canonical(selection_trace) == std::filesystem::weakly_canonical(selection_replay)) {
            throw std::invalid_argument("selection replay and output paths must differ");
        }
        replay_.open(selection_replay, std::ios::binary);
        if (!replay_) throw std::runtime_error("cannot open selection replay");
    }
    if (!selection_trace.empty()) {
        if (std::filesystem::exists(selection_trace)) throw std::runtime_error("selection trace already exists");
        trace_.open(selection_trace, std::ios::binary);
        if (!trace_) throw std::runtime_error("cannot create selection trace");
    }
}
std::vector<float> AttentionSession::process(std::uint32_t layer, std::uint64_t first_position, std::size_t tokens,
                                            const std::vector<float> & queries, const std::vector<float> & keys,
                                            const std::vector<float> & values, float scale, bool interleaved_inputs) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    const auto query_elements = static_cast<std::size_t>(query_heads_) * g.dimension;
    const auto kv_elements = static_cast<std::size_t>(g.heads) * g.dimension;
    if (tokens == 0 || tokens > g.capacity_tokens || first_position > g.capacity_tokens - tokens ||
        queries.size() / tokens != query_elements || queries.size() % tokens != 0 ||
        keys.size() / tokens != kv_elements * (interleaved_inputs ? 2 : 1) || keys.size() % tokens != 0 ||
        (interleaved_inputs ? !values.empty() : values.size() != keys.size())) {
        throw std::invalid_argument("attention batch shape or capacity mismatch");
    }
    if ((prefill_ || rope_) && first_position < prefill_tokens_) {
        const auto count = std::min<std::uint64_t>(tokens, prefill_tokens_-first_position);
        auto output = rope_ ? infllm_prefill(layer,first_position,count,queries,keys,values,scale,interleaved_inputs) :
            tiled_prefill(layer,first_position,count,queries,keys,values,scale,interleaved_inputs);
        if (count < tokens) {
            const auto q = queries.begin()+count*query_elements;
            const auto k = keys.begin()+count*kv_elements*(interleaved_inputs ? 2 : 1);
            std::vector<float> tail_values;
            if (!interleaved_inputs) tail_values.assign(values.begin()+count*kv_elements,values.end());
            auto tail = process(layer,first_position+count,tokens-count,{q,queries.end()},{k,keys.end()},tail_values,scale,interleaved_inputs);
            output.insert(output.end(),tail.begin(),tail.end());
        }
        return output;
    }
    std::vector<float> output(queries.size());
    std::vector<float> q(query_elements);
    std::vector<float> probabilities;
    try {
        for (std::size_t token = 0; token < tokens; ++token) {
            const auto position = first_position + token;
            std::copy_n(queries.data() + token * query_elements, query_elements, q.data());
            append_token(layer, position, q, keys.data() + token * kv_elements * (interleaved_inputs ? 2 : 1),
                         interleaved_inputs ? nullptr : values.data() + token * kv_elements);
            std::vector<HeadSelection> selection;
            const bool use_sparse = sparse_ && (rope_ || position >= prefill_tokens_);
            if (use_sparse) {
                const auto start = Clock::now();
                selection = position < prefill_tokens_ ? select_blocks(g, budget_, *representatives_, layer, position, q) :
                    select(layer, position, q);
                stats_.selection_ns += elapsed(start);
                if (position >= prefill_tokens_) record(layer, position, selection);
                for (const auto & head : selection) stats_.selected_tokens += head.positions.size();
            }
            const auto attention_start = Clock::now();
            auto attended = streamed_attention(*store_, layer, position, query_heads_, q, scale, use_sparse ? &selection : nullptr,
                rope_ ? &*rope_ : nullptr, budget_.local, rope_ ? &probabilities : nullptr);
            if (rope_) observe_probabilities(layer, position, probabilities);
            stats_.streaming_attention_ns += elapsed(attention_start);
            std::copy(attended.begin(), attended.end(), output.begin() + token * query_elements);
            ++stats_.attention_queries;
            std::uint64_t workspace = (output.capacity() + q.capacity() + attended.capacity()) * sizeof(float) +
                (3 * kv_elements + static_cast<std::size_t>(g.block_tokens) * 2 * g.dimension) * sizeof(Half) +
                static_cast<std::size_t>(query_heads_ / g.heads) * (g.dimension + 2) * sizeof(double);
            for (const auto & head : selection) workspace += (head.positions.capacity() + head.blocks.capacity()) * sizeof(std::uint64_t);
            stats_.workspace_peak_bytes = std::max(stats_.workspace_peak_bytes, workspace);
        }
    } catch (...) { failed_ = true; throw; }
    return output;
}
std::vector<float> AttentionSession::infllm_prefill(std::uint32_t layer, std::uint64_t first, std::size_t tokens,
    const std::vector<float> & queries, const std::vector<float> & keys, const std::vector<float> & values,
    float scale, bool packed, DeviceQuery device_query, DeviceOutput device_output, DeviceKvBatch device_kv,
    const std::vector<Half> * archived) {
    const auto & g = geometry();
    if (store_->length(layer) != first || !std::isfinite(scale) || scale <= 0) {
        throw std::invalid_argument("InfLLM prefill position or scale");
    }
    const auto qwidth = static_cast<std::size_t>(query_heads_) * g.dimension;
    const auto kwidth = static_cast<std::size_t>(g.heads) * g.dimension;
    const auto capacity = prefill_ ? std::min(budget_.prefill_group_tokens(), prefill_->query_capacity())
                                   : budget_.prefill_group_tokens();
    stats_.infllm_prefill_group_token_limit = capacity;
    std::vector<float> output(archived ? 0 : tokens * qwidth), mean(archived ? 0 : qwidth), q(prefill_ ? 0 : qwidth), probabilities, batch;
    std::vector<Half> row(2 * kwidth), key_row(prefill_ ? 0 : kwidth), current, planes;
    try {
        if (archived) for (const auto value : *archived) {
            if ((value & 0x7c00u) == 0x7c00u) throw std::invalid_argument("non-finite prefill KV offload");
        }
        for (std::size_t start = 0; start < tokens;) {
            const auto position = first + start;
            auto count = static_cast<std::uint32_t>(std::min<std::size_t>(capacity, tokens - start));
            if (position < budget_.total) {
                count = std::min(count, budget_.total - static_cast<std::uint32_t>(position));
            }
            const float * query = archived ? nullptr : queries.data() + start * qwidth;
            const DeviceQuery chunk_query = archived ?
                DeviceQuery{device_query.data + start * qwidth, count * qwidth, device_query.memory} : DeviceQuery{};
            const auto selection_start = Clock::now();
            const auto context = [&] {
                if (archived) {
                    const auto mean_query = position >= budget_.total ? prefill_->mean_sparse_query_async(chunk_query, count) : DeviceQueryWithEvent{};
                    return select_prefill_context(g, budget_, *representatives_, layer, position, count, mean_query);
                }
                if (position >= budget_.total) {
                    mean_infllm_query(query, mean.data(), qwidth, count);
                }
                return select_prefill_context(g, budget_, *representatives_, layer, position, count, mean);
            }();
            const auto & positions = context.positions;
            stats_.selection_ns += elapsed(selection_start);
            if (!device_probabilities()) current.resize(count * 2 * kwidth);
            const auto append_start = Clock::now();
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto offset = (start + i) * kwidth;
                if (archived) std::copy_n(archived->data() + offset * 2, row.size(), row.data());
                else pack_token(g, keys.data() + offset * (packed ? 2 : 1), packed ? nullptr : values.data() + offset, row.data());
                store_->append_interleaved(layer, position + i, row);
                if (!device_probabilities()) std::copy(row.begin(), row.end(), current.data() + i * 2 * kwidth);
            }
            stats_.append_ns += elapsed(append_start);

            const auto attention_start = Clock::now();
            SparsePrefillOutput device;
            if (prefill_) {
                const auto rows = positions.size();
                const auto stride = static_cast<std::size_t>(2) * g.dimension;
                prefill_tile_.resize(rows * g.heads * stride);
                for (std::uint32_t head = 0; head < g.heads; ++head) {
                    auto * target = prefill_tile_.data() + head * rows * stride;
                    std::size_t i = 0;
                    while (i < rows && positions[i] < position) {
                        const auto begin = i++;
                        while (i < rows && positions[i] < position && positions[i] == positions[i - 1] + 1) ++i;
                        store_->read_rows(layer, head, positions[begin], static_cast<std::uint32_t>(i - begin),
                                          target + begin * stride, nullptr, planes);
                    }
                    for (; !archived && i < rows; ++i) {
                        const auto src = (positions[i] - position) * 2 * kwidth + head * stride;
                        std::copy_n(current.data() + src, stride, target + i * stride);
                    }
                }
                if (!archived) batch.assign(query, query + count * qwidth);
                const DeviceOutput chunk_output = archived ?
                    DeviceOutput{device_output.data + start * qwidth, count * qwidth, device_output.memory} : DeviceOutput{};
                const DeviceKvBatch chunk_kv = archived ?
                    DeviceKvBatch{device_kv.data + start * 2 * kwidth, count * 2 * kwidth, device_kv.memory} : DeviceKvBatch{};
                device = prefill_->sparse({position, count, scale, budget_.local, *rope_, positions, batch,
                                           prefill_tile_, context.local_only_from, chunk_query, chunk_output, chunk_kv,
                                           device_probabilities()});
                if (!archived) {
                    if (device.attention.size() != count * qwidth) throw std::runtime_error("sparse prefill output shape");
                    std::copy(device.attention.begin(), device.attention.end(), output.data() + start * qwidth);
                } else {
                    stats_.direct_prefill_queries += count;
                    stats_.prefill_kv_download_bytes += count * 2 * kwidth * sizeof(Half);
                }
                if (device_probabilities()) {
                    representatives_->observe_prefill_device_scores(layer, position, count, chunk_kv, device.device_scores);
                } else representatives_->observe_prefill_scores(layer, position, count, current, device.key_scores);
            }
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto p = position + i;
                const auto visible = std::upper_bound(positions.begin(), positions.end(), p);
                const auto recent = std::min<std::uint64_t>(p + 1, budget_.local);
                const auto local_only = std::min(positions.begin() + context.local_only_from, visible);
                const auto local_begin = std::lower_bound(local_only, visible, p + 1 - recent);
                if (!prefill_) {
                    std::copy_n(query + i * qwidth, qwidth, q.data());
                    HeadSelection selection{{positions.begin(), local_only}, {}};
                    selection.positions.insert(selection.positions.end(), local_begin, visible);
                    const std::vector<HeadSelection> selected(g.heads, selection);
                    const auto attended = streamed_attention(*store_, layer, p, query_heads_, q, scale,
                        &selected, &*rope_, budget_.local, &probabilities);
                    std::copy(attended.begin(), attended.end(), output.data() + (start + i) * qwidth);
                    copy_keys(g, current.data() + i * 2 * kwidth, key_row);
                    representatives_->observe(layer, p, q, key_row);
                    observe_probabilities(layer, p, probabilities);
                }
                stats_.selected_tokens += ((local_only - positions.begin()) + (visible - local_begin)) * g.heads;
            }
            if (prefill_) stats_.tiled_prefill_ns += elapsed(attention_start);
            else stats_.streaming_attention_ns += elapsed(attention_start);
            stats_.attention_queries += count;
            const auto workspace = (output.capacity() + mean.capacity() + q.capacity() + probabilities.capacity() +
                                    device.attention.capacity() + device.key_scores.capacity()) * sizeof(float) +
                (row.capacity() + key_row.capacity() + current.capacity() + planes.capacity()) * sizeof(Half);
            stats_.workspace_peak_bytes = std::max<std::uint64_t>(stats_.workspace_peak_bytes, workspace);
            ++stats_.infllm_prefill_groups;
            if (position >= budget_.total) ++stats_.infllm_prefill_rank_groups;
            start += count;
        }
    } catch (...) {
        failed_ = true;
        if (prefill_) try { prefill_->abort(); } catch (...) {}
        throw;
    }
    return output;
}
void AttentionSession::observe_probabilities(std::uint32_t layer, std::uint64_t position, const std::vector<float> & probabilities) {
    if (failed_ || !rope_ || early_) throw std::logic_error("InfLLM attention receipt without an active query");
    try { representatives_->observe_probabilities(layer, position, probabilities); }
    catch (...) { failed_ = true; throw; }
}
void AttentionSession::observe_device_probabilities(std::uint32_t layer, std::uint64_t position, DeviceKvBatch kv, DeviceQuery scores,
    void * compute_stream, bool on_compute) {
    if (failed_ || !rope_ || early_) throw std::logic_error("device probability receipt without an active query");
    try { representatives_->observe_device_probabilities(layer, position, 1, kv, scores, false, compute_stream, on_compute); }
    catch (...) { failed_ = true; throw; }
}
void AttentionSession::finish_device_probabilities() {
    try { if (representatives_) representatives_->finish_device_probabilities(); }
    catch (...) { failed_ = true; throw; }
}
std::vector<float> AttentionSession::tiled_prefill(std::uint32_t layer, std::uint64_t first, std::size_t tokens,
    const std::vector<float> & queries,const std::vector<float> & keys,const std::vector<float> & values,float scale,bool packed) {
    const auto & g=geometry();
    const auto qwidth=static_cast<std::size_t>(query_heads_)*g.dimension;
    const auto kwidth=static_cast<std::size_t>(g.heads)*g.dimension;
    std::vector<float> output(tokens*qwidth), query(qwidth);
    try {
        for (std::size_t start=0; start<tokens;) {
            const auto count=static_cast<std::uint32_t>(std::min<std::size_t>(prefill_->query_capacity(),tokens-start));
            std::vector<float> batch(queries.begin()+start*qwidth,queries.begin()+(start+count)*qwidth);
            for (std::uint32_t i=0;i<count;++i) {
                std::copy_n(batch.data()+i*qwidth,qwidth,query.data());
                append_token(layer,first+start+i,query,keys.data()+(start+i)*kwidth*(packed?2:1),packed?nullptr:values.data()+(start+i)*kwidth);
            }
            const auto attention_start=Clock::now();
            prefill_->begin(first+start,count,scale,batch);
            const auto end=first+start+count;
            std::vector<Half> planes;
            for (std::uint64_t tile=0;tile<end;tile+=prefill_->tile_capacity()) {
                const auto valid=static_cast<std::uint32_t>(std::min<std::uint64_t>(prefill_->tile_capacity(),end-tile));
                for (std::uint32_t h=0;h<g.heads;++h) {
                    auto * target = prefill_tile_.data() + static_cast<std::size_t>(h) * prefill_->tile_capacity() * 2 * g.dimension;
                    store_->read_rows(layer, h, tile, valid, target, nullptr, planes);
                }
                prefill_->consume(tile,valid,prefill_tile_);
            }
            auto attended=prefill_->finish();
            if (attended.size()!=batch.size()) throw std::runtime_error("tiled prefill output shape");
            for (const auto value:attended) if (!std::isfinite(value)) throw std::runtime_error("non-finite tiled prefill output");
            std::copy(attended.begin(),attended.end(),output.begin()+start*qwidth);
            stats_.attention_queries+=count;
            stats_.workspace_peak_bytes=std::max(stats_.workspace_peak_bytes,static_cast<std::uint64_t>(
                (output.capacity()+batch.capacity()+query.capacity()+attended.capacity())*sizeof(float)+planes.capacity()*sizeof(Half)));
            // Inclusive host interval covers storage reads and the device frame.
            stats_.tiled_prefill_ns+=elapsed(attention_start);
            start+=count;
        }
    } catch (...) {
        failed_=true;
        const auto error=std::current_exception();
        try { prefill_->abort(); } catch (...) {}
        std::rethrow_exception(error);
    }
    return output;
}
void AttentionSession::append_token(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                    const float * keys, const float * values, bool query_observed) {
    const auto elements = static_cast<std::size_t>(geometry().heads) * geometry().dimension;
    const auto start = Clock::now();
    for (const auto value : query) if (!std::isfinite(value)) throw std::invalid_argument("non-finite query");
    std::vector<Half> kv(elements * 2);
    pack_token(geometry(), keys, values, kv.data());
    if (prefetch_) prefetch_->wait_store(layer);
    store_->append_interleaved(layer, position, kv);
    if (representatives_) {
        std::vector<Half> k(elements);
        copy_keys(geometry(), kv.data(), k);
        if (query_observed) representatives_->append_keys(layer, position, k);
        else representatives_->observe(layer, position, query, k);
    }
    stats_.append_ns += elapsed(start);
}
std::optional<DeviceMemory> AttentionSession::prefill_direct_memory() const {
    if (!prefill_) return {};
    const auto memory=prefill_->direct_memory();
    if (rope_) return memory == DeviceMemory::cuda && representatives_->rank_memory() == memory ? memory : std::nullopt;
    if (representatives_ && representatives_->direct_memory()!=memory) return {};
    return memory;
}
void AttentionSession::retain_prefill_graph_output(std::uint64_t bytes) {
    if (bytes>UINT64_MAX-stats_.prefill_graph_device_bytes) throw std::overflow_error("prefill graph output accounting");
    stats_.prefill_graph_device_bytes+=bytes;
    stats_.prefill_graph_device_peak_bytes=std::max(stats_.prefill_graph_device_peak_bytes,stats_.prefill_graph_device_bytes);
}
void AttentionSession::release_prefill_graph_output(std::uint64_t bytes) noexcept {
    if (bytes>stats_.prefill_graph_device_bytes) std::terminate();
    stats_.prefill_graph_device_bytes-=bytes;
}
void AttentionSession::process_prefill_device(std::uint32_t layer,std::uint64_t first,std::uint32_t tokens,DeviceQuery queries,
    DeviceKvBatch device_kv,const std::vector<Half> & host_kv,float scale,DeviceOutput output) {
    trace::Scope phase("GPU Attention", layer, first);
    const auto memory=prefill_direct_memory();
    if (failed_ || early_ || !memory || !tokens || tokens>prefill_tokens_ || first>prefill_tokens_-tokens ||
        store_->length(layer)!=first || !std::isfinite(scale) || scale<=0) throw std::invalid_argument("device prefill frame state or position");
    const auto & g=geometry();
    validate_device_query(queries,*memory,query_heads_,g.dimension,tokens);
    validate_device_kv(device_kv,*memory,g.heads,g.dimension,tokens);
    validate_device_output(output,*memory,query_heads_,g.dimension,tokens);
    if (host_kv.size()!=device_kv.elements) throw std::invalid_argument("device prefill offload shape");
    if (device_ranges_overlap(output.data,output.elements,sizeof(float),queries.data,queries.elements,sizeof(float)) ||
        device_ranges_overlap(output.data,output.elements,sizeof(float),device_kv.data,device_kv.elements,sizeof(Half))) {
        throw std::invalid_argument("device prefill output overlaps borrowed inputs");
    }
    if (rope_) {
        (void) infllm_prefill(layer, first, tokens, {}, {}, {}, scale, true, queries, output, device_kv, &host_kv);
        return;
    }
    const auto qwidth=device_elements(1,query_heads_,g.dimension),kwidth=device_elements(1,g.heads,g.dimension,2);
    std::vector<Half> row(kwidth);
    try {
        for (auto value:host_kv) if (!std::isfinite(from_half(value))) throw std::invalid_argument("non-finite prefill KV offload");
        for (std::uint32_t start=0;start<tokens;) {
            const auto count=std::min(prefill_->query_capacity(),tokens-start);
            const DeviceQuery query{queries.data+start*qwidth,count*qwidth,*memory};
            const DeviceKvBatch kv{device_kv.data+start*kwidth,count*kwidth,*memory};
            const auto append_start=Clock::now();
            if (representatives_) representatives_->observe_batch_device(layer,first+start,count,query,kv);
            for (std::uint32_t i=0;i<count;++i) {
                std::copy_n(host_kv.data()+(start+i)*kwidth,kwidth,row.data());
                store_->append_interleaved(layer,first+start+i,row);
            }
            stats_.append_ns+=elapsed(append_start);
            const auto attention_start=Clock::now();
            prefill_->begin_device(first+start,count,scale,query);
            const auto end=first+start+count;
            std::vector<Half> planes;
            for (std::uint64_t key=0;key<end;) {
                const auto valid=static_cast<std::uint32_t>(std::min<std::uint64_t>(prefill_->tile_capacity(),
                    key<first?first-key:end-key));
                if (key>=first) {
                    prefill_->consume_device(key,valid,{device_kv.data+(key-first)*kwidth,valid*kwidth,*memory});
                } else {
                    for (std::uint32_t h=0;h<g.heads;++h) {
                        auto * target = prefill_tile_.data() + static_cast<std::size_t>(h) * prefill_->tile_capacity() * 2 * g.dimension;
                        store_->read_rows(layer, h, key, valid, target, nullptr, planes);
                    }
                    prefill_->consume(key,valid,prefill_tile_);
                }
                key+=valid;
            }
            prefill_->finish_to({output.data+start*qwidth,count*qwidth,*memory});
            stats_.attention_queries+=count; stats_.direct_prefill_queries+=count;
            stats_.prefill_kv_download_bytes+=count*kwidth*sizeof(Half);
            stats_.workspace_peak_bytes=std::max(stats_.workspace_peak_bytes,static_cast<std::uint64_t>((host_kv.capacity()+row.capacity()+planes.capacity())*sizeof(Half)));
            stats_.tiled_prefill_ns+=elapsed(attention_start);
            start+=count;
        }
    } catch (...) {
        failed_=true; const auto error=std::current_exception();
        try { prefill_->abort(); } catch (...) {}
        std::rethrow_exception(error);
    }
}
std::uint32_t AttentionSession::gather_slots() const {
    const auto tokens = sparse_ ? budget_.total : geometry().capacity_tokens;
    const auto padded = (static_cast<std::uint64_t>(tokens) + 31) / 32 * 32;
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::overflow_error("gather length exceeds backend limits");
    return static_cast<std::uint32_t>(padded);
}
std::uint64_t AttentionSession::gather_elements() const {
    return static_cast<std::uint64_t>(gather_slots()) *
        (2ull * geometry().heads * geometry().dimension + query_heads_ + (rope_ ? geometry().heads : 0));
}
void AttentionSession::read_decode_rows(std::uint32_t layer, const std::vector<KvRowRange> & ranges, AsyncReader * reader, std::size_t depth) {
    if (auto * file = dynamic_cast<FileKvStore *>(store_.get()); file && geometry().contiguous_planes) {
        file->read_separate_rows(layer, ranges, reader, depth);
        return;
    }
    std::vector<Half> scratch;
    for (const auto & range : ranges) store_->read_rows(layer, range.head, range.first, range.count, range.keys, range.values, scratch);
}
void AttentionSession::read_dense_history(std::uint32_t layer, std::uint32_t position, Half * destination, AsyncReader * reader) {
    auto * file = dynamic_cast<FileKvStore *>(store_.get());
    if (failed_ || sparse_ || !file || position < prefill_tokens_ || position >= geometry().capacity_tokens) {
        throw std::logic_error("invalid dense offload history request");
    }
    file->read_dense_layer(layer, position, gather_slots(), destination, reader);
}
void AttentionSession::append_dense_decode(std::uint32_t layer, std::uint64_t position, const float * keys, const float * values) {
    if (failed_ || sparse_ || position < prefill_tokens_ || !keys || !values) throw std::logic_error("invalid dense decode archive");
    try {
        const auto start = Clock::now();
        std::vector<Half> kv(static_cast<std::size_t>(geometry().heads) * 2 * geometry().dimension);
        pack_token(geometry(), keys, values, kv.data());
        store_->append_interleaved(layer, position, kv);
        stats_.append_ns += elapsed(start);
        if (trace_.is_open()) {
            std::vector<HeadSelection> selection(geometry().heads);
            for (auto & head : selection) {
                head.positions.resize(position + 1);
                std::iota(head.positions.begin(), head.positions.end(), std::uint64_t{0});
            }
            record(layer, position, selection);
        }
        stats_.selected_tokens += (position + 1) * geometry().heads;
        ++stats_.attention_queries;
    } catch (...) { failed_ = true; throw; }
}
bool AttentionSession::layer_write_ready(std::uint32_t layer) const {
    const auto * file = dynamic_cast<const FileKvStore *>(store_.get());
    return file && file->write_buffer_full(layer);
}
std::vector<Half> AttentionSession::gather(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                          const std::vector<float> & keys, const std::vector<float> & values, bool interleaved_inputs) {
    std::vector<Half> packed(static_cast<std::size_t>(gather_elements()));
    gather_into(layer,position,query,keys,values,packed.data(),packed.size(),interleaved_inputs);
    stats_.workspace_peak_bytes=std::max<std::uint64_t>(stats_.workspace_peak_bytes,
        packed.capacity()*sizeof(Half)+static_cast<std::size_t>(geometry().block_tokens)*2*geometry().dimension*sizeof(Half));
    return packed;
}
void AttentionSession::gather_into(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                   const std::vector<float> & keys, const std::vector<float> & values,
                                   Half * packed, std::size_t elements, bool interleaved_inputs) {
    if (failed_) throw std::runtime_error("attention session failed");
    const auto & g = geometry();
    if (position < prefill_tokens_ || query.size() != static_cast<std::size_t>(query_heads_) * g.dimension ||
        keys.size() != static_cast<std::size_t>(g.heads) * g.dimension * (interleaved_inputs ? 2 : 1) ||
        (interleaved_inputs ? !values.empty() : keys.size() != values.size())) {
        throw std::invalid_argument("gather requires one decode query");
    }
    if (!packed || elements!=gather_elements()) throw std::invalid_argument("gather output shape");
    try {
        append_token(layer, position, query, keys.data(), interleaved_inputs ? nullptr : values.data());
        std::vector<HeadSelection> selection;
        const auto selection_start = Clock::now();
        if (sparse_) selection = select(layer, position, query);
        else if (trace_.is_open()) {
            selection.resize(g.heads);
            for (auto & head : selection) {
                head.positions.resize(static_cast<std::size_t>(position + 1));
                std::iota(head.positions.begin(), head.positions.end(), std::uint64_t{0});
            }
        }
        stats_.selection_ns += elapsed(selection_start);
        record(layer, position, selection);
        const auto gather_start = Clock::now();
        const auto slots = gather_slots();
        const auto head_elements=static_cast<std::size_t>(slots)*g.dimension;
        const auto plane_elements=head_elements*g.heads;
        auto * mask=packed+2*plane_elements;
        auto * markers = rope_ ? mask + static_cast<std::size_t>(slots) * query_heads_ : nullptr;
        std::vector<Half> range_scratch;
        const auto group = query_heads_ / g.heads;
        for (std::uint32_t head = 0; head < g.heads; ++head) {
            const auto * positions=sparse_?&selection[head].positions:nullptr;
            const auto count=positions?positions->size():static_cast<std::size_t>(position+1);
            if (!count || count>slots) throw std::logic_error("invalid gathered selection size");
            if (markers) {
                auto * row = markers + static_cast<std::size_t>(head) * slots;
                std::fill_n(row, slots, Half{0});
                for (std::size_t i = 0; i < count; ++i) {
                    const auto distance = position - (*positions)[i];
                    if (distance < budget_.local) row[i] = to_half(static_cast<float>(budget_.local - distance));
                }
            }
            for (std::size_t i=0;i<count;) {
                const auto token=positions?(*positions)[i]:i;
                std::uint32_t run = static_cast<std::uint32_t>(count - i);
                if (positions) {
                    run=1;
                    while (i + run < count && (*positions)[i+run]==token+run) ++run;
                }
                const auto offset=head*head_elements+i*g.dimension;
                store_->read_rows(layer, head, token, run,
                                  packed + (g.layout == KvLayout::interleaved ? 2 * offset : offset),
                                  g.layout == KvLayout::interleaved ? nullptr : packed + plane_elements + offset, range_scratch);
                i+=run;
            }
            const auto tail=(slots-count)*g.dimension;
            const auto tail_offset=head*head_elements+count*g.dimension;
            if (g.layout==KvLayout::interleaved) std::fill_n(packed+2*tail_offset,2*tail,Half{0});
            else {
                std::fill_n(packed+tail_offset,tail,Half{0});
                std::fill_n(packed+plane_elements+tail_offset,tail,Half{0});
            }
            for (std::uint32_t q = head * group; q < (head + 1) * group; ++q) {
                auto * row=mask+static_cast<std::size_t>(q)*slots;
                std::fill_n(row,count,Half{0});
                std::fill_n(row+count,slots-count,Half{0xfc00});
            }
            stats_.gather_copy_bytes+=2ull*count*g.dimension*sizeof(Half);
            stats_.selected_tokens+=count;
        }
        stats_.gather_ns += elapsed(gather_start);
        ++stats_.attention_queries;
        stats_.workspace_peak_bytes = std::max<std::uint64_t>(stats_.workspace_peak_bytes,
            range_scratch.capacity()*sizeof(Half));
    } catch (...) { failed_ = true; throw; }
}
Prefetcher & AttentionSession::prefetcher() {
    if (!prefetch_) throw std::logic_error("prefetch is disabled");
    return *prefetch_;
}
void AttentionSession::begin_prefetch(std::uint32_t layer, std::uint64_t position, PredictionCosts costs) {
    if (failed_) throw std::runtime_error("attention session failed");
    if (prefetch_ && position >= prefill_tokens_ && position < geometry().capacity_tokens) {
        try { prefetch_->begin(layer, position, costs); } catch (...) { failed_ = true; throw; }
    }
}
std::vector<HeadSelection> AttentionSession::select_decode(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                                          const std::vector<float> & keys, const std::vector<float> & values,
                                                          bool interleaved_inputs, const std::function<void()> & after_append) {
    if (failed_) throw std::runtime_error("attention session failed");
    if (!sparse_ || position < prefill_tokens_ || query.size() != static_cast<std::size_t>(query_heads_) * geometry().dimension ||
        keys.size() != static_cast<std::size_t>(geometry().heads) * geometry().dimension * (interleaved_inputs ? 2 : 1) ||
        (interleaved_inputs ? !values.empty() : values.size() != keys.size())) throw std::invalid_argument("invalid selected decode input");
    try {
        append_token(layer, position, query, keys.data(), interleaved_inputs ? nullptr : values.data());
        if (after_append) after_append();
        const auto start = Clock::now();
        auto selected = select(layer, position, query);
        stats_.selection_ns += elapsed(start);
        record(layer, position, selected);
        for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
        ++stats_.attention_queries;
        return selected;
    } catch (...) { failed_ = true; throw; }
}
std::vector<HeadSelection> AttentionSession::select_decode_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query,
    const std::vector<float> & keys, const std::vector<float> & values, const std::function<void()> & after_append) {
    const auto memory = direct_memory();
    const auto width = static_cast<std::size_t>(geometry().heads) * geometry().dimension;
    if (failed_ || early_ || !rope_ || !memory || !sparse_ || prefetch_ || position < prefill_tokens_ ||
        store_->length(layer) != position || keys.size() != width || values.size() != width) {
        throw std::invalid_argument("invalid selected device decode input");
    }
    validate_device_query(query, *memory, query_heads_, geometry().dimension);
    try {
        representatives_->begin_probability_query(layer, position);
        append_token(layer, position, {}, keys.data(), values.data(), true);
        if (after_append) after_append();
        const auto start = Clock::now();
        auto selected = select_blocks(geometry(), budget_, *representatives_, layer, position, query);
        stats_.selection_ns += elapsed(start);
        record(layer, position, selected);
        for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
        ++stats_.attention_queries;
        return selected;
    } catch (...) { failed_ = true; throw; }
}
std::vector<HeadSelection> AttentionSession::select_before_kv(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) {
    if (failed_ || early_ || pending_device_selection_ || !prefetch_ || position < prefill_tokens_ || store_->length(layer) != position) {
        throw std::logic_error("early selection requires an uncommitted decode position");
    }
    try {
        const auto start = Clock::now();
        representatives_->observe_query(layer, position, query);
        auto selected = select(layer, position, query);
        stats_.selection_ns += elapsed(start);
        early_ = std::make_pair(layer, position);
        direct_early_ = false;
        record(layer, position, selected);
        for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
        return selected;
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::append_after_selection(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query,
                                             const std::vector<float> & keys, const std::vector<float> & values, bool packed) {
    if (failed_ || !early_ || direct_early_ || *early_ != std::make_pair(layer, position) ||
        query.size() != static_cast<std::size_t>(query_heads_) * geometry().dimension ||
        keys.size() != static_cast<std::size_t>(geometry().heads) * geometry().dimension * (packed ? 2 : 1) ||
        (packed ? !values.empty() : values.size() != keys.size())) throw std::invalid_argument("late KV does not match early selection");
    try {
        append_token(layer, position, query, keys.data(), packed ? nullptr : values.data(), true);
        early_.reset();
        ++stats_.attention_queries;
    } catch (...) { failed_ = true; throw; }
}
std::optional<DeviceMemory> AttentionSession::direct_memory() const {
    if (!representatives_ || replay_.is_open()) return {};
    return rope_ ? representatives_->rank_memory() : representatives_->direct_memory();
}
std::vector<HeadSelection> AttentionSession::select_before_kv_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query) {
    submit_before_kv_device(layer, position, query);
    return finish_before_kv_device();
}
void AttentionSession::submit_before_kv_device(std::uint32_t layer, std::uint64_t position, DeviceQuery query) {
    if (failed_ || early_ || pending_device_selection_ || !prefetch_ || !direct_memory() || position < prefill_tokens_ || store_->length(layer) != position) {
        throw std::logic_error("device early selection requires an uncommitted decode position");
    }
    try {
        const auto start = Clock::now();
        if (rope_) representatives_->begin_probability_query(layer, position);
        else representatives_->observe_query_device(layer, position, query,true);
        auto pending = submit_select_blocks(geometry(), budget_, *representatives_, layer, position, query);
        pending_device_selection_ = [this, layer, position, start, pending = std::move(pending)]() mutable {
            auto selected = pending();
            if (!rope_) representatives_->finish_query_for_rank_device();
            stats_.selection_ns += elapsed(start);
            early_ = std::make_pair(layer, position);
            direct_early_ = true;
            record(layer, position, selected);
            for (const auto & head : selected) stats_.selected_tokens += head.positions.size();
            return selected;
        };
    } catch (...) {
        failed_ = true;
        if (!rope_) try { representatives_->finish_query_for_rank_device(); } catch (...) {}
        throw;
    }
}
std::vector<HeadSelection> AttentionSession::finish_before_kv_device() {
    if (failed_ || !pending_device_selection_) throw std::logic_error("no pending device selection");
    auto pending = std::move(pending_device_selection_);
    pending_device_selection_ = {};
    try { return pending(); }
    catch (...) {
        failed_ = true;
        if (!rope_) try { representatives_->finish_query_for_rank_device(); } catch (...) {}
        throw;
    }
}
void AttentionSession::append_after_selection_half(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv, DeviceKeys keys) {
    stage_device_keys(layer, position, keys);
    archive_device_kv(layer, position, kv);
}
void AttentionSession::stage_device_keys(std::uint32_t layer, std::uint64_t position, DeviceKeys keys,void * compute_stream) {
    if (failed_ || !early_ || !direct_early_ || device_keys_staged_ || *early_ != std::make_pair(layer, position)) {
        throw std::invalid_argument("device late KV does not match early selection");
    }
    try {
        const auto start = Clock::now();
        if (rope_) validate_device_keys(keys, *direct_memory(), geometry().heads, geometry().dimension);
        else representatives_->append_keys_device(layer, position, keys,compute_stream);
        stats_.append_ns += elapsed(start);
        device_keys_staged_ = true;
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::archive_device_kv(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & kv) {
    if (failed_ || !early_ || !device_keys_staged_ || *early_ != std::make_pair(layer, position) || store_->length(layer)!=position ||
        kv.size() != static_cast<std::size_t>(geometry().heads)*geometry().dimension*2) {
        throw std::invalid_argument("device archive does not match the outstanding KV position");
    }
    try {
        for (const auto value : kv) if (!std::isfinite(from_half(value))) throw std::invalid_argument("non-finite device KV offload");
        const auto start = Clock::now();
        prefetch_->wait_store(layer);
        store_->append_interleaved(layer, position, kv);
        if (rope_) {
            std::vector<Half> keys(kv.size() / 2);
            copy_keys(geometry(), kv.data(), keys);
            representatives_->append_keys(layer, position, keys);
        }
        stats_.append_ns += elapsed(start);
        stats_.decode_kv_download_bytes += kv.size()*sizeof(Half);
        ++stats_.direct_decode_queries;
        ++stats_.attention_queries;
        early_.reset(); direct_early_ = false; device_keys_staged_ = false;
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::record(std::uint32_t layer, std::uint64_t position, const std::vector<HeadSelection> & selection) {
    if (!trace_.is_open()) return;
    for (std::size_t head = 0; head < selection.size(); ++head) {
        trace_ << "{\"layer\":" << layer << ",\"position\":" << position << ",\"kv_head\":" << head << ",\"ranges\":[";
        const auto & positions = selection[head].positions;
        for (std::size_t i = 0; i < positions.size();) {
            const auto first = i++;
            while (i < positions.size() && positions[i] == positions[i - 1] + 1) ++i;
            if (first > 0) trace_ << ',';
            trace_ << '[' << positions[first] << ',' << i - first << ']';
        }
        trace_ << "]}\n";
    }
    if (!trace_) throw std::runtime_error("selection trace write failed");
}
std::vector<HeadSelection> AttentionSession::select(std::uint32_t layer, std::uint64_t position, const std::vector<float> & query) {
    if (!replay_.is_open()) return select_blocks(geometry(), budget_, *representatives_, layer, position, query);
    std::vector<HeadSelection> selection(geometry().heads);
    for (std::uint32_t head = 0; head < geometry().heads; ++head) {
        std::string line;
        // Each range consumes at least one token. Bound input before allocation.
        const auto limit = static_cast<std::uint64_t>(budget_.total) * 48 + 256;
        for (char ch; replay_.get(ch) && ch != '\n';) {
            if (line.size() >= limit) throw std::invalid_argument("selection replay line exceeds budget");
            line.push_back(ch);
        }
        if (line.empty()) throw std::invalid_argument("selection replay ended before requested query");
        const auto item = nlohmann::json::parse(line);
        const auto integer = [](const nlohmann::json & value) -> std::uint64_t {
            if (!value.is_number_integer() || value.get<std::int64_t>() < 0) throw std::invalid_argument("invalid replay integer");
            return value.get<std::uint64_t>();
        };
        if (!item.is_object() || item.size() != 4 || integer(item.at("layer")) != layer ||
            integer(item.at("position")) != position || integer(item.at("kv_head")) != head || !item.at("ranges").is_array()) {
            throw std::invalid_argument("selection replay query identity mismatch");
        }
        auto & result = selection[head];
        for (const auto & range : item.at("ranges")) {
            if (!range.is_array() || range.size() != 2) throw std::invalid_argument("invalid replay range");
            const auto first = integer(range[0]), count = integer(range[1]);
            if (count == 0 || first > position || count > position - first + 1 ||
                count > budget_.total - result.positions.size() || (!result.positions.empty() && first <= result.positions.back())) {
                throw std::invalid_argument("replay range exceeds history or token budget");
            }
            for (std::uint64_t i = 0; i < count; ++i) {
                const auto p = first + i;
                result.positions.push_back(p);
                const auto block = p / geometry().block_tokens;
                if (result.blocks.empty() || result.blocks.back() != block) result.blocks.push_back(block);
            }
        }
        if (result.positions.empty()) throw std::invalid_argument("empty selection replay");
        const auto contains = [&](std::uint64_t p) { return std::binary_search(result.positions.begin(), result.positions.end(), p); };
        for (std::uint64_t p = 0; p < std::min<std::uint64_t>(budget_.initial, position + 1); ++p) {
            if (!contains(p)) throw std::invalid_argument("replay omits initial tokens");
        }
        for (auto p = position + 1 - std::min<std::uint64_t>(budget_.local, position + 1); p <= position; ++p) {
            if (!contains(p)) throw std::invalid_argument("replay omits local tokens");
        }
    }
    return selection;
}
void AttentionSession::enable_direct_reads() {
    store_->enable_direct_reads();
}
void AttentionSession::flush() {
    if (failed_) throw std::runtime_error("attention session failed");
    if (pending_device_selection_) throw std::logic_error("collect device selection before flushing the session");
    try {
        finish_device_probabilities();
        if (prefetch_) prefetch_->finish_stores();
        store_->flush();
        if (trace_.is_open()) { trace_.flush(); if (!trace_) throw std::runtime_error("selection trace flush failed"); }
    } catch (...) { failed_ = true; throw; }
}
void AttentionSession::finish() {
    if (prefetch_) prefetch_->finish();
    flush();
    if (replay_.is_open()) {
        replay_ >> std::ws;
        if (replay_.peek() != std::char_traits<char>::eof()) {
            failed_ = true;
            throw std::invalid_argument("selection replay has unused queries");
        }
    }
}
SessionStats AttentionSession::stats() const {
    auto result = stats_;
    result.kv_resident_bytes = store_->resident_bytes();
    result.kv_write_buffer_bytes = store_->write_buffer_bytes();
    result.representative_bytes = representatives_ ? representatives_->resident_bytes() : 0;
    if (representatives_) {
        result.ranking = representatives_->ranker_stats();
        result.selection_backend = replay_.is_open() ? "replay" : representatives_->ranker_name();
        result.representative_backend = representatives_->builder_name();
    }
    result.io = store_->io_stats();
    if (prefetch_) {
        result.prefetch = prefetch_->stats();
        result.io.write_requests += result.prefetch.io.write_requests;
        result.io.write_bytes += result.prefetch.io.write_bytes;
        result.io.write_ns += result.prefetch.io.write_ns;
    }
    if (prefill_) result.prefill=prefill_->stats();
    result.prefill_host_bytes=prefill_tile_.capacity()*sizeof(Half);
    return result;
}
void AttentionSession::set_ranker(std::unique_ptr<BlockRanker> ranker) {
    if (failed_ || !representatives_ || replay_.is_open() || early_ || pending_device_selection_) throw std::logic_error("ranker requires a dynamic sparse session at a query boundary");
    try { representatives_->set_ranker(std::move(ranker)); }
    catch (...) { failed_ = true; throw; }
}
void AttentionSession::set_prefill(std::unique_ptr<TiledPrefill> prefill) {
    const auto * original = rope_ ? "cpu_infllm_sparse" : "cpu_streaming";
    if (failed_ || early_ || pending_device_selection_ || prefill_ || stats_.prefill_backend != original || !prefill ||
        prefill->supports_sparse() != rope_.has_value() || !prefill->query_capacity() ||
        !prefill->tile_capacity() || prefill->tile_capacity()%geometry().block_tokens) throw std::invalid_argument("invalid tiled prefill attachment");
    const auto width=static_cast<std::uint64_t>(2)*geometry().dimension;
    if (prefill->tile_capacity()>SIZE_MAX/sizeof(Half)/width/geometry().heads) throw std::overflow_error("prefill host tile allocation");
    const auto elements=static_cast<std::uint64_t>(geometry().heads)*prefill->tile_capacity()*width;
    std::vector<Half> tile(static_cast<std::size_t>(elements));
    stats_.prefill_backend=prefill->name();
    stats_.prefill_host_peak_bytes=tile.capacity()*sizeof(Half);
    prefill_tile_=std::move(tile); prefill_=std::move(prefill);
}
void AttentionSession::release_prefill_if_complete() {
    if (failed_ || !prefill_) return;
    for (std::uint32_t layer=0;layer<geometry().layers;++layer) if (store_->length(layer)<prefill_tokens_) return;
    stats_.prefill=prefill_->stats();
    stats_.prefill.kv_device_bytes=0; stats_.prefill.workspace_device_bytes=0;
    prefill_.reset();
    std::vector<Half>().swap(prefill_tile_);
}
} // namespace solid
