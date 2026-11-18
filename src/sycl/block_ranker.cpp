#include "solid/block_ranker.hpp"
#include "solid/representative_math.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <type_traits>

namespace solid {
namespace {

// A normalized two-float expansion for devices without fp64. The explicit
// FMAs recover the rounding error of products and quotients.
struct FloatPair {
    float high = 0;
    float low = 0;

    static FloatPair sum(float a, float b) {
        const float s = a + b;
        const float v = s - a;
        return {s, (a - (s - v)) + (b - v)};
    }
    friend FloatPair operator+(FloatPair a, FloatPair b) {
        const auto s = sum(a.high, b.high);
        return sum(s.high, s.low + (a.low + b.low));
    }
    friend FloatPair operator*(FloatPair a, float b) {
        const float p = a.high * b;
        return sum(p, sycl::fma(a.high, b, -p) + a.low * b);
    }
    friend FloatPair operator/(FloatPair a, std::uint32_t b) {
        const float divisor = static_cast<float>(b);
        const float q = a.high / divisor;
        return sum(q, (sycl::fma(-q, divisor, a.high) + a.low) / divisor);
    }
    friend bool operator>(FloatPair a, FloatPair b) {
        return a.high > b.high || (a.high == b.high && a.low > b.low);
    }
    friend bool operator==(FloatPair a, FloatPair b) { return a.high == b.high && a.low == b.low; }
};

float decode_half(Half bits) { return static_cast<float>(sycl::bit_cast<sycl::half>(bits)); }

template<class Score> Score device_score(float value) {
    if constexpr (std::is_same_v<Score, FloatPair>) return {value, 0};
    else return value;
}
template<class Score> bool finite(Score score) {
    if constexpr (std::is_same_v<Score, FloatPair>) return sycl::isfinite(score.high) && sycl::isfinite(score.low);
    else return sycl::isfinite(score);
}
template<class Score> Score seed_score(double value) {
    if constexpr (std::is_same_v<Score, FloatPair>) {
        const float high = static_cast<float>(value);
        if (!std::isfinite(high)) throw std::invalid_argument("representative seed exceeds SYCL score range");
        return {high, static_cast<float>(value - high)};
    } else return value;
}
template<class Score> double host_score(Score value) {
    if constexpr (std::is_same_v<Score, FloatPair>) return static_cast<double>(value.high) + value.low;
    else return value;
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("SYCL ranker buffer size");
    return a * b;
}

struct UsmDelete {
    sycl::queue * queue = nullptr;
    void operator()(void * pointer) const { sycl::free(pointer, *queue); }
};
template<class T> using Buffer = std::unique_ptr<T, UsmDelete>;

void flag_invalid(int * status) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*status).store(1);
}

template<class Score> class SyclBlockRanker final : public BlockRanker {
    struct Layer {
        Buffer<Half> bank, pending_keys, top_keys;
        Buffer<Score> pending_scores, top_scores;
        Buffer<std::uint64_t> top_positions;
        std::uint64_t published = 0, length = 0;
        bool initialized = false, query_pending = false;
    };
public:
    SyclBlockRanker(KvGeometry geometry, std::uint32_t query_heads, std::uint32_t representatives,
                    std::uint32_t budget, const sycl::queue & backend_queue, std::uint32_t local) :
        queue_(backend_queue.get_context(), backend_queue.get_device(),
               [](sycl::exception_list errors) { for (const auto & error : errors) std::rethrow_exception(error); },
               sycl::property::queue::in_order{}),
        geometry_(geometry), query_heads_(query_heads), representatives_(representatives), budget_(budget), local_(local) {
        geometry_.validate();
        if (!query_heads || query_heads % geometry_.heads || !representatives || representatives > geometry_.block_tokens ||
            !budget || budget > geometry_.capacity_tokens || local > budget) {
            throw std::invalid_argument("invalid SYCL ranker geometry");
        }
        const auto device = queue_.get_device();
        if (!device.has(sycl::aspect::fp16) || !device.has(sycl::aspect::usm_device_allocations) ||
            device.get_info<sycl::info::device::max_work_group_size>() < work_group_size) {
            throw std::invalid_argument("SYCL selection requires FP16, device USM and 64 work items per group");
        }
        const auto blocks = geometry_.blocks_per_head();
        const auto max_result = std::min<std::uint64_t>(budget, blocks);
        keys_per_token_ = product(geometry_.heads, geometry_.dimension);
        block_elements_ = product(keys_per_token_, representatives);
        bank_elements_ = product(blocks, block_elements_);
        layers_.resize(geometry_.layers);
        query_ = allocate<float>(product(query_heads, geometry_.dimension));
        ids_ = allocate<std::uint64_t>(blocks);
        costs_ = allocate<std::uint32_t>(blocks);
        scores_ = allocate<Score>(product(geometry_.heads, blocks));
        output_host_.resize(product(geometry_.heads, max_result));
        output_ = allocate<std::uint64_t>(output_host_.size());
        error_ = allocate<int>(1);
        stats_.host_bytes = layers_.capacity() * sizeof(Layer) + output_host_.capacity() * sizeof(std::uint64_t);
        if (local_) {
            mean_ = allocate<Score>(keys_per_token_, true);
            const auto slots = product(geometry_.heads, representatives_);
            built_positions_ = allocate<std::uint64_t>(slots, true);
            built_scores_ = allocate<Score>(slots, true);
        }
    }

    void publish(std::uint32_t layer, std::uint64_t first, std::uint64_t count,
                 const std::vector<Half> & keys) override {
        auto & state = checked_layer(layer);
        if (state.initialized || first != state.published || !count || count > geometry_.blocks_per_head() - first ||
            keys.size() != product(count, block_elements_)) {
            throw std::invalid_argument("SYCL representative publication must append complete immutable blocks");
        }
        validate_keys(keys);
        ensure_bank(state);
        execute([&] { queue_.memcpy(state.bank.get() + first * block_elements_, keys.data(), keys.size() * sizeof(Half)); });
        state.published += count;
        stats_.published_blocks += count;
    }

    std::vector<std::vector<std::uint64_t>> rank(std::uint32_t layer, const std::vector<float> & query,
                                                 const BlockCandidates & candidates) override {
        validate_query(query);
        return rank_impl(layer, query.data(), candidates, false);
    }

    std::vector<std::vector<std::uint64_t>> rank_device(std::uint32_t layer, DeviceQuery query,
                                                        const BlockCandidates & candidates) override {
        validate_device_query(query, DeviceMemory::sycl, query_heads_, geometry_.dimension);
        validate_pointer(query.data);
        auto result = rank_impl(layer, query.data, candidates, true);
        ++stats_.device_rank_queries;
        return result;
    }

private:
    std::vector<std::vector<std::uint64_t>> rank_impl(std::uint32_t layer, const float * input,
                                                      const BlockCandidates & candidates, bool direct) {
        const auto & state = checked_layer(layer);
        const auto count = candidates.blocks.size();
        const auto remaining = candidates.remaining;
        if (count != candidates.tokens.size() || count > geometry_.blocks_per_head() || remaining > budget_) {
            throw std::invalid_argument("SYCL ranking shape or budget");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (candidates.blocks[i] >= state.published || (i && candidates.blocks[i] <= candidates.blocks[i - 1]) ||
                !candidates.tokens[i] || candidates.tokens[i] > geometry_.block_tokens) {
                throw std::invalid_argument("invalid SYCL rank candidate");
            }
        }
        const auto heads = geometry_.heads;
        std::vector<std::vector<std::uint64_t>> result(heads);
        if (!remaining || !count) {
            if (direct) {
                int error = 0;
                execute([&] {
                    queue_.fill(error_.get(), 0, 1);
                    check_device_query(input);
                    queue_.memcpy(&error, error_.get(), sizeof(error));
                }, &error);
            }
            return result;
        }

        const auto stride = std::min<std::size_t>(remaining, count);
        const auto result_count = product(heads, stride);
        const auto query_bytes = product(product(query_heads_, geometry_.dimension), sizeof(float));
        int error = 0;
        execute([&] {
            if (!direct) queue_.memcpy(query_.get(), input, query_bytes);
            queue_.memcpy(ids_.get(), candidates.blocks.data(), count * sizeof(std::uint64_t));
            queue_.memcpy(costs_.get(), candidates.tokens.data(), count * sizeof(std::uint32_t));
            queue_.fill(error_.get(), 0, 1);
            queue_.fill(output_.get(), std::uint64_t{UINT64_MAX}, result_count);
            score_candidates(state.bank.get(), direct ? input : query_.get(), count);
            select_candidates(count, remaining, stride);
            queue_.memcpy(output_host_.data(), output_.get(), result_count * sizeof(std::uint64_t));
            queue_.memcpy(&error, error_.get(), sizeof(error));
        }, &error);
        for (std::uint32_t h = 0; h < heads; ++h) {
            const auto start = output_host_.data() + h * stride;
            result[h].assign(start, std::find(start, start + stride, UINT64_MAX));
        }
        ++stats_.calls;
        stats_.scored_blocks += count * heads;
        if (!direct) stats_.query_upload_bytes += query_bytes;
        return result;
    }

public:
    std::uint32_t builder_window() const override { return local_; }
    std::optional<DeviceMemory> direct_memory() const override { return DeviceMemory::sycl; }
    const char * builder_name() const override {
        return !local_ ? "cpu" : std::is_same_v<Score, double> ? "sycl_window_fp64" : "sycl_window_float_pair";
    }
    const char * name() const override { return std::is_same_v<Score, double> ? "sycl_fp64_topk" : "sycl_float_pair_topk"; }
    RankerStats stats() const override { return stats_; }

    void seed_builder(std::uint32_t layer, const RepresentativeSeed & seed) override {
        auto & state = checked_layer(layer);
        if (!local_ || state.initialized) throw std::invalid_argument("SYCL builder initialization state");
        validate_representative_seed(geometry_, representatives_, local_, seed);
        const auto pending = std::min<std::uint64_t>(seed.length, local_);
        const auto first = seed.length - pending;
        if (state.published != first / geometry_.block_tokens) throw std::invalid_argument("SYCL builder published prefix differs from seed");

        const auto heads = geometry_.heads;
        const auto width = geometry_.dimension;
        const auto slots = product(heads, representatives_);
        std::vector<Score> pending_scores;
        pending_scores.reserve(seed.pending_scores.size());
        for (const auto score : seed.pending_scores) pending_scores.push_back(seed_score<Score>(score));
        std::vector<Score> top_scores(slots);
        std::vector<std::uint64_t> top_positions(slots, UINT64_MAX);
        std::vector<Half> top_keys(block_elements_);
        for (std::uint32_t h = 0; h < heads; ++h) {
            const auto base = static_cast<std::size_t>(h) * representatives_;
            for (std::uint32_t r = 0; r < representatives_; ++r) {
                const auto i = base + r;
                if (seed.partial_positions[i] == UINT64_MAX) break;
                insert_representative(seed.partial_positions[i], seed_score<Score>(seed.partial_scores[i]),
                    seed.partial_keys.data() + i * width, representatives_, width,
                    top_positions.data() + base, top_scores.data() + base, top_keys.data() + base * width);
            }
        }

        ensure_bank(state);
        execute([&] {
            state.pending_keys = allocate<Half>(product(local_, keys_per_token_), true);
            state.pending_scores = allocate<Score>(product(local_, heads), true);
            state.top_keys = allocate<Half>(block_elements_, true);
            state.top_positions = allocate<std::uint64_t>(slots, true);
            state.top_scores = allocate<Score>(slots, true);
            const auto copy_ring = [&](std::uint64_t dest, std::uint64_t src, std::uint64_t rows) {
                if (!rows) return;
                queue_.memcpy(state.pending_keys.get() + dest * keys_per_token_, seed.pending_keys.data() + src * keys_per_token_,
                              rows * keys_per_token_ * sizeof(Half));
                queue_.memcpy(state.pending_scores.get() + dest * heads, pending_scores.data() + src * heads,
                              rows * heads * sizeof(Score));
            };
            const auto ring_start = first % local_;
            const auto before_wrap = std::min<std::uint64_t>(pending, local_ - ring_start);
            copy_ring(ring_start, 0, before_wrap);
            copy_ring(0, before_wrap, pending - before_wrap);
            queue_.memcpy(state.top_keys.get(), top_keys.data(), block_elements_ * sizeof(Half));
            queue_.memcpy(state.top_positions.get(), top_positions.data(), slots * sizeof(std::uint64_t));
            queue_.memcpy(state.top_scores.get(), top_scores.data(), slots * sizeof(Score));
        });
        state.length = seed.length;
        state.initialized = true;
    }

    std::optional<BuiltRepresentatives> build_query(std::uint32_t layer, std::uint64_t position,
                                                    const std::vector<float> & query) override {
        validate_query(query);
        return build_query_impl(layer, position, query.data(), false);
    }

    std::optional<BuiltRepresentatives> build_query_device(std::uint32_t layer, std::uint64_t position,
                                                           DeviceQuery query) override {
        validate_device_query(query, DeviceMemory::sycl, query_heads_, geometry_.dimension);
        validate_pointer(query.data);
        auto result = build_query_impl(layer, position, query.data, true);
        ++stats_.device_builder_queries;
        return result;
    }

private:
    std::optional<BuiltRepresentatives> build_query_impl(std::uint32_t layer, std::uint64_t position,
                                                         const float * input, bool direct) {
        auto & state = builder_layer(layer, position, false);
        const auto pending = std::min<std::uint64_t>(position, local_);
        const bool finalize = position >= local_;
        const bool complete = finalize && (position - local_ + 1) % geometry_.block_tokens == 0;
        std::optional<BuiltRepresentatives> built;
        std::vector<Score> scores;
        if (complete) {
            built.emplace();
            built->block = (position - local_) / geometry_.block_tokens;
            if (built->block != state.published) throw std::logic_error("SYCL representative publication boundary");
            const auto slots = product(geometry_.heads, representatives_);
            built->keys.resize(block_elements_);
            built->positions.resize(slots);
            built->scores.resize(slots);
            scores.resize(slots);
        }
        if (pending || direct) {
            int error = 0;
            const auto bytes = product(product(query_heads_, geometry_.dimension), sizeof(float));
            execute([&] {
                if (!direct) queue_.memcpy(query_.get(), input, bytes);
                queue_.fill(error_.get(), 0, 1);
                if (pending) advance_window(state, direct ? input : query_.get(), position - pending, pending);
                else check_device_query(input);
                if (finalize) finalize_row(state, position - local_, complete);
                if (built) {
                    queue_.memcpy(built->keys.data(), state.bank.get() + built->block * block_elements_, block_elements_ * sizeof(Half));
                    queue_.memcpy(built->positions.data(), built_positions_.get(), scores.size() * sizeof(std::uint64_t));
                    queue_.memcpy(scores.data(), built_scores_.get(), scores.size() * sizeof(Score));
                }
                queue_.memcpy(&error, error_.get(), sizeof(error));
            }, &error);
            if (!direct) stats_.query_upload_bytes += bytes;
        }
        if (built) {
            // Project the device ordering into the host score type. A sub-ULP
            // residual can turn adjacent device scores into a tie here.
            std::transform(scores.begin(), scores.end(), built->scores.begin(), host_score<Score>);
            order_host_ties(*built);
            ++state.published;
            ++stats_.published_blocks;
            ++stats_.built_blocks;
        }
        state.query_pending = true;
        ++stats_.builder_queries;
        return built;
    }

public:
    void build_append(std::uint32_t layer, std::uint64_t position, const std::vector<Half> & keys) override {
        auto & state = builder_layer(layer, position, true);
        if (keys.size() != keys_per_token_) throw std::invalid_argument("SYCL builder key shape");
        validate_keys(keys);
        const auto bytes = keys.size() * sizeof(Half);
        const auto slot = position % local_;
        execute([&] {
            queue_.memcpy(state.pending_keys.get() + slot * keys_per_token_, keys.data(), bytes);
            queue_.fill(state.pending_scores.get() + slot * geometry_.heads, Score{}, geometry_.heads);
        });
        ++state.length;
        state.query_pending = false;
        ++stats_.builder_appends;
        stats_.builder_key_upload_bytes += bytes;
    }

    void build_append_device(std::uint32_t layer, std::uint64_t position, DeviceKeys keys) override {
        auto & state = builder_layer(layer, position, true);
        const auto heads = geometry_.heads;
        const auto width = geometry_.dimension;
        validate_device_keys(keys, DeviceMemory::sycl, heads, width);
        validate_pointer(keys.data);
        const auto slot = position % local_;
        auto * target = state.pending_keys.get() + slot * keys_per_token_;
        const auto status = error_.get();
        int error = 0;
        execute([&] {
            queue_.fill(status, 0, 1);
            queue_.parallel_for(sycl::range<1>(keys_per_token_), [=](sycl::id<1> i) {
                const auto key = keys.data[(i[0] / width) * keys.head_stride + i[0] % width];
                if (!sycl::isfinite(decode_half(key))) flag_invalid(status);
                target[i[0]] = key;
            });
            queue_.fill(state.pending_scores.get() + slot * heads, Score{}, heads);
            queue_.memcpy(&error, status, sizeof(error));
        }, &error);
        ++state.length;
        state.query_pending = false;
        ++stats_.builder_appends;
        ++stats_.device_builder_appends;
    }

private:
    static constexpr std::size_t work_group_size = 64;

    void score_candidates(const Half * bank, const float * query, std::size_t count) {
        const auto heads = geometry_.heads;
        const auto width = geometry_.dimension;
        const auto group = query_heads_ / heads;
        const auto reps = representatives_;
        const auto ids = ids_.get();
        const auto scores = scores_.get();
        const auto status = error_.get();
        queue_.parallel_for(sycl::range<1>(product(heads, count)), [=](sycl::id<1> id) {
            const auto head = id[0] / count;
            const auto block = ids[id[0] % count];
            const auto keys = bank + (block * heads + head) * reps * width;
            const auto queries = query + head * group * width;
            Score relevance{};
            for (std::uint32_t r = 0; r < reps; ++r) {
                for (std::uint32_t q = 0; q < group; ++q) {
                    for (std::uint32_t d = 0; d < width; ++d) {
                        relevance = relevance + device_score<Score>(queries[static_cast<std::size_t>(q) * width + d]) *
                                                decode_half(keys[static_cast<std::size_t>(r) * width + d]) / group;
                    }
                }
            }
            if (!finite(relevance)) flag_invalid(status);
            scores[id[0]] = relevance;
        });
    }

    void select_candidates(std::size_t count, std::uint32_t remaining, std::size_t stride) {
        const auto ids = ids_.get();
        const auto costs = costs_.get();
        const auto scores = scores_.get();
        const auto output = output_.get();
        const auto heads = geometry_.heads;
        queue_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<std::uint64_t, 1> best(sycl::range<1>(work_group_size), cgh);
            sycl::local_accessor<std::uint32_t, 1> budget(sycl::range<1>(1), cgh);
            cgh.parallel_for(sycl::nd_range<1>(product(heads, work_group_size), work_group_size), [=](sycl::nd_item<1> item) {
                const auto lane = item.get_local_linear_id();
                const auto head = item.get_group_linear_id();
                const auto head_scores = scores + head * count;
                const auto choose = [&](std::uint64_t a, std::uint64_t b) {
                    if (a == UINT64_MAX) return b;
                    if (b == UINT64_MAX) return a;
                    return head_scores[b] > head_scores[a] || (head_scores[b] == head_scores[a] && ids[b] < ids[a]) ? b : a;
                };
                if (lane == 0) budget[0] = remaining;
                item.barrier(sycl::access::fence_space::local_space);
                for (std::size_t i = 0; i < stride && budget[0]; ++i) {
                    std::uint64_t candidate = UINT64_MAX;
                    for (auto j = lane; j < count; j += work_group_size) {
                        if (finite(head_scores[j])) candidate = choose(candidate, j);
                    }
                    best[lane] = candidate;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (auto span = work_group_size / 2; span; span /= 2) {
                        if (lane < span) best[lane] = choose(best[lane], best[lane + span]);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    const auto winner = best[0];
                    if (winner == UINT64_MAX) break;
                    if (lane == 0) {
                        output[head * stride + i] = ids[winner];
                        head_scores[winner] = device_score<Score>(-std::numeric_limits<float>::infinity());
                        budget[0] -= sycl::min(budget[0], costs[winner]);
                    }
                    item.barrier(sycl::access::fence_space::global_and_local);
                }
            });
        });
    }

    void advance_window(Layer & state, const float * query, std::uint64_t first, std::uint64_t count) {
        const auto heads = geometry_.heads;
        const auto width = geometry_.dimension;
        const auto group = query_heads_ / heads;
        const auto local = local_;
        const auto mean = mean_.get();
        const auto keys = state.pending_keys.get();
        const auto scores = state.pending_scores.get();
        const auto status = error_.get();
        queue_.parallel_for(sycl::range<1>(keys_per_token_), [=](sycl::id<1> id) {
            const auto head = id[0] / width;
            const auto d = id[0] % width;
            Score value{};
            for (std::uint32_t q = 0; q < group; ++q) {
                value = value + device_score<Score>(query[(head * group + q) * width + d]) / group;
            }
            mean[id[0]] = value;
        });
        queue_.parallel_for(sycl::range<1>(product(count, heads)), [=](sycl::id<1> id) {
            const auto h = id[0] % heads;
            const auto slot = (first + id[0] / heads) % local;
            const auto row = slot * heads + h;
            Score value = scores[row];
            for (std::uint32_t d = 0; d < width; ++d) value = value + mean[h * width + d] * decode_half(keys[row * width + d]);
            if (!finite(value)) flag_invalid(status);
            scores[row] = value;
        });
    }

    void check_device_query(const float * query) {
        const auto status = error_.get();
        queue_.parallel_for(sycl::range<1>(product(query_heads_, geometry_.dimension)), [=](sycl::id<1> i) {
            if (!sycl::isfinite(query[i[0]])) flag_invalid(status);
        });
    }

    void finalize_row(Layer & state, std::uint64_t position, bool complete) {
        const auto heads = geometry_.heads;
        const auto width = geometry_.dimension;
        const auto reps = representatives_;
        const auto slot = position % local_;
        const auto block = position / geometry_.block_tokens;
        const auto keys = state.pending_keys.get();
        const auto scores = state.pending_scores.get();
        const auto top_keys = state.top_keys.get();
        const auto top_positions = state.top_positions.get();
        const auto top_scores = state.top_scores.get();
        const auto bank = state.bank.get() + block * block_elements_;
        const auto finished_positions = built_positions_.get();
        const auto finished_scores = built_scores_.get();
        queue_.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> id) {
            const auto row = slot * heads + id[0];
            const auto base = id[0] * reps;
            insert_representative(position, scores[row], keys + row * width, reps, width,
                                  top_positions + base, top_scores + base, top_keys + base * width);
            if (!complete) return;
            for (std::uint32_t r = 0; r < reps; ++r) {
                const auto i = base + r;
                finished_positions[i] = top_positions[i];
                finished_scores[i] = top_scores[i];
                top_positions[i] = UINT64_MAX;
                for (std::uint32_t d = 0; d < width; ++d) bank[i * width + d] = top_keys[i * width + d];
            }
        });
    }

    void order_host_ties(BuiltRepresentatives & built) const {
        if constexpr (std::is_same_v<Score, FloatPair>) {
            const auto width = geometry_.dimension;
            for (std::uint32_t h = 0; h < geometry_.heads; ++h) {
                const auto start = static_cast<std::size_t>(h) * representatives_;
                for (auto i = start + 1; i < start + representatives_; ++i) {
                    for (auto j = i; j > start && built.scores[j] == built.scores[j - 1] &&
                                     built.positions[j] < built.positions[j - 1]; --j) {
                        std::swap(built.positions[j], built.positions[j - 1]);
                        auto current = built.keys.begin() + j * width;
                        std::swap_ranges(current, current + width, current - width);
                    }
                }
            }
        }
    }

    Layer & checked_layer(std::uint32_t layer) {
        if (failed_) throw std::runtime_error("SYCL block ranker failed");
        if (layer >= layers_.size()) throw std::invalid_argument("SYCL ranker layer");
        return layers_[layer];
    }
    Layer & builder_layer(std::uint32_t layer, std::uint64_t position, bool pending) {
        auto & state = checked_layer(layer);
        if (!state.initialized || state.query_pending != pending || state.length != position || position >= geometry_.capacity_tokens) {
            throw std::invalid_argument("SYCL builder query/append sequence");
        }
        return state;
    }
    void validate_query(const std::vector<float> & query) const {
        if (query.size() != product(query_heads_, geometry_.dimension) ||
            !std::all_of(query.begin(), query.end(), [](float q) { return std::isfinite(q); })) {
            throw std::invalid_argument("invalid SYCL selection query");
        }
    }
    void validate_pointer(const void * pointer) const {
        const auto context = queue_.get_context();
        const auto kind = sycl::get_pointer_type(pointer, context);
        const auto device = queue_.get_device();
        if (kind == sycl::usm::alloc::host || (kind == sycl::usm::alloc::unknown && device.has(sycl::aspect::usm_system_allocations))) return;
        if (kind == sycl::usm::alloc::unknown || sycl::get_pointer_device(pointer, context) != device) {
            throw std::invalid_argument("borrowed SYCL input belongs to another device or context");
        }
    }
    static void validate_keys(const std::vector<Half> & keys) {
        if (!std::all_of(keys.begin(), keys.end(), [](Half key) { return std::isfinite(from_half(key)); })) {
            throw std::invalid_argument("non-finite SYCL representative key");
        }
    }
    template<class T> Buffer<T> allocate(std::size_t count, bool builder = false) {
        const auto bytes = product(count, sizeof(T));
        Buffer<T> buffer(sycl::malloc_device<T>(count, queue_), UsmDelete{&queue_});
        if (!buffer) throw std::bad_alloc();
        stats_.device_bytes += bytes;
        if (builder) stats_.builder_device_bytes += bytes;
        return buffer;
    }
    void ensure_bank(Layer & state) { if (!state.bank) state.bank = allocate<Half>(bank_elements_); }
    template<class Submit> void execute(Submit submit, const int * error = nullptr) {
        try {
            submit();
            queue_.wait_and_throw();
            if (error && *error) throw std::runtime_error("non-finite SYCL selection score");
        } catch (...) {
            failed_ = true;
            try { queue_.wait(); } catch (...) {}
            throw;
        }
    }

    sycl::queue queue_;
    KvGeometry geometry_;
    std::uint32_t query_heads_, representatives_, budget_, local_;
    std::size_t keys_per_token_ = 0, block_elements_ = 0, bank_elements_ = 0;
    std::vector<Layer> layers_;
    Buffer<float> query_;
    Buffer<std::uint32_t> costs_;
    Buffer<std::uint64_t> ids_, output_, built_positions_;
    Buffer<Score> scores_, mean_, built_scores_;
    Buffer<int> error_;
    std::vector<std::uint64_t> output_host_;
    RankerStats stats_;
    bool failed_ = false;
};
} // namespace

std::unique_ptr<BlockRanker> make_sycl_block_ranker(const KvGeometry & geometry, std::uint32_t query_heads,
                                                   std::uint32_t representatives, std::uint32_t budget, void * backend_queue,
                                                   std::uint32_t local) {
    if (!backend_queue) throw std::invalid_argument("SYCL ranker requires an attention backend queue");
    const auto & queue = *static_cast<sycl::queue *>(backend_queue);
    if (queue.get_device().has(sycl::aspect::fp64)) {
        return std::make_unique<SyclBlockRanker<double>>(geometry, query_heads, representatives, budget, queue, local);
    }
    return std::make_unique<SyclBlockRanker<FloatPair>>(geometry, query_heads, representatives, budget, queue, local);
}
} // namespace solid
