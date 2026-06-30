/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
#include <immintrin.h>
#endif

namespace faiss {

/**
 * Implements the core logic of Panorama-based refinement.
 * arXiv: https://arxiv.org/abs/2510.00566
 *
 * Panorama partitions the dimensions of all vectors into L contiguous levels.
 * During the refinement stage of ANNS, it computes distances between the query
 * and its candidates level-by-level. After processing each level, it prunes the
 * candidates whose lower bound exceeds the k-th best distance.
 *
 * In order to enable speedups, the dimensions (or codes) of each vector are
 * stored in a batched, level-major manner. Within each batch of b vectors, the
 * dimensions corresponding to level 1 will be stored first (for all elements in
 * that batch), followed by level 2, and so on. This allows for efficient memory
 * access patterns.
 *
 * Coupled with the appropriate orthogonal PreTransform (e.g. PCA, Cayley,
 * etc.), Panorama can prune the vast majority of dimensions, greatly
 * accelerating the refinement stage.
 *
 * This is the abstract base class. Concrete subclasses (PanoramaFlat,
 * PanoramaPQ) implement compute_cumulative_sums and progressive_filter_batch
 * for their respective code formats.
 */
struct Panorama {
    static constexpr size_t kDefaultBatchSize = 128;

    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width_bytes = 0;
    size_t batch_size = 0;

    float epsilon = 1.0f;

    /// Ablation toggle: if true, search uses the unoptimized kernel
    /// (fused update+prune+compact loop with no Direct/FixedWidth/_pext
    /// optimizations). Default false (use the optimized kernel).
    /// Only respected by PanoramaFlat / IndexIVFFlatPanorama.
    bool use_unoptimized_kernel = false;

    /// Ablation toggle: if true, the optimized kernel skips the
    /// with_level_width compile-time dispatch and always invokes
    /// compute_level_dot_products_flat<Direct, /*FixedWidth=*/0>.
    /// Ignored when use_unoptimized_kernel == true. Default false
    /// (use the templated kernels).
    bool disable_fixed_width = false;

    Panorama() = default;
    Panorama(size_t d, size_t code_size, size_t n_levels, size_t batch_size);

    virtual ~Panorama() = default;

    void set_derived_values();

    /// Helper method to copy codes into level-oriented batch layout at a given
    /// offset in the list.
    /// PanoramaFlat uses row-major within each level (point bytes contiguous).
    /// PanoramaPQ overrides to use column-major (subquantizer columns
    /// contiguous).
    virtual void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

    /// Compute the cumulative sums (suffix norms) for database vectors.
    /// The cumsums follow the level-oriented batch layout to minimize the
    /// number of random memory accesses.
    /// Subclasses interpret the raw code bytes according to their format:
    /// PanoramaFlat reinterprets as float*, PanoramaPQ decodes via PQ.
    virtual void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const = 0;

    /// Compute the cumulative sums of the query vector.
    void compute_query_cum_sums(const float* query, float* query_cum_sums)
            const;

    /// Copy single entry (code and cum_sum) from one location to another.
    void copy_entry(
            uint8_t* dest_codes,
            uint8_t* src_codes,
            float* dest_cum_sums,
            float* src_cum_sums,
            size_t dest_idx,
            size_t src_idx) const;

    virtual void reconstruct(
            idx_t key,
            float* recons,
            const uint8_t* codes_base) const;
};

/// Compute dot products between one query level and multiple database vectors
/// at that level, using a batch-of-4 pattern for amortized query loads and
/// better ILP. Wrapped in FAISS_PRAGMA_IMPRECISE to enable FMA and
/// reassociation for autovectorization.
///
/// When Direct=true, active_indices is the identity mapping (i.e. the
/// i-th active element lives at position i in level_storage). This lets
/// the compiler see a simple strided access pattern and vectorize
/// aggressively. Use on level 0 before any pruning has occurred.
///
/// FixedWidth: when non-zero, the compiler sees the inner loop bound as a
/// compile-time constant, enabling full unrolling and permanent query
/// register allocation (e.g. for FixedWidth=32, the query stays in 2 zmm
/// registers across all vectors regardless of dataset).
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool Direct = false, size_t FixedWidth = 0>
static inline void compute_level_dot_products_flat(
        const float* FAISS_RESTRICT query_level,
        const float* FAISS_RESTRICT level_storage,
        const uint32_t* active_indices,
        size_t num_active,
        size_t level_width_dims,
        float* FAISS_RESTRICT dot_products,
        size_t stride) {
    const size_t width = FixedWidth > 0 ? FixedWidth : level_width_dims;

    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage +
                (Direct ? (i + 0) : active_indices[i + 0]) * stride;
        const float* y1 = level_storage +
                (Direct ? (i + 1) : active_indices[i + 1]) * stride;
        const float* y2 = level_storage +
                (Direct ? (i + 2) : active_indices[i + 2]) * stride;
        const float* y3 = level_storage +
                (Direct ? (i + 3) : active_indices[i + 3]) * stride;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (size_t j = 0; j < width; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }

    for (; i < num_active; i++) {
        const float* yj =
                level_storage + (Direct ? i : active_indices[i]) * stride;
        float dp = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (size_t j = 0; j < width; j++) {
            dp += query_level[j] * yj[j];
        }
        dot_products[i] = dp;
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Compact active_indices in-place, removing entries where keep_mask[i]
/// is zero. Returns the new count of active elements. Uses a branchless BMI2 +
/// AVX2 fast path (8 elements/iteration via _pext_u64 permutation) with a
/// scalar fallback for the tail and non-x86 platforms.
static inline size_t compact_active_pext(
        uint32_t* active_indices,
        const uint8_t* keep_mask,
        size_t num_active) {
    size_t next_active = 0;
    size_t i = 0;

#if defined(__BMI2__) && defined(__AVX2__)
    for (; i + 8 <= num_active; i += 8) {
        uint64_t bytes;
        memcpy(&bytes, &keep_mask[i], 8);

        uint64_t expanded = bytes * 0xFFULL;
        // Lane indices in little-endian order (safe: BMI2+AVX2 implies x86).
        uint64_t packed = _pext_u64(0x0706050403020100ULL, expanded);

        __m256i perm = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((int64_t)packed));
        __m256i data = _mm256_loadu_si256((const __m256i*)&active_indices[i]);
        __m256i compacted = _mm256_permutevar8x32_epi32(data, perm);
        _mm256_storeu_si256((__m256i*)&active_indices[next_active], compacted);

        next_active += __builtin_popcountll(bytes);
    }
#endif

    for (; i < num_active; i++) {
        active_indices[next_active] = active_indices[i];
        next_active += keep_mask[i] ? 1 : 0;
    }

    return next_active;
}

/// Distance-update + Cauchy-Schwarz pruning kernel. Separate function with
/// FAISS_RESTRICT so the compiler proves no aliasing and auto-vectorizes
/// the contiguous (Direct=true) path. The IMPRECISE pragma creates a hard
/// optimization boundary that enforces the restrict contract.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool Direct, typename C, MetricType M>
static inline void prune_level_kernel(
        float* FAISS_RESTRICT exact_distances,
        const float* FAISS_RESTRICT dot_buffer,
        const float* FAISS_RESTRICT level_cum_sums,
        uint8_t* FAISS_RESTRICT keep_mask,
        const uint32_t* FAISS_RESTRICT active_indices,
        uint32_t num_active,
        float query_cum_norm,
        float threshold,
        float epsilon) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (uint32_t i = 0; i < num_active; i++) {
        uint32_t idx = Direct ? i : active_indices[i];
        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[idx] += dot_buffer[i];
        } else {
            exact_distances[idx] -= 2.0f * dot_buffer[i];
        }

        float cum_sum = level_cum_sums[idx];
        float cauchy_schwarz_bound;
        if constexpr (M == METRIC_INNER_PRODUCT) {
            cauchy_schwarz_bound = -cum_sum * query_cum_norm;
        } else {
            cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
        }

        float lower_bound =
                exact_distances[idx] - (epsilon * cauchy_schwarz_bound);
        // Inline the comparison instead of calling C::cmp — the GCC
        // optimize pragma boundary prevents cross-TU inlining, which
        // turns C::cmp into a PLT call and kills vectorization.
        if constexpr (C::is_max) {
            keep_mask[i] = (threshold > lower_bound) ? 1 : 0;
        } else {
            keep_mask[i] = (threshold < lower_bound) ? 1 : 0;
        }
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// Runtime-to-compile-time dispatch for level widths.
/// Specializes for multiples of 8 up to 128; falls back to W=0 (generic).
namespace detail {
template <size_t Lo, size_t Hi, size_t Step, typename Lambda>
inline auto dispatch_width(size_t width, Lambda&& fn) {
    if constexpr (Lo > Hi) {
        return fn.template operator()<0>();
    } else {
        if (width == Lo) {
            return fn.template operator()<Lo>();
        }
        return dispatch_width<Lo + Step, Hi, Step>(
                width, std::forward<Lambda>(fn));
    }
}
} // namespace detail

template <typename LambdaType>
inline auto with_level_width(size_t width, LambdaType&& action) {
    return detail::dispatch_width<8, 128, 8>(
            width, std::forward<LambdaType>(action));
}

template <typename Lambda>
inline auto with_bool(bool value, Lambda&& fn) {
    if (value) {
        return fn.template operator()<true>();
    } else {
        return fn.template operator()<false>();
    }
}

/// Processes one level of Panorama flat filtering. Templated on Direct
/// (identity mapping vs indirect) to enable full vectorization of the
/// distance-update and pruning loops when active_indices[i] == i.
///
/// When DisableFixedWidth=true, we bypass the with_level_width
/// compile-time dispatch and always call
/// compute_level_dot_products_flat<Direct, /*FixedWidth=*/0>. This
/// disables the templated unrolling/permanent-register optimization and
/// is used by the systems-ablation experiments.
template <bool Direct, typename C, MetricType M, bool DisableFixedWidth = false>
static inline size_t panorama_flat_level_body(
        const float* query_level,
        const float* level_storage,
        uint32_t* active_indices,
        size_t num_active,
        size_t actual_level_width,
        float* dot_buffer,
        uint8_t* keep_mask,
        float* exact_distances,
        const float* level_cum_sums,
        float query_cum_norm,
        float threshold,
        float epsilon) {
    if constexpr (DisableFixedWidth) {
        compute_level_dot_products_flat<Direct, /*FixedWidth=*/0>(
                query_level,
                level_storage,
                active_indices,
                num_active,
                actual_level_width,
                dot_buffer,
                actual_level_width);
    } else {
        with_level_width(actual_level_width, [&]<size_t W>() {
            compute_level_dot_products_flat<Direct, W>(
                    query_level,
                    level_storage,
                    active_indices,
                    num_active,
                    actual_level_width,
                    dot_buffer,
                    actual_level_width);
        });
    }

    prune_level_kernel<Direct, C, M>(
            exact_distances,
            dot_buffer,
            level_cum_sums,
            keep_mask,
            active_indices,
            (uint32_t)num_active,
            query_cum_norm,
            threshold,
            epsilon);

    return compact_active_pext(active_indices, keep_mask, num_active);
}

/**
 * Panorama for flat (uncompressed) float vectors.
 *
 * Codes are raw float vectors (code_size = d * sizeof(float)).
 * compute_cumulative_sums interprets codes as floats.
 * progressive_filter_batch computes dot products on raw float storage.
 */
struct PanoramaFlat : Panorama {
    size_t level_width_dims = 0;

    PanoramaFlat() = default;
    PanoramaFlat(size_t d, size_t n_levels, size_t batch_size);

    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const override;

    /// Panorama's core progressive filtering algorithm for flat codes:
    /// Process vectors in batches for cache efficiency. For each batch:
    /// 1. Apply ID selection filter and initialize distances
    /// (||y||^2 + ||x||^2).
    /// 2. Maintain an "active set" of candidate indices that haven't been
    /// pruned yet.
    /// 3. For each level, use a two-pass approach:
    ///    Pass 1: Compute dot products for all active points via
    ///            compute_level_dot_products_flat (batch-of-4, autovectorized).
    ///    Pass 2: Update distances, apply Cauchy-Schwarz pruning, and compact
    ///            the active set.
    /// 4. After all levels, survivors are exact distances; update heap.
    ///
    /// DisableFixedWidth=true bypasses the with_level_width compile-time
    /// dispatch in panorama_flat_level_body (used for the systems
    /// ablation; it isolates the speedup contribution of the templated
    /// FixedWidth distance kernels from the rest of the new pipeline).
    template <typename C, MetricType M, bool DisableFixedWidth = false>
    size_t progressive_filter_batch(
            const uint8_t* codes_base,
            const float* cum_sums,
            const float* query,
            const float* query_cum_sums,
            size_t batch_no,
            size_t list_size,
            const IDSelector* sel,
            const idx_t* ids,
            bool use_sel,
            std::vector<uint32_t>& active_indices,
            std::vector<float>& exact_distances,
            std::vector<float>& dot_buffer,
            std::vector<uint8_t>& keep_mask,
            float threshold,
            PanoramaStats& local_stats) const;

    /// Unoptimized variant of progressive_filter_batch. Used as the
    /// systems-ablation baseline. Differences vs.
    /// progressive_filter_batch:
    ///   - No batched dot-product pass: the per-candidate dot product
    ///     is computed *inside* the inner loop via fvec_inner_product,
    ///     i.e. one query/code dot product at a time. No dot_buffer,
    ///     no batch-of-4, no FixedWidth specialization, no Direct
    ///     specialization at level 0.
    ///   - distance update + Cauchy-Schwarz pruning + active-set
    ///     compaction are FUSED into a single branchy loop (no separate
    ///     prune_level_kernel, no compact_active_pext / BMI2-AVX2
    ///     compaction).
    ///   - No `epsilon` relaxation on the Cauchy-Schwarz bound: this
    ///     is the historical code from before `epsilon` was introduced,
    ///     equivalent to `epsilon = 1.0`. Variants that schedule the
    ///     unoptimized path therefore force `epsilon = 1.0` upstream
    ///     so results stay apples-to-apples with the optimized path.
    ///   - No early break when num_active drops to 0 mid-batch.
    /// Otherwise produces bit-identical results (modulo float
    /// reordering inside fvec_inner_product) to the optimized kernel
    /// at `epsilon = 1.0`.
    template <typename C, MetricType M>
    size_t progressive_filter_batch_unoptimized(
            const uint8_t* codes_base,
            const float* cum_sums,
            const float* query,
            const float* query_cum_sums,
            size_t batch_no,
            size_t list_size,
            const IDSelector* sel,
            const idx_t* ids,
            bool use_sel,
            std::vector<uint32_t>& active_indices,
            std::vector<float>& exact_distances,
            float threshold,
            PanoramaStats& local_stats) const;
};

template <typename C, MetricType M, bool DisableFixedWidth>
size_t PanoramaFlat::progressive_filter_batch(
        const uint8_t* codes_base,
        const float* cum_sums,
        const float* query,
        const float* query_cum_sums,
        size_t batch_no,
        size_t list_size,
        const IDSelector* sel,
        const idx_t* ids,
        bool use_sel,
        std::vector<uint32_t>& active_indices,
        std::vector<float>& exact_distances,
        std::vector<float>& dot_buffer,
        std::vector<uint8_t>& keep_mask,
        float threshold,
        PanoramaStats& local_stats) const {
    size_t batch_start = batch_no * batch_size;
    size_t curr_batch_size = std::min(list_size - batch_start, batch_size);

    size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);
    const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
    const float* level_cum_sums = batch_cum_sums + batch_size;
    float q_norm = query_cum_sums[0] * query_cum_sums[0];

    size_t batch_offset = batch_no * batch_size * code_size;
    const uint8_t* storage_base = codes_base + batch_offset;

    // Initialize active set with ID-filtered vectors.
    size_t num_active = 0;
    for (size_t i = 0; i < curr_batch_size; i++) {
        size_t global_idx = batch_start + i;
        idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
        bool include = !use_sel || sel->is_member(id);

        active_indices[num_active] = i;
        float cum_sum = batch_cum_sums[i];

        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[i] = 0.0f;
        } else {
            exact_distances[i] = cum_sum * cum_sum + q_norm;
        }

        num_active += include;
    }

    size_t total_active = num_active;
    bool first_level_direct = (num_active == curr_batch_size);

    local_stats.total_dims += total_active * n_levels;

    for (size_t level = 0; (level < n_levels) && (num_active > 0); level++) {
        local_stats.total_dims_scanned += num_active;

        float query_cum_norm = query_cum_sums[level + 1];

        size_t level_offset = level * level_width_bytes * batch_size;
        const float* level_storage =
                (const float*)(storage_base + level_offset);
        const float* query_level = query + level * level_width_dims;
        size_t actual_level_width =
                std::min(level_width_dims, d - level * level_width_dims);

        num_active =
                with_bool(level == 0 && first_level_direct, [&]<bool Direct>() {
                    return panorama_flat_level_body<
                            Direct, C, M, DisableFixedWidth>(
                            query_level,
                            level_storage,
                            active_indices.data(),
                            num_active,
                            actual_level_width,
                            dot_buffer.data(),
                            keep_mask.data(),
                            exact_distances.data(),
                            level_cum_sums,
                            query_cum_norm,
                            threshold,
                            epsilon);
                });

        level_cum_sums += batch_size;
    }

    return num_active;
}

template <typename C, MetricType M>
size_t PanoramaFlat::progressive_filter_batch_unoptimized(
        const uint8_t* codes_base,
        const float* cum_sums,
        const float* query,
        const float* query_cum_sums,
        size_t batch_no,
        size_t list_size,
        const IDSelector* sel,
        const idx_t* ids,
        bool use_sel,
        std::vector<uint32_t>& active_indices,
        std::vector<float>& exact_distances,
        float threshold,
        PanoramaStats& local_stats) const {
    size_t batch_start = batch_no * batch_size;
    size_t curr_batch_size = std::min(list_size - batch_start, batch_size);

    size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);
    const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
    const float* level_cum_sums = batch_cum_sums + batch_size;
    float q_norm = query_cum_sums[0] * query_cum_sums[0];

    size_t batch_offset = batch_no * batch_size * code_size;
    const uint8_t* storage_base = codes_base + batch_offset;

    // Initialize active set with ID-filtered vectors.
    size_t num_active = 0;
    for (size_t i = 0; i < curr_batch_size; i++) {
        size_t global_idx = batch_start + i;
        idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
        bool include = !use_sel || sel->is_member(id);

        active_indices[num_active] = i;
        float cum_sum = batch_cum_sums[i];

        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[i] = 0.0f;
        } else {
            exact_distances[i] = cum_sum * cum_sum + q_norm;
        }

        num_active += include;
    }

    if (num_active == 0) {
        return 0;
    }

    size_t total_active = num_active;
    for (size_t level = 0; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;
        local_stats.total_dims += total_active;

        float query_cum_norm = query_cum_sums[level + 1];

        size_t level_offset = level * level_width_bytes * batch_size;
        const float* level_storage =
                (const float*)(storage_base + level_offset);

        size_t next_active = 0;
        for (size_t i = 0; i < num_active; i++) {
            uint32_t idx = active_indices[i];
            size_t actual_level_width = std::min(
                    level_width_dims, d - level * level_width_dims);

            const float* yj = level_storage + idx * actual_level_width;
            const float* query_level = query + level * level_width_dims;

            float dot_product =
                    fvec_inner_product(query_level, yj, actual_level_width);

            if constexpr (M == METRIC_INNER_PRODUCT) {
                exact_distances[idx] += dot_product;
            } else {
                exact_distances[idx] -= 2.0f * dot_product;
            }

            float cum_sum = level_cum_sums[idx];
            float cauchy_schwarz_bound;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                cauchy_schwarz_bound = -cum_sum * query_cum_norm;
            } else {
                cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
            }

            float lower_bound = exact_distances[idx] - cauchy_schwarz_bound;

            active_indices[next_active] = idx;
            next_active += C::cmp(threshold, lower_bound) ? 1 : 0;
        }

        num_active = next_active;
        level_cum_sums += batch_size;
    }

    return num_active;
}
} // namespace faiss

#endif
