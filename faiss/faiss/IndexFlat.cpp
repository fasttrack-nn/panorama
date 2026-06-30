/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/sorting.h>
#include <omp.h>
#include <cstring>

namespace faiss {

IndexFlat::IndexFlat(idx_t d_, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d_, d_, metric) {}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
    } else {
        knn_extra_metrics(
                x,
                get_xb(),
                d,
                n,
                ntotal,
                metric_type,
                metric_arg,
                k,
                distances,
                labels,
                sel);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

template <SIMDLevel SL>
struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* b;
    size_t ndis;
    size_t npartial_dot_products;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr<SL>(q, (float*)code, d);
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products++;
        return fvec_inner_product<SL>(
                q + offset, b + i * d + offset, num_components);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr<SL>(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q_ = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size,
                      q_),
              d(storage.d),
              nb(storage.ntotal),
              b(storage.get_xb()),
              ndis(0),
              npartial_dot_products(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4<SL>(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

    void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0_ = 0;
        float dp1_ = 0;
        float dp2_ = 0;
        float dp3_ = 0;
        fvec_inner_product_batch_4<SL>(
                q + offset,
                y0 + offset,
                y1 + offset,
                y2 + offset,
                y3 + offset,
                num_components,
                dp0_,
                dp1_,
                dp2_,
                dp3_);
        dp0 = dp0_;
        dp1 = dp1_;
        dp2 = dp2_;
        dp3 = dp3_;
    }
};

template <SIMDLevel SL>
struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product<SL>(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product<SL>(q, (const float*)code, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q_in = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q_in),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4<SL>(
                q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    FlatCodesDistanceComputer* dc = nullptr;
    if (metric_type == METRIC_L2) {
        with_simd_level([&]<SIMDLevel SL>() { dc = new FlatL2Dis<SL>(*this); });
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        with_simd_level([&]<SIMDLevel SL>() { dc = new FlatIPDis<SL>(*this); });
    } else {
        dc = get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
    return dc;
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key < ntotal);
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

/***************************************************
 * IndexFlatL2
 ***************************************************/

namespace {
template <SIMDLevel SL>
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* l2norms;
    float query_l2norm;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr<SL>(q, (float*)code, d);
    }

    float operator()(const idx_t i) final override {
        const float* __restrict y =
                reinterpret_cast<const float*>(codes + i * code_size);

        prefetch_L2(l2norms + i);
        const float dp0 = fvec_inner_product<SL>(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict yi =
                reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict yj =
                reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);
        const float dp0 = fvec_inner_product<SL>(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDis(
            const IndexFlatL2& storage,
            const float* q_in = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q_in),
              b(storage.get_xb()),
              ndis(0),
              l2norms(storage.cached_l2norms.data()),
              query_l2norm(0) {}

    void set_query(const float* x) override {
        q = x;
        query_l2norm = fvec_norm_L2sqr<SL>(q, d);
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4<SL>(
                q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};

} // namespace

void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
            cached_l2norms.data(),
            reinterpret_cast<const float*>(codes.data()),
            d,
            ntotal);
}

void IndexFlatL2::clear_l2norms() {
    cached_l2norms.clear();
    cached_l2norms.shrink_to_fit();
}

FlatCodesDistanceComputer* IndexFlatL2::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        if (!cached_l2norms.empty()) {
            FlatCodesDistanceComputer* dc = nullptr;
            with_simd_level([&]<SIMDLevel SL>() {
                dc = new FlatL2WithNormsDis<SL>(*this);
            });
            return dc;
        }
    }

    return IndexFlat::get_FlatCodesDistanceComputer();
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update_in)
        : IndexFlatL2(1), continuous_update(continuous_update_in) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update) {
        update_permutation();
    }
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            perm.size() == static_cast<size_t>(ntotal),
            "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            goto done;
        }

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) {
                i0 = imed;
            } else {
                i1 = imed;
            }
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

/**************************************************************
 * shared flat Panorama search code
 **************************************************************/

namespace {

template <typename Fn>
inline auto dispatch_metric_compare(MetricType metric, Fn&& fn) {
    if (is_similarity_metric(metric)) {
        using C = CMin<float, int64_t>;
        return fn.template operator()<C>();
    }
    using C = CMax<float, int64_t>;
    return fn.template operator()<C>();
}

template <bool use_radius, typename C, typename BlockHandler>
inline void flat_pano_search_core(
        const IndexFlatPanorama& index,
        BlockHandler& handler,
        idx_t n,
        const float* x,
        float radius,
        const SearchParameters* params) {
    using SingleResultHandler = typename BlockHandler::SingleResultHandler;

    IDSelector* sel = params ? params->sel : nullptr;
    bool use_sel = sel != nullptr;

    [[maybe_unused]] int nt = std::min(int(n), omp_get_max_threads());
    size_t n_batches = (index.ntotal + index.batch_size - 1) / index.batch_size;

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler res(handler);

        std::vector<float> query_cum_norms(index.n_levels + 1);
        std::vector<float> exact_distances(index.batch_size);
        std::vector<uint32_t> active_indices(index.batch_size);
        std::vector<float> dot_buffer(index.batch_size);
        std::vector<uint8_t> keep_mask(index.batch_size);

#pragma omp for
        for (int64_t i = 0; i < n; i++) {
            const float* xi = x + i * index.d;
            index.pano.compute_query_cum_sums(xi, query_cum_norms.data());

            PanoramaStats local_stats;
            local_stats.reset();

            res.begin(i);

            for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
                size_t batch_start = batch_no * index.batch_size;

                float threshold;
                if constexpr (use_radius) {
                    threshold = radius;
                } else {
                    threshold = res.heap_dis[0];
                }

                size_t num_active = with_metric_type(
                        index.metric_type, [&]<MetricType M>() {
                            return index.pano.progressive_filter_batch<C, M>(
                                    index.codes.data(),
                                    index.cum_sums.data(),
                                    xi,
                                    query_cum_norms.data(),
                                    batch_no,
                                    index.ntotal,
                                    sel,
                                    nullptr,
                                    use_sel,
                                    active_indices,
                                    exact_distances,
                                    dot_buffer,
                                    keep_mask,
                                    threshold,
                                    local_stats);
                        });

                for (size_t j = 0; j < num_active; j++) {
                    res.add_result(
                            exact_distances[active_indices[j]],
                            batch_start + active_indices[j]);
                }
            }

            res.end();
            indexPanorama_stats.add(local_stats);
        }
    }
}

} // anonymous namespace

/***************************************************
 * IndexFlatPanorama
 ***************************************************/

void IndexFlatPanorama::add(idx_t n, const float* x) {
    size_t offset = ntotal;
    ntotal += n;
    size_t num_batches = (ntotal + batch_size - 1) / batch_size;

    codes.resize(num_batches * batch_size * code_size);
    cum_sums.resize(num_batches * batch_size * (n_levels + 1));

    const uint8_t* code = reinterpret_cast<const uint8_t*>(x);
    pano.copy_codes_to_level_layout(codes.data(), offset, n, code);
    pano.compute_cumulative_sums(cum_sums.data(), offset, n, code);
}

void IndexFlatPanorama::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(batch_size >= static_cast<size_t>(k));

    dispatch_metric_compare(metric_type, [&]<typename C>() {
        HeapBlockResultHandler<C, false> handler(
                size_t(n), distances, labels, size_t(k), nullptr);
        flat_pano_search_core<false, C>(*this, handler, n, x, 0.0f, params);
    });
}

void IndexFlatPanorama::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    dispatch_metric_compare(metric_type, [&]<typename C>() {
        RangeSearchBlockResultHandler<C, false> handler(
                result, radius, nullptr);
        flat_pano_search_core<true, C>(*this, handler, n, x, radius, params);
    });
}

void IndexFlatPanorama::reset() {
    IndexFlat::reset();
    cum_sums.clear();
}

void IndexFlatPanorama::reconstruct(idx_t key, float* recons) const {
    pano.reconstruct(key, recons, codes.data());
}

void IndexFlatPanorama::reconstruct_n(idx_t i, idx_t n, float* recons) const {
    Index::reconstruct_n(i, n, recons);
}

size_t IndexFlatPanorama::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                pano.copy_entry(
                        codes.data(),
                        codes.data(),
                        cum_sums.data(),
                        cum_sums.data(),
                        j,
                        i);
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        size_t num_batches = (ntotal + batch_size - 1) / batch_size;
        codes.resize(num_batches * batch_size * code_size);
        cum_sums.resize(num_batches * batch_size * (n_levels + 1));
    }
    return nremove;
}

void IndexFlatPanorama::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatPanorama index");
    check_compatible_for_merge(otherIndex);
    IndexFlatPanorama* other = static_cast<IndexFlatPanorama*>(&otherIndex);

    std::vector<float> buffer(other->ntotal * code_size);
    otherIndex.reconstruct_n(0, other->ntotal, buffer.data());

    add(other->ntotal, buffer.data());
    other->reset();
}

void IndexFlatPanorama::add_sa_codes(
        idx_t /* n */,
        const uint8_t* /* codes_in */,
        const idx_t* /* xids */) {
    FAISS_THROW_MSG("add_sa_codes not implemented for IndexFlatPanorama");
}

void IndexFlatPanorama::permute_entries(const idx_t* perm) {
    MaybeOwnedVector<uint8_t> new_codes(codes.size());
    std::vector<float> new_cum_sums(cum_sums.size());

    for (idx_t i = 0; i < ntotal; i++) {
        pano.copy_entry(
                new_codes.data(),
                codes.data(),
                new_cum_sums.data(),
                cum_sums.data(),
                i,
                perm[i]);
    }

    std::swap(codes, new_codes);
    std::swap(cum_sums, new_cum_sums);
}

/// Branchless distance-update + Cauchy-Schwarz pruning + keep-mask kernel.
/// Pulled out so the compiler sees __restrict pointers and the
/// FAISS_PRAGMA_IMPRECISE optimization barrier, both of which are critical
/// for autovectorization with AVX-512 mask ops.
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <bool is_sim>
static inline void refine_pano_prune_kernel(
        float* FAISS_RESTRICT exact_distances,
        const float* FAISS_RESTRICT dot_buffer,
        const float* FAISS_RESTRICT level_cs,
        uint8_t* FAISS_RESTRICT keep_mask,
        size_t batch_size,
        float query_cum_norm,
        float two_qc,
        float eps,
        float threshold) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t b = 0; b < batch_size; b++) {
        float d_new;
        if constexpr (is_sim) {
            d_new = exact_distances[b] + dot_buffer[b];
        } else {
            d_new = exact_distances[b] - 2.0f * dot_buffer[b];
        }
        exact_distances[b] = d_new;

        float cs_bound;
        if constexpr (is_sim) {
            cs_bound = -level_cs[b] * query_cum_norm;
        } else {
            cs_bound = two_qc * level_cs[b];
        }
        float lower_bound = d_new - eps * cs_bound;
        if constexpr (is_sim) {
            keep_mask[b] = (lower_bound >= threshold) ? 1 : 0;
        } else {
            keep_mask[b] = (lower_bound <= threshold) ? 1 : 0;
        }
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/// In-place compaction of the four parallel arrays based on keep_mask.
/// Returns the new active count. Branchy gather but fast for small batches
/// (BLOCK ~ 64); the dominant cost is the dot product, not this.
static inline size_t refine_pano_compact(
        uint32_t* index_array,
        float* exact_distances,
        const float** cum_ptrs,
        float* level_cs,
        const uint8_t* keep_mask,
        size_t batch_size) {
    size_t next = 0;
    for (size_t b = 0; b < batch_size; b++) {
        index_array[next] = index_array[b];
        exact_distances[next] = exact_distances[b];
        cum_ptrs[next] = cum_ptrs[b];
        // level_cs gets re-gathered next iteration, no need to keep.
        (void)level_cs;
        next += keep_mask[b];
    }
    return next;
}

/// Hyperoptimized refine kernel inspired by IndexHNSWFlatPanorama's
/// search_from_candidates_panorama hot loop.
///
/// Key wins over the naive "one candidate at a time" loop:
///   1. Process candidates in mini-batches of REFINE_BLOCK so the L1-resident
///      query level can be reused across many vectors and the inner
///      `compute_level_dot_products_flat` runs with a compile-time-constant
///      width (full unroll, query held in registers).
///   2. Pass-1 collection: prefetch every candidate's full vector + its
///      cum_sum row before any arithmetic touches them. One sweep through
///      `base_labels` per block hides the random-access latency.
///   3. Hoist all per-level constants (`two_qc`, `eps`, `query_level`,
///      `level_base`, `dim_span`) outside the per-candidate inner loop.
///   4. Gather `level_cum_sums` into a contiguous buffer per level so the
///      prune kernel sees __restrict pointers and autovectorizes.
///   5. Branchless prune kernel + simple compaction; survivors stay
///      contiguous so the next level keeps full SIMD throughput.
///   6. Single SIMD level + metric dispatch at the outermost scope - no
///      runtime dispatch in the loop body.
template <typename C, MetricType M, bool is_sim>
static void refine_panorama_block(
        const IndexFlatPanorama& index,
        idx_t i,
        const idx_t* __restrict idsi,
        idx_t k_base,
        const float* __restrict xi,
        const float* __restrict query_cum_norms,
        float query_cum_norm_sq,
        typename HeapBlockResultHandler<C, false>::SingleResultHandler& res,
        std::vector<uint32_t>& index_array,
        std::vector<float>& exact_distances,
        std::vector<float>& dot_buffer,
        std::vector<float>& level_cs_buffer,
        std::vector<uint8_t>& keep_mask,
        std::vector<const float*>& cum_ptrs,
        PanoramaStats& local_stats) {
    constexpr size_t REFINE_BLOCK = 32;
    const size_t d = index.d;
    const auto& pano = index.pano;
    const size_t n_levels = pano.n_levels;
    const size_t level_width_dims = pano.level_width_dims;
    const size_t cum_stride = n_levels + 1;
    const float eps = pano.epsilon;
    const float* cum_base = index.cum_sums.data();
    const float* codes_base =
            reinterpret_cast<const float*>(index.codes.data());

    res.begin(i);

    for (idx_t block_start = 0; block_start < k_base;
         block_start += static_cast<idx_t>(REFINE_BLOCK)) {
        const idx_t block_end = std::min<idx_t>(
                block_start + static_cast<idx_t>(REFINE_BLOCK), k_base);

        // Pass 1: collect valid candidate ids, prefetch cum_sum row +
        // first cache lines of the vector data, initialize exact distance
        // to ||y||^2 + ||x||^2 (or 0 for IP).
        size_t batch_size = 0;
        for (idx_t j = block_start; j < block_end; j++) {
            const idx_t idx = idsi[j];
            if (idx < 0) {
                continue;
            }
            const float* row =
                    cum_base + static_cast<size_t>(idx) * cum_stride;
            // Prefetch the cum_sum row (n_levels+1 floats, fits a line)
            // and the first cache lines of the vector data (d floats).
            prefetch_L2(row);
            const float* code_row =
                    codes_base + static_cast<size_t>(idx) * d;
            prefetch_L2(code_row);
            if (d * sizeof(float) > 64) {
                prefetch_L2(reinterpret_cast<const uint8_t*>(code_row) + 64);
            }

            float cum0 = row[0];
            index_array[batch_size] = static_cast<uint32_t>(idx);
            cum_ptrs[batch_size] = row;
            if constexpr (is_sim) {
                exact_distances[batch_size] = 0.0f;
            } else {
                exact_distances[batch_size] =
                        query_cum_norm_sq + cum0 * cum0;
            }
            batch_size++;
        }

        if (batch_size == 0) {
            continue;
        }

        local_stats.total_dims += batch_size * d;

        // Pass 2: progressive refinement, one level at a time, batched.
        size_t curr_level = 0;
        while (curr_level < n_levels && batch_size > 0) {
            const size_t cs_level_idx = curr_level + 1;
            const float query_cum_norm = query_cum_norms[cs_level_idx];
            const float two_qc = 2.0f * query_cum_norm;

            const size_t start_dim = curr_level * level_width_dims;
            const size_t end_dim =
                    std::min((curr_level + 1) * level_width_dims, d);
            const size_t dim_span = end_dim - start_dim;

            const float* query_level = xi + start_dim;
            const float* level_base = codes_base + start_dim;

            // Compile-time width specialization → full unroll, query held in
            // SIMD registers across all candidates in the batch.
            with_level_width(dim_span, [&]<size_t W>() {
                compute_level_dot_products_flat</*Direct=*/false, W>(
                        query_level,
                        level_base,
                        index_array.data(),
                        batch_size,
                        dim_span,
                        dot_buffer.data(),
                        d);
            });

            local_stats.total_dims_scanned += batch_size * dim_span;

            // Gather this level's cum_sum into a contiguous buffer so the
            // prune kernel can autovectorize without indirect loads.
            for (size_t b = 0; b < batch_size; b++) {
                level_cs_buffer[b] = cum_ptrs[b][cs_level_idx];
            }

            // Re-read threshold once per level (it tightens as the heap
            // fills). Within the level, the threshold is fixed.
            const float threshold = res.heap_dis[0];
            refine_pano_prune_kernel<is_sim>(
                    exact_distances.data(),
                    dot_buffer.data(),
                    level_cs_buffer.data(),
                    keep_mask.data(),
                    batch_size,
                    query_cum_norm,
                    two_qc,
                    eps,
                    threshold);

            batch_size = refine_pano_compact(
                    index_array.data(),
                    exact_distances.data(),
                    cum_ptrs.data(),
                    level_cs_buffer.data(),
                    keep_mask.data(),
                    batch_size);
            curr_level++;
        }

        // Survivors are now at exact distance — push to heap.
        for (size_t b = 0; b < batch_size; b++) {
            res.add_result(
                    exact_distances[b],
                    static_cast<idx_t>(index_array[b]));
        }
    }

    res.end();
}

void IndexFlatPanorama::search_subset(
        idx_t n,
        const float* x,
        idx_t k_base,
        const idx_t* base_labels,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    with_simd_level([&]<SIMDLevel SL>() {
        (void)SL; // SIMD level captured for ABI-correct dispatch; the inner
                  // loop uses with_level_width<W> + compute_level_dot_products_flat
                  // (autovectorized) so no per-call SIMDLevel is needed.
        with_metric_type(metric_type, [&]<MetricType M>() {
            constexpr bool is_sim = is_similarity_metric(M);
            using C = std::conditional_t<
                    is_sim,
                    CMin<float, int64_t>,
                    CMax<float, int64_t>>;
            HeapBlockResultHandler<C, false> handler(
                    size_t(n), distances, labels, size_t(k), nullptr);

            FAISS_THROW_IF_NOT(k > 0);
            FAISS_THROW_IF_NOT(batch_size == 1);

            // Block size for the refine inner loop. 32 balances two things:
            //   - large enough to amortize the per-block setup +
            //     compute_level_dot_products_flat call overhead, and to keep
            //     the inner dot-product loop running with full SIMD throughput
            //     (1 query level cached in registers, 32 vectors processed
            //     using all available FMA units);
            //   - small enough that the heap threshold tightens often, giving
            //     us most of the per-candidate pruning power of the unbatched
            //     baseline.
            constexpr size_t REFINE_BLOCK = 32;

            [[maybe_unused]] int nt = std::min(int(n), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
            {
                typename HeapBlockResultHandler<C, false>::SingleResultHandler
                        res(handler);

                std::vector<float> query_cum_norms(n_levels + 1);
                std::vector<uint32_t> index_array(REFINE_BLOCK);
                std::vector<float> exact_distances(REFINE_BLOCK);
                std::vector<float> dot_buffer(REFINE_BLOCK);
                std::vector<float> level_cs_buffer(REFINE_BLOCK);
                std::vector<uint8_t> keep_mask(REFINE_BLOCK);
                std::vector<const float*> cum_ptrs(REFINE_BLOCK);

#pragma omp for
                for (idx_t i = 0; i < n; i++) {
                    const idx_t* __restrict idsi = base_labels + i * k_base;
                    const float* xi = x + i * d;

                    PanoramaStats local_stats;
                    local_stats.reset();

                    pano.compute_query_cum_sums(xi, query_cum_norms.data());
                    const float query_cum_norm_sq =
                            query_cum_norms[0] * query_cum_norms[0];

                    refine_panorama_block<C, M, is_sim>(
                            *this,
                            i,
                            idsi,
                            k_base,
                            xi,
                            query_cum_norms.data(),
                            query_cum_norm_sq,
                            res,
                            index_array,
                            exact_distances,
                            dot_buffer,
                            level_cs_buffer,
                            keep_mask,
                            cum_ptrs,
                            local_stats);

                    indexPanorama_stats.add(local_stats);
                }
            }
        });
    });
}

/***************************************************
 * IndexFlatPanoramaInline
 ***************************************************/

IndexFlatPanoramaInline::IndexFlatPanoramaInline(idx_t d_in, size_t n_levels_in)
        : IndexFlatL2(d_in),
          n_levels(n_levels_in),
          pano(d_in, n_levels_in, /*batch_size=*/1),
          cs_per_row(n_levels_in + 1) {
    // Each row is laid out as [ cum_sums (cs_per_row floats) | xb (d floats) ]
    code_size = (cs_per_row + static_cast<size_t>(d_in)) * sizeof(float);
}

void IndexFlatPanoramaInline::sa_encode(
        idx_t n,
        const float* x,
        uint8_t* bytes) const {
    if (n <= 0) {
        return;
    }
    const size_t cs = cs_per_row;
    const size_t row_n = cs + static_cast<size_t>(d);

    // Compute cum-sums for all `n` rows into a contiguous temp buffer
    // (PanoramaFlat::compute_cumulative_sums writes a tightly packed
    // n * (n_levels+1) layout), then scatter the prefix into each
    // output row alongside the raw vector data.
    std::vector<float> tmp_cum(static_cast<size_t>(n) * cs);
    pano.compute_cumulative_sums(
            tmp_cum.data(),
            /*offset=*/0,
            n,
            reinterpret_cast<const uint8_t*>(x));

    float* dst_floats = reinterpret_cast<float*>(bytes);
#pragma omp parallel for if (n >= 1024)
    for (idx_t i = 0; i < n; i++) {
        float* dst = dst_floats + static_cast<size_t>(i) * row_n;
        memcpy(dst, tmp_cum.data() + static_cast<size_t>(i) * cs,
               cs * sizeof(float));
        memcpy(dst + cs, x + static_cast<size_t>(i) * d,
               static_cast<size_t>(d) * sizeof(float));
    }
}

void IndexFlatPanoramaInline::sa_decode(
        idx_t n,
        const uint8_t* bytes,
        float* x) const {
    if (n <= 0) {
        return;
    }
    const size_t cs = cs_per_row;
    const size_t row_n = cs + static_cast<size_t>(d);
    const float* src_floats = reinterpret_cast<const float*>(bytes);
    for (idx_t i = 0; i < n; i++) {
        memcpy(x + static_cast<size_t>(i) * d,
               src_floats + static_cast<size_t>(i) * row_n + cs,
               static_cast<size_t>(d) * sizeof(float));
    }
}

void IndexFlatPanoramaInline::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key < ntotal);
    memcpy(recons, get_inline_xb(key), static_cast<size_t>(d) * sizeof(float));
}

void IndexFlatPanoramaInline::reconstruct_n(idx_t i0, idx_t ni, float* recons)
        const {
    FAISS_THROW_IF_NOT(i0 + ni <= ntotal);
    for (idx_t i = 0; i < ni; i++) {
        memcpy(recons + static_cast<size_t>(i) * d,
               get_inline_xb(i0 + i),
               static_cast<size_t>(d) * sizeof(float));
    }
}

namespace {

/// Distance computer over `IndexFlatPanoramaInline` storage.
///
/// Each row in `storage.codes` is laid out as
///   [ cum_sums (cs_per_row floats) | xb (d floats) ]
/// `code_size` strides one full row, so we just rebase `codes` past the
/// prefix and reuse the standard L2 SIMD kernels — no algorithmic change
/// versus `FlatL2Dis`, just a different starting offset.
template <SIMDLevel SL>
struct FlatPanoramaInlineL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* b; // pointer to the first vector's xb (post-prefix)
    size_t row_floats; // = cs_per_row + d
    size_t prefix_floats; // = cs_per_row
    size_t ndis;
    size_t npartial_dot_products;

    explicit FlatPanoramaInlineL2Dis(
            const IndexFlatPanoramaInline& storage,
            const float* q_ = nullptr)
            : FlatCodesDistanceComputer(
                      // codes pointer rebased to the first vector's xb so
                      // `codes + i * code_size` lands at row i's xb start.
                      storage.codes.data() +
                              storage.cs_per_row * sizeof(float),
                      storage.code_size,
                      q_),
              d(storage.d),
              nb(storage.ntotal),
              b(storage.get_inline_xb(0)),
              row_floats(storage.row_floats()),
              prefix_floats(storage.cs_per_row),
              ndis(0),
              npartial_dot_products(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr<SL>(q, (const float*)code, d);
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products++;
        return fvec_inner_product<SL>(
                q + offset,
                b + static_cast<size_t>(i) * row_floats + offset,
                num_components);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr<SL>(
                b + static_cast<size_t>(j) * row_floats,
                b + static_cast<size_t>(i) * row_floats,
                d);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_L2sqr_batch_4<SL>(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

    void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products += 4;
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0_ = 0, dp1_ = 0, dp2_ = 0, dp3_ = 0;
        fvec_inner_product_batch_4<SL>(
                q + offset,
                y0 + offset,
                y1 + offset,
                y2 + offset,
                y3 + offset,
                num_components,
                dp0_,
                dp1_,
                dp2_,
                dp3_);
        dp0 = dp0_;
        dp1 = dp1_;
        dp2 = dp2_;
        dp3 = dp3_;
    }
};

} // namespace

FlatCodesDistanceComputer*
IndexFlatPanoramaInline::get_FlatCodesDistanceComputer() const {
    FlatCodesDistanceComputer* dc = nullptr;
    with_simd_level([&]<SIMDLevel SL>() {
        dc = new FlatPanoramaInlineL2Dis<SL>(*this);
    });
    return dc;
}

} // namespace faiss
