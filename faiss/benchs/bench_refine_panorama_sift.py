# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SIFT1M-focused micro-benchmark for IndexRefinePanorama.

Compares IndexRefinePanorama against IndexRefineFlat on SIFT1M with a
PQ base index. Single-thread numbers so we measure the pointwise
search_subset hot loop, not OpenMP parallelism.

Reports:
  - Recall@10 of both refine variants
  - QPS of both refine variants
  - Speedup (panorama / flat)
  - % of dimensions actually scanned
"""

import argparse
import multiprocessing as mp
import time

import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M


def eval_once(index, queries, gt, k, params=None, repeats=3):
    """Run search a few times and report best-of-N timing."""
    best_t = float("inf")
    D = I = None
    for _ in range(repeats):
        t0 = time.time()
        D, I = index.search(queries, k=k, params=params)
        t = time.time() - t0
        if t < best_t:
            best_t = t
    nq = queries.shape[0]
    speed_ms = best_t * 1000 / nq
    qps = 1000.0 / speed_ms
    corrects = (gt == I).sum()
    recall = corrects / (nq * k)
    return recall, qps, D, I


def build_indexes(d, xt, xb, factory_string, n_levels):
    base_index = faiss.index_factory(d, factory_string)
    faiss.omp_set_num_threads(mp.cpu_count())
    base_index.train(xt)
    base_index.add(xb)

    refine_pano_factory = f"PCA{d},FlatL2Panorama{n_levels}_1"
    refine_pano = faiss.index_factory(d, refine_pano_factory)
    refine_pano.train(xt)
    refine_pano.add(xb)

    idx_flat = faiss.IndexRefineFlat(base_index, faiss.swig_ptr(xb))
    idx_pano = faiss.IndexRefinePanorama(base_index, refine_pano)
    return base_index, idx_flat, idx_pano


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=1000,
                    help="number of queries (subset of 10k)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlevels", type=int, default=4,
                    help="Panorama levels (SIFT128 -> 4 is a good default)")
    ap.add_argument("--factory", type=str, default="IVF1024,PQ32x4fs")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    ds = DatasetSIFT1M()
    xq = ds.get_queries()[:args.nq]
    xb = ds.get_database()
    gt = ds.get_groundtruth()[:args.nq, :args.k]
    xt = ds.get_train()
    nb, d = xb.shape

    print(f"# Dataset SIFT1M  d={d}  nb={nb}  nq={args.nq}  k={args.k}")
    print(f"# base factory    = {args.factory}")
    print(f"# nlevels         = {args.nlevels}")
    print(f"# threads (search)= {args.threads}")
    print(f"# repeats per cfg = {args.repeats}")

    base_index, idx_flat, idx_pano = build_indexes(
        d, xt, xb, args.factory, args.nlevels)

    faiss.omp_set_num_threads(args.threads)

    nprobe_list = [4, 16, 64]
    kfactor_list = [1, 8, 64, 256, 1024]

    print()
    print("nprobe  k_factor   recall_flat   qps_flat   "
          "recall_pano   qps_pano   dims(%)   speedup(x)")
    for nprobe in nprobe_list:
        base_index.nprobe = nprobe
        for kf in kfactor_list:
            params = faiss.IndexRefineSearchParameters(k_factor=kf)
            recall_f, qps_f, _, _ = eval_once(
                idx_flat, xq, gt, args.k, params=params,
                repeats=args.repeats)
            faiss.cvar.indexPanorama_stats.reset()
            recall_p, qps_p, _, _ = eval_once(
                idx_pano, xq, gt, args.k, params=params,
                repeats=args.repeats)
            dims_pct = (
                faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100.0)
            speedup = qps_p / qps_f if qps_f > 0 else float("nan")
            print(f"{nprobe:6d}  {kf:7.1f}   "
                  f"{recall_f:11.6f}  {qps_f:9.2f}   "
                  f"{recall_p:11.6f}  {qps_p:9.2f}   "
                  f"{dims_pct:7.2f}   {speedup:9.3f}x")


if __name__ == "__main__":
    main()
