#!/usr/bin/env python3
"""
Benchmark query-transform time vs thread count for PDX and faiss PCA.

Measures wall-clock time of the matrix multiply (sgemm / DCT) applied to
queries only.  No full IVF search is performed.

Usage:
    python3 bench_transform_threads.py [--datasets-dir /datasets/datasets]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench import read_fvecs
from transforms import get_pca_linear_transform

import faiss
import pdxearch

DATASET_DIR_DEFAULT = "/datasets/datasets"
BASE_FILE = "openai_base.fvec"
QUERY_FILE = "openai_query.fvec"

NQ_VALUES = [10, 100, 1000]
THREAD_COUNTS = [1, 4, 8, 16, 32]
WARMUP_ITERS = 3
TIMED_ITERS = 20
NB_SAMPLE = 10_000
NLIST_PDX = 16
DIM = 3072


def _set_global_threads(n: int) -> None:
    faiss.omp_set_num_threads(n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)


def bench_faiss_pca(lt: faiss.LinearTransform, xq: np.ndarray, nq: int, n_threads: int) -> dict:
    _set_global_threads(n_threads)
    xq_slice = np.ascontiguousarray(xq[:nq], dtype=np.float32)

    for _ in range(WARMUP_ITERS):
        lt.apply(xq_slice)

    times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        lt.apply(xq_slice)
        times.append(time.perf_counter() - t0)

    arr = np.array(times) * 1000.0
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std())}


def bench_pdx_transform(xb_sample: np.ndarray, xq: np.ndarray, nq: int, n_threads: int) -> dict:
    _set_global_threads(n_threads)
    idx = pdxearch.IndexPDXIVF(
        num_dimensions=DIM, num_clusters=NLIST_PDX, normalize=False, n_threads=n_threads,
    )
    idx.build(xb_sample)

    xq_slice = np.ascontiguousarray(xq[:nq], dtype=np.float32)

    for _ in range(WARMUP_ITERS):
        idx.transform_queries(xq_slice)

    times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        idx.transform_queries(xq_slice)
        times.append(time.perf_counter() - t0)

    arr = np.array(times) * 1000.0
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default=DATASET_DIR_DEFAULT)
    args = parser.parse_args()

    base_path = os.path.join(args.datasets_dir, BASE_FILE)
    query_path = os.path.join(args.datasets_dir, QUERY_FILE)

    print(f"Loading {NB_SAMPLE} base vectors for PCA fitting ...")
    xb_sample = read_fvecs(base_path, max_vectors=NB_SAMPLE)
    print(f"Loading queries ...")
    xq = read_fvecs(query_path, max_vectors=max(NQ_VALUES))
    d = xq.shape[1]
    print(f"  d={d}, xb_sample={xb_sample.shape}, xq={xq.shape}")

    print("Fitting faiss PCA LinearTransform ...")
    lt = get_pca_linear_transform(xb_sample, "bench_threads", NB_SAMPLE)
    print("Done.\n")

    print(f"Warmup: {WARMUP_ITERS} iters, Timed: {TIMED_ITERS} iters")
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"nq values: {NQ_VALUES}\n")

    rows: list[dict] = []

    for nq in NQ_VALUES:
        for nt in THREAD_COUNTS:
            print(f"  faiss_pca  nq={nq:>5}  threads={nt:>3} ... ", end="", flush=True)
            r = bench_faiss_pca(lt, xq, nq, nt)
            r.update(index="faiss_pca", nq=nq, threads=nt)
            rows.append(r)
            print(f"{r['mean_ms']:>8.3f} ms  (std {r['std_ms']:.3f})")

    for nq in NQ_VALUES:
        for nt in THREAD_COUNTS:
            print(f"  pdx_ivf    nq={nq:>5}  threads={nt:>3} ... ", end="", flush=True)
            r = bench_pdx_transform(xb_sample, xq, nq, nt)
            r.update(index="pdx_ivf", nq=nq, threads=nt)
            rows.append(r)
            print(f"{r['mean_ms']:>8.3f} ms  (std {r['std_ms']:.3f})")

    print("\n" + "=" * 72)
    print(f"Dataset: openai (d={d})")
    print(f"Warmup: {WARMUP_ITERS}, Timed: {TIMED_ITERS}")
    print("=" * 72)
    hdr = f"{'index':>12}  {'nq':>5}  {'threads':>7}  {'mean_ms':>10}  {'std_ms':>8}  {'vs_1t':>8}"
    print(hdr)
    print("-" * len(hdr))

    baseline: dict[tuple[str, int], float] = {}
    for r in rows:
        key = (r["index"], r["nq"])
        if r["threads"] == 1:
            baseline[key] = r["mean_ms"]

    for r in rows:
        key = (r["index"], r["nq"])
        base = baseline.get(key, r["mean_ms"])
        speedup = base / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
        print(
            f"{r['index']:>12}  {r['nq']:>5}  {r['threads']:>7}  "
            f"{r['mean_ms']:>10.3f}  {r['std_ms']:>8.3f}  {speedup:>7.2f}x"
        )


if __name__ == "__main__":
    main()
