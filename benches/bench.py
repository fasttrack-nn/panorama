#!/usr/bin/env python3
"""
Benchmark script for Panorama ANN experiments.

Reads a JSON config describing datasets and experiments, runs benchmarks,
caches ground truth in DuckDB, and appends results to a raw_data table.
"""

import argparse
import json
import yaml
import itertools
import os
import pickle
import resource
import struct
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import ctypes
import ctypes.util
import duckdb
import faiss
import numpy as np

# Force the flat-array `VisitedTable` (default switches to a
# std::unordered_set above 500K vectors for memory savings, but the
# hashset's per-pop malloc + insert costs roughly halve search QPS at
# paper scale). One-time global setting; affects every HNSW index built
# via this harness — vanilla, Pano, hnswlib, etc.
faiss.cvar.visited_table_hashset_threshold = 10**18

from transforms import (
    get_ivfpq_pano_transform,
    get_pca_linear_transform,
)

def _get_openblas_set_num_threads():
    """Return openblas_set_num_threads function, or None if unavailable."""
    for name in ("openblas", "openblas64", "blas"):
        path = ctypes.util.find_library(name)
        if path:
            try:
                lib = ctypes.CDLL(path)
                fn = lib.openblas_set_num_threads
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                return fn
            except (OSError, AttributeError):
                continue
    return None

_openblas_set_num_threads = _get_openblas_set_num_threads()


def set_threads(n: int) -> None:
    """Set thread count for faiss OMP, OpenBLAS, and MKL (env + runtime)."""
    faiss.omp_set_num_threads(n)
    if _openblas_set_num_threads:
        _openblas_set_num_threads(n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)

print("faiss compile options:", faiss.get_compile_options())

# Optional third-party indexes

def _find_repo_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return root
    except Exception:
        pass
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

_REPO_ROOT = _find_repo_root()

try:
    import pdxearch
    PDX_AVAILABLE = True
except ImportError:
    print("Warning: pdxearch not available.")
    PDX_AVAILABLE = False

_ADS_INDEX_BIN = os.path.join(_REPO_ROOT, "ADSampling", "build", "index_hnsw")
_ADS_SEARCH_BIN = os.path.join(_REPO_ROOT, "ADSampling", "build", "search_hnsw")
ADS_AVAILABLE = os.path.isfile(_ADS_INDEX_BIN) and os.path.isfile(_ADS_SEARCH_BIN)
if not ADS_AVAILABLE:
    print("Warning: ADSampling binaries not found. Run ADSampling/build.sh")

_ADS_IVF_INDEX_BIN = os.path.join(_REPO_ROOT, "ADSampling", "build", "index_ivf")
_ADS_IVF_SEARCH_BIN = os.path.join(_REPO_ROOT, "ADSampling", "build", "search_ivf")
ADS_IVF_AVAILABLE = (
    os.path.isfile(_ADS_IVF_INDEX_BIN) and os.path.isfile(_ADS_IVF_SEARCH_BIN)
)
if not ADS_IVF_AVAILABLE:
    print("Warning: ADSampling IVF binaries not found. Run ADSampling/build.sh")

# Vanilla upstream hnswlib bench driver. Build via
# benches/external_bins/build.sh (uses ADSampling's bundled hnswlib
# headers, which are clean upstream when adaptive=0 — the default).
_HNSWLIB_BIN = os.path.join(_REPO_ROOT, "benches", "external_bins", "bench_hnswlib")
HNSWLIB_AVAILABLE = os.path.isfile(_HNSWLIB_BIN)
if not HNSWLIB_AVAILABLE:
    print(
        "Warning: hnswlib bench driver not found. "
        "Run benches/external_bins/build.sh"
    )

# PKF / KS2 (Lu et al. 2025, arXiv 2505.20274). The official KS2 binary
# from the repo at /home/lutex/panorama/KS/KS2/l2.
_PKF_BIN = os.path.join(_REPO_ROOT, "KS", "KS2", "l2", "build", "KS2")
PKF_AVAILABLE = os.path.isfile(_PKF_BIN)
if not PKF_AVAILABLE:
    print("Warning: PKF/KS2 binary not found. cd KS/KS2/l2 && mkdir -p build && cd build && cmake .. && make -j")


def _pkf_default_L(d: int) -> int:
    """Pick the PKF level count L such that d/L is the divisor of d
    closest to 16 (the paper's empirically optimal d', see Sec 6.2:
    "When d' = d/L is around 16, HNSW+KS2 achieves the best search
    performance"). Falls back to L=1 if no divisor exists.
    """
    target = 16
    best_L, best_diff = 1, float("inf")
    for L in range(1, d + 1):
        if d % L != 0:
            continue
        diff = abs(d // L - target)
        if diff < best_diff:
            best_diff = diff
            best_L = L
    return best_L

# Per-dataset cache directories for hnswlib (saved index files) and PKF
# (KS2 index.bin + ProjInfo + KS2-format fvecs/ivecs). Each subdirectory
# is keyed by (dataset, scale, M_HNSW, efC, [L]) so different combos
# don't collide.
_HNSWLIB_CACHE_ROOT = os.path.join(_REPO_ROOT, "benches", "external_bins", "_hnswlib_cache")
_PKF_CACHE_ROOT = os.path.join(_REPO_ROOT, "benches", "external_bins", "_pkf_cache")

# A shared "external bench data" cache for the per-(dataset, scale, nq)
# fvecs/ivecs files that both hnswlib and PKF binaries need (standard
# .fvecs format, not our .fvec). Keyed by dataset name + slice size so
# multiple jobs share the conversion cost.
_EXTERNAL_DATA_ROOT = os.path.join(_REPO_ROOT, "benches", "external_bins", "_data_cache")


def _prepare_external_bench_data(
    ds: "DatasetState",
    k_gt: int = 100,
) -> Dict[str, str]:
    """Materialize KS2-format .fvecs/.ivecs files for the dataset slice.

    Both the hnswlib and PKF subprocess drivers expect the standard
    fvecs format (each row = int32 dim header + dim*float32 data) and
    standard ivecs (same shape, int32). We share the conversion across
    runs by caching under
        _data_cache/<dataset>__nb<nb>__nq<nq>__k<k_gt>/
    Returns a dict with keys: base, query, truth.
    """
    cache_dir = os.path.join(
        _EXTERNAL_DATA_ROOT,
        f"{ds.name}__nb{ds.nb}__nq{ds.nq}__k{k_gt}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    base_p = os.path.join(cache_dir, "base.fvecs")
    query_p = os.path.join(cache_dir, "query.fvecs")
    truth_p = os.path.join(cache_dir, "truth.ivecs")
    sentinel = os.path.join(cache_dir, ".ready")
    if os.path.exists(sentinel):
        return {"base": base_p, "query": query_p, "truth": truth_p}

    print(
        f"  [external] materializing fvecs cache for {ds.name} "
        f"(nb={ds.nb}, nq={ds.nq}, k={k_gt}) ..."
    )

    def write_fvecs(path: str, arr: np.ndarray) -> None:
        with open(path, "wb") as f:
            d = arr.shape[1]
            for v in arr:
                f.write(struct.pack("i", d))
                f.write(v.astype(np.float32).tobytes())

    def write_ivecs(path: str, arr: np.ndarray) -> None:
        with open(path, "wb") as f:
            d = arr.shape[1]
            for v in arr:
                f.write(struct.pack("i", d))
                f.write(v.astype(np.int32).tobytes())

    write_fvecs(base_p, ds.xb)
    write_fvecs(query_p, ds.xq)
    # KS2 always reads 100 ints per row (hardcoded maxk). If the GT
    # we have only stores up to k_gt < 100, pad with -1.
    gt = ds.I_gt
    if gt.shape[1] < 100:
        pad = np.full((gt.shape[0], 100 - gt.shape[1]), -1, dtype=np.int32)
        gt = np.concatenate([gt.astype(np.int32), pad], axis=1)
    elif gt.shape[1] > 100:
        gt = gt[:, :100]
    write_ivecs(truth_p, gt.astype(np.int32))
    open(sentinel, "w").write("ok\n")
    return {"base": base_p, "query": query_p, "truth": truth_p}

# Constants
US_IN_S = 1_000_000
NBITS = 8
NUM_SEARCH_ITERATIONS = 3

# Maps index_type -> (list of build param names, list of search param names)
INDEX_PARAM_SPEC: Dict[str, Tuple[List[str], List[str]]] = {
    "ivfpq":          (["nlist", "M"],                               ["nprobe", "n_threads"]),
    "ivfpq_pano":     (["nlist", "M", "nlevels", "alpha"],           ["nprobe", "epsilon", "n_threads"]),
    "ivf_flat":       (["nlist"],                                     ["nprobe", "n_threads"]),
    "ivf_flat_pano":  (["nlist", "nlevels", "batch_size"],              ["nprobe", "epsilon", "n_threads", "use_unoptimized_kernel", "disable_fixed_width"]),
    "hnsw":           (["M_HNSW", "efConstruction"],                  ["efSearch", "n_threads"]),
    "hnsw_pano":      (["M_HNSW", "efConstruction", "nlevels"],       ["efSearch", "epsilon", "n_threads"]),
    "pdx_ivf":        (["nlist"],                                     ["nprobe", "n_threads"]),
    # PDX-BOND-IVF: F32 IVF using BOND pruner (running heap-top threshold,
    # no random rotation). Restored from upstream PDX commit 4a2e65e and
    # reintegrated into the post-refactor PDX layout. Same param surface as
    # pdx_ivf so the two are scheduled apples-to-apples.
    "pdx_bond_ivf":   (["nlist"],                                     ["nprobe", "n_threads"]),
    "naive":          ([],                                            ["n_threads"]),
    "naive_pano":     (["nlevels"],                                    ["epsilon", "n_threads"]),
    "ads_hnsw":       (["M_HNSW", "efConstruction"],                ["efSearch", "n_threads"]),
    # Vanilla upstream hnswlib (Malkov & Yashunin reference impl). Run via
    # the external bench driver in benches/external_bins/bench_hnswlib so
    # we get the exact upstream search loop without Faiss's prune_headroom
    # graph-construction tweak. Apples-to-apples with `hnsw` (same M_HNSW,
    # efConstruction, efSearch, single-thread).
    "hnswlib":        (["M_HNSW", "efConstruction"],                ["efSearch", "n_threads"]),
    # PKF / KS2 (Lu et al. 2025, arXiv 2505.20274) — projection-based
    # routing test on top of hnswlib. Same graph as `hnswlib` but adds the
    # KS2 cheap test before each full L2. `L` controls dimension splitting
    # (paper-optimal: d/L ~ 16). Run via the official KS2 binary at
    # /home/lutex/panorama/KS/KS2/l2/build/KS2.
    "pkf":            (["M_HNSW", "efConstruction", "L"],            ["efSearch", "n_threads"]),
    # IVF++ from ADSampling (adaptive=1). epsilon0 is left at the C++
    # default (2.1). Same param surface as ivf_flat so they are scheduled
    # apples-to-apples.
    "ads_ivf":        (["nlist"],                                     ["nprobe", "n_threads"]),
    # 4-bit FastScan PQ via faiss IVF{nlist},PQ{M}x4fs as the base. M is the
    # number of subquantizers; per-vector size = M*4 bits. Compression vs.
    # float32 is d*32 / (M*4) = 8 * (d/M). With M = d/16 this is 128x.
    #
    # We never run the bare base because its quantization caps recall at
    # ~0.2 - it's only useful as a candidate generator for refine. Two
    # variants for apples-to-apples Panorama benchmarking:
    #   fastscan_refine        = base + IndexRefineFlat (full-precision refine)
    #   fastscan_refine_pano   = base + IndexRefinePanorama (Cauchy-Schwarz pruned)
    # Compare them at the same recall to measure the speedup attributable
    # to Panorama's level-pruning specifically.
    "fastscan_refine":      (["nlist", "M", "bbs"],                  ["nprobe", "k_factor", "n_threads"]),
    "fastscan_refine_pano": (["nlist", "M", "bbs", "nlevels"],       ["nprobe", "k_factor", "epsilon", "n_threads"]),
    # RaBitQ FastScan refine family. Mirrors the fastscan_refine family
    # but uses faiss IndexIVFRaBitQFastScan as the candidate generator
    # instead of PQ4FastScan. nb_bits = bits per dimension; nb_bits=1 is
    # the cheapest (32x compression vs. float32) and the only setting
    # that stays in the pure SIMD XOR-popcount kernel - higher nb_bits
    # triggers per-candidate multibit refinement that loses most of the
    # FastScan speed advantage. Recall is recovered downstream by the
    # refine wrapper (full-precision flat or panorama-pruned flat).
    "rabitq_refine":        (["nlist", "bbs", "nb_bits"],            ["nprobe", "k_factor", "n_threads"]),
    "rabitq_refine_pano":   (["nlist", "bbs", "nb_bits", "nlevels"], ["nprobe", "k_factor", "epsilon", "n_threads"]),
}

# Perf helpers

@dataclass
class PerfCounters:
    wall_time_s: float = 0.0
    user_time_s: float = 0.0
    system_time_s: float = 0.0

@contextmanager
def timed_execution() -> Generator[PerfCounters, None, None]:
    pcounters = PerfCounters()
    wall_start = time.perf_counter()
    ru_start = resource.getrusage(resource.RUSAGE_SELF)
    yield pcounters
    wall_end = time.perf_counter()
    ru_end = resource.getrusage(resource.RUSAGE_SELF)
    pcounters.wall_time_s = wall_end - wall_start
    pcounters.user_time_s = ru_end.ru_utime - ru_start.ru_utime
    pcounters.system_time_s = ru_end.ru_stime - ru_start.ru_stime

# fvec I/O

def read_fvecs(filename: str, max_vectors: Optional[int] = None) -> np.ndarray:
    """Read fvec/fvecs format file, optionally limited to max_vectors."""
    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Cannot read dimension from {filename}")
        dim = int.from_bytes(dim_bytes, byteorder="little")
        bytes_per_vec = 4 + dim * 4

        f.seek(0, 2)
        file_size = f.tell()
        total_vecs = file_size // bytes_per_vec

        n = total_vecs if max_vectors is None else min(total_vecs, max_vectors)
        f.seek(0)

        buf = np.empty((n, dim), dtype=np.float32)
        for i in range(n):
            _ = f.read(4)  # dimension prefix
            buf[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        return buf


def fvec_dimension_and_count(path: str) -> Tuple[int, int]:
    """Return (dimension, vector_count) without loading the whole file."""
    with open(path, "rb") as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            return 0, 0
        dim = int.from_bytes(dim_bytes, byteorder="little")
        f.seek(0, 2)
        return dim, f.tell() // (4 + dim * 4)


def write_fvecs(filename: str, data: np.ndarray):
    """Write float32 matrix to fvecs format."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(data[i].tobytes())


def write_ivecs(filename: str, data: np.ndarray):
    """Write integer matrix to ivecs format (used for ground truth)."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    n, d = data.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(data[i].tobytes())


def _prepare_adsampling_data(
    dataset_name: str,
    xb: np.ndarray,
    xq: np.ndarray,
    I_gt: np.ndarray,
    datasets_dir: str,
) -> str:
    """Prepare ADSampling data files and return the cache directory path.

    Creates: base fvecs, query fvecs, orthogonal matrix O.fvecs,
    transformed base vectors, and ground truth ivecs.
    All files are cached (skipped if already present).
    Directory is keyed by (dataset_name, nb) so different scale factors
    don't collide.
    """
    nb = xb.shape[0]
    nq = xq.shape[0]
    ads_dir = os.path.join(datasets_dir, dataset_name, f"adsampling_nb{nb}")
    os.makedirs(ads_dir, exist_ok=True)
    d = xb.shape[1]

    base_path = os.path.join(ads_dir, f"{dataset_name}_base.fvecs")
    if not os.path.exists(base_path):
        print("    [ads] Writing base vectors ...")
        write_fvecs(base_path, xb)

    query_path = os.path.join(ads_dir, f"{dataset_name}_query_nq{nq}.fvecs")
    if not os.path.exists(query_path):
        print("    [ads] Writing query vectors ...")
        write_fvecs(query_path, xq)

    ortho_path = os.path.join(ads_dir, "O.fvecs")
    trans_path = os.path.join(ads_dir, f"O{dataset_name}_base.fvecs")
    if not os.path.exists(ortho_path):
        print(f"    [ads] Generating orthogonal matrix (d={d}, seed=0) ...")
        rng = np.random.RandomState(0)
        G = rng.randn(d, d).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        Q = Q.astype(np.float32)
        write_fvecs(ortho_path, Q)
        xb_t = (xb @ Q).astype(np.float32)
        print("    [ads] Writing transformed base vectors ...")
        write_fvecs(trans_path, xb_t)

    gt_path = os.path.join(ads_dir, f"{dataset_name}_groundtruth_nq{nq}.ivecs")
    if not os.path.exists(gt_path):
        print("    [ads] Writing ground truth ivecs ...")
        write_ivecs(gt_path, I_gt.astype(np.int32))

    return ads_dir


def _prepare_adsampling_centroids(
    dataset_name: str,
    xb: np.ndarray,
    ads_dir: str,
    nlist: int,
) -> Tuple[str, str]:
    """Generate (and cache) ADSampling IVF centroid files for the given nlist.

    Returns (raw_centroid_path, transformed_centroid_path).
    Centroids are computed via faiss k-means on the UNTRANSFORMED base
    vectors, then the orthogonal matrix saved by ``_prepare_adsampling_data``
    is applied to produce the transformed centroids that ADSampling's
    ``index_ivf`` binary expects.
    """
    raw_path = os.path.join(ads_dir, f"{dataset_name}_centroid_{nlist}.fvecs")
    trans_path = os.path.join(ads_dir, f"O{dataset_name}_centroid_{nlist}.fvecs")
    if os.path.exists(raw_path) and os.path.exists(trans_path):
        return raw_path, trans_path

    print(f"    [ads] Clustering nlist={nlist} centroids on raw vectors ...")
    d = xb.shape[1]
    kmeans = faiss.Kmeans(d, nlist, niter=20, verbose=False, seed=0)
    kmeans.train(np.ascontiguousarray(xb, dtype=np.float32))
    centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))

    # Apply the orthogonal rotation cached by _prepare_adsampling_data
    ortho_path = os.path.join(ads_dir, "O.fvecs")
    if not os.path.exists(ortho_path):
        raise RuntimeError(
            f"orthogonal matrix not found at {ortho_path}; "
            "ensure _prepare_adsampling_data ran first"
        )
    Q = read_fvecs(ortho_path)
    centroids_t = np.ascontiguousarray((centroids @ Q).astype(np.float32))

    write_fvecs(raw_path, centroids)
    write_fvecs(trans_path, centroids_t)
    print(f"    [ads] Wrote centroid files for nlist={nlist}")
    return raw_path, trans_path


def resolve_dataset_paths(
    name: str,
    original_dir: str,
) -> Dict[str, Optional[str]]:
    """Locate original base/query files for a dataset.

    Returns dict with keys 'orig_base', 'orig_query'.
    Values are None when the file doesn't exist.
    """
    paths: Dict[str, Optional[str]] = {}
    for key, (prefix, suffixes) in {
        "orig_base":  (f"{name}_base",  [".fvec", ".fvecs"]),
        "orig_query": (f"{name}_query", [".fvec", ".fvecs"]),
    }.items():
        found = None
        if os.path.isdir(original_dir):
            for fname in os.listdir(original_dir):
                if fname.startswith(prefix) and any(fname.endswith(s) for s in suffixes):
                    found = os.path.join(original_dir, fname)
                    break
        paths[key] = found
    return paths


# Ground truth (DuckDB-cached)

def _gt_table_ensure(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth (
            dataset_name  VARCHAR,
            nb            BIGINT,
            nq            BIGINT,
            k             INTEGER,
            query_path    VARCHAR,
            D_blob        BLOB,
            I_blob        BLOB,
            PRIMARY KEY (dataset_name, nb, nq, k, query_path)
        )
    """)


def get_or_compute_ground_truth(
    db_path: str,
    dataset_name: str,
    xb: np.ndarray,
    xq: np.ndarray,
    k: int,
    query_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    nb = xb.shape[0]
    nq = xq.shape[0]

    # Read (also needs retry -- a concurrent writer blocks readers)
    cached_row = [None]
    def _read(con):
        _gt_table_ensure(con)
        cached_row[0] = con.execute(
            "SELECT D_blob, I_blob FROM ground_truth "
            "WHERE dataset_name=? AND nb=? AND nq=? AND k=? AND query_path=?",
            [dataset_name, nb, nq, k, query_path],
        ).fetchone()
    _db_write(db_path, _read)

    if cached_row[0] is not None:
        D_gt = pickle.loads(cached_row[0][0])
        I_gt = pickle.loads(cached_row[0][1])
        print(f"  Loaded cached ground truth ({nb} db, {nq} queries, k={k})")
        return D_gt, I_gt

    print(f"  Computing ground truth (brute-force, {nb} db, {nq} queries, k={k}) ...")
    set_threads(os.cpu_count() or 1)
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)
    D_gt, I_gt = flat.search(xq, k)

    D_blob = pickle.dumps(D_gt)
    I_blob = pickle.dumps(I_gt)
    def _write(con):
        _gt_table_ensure(con)
        con.execute(
            "INSERT INTO ground_truth VALUES (?, ?, ?, ?, ?, ?, ?)",
            [dataset_name, nb, nq, k, query_path, D_blob, I_blob],
        )
    _db_write(db_path, _write)
    print("  Ground truth cached.")
    return D_gt, I_gt


# DuckDB helpers

_RAW_DATA_DDL = """
CREATE TABLE IF NOT EXISTS raw_data (
    index_type           VARCHAR,
    file_path            VARCHAR,
    dataset_name         VARCHAR,
    nb                   BIGINT,
    nq                   BIGINT,
    nlevels              INTEGER,
    qps_mean             DOUBLE,
    qps_std              DOUBLE,
    search_mean_ms       DOUBLE,
    search_std_ms        DOUBLE,
    verification_mean_ms DOUBLE,
    verification_std_ms  DOUBLE,
    recall               DOUBLE,
    avg_level_percent    DOUBLE,
    M                    INTEGER,
    nlist                INTEGER,
    nprobe               INTEGER,
    n_trees              INTEGER,
    search_k             INTEGER,
    k                    INTEGER,
    M_HNSW               INTEGER,
    efSearch              INTEGER,
    batch_size           INTEGER,
    alpha                DOUBLE,
    scale_factor         DOUBLE,
    experiment_id        VARCHAR,
    run_id               VARCHAR,
    created_at           VARCHAR,
    valid                BOOLEAN,
    note                 VARCHAR,
    epsilon              DOUBLE,
    n_threads            INTEGER,
    k_factor             DOUBLE,
    bbs                  INTEGER,
    nb_bits              INTEGER
)
"""


# Additive migrations applied on every connection. Use IF NOT EXISTS only
# (DuckDB supports it on ADD COLUMN since v0.9). Never drop or rename columns.
_RAW_DATA_MIGRATIONS = (
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS n_threads INTEGER",
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS k_factor DOUBLE",
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS bbs INTEGER",
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS nb_bits INTEGER",
    # Systems-ablation columns for ivf_flat_pano:
    #   pca_transform           -- build-time: was the leading PCA applied?
    #   use_unoptimized_kernel  -- search-time: unoptimized fused-loop kernel?
    #   disable_fixed_width     -- search-time: skip with_level_width dispatch?
    # Defaults to NULL on existing rows so older queries keep working.
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS pca_transform BOOLEAN",
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS use_unoptimized_kernel BOOLEAN",
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS disable_fixed_width BOOLEAN",
    # L (PKF/KS2 dimension chunk count). Stored as integer; default
    # NULL for non-PKF rows.
    "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS L INTEGER",
)


def _raw_data_ensure(con: duckdb.DuckDBPyConnection):
    con.execute(_RAW_DATA_DDL)
    for stmt in _RAW_DATA_MIGRATIONS:
        con.execute(stmt)


def _db_write(db_path: str, fn, max_retries: int = 30, base_delay: float = 0.5):
    """Open a connection, run fn(con), close. Retries on lock conflicts."""
    for attempt in range(max_retries):
        try:
            con = duckdb.connect(db_path)
            try:
                fn(con)
            finally:
                con.close()
            return
        except duckdb.IOException:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** min(attempt, 4))
            print(f"    [db] lock busy, retrying in {delay:.1f}s ...")
            time.sleep(delay)


def insert_result(db_path: str, row: Dict[str, Any]):
    def _do(con):
        _raw_data_ensure(con)
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(f'"{c}"' for c in cols)
        con.execute(
            f"INSERT INTO raw_data ({col_names}) VALUES ({placeholders})",
            list(row.values()),
        )
    _db_write(db_path, _do)


# Index build + search

def build_index(
    index_type: str,
    xb: np.ndarray,
    d: int,
    build_params: Dict[str, Any],
    dataset_name: str,
    nb: int,
) -> Tuple[Any, Optional[Any]]:
    """Build and return (index, pca_model_or_None).

    The second return value is currently always None and is preserved
    only to keep the existing call signature.
    """
    nlevels = build_params.get("nlevels", 1)
    nlist = build_params.get("nlist", 100)
    M = build_params.get("M", 8)
    M_HNSW = build_params.get("M_HNSW", 32)
    efConstruction = build_params.get("efConstruction", 40)
    alpha = build_params.get("alpha", 8)

    ncpu = os.cpu_count() or 1
    set_threads(ncpu)

    if index_type == "ivfpq":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, NBITS)
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "ivfpq_pano":
        batch_size = build_params.get("batch_size", 128)
        quantizer = faiss.IndexFlatL2(d)
        sub = faiss.IndexIVFPQPanorama(quantizer, d, nlist, M, NBITS, nlevels, batch_size)
        # `pca_transform: false` skips the entire PCA + energy-spill +
        # level-equalization pipeline so Panorama operates on the raw
        # axes. Useful as an ablation against the standard pipeline that
        # depends on PCA to put energy on the leading dimensions.
        if build_params.get("pca_transform", True):
            lt = get_ivfpq_pano_transform(xb, dataset_name, nb, nlevels, alpha)
            index = faiss.IndexPreTransform(lt, sub)
        else:
            index = sub
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "ivf_flat":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "ivf_flat_pano":
        batch_size = build_params.get("batch_size", 128)
        sub = faiss.index_factory(d, f"IVF{nlist},FlatPanorama{nlevels}_{batch_size}")
        if build_params.get("pca_transform", True):
            lt = get_pca_linear_transform(xb, dataset_name, nb)
            index = faiss.IndexPreTransform(lt, sub)
        else:
            index = sub
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "hnsw":
        index = faiss.index_factory(d, f"HNSW{M_HNSW},Flat")
        hnsw_sub = faiss.downcast_index(index)
        hnsw_sub.hnsw.efConstruction = efConstruction
        hnsw_sub.hnsw.search_bounded_queue = True
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "hnsw_pano":
        sub = faiss.index_factory(d, f"HNSW{M_HNSW},FlatPanorama{nlevels}")
        hnsw_sub = faiss.downcast_index(sub)
        hnsw_sub.hnsw.efConstruction = efConstruction
        hnsw_sub.hnsw.search_bounded_queue = True
        if build_params.get("pca_transform", True):
            lt = get_pca_linear_transform(xb, dataset_name, nb)
            index = faiss.IndexPreTransform(lt, sub)
        else:
            index = sub
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "pdx_ivf":
        if not PDX_AVAILABLE:
            raise RuntimeError("pdxearch not installed")
        index = pdxearch.IndexPDXIVF(
            num_dimensions=d, num_clusters=nlist, normalize=False, n_threads=ncpu,
        )
        index.build(xb)
        return index, None

    if index_type == "pdx_bond_ivf":
        if not PDX_AVAILABLE:
            raise RuntimeError("pdxearch not installed")
        if not hasattr(pdxearch, "IndexPDXBONDIVF"):
            raise RuntimeError(
                "pdxearch is installed but does not expose IndexPDXBONDIVF; "
                "rebuild PDX (panorama/PDX/build.sh) to pick up the BOND pruner."
            )
        index = pdxearch.IndexPDXBONDIVF(
            num_dimensions=d, num_clusters=nlist, normalize=False, n_threads=ncpu,
        )
        index.build(xb)
        return index, None

    if index_type == "naive":
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        return index, None

    if index_type == "naive_pano":
        batch_size = build_params.get("batch_size", 512)
        sub = faiss.index_factory(d, f"FlatL2Panorama{nlevels}_{batch_size}")
        if build_params.get("pca_transform", True):
            lt = get_pca_linear_transform(xb, dataset_name, nb)
            index = faiss.IndexPreTransform(lt, sub)
        else:
            index = sub
        index.train(xb)
        index.add(xb)
        return index, None

    if index_type == "hnswlib":
        if not HNSWLIB_AVAILABLE:
            raise RuntimeError(
                "hnswlib bench driver not built. "
                "Run benches/external_bins/build.sh"
            )
        ext_dir = build_params.get("_external_data_dir")
        if ext_dir is None:
            raise RuntimeError(
                "hnswlib needs _external_data_dir; ensure load_dataset_state "
                "was called with needs_external_fvecs=True"
            )
        cache_dir = os.path.join(
            _HNSWLIB_CACHE_ROOT,
            f"{dataset_name}__nb{nb}__M{M_HNSW}__efC{efConstruction}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        index_file = os.path.join(cache_dir, "hnsw.bin")
        base_p = os.path.join(ext_dir, "base.fvecs")
        query_p = os.path.join(ext_dir, "query.fvecs")
        truth_p = os.path.join(ext_dir, "truth.ivecs")
        # Eager build with multi-threading. Subsequent search_index
        # calls invoke the same binary with --index pointing to the
        # cached hnsw.bin, single-threaded, just for the search phase.
        if not os.path.exists(index_file):
            print(
                f"    [hnswlib] BUILD pass M={M_HNSW} efC={efConstruction} "
                f"(multi-threaded) ..."
            )
            t0 = time.time()
            res = subprocess.run(
                [
                    _HNSWLIB_BIN, "--index", index_file,
                    base_p, query_p, truth_p,
                    str(nb), str(1000), str(d), str(10),
                    str(efConstruction), str(M_HNSW),
                    "16",  # placeholder ef; we ignore the search line
                ],
                env=dict(os.environ, OMP_NUM_THREADS=str(os.cpu_count() or 1)),
                capture_output=True, text=True,
            )
            if not os.path.exists(index_file):
                raise RuntimeError(
                    f"hnswlib build failed (no hnsw.bin produced): "
                    f"stdout={res.stdout[-500:]!r} stderr={res.stderr[-500:]!r}"
                )
            print(f"    [hnswlib] BUILD done in {time.time() - t0:.1f}s")
        handle = {
            "type": "hnswlib",
            "index_file": index_file,
            "base_path": base_p,
            "query_path": query_p,
            "truth_path": truth_p,
            "dataset_name": dataset_name,
            "nb": nb,
            "d": d,
            "M_HNSW": M_HNSW,
            "efC": efConstruction,
        }
        return handle, None

    if index_type == "pkf":
        if not PKF_AVAILABLE:
            raise RuntimeError(
                "PKF/KS2 binary not built. "
                "cd KS/KS2/l2 && mkdir -p build && cd build && cmake .. && make -j"
            )
        ext_dir = build_params.get("_external_data_dir")
        if ext_dir is None:
            raise RuntimeError(
                "pkf needs _external_data_dir; ensure load_dataset_state "
                "was called with needs_external_fvecs=True"
            )
        L = int(build_params.get("L", _pkf_default_L(d)))
        cwd = os.path.join(
            _PKF_CACHE_ROOT,
            f"{dataset_name}__nb{nb}__M{M_HNSW}__efC{efConstruction}__L{L}",
        )
        os.makedirs(cwd, exist_ok=True)
        # KS2 builds index.bin + ProjInfo lazily on first invocation in
        # cwd. We pre-build them here so search_index can be called with
        # any efSearch and run only the search phase.
        index_bin = os.path.join(cwd, "index.bin")
        if not os.path.exists(index_bin):
            # First invocation in this cwd: PKF builds index.bin +
            # ProjInfo and exit(0)'s without reaching the search loop.
            # We still pass an efSearch (16) as a placeholder for the
            # CLI parser; it's ignored on the build path.
            cmd = [
                _PKF_BIN,
                os.path.join(ext_dir, "base.fvecs"),
                os.path.join(ext_dir, "query.fvecs"),
                os.path.join(ext_dir, "truth.ivecs"),
                str(nb), str(1000), str(d), str(10),
                str(efConstruction), str(M_HNSW), str(L),
                "16",
            ]
            print(
                f"    [pkf] BUILD pass M={M_HNSW} efC={efConstruction} L={L} "
                f"d'={d // L} ..."
            )
            t0 = time.time()
            res = subprocess.run(
                cmd, cwd=cwd,
                env=dict(os.environ, OMP_NUM_THREADS=str(os.cpu_count() or 1)),
                capture_output=True, text=True,
            )
            if not os.path.exists(index_bin):
                raise RuntimeError(
                    f"PKF build failed (no index.bin produced): "
                    f"stdout={res.stdout[-500:]!r} stderr={res.stderr[-500:]!r}"
                )
            print(f"    [pkf] BUILD done in {time.time() - t0:.1f}s")
        handle = {
            "type": "pkf",
            "cwd": cwd,
            "base_path": os.path.join(ext_dir, "base.fvecs"),
            "query_path": os.path.join(ext_dir, "query.fvecs"),
            "truth_path": os.path.join(ext_dir, "truth.ivecs"),
            "dataset_name": dataset_name,
            "nb": nb,
            "d": d,
            "M_HNSW": M_HNSW,
            "efC": efConstruction,
            "L": L,
        }
        return handle, None

    if index_type == "ads_hnsw":
        if not ADS_AVAILABLE:
            raise RuntimeError("ADSampling binaries not built. Run ADSampling/build.sh")
        ads_dir = build_params["_ads_dir"]
        data_file = os.path.join(ads_dir, f"O{dataset_name}_base.fvecs")

        index_file = os.path.join(
            ads_dir,
            f"O{dataset_name}_ef{efConstruction}_M{M_HNSW}.index",
        )
        if not os.path.exists(index_file):
            cmd = [
                _ADS_INDEX_BIN,
                "-d", data_file,
                "-i", index_file,
                "-e", str(efConstruction),
                "-m", str(M_HNSW),
            ]
            print(f"    [ads] Building HNSW++ index: M={M_HNSW} efC={efConstruction}")
            subprocess.run(cmd, check=True)
        else:
            print(f"    [ads] Reusing cached index: {os.path.basename(index_file)}")

        handle = {
            "type": "ads_hnsw",
            "index_path": index_file,
            "ads_dir": ads_dir,
            "dataset_name": dataset_name,
        }
        return handle, None

    if index_type == "ads_ivf":
        if not ADS_IVF_AVAILABLE:
            raise RuntimeError(
                "ADSampling IVF binaries not built. Run ADSampling/build.sh"
            )
        ads_dir = build_params["_ads_dir"]
        # Build (or reuse) the per-nlist centroid files.
        _, trans_centroid_path = _prepare_adsampling_centroids(
            dataset_name, xb, ads_dir, nlist
        )
        data_file = os.path.join(ads_dir, f"O{dataset_name}_base.fvecs")
        index_file = os.path.join(
            ads_dir, f"O{dataset_name}_ivf_{nlist}_pp.index"
        )
        if not os.path.exists(index_file):
            cmd = [
                _ADS_IVF_INDEX_BIN,
                "-d", data_file,
                "-c", trans_centroid_path,
                "-i", index_file,
                "-a", "1",  # adaptive=1 -> IVF++ (cache-optimised)
            ]
            print(f"    [ads] Building IVF++ index: nlist={nlist}")
            subprocess.run(cmd, check=True)
        else:
            print(f"    [ads] Reusing cached IVF index: {os.path.basename(index_file)}")
        handle = {
            "type": "ads_ivf",
            "index_path": index_file,
            "ads_dir": ads_dir,
            "dataset_name": dataset_name,
        }
        return handle, None

    # ---- 4-bit FastScan refine family -------------------------------------
    # Both share the same base build (IVF{nlist},PQ{M}x4fs):
    #   fastscan_refine       = base + IndexRefineFlat       (baseline)
    #   fastscan_refine_pano  = base + IndexRefinePanorama   (with pruning)
    if index_type in ("fastscan_refine", "fastscan_refine_pano"):
        bbs = int(build_params.get("bbs", 32))
        base = faiss.index_factory(d, f"IVF{nlist},PQ{M}x4fs")
        if hasattr(base, "bbs"):
            base.bbs = bbs
        base.train(xb)
        base.add(xb)

        if index_type == "fastscan_refine":
            # IndexRefineFlat refines candidates with full-precision L2.
            # Pass xb via a contiguous swig pointer; faiss does NOT copy,
            # so we must hold a reference to the original buffer.
            xb_contig = np.ascontiguousarray(xb)
            wrapper = faiss.IndexRefineFlat(base, faiss.swig_ptr(xb_contig))
            handle = {
                "type": "fastscan_refine",
                "wrapper": wrapper,
                "base": base,
                "_xb_ref": xb_contig,
            }
            return handle, None

        # fastscan_refine_pano: build the FlatL2Panorama refine index too.
        # `pca_transform: false` drops the leading PCA so the refine head
        # operates on raw axes (matching the no-PCA pano ablation in the
        # other index families).
        if build_params.get("pca_transform", True):
            factory = f"PCA{d},FlatL2Panorama{nlevels}_1"
        else:
            factory = f"FlatL2Panorama{nlevels}_1"
        refine_pano = faiss.index_factory(d, factory)
        refine_pano.train(xb)
        refine_pano.add(xb)
        wrapper = faiss.IndexRefinePanorama(base, refine_pano)
        handle = {
            "type": "fastscan_refine_pano",
            "wrapper": wrapper,
            "base": base,
            "refine_pano": refine_pano,
        }
        return handle, None

    # ---- RaBitQ FastScan refine family ------------------------------------
    # Both share the same base build (IndexIVFRaBitQFastScan):
    #   rabitq_refine       = base + IndexRefineFlat       (baseline)
    #   rabitq_refine_pano  = base + IndexRefinePanorama   (with pruning)
    if index_type in ("rabitq_refine", "rabitq_refine_pano"):
        bbs = int(build_params.get("bbs", 32))
        nb_bits = int(build_params.get("nb_bits", 1))
        quantizer = faiss.IndexFlatL2(d)
        # Constructor signature (verified in faiss/IndexIVFRaBitQFastScan.h):
        #   (quantizer, d, nlist, metric, bbs, own_invlists, nb_bits)
        base = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2, bbs, True, nb_bits,
        )
        base.train(xb)
        base.add(xb)

        if index_type == "rabitq_refine":
            xb_contig = np.ascontiguousarray(xb)
            wrapper = faiss.IndexRefineFlat(base, faiss.swig_ptr(xb_contig))
            handle = {
                "type": "rabitq_refine",
                "wrapper": wrapper,
                "base": base,
                "_xb_ref": xb_contig,
            }
            return handle, None

        # rabitq_refine_pano: PCA + FlatL2Panorama refine head.
        # `pca_transform: false` drops the leading PCA, mirroring the
        # fastscan_refine_pano ablation.
        if build_params.get("pca_transform", True):
            factory = f"PCA{d},FlatL2Panorama{nlevels}_1"
        else:
            factory = f"FlatL2Panorama{nlevels}_1"
        refine_pano = faiss.index_factory(d, factory)
        refine_pano.train(xb)
        refine_pano.add(xb)
        wrapper = faiss.IndexRefinePanorama(base, refine_pano)
        handle = {
            "type": "rabitq_refine_pano",
            "wrapper": wrapper,
            "base": base,
            "refine_pano": refine_pano,
        }
        return handle, None

    raise ValueError(f"Unknown index type: {index_type}")


def search_index(
    index_type: str,
    index: Any,
    xq: np.ndarray,
    k: int,
    search_params: Dict[str, Any],
    build_params: Dict[str, Any],
    pca_model: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run search and return (D, I, timing_stats).

    timing_stats keys: qps_mean, qps_std, search_mean_ms, search_std_ms.
    (ads_hnsw also includes recall from the external binary.)
    """
    nq = xq.shape[0]
    nprobe = search_params.get("nprobe", 8)
    efSearch = search_params.get("efSearch", 64)
    epsilon = search_params.get("epsilon", 1.0)
    k_factor = float(search_params.get("k_factor", 1.0))
    n_threads = max(1, int(search_params.get("n_threads", 1)))

    if index_type == "ads_hnsw":
        handle = index
        ads_dir = handle["ads_dir"]
        ds = handle["dataset_name"]
        efSearch_val = search_params.get("efSearch", 64)

        query_path = os.path.join(ads_dir, f"{ds}_query_nq{nq}.fvecs")
        gt_path = os.path.join(ads_dir, f"{ds}_groundtruth_nq{nq}.ivecs")
        trans_path = os.path.join(ads_dir, "O.fvecs")
        result_path = os.path.join(ads_dir, "search_result.log")

        if os.path.exists(result_path):
            os.remove(result_path)

        cmd = [
            _ADS_SEARCH_BIN,
            "-d", "1",
            "-i", handle["index_path"],
            "-q", query_path,
            "-g", gt_path,
            "-t", trans_path,
            "-r", result_path,
            "-k", str(k),
            "-f", str(efSearch_val),
        ]
        # ADSampling search binary uses OpenMP; control parallelism via env.
        ads_env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ads_env)

        with open(result_path) as f:
            line = f.readline().strip()
        parts = line.split()
        recall_pct = float(parts[1])
        time_us_per_query = float(parts[2])

        total_time_s = time_us_per_query * nq * 1e-6
        qps = nq / total_time_s if total_time_s > 0 else 0

        stats = {
            "qps_mean": qps,
            "qps_std": 0.0,
            "search_mean_ms": total_time_s * 1000.0,
            "search_std_ms": 0.0,
            "recall": recall_pct / 100.0,
        }
        D_dummy = np.zeros((nq, k), dtype=np.float32)
        I_dummy = np.zeros((nq, k), dtype=np.int64)
        return D_dummy, I_dummy, stats

    if index_type == "ads_ivf":
        handle = index
        ads_dir = handle["ads_dir"]
        ds = handle["dataset_name"]

        query_path = os.path.join(ads_dir, f"{ds}_query_nq{nq}.fvecs")
        gt_path = os.path.join(ads_dir, f"{ds}_groundtruth_nq{nq}.ivecs")
        trans_path = os.path.join(ads_dir, "O.fvecs")
        result_path = os.path.join(ads_dir, "search_ivf_result.log")

        ads_env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        per_iter_qps: List[float] = []
        per_iter_time_ms: List[float] = []
        recall_pct = 0.0
        for _ in range(NUM_SEARCH_ITERATIONS):
            if os.path.exists(result_path):
                os.remove(result_path)
            cmd = [
                _ADS_IVF_SEARCH_BIN,
                "-d", "1",
                "-i", handle["index_path"],
                "-q", query_path,
                "-g", gt_path,
                "-t", trans_path,
                "-r", result_path,
                "-k", str(k),
                "-f", str(nprobe),
            ]
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=ads_env,
            )
            with open(result_path) as f:
                line = f.readline().strip()
            parts = line.split()
            recall_pct = float(parts[1])
            time_us_per_query = float(parts[2])
            total_time_s = time_us_per_query * nq * 1e-6
            per_iter_qps.append(nq / total_time_s if total_time_s > 0 else 0.0)
            per_iter_time_ms.append(total_time_s * 1000.0)

        qps_arr = np.array(per_iter_qps)
        time_arr = np.array(per_iter_time_ms)
        stats = {
            "qps_mean": float(np.mean(qps_arr)),
            "qps_std": float(np.std(qps_arr)),
            "search_mean_ms": float(np.mean(time_arr)),
            "search_std_ms": float(np.std(time_arr)),
            "recall": recall_pct / 100.0,
        }
        D_dummy = np.zeros((nq, k), dtype=np.float32)
        I_dummy = np.zeros((nq, k), dtype=np.int64)
        return D_dummy, I_dummy, stats

    if index_type == "hnswlib":
        handle = index
        efSearch_val = int(search_params.get("efSearch", 64))
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        cmd = [
            _HNSWLIB_BIN,
            "--index", handle["index_file"],
            handle["base_path"], handle["query_path"], handle["truth_path"],
            str(handle["nb"]), str(nq), str(handle["d"]), str(k),
            str(handle["efC"]), str(handle["M_HNSW"]),
            str(efSearch_val),
        ]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"hnswlib failed (rc={res.returncode}): "
                f"stdout={res.stdout[-500:]!r} stderr={res.stderr[-500:]!r}"
            )
        # Output: header line then "<ef>\t<recall>\t<qps>"
        recall_val, qps_val = None, None
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("ef\t"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    if int(parts[0]) == efSearch_val:
                        recall_val = float(parts[1])
                        qps_val = float(parts[2])
                except ValueError:
                    continue
        if recall_val is None:
            raise RuntimeError(
                f"hnswlib produced no result line for ef={efSearch_val}: "
                f"stdout={res.stdout!r}"
            )
        total_time_s = nq / qps_val if qps_val > 0 else 0.0
        stats = {
            "qps_mean": qps_val,
            "qps_std": 0.0,
            "search_mean_ms": total_time_s * 1000.0,
            "search_std_ms": 0.0,
            "recall": recall_val,
        }
        return (
            np.zeros((nq, k), dtype=np.float32),
            np.zeros((nq, k), dtype=np.int64),
            stats,
        )

    if index_type == "pkf":
        handle = index
        efSearch_val = int(search_params.get("efSearch", 64))
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        # The PKF binary loads the cached index.bin/ProjInfo from cwd
        # and accepts a list of efSearch values as trailing argv.
        cmd = [
            _PKF_BIN,
            handle["base_path"], handle["query_path"], handle["truth_path"],
            str(handle["nb"]), str(nq), str(handle["d"]), str(k),
            str(handle["efC"]), str(handle["M_HNSW"]), str(handle["L"]),
            str(efSearch_val),
        ]
        per_qps: List[float] = []
        recall_val = None
        for _ in range(NUM_SEARCH_ITERATIONS):
            res = subprocess.run(
                cmd, cwd=handle["cwd"], env=env,
                capture_output=True, text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"pkf failed (rc={res.returncode}): "
                    f"stdout={res.stdout[-500:]!r} stderr={res.stderr[-500:]!r}"
                )
            # Output: "<ef>\t<recall>\t<qps> QPS"
            for line in res.stdout.splitlines():
                s = line.strip().replace("QPS", "").strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    line_ef = int(parts[0])
                except ValueError:
                    continue
                if line_ef != efSearch_val:
                    continue
                recall_val = float(parts[1])
                per_qps.append(float(parts[2]))
                break
        if not per_qps or recall_val is None:
            raise RuntimeError(
                f"pkf produced no result line for ef={efSearch_val}: "
                f"stdout={res.stdout!r}"
            )
        qps_arr = np.array(per_qps)
        qps_mean = float(np.mean(qps_arr))
        total_time_s = nq / qps_mean if qps_mean > 0 else 0.0
        stats = {
            "qps_mean": qps_mean,
            "qps_std": float(np.std(qps_arr)),
            "search_mean_ms": total_time_s * 1000.0,
            "search_std_ms": 0.0,
            "recall": recall_val,
        }
        return (
            np.zeros((nq, k), dtype=np.float32),
            np.zeros((nq, k), dtype=np.int64),
            stats,
        )

    set_threads(n_threads)

    # ---- FastScan / RaBitQ refine family ---------------------------------
    # *_refine:       IndexRefineFlat wrapper, k_factor controlled.
    # *_refine_pano:  IndexRefinePanorama wrapper, k_factor + epsilon both
    #                 flow via IndexRefineSearchParameters (epsilon field
    #                 was added in our faiss patch).
    # The fastscan_* and rabitq_* variants share an identical wrapper API
    # (the only difference is the candidate-generator base index built in
    # build_index), so the search code path is the same.
    if index_type in ("fastscan_refine", "rabitq_refine"):
        handle = index
        wrapper = handle["wrapper"]
        base = handle["base"]
        base.nprobe = nprobe

        params = faiss.IndexRefineSearchParameters()
        params.k_factor = k_factor

        search_times: List[float] = []
        for _ in range(NUM_SEARCH_ITERATIONS):
            with timed_execution() as t:
                D, I = wrapper.search(xq, k, params=params)
            search_times.append(t.wall_time_s)
        search_arr = np.array(search_times)
        qps_per = [nq / st for st in search_times]
        return D, I, {
            "qps_mean": float(np.mean(qps_per)),
            "qps_std": float(np.std(qps_per)),
            "search_mean_ms": float(np.mean(search_arr)) * 1000.0,
            "search_std_ms": float(np.std(search_arr)) * 1000.0,
        }

    if index_type in ("fastscan_refine_pano", "rabitq_refine_pano"):
        handle = index
        wrapper = handle["wrapper"]
        base = handle["base"]
        base.nprobe = nprobe

        params = faiss.IndexRefineSearchParameters()
        params.k_factor = k_factor
        params.epsilon = epsilon
        faiss.cvar.indexPanorama_stats.reset()

        search_times = []
        for _ in range(NUM_SEARCH_ITERATIONS):
            with timed_execution() as t:
                D, I = wrapper.search(xq, k, params=params)
            search_times.append(t.wall_time_s)
        search_arr = np.array(search_times)
        qps_per = [nq / st for st in search_times]
        return D, I, {
            "qps_mean": float(np.mean(qps_per)),
            "qps_std": float(np.std(qps_per)),
            "search_mean_ms": float(np.mean(search_arr)) * 1000.0,
            "search_std_ms": float(np.std(search_arr)) * 1000.0,
            "avg_level_percent": (
                faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100.0
            ),
        }

    # Apply search-time params to IndexPreTransform-wrapped IVF indexes
    if index_type in ("ivfpq", "ivfpq_pano", "ivf_flat", "fs"):
        faiss.extract_index_ivf(index).nprobe = nprobe
    elif index_type == "ivf_flat_pano":
        faiss.extract_index_ivf(index).nprobe = nprobe
    elif index_type.startswith("hnsw"):
        # Navigate through IndexPreTransform to the HNSW sub-index
        inner = index.index if isinstance(index, faiss.IndexPreTransform) else index
        hnsw_sub = faiss.downcast_index(inner)
        hnsw_sub.hnsw.efSearch = efSearch
        hnsw_sub.hnsw.search_bounded_queue = True

    is_pano = "pano" in index_type
    if is_pano:
        # Set epsilon on the Panorama index for Cauchy-Schwarz bound scaling.
        if index_type in ("ivfpq_pano", "ivf_flat_pano"):
            faiss.downcast_index(faiss.extract_index_ivf(index)).set_epsilon(epsilon)
        elif index_type == "hnsw_pano":
            inner = index.index if isinstance(index, faiss.IndexPreTransform) else index
            faiss.downcast_index(inner).set_epsilon(epsilon)
        elif index_type == "naive_pano":
            faiss.downcast_index(index).set_epsilon(epsilon)

        # Systems-ablation toggles (only meaningful for ivf_flat_pano; the
        # setters live on IndexIVFFlatPanorama). Default false on every
        # other index type so they're effectively no-ops there.
        if index_type == "ivf_flat_pano":
            inner = faiss.downcast_index(faiss.extract_index_ivf(index))
            inner.set_use_unoptimized_kernel(
                bool(search_params.get("use_unoptimized_kernel", False))
            )
            inner.set_disable_fixed_width(
                bool(search_params.get("disable_fixed_width", False))
            )

        faiss.cvar.indexPanorama_stats.reset()

    search_times: List[float] = []

    if index_type == "pdx_ivf":
        D_final = np.full((nq, k), np.inf, dtype=np.float32)
        I_final = np.full((nq, k), -1, dtype=np.int64)

        with timed_execution() as transform_t:
            xq_t = index.transform_queries(xq)
        transform_overhead_s = transform_t.wall_time_s

        xq_prepared = [np.ascontiguousarray(xq_t[i]) for i in range(nq)]

        for _ in range(NUM_SEARCH_ITERATIONS):
            query_times: List[float] = []
            for i in range(nq):
                with timed_execution() as t:
                    index.search(
                        xq_prepared[i], k, nprobe=nprobe, is_query_transformed=True,
                    )
                query_times.append(t.wall_time_s)
            search_times.append(sum(query_times) + transform_overhead_s)

        index.reset_stats()
        for i in range(nq):
            ids, dists = index.search(
                xq_prepared[i], k, nprobe=nprobe, is_query_transformed=True,
            )
            n = min(len(ids), k)
            I_final[i, :n] = ids[:n]
            D_final[i, :n] = dists[:n]
        D, I = D_final, I_final

    elif index_type == "pdx_bond_ivf":
        # BOND has no rotation, so we hand the raw queries straight to the
        # searcher. Mirror the per-query timing loop used by pdx_ivf so the
        # two methods are timed apples-to-apples.
        D_final = np.full((nq, k), np.inf, dtype=np.float32)
        I_final = np.full((nq, k), -1, dtype=np.int64)
        xq_prepared = [np.ascontiguousarray(xq[i]) for i in range(nq)]

        for _ in range(NUM_SEARCH_ITERATIONS):
            query_times: List[float] = []
            for i in range(nq):
                with timed_execution() as t:
                    index.search(xq_prepared[i], k, nprobe=nprobe)
                query_times.append(t.wall_time_s)
            search_times.append(sum(query_times))

        index.reset_stats()
        for i in range(nq):
            ids, dists = index.search(xq_prepared[i], k, nprobe=nprobe)
            n = min(len(ids), k)
            I_final[i, :n] = ids[:n]
            D_final[i, :n] = dists[:n]
        D, I = D_final, I_final

    else:
        for _ in range(NUM_SEARCH_ITERATIONS):
            with timed_execution() as t:
                D, I = index.search(xq, k)
            search_times.append(t.wall_time_s)
    search_arr = np.array(search_times)
    mean_s = float(np.mean(search_arr))
    std_s = float(np.std(search_arr))
    qps_per = [nq / st for st in search_times]

    stats = {
        "qps_mean": float(np.mean(qps_per)),
        "qps_std": float(np.std(qps_per)),
        "search_mean_ms": mean_s * 1000.0,
        "search_std_ms": std_s * 1000.0,
    }
    if is_pano:
        stats["avg_level_percent"] = faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100.0
    elif index_type in ("pdx_ivf", "pdx_bond_ivf"):
        stats["avg_level_percent"] = index.get_ratio_dims_scanned() * 100.0
    return D, I, stats


# Main logic

def _param_grid(params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Cartesian product of parameter lists -> list of dicts."""
    if not params:
        return [{}]
    keys = list(params.keys())
    values = [params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


@dataclass
class DatasetState:
    """Loaded dataset + cached ground truth for one (dataset, scale) combo.

    Created once per dataset by ``load_dataset_state`` and then passed to
    repeated ``run_one_combo`` calls so we never re-load vectors or
    recompute GT. Used by both ``run_experiments`` (CLI) and the
    dashboard worker.
    """

    name: str
    dim: int
    nb: int
    nq: int
    k: int
    scale_factor: float
    xb: "np.ndarray"
    xq: "np.ndarray"
    I_gt: "np.ndarray"
    D_gt: "np.ndarray"
    base_path: str
    query_path: str
    ads_dir: Optional[str] = None
    # Path to the .fvecs/.ivecs cache used by `hnswlib` and `pkf`. Only
    # populated when the worker requests it.
    external_data_dir: Optional[str] = None


def load_dataset_state(
    *,
    dataset_name: str,
    db_path: str,
    datasets_dir: str,
    scale_factor: float,
    k: int,
    nq: int,
    needs_ads: bool = False,
    needs_external_fvecs: bool = False,
) -> Optional[DatasetState]:
    """Resolve paths, load vectors, compute/load GT for one dataset.

    Returns ``None`` if the dataset's base/query files are missing or
    the scaled size is zero.
    """
    paths = resolve_dataset_paths(dataset_name, datasets_dir)
    if paths["orig_base"] is None or paths["orig_query"] is None:
        return None

    dim, total_nb = fvec_dimension_and_count(paths["orig_base"])
    nb = int(total_nb * scale_factor)
    if nb == 0:
        return None

    xb = read_fvecs(paths["orig_base"], max_vectors=nb)
    xq = read_fvecs(paths["orig_query"], max_vectors=nq)
    D_gt, I_gt = get_or_compute_ground_truth(
        db_path, dataset_name, xb, xq, k, paths["orig_query"],
    )

    ads_dir = None
    if needs_ads:
        ads_dir = _prepare_adsampling_data(dataset_name, xb, xq, I_gt, datasets_dir)

    ds = DatasetState(
        name=dataset_name,
        dim=dim,
        nb=nb,
        nq=xq.shape[0],
        k=k,
        scale_factor=scale_factor,
        xb=xb,
        xq=xq,
        I_gt=I_gt,
        D_gt=D_gt,
        base_path=paths["orig_base"],
        query_path=paths["orig_query"],
        ads_dir=ads_dir,
    )

    if needs_external_fvecs:
        # Materialize KS2-format fvecs/ivecs for hnswlib & pkf. The dict
        # returned has keys: base, query, truth (all .fvecs/.ivecs paths).
        prep = _prepare_external_bench_data(ds)
        ds.external_data_dir = os.path.dirname(prep["base"])

    return ds


def build_index_for_combo(
    ds: DatasetState,
    index_type: str,
    build_params: Dict[str, Any],
) -> Tuple[Any, Optional[Any]]:
    """Build an index for one combo. Adds the ADS dir hint when needed."""
    bp = dict(build_params)
    if index_type in ("ads_hnsw", "ads_ivf") and ds.ads_dir is not None:
        bp["_ads_dir"] = ds.ads_dir
    if index_type in ("hnswlib", "pkf") and ds.external_data_dir is not None:
        bp["_external_data_dir"] = ds.external_data_dir
    return build_index(index_type, ds.xb, ds.dim, bp, ds.name, ds.nb)


def run_one_combo(
    ds: DatasetState,
    index_type: str,
    index: Any,
    pca_model: Optional[Any],
    build_params: Dict[str, Any],
    search_params: Dict[str, Any],
    *,
    db_path: str,
    run_id: str,
    note: Optional[str],
) -> Dict[str, Any]:
    """Run search for ONE (build_combo, search_combo), insert one
    raw_data row, and return the inserted row dict.

    Raises on search failure; caller decides how to handle the exception.
    """
    bp = dict(build_params)
    if index_type in ("ads_hnsw", "ads_ivf") and ds.ads_dir is not None:
        bp["_ads_dir"] = ds.ads_dir

    D, I, stats = search_index(
        index_type, index, ds.xq, ds.k, search_params, bp,
        pca_model=pca_model,
    )

    if "recall" in stats and stats["recall"] is not None:
        recall = stats["recall"]
    else:
        recall = float(np.mean([
            len(set(ds.I_gt[i].tolist()) & set(I[i].tolist())) / ds.k
            for i in range(len(ds.I_gt))
        ]))

    row = {
        "index_type": index_type,
        "file_path": ds.base_path,
        "dataset_name": ds.name,
        "nb": ds.nb,
        "nq": ds.xq.shape[0],
        "nlevels": bp.get("nlevels", -1),
        "qps_mean": stats["qps_mean"],
        "qps_std": stats["qps_std"],
        "search_mean_ms": stats["search_mean_ms"],
        "search_std_ms": stats["search_std_ms"],
        "verification_mean_ms": None,
        "verification_std_ms": None,
        "recall": recall,
        "avg_level_percent": stats.get("avg_level_percent"),
        "M": bp.get("M", -1),
        "nlist": bp.get("nlist", -1),
        "nprobe": search_params.get("nprobe", -1),
        "n_trees": bp.get("n_trees", -1),
        "search_k": search_params.get("search_k", -1),
        "k": ds.k,
        "M_HNSW": bp.get("M_HNSW", -1),
        "efSearch": search_params.get("efSearch", -1),
        "batch_size": bp.get("batch_size", -1),
        "alpha": bp.get("alpha", -1.0),
        "scale_factor": ds.scale_factor,
        "experiment_id": str(uuid.uuid4()),
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "valid": True,
        "note": note,
        "epsilon": search_params.get("epsilon"),
        "n_threads": int(search_params.get("n_threads", 1)),
        "k_factor": float(search_params.get("k_factor", -1.0)),
        "bbs": int(bp.get("bbs", -1)),
        "nb_bits": int(bp.get("nb_bits", -1)),
        "pca_transform": bool(bp.get("pca_transform", True)),
        "use_unoptimized_kernel": bool(
            search_params.get("use_unoptimized_kernel", False)
        ),
        "disable_fixed_width": bool(
            search_params.get("disable_fixed_width", False)
        ),
        "L": int(bp.get("L", -1)),
    }
    insert_result(db_path, row)
    return row


def run_experiments(
    config: Dict,
    db_path: str,
    run_id: str,
    original_dir: str,
    scale_factor: float,
    note: Optional[str] = None,
):
    datasets = config["datasets"]
    k = config.get("k", 10)
    nq_cfg = config.get("nq", 100)

    for ds_name, ds_cfg in datasets.items():
        experiments = ds_cfg["experiments"]
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        needs_ads = any(e["index_type"] in ("ads_hnsw", "ads_ivf") for e in experiments)
        needs_external_fvecs = any(
            e["index_type"] in ("hnswlib", "pkf") for e in experiments)
        ds = load_dataset_state(
            dataset_name=ds_name,
            db_path=db_path,
            datasets_dir=original_dir,
            scale_factor=scale_factor,
            k=k,
            nq=nq_cfg,
            needs_ads=needs_ads,
            needs_external_fvecs=needs_external_fvecs,
        )
        if ds is None:
            print(f"  SKIP: dataset {ds_name} unavailable or empty after scale_factor={scale_factor}")
            continue

        print(f"  dim={ds.dim}, nb={ds.nb}, nq={ds.nq}, k={ds.k}")

        for exp in experiments:
            idx_type = exp["index_type"]
            raw_build = exp.get("build_params", {})
            raw_search = exp.get("search_params", {})

            if idx_type not in INDEX_PARAM_SPEC:
                print(f"  SKIP unknown index_type: {idx_type}")
                continue

            build_combos = _param_grid(raw_build)
            search_combos = _param_grid(raw_search)

            for bp in build_combos:
                print(f"\n  [{idx_type}] build_params={bp}")
                with timed_execution() as build_t:
                    try:
                        index, pca_model = build_index_for_combo(ds, idx_type, bp)
                    except Exception as exc:
                        print(f"    BUILD FAILED: {exc}")
                        continue
                print(f"    Built in {build_t.wall_time_s * 1000.0:.0f} ms")

                for sp in search_combos:
                    print(f"    search_params={sp} ... ", end="", flush=True)
                    try:
                        row = run_one_combo(
                            ds, idx_type, index, pca_model, bp, sp,
                            db_path=db_path, run_id=run_id, note=note,
                        )
                    except Exception as exc:
                        print(f"SEARCH FAILED: {exc}")
                        continue
                    avg_lvl = row.get("avg_level_percent")
                    lvl_str = f"  dims%={avg_lvl:.1f}%" if avg_lvl is not None else ""
                    print(
                        f"recall={row['recall']:.4f}  "
                        f"qps={row['qps_mean']:.1f}{lvl_str}"
                    )

                del index


# CLI entry point

def main():
    parser = argparse.ArgumentParser(
        description="Unified Panorama ANN benchmark runner",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON experiment config file",
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--db", default=os.path.join(repo_root, "data", "benchmarks.duckdb"),
        help="Path to DuckDB database file (default: <repo>/data/benchmarks.duckdb)",
    )
    parser.add_argument(
        "--datasets-dir", default=os.path.join(repo_root, "data", "datasets"),
        help="Directory containing fvec/fvecs dataset files (default: <repo>/data/datasets)",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.0,
        help="Scale factor for database size in (0, 1]  (default: 1.0)",
    )
    parser.add_argument(
        "--note", type=str, required=True,
        help="Free-text note stored with every result row for this run",
    )
    args = parser.parse_args()

    if not 0 < args.scale_factor <= 1.0:
        parser.error("--scale-factor must be in (0, 1]")

    with open(args.config) as f:
        if args.config.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    print(f"Config: {args.config}")
    print(f"DB:     {args.db}")
    print(f"Scale:  {args.scale_factor}")
    if args.note:
        print(f"Note:   {args.note}")

    _db_write(args.db, _raw_data_ensure)

    run_experiments(
        config=config,
        db_path=args.db,
        run_id=run_id,
        original_dir=args.datasets_dir,
        scale_factor=args.scale_factor,
        note=args.note,
    )

    print(f"\nDone. Results written to {args.db} (run_id={run_id})")


if __name__ == "__main__":
    main()
