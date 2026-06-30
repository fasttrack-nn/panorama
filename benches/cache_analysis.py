#!/usr/bin/env python3
"""Measure PCA transform time vs search time to understand super-theoretical speedups."""
import os, sys, time, numpy as np, faiss

sys.path.insert(0, os.path.dirname(__file__))
from transforms import get_pca_linear_transform

DATASETS_DIR = "/datasets/datasets"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_fvecs(filename, max_vectors=None):
    with open(filename, "rb") as f:
        dim = int.from_bytes(f.read(4), "little")
        bpv = 4 + dim * 4
        f.seek(0, 2)
        total = f.tell() // bpv
        n = total if max_vectors is None else min(total, max_vectors)
        f.seek(0)
        buf = np.empty((n, dim), dtype=np.float32)
        for i in range(n):
            f.read(4)
            buf[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return buf

def find_file(ds_name, suffix):
    d = os.path.join(DATASETS_DIR, ds_name) if os.path.isdir(os.path.join(DATASETS_DIR, ds_name)) else DATASETS_DIR
    for f in os.listdir(d):
        if f.startswith(f"{ds_name}_{suffix}") and f.endswith((".fvec", ".fvecs")):
            return os.path.join(d, f)
    for f in os.listdir(DATASETS_DIR):
        if f.startswith(f"{ds_name}_{suffix}") and f.endswith((".fvec", ".fvecs")):
            return os.path.join(DATASETS_DIR, f)
    return None

def median_ms(fn, iters=5):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

configs = [
    ("cifar10",      50000,  10, 32, 1024, [1, 3, 5, 10]),
    ("fashionmnist", 60000, 256, 16, 1024, [2, 10, 32, 128]),
    ("sift10m",   10000000, 1024, 4, 1024, [2, 10, 32, 128]),
    ("gist1m",     1000000, 128, 16, 1024, [2, 10, 32, 128]),
]

nq = 1000

for ds_name, nb, nlist, nlevels, batch_size, nprobes in configs:
    base_path = find_file(ds_name, "base")
    query_path = find_file(ds_name, "query")
    if not base_path or not query_path:
        print(f"SKIP {ds_name}: files not found")
        continue

    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}  d=?, nb={nb}, nlist={nlist}, nlevels={nlevels}")
    print(f"{'='*70}")

    xb = read_fvecs(base_path, nb)
    xq = read_fvecs(query_path, nq)
    d = xb.shape[1]
    print(f"  d={d}, nb={xb.shape[0]}, nq={xq.shape[0]}")
    vecs_per_cluster = nb // nlist

    # Build indexes
    faiss.omp_set_num_threads(os.cpu_count())
    lt = get_pca_linear_transform(xb, f"cache_analysis_{ds_name}", nb)
    sub_pano = faiss.index_factory(d, f"IVF{nlist},FlatPanorama{nlevels}_{batch_size}")
    pano = faiss.IndexPreTransform(lt, sub_pano)
    pano.train(xb); pano.add(xb)

    q_flat = faiss.IndexFlatL2(d)
    ivf_flat = faiss.IndexIVFFlat(q_flat, d, nlist)
    ivf_flat.train(xb); ivf_flat.add(xb)

    faiss.omp_set_num_threads(1)

    # Measure PCA transform cost by timing pano search vs search without transform
    # Build a second pano index without the PreTransform wrapper to isolate search-only
    # Time just the PCA transform
    pca_ms = median_ms(lambda: lt.apply(xq))
    pca_per_query_ms = pca_ms / nq

    print(f"\n  PCA transform (1000 queries, d={d}): {pca_ms:.1f} ms total, {pca_per_query_ms:.3f} ms/query")

    print(f"\n  {'nprobe':>6} {'vecs':>7} | {'flat_ms':>8} {'pano_ms':>8} {'pca_ms':>7} | {'search_only':>11} {'flat_search':>11} | {'speedup':>7} {'theory':>7} {'%theo':>6} | {'cache_bonus':>11}")
    print(f"  {'-'*110}")

    for nprobe in nprobes:
        ivf_flat.nprobe = nprobe
        faiss.extract_index_ivf(pano).nprobe = nprobe
        n_vecs = nprobe * vecs_per_cluster

        flat_ms = median_ms(lambda: ivf_flat.search(xq, 10))

        faiss.cvar.indexPanorama_stats.reset()
        pano_ms_total = median_ms(lambda: pano.search(xq, 10))
        dim_pct = faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100

        pano_search_only = pano_ms_total - pca_ms
        flat_search = flat_ms

        actual_speedup = flat_ms / pano_ms_total
        theoretical = 1.0 / (dim_pct / 100.0) if dim_pct > 0 else float('inf')
        pct_theo = (actual_speedup / theoretical * 100) if theoretical > 0 else 0

        # What speedup would we get from dim reduction alone?
        # flat_search / (flat_search * dim%) = 1/dim% = theoretical
        # actual search-only speedup (excluding PCA):
        search_speedup = flat_search / pano_search_only if pano_search_only > 0 else float('inf')
        cache_bonus = search_speedup / theoretical * 100 if theoretical > 0 else 0

        print(f"  {nprobe:>6} {n_vecs:>7} | {flat_ms:>7.1f}  {pano_ms_total:>7.1f}  {pca_ms:>6.1f} | {pano_search_only:>10.1f}  {flat_search:>10.1f} | {actual_speedup:>6.2f}x {theoretical:>6.1f}x {pct_theo:>5.1f}% | {cache_bonus:>10.1f}%")

    print()
    print(f"  PCA overhead as % of pano total at max nprobe: {pca_ms / pano_ms_total * 100:.1f}%")
    print(f"  PCA overhead as % of pano total at min nprobe: ", end="")
    faiss.extract_index_ivf(pano).nprobe = nprobes[0]
    pano_min = median_ms(lambda: pano.search(xq, 10))
    print(f"{pca_ms / pano_min * 100:.1f}%")
