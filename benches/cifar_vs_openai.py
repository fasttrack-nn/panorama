#!/usr/bin/env python3
"""Head-to-head: CIFAR vs OpenAI — why does CIFAR exceed theoretical while OpenAI doesn't?"""
import os, sys, time, numpy as np, faiss, mmap

sys.path.insert(0, os.path.dirname(__file__))
from transforms import get_pca_linear_transform

DATASETS_DIR = "/datasets/datasets"

def read_fvecs_fast(filename, max_vectors=None):
    """Memory-mapped fvecs reader — much faster than per-vector Python loop."""
    with open(filename, "rb") as f:
        dim = int.from_bytes(f.read(4), "little")
        bpv = 4 + dim * 4
        f.seek(0, 2)
        total = f.tell() // bpv
        n = total if max_vectors is None else min(total, max_vectors)
    raw = np.memmap(filename, dtype=np.uint8, mode='r', shape=(n, bpv))
    return np.frombuffer(raw[:, 4:].tobytes(), dtype=np.float32).reshape(n, dim).copy()

def find_file(name, kind):
    for d in [os.path.join(DATASETS_DIR, name), DATASETS_DIR]:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.startswith(f"{name}_{kind}") and f.endswith((".fvec", ".fvecs")):
                return os.path.join(d, f)
    return None

def median_ms(fn, iters=3):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

nq = 1000

configs = [
    ("cifar10",  50000,   10, 32, 1024),
    ("openai",  100000,  256, 32, 1024),
]

for ds_name, nb, nlist, nlevels, batch_size in configs:
    print(f"\n{'='*70}", flush=True)
    print(f"Loading {ds_name} (nb={nb})...", flush=True)
    t0 = time.perf_counter()
    xb = read_fvecs_fast(find_file(ds_name, "base"), nb)
    xq = read_fvecs_fast(find_file(ds_name, "query"), nq)
    d = xb.shape[1]
    vpc = nb // nlist
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s. d={d}, ~{vpc} vecs/cluster", flush=True)

    faiss.omp_set_num_threads(os.cpu_count())
    print("  Building indexes...", flush=True)
    t0 = time.perf_counter()
    lt = get_pca_linear_transform(xb, f"cvo2_{ds_name}", nb)
    sub = faiss.index_factory(d, f"IVF{nlist},FlatPanorama{nlevels}_{batch_size}")
    pano = faiss.IndexPreTransform(lt, sub)
    pano.train(xb); pano.add(xb)

    q_flat = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(q_flat, d, nlist)
    ivf.train(xb); ivf.add(xb)
    print(f"  Built in {time.perf_counter()-t0:.1f}s", flush=True)

    faiss.omp_set_num_threads(1)

    pca_ms = median_ms(lambda: lt.apply(xq))
    dims_per_level = d // nlevels
    level_kb = vpc * dims_per_level * 4 / 1024
    print(f"  PCA: {pca_ms:.1f}ms | dims/level: {dims_per_level} | level data/cluster: {level_kb:.0f}KB")

    nprobes = [1, 2, 5, 10] if ds_name == "cifar10" else [1, 2, 4, 8, 16, 32]

    print(f"\n  {'np':>4} {'vecs':>7} {'MB':>5} | {'flat':>8} {'pano':>8} {'pca':>6} {'srch':>8} | {'speed':>6} {'thry':>6} {'%th':>5} | {'dim%':>5}")
    print(f"  {'-'*85}")

    for nprobe in nprobes:
        if nprobe > nlist: continue
        ivf.nprobe = nprobe
        faiss.extract_index_ivf(pano).nprobe = nprobe
        n_vecs = nprobe * vpc
        mb = n_vecs * d * 4 / 1e6

        flat_ms = median_ms(lambda: ivf.search(xq, 10))

        faiss.cvar.indexPanorama_stats.reset()
        pano_ms = median_ms(lambda: pano.search(xq, 10))
        dim_pct = faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100

        actual = flat_ms / pano_ms
        theory = 1.0 / (dim_pct / 100.0) if dim_pct > 0 else 0
        pct = actual / theory * 100 if theory > 0 else 0
        srch = pano_ms - pca_ms

        print(f"  {nprobe:>4} {n_vecs:>7} {mb:>4.0f}M | {flat_ms:>7.1f}  {pano_ms:>7.1f}  {pca_ms:>5.1f}  {srch:>7.1f} | {actual:>5.1f}x {theory:>5.1f}x {pct:>4.0f}% | {dim_pct:>4.1f}%", flush=True)

    del pano, ivf, xb
