#!/usr/bin/env python3
"""Fairness audit: Panorama vs PDX benchmarking methodology."""
import numpy as np, time, os, faiss, pdxearch, sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from transforms import get_pca_linear_transform

d = 3072; nb = 50000; nq = 1000; k = 10; nlist = 10; ITERS = 5
np.random.seed(42)
xb = np.random.randn(nb, d).astype(np.float32)
xq = np.random.randn(nq, d).astype(np.float32)

print(f"=== FAIRNESS AUDIT: d={d}, nb={nb}, nq={nq}, k={k}, nlist={nlist} ===\n")

faiss.omp_set_num_threads(os.cpu_count())

q_flat = faiss.IndexFlatL2(d)
ivf_flat = faiss.IndexIVFFlat(q_flat, d, nlist)
ivf_flat.train(xb); ivf_flat.add(xb)

lt = get_pca_linear_transform(xb, "audit", nb)
sub_pano = faiss.index_factory(d, f"IVF{nlist},FlatPanorama32_128")
pano = faiss.IndexPreTransform(lt, sub_pano)
pano.train(xb); pano.add(xb)

pdx = pdxearch.IndexPDXIVF(num_dimensions=d, num_clusters=nlist, normalize=False)
pdx.build(xb)
print("All indexes built.\n")

print("── Threading ──")
faiss.omp_set_num_threads(1)
print(f"faiss OMP threads:      {faiss.omp_get_max_threads()}")
print(f"OPENBLAS_NUM_THREADS:   {os.environ.get('OPENBLAS_NUM_THREADS','unset')}")

# Transform cost
times = []
for _ in range(5):
    t0 = time.perf_counter(); pdx.transform_queries(xq); times.append(time.perf_counter()-t0)
print(f"PDX batch transform:    {np.median(times)*1000:.1f} ms\n")

def median_time(fn, iters=ITERS):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter()-t0)
    return np.median(times)

print("── Search timing (nprobe=5, median of 5 iters) ──")
faiss.omp_set_num_threads(1)

ivf_flat.nprobe = 5
t = median_time(lambda: ivf_flat.search(xq, k))
flat_batch = t*1000

def flat_perq():
    for i in range(nq):
        ivf_flat.search(xq[i:i+1], k)
t = median_time(flat_perq)
flat_pq = t*1000

faiss.extract_index_ivf(pano).nprobe = 5
t = median_time(lambda: pano.search(xq, k))
pano_batch = t*1000

def pano_perq():
    for i in range(nq):
        pano.search(xq[i:i+1], k)
t = median_time(pano_perq)
pano_pq = t*1000

def pdx_benchpy():
    xq_t = pdx.transform_queries(xq)
    for i in range(nq):
        q = np.ascontiguousarray(xq_t[i])
        pdx.search(q, k, nprobe=5, is_query_transformed=True)
t = median_time(pdx_benchpy)
pdx_new = t*1000

xq_t_cached = pdx.transform_queries(xq)
def pdx_search_only():
    for i in range(nq):
        q = np.ascontiguousarray(xq_t_cached[i])
        pdx.search(q, k, nprobe=5, is_query_transformed=True)
t = median_time(pdx_search_only)
pdx_so = t*1000

def pdx_old():
    for i in range(nq):
        q = np.ascontiguousarray(xq[i])
        pdx.search(q, k, nprobe=5)
t = median_time(pdx_old)
pdx_old_ms = t*1000

print(f"{'Method':<50} {'ms':>8} {'QPS':>8}")
print("-" * 68)
print(f"{'Faiss IVF flat (batched, no transform)':<50} {flat_batch:>8.1f} {nq/(flat_batch/1000):>8.0f}")
print(f"{'Faiss IVF flat (per-query loop)':<50} {flat_pq:>8.1f} {nq/(flat_pq/1000):>8.0f}")
print(f"{'Panorama (batched = bench.py method)':<50} {pano_batch:>8.1f} {nq/(pano_batch/1000):>8.0f}")
print(f"{'Panorama (per-query loop)':<50} {pano_pq:>8.1f} {nq/(pano_pq/1000):>8.0f}")
print(f"{'PDX (batch xform + per-q search = bench.py)':<50} {pdx_new:>8.1f} {nq/(pdx_new/1000):>8.0f}")
print(f"{'PDX (search-only, pre-transformed)':<50} {pdx_so:>8.1f} {nq/(pdx_so/1000):>8.0f}")
print(f"{'PDX (OLD: per-query xform inside search)':<50} {pdx_old_ms:>8.1f} {nq/(pdx_old_ms/1000):>8.0f}")

print()
print("── Breakdown ──")
pdx_xf = pdx_new - pdx_so
print(f"PDX batch transform overhead:          {pdx_xf:.1f} ms ({pdx_xf/pdx_new*100:.1f}% of total)")
print(f"PDX per-query search:                  {pdx_so:.1f} ms ({pdx_so/pdx_new*100:.1f}% of total)")
print(f"PDX OLD per-query transform overhead:  {pdx_old_ms - pdx_so:.1f} ms (eliminated by batching)")
dispatch = flat_pq - flat_batch
print(f"Python per-query dispatch overhead:    {dispatch:.1f} ms")

print()
print("── Fairness analysis ──")
print(f"Panorama bench.py: batched C++ (transform + search)  = {pano_batch:.1f} ms")
print(f"PDX bench.py:      batch transform + Python per-q    = {pdx_new:.1f} ms")
print()
print(f"If Panorama also ran per-query (same Python overhead): {pano_pq:.1f} ms")
print(f"  Panorama batched vs per-query speedup: {pano_pq/pano_batch:.2f}x")
print(f"  This advantage comes from batching all queries in one C++ call.")
print()
bias_pct = dispatch / pdx_new * 100
print(f"Estimated bias against PDX: ~{dispatch:.0f} ms ({bias_pct:.1f}% of PDX total)")
if bias_pct > 5:
    print(f"  Non-trivial, but reflects PDX's actual single-query API limitation.")
else:
    print(f"  Negligible — benchmark is fair.")
