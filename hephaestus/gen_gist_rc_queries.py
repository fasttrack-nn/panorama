"""Generate synthetic GIST queries at three Relative-Contrast targets.

Targets (calibrated against the natural GIST query distribution, where
RC@10 has p10=1.47, median=1.62, p90=2.23, p99=5.07, hardest-natural=1.35):

    rc3p0   ->  RC = 3.00  (easy,    ~p95 of natural)
    rc1p7   ->  RC = 1.70  (medium,  ~p55 of natural)
    rc1p35  ->  RC = 1.35  (hard,    matches the hardest natural query)

Output: 3 .fvec files with 1000 queries x 960 dims each, written next
to the GIST base file as gist1m_query_rc{label}.fvec so panorama's
bench.resolve_dataset_paths can pick them up via a `dataset_name` of
`gist1m_rc{label}` (we leave it to a follow-up to wire that into
bench.py if desired).

Why we don't use hephaestus.HephaestusGradient verbatim:

  1) Speed: hephaestus's distance kernel is a JAX-jitted
     `||X - q||_2` that materialises a 1M x 960 float32 difference
     tensor every call -- ~1.7s per gradient step on this CPU. We
     replace it with the standard BLAS identity
         ||X - q||^2 = ||X||^2 + ||q||^2 - 2 (X @ q)
     which collapses to a single GEMV plus two reductions. Measured
     ~50ms per step on the same machine (~30x speedup). For 1000
     queries x 3 targets that's the difference between days and minutes.

  2) Convergence: starting from a random data point and chasing a
     fixed RC target is fragile. The gradient of RC vanishes near
     RC ~ 1 (the floor where every point is equidistant), so the
     optimiser easily falls into the floor plateau and never climbs
     out. We initialise smartly from a pre-scored pool: sample N base
     vectors once, score their natural RC in one tiled GEMM, then for
     each query pick the unused pool entry whose natural RC is
     closest to the target and run a short Adam refinement starting
     from there. The pool is shared across all RC targets and queries,
     so the dominant cost is ~12 minutes of (pool x base) GEMMs paid
     once, after which per-query smart-init is a free argmin.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
GIST_BASE_FVEC = Path("/datasets/datasets/gist1m_base.fvec")
DATASETS_DIR = Path("/datasets/datasets")

DATASET_NAME = "gist1m"

# (fvec-suffix-label, RC target). Suffix uses 'p' for the decimal so
# filenames stay shell-friendly: gist1m_query_rc1p35.fvec.
#
# All targets share one pre-scored candidate pool sampled from the base
# set, so per-target overrides for n_init / max_iter are unnecessary --
# refinement is the only per-query cost left.
RC_TARGETS = [
    # label,    target_rc
    ("rc3p0",   3.00),
    ("rc1p7",   1.70),
    ("rc1p35",  1.35),
]

K = 10
N_QUERIES_PER_TARGET = 1000
DELTA = 0.03            # +-3% acceptance band around target RC
LEARNING_RATE = 0.3     # Adam step; lower than hephaestus's default 1.0
                        # because our scorer is sharper and lr=1 overshoots

# Shared smart-init pool. We sample this many base vectors once,
# precompute their natural RC, and then per-query smart-init becomes a
# cheap argmin over the pool instead of a fresh GEMM. Sized so that
# even the hardest target (RC=1.35, ~1% prevalence) has comfortable
# headroom for 1000 unique queries.
POOL_SIZE = 200_000
POOL_TILE = 256          # rows per GEMM tile; (256, 1M) f32 = 1 GB

SEED = 1234


def read_fvecs(path: Path) -> np.ndarray:
    """Read standard {int32 dim, float32[dim]} per-vector format."""
    with open(path, "rb") as f:
        dim = int.from_bytes(f.read(4), "little")
        f.seek(0, 2)
        n = f.tell() // (4 + dim * 4)
        f.seek(0)
        out = np.empty((n, dim), dtype=np.float32)
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return out


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    dim_prefix = np.int32(d).tobytes()
    with open(path, "wb") as f:
        for row in arr:
            f.write(dim_prefix)
            f.write(row.tobytes())


# Fast RC kernel (NumPy + BLAS).

def _dist_sq(q: np.ndarray, X: np.ndarray, X_sq: np.ndarray) -> np.ndarray:
    """||X - q||^2 expanded to ||X||^2 + ||q||^2 - 2 X.q."""
    q_sq = float(np.dot(q, q))
    inner = X @ q  # BLAS GEMV: O(n*d) with high throughput
    return X_sq + q_sq - 2.0 * inner


def fast_rc(
    q: np.ndarray, X: np.ndarray, X_sq: np.ndarray, k: int,
) -> float:
    d = np.sqrt(np.maximum(_dist_sq(q, X, X_sq), 0.0))
    idx = np.argpartition(d, k)
    return float(d.mean() / d[idx[k]])


def fast_rc_and_grad(
    q: np.ndarray, X: np.ndarray, X_sq: np.ndarray, k: int,
) -> tuple[float, np.ndarray]:
    """RC and analytic gradient w.r.t. q.

    Derivation:
        d_i  = ||X_i - q||
        avg  = (1/n) sum_i d_i
        d_k  = the (k+1)-th smallest d_i  (i.e. the k-th nearest neighbour
               distance under 0-indexed argpartition)
        RC   = avg / d_k

        d/dq d_i      = (q - X_i) / d_i
        d/dq avg      = (1/n) sum_i (q - X_i)/d_i
                      = q * (1/n) * sum 1/d_i  -  (1/n) * X^T @ (1/d)
        d/dq d_k      = (q - X_{k_idx}) / d_k

        d/dq RC       = (g_avg * d_k - avg * g_kth) / d_k^2

    The numerically dangerous case is d_i ~= 0 (q coincides with a data
    point); we mask 1/d_i to 0 there which kills the singular term.
    """
    d_sq = _dist_sq(q, X, X_sq)
    d = np.sqrt(np.maximum(d_sq, 1e-12))
    idx = np.argpartition(d, k)
    k_idx = int(idx[k])
    d_k = float(d[k_idx])

    safe_d = np.where(d > 1e-6, d, 1.0)
    inv_d = (1.0 / safe_d).astype(np.float32)
    inv_d[d <= 1e-6] = 0.0

    n = X.shape[0]
    sum_inv = float(inv_d.sum()) / n
    weighted_X = (X.T @ inv_d) / n           # (dim,)  -- another BLAS GEMV
    g_avg = q * sum_inv - weighted_X

    if d_k > 1e-6:
        g_kth = (q - X[k_idx]) / d_k
    else:
        g_kth = np.zeros_like(q)

    avg = float(d.mean())
    rc = avg / d_k
    grad = (g_avg * d_k - avg * g_kth) / (d_k * d_k)
    return rc, grad.astype(np.float32)


# Smart init via a shared candidate pool.
#
# Key idea: smart-init for one query at the hard target needs to scan
# ~2k candidates (because <1% of base points sit naturally near RC=1.35).
# Doing that 1000 times is ~2M candidate scorings. Instead we score one
# big pool ONCE -- typically pool_size in [50k, 200k] -- as a single
# tiled (pool_size, dim) @ (dim, N) GEMM, store the resulting natural
# RCs, and then per-query smart-init becomes a cheap argmin over the
# precomputed RC array. This collapses ~2M scorings to ~pool_size
# scorings shared across all targets and all queries.
#
# The tile size is chosen so the (tile, N) distance matrix fits in
# memory: with tile=512 and N=1M float32 we use 2 GB per tile, so we
# walk the pool in tiles and accumulate the result.

def score_candidate_pool(
    *,
    X: np.ndarray,
    X_sq: np.ndarray,
    k: int,
    pool_idx: np.ndarray,
    tile_size: int = 512,
    log=None,
) -> np.ndarray:
    """Compute natural RC for every base vector in pool_idx.

    Returns an array `rcs` of shape (len(pool_idx),) in the same order
    as pool_idx. Uses the (k+1)-th smallest distance as the k-NN
    distance (skips the trivial self-match at distance 0).
    """
    n_pool = len(pool_idx)
    rcs = np.empty(n_pool, dtype=np.float32)
    n_tiles = (n_pool + tile_size - 1) // tile_size

    t_start = time.time()
    for ti in range(n_tiles):
        lo = ti * tile_size
        hi = min(lo + tile_size, n_pool)
        sl = pool_idx[lo:hi]
        cand = X[sl]                            # (tile, dim)
        cand_sq = X_sq[sl]                      # (tile,)

        # (tile, N) distance matrix in float32 -- the dominant cost.
        inner = cand @ X.T                      # one GEMM per tile
        d_sq = X_sq[None, :] + cand_sq[:, None] - 2.0 * inner
        np.maximum(d_sq, 0.0, out=d_sq)
        d = np.sqrt(d_sq, out=d_sq)

        avg = d.mean(axis=1)
        # k+1 smallest per row; the largest of those (k+1) is the
        # k-th non-self nearest distance.
        part = np.argpartition(d, k, axis=1)[:, :k + 1]
        rows = np.arange(hi - lo)[:, None]
        kth = d[rows, part].max(axis=1)

        # Pool members that have a near-duplicate in the dataset get
        # kth ~ 0 and an absurd RC. We only ever want pool entries
        # near a sane RC target, so clip the upper tail to a sentinel
        # so they're never picked as init candidates.
        DEGENERATE_KTH = 1e-3
        rc_tile = avg / np.maximum(kth, 1e-12)
        rc_tile[kth < DEGENERATE_KTH] = np.inf
        rcs[lo:hi] = rc_tile

        if log is not None and (ti + 1) % max(1, n_tiles // 10) == 0:
            elapsed = time.time() - t_start
            log.info(
                "  pool tile %4d/%d  (%5.0f%%)  elapsed=%5.1fs",
                ti + 1, n_tiles, 100 * (ti + 1) / n_tiles, elapsed,
            )

    return rcs


def smart_init_from_pool(
    *,
    X: np.ndarray,
    pool_idx: np.ndarray,
    pool_rcs: np.ndarray,
    target_rc: float,
    used_mask: np.ndarray,
) -> tuple[np.ndarray, float, int] | None:
    """Pick the unused pool entry whose natural RC is closest to target.

    `used_mask` is mutated to mark the chosen entry. Returns
    (init_query, init_rc, pool_position) or None if every entry has
    been used.
    """
    if used_mask.all():
        return None
    gaps = np.where(used_mask, np.inf, np.abs(pool_rcs - target_rc))
    best = int(np.argmin(gaps))
    used_mask[best] = True
    return X[pool_idx[best]].copy(), float(pool_rcs[best]), best


# Per-query refinement: short Adam ascent/descent on RC.

def refine_query(
    *,
    X: np.ndarray,
    X_sq: np.ndarray,
    k: int,
    target_rc: float,
    delta: float,
    learning_rate: float,
    max_iter: int,
    init_q: np.ndarray,
    init_rc: float,
) -> tuple[np.ndarray, float, int]:
    """Drive RC into [target/(1+delta), target*(1+delta)] starting from
    init_q. Returns (final_q, final_rc, iterations_used).

    Sign convention: gradient of RC w.r.t. q points in the direction
    that increases RC. If we're below the target band we apply the
    gradient as-is (Adam will move us up); if we're above, we negate.
    Inside the band -> stop early.
    """
    target_low = target_rc / (1 + delta)
    target_high = target_rc * (1 + delta)

    if target_low <= init_rc <= target_high:
        return init_q, init_rc, 0

    q = init_q.copy()
    opt = optax.adam(learning_rate)
    state = opt.init(q)

    rc = init_rc
    best_q, best_gap = q, abs(init_rc - target_rc)

    for it in range(1, max_iter + 1):
        rc, g = fast_rc_and_grad(q, X, X_sq, k)
        gap = abs(rc - target_rc)
        if gap < best_gap:
            best_q, best_gap = q.copy(), gap
        if target_low <= rc <= target_high:
            return q, rc, it
        if rc < target_low:
            # We want RC to go UP; gradient already points that way.
            direction = g
        else:
            # We want RC to go DOWN; flip.
            direction = -g
        upd, state = opt.update(direction, state)
        q = optax.apply_updates(q, upd)

    # Out of budget: return the best-so-far position rather than the
    # last (which may have overshot).
    final_rc = fast_rc(best_q, X, X_sq, k)
    return best_q, final_rc, max_iter


def generate_batch(
    *,
    X: np.ndarray,
    X_sq: np.ndarray,
    k: int,
    target_rc: float,
    delta: float,
    n_queries: int,
    learning_rate: float,
    max_iter: int,
    pool_idx: np.ndarray,
    pool_rcs: np.ndarray,
    used_mask: np.ndarray,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate n_queries queries at the given RC target.

    Uses a shared pre-scored candidate pool (pool_idx / pool_rcs) for
    smart-init: each query consumes one unused pool entry (the one
    with natural RC closest to target) and then runs a short Adam
    refinement to land it inside the acceptance band.

    `used_mask` is shared across targets and mutated in place so each
    base vector becomes init for at most one query across the whole
    run.

    Returns (queries (n,dim), achieved_rcs (n,), stats dict)."""
    queries = np.empty((n_queries, X.shape[1]), dtype=np.float32)
    achieved = np.empty(n_queries, dtype=np.float32)
    iters_used = np.empty(n_queries, dtype=np.int32)

    target_low = target_rc / (1 + delta)
    target_high = target_rc * (1 + delta)

    log_every = max(1, n_queries // 20)
    t_batch = time.time()

    for qi in range(n_queries):
        picked = smart_init_from_pool(
            X=X, pool_idx=pool_idx, pool_rcs=pool_rcs,
            target_rc=target_rc, used_mask=used_mask,
        )
        if picked is None:
            log.warning(
                "  ran out of unused pool candidates after %d/%d queries; "
                "stopping early",
                qi, n_queries,
            )
            queries = queries[:qi]
            achieved = achieved[:qi]
            iters_used = iters_used[:qi]
            break

        init_q, init_rc, _ = picked
        q, rc, it = refine_query(
            X=X, X_sq=X_sq, k=k,
            target_rc=target_rc, delta=delta,
            learning_rate=learning_rate,
            max_iter=max_iter,
            init_q=init_q, init_rc=init_rc,
        )
        queries[qi] = q
        achieved[qi] = rc
        iters_used[qi] = it

        if (qi + 1) % log_every == 0 or qi == n_queries - 1:
            elapsed = time.time() - t_batch
            done = qi + 1
            in_band = ((achieved[:done] >= target_low)
                       & (achieved[:done] <= target_high))
            log.info(
                "  [%4d/%d] elapsed=%5.1fs  "
                "in-band=%4d/%d (%.0f%%)  "
                "rc med=%.3f, mean=%.3f  "
                "iters med=%d",
                done, n_queries, elapsed,
                int(in_band.sum()), done, 100 * in_band.mean(),
                float(np.median(achieved[:done])),
                float(achieved[:done].mean()),
                int(np.median(iters_used[:done])),
            )

    in_band_mask = (achieved >= target_low) & (achieved <= target_high)
    stats = {
        "target_rc": target_rc,
        "delta": delta,
        "n_queries": int(len(achieved)),
        "n_in_band": int(in_band_mask.sum()),
        "frac_in_band": float(in_band_mask.mean()) if len(achieved) else 0.0,
        "rc_min": float(achieved.min()) if len(achieved) else float("nan"),
        "rc_p10": float(np.percentile(achieved, 10)) if len(achieved) else float("nan"),
        "rc_median": float(np.median(achieved)) if len(achieved) else float("nan"),
        "rc_mean": float(achieved.mean()) if len(achieved) else float("nan"),
        "rc_p90": float(np.percentile(achieved, 90)) if len(achieved) else float("nan"),
        "rc_max": float(achieved.max()) if len(achieved) else float("nan"),
        "iters_median": int(np.median(iters_used)) if len(iters_used) else 0,
        "iters_mean": float(iters_used.mean()) if len(iters_used) else 0.0,
        "iters_max": int(iters_used.max()) if len(iters_used) else 0,
    }
    return queries, achieved, stats


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=N_QUERIES_PER_TARGET,
                   help="queries per RC target")
    p.add_argument("--max-iter", type=int, default=80,
                   help="max Adam iterations per query post-init")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--delta", type=float, default=DELTA,
                   help="acceptance band around each target RC")
    p.add_argument("--pool-size", type=int, default=POOL_SIZE,
                   help="shared smart-init candidate pool size")
    p.add_argument("--pool-tile", type=int, default=POOL_TILE,
                   help="rows per GEMM tile when scoring the pool")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--targets", nargs="*", default=None,
                   help="override targets; entry is 'label:rc' "
                        "(e.g. 'rc3p0:3.0')")
    p.add_argument("--dry-run", type=int, default=0,
                   help="if >0, generate this many queries per target "
                        "as a smoke test")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("gen_gist_rc")

    if args.targets:
        targets = []
        for spec in args.targets:
            label, rc = spec.split(":")
            targets.append((label, float(rc)))
    else:
        targets = list(RC_TARGETS)

    n_per_target = args.dry_run if args.dry_run else args.n

    log.info("Loading %s ...", GIST_BASE_FVEC)
    t0 = time.time()
    X = read_fvecs(GIST_BASE_FVEC)
    log.info("  loaded %s in %.1fs", X.shape, time.time() - t0)

    log.info("Precomputing ||X||^2 ...")
    t0 = time.time()
    X_sq = np.einsum("ij,ij->i", X, X).astype(np.float32)
    log.info("  done in %.2fs", time.time() - t0)

    pool_size = min(args.pool_size, X.shape[0])
    log.info(
        "Building shared candidate pool (size=%d, tile=%d) ...",
        pool_size, args.pool_tile,
    )
    pool_rng = np.random.default_rng(args.seed)
    pool_idx = pool_rng.choice(X.shape[0], size=pool_size, replace=False)
    t0 = time.time()
    pool_rcs = score_candidate_pool(
        X=X, X_sq=X_sq, k=K,
        pool_idx=pool_idx, tile_size=args.pool_tile, log=log,
    )
    finite = np.isfinite(pool_rcs)
    n_degenerate = int((~finite).sum())
    finite_rcs = pool_rcs[finite]
    log.info(
        "  pool done in %.1fs.  finite=%d/%d  degenerate=%d.  "
        "RC dist (finite): min=%.3f p1=%.3f p10=%.3f median=%.3f "
        "p90=%.3f p99=%.3f max=%.3f",
        time.time() - t0,
        int(finite.sum()), pool_size, n_degenerate,
        float(finite_rcs.min()),
        float(np.percentile(finite_rcs, 1)),
        float(np.percentile(finite_rcs, 10)),
        float(np.median(finite_rcs)),
        float(np.percentile(finite_rcs, 90)),
        float(np.percentile(finite_rcs, 99)),
        float(finite_rcs.max()),
    )
    for label, target_rc in targets:
        in_band_pool = int(np.sum(
            (pool_rcs >= target_rc / (1 + args.delta))
            & (pool_rcs <= target_rc * (1 + args.delta))
        ))
        log.info(
            "  pool prevalence at %s (RC=%.3f +/-%.0f%%): %d / %d (%.2f%%)",
            label, target_rc, 100 * args.delta,
            in_band_pool, pool_size, 100 * in_band_pool / pool_size,
        )

    used_mask = np.zeros(len(pool_idx), dtype=bool)
    overall_stats = []
    for label, target_rc in targets:
        log.info("")
        log.info(
            "=== target %s (RC=%.3f, +/-%.0f%%): generating %d queries ===",
            label, target_rc, 100 * args.delta, n_per_target,
        )

        queries, achieved, stats = generate_batch(
            X=X, X_sq=X_sq, k=K,
            target_rc=target_rc, delta=args.delta,
            n_queries=n_per_target,
            learning_rate=args.lr,
            max_iter=args.max_iter,
            pool_idx=pool_idx, pool_rcs=pool_rcs,
            used_mask=used_mask,
            log=log,
        )

        log.info("[%s] summary: in_band=%d/%d (%.0f%%)  "
                 "rc med=%.3f mean=%.3f range=[%.3f,%.3f]  "
                 "iters med=%d max=%d",
                 label, stats["n_in_band"], stats["n_queries"],
                 100 * stats["frac_in_band"],
                 stats["rc_median"], stats["rc_mean"],
                 stats["rc_min"], stats["rc_max"],
                 stats["iters_median"], stats["iters_max"])

        out_path = DATASETS_DIR / f"{DATASET_NAME}_query_{label}.fvec"
        write_fvecs(out_path, queries)
        log.info("[%s] wrote %s  (%d x %d)",
                 label, out_path, queries.shape[0], queries.shape[1])

        # Also save the achieved-RC vector next to the fvec for later
        # auditing without recomputing.
        rc_path = DATASETS_DIR / f"{DATASET_NAME}_query_{label}.rc.npy"
        np.save(rc_path, achieved)
        log.info("[%s] wrote %s", label, rc_path)
        overall_stats.append((label, stats))

    log.info("")
    log.info("=== ALL DONE ===")
    for label, stats in overall_stats:
        log.info("  %s -> target %.3f  achieved median %.3f  in-band %d/%d (%.0f%%)",
                 label, stats["target_rc"], stats["rc_median"],
                 stats["n_in_band"], stats["n_queries"],
                 100 * stats["frac_in_band"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
