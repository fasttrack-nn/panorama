"""
Per-index-type data transforms for Panorama benchmarks.

Caches:
  - sklearn PCA model keyed by (dataset_name, nb) - shared by all panorama indexes
  - Plain PCA LinearTransform keyed by (dataset_name, nb)
  - IVFPQPano full transform keyed by (dataset_name, nb, nlevels, alpha)
"""

from typing import List, Tuple

import faiss
import numpy as np
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

# Module-level caches (alive for the duration of a script run)

_pca_model_cache: dict = {}
_pca_lt_cache: dict = {}
_ivfpq_pano_lt_cache: dict = {}

# Public API

def _pick_svd_solver(nb: int, d: int) -> str:
    """Pick an SVD solver that doesn't overflow LAPACK 32-bit indices.

    sklearn's default ``svd_solver="auto"`` picks ``"full"`` whenever
    ``n_components`` equals the feature count, which routes through
    scipy's LAPACK ``gesdd`` and fails on ``nb*d > 2**31`` matrices
    (e.g. openai 900_000 x 3072 = 2.76e9 > 2.15e9).

    We force ``"randomized"`` for any matrix above ~1.5e9 elements.
    Randomized SVD does not call LAPACK on the full matrix, so it is
    immune to the overflow. For PCA(n_components=d) with d in the
    thousands and n in the hundreds of thousands, randomized SVD is also
    several times faster than the exact LAPACK path with negligible loss
    of accuracy in the leading components.
    """
    OVERFLOW_GUARD = 1_500_000_000  # well below 2**31, leaves headroom
    if nb * d > OVERFLOW_GUARD:
        return "randomized"
    return "auto"


def get_pca_model(xb: np.ndarray, dataset_name: str, nb: int) -> PCA:
    """Return a fitted sklearn PCA (n_components=d), cached by (dataset, nb)."""
    key = (dataset_name, nb)
    if key not in _pca_model_cache:
        d = xb.shape[1]
        solver = _pick_svd_solver(nb, d)
        print(
            f"    [transforms] Fitting PCA (d={d}, nb={nb}, solver={solver}) "
            f"for {dataset_name} ..."
        )
        # iterated_power controls power-method iterations for the randomized
        # solver. Bump above the default 'auto' so the leading eigenvectors
        # are accurate even when we ask for full d components.
        pca_kwargs = {"n_components": d, "svd_solver": solver}
        if solver == "randomized":
            pca_kwargs["iterated_power"] = 7
            pca_kwargs["random_state"] = 0
        pca = PCA(**pca_kwargs)
        pca.fit(xb)
        _pca_model_cache[key] = pca
    return _pca_model_cache[key]


def get_pca_linear_transform(
    xb: np.ndarray, dataset_name: str, nb: int,
) -> faiss.LinearTransform:
    """Return a faiss.LinearTransform wrapping a plain PCA.

    Cached by (dataset_name, nb). Reused across nlevels and index types.
    """
    key = (dataset_name, nb)
    if key not in _pca_lt_cache:
        pca = get_pca_model(xb, dataset_name, nb)
        _pca_lt_cache[key] = _sklearn_pca_to_linear_transform(pca)
    return _pca_lt_cache[key]


def get_ivfpq_pano_transform(
    xb: np.ndarray,
    dataset_name: str,
    nb: int,
    nlevels: int,
    alpha: float,
) -> faiss.LinearTransform:
    """Return a faiss.LinearTransform with PCA + energy-spill + level-equalization.

    Cached by (dataset_name, nb, nlevels, alpha).
    """
    key = (dataset_name, nb, nlevels, alpha)
    if key not in _ivfpq_pano_lt_cache:
        print(f"    [transforms] Building IVFPQPano transform "
              f"(nlevels={nlevels}, alpha={alpha}) for {dataset_name} nb={nb} ...")
        lt = _make_pca_level_rotation_transform(xb, nlevels, alpha)
        _ivfpq_pano_lt_cache[key] = lt
    return _ivfpq_pano_lt_cache[key]


# Internal: plain PCA -> faiss LinearTransform

def _sklearn_pca_to_linear_transform(pca: PCA) -> faiss.LinearTransform:
    d = pca.n_components_
    P = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)

    lt = faiss.LinearTransform(d, d, True)
    faiss.copy_array_to_vector(P.ravel(), lt.A)
    faiss.copy_array_to_vector(-(P @ mean).ravel(), lt.b)
    lt.is_trained = True
    lt.have_bias = True
    return lt


# Internal: IVFPQPano full transform (ported from faiss/benchs/bench_ivfpq_panorama.py)
#
# Level boundaries MUST match the C++ Panorama kernel which uses ceiling
# division:  level_width_dims = (d + n_levels - 1) / n_levels
# The last level may be shorter than the others.


def _level_ranges(d: int, n_levels: int) -> List[Tuple[int, int]]:
    """Return (start, end) pairs matching C++ Panorama level boundaries."""
    block_size = (d + n_levels - 1) // n_levels
    return [(l * block_size, min((l + 1) * block_size, d)) for l in range(n_levels)]


def _find_n_spill(
    variances: np.ndarray,
    level_start: int,
    level_width: int,
    max_energy_per_level: float,
    d: int,
) -> int:
    """Smallest number of extra dims to spill into so the level stays under cap."""
    level_end = level_start + level_width
    max_extra = d - level_end
    if max_extra == 0:
        return 0

    total = float(np.sum(variances[level_start:level_end]))
    for n in range(1, max_extra + 1):
        total += float(variances[level_end + n - 1])
        if level_width * total / (level_width + n) <= max_energy_per_level:
            return n
    return max_extra


def _random_orthogonal(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    H = rng.randn(size, size).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))[:, None]
    return Q


def _build_energy_spill_rotation(
    eigenvalues: np.ndarray,
    n_levels: int,
    alpha: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonal matrix that caps per-level energy via localized rotations."""
    d = len(eigenvalues)
    total_energy = float(np.sum(eigenvalues))
    max_energy_per_level = alpha * total_energy / n_levels

    variances = eigenvalues.astype(np.float32).copy()
    spill_matrix = np.eye(d, dtype=np.float32)
    rng = np.random.RandomState(seed)

    for start, end in _level_ranges(d, n_levels):
        level_width = end - start
        level_energy = float(np.sum(variances[start:end]))

        if level_energy <= max_energy_per_level:
            continue

        n_spill = _find_n_spill(
            variances, start, level_width, max_energy_per_level, d,
        )
        if n_spill == 0:
            continue

        sub_end = end + n_spill
        Q = _random_orthogonal(level_width + n_spill, rng)

        full_Q = np.eye(d, dtype=np.float32)
        full_Q[start:sub_end, start:sub_end] = Q
        spill_matrix = full_Q @ spill_matrix

        avg_var = float(np.sum(variances[start:sub_end])) / (level_width + n_spill)
        variances[start:sub_end] = avg_var

    return spill_matrix, variances


def _build_level_equalization_rotation(
    d: int, n_levels: int, seed: int = 77,
) -> np.ndarray:
    """Block-diagonal random rotation for within-level energy equalization."""
    rng = np.random.RandomState(seed)
    blocks = [_random_orthogonal(end - start, rng)
              for start, end in _level_ranges(d, n_levels)]
    return block_diag(*blocks).astype(np.float32)


def _make_pca_level_rotation_transform(
    xb: np.ndarray, n_levels: int, alpha: float, seed: int = 77,
) -> faiss.LinearTransform:
    """PCA + energy-spill + per-level rotation as one LinearTransform.

    Pipeline:  y = R_eq @ R_spill @ P @ (x - mean)
    """
    nb, dim = xb.shape

    solver = _pick_svd_solver(nb, dim)
    pca_kwargs = {"n_components": dim, "svd_solver": solver}
    if solver == "randomized":
        pca_kwargs["iterated_power"] = 7
        pca_kwargs["random_state"] = 0
    pca = PCA(**pca_kwargs)
    pca.fit(xb)
    P = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    eigenvalues = pca.explained_variance_.astype(np.float32)

    R_spill, _ = _build_energy_spill_rotation(
        eigenvalues, n_levels, alpha, seed=seed,
    )

    R_eq = _build_level_equalization_rotation(dim, n_levels, seed=seed + 1)

    combined = (R_eq @ R_spill @ P).astype(np.float32)

    lt = faiss.LinearTransform(dim, dim, True)
    faiss.copy_array_to_vector(combined.ravel(), lt.A)
    faiss.copy_array_to_vector(-(combined @ mean).ravel(), lt.b)
    lt.is_trained = True
    lt.have_bias = True
    return lt
