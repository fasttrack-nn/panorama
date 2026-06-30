"""Debug script: verify energy distribution after PCA and per-level rotations.

Shows that the current bench code composes the PCA+rotation transform
incorrectly, scrambling the per-level energy distribution.

FAISS LinearTransform convention (from VectorTransform.cpp sgemm call):
    y = A_stored @ x_col + b       (column-vector form)
  or equivalently
    y = x_row @ A_stored^T + b     (row-vector form)

  Bias:  b = -A_stored @ mean

The correct panorama transform is:
    y = A_block @ P @ (x - mean)
    i.e.  center -> PCA -> per-level rotation

The buggy code does:
    combined = P @ A_block          (WRONG ORDER)
    b = -mean @ combined            (WRONG: vector @ matrix instead of matrix @ vector)
"""

import numpy as np
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

try:
    from faiss.contrib.datasets_fb import DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import DatasetGIST1M

ds = DatasetGIST1M()
xt = ds.get_train()[:50000]
d = xt.shape[1]

n_levels = 16
block_size = d // n_levels
seed = 42

print(f"d = {d}, n_levels = {n_levels}, block_size = {block_size}")
print()

# Fit PCA
pca = PCA(n_components=d)
pca.fit(xt)
P = pca.components_.astype(np.float32)   # (d, d), rows = principal components
mean = pca.mean_.astype(np.float32)      # (d,)
eigenvalues = pca.explained_variance_     # (d,)

# Block-diagonal random rotation
rng = np.random.RandomState(seed)
blocks = []
for _ in range(n_levels):
    H = rng.randn(block_size, block_size).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))[:, None]
    blocks.append(Q)
A_block = block_diag(*blocks).astype(np.float32)  # (d, d)

# Apply transforms to training data
x_centered = xt - mean  # (N, d)

z_pca = x_centered @ P.T  # correct PCA-transformed data (N, d)

# Correct: PCA first, then per-level rotation
z_correct = z_pca @ A_block.T  # (N, d)

# Buggy (what the bench code effectively computes):
# combined_buggy = P @ A_block, transform = x @ combined_buggy^T
# = x @ A_block^T @ P^T  (rotation in original space, THEN PCA)
z_buggy = x_centered @ A_block.T @ P.T  # bug: rotation before PCA

# Per-dimension variance
var_pca = np.var(z_pca, axis=0)
var_correct = np.var(z_correct, axis=0)
var_buggy = np.var(z_buggy, axis=0)

# Per-level energy (sum of variances in each block)
def per_level_energy(variances, n_levels, block_size):
    energies = []
    for lev in range(n_levels):
        s = lev * block_size
        energies.append(np.sum(variances[s : s + block_size]))
    return np.array(energies)

level_energy_pca = per_level_energy(var_pca, n_levels, block_size)
level_energy_correct = per_level_energy(var_correct, n_levels, block_size)
level_energy_buggy = per_level_energy(var_buggy, n_levels, block_size)

# Print results
print("=" * 70)
print("PER-LEVEL ENERGY (sum of per-dim variance in each level's block)")
print("=" * 70)
print(f"{'Level':>5}  {'After PCA':>14}  {'PCA+Rot(correct)':>18}  {'PCA+Rot(buggy)':>18}  {'buggy/PCA ratio':>16}")
print("-" * 70)
for lev in range(n_levels):
    ratio = level_energy_buggy[lev] / level_energy_pca[lev] if level_energy_pca[lev] > 0 else 0
    print(
        f"{lev:5d}  {level_energy_pca[lev]:14.2f}  "
        f"{level_energy_correct[lev]:18.2f}  "
        f"{level_energy_buggy[lev]:18.2f}  "
        f"{ratio:16.4f}"
    )

print()
total_pca = level_energy_pca.sum()
total_correct = level_energy_correct.sum()
total_buggy = level_energy_buggy.sum()
print(f"Total energy:  PCA={total_pca:.2f}  correct={total_correct:.2f}  buggy={total_buggy:.2f}")

print()
print("=" * 70)
print("ENERGY RATIOS  level[i] / level[0]  (should decay for panorama)")
print("=" * 70)
print(f"{'Level':>5}  {'After PCA':>14}  {'PCA+Rot(correct)':>18}  {'PCA+Rot(buggy)':>18}")
print("-" * 70)
for lev in range(n_levels):
    r_pca = level_energy_pca[lev] / level_energy_pca[0]
    r_correct = level_energy_correct[lev] / level_energy_correct[0]
    r_buggy = level_energy_buggy[lev] / level_energy_buggy[0]
    print(
        f"{lev:5d}  {r_pca:14.6f}  "
        f"{r_correct:18.6f}  "
        f"{r_buggy:18.6f}"
    )

print()
print("=" * 70)
print("FIRST 20 PER-DIMENSION VARIANCES (should NOT be sorted after rotation,")
print("but per-level totals should match PCA)")
print("=" * 70)
print(f"{'Dim':>5}  {'PCA eigenval':>14}  {'PCA var':>14}  {'correct var':>14}  {'buggy var':>14}")
print("-" * 70)
for i in range(min(20, d)):
    print(
        f"{i:5d}  {eigenvalues[i]:14.4f}  "
        f"{var_pca[i]:14.4f}  "
        f"{var_correct[i]:14.4f}  "
        f"{var_buggy[i]:14.4f}"
    )

# Verify bias computation
print()
print("=" * 70)
print("BIAS VERIFICATION")
print("=" * 70)

combined_buggy = P @ A_block
combined_correct = A_block @ P

bias_buggy = (-mean @ combined_buggy)        # bench code: vector @ matrix
bias_correct = -(combined_correct @ mean)     # correct: matrix @ vector

# What the buggy bias actually equals:
bias_buggy_equiv = -(combined_buggy.T @ mean)  # = -(A_block^T @ P^T @ mean)

print(f"bias_buggy  == -(combined_buggy^T @ mean)? {np.allclose(bias_buggy, bias_buggy_equiv)}")
print(f"bias_buggy  == bias_correct?                {np.allclose(bias_buggy, bias_correct)}")
print(f"||bias_correct - bias_buggy||  = {np.linalg.norm(bias_correct - bias_buggy):.4f}")
print(f"||bias_correct||               = {np.linalg.norm(bias_correct):.4f}")

# Check: per-level energy preserved by correct rotation?
print()
print("=" * 70)
print("KEY CHECK: per-level energy preserved by correct rotation?")
print("=" * 70)
for lev in range(n_levels):
    diff = abs(level_energy_correct[lev] - level_energy_pca[lev])
    pct = 100 * diff / level_energy_pca[lev] if level_energy_pca[lev] > 0 else 0
    ok = "OK" if pct < 0.01 else "FAIL"
    print(f"  Level {lev:2d}: PCA={level_energy_pca[lev]:.2f}  correct={level_energy_correct[lev]:.2f}  diff={pct:.6f}%  {ok}")
