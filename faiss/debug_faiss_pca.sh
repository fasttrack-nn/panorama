#!/bin/bash
# Deep FAISS PCA Diagnostic - Run this to identify the exact issue

echo "=========================================="
echo "DEEP FAISS PCA DIAGNOSTIC"
echo "=========================================="
echo ""

# 1. Identify which FAISS is actually being used
echo "1. WHICH FAISS IS BEING IMPORTED?"
echo "===================================="
python3 << 'PYEOF'
import faiss
import sys
print(f"Python executable: {sys.executable}")
print(f"FAISS module location: {faiss.__file__}")
print(f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
PYEOF
echo ""

# 2. Check the actual .so file being used
echo "2. ACTUAL .SO FILE BEING LOADED"
echo "================================"
python3 << 'PYEOF'
import faiss
import os
so_path = os.path.dirname(faiss.__file__)
print(f"FAISS library directory: {so_path}")
for f in os.listdir(so_path):
    if f.endswith('.so'):
        full_path = os.path.join(so_path, f)
        print(f"  {f}")
PYEOF
echo ""

# 3. Run ldd on the actual FAISS .so
echo "3. DEPENDENCIES OF ACTUAL FAISS .SO"
echo "===================================="
python3 << 'PYEOF'
import faiss
import os
import subprocess
so_path = os.path.dirname(faiss.__file__)
for f in os.listdir(so_path):
    if '_module.so' in f and 'faiss' in f:
        full_path = os.path.join(so_path, f)
        print(f"File: {full_path}")
        result = subprocess.run(['ldd', full_path], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'blas' in line.lower() or 'lapack' in line.lower():
                print(f"  {line}")
        break
PYEOF
echo ""

# 4. Check NumPy BLAS
echo "4. NUMPY BLAS CONFIGURATION"
echo "============================"
python3 << 'PYEOF'
import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"NumPy file: {np.__file__}")
# Try the correct way to get BLAS info
try:
    import numpy.core._multiarray_umath as mu
    if hasattr(mu, '__file__'):
        print(f"NumPy multiarray location: {mu.__file__}")
except:
    pass

# Try using numpy's internal config
try:
    config = np.show_config()
    print("NumPy config obtained")
except:
    print("Could not get NumPy config")
PYEOF
echo ""

# 6. Test with index_factory PCA+IVF+Flat
echo "6. INDEX_FACTORY: PCA64,IVF10,Flat TEST"
echo "========================================"
python3 << 'PYEOF'
import numpy as np
import faiss

np.random.seed(42)
d = 64
n_train = 1000
n_test = 100

train_data = np.random.randn(n_train, d).astype('float32')
test_data = np.random.randn(n_test, d).astype('float32')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print()

# Build index using index_factory
index_str = "PCA64,IVF10,Flat"
print(f"Building index with: {index_str}")
index = faiss.index_factory(d, index_str)

print(f"Index is_trained: {index.is_trained}")
index.train(train_data)
print(f"After training, is_trained: {index.is_trained}")

index.add(train_data)
print(f"Added {index.ntotal} vectors")
print()

# Search
D, I = index.search(test_data[:10], k=5)

print(f"Search results (distances): {D[0]}")
print(f"Search results (indices): {I[0]}")
print(f"Any invalid indices (-1)? {(I == -1).any()}")
print(f"Any zero or infinite distances? {np.any((D == 0) | np.isinf(D))}")
print()

# Compare with flat index
flat_index = faiss.IndexFlatL2(d)
flat_index.add(train_data)
D_flat, I_flat = flat_index.search(test_data[:10], k=5)

print(f"Flat index search results: {I_flat[0]}")
print(f"Flat index distances: {D_flat[0]}")
print()

# Check if results are reasonable
print("Result analysis:")
print(f"  PCA index valid results? {not (I == -1).any()}")
print(f"  Flat index valid results? {not (I_flat == -1).any()}")
if not (I == -1).any() and not (I_flat == -1).any():
    print(f"  Recall@5: {np.mean([len(set(I[i]) & set(I_flat[i])) for i in range(len(I))]) / 5}")
PYEOF
echo ""

# 7. Check if it's a linking issue with OpenBLAS
echo "7. OPENBLAS VERSION & SYMBOLS"
echo "=============================="
nm -D /lib/x86_64-linux-gnu/libopenblas.so.0 2>/dev/null | grep -E "^[0-9a-f]+ T (cblas_|dgemm|dgesvd)" | head -20 || echo "Could not inspect libopenblas symbols"
echo ""

# 8. Test with explicit BLAS calls
echo "8. DIRECT BLAS TEST"
echo "==================="
python3 << 'PYEOF'
import numpy as np
import ctypes

# Test if BLAS is working correctly
A = np.random.randn(10, 10).astype('float32')
B = np.random.randn(10, 10).astype('float32')
C = np.dot(A, B)

print(f"Basic matrix multiply result (not NaN): {not np.isnan(C).any()}")
print(f"Result contains valid values: {np.isfinite(C).all()}")
print(f"First element: {C[0, 0]}")
PYEOF
echo ""

echo "=========================================="
echo "END DIAGNOSTIC"
echo "=========================================="