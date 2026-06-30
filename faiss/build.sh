#!/bin/bash
# Build Faiss with optimized AVX512.
set -e

OPT_LEVEL="${FAISS_OPT_LEVEL:-avx512}"

echo "Configuring Faiss (opt_level=$OPT_LEVEL)..."
cmake -B build . \
    -DFAISS_OPT_LEVEL="$OPT_LEVEL" \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ "$OPT_LEVEL" = "generic" ]; then
    FAISS_TARGET=faiss
    SWIG_TARGET=swigfaiss
else
    FAISS_TARGET="faiss_${OPT_LEVEL}"
    SWIG_TARGET="swigfaiss_${OPT_LEVEL}"
fi

echo "Building Faiss core library..."
make -C build -j "$FAISS_TARGET"

echo "Building SWIG Python bindings..."
make -C build -j "$SWIG_TARGET"

echo "Installing Python package..."
cd build/faiss/python && python3 setup.py install --user && cd ../../..

echo "Faiss built."
