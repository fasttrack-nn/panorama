#!/bin/bash
# Build & setup script for Panorama.
# Usage: FAISS_OPT_LEVEL=generic ./setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Platform: $OS / $ARCH"

# System dependencies
if [ "$OS" = "Linux" ]; then
    REQUIRED_PKGS=(
        build-essential
        cmake
        swig
        python3-dev
        python3-numpy
        libopenblas-dev
        liblapack-dev
        libomp-dev
    )
    MISSING=()
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            MISSING+=("$pkg")
        fi
    done
    if [ ${#MISSING[@]} -ne 0 ]; then
        echo "Installing missing system packages: ${MISSING[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${MISSING[@]}"
    else
        echo "All system dependencies present."
    fi

elif [ "$OS" = "Darwin" ]; then
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew is required on macOS. Install from https://brew.sh"
        exit 1
    fi
    REQUIRED_FORMULAE=(cmake swig libomp openblas)
    MISSING=()
    for formula in "${REQUIRED_FORMULAE[@]}"; do
        if ! brew list "$formula" &>/dev/null; then
            MISSING+=("$formula")
        fi
    done
    if [ ${#MISSING[@]} -ne 0 ]; then
        echo "Installing missing Homebrew formulae: ${MISSING[*]}"
        brew install "${MISSING[@]}"
    else
        echo "All system dependencies present."
    fi

    if ! xcode-select -p &>/dev/null; then
        echo "Xcode Command Line Tools not found. Installing ..."
        xcode-select --install
        echo "Re-run this script after the install finishes."
        exit 1
    fi
else
    echo "ERROR: Unsupported OS '$OS'. This script supports Linux and macOS."
    exit 1
fi

# Validate FAISS_OPT_LEVEL
if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
    VALID_OPT_LEVELS="generic avx2 avx512 avx512_spr"
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    if [ "$OS" = "Darwin" ]; then
        VALID_OPT_LEVELS="generic"
    else
        VALID_OPT_LEVELS="generic sve"
    fi
else
    echo "WARNING: Unknown architecture '$ARCH'; allowing all opt levels."
    VALID_OPT_LEVELS="generic avx2 avx512 avx512_spr sve"
fi

OPT_LEVEL="${FAISS_OPT_LEVEL:-}"
if [ -z "$OPT_LEVEL" ]; then
    echo "ERROR: FAISS_OPT_LEVEL is not set."
    echo "Valid options for $ARCH: $VALID_OPT_LEVELS"
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        echo "  (NEON is always enabled on ARM64; use 'generic' for NEON-only)"
    fi
    echo "Example: FAISS_OPT_LEVEL=generic ./setup.sh"
    exit 1
fi
case " $VALID_OPT_LEVELS " in
    *" $OPT_LEVEL "*) ;;
    *)
        echo "ERROR: Invalid FAISS_OPT_LEVEL='$OPT_LEVEL' for $ARCH"
        echo "Valid options: $VALID_OPT_LEVELS"
        if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
            echo "  (NEON is always enabled on ARM64; use 'generic' for NEON-only)"
        fi
        exit 1
        ;;
esac
echo "Using FAISS_OPT_LEVEL=$OPT_LEVEL"

# Virtual environment
if [ ! -d venv ]; then
    echo "Creating virtual environment ..."
    python3 -m venv venv --system-site-packages
fi
source venv/bin/activate
echo "Using python: $(which python3)  ($(python3 --version))"

# Python dependencies
echo ""
echo "Installing Python dependencies ..."
pip install --upgrade pip setuptools
pip install -r requirements.txt

# Build faiss (C++ + SWIG, linked via .pth)
echo ""
echo "Building faiss ..."
cd "$SCRIPT_DIR/faiss"

CMAKE_EXTRA_FLAGS=()
if [ "$OS" = "Darwin" ]; then
    OMP_PREFIX="$(brew --prefix libomp)"
    CMAKE_EXTRA_FLAGS+=(
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
        -DOpenMP_C_LIB_NAMES="omp"
        -DOpenMP_CXX_LIB_NAMES="omp"
        -DOpenMP_omp_LIBRARY="${OMP_PREFIX}/lib/libomp.dylib"
    )
fi

cmake -B build . \
    -DFAISS_OPT_LEVEL="$OPT_LEVEL" \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "${CMAKE_EXTRA_FLAGS[@]}"

if [ "$OPT_LEVEL" = "generic" ]; then
    FAISS_TARGET=faiss; SWIG_TARGET=swigfaiss
else
    FAISS_TARGET="faiss_${OPT_LEVEL}"; SWIG_TARGET="swigfaiss_${OPT_LEVEL}"
fi

make -C build -j "$FAISS_TARGET"
make -C build -j "$SWIG_TARGET"

# Run setup.py build (but not install) to generate the Python package tree
cd build/faiss/python
python3 setup.py build
cd "$SCRIPT_DIR"

# Add the faiss build output to the venv's import path via a .pth file
FAISS_LIB="$SCRIPT_DIR/faiss/build/faiss/python/build/lib"
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
echo "$FAISS_LIB" > "$SITE_PACKAGES/faiss-local.pth"
echo "Linked faiss into venv via .pth file"

# Build PDX (pdxearch)
echo ""
echo "Building pdxearch ..."
cd "$SCRIPT_DIR/PDX"
if [ "$OS" = "Darwin" ]; then
    OMP_PREFIX="$(brew --prefix libomp)"
    CMAKE_ARGS="-DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp;-I${OMP_PREFIX}/include \
-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp;-I${OMP_PREFIX}/include \
-DOpenMP_C_LIB_NAMES=omp \
-DOpenMP_CXX_LIB_NAMES=omp \
-DOpenMP_omp_LIBRARY=${OMP_PREFIX}/lib/libomp.dylib" \
    pip install .
else
    pip install .
fi
cd "$SCRIPT_DIR"

# Build ADSampling
echo ""
echo "Building ADSampling ..."
cd "$SCRIPT_DIR/ADSampling"
bash build.sh
cd "$SCRIPT_DIR"

# Verify imports
echo ""
echo "Verifying imports ..."
python3 -c "
import faiss;      print(f'  faiss      OK  ({faiss.get_compile_options()})')
import duckdb;     print(f'  duckdb     OK  (v{duckdb.__version__})')
import sklearn;    print(f'  sklearn    OK  (v{sklearn.__version__})')
import matplotlib; print(f'  matplotlib OK  (v{matplotlib.__version__})')

import pdxearch;   print('  pdxearch   OK')
"

echo ""
echo "Setup complete! Next steps:"
echo ""
echo "  source venv/bin/activate"
echo "  python3 data/install_datasets.py [datasets-dir]"
echo "  cd benches && python3 bench.py --config example_config.json --scale-factor 0.1 --datasets-dir [datasets-dir]"
echo ""
