#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Download Eigen 3.4.0 headers if not present
if [ ! -d "src/Eigen" ]; then
    echo "Downloading Eigen 3.4.0 ..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -O /tmp/eigen-3.4.0.tar.gz
    tar -xzf /tmp/eigen-3.4.0.tar.gz -C /tmp
    cp -r /tmp/eigen-3.4.0/Eigen src/Eigen
    rm -rf /tmp/eigen-3.4.0 /tmp/eigen-3.4.0.tar.gz
    echo "Eigen installed to src/Eigen/"
else
    echo "Eigen already present."
fi

mkdir -p build

echo "Compiling index_hnsw ..."
g++ -O3 -march=native -o build/index_hnsw src/index_hnsw.cpp -I src/

echo "Compiling search_hnsw ..."
g++ -O3 -march=native -o build/search_hnsw src/search_hnsw.cpp -I src/

echo "Compiling index_ivf ..."
g++ -O3 -march=native -o build/index_ivf src/index_ivf.cpp -I src/

echo "Compiling search_ivf ..."
g++ -O3 -march=native -o build/search_ivf src/search_ivf.cpp -I src/

echo "ADSampling binaries built in build/"
ls -l build/
