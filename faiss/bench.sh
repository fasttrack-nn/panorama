#!/bin/bash
set -e
CXX=/usr/bin/c++
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
FLAGS="-O3 -DNDEBUG -std=gnu++20 -mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt -mbmi2 $EXTRA_FLAGS"

echo "Compiling: $CXX $FLAGS -o kernel_bench kernel_bench.cpp"
$CXX $FLAGS -o kernel_bench kernel_bench.cpp

echo "Running..."
./kernel_bench

echo ""
echo "Code sizes:"
objdump -d -C --no-show-raw-insn kernel_bench | grep -E '^[0-9a-f]+ <.*v[0-9]+_' | while read line; do
    echo "  $line"
done
