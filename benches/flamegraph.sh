#!/bin/bash
#
# Profile bench.py with perf and generate a FlameGraph SVG.
#
# Usage:
#   ./flamegraph.sh --config big_bench.yaml --scale-factor 0.01
#
# All arguments are forwarded to bench.py. Output goes to benches/profiles/.
#
# Prerequisites:
#   1. Build faiss with profiling symbols:
#        cd faiss && FAISS_PROFILE=1 ./build.sh
#      (or: FAISS_PROFILE=1 ./setup.sh)
#   2. Linux perf tools installed (apt install linux-tools-$(uname -r))
#   3. FlameGraph repo cloned at <repo>/FlameGraph/
#
# Environment variables:
#   PERF_FREQ       - sampling frequency in Hz (default: 997)
#   PERF_CALLGRAPH  - call-graph mode: fp, dwarf, lbr (default: fp)
#   FLAMEGRAPH_OPTS - extra flags for flamegraph.pl (e.g. --width 1800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FLAMEGRAPH_DIR="$REPO_ROOT/FlameGraph"
PROFILE_DIR="$SCRIPT_DIR/profiles"

FREQ="${PERF_FREQ:-997}"
CALLGRAPH="${PERF_CALLGRAPH:-fp}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 [bench.py arguments]"
    echo ""
    echo "Example:"
    echo "  $0 --config big_bench.yaml --scale-factor 0.01 --datasets-dir /datasets/datasets"
    echo ""
    echo "Environment:"
    echo "  PERF_FREQ=997       sampling frequency (Hz)"
    echo "  PERF_CALLGRAPH=fp   call-graph mode (fp|dwarf|lbr)"
    exit 1
fi

for tool in perf python3; do
    if ! command -v "$tool" &>/dev/null; then
        echo "ERROR: $tool not found in PATH" >&2
        exit 1
    fi
done
for script in flamegraph.pl stackcollapse-perf.pl; do
    if [ ! -f "$FLAMEGRAPH_DIR/$script" ]; then
        echo "ERROR: $FLAMEGRAPH_DIR/$script not found" >&2
        echo "Clone FlameGraph: git clone https://github.com/brendangregg/FlameGraph $FLAMEGRAPH_DIR" >&2
        exit 1
    fi
done

mkdir -p "$PROFILE_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PERF_DATA="$PROFILE_DIR/perf_${TIMESTAMP}.data"
FOLDED_RAW="$PROFILE_DIR/folded_raw_${TIMESTAMP}.txt"
FOLDED="$PROFILE_DIR/folded_${TIMESTAMP}.txt"
SVG="$PROFILE_DIR/flamegraph_${TIMESTAMP}.svg"

echo "============================================"
echo " FlameGraph profiling for bench.py"
echo "============================================"
echo "  perf freq:    $FREQ Hz"
echo "  call-graph:   $CALLGRAPH"
echo "  perf data:    $PERF_DATA"
echo "  output SVG:   $SVG"
echo "  bench args:   $*"
echo "============================================"
echo ""

echo "[1/4] Recording with perf ..."
perf record \
    -F "$FREQ" \
    --call-graph "$CALLGRAPH" \
    -o "$PERF_DATA" \
    -- python3 "$SCRIPT_DIR/bench.py" "$@"

echo ""
echo "[2/4] Collapsing stacks ..."
perf script -i "$PERF_DATA" --no-inline \
    | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" --kernel \
    > "$FOLDED_RAW"

echo "[3/4] Cleaning up [unknown] frames ..."
# Strip [unknown] frames that break stack continuity, and drop
# pure-[unknown] stacks that carry no information.
sed 's/;\[unknown\]//g' "$FOLDED_RAW" \
    | grep -v '^\[unknown\] ' \
    > "$FOLDED"

TOTAL_SAMPLES=$(wc -l < "$FOLDED")
echo "  $TOTAL_SAMPLES unique stacks after cleanup"

echo ""
echo "[4/4] Generating flamegraph ..."
"$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "bench.py – $(date +%Y-%m-%d\ %H:%M)" \
    --subtitle "perf @ ${FREQ}Hz, call-graph=${CALLGRAPH}" \
    --countname samples \
    --width 1600 \
    ${FLAMEGRAPH_OPTS:-} \
    "$FOLDED" > "$SVG"

echo ""
echo "============================================"
echo " Done!"
echo "  SVG:    $SVG"
echo "  Folded: $FOLDED"
echo "  Raw:    $PERF_DATA"
echo ""
echo "  Open in browser:  xdg-open $SVG"
echo "============================================"
