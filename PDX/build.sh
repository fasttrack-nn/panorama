#!/bin/bash
# Build and install pdxearch into the project venv.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not found at $VENV — run setup.sh first."
    exit 1
fi

source "$VENV/bin/activate"
echo "Installing pdxearch into $(which python3) ..."
pip install "$SCRIPT_DIR"
echo "Done."
