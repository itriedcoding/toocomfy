#!/usr/bin/env bash
# Build the toocomfy desktop app (Electron) on macOS or Linux.
# Produces dist/toocomfy-darwin/ and dist/toocomfy-linux/ (per-platform).
#
# Usage:
#     bash scripts/build_desktop.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Python backend in an isolated venv
if [[ ! -d "$ROOT/venv" ]]; then
    python3 -m venv "$ROOT/venv"
fi
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
pip install -q -r "$ROOT/requirements.txt"

# 2. Electron app
cd "$ROOT/desktop"
npm install
npm run build

echo ""
echo "Build output: $ROOT/desktop/dist/"
