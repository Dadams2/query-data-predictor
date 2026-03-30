#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/paper_env.sh"

cd "$ROOT_DIR"

paper_sync
paper_python extract_table_data.py --check
