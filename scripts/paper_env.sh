#!/usr/bin/env bash

# Shared environment defaults for review/reproduction commands.
# Keeps uv, matplotlib, and fontconfig caches inside the repository so the
# artifact behaves well in restricted environments.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  export PAPER_PYTHON="$ROOT_DIR/.venv/bin/python"
  export PAPER_PYTEST="$ROOT_DIR/.venv/bin/pytest"
  export PAPER_QDP="$ROOT_DIR/.venv/bin/query-data-predictor"
else
  export PAPER_PYTHON=""
  export PAPER_PYTEST=""
  export PAPER_QDP=""
fi

paper_sync() {
  if [[ -n "$PAPER_PYTHON" ]]; then
    echo "Using existing .venv at $ROOT_DIR/.venv"
  else
    uv sync --frozen --group dev
  fi
}

paper_python() {
  if [[ -n "$PAPER_PYTHON" ]]; then
    "$PAPER_PYTHON" "$@"
  else
    uv run python "$@"
  fi
}

paper_pytest() {
  if [[ -n "$PAPER_PYTEST" ]]; then
    "$PAPER_PYTEST" "$@"
  else
    uv run pytest "$@"
  fi
}

paper_qdp() {
  if [[ -n "$PAPER_QDP" ]]; then
    "$PAPER_QDP" "$@"
  else
    uv run query-data-predictor "$@"
  fi
}
