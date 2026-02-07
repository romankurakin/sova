#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This build script is for macOS only."
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$ROOT_DIR/.pyinstaller}"
PYINSTALLER_SPEC="${PYINSTALLER_SPEC:-pyinstaller==6.18.0}"
SITE_PACKAGES="$ROOT_DIR/.venv/lib/python3.13/site-packages"
PYTHONPATH_VALUE="$SITE_PACKAGES:$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$PYTHONPATH"
fi

PYI_ARGS=(
  --clean
  --noconfirm
  --onefile
  --name sova
  --specpath "$ROOT_DIR/build"
  --workpath "$ROOT_DIR/build/pyinstaller"
  --distpath "$ROOT_DIR/dist"
  --collect-all sqlite_vector
  --collect-all rich
  --collect-all pymupdf
  --collect-all pymupdf.layout
  --collect-all pymupdf4llm
  sova/__main__.py
)

run_with_uv() {
  UV_CACHE_DIR="$UV_CACHE_DIR" \
  PYINSTALLER_CONFIG_DIR="$PYINSTALLER_CONFIG_DIR" \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  uv run --no-project --with "$PYINSTALLER_SPEC" pyinstaller "${PYI_ARGS[@]}"
}

# Freeze the CLI into a single binary: dist/sova
if ! run_with_uv; then
  PYINSTALLER_BIN="$(find "$UV_CACHE_DIR/archive-v0" -type f -path "*/bin/pyinstaller" 2>/dev/null | head -n 1)"
  if [[ -z "$PYINSTALLER_BIN" ]]; then
    echo "Failed to fetch pyinstaller and no cached binary found in $UV_CACHE_DIR/archive-v0"
    exit 1
  fi
  echo "uv fetch failed, using cached pyinstaller: $PYINSTALLER_BIN"
  PYINSTALLER_CONFIG_DIR="$PYINSTALLER_CONFIG_DIR" \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  "$PYINSTALLER_BIN" "${PYI_ARGS[@]}"
fi

echo "Built binary: $ROOT_DIR/dist/sova"
echo "Run it with: $ROOT_DIR/dist/sova -s \"your query\""
