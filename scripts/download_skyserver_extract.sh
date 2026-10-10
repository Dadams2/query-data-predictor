#!/usr/bin/env bash
set -euo pipefail

# Downloads the query-only SkyServer extract read by tools/sdss_characterise.py:
# for each of 462 sessions, every query's text, type and result size (no result
# rows). It is pinned to the EDBT 2027 artifact commit (tag edbt2027-submission)
# and checked against its SHA-256.

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  echo "Usage: $0"
  echo "Writes data/skyserver_sessions.csv.gz (2.6 MB)."
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="06f2f6af8de2daf8a09f536fdd4843b456d12dd6"
URL="https://raw.githubusercontent.com/Dadams2/query-data-predictor/$COMMIT/data/skyserver_sessions.csv.gz"
SHA256="4bc660bf8606b95bee1cc252e60cf76227a6e28234a7a122e53c775a3154711e"
DEST="$ROOT_DIR/data/skyserver_sessions.csv.gz"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

if [[ -f "$DEST" && "$(sha256 "$DEST")" == "$SHA256" ]]; then
  echo "Already present: $DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp "$DEST.XXXXXX")"
trap 'rm -f "$TMP"' EXIT

echo "Downloading $URL"
curl -fsSL -o "$TMP" "$URL"

ACTUAL="$(sha256 "$TMP")"
if [[ "$ACTUAL" != "$SHA256" ]]; then
  echo "Checksum mismatch: expected $SHA256, got $ACTUAL" >&2
  exit 1
fi

mv "$TMP" "$DEST"
echo "Wrote $DEST"
