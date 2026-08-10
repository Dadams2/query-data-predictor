#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  echo "Usage: $0 DB_NAME"
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DB_NAME"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="$ROOT_DIR/docker/postgres/backups"
DB_NAME="$1"

set -a
[ ! -f "$ROOT_DIR/.env" ] || source "$ROOT_DIR/.env"
set +a

mkdir -p "$BACKUP_DIR"

PGHOST="${PG_HOST:-localhost}" \
PGPORT="${PG_PORT:-5432}" \
PGUSER="${PG_DATA_USER:-postgres}" \
PGPASSWORD="${PG_SESSION_PASSWORD:-}" \
pg_dump --format=custom --no-owner --no-acl --dbname "$DB_NAME" --file "$BACKUP_DIR/$DB_NAME.dump"

echo "Wrote $BACKUP_DIR/$DB_NAME.dump"
