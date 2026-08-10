#!/usr/bin/env bash
set -euo pipefail

for backup in /backups/*; do
  [ -f "$backup" ] || continue

  file="$(basename "$backup")"
  db="${file%.*}"

  case "$file" in
    *.sql)
      createdb --username "$POSTGRES_USER" "$db"
      psql --username "$POSTGRES_USER" --dbname "$db" --file "$backup"
      ;;
    *.dump|*.backup)
      createdb --username "$POSTGRES_USER" "$db"
      pg_restore --username "$POSTGRES_USER" --dbname "$db" --no-owner --no-acl "$backup"
      ;;
    *)
      echo "Skipping unsupported backup: $file"
      ;;
  esac
done
