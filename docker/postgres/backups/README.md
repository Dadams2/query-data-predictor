# Local Postgres Backups

Put logical backups here before first `docker compose up`.

- `sdss.sql` restores into database `sdss`
- `simba_sdss.dump` restores into database `simba_sdss`

Backups are ignored by git. Reset the Docker volume to rerun restores:

```bash
docker compose down -v
docker compose up -d
```
