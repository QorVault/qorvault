# BoardDocs Data Loader

Ingests Kent School District BoardDocs meeting records into PostgreSQL from scraped data directories. Handles three meeting formats automatically: structured (item subdirectories), flat (agenda.txt), and empty (meeting.json only).

## Install

```bash
cd ~/ksd-boarddocs-rag/boarddocs_loader
pip install -e ".[dev]"
```

## Usage

```bash
# Format report — see how many meetings of each type
python -m boarddocs_loader --format-report

# Dry run — parse and validate without writing to DB
python -m boarddocs_loader --dry-run --limit 10

# Full load
python -m boarddocs_loader \
    --data-dir /home/qorvault/projects/ksd_forensic/boarddocs/data \
    --pg-dsn "postgresql://boarddocs:CHANGE_ME_ON_FIRST_DEPLOY@localhost:5432/boarddocs" \
    --tenant kent_sd

# Verbose mode (DEBUG logging)
python -m boarddocs_loader --dry-run --verbose --limit 5
```

## CLI Options

| Flag | Description |
|---|---|
| `--data-dir` | Path to meeting data root (default: `/home/qorvault/projects/ksd_forensic/boarddocs/data`) |
| `--pg-dsn` | PostgreSQL connection string |
| `--tenant` | Tenant ID (default: `kent_sd`) |
| `--dry-run` | Parse without writing to database |
| `--limit N` | Process only first N meetings |
| `--verbose` | Enable DEBUG logging |
| `--format-report` | Print format counts and exit |

## Meeting Formats

- **Structured**: Item subdirectories with `item.json` + attachments. Creates `agenda_item` records.
- **Flat**: `agenda.txt` + `agenda.html` + loose attachments. Creates `agenda` records.
- **Empty**: Only `meeting.json`. Creates `agenda` record with `processing_status='complete'`.

## Tests

```bash
pytest -v
```
