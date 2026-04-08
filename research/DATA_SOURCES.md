# Research Data Sources

## Why data files are not in git

The JSON dataset files in `research/ospi_data/` and
`research/kent_school_data/` are excluded from version control. They are
downloaded reference datasets whose authoritative source is OSPI (Office
of Superintendent of Public Instruction) via Washington State's open data
portal at data.wa.gov. Storing them in git would bloat the repository
(several files exceed 100 MB) without benefit — the data can be
re-fetched at any time using the scripts below, and git is not the right
tool for caching external data.

The same applies to `manual_extraction/` which contains large PDF files
from board meetings. These are backed up to a separate NAS and are not
suitable for git.

## How to restore data files after a fresh clone

### OSPI Report Card data (`research/ospi_data/`)

Downloads enrollment, graduation, assessment, attendance, discipline,
growth, teacher demographics, teacher experience, SQSS, and WaKIDS
datasets from data.wa.gov filtered to Kent School District:

```bash
cd ~/ksd-boarddocs-rag
python3 research/ospi_data/download_ospi.py
```

Output: one JSON file per category in `research/ospi_data/`.

### Extended Kent data (`research/kent_school_data/`)

Downloads additional datasets via Socrata catalog search, including
highly capable, school improvement, and discovered datasets:

```bash
cd ~/ksd-boarddocs-rag
python3 scripts/download_kent_data.py
```

Output: organized JSON files under `research/kent_school_data/`.

### Ingesting into PostgreSQL

After downloading, load the data into the RAG pipeline:

```bash
cd ~/ksd-boarddocs-rag
document_processor/venv/bin/python scripts/ingest_ospi_data.py
document_processor/venv/bin/python scripts/ingest_new_ospi_data.py
```

## Data update frequency

OSPI publishes updated Report Card data annually, typically in the fall
after the prior school year ends. The download scripts can be re-run at
any time to fetch the latest data — they use `ON CONFLICT DO NOTHING`
for idempotent ingestion.

## Files excluded from git

The following patterns are in `.gitignore`:

- `research/ospi_data/*.json` — downloaded OSPI datasets
- `research/kent_school_data/` — downloaded Kent-specific datasets
- `manual_extraction/` — large PDF files, backed up to a separate NAS
