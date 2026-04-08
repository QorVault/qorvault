"""CLI entry point: python -m boarddocs_loader."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .db import Database
from .loader import DataLoader


def setup_logging(verbose: bool) -> None:
    """Configure logging to stdout and optionally a log file."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_dir = Path("/var/log/boarddocs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "loader.log"))
    except PermissionError:
        pass  # Skip file logging if no write access

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="boarddocs_loader",
        description="Load BoardDocs meeting data into PostgreSQL",
    )
    parser.add_argument("--data-dir", type=Path, help="Path to meeting data root")
    parser.add_argument("--pg-dsn", type=str, help="PostgreSQL connection string")
    parser.add_argument("--tenant", type=str, help="Tenant ID (default: kent_sd)")
    parser.add_argument("--dry-run", action="store_true", help="Parse without DB writes")
    parser.add_argument("--limit", type=int, default=0, help="Process only N meetings")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--format-report", action="store_true", help="Print format counts and exit")

    args = parser.parse_args()
    settings = Settings()

    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.pg_dsn:
        settings.pg_dsn = args.pg_dsn
    if args.tenant:
        settings.tenant = args.tenant
    if args.dry_run:
        settings.dry_run = True
    if args.limit:
        settings.limit = args.limit
    if args.verbose:
        settings.verbose = True
    if args.format_report:
        settings.format_report = True

    setup_logging(settings.verbose)
    logger = logging.getLogger(__name__)

    if not settings.data_dir.exists():
        logger.error("Data directory does not exist: %s", settings.data_dir)
        sys.exit(1)

    loader = DataLoader(
        data_dir=settings.data_dir,
        tenant=settings.tenant,
        dry_run=settings.dry_run,
        limit=settings.limit,
        verbose=settings.verbose,
    )

    if settings.format_report:
        stats = loader.format_report()
        print("\nFormat Report")
        print(f"{'='*40}")
        print(f"Total meetings:  {stats.meetings_processed}")
        print(f"  Structured:    {stats.structured_count}")
        print(f"  Flat:          {stats.flat_count}")
        print(f"  Empty:         {stats.empty_count}")
        sys.exit(0)

    # Connect to database unless dry-run
    db = None
    if not settings.dry_run:
        db = Database(settings.pg_dsn)
        try:
            db.connect()
        except Exception as e:
            logger.error("Database connection failed: %s", e)
            sys.exit(1)
        loader.db = db

    try:
        stats = loader.run()
    finally:
        if db:
            db.close()

    print("\nLoad Complete")
    print(f"{'='*40}")
    print(f"Meetings processed:     {stats.meetings_processed}")
    print(f"  Structured:           {stats.structured_count}")
    print(f"  Flat:                 {stats.flat_count}")
    print(f"  Empty:                {stats.empty_count}")
    print(f"Agenda item records:    {stats.agenda_items_created}")
    print(f"Agenda records:         {stats.agenda_records_created}")
    print(f"Attachment records:     {stats.attachment_records_created}")
    print(f"Skipped (already in DB):{stats.skipped}")
    print(f"Errors:                 {stats.errors}")
    print(f"Elapsed:                {stats.elapsed:.1f}s")

    if stats.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
