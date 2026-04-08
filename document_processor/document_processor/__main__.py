"""CLI entry point: python -m document_processor."""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import parse_args
from .processor import run


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    stats = asyncio.run(
        run(
            dsn=args.dsn,
            ocr_url=args.ocr_url,
            tenant=args.tenant,
            workers=args.workers,
            dry_run=args.dry_run,
            document_type=args.document_type,
        )
    )

    print("\nProcessing Complete")
    print(f"{'=' * 40}")
    print(f"Total documents:   {stats.total}")
    print(f"Completed:         {stats.completed}")
    print(f"Failed:            {stats.failed}")
    print(f"Chunks created:    {stats.chunks_created}")

    if stats.errors:
        print(f"\nFailed documents ({len(stats.errors)}):")
        for err in stats.errors[:20]:
            print(f"  - {err}")

    if stats.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
