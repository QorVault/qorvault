"""CLI entry point: python -m embedding_pipeline."""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import parse_args
from .pipeline import run


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    stats = asyncio.run(
        run(
            dsn=args.dsn,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            tenant=args.tenant,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    )

    print("\nEmbedding Complete", file=sys.stderr)
    print(f"{'=' * 40}", file=sys.stderr)
    print(f"Chunks embedded:   {stats.total_embedded}", file=sys.stderr)
    print(f"Failed:            {stats.total_failed}", file=sys.stderr)
    print(f"Batches:           {stats.batches_processed}", file=sys.stderr)

    if stats.total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
