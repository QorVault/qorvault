"""Attach ``documents`` ids to voucher rows whose files were not yet ingested.

44 voucher PDFs exist on disk with no ``documents`` row -- including every
file in the 2026 sets that carry the hard fixtures -- and operator-staged
packets arrive before the scrape that would collect them. Those rows are
written with ``locator_document_id IS NULL`` and ``source = 'staged_pdf'``,
never with a guessed id.

When ingest later produces a row for the same file, this script attaches it.
Matching is on **SHA-256 of the file content**, not on path or name: a path
changes when a directory is renamed, which has already happened once in this
corpus, and two differently named copies of the same listing are the same
evidence.

The script is idempotent. Running it twice changes nothing the second time,
and it only ever fills a NULL -- it never overwrites an id that is already
there, and it never touches a row outside schema ``facts``.
"""

from __future__ import annotations

import argparse

import census
import db

UPDATES = (
    (
        "facts.voucher_set",
        "locator_document_id",
        """
        UPDATE facts.voucher_set
           SET locator_document_id = %s, source_document_id = COALESCE(source_document_id, %s)
         WHERE locator_file_sha256 = %s AND locator_document_id IS NULL
        """,
    ),
    (
        "facts.voucher_line",
        "locator_document_id",
        """
        UPDATE facts.voucher_line
           SET locator_document_id = %s
         WHERE locator_file_sha256 = %s AND locator_document_id IS NULL
        """,
    ),
    (
        "facts.voucher_reconciliation",
        "register_document_id",
        """
        UPDATE facts.voucher_reconciliation
           SET register_document_id = %s
         WHERE register_file_sha256 = %s AND register_document_id IS NULL
        """,
    ),
    (
        "facts.voucher_parse_log",
        "document_id",
        """
        UPDATE facts.voucher_parse_log
           SET document_id = %s
         WHERE file_sha256 = %s AND document_id IS NULL
        """,
    ),
)


def digest_to_document_id() -> dict[str, str]:
    """Build a map from file digest to ``documents.id``.

    Returns:
        Mapping of SHA-256 to document id, for every voucher artifact that
        both resolves to a file and has a ``documents`` row.
    """
    artifacts = census.merge(census.db_artifacts(db.query_dicts), census.disk_artifacts())
    return {artifact.sha256: artifact.document_id for artifact in artifacts if artifact.sha256 and artifact.document_id}


def main() -> int:
    """Attach document ids by digest and report what changed.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    mapping = digest_to_document_id()
    print(f"{len(mapping)} voucher files have both a digest and a document id")

    pending = db.query_dicts(
        """
        SELECT locator_file_sha256 AS sha256, count(*) AS n
        FROM facts.voucher_set
        WHERE locator_document_id IS NULL
        GROUP BY 1
        """,
        None,
    )
    linkable = [row for row in pending if row["sha256"] in mapping]
    print(f"{len(pending)} distinct files behind unlinked sets; {len(linkable)} now have a document id")
    if args.dry_run:
        for row in linkable:
            print(f"  would link {row['sha256'][:12]}... -> {mapping[row['sha256']]}")
        return 0

    updated = dict.fromkeys((table for table, _, _ in UPDATES), 0)
    with db.connect() as conn:
        with conn.cursor() as cur:
            for digest, document_id in mapping.items():
                for table, column, sql in UPDATES:
                    params = (
                        (document_id, document_id, digest) if table == "facts.voucher_set" else (document_id, digest)
                    )
                    cur.execute(sql, params)
                    updated[table] += cur.rowcount
    for table, count in updated.items():
        print(f"  {table}: {count} row(s) linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
