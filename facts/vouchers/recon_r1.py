"""Header-edge survey for the R1 column-assignment change.

Answers three questions with measurements rather than with assertions:

1. Does every listing format in the corpus print a column header from which
   column edges can be derived?
2. Do the header edges of one document drift between its own pages?
3. Where do the data tokens actually sit under each printed header, and how
   much empty space separates one column from the next?

``--survey`` prints the per-document detail the report needs.
``--corpus`` sweeps every detail listing and reports header availability,
drift and the narrowest corridor, so a format that cannot be read is found
before the parser depends on it and not after.

Reads PDFs and ``documents`` only; writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import census
import db
import geometry
import pdfplumber
from geometry import ColumnGrid, Label, TextRow


@dataclass
class DocumentGrid:
    """A document's grid together with what was measured building it.

    Attributes:
        path: Filesystem path of the PDF.
        grid: The derived grid, or None when no header matched.
        pages: Page count.
        headers: One ``(page, top)`` per page that printed a header.
        drift: Header disagreements found between pages.
        note: Why no grid could be built, when that is the case.
    """

    path: str
    grid: ColumnGrid | None
    pages: int
    headers: list[tuple[int, float]]
    drift: list[str]
    note: str | None = None


def read_grid(path: str, max_pages: int | None = None) -> DocumentGrid:
    """Derive one document's column grid and check it against every header.

    Args:
        path: Filesystem path to a PDF.
        max_pages: Stop scanning for extra headers after this many pages.
            The grid itself always comes from the first header found.

    Returns:
        The document's grid and the evidence gathered for it.
    """
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        first: tuple[geometry.HeaderSchema, list[Label], float, int] | None = None
        headers: list[tuple[int, float]] = []
        drift: list[str] = []
        all_rows: list[TextRow] = []
        header_tops: set[float] = set()

        limit = page_count if max_pages is None else min(page_count, max_pages)
        for number in range(1, limit + 1):
            rows = geometry.page_rows(pdf.pages[number - 1])
            all_rows.extend(rows)
            found = geometry.find_header(rows)
            if found is None:
                continue
            schema, labels, top = found
            headers.append((number, top))
            header_tops.add(top)
            if first is None:
                first = (schema, labels, top, number)
                continue
            drift.extend(_compare(first[1], labels, number))

        if first is None:
            return DocumentGrid(path, None, page_count, headers, drift, "no header matched any known format")

        schema, labels, _, header_page = first
        grid, note = geometry.build_grid(schema.name, labels, all_rows, header_tops, header_page)
        if grid is None:
            return DocumentGrid(path, None, page_count, headers, drift, note)
        grid.drift = drift
        return DocumentGrid(path, grid, page_count, headers, drift)


def _compare(reference: list[Label], other: list[Label], page: int) -> list[str]:
    """Report header labels that moved between two pages of one document.

    Args:
        reference: Labels from the document's first header.
        other: Labels from a later page's header.
        page: The later page's number.

    Returns:
        One message per label that differs by more than the tolerance.
    """
    out: list[str] = []
    by_key = {label.key: label for label in other}
    for label in reference:
        twin = by_key.get(label.key)
        if twin is None:
            out.append(f"page {page}: column {label.key!r} is absent from this page's header")
            continue
        for edge in ("x0", "x1"):
            delta = abs(getattr(label, edge) - getattr(twin, edge))
            if delta > geometry.DRIFT_TOLERANCE:
                out.append(
                    f"page {page}: {label.key}.{edge} moved {delta:.2f} pt "
                    f"({getattr(label, edge):.2f} -> {getattr(twin, edge):.2f})"
                )
    return out


def column_extents(path: str, grid: ColumnGrid, sample_pages: int = 3) -> dict[str, tuple[float, float, int]]:
    """Measure where data tokens actually sit under each column.

    Args:
        path: Filesystem path to the PDF.
        grid: The document's grid.
        sample_pages: How many pages of data rows to measure.

    Returns:
        Column key to ``(min x0, max x1, rows seen)``.
    """
    seen: dict[str, tuple[float, float, int]] = {}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:sample_pages]:
            for row in geometry.page_rows(page):
                if not geometry.is_data_row(row, grid.left_margin):
                    continue
                for char in row.chars:
                    if char["text"].isspace():
                        continue
                    index = grid.column_of((char["x0"] + char["x1"]) / 2.0)
                    if index >= len(grid.labels):
                        continue
                    key = grid.labels[index].key
                    low, high, count = seen.get(key, (char["x0"], char["x1"], 0))
                    seen[key] = (min(low, char["x0"]), max(high, char["x1"]), count + 1)
    return seen


def survey(paths: list[str]) -> None:
    """Print the per-document header-edge survey.

    Args:
        paths: PDFs to survey.
    """
    for path in paths:
        record = read_grid(path)
        print(f"\n### {path}")
        print(f"pages={record.pages}  headers_on_pages={[p for p, _ in record.headers]}")
        if record.grid is None:
            print(f"  NO GRID: {record.note}")
            continue
        grid = record.grid
        print(
            f"  schema={grid.schema}  method={grid.method}  header_page={grid.header_page}"
            f"  left_margin={grid.left_margin}  rows_measured={grid.rows_used}"
        )
        print(f"  {'column':18s} {'header x0':>10s} {'header x1':>10s} {'data x0':>10s} {'data x1':>10s}  label")
        for label, (low, high) in zip(grid.labels, grid.extents, strict=False):
            print(f"  {label.key:18s} {label.x0:10.2f} {label.x1:10.2f} {low:10.2f} {high:10.2f}  {label.text!r}")
        print(f"  {'boundary':32s} {'x':>10s} {'corridor':>10s}")
        for (left, right), boundary, corridor in zip(
            list(zip(grid.labels, grid.labels[1:], strict=False)),
            grid.boundaries,
            grid.corridors,
            strict=False,
        ):
            flag = "  <-- NARROW" if corridor < geometry.MIN_CORRIDOR else ""
            print(f"  {left.key + '|' + right.key:32s} {boundary:10.2f} {corridor:10.2f}{flag}")
        for message in grid.outside_header_band:
            print(f"  OUTSIDE HEADER BAND {message}")
        for message in record.drift:
            print(f"  DRIFT {message}")
        if not record.drift and len(record.headers) > 1:
            print(f"  drift: none across {len(record.headers)} printed headers")


def corpus_sweep(limit: int | None, out_path: str | None) -> int:
    """Sweep every detail listing for header availability and drift.

    Args:
        limit: Stop after this many artifacts, or None for all.
        out_path: Where to write the JSON findings.

    Returns:
        Number of listings with no derivable header.
    """
    artifacts = census.merge(census.db_artifacts(db.query_dicts), census.disk_artifacts())
    listings = [a for a in artifacts if a.exclusion is None and a.resolved_path and a.doc_class == "detail_listing"]
    if limit:
        listings = listings[:limit]

    findings: list[dict] = []
    missing = 0
    for index, artifact in enumerate(listings, start=1):
        if index % 25 == 0:
            print(f"  {index}/{len(listings)}", file=sys.stderr)
        try:
            record = read_grid(artifact.resolved_path, max_pages=6)
        except Exception as exc:  # noqa: BLE001 - an unreadable PDF is data, not a crash
            findings.append(
                {
                    "path": artifact.resolved_path,
                    "meeting_date": artifact.meeting_date,
                    "fund": artifact.fund,
                    "status": "unreadable",
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )
            missing += 1
            continue
        entry = {
            "path": artifact.resolved_path,
            "meeting_date": artifact.meeting_date,
            "fund": artifact.fund,
            "pages": record.pages,
            "headers": len(record.headers),
            "drift": record.drift,
        }
        if record.grid is None:
            entry["status"] = "no_header"
            entry["note"] = record.note
            missing += 1
        else:
            entry["status"] = "ok"
            entry["schema"] = record.grid.schema
            entry["method"] = record.grid.method
            entry["rows_used"] = record.grid.rows_used
            entry["outside_header_band"] = record.grid.outside_header_band
            entry["columns"] = list(record.grid.keys)
            entry["narrowest_corridor"] = round(record.grid.narrowest_corridor, 3)
            entry["boundaries"] = [round(b, 2) for b in record.grid.boundaries]
        findings.append(entry)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=2)
    ok = [f for f in findings if f.get("status") == "ok"]
    drifted = [f for f in ok if f["drift"]]
    narrow = [f for f in ok if f.get("narrowest_corridor", 9) < geometry.MIN_CORRIDOR]
    print(
        f"listings={len(findings)} header_ok={len(ok)} no_header={missing} drifted={len(drifted)} narrow={len(narrow)}"
    )
    by_schema: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for entry in ok:
        by_schema[entry["schema"]] = by_schema.get(entry["schema"], 0) + 1
        by_method[entry["method"]] = by_method.get(entry["method"], 0) + 1
    print("by schema:", by_schema)
    print("by method:", by_method)
    for entry in findings:
        if entry.get("status") != "ok":
            print(f"  NO HEADER {entry['meeting_date']} {entry['fund']}: {entry.get('note')}  {entry['path']}")
    for entry in drifted:
        print(f"  DRIFT {entry['meeting_date']} {entry['fund']}: {entry['drift'][:3]}")
    for entry in narrow:
        print(f"  NARROW {entry['meeting_date']} {entry['fund']}: {entry['narrowest_corridor']} pt")
    return missing


def main() -> int:
    """Run the survey or the corpus sweep.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", nargs="*", help="PDF paths to survey in detail")
    parser.add_argument("--corpus", action="store_true", help="sweep every detail listing")
    parser.add_argument("--limit", type=int, help="stop the sweep after N listings")
    parser.add_argument("--out", help="write sweep findings to this JSON file")
    args = parser.parse_args()

    if args.survey:
        survey(args.survey)
    if args.corpus:
        corpus_sweep(args.limit, args.out)
    if not args.survey and not args.corpus:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
