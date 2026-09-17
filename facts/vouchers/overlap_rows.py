"""Count regex-vs-geometry disagreement on rows where glyph boxes overlap.

The two parse paths disagree on some rows, and a bare disagreement count is
not very informative: most of it is the regex path failing on layouts it was
never able to read. The interesting population is much narrower -- the rows
where a glyph box from one column overlaps a glyph box from the next, so
that no boundary *point* is outside both boxes and the midpoint rule is the
only thing deciding the split.

Those rows are where the R1 design decision actually bites. On 2026-03-25
ASB page 3 the final ``N`` of a truncated ``...HUDSONS BAR AN`` spans
255.685-262.379 and the first ``0`` of the check date spans 260.949-265.773:
an overlap of 1.43 pt. A box-straddle test would have rejected that row. The
midpoint test keeps it, and it carries 1,322.35 of a set that reconciles to
the cent.

This module reports how the two paths compare **on that population alone**,
per set. No threshold is applied and nothing here fails: it is a
measurement, and what it measures is how much work the midpoint rule is
doing.

Reads PDFs only. Touches no database, so it can be run with no credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import geometry as geo
import parsers
from locators import PdfText

# The largest glyph-box overlap measured anywhere in this corpus, on
# 2026-03-25 ASB page 3. Rows are counted when an overlap at a column
# boundary is greater than zero and no wider than this: a wider one would
# not be the truncation artifact this measures but a broken grid, and it
# would be a different finding.
#
# The comparison is made on the overlap rounded to two places, because 1.43
# is itself the rounded figure. The row it was measured from overlaps by
# 1.430069519999961, and an exact `<= 1.43` excluded the one row the bound
# was derived from -- a bound that rejects its own witness is not a bound.
MAX_OVERLAP_PT = 1.43


@dataclass
class SetOverlap:
    """Overlap and verdict counts for one listing.

    Attributes:
        label: Set label, derived from the file name.
        path: Absolute path to the PDF.
        data_rows: Data rows the geometry path read.
        overlap_rows: Rows carrying a boundary glyph overlap in range.
        widest: Widest overlap seen, in points.
        verdicts: Regex verdict counts over the overlap rows only.
        all_verdicts: Regex verdict counts over every data row.
        note: Why no grid could be built, when that is the case.
    """

    label: str
    path: str
    data_rows: int = 0
    overlap_rows: int = 0
    widest: float = 0.0
    verdicts: dict[str, int] = field(default_factory=lambda: {"agree": 0, "disagree": 0, "regex_miss": 0})
    all_verdicts: dict[str, int] = field(default_factory=lambda: {"agree": 0, "disagree": 0, "regex_miss": 0})
    note: str | None = None


def boundary_overlap(row: geo.TextRow, boundaries: tuple[float, ...]) -> float:
    """Return the widest in-range glyph overlap across a column boundary.

    For each boundary, the rightmost character whose midpoint falls left of
    it and the leftmost whose midpoint falls right of it are the two
    characters the boundary separates. When those two boxes overlap, no
    point on the x axis lies outside both, and the row can only be split by
    comparing midpoints.

    Args:
        row: A printed row.
        boundaries: The grid's column boundaries.

    Returns:
        The widest overlap in points, or 0.0 when none is in range.
    """
    chars = [c for c in row.chars if not c["text"].isspace()]
    if not chars:
        return 0.0
    widest = 0.0
    for boundary in boundaries:
        left = [c for c in chars if (c["x0"] + c["x1"]) / 2.0 < boundary]
        right = [c for c in chars if (c["x0"] + c["x1"]) / 2.0 >= boundary]
        if not left or not right:
            continue
        last = max(left, key=lambda c: c["x1"])
        first = min(right, key=lambda c: c["x0"])
        overlap = last["x1"] - first["x0"]
        if 0.0 < round(overlap, 2) <= MAX_OVERLAP_PT:
            widest = max(widest, overlap)
    return widest


def measure(path: str, meeting_date: str | None = None) -> SetOverlap:
    """Measure one listing.

    Args:
        path: Absolute path to a detail-listing PDF.
        meeting_date: ISO meeting date, used only as an era tie-break.

    Returns:
        The set's counts.
    """
    out = SetOverlap(label=os.path.basename(path), path=path)
    pdf = PdfText(path, geometry_too=True)
    grid = pdf.grid
    if grid is None:
        out.note = pdf.grid_note or "no column grid could be derived"
        return out

    era = parsers.detect_era(pdf.pages[0].text if pdf.pages else "", meeting_date)
    keys = grid.keys
    for page in pdf.pages:
        for placed in page.rows:
            row_geo, line = placed.row, placed.line
            if row_geo.top in page.header_tops:
                continue
            if not line.strip() or parsers.NOISE_RX.match(line) or parsers.TOTAL_LINE_RX.match(line):
                continue
            if not geo.is_data_row(row_geo, grid.left_margin):
                continue
            out.data_rows += 1
            parsed, problems = parsers.read_cells(geo.assign_row(row_geo, grid), keys)
            if problems:
                parsed.reason_code = parsers.COLUMN_AMBIGUOUS
            verdict = parsers._regex_verdict(line, era, parsed)
            out.all_verdicts[verdict] += 1
            overlap = boundary_overlap(row_geo, grid.boundaries)
            if overlap > 0.0:
                out.overlap_rows += 1
                out.widest = max(out.widest, overlap)
                out.verdicts[verdict] += 1
    return out


def main() -> int:
    """Measure every listing named on the command line.

    Returns:
        Process exit code. Always 0: this reports, it does not judge.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", help="detail-listing PDFs to measure")
    parser.add_argument("--date", help="ISO meeting date, used only as an era tie-break")
    args = parser.parse_args()

    results = []
    for path in args.pdfs:
        try:
            results.append(measure(path, args.date))
        except Exception as exc:  # noqa: BLE001 - a corrupt PDF is data, not a crash
            failed = SetOverlap(label=os.path.basename(path), path=path)
            failed.note = f"{type(exc).__name__}: {exc}"
            results.append(failed)

    width = max(len(r.label) for r in results)
    print(f"{'listing':{width}s}  rows  overlap  widest  overlap-row verdicts (agree/disagree/miss)")
    totals = {"rows": 0, "overlap": 0, "agree": 0, "disagree": 0, "regex_miss": 0}
    for r in results:
        if r.note:
            print(f"{r.label:{width}s}  -- {r.note}")
            continue
        v = r.verdicts
        print(
            f"{r.label:{width}s}  {r.data_rows:5d}  {r.overlap_rows:7d}  {r.widest:6.2f}  "
            f"{v['agree']}/{v['disagree']}/{v['regex_miss']}"
        )
        totals["rows"] += r.data_rows
        totals["overlap"] += r.overlap_rows
        for key in ("agree", "disagree", "regex_miss"):
            totals[key] += v[key]
    print()
    print(
        f"TOTAL data rows {totals['rows']}, of which {totals['overlap']} carry a boundary glyph overlap "
        f"of 0 < overlap <= {MAX_OVERLAP_PT} pt."
    )
    print(
        f"On those overlap rows the retired regex path agrees {totals['agree']}, "
        f"disagrees {totals['disagree']}, and cannot read {totals['regex_miss']}."
    )
    print("No threshold is applied. Geometry is authoritative; this is a measurement, not a gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
