"""Column assignment by page geometry.

The regex row parsers read a line of text. Text has no column boundaries in
it: when the accounting system prints an invoice amount with no cents
immediately followed by a description that starts with a digit, the two
columns become one token and no regular expression over that text can tell
where one ends and the other begins. That defect (R1) cost six HARD
fixtures on the 2026-06-24 and 2026-07-22 packets, and on 2026-07-22 the
whole ACH listing, because that packet is rendered with no inter-column
whitespace at all.

The characters themselves are not ambiguous. ``pdfplumber`` reports an x0
and an x1 for every glyph, and the accounting system lays every column out
at the same x on every row of a listing. This module reads those
coordinates and decides, per character, which printed column it belongs to.

How a boundary is decided
-------------------------

1. **The printed header names the columns and fixes their order.** Header
   labels are found by matching the page's header text against an ordered
   vocabulary per listing format. A document whose header matches no known
   format gets no grid at all, and its rows are recorded as failures
   rather than parsed by guesswork.

2. **The boundary itself is the empty corridor between two columns' own
   printed runs**, measured over every well-formed data row of the
   document. Phase 0 established that the header's x-extents cannot fix
   the point, because label alignment inside the column is not consistent
   between formats: on 2026-07-22 ACH the ``Amount`` label's right edge
   sits within 0.2 pt of the amounts beneath it, while on 2026-03-25 ASB
   the same label sits about 30 pt to the *left* of its own data, entirely
   outside the band between that label and the next. Every rule that took
   a fixed side of the header band -- left edge, right edge or midpoint --
   was measured to fall inside real data on at least one format present in
   this corpus. The corridor is measured, not chosen: it is the gap
   between the rightmost character the left column prints anywhere in the
   document and the leftmost character the right column prints anywhere.
   Where the header band and the measured corridor disagree, the
   disagreement is recorded on the grid rather than averaged away.

3. **A page with no derivable header inherits the grid of the last header
   above it.** 2026-07-22 prints its header on page 1 only. The grid is
   therefore a property of the document, and every header a document does
   print is checked against it: drift beyond ``DRIFT_TOLERANCE`` is
   recorded, never averaged away.

Why characters are assigned by midpoint
---------------------------------------

Glyph boxes in this corpus overlap their neighbours. Inside a single word
the overlap is around 0.05 pt, but where the accounting system truncates a
vendor name hard against the next column it is much larger: on
2026-03-25 ASB page 3 the final ``N`` of ``...HUDSONS BAR AN`` spans
255.685-262.379 and the first ``0`` of the check date spans 260.949-265.773,
an overlap of 1.43 pt. No boundary point exists that is outside both boxes.
Assignment is therefore by the character's midpoint, and a character is
ambiguous when its *midpoint* lies within ``MIDPOINT_CLEARANCE`` of a
boundary. Testing the whole box instead would reject that row, and that row
is 1,322.35 of a set that reconciles to the cent today.

Nothing here uses an LLM, and no x coordinate is hard-coded: every number
in this module is a tolerance in points, stated and justified above or at
its definition.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field

# Characters on the same printed row never differ in ``top`` by more than a
# fraction of a line. Measured line pitch in this corpus is 5.0-7.4 pt, so
# 1.5 pt groups a row without ever merging two.
ROW_TOLERANCE = 1.5

# A gap wider than this between two glyph boxes starts a new run. Inside a
# word, boxes touch or overlap; between words the accounting system leaves
# at least a space. 1.0 pt is below the narrowest space glyph measured
# (1.25 pt) and above the widest intra-word overlap measured (0.08 pt).
RUN_GAP = 1.0

# A character whose midpoint falls this close to a boundary is not safely
# on either side of it. The tightest real clearance measured is 1.06 pt
# (2026-03-25 ASB page 3, where a truncated vendor name runs into the check
# date), so 0.5 pt flags genuine ambiguity without rejecting the tightest
# row this corpus actually prints. Across 482,395 rows it fires 3 times.
MIDPOINT_CLEARANCE = 0.5

# Two headers of the same document must agree to this. Phase 0 measured
# zero drift on every multi-header document in the corpus; 2.0 pt is the
# operator's stated threshold, kept as the alarm rather than tightened to
# the measurement.
DRIFT_TOLERANCE = 2.0

# A boundary corridor narrower than this is reported rather than trusted.
# The narrowest corridor measured anywhere in the corpus is 1.14 pt
# (2024-12-11 ASB); the next narrowest is 2.11 pt and the median is 5.4 pt.
# Nothing in the corpus sits near 0.6, so this is an alarm rather than a
# threshold the data pushes against.
MIN_CORRIDOR = 0.6

# A grid needs at least this many rows that print one run per column. One
# is enough and is not a compromise: a listing with a single data row has
# exactly one row to measure, and that row's own corridors are the correct
# ones for it. 2026-08-26 Trust and every Custodial listing in the corpus
# print two rows.
MIN_SAMPLE_ROWS = 1

# Rows whose leftmost glyph starts this far from the listing's left margin
# are not data rows. Wrapped description lines, page furniture and the
# subtotal lines Era B and C print in the middle of the page are all
# excluded by it.
MARGIN_TOLERANCE = 2.0


@dataclass(frozen=True)
class Word:
    """One run of characters with no internal gap.

    Attributes:
        text: The characters, in x order.
        x0: Left edge of the first glyph box.
        x1: Right edge of the last glyph box.
    """

    text: str
    x0: float
    x1: float


@dataclass(frozen=True)
class TextRow:
    """One printed row of a page, with its characters.

    Attributes:
        top: Distance from the top of the page to the row.
        chars: Character dicts as ``pdfplumber`` reports them, in x order.
        text: The characters concatenated, with no separators added.
    """

    top: float
    chars: tuple[dict, ...]
    text: str

    @property
    def x0(self) -> float:
        """Left edge of the row's leftmost glyph, or 0.0 when empty."""
        return self.chars[0]["x0"] if self.chars else 0.0

    @property
    def x1(self) -> float:
        """Right edge of the row's rightmost glyph, or 0.0 when empty."""
        return max((c["x1"] for c in self.chars), default=0.0)


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a listing format.

    Attributes:
        key: Canonical name used by the parser and the fact tables.
        labels: Printed header spellings, longest first.
        kind: ``text``, ``date``, ``digits`` or ``amount``; decides the
            type check the column's own content must pass.
        required: Whether a data row must carry usable content here, and
            whether the printed header must name the column at all. Some
            listings print no description column: 2026-02-11 Trust ends at
            the invoice amount, and 2022-03-09 ASB does the same.
    """

    key: str
    labels: tuple[str, ...]
    kind: str
    required: bool = True


@dataclass(frozen=True)
class HeaderSchema:
    """An ordered set of columns that one listing format prints.

    Attributes:
        name: Identifier recorded on the grid.
        columns: The columns, left to right.
    """

    name: str
    columns: tuple[ColumnSpec, ...]


# The three printed layouts. Column ORDER is the key, not the wording:
# Phase 0 found 55 distinct header spellings for four actual layouts
# because the header wraps differently at different page widths.
SCHEMAS: tuple[HeaderSchema, ...] = (
    HeaderSchema(
        name="vendor_first",
        columns=(
            ColumnSpec("vendor", ("Vendor Name", "Vendor"), "text"),
            ColumnSpec("check_date", ("Check Date",), "date"),
            ColumnSpec("check_number", ("Check Number", "Check No.", "Check #"), "digits"),
            ColumnSpec("check_amount", ("Check Amount", "Check Amt", "Amount"), "amount"),
            ColumnSpec("invoice_amount", ("Invoice Amount", "Invoice Amt", "Inv Amt", "Amount"), "amount"),
            ColumnSpec(
                "description",
                ("Invoice Description", "Work Performed", "Description"),
                "text",
                required=False,
            ),
        ),
    ),
    HeaderSchema(
        name="check_first",
        columns=(
            # The 2008 header names its two money columns the other way
            # round from the 2017 one. The POSITIONS are what this layer
            # keeps: fifth column to check_amount, sixth to invoice_amount,
            # exactly as the regex parser this replaces did, so historical
            # figures do not silently change meaning. Whether the 2008
            # wording means the district intended the reverse is recorded
            # as an open item, not decided here.
            ColumnSpec("check_number", ("Check No.", "Check #"), "digits"),
            ColumnSpec("vendor", ("Vendor Name", "Vendor"), "text"),
            ColumnSpec("check_date", ("Date",), "date"),
            ColumnSpec("description", ("Description",), "text", required=False),
            ColumnSpec("check_amount", ("Invoice", "Inv. Amt."), "amount"),
            ColumnSpec("invoice_amount", ("Inv. Total", "Total Amt."), "amount"),
        ),
    ),
    HeaderSchema(
        name="voucher_first",
        columns=(
            ColumnSpec("check_number", ("Voucher Number",), "digits"),
            ColumnSpec("vendor", ("Vendor Name", "Vendor"), "text"),
            ColumnSpec("invoice_amount", ("Voucher Amount", "Amount"), "amount"),
            ColumnSpec("description", ("Description",), "text", required=False),
        ),
    ),
)

# Key of the synthetic column that holds everything printed to the right of
# the last column this layer reads.
TRAILING_KEY = "trailing"

# How far a boundary is kept from the column it must not cut into, when one
# of the two columns is elastic toward the other. Larger than
# MIDPOINT_CLEARANCE so a hugged boundary still leaves a character's
# midpoint unambiguous, and smaller than half the narrowest corridor
# measured in the corpus (2.36 pt).
HUG_MARGIN = 1.0

# Spread of a column's run edges, in points, below which that edge is taken
# to be fixed rather than elastic. Measured edge spread inside one document
# is either under 0.5 pt or many points; nothing in the corpus sits near
# this value.
ALIGNMENT_SPREAD = 1.0

# A header row always contains one of these words. Used only to prune the
# rows a header search looks at; it decides nothing.
HEADER_HINT_RX = re.compile(r"amount|description|vendor|voucher|check", re.I)

# A row carrying a printed money value is data, never a header.
DATA_HINT_RX = re.compile(r"\d[\d,]*\.\d{2}")


@dataclass(frozen=True)
class Label:
    """A matched header label with the extent of its printed text.

    Attributes:
        key: Canonical column name.
        text: Header text as printed, joined across wrapped rows.
        x0: Left edge of the label.
        x1: Right edge of the label.
    """

    key: str
    text: str
    x0: float
    x1: float


@dataclass
class ColumnGrid:
    """Column boundaries for one document, with the evidence behind them.

    Attributes:
        schema: Name of the matched listing format.
        labels: Matched header labels, left to right.
        boundaries: One x per adjacent label pair.
        corridors: Width of the unoccupied corridor each boundary sits in.
        header_page: Page the grid's header was read from.
        left_margin: Modal left edge of the listing's data rows.
        extents: Measured ``(x0, x1)`` of each column's printed runs.
        rows_used: Well-formed rows the corridors were measured from.
        outside_header_band: Boundaries that fell outside the band between
            the header labels they separate.
        method: ``runs`` when the corridors were measured between whole
            column runs, ``header_band`` when the renderer fragmented its
            amounts and the corridors had to be searched inside the header
            bands instead.
        drift: Recorded disagreements between headers of the same document.
    """

    schema: str
    labels: tuple[Label, ...]
    boundaries: tuple[float, ...]
    corridors: tuple[float, ...]
    header_page: int
    left_margin: float
    extents: tuple[tuple[float, float], ...] = ()
    rows_used: int = 0
    outside_header_band: list[str] = field(default_factory=list)
    method: str = "runs"
    drift: list[str] = field(default_factory=list)

    @property
    def keys(self) -> tuple[str, ...]:
        """Canonical column names, left to right."""
        return tuple(label.key for label in self.labels)

    def column_of(self, midpoint: float) -> int:
        """Return the index of the column a midpoint falls in.

        Args:
            midpoint: x midpoint of a glyph box.

        Returns:
            Index into ``labels``.
        """
        return bisect_left(self.boundaries, midpoint)

    @property
    def narrowest_corridor(self) -> float:
        """Width of the tightest boundary corridor on this grid."""
        return min(self.corridors) if self.corridors else 0.0


def page_rows(page) -> list[TextRow]:
    """Group a page's characters into printed rows.

    Args:
        page: A ``pdfplumber`` page.

    Returns:
        Rows in printed order, each with its characters in x order.
    """
    buckets: list[tuple[float, list[dict]]] = []
    for char in sorted(page.chars, key=lambda c: (c["top"], c["x0"])):
        if buckets and abs(char["top"] - buckets[-1][0]) <= ROW_TOLERANCE:
            buckets[-1][1].append(char)
        else:
            buckets.append((char["top"], [char]))
    rows = []
    for top, chars in buckets:
        ordered = tuple(sorted(chars, key=lambda c: c["x0"]))
        rows.append(TextRow(top, ordered, "".join(c["text"] for c in ordered)))
    return rows


def row_words(row: TextRow) -> list[Word]:
    """Split a row into runs separated by a gap or by a space character.

    Args:
        row: The row to split.

    Returns:
        Words in x order. Whitespace-only runs are dropped.
    """
    words: list[Word] = []
    current: list[dict] = []
    for char in row.chars:
        if char["text"].isspace():
            if current:
                words.append(_word(current))
                current = []
            continue
        if current and char["x0"] - current[-1]["x1"] > RUN_GAP:
            words.append(_word(current))
            current = []
        current.append(char)
    if current:
        words.append(_word(current))
    return words


def row_runs(row: TextRow) -> list[Word]:
    """Split a row into runs separated only by an x gap.

    Unlike :func:`row_words` this does **not** break on a space character.
    A space inside a description is part of that column's content; the only
    thing that separates one column from the next is empty space on the
    page. This is the segmentation the column corridors are measured from,
    and it is why 2026-07-22 -- whose extracted *text* has no gaps between
    columns at all -- still segments cleanly here.

    Args:
        row: The row to split.

    Returns:
        Runs in x order, with outer whitespace stripped from the text.
        Runs that hold nothing but whitespace are dropped.
    """
    runs: list[Word] = []
    current: list[dict] = []
    reach = float("-inf")
    for char in row.chars:
        if current and char["x0"] - reach > RUN_GAP:
            runs.append(_word(current))
            current = []
            reach = float("-inf")
        current.append(char)
        reach = max(reach, char["x1"])
    if current:
        runs.append(_word(current))
    return [run for run in runs if run.text]


def _word(chars: list[dict]) -> Word:
    """Build a Word from a list of character dicts.

    Leading and trailing whitespace characters are dropped before the
    extent is taken: a padding space carries a real x range in this corpus
    and would otherwise widen the column it sits at the edge of.

    Args:
        chars: Characters in x order.

    Returns:
        The word, with whitespace trimmed from both ends.
    """
    start, end = 0, len(chars)
    while start < end and chars[start]["text"].isspace():
        start += 1
    while end > start and chars[end - 1]["text"].isspace():
        end -= 1
    trimmed = chars[start:end]
    if not trimmed:
        return Word("", chars[0]["x0"], chars[0]["x0"])
    return Word("".join(c["text"] for c in trimmed), trimmed[0]["x0"], max(c["x1"] for c in trimmed))


def _normalise(text: str) -> str:
    """Fold a header label for comparison.

    Whitespace and full stops are removed rather than collapsed. The
    accounting system writes the same label as ``Check #``, ``Check#``,
    ``Check No.`` and ``CHECK NO``, and kerning splits ``Work performed``
    into ``Work perfor`` and ``med``; none of those differences mean
    anything.

    Args:
        text: Header text as printed.

    Returns:
        Lower-cased text with every space and full stop removed.
    """
    return re.sub(r"[\s.]+", "", text).lower()


def _stacks(block: list[TextRow]) -> list[Label]:
    """Merge vertically stacked header words into one label each.

    A header that wraps prints ``Check`` on one row and ``Amount`` on the
    next, over the same column. Those two words overlap in x and belong to
    one label; ``Check`` and ``Date`` on a single row do not overlap and are
    two labels until the schema match joins them.

    Args:
        block: The rows making up the header.

    Returns:
        One label per x-overlapping group, left to right, with ``key``
        still unset.
    """
    entries: list[tuple[int, Word]] = []
    for index, row in enumerate(block):
        for word in row_words(row):
            entries.append((index, word))
    entries.sort(key=lambda item: item[1].x0)

    groups: list[list[tuple[int, Word]]] = []
    for index, word in entries:
        placed = False
        for group in groups:
            if any(word.x0 < other.x1 and other.x0 < word.x1 for _, other in group):
                group.append((index, word))
                placed = True
                break
        if not placed:
            groups.append([(index, word)])

    labels: list[Label] = []
    for group in groups:
        group.sort(key=lambda item: (item[0], item[1].x0))
        text = " ".join(word.text.strip() for _, word in group if word.text.strip())
        labels.append(
            Label(
                key="",
                text=text,
                x0=min(word.x0 for _, word in group),
                x1=max(word.x1 for _, word in group),
            )
        )
    labels.sort(key=lambda label: label.x0)
    return labels


def match_schema(stacks: list[Label]) -> tuple[HeaderSchema, list[Label]] | None:
    """Match a header's word stacks against the known listing formats.

    Args:
        stacks: Header word stacks, left to right.

    Returns:
        ``(schema, labels)`` where labels carry canonical keys and cover
        every column of the schema, or None when no format matches.
    """
    best: tuple[HeaderSchema, list[Label]] | None = None
    for schema in SCHEMAS:
        matched = _match_one(schema, stacks)
        if matched is None:
            continue
        if best is None or len(matched) > len(best[1]):
            best = (schema, matched)
    return best


def _match_one(schema: HeaderSchema, stacks: list[Label]) -> list[Label] | None:
    """Consume stacks left to right against one schema.

    Args:
        schema: The format to try.
        stacks: Header word stacks.

    Returns:
        Labels with canonical keys, or None when the schema does not match.
    """
    out: list[Label] = []
    position = 0
    for spec in schema.columns:
        consumed = _consume(spec, stacks, position)
        if consumed is None:
            if spec.required:
                return None
            # A column the format does not print at all is not a failure to
            # read the header; it is a listing with one fewer column.
            continue
        label, position = consumed
        out.append(label)
    # Era B and Era C print eight account-code columns and two more
    # description columns to the right of the ones this layer reads. They
    # are collapsed into a single trailing column: their own boundaries are
    # never needed, but their left edge is, because without it a long
    # description would bleed into them.
    leftovers = stacks[position:]
    if leftovers:
        out.append(
            Label(
                key=TRAILING_KEY,
                text=" ".join(stack.text for stack in leftovers)[:80],
                x0=min(stack.x0 for stack in leftovers),
                x1=max(stack.x1 for stack in leftovers),
            )
        )
    return out


def _consume(spec: ColumnSpec, stacks: list[Label], position: int) -> tuple[Label, int] | None:
    """Match one column spec against the next few stacks.

    Args:
        spec: The column to match.
        stacks: Header word stacks.
        position: Index of the next unconsumed stack.

    Returns:
        ``(label, next_position)``, or None when nothing matches here.
    """
    for label_text in spec.labels:
        wanted = _normalise(label_text)
        for span in range(1, 5):
            if position + span > len(stacks):
                break
            group = stacks[position : position + span]
            joined = _normalise(" ".join(item.text for item in group))
            if joined == wanted:
                return (
                    Label(
                        key=spec.key,
                        text=" ".join(item.text for item in group),
                        x0=min(item.x0 for item in group),
                        x1=max(item.x1 for item in group),
                    ),
                    position + span,
                )
    return None


def find_header(rows: list[TextRow]) -> tuple[HeaderSchema, list[Label], float] | None:
    """Locate the printed column header on one page.

    Args:
        rows: The page's rows in printed order.

    Returns:
        ``(schema, labels, top)`` for the best-matching header, or None.
    """
    best: tuple[HeaderSchema, list[Label], float] | None = None
    pitch = _line_pitch(rows)
    for index, row in enumerate(rows):
        if not HEADER_HINT_RX.search(row.text) or DATA_HINT_RX.search(row.text):
            continue
        for block in _header_blocks(rows, index, pitch):
            matched = match_schema(_stacks(block))
            if matched is None:
                continue
            schema, labels = matched
            if best is None or len(labels) > len(best[1]):
                best = (schema, labels, row.top)
    return best


def _header_blocks(rows: list[TextRow], index: int, pitch: float) -> list[list[TextRow]]:
    """Enumerate the row groups a wrapped header might occupy.

    A header wraps onto as many as three printed rows and the wrapped part
    can sit above the spine, below it, or both: 2026-08-26 Trust prints
    ``Check`` above and ``Amount`` below, while 2026-06-24 GF prints
    ``Check`` above and ``Number`` below. Rather than guess which, every
    contiguous group around the spine is offered to the schema match and
    the group that matches the most columns wins.

    Args:
        rows: The page's rows.
        index: Index of the spine row.
        pitch: Median vertical distance between printed rows.

    Returns:
        Candidate blocks, each in printed order.
    """
    spine = rows[index]
    above: list[TextRow] = []
    below: list[TextRow] = []
    for step, sink in ((-1, above), (1, below)):
        cursor = index + step
        while 0 <= cursor < len(rows) and len(sink) < 2:
            candidate = rows[cursor]
            if abs(candidate.top - spine.top) > pitch * 2.4:
                break
            # A row with money in it is data. A row reaching left of the
            # spine's own left margin is a rule or a full-width title, not
            # a wrapped header cell. A row reaching further RIGHT is not
            # excluded: 2026-02-11 Trust prints no description column, so
            # its wrapped "Amount" cells sit right of everything on the
            # spine row.
            if DATA_HINT_RX.search(candidate.text):
                break
            if candidate.x0 < spine.x0 - MARGIN_TOLERANCE:
                break
            sink.append(candidate)
            cursor += step
    blocks: list[list[TextRow]] = []
    for up in range(len(above) + 1):
        for down in range(len(below) + 1):
            block = [*above[:up], spine, *below[:down]]
            blocks.append(sorted(block, key=lambda row: row.top))
    return blocks


def _line_pitch(rows: list[TextRow]) -> float:
    """Estimate the vertical distance between consecutive printed rows.

    Args:
        rows: The page's rows.

    Returns:
        Median gap in points, or 8.0 when the page has too few rows.
    """
    gaps = sorted(b.top - a.top for a, b in zip(rows, rows[1:], strict=False) if 0 < b.top - a.top < 40)
    if not gaps:
        return 8.0
    return gaps[len(gaps) // 2]


def left_margin(rows: list[TextRow], header_tops: set[float], read_columns: int, cut: float) -> float | None:
    """Find the x the listing's data rows start at.

    Only rows that print one run per column vote. Counting every row was
    tried and is wrong: the 2017-01-25 ASB listing wraps more descriptions
    than it has data rows, so the modal left edge of *all* rows is the
    description column and every data row is then excluded from the
    measurement.

    Args:
        rows: Every row of the document.
        header_tops: Tops of rows that belong to a header.
        read_columns: Columns this layer reads, excluding trailing ones.
        cut: x separating read columns from trailing ones.

    Returns:
        The modal left edge, or None when no row is well formed.
    """
    counts: dict[float, int] = {}
    for row in rows:
        if not row.chars or row.top in header_tops:
            continue
        if len([run for run in row_runs(row) if run.x0 < cut]) != read_columns:
            continue
        key = round(row.x0, 1)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def modal_margin(rows: list[TextRow], header_tops: set[float]) -> float | None:
    """Return the modal left edge of every row that is not a header.

    Used only when no row of the document segments into one run per
    column, which happens where the renderer fragments its amounts.

    Args:
        rows: Every row of the document.
        header_tops: Tops of rows that belong to a header.

    Returns:
        The modal left edge, or None when there are no rows.
    """
    counts: dict[float, int] = {}
    for row in rows:
        if not row.chars or row.top in header_tops:
            continue
        key = round(row.x0, 1)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def band_boundaries(labels: list[Label], samples: list[TextRow]) -> Corridors:
    """Place each boundary in the widest gap inside its header band.

    The fallback for a document whose renderer fragments its amounts so
    badly that no row segments into one run per column: 2022-11-09 ASB
    prints ``$     1 ,120.00`` as three separately positioned pieces, and
    the gaps *inside* one amount there are wider than the gap between two
    columns. Run counting cannot work on such a page; character midpoints
    still can, and the header band bounds the search so the corridor
    chosen is one that separates two named columns rather than two words
    of a description.

    Args:
        labels: Matched header labels, left to right.
        samples: Data rows of the document.

    Returns:
        The placed boundaries. Extents are the header labels' own, since
        no per-column extent was measured.
    """
    midpoints = sorted(
        (char["x0"] + char["x1"]) / 2.0 for row in samples for char in row.chars if not char["text"].isspace()
    )
    boundaries: list[float] = []
    widths: list[float] = []
    for left, right in zip(labels, labels[1:], strict=False):
        low, high = left.x1, right.x0
        if high <= low:
            boundaries.append((low + high) / 2.0)
            widths.append(0.0)
            continue
        edges = [low, *[m for m in midpoints if low < m < high], high]
        best_gap, best_at = 0.0, (low + high) / 2.0
        for a, b in zip(edges, edges[1:], strict=False):
            if b - a > best_gap:
                best_gap, best_at = b - a, (a + b) / 2.0
        boundaries.append(best_at)
        widths.append(best_gap)
    return Corridors(
        boundaries=boundaries,
        widths=widths,
        extents=[(label.x0, label.x1) for label in labels],
        rows_used=0,
        outside_header_band=[],
    )


def build_grid(
    schema_name: str,
    labels: list[Label],
    rows: list[TextRow],
    header_tops: set[float],
    header_page: int,
) -> tuple[ColumnGrid | None, str | None]:
    """Derive one document's column grid from its header and its rows.

    Args:
        schema_name: Name of the matched listing format.
        labels: Matched header labels, left to right.
        rows: Every row of the document.
        header_tops: Tops of rows that belong to a header.
        header_page: Page the header was read from.

    Returns:
        ``(grid, note)``. ``grid`` is None when no boundaries could be
        placed at all, and ``note`` then says why.
    """
    cut = trailing_cut(labels)
    read_columns = len(labels) - 1 if cut != float("inf") else len(labels)
    method = "runs"
    margin = left_margin(rows, header_tops, read_columns, cut)
    if margin is None:
        method = "header_band"
        margin = modal_margin(rows, header_tops)
    if margin is None:
        return None, "the document prints no rows below its header"
    samples = [row for row in rows if row.top not in header_tops and is_data_row(row, margin)]

    measured = build_boundaries(labels, samples) if method == "runs" else None
    if measured is None or any(width <= 0 for width in measured.widths):
        method = "header_band"
        measured = band_boundaries(labels, samples)
    if any(width < 0 for width in measured.widths):
        return None, "two columns overlap in x; no corridor separates them"
    return (
        ColumnGrid(
            schema=schema_name,
            labels=tuple(labels),
            boundaries=tuple(measured.boundaries),
            corridors=tuple(measured.widths),
            header_page=header_page,
            left_margin=margin,
            extents=tuple(measured.extents),
            rows_used=measured.rows_used,
            outside_header_band=measured.outside_header_band,
            method=method,
        ),
        None,
    )


def is_data_row(row: TextRow, margin: float) -> bool:
    """Whether a row starts at the listing's data margin.

    Args:
        row: The row.
        margin: The document's modal left edge.

    Returns:
        True when the row begins at the data margin.
    """
    return bool(row.chars) and abs(row.x0 - margin) <= MARGIN_TOLERANCE


@dataclass
class Corridors:
    """The measured column corridors of one document.

    Attributes:
        boundaries: One x per adjacent column pair.
        widths: Width of the empty corridor each boundary sits in.
        extents: Measured ``(x0, x1)`` of each column's printed runs.
        rows_used: How many well-formed rows the measurement came from.
        outside_header_band: Boundaries that fell outside the band between
            the two header labels they separate. Recorded, not corrected.
    """

    boundaries: list[float]
    widths: list[float]
    extents: list[tuple[float, float]]
    rows_used: int
    outside_header_band: list[str]


def trailing_cut(labels: list[Label]) -> float:
    """Return the x that separates read columns from trailing ones.

    Used only to decide which runs of a row belong to the columns this
    layer reads while the corridors are being measured. The boundary that
    is finally used is measured like every other one.

    Args:
        labels: Matched header labels, left to right.

    Returns:
        The cut, or infinity when the format prints no trailing columns.
    """
    if not labels or labels[-1].key != TRAILING_KEY or len(labels) < 2:
        return float("inf")
    return (labels[-2].x1 + labels[-1].x0) / 2.0


def build_boundaries(labels: list[Label], samples: list[TextRow]) -> Corridors | None:
    """Measure the empty corridor between each adjacent pair of columns.

    A well-formed row prints exactly one run per column. Those rows -- the
    overwhelming majority -- give each column's true extent across the
    whole document, and the corridor between two neighbouring extents is
    where the boundary goes. Rows whose columns have run together, which
    are the rows this whole module exists for, are excluded from the
    measurement and then assigned by the boundaries it produces.

    Where inside the corridor the boundary sits depends on which of the two
    columns can grow toward the other, and that is measured too: a column
    whose runs all start at the same x is anchored on the left and can only
    grow right, and one whose runs all end at the same x is anchored on the
    right and can only grow left. The boundary is kept clear of whichever
    side is anchored, so a vendor name longer than any seen here still
    lands in the vendor column rather than in the check date. When both
    sides can grow the corridor is split down the middle, which is the only
    honest answer available and leaves any row that exceeds the measured
    extents to fail its type check rather than be guessed at.

    Args:
        labels: Matched header labels, left to right.
        samples: Data rows of the document.

    Returns:
        The measured corridors, or None when too few rows are well formed
        to measure from.
    """
    expected = len(labels)
    cut = trailing_cut(labels)
    read_columns = expected - 1 if cut != float("inf") else expected
    lows = [float("inf")] * expected
    highs = [float("-inf")] * expected
    starts: list[list[float]] = [[] for _ in range(expected)]
    ends: list[list[float]] = [[] for _ in range(expected)]
    used = 0

    for row in samples:
        runs = row_runs(row)
        in_scope = [run for run in runs if run.x0 < cut]
        out_of_scope = [run for run in runs if run.x0 >= cut]
        if len(in_scope) != read_columns:
            continue
        used += 1
        for index, run in enumerate(in_scope):
            lows[index] = min(lows[index], run.x0)
            highs[index] = max(highs[index], run.x1)
            starts[index].append(run.x0)
            ends[index].append(run.x1)
        if out_of_scope:
            lows[-1] = min(lows[-1], min(run.x0 for run in out_of_scope))
            highs[-1] = max(highs[-1], max(run.x1 for run in out_of_scope))

    if used < MIN_SAMPLE_ROWS or any(low == float("inf") for low in lows[:read_columns]):
        return None
    if lows[-1] == float("inf"):
        # A trailing column the sampled rows never printed cannot bound the
        # column before it; fall back to its header label.
        lows[-1], highs[-1] = labels[-1].x0, labels[-1].x1

    boundaries: list[float] = []
    widths: list[float] = []
    outside: list[str] = []
    for index in range(expected - 1):
        left_edge, right_edge = highs[index], lows[index + 1]
        corridor = right_edge - left_edge
        grows_right = _spread(ends[index]) > ALIGNMENT_SPREAD or not ends[index]
        grows_left = _spread(starts[index + 1]) > ALIGNMENT_SPREAD or not starts[index + 1]
        boundaries.append(_place(left_edge, right_edge, grows_right, grows_left))
        widths.append(corridor)
        band_low, band_high = labels[index].x1, labels[index + 1].x0
        if not band_low <= boundaries[-1] <= band_high:
            outside.append(
                f"{labels[index].key}|{labels[index + 1].key}: corridor at {boundaries[-1]:.2f} "
                f"is outside the header band [{band_low:.2f}, {band_high:.2f}]"
            )
    return Corridors(
        boundaries=boundaries,
        widths=widths,
        extents=list(zip(lows, highs, strict=True)),
        rows_used=used,
        outside_header_band=outside,
    )


def _spread(values: list[float]) -> float:
    """Return how far apart a column's run edges sit.

    Args:
        values: One edge per sampled run.

    Returns:
        Range of the values, or 0.0 when there are none.
    """
    return max(values) - min(values) if values else 0.0


def _place(left_edge: float, right_edge: float, grows_right: bool, grows_left: bool) -> float:
    """Choose a point inside a corridor.

    Args:
        left_edge: Rightmost x the left column was measured at.
        right_edge: Leftmost x the right column was measured at.
        grows_right: Whether the left column can extend further right.
        grows_left: Whether the right column can extend further left.

    Returns:
        The boundary x.
    """
    corridor = right_edge - left_edge
    if corridor <= 0:
        return (left_edge + right_edge) / 2.0
    margin = min(HUG_MARGIN, corridor / 2.0)
    if grows_right and not grows_left:
        return right_edge - margin
    if grows_left and not grows_right:
        return left_edge + margin
    return (left_edge + right_edge) / 2.0


@dataclass(frozen=True)
class AssignedRow:
    """One row's characters sorted into the grid's columns.

    Attributes:
        cells: Column key to the text assigned to it.
        ambiguous: Characters whose midpoint sat on a boundary.
    """

    cells: dict[str, str]
    ambiguous: tuple[str, ...]

    def get(self, key: str) -> str:
        """Return a column's text, or the empty string.

        Args:
            key: Canonical column name.

        Returns:
            The column's text with outer whitespace stripped.
        """
        return self.cells.get(key, "").strip()


def assign_row(row: TextRow, grid: ColumnGrid) -> AssignedRow:
    """Sort a row's characters into columns by x midpoint.

    Args:
        row: The row to assign.
        grid: The document's column grid.

    Returns:
        The row's text per column, and any characters that sat on a
        boundary.
    """
    buckets: dict[int, list[dict]] = {}
    ambiguous: list[str] = []
    for char in row.chars:
        midpoint = (char["x0"] + char["x1"]) / 2.0
        index = grid.column_of(midpoint)
        if not char["text"].isspace():
            for boundary in grid.boundaries:
                if abs(midpoint - boundary) < MIDPOINT_CLEARANCE:
                    ambiguous.append(f"{char['text']!r} at x={midpoint:.2f} sits on the boundary at {boundary:.2f}")
                    break
        buckets.setdefault(index, []).append(char)

    cells: dict[str, str] = {}
    for index, chars in buckets.items():
        if index >= len(grid.labels):
            continue
        ordered = sorted(chars, key=lambda c: c["x0"])
        pieces: list[str] = []
        reach = float("-inf")
        for char in ordered:
            # A gap with no space character in it is still a separation:
            # 2026-07-22 prints its columns with no spaces at all.
            if reach > float("-inf") and char["x0"] - reach > RUN_GAP:
                pieces.append(" ")
            pieces.append(char["text"])
            reach = max(reach, char["x1"])
        cells[grid.labels[index].key] = re.sub(r"\s+", " ", "".join(pieces)).strip()
    return AssignedRow(cells=cells, ambiguous=tuple(ambiguous))
