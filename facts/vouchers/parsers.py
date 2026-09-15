"""Row and total parsers for the four voucher listing eras.

Phase 0 established that the district has changed its voucher report format
four times since 2005, and that the printed column header is **not** a
reliable era key -- it is two physical lines that wrap differently depending
on page width, which produced 55 distinct "signatures" for four actual
layouts. The reliable key is the column **order** plus whether amounts carry
a ``$``.

    Era A  2005-2008   Voucher Number | Vendor Name | Amount | Description
                       No check date, no check number, one amount.
    Era B  2017-2019   CHECK NO. | VENDOR | DATE | DESCRIPTION | INVOICE | INV. TOTAL
                       Check number FIRST. Amounts $-prefixed.
    Era C  2020-2022   Vendor | Check date | Check # | Check Amt | Invoice Amt | Work performed
                       Vendor first. Amounts $-prefixed. Listings are CUMULATIVE.
    Era D  2023-2026   Vendor | Check Date | Check Number | Check Amount | Invoice Amount | Description
                       First era to print a whole-line TOTAL. $ optional.

Nothing here uses an LLM. Every value is produced by a regular expression
and exact ``Decimal`` arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

# --------------------------------------------------------------- amounts --

# pdfplumber's layout mode can split a number with a space ("$ 3 ,609,064.78",
# "$ 8 4,009.42"). Spaces are a rendering artifact, not content, so they are
# admitted in the pattern and stripped after capture.
#
# The decimal part is OPTIONAL, and that is load-bearing. This corpus prints
# invoice amounts with one decimal place and with none at all -- the
# 2025-03-26 ASB listing has "123.4", "293.1" and a bare "132" alongside
# check amounts that do carry cents. Requiring two decimals was tried and
# reverted: it dropped 2,108 real rows corpus-wide and took nine sets from
# reconciled to out-of-balance. The change had been measured on twelve
# well-behaved 2023 and 2026 files, where it genuinely changed nothing --
# too narrow a sample to justify the rule.
#
# The safeguard against a capture running into the next column is not this
# pattern: it is the comma-grouping check in `money`, plus the whitespace
# boundary required of the register's own amount capture.
AMOUNT_BODY = r"-?\$?\s?-?[\d, ]*\.?\d+-?"


# After the layout spaces are removed, a real amount has either no commas
# at all or thousands groups of exactly three digits. This one check is what
# separates a genuine pdfplumber split ("8 4,009.42" -> "84,009.42") from a
# capture that has swallowed the column to its left ("2/12/2026 384,438.45"
# -> "2026384,438.45"). Without it the register cross-check silently reads a
# warrant issue date as part of the money.
GROUPING_RX = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")


def money(raw: str | None) -> Decimal | None:
    """Parse a printed amount into an exact Decimal.

    Handles the three ways this corpus writes a negative number -- a leading
    minus, a minus after the dollar sign, and a trailing minus -- and the
    pdfplumber layout artifact that splits a number with a space.

    A value whose thousands separators do not group into threes is rejected
    rather than returned, because in this corpus that always means the
    capture reached into the neighbouring column.

    Args:
        raw: Amount as printed, possibly with ``$``, commas or stray spaces.

    Returns:
        Exact decimal value, or None when the text is not a well-formed
        amount.
    """
    if raw is None:
        return None
    cleaned = raw.replace(" ", "").replace("$", "")
    negative = cleaned.startswith("-") or cleaned.endswith("-")
    cleaned = cleaned.strip("-")
    if not cleaned or cleaned == "." or not GROUPING_RX.match(cleaned):
        return None
    try:
        value = Decimal(cleaned.replace(",", ""))
    except InvalidOperation:
        return None
    return -value if negative else value


def money_tail(raw: str | None) -> Decimal | None:
    """Parse the amount at the end of a captured span.

    Tries the whole span first, so a pdfplumber-split number stays intact,
    then drops leading whitespace-separated chunks until what remains is a
    well-formed amount. That ordering matters: preferring the shortest
    suffix would read ``$ 8 4,009.42`` as ``4,009.42``.

    Args:
        raw: Captured text ending in an amount.

    Returns:
        Exact decimal value, or None.
    """
    if raw is None:
        return None
    parts = raw.split()
    for start in range(len(parts)):
        value = money("".join(parts[start:]))
        if value is not None:
            return value
    return None


def parse_date(raw: str | None) -> date | None:
    """Parse a printed check date.

    Args:
        raw: Date as printed, possibly carrying a pdfplumber-injected space
            ("03 /12/2026") and either a two- or four-digit year.

    Returns:
        A date, or None when the text is not a date this corpus uses.
    """
    if not raw:
        return None
    cleaned = raw.replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", cleaned)
    if not m:
        return None
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        # The corpus spans 2005-2026; a two-digit year is always this century.
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------- totals --

# A TOTAL is matched only as a whole line. "Total Technology" is a real
# vendor on the 2026-03-25 General Fund listing, page 16; matching it as a
# total would silently truncate the set 19,000 characters early.
TOTAL_LINE_RX = re.compile(
    r"^[ \t]*(?P<label>(?:GRAND[ \t]+)?TOTALS?(?:[ \t]+[A-Z&' ]+?)?)"
    r"[ \t]*\$?[ \t]*(?P<amount>-?[\d, ]+\.\d{2})[ \t]*$",
    re.M,
)

# The register prints a per-fund total with the fund named in the label.
REGISTER_TOTAL_RX = re.compile(
    r"^[ \t]*TOTAL[ \t]+(?P<fund>[A-Z][A-Z&' ]*?)[ \t]*\$?[ \t]*" r"(?P<amount>-?[\d, ]+\.\d{2})[ \t]*$",
    re.M,
)

REGISTER_FUND_NAMES = {
    "GENERAL FUND": "GF",
    "CAPITAL PROJECTS FUND": "Capital",
    "ASB FUND": "ASB",
    "CUSTODIAL FUND": "Custodial",
    "FIDUCIARY FUND": "Trust",
    "TRANSPORTATION VEHICLE FUND": "Transportation",
    "PERMANENT FUND": "Permanent",
    "TRUST FUND": "Trust",
}

# ------------------------------------------------------------ era D rows --

# The build brief's starting-point regex, verbatim. Retained so tests can
# show exactly what it does and does not catch. Not used for parsing.
ROW_RX_BRIEF = re.compile(
    r"^\s*(?P<vendor>.+?)\s{1,}"
    r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<chk>\d{5,11})\s+"
    r"(?P<chkamt>-?[\d,]*\.?\d+)\s+"
    r"(?P<invamt>-?[\d,]*\.?\d+)\s*"
    r"(?P<desc>.*)$"
)

# Eras C and D: vendor, then date, then check number, then two amounts.
#
# Three differences from the brief's pattern, each forced by a measured
# failure on a real document rather than by anticipation:
#
#  1. ``\s*`` rather than ``\s{1,}`` between vendor and date. On 2026-03-25
#     ASB the vendor "THE HEATHMAN LODGE AND HUDSONS BAR AN" runs straight
#     into its date with no gap: "...BAR AN03 /12/2026". That one row is the
#     whole of the $1,322.35 shortfall recorded against that set.
#  2. The date may carry an injected space, and may use a two-digit year
#     (Era C writes "9/24/2020" but also "12/03/2020" and "9/30/21").
#  3. Amounts may be $-prefixed. This single omission is why the brief's
#     pattern parses zero rows from 268 of the 446 detail listings.
ROW_RX_VENDOR_FIRST = re.compile(
    r"^\s*(?P<vendor>.+?)\s*"
    r"(?P<date>\d{1,2}\s?/\s?\d{1,2}\s?/\s?(?:\d{4}|\d{2}))\s+"
    r"(?P<chk>\d{5,11})\s+"
    rf"(?P<chkamt>{AMOUNT_BODY})\s+"
    rf"(?P<invamt>{AMOUNT_BODY})\s*"
    r"(?P<desc>.*)$"
)

# Era B: check number first, then vendor, then date, then description, then
# two $-prefixed amounts at the right margin.
ROW_RX_CHECK_FIRST = re.compile(
    r"^\s*(?P<chk>\d{5,11})\s+"
    r"(?P<vendor>.+?)\s+"
    r"(?P<date>\d{1,2}\s?/\s?\d{1,2}\s?/\s?(?:\d{4}|\d{2}))\s+"
    r"(?P<desc>.*?)\s*"
    rf"\$\s*(?P<chkamt>{AMOUNT_BODY})\s+"
    rf"\$\s*(?P<invamt>{AMOUNT_BODY})\s*$"
)

# Era A: voucher number, vendor, one amount, description. There is no check
# date and no separate invoice amount anywhere in this era's report.
ROW_RX_ERA_A = re.compile(
    r"^\s*(?P<chk>\d{5,9})\s+" r"(?P<vendor>.+?)\s{2,}" rf"(?P<invamt>{AMOUNT_BODY})\s{{2,}}" r"(?P<desc>.*)$"
)

# Lines that look like data but are furniture.
NOISE_RX = re.compile(
    r"^\s*(?:Kent School District|Voucher Register|Board Agenda Vouchers"
    r"|Page\s+\d|PAGE\s+\d|Date:|DATE:|-{5,}|={5,})",
    re.I,
)

# Era detection signals, applied to the first page only.
ERA_A_RX = re.compile(r"KSD\s+VOUCHER\s+REGISTER", re.I)
ERA_B_RX = re.compile(r"CHECK\s+NO\.", re.I)
ERA_C_RX = re.compile(r"Check\s*date\s+Check\s*#|Work\s+performed", re.I)
ERA_D_RX = re.compile(r"Check\s+Date\s+(?:Check\s+)?Number|Check\s+Number", re.I)


# The one reason code added for column assignment. Every way a row can
# fail to be read by geometry -- a character sitting on a boundary, a
# column whose content fails its own type check, a required column with
# nothing in it, or a page whose grid could not be derived -- is this code,
# with the specific failure in ``reason_detail``. One code, because a row
# that cannot be read is one kind of fact regardless of which column broke.
COLUMN_AMBIGUOUS = "COLUMN_AMBIGUOUS"

# A check number as this corpus prints it. Matches the bound the regex row
# patterns have always used, so the geometry path accepts and rejects the
# same numbers the regex path did.
CHECK_NUMBER_RX = re.compile(r"^\d{5,11}$")


@dataclass
class Row:
    """One parsed voucher line with its locator offsets.

    Attributes:
        vendor_raw: Vendor exactly as printed, whitespace-collapsed.
        check_date: Parsed check date, when the era prints one.
        check_number: Check or pseudo-check number as printed.
        check_amount: Total amount of the check.
        invoice_amount: Amount of this invoice against that check.
        description: Description, including any continuation lines.
        char_offset: Offset of the row's first character in the PDF text.
        char_end: Offset just past the row's last character.
        line_text: The raw line, for the locator quote.
        reason_code: ``COLUMN_AMBIGUOUS`` when the row could not be read,
            otherwise None. A row with a reason code is still recorded:
            a row that is on the page and not in the table is a silent
            drop, which is the failure mode this layer exists to prevent.
        reason_detail: What specifically could not be read.
        regex_verdict: ``agree``, ``disagree`` or ``regex_miss`` from the
            retired regex path, kept as a per-line cross-check only.
    """

    vendor_raw: str
    check_date: date | None
    check_number: str | None
    check_amount: Decimal | None
    invoice_amount: Decimal | None
    description: str
    char_offset: int
    char_end: int
    line_text: str
    reason_code: str | None = None
    reason_detail: str | None = None
    regex_verdict: str | None = None


@dataclass
class ParsedListing:
    """Everything a parser extracts from one voucher listing.

    Attributes:
        era: Detected format era.
        rows: Parsed rows in document order.
        totals: Every whole-line TOTAL found, as (label, amount, offset).
        stated_total: The total chosen by the first-after-last-row rule.
        stated_total_offset: Offset of the chosen total.
        stated_total_rule: Which rule chose it.
        extra_totals: Totals found after the chosen one.
        period: (start, end) of the warrant period as printed.
        pcard_period: (start, end) of the P-card period as printed.
        fund_from_text: Fund named by the printed header, when present.
        notes: Observations worth carrying onto the set row.
        grid_schema: Listing format the column grid matched, when one did.
        grid_method: How the grid's corridors were measured.
        grid_note: Why no grid could be built, when that is the case.
        grid_columns: Canonical column names the grid carries.
        narrowest_corridor: Tightest boundary corridor on the grid.
        regex_agree: Rows where the retired regex agreed with geometry.
        regex_disagree: Rows where it produced something different.
        regex_miss: Rows it could not read at all.
    """

    era: str | None
    rows: list[Row] = field(default_factory=list)
    totals: list[tuple[str, Decimal, int]] = field(default_factory=list)
    stated_total: Decimal | None = None
    stated_total_offset: int | None = None
    stated_total_rule: str | None = None
    extra_totals: list[tuple[str, Decimal, int]] = field(default_factory=list)
    period: tuple[date | None, date | None] = (None, None)
    pcard_period: tuple[date | None, date | None] = (None, None)
    fund_from_text: str | None = None
    notes: list[str] = field(default_factory=list)
    grid_schema: str | None = None
    grid_method: str | None = None
    grid_note: str | None = None
    grid_columns: tuple[str, ...] = ()
    narrowest_corridor: float | None = None
    regex_agree: int = 0
    regex_disagree: int = 0
    regex_miss: int = 0


PERIOD_RX = re.compile(
    r"(?P<d1>\d{1,2}/\d{1,2}/\d{2,4})\s*(?:through|to|-)\s*(?P<d2>\d{1,2}/\d{1,2}/\d{2,4})",
    re.I,
)
PCARD_RX = re.compile(
    r"P-?Cards?\s+(?P<d1>\d{1,2}/\d{1,2}/\d{2,4})\s*(?:through|to|-)\s*" r"(?P<d2>\d{1,2}/\d{1,2}/\d{2,4})",
    re.I,
)


def detect_era(first_page: str, meeting_date: str | None = None) -> str | None:
    """Identify the format era of a listing from its first page.

    The printed column header wraps differently at different page widths, so
    era detection keys on the distinctive tokens of each generation's report
    rather than on the header line as a whole.

    Args:
        first_page: Text of the listing's first page.
        meeting_date: ISO meeting date, used only to break a tie.

    Returns:
        ``A``, ``B``, ``C``, ``D``, or None when nothing matches.
    """
    head = first_page[:4000]
    if ERA_A_RX.search(head):
        return "A"
    if ERA_B_RX.search(head):
        return "B"
    if ERA_C_RX.search(head):
        return "C"
    if ERA_D_RX.search(head):
        return "D"
    # No header survived extraction. Fall back to the date, which is a
    # weaker signal and is recorded as such by the caller.
    if meeting_date:
        year = int(meeting_date[:4])
        if year <= 2009:
            return "A"
        if year <= 2019:
            return "B"
        if year <= 2022:
            return "C"
        return "D"
    return None


def _clean(text: str) -> str:
    """Collapse runs of whitespace in a captured field.

    Args:
        text: Raw captured text.

    Returns:
        Whitespace-normalized text.
    """
    return re.sub(r"\s+", " ", text).strip()


def _row_from_match(match: re.Match, offset: int, line: str, era: str) -> Row | None:
    """Build a Row from a regex match, or None when it is not a data row.

    Args:
        match: A row-regex match.
        offset: Character offset of the line within the PDF text.
        line: The raw line.
        era: Format era, which decides how the columns are read.

    Returns:
        A Row, or None when the captured values are not usable.
    """
    groups = match.groupdict()
    vendor = _clean(groups.get("vendor") or "")
    if not vendor:
        return None
    invoice = money(groups.get("invamt"))
    if invoice is None:
        return None
    # Era A prints one amount; the check amount and invoice amount are the
    # same figure and saying otherwise would invent a distinction.
    check_amount = money(groups.get("chkamt")) if era != "A" else invoice
    return Row(
        vendor_raw=vendor,
        check_date=parse_date(groups.get("date")),
        check_number=(groups.get("chk") or "").strip() or None,
        check_amount=check_amount,
        invoice_amount=invoice,
        description=_clean(groups.get("desc") or ""),
        char_offset=offset,
        char_end=offset + len(line),
        line_text=line,
    )


def _row_regex_for(era: str) -> re.Pattern:
    """Return the row pattern for an era.

    Args:
        era: Format era.

    Returns:
        The compiled row pattern.
    """
    if era == "A":
        return ROW_RX_ERA_A
    if era == "B":
        return ROW_RX_CHECK_FIRST
    return ROW_RX_VENDOR_FIRST


# A continuation line carries the tail of a wrapped vendor name or
# description. It has no date, no check number and no amount -- a line with
# an amount is always a data row, and treating it as continuation would lose
# money.
CONTINUATION_RX = re.compile(r"^\s{2,}\S")
HAS_AMOUNT_RX = re.compile(r"\d[\d,]*\.\d{2}")


def parse_listing(text: str, first_page: str, meeting_date: str | None = None) -> ParsedListing:
    """Parse a voucher detail listing into rows and totals.

    Args:
        text: Full layout-preserved text of the PDF.
        first_page: Text of page 1, used for era and header detection.
        meeting_date: ISO meeting date, used only as an era tie-break.

    Returns:
        A ParsedListing. ``era`` is None when no era could be identified.
    """
    era = detect_era(first_page, meeting_date)
    result = ParsedListing(era=era)
    if era is None:
        return result

    row_rx = _row_regex_for(era)

    offset = 0
    last_row_end = 0
    for line in text.split("\n"):
        line_start = offset
        offset += len(line) + 1
        if not line.strip() or NOISE_RX.match(line):
            continue
        match = row_rx.match(line)
        if match:
            row = _row_from_match(match, line_start, line, era)
            if row is not None:
                result.rows.append(row)
                last_row_end = row.char_end
                continue
        # Not a data row. If it carries no amount and we already have a row,
        # it is the wrapped tail of the previous one.
        if result.rows and not HAS_AMOUNT_RX.search(line) and CONTINUATION_RX.match(line):
            tail = _clean(line)
            if tail and not TOTAL_LINE_RX.match(line):
                previous = result.rows[-1]
                previous.description = f"{previous.description} {tail}".strip()
                previous.char_end = line_start + len(line)

    _read_totals(result, text, first_page, last_row_end)
    return result


def _read_totals(result: ParsedListing, text: str, first_page: str, last_row_end: int) -> None:
    """Read the printed TOTAL lines and the header's periods.

    Unchanged by the geometry work and shared by both parse paths: a TOTAL
    is a whole line of text, not a column, and reading it from coordinates
    would be a change with nothing to gain.

    Args:
        result: The listing being built, mutated in place.
        text: Full layout-preserved text of the PDF.
        first_page: Text of page 1.
        last_row_end: Offset just past the last data row found.
    """
    for match in TOTAL_LINE_RX.finditer(text):
        amount = money(match.group("amount"))
        if amount is not None:
            result.totals.append((_clean(match.group("label")), amount, match.start()))

    # The stated total is the first whole-line TOTAL at or after the last
    # data row. A trailing TOTAL-only page -- the 2026-05-27 Transportation
    # listing prints a second TOTAL on page 5 at exactly twice the real
    # figure -- is recorded but does not win.
    for label, amount, start in result.totals:
        if start >= last_row_end:
            result.stated_total = amount
            result.stated_total_offset = start
            result.stated_total_rule = "first_total_after_last_row"
            break
    if result.stated_total is None and result.totals:
        label, amount, start = result.totals[-1]
        result.stated_total = amount
        result.stated_total_offset = start
        result.stated_total_rule = "last_total_no_row_anchor"
    if result.stated_total_offset is not None:
        result.extra_totals = [t for t in result.totals if t[2] > result.stated_total_offset]

    period = PERIOD_RX.search(first_page)
    pcard = PCARD_RX.search(first_page)
    if period and pcard and period.start() >= pcard.start():
        period = None
    if period:
        result.period = (parse_date(period.group("d1")), parse_date(period.group("d2")))
    if pcard:
        result.pcard_period = (parse_date(pcard.group("d1")), parse_date(pcard.group("d2")))


def read_cells(cells, keys: tuple[str, ...]) -> tuple[Row | None, list[str]]:
    """Turn one row's assigned columns into a Row, or say why it cannot be.

    Every column is parsed independently and against its own type. A
    boundary is never moved to make a column parse, and a column that fails
    is never filled in from a neighbour.

    Args:
        cells: The row's assigned columns.
        keys: Canonical column names the grid carries, left to right.

    Returns:
        ``(row, problems)``. ``row`` carries the values that did parse even
        when ``problems`` is non-empty, so a failed row still shows the
        operator what was on the page.
    """
    problems: list[str] = []
    if cells.ambiguous:
        problems.append(f"character on a column boundary: {cells.ambiguous[0]}")

    vendor = cells.get("vendor")
    if not vendor:
        problems.append("the vendor column is empty")

    check_date = None
    if "check_date" in keys:
        raw = cells.get("check_date")
        check_date = parse_date(raw)
        if check_date is None:
            problems.append(f"the check date column holds {raw!r}, which is not a date")

    check_number = None
    if "check_number" in keys:
        raw = cells.get("check_number").replace(" ", "")
        if CHECK_NUMBER_RX.match(raw):
            check_number = raw
        else:
            problems.append(f"the check number column holds {raw!r}, which is not a check number")

    invoice = money(cells.get("invoice_amount"))
    if invoice is None:
        problems.append(f"the invoice amount column holds {cells.get('invoice_amount')!r}, which is not an amount")

    if "check_amount" in keys:
        check_amount = money(cells.get("check_amount"))
        if check_amount is None:
            problems.append(f"the check amount column holds {cells.get('check_amount')!r}, which is not an amount")
    else:
        # Era A prints one amount. Saying the check total differs from the
        # invoice total would invent a distinction the document does not
        # make.
        check_amount = invoice

    row = Row(
        vendor_raw=_clean(vendor),
        check_date=check_date,
        check_number=check_number,
        check_amount=check_amount,
        invoice_amount=invoice,
        description=_clean(cells.get("description")),
        char_offset=0,
        char_end=0,
        line_text="",
    )
    return row, problems


def _regex_verdict(line: str, era: str | None, row: Row) -> str:
    """Compare the retired regex path against what geometry read.

    Geometry is authoritative. This runs only so the parse log can say how
    often the two agree, and on which sets they do not.

    Args:
        line: The row's layout text line.
        era: Detected format era.
        row: The row geometry produced.

    Returns:
        ``agree``, ``disagree`` or ``regex_miss``.
    """
    match = _row_regex_for(era or "D").match(line)
    if not match:
        return "regex_miss"
    other = _row_from_match(match, 0, line, era or "D")
    if other is None:
        return "regex_miss"
    same = other.invoice_amount == row.invoice_amount and other.check_number == row.check_number
    return "agree" if same else "disagree"


def parse_listing_geometric(pdf, meeting_date: str | None = None) -> ParsedListing:
    """Parse a voucher detail listing by column geometry.

    Args:
        pdf: A ``PdfText`` carrying both the layout text and the page
            geometry.
        meeting_date: ISO meeting date, used only as an era tie-break.

    Returns:
        A ParsedListing. Rows that could not be read carry a reason code
        and are still present.
    """
    import geometry as geo

    first_page = pdf.pages[0].text if pdf.pages else ""
    era = detect_era(first_page, meeting_date)
    result = ParsedListing(era=era)
    grid = pdf.grid
    result.grid_note = pdf.grid_note
    if grid is None:
        _read_totals(result, pdf.text, first_page, 0)
        return result

    result.grid_schema = grid.schema
    result.grid_method = grid.method
    result.grid_columns = grid.keys
    result.narrowest_corridor = grid.narrowest_corridor
    if grid.outside_header_band:
        result.notes.append(
            f"{len(grid.outside_header_band)} column boundary(ies) sit outside the band between the "
            f"header labels they separate: {'; '.join(grid.outside_header_band)}"
        )
    if pdf.grid_drift:
        result.notes.append(f"header drift between pages: {'; '.join(pdf.grid_drift)}")
    if pdf.misaligned_pages:
        result.notes.append(
            f"page(s) {pdf.misaligned_pages} render a different number of text lines than printed rows; "
            f"their locator offsets are page-level rather than line-level"
        )

    keys = grid.keys
    last_row_end = 0
    for page in pdf.pages:
        for placed in page.rows:
            row_geo, line, offset = placed.row, placed.line, placed.char_offset
            if row_geo.top in page.header_tops:
                continue
            if not line.strip() or NOISE_RX.match(line) or TOTAL_LINE_RX.match(line):
                continue
            if geo.is_data_row(row_geo, grid.left_margin):
                parsed, problems = read_cells(geo.assign_row(row_geo, grid), keys)
                parsed.char_offset = offset
                parsed.char_end = offset + len(line)
                parsed.line_text = line
                if problems:
                    parsed.reason_code = COLUMN_AMBIGUOUS
                    parsed.reason_detail = "; ".join(problems)
                    if not parsed.vendor_raw:
                        parsed.vendor_raw = _clean(line)[:300] or "(blank row)"
                parsed.regex_verdict = _regex_verdict(line, era, parsed)
                if parsed.regex_verdict == "agree":
                    result.regex_agree += 1
                elif parsed.regex_verdict == "disagree":
                    result.regex_disagree += 1
                else:
                    result.regex_miss += 1
                result.rows.append(parsed)
                last_row_end = parsed.char_end
                continue
            # Not a data row. If it carries no amount and we already have a
            # row, it is the wrapped tail of the previous one.
            if result.rows and not HAS_AMOUNT_RX.search(line) and CONTINUATION_RX.match(line):
                tail = _clean(line)
                if tail:
                    previous = result.rows[-1]
                    previous.description = f"{previous.description} {tail}".strip()
                    previous.char_end = offset + len(line)
                    # last_row_end deliberately does NOT move here. It
                    # anchors which printed TOTAL is the set's own, and a
                    # wrapped description is not a new data row. The
                    # 2026-05-27 Transportation listing has one data row on
                    # page 1 followed by four title-only pages and a second
                    # TOTAL at exactly twice the real figure; advancing the
                    # anchor over those titles hands the set the wrong
                    # total.

    _read_totals(result, pdf.text, first_page, last_row_end)
    return result


def parse_register(text: str) -> dict[str, list[dict]]:
    """Parse a signed warrant register into its per-line detail.

    The register is organised by fund and payment type, listing warrant
    number ranges and issue dates rather than vendors. It is the board's own
    approval record, certified under penalty of perjury, and it is produced
    from a different query than the detail listings -- which is what makes it
    an independent cross-check rather than a restatement.

    Args:
        text: Full layout-preserved text of the register PDF.

    Returns:
        A dict with ``lines`` (one per register row) and ``fund_totals``
        (one per printed ``TOTAL <FUND>`` line).
    """
    lines: list[dict] = []
    # The amount must begin at a whitespace boundary. Without that guard the
    # capture starts mid-token after a date separator and reads the year as
    # part of the money: "530157-530158 3/5/2026 355.27" becomes 2,026,355.27
    # and "PURCHASING CARD 2/11/26-3/12/26 403.45" becomes 26,403.45. Both
    # were live in this parser until a unit test on a four-line synthetic
    # register caught them. A decimal point is required for the same reason.
    amount_rx = re.compile(r"(?:^|(?<=\s))(-?\$?\s?-?\d[\d, ]*\.\d{2}-?)\s*$")
    range_rx = re.compile(r"\b(\d{5,10})\s*-\s*(\d{5,10})\b")
    offset = 0
    for raw in text.split("\n"):
        start = offset
        offset += len(raw) + 1
        line = _clean(raw)
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("TOTAL", "GRAND", "FUND ", "I, THE", "SERVICES", "OBLIGATIONS", "SIGNATURE")):
            continue
        match = amount_rx.search(line)
        if not match:
            continue
        amount = money_tail(match.group(1))
        if amount is None:
            continue
        rng = range_rx.search(line)
        lines.append(
            {
                "text": line,
                "amount": amount,
                "offset": start,
                "warrant_range": f"{rng.group(1)}-{rng.group(2)}" if rng else None,
                "is_direct_deposit": "DIRECT DEPOSIT" in upper,
                "is_pcard": "PURCHASING CARD" in upper,
                "is_payroll": " PAYROLL " in f" {upper} ",
                "is_accounts_payable": "ACCOUNTS PAYABLE" in upper,
                "fund": _register_fund(upper),
            }
        )

    fund_totals: list[dict] = []
    for match in REGISTER_TOTAL_RX.finditer(text):
        amount = money(match.group("amount"))
        label = _clean(match.group("fund"))
        if amount is None or label.upper() == "":
            continue
        fund_totals.append(
            {
                "label": label,
                "fund": REGISTER_FUND_NAMES.get(label.upper()),
                "amount": amount,
                "offset": match.start(),
                "quote": _clean(match.group(0)),
            }
        )
    return {"lines": lines, "fund_totals": fund_totals}


def _register_fund(upper_line: str) -> str | None:
    """Identify which fund a register line belongs to.

    Args:
        upper_line: The register line, upper-cased.

    Returns:
        A fund code, or None.
    """
    for name, code in REGISTER_FUND_NAMES.items():
        if upper_line.startswith(name.split()[0]):
            # "GENERAL", "CAPITAL", "ASB", "CUSTODIAL", "TRANSPORTATION"
            if name.split()[0] in {
                "GENERAL",
                "CAPITAL",
                "ASB",
                "CUSTODIAL",
                "TRANSPORTATION",
                "FIDUCIARY",
                "PERMANENT",
                "TRUST",
            }:
                return code
    return None
