"""Fixtures for the voucher fact layer.

A fixture is a number somebody stated independently of this code, checked
against what the build produced. Three outcomes, and they are not
interchangeable:

``PASS``
    The build reproduces the stated figure exactly.
``FAIL``
    The build produces something else. Never loosened to make it green.
``BLOCKED``
    The fixture could not run because its source document is not on this
    machine. **A check that never executed is not a check that succeeded**,
    so it is reported separately and never counted as a pass.

``main`` exits non-zero when any HARD fixture fails. Blocked fixtures do not
fail the run -- they are a missing input, not a defect -- but they are
printed every time so they cannot be quietly forgotten.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from decimal import Decimal

import db

# ------------------------------------------------------------------ HARD --
#
# Stated totals as given in the build brief. Every one of these is a figure
# printed on the PDF's own TOTAL line. Where Phase 0 could reach the
# document, the value below was read out of it and matched the brief.
HARD_SET_TOTALS: dict[str, dict[str, Decimal]] = {
    "2026-03-25": {
        "GF": Decimal("5609073.26"),
        "ACH": Decimal("3609064.78"),
        "Capital": Decimal("84009.42"),
        "ASB": Decimal("136796.65"),
    },
    # Added by the operator on 2026-09-14: five funds, all verified against
    # the documents in Phase 0, plus the double-TOTAL reproduction.
    "2026-05-27": {
        "GF": Decimal("3388060.86"),
        "ACH": Decimal("9394987.52"),
        "Capital": Decimal("1047509.82"),
        "ASB": Decimal("181039.41"),
        "Transportation": Decimal("173922.13"),
    },
    "2026-06-24": {
        "GF": Decimal("2026444.30"),
        "ACH": Decimal("5490076.68"),
        "Capital": Decimal("932245.73"),
        "ASB": Decimal("149650.79"),
    },
    "2026-07-22": {
        "ACH": Decimal("5585581.92"),
        "Capital": Decimal("127028.73"),
    },
    "2026-08-26": {
        "GF": Decimal("2474622.84"),
        "ACH": Decimal("15001072.89"),
        "Capital": Decimal("403772.77"),
        "ASB": Decimal("14330.26"),
    },
}

# The pre-2024 set chosen in Phase 0, with its stated total read from the
# document's own TOTAL line and recorded in the recon report before any
# parsing was done against it.
HARD_PRE_2024 = {
    "meeting_date": "2023-08-23",
    "fund": "Capital",
    "stated_total": Decimal("2120786.72"),
    "document_id": "1eb3a878-8998-4f41-a966-2b2c3b3ef38b",
    "sha256": "eb157ca7a6109cdc93c47c2294b252c580214253fd2d3db58f2261237d51e345",
    "note": "Era C header, first generation to print a TOTAL. Read from page 2.",
}

# 2026-06-24 General Fund, decomposed. The three parts sum to the whole:
#   1,800,211.05 + 224,719.75 + 1,513.50 = 2,026,444.30
HARD_GF_COMPONENTS = {
    "meeting_date": "2026-06-24",
    "fund": "GF",
    "warrant_range": (608270, 608506),
    "warrant_sum": Decimal("1800211.05"),
    "pcard_range": (9261000039, 9261000042),
    "pcard_sum": Decimal("224719.75"),
    "payroll_check": "530162",
    "payroll_amount": Decimal("1513.50"),
}

# Individual printed lines, each traced to a value printed on the staged
# PDF. These are the R1 fixtures: every one of them was read wrongly by the
# regex parser because the amount column had no right-hand edge in the
# extracted text, and every one of them is a figure a reader could check
# against the document in under a minute.
HARD_LINES: tuple[dict, ...] = (
    {
        "name": "2026-06-24 ASB 418449 Head Quarters Corp, the 240 line",
        "meeting_date": "2026-06-24",
        "fund": "ASB",
        "check_number": "418449",
        "vendor_like": "%Head Quarters%",
        "check_amount": Decimal("355.00"),
        "invoice_amount": Decimal("240.00"),
        "note": "printed: check 355, invoice 240, '2 Standard Portable Toilets for KM Athletics Use'. "
        "The regex parser read check 355240.00 and invoice 2.00.",
    },
    {
        "name": "2026-06-24 ACH 9252601789 Pacifica Law Group, the 106 line",
        "meeting_date": "2026-06-24",
        "fund": "ACH",
        "check_number": "9252601789",
        "vendor_like": "%Pacifica%",
        "check_amount": Decimal("40817.50"),
        "invoice_amount": Decimal("106.00"),
        "note": "printed invoice 106, not -10,625.00. The description begins '25-26' and this corpus "
        "writes some negatives with a trailing minus, so '106 25-' read as a credit.",
    },
    {
        "name": "2026-06-24 ACH 9252601789 Pacifica Law Group, the 53 line",
        "meeting_date": "2026-06-24",
        "fund": "ACH",
        "check_number": "9252601789",
        "vendor_like": "%Pacifica%",
        "check_amount": Decimal("40817.50"),
        "invoice_amount": Decimal("53.00"),
        "note": "printed invoice 53, not -5,325.00.",
    },
    {
        "name": "2026-06-24 GF 608288 GRMEA-Enumclaw HS, the 700 line",
        "meeting_date": "2026-06-24",
        "fund": "GF",
        "check_number": "608288",
        "vendor_like": "%GRMEA%",
        "check_amount": Decimal("760.00"),
        "invoice_amount": Decimal("700.00"),
        "note": "printed: check 760, invoice 700. The regex parser read check 760700.00 and invoice 2.00.",
    },
    {
        "name": "2026-06-24 GF 608502 UW Botanic Gardens",
        "meeting_date": "2026-06-24",
        "fund": "GF",
        "check_number": "608502",
        "vendor_like": "%Botanic%",
        "check_amount": Decimal("388.00"),
        "invoice_amount": Decimal("388.00"),
        "note": "printed: check 388, invoice 388. The regex parser read check 388388.00 and invoice 2.00.",
    },
)

# Whole checks whose invoice lines must sum to the check amount printed on
# every one of those lines. This is the document's own arithmetic.
#
# line_count for check 9252601789 is 12, read off the document. The build
# brief says 13; 2026-06-24 ACH prints twelve rows carrying that check
# number, eleven on page 27 and one on page 28, and those twelve sum to
# 40,817.50 exactly. The deviation is reported, and the assertion is
# anchored on the printed check amount rather than on the count.
HARD_CHECKS: tuple[dict, ...] = (
    {
        "name": "2026-06-24 ACH 9252601789 Pacifica Law Group",
        "meeting_date": "2026-06-24",
        "fund": "ACH",
        "check_number": "9252601789",
        "check_amount": Decimal("40817.50"),
        "line_count": 12,
    },
    {
        "name": "2026-06-24 GF 608288 GRMEA-Enumclaw HS",
        "meeting_date": "2026-06-24",
        "fund": "GF",
        "check_number": "608288",
        "check_amount": Decimal("760.00"),
        "line_count": 3,
    },
    {
        "name": "2026-06-24 GF 608502 UW Botanic Gardens",
        "meeting_date": "2026-06-24",
        "fund": "GF",
        "check_number": "608502",
        "check_amount": Decimal("388.00"),
        "line_count": 1,
    },
    {
        "name": "2026-06-24 ASB 418449 Head Quarters Corp",
        "meeting_date": "2026-06-24",
        "fund": "ASB",
        "check_number": "418449",
        "check_amount": Decimal("355.00"),
        "line_count": 2,
    },
)

# 2026-07-22 is the packet rendered with no inter-column whitespace at all.
# Every printed row must land in the table carrying either an amount or a
# reason code: a row that is on the page and not in the table is a silent
# drop, and this fixture exists to make that impossible to ship.
HARD_NO_SILENT_DROPS: tuple[dict, ...] = (
    {"meeting_date": "2026-07-22", "fund": "ACH", "rows": 1541},
    {"meeting_date": "2026-07-22", "fund": "Capital", "rows": 12},
)

# The cumulative-listing fixture the operator asked for. The 2021-02-10 and
# 2021-03-10 Transportation listings both total 1,175,094.00 while their
# recaps state 783,396.00 and 391,698.00 -- and those two sum to the third.
# This asserts the arithmetic AND that the build flags the overlap.
HARD_CUMULATIVE = {
    "fund": "Transportation",
    "dates": ("2021-02-10", "2021-03-10"),
    "listing_total": Decimal("1175094.00"),
    "recap_totals": (Decimal("783396.00"), Decimal("391698.00")),
}

# -------------------------------------------------------------- ADVISORY --
#
# From a prior manual parse. Deviations are reported, never failed on: a
# hand count is evidence, not an oracle, and where the two disagree the
# right response is to look, not to change the code until it agrees.
ADVISORY_COUNTS: dict[str, dict[str, tuple[int, int]]] = {
    # meeting_date -> fund -> (check_count, line_count)
    "2026-03-25": {"GF": (384, 1533), "ACH": (241, 1101), "Capital": (29, 40)},
    "2026-06-24": {
        "GF": (242, 1129),
        "ACH": (342, 1596),
        "Capital": (11, 13),
        "ASB": (75, 206),
    },
    "2026-07-22": {"ACH": (336, 1541), "Capital": (9, 12)},
    "2026-08-26": {
        "GF": (272, 708),
        "ACH": (382, 981),
        "Capital": (21, 26),
        "ASB": (12, 14),
    },
}

# Sum of invoice_amount across GF and ACH, vendor name containing the key.
ADVISORY_VENDORS: dict[str, dict[str, Decimal]] = {
    "2026-03-25": {
        "Sunburst Workforce Advisors": Decimal("722422.23"),
        "Blazerworks": Decimal("105425.90"),
        "CBPI": Decimal("12303.75"),
    },
    "2026-06-24": {"Sunburst Workforce Advisors": Decimal("300252.85")},
    "2026-07-22": {"Sunburst Workforce Advisors": Decimal("949107.73")},
    "2026-08-26": {
        "Sunburst Workforce Advisors": Decimal("308254.73"),
        "Elevation Healthcare": Decimal("191551.46"),
        "Blazerworks": Decimal("45891.20"),
        "CBPI": Decimal("11642.00"),
        "Positive Behavior Supports": Decimal("15112.50"),
    },
}


@dataclass
class Result:
    """Outcome of one fixture.

    Attributes:
        name: Identifier for the fixture.
        status: ``PASS``, ``FAIL``, ``BLOCKED`` or ``REPORT``.
        expected: What was asserted.
        actual: What the build produced.
        detail: Free text explaining the outcome.
    """

    name: str
    status: str
    expected: str = ""
    actual: str = ""
    detail: str = ""


@dataclass
class Suite:
    """Collected fixture results.

    Attributes:
        results: Every result, in run order.
    """

    results: list[Result] = field(default_factory=list)

    def add(self, result: Result) -> None:
        """Record a result.

        Args:
            result: The result to record.
        """
        self.results.append(result)

    def count(self, status: str) -> int:
        """Count results with a status.

        Args:
            status: Status to count.

        Returns:
            Number of matching results.
        """
        return sum(1 for r in self.results if r.status == status)

    @property
    def hard_failures(self) -> list[Result]:
        """Results that failed rather than being blocked."""
        return [r for r in self.results if r.status == "FAIL"]


def _sets_present() -> dict[tuple[str, str], dict]:
    """Load every voucher set, keyed by (meeting_date, fund).

    Returns:
        Mapping of key to the set's row.
    """
    rows = db.query_dicts(
        """
        SELECT set_id, meeting_date::text AS meeting_date, fund, stated_total,
               parsed_total, sum_check_dedup, line_count, check_count,
               hash_total, reconciled, delta, reason_code, notes, source
        FROM facts.voucher_set
        """,
        None,
    )
    return {(r["meeting_date"], r["fund"]): r for r in rows}


def check_hard_set_totals(suite: Suite, sets: dict) -> None:
    """Assert every stated set total the brief gives.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for meeting_date, funds in sorted(HARD_SET_TOTALS.items()):
        for fund, expected in sorted(funds.items()):
            name = f"hard_total {meeting_date} {fund}"
            row = sets.get((meeting_date, fund))
            if row is None:
                suite.add(
                    Result(
                        name,
                        "BLOCKED",
                        str(expected),
                        "no set",
                        "source document is not on this machine; stage it under "
                        "~/workspace/staging/vouchers-2026/<date>/ and re-run build.py",
                    )
                )
                continue
            stated = row["stated_total"]
            parsed = row["parsed_total"]
            if stated == expected and parsed == expected:
                suite.add(Result(name, "PASS", str(expected), str(parsed)))
            else:
                suite.add(
                    Result(
                        name,
                        "FAIL",
                        str(expected),
                        f"stated={stated} parsed={parsed}",
                        f"delta={row['delta']} reason={row['reason_code']}",
                    )
                )


def check_pre_2024(suite: Suite, sets: dict) -> None:
    """Assert the pre-2024 set chosen in Phase 0.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    spec = HARD_PRE_2024
    name = f"hard_pre_2024 {spec['meeting_date']} {spec['fund']}"
    row = sets.get((spec["meeting_date"], spec["fund"]))
    if row is None:
        suite.add(Result(name, "BLOCKED", str(spec["stated_total"]), "no set"))
        return
    if row["stated_total"] == spec["stated_total"] == row["parsed_total"]:
        suite.add(Result(name, "PASS", str(spec["stated_total"]), str(row["parsed_total"]), spec["note"]))
    else:
        suite.add(
            Result(
                name,
                "FAIL",
                str(spec["stated_total"]),
                f"stated={row['stated_total']} parsed={row['parsed_total']}",
                spec["note"],
            )
        )


def check_gf_components(suite: Suite, sets: dict) -> None:
    """Assert the 2026-06-24 General Fund decomposition.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    spec = HARD_GF_COMPONENTS
    name = "hard_components 2026-06-24 GF"
    if (spec["meeting_date"], spec["fund"]) not in sets:
        suite.add(
            Result(
                name,
                "BLOCKED",
                f"warrants {spec['warrant_sum']} + p-card {spec['pcard_sum']} + payroll {spec['payroll_amount']}",
                "no set",
                "arithmetic checks out (1,800,211.05 + 224,719.75 + 1,513.50 = "
                "2,026,444.30) but cannot be verified against a document",
            )
        )
        return
    lo, hi = spec["warrant_range"]
    rows = db.query_dicts(
        """
        SELECT COALESCE(sum(d.amount), 0) AS total
        FROM (
            SELECT DISTINCT l.check_number, l.check_amount AS amount
            FROM facts.voucher_line l
            JOIN facts.voucher_set s ON s.set_id = l.set_id
            WHERE s.meeting_date = %s AND s.fund = %s
              AND l.check_number ~ '^[0-9]+$'
              AND l.check_number::bigint BETWEEN %s AND %s
        ) AS d
        """,
        (spec["meeting_date"], spec["fund"], lo, hi),
    )
    warrant_total = rows[0]["total"] if rows else Decimal("0")
    plo, phi = spec["pcard_range"]
    rows = db.query_dicts(
        """
        SELECT COALESCE(sum(d.amount), 0) AS total
        FROM (
            SELECT DISTINCT l.check_number, l.check_amount AS amount
            FROM facts.voucher_line l
            JOIN facts.voucher_set s ON s.set_id = l.set_id
            WHERE s.meeting_date = %s AND s.fund = %s
              AND l.check_number ~ '^[0-9]+$'
              AND l.check_number::bigint BETWEEN %s AND %s
        ) AS d
        """,
        (spec["meeting_date"], spec["fund"], plo, phi),
    )
    pcard_total = rows[0]["total"] if rows else Decimal("0")
    rows = db.query_dicts(
        """
        SELECT COALESCE(sum(l.invoice_amount), 0) AS total
        FROM facts.voucher_line l
        JOIN facts.voucher_set s ON s.set_id = l.set_id
        WHERE s.meeting_date = %s AND s.fund = %s AND l.check_number = %s
        """,
        (spec["meeting_date"], spec["fund"], spec["payroll_check"]),
    )
    payroll_total = rows[0]["total"] if rows else Decimal("0")

    parts = [
        ("warrants", warrant_total, spec["warrant_sum"]),
        ("p-card", pcard_total, spec["pcard_sum"]),
        ("payroll", payroll_total, spec["payroll_amount"]),
    ]
    bad = [f"{label} {actual} != {want}" for label, actual, want in parts if actual != want]
    if bad:
        suite.add(Result(name, "FAIL", "three components", "; ".join(bad)))
    else:
        suite.add(Result(name, "PASS", "three components", "all three match"))


def check_hard_lines(suite: Suite, sets: dict) -> None:
    """Assert individual printed lines, as printed.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for spec in HARD_LINES:
        name = f"hard_line {spec['name']}"
        if (spec["meeting_date"], spec["fund"]) not in sets:
            suite.add(Result(name, "BLOCKED", str(spec["invoice_amount"]), "no set"))
            continue
        rows = db.query_dicts(
            """
            SELECT l.check_amount, l.invoice_amount, l.description, l.reason_code, l.locator_page
            FROM facts.voucher_line l
            JOIN facts.voucher_set s ON s.set_id = l.set_id
            WHERE s.meeting_date = %s AND s.fund = %s AND l.check_number = %s AND l.vendor_raw ILIKE %s
            ORDER BY l.line_seq
            """,
            (spec["meeting_date"], spec["fund"], spec["check_number"], spec["vendor_like"]),
        )
        expected = f"a line with check {spec['check_amount']} and invoice {spec['invoice_amount']}"
        hit = next(
            (
                r
                for r in rows
                if r["check_amount"] == spec["check_amount"]
                and r["invoice_amount"] == spec["invoice_amount"]
                and r["reason_code"] is None
            ),
            None,
        )
        if hit is not None:
            suite.add(Result(name, "PASS", expected, f"page {hit['locator_page']}: {hit['description']}"[:120]))
        else:
            seen = ", ".join(f"check {r['check_amount']}/invoice {r['invoice_amount']}" for r in rows[:6]) or "no lines"
            suite.add(Result(name, "FAIL", expected, seen, spec["note"]))


def check_hard_checks(suite: Suite, sets: dict) -> None:
    """Assert that a check's lines sum to the check amount printed on them.

    This is the document's own arithmetic, not an assumption: every row of
    a multi-invoice check repeats the same check amount, and the invoices
    against it must add up to it.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for spec in HARD_CHECKS:
        name = f"hard_check {spec['name']}"
        if (spec["meeting_date"], spec["fund"]) not in sets:
            suite.add(Result(name, "BLOCKED", str(spec["check_amount"]), "no set"))
            continue
        rows = db.query_dicts(
            """
            SELECT count(*) AS lines,
                   count(DISTINCT l.check_amount) AS distinct_check_amounts,
                   min(l.check_amount) AS check_amount,
                   COALESCE(sum(l.invoice_amount), 0) AS invoice_sum,
                   count(*) FILTER (WHERE l.reason_code IS NOT NULL) AS unread
            FROM facts.voucher_line l
            JOIN facts.voucher_set s ON s.set_id = l.set_id
            WHERE s.meeting_date = %s AND s.fund = %s AND l.check_number = %s
            """,
            (spec["meeting_date"], spec["fund"], spec["check_number"]),
        )
        row = rows[0]
        problems = []
        if row["distinct_check_amounts"] != 1 or row["check_amount"] != spec["check_amount"]:
            problems.append(
                f"check amount {row['check_amount']} over {row['distinct_check_amounts']} distinct value(s), "
                f"expected exactly {spec['check_amount']}"
            )
        if row["invoice_sum"] != spec["check_amount"]:
            problems.append(f"invoice lines sum to {row['invoice_sum']}, not {spec['check_amount']}")
        if spec["line_count"] is not None and row["lines"] != spec["line_count"]:
            problems.append(f"{row['lines']} lines, expected {spec['line_count']}")
        if row["unread"]:
            problems.append(f"{row['unread']} line(s) carry a reason code")
        expected = f"{spec['line_count'] or 'all'} lines summing to {spec['check_amount']}"
        if problems:
            suite.add(Result(name, "FAIL", expected, "; ".join(problems)))
        else:
            suite.add(Result(name, "PASS", expected, f"{row['lines']} lines, sum {row['invoice_sum']}"))


def check_no_silent_drops(suite: Suite, sets: dict) -> None:
    """Assert every printed row is in the table with an amount or a reason.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for spec in HARD_NO_SILENT_DROPS:
        name = f"hard_no_silent_drops {spec['meeting_date']} {spec['fund']}"
        row = sets.get((spec["meeting_date"], spec["fund"]))
        if row is None:
            suite.add(Result(name, "BLOCKED", f"{spec['rows']} rows", "no set"))
            continue
        counts = db.query_dicts(
            """
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE l.invoice_amount IS NOT NULL AND l.reason_code IS NULL) AS with_amount,
                   count(*) FILTER (WHERE l.reason_code IS NOT NULL) AS with_reason,
                   count(*) FILTER (WHERE l.invoice_amount IS NULL AND l.reason_code IS NULL) AS neither
            FROM facts.voucher_line l
            JOIN facts.voucher_set s ON s.set_id = l.set_id
            WHERE s.meeting_date = %s AND s.fund = %s
            """,
            (spec["meeting_date"], spec["fund"]),
        )[0]
        problems = []
        if counts["rows"] != spec["rows"]:
            problems.append(f"{counts['rows']} rows in the table, {spec['rows']} printed")
        if counts["neither"]:
            problems.append(f"{counts['neither']} row(s) carry neither an amount nor a reason code")
        expected = f"{spec['rows']} rows, each with an amount or a reason code"
        actual = f"{counts['rows']} rows: {counts['with_amount']} with an amount, {counts['with_reason']} with a reason"
        if problems:
            suite.add(Result(name, "FAIL", expected, actual, "; ".join(problems)))
        else:
            suite.add(Result(name, "PASS", expected, actual))


def check_cumulative(suite: Suite, sets: dict) -> None:
    """Assert the cumulative-listing arithmetic and that it is flagged.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    spec = HARD_CUMULATIVE
    a, b = spec["recap_totals"]
    name = "hard_cumulative Transportation 2021-02-10 / 2021-03-10"
    if a + b != spec["listing_total"]:
        suite.add(Result(name, "FAIL", str(spec["listing_total"]), str(a + b), "recap arithmetic"))
        return
    rows = [sets.get((d, spec["fund"])) for d in spec["dates"]]
    if any(r is None for r in rows):
        suite.add(Result(name, "BLOCKED", str(spec["listing_total"]), "set missing"))
        return
    totals = [r["parsed_total"] for r in rows]
    flagged = [r for r in rows if r["notes"] and "cumulative" in r["notes"]]
    problems = []
    if not all(t == spec["listing_total"] for t in totals):
        problems.append(f"listing totals {totals} != {spec['listing_total']}")
    if not flagged:
        problems.append("neither set is flagged as cumulative")
    if problems:
        suite.add(Result(name, "FAIL", str(spec["listing_total"]), "; ".join(problems)))
    else:
        suite.add(
            Result(
                name,
                "PASS",
                f"{a} + {b} = {spec['listing_total']}",
                f"both listings total {spec['listing_total']}, {len(flagged)} flagged cumulative",
            )
        )


def check_no_silent_acceptance(suite: Suite) -> None:
    """Assert that every unreconciled set carries a reason code.

    This is the "reconcile or flag" contract itself. A set that neither
    reconciles nor explains itself is the one failure mode the whole design
    exists to prevent.

    Args:
        suite: Suite to record into.
    """
    rows = db.query_dicts(
        """
        SELECT set_id, meeting_date::text AS meeting_date, fund, reconciled, delta
        FROM facts.voucher_set
        WHERE reconciled IS DISTINCT FROM TRUE AND reason_code IS NULL
        """,
        None,
    )
    if rows:
        suite.add(
            Result(
                "contract_no_silent_acceptance",
                "FAIL",
                "0 unreconciled sets without a reason code",
                f"{len(rows)}: {[r['set_id'] for r in rows[:10]]}",
            )
        )
    else:
        suite.add(
            Result(
                "contract_no_silent_acceptance",
                "PASS",
                "0 unreconciled sets without a reason code",
                "0",
            )
        )


def check_every_line_has_a_locator(suite: Suite) -> None:
    """Assert that every line carries a resolvable locator.

    Args:
        suite: Suite to record into.
    """
    rows = db.query_dicts(
        """
        SELECT count(*) AS n FROM facts.voucher_line
        WHERE locator_file_path IS NULL
           OR locator_file_sha256 IS NULL
           OR locator_char_offset IS NULL
           OR locator_quote IS NULL
           OR locator_quote = ''
        """,
        None,
    )
    missing = rows[0]["n"]
    status = "PASS" if missing == 0 else "FAIL"
    suite.add(Result("contract_every_line_has_a_locator", status, "0", str(missing)))


def check_advisory_counts(suite: Suite, sets: dict) -> None:
    """Report deviations from the prior manual parse's counts.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for meeting_date, funds in sorted(ADVISORY_COUNTS.items()):
        for fund, (checks, lines) in sorted(funds.items()):
            name = f"advisory_counts {meeting_date} {fund}"
            row = sets.get((meeting_date, fund))
            if row is None:
                suite.add(Result(name, "BLOCKED", f"{checks} checks / {lines} lines", "no set"))
                continue
            actual = f"{row['check_count']} checks / {row['line_count']} lines"
            expected = f"{checks} checks / {lines} lines"
            match = row["check_count"] == checks and row["line_count"] == lines
            suite.add(
                Result(
                    name,
                    "REPORT",
                    expected,
                    actual,
                    "match" if match else "DEVIATION -- reported, not failed",
                )
            )


def check_advisory_vendors(suite: Suite, sets: dict) -> None:
    """Report deviations from the prior manual parse's vendor totals.

    Args:
        suite: Suite to record into.
        sets: Loaded voucher sets.
    """
    for meeting_date, vendors in sorted(ADVISORY_VENDORS.items()):
        have_any = any((meeting_date, f) in sets for f in ("GF", "ACH"))
        for needle, expected in sorted(vendors.items()):
            name = f"advisory_vendor {meeting_date} {needle}"
            if not have_any:
                suite.add(Result(name, "BLOCKED", str(expected), "no GF/ACH set"))
                continue
            rows = db.query_dicts(
                """
                SELECT COALESCE(sum(l.invoice_amount), 0) AS total
                FROM facts.voucher_line l
                JOIN facts.voucher_set s ON s.set_id = l.set_id
                WHERE s.meeting_date = %s AND s.fund IN ('GF', 'ACH')
                  AND l.vendor_raw ILIKE %s
                """,
                (meeting_date, f"%{needle}%"),
            )
            actual = rows[0]["total"]
            suite.add(
                Result(
                    name,
                    "REPORT",
                    str(expected),
                    str(actual),
                    "match" if actual == expected else f"DEVIATION delta={actual - expected}",
                )
            )


def run() -> Suite:
    """Run every fixture.

    Returns:
        The completed suite.
    """
    suite = Suite()
    sets = _sets_present()
    check_hard_set_totals(suite, sets)
    check_hard_lines(suite, sets)
    check_hard_checks(suite, sets)
    check_no_silent_drops(suite, sets)
    check_pre_2024(suite, sets)
    check_gf_components(suite, sets)
    check_cumulative(suite, sets)
    check_no_silent_acceptance(suite)
    check_every_line_has_a_locator(suite)
    check_advisory_counts(suite, sets)
    check_advisory_vendors(suite, sets)
    return suite


def main() -> int:
    """Run the fixtures and print a report.

    Returns:
        Process exit code: 1 when any HARD fixture failed, else 0.
    """
    suite = run()
    width = max(len(r.name) for r in suite.results)
    for result in suite.results:
        line = f"{result.status:8s} {result.name:{width}s}"
        if result.expected or result.actual:
            line += f"  expected={result.expected}  actual={result.actual}"
        if result.detail:
            line += f"\n{'':9s}{result.detail}"
        print(line)
    print()
    print(
        f"PASS {suite.count('PASS')}   FAIL {suite.count('FAIL')}   "
        f"BLOCKED {suite.count('BLOCKED')}   REPORT {suite.count('REPORT')}"
    )
    if suite.count("BLOCKED"):
        print("\nBLOCKED fixtures did not run. A check that never executed is not a check that succeeded.")
    return 1 if suite.hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
