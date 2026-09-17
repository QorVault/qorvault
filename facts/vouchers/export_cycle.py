"""Export one voucher cycle as a briefing a director can read out loud.

Two rules govern what leaves this script:

**Nothing unreconciled is presented as a figure.** Every total carries the
reconciliation status of the set it came from, and a set that does not tie
to its own printed total is labelled on the line where its number appears,
not in a footnote.

**No personal name is ever published.** The export uses an allow-list --
watch-list vendors plus names that positively identify as organizations --
rather than a block-list of personal names. A block-list has to recognise
every individual to be safe; an allow-list only has to recognise companies.
Refund and reimbursement lines to individuals stay in ``facts.voucher_line``
and stay queryable; they are withheld from the published file. Some small
businesses are withheld as a side effect, and that is the correct direction
for a privacy control to fail.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date
from decimal import Decimal

import db
from vendors import is_exportable, normalize_vendor

# Vendors the operator has named for standing attention. These are published
# by name regardless of whether the suffix rules recognise them: they are an
# explicit, reviewable decision rather than a pattern match.
WATCH_LIST: tuple[str, ...] = (
    "Sunburst Workforce Advisors",
    "Elevation Healthcare",
    "Blazerworks",
    "CBPI",
    "Positive Behavior Supports",
    "Robert Half",
    "Pacifica Law Group",
    "Foster Garvey",
    "KCABA",
    "Gersh Academy",
    "Renton SD",
    "Tacoma SD",
    "Yellow Wood Academy",
    "Hazel Health",
    "Kent Youth & Family Services",
    "St Vincent de Paul",
    "Communities in Schools",
)

WATCH_LIST_NORMS = frozenset(normalize_vendor(name) for name in WATCH_LIST)

TOP_VENDORS = 25
WATCH_CYCLES = 12
MAX_LINES = 300

SETS_SQL = """
    SELECT t.set_id, t.fund, t.stated_total, t.parsed_total, t.sum_check_dedup,
           t.line_count, t.check_count, t.hash_total, t.reconciled, t.delta,
           t.reason_code, t.status, t.notes, t.period_start, t.period_end,
           t.pcard_period_start, t.pcard_period_end, t.source,
           t.locator_file_path, t.locator_page, t.locator_quote, t.dan,
           d.parsed_total_new_checks, d.restated_from_earlier,
           d.restated_lines, d.restated_checks, d.unread_lines,
           a.reason        AS crosscheck_reason,
           a.has_crosscheck,
           a.crosscheck_note
    FROM facts.set_totals t
    JOIN facts.set_totals_deduped     d ON d.set_id = t.set_id
    LEFT JOIN facts.register_availability a ON a.set_id = t.set_id
    WHERE t.meeting_date = %s
    ORDER BY t.fund
"""

VENDORS_SQL = """
    SELECT l.vendor_raw, l.vendor_norm,
           sum(l.invoice_amount) FILTER (WHERE l.reason_code IS NULL)  AS invoice_total,
           sum(l.invoice_amount) FILTER (
               WHERE l.reason_code IS NULL AND l.is_first_cycle_for_check)
                                                                       AS invoice_total_new_checks,
           count(*) FILTER (WHERE l.reason_code IS NULL)               AS line_count,
           count(DISTINCT l.check_number)                              AS check_count,
           count(*) FILTER (WHERE l.reason_code IS NOT NULL)           AS unread_lines,
           bool_and(l.reconciled IS TRUE)                              AS all_reconciled,
           -- Feeds the classifier's Payroll Handwrite override. Measured
           -- over the payee's WHOLE history, not just this cycle: a payee
           -- who is an employee in March is an employee in May, and a
           -- cycle-local test would publish them in the cycles where the
           -- hand-written cheque happens not to fall.
           EXISTS (SELECT 1 FROM facts.voucher_line h
                   WHERE h.vendor_norm = l.vendor_norm
                     AND h.description ~* 'payroll\\s+handwrite')      AS payroll_handwrite,
           min(l.locator_page)                                         AS first_page
    FROM facts.voucher_line_deduped l
    WHERE l.meeting_date = %s
    GROUP BY l.vendor_raw, l.vendor_norm
    HAVING sum(l.invoice_amount) FILTER (WHERE l.reason_code IS NULL) IS NOT NULL
    ORDER BY sum(l.invoice_amount) FILTER (WHERE l.reason_code IS NULL) DESC
"""

CYCLES_SQL = """
    SELECT DISTINCT meeting_date::text AS meeting_date
    FROM facts.voucher_set
    WHERE meeting_date <= %s
    ORDER BY meeting_date DESC
    LIMIT %s
"""

WATCH_SQL = """
    SELECT l.meeting_date::text AS meeting_date,
           sum(l.invoice_amount) FILTER (
               WHERE l.reason_code IS NULL AND l.is_first_cycle_for_check) AS invoice_total
    FROM facts.voucher_line_deduped l
    WHERE l.meeting_date = ANY(%s::date[])
      AND l.vendor_raw ILIKE %s
    GROUP BY 1
"""

RECON_SQL = """
    SELECT fund, basis, register_total, detail_total, delta, match, reason,
           register_warrant_range, locator_register_page, locator_register_quote
    FROM facts.register_crosschecks
    WHERE meeting_date = %s
    ORDER BY fund, basis
"""


def _crosscheck(row: dict) -> str:
    """Describe whether a set has an independent register cross-check.

    Stated per set rather than left to a reader to notice. The signed
    registers for 2026-06-24 and 2026-07-22 are scans with no text layer,
    so those two nights have no second opinion on this machine and the
    listing is the only machine-readable source for them.

    Args:
        row: A set row carrying the cross-check reason.

    Returns:
        A short phrase for the table cell.
    """
    return {
        "MATCH": "yes — ties to the signed register",
        "REGISTER_MISMATCH": "yes — differs from the signed register",
        "SCOPE_DIFFERS": "Warrant Recap only (wider scope)",
        "REGISTER_NO_TEXT": "**none** — register is a scan with no text layer",
        "REGISTER_NOT_FOUND": "**none** — no register on this machine",
    }.get(row["crosscheck_reason"], "**none recorded**")


def money(value: Decimal | None) -> str:
    """Format an amount for a printed table.

    Args:
        value: Amount, or None.

    Returns:
        A comma-grouped string, or an em dash when the value is absent.
    """
    if value is None:
        return "—"
    return f"{value:,.2f}"


def load_cycle(meeting_date: str) -> dict:
    """Load everything the export needs for one voucher night.

    Args:
        meeting_date: ISO meeting date.

    Returns:
        A dict of sets, vendors, watch-list rows and register cross-checks.
    """
    sets = db.query_dicts(SETS_SQL, (meeting_date,))
    vendors = db.query_dicts(VENDORS_SQL, (meeting_date,))
    recon = db.query_dicts(RECON_SQL, (meeting_date,))
    cycles = [r["meeting_date"] for r in db.query_dicts(CYCLES_SQL, (meeting_date, WATCH_CYCLES))]
    cycles.reverse()
    watch: dict[str, dict[str, Decimal]] = {}
    for name in WATCH_LIST:
        rows = db.query_dicts(WATCH_SQL, (cycles, f"%{name}%"))
        watch[name] = {r["meeting_date"]: r["invoice_total"] for r in rows}
    return {
        "meeting_date": meeting_date,
        "sets": sets,
        "vendors": vendors,
        "recon": recon,
        "cycles": cycles,
        "watch": watch,
    }


def exportable_vendors(vendors: list[dict]) -> tuple[list[dict], int, Decimal]:
    """Split vendors into those that may be published and those that may not.

    Args:
        vendors: Vendor rows for the cycle.

    Returns:
        ``(publishable, withheld_count, withheld_total)``.
    """
    publishable, withheld, withheld_total = [], 0, Decimal("0")
    for row in vendors:
        if is_exportable(row["vendor_raw"], WATCH_LIST_NORMS, bool(row.get("payroll_handwrite"))):
            publishable.append(row)
        else:
            withheld += 1
            withheld_total += row["invoice_total"] or Decimal("0")
    return publishable, withheld, withheld_total


def render(cycle: dict) -> str:
    """Render the cycle as markdown.

    Args:
        cycle: Output of :func:`load_cycle`.

    Returns:
        Markdown text, bounded to ``MAX_LINES`` lines.
    """
    meeting_date = cycle["meeting_date"]
    out: list[str] = []
    add = out.append

    add(f"# Vouchers — {meeting_date}")
    add("")
    add(
        "Generated from `facts.voucher_set` / `facts.voucher_line`. Every figure "
        "traces to a page in a source PDF; the locator column names the page."
    )
    add("")

    if not cycle["sets"]:
        add(f"**No voucher set is on file for {meeting_date}.**")
        return "\n".join(out) + "\n"

    # ---------------------------------------------------------- set totals --
    add("## Set totals")
    add("")
    add("| Fund | Stated total | Parsed total | Lines | Held | Checks | Status | Register cross-check | Page |")
    add("|---|---:|---:|---:|---:|---:|---|---|---:|")
    grand_reconciled = Decimal("0")
    any_unreconciled = False
    for row in cycle["sets"]:
        add(
            f"| {row['fund']} | {money(row['stated_total'])} | "
            f"{money(row['parsed_total'])} | {row['line_count']:,} | "
            f"{row['unread_lines'] or 0:,} | {row['check_count']:,} | {row['status']} | "
            f"{_crosscheck(row)} | {row['locator_page'] or '—'} |"
        )
        if row["reconciled"] is True:
            grand_reconciled += row["parsed_total"] or Decimal("0")
        else:
            any_unreconciled = True
    add("")
    add(f"**Total across sets that reconcile: {money(grand_reconciled)}**")
    periods = [
        f"{r['fund']} {r['period_start']} → {r['period_end']}"
        for r in cycle["sets"]
        if r["period_start"] and r["period_end"]
    ]
    if periods:
        add("")
        add("Warrant periods: " + "; ".join(periods) + ".")
    held = sum(r["unread_lines"] or 0 for r in cycle["sets"])
    if held:
        add("")
        add(
            f"**Held:** {held:,} printed row(s) across these sets could not be assigned to columns and are "
            f"excluded from every total above. They are not lost: each is in `facts.unread_lines` with the "
            f"page it is on and the reason it could not be read."
        )
    restated = [r for r in cycle["sets"] if (r["restated_lines"] or 0) > 0]
    if restated:
        add("")
        add(
            "**Already counted at an earlier meeting:** "
            + "; ".join(
                f"{r['fund']} {money(r['restated_from_earlier'])} on {r['restated_checks']:,} check(s)"
                for r in restated
            )
            + ". These listings restate warrants from earlier cycles. Adding this cycle's totals to an "
            "earlier cycle's counts that money twice; the deduplicated figure is in "
            "`facts.set_totals_deduped.parsed_total_new_checks`."
        )
    if any_unreconciled:
        add("")
        add(
            "> One or more sets above do not reconcile to their own printed total, "
            "or print no total at all. Their figures are shown but are **not** "
            "included in the total on the line above, and should not be quoted as "
            "settled."
        )
    add("")

    # P-card and control-total notes, only when they apply.
    pcard = [r for r in cycle["sets"] if r["pcard_period_start"]]
    if pcard:
        add(
            "P-card periods differ from warrant periods: "
            + "; ".join(f"{r['fund']} {r['pcard_period_start']} → {r['pcard_period_end']}" for r in pcard)
            + "."
        )
        add("")
    drift = [r for r in cycle["sets"] if (r["parsed_total"] or 0) != (r["sum_check_dedup"] or 0)]
    if drift:
        add(
            "Second control total (sum of deduplicated check amounts) differs from "
            "the invoice total on: "
            + "; ".join(
                f"{r['fund']} by {money((r['sum_check_dedup'] or Decimal(0)) - (r['parsed_total'] or Decimal(0)))}"
                for r in drift
            )
            + ". This is normal where a P-card statement total exceeds the "
            "transactions itemised, and where a credit rides on a sentinel check "
            "number."
        )
        add("")

    # ------------------------------------------------- register cross-check --
    if cycle["recon"]:
        add("## Checked against the board's signed register")
        add("")
        add("| Fund | Basis | Register | Listing | Δ | Result | Warrant range |")
        add("|---|---|---:|---:|---:|---|---|")
        for row in cycle["recon"]:
            rng = (row["register_warrant_range"] or "—")[:40]
            add(
                f"| {row['fund']} | {row['basis']} | {money(row['register_total'])} | "
                f"{money(row['detail_total'])} | {money(row['delta'])} | "
                f"{row['reason']} | {rng} |"
            )
        add("")

    # ------------------------------------------------------------- vendors --
    publishable, withheld, withheld_total = exportable_vendors(cycle["vendors"])
    add(f"## Top {TOP_VENDORS} vendors this cycle")
    add("")
    add("| Vendor | Invoice total | Of which new checks | Lines | Checks | From reconciled sets |")
    add("|---|---:|---:|---:|---:|---|")
    for row in publishable[:TOP_VENDORS]:
        add(
            f"| {row['vendor_raw']} | {money(row['invoice_total'])} | "
            f"{money(row['invoice_total_new_checks'])} | "
            f"{row['line_count']:,} | {row['check_count']:,} | "
            f"{'yes' if row['all_reconciled'] else 'NO'} |"
        )
    add("")
    add(
        f"{withheld} payee(s) totalling {money(withheld_total)} are withheld from "
        f"this export because they are not positively identifiable as "
        f"organizations. They remain in `facts.voucher_line` and are queryable; "
        f"they are not published because refunds and reimbursements to named "
        f"individuals should not appear in a public briefing."
    )
    add("")

    # ---------------------------------------------------------- watch list --
    add(f"## Watch list, last {len(cycle['cycles'])} cycles")
    add("")
    header = " | ".join(c[5:] for c in cycle["cycles"])
    add(f"| Vendor | {header} |")
    add("|---" * (len(cycle["cycles"]) + 1) + "|")
    for name in WATCH_LIST:
        by_cycle = cycle["watch"][name]
        if not by_cycle:
            continue
        cells = " | ".join(money(by_cycle.get(c)) if by_cycle.get(c) is not None else "—" for c in cycle["cycles"])
        add(f"| {name} | {cells} |")
    add("")
    add(
        "Blank cells mean the vendor does not appear in that cycle's listings, not "
        "that it was paid nothing outside them."
    )
    add("")
    add("---")
    add("")
    add(
        f"Records retention: DAN {cycle['sets'][0]['dan']} (Washington State "
        f"Archives). Source files are named in `facts.voucher_set.locator_file_path`."
    )

    while out and not out[-1]:
        out.pop()
    text = "\n".join(out)
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        lines = lines[: MAX_LINES - 2] + ["", f"_(truncated at {MAX_LINES} lines)_"]
        text = "\n".join(lines)
    return text + "\n"


def write_csv(cycle: dict, path: str) -> None:
    """Write the same content as CSV.

    Args:
        cycle: Output of :func:`load_cycle`.
        path: Destination path.
    """
    publishable, _, _ = exportable_vendors(cycle["vendors"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "fund", "value", "detail"])
        for row in cycle["sets"]:
            writer.writerow(
                [
                    "set_total",
                    cycle["meeting_date"],
                    row["fund"],
                    row["parsed_total"],
                    f"stated={row['stated_total']};status={row['status']};"
                    f"lines={row['line_count']};checks={row['check_count']};"
                    f"hash_total={row['hash_total']}",
                ]
            )
        for row in cycle["recon"]:
            writer.writerow(
                [
                    "register_crosscheck",
                    cycle["meeting_date"],
                    row["fund"],
                    row["delta"],
                    f"basis={row['basis']};register={row['register_total']};"
                    f"listing={row['detail_total']};reason={row['reason']}",
                ]
            )
        for row in publishable[:TOP_VENDORS]:
            writer.writerow(
                [
                    "top_vendor",
                    row["vendor_raw"],
                    "",
                    row["invoice_total"],
                    f"lines={row['line_count']};checks={row['check_count']};all_reconciled={row['all_reconciled']}",
                ]
            )
        for name in WATCH_LIST:
            for meeting_date, total in sorted(cycle["watch"][name].items()):
                writer.writerow(["watch_list", name, "", total, meeting_date])


def main() -> int:
    """Render one cycle to markdown and CSV.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_date", help="ISO meeting date, e.g. 2026-03-25")
    parser.add_argument("-o", "--output", help="markdown output path")
    args = parser.parse_args()

    date.fromisoformat(args.meeting_date)  # reject a malformed date early

    cycle = load_cycle(args.meeting_date)
    text = render(cycle)

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exports",
        f"facts-vouchers-{args.meeting_date}.md",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    write_csv(cycle, csv_path)
    print(f"wrote {out_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
