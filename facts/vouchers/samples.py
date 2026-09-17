"""Draw traceable samples so the operator can audit the parse by hand.

For every 2026 fund-set that reconciles, this writes a seeded random sample
of up to 300 lines with the page and quote needed to find each one in the
source PDF. The operator checks them against the document; this script does
not judge them, and nothing here decides whether a line is right.

Why 300, and what a clean result licenses you to say:

    With 300 lines drawn at random from a set, **zero** errors found puts
    the error rate below about **1% at 95% confidence**. That is the
    "rule of three": with no failures in n trials, the upper bound on the
    failure rate is roughly 3/n, so 3/300 = 1%.

That statement is only available if the sample is genuinely random and the
seed is recorded, which is why the seed is written into every file. It says
nothing about sets that were not sampled, and nothing about the sets that
did not reconcile.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from decimal import Decimal

import db
from vendors import is_exportable, publishable_name, redact_name

SAMPLE_SIZE = 300
DEFAULT_SEED = 20260914

# Cycles whose sample size is doubled, and why.
#
# 2026-07-22 is rendered with no inter-column whitespace at all, so its
# figures reach these tables on the column geometry alone. Its signed
# register is a scan with no text layer, so there is no second machine
# readable source to check them against. The operator's trace-to-source is
# the only independent check those numbers get, and it is worth twice as
# many lines.
DOUBLE_SAMPLE_CYCLES: dict[str, str] = {
    "2026-07-22": (
        "This packet prints no whitespace between its columns, so every figure here was read from the "
        "characters' x coordinates alone, and the signed register for this meeting is a scan with no text "
        "layer. There is no second machine-readable source for these numbers, so this sample is twice the "
        "usual size and it is the only independent check they get."
    ),
}

SETS_SQL = """
    SELECT set_id, meeting_date::text AS meeting_date, fund, stated_total,
           parsed_total, line_count, check_count, reconciled,
           locator_file_path, locator_file_sha256
    FROM facts.voucher_set
    WHERE meeting_date >= %s AND reconciled IS TRUE
    ORDER BY meeting_date, fund
"""

LINES_SQL = """
    SELECT l.line_seq, l.vendor_raw, l.check_date::text AS check_date, l.check_number,
           l.check_amount, l.invoice_amount, l.description, l.is_pcard,
           l.is_payroll_warrant, l.is_credit, l.amount_paren, l.reason_code,
           l.locator_page, l.locator_char_offset, l.locator_quote,
           EXISTS (SELECT 1 FROM facts.voucher_line h
                   WHERE h.vendor_norm = l.vendor_norm
                     AND h.description ~* 'payroll\\s+handwrite') AS payroll_handwrite
    FROM facts.voucher_line l
    WHERE l.set_id = %s
    ORDER BY l.line_seq
"""


def _withheld(row: dict) -> bool:
    """Whether this row's payee name must not be printed.

    Delegates to the export layer's classifier rather than repeating its
    rule. A sample file and an export that decide this separately will
    disagree eventually, and the first sign of the disagreement would be a
    person's name in a document that has already been handed out.

    Args:
        row: A voucher line row.

    Returns:
        True when the payee name is withheld.
    """
    return not is_exportable(row["vendor_raw"], None, bool(row.get("payroll_handwrite")))


def money(value: Decimal | None) -> str:
    """Format an amount for a sample table.

    Args:
        value: Amount, or None.

    Returns:
        A comma-grouped string, or an em dash.
    """
    return "—" if value is None else f"{value:,.2f}"


def _filename(set_id: str) -> str:
    """Turn a set id into a file name that cannot collide.

    Args:
        set_id: The set's identifier, e.g. ``2026-02-11:Permanent#2``.

    Returns:
        A file name, e.g. ``2026-02-11-Permanent-2.md``.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", set_id.replace(":", "-").replace("#", "-")) + ".md"


def sample_size(meeting_date: str, base: int) -> int:
    """Return the sample size for a cycle.

    Args:
        meeting_date: ISO meeting date.
        base: The default sample size.

    Returns:
        The size to draw.
    """
    return base * 2 if meeting_date in DOUBLE_SAMPLE_CYCLES else base


def render(set_row: dict, lines: list[dict], seed: int, drawn: int) -> str:
    """Render one set's sample as markdown.

    Args:
        set_row: The voucher set.
        lines: The sampled lines, in document order.
        seed: The RNG seed used.
        drawn: How many lines were drawn.

    Returns:
        Markdown text.
    """
    out: list[str] = []
    add = out.append
    add(f"# Sample — {set_row['meeting_date']} {set_row['fund']}")
    add("")
    add(f"- **Set:** `{set_row['set_id']}`")
    add(f"- **Source file:** `{set_row['locator_file_path']}`")
    add(f"- **SHA-256:** `{set_row['locator_file_sha256']}`")
    add(
        f"- **Set totals:** stated {money(set_row['stated_total'])}, "
        f"parsed {money(set_row['parsed_total'])}, "
        f"{set_row['line_count']:,} lines, {set_row['check_count']:,} checks"
    )
    add(f"- **Sample:** {drawn} of {set_row['line_count']:,} lines, seed `{seed}`")
    add("")
    if set_row["meeting_date"] in DOUBLE_SAMPLE_CYCLES:
        add(f"> **{DOUBLE_SAMPLE_CYCLES[set_row['meeting_date']]}**")
        add("")
    add(
        "Open the source file at the page in each row and confirm the vendor, "
        "date, check number and both amounts match. Mark anything that does not."
    )
    add("")
    add(
        f"**If you find zero errors in these {drawn} lines**, the error rate for this set is below roughly "
        f"{3 / drawn:.1%} at 95% confidence (the rule of three: 3/{drawn}). A smaller sample gives a "
        f"correspondingly weaker bound, and this says nothing about any set that was not sampled."
    )
    add("")
    withheld_rows = sum(1 for row in lines if _withheld(row))
    if withheld_rows:
        add(
            f"> **{withheld_rows} of these {len(lines)} rows name an individual rather than a business, and "
            f"the name is withheld.** The same classifier the published exports use decides this, so the two "
            f"cannot drift apart. A withheld row still carries its page, check number and both amounts, which "
            f"is everything needed to find it in the source PDF and check the arithmetic — open the page and "
            f"the name is there. The verbatim quote is redacted for the same reason the column is: the quote "
            f"is the source line, and it carries the name."
        )
        add("")
    add("| # | Page | Payee | Check date | Check no. | Check amt | Invoice amt | Description | Flags |")
    add("|---:|---:|---|---|---|---:|---:|---|---|")
    for row in lines:
        flags = []
        if row["is_pcard"]:
            flags.append("P-card")
        if row["is_payroll_warrant"]:
            flags.append("payroll")
        if row["is_credit"]:
            flags.append("credit")
        if row.get("amount_paren"):
            flags.append("(negative)")
        if row["reason_code"]:
            flags.append(f"**{row['reason_code']}**")
        description = (row["description"] or "").replace("|", "\\|")[:70]
        vendor = publishable_name(row["vendor_raw"], None, bool(row.get("payroll_handwrite")))
        add(
            f"| {row['line_seq']} | {row['locator_page'] or '—'} | {vendor.replace('|', chr(92) + '|')} | "
            f"{row['check_date'] or '—'} | {row['check_number'] or '—'} | "
            f"{money(row['check_amount'])} | {money(row['invoice_amount'])} | "
            f"{description} | {' '.join(flags)} |"
        )
    add("")
    add("## Verbatim quotes")
    add("")
    add("Each line as extracted, for exact comparison against the page:")
    add("")
    for row in lines:
        quote = (row["locator_quote"] or "")[:200]
        if _withheld(row):
            quote = redact_name(quote, row["vendor_raw"])
        add(f"- `#{row['line_seq']}` p{row['locator_page']} @{row['locator_char_offset']}: `{quote}`")
    # No trailing blank line: the file must end with exactly one newline or
    # the end-of-file-fixer pre-commit hook rewrites it on every regeneration.
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def main() -> int:
    """Write one sample file per reconciling 2026 fund-set.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-01-01", help="earliest meeting date")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE, help="lines per set")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"),
        help="output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sets = db.query_dicts(SETS_SQL, (args.since,))
    if not sets:
        print(f"no reconciling sets on or after {args.since}")
        return 0

    written = []
    for set_row in sets:
        lines = db.query_dicts(LINES_SQL, (set_row["set_id"],))
        if not lines:
            continue
        # Seeded per set so adding a set does not reshuffle the others, and
        # so the operator can re-draw exactly the same sample later.
        rng = random.Random(f"{args.seed}:{set_row['set_id']}")  # noqa: S311 - sampling, not key material
        drawn = min(sample_size(set_row["meeting_date"], args.size), len(lines))
        sample = sorted(rng.sample(lines, drawn), key=lambda r: r["line_seq"])
        # Named from the set id, not from (date, fund). Two listings for
        # the same fund and meeting do occur -- 2026-02-11 carries two
        # Permanent Fund sets whose contents differ -- and naming the file
        # after the fund alone made the second silently overwrite the
        # first, which is a sample file the operator never got to see.
        name = _filename(set_row["set_id"])
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render(set_row, sample, args.seed, drawn))
        written.append((path, drawn, len(lines)))

    for path, drawn, total in written:
        print(f"wrote {path}  ({drawn} of {total} lines)")
    print(f"\n{len(written)} sample file(s), seed {args.seed}")
    print(
        "Accept-on-zero: 0 errors in n sampled lines puts that set's error rate below about 3/n "
        "at 95% confidence. Cycles in DOUBLE_SAMPLE_CYCLES are drawn at twice the size."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
