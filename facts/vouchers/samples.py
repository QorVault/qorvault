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
from decimal import Decimal

import db

SAMPLE_SIZE = 300
DEFAULT_SEED = 20260914

SETS_SQL = """
    SELECT set_id, meeting_date::text AS meeting_date, fund, stated_total,
           parsed_total, line_count, check_count, reconciled,
           locator_file_path, locator_file_sha256
    FROM facts.voucher_set
    WHERE meeting_date >= %s AND reconciled IS TRUE
    ORDER BY meeting_date, fund
"""

LINES_SQL = """
    SELECT line_seq, vendor_raw, check_date::text AS check_date, check_number,
           check_amount, invoice_amount, description, is_pcard,
           is_payroll_warrant, is_credit, locator_page, locator_char_offset,
           locator_quote
    FROM facts.voucher_line
    WHERE set_id = %s
    ORDER BY line_seq
"""


def money(value: Decimal | None) -> str:
    """Format an amount for a sample table.

    Args:
        value: Amount, or None.

    Returns:
        A comma-grouped string, or an em dash.
    """
    return "—" if value is None else f"{value:,.2f}"


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
    add(
        "Open the source file at the page in each row and confirm the vendor, "
        "date, check number and both amounts match. Mark anything that does not."
    )
    add("")
    add(
        "**If you find zero errors in 300 lines**, the error rate for this set is "
        "below roughly 1% at 95% confidence (the rule of three: 3/300). Fewer than "
        "300 lines gives a correspondingly weaker bound, and this says nothing "
        "about any set not sampled."
    )
    add("")
    add("| # | Page | Vendor | Check date | Check no. | Check amt | Invoice amt | Description | Flags |")
    add("|---:|---:|---|---|---|---:|---:|---|---|")
    for row in lines:
        flags = []
        if row["is_pcard"]:
            flags.append("P-card")
        if row["is_payroll_warrant"]:
            flags.append("payroll")
        if row["is_credit"]:
            flags.append("credit")
        description = (row["description"] or "").replace("|", "\\|")[:70]
        vendor = row["vendor_raw"].replace("|", "\\|")
        add(
            f"| {row['line_seq']} | {row['locator_page'] or '—'} | {vendor} | "
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
        add(
            f"- `#{row['line_seq']}` p{row['locator_page']} @{row['locator_char_offset']}: "
            f"`{(row['locator_quote'] or '')[:200]}`"
        )
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
        drawn = min(args.size, len(lines))
        sample = sorted(rng.sample(lines, drawn), key=lambda r: r["line_seq"])
        name = f"{set_row['meeting_date']}-{set_row['fund']}.md"
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render(set_row, sample, args.seed, drawn))
        written.append((path, drawn, len(lines)))

    for path, drawn, total in written:
        print(f"wrote {path}  ({drawn} of {total} lines)")
    print(f"\n{len(written)} sample file(s), seed {args.seed}")
    print("Accept-on-zero: 0 errors in 300 sampled lines puts that set's error rate below about 1% at 95% confidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
