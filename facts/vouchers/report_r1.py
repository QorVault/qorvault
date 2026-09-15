"""Before-and-after tables for the R1 column-assignment change.

Compares a snapshot of ``facts.voucher_set`` taken before the change with
the table as it stands now, and prints the Markdown the report needs: the
per-set table, the reason-code counts, the residual out-of-balance list by
year and fund, and how often the retired regex row parser disagreed with
the geometry that replaced it.

The "before" snapshot is a CSV of ``facts.voucher_set`` taken at the
commit being compared against. To rebuild one from scratch, check that
commit out into a separate worktree and run ``build.py --dry-run``.

Reads only. Writes nothing to the database.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from decimal import Decimal

import db


def load_before(path: str) -> dict[str, dict]:
    """Load the pre-change snapshot.

    Args:
        path: CSV written from ``facts.voucher_set``.

    Returns:
        Rows keyed by set id.
    """
    with open(path, encoding="utf-8") as handle:
        return {row["set_id"]: row for row in csv.DictReader(handle)}


def load_after() -> dict[str, dict]:
    """Load the current state of every voucher set.

    Returns:
        Rows keyed by set id.
    """
    rows = db.query_dicts(
        """
        SELECT s.set_id, s.meeting_date::text AS meeting_date, s.fund, s.format_era, s.source,
               s.stated_total, s.parsed_total, s.line_count, s.check_count,
               s.reconciled, s.delta, s.reason_code,
               COALESCE(u.unread, 0) AS unread
        FROM facts.voucher_set s
        LEFT JOIN (
            SELECT set_id, count(*) AS unread
            FROM facts.voucher_line WHERE reason_code IS NOT NULL GROUP BY set_id
        ) u ON u.set_id = s.set_id
        ORDER BY s.meeting_date, s.fund
        """,
        None,
    )
    return {row["set_id"]: row for row in rows}


def _decimal(value) -> Decimal | None:
    """Coerce a CSV cell to a Decimal.

    Args:
        value: Cell contents.

    Returns:
        The decimal, or None when the cell was empty.
    """
    if value in (None, "", "None"):
        return None
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _bool(value) -> bool | None:
    """Coerce a CSV cell to a tri-state boolean.

    Args:
        value: Cell contents.

    Returns:
        True, False or None.
    """
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "t", "1"}


def ties(row: dict) -> bool:
    """Whether a set reconciles to the cent.

    Args:
        row: A set row from either side.

    Returns:
        True only when the document states a total and the lines match it.
    """
    return _bool(row["reconciled"]) is True


def _tie_cell(row: dict | None) -> str:
    """Render whether a set reconciles, keeping the three states distinct.

    Args:
        row: A set row, or None when the set is absent from that side.

    Returns:
        ``Y``, ``N``, or an em dash when the document states no total to
        reconcile against and there is therefore nothing to tie.
    """
    if row is None:
        return "(absent)"
    state = _bool(row["reconciled"])
    return {True: "Y", False: "N", None: "n/a"}[state]


def per_set_table(before: dict, after: dict) -> list[str]:
    """Build the per-set before-and-after table.

    Args:
        before: Snapshot rows keyed by set id.
        after: Current rows keyed by set id.

    Returns:
        Markdown lines.
    """
    out = [
        "| set | era | printed total | parsed before | parsed after | ties before | ties after "
        "| reason before | reason after | unread rows after |",
        "|---|---|---:|---:|---:|:-:|:-:|---|---|---:|",
    ]
    for set_id in sorted(set(before) | set(after)):
        b = before.get(set_id)
        a = after.get(set_id)
        printed = _decimal((a or b)["stated_total"])
        out.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                set_id,
                (a or b)["format_era"] or "-",
                "—" if printed is None else f"{printed:,}",
                "(absent)" if b is None else f"{_decimal(b['parsed_total']):,}",
                "(absent)" if a is None else f"{_decimal(a['parsed_total']):,}",
                _tie_cell(b),
                _tie_cell(a),
                "—" if b is None else (b["reason_code"] or "—"),
                "—" if a is None else (a["reason_code"] or "—"),
                "—" if a is None else a["unread"],
            )
        )
    return out


def regressions(before: dict, after: dict) -> list[str]:
    """List sets that tied before the change and do not tie after it.

    Args:
        before: Snapshot rows.
        after: Current rows.

    Returns:
        Human-readable lines, empty when there are none.
    """
    out = []
    for set_id, b in sorted(before.items()):
        if not ties(b):
            continue
        a = after.get(set_id)
        if a is None:
            out.append(f"{set_id}: reconciled before, absent from the table now")
        elif not ties(a):
            out.append(
                f"{set_id}: reconciled before at {b['parsed_total']}, now {a['parsed_total']} ({a['reason_code']})"
            )
    return out


def main() -> int:
    """Print the report tables.

    Returns:
        Process exit code: 1 when any set stopped reconciling.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="CSV snapshot of facts.voucher_set before the change")
    parser.add_argument("--table", action="store_true", help="print the full per-set table")
    args = parser.parse_args()

    before, after = load_before(args.before), load_after()

    tied_before = sum(1 for r in before.values() if ties(r))
    tied_after = sum(1 for r in after.values() if ties(r))
    lines_before = sum(int(r["line_count"]) for r in before.values())
    lines_after = sum(r["line_count"] for r in after.values())
    unread_after = sum(r["unread"] for r in after.values())

    print("## Totals\n")
    print(f"- sets: {len(before)} before, {len(after)} after")
    print(f"- reconciling to the cent: **{tied_before} before, {tied_after} after**")
    print(f"- voucher lines: {lines_before:,} before, {lines_after:,} after")
    print(f"- rows held with a reason code: **{unread_after:,}** of {lines_after:,} after")

    print("\n## Reason codes\n")
    before_codes = Counter(r["reason_code"] or "(reconciled)" for r in before.values())
    after_codes = Counter(r["reason_code"] or "(reconciled)" for r in after.values())
    print("| reason code | sets before | sets after |")
    print("|---|---:|---:|")
    for code in sorted(set(before_codes) | set(after_codes)):
        print(f"| {code} | {before_codes.get(code, 0)} | {after_codes.get(code, 0)} |")

    print("\n## Sets that stopped reconciling\n")
    broken = regressions(before, after)
    print("\n".join(f"- {line}" for line in broken) if broken else "None.")

    print("\n## Residual: sets with a printed total that do not tie\n")
    residual = [
        r
        for r in after.values()
        if _bool(r["reconciled"]) is False and r["reason_code"] != "REGEX_MISS" and r["stated_total"] is not None
    ]
    by_year: dict[str, Counter] = defaultdict(Counter)
    for row in residual:
        by_year[row["meeting_date"][:4]][row["fund"]] += 1
    print(f"{len(residual)} sets.\n")
    print("| year | " + " | ".join(sorted({r["fund"] for r in residual})) + " | total |")
    funds = sorted({r["fund"] for r in residual})
    print("|---" * (len(funds) + 2) + "|")
    for year in sorted(by_year):
        counts = by_year[year]
        print(f"| {year} | " + " | ".join(str(counts.get(f, 0)) for f in funds) + f" | {sum(counts.values())} |")

    print("\n| set | printed total | parsed | delta | unread rows |")
    print("|---|---:|---:|---:|---:|")
    for row in sorted(residual, key=lambda r: r["set_id"]):
        print(
            f"| `{row['set_id']}` | {row['stated_total']:,} | {row['parsed_total']:,} "
            f"| {row['delta']:,} | {row['unread']} |"
        )

    print("\n## Retired regex cross-check, by set\n")
    checks = db.query_dicts(
        """
        SELECT meeting_date::text AS meeting_date, fund, regex_agree, regex_disagree, regex_miss
        FROM facts.voucher_parse_log
        WHERE regex_disagree > 0 OR regex_miss > 0
        ORDER BY regex_disagree DESC, regex_miss DESC
        """,
        None,
    )
    totals = db.query_dicts(
        "SELECT sum(regex_agree) a, sum(regex_disagree) d, sum(regex_miss) m FROM facts.voucher_parse_log", None
    )[0]
    print(f"Across the corpus: {totals['a']:,} agree, **{totals['d']:,} disagree**, {totals['m']:,} regex miss.\n")
    print("| meeting | fund | agree | disagree | regex miss |")
    print("|---|---|---:|---:|---:|")
    for row in checks[:60]:
        print(
            f"| {row['meeting_date']} | {row['fund']} | {row['regex_agree']:,} "
            f"| {row['regex_disagree']:,} | {row['regex_miss']:,} |"
        )
    if len(checks) > 60:
        print(f"\n{len(checks) - 60} further artifacts with a disagreement or a miss are not listed here.")

    if args.table:
        print("\n## Every set, before and after\n")
        print("\n".join(per_set_table(before, after)))

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
