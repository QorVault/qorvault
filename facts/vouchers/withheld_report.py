"""List every payee the classifier withholds, as a worksheet for the allowlist.

This is the operator's triage sheet for ``fixtures/payee_allowlist.txt``:
it names each withheld payee, says how much of the corpus that payee
accounts for, when they first and last appear, why they were withheld, and
whether the name has the shape the marker rule is worst at -- a bare
acronym or an all-caps name.

**The output contains individuals' names and must not be committed.** That
is the whole reason the classifier exists. The file is written outside the
code tree's published artifacts and the session debrief carries the command
to keep git from ever taking it. Read it, decide, put the organizations you
choose into the allowlist, and the allowlist -- which by then contains only
names you have decided may be published -- is what goes into git.

No LLM decides anything here. The reason column is
``vendors.classify_payee``'s own verdict, so this sheet and the export
writers cannot disagree about who is withheld.
"""

from __future__ import annotations

import argparse
import csv
import os

import db
from vendors import ACRONYM_RX, classify_payee, display_name

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports",
    "withheld-payees-2026-09-16.csv",
)

# Cycle is the meeting the listing was presented at, taken from the set the
# line is printed in. Not the check date: a warrant cut in one month and
# presented in the next belongs to the cycle that presented it, which is
# the same rule the deduped views use.
PAYEES_SQL = """
    SELECT v.display_name,
           count(*)                    AS line_count,
           min(s.meeting_date)::text   AS first_cycle,
           max(s.meeting_date)::text   AS last_cycle,
           EXISTS (SELECT 1 FROM facts.voucher_line h
                   WHERE h.vendor_norm = v.vendor_norm
                     AND h.description ~* 'payroll\\s+handwrite') AS payroll_handwrite
    FROM facts.vendor v
    JOIN facts.voucher_line l ON l.vendor_norm = v.vendor_norm
    JOIN facts.voucher_set  s ON s.set_id = l.set_id
    GROUP BY v.display_name, v.vendor_norm
    ORDER BY count(*) DESC, v.display_name
"""

COLUMNS = ("payee", "line_count", "first_cycle", "last_cycle", "reason", "flag")


def name_flag(raw: str) -> str:
    """Flag the two name shapes the marker rule is worst at.

    Args:
        raw: Payee name exactly as printed.

    Returns:
        ``bare_acronym`` for a single all-caps token such as a purchasing
        co-operative's initials, ``all_caps`` for a multi-token name
        printed entirely in capitals, or the empty string.
    """
    text = display_name(raw)
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) == 1 and ACRONYM_RX.match(text):
        return "bare_acronym"
    if any(ch.isalpha() for ch in text) and text == text.upper():
        return "all_caps"
    return ""


def withheld_rows() -> list[dict]:
    """Every withheld payee, already sorted by line count descending.

    Returns:
        Rows ready for the CSV writer.
    """
    out = []
    for row in db.query_dicts(PAYEES_SQL, None):
        raw = row["display_name"]
        publish, reason = classify_payee(raw, bool(row["payroll_handwrite"]))
        if publish:
            continue
        out.append(
            {
                "payee": display_name(raw),
                "line_count": row["line_count"],
                "first_cycle": row["first_cycle"],
                "last_cycle": row["last_cycle"],
                "reason": reason,
                "flag": name_flag(raw),
            }
        )
    return out


def main() -> int:
    """Write the withheld-payee worksheet.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    args = parser.parse_args()

    rows = withheld_rows()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_flag: dict[str, list[dict]] = {}
    for row in rows:
        by_flag.setdefault(row["flag"] or "(none)", []).append(row)
    by_reason: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1

    print(f"wrote {args.out}")
    print(f"{len(rows):,} withheld payees covering {sum(r['line_count'] for r in rows):,} lines")
    for flag, group in sorted(by_flag.items()):
        print(f"  flag {flag:<13} {len(group):>6,} payees  {sum(r['line_count'] for r in group):>8,} lines")
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  reason {reason:<20} {count:>6,} payees")
    print("\nThis file names individuals. Do not commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
