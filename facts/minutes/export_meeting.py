"""Export one meeting's motions, votes and executive sessions as markdown.

Every line carries a locator so any statement can be traced to a source
document, page and verbatim quote.

Usage::

    set -a && source ~/workspace/projects/ksd-main/.env && set +a
    export PGPASSWORD=...
    .venv/bin/python export_meeting.py 2026-02-04
    .venv/bin/python export_meeting.py 2026-02-04 --type regular
"""

from __future__ import annotations

import argparse
import sys

import db


def locator_str(doc_id: str | None, page: int | None, offset: int | None) -> str:
    """Render a locator as a compact citation.

    Args:
        doc_id: Source document id.
        page: Page number, or None when the page could not be resolved.
        offset: Character offset into the extracted text.

    Returns:
        Citation string such as ``doc a1b2c3d4 p.3 @12045``.
    """
    if not doc_id:
        return "no locator"
    short = str(doc_id)[:8]
    page_part = f"p.{page}" if page is not None else "p.?"
    off_part = f"@{offset}" if offset is not None else ""
    return f"doc {short} {page_part} {off_part}".strip()


def export(meeting_date: str, meeting_type: str | None) -> str:
    """Build the markdown export for a meeting date.

    Args:
        meeting_date: ISO date of the meeting.
        meeting_type: Optional meeting type filter.

    Returns:
        Markdown text.
    """
    params: list = [meeting_date]
    sql = """
        SELECT meeting_id, meeting_date, meeting_type, format_era, source,
               minutes_document_id::text, approved_at_meeting_id, census_slug
        FROM facts.meeting WHERE meeting_date = %s
    """
    if meeting_type:
        sql += " AND meeting_type = %s"
        params.append(meeting_type)
    sql += " ORDER BY meeting_type"
    meetings = db.query(sql, tuple(params))

    if not meetings:
        return f"# No meeting found on {meeting_date}\n"

    out: list[str] = [f"# Kent School District Board — {meeting_date}", ""]

    for mid, mdate, mtype, era, source, mdoc, approved_at, slug in meetings:
        out.append(f"## {mtype.replace('_', ' ').title()} — `{mid}`")
        out.append("")
        out.append(f"- Format era: **{era or 'n/a'}**  ")
        out.append(f"- Record source: **{source}**  ")
        if mdoc:
            out.append(f"- Minutes document: `{mdoc[:8]}`  ")
        else:
            out.append("- Minutes document: **none in corpus**  ")
        if approved_at:
            out.append(f"- Minutes approved at meeting dated: {approved_at}  ")
        if slug:
            out.append(f"- BoardDocs meeting: `{slug}`  ")
        out.append("")

        attendance = db.query(
            """
            SELECT director_raw, role_raw, status, locator_document_id::text,
                   locator_page, locator_char_offset
            FROM facts.attendance WHERE meeting_id = %s
            ORDER BY director_raw
        """,
            (mid,),
        )
        if attendance:
            out.append("### Attendance")
            out.append("")
            for name, role, status, d, p, o in attendance:
                label = f"{role} {name}" if role else name
                out.append(f"- **{label}** — {status}  \n  <sub>{locator_str(d, p, o)}</sub>")
            out.append("")

        motions = db.query(
            """
            SELECT motion_id, motion_seq, motion_number_raw, motion_text,
                   disposition, vote_format, mover_raw, second_raw,
                   tally_yes, tally_no, tally_abstain, is_consent_agenda,
                   source, locator_document_id::text, locator_page,
                   locator_char_offset, locator_quote
            FROM facts.motion WHERE meeting_id = %s
            ORDER BY source, motion_seq
        """,
            (mid,),
        )
        if motions:
            out.append(f"### Motions ({len(motions)})")
            out.append("")
            for m in motions:
                (
                    motion_id,
                    seq,
                    number,
                    text,
                    disposition,
                    vfmt,
                    mover,
                    second,
                    ty,
                    tn,
                    ta,
                    consent,
                    msource,
                    d,
                    p,
                    o,
                    quote,
                ) = m
                head = number or f"Motion {seq}"
                flag = " _(consent agenda)_" if consent else ""
                out.append(f"#### {head}{flag} — **{disposition.upper()}**")
                out.append("")
                out.append(f"> {text[:600]}")
                out.append("")
                bits = [f"vote format: `{vfmt}`", f"source: `{msource}`"]
                if mover:
                    bits.append(f"moved by **{mover}**")
                if second:
                    bits.append(f"seconded by **{second}**")
                if ty is not None:
                    bits.append(f"tally {ty}-{tn}-{ta}")
                out.append("- " + " · ".join(bits))
                out.append(f'- <sub>{locator_str(d, p, o)} — "{quote[:160]}"</sub>')

                votes = db.query(
                    """
                    SELECT director_raw, vote, locator_document_id::text,
                           locator_page, locator_char_offset
                    FROM facts.vote WHERE motion_id = %s
                    ORDER BY vote, director_raw
                """,
                    (motion_id,),
                )
                if votes:
                    out.append("")
                    for name, vote, vd, vp, vo in votes:
                        out.append(f"  - {name}: **{vote}** " f"<sub>{locator_str(vd, vp, vo)}</sub>")
                out.append("")

        execs = db.query(
            """
            SELECT announcement_seq, announcement_kind, announced_purpose,
                   purpose_category, announced_at, actual_end_time,
                   is_extension, locator_document_id::text, locator_page,
                   locator_char_offset, locator_quote
            FROM facts.executive_session WHERE meeting_id = %s
            ORDER BY announcement_seq
        """,
            (mid,),
        )
        if execs:
            out.append(f"### Executive sessions ({len(execs)})")
            out.append("")
            for seq, kind, purpose, cat, start, end, ext, d, p, o, q in execs:
                out.append(f"- **#{seq}** ({kind}{', extension' if ext else ''})")
                out.append(f"  - Purpose: {purpose or '_not stated_'}")
                out.append(f"  - RCW category: {cat or '_not cited_'}")
                if start:
                    out.append(f"  - Announced at: {start}")
                if end:
                    out.append(f"  - Ended: {end}")
                out.append(f'  - <sub>{locator_str(d, p, o)} — "{q[:160]}"</sub>')
            out.append("")

        if not motions and not execs:
            out.append(
                "_No motions or executive sessions recorded for this "
                "meeting. For work and study sessions this is normal, "
                "not a parse failure._"
            )
            out.append("")

    return "\n".join(out)


def main() -> int:
    """Print the markdown export for a meeting.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description="Export one meeting as markdown.")
    ap.add_argument("meeting_date", help="ISO date, e.g. 2026-02-04")
    ap.add_argument("--type", dest="meeting_type", default=None, help="restrict to one meeting type")
    ap.add_argument("-o", "--output", default=None, help="write to a file")
    args = ap.parse_args()

    md = export(args.meeting_date, args.meeting_type)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
