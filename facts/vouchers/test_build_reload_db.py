"""The single-date reload regression, run against the real database.

``test_build_reload.py`` proves the scoping logic against an in-memory
table model. This file proves the SQL that ``PostgresStore`` actually emits,
which the model cannot: that a ``--reload --only-date`` run against Postgres
leaves every row it was not asked to touch byte-identical.

**It never modifies the live schema.** Everything happens inside one
transaction on a raw connection that is rolled back in ``finally`` --
``db.connect()`` is deliberately not used because it commits on success.
The rows it seeds are synthetic (meeting dates in 2031, vendor names
prefixed ``zz test``), so they cannot collide with a real set, and the
assertions on real data are count-and-hash comparisons taken before and
after.

Like ``test_no_leaks.py`` it SKIPS without a write credential, and a skip
here is a stop condition before merge, not a pass: the operator runs it with
``POSTGRES_PASSWORD`` from ``ksd-main/.env``. The credential-free podman
transport is read-only and cannot run it.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal

import db
import psycopg2
import pytest
from build import (
    DATE_SCOPED_TABLES,
    LINE_COLUMNS,
    RECON_COLUMNS,
    SET_COLUMNS,
    FactsPayload,
    PostgresStore,
    build_vendors,
    write_facts,
)
from test_build_reload import EARLY, LATE, _line, _log, _recon, _set

SYNTHETIC_PREFIX = "2031-"
VENDOR_PREFIX = "zz test "


def _connection():
    """A raw psycopg2 connection, or a loud skip."""
    if db.use_psql_transport():
        pytest.skip("VOUCHERS_DB_TRANSPORT=podman is read-only; this test needs the write credential")
    try:
        return psycopg2.connect(**db.connection_kwargs())
    except psycopg2.OperationalError as exc:
        pytest.skip(
            "no database connection, so the reload regression did not run against Postgres: "
            f"{type(exc).__name__}: {exc}"
        )


def _payload(meeting_date: str, amounts: tuple[str, str], vendors: tuple[str, str]) -> FactsPayload:
    """One GF set with two lines, shaped to satisfy every CHECK constraint in schema.sql."""
    set_id = f"{meeting_date}:GF"
    real = {"source": "corpus_pdf"}
    lines = [
        _line(set_id, 1, VENDOR_PREFIX + vendors[0], amounts[0], **real),
        _line(set_id, 2, VENDOR_PREFIX + vendors[1], amounts[1], **real),
    ]
    total = str(Decimal(amounts[0]) + Decimal(amounts[1]))
    set_rows = [_set(meeting_date, "GF", total, **real)]
    return FactsPayload(
        set_rows=set_rows,
        lines=lines,
        vendors=build_vendors(lines, {set_id: date.fromisoformat(meeting_date)}),
        recon_rows=[_recon(set_id, basis="recap_fund_total", reason="MATCH")],
        logs=[_log(meeting_date, "GF", **real)],
    )


def _combined(*payloads: FactsPayload) -> FactsPayload:
    lines = [ln for p in payloads for ln in p.lines]
    set_rows = [s for p in payloads for s in p.set_rows]
    set_dates = {s["set_id"]: date.fromisoformat(s["meeting_date"]) for s in set_rows}
    return FactsPayload(
        set_rows=set_rows,
        lines=lines,
        vendors=build_vendors(lines, set_dates),
        recon_rows=[r for p in payloads for r in p.recon_rows],
        logs=[lg for p in payloads for lg in p.logs],
    )


# The real-data snapshots exclude synthetic rows by prefix and are taken as
# (count, md5 of every row rendered as text, in key order). ``vendor`` is
# compared on its order-independent columns only: the recompute meets lines
# in (meeting_date, set_id, line_seq) order, which can differ from a full
# build's artifact order where one date has two listings for the same fund,
# and that would move the ``observed_locator_*`` columns without changing
# any fact.
_REAL_SNAPSHOTS = {
    "voucher_set": (
        "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY set_id), '')) "
        "FROM facts.voucher_set t WHERE set_id NOT LIKE %s"
    ),
    "voucher_line": (
        "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY set_id, line_seq), '')) "
        "FROM facts.voucher_line t WHERE set_id NOT LIKE %s"
    ),
    "voucher_reconciliation": (
        "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY set_id, basis), '')) "
        "FROM facts.voucher_reconciliation t WHERE set_id NOT LIKE %s"
    ),
    "voucher_parse_log": (
        "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY parse_id), '')) "
        "FROM facts.voucher_parse_log t WHERE coalesce(meeting_date::text, '') NOT LIKE %s"
    ),
}
_REAL_VENDORS = (
    "SELECT vendor_norm, display_name, aliases, is_person_shaped, first_seen, last_seen, line_count, "
    "total_invoice, category, tag_source FROM facts.vendor WHERE vendor_norm NOT LIKE %s ORDER BY vendor_norm"
)


def _snapshot_real(cur) -> dict:
    out = {}
    for table, statement in _REAL_SNAPSHOTS.items():
        cur.execute(statement, (SYNTHETIC_PREFIX + "%",))
        out[table] = cur.fetchone()
    cur.execute(_REAL_VENDORS, (VENDOR_PREFIX + "%",))
    out["vendor"] = {row[0]: row[1:] for row in cur.fetchall()}
    return out


def _snapshot_date(cur, meeting_date: str) -> dict:
    """Every row a meeting date owns, rendered as text, per date-scoped table."""
    out = {}
    cur.execute("SELECT t::text FROM facts.voucher_set t WHERE meeting_date = %s ORDER BY set_id", (meeting_date,))
    out["voucher_set"] = [r[0] for r in cur.fetchall()]
    cur.execute(
        "SELECT t::text FROM facts.voucher_line t "
        "WHERE set_id IN (SELECT set_id FROM facts.voucher_set WHERE meeting_date = %s) ORDER BY set_id, line_seq",
        (meeting_date,),
    )
    out["voucher_line"] = [r[0] for r in cur.fetchall()]
    cur.execute(
        "SELECT t::text FROM facts.voucher_reconciliation t "
        "WHERE set_id IN (SELECT set_id FROM facts.voucher_set WHERE meeting_date = %s) ORDER BY set_id, basis",
        (meeting_date,),
    )
    out["voucher_reconciliation"] = [r[0] for r in cur.fetchall()]
    cur.execute(
        "SELECT t::text FROM facts.voucher_parse_log t WHERE meeting_date = %s ORDER BY parse_id", (meeting_date,)
    )
    out["voucher_parse_log"] = [r[0] for r in cur.fetchall()]
    assert set(out) == set(DATE_SCOPED_TABLES)
    return out


def _fp(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class TestSingleDateReloadAgainstPostgres:
    """The SQL behind ``--reload --only-date`` confines itself to that date."""

    def test_reloading_one_date_leaves_every_other_row_byte_identical(self):
        """Real rows and the other synthetic date survive, the target date is replaced, and vendor tags are kept."""
        conn = _connection()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                store = PostgresStore(cur)
                real_before = _snapshot_real(cur)

                early = _payload(EARLY, ("100.00", "250.50"), ("Acme Widgets Inc", "Shared Vendor LLC"))
                late = _payload(LATE, ("75.25", "400.00"), ("Shared Vendor LLC", "Late Only Corp"))
                # Seed with a plain insert: no delete of any kind touches the live tables here.
                write_facts(store, _combined(early, late), reload=False, only_date=None)
                cur.execute(
                    "UPDATE facts.vendor SET category = %s, tag_source = %s WHERE vendor_norm = %s",
                    ("utilities", "manual", VENDOR_PREFIX + "shared vendor llc"),
                )
                assert cur.rowcount == 1
                early_before = _snapshot_date(cur, EARLY)
                assert all(early_before.values()), "fixture must seed every date-scoped table"

                replacement = _payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))
                write_facts(store, replacement, reload=True, only_date=LATE)

                # 1. The other synthetic date: byte-identical, every table.
                assert _snapshot_date(cur, EARLY) == early_before

                # 2. Every real row: byte-identical counts and hashes, every date-scoped table.
                real_after = _snapshot_real(cur)
                for table in DATE_SCOPED_TABLES:
                    assert real_after[table] == real_before[table], f"real rows changed in facts.{table}"

                # 3. Real vendors: same facts, same tags. Differences are reported by fingerprint, never by name.
                changed = sorted(
                    n for n in real_before["vendor"] if real_after["vendor"].get(n) != real_before["vendor"][n]
                )
                missing = sorted(set(real_before["vendor"]) - set(real_after["vendor"]))
                added = sorted(set(real_after["vendor"]) - set(real_before["vendor"]))
                assert not (changed or missing or added), (
                    f"real vendor rows differ after the recompute: changed={len(changed)} missing={len(missing)} "
                    f"added={len(added)}; first fingerprints {[_fp(n) for n in (changed + missing + added)[:5]]}"
                )

                # 4. The target date holds exactly the replacement.
                cur.execute("SELECT set_id, stated_total FROM facts.voucher_set WHERE meeting_date = %s", (LATE,))
                assert cur.fetchall() == [(f"{LATE}:GF", Decimal("1001.00"))]
                cur.execute(
                    "SELECT vendor_norm, invoice_amount FROM facts.voucher_line WHERE set_id = %s ORDER BY line_seq",
                    (f"{LATE}:GF",),
                )
                assert cur.fetchall() == [
                    (VENDOR_PREFIX + "shared vendor llc", Decimal("999.99")),
                    (VENDOR_PREFIX + "brand new vendor", Decimal("1.01")),
                ]
                cur.execute("SELECT count(*) FROM facts.voucher_reconciliation WHERE set_id = %s", (f"{LATE}:GF",))
                assert cur.fetchone() == (len(replacement.recon_rows),)
                cur.execute("SELECT count(*) FROM facts.voucher_parse_log WHERE meeting_date = %s", (LATE,))
                assert cur.fetchone() == (len(replacement.logs),)

                # 5. Synthetic vendors: aggregated across both dates, old late-only vendor gone, tag carried.
                cur.execute(
                    "SELECT vendor_norm, first_seen, last_seen, total_invoice, category, tag_source "
                    "FROM facts.vendor WHERE vendor_norm LIKE %s ORDER BY vendor_norm",
                    (VENDOR_PREFIX + "%",),
                )
                got = {r[0]: r[1:] for r in cur.fetchall()}
                assert set(got) == {
                    VENDOR_PREFIX + n for n in ("acme widgets inc", "shared vendor llc", "brand new vendor")
                }
                assert got[VENDOR_PREFIX + "shared vendor llc"] == (
                    date.fromisoformat(EARLY),
                    date.fromisoformat(LATE),
                    Decimal("250.50") + Decimal("999.99"),
                    "utilities",
                    "manual",
                )
        finally:
            conn.rollback()
            conn.close()

    def test_columns_named_here_exist(self):
        """Guard for the snapshot SQL: every column it names is one the build writes."""
        for column in ("set_id", "meeting_date", "stated_total"):
            assert column in SET_COLUMNS
        for column in ("set_id", "line_seq", "vendor_norm", "invoice_amount"):
            assert column in LINE_COLUMNS
        assert "basis" in RECON_COLUMNS
