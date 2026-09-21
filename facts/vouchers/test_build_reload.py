"""Regression tests for the reload path of ``build.py``.

The bug these pin: ``--only-date`` narrowed what was *parsed*, but
``--reload`` deleted every fact table unconditionally, so the obvious
"reload one meeting" command destroyed every other cycle's facts and then
inserted one date's worth. Confirmed as a data-loss bug on 2026-09-21.

These tests run without a database. The write step takes a *store*; here it
is an in-memory table model that records every delete it is asked to make,
so a test can assert on the resulting rows AND on which deletes happened.
The SQL that ``PostgresStore`` emits is exercised by
``test_build_reload_db.py`` against a real database, inside a rolled-back
transaction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal

import pytest
from build import (
    DATE_SCOPED_TABLES,
    FACT_TABLES,
    LINE_COLUMNS,
    LOG_COLUMNS,
    RECON_COLUMNS,
    SET_COLUMNS,
    VENDOR_COLUMNS,
    FactsPayload,
    ReloadScopeError,
    build_vendors,
    single_date_caveat,
    write_facts,
)

# Synthetic dates far outside the corpus so that the same fixtures can be
# reused verbatim by the database-backed test without colliding with real
# sets.
EARLY = "2031-01-14"
LATE = "2031-02-11"


# ------------------------------------------------------------ the store --
class MemoryStore:
    """The smallest table model that can stand in for Postgres here.

    Rows are plain dicts. Every delete is appended to ``deletes`` so a test
    can prove not only what survived but which statements were issued.
    """

    def __init__(self) -> None:
        """Start with every fact table empty and no deletes recorded."""
        self.tables: dict[str, list[dict]] = {name: [] for name in FACT_TABLES}
        self.deletes: list[tuple] = []

    # -- writes ---------------------------------------------------------
    def delete_all(self, table: str) -> None:
        """Delete every row of one fact table."""
        self.deletes.append(("all", table))
        self.tables[table] = []

    def delete_where_date(self, table: str, meeting_date: str) -> None:
        """Delete one meeting date's rows from a table that carries ``meeting_date``."""
        self.deletes.append(("date", table, meeting_date))
        self.tables[table] = [r for r in self.tables[table] if r["meeting_date"] != meeting_date]

    def delete_where_set_in(self, table: str, set_ids: list[str]) -> None:
        """Delete the rows of the given sets from a table keyed by ``set_id``."""
        self.deletes.append(("sets", table, tuple(set_ids)))
        wanted = set(set_ids)
        self.tables[table] = [r for r in self.tables[table] if r["set_id"] not in wanted]

    def insert(self, table: str, columns: tuple[str, ...], rows: list[dict]) -> None:
        """Insert rows, binding every value."""
        self.tables[table].extend({c: r.get(c) for c in columns} for r in rows)

    # -- reads ----------------------------------------------------------
    def set_ids_for_date(self, meeting_date: str) -> list[str]:
        """The set ids stored for one meeting date."""
        return [r["set_id"] for r in self.tables["voucher_set"] if r["meeting_date"] == meeting_date]

    def lines_for_vendors(self) -> tuple[list[dict], dict[str, date]]:
        """Every stored line, oldest set first, plus each set's date."""
        set_dates = {r["set_id"]: date.fromisoformat(r["meeting_date"]) for r in self.tables["voucher_set"]}
        lines = sorted(
            self.tables["voucher_line"],
            key=lambda r: (set_dates[r["set_id"]], r["set_id"], r["line_seq"]),
        )
        return lines, set_dates

    def vendor_tags(self) -> dict[str, tuple[str | None, str | None]]:
        """Operator and LLM tags currently on vendors, by ``vendor_norm``."""
        return {
            r["vendor_norm"]: (r.get("category"), r.get("tag_source"))
            for r in self.tables["vendor"]
            if r.get("category") is not None or r.get("tag_source") is not None
        }


# --------------------------------------------------------------- fixtures --
def _sha(*parts: str) -> str:
    """A distinct, stable sha256 per synthetic file: parse_log has UNIQUE (file_sha256)."""
    return hashlib.sha256(("test:" + ":".join(parts)).encode()).hexdigest()


def _line(set_id: str, seq: int, vendor: str, amount: str, **overrides) -> dict:
    row = {c: None for c in LINE_COLUMNS}
    row.update(
        set_id=set_id,
        line_seq=seq,
        vendor_raw=vendor,
        vendor_norm=vendor.lower(),
        check_number=f"{600000 + seq}",
        check_amount=Decimal(amount),
        invoice_amount=Decimal(amount),
        is_pcard=False,
        is_payroll_warrant=False,
        is_person_shaped=False,
        is_credit=False,
        amount_paren=False,
        reason_code=None,
        source="corpus_pdf",
        locator_file_path=f"/test/{set_id}.pdf",
        locator_file_sha256=_sha(set_id),
        locator_page=1,
        locator_char_offset=seq * 10,
        locator_quote=f"{vendor} {amount}",
    )
    row.update(overrides)
    return row


def _set(meeting_date: str, fund: str, total: str, **overrides) -> dict:
    row = {c: None for c in SET_COLUMNS}
    row.update(
        set_id=f"{meeting_date}:{fund}",
        meeting_date=meeting_date,
        fund=fund,
        doc_class="detail_listing",
        format_era="C",
        stated_total=Decimal(total),
        parsed_total=Decimal(total),
        line_count=2,
        check_count=2,
        reconciled=True,
        delta=Decimal("0"),
        source="corpus_pdf",
        locator_file_path=f"/test/{meeting_date}-{fund}.pdf",
        locator_file_sha256=_sha(f"{meeting_date}:{fund}"),
    )
    row.update(overrides)
    return row


def _recon(set_id: str, **overrides) -> dict:
    row = {c: None for c in RECON_COLUMNS}
    row.update(set_id=set_id, basis="recap_fund_total", match=True, delta=Decimal("0"))
    row.update(overrides)
    return row


def _log(meeting_date: str, fund: str, **overrides) -> dict:
    row = {c: None for c in LOG_COLUMNS}
    row.update(
        file_sha256=_sha(f"{meeting_date}:{fund}"),
        file_path=f"/test/{meeting_date}-{fund}.pdf",
        meeting_date=meeting_date,
        fund=fund,
        doc_class="detail_listing",
        status="parsed",
        lines_found=2,
        checks_found=2,
        # NOT NULL DEFAULT 0 in the schema: an explicit NULL in an INSERT
        # overrides the default, so the fixture must set them as the build does.
        unread_lines=0,
        regex_agree=0,
        regex_disagree=0,
        regex_miss=0,
        source="corpus_pdf",
    )
    row.update(overrides)
    return row


def _date_payload(meeting_date: str, amounts: tuple[str, str], vendors: tuple[str, str]) -> FactsPayload:
    """One GF set with two lines for ``meeting_date``."""
    set_id = f"{meeting_date}:GF"
    lines = [
        _line(set_id, 1, vendors[0], amounts[0]),
        _line(set_id, 2, vendors[1], amounts[1]),
    ]
    total = str(Decimal(amounts[0]) + Decimal(amounts[1]))
    set_rows = [_set(meeting_date, "GF", total)]
    set_dates = {set_id: date.fromisoformat(meeting_date)}
    return FactsPayload(
        set_rows=set_rows,
        lines=lines,
        vendors=build_vendors(lines, set_dates),
        recon_rows=[_recon(set_id)],
        logs=[_log(meeting_date, "GF")],
    )


def _combined(*payloads: FactsPayload) -> FactsPayload:
    """What a full build of several dates would produce, vendors aggregated across all of them."""
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


def _seeded_store() -> tuple[MemoryStore, FactsPayload, FactsPayload]:
    """A store holding two dates, loaded the way a full reload would load them."""
    early = _date_payload(EARLY, ("100.00", "250.50"), ("Acme Widgets Inc", "Shared Vendor LLC"))
    late = _date_payload(LATE, ("75.25", "400.00"), ("Shared Vendor LLC", "Late Only Corp"))
    store = MemoryStore()
    write_facts(store, _combined(early, late), reload=True, only_date=None)
    # An operator tag on a vendor that both dates pay. build_vendors never
    # produces this column; it exists only because someone set it.
    for row in store.tables["vendor"]:
        if row["vendor_norm"] == "shared vendor llc":
            row["category"] = "utilities"
            row["tag_source"] = "manual"
    store.deletes.clear()
    return store, early, late


def _rows_for_date(store: MemoryStore, table: str, meeting_date: str) -> list[dict]:
    if table in ("voucher_set", "voucher_parse_log"):
        return [r for r in store.tables[table] if r["meeting_date"] == meeting_date]
    ids = set(store.set_ids_for_date(meeting_date))
    return [r for r in store.tables[table] if r["set_id"] in ids]


def _content_hash(rows: list[dict]) -> str:
    canonical = json.dumps(
        sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str)), sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# ------------------------------------------------------------------ tests --
class TestSingleDateReload:
    """``--reload --only-date D`` replaces D and touches nothing else."""

    def test_other_date_rows_are_byte_identical(self):
        """Every date-scoped table keeps the untouched date's rows, by count and by content."""
        store, early, _ = _seeded_store()
        before = {
            t: (len(_rows_for_date(store, t, EARLY)), _content_hash(_rows_for_date(store, t, EARLY)))
            for t in DATE_SCOPED_TABLES
        }
        assert all(count > 0 for count, _ in before.values()), "fixture must seed every date-scoped table"

        replacement = _date_payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))
        write_facts(store, replacement, reload=True, only_date=LATE)

        after = {
            t: (len(_rows_for_date(store, t, EARLY)), _content_hash(_rows_for_date(store, t, EARLY)))
            for t in DATE_SCOPED_TABLES
        }
        assert after == before

    def test_target_date_rows_are_replaced(self):
        """The reloaded date holds exactly the new payload, with none of its old rows left behind."""
        store, _, _ = _seeded_store()
        replacement = _date_payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))

        write_facts(store, replacement, reload=True, only_date=LATE)

        assert _content_hash(_rows_for_date(store, "voucher_set", LATE)) == _content_hash(
            [{c: r.get(c) for c in SET_COLUMNS} for r in replacement.set_rows]
        )
        assert _content_hash(_rows_for_date(store, "voucher_line", LATE)) == _content_hash(
            [{c: r.get(c) for c in LINE_COLUMNS} for r in replacement.lines]
        )
        assert _content_hash(_rows_for_date(store, "voucher_reconciliation", LATE)) == _content_hash(
            [{c: r.get(c) for c in RECON_COLUMNS} for r in replacement.recon_rows]
        )
        assert _content_hash(_rows_for_date(store, "voucher_parse_log", LATE)) == _content_hash(
            [{c: r.get(c) for c in LOG_COLUMNS} for r in replacement.logs]
        )

    def test_vendor_table_equals_full_aggregate_of_all_dates(self):
        """Vendors are cross-date aggregates, so they are recomputed from every line now stored."""
        store, early, _ = _seeded_store()
        replacement = _date_payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))

        write_facts(store, replacement, reload=True, only_date=LATE)

        expected = build_vendors(*MemoryStore.lines_for_vendors(store))
        got = {r["vendor_norm"]: {c: r.get(c) for c in VENDOR_COLUMNS} for r in store.tables["vendor"]}
        assert got == {r["vendor_norm"]: {c: r.get(c) for c in VENDOR_COLUMNS} for r in expected}
        # The early date's exclusive vendor survives; the late date's old exclusive vendor is gone.
        assert "acme widgets inc" in got
        assert "late only corp" not in got
        assert "brand new vendor" in got
        # And the shared vendor now spans both dates with both dates' money.
        shared = got["shared vendor llc"]
        assert (shared["first_seen"], shared["last_seen"]) == (date.fromisoformat(EARLY), date.fromisoformat(LATE))
        assert shared["total_invoice"] == Decimal("250.50") + Decimal("999.99")

    def test_vendor_tags_are_carried_across_the_recompute(self):
        """An operator's category tag on a vendor is not destroyed by reloading one date."""
        store, _, _ = _seeded_store()
        replacement = _date_payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))

        write_facts(store, replacement, reload=True, only_date=LATE)

        tagged = {r["vendor_norm"]: (r.get("category"), r.get("tag_source")) for r in store.tables["vendor"]}
        assert tagged["shared vendor llc"] == ("utilities", "manual")
        assert tagged["acme widgets inc"] == (None, None)

    def test_no_unscoped_delete_except_the_vendor_recompute(self):
        """The only whole-table delete on this path is the one the vendor recompute needs."""
        store, _, _ = _seeded_store()
        replacement = _date_payload(LATE, ("999.99", "1.01"), ("Shared Vendor LLC", "Brand New Vendor"))

        write_facts(store, replacement, reload=True, only_date=LATE)

        unscoped = [d for d in store.deletes if d[0] == "all"]
        assert unscoped == [("all", "vendor")]
        scoped_tables = {d[1] for d in store.deletes if d[0] in ("date", "sets")}
        assert scoped_tables == set(DATE_SCOPED_TABLES)

    def test_refuses_a_payload_row_outside_the_date(self):
        """A row for another date in a single-date reload is a scoping error, raised before any delete."""
        store, early, _ = _seeded_store()
        stray = _date_payload(LATE, ("1.00", "2.00"), ("A Vendor", "B Vendor"))
        stray.set_rows.append(early.set_rows[0])

        with pytest.raises(ReloadScopeError):
            write_facts(store, stray, reload=True, only_date=LATE)
        assert store.deletes == []

    def test_refuses_a_fact_table_it_does_not_know_how_to_scope(self, monkeypatch):
        """A sixth fact table with no per-date strategy makes the combination refuse, before any delete."""
        store, _, _ = _seeded_store()
        monkeypatch.setattr("build.FACT_TABLES", (*FACT_TABLES, "voucher_future_table"))
        store.tables["voucher_future_table"] = [{"set_id": f"{EARLY}:GF", "meeting_date": EARLY}]
        replacement = _date_payload(LATE, ("1.00", "2.00"), ("A Vendor", "B Vendor"))

        with pytest.raises(ReloadScopeError, match="voucher_future_table"):
            write_facts(store, replacement, reload=True, only_date=LATE)
        assert store.deletes == []


class TestFullReloadUnchanged:
    """``--reload`` without a date still wipes and rebuilds all five tables, in the same order."""

    def test_full_reload_deletes_every_fact_table_unscoped_in_the_original_order(self):
        """Full reload issues exactly the five unconditional deletes, in the order build.py has always used."""
        store, early, late = _seeded_store()

        write_facts(store, _combined(early, late), reload=True, only_date=None)

        assert store.deletes == [
            ("all", "voucher_reconciliation"),
            ("all", "voucher_line"),
            ("all", "voucher_set"),
            ("all", "vendor"),
            ("all", "voucher_parse_log"),
        ]

    def test_full_reload_uses_the_payload_vendors_as_given(self):
        """Full reload writes the vendors the build computed; it does not re-read lines from the store."""
        store, early, late = _seeded_store()
        payload = _combined(early, late)

        write_facts(store, payload, reload=True, only_date=None)

        got = [{c: r.get(c) for c in VENDOR_COLUMNS} for r in store.tables["vendor"]]
        assert got == [{c: r.get(c) for c in VENDOR_COLUMNS} for r in payload.vendors]

    def test_without_reload_nothing_is_deleted(self):
        """Plain insert mode issues no delete at all, as before."""
        store = MemoryStore()
        early = _date_payload(EARLY, ("1.00", "2.00"), ("A Vendor", "B Vendor"))

        write_facts(store, early, reload=False, only_date=None)

        assert store.deletes == []
        assert len(store.tables["voucher_set"]) == 1


class TestTableRegistry:
    """The registry is what keeps a future table from reintroducing the bug."""

    def test_every_fact_table_is_either_date_scoped_or_recomputed(self):
        """No fact table is left in an undefined third state."""
        assert set(FACT_TABLES) == set(DATE_SCOPED_TABLES) | {"vendor"}


class TestSingleDateCaveat:
    """A single-date run cannot see earlier cycles, so its cumulative-overlap note cannot fire. Say so."""

    def test_caveat_names_the_date_and_the_gap(self):
        """The caveat text is specific enough to act on: which date, which fact, and why."""
        text = single_date_caveat(LATE)
        assert LATE in text
        assert "cumulative" in text
        assert "earlier" in text
