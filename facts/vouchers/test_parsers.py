"""Tests for the voucher parsers, vendor rules and build arithmetic.

Every case here is drawn from a line that actually occurs in this corpus.
Where a test looks pedantic, the behaviour it pins was wrong at some point
during the build and this is what caught it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
import vendors
from build import PCARD_PREFIX, SENTINEL_CHECK_NUMBERS, SetRow, mark_cumulative, summarize
from parsers import (
    ROW_RX_CHECK_FIRST,
    ROW_RX_ERA_A,
    ROW_RX_VENDOR_FIRST,
    TOTAL_LINE_RX,
    ParsedListing,
    detect_era,
    money,
    parse_date,
    parse_listing,
    parse_register,
)
from vendors import (
    display_name,
    is_exportable,
    is_organization,
    is_person_shaped,
    load_payee_allowlist,
    normalize_vendor,
)


class TestMoney:
    """Amount parsing, including the three ways this corpus writes a minus."""

    @pytest.mark.parametrize(
        ("raw", "value"),
        [
            ("5,609,073.26", Decimal("5609073.26")),
            ("$ 5,609,073.26", Decimal("5609073.26")),
            ("3 ,609,064.78", Decimal("3609064.78")),
            ("$ 8 4,009.42", Decimal("84009.42")),
            ("$ 2 25.00", Decimal("225.00")),
            ("-1,322.35", Decimal("-1322.35")),
            ("$-2,887.19", Decimal("-2887.19")),
            ("2,887.19-", Decimal("-2887.19")),
            ("0.00", Decimal("0.00")),
        ],
    )
    def test_money(self, raw, value):
        """Amounts parse exactly, whatever punctuation surrounds them."""
        assert money(raw) == value

    def test_money_rejects_non_numbers(self):
        """Empty and punctuation-only strings are not amounts."""
        assert money("") is None
        assert money(None) is None
        assert money(".") is None

    def test_money_is_exact_not_float(self):
        """Money is Decimal end to end; a float would not sum to the cent."""
        total = sum((money(x) for x in ("0.10", "0.20", "0.30")), Decimal("0"))
        assert total == Decimal("0.60")


class TestParseDate:
    """Check dates, including two-digit years and injected spaces."""

    @pytest.mark.parametrize(
        ("raw", "value"),
        [
            ("02/12/2026", date(2026, 2, 12)),
            ("03 /12/2026", date(2026, 3, 12)),
            ("9/24/2020", date(2020, 9, 24)),
            ("9/30/21", date(2021, 9, 30)),
            ("1/5/17", date(2017, 1, 5)),
        ],
    )
    def test_parse_date(self, raw, value):
        """Dates parse in every notation the corpus prints."""
        assert parse_date(raw) == value

    def test_rejects_nonsense(self):
        """A non-date is None rather than a guess."""
        assert parse_date("13/45/2026") is None
        assert parse_date("not a date") is None
        assert parse_date(None) is None


class TestEraDetection:
    """Four generations of report, identified from their first page."""

    @pytest.mark.parametrize(
        ("header", "era"),
        [
            ("KSD VOUCHER REGISTER\nVOUCHER DATES: 09-MAR-07 TO 22-MAR-07", "A"),
            ("Kent School District\nVoucher Register\nCHECK NO. VENDOR DATE", "B"),
            ("Kent School District\nVendor Check date Check # Check Amt Invoice Amt", "C"),
            ("Vendor Check Date Check Number Check Amount Invoice Amount", "D"),
        ],
    )
    def test_detect_era(self, header, era):
        """Each generation's distinctive tokens identify it."""
        assert detect_era(header) == era

    def test_era_c_header_survives_into_2023(self):
        """The Era C header is still in use in 2023, so era is not a date range.

        This is why "cumulative" is measured from check-number overlap rather
        than inferred from the era label -- tagging a whole era would wrongly
        brand current sets.
        """
        header = (
            "Kent School District\nCapital Projects Vouchers\n7/21/2023 - 8/10/2023\n"
            "Vendor Check date Check # Check Amt Invoice Amt Work performed"
        )
        assert detect_era(header, "2023-08-23") == "C"

    def test_date_fallback_when_no_header_survived(self):
        """A date is a weaker signal, used only when the header is gone."""
        assert detect_era("", "2007-03-28") == "A"
        assert detect_era("", "2026-03-25") == "D"

    def test_no_signal_at_all(self):
        """With neither header nor date, the era is unknown, not guessed."""
        assert detect_era("") is None


class TestTotalLine:
    """A TOTAL is a whole line. A vendor named Total is a row."""

    def test_plain_total(self):
        """A bare TOTAL line parses."""
        m = TOTAL_LINE_RX.search("            TOTAL   $  5,609,073.26   \n")
        assert m and money(m.group("amount")) == Decimal("5609073.26")

    def test_fund_labelled_total(self):
        """The register labels its totals with a fund name."""
        m = TOTAL_LINE_RX.search("TOTAL GENERAL FUND    $   45,109,568.76\n")
        assert m and m.group("label").strip() == "TOTAL GENERAL FUND"

    def test_vendor_beginning_total_is_not_a_total(self):
        """A vendor whose name begins "Total" is a row, not a total.

        "Total Technology" is a real vendor on 2026-03-25 GF, page 16.

        Matching it as a total would truncate the set 19,000 characters
        early and the set would still appear to reconcile against the wrong
        figure.
        """
        line = "     Total Technology    02/19/2026 607476     39,799.20    39,799.20 TSS Team - Type Covers\n"
        assert TOTAL_LINE_RX.search(line) is None


class TestRowRegexes:
    """One row pattern per era, each against a real line."""

    def test_era_d_plain(self):
        """Era D without a dollar sign, as 2026-03-25 prints it."""
        line = (
            "     911 Interpreters Inc     02/12/2026 607335       1,435.54"
            "    1,435.54 Open PO for 2025-2026 school year"
        )
        m = ROW_RX_VENDOR_FIRST.match(line)
        assert m
        assert m.group("vendor").strip() == "911 Interpreters Inc"
        assert m.group("chk") == "607335"
        assert money(m.group("invamt")) == Decimal("1435.54")

    def test_era_d_dollar_prefixed(self):
        """Era D with a dollar sign, as 2026-02-11 prints it.

        The brief's pattern admits no dollar sign, which is why it parsed
        zero rows from 268 of the 446 detail listings.
        """
        line = "  Academy Schs      01/29/2026 607141 $ 8 ,108.50 $ 8 ,108.50 Out of district placement"
        m = ROW_RX_VENDOR_FIRST.match(line)
        assert m
        assert money(m.group("chkamt")) == Decimal("8108.50")
        assert money(m.group("invamt")) == Decimal("8108.50")

    def test_era_c_two_digit_year(self):
        """Era C mixes two- and four-digit years within one document."""
        line = "  Rosalind Vance Harper 10/7/21 415040 $ 8 .00 $ 8.00 Refund soccer socks"
        m = ROW_RX_VENDOR_FIRST.match(line)
        assert m
        assert parse_date(m.group("date")) == date(2021, 10, 7)

    def test_era_b_check_number_first(self):
        """Era B puts the check number before the vendor."""
        line = "  411038   Vance, Gregory L   1/12/2017 Choreography svcs-KW Dance $  750.00  $  750.00"
        m = ROW_RX_CHECK_FIRST.match(line)
        assert m
        assert m.group("chk") == "411038"
        assert m.group("vendor").strip() == "Vance, Gregory L"
        assert money(m.group("invamt")) == Decimal("750.00")

    def test_era_a_no_check_date(self):
        """Era A prints a voucher number, a vendor, one amount, a description."""
        line = "  1172920  VAN SICLEN STOCKS &        12,078.85    LEGAL FEES FIRKINS"
        m = ROW_RX_ERA_A.match(line)
        assert m
        assert m.group("chk") == "1172920"
        assert money(m.group("invamt")) == Decimal("12078.85")

    def test_heathman_lodge_row(self):
        """The one row that cost $1,322.35 on 2026-03-25 ASB.

        The vendor name is long enough to collide with the date column, so
        there is no whitespace between them at all, and pdfplumber has split
        the date. Both are why the brief's pattern misses it.
        """
        line = (
            "  THE HEATHMAN LODGE AND HUDSONS BAR AN03 /12/2026 418256      "
            "1,322.35    1,322.35 Cheet to State Hotel fee"
        )
        m = ROW_RX_VENDOR_FIRST.match(line)
        assert m
        assert m.group("vendor").strip().endswith("BAR AN")
        assert m.group("chk") == "418256"
        assert money(m.group("invamt")) == Decimal("1322.35")


class TestPartialDecimalAmounts:
    """This corpus prints invoice amounts with fewer than two decimals.

    Requiring cents was tried and reverted: it dropped 2,108 real rows and
    took nine sets from reconciled to out-of-balance. These lines are from
    the 2025-03-26 ASB listing.
    """

    @pytest.mark.parametrize(
        ("line", "invoice"),
        [
            (
                "  Amazon Capital Services  3/6/2025  417428  1,595.02  123.4 POM POMS",
                Decimal("123.4"),
            ),
            (
                "  Amazon Capital Services  3/13/2025  417439  4,614.54  132 Yarn, cups",
                Decimal("132"),
            ),
            (
                "  Amazon Capital Services  3/6/2025  417428  1,595.02  -8.8 KLA supplies",
                Decimal("-8.8"),
            ),
        ],
    )
    def test_partial_decimals_parse(self, line, invoice):
        """One decimal place, none at all, and a negative all parse."""
        m = ROW_RX_VENDOR_FIRST.match(line)
        assert m is not None
        assert money(m.group("invamt")) == invoice

    def test_one_decimal_is_numerically_equal_to_two(self):
        """123.4 and 123.40 are the same money and sum identically."""
        assert money("123.4") == Decimal("123.40")


class TestParseListing:
    """End-to-end parsing of a synthetic listing."""

    LISTING = (
        "                    General Fund Warrants 02/06/26 through 03/12/26 "
        "and P-Cards 01/17/26 through 02/28/26\n"
        "     Vendor        Check Date Check Number Check Amount Invoice Amount Description\n"
        "     Alpha Inc     02/12/2026 607335       1,000.00     600.00 First invoice\n"
        "     Alpha Inc     02/12/2026 607335       1,000.00     400.00 Second invoice\n"
        "     Beta LLC      02/19/2026 607336         250.00     250.00 Something\n"
        "          continued description text\n"
        "                    TOTAL    $   1,250.00\n"
    )

    def test_rows_and_total(self):
        """Rows parse, the total is found, and the two agree."""
        result = parse_listing(self.LISTING, self.LISTING, "2026-03-25")
        assert result.era == "D"
        assert len(result.rows) == 3
        assert result.stated_total == Decimal("1250.00")
        assert sum(r.invoice_amount for r in result.rows) == Decimal("1250.00")

    def test_continuation_appends_to_description(self):
        """A line with no amount is the wrapped tail of the row above it."""
        result = parse_listing(self.LISTING, self.LISTING, "2026-03-25")
        assert result.rows[-1].description.endswith("continued description text")

    def test_periods_are_read_separately(self):
        """The warrant period and the P-card period are different periods."""
        result = parse_listing(self.LISTING, self.LISTING, "2026-03-25")
        assert result.period == (date(2026, 2, 6), date(2026, 3, 12))
        assert result.pcard_period == (date(2026, 1, 17), date(2026, 2, 28))

    def test_multi_invoice_check_is_deduplicated_by_the_caller(self):
        """Check 607335 is printed twice with the same check amount.

        Summing check_amount without deduplicating would double-count the
        money; summing invoice_amount is what equals the printed total.
        """
        result = parse_listing(self.LISTING, self.LISTING, "2026-03-25")
        first_two = result.rows[:2]
        assert {r.check_number for r in first_two} == {"607335"}
        assert all(r.check_amount == Decimal("1000.00") for r in first_two)
        assert sum(r.invoice_amount for r in first_two) == Decimal("1000.00")

    def test_trailing_total_page_does_not_win(self):
        """A lone TOTAL page after the data is recorded but does not win.

        The 2026-05-27 Transportation listing prints TOTAL $173,922.13 on
        page 1 and TOTAL $347,844.26 -- exactly twice that -- on page 5.
        """
        listing = (
            "  Transportation Vehicle Fund Warrants 03/13/26 through 04/08/26\n"
            "  Vendor  Check Date Check Number Check Amount Invoice Amount Description\n"
            "  Schetky Northwest Sales Inc 04/09/2026 900062  173,922.13 173,922.13 One bus\n"
            "                    TOTAL  $ 173,922.13\n"
            "  Transportation Vehicle Fund Warrants 03/13/26 through 04/08/26\n"
            "                    TOTAL  $ 347,844.26\n"
        )
        result = parse_listing(listing, listing, "2026-05-27")
        assert result.stated_total == Decimal("173922.13")
        assert result.stated_total_rule == "first_total_after_last_row"
        assert [a for _, a, _ in result.extra_totals] == [Decimal("347844.26")]

    def test_unknown_era_yields_no_rows(self):
        """With no era there is no parse, and that is recorded, not guessed."""
        result = parse_listing("nothing here", "nothing here")
        assert result.era is None
        assert result.rows == []


class TestParseRegister:
    """The signed register is a different document with a different shape."""

    REGISTER = (
        "KENT SCHOOL DISTRICT\n"
        "FUND TYPE WARRANT NUMBER ISSUE DATE AMOUNT\n"
        "GENERAL  ACCOUNTS PAYABLE 607335-607409  2/12/2026  1,503,279.74\n"
        "GENERAL  ACCOUNTS PAYABLE DIRECT DEPOSIT-ELECTRONIC TRANSFER 2/12/2026 384,438.45\n"
        "GENERAL  PAYROLL  DIRECT DEPOSIT-ELECTRONIC TRANSFER 2/27/2026 19,437,581.87\n"
        "GENERAL  ACCOUNTS PAYABLE PURCHASING CARD-ELECTRONIC TRANSFER 2/11/26-3/12/26 308,491.87\n"
        "TOTAL GENERAL FUND  $  45,109,568.76\n"
        "CAPITAL  ACCOUNTS PAYABLE 208848-208852  2/12/2026  15,441.14\n"
        "TOTAL CAPITAL PROJECTS FUND  $  1,169,819.12\n"
        "GRAND TOTAL  $  46,434,351.39\n"
    )

    def test_warrant_ranges_are_captured(self):
        """A warrant range is what lets the register be cross-checked."""
        parsed = parse_register(self.REGISTER)
        ranges = [line["warrant_range"] for line in parsed["lines"] if line["warrant_range"]]
        assert "607335-607409" in ranges
        assert "208848-208852" in ranges

    def test_payroll_and_accounts_payable_are_distinguished(self):
        """Payroll direct deposit is not an ACH voucher.

        The 2026-03-25 ACH listing total equals the sum of the ACCOUNTS
        PAYABLE direct-deposit lines only. Including the payroll one would
        overstate it by $19.4M.
        """
        parsed = parse_register(self.REGISTER)
        ap_dd = [line for line in parsed["lines"] if line["is_direct_deposit"] and line["is_accounts_payable"]]
        payroll_dd = [line for line in parsed["lines"] if line["is_direct_deposit"] and line["is_payroll"]]
        assert sum(line["amount"] for line in ap_dd) == Decimal("384438.45")
        assert sum(line["amount"] for line in payroll_dd) == Decimal("19437581.87")

    def test_fund_totals_are_mapped_to_fund_codes(self):
        """The register's fund labels map onto the fact layer's codes."""
        parsed = parse_register(self.REGISTER)
        by_fund = {t["fund"]: t["amount"] for t in parsed["fund_totals"]}
        assert by_fund["GF"] == Decimal("45109568.76")
        assert by_fund["Capital"] == Decimal("1169819.12")

    def test_amount_capture_stops_at_the_date_boundary(self):
        """A register amount must not swallow the year of its issue date.

        Both of these were live in this parser and produced numbers that
        looked entirely plausible: "530157-530158 3/5/2026 355.27" read as
        $2,026,355.27, and "PURCHASING CARD 2/11/26-3/12/26 403.45" read as
        $26,403.45. Neither is out of range for a school district, which is
        exactly why an eye would not catch them.
        """
        register = (
            "GENERAL  PAYROLL  530157-530158  3/5/2026  355.27\n"
            "CAPITAL  ACCOUNTS PAYABLE PURCHASING CARD-ELECTRONIC TRANSFER 2/11/26-3/12/26  403.45\n"
        )
        amounts = [line["amount"] for line in parse_register(register)["lines"]]
        assert amounts == [Decimal("355.27"), Decimal("403.45")]

    def test_a_bare_date_line_is_not_an_amount(self):
        """The register's header date line carries no money."""
        assert parse_register("                 3/25/2026\n")["lines"] == []

    def test_grand_total_is_not_a_fund(self):
        """GRAND TOTAL is not a fund and must not become one."""
        parsed = parse_register(self.REGISTER)
        assert all(t["label"] != "GRAND TOTAL" for t in parsed["fund_totals"])


class TestVendorRules:
    """Normalization, measured against the corpus rather than assumed."""

    @pytest.mark.parametrize(
        ("raw", "norm"),
        [
            ("Amazon Capital Services", "amazon capital services"),
            ("AMAZON CAPITAL SERVICES", "amazon capital services"),
            ("  Amazon   Capital  Services  ", "amazon capital services"),
            ("Consolidated Press Printing In.", "consolidated press printing in"),
            ("KCDA", "kcda"),
        ],
    )
    def test_normalize(self, raw, norm):
        """Trim, collapse, strip trailing punctuation, casefold. Nothing else."""
        assert normalize_vendor(raw) == norm

    def test_case_only_duplicates_merge(self):
        """Casefolding merges exactly the 11 case-only pairs in the corpus."""
        assert normalize_vendor("THE PLUMBING JOINT INC") == normalize_vendor("The Plumbing Joint Inc")

    def test_display_name_never_changes_case(self):
        """KCDA must not become Kcda on a page a director reads aloud."""
        assert display_name("KCDA") == "KCDA"
        assert display_name("WA ST Patrol") == "WA ST Patrol"

    def test_corporate_suffixes_are_not_stripped(self):
        """Smith Inc and Smith LLC can be different legal entities."""
        assert normalize_vendor("Smith Inc") != normalize_vendor("Smith LLC")

    @pytest.mark.parametrize(
        "raw",
        [
            "Harrow, Jason Christopher",
            "Pemberton, Jennifer Marie",
            "Okonkwo, Justin W",
            "Ashby, DeVona L",
            "Calloway, Amy Rose",
        ],
    )
    def test_person_shaped(self, raw):
        """The comma form catches 654 of 3,527 distinct names."""
        assert is_person_shaped(raw)

    @pytest.mark.parametrize(
        "raw",
        ["Amazon Capital Services", "JW Pepper & Son Inc", "KCDA", "Comcast"],
    )
    def test_not_person_shaped(self, raw):
        """A company is not a person, even a one-word company."""
        assert not is_person_shaped(raw)

    @pytest.mark.parametrize(
        "raw",
        ["NWAP, Inc", "Hearing, Speech & Deafness Ctr", "Smith, LLC"],
    )
    def test_comma_shaped_organizations_are_not_people(self, raw):
        """A comma before a legal form is not a surname before a given name.

        Both of the real examples here were withheld from the export as
        though they were individuals until the guard was added.
        """
        assert not is_person_shaped(raw)
        assert is_organization(raw)
        assert is_exportable(raw)

    def test_given_surname_payees_are_not_caught_by_the_comma_rule(self):
        """This is the gap the export's allow-list exists to cover.

        "Bradley Quorvin" is a person being reimbursed. No deterministic rule
        separates that from a two-word company, so the export must not rely
        on detecting it.
        """
        assert not is_person_shaped("Bradley Quorvin")
        assert not is_organization("Bradley Quorvin")
        assert not is_exportable("Bradley Quorvin")

    @pytest.mark.parametrize(
        "raw",
        ["Amazon Capital Services", "JW Pepper & Son Inc", "Micro Computer Systems Inc"],
    )
    def test_organizations_are_exportable(self, raw):
        """Names carrying a business marker may be published."""
        assert is_exportable(raw)

    def test_a_bare_acronym_is_no_longer_exportable(self, monkeypatch, tmp_path):
        """KCDA is a purchasing cooperative and the rule withholds it.

        The classifier is a marker list, and a bare acronym carries no
        marker. The old rule published any single all-caps token; 116
        payees in this corpus qualified that way. Withholding them is the
        safe direction and is what the specified rule does, but it is a
        real loss of context and was raised for the operator in the
        close-out report rather than quietly restored here. The operator
        allowlisted KCDA on 2026-09-18, so this test -- which is about the
        rule, not the operator's file -- reads an empty allowlist.
        """
        monkeypatch.setattr(vendors, "PAYEE_ALLOWLIST_PATH", str(tmp_path / "empty-allowlist.txt"))
        load_payee_allowlist.cache_clear()
        assert not is_exportable("KCDA")
        assert is_exportable("KCDA", frozenset({normalize_vendor("KCDA")}))

    def test_watch_list_overrides(self):
        """A watch-list name is published because the operator chose it.

        "Robert Half" is the example that makes the case: it is a staffing
        company whose name is indistinguishable from an individual's, so no
        rule will ever admit it. The watch list is how a human decision
        overrides a pattern.
        """
        watch = frozenset({normalize_vendor("Robert Half")})
        assert not is_organization("Robert Half")
        assert not is_exportable("Robert Half")
        assert is_exportable("Robert Half", watch)

    def test_person_is_never_exportable_even_on_the_watch_list_path(self):
        """A personal name not on the watch list is never published."""
        watch = frozenset({normalize_vendor("Some Company Inc")})
        assert not is_exportable("Harrow, Jason Christopher", watch)


def _set(set_id: str, lines: list[dict], stated: Decimal | None, extra=None) -> SetRow:
    """Build a SetRow for arithmetic tests.

    Args:
        set_id: Identifier.
        lines: Line dicts.
        stated: The stated total, or None.
        extra: Extra totals found after the chosen one.

    Returns:
        A SetRow with its parser output stubbed.
    """
    parsed = ParsedListing(era="D")
    parsed.stated_total = stated
    parsed.extra_totals = extra or []
    row = SetRow(set_id=set_id, artifact=None, parsed=parsed, fund="GF")
    row.lines = lines
    return row


def _line(number: str, check: str, invoice: str, reason: str | None = None, paren: bool = False) -> dict:
    """Build a minimal line dict.

    Check numbers here are six digits because that is what this corpus
    prints: ``CHECK_NUMBER_RX`` accepts five to eleven. The earlier
    fixtures used ``"1"`` and ``"2"``, which the sentinel rule now reads as
    all-one-digit placeholders -- correctly, and only because the fixture
    was never a shape the documents produce.

    Args:
        number: Check number.
        check: Check amount.
        invoice: Invoice amount.
        reason: Row-level reason code, when the row could not be read.
        paren: Whether the amount was printed in parentheses.

    Returns:
        A line dict with the fields summarize reads.
    """
    return {
        "check_number": number,
        "check_amount": Decimal(check),
        "invoice_amount": Decimal(invoice),
        "reason_code": reason,
        "reason_detail": None,
        "amount_paren": paren,
    }


class TestSummarize:
    """The three states of reconciled, which are not interchangeable."""

    def test_reconciles(self):
        """Lines summing to the stated total, to the cent."""
        row = _set("s", [_line("607001", "10.00", "6.00"), _line("607001", "10.00", "4.00")], Decimal("10.00"))
        summarize(row)
        assert row.reconciled is True
        assert row.delta == Decimal("0")
        assert row.reason_code is None
        assert row.check_count == 1
        assert row.sum_check_dedup == Decimal("10.00")

    def test_out_of_balance_is_false_with_a_reason(self):
        """A real disagreement is false, with a delta and a reason."""
        row = _set("s", [_line("607001", "10.00", "9.00")], Decimal("10.00"))
        summarize(row)
        assert row.reconciled is False
        assert row.delta == Decimal("-1.00")
        assert row.reason_code == "OUT_OF_BALANCE"

    def test_no_stated_total_is_null_not_false(self):
        """303 pre-2023 listings print no total.

        Marking them false would assert the district's arithmetic is wrong
        when what is true is that the document states no arithmetic.
        """
        row = _set("s", [_line("607001", "10.00", "10.00")], None)
        summarize(row)
        assert row.reconciled is None
        assert row.reason_code == "TOTAL_NOT_FOUND"

    def test_an_unread_row_is_held_but_never_summed(self):
        """A row with a reason code is counted, not added and not dropped.

        Adding it would invent money; dropping it would hide a printed
        payment. It is held with its reason and excluded from every total.
        """
        row = _set(
            "s",
            [_line("607001", "10.00", "6.00"), _line("607002", "99.00", "99.00", reason="COLUMN_AMBIGUOUS")],
            Decimal("6.00"),
        )
        summarize(row)
        assert row.unread_lines == 1
        assert row.parsed_total == Decimal("6.00")
        assert row.check_count == 1
        assert row.reconciled is True
        assert "could not be assigned to columns" in " ".join(row.notes)

    def test_a_set_whose_every_row_is_unread_is_flagged(self):
        """Zero readable rows is not a reconciled set at zero."""
        row = _set("s", [_line("607001", "10.00", "10.00", reason="COLUMN_AMBIGUOUS")], Decimal("10.00"))
        summarize(row)
        assert row.reconciled is False
        assert row.reason_code == "COLUMN_AMBIGUOUS"

    def test_zero_rows_is_a_regex_miss(self):
        """No rows at all is a parser failure, and says so."""
        row = _set("s", [], Decimal("10.00"))
        summarize(row)
        assert row.reconciled is False
        assert row.reason_code == "REGEX_MISS"

    def test_multiple_totals_reason_only_when_it_does_not_reconcile(self):
        """An extra TOTAL page is only a reason code when the choice fails."""
        good = _set(
            "s",
            [_line("607001", "10.00", "10.00")],
            Decimal("10.00"),
            extra=[("TOTAL", Decimal("20.00"), 999)],
        )
        summarize(good)
        assert good.reconciled is True
        assert good.reason_code is None
        assert "further TOTAL line" in (good.notes[0] if good.notes else "")

        bad = _set(
            "s",
            [_line("607001", "10.00", "9.00")],
            Decimal("10.00"),
            extra=[("TOTAL", Decimal("20.00"), 999)],
        )
        summarize(bad)
        assert bad.reason_code == "MULTIPLE_TOTALS"

    def test_hash_total_is_the_sum_of_distinct_check_numbers(self):
        """The third control total: a transposition changes it."""
        row = _set(
            "s",
            [_line("607335", "1.00", "1.00"), _line("607336", "2.00", "2.00")],
            Decimal("3.00"),
        )
        summarize(row)
        assert row.hash_total == 607335 + 607336


class TestCumulative:
    """Restated cycles are detected from the data, not from the era."""

    def test_overlap_is_flagged_on_the_later_set(self):
        """A set repeating an earlier set's checks is marked, and says how much."""

        class FakeArtifact:
            def __init__(self, meeting_date):
                self.meeting_date = meeting_date

        first = SetRow("a", FakeArtifact("2021-02-10"), ParsedListing(era="C"), fund="Transportation")
        first.lines = [_line("900001", "100.00", "100.00")]
        second = SetRow("b", FakeArtifact("2021-03-10"), ParsedListing(era="C"), fund="Transportation")
        second.lines = [
            _line("900001", "100.00", "100.00"),
            _line("900002", "50.00", "50.00"),
        ]
        for row in (first, second):
            summarize(row)
        mark_cumulative([second, first])
        assert not any("cumulative" in n for n in first.notes)
        assert any("cumulative" in n for n in second.notes)
        assert "1 of 2 check numbers" in second.notes[0]

    def test_distinct_funds_do_not_collide(self):
        """Two funds may legitimately reuse a check number series."""

        class FakeArtifact:
            def __init__(self, meeting_date):
                self.meeting_date = meeting_date

        gf = SetRow("gf", FakeArtifact("2021-02-10"), ParsedListing(era="C"), fund="GF")
        gf.lines = [_line("601001", "1.00", "1.00")]
        asb = SetRow("asb", FakeArtifact("2021-03-10"), ParsedListing(era="C"), fund="ASB")
        asb.lines = [_line("601001", "1.00", "1.00")]
        mark_cumulative([gf, asb])
        assert not gf.notes
        assert not asb.notes


class TestFlags:
    """P-card, payroll and credit flags."""

    def test_pcard_prefix(self):
        """P-card pseudo-checks begin 926."""
        assert "9261000039".startswith(PCARD_PREFIX)
        assert not "607335".startswith(PCARD_PREFIX)

    def test_sentinel_credit_numbers(self):
        """A credit rides on a sentinel check number, confirmed on 2026-05-27."""
        assert "8888888888" in SENTINEL_CHECK_NUMBERS
