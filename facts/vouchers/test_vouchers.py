"""Tests for the voucher fact layer.

Every test here pins a behaviour that a real document in this corpus
depends on. Where a test looks pedantic, the corresponding bug was live
in this package at some point during Phase 0 and the test is what caught
it.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from classify import (
    doc_class_from_title,
    era_band,
    excluded_reason,
    fiscal_year,
    fund_from_text,
    fund_from_title,
    normalize_title,
)
from locators import PATH_REWRITES, make_quote, meeting_date_from_path
from recon_phase0 import ROW_RX, ROW_RX_BRIEF, TOTAL_LINE_RX, money


class TestTitleNormalization:
    r"""Separator normalization, which a whole class of near-misses needs.

    Regex ``\b`` does not fire between a letter and an underscore. Before ``normalize_title`` existed, every
    ``Warrant_Recap_3-8-17.pdf`` in the corpus fell through unclassified while ``Warrant Recap 3-8-17.pdf``
    classified correctly -- 26 recaps and 92 detail listings were silently invisible.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Warrant_Recap.pdf", "Warrant Recap.pdf"),
            ("Warrant+Recap.pdf", "Warrant Recap.pdf"),
            ("WARRANT_RECAP.pdf", "WARRANT RECAP.pdf"),
            ("Capital_Projects_Vouchers_2-8-17.pdf", "Capital Projects Vouchers 2 8 17.pdf"),
            ("GF Voucher Rpt_091119.pdf", "GF Voucher Rpt 091119.pdf"),
            ("Vouchers", "Vouchers"),
        ],
    )
    def test_separators_become_spaces(self, raw, expected):
        """Separators become spaces."""
        assert normalize_title(raw) == expected

    def test_extension_is_preserved(self):
        """The file extension survives normalization.

        A ".docx" suffix is the signal that an artifact is a resolution rather
        than a payment listing; collapsing it into the body would lose that.
        """
        assert normalize_title("Resolution_1370_-_Cancellation.docx").endswith(".docx")

    def test_underscore_titles_classify(self):
        """Underscore titles classify."""
        assert doc_class_from_title("Warrant_Recap_3-8-17.pdf") == "warrant_recap"
        assert fund_from_title("Capital_Projects_Vouchers_2-8-17.pdf") == "Capital"
        assert fund_from_title("General_Fund.pdf") == "GF"


class TestFundClassification:
    """Fund rules against titles that actually occur in the corpus."""

    @pytest.mark.parametrize(
        ("title", "fund"),
        [
            ("ACH Vouchers 03-25-26.pdf", "ACH"),
            ("ACH Funds Vouchers 10-09-24.pdf", "ACH"),
            ("General Fund Vouchers 03-25-26.pdf", "GF"),
            ("GF Voucher Rpt_091119.pdf", "GF"),
            ("General_fund_Vouchers_1-25-17.pdf", "GF"),
            ("Capital Vouchers 03-25-26.pdf", "Capital"),
            ("CP Vouchers 9-11-19.pdf", "Capital"),
            ("CPF Vouchers 1-25-23 to 2-7-23.pdf", "Capital"),
            ("Captial Project Fund Vouchers 12.8.2020.pdf", "Capital"),
            ("ASB Vouchers 03-25-26.pdf", "ASB"),
            ("Associate Student Body Funds Vouchers 01-14-26.pdf", "ASB"),
            ("Trust Vouchers 5.13.2020.pdf", "Trust"),
            ("Vision_Trust_Vouchers_8-23-17.pdf", "Trust"),
            ("TVF Vouchers 05-27-26.pdf", "Transportation"),
            ("Transporation Vouchers 5-27-2020.pdf", "Transportation"),
            ("TR Vouchers 10-9-24 to 10-22-24.pdf", "Transportation"),
            ("Custodial Vouchers 01-22-25 to 02-04-25.pdf", "Custodial"),
            ("Permanent Funds Vouchers 01-14-26.pdf", "Permanent"),
        ],
    )
    def test_fund_from_title(self, title, fund):
        """Fund from title."""
        assert fund_from_title(title) == fund

    def test_capital_wins_over_fund_word(self):
        """Capital beats the generic Fund rule, which would otherwise have to guess."""
        assert fund_from_title("Capital Projects Fund Vouchers 6.24.2020.pdf") == "Capital"

    def test_ach_wins_over_fund_word(self):
        """Ach wins over fund word."""
        assert fund_from_title("ACH Funds Vouchers 10-09-24.pdf") == "ACH"

    @pytest.mark.parametrize(
        ("phrase", "fund"),
        [
            ("General Fund Warrants 02/06/26 through 03/12/26 and P-Cards", "GF"),
            ("ACH Payments 02/06/26 through 03/12/26", "ACH"),
            ("Capital Projects Fund Warrants 01/09/26 through 02/05/26", "Capital"),
            ("Associated Student Body Fund Warrants 02/06/26 through 03/12/26", "ASB"),
            ("Transportation Vehicle Fund Warrants 03/13/26 through 04/08/26", "Transportation"),
        ],
    )
    def test_fund_from_printed_phrase(self, phrase, fund):
        """Fund from printed phrase."""
        assert fund_from_text(phrase) == fund

    def test_printed_phrase_is_authoritative_over_title(self):
        """The printed header is the only fund signal some files carry.

        "Board Summary Vouchers 9.28.2022.pdf" names no fund in its title at all.
        """
        assert fund_from_title("Board Summary Vouchers 9.28.2022.pdf") is None
        assert fund_from_text("General Fund Warrants 9/1/22 through 9/28/22") == "GF"


class TestDocumentClass:
    """The four artifact kinds are not interchangeable evidence."""

    @pytest.mark.parametrize(
        ("title", "doc_class"),
        [
            ("Board Mtg 03-25-26 SIGNED.pdf", "warrant_register"),
            ("BDMTG - 5-27-2026 SIGNED.pdf", "warrant_register"),
            ("BDMTG - 10-26-22 - Signed.pdf", "warrant_register"),
            ("10-26-22 BOARD MTG SIGNED.pdf", "warrant_register"),
            ("Board Summary Vouchers 9.28.2022.pdf", "warrant_register"),
            ("Warrant_Recap.pdf", "warrant_recap"),
            ("Warrant Summary 9-11-19.pdf", "warrant_recap"),
            ("Voucher Recap 11-14-18.pdf", "warrant_recap"),
            ("Resolution No. 1370 Cancellation of Warrants", "warrant_cancellation"),
            ("List_of_Warrants_for_Cancelation_9-13-17.pdf", "warrant_cancellation"),
            ("OldWarrants.pdf", "warrant_cancellation"),
            ("GF Vendor Rpt_YTD032520.pdf", "voucher_by_vendor"),
            ("YTD_GF Vendor Report 1.1.2021.pdf", "voucher_by_vendor"),
            ("General Fund Vouchers 03-25-26.pdf", "detail_listing"),
            ("General_Fund.pdf", "detail_listing"),
            ("Trust 1-22-20.pdf", "detail_listing"),
        ],
    )
    def test_doc_class(self, title, doc_class):
        """Doc class."""
        assert doc_class_from_title(title) == doc_class

    def test_register_beats_voucher_word(self):
        """A title naming both the register and vouchers is the register.

        "10-26-22 BOARD MTG VOUCHERS SIGNED.pdf" would be parsed as a vendor
        listing if the detail rule ran first, and the cross-check would then
        compare a document against itself.
        """
        assert doc_class_from_title("10-26-22 BOARD MTG VOUCHERS SIGNED.pdf") == "warrant_register"

    def test_vendor_report_is_not_a_monthly_set(self):
        """A year-to-date vendor report overlaps the monthly listings.

        Taking it as a set would double-count a year of payments.
        """
        assert doc_class_from_title("GF Vendor Rpt_YTD032520.pdf") == "voucher_by_vendor"


class TestExclusions:
    """Words that mean something else elsewhere in the corpus."""

    @pytest.mark.parametrize(
        ("title", "reason"),
        [
            ("Board Meeting Transcript - 2019-05-22", "meeting_transcript"),
            ("Regular Board Meeting Transcript - 2021-01-13", "meeting_transcript"),
            ("Board Meeting Minutes Executive Session 20220914.pdf", "meeting_minutes"),
            ("Attachment for 20240208 Board Meeting - NS Equipment 2.pdf", "meeting_attachment"),
            ("2024-25 Device Warranties & Services - Micro Computer Systems.pdf", "warranty_not_warrant"),
            ("3b_KW_E_Parking_Lot_Statutory_Warranty_Deed.pdf", "warranty_not_warrant"),
            ("2024 - Audit of Expenditures - Final.pdf", "audit_report"),
        ],
    )
    def test_excluded(self, title, reason):
        """Excluded."""
        assert excluded_reason(title) == reason

    def test_transcripts_never_reach_the_candidate_sweep(self):
        """Transcripts are kept out one step earlier, by the title sweep.

        The sweep matches "%board mtg%" and "%bdmtg%" but not "%board
        meeting%". The exclusion rule above is a guard in case the sweep is
        ever widened, so both layers are pinned rather than one.
        """
        from census import TITLE_PATTERNS

        assert "%board meeting%" not in TITLE_PATTERNS
        assert "%board mtg%" in TITLE_PATTERNS
        assert "%bdmtg%" in TITLE_PATTERNS

    def test_voucher_titles_are_not_excluded(self):
        """Voucher titles are not excluded."""
        for title in (
            "General Fund Vouchers 03-25-26.pdf",
            "BDMTG - 5-27-2026 SIGNED.pdf",
            "Warrant_Recap.pdf",
        ):
            assert excluded_reason(title) is None


class TestFiscalYearAndBand:
    """Washington school fiscal years run 1 September to 31 August."""

    @pytest.mark.parametrize(
        ("date", "fy"),
        [
            ("2025-08-26", "FY2025"),
            ("2025-09-10", "FY2026"),
            ("2026-03-25", "FY2026"),
            ("2026-08-26", "FY2026"),
            ("2026-09-23", "FY2027"),
        ],
    )
    def test_fiscal_year(self, date, fy):
        """Fiscal year."""
        assert fiscal_year(date) == fy

    def test_fiscal_year_of_unknown_date(self):
        """Fiscal year of unknown date."""
        assert fiscal_year(None) is None

    @pytest.mark.parametrize(
        ("date", "band"),
        [("2005-05-11", "2005-2009"), ("2017-02-08", "2015-2019"), ("2026-03-25", "2025-2029")],
    )
    def test_era_band(self, date, band):
        """Era band."""
        assert era_band(date) == band


class TestPathResolution:
    """Two stale roots, not one."""

    def test_workspace_rewrite_is_tried_first(self):
        """Both stale roots begin "/home/donald/".

        If the generic archive rewrite ran first, every 2026 re-scrape path would resolve to a directory that does
        not exist and look like a missing file.
        """
        stale_prefixes = [stale for stale, _ in PATH_REWRITES]
        workspace = "/home/donald/workspace/projects/ksd_forensic/boarddocs/data/"
        generic = "/home/donald/"
        assert stale_prefixes.index(workspace) < stale_prefixes.index(generic)

    def test_meeting_date_from_path(self):
        """Meeting date from path."""
        path = (
            "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic"
            "/boarddocs/data/2026-03-25-regular-meeting-6-30-p-m-"
            "/9-12-ds4muh5ca61b-vouchers/General Fund Vouchers 03-25-26.pdf"
        )
        assert meeting_date_from_path(path) == "2026-03-25"

    def test_meeting_date_absent(self):
        """A path with no meeting slug in it. Nothing is created or read."""
        assert meeting_date_from_path("/nonexistent/nowhere/file.pdf") is None


class TestMoney:
    """pdfplumber's layout mode can split a number with a space."""

    @pytest.mark.parametrize(
        ("raw", "value"),
        [
            ("5,609,073.26", Decimal("5609073.26")),
            ("3 ,609,064.78", Decimal("3609064.78")),
            ("8 4,009.42", Decimal("84009.42")),
            ("-1,322.35", Decimal("-1322.35")),
            ("0.00", Decimal("0.00")),
        ],
    )
    def test_money(self, raw, value):
        """Money."""
        assert money(raw) == value

    def test_money_rejects_non_numbers(self):
        """Money rejects non numbers."""
        assert money("") is None
        assert money("-") is None


class TestTotalLine:
    """A TOTAL is a whole line. A vendor named Total is a row."""

    def test_plain_total(self):
        """Plain total."""
        m = TOTAL_LINE_RX.search("                    TOTAL   $  5,609,073.26   \n")
        assert m is not None
        assert money(m.group("amount")) == Decimal("5609073.26")

    def test_total_with_space_inside_number(self):
        """Total with space inside number."""
        m = TOTAL_LINE_RX.search("      TOTAL           $ 3 ,609,064.78\n")
        assert m is not None
        assert money(m.group("amount")) == Decimal("3609064.78")

    def test_fund_labelled_total(self):
        """Fund labelled total."""
        m = TOTAL_LINE_RX.search("TOTAL GENERAL FUND    $   45,109,568.76\n")
        assert m is not None
        assert m.group("label").strip() == "TOTAL GENERAL FUND"

    def test_grand_total(self):
        """Grand total."""
        m = TOTAL_LINE_RX.search("GRAND TOTAL           $  46,434,351.39\n")
        assert m is not None
        assert m.group("label").strip() == "GRAND TOTAL"

    def test_vendor_beginning_total_is_not_a_total(self):
        """This line is real: "Total Technology" is a vendor on the 2026-03-25 General Fund listing, page 16.

        Matching it as a total would truncate the set 19,000 characters early.
        """
        line = (
            "     Total Technology    02/19/2026 607476     39,799.20    39,799.20 "
            "TSS Team - Type Covers - Microsoft Surface Pro 7 Keyboards\n"
        )
        assert TOTAL_LINE_RX.search(line) is None

    def test_vendor_beginning_total_still_parses_as_a_row(self):
        """Vendor beginning total still parses as a row."""
        line = (
            "     Total Technology    02/19/2026 607476     39,799.20    39,799.20 "
            "TSS Team - Type Covers - Microsoft Surface Pro 7 Keyboards"
        )
        m = ROW_RX.match(line)
        assert m is not None
        assert m.group("vendor").strip() == "Total Technology"
        assert m.group("chk") == "607476"
        assert money(m.group("chkamt")) == Decimal("39799.20")


class TestRowRegex:
    """Row shape, taken from real 2026-03-25 General Fund lines."""

    def test_simple_row(self):
        """Simple row."""
        line = (
            "               911 Interpreters Inc            02/12/2026 607335"
            "       1,435.54    1,435.54 Open PO for 2025-2026 school year"
        )
        m = ROW_RX.match(line)
        assert m is not None
        assert m.group("vendor").strip() == "911 Interpreters Inc"
        assert m.group("date") == "02/12/2026"
        assert m.group("chk") == "607335"
        assert money(m.group("chkamt")) == Decimal("1435.54")
        assert money(m.group("invamt")) == Decimal("1435.54")
        assert m.group("desc").startswith("Open PO")

    def test_multi_invoice_check_repeats_the_check_amount(self):
        """Check 607487 is printed once per invoice with the same check amount.

        Summing check_amount without deduplicating by check number double-counts the money.
        """
        lines = [
            "   911 Interpreters Inc   02/26/2026 607487       4,950.81    3,146.15 Open PO",
            "   911 Interpreters Inc   02/26/2026 607487       4,950.81    1,804.66 Open PO",
        ]
        parsed = [ROW_RX.match(ln) for ln in lines]
        assert all(parsed)
        assert {m.group("chk") for m in parsed} == {"607487"}
        assert sum(money(m.group("invamt")) for m in parsed) == Decimal("4950.81")
        assert money(parsed[0].group("chkamt")) == Decimal("4950.81")

    def test_vendor_with_digits_and_punctuation(self):
        """Vendor with digits and punctuation."""
        line = "   ALL HANDS CMTY INTERP SVCS   02/12/2026 607336       7,191.41     425.00 ASL"
        m = ROW_RX.match(line)
        assert m is not None
        assert m.group("vendor").strip() == "ALL HANDS CMTY INTERP SVCS"

    def test_header_line_is_not_a_row(self):
        """Header line is not a row."""
        line = "      Vendor       Check Date  Check Number  Check Amount Invoice Amount Description"
        assert ROW_RX.match(line) is None

    def test_pcard_pseudo_check_number_is_accepted(self):
        """P-card pseudo-checks are 10 digits beginning 926; the brief's 5-11 digit window has to admit them."""
        line = "   US BANK PCARD   06/12/2026 9261000039      224,719.75   1,204.55 P-Card purchases"
        m = ROW_RX.match(line)
        assert m is not None
        assert m.group("chk") == "9261000039"

    def test_payroll_warrant_number_is_accepted(self):
        """Payroll warrant number is accepted."""
        line = "   A DIRECTOR   06/18/2026 530162      1,513.50   1,513.50 Travel reimbursement"
        m = ROW_RX.match(line)
        assert m is not None
        assert m.group("chk") == "530162"


class TestBriefRegexVersusFixed:
    """The two changes to the brief's row regex, each pinned to its cause.

    Both were found by running the brief's own regex against the brief's own hard fixture and measuring the
    shortfall, not by reading the pattern.
    """

    HEATHMAN = (
        "  THE HEATHMAN LODGE AND HUDSONS BAR AN03 /12/2026 418256      1,322.35    1,322.35 Cheet to State Hotel fee"
    )

    def test_brief_regex_misses_the_heathman_row(self):
        """This single line is the entire $1,322.35 gap the brief records against the 2026-03-25 ASB set.

        The vendor name is long enough to collide with the date column, so there is no whitespace at all between
        them, and pdfplumber splits the date as "03 /12/2026".
        """
        assert ROW_RX_BRIEF.match(self.HEATHMAN) is None

    def test_fixed_regex_catches_it(self):
        """Fixed regex catches it."""
        m = ROW_RX.match(self.HEATHMAN)
        assert m is not None
        assert m.group("chk") == "418256"
        assert money(m.group("invamt")) == Decimal("1322.35")
        assert m.group("date").replace(" ", "") == "03/12/2026"

    def test_fixed_regex_does_not_eat_digits_off_the_vendor(self):
        r"""Relaxing the vendor/date separator to \\s* risks the vendor donating its trailing digits to the date.

        Non- greedy matching takes the earliest possible date, so it does not.
        """
        m = ROW_RX.match(self.HEATHMAN)
        assert m.group("vendor").strip().endswith("BAR AN")

    def test_both_regexes_agree_on_a_well_spaced_row(self):
        """Both regexes agree on a well spaced row."""
        line = (
            "   911 Interpreters Inc      02/12/2026 607335       1,435.54"
            "    1,435.54 Open PO for 2025-2026 school year"
        )
        a, b = ROW_RX_BRIEF.match(line), ROW_RX.match(line)
        assert a and b
        assert a.group("vendor").strip() == b.group("vendor").strip()
        assert a.group("chk") == b.group("chk")
        assert money(a.group("invamt")) == money(b.group("invamt"))


class TestQuote:
    """Locator quotes are verbatim, whitespace-normalized, and bounded."""

    def test_collapses_whitespace(self):
        """Collapses whitespace."""
        text = "TOTAL      $   5,609,073.26"
        assert make_quote(text, 0, len(text)) == "TOTAL $ 5,609,073.26"

    def test_respects_max_len(self):
        """Respects max len."""
        text = "x" * 500
        assert len(make_quote(text, 0, 500, max_len=100)) == 100


class TestRegexHygiene:
    """The patterns compile and stay anchored."""

    def test_total_line_is_multiline_anchored(self):
        """Total line is multiline anchored."""
        assert TOTAL_LINE_RX.flags & re.M

    def test_row_regex_requires_two_amounts(self):
        """One amount is not a data row; it is usually a wrapped description."""
        assert ROW_RX.match("   Vendor Name   02/12/2026 607335    1,435.54") is None
