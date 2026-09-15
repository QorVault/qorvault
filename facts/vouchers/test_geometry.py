"""Tests for column assignment by page geometry.

Every case here is a row shape that actually occurs in this corpus, with
the x coordinates measured off the document named in the test. Where a
test looks pedantic, the behaviour it pins was wrong at some point while
R1 was being built, and this is what caught it.

No PDF is opened. A page is a list of character boxes, which is all
``pdfplumber`` gives the module under test.
"""

from __future__ import annotations

from decimal import Decimal

import geometry
import pytest
from geometry import Label, TextRow
from parsers import COLUMN_AMBIGUOUS, read_cells


def glyphs(text: str, x0: float, advance: float = 5.0, top: float = 100.0) -> list[dict]:
    """Lay a string out as character boxes starting at x0.

    Args:
        text: The characters.
        x0: Left edge of the first box.
        advance: Width of each box.
        top: Row position.

    Returns:
        Character dicts in the shape ``pdfplumber`` reports.
    """
    return [
        {"text": char, "x0": x0 + index * advance, "x1": x0 + (index + 1) * advance, "top": top}
        for index, char in enumerate(text)
    ]


def make_row(*pieces: tuple[str, float], advance: float = 5.0, top: float = 100.0) -> TextRow:
    """Build a row from (text, x0) pieces.

    Args:
        *pieces: Each piece's text and left edge.
        advance: Character width.
        top: Row position.

    Returns:
        The row, with its characters in x order.
    """
    chars: list[dict] = []
    for text, x0 in pieces:
        chars.extend(glyphs(text, x0, advance, top))
    chars.sort(key=lambda c: c["x0"])
    return TextRow(top, tuple(chars), "".join(c["text"] for c in chars))


class TestRunSegmentation:
    """Runs are separated by empty space, never by a space character."""

    def test_july_row_segments_without_any_whitespace(self):
        """2026-07-22 prints no gap in the text and six runs on the page.

        This is the row the whole change exists for: extracted as text it
        reads ``AIRGAS USA LLC6/18/202692526018813.153.152025-26 Helium``
        and no regular expression can find the column edges in it.
        """
        row = make_row(
            ("AIRGAS USA LLC", 40.08),
            ("6/18/2026", 202.56),
            ("9252601881", 228.60),
            ("3.15", 284.64),
            ("3.15", 326.52),
            ("2025-26 Helium tank rental", 338.64),
            advance=2.4,
        )
        runs = geometry.row_runs(row)
        assert [run.text for run in runs] == [
            "AIRGAS USA LLC",
            "6/18/2026",
            "9252601881",
            "3.15",
            "3.15",
            "2025-26 Helium tank rental",
        ]

    def test_a_space_inside_a_description_does_not_start_a_run(self):
        """``Helium tank`` is one column, not two."""
        row = make_row(("Helium tank rental", 338.64), advance=2.4)
        assert len(geometry.row_runs(row)) == 1

    def test_words_split_on_spaces_but_runs_do_not(self):
        """Header matching needs words; corridor measuring needs runs."""
        row = make_row(("Check Date", 198.0), advance=2.4)
        assert [word.text for word in geometry.row_words(row)] == ["Check", "Date"]
        assert [run.text for run in geometry.row_runs(row)] == ["Check Date"]

    def test_a_padding_space_does_not_widen_a_run(self):
        """``$     20,360.00`` is one run whose extent is the printed ink."""
        row = make_row(("$     20,360.00", 327.67), advance=3.0)
        run = geometry.row_runs(row)[0]
        assert run.text == "$     20,360.00"
        assert run.x0 == pytest.approx(327.67)


class TestHeaderMatching:
    """The printed header names the columns and fixes their order."""

    def test_wrapped_check_amount_above_the_spine(self):
        """2026-07-22 prints ``Check`` a line above ``Amount``."""
        above = make_row(("Check", 279.84), advance=2.5, top=44.5)
        spine = make_row(
            ("Vendor", 40.08),
            ("Check Date", 198.0),
            ("Check Number", 228.6),
            ("Amount", 275.88),
            ("Invoice Amount", 300.12),
            ("Description", 338.64),
            advance=2.5,
            top=51.8,
        )
        found = geometry.find_header([above, spine])
        assert found is not None
        schema, labels, _ = found
        assert schema.name == "vendor_first"
        assert [label.key for label in labels] == [
            "vendor",
            "check_date",
            "check_number",
            "check_amount",
            "invoice_amount",
            "description",
        ]

    def test_wrapped_check_number_below_the_spine(self):
        """2026-06-24 GF prints ``Check`` above and ``Number`` below it."""
        above = make_row(("Check", 374.14), advance=2.9, top=80.1)
        spine = make_row(
            ("Vendor", 71.78),
            ("Check Date", 308.14),
            ("Check Amount", 436.29),
            ("Invoice Amount", 521.35),
            ("Description", 608.14),
            advance=2.9,
            top=87.17,
        )
        below = make_row(("Number", 374.14), advance=2.9, top=94.46)
        found = geometry.find_header([above, spine, below])
        assert found is not None
        assert [label.key for label in found[1]] == [
            "vendor",
            "check_date",
            "check_number",
            "check_amount",
            "invoice_amount",
            "description",
        ]

    def test_a_listing_with_no_description_column(self):
        """2026-02-11 Trust ends at the invoice amount and that is valid."""
        spine = make_row(
            ("Vendor", 41.28),
            ("Check Date", 239.76),
            ("Check Number", 304.32),
            ("Check Amount", 376.56),
            ("Invoice Amount", 448.32),
            advance=2.9,
            top=54.0,
        )
        found = geometry.find_header([spine])
        assert found is not None
        assert [label.key for label in found[1]] == [
            "vendor",
            "check_date",
            "check_number",
            "check_amount",
            "invoice_amount",
        ]

    @pytest.mark.parametrize("spelling", ["Check #", "Check#", "Check No.", "CHECK NO"])
    def test_check_number_spellings(self, spelling):
        """Spacing and full stops in a header label mean nothing."""
        spine = make_row(
            ("Vendor Name", 40.0),
            ("Check Date", 150.0),
            (spelling, 220.0),
            ("Check Amt", 280.0),
            ("Invoice Amt", 340.0),
            ("Work Performed", 400.0),
            advance=2.9,
            top=60.0,
        )
        found = geometry.find_header([spine])
        assert found is not None
        assert [label.key for label in found[1]][:3] == ["vendor", "check_date", "check_number"]

    def test_account_code_columns_collapse_into_one_trailing_column(self):
        """Era B and C print eight account-code columns this layer ignores.

        Their left edge is kept, because without it a long description
        would bleed into them.
        """
        spine = make_row(
            ("Vendor Name", 22.2),
            ("Check Date", 109.92),
            ("Check #", 135.12),
            ("Check Amt", 159.0),
            ("Invoice Amt", 186.72),
            ("Invoice Description", 218.04),
            ("FD", 496.92),
            ("LDG", 503.88),
            ("PRGM", 525.0),
            advance=2.2,
            top=62.86,
        )
        found = geometry.find_header([spine])
        assert found is not None
        keys = [label.key for label in found[1]]
        assert keys[-1] == geometry.TRAILING_KEY
        assert keys[:6] == [
            "vendor",
            "check_date",
            "check_number",
            "check_amount",
            "invoice_amount",
            "description",
        ]

    def test_a_header_that_matches_nothing_yields_no_grid(self):
        """The by-vendor year-to-date summary is not a voucher listing."""
        spine = make_row(("Vendor Name", 40.92), ("Year-to-Date", 300.0), advance=2.9, top=108.26)
        assert geometry.find_header([spine]) is None


def labels_for(*spec: tuple[str, float, float]) -> list[Label]:
    """Build header labels from (key, x0, x1) triples.

    Args:
        *spec: Each label's key and extent.

    Returns:
        The labels.
    """
    return [Label(key=key, text=key, x0=x0, x1=x1) for key, x0, x1 in spec]


class TestBoundaryPlacement:
    """Where inside a corridor the boundary goes, and why."""

    def test_boundary_hugs_the_anchored_side_when_the_left_column_grows(self):
        """A vendor column grows right; a date column starts at a fixed x.

        2026-03-25 ASB truncates ``THE HEATHMAN LODGE AND HUDSONS BAR AN``
        hard against the check date. Splitting that corridor down the
        middle puts the end of the vendor name in the date column and
        loses the row -- 1,322.35 of a set that reconciles to the cent.
        """
        labels = labels_for(("vendor", 67.93, 101.53), ("check_date", 261.15, 312.64))
        samples = [
            make_row(("ABC Co", 67.59), ("02/12/2026", 260.81), advance=4.6),
            make_row(("A Longer Vendor Name", 67.59), ("03/12/2026", 260.81), advance=4.6),
        ]
        measured = geometry.build_boundaries(labels, samples)
        assert measured is not None
        assert measured.boundaries[0] == pytest.approx(260.81 - geometry.HUG_MARGIN)

    def test_boundary_hugs_left_when_the_right_column_is_right_aligned(self):
        """An amount column grows leftwards as its values get wider.

        2026-07-22 ACH prints check amounts that end at a fixed x and
        start wherever the value is wide enough to reach.
        """
        labels = labels_for(("check_number", 228.6, 262.12), ("check_amount", 275.88, 294.61))
        samples = [
            make_row(("9252601881", 228.6), ("3.15", 284.64), advance=2.4),
            make_row(("9252601882", 228.6), ("20,594.57", 272.28), advance=2.4),
        ]
        measured = geometry.build_boundaries(labels, samples)
        assert measured is not None
        # Right edge of the check numbers, plus the clearance margin.
        assert measured.boundaries[0] == pytest.approx(252.6 + geometry.HUG_MARGIN)

    def test_boundary_splits_the_corridor_when_both_columns_grow(self):
        """With nothing anchored there is no better answer than the middle."""
        labels = labels_for(("vendor", 40.0, 60.0), ("check_amount", 200.0, 230.0))
        samples = [
            make_row(("Short", 40.0), ("1.00", 210.0), advance=2.0),
            make_row(("A much longer vendor", 40.0), ("123,456.00", 190.0), advance=2.0),
        ]
        measured = geometry.build_boundaries(labels, samples)
        assert measured is not None
        assert measured.boundaries[0] == pytest.approx((80.0 + 190.0) / 2)

    def test_a_boundary_outside_the_header_band_is_recorded(self):
        """The header does not always bracket the true corridor.

        On 2026-03-25 ASB the ``Check Amount`` label sits about 30 pt left
        of its own data. The corridor is still measured correctly; the
        disagreement with the header is recorded rather than hidden.
        """
        labels = labels_for(("check_amount", 394.37, 431.54), ("invoice_amount", 464.34, 501.51))
        samples = [
            make_row(("1,322.35", 427.53), ("1,322.35", 498.91), advance=4.16),
            make_row(("3.15", 444.21), ("3.15", 515.6), advance=4.16),
        ]
        measured = geometry.build_boundaries(labels, samples)
        assert measured is not None
        # The corridor is 30 pt to the right of the label's own right edge.
        assert measured.boundaries[0] > labels[0].x1
        assert measured.extents[0][0] > labels[0].x0

    def test_a_corridor_outside_the_header_band_is_recorded(self):
        """A boundary the header does not bracket is reported, not hidden."""
        labels = labels_for(("check_number", 324.0, 391.15), ("check_amount", 394.37, 431.54))
        samples = [
            make_row(("418200", 323.8), ("3.15", 444.21), advance=4.86),
            make_row(("418256", 323.8), ("1,322.35", 427.53), advance=4.16),
        ]
        measured = geometry.build_boundaries(labels, samples)
        assert measured is not None
        assert measured.outside_header_band
        assert "check_number|check_amount" in measured.outside_header_band[0]

    def test_a_document_with_no_well_formed_row_yields_nothing(self):
        """Measuring corridors needs at least one row that segments."""
        labels = labels_for(("vendor", 40.0, 60.0), ("check_amount", 200.0, 230.0))
        assert geometry.build_boundaries(labels, []) is None


class TestAssignment:
    """Characters go to columns by midpoint, and overlap is not ambiguity."""

    def test_overlapping_glyph_boxes_still_assign_correctly(self):
        """Measured off 2026-03-25 ASB page 3.

        The vendor's final ``N`` spans 255.685-262.379 and the date's first
        ``0`` spans 260.949-265.773: they overlap by 1.43 pt, so no
        boundary exists that is outside both boxes. Their midpoints are
        4.3 pt apart and are not close.
        """
        row = TextRow(
            868.08,
            (
                {"text": "N", "x0": 255.685, "x1": 262.379, "top": 868.08},
                {"text": "0", "x0": 260.949, "x1": 265.773, "top": 868.08},
                {"text": "3", "x0": 265.827, "x1": 270.650, "top": 868.08},
            ),
            "N03",
        )
        grid = geometry.ColumnGrid(
            schema="vendor_first",
            labels=tuple(labels_for(("vendor", 67.93, 101.53), ("check_date", 261.15, 312.64))),
            boundaries=(259.81,),
            corridors=(46.23,),
            header_page=1,
            left_margin=67.6,
        )
        assigned = geometry.assign_row(row, grid)
        assert assigned.get("vendor") == "N"
        assert assigned.get("check_date") == "03"
        assert not assigned.ambiguous

    def test_a_character_centred_on_a_boundary_is_ambiguous(self):
        """A glyph whose midpoint sits on the boundary is not on a side."""
        row = TextRow(
            100.0,
            ({"text": "7", "x0": 258.0, "x1": 261.6, "top": 100.0},),
            "7",
        )
        grid = geometry.ColumnGrid(
            schema="vendor_first",
            labels=tuple(labels_for(("vendor", 60.0, 100.0), ("check_date", 261.15, 312.64))),
            boundaries=(259.81,),
            corridors=(46.23,),
            header_page=1,
            left_margin=60.0,
        )
        assert geometry.assign_row(row, grid).ambiguous

    def test_the_r1_row_reads_as_printed(self):
        """The 2026-06-24 ASB line the regex parser lost 238.00 on.

        Printed: check 355, invoice 240, description beginning with a
        digit. Read as text the three run together as ``3552402``.
        """
        row = make_row(
            ("Head Quarters Corp", 50.0),
            ("6/11/2026", 240.0),
            ("418449", 300.0),
            ("355", 360.0),
            ("240", 420.0),
            ("2 Standard Portable Toilets for KM Athletics Use", 470.0),
            advance=2.4,
        )
        grid = geometry.ColumnGrid(
            schema="vendor_first",
            labels=tuple(
                labels_for(
                    ("vendor", 50.0, 90.0),
                    ("check_date", 240.0, 262.0),
                    ("check_number", 300.0, 315.0),
                    ("check_amount", 360.0, 368.0),
                    ("invoice_amount", 420.0, 428.0),
                    ("description", 470.0, 500.0),
                )
            ),
            boundaries=(235.0, 295.0, 355.0, 415.0, 465.0),
            corridors=(140.0, 40.0, 40.0, 45.0, 40.0),
            header_page=1,
            left_margin=50.0,
        )
        assigned = geometry.assign_row(row, grid)
        parsed, problems = read_cells(assigned, grid.keys)
        assert not problems
        assert parsed.check_amount == Decimal("355")
        assert parsed.invoice_amount == Decimal("240")
        assert parsed.description == "2 Standard Portable Toilets for KM Athletics Use"

    def test_a_trailing_minus_in_the_description_is_text(self):
        """``25-26 Legal Services`` is a description, not a credit.

        The regex parser read ``106 25-`` as -10,625.00 and turned a $106
        legal invoice into a $10,625 credit.
        """
        row = make_row(
            ("Pacifica Law Group LLP", 50.0),
            ("6/4/2026", 240.0),
            ("9252601789", 300.0),
            ("40,817.50", 350.0),
            ("106", 420.0),
            ("25-26 Legal Services", 470.0),
            advance=2.4,
        )
        grid = geometry.ColumnGrid(
            schema="vendor_first",
            labels=tuple(
                labels_for(
                    ("vendor", 50.0, 90.0),
                    ("check_date", 240.0, 262.0),
                    ("check_number", 300.0, 325.0),
                    ("check_amount", 350.0, 372.0),
                    ("invoice_amount", 420.0, 428.0),
                    ("description", 470.0, 500.0),
                )
            ),
            boundaries=(235.0, 295.0, 345.0, 415.0, 465.0),
            corridors=(140.0, 40.0, 25.0, 45.0, 40.0),
            header_page=1,
            left_margin=50.0,
        )
        parsed, problems = read_cells(geometry.assign_row(row, grid), grid.keys)
        assert not problems
        assert parsed.invoice_amount == Decimal("106")
        assert parsed.check_amount == Decimal("40817.50")


class TestTypeChecks:
    """A column that fails its own type check is a failure, not a guess."""

    def _grid(self):
        """Build a six-column Era D grid with generous corridors.

        Returns:
            The grid.
        """
        return geometry.ColumnGrid(
            schema="vendor_first",
            labels=tuple(
                labels_for(
                    ("vendor", 50.0, 90.0),
                    ("check_date", 240.0, 262.0),
                    ("check_number", 300.0, 325.0),
                    ("check_amount", 350.0, 372.0),
                    ("invoice_amount", 420.0, 428.0),
                    ("description", 470.0, 500.0),
                )
            ),
            boundaries=(235.0, 295.0, 345.0, 415.0, 465.0),
            corridors=(140.0, 40.0, 25.0, 45.0, 40.0),
            header_page=1,
            left_margin=50.0,
        )

    def test_a_non_date_in_the_date_column_fails_the_row(self):
        """Nothing is borrowed from the vendor column to make it parse."""
        grid = self._grid()
        row = make_row(
            ("Some Vendor", 50.0),
            ("NOT A DATE", 240.0),
            ("418449", 300.0),
            ("355.00", 350.0),
            ("240.00", 420.0),
            advance=2.0,
        )
        _, problems = read_cells(geometry.assign_row(row, grid), grid.keys)
        assert any("not a date" in problem for problem in problems)

    def test_a_non_numeric_check_number_fails_the_row(self):
        """A check number with a letter in it is not a check number."""
        grid = self._grid()
        row = make_row(
            ("Some Vendor", 50.0),
            ("6/11/2026", 240.0),
            ("41A449", 300.0),
            ("355.00", 350.0),
            ("240.00", 420.0),
            advance=2.0,
        )
        _, problems = read_cells(geometry.assign_row(row, grid), grid.keys)
        assert any("not a check number" in problem for problem in problems)

    def test_an_empty_invoice_amount_fails_the_row(self):
        """A missing invoice amount is a failure, never a zero."""
        grid = self._grid()
        row = make_row(("Some Vendor", 50.0), ("6/11/2026", 240.0), ("418449", 300.0), ("355.00", 350.0), advance=2.0)
        parsed, problems = read_cells(geometry.assign_row(row, grid), grid.keys)
        assert any("not an amount" in problem for problem in problems)
        assert parsed.invoice_amount is None

    def test_the_failure_code_is_the_only_one(self):
        """One code covers every way column assignment can fail."""
        assert COLUMN_AMBIGUOUS == "COLUMN_AMBIGUOUS"
