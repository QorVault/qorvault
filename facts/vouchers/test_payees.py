"""Tests for the payee-name classifier shared by exports and samples.

Every name in this file is a real payee in this corpus. The two lists that
carry the word HARD are the operator's acceptance fixtures for findings F1
and F4: if either regresses, a real person's name reaches a published
document, and that is not a defect anyone can take back.

The classifier is a pure function over a printed name plus one boolean
about that payee's own rows. No LLM is involved in classifying a payee.
"""

from __future__ import annotations

import pytest
from vendors import (
    BUSINESS_MARKERS,
    NAME_SUFFIXES,
    PAYEE_ALLOWLIST_PATH,
    SURNAME_LIKE_MARKERS,
    WITHHELD_LABEL,
    business_marker,
    classify_payee,
    is_allowlisted,
    is_exportable,
    load_payee_allowlist,
    normalize_vendor,
    publishable_name,
    redact_name,
)

# ---------------------------------------------------------------- F1 --
#
# Payees printed "Surname, Given" that carried is_person_shaped = false.
# Seventeen individuals; several carry a generational suffix, which was the
# specific shape the old flag missed.
#
# The two names the operator carved out of F1 -- "Hearing, Speech &
# Deafness Ctr" and "NWAP, Inc" -- are businesses and are tested separately
# below. F1 as a finding lists nineteen payees; seventeen of them are
# people.
F1_INDIVIDUALS = [
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
]

# The two F1 payees that are organizations wearing the comma shape.
F1_ORGANIZATIONS = ["Hearing, Speech & Deafness Ctr", "NWAP, Inc"]

# ---------------------------------------------------------------- F4 --
#
# Payees printed "Given Surname". No comma, no suffix, no marker -- the
# shape no pattern can distinguish from a two-word company, which is why
# the classifier is an allow-list and these are withheld by default rather
# than detected.
F4_INDIVIDUALS = [
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
    "(individual payee, name withheld)",
]

# F4 lists fifteen payees. "(individual payee, name withheld)" is the fifteenth and is kept
# separate only because its rows are what proved the Payroll Handwrite
# override: the name alone is indistinguishable from a company.
F4_PAYROLL_HANDWRITE = "(individual payee, name withheld)"


class TestF1HardFixture:
    """HARD: the seventeen F1 individuals are withheld."""

    @pytest.mark.parametrize("raw", F1_INDIVIDUALS)
    def test_f1_individual_is_withheld(self, raw):
        """A "Surname, Given" payee is never published."""
        publish, reason = classify_payee(raw)
        assert publish is False, f"{raw} would be published as {reason}"
        assert not is_exportable(raw)
        assert publishable_name(raw) == WITHHELD_LABEL

    @pytest.mark.parametrize("raw", F1_INDIVIDUALS)
    def test_a_suffix_never_makes_a_name_a_business(self, raw):
        """JR, SR, II, III and IV are facts about a person.

        "(individual payee, name withheld)" is the case that makes the point: the
        suffix is stripped before marker matching, so it can neither
        qualify a name nor disqualify one.
        """
        assert business_marker(raw) is None

    def test_suffixes_are_not_markers(self):
        """The two lists must not overlap, or a suffix becomes evidence."""
        assert not (NAME_SUFFIXES & BUSINESS_MARKERS)

    @pytest.mark.parametrize("raw", F1_ORGANIZATIONS)
    def test_f1_organizations_survive(self, raw):
        """HARD: the two comma-shaped organizations are still published.

        Both were withheld as though they were individuals before the
        person-shape rule learned to look at what follows the comma.
        """
        publish, reason = classify_payee(raw)
        assert publish is True, f"{raw} would be withheld as {reason}"
        assert publishable_name(raw) != WITHHELD_LABEL


class TestF4HardFixture:
    """HARD: the fifteen F4 individuals are withheld."""

    @pytest.mark.parametrize("raw", F4_INDIVIDUALS)
    def test_f4_individual_is_withheld(self, raw):
        """A "Given Surname" payee carries no marker and is withheld."""
        publish, reason = classify_payee(raw)
        assert publish is False, f"{raw} would be published as {reason}"
        assert reason == "no_marker"
        assert publishable_name(raw) == WITHHELD_LABEL

    def test_payroll_handwrite_overrides_everything(self):
        """A Payroll Handwrite row means an employee, whatever the name.

        The override is checked before the marker, so it holds even for a
        name that would otherwise qualify outright.
        """
        assert classify_payee(F4_PAYROLL_HANDWRITE, has_payroll_handwrite=True) == (
            False,
            "payroll_handwrite",
        )
        assert classify_payee("Acme Solutions Inc", has_payroll_handwrite=True) == (
            False,
            "payroll_handwrite",
        )
        assert not is_exportable("Acme Solutions Inc", None, True)

    def test_the_watch_list_does_not_override_payroll_handwrite(self):
        """An operator approval is for a company, not for an employee row."""
        watch = frozenset({normalize_vendor(F4_PAYROLL_HANDWRITE)})
        # The watch list is an explicit human decision and does still win;
        # this test pins that it is the ONLY thing that does, so the
        # behaviour is a choice on the record rather than an accident.
        assert is_exportable(F4_PAYROLL_HANDWRITE, watch, True)
        assert not is_exportable(F4_PAYROLL_HANDWRITE, None, True)


class TestMarkerMatching:
    """Whole-token matching, and the surnames that are also markers."""

    @pytest.mark.parametrize(
        ("raw", "marker"),
        [
            ("Apple Inc", "inc"),
            ("ArbiterSports LLC", "llc"),
            ("AccuTrain Corporation", "corporation"),
            ("Bellevue College", "college"),
            ("Auburn SD", "sd"),
            ("Crestwood PTA", "pta"),
            ("Kentridge HS Booster Club", "hs"),
            ("Akurate Solutions", "solutions"),
            ("United States Treasury", "treasury"),
        ],
    )
    def test_markers_publish(self, raw, marker):
        """A payee carrying a marker is published, and says which."""
        assert classify_payee(raw) == (True, f"marker:{marker}")

    @pytest.mark.parametrize("raw", ["Cortez Landscaping", "Cochran Electric", "Increase Miller"])
    def test_substring_is_not_a_marker(self, raw):
        """Matching is whole-token.

        "Cortez" contains "corp", "Cochran" contains "co" and "Increase"
        contains "inc". Substring matching would publish all three, and two
        of them are surnames.
        """
        assert business_marker(raw) is None
        assert not is_exportable(raw)

    @pytest.mark.parametrize("raw", ["(individual payee, name withheld)", "(individual payee, name withheld)"])
    def test_a_surname_that_is_also_a_marker_is_withheld(self, raw):
        """HARD: "Church" as a two-token name is a person.

        The first run of this classifier over the corpus published both of
        these. They are the reason SURNAME_LIKE_MARKERS exists.
        """
        assert business_marker(raw) is None
        assert classify_payee(raw) == (False, "no_marker")

    @pytest.mark.parametrize(
        "raw",
        [
            "Faith Baptist Church",
            "Kent Covenant Church",
            "Lake Sawyer Christian Church",
            "Seattle Buddhist Church Matsuri Taiko",
        ],
    )
    def test_a_congregation_is_published(self, raw):
        """Three tokens or more, and "Church" is a description again."""
        assert classify_payee(raw) == (True, "marker:church")

    def test_church_is_the_only_surname_like_marker(self):
        """Pin the list so widening it is a deliberate, tested act."""
        assert SURNAME_LIKE_MARKERS == {"church"}

    def test_person_shaped_beats_a_marker_in_the_surname(self):
        """HARD: "(individual payee, name withheld)" is a person named Church.

        The first run published him. This is the regression test.
        """
        assert classify_payee("(individual payee, name withheld)") == (False, "person_shaped")
        assert publishable_name("(individual payee, name withheld)") == WITHHELD_LABEL


class TestWithholdingIsTheDefault:
    """Anything the rules cannot positively identify is withheld."""

    @pytest.mark.parametrize("raw", ["", None, "   "])
    def test_empty_is_withheld(self, raw):
        """No name is not a business."""
        assert not is_exportable(raw)
        assert publishable_name(raw) == WITHHELD_LABEL

    @pytest.mark.parametrize("raw", ["KCDA", "AFSCME", "Teamsters", "Robert Half"])
    def test_organizations_without_a_marker_are_withheld(self, raw):
        """Being obviously a company is not the test; carrying a marker is.

        All four are organizations and all four are withheld. That is the
        cost side of the allow-list, and it is the side that is safe to be
        wrong on. The watch list is how the operator publishes one anyway.
        """
        assert not is_exportable(raw)
        assert is_exportable(raw, frozenset({normalize_vendor(raw)}))


class TestPayeeAllowlist:
    """The operator's allowlist file: what it releases and what it cannot.

    Every test here writes its own temporary file. None of them reads or
    edits the packaged allowlist, except the one that asserts it is empty.
    """

    @staticmethod
    def _write(tmp_path, *lines):
        """Write a temporary allowlist and clear the loader's cache.

        Args:
            tmp_path: pytest temporary directory.
            *lines: File lines, written verbatim.

        Returns:
            Path to the file.
        """
        path = tmp_path / "allowlist.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        load_payee_allowlist.cache_clear()
        return str(path)

    def test_the_packaged_allowlist_ships_empty(self):
        """HARD: nothing is published by default.

        An allowlist that arrived with entries in it would be an agent
        deciding to publish names. If this fails, read the diff before the
        code: someone added a payee.
        """
        load_payee_allowlist.cache_clear()
        assert load_payee_allowlist(PAYEE_ALLOWLIST_PATH) == frozenset()

    def test_an_entry_publishes_a_payee_that_carries_no_marker(self, tmp_path):
        """The whole point: release a bare acronym without widening a rule."""
        path = self._write(tmp_path, "KCDA")
        assert is_allowlisted("KCDA", path)
        assert not is_allowlisted("AFSCME", path)

    def test_matching_is_the_corpus_key_not_the_literal_string(self, tmp_path):
        """Case and spacing do not matter, because they do not matter here.

        ``normalize_vendor`` already merges these spellings into one payee,
        so an entry that released only one of them would withhold the same
        payee on the cycles where it is typed differently.
        """
        path = self._write(tmp_path, "  Amazon Capital Services  ")
        assert is_allowlisted("AMAZON CAPITAL SERVICES", path)
        assert is_allowlisted("Amazon Capital Services.", path)

    def test_an_entry_is_not_a_substring_or_a_pattern(self, tmp_path):
        """One entry releases one payee identity and no other."""
        path = self._write(tmp_path, "KCDA")
        assert not is_allowlisted("KCDA Warehouse", path)
        assert not is_allowlisted("Friends of KCDA", path)
        assert not is_allowlisted("KCD", path)

    def test_comments_and_blank_lines_are_ignored(self, tmp_path):
        """The file is edited by a person, so it has to tolerate notes."""
        path = self._write(tmp_path, "# a comment", "", "   ", "  # indented comment", "KCDA")
        assert load_payee_allowlist(path) == frozenset({normalize_vendor("KCDA")})

    def test_a_missing_file_is_an_empty_allowlist(self, tmp_path):
        """The scaffold has to work before the operator fills it in."""
        load_payee_allowlist.cache_clear()
        assert load_payee_allowlist(str(tmp_path / "nope.txt")) == frozenset()

    def test_an_allowlisted_name_outranks_the_person_shape_guard(self, tmp_path):
        """HARD-adjacent: the operator's explicit decision beats a pattern.

        ``Hearing, Speech & Deafness Ctr`` is the real case -- a business
        wearing the comma shape. The guard is a heuristic; the allowlist is
        a decision.
        """
        path = self._write(tmp_path, "Hearing, Speech & Deafness Ctr")
        assert is_allowlisted("Hearing, Speech & Deafness Ctr", path)

    def test_payroll_handwrite_still_withholds_an_allowlisted_payee(self, tmp_path, monkeypatch):
        """HARD: an allowlist entry cannot publish an employee.

        The Payroll Handwrite signal is not a pattern over the name -- it is
        the district's own record that this payee was handed a cheque as a
        person. A typo in an operator-edited file must not be able to
        override it, because that is the exact failure this control exists
        to prevent and it cannot be taken back.
        """
        path = self._write(tmp_path, "(individual payee, name withheld)")
        monkeypatch.setattr("vendors.PAYEE_ALLOWLIST_PATH", path)
        load_payee_allowlist.cache_clear()
        try:
            publish, reason = classify_payee("(individual payee, name withheld)", has_payroll_handwrite=True)
            assert publish is False
            assert reason == "payroll_handwrite"
            assert publishable_name("(individual payee, name withheld)", None, True) == WITHHELD_LABEL
        finally:
            load_payee_allowlist.cache_clear()

    def test_the_classifier_reports_the_allowlist_as_its_reason(self, tmp_path, monkeypatch):
        """A published name says why, so a report column can show it."""
        path = self._write(tmp_path, "AFSCME")
        monkeypatch.setattr("vendors.PAYEE_ALLOWLIST_PATH", path)
        load_payee_allowlist.cache_clear()
        try:
            assert classify_payee("AFSCME") == (True, "allowlist")
            assert is_exportable("AFSCME")
            assert publishable_name("AFSCME") == "AFSCME"
        finally:
            load_payee_allowlist.cache_clear()


class TestQuoteRedaction:
    """A withheld name must not survive in the verbatim quote beside it.

    The sample files print the source line as extracted, so withholding the
    vendor column alone publishes the name one column to the right. These
    tests are on the same function the sample writer calls.
    """

    def test_the_name_is_removed_from_the_quote(self):
        """The printed name is replaced where it appears."""
        quote = "(individual payee, name withheld) 6/4/2026 608288 760.00 700.00 Reimburse mileage"
        out = redact_name(quote, "(individual payee, name withheld)")
        assert "(individual payee, name withheld)" not in out
        assert WITHHELD_LABEL in out
        # Everything a reader needs to find and check the row survives.
        assert "608288" in out and "700.00" in out and "6/4/2026" in out

    def test_layout_spacing_inside_the_name_is_handled(self):
        """The quote carries the PDF's spacing, not the vendor column's.

        A name that reaches the table as one space appears on the page with
        the layout's runs of spaces, and a literal replace would miss it.
        """
        quote = "Hearing,   Speech &  Deafness  Ctr 6/4/2026 608288 760.00"
        out = redact_name(quote, "Hearing, Speech & Deafness Ctr")
        assert "Deafness" not in out
        assert WITHHELD_LABEL in out

    def test_an_unlocatable_name_suppresses_the_whole_quote(self):
        """If the name cannot be found, the quote does not go out.

        Publishing it on the assumption the name is absent is the failure
        this branch exists to prevent: a hyphenation or a pdfplumber split
        inside the name would defeat the pattern while leaving the name
        perfectly readable to a person.
        """
        quote = "Corrie-\nanne Kelly 6/4/2026 608288 760.00"
        out = redact_name(quote, "(individual payee, name withheld)")
        assert "quote suppressed" in out
        assert "608288" not in out

    def test_empty_quote_is_safe(self):
        """A row with no quote redacts to nothing, not to a crash."""
        assert redact_name("", "(individual payee, name withheld)") == ""
        assert redact_name(None, "(individual payee, name withheld)") == ""

    @pytest.mark.parametrize("raw", F1_INDIVIDUALS + F4_INDIVIDUALS)
    def test_every_withheld_fixture_name_can_be_redacted(self, raw):
        """HARD: every name in F1 and F4 is removable from its own line.

        Generated the way the listings print them, so a name the pattern
        cannot handle fails here rather than in a published file.
        """
        quote = f"{raw} 6/4/2026 608288 760.00 700.00 Description"
        out = redact_name(quote, raw)
        assert raw not in out
        assert "quote suppressed" not in out
