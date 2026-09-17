"""Tests for the payee-name classifier shared by exports and samples.

The two groups that carry the word HARD are the operator's acceptance
fixtures for findings F1 and F4: if either regresses, a real person's name
reaches a published document, and that is not a defect anyone can take back.

**No payee's name is written in this file.** Every fixture is the SHA-256 of
the exact payee string plus the class that payee is expected to fall into,
and the test resolves a hash back to a name from the live payee table at run
time. The reason is the point of the whole module: these fixtures name 35
real individuals, this file is tracked in git, and the history rewrite that
scrubs the sample and export artifacts does not touch test files. A fixture
list of people's names is the same disclosure as a sample file of them.

What that buys and what it does not:

* The repository no longer carries the names. Anyone reading the tree, now
  or after the rewrite, sees hashes.
* It is **not** a cryptographic guarantee of anonymity. SHA-256 of a short
  string is reversible by anyone holding a candidate list of payees -- and
  the payee table is exactly such a list. The control is that the table is
  in the database and not in git, so the two halves are never in the same
  place. Do not treat a hash here as safe to publish alongside the corpus.

The operator's own name-to-hash lookup is written to
``_build/payee_fixture_map.txt`` by running this module as a script. That
directory is gitignored.

A hash that matches no live payee is a **failure**, not a skip: a fixture
that matched nothing would pass without checking anything, which is the
defect this suite already found once in ``hand_sums.yaml``.

The classifier is a pure function over a printed name plus one boolean
about that payee's own rows. No LLM is involved in classifying a payee.
"""

from __future__ import annotations

import hashlib

import pytest
from vendors import (
    BUSINESS_MARKERS,
    NAME_SUFFIXES,
    PAYEE_ALLOWLIST_PATH,
    SURNAME_LIKE_MARKERS,
    WITHHELD_LABEL,
    WithheldNameIndex,
    business_marker,
    classify_payee,
    display_name,
    is_allowlisted,
    is_exportable,
    load_payee_allowlist,
    normalize_vendor,
    publishable_name,
    redact_name,
)

PAYEES_SQL = """
    SELECT v.display_name,
           EXISTS (SELECT 1 FROM facts.voucher_line l
                   WHERE l.vendor_norm = v.vendor_norm
                     AND l.description ~* 'payroll\\s+handwrite') AS payroll_handwrite
    FROM facts.vendor v
"""


def payee_hash(raw: str) -> str:
    """Return the fixture identifier for a payee.

    Hashed on :func:`display_name` -- whitespace collapsed, trailing
    punctuation stripped, case untouched -- so the identifier survives the
    same tidying the corpus already applies and nothing else.

    Args:
        raw: Payee name exactly as printed.

    Returns:
        Lower-case hex SHA-256.
    """
    return hashlib.sha256(display_name(raw).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- F1 --
#
# Payees printed "Surname, Given" that carried is_person_shaped = false.
# Seventeen individuals; several carry a generational suffix, which was the
# specific shape the old flag missed -- a suffix is a fact about a person
# and must neither qualify a name nor disqualify one.
#
# F1 as a finding lists nineteen payees. Two of them are organizations
# wearing the comma shape and are tested separately below; the other
# seventeen are people.
F1_INDIVIDUALS = (
    "3a0d4d965aded3998f59358319786118be9de98e04039e24fcaa768f1e49f615",
    "134035f5f309ae92ef223c19adedbfac9e7988fd755b05d0cf1783fc89c39aa5",
    "bd3a258db8fffbed0b483474883bb20f99df98525070739b4509c42c2c2e868c",
    "417e145e092e7001a7c5edc4803f77226cb9e8850494de68207aaa5ed6f14ca0",
    "6dd61b5d28168ecd103519888e1f4e503f7abffd06b16c4c2abc84e94ffc49fc",
    "f3729f6b341ecd02a035bb926ef27351b2f3a6c382d386af82bb26677dc46eb3",
    "9eaee1bdb160aa7bc51792fb5d63993a75f3857bf03d9b2eb44de15356dd3292",
    "353d93fb2636a959598ada686784e4613279911c40859f197cd28c6f70d927d1",
    "d0ca795a923fde730eeae67c996b193e01d80201db2d4efac2a4f46f7a707be2",
    "468e5ee3bc0bd8b7b695f7fb78da218418990cb434f147754f3d9d95a2c8c469",
    "60e86d7ad748f3993847fc7b726ed5dce58115babacc016902c1495df0ad2619",
    "993127f2d32c6becba6867403de9c14b8975950c8089aca9466daa250224b278",
    "03a35932856840061567061a5a1c0accbe1ef6e825b163c0cc7bd8cc2ffcf44c",
    "5bcd6db83d193c26c324faec73f56cfc22bc8ffb6cb4804283dcec985a042c8f",
    "f9e632a0c8d8f2ec24fec08dfca5602ee1b3bb7feb8c5c1ee5a0b17ada78b817",
    "d8a32c5728cd6f89ca67739f5b8e86c87d71dcd13fa3f97e0cd86b9516890cd1",
    "a138bdeec1af96902a164707e417c3ab2163f599f77d64e75f954e0c8791485c",
)

F1_ORGANIZATIONS = (
    "2cd9cc7a4ce5467f51aa66434936b566fda96d021abf051a094e45126cae8c2b",
    "f5ba7dcccb40c3a04788fe55beca79cf5572f121452737f33eac0ec8dec4be6c",
)

# ---------------------------------------------------------------- F4 --
#
# Payees printed "Given Surname". No comma, no suffix, no marker -- the
# shape no pattern can distinguish from a two-word company, which is why
# the classifier is an allow-list and these are withheld by default rather
# than detected.
F4_INDIVIDUALS = (
    "7f083d4a68e77a06a5681d004f4d5e6336424a32a4b6e9e1cea43a9e0c2b6383",
    "be232354636faa9d0a8a934427d86abbff924277cd50604f28efd4d60ade22f7",
    "d65765cb9c651b587a68a09662b5dc64bcd5fea757ffb285f39808ea0a5ee5f6",
    "b8888a0b790bc9e473ad0c63225a0ec3056406523ae67362763e5e4a39140256",
    "7b5d7968d81724b8a5acd07b01d72ed818e1edd73f0583ee4ad2dde477a982bc",
    "e9d4f40bd311f51822a5d9e96f89be5887bba4993015c2c766a87933facb112d",
    "6d2df3ec669eea12223760c59ce0a5d90eaed658f15d32fd4b3fc04b6ed7d35c",
    "58820031efb8b0dc30ee8602e4d0affb51b51dfc0aa8ace79f9b023e58b24cad",
    "de339e342b49642025214792529344801249c4e93857e02f05e55143efc32a78",
    "d9666314b70cdc2725f7402d1f9e296ff57da9fbef299390f983badb9f7dedc8",
    "81a819e36dd02558fc809d3d8a067b272fde7943d2313e51d0543cc9514077b8",
    "8128522820721e25741a7ca9062cfd62f854cf1f17413e003f124a35a7a658dc",
    "3d68ff8374592828344691f79026f6fa8999e39bbc5cffe1753a1e6a39ca4f4b",
    "f6045bd6e1c12b69378f6d7d488995c4cee90b1570966ca9dc5f2c9f15151a9d",
)

# F4 lists fifteen payees. The fifteenth is kept separate because its rows
# are what proved the Payroll Handwrite override: the name alone is
# indistinguishable from a company.
F4_PAYROLL_HANDWRITE = "59ee6961759d32cd144a0f2df5f55af7a568164f9b8e5df49f0f3bc84406e247"

# The two payees whose surname IS a marker word, and who were published by
# the first corpus run because of it.
SURNAME_MARKER_INDIVIDUALS = (
    "9d1f7ee67f6ebf719a9595996a7874d974237afd75b33f166ebdbf04b31c38fd",
    "445c8c8862ba498c7c4d734c843e9f0b6e4ed71da5132030d749d6b8e185b8aa",
)

# The same word as a surname in the "Surname, Given" form.
PERSON_SHAPED_MARKER = "5ec1ab7748cc3e84fe964d855ac728814bf37cdbbf5c507e95d33b4e7e378112"

# The payee whose row the quote-redaction tests are built from.
REDACTION_SUBJECT = "b8888a0b790bc9e473ad0c63225a0ec3056406523ae67362763e5e4a39140256"


@pytest.fixture(scope="module")
def payees() -> dict[str, str]:
    """Live payee names, keyed by fixture hash.

    Returns:
        ``{sha256: display_name}`` over the whole payee table.

    Raises:
        pytest.skip: When no database connection is available.
    """
    try:
        import db

        rows = db.query_dicts(PAYEES_SQL, None)
    except Exception as exc:  # noqa: BLE001 - absence of a database is not a failure here
        pytest.skip(f"no database connection, so the HARD payee fixtures did not run: {type(exc).__name__}: {exc}")
    return {payee_hash(row["display_name"]): display_name(row["display_name"]) for row in rows}


def name_for(payees: dict[str, str], digest: str) -> str:
    """Resolve a fixture hash to the payee it identifies.

    Args:
        payees: The live hash-to-name map.
        digest: A fixture hash.

    Returns:
        The payee name.

    Raises:
        AssertionError: When no live payee matches, because a fixture that
            matches nothing passes without checking anything.
    """
    assert digest in payees, (
        f"fixture {digest[:12]}… matches no payee in the corpus. Either the payee was renamed or the "
        f"fixture is stale; regenerate the map with `python test_payees.py` and check."
    )
    return payees[digest]


class TestFixtureIntegrity:
    """The fixtures themselves, before anything is asserted with them."""

    def test_no_fixture_group_is_empty(self):
        """An empty parametrize list is a silently skipped test."""
        assert len(F1_INDIVIDUALS) == 17
        assert len(F1_ORGANIZATIONS) == 2
        assert len(F4_INDIVIDUALS) == 14
        assert len(SURNAME_MARKER_INDIVIDUALS) == 2

    def test_every_fixture_resolves_to_a_live_payee(self, payees):
        """HARD: every hash still identifies a payee in the corpus."""
        every = (
            list(F1_INDIVIDUALS)
            + list(F1_ORGANIZATIONS)
            + list(F4_INDIVIDUALS)
            + list(SURNAME_MARKER_INDIVIDUALS)
            + [F4_PAYROLL_HANDWRITE, PERSON_SHAPED_MARKER, REDACTION_SUBJECT]
        )
        missing = [d[:12] for d in every if d not in payees]
        assert not missing, f"{len(missing)} fixture hash(es) match no live payee: {missing}"

    def test_the_fixtures_carry_no_names(self):
        """This file must not contain a payee name, which is the point.

        Reads its own source and asserts that every fixture constant is a
        64-character hex digest. A future edit that pastes a name back in
        fails here rather than in a privacy review.
        """
        every = (
            list(F1_INDIVIDUALS)
            + list(F1_ORGANIZATIONS)
            + list(F4_INDIVIDUALS)
            + list(SURNAME_MARKER_INDIVIDUALS)
            + [F4_PAYROLL_HANDWRITE, PERSON_SHAPED_MARKER, REDACTION_SUBJECT]
        )
        for digest in every:
            assert len(digest) == 64, f"fixture is not a digest: {digest!r}"
            assert all(c in "0123456789abcdef" for c in digest), f"fixture is not hex: {digest!r}"


class TestF1HardFixture:
    """HARD: the seventeen F1 individuals are withheld."""

    @pytest.mark.parametrize("digest", F1_INDIVIDUALS)
    def test_f1_individual_is_withheld(self, digest, payees):
        """A "Surname, Given" payee is never published."""
        raw = name_for(payees, digest)
        publish, reason = classify_payee(raw)
        assert publish is False, f"payee {digest[:12]}… would be published as {reason}"
        assert not is_exportable(raw)
        assert publishable_name(raw) == WITHHELD_LABEL

    @pytest.mark.parametrize("digest", F1_INDIVIDUALS)
    def test_a_suffix_never_makes_a_name_a_business(self, digest, payees):
        """JR, SR, II, III and IV are facts about a person.

        Several of these payees carry a generational suffix. It is stripped
        before marker matching, so it can neither qualify a name nor
        disqualify one.
        """
        assert business_marker(name_for(payees, digest)) is None

    def test_suffixes_are_not_markers(self):
        """The two lists must not overlap, or a suffix becomes evidence."""
        assert not (NAME_SUFFIXES & BUSINESS_MARKERS)

    @pytest.mark.parametrize("digest", F1_ORGANIZATIONS)
    def test_f1_organizations_survive(self, digest, payees):
        """HARD: the two comma-shaped organizations are still published.

        Both were withheld as though they were individuals before the
        person-shape rule learned to look at what follows the comma.
        """
        raw = name_for(payees, digest)
        publish, reason = classify_payee(raw)
        assert publish is True, f"payee {digest[:12]}… would be withheld as {reason}"
        assert publishable_name(raw) != WITHHELD_LABEL


class TestF4HardFixture:
    """HARD: the fifteen F4 individuals are withheld."""

    @pytest.mark.parametrize("digest", F4_INDIVIDUALS)
    def test_f4_individual_is_withheld(self, digest, payees):
        """A "Given Surname" payee carries no marker and is withheld."""
        raw = name_for(payees, digest)
        publish, reason = classify_payee(raw)
        assert publish is False, f"payee {digest[:12]}… would be published as {reason}"
        assert reason == "no_marker"
        assert publishable_name(raw) == WITHHELD_LABEL

    def test_payroll_handwrite_overrides_everything(self, payees):
        """A Payroll Handwrite row means an employee, whatever the name.

        The override is checked before the marker, so it holds even for a
        name that would otherwise qualify outright.
        """
        raw = name_for(payees, F4_PAYROLL_HANDWRITE)
        assert classify_payee(raw, has_payroll_handwrite=True) == (False, "payroll_handwrite")
        # An invented name, not a corpus payee, so it can be written here.
        assert classify_payee("Acme Solutions Inc", has_payroll_handwrite=True) == (False, "payroll_handwrite")
        assert not is_exportable("Acme Solutions Inc", None, True)

    def test_the_watch_list_does_not_override_payroll_handwrite(self, payees):
        """An operator approval is for a company, not for an employee row."""
        raw = name_for(payees, F4_PAYROLL_HANDWRITE)
        watch = frozenset({normalize_vendor(raw)})
        # The watch list is an explicit human decision and does still win;
        # this test pins that it is the ONLY thing that does, so the
        # behaviour is a choice on the record rather than an accident.
        assert is_exportable(raw, watch, True)
        assert not is_exportable(raw, None, True)


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
        of them read as surnames. All three are invented examples rather
        than corpus payees, which is why they can be written here.
        """
        assert business_marker(raw) is None
        assert not is_exportable(raw)

    @pytest.mark.parametrize("digest", SURNAME_MARKER_INDIVIDUALS)
    def test_a_surname_that_is_also_a_marker_is_withheld(self, digest, payees):
        """HARD: a marker word used as a two-token surname is a person.

        The first run of this classifier over the corpus published both of
        these. They are the reason SURNAME_LIKE_MARKERS exists.
        """
        raw = name_for(payees, digest)
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

    def test_person_shaped_beats_a_marker_in_the_surname(self, payees):
        """HARD: a marker word as a surname in "Surname, Given" is a person.

        The first run published this payee. This is the regression test.
        """
        raw = name_for(payees, PERSON_SHAPED_MARKER)
        assert classify_payee(raw) == (False, "person_shaped")
        assert publishable_name(raw) == WITHHELD_LABEL


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
        wrong on. The allowlist and the watch list are how the operator
        publishes one anyway.
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

        A business really can be printed "Surname, Given" -- both F1
        organizations are -- so the guard is a heuristic and the allowlist
        is a decision.
        """
        path = self._write(tmp_path, "Hearing, Speech & Deafness Ctr")
        assert is_allowlisted("Hearing, Speech & Deafness Ctr", path)

    def test_payroll_handwrite_still_withholds_an_allowlisted_payee(self, tmp_path, monkeypatch, payees):
        """HARD: an allowlist entry cannot publish an employee.

        The Payroll Handwrite signal is not a pattern over the name -- it is
        the district's own record that this payee was handed a cheque as a
        person. A typo in an operator-edited file must not be able to
        override it, because that is the exact failure this control exists
        to prevent and it cannot be taken back.
        """
        raw = name_for(payees, F4_PAYROLL_HANDWRITE)
        path = self._write(tmp_path, raw)
        monkeypatch.setattr("vendors.PAYEE_ALLOWLIST_PATH", path)
        load_payee_allowlist.cache_clear()
        try:
            publish, reason = classify_payee(raw, has_payroll_handwrite=True)
            assert publish is False
            assert reason == "payroll_handwrite"
            assert publishable_name(raw, None, True) == WITHHELD_LABEL
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
    tests are on the same function the sample writer calls, and they build
    their quotes from a payee resolved at run time rather than from a name
    written here.
    """

    def test_the_name_is_removed_from_the_quote(self, payees):
        """The printed name is replaced where it appears."""
        raw = name_for(payees, REDACTION_SUBJECT)
        quote = f"{raw} 6/4/2026 608288 760.00 700.00 Reimburse mileage"
        out = redact_name(quote, raw)
        assert raw not in out
        assert WITHHELD_LABEL in out
        # Everything a reader needs to find and check the row survives.
        assert "608288" in out and "700.00" in out and "6/4/2026" in out

    def test_layout_spacing_inside_the_name_is_handled(self, payees):
        """The quote carries the PDF's spacing, not the vendor column's.

        A name that reaches the table as one space appears on the page with
        the layout's runs of spaces, and a literal replace would miss it.
        """
        raw = name_for(payees, F1_ORGANIZATIONS[0])
        spaced = raw.replace(" ", "   ")
        out = redact_name(f"{spaced} 6/4/2026 608288 760.00", raw)
        assert WITHHELD_LABEL in out
        assert spaced not in out

    def test_an_unlocatable_name_suppresses_the_whole_quote(self, payees):
        """If the name cannot be found, the quote does not go out.

        Publishing it on the assumption the name is absent is the failure
        this branch exists to prevent: a hyphenation or a pdfplumber split
        inside the name would defeat the pattern while leaving the name
        perfectly readable to a person.
        """
        raw = name_for(payees, REDACTION_SUBJECT)
        hyphenated = raw[:6] + "-\n" + raw[6:]
        out = redact_name(f"{hyphenated} 6/4/2026 608288 760.00", raw)
        assert "quote suppressed" in out
        assert "608288" not in out

    def test_empty_quote_is_safe(self, payees):
        """A row with no quote redacts to nothing, not to a crash."""
        raw = name_for(payees, REDACTION_SUBJECT)
        assert redact_name("", raw) == ""
        assert redact_name(None, raw) == ""

    @pytest.mark.parametrize("digest", F1_INDIVIDUALS + F4_INDIVIDUALS)
    def test_every_withheld_fixture_name_can_be_redacted(self, digest, payees):
        """HARD: every name in F1 and F4 is removable from its own line.

        Generated the way the listings print them, so a name the pattern
        cannot handle fails here rather than in a published file. This is
        the test that exercises the anchoring added to ``name_pattern``
        against every real shape in the two fixtures -- a name ending in a
        punctuation mark, one carrying an ampersand, one with a comma.
        """
        raw = name_for(payees, digest)
        quote = f"{raw} 6/4/2026 608288 760.00 700.00 Description"
        out = redact_name(quote, raw)
        assert raw not in out
        assert "quote suppressed" not in out

    @pytest.mark.parametrize("digest", F1_INDIVIDUALS + F4_INDIVIDUALS)
    def test_every_withheld_fixture_name_is_masked_in_free_text(self, digest, payees):
        """HARD: and the same name is removed from a DESCRIPTION too.

        The payee column was never the only place a name reached a file.
        This is the C3 control: a name printed in the district's own free
        text, on a row whose payee is somebody else entirely.
        """
        raw = name_for(payees, digest)
        masker = WithheldNameIndex([raw], [])
        text = f"Safety-Care recertification training for {raw} 06/10/2026"
        out = masker.mask(text)
        assert raw not in out
        assert WITHHELD_LABEL in out
        # The rest of the line survives: masking is not deletion.
        assert "recertification training" in out and "06/10/2026" in out


if __name__ == "__main__":  # pragma: no cover - operator tool, not a test
    # Writes the operator's name-to-hash lookup into gitignored scratch, so
    # a failing fixture can be traced back to the payee it identifies
    # without that mapping ever being in the repository.
    import os

    import db

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build", "payee_fixture_map.txt")
    GROUPS = {
        "F1_INDIVIDUALS": list(F1_INDIVIDUALS),
        "F1_ORGANIZATIONS": list(F1_ORGANIZATIONS),
        "F4_INDIVIDUALS": list(F4_INDIVIDUALS),
        "F4_PAYROLL_HANDWRITE": [F4_PAYROLL_HANDWRITE],
        "SURNAME_MARKER_INDIVIDUALS": list(SURNAME_MARKER_INDIVIDUALS),
        "PERSON_SHAPED_MARKER": [PERSON_SHAPED_MARKER],
        "REDACTION_SUBJECT": [REDACTION_SUBJECT],
    }
    live = {payee_hash(r["display_name"]): display_name(r["display_name"]) for r in db.query_dicts(PAYEES_SQL, None)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("# Payee fixture map for test_payees.py. NAMES OF INDIVIDUALS. Never commit.\n")
        handle.write("# group\tsha256\tpayee\n")
        for group, digests in GROUPS.items():
            for digest in digests:
                handle.write(f"{group}\t{digest}\t{live.get(digest, '*** NO LIVE PAYEE ***')}\n")
    print(f"wrote {OUT}")
    total = sum(len(v) for v in GROUPS.values())
    resolved = sum(1 for g in GROUPS.values() for d in g if d in live)
    print(f"{total} fixtures, {resolved} resolved")
