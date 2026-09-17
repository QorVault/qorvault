"""Tests for the hand-sum fixture file and its reader.

The reader is hand-rolled rather than pyyaml, which is a decision this
package already made for facts/minutes. A hand-rolled reader has to be
tested harder than a library one, because the failure mode is silent: a
fixture whose field did not parse is a fixture that quietly did not run,
and a check that did not run is not a check that passed.

The figures themselves are asserted against the database by
``fixtures.check_hand_sums``. These tests are about the file.
"""

from __future__ import annotations

from decimal import Decimal

import fixtures
import pytest

# `import fixtures` pulls in `db`, but db.py opens no connection at import
# time -- it only builds connection kwargs when a query runs. So these tests
# need psycopg2 installed and nothing else.
#
# An earlier version of this file stubbed `db` into sys.modules to avoid
# even that. It worked, and it also silently disabled test_no_leaks.py: that
# module imports the real db inside a try/except and SKIPS when the import
# misbehaves, so the stub turned the HARD privacy check into a skip whenever
# these tests ran first. Do not reintroduce it.


@pytest.fixture(scope="module")
def entries() -> dict:
    """The parsed hand-sum file.

    Returns:
        Mapping of fixture name to fields.
    """
    return fixtures.load_hand_sums()


class TestTheFileParses:
    """The file is present and every entry survives the reader."""

    def test_all_three_entries_are_present(self, entries):
        """Two HARD hand sums and one ADVISORY provenance record."""
        assert set(entries) == {
            "fixture_A_teamsters_2026",
            "fixture_B_united_volleyball",
            "manual_parse_2026",
        }

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        """An absent file blocks the fixture; it does not crash the run."""
        assert fixtures.load_hand_sums(str(tmp_path / "nope.yaml")) == {}

    def test_a_malformed_entry_raises(self, tmp_path):
        """A line that is neither a pair nor a continuation fails loudly.

        Silently skipping it would drop a field, and a fixture missing its
        expected value passes vacuously.
        """
        path = tmp_path / "bad.yaml"
        path.write_text("  orphan continuation with no key\n", encoding="utf-8")
        with pytest.raises(ValueError):
            fixtures.load_hand_sums(str(path))

    def test_wrapped_prose_is_folded_not_truncated(self, entries):
        """The operator wrapped long values; the reader must rejoin them.

        Truncating at the wrap would silently shorten `covers`, which is the
        field that says what a fixture does NOT demonstrate -- the most
        dangerous field to lose.
        """
        covers = entries["fixture_B_united_volleyball"]["covers"]
        assert covers.startswith("the dedupe only.")
        assert "no figure from these sets is speakable" in covers

    def test_an_inline_comment_does_not_become_part_of_the_value(self, entries):
        """`value: 30344.00   # 10 lines, 5 checks` is the number alone."""
        assert entries["fixture_A_teamsters_2026"]["value"] == "30344.00"


class TestProvenanceIsRecorded:
    """Every fixture says who produced it, when, and how."""

    @pytest.mark.parametrize("name", ["fixture_A_teamsters_2026", "fixture_B_united_volleyball"])
    def test_hard_fixtures_name_a_person_and_a_method(self, entries, name):
        """A HARD fixture must be attributable, or it cannot be an oracle."""
        entry = entries[name]
        assert entry["tier"] == "HARD"
        assert entry["who"].startswith("Don Cook (operator)")
        assert entry["when"] == "2026-09-15"
        assert entry["method"] and entry["method"] != "unknown"

    def test_the_unattributed_parse_is_advisory_with_who_unknown(self, entries):
        """Retiering this was the point: nobody knows who made it.

        An unattributed figure cannot be asked what it counted, so it is
        context and never a failure.
        """
        entry = entries["manual_parse_2026"]
        assert entry["tier"] == "ADVISORY"
        assert entry["who"] == "unknown"
        assert entry["method"] == "unknown"


class TestTheAssertedFiguresMatchTheProse:
    """The assert_* keys restate `value`; they must not drift from it."""

    def test_fixture_a_assertions(self, entries):
        """Pass-through: raw and deduplicated are the same figure."""
        entry = entries["fixture_A_teamsters_2026"]
        assert Decimal(entry["assert_raw_sum"]) == Decimal("30344.00")
        assert Decimal(entry["assert_deduped_sum"]) == Decimal("30344.00")
        assert Decimal(entry["value"]) == Decimal(entry["assert_raw_sum"])
        assert int(entry["assert_line_count"]) == 10
        assert int(entry["assert_check_count"]) == 5
        assert len(entry["assert_cycles"].split(",")) == 5

    def test_fixture_b_assertions(self, entries):
        """The dedupe removes exactly one line worth 409.65 and no more."""
        entry = entries["fixture_B_united_volleyball"]
        raw = Decimal(entry["assert_raw_sum"])
        deduped = Decimal(entry["assert_deduped_sum"])
        difference = Decimal(entry["assert_difference"])
        assert raw == Decimal("4324.26")
        assert deduped == Decimal("3914.61")
        assert raw - deduped == difference == Decimal("409.65")
        assert int(entry["assert_line_count"]) - int(entry["assert_deduped_line_count"]) == 1
        assert entry["assert_restated_check"] == "417751"

    def test_the_two_fixtures_bound_the_dedupe_from_both_sides(self, entries):
        """A: removes nothing when nothing repeats. B: removes exactly one.

        Either alone can be satisfied by a broken dedupe -- A by one that
        never removes anything, B by one that removes too much elsewhere.
        Together they cannot.
        """
        a = entries["fixture_A_teamsters_2026"]
        b = entries["fixture_B_united_volleyball"]
        assert Decimal(a["assert_raw_sum"]) - Decimal(a["assert_deduped_sum"]) == Decimal("0")
        assert Decimal(b["assert_raw_sum"]) - Decimal(b["assert_deduped_sum"]) > Decimal("0")

    def test_fixture_b_records_that_its_sets_are_not_speakable(self, entries):
        """The caveat travels with the number or it will be lost.

        Three of the four source sets are OUT_OF_BALANCE and one is
        TOTAL_INCONSISTENT_AT_SOURCE. The fixture verifies the dedupe, not
        the sets, and the file has to say so where the figure is.
        """
        covers = entries["fixture_B_united_volleyball"]["covers"]
        assert "OUT_OF_BALANCE" in covers
        assert "TOTAL_INCONSISTENT_AT_SOURCE" in covers
        assert "not speakable" in covers or "no figure from these sets is speakable" in covers
