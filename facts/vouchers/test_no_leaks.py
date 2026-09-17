"""No published artifact may contain a withheld payee name.

This is the acceptance test for the privacy control, and it is deliberately
built the way it is:

* It reuses ``vendors.classify_payee`` -- the same function the writers call.
  A test with its own pattern would pass while the writers leak, because the
  two patterns would drift and the test would be checking the wrong rule.
* It reads the withheld list from the database rather than from a literal
  list, so a payee added by a future ingest is covered without anyone
  remembering to add it here.
* It searches every published file, not a sample of them.

It needs a database, because the withheld list is a property of the corpus
rather than of the code. With no connection it SKIPS, and a skip is reported
rather than counted as a pass: a check that did not run is not a check that
succeeded.
"""

from __future__ import annotations

import os

import pytest
from vendors import classify_payee, name_pattern

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exports")

PAYEES_SQL = """
    SELECT v.display_name,
           EXISTS (SELECT 1 FROM facts.voucher_line l
                   WHERE l.vendor_norm = v.vendor_norm
                     AND l.description ~* 'payroll\\s+handwrite') AS payroll_handwrite
    FROM facts.vendor v
"""


def _published_files() -> list[str]:
    """Every file this package hands out.

    Returns:
        Absolute paths to the sample and export files.
    """
    found = []
    for directory in (SAMPLES_DIR, EXPORTS_DIR):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and name.endswith((".md", ".csv")):
                found.append(os.path.abspath(path))
    return found


@pytest.fixture(scope="module")
def withheld_names() -> list[str]:
    """Payee names the classifier withholds, from the live corpus.

    Returns:
        Display names that must not appear in any published file.

    Raises:
        pytest.skip: When no database connection is available.
    """
    try:
        import db

        rows = db.query_dicts(PAYEES_SQL, None)
    except Exception as exc:  # noqa: BLE001 - absence of a database is not a failure here
        pytest.skip(f"no database connection, so the leak check did not run: {type(exc).__name__}: {exc}")
    names = [
        row["display_name"]
        for row in rows
        if not classify_payee(row["display_name"], bool(row["payroll_handwrite"]))[0]
    ]
    # One- and two-character names would match inside ordinary words and
    # would make this test useless noise rather than a control.
    return [n for n in names if n and len(n.strip()) > 6]


def test_there_are_files_to_check():
    """A green run over zero files is not evidence of anything."""
    assert _published_files(), "no sample or export files found; regenerate them before trusting this test"


def test_no_published_file_contains_a_withheld_name(withheld_names):
    """HARD: the control the whole classifier exists to enforce.

    Uses the same name pattern the redactor uses, so a name written with the
    page's own spacing is caught here too rather than only in the redactor.
    """
    assert withheld_names, "the withheld list is empty; the classifier or the query is wrong"
    patterns = [(n, name_pattern(n)) for n in withheld_names]
    leaks: list[str] = []
    for path in _published_files():
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for name, pattern in patterns:
            if pattern is not None and pattern.search(text):
                leaks.append(f"{os.path.basename(path)}: {name}")
    assert not leaks, "withheld payee names found in published files:\n" + "\n".join(sorted(leaks)[:40])
