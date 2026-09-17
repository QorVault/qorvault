"""No published artifact may contain a withheld payee name.

This is the acceptance test for the privacy control, and it is deliberately
built the way it is:

* It reuses ``vendors.WithheldNameIndex`` -- the same object the export and
  sample writers mask with. A test with its own pattern would pass while the
  writers leak, because the two patterns would drift and the test would be
  checking the wrong rule.
* It asks the question the writers answer, **with the watch list**. The
  previous version called ``classify_payee`` directly and so could not see
  the operator's standing instruction to publish certain vendors by name; it
  reported 25 of those as leaks. That is the lesson from the last round:
  sharing an implementation does not give you a shared decision if the
  callers pass different arguments.
* It reads the withheld list from the database rather than from a literal
  list, so a payee added by a future ingest is covered without anyone
  remembering to add it here.
* It searches every published file, not a sample of them.

It needs a database, because the withheld list is a property of the corpus
rather than of the code. With no connection it SKIPS, and a skip is reported
rather than counted as a pass: a check that did not run is not a check that
succeeded. Treat a skip on this test as a stop condition -- on 2026-09-15 it
skipped, and the artifacts it did not check named hundreds of individuals.
"""

from __future__ import annotations

import os

import pytest
from export_cycle import WATCH_LIST_NORMS
from vendors import WithheldNameIndex, is_exportable

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
def masker() -> WithheldNameIndex:
    """The live corpus's withheld names, indexed exactly as the writers do.

    Returns:
        The index the writers mask with.

    Raises:
        pytest.skip: When no database connection is available.
    """
    try:
        import db

        rows = db.query_dicts(PAYEES_SQL, None)
    except Exception as exc:  # noqa: BLE001 - absence of a database is not a failure here
        pytest.skip(f"no database connection, so the leak check did not run: {type(exc).__name__}: {exc}")
    withheld, published = [], []
    for row in rows:
        name = row["display_name"]
        exportable = is_exportable(name, WATCH_LIST_NORMS, bool(row["payroll_handwrite"]))
        (published if exportable else withheld).append(name)
    return WithheldNameIndex(withheld, published)


def test_there_are_files_to_check():
    """A green run over zero files is not evidence of anything."""
    assert _published_files(), "no sample or export files found; regenerate them before trusting this test"


def test_no_published_file_contains_a_withheld_name(masker):
    """HARD: the control the whole classifier exists to enforce.

    Uses the writers' own index, so a name written with the page's own
    spacing is caught here too rather than only in the masker, and a name
    that merely sits inside a longer published payee name -- ``ANIXTER``
    inside ``Anixter Inc``, one company spelled two ways -- is not reported
    as a leak it is not.
    """
    assert len(masker), "the withheld list is empty; the classifier or the query is wrong"
    leaks: list[str] = []
    for path in _published_files():
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        leaks.extend(f"{os.path.basename(path)}: {name}" for _, _, name in masker.occurrences(text))
    assert not leaks, "withheld payee names found in published files:\n" + "\n".join(sorted(set(leaks))[:40])
