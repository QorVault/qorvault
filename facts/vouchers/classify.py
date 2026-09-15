"""Classification of voucher artifacts by fund and document class.

Two independent signals are available and both are used:

* the **title** (``documents.title``, or the file name on disk), and
* the **first-page phrase** printed by the accounting system
  ("General Fund Warrants", "ACH Payments", "Capital Projects Fund
  Warrants").

Titles are operator-typed and inconsistent across twenty years -- the same
fund appears as ``GF``, ``General Fund``, ``General_Fund`` and
``General fund``, and the same artifact as ``Warrant_Recap.pdf``,
``WARRANT_RECAP.pdf`` and ``Warrant+Recap.pdf``. The printed phrase is
machine-generated and therefore authoritative where the two disagree; the
title is what is available when there is no text layer.

Every rule here is a literal regex over a closed vocabulary. No LLM is
involved in classifying a fund, a document class, or anything downstream
of them.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------ excluded --
#
# The words "warrant" and "board meeting" are load-bearing elsewhere in the
# corpus. Excluding these by name is deliberate: a silent filter that drops
# 216 transcripts would look identical to a filter that drops 216 voucher
# sets, and Phase 0's job is to be able to tell the difference.

EXCLUDE_RULES: tuple[tuple[str, str], ...] = (
    ("meeting_transcript", r"\bTranscript\b"),
    ("meeting_minutes", r"Board\s+Meeting\s+Minutes"),
    # "Attachment for 20240208 Board Meeting - NS Equipment 2.pdf"
    ("meeting_attachment", r"^Attachment\s+for\b|\bfor\s+\d{6,8}\s+Board\s+Meeting\b"),
    # "2024-25 Device Warranties & Services", "Statutory Warranty Deed"
    ("warranty_not_warrant", r"Warrant(?:y|ies)(?![a-z])"),
    ("audit_report", r"Audit\s+of\s+Expenditures"),
    ("policy_or_contract", r"\bEmployment\b|Capital\s+Facilities\s+Plan|\bContract\b"),
    ("donation_listing", r"^Donations\b|\bDonations\s+for\b"),
)

EXCLUDE_COMPILED = tuple((name, re.compile(rx, re.I)) for name, rx in EXCLUDE_RULES)

# --------------------------------------------------------------- funds --
#
# The build task names five funds (GF, ACH, Capital, ASB, Trust). The corpus
# holds four more that pay real district money -- Transportation Vehicle
# Fund, Custodial, Permanent and a short-lived "Vision Trust". They are
# classified here rather than swept into "other", so Phase 0 can report how
# much money a five-fund vocabulary would make invisible.
#
# ACH is a payment METHOD, not a fund: Phase 0 proved the 2026-03-25 ACH
# listing total equals the sum of the accounts-payable direct-deposit lines
# across General, Capital, ASB and Custodial in the signed register. It is
# modelled as a fund here only because the district publishes it as its own
# listing with its own TOTAL.

FUND_TITLE_RULES: tuple[tuple[str, str], ...] = (
    # ACH first: an ACH file never also names another fund, and "ACH Funds
    # Vouchers" would otherwise be caught by a generic "Fund" rule.
    ("ACH", r"\bACH\b"),
    # "Associate Student Body" (sic) appears on 2026-01-14; the corpus is
    # typed by hand and the typos are part of the data.
    ("ASB", r"\bASB\b|Associated?\s+Student\s+Body"),
    # Capital before Trust/GF: "Capital Projects Fund" contains "Fund".
    # "Captial" (sic) appears on 2020-12-08.
    ("Capital", r"\bCapital\b|\bCaptial\b|\bCPF\b|\bCP\b(?=\s*(?:Vouchers?|\d))"),
    ("Transportation", r"Transport|Transporation|\bTVF\b|\bTR\b(?=\s*Vouchers?)"),
    ("Custodial", r"\bCustodial\b"),
    ("Permanent", r"\bPermanent\b"),
    ("Trust", r"\bTrust\b"),
    ("GF", r"\bGF\b|General[_ ]?[Ff]und|\bGeneral\s+[Ff]und\b"),
)

# Phrases the accounting system prints at the head of a fund's listing.
# These are the authoritative signal where they exist.
FUND_PHRASE_RULES: tuple[tuple[str, str], ...] = (
    ("ACH", r"ACH\s+(?:Payments?|Vouchers?|Warrants?)"),
    ("ASB", r"(?:Associated\s+Student\s+Body|ASB)\s+Fund\s+Warrants?"),
    ("Capital", r"Capital\s+Projects?\s+Fund\s+Warrants?"),
    ("Transportation", r"Transportation\s+Vehicle\s+Fund\s+Warrants?"),
    ("Custodial", r"Custodial\s+Fund\s+Warrants?"),
    ("Permanent", r"Permanent\s+Funds?\s+Warrants?"),
    ("Trust", r"(?:Private\s+Purpose\s+)?Trust\s+Fund\s+Warrants?"),
    ("GF", r"General\s+Fund\s+Warrants?"),
)

SEPARATOR_RX = re.compile(r"[_+.\-]+")
WHITESPACE_RX = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    r"""Turn file-name separators into spaces before pattern matching.

    Titles in this corpus use ``_``, ``+``, ``.`` and ``-`` interchangeably
    as word separators: ``Warrant_Recap.pdf``, ``Warrant+Recap.pdf``,
    ``Warrant Recap 3-8-17.pdf``. Regex ``\b`` does **not** fire between a
    letter and an underscore -- both are word characters -- so a pattern
    like ``Warrant[_ ]*Recap\b`` silently fails on ``Warrant_Recap_3-8-17``
    and the artifact falls through unclassified. Normalizing first removes
    that entire class of near-miss.

    The file extension is kept, because ``.docx`` and ``.xlsx`` are a real
    signal that an artifact is a resolution rather than a payment listing.

    Args:
        title: Raw title or file name.

    Returns:
        Title with separators collapsed to single spaces.
    """
    stem, dot, ext = title.rpartition(".")
    if dot and len(ext) <= 5 and ext.isalnum():
        body, suffix = stem, f".{ext.lower()}"
    else:
        body, suffix = title, ""
    body = SEPARATOR_RX.sub(" ", body)
    return WHITESPACE_RX.sub(" ", body).strip() + suffix


FUNDS_IN_SCOPE: tuple[str, ...] = ("GF", "ACH", "Capital", "ASB", "Trust")
FUNDS_EXTRA: tuple[str, ...] = ("Transportation", "Custodial", "Permanent")
ALL_FUNDS: tuple[str, ...] = FUNDS_IN_SCOPE + FUNDS_EXTRA

# ------------------------------------------------------ document class --
#
# A voucher night produces several different artifacts and they are not
# interchangeable evidence. Conflating the signed register (the board's own
# approval record, certified under penalty of perjury) with the detail
# listing (the accounting system's export) would destroy the only
# independent cross-check available.

DOC_CLASS_RULES: tuple[tuple[str, str], ...] = (
    # Signed warrant register: "BDMTG - 6-24-2026 SIGNED.pdf",
    # "Board Mtg 03-25-26 SIGNED.pdf", "10-26-22 BOARD MTG SIGNED.pdf".
    (
        "warrant_register",
        r"\bBD ?MTG\b|\bBOARD MTG\b|\bBoard Mtg\b|\bBoard Summary\b",
    ),
    # Cancellation / stale warrant resolutions: an accounting correction,
    # not a payment listing.
    (
        "warrant_cancellation",
        r"Cancel(?:l)?ations?\s+of\s+Warrants|Cancel\s+Stale|Stale\s*Warrants?"
        r"|Outstanding\s*Warrants?|Warrants\s+Outstanding|OldWarrants"
        r"|Warrants?\s+for\s+Cancel",
    ),
    # Fund-level recap or summary: totals only, no vendor rows. The
    # pre-2018 era publishes only this.
    ("warrant_recap", r"(?:Warrants?|Vouchers?)\s*(?:Recap|Summary|Request|List)(?![a-z])"),
    # Year-to-date listing sorted by vendor rather than by check. A
    # different view of overlapping data -- never a monthly set.
    ("voucher_by_vendor", r"Vendor\s*(?:Rpt|Report)"),
    # The monthly per-check listing this build exists to parse.
    ("detail_listing", r"Voucher"),
    # Bare fund-and-date file names: "General_Fund.pdf", "Trust 1-22-20.pdf",
    # "Capital Projects 3-11-20.pdf". Confirmed against the printed header.
    (
        "detail_listing",
        r"^(?:General Fund|Capital Projects?|Trust|ASB|ACH|TVF)(?:\s*\d|\.pdf$)",
    ),
)

DOC_CLASS_COMPILED = tuple((c, re.compile(rx, re.I)) for c, rx in DOC_CLASS_RULES)

# Classes that carry per-check vendor rows and therefore become voucher sets.
PARSEABLE_CLASSES: tuple[str, ...] = ("detail_listing",)

PARSEABLE_SUFFIXES: tuple[str, ...] = (".pdf",)


def excluded_reason(title: str | None) -> str | None:
    """Return why a title is not a voucher artifact, or None if it may be.

    Args:
        title: ``documents.title`` or a file name.

    Returns:
        An exclusion reason code, or None.
    """
    if not title:
        return "empty_title"
    normalized = normalize_title(title)
    for name, rx in EXCLUDE_COMPILED:
        if rx.search(normalized):
            return name
    return None


def fund_from_title(title: str | None) -> str | None:
    """Classify a fund from an artifact title or file name.

    Args:
        title: ``documents.title`` or a file name.

    Returns:
        A fund code, or None when the title names no fund.
    """
    if not title:
        return None
    normalized = normalize_title(title)
    for fund, pattern in FUND_TITLE_RULES:
        if re.search(pattern, normalized, re.I):
            return fund
    return None


def fund_from_text(text: str | None, window: int = 4000) -> str | None:
    """Classify a fund from the phrase the accounting system prints.

    Args:
        text: Extracted document text; only the head is inspected.
        window: How many leading characters to search.

    Returns:
        A fund code, or None when no known phrase appears.
    """
    if not text:
        return None
    head = text[:window]
    for fund, pattern in FUND_PHRASE_RULES:
        if re.search(pattern, head, re.I):
            return fund
    return None


def doc_class_from_title(title: str | None) -> str | None:
    """Classify the kind of voucher artifact from its title.

    Args:
        title: ``documents.title`` or a file name.

    Returns:
        One of ``warrant_register``, ``warrant_cancellation``,
        ``warrant_recap``, ``voucher_by_vendor``, ``detail_listing``,
        or None when the title matches no rule.
    """
    if not title:
        return None
    normalized = normalize_title(title)
    for doc_class, rx in DOC_CLASS_COMPILED:
        if rx.search(normalized):
            return doc_class
    return None


def fiscal_year(meeting_date: str | None) -> str | None:
    """Return the Washington school fiscal year for a meeting date.

    Washington school district fiscal years run 1 September to 31 August,
    so a voucher night in September opens a new fiscal year.

    Args:
        meeting_date: ISO date string.

    Returns:
        Fiscal year label such as ``FY2026`` (1 Sep 2025 - 31 Aug 2026),
        or None when the date is unknown.
    """
    if not meeting_date or len(meeting_date) < 7:
        return None
    year, month = int(meeting_date[:4]), int(meeting_date[5:7])
    return f"FY{year + 1}" if month >= 9 else f"FY{year}"


def era_band(meeting_date: str | None) -> str | None:
    """Return the five-year band a meeting date falls in.

    Args:
        meeting_date: ISO date string.

    Returns:
        A band label such as ``2015-2019``, or None.
    """
    if not meeting_date or len(meeting_date) < 4:
        return None
    year = int(meeting_date[:4])
    start = year - (year - 2005) % 5
    return f"{start}-{start + 4}"
