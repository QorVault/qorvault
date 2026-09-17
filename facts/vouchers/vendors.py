"""Deterministic vendor-name normalization.

Every rule here was derived from measuring 51,338 vendor strings (3,527
distinct) drawn from the parsed listings in Phase 0, and each is recorded
with the count that justified it. Matching is **exact and case-insensitive
only**. There is no fuzzy matching, no edit distance, no dictionary and no
LLM anywhere in this module: a rule that merges two vendors cannot be undone
once the rows are written, and in a financial table a wrong merge is a
wrong answer about who was paid.

What the rules do:

* trim and collapse internal whitespace -- 0 names need it today, but it
  costs nothing and guards against a less tidy extractor later;
* strip trailing ``.``, ``,``, ``-`` and ``&`` -- 8 names need it;
* casefold for the key -- this merges exactly **11** pairs, such as
  ``AMAZON CAPITAL SERVICES`` and ``Amazon Capital Services``, and nothing
  else. The raw spelling is kept for display so ``KCDA`` does not become
  ``Kcda``.

What the rules deliberately do **not** do:

* strip corporate suffixes. 890 of 3,527 names carry one, and ``Smith Inc``
  and ``Smith LLC`` can be different legal entities;
* expand abbreviations. The corpus is full of ``Svc``, ``Svcs``, ``Sys``,
  ``Sol``, ``Publ``, ``Prod``, ``Mgmt``, ``Intl``, ``Lrning Diff``. Every
  expansion is a guess.
"""

from __future__ import annotations

import functools
import os
import re

WHITESPACE_RX = re.compile(r"\s+")
TRAILING_PUNCT = ".,-&; "

# "Harrow, Jason Christopher", "Okonkwo, Justin W", "Ashby, DeVona L".
# 654 of 3,527 distinct names, 18.5%. These are refund and reimbursement
# payees -- real people being paid back by their school district.
#
# The surname part is at most two tokens ("Van Sielen, Mary"), which is what
# keeps "Hearing, Speech & Deafness Ctr" and "NWAP, Inc" out: both matched
# an earlier, looser rule and were withheld from the export as though they
# were individuals.
PERSON_COMMA_RX = re.compile(r"^[A-Z][A-Za-z''\-]+(?:\s+[A-Z][A-Za-z''\-]+)?,\s+[A-Z][A-Za-z.''\-]*\s*[A-Za-z.''\-]*$")

# Names that carry a corporate suffix are organizations, not individuals.
# This is the positive half of the export rule: an allow-list, because no
# deterministic rule separates "Bradley Quorvin" from a two-word company.
CORPORATE_SUFFIX_RX = re.compile(
    r"(?:^|\s)(?:inc|llc|llp|ltd|co|corp|corporation|company|pllc|p\.?c|lp"
    r"|assn|association|dist|district|univ|university|college|school|schs"
    r"|hs|ms|es|foundation|fund|trust|bank|group|partners|svcs?|services?"
    r"|systems?|sys|solutions?|sol|supply|products?|prod|mgmt|management"
    r"|intl|international|academy|center|centre|ctr|institute|inst|clinic"
    r"|hospital|health|dept|department|city|county|state|usa|us|nw|wa)"
    r"\.?$",
    re.I,
)

# Organization names that are a single all-caps token or acronym.
#
# All-caps alone is NOT a safe organization signal in this corpus: it also
# pays individuals in all caps ("(individual payee, name withheld)" on the 2022 ASB listing).
# Only a single token qualifies.
ACRONYM_RX = re.compile(r"^[A-Z0-9&.\-]{2,}$")

# Signals that a name belongs to an organization and cannot belong to a
# "Given Surname" individual. Each is here because it appears in this
# corpus's withheld bucket on a name that is plainly a company or a public
# body -- Puget Sound Energy, City of Kent, Federal Way Public Schools,
# Soos Creek Water & Sewer, Monday.com. None of them can occur in a
# personal name, which is what makes widening the allow-list with them safe.
ORG_WORDS = frozenset(
    """
    energy water sewer power utility utilities telecom wireless communications
    school schools schs district districts sd esd college university univ
    academy institute inst center centre ctr clinic hospital health healthcare
    city county state federal municipal department dept office bureau agency
    authority commission board association assn society soc union
    bank credit insurance financial capital investments press printing publishing
    media news journal broadcasting productions studios
    foods food produce farms farm bakery bakeries dairy beverage catering
    supply supplies equipment materials products manufacturing industries
    services service svc svcs solutions systems technologies technology
    consulting consultants advisors engineering architects architecture arch
    construction contractors builders landscaping plumbing electric electrical
    roofing flooring floors glass concrete paving mechanical hvac
    transport transportation logistics freight delivery couriers
    security safety maintenance janitorial cleaning restoration
    seminars training institute academy learning education educational
    partnership partners group holdings enterprises ventures brothers sons
    hotel lodge inn resort restaurant cafe grill
    pharmacy medical dental vision therapy therapies rehabilitation
    library museum theatre theater arts athletics sports fitness
    payroll deductions benefits retirement trust foundation charities charity
    employees workers staffing recruiting personnel
    """.split()
)

# A token that can only belong to an organization: it carries a digit, a
# dot, an ampersand, or a slash.
NON_PERSONAL_CHAR_RX = re.compile(r"[&/0-9]|\w\.\w")


def normalize_vendor(raw: str | None) -> str:
    """Return the deterministic matching key for a vendor name.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        The normalized key. Empty string only when the input is empty.
    """
    if not raw:
        return ""
    text = WHITESPACE_RX.sub(" ", raw).strip()
    text = text.strip(TRAILING_PUNCT).strip()
    return text.casefold()


def display_name(raw: str | None) -> str:
    """Return the name as it should be shown, unedited.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        Whitespace-normalized name with trailing punctuation removed. Case
        is never changed: ``KCDA`` stays ``KCDA``.
    """
    if not raw:
        return ""
    return WHITESPACE_RX.sub(" ", raw).strip().strip(TRAILING_PUNCT).strip()


def is_person_shaped(raw: str | None) -> bool:
    """Whether a vendor name is in the ``Surname, Given`` personal form.

    This catches 654 of the 3,527 distinct names in the corpus. It does
    **not** catch ``Given Surname`` payees such as ``Bradley Quorvin``,
    because no deterministic rule separates those from a two-word company.
    The export therefore uses :func:`is_exportable` rather than the negation
    of this function.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        True when the name is in the comma-separated personal form.
    """
    if not raw:
        return False
    text = raw.strip()
    if not PERSON_COMMA_RX.match(text):
        return False
    # "NWAP, Inc" and "Smith, LLC" have the comma shape but the part after
    # the comma is a legal form, not a given name. Without this guard they
    # are withheld from the export as though they were individuals.
    _, _, after = text.partition(",")
    return not CORPORATE_SUFFIX_RX.search(after.strip())


def is_organization(raw: str | None) -> bool:
    """Whether a name is positively identifiable as an organization.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        True when the name carries a corporate suffix or is an acronym.
    """
    if not raw:
        return False
    text = display_name(raw)
    if is_person_shaped(text):
        return False
    if ACRONYM_RX.match(text) and len(text.split()) == 1:
        return True
    if NON_PERSONAL_CHAR_RX.search(text):
        return True
    if any(token.strip(".,'-").casefold() in ORG_WORDS for token in text.split()):
        return True
    return bool(CORPORATE_SUFFIX_RX.search(text))


# ------------------------------------------------- the payee classifier --
#
# ONE classifier, used by the export layer and by the sample files. Two
# rules that are meant to be the same rule will drift, and the first anyone
# would learn of the drift is a person's name in a published document.
#
# A payee is published only if it carries a marker from this list. Anything
# else is withheld, including small businesses that name no legal form.
# That asymmetry is deliberate: withholding a company costs a reader some
# context, and publishing a person costs that person their privacy, and
# those are not comparable errors.
#
# Matching is whole-token and case-insensitive. Substring matching would
# publish "Cortez" for carrying "Corp" and "Cochran" for carrying "Co".

# Operator's list, verbatim.
MARKERS_SPECIFIED = (
    "llc",
    "inc",
    "corp",
    "co",
    "ltd",
    "llp",
    "ctr",
    "center",
    "district",
    "dept",
    "school",
    "hs",
    "pta",
    "association",
    "assn",
    "foundation",
    "church",
    "college",
    "university",
    "services",
    "solutions",
)

# Additions, each justified in the close-out report by a count of the
# payees it releases and by the argument that the token cannot occur inside
# a personal name. Two families only:
#
#   * legal forms the specified list happens not to name. "Ltd" is on the
#     list but "Limited" is not; "Corp" is but "Corporation" is not. A payee
#     should not turn on which abbreviation its bookkeeper typed.
#   * public bodies. A unit of government is never a private individual,
#     and this corpus pays a lot of them.
#
# Nothing describing an INDUSTRY is added -- no "energy", "supply",
# "services" beyond the specified word, no "consulting". Those read as
# organization words to a human but none of them is impossible in a
# business that is one person trading under their own name, which is
# exactly the payee this control exists to protect.
MARKERS_ADDED = (
    # legal forms
    "incorporated",
    "corporation",
    "company",
    "limited",
    "pllc",
    "plc",
    "lp",
    "pc",
    # public bodies and the units this district actually pays
    "sd",
    "esd",
    "county",
    "city",
    "state",
    "treasury",
    "authority",
    "commission",
    "bureau",
    "agency",
    "municipality",
    "schools",
    "schs",
    "universities",
    "colleges",
    "districts",
    "departments",
)

BUSINESS_MARKERS = frozenset(MARKERS_SPECIFIED) | frozenset(MARKERS_ADDED)

# Markers that are also ordinary surnames. One, found by running the
# classifier over all 13,291 payees and reading what it newly released:
# "(individual payee, name withheld)" and "(individual payee, name withheld)" are two people, and "Faith Baptist
# Church" is a congregation. A surname-like marker therefore only counts
# when the name is long enough to be a description of an organization
# rather than a person's two-token name -- which is what separates those
# two payees from "Kent Covenant Church" and "Seattle Buddhist Church
# Matsuri Taiko".
#
# This is not a list of "words that look like names". It is a list of words
# that ARE markers on this list AND are attested surnames in this corpus.
# Adding to it requires the same evidence.
SURNAME_LIKE_MARKERS = frozenset({"church"})

# Below this many markable tokens, a surname-like marker is ambiguous, and
# ambiguous is withheld.
SURNAME_LIKE_MIN_TOKENS = 3

# Generational suffixes. Listed so they can be removed before marker
# matching and so the report can say plainly that they are inert: a suffix
# is a fact about a person, and treating one as evidence of a company is
# how "(individual payee, name withheld)" came to be classified as a business.
NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# The description the district prints on a hand-cut payroll cheque. The
# payee on such a row is an employee, whatever the name looks like, so it
# overrides every marker.
PAYROLL_HANDWRITE_RX = re.compile(r"payroll\s+handwrite", re.I)

TOKEN_SPLIT_RX = re.compile(r"[^A-Za-z0-9&]+")

WITHHELD_LABEL = "(name withheld)"

# ------------------------------------------------- the operator allowlist --
#
# A file of exact payee strings the operator has decided are organizations.
# An entry publishes that payee whatever the patterns above conclude, which
# is how a bare acronym like a purchasing co-operative gets published without
# widening a rule that would also release somebody's two-word name.
#
# It is a FILE and not a code constant on purpose: adding a payee is an
# operator decision, it wants to be reviewable on its own in a diff, and it
# must not require editing Python to make.
#
# THE ENTRY IS MATCHED ON normalize_vendor(), the same key this package uses
# everywhere to decide that two printed spellings are one payee. So an entry
# releases exactly one payee identity -- it is not a substring match, not a
# pattern, and it cannot release a second payee that merely resembles it.
# Case and surrounding whitespace do not matter, because they do not matter
# to the corpus either: "AMAZON CAPITAL SERVICES" and "Amazon Capital
# Services" are already the same vendor here.
#
# WHAT IT DOES NOT OVERRIDE: the Payroll Handwrite signal. That is not a
# pattern over the name, it is the district's own evidence on the payee's
# rows that this payee is an employee being handed a cheque. An allowlist
# is meant to release organizations, and a typo that released an employee
# is the exact failure this whole control exists to prevent. This ordering
# is pinned by a test and is flagged for the operator in the session report.
PAYEE_ALLOWLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "payee_allowlist.txt")

ALLOWLIST_COMMENT_PREFIX = "#"


@functools.lru_cache(maxsize=8)
def load_payee_allowlist(path: str = PAYEE_ALLOWLIST_PATH) -> frozenset[str]:
    """Read the operator's allowlist of exact payee strings.

    Blank lines are skipped and a line whose first non-space character is
    ``#`` is a comment. No payee in this corpus begins with ``#``, which was
    measured before the format was chosen.

    A missing file is an empty allowlist, not an error: the file is a
    scaffold the operator fills in, and the classifier has to work before
    they do.

    Args:
        path: Allowlist file. Defaults to the packaged one.

    Returns:
        Normalized payee keys that publish regardless of pattern.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return frozenset()
    names = set()
    for line in lines:
        text = line.strip()
        if not text or text.startswith(ALLOWLIST_COMMENT_PREFIX):
            continue
        key = normalize_vendor(text)
        if key:
            names.add(key)
    return frozenset(names)


def is_allowlisted(raw: str | None, path: str | None = None) -> bool:
    """Whether the operator has allowlisted this exact payee.

    The default path is resolved on each call rather than bound as a
    default argument, so a test that points the module at a temporary file
    is actually followed. A privacy control whose tests silently exercise
    the shipped file instead of the one under test is not being tested.

    Args:
        raw: Vendor name exactly as printed.
        path: Allowlist file. Defaults to the packaged one.

    Returns:
        True when the payee's normalized key is in the allowlist.
    """
    if not raw:
        return False
    return normalize_vendor(raw) in load_payee_allowlist(path or PAYEE_ALLOWLIST_PATH)


def _markable_tokens(raw: str) -> list[str]:
    """Split a payee name into tokens eligible to carry a marker.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        Lower-cased tokens with generational suffixes removed.
    """
    tokens = [t.casefold() for t in TOKEN_SPLIT_RX.split(raw) if t]
    return [t for t in tokens if t not in NAME_SUFFIXES]


def business_marker(raw: str | None) -> str | None:
    """Return the business marker a payee name carries, if any.

    A surname-like marker only counts on a name of at least
    ``SURNAME_LIKE_MIN_TOKENS`` tokens; on a shorter name it is more likely
    to be the payee's surname than their legal form.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        The marker token found, or None.
    """
    if not raw:
        return None
    tokens = _markable_tokens(raw)
    found = [t for t in tokens if t in BUSINESS_MARKERS]
    for token in found:
        if token in SURNAME_LIKE_MARKERS and len(tokens) < SURNAME_LIKE_MIN_TOKENS:
            continue
        return token
    return None


def classify_payee(raw: str | None, has_payroll_handwrite: bool = False) -> tuple[bool, str]:
    """Decide whether a payee name may be published, and say why.

    Args:
        raw: Vendor name exactly as printed.
        has_payroll_handwrite: True when any of this payee's own lines
            carries a "Payroll Handwrite" description.

    Returns:
        ``(publish, reason)``. ``reason`` is a short code suitable for a
        report column: ``allowlist``, ``marker:<token>``,
        ``payroll_handwrite``, ``person_shaped``, ``no_marker`` or
        ``empty``.
    """
    if not raw or not raw.strip():
        return False, "empty"
    if has_payroll_handwrite:
        # Checked before the marker AND before the allowlist. See the
        # comment on PAYEE_ALLOWLIST_PATH: this is the district's own
        # evidence that the payee is an employee, and an allowlist typo
        # must not be able to publish one.
        return False, "payroll_handwrite"
    if is_allowlisted(raw):
        # The operator's explicit, exact decision, so it outranks every
        # pattern below -- including the person-shape guard, because a
        # business really can be printed "Hearing, Speech & Deafness Ctr".
        return True, "allowlist"
    if is_person_shaped(raw):
        # A surname can be a marker word. "(individual payee, name withheld)" is a person
        # named Church, and the first run of this classifier over the corpus
        # published him. is_person_shaped already distinguishes that from
        # "NWAP, Inc" and "Smith, LLC", where the text after the comma is a
        # legal form rather than a given name, so the two survive this guard.
        return False, "person_shaped"
    marker = business_marker(raw)
    if marker:
        return True, f"marker:{marker}"
    return False, "no_marker"


def is_exportable(
    raw: str | None,
    watch_list_norms: frozenset[str] | None = None,
    has_payroll_handwrite: bool = False,
) -> bool:
    """Whether a payee name may appear in a published export or sample.

    The rule is an **allow-list**, not a block-list, and that direction is
    the whole point. A block-list has to recognise every personal name to be
    safe; this only has to recognise organizations. A small business that
    names no legal form is withheld as a side effect -- the row stays in the
    table and stays queryable, which is the right way round for a privacy
    control.

    Args:
        raw: Vendor name exactly as printed.
        watch_list_norms: Normalized names the operator has explicitly
            approved for publication.
        has_payroll_handwrite: True when any of this payee's own lines
            carries a "Payroll Handwrite" description.

    Returns:
        True when the name may be published.
    """
    if not raw:
        return False
    if watch_list_norms and normalize_vendor(raw) in watch_list_norms:
        return True
    return classify_payee(raw, has_payroll_handwrite)[0]


def publishable_name(
    raw: str | None,
    watch_list_norms: frozenset[str] | None = None,
    has_payroll_handwrite: bool = False,
) -> str:
    """Return the payee name to print, or the withheld placeholder.

    The single call every export and sample writer goes through, so that
    "did this file withhold names?" has one answer rather than one per
    writer.

    Args:
        raw: Vendor name exactly as printed.
        watch_list_norms: Normalized names approved for publication.
        has_payroll_handwrite: True when any of this payee's own lines
            carries a "Payroll Handwrite" description.

    Returns:
        The display name, or ``WITHHELD_LABEL``.
    """
    if is_exportable(raw, watch_list_norms, has_payroll_handwrite):
        return display_name(raw)
    return WITHHELD_LABEL


def name_pattern(raw: str | None) -> re.Pattern | None:
    r"""Build the pattern that finds a payee's printed name in free text.

    Tokens are joined with ``\\s+`` because the layout text a locator quote
    is cut from carries the PDF's own spacing, which is not the spacing in
    ``vendor_raw``: "Hearing,  Speech &   Deafness Ctr" and "Hearing, Speech
    & Deafness Ctr" are the same name on the same page.

    Args:
        raw: Vendor name exactly as printed.

    Returns:
        A compiled case-insensitive pattern, or None for an empty name.
    """
    tokens = [re.escape(t) for t in display_name(raw).split()]
    if not tokens:
        return None
    return re.compile(r"\s+".join(tokens), re.I)


def redact_name(text: str | None, raw: str | None) -> str:
    """Remove a withheld payee's name from a verbatim quote.

    A locator quote is the source line copied from the page, so it carries
    the payee's name whether or not the vendor column was withheld. Printing
    the quote unredacted beside a withheld vendor column would publish the
    name anyway, one row to the right.

    Where the name cannot be located in the text -- a hyphenation or a
    pdfplumber split inside the name -- the **whole quote** is dropped
    rather than published on the assumption that the name is not in it.

    Args:
        text: The quote as extracted.
        raw: The payee name to remove.

    Returns:
        The quote with the name replaced, or a placeholder when it could
        not be removed safely.
    """
    if not text:
        return ""
    pattern = name_pattern(raw)
    if pattern is None:
        return text
    redacted, count = pattern.subn(WITHHELD_LABEL, text)
    if count:
        return redacted
    # The name is not findable as printed. It may still be in there, split
    # or hyphenated, so the quote does not go out.
    return f"{WITHHELD_LABEL} — quote suppressed: the payee name could not be located to remove it"
