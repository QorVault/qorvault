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


def is_exportable(raw: str | None, watch_list_norms: frozenset[str] | None = None) -> bool:
    """Whether a vendor name may appear in a published export.

    The rule is an **allow-list**, not a block-list, and that direction is
    the whole point. A block-list has to recognise every personal name to be
    safe; this only has to recognise organizations. A small business that
    names no legal form is withheld from the export as a side effect -- the
    row stays in the table and stays queryable, which is the right way round
    for a privacy control.

    Args:
        raw: Vendor name exactly as printed.
        watch_list_norms: Normalized names the operator has explicitly
            approved for publication.

    Returns:
        True when the name may be published.
    """
    if not raw:
        return False
    if watch_list_norms and normalize_vendor(raw) in watch_list_norms:
        return True
    return is_organization(raw)
