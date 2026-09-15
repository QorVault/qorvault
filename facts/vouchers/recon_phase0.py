"""Phase 0 reconnaissance for the voucher fact tables.

Read-only against ``documents``; read-only against the corpus on disk. This
script writes nothing except its own cache and its JSON findings.

It answers the five Phase 0 questions:

1. What voucher documents exist, by fiscal year and fund, and which voucher
   nights have an agenda item but no PDF.
2. How many resolve to a file on disk, and which of those carry a text
   layer -- a full pass over every candidate, not a sample.
3. What format eras exist, dated, with one locator each.
4. For how many meetings a signed warrant register exists alongside the
   detail listing, and whether the register is independent evidence.
5. What vendor-name normalization rules the data actually requires.

No LLM is involved in any amount, date, vendor, check-number or total path.
Every number below comes from regex and arithmetic over extracted PDF text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

import census
import db
from classify import ALL_FUNDS, era_band, fund_from_text
from locators import PdfText

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build", "recon_cache.json")

# Artifact classes worth opening. Cancellation resolutions are excluded: they
# are an accounting correction with no vendor rows and no fund total.
PROBE_CLASSES = frozenset({"detail_listing", "warrant_register", "warrant_recap", "voucher_by_vendor"})

# -------------------------------------------------------------- patterns --

# A TOTAL line is matched only as a whole line. A vendor whose name begins
# "Total" (the corpus contains "Total Technology", a real company) is a data
# row, and treating it as a total would silently truncate a set.
TOTAL_LINE_RX = re.compile(
    r"^[ \t]*(?P<label>(?:GRAND[ \t]+)?TOTAL(?:[ \t]+[A-Z&' ]+?)?)"
    r"[ \t]*\$?[ \t]*(?P<amount>-?[\d, ]+\.\d{2})[ \t]*$",
    re.M,
)

# The starting-point row regex exactly as written in the build brief. Kept
# so the report can state what it does and does not catch, and pinned by a
# test -- not used for any measurement.
ROW_RX_BRIEF = re.compile(
    r"^\s*(?P<vendor>.+?)\s{1,}"
    r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<chk>\d{5,11})\s+"
    r"(?P<chkamt>-?[\d,]*\.?\d+)\s+"
    r"(?P<invamt>-?[\d,]*\.?\d+)\s*"
    r"(?P<desc>.*)$"
)

# The brief's regex with two changes, each forced by a measured failure on
# real 2026 documents rather than by anticipation:
#
#  1. ``\s{1,}`` between vendor and date becomes ``\s*``. On the 2026-03-25
#     ASB listing the vendor "THE HEATHMAN LODGE AND HUDSONS BAR AN" runs
#     straight into its date with no gap at all: "...BAR AN03 /12/2026".
#  2. The date may carry a pdfplumber-injected space ("03 /12/2026"), and so
#     may an amount ("$ 3 ,609,064.78"). Both are layout artifacts, not
#     content, and are stripped after capture.
#
# Together these two changes are the whole of the $1,322.35 shortfall the
# brief records against 2026-03-25 ASB. With them the set reconciles to the
# cent; without them it is short by exactly that one row.
ROW_RX = re.compile(
    r"^\s*(?P<vendor>.+?)\s*"
    r"(?P<date>\d{1,2}\s?/\s?\d{1,2}\s?/\s?\d{4})\s+"
    r"(?P<chk>\d{5,11})\s+"
    r"(?P<chkamt>-?[\d, ]*\.?\d+)\s+"
    r"(?P<invamt>-?[\d, ]*\.?\d+)\s*"
    r"(?P<desc>.*)$"
)

# Bumped whenever ``probe`` starts recording something new, so a cached
# entry from an earlier shape is re-probed instead of silently reported.
PROBE_VERSION = 2

# "General Fund Warrants 02/06/26 through 03/12/26 and P-Cards 01/17/26
# through 02/28/26"
PERIOD_RX = re.compile(
    r"(?P<d1>\d{1,2}/\d{1,2}/\d{2,4})\s+through\s+(?P<d2>\d{1,2}/\d{1,2}/\d{2,4})",
    re.I,
)
PCARD_RX = re.compile(
    r"P-?Cards?\s+(?P<d1>\d{1,2}/\d{1,2}/\d{2,4})\s+through\s+(?P<d2>\d{1,2}/\d{1,2}/\d{2,4})",
    re.I,
)

# The column header line identifies a layout era more reliably than a date
# does: the district changed report generators without changing file names.
HEADER_TOKENS = ("Vendor", "Check", "Invoice", "Description", "Amount", "Number", "Date")


def money(raw: str) -> Decimal | None:
    """Parse a printed amount into an exact Decimal.

    This is the Phase 0 version and is deliberately left as it was. The
    canonical parser is ``parsers.money``, which adds a thousands-grouping
    check. Changing this one would silently alter the figures in
    ``reports/facts-vouchers-recon-2026-09-14.md``, which is the record of
    what Phase 0 actually measured.

    Args:
        raw: Amount as printed, possibly with commas or stray spaces.

    Returns:
        Exact decimal value, or None when the text is not a number.
    """
    cleaned = raw.replace(",", "").replace(" ", "").replace("$", "")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def header_signature(page_text: str) -> str:
    """Summarize the column header of a listing page.

    Args:
        page_text: Text of the first page of a listing.

    Returns:
        A compact signature such as ``Vendor|Check Date|Check Number|...``,
        or ``(none)`` when no header line is present.
    """
    for raw in page_text.split("\n")[:40]:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        hits = sum(1 for token in HEADER_TOKENS if token in line)
        if hits >= 3:
            return re.sub(r"\s{2,}", "|", raw.strip())
    return "(none)"


def probe(path: str) -> dict:
    """Extract everything Phase 0 needs from one voucher PDF.

    Args:
        path: Filesystem path to a PDF.

    Returns:
        A findings dict. ``error`` is set when the PDF could not be read.
    """
    try:
        pdf = PdfText(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt PDF is data, not a crash
        return {"error": f"{type(exc).__name__}: {exc}"}

    head = pdf.pages[0].text if pdf.pages else ""
    first_lines = [re.sub(r"\s+", " ", ln).strip() for ln in head.split("\n")]
    first_lines = [ln for ln in first_lines if ln][:6]

    totals = []
    for match in TOTAL_LINE_RX.finditer(pdf.text):
        amount = money(match.group("amount"))
        totals.append(
            {
                "label": re.sub(r"\s+", " ", match.group("label")).strip(),
                "amount": str(amount) if amount is not None else None,
                "page": pdf.page_for_offset(match.start()),
                "offset": match.start(),
                "quote": re.sub(r"\s+", " ", match.group(0)).strip(),
            }
        )

    # Rows, with their offsets, so "the first TOTAL after the last data row"
    # can be evaluated rather than assumed.
    rows = []
    offset = 0
    last_row_end = 0
    for line in pdf.text.split("\n"):
        match = ROW_RX.match(line)
        if match:
            rows.append(match.groupdict())
            last_row_end = offset + len(line)
        offset += len(line) + 1

    # Three control totals, all recorded. The dollar total is the one the
    # printed TOTAL is expected to equal; the other two are independent
    # checks on the same rows.
    sum_invoice = Decimal("0")
    check_amounts: dict[str, Decimal] = {}
    hash_total = 0
    for row in rows:
        invoice = money(row["invamt"])
        if invoice is not None:
            sum_invoice += invoice
        number = row["chk"].strip()
        if number not in check_amounts:
            amount = money(row["chkamt"])
            check_amounts[number] = amount if amount is not None else Decimal("0")
            if number.isdigit():
                hash_total += int(number)
    sum_check = sum(check_amounts.values(), Decimal("0"))

    # The stated total is the first whole-line TOTAL at or after the last
    # data row. Trailing TOTAL-only pages -- the 2026-05-27 Transportation
    # listing prints a second TOTAL on page 5 at exactly twice the real
    # figure -- are recorded but do not win.
    chosen = None
    rule = None
    for total in totals:
        if total["offset"] >= last_row_end:
            chosen = total
            rule = "first_total_after_last_row"
            break
    if chosen is None and totals:
        chosen = totals[-1]
        rule = "last_total_no_row_anchor"

    stated = money(chosen["amount"]) if chosen and chosen["amount"] else None
    delta = (sum_invoice - stated) if stated is not None else None

    period = PERIOD_RX.search(head)
    pcard = PCARD_RX.search(head)
    # The P-card clause also matches the generic period pattern, so the
    # first "X through Y" is the warrant period only when it is not the
    # P-card one.
    if period and pcard and period.start() >= pcard.start():
        period = None

    return {
        "_v": PROBE_VERSION,
        "pages": pdf.page_count,
        "chars": len(pdf.text),
        "has_text_layer": pdf.has_text_layer,
        "first_lines": first_lines,
        "fund_from_text": fund_from_text(head),
        "header_signature": header_signature(head),
        "period": [period.group("d1"), period.group("d2")] if period else None,
        "pcard_period": [pcard.group("d1"), pcard.group("d2")] if pcard else None,
        "totals": totals,
        "total_count": len(totals),
        "chosen_total": chosen,
        "chosen_total_rule": rule,
        "stated_total": str(stated) if stated is not None else None,
        "row_count": len(rows),
        "check_count": len(check_amounts),
        "sum_invoice": str(sum_invoice),
        "sum_check_dedup": str(sum_check),
        "hash_total": hash_total,
        "delta": str(delta) if delta is not None else None,
        "reconciles": bool(stated is not None and rows and delta == 0),
        "invoice_equals_check": sum_invoice == sum_check,
        "sample_rows": rows[:12],
        "sample_vendors": [r["vendor"].strip() for r in rows[:400]],
    }


def load_cache() -> dict:
    """Read the probe cache, or an empty cache if none exists.

    Returns:
        Mapping of SHA-256 digest to probe findings.
    """
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(cache: dict) -> None:
    """Write the probe cache.

    Args:
        cache: Mapping of SHA-256 digest to probe findings.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = f"{CACHE_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(cache, handle)
    os.replace(tmp, CACHE_PATH)


def build_inventory() -> list:
    """Assemble the merged artifact inventory.

    Returns:
        Artifacts, deduplicated by content, classified.
    """
    from_db = census.db_artifacts(db.query_dicts)
    from_disk = census.disk_artifacts()
    return census.merge(from_db, from_disk)


def main() -> None:
    """Run Phase 0 recon and print JSON findings to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="skip PDF extraction; inventory only",
    )
    parser.add_argument("--json-out", help="also write findings to this path")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="report probe progress on stderr",
    )
    args = parser.parse_args()

    artifacts = build_inventory()
    agenda_items = census.voucher_agenda_items(db.query_dicts)
    kept = [a for a in artifacts if a.exclusion is None]

    findings: dict = {}

    # ------------------------------------------------------ 1. inventory --
    by_class = Counter(a.doc_class for a in kept)
    detail = [a for a in kept if a.doc_class == "detail_listing"]
    register = [a for a in kept if a.doc_class == "warrant_register"]

    fy_fund: dict[str, Counter] = defaultdict(Counter)
    for art in detail:
        fy_fund[art.fiscal_year or "unknown"][art.fund or "unknown"] += 1

    findings["inventory"] = {
        "artifacts_unique_content": len(artifacts),
        "kept": len(kept),
        "excluded": Counter(a.exclusion for a in artifacts if a.exclusion),
        "excluded_titles": sorted({a.title for a in artifacts if a.exclusion == "unclassified"}),
        "by_doc_class": by_class,
        "detail_by_fiscal_year_fund": {k: dict(v) for k, v in sorted(fy_fund.items())},
        "detail_by_fund": Counter(a.fund for a in detail),
        "earliest_detail": min((a.meeting_date for a in detail if a.meeting_date), default=None),
        "latest_detail": max((a.meeting_date for a in detail if a.meeting_date), default=None),
        "earliest_any": min((a.meeting_date for a in kept if a.meeting_date), default=None),
        "latest_any": max((a.meeting_date for a in kept if a.meeting_date), default=None),
        "origin": Counter(a.origin for a in kept),
    }

    # --------------------------------------------- completeness check --
    #
    # A filter that silently drops a voucher set looks exactly like one that
    # drops a transcript. This re-walks the corpus and proves, by content
    # hash, that every PDF filed under a voucher agenda item is either in the
    # inventory or is a byte-identical duplicate of something that is.
    have_hash = {a.sha256 for a in artifacts if a.sha256}
    have_path = {os.path.realpath(a.resolved_path) for a in artifacts if a.resolved_path}
    duplicates = 0
    absent: list[str] = []
    for root in census.CORPUS_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if not census.VOUCHER_ITEM_DIR_RX.search(os.path.basename(dirpath)):
                continue
            for name in filenames:
                if not name.lower().endswith(".pdf"):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.realpath(path) in have_path:
                    continue
                if census.sha256_of(path) in have_hash:
                    duplicates += 1
                else:
                    absent.append(path)
    findings["coverage_check"] = {
        "voucher_dir_pdfs_not_in_inventory": duplicates + len(absent),
        "content_duplicates": duplicates,
        "truly_absent": len(absent),
        "truly_absent_paths": absent[:40],
    }

    # Voucher nights on the agenda with no listing of any kind.
    dates_with_detail = {a.meeting_date for a in detail if a.meeting_date}
    dates_with_any = {
        a.meeting_date for a in kept if a.meeting_date and a.doc_class in {"detail_listing", "warrant_recap"}
    }
    agenda_dates = sorted({r["meeting_date"] for r in agenda_items if r["meeting_date"]})
    findings["agenda_items"] = {
        "voucher_agenda_items": len(agenda_items),
        "distinct_meeting_dates": len(agenda_dates),
        "earliest": agenda_dates[0] if agenda_dates else None,
        "latest": agenda_dates[-1] if agenda_dates else None,
        "agenda_but_no_detail": sorted(set(agenda_dates) - dates_with_detail),
        "agenda_but_no_listing_at_all": sorted(set(agenda_dates) - dates_with_any),
    }

    # -------------------------------------------- 2. resolution + text --
    #
    # The recaps are probed too, even though they carry no vendor rows. They
    # are the ONLY voucher artifact the corpus holds for 2010-2016, so a
    # text-layer answer that skipped them would report on seven years the
    # question was never asked about.
    parseable = [a for a in kept if a.doc_class in PROBE_CLASSES]
    resolved = [a for a in parseable if a.resolved_path]
    findings["resolution"] = {
        "parseable_artifacts": len(parseable),
        "resolved_to_disk": len(resolved),
        "unresolved": len(parseable) - len(resolved),
        "unresolved_examples": [
            {"date": a.meeting_date, "title": a.title, "stored_path": a.file_path}
            for a in parseable
            if not a.resolved_path
        ][:25],
        "db_rows_with_document_id": sum(1 for a in resolved if a.document_id),
        "disk_only_no_document_id": sum(1 for a in resolved if not a.document_id),
    }

    if args.no_probe:
        print(json.dumps(findings, indent=1, default=str))
        return

    # ----------------------------------------------- full extraction pass --
    cache = load_cache()
    probes: dict[str, dict] = {}
    for index, art in enumerate(resolved, start=1):
        key = art.sha256 or art.resolved_path or ""
        stale = cache.get(key, {}).get("_v") != PROBE_VERSION and "error" not in cache.get(key, {})
        if key not in cache or stale:
            cache[key] = probe(art.resolved_path)
            if index % 25 == 0:
                save_cache(cache)
        probes[key] = cache[key]
        if args.progress and index % 25 == 0:
            print(f"  probed {index}/{len(resolved)}", file=sys.stderr)
    save_cache(cache)

    def probe_of(art) -> dict:
        return probes.get(art.sha256 or art.resolved_path or "", {})

    text_by_year: dict[str, Counter] = defaultdict(Counter)
    for art in resolved:
        info = probe_of(art)
        year = (art.meeting_date or "????")[:4]
        if info.get("error"):
            text_by_year[year]["unreadable"] += 1
        elif info.get("has_text_layer"):
            text_by_year[year]["text"] += 1
        else:
            text_by_year[year]["image_only"] += 1
    findings["text_layer_by_year"] = {k: dict(v) for k, v in sorted(text_by_year.items())}

    since_2015 = [a for a in resolved if (a.meeting_date or "") >= "2015-01-01"]
    with_text = sum(1 for a in since_2015 if probe_of(a).get("has_text_layer"))
    findings["text_layer_since_2015"] = {
        "resolved": len(since_2015),
        "with_text": with_text,
        "pct": round(100.0 * with_text / len(since_2015), 1) if since_2015 else None,
    }

    # --------------------------------------------------- 3. format eras --
    era_groups: dict[str, list] = defaultdict(list)
    for art in detail:
        info = probe_of(art)
        if not info or info.get("error") or not info.get("has_text_layer"):
            continue
        era_groups[info.get("header_signature", "(none)")].append(art)

    eras = []
    for signature, members in sorted(era_groups.items(), key=lambda kv: -len(kv[1])):
        dates = sorted(a.meeting_date for a in members if a.meeting_date)
        exemplar = members[0]
        info = probe_of(exemplar)
        eras.append(
            {
                "header_signature": signature,
                "n": len(members),
                "first": dates[0] if dates else None,
                "last": dates[-1] if dates else None,
                "funds": dict(Counter(a.fund for a in members)),
                "locator": {
                    "title": exemplar.title,
                    "meeting_date": exemplar.meeting_date,
                    "document_id": exemplar.document_id,
                    "path": exemplar.resolved_path,
                    "page": 1,
                    "quote": (info.get("first_lines") or [""])[0],
                },
                "rows_in_exemplar": info.get("row_count"),
                "totals_in_exemplar": len(info.get("totals") or []),
            }
        )
    findings["format_eras"] = eras

    # ------------------------------------- reconciliation feasibility --
    #
    # The single number that decides whether Phase 1 is a build or a
    # research project: of the detail listings that have a text layer, how
    # many already reconcile to their own printed TOTAL, to the cent,
    # using nothing but the row regex and arithmetic.
    def recon_bucket(art) -> str:
        info = probe_of(art)
        if info.get("error"):
            return "unreadable"
        if not info.get("has_text_layer"):
            return "no_text_layer"
        if not info.get("row_count"):
            return "regex_miss"
        if info.get("stated_total") is None:
            return "total_not_found"
        if info.get("reconciles"):
            return "reconciles"
        return "out_of_balance"

    by_year_recon: dict[str, Counter] = defaultdict(Counter)
    by_fund_recon: dict[str, Counter] = defaultdict(Counter)
    by_era_recon: dict[str, Counter] = defaultdict(Counter)
    for art in detail:
        bucket = recon_bucket(art)
        by_year_recon[(art.meeting_date or "????")[:4]][bucket] += 1
        by_fund_recon[art.fund or "unknown"][bucket] += 1
        by_era_recon[probe_of(art).get("header_signature") or "(none)"][bucket] += 1

    overall = Counter(recon_bucket(a) for a in detail)
    findings["reconciliation_feasibility"] = {
        "overall": dict(overall),
        "detail_listings": len(detail),
        "by_year": {k: dict(v) for k, v in sorted(by_year_recon.items())},
        "by_fund": {k: dict(v) for k, v in sorted(by_fund_recon.items())},
        "out_of_balance_examples": [
            {
                "date": a.meeting_date,
                "fund": a.fund,
                "title": a.title,
                "stated": probe_of(a).get("stated_total"),
                "parsed": probe_of(a).get("sum_invoice"),
                "delta": probe_of(a).get("delta"),
                "rows": probe_of(a).get("row_count"),
                "totals_seen": probe_of(a).get("total_count"),
            }
            for a in detail
            if recon_bucket(a) == "out_of_balance"
        ][:40],
        "multiple_totals": sum(1 for a in detail if (probe_of(a).get("total_count") or 0) > 1),
        "invoice_vs_check_mismatch": sum(
            1 for a in detail if probe_of(a).get("row_count") and not probe_of(a).get("invoice_equals_check")
        ),
        "chosen_total_rule": Counter(
            probe_of(a).get("chosen_total_rule") for a in detail if probe_of(a).get("row_count")
        ),
    }
    findings["reconciliation_by_layout"] = [
        {"header_signature": sig, **dict(counts)}
        for sig, counts in sorted(by_era_recon.items(), key=lambda kv: -sum(kv[1].values()))
    ]

    # Five-year band sample, three sets per band, as the brief asks.
    band_sample: dict[str, list] = {}
    # Sampling for a report, not key material. Seeded so the same
    # three sets per band are drawn on every run and the operator can
    # trace the same documents this report cites.
    rng = random.Random(20260914)  # noqa: S311
    by_band: dict[str, list] = defaultdict(list)
    for art in detail:
        band = era_band(art.meeting_date)
        if band:
            by_band[band].append(art)
    for band, members in sorted(by_band.items()):
        picks = rng.sample(members, min(3, len(members)))
        band_sample[band] = [
            {
                "title": a.title,
                "meeting_date": a.meeting_date,
                "fund": a.fund,
                "path": a.resolved_path,
                "header_signature": probe_of(a).get("header_signature"),
                "has_text_layer": probe_of(a).get("has_text_layer"),
                "rows": probe_of(a).get("row_count"),
                "totals": probe_of(a).get("totals"),
                "first_lines": probe_of(a).get("first_lines"),
            }
            for a in picks
        ]
    findings["five_year_band_sample"] = band_sample

    # ------------------------------------------------------ 4. registers --
    reg_by_year: Counter = Counter()
    det_dates_by_year: dict[str, set] = defaultdict(set)
    reg_dates_by_year: dict[str, set] = defaultdict(set)
    for art in register:
        year = (art.meeting_date or "????")[:4]
        reg_by_year[year] += 1
        reg_dates_by_year[year].add(art.meeting_date)
    for art in detail:
        det_dates_by_year[(art.meeting_date or "????")[:4]].add(art.meeting_date)

    findings["registers"] = {
        "total": len(register),
        "by_year": {
            year: {
                "registers": reg_by_year[year],
                "meetings_with_register": len(reg_dates_by_year[year]),
                "meetings_with_detail": len(det_dates_by_year.get(year, ())),
                "both": len(reg_dates_by_year[year] & det_dates_by_year.get(year, set())),
            }
            for year in sorted(set(reg_by_year) | set(det_dates_by_year))
        },
        "with_text_layer": sum(1 for a in register if probe_of(a).get("has_text_layer")),
    }

    # ---------------------------------------------------- 5. vendor shape --
    vendors: list[str] = []
    for art in detail:
        vendors.extend(probe_of(art).get("sample_vendors") or [])
    # Sampling for a report, not key material; the seed is recorded so
    # the same 200 vendor strings are drawn on every run.
    rng2 = random.Random(20260914)  # noqa: S311
    # Sampling for a report, not key material; seed recorded above.
    vendor_sample = rng2.sample(vendors, min(200, len(vendors))) if vendors else []  # noqa: S311
    findings["vendor_shape"] = {
        "vendor_lines_available": len(vendors),
        "sample_size": len(vendor_sample),
        "sample": sorted(vendor_sample),
        "signals": {
            "all_caps": sum(1 for v in vendor_sample if v.isupper()),
            "mixed_case": sum(1 for v in vendor_sample if not v.isupper() and not v.islower()),
            "trailing_comma": sum(1 for v in vendor_sample if v.endswith(",")),
            "contains_comma": sum(1 for v in vendor_sample if "," in v),
            "has_corporate_suffix": sum(
                1 for v in vendor_sample if re.search(r"\b(?:inc|llc|llp|ltd|co|corp|pllc|pc|lp)\b\.?$", v, re.I)
            ),
            "has_ampersand": sum(1 for v in vendor_sample if "&" in v),
            "has_period": sum(1 for v in vendor_sample if "." in v),
            "has_digit": sum(1 for v in vendor_sample if any(c.isdigit() for c in v)),
            "single_token": sum(1 for v in vendor_sample if len(v.split()) == 1),
        },
    }

    # ------------------------------------------ fixture-month verification --
    fixture_months = ["2026-03-25", "2026-05-27", "2026-06-24", "2026-07-22", "2026-08-26"]
    fixtures = {}
    for month in fixture_months:
        sets = [a for a in kept if a.meeting_date == month]
        fixtures[month] = [
            {
                "title": a.title,
                "fund": a.fund,
                "doc_class": a.doc_class,
                "document_id": a.document_id,
                "path": a.resolved_path,
                "totals": probe_of(a).get("totals"),
                "rows": probe_of(a).get("row_count"),
                "pages": probe_of(a).get("pages"),
                "period": probe_of(a).get("period"),
                "pcard_period": probe_of(a).get("pcard_period"),
            }
            for a in sets
        ]
    findings["fixture_months"] = fixtures

    # Monthly-set count since 2024-09, the first STOP threshold.
    recent = [a for a in detail if (a.meeting_date or "") >= "2024-09-01"]
    recent_dates = sorted({a.meeting_date for a in recent})
    findings["stop_checks"] = {
        "monthly_sets_since_2024_09": len(recent),
        "distinct_voucher_nights_since_2024_09": len(recent_dates),
        "voucher_nights": recent_dates,
        "funds_in_scope_present": sorted({a.fund for a in detail} & set(ALL_FUNDS)),
    }

    print(json.dumps(findings, indent=1, default=str))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(findings, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
