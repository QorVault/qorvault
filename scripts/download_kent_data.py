#!/usr/bin/env python3
"""Download all available Kent School District data from Washington State
open data portals (data.wa.gov Socrata API) and save as organized JSON files.

JSON is used as the primary format because:
- It preserves nested/structured data from the Socrata API
- It's directly consumable by the RAG ingestion pipeline
- CSV flattens structure and loses type info (numbers become strings)

Before saving, checks each dataset against the existing PostgreSQL database
to flag records that are already ingested (from OSPI or kent.k12.wa.us sources).

Usage:
    python3 scripts/download_kent_data.py [--skip-catalog-search] [--dry-run]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISTRICT_ORG_ID = "100117"
DISTRICT_NAME = "Kent School District"
DISTRICT_CODE = "17415"
COUNTY = "King"

PAGE_LIMIT = 50000
BASE_URL = "https://data.wa.gov/resource/{dataset_id}.json"
CATALOG_URL = "https://data.wa.gov/api/catalog/v1"

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OUTPUT_DIR = Path("/home/qorvault/projects/ksd-boarddocs-rag/research/kent_school_data")
OSPI_DATA_DIR = Path("/home/qorvault/projects/ksd-boarddocs-rag/research/ospi_data")


# PostgreSQL connection (for dedup check) — built from env vars
def _build_dsn() -> str:
    """Build PostgreSQL URL from POSTGRES_* env vars. No hardcoded credentials."""
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "qorvault")
    user = os.environ.get("POSTGRES_USER", "qorvault")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. " "Copy .env.example to .env and fill in credentials."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DB_DSN = _build_dsn()
TENANT_ID = "kent_sd"

# Known OSPI/kent.k12.wa.us domains to check for duplicates
DEDUP_DOMAINS = ["ospi.k12.wa.us", "kent.k12.wa.us", "data.wa.gov", "reportcard.ospi"]

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each entry: (dataset_id, filter_field, filter_value_type)
# filter_value_type: "org_id" = use 100117, "name" = use "Kent School District"
DATASETS = {
    "enrollment": {
        "description": "Student enrollment by year, grade, demographics",
        "sources": [
            # Historical (2014-current) — comprehensive multi-year
            ("rxjk-6ieq", "districtorganizationid", "org_id"),
            # Single-year supplements
            ("2rwv-gs2e", "districtorganizationid", "org_id"),  # 2024-25
            ("q4ba-s3jc", "districtorganizationid", "org_id"),  # 2023-24
        ],
    },
    "assessment": {
        "description": "Smarter Balanced test scores (ELA, Math, Science)",
        "sources": [
            ("292v-tb9r", "districtorganizationid", "org_id"),  # 2014-15 to 2021-22
            ("xh7m-utwp", "districtorganizationid", "org_id"),  # 2022-23
            ("x73g-mrqp", "districtorganizationid", "org_id"),  # 2023-24
            ("h5d9-vgwi", "districtorganizationid", "org_id"),  # 2024-25
        ],
    },
    "graduation": {
        "description": "Graduation rates by year, cohort, demographics",
        "sources": [
            ("9dvy-pnhx", "districtorganizationid", "org_id"),  # 2014-15 to 2020-21
            ("i23g-ymbg", "districtorganizationid", "org_id"),  # 2021-22
            ("kigx-4b2d", "districtorganizationid", "org_id"),  # 2022-23
            ("76iv-8ed4", "districtorganizationid", "org_id"),  # 2023-24
            ("isxb-523t", "districtorganizationid", "org_id"),  # 2024-25
            ("vp4a-8vq8", "districtorganizationid", "org_id"),  # alt 2023-24
        ],
    },
    "discipline": {
        "description": "Exclusion/suspension rates",
        "sources": [
            ("fwbr-3ker", "districtorganizationid", "org_id"),  # 2014-15 to 2021-22
            ("ixvm-ww8s", "districtorganizationid", "org_id"),  # 2022-23
            ("sm68-769y", "districtorganizationid", "org_id"),  # 2023-24
        ],
    },
    "sqss": {
        "description": "SQSS measures (dual credit, 9th grade on track, regular attendance)",
        "sources": [
            ("gjiu-inph", "districtorganizationid", "org_id"),  # 2014-15
            ("nvyx-ge76", "districtorganizationid", "org_id"),  # 2015-16
            ("ayz3-kckw", "districtorganizationid", "org_id"),  # 2016-17
            ("h5ih-67hr", "districtorganizationid", "org_id"),  # 2017-18
            ("2zsf-krin", "districtorganizationid", "org_id"),  # 2018-19
            ("nfpj-mzp6", "districtorganizationid", "org_id"),  # 2019-20
            ("34y8-8dsi", "districtorganizationid", "org_id"),  # 2020-21
            ("tfs4-sdfn", "districtorganizationid", "org_id"),  # 2021-22
            ("hs5t-6yez", "districtorganizationid", "org_id"),  # 2022-23
            ("q9gf-prrp", "districtorganizationid", "org_id"),  # 2023-24
        ],
    },
    "growth": {
        "description": "Student growth percentiles",
        "sources": [
            ("ufi5-ki2f", "districtorganizationid", "org_id"),  # 2014-15 to 2018-19
            ("jum2-3mgi", "districtorganizationid", "org_id"),  # 2022-23
            ("cxts-amj6", "districtorganizationid", "org_id"),  # 2023-24
            ("hv7j-ib7g", "districtorganizationid", "org_id"),  # 2024-25
        ],
    },
    "teacher_demographics": {
        "description": "Teacher race, gender, experience (2017-18 to 2024-25)",
        "sources": [
            ("yp28-ks6d", "leaorganizationid", "org_id"),
        ],
    },
    "teacher_experience": {
        "description": "Teacher experience distribution (2017-18 to 2024-25)",
        "sources": [
            ("bdjb-hg6t", "leaorganizationid", "org_id"),
        ],
    },
    "wakids": {
        "description": "Kindergarten readiness (WaKIDS)",
        "sources": [
            ("fmqr-vuub", "districtorganizationid", "org_id"),
        ],
    },
    "highly_capable": {
        "description": "Highly Capable program data",
        "sources": [
            ("85wj-zd4e", "districtorganizationid", "org_id"),
        ],
    },
    "school_improvement": {
        "description": "Washington School Improvement Framework",
        "sources": [
            ("v8by-xqk3", "districtorganizationid", "org_id"),
        ],
    },
    "schools_report_card": {
        "description": "Schools Report Card Data",
        "sources": [
            ("7m7a-urs7", "districtorganizationid", "org_id"),
        ],
    },
}

# Alternative filter field names to try if the primary fails
ALT_FILTER_FIELDS = {
    "org_id": [
        ("districtorganizationid", DISTRICT_ORG_ID),
        ("leaorganizationid", DISTRICT_ORG_ID),
        ("organizationid", DISTRICT_ORG_ID),
        ("districtcode", DISTRICT_CODE),
    ],
    "name": [
        ("districtname", DISTRICT_NAME),
        ("district", DISTRICT_NAME),
        ("district_name", DISTRICT_NAME),
    ],
}


# ---------------------------------------------------------------------------
# Database dedup: load existing external_ids, source_urls, and content hashes
# ---------------------------------------------------------------------------
def load_existing_documents():
    """Query the PostgreSQL database for all existing documents from OSPI or
    kent.k12.wa.us sources. Returns a dict with multiple lookup keys for
    fast dedup checking.

    Returns dict with keys:
        external_ids: set of existing external_id values
        source_urls: set of existing source_url values
        title_hashes: set of MD5(title) for fuzzy matching
        content_hashes: set of MD5(first 500 chars of content_text)
    """
    existing = {
        "external_ids": set(),
        "source_urls": set(),
        "title_hashes": set(),
        "content_hashes": set(),
        "dataset_keys": set(),
    }

    try:
        import subprocess

        # Query all documents that could overlap with OSPI/data.wa.gov data
        query = """
            SELECT external_id, source_url, title,
                   LEFT(content_text, 500) as content_prefix,
                   metadata::text as meta
            FROM documents
            WHERE tenant_id = 'kent_sd'
              AND (
                document_type = 'ospi_data'
                OR source_url LIKE '%ospi%'
                OR source_url LIKE '%kent.k12%'
                OR source_url LIKE '%data.wa.gov%'
                OR external_id LIKE 'ospi-%'
                OR external_id LIKE 'kent-k12-%'
                OR external_id LIKE 'datawagov-%'
              )
        """
        result = subprocess.run(
            [
                "podman",
                "exec",
                "boarddocs-postgres",
                "psql",
                "-U",
                os.environ.get("POSTGRES_USER", "qorvault"),
                "-d",
                os.environ.get("POSTGRES_DB", "qorvault"),
                "-t",
                "-A",
                "-F",
                "\t",
                "-c",
                query,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    ext_id = parts[0].strip()
                    src_url = parts[1].strip()
                    title = parts[2].strip()
                    content = parts[3].strip() if len(parts) > 3 else ""
                    meta = parts[4].strip() if len(parts) > 4 else ""

                    if ext_id:
                        existing["external_ids"].add(ext_id)
                    if src_url:
                        existing["source_urls"].add(src_url)
                    if title:
                        existing["title_hashes"].add(hashlib.md5(title.encode()).hexdigest())
                    if content:
                        existing["content_hashes"].add(hashlib.md5(content.encode()).hexdigest())
                    # Extract dataset_key from metadata if present
                    if meta and "dataset_key" in meta:
                        try:
                            m = json.loads(meta)
                            dk = m.get("dataset_key", "")
                            if dk:
                                existing["dataset_keys"].add(dk)
                        except (json.JSONDecodeError, TypeError):
                            pass

        print(f"  Loaded {len(existing['external_ids'])} existing OSPI/kent.k12 documents from DB")
        print(f"  Existing dataset keys: {sorted(existing['dataset_keys'])}")

    except Exception as e:
        print(f"  WARNING: Could not query DB for dedup: {e}")
        print("  Continuing without dedup check")

    return existing


def check_already_ingested(category, dataset_id, records, existing):
    """Check if this dataset's records overlap with what's already in the database.

    Returns a dict with:
        is_ingested: bool — True if this dataset category already exists in DB
        overlap_count: int — number of records that match existing data
        new_records: list — records that are NOT already in the DB
        message: str — human-readable summary
    """
    result = {
        "is_ingested": False,
        "overlap_count": 0,
        "new_records": records,
        "message": "",
    }

    # Check 1: Does the category match an existing OSPI external_id?
    ospi_ext_id = f"ospi-reportcard-{category}"
    if ospi_ext_id in existing["external_ids"]:
        result["is_ingested"] = True
        result["message"] = f"Category '{category}' already exists as document " f"'{ospi_ext_id}' in the database."

    # Check 2: Does the dataset_key match?
    if category in existing["dataset_keys"]:
        result["is_ingested"] = True
        result["message"] = f"Category '{category}' already ingested (dataset_key in DB metadata)."

    # Check 3: Does the data.wa.gov source URL match?
    source_url = f"https://data.wa.gov/resource/{dataset_id}"
    if source_url in existing["source_urls"]:
        result["is_ingested"] = True
        result["message"] = f"Dataset {dataset_id} source URL already in database."

    return result


# ---------------------------------------------------------------------------
# Socrata API download functions
# ---------------------------------------------------------------------------
def download_page(dataset_id, where_clause, offset=0):
    """Download one page of records from data.wa.gov."""
    params = {
        "$where": where_clause,
        "$limit": str(PAGE_LIMIT),
        "$offset": str(offset),
    }
    url = BASE_URL.format(dataset_id=dataset_id) + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return data
            print(f"    WARNING: {dataset_id} returned non-list: {type(data)}", file=sys.stderr)
            return []
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"    HTTP {e.code}: {body}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        return []


def build_where_clause(filter_field, filter_value_type):
    """Build Socrata $where clause for Kent School District."""
    if filter_value_type == "org_id":
        return f"{filter_field}='{DISTRICT_ORG_ID}'"
    else:
        return f"{filter_field}='{DISTRICT_NAME}'"


def download_dataset(dataset_id, filter_field, filter_value_type):
    """Download all Kent SD records from a Socrata dataset with pagination.
    If the primary filter fails, tries alternative filter fields.
    """
    where = build_where_clause(filter_field, filter_value_type)
    all_records = []
    offset = 0

    while True:
        page = download_page(dataset_id, where, offset)
        all_records.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        print(f"      {len(all_records)}... ", end="", flush=True)
        time.sleep(0.3)

    # If primary filter returned nothing, try alternatives
    if not all_records:
        for alt_field, alt_value in ALT_FILTER_FIELDS.get(filter_value_type, []):
            if alt_field == filter_field:
                continue
            alt_where = f"{alt_field}='{alt_value}'"
            print(f"      Trying alt filter: {alt_field}={alt_value}... ", end="", flush=True)
            page = download_page(dataset_id, alt_where)
            if page:
                all_records = page
                # Paginate the rest
                offset = PAGE_LIMIT
                while len(page) == PAGE_LIMIT:
                    page = download_page(dataset_id, alt_where, offset)
                    all_records.extend(page)
                    offset += PAGE_LIMIT
                    time.sleep(0.3)
                print(f"{len(all_records)} records")
                break
            else:
                print("0 records")

    # If still nothing, try by district name
    if not all_records and filter_value_type == "org_id":
        for alt_field, alt_value in ALT_FILTER_FIELDS["name"]:
            alt_where = f"{alt_field}='{alt_value}'"
            print(f"      Trying name filter: {alt_field}={alt_value}... ", end="", flush=True)
            page = download_page(dataset_id, alt_where)
            if page:
                all_records = page
                offset = PAGE_LIMIT
                while len(page) == PAGE_LIMIT:
                    page = download_page(dataset_id, alt_where, offset)
                    all_records.extend(page)
                    offset += PAGE_LIMIT
                    time.sleep(0.3)
                print(f"{len(all_records)} records")
                break
            else:
                print("0 records")

    return all_records


def deduplicate(records):
    """Remove exact duplicate records based on JSON serialization."""
    seen = set()
    unique = []
    for r in records:
        r_clean = {k: v for k, v in r.items() if k not in ("rowid", ":id")}
        key = json.dumps(r_clean, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def save_csv(records, filepath):
    """Save records as CSV."""
    if not records:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    # Collect all fields across all records
    for r in records[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_json(records, filepath):
    """Save records as JSON (for backward compat with existing ospi_data/)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# Catalog search: discover additional datasets
# ---------------------------------------------------------------------------
def search_catalog():
    """Search data.wa.gov catalog for education-related datasets we might be missing."""
    discovered = []
    search_terms = [
        "school district report card",
        "K-12 education Washington",
        "OSPI school district",
        "school finance expenditure Washington",
        "CTE career technical education Washington",
    ]

    for term in search_terms:
        params = {
            "q": term,
            "domains": "data.wa.gov",
            "search_context": "data.wa.gov",
            "limit": "50",
            "categories": "Education",
        }
        url = CATALOG_URL + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("results", []):
                    resource = item.get("resource", {})
                    ds_id = resource.get("id", "")
                    name = resource.get("name", "")
                    description = resource.get("description", "")[:200]
                    ds_type = resource.get("type", "")

                    # Only tabular datasets (not charts/maps)
                    if ds_type not in ("dataset",):
                        continue

                    # Skip datasets we already know about
                    known_ids = set()
                    for cat_config in DATASETS.values():
                        for src in cat_config["sources"]:
                            known_ids.add(src[0])
                    if ds_id in known_ids:
                        continue

                    discovered.append(
                        {
                            "id": ds_id,
                            "name": name,
                            "description": description,
                            "search_term": term,
                        }
                    )
            time.sleep(0.5)
        except Exception as e:
            print(f"  Catalog search error for '{term}': {e}")

    # Deduplicate by dataset ID
    seen_ids = set()
    unique = []
    for d in discovered:
        if d["id"] not in seen_ids:
            seen_ids.add(d["id"])
            unique.append(d)

    return unique


def probe_dataset_for_kent(dataset_id):
    """Try to fetch Kent SD data from a discovered dataset using various filter fields.
    Returns (records, filter_field_used) or ([], None).
    """
    all_filters = ALT_FILTER_FIELDS["org_id"] + ALT_FILTER_FIELDS["name"] + [("county", COUNTY)]

    for field, value in all_filters:
        where = f"{field}='{value}'"
        page = download_page(dataset_id, where)
        if page:
            return page, field
        time.sleep(0.3)

    return [], None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download Kent SD data from data.wa.gov")
    parser.add_argument(
        "--skip-catalog-search", action="store_true", help="Skip catalog discovery of additional datasets"
    )
    parser.add_argument("--dry-run", action="store_true", help="Check what would be downloaded without saving")
    parser.add_argument("--also-csv", action="store_true", help="Also save CSV copies alongside JSON files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Kent School District — Comprehensive Data Download")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Load existing documents for dedup
    print("\n[1/4] Loading existing documents from PostgreSQL for dedup check...")
    existing = load_existing_documents()

    # Step 2: Download all known datasets
    print("\n[2/4] Downloading known datasets from data.wa.gov...")
    metadata = {
        "download_date": datetime.now().isoformat(),
        "district": DISTRICT_NAME,
        "district_code": DISTRICT_CODE,
        "organization_id": DISTRICT_ORG_ID,
        "datasets": [],
        "already_ingested": [],
        "failed": [],
    }

    for category, config in DATASETS.items():
        print(f"\n  {'─' * 60}")
        print(f"  {category}: {config['description']}")
        print(f"  {'─' * 60}")

        all_records = []
        source_ids_used = []

        for dataset_id, filter_field, filter_value_type in config["sources"]:
            print(f"    Fetching {dataset_id} ({filter_field})... ", end="", flush=True)
            records = download_dataset(dataset_id, filter_field, filter_value_type)
            print(f"{len(records)} records")
            all_records.extend(records)
            if records:
                source_ids_used.append(dataset_id)
            time.sleep(0.3)

        # Deduplicate
        before = len(all_records)
        all_records = deduplicate(all_records)
        after = len(all_records)
        if before != after:
            print(f"    Deduplicated: {before} -> {after} ({before - after} dupes removed)")

        if not all_records:
            print(f"    NO DATA FOUND for {category}")
            metadata["failed"].append(
                {
                    "category": category,
                    "reason": "No records returned from any source",
                }
            )
            continue

        # Dedup check against existing DB
        for ds_id in source_ids_used:
            dedup = check_already_ingested(category, ds_id, all_records, existing)
            if dedup["is_ingested"]:
                print(f"    ** ALREADY IN DB: {dedup['message']}")
                metadata["already_ingested"].append(
                    {
                        "category": category,
                        "dataset_id": ds_id,
                        "message": dedup["message"],
                        "records_in_download": len(all_records),
                    }
                )

        if args.dry_run:
            print(f"    [DRY RUN] Would save {len(all_records)} records")
            continue

        # Save JSON (primary format for RAG ingestion)
        category_dir = OUTPUT_DIR / category
        json_path = category_dir / f"{category}.json"
        save_json(all_records, json_path)
        print(f"    Saved JSON: {json_path} ({len(all_records)} records)")

        # Optionally also save CSV for human inspection
        if args.also_csv:
            csv_path = category_dir / f"{category}.csv"
            save_csv(all_records, csv_path)
            print(f"    Saved CSV: {csv_path}")

        # Also update existing ospi_data/ JSON if the category matches
        ospi_json = OSPI_DATA_DIR / f"{category}.json"
        if ospi_json.exists():
            # Load existing, merge, deduplicate
            with open(ospi_json) as f:
                existing_records = json.load(f)
            merged = existing_records + all_records
            merged = deduplicate(merged)
            if len(merged) > len(existing_records):
                save_json(merged, ospi_json)
                print(
                    f"    Updated ospi_data/{category}.json: "
                    f"{len(existing_records)} -> {len(merged)} records "
                    f"(+{len(merged) - len(existing_records)} new)"
                )
            else:
                print(f"    ospi_data/{category}.json already up to date " f"({len(existing_records)} records)")

        metadata["datasets"].append(
            {
                "category": category,
                "description": config["description"],
                "source_dataset_ids": source_ids_used,
                "total_records": len(all_records),
                "columns": list(all_records[0].keys()) if all_records else [],
            }
        )

    # Step 3: Extract attendance from SQSS
    print(f"\n  {'─' * 60}")
    print("  Extracting attendance subset from SQSS data...")
    print(f"  {'─' * 60}")
    sqss_json = OUTPUT_DIR / "sqss" / "sqss.json"
    if sqss_json.exists():
        with open(sqss_json, encoding="utf-8") as f:
            sqss_records = json.load(f)
        attendance_records = [
            r
            for r in sqss_records
            if r.get("measure") == "Regular Attendance" or r.get("measures") == "Regular Attendance"
        ]
        if attendance_records:
            att_dir = OUTPUT_DIR / "attendance"
            save_json(attendance_records, att_dir / "attendance.json")
            print(f"    Extracted {len(attendance_records)} attendance records")
            if args.also_csv:
                save_csv(attendance_records, att_dir / "attendance.csv")

    # Step 4: Search catalog for additional datasets
    if not args.skip_catalog_search:
        print("\n[3/4] Searching data.wa.gov catalog for additional datasets...")
        discovered = search_catalog()
        print(f"  Found {len(discovered)} candidate datasets not in our known list")

        if discovered:
            # Save the discovery list
            disc_path = OUTPUT_DIR / "metadata" / "discovered_datasets.json"
            disc_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(discovered, disc_path)
            print(f"  Saved discovery list to {disc_path}")

            # Try to fetch Kent data from top candidates
            print("\n  Probing top discovered datasets for Kent SD data...")
            probed_count = 0
            for ds in discovered[:15]:  # Probe top 15 to avoid API abuse
                print(f"    {ds['id']}: {ds['name'][:60]}... ", end="", flush=True)
                records, field_used = probe_dataset_for_kent(ds["id"])
                if records:
                    records = deduplicate(records)
                    print(f"{len(records)} records (filter: {field_used})")

                    if not args.dry_run:
                        # Save to a "discovered" subdirectory
                        safe_name = ds["id"].replace("-", "_")
                        disc_dir = OUTPUT_DIR / "discovered"
                        save_json(records, disc_dir / f"{safe_name}.json")
                        if args.also_csv:
                            save_csv(records, disc_dir / f"{safe_name}.csv")

                    metadata["datasets"].append(
                        {
                            "category": f"discovered/{ds['id']}",
                            "description": ds["name"],
                            "source_dataset_ids": [ds["id"]],
                            "total_records": len(records),
                            "discovered": True,
                        }
                    )
                    probed_count += 1
                else:
                    print("no Kent data")
                time.sleep(0.5)

            print(f"  Found Kent data in {probed_count}/{min(len(discovered), 15)} probed datasets")
    else:
        print("\n[3/4] Skipping catalog search (--skip-catalog-search)")

    # Step 5: Save metadata
    print("\n[4/4] Saving download metadata...")
    meta_dir = OUTPUT_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "download_log.json"
    save_json(metadata, meta_path)
    print(f"  Saved: {meta_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    total_records = 0
    for ds in metadata["datasets"]:
        status = "OK" if ds["total_records"] > 0 else "EMPTY"
        discovered = " [DISCOVERED]" if ds.get("discovered") else ""
        print(f"  {ds['category']:30s}: {ds['total_records']:>7,} records  [{status}]{discovered}")
        total_records += ds["total_records"]
    print(f"  {'TOTAL':30s}: {total_records:>7,} records")

    if metadata["already_ingested"]:
        print(f"\nALREADY IN DATABASE ({len(metadata['already_ingested'])} categories):")
        for item in metadata["already_ingested"]:
            print(f"  {item['category']:30s}: {item['message']}")

    if metadata["failed"]:
        print(f"\nFAILED ({len(metadata['failed'])} categories):")
        for item in metadata["failed"]:
            print(f"  {item['category']:30s}: {item['reason']}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
