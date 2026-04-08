#!/usr/bin/env python3
"""Download all OSPI Report Card datasets for Kent School District from data.wa.gov.

For each category, downloads from multiple Socrata dataset IDs (multi-year + individual
year supplements), merges records, and deduplicates.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DISTRICT_ORG_ID = "100117"
LIMIT = 50000
BASE_URL = "https://data.wa.gov/resource/{dataset_id}.json"
OUTPUT_DIR = "/home/qorvault/projects/ksd-boarddocs-rag/research/ospi_data"

# Dataset definitions: each category has a list of (dataset_id, filter_field) tuples
# to handle both districtorganizationid and leaorganizationid naming differences
DATASETS = {
    "enrollment": {
        "description": "Student enrollment by year, grade, demographics",
        "sources": [
            ("rxjk-6ieq", "districtorganizationid"),  # 2014-15 to current
        ],
    },
    "graduation": {
        "description": "Graduation rates by year, cohort, demographics",
        "sources": [
            ("9dvy-pnhx", "districtorganizationid"),  # 2014-15 to 2020-21
            ("i23g-ymbg", "districtorganizationid"),  # 2021-22
            ("kigx-4b2d", "districtorganizationid"),  # 2022-23
            ("76iv-8ed4", "districtorganizationid"),  # 2023-24
            ("isxb-523t", "districtorganizationid"),  # 2024-25
        ],
    },
    "assessment": {
        "description": "Smarter Balanced test scores (ELA, Math, Science)",
        "sources": [
            ("292v-tb9r", "districtorganizationid"),  # 2014-15 to 2021-22
            ("xh7m-utwp", "districtorganizationid"),  # 2022-23
            ("x73g-mrqp", "districtorganizationid"),  # 2023-24
        ],
    },
    "discipline": {
        "description": "Exclusion/suspension rates",
        "sources": [
            ("fwbr-3ker", "districtorganizationid"),  # 2014-15 to 2021-22
            ("ixvm-ww8s", "districtorganizationid"),  # 2022-23
            ("sm68-769y", "districtorganizationid"),  # 2023-24
        ],
    },
    "sqss": {
        "description": "SQSS measures (dual credit, 9th grade on track, regular attendance)",
        "sources": [
            ("gjiu-inph", "districtorganizationid"),  # 2014-15
            ("nvyx-ge76", "districtorganizationid"),  # 2015-16
            ("ayz3-kckw", "districtorganizationid"),  # 2016-17
            ("h5ih-67hr", "districtorganizationid"),  # 2017-18
            ("2zsf-krin", "districtorganizationid"),  # 2018-19
            ("nfpj-mzp6", "districtorganizationid"),  # 2019-20
            ("34y8-8dsi", "districtorganizationid"),  # 2020-21
            ("tfs4-sdfn", "districtorganizationid"),  # 2021-22
            ("hs5t-6yez", "districtorganizationid"),  # 2022-23
            ("q9gf-prrp", "districtorganizationid"),  # 2023-24
        ],
    },
    "growth": {
        "description": "Student growth percentiles",
        "sources": [
            ("ufi5-ki2f", "districtorganizationid"),  # 2014-15 to 2018-19
            ("jum2-3mgi", "districtorganizationid"),  # 2022-23
            ("cxts-amj6", "districtorganizationid"),  # 2023-24
            ("hv7j-ib7g", "districtorganizationid"),  # 2024-25
        ],
    },
    "teacher_demographics": {
        "description": "Teacher race, gender, experience (multi-year 2017-18 to 2024-25)",
        "sources": [
            ("yp28-ks6d", "leaorganizationid"),  # 2017-18 to 2024-25
        ],
    },
    "teacher_experience": {
        "description": "Teacher experience distribution (multi-year 2017-18 to 2024-25)",
        "sources": [
            ("bdjb-hg6t", "leaorganizationid"),  # 2017-18 to 2024-25
        ],
    },
    "wakids": {
        "description": "Kindergarten readiness (WaKIDS)",
        "sources": [
            ("fmqr-vuub", "districtorganizationid"),  # 2014-15 to current
        ],
    },
}


def download_page(dataset_id: str, filter_field: str, offset: int = 0) -> list[dict]:
    """Download one page of records for Kent SD from a single Socrata dataset."""
    params = {
        "$where": f"{filter_field}='{DISTRICT_ORG_ID}'",
        "$limit": str(LIMIT),
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
            else:
                print(f"  WARNING: {dataset_id} returned non-list: {type(data)}", file=sys.stderr)
                return []
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"  ERROR: {dataset_id} HTTP {e.code}: {body}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  ERROR: {dataset_id}: {e}", file=sys.stderr)
        return []


def download_dataset(dataset_id: str, filter_field: str) -> list[dict]:
    """Download ALL records for Kent SD from a single Socrata dataset, with pagination."""
    all_records = []
    offset = 0
    while True:
        page = download_page(dataset_id, filter_field, offset)
        all_records.extend(page)
        if len(page) < LIMIT:
            break  # Last page
        offset += LIMIT
        print(f"{len(all_records)}... ", end="", flush=True)
        time.sleep(0.5)  # Be polite between pages
    return all_records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact duplicate records based on JSON serialization of each record."""
    seen = set()
    unique = []
    for r in records:
        # Remove rowid if present (different across datasets for same record)
        r_copy = {k: v for k, v in r.items() if k != "rowid"}
        key = json.dumps(r_copy, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main():
    results = {}

    for category, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {category} - {config['description']}")
        print(f"{'='*60}")

        all_records = []
        for dataset_id, filter_field in config["sources"]:
            print(f"  Fetching {dataset_id} ({filter_field})... ", end="", flush=True)
            records = download_dataset(dataset_id, filter_field)
            print(f"{len(records)} records")
            all_records.extend(records)
            time.sleep(0.5)  # Be polite to the API

        # Deduplicate
        before = len(all_records)
        unique_records = deduplicate(all_records)
        after = len(unique_records)

        if before != after:
            print(f"  Deduplicated: {before} -> {after} records ({before - after} dupes removed)")

        # Save
        output_path = f"{OUTPUT_DIR}/{category}.json"
        with open(output_path, "w") as f:
            json.dump(unique_records, f, indent=2)

        print(f"  Saved: {output_path} ({after} records)")
        results[category] = after

    # Extract attendance.json from SQSS data (Regular Attendance measure)
    print(f"\n{'='*60}")
    print("Extracting: attendance - Regular attendance rates (from SQSS)")
    print(f"{'='*60}")
    sqss_path = f"{OUTPUT_DIR}/sqss.json"
    with open(sqss_path) as f:
        sqss_all = json.load(f)
    # Older SQSS datasets use "measures" (plural), newer use "measure" (singular)
    attendance_records = [
        r for r in sqss_all if r.get("measure") == "Regular Attendance" or r.get("measures") == "Regular Attendance"
    ]
    attendance_path = f"{OUTPUT_DIR}/attendance.json"
    with open(attendance_path, "w") as f:
        json.dump(attendance_records, f, indent=2)
    print(f"  Extracted {len(attendance_records)} attendance records from {len(sqss_all)} SQSS records")
    results["attendance"] = len(attendance_records)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    total = 0
    for category, count in results.items():
        status = "OK" if count > 0 else "EMPTY!"
        print(f"  {category:25s}: {count:6d} records  [{status}]")
        total += count
    print(f"  {'TOTAL':25s}: {total:6d} records")


if __name__ == "__main__":
    main()
