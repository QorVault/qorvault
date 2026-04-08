#!/usr/bin/env python3
"""Ingest NEW OSPI/data.wa.gov datasets into PostgreSQL as documents + chunks.

Covers 5 categories not in the original ingest_ospi_data.py:
  1. Highly Capable program data (867 records)
  2. School Improvement Framework / WSIF (491 records)
  3. English Learner Assessment — WIDA ACCESS/ALT + ELPA (115 records)
  4. Per Pupil Expenditure (6 records)
  5. 1003 School Improvement Funds (12 records)

Follows the same narrative chunk format as the existing OSPI data so the
RAG system can answer questions consistently.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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


DSN = _build_dsn()


def safe_float(val, default=0.0):
    """Parse a value that might be a suppressed rate like '<3.6%' or 'N<20'."""
    if not val:
        return default
    s = str(val).replace("%", "").replace(",", "").strip()
    # Handle suppression markers
    if s.startswith("<") or s.startswith(">"):
        s = s[1:]
    if s.startswith("N<") or s.startswith("N>"):
        return default
    if s in ("N/A", "NULL", "Suppress", ""):
        return default
    # Handle "Suppress:" prefix
    if "Suppress" in s or "suppress" in s:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


TENANT_ID = "kent_sd"
DATA_DIR = Path(os.path.expanduser("~/ksd-boarddocs-rag/research/kent_school_data"))
DISCOVERED_DIR = DATA_DIR / "discovered"
SOURCE_URL_BASE = "https://data.wa.gov/resource"
REPORT_CARD_URL = "https://reportcard.ospi.k12.wa.us/ReportCard/ViewSchoolOrDistrict/100117"


# ---------------------------------------------------------------------------
# 1. HIGHLY CAPABLE
# ---------------------------------------------------------------------------
def build_highly_capable_chunks() -> list[str]:
    """Build chunks from Highly Capable program participation data."""
    path = DATA_DIR / "highly_capable" / "highly_capable.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    if not data:
        return []

    chunks = []
    year = data[0].get("schoolyear", "2023")

    # District-level overview — All Students by school
    school_rows = [
        r for r in data if r.get("orglevel", "").lower() == "school" and r.get("studentgroup") == "All Students"
    ]

    if school_rows:
        school_rows.sort(key=lambda r: safe_float(r.get("highlycapablerate", "0")), reverse=True)
        lines = [f"OSPI Report Card: Kent School District — Highly Capable Identification by School ({year})"]
        lines.append("Source: Washington State OSPI Highly Capable Data (data.wa.gov)\n")
        for r in school_rows:
            name = r.get("schoolname", "Unknown")
            rate = r.get("highlycapablerate", "N/A")
            hc_count = r.get("highlycapabletotal", "")
            total = r.get("totalstudents", "")
            extra = f" ({hc_count}/{total} students)" if hc_count and total else ""
            lines.append(f"  {name}: {rate}{extra}")
        chunks.append("\n".join(lines))

    # District-level — All Students aggregate
    district_all = [
        r for r in data if r.get("orglevel", "").lower() == "district" and r.get("studentgroup") == "All Students"
    ]
    if not district_all:
        # Some datasets only have school-level; compute district from schools
        district_all = school_rows

    # Demographic equity breakdown (district or aggregated)
    demo_rows = [
        r
        for r in data
        if r.get("orglevel", "").lower() == "district"
        and r.get("studentgroup") != "All Students"
        and not r.get("studentgroup", "").startswith("Non")
    ]
    if not demo_rows:
        # Fall back to school-level aggregation — pick any school for groups
        demo_rows = [
            r
            for r in data
            if r.get("orglevel", "").lower() == "school"
            and r.get("studentgroup") != "All Students"
            and not r.get("studentgroup", "").startswith("Non")
        ]
        # Deduplicate to one per student group (pick highest-enrollment school)
        by_group = {}
        for r in demo_rows:
            g = r.get("studentgroup", "")
            if g not in by_group:
                by_group[g] = []
            by_group[g].append(r)

    if demo_rows:
        # Group by studentgroup, compute district-wide averages
        by_group = {}
        for r in demo_rows:
            g = r.get("studentgroup", "")
            if g not in by_group:
                by_group[g] = {"hc": 0, "total": 0}
            try:
                by_group[g]["hc"] += int(r.get("highlycapabletotal", 0) or 0)
                by_group[g]["total"] += int(r.get("totalstudents", 0) or 0)
            except (ValueError, TypeError):
                pass

        lines = [f"OSPI Report Card: Kent School District — Highly Capable Identification by Student Group ({year})"]
        lines.append("Source: Washington State OSPI Highly Capable Data (data.wa.gov)")
        lines.append("Shows equity gaps in Highly Capable program identification.\n")

        sorted_groups = sorted(by_group.items(), key=lambda x: x[1]["hc"] / max(x[1]["total"], 1), reverse=True)
        for group, counts in sorted_groups:
            if counts["total"] > 0:
                rate = counts["hc"] / counts["total"] * 100
                lines.append(f"  {group}: {rate:.1f}% ({counts['hc']}/{counts['total']} students)")
        chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 2. SCHOOL IMPROVEMENT FRAMEWORK (WSIF)
# ---------------------------------------------------------------------------
def build_school_improvement_chunks() -> list[str]:
    """Build chunks from Washington School Improvement Framework data."""
    path = DATA_DIR / "school_improvement" / "school_improvement.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    if not data:
        return []

    chunks = []
    year = data[0].get("wsif_year", "2022")

    # Get All Students rows (one per school)
    all_students = [r for r in data if r.get("student_group") == "All Students"]

    # Support tier distribution
    tier_counts = {}
    for r in all_students:
        tier = r.get("_2022_support_tier", r.get("_2022_cycle_support", "Unknown"))
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    if tier_counts:
        lines = [f"OSPI Report Card: Kent School District — School Improvement Support Tiers ({year} WSIF Cycle)"]
        lines.append("Source: Washington School Improvement Framework (data.wa.gov)")
        lines.append("Schools are assigned support tiers based on performance indicators.\n")
        tier_order = ["Foundational Supports", "Targeted 1-2", "Targeted 3+", "Comprehensive", "Comprehensive_LowGrad"]
        for tier in tier_order:
            if tier in tier_counts:
                lines.append(f"  {tier}: {tier_counts[tier]} schools")
        # Any tiers not in the expected order
        for tier, count in sorted(tier_counts.items()):
            if tier not in tier_order:
                lines.append(f"  {tier}: {count} schools")
        total = sum(tier_counts.values())
        lines.append(f"\n  Total schools assessed: {total}")
        chunks.append("\n".join(lines))

    # Schools requiring intervention (Targeted 3+ and Comprehensive)
    intervention_keywords = ("Targeted 3+", "Targeted_EL", "Comprehensive")
    intervention_schools = [
        r for r in all_students if any(kw in r.get("_2022_support_tier", "") for kw in intervention_keywords)
    ]
    if intervention_schools:
        lines = [f"OSPI Report Card: Kent School District — Schools Requiring State Intervention ({year} WSIF)"]
        lines.append("Source: Washington School Improvement Framework (data.wa.gov)")
        lines.append("These schools are identified for additional state support.\n")
        for r in sorted(intervention_schools, key=lambda x: x.get("_2022_support_tier", "")):
            name = r.get("school_name", "Unknown")
            tier = r.get("_2022_support_tier", "Unknown")
            stype = {"P": "Primary", "S": "Secondary", "A": "Alternative", "R": "Regular"}.get(
                r.get("school_type", ""), r.get("school_type", "")
            )
            att = r.get("regularattendance_rate", "N/A")
            grad = r.get("grad_fouryear_rate", "N/A")
            title_i = "Title I" if r.get("_2022_titlei") else "Non-Title I"
            lines.append(f"  {name} ({stype}, {title_i}): {tier}")
            details = []
            if att and att != "N/A":
                details.append(f"Attendance: {att}")
            if grad and grad != "N/A":
                details.append(f"Graduation: {grad}")
            if details:
                lines.append(f"    {', '.join(details)}")
        chunks.append("\n".join(lines))

    # Per-school performance summary (All Students, sorted by combined decile)
    school_perf = [r for r in all_students if r.get("sqss_combined_decile")]
    if school_perf:
        school_perf.sort(key=lambda r: safe_float(r.get("sqss_combined_decile"), 5))
        lines = [f"OSPI Report Card: Kent School District — School Performance Rankings ({year} WSIF)"]
        lines.append("Source: Washington School Improvement Framework (data.wa.gov)")
        lines.append("SQSS Combined Decile: 1 = lowest performing, 10 = highest performing.\n")
        for r in school_perf:
            name = r.get("school_name", "Unknown")
            decile = r.get("sqss_combined_decile", "N/A")
            tier = r.get("_2022_support_tier", "")
            att_decile = r.get("regularattendance_decile", "")
            dual = r.get("dualcredit_rate", "")
            parts = [f"Decile {decile}"]
            if tier:
                parts.append(tier)
            if att_decile:
                parts.append(f"Attendance decile: {att_decile}")
            lines.append(f"  {name}: {', '.join(parts)}")
        chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 3. ENGLISH LEARNER ASSESSMENT
# ---------------------------------------------------------------------------
def build_el_assessment_chunks() -> list[str]:
    """Build chunks from English Learner Assessment (WIDA ACCESS/ALT, ELPA) data."""
    # Merge all 4 discovered EL assessment files
    el_files = [
        DISCOVERED_DIR / "2mv4_s52p.json",  # 2024-25
        DISCOVERED_DIR / "qrns_2pnm.json",  # 2023-24
        DISCOVERED_DIR / "43ir_hnt6.json",  # 2022-23
        DISCOVERED_DIR / "2r36_43nk.json",  # Historical
    ]
    all_records = []
    for fp in el_files:
        if fp.exists():
            with open(fp) as f:
                all_records.extend(json.load(f))
    if not all_records:
        print("  WARNING: No EL assessment data found")
        return []

    chunks = []

    # District-level records only
    district = [r for r in all_records if r.get("orglevel", "").lower() == "district"]

    # WIDA ACCESS proficiency trend (all grades)
    wida_all = [r for r in district if r.get("test") == "WIDAACC" and r.get("gradelevel") == "All Grades"]
    # Also try without grade filter if no "All Grades" rows
    if not wida_all:
        wida_all = [r for r in district if r.get("test") == "WIDAACC"]

    by_year = {}
    for r in wida_all:
        yr = r.get("schoolyear", "")
        grade = r.get("gradelevel", "All Grades")
        if yr not in by_year:
            by_year[yr] = {}
        by_year[yr][grade] = r

    years = sorted(by_year.keys())

    # Overall proficiency trend
    if years:
        lines = ["OSPI Report Card: Kent School District — English Learner Proficiency Trend (WIDA ACCESS)"]
        lines.append("Source: Washington State OSPI English Learner Assessment Data (data.wa.gov)")
        lines.append("Shows percentage of English Learners meeting proficiency standard on WIDA ACCESS.\n")
        for yr in years:
            grades = by_year[yr]
            # Try "All Grades" first, then aggregate
            if "All Grades" in grades:
                r = grades["All Grades"]
                prof_rate = r.get("proficientlabel", r.get("metstandarddat", "N/A"))
                denom = r.get("proficientdenominatordat", r.get("metstandarddenominatordat", ""))
                extra = f" ({denom} students tested)" if denom else ""
                lines.append(f"  {yr}: {prof_rate} proficient{extra}")
            else:
                # List by grade
                for grade in sorted(grades.keys()):
                    r = grades[grade]
                    prof_rate = r.get("proficientlabel", r.get("metstandarddat", "N/A"))
                    denom = r.get("proficientdenominatordat", r.get("metstandarddenominatordat", ""))
                    lines.append(f"  {yr} {grade}: {prof_rate} proficient ({denom} tested)")
        chunks.append("\n".join(lines))

    # Grade-level breakdown for most recent year
    if years:
        latest_yr = years[-1]
        grade_rows = [
            r
            for r in district
            if r.get("test") == "WIDAACC"
            and r.get("schoolyear") == latest_yr
            and r.get("gradelevel") not in ("All Grades", None, "")
        ]
        if grade_rows:
            lines = [f"OSPI Report Card: Kent School District — EL Proficiency by Grade ({latest_yr}, WIDA ACCESS)"]
            lines.append("Source: Washington State OSPI English Learner Assessment Data (data.wa.gov)\n")
            grade_order = [
                "Kindergarten",
                "1st Grade",
                "2nd Grade",
                "3rd Grade",
                "4th Grade",
                "5th Grade",
                "6th Grade",
                "7th Grade",
                "8th Grade",
                "9th Grade",
                "10th Grade",
                "11th Grade",
                "12th Grade",
            ]
            grade_map = {r.get("gradelevel"): r for r in grade_rows}
            for g in grade_order:
                if g in grade_map:
                    r = grade_map[g]
                    prof = r.get("proficientlabel", r.get("metstandarddat", "N/A"))
                    prog = r.get("progressinglabel", "")
                    denom = r.get("proficientdenominatordat", r.get("metstandarddenominatordat", ""))
                    parts = [f"Proficient: {prof}"]
                    if prog:
                        parts.append(f"Progressing: {prog}")
                    if denom:
                        parts.append(f"{denom} tested")
                    lines.append(f"  {g}: {', '.join(parts)}")
            # Add any grades not in the standard order
            for g, r in sorted(grade_map.items()):
                if g not in grade_order:
                    prof = r.get("proficientlabel", r.get("metstandarddat", "N/A"))
                    lines.append(f"  {g}: Proficient: {prof}")
            chunks.append("\n".join(lines))

    # WIDA ALT data (if present)
    wida_alt = [r for r in district if r.get("test") == "WIDAALT"]
    if wida_alt:
        lines = ["OSPI Report Card: Kent School District — EL Alternate Assessment (WIDA ALT)"]
        lines.append("Source: Washington State OSPI English Learner Assessment Data (data.wa.gov)")
        lines.append("WIDA ALT is for English Learners with significant cognitive disabilities.\n")
        for r in sorted(wida_alt, key=lambda x: (x.get("schoolyear", ""), x.get("gradelevel", ""))):
            yr = r.get("schoolyear", "")
            grade = r.get("gradelevel", "All")
            prof = r.get("proficientlabel", r.get("metstandarddat", "N/A"))
            denom = r.get("proficientdenominatordat", r.get("metstandarddenominatordat", ""))
            extra = f" ({denom} tested)" if denom else ""
            lines.append(f"  {yr} {grade}: {prof} proficient{extra}")
        chunks.append("\n".join(lines))

    # Historical ELPA data (if present)
    elpa = [r for r in district if r.get("test") == "ELPA"]
    if elpa:
        lines = ["OSPI Report Card: Kent School District — EL Assessment Historical (ELPA)"]
        lines.append("Source: Washington State OSPI English Learner Assessment Data (data.wa.gov)")
        lines.append("ELPA was the prior English proficiency assessment before WIDA ACCESS.\n")
        for r in sorted(elpa, key=lambda x: (x.get("schoolyear", ""), x.get("gradelevel", ""))):
            yr = r.get("schoolyear", "")
            grade = r.get("gradelevel", "All")
            prof = r.get("metstandarddat", r.get("proficientlabel", "N/A"))
            denom = r.get("metstandarddenominatordat", r.get("proficientdenominatordat", ""))
            extra = f" ({denom} tested)" if denom else ""
            lines.append(f"  {yr} {grade}: {prof}% met standard{extra}")
        chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 4. PER PUPIL EXPENDITURE
# ---------------------------------------------------------------------------
def build_expenditure_chunks() -> list[str]:
    """Build chunks from per-pupil expenditure data."""
    path = DISCOVERED_DIR / "vnm3_j8pe.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    if not data:
        return []

    chunks = []
    data.sort(key=lambda r: r.get("school_year_code", ""))

    lines = ["OSPI Report Card: Kent School District — Per Pupil Expenditure Trend"]
    lines.append("Source: Washington State OSPI Per Pupil Expenditure Data (data.wa.gov)")
    lines.append("Shows total spending per student broken down by funding source.\n")

    for r in data:
        yr = r.get("school_year_code", "Unknown")
        enrollment = r.get("enrollment", "N/A")
        total_ppe = r.get("total_ppe", "N/A")
        total_exp = r.get("total_expenditures1", "N/A")
        local_ppe = r.get("local_ppe", "N/A")
        state_ppe = r.get("state_ppe", "N/A")
        federal_ppe = r.get("federal_ppe", "N/A")

        # Format currency values
        try:
            total_exp_fmt = f"${float(total_exp):,.0f}"
        except (ValueError, TypeError):
            total_exp_fmt = total_exp
        try:
            total_ppe_fmt = f"${float(total_ppe):,.0f}"
        except (ValueError, TypeError):
            total_ppe_fmt = total_ppe

        lines.append(f"  {yr}: {total_ppe_fmt}/student (Total: {total_exp_fmt})")
        lines.append(
            f"    Enrollment: {float(enrollment):,.0f} | "
            f"Local: ${float(local_ppe):,.0f} | "
            f"State: ${float(state_ppe):,.0f} | "
            f"Federal: ${float(federal_ppe):,.0f}"
        )

    # Add trend analysis
    if len(data) >= 2:
        first = data[0]
        last = data[-1]
        try:
            first_ppe = float(first.get("total_ppe", 0))
            last_ppe = float(last.get("total_ppe", 0))
            pct_change = (last_ppe - first_ppe) / first_ppe * 100
            lines.append(
                f"\n  Per-pupil spending increased {pct_change:.1f}% "
                f"from ${first_ppe:,.0f} ({first.get('school_year_code')}) "
                f"to ${last_ppe:,.0f} ({last.get('school_year_code')})."
            )
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # Federal funding spike analysis
        try:
            first_fed = float(first.get("federal_ppe", 0))
            last_fed = float(last.get("federal_ppe", 0))
            mid_data = [r for r in data if r.get("school_year_code") in ("2021-22", "2022-23")]
            if mid_data:
                peak_fed = max(float(r.get("federal_ppe", 0)) for r in mid_data)
                if peak_fed > first_fed * 1.3:
                    lines.append(
                        f"  Federal funding peaked at ${peak_fed:,.0f}/student during COVID relief, "
                        f"now declining to ${last_fed:,.0f}/student."
                    )
        except (ValueError, TypeError):
            pass

    chunks.append("\n".join(lines))
    return chunks


# ---------------------------------------------------------------------------
# 5. 1003 SCHOOL IMPROVEMENT FUNDS
# ---------------------------------------------------------------------------
def build_1003_funds_chunks() -> list[str]:
    """Build chunks from federal 1003 school improvement grant data."""
    files = [
        DISCOVERED_DIR / "8g6g_x265.json",  # 2022-23
        DISCOVERED_DIR / "xnw8_pe32.json",  # 2021-22
    ]
    all_records = []
    for fp in files:
        if fp.exists():
            with open(fp) as f:
                all_records.extend(json.load(f))
    if not all_records:
        print("  WARNING: No 1003 funds data found")
        return []

    chunks = []

    # Group by year
    by_year = {}
    for r in all_records:
        yr = r.get("schoolyear", r.get("school_year", "Unknown"))
        if yr not in by_year:
            by_year[yr] = []
        by_year[yr].append(r)

    lines = ["OSPI Report Card: Kent School District — Federal 1003 School Improvement Grants"]
    lines.append("Source: Washington State OSPI 1003 Funds Data (data.wa.gov)")
    lines.append("Section 1003 grants fund improvements at schools identified for state intervention.\n")

    for yr in sorted(by_year.keys()):
        schools = by_year[yr]
        total_award = sum(int(r.get("_1003_award", 0) or 0) for r in schools)
        lines.append(f"  {yr}: ${total_award:,} total across {len(schools)} schools")
        for r in sorted(schools, key=lambda x: int(x.get("_1003_award", 0) or 0), reverse=True):
            name = r.get("school_name", "Unknown")
            award = int(r.get("_1003_award", 0) or 0)
            status = r.get("accountability_status", "")
            # What the money was spent on
            spending = []
            spend_fields = [
                ("teaching", "Teaching"),
                ("instructional_professional", "Professional Development"),
                ("curriculum", "Curriculum"),
                ("guidance_and_counseling", "Counseling"),
                ("pupil_management_and_safety", "Pupil Safety"),
                ("instructional_technology", "Technology"),
                ("health_related_services", "Health Services"),
                ("learning_resources", "Learning Resources"),
            ]
            for field, label in spend_fields:
                if r.get(field) is True or r.get(field) == "true":
                    spending.append(label)
            spend_str = f" → {', '.join(spending)}" if spending else ""
            lines.append(f"    {name} ({status}): ${award:,}{spend_str}")

    chunks.append("\n".join(lines))
    return chunks


# ---------------------------------------------------------------------------
# MAIN — INSERT INTO DATABASE
# ---------------------------------------------------------------------------
DATASET_BUILDERS = [
    (
        "highly_capable",
        "OSPI Report Card — Highly Capable Program",
        build_highly_capable_chunks,
        "https://data.wa.gov/resource/85wj-zd4e",
    ),
    (
        "school_improvement",
        "OSPI Report Card — School Improvement Framework (WSIF)",
        build_school_improvement_chunks,
        "https://data.wa.gov/resource/v8by-xqk3",
    ),
    (
        "el_assessment",
        "OSPI Report Card — English Learner Assessment (WIDA/ELPA)",
        build_el_assessment_chunks,
        "https://data.wa.gov/resource/2mv4-s52p",
    ),
    (
        "expenditure",
        "OSPI Report Card — Per Pupil Expenditure",
        build_expenditure_chunks,
        "https://data.wa.gov/resource/vnm3-j8pe",
    ),
    (
        "1003_funds",
        "OSPI Report Card — Federal 1003 School Improvement Grants",
        build_1003_funds_chunks,
        "https://data.wa.gov/resource/8g6g-x265",
    ),
]


async def main():
    conn = await asyncpg.connect(DSN)
    doc_ids = []
    total_chunks = 0

    for dataset_key, doc_title, builder, source_url in DATASET_BUILDERS:
        print(f"\nBuilding chunks for: {doc_title}")
        text_chunks = builder()
        if not text_chunks:
            print("  No data — skipping")
            continue

        print(f"  Generated {len(text_chunks)} chunks")

        doc_id = uuid.uuid4()
        external_id = f"ospi-reportcard-{dataset_key}"

        # Insert document (ON CONFLICT DO NOTHING for idempotency)
        await conn.execute(
            """
            INSERT INTO documents (
                id, tenant_id, external_id, document_type, title,
                source_url, meeting_date, committee_name,
                processing_status, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (tenant_id, external_id) DO NOTHING
        """,
            doc_id,
            TENANT_ID,
            external_id,
            "ospi_data",
            doc_title,
            source_url,
            None,
            "Washington State OSPI",
            "complete",
            json.dumps(
                {
                    "source": "ospi_report_card",
                    "data_portal": "data.wa.gov",
                    "district_id": "100117",
                    "dataset_key": dataset_key,
                }
            ),
        )

        # Check if doc was actually inserted (might already exist from prior run)
        existing = await conn.fetchval(
            "SELECT id FROM documents WHERE tenant_id=$1 AND external_id=$2",
            TENANT_ID,
            external_id,
        )
        if existing != doc_id:
            print(f"  Document already exists ({existing}), replacing chunks...")
            doc_id = existing
            deleted = await conn.execute("DELETE FROM chunks WHERE document_id=$1", doc_id)
            print(f"  Deleted old chunks: {deleted}")

        # Insert chunks
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = uuid.uuid4()
            word_count = len(chunk_text.split())
            await conn.execute(
                """
                INSERT INTO chunks (
                    id, tenant_id, document_id, chunk_index, content,
                    token_count, embedding_status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (document_id, chunk_index) DO NOTHING
            """,
                chunk_id,
                TENANT_ID,
                doc_id,
                idx,
                chunk_text,
                int(word_count * 1.3),
                "pending",
                json.dumps(
                    {
                        "document_type": "ospi_data",
                        "dataset_key": dataset_key,
                    }
                ),
            )

        doc_ids.append(doc_id)
        total_chunks += len(text_chunks)
        print(f"  Inserted doc {doc_id} with {len(text_chunks)} chunks")

    await conn.close()

    print(f"\n{'=' * 60}")
    print(f"Inserted {len(doc_ids)} documents, {total_chunks} chunks total")
    print("All chunks have embedding_status='pending'")
    print("Run the embedding pipeline to embed them into Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
