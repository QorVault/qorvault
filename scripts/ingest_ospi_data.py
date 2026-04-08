#!/usr/bin/env python3
"""Ingest OSPI Report Card data into PostgreSQL as documents + chunks,
prioritized by consequence for RAG retrieval.

Creates narrative text chunks from structured JSON data so the RAG
system can answer questions about district performance, demographics,
test scores, graduation rates, attendance, discipline, and staffing.
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
TENANT_ID = "kent_sd"
DATA_DIR = Path(os.path.expanduser("~/ksd-boarddocs-rag/research/ospi_data"))
SOURCE_URL = "https://reportcard.ospi.k12.wa.us/ReportCard/ViewSchoolOrDistrict/100117"


def make_chunks(text_blocks: list[str]) -> list[str]:
    """Return text blocks as-is — each block is already a self-contained chunk."""
    return [b.strip() for b in text_blocks if b.strip()]


def load_json(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. ENROLLMENT
# ---------------------------------------------------------------------------
def build_enrollment_chunks() -> list[str]:
    """Build chunks from enrollment data — demographics, trends, special pops."""
    data = load_json("enrollment.json")
    if not data:
        return []

    chunks = []

    # District-level totals by year
    district_rows = [
        r for r in data if r.get("organizationlevel") == "District" and r.get("gradelevel") == "All Grades"
    ]

    # Group by year
    by_year = {}
    for r in district_rows:
        yr = r.get("schoolyear", "")
        if yr not in by_year:
            by_year[yr] = r

    years_sorted = sorted(by_year.keys())

    # Enrollment trend chunk
    lines = ["OSPI Report Card: Kent School District — Enrollment Trends"]
    lines.append("Source: Washington State OSPI Report Card (data.wa.gov)")
    lines.append(f"Data covers school years {years_sorted[0]} through {years_sorted[-1]}.\n")
    for yr in years_sorted:
        r = by_year[yr]
        total = r.get("all_students", "N/A")
        female = r.get("female", "N/A")
        male = r.get("male", "N/A")
        lines.append(f"{yr}: Total {total} students (Female: {female}, Male: {male})")
    chunks.append("\n".join(lines))

    # Racial/ethnic composition for most recent year
    if years_sorted:
        latest = by_year[years_sorted[-1]]
        yr = years_sorted[-1]
        lines = [f"OSPI Report Card: Kent School District — Racial/Ethnic Enrollment ({yr})"]
        lines.append("Source: Washington State OSPI Report Card\n")
        race_fields = [
            ("white", "White"),
            ("asian", "Asian"),
            ("hispanic_latino_of_any_race_s_", "Hispanic/Latino"),
            ("black_african_american", "Black/African American"),
            ("two_or_more_races", "Two or More Races"),
            ("native_hawaiian_other_pacific_islander", "Native Hawaiian/Pacific Islander"),
            ("american_indian_alaskan_native", "American Indian/Alaska Native"),
        ]
        total = int(latest.get("all_students", 0) or 0)
        for field, label in race_fields:
            val = latest.get(field, "N/A")
            if val and val != "N/A" and total > 0:
                try:
                    pct = f" ({int(val)/total*100:.1f}%)"
                except (ValueError, TypeError):
                    pct = ""
            else:
                pct = ""
            lines.append(f"  {label}: {val}{pct}")
        lines.append("\nKent is a majority-minority district with no single racial group above 27%.")
        chunks.append("\n".join(lines))

    # Special populations for most recent year
    if years_sorted:
        latest = by_year[years_sorted[-1]]
        yr = years_sorted[-1]
        lines = [f"OSPI Report Card: Kent School District — Special Populations ({yr})"]
        lines.append("Source: Washington State OSPI Report Card\n")
        pop_fields = [
            ("english_language_learners", "English Language Learners"),
            ("low_income", "Low Income"),
            ("students_with_disabilities", "Students with Disabilities"),
            ("section_504", "Section 504"),
            ("homeless", "Homeless"),
            ("foster_care", "Foster Care"),
            ("migrant", "Migrant"),
            ("military_parent", "Military Parent"),
            ("highly_capable", "Highly Capable"),
        ]
        total = int(latest.get("all_students", 0) or 0)
        for field, label in pop_fields:
            val = latest.get(field, "N/A")
            if val and val != "N/A" and total > 0:
                try:
                    pct = f" ({int(val)/total*100:.1f}%)"
                except (ValueError, TypeError):
                    pct = ""
            else:
                pct = ""
            lines.append(f"  {label}: {val}{pct}")
        chunks.append("\n".join(lines))

    # Racial composition change over time
    if len(years_sorted) >= 2:
        first_yr = years_sorted[0]
        last_yr = years_sorted[-1]
        first = by_year[first_yr]
        last = by_year[last_yr]
        lines = [f"OSPI Report Card: Kent School District — Demographic Shift ({first_yr} to {last_yr})"]
        lines.append("Source: Washington State OSPI Report Card\n")
        race_fields = [
            ("white", "White"),
            ("asian", "Asian"),
            ("hispanic_latino_of_any_race_s_", "Hispanic/Latino"),
            ("black_african_american", "Black/African American"),
            ("english_language_learners", "English Language Learners"),
        ]
        for field, label in race_fields:
            v1 = first.get(field, 0)
            v2 = last.get(field, 0)
            try:
                v1, v2 = int(v1 or 0), int(v2 or 0)
                diff = v2 - v1
                sign = "+" if diff > 0 else ""
                lines.append(f"  {label}: {v1} → {v2} ({sign}{diff})")
            except (ValueError, TypeError):
                pass
        chunks.append("\n".join(lines))

    # Grade-level enrollment for most recent year
    grade_rows = [
        r
        for r in data
        if r.get("organizationlevel") == "District"
        and r.get("gradelevel") not in ("All Grades", None)
        and r.get("schoolyear") == years_sorted[-1]
    ]
    if grade_rows:
        yr = years_sorted[-1]
        lines = [f"OSPI Report Card: Kent School District — Enrollment by Grade ({yr})"]
        lines.append("Source: Washington State OSPI Report Card\n")
        grade_order = [
            "Pre-Kindergarten",
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
                lines.append(f"  {g}: {r.get('all_students', 'N/A')} students")
        chunks.append("\n".join(lines))

    # School-level enrollment for most recent year
    school_rows = [
        r
        for r in data
        if r.get("organizationlevel") == "School"
        and r.get("gradelevel") == "All Grades"
        and r.get("schoolyear") == years_sorted[-1]
    ]
    if school_rows:
        yr = years_sorted[-1]
        school_rows.sort(key=lambda r: int(r.get("all_students", 0) or 0), reverse=True)
        lines = [f"OSPI Report Card: Kent School District — School Enrollment ({yr})"]
        lines.append("Source: Washington State OSPI Report Card\n")
        for r in school_rows:
            name = r.get("schoolname", "Unknown")
            total = r.get("all_students", "N/A")
            lines.append(f"  {name}: {total} students")
        chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 2. ASSESSMENT (TEST SCORES)
# ---------------------------------------------------------------------------
def build_assessment_chunks() -> list[str]:
    """Build chunks from Smarter Balanced assessment data."""
    data = load_json("assessment.json")
    if not data:
        return []

    chunks = []

    # Filter to district-level, all students
    district_all = [
        r
        for r in data
        if r.get("organizationlevel") == "District"
        and r.get("studentgroup") == "All Students"
        and r.get("studentgrouptypeinchart") in ("All", "AllStudents", None, "")
    ]

    # Group by subject and year
    by_subject_year = {}
    for r in district_all:
        subj = r.get("testsubject", "")
        yr = r.get("schoolyear", "")
        grade = r.get("gradelevel", "")
        key = (subj, yr, grade)
        by_subject_year[key] = r

    # ELA trend (All Grades)
    for subject, label in [("ELA", "English Language Arts (ELA)"), ("Math", "Mathematics"), ("Science", "Science")]:
        trend_rows = {k: v for k, v in by_subject_year.items() if k[0] == subject and k[2] == "All Grades"}
        if not trend_rows:
            # Try other grade level names
            trend_rows = {k: v for k, v in by_subject_year.items() if k[0] == subject and "all" in k[2].lower()}
        years = sorted(trend_rows.keys(), key=lambda k: k[1])
        if years:
            lines = [f"OSPI Report Card: Kent School District — {label} Proficiency Trend"]
            lines.append("Source: Washington State OSPI Report Card (Smarter Balanced Assessment)\n")
            for k in years:
                r = trend_rows[k]
                pct = r.get("percentmetstandard", r.get("percentlevel3", "N/A"))
                count = r.get("countmetstandard", "")
                tested = r.get("countofstudentsexpectedtotestingroup", "")
                extra = f" ({count}/{tested} students)" if count and tested else ""
                lines.append(f"  {k[1]}: {pct}% met standard{extra}")
            chunks.append("\n".join(lines))

    # Grade-level breakdown for most recent year
    years_available = sorted(set(k[1] for k in by_subject_year.keys()))
    if years_available:
        latest_yr = years_available[-1]
        for subject, label in [("ELA", "ELA"), ("Math", "Math"), ("Science", "Science")]:
            grade_rows = {
                k: v
                for k, v in by_subject_year.items()
                if k[0] == subject and k[1] == latest_yr and k[2] not in ("All Grades",)
            }
            if grade_rows:
                lines = [f"OSPI Report Card: Kent School District — {label} Scores by Grade ({latest_yr})"]
                lines.append("Source: Washington State OSPI Report Card\n")
                for k in sorted(grade_rows.keys(), key=lambda x: x[2]):
                    r = grade_rows[k]
                    pct = r.get("percentmetstandard", r.get("percentlevel3", "N/A"))
                    grade = k[2]
                    l1 = r.get("percentlevel1", "")
                    l2 = r.get("percentlevel2", "")
                    l3 = r.get("percentlevel3", "")
                    l4 = r.get("percentlevel4", "")
                    detail = ""
                    if l1 and l2 and l3 and l4:
                        detail = f" (L1: {l1}%, L2: {l2}%, L3: {l3}%, L4: {l4}%)"
                    lines.append(f"  Grade {grade}: {pct}% proficient{detail}")
                chunks.append("\n".join(lines))

    # Demographic gaps for most recent year (ELA and Math)
    if years_available:
        latest_yr = years_available[-1]
        demo_rows = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("schoolyear") == latest_yr
            and r.get("gradelevel") in ("All Grades",)
            and r.get("studentgroup") != "All Students"
        ]
        for subject in ["ELA", "Math"]:
            subj_rows = [r for r in demo_rows if r.get("testsubject") == subject]
            if subj_rows:
                lines = [
                    f"OSPI Report Card: Kent School District — {subject} Proficiency by Student Group ({latest_yr})"
                ]
                lines.append("Source: Washington State OSPI Report Card\n")
                subj_rows.sort(key=lambda r: float(r.get("percentmetstandard", 0) or 0), reverse=True)
                for r in subj_rows[:20]:  # top 20 groups
                    group = r.get("studentgroup", "Unknown")
                    pct = r.get("percentmetstandard", "N/A")
                    if pct and pct != "N/A":
                        lines.append(f"  {group}: {pct}%")
                chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 3. GRADUATION
# ---------------------------------------------------------------------------
def build_graduation_chunks() -> list[str]:
    """Build chunks from graduation rate data."""
    data = load_json("graduation.json")
    if not data:
        return []

    chunks = []

    # District-level, all students, 4-year cohort
    four_year = [
        r
        for r in data
        if r.get("organizationlevel") == "District"
        and r.get("studentgroup") == "All Students"
        and r.get("cohort") in ("Four Year", "Four-Year", "4")
    ]

    by_year = {}
    for r in four_year:
        yr = r.get("schoolyear", "")
        by_year[yr] = r

    years = sorted(by_year.keys())
    if years:
        lines = ["OSPI Report Card: Kent School District — Four-Year Graduation Rate Trend"]
        lines.append("Source: Washington State OSPI Report Card\n")
        for yr in years:
            r = by_year[yr]
            rate = r.get("graduationrate", "N/A")
            grads = r.get("graduate", "")
            cohort = r.get("finalcohort", "")
            extra = f" ({grads}/{cohort})" if grads and cohort and grads != "NULL" and cohort != "NULL" else ""
            lines.append(f"  {yr}: {rate}%{extra}")
        chunks.append("\n".join(lines))

    # Demographic gaps for most recent year
    if years:
        latest_yr = years[-1]
        demo_rows = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("schoolyear") == latest_yr
            and r.get("cohort") in ("Four Year", "Four-Year", "4")
            and r.get("studentgroup") != "All Students"
        ]
        if demo_rows:
            lines = [f"OSPI Report Card: Kent School District — Graduation Rate by Student Group ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            demo_rows.sort(key=lambda r: float(r.get("graduationrate", 0) or 0), reverse=True)
            for r in demo_rows[:25]:
                group = r.get("studentgroup", "Unknown")
                rate = r.get("graduationrate", "N/A")
                if rate and rate != "N/A" and rate != "NULL":
                    lines.append(f"  {group}: {rate}%")
            chunks.append("\n".join(lines))

    # Extended cohort rates for most recent year
    if years:
        latest_yr = years[-1]
        extended = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("studentgroup") == "All Students"
            and r.get("schoolyear") == latest_yr
            and r.get("cohort") not in ("Four Year", "Four-Year", "4")
        ]
        if extended:
            lines = [f"OSPI Report Card: Kent School District — Extended Graduation Rates ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for r in sorted(extended, key=lambda x: x.get("cohort", "")):
                period = r.get("cohort", "Unknown")
                rate = r.get("graduationrate", "N/A")
                lines.append(f"  {period} cohort: {rate}%")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 4. ATTENDANCE
# ---------------------------------------------------------------------------
def build_attendance_chunks() -> list[str]:
    """Build chunks from attendance/regular attendance data."""
    data = load_json("attendance.json")
    if not data:
        return []

    chunks = []

    # District-level, all students
    district_all = [
        r
        for r in data
        if r.get("organizationlevel") == "District"
        and r.get("studentgroup") == "All Students"
        and r.get("gradelevel") in ("All Grades", None, "")
    ]

    by_year = {}
    for r in district_all:
        yr = r.get("schoolyear", r.get("reportingyear", ""))
        by_year[yr] = r

    years = sorted(by_year.keys())
    if years:
        lines = ["OSPI Report Card: Kent School District — Regular Attendance Rate Trend"]
        lines.append("Regular attendance = attending 90% or more of school days.")
        lines.append("Source: Washington State OSPI Report Card\n")
        for yr in years:
            r = by_year[yr]
            rate = r.get("percentregularattenders", r.get("percentmetstandard", r.get("rate", "N/A")))
            count = r.get("numerator", r.get("numberregularattenders", ""))
            total = r.get("denominator", r.get("numberstudents", ""))
            extra = f" ({count}/{total} students)" if count and total else ""
            lines.append(f"  {yr}: {rate}%{extra}")
        chunks.append("\n".join(lines))

    # Grade-level breakdown for most recent year
    if years:
        latest_yr = years[-1]
        grade_rows = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("studentgroup") == "All Students"
            and r.get("gradelevel") not in ("All Grades", None, "")
            and (r.get("schoolyear", r.get("reportingyear", "")) == latest_yr)
        ]
        if grade_rows:
            lines = [f"OSPI Report Card: Kent School District — Attendance by Grade ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            grade_rows.sort(key=lambda r: r.get("gradelevel", ""))
            for r in grade_rows:
                grade = r.get("gradelevel", "Unknown")
                rate = r.get("percentregularattenders", r.get("percentmetstandard", r.get("rate", "N/A")))
                lines.append(f"  {grade}: {rate}% regular attendance")
            chunks.append("\n".join(lines))

    # Demographic breakdown for most recent year
    if years:
        latest_yr = years[-1]
        demo_rows = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("studentgroup") != "All Students"
            and r.get("gradelevel") in ("All Grades", None, "")
            and (r.get("schoolyear", r.get("reportingyear", "")) == latest_yr)
        ]
        if demo_rows:
            lines = [f"OSPI Report Card: Kent School District — Attendance by Student Group ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            demo_rows.sort(
                key=lambda r: float(
                    r.get("percentregularattenders", r.get("percentmetstandard", r.get("rate", 0))) or 0
                ),
                reverse=True,
            )
            for r in demo_rows[:20]:
                group = r.get("studentgroup", "Unknown")
                rate = r.get("percentregularattenders", r.get("percentmetstandard", r.get("rate", "N/A")))
                if rate and rate != "N/A":
                    lines.append(f"  {group}: {rate}%")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 5. DISCIPLINE
# ---------------------------------------------------------------------------
def build_discipline_chunks() -> list[str]:
    """Build chunks from discipline/exclusion data."""
    data = load_json("discipline.json")
    if not data:
        return []

    chunks = []

    # District-level, all students, all grades
    district_all = [
        r
        for r in data
        if r.get("organizationlevel") == "District"
        and r.get("student_group") == "All Students"
        and r.get("gradelevel") in ("All Grades", "All", None, "")
    ]

    by_year = {}
    for r in district_all:
        yr = r.get("schoolyear", "")
        by_year[yr] = r

    years = sorted(by_year.keys())
    if years:
        lines = ["OSPI Report Card: Kent School District — Discipline Exclusion Rate Trend"]
        lines.append("Exclusion rate = percentage of students receiving at least one exclusionary discipline action.")
        lines.append("Source: Washington State OSPI Report Card\n")
        for yr in years:
            r = by_year[yr]
            rate = r.get("disciplinerate", "N/A")
            excluded = r.get("disciplinenumerator", "")
            enrolled = r.get("disciplinedenominator", "")
            extra = f" ({excluded}/{enrolled})" if excluded and enrolled else ""
            lines.append(f"  {yr}: {rate}%{extra}")
        chunks.append("\n".join(lines))

    # Grade-level breakdown for most recent year
    if years:
        latest_yr = years[-1]
        grade_rows = [
            r
            for r in data
            if r.get("organizationlevel") == "District"
            and r.get("student_group") == "All Students"
            and r.get("gradelevel") not in ("All Grades", "All", None, "")
            and r.get("schoolyear") == latest_yr
        ]
        if grade_rows:
            lines = [f"OSPI Report Card: Kent School District — Discipline by Grade ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            grade_rows.sort(key=lambda r: r.get("gradelevel", ""))
            for r in grade_rows:
                grade = r.get("gradelevel", "Unknown")
                rate = r.get("disciplinerate", "N/A")
                excluded = r.get("disciplinenumerator", "")
                extra = f" ({excluded} students)" if excluded else ""
                lines.append(f"  {grade}: {rate}% exclusion rate{extra}")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 6. TEACHER DATA
# ---------------------------------------------------------------------------
def build_teacher_chunks() -> list[str]:
    """Build chunks from teacher demographics and experience data."""
    chunks = []

    # Teacher demographics
    data = load_json("teacher_demographics.json")
    if data:
        district_all = [r for r in data if r.get("organizationlevel") in ("District", "LEA")]

        by_year = {}
        for r in district_all:
            yr = r.get("schoolyear", "")
            if yr not in by_year:
                by_year[yr] = []
            by_year[yr].append(r)

        years = sorted(by_year.keys())
        if years:
            latest_yr = years[-1]
            rows = by_year[latest_yr]
            lines = [f"OSPI Report Card: Kent School District — Teacher Demographics ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for r in sorted(
                rows, key=lambda x: int(x.get("countofteachers", x.get("teachercount", 0)) or 0), reverse=True
            ):
                race = r.get("raceethnicity", r.get("race_ethnicity", "Unknown"))
                count = r.get("countofteachers", r.get("teachercount", "N/A"))
                pct = r.get("percentofteachers", r.get("teacherpercent", ""))
                avg_exp = r.get("averageexperience", r.get("avgexperience", ""))
                extra_parts = []
                if pct:
                    extra_parts.append(f"{pct}%")
                if avg_exp:
                    extra_parts.append(f"avg {avg_exp} yrs experience")
                extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
                lines.append(f"  {race}: {count} teachers{extra}")
            chunks.append("\n".join(lines))

        # Teacher count trend
        if len(years) >= 2:
            lines = ["OSPI Report Card: Kent School District — Teacher Staffing Trend"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for yr in years:
                rows = by_year[yr]
                total = sum(int(r.get("countofteachers", r.get("teachercount", 0)) or 0) for r in rows)
                if total > 0:
                    lines.append(f"  {yr}: {total} teachers")
            chunks.append("\n".join(lines))

    # Teacher experience distribution
    exp_data = load_json("teacher_experience.json")
    if exp_data:
        district_exp = [r for r in exp_data if r.get("organizationlevel") in ("District", "LEA")]

        by_year = {}
        for r in district_exp:
            yr = r.get("schoolyear", "")
            if yr not in by_year:
                by_year[yr] = []
            by_year[yr].append(r)

        years = sorted(by_year.keys())
        if years:
            latest_yr = years[-1]
            rows = by_year[latest_yr]
            lines = [f"OSPI Report Card: Kent School District — Teacher Experience Distribution ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for r in sorted(rows, key=lambda x: x.get("experiencerange", x.get("experience_range", ""))):
                exp_range = r.get("experiencerange", r.get("experience_range", "Unknown"))
                count = r.get("countofteachers", r.get("teachercount", "N/A"))
                pct = r.get("percentofteachers", r.get("teacherpercent", ""))
                extra = f" ({pct}%)" if pct else ""
                lines.append(f"  {exp_range}: {count} teachers{extra}")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 7. SQSS (Dual Credit, 9th Grade On Track)
# ---------------------------------------------------------------------------
def build_sqss_chunks() -> list[str]:
    """Build chunks from SQSS indicator data."""
    data = load_json("sqss.json")
    if not data:
        return []

    chunks = []

    # Find measure field name
    measure_field = "measure"
    if data and "measures" in data[0] and "measure" not in data[0]:
        measure_field = "measures"

    for measure_name, title in [
        ("Dual Credit", "Dual Credit Participation Rate"),
        ("9th Grade on Track", "9th Grade On-Track Rate"),
    ]:
        measure_rows = [
            r
            for r in data
            if r.get(measure_field, "") == measure_name
            or r.get("measure", "") == measure_name
            or r.get("measures", "") == measure_name
        ]

        district_all = [
            r
            for r in measure_rows
            if r.get("organizationlevel") == "District" and r.get("studentgroup") == "All Students"
        ]

        by_year = {}
        for r in district_all:
            yr = r.get("schoolyear", r.get("reportingyear", ""))
            by_year[yr] = r

        years = sorted(by_year.keys())
        if years:
            lines = [f"OSPI Report Card: Kent School District — {title} Trend"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for yr in years:
                r = by_year[yr]
                rate = r.get("percentmetstandard", r.get("rate", r.get("percent", "N/A")))
                count = r.get("numerator", "")
                total = r.get("denominator", "")
                extra = f" ({count}/{total} students)" if count and total else ""
                lines.append(f"  {yr}: {rate}%{extra}")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 8. STUDENT GROWTH
# ---------------------------------------------------------------------------
def build_growth_chunks() -> list[str]:
    """Build chunks from student growth percentile data."""
    data = load_json("growth.json")
    if not data:
        return []

    chunks = []

    # District-level, all students
    district_all = [
        r for r in data if r.get("organizationlevel") == "District" and r.get("studentgroup") == "All Students"
    ]

    for subject_value, label in [("English Language Arts", "ELA"), ("Math", "Math")]:
        subj_rows = [r for r in district_all if r.get("subject") == subject_value]

        by_year = {}
        for r in subj_rows:
            yr = r.get("schoolyear", "")
            grade = r.get("gradelevel", "")
            if yr not in by_year:
                by_year[yr] = {}
            by_year[yr][grade] = r

        years = sorted(by_year.keys())
        if years:
            latest_yr = years[-1]
            grades = by_year[latest_yr]
            lines = [f"OSPI Report Card: Kent School District — {label} Student Growth Percentiles ({latest_yr})"]
            lines.append("Median SGP of 50 = average growth statewide.")
            lines.append("Source: Washington State OSPI Report Card\n")
            for grade in sorted(grades.keys()):
                r = grades[grade]
                sgp = r.get("mediansgp", "N/A")
                count = r.get("studentcount", "")
                extra = f" ({count} students)" if count else ""
                lines.append(f"  Grade {grade}: Median SGP {sgp}{extra}")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 9. WaKIDS (Kindergarten Readiness)
# ---------------------------------------------------------------------------
def build_wakids_chunks() -> list[str]:
    """Build chunks from WaKIDS kindergarten readiness data."""
    data = load_json("wakids.json")
    if not data:
        return []

    chunks = []

    # District-level, all students, all domains
    district_all = [
        r for r in data if r.get("organizationlevel") == "District" and r.get("studentgroup") == "All Students"
    ]

    # Group by year and domain
    by_year_domain = {}
    for r in district_all:
        yr = r.get("schoolyear", "")
        domain = r.get("domain", r.get("subdomain", ""))
        if yr and domain:
            if yr not in by_year_domain:
                by_year_domain[yr] = {}
            by_year_domain[yr][domain] = r

    years = sorted(by_year_domain.keys())

    # Trend by domain
    if years:
        lines = ["OSPI Report Card: Kent School District — Kindergarten Readiness (WaKIDS) Trend"]
        lines.append("Percentage of kindergarteners demonstrating readiness by developmental domain.")
        lines.append("Source: Washington State OSPI Report Card\n")
        # Get all domain names
        all_domains = set()
        for yr_data in by_year_domain.values():
            all_domains.update(yr_data.keys())
        for yr in years:
            domains = by_year_domain[yr]
            parts = []
            for d in sorted(all_domains):
                if d in domains:
                    r = domains[d]
                    pct = r.get("percentmetstandard", r.get("percentready", r.get("percent", "N/A")))
                    if pct and pct != "N/A":
                        parts.append(f"{d}: {pct}%")
            if parts:
                lines.append(f"  {yr}: {', '.join(parts)}")
        chunks.append("\n".join(lines))

    # Most recent year detail
    if years:
        latest_yr = years[-1]
        domains = by_year_domain.get(latest_yr, {})
        if domains:
            lines = [f"OSPI Report Card: Kent School District — Kindergarten Readiness Detail ({latest_yr})"]
            lines.append("Source: Washington State OSPI Report Card\n")
            for d in sorted(domains.keys()):
                r = domains[d]
                pct = r.get("percentmetstandard", r.get("percentready", r.get("percent", "N/A")))
                count = r.get("numerator", r.get("countready", ""))
                total = r.get("denominator", r.get("counttested", ""))
                extra = f" ({count}/{total} students)" if count and total else ""
                lines.append(f"  {d}: {pct}% ready{extra}")
            chunks.append("\n".join(lines))

    return chunks


# ---------------------------------------------------------------------------
# 10. SUMMARY CHUNK (key findings)
# ---------------------------------------------------------------------------
def build_summary_chunk() -> list[str]:
    """Build a high-level summary chunk of key findings."""
    return [
        """OSPI Report Card: Kent School District — Key Findings Summary (Data through 2025-26)
Source: Washington State OSPI Report Card (data.wa.gov), retrieved February 2026.

ENROLLMENT: 25,377 students (2025-26), down 9.8% from 28,151 in 2014-15. COVID caused an 11.2% enrollment drop. Kent is now a majority-minority district — no single racial group above 27%. White students declined from 36.7% to 26.1%, Asian grew from 17.4% to 24.5%. English Language Learners nearly doubled from 18% to 31.8% of enrollment.

CHRONIC ABSENCE CRISIS: Only 67.1% of students attend regularly (90%+ of days) in 2024-25, down from 85-87% pre-pandemic. 10th graders worst at 51.2%. About 1 in 3 students is chronically absent — 20 points below pre-pandemic levels with no sign of recovery.

MATH PROFICIENCY GUTTED: Only 32.9% of students meet math standards (2024-25), down from 46-49% pre-pandemic. 6th grade worst at 26.7%. ELA proficiency at 42.6%, also below pre-pandemic ~57%.

GRADUATION: 86.6% four-year rate (2024-25). Equity gaps: NHPI students 63.0%, ELL 69.6%, students with disabilities 71.0% vs. Highly Capable 97.2%.

9TH GRADE ON-TRACK COLLAPSE: 61.6%, down from 73% pre-pandemic — a leading indicator for future graduation declines.

DISCIPLINE: Exclusion rate hit decade-high 4.55% in 2023-24, moderated to 3.61%. 6th grade highest at 7.76%.

TEACHER DIVERSITY GAP: 79.4% of teachers are White while 73.9% of students are non-White. Teacher count dropped 7.5% while class sizes jumped to 22.1.

KINDERGARTEN READINESS DECLINING: Math readiness at 56.6% in 2025-26, the lowest in a decade. All 6 WaKIDS domains declining.

BRIGHT SPOT: Dual credit participation at 78.4%, a new high.""",
    ]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
DATASET_BUILDERS = [
    # Priority order: most consequential first
    ("assessment", "OSPI Report Card — Assessment/Test Scores", build_assessment_chunks),
    ("graduation", "OSPI Report Card — Graduation Rates", build_graduation_chunks),
    ("attendance", "OSPI Report Card — Attendance", build_attendance_chunks),
    ("enrollment", "OSPI Report Card — Enrollment & Demographics", build_enrollment_chunks),
    ("discipline", "OSPI Report Card — Discipline", build_discipline_chunks),
    ("teacher", "OSPI Report Card — Teacher Data", build_teacher_chunks),
    ("sqss", "OSPI Report Card — School Quality (SQSS)", build_sqss_chunks),
    ("growth", "OSPI Report Card — Student Growth", build_growth_chunks),
    ("wakids", "OSPI Report Card — Kindergarten Readiness (WaKIDS)", build_wakids_chunks),
    ("summary", "OSPI Report Card — Key Findings Summary", build_summary_chunk),
]


async def main():
    conn = await asyncpg.connect(DSN)
    doc_ids = []
    total_chunks = 0

    for dataset_key, doc_title, builder in DATASET_BUILDERS:
        print(f"\nBuilding chunks for: {doc_title}")
        text_chunks = builder()
        if not text_chunks:
            print("  No data — skipping")
            continue

        print(f"  Generated {len(text_chunks)} chunks")

        doc_id = uuid.uuid4()
        external_id = f"ospi-reportcard-{dataset_key}"

        # Insert document
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
            SOURCE_URL,
            None,  # no specific meeting date
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

        # Check if doc was actually inserted (might already exist)
        existing = await conn.fetchval(
            "SELECT id FROM documents WHERE tenant_id=$1 AND external_id=$2",
            TENANT_ID,
            external_id,
        )
        if existing != doc_id:
            print(f"  Document already exists ({existing}), updating chunks...")
            doc_id = existing
            # Delete old chunks to replace with fresh data
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
                int(word_count * 1.3),  # approximate token count
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

    print(f"\n{'='*60}")
    print(f"Inserted {len(doc_ids)} documents, {total_chunks} chunks total")
    print("All chunks have embedding_status='pending'")
    print("Run the embedding pipeline to embed them into Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
