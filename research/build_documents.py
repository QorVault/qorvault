#!/usr/bin/env python3
"""Build the meeting intelligence document and cheat sheet from extracted contradictions."""

import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

CONTRADICTIONS_FILE = Path(__file__).parent / "contradictions.json"
CLAIMS_FILE = Path(__file__).parent / "extracted_claims.json"
KEYDOCS_DIR = Path(__file__).parent / "keydocs"

today = date.today().isoformat()
INTEL_FILE = Path(__file__).parent / f"meeting_intelligence_{today}.md"
CHEAT_FILE = Path(__file__).parent / f"cheat_sheet_{today}.md"


def load_key_doc_excerpts() -> str:
    """Load key financial figures from budget documents for the quick reference section."""
    docs = {}
    for f in sorted(KEYDOCS_DIR.glob("*.txt")):
        docs[f.stem] = f.read_text()[:15000]
    return docs


def build_intel_doc():
    data = json.loads(CONTRADICTIONS_FILE.read_text())
    claims_data = json.loads(CLAIMS_FILE.read_text())
    contradictions = data["contradictions"]

    high = [c for c in contradictions if c["contradiction"]["confidence"] == "HIGH"]
    medium = [c for c in contradictions if c["contradiction"]["confidence"] == "MEDIUM"]
    low = [c for c in contradictions if c["contradiction"]["confidence"] == "LOW"]

    lines = []
    lines.append(f"# KSD Board Meeting Intelligence Brief — {today}")
    lines.append("")
    lines.append("**Prepared for tonight's board meeting on the budget shortfall.**")
    lines.append(
        "**Based on analysis of 7 board meeting transcripts (Dec 2023 – Jun 2025) cross-referenced against official budget documents.**"
    )
    lines.append("")
    lines.append(f"Total claims analyzed: {data['claims_analyzed']}")
    lines.append(
        f"Contradictions found: {data['total_contradictions']} (HIGH: {data['by_confidence']['HIGH']}, MEDIUM: {data['by_confidence']['MEDIUM']}, LOW: {data['by_confidence']['LOW']})"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: HIGH confidence
    lines.append("## Section 1: HIGH CONFIDENCE CONTRADICTIONS")
    lines.append("")
    for i, c in enumerate(high, 1):
        claim = c["claim"]
        cont = c["contradiction"]
        lines.append(f"### Contradiction #{i}")
        lines.append("")
        lines.append(f'**CLAIM:** "{claim["quote"]}"')
        lines.append(f'**Speaker:** {claim["speaker_name"]}, {claim.get("speaker_role", "Unknown role")}')
        lines.append(f'**Meeting:** {claim["meeting_date"]} — Timestamp: {claim.get("timestamp", "Unknown")}')
        lines.append(
            f'**Video:** {claim["meeting_date"]} recording — {claim.get("timestamp", "see transcript")} — USE THIS TO ROLL THE TAPE'
        )
        lines.append("")
        lines.append(f'**RECORD SHOWS:** {cont["evidence_details"]}')
        lines.append(f'**Source:** {cont["evidence_source"]}')
        lines.append("")
        lines.append(f'**RESPONSE READY:** "{cont["summary"]}"')
        lines.append("")
        lines.append("---")
        lines.append("")

    # Section 2: MEDIUM confidence
    lines.append("## Section 2: MEDIUM CONFIDENCE CONTRADICTIONS")
    lines.append("")
    for i, c in enumerate(medium, 1):
        claim = c["claim"]
        cont = c["contradiction"]
        lines.append(f"### Inconsistency #{i}")
        lines.append("")
        lines.append(f'**CLAIM:** "{claim["quote"]}"')
        lines.append(f'**Speaker:** {claim["speaker_name"]}, {claim.get("speaker_role", "Unknown role")}')
        lines.append(f'**Meeting:** {claim["meeting_date"]} — Timestamp: {claim.get("timestamp", "Unknown")}')
        lines.append("")
        lines.append(f'**INCONSISTENCY:** {cont["evidence_details"]}')
        lines.append(f'**Source:** {cont["evidence_source"]}')
        lines.append("")
        lines.append(f'**RESPONSE READY:** "{cont["summary"]}"')
        lines.append("")
        lines.append("---")
        lines.append("")

    # Section 3: Claims to watch tonight
    lines.append("## Section 3: CLAIMS TO WATCH TONIGHT")
    lines.append("")
    lines.append(
        "Based on patterns from previous meetings, these are the most likely claims to be made tonight with pre-prepared responses:"
    )
    lines.append("")

    # Use Claude to generate predictions based on patterns
    client = anthropic.Anthropic()

    # Gather all claims for pattern analysis
    all_claims_text = json.dumps(
        [
            {
                "speaker": c["claim"]["speaker_name"],
                "quote": c["claim"]["quote"][:200],
                "date": c["claim"]["meeting_date"],
                "category": c["claim"]["category"],
            }
            for c in contradictions
        ],
        indent=2,
    )

    # Load key recent doc for current figures
    budget_update = (KEYDOCS_DIR / "Budget_Update_2.4.26.txt").read_text()[:10000]
    fin_stmt = (KEYDOCS_DIR / "Financial_Statement_Dec_2025.txt").read_text()[:10000]

    prediction_prompt = f"""Based on these patterns of claims made in past Kent School District board meetings about the budget shortfall, predict the 5 most likely claims to be made TONIGHT and provide a documentary response for each.

PAST CLAIM PATTERNS:
{all_claims_text}

CURRENT BUDGET FIGURES (Feb 4, 2026 Budget Update):
{budget_update}

CURRENT FINANCIAL STATEMENT (Dec 2025):
{fin_stmt}

For each predicted claim, provide:
1. The likely claim (what they'll probably say)
2. The documentary response (what the records actually show)
3. The source document

Format as numbered list with clear headers. Be specific with numbers."""

    print("Generating predictions for tonight...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prediction_prompt}],
    )
    lines.append(response.content[0].text)
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 4: Quick Reference Key Figures
    lines.append("## Section 4: QUICK REFERENCE — KEY FIGURES")
    lines.append("")

    figures_prompt = f"""From these official Kent School District budget documents, extract a reference table of the most important financial figures. Include:

1. Current year (FY2025-26) projected deficit
2. Ending fund balance percentage (current and projected)
3. Enrollment figures (most recent 5 years if available)
4. Monthly operating cost (burn rate)
5. Personnel costs as % of budget
6. ESSER/federal funds at risk
7. Levy amounts and expiration dates
8. Special education overspending
9. Any figures that have changed significantly between presentations

Format as a markdown table with columns: Figure | Value | Source Document | Date

BUDGET UPDATE (Feb 4, 2026):
{budget_update}

FINANCIAL STATEMENT (Dec 2025):
{fin_stmt}"""

    print("Generating key figures table...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": figures_prompt}],
    )
    lines.append(response.content[0].text)
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 5: RAG Quick Queries
    lines.append("## Section 5: RAG QUICK QUERIES")
    lines.append("")
    lines.append("Pre-tested queries to run during the meeting via the RAG API at http://127.0.0.1:8000/api/v1/query")
    lines.append("")
    lines.append("```bash")
    lines.append("# Usage: curl -s -X POST http://127.0.0.1:8000/api/v1/query \\")
    lines.append('#   -H "Content-Type: application/json" \\')
    lines.append('#   -d \'{"query": "YOUR QUERY HERE", "rewrite_query": true}\'')
    lines.append("```")
    lines.append("")

    queries = [
        "What is the current projected deficit for Kent School District FY2025-26?",
        "What is the ending fund balance percentage for Kent School District?",
        "How much does Kent School District spend on special education versus state funding?",
        "What are Kent School District enrollment trends over the past 5 years?",
        "What federal funding is at risk of being cut for Kent School District?",
        "What is the monthly operating cost (burn rate) for Kent School District general fund?",
        "What budget cuts has Kent School District proposed for 2025-26?",
        "When does the Kent School District EP&O levy expire and what is the renewal plan?",
        "What is the MSOC underfunding gap for Kent School District?",
        "What deferred maintenance has Kent School District designated and how has it changed?",
    ]

    for i, q in enumerate(queries, 1):
        lines.append(f"{i}. **{q}**")
        lines.append("   ```")
        lines.append(
            f'   curl -s -X POST http://127.0.0.1:8000/api/v1/query -H "Content-Type: application/json" -d \'{{"query": "{q}", "rewrite_query": true}}\' | python3 -m json.tool | head -30'
        )
        lines.append("   ```")
        lines.append("")

    # Write the full intel doc
    INTEL_FILE.write_text("\n".join(lines))
    print(f"Intelligence document saved: {INTEL_FILE} ({len('\n'.join(lines)):,} chars)")

    # Build the cheat sheet (condensed one-pager)
    cheat = []
    cheat.append(f"# KSD Budget Meeting Cheat Sheet — {today}")
    cheat.append("")
    cheat.append("## HIGH CONFIDENCE CONTRADICTIONS")
    cheat.append("")

    for i, c in enumerate(high, 1):
        claim = c["claim"]
        cont = c["contradiction"]
        cheat.append(f"**{i}. {claim['speaker_name']} ({claim['meeting_date']}):**")
        cheat.append(f'Said: "{claim["quote"][:150]}..."')
        cheat.append(f'Record: {cont["summary"][:200]}')
        cheat.append(f'Source: {cont["evidence_source"]}')
        cheat.append("")

    cheat.append("---")
    cheat.append("")
    cheat.append("## KEY FIGURES (from official documents)")
    cheat.append("")
    cheat.append("| Metric | Value | Source |")
    cheat.append("|--------|-------|--------|")
    cheat.append("| FY25-26 Operating Deficit | $22.8M (revised Dec 2025) | Budget Update Dec 2025 |")
    cheat.append("| FY25-26 Revenue (revised) | $541.2M | Budget Update Dec 2025 |")
    cheat.append("| Monthly Burn Rate | ~$47M/month | Budget Update Dec/Jan |")
    cheat.append("| Enrollment (FY24-25 actual) | 23,902 AAFTE | Financial Stmt Dec 2025 |")
    cheat.append("| Enrollment (FY25-26 revised) | 23,430 AAFTE | Financial Stmt Dec 2025 |")
    cheat.append("| Personnel Costs | ~82% of budget | Budget Update Dec 2025 |")
    cheat.append("| Special Ed Overspend (FY25) | $8.3M over budget | Budget Update Dec 2025 |")
    cheat.append("| EP&O Levy Expiry | Dec 31, 2027 | Transcript Jun 2025 |")
    cheat.append("| Budget Cuts Needed (FY26-27) | $35M | Budget Update Dec 2025 |")
    cheat.append("| Federal Funds at Risk | ~$17M (Title I/II/III/IV) | Budget Update Dec 2025 |")
    cheat.append("")
    cheat.append("---")
    cheat.append("")
    cheat.append("## WATCH FOR TONIGHT")
    cheat.append("")
    cheat.append("- Deficit figures: Was $14.9M in Jun 2025, now $22.8M+ — watch for use of outdated number")
    cheat.append("- Enrollment: May claim '25,000+ students' — actual is 23,430")
    cheat.append("- Monthly costs: Was $45M in Aug 2024, now $47M — watch for old figure")
    cheat.append("- Fund balance: Aug 2024 projected $46.2M, actual was $58.75M — projections were wrong")
    cheat.append("- Budget cuts needed: Was $7.5M/yr in Jun 2025, now $35M in one year — massive escalation")
    cheat.append("")

    CHEAT_FILE.write_text("\n".join(cheat))
    print(f"Cheat sheet saved: {CHEAT_FILE} ({len('\n'.join(cheat)):,} chars)")


if __name__ == "__main__":
    build_intel_doc()
