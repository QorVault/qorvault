#!/usr/bin/env python3
"""Find contradictions between extracted claims and documentary evidence via RAG."""

import json
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

RAG_API = "http://127.0.0.1:8000/api/v1/query"
CLAIMS_FILE = Path(__file__).parent / "extracted_claims.json"
OUTPUT_FILE = Path(__file__).parent / "contradictions.json"

# Query templates by category
QUERY_TEMPLATES = {
    "FINANCIAL_FIGURE": "Kent School District {verifiable_element} budget financial figure official document",
    "ENROLLMENT_CLAIM": "Kent School District enrollment {verifiable_element} official report data",
    "COMMITMENT": "Kent School District {verifiable_element} follow-up action report status",
    "PROGRAM_STATUS": "Kent School District {verifiable_element} program status report",
    "PROCESS_CLAIM": "Kent School District {verifiable_element} process policy procedure",
}

CONTRADICTION_PROMPT = """You are analyzing whether documentary evidence contradicts a specific claim made in a Kent School District board meeting.

CLAIM made on {meeting_date} by {speaker_name} ({speaker_role}):
"{quote}"

Verifiable element: {verifiable_element}

DOCUMENTARY EVIDENCE retrieved from official records:
{evidence}

Analyze whether the evidence CONTRADICTS, SUPPORTS, or is INCONCLUSIVE regarding this claim.

If there IS a contradiction, respond with JSON:
{{
  "has_contradiction": true,
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "contradiction_summary": "One sentence summary of the contradiction suitable for reading aloud at a board meeting — factual, professional, citing the specific document",
  "evidence_details": "What the documents actually show, with specific numbers/dates",
  "evidence_source": "Document title and date of the contradicting evidence"
}}

HIGH = direct numerical contradiction or proven false statement
MEDIUM = inconsistent framing, different numbers for similar metrics, or missing follow-through on commitment
LOW = circumstantial tension worth investigating

If there is NO contradiction (evidence supports or is inconclusive), respond with:
{{
  "has_contradiction": false,
  "notes": "Brief note on what the evidence shows"
}}

Respond ONLY with the JSON object, no other text."""


def build_rag_query(claim: dict) -> str:
    """Build a targeted RAG query for a claim."""
    category = claim.get("category", "PROCESS_CLAIM")
    template = QUERY_TEMPLATES.get(category, QUERY_TEMPLATES["PROCESS_CLAIM"])
    verifiable = claim.get("verifiable_element", claim.get("quote", "")[:100])
    return template.format(verifiable_element=verifiable)


def query_rag(query: str) -> dict:
    """Query the RAG API."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(RAG_API, json={"query": query, "rewrite_query": True})
        resp.raise_for_status()
        return resp.json()


def check_contradiction(claim: dict, rag_result: dict, claude_client: anthropic.Anthropic) -> dict | None:
    """Use Claude to check if RAG evidence contradicts the claim."""
    # Build evidence text from RAG citations
    evidence_parts = []
    for i, citation in enumerate(rag_result.get("citations", [])[:5], 1):
        evidence_parts.append(
            f"Document {i}: {citation.get('title', 'Unknown')} "
            f"(Meeting: {citation.get('meeting_date', 'Unknown')})\n"
            f"Content: {citation.get('text', citation.get('chunk_text', 'N/A'))[:1000]}"
        )

    if not evidence_parts:
        return None

    evidence_text = "\n\n".join(evidence_parts)

    prompt = CONTRADICTION_PROMPT.format(
        meeting_date=claim.get("meeting_date", "Unknown"),
        speaker_name=claim.get("speaker_name", "Unknown"),
        speaker_role=claim.get("speaker_role", "Unknown"),
        quote=claim.get("quote", ""),
        verifiable_element=claim.get("verifiable_element", ""),
        evidence=evidence_text,
    )

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        print(f"      Error analyzing contradiction: {e}")
        return None


def main():
    claims_data = json.loads(CLAIMS_FILE.read_text())
    claude_client = anthropic.Anthropic()

    all_claims = []
    for transcript in claims_data["transcripts"]:
        for claim in transcript.get("claims", []):
            all_claims.append(claim)

    # Prioritize: FINANCIAL_FIGURE and COMMITMENT first, then others
    priority_order = ["FINANCIAL_FIGURE", "COMMITMENT", "ENROLLMENT_CLAIM", "PROGRAM_STATUS", "PROCESS_CLAIM"]
    all_claims.sort(
        key=lambda c: priority_order.index(c.get("category", "PROCESS_CLAIM"))
        if c.get("category") in priority_order
        else 99
    )

    print(f"Processing {len(all_claims)} claims...")

    contradictions = []
    checked = 0
    skipped = 0

    for i, claim in enumerate(all_claims):
        checked += 1
        cat = claim.get("category", "?")
        speaker = claim.get("speaker_name", "?")
        verifiable = claim.get("verifiable_element", "?")[:60]

        print(f"  [{checked}/{len(all_claims)}] {cat}: {speaker} — {verifiable}")

        # Query RAG
        try:
            query = build_rag_query(claim)
            rag_result = query_rag(query)
        except Exception as e:
            print(f"    RAG query failed: {e}")
            continue

        # Check if RAG returned anything useful
        if not rag_result.get("citations"):
            skipped += 1
            continue

        # Analyze for contradiction
        result = check_contradiction(claim, rag_result, claude_client)
        if result and result.get("has_contradiction"):
            contradiction = {
                "claim": {
                    "speaker_name": claim.get("speaker_name"),
                    "speaker_role": claim.get("speaker_role"),
                    "quote": claim.get("quote"),
                    "timestamp": claim.get("timestamp"),
                    "meeting_date": claim.get("meeting_date"),
                    "meeting_title": claim.get("meeting_title"),
                    "transcript_id": claim.get("transcript_id"),
                    "category": claim.get("category"),
                    "verifiable_element": claim.get("verifiable_element"),
                },
                "contradiction": {
                    "confidence": result.get("confidence"),
                    "summary": result.get("contradiction_summary"),
                    "evidence_details": result.get("evidence_details"),
                    "evidence_source": result.get("evidence_source"),
                },
                "rag_query": query,
            }
            contradictions.append(contradiction)
            conf = result.get("confidence", "?")
            print(f"    *** CONTRADICTION ({conf}): {result.get('contradiction_summary', '')[:80]}")

        # Brief pause to avoid hammering the API
        time.sleep(0.5)

        # Save intermediate results every 20 claims
        if checked % 20 == 0:
            _save(contradictions, checked, len(all_claims))

    _save(contradictions, checked, len(all_claims))
    print(
        f"\nDone. {len(contradictions)} contradictions found from {checked} claims checked ({skipped} had no RAG results)."
    )


def _save(contradictions, checked, total):
    output = {
        "total_contradictions": len(contradictions),
        "claims_checked": checked,
        "by_confidence": {
            "HIGH": sum(1 for c in contradictions if c["contradiction"]["confidence"] == "HIGH"),
            "MEDIUM": sum(1 for c in contradictions if c["contradiction"]["confidence"] == "MEDIUM"),
            "LOW": sum(1 for c in contradictions if c["contradiction"]["confidence"] == "LOW"),
        },
        "contradictions": contradictions,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"  [Saved {len(contradictions)} contradictions after {checked}/{total} claims]")


if __name__ == "__main__":
    main()
