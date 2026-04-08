#!/usr/bin/env python3
"""Batch contradiction analysis — send claims + key documents to Claude in groups."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

CLAIMS_FILE = Path(__file__).parent / "extracted_claims.json"
KEYDOCS_DIR = Path(__file__).parent / "keydocs"
OUTPUT_FILE = Path(__file__).parent / "contradictions.json"


def load_key_documents() -> str:
    """Load and concatenate key budget documents."""
    docs = []
    for f in sorted(KEYDOCS_DIR.glob("*.txt")):
        content = f.read_text()
        if len(content) > 20000:
            content = content[:20000] + "\n[...TRUNCATED...]"
        docs.append(f"=== DOCUMENT: {f.stem} ===\n{content}")
    return "\n\n".join(docs)


def analyze_batch(client: anthropic.Anthropic, claims: list[dict], key_docs_text: str, batch_label: str) -> list[dict]:
    """Send a batch of claims + key documents to Claude for contradiction analysis."""
    claims_text = json.dumps(claims, indent=2)

    prompt = f"""You are a civic accountability analyst preparing for a school board meeting tonight. You have two inputs:

1. CLAIMS: Specific factual claims made by district officials in past board meetings
2. KEY DOCUMENTS: Official budget documents, financial statements, and superintendent communications

Your job: Find contradictions between what officials SAID and what the DOCUMENTS show.

Focus on:
- Numbers that don't match (e.g., claimed deficit vs. actual deficit in financial statements)
- Commitments that weren't followed through
- Claims about enrollment, staffing, or program status that conflict with official reports
- Inconsistencies between what was said at different meetings
- Financial figures that changed without explanation between presentations

For each contradiction found, provide:
- claim_index: index into the claims array
- confidence: "HIGH" (direct numerical contradiction or provably false), "MEDIUM" (inconsistent framing, unexplained changes), or "LOW" (circumstantial tension)
- contradiction_summary: One professional sentence suitable for reading aloud at a board meeting, citing the specific document
- evidence_details: What the documents actually show, with specific numbers/dates
- evidence_source: Which key document contains the contradicting evidence

Return a JSON array of contradiction objects. If no contradictions found, return an empty array.
Be thorough but precise — only flag genuine contradictions, not reasonable rounding or scope differences.

CLAIMS:
{claims_text}

KEY DOCUMENTS:
{key_docs_text}"""

    print(f"  Analyzing batch: {batch_label} ({len(claims)} claims)...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        results = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        print(f"    Raw: {text[:300]}")
        return []

    # Enrich results with full claim data
    enriched = []
    for r in results:
        idx = r.get("claim_index", 0)
        if idx < len(claims):
            claim = claims[idx]
            enriched.append(
                {
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
                        "confidence": r.get("confidence"),
                        "summary": r.get("contradiction_summary"),
                        "evidence_details": r.get("evidence_details"),
                        "evidence_source": r.get("evidence_source"),
                    },
                }
            )
            conf = r.get("confidence", "?")
            print(f"    [{conf}] {r.get('contradiction_summary', '')[:100]}")

    return enriched


def main():
    client = anthropic.Anthropic()
    claims_data = json.loads(CLAIMS_FILE.read_text())

    # Load key documents (truncated to fit context)
    key_docs_text = load_key_documents()
    print(f"Key documents loaded: {len(key_docs_text):,} chars")

    # Group claims by transcript
    all_contradictions = []

    for transcript in claims_data["transcripts"]:
        claims = transcript.get("claims", [])
        if not claims:
            continue

        title = transcript.get("title", "Unknown")
        date = transcript.get("meeting_date", "Unknown")
        label = f"{title} ({date})"

        # Split into batches of ~40 claims to stay within context
        batch_size = 40
        for i in range(0, len(claims), batch_size):
            batch = claims[i : i + batch_size]
            batch_label = f"{label} [{i+1}-{i+len(batch)}]"

            try:
                contradictions = analyze_batch(client, batch, key_docs_text, batch_label)
                all_contradictions.extend(contradictions)
            except Exception as e:
                print(f"    Error: {e}")

    # Save results
    output = {
        "total_contradictions": len(all_contradictions),
        "claims_analyzed": claims_data["total_claims"],
        "by_confidence": {
            "HIGH": sum(1 for c in all_contradictions if c["contradiction"]["confidence"] == "HIGH"),
            "MEDIUM": sum(1 for c in all_contradictions if c["contradiction"]["confidence"] == "MEDIUM"),
            "LOW": sum(1 for c in all_contradictions if c["contradiction"]["confidence"] == "LOW"),
        },
        "contradictions": all_contradictions,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nDone. {len(all_contradictions)} contradictions found.")
    print(f"  HIGH: {output['by_confidence']['HIGH']}")
    print(f"  MEDIUM: {output['by_confidence']['MEDIUM']}")
    print(f"  LOW: {output['by_confidence']['LOW']}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
