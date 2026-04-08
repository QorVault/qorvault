#!/usr/bin/env python3
"""Retry contradiction analysis for transcripts that were rate-limited."""

import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

CLAIMS_FILE = Path(__file__).parent / "extracted_claims.json"
KEYDOCS_DIR = Path(__file__).parent / "keydocs"
OUTPUT_FILE = Path(__file__).parent / "contradictions.json"

# Already processed transcript IDs (Aug 2024, Jun 2025)
PROCESSED_IDS = {
    "250ea6dc-ad2f-42a7-9663-f5e04de50907",  # Aug 2024
    "289328b0-20e3-41c6-b475-d858062de03d",  # Jun 2025
}

QUERY_TEMPLATES = {
    "FINANCIAL_FIGURE": "Kent School District {verifiable_element} budget financial figure",
    "ENROLLMENT_CLAIM": "Kent School District enrollment {verifiable_element}",
    "COMMITMENT": "Kent School District {verifiable_element} follow-up action",
    "PROGRAM_STATUS": "Kent School District {verifiable_element} program status",
    "PROCESS_CLAIM": "Kent School District {verifiable_element} process procedure",
}


def load_key_documents() -> str:
    docs = []
    for f in sorted(KEYDOCS_DIR.glob("*.txt")):
        content = f.read_text()
        if len(content) > 20000:
            content = content[:20000] + "\n[...TRUNCATED...]"
        docs.append(f"=== DOCUMENT: {f.stem} ===\n{content}")
    return "\n\n".join(docs)


def analyze_batch(client, claims, key_docs_text, batch_label):
    claims_text = json.dumps(claims, indent=2)

    prompt = f"""You are analyzing school board meeting claims against official budget documents.

Find contradictions between what officials SAID and what the DOCUMENTS show. Focus on numbers that don't match, commitments not followed, and claims that conflict with official reports.

For each contradiction, provide JSON with: claim_index, confidence (HIGH/MEDIUM/LOW), contradiction_summary (one sentence for board meeting use), evidence_details, evidence_source.

Return a JSON array. Empty array if no contradictions.

CLAIMS:
{claims_text}

KEY DOCUMENTS:
{key_docs_text}"""

    print(f"  Analyzing: {batch_label} ({len(claims)} claims)...")

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
        print(f"    Parse error: {e}")
        return []

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
            print(f"    [{r.get('confidence')}] {r.get('contradiction_summary', '')[:100]}")
    return enriched


def main():
    client = anthropic.Anthropic()
    claims_data = json.loads(CLAIMS_FILE.read_text())
    existing = json.loads(OUTPUT_FILE.read_text())
    key_docs_text = load_key_documents()

    new_contradictions = []

    for transcript in claims_data["transcripts"]:
        tid = transcript.get("transcript_id", "")
        if tid in PROCESSED_IDS:
            print(f"Skipping already-processed: {transcript.get('title')}")
            continue

        claims = transcript.get("claims", [])
        if not claims:
            continue

        title = transcript.get("title", "Unknown")
        date = transcript.get("meeting_date", "Unknown")

        batch_size = 40
        for i in range(0, len(claims), batch_size):
            batch = claims[i : i + batch_size]
            label = f"{title} ({date}) [{i+1}-{i+len(batch)}]"

            try:
                results = analyze_batch(client, batch, key_docs_text, label)
                new_contradictions.extend(results)
            except anthropic.RateLimitError:
                print("    Rate limited. Waiting 90s...")
                time.sleep(90)
                try:
                    results = analyze_batch(client, batch, key_docs_text, label)
                    new_contradictions.extend(results)
                except Exception as e:
                    print(f"    Failed again: {e}")
            except Exception as e:
                print(f"    Error: {e}")

            # Rate limit: wait between batches
            print("    Waiting 70s for rate limit...")
            time.sleep(70)

    # Merge with existing
    all_contradictions = existing["contradictions"] + new_contradictions
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
    print(f"\nDone. {len(new_contradictions)} new contradictions. Total: {len(all_contradictions)}")
    print(
        f"  HIGH: {output['by_confidence']['HIGH']}, MEDIUM: {output['by_confidence']['MEDIUM']}, LOW: {output['by_confidence']['LOW']}"
    )


if __name__ == "__main__":
    main()
