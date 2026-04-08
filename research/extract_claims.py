#!/usr/bin/env python3
"""Extract verifiable factual claims from board meeting transcripts using Claude API."""

import asyncio
import json
from pathlib import Path

# Load env
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
OUTPUT_FILE = Path(__file__).parent / "extracted_claims.json"

EXTRACTION_PROMPT = """You are analyzing a school board meeting transcript to extract specific verifiable factual claims made by district administration or board majority members.

Extract ONLY claims that are:
1. Specific and verifiable (contain numbers, dates, percentages, named programs, or specific commitments)
2. Made by an identified speaker (not anonymous public commenters)
3. Related to budget, finance, enrollment, staffing levels, program status, or district performance

For each claim extract:
- speaker_name: Name of the speaker
- speaker_role: Their role if identifiable (superintendent, board member, CFO, etc.)
- quote: Exact quote or close paraphrase
- timestamp: Timestamp in the transcript (HH:MM:SS if available, or position indicator like "early/middle/late in meeting")
- category: One of FINANCIAL_FIGURE, ENROLLMENT_CLAIM, PROGRAM_STATUS, COMMITMENT, PROCESS_CLAIM
- verifiable_element: The specific thing that could be checked against records

Return as a JSON array. Be precise — only include claims where there is something specific to verify. Do not include vague statements like "we are working hard on this."

TRANSCRIPT:
"""

# Max chars to send per transcript — Claude opus context is large but let's be safe
MAX_CHARS = 180000


async def extract_from_transcript(client, transcript_id: str) -> dict:
    meta_file = TRANSCRIPTS_DIR / f"{transcript_id}.meta"
    text_file = TRANSCRIPTS_DIR / f"{transcript_id}.txt"

    meta = meta_file.read_text().strip().split("|")
    title = meta[0] if meta else "Unknown"
    meeting_date = meta[1] if len(meta) > 1 else "Unknown"

    content = text_file.read_text()
    if len(content) > MAX_CHARS:
        # Truncate but note it
        content = content[:MAX_CHARS] + "\n\n[TRANSCRIPT TRUNCATED AT CHARACTER LIMIT]"

    print(f"  Processing: {title} ({meeting_date}) — {len(content):,} chars")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + content}],
        )

        response_text = response.content[0].text

        # Extract JSON from response (might be wrapped in markdown code block)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        claims = json.loads(response_text)

        # Add metadata to each claim
        for claim in claims:
            claim["transcript_id"] = transcript_id
            claim["meeting_title"] = title
            claim["meeting_date"] = meeting_date

        print(f"    → {len(claims)} claims extracted")
        return {"transcript_id": transcript_id, "title": title, "meeting_date": meeting_date, "claims": claims}

    except json.JSONDecodeError as e:
        print(f"    → JSON parse error: {e}")
        print(f"    → Raw response: {response_text[:500]}")
        return {
            "transcript_id": transcript_id,
            "title": title,
            "meeting_date": meeting_date,
            "claims": [],
            "error": str(e),
        }
    except Exception as e:
        print(f"    → Error: {e}")
        return {
            "transcript_id": transcript_id,
            "title": title,
            "meeting_date": meeting_date,
            "claims": [],
            "error": str(e),
        }


def main():
    client = anthropic.Anthropic()

    # Get all transcript IDs
    transcript_ids = [f.stem for f in TRANSCRIPTS_DIR.glob("*.txt")]
    transcript_ids.sort()

    print(f"Found {len(transcript_ids)} transcripts")

    all_results = []
    total_claims = 0
    category_counts = {}

    for tid in transcript_ids:
        result = asyncio.run(extract_from_transcript(client, tid))
        all_results.append(result)

        for claim in result.get("claims", []):
            total_claims += 1
            cat = claim.get("category", "UNKNOWN")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Save results
    output = {"total_claims": total_claims, "category_breakdown": category_counts, "transcripts": all_results}

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nTotal claims extracted: {total_claims}")
    print(f"Category breakdown: {json.dumps(category_counts, indent=2)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
