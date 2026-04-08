#!/usr/bin/env python3
"""Compare speaker resolution quality across LLM backends.

Tests the same transcripts against Claude Haiku, Sonnet, Opus,
local Mistral Small 3.1 24B, and local Qwen 2.5-72B-Instruct.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

TRANSCRIPT_DIR = Path("/home/qorvault/projects/ksd_forensic/output/whisperx_transcripts")
LLAMA_CLI = Path("/opt/llama.cpp-vulkan/build/bin/llama-cli")
MISTRAL_MODEL = Path("/home/qorvault/models/chat--Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf")
QWEN_MODEL = Path("/home/qorvault/models/chat--Qwen2.5-72B-Instruct-Q4_K_M.gguf")

# Test transcripts
TEST_FILES = [
    ("20240912_-_KSD_Regular_Board_Meeting_-_09_11_24.json", date(2024, 9, 11)),
    ("20210818_-_KSD_Board_Executive_Session_-_08_17_2021.json", date(2021, 8, 17)),
    (
        "20230511_-_KSD_Special_Board_Meeting__Executive_Session_and_" "Regular_Board_Meeting_-_05_10_2023.json",
        date(2023, 5, 10),
    ),
]

# Board compositions by era
COMPOSITIONS = {
    date(2024, 9, 11): {
        "members": [
            "Meghin Margel",
            "Joe Farah",
            "Tim Clark",
            "Donald Cook",
            "Hyun-Jin Song",
        ],
        "president": "Meghin Margel",
    },
    date(2021, 8, 17): {
        "members": [
            "Maya Vengadasalam",
            "Meghin Margel",
            "Donald Cook",
            "Joe Farah",
            "Leslie Hamada",
            "Tim Clark",
        ],
        "president": "Maya Vengadasalam",
    },
    date(2023, 5, 10): {
        "members": [
            "Maya Vengadasalam",
            "Meghin Margel",
            "Donald Cook",
            "Joe Farah",
            "Leslie Hamada",
            "Tim Clark",
        ],
        "president": "Maya Vengadasalam",
    },
}

OPENING_SEGMENTS = 150


def build_prompt(segments: list[dict], meeting_date: date) -> str:
    """Build speaker resolution prompt from transcript segments."""
    comp = COMPOSITIONS[meeting_date]
    board_members = comp["members"]
    president = comp["president"]
    superintendent = "Israel Vela"

    opening_parts = []
    for seg in segments[:OPENING_SEGMENTS]:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        opening_parts.append(f"[{speaker}] ({start:.1f}s): {text}")

    roll_parts = []
    for seg in segments[OPENING_SEGMENTS:]:
        text_lower = seg.get("text", "").lower()
        if any(kw in text_lower for kw in ["roll call", "aye", "nay", "motion", "second", "all in favor"]):
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0)
            text = seg.get("text", "").strip()
            roll_parts.append(f"[{speaker}] ({start:.1f}s): {text}")
            if len(roll_parts) >= 50:
                break

    opening_text = "\n".join(opening_parts)
    roll_text = "\n".join(roll_parts) if roll_parts else "(no roll call votes found)"

    return (
        f"Here is the opening of a Kent School District board meeting"
        f" on {meeting_date.isoformat()}.\n"
        f"\n"
        f"The board members at this time were: {', '.join(board_members)}\n"
        f"The board president is: {president}\n"
        f"The superintendent is: {superintendent}\n"
        f"\n"
        f"The president typically calls the meeting to order and conducts roll call.\n"
        f'During roll call, the president calls each director by name ("Director Cook?")'
        f' and they respond "Present" or "Here".\n'
        f'Roll call votes later have longer "Aye"/"Nay" responses that are more'
        f" reliably diarized.\n"
        f'Other speakers may say "Thank you, President {president.split()[-1]}"'
        f" which confirms the president's label.\n"
        f"\n"
        f"=== OPENING OF MEETING ===\n"
        f"{opening_text}\n"
        f"\n"
        f"=== ROLL CALL VOTES (later in meeting) ===\n"
        f"{roll_text}\n"
        f"\n"
        f"Map SPEAKER_XX labels to real names. Only map speakers you are confident about.\n"
        f"Return a JSON object mapping speaker labels to names, e.g.:\n"
        f'{{"SPEAKER_14": "Meghin Margel", "SPEAKER_20": "Israel Vela"}}\n'
        f"\n"
        f"Return ONLY the JSON object, no other text."
    )


def extract_json(text: str) -> dict:
    """Extract a JSON object from model output text."""
    # Strip markdown code blocks
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.rstrip("`").strip()

    # Find all JSON-like blocks and take the last one (model response, not leaked prompt)
    matches = re.findall(r"\{[^}]+\}", cleaned)
    if matches:
        return json.loads(matches[-1])
    return {}


def test_claude(prompt: str, model: str) -> tuple[dict, float]:
    """Run speaker resolution via Claude API. Returns (speaker_map, seconds)."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    response_text = resp.content[0].text.strip()
    speaker_map = extract_json(response_text)
    return speaker_map, elapsed


def test_local(prompt: str, model_path: Path, model_name: str) -> tuple[dict, float]:
    """Run speaker resolution via local llama.cpp. Returns (speaker_map, seconds)."""
    if not model_path.exists():
        print(f"  {model_name}: MODEL NOT FOUND at {model_path}")
        return {}, 0.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name

    try:
        cmd = [
            str(LLAMA_CLI),
            "-m",
            str(model_path),
            "-ngl",
            "99",
            "-f",
            prompt_path,
            "-n",
            "1024",
            "--temp",
            "0.1",
            "-c",
            "8192",
            "--no-display-prompt",
            "--single-turn",
            "--jinja",
        ]
        t0 = time.time()
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  {model_name}: llama-cli error (exit {result.returncode})")
            return {}, elapsed

        speaker_map = extract_json(result.stdout)
        return speaker_map, elapsed

    except subprocess.TimeoutExpired:
        print(f"  {model_name}: TIMEOUT (600s)")
        return {}, 600.0
    except Exception as e:
        print(f"  {model_name}: ERROR — {e}")
        return {}, 0.0
    finally:
        Path(prompt_path).unlink(missing_ok=True)


def main() -> None:
    """Run comparison across all backends."""
    backends = [
        ("Claude Haiku 4.5", "api", "claude-haiku-4-5-20251001"),
        ("Claude Sonnet 4", "api", "claude-sonnet-4-20250514"),
        ("Claude Opus 4", "api", "claude-opus-4-20250514"),
        ("Mistral Small 3.1 24B (Q8)", "local", str(MISTRAL_MODEL)),
        ("Qwen 2.5-72B-Instruct (Q4_K_M)", "local", str(QWEN_MODEL)),
    ]

    all_results: dict[str, dict] = {}

    for fname, meeting_date in TEST_FILES:
        path = TRANSCRIPT_DIR / fname
        if not path.exists():
            print(f"\nSKIPPING {fname} — not found")
            continue

        data = json.loads(path.read_text())
        segments = data.get("segments", [])
        unique_speakers = sorted(set(s.get("speaker") for s in segments if s.get("speaker")))

        print(f"\n{'=' * 80}")
        print(f"Transcript: {fname}")
        print(f"Date: {meeting_date}, Segments: {len(segments)}, Speakers: {len(unique_speakers)}")
        print(f"{'=' * 80}")

        prompt = build_prompt(segments, meeting_date)
        print(f"Prompt length: {len(prompt)} chars")

        meeting_results = {}

        for name, backend_type, model_id in backends:
            print(f"\n  Testing: {name}...")
            try:
                if backend_type == "api":
                    speaker_map, elapsed = test_claude(prompt, model_id)
                else:
                    speaker_map, elapsed = test_local(prompt, Path(model_id), name)

                meeting_results[name] = {
                    "speakers_resolved": len(speaker_map),
                    "elapsed_seconds": round(elapsed, 1),
                    "mapping": speaker_map,
                }

                print(f"    Resolved: {len(speaker_map)} speakers in {elapsed:.1f}s")
                for spk_label, spk_name in sorted(speaker_map.items()):
                    print(f"      {spk_label} → {spk_name}")

            except Exception as e:
                print(f"    ERROR: {e}")
                meeting_results[name] = {
                    "speakers_resolved": 0,
                    "elapsed_seconds": 0,
                    "mapping": {},
                    "error": str(e),
                }

        all_results[fname] = meeting_results

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    backend_names = [name for name, _, _ in backends]
    header = f"{'Transcript':<45} | " + " | ".join(f"{n:>15}" for n in backend_names)
    print("\nSpeakers resolved:")
    print(header)
    print("-" * len(header))

    for fname, meeting_date in TEST_FILES:
        if fname not in all_results:
            continue
        meeting_res = all_results[fname]
        short = f"{meeting_date} ({fname[:30]}...)"
        counts = []
        for name in backend_names:
            r = meeting_res.get(name, {})
            count = r.get("speakers_resolved", 0)
            elapsed = r.get("elapsed_seconds", 0)
            counts.append(f"{count:>3} ({elapsed:>5.1f}s)")
        print(f"{short:<45} | " + " | ".join(f"{c:>15}" for c in counts))

    # Agreement analysis
    print("\n\nAgreement analysis (where models agree on speaker identity):")
    for fname, meeting_date in TEST_FILES:
        if fname not in all_results:
            continue
        print(f"\n  {meeting_date}:")
        meeting_res = all_results[fname]

        # Collect all speaker labels mentioned across all backends
        all_labels = set()
        for name in backend_names:
            mapping = meeting_res.get(name, {}).get("mapping", {})
            all_labels.update(mapping.keys())

        for label in sorted(all_labels):
            names_by_backend = {}
            for name in backend_names:
                mapping = meeting_res.get(name, {}).get("mapping", {})
                if label in mapping:
                    names_by_backend[name] = mapping[label]

            resolved_names = list(names_by_backend.values())
            unique_names = set(resolved_names)

            if len(unique_names) == 1 and len(resolved_names) > 1:
                consensus = f"CONSENSUS: {resolved_names[0]}"
            elif len(unique_names) > 1:
                consensus = "DISAGREEMENT"
            else:
                consensus = ""

            assignments = ", ".join(f"{bk[:12]}={nm}" for bk, nm in names_by_backend.items())
            print(f"    {label}: {assignments}  [{consensus}]")

    # Save raw results
    out_path = Path(__file__).parent / "speaker_resolution_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
