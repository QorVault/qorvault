"""Query routing — classifies queries into rag, database, or hybrid routes.

Uses Claude Sonnet for classification and spaCy NER for PERSON entity
detection, following the pattern from scripts/name_frequency.py.
"""

from __future__ import annotations

import json
import logging
import re
import time

import anthropic

logger = logging.getLogger(__name__)

ROUTER_MODEL = "claude-sonnet-4-6"

ROUTER_SYSTEM_PROMPT = """\
You are a query router for a school district document search system.
Classify the user's question into one of three routes:

1. "rag" — Questions requiring semantic understanding of document content.
   Examples: "What was discussed about school safety?", "Summarize the superintendent's goals"

2. "database" — Questions that can be answered with structured data from the database.
   The database has: documents (title, document_type, meeting_date, committee_name, source_url),
   chunks (content, token_count), tenants. document_type is one of: agenda_item, agenda, attachment.
   Examples: "How many meetings were there in 2024?", "List all Budget Committee documents"

3. "hybrid" — Questions needing both semantic search and structured data analysis.
   Examples: "What budget topics were discussed most frequently in 2023?",
   "Compare what the superintendent said in January vs June 2025"

Return JSON: {"route": "rag"|"database"|"hybrid", "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "extracted_filters": {"date_from": null, "date_to": null, "committee_name": null,
    "document_type": null, "person_names": []}}

Rules:
- Default to "rag" when uncertain
- "database" only for questions answerable by counting, listing, or filtering metadata
- "hybrid" when the question needs both semantic content and structured aggregation
- Extract any dates, committees, document types, or person names mentioned in the query"""

# ---------------------------------------------------------------------------
# spaCy lazy singleton — loaded once, reused across requests
# ---------------------------------------------------------------------------

# Names that spaCy frequently misclassifies as PERSON
_FALSE_POSITIVES = {
    "board",
    "director",
    "superintendent",
    "president",
    "council",
    "committee",
    "state",
    "washington",
    "kent",
    "district",
    "school",
    "staff",
    "public",
    "speaker",
    "member",
}

_MIN_NAME_LEN = 4
_MAX_NAME_LEN = 40

_nlp = None


def _get_nlp():
    """Load the best available spaCy model (lazy singleton)."""
    global _nlp  # noqa: PLW0603
    if _nlp is not None:
        return _nlp

    try:
        import spacy
    except ImportError:
        logger.warning("spaCy not installed — NER disabled for query routing")
        return None

    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            _nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
            logger.info("Loaded spaCy model: %s", model_name)
            return _nlp
        except OSError:
            continue

    logger.warning("No spaCy English model found — NER disabled for query routing")
    return None


def _normalize_name(name: str) -> str | None:
    """Normalize a name or return None if it should be filtered out."""
    name = name.strip()
    if not name:
        return None
    name = name.title()
    if len(name) < _MIN_NAME_LEN or len(name) > _MAX_NAME_LEN:
        return None
    if name.lower() in _FALSE_POSITIVES:
        return None
    words = name.split()
    if len(words) == 1 and words[0].lower() in _FALSE_POSITIVES:
        return None
    return name


def _extract_person_names(text: str) -> list[str]:
    """Extract PERSON entities from text using spaCy NER."""
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    names = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = _normalize_name(ent.text)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def classify_query(client: anthropic.Anthropic, user_query: str) -> dict:
    """Classify a user query into rag, database, or hybrid route.

    Args:
        client: An initialized Anthropic client instance.
        user_query: The raw user question.

    Returns:
        Dict with keys: route, confidence, reasoning, extracted_filters.
        Falls back to rag route on any error.
    """
    t0 = time.time()

    # Run NER to detect person names before calling Claude
    person_names = _extract_person_names(user_query)

    # Build the message sent to the classifier
    classifier_input = user_query
    if person_names:
        classifier_input += f"\n\n[Detected person names: {', '.join(person_names)}]"

    try:
        response = client.messages.create(
            model=ROUTER_MODEL,
            max_tokens=512,
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": classifier_input}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)

        result = json.loads(text.strip())

        route = result.get("route", "rag")
        if route not in ("rag", "database", "hybrid"):
            route = "rag"

        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")

        extracted_filters = result.get("extracted_filters", {})
        # Ensure person_names from spaCy are included
        if person_names:
            existing = extracted_filters.get("person_names", []) or []
            merged = list(dict.fromkeys(existing + person_names))
            extracted_filters["person_names"] = merged

        elapsed = time.time() - t0
        logger.info(
            "Query routing: route=%s confidence=%.2f (%.1fs) — %s",
            route,
            confidence,
            elapsed,
            reasoning,
        )

        return {
            "route": route,
            "confidence": confidence,
            "reasoning": reasoning,
            "extracted_filters": extracted_filters,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        logger.warning("Query classification failed (%.1fs): %s", elapsed, exc)
        return {
            "route": "rag",
            "confidence": 0.0,
            "reasoning": f"classification_error: {exc}",
            "extracted_filters": {"person_names": person_names} if person_names else {},
        }
