"""Query rewriting and decomposition via Claude Sonnet.

Rewrites conversational questions using official school district vocabulary
and decomposes complex questions into focused sub-queries for better
vector retrieval.
"""

from __future__ import annotations

import json
import logging
import re
import time

import anthropic

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """\
You are a query optimization assistant for a school district document RAG system. \
Your job is to rewrite user questions to maximize retrieval quality from a corpus \
of official school district documents including board meeting minutes, budget \
presentations, policy documents, financial reports, audit findings, and \
superintendent communications spanning 20 years.

Given a user question, produce a JSON response with this exact structure:
{
  "rewritten_query": "the primary query rewritten in precise technical language matching official school district documentation vocabulary",
  "sub_queries": [
    "focused sub-query 1 targeting a specific aspect",
    "focused sub-query 2 targeting another aspect"
  ],
  "reasoning": "brief explanation of what vocabulary choices and decomposition strategy you used"
}

Rules:
- Use official terminology: "general fund", "ending fund balance", "per-pupil allocation", \
"levy equalization", "I-728", "ESSER", "ARRA", "certificated staff", "classified staff", \
"ASB fund", "capital projects fund", "debt service fund", "CTE", "LAP", "Title I", \
"basic education allocation", "student FTE", "levy lid", "excess levy", "bond measure"
- Decompose questions that ask about causes, comparisons, trends, or multiple time periods
- Keep sub_queries between 2 and 4 — do not over-decompose simple factual questions
- For simple factual questions (single specific fact, single time period), use 0 sub_queries"""

REWRITE_MODEL = "claude-sonnet-4-6"


def rewrite_query(client: anthropic.Anthropic, user_query: str) -> dict:
    """Rewrite a user query using technical vocabulary and decompose into sub-queries.

    Args:
        client: An initialized Anthropic client instance.
        user_query: The raw user question.

    Returns:
        Dict with keys: rewritten_query, sub_queries, reasoning.
        Falls back to the original query on any error.
    """
    t0 = time.time()

    try:
        response = client.messages.create(
            model=REWRITE_MODEL,
            max_tokens=1024,
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_query}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Strip markdown code fences if present (```json ... ```)
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)

        result = json.loads(text.strip())

        # Validate expected keys
        rewritten = result.get("rewritten_query", user_query)
        sub_queries = result.get("sub_queries", [])
        reasoning = result.get("reasoning", "")

        # Clamp sub_queries to 4 max
        if len(sub_queries) > 4:
            sub_queries = sub_queries[:4]

        elapsed = time.time() - t0
        logger.info(
            "Query rewrite: %d sub-queries, %.1fs — %s",
            len(sub_queries),
            elapsed,
            reasoning,
        )

        return {
            "rewritten_query": rewritten,
            "sub_queries": sub_queries,
            "reasoning": reasoning,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        logger.warning("Query rewrite failed (%.1fs), using original: %s", elapsed, exc)
        return {
            "rewritten_query": user_query,
            "sub_queries": [],
            "reasoning": f"rewrite_error: {exc}",
        }
