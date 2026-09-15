# Session Debrief — 2026-09-08 — Retrieval smoke test on Smeltor

**Session scope**: Read-only diagnostic: are known-important BoardDocs documents present, parsed, chunked, and retrievable in the existing QorVault corpus on Smeltor?
**Branch/project**: `ksd-boarddocs-rag` on Smeltor; PostgreSQL + Qdrant dev corpus. Report at `reports/smoke-test-2026-09-08.md`. Diagnostic only; nothing modified.

## Decisions

None in-session. Two are now forced (see Open Items 1 and 2) and must be resolved before any conclusion in this report is acted on.

## What Changed

Nothing. Read-only.

## Findings

**Parse and coverage are good for named documents.** All operator-supplied documents exist, parsed cleanly, and chunked at sensible sizes: Resolution 1680 agenda item (4 chunks) and full legal PDF (14), Resolution 1679 (2), superintendent contract (15), SRO agreement (20) and all four renewals, Resolution 1669 original and repeal. Re-ingestion for parse quality is **not** justified by this test.

**Retrieval ranking of formal documents is poor under the pipeline that was tested.**
- Resolution 1680 agenda item: rank 16 on conversational query (outside top-10)
- Resolution 1680 PDF: rank 12 on title query, rank 2 on natural-language query
- Superintendent contract: not in top 50 on title query; rank 41 on NL
- SRO renewals: fine on title, bottom of top-50 on NL
- Resolution 1669 original: misses entirely on NL
- Resolution 1679 and SRO original: rank 1–5, no issue

Report attributes this to pure cosine similarity with no BM25 and no reranker — formal/legal text embeds into a region outcompeted by the larger population of conversational minutes and agenda items on the same topics. **This contradicts the documented and branch-verified architecture of `main`** (HybridRetriever + Postgres FTS + RRF + cross-encoder reranker). See Open Item 1.

**The $97.8M levy total does not exist in source text.** Resolution 1680 states three annual amounts ($32.2M, $32.6M, $33.0M). No chunk contains "97.8". Any query on the total will miss regardless of retrieval quality; the total is a derived fact.

**ESSER allocation resolution not found.** Only ESSER-tagged items are project-level bid documents. Either allocations were adopted under a different instrument (budget amendment, extension) or the scraper missed a category.

**Qdrant and PostgreSQL disagree on corpus size.** Qdrant 230,587 points; PostgreSQL 179,081 chunk rows; difference 51,506. Report calls the excess Qdrant "orphans." Direction of truth not established — the ~230k figure matches prior documentation of the corpus, so Postgres may be the side missing rows. See Open Item 2.

**Step 0 (corpus fidelity vs. production) was not reported** in the summary. Unknown whether the dev corpus matches production.

## Verification (follow-up, same day)

**Q1 — pipeline tested: the pre-hybrid retriever, not main.** Working tree is `claude/infra-vllm-quadlet` at `8b0b8db` (2026-03-31). `rag_api` was not running; only Qdrant and Postgres containers were up. The smoke test queried Qdrant directly. The checked-out `retriever.py` has a single one-stage vector `Retriever`; no HybridRetriever, KeywordRetriever, or reranker in that tree. Ranks are equivalent to what the *March-31* `rag_api` would return (same ONNX embedder, cache path, collection, tenant filter, search body; only top_k differs, 10 vs 50). **All ranking conclusions apply to the pre-hybrid pipeline only.** Unresolved contradiction: the 2026-09-08 branch-conflict session reported `main` imports HybridRetriever/KeywordRetriever/Reranker; this session says none exist "anywhere in the codebase" — most likely scope (checked-out tree vs `main`), to be confirmed with `git show main:...`.

**Q2 — Postgres is truth.** 51,506 = 50,125 pre-tenant-schema points (no `tenant_id` payload; invisible to tenant-filtered search; harmless) + 1,381 stale `kent_sd` duplicates (old chunk UUIDs from a re-chunking event, never purged; live in search; all large attachments — vouchers, budgets, contracts). 38/38 sampled `kent_sd` orphans had their parent document present in Postgres with current chunks. Zero documents absent from Postgres.

## Verification 2 — main pipeline (same day)

**Setup.** Worktree `~/workspace/projects/ksd-main` on `main`. `rag_api` launched from the worktree with system `python3` (`uvicorn rag_api.main:app --host 127.0.0.1 --port 8000`), Postgres password injected at runtime (`.env` copy is stale), Anthropic key via copied `.env`. Startup log confirms: reranker SHA-256 verified, `Cross-encoder reranker loaded`, `hybrid_search=enabled`, `model=claude-opus-4-6`, collection `boarddocs_chunks`. Pre-checks: `chunks.search_vector` 179,081/179,081 populated, both FTS indexes present. `main` has HybridRetriever/KeywordRetriever/Reranker in `hybrid_retriever.py`, `keyword_retriever.py`, `reranker.py` (Verification 1's "nowhere in the codebase" was scoped to the March-31 checkout and was wrong). Server stopped after the run; production checkout untouched.

**Method note.** `/api/v1/query` has no retrieval-only mode; every query cost an Opus call. `citations[].source_number` is the post-rerank position (built unconditionally from `hybrid_retriever.search()` output; 25 entries, monotonically descending cross-encoder scores), so it is a valid rank. Logs record only counts ("25 vector + 18 keyword -> 35 fused"), never per-stage document positions. top_k capped at 25.

**Results (old pre-hybrid rank → main rank, 12 queries):** 8 improved, 3 unchanged, 1 regressed, 2 still missing, 2 indeterminate. All four major gains were title/identifier queries (12→1, 8→1, 4→1, 2→1). Conversational queries barely moved (16→14, 5→4) or worsened (2→7). **Superintendent contract still not retrievable** on the NL query: vector >25, keyword >25, hybrid >25.

**Mechanism.** `plainto_tsquery` ANDs all terms; the NL contract query matches 0 chunks of the contract (document text says "agreement"; "superintendent agreement" matches 14; "Vela" matches 2). Keyword leg contributes nothing and hybrid degrades to pure vector. Deeper cause: "Vela" appears in only 2 of the contract's 15 chunks — chunks lack document identity. 16,640 chunks corpus-wide match "superintendent".

**Regression.** Res 1680 PDF 2→7 (CE score 0.5330); likely `recency_weight=0.3` demoting a 2024 document below 2025–26 minutes.

**Verdict.** `main` is deployable and a large improvement on identifier queries; neutral-to-negative on prose queries. Smoke-test recommendations 1–2 (add keyword search, add reranker) were already built and do not solve the motivating case.

## Rebuild design inputs (from this session)

1. Contextual chunk headers: prepend document title, type, date, parties to every chunk before embedding and FTS indexing.
2. Document-type routing: query intent "contract"/"resolution"/"policy" → metadata filter before scoring.
3. Keyword query construction: OR-with-weights / `websearch_to_tsquery` / real BM25; never degrade to zero.
4. Intent-conditional recency: recency weight applied by the router per query type, not globally.
5. `/retrieve` endpoint returning per-stage candidates and scores (vector, keyword, fused, reranked); eval harness targets this.
6. Fact layer (unchanged): extracted amounts, terms, votes, with line-level citations.
Ingestion/parse layer retained as-is.

## Open Items

1. **Re-run against main** — DONE (see Verification 2).
1b. **Re-run against main (original)** — Symptom: smoke test measured the March-31 retriever. Next step: confirm hybrid exists on `main` (`git show main:rag_api/rag_api/retriever.py | grep -c Hybrid`), add a separate worktree on `main`, start `rag_api`, re-run the identical document/query set through the endpoint at top_k 50, report old vs new rank. Urgency: **high** — decides whether document-type routing enters the rebuild. *(Supersedes prior item 1.)*
1a. **Which pipeline was tested? (original)** — Symptom: report describes pure-vector retrieval; `main` carries hybrid + reranker. Possibilities: (a) queries went straight to Qdrant, bypassing `rag_api`; (b) the running dev service is checked out on the quality branch (which removed HybridRetriever/Reranker); (c) dev deployment predates main. Tried: nothing. Next step: `git -C ~/workspace/projects/ksd-boarddocs-rag branch --show-current`, confirm what process serves the API and from which checkout, then re-run the same document set through the `rag_api` query endpoint. Urgency: **high** — every ranking conclusion depends on it.
2. **Qdrant cleanup** — RESOLVED direction: Postgres is truth. Next step: delete the 1,381 stale `kent_sd` duplicates on dev (parents verified present); the 50,125 pre-tenant points can go too but are inert. Then check whether **production** Qdrant carries the same 1,381 — if so, stale contract/budget chunks can surface to real users; fix before the rebuild. Urgency: medium.
3. **Step 0 corpus fidelity** — Next step: pull the step 0 figures from the full report and compare against production counts. Urgency: medium.
4. **ESSER coverage** — Next step: search BoardDocs directly for how ESSER funds were adopted (budget extension? grant acceptance?) and check whether the scraper captured that instrument type. Urgency: medium — this is the first confirmed coverage gap.
5. **Derived-fact retrieval ($97.8M)** — Design input for the rebuild, not a bug: totals, sums, and per-year amounts need an extracted facts table with line-level citations. Urgency: n/a — feeds the canonical-model design.
6. **Formal-document ranking** — RESOLVED: hybrid+reranker tested; formal documents still sink on prose queries. Document-type routing and contextual chunk headers enter the rebuild (see Rebuild design inputs).
7. **Stale `.env` Postgres password** — Symptom: `rag_api` cannot start via the documented CLAUDE.md procedure; runtime injection used for this test. Next step: operator updates `.env` out-of-band. Urgency: medium.
8. **`block-production-path.sh` gaps** — Symptom: literal string match on the absolute path; `~`-prefixed paths bypass it (the `git worktree add` passed this way); `python3 -c` is allowlisted. Also its suggested dev path `/mnt/qorvault-dev/repo/` does not exist. Next step: operator decision — re-scope for Smeltor (drop production-path matching; keep destructive-data and promotion-path denies, `realpath`-resolved; add backups) or fix as-is (`realpath` + drop `python3 -c`). Urgency: medium; higher before any local-model executor trial.
9. **Res 1680 recency regression** — Next step: confirm `recency_weight` is the cause by re-running with it at 0; feeds design input 4. Urgency: low.
10. **Deployed generator pinned to `claude-opus-4-6`** — config staleness; note for docs. Urgency: low.

## Documentation Impact

- Anything stating Q7 (levy resolution) was an ingestion/parse failure is wrong; it is a ranking and/or derived-fact problem.
- Corpus size claims (~230k chunks) need a footnote until Open Item 2 is resolved.
- Rebuild plan: the "re-ingest for parse quality" branch of the plan is retired; the "fact layer" and "retrieval routing by document type" branches are strengthened.

## System State Summary

On Smeltor, `main` (worktree `~/workspace/projects/ksd-main`) runs the HybridRetriever + Postgres FTS + RRF + cross-encoder reranker pipeline cleanly against the dev corpus; it is not left running. The QorVault dev corpus holds every operator-named BoardDocs document with clean parsed text and reasonable chunk counts. Qdrant reports 230,587 points and PostgreSQL 179,081 chunk rows; Postgres is authoritative — the 51,506 excess is 50,125 inert pre-tenant points plus 1,381 live stale `kent_sd` duplicate chunks of large attachments, none yet deleted. The Smeltor working tree is `claude/infra-vllm-quadlet` at `8b0b8db` (2026-03-31), which predates the hybrid architecture; `rag_api` is not running there. Under that pre-hybrid single-stage vector retriever (queried directly in Qdrant, rank-equivalent to the March-31 `rag_api`) — formal documents (contracts, resolutions) rank poorly against conversational queries while agenda items and minutes rank well; under `main`'s hybrid + reranker pipeline, identifier/title queries now rank 1 but prose queries are unchanged or worse, because `plainto_tsquery` AND-matching yields zero keyword hits and chunks lack document identity ("Vela" in 2 of 15 contract chunks); the superintendent contract remains unretrievable on a natural-language query. `rag_api` has no retrieval-only endpoint and logs no per-stage ranks. Resolution 1680's $97.8M total is not present in any chunk (source states three annual amounts). No ESSER allocation resolution exists in the corpus. Nothing was modified.
