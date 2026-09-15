# Session Debrief — 2026-09-08 — vLLM status check and d7f2f55 vs main conflict assessment

**Session scope**: Verify the local vLLM reviewer on Smeltor is live, then assess whether quality branch d7f2f55 can merge into main.
**Branch/project**: qorvault repo on Smeltor; branches `main` and `d7f2f55` (quality branch). Session date unknown — transcript fragment only; debriefed 2026-09-08 from a partial paste (report sections "vLLM Status" and "Step 5" only; steps 1–4 not seen).

## Decisions

None recorded in the fragment. The session produced findings, not decisions. One decision is now forced and is listed under Open Items: what to do with d7f2f55.

## What Changed

Nothing was modified. This was a read-only investigation.

## Findings

**vLLM on Smeltor is running and is the Stage 1 hook reviewer.**
- systemd user unit `civic-vllm.service`, Podman quadlet at `~/.config/containers/systemd/civic-vllm.container` (the deployed file; an untracked copy in the project repo is not the one in use)
- Model `Qwen2.5-72B-Instruct-AWQ`, `awq_marlin`, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, ~45.8 GB VRAM
- Host `127.0.0.1:8080` → container `:8000` via pasta; health-checked every ~30 s
- During the commit attempt the Stage 1 probe hit `/v1/models` at 04:12 and returned **ESCALATE** (local model flagged a commit touching the production path), which triggered Stage 2. Stage 1 did not ALLOW.

**Sidebar:** open-webui is also running as a Podman container on port 3000. Previously flagged as state-persisting by default; not part of this session's scope.

**d7f2f55 and main have completely unrelated histories.** `git merge-base` returns nothing; main descends from `d0c57c5` ("Initial public release"), the quality branch from `1706502` ("initial: project structure"). Existing branches `backup/local-main-unrelated-20260502T003246Z` and `backup/unrelated-history-hybrid-route-20260501` show this was already known in early May.

Conflict map:
- `rag_api/rag_api/retriever.py` — hard conflict. main has `validate_url_scheme()` (XSS guard) and `qdrant_filter`; d7f2f55 replaces the guard with `_resolve_source_url()`, changes the filter shape to `{"must": ...}`, adds `authority_score_boost`.
- `rag_api/rag_api/config.py` — hard conflict. main has reranker and recency settings, `excluded_document_types`, `qorvault` DB defaults; d7f2f55 removes reranker/recency config, adds `authority_score_boost`, uses `boarddocs` defaults.
- `rag_api/rag_api/main.py` — hard conflict. d7f2f55 drops `HybridRetriever`, `KeywordRetriever`, `Reranker` imports that main depends on.
- `rag_api/rag_api/prompts.py` — clean. d7f2f55 appends rule 8 (number-fabrication guard) at the end of a string main also has. Cherry-pickable.
- `embedding_pipeline/embedding_pipeline/pipeline.py` — not assessed; likely differs.

Root cause: d7f2f55 predates main's HybridRetriever / Reranker / ProviderRegistry architecture. Its retriever changes would have to be re-implemented against main, not merged.

## Open Items

1. **Disposition of d7f2f55** — Symptom: unmergeable quality branch. Tried: conflict assessment only. Diagnosis: architecture predates main; only `prompts.py` rule 8 is a clean pick. Proposed next step: cherry-pick rule 8 onto main if wanted; tag d7f2f55 as `archive/quality-branch-pre-hybrid` and stop carrying it. Note that `authority_score_boost` on d7f2f55 is functionally the authority weighting main already has — verify main's version covers it before discarding. Urgency: low; the rebuild supersedes this.
2. **Stage 1 hook over-escalates on Smeltor** — Symptom: local reviewer escalated a commit for touching "the production path" on a machine that cannot reach production. Tried: nothing. Diagnosis: hook rules still match production-path patterns regardless of host. Proposed next step: decide whether Smeltor's hook config should distinguish dev from prod paths, or leave it (escalation to Stage 2 is cheap). Urgency: low.
3. **Untracked quadlet copy in the repo** — Symptom: `civic-vllm.container` exists in the project repo but is not the deployed file. Proposed next step: either track the deployed file or delete the stray copy so the repo does not misrepresent runtime config. Urgency: low.
4. **open-webui state persistence** — previously flagged, still running on :3000. Not assessed here.
5. **Steps 1–4 of this session's report were not captured** in the paste. If the full report exists on Smeltor, recover it before the findings above are treated as complete.

## Documentation Impact

- Any project doc describing the hook pipeline should state that Stage 1 is Qwen2.5-72B-Instruct-AWQ via vLLM on `127.0.0.1:8080` (quadlet `civic-vllm.container`), if it does not already.
- Any doc that references the "quality branch" as pending work is stale; it is unmergeable as-is.

## System State Summary

On Smeltor, the qorvault repo's `main` branch carries the HybridRetriever + KeywordRetriever + cross-encoder Reranker + ProviderRegistry architecture with recency weighting, reranker SHA-256 verification, and `validate_url_scheme()` in `rag_api/rag_api/retriever.py`. A separate quality branch `d7f2f55`, with no common ancestor, holds an older single-retriever design plus an `authority_score_boost` and a prompts.py number-fabrication rule; it cannot be merged and is effectively archival. Commit-time review runs a two-stage hook: Stage 1 queries a local vLLM server (`civic-vllm.service`, Podman quadlet, Qwen2.5-72B-Instruct-AWQ, host port 8080, ~45.8 GB VRAM) and escalates to Stage 2 on production-path matches. open-webui runs alongside on port 3000. Smeltor has no network path to production by design.
