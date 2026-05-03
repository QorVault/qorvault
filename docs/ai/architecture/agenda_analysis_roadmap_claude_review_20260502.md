# Agenda Analysis System — Architecture Decision Memo

**Author:** Claude Code review (Opus 4.7, 1M context)
**Date:** 2026-05-02
**Repo HEAD at review:** `62bd8a6ba9a975ba6fd7fe77d5b9b5c99c127dd2` (post-PR #28)
**Status:** DRAFT for Donald's review. Owner decisions §(a)–(g) applied
2026-05-02. Prepared for a dedicated docs-only PR. No code written.

This memo converts Donald's autonomous-weekly-agenda roadmap into an
architecture review with deterministic-first interpretation **B**: LLMs are
allowed only as bounded structured-IO components with strict input contracts,
fixed templates, JSON output schemas validated in code, deterministic
orchestration, and deterministic verification gates.

---

## Table of contents

- Owner decisions applied (V1)
- Open decisions remaining
- 0. Working assumptions (challenge any that are wrong)
- 1. Existing implementation vs. proposed future system
- 2. Strongest parts of the roadmap
- 3. Unsafe assumptions / missing risks
- 4. Recommended Stage 0 / Stage 1 docs and skeletons
- 5. Durable run-controller design
- 6. Agenda watcher / downloader boundary
- 7. Delta ingest + embedding boundary
- 8. Per-item analysis JSON contract (`agenda-item-analysis/v1`)
- 9. Question-generation strategy
- 10. Likely-district-answer strategy
- 11. Follow-up-question strategy
- 12. Consequence-ranking rubric
- 13. Verification / redrive gates
- 14. Provider abstraction
- 15. Local-LLM parity plan
- 16. Notification design
- 17. Recommended implementation order
- 18. Plain-English decision points for Donald (RESOLVED)
- Top 5 risks (ranked)
- Top 5 next implementation steps
- Open decisions for Donald
- Safety attestation

---

## Owner decisions applied (V1)

The seven decision points raised in §18 have been resolved by Donald and
applied to the design below.

| # | Decision | Resolution |
|---|---|---|
| (a) | Failed verification handling | **Withhold failed items.** Verified items render normally; failed items are replaced by a "Withheld — verification failed: <reason>" stub listed at the top of the report. The report is delivered even with withholdings; nothing is hidden. |
| (b) | `codex_remote` provider | **Deferred. No CLI wrapper.** `codex_remote` stays a stub in V1. A real implementation is allowed only via the OpenAI Codex API once a stable non-interactive interface is explicitly approved by Donald. Driving Codex CLI headlessly is out of scope. |
| (c) | Per-run cost ceiling | **Both limits.** A soft limit warns/flags. A hard limit halts further model calls; remaining work is marked withheld or deferred. Both values live in versioned config so revisions go through PR. |
| (d) | Rubric weights V1 | **Accepted as versioned draft.** Weights live in `config/rubric/v1.toml`. Every artifact stamps `weights_version`. Tuning is allowed via PR; historical scores remain reproducible. |
| (e) | Predicted-answer display | **Internal-only in V1.** Predictions are stored and drive follow-up questions only; they never appear in the public/final report until the reliability risk is better understood. |
| (f) | Resumability | **Within-run + cross-run reuse.** A single run can survive process restarts within its 24h TTL. Prior-run artifacts are reusable across runs only when the input identity matches; reuse is opt-in per artifact kind. Default reuse table: evidence packs and classifications **reuse**; questions, predicted answers, follow-ups, rubric, verifications **regenerate**. |
| (g) | Revision identity | **Stable GUID + content hash.** Item identity is `(meeting_id, item_guid)` — preserved across runs. `content_hash` detects revision; a mismatch with the same identity flags the item as `revised`, triggers re-analysis of that item only, and records a revision history. |

---

## Open decisions remaining

These items are still owner-side and need Donald's input before the
relevant implementation issues can move.

| # | Item | Why it's open | Suggested resolution path |
|---|---|---|---|
| O1 | Working assumptions A1–A8 (§0) | Each is a starting assumption I made to keep the memo moving. | Donald reads §0; flags any "no" items as new decision points. |
| O2 | Cost-ceiling values | Decision (c) is **yes**, but the soft and hard dollar amounts aren't set. | Run a synthetic-data cost simulation as part of Issue 9 and propose numbers. Donald approves both before any production run. |
| O3 | Rubric weight first-tune checkpoint | Decision (d) accepts V1 weights. Donald should pick a review checkpoint. | Calendar a weights-review pass after weekly run #4. |
| O4 | Predicted-answer promotion criteria | Decision (e) is internal-only. Need a written rule for when to promote to appendix or public. | Define a clean-runs streak threshold (e.g., 8 consecutive runs with zero withheld predictions) before lifting the suppression. |
| O5 | SMTP / email infrastructure (A3) | Notification design assumes an SMTP relay + sender identity. | Confirm A3 yes/no. If no, pick a transactional-email API provider before Issue 14. |
| O6 | `codex_remote` API approval (future) | Decision (b) defers `codex_remote`. Future API path is gated on explicit owner approval. | Revisit only if `claude_remote` + `llama_local` prove insufficient after parity phase. |
| O7 | Per-kind cross-run reuse table | Decision (f) approves cross-run reuse but lists a default reuse table; Donald should confirm. | Lock the per-kind reuse table in Issue 1 (DB schema + JSON Schemas). |

---

## 0. Working assumptions (challenge any that are wrong)

These are non-blocker assumptions I made to keep the memo moving. Each is
labeled so Donald can reject or amend.

- **A1.** The analysis worker uses the existing retriever module directly
  (the "retrieval-only" hybrid path clarified by PR #26 / issue #27), **not**
  the `/api/v1/query` endpoint. The endpoint orchestrates an LLM; the worker
  bypasses LLM routing and gets raw retrieval.
- **A2.** Anthropic API access is available today (already used by `rag_api`).
  Local llama-server is also available (already running). `codex_remote` is
  **deferred per decision (b)**; it remains a stub in V1 and is not exercised
  by any code path. A future API path requires explicit owner approval.
- **A3.** Email infrastructure: an SMTP relay is reachable from Smeltor (or
  will be), and a sender identity (e.g. `runs@qorvault.local`) is acceptable.
  If neither is true, notification design needs to grow a "send via API"
  branch (Postmark / SES). SMS is deferred.
- **A4.** BoardDocs continues to expose stable per-item GUIDs (the existing
  loader captures them). Item identity is `(meeting_id, item_guid)`; content
  changes against the same identity are revisions.
- **A5.** The runs/items/artifacts/claims/notifications tables live in the
  same `qorvault` Postgres database as the existing schema, under a new
  `agenda_runs` schema (Postgres schema, not table prefix) for namespace
  isolation. tenant_id stays `kent_sd` everywhere.
- **A6.** Analysis runs are weekly, scheduled by a systemd user timer. No
  ad-hoc runs in V1; manual redrive of failed items is the only off-schedule
  trigger.
- **A7.** Per-item analysis depth target: ~600–1,200 output tokens per item
  for the full JSON (questions + predicted answers + follow-ups + rubric).
  Real budget gets refined after first cost estimate.
- **A8.** "Final report" = a single markdown file written under
  `docs/ai/agenda_runs/<run_id>/report.md`, plus a summary email body. Not a
  web page in V1.

---

## 1. Existing implementation vs. proposed future system

**Existing today (post-PR #28):**

- `boarddocs_loader` ingests scraped meeting JSON to Postgres; full-load mode.
- `document_processor` extracts/chunks documents (manual or bulk).
- `embedding_pipeline` embeds chunks into Qdrant (`mxbai-embed-large-v1`,
  1024-dim, cosine).
- `rag_api` exposes `/api/v1/query` for ad-hoc Q&A with citations
  (Anthropic-backed).
- `infrastructure/` runs Postgres (with pgvector) and Qdrant under Podman
  quadlets; llama-server runs under systemd.
- No automated run controller. No scheduled detection of new agendas.
  No batch analysis. No verification layer. No notifications.

**Proposed additions (the "agenda analysis system"):**

1. **Agenda watcher** — weekly poller; produces `pending_meeting` rows.
2. **Agenda downloader** — fetches new packets to staging.
3. **Delta-aware loader** — extends `boarddocs_loader` with `--delta`.
4. **Run controller** — durable Postgres-backed state machine.
5. **Per-item analysis worker** — bounded LLM calls for questions /
   predicted answers / follow-ups, structured JSON only.
6. **Consequence ranker** — deterministic rubric with structured-IO LLM
   classifications for individual fields.
7. **Verifier** — deterministic citation-integrity and numeric-claim
   checks; redrive logic.
8. **Renderer** — deterministic markdown report generator.
9. **Notifier** — email V1, SMS V2.
10. **Provider abstraction** — `claude_remote`, `codex_remote`, `llama_local`.
11. **Local-LLM parity test harness** — promotion gate per task.

The new pieces share the existing Postgres + Qdrant + llama-server
infrastructure; they do not introduce new datastores or new languages.

---

## 2. Strongest parts of the roadmap

- **Deterministic-first with structured-IO LLM components** is the right
  shape. It gives you the LLM's reasoning surface where it's needed
  (question phrasing, classification) without letting it touch the data
  path or system state.
- **Per-item JSON contract enforced before any prose rendering** is the
  single highest-leverage discipline. It eliminates citation drift and
  prevents the LLM from inventing claims that survive into the final report.
- **24h resumable runs** is appropriate for a weekly batch — implies
  durable state, which you already have via Postgres.
- **Provider abstraction with explicit local-parity gate** is excellent
  governance. It avoids the trap of promoting a local model based on vibes.
- **Verification + redrive before render** is the second-highest leverage
  discipline.
- **Email-first, SMS-later** is pragmatic; defers a billing relationship
  until you've proven the pipeline works.

---

## 3. Unsafe assumptions / missing risks

| # | Risk | Mitigation in this design |
|---|------|---------------------------|
| R1 | "Predict 5 likely district answers" risks invention even with structured IO; the model can produce plausible-but-unsupported predictions. | All predicted answers tagged with `speculation_grade` and `evidence_chunks`. Decision (e) below proposes suppressing them in the public report by V1. |
| R2 | "Verify every numeric claim" is harder than it sounds — claims may aggregate across multiple sources or use rounding. | Two-stage check: substring match with normalization (commas, $, %), then parse-and-compare with tolerance. Failures redrive once, then withhold. |
| R3 | `codex_remote` is not a real API surface today — Codex CLI is interactive. | Stub `codex_remote` initially. Donald to decide CLI-wrapper vs. OpenAI Codex API (see decision b). |
| R4 | Per-meeting LLM cost can spike (a 30-item meeting × 1 question gen + 1 answer gen + 5×1 follow-up gen + 1 rank + 1 verify retry ≈ 270+ calls). | Soft + hard cost ceilings with run abort; per-item token budgets enforced before dispatch. |
| R5 | BoardDocs HTML/JSON shape can drift; the watcher may silently miss meetings. | Watcher emits a structured "candidate meetings" list every run, even on days with zero new meetings, so a sustained zero is detectable. Alert if zero new meetings for >2 weeks. |
| R6 | Embedding-collision on revised agendas: a corrected item could be treated as new, doubling embeddings. | Identity = `(meeting_id, item_guid)`. If GUID match + content_hash differs, mark as `revised`; only re-embed changed chunks. |
| R7 | Run-controller single-writer assumption breaks if a stuck process holds the advisory lock. | Lock acquisition with timeout; expired locks broken by an external `agenda-lock-reaper` after configurable threshold (default 6h). |
| R8 | Email delivery is silently lossy; SMTP relay drops mean Donald never sees a "completed" notification. | `notifications` table records every send attempt; a separate `notification-watchdog` timer raises a system-level error if a `completed` run has no successful notification within 1h. |
| R9 | Schema-validated LLM outputs can still be semantically wrong (e.g., a `text` field that just says "TBD"). | Schema includes minimum-content constraints (regex for non-trivial content; minimum length thresholds). Verification re-checks. |
| R10 | Local model promotion via parity test risks a "the test set is what we trained on" effect; parity ≠ generalization. | Parity test set is rotated quarterly; held-out items are added each quarter and never seen by the local model during evaluation. |

---

## 4. Recommended Stage 0 / Stage 1 docs and skeletons

**Stage 0 — design only, no code:**

- This memo (after Donald's edits and approval).
- DB migration plan: additive-only DDL committed as a markdown design,
  not yet applied.
- JSON Schema files (Draft 2020-12) for: per-item analysis,
  run-state envelope, artifact envelope, ranking-rubric output,
  notification record. Stored under `schemas/agenda_runs/`.
- Operations runbook stub: how to start a run, check status, redrive an
  item, abort a run, rotate the parity test set.
- Cost-estimation spreadsheet for per-meeting LLM cost.

**Stage 1 — skeleton code, no production runs:**

- DB migrations applied to **dev DB only**, idempotent.
- Provider abstraction package with `claude_remote` real, `codex_remote`
  stubbed, `llama_local` real but tested only on a fixed prompt.
- Run controller skeleton: walks an item through the state machine using
  hard-coded fake LLM responses; demonstrates state transitions, claim
  ownership, retry policy, redrive policy.
- Notification stub: writes to `notifications` table with channel = `log`.
- One end-to-end synthetic-data test that walks a fake meeting (3 items)
  through every state without contacting BoardDocs, Anthropic, llama, or
  the real database (uses a test schema).

---

## 5. Durable run-controller design

**Process model:** Single long-running Python process per active run, started
by systemd-user one-shot from the weekly timer. Restarts pick up where they
left off by re-scanning Postgres. No in-memory state.

**State machine (per run_item):**

```
pending → claimed → retrieving_evidence → questions_generating
        → answers_generating → followups_generating → ranking
        → verifying → verified
                  ↘ verification_failed → redrive_queued → claimed (retry)
                                       ↘ withheld (after max retries)
        ↘ failed_timeout (after 24h)
```

Run-level states: `created → ingesting → analyzing → rendering → completed`
or `aborted` or `failed`.

**Persistence:** Postgres tables (additive, namespaced under schema
`agenda_runs`):

- `agenda_runs.runs (run_id, started_at, ended_at, status, weights_version,
  cost_soft_cap, cost_hard_cap, totals_json)`.
- `agenda_runs.run_items (run_item_id, run_id, meeting_id, item_guid,
  state, attempts, last_error, evidence_pack_id, claim_owner,
  claimed_at, ...)`.
- `agenda_runs.artifacts (artifact_id, run_item_id, kind, schema_version,
  content_jsonb, model_provenance_jsonb, created_at)`.
  `kind ∈ {evidence_pack, questions, predicted_answers, follow_ups,
  rubric, verification_report, render_fragment}`.
- `agenda_runs.claims (claim_id, run_item_id, kind, value_text,
  numeric_value, source_chunk_id, verified_bool, verifier_notes)`.
- `agenda_runs.numeric_claims` (split out for indexing if volume warrants).
- `agenda_runs.notifications (notification_id, run_id, channel, address,
  template, body, status, attempts, last_error, created_at, sent_at)`.
- `agenda_runs.verification_results (id, run_item_id, gate, status,
  details_jsonb, created_at)`.

**Concurrency:** `SELECT … FOR UPDATE SKIP LOCKED` to claim items; per-run
advisory lock so only one controller drives a given run at a time;
per-item advisory lock during transitions.

**Idempotency:** every state transition is `UPDATE … WHERE state = expected`
(optimistic). Workers re-check artifact existence before regenerating.

**Backoff:** per-item attempt counter + exponential delay (1m, 5m, 15m).

**TTL:** 24h soft cap → state `failed_timeout` for items not in a terminal
state. Run is still allowed to render with the verified subset.

**Cross-run reuse (per decision f):** the run controller can re-attach prior
artifacts when a new run's input identity matches an existing one. Reuse is
opt-in per artifact kind, locked at the JSON-Schema layer. Default reuse
table:

| Artifact kind | Reuse policy |
|---|---|
| `evidence_pack` | **reuse** if the underlying chunk IDs and retrieval params match |
| `rubric.classifications` (policy_area, reversibility, time_horizon) | **reuse** if the item content_hash matches |
| `questions` | **regenerate** every run |
| `predicted_answers` | **regenerate** every run |
| `follow_ups` | **regenerate** every run |
| `rubric.score` | **regenerate** every run (uses current weights_version) |
| `verification_results` | **regenerate** every run |

This default lets the controller skip retrieval and classification work when
they haven't changed, but always re-derives the analytical output so it
reflects the current model and current weights. Donald confirms or amends in
Issue 1.

---

## 6. Agenda watcher / downloader boundary

**Watcher** (`scripts/agenda_watcher.py`, systemd timer Sundays 6:00 AM):

- Reads `agenda_cursor` (last-known meeting datetime per committee).
- Calls BoardDocs listing endpoint(s) with cursor; collects candidate
  meetings.
- Filters by status (Final/Approved/Posted, **not** Draft).
- Writes one `pending_meeting` row per new meeting with status `discovered`.
- Emits one structured log line per run with `(committees_scanned,
  candidates_found, posted_filtered_in, posted_filtered_out)` so a
  sustained zero is observable.
- **Does not** download attachments. Does not touch documents/chunks.

**Downloader** (separate service or watcher subcommand):

- Picks up `pending_meeting` rows in `discovered` state.
- Downloads files to `runs_staging/<meeting_id>/` (gitignored).
- Computes content hashes and writes them into `pending_meeting`.
- Transitions to `downloaded`.

**Retries:** transient HTTP errors retry with backoff; hard 404 marks the
meeting as `unavailable` with notes.

**Boundary contract:** the run controller only consumes meetings in state
`downloaded`. It never re-fetches from BoardDocs.

---

## 7. Delta ingest + embedding boundary

Extend `boarddocs_loader` with a `--delta --pending-meeting-id <uuid>` mode
that processes exactly one meeting and exits.

- Loader inserts `documents` and `document_pages` rows; existing
  `ON CONFLICT DO NOTHING` handles re-runs.
- After loader exits, controller invokes `document_processor` filtered
  by `document_id IN (...)` from the just-loaded set.
- After processor exits, controller invokes `embedding_pipeline` filtered
  to chunks with `embedding_id IS NULL`. (Pipeline already supports this
  shape; cron is just disabled.)
- Each step emits a `run_artifact` row recording counts and document IDs.

**Revision handling (per decision g — stable GUID + content hash):** identity
is `(meeting_id, item_guid)` and is preserved across runs. If a document
with same identity exists with a different `content_hash`, mark old chunks as
`superseded` and re-process. Old embeddings remain in Qdrant, retrievable but
de-prioritized via payload filter. The revision history (`(item_guid, prior
content_hash, new content_hash, observed_at)`) is recorded in
`agenda_runs.revision_history` so per-item re-analysis can be audited.

**Boundary contract:** delta ingest never deletes rows or vectors; it only
adds and marks. Failures during delta can re-run the same
`pending_meeting_id` safely.

---

## 8. Per-item analysis JSON contract (`agenda-item-analysis/v1`)

```json
{
  "schema_version": "agenda-item-analysis/v1",
  "run_id": "uuid",
  "meeting_id": "string",
  "item_guid": "string",
  "item_title": "string",
  "item_type": "consent|action|discussion|presentation|workshop|study|other",
  "evidence_pack_id": "uuid",
  "primary_questions": [
    { "id": "q1", "text": "≥40 chars", "rationale": "≥80 chars",
      "evidence_chunks": ["chunk_id", "chunk_id"] }
  ],
  "primary_questions_count": 3,
  "predicted_answers": [
    { "id": "a1", "text": "≥40 chars",
      "speculation_grade": "high|medium|low",
      "supporting_history_chunks": ["chunk_id"] }
  ],
  "predicted_answers_count": 5,
  "follow_up_questions": [
    { "answer_id": "a1",
      "questions": [
        { "id": "fq1", "text": "≥40 chars",
          "evidence_chunks": ["chunk_id"] }
      ],
      "questions_count": 3 }
  ],
  "consequence_rubric": {
    "weights_version": "v1",
    "components": {
      "dollar_magnitude": { "raw": 1234567, "score_0_100": 78 },
      "stakeholder_count_proxy": { "groups": ["students","families"], "score_0_100": 40 },
      "policy_area": { "value": "finance", "weight": 1.0, "score_0_100": 100 },
      "action_type": { "value": "action", "score_0_100": 100 },
      "reversibility": { "value": false, "score_0_100": 100 },
      "time_horizon": { "value": "long", "score_0_100": 75 },
      "community_input_window": { "value": true, "score_0_100": 100 }
    },
    "score_0_100": 84.2,
    "justification_text": "must contain ≥1 verbatim quote from item"
  },
  "citations": [
    { "chunk_id": "...", "source_url": "https://...", "title": "...",
      "meeting_date": "YYYY-MM-DD" }
  ],
  "numeric_claims": [
    { "id": "n1", "claim_text": "...", "value": "1234567",
      "value_kind": "currency|percent|count|date|other",
      "source_chunk": "chunk_id", "verified": false,
      "verification_method": null, "verification_notes": null }
  ],
  "verification_status": "pending|passed|failed|withheld",
  "verification_notes": [],
  "model_provenance": {
    "provider": "claude_remote",
    "model": "claude-opus-4-7",
    "prompt_template_id": "questions/v1",
    "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
    "schema_repaired_attempts": 0
  },
  "generated_at": "ISO8601"
}
```

Schema enforcement: Pydantic v2 model with strict types; cardinality
constraints (exactly 3 / exactly 5 / exactly 3 follow-ups per answer);
JSON Schema mirror in `schemas/agenda_runs/agenda-item-analysis.v1.json`
for cross-language validation. Reject + repair-prompt once on schema
violation; otherwise mark `redrive`.

---

## 9. Question-generation strategy

- **Evidence pack:** item content (full), parent meeting context (title,
  date, committee), top-K (default 8) retrieved chunks via the
  retrieval-only path, deduplicated by document_id.
- **Prompt template:** fixed; references the evidence pack by chunk IDs.
- **Output schema:** exactly 3 questions; each `evidence_chunks` MUST be
  a subset of pack chunk IDs.
- **Validation:** schema, count, evidence-membership, minimum text length,
  minimum rationale length.
- **Failure path:** schema-repair retry (1×) → redrive → withhold.
- **Provider:** `claude_remote` in V1.

---

## 10. Likely-district-answer strategy

- **Per item, not per question.** Generate 5 predictions for the item
  overall.
- **Evidence pack:** item content + 3 primary questions + retrieval over
  prior meetings filtered to similar `item_type` and matching policy area
  (top-K = 12).
- **Output schema:** exactly 5 predictions, each with
  `speculation_grade` and `supporting_history_chunks`. `low` grade
  predictions must include explicit "no precedent found" note.
- **Display policy:** see decision (e). Recommend suppressing from the
  public report in V1; use them only as scaffolding for follow-ups.
- **Provider:** `claude_remote` in V1.

---

## 11. Follow-up-question strategy

- **Per predicted answer, generate exactly 3.**
- **Evidence pack:** item content + the predicted answer's text + the
  answer's `supporting_history_chunks`.
- **Output schema:** 3 questions, each with `evidence_chunks`
  (subset of pack).
- **Provider:** `claude_remote` in V1; later candidate for `llama_local`
  promotion (relatively short context, repetitive structure).

---

## 12. Consequence-ranking rubric

**Goal:** an auditable score that Donald can defend, not an LLM opinion.

**Components and proposed weights (V1, subject to Donald's review — see
decision d):**

| Component | Weight | Source |
|---|---|---|
| dollar_magnitude | 25% | regex + structured extraction; log-scaled to 0–100 |
| policy_area | 25% | LLM classifier into fixed enum × per-area policy weight |
| stakeholder_count_proxy | 15% | keyword count over fixed lexicon, capped |
| action_type | 15% | enum (action > discussion > presentation > consent) |
| reversibility | 10% | LLM yes/no with mandatory evidence quote |
| time_horizon | 5% | enum (short/medium/long), regex-extracted dates first |
| community_input_window | 5% | structured field from scrape if available, else 0 |

**Per-area policy weights** (V1, Donald to confirm):
curriculum 1.0, finance 1.0, personnel 0.9, safety 1.0,
facilities 0.7, governance 0.6.

**Formula:** `score = 100 * sum(component_weight * component_score / 100)`.

**Determinism:** all weights live in a versioned config file
(`config/rubric/v1.toml`). `weights_version` stored on every rubric
artifact so historical scores remain reproducible after weight changes.

**LLM role:** classify individual fields under strict schemas; cannot
influence formula or weights; must include an evidence quote in the
`justification_text`.

---

## 13. Verification / redrive gates

**Gate 1 — citation integrity** (deterministic, no LLM):

- Every `evidence_chunks[*]` and `supporting_history_chunks[*]` ID must
  exist in Qdrant `boarddocs_chunks`.
- Every `citations[*].chunk_id` must exist; `source_url` must equal the
  chunk's stored `source_url`.
- Failures: schema repair attempt 1× → redrive 1× → withhold.

**Gate 2 — numeric-claim integrity** (deterministic):

- For each `numeric_claim`, fetch the cited chunk's content text.
- Substring check with normalization: strip commas, normalize `$`, `%`,
  unicode dashes, whitespace.
- If substring fails, parse-and-compare: extract claimed value, parse all
  numbers from chunk; require exact match for currency/count, ±0.05 abs
  tolerance for percentages.
- Failures: redrive 1× → withhold.

**Run-level rendering policy:** **withhold failed items, render the rest,
list withheld items at top of report with reasons.** See decision (a).

**Redrive bookkeeping:** every retry increments `attempts`; max 2 attempts
per item per run. After max attempts, item is `withheld` with the failure
mode preserved in `verification_results`.

---

## 14. Provider abstraction

**Interface (V1, Python):**

```python
class Provider(Protocol):
    name: str
    capabilities: set[str]   # e.g. {"json_schema", "tool_use"}

    async def complete(
        self,
        *,
        prompt_template_id: str,
        rendered_prompt: str,
        output_schema: dict,
        max_input_tokens: int,
        max_output_tokens: int,
        timeout_seconds: int,
        idempotency_key: str,
    ) -> ProviderResponse: ...
```

`ProviderResponse` carries `content_json`, `model`, `input_tokens`,
`output_tokens`, `latency_ms`, `provider_request_id`,
`schema_repaired_attempts`.

**Per-task provider selection** (V1 defaults):

| Task | V1 provider | V2 candidate |
|---|---|---|
| questions | claude_remote | llama_local (after parity) |
| predicted_answers | claude_remote | llama_local |
| follow_ups | claude_remote | llama_local |
| rubric.policy_area | claude_remote | llama_local |
| rubric.reversibility | claude_remote | llama_local |
| numeric extraction | claude_remote | rule-based only |

**Failure semantics:** every provider call has a hard timeout and a
schema-repair retry; transport errors mark the artifact `transport_failed`
and trigger run-controller redrive (separate from schema-repair).

**Provider implementations:**

- `claude_remote`: Anthropic SDK with tool-use enforcing JSON output
  schema. Existing pattern in `rag_api`.
- `codex_remote`: **deferred in V1 per decision (b).** Stays a stub; no CLI
  wrapper is built. A future real implementation may use the OpenAI Codex API
  only after explicit owner approval (separate billing relationship). Driving
  Codex CLI headlessly is out of scope and not pursued.
- `llama_local`: HTTP to local llama-server's OpenAI-compatible endpoint
  with `--json-schema` constraint or GBNF grammar. Validate that the
  loaded model honors the constraint reliably before promotion.

---

## 15. Local-LLM parity plan

**Parity test set:** 20 cached items from prior meetings, with `claude_remote`
outputs taken as gold. Stored under `tests/agenda_runs/parity/v1/`.

**Metrics per task:**

- Schema validity rate (must be 100%).
- Citation integrity rate (must be ≥ claude's, typically 100%).
- Numeric-claim integrity rate (must be ≥ claude's).
- Content-similarity to claude (cosine over 1024-dim embeddings of
  serialized JSON, weighted by section).
- Latency p50 / p95.

**Promotion gate per task:** parity ≥ 0.90, schema validity = 100%,
citation/numeric integrity ≥ claude's. Promotion is task-scoped: ranking
classifications might pass before question generation does.

**Quarterly rotation:** add 5 held-out items each quarter; retire the
oldest 5. Prevents the test set from becoming a training proxy.

**Provenance:** `model_provenance` on every artifact records which
provider produced it. A run can mix providers across tasks.

---

## 16. Notification design

**V1 — email:**

- Recipients in `agenda_runs.notification_targets (channel='email',
  address, enabled, weekly_default)`.
- On run state transition to `completed` or `failed`, controller writes
  one `notifications` row per recipient.
- Separate worker (`agenda-notification-sender`) drains the queue with
  retry/backoff, updates status.
- Body: short summary (run_id, meeting count, item totals, withheld count,
  top-3 ranked items), plus a link/path to the rendered report.
- **Watchdog:** a separate timer raises a system-level error if a
  `completed` run has zero successful sends within 1h.

**V2 — SMS:**

- Same `notifications` table, channel = `sms`.
- Triggered only on explicit success or specific failure conditions (e.g.,
  abort), not for every send.
- Provider gateway (Twilio etc.) integrated via the same provider
  abstraction shape (input contract, structured response, retry policy).

**Both channels:** notifications fire **only** after the run reaches a
terminal state. No mid-run pings in V1.

---

## 17. Recommended implementation order

Sequencing optimized for safe verifiability and clear "stop here" points if
any stage proves unsatisfactory.

| # | Issue | Why it goes here |
|---|---|---|
| 1 | DB schema design + migration plan (markdown only) | Foundation; no code yet. |
| 2 | JSON Schema files for all artifact kinds | Locks the contracts before anything writes. |
| 3 | Provider abstraction + `claude_remote` real impl | Enables Issue 7+ to be tested. |
| 4 | Run-controller skeleton with mock workers | State machine validated on synthetic data, no LLM. |
| 5 | Apply DB migrations to **dev DB only** | First real persistence. |
| 6 | Agenda watcher (dry-run mode only) | Detects but doesn't download; observe for 4 weeks. |
| 7 | Agenda downloader | Only after watcher passes a dry-run window. |
| 8 | Delta ingest (`boarddocs_loader --delta`) | Reuses existing pipelines safely. |
| 9 | Per-item analysis worker — questions only | First real LLM call inside the system. |
| 10 | Predicted answers + follow-ups | Adds the answer-and-follow-up loop. |
| 11 | Consequence ranker (rules + classifier calls) | Deterministic surface; should be quick once Issue 9 lands. |
| 12 | Verification gates + redrive | Single most important quality gate. |
| 13 | Renderer (markdown report) | Produces a report from verified artifacts only. |
| 14 | Email notification + watchdog | First end-to-end run can complete here. |
| 15 | Parity test harness | Establishes baseline before any local promotion. |
| 16 | SMS notification | Only after parity baseline is healthy. |
| 17 | Local promotion of first task (likely follow-ups) | Smallest semantic surface; safest first promotion. |
| 18 | Promote ranking classifications to local | Bigger scope; only after #17 holds. |

Cross-cutting prerequisites already on your tracker:

- **#27** (route semantics clarification) — must land before Issue 9 so
  the analysis worker has an unambiguous retrieval contract.
- **#29** (validator trailing bytes) — independent, can land anytime.
- **#30** (Stage 1 CI) — should land **before Issue 9** so all subsequent
  PRs are checked.

---

## 18. Plain-English decision points for Donald (RESOLVED)

> **Status:** all seven decisions were resolved by Donald on 2026-05-02 and
> applied throughout this memo. The original options and rationale are
> preserved below for historical reference. The chosen option for each is
> marked **DECIDED**.

### (a) Failed-item rendering policy — **DECIDED: withhold**

When verification rejects an item, do we:

- **(a-block)** block the entire report until every item passes?
- **(a-withhold)** render the report with that item replaced by a
  "Withheld — verification failed: <reason>" stub, listed at the top? ✶
  **DECIDED**

Reasons: blocking the whole run on one bad item makes the system fragile
during model upgrades or BoardDocs schema drift; surfacing failures
preserves transparency and lets you investigate without losing the rest of
the analysis. Verified items still get delivered; failed items are clearly
labeled as withheld with reasons.

### (b) `codex_remote` reality — **DECIDED: deferred (no CLI wrapper)**

What does `codex_remote` actually call?

- **(b-API)** OpenAI Codex / GPT API, separate billing from your CLI usage.
- **(b-CLI-wrapper)** Drive `codex exec ...` headlessly from a wrapper. **explicitly excluded**
- **(b-defer)** Stub indefinitely; rely only on `claude_remote` and
  `llama_local`. ✶ **DECIDED for V1**

A future API path is allowed only on explicit owner approval, after a
stable non-interactive interface is available. CLI wrapper option is
permanently off the table.

### (c) Per-run cost ceiling — **DECIDED: both limits**

- **(c-yes)** Soft cap (warn/flag) + hard cap (halt further model calls;
  remaining work marked withheld or deferred). ✶ **DECIDED**
- **(c-no)** No automatic abort.

Soft and hard dollar values are still TBD and live as Open Decision **O2**.

### (d) Rubric weights — **DECIDED: accept V1 as versioned draft**

Are the V1 weights in §12 (25/25/15/15/10/5/5 with policy-area
sub-weights) acceptable as the starting point, subject to revision via PR?

- **(d-accept-v1)** Yes, ship V1 weights; revise via PR if behavior is
  wrong. ✶ **DECIDED**
- **(d-redo)** No, give me different starting weights.

Weights live in `config/rubric/v1.toml`; every artifact stamps
`weights_version` so historical scores remain reproducible after weight
changes.

### (e) Predicted-answer display — **DECIDED: internal-only for V1**

- **(e-public)** Show in the public-facing report with a "speculative" label.
- **(e-appendix)** Show only in an appendix section.
- **(e-internal)** Suppress from the public report; use only as scaffolding
  for follow-up questions and store in DB. ✶ **DECIDED for V1**

Promotion criteria to lift the suppression are tracked under Open
Decision **O4**.

### (f) Resumability scope — **DECIDED: within-run + cross-run reuse**

- **(f-both)** Within-run resumability **and** cross-run artifact reuse. ✶ **DECIDED**
- **(f-within-only)** Within-run only.

Per-kind cross-run reuse defaults are listed in §5 (run-controller design).
Donald confirms or amends the per-kind reuse table in Issue 1.

### (g) Revision identity — **DECIDED: stable GUID + content hash**

- **(g-stable-guid)** Identity = `(meeting_id, item_guid)`; content_hash
  changes mean `revised`. ✶ **DECIDED** (combined with content_hash
  detection)
- **(g-content-hash)** Identity = content_hash; revisions are new items.

Stable GUID preserves identity across runs; `content_hash` detects revision.
A revision history table records `(item_guid, prior_content_hash,
new_content_hash, observed_at)` so per-item re-analysis is auditable.

---

## Top 5 risks (ranked)

1. **R4 — runaway LLM cost on a large meeting.** Mitigation: cost ceilings + per-item token budgets enforced before dispatch.
2. **R3 — `codex_remote` is not a real API surface today.** Mitigation: stub in V1; Donald decides API vs. CLI-wrapper before promotion to a runtime dependency.
3. **R2 — numeric-claim verification is harder than substring match.** Mitigation: two-stage check (substring + parse-with-tolerance) and explicit redrive policy.
4. **R5 — BoardDocs HTML/JSON schema drift breaks watcher silently.** Mitigation: structured "candidates_found" log + alert on sustained zero.
5. **R1 — predicted answers may invent.** Mitigation: speculation-grade tags + suppress from public report in V1.

## Top 5 next implementation steps

1. **Memo updated with Donald's decisions (a)–(g) on 2026-05-02.** Donald reviews the top-of-memo "Owner decisions applied" and the working assumptions A1–A8; Donald authorized a dedicated docs-only PR for this memo before any implementation work.
2. **Open Issue #31: DB schema + JSON Schemas for `agenda_runs/v1`** (design only; no DDL applied).
3. **Open Issue #32: Provider abstraction interface + `claude_remote` real implementation + contract test on synthetic prompts.**
4. **Open Issue #33: Run-controller skeleton with mock workers** (state-machine validation on synthetic data; uses dev DB).
5. **Coordinate with #27 and #30**: route-semantics clarification (blocks Issue #9 of the order above) and Stage-1 CI (blocks Issue #9 onward).

## Open decisions for Donald

§18 decisions (a)–(g) are **resolved** as of 2026-05-02. See "Owner
decisions applied" near the top of this memo for the resolutions.

Sub-questions that emerged from those decisions, plus working assumptions
A1–A8, are tracked under "Open decisions remaining" near the top
(items O1–O7).

---

## Safety attestation

- **Docs-only artifact:** this memo records architecture decisions and does
  not add runtime code.
- **Dedicated PR scope:** this memo is intended to be committed only as
  `docs/ai/architecture/agenda_analysis_roadmap_claude_review_20260502.md`
  in a dedicated docs-only PR.
- **No implementation authorization:** future code, schema, provider, run
  controller, notification, or deployment work requires separate issues and
  approval.
- **No production:** no Smeltor production services touched.
- **No Framework:** Framework directory not accessed.
- **No database:** no DB connections; no DDL applied; no migrations.
- **No OSPI:** OSPI scripts not run; OSPI corpus not read or modified;
  `download_ospi.py` not invoked; real OSPI data not validated.
- **No BoardDocs / KSD scraping:** no live downloads.
- **No ingestion / indexing / embedding:** none triggered.
- **No deployment / promotion / sync:** none performed.
