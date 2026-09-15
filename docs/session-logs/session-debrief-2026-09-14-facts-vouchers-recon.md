# Session Debrief — Voucher Fact Tables, Phase 0 Recon

**Date:** 2026-09-14
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers`, branched from `main` @ `29557ed`
**Git ref:** `da22888` (Phase 0 recon)
**Package:** `facts/vouchers/`
**Report:** `reports/facts-vouchers-recon-2026-09-14.md`

Scope was Phase 0 of the voucher fact-table build: inventory, PDF resolution and
text layer, format eras, signed warrant registers, vendor-name shape — then stop
for operator review. **Phase 1 was not started.** No `facts` table was created, no
row was written anywhere, no schema was applied.

---

## Decisions

**Phase 1 is stopped, and the stop is the brief's own.** Three of the four 2026
fixture months — 2026-06-24, 2026-07-22, 2026-08-26 — do not exist anywhere on this
machine. I searched four independent ways and all four came back empty. The scraped
corpus stops at 2026-03-25; the only later voucher artifact is a hand-placed
2026-05-27 directory that no ingest run has seen. Ten of the fourteen HARD fixture
totals belong to the missing months. **I did not weaken a fixture, substitute a
different month's figures, or proceed on the four that remain** — the brief says stop
if the fixture months are absent, and they are.

**The recon ran to completion anyway.** Stopping on the fixtures is not a reason to
stop reconnaissance. Every other Phase 0 question was answered in full, including
the full-pass text-layer survey over all 693 resolvable voucher PDFs, so the
operator's go/no-go decision is made with the whole picture rather than half of it.

**I verified the brief's fixtures against the documents rather than trusting them.**
All four 2026-03-25 HARD totals match the PDFs' own `TOTAL` lines exactly, and all
four reconcile when parsed — Δ 0.00 on each. The advisory line and check counts
match to the row (GF 1,533/384; ACH 1,101/241; Capital 40/29). A fixture I have not
checked against its source is a claim, not a check.

**Two changes to the brief's row regex, each forced by a measured failure.** The
brief's `\s{1,}` between vendor and date, and its date pattern with no internal
space, are what cost the $1,322.35 on 2026-03-25 ASB. Both changes were derived from
the one failing row, not anticipated, and both are pinned by tests that fail against
the brief's original pattern. The original is retained in source as `ROW_RX_BRIEF`
so the report can say precisely what it does and does not catch.

**Deduplication is by content hash, not by name or path.** The same money is filed
under `GF Vouchers 3-25-26.pdf` in one corpus root and
`General Fund Vouchers 03-25-26.pdf` in another, and several meetings are scraped
twice under differently punctuated slugs. 1,266 voucher-directory PDFs collapse to
859 distinct files. A name-based rule would have counted the same warrants twice.

**No LLM in any amount, date, vendor, check-number or total path.** Everything in
this session is `pdfplumber` extraction, regular expressions and exact `Decimal`
arithmetic. No `category` column was written, because no table was written.

## What changed

New package `facts/vouchers/` (9 files) and one report. Nothing else in the repo was
touched; `facts/minutes/` was read for conventions and not modified.

| File | Purpose |
|---|---|
| `facts/vouchers/locators.py` | Path resolution across **two** stale roots, `PdfText` with exact per-page offsets, quote builder. |
| `facts/vouchers/classify.py` | Fund and document-class rules over a closed vocabulary; title separator normalization. |
| `facts/vouchers/census.py` | Merged inventory across `documents` and the corpus on disk, deduplicated by SHA-256. |
| `facts/vouchers/db.py` | `READ ONLY` Postgres access; credentials injected at runtime. |
| `facts/vouchers/recon_phase0.py` | The recon itself. Full extraction pass, cached by content hash. |
| `facts/vouchers/test_vouchers.py` | 99 tests. |
| `requirements.txt` / `-dev.txt` / `-lock.txt` | Same pins as `facts/minutes`, so the two packages cannot disagree about how a PDF is read. |

Database: **read-only queries only.** No `CREATE`, no `INSERT`, no `UPDATE`, no
`DELETE`, anywhere, in any schema.

## Findings

**Three of four fixture months are missing, and the cause is upstream.** The newest
scraped meeting directory in either corpus root is 2026-03-25; the newest voucher PDF
with a `documents` row is 2026-02-11. Today is 2026-09-14 and the next voucher night
is 2026-09-23. **The voucher record on this machine is six months stale.** No parser
fixes that.

**`documents.file_path` is stale under two roots, not one — and `facts/minutes` only
handles one of them.** The bulk 2005–2026 corpus rewrites to the backup archive; a
later 2026 re-scrape rewrites to `boarddocs/data_DO_NOT_LOAD/`. The minutes package's
single generic `/home/donald/` → archive rewrite produces a non-existent path for the
second root, and a missing rewrite is indistinguishable from a missing file. With
both rewrites, tried most-specific-first, **693 of 693 voucher PDFs resolve.** This
is a live defect in shipped minutes code, flagged and not touched.

**44 voucher PDFs have no `documents` row at all** — including *every file* in both
2026 sets that carry the reachable fixtures. `locator_document_id uuid REFERENCES
documents(id)` cannot be satisfied for them, so the locator contract as written does
not survive contact with the 2026 data.

**Before 2023 the detail listings print no total.** 303 listings, 2017–2022, with
vendor rows and nothing to reconcile against. "Reconcile or flag" has no *it* for
those years. The recaps that could supply one are scanned images until 2020, and the
signed registers do not begin until late 2021.

**2010–2016 has no machine-readable voucher data at all.** 106 `Warrant_Recap.pdf`
files, zero with a text layer. Seven years where the record shows that money was
approved and how much in total, and nothing about who was paid.

**Era C listings are cumulative, and this is the sharpest edge found.** The
2021-02-10 and 2021-03-10 Transportation listings both parse to exactly
$1,175,094.00, while their recaps state $783,396.00 and $391,698.00 —
`783,396.00 + 391,698.00 = 1,175,094.00`. The same pattern holds at 2020-06-24.
**Summing Era C listings across meetings double- and triple-counts real money.**
Until a deduplication rule is designed, no 2020–2022 figure should be said aloud.

**The signed register is genuinely independent evidence, proven arithmetically.**
On 2026-03-25 the ACH listing total equals the sum of every accounts-payable
direct-deposit line across General, Capital, ASB and Custodial — **Δ 0.00**. Capital
equals its warrant ranges plus its P-card line — **Δ 0.00**. ASB differs by exactly
$581.50, which is two single-warrant register lines (`418227` at $30.00 and `418236`
at $551.50) that appear on the register and not in the listing. I have not determined
why they are withheld and have not guessed.

**ACH is a payment method, not a fund.** The same arithmetic proves the ACH listing
spans four funds. It is modelled as a fund only because the district publishes it
with its own total.

**The control-total contract in the brief is wrong in one respect.**
`sum(invoice_amount) == sum(deduped check_amount)` is false on real data, benignly:
P-card pseudo-checks carry a statement total that exceeds the transactions itemised
(2026-05-27 ASB, check `9264000032`, Δ $468.40), and credits ride on sentinel check
numbers (`8888888888`, −$2,887.19). 127 of the sets that parse would fail that
assertion, every one for a correct reason. `sum(invoice_amount) == stated_total` is
the hard check; the other is a second control total whose difference should be
stored.

**The corpus publishes nine funds, not five.** Transportation Vehicle (30 listings),
Custodial (9), Permanent (1) and a short-lived Vision Trust fall outside the brief's
five-value vocabulary. A five-value `CHECK` constraint would reject 40 real sets.

**Regex `\b` does not fire next to an underscore, and it cost 118 artifacts.** Before
`normalize_title` existed, every `Warrant_Recap_3-8-17.pdf` and
`Capital_Projects_Vouchers_2-8-17.pdf` fell through unclassified while the
space-separated spelling classified correctly. Found by inspecting the unclassified
list rather than by reading the pattern; pinned by tests.

**One district labelling error found.** `TVF_Vouchers_02-11-26.pdf` (doc
`a2325718-…`) prints `Permanent Fund Warrants` as its header. This is why the printed
phrase must beat the file name where they disagree.

## Open Items

### 1. D1 — the three missing fixture months *(blocking Phase 1)*

- **Symptom:** 10 of 14 HARD fixture totals cannot be verified against a document.
- **Diagnosis:** The PDFs are not on this machine. Confirmed four ways (report §A4).
  BoardDocs fetching is out of scope and `curl` is blocked by policy.
- **Next step:** Operator chooses — (a) supply the 06-24 / 07-22 / 08-26 packets,
  (b) re-scope the fixtures to 2026-03-25 + 2026-05-27 (nine to-the-cent fixtures
  across two months, including a live reproduction of the double-TOTAL pitfall), or
  (c) fix the scrape first. (a) and (c) are not exclusive; (b) blocks neither.
- **Urgency:** **High** — nothing else in Phase 1 starts until this is answered.

### 2. D2 — extend the fund vocabulary from five to nine

- **Next step:** Confirm `Transportation`, `Custodial`, `Permanent` join the `CHECK`
  constraint. Recommended.
- **Urgency:** Medium — cheap now, a migration later.

### 3. D3 — locators for the 44 PDFs with no `documents` row

- **Next step:** Approve nullable `locator_document_id` plus `source_path`,
  `source_sha256` and `agenda_item_document_id` on `voucher_set`. The alternative —
  re-ingesting those files into `documents` — writes outside `facts` and is forbidden
  by this task's rules; I will not do it without an explicit instruction.
- **Urgency:** **High** — it blocks the schema, and it affects the fixture sets.

### 4. D4 — what to do with 2017–2022

- **Symptom:** 303 listings that can be parsed and cannot be reconciled, of which the
  2020–2022 subset is additionally cumulative.
- **Next step:** Operator chooses scope. I lean toward Phase 1 covering 2023-onward
  first and treating the earlier era as a follow-on — it gets correct numbers in
  place for 2026-09-23 without risking a double-counted figure being quoted.
- **Urgency:** Medium.

### 5. D5 — the export's personal-name rule

- **Symptom:** 654 of 3,527 distinct vendor names are `Surname, Given` and catchable.
  The corpus also pays individuals as `Given Surname`, which **no deterministic rule**
  separates from a two-word company.
- **Next step:** Approve an allow-list (watch-list vendors plus names carrying a
  corporate suffix). It is conservative and will also withhold some small businesses
  from the export; they stay in the table.
- **Urgency:** Medium — it is a privacy control, not a convenience.

### 6. D6 — carry the two-root path fix back to `facts/minutes`

- **Next step:** Apply the ordered `PATH_REWRITES` from
  `facts/vouchers/locators.py` to `facts/minutes/locators.py`. Out of scope this
  session; not touched.
- **Urgency:** Medium — it is silently wrong today for any 2026 re-scrape document.

### 7. D7 — accept the corrected control-total contract

- **Next step:** Confirm `sum(invoice_amount) == stated_total` as the hard
  reconciliation, with `sum(deduped check_amount)` stored and its difference recorded.
- **Urgency:** Medium — it changes what `reconciled` means.

### 8. Parser defects surfaced but not fixed (Phase 1 work)

- Five out-of-balance sets parse to fractional cents (e.g. 2023-05-24 Capital at
  `1,874,334.4036`), and 2025-04-23 GF parses to $127.4M against a stated $10.9M.
  Both are the permissive amount pattern capturing across a column boundary. They
  surface as **out of balance** rather than as silently wrong numbers, which is the
  design working — but they are real and unfixed.
- 268 of 446 listings parse zero rows with the current-era regex. A stratified sample
  of 37 shows **30 recover** once a `$` prefix and the older column orders are
  admitted. The corpus needs four era parsers; it is not unparseable.

### 9. Two corrupt PDFs

One 2017 and one 2022 file could not be opened at all. Counted as `unreadable`, not
as missing.

## Documentation impact

- `CLAUDE.md` documents the database as `qorvault/qorvault`; it is
  `boarddocs/boarddocs`. Flagged only — CLAUDE.md files are never modified.
- `documents.file_path` needs **two** rewrites, not one. Anything reading that column
  with only the minutes package's rule is silently broken for 2026 documents.
- The pre-2019 corpus still exists only inside
  `/home/donald/qorvault-dev-archive/framework-backup/`. Fourteen years of financial
  records with a backup as their sole copy — the same durability risk the 2026-09-13
  minutes debrief raised, unchanged.
- The live corpus directory is named `data_DO_NOT_LOAD`, which suggests somebody
  deliberately removed it from the ingest path. I only read from it.

## System state summary

- **Nothing was written outside `facts/vouchers/`, `reports/` and `docs/`.** No
  `facts` table was created. No row was written or deleted in any schema.
- `documents`, `chunks`, Qdrant, `rag_api`, `ksd-boarddocs-rag` and production: never
  written. Every corpus read used a `READ ONLY` Postgres session.
- Branch `claude/facts-vouchers`, one commit ahead of `main` @ `29557ed`. **Not
  pushed, not merged.** Commit `da22888` is SSH-signed and verified.
- Pre-commit hooks all ran and passed — private-key detection, large files,
  branch guard, merge conflicts, debug statements, secret detection, bandit, ruff,
  ruff-format, interrogate. **None bypassed.** No hook blocked anything this session.
- 99 tests passing. `ruff check` clean, `ruff format` clean, `interrogate` 98%
  (floor 80%), `bandit` 0 medium / 0 high.
- Virtualenv at `facts/vouchers/.venv`, gitignored, built from
  `requirements-lock.txt` only. `pip-audit` reports **no known vulnerabilities**.
- Credentials injected at runtime from the container config. `.env` was neither read
  nor edited; no password was printed.

## Regenerating

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Inventory only -- no PDF extraction, ~2 s.
.venv/bin/python recon_phase0.py --no-probe

# Full recon. First run opens all 693 voucher PDFs (~25 min) and caches by
# content hash under _build/; later runs are ~2 s.
.venv/bin/python recon_phase0.py --progress \
  --json-out _build/recon_findings.json \
  > _build/recon_stdout.json 2> _build/recon_progress.log

.venv/bin/python -m pytest test_vouchers.py -q     # 99 passed
.venv/bin/pip-audit -r requirements-lock.txt       # no known vulnerabilities
```

Rebuilding the venv in a fresh worktree:

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-lock.txt
```

There is no export to regenerate yet: `export_cycle.py`, the views and the sample
files are Phase 1 deliverables and Phase 1 has not been authorised. The Phase 0
report's Appendix A carries every command run this session.
