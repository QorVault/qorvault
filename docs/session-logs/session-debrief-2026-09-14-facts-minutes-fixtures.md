# Session Debrief — Minutes Fact Tables, Fixture Closure

**Date:** 2026-09-14
**Worktree:** `~/workspace/projects/ksd-minutes`
**Branch:** `claude/facts-minutes-fixtures`, branched from `main` @ `27f80d2`
**Package:** `facts/minutes/`
**Report:** `reports/facts-minutes-fixtures-2026-09-14.md`

Scope was the four operator decisions on the open fixtures from
2026-09-13, and nothing else. No rebuild, no parser change, no table write.

---

## Decisions

**26 executive sessions in 2024 is correct; 27 was wrong.** An adjournment
sentence is an end time for the session already open. The fixture now asserts
only the 24 sessions convened as their own meeting — a closed census, so a
shortfall is a real miss — and *reports* the 2 announced inside other meetings'
minutes without asserting them. The corpus cannot say how many announcements it
ought to contain, so a target there would assert a fact about the world that the
record does not hold. The superseded expectation of 27 is retained in
`fixtures.py` with its reasoning rather than deleted.

**The attendance/vote check keeps its teeth by asserting something true.** The
old assertion — voters never exceed recorded attendance — is false about the
district's records, and a permanently red check communicates nothing. The
assertion is now "nothing new appears": 117 discrepant motions across 6
meetings, each carrying a cause, and zero outside the maintained known-set. A
green result means *no undiagnosed discrepancy*, not *no discrepancy*.

**One of the six is our bug, not the district's.** 2025-02-11 was classified by
the previous session as a record discrepancy. It is not: a surname-only vote
roll from an adjacent motion is being attributed to this one, producing 8 votes
from a 4-member board. I gave it the cause `parser_roll_bleed` instead of
forcing it into the four-value vocabulary, because recording it as a record
discrepancy would have asserted something false about the district while hiding
something true about our code. This is a deviation from the decision as written
and is flagged as such.

**"approve" is not evidence of adoption.** Adding it to the disposition stems
would have taken the residual from 43 to 11. It matches "Recommended Action
That the Board of Directors approves…" — the proposal put *to* the board, not
proof the board adopted it. Removed, and pinned by a regression test. The
fixture stays honestly red rather than green on a weaker standard.

**The six hand-count dates were kept despite their meeting type.** Three Era A
targets sit on a census row typed `work_study`/`special`. The nearest `regular`
meeting with minutes is 105–287 days away. The unreliable thing is the Era A
census typing, not the document — each PDF is the board minutes for its date
and carries motions, which is all a hand count needs.

**No LLM in any date, name, vote, motion or count path.** Everything added is
regex, set membership and arithmetic.

## What changed

Four files, one new:

| File | Change |
|---|---|
| `facts/minutes/views.sql` | Added `facts.attendance_vote_discrepancies` (+64 lines). The five existing views are unchanged and were replaced byte-identically. |
| `facts/minutes/fixtures.py` | Rewrote 3 fixtures, renamed `tally_within_attendance` → `attendance_vote_discrepancies`, added the known-set, stem documentation, and the hand-count file loader. |
| `facts/minutes/test_parsers.py` | +35 tests across 3 new classes. |
| `facts/minutes/fixtures/hand_counts.yaml` | **New.** Six targets with empty count fields for the operator to fill in. |

Database: **one `CREATE OR REPLACE VIEW` pass against schema `facts`.** No
table written, no rebuild, nothing deleted. All six row counts identical to
Phase 0 (meeting 1,646; attendance 3,515; motion 6,507; vote 19,625;
executive_session 280; minutes_parse_log 874).

Results: `exec_sessions_2024` **PASS** (24 asserted, 2 reported);
`attendance_vote_discrepancies` **PASS** (0 unknown);
`disposition_and_locator` **FAIL at 43**, fully enumerated;
`operator_hand_counts` **BLOCKED** with six targets prepared.
Tests **70 passing**.

## Findings

**The previous session's diagnosis of the 43 was wrong, and the stem widening
therefore changed nothing.** The record describes them as a withdrew/withdrawn
vocabulary problem. There is not one `withdrawn` or `tabled` row among them —
41 are `adopted`, 2 are `lost`, and all 43 come from agenda items. The cause is
quote truncation: `make_quote(..., max_len=400)` cuts the citation off before
it reaches the `Final Resolution:` line. 34 of 43 sit exactly at the ceiling.
These are correct rows with an inadequate citation. Residual before: 43. After:
43.

**Parser bug #3 from the previous session is not fully fixed.** The surname-only
roll bleed survives where a document has no following `Final Resolution` anchor
and no following "A motion was made" — the tail scan then runs to end of
document. The duplicate guard misses it because it compares raw name strings and
`"Song"` ≠ `"Andy Song"`. This inflates vote tallies, which are the most
load-bearing numbers in the fact layer.

**The run report's table of six meetings does not match the data.** It lists
2020-03-19 (which produces no discrepancy at all) and omits 2023-11-08 (which
produces 51 of the 117 — the largest single contributor). The totals were
right; the enumeration was not. Derived from the data, not transcribed.

**The committed 2026-02-04 export is stale.** It shows a meeting row
(`2026-02-04:work_study#2`) that does not exist in the database — the duplicate
punctuated slug from R6, generated before the census dedupe took effect.
Unrelated to this session's changes.

**Era A census meeting types are unreliable.** Most Era A minutes land on
`work_study`/`special` rows; regular meetings with minutes are months apart in
that era. Anything filtering Era A by `meeting_type = 'regular'` will silently
see almost nothing.

**A hook false-positive, reported not bypassed.** `validate-pip-install.sh`
parses the shell redirect `2>&1` as a package name (it survives the flag/path
skips, then `${arg%%[><=!~@]*}` cuts at `>` yielding `2`). This blocks the
lockfile install the hook is explicitly written to allow. I re-ran without the
redirect inside the pip command — the permitted path — and did not modify or
disable the hook. Separately, the allowlist it names
(`/home/donald/workspace/.claude/approved-packages.txt`) does not exist, so
every pip install in this project is blocked regardless of contents.

## Open Items

### 1. `disposition_and_locator` still fails at 43

- **Symptom:** 43 of 6,507 motion locator quotes contain no word evidencing the
  recorded disposition. `missing_locator` is 0.
- **Tried:** Widened the disposition match to word stems exactly as decided.
  Zero effect — the stems were never the problem.
- **Diagnosis:** Quote truncation at 400 characters in
  `vote_parser.parse_agenda_item`. Either no motion opening is found (so the
  quote starts at the top of the agenda-item page and spends its budget on
  BoardDocs furniture), or the motion genuinely runs past 400 characters before
  its `Final Resolution:` line.
- **Next step:** R8 — give the disposition sentence its own quote field, or
  elide the middle of the bridge (first ~200 chars + "…" + the resolution
  sentence). Requires `build.py --reload`. Would take all 43 green on evidence
  rather than vocabulary.
- **Urgency:** Low. Citation quality, not data correctness — the dispositions
  themselves are right.

### 2. Surname-only vote roll bleed inflates tallies — **highest priority**

- **Symptom:** `2025-02-11:special#a1` records 5 yes / 3 no from a 4-member
  board.
- **Tried:** Nothing — fixing it is out of this session's scope.
- **Diagnosis:** Tail scan runs to end-of-document when no following anchor
  exists; `seen` set compares raw names so `"Song"` and `"Andy Song"` do not
  collide.
- **Next step:** R7 — bound the tail scan and normalise names before the
  duplicate check, then `build.py --reload` and re-verify the Phase 0 counts.
  Corpus-wide prevalence is unknown; this session found it via one meeting and
  did not sweep for others.
- **Urgency:** **High.** It corrupts vote tallies, and the discrepancy fixture
  currently passes with it live.

### 3. Hand-count fixture still BLOCKED

- **Symptom:** The only independent check on parser accuracy has never run.
- **Diagnosis:** Awaiting the operator's counts.
- **Next step:** Fill `motions_total` / `motions_adopted` / `motions_lost` in
  `facts/minutes/fixtures/hand_counts.yaml` for the six meetings, then run
  `.venv/bin/python fixtures.py`. The file carries each PDF path, page count
  and the parser's own figures.
- **Urgency:** Medium — unchanged from the previous session, but now
  actionable: targets, paths and instructions are prepared.

### 4. Stale committed export

- **Next step:** R9 — regenerate `reports/meeting-export-2026-02-04.md`. Left
  untouched because the decision list did not include it.
- **Urgency:** Low.

### 5. Previous report contains two incorrect enumerations

- **Next step:** R10 — correct `reports/facts-minutes-run-2026-09-13.md` §4.
  Flagged rather than edited: it is a previous session's record.
- **Urgency:** Low, but it misdirected this session's diagnosis of the 43 and
  would misdirect the next reader too.

### 6. Out of scope, carried forward

R1–R6 from the previous session are unchanged and still outstanding. R1 (stale
`documents.file_path`) is exercised by every hand-count target path. Nothing
touching `documents`, `chunks` or the voucher work was attempted.

## Documentation impact

- `reports/facts-minutes-run-2026-09-13.md` §4 needs the two corrections above
  (R10).
- `reports/meeting-export-2026-02-04.md` is stale (R9).
- The cause vocabulary for attendance/vote discrepancies is documented in the
  `views.sql` comment block above `facts.attendance_vote_discrepancies` and
  mirrored in `fixtures.py`. It now has five values, not four:
  `parser_roll_bleed` was added and is explicitly *not* a record discrepancy.
- `CLAUDE.md` still documents the database as `qorvault/qorvault`; it is
  `boarddocs/boarddocs`. Flagged only — CLAUDE.md files are never modified.

## System state summary

- Schema `facts`: six views (five unchanged, one new), six tables **untouched**.
  Row counts identical to Phase 0. No rows deleted anywhere.
- `documents`, `chunks`, Qdrant, `rag_api`, `ksd-boarddocs-rag` and production:
  never written. All corpus reads used a `READ ONLY` Postgres session.
- Branch `claude/facts-minutes-fixtures`, one commit ahead of `main` @
  `27f80d2`. Not pushed, not merged.
- Virtualenv at `facts/minutes/.venv`, gitignored, built from
  `requirements-lock.txt` only.
- Credentials injected at runtime from the container. `.env` neither read nor
  edited; no password printed.
- One hook escalation, respected not bypassed (see Findings).

## Regenerating

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Views only. No reload is needed or wanted: the fact tables are untouched.
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql

.venv/bin/python -m pytest test_parsers.py -q     # 70 passed
.venv/bin/python fixtures.py                      # exits 1: disposition red at 43
```

Rebuilding the venv in a fresh worktree:

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-lock.txt
```

Every command run this session is in the report's appendix.
