# Minutes Fact Tables — Closing the Four Open Fixtures

**Date:** 2026-09-14
**Worktree:** `~/workspace/projects/ksd-minutes`
**Branch:** `claude/facts-minutes-fixtures` (from `main` @ `27f80d2`)
**Database:** `boarddocs` on `boarddocs-postgres`, schema `facts`
**Package:** `facts/minutes/`

**Status: three of the four decisions applied and green. One —
`disposition_and_locator` — is applied exactly as decided and is still red at
43, because the operator's stem widening does not touch the actual cause. All
43 are enumerated in §4. Nothing was loosened.**

| Fixture | Before | After |
|---|---|---|
| `exec_sessions_2024` | FAIL (26 vs 27) | **PASS** — 24 asserted, 2 reported |
| `tally_within_attendance` → `attendance_vote_discrepancies` | FAIL (117 of 2,613) | **PASS** — 117 discrepancies, 0 outside the known-set |
| `disposition_and_locator` | FAIL (43 of 6,507) | **FAIL (43)** — unchanged, all 43 enumerated |
| `operator_hand_counts` | BLOCKED (no targets) | **BLOCKED** — six targets prepared |

Unit tests: **70 passing** (35 before, 35 added).

---

## 1. Phase 0 — recon

All three STOP gates cleared.

**Worktree.** Created fresh off `main`. `main` and the previous session's branch
`claude/feat-facts-minutes` both point at `27f80d2`, and `facts/` is fully
present on `main`, so `facts/minutes/` in this worktree is byte-identical to
`main` (`git diff main HEAD -- facts/` is empty).

**Virtualenv.** Rebuilt from `requirements-lock.txt` into `facts/minutes/.venv`.
Nothing outside the lock file was installed.

**Tests.** `35 passed in 0.04s` — matches the run report.

**Row counts — all six match exactly, no STOP:**

| Table | Run report | Observed |
|---|---:|---:|
| `facts.meeting` | 1,646 | 1,646 |
| `facts.attendance` | 3,515 | 3,515 |
| `facts.motion` | 6,507 | 6,507 |
| `facts.vote` | 19,625 | 19,625 |
| `facts.executive_session` | 280 | 280 |
| `facts.minutes_parse_log` | 874 | 874 |

**Fixtures reproduce exactly, no STOP:** `exec_sessions_2024` 26 vs 27;
`tally_within_attendance` 117 of 2,613; `disposition_and_locator` 43 of 6,507;
`operator_hand_counts` blocked.

## 2. Decision 1 — `exec_sessions_2024`: accept 26 — **PASS**

The adjournment sentence in the 2024-09-11 minutes ("The Executive Session was
adjourned at 10:25 p.m.") is an end time for the session already open, not a
third session. The fixture now asserts only what the corpus can actually be
held to:

- **HARD, asserted:** executive sessions convened as their own meeting in the
  2024 census have a row in `facts.executive_session` — **expected 24,
  actual 24**.
- **REPORTED, not asserted:** sessions announced inside another meeting's
  minutes in 2024 — **2**, as expected.
- Closings and adjournments are never counted as sessions. The one 2024 closing
  is reported under `closings_never_counted_as_sessions` so it stays visible.

The distinction matters and is the reason the split was worth making. The
census is a closed list, so a shortfall against 24 is a real miss and deserves
an assertion. The announcements are not a closed list — the corpus cannot tell
us how many executive sessions the district *should* have announced inside
other meetings' minutes, so asserting a number there would be asserting a fact
about the world that the record does not contain.

The original expectation is retained in `fixtures.py`, not deleted:

```python
SUPERSEDED_EXEC_SESSIONS_2024_EXPECTATION = 27
```

with the one-line reasoning that 27 is only reachable by counting an
adjournment sentence as a session, which would double-count every session whose
close happens to be recorded. It is also echoed into the fixture's own output
under `superseded_expectation` / `superseded_reason`, so anyone reading a run
log sees the history without reading the source.

## 3. Decision 2 — `attendance_vote_discrepancies`: keep the check, change its role — **PASS**

New view `facts.attendance_vote_discrepancies` (in `views.sql`): one row per
(meeting, motion) where recorded voters exceed recorded present, carrying the
full locator columns and a `cause`. The fixture now asserts **no rows outside
the six known meetings**. It reports 117 discrepant motions across 6 meetings
and 0 unknown.

A green result here means *"no undiagnosed discrepancy"*, not *"no
discrepancy"*. That is the whole point of the change: the previous assertion —
that voters never exceed recorded attendance — is simply false about the
district's own records, and holding it produced a permanently red check that
told the reader nothing. The list of six with their causes is the useful
output.

| Meeting | Present | Max cast | Motions | Cause |
|---|---:|---:|---:|---|
| 2022-06-29 special | 1 | 4 | 4 | `presiding_only` |
| 2022-10-05 special | 4 | 5 | 1 | `attendance_short` |
| 2023-11-08 regular | 4 | 5 | 51 | `status_excluded` |
| 2023-12-13 regular | 4 | 5 | 36 | `board_transition` |
| 2024-07-10 special | 3 | 4 | 24 | `status_excluded` |
| 2025-02-11 special | 4 | 8 | 1 | `parser_roll_bleed` |

The known-set is maintained in `fixtures.py` as
`KNOWN_ATTENDANCE_VOTE_DISCREPANCIES`, keyed by `meeting_id` with its date and
cause. The fixture also reports `known_but_no_longer_present`, so if the
underlying data moves and a known meeting stops being discrepant, the stale
entry surfaces instead of quietly rotting.

### 3a. Two corrections to the run report's list of six

The run report §4 table does not match the data. Two of its rows are wrong:

- **2020-03-19 special is listed but does not violate.** It produces no
  discrepancy rows at all.
- **2023-11-08 regular violates but is not listed.** It is the single largest
  contributor, 51 of the 117 motions.

The other four meetings match. The totals in the report (117 violations, 6
meetings) were right; the enumeration of *which* six was not. The set above is
derived from the data, not transcribed.

### 3b. Deviation: 2025-02-11 is a parser defect, not a record discrepancy

**This is the most important finding in the session and it is a deviation from
the decision as written.**

The decision specified four cause values (`board_transition` /
`attendance_short` / `presiding_only` / `status_excluded`), all of which
describe ways the *district's record* can be internally inconsistent. Five of
the six meetings fit. 2025-02-11 does not — it is a live bug in our parser, and
recording it as a record discrepancy would have been asserting something false
about the district while hiding something true about our code.

Motion `2025-02-11:special#a1` carries a tally of 5 yes / 3 no = 8 votes from a
board of 4 people:

```
Tim Clark      yes  off=814   "Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song"
Meghin Margel  yes  off=814   "Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song"
Donald Cook    yes  off=814   "Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song"
Andy Song      yes  off=814   "Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song"
Song           yes  off=1140  "Yea: Song"
Clark          no   off=1150  "Nay: Clark, Cook, Margel"
Cook           no   off=1150  "Nay: Clark, Cook, Margel"
Margel         no   off=1150  "Nay: Clark, Cook, Margel"
```

The first four are the canonical full-name roll for this motion. The last four
are a **surname-only roll belonging to an adjacent motion**, swept in because
the tail scan in `vote_parser.parse_agenda_item` runs to the end of the
document when there is no following `Final Resolution` anchor and no following
"A motion was made". The duplicate-suppression `seen` set does not catch it
because it matches on the raw name string, and `"Song"` is not `"Andy Song"`.

This is parser bug #3 from the previous session — surname-only pre-rolls being
attributed to the wrong motion — **not fully fixed**. It was found then,
patched for the common case, and survives in this shape.

I gave it the cause `parser_roll_bleed` rather than forcing it into
`status_excluded` (which is how the run report classified it, and which is
wrong — 2025-02-11 records 4 present and 4 directors, with no absent or excused
status involved at all). The view's comment block states explicitly that
`parser_roll_bleed` is **not** a record discrepancy.

**The fixture passes with this row in the known-set.** I want to be direct
about what that means: a hard check is green while a known parser defect is
live. I judged that better than the alternative of a silent false
classification, but it is a real cost, and the fix belongs in §7 as a
recommended change rather than something I did unasked — repairing it changes
`facts.vote` and `facts.motion` corpus-wide and would move the Phase 0 numbers
this session was told to hold fixed.

## 4. Decision 3 — `disposition_and_locator`: widen to stems — **still FAIL (43)**

Applied exactly as decided. `DISPOSITION_WORDS` now documents and covers the
stems requested:

| Disposition | Pattern | Covers |
|---|---|---|
| `adopted` | `carri\|carry\|pass\|adopt` | carried / carries / carry / passed / passes / pass / adopted / adopts / adopt |
| `lost` | `fail\|lost\|lose\|died\|not\s+carri` | failed / fails / fail / lost / loses / died / not carried |
| `tabled` | `tabl` | tabled / tables / table |
| `withdrawn` | `withdr` | withdrew / withdrawn / withdraws / withdraw |

**The residual is unchanged: 43 before, 43 after.** The widening matched
nothing new, and this is not a surprise once the residual is actually
inspected: the previous session's diagnosis of these 43 was wrong.

### 4a. The recorded diagnosis does not describe these rows

The run report §4 and the debrief both describe the residual as a vocabulary
problem — *"a motion marked `withdrawn` because the minutes say the mover
'withdrew' it"*. There is **not one such row**. The actual population:

| Property | Value |
|---|---|
| Dispositions involved | `adopted` (41), `lost` (2) |
| `withdrawn` or `tabled` rows | **0** |
| Source | `agenda_item` (43 of 43) — no minutes-sourced row is affected |
| Distinct documents | 37 |
| `missing_locator` | 0 |

The real cause is **quote truncation, not vocabulary.** In
`vote_parser.parse_agenda_item` the motion quote is built as:

```python
quote=make_quote(text, start, anchor.end(), max_len=400)
```

`make_quote` hard-truncates at `start + max_len`. The span is meant to bridge
the motion opening to its `Final Resolution:` line so the citation lands on the
disposition — but when that span exceeds 400 characters, the quote stops before
the disposition ever appears. 34 of the 43 quotes are at the 400-character
ceiling.

Two shapes produce it:

1. **No motion opening found** (the majority). When no "A motion was made"
   precedes the resolution, `start` falls back to the previous anchor's end —
   0 for the first motion in a document. The quote then begins at the top of
   the agenda-item page ("Agenda Item Details Meeting Jun 26, 2019 …") and
   spends its whole 400-character budget on BoardDocs page furniture.
2. **A genuinely long motion.** Consent-agenda and multi-part motions run past
   400 characters before reaching "Final Resolution".

These are *correct rows with an inadequate citation*. The disposition is right;
the quote does not prove it.

### 4b. Why I did not widen further to make it green

An earlier iteration of this change added `approv` to the `adopted` stems. That
dropped the residual from 43 to 11 and I removed it, because it is exactly the
move this fixture exists to prevent. The text it matches is

> Recommended Action **That the Board of Directors approves** the bids for Old
> Panther Lake Elementary Demolition…

which is the proposal *put to* the board, not evidence the board adopted it. 32
truncated citations would have gone green without a single one of them gaining
a word that shows an outcome. A regression test now pins this: see
`TestDispositionStems.test_stem_rejects_non_evidence`.

### 4c. Full enumeration of all 43 residual cases

Per the decision — no count is accepted without the list. Page is the true PDF
page from `pdfplumber`; agenda items are single scraped web pages, so page 1 is
a fact about them. Offset is the character offset into the extracted text.
Quotes are verbatim, truncated to 150 characters for table width only; the
fixture emits them in full.

| # | motion_id | disp | document_id | page | offset | quote (verbatim, truncated to 150 ch for the table) |
|---:|---|---|---|---:|---:|---|
| 1 | `2018-11-14:regular#a4` | adopted | `313e1bf5-efa8-4d40-a7e3-ecb3624c5f7b` | 1 | 0 | Agenda Item Details Meeting Nov 14, 2018 - Regular Meeting - 7 p.m. Category 5. Board of Directors - Discussion & Approval Subject 5.04 Inclusive Educ… |
| 2 | `2018-12-12:regular#a1` | adopted | `1f5c1929-98c1-45a1-80df-10bb6777ca66` | 1 | 143 | A motion was made to approve the addition of agenda item 3.01 Fiscal Recovery Task Force Application Extension; the addition of consent agenda item 5.… |
| 3 | `2019-06-26:regular#a1` | adopted | `06a33aff-c9ca-421c-bd73-ff18c0aa6d09` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.20 Old Panther La… |
| 4 | `2019-06-26:regular#a10` | adopted | `492b4d58-9348-409b-a4b5-8bb6ca1fdcc4` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.07 Gates Foundati… |
| 5 | `2019-06-26:regular#a11` | adopted | `0a028113-bacb-4506-9caf-fbe13bbb2557` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.11 Hobsons - Navi… |
| 6 | `2019-06-26:regular#a12` | adopted | `af4a9217-5857-49b2-875d-569bff2efe76` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.13 Inclusive Educ… |
| 7 | `2019-06-26:regular#a13` | adopted | `2e8e94bd-ab68-46e7-abb7-402a5ec588e9` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.22 Change Order N… |
| 8 | `2019-06-26:regular#a14` | adopted | `a54fdfa2-e7d9-4724-94ff-3c707f632467` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.05 Affirmative Ac… |
| 9 | `2019-06-26:regular#a15` | adopted | `831de5fe-92d7-4c3b-8d52-f92649c035b4` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.08 School Climate… |
| 10 | `2019-06-26:regular#a16` | adopted | `3b4f20e9-0e61-49f8-a7d4-9e1130d9f02c` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.12 Inclusive Educ… |
| 11 | `2019-06-26:regular#a17` | adopted | `113aa711-6516-4816-93ff-32f52d4816ce` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.16 Technology Lev… |
| 12 | `2019-06-26:regular#a18` | adopted | `598d83b5-89b1-406f-bbbb-eb30b8264b98` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.24 Recommendation… |
| 13 | `2019-06-26:regular#a2` | adopted | `965c754e-b967-4deb-b5c9-c65be92c2fb9` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.14 Inclusive Educ… |
| 14 | `2019-06-26:regular#a20` | adopted | `a24d6cf4-19f0-4379-b588-1b489d4ad1a0` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.06 First Reading … |
| 15 | `2019-06-26:regular#a21` | adopted | `db9d4693-e324-4ad0-9c16-4d535cc928a8` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.10 Volunteers in … |
| 16 | `2019-06-26:regular#a22` | adopted | `0b570ef9-3575-4efe-b345-db586901e5c0` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.15 Actively Learn… |
| 17 | `2019-06-26:regular#a23` | adopted | `61964610-58f8-4abf-b870-5f1628685dc7` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.18 Fire Sprinkler… |
| 18 | `2019-06-26:regular#a24` | adopted | `ed12fec9-65ca-4d67-a231-2c7148b44426` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.19 New Valley Ele… |
| 19 | `2019-06-26:regular#a25` | adopted | `dcd1676f-8e04-4aee-83ea-7241cc9518c2` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.21 Pine Tree Elem… |
| 20 | `2019-06-26:regular#a32` | adopted | `53db1b90-aa33-4fc3-bcde-7b1f8a25f3f5` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.23 Recommendation… |
| 21 | `2019-06-26:regular#a36` | adopted | `147600c1-0f2d-4bf5-8ea2-9f65d2f49f4a` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.17 Fairwood Eleme… |
| 22 | `2019-06-26:regular#a5` | adopted | `ca2e64fd-e18a-4c2c-892a-7a4458610421` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.01 Decision Regar… |
| 23 | `2019-06-26:regular#a6` | adopted | `67e6bbf3-8804-4b5c-9447-672f62b0ef44` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.09 Thunderbird Co… |
| 24 | `2019-06-26:regular#a8` | adopted | `03937280-d47b-498e-8719-c4eb2239a6c6` | 1 | 0 | Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.03 Washington Sta… |
| 25 | `2020-08-26:regular#a22` | adopted | `ee77573c-21a5-408d-8ea4-3c4b87734ecf` | 1 | 0 | Agenda Item Details Meeting Aug 26, 2020 - VIRTUAL Regular Meeting - 7 p.m. Category 7. Board of Directors - Discussion & Approval Subject 7.13 N2Y, L… |
| 26 | `2020-08-26:regular#a29` | adopted | `7bf1ce58-27f9-48c4-a5d6-e5ee8aa9798a` | 1 | 151 | A motion was made to approve the addition of two supplemental personnel reports ("A" and "B"), the excusal of Director Bowen from tonight's Exempt Ses… |
| 27 | `2021-08-17:special#a4` | adopted | `e00a62cf-e453-4275-b1ab-ba3e3aef110d` | 1 | 836 | A motion was made to approve the Interim Superintendent Job Posting - Long Term with the following amendments: 1. Add the word "pandemic" to the posti… |
| 28 | `2022-10-12:regular#a3` | adopted | `20e59b42-58a6-47c6-b3bd-112034b5bb1d` | 1 | 890 | A motion was made to approve Suspend Board Policy 1420 Requiring Three (3) Days Advance Notice to the Board of Relevant Information on a Motion Item P… |
| 29 | `2023-03-08:regular#a4` | adopted | `b9d7da18-5d2e-4f92-a760-ee76a083fb8c` | 1 | 0 | Agenda Item Details Meeting Mar 08, 2023 - Regular Meeting - 6:30 p.m. Category 7. Discussion and Approval Subject 7.04 Resolution No. 1637 - Conditio… |
| 30 | `2023-04-26:regular#a98` | adopted | `e5fb6880-8060-42b7-8a24-b44a2c2f6a18` | 1 | 361 | A motion was made to move Reorganization of the Board of Directors and Election of Officers (*ADDED AT ITEM 1.07 - AGENDA REVIEW) from its initial pla… |
| 31 | `2023-09-27:regular#a7` | adopted | `d4515094-a162-460e-8d2d-82eb808cb5d8` | 1 | 1068 | A motion was made to approve Ninth Reading and Approval of Policy 4220 Complaints Concerning Staff or Programs, which was made by Director Hamada and … |
| 32 | `2023-12-13:regular#a18` | adopted | `ffae9e0a-8dc6-4f95-adac-3692f73a83ae` | 1 | 0 | Agenda Item Details Meeting Dec 13, 2023 - Regular Meeting - 6:30 p.m. Category 4. Reorganization of the Board of Directors and Election of Officers S… |
| 33 | `2023-12-13:regular#a19` | adopted | `ffae9e0a-8dc6-4f95-adac-3692f73a83ae` | 1 | 2497 | Yea: Awale Farah, Tim Clark, Meghin Margel, Donald Cook, Andy Song Elections for the 2024 school board officers took place via nominations and roll ca… |
| 34 | `2023-12-13:regular#a21` | adopted | `ffae9e0a-8dc6-4f95-adac-3692f73a83ae` | 1 | 3352 | Yea: Donald Cook, Andy Song Nay: Awale Farah, Tim Clark, Meghin Margel Elections for the 2024 school board officers took place via nominations and rol… |
| 35 | `2024-06-26:regular#a1` | adopted | `f52ac3c5-656a-4351-a651-a5da85de62d4` | 1 | 259 | A motion was made to remove Item 7.02 from the published agenda, City of Kent American Rescue Plan Act Federal Grant Fund Acceptance 2024-2026 due to … |
| 36 | `2024-12-11:regular#a42` | adopted | `43d5e9da-2032-476a-a6d2-be3c6e42fc4d` | 1 | 0 | Agenda Item Details Meeting Dec 11, 2024 - Regular Meeting - 6:30 p.m. Category 2. Reorganization of the Board of Directors and Election of Officers S… |
| 37 | `2024-12-11:regular#a43` | lost | `43d5e9da-2032-476a-a6d2-be3c6e42fc4d` | 1 | 2781 | Yea: Awale Farah, Tim Clark, Meghin Margel Nay: Donald Cook, Andy Song Elections for the 2025 school board officers took place via nominations and rol… |
| 38 | `2024-12-11:regular#a44` | adopted | `43d5e9da-2032-476a-a6d2-be3c6e42fc4d` | 1 | 3218 | Yea: Tim Clark, Meghin Margel Nay: Awale Farah, Donald Cook, Andy Song Elections for the 2025 school board officers took place via nominations and rol… |
| 39 | `2025-09-10:regular#a12` | adopted | `83fc7ead-25db-4bb5-b9bc-1f7555894864` | 1 | 561 | A motion was made to add a new item, ‘5th Grade Camp/Outdoor Education Discussion’, to the agenda for discussion. Voting took place via roll call vote… |
| 40 | `2025-12-10:regular#a18` | lost | `1479b410-dc79-40f4-b452-c5af3d306f1f` | 1 | 0 | Agenda Item Details Meeting Dec 10, 2025 - Regular Meeting - 6:30 p.m. Category 4. Reorganization of the Board of Directors and Election of Officers S… |
| 41 | `2025-12-10:regular#a20` | adopted | `1479b410-dc79-40f4-b452-c5af3d306f1f` | 1 | 3919 | . Elections for the 2026 school board officers took place via nominations and roll call vote was as follows for Vice President Nominee: Teresa Gregory… |
| 42 | `2025-12-10:regular#a21` | adopted | `1479b410-dc79-40f4-b452-c5af3d306f1f` | 1 | 4339 | . Elections for the 2026 school board officers took place via nominations and roll call vote was as follows for Legislative Representative Nominees: A… |
| 43 | `2026-01-28:regular#a45` | adopted | `ac6f74d1-4420-48bb-9861-b8811e0458c2` | 1 | 0 | Agenda Item Details Meeting Jan 28, 2026 - Regular Meeting - 6:30 p.m. Category 9. Discussion and Approval Subject 9.06 Second Reading and Approval of… |

The complete, untruncated quotes are available from the fixture itself:

```bash
.venv/bin/python fixtures.py \
  | python3 -c "import json,sys; [print(r) for r in json.load(sys.stdin)['hard'][2]['detail']['quote_mismatch']]"
```

## 5. Decision 4 — hand-count preparation — **BLOCKED, six targets ready**

The fixture now reads the operator's counts from
`facts/minutes/fixtures/hand_counts.yaml` and stays BLOCKED until every
`motions_total` is filled in. It reports which meetings it is still waiting on.
It did not run, and is not reported as passing.

**All six operator-chosen dates have a minutes document in the corpus, and all
six PDFs resolve on disk.** No substitution was needed.

| Era | Meeting | Document id | Title | Page count | Parser motions (total / adopted / lost) |
|---|---|---|---|---:|---|
| A | `2011-06-08:work_study` | `80444646-57cf-4912-9143-c62db4af377b` | `Board_Meeting_Minutes_060811.pdf` | 5 | 6 / 6 / 0 |
| A | `2016-04-27:work_study` | `29bd0888-38c6-45e0-9a72-5162a4792965` | `Board_Minutes_042716.pdf` | 5 | 6 / 6 / 0 |
| A | `2019-06-12:special` | `4652f94e-75c3-439c-9dd3-3694dc799683` | `Board Minutes 061219.pdf` | 6 | 11 / 11 / 0 |
| B | `2023-05-24:regular` | `1a194cee-bbe5-4f28-b4fe-b3cc2b36220e` | `Board Minutes 052423.pdf` | 3 | 11 / 11 / 0 |
| B | `2024-09-11:regular` | `87469e3a-8e4e-402b-b634-b93b4e3251ad` | `Board Minutes 2024 09 11.pdf` | 6 | 8 / 4 / 4 |
| B | `2025-02-26:regular` | `af4c1f49-c35a-4d0d-b8e7-0a9ca7ca036a` | `Board Minutes 2025 02 26.pdf` | 4 | 13 / 12 / 1 |

Source PDF paths as resolved by `locators.resolve_pdf_path` (all exist; the
stored `documents.file_path` values are stale and rewritten onto the archive
root — recommendation R1 from the previous session, still outstanding):

```
/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data/
  2011-06-22-regular-meeting-700-pm/Board_Meeting_Minutes_060811.pdf
  2016-05-11-regular-meeting-7-pm/Board_Minutes_042716.pdf
  2019-06-26-regular-meeting-7-p-m-/9-01-b4ha7a7a4dba-minutes-of-the-12-june-2019-special-meeting-e/Board Minutes 061219.pdf
  2023-06-14-regular-meeting-6-30-p-m-/8-15-cs6vrt81e30e-minutes-of-24-may-2023-regular-meeting-specia/Board Minutes 052423.pdf
  2024-09-25-regular-meeting-6-30-p-m-/7-04-d96n2t5d7068-minutes-of-11-september-2024-regular-meeting-/Board Minutes 2024 09 11.pdf
  2025-03-12-regular-meeting-6-30-p-m-/9-12-dea5ql111fea-minutes-of-26-february-2025-regular-meeting-a/Board Minutes 2025 02 26.pdf
```

Note that each minutes PDF lives in the directory of the *later* meeting that
approved it — the attachment-offset finding from Phase 0, visible here in the
paths.

**The parser counts above are minutes-sourced motions only.** Motions sourced
from BoardDocs agenda items are deliberately excluded from the comparison: the
operator is counting one PDF, and the agenda-item record is a separate source
describing the same meeting differently. Mixing them would compare a count of
one document against a count of two. For 2023-05-24 the difference is stark —
11 motions in the minutes, 58 more from agenda items.

### 5a. Why the three Era A targets are typed `work_study` / `special`

The three Era A dates carry their minutes on a census row typed `work_study` or
`special` rather than `regular`. The decision said to propose the nearest
regular meeting in that case. **I did not substitute, and I want to be explicit
about why**, because it is a judgement call and the operator may overrule it.

The nearest regular meeting that actually has minutes is a long way away:

| Requested date | Nearest `regular` with minutes | Gap |
|---|---|---:|
| 2011-06-08 | 2010-11-10 | 210 days |
| 2016-04-27 | 2016-01-13 | 105 days |
| 2019-06-12 | 2020-03-25 | 287 days |

Substituting would move each target by months and, worse, it would be
correcting the wrong thing. The documents at the requested dates are plainly
the board meeting minutes for those dates — `Board_Meeting_Minutes_060811.pdf`
carries 6 motions, which a work session does not have. What is unreliable is
the census `meeting_type` in Era A, not the document. A hand count needs a
minutes PDF with motions in it, and all six have exactly that.

`hand_counts.yaml` carries this note inline so whoever does the counting is not
confused by the type, and instructs them to count the PDF rather than the
label.

### 5b. Why the file is parsed without PyYAML

`load_hand_counts` is a ~30-line parser for the flat `key: value` subset the
template uses. PyYAML is in neither `requirements.txt` nor
`requirements-lock.txt`, and the dependency manifest is the authority for this
package. Adding a dependency to read one flat list of scalars — for a file this
same module writes the template of — is not a trade worth making. Malformed
lines raise `ValueError` rather than silently yielding fewer meetings than the
file names, which is the failure mode that would actually hurt: a hand-count
check that quietly verifies four meetings instead of six.

## 6. Phase 2 — verification

**Unit tests: 70 passed.** 35 pre-existing, unchanged; 35 added across three
classes:

- `TestDispositionStems` — 17 cases pinning every inflection the widened stems
  must accept, and 5 pinning what they must reject (including `approve`, and
  cross-disposition bleed such as "Motion carried." never evidencing `lost`).
- `TestKnownAttendanceVoteDiscrepancies` — 6 cases over the known-set logic:
  known rows suppressed, new rows surfaced with a NULL cause, mixed batches
  separated in one pass, the set closed at exactly six, every entry carrying a
  cause and a date consistent with its `meeting_id`, and 2025-02-11 pinned as
  `parser_roll_bleed` so a later edit cannot quietly reclassify a parser bug as
  a record quirk.
- `TestHandCountFile` — 7 cases: three targets per era, every target having a
  document/PDF/page count, counts shipping empty so the fixture cannot go
  green, scalar typing, absent file, malformed line raising, and comment
  stripping.

The known-set logic was deliberately split out of the fixture into
`unknown_attendance_vote_discrepancies()` so it is testable without a database.

**Fixture results:**

```
exec_sessions_2024             PASS   asserted 24/24; reported 2 announcements
attendance_vote_discrepancies  PASS   117 motions, 6 meetings, 0 unknown
disposition_and_locator        FAIL   43 of 6,507 — all enumerated in §4c
operator_hand_counts           BLOCKED  awaiting counts for all six targets
```

`fixtures.py` still exits non-zero, which is correct — `disposition_and_locator`
is genuinely red.

### 6a. Export re-run — one pre-existing difference, not caused by this session

`export_meeting.py 2026-02-04` re-run and diffed against
`reports/meeting-export-2026-02-04.md`. Two differences:

1. **Cosmetic.** The committed file has trailing double-spaces (markdown hard
   line breaks) stripped, presumably by the trailing-whitespace pre-commit
   hook. The generator still emits them. No data meaning.
2. **Substantive.** The committed export contains a second meeting block,
   `2026-02-04:work_study#2` (source `census`, no minutes, slug
   `2026-02-04-special-meeting-work-session-500-pm`). **That row does not exist
   in the database.** Only `2026-02-04:work_study` does.

This is the duplicate-slug dedupe from recommendation R6 — the same 2026 work
session scraped twice under differently punctuated slugs. The committed export
was generated *before* the census dedupe took effect and is stale relative to
the final loaded state.

It is not caused by this session's changes: `export_meeting.py`, `build.py`,
`parsers.py` and the fact tables were not touched. `git diff --stat` covers
only `fixtures.py`, `test_parsers.py` and `views.sql`, plus the new
`fixtures/hand_counts.yaml`. The only database write was `CREATE OR REPLACE
VIEW`.

I did not regenerate the committed export — the decision list did not include
it and the instruction was to report any change. See R9.

## 7. Recommended changes (requires operator approval — none made)

The task expected nothing here. Four things surfaced.

- **R7 — Fix the surname-only roll bleed in `vote_parser.parse_agenda_item`
  (the 2025-02-11 defect, §3b).** This is the one with real consequences: it
  inflates vote tallies by attributing an adjacent motion's roll to the wrong
  motion. Two changes are needed together — bound the tail scan so it cannot
  run to end-of-document when no following anchor exists, and normalise names
  before the duplicate check so `"Song"` and `"Andy Song"` collide. The fix
  touches `facts.vote` and `facts.motion` corpus-wide and requires
  `build.py --reload`, which would move the Phase 0 row counts this session was
  instructed to hold fixed. That is why it is a recommendation and not a
  change. **I would prioritise this above the remaining fixture work** — vote
  tallies are the most load-bearing numbers in the fact layer. Note the
  discrepancy fixture currently passes *with this bug live*.

- **R8 — Fix the truncated agenda-item motion quotes (the 43, §4a).** The
  citation should reach the disposition. Two options: give the disposition
  sentence its own short quote field rather than bridging from the motion
  opening, or keep the bridge but elide the middle (first ~200 chars + "…" +
  the `Final Resolution:` sentence). The second keeps one quote per motion and
  would take all 43 green *on evidence rather than on vocabulary*. Also
  requires a reload. Until then the fixture stays honestly red.

- **R9 — Regenerate `reports/meeting-export-2026-02-04.md`.** It is stale: it
  shows a meeting row that no longer exists (§6a). Anyone treating it as
  current would count one 2026-02-04 work session too many.

- **R10 — Correct the run report's six-meeting table** (`reports/facts-minutes-run-2026-09-13.md`
  §4), which lists 2020-03-19 (does not violate) and omits 2023-11-08 (51 of
  the 117), and describes the 43 as a withdrew/withdrawn vocabulary issue when
  no such row exists (§3a, §4a). The totals were right; the enumerations were
  not. Flagging rather than editing a previous session's report.

Previous recommendations R1–R6 are unchanged and still outstanding; R1 (stale
`documents.file_path`) is exercised by every hand-count target path in §5.

## 8. Guard gap observed (reported, not used)

`~/.claude/hooks/validate-pip-install.sh` blocked a legitimate lockfile
install. The command was:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install \
  --disable-pip-version-check -q -r requirements-lock.txt 2>&1 | tail -20
```

The hook extracts everything after `install` up to the first `;`, `|` or `&`,
which captures the shell redirect `2>&1` as if it were an argument. It is not a
flag and not a path, so it reaches the version-stripping step, where
`PKGNAME="${arg%%[><=!~@]*}"` cuts at the `>` and yields a package named `2`.
`2` is not on the allowlist, so the install is blocked.

Two observations:

- **False positive.** The hook explicitly intends to allow `-r lockfile`
  installs — with no package names parsed it exits 0 at line 106. A shell
  redirect in the same command defeats that. I re-ran the identical install
  without the redirect inside the pip command, which is the path the hook
  permits; I did not modify, disable or bypass the hook.
- **Secondary.** The block message names the allowlist path
  `/home/donald/workspace/.claude/approved-packages.txt`, which does not exist,
  so *every* pip install in this project is blocked regardless of contents.
  That is arguably correct-by-default, but it means the allowlist has never
  been exercised here.

Suggested hook change (operator's call, I have not touched it): strip shell
redirect tokens before parsing, e.g. skip any argument matching
`^[0-9]*[<>]` — a one-line addition alongside the existing URL and path skips.

## 9. Compliance notes

- All reads against `documents` and any non-`facts` table went through
  `db.query`, which opens the Postgres session `READ ONLY`. An accidental write
  would have failed at the database.
- The only write to the database was `CREATE OR REPLACE VIEW` against schema
  `facts` (six views, the five existing ones replaced unchanged plus the new
  one). **No table was written, no rebuild was run, nothing was deleted
  anywhere.** All six row counts are unchanged from Phase 0.
- `documents`, `chunks`, Qdrant and `rag_api` were never written.
  `ksd-boarddocs-rag`, production and the hooks were never touched.
- No LLM is in any date, name, vote, motion or count path. Everything added
  this session is regex, set membership and arithmetic.
- Credentials were injected at runtime from the container environment. `.env`
  was neither read nor edited, and no password was printed.
- Only packages in `requirements-lock.txt` were installed.

---

## Appendix — every command run this session

Credentials come from the running container at runtime. `.env` is stale for
Postgres and was not read or edited. The password is never echoed.

### Setup

```bash
# Worktree, fresh off main
cd ~/workspace/projects/ksd-main
git worktree add ~/workspace/projects/ksd-minutes -b claude/facts-minutes-fixtures main
git worktree list

# Confirm facts/minutes matches main
git log --oneline -5 main
git ls-tree -r --name-only main -- facts/
git diff --stat main HEAD -- facts/      # empty

# Virtualenv from the lock file
cd ~/workspace/projects/ksd-minutes/facts/minutes
python3 -m venv .venv
.venv/bin/python -V
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-lock.txt
```

### Credential injection (prefixes every database command below)

```bash
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)
```

### Phase 0 — recon

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes

podman ps --format '{{.Names}} {{.Status}}'

.venv/bin/python -m pytest test_parsers.py -q          # 35 passed

# Row counts against the run report
.venv/bin/python -c "
import db
for t in ['meeting','attendance','motion','vote','executive_session','minutes_parse_log']:
    print(t, db.query('SELECT count(*) FROM facts.'+t)[0][0])
"

# Baseline fixtures
.venv/bin/python fixtures.py > /tmp/fx_baseline.json
```

### Diagnosis

```bash
# Characterise all 43 disposition mismatches
.venv/bin/python -c "
import re, db
from fixtures import DISPOSITION_WORDS
rows = db.query('''SELECT motion_id, disposition, source, locator_document_id::text,
       locator_page, locator_char_offset, coalesce(locator_quote,'')
       FROM facts.motion ORDER BY motion_id''')
bad=[r for r in rows if DISPOSITION_WORDS.get(r[1]) and not re.search(DISPOSITION_WORDS[r[1]],r[6],re.I)]
from collections import Counter
print(len(bad), Counter(b[1] for b in bad), Counter(b[2] for b in bad))
"

# The actual six discrepant meetings (the run report's table is wrong)
.venv/bin/python -c "
import db
for r in db.query('''
WITH present AS (
  SELECT meeting_id, count(*) AS n FROM facts.attendance
  WHERE status IN ('present','present_virtual','arrived_late','left_early')
  GROUP BY meeting_id)
SELECT m.meeting_date, m.meeting_type, mo.meeting_id, count(*),
       min(p.n), max(coalesce(mo.tally_yes,0)+coalesce(mo.tally_no,0)+coalesce(mo.tally_abstain,0))
FROM facts.motion mo
JOIN present p ON p.meeting_id = mo.meeting_id
JOIN facts.meeting m ON m.meeting_id = mo.meeting_id
WHERE (mo.tally_yes IS NOT NULL OR mo.tally_no IS NOT NULL OR mo.tally_abstain IS NOT NULL)
  AND coalesce(mo.tally_yes,0)+coalesce(mo.tally_no,0)+coalesce(mo.tally_abstain,0) > p.n
GROUP BY 1,2,3 ORDER BY 1'''): print(r)
"

# Attendance vs voters per meeting (cause assignment)
.venv/bin/python -c "
import db
for mid in ['2023-11-08:regular','2025-02-11:special','2024-07-10:special',
            '2022-10-05:special','2023-12-13:regular','2022-06-29:special']:
    print(mid)
    print(' ', db.query('SELECT director_raw, status FROM facts.attendance WHERE meeting_id=%s ORDER BY 1',(mid,)))
    print(' ', db.query('''SELECT v.director_raw, count(*) FROM facts.vote v
        JOIN facts.motion mo USING (motion_id) WHERE mo.meeting_id=%s GROUP BY 1 ORDER BY 1''',(mid,)))
"

# 2025-02-11 roll bleed evidence
.venv/bin/python -c "
import db
for r in db.query('''SELECT v.director_raw, v.vote, v.locator_char_offset, v.locator_quote
   FROM facts.vote v JOIN facts.motion mo USING (motion_id)
   WHERE mo.meeting_id='2025-02-11:special' ORDER BY v.locator_char_offset'''): print(r)
"

# Hand-count target resolution
.venv/bin/python -c "
import db, locators
ids = [('2011-06-08:work_study','80444646-57cf-4912-9143-c62db4af377b'),
 ('2016-04-27:work_study','29bd0888-38c6-45e0-9a72-5162a4792965'),
 ('2019-06-12:special','4652f94e-75c3-439c-9dd3-3694dc799683'),
 ('2023-05-24:regular','1a194cee-bbe5-4f28-b4fe-b3cc2b36220e'),
 ('2024-09-11:regular','87469e3a-8e4e-402b-b634-b93b4e3251ad'),
 ('2025-02-26:regular','af4c1f49-c35a-4d0d-b8e7-0a9ca7ca036a')]
for mid, did in ids:
    t,fp,pc,dt = db.query('SELECT title, file_path, page_count, document_type FROM documents WHERE id=%s',(did,))[0]
    print(mid, t, pc, locators.resolve_pdf_path(fp))
    print('  ', db.query('SELECT source, disposition, count(*) FROM facts.motion WHERE meeting_id=%s GROUP BY 1,2 ORDER BY 1,2',(mid,)))
"

# Nearest regular meetings with minutes (for the Era A substitution question)
.venv/bin/python -c "
import db
for d in ['2011-06-08','2016-04-27','2019-06-12']:
    print(d, db.query('''SELECT meeting_id, abs(meeting_date - %s::date) FROM facts.meeting
      WHERE meeting_type='regular' AND minutes_document_id IS NOT NULL
      ORDER BY 2 LIMIT 3''',(d,)))
"
```

### Phase 1 — apply

```bash
# Views (five replaced unchanged, facts.attendance_vote_discrepancies added)
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql
```

Edits: `fixtures.py`, `test_parsers.py`, `views.sql`; new
`fixtures/hand_counts.yaml`.

### Phase 2 — verify

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes

.venv/bin/python -m pytest test_parsers.py -q          # 70 passed

.venv/bin/python fixtures.py > /tmp/fx_final.json      # exits 1 (disposition red)

# Export regression
.venv/bin/python export_meeting.py 2026-02-04 -o /tmp/meeting-export-2026-02-04.new.md
diff -u ../../reports/meeting-export-2026-02-04.md /tmp/meeting-export-2026-02-04.new.md

# Confirm blast radius
cd ~/workspace/projects/ksd-minutes && git status --short && git diff --stat
```

### Reproducing this session's results from scratch

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql
.venv/bin/python -m pytest test_parsers.py -q
.venv/bin/python fixtures.py
```

No `build.py --reload` was run and none is needed to reproduce: the fact tables
are untouched.

### Useful queries added by this session

```sql
SELECT meeting_id, cause, count(*) AS motions,
       min(present_recorded), max(cast_votes)
  FROM facts.attendance_vote_discrepancies
 GROUP BY 1,2 ORDER BY 1;

SELECT * FROM facts.attendance_vote_discrepancies WHERE cause IS NULL;  -- must be empty
```

---

# Part 2 — R7 and R8 applied (same day, same branch)

Operator approved R7 (vote-roll bleed) and R8 (locator quote cap), followed by
a full reload, export regeneration and a dated addendum to the 2026-09-13 run
report. This part records that work. Everything above describes the state
*before* the reload.

**All four hard fixtures are now green or legitimately blocked, and
`fixtures.py` exits 0 for the first time.**

| Fixture | Part 1 | Part 2 |
|---|---|---|
| `exec_sessions_2024` | PASS | **PASS** |
| `attendance_vote_discrepancies` | PASS (6 known) | **PASS (5 known)** |
| `disposition_and_locator` | FAIL (43) | **PASS (0)** |
| `operator_hand_counts` | BLOCKED | **BLOCKED** (awaiting counts) |

Unit tests: **80 passing** (70 → 80).

## 1. Row counts before and after the reload

`build.py --reload`, 1m12s, 874 minutes documents and 2,543 agenda items.

| Table | Before | After | Delta |
|---|---:|---:|---:|
| `facts.meeting` | 1,646 | 1,646 | 0 |
| `facts.attendance` | 3,515 | 3,515 | 0 |
| `facts.motion` | 6,507 | 6,507 | 0 |
| `facts.vote` | 19,625 | **19,613** | **−12** |
| `facts.executive_session` | 280 | 280 | 0 |
| `facts.minutes_parse_log` | 874 | 874 | 0 |

Secondary figures:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `motion` by source: agenda_item / minutes | 4,214 / 2,293 | 4,214 / 2,293 | 0 / 0 |
| `motion` by disposition: adopted / lost / withdrawn | 6,391 / 115 / 1 | 6,391 / 115 / 1 | 0 |
| Motions with `vote_format='named'` | 4,213 | **4,210** | **−3** |
| Longest `motion.locator_quote` | 400 | **3,516** | +3,116 |
| Motion quotes over 400 chars | 0 | **45** | +45 |
| Parse status: parsed / superseded | 795 / 79 | 795 / 79 | 0 |

**There are exactly two deltas, and both are explained below in full. Nothing
else moved.**

### Delta 1 — `facts.vote` −12 rows

Every one of the 12 was spurious. I verified this by running the old and new
parsers side by side over all 2,543 agenda items and diffing their output, not
by reasoning about it. Only **two documents** differ, across 4 motions:

| Document | Meeting | Motion | Votes removed |
|---|---|---|---|
| `1479b410` | 2025-12-10 regular | `#a18` | Donald Cook (no), Andy Song (no) |
| `1479b410` | 2025-12-10 regular | `#a19` | **`None` (no)** |
| `1479b410` | 2025-12-10 regular | `#a20` | Andy Song (yes), Laura Williams (yes), Teresa Gregory (yes), Donald Cook (no), Meghin Margel (no) |
| `69513d40` | 2025-02-11 special | `#a1` | Song (yes), Clark (no), Cook (no), Margel (no) |

**No votes were added anywhere, and no motion changed its disposition, mover,
second or count of motions.**

`69513d40` is the 2025-02-11 defect from Part 1 §3b: four nomination-round
votes swept onto an unrelated scheduling motion. Its legitimate roll survives —
`2025-02-11:special#a1` still carries its 4 Yea votes (Tim Clark, Meghin
Margel, Donald Cook, Andy Song) and is still `vote_format='named'`.

`1479b410` is the 2025-12-10 board reorganization, and it was worse than the
2025-02-11 case. Its rolls are printed **before** each `Final Resolution:` line,
so the old parser handed every motion the *following* motion's roll — a
systematic off-by-one. It also recorded a director named **`None`**, parsed
from the line "Nay: None." That row asserted that a person called None voted
against seating the Vice President.

### Delta 2 — `vote_format='named'` −3 motions

`2025-12-10:regular#a18`, `#a19` and `#a20` no longer have any votes attached,
so they fall back to `carried_no_names`. This is a direct consequence of
Delta 1 and affects no other year: 2025 named motions go 723 → 720, and the
percentage unnamed for 2025 goes 17.3% → 17.6%. Every other year in
`facts.votes_unnamed_by_year` is byte-identical, including 2018 at 130.

The pre-2018 headline finding is untouched: still zero named votes anywhere in
the record before 2018.

## 2. R7 — the vote-roll bleed

### 2a. The decided bound was necessary but not sufficient

The decision was to bound the surname-only scan to the current motion's block,
"never past the next `Motion & Voting` or `Recommended Action` heading,
whichever comes first." I implemented that bound. **On its own it does not fix
2025-02-11**, and I want to be explicit about that rather than report R7 as
done-as-specified.

In document `69513d40` the only `Motion & Voting` heading is at offset 303,
*before* the `Final Resolution:` anchor at 782. `Recommended Action` does not
appear at all. There is no later heading to stop at, so the bound falls through
to end-of-document exactly as before, and all eight `ROLL CALL VOTING ROUND`
blocks are still swept in.

What actually ends the motion's roll is structural: a motion's canonical roll
is the **contiguous run** of `Yea:`/`Nay:`/`Abstain:`/`Absent:` lines
immediately following its resolution. In `69513d40` that run is one line long,
and the prose sentence after it ("The process to select the new Director
District 4 board position…") is where the motion's roll ends.

So the implementation does both:

1. **Bound the block** by the earliest of: the next `Final Resolution:` anchor,
   the next "A motion was made", the next `Motion & Voting` heading, the next
   `Recommended Action` heading. (The decided rule, plus the two pre-existing
   bounds.)
2. **Trim to the canonical roll** within that block —
   `vote_parser._canonical_roll_block`. Blank lines are permitted inside a
   roll; the first non-blank, non-roll line ends it. Before the first roll
   line, up to 200 characters of interstitial text are tolerated (some items
   print "Voting took place via roll call vote." between the resolution and its
   roll), bounded so that a motion with no roll cannot reach forward into an
   unrelated one.

Step 2 is what does the work. Step 1 is retained because it is cheap, it is
correct, and it stops the scan earlier in the common multi-motion case.

### 2b. The regression test

`TestNominationRollCallDoesNotBleed`, built on a faithful excerpt of
`69513d40`. It uses only `parse_agenda_item`, which exists in both the old and
new parser, so it fails on an **assertion**, not an import error. Verified by
stashing the new `vote_parser.py` and running against `HEAD`:

```
FAILED test_only_the_canonical_roll_is_attributed_to_the_motion
FAILED test_tally_cannot_exceed_a_four_member_board
FAILED test_surname_only_nomination_names_are_not_recorded
FAILED test_a_later_heading_bounds_the_block
4 failed, 1 passed
```

with the defect's exact signature:

```
E       AssertionError: assert ((5) + (3)) == 4
E       AssertionError: assert 'Song' not in ['Tim Clark', 'Meghin Margel',
                                'Donald Cook', 'Andy Song', 'Song', 'Clark', ...]
```

All five pass on the new parser. `test_multi_motion_item_is_unaffected` pins
the ordinary two-motion shape so the fix cannot quietly narrow normal parsing.

The cap constant is imported *inside* its own test rather than at module scope,
specifically so it cannot turn the above assertion failures into an
`ImportError` when the file is run against the old parser.

### 2c. New gap this fix exposes — not a regression, but a real hole

`1479b410` (2025-12-10 board reorganization) now yields **no named votes at
all** for its four officer elections. Previously it yielded eight, all
misattributed by one motion. Wrong data became absent data, which is the right
direction, but neither is correct.

The cause is a format this parser has never handled: the rolls precede the
`Final Resolution:` line, and they use **`Aye:`** — a label `ROLL_RX` does not
recognise at all (it matches only `Yea|Nay|Abstain|Absent`). That is why the
old parser captured the `Nay:` lines but never the `Aye:` lines, and why one
row came through as a director named `None`.

This is a genuine civic record — how each director voted on seating the 2026
board president, vice president and legislative representative — and it is
currently not in `facts.vote`. Filing as **R11** below rather than fixing it
here: it is a new capability, not the bleed fix, and it would move the vote
counts again.

## 3. R8 — the locator quote cap

Measured the motion-opening-to-`Final Resolution:` span across all 4,214
agenda-item motions before choosing a cap:

| Statistic | Chars |
|---|---:|
| Minimum | 122 |
| Median | 152 |
| 95th percentile | 233 |
| 99th percentile | 420 |
| **Maximum** | **3,518** |

| Candidate cap | Motions still truncated |
|---:|---:|
| 400 (old) | 45 |
| 1,000 | 30 |
| 2,000 | 14 |
| 3,000 | 3 |
| **4,000** | **0** |

**New cap: `MOTION_QUOTE_MAX_LEN = 4000`**, clearing the longest span in the
corpus with headroom and matching the existing `motion_text[:4000]` ceiling.
The median motion is 152 characters, so the overwhelming majority of quotes are
unchanged — only 45 of 6,507 quotes now exceed 400 characters.

**Longest quote in the corpus after the reload: 3,516 characters.** Two
characters shorter than the 3,518-character span, because `make_quote`
collapses whitespace runs after slicing.

`disposition_and_locator` is now **PASS: 0 quote mismatches, 0 missing
locators, 6,507 of 6,507 checked.** Nothing is left to enumerate — the 43 cases
listed in §4c above are all resolved, and they were resolved by making the
citation reach the disposition rather than by widening what counts as evidence.
The `approve` stem is still excluded and still pinned by a test.

## 4. Known-set shrunk to five

With 2025-02-11 no longer discrepant, `facts.attendance_vote_discrepancies`
holds **116 rows across 5 meetings** (was 117 across 6):

| Meeting | Present | Max cast | Motions | Cause |
|---|---:|---:|---:|---|
| 2022-06-29 special | 1 | 4 | 4 | `presiding_only` |
| 2022-10-05 special | 4 | 5 | 1 | `attendance_short` |
| 2023-11-08 regular | 4 | 5 | 51 | `status_excluded` |
| 2023-12-13 regular | 4 | 5 | 36 | `board_transition` |
| 2024-07-10 special | 3 | 4 | 24 | `status_excluded` |

`2025-02-11:special` and the `parser_roll_bleed` cause are removed from both
`views.sql` and `KNOWN_ATTENDANCE_VOTE_DISCREPANCIES`. The remaining four cause
values all describe the district's record; none describes our parsing.

**The removal was not spotted by hand.** The `known_but_no_longer_present`
field I added in Part 1 reported `['2025-02-11:special']` on the first post-
reload run, which is exactly what it exists for — a known-set that silently
retains entries becomes a list of bugs the check has been taught to ignore.

Two tests enforce the shrink: `test_fixed_parser_defect_is_not_still_exempted`
asserts 2025-02-11 is gone, and `test_the_fixed_meeting_would_now_be_reported_
as_new` asserts that if the bleed ever regressed, that meeting would fail the
check rather than be waved through. `test_every_cause_describes_the_record_not_
the_parser` now rejects any cause outside the four record-level values, so a
future parser defect cannot be parked in the known-set the way this one was.

## 5. Export regenerated (R9)

`reports/meeting-export-2026-02-04.md` regenerated and committed. The stale row
is gone:

```diff
-## Work Study — `2026-02-04:work_study#2`
-- Record source: **census**
-- Minutes document: **none in corpus**
-- BoardDocs meeting: `2026-02-04-special-meeting-work-session-500-pm`
```

The only other change is the restoration of trailing double-spaces (markdown
hard line breaks) that the generator emits and the pre-commit
trailing-whitespace hook strips again on commit. No motion, vote, attendance or
locator value changed.

## 6. Run report addendum (R10)

`reports/facts-minutes-run-2026-09-13.md` now carries a dated
**Addendum — 2026-09-14** appended to the end. The original body is untouched.
It records: A1 the corrected six-meeting table, A2 that parser bug 3 was not
fully fixed and what it cost, A3 that the 43-mismatch diagnosis was wrong, A4
the figures superseded by the reload, A5 the stale export.

## 7. Verification

```
pytest test_parsers.py -q          80 passed
fixtures.py                        exit 0
  exec_sessions_2024               PASS  (24 asserted, 2 reported)
  attendance_vote_discrepancies    PASS  (116 rows, 5 meetings, 0 unknown, 0 stale)
  disposition_and_locator          PASS  (0 of 6,507)
  operator_hand_counts             BLOCKED (awaiting six counts)
```

One test needed updating beyond the known-set change:
`test_known_and_unknown_are_separated` used 2025-02-11 as its example of a
known meeting and was re-pointed at 2024-07-10.

## 8. Recommended changes (requires operator approval — none made)

- **R11 — Capture vote rolls printed *before* the `Final Resolution:` line, and
  recognise `Aye:` as a yes label.** The 2025-12-10 board reorganization (§2c)
  records how each director voted on seating the 2026 president, vice president
  and legislative representative, and none of it is in `facts.vote`. `Aye:` is
  not in `ROLL_RX` at all. Corpus-wide prevalence is unmeasured — I found this
  document only because the bleed fix changed its output. **I'd suggest
  measuring `Aye:` across the agenda-item corpus before designing the fix**,
  since it may be a one-off format or may affect every reorganization meeting
  in the record (these happen annually, every December).
- **R12 — Sweep for other motions whose roll is the last thing in its document.**
  The bleed was found via one meeting, not a search. The new
  `_canonical_roll_block` bounds all of them structurally, but a count of how
  many motions sit in that position would confirm the blast radius was really
  two documents.

R1–R6 from 2026-09-13 remain outstanding and untouched. R7, R8, R9 and R10 are
now done.

## 9. Compliance notes for Part 2

- Writes were confined to schema `facts`: `build.py --reload` rewrote the six
  fact tables, and `views.sql` was reapplied. No table outside `facts` was
  written; no row outside `facts` was deleted.
- All reads of `documents` used `db.query`, which opens the session
  `READ ONLY`. The old-vs-new parser diff in §1 read `documents.content_text`
  through that same read-only path.
- `documents`, `chunks`, Qdrant, `rag_api`, `ksd-boarddocs-rag` and production
  were never written. No hook was modified or bypassed.
- No LLM is in any date, name, vote, motion or count path.
- Credentials injected at runtime from the container; `.env` untouched; no
  password printed.

---

## Appendix — Part 2 commands

All prefixed by the same runtime credential injection shown in the Part 1
appendix.

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes

# Measure the motion-to-resolution span before choosing a cap (R8)
.venv/bin/python -c "
import re, db, statistics
from vote_parser import FINAL_RES_RX, MADE_RX, MOTION_VOTING_RX
rows = db.query('''SELECT id::text, content_text FROM documents
                   WHERE document_type='agenda_item' AND content_text ~* 'motion *& *voting' ''')
spans=[]
for did, txt in rows:
    if not MOTION_VOTING_RX.search(txt): continue
    anchors=list(FINAL_RES_RX.finditer(txt)); made=[m.start() for m in MADE_RX.finditer(txt)]
    for i,a in enumerate(anchors):
        prev_end = anchors[i-1].end() if i else 0
        cand=[p for p in made if prev_end<=p<a.start()]
        spans.append(a.end()-(cand[-1] if cand else prev_end))
spans.sort()
print(spans[0], statistics.median(spans), spans[-1])
for cap in (400,1000,2000,3000,4000): print(cap, sum(1 for s in spans if s>cap))
"

# Prove the regression test fails on the OLD parser with an assertion
cp vote_parser.py /tmp/vote_parser.new.py
git stash push -- vote_parser.py
.venv/bin/python -m pytest test_parsers.py -k TestNominationRollCallDoesNotBleed -q
git checkout -- vote_parser.py; git stash pop

# Snapshot before the reload
.venv/bin/python -c "
import db, json
snap={t: db.query('SELECT count(*) FROM facts.'+t)[0][0] for t in
      ['meeting','attendance','motion','vote','executive_session','minutes_parse_log']}
snap['max_quote_len']=db.query('SELECT max(length(locator_quote)) FROM facts.motion')[0][0]
snap['vote_named_motions']=db.query(\"SELECT count(*) FROM facts.motion WHERE vote_format='named'\")[0][0]
json.dump(snap, open('/tmp/before_reload.json','w'), indent=1)
"

# Rebuild (~73s)
.venv/bin/python build.py --reload

# Explain the vote delta: old vs new parser, side by side over every agenda item
git show HEAD:facts/minutes/vote_parser.py > /tmp/vote_parser_old.py
.venv/bin/python -c "
import sys, importlib.util, db
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
old=load('vp_old','/tmp/vote_parser_old.py'); import vote_parser as new
rows=db.query('''SELECT id::text, coalesce(agenda_item_id,''), content_text FROM documents
                 WHERE document_type='agenda_item' AND content_text ~* 'motion *& *voting' ''')
for did, aid, txt in rows:
    o=old.parse_agenda_item(txt); n=new.parse_agenda_item(txt)
    assert len(o)==len(n), ('motion count changed', did)
    for om, nm in zip(o,n):
        ov=[(v.director_raw,v.vote) for v in om.votes]; nv=[(v.director_raw,v.vote) for v in nm.votes]
        if ov!=nv: print(did, aid, om.seq, 'REMOVED', [x for x in ov if x not in nv],
                         'ADDED', [x for x in nv if x not in ov])
"

# Reapply views with the five-meeting known-set
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql

# Verify
.venv/bin/python -m pytest test_parsers.py -q      # 80 passed
.venv/bin/python fixtures.py                       # exit 0

# Regenerate the export and diff (R9)
.venv/bin/python export_meeting.py 2026-02-04 -o ../../reports/meeting-export-2026-02-04.md
cd ~/workspace/projects/ksd-minutes && git diff reports/meeting-export-2026-02-04.md
```

### Reproducing Part 2 from a clean checkout of this branch

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < schema.sql
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql

.venv/bin/python build.py --reload                 # ~73s, vote = 19,613
.venv/bin/python -m pytest test_parsers.py -q      # 80 passed
.venv/bin/python fixtures.py                       # exit 0
```

---

# Part 3 — R11 applied (same day, same branch)

Operator approved R11: recognise `Aye:`, handle a roll printed above its
resolution, and treat `None.` as zero votes. Measured first, as instructed.

**Result: the four 2025-12-10 officer elections are now recorded correctly —
20 named votes where there had been 8 wrong ones and then none. All four hard
fixtures stay green or blocked; `fixtures.py` still exits 0. Tests 80 → 93.**

## 1. Measurement, before changing anything

Over all 2,543 agenda items carrying a `Motion & Voting` block (4,214 motions):

| Year | Items | `Aye:` items | `Yea:` items | `Nay:` items | `<label>: None.` items |
|---|---:|---:|---:|---:|---:|
| 2018 | 113 | 0 | 113 | 20 | 0 |
| 2019 | 415 | 0 | 411 | 23 | 0 |
| 2020 | 381 | 0 | 378 | 72 | 0 |
| 2021 | 363 | 0 | 358 | 38 | 0 |
| 2022 | 321 | 0 | 318 | 31 | 0 |
| 2023 | 361 | 0 | 361 | 25 | 0 |
| 2024 | 214 | 0 | 214 | 59 | 0 |
| 2025 | 330 | **1** | 326 | 73 | **1** |
| 2026 | 45 | 0 | 43 | 14 | 0 |
| **All** | **2,543** | **1** | **2,522** | **355** | **1** |

Roll position relative to the `Final Resolution:` line, per motion:

| Roll position | Motions |
|---|---:|
| Roll on the first non-blank line after the resolution | 4,200 |
| Roll both above and below | 13 |
| Roll above only | 1 |
| No roll at all | 0 |

**`Aye:` appears in exactly one agenda item in the corpus, and so does
`<label>: None.` — both in `1479b410`, the 2025-12-10 board reorganization.**

This answers the open question I raised when filing R11. I had flagged that
reorganization meetings happen every December and this might therefore affect
the whole record. **It does not.** Every other year records officer elections
in a form this parser already reads, or does not record them in an agenda item
at all. R11 is a one-document fix, and I would not have known that without
measuring.

### 1a. A nuance in the "roll above only" count

Only 1 of `1479b410`'s 4 motions classifies as "above only" on the *old*
geometry, because each motion's apparent below-the-line roll was really the
next motion's above-the-line roll. After R7's contiguity rule those false
below-blocks are gone, so all four motions genuinely have no roll below — which
is exactly the condition the decided rule keys on. The rule works because R7
landed first; on the pre-R7 parser it would have fixed only the last election.

## 2. What changed in the parser

| Change | Why |
|---|---|
| `Aye` added to `ROLL_RX`, `ROLL_LINE_RX` and `VOTE_MAP` (`aye → yes`) | An affirmative label the parser did not recognise. Its absence is why the old parser read these rolls as Nay-only. |
| `NONE_ROLL_RX` + guards in `_split_names` | `Nay: None.` is a count of zero. Matched against the **whole** roll value (and against a bare comma-separated token), so a director whose name merely contained the word is unaffected — pinned by `test_a_real_name_is_not_dropped`. |
| `_preceding_roll_block(text, lower_bound, anchor_start)` | Fallback used **only** when a motion has no roll below its resolution. Scans backwards from the resolution for the contiguous roll immediately above it. |
| `prev_roll_end` carried across the motion loop | Bounds the look-back so a motion with genuinely no roll cannot reach back and claim the previous motion's. |
| Vote locator offsets now keyed to `roll_start` | The roll may sit above the resolution, so `tail_start` is no longer the right base for the citation. Pinned by `test_locator_quotes_point_at_the_roll_above_the_resolution`. |

Precedence is explicit: **a roll below the resolution always wins.** The
look-back is consulted only when there is nothing below. That is what keeps the
13 "above and below" motions — the original bug-3 shape, where the roll above
belongs to the *next* motion — parsing exactly as before.

### 2a. Correction to my own R7 implementation

Writing the R11 regression test exposed a real weakness in the code I committed
earlier today. `_canonical_roll_block` tolerated up to 200 characters of prose
between a resolution and its roll, on my speculation that some items print an
interstitial line. **On the excerpt, that tolerance was enough to bridge the
narrative and let motion 3 claim motion 4's roll — reintroducing the very
off-by-one R7 exists to stop.**

I measured whether the tolerance was earning anything: **4,210 of 4,214 motions
put the roll on the first non-blank line after the resolution, and not one
motion in the corpus needs a gap.** The only motions with a roll further down
are the 2025-12-10 elections, where that roll belongs to the next motion. The
tolerance was pure risk with zero benefit, so it is removed: any non-blank
content now ends the roll.

This is a case where a test written for one defect caught a latent one in the
fix for another. The 200-character tolerance was my invention, not something
the corpus asked for, and I should not have added it speculatively.

## 3. Regression tests

`TestPreResolutionAyeRolls` (7 tests) on a faithful excerpt of `1479b410`, and
`TestNoneRollHandling` (6 tests). Verified against the committed post-R7 parser
by stashing `vote_parser.py`:

```
FAILED TestPreResolutionAyeRolls::test_four_officer_elections_are_parsed
FAILED TestPreResolutionAyeRolls::test_each_election_gets_its_own_roll_not_the_next_ones
FAILED TestPreResolutionAyeRolls::test_cook_for_president_fails_two_to_three
FAILED TestPreResolutionAyeRolls::test_aye_is_an_affirmative_label
FAILED TestPreResolutionAyeRolls::test_nay_none_is_zero_votes_not_a_director
FAILED TestNoneRollHandling::test_none_yields_no_names[None.]
FAILED TestNoneRollHandling::test_none_yields_no_names[None]
FAILED TestNoneRollHandling::test_none_yields_no_names[ NONE ]
FAILED TestNoneRollHandling::test_none_yields_no_names[None .]
9 failed, 4 passed
```

All assertion failures, no import errors — `_split_names` is imported inside its
test for exactly that reason. Representative output:

```
E  AssertionError: assert ['None'] == []
E    Left contains one more item: 'None'
```

`test_a_post_resolution_roll_still_wins` pins the ordinary shape so the
fallback cannot start displacing normal parsing.

## 4. Reload and the old-vs-new diff

`build.py --reload`, 1m13s. Diffed the committed post-R7 parser against the R11
parser over all 2,543 agenda items, as for R7.

**Exactly one document differs. Nothing was removed anywhere. No motion changed
disposition, and no document changed its motion count.**

| Document | Meeting | Motion | Votes | Added |
|---|---|---|---:|---|
| `1479b410` | 2025-12-10 regular | `#a18` | 0 → 5 | Cook (yes), Song (yes), Williams (no), Gregory (no), Margel (no) |
| `1479b410` | 2025-12-10 regular | `#a19` | 0 → 5 | Williams (yes), Gregory (yes), Margel (yes), Cook (no), Song (no) |
| `1479b410` | 2025-12-10 regular | `#a20` | 0 → 5 | Cook, Song, Williams, Gregory, Margel (all yes) |
| `1479b410` | 2025-12-10 regular | `#a21` | 0 → 5 | Song (yes), Williams (yes), Gregory (yes), Cook (no), Margel (no) |

Row counts:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `facts.meeting` | 1,646 | 1,646 | 0 |
| `facts.attendance` | 3,515 | 3,515 | 0 |
| `facts.motion` | 6,507 | 6,507 | 0 |
| **`facts.vote`** | 19,613 | **19,633** | **+20** |
| `facts.executive_session` | 280 | 280 | 0 |
| `facts.minutes_parse_log` | 874 | 874 | 0 |
| vote: yes / no / abstain | 19,042 / 455 / 116 | 19,055 / 462 / 116 | **+13 / +7 / 0** |
| disposition: adopted / lost / withdrawn | 6,391 / 115 / 1 | 6,391 / 115 / 1 | **0 / 0 / 0** |
| motion source: agenda_item / minutes | 4,214 / 2,293 | 4,214 / 2,293 | 0 / 0 |
| `vote_format='named'` motions | 4,210 | **4,214** | **+4** |
| Longest motion locator quote | 3,516 | 3,516 | 0 |
| Parse status parsed / superseded | 795 / 79 | 795 / 79 | 0 / 0 |

Every delta accounted for:

- **+20 votes** = 4 elections × 5 directors, all in `1479b410`. The 13 yes / 7 no
  split is arithmetic on the four rolls: 2+3+5+3 = 13 affirmative,
  3+2+0+2 = 7 negative.
- **+4 named motions** — `#a18`–`#a21` now have votes, so they move from
  `carried_no_names` to `named`. All 4,214 agenda-item motions are now named.
- **Zero disposition changes**, verified directly rather than by inference: all
  6,507 dispositions were snapshotted before the reload and compared by
  `motion_id` afterwards. 0 changed, 0 motion ids disappeared, 0 appeared.
- **Directors named `None` in `facts.vote`: 0.**

Motions with `vote_format='named'` now equals the full agenda-item motion count
(4,214), which is the expected end state: every BoardDocs `Motion & Voting`
block carries a roll.

## 5. Fixtures, known-set and hand-count targets

```
pytest test_parsers.py -q          93 passed
fixtures.py                        exit 0
  exec_sessions_2024               PASS
  attendance_vote_discrepancies    PASS  (116 rows, 5 meetings, 0 unknown, 0 stale)
  disposition_and_locator          PASS  (0 of 6,507)
  operator_hand_counts             BLOCKED (awaiting six counts)
```

**Known-set unaffected** — still the same five meetings with the same causes and
the same 116 rows. 2025-12-10 does **not** enter the view.

**One caveat I want on the record about that.** 2025-12-10 does not appear in
`facts.attendance_vote_discrepancies` because it has **no attendance rows at
all** (no minutes document for that meeting in the corpus), and the view inner-
joins on recorded attendance. So its 5-voter rolls are not checked against a
roll call — not because they passed a check, but because there is nothing to
check them against. That is the documented behaviour of the fixture ("a motion
whose meeting has no attendance record cannot violate the constraint, it simply
cannot be checked"), and it is correct, but a reader should not mistake the
green result for confirmation of these 20 new rows.

**Hand-count targets unaffected** — all six re-verified against the database
after the reload; every `parser_motions_*` figure in `hand_counts.yaml` still
matches. None of the six is an agenda-item-sourced meeting, so R11 could not
have touched them.

**2026-02-04 export unchanged**, ignoring the trailing whitespace the
pre-commit hook strips.

## 6. Recommended changes (requires operator approval — none made)

- **R12 (unchanged, still open)** — sweep for motions whose roll is the last
  content in its document, to confirm the R7 blast radius really was two
  documents. The Part 3 measurement partly covers this (the roll-position table
  shows only 14 motions are not in the plain below-the-line shape), so this is
  now close to confirmed rather than open.
- **R13 — 2025-12-10 has no minutes document in the corpus.** That is why its
  votes cannot be cross-checked against attendance (§5). It is one of the
  missing-minutes meetings already visible in
  `facts.meetings_missing_minutes`; flagging it here only because it is now the
  source of 20 vote rows that no roll call constrains.

R1–R6 from 2026-09-13 remain outstanding. R7, R8, R9, R10 and R11 are done.

## 7. Compliance notes for Part 3

- Writes confined to schema `facts` (`build.py --reload`). No table outside
  `facts` written; nothing deleted outside `facts`; views unchanged this part.
- All `documents` reads via `db.query`, session `READ ONLY`, including the
  measurement pass and the old-vs-new parser diff.
- No LLM in any date, name, vote, motion or count path. The `Aye`/`None`
  handling and the backward roll scan are regex and line arithmetic.
- Credentials injected at runtime; `.env` untouched; no password printed.
- No hook modified or bypassed.

## Appendix — Part 3 commands

```bash
cd ~/workspace/projects/ksd-minutes/facts/minutes

# 1. Measure before changing anything
.venv/bin/python -c "
import re, db
from collections import Counter, defaultdict
from vote_parser import FINAL_RES_RX, MADE_RX, MOTION_VOTING_RX
AYE=re.compile(r'^[ \t]*Aye[ \t]*:', re.I|re.M); YEA=re.compile(r'^[ \t]*Yea[ \t]*:', re.I|re.M)
NAY=re.compile(r'^[ \t]*Nay[ \t]*:', re.I|re.M)
NONE=re.compile(r'^[ \t]*(?:Aye|Yea|Nay|Abstain|Absent)[ \t]*:[ \t]*None[ \t.]*\$', re.I|re.M)
ANY=re.compile(r'^[ \t]*(Aye|Yea|Nay|Abstain|Absent)[ \t]*:', re.I|re.M)
# ... full script in the session transcript; counts by year and roll position
"

# 2. Confirm the interstitial tolerance was unnecessary (leads to the R7 correction)
#    -> 4,210 roll_on_first_line, 3 roll_after_>200_chars, 0 in between

# 3. Prove the new tests fail on the committed post-R7 parser
cp vote_parser.py /tmp/vote_parser.r11.py
git stash push -- vote_parser.py
.venv/bin/python -m pytest test_parsers.py \
  -k "TestPreResolutionAyeRolls or TestNoneRollHandling" -q
git checkout -- vote_parser.py; git stash pop

# 4. Snapshot (including all 6,507 dispositions), reload, diff, compare
.venv/bin/python -c "
import db, json
json.dump({m:d for m,d in db.query('SELECT motion_id, disposition FROM facts.motion')},
          open('/tmp/disp_before_r11.json','w'))
"
.venv/bin/python build.py --reload                 # ~73s -> vote = 19,633

git show HEAD:facts/minutes/vote_parser.py > /tmp/vote_parser_r7.py
.venv/bin/python -c "
import sys, importlib.util, db
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
old=load('vp_r7','/tmp/vote_parser_r7.py'); import vote_parser as new
# per-document vote diff + disposition-change check; see transcript
"

# 5. Verify
.venv/bin/python -m pytest test_parsers.py -q      # 93 passed
.venv/bin/python fixtures.py                       # exit 0
.venv/bin/python export_meeting.py 2026-02-04 -o /tmp/exp_r11.md
diff <(sed 's/[[:space:]]*\$//' ../../reports/meeting-export-2026-02-04.md) \
     <(sed 's/[[:space:]]*\$//' /tmp/exp_r11.md)   # identical
```
