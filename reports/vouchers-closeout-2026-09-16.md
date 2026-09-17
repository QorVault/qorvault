# Close-out 2 — reload confirmed, allowlist scaffolded, one name still in a description

**Date:** 2026-09-16
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers`
**Parent commit at session start:** `7659f6d`
**Previous report:** `reports/vouchers-closeout-2026-09-15.md`
**Debrief:** `docs/session-logs/session-debrief-2026-09-16-vouchers-closeout-2.md`

> **Superseded in part.** The operator ruled on everything raised below on
> 2026-09-16 and the rulings are implemented. Sections 1–3 describe the state
> that produced those rulings and are left as written. **Read
> "Rulings implemented" at the foot of this report for what is true now** —
> in particular, the leak check passes, the artifacts are clean, and the
> voucher package no longer names an individual anywhere.

## Outcome first

**The reload is confirmed good. The code is committed. The regenerated
sample and export files are NOT committed, because one real person's name
is still printed in one of them — in a description, not in a payee column.**

| Step | Result |
|---|---|
| 1. Confirm the reload | **Clean.** All 88 baseline sets still reconcile, 20 more now do, 0 stopped. Fixtures 37 PASS / 0 FAIL, both HARD hand sums tie. |
| 2. Allowlist scaffold | **Done.** `fixtures/payee_allowlist.txt` created empty and wired into the one classifier; `reports/withheld-payees-2026-09-16.csv` written, 10,659 payees. |
| 3. Regenerate and commit | **Partly done.** Samples and exports regenerated; the leak check runs and **fails**; code committed, artifacts deliberately withheld from the commit. Rewrite commands written, unexecuted. |
| 4. Debrief | Done. |

Three things changed the shape of this session and each is set out in full
below:

1. **The leak check no longer skips.** Last session it could not reach the
   database and skipped, and a skip reads like a pass. It now runs, and its
   first honest verdict was that the artifacts then in the tree contained
   hundreds of individuals' names. After regeneration that is down to
   **one**.
2. **That one is in a description, not a payee column.** It is open finding
   C3, which the previous session raised and the operator has not yet ruled
   on. The classifier is doing its job; the description column is not
   covered by it.
3. **41 of the 42 names the leak check reports are not disclosures at all.**
   They are three separate defects in the test, itemised below with the
   exact fix for each. I did not apply those fixes — changing what a HARD
   privacy test accepts is the operator's call, not mine.

---

# Step 1 — the reload is confirmed

## What was compared, and the evidence

| Artifact | SHA-256 |
|---|---|
| `_build/baseline-sets-at-7659f6d.psv` | `091dc03ecb4d0f400b7bb229c6d067446501b73ff5a8fce67796aede135e348f` |
| `_build/sets_after_closeout.psv` | `e1c98661383f2fa3699c246634c71cbe6f6b5f8cd666887e15caf065b1637c86` |
| `_build/dryrun.log` | `2522776914a6fdda60a852acbd2e9165e400ac61f51c6bf88db352e55492e68d` |

All three describe the same **459 sets** with identical set-id membership —
no set appeared, none vanished. (Both `.psv` files are 461 lines: `psql`
echoes `BEGIN` and `SET` before the 459 data rows.)

**The 88-count holds.** 88 sets reconcile at `7659f6d`: 87 with no reason
code plus `2026-02-11:Permanent#2`, which reconciles and carries
`DUPLICATE_SET`. This is the count the operator's `rebuild.sh` asserted
before it was willing to touch the database, and it is the guard against
the failure mode the previous session documented — a comparison that
matches zero rows on both sides and prints nothing, which looks exactly
like a pass.

## The STOP check: nothing stopped

```
baseline reconciling: 88
after    reconciling: 108
sets that stopped reconciling: (none)
sets that started reconciling:  20
```

The dry run and the loaded database agree exactly on which 108 sets
reconcile — neither direction of the comparison prints a set. That matters
more than it looks: it means what landed in the tables is what the parse
said it would be, so the STOP check the operator ran against the dry run
(before any write) and the one run against the database afterwards are
testing the same thing.

## Reason codes, before and after

| reconciled | reason code | at `7659f6d` | after reload |
|---|---|---:|---:|
| (null) | `TOTAL_NOT_FOUND` | 335 | 335 |
| true | (none) | 87 | **107** |
| true | `DUPLICATE_SET` | 1 | 1 |
| false | `OUT_OF_BALANCE` | 26 | **5** |
| false | `COLUMN_AMBIGUOUS` | 6 | 6 |
| false | `REGEX_MISS` | 4 | 4 |
| false | `TOTAL_INCONSISTENT_AT_SOURCE` | — | **1** |

**21 sets changed, all in one direction.** 20 went
`OUT_OF_BALANCE → reconciled`, and 1 went
`OUT_OF_BALANCE → TOTAL_INCONSISTENT_AT_SOURCE`. No set moved from
reconciling to anything else, and no set acquired `COLUMN_AMBIGUOUS` or
`REGEX_MISS` that did not already have it.

## The confirmation set

Twenty sets both reconcile and contain rows whose sign came from a
bracket — and they are exactly the twenty sets that started reconciling.
The parenthesis rule is what moved them, and each one's own printed TOTAL
is what confirmed it:

```
2025-01-22:ASB  2025-01-22:GF   2025-02-26:ACH  2025-02-26:ASB
2025-02-26:GF   2025-04-23:ASB  2025-06-25:ASB  2025-06-25:GF
2025-07-23:ASB  2025-07-23:GF   2025-08-13:ASB  2025-09-24:ASB
2025-09-24:GF   2025-10-22:ASB  2025-10-22:GF   2025-12-10:GF
2026-01-14:ACH  2026-01-14:ASB  2026-01-14:GF   2026-02-11:GF
```

Every one ties to the cent: `stated_total = parsed_total` on all 20.

**The residual is one set.** `2026-02-11:ASB` holds 8 parenthesised rows,
is out of balance by −475.00, and its TOTAL ties under neither reading. It
keeps `OUT_OF_BALANCE`, which is correct: that bucket means go and check
the parse, and here the parse is what is in question.

The other 65 sets carrying parenthesised rows all print no TOTAL
(`TOTAL_NOT_FOUND`), so there is no arithmetic to confirm the reading
against. **This is worth stating plainly because it is a limit on the
standing ruling.** The ruling is "parentheses = negative, accepted per set
where the printed total then ties." In a set with no printed total there is
nothing to tie to, and `build.py` stores the negative reading anyway,
recording a note on the set that says the reading could not be confirmed.
That is a defensible default and it is documented at the point of use, but
it is not what the ruling says, and it covers the large majority of the
affected sets. Raised under Recommended changes.

## `TOTAL_INCONSISTENT_AT_SOURCE` — one set, as predicted

```
set_id             | fund    | stated  | parsed   | notes
2025-08-13:Capital | Capital | 1670.04 | 11202.17 | printed TOTAL is smaller than a single
                                                    line in the set (5090.00); every row was
                                                    read and none is negative
```

This is the instance the previous session predicted from a read-only query
before the code existed, and it is the only one in 459 sets. Nothing else
in the corpus trips the rule, which is the right shape for a rule that
accuses a document of contradicting itself.

## Parenthesised rows — more than the pre-change estimate, and why

|  | rows | sets |
|---|---:|---:|
| pre-change estimate (2026-09-15) | 744 | 59 |
| after reload, `amount_paren = true` | **1,388** | **86** |

**These count different things and neither is wrong.** The pre-change
figure counted rows that had *failed their column check* with a message
naming a parenthesised amount — that is, rows the old code could not read.
`amount_paren` records every row whose sign came from a bracket, including
rows the old code read without complaint. The larger number is the honest
one now that the column exists. It is called out because a reader comparing
the two reports would otherwise see a number nearly double and assume a
regression.

## Sentinel check numbers in 2026

One listing, six rows: `2026-05-27 ACH`, the `8888888888` Electrocom
family. Unchanged from the pre-rebuild measurement. Neither hand-sum
fixture touches a sentinel.

## Fixtures and tests

```
fixtures.py:  PASS 37   FAIL 0   BLOCKED 0   REPORT 24
  fixture_A_teamsters_2026      raw 30,344.00  dedup 30,344.00  10 lines  5 checks   PASS
  fixture_B_united_volleyball   raw  4,324.26  dedup  3,914.61  5→4 lines  diff 409.65  PASS
```

Both HARD hand sums tie against the rebuilt tables. R1 recorded 35 HARD
passes; the two additional ones are the hand sums, which did not exist then.

```
pytest:  351 passed, 1 failed, 0 skipped
```

Last session: 342 passed, 1 **skipped**. The skip was the leak check. It
now runs. The failure is the leak check and is the subject of Step 3.

**No STOP condition was tripped in Step 1.**

---

# Step 2 — the allowlist scaffold

## `facts/vouchers/fixtures/payee_allowlist.txt`

Created, **with no entries**, and populated by nobody. The file carries a
header explaining the format, which is documentation rather than data — an
operator-edited file that does not say how to edit it is a trap. No payee
in this corpus begins with `#`, measured across all 13,306 payees, count
zero, so a comment character cannot collide with a real name.

**An entry is matched on `normalize_vendor()`** — whitespace collapsed,
trailing punctuation stripped, case folded. That is not a loosening; it is
the key this package already uses everywhere to decide that two printed
spellings are one payee. Matching the literal string instead would withhold
the same payee on the cycles where the district typed it in a different
case, which is a control that fails in an unpredictable direction. It is
**not** a substring match: `KCDA` does not release `KCDA Warehouse`.

**An entry overrides every name pattern**, including the `Surname, Given`
person guard, because a real business is printed
`Hearing, Speech & Deafness Ctr`.

**An entry does not override the Payroll Handwrite signal.** That is the
one ordering decision I made and it is the one worth arguing about. Payroll
Handwrite is not a pattern over a name — it is the district's own record
that this payee was handed a cheque as a person. An allowlist is edited by
hand, and a hand-edited file will eventually contain a typo; the direction
in which that typo fails should be "an organization stays withheld", not "an
employee is published". It is pinned by a test and it is raised for the
operator under Recommended changes, because it is a rule about what the
operator's own instruction means.

## `reports/withheld-payees-2026-09-16.csv`

**10,659 withheld payees across 241,608 lines**, sorted by line count
descending, with the columns asked for plus `flag`.

| flag | payees | lines |
|---|---:|---:|
| `bare_acronym` | 116 | 18,082 |
| `all_caps` | 914 | 80,431 |
| (neither) | 9,629 | 143,095 |

| reason | payees |
|---|---:|
| `no_marker` | 6,899 |
| `person_shaped` | 3,131 |
| `payroll_handwrite` | 629 |

Published: 2,647 of 13,306, which is the same published count the previous
session measured over 13,291 payees — the rebuild added 15 payees and
released none of them.

**The top three withheld payees by volume are `BANK OF AMERICA` (59,042
lines), `Amazon.Com` (23,574) and `KCDA` (12,578).** That is the cost of
the specified rule in one line: the district's three highest-volume
counterparties are all withheld, none is a person, and a reader of the
export cannot see any of them.

Note that `Amazon.Com` carries **neither** flag — it is not a bare acronym
and it is not all caps. The flag column is a triage aid, not the answer;
the operator will need to read past it.

**This CSV is not committed and must not be.** It is a list of 10,659
names, ~3,760 of which the classifier believes are individuals. The command
to stop git from ever taking it is in the debrief. It is written to
`reports/` because that is where the instruction said to put it and because
the operator needs it beside the allowlist it feeds, not because it belongs
in the repository.

---

# Step 3 — regeneration, and the leak check

## What regenerated

- **32 sample files**, up from 28. The four new ones are
  `2026-01-14-ACH`, `2026-01-14-ASB`, `2026-01-14-GF` and `2026-02-11-GF` —
  the 2026 sets that started reconciling in the reload. Nothing became
  stale: reconciliation only grew, so every existing sample file was
  rewritten rather than orphaned.
- **10 export files**, the five 2026 cycles in `.md` and `.csv`.

**The regeneration command in the previous debrief was wrong.**
`export_cycle.py --date 2026-03-25` fails — the argument is positional,
`export_cycle.py 2026-03-25`. Five failed invocations, no output written,
exit code lost in the loop. Corrected in this report's appendix and in the
debrief.

## The leak check runs now, and here is what it found

Before regeneration, against the artifacts as they stood at `7659f6d`, the
check failed on hundreds of names including many individuals in
`Surname, Given` form. That is the leak the history rewrite exists to
remove, now demonstrated rather than inferred.

After regeneration it still fails, on **174 hits across 42 distinct names in
35 of 42 files**. Every hit was traced to its exact position in the file and
assigned a cause by span containment:

| cause | hits | names | is it a disclosure? |
|---|---:|---:|---|
| `suffix_variant` | 129 | 31 | **No.** Test defect. |
| `watch_list` | 25 | 2 | **No.** Test defect. |
| `free_text` | 20 | 11 | 10 organizations; **1 individual**. |

### `suffix_variant` — 129 hits, 31 names. A false positive.

The corpus prints the same business two ways, with and without its legal
suffix, and `normalize_vendor` deliberately does not strip suffixes
(`Smith Inc` and `Smith LLC` can be different legal entities). So
`Anixter Inc` carries a marker and publishes, while `ANIXTER` carries none
and is withheld — **the same company, two vendor rows.** The test then
searches the published file for the string `ANIXTER`, finds it inside
`Anixter Inc`, and calls it a leak.

All 31 are of this form. A sample:

```
withheld                     published superstring
ANIXTER                      Anixter Inc
CenturyLink                  CenturyLink Inc
DAILY JOURNAL OF COMMERCE    Daily Journal of Commerce Inc
STANDARD INSURANCE           Standard Insurance Company
PUGET SOUND REGIONAL         Puget Sound Regional Fire Authority
Child Support Enforcement    Child Support Enforcement Agency
```

None is a person. Nothing is disclosed that the classifier did not decide
to publish. **This cannot be fixed in the writers** — to make the test pass
you would have to withhold `Anixter Inc` because a different vendor row
spells it `ANIXTER`, and that reasoning cascades until nothing is
publishable. It is a defect in the test.

### `watch_list` — 25 hits, 2 names. Test and writer have drifted.

`YELLOW WOOD ACADEMY` and `BlazerWorks` appear in the export files' watch
list and top-vendor tables because `export_cycle.py` passes
`WATCH_LIST_NORMS` into `is_exportable`, and both are on the operator's
watch list. The export published them **on the operator's standing
instruction.**

`test_no_leaks.py` calls `classify_payee(name, handwrite)` directly, with no
watch list, so it cannot see that decision and reports it as a leak.

This one is worth dwelling on, because the whole design of this control is
"one classifier, so the test and the writers cannot drift." They drifted
anyway — not through a second copy of the rule, but through the test
calling the shared function with a different argument than the writer does.
A shared implementation does not give you a shared decision if the callers
pass different inputs.

### `free_text` — 20 hits, 11 names. Nine organizations, one person.

These are names printed in the **description** column and carried into the
verbatim quote. The payee column on those rows is correctly `(name
withheld)` or a correctly published organization; the name arrives from the
text of the line item:

```
| (name withheld) | ... | Due05 - TEAMSTERS DUES for 2025-11-26 Regular Payroll |
| (name withheld) | ... | Comcast Fiber WAN Lit KVA, iGrad, River Ridge ...     |
| Amazon Capital Services | ... | Nutritional Snacks ... using Safeway Grant funds |
| (name withheld) | ... | Gr. 3 FT to Museum of Flight Bus12/04/25             |
```

Ten of the eleven are organizations: `Teamsters`, `COMCAST`, `Safeway`,
`Fred Meyer`, `MUSEUM OF FLIGHT`, `Tacoma Art Museum`,
`WA ST THESPIAN SOCIETY`, `Elite Performance Dance Camp`,
`AASA Membership`, and `G GROUP`.

`G GROUP` is a third, separate test defect: `name_pattern` joins tokens
with `\s+` and anchors on nothing, so `G GROUP` matches across the word
boundary in "LAP Learnin**g Group** Supplies". The test already carries a
crude guard against this class of noise — it drops names of six characters
or fewer — which is the same problem recognised and half-solved.

**The eleventh is a real individual**, named in the description of line 908
of `facts/vouchers/samples/2026-07-22-GF.md`: a person receiving
recertification training, on a row whose payee (`QBS LLC`) is legitimately
published. I have not written the name into this report or the debrief;
the point of the exercise is to keep it out of git, and a report is in git.
Open the file at that line.

The name is withheld as `no_marker`, not `person_shaped` — which is finding
F4 in a single row. `Fred Meyer`, three names above it in the same list,
has the identical shape and is a supermarket chain. No pattern separates
them, which is exactly why the classifier is an allow-list.

## What was committed, and what was not

**Committed:** the code, the tests, the allowlist scaffold, the two
narrative documents, and the previous session's uncommitted work.

**Not committed:**

- `facts/vouchers/samples/` and `exports/` — the regenerated artifacts.
  They contain one individual's name and the operator has not ruled on C3.
  They are strictly better than what is at `HEAD` (which names hundreds of
  individuals in the payee column itself), but "better" is not the standard
  for a privacy control.
- `reports/withheld-payees-2026-09-16.csv` — 10,659 payee names.

**I did not modify `test_no_leaks.py`.** Three of its four failure causes
are defects in the test and I am confident of the diagnosis, but changing
what a HARD privacy test accepts is a decision with an evidence burden that
belongs to the operator. The exact patch for each is under Recommended
changes. The standing rule is that a HARD fixture is reported, not
loosened, and a test that is wrong in the *safe* direction is not urgent.

---

# Findings

**D1 — A skipped check reads exactly like a passing one, and this one hid a
real leak for a whole session.** `test_no_leaks.py` skipped on 2026-09-15
for want of a database. The previous session flagged the skip honestly and
in bold. It still took until this session, when the check actually ran, to
learn that the tracked artifacts contained hundreds of individuals' names —
which had been true since `95a6880` in every one of five commits. The
lesson is not "the skip was hidden"; it was declared. It is that a declared
skip on a HARD control is a stop condition, not a footnote.

**D2 — One shared classifier did not prevent drift, because the callers
disagree.** `test_no_leaks.py` and `export_cycle.py` both call
`is_exportable`/`classify_payee`, exactly as designed — and disagree on 25
hits, because the writer passes the watch list and the test does not. The
single-implementation rule is necessary and is not sufficient; the callers'
arguments are part of the rule.

**D3 — A substring search over payee names cannot be a privacy control on
this corpus.** 129 of 174 hits are a business's own name found inside its
own longer name. The corpus's deliberate refusal to strip legal suffixes —
correct, because `Smith Inc` and `Smith LLC` may differ — guarantees this
collision, so the test's search strategy and the normalizer's design are in
direct conflict.

**D4 — The description column leaks, and now there is a person in it.** C3
was raised on 2026-09-15 with `Payroll Handwrite - Kelly` as the example, a
surname beside an amount. The regenerated artifacts contain a full personal
name in a description on a row whose payee is a published company, so the
payee classifier cannot reach it by construction. C3 is no longer a
theoretical exposure.

**D5 — A source document prints an impossible voucher period, and nothing
checks.** `2026-01-14:GF` records `period_start 2025-11-14`,
`period_end 2025-01-08` — a period that ends ten months before it begins.
The source PDF says so verbatim: *"General Fund Warrants 11/14/25 through
01/08/25 and P-Cards 10/25/25 through 12/05/25"*. Every other fund on that
night prints `01/08/26` and the listing's own check dates run to
`01/08/2026`. **The district typed the wrong year; the parse is faithful.**

**D6 — And the same one-line invariant catches a real parse defect.**
`2025-08-13:GF` records `period_start 2025-07-01`, `period_end 2025-06-30`.
That listing prints **no period statement at all** — page 1 carries only
the date `7/31/2025`. The range was taken from the first data row's
description, *"Software License Renewal 07/01/25-06/30/25"*. The period
locator matched a date range inside a line item. Two sets in 459 have a
backwards period; one is the district's typo and one is this code's bug,
and `period_end >= period_start` would have found both.

**D7 — The rule for parenthesised amounts is confirmed on 20 sets and
unconfirmed on 65.** The standing ruling accepts the negative reading
"per set where the printed total then ties". 65 of the 86 affected sets
print no total. The code stores the negative reading and notes that it is
unconfirmed, which is reasonable and is not what the ruling says.

**D8 — The HARD payee fixtures are 31 individuals' names in a tracked test
file.** `test_payees.py` lists the F1 and F4 payees verbatim. The history
rewrite scrubs `facts/vouchers/samples` and `exports`; it does not touch
`test_payees.py`, and `test_no_leaks.py` does not scan it. The commit made
this session therefore puts 31 real individuals' names into git
permanently. They came from the operator's own findings, so this may be
intended — but it is the one place where the branch, after the rewrite,
still names people.

**D9 — The credential-free read path exists and the whole package works
through it.** The host reaches Postgres over a rootless Podman
port-forward, so Postgres sees a non-loopback source address and `pg_hba`
falls past its three `trust` lines to `scram-sha-256`. Inside the container
`local all all trust` applies. Every read path in this package now runs
with no credential at all. See "What changed" below.

---

# What changed in the code

| File | Change |
|---|---|
| `facts/vouchers/db.py` | **New read-only transport.** Opt-in via `VOUCHERS_DB_TRANSPORT=podman`; runs each query inside the database container through `podman exec ... psql`, wrapped in `BEGIN; SET TRANSACTION READ ONLY`. |
| `facts/vouchers/vendors.py` | `PAYEE_ALLOWLIST_PATH`, `load_payee_allowlist()`, `is_allowlisted()`; allowlist consulted inside `classify_payee` so the test and the writers see the same verdict. |
| `facts/vouchers/fixtures/payee_allowlist.txt` | **New, empty.** |
| `facts/vouchers/withheld_report.py` | **New.** Writes the withheld-payee worksheet. |
| `facts/vouchers/test_payees.py` | **New class**, 9 tests: the packaged allowlist ships empty, matching is the corpus key, an entry is not a substring, comments, missing file, person-shape override, and that Payroll Handwrite still withholds. |

## Why the transport is built the way it is

Three properties, because a second route into a database needs to be
boring:

- **Opt-in.** Without `VOUCHERS_DB_TRANSPORT=podman` nothing changes. The
  default path is byte-for-byte what it was, and still needs a credential.
- **Read-only enforced by Postgres, not by convention.** Every statement is
  wrapped in `BEGIN; SET TRANSACTION READ ONLY`. Verified live:
  `current_setting('transaction_read_only')` returns `on` inside the
  transport. A second, structural barrier comes free — the caller's SQL is
  nested inside a subquery, and PostgreSQL rejects a data-modifying
  statement there outright.
- **No string interpolation.** Parameters are rendered by **psycopg2's own
  adapters** — the same code that binds them on the normal path — and only
  where a connection-less adapter is provably exact. A text parameter
  carrying a backslash or a non-ASCII character **raises** rather than
  being rendered differently than psycopg2 would have. Placeholders are
  substituted by split-and-rejoin, not `%`-formatting, so a literal percent
  in a query cannot be mistaken for one.

One more thing was necessary and is easy to miss: `json_agg` has **no
ordering guarantee** over a subquery. Rows are numbered as they leave the
inner query and the aggregate is ordered on that number. Without it,
`ORDER BY` in a caller's query would be advisory — which would silently
reshuffle `export_cycle.py`'s top-vendor table and change which lines
`samples.py` draws from its seeded RNG.

---

# Open items

1. **The one individual's name in a description** —
   `facts/vouchers/samples/2026-07-22-GF.md`, line 908. Blocks the artifact
   commit. This is C3 and needs the operator's ruling.
   *Symptom:* leak check fails; a person is named beside an amount.
   *Tried:* full span-level triage of all 174 hits to separate it from 41
   non-disclosures. *Diagnosis:* the payee classifier cannot reach the
   description column by construction — the row's payee is a published
   company. *Next:* rule on C3 (options under Recommended changes).
   *Urgency:* **high** — it is the only thing between here and a clean
   artifact commit.
2. **The three `test_no_leaks.py` defects.** Patches written, not applied.
   *Urgency:* high — the test cannot go green without them even after C3.
3. **`test_payees.py` carries 31 individuals' names into git** (D8).
   *Urgency:* medium — decide before the branch is pushed anywhere.
4. **`2025-08-13:GF` period parsed from a line-item description** (D6).
   *Symptom:* `period_end` ten months before `period_start`. *Tried:*
   read page 1 of the source; it prints no period statement. *Diagnosis:*
   the period locator matches a date range anywhere on the page.
   *Next:* anchor the locator to the header band, add
   `period_end >= period_start` as a contract check, rebuild.
   *Urgency:* low — it affects one set's metadata, no money.
5. **`2026-01-14:GF` prints an impossible period in the source** (D5). No
   code change; it wants a reason code or a note so a reader is not
   misled. *Urgency:* low.
6. **65 sets carry an unconfirmed negative reading** (D7). *Urgency:* low
   — none of them reconciles or could.
7. Carried forward unchanged from 2026-09-15: the stale 40,658.50 figure in
   `reports/facts-vouchers-reconciliation-2026-09-15-addendum-staged-build.md:226`;
   `ruff`/`bandit`/`interrogate` now run via pre-commit on this commit;
   `overlap_rows.py` still measured over five cycles only, not 459 sets;
   whether the export's status vocabulary should show
   `TOTAL_INCONSISTENT_AT_SOURCE`.
8. The previous session's `/tmp/newly_withheld.txt` and friends did not
   survive. `reports/withheld-payees-2026-09-16.csv` supersedes them.

---

# Recommended changes (requires operator approval)

1. **Fix the three `test_no_leaks.py` defects.** In priority order, and all
   three are corrections rather than loosenings — each removes a report of
   a leak where no name is disclosed:
   - *Watch list.* Build the withheld list with
     `is_exportable(name, WATCH_LIST_NORMS, handwrite)` instead of
     `classify_payee(name, handwrite)`, so the test asks the same question
     the writer answers. **This is the important one** — without it the
     test contradicts a standing operator instruction.
   - *Suffix variants.* Ignore a hit whose matched span lies inside a
     longer published payee name at the same position. Measured effect:
     129 of 174 hits.
   - *Word boundaries.* Anchor `name_pattern` with `\b` at both ends. This
     also lets the `len > 6` filter be reconsidered, since it exists to
     paper over the same problem.
2. **Rule on C3 — the description column.** Three options:
   *(a)* redact withheld payee names from descriptions and quotes as well
   as the payee column. Makes the test passable, costs real civic context
   ("TEAMSTERS DUES" becomes unreadable), and is a large behaviour change.
   *(b)* redact only where the description names a payee the classifier
   believes is an **individual** — narrower, keeps `TEAMSTERS DUES` and
   `Comcast Fiber WAN` intact, and covers the actual disclosure. **I would
   do this one.** *(c)* accept it and exclude descriptions from the leak
   check explicitly, with the reasoning recorded. Doing nothing is not on
   the list: the artifacts cannot be committed as they stand.
3. **Decide whether the allowlist should override Payroll Handwrite.** I
   ordered it so it does not, and pinned that with a test. If you want the
   allowlist to be absolute, say so and the test changes with it.
4. **Decide the bare-acronym question**, carried from 2026-09-15 and now
   with better numbers: 116 payees / 18,082 lines are bare acronyms, and a
   further 914 payees / 80,431 lines are multi-token all-caps names. The
   allowlist now exists as the deliberate, per-name route, and
   `reports/withheld-payees-2026-09-16.csv` is sorted so the highest-volume
   candidates are at the top. My view is unchanged from last session: the
   structural all-caps-single-token rule cannot admit a `Given Surname`
   payee, so it is safe — but it is a widening of a privacy control.
5. **Add `period_end >= period_start` as a contract check** in
   `fixtures.py`. One line, and it would have caught D6, a real parse
   defect, on its own.
6. **Consider consolidating `WATCH_LIST` into the allowlist file.** There
   are now two operator-owned lists of names-to-publish: a tuple in
   `export_cycle.py` and a text file in `fixtures/`. They already disagree
   about who the test thinks is published (D2). One list, one file.
7. **Decide what to do about `test_payees.py`'s 31 names** (D8).

---

# Appendix — commands

Every database read this session went through the container, either
directly or through the new transport:

```bash
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs   # BEGIN; SET TRANSACTION READ ONLY;
```

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
export VOUCHERS_DB_TRANSPORT=podman        # no credential in this session

.venv/bin/python fixtures.py                              # 37 PASS / 0 FAIL / 24 REPORT
.venv/bin/python withheld_report.py                       # 10,659 withheld payees
.venv/bin/python samples.py                               # 32 files
for d in 2026-03-25 2026-05-27 2026-06-24 2026-07-22 2026-08-26; do
  .venv/bin/python export_cycle.py "$d"                   # POSITIONAL, not --date
done
.venv/bin/python -m pytest -q -rs                         # 351 passed, 1 failed, 0 skipped
```

STOP check, recomputed independently of the operator's `rebuild.sh`:

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers/_build
sha256sum baseline-sets-at-7659f6d.psv sets_after_closeout.psv
awk -F'|' '$3=="true"{print $1}' baseline-sets-at-7659f6d.psv | sort > /tmp/v_before.txt
awk -F'|' '$3=="true"{print $1}' sets_after_closeout.psv      | sort > /tmp/v_after.txt
test "$(wc -l < /tmp/v_before.txt)" -eq 88 || echo "BASELINE PARSE IS WRONG"
comm -23 /tmp/v_before.txt /tmp/v_after.txt      # printed nothing
```

The impossible periods (D5, D6):

```sql
SELECT set_id, period_start, period_end FROM facts.voucher_set
WHERE period_end < period_start ORDER BY set_id;
--  2025-08-13:GF | 2025-07-01 | 2025-06-30     <- parse defect
--  2026-01-14:GF | 2025-11-14 | 2025-01-08     <- source typo
```

---

# Rulings implemented — 2026-09-16

The operator ruled on all four open questions and the rulings are built.
**Everything above this line describes the state that produced them.**

| Ruling | State |
|---|---|
| 1. C3 masking + the three leak-test patches | **Done.** Leak check passes on the regenerated artifacts. |
| 2. Hash the 31 verbatim fixture names | **Done.** 38 fixtures, all resolving; map in gitignored scratch. |
| 3. Period invariant, two reason codes, parser fix | **Done in code.** Takes effect on the next reload. |
| 4. Transport stays opt-in; `pg_hba` documented | **Done**, and the transport was hardened along the way. |

**Tests: 386 passed, 0 failed, 0 skipped.** Fixtures: 37 PASS, **1 FAIL**,
24 REPORT — the one failure is the new period invariant reporting truthfully
that the database has not been reloaded yet. See ruling 3.

## Ruling 1 — descriptions are masked, and the leak test is fixed

**One object does the masking and the checking.** `vendors.WithheldNameIndex`
is built by `samples.py`, by `export_cycle.py` and by `test_no_leaks.py`, and
each builds it from **its own** view of who is published — samples without
the watch list, exports with it, exactly as each already decided its payee
column. That is the direct answer to finding D2: the previous drift did not
come from two copies of the rule, it came from two callers passing different
arguments to one function, so the fix is a shared object rather than a shared
function.

Masking uses the same token as the payee column, `(name withheld)`, and it
covers the sample description column, the sample verbatim quote, the whole
rendered export markdown and every field of the export CSV.

```
| (name withheld) | ... | Due05 - (name withheld) DUES for 2025-11-26 Regular Payroll |
| Amazon Capital Services | ... | Nutritional Snacks ... using (name withheld) Grant funds |
```

Ordering matters and is pinned in the code: the row's own payee goes through
`redact_name` **first**, because that function suppresses the whole quote when
the name cannot be located in it, and only then is the rest of the text
masked.

**The three patches, all applied:**

- *Watch list.* The test now builds its withheld list with
  `is_exportable(name, WATCH_LIST_NORMS, handwrite)`. Removes 25 false hits.
- *Suffix variants.* A hit whose span lies inside a longer published payee
  name is not a leak — `ANIXTER` inside `Anixter Inc` is one company spelled
  two ways. This lives in `WithheldNameIndex.occurrences`, so the writers get
  it too and never mask a published company's name out of its own text.
  Removes 129 false hits.
- *Word boundaries.* `name_pattern` is anchored with `(?<!\w)` / `(?!\w)`
  rather than `\b`, so a name that begins or ends with punctuation anchors the
  same way. `G GROUP` no longer matches inside "LAP Learnin**g Group**
  Supplies".

### The length cut-off was replaced, and the reason is not cosmetic

The old test dropped any withheld name of six characters or fewer, because
without anchoring a short name matches inside ordinary words. Carrying that
straight over would have been a mistake. Measured over the corpus:

| withheld names shorter than 7 characters | 131 |
|---|---:|
| single token — every one an organization or acronym (`AFLAC`, `KCDA`, `Costco`, and also `CASH`, `Club`, `Acct`) | 107 |
| two tokens | 24 |
| ...of those 24, names that read as personal | **23** |

**A flat length cut-off leaves exactly the people outside the control, and it
does it to the payees with the shortest names** — two short tokens each, the
shape common in Vietnamese, Chinese and Korean names. The rule is now
`is_maskable`: two or more tokens, **or** seven characters or more. Two
anchored tokens are specific enough to mask on at any length; a single short
token is not, which is what keeps `CASH` and `Club` from shredding
descriptions.

### Result

- `test_no_leaks.py`: **2 passed.**
- Every tracked file under `facts/vouchers/samples/` and `exports/` searched
  with the full 10,549-name index: **0 hits.**
- Exports reported `withheld names masked out of the rendered markdown: 0`
  on all five cycles. That is the expected answer, not a failure — the export
  composes its prose from structured values and carries no description text.
  The pass is in place so that a future field cannot leak by being forgotten.

### Organisations masked or withheld only because the allowlist is empty

Requested separately, and **not a reason to hold the artifacts**. These 45
names are withheld by the specified marker rule alone; none is an individual.
They appear in the voucher package's code comments, its fixtures, or its own
reports, and where they occur in a published artifact's free text they are
now masked. Populating `fixtures/payee_allowlist.txt` releases any of them:

```
AASA Membership              ALBERTSONS                   ALL HANDS CMTY INTERP SVCS
ANIXTER                      ARAMARK                      Amazon.Com
Area 5 DECA                  B & H PHOTO-VIDEO            BANK OF AMERICA
COMCAST                      CenturyLink                  Child Support Enforcement
Consolidated Press Printing In  DAILY JOURNAL OF COMMERCE  DEPARTMENT OF RETIREMENT
ELECTROCOM                   Elite Performance Dance Camp Elite Performance Dance Camps
Fred Meyer                   G GROUP                      Happy Feet Boots
KENT YOUTH & FAMILY          MICRO COMPUTER SYSTEMS       MUSEUM OF FLIGHT
Monday.com                   PUGET SOUND ENERGY           PUGET SOUND REGIONAL
RESTORX OF WASHINGTON        SOOS CREEK WATER & SEWER     STANDARD INSURANCE
Safeway                      THE HEATHMAN LODGE AND HUDSONS BAR AN
Tacoma Art Museum            Teamsters                    Threshold
Total Technology             US Bank                      US Foods - Seattle
UW Botanic Gardens           VAN SICLEN STOCKS            WA ST Patrol
WA ST THESPIAN SOCIETY       WEA/APA-BLUE CROSS           WEA/APA-WA DENTAL
Weissman
```

Three of those are worth a second look before they go on any allowlist:
**`Threshold`, `Total Technology` and `Weissman`** are payee names that are
also ordinary words or common nouns, and they are why most of the hits in
this list are in prose about parsing rather than in anything to do with
money. **The artifacts are still not committed**, per the ruling.

## Ruling 2 — the fixture names are hashes now

`test_payees.py` carries **no payee name at all**. Each fixture is the
SHA-256 of the exact payee string, hashed on `display_name`, plus the class
it belongs to; the test resolves a hash to a name from the live payee table
at run time. 38 fixtures, **all 38 resolving**. A hash that matches nothing
is a failure, not a skip.

Three tests guard the fixtures themselves: the group sizes are pinned (17,
2, 14, 2), every hash must resolve, and every constant must be 64 hex
characters — so a future edit that pastes a name back in fails in CI rather
than in a privacy review.

**What this buys and what it does not**, stated in the module docstring
rather than left implied: SHA-256 of a short string is reversible by anyone
holding a candidate list, and the payee table is exactly such a list. The
control is that the table lives in the database and not in git, so the two
halves are never in the same place. A hash here is not safe to publish
*alongside* the corpus.

Map written to `facts/vouchers/_build/payee_fixture_map.txt` (gitignored),
regenerable with `python test_payees.py`.

### Names found elsewhere, and removed

Hashing the fixtures was not sufficient. Searching every tracked file with
the full withheld list found individuals in five more places, all now fixed:

| File | What was there |
|---|---|
| `facts/vouchers/test_parsers.py` | 8 real payees used as parse fixtures. Replaced with invented names of the same shape, each verified not to collide with a real payee. The shapes are what those tests exercise; the identities never were. |
| `facts/vouchers/vendors.py` | 5 real payees named in comments as examples — **3 of them put there by me earlier in this same session**, while writing the comment that explains why short names must be masked. Replaced with descriptions of the shape. |
| `reports/vouchers-closeout-2026-09-15.md` | 4 individuals in the C2 finding. |
| `docs/session-logs/session-debrief-2026-09-15-vouchers-closeout.md` | 3 individuals in the same finding. |
| `reports/facts-vouchers-recon-2026-09-14.md`, `reports/vouchers-r1-2026-09-15.md` | 8 individuals in older reports. |

Editing historical reports deserves a word, because this package has a
standing rule against rewriting them: the previous session declined to
correct a stale *figure* in an old report, on the grounds that doing so
destroys the record of what was believed when. **A name is not a finding.**
Every figure, conclusion and date in those reports is untouched; only
personal names were replaced, with a description of what the name was. The
record of what was believed is intact.

### Confirmation — the count the ruling asked for

Searched: **323 tracked text files**, against all 10,656 withheld payees
(10,549 of them specific enough to search for).

| | files |
|---|---:|
| A. `facts/vouchers/samples/` and `exports/` containing any withheld name | **0** |
| B. voucher package, its reports and its debriefs, containing an **individual** | **0** |
| C. rest of the repository, containing an individual | 3 |

**B is the number the ruling asked for and it is 0.** Twenty-one files in
the voucher package still contain an organisation's name — the list above —
which is not a disclosure.

**C is outside this session's write scope and is reported, not fixed.** The
three files are `facts/minutes/parsers.py`, `facts/minutes/test_parsers.py`
and `reports/facts-minutes-recon-2026-09-12.md`, one individual each. They
belong to the `facts/minutes` package. The same treatment applied here would
work there.

There is also a category the count above does not capture, and it should be
said plainly: files under `transcription/` and `research/` name **board
directors and district officials** who are also voucher payees, because they
are reimbursed. Those are public officials named in public meeting records,
which is a different thing from a private individual receiving a refund, and
nothing here should be read as proposing to redact them.

## Ruling 3 — the period invariant

**Parser.** `PERIOD_RX` and `PCARD_RX` now search only `header_region()` —
page 1 down to, but not including, the first line carrying two or more
printed amounts. Two amounts is what makes a line a data row in every era in
this corpus, and no title, fund name or period statement carries them.

**The defect was wider than the one set.** Three sets lose a period that was
read out of a line item, not one:

| set | period it had | why |
|---|---|---|
| `2025-08-13:GF` | 2025-07-01 → 2025-06-30 | from `"Software License Renewal 07/01/25-06/30/25"` |
| `2017-03-22:ASB` | 2016-11-29 → 2017-01-17 | from a description ~2,600 characters into page 1 |
| `2017-04-26:ASB` | 2016-12-06 → 2017-01-24 | same shape |

The two 2017 sets never showed up in the backwards-period check because
their invented ranges happened to run forwards. **They were wrong and nobody
could have seen it.** That is the argument for the invariant in one line: it
found a bug it was not looking for.

**And it exposed a second, separate gap.** Both 2017 listings *do* state a
period in their header — `3-9-17 through 3-16-17` — with hyphens.
`PERIOD_RX` matches slashes only, so it never read them. Those sets will now
be recorded as `PERIOD_NOT_STATED` when the truth is "stated in a form this
layer cannot read". **I did not widen the date pattern** — that adds data
rather than removing wrong data, it is not what the ruling asked for, and it
needs its own verification pass. Instead the note attached to the set says
what is actually true: *"no warrant period could be read from this listing's
header … either the listing prints none, or it prints one in a format this
layer does not read."* Open item.

**Reason codes.** `PERIOD_INVALID_AT_SOURCE` where the stated period ends
before it begins, **with the parsed dates kept exactly as printed** — blanking
them would hide the district's error behind what reads as a missing field.
`PERIOD_NOT_STATED` where none could be read. Both are written **only when
`reason_code` is otherwise empty**: a reconciliation finding outranks a
metadata one, and the column holds one value. Both added to the
`voucher_set` CHECK constraint and to the `voucher_parse_log` status list.

> While adding them I found the `CREATE TABLE` status list had drifted from
> the `ALTER` block at the foot of `schema.sql` — `total_inconsistent_at_source`
> was added to one and not the other, so a **fresh install from that file
> would have rejected a status the running database accepts**. Fixed, and the
> two lists now carry a comment saying they must move together.

**Contract check.** `check_period_invariant` in `fixtures.py` fails on any
set with `period_end < period_start` that does **not** carry
`PERIOD_INVALID_AT_SOURCE` — so a period this code invented fails, while a
district typo faithfully recorded is reported. It is registered in the suite
runner.

**It currently FAILS, and that is correct.** The database still holds the
pre-ruling build, where both backwards-period sets carry no reason code:

```
FAIL  contract_period_invariant
      expected = 0 sets with period_end < period_start and no PERIOD_INVALID_AT_SOURCE
      actual   = 2: ['2025-08-13:GF', '2026-01-14:GF']
```

It passes after `build.py --reload`. Fixtures are 37 PASS / 1 FAIL until then.

### Confirmation: no dollar figure and no reconciliation status changes

`build.py --dry-run` over all 459 sets — which now runs credential-free
through the transport — compared against the loaded database:

```
sets in dry run: 459    sets in database: 459    membership identical
DOLLAR FIGURES changed (stated or parsed total, compared numerically):   0
LINE COUNTS changed:                                                     0
RECONCILIATION STATUS changed:                                           0
REASON CODE changed:                                                    12
```

All twelve move from no reason code to a period code: eleven to
`PERIOD_NOT_STATED`, one — `2026-01-14:GF` — to `PERIOD_INVALID_AT_SOURCE`.
The two 2017 ASB sets are **not** among them, because they already carry
`TOTAL_NOT_FOUND`; they lose their invented period silently, which is the
precedence rule working.

> A caution on how that comparison was made. A first pass reported ten
> "changed" dollar figures which were all `0` against `0.00` — the dry-run log
> prints a zero Decimal one way and the database stores it another. Comparing
> the rendered strings would have reported a ten-set money change that does
> not exist. The figures above are compared as `Decimal`.

## Ruling 4 — the transport, and what running it through `build.py` found

Kept opt-in, default psycopg2, read-only enforced by
`BEGIN; SET TRANSACTION READ ONLY`. The `pg_hba` explanation is now in
`docs/data-paths.md`, including the exact log line that identifies the cause
and the statement that this is the intended posture rather than a
misconfiguration to fix.

**The transport was also wrong, and the dry run found it.** The first
attempt to run `build.py --dry-run` through it aborted:

```
ValueError: VOUCHERS_DB_TRANSPORT=podman cannot render this parameter exactly
(non-ASCII or backslash): '(General Fund|ACH|...)\\s+(Warrants?|Payments?)'
```

`census.voucher_agenda_items` binds a regex, and psycopg2's adapter — used
with no connection — assumes `standard_conforming_strings` is **off** and
doubles every backslash. That would have turned `\s+` into something matching
different rows. **The guard caught it rather than letting it through**, which
is the one thing that had to work, and it is the clearest evidence that
refusing to guess was the right default.

Now fixed properly rather than by refusing:

- The server's `standard_conforming_strings` is **queried once per process**,
  not assumed, and the transport refuses to run if it is off.
- Under that setting the complete rule for a literal is that `'` is doubled
  and every other character stands for itself. The renderer is cross-checked
  against psycopg2's adapter for every string where the adapter is known to
  be exact, so the hand-written rule cannot drift without failing.
- UTF-8 is pinned on both sides (`PGCLIENTENCODING=UTF8`, bytes in and out)
  rather than inherited from the locale.

Round-tripped through the server: `a\s+b`, `O'Brien & Sons`, `café`,
`back\slash`, `%Robert Half%`, `quote'' and \ both` — **all six return
byte-identical.**

Every read path in the package now runs with no credential: `samples.py`,
`export_cycle.py`, `fixtures.py`, `withheld_report.py`, `build.py --dry-run`,
`withheld_report.py` and the whole test suite. `build.py --reload` writes and
still needs the password from `ksd-main/.env`.

## Open items added by this round

1. **`PERIOD_RX` cannot read a hyphenated period.** At least two era-B/C
   listings state one and will be recorded `PERIOD_NOT_STATED`.
   *Next:* widen the pattern and re-verify with the same dry-run diff.
   *Urgency:* low — metadata only, no money.
2. **Three files in `facts/minutes` name an individual.** Outside this
   session's write scope. *Urgency:* medium, same class as the fixture
   hashing just done here.
3. **`Threshold`, `Total Technology`, `Weissman`** are payee names that are
   also ordinary words. Harmless today because nothing masks them in the
   voucher artifacts, but worth knowing before anyone lowers `is_maskable`
   further. *Urgency:* low.
4. **The period reason codes are written only when `reason_code` is free.**
   Two sets (`2017-03-22:ASB`, `2017-04-26:ASB`) therefore lose an invented
   period with no code recording it; the note on the set says so, but a query
   on `reason_code` will not find them. *Urgency:* low.

---

# Additions of 2026-09-16 — the second reload, and two more verifications

Both written, **neither run**. The operator runs the rewrite, then
`rebuild2.sh`.

## `_build/rebuild2.sh` — the second reload

Same shape as the `rebuild.sh` that produced the first one: apply the
schema, dry run, gate, and only then write. Every step that can refuse
refuses **before** the reload rather than after it.

**Schema first, and why it cannot be second.** `facts.voucher_set.reason_code`
and `facts.voucher_parse_log.status` both carry CHECK constraints listing the
permitted values, and neither list in the running database contains
`PERIOD_INVALID_AT_SOURCE` or `PERIOD_NOT_STATED`. Reloading first and
migrating afterwards fails part-way through a write. `schema.sql` is
idempotent by construction — `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT
EXISTS`, `CREATE INDEX IF NOT EXISTS`, and constraints re-stated inside a
`DO` block that does `DROP CONSTRAINT IF EXISTS` before `ADD CONSTRAINT`, so
re-running replaces a constraint rather than colliding with it. `views.sql`
is unchanged since 2026-09-15 and is deliberately not re-applied.

**The baseline is new and was captured from the live tables this session**,
not predicted:

| | |
|---|---|
| file | `_build/baseline-sets-before-rebuild2.psv` |
| sha256 | `e1c98661383f2fa3699c246634c71cbe6f6b5f8cd666887e15caf065b1637c86` |
| sets | 459 |
| reconciling | **108** |

The count is asserted before the comparison runs, and the dry run's count is
asserted too. That guard exists because of the failure the first rebuild
documented: a comparison that matches zero rows on **both** sides prints
nothing and reads exactly like a pass.

**What it asserts after the write:**

```
459 sets, membership unchanged
108 reconciling, none of the original 108 stopped
 12 reason codes changed   (11 -> PERIOD_NOT_STATED, 1 -> PERIOD_INVALID_AT_SOURCE)
  0 dollar figures changed
  0 line counts changed
  0 reconciliation statuses changed
fixtures.py reports FAIL 0
```

Anything else and it exits non-zero with the rows printed.

**`_build/compare_sets.py` does the classification, and it compares money as
`Decimal`.** This is the one piece of the script that is not obvious and it
is not fussiness: the first attempt at exactly this comparison during the
session reported **ten changed dollar figures that were all `0` against
`0.00`** — the dry-run log renders a zero Decimal one way and the database
stores it another. Comparing rendered strings would have reported a ten-set
money change that does not exist, on a script whose entire job is to notice
a money change.

**Pre-flighted, both directions.** The expectation was checked against the
2026-09-16 dry run before being written into the script:

```
DOLLAR FIGURES changed (Decimal comparison): 0
LINE COUNTS changed: 0
RECONCILIATION STATUS changed: 0
REASON CODES changed: 12
OK: every difference is one that was expected.        exit=0
```

and the gate was tested failing, because a gate that cannot fail is not a
gate: nudging one parsed total by a cent gives `exit=1`, and asking it to
expect 11 reason-code changes instead of 12 gives `exit=1`.

`bash -n` parses the script clean. It has not been executed.

## Verification 4 — the content rewrite changed 49 substitutions and nothing else

Verification 2 proves no individual's name survives the rewrite. It does not
prove the filter left everything else alone — a substitution script that also
mangled an unrelated line would pass it.

The method is stronger than reading a diff: for every commit and every one of
the seven rewritten paths, take the **original** blob from the safety tag's
history, pipe it through the same filter, and require the result to be
**byte-identical** to the rewritten commit's blob. Anything the filter did
that the substitution set does not explain shows up as a `MISMATCH`.

Commits pair by subject line, which is unique across all twelve
(`git log --format=%s | sort | uniq -d` prints nothing). The pairing would
survive `--prune-empty` dropping a commit — it will not drop one, because no
commit on this branch touches only the artifact paths, and that was checked.

**Expected: `checked=67`, `rewritten=46`, and `MISMATCH` never printed.**
Both counts were computed against the current history rather than guessed.

## Verification 5 — the substitution file is gone

49 individuals' names, useful only until Verification 4 passes and a
liability afterwards. Deleted **after** Verifications 1–4, never before,
because every one of them needs it. Three proofs: the files are gone, nothing
left under `_build/` still carries the substitution table, and git has never
heard of any of them.

`_build/payee_fixture_map.txt` is deliberately **kept** — it is the
hash-to-name lookup that lets a failing `test_payees.py` fixture be traced
back to the payee it identifies, it is gitignored, and it regenerates with
`python test_payees.py`. Delete it too if you would rather regenerate on
demand; nothing depends on it existing.
