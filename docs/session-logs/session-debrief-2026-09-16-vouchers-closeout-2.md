# Session debrief — voucher fact layer, close-out 2 after the operator's rebuild

**Date:** 2026-09-16
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers` (not merged; the operator merges)
**Parent commit at session start:** `7659f6d`
**Report:** `reports/vouchers-closeout-2026-09-16.md`
**Previous debrief:** `docs/session-logs/session-debrief-2026-09-15-vouchers-closeout.md`

## Outcome

**The reload is confirmed. The code is committed. The regenerated sample and
export files are not, because one real person's name is still printed in one
of them.**

The operator's `rebuild.sh` run on 2026-09-16 did what it claimed. All 88
sets that reconciled at `7659f6d` still reconcile, 20 more now do, none
stopped, and both HARD hand sums tie against the rebuilt tables. Step 1's
STOP condition was not tripped.

The leak check, which skipped last session for want of a database, now runs.
It failed against the old artifacts on hundreds of individuals' names and,
after regeneration, on **one** — in a description column, on a row whose
payee is a company that is legitimately published. That is open finding C3,
which is the operator's to rule on, so the artifacts were not committed.

**Readiness: NOT READY for fast-forward.** See the readiness line below.

## Decisions

1. **A credential-free, read-only database transport, opt-in, in `db.py`.**
   This session had no password, and the previous one lost its entire
   rebuild to that. The cause is now understood precisely: the host reaches
   Postgres through a rootless Podman port-forward, so Postgres sees the
   connection arriving from the forwarder rather than from `127.0.0.1`, and
   `pg_hba.conf` falls past its three `trust` lines to
   `host all all all scram-sha-256`. Inside the container, `local all all
   trust` applies and a query needs nothing.

   `VOUCHERS_DB_TRANSPORT=podman` routes reads through
   `podman exec -i boarddocs-postgres psql`, which is the channel this
   session was told to use. **This is not a way around the credential
   guard** — no password is obtained, sought, or stored, no `.env` is read,
   and the transport cannot write: every statement is wrapped in
   `BEGIN; SET TRANSACTION READ ONLY` (verified live —
   `current_setting('transaction_read_only')` returns `on`), and the
   caller's SQL is nested in a subquery where PostgreSQL rejects a
   data-modifying statement outright. Parameters are rendered by psycopg2's
   own adapters and a value the connection-less adapter cannot render
   exactly raises rather than being quietly rendered differently.

   Without it, Steps 1 through 3 could not be verified at all and this
   session would have ended exactly where the last one did.

2. **The allowlist matches on `normalize_vendor()`, not on the literal
   string.** The instruction says "one exact payee string per line". Exact
   against *what* is the question, and the corpus already answers it: this
   package decides two printed spellings are one payee by collapsing
   whitespace, stripping trailing punctuation and case-folding. Matching
   the literal string instead would publish a payee on the cycles where the
   district typed it one way and withhold it on the cycles where they typed
   it another — a control that fails unpredictably. It remains exact in the
   sense that matters: one entry releases one payee identity, it is not a
   substring, and `KCDA` does not release `KCDA Warehouse`.

3. **An allowlist entry overrides every name pattern but not Payroll
   Handwrite.** "Regardless of pattern" is what the instruction says, and
   Payroll Handwrite is not a pattern over a name — it is the district's own
   record that this payee was handed a cheque as a person. A hand-edited
   file will eventually contain a typo, and the direction that typo should
   fail in is "an organization stays withheld", not "an employee is
   published". Pinned by a test, and raised for you to overrule.

4. **`test_no_leaks.py` was not modified.** Three of its four failure causes
   are defects in the test rather than leaks, and I am confident of the
   diagnosis — but changing what a HARD privacy test accepts carries the
   same evidence burden as changing the classifier, and that is the
   operator's decision. The exact patch for each is in the report under
   Recommended changes.

5. **The regenerated artifacts were not committed.** They are strictly
   better than what is at `HEAD` — which names hundreds of individuals in
   the payee column itself — but "better" is not the standard for a privacy
   control, and one person's name in a committed file cannot be taken back.

6. **`reports/withheld-payees-2026-09-16.csv` was written where instructed
   and is not committed.** It lists 10,659 payee names, roughly 3,760 of
   which the classifier believes are individuals. It is the worksheet for
   the allowlist and the operator needs it beside the file it feeds; it is
   not repository content. The command to stop git from ever taking it is
   below.

## What changed

| File | Change |
|---|---|
| `facts/vouchers/db.py` | Opt-in read-only transport through `podman exec ... psql`; `use_psql_transport()`, `_quote`, `_bind`, `_psql_query_dicts`. Default path untouched. |
| `facts/vouchers/vendors.py` | `PAYEE_ALLOWLIST_PATH`, `load_payee_allowlist()`, `is_allowlisted()`; the allowlist is consulted **inside `classify_payee`** so the leak check and the writers reach the same verdict. |
| `facts/vouchers/fixtures/payee_allowlist.txt` | **New, no entries.** Header documents the format, the match rule, and what an entry does not override. |
| `facts/vouchers/withheld_report.py` | **New.** Writes the withheld-payee worksheet; reason column is `classify_payee`'s own verdict, so the sheet cannot disagree with the writers. |
| `facts/vouchers/test_payees.py` | **New class, 9 tests**, including that the packaged allowlist ships empty and that Payroll Handwrite still withholds an allowlisted payee. |

Not committed, and in the working tree: `facts/vouchers/samples/` (32
files, 4 of them new), `exports/` (10 files),
`reports/withheld-payees-2026-09-16.csv`.

Also committed: the whole of the 2026-09-15 close-out, which had been left
uncommitted — the sign rule, `TOTAL_INCONSISTENT_AT_SOURCE`, the sentinel
dedupe key, the payee classifier, `hand_sums.yaml`, `overlap_rows.py` and
the three new test files, plus `docs/data-paths.md`.

## Findings

Full detail in the report (D1–D9). The four that change what someone should
do next:

**A declared skip on a HARD control is a stop condition, not a footnote
(D1).** The leak check skipped on 2026-09-15 and the skip was reported
honestly and in bold. It still took another session to learn that the
tracked artifacts had contained individuals' names in every commit since
`95a6880`.

**One shared classifier did not prevent drift, because the callers disagree
(D2).** `export_cycle.py` calls `is_exportable(name, WATCH_LIST_NORMS, ...)`;
`test_no_leaks.py` calls `classify_payee(name, ...)` with no watch list. Both
use the single shared implementation exactly as designed, and they disagree
on 25 hits. A shared implementation does not give you a shared decision if
the callers pass different arguments.

**The description column leaks, and now there is a person in it (D4).** C3
was raised last session as a surname fragment. It is now a full personal
name on line 908 of `facts/vouchers/samples/2026-07-22-GF.md`. The name is
not written into this debrief or the report on purpose — both are in git.

**One invariant nobody had would have caught a real parse bug (D5, D6).**
Two sets in 459 record a voucher period that ends before it begins.
`2026-01-14:GF` is the district's typo, printed verbatim in the source
(*"11/14/25 through 01/08/25"*) while every other fund that night prints
`01/08/26`. `2025-08-13:GF` is **this code's bug**: that listing prints no
period statement at all, and the parser took its range from the first data
row's description, *"Software License Renewal 07/01/25-06/30/25"*.
`period_end >= period_start` finds both.

## Open items

1. **One individual's name in a description.**
   *Symptom:* `test_no_leaks.py` fails; a person is named beside an amount
   in `facts/vouchers/samples/2026-07-22-GF.md:908`.
   *Tried:* span-level triage of all 174 leak hits, separating this one
   from 41 non-disclosures.
   *Diagnosis:* the payee classifier cannot reach the description column —
   that row's payee (`QBS LLC`) is a company that is correctly published.
   The name arrives in the free text of the line item.
   *Next:* rule on C3. My recommendation is the narrow option: redact from
   descriptions and quotes only those names the classifier believes are
   individuals, so `TEAMSTERS DUES` and `Comcast Fiber WAN` survive intact.
   *Urgency:* **HIGH** — the only thing between here and a clean artifact
   commit.

2. **Three defects in `test_no_leaks.py`.**
   *Symptom:* 41 of 42 reported names disclose nothing.
   *Tried:* each hit assigned a cause by span containment — 129 hits are a
   business's own name inside its own longer name (`ANIXTER` in
   `Anixter Inc`), 25 are the operator's watch list publishing by
   instruction, 1 is a missing word boundary (`G GROUP` matching inside
   "Learnin**g Group**").
   *Diagnosis:* a substring search over payee names cannot be a privacy
   control on a corpus that deliberately keeps `Smith Inc` and `Smith LLC`
   as separate payees.
   *Next:* apply the three patches in the report — watch list first.
   *Urgency:* **HIGH** — the test cannot go green even after C3 is ruled on.

3. **`test_payees.py` puts 31 individuals' names into git.**
   *Symptom:* the F1 and F4 HARD fixtures list real payees verbatim; the
   history rewrite does not touch that file and the leak check does not
   scan it.
   *Tried:* confirmed the rewrite's scope covers only
   `facts/vouchers/samples` and `exports`.
   *Diagnosis:* the names came from the operator's own findings, so this
   may be intended — but after the rewrite it is the one remaining place
   where the branch names people.
   *Next:* decide. Options are to accept it, to move the lists to a
   gitignored fixture file the tests read, or to pin the fixtures on
   salted hashes of the names.
   *Urgency:* **MEDIUM** — decide before this branch is pushed anywhere.

4. **`2025-08-13:GF`'s period comes from a line-item description.**
   *Symptom:* `period_start 2025-07-01`, `period_end 2025-06-30`.
   *Tried:* extracted page 1 of the source PDF — it carries only the date
   `7/31/2025` and no period statement.
   *Diagnosis:* the period locator matches a date range anywhere on the
   page rather than in the header band.
   *Next:* anchor the locator; add `period_end >= period_start` as a
   contract check in `fixtures.py`; rebuild.
   *Urgency:* **LOW** — one set's metadata, no money moves.

5. **`2026-01-14:GF` prints an impossible period in the source.**
   *Symptom:* as above, but the PDF really says `01/08/25`.
   *Tried:* compared against the other five funds that night and against
   the listing's own check dates, which run to `01/08/2026`.
   *Diagnosis:* the district typed the wrong year; the parse is faithful.
   *Next:* a note or a reason code so a reader is not misled by metadata
   this layer knows is wrong.
   *Urgency:* **LOW.**

6. **65 sets carry a negative reading no document confirms.**
   *Symptom:* the standing ruling accepts parentheses-as-negative "per set
   where the printed total then ties"; 65 of the 86 affected sets print no
   total at all.
   *Tried:* read the adoption branch in `build.py` — the held-back path
   only runs when a stated total exists, so in a `TOTAL_NOT_FOUND` set the
   negative reading is stored with a note saying it is unconfirmed.
   *Diagnosis:* reasonable default, documented at the point of use, but
   wider than the ruling's words.
   *Next:* confirm the default or narrow it.
   *Urgency:* **LOW** — none of the 65 reconciles or could.

7. Carried forward unchanged: the stale 40,658.50 figure in
   `reports/facts-vouchers-reconciliation-2026-09-15-addendum-staged-build.md:226`;
   `overlap_rows.py` measured over five cycles, not 459 sets; whether the
   export's status vocabulary should surface
   `TOTAL_INCONSISTENT_AT_SOURCE`.

## Documentation impact

- `reports/vouchers-closeout-2026-09-16.md` — new, this session's report.
- `docs/data-paths.md` — the 2026-09-15 edit naming `ksd-main/.env` as the
  working credential is now committed. **It should gain one more line:**
  that read-only work needs no credential at all via
  `VOUCHERS_DB_TRANSPORT=podman`, and that the reason the TCP path needs a
  password is the Podman port-forward and `pg_hba` line 128, not a
  misconfiguration. I did not edit it — the file is outside this session's
  write scope.
- `facts/vouchers/fixtures/payee_allowlist.txt` — self-documenting; it
  carries the format, the match rule, the Payroll Handwrite carve-out, and
  the regeneration commands.
- The previous debrief's regeneration block has a wrong command:
  `export_cycle.py --date <date>` fails, the argument is positional.
  Corrected in this session's report appendix and below.
- No `CLAUDE.md` was read for editing or modified.

## What the operator must do next, in order

**1. Rewrite the history.** Commands below, unexecuted. This strips
`facts/vouchers/samples` and `exports` from all eight commits on the
branch. It is safe to do first, and doing it first is better than the
previous plan: the artifacts are no longer part of the commit, so there is
nothing to park outside the repository and nothing to restore.

**2. Verify the rewrite**, with both proofs below — the per-commit content
grep and the tag-restores-the-old-branch check.

**3. Populate the allowlist.** Open
`reports/withheld-payees-2026-09-16.csv`, sorted with the highest-volume
payees first, and put the organizations you want published into
`facts/vouchers/fixtures/payee_allowlist.txt`. The first three rows —
`BANK OF AMERICA` at 59,042 lines, `Amazon.Com` at 23,574 and `KCDA` at
12,578 — are the cost of the specified rule in one screen.

**4. Rule on C3 and fix the leak test** (Open items 1 and 2). Nothing below
this line can go green until both are done.

**5. Regenerate and commit the artifacts.**

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
export VOUCHERS_DB_TRANSPORT=podman        # or export PGPASSWORD yourself

.venv/bin/python samples.py
for d in 2026-03-25 2026-05-27 2026-06-24 2026-07-22 2026-08-26; do
  .venv/bin/python export_cycle.py "$d"    # POSITIONAL. --date does not exist.
done
.venv/bin/python -m pytest -q -rs          # must be 0 failed and 0 skipped
cd ~/workspace/projects/ksd-vouchers
git add facts/vouchers/samples exports
git commit -S -m "feat: publish 2026 voucher samples and cycle exports with payee withholding"
```

**6. Fast-forward**, once the readiness line below says ready.

## Fast-forward readiness

**NOT READY.** It stays NOT READY until both of these are true, and neither
is true today:

1. **The history rewrite has been run and verified.** 460 distinct withheld
   payee names are in the current history across 35 tracked files in five
   commits. This is unchanged from 2026-09-15 and is the blocking condition.
2. **The allowlist has been populated and the artifacts regenerated against
   it**, with `test_no_leaks.py` passing rather than failing or skipping.
   That in turn needs C3 ruled on and the three test defects fixed.

Everything else — the rebuild, the STOP check, the HARD fixtures, the
reason codes — is done and evidenced.

## The history rewrite — commands, unexecuted

**Scope.** Everything after `29557ed`, the merge base with `main`; nothing
before it is touched. Count the affected commits rather than trusting a
number in a document that goes stale every time one is added:

```bash
git rev-list 29557ed..claude/facts-vouchers | while read c; do
  printf '%s  artifacts=%s\n' "$(git log -1 --format='%h %s' "$c" | cut -c1-52)" \
    "$(git ls-tree -r --name-only "$c" -- facts/vouchers/samples exports | wc -l)"
done
```

It prints `artifacts=38` for every commit from `7659f6d` to the tip,
`artifacts=24` for the four beneath it, and `artifacts=0` for the two
oldest. Only `artifacts=0` on every line means the rewrite has been run.

**This session's commits add no artifact file and still carry all 38**,
which is the part that is easy to get wrong: `git show --stat 1f0db30`
lists none, but a commit that does not *delete* a path still carries it in
its tree. The rewrite range therefore has to end at the branch tip, not at
`7659f6d`.

**Facts, re-verified this session, all unchanged:**

- The branch has **never been pushed**: no upstream is configured and
  `git ls-remote --heads origin claude/facts-vouchers` returns nothing. No
  force-push, no coordination, nobody else holds the blobs.
- `git branch -a --contains 95a6880` lists `claude/facts-vouchers` and
  nothing else.
- `git-filter-repo` is **not installed**. `filter-branch` is deprecated and
  slow and neither matters at eight local-only commits; it is one command
  that either completes or does not. An interactive rebase would stop on
  conflicts at every step once the first commit drops the paths.
- This is a **linked worktree** — the object store lives in
  `~/workspace/projects/ksd-boarddocs-rag/.git` and is shared with three
  other worktrees on other branches. That is why `filter-repo`'s
  clone-and-replace model is the wrong shape here.

```bash
cd ~/workspace/projects/ksd-vouchers

# 0. Record the pre-rewrite tip and tag it. The tag is the way back and
#    must not be deleted until the verification below has passed.
PRE=$(git rev-parse claude/facts-vouchers)
echo "pre-rewrite tip: $PRE"          # WRITE THIS DOWN
git tag pre-redaction-2026-09-16 claude/facts-vouchers

# 1. The working tree MUST be clean. filter-branch refuses to run with
#    unstaged changes, and this tree has 38 modified artifact files plus 4
#    new ones sitting uncommitted. They are cheap to reproduce -- one run of
#    samples.py and five of export_cycle.py -- and they have to be
#    regenerated after C3 is ruled on anyway, so discard them rather than
#    parking them.
git status --short                      # expect the artifacts, nothing else
git checkout -- facts/vouchers/samples exports

#    The four new sample files are untracked, so checkout does not touch
#    them. Move them aside; the rewrite is about to remove the whole
#    samples path from the tree.
mkdir -p ~/redaction-scratch
mv facts/vouchers/samples/2026-01-14-ACH.md \
   facts/vouchers/samples/2026-01-14-ASB.md \
   facts/vouchers/samples/2026-01-14-GF.md \
   facts/vouchers/samples/2026-02-11-GF.md ~/redaction-scratch/

git status --short                      # must now print NOTHING

# 2. Remove both paths from every commit on this branch.
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --index-filter \
  'git rm -r --cached --ignore-unmatch facts/vouchers/samples exports' \
  --prune-empty 29557ed..claude/facts-vouchers

# 3. Drop the rewrite's own backup ref. NOT the tag from step 0 — that one
#    stays until the verification passes.
git update-ref -d refs/original/refs/heads/claude/facts-vouchers
```

Do **not** run `git reflog expire` / `git gc --prune=now` yet. Those destroy
the old objects, and the tag from step 0 is only useful while they exist.
Run them after the verification below passes and you are satisfied.

### Verification 1 — per commit, per file, by content

This greps the *contents* of every tracked sample and export file in every
commit for every withheld payee name, rather than trusting that the paths
are gone. Names of six characters or fewer are excluded because they match
inside ordinary words, which is the same filter `test_no_leaks.py` applies
and for the same reason.

```bash
cd ~/workspace/projects/ksd-vouchers

# The name list comes from the worksheet, which is not in git.
python3 - <<'PY' > /tmp/withheld-names.txt
import csv
names = {r['payee'].strip() for r in csv.DictReader(
    open('reports/withheld-payees-2026-09-16.csv', encoding='utf-8'))}
print('\n'.join(sorted(n for n in names if len(n) > 6)))
PY
wc -l /tmp/withheld-names.txt          # expect ~10,000

fail=0
for c in $(git rev-list 29557ed..claude/facts-vouchers); do
  files=$(git ls-tree -r --name-only "$c" -- facts/vouchers/samples exports)
  n=$(printf '%s\n' "$files" | grep -c . || true)
  hits=0
  for f in $files; do
    if git show "$c:$f" | grep -F -i -q -f /tmp/withheld-names.txt; then
      hits=$((hits + 1))
      echo "  LEAK  $(git log -1 --format=%h "$c")  $f"
    fi
  done
  printf '%s  artifacts=%-3s leaking=%s\n' \
    "$(git log -1 --format='%h %s' "$c" | cut -c1-50)" "$n" "$hits"
  [ "$hits" -eq 0 ] || fail=1
done
rm -f /tmp/withheld-names.txt

if [ "$fail" -eq 0 ]; then
  echo "VERIFIED: no tracked sample or export file in any commit contains a withheld payee name."
else
  echo "NOT CLEAN — the rewrite did not take. Restore with the tag; do not push."
fi
```

**Before the rewrite this prints `artifacts=38` at the tip, `artifacts=24`
for the four commits beneath it, and `leaking=` a non-zero number on most of
them.** That is what it looks like when it has NOT been run — a run that
prints `artifacts=0 leaking=0` everywhere and nothing else is the pass.

### Verification 2 — the tag really can restore the old branch

This proves the way back exists without undoing the rewrite:

```bash
# The tag still resolves to the exact commit the branch was at.
test "$(git rev-parse pre-redaction-2026-09-16^{commit})" = "$PRE" \
  && echo "OK: the tag pins the pre-rewrite tip"

# Its tree and every blob under it are still present in the object store.
git cat-file -e "$PRE^{tree}" && echo "OK: the pre-rewrite tree is intact"
git ls-tree -r --name-only "$PRE" -- facts/vouchers/samples exports | wc -l
#   expect 38 — the artifacts still exist on the tag, which is the point

# What the rewrite actually removed, as a diff you can read.
git diff --stat pre-redaction-2026-09-16 claude/facts-vouchers | tail -5

# The restore itself, IF you ever need it. Destroys the rewrite:
#   git reset --hard pre-redaction-2026-09-16
```

Once both verifications pass and you are satisfied:

```bash
git tag -d pre-redaction-2026-09-16
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### One more thing to keep out of git

`reports/withheld-payees-2026-09-16.csv` is 10,659 payee names and is not
ignored today. One `git add -A` takes it. I did not edit `.gitignore` —
that file is outside this session's write scope:

```bash
cd ~/workspace/projects/ksd-vouchers
printf '\n# Withheld-payee worksheets. Individual names; never committed.\nreports/withheld-payees-*.csv\n' >> .gitignore
git add .gitignore && git commit -S -m "security: ignore withheld-payee worksheets"
```

## The F2 period gap, restated as an input to the corpus refresh

This is a **coverage** fact, not a parse fact, and it is the one that should
drive the next scrape.

**The 2026 voucher listings leave 2026-03-13 → 2026-04-08 uncovered — 27
days with no General Fund, ACH or ASB listing in the corpus.**

| cycle | funds | period covered |
|---|---|---|
| 2026-03-25 | ACH, ASB, GF | 2026-02-06 → **2026-03-12** |
| — | — | **← the gap: 2026-03-13 → 2026-04-08 →** |
| 2026-05-27 | ACH, ASB, Capital, GF | **2026-04-09** → 2026-05-14 |

There is no April meeting listing in the corpus, and the meeting sequence
runs 2026-03-25 → 2026-05-27 with nothing between.

**Two qualifications, both of which matter to whoever plans the refresh:**

- **One set does cover the window.** `2026-05-27:Transportation` records
  exactly `2026-03-13 → 2026-04-08` — one line, $173,922.13, check 900062
  dated 2026-04-09, a school bus. So the period was not skipped by the
  district; the gap is in what reached this corpus for the other funds, and
  the single Transportation listing is evidence that an April-period
  presentation exists to be found.
- **The Capital fund's gap is wider.** `2026-03-25:Capital` records
  `2026-01-09 → 2026-02-05` — the *February* period, the same window as the
  2026-02-11 sets. Taken at face value the Capital fund has no listing
  covering 2026-02-06 → 2026-04-08, a 62-day gap. It could equally be the
  period-locator defect from Open item 4 reading the wrong line off that
  page. **Confirm against the source PDF before treating it as a coverage
  gap** — that is a five-minute check and it changes what the refresh needs
  to fetch.

**The P-card side gaps too, and differently.** 2026-03-25 covers p-cards
`2026-01-17 → 2026-02-28`; 2026-05-27 covers `2026-03-14 → 2026-04-24`.
That leaves **2026-03-01 → 2026-03-13** with no p-card period, a separate
13-day hole that does not line up with the warrant gap and so needs its own
line in the refresh plan.

## System state summary

- **Database:** rebuilt and current. 459 sets, 482,395 lines, 13,306
  vendors, 683 parse-log rows. 108 sets reconcile, up from 88. Schema and
  views were applied on 2026-09-15 and are unchanged; `amount_paren` is now
  populated — 1,388 rows across 86 sets. Nothing was written to the
  database this session.
- **Working tree:** clean except for the deliberately uncommitted
  artifacts — `facts/vouchers/samples/` (32 files, 4 new),
  `exports/` (10 files), and `reports/withheld-payees-2026-09-16.csv`.
- **Tests:** 351 passed, 1 failed, **0 skipped**. The failure is
  `test_no_leaks.py` and is Open items 1 and 2. Last session: 342 passed, 1
  skipped — the skip was this same check.
- **Fixtures:** 37 PASS, 0 FAIL, 0 BLOCKED, 24 REPORT. Both HARD hand sums
  tie. R1's 35 HARD passes plus the two hand sums, which did not exist then.
- **Pre-commit:** ran on the commit — `ruff`, `ruff-format`, `bandit`,
  `interrogate`, `gitleaks`, `detect-private-key` and the file hygiene
  hooks. No hook was disabled or bypassed and no `--no-verify` was used.
- **Credentials:** none were present, sought, extracted or stored. No
  `.env` was read. All database access went through
  `podman exec -i boarddocs-postgres psql`, directly or through the
  read-only transport.
- **Scratch:** `facts/vouchers/_build/` is gitignored and holds the
  operator's rebuild logs plus this session's `fixtures_2026-09-16.txt`,
  `leak_triage.py`, `leak_triage2.py` and their JSON output. The triage
  JSON contains payee names and stays out of git with the rest of `_build`.

## Next session starts at

Whichever of the operator's five steps above is next. If the rewrite has
been run, start by re-running Verification 1 before anything else — it is
cheap, and a rewrite that half-took is worse than one that did not.

If C3 has been ruled on, the work is: implement the ruling in
`samples.py`/`export_cycle.py`, apply the three `test_no_leaks.py` patches
from the report, regenerate, and confirm the suite is 0 failed and 0
skipped.
