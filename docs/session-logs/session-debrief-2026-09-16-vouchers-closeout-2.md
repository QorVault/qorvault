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

**Update, later the same day.** The operator ruled on all four open
questions and the rulings are implemented in a second commit. What changed:
descriptions and quotes are masked, the leak check **passes**, the fixture
names are hashes, the period invariant exists with its two reason codes and
a parser that no longer reads a period out of a line item, and the transport
is documented and hardened. **386 tests pass, 0 fail, 0 skip.** The
artifacts are still not committed, per the ruling. Full detail is in
`reports/vouchers-closeout-2026-09-16.md` under "Rulings implemented"; the
sections below are the state that produced the rulings and the operator
sequence, which is unchanged except where marked.

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

4. **`test_no_leaks.py` was not modified.** *(Superseded by the rulings —
   all three patches are now applied. See the rulings section below.)*
   Three of its four failure causes
   are defects in the test rather than leaks, and I am confident of the
   diagnosis — but changing what a HARD privacy test accepts carries the
   same evidence burden as changing the classifier, and that is the
   operator's decision. The exact patch for each is in the report under
   Recommended changes.

5. **The regenerated artifacts were not committed.** *(Still true after the
   rulings, but for a different reason: they are now clean, and the operator
   instructed that they be held until the allowlist is populated.)*
   They are strictly
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

**4. ~~Rule on C3 and fix the leak test.~~ DONE — both are built and the
leak check passes.** Replaced by: **run `_build/rebuild2.sh`**, the second
reload. Written this session, **not run**, same shape as the `rebuild.sh`
you ran on 2026-09-16 — schema, dry run, STOP check, and only then a write.

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
bash _build/rebuild2.sh
```

What it does, in order, and what each step refuses on:

| Step | Refuses if |
|---|---|
| applies `schema.sql` | it errors — the reload writes rows the OLD CHECK constraint rejects, so this must go first |
| `build.py --dry-run` | the parse fails |
| STOP check vs `_build/baseline-sets-before-rebuild2.psv` | the baseline does not hold exactly **108** reconciling sets, or the dry run reconciles anything other than 108, or any of those 108 stops |
| `build.py --reload` | the load fails |
| re-check against the tables | any of the 108 stopped reconciling |
| `_build/compare_sets.py` | **any** dollar figure, line count or reconciliation status moved, or the reason-code changes are not exactly **12** |
| `fixtures.py` | the suite does not report `FAIL 0` |

The baseline was captured from the live tables this session —
`sha256 e1c98661…`, 459 sets, 108 reconciling — so the STOP check compares
against what is really there rather than against a prediction. The
`108` assertion exists for the reason the first rebuild recorded: a
comparison that matches zero rows on both sides prints nothing and reads
exactly like a pass.

`compare_sets.py` compares money as `Decimal`, never as text. That is not
fussiness — a first attempt at this comparison during the session reported
ten changed dollar figures that were all `0` against `0.00`. The gate was
pre-flighted against the dry run and was tested in both directions: it
returns 0 on the expected outcome and 1 when a dollar figure is nudged by a
cent or the reason-code count is off by one.

**5. Regenerate and commit the artifacts.**

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
export VOUCHERS_DB_TRANSPORT=podman        # or export PGPASSWORD yourself

.venv/bin/python samples.py
for d in 2026-03-25 2026-05-27 2026-06-24 2026-07-22 2026-08-26; do
  .venv/bin/python export_cycle.py "$d"    # POSITIONAL. --date does not exist.
done
.venv/bin/python -m pytest -q -rs          # must be 0 failed and 0 skipped
                                           # 386 tests as of 2026-09-16
cd ~/workspace/projects/ksd-vouchers
git add facts/vouchers/samples exports
git commit -S -m "feat: publish 2026 voucher samples and cycle exports with payee withholding"
```

**6. Fast-forward**, once the readiness line below says ready.

## Fast-forward readiness

**NOT READY.** It stays NOT READY until both of these are true, and neither
is true today:

1. **The history rewrite has been run and verified.** 460 distinct withheld
   payee names are in the current history across 35 tracked files. This is
   unchanged from 2026-09-15 and is the blocking condition.
2. **The allowlist has been populated and the artifacts regenerated against
   it**, then committed.

What the rulings did change is *why* item 2 is outstanding. It is no longer
blocked on a defect: C3 is ruled on and built, the three leak-test defects
are fixed, `test_no_leaks.py` **passes** on the regenerated artifacts, and
the search of every tracked file finds **no individual's name anywhere in
the voucher package or its documents**. What remains is a decision — which
organisations to publish — and 45 candidates are listed in the report. The
artifacts are held out of the commit on the operator's explicit instruction,
not because anything is wrong with them.

One more thing must happen before the branch is fit to merge, and it is new:
**`build.py --reload` has to run again.** The period work is in the code and
not in the tables, and the new `contract_period_invariant` fixture check
fails until it does — truthfully, because the database still holds the
pre-ruling build.

Everything else — the rebuild, the STOP check, the HARD fixtures, the
reason codes — is done and evidenced.

## The history rewrite — commands, unexecuted (RE-ISSUED 2026-09-16)

**The earlier version of this section is superseded. Do not use it — its
scope is too narrow.** It removed `facts/vouchers/samples` and `exports`
from every tree, which was the whole job when it was written. It is no
longer the whole job.

**Why the scope grew.** Ruling 2 replaced individuals' names with hashes and
with invented fixtures, and redacted them from four reports and debriefs.
That fixed the **working tree and `HEAD`** — but a commit keeps what it
recorded, and those names are still in the older commits. Measured, with
`git grep` over every commit for all 49 individuals, **excluding
`facts/vouchers/samples` and `exports`** -- those carry names in every commit
and this same rewrite deletes them outright:

```
HEAD     files=0     <- clean; every commit below it is not
7cebf3f  files=7
e644038  files=7
1e23308  files=7
1f0db30  files=7
7659f6d  files=4
512e837  files=3     <- and 3 in each commit below it
...
da22888  files=1
```

Seven paths carry them across eleven commits:

```
facts/vouchers/vendors.py
facts/vouchers/test_parsers.py
facts/vouchers/test_payees.py
docs/session-logs/session-debrief-2026-09-15-vouchers-closeout.md
reports/facts-vouchers-recon-2026-09-14.md
reports/vouchers-closeout-2026-09-15.md
reports/vouchers-r1-2026-09-15.md
```

So the rewrite now does **two** things in one pass: drop the artifact paths,
and rewrite the content of those seven files wherever they appear.

**Facts, re-verified, all unchanged:** the branch has never been pushed
(no upstream, `git ls-remote --heads origin claude/facts-vouchers` returns
nothing); `git branch -a --contains 95a6880` lists only this branch;
`git-filter-repo` is not installed; this is a linked worktree whose object
store is shared with three other worktrees, which is why filter-repo's
clone-and-replace model is the wrong shape here.

### 0. The substitution set, and why it is not in the repository

`facts/vouchers/_build/redaction-subs.tsv` holds 49 `old<TAB>new` pairs. It
is a list of individuals' names, so it lives in gitignored scratch and is
moved out of the tree entirely before the rewrite runs.

It has been **verified**: every affected blob in every commit was piped
through the filter and re-searched, and no individual's name survives in any
of them.

```bash
cd ~/workspace/projects/ksd-vouchers
cp facts/vouchers/_build/redaction-subs.tsv  ~/redaction-subs.tsv
cp facts/vouchers/_build/redact_stream.py    ~/redact_stream.py
export REDACTION_SUBS=~/redaction-subs.tsv
```

Write the index filter to its own file — quoting a loop inside
`--index-filter '...'` is how this goes wrong:

```bash
cat > ~/redact-index-filter.sh <<'SCRIPT'
set -e
git rm -r --cached --ignore-unmatch facts/vouchers/samples exports >/dev/null
for f in facts/vouchers/vendors.py \
         facts/vouchers/test_parsers.py \
         facts/vouchers/test_payees.py \
         docs/session-logs/session-debrief-2026-09-15-vouchers-closeout.md \
         reports/facts-vouchers-recon-2026-09-14.md \
         reports/vouchers-closeout-2026-09-15.md \
         reports/vouchers-r1-2026-09-15.md ; do
    entry=$(git ls-files -s -- "$f")
    [ -n "$entry" ] || continue
    mode=$(printf '%s' "$entry" | cut -d' ' -f1)
    blob=$(printf '%s' "$entry" | cut -d' ' -f2)
    new=$(git cat-file blob "$blob" | python3 "$HOME/redact_stream.py" | git hash-object -w --stdin)
    git update-index --cacheinfo "$mode,$new,$f"
done
SCRIPT
chmod +x ~/redact-index-filter.sh
```

### 1. Safety tag, and a clean tree

```bash
cd ~/workspace/projects/ksd-vouchers
PRE=$(git rev-parse claude/facts-vouchers)
echo "pre-rewrite tip: $PRE"          # WRITE THIS DOWN
git tag pre-redaction-2026-09-16 claude/facts-vouchers
```

filter-branch refuses to run with unstaged changes, and this tree has the
regenerated artifacts sitting uncommitted. They are cheap to reproduce — one
run of `samples.py` and five of `export_cycle.py` — and they have to be
regenerated after the allowlist is populated anyway, so discard them:

```bash
git status --short                      # expect the artifacts, nothing else
git checkout -- facts/vouchers/samples exports
mkdir -p ~/redaction-scratch
mv facts/vouchers/samples/2026-01-14-ACH.md \
   facts/vouchers/samples/2026-01-14-ASB.md \
   facts/vouchers/samples/2026-01-14-GF.md \
   facts/vouchers/samples/2026-02-11-GF.md ~/redaction-scratch/
git status --short                      # must now print NOTHING
```

### 2. The rewrite

```bash
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch \
  --index-filter 'bash ~/redact-index-filter.sh' \
  --prune-empty 29557ed..claude/facts-vouchers

git update-ref -d refs/original/refs/heads/claude/facts-vouchers
```

Do **not** run `git reflog expire` / `git gc --prune=now` yet. Those destroy
the old objects, and the tag from step 1 is only useful while they exist.

### Verification 1 — artifacts gone, by content not by path

```bash
cd ~/workspace/projects/ksd-vouchers

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
      hits=$((hits + 1)); echo "  LEAK  $(git log -1 --format=%h "$c")  $f"
    fi
  done
  printf '%s  artifacts=%-3s leaking=%s\n' \
    "$(git log -1 --format='%h %s' "$c" | cut -c1-50)" "$n" "$hits"
  [ "$hits" -eq 0 ] || fail=1
done
rm -f /tmp/withheld-names.txt
[ "$fail" -eq 0 ] && echo "VERIFIED: no tracked sample or export file in any commit contains a withheld payee name."
```

Before the rewrite this prints `artifacts=38` for every commit from
`7659f6d` up and `artifacts=24` for the four below it, with `leaking=` a
non-zero number on most. **`artifacts=0 leaking=0` on every line is the
pass.**

### Verification 2 — no individual's name in any commit, anywhere

This is the new one, and it is the check that ruling 2 is actually enforced
in history rather than only at the tip:

```bash
cd ~/workspace/projects/ksd-vouchers
grep -v '^#' ~/redaction-subs.tsv | cut -f1 > /tmp/individuals.txt

fail=0
for c in $(git rev-list 29557ed..claude/facts-vouchers); do
  n=$(git grep -I -F -f /tmp/individuals.txt -l "$c" 2>/dev/null | wc -l)
  printf '  %s  files-naming-an-individual=%s\n' "$(git log -1 --format=%h "$c")" "$n"
  [ "$n" -eq 0 ] || fail=1
done
rm -f /tmp/individuals.txt
[ "$fail" -eq 0 ] && echo "VERIFIED: no commit on this branch names an individual."
```

Every line must read `files-naming-an-individual=0`.

**Run before the rewrite it reports non-zero on every commit including the
tip** -- at the time of writing, 7 at `HEAD` and up to 14 further down. That
is not a contradiction of "HEAD is clean": this check deliberately does
**not** exclude `facts/vouchers/samples` and `exports`, and at `HEAD` all
seven hits are tracked sample files still holding their pre-masking content.
The rewrite deletes those paths, so after it runs the same command covers
everything and every line reads zero.

### Verification 3 — the tag really can restore the old branch

```bash
test "$(git rev-parse pre-redaction-2026-09-16^{commit})" = "$PRE" \
  && echo "OK: the tag pins the pre-rewrite tip"
git cat-file -e "$PRE^{tree}" && echo "OK: the pre-rewrite tree is intact"
git ls-tree -r --name-only "$PRE" -- facts/vouchers/samples exports | wc -l
#   expect 38 — the artifacts still exist on the tag, which is the point
git diff --stat pre-redaction-2026-09-16 claude/facts-vouchers | tail -5

# The restore itself, IF you ever need it. Destroys the rewrite:
#   git reset --hard pre-redaction-2026-09-16
```

### Verification 4 — the content rewrite changed the 49 substitutions and nothing else

Verification 2 proves no individual's name survives. It does **not** prove
the filter left everything else alone — a substitution script that also
mangled an unrelated line would pass it. This one closes that gap, and it is
the check that makes the content rewrite safe to accept.

The method is stronger than reading a diff: for every commit and every one of
the seven rewritten paths, take the **original** blob from the safety tag's
history, pipe it through the same filter, and require the result to be
**byte-identical** to what is in the rewritten commit. Anything the filter
did that the substitution set does not account for shows up as a mismatch.

Commits are paired by subject line, which is unique across all of them
(checked: `git log --format=%s | sort | uniq -d` prints nothing), so the
pairing survives even if `--prune-empty` were to drop one. It will not —
no commit on this branch touches only the artifact paths.

```bash
cd ~/workspace/projects/ksd-vouchers
export REDACTION_SUBS=~/redaction-subs.tsv

PATHS="facts/vouchers/vendors.py
facts/vouchers/test_parsers.py
facts/vouchers/test_payees.py
docs/session-logs/session-debrief-2026-09-15-vouchers-closeout.md
reports/facts-vouchers-recon-2026-09-14.md
reports/vouchers-closeout-2026-09-15.md
reports/vouchers-r1-2026-09-15.md"

fail=0; checked=0; rewritten=0
for new in $(git rev-list 29557ed..claude/facts-vouchers); do
  subject=$(git log -1 --format=%s "$new")
  old=$(git log --format='%H %s' 29557ed..pre-redaction-2026-09-16 \
        | grep -F -- "$subject" | head -1 | cut -d' ' -f1)
  if [ -z "$old" ]; then
    echo "  UNPAIRED: $(git log -1 --format=%h "$new")  $subject"; fail=1; continue
  fi
  for f in $PATHS; do
    git cat-file -e "$old:$f" 2>/dev/null || continue
    checked=$((checked + 1))
    git show "$old:$f" | python3 "$HOME/redact_stream.py" > /tmp/expected.blob
    git show "$new:$f" > /tmp/actual.blob
    if cmp -s /tmp/expected.blob /tmp/actual.blob; then
      if ! cmp -s <(git show "$old:$f") /tmp/actual.blob; then
        rewritten=$((rewritten + 1))
      fi
    else
      echo "  MISMATCH  $(git log -1 --format=%h "$new")  $f"
      diff <(git show "$old:$f") /tmp/actual.blob | head -20
      fail=1
    fi
  done
done
rm -f /tmp/expected.blob /tmp/actual.blob

echo "checked $checked file-versions; $rewritten of them were rewritten"
[ "$fail" -eq 0 ] && echo "VERIFIED: every rewritten file differs from its original by the substitutions and by nothing else."
```

**Expect `checked=67` and `rewritten=46`** at the time of writing — the
seven paths do not all exist in all twelve commits, and `HEAD` was already
clean so its files pass unchanged. What matters is that **`MISMATCH` never
prints**. A mismatch means the filter did something the substitution set
does not explain; do not proceed, and restore from the tag.

### Verification 5 — the substitution file is gone

The substitution set is 49 individuals' names. It has done its job the moment
Verification 4 passes, and from then on it is only a liability sitting in the
working tree. Delete it and prove it is gone — **after** Verifications 1
through 4, never before, because every one of them needs it.

```bash
cd ~/workspace/projects/ksd-vouchers

rm -f facts/vouchers/_build/redaction-subs.tsv \
      ~/redaction-subs.tsv \
      ~/redact-index-filter.sh \
      ~/redact_stream.py

# 1. The files themselves are gone.
for f in facts/vouchers/_build/redaction-subs.tsv ~/redaction-subs.tsv \
         ~/redact-index-filter.sh ~/redact_stream.py; do
  if [ -e "$f" ]; then echo "  STILL PRESENT: $f"; else echo "  gone: $f"; fi
done

# 2. Nothing anywhere under _build/ still holds the substitution table.
#    redact_stream.py is a copy of the one in _build; that copy goes too.
rm -f facts/vouchers/_build/redact_stream.py \
      facts/vouchers/_build/make_redaction_subs.py
grep -rl "individual payee, name withheld" facts/vouchers/_build/ 2>/dev/null \
  && echo "  ^ these still reference the substitution set -- read them and decide" \
  || echo "  gone: no file under _build/ references the substitution set"

# 3. It was never in git, and is not now.
git log --all --oneline -- '*redaction-subs*' | head -1
git ls-files --error-unmatch facts/vouchers/_build/redaction-subs.tsv 2>&1 | head -1
#   both must report that git has never heard of it
```

**What is deliberately NOT deleted:** `_build/payee_fixture_map.txt`. That is
the hash-to-name lookup for `test_payees.py`, and it is how a failing fixture
is traced back to the payee it identifies. It is gitignored, it is
regenerable with `python test_payees.py`, and it is the operator's working
tool rather than a leftover. Delete it too if you would rather regenerate it
on demand — nothing depends on it existing.

### Then, and only then

```bash
cd facts/vouchers && VOUCHERS_DB_TRANSPORT=podman .venv/bin/python -m pytest -q -rs
#   the rewrite changed test_parsers.py and test_payees.py in history but
#   not at the tip, so this must still be 386 passed, 0 failed, 0 skipped

cd ~/workspace/projects/ksd-vouchers
git tag -d pre-redaction-2026-09-16
git reflog expire --expire=now --all
git gc --prune=now --aggressive
rm -f ~/redaction-subs.tsv ~/redact_stream.py ~/redact-index-filter.sh
rm -rf ~/redaction-scratch
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

- **Database:** holds the 2026-09-16 rebuild. 459 sets, 482,395 lines,
  13,306 vendors, 683 parse-log rows. 108 sets reconcile, up from 88;
  `amount_paren` populated on 1,388 rows across 86 sets. **Nothing was
  written to the database this session**, and it is now one build behind the
  code: the period work and the two new reason codes are in `build.py`,
  `parsers.py` and `schema.sql` and not in the tables. `schema.sql` must be
  re-applied before the next reload, because the reload writes rows the old
  CHECK constraint rejects.
- **Working tree:** clean except for the deliberately uncommitted
  artifacts — `facts/vouchers/samples/` (32 files, 4 new),
  `exports/` (10 files), and `reports/withheld-payees-2026-09-16.csv`.
- **Tests:** **386 passed, 0 failed, 0 skipped.** Last session: 342 passed
  and 1 skipped, the skip being the leak check; earlier today, 351 passed
  and 1 failed, the failure being the same check once it could finally run.
  It passes now.
- **Fixtures:** 37 PASS, **1 FAIL**, 0 BLOCKED, 24 REPORT. Both HARD hand
  sums tie. The one failure is `contract_period_invariant`, new today,
  reporting truthfully that the database still holds the pre-ruling build:
  two sets have a period ending before it begins and neither yet carries
  `PERIOD_INVALID_AT_SOURCE`. **It passes after `build.py --reload`.**
- **Pre-commit:** ran on the commit — `ruff`, `ruff-format`, `bandit`,
  `interrogate`, `gitleaks`, `detect-private-key` and the file hygiene
  hooks. No hook was disabled or bypassed and no `--no-verify` was used.
- **Credentials:** none were present, sought, extracted or stored. No
  `.env` was read. All database access went through
  `podman exec -i boarddocs-postgres psql`, directly or through the
  read-only transport.
- **Scratch:** `facts/vouchers/_build/` is gitignored. It holds the
  operator's rebuild logs, this session's `dryrun_2026-09-16.log`, the
  leak-triage scripts and their JSON, and three files the next steps need:
  **`redaction-subs.tsv`** (49 name substitutions for the rewrite),
  **`redact_stream.py`** (the filter that applies them) and
  **`payee_fixture_map.txt`** (hash-to-name lookup for `test_payees.py`).
  All three contain or resolve individuals' names and must stay out of git.
  It also holds the second reload, written and **not run**:
  **`rebuild2.sh`**, **`compare_sets.py`**, and the baseline it gates on,
  **`baseline-sets-before-rebuild2.psv`**.

## Rulings of 2026-09-16 — decisions and what changed

The operator's rulings arrived after the first commit. Five decisions were
mine to make inside them, and each is here because it could reasonably have
gone the other way.

7. **One object, not one function, does the masking and the checking.**
   `WithheldNameIndex` is constructed by `samples.py`, `export_cycle.py` and
   `test_no_leaks.py`, each from its **own** view of who is published —
   samples without the watch list, exports with it, matching what each
   already does in its payee column. Finding D2 was that a shared *function*
   did not prevent drift because the callers passed different arguments; a
   shared object that carries the decision is the fix.

8. **The length cut-off became a token rule, and this one matters.** The old
   test ignored withheld names of six characters or fewer, because without
   anchoring a short name matches inside ordinary words. Carrying that over
   would have excluded 23 real people from the control — two short tokens
   each, the shape common in Vietnamese, Chinese and Korean names — while
   protecting nobody, since all 107 single-token short names are acronyms.
   The rule is now two-or-more tokens **or** seven-plus characters. A flat
   length cut-off fails hardest on the payees with the shortest names, and
   that is not a neutral way for a privacy control to fail.

9. **An invalid period keeps its dates.** `PERIOD_INVALID_AT_SOURCE` flags
   the set and leaves `period_start` and `period_end` exactly as printed.
   Blanking them would hide the district's own error behind what reads as a
   missing field, and a reader checking this layer against the page needs to
   see what the page shows.

10. **Period codes never overwrite a reconciliation code.** `reason_code`
    holds one value and a set that does not balance has something more
    important to say than that its header was quiet. The invariant is
    asserted independently in `fixtures.py`, so nothing depends on the code
    winning the slot. Consequence, recorded because it is a real gap: two
    sets lose an invented period with no code marking it, and only the note
    on the set records it.

11. **The date pattern was NOT widened.** Two era-B/C listings state a
    hyphenated period — `3-9-17 through 3-16-17` — that `PERIOD_RX` cannot
    read, so they will be recorded `PERIOD_NOT_STATED` when the truth is
    "stated in a form this layer does not read". Widening the pattern adds
    data rather than removing wrong data, was not what the ruling asked for,
    and needs its own verification pass. The note attached to the set says
    what is actually true instead. Open item.

### What changed, second commit

| File | Change |
|---|---|
| `facts/vouchers/vendors.py` | `WithheldNameIndex` — bucketed find-and-mask over free text, skipping any hit inside a longer published payee name; `is_maskable`; `name_pattern` anchored with lookarounds; real payee names removed from comments. |
| `facts/vouchers/samples.py` | Masks the description column and the verbatim quote; own-payee `redact_name` runs first so an unlocatable name still suppresses the whole quote. |
| `facts/vouchers/export_cycle.py` | Masks the rendered markdown and every CSV cell; reports how many names it masked. |
| `facts/vouchers/test_no_leaks.py` | All three patches: watch list, suffix variants, word boundaries — by using the writers' own index. |
| `facts/vouchers/test_payees.py` | Fixtures are SHA-256 hashes resolved against the live payee table; three integrity tests; a new masking test over all 31 F1+F4 shapes. |
| `facts/vouchers/parsers.py` | `header_region()`; the period is read from the header only, never from a line item. |
| `facts/vouchers/build.py` | `apply_period_status()`, the two reason codes, and their parse-log statuses. |
| `facts/vouchers/fixtures.py` | `check_period_invariant`, registered in the runner. |
| `facts/vouchers/schema.sql` | Both new codes in both CHECK constraints; `CREATE TABLE` status list re-synced with the `ALTER` block it had drifted from. |
| `facts/vouchers/db.py` | `standard_conforming_strings` verified rather than assumed; correct literal rendering for backslashes; UTF-8 pinned end to end. |
| `facts/vouchers/test_parsers.py` | 8 real payees replaced with invented names of the same shape. |
| `docs/data-paths.md` | Why `pg_hba` demands a password from the host, and the credential-free read path. |
| `reports/*.md`, `docs/session-logs/*.md` | Individuals' names replaced with descriptions. Figures and findings untouched. |

### The number the ruling asked for

323 tracked text files searched against all 10,656 withheld payees:

- `facts/vouchers/samples/` and `exports/` with any withheld name: **0**
- voucher package, its reports and debriefs, with an **individual**: **0**
- rest of the repository with an individual: **3**, all in `facts/minutes`,
  outside this session's write scope and reported not fixed

## Next session starts at

Whichever of the operator's steps is next, in the order in "What the
operator must do next".

**If the rewrite has been run**, start by re-running Verifications 1 and 2
before anything else. They are cheap, and a rewrite that half-took is worse
than one that did not. Verification 2 is the new one and the one that
matters most: every commit must report
`files-naming-an-individual=0`.

**If the reload has been run**, confirm `fixtures.py` reports 38 PASS / 0
FAIL, and confirm the 12 expected reason-code changes and no others —
`build.py --dry-run` produced that list today and the comparison method is
in the report.

**If the allowlist has been populated**, regenerate samples and exports,
confirm `test_no_leaks.py` still passes, and commit the artifacts. That is
the last blocker before fast-forward.

## Rewrite executed — 2026-09-17

**Pre-rewrite tip: `35738cf2849a4ffa2f77de22dc85007de3c3254e`.**
Post-rewrite tip: `782dd5a`. The safety tag `pre-redaction-2026-09-16` is
still in place and the "Then, and only then" block — tag delete, reflog
expire, `gc` — was **not** run, so `git reset --hard
pre-redaction-2026-09-16` remains available. `refs/original` was deleted as
the plan specifies.

The tip hash was recovered from `claude/facts-vouchers@{1}`, the reflog entry
immediately below `filter-branch: rewrite`, and cross-checked against the
commit line printed when the `.gitignore` commit was made. It was
deliberately **not** read off the tag: Verification 3 exists to compare the
tag against that value, and sourcing it from the tag would have made the
check compare the tag to itself.

### Deviations from the plan as written, and why

- **The `.gitignore` commit ran first**, ahead of Step 1. This is why
  `git status --short` was already clean at Step 1 despite
  `reports/withheld-payees-2026-09-16.csv` sitting untracked — the new
  ignore rule covered it. Running it first is the better order and should be
  the recorded one.
- **`git checkout -- facts/vouchers/samples exports` was omitted.** The 38
  regenerated artifacts were not dirty; they had been stashed as `stash@{0}
  regenerated artifacts pre-rewrite`, where they remain. The `mkdir`/`mv` of
  the four untracked new samples to `~/redaction-scratch/` was kept and ran.
- **Blocks 0, the safety tag, the rewrite, and Verification 5 were run by
  the operator by hand.** The command-review hook escalated the Step 0 block
  on the literal text `git rm -r --cached` inside the index-filter heredoc,
  before anything executed. The hook was not bypassed or worked around.

### Blocks

| Block | Result |
|---|---|
| `.gitignore` commit | PASS — `35738cf`, all pre-commit hooks green |
| Step 1, clean tree (`git checkout --` omitted) | PASS — four untracked samples moved to `~/redaction-scratch/`; `git status --short` printed nothing |
| Step 0, substitution set out of tree + index filter | PASS — run by hand after hook escalation |
| Safety tag | PASS — `pre-redaction-2026-09-16` |
| Step 2, the rewrite | PASS — run by hand; 16 commits rewritten, `refs/original` deleted |

### Verifications

| # | Result |
|---|---|
| 1 — artifacts gone by content | **PASS** — `artifacts=0 leaking=0` on all 16 commits, no `LEAK` line. Name list built from 10,527 withheld payees. |
| 2 — no individual named in any commit | **PASS** — `files-naming-an-individual=0` on all 16, tip included. |
| 3 — the tag restores the old branch | **PASS** — tag pins the pre-rewrite tip, tree intact, 38 artifacts still reachable from the tag, diff 38 files / 11,370 deletions. |
| 4 — substitutions only, nothing else | **PASS** — `checked 81 file-versions; 46 of them were rewritten`; no `MISMATCH`, no `UNPAIRED`. |
| 5 — the substitution file is gone | **PASS** — all four `gone:` lines, git has never heard of the file. |

**On Verification 4's counts.** This section predicted `checked=67,
rewritten=46`. Checked came in at **81** because the rewrite range holds 16
commits rather than the twelve it held when the prediction was written; the
extra commits carry copies of the seven paths that were already clean, so
they verify and pass unchanged. **`rewritten=46` matched exactly** — the
same 46 file-versions that were supposed to change are the ones that
changed, each byte-identical to its original piped through the filter.

**On Verification 5's one flagged path.** `_build/verify_section.md` matched
the replacement phrase but not the name list. It is the drafting copy of
Verifications 4 and 5, and its whole text is a verbatim substring of this
debrief (91 of 91 non-blank lines). The match is the check finding its own
source: line 92 of that file *is* the V5 `grep` command, which quotes the
phrase as its search string. It holds no individual's name.

### The pytest line — FAILED the stated condition

```
1 failed, 385 passed in 1.26s
```

This section required `386 passed, 0 failed, 0 skipped`. The failure is
`test_no_leaks.py::test_there_are_files_to_check` —
*"A green run over zero files is not evidence of anything"* — asserting that
`_published_files()` is non-empty. It is empty: `facts/vouchers/samples/`
and `exports/` no longer exist in the working tree, because the rewrite
removed them from every commit and `filter-branch` checked out the rewritten
tip. `git ls-tree -r HEAD -- facts/vouchers/samples exports` returns 0.

**This is the guard doing its job, not damage from the rewrite.** The suite
still holds 386 tests (1 + 385); none were lost. The expectation written
here accounted for the content rewrite of `test_parsers.py` and
`test_payees.py` but not for the artifact deletion that the same rewrite
performs — and the `386 passed` figure was originally measured *with* the
regenerated artifacts present in the tree. The same failure would have
occurred under the plan as written, because after the rewrite there is no
committed copy of the artifacts left to restore.

Two consequences worth carrying forward. First, with zero files to scan,
`test_no_published_file_contains_a_withheld_name` — the HARD control the
classifier exists to enforce — is **passing vacuously**. That is precisely
what the guard test was added to expose. Second, this resolves at step 5:
regenerate the artifacts against the populated allowlist and re-run, at
which point `386 passed, 0 failed, 0 skipped` becomes the real expectation.
Nothing was unstashed or regenerated to make the suite green.

### Scratch deletions

Five recon files were deleted from `facts/vouchers/_build/` before
Verification 5, after a name sweep found individuals' names in eight files
there. None was ever in git; all are ignored by `facts/.gitignore:8`
(`*/_build/`).

| File | Why deleted |
|---|---|
| `recon_stdout.json` | Byte-for-byte the same findings as `recon_findings.json` (identical parsed JSON; the 1-byte difference is `print()`'s trailing newline). |
| `recon_findings_run1.json` | Earlier partial run — 11 of the final 14 sections, missing `coverage_check`, `reconciliation_feasibility`, `reconciliation_by_layout`; carried 8 names against the final run's 5. |
| `recon_findings_run2.json` | Same, superseded by the 22:19 run. |
| `recon_findings.json` | Canonical Phase 0 findings, but read by nothing; its conclusions are the numbered sections of `reports/facts-vouchers-recon-2026-09-14.md`, which at line 854 already declares this file gitignored "because it contains verbatim vendor and description text." Regenerable via the committed A5 command (~25 min) while the source corpus is present. |
| `recon_cache.json` | 693-entry SHA-256 probe cache; 78 entries held names. Deleting costs a ~25-minute re-probe on the next recon run, not information. |

**`_build/payee_fixture_map.txt` is now the one remaining name-bearing file
under `_build/`.** It is kept deliberately, per the Verification 5 note
above: it is the hash-to-name lookup that traces a failing fixture back to
the payee it identifies, it is gitignored, and it is regenerable with
`python test_payees.py`.

One consequence of deleting `redaction-subs.tsv` and
`make_redaction_subs.py` together: the 49-name list no longer exists, so a
name-based sweep of anything — logs, transcripts, scratch — can no longer be
run as specified. `payee_fixture_map.txt` is now the only local source of
those names.

### Monitoring infrastructure — what it captured

Checked read-only, on the question of whether the per-minute git
auto-snapshot or terminal session recording preserved `redaction-subs.tsv`,
`redact-index-filter.sh`, or the replacement phrase.

**The git auto-snapshot is not running and never saw these files.** There is
no crontab for `donald`, `~/.config/systemd/user/` holds no unit files at
all, and `ksd-fs-watcher.service` and `ksd-log-aggregator.service` both
report `inactive`. Forty `auto-snapshot` commits do exist, all between
2026-02-27 and 2026-03-02 — six months before the voucher work — on a ref
unrelated to this branch, with **zero** inside the rewrite range. No
auto-snapshot commit ever touched a `_build/` path, which is expected since
`_build/` is gitignored.

**Terminal session recording is not installed either** — `claude-session`
and `cs` are not defined in this shell, and there are no `.cast` or
`typescript` capture files under `~/workspace`.

**What does record is Claude Code's own logging**, and it captured the
**filenames and command text but not the file contents**:

| Location | `redaction-subs` | `redact-index-filter` | the phrase |
|---|---|---|---|
| `~/.claude/history.jsonl` | 3 | 2 | 1 |
| `~/.claude/logs/ai-review.jsonl` | 7 | 2 | 0 |
| `~/.claude/file-history/<session>/` | 4 files | 4 files | 2 files |
| `~/.claude/projects/-home-donald-workspace/*.jsonl` | 2 files | 2 files | 2 files |

The substitution table has 49 rows and every row contains the replacement
phrase exactly once, so any file holding the table would show **at least 49**
phrase occurrences. The highest count anywhere is **8**. All four
`file-history` entries are markdown — 0 tab-delimited lines, 7 to 36
headings, code fences throughout, one of them 874 lines, matching this
debrief — not the TSV. The body of `redact-index-filter.sh` **is** recorded,
in the hook's review log and the session transcript, but that script names
seven paths and contains no individual's name.

Not covered by this check: whether any of the 49 names appear in those logs
on their own, independent of these three strings. That sweep was then run
against `payee_fixture_map.txt` as the name source — see the next
subsection, which supersedes this paragraph.

### The name sweep of Claude Code's logs — run 2026-09-17

**It covers 31 of the 49 names.** `redaction-subs.tsv` is gone, so the name
source is `_build/payee_fixture_map.txt`, whose `F*_INDIVIDUALS` groups hold
31 individuals. The map's other six names are organizations and single-name
markers and are not members of the substitution set. Every count below is
therefore a **lower bound**: 18 of the 49 names were not searched for at
all, and no local source for them now exists.

Read-only throughout. Nothing was deleted or modified, and no name was
printed — the scripts load names at run time and only ever count them,
because this sweep's own output lands in the transcripts being swept.

Scripts, in `_build/` and gitignored: `log_name_sweep.py` (the sweep),
`log_name_sweep_detail.py` (the per-file breakdown). Matching runs in three
tiers, each anchored at both ends so a short token cannot match inside a
longer word, and each tolerant of the separators these files actually use —
a comma, a lost space, a markdown cell divider, or a JSON-escaped `\n`,
`\r\n` or `\\n` where a name wrapped across a line. Tier A is the name in
corpus order, tier B the same name in prose order (`Surname, Given Middle`
reversed, and shortened to `Given Surname`), tier C the surname alone.
Sixteen matcher cases on fabricated names pass, including all four escape
forms and five must-not-match cases.

Scope: all of `~/.claude` — **4,406 files, 235.3 MB** — not only the four
locations in the table above. Eleven binaries under `plugins/` would not
decode as UTF-8 and were skipped; `.credentials.json` was deliberately not
opened.

**Tier A+B — a full name. Twelve files hold one, and between them they hold
all 31.**

| Location | Files scanned | Files with a hit | Occurrences | Names |
|---|---|---|---|---|
| `projects/` | 335 | 3 | 611 | 31 |
| `file-history/` | 1,774 | 9 | 168 | 31 |
| all 37 other locations | 2,297 | 0 | 0 | 0 |
| **total** | **4,406** | **12** | **779** | **31** |

`history.jsonl`, `logs/`, `debug/` (107.9 MB), `paste-cache/`, `tasks/`,
`plans/`, `shell-snapshots/` and `backups/` are **clean at tier A+B** — a
useful result on its own, since the earlier check found the *filenames* in
`history.jsonl` and `logs/ai-review.jsonl` and it would have been reasonable
to assume the names rode along with them. They did not.

The twelve files, and what each one is:

| Occurrences | Names | File | What it is |
|---|---|---|---|
| 367 | 31 | `projects/-home-donald-workspace/1a886dcf….jsonl` | Session transcript, 2026-09-15 |
| 240 | 31 | `projects/-home-donald-workspace/ff25cce9….jsonl` | Session transcript, 2026-09-16/17 |
| 4 | 1 | `projects/-home-donald-workspace/91d4328a….jsonl` | Session transcript, 2026-09-15 |
| 38 ×3 | 31 | `file-history/{1a886dcf,ff25cce9}/d9c1375…@v1,v2` | `facts/vouchers/test_payees.py`, pre-masking |
| 32, 18 | 31, 17 | `file-history/1a886dcf/8a05629…@v2,v3` | An uncommitted working document — see below |
| 1 | 1 | `file-history/1a886dcf/7fd7fd2…@v2` | `reports/vouchers-closeout-2026-09-15.md`, pre-masking |
| 1 ×3 | 1 | `file-history/{1a886dcf,ff25cce9}/c267f57…@v1,v2` | `facts/vouchers/vendors.py`, pre-masking |

Identification did not need the files' text: three of the nine
`file-history` entries hash to git blobs that still exist in the object
store, and the other six share a per-path filename hash with one of those
three. The odd one out is `8a05629…`, two versions of a 909-line, 46 KB
working document — "Hand-sum fixture candidates" and "Step 2 — Locators"
from the 2026-09-15 fixture selection — that **exists in no commit on any
ref and nowhere on disk.** It was never committed and has been deleted.
Claude Code's `file-history` is the only place it survives, and it carries
all 31 names.

**The three blobs that are still in git are reachable only from
`pre-redaction-2026-09-16`, not from `claude/facts-vouchers`.** Checked
per blob with `git rev-list --objects`. That is the safety tag doing exactly
what Verification 3 describes, not a failure of the rewrite.

**Tier C — a surname alone. An upper bound, not a finding.**

| | Occurrences | Files | Surnames |
|---|---|---|---|
| inside the 12 files above | 1,193 | 12 | 29 |
| everywhere else | 159 | 35 | 6 |
| **total** | **1,352** | **47** | **29** |

Eighty-eight percent of tier C sits inside files tier A+B already names. The
159 that do not involve **six** surnames spread thinly across unrelated
sessions, `debug/`, `plugins/` and one `paste-cache/` entry — the shape of
common surnames colliding with ordinary text rather than of a leak. Tier C
is reported because a surname written on its own is still a disclosure; it
is reported separately because most of these are coincidence.

**What this means.** The history rewrite cleaned git. It could not reach
`~/.claude`, which is outside the repository, and which holds pre-masking
copies of three source files, one deleted uncommitted document, and three
session transcripts — 779 full-name occurrences of all 31 searchable names,
plus whatever share of the unsearched 18 is in there.

**Disposition — delete all twelve.** Written as a script for the operator to
run rather than run here: `_build/claude-log-cleanup.sh`. Twelve `rm -f`
lines, then a re-run of `log_name_sweep.py` that must report zero files and
zero occurrences at the full-name tier, then removal of both sweep scripts.
The verification **gates** the removal — if the tier is not clean the scripts
stay, so whatever remains can still be found; deleting the only tool that
can check would be the wrong order. `bash -n` and `shellcheck` are clean, the
twelve paths were diffed against the sweep's own hit list with no drift, all
twelve were confirmed to exist so `rm -f` is not hiding a typo, and the gate
was tested against both a dirty and a clean sweep output as well as against
unparseable input. Nothing holds any of the twelve open and none is the live
session's transcript. **Not run.**

Two costs, weighed and accepted: deleting the three transcripts ends
`claude --resume` for sessions `1a886dcf`, `ff25cce9` and `91d4328a`, and
`8a05629…` is the last copy of that working document in existence.

A clean result from that script will mean "none of the 31", not "none of the
49". The 18 unsearched names remain unsearchable for want of a source.

**Also outstanding: `~/backups/claude-config/`.** Two tarballs dated
2026-09-16, 15:03 and 16:48, 41 MB each, hold pre-masking copies of the same
material — 36 archive members for session `1a886dcf` and 2 for `91d4328a`.
Neither contains `ff25cce9`, whose entries were written between 18:36 and
20:02 that evening, after both backups were taken. These sit outside
`~/.claude`, so the sweep never saw them and `claude-log-cleanup.sh` does not
touch them. **Scheduled for deletion after the hook batch.**

### Readiness

**NOT READY**, unchanged. The history rewrite — blocking condition 1 — is
now run and verified on all five checks. Condition 2 is untouched: the
allowlist is still unpopulated, the artifacts are still unregenerated and
uncommitted, and `build.py --reload` (`_build/rebuild2.sh`) has still not
run. The pytest line will not return `386 passed, 0 failed, 0 skipped` until
the artifacts are regenerated, and until then the leak control passes
vacuously.
