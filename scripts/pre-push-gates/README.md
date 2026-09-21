# Pre-push gates

Checks what a range of commits would publish **before** it leaves this
machine. Published git history is forever: a leaked key or a person's name
pushed to GitHub cannot be un-published, only revoked or scrubbed after the
fact. These gates are the last read-only look before `git push`.

```bash
# from anywhere inside the repo, with the vouchers venv:
facts/vouchers/.venv/bin/python scripts/pre-push-gates/run_gates.py origin/main..main
```

Exit `0` means every gate passed. Exit `1` means at least one failed and the
output says which. Exit `2` is a usage error. The runner **never writes inside
the working tree, never touches a ref, never pushes** — it reads the
repository, extracts content to a temporary directory outside it, scans, and
deletes that directory on exit.

## What "the range" means

`OLD..NEW` is read the way `git rev-list` reads it. The runner extracts every
blob reachable from `NEW` but not from `OLD` — every file version the remote
would receive for the first time, **intermediate versions included**, not
just the net diff. A file unchanged since `OLD` is not re-scanned; an old blob
reappearing under a new path is not re-scanned; a file edited in the range
is scanned as a whole, which is why a pre-existing name in an untouched
comment of an edited file still surfaces (labelled already-public, see
below).

## The four gates

| Gate | What it checks | Instrument |
|---|---|---|
| `credential-scan` | Anthropic-style key shapes (`sk-ant-api…`, OAuth, cookies) in every new blob, plus a pass inside any archive blob | `secret-scan.sh`, vendored here |
| `gitleaks` | The broad ruleset — AWS/GitHub/private keys/high-entropy strings — over the commit range itself, values redacted | `gitleaks git --log-opts=OLD..NEW`, default rules |
| `person-names` | Every new blob against the person-shaped full-name list | fixed-string, case-insensitive grep |
| `withheld-list` | New blobs under `facts/vouchers/samples/` and `exports/` against the **whole** withheld-payee list minus the allowlist. Every other blob is checked against the same list **report-only** | same |

A gate that cannot run **fails**. No gitleaks binary, no worksheet, no
scanner: the run says so and exits `1`. A check that did not run is not a
clean check — the same rule `facts/vouchers/test_no_leaks.py` follows.

## Where the name patterns come from

Patterns are **rebuilt on every run** and never stored:

- Source: the local worksheets `reports/withheld-payees-*.csv`. The main
  sheet is produced by `facts/vouchers/withheld_report.py` from the
  database (`cd facts/vouchers && VOUCHERS_DB_TRANSPORT=podman .venv/bin/python withheld_report.py`);
  the `-officials-` and `-pass2-` sheets came from the operator sessions
  recorded in `docs/session-logs/session-debrief-2026-09-16-vouchers-closeout-2.md`.
  All of them name individuals and are gitignored (`.gitignore`, the
  `reports/withheld-payees-*.csv` rule). Column shapes the runner reads:
  `payee` and `reason` (`person_shaped`, `payroll_handwrite`, `no_marker`),
  `tier` (`1_full_name`), `PROPOSED` (`PERSON`). Lines starting with `#` are
  commentary.
- Subtracted: `facts/vouchers/fixtures/payee_allowlist.txt` at `NEW` (or
  `--allowlist-ref`), matched under the package's own key — trim, collapse
  whitespace, strip trailing `. , - & ;`, casefold. Not a substring match.
- Dropped: names of six characters or fewer (a word, not an identity).
- The person list keeps only full names (a comma or a space inside), because
  a lone surname matches ordinary prose.

The generated pattern files exist only in the temporary directory for the
life of the run. `--keep-work DIR` retains the extracted blobs and the
gitleaks report for inspection; the pattern files are removed even then.

## Reading a failure

Every line of evidence is a fingerprint, a path, a commit or a rule —
**never a name, never a token**. A fingerprint is the first twelve hex
digits of sha256 over the lowercased match; to see which name it is, compute
it locally against the worksheet: `printf '%s' "name" | tr A-Z a-z | sha256sum | cut -c1-12`.

```
FAIL  person-names: 2 blob(s) with hits (1 already public)
        persons: blob=16d1393752 paths=['facts/minutes/test_parsers.py'] commits=['51dea1b'] patterns=3 fps=… NEW
        persons: blob=a31aefdb12 paths=['reports/x.md'] commits=['e9a396a'] patterns=1 fps=… ALREADY-PUBLIC
```

- `NEW` — the exact string does not occur anywhere in the tree at `OLD`.
  Pushing would disclose it for the first time. Stop; rule; scrub the range
  (a new commit on top does **not** remove it from history) or accept in
  writing.
- `ALREADY-PUBLIC` — the string already exists in `OLD`'s tree (elsewhere,
  or in the unedited part of the same file). Still a hit, because the
  question "should this be public at all" may never have been asked. With
  `--allow-preexisting` these are reported but do not fail the gate; use it
  only after that question has been answered.
- `report-only` lines under `withheld-list` are organization spellings
  (classifier class `no_marker`) found outside the voucher artifacts — a
  vendor named in a code comment, for instance. They do not fail the gate.
  Read them; they are usually the same organization printed with and
  without a corporate suffix.
- `withheld-list` hits **inside** the artifacts most often turn out to be
  that same suffix-variant class: the withheld marker-less spelling is a
  prefix of a published payee that carries `INC`/`LLC`/`CORP`. The 2026-09-16
  closeout ruled that class a test defect, not a disclosure — but the gate
  cannot tell it from a genuine withheld payee, so a human looks every time.
- A `credential-scan` or `gitleaks` failure means a key-shaped string is in
  the content or the history. Revoke it first, then scrub, then re-run.

## Files

| File | Role |
|---|---|
| `run_gates.py` | The entry point and all four gates. Importable; `tests/test_pre_push_gates.py` drives it without network or database. |
| `secret-scan.sh` | Vendored verbatim from `~/workspace/projects/ksd_forensic/scripts/secret-scan.sh` (not a git repository) on 2026-09-21; sha256 `a1185b48645e6a0fde76de2c7f0448096b669792a1386f5cd625a416abf1c374`. Read-only; prints fingerprints, never tokens. |

`gitleaks` is not vendored. The runner looks for it on `PATH`, then in
pre-commit's cache (`~/.cache/pre-commit/*/golangenv-default/bin/gitleaks`,
built the first time the `gitleaks` hook runs), then fails.

## Options

```
run_gates.py OLD..NEW
    --repo PATH            repository (default: the one containing the cwd)
    --worksheets DIR       where withheld-payees-*.csv live (default: <repo>/reports)
    --allowlist-ref REF    ref to read the allowlist from (default: NEW)
    --gitleaks PATH        gitleaks binary (default: PATH, then pre-commit cache)
    --secret-scan PATH     scanner script (default: the vendored copy)
    --keep-work DIR        keep extracted blobs and reports; pattern files are still removed
    --allow-preexisting    already-public name hits warn instead of fail
```

## History

Written after the 2026-09-20/21 promotion of `main` to GitHub, where these
checks were run by hand from session scratch files. The hand run and the
runner agree on that range: 39 commits, 215 blobs, credentials and gitleaks
clean, six person fingerprints across 16 blobs (two already public), 33
artifact blobs in the suffix-variant class.
