"""Synthetic tests for scripts/pre-push-gates. No network, no database, no real names.

Every name in this file is invented. Every key-shaped string is assembled at
runtime from pieces so that no test file ever contains a token-shaped
literal, which would trip the repo's own gitleaks hook.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATES_DIR = REPO_ROOT / "scripts" / "pre-push-gates"
RUNNER = GATES_DIR / "run_gates.py"
SECRET_SCAN = GATES_DIR / "secret-scan.sh"
GIT = shutil.which("git") or "/usr/bin/git"

# Invented. Shapes matter, identities do not.
PERSON_COMMA = "Quixote, Zebulon"
PERSON_SPACE = "Quibble Merryweather"
OFFICIAL_FULL = "Ottoline Vossberg"
OFFICIAL_SURNAME_ORG = "VOSSBERG PLUMBING"
ORG_ALLOWLISTED_DOTTED = "ACME WIDGET CO."
ORG_WITHHELD = "BIG ORG SUPPLY"
PAYROLL = "PAYROLL PERSONNAME"
SHORT = "Tiny"


def load_gates():
    """Import the runner directly from its repo-local path."""
    spec = importlib.util.spec_from_file_location("run_gates", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Dataclasses resolve postponed annotations through sys.modules, so a
    # path-loaded module must be registered before it executes.
    sys.modules["run_gates"] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    """Run git on a throwaway repo with signing off and a fixed identity."""
    result = subprocess.run(  # noqa: S603 - test drives git on a throwaway repo under tmp_path.
        [
            GIT,
            "-C",
            str(repo),
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def blob_sha(content: str) -> str:
    """The git blob id of ``content``, to assert on exactly which blobs came back."""
    data = content.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git object id, not a security hash.


def fake_token() -> str:
    """A string shaped like an Anthropic key, built at runtime, never written as a literal."""
    return "sk-" + "ant-" + "api03-" + "x" * 30


def write(path: Path, text: str) -> None:
    """Write text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- fixtures --
@pytest.fixture
def two_commit_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo with two commits: one changed file, one unchanged, one old blob under a new path, one new file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo / "a.txt", "alpha\n")
    write(repo / "b.txt", "beta\n")
    write(repo / "facts/vouchers/samples/s.md", "sample one\n")
    write(repo / "facts/vouchers/fixtures/payee_allowlist.txt", "# allowlist\nACME WIDGET CO\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    old = git(repo, "rev-parse", "HEAD")
    write(repo / "a.txt", "alpha two\n")
    write(repo / "c.txt", "beta\n")  # same blob as b.txt: already reachable from old
    write(repo / "d.txt", "delta\n")
    write(repo / "facts/vouchers/samples/s.md", "sample two\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "two")
    new = git(repo, "rev-parse", "HEAD")
    return repo, old, new


@pytest.fixture
def worksheets(tmp_path: Path) -> Path:
    """Three synthetic worksheets in the shapes the real ones have."""
    wsdir = tmp_path / "reports"
    wsdir.mkdir()
    write(
        wsdir / "withheld-payees-2026-09-16.csv",
        "payee,line_count,first_cycle,last_cycle,reason,flag\n"
        f'"{PERSON_COMMA}",3,2020-01-01,2020-02-01,person_shaped,\n'
        f"{ORG_ALLOWLISTED_DOTTED},9,2020-01-01,2020-02-01,no_marker,all_caps\n"
        f"{ORG_WITHHELD},9,2020-01-01,2020-02-01,no_marker,all_caps\n"
        f"{SHORT},1,2020-01-01,2020-02-01,no_marker,\n"
        f"{PAYROLL},2,2020-01-01,2020-02-01,payroll_handwrite,\n",
    )
    write(
        wsdir / "withheld-payees-officials-2026-09-18.csv",
        "surname,tier,payee,line_count,first_cycle,last_cycle,reason,flag\n"
        f"Vossberg,1_full_name,{OFFICIAL_FULL},2,2020-01-01,2020-02-01,no_marker,\n"
        f"Vossberg,2_surname_only,{OFFICIAL_SURNAME_ORG},2,2020-01-01,2020-02-01,no_marker,all_caps\n",
    )
    write(
        wsdir / "withheld-payees-pass2-2026-09-18.csv",
        "# RULE SET (comment lines precede the header)\n"
        "# more commentary\n"
        "rank,payee,line_count,first_cycle,last_cycle,reason,flag,PROPOSED,rule,shape\n"
        f"1,{PERSON_SPACE},40,2020-01-01,2020-02-01,no_marker,,PERSON,R1,TT\n"
        f"2,{ORG_WITHHELD},40,2020-01-01,2020-02-01,no_marker,all_caps,ORG,R3b,CCC\n",
    )
    return wsdir


@pytest.fixture
def fake_gitleaks_factory(tmp_path: Path):
    """Build a stand-in gitleaks executable that writes a chosen report and exit code."""

    def make(findings: list[dict], exit_code: int) -> Path:
        """Write the stand-in script and make it executable."""
        script = tmp_path / f"gitleaks-{exit_code}-{len(findings)}"
        body = (
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "args = sys.argv[1:]\n"
            "path = args[args.index('--report-path') + 1]\n"
            f"json.dump({json.dumps(findings)}, open(path, 'w'))\n"
            "print('fake gitleaks ran')\n"
            f"sys.exit({exit_code})\n"
        )
        script.write_text(body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return script

    return make


# --------------------------------------------------------------- extractor --
class TestExtractBlobs:
    """The extractor returns exactly what a push would publish for the first time."""

    def test_returns_exactly_the_blobs_new_to_the_range(self, two_commit_repo, tmp_path):
        """Changed and new files are in; unchanged files and old blobs under new paths are out."""
        gates = load_gates()
        repo, old, new = two_commit_repo
        outdir = tmp_path / "blobs"

        blobs = gates.extract_blobs(repo, old, new, outdir)

        assert {b.sha for b in blobs} == {blob_sha("alpha two\n"), blob_sha("delta\n"), blob_sha("sample two\n")}
        assert blob_sha("beta\n") not in {b.sha for b in blobs}, "an old blob under a new path is already public"
        assert blob_sha("alpha\n") not in {b.sha for b in blobs}

    def test_records_paths_and_writes_content(self, two_commit_repo, tmp_path):
        """Each blob carries its repo paths and is written under its sha and basename."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        blobs = {b.sha: b for b in gates.extract_blobs(repo, old, new, tmp_path / "blobs")}

        sample = blobs[blob_sha("sample two\n")]
        assert sample.paths == ("facts/vouchers/samples/s.md",)
        assert sample.file.read_text() == "sample two\n"
        assert sample.file.name.endswith("__s.md")
        assert sample.file.parent == tmp_path / "blobs"

    def test_empty_range_yields_no_blobs(self, two_commit_repo, tmp_path):
        """An empty range extracts nothing."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        assert gates.extract_blobs(repo, new, new, tmp_path / "blobs") == []


# ---------------------------------------------------------------- patterns --
class TestPatterns:
    """Pattern lists are rebuilt from the worksheets and the allowlist, by the package's rules."""

    def test_match_key_matches_the_allowlist_contract(self):
        """The key trims, collapses, strips trailing punctuation and casefolds."""
        gates = load_gates()
        assert gates.match_key("  ACME   WIDGET CO. ") == "acme widget co"
        assert gates.match_key("Name & ;") == "name"

    def test_builds_both_lists_from_the_three_worksheets(self, worksheets):
        """Allowlisted and short names are dropped; the person subset follows the row classes."""
        gates = load_gates()
        allowlist = "# comment\nACME WIDGET CO\n"

        patterns = gates.build_patterns(gates.worksheets_in(worksheets), allowlist)

        assert patterns.all_names == frozenset(
            {PERSON_COMMA, ORG_WITHHELD, PAYROLL, OFFICIAL_FULL, OFFICIAL_SURNAME_ORG, PERSON_SPACE}
        )
        assert ORG_ALLOWLISTED_DOTTED not in patterns.all_names, "allowlisted under the match key despite the dot"
        assert SHORT not in patterns.all_names, "length floor"
        assert patterns.persons == frozenset({PERSON_COMMA, PAYROLL, OFFICIAL_FULL, PERSON_SPACE})

    def test_worksheets_in_finds_only_withheld_payee_sheets(self, worksheets):
        """Only withheld-payees-*.csv counts as a worksheet."""
        gates = load_gates()
        write(worksheets / "unrelated.csv", "a,b\n1,2\n")

        found = sorted(p.name for p in gates.worksheets_in(worksheets))

        assert found == [
            "withheld-payees-2026-09-16.csv",
            "withheld-payees-officials-2026-09-18.csv",
            "withheld-payees-pass2-2026-09-18.csv",
        ]

    def test_fingerprint_is_case_insensitive_and_short(self):
        """Fingerprints are twelve hex digits of sha256 over the lowercased text."""
        gates = load_gates()
        assert gates.fingerprint("Quixote, Zebulon") == gates.fingerprint("quixote, zebulon")
        assert gates.fingerprint("x") == hashlib.sha256(b"x").hexdigest()[:12]


# --------------------------------------------------------------- name gates --
def _blob(gates, tmp_path: Path, rel_path: str, text: str):
    """A Blob object over a file written under tmp_path/blobs, named the way the extractor names them."""
    sha = blob_sha(text)
    file = tmp_path / "blobs" / f"{sha}__{Path(rel_path).name}"
    write(file, text)
    return gates.Blob(sha=sha, paths=(rel_path,), file=file)


def _patterns(gates, worksheets):
    """Patterns built from the synthetic worksheets with one allowlisted organization."""
    return gates.build_patterns(gates.worksheets_in(worksheets), "ACME WIDGET CO\n")


class TestNameGates:
    """The person and withheld gates flag by fingerprint and never print a name."""

    def test_clean_blobs_pass_both_gates(self, two_commit_repo, worksheets, tmp_path):
        """Content matching nothing passes both gates with no hits."""
        gates = load_gates()
        repo, old, _ = two_commit_repo
        blobs = [_blob(gates, tmp_path, "src/code.py", "x = 1  # nothing to see\n")]

        persons, withheld = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets))

        assert persons.passed and withheld.passed
        assert persons.hits == [] and withheld.hits == []

    def test_planted_person_name_fails_the_person_gate_by_fingerprint_only(self, two_commit_repo, worksheets, tmp_path):
        """A person name in code fails the gate; the report carries its fingerprint and path, not the name."""
        gates = load_gates()
        repo, old, _ = two_commit_repo
        blobs = [_blob(gates, tmp_path, "src/code.py", f"# reviewed with {PERSON_COMMA} on Tuesday\n")]

        persons, _ = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets))

        assert not persons.passed
        assert len(persons.hits) == 1
        hit = persons.hits[0]
        assert hit.fingerprints == (gates.fingerprint(PERSON_COMMA),)
        assert hit.already_public is False
        rendered = "\n".join([persons.summary, *persons.details])
        assert "src/code.py" in rendered
        assert PERSON_COMMA.lower() not in rendered.lower(), "a name must never reach the log"

    def test_withheld_org_in_an_artifact_fails_the_withheld_gate(self, two_commit_repo, worksheets, tmp_path):
        """A withheld spelling inside a voucher artifact is fatal."""
        gates = load_gates()
        repo, old, _ = two_commit_repo
        blobs = [
            _blob(gates, tmp_path, "facts/vouchers/samples/2031-01-14-GF.md", f"| 1 | **{ORG_WITHHELD}** | 10.00 |\n")
        ]

        persons, withheld = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets))

        assert persons.passed
        assert not withheld.passed
        assert withheld.hits[0].fingerprints == (gates.fingerprint(ORG_WITHHELD),)

    def test_withheld_org_in_code_is_report_only(self, two_commit_repo, worksheets, tmp_path):
        """The same spelling outside the artifacts is reported but not fatal."""
        gates = load_gates()
        repo, old, _ = two_commit_repo
        blobs = [_blob(gates, tmp_path, "src/code.py", f"# vendor example: {ORG_WITHHELD}\n")]

        _, withheld = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets))

        assert withheld.passed, "an organization spelling in code is reported, not failed"
        assert withheld.hits == []
        assert any(gates.fingerprint(ORG_WITHHELD) in line for line in withheld.details)

    def test_already_public_hit_is_labelled_and_can_be_allowed(self, tmp_path, worksheets):
        """A string already in OLD's tree is labelled already-public and can be downgraded to a warning."""
        gates = load_gates()
        repo = tmp_path / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        write(repo / "notes.md", f"Attendees: {PERSON_SPACE}.\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "public already")
        old = git(repo, "rev-parse", "HEAD")
        blobs = [_blob(gates, tmp_path, "src/roster.py", f'ROSTER = ["{PERSON_SPACE}"]\n')]

        strict, _ = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets))
        lenient, _ = gates.name_gates(repo, old, blobs, _patterns(gates, worksheets), allow_preexisting=True)

        assert strict.hits[0].already_public is True
        assert not strict.passed
        assert lenient.passed
        assert lenient.hits[0].already_public is True, "still reported, just not fatal"


# ------------------------------------------------------------ credential gate --
class TestSecretScan:
    """The vendored credential scanner is present and behaves."""

    def test_vendored_script_is_present_and_executable(self):
        """The scanner ships with the runner and is executable."""
        assert SECRET_SCAN.exists()
        assert os.access(SECRET_SCAN, os.X_OK)

    def test_clean_directory_passes(self, tmp_path):
        """A directory without key shapes is clean."""
        gates = load_gates()
        blobdir = tmp_path / "blobs"
        write(blobdir / "abc__code.py", "print('hello')\n")

        result = gates.run_secret_scan(SECRET_SCAN, blobdir)

        assert result.passed
        assert "clean" in result.summary

    def test_key_shaped_string_fails_without_echoing_it(self, tmp_path):
        """A key-shaped string fails the gate and never appears in the report."""
        gates = load_gates()
        blobdir = tmp_path / "blobs"
        token = fake_token()
        write(blobdir / "abc__config.py", f'KEY = "{token}"\n')

        result = gates.run_secret_scan(SECRET_SCAN, blobdir)

        assert not result.passed
        assert token not in "\n".join([result.summary, *result.details])


# --------------------------------------------------------------- gitleaks --
class TestGitleaks:
    """gitleaks is run over the range and its absence is a failure."""

    def test_missing_binary_is_a_failure_not_a_skip(self, two_commit_repo, tmp_path):
        """No binary means the gate fails with a reason, not a skip."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        result = gates.run_gitleaks(None, repo, old, new, tmp_path / "report.json")

        assert not result.passed
        assert "not found" in result.summary

    def test_clean_report_passes(self, two_commit_repo, tmp_path, fake_gitleaks_factory):
        """An empty report with exit 0 passes."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        result = gates.run_gitleaks(fake_gitleaks_factory([], 0), repo, old, new, tmp_path / "report.json")

        assert result.passed

    def test_findings_fail_with_rule_file_and_commit(self, two_commit_repo, tmp_path, fake_gitleaks_factory):
        """Findings fail the gate and are reported by rule, file and commit."""
        gates = load_gates()
        repo, old, new = two_commit_repo
        finding = {"RuleID": "generic-api-key", "File": "cfg.py", "Commit": "abcdef0123456789", "Secret": "REDACTED"}

        result = gates.run_gitleaks(fake_gitleaks_factory([finding], 1), repo, old, new, tmp_path / "report.json")

        assert not result.passed
        assert any("generic-api-key" in d and "cfg.py" in d and "abcdef0" in d for d in result.details)


# -------------------------------------------------------------- entry point --
def _run_main(gates, repo, old, new, worksheets, gitleaks, extra=()):
    """Invoke the entry point in-process with the test's repo, worksheets and gitleaks stand-in."""
    return gates.main(
        [
            f"{old}..{new}",
            "--repo",
            str(repo),
            "--worksheets",
            str(worksheets),
            "--gitleaks",
            str(gitleaks),
            *extra,
        ]
    )


class TestEntryPoint:
    """One command runs every gate and exits non-zero when any fails."""

    def test_clean_range_exits_zero_and_reports_every_gate(
        self, two_commit_repo, worksheets, fake_gitleaks_factory, capsys
    ):
        """A clean range prints PASS for all four gates and exits 0."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        code = _run_main(gates, repo, old, new, worksheets, fake_gitleaks_factory([], 0))

        out = capsys.readouterr().out
        assert code == 0
        for gate in ("credential-scan", "gitleaks", "person-names", "withheld-list"):
            assert f"PASS  {gate}" in out, out
        assert "OVERALL: PASS" in out

    def test_gitleaks_finding_makes_the_run_fail(self, two_commit_repo, worksheets, fake_gitleaks_factory, capsys):
        """A gitleaks finding fails the run."""
        gates = load_gates()
        repo, old, new = two_commit_repo
        leaky = fake_gitleaks_factory([{"RuleID": "r", "File": "f", "Commit": "c" * 40}], 1)

        code = _run_main(gates, repo, old, new, worksheets, leaky)

        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  gitleaks" in out
        assert "OVERALL: FAIL" in out

    def test_planted_person_name_in_the_range_makes_the_run_fail(
        self, two_commit_repo, worksheets, fake_gitleaks_factory, capsys
    ):
        """A person name committed in the range fails the run without being printed."""
        gates = load_gates()
        repo, old, _ = two_commit_repo
        write(repo / "src/roster.py", f"# {PERSON_COMMA}\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "three")
        new = git(repo, "rev-parse", "HEAD")

        code = _run_main(gates, repo, old, new, worksheets, fake_gitleaks_factory([], 0))

        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  person-names" in out
        assert PERSON_COMMA.lower() not in out.lower()

    def test_missing_worksheets_fail_the_name_gates(self, two_commit_repo, tmp_path, fake_gitleaks_factory, capsys):
        """No worksheet means both name gates fail and the output says how to regenerate one."""
        gates = load_gates()
        repo, old, new = two_commit_repo
        empty = tmp_path / "no-reports"
        empty.mkdir()

        code = _run_main(gates, repo, old, new, empty, fake_gitleaks_factory([], 0))

        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  person-names" in out and "FAIL  withheld-list" in out
        assert "withheld_report.py" in out, "the failure must say how to regenerate the source"

    def test_missing_gitleaks_binary_fails_the_run(self, two_commit_repo, worksheets, tmp_path):
        """A missing gitleaks binary fails the run."""
        gates = load_gates()
        repo, old, new = two_commit_repo

        code = _run_main(gates, repo, old, new, worksheets, tmp_path / "no-such-gitleaks")

        assert code == 1

    def test_runner_leaves_the_repo_untouched(self, two_commit_repo, worksheets, fake_gitleaks_factory):
        """The run changes neither the working tree nor HEAD, and writes no pattern file inside it."""
        gates = load_gates()
        repo, old, new = two_commit_repo
        before = git(repo, "status", "--porcelain"), git(repo, "rev-parse", "HEAD")

        _run_main(gates, repo, old, new, worksheets, fake_gitleaks_factory([], 0))

        assert (git(repo, "status", "--porcelain"), git(repo, "rev-parse", "HEAD")) == before
        assert not list(repo.rglob("pat_*.txt")), "pattern files never land inside the tree"

    def test_cli_is_runnable_as_a_script(self, two_commit_repo, worksheets, fake_gitleaks_factory):
        """The runner works as a command, not only as an import."""
        repo, old, new = two_commit_repo
        result = subprocess.run(  # noqa: S603 - test runs this repo-local script with synthetic tmp_path fixtures.
            [
                sys.executable,
                str(RUNNER),
                f"{old}..{new}",
                "--repo",
                str(repo),
                "--worksheets",
                str(worksheets),
                "--gitleaks",
                str(fake_gitleaks_factory([], 0)),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OVERALL: PASS" in result.stdout
