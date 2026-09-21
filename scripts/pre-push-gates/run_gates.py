#!/usr/bin/env python3
"""Pre-push gates: check what a commit range would publish, before it leaves this machine.

    run_gates.py OLD..NEW [--repo PATH] [--worksheets DIR] [--allowlist-ref REF]
                          [--gitleaks PATH] [--secret-scan PATH] [--keep-work DIR]
                          [--allow-preexisting]

Every blob reachable from NEW but not from OLD -- every file version the
remote would receive for the first time, intermediate versions included --
is extracted to a temporary directory outside the repository and put through
four gates:

    credential-scan   secret-scan.sh over the blobs (Anthropic key shapes)
    gitleaks          gitleaks over the commit range, default ruleset, redacted
    person-names      every blob against the person-shaped full-name list
    withheld-list     voucher artifact blobs against the whole withheld list;
                      every other blob against it as a report-only layer

Name patterns are rebuilt on every run from the local, gitignored worksheets
``reports/withheld-payees-*.csv`` minus the committed allowlist, live only in
the temporary directory, and are deleted with it. Nothing is written inside
the working tree, no ref is touched, nothing is pushed. A gate that cannot
run (no scanner, no worksheet) FAILS: a check that did not run is not a
clean check.

Reports name fingerprints (sha256 of the lowercased match, first 12 hex
digits), paths and commits -- never a name and never a token.

Exit 0 when every gate passes, 1 when any gate fails, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SECRET_SCAN = HERE / "secret-scan.sh"
ALLOWLIST_PATH = "facts/vouchers/fixtures/payee_allowlist.txt"
ARTIFACT_PREFIXES = ("facts/vouchers/samples/", "exports/")
WORKSHEET_GLOB = "withheld-payees-*.csv"
# Names of six characters or fewer are dropped, as the vouchers package's own
# leak check drops them: at that length a pattern is a word, not an identity.
MIN_NAME_LEN = 7
REGENERATE_HINT = (
    "no worksheet matching reports/withheld-payees-*.csv; regenerate with "
    "`cd facts/vouchers && VOUCHERS_DB_TRANSPORT=podman .venv/bin/python withheld_report.py` "
    "(the output names individuals and is gitignored)"
)


# ------------------------------------------------------------------- git --
def git(repo: Path, *args: str, stdin: str | None = None) -> str:
    """Run one git command in ``repo`` and return its stdout as text."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def git_bytes(repo: Path, *args: str) -> bytes:
    """Run one git command in ``repo`` and return its stdout as bytes."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


# -------------------------------------------------------------- extractor --
@dataclass(frozen=True)
class Blob:
    """One file version new to the range, extracted to disk.

    Attributes:
        sha: The git blob id.
        paths: Every repository path the blob appears under, sorted.
        file: The extracted copy, named ``<sha>__<basename>``.
    """

    sha: str
    paths: tuple[str, ...]
    file: Path


def extract_blobs(repo: Path, old: str, new: str, outdir: Path) -> list[Blob]:
    """Write every blob reachable from ``new`` but not ``old`` into ``outdir``.

    This is exactly the content a push of ``new`` publishes for the first
    time: a file unchanged since ``old`` is excluded, an old blob reappearing
    under a new path is excluded, every intermediate version is included.

    Returns:
        The blobs, sorted by id. Empty when the range is empty.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    listed = git(repo, "rev-list", "--objects", f"{old}..{new}").splitlines()
    if not listed:
        return []
    paths: dict[str, set[str]] = defaultdict(set)
    shas: list[str] = []
    for line in listed:
        sha, _, path = line.partition(" ")
        shas.append(sha)
        if path:
            paths[sha].add(path)
    typed = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype)", stdin="\n".join(shas) + "\n")
    blob_shas = sorted({ln.split()[0] for ln in typed.splitlines() if ln.split()[1] == "blob"})
    blobs: list[Blob] = []
    for sha in blob_shas:
        found = tuple(sorted(paths.get(sha, ())))
        base = Path(found[0]).name if found else "noname"
        file = outdir / f"{sha}__{base}"
        file.write_bytes(git_bytes(repo, "cat-file", "blob", sha))
        blobs.append(Blob(sha=sha, paths=found, file=file))
    return blobs


# --------------------------------------------------------------- patterns --
def match_key(name: str) -> str:
    """The vouchers package's allowlist key: trim, collapse spaces, strip trailing ``. , - & ;``, casefold."""
    collapsed = " ".join(name.split())
    return re.sub(r"[.,\-&;\s]+$", "", collapsed).casefold()


def fingerprint(text: str) -> str:
    """Twelve hex digits of sha256 over the lowercased text: correlatable, never readable."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:12]


def worksheets_in(directory: Path) -> list[Path]:
    """The withheld-payee worksheets present in ``directory``, sorted."""
    return sorted(directory.glob(WORKSHEET_GLOB))


def read_worksheet(path: Path) -> list[dict]:
    """Rows of one worksheet. Lines starting with ``#`` are commentary and skipped."""
    text = "".join(
        line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith("#")
    )
    return list(csv.DictReader(io.StringIO(text)))


def is_person_row(row: dict) -> bool:
    """Whether a worksheet row describes an individual rather than an organization."""
    return (
        row.get("reason") in ("person_shaped", "payroll_handwrite")
        or row.get("tier") == "1_full_name"
        or row.get("PROPOSED") == "PERSON"
    )


@dataclass(frozen=True)
class Patterns:
    """The two pattern lists the name gates use.

    Attributes:
        all_names: Every withheld payee spelling, minus the allowlist.
        persons: The person-shaped full names among them.
    """

    all_names: frozenset[str]
    persons: frozenset[str]


def build_patterns(worksheets: Iterable[Path], allowlist_text: str) -> Patterns:
    """Rebuild the pattern lists from worksheets and the allowlist file's text.

    Both lists drop names shorter than ``MIN_NAME_LEN`` and anything the
    allowlist releases under the package's match key. The person list keeps
    only full names (a comma or a space inside), because a lone surname
    matches ordinary prose.
    """
    allow = {match_key(ln) for ln in allowlist_text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}
    all_names: set[str] = set()
    persons: set[str] = set()
    for sheet in worksheets:
        for row in read_worksheet(sheet):
            name = " ".join((row.get("payee") or "").split())
            if len(name) < MIN_NAME_LEN or match_key(name) in allow:
                continue
            all_names.add(name)
            if is_person_row(row) and ("," in name or " " in name):
                persons.add(name)
    return Patterns(all_names=frozenset(all_names), persons=frozenset(persons))


# --------------------------------------------------------------- results --
@dataclass
class GateResult:
    """One gate's verdict and the evidence behind it.

    Attributes:
        name: The gate's name as printed.
        passed: Whether the gate passed.
        summary: One line of counts.
        details: Indented evidence lines: fingerprints, paths, commits, rules. Never names, never tokens.
        hits: Structured hits, for callers and tests.
    """

    name: str
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    hits: list = field(default_factory=list)


@dataclass(frozen=True)
class NameHit:
    """A blob in which at least one pattern of a fatal layer matched."""

    blob: Blob
    layer: str
    fingerprints: tuple[str, ...]
    already_public: bool


# -------------------------------------------------------------- name gates --
def _grep_matches(patfile: Path, file: Path) -> set[str]:
    """The distinct lowercased strings in ``file`` that match a pattern line, fixed-string and case-insensitive."""
    result = subprocess.run(
        ["grep", "-o", "-h", "-a", "-i", "-F", "-f", str(patfile), str(file)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"grep failed on {file.name}: {result.stderr.strip()[:200]}")
    return {m.strip().lower() for m in result.stdout.splitlines() if m.strip()}


def _in_tree(repo: Path, ref: str, text: str, workdir: Path) -> bool:
    """Whether ``text`` already occurs anywhere in the tree at ``ref`` (case-insensitive)."""
    probe = workdir / "probe.txt"
    probe.write_text(text + "\n", encoding="utf-8")
    result = subprocess.run(
        ["git", "-C", str(repo), "grep", "-I", "-i", "-F", "-q", "-f", str(probe), ref],
        capture_output=True,
    )
    probe.unlink(missing_ok=True)
    return result.returncode == 0


def _commits_with(repo: Path, sha: str, old: str, new: str) -> list[str]:
    """Abbreviated ids of the commits in the range that carry the blob."""
    return git(repo, "log", "--format=%h", f"--find-object={sha}", f"{old}..{new}").split()


def _is_artifact(blob: Blob) -> bool:
    """Whether any of the blob's paths is a voucher artifact path."""
    return any(p.startswith(ARTIFACT_PREFIXES) for p in blob.paths)


def _write_patterns(names: Iterable[str], path: Path) -> Path:
    """Write one pattern per line, sorted, and return the path."""
    path.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    return path


def name_gates(
    repo: Path,
    old: str,
    blobs: list[Blob],
    patterns: Patterns,
    *,
    new: str | None = None,
    allow_preexisting: bool = False,
    workdir: Path | None = None,
) -> tuple[GateResult, GateResult]:
    """Run the two name gates over the extracted blobs.

    ``person-names`` matches every blob against the person list; a hit fails
    the gate. ``withheld-list`` matches voucher artifact blobs against every
    withheld spelling; a hit fails the gate. Every other blob is matched
    against the whole list too, but only reported: an organization's name in
    a code comment is worth a look, not a red light.

    A hit whose exact string already exists in the tree at ``old`` is labelled
    already public. With ``allow_preexisting`` such hits are reported but do
    not fail the gate.

    Args:
        repo: The repository.
        old: The already-published ref the range starts from.
        blobs: Output of ``extract_blobs``.
        patterns: Output of ``build_patterns``.
        new: The range's tip; when given, hits name the commits that carry them.
        allow_preexisting: Downgrade already-public hits to warnings.
        workdir: Where pattern files may be written. Defaults to a fresh
            temporary directory that is removed on return.

    Returns:
        ``(person-names result, withheld-list result)``.
    """
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="pre-push-gates-names-")) if own_tmp else workdir
    try:
        if not patterns.persons and not patterns.all_names:
            missing = "no patterns were built; refusing to call an empty check a pass"
            return (
                GateResult("person-names", False, missing),
                GateResult("withheld-list", False, missing),
            )
        pat_persons = _write_patterns(patterns.persons, work / "pat_persons.txt")
        pat_all = _write_patterns(patterns.all_names, work / "pat_all.txt")

        def scan(layer: str, patfile: Path, candidates: list[Blob]) -> tuple[list[NameHit], list[str]]:
            """Match every candidate blob against one pattern file; return hits and their evidence lines."""
            hits: list[NameHit] = []
            lines: list[str] = []
            for blob in candidates:
                matched = _grep_matches(patfile, blob.file)
                if not matched:
                    continue
                public = all(_in_tree(repo, old, m, work) for m in matched)
                fps = tuple(sorted(fingerprint(m) for m in matched))
                where = f"paths={list(blob.paths)}"
                if new:
                    where += f" commits={_commits_with(repo, blob.sha, old, new)}"
                lines.append(
                    f"{layer}: blob={blob.sha[:10]} {where} patterns={len(fps)} "
                    f"fps={' '.join(fps)} {'ALREADY-PUBLIC' if public else 'NEW'}"
                )
                hits.append(NameHit(blob=blob, layer=layer, fingerprints=fps, already_public=public))
            return hits, lines

        def verdict(name: str, hits: list[NameHit], lines: list[str], extra: list[str]) -> GateResult:
            """Fold hits into a gate result, honouring --allow-preexisting."""
            fatal = [h for h in hits if not (h.already_public and allow_preexisting)]
            public = sum(1 for h in hits if h.already_public)
            summary = f"{len(hits)} blob(s) with hits ({public} already public)"
            if allow_preexisting and hits and not fatal:
                summary += "; all already public, allowed by --allow-preexisting"
            return GateResult(name, not fatal, summary, details=lines + extra, hits=hits)

        person_hits, person_lines = scan("persons", pat_persons, blobs)
        artifacts = [b for b in blobs if _is_artifact(b)]
        others = [b for b in blobs if not _is_artifact(b)]
        withheld_hits, withheld_lines = scan("artifacts", pat_all, artifacts)
        _, report_only = scan("report-only", pat_all, others)
        report_only = [
            f"{ln}  (organization spelling in non-artifact content: reported, not fatal)" for ln in report_only
        ]
        return (
            verdict("person-names", person_hits, person_lines, []),
            verdict("withheld-list", withheld_hits, withheld_lines, report_only),
        )
    finally:
        for stale in work.glob("pat_*.txt"):
            stale.unlink(missing_ok=True)
        if own_tmp:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------- credential gates --
def run_secret_scan(script: Path, blobdir: Path) -> GateResult:
    """Run the vendored ``secret-scan.sh`` over the extracted blobs."""
    if not script.exists():
        return GateResult("credential-scan", False, f"scanner not found at {script}")
    proc = subprocess.run(["bash", str(script), str(blobdir)], capture_output=True, text=True, errors="replace")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    verdict = next(
        (ln for ln in lines if ln.startswith("RESULT:")), f"scanner exited {proc.returncode} without a RESULT line"
    )
    counts = [ln for ln in lines if ln.startswith(("pass 1:", "pass 2:"))]
    passed = proc.returncode == 0 and verdict == "RESULT: clean"
    details = counts if passed else [ln for ln in lines if not ln.startswith(("secret-scan", "targets:", "pattern:"))]
    return GateResult("credential-scan", passed, verdict.removeprefix("RESULT: "), details=details)


def find_gitleaks() -> Path | None:
    """``gitleaks`` on PATH, else the newest binary pre-commit has built into its cache."""
    on_path = shutil.which("gitleaks")
    if on_path:
        return Path(on_path)
    cache = Path.home() / ".cache" / "pre-commit"
    candidates = sorted(cache.glob("*/golangenv-default/bin/gitleaks"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_gitleaks(binary: Path | None, repo: Path, old: str, new: str, report: Path) -> GateResult:
    """Run gitleaks over the commit range with values redacted; a missing binary is a failure."""
    if binary is None or not Path(binary).exists():
        return GateResult(
            "gitleaks",
            False,
            "gitleaks binary not found (install it, or run `pre-commit run gitleaks` once to build it); "
            "a scanner that did not run is not a clean scan",
        )
    proc = subprocess.run(
        [
            str(binary),
            "git",
            f"--log-opts={old}..{new}",
            "--redact=100",
            "--no-banner",
            "--exit-code",
            "1",
            "--report-format",
            "json",
            "--report-path",
            str(report),
            str(repo),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    findings: list[dict] = []
    if report.exists() and report.stat().st_size:
        try:
            findings = json.loads(report.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            return GateResult(
                "gitleaks", False, "gitleaks wrote an unreadable report", details=[proc.stderr.strip()[-400:]]
            )
    if proc.returncode not in (0, 1):
        return GateResult("gitleaks", False, f"gitleaks exited {proc.returncode}", details=[proc.stderr.strip()[-400:]])
    passed = proc.returncode == 0 and not findings
    details = [f"{f.get('RuleID')}  {f.get('File')}  {str(f.get('Commit', ''))[:7]}" for f in findings]
    return GateResult("gitleaks", passed, f"{len(findings)} finding(s)", details=details)


# ------------------------------------------------------------- entry point --
def _parse(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("range", help="OLD..NEW, e.g. origin/main..main")
    parser.add_argument("--repo", help="repository path (default: the one containing the current directory)")
    parser.add_argument("--worksheets", help="directory holding withheld-payees-*.csv (default: <repo>/reports)")
    parser.add_argument("--allowlist-ref", help=f"ref to read {ALLOWLIST_PATH} from (default: NEW)")
    parser.add_argument("--gitleaks", help="gitleaks binary (default: PATH, then pre-commit's cache)")
    parser.add_argument("--secret-scan", default=str(DEFAULT_SECRET_SCAN), help="secret-scan.sh to run")
    parser.add_argument("--keep-work", help="keep extracted blobs and reports here (pattern files are still removed)")
    parser.add_argument(
        "--allow-preexisting", action="store_true", help="already-public name hits warn instead of fail"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every gate over the range and print one verdict line per gate.

    Returns:
        0 when every gate passed, 1 when any failed, 2 on a usage error.
    """
    args = _parse(argv)
    old_ref, sep, new_ref = args.range.partition("..")
    if not sep or not old_ref or not new_ref:
        print(f"range must be OLD..NEW, got {args.range!r}", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve() if args.repo else Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    try:
        old = git(repo, "rev-parse", "--verify", f"{old_ref}^{{commit}}").strip()
        new = git(repo, "rev-parse", "--verify", f"{new_ref}^{{commit}}").strip()
    except subprocess.CalledProcessError as exc:
        print(f"cannot resolve range in {repo}: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    work = Path(args.keep_work).resolve() if args.keep_work else Path(tempfile.mkdtemp(prefix="pre-push-gates-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        blobs = extract_blobs(repo, old, new, work / "blobs")
        commits = git(repo, "rev-list", "--count", f"{old}..{new}").strip()
        print(f"pre-push gates  repo={repo}  range={old[:7]}..{new[:7]}  commits={commits}  new blobs={len(blobs)}")

        results: list[GateResult] = [run_secret_scan(Path(args.secret_scan), work / "blobs")]
        gitleaks = Path(args.gitleaks) if args.gitleaks else find_gitleaks()
        results.append(run_gitleaks(gitleaks, repo, old, new, work / "gitleaks.json"))

        wsdir = Path(args.worksheets) if args.worksheets else repo / "reports"
        sheets = worksheets_in(wsdir) if wsdir.is_dir() else []
        if not sheets:
            results.append(GateResult("person-names", False, REGENERATE_HINT))
            results.append(GateResult("withheld-list", False, REGENERATE_HINT))
        else:
            try:
                allowlist_text = git(repo, "show", f"{args.allowlist_ref or new}:{ALLOWLIST_PATH}")
                allow_note: list[str] = []
            except subprocess.CalledProcessError:
                allowlist_text = ""
                allow_note = [
                    f"note: {ALLOWLIST_PATH} not found at the ref; treating the allowlist as empty (stricter)"
                ]
            patterns = build_patterns(sheets, allowlist_text)
            persons, withheld = name_gates(
                repo, old, blobs, patterns, new=new, allow_preexisting=args.allow_preexisting, workdir=work
            )
            source = (
                f"patterns: persons={len(patterns.persons)} all={len(patterns.all_names)} "
                f"from {len(sheets)} worksheet(s)"
            )
            persons.details = [source, *allow_note, *persons.details]
            results.extend([persons, withheld])

        for result in results:
            print(f"{'PASS' if result.passed else 'FAIL'}  {result.name}: {result.summary}")
            for line in result.details:
                print(f"        {line}")
        overall = all(r.passed for r in results)
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
        return 0 if overall else 1
    finally:
        for stale in work.glob("pat_*.txt"):
            stale.unlink(missing_ok=True)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
