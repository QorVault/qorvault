#!/usr/bin/env bash
# secret-scan.sh — locate Anthropic-style credentials on this workstation.
#
# READ-ONLY. Prints locations, token types and counts. NEVER prints a token
# value. The only per-token identifier shown is sha256(token)[0:12], so one key
# can be correlated across files without any of its bytes appearing. To match a
# key you hold:   printf '%s' "$KEY" | sha256sum | cut -c1-12
#
# Usage:  secret-scan.sh [--skip-archives] [DIR ...]
#         default DIRs: ~/workspace ~/.claude ~/qorvault-dev-archive ~/backups
# Exit:   0 = nothing found   1 = credentials found   2 = usage error
#
# Pass 1  plain files — grep -rlaE. The -a is load-bearing: this box's grep is
#         ugrep, which skips binary files (SQLite, WAL, browser cache) SILENTLY
#         (no stderr) without it.
# Pass 2  archives — .tar[.gz|.xz|.bz2|.zst], .tgz, .zip and bare .gz are
#         stream-decompressed in memory by python3 stdlib; nothing touches disk.
#         grep cannot see inside these, and `ugrep -z` was observed to miss them.
#
# Token types (length floor of 20 drops doc placeholders like sk-ant-your-key):
#   api = API key   oat/ort = Claude Code OAuth   sid/rh = claude.ai cookies
#
# Known gaps: nested archives, encrypted zips, git object stores (.git/objects
# is zlib-compressed — use gitleaks/trufflehog for history), anything outside
# the DIRs given (e.g. ~/.bash_history, ~/*.yaml at the home root).

set -uo pipefail

PATTERN='sk-ant-(api|oat|ort|sid|rh)[0-9]*-[A-Za-z0-9_-]{20,}'

skip_archives=0
dirs=()
for a in "$@"; do
  case "$a" in
    --skip-archives) skip_archives=1 ;;
    -h|--help)       sed -n '2,25p' "$0"; exit 2 ;;
    -*)              echo "unknown option: $a" >&2; exit 2 ;;
    *)               dirs+=("$a") ;;
  esac
done
[ ${#dirs[@]} -eq 0 ] && dirs=("$HOME/workspace" "$HOME/.claude" "$HOME/qorvault-dev-archive" "$HOME/backups")

targets=()
for d in "${dirs[@]}"; do
  if [ -d "$d" ]; then targets+=("$d"); else echo "skip (not a directory): $d" >&2; fi
done
[ ${#targets[@]} -eq 0 ] && { echo "nothing to scan" >&2; exit 2; }

ttype() { sed -E 's/^sk-ant-([a-z]+)[0-9]*-.*/\1/'; }   # stdin: token -> api|oat|ort|sid|rh

echo "secret-scan  $(date -Iseconds)"
echo "targets:     ${targets[*]}"
echo "pattern:     $PATTERN"
echo

# ---------------------------------------------------------------- pass 1
echo "== PASS 1: plain files =="
files=$(grep -rlaE "$PATTERN" "${targets[@]}" 2>/dev/null | sort)
nfiles=$(printf '%s\n' "$files" | grep -c .)

if [ "$nfiles" -gt 0 ]; then
  echo "-- by directory (files) --"
  printf '%s\n' "$files" | xargs -d '\n' -n1 dirname | sort | uniq -c | sort -rn
  echo
  byfile=(); fps=()
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    toks=$(grep -aoE "$PATTERN" "$f" 2>/dev/null | sort -u)
    n=$(printf '%s\n' "$toks" | grep -c .)
    types=$(printf '%s\n' "$toks" | ttype | sort -u | paste -sd, -)
    byfile+=("$(printf '%4d  %-12s %s' "$n" "$types" "$f")")
    while IFS= read -r tok; do
      [ -z "$tok" ] && continue
      fps+=("$(printf '%s' "$tok" | sha256sum | cut -c1-12) $(printf '%s' "$tok" | ttype)")
    done <<< "$toks"
  done <<< "$files"
  echo "-- by file: distinct tokens, types, path --"
  printf '%s\n' "${byfile[@]}" | sort -rn
  echo
  echo "-- distinct tokens: fingerprint, type, files containing it --"
  printf '%s\n' "${fps[@]}" | sort | uniq -c | awk '{printf "  %s  %-4s %d file(s)\n", $2, $3, $1}' | sort -k2,2 -k3,3rn
fi
echo
echo "pass 1: $nfiles file(s) with credentials"
echo

# ---------------------------------------------------------------- pass 2
arch_rc=0
if [ "$skip_archives" -eq 0 ]; then
  echo "== PASS 2: archives (stream-decompressed in memory; nothing written) =="
  python3 - "$PATTERN" "${targets[@]}" <<'PY'
import sys, os, re, tarfile, zipfile, gzip, hashlib, shutil, subprocess
pat = re.compile(sys.argv[1].encode()); roots = sys.argv[2:]
TAR = ('.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz', '.tar.bz2', '.tbz2', '.tar.zst', '.tzst')
fp = lambda t: hashlib.sha256(t).hexdigest()[:12]
kind = lambda t: re.match(rb'sk-ant-([a-z]+)', t).group(1).decode()

def scan(fh, chunk=1 << 20, overlap=512):
    """Regex over a stream in bounded memory; a match cut off at a chunk edge is re-seen via overlap."""
    toks, tail = set(), b''
    while True:
        buf = fh.read(chunk); last = not buf; data = tail + buf
        for m in pat.finditer(data):
            if m.end() == len(data) and not last: continue
            toks.add(m.group(0))
        if last: return toks
        tail = data[-overlap:]

def report(path, member, toks):
    hits.add(path)
    print(f"  {path}\n      {member}  ->  {len(toks)} token(s) [{','.join(sorted({kind(t) for t in toks}))}]  {' '.join(sorted(fp(t) for t in toks))}")

def open_tar(path):
    if path.endswith(('.tar.zst', '.tzst')):
        if not shutil.which('zstd'): raise RuntimeError('zstd not installed')
        p = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return tarfile.open(fileobj=p.stdout, mode='r|')
    return tarfile.open(path, 'r:*')

hits, skipped, scanned = set(), [], 0
for root in roots:
    for dp, _, fn in os.walk(root, onerror=lambda e: None):
        for name in fn:
            path = os.path.join(dp, name); low = name.lower()
            if os.path.islink(path): continue
            try:
                if low.endswith(TAR):
                    scanned += 1
                    with open_tar(path) as tf:
                        for m in tf:
                            if not m.isfile(): continue
                            fh = tf.extractfile(m)
                            if fh is None: continue
                            t = scan(fh)
                            if t: report(path, m.name, t)
                elif low.endswith('.zip'):
                    scanned += 1
                    with zipfile.ZipFile(path) as zf:
                        for m in zf.infolist():
                            if m.is_dir(): continue
                            with zf.open(m) as fh:
                                t = scan(fh)
                            if t: report(path, m.filename, t)
                elif low.endswith('.gz'):
                    scanned += 1
                    with gzip.open(path, 'rb') as fh:
                        t = scan(fh)
                    if t: report(path, '(gzip payload)', t)
            except Exception as e:
                skipped.append((path, f"{type(e).__name__}: {e}"[:80]))

print(f"\npass 2: {scanned} archive(s) scanned, {len(hits)} with credentials, {len(skipped)} unreadable")
for p, why in skipped[:15]: print(f"  unreadable: {p}  ({why})")
if len(skipped) > 15: print(f"  ... and {len(skipped) - 15} more")
sys.exit(1 if hits else 0)
PY
  arch_rc=$?
fi

echo
if [ "$nfiles" -gt 0 ] || [ "$arch_rc" -eq 1 ]; then echo "RESULT: credentials present"; exit 1; fi
echo "RESULT: clean"; exit 0
