#!/bin/bash
# release_rename.sh -- fLLiper rename in ONE command after the freeze (RM, FL7 28.09., Variante B of the operator).
#
#   release_rename.sh <base-ref> [--branch <new-branch>] [--push]
#
# On a fresh detached worktree of <base-ref> it produces three commits and proves each one:
#   1. mechanical rename  (rename_to_flliper.py apply --weg2 --ident-map; determinism + idempotency + verify PASS)
#   2. the German identifiers the mechanical pass cannot take, WITH their pins (ident_fix.py; refuses collisions)
#   3. translation of the user-facing prose only: log messages, argparse help, markdown lines (translation memory
#      of the FL7 probe first, cachy for what is new; a unit whose German text a test/evaluator matches stays German
#      (pinned_units.py); english_audit apply + check gate PASS)
# then bytecompile, import, the rename test set (old tree vs renamed: failure set must be identical modulo renamed
# node ids), and launcher dry-runs nf/27b with the tool-converted release profiles.
# Comments and docstrings stay German here; they are translated after the release in shards (Variante B).
# Every step prints "STEP <name> <seconds>"; the script stops at the first failed proof (exit 3) and leaves the
# worktree for inspection. It never touches desk/27b-unified-0926 and never force-pushes; --push only pushes the NEW
# branch (fast-forward from nothing) with an ls-remote check before and after.
set -uo pipefail
BASE=${1:?base ref}; shift
BRANCH=""; PUSH=0
while [ $# -gt 0 ]; do case $1 in --branch) BRANCH=$2; shift 2;; --push) PUSH=1; shift;; *) echo "usage"; exit 2;; esac; done
F=${RELEASE_KIT:-$(cd "$(dirname "$0")" && pwd)}            # kit dir (tools/release of the tree): engine, maps, TM, helpers
T=$F; PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}; REPO=${REPO:-/spinning/htsglang}
# KIT_WORK: scratch space of the kit (run directories, test HOMEs, the kit lock) -- outside the tracked tree
KIT_WORK=${KIT_WORK:-/spinning/flliper/work}; export RELEASE_KIT=$F RELEASE_WORK=${RELEASE_WORK:-$KIT_WORK/release}
# Test wrapper: default = the rig's agent-test queue; the capped form of item 600 (RIG_TEST_WRAP=$F/capped_run.sh) is
# flock pytest.lock + systemd-run --scope MemoryMax=4G, no GPU. RUN_BASE moves the run directories out of the kit dir;
# KIT_TM points at a private copy of the translation memory; SKIP_DRY=1 skips the launcher dry-runs (item 600: no
# launcher process); TESTS_LIST selects a targeted test set (run_tests.sh).
TT=${RIG_TEST_WRAP:-$F/capped_run.sh}; export RIG_TEST_WRAP=$TT
# F0-A fix round 1: reviewed ident-map collisions (different functions, never one scope) are exempt by default; the name-rule exemption
# per line is chosen by the caller: COLLISION_OK_FILE=$F/data/collision_ok_1007_<27b|nf>.json (collision_auto merges it).
[ -z "${IDENT_COLLISION_OK_FILE:-}" ] && [ -f "$F/data/ident_collision_ok_1007.json" ] && export IDENT_COLLISION_OK_FILE=$F/data/ident_collision_ok_1007.json
MAP=$F/data/merged_0928.json; FIXMAP=$F/data/identfix_map.json; TM=${KIT_TM:-$F/data/tm.jsonl}
RUN=${RUN_BASE:-$KIT_WORK/runs}/run-$(date -u +%m%d%H%M%S); mkdir -p "$RUN"; W=$RUN/wt; B=$RUN/wt-base
# FIXMAP_EXTRA="<table.json> ..." (F0-A, 07.10.): further identifier tables for step 2 (ident_fix.py), merged over
# identfix_map.json, later tables win on the same key (e.g. FIXMAP_EXTRA=$F/data/ident_map_1007.json). Unset = unchanged behaviour.
if [ -n "${FIXMAP_EXTRA:-}" ]; then
  $PY - "$FIXMAP" "$RUN/identfix_merged.json" $FIXMAP_EXTRA <<'E2' || { echo "ABORT: FIXMAP_EXTRA merge"; exit 3; }
import json, sys
out = {}
for f in sys.argv[1:2] + sys.argv[3:]:
    out.update({k: v for k, v in json.load(open(f)).items() if not k.startswith("_")})
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
E2
  FIXMAP=$RUN/identfix_merged.json
fi
# the commit identity is the repo config's (efschu); never -c user.*
GIT(){ git -C "$W" "$@"; }
# kit runs are serialised (order 910): a second run waits here for the first (load-sensitive timing tests); KIT_LOCK=0 skips
[ "${KIT_LOCK:-1}" = 1 ] && { mkdir -p "$KIT_WORK"; exec 9>"$KIT_WORK/.kit_release.lock"; flock 9; }
T0=$(date +%s); START=$T0; step(){ echo "STEP $1 $(( $(date +%s) - T0 )) s"; T0=$(date +%s); }
die(){ echo "ABORT: $*"; echo "worktree left at $W"; exit 3; }
SHA=$(git -C $REPO rev-parse --verify "$BASE^{commit}") || die "unknown base $BASE"
git -C $REPO worktree add --detach "$W" "$SHA" >/dev/null 2>&1 || die "worktree"
git -C $REPO worktree add --detach "$B" "$SHA" >/dev/null 2>&1 || die "worktree base"
echo "BASE $SHA  RUN $RUN"; step worktrees

# 0. collision exemption per line, chosen automatically (order 910): the ticket label PDFLIP-A..X next to the old WEG2-*
# markers is the one cosmetic clash class; collision_auto.py exempts exactly it (full-token proof per file) from the
# unchanged base tree and writes $RUN/collision_ok.json. A COLLISION_OK_FILE set by hand is merged in. Anything outside
# that class still aborts in step 1. COLLISION_AUTO=0 switches this off.
if [ "${COLLISION_AUTO:-1}" = 1 ]; then
  $PY $F/collision_auto.py "$B" $RUN/collision_ok.json > $RUN/collision_auto.log 2>&1 || die "collision_auto: $(tail -3 $RUN/collision_auto.log)"
  export COLLISION_OK_FILE=$RUN/collision_ok.json
  echo "COLLISION-AUTO $(tail -1 $RUN/collision_auto.log)"; step collision_auto
fi

# 1. mechanical rename -------------------------------------------------------------------------------------------
$PY $T/rename_to_flliper.py apply --root "$W" --weg2 --ident-map "$MAP" --manifest $RUN/manifest.json > $RUN/apply.log 2>&1 || die "apply rc"
grep -q "ident-map collision" $RUN/apply.log && die "$(grep 'ident-map collision' $RUN/apply.log | head -2)"
TREE1=$(GIT write-tree)
$PY $T/rename_to_flliper.py apply --root "$W" --weg2 --ident-map "$MAP" > $RUN/apply2.log 2>&1
[ "$(GIT write-tree)" = "$TREE1" ] || die "second pass changed the tree (not idempotent)"
$PY $T/rename_to_flliper.py verify --base "$SHA" --root "$W" --weg2 --ident-map "$MAP" > $RUN/verify.log 2>&1
tail -1 $RUN/verify.log | grep -q PASS || die "verify: $(tail -1 $RUN/verify.log)"
GIT commit -q -m "fLLiper rename: mechanical (sglang -> flliper, weg2 -> pdflip, German identifiers via ident map)" \
  -m "rename_to_flliper.py apply --weg2 --ident-map $(basename $MAP) on $SHA; second pass no-op; verify PASS." || die commit1
C1=$(GIT rev-parse HEAD); echo "COMMIT1 $C1 $(grep -E '"(files_changed|paths_moved|replacements_total)"' $RUN/apply.log | tr -d ' \n')"
step mechanical

# 2. remaining German identifiers with their pins -------------------------------------------------------------------
$PY $F/ident_fix.py "$W" "$FIXMAP" > $RUN/identfix.log 2>&1 || die "ident_fix: $(tail -3 $RUN/identfix.log)"
# golden fixtures of the code's own output (HW-GENERIC plan fingerprint) carry marker text WEG2-...: renamed in step (item 600)
$PY $F/fixture_sync.py "$W" > $RUN/fixture.log 2>&1 || die "fixture_sync: $(tail -3 $RUN/fixture.log)"
GIT add -A && GIT commit -q -m "fLLiper rename: the German identifiers the mechanical pass cannot take, with their pins" \
  -m "ident_fix.py $(basename $FIXMAP): $(tail -1 $RUN/identfix.log); $(tail -1 $RUN/fixture.log)" || die commit2
C2=$(GIT rev-parse HEAD); echo "COMMIT2 $C2 $(tail -1 $RUN/identfix.log)"
step identifiers

# 3. user-facing prose ----------------------------------------------------------------------------------------------
mkdir -p $RUN/units
$PY $T/english_audit.py extract --repo $REPO --root "$W" --out $RUN/units --shards 1 > $RUN/extract.log 2>&1 || die extract
$PY $F/translate_shard.py $RUN/units/units-00.jsonl $RUN/uf.translated.jsonl $RUN/uf_timing.jsonl \
  Qwen3.8-27B-cachy-think log,help,doc "$TM" > $RUN/translate.log 2>&1 || die "translate: $(tail -2 $RUN/translate.log)"
echo "TRANSLATE $(tail -1 $RUN/translate.log)"
# a translation that a test/evaluator still matches in its German form stays German (pinned_units.py)
$PY $F/pinned_units.py "$W" $RUN/uf.translated.jsonl $RUN/uf.unpinned.jsonl 2> $RUN/pinned.log > $RUN/pinned.sum || die pinned
echo "PINNED $(cat $RUN/pinned.sum)"
$PY $T/english_audit.py apply --repo $REPO --root "$W" $RUN/uf.unpinned.jsonl > $RUN/tapply.log 2>&1 || die "apply units: $(tail -3 $RUN/tapply.log)"
$PY $T/english_audit.py check --repo $REPO --root "$W" --base "$C2" --must-keep $F/data/must_keep.txt \
  --units $RUN/uf.unpinned.jsonl > $RUN/gate.log 2>&1 || die "gate: $(tail -5 $RUN/gate.log)"
GIT add -A && GIT commit -q -m "fLLiper: translate the user-facing prose (log messages, argparse help, markdown) German -> English" \
  -m "english_audit extract/apply/check gate PASS; comments and docstrings follow after the release." || die commit3
C3=$(GIT rev-parse HEAD); echo "COMMIT3 $C3"
step translation

# proofs ----------------------------------------------------------------------------------------------------------
$PY $F/bytecompile.py "$W" | head -1
PYTHONPATH="$W/python" HOME=$RUN/home $TT $PY -W ignore $F/import_check.py > $RUN/import.log 2>&1; grep -E '^IMPORTS' $RUN/import.log
grep -q 'IMPORTS PASS' $RUN/import.log || die "import"
step compile_import
for side in base new; do
  wt=$B; r=0; [ $side = new ] && { wt=$W; r=1; }
  bash $F/run_tests.sh "$wt" $r > $RUN/tests_$side.log 2>&1      # serial (order 910): timing tests flake under parallel load
done
echo "TESTS base: $(tail -1 $RUN/tests_base.log)"; echo "TESTS new:  $(tail -1 $RUN/tests_new.log)"
# order 910: a run that was killed (lock wait aborted, OOM) leaves a log without pytest summary -- two empty logs compare as
# "0 new failures". A missing summary line is an abort, never a green.
for side in base new; do
  tail -5 $RUN/tests_$side.log | grep -qE '[0-9]+ (passed|failed|error)' || die "tests $side: no pytest summary line in tests_$side.log (run killed or never started)"
done
$PY - "$RUN" <<'E' || die "test comparison"
import sys
run = sys.argv[1]
def fails(p):
    out = set()
    for l in open(p, errors="ignore"):
        if l.startswith(("FAILED ", "ERROR ")):
            n = l.split(" ", 1)[1].split(" - ")[0].strip()
            out.add(n.replace("sglang", "flliper").replace("weg2", "pdflip").replace("Weg2", "PdFlip"))
    return out
b, n = fails(f"{run}/tests_base.log"), fails(f"{run}/tests_new.log")
new = sorted(x for x in n - b if not x.startswith("test::"))
print("new failures (renamed node ids ignored):", len(new)); print("\n".join(new[:20]))
open(f"{run}/new_failures.txt", "w").write("\n".join(new) + ("\n" if new else ""))
sys.exit(0)
E
# order 910: a newly red test (timing tests under load, e.g. ...prewarm_holds_the_loop_under_120_ms) is repeated ALONE,
# capped and serial (retry_failed.sh, up to RETRY_TRIES=3); only a test that stays red in every single run aborts.
NEWRED=0
while IFS= read -r nid; do
  [ -n "$nid" ] || continue
  if bash $F/retry_failed.sh "$W" "$nid" "${RETRY_TRIES:-3}" | tee -a $RUN/retry.log | tail -1 | grep -q '^RETRY-PASS'; then
    echo "FLAKE (green when repeated alone): $nid"
  else
    echo "STILL RED when repeated alone: $nid"; NEWRED=$((NEWRED+1))
  fi
done < $RUN/new_failures.txt
[ $NEWRED = 0 ] || die "new test failures ($NEWRED stay red when repeated alone)"
step tests
# Dry-run profiles per line (order 910): LINE=nf|int8|dual picks the tool-converted release profile of THAT line
# (profconv/<p>.env, made by profconv.py from docker/profiles_release); default "nf 27b" as before.
case "${LINE:-}" in nf) DRYP="nf";; nfabl) DRYP="nf-int4-h6-abl";; int8) DRYP="int8";; dual) DRYP="dual";; *) DRYP="nf 27b";; esac
lastmsg(){ grep -v '^EXIT=' "$1" | tail -1 | sed "s#$2#<tree>#g;s/WEG2/PDFLIP/g;s/Weg2/PdFlip/g;s/weg2/pdflip/g;s/SGLANG/FLLIPER/g;s/sglang/flliper/g"; }
oldprof(){ case $1 in dual) echo /spinning/gpu-arb/docker/profiles_release/27b-nvfp4-dual.env;; int8) echo /spinning/gpu-arb/docker/profiles_release/27b.env;; *) echo /spinning/gpu-arb/docker/profiles/$1.env;; esac; }
for p in $DRYP; do
  [ "${SKIP_DRY:-0}" = 1 ] && { echo "DRY $p: SKIPPED (SKIP_DRY=1)"; continue; }
  $TT bash $F/dry.sh $p "$W" $RUN/dry_${p}.log $F/profconv/$p.env; drc=$?
  if [ $drc != 4 ] && ! grep -q '^EXIT=' $RUN/dry_${p}.log 2>/dev/null; then die "dry-run $p did not finish (rc=$drc, no EXIT line in dry_${p}.log): killed or never started"; fi
  if [ $drc = 4 ]; then
    # dry.sh exit 4 = a path the profile REQUIRES (model/draft files) is not visible on this box (e.g. the agent container
    # sees only the empty mount point of a model dataset): not a rename finding, but never silent -- the operator step F0-2 runs it
    echo "DRY $p: SKIPPED -- a REQUIRED profile path is not visible on this box (operator dry-run F0-2 must cover it)"; continue
  fi
  echo "DRY $p conv: $(tail -1 $RUN/dry_${p}.log)"
  if ! tail -1 $RUN/dry_${p}.log | grep -q 'EXIT=0'; then
    # the launcher dry-run reads the LIVE box (disk, host RAM): a refusal the OLD tree with the rig profile shares is
    # the box's state, not the rename's -- named, not hidden
    $TT bash $F/dry.sh $p "$B" $RUN/dry_${p}_old.log $(oldprof $p)
    grep -q '^EXIT=' $RUN/dry_${p}_old.log 2>/dev/null || die "dry-run $p (old tree) did not finish: killed or never started"
    cn=$(grep -v '^PROFILE_ARGS' $RUN/dry_${p}.log | grep -oE '(Weg2|PdFlip)[A-Za-z]+Refused|[^0-9A-Za-z]W[0-9]{1,3} ' | tail -1 | sed 's/^Weg2\|^PdFlip//')
    co=$(grep -v '^PROFILE_ARGS' $RUN/dry_${p}_old.log | grep -oE '(Weg2|PdFlip)[A-Za-z]+Refused|[^0-9A-Za-z]W[0-9]{1,3} ' | tail -1 | sed 's/^Weg2\|^PdFlip//')
    if [ -n "$cn" ] && [ "$cn" = "$co" ]; then echo "DRY $p: BOX REFUSAL shared by the old tree ($cn) -- environment, not the rename"
    elif [ -z "$cn" ] && [ -z "$co" ] && [ "$(lastmsg $RUN/dry_${p}.log "$W")" = "$(lastmsg $RUN/dry_${p}_old.log "$B")" ]; then
      # a refusal that is not a numbered launcher refusal (missing model config, ...) with the SAME message in both generations
      echo "DRY $p: BOX REFUSAL shared by the old tree ($(lastmsg $RUN/dry_${p}.log "$W" | cut -c1-160)) -- environment, not the rename"
    else die "dry-run $p (new: ${cn:-?}, old: ${co:-?})"; fi
  fi
done
step dryruns

if [ -n "$BRANCH" ]; then
  GIT branch "$BRANCH" "$C3" || die "branch exists"
  if [ $PUSH = 1 ]; then
    before=$(git -C $REPO ls-remote origin "refs/heads/$BRANCH"); [ -z "$before" ] || die "remote branch $BRANCH exists"
    GIT -c credential.helper='!f() { echo username=efschu; echo password=$(tr -d " \t\r\n" < /root/GITHUB_PAT); }; f' \
      push -q origin "$C3:refs/heads/$BRANCH" 2>&1 | grep -v '^remote' || true
    echo "PUSHED $(git -C $REPO ls-remote origin "refs/heads/$BRANCH")"
  fi
fi
echo "DONE total $(( $(date +%s) - START )) s  base=$SHA  commits: $C1 $C2 $C3  worktree $W"
