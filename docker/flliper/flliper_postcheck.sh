#!/bin/bash
# flliper_postcheck.sh -- checks of a BUILT flat fLLiper image, run by the operator on the build host after host_build.sh
# (FLAT_IMAGE_PLAN.md section 5). No GPU: every container runs with --network none and without --gpus.
#
#   flliper_postcheck.sh <image> [--plan] [--ctx <ctx dir>] [--models <models-cache dir>] [--only "C1 C4 ..."]
#
#   --plan    print the checks and the docker commands, run nothing (no docker call at all)
#   --ctx     the ctx the image was built from: its BUILD_INFO.json revisions must equal the image labels (C1)
#   --models  also run `dryrun` for the accepted profiles with the models mounted read-only (C10); without it C10 is SKIP
#   --only    run only the named checks
#   env DOCKER (default docker) -- the tests put a stub here; PROFILES_CHECK (default: the image's accepted profiles)
#
# Checks (one line each: "CHECK <id> PASS|FAIL|WARN|SKIP <detail>"):
#   C1  labels: title fLLiper, revision = revision.27b (40 hex), revision.nf 40 hex, build=flat, one_tree 0|1 agrees
#       with the two revisions, placeholders=none (WARN otherwise); with --ctx equal to BUILD_INFO.json lines
#   C2  config: HEALTHCHECK, STOPSIGNAL SIGTERM, ENTRYPOINT /opt/htsglang/entrypoint.sh, <= 70 layers
#   C3  slot trees: /opt/htsglang/src-<slot> HEAD = label revision.<slot>, clean (git status empty)
#   C4  D1 selfcheck per slot: `MODE=weg2 HTSGLANG_PROFILE=<slot> <image> selfcheck` rc 0
#   C5  JIT reports: every /opt/htsglang/JIT_PREBUILD-*.json verdict OK, 0 problems
#   C6  import smoke per slot (PYTHONPATH of the slot): package, launcher, front, engine, kernel_dist_guard, sgl_kernel,
#       flashinfer 0.7.0, cutlass, tvm_ffi, triton, torch 2.11; no sglang/flliper importable from site-packages
#   C7  profiles parse: every /opt/htsglang/profiles/*.env sources in a subshell (_form stub), status in
#       abgenommen|experimentell|geplant, PROFILE_MODEL set; the accepted ones (BUILD_INFO.accepted_first_docker) abgenommen
#   C8  release defaults (Leistungsschalter, 27B): the qwen27b registry row of the 27b slot turns its proven switches on
#       (front_exact_tokens, p_row_authority, d_hostgap_levers, d_hostgap_base, d_release_fixes) -- FAIL if a field is
#       missing (image built from a tree before ad8b383b08) or off
#   C9  pip check: findings counted (WARN; the allow-list is frozen from the first image, FLAT_IMAGE_PLAN 5.4)
#   C10 dryrun of the accepted profiles with --models (launcher argv without a boot)
#   C11 deadman in the image (30.09., DEADMAN-IM-IMAGE): /opt/htsglang/devtools/boot_deadman.sh serves BOTH slots; FAIL if
#       it calls `state_file.py progress-check` and a slot's state_file.py has none (or the reverse: a tree with it and
#       a deadman without = the host copy baked in), if a slot has note_rank_death without "rank" in WRITERS, or if its
#       md5 differs from BUILD_INFO.json .deadman.md5 (make_flat_ctx 0930dm). An old deadman with old trees = WARN;
#       env FLAT_REQUIRE_DEADMAN=1 makes every missing part a FAIL (release/freeze image).
# Exit: 0 no FAIL, 1 at least one FAIL, 2 usage.
set -uo pipefail

IMG=""; PLAN=0; CTX=""; MODELS=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) PLAN=1; shift ;;
    --ctx) CTX=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [ -z "$IMG" ] || { echo "one image only" >&2; exit 2; }; IMG=$1; shift ;;
  esac
done
[ -n "$IMG" ] || { echo "usage: flliper_postcheck.sh <image> [--plan] [--ctx dir] [--models dir] [--only ids]" >&2; exit 2; }
DOCKER=${DOCKER:-docker}
RUN=("$DOCKER" run --rm --network none)
NFAIL=0; NWARN=0
res(){ printf 'CHECK %s %s %s\n' "$1" "$2" "$3"; case "$2" in FAIL) NFAIL=$((NFAIL + 1)) ;; WARN) NWARN=$((NWARN + 1)) ;; esac; }
want(){ [ -z "$ONLY" ] && return 0; case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
say(){ printf '%s\n' "$*"; }

if [ "$PLAN" = 1 ]; then
  say "flliper_postcheck.sh --plan $IMG (nothing runs)"
  say "  C1/C2  $DOCKER image inspect $IMG"
  say "  C3     ${RUN[*]} --entrypoint bash $IMG -c '<git rev-parse/status per slot>'"
  say "  C4     ${RUN[*]} -e MODE=weg2 -e HTSGLANG_PROFILE=27b $IMG selfcheck   (and nf)"
  say "  C5     ${RUN[*]} --entrypoint /opt/venv/bin/python $IMG -c '<JIT_PREBUILD-*.json verdicts>'"
  say "  C6     ${RUN[*]} --entrypoint /opt/venv/bin/python -e PYTHONPATH=/opt/htsglang/src-<slot>/python $IMG -c '<imports>'"
  say "  C7     ${RUN[*]} --entrypoint bash $IMG -c '<source every profile with a _form stub>'"
  say "  C8     ${RUN[*]} --entrypoint /opt/venv/bin/python -e PYTHONPATH=/opt/htsglang/src-27b/python $IMG -c '<registry row>'"
  say "  C9     ${RUN[*]} --entrypoint /opt/venv/bin/python $IMG -m pip check"
  say "  C10    ${RUN[*]} -v <models>:/spinning/llm_stuff/club-3090/models-cache:ro -e MODE=weg2 -e HTSGLANG_PROFILE=<p> $IMG dryrun"
  say "  C11    ${RUN[*]} --entrypoint /opt/venv/bin/python $IMG -c '<boot_deadman.sh md5/progress-check vs each slot's state_file.py, BUILD_INFO deadman.md5>'"
  exit 0
fi

# ---- C1 / C2: image metadata (one inspect) ------------------------------------------------------------------------
META=$("$DOCKER" image inspect "$IMG" 2>/dev/null) || { res C1 FAIL "image $IMG not found ($DOCKER image inspect)"; exit 1; }
LAB=$(printf '%s' "$META" | python3 -c '
import json, sys
d = json.load(sys.stdin); d = d[0] if isinstance(d, list) else d
c = d.get("Config") or {}
lab = c.get("Labels") or {}
p = "io.github.efschu.flliper."
out = {"title": lab.get("org.opencontainers.image.title", ""), "rev": lab.get("org.opencontainers.image.revision", ""),
       "r27": lab.get(p + "revision.27b", ""), "rnf": lab.get(p + "revision.nf", ""), "build": lab.get(p + "build", ""),
       "one": lab.get(p + "one_tree", ""), "ph": lab.get(p + "placeholders", ""),
       "version": lab.get("org.opencontainers.image.version", ""),
       "hc": "yes" if (c.get("Healthcheck") or {}).get("Test") else "no", "stop": c.get("StopSignal") or "",
       "ep": " ".join(c.get("Entrypoint") or []), "layers": len((d.get("RootFS") or {}).get("Layers") or [])}
for k, v in out.items():
    print(f"{k}={v}")
') || { res C1 FAIL "image inspect output not parseable"; exit 1; }
lab(){ printf '%s\n' "$LAB" | sed -n "s/^$1=//p" | head -1; }
R27=$(lab r27); RNF=$(lab rnf)
is40(){ printf '%s' "$1" | grep -qxE '[0-9a-f]{40}'; }
if want C1; then
  bad=()
  [ "$(lab title)" = fLLiper ] || bad+=("title='$(lab title)'")
  is40 "$(lab rev)" || bad+=("revision not 40 hex")
  [ "$(lab rev)" = "$R27" ] || bad+=("revision != revision.27b")
  is40 "$R27" || bad+=("revision.27b not 40 hex"); is40 "$RNF" || bad+=("revision.nf not 40 hex")
  [ "$(lab build)" = flat ] || bad+=("build='$(lab build)'")
  case "$(lab one)" in
    1) [ "$R27" = "$RNF" ] || bad+=("one_tree=1 but two revisions") ;;
    0) [ "$R27" != "$RNF" ] || bad+=("one_tree=0 but one revision") ;;
    "") [ "$R27" = "$RNF" ] || bad+=("no one_tree label (pre-0930 Dockerfile) and two revisions") ;;
    *) bad+=("one_tree='$(lab one)'") ;;
  esac
  if [ -n "$CTX" ]; then
    if [ -f "$CTX/BUILD_INFO.json" ]; then
      c27=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lines"]["27b"]["revision"])' "$CTX/BUILD_INFO.json")
      cnf=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lines"]["nf"]["revision"])' "$CTX/BUILD_INFO.json")
      [ "$c27" = "$R27" ] || bad+=("ctx 27b ${c27:0:10} != label ${R27:0:10}")
      [ "$cnf" = "$RNF" ] || bad+=("ctx nf ${cnf:0:10} != label ${RNF:0:10}")
    else bad+=("--ctx $CTX has no BUILD_INFO.json"); fi
  fi
  if [ "${#bad[@]}" = 0 ]; then res C1 PASS "version $(lab version), 27b ${R27:0:10}, nf ${RNF:0:10}, one_tree $(lab one)"
  else res C1 FAIL "${bad[*]}"; fi
  [ "$(lab ph)" = none ] || res C1 WARN "placeholders='$(lab ph)' (B4 values still open)"
fi
if want C2; then
  bad=()
  [ "$(lab hc)" = yes ] || bad+=("no HEALTHCHECK")
  [ "$(lab stop)" = SIGTERM ] || bad+=("StopSignal='$(lab stop)'")
  [ "$(lab ep)" = /opt/htsglang/entrypoint.sh ] || bad+=("Entrypoint='$(lab ep)'")
  [ "$(lab layers)" -le 70 ] 2>/dev/null || bad+=("$(lab layers) layers > 70")
  if [ "${#bad[@]}" = 0 ]; then res C2 PASS "healthcheck, SIGTERM, entrypoint, $(lab layers) layers"; else res C2 FAIL "${bad[*]}"; fi
fi

# ---- C3: slot trees ---------------------------------------------------------------------------------------------------
if want C3; then
  out=$("${RUN[@]}" --entrypoint bash "$IMG" -c 'for f in 27b nf; do d=/opt/htsglang/src-$f; echo "$f $(git -C $d rev-parse HEAD 2>/dev/null || echo none) $(git -C $d status --porcelain 2>/dev/null | wc -l)"; done' 2>&1)
  h27=$(printf '%s\n' "$out" | awk '$1=="27b"{print $2}'); d27=$(printf '%s\n' "$out" | awk '$1=="27b"{print $3}')
  hnf=$(printf '%s\n' "$out" | awk '$1=="nf"{print $2}'); dnf=$(printf '%s\n' "$out" | awk '$1=="nf"{print $3}')
  if [ "$h27" = "$R27" ] && [ "$hnf" = "$RNF" ] && [ "$d27" = 0 ] && [ "$dnf" = 0 ]; then res C3 PASS "src-27b @ ${h27:0:10}, src-nf @ ${hnf:0:10}, both clean"
  else res C3 FAIL "src-27b ${h27:-?} (dirty ${d27:-?}) vs ${R27:0:10}; src-nf ${hnf:-?} (dirty ${dnf:-?}) vs ${RNF:0:10}"; fi
fi

# ---- C4: D1 selfcheck per slot ---------------------------------------------------------------------------------------
if want C4; then
  for slot in 27b nf; do
    if o=$("${RUN[@]}" -e MODE=weg2 -e HTSGLANG_PROFILE="$slot" "$IMG" selfcheck 2>&1); then res C4 PASS "selfcheck $slot rc 0"
    else res C4 FAIL "selfcheck $slot rc != 0: $(printf '%s\n' "$o" | grep -iE 'fatal|fehl|error|fail' | head -2 | tr '\n' ' ' | cut -c1-200)"; fi
  done
fi

# ---- C5: JIT reports ---------------------------------------------------------------------------------------------------
if want C5; then
  o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python "$IMG" -c '
import glob, json
fs = sorted(glob.glob("/opt/htsglang/JIT_PREBUILD-*.json"))
print("N", len(fs))
for f in fs:
    d = json.load(open(f))
    print("R", f.rsplit("/", 1)[-1], d.get("verdict"), len(d.get("problems") or []))
' 2>&1)
  n=$(printf '%s\n' "$o" | awk '$1=="N"{print $2}')
  badr=$(printf '%s\n' "$o" | awk '$1=="R" && ($3!="OK" || $4!=0){print $2"="$3"/"$4}' | paste -sd' ' -)
  if [ -z "$n" ] || [ "$n" = 0 ]; then res C5 FAIL "no JIT_PREBUILD reports ($(printf '%s' "$o" | tail -1 | cut -c1-120))"
  elif [ -n "$badr" ]; then res C5 FAIL "reports not OK: $badr"
  else res C5 PASS "$n reports verdict OK, 0 problems"; fi
fi

# ---- C6: import smoke per slot -----------------------------------------------------------------------------------------
IMPORT_PY='
import importlib, os, sys
root = os.environ["SLOT_ROOT"]
pkg = "flliper" if os.path.isdir(os.path.join(root, "python", "flliper")) else "sglang"
sub = "pdflip" if pkg == "flliper" else "weg2"
mods = [pkg, f"{pkg}.srt.{sub}.launcher", f"{pkg}.srt.{sub}.front", f"{pkg}.srt.{sub}.form",
        f"{pkg}.srt.entrypoints.engine", f"{pkg}.srt.utils.kernel_dist_guard",
        "sgl_kernel", "flashinfer", "cutlass", "tvm_ffi", "triton", "torch"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        if m == pkg and not os.path.realpath(mod.__file__).startswith(os.path.realpath(root)):
            bad.append(f"{m} from {mod.__file__} (not the slot tree)")
    except Exception as e:  # noqa: BLE001
        bad.append(f"{m}: {type(e).__name__}: {str(e)[:80]}")
import importlib.metadata as md
fi = md.version("flashinfer-python")
if not fi.startswith("0.7.0"): bad.append(f"flashinfer {fi}")
import torch
if not torch.__version__.startswith("2.11"): bad.append(f"torch {torch.__version__}")
print("PKG", pkg, "fi", fi, "torch", torch.__version__)
print("BAD", " | ".join(bad) if bad else "-")
'
if want C6; then
  for slot in 27b nf; do
    o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python -e SLOT_ROOT=/opt/htsglang/src-$slot \
          -e PYTHONPATH=/opt/htsglang/src-$slot/python "$IMG" -c "$IMPORT_PY" 2>&1)
    b=$(printf '%s\n' "$o" | sed -n 's/^BAD //p' | tail -1)
    if [ "$b" = "-" ]; then res C6 PASS "$slot: $(printf '%s\n' "$o" | sed -n 's/^PKG //p' | tail -1)"
    else res C6 FAIL "$slot: ${b:-no result: $(printf '%s' "$o" | tail -1 | cut -c1-160)}"; fi
  done
  # no-shadow: without a slot PYTHONPATH neither package may import (both live only in the slot trees)
  o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python -e PYTHONPATH= "$IMG" -c '
import importlib.util as u
print("SHADOW", " ".join(m for m in ("sglang", "flliper") if u.find_spec(m)) or "-")' 2>&1)
  s=$(printf '%s\n' "$o" | sed -n 's/^SHADOW //p' | tail -1)
  if [ "$s" = "-" ]; then res C6 PASS "no sglang/flliper in site-packages"; else res C6 FAIL "site-packages shadow: ${s:-? $(printf '%s' "$o" | tail -1)}"; fi
fi

# ---- C7: profiles parse ------------------------------------------------------------------------------------------------
PROF_SH='
acc=$(/opt/venv/bin/python -c "import json; print(\" \".join(json.load(open(\"/opt/htsglang/BUILD_INFO.json\")).get(\"accepted_first_docker\", [])))" 2>/dev/null)
echo "ACCEPTED $acc"
for f in /opt/htsglang/profiles/*.env; do
  n=$(basename "$f" .env)
  r=$( ( _form(){ :; }; HTSGLANG_TAG=postcheck; SGLANG_WEG2_EVIDENCE_DIR=/tmp; cd /opt/htsglang/profiles && source "$f" >/dev/null 2>&1 \
        && { declare -F profile_form_env >/dev/null && profile_form_env >/dev/null 2>&1; true; } \
        && echo "OK ${PROFILE_STATUS:-abgenommen} ${#PROFILE_ARGS[@]} ${PROFILE_MODEL:+model}" ) 2>&1 | tail -1)
  echo "P $n $r"
done'
if want C7; then
  o=$("${RUN[@]}" --entrypoint bash "$IMG" -c "$PROF_SH" 2>&1)
  acc=$(printf '%s\n' "$o" | sed -n 's/^ACCEPTED //p' | head -1)
  np=$(printf '%s\n' "$o" | grep -c '^P ' || true)
  bad=$(printf '%s\n' "$o" | awk '$1=="P" && ($3!="OK" || ($4!="abgenommen" && $4!="experimentell" && $4!="geplant")){print $2}' | paste -sd' ' -)
  nomodel=$(printf '%s\n' "$o" | awk '$1=="P" && $3=="OK" && $4!="geplant" && $6!="model"{print $2}' | paste -sd' ' -)
  notacc=""
  for a in ${PROFILES_CHECK:-$acc}; do
    st=$(printf '%s\n' "$o" | awk -v n="$a" '$1=="P" && $2==n{print $4}')
    [ "$st" = abgenommen ] || notacc+="$a=${st:-missing} "
  done
  if [ "$np" = 0 ]; then res C7 FAIL "no profiles under /opt/htsglang/profiles"
  elif [ -n "$bad$notacc" ]; then res C7 FAIL "unparsable/bad status: ${bad:--}; accepted not abgenommen: ${notacc:--}"
  else res C7 PASS "$np profiles source cleanly; accepted ($acc) abgenommen"; fi
  [ -z "$nomodel" ] || res C7 WARN "no PROFILE_MODEL after sourcing: $nomodel"
fi

# ---- C8: release defaults (27B Leistungsschalter in the registry row of the 27b slot) ----------------------------------
DEF_PY='
import importlib, os
root = "/opt/htsglang/src-27b"
pkg = "flliper" if os.path.isdir(root + "/python/flliper") else "sglang"
sub = "pdflip" if pkg == "flliper" else "weg2"
FM = importlib.import_module(f"{pkg}.srt.{sub}.form")
row = FM.PROFILES.get("qwen27b")
fields = ("front_exact_tokens", "p_row_authority", "d_hostgap_levers", "d_hostgap_base", "d_release_fixes")
miss = [f for f in fields if not hasattr(row, f)]
off = [f for f in fields if hasattr(row, f) and not getattr(row, f)]
sd = FM.PROFILE_SWITCH_DEFAULTS.get("qwen27b", {})
print("MISS", " ".join(miss) or "-"); print("OFF", " ".join(off) or "-")
print("ON", sum(1 for v in sd.values() if v is True), "of", len(sd))
'
if want C8; then
  o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python -e PYTHONPATH=/opt/htsglang/src-27b/python "$IMG" -c "$DEF_PY" 2>&1)
  miss=$(printf '%s\n' "$o" | sed -n 's/^MISS //p' | tail -1); off=$(printf '%s\n' "$o" | sed -n 's/^OFF //p' | tail -1)
  if [ "$miss" = "-" ] && [ "$off" = "-" ]; then res C8 PASS "qwen27b row: 5/5 proven switch fields on; $(printf '%s\n' "$o" | sed -n 's/^ON //p' | tail -1) switch defaults true"
  else res C8 FAIL "qwen27b row: missing ${miss:-?} / off ${off:-?} ($(printf '%s' "$o" | tail -1 | cut -c1-120))"; fi
fi

# ---- C9: pip check -----------------------------------------------------------------------------------------------------
if want C9; then
  o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python "$IMG" -m pip check 2>&1); n=$(printf '%s\n' "$o" | grep -c . || true)
  if printf '%s' "$o" | grep -q 'No broken requirements'; then res C9 PASS "pip check clean"
  else res C9 WARN "pip check: $n findings (compare with the frozen allow-list; reference venv had 8)"; fi
fi

# ---- C10: dryrun with models --------------------------------------------------------------------------------------------
if want C10; then
  if [ -z "$MODELS" ]; then res C10 SKIP "no --models (dryrun reads the checkpoint config)"
  else
    acc=$("${RUN[@]}" --entrypoint /opt/venv/bin/python "$IMG" -c 'import json; print(" ".join(json.load(open("/opt/htsglang/BUILD_INFO.json")).get("accepted_first_docker", [])))' 2>/dev/null)
    for p in ${PROFILES_CHECK:-$acc}; do
      if "${RUN[@]}" -v "$MODELS:/spinning/llm_stuff/club-3090/models-cache:ro" -e MODE=weg2 -e HTSGLANG_PROFILE="$p" "$IMG" dryrun >/dev/null 2>&1
      then res C10 PASS "dryrun $p rc 0"; else res C10 FAIL "dryrun $p rc != 0"; fi
    done
  fi
fi

# ---- C11: deadman in the image ------------------------------------------------------------------------------------------
if want C11; then
  o=$("${RUN[@]}" --entrypoint /opt/venv/bin/python "$IMG" -c '
import hashlib, json, os, re
dm = "/opt/htsglang/devtools/boot_deadman.sh"   # boot_deadman
b = open(dm, "rb").read() if os.path.exists(dm) else b""
try:
    want = (json.load(open("/opt/htsglang/BUILD_INFO.json")).get("deadman") or {}).get("md5") or "-"
except Exception:
    want = "-"
print("DM md5=%s progress=%d want=%s" % (hashlib.md5(b).hexdigest() if b else "none", b.count(b"progress-check"), want))
for slot in ("27b", "nf"):
    sf = next((p for p in ("/opt/htsglang/src-%s/python/flliper/srt/pdflip/state_file.py" % slot,
                           "/opt/htsglang/src-%s/python/sglang/srt/weg2/state_file.py" % slot) if os.path.exists(p)), None)
    t = open(sf).read() if sf else ""
    print("S %s sf=%s progress=%d rank=%d death=%d" % (slot, sf or "none", t.count("\"progress-check\""),
          len(re.findall(r"(?m)^WRITERS = .*\"rank\"", t)), len(re.findall(r"(?m)^def note_rank_death", t))))
' 2>&1)
  dmp=$(printf '%s\n' "$o" | sed -n 's/^DM .* progress=\([0-9]*\) .*/\1/p'); dmm=$(printf '%s\n' "$o" | sed -n 's/^DM md5=\([^ ]*\) .*/\1/p')
  dmw=$(printf '%s\n' "$o" | sed -n 's/^DM .* want=\(.*\)$/\1/p'); bad=""; old=""
  [ -n "$dmp" ] || bad="no deadman answer ($(printf '%s' "$o" | tail -1 | cut -c1-120))"
  for slot in 27b nf; do
    l=$(printf '%s\n' "$o" | grep "^S $slot "); sp=$(sed -n 's/.* progress=\([0-9]*\) .*/\1/p' <<< "$l")
    sr=$(sed -n 's/.* rank=\([0-9]*\) .*/\1/p' <<< "$l"); sd=$(sed -n 's/.* death=\([0-9]*\)$/\1/p' <<< "$l")
    [ -n "$sp" ] || { bad="$bad; $slot: no state_file answer"; continue; }
    if [ "${dmp:-0}" -gt 0 ] && [ "$sp" = 0 ]; then bad="$bad; the deadman calls progress-check, $slot state_file.py has none"
    elif [ "${dmp:-0}" = 0 ] && [ "$sp" -gt 0 ]; then bad="$bad; $slot tree has progress-check, the deadman not (host copy baked in)"; fi
    [ "${sd:-0}" -gt 0 ] && [ "${sr:-0}" = 0 ] && bad="$bad; $slot: note_rank_death without the rank writer"
    { [ "$sp" = 0 ] || [ "${sr:-0}" = 0 ]; } && old="$old $slot"
  done
  [ "${dmw:--}" != "-" ] && [ "$dmm" != "$dmw" ] && bad="$bad; md5 $dmm != BUILD_INFO deadman.md5 $dmw"
  [ "${dmp:-0}" = 0 ] && old="$old deadman"
  if [ -n "$bad" ]; then res C11 FAIL "${bad#; }"
  elif [ -n "$old" ] && [ "${FLAT_REQUIRE_DEADMAN:-0}" = 1 ]; then res C11 FAIL "FLAT_REQUIRE_DEADMAN=1: PROGRESS-STALL/rank writer missing in:$old"
  elif [ -n "$old" ]; then res C11 WARN "old generation (no PROGRESS-STALL / rank writer) in:$old -- consistent, not the release form"
  else res C11 PASS "deadman md5 ${dmm:0:8} progress-check, both slots' state_file.py fit (rank writer), = BUILD_INFO"; fi
fi

say "POSTCHECK $IMG: $NFAIL FAIL, $NWARN WARN -> $([ "$NFAIL" = 0 ] && echo GREEN || echo RED)"
[ "$NFAIL" = 0 ]
