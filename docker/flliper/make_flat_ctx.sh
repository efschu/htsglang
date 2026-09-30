#!/bin/bash
# make_flat_ctx.sh -- context generator for the flat fLLiper release image (release table row 7).
# FB for the 27B operator, 2026-09-26. Plan: /spinning/gpu-arb/docs/FLAT_IMAGE_PLAN.md
#
#   ./make_flat_ctx.sh --plan  [options]     check every input, print the context/build/tag/labels/blockers (read-only)
#   ./make_flat_ctx.sh --write [options]     same checks, then create ctx/flat-<release>-<sha10>-<cu>/ (builds NOTHING)
#   options: [--rev <sha|ref>] [--branch <branch>] [--release <version>] [--cu cu130] [--since 2026-09-01]
#            [--kernels <list> ...] [--profiles "<name> ..."] [--source <url>] [--allow-unpushed]
#            [--host] [--emit-kernels] [--no-hash] [--accepted "27b nf"] [--mem-profile auto|full|nf]
#   --mem-profile nf: build NEXT TO a running boot (NF Dauerbetrieb): builder cap 16g, 8 CPUs, the prebuild concurrency is
#   baked into the ctx Dockerfile (ctx name gets -nf); full: no boot, full parallelism; auto: nf if --host sees a boot.
#
# --write writes ONLY below $HERE/ctx/flat-* (first into <ctx>.partial, renamed at the end; an existing ctx is never
# overwritten). It reads the repo, the reference venv, the pinned wheels and the rig files the profiles need. Until B1
# (renamed tree) and B2 (entrypoint FLLIPER_*) are closed it ALSO accepts the old layout (operator 26.09.: procedure test
# before B1) -- those two are recorded as release blockers in BUILD_INFO.json, not as ctx refusals. The revision must be
# b3180fb493 or a descendant. B4 values (source repo, version scheme) are placeholders unless given (--source/--release)
# and are named in the label io.github.efschu.flliper.placeholders.
#
# --plan is strictly read-only:
#   * no file is written anywhere (no ctx/, no log, nothing under /root/.cache), stdout only;
#   * git runs with --no-optional-locks and only reads objects (rev-parse, cat-file, ls-tree, grep, branch -r);
#   * --host reads the Proxmox host over `ssh -o BatchMode=yes proxmox` (meminfo, nproc, zfs avail, docker image
#     inspect, buildx inspect, boot_procs.sh) -- no docker build, no container start, no pull, no builder start;
#   * no GPU, no secrets (wheels and lock are hashed, never keys).
#
# Exit: --plan 0 = no blocker, 3 = blockers named (BLOCKER lines); --write 0 = ctx created (release blockers may remain,
# listed), 3 = refused by a ctx blocker (nothing kept); 2 = usage.
set -uo pipefail

HERE=${PREPARE_HERE:-/spinning/gpu-arb/docker}
REPO=${REPO:-/spinning/htsglang}
REF_VENV=${REF_VENV:-/spinning/htsglang-gpu/.venv}
# Pins of the cu130 stack (rc8b full build 2026-09-25, carried unchanged through rc10z)
KW_FILE=${KW_FILE:-/spinning/wt-kernel-cu13-120a-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl}
KW_SHA=605c54e76b351a3715ef8a875ca6101e4a88e18f1b1206576fbd81ddfdde1de9
RC8B_CTX=$HERE/ctx/duo-rc8b-27bd342caa832-nf8f1db7a6e5-cu130
FIPY_FILE=${FIPY_FILE:-/spinning/fi-wheels-0925/src-2f3bc5ac/flashinfer_python-0.7.0-py3-none-any.whl}   # operator 26.09. 18:04Z
FIPY_SHA=708abc835862ecc0b35ccecda84f608851026680542d8e2eb9dfeabfd1ee5e1b   # source build flashinfer 2f3bc5ac (#5242)
FICUBIN_FILE=${FICUBIN_FILE:-/spinning/fi-wheels-0925/flashinfer_cubin-0.7.0-py3-none-any.whl}
FICUBIN_SHA=f1821e11ad4ea9666a09c2b04cc16b1e34f601296dc7a7b689649281c0358a9c
OVERRIDES=${OVERRIDES:-$HERE/overrides_fi070.txt}
LOCK_SHA_RC8B=ae10b55f1ca9e1932fbad841e08d1b659fcc7e68dea6da7912a7e30a2bd0cd95
NCCL_SHA=1c8618b866734cbdd5401715d6178be763ece283b7f808ecf86dedab211162c1
NCCL_BANNER="NCCL version 2.28.9+cuda13.0"
BASE_TAG=nvidia/cuda:13.0.2-devel-ubuntu24.04
BASE_DIGEST_RC8B=sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
# Release profile whitelist (row 8, user 26.09. 17:20Z: first acceptance only 27b INT8 + nf INT4, the rest experimental)
# E4 (Operator 28.09.): the release profiles are `27b` (= 27b-release-draft + Fix B) and `nf` (= nf-h91-dpr-sa-vis, vision);
# the old names ship as experimental aliases, 27b-base is the old 27b form (base of 27b and of the 27b format siblings).
# Their files come from the release overlay PROFILE_OVERLAY (profiles_release/); every other profile from profiles/.
PROFILES_DEFAULT="27b 27b-base 27b-release-draft 27b-row-authority 27b-nvfp4 27b-fp8 27b-gguf nf nf-h91-dpr-sa-vis nf-nvfp4 nf-nvfp4-d nf-gguf"
# accepted for the first Docker acceptance (user 26.09. 17:20Z); every other whitelisted profile that is "abgenommen" on the
# rig is gated to "experimentell" IN THE CTX COPY (the rig profile file is not touched)
ACCEPTED_DEFAULT="27b nf"
PROFILE_OVERLAY=${PROFILE_OVERLAY:-$HERE/profiles_release}
# pf <profile>: the file this profile ships from -- the E4 release overlay first, the rig table otherwise
pf(){ if [ -f "$PROFILE_OVERLAY/$1.env" ]; then echo "$PROFILE_OVERLAY/$1.env"; else echo "$HERE/profiles/$1.env"; fi; }
MIN_REV=b3180fb493c02d4a27ae4108a4bccaf3f7719c17   # operator 26.09.: b3180fb493 or newer
ARB=/spinning/gpu-arb
KERNEL_LISTS_DEFAULT="delta_kernels_rc10u.txt delta_kernels_rc9dwin.txt delta_kernels_rc9f.txt delta_kernels_rc9c.txt delta_kernels_rc9b.txt delta_kernels_rc9.txt"

PLAN=0; WRITE=0; REV_IN=""; BRANCH=""; REL=""; CU=cu130; KLISTS=(); PROFILES=$PROFILES_DEFAULT; HOST=0; EMIT=0; HASH=1
SINCE=2026-09-01; SOURCE=""; ALLOW_UNPUSHED=0; ACCEPTED=$ACCEPTED_DEFAULT; MEMPROF=auto; LOCK_FOLD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) PLAN=1; shift ;;
    --write) WRITE=1; shift ;;
    --since) SINCE=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --allow-unpushed) ALLOW_UNPUSHED=1; shift ;;
    --accepted) ACCEPTED=$2; shift 2 ;;
    --mem-profile) MEMPROF=$2; shift 2 ;;
    --lock-fold) LOCK_FOLD=1; shift ;;   # PA 28.09. (Checkliste Punkt 8): PyPI-Pins der Overrides in den Lock, qwen-tts raus
    --rev) REV_IN=$2; shift 2 ;;
    --branch) BRANCH=$2; shift 2 ;;
    --release) REL=$2; shift 2 ;;
    --cu) CU=$2; shift 2 ;;
    --kernels) KLISTS+=("$2"); shift 2 ;;
    --profiles) PROFILES=$2; shift 2 ;;
    --host) HOST=1; shift ;;
    --emit-kernels) EMIT=1; shift ;;
    --no-hash) HASH=0; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ $((PLAN + WRITE)) != 1 ]; then echo "exactly one of --plan / --write (see --help); nothing was done" >&2; exit 2; fi
# B4 placeholders (user decision pending): the version scheme and the public source repo
PH=()
# B4 decided by the user 26.09. ~18:55Z (memory flliper-source-und-version-0926): source efschu/fLLiper, flliper:0.1.0-cu130.
# A probe build must pass its own --release (e.g. 0.1.0-probe1) so it never takes the release tag.
if [ -z "$REL" ]; then REL=0.1.0; fi
if [ -z "$SOURCE" ]; then SOURCE=https://github.com/efschu/fLLiper; fi
PLACEHOLDERS=$( [ "${#PH[@]}" -gt 0 ] && echo "${PH[*]} (B4, user decision pending)" || echo none )
[ "${#KLISTS[@]}" -gt 0 ] || for k in $KERNEL_LISTS_DEFAULT; do KLISTS+=("$HERE/$k"); done
case "$REL" in *[!A-Za-z0-9._-]*|"") echo "release '$REL' is not a valid docker tag part" >&2; exit 2 ;; esac

G=(git --no-optional-locks -c safe.directory='*' -C "$REPO")
BLOCKERS=(); WARNS=()
say(){ printf '%s\n' "$*"; }
sec(){ printf '\n== %s\n' "$*"; }
blocker(){ BLOCKERS+=("$*"); say "   BLOCKER: $*"; }
# relblock: blocks the RELEASE (B1/B2) -- a blocker in --plan, recorded but not refusing in --write (procedure test)
RELBLOCKERS=()
relblock(){ if [ "$WRITE" = 1 ]; then RELBLOCKERS+=("$*"); say "   RELEASE-BLOCKER (ctx allowed): $*"; else blocker "$*"; fi; }
# buildblock: blocks the BUILD on the host (tag, disk, boot, RAM) -- a blocker in --plan, a warning for the ctx
buildblock(){ if [ "$WRITE" = 1 ]; then warn "build-time: $*"; else blocker "$*"; fi; }
warn(){ WARNS+=("$*"); say "   WARN: $*"; }
ok(){ say "   ok: $*"; }

if [ "$WRITE" = 1 ]; then say "make_flat_ctx.sh --write  $(date -u +%FT%TZ)  (writes only below $HERE/ctx/flat-*; nothing is built)"
else say "make_flat_ctx.sh --plan  $(date -u +%FT%TZ)  (read-only; nothing is written, nothing is built)"; fi

# --- 1. Source tree -------------------------------------------------------------------------------------------------
sec "1. Source tree (ONE unified tree for both slots)"
if [ -z "$REV_IN" ]; then
  REV_IN=desk/27b-unified-0926
  git --no-optional-locks -C "$REPO" rev-parse -q --verify "refs/remotes/origin/$REV_IN" >/dev/null && REV_IN=origin/$REV_IN
  say "   no --rev given: using the unified line head $REV_IN (NOT the release input -- the renamed commit does not exist yet)"
fi
[ -n "$BRANCH" ] || BRANCH=${REV_IN#origin/}
REV=$("${G[@]}" rev-parse --verify -q "${REV_IN}^{commit}") || { say "   revision $REV_IN unknown in $REPO"; exit 2; }
SHA10=${REV:0:10}
say "   revision: $REV ($("${G[@]}" log -1 --format='%cI %s' "$REV" | cut -c1-110))"
BR=""
for b in "refs/heads/$BRANCH" "refs/remotes/origin/$BRANCH"; do
  "${G[@]}" rev-parse -q --verify "$b" >/dev/null && "${G[@]}" merge-base --is-ancestor "$REV" "$b" 2>/dev/null && { BR=$b; break; }
done
if [ -n "$BR" ]; then ok "on branch $BR"; else blocker "revision ${SHA10} is on neither refs/heads/$BRANCH nor origin/$BRANCH"; fi
if "${G[@]}" merge-base --is-ancestor "$MIN_REV" "$REV" 2>/dev/null; then ok "descendant of ${MIN_REV:0:10} (operator minimum)"
else blocker "revision ${SHA10} is not ${MIN_REV:0:10} or newer (operator minimum)"; fi
PUSHED=$("${G[@]}" branch -r --contains "$REV" 2>/dev/null | sed 's/^[* ]*//' | grep -v -- '->' | grep -xF "origin/$BRANCH")
[ -n "$PUSHED" ] || PUSHED=$("${G[@]}" branch -r --contains "$REV" 2>/dev/null | sed 's/^[* ]*//' | grep -v -- '->' | head -3 | paste -sd, -)
if [ -n "$PUSHED" ]; then ok "pushed: $PUSHED (remote refs as of the last fetch)"
elif [ "$ALLOW_UNPUSHED" = 1 ]; then PUSHED=""; warn "revision ${SHA10} UNPUSHED (--allow-unpushed): BUILD_INFO/label carry UNPUSHED, publication locked"
else blocker "revision ${SHA10} is on no remote branch (UNPUSHED) -- push first or --allow-unpushed"; fi
PUSH_STATE=$([ -n "$PUSHED" ] && echo "pushed: $PUSHED" || echo "UNPUSHED, publication only after the user's push")
if "${G[@]}" cat-file -e "$REV:python/flliper" 2>/dev/null; then PKG=flliper; PDF=pdflip; RENAMED=1
  ok "renamed layout: python/flliper, subsystem pdflip"
else PKG=sglang; PDF=weg2; RENAMED=0
  relblock "B1: tree ${SHA10} is NOT renamed (python/sglang, weg2) -- RENAME_PLAN 8.15 steps 2-5 + translation missing; the flat fLLiper image needs the renamed commit"
fi
"${G[@]}" cat-file -e "$REV:python/$PKG/__init__.py" 2>/dev/null || blocker "python/$PKG/__init__.py missing in ${SHA10}"
# _version.py must be ignored by the tree's own .gitignore (Dockerfile step 4 check-ignore)
if "${G[@]}" ls-tree -r --name-only "$REV" | grep -qx "python/$PKG/_version.py"; then blocker "python/$PKG/_version.py is TRACKED in ${SHA10} (Dockerfile writes it)"
elif { "${G[@]}" show "$REV:python/.gitignore" 2>/dev/null; "${G[@]}" show "$REV:.gitignore" 2>/dev/null; } | grep -q '_version.py'; then ok "_version.py untracked and listed in a .gitignore"
else warn "_version.py not found in python/.gitignore or .gitignore of ${SHA10} -- Dockerfile check-ignore may FATAL"; fi
# stage-B seams (prepare_context.sh step 0, both spellings)
missing=()
for f in "python/$PKG/srt/$PDF/launcher.py" "python/$PKG/srt/$PDF/host_ledger.py" "python/$PKG/srt/$PDF/corridor_budget.py" \
         "scripts/$PDF/tms/build_tms_preload.sh" "test/registered/unit/$PDF/test_${PDF}_rig_paths_env_docker.py"; do
  "${G[@]}" grep -q -E '(SGLANG_WEG2|FLLIPER_PDFLIP)_(GPU_ARB|EVIDENCE_DIR|VENV|TMS_OUT_DIR)' "$REV" -- "$f" 2>/dev/null || missing+=("$f")
done
if [ "${#missing[@]}" = 0 ]; then ok "stage-B path seams present in all five files"; else blocker "stage-B seams missing in: ${missing[*]}"; fi
for f in docker/htsglang-chat_template.jinja docker/htsglang-entrypoint.sh; do
  "${G[@]}" cat-file -e "$REV:$f" 2>/dev/null && ok "Dockerfile step 7 source present: $f" || warn "$f missing in ${SHA10} (Dockerfile step 7 copies it; renamed path?)"
done
NFILES=$("${G[@]}" ls-tree -r --name-only "$REV" | wc -l)
say "   tree: $NFILES tracked files; both slots src-27b/ and src-nf/ = the same revision (as rc10u..rc10z)"
if [ "$RENAMED" = 1 ]; then
  n_old=$("${G[@]}" grep -I -c -E '\bweg2\b|SGLANG_WEG2_' "$REV" -- "python/$PKG" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
  say "   residue in python/$PKG: $n_old lines with weg2/SGLANG_WEG2_ (compat shims are expected; the translation gate owns the verdict)"
fi

# --- 2. Pinned inputs -----------------------------------------------------------------------------------------------
sec "2. Pinned inputs (by sha256; models are NOT in the image)"
chk(){ # chk <label> <file> <sha>
  if [ ! -f "$2" ]; then blocker "$1 missing: $2"; return; fi
  if [ "$HASH" = 0 ]; then say "   $1: $2 ($(stat -c %s "$2") B, hash not checked: --no-hash)"; return; fi
  local h; h=$(sha256sum "$2" | cut -d' ' -f1)
  if [ "$h" = "$3" ]; then ok "$1 $(basename "$2") sha256 ${h:0:12}"; else blocker "$1 $2 sha256 ${h:0:12} != pin ${3:0:12}"; fi
}
chk "kernel wheel (sgl-kernel 0.4.4, 86;120a, cu13)" "$KW_FILE" "$KW_SHA"
chk "flashinfer-python 0.7.0 (source build 2f3bc5ac)" "$FIPY_FILE" "$FIPY_SHA"
chk "flashinfer-cubin 0.7.0" "$FICUBIN_FILE" "$FICUBIN_SHA"
if [ -f "$OVERRIDES" ]; then
  for p in 'flashinfer_python-0.7.0' 'flashinfer_cubin-0.7.0' 'nvidia-cutlass-dsl==4.7.1' 'nvidia-cutlass-dsl-libs-cu13==4.7.1' 'flash-attn-4==4.0.0b32'; do
    grep -q -F "$p" "$OVERRIDES" || blocker "overrides $OVERRIDES lacks $p"
  done
  ok "overrides $(basename "$OVERRIDES"): $(grep -cvE '^\s*(#|$)' "$OVERRIDES") pins (FlashInfer 0.7.0@2f3bc5ac, CuTe DSL 4.7.1, ...)"
  if [ "$LOCK_FOLD" = 1 ]; then   # plan view of --lock-fold (the fold itself runs in --write, step 2)
    say "   lock-fold: the $(grep -cvE '^\s*(#|/|$)' "$OVERRIDES") PyPI pins of $(basename "$OVERRIDES") go INTO the lock, flashinfer-python 0.6.14 and qwen-tts 0.1.1 leave it; overrides keep only the $(grep -c '^/' "$OVERRIDES") FlashInfer wheel(s); the lock sha changes (pip layer rebuilds once, ~2.5 min)"
  fi
else blocker "overrides file missing: $OVERRIDES"; fi
if [ -x "$REF_VENV/bin/python" ]; then
  LOCK_SHA=$("$REF_VENV/bin/python" -m pip freeze --exclude-editable 2>/dev/null | grep -v -E '^sglang-kernel @ ' | sha256sum | cut -d' ' -f1)
  if [ "$LOCK_SHA" = "$LOCK_SHA_RC8B" ]; then ok "lock from $REF_VENV = rc8b lock ${LOCK_SHA:0:12} (pip layer can hit the builder cache)"
  else warn "lock ${LOCK_SHA:0:12} != rc8b ${LOCK_SHA_RC8B:0:12} -- reference venv changed since 25.09.; pip layer rebuilds, diff the freeze before building"; fi
  "$REF_VENV/bin/python" - <<'PY'
import importlib.metadata as m
v = {d: m.version(d) for d in ("torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl", "apache-tvm-ffi", "nvidia-nccl-cu13")}
print("   reference venv: " + ", ".join(f"{k} {x}" for k, x in v.items()) + "  (flashinfer/cutlass-dsl replaced by the overrides in the image)")
try:
    m.version("qwen-tts"); print("   WARN-INFO: qwen-tts is in the reference venv (not needed by the product; source of pip-check findings)")
except m.PackageNotFoundError:
    pass
PY
  L=$REF_VENV/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
  if [ -f "$L" ]; then
    h=$(sha256sum "$L" | cut -d' ' -f1); b=$(strings "$L" | grep -m1 -oE 'NCCL version [0-9.]+\+cuda[0-9.]+' || true)
    [ "$h" = "$NCCL_SHA" ] && [ "$b" = "$NCCL_BANNER" ] && ok "NCCL $b sha256 ${h:0:12}" || blocker "NCCL $L: '$b' ${h:0:12} != pin '$NCCL_BANNER' ${NCCL_SHA:0:12}"
  fi
  PIPCHK=$("$REF_VENV/bin/python" -m pip check 2>/dev/null | wc -l)
  say "   pip check baseline on the reference venv: $PIPCHK findings (image gate = no finding outside the allow-list, plan §5)"
else blocker "reference venv $REF_VENV missing"; fi
FIM=$RC8B_CTX/assets/flashinfer_modules.txt
if [ -f "$FIM" ]; then ok "FlashInfer module list (rc8b ctx, rig 0.6.14 build.ninja): $(grep -cv '^#' "$FIM") modules -> Dockerfile step 3b"
else blocker "FlashInfer module list missing: $FIM"; fi
say "   models: not in the image; profiles name them under /spinning/llm_stuff/club-3090/models-cache (bind mount)"

# --- 3. JIT prebuild: kernel list (union), tvm-ffi seed, Triton seed ------------------------------------------------
sec "3. JIT prebuild for both slots"
UNION=""
for l in "${KLISTS[@]}"; do
  [ -f "$l" ] || { blocker "kernel list missing: $l"; continue; }
  UNION+=$(grep -vE '^\s*(#|$)' "$l" | sed 's/[[:space:]]\+/ /g; s/ $//')$'\n'
done
UNION=$(printf '%s' "$UNION" | awk 'NF && !seen[$0]++')
N_ALL=$(printf '%s\n' "$UNION" | grep -c . || true)
say "   kernel lists (start: $(basename "${KLISTS[0]}")): ${#KLISTS[@]} files -> $N_ALL distinct lines"
printf '%s\n' "$UNION" | awk '{k=$1; s=($0 ~ /src-nf\/python/) ? "nf" : "27b"; c[k" "s]++} END{for (x in c) printf "     %-14s %d\n", x, c[x]}' | sort
# module existence in the target tree (renamed tree: sglang. -> flliper., .weg2. -> .pdflip., as delta_prebuild.py maps)
miss=0; nchk=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  read -r -a W <<< "$line"; kind=${W[0]}; mod=""
  case "$kind" in
    "tvmffi"|"barlink") mod=${W[2]%%:*} ;; "module") mod=${W[1]} ;; "py") mod=${W[1]%%:*} ;; "fi") continue ;;
    *) mod=${W[0]%%:*} ;;
  esac
  [ -n "$mod" ] || continue
  if [ "$RENAMED" = 1 ]; then mod=$(printf '%s' "$mod" | sed -E 's/^sglang\./flliper./; s/\.weg2\./.pdflip./g'); fi
  p=python/${mod//.//}; nchk=$((nchk + 1))
  if ! "${G[@]}" cat-file -e "$REV:$p.py" 2>/dev/null && ! "${G[@]}" cat-file -e "$REV:$p/__init__.py" 2>/dev/null; then
    miss=$((miss + 1)); say "     module missing in ${SHA10}: $mod"
  fi
done <<< "$UNION"
if [ "$miss" = 0 ]; then ok "$nchk module references resolve in ${SHA10}"; else blocker "$miss of $nchk kernel-list modules do not exist in ${SHA10}"; fi
printf '%s\n' "$UNION" | grep -q '/opt/htsglang/' && say "   note: lines carry /opt/htsglang paths (PYTHONPATH=/opt/htsglang/src-nf/python, --report); keep them while the image paths stay /opt/htsglang"
say "   the full Dockerfile has NO kernel-list step (only Dockerfile.delta*): tvm-ffi + 13 'fi' lines need Dockerfile.flliper step 6b (plan B3)"
# tvm-ffi seed on the rig: usable in a cu13 base only if it links libcudart.so.13
if [ -d /root/.cache/tvm-ffi ]; then
  n=0; n13=0; n12=0
  for d in /root/.cache/tvm-ffi/*/; do
    [ -d "$d" ] || continue; n=$((n + 1))
    s=$(ls "$d"*.so 2>/dev/null | head -1); [ -n "$s" ] || continue
    c=$(readelf -d "$s" 2>/dev/null | grep -oE 'libcudart\.so\.[0-9]+' | head -1)
    case "$c" in libcudart.so.13) n13=$((n13 + 1)) ;; libcudart.so.12) n12=$((n12 + 1)) ;; esac
  done
  say "   rig tvm-ffi cache: $n entries ($n13 link cudart 13, $n12 link cudart 12). Seed policy: NO rig seed in the image --"
  say "     names hash the JIT sources, and the renamed/translated tree changes them; the image builds its own entries from the list"
fi
if [ -d /root/.triton/cache ]; then
  say "   rig Triton cache: $(ls /root/.triton/cache | wc -l) entries (~6.4 GB). Seed policy: NOT in the image (host_acceptance.sh seed ->"
  say "     \$ACC/triton volume, as today); keys hash the kernel source text, so a renamed-tree seed must come from its first metal boot"
fi

# --- 4. Profiles ----------------------------------------------------------------------------------------------------
sec "4. Profiles (release whitelist)"
# eff_status <profile>: last PROFILE_STATUS= of the file, else of the profile it sources, else "abgenommen" (entrypoint rule)
eff_status(){ local f; f=$(pf "$1"); local st dep
  st=$(grep -oE '^PROFILE_STATUS=[^ #]*' "$f" | tail -1 | cut -d= -f2)
  if [ -z "$st" ]; then dep=$(grep -oE '^[[:space:]]*source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[A-Za-z0-9._-]+\.env"' "$f" | grep -oE '[A-Za-z0-9._-]+\.env"$' | tr -d '"' | tail -1)
    [ -n "$dep" ] && st=$(eff_status "${dep%.env}"); fi
  echo "${st:-abgenommen}"; }
# img_status <profile>: status the profile gets IN THE IMAGE (gate: only $ACCEPTED stay abgenommen)
img_status(){ local st; st=$(eff_status "$1"); case " $ACCEPTED " in *" $1 "*) echo "$st"; return;; esac
  [ "$st" = abgenommen ] && echo experimentell || echo "$st"; }
say "   accepted for the first Docker acceptance: $ACCEPTED (the others run only with HTSGLANG_ALLOW_EXPERIMENTAL=1 or are refused)"
for p in $PROFILES; do
  f=$(pf "$p")
  if [ ! -f "$f" ]; then blocker "profile $p.env missing in $HERE/profiles (and $PROFILE_OVERLAY)"; continue; fi
  st=$(grep -m1 -oE '^PROFILE_STATUS=[^ #]*' "$f" | cut -d= -f2); md=$(grep -m1 -oE '^PROFILE_MODEL=[^ #]*' "$f" | cut -d= -f2)
  old=$(grep -c -E 'weg2|SGLANG_WEG2|HTSGLANG_' "$f")
  st=$(eff_status "$p"); ist=$(img_status "$p")
  say "   $p: rig status=$st, image status=$ist$([ "$st" != "$ist" ] && echo ' (GATED in the ctx copy)'), model=${md:-(from base/profile chain)}, old-name lines=$old, file $( [ "$f" = "$HERE/profiles/$p.env" ] && echo profiles/ || echo "${PROFILE_OVERLAY##*/}/ (E4)")"
done
PDATA=()
for p in $PROFILES; do
  f=$(pf "$p"); [ -f "$f" ] || continue
  for dep in $(grep -oE '^[[:space:]]*source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/[A-Za-z0-9._-]+\.env"' "$f" | grep -oE '[A-Za-z0-9._-]+\.env"$' | tr -d '"'); do
    case " $PROFILES " in *" ${dep%.env} "*) ;; *) blocker "profile $p sources $dep, which is not in the whitelist" ;; esac
  done
  for j in "$HERE/profiles/$p".*.json; do [ -f "$j" ] && PDATA+=("$j"); done
done
for j in "${PDATA[@]}"; do say "   data: $(basename "$j")"; done
all=$(ls "$HERE"/profiles/*.env | grep -v bak_ | wc -l); inw=$(echo $PROFILES | wc -w)
say "   excluded: $((all - inw)) arm/experiment profiles (prepare_context.sh copy_profiles would ship all $all -- plan §3.6)"
[ -f "$HERE/profiles/nf.assets" ] && say "   nf.assets: $(grep -cvE '^\s*(#|$)' "$HERE/profiles/nf.assets") rig files (wake-credit reference logs, measured record, planner cache)"

# --- 5. Host (optional) ---------------------------------------------------------------------------------------------
TAG="flliper:${REL}-${CU}"
TAG2="flliper:${CU}-${SHA10}"
sec "5. Build host"
if [ "$HOST" = 1 ]; then
  H=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 proxmox "
    awk '/^MemAvailable:/{printf \"avail=%d\n\", \$2/1048576} /^MemTotal:/{printf \"total=%d\n\", \$2/1048576}' /proc/meminfo
    echo nproc=\$(nproc)
    echo zfs_free=\$(( \$(zfs list -H -p -o avail spinning/docker) / 1073741824 ))
    docker image inspect '$TAG' >/dev/null 2>&1 && echo tag1=exists || echo tag1=free
    docker image inspect '$TAG2' >/dev/null 2>&1 && echo tag2=exists || echo tag2=free
    echo base=\$(docker image inspect '$BASE_TAG' --format '{{index .RepoDigests 0}}' 2>/dev/null || echo missing)
    docker buildx inspect htsglang-build >/dev/null 2>&1 && echo builder=present || echo builder=absent
    echo boots=\$(bash /spinning/subvol-999-disk-0/spinning/gpu-arb/devtools/boot_procs.sh sched launcher launch_server 2>/dev/null | wc -l)
  " 2>/dev/null)
  if [ -z "$H" ]; then warn "host not readable over ssh (BatchMode) -- resource plan uses defaults"; H="avail=0"; fi
  eval "$(printf '%s\n' "$H" | grep -E '^[a-z0-9_]+=[A-Za-z0-9@:._/-]*$')"
  say "   host: MemAvailable ${avail:-?}/${total:-?} GiB, ${nproc:-?} threads, spinning/docker free ${zfs_free:-?} GiB, boots running: ${boots:-?}"
  say "   base $BASE_TAG -> ${base:-?} $([ "${base:-}" = "nvidia/cuda@$BASE_DIGEST_RC8B" ] && echo '(= rc8b digest)' || echo '(!= rc8b digest -- check)')"
  say "   builder htsglang-build: ${builder:-?} (stopped between builds; its cache volume keeps apt/pip layers)"
  [ "${tag1:-}" = exists ] && buildblock "tag $TAG exists on the host -- never overwrite (F14), pick a new release"
  [ "${tag2:-}" = exists ] && buildblock "tag $TAG2 exists on the host"
  [ "${zfs_free:-0}" -ge 150 ] || buildblock "spinning/docker free ${zfs_free:-?} GiB < 150 GiB"
  [ "${boots:-1}" = 0 ] || say "   a boot runs on the host/CT999 (${boots} processes): only --mem-profile nf may build now (checked below)"
else
  say "   (not queried; add --host for MemAvailable, disk, tag collision, boot check)"; avail=${AVAIL_GIB:-83}; nproc=32
fi
# Resource plan, two profiles. Memory model anchored on MEASURED builds (builder cgroup anon, 5-s samples):
#   * light FlashInfer/tvm-ffi job: ~1 GiB worker + ~0.8 GiB per nvcc;
#   * CUTLASS nvcc (gemm_sm120, fp4_gemm_cutlass_sm120): ~6.1 GiB each -- serial build 25.09. 02:44Z (MAX_JOBS=3, one module
#     at a time): anon peak 19.4 GiB; 24g cap with 4 parallel CUTLASS steps: OOM (02:36Z);
#   * rc8b 25.09. (16/6, 10 light modules x 2): anon peak 41.9 GiB, all in the FlashInfer prebuild step; every other step
#     <= 5.4 GiB (prebuild B), export/load 0.1 GiB in the builder.
# The model adds heavy and light peaks (upper bound; rc8b: model 63 vs measured 41.9 GiB).
if [ "$MEMPROF" = auto ]; then if [ "$HOST" = 1 ] && [ "${boots:-0}" != 0 ]; then MEMPROF=nf; else MEMPROF=full; fi; fi
case "$MEMPROF" in
  full)
    FLOOR=20; cap=$(( ${avail:-0} - FLOOR )); [ "$cap" -gt 72 ] && cap=72
    if   [ "$cap" -ge 64 ]; then MJ=16; HJ=6; MOD=10
    elif [ "$cap" -ge 48 ]; then MJ=12; HJ=4; MOD=8
    elif [ "$cap" -ge 32 ]; then MJ=8;  HJ=2; MOD=6
    else MJ=3; HJ=1; MOD=4; buildblock "builder cap ${cap} GiB < 32 GiB (MemAvailable ${avail:-?} - floor $FLOOR) -- free host RAM or use --mem-profile nf"; fi
    LIGHTJ=2; HMODJ=1; DJ=$(( MJ * 3 / 2 )); CPUS=$(( ${nproc:-32} - 2 ))
    OVR="HOST_FLOOR_GIB=$FLOOR"; [ "${boots:-0}" != 0 ] && buildblock "profile full while a boot runs -- wait for the boot to end or use --mem-profile nf" ;;
  nf)
    # cap = MemAvailable - 4 (host_build.sh MARGIN_GIB=4 and HOST_FLOOR_GIB=4) - 1, at most 16g; below 13g the model peak
    # (12.3 GiB) has no room -> refused. host_build.sh re-checks MemAvailable itself at start.
    MJ=4; HJ=1; MOD=2; LIGHTJ=2; HMODJ=1; DJ=4; CPUS=8; FLOOR=4
    cap=$(( ${avail:-0} - 5 )); [ "$cap" -gt 16 ] && cap=16
    [ "$cap" -ge 13 ] || buildblock "profile nf: MemAvailable ${avail:-?} GiB leaves a cap of ${cap}g < 13g (model peak 12.3 GiB) -- wait"
    OVR="HOST_FLOOR_GIB=4 MARGIN_GIB=4 ALLOW_WITH_CT999_BOOT=1" ;;
  *) echo "--mem-profile must be auto|full|nf" >&2; exit 2 ;;
esac
read -r FIPK B6PK PEAK NEED <<< "$(awk -v hm="$HMODJ" -v hj="$HJ" -v m="$MOD" -v l="$LIGHTJ" -v dj="$DJ" 'BEGIN{
  fi=hm*(1+6.1*hj)+m*(1+0.8*l); h6=1+6.1*hj; l6=dj*(1+0.8*2); b6=(h6>l6)?h6:l6; pk=fi; if(b6>pk)pk=b6; if(5.4>pk)pk=5.4;
  printf "%.1f %.1f %.1f %.1f", fi, b6, pk, pk+2 }')"
say "   resource plan, profile $MEMPROF$([ "$MEMPROF" = nf ] && echo ' (next to a running boot)' || echo ' (no boot, full parallelism)'):"
say "     BUILD_MEM=${cap}g BUILD_CPUS=$CPUS MAX_JOBS=$MJ HEAVY_MAX_JOBS=$HJ  |  ctx-baked: PREBUILD_MODULE_JOBS=$MOD PREBUILD_LIGHT_MAX_JOBS=$LIGHTJ PREBUILD_HEAVY_MODULE_JOBS=$HMODJ DELTA_JOBS=$DJ"
say "     model: FlashInfer step ${FIPK} GiB, step 6b ${B6PK} GiB -> builder peak ${PEAK} GiB (cap ${cap}g); host need incl. dockerd/CLI ~${NEED} GiB"
if [ "$MEMPROF" = nf ]; then
  say "     time: FlashInfer prebuild ~45-50 min (gemm_sm120 serial: 29 TUs = 2471 s rig ninja log; light 10510 s / 4 nvcc),"
  say "           step 6b ~25 min (fp4_gemm_cutlass_sm120 17 TUs serial), total ~90-110 min"
  fr=$(awk -v a="${avail:-0}" -v c="$cap" 'BEGIN{print (a-c-2>=2)?1:0}')
  [ "$fr" = 1 ] || buildblock "cap ${cap}g + ~2 GiB dockerd/CLI leaves < 2 GiB of MemAvailable ${avail:-?} GiB -- wait"
  awk -v n="$NEED" 'BEGIN{exit !(n<=30)}' && say "     operator bound: need ${NEED} GiB <= 30 GiB -> allowed next to NF (overrides: $OVR)" \
    || buildblock "host need ${NEED} GiB > 30 GiB operator bound"
else
  say "     time: 40-60 min (rc8b 29 min at 16/6 with a warm cache; FI prebuild 14-18 min, step 6b 5-7 min, export+load ~8 min)"
fi

# --- 6. The context it would create, the build, tag and labels ------------------------------------------------------
CTX=$HERE/ctx/flat-${REL}-${SHA10}-${CU}$([ "$MEMPROF" = nf ] && echo -nf)
sec "6. Context that the write mode would create (NOT created)"
say "   $CTX/"
say "     Dockerfile            <- Dockerfile.flliper (plan §3; B2/B3)"
say "     src-27b/ src-nf/      <- two shallow clones of ${SHA10} (--shallow-since 2026-09-01), clean, with .git"
say "     lock/requirements.lock (pip freeze of $REF_VENV, without sglang-kernel), lock/overrides.txt <- $(basename "$OVERRIDES")"
say "     assets/wheels/        <- $(basename "$KW_FILE") (${KW_SHA:0:12})"
say "     assets/fi-wheels/     <- flashinfer_python 0.7.0 (${FIPY_SHA:0:12}), flashinfer_cubin 0.7.0 (${FICUBIN_SHA:0:12})"
say "     assets/flashinfer_modules.txt (52), assets/tvm-ffi/ (EMPTY), assets/nvidia-open-595/ (EMPTY, WITH_NV_HEADERS=0)"
say "     assets/devtools/ (boot_deadman.sh host_ledger_preflight.sh mem_timeseries.sh), assets/arb-seed-{27b,nf}/, assets/profiles/{27b,nf}/"
say "     tools/{entrypoint.sh,healthcheck.sh,prebuild_jit.py,delta_prebuild.py,delta_postcheck.py}, tools/profiles/ ($(echo $PROFILES | wc -w) whitelisted)"
say "     tools/kernels.txt     <- union list above ($N_ALL lines)"
say "     BUILD_INFO.json (layout flat, release $REL, revision ${SHA10}, pins), BUILD_INFO-{27b,nf}.json, MANIFEST.sha256"
sec "7. Build command (Proxmox host; host_build.sh route with IMAGE= and the flat Dockerfile)"
say "   CTX=$CTX EXPECT_MANIFEST=<sha256 of MANIFEST.sha256> IMAGE=$TAG \\"
say "     BUILD_MEM=${cap}g BUILD_CPUS=$CPUS MAX_JOBS=$MJ HEAVY_MAX_JOBS=$HJ PREBUILD_STRICT=1 $OVR \\"
say "     bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_build.sh --dry-run    # then without --dry-run, via host_build_detached.sh"
say "   then: docker tag $TAG $TAG2   (second, immutable tag)"
sec "8. Tag and labels"
say "   tags: $TAG  and  $TAG2   (registry after the user's Go: ghcr.io/efschu/flliper:<same>) -- no sglang/weg2/htsglang/27b-nf in a tag"
say "   org.opencontainers.image.title=fLLiper version=$REL revision=$REV (plain sha, one tree)"
say "   org.opencontainers.image.source=$SOURCE licenses=Apache-2.0 base.name=<CUDA_BASE digest from host_build.sh>"
say "   io.github.efschu.flliper.placeholders=\"$PLACEHOLDERS\""
say "   io.github.efschu.flliper.slots=\"27b nf\" io.github.efschu.flliper.revision.27b=${SHA10}.. io.github.efschu.flliper.revision.nf=${SHA10}.. io.github.efschu.flliper.push_state=\"$PUSH_STATE\""
say "   io.github.efschu.flliper.cuda=$CU io.github.efschu.flliper.kernel_wheel.sha256=${KW_SHA:0:12}.. io.github.efschu.flliper.flashinfer=0.7.0+2f3bc5ac io.github.efschu.flliper.cute_dsl=4.7.1"
say "   io.github.efschu.flliper.nccl=2.28.9+cuda13.0 io.github.efschu.flliper.driver.expected=595.58.03 io.github.efschu.flliper.build=flat"
if [ "$EMIT" = 1 ]; then sec "kernel list (union, stdout only)"; printf '%s\n' "$UNION"; fi

# --- 7. Verdict -----------------------------------------------------------------------------------------------------
sec "VERDICT"
# Standing blockers that no input can clear (FLAT_IMAGE_PLAN.md §6)
DFF=$HERE/Dockerfile.flliper
if [ ! -f "$DFF" ]; then blocker "B3: Dockerfile.flliper missing (kernel-list step 6b, fLLiper labels, plain revision label) -- plan §3"
else for ph in __FLLIPER_VERSION__ __FLLIPER_SOURCE__ __FLLIPER_PLACEHOLDERS__ __FLLIPER_LOCK_SHA256__; do
       grep -q "=$ph\$" "$DFF" || blocker "Dockerfile.flliper lacks the bake point ARG ...=$ph"; done
     ok "Dockerfile.flliper present ($(grep -c '^RUN ' "$DFF") RUN, bake points 4/4)"; fi
grep -q 'FLLIPER_PROFILE' "$HERE/entrypoint.sh" || relblock "B2: entrypoint.sh reads HTSGLANG_PROFILE only; README promises FLLIPER_PROFILE, MODE=pdflip, /var/lib/flliper -- RENAME step 5"
say "   warnings: ${#WARNS[@]}; blockers: ${#BLOCKERS[@]}; release blockers (ctx allowed): ${#RELBLOCKERS[@]}"
for b in "${BLOCKERS[@]}"; do say "   - $b"; done
for b in "${RELBLOCKERS[@]}"; do say "   - (release) $b"; done
say "   estimate: $([ "$MEMPROF" = nf ] && echo "90-110" || echo "40-60") min on the host (no GPU), image ~34-35 GB uncompressed, ~45-60 layers (no delta chain), builder peak model ${PEAK} GiB (cap ${cap}g)"
if [ "$WRITE" = 0 ]; then [ "${#BLOCKERS[@]}" -gt 0 ] && exit 3; exit 0; fi
if [ "${#BLOCKERS[@]}" -gt 0 ]; then say "WRITE REFUSED: ${#BLOCKERS[@]} ctx blocker(s) above -- nothing was written"; exit 3; fi
# ======================================================================================================================
# WRITE MODE -- creates the context below $HERE/ctx/flat-* and nothing else. No docker, no GPU, no build.
# ======================================================================================================================
OUT=$CTX; PART=$OUT.partial; DONE=0
case "$OUT" in "$HERE"/ctx/flat-*) ;; *) say "WRITE REFUSED: target $OUT is not below $HERE/ctx/flat-*"; exit 3 ;; esac
[ -e "$OUT" ] && { say "WRITE REFUSED: $OUT exists -- a ctx is never overwritten (new --release or revision)"; exit 3; }
[ -e "$PART" ] && { say "WRITE REFUSED: $PART exists (earlier aborted run?) -- look at it and remove it by hand"; exit 3; }
cleanup(){ if [ "$DONE" != 1 ]; then case "$PART" in "$HERE"/ctx/flat-*.partial) rm -rf -- "$PART" ;; esac; fi; }
trap cleanup EXIT
fail(){ say "WRITE FAILED: $* -- partial ctx removed, nothing kept"; exit 3; }
sec "WRITE $OUT"
mkdir -p "$PART"/{lock,assets/wheels,assets/fi-wheels,assets/devtools,assets/tvm-ffi,assets/nvidia-open-595,assets/profiles/27b,assets/profiles/nf,tools/profiles} \
  "$PART"/assets/arb-seed-27b/weg2/calib "$PART"/assets/arb-seed-nf/weg2/calib || fail "mkdir"

say "1/8 tree ${SHA10}: src-27b/ = shallow fetch since $SINCE, src-nf/ = copy (ONE tree, both slots)"
SRCREF=refs/heads/$BRANCH
if ! { "${G[@]}" rev-parse -q --verify "$SRCREF" >/dev/null && "${G[@]}" merge-base --is-ancestor "$REV" "$SRCREF"; }; then SRCREF=refs/remotes/origin/$BRANCH; fi
S=(git -c init.defaultBranch=main -c advice.detachedHead=false -C "$PART/src-27b")
git -c init.defaultBranch=main init -q "$PART/src-27b" || fail "git init"
"${S[@]}" remote add origin "file://$REPO" || fail "remote add"
"${S[@]}" fetch -q --shallow-since="$SINCE" origin "+$SRCREF:refs/remotes/origin/$BRANCH" || fail "fetch $SRCREF from $REPO"
"${S[@]}" cat-file -e "$REV^{commit}" 2>/dev/null || fail "${SHA10} is not within --shallow-since $SINCE of $SRCREF"
"${S[@]}" checkout -q --detach "$REV" || fail "checkout ${SHA10}"
[ "$("${S[@]}" rev-parse HEAD)" = "$REV" ] || fail "HEAD != ${SHA10}"
[ -z "$("${S[@]}" status --porcelain)" ] || fail "src-27b not clean"
"${S[@]}" check-ignore -q "python/$PKG/_version.py" || fail "python/$PKG/_version.py not ignored (Dockerfile step 4 would FATAL)"
cp -a "$PART/src-27b" "$PART/src-nf" || fail "copy src-nf"
[ "$(git -C "$PART/src-nf" rev-parse HEAD)" = "$REV" ] && [ -z "$(git -C "$PART/src-nf" status --porcelain)" ] || fail "src-nf not clean at ${SHA10}"
for f in 27b nf; do grep -q -E '://[^/@]+:[^/@]+@' "$PART/src-$f/.git/config" && fail "credentials in src-$f/.git/config"; done
say "   $("${S[@]}" rev-list --count HEAD) commits (shallow), $("${S[@]}" ls-files | wc -l) files, package $PKG, source ref $SRCREF"

say "2/8 lock (pip freeze of $REF_VENV without sglang-kernel) + overrides"
"$REF_VENV/bin/python" -m pip freeze --exclude-editable > "$PART/lock/venv-freeze.full.txt" || fail "pip freeze"
grep -v -E '^sglang-kernel @ ' "$PART/lock/venv-freeze.full.txt" > "$PART/lock/requirements.lock"
grep -q -E ' @ (file|git)' "$PART/lock/requirements.lock" && fail "lock carries further local/VCS installs"
LOCKW=$(sha256sum "$PART/lock/requirements.lock" | cut -d' ' -f1)
[ "$LOCKW" = "$LOCK_SHA" ] || fail "lock changed between check and write (${LOCK_SHA:0:12} -> ${LOCKW:0:12})"
cp -p "$OVERRIDES" "$PART/lock/overrides.txt"
if [ "$LOCK_FOLD" = 1 ]; then
  # Lock-Fold: jede PyPI-Zeile NAME==VER der Overrides ersetzt/ergaenzt den Lock; flashinfer-python (kommt als geprueftes
  # Wheel in Schritt 2b) und qwen-tts (#466-Uebersetzer, nicht im Release, 6 pip-check-Befunde) gehen raus. In den Overrides
  # bleiben nur die Wheel-Dateien (/tmp/htsglang-fi-wheels/...). Die rohe Freeze bleibt als lock/venv-freeze.full.txt.
  python3 - "$PART/lock/requirements.lock" "$PART/lock/overrides.txt" <<'PY' || fail "lock fold"
import re, sys
lp, op = sys.argv[1], sys.argv[2]
lock = open(lp).read().splitlines(); ovr = open(op).read().splitlines()
name = lambda l: re.split(r"==| @ ", l)[0].strip().lower()
pins = {name(l): l.strip() for l in ovr if l.strip() and not l.lstrip().startswith(("#", "/"))}
bad = [l for l in pins.values() if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", l)]
if bad: sys.exit("lock fold: override line is not NAME==VERSION: %s" % bad)
out, seen, drop = [], set(), {"flashinfer-python", "qwen-tts"}
for l in lock:
    n = name(l)
    if n in drop: continue
    if n in pins: out.append(pins[n]); seen.add(n); continue
    out.append(l)
for n, l in pins.items():
    if n not in seen:
        i = next((k for k, x in enumerate(out) if name(x) > n), len(out)); out.insert(i, l)
open(lp, "w").write("\n".join(out) + "\n")
keep = ["# lock-fold: PyPI pins moved into lock/requirements.lock; only the checked FlashInfer wheels stay here"]
keep += [l for l in ovr if l.lstrip().startswith("/")]
open(op, "w").write("\n".join(keep) + "\n")
print("   lock-fold: %d pins into the lock (%d new), dropped %s" % (len(pins), len(pins) - len(seen), sorted(drop)))
PY
  LOCKW=$(sha256sum "$PART/lock/requirements.lock" | cut -d' ' -f1)
fi
say "   $(wc -l < "$PART/lock/requirements.lock") packages, sha ${LOCKW:0:12}; overrides $(grep -cvE '^\s*(#|$)' "$PART/lock/overrides.txt") pins"

say "3/8 pinned wheels (copied, then hashed again in the ctx)"
cpk(){ cp -p "$1" "$2/" || fail "copy $1"; local h; h=$(sha256sum "$2/$(basename "$1")" | cut -d' ' -f1); [ "$h" = "$3" ] || fail "$(basename "$1") in ctx sha ${h:0:12} != pin ${3:0:12}"; say "   ok $(basename "$1") ${h:0:12}"; }
cpk "$KW_FILE" "$PART/assets/wheels" "$KW_SHA"
cpk "$FIPY_FILE" "$PART/assets/fi-wheels" "$FIPY_SHA"
cpk "$FICUBIN_FILE" "$PART/assets/fi-wheels" "$FICUBIN_SHA"

say "4/8 JIT inputs: FlashInfer module list, kernel list (union), empty tvm-ffi seed, no NV headers"
cp -p "$FIM" "$PART/assets/flashinfer_modules.txt"
echo "empty: no rig seed (cu12 entries; names hash the sources) -- FLAT_IMAGE_PLAN.md section 2" > "$PART/assets/tvm-ffi/.keep"
echo "empty: image built without NV headers (WITH_NV_HEADERS=0, user F6)" > "$PART/assets/nvidia-open-595/.keep"
{ echo "# flat kernel list (make_flat_ctx.sh $(date -u +%FT%TZ)): union of $(for l in "${KLISTS[@]}"; do printf '%s ' "$(basename "$l")"; done)"
  echo "# runs in Dockerfile.flliper step 6b with the 27B slot env; lines with MAX_JOBS= run first and alone (HEAVY_MAX_JOBS)"
  printf '%s\n' "$UNION"; } > "$PART/tools/kernels.txt"
{ echo "# barlink lines again, built into the NF slot's TORCH_EXTENSIONS_DIR (Dockerfile.flliper step 6b)"
  printf '%s\n' "$UNION" | grep -E '^barlink ' || true; } > "$PART/tools/kernels-nf.txt"
say "   kernels.txt $N_ALL lines ($(printf '%s\n' "$UNION" | grep -cE '(^|[[:space:]])MAX_JOBS=' || true) heavy), kernels-nf.txt $(grep -cE '^barlink ' "$PART/tools/kernels-nf.txt" || true) lines"

say "5/8 rig tools, ARB seed per slot, profile data"
for f in boot_deadman.sh host_ledger_preflight.sh mem_timeseries.sh; do cp -p "$ARB/devtools/$f" "$PART/assets/devtools/" || fail "devtools $f"; done
for f in 27b nf; do
  cp -p "$ARB/weg2/PROBE_RING_0907.md" "$PART/assets/arb-seed-$f/weg2/" && cp -p "$ARB"/weg2/calib/*.json "$PART/assets/arb-seed-$f/weg2/calib/" || fail "arb seed $f"
done
cp -p "$ARB/weg2/corridor_budget_sample.json" "$PART/assets/arb-seed-27b/weg2/" || fail "27b corridor sample"
cp -p "$ARB/weg2/census/xchg_census_weg2xsn246_27198a2711.json" "$PART/assets/profiles/27b/" || fail "27b census"
cp -p "$ARB/weg2/census/xchg_census_fnFL2_graph.json" "$ARB/weg2/census/xchg_census_fnFL2_computed.json" \
      "$ARB/weg2/corridor_budget_sample_nextflash_0921.json" "$PART/assets/profiles/nf/" || fail "nf profile data"
for fam in 27b nf; do
  [ -f "$HERE/profiles/$fam.assets" ] || continue
  while IFS= read -r a; do
    a=${a%%#*}; a=$(echo "$a" | xargs); [ -n "$a" ] || continue
    [ -f "$a" ] || fail "profile data missing: $a (profiles/$fam.assets)"
    cp -p "$a" "$PART/assets/profiles/$fam/"
  done < "$HERE/profiles/$fam.assets"
done
say "   seed 27b $(find "$PART/assets/arb-seed-27b" -type f | wc -l) files, nf $(find "$PART/assets/arb-seed-nf" -type f | wc -l); profile data 27b $(ls "$PART/assets/profiles/27b" | wc -l), nf $(ls "$PART/assets/profiles/nf" | wc -l) ($(du -sh --apparent-size "$PART/assets/profiles" | cut -f1))"

say "6/8 tools and the $(echo $PROFILES | wc -w) whitelisted profiles"
cp -p "$HERE/entrypoint.sh" "$HERE/healthcheck.sh" "$HERE/prebuild_jit.py" "$HERE/delta_prebuild.py" "$HERE/delta_postcheck.py" "$PART/tools/" || fail "tools"
for pr in $PROFILES; do
  cp -p "$(pf "$pr")" "$PART/tools/profiles/$pr.env" || fail "profile $pr"
  st=$(eff_status "$pr"); ist=$(img_status "$pr")
  if [ "$st" != "$ist" ]; then
    printf '\n# flat release gate (make_flat_ctx.sh, user 26.09. 17:20Z): first Docker acceptance accepts only %s;\n# this profile ships gated (rig status %s).\nPROFILE_STATUS=%s\n' "$ACCEPTED" "$st" "$ist" >> "$PART/tools/profiles/$pr.env"
    say "   gated: $pr $st -> $ist (ctx copy only)"
  fi
done
for j in "${PDATA[@]}"; do cp -p "$j" "$PART/tools/profiles/"; done
# every /opt/htsglang/profiles/<x> a whitelisted profile names must exist in the image (tools/profiles + assets/profiles)
nref=0
for r in $(grep -ohE '/opt/htsglang/profiles/[A-Za-z0-9_./-]+' "$PART"/tools/profiles/*.env | sort -u); do
  rel=${r#/opt/htsglang/profiles/}; rel=${rel%/}; nref=$((nref + 1))
  [ -e "$PART/tools/profiles/$rel" ] || [ -e "$PART/assets/profiles/$rel" ] || fail "profile reference $r has no file in the ctx"
done
say "   $(ls "$PART"/tools/profiles/*.env | wc -l) profiles + ${#PDATA[@]} data files; $nref /opt/htsglang/profiles references all resolve"

say "7/8 Dockerfile (Dockerfile.flliper, B4 values baked), .dockerignore, BUILD_INFO"
case "$SOURCE" in *[!A-Za-z0-9:/._-]*) fail "--source '$SOURCE' has characters outside [A-Za-z0-9:/._-]" ;; esac
sed -e "s#^ARG FLLIPER_VERSION=__FLLIPER_VERSION__\$#ARG FLLIPER_VERSION=$REL#" \
    -e "s#^ARG FLLIPER_SOURCE=__FLLIPER_SOURCE__\$#ARG FLLIPER_SOURCE=$SOURCE#" \
    -e "s#^ARG FLLIPER_PLACEHOLDERS=__FLLIPER_PLACEHOLDERS__\$#ARG FLLIPER_PLACEHOLDERS=\"$PLACEHOLDERS\"#" \
    -e "s#^ARG FLLIPER_LOCK_SHA256=__FLLIPER_LOCK_SHA256__\$#ARG FLLIPER_LOCK_SHA256=$LOCKW#" \
    "$DFF" > "$PART/Dockerfile"
grep -q '__FLLIPER_' "$PART/Dockerfile" && fail "Dockerfile: unbaked __FLLIPER_ point left"
if [ "$MEMPROF" = nf ]; then   # concurrency next to a running boot (host_build.sh passes only MAX_JOBS / HEAVY_MAX_JOBS)
  sed -i -e "s#^ARG PREBUILD_MODULE_JOBS=10\$#ARG PREBUILD_MODULE_JOBS=$MOD#" -e "s#^ARG PREBUILD_LIGHT_MAX_JOBS=2\$#ARG PREBUILD_LIGHT_MAX_JOBS=$LIGHTJ#" \
         -e "s#^ARG PREBUILD_HEAVY_MODULE_JOBS=1\$#ARG PREBUILD_HEAVY_MODULE_JOBS=$HMODJ#" -e "s#^ARG DELTA_JOBS=\$#ARG DELTA_JOBS=$DJ#" "$PART/Dockerfile"
  for a in "PREBUILD_MODULE_JOBS=$MOD" "PREBUILD_LIGHT_MAX_JOBS=$LIGHTJ" "PREBUILD_HEAVY_MODULE_JOBS=$HMODJ" "DELTA_JOBS=$DJ"; do
    grep -qx "ARG $a" "$PART/Dockerfile" || fail "nf profile: ARG $a not baked"; done
  say "   nf profile baked: PREBUILD_MODULE_JOBS=$MOD LIGHT=$LIGHTJ HEAVY_MODULE_JOBS=$HMODJ DELTA_JOBS=$DJ"
fi
cp -p "$HERE/.dockerignore" "$PART/.dockerignore"
STAGE_B=""
if "${G[@]}" merge-base --is-ancestor bae049a3b4 "$REV" 2>/dev/null; then STAGE_B="ancestor bae049a3b4"
else
  PID_B=$("${G[@]}" show bae049a3b4 | git patch-id --stable | cut -d' ' -f1)
  for c in $("${G[@]}" rev-list --no-merges --since=2026-09-24 "$REV" -- python/sglang/srt/weg2/launcher.py "python/$PKG/srt/$PDF/launcher.py"); do
    if [ "$("${G[@]}" show "$c" | git patch-id --stable | cut -d' ' -f1)" = "$PID_B" ]; then STAGE_B="pick $c (patch-id = bae049a3b4)"; break; fi
  done
fi
[ -n "$STAGE_B" ] || STAGE_B="content seams present (5/5), no pick id found"
DRV=$(awk '/NVRM version/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/){print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)
REL_JSON=$(printf '%s\n' "${RELBLOCKERS[@]}")
PRS=$(for pr in $PROFILES; do printf '%s=%s\n' "$pr" "$(img_status "$pr")"; done)
KL=$(for l in "${KLISTS[@]}"; do basename "$l"; done)
FLAT_LOCK_FOLD=$LOCK_FOLD FLAT_MEMPROF=$MEMPROF python3 - "$PART" "$REL" "$CU" "$REV" "$BRANCH" "$PUSH_STATE" "$SINCE" "$REF_VENV" "$LOCKW" "$KW_SHA" "$(basename "$KW_FILE")" \
          "$DRV" "$STAGE_B" "$PKG" "$SOURCE" "$PLACEHOLDERS" "$TAG" "$TAG2" "$REL_JSON" "$PRS" "$KL" "$FIM" <<'PYW' || fail "BUILD_INFO"
import hashlib, json, os, pathlib, re, sys, time, glob, base64
(out, rel, cu, rev, branch, push, since, venv, lock, kwsha, kwname, drv, stage_b, pkg, source, ph, tag, tag2,
 relb, prs, kl, fim) = sys.argv[1:]
o = pathlib.Path(out); now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
sp = pathlib.Path(venv, "lib/python3.12/site-packages"); lib = sp / "nvidia/nccl/lib/libnccl.so.2"; data = lib.read_bytes()
m = re.search(rb"NCCL version [0-9.]+\+cuda[0-9.]+", data)
rh = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
claims = {}
for rec in sorted(glob.glob(str(sp / "nvidia_nccl_cu1*.dist-info/RECORD"))):
    for row in open(rec):
        if row.startswith("nvidia/nccl/lib/libnccl.so.2,"):
            claims[os.path.basename(os.path.dirname(rec))] = row.split(",")[1] == rh
nccl = {"file": str(lib), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "banner": m.group(0).decode() if m else None, "record_claims_match": claims}
ovr = [l.strip() for l in (o / "lock/overrides.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
fiw = {w.name: hashlib.sha256(w.read_bytes()).hexdigest() for w in sorted((o / "assets/fi-wheels").glob("*.whl"))}
kern = (o / "tools/kernels.txt").read_bytes()
nfi = len([l for l in (o / "assets/flashinfer_modules.txt").read_text().splitlines() if l and not l.startswith("#")])
common = {"revision": rev, "branch": branch, "push_state": push, "shallow_since": since, "created_utc": now,
          "reference_venv": venv, "lock_sha256": lock, "kernel_wheel_sha256": kwsha, "flashinfer": "0.7.0",
          "flashinfer_lock": "none (lock-fold)" if os.environ.get("FLAT_LOCK_FOLD") == "1" else "0.6.14",
          "lock_fold": os.environ.get("FLAT_LOCK_FOLD") == "1", "driver_expected": drv or "595.58.03",
          "nv_headers": {"in_image": False, "describe": "none", "patch_sha256_16": "none"}, "tvm_ffi_seed": False,
          "stage_b": stage_b, "nccl": nccl, "cuda": cu, "package": pkg,
          "base_image": "per --build-arg CUDA_BASE (digest) from host_build.sh"}
for slot in ("27b", "nf"):
    (o / f"BUILD_INFO-{slot}.json").write_text(json.dumps({"line": slot} | common, indent=1))
bi = {"layout": "duo", "build": "flat", "release": rel, "cuda": cu, "created_utc": now,
      "lines": {s: {"revision": rev, "branch": branch, "push_state": push} for s in ("27b", "nf")},
      "one_tree": True, "package": pkg, "lock_sha256": lock, "overrides": ovr,
      "kernel_wheel": {"file": kwname, "sha256": kwsha}, "kernel_wheel_sha256": kwsha, "fi_wheels": fiw,
      "nccl": nccl, "nv_headers": common["nv_headers"], "driver_expected": common["driver_expected"],
      "flashinfer": "0.7.0", "flashinfer_source": "flashinfer-ai/flashinfer 2f3bc5ac (#5242)", "flashinfer_lock": common["flashinfer_lock"], "lock_fold": common["lock_fold"],
      "tvm_ffi_seed": False, "push_state": f"27b: {push}; nf: {push}", "stage_b": stage_b,
      "jit": {"flashinfer_modules": nfi, "flashinfer_modules_source": fim,
              "kernel_list": {"lines": len([l for l in kern.decode().splitlines() if l and not l.startswith("#")]),
                              "sha256": hashlib.sha256(kern).hexdigest(), "sources": kl.split()}},
      "profiles": dict(l.split("=", 1) for l in prs.splitlines() if "=" in l),
      "accepted_first_docker": sorted(k for k, v in dict(l.split("=", 1) for l in prs.splitlines() if "=" in l).items() if v == "abgenommen"),
      "flliper": {"version": rel, "source": source, "placeholders": ph, "tags": [tag, tag2]},
      "mem_profile": os.environ.get("FLAT_MEMPROF", "full"),
      "release_blockers": [l for l in relb.splitlines() if l]}
(o / "BUILD_INFO.json").write_text(json.dumps(bi, indent=1))
PYW
say "   BUILD_INFO: layout duo + build flat, one tree ${SHA10}, stage B: $STAGE_B"

say "8/8 secret scan, manifest"
hits=$(find "$PART" -path "$PART/src-27b" -prune -o -path "$PART/src-nf" -prune -o \( -name '*.adminkey' -o -name 'gpuq_booking.json' \
        -o -name 'GITHUB_PAT*' -o -name '.git-credentials' -o -name '*.pem' -o -name '.netrc' \) -print)
[ -z "$hits" ] || fail "forbidden files in the ctx: $hits"
if grep -rIl -E 'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-or-v1-[a-f0-9]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|admin[-_]api[-_]key[=": ]+[A-Za-z0-9_-]{20,}' \
     "$PART/assets" "$PART/lock" "$PART/tools" "$PART"/BUILD_INFO*.json "$PART/Dockerfile" 2>/dev/null; then fail "key pattern in the ctx (files above)"; fi
( cd "$PART" && find Dockerfile assets lock tools BUILD_INFO.json BUILD_INFO-27b.json BUILD_INFO-nf.json .dockerignore -type f -print0 \
    | sort -z | xargs -0 sha256sum ) > "$PART/MANIFEST.sha256" || fail "manifest"
echo "$(date -u +%FT%TZ) created by make_flat_ctx.sh --write (rev ${SHA10}, release $REL)" > "$PART/MANIFEST.history"
mv -- "$PART" "$OUT" || fail "rename $PART -> $OUT"
DONE=1
MAN=$(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1)
sec "CTX READY (nothing built)"
say "   $OUT  ($(du -sh "$OUT" | cut -f1), $(wc -l < "$OUT/MANIFEST.sha256") manifest entries)"
say "   MANIFEST.sha256 digest: $MAN"
say "   release blockers still open: ${#RELBLOCKERS[@]}$( [ "${#RELBLOCKERS[@]}" -gt 0 ] && echo ' -- a build from this ctx is a PROCEDURE TEST, not a release image')"
say "   build (Proxmox host; profile $MEMPROF: $([ "$MEMPROF" = nf ] && echo "allowed next to the running boot, overrides named" || echo "operator window, no boot running")):"
say "     CTX=$OUT EXPECT_MANIFEST=$MAN IMAGE=$TAG $OVR \\"
say "       BUILD_MEM=${cap}g BUILD_CPUS=$CPUS MAX_JOBS=$MJ HEAVY_MAX_JOBS=$HJ PREBUILD_STRICT=1 \\"
say "       bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_build.sh --dry-run"
exit 0
