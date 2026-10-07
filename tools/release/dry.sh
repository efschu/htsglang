#!/bin/bash
# dry.sh <profile> <tree> <outlog> <profile-file>  -- UN4/UN6 dry.sh form; own evidence dir + HOME; run via test27b.sh
set -euo pipefail
PROF=$1; WT=$2; OUTLOG=$(realpath -m "$3"); PF=$4
F=${RELEASE_WORK:-/spinning/flliper/work/release}
_form(){ :; }
source "$PF"
# order 910: the entrypoint exports the profile's FORM variables (profile_form_env) into the launcher's env; a dry-run without
# them refuses profiles whose flags need one (e.g. dual: --dual-mps on needs ..._DUAL_MPS_OPT_IN=1 from the form). Same
# semantics as the entrypoint's _form: an explicit value already in the env wins. DRY_FORM_ENV=0 switches this off.
if [ "${DRY_FORM_ENV:-1}" = 1 ] && declare -F profile_form_env >/dev/null; then
  export HTSGLANG_TAG=${HTSGLANG_TAG:-fl7dry${PROF//-/}}   # the entrypoint sets the boot tag before profile_form_env
  _form(){ local var=$1 best=$2; [ -n "${!var+x}" ] || export "$var=$best"; }
  profile_form_env
fi
map() {
  local a=$1
  a=${a//\/opt\/htsglang\/profiles\/27b\/xchg_census_weg2xsn246_27198a2711.json//spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json}
  a=${a//\/opt\/htsglang\/profiles\/nf\/xchg_census_/\/spinning\/gpu-arb\/weg2\/census\/xchg_census_}
  a=${a//\/opt\/htsglang\/profiles\/nf\/corridor_budget_/\/spinning\/gpu-arb\/weg2\/corridor_budget_}
  a=${a//\/opt\/htsglang\/profiles\/nf\/boot_weg2_/\/spinning\/evidence-665-f1\/boot_weg2_}
  printf '%s' "$a"
}
args=(); for a in "${PROFILE_ARGS[@]}"; do args+=("$(map "$a")"); done
for r in "${PROFILE_REQUIRED_PATHS[@]:-}"; do [ -z "$r" ] && continue; r=$(map "$r"); [ -e "$r" ] || { echo "REQUIRED fehlt: $r"; exit 4; }; done
PKG=sglang; [ -d "$WT/python/flliper" ] && PKG=flliper
RUN=$(basename "$OUTLOG" .log)
EV=$F/ev/$RUN; rm -rf "$EV"; mkdir -p "$EV"
# the measured record under its KEPT name (RENAME_PLAN 8.12-1): both trees must read it
[[ "$PROF" == nf* ]] && cp /spinning/evidence-665-f1/weg2_measured_record.json "$EV/weg2_measured_record.json"
H=$F/home/$RUN; rm -rf "$H"; mkdir -p "$H/.cache"; cp -a /root/.cache/sglang "$H/.cache/sglang"
printf 'PROFILE_ARGS(%s):' "$PROF" > "$OUTLOG"; printf ' %q' "${args[@]}" >> "$OUTLOG"; echo >> "$OUTLOG"
cd "$WT"
set +e
# the store lives where the REAL boot has it: the XFS bind /l3/<line> (user rule: XFS, not ZFS); without this the
# launcher falls back to /spinning/hicache-weg2 on the ZFS pool and refuses with W57 on a full pool (FL7 28.09.)
# The live stores (nf/store, 27b/store) are NEVER the dry-run root: plan_store() makedirs+statvfs its root and the
# L3 attach reads it. An empty probe dir on the SAME XFS (only here for the measurement) gives the same statvfs.
SR=${DRY_STORE_ROOT:-/spinning/docker-acceptance/27b/store/rm-dryrun-probe}
# order 910: where the XFS bind is not writable for this user (agent container), the probe dir moves next to the kit's work
# dir (same rule: an EMPTY dir, never a live store). DRY_STORE_ROOT set explicitly always wins.
if [ -z "${DRY_STORE_ROOT:-}" ] && ! mkdir -p "$SR" 2>/dev/null; then SR=$F/store-probe; mkdir -p "$SR"; fi
HOME=$H SGLANG_WEG2_STORE_ROOT=$SR FLLIPER_PDFLIP_STORE_ROOT=$SR PYTHONPATH="$WT/python" CUDA_VISIBLE_DEVICES="" timeout 900 /spinning/htsglang-gpu/.venv/bin/python ${RELEASE_KIT:-$(cd "$(dirname "$0")" && pwd)}/envdump.py $PKG \
  --tree "$WT" --tag fl7dry${PROF//-/} --transport bar1 "${args[@]}" --evidence-dir "$EV" --dry-run >> "$OUTLOG" 2>&1
echo "EXIT=$?" >> "$OUTLOG"
