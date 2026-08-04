#!/usr/bin/env bash
# MANDATORY END-OF-WINDOW RESTORE.
#
# At the end of the window the INT8-W8A8 serving instance and the translator
# tenant must be back up and the user must find a SERVING system -- not a rig
# left in whatever state the last arm happened to leave it in.
#
# Recipe source: /tmp/w530_boot.sh (read verbatim; this script re-executes it
# rather than reimplementing it, so there is one recipe and not two that drift).
# Arbitration rules: /spinning/gpu-arb/README.md.
#
# ORDER, and why:
#   1. every window server is down and every window sampler is stopped
#   2. the incumbent SERVING holder line is on disk, verbatim
#   3. boot INT8 via /tmp/w530_boot.sh (which claims its OWN holder + heartbeat)
#   4. smoke: /health, then one MT probe through the translator path
#   5. STOP OUR heartbeat -- BEFORE any release. Standing rule, no exceptions.
#   6. rewrite $ARB/holder back to the SERVING line, byte for byte
#
# DESK-WRITTEN, NEVER EXECUTED. `bash -n` only. Do not run any of this until
# the window is actually over.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
. "$HERE/lib.sh"

W530_BOOT="${W530_BOOT:-/tmp/w530_boot.sh}"
INT8_PORT="${INT8_PORT:-30030}"
TRANSLATOR_PORT="${TRANSLATOR_PORT:-30800}"
HOLDER_BACKUP="${HOLDER_BACKUP:-$RUN/holder_serving_original.txt}"

log "=== end-of-window restore ==="

# --- 1. nothing of ours may still hold a card ------------------------------
for arm in 478_iq3xxs 478_q3kxl 470_a_base 470_a_cut 470_b_dspark \
           462_eager 462_f2 462_breakable_clean; do
    [ -e "$RUN/${arm}.pid" ] && stop_server "$arm"
    [ -e "$RUN/ram_${arm}.pid" ] && rammon_stop "$arm"
done

apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
if [ -n "$apps" ]; then
    log "compute processes still present after teardown:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    die "refusing to boot the serving instance on top of a leftover process. Identify it (it may be foreign -- never a broad kill) and clear it, then re-run."
fi

# Corridor check per card BEFORE the serving boot, keyed by UUID.
while IFS=, read -r nvml_idx uuid name free; do
    nvml_idx="${nvml_idx// /}"; uuid="${uuid// /}"; free="${free// /}"
    [ -n "$nvml_idx" ] || continue
    log "post-window: NVML ${nvml_idx} ${name} ${uuid} free=${free} MiB"
done < <(nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits)

# --- 2. the SERVING holder line must be preserved --------------------------
if [ ! -s "$HOLDER_BACKUP" ]; then
    die "no preserved SERVING holder line at $HOLDER_BACKUP. arb_claim() copies it before the window takes the cards; without it the original text cannot be restored VERBATIM, which is what this restore promises. Recover it from $ARB/log or ask before writing an approximation."
fi
log "preserved SERVING holder line:"
cat "$HOLDER_BACKUP" >&2

# --- 2b. stop OUR heartbeat BEFORE booting, not after ----------------------
# Ordering defect found at restore time. w530_boot.sh REFUSES if $ARB/holder is
# younger than 300s ("arb holder is live"), and our heartbeat rewrites it every
# 30s -- so booting the serving instance while our heartbeat still runs is a
# guaranteed refusal. The standing rule is "stop the heartbeat BEFORE
# releasing"; stopping it here satisfies that and unblocks the boot, since the
# release itself is step 6. w530_boot.sh then claims its own holder.
arb_heartbeat_stop

# --- 3. boot INT8 serving via its own recipe -------------------------------
[ -x "$W530_BOOT" ] || [ -r "$W530_BOOT" ] || die "serving recipe not found: $W530_BOOT"
log "booting INT8-W8A8 serving via $W530_BOOT (it claims its own holder + heartbeat)"
# w530_boot.sh refuses rather than displaces, and does its own bounded
# readiness wait (120 x 10s). Its exit code is the boot verdict.
bash "$W530_BOOT"
rc=$?
[ "$rc" -eq 0 ] || die "$W530_BOOT exited ${rc} -- the serving instance is NOT up. Do not release the cards; read /tmp/w530_boot.log."

# --- 4. smoke ---------------------------------------------------------------
log "smoke 1/3: /health"
curl -s -m 10 -o /dev/null -w 'health http=%{http_code}\n' \
    "http://127.0.0.1:${INT8_PORT}/health" >&2 \
    || die "the serving instance does not answer /health on ${INT8_PORT}"

log "smoke 2/3: MT probe through the serving engine (the translator's backend)"
MT_OUT="$RUN/restore_mt_probe.json"
curl -s -m 120 -X POST "http://127.0.0.1:${INT8_PORT}/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen3.6-27B","temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"system","content":"You are a translation engine. Reply with the translation only, no commentary."},{"role":"user","content":"Translate to Spanish: The train leaves at seven in the morning."}]}' \
     -o "$MT_OUT" -w 'mt http=%{http_code}\n' >&2
"$PY" - "$MT_OUT" <<'PYEOF' || die "the MT probe did not return a usable translation -- the translator tenant would be broken"
import json, sys
doc = json.load(open(sys.argv[1]))
msg = doc["choices"][0]["message"]
text = (msg.get("content") or "").strip()
# A thinking model with a small token budget spends the whole budget in
# reasoning_content and returns an EMPTY content with finish_reason "length".
# That is a probe artefact, not a broken engine -- it happened on the first
# restore attempt at max_tokens=64 and would have reported a perfectly healthy
# serving instance as broken. The budget is now 512 with thinking off; if the
# answer still lands in reasoning only, say which failure this is.
if not text and (msg.get("reasoning_content") or "").strip():
    raise SystemExit(
        "PROBE ARTEFACT, not an engine fault: the answer is in reasoning_content "
        f"and content is empty (finish_reason="
        f"{doc['choices'][0].get('finish_reason')!r}). Raise max_tokens or "
        "disable thinking; do not report the engine broken on this alone."
    )
print(f"MT probe answer: {text!r}")
if not text:
    raise SystemExit("empty MT answer")
# Discriminating, not decorative: an untranslated echo means the request went
# through but the engine did not do the job. Any of these Spanish markers is
# enough; their absence together with the English source words is the failure.
low = text.lower()
if "train" in low and "tren" not in low:
    raise SystemExit(f"answer looks untranslated: {text!r}")
PYEOF

log "smoke 3/3: translator front door still listening on ${TRANSLATOR_PORT}"
if curl -s -m 10 -o /dev/null "http://127.0.0.1:${TRANSLATOR_PORT}/metrics" 2>/dev/null; then
    log "translator front door answers"
else
    log "WARNING: the translator front door on ${TRANSLATOR_PORT} did not answer /metrics."
    log "It runs as its own process (scripts/translator/ in /spinning/wt-466-translator);"
    log "the engine restart above does not restart IT. Bring it back and re-run the"
    log "full front-door check: scripts/translator/front_door_test.py --url ws://127.0.0.1:${TRANSLATOR_PORT}"
    log "Do NOT report the window closed until that passes."
fi

# --- 5. heartbeat already stopped in step 2b (see the note there) ----------
arb_heartbeat_stop   # idempotent; the stop file is already in place

# --- 6. restore the SERVING holder line, byte for byte ---------------------
# w530_boot.sh wrote its own holder line in step 3. The user's rig convention
# is that the SERVING line -- the one preserved before the window -- is what
# belongs there afterwards, so it is restored verbatim here and the fact is
# logged rather than silently overwritten.
log "current holder after the serving boot:"
cat "$ARB/holder" >&2
cp -a "$ARB/holder" "$RUN/holder_after_w530_boot.txt"
cp -a "$HOLDER_BACKUP" "$ARB/holder"
printf '%s  %s  RELEASE cards=0,1,2 -- DSV4F window closed, SERVING holder line restored verbatim from %s\n' \
    "$(utc)" "$ARB_SESSION" "$HOLDER_BACKUP" >> "$ARB/log"

log "restored holder:"
cat "$ARB/holder" >&2

log "=== restore complete: INT8 serving up on ${INT8_PORT}, holder back to the SERVING line ==="
log "Remaining manual item if smoke 3/3 warned: restart the translator front door."
