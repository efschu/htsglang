#!/usr/bin/env bash
# boot_deadman.sh — ONE-SHOT dead-man watcher for a serving boot.
#
# Standing user order 2026-08-30: crashes/hangs MUST be noticed reliably,
# WITHOUT a persistent monitor that burns tokens. This is the sanctioned
# form: a bounded single-event waiter (allowed class per memory
# serving-liveness-monitor) armed once per boot via Bash run_in_background.
# It consumes ZERO model tokens while waiting; its process EXIT (one task
# notification) IS the crash/hang signal. It never emits periodic events.
#
# Detection tiers (the #698 lesson: /health 200 while dead => only a real
# generate proves liveness):
#   1. CRASH: no serving process matches PATTERN            (checked every CHECK_S)
#   2. DEAD-OR-USELESS: /health_generate fails/times out    (probed every PROBE_S,
#      twice in a row before firing — catches hangs AND livelocks that keep
#      logging and keep /health green; costs 1 token per probe)
#   3. FLIP-STALL (#1262): the Weg-2 front's OWN 'WEG2-FLIP STALL' line, read
#      from the same appended chunk tier 2 already reads. Tiers 1 and 2 are
#      STRUCTURALLY BLIND to a livelock BEHIND a healthy front — boot weg2t2a
#      (2026-09-08) held state=flipping for 7 min with six live processes and
#      HTTP 200 on all three ports, and neither tier fired, correctly. A
#      LATCH, because the front emits the line once per stalled flip.
# The verdict line names tier, timestamps, log-mtime age, and last log line.
#
# PASSIVE FIRST (#942 addendum, user 2026-08-30: "warum probt er periodisch?
# das koennte man doch auch viel schlauer machen, dass keine oder so gut wie
# keine 1 token requests mehr kommen"). Tier 2 is SKIPPED whenever the log
# itself already proves organic liveness in the window that just passed, so a
# loaded box emits ~zero synthetic 1-token requests and only an idle box still
# probes. The synthetic probes were not free: on
# /spinning/evidence-665-f1/boot_855_gdncov_0840f82601_0830_042735.log every
# one of them arrived as a sub-chunk prefill in the TP layout and the flip
# policy armed a full ~4 s cutover for it — 55 armings, all below the 4096
# chunk cap, 12 of them for a single token (#942, fixed on the policy side).
#
# WHAT COUNTS AS PROOF, AND WHY IT IS NOT "the log moved" (#698 measured a
# livelock that kept logging while /health stayed 200). Two terms, both read
# off the SAME line and both required:
#   (a) a `Decode batch ... gen throughput (token/s): X` line with X > 0
#       appended since the previous check — tokens were actually EMITTED;
#   (b) that line's `#full token:` counter DIFFERS from the one seen at the
#       previous check — the generation state MOVED. A stuck loop reprinting
#       identical decode lines satisfies (a) and fails (b).
# Log mtime and generic log activity are deliberately NOT used, and neither is
# the mere existence of a line: only appended-since-last-check bytes are read.
#
# THE MARKER IS DERIVED FROM THE LIVE LOG, NOT GUESSED. That boot runs
# `log_requests=False`, so per-request completion lines do not exist in it at
# all: `finish` occurs 377x and trapsafe_count.py scores 0 genuine (every hit
# is the phase policy's own prose — the #995 trap, exactly). `Decode batch ...
# gen throughput (token/s)` occurs 666x, never with a zero value, with the
# token counter advancing (131 -> 5431). It is therefore the strongest
# emitted-token evidence the log actually carries, and the DECISION to use it
# instead of a finish line is recorded here rather than assumed.
#
# VERDICT SEMANTICS UNCHANGED: a skipped probe is positive liveness evidence
# and clears the streak exactly as a successful probe does, so the rule stays
# "two consecutive FAILED probes -> HANG-OR-LIVELOCK, one notification".
#
# 503 IS A REAL ANSWER, NOT A FAILURE (measured 2026-08-31, boot 7: a load
# probe drove /health_generate to 503 twice in a row -> the deadman fired
# HANG-OR-LIVELOCK while 3 schedulers lived and served; `curl -sf` scores 503
# as failure). A 503 proves the HTTP layer and admission path answered — a
# busy box, not a dead one. It is NOT liveness proof either (a starved
# scheduler could 503 forever), so it gets its OWN streak: fire BUSY-STARVED
# only after BUSY_LIMIT consecutive 503s with zero organic decode progress in
# between (passive_alive resets the streak). Hard failures (timeout,
# connection refused, any other code) keep the strict 2x rule.
#
# Usage:
#   boot_deadman.sh LOGFILE [PORT] [PATTERN] [PROBE_S] [CHECK_S] [GRACE_S]
# Arm (from the booting agent, right after launch):
#   Bash(run_in_background): /spinning/gpu-arb/devtools/boot_deadman.sh \
#       /root/current_boot.log 30030
# Smoke (can-fire proof): PROBE_S=2 GRACE_S=0 against a dead port -> fires <10s.
set -u
LOG="${1:?usage: boot_deadman.sh LOGFILE [PORT] [PATTERN] [PROBE_S] [CHECK_S] [GRACE_S]}"
PORT="${2:-30030}"
# FL5 26.09. (RENAME_PLAN 8.13 step 1): both generations, old `sglang::scheduler` and renamed `flliper::scheduler`.
PATTERN="${3:-(sglang|flliper)(::scheduler|\.srt\.entrypoints\.(http_server|openai))|launch_server}"
PROBE_S="${4:-${PROBE_S:-120}}"
CHECK_S="${5:-${CHECK_S:-15}}"
GRACE_S="${6:-${GRACE_S:-240}}"

# pgrep -f matches every process whose ARGV carries the pattern text — this
# script itself, and any shell/timeout wrapper up the ancestor chain (the
# agent-wait-disziplin trap, caught twice by smoke). A real serving process is
# a python/sglang worker, never a shell — so filter matches by comm.
serving_alive() {
  local p comm
  for p in $(pgrep -f "$PATTERN" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    comm=$(cat "/proc/$p/comm" 2>/dev/null) || continue
    case "$comm" in
      bash|sh|dash|zsh|timeout|pgrep|grep|boot_deadman*) continue ;;
      *) return 0 ;;
    esac
  done
  return 1
}

# ---------------------------------------------------------------------------
# #1263 HOIST: THESE THREE ARE TOP-LEVEL, AND THEY MUST STAY THAT WAY.
#
# They were defined INSIDE `scan_admission_alarms`'s body (between its `local`
# line and its closing brace). Bash creates a nested function only when the
# ENCLOSING function RUNS, so until the first passive probe had completed a
# chunk scan, `counter_triple_whole_log`, `line_epoch` and
# `fatal_supersedes_progress` did not exist -- while `verdict` calls
# `counter_triple_whole_log` UNCONDITIONALLY on every branch (#1174 b).
#
# THE CONSEQUENCE, MEASURED rather than assumed (case 8c against the pre-hoist
# file, 2026-09-08), because the obvious guess is wrong and would have set the
# severity wrong: the verdict is NOT lost. Bash prints
# `boot_deadman.sh: line N: counter_triple_whole_log: command not found` to
# stderr, the command substitution yields the empty string, and the verdict line
# still prints -- with its #1174 triple BLANK:
#     DEADMAN[CRASH] ... counters(whole-log,#995-anchored):  log_age=5s ...
# So the failure mode is a deadman that fires correctly while silently dropping
# the evidence #1174 exists to guarantee on EVERY branch, plus one stderr line
# that reads like a fault in the watcher at exactly the moment it is the only
# witness. That is the whole defect, and it is enough.
#
# EXPOSURE, enumerated rather than asserted: `verdict` is the only caller that
# could be reached before the first probe. `fatal_supersedes_progress` runs only
# from `organic_alive`, which calls `passive_alive` first, and `line_epoch` runs
# only from `fatal_supersedes_progress` -- both were therefore always defined by
# the time they were called. Tier 1 (no process matches the pattern) and the
# tier-3 FLIP-STALL branch are the verdicts a boot can reach early; #1262's own
# record named this residual when it added that tier.
#
# It never fired because the ORDER hid it: --selftest exercises
# `scan_admission_alarms` (cases 7/7b/7c) before it reaches the per-verdict
# triple loop (case 8b), and on a real boot the grace wait means a probe
# normally lands first. Order is not a guarantee. Case 8c below drives a
# verdict in a shell where `scan_admission_alarms` has NEVER run, which is the
# only shape that can fail on this.
#
# Definitions only -- no verdict semantics change. The two FATAL_* assignments
# move with `fatal_supersedes_progress` because they are its state: the outer
# `FATAL_DETAIL=""` was redundant (the function clears it at its own top) and
# `FATAL_SCAN_LINES="${FATAL_SCAN_LINES:-40000}"` is idempotent, so hoisting
# them only makes them exist BEFORE the first scan instead of after it.
# ---------------------------------------------------------------------------
# #1174 (b): THE ANCHORED COUNTER TRIPLE BELONGS ON EVERY VERDICT, NOT ONLY THE
# WEDGE ONE. weg1b4 (2026-09-03) died with verdict CRASH and deadman_855_weg1b4
# .out therefore carried NO wedge_alarms=/flip_abandoned=/arm_unfunded= line at
# all, so the #1159 "counters anchored, no prose counted" claim had no specimen
# on the very boot that was supposed to supply one. Scope is the WHOLE LOG,
# because that is trapsafe_count.py's denominator and the comparison is against
# it. Same #995 anchors as scan_admission_alarms -- prose that merely NAMES the
# three markers (http_server.py:1029) must not be counted.
counter_triple_whole_log() {
  local real nw="0" na="0" nu="0"
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  if [ -f "$real" ]; then
    nw=$(grep -cE 'ADMISSION-WEDGE: [0-9]+ queued' "$real" 2>/dev/null) || true
    na=$(grep -cE '\] PHASE-FLIP FLIP ABANDONED' "$real" 2>/dev/null) || true
    nu=$(grep -cE '\] PHASE-POLICY ARM-UNFUNDED:' "$real" 2>/dev/null) || true
  fi
  echo "wedge_alarms=${nw:-0} flip_abandoned=${na:-0} arm_unfunded=${nu:-0}"
}

# Epoch seconds of a log line's OWN '[YYYY-MM-DD HH:MM:SS ...]' prefix.
# Both shapes in the boot log are covered: '[2026-09-03 08:07:50 PP0]' (rank
# lines) and '[2026-09-03T08:08:56Z]' (launcher lines).
line_epoch() {
  local ts
  ts=$(printf '%s\n' "$1" | sed -n 's/^\[\([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\)[T ]\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\).*/\1 \2/p')
  [ -n "$ts" ] || return 1
  date -d "$ts" +%s 2>/dev/null || return 1
}

# #1174 (a): PASSIVE-ALIVE MUST NOT OUTRANK A FATAL LINE.
#
# MEASURED, boot weg1b4 (boot_855_weg1b4_f58a71bde0_0903_080200): PP0 raised
# ObjectRecvStalled at 08:07:50 (log line 90592) and the launcher logged SIGQUIT
# in the same second. At 08:08:29 -- 39 s later -- this watcher printed
# "DEADMAN passive-alive via completion ... organic decode progress in the last
# 120s (#full token=83)". Both of passive_alive's terms were honestly met: the
# window it read still contained the boot's ONLY decode line (08:06:31, 118 s
# old at that instant, the whole decode history of the boot was one print
# event), and its token counter had never been seen before. The verdict "alive"
# was wrong for 74 s, until process absence produced DEADMAN[CRASH] at 08:09:04.
#
# THE MISSING QUESTION, in the same shape as the #1033 admission question: a
# decode line proves that something decoded ONCE, never that it decoded AFTER
# the group started dying. So: the newest forward-progress line must be YOUNGER
# than the newest fatal line. Returns 0 when the fatal line supersedes (=> do
# not claim alive), 1 when it does not.
#
# Timestamps come from the lines' OWN prefixes, not from mtime; append order in
# the file is the fallback when a line carries no parseable prefix, and it is
# also the tie-break inside the same second.
FATAL_DETAIL=""
FATAL_SCAN_LINES="${FATAL_SCAN_LINES:-40000}"
fatal_supersedes_progress() {
  local real buf fhit phit fline pline fno pno fts pts
  FATAL_DETAIL=""
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  [ -f "$real" ] || return 1
  buf=$(tail -n "$FATAL_SCAN_LINES" "$real" 2>/dev/null)
  fhit=$(printf '%s\n' "$buf" | grep -nE 'Scheduler hit an exception|SIGQUIT received|kill_process_tree called' | tail -1)
  [ -n "$fhit" ] || return 1
  fno=${fhit%%:*}; fline=${fhit#*:}
  phit=$(printf '%s\n' "$buf" | grep -nE 'Decode batch|Prefill batch' | tail -1)
  if [ -z "$phit" ]; then
    FATAL_DETAIL="no Decode/Prefill batch line at all in the last ${FATAL_SCAN_LINES} lines, but a fatal line is present: ${fline:0:140}"
    return 0
  fi
  pno=${phit%%:*}; pline=${phit#*:}
  fts=$(line_epoch "$fline") || fts=""
  pts=$(line_epoch "$pline") || pts=""
  if [ -n "$fts" ] && [ -n "$pts" ] && [ "$fts" -ne "$pts" ]; then
    [ "$fts" -gt "$pts" ] || return 1
    FATAL_DETAIL="newest progress line is $(( fts - pts ))s OLDER than the newest fatal line -- progress: ${pline:0:80} | fatal: ${fline:0:120}"
    return 0
  fi
  # same second, or a line without a parseable prefix: append order decides.
  [ "$fno" -gt "$pno" ] || return 1
  FATAL_DETAIL="newest fatal line was appended AFTER the newest progress line -- progress: ${pline:0:80} | fatal: ${fline:0:120}"
  return 0
}

# #1033 posten (a) -- THE TIER THAT WAS MISSING, and why passive-alive alone
# could not supply it.
#
# MEASURED, boot 20 (boot_855_1050dev_0840f82601_0831_115825): from 12:09:56 the
# instance livelocked -- 101075 tokens of prefill queued, no request reaching a
# first token, PP0 arming and abandoning the flip 48 times. The deadman scored
# it ALIVE for 26 minutes and never fired, and lowering BUSY_LIMIT would not
# have changed that by one second: `busy` only counts SYNTHETIC 503s, and no
# synthetic probe ever ran, because ONE resident request kept decoding and
# `passive_alive` therefore kept succeeding on its own terms. Both of its
# conditions were honestly met -- throughput > 0 and #full token moving.
#
# THE HONEST READING: passive-alive proves that SOMETHING DECODES. It has never
# proved that ADMISSION WORKS, and the wedge is exactly the state where those
# two come apart. So this is not a threshold to tune; it is a missing question.
#
# The server already answers it, in its own words, and the deadman simply never
# listened: `ADMISSION-WEDGE: N queued, M running, and NO first token for X s`.
# Scanned from the SAME appended chunk `passive_alive` already reads, so it
# costs no extra I/O and no extra file position.
#
# TRAP-SAFE ANCHORING (#995), and it bites here: the log ALSO carries
# `ADMISSION-WEDGE RECOVERY:` lines, which say the opposite. Measured on boot
# 20: 375 alarm lines against 238 RECOVERY lines. Matching the bare token
# `ADMISSION-WEDGE` would score the recoveries as alarms. The alarm is anchored
# on the COLON, which RECOVERY does not have in that position.
scan_admission_alarms() {
  local chunk="$1" n_wedge n_abandon n_unfunded n_flipstall
  # `grep -c` PRINTS 0 and EXITS 1 on no match, so a `|| echo 0` fallback
  # appends a SECOND zero and the capture becomes the two-line string "0\n0",
  # which `[ -gt ]` then rejects as "integer expression expected". Caught by
  # the self-test below; keep the `|| true`, never an `|| echo`.
  # #1159 TRAP FIRED ON THE ANCHOR THAT WAS NOT THERE (weg1b3, 2026-09-02).
  #
  # The three counters above were anchored against ADMISSION-WEDGE RECOVERY and
  # against nothing else. The log carries a SECOND quoting shape the #995 rule
  # covers and this function did not: the #1033 health-check line
  # (http_server.py:1029) NAMES all three markers in its own prose, in
  # backticks --
  #     "... check the scheduler's own alarms -- `ADMISSION-WEDGE: N queued`,
  #      `PHASE-FLIP FLIP ABANDONED`, `PHASE-POLICY ARM-UNFUNDED` -- which name
  #      the link that actually stopped."
  # Five such lines exist in boot weg1b3 (log lines 2566, 2568, 5446, 69296,
  # 150583), and they are the ENTIRE population of both abandon counters:
  # bare 5 / genuine 0 for each. The deadman's own verdict at 23:58:02 read
  # "flip_abandoned=1 arm_unfunded=1" and NO flip was abandoned and NO arm was
  # unfunded on that boot. A deadman that miscounts its own evidence sends the
  # reader to the wrong link.
  #
  # THE ANCHORS, each keyed on what the GENUINE emitter writes and the prose
  # cannot:
  #   wedge     'ADMISSION-WEDGE: <digits> queued' -- the prose says "N queued".
  #   abandon   '] PHASE-FLIP FLIP ABANDONED' -- LOG_PREFIX at the START of the
  #             message (phase_flip_runtime.py:7111, :11176, :11193); in the
  #             prose the marker is preceded by a backtick, never by "] ".
  #   unfunded  '] PHASE-POLICY ARM-UNFUNDED:' -- the genuine emitter
  #             (scheduler.py:3700) writes a COLON; the prose writes a backtick.
  # Pinned by case 7 of --selftest, which replays the real prose line.
  n_wedge=$(printf '%s\n' "$chunk" | grep -cE 'ADMISSION-WEDGE: [0-9]+ queued') || true
  n_abandon=$(printf '%s\n' "$chunk" | grep -cE '\] PHASE-FLIP FLIP ABANDONED') || true
  n_unfunded=$(printf '%s\n' "$chunk" | grep -cE '\] PHASE-POLICY ARM-UNFUNDED:') || true
  # #1262 TIER 3 -- THE TIER BOTH EXISTING ONES ARE STRUCTURALLY BLIND TO.
  #
  # Boot weg2t2a (2026-09-08) reached its first full three-group READY, logged
  # `WEG2-FLIP begin epoch=0` at 11:57:07Z, and never finished it: all six
  # scheduler ranks sat active+gil inside an idle-time invariant check
  # (read_free_rows, 410 857 rows per pass per rank). Tier 1 saw six healthy
  # processes. Tier 2 got HTTP 200 on all three ports for the whole seven
  # minutes -- correctly, because the FRONT was never wedged, only the groups
  # behind it. Neither tier is at fault and no threshold of theirs would have
  # helped; the question they do not ask is "did a flip that started ever
  # finish". The front answers it in its own words, once per flip, past a
  # bound DERIVED from that boot's own measured flip cost
  # (weg2/front.py::flip_stall_check / _flip_stall_bound_s -- no literal
  # seconds), and this is where the deadman listens.
  #
  # LATCH, not a consecutive-window counter like the wedge tier above: the
  # front emits ONE line per stalled flip (a repeating alarm would be a
  # persistent monitor, which this rig forbids), so a tier that demanded N
  # consecutive windows could never fire on it. Once seen, it stands.
  #
  # TRAP-SAFE ANCHOR (#995): 'WEG2-FLIP STALL epoch=<digit>'. The bare token
  # `WEG2-FLIP STALL` also appears in this file's own prose and in the front's
  # source comments; the genuine emitter is the only writer that follows it
  # with `epoch=` and a digit. Pinned by case 7c of --selftest.
  n_flipstall=$(printf '%s\n' "$chunk" | grep -cE '(WEG2|PDFLIP)-FLIP STALL epoch=[0-9]') || true
  if [ "${n_flipstall:-0}" -gt 0 ]; then
    FLIP_STALL_SEEN=1
    FLIP_STALL_DETAIL=$(printf '%s\n' "$chunk" \
      | grep -E '(WEG2|PDFLIP)-FLIP STALL epoch=[0-9]' | tail -1 | cut -c1-300)
  fi
  # 28.09. (FS-RELEASE): the latch answers "did a flip that started ever
  # finish" -- a LATER '<front>-FLIP done epoch=M' with M > the stalled begin
  # epoch IS that answer, and it is yes: the flip was slow, not stuck. Boot
  # ...noadopts0bar1dauer09281111 (rc12z17): stall line 11:23:15Z (begin
  # epoch=10, 17.8 s > bound 9.9 s), 'done epoch=11' 11:23:16Z (19.0 s), three
  # more flips completed -- the unreleased latch killed a live group 2.5 min
  # later. A flip that never finishes never writes 'done', so a real stall
  # still fires exactly as before. Pinned by case 7d of --selftest.
  if [ "$FLIP_STALL_SEEN" = "1" ]; then
    fs_ep=$(printf '%s\n' "$FLIP_STALL_DETAIL" | grep -oE 'STALL epoch=[0-9]+' | grep -oE '[0-9]+$')
    done_ep=$(printf '%s\n' "$chunk" | grep -oE '(WEG2|PDFLIP)-FLIP done epoch=[0-9]+' \
      | grep -oE '[0-9]+$' | sort -n | tail -1)
    if [ -n "$fs_ep" ] && [ -n "$done_ep" ] && [ "$done_ep" -gt "$fs_ep" ]; then
      echo "DEADMAN FLIP-STALL-RELEASED $(date -Is) port=${PORT:-?} stalled begin epoch=$fs_ep" \
           "completed (done epoch=$done_ep) -- a slow flip, not a stall; latch cleared" \
           "($FLIP_STALL_DETAIL)"
      FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
    fi
  fi

  # #1264 fix 2b (2): THE CONTROLLER'S DEATH, which is EARLIER and more precise
  # than the stall it eventually causes.
  #
  # Boot weg2t2b (2026-09-08): the controller raised at 12:43:04,131Z, 0.6 s
  # into the first flip. It logged `controller error: cannot unpack
  # non-iterable CardFree object` -- a generic handler message carrying no
  # marker any watcher could match -- and continued into a loop whose first
  # statement skips every iteration while `state != "serving"`. The front was
  # dead from that instant; the ONLY thing that ever spoke was tier 3, 123.7 s
  # later, and it named `stage=sleep-kv`, a stage that had COMPLETED before the
  # exception. That one word sent the whole triage down the HiCache drain.
  #
  # So this tier listens for the death itself, in the two shapes the front now
  # emits at that moment: its own named line, and the W4 verdict that follows
  # it. Either one alone is sufficient -- they are one event, and requiring
  # both would make the tier fail if a future edit dropped one.
  #
  # TRAP-SAFE ANCHORS (#995), both needed, because both tokens appear in prose:
  #   * `WEG2-FLIP CONTROLLER-DEAD` is written in this file's own comments and
  #     in weg2/front.py's source comments -> anchor on `epoch=<digit>`, which
  #     only `controller_dead_line()` emits.
  #   * `W4 Weg2WakeRefused` is a bare CODE and appears in prose all over the
  #     front, the launcher and this script -> anchor on the GENUINE EMITTER's
  #     shape, `WEG2 STOP W4 Weg2WakeRefused`, which only `do_stop()` writes
  #     (front.py: logger.error("WEG2 STOP %s -- %s", name, detail)).
  # Pinned by case 7d of --selftest, which replays both prose forms first.
  n_ctldead=$(printf '%s\n' "$chunk" \
    | grep -cE '(WEG2|PDFLIP)-FLIP CONTROLLER-DEAD epoch=[0-9]|(WEG2|PDFLIP) STOP W4 (Weg2|PdFlip)WakeRefused|(WEG2|PDFLIP) STOP W22 (Weg2|PdFlip)HostWatermarkBreached') || true
  if [ "${n_ctldead:-0}" -gt 0 ]; then
    CONTROLLER_DEAD_SEEN=1
    CONTROLLER_DEAD_DETAIL=$(printf '%s\n' "$chunk" \
      | grep -E '(WEG2|PDFLIP)-FLIP CONTROLLER-DEAD epoch=[0-9]|(WEG2|PDFLIP) STOP W4 (Weg2|PdFlip)WakeRefused|(WEG2|PDFLIP) STOP W22 (Weg2|PdFlip)HostWatermarkBreached' \
      | tail -1 | cut -c1-300)
  fi

  if [ "${n_wedge:-0}" -gt 0 ]; then
    WEDGE_IN_CHUNK=1
    # #1174 (b): SCOPE. These three are counted over the CHUNK APPENDED SINCE
    # THE LAST PROBE. trapsafe_count.py counts the WHOLE FILE, so the two were
    # never comparable, and the verdict line now carries a whole-log triple
    # under the bare names (counter_triple_whole_log). To keep exactly ONE
    # occurrence of each bare anchored name per verdict line, the in-window
    # counters are named window_*.
    WEDGE_DETAIL="window_wedge_alarms=$n_wedge window_flip_abandoned=$n_abandon window_arm_unfunded=$n_unfunded"
  else
    WEDGE_IN_CHUNK=0
    WEDGE_DETAIL=""
  fi
}

# Organic-liveness reader. Sets PASSIVE_TOK to the proving line's token counter
# and returns 0 on success; returns 1 otherwise.
# Reads ONLY the bytes appended since the last call — never the whole log, and
# never the file's mtime.
#
# #1033: IT MUST NOT BE CALLED IN A COMMAND SUBSTITUTION, AND IT USED TO BE.
# The shipped form was `if tok=$(passive_alive); then`. `$( )` is a SUBSHELL, so
# every state update this function makes -- `LAST_OFF=$size` and the caller's
# `LAST_TOK=$tok` -- was discarded the instant it returned. Two consequences,
# both silent and both fatal to the watcher's purpose:
#
#   * LAST_OFF never left 0, so every probe re-read the log FROM BYTE 1 under
#     the `head -c 4000000` cap. Once a boot log passes 4 MB (boot 20 reached
#     39.4 MB) the function reads a FIXED EARLY PREFIX for the rest of the boot
#     and reports whatever decode line happened to be in it -- for ever.
#   * LAST_TOK never left "", so the anti-livelock term
#     `[ "$tok" = "$LAST_TOK" ] && return 1` could never fire. Its own comment
#     says "a first observation has no predecessor and passes on (a) alone" --
#     under the subshell EVERY observation was a first observation.
#
# Net effect, and it matches the recorded boot-20 symptom exactly ("last
# passive-alive verdict 12:08:26, wedge from 12:09:56, never synthetically
# probed again"): the deadman scored a livelocked instance alive off a stale
# line, skipped the synthetic probe on every subsequent round, and could not
# fire. This is the ROOT of #1033 detection posten (a) -- deeper than the
# BUSY_LIMIT threshold that posten originally named.
#
# Proof, not inference: `f(){ LAST_OFF=999; }; x=$(f)` leaves LAST_OFF at 0,
# a direct `f` sets it to 999. Pinned by case 5 of --selftest.
LAST_OFF=0
LAST_TOK=""
# #1291: the inode of the log file at the last successful read, so a
# rotation (log path retargeted to a NEW file, e.g. a symlink swap) can be
# told apart from an in-place truncation of the SAME file -- both re-anchor
# at offset 0, but only rotation changes this value. Set by arm_baseline()
# and refreshed on every passive_alive() read.
LAST_INO=""
# #1033 posten (a): what the SAME chunk says about ADMISSION, set as a side
# effect so it costs no extra read. See the STARVED-ADMISSION tier below.
WEDGE_IN_CHUNK=0
WEDGE_DETAIL=""
# #1262 tier 3, set by the SAME chunk read (scan_admission_alarms). A LATCH:
# the front emits one WEG2-FLIP STALL line per stalled flip, so this must not
# be reset per window the way WEDGE_IN_CHUNK is.
FLIP_STALL_SEEN=0
FLIP_STALL_DETAIL=""
# #1264 fix 2b (2), same chunk read, same LATCH discipline: the front emits the
# controller-death line and its W4 verdict ONCE, so a per-window flag would be
# cleared before the next probe ever looked at it.
CONTROLLER_DEAD_SEEN=0
CONTROLLER_DEAD_DETAIL=""
passive_alive() {
  local real size ino chunk line tok thr
  WEDGE_IN_CHUNK=0
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  [ -f "$real" ] || return 1
  size=$(stat -c %s "$real" 2>/dev/null) || return 1
  ino=$(stat -c %i "$real" 2>/dev/null) || ino=""
  # #1291: an inode change means the path now names a DIFFERENT file (a
  # rotation or a symlink retarget) -- named separately from a same-file
  # truncation because #995 wants every re-anchor traceable to its cause,
  # not folded into one silent "size went backwards" branch. Both re-anchor
  # at offset 0: neither a rotated-in file nor a truncated one owes us its
  # pre-anchor history, and re-scanning it from 0 here is the SAME shape as
  # the #1291 bug (a fresh arm re-reading historical lines), just triggered
  # mid-watch instead of at arm time -- so it must land on the new file's
  # byte 0, never replay bytes from before the rotation/truncation.
  if [ -n "$LAST_INO" ] && [ -n "$ino" ] && [ "$ino" != "$LAST_INO" ]; then
    echo "DEADMAN log rotated $(date -Is): inode $LAST_INO -> $ino, re-anchoring at offset 0"
    LAST_OFF=0; LAST_TOK=""
  elif [ "$size" -lt "$LAST_OFF" ]; then
    echo "DEADMAN log truncated $(date -Is): size ${LAST_OFF} -> ${size}, re-anchoring at offset 0"
    LAST_OFF=0; LAST_TOK=""
  fi
  LAST_INO="$ino"
  [ "$size" -gt "$LAST_OFF" ] || return 1
  chunk=$(tail -c "+$(( LAST_OFF + 1 ))" "$real" 2>/dev/null | head -c 4000000)
  LAST_OFF=$size
  scan_admission_alarms "$chunk"
  line=$(printf '%s\n' "$chunk" | grep -F 'gen throughput (token/s):' | tail -1)
  [ -n "$line" ] || return 1
  thr=$(printf '%s\n' "$line" | grep -o 'gen throughput (token/s): [0-9.]*' | grep -o '[0-9.]*$')
  tok=$(printf '%s\n' "$line" | grep -o '#full token: [0-9]*' | grep -o '[0-9]*$')
  [ -n "$thr" ] && [ -n "$tok" ] || return 1
  # (a) tokens were emitted in this window.
  case "$thr" in ''|0|0.0|0.00|0.000) return 1 ;; esac
  # (b) the generation state MOVED since the last check -- the anti-livelock
  # term. A first observation has no predecessor and passes on (a) alone.
  [ "$tok" = "$LAST_TOK" ] && return 1
  # #1033: the state updates live HERE, in the function that owns them, and the
  # result travels in a GLOBAL rather than on stdout -- so no caller can put
  # this behind a `$( )` and silently throw the bookkeeping away again.
  LAST_TOK=$tok
  PASSIVE_TOK=$tok
  return 0
}

# #1174 (a): the composite the main loop applies, factored out so the self-test
# can pin it directly instead of re-implementing the two terms.
ALIVE_WITHHELD=""
organic_alive() {
  ALIVE_WITHHELD=""
  PASSIVE_TOK=""
  passive_alive || return 1
  if fatal_supersedes_progress; then
    ALIVE_WITHHELD="$FATAL_DETAIL"
    return 1
  fi
  return 0
}

# #1231 (S6 slice boot 2026-09-06, weg1b12s6): the group sat wedged-but-alive
# for five minutes after this script said HANG-OR-LIVELOCK, and nobody dumped
# it -- the boot driver was waiting on the acceptance arm's 'O-1 VERDICT:'
# line, which only appears AFTER the death. The deadman is the first process
# that KNOWS the group is wedged, so it takes the py-spy dumps itself, here,
# before any teardown can run. Dumps land beside the log as
# <log>_pyspy_<tier>_<pid>_<hhmmss>.txt. CRASH tier = nothing alive to dump.
PYSPY="${PYSPY:-/spinning/htsglang-gpu/.venv/bin/py-spy}"
dump_live_stacks() {
  local tier="$1" real base p comm out rc n=0
  if [ ! -x "$PYSPY" ]; then
    PYSPY=$(command -v py-spy 2>/dev/null) || { echo "DEADMAN-DUMP[$tier] skipped: py-spy not found"; return 0; }
  fi
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  base="${real%.log}"
  for p in $(pgrep -f "$PATTERN" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    comm=$(cat "/proc/$p/comm" 2>/dev/null) || continue
    case "$comm" in
      bash|sh|dash|zsh|timeout|pgrep|grep|boot_deadman*) continue ;;
    esac
    out="${base}_pyspy_${tier}_${p}_$(date -u +%H%M%S).txt"
    if timeout 30 "$PYSPY" dump --pid "$p" > "$out" 2>&1; then
      n=$(( n + 1 ))
    else
      rc=$?; echo "DEADMAN-DUMP[$tier] pid=$p comm=$comm failed rc=$rc ($out)"
    fi
  done
  echo "DEADMAN-DUMP[$tier] $(date -Is) dumped=$n live stacks -> ${base}_pyspy_${tier}_*.txt"
}

verdict() {
  local tier="$1" detail="$2"
  local age="n/a" last="n/a" real
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  if [ -f "$real" ]; then
    age="$(( $(date +%s) - $(stat -c %Y "$real") ))s"
    last=$(tail -1 "$real" 2>/dev/null | cut -c1-200)
  fi
  # #1174 (b): the anchored whole-log triple rides on EVERY verdict branch.
  echo "DEADMAN[$tier] $(date -Is) port=$PORT $detail" \
       "counters(whole-log,#995-anchored): $(counter_triple_whole_log)" \
       "log_age=$age last_line: $last"
  # #1231: dump the live group on every wedged-but-alive tier, never in the
  # self-test (which calls verdict with detail "self-test" and must not attach).
  if [ "$tier" != "CRASH" ] && [ "$detail" != "self-test" ]; then
    dump_live_stacks "$tier"
  fi
  exit 0
}

grace_wait() {
  local deadline real elapsed=0
  deadline=$(( $(date +%s) + GRACE_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
    if [ -f "$real" ] && grep -qE 'The server is fired up and ready to roll|Uvicorn running on' "$real" 2>/dev/null; then
      GRACE_ENDED_BY="READY"
      return 0
    fi
    sleep 5
  done
  GRACE_ENDED_BY="CEILING"
  return 0
}

# ---------------------------------------------------------------------------
# #1291 ROOT CAUSE, boot weg2sb5f (BOOT_weg2sb5f_0909.md): LAST_OFF started
# at 0 (byte 1 of the log) and was never re-anchored at arm time, so the
# very first passive_alive() call after grace_wait() handed
# scan_admission_alarms() the WHOLE log, including any WEG2-FLIP STALL line
# that had already latched FLIP_STALL_SEEN long before this watcher process
# existed. On sb5f the front stalled ONCE, at 05:46:11 (epoch=12,
# BOOT_weg2sb5f_0909.md line ~4500 of the front log); every re-armed front
# watcher after that instant matched the SAME historical line and fired
# DEADMAN-DUMP[FLIP-STALL] within seconds -- three duplicate dumps, at
# 05:47:15, 05:59:40 and 06:24:39, all quoting the identical epoch=12 event,
# while the front's own genuine stall counter stayed at 1 and the base
# (tier-2 /health_generate on all three ports) stayed 200/200/200 the whole
# time. The front was never actually unhealthy; the watcher was reading its
# own arm as if it were the log's beginning.
#
# FIX: arm_baseline() anchors LAST_OFF (and LAST_INO, for rotation) at the
# log's CURRENT size at arm time, so historical bytes -- anything written
# before this process existed -- are permanently out of scope for every
# tier that reads through passive_alive()'s chunk (that's tiers 2/3/4: the
# organic-alive check, FLIP-STALL, CONTROLLER-DEAD, STARVED-ADMISSION all
# ride the same `chunk`). Only bytes appended AFTER arm are ever scanned.
# Pinned by --selftest cases 9/9b/9c/9d below.
# ---------------------------------------------------------------------------
arm_baseline() {
  local real
  real=$(readlink -f "$LOG" 2>/dev/null || echo "$LOG")
  if [ -f "$real" ]; then
    LAST_OFF=$(stat -c %s "$real" 2>/dev/null) || LAST_OFF=0
    LAST_INO=$(stat -c %i "$real" 2>/dev/null) || LAST_INO=""
    echo "DEADMAN armed $(date -Is) log=$real offset=$LAST_OFF (historical bytes at or before this offset are never scanned by any tier)"
  else
    LAST_OFF=0
    LAST_INO=""
    echo "DEADMAN armed $(date -Is) log=$real does not exist yet, baseline offset=0"
  fi
  LAST_TOK=""
}

# ---------------------------------------------------------------------------
# #1033 posten (c): THE REAL PASSIVE-ALIVE CASE, IN THE SELF-TEST.
# `boot_deadman.sh --selftest` replays the exact line shapes of the boot-20
# livelock instead of a synthetic staircase -- the same lesson
# `unaccounted_watchman.py` learned the hard way (its first version passed its
# synthetic test and read SILENT on the real log). Case 3 is the one that
# matters: decode progress AND the wedge alarm in the same window, which the
# pre-#1033 deadman scored as unambiguously alive.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# #1264 fix 2b: --selftest-wiring, THE END-TO-END HALF.
#
# MEASURED GAP, and it is why this flag exists rather than one more case in
# --selftest: unwiring the CONTROLLER-DEAD verdict branch (replacing its
# condition with a constant false) leaves `--selftest` FULLY GREEN. That suite
# exercises `scan_admission_alarms` -- the pattern half -- and never the main
# loop, so it proves a tier RECOGNISES a line and not that anything HAPPENS
# when it does. "A tool is built only when it is wired", and this is the wire.
#
# So: spawn a process the pattern matches, write a synthetic log carrying the
# READY marker and the front's own death line, run THIS script as a child with
# short timers, and assert the verdict it prints. No GPU, no server, ~10 s.
# ---------------------------------------------------------------------------
if [ "$LOG" = "--selftest-wiring" ]; then
  self="$(readlink -f "$0")"
  tmp="$(mktemp -d)"
  rc=0
  wire_case() {
    local name="$1" inject="$2" want="$3" marker log outfile out kid pid
    marker="deadman-wiring-$$-$RANDOM"
    log="$tmp/$marker.log"
    outfile="$tmp/$marker.out"
    # The READY marker ends the grace at once. #1291: the injected line is no
    # longer baked into the log before the child starts -- arm_baseline now
    # anchors at the log's CURRENT size the moment the child is armed, so a
    # line already sitting in the file at that instant is, correctly,
    # historical and invisible (that is the entire fix this ticket exists
    # for). Append it AFTER the child is running instead, exactly like a real
    # event does relative to a deadman armed hours earlier. A `gen throughput`
    # line is deliberately ABSENT from `inject`: the tier has to fire without
    # any decode progress to lean on.
    echo "[2026-09-08 12:40:00] The server is fired up and ready to roll!" > "$log"
    # A real process for tier 1, matched by a token unique to this case so the
    # pattern can never match the deadman itself or another test's helper.
    sleep 120 2>/dev/null &
    kid=$!
    # PATTERN matches the child by PID via its own /proc cmdline is not
    # possible for `sleep`, so match this shell's helper through a marker file
    # instead: PATTERN is a regex over `pgrep -f`, and `sleep 120` is matched
    # by the literal it was started with. Backgrounded (not `$( )`) so the
    # inject line can land on the log while it is still running.
    GRACE_UNTIL_READY=1 timeout 25 "$self" "$log" 0 "sleep 120" 1 1 5 > "$outfile" 2>&1 &
    pid=$!
    sleep 1  # let arm_baseline (+ the already-satisfied READY grace) land first
    echo "$inject" >> "$log"
    wait "$pid" 2>/dev/null
    out=$(cat "$outfile" 2>/dev/null)
    kill "$kid" 2>/dev/null || true
    wait "$kid" 2>/dev/null || true
    # `want` is the literal the verdict line carries; a leading '!' asserts
    # ABSENCE, which is what makes the healthy control a real control rather
    # than a match on the word DEADMAN that every line carries.
    case "$want" in
      "!"*)
        case "$out" in
          *"${want#!}"*)
            echo "  FAIL $name: '${want#!}' must NOT appear, got:"
            printf '%s\n' "$out" | grep -o "${want#!}" | head -2 | sed 's/^/         /'
            rc=1 ;;
          *) echo "  ok   $name" ;;
        esac ;;
      *)
        case "$out" in
          *"$want"*) echo "  ok   $name" ;;
          *) echo "  FAIL $name: wanted '$want', got:"
             printf '%s\n' "$out" | head -4 | cut -c1-160 | sed 's/^/         /'
             rc=1 ;;
        esac ;;
    esac
  }
  echo "== W1. the CONTROLLER-DEAD line reaches an actual verdict"
  wire_case "front's own death line -> verdict CONTROLLER-DEAD" \
    '[2026-09-08 12:43:04] ERROR weg2.front: WEG2-FLIP CONTROLLER-DEAD epoch=0 stage_last_known=sleep-kv age_s=123.1 exc=TypeError -- the controller loop raised while a flip was OPEN' \
    "DEADMAN[CONTROLLER-DEAD]"
  echo "== W2. do_stop's W4 verdict reaches the same verdict on its own"
  wire_case "WEG2 STOP W22 Weg2HostWatermarkBreached -> verdict CONTROLLER-DEAD" \
    '[2026-09-08 18:02:11] ERROR weg2.front: WEG2 STOP W22 Weg2HostWatermarkBreached -- W22 Weg2HostWatermarkBreached current=96.60 watermark=95.90 margin=11.64' \
    1
  wire_case "WEG2 STOP W4 Weg2WakeRefused -> verdict CONTROLLER-DEAD" \
    '[2026-09-08 12:43:04] ERROR weg2.front: WEG2 STOP W4 Weg2WakeRefused -- unhandled TypeError during the flip of epoch 0' \
    "DEADMAN[CONTROLLER-DEAD]"
  echo "== W2b. FL5 26.09.: the renamed front's death line (PDFLIP) reaches the same verdict"
  wire_case "renamed PDFLIP-FLIP CONTROLLER-DEAD -> verdict CONTROLLER-DEAD" \
    '[2026-09-26 12:43:04] ERROR pdflip.front: PDFLIP-FLIP CONTROLLER-DEAD epoch=0 stage_last_known=sleep-kv age_s=123.1 exc=TypeError -- the controller loop raised while a flip was OPEN' \
    "DEADMAN[CONTROLLER-DEAD]"
  echo "== W3. a healthy log reaches NO tier-3 verdict (the can-fail control)"
  wire_case "an ordinary decode line -> no CONTROLLER-DEAD" \
    '[2026-09-08 12:43:04 PP0] Decode batch phase=tp, #running-req: 1, #full token: 16443, gen throughput (token/s): 40.1' \
    "!CONTROLLER-DEAD"
  rm -rf "$tmp"
  if [ "$rc" = "0" ]; then
    echo; echo "SELFTEST-WIRING PASS -- the tier is wired, not merely present"
  else
    echo; echo "SELFTEST-WIRING FAIL"
  fi
  exit "$rc"
fi

if [ "$LOG" = "--selftest" ]; then
  st_fail=0
  st() {
    if [ "$2" = "$3" ]; then echo "  ok   $1"; else
      echo "  FAIL $1: got '$2' want '$3'"; st_fail=$(( st_fail + 1 )); fi
  }
  DECODE_LINE='[2026-08-31 12:22:20 PP0] Decode batch phase=tp, #running-req: 1, #full token: 16342, full token usage: 0.02, mamba num: 3, mamba usage: 0.30, accept len: 2.80, accept rate: 0.90, cuda graph: True, gen throughput (token/s): 41.2'
  WEDGE_LINE='[2026-08-31 12:22:56 PP0] queue age 37.9s since last first-token progress (perf_counter=2222408.2): ADMISSION-WEDGE: 2 queued, 0 running, and NO first token for 37.9s (>= 20.0s) and no prefill chunk either. Work is waiting'
  RECOVERY_LINE='[2026-08-31 12:20:11 PP0] ADMISSION-WEDGE RECOVERY: request 5cfa9d46 was NOT APPLICABLE -- the wedge cleared on its own'
  ABANDON_LINE='[2026-08-31 12:22:49 PP0] PHASE-FLIP FLIP ABANDONED: tp_to_pp was armed for 30.0s without the group reaching a quiescent boundary (deadline 30s; this rank ready=0).'

  echo "== 1. RECOVERY lines are not alarms (#995 anchoring)"
  scan_admission_alarms "$RECOVERY_LINE"
  st "bare 'ADMISSION-WEDGE' in a RECOVERY line does NOT arm" "$WEDGE_IN_CHUNK" "0"

  echo "== 2. the real alarm arms"
  scan_admission_alarms "$WEDGE_LINE"
  st "'ADMISSION-WEDGE: N queued' arms" "$WEDGE_IN_CHUNK" "1"

  echo "== 3. THE BOOT-20 CASE: decode progress AND the alarm together"
  scan_admission_alarms "$DECODE_LINE
$WEDGE_LINE
$ABANDON_LINE"
  st "alarm arms even with decode progress in the same window" "$WEDGE_IN_CHUNK" "1"
  case "$WEDGE_DETAIL" in
    *flip_abandoned=1*) st "detail names the abandon count" "yes" "yes" ;;
    *) st "detail names the abandon count" "$WEDGE_DETAIL" "flip_abandoned=1" ;;
  esac

  echo "== 4. a healthy window with decode and no alarm does NOT arm"
  scan_admission_alarms "$DECODE_LINE"
  st "healthy decode window" "$WEDGE_IN_CHUNK" "0"

  echo "== 5. passive_alive against a real growing file -- the SUBSHELL defect"
  st_tmp=$(mktemp); LOG="$st_tmp"; LAST_OFF=0; LAST_TOK=""; PASSIVE_TOK=""
  printf '%s\n' "$DECODE_LINE" > "$st_tmp"
  passive_alive && rc=0 || rc=1
  st "first observation is alive" "$rc:$PASSIVE_TOK" "0:16342"
  st "and it set no alarm" "$WEDGE_IN_CHUNK" "0"
  st "LAST_OFF ADVANCED in the caller (the shipped \$( ) form left it 0)" \
     "$([ "$LAST_OFF" -gt 0 ] && echo advanced || echo stuck)" "advanced"
  # the anti-livelock term -- it can only work because LAST_TOK survived
  printf '%s\n' "$DECODE_LINE" >> "$st_tmp"
  passive_alive && rc=0 || rc=1
  st "an unchanged #full token is NOT progress" "$rc" "1"
  # and now the boot-20 window: progress AND alarm, both reported in one pass
  printf '%s\n' "[x] Decode batch phase=tp, #running-req: 1, #full token: 16443, gen throughput (token/s): 40.1" >> "$st_tmp"
  printf '%s\n' "$WEDGE_LINE" >> "$st_tmp"
  passive_alive && rc=0 || rc=1
  st "boot-20 window reads ALIVE ..." "$rc:$PASSIVE_TOK" "0:16443"
  st "... AND raises the admission alarm in the SAME pass" "$WEDGE_IN_CHUNK" "1"
  # the regression that made the watcher blind: a caller using $( ) throws the
  # position away. Pinned as a NEGATIVE so nobody reintroduces the form.
  LAST_OFF=0; LAST_TOK=""
  _ignored=$(passive_alive) || true
  st "a \$( ) caller loses LAST_OFF -- this is the defect, kept visible" \
     "$LAST_OFF" "0"
  rm -f "$st_tmp"

  echo "== 6. grace ends at READY, not at the ceiling (#1033 posten d)"
  st_tmp2=$(mktemp); LOG="$st_tmp2"; GRACE_S=7
  printf '%s\n' "[boot] loading weights" > "$st_tmp2"
  t0=$(date +%s); grace_wait; t1=$(date +%s)
  st "no READY line -> waits to the ceiling (7s here)" \
     "$([ $(( t1 - t0 )) -ge 5 ] && echo waited || echo returned-instantly)" "waited"
  st "and reports the CEILING as the reason" "$GRACE_ENDED_BY" "CEILING"
  printf '%s\n' "[boot] The server is fired up and ready to roll" >> "$st_tmp2"
  t0=$(date +%s); grace_wait; t1=$(date +%s)
  st "READY present -> returns at once" \
     "$([ $(( t1 - t0 )) -le 2 ] && echo fast || echo slow)" "fast"
  st "and reports READY as the reason" "$GRACE_ENDED_BY" "READY"
  rm -f "$st_tmp2"

  echo "== 7. the #1033 health-check PROSE line is not an event (#1159/#995)"
  PROSE_LINE="[2026-09-02 23:52:12] Health check failed. Server couldn't get a response from detokenizer for last 20 seconds. -- #1033 READ THIS AS: NOBODY SENT. Before suspecting the detokenizer, check the scheduler's own alarms -- \`ADMISSION-WEDGE: N queued\`, \`PHASE-FLIP FLIP ABANDONED\`, \`PHASE-POLICY ARM-UNFUNDED\` -- which name the link that actually stopped."
  scan_admission_alarms "$PROSE_LINE"
  st "prose naming all three markers arms NOTHING" "$WEDGE_IN_CHUNK" "0"
  UNFUNDED_LINE='[2026-09-02 23:52:49 PP0] PHASE-POLICY ARM-UNFUNDED: armed tp_to_pp (stall) and the cutover has not committed after 40 rounds'
  scan_admission_alarms "$WEDGE_LINE
$PROSE_LINE
$ABANDON_LINE
$UNFUNDED_LINE"
  st "genuine abandon+unfunded counted once each beside the prose" \
     "$WEDGE_DETAIL" "window_wedge_alarms=1 window_flip_abandoned=1 window_arm_unfunded=1"

  echo "== 7c. #1262 tier 3: the flip-stall latch, and its #995 anchor"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  FS_PROSE='[2026-09-08 12:04:05] this deadman treats a `WEG2-FLIP STALL` line as a tier-3 kill signal'
  scan_admission_alarms "$FS_PROSE"
  st "prose naming WEG2-FLIP STALL arms NOTHING" "$FLIP_STALL_SEEN" "0"
  FS_LINE='[2026-09-08 12:04:05] ERROR weg2.front: WEG2-FLIP STALL epoch=0 elapsed=418.0 s bound=120.0 s stage=drain awake=D queue=1 flips=0 (bound provenance: no flip measured on this boot yet -- the front own published drain deadline (--drain-deadline-s) stands in).'
  scan_admission_alarms "$FS_LINE"
  st "the genuine 'WEG2-FLIP STALL epoch=<digit>' line arms tier 3" "$FLIP_STALL_SEEN" "1"
  case "$FLIP_STALL_DETAIL" in
    *"epoch=0 elapsed=418.0 s"*) st "and the detail carries epoch + elapsed" "ok" "ok" ;;
    *) st "and the detail carries epoch + elapsed" "$FLIP_STALL_DETAIL" "epoch=0 elapsed=418.0 s" ;;
  esac
  # LATCH, not a per-window flag: a later quiet window must not clear it.
  scan_admission_alarms "$DECODE_LINE"
  st "a later quiet window does NOT clear the latch" "$FLIP_STALL_SEEN" "1"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  echo "== 7d. FS-RELEASE: a later 'done' of the stalled flip clears the latch, an older one does not"
  FS_STALL10='[2026-09-28 11:23:15,596] ERROR weg2.front: WEG2-FLIP STALL epoch=10 elapsed=17.8 s bound=9.9 s stage_last_known=gathered-legs age_s=1.2 awake=D queue=3 flips=10'
  FS_DONE10='[2026-09-28 11:22:13,372] INFO weg2.front: WEG2-FLIP done epoch=10 slept=P woke=D drain+quiesce=144 ms'
  FS_DONE11='[2026-09-28 11:23:16,789] INFO weg2.front: WEG2-FLIP done epoch=11 slept=D woke=P drain+quiesce=16552 ms'
  scan_admission_alarms "$FS_STALL10" >/dev/null
  scan_admission_alarms "$FS_DONE10" >/dev/null
  st "an OLDER flip's done (epoch=10) keeps the latch" "$FLIP_STALL_SEEN" "1"
  scan_admission_alarms "$FS_DONE11" >/dev/null
  st "the stalled flip's own done (epoch=11) clears the latch" "$FLIP_STALL_SEEN" "0"
  scan_admission_alarms "$FS_STALL10
$FS_DONE11" >/dev/null
  st "stall + its done in ONE chunk leaves nothing latched" "$FLIP_STALL_SEEN" "0"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""

  echo "== 7d. #1264 fix 2b: the controller's death is its own verdict"
  # RED FIRST, both prose forms: these two tokens appear in this script's own
  # comments and throughout weg2/front.py, so an unanchored pattern would arm
  # the tier off documentation. This is the #995 trap, replayed.
  CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""
  CD_PROSE1='[2026-09-08 12:43:04] this deadman treats a `WEG2-FLIP CONTROLLER-DEAD` line as a tier-3 kill signal'
  scan_admission_alarms "$CD_PROSE1"
  st "prose naming WEG2-FLIP CONTROLLER-DEAD arms NOTHING" "$CONTROLLER_DEAD_SEEN" "0"
  CD_PROSE2='[2026-09-08 12:43:04] the flip refuses with W4 Weg2WakeRefused when a leg fails'
  scan_admission_alarms "$CD_PROSE2"
  st "the bare code W4 Weg2WakeRefused in prose arms NOTHING" "$CONTROLLER_DEAD_SEEN" "0"

  # GREEN: the two shapes the front actually emits at the moment of death.
  # Line 1 is `controller_dead_line()`, verbatim in shape including the fix
  # 2b (1) stage report -- note age_s=123.1 against a sleep RPC that answered
  # in 0.6 s, which is the reading the bare `stage=` spelling hid.
  CD_LINE='[2026-09-08 12:43:04] ERROR weg2.front: WEG2-FLIP CONTROLLER-DEAD epoch=0 stage_last_known=sleep-kv age_s=123.1 exc=TypeError -- the controller loop raised while a flip was OPEN'
  scan_admission_alarms "$CD_LINE"
  st "the genuine CONTROLLER-DEAD line arms the tier" "$CONTROLLER_DEAD_SEEN" "1"
  case "$CONTROLLER_DEAD_DETAIL" in
    *"stage_last_known=sleep-kv age_s=123.1"*|*"exc=TypeError"*)
      st "and the detail carries the stage, its age and the exception" "ok" "ok" ;;
    *) st "and the detail carries the stage, its age and the exception" \
          "$CONTROLLER_DEAD_DETAIL" "stage_last_known=... exc=..." ;;
  esac
  # The latch, same discipline as 7c: one line per death, so a quiet window
  # afterwards must not clear it.
  scan_admission_alarms "$DECODE_LINE"
  st "a later quiet window does NOT clear the CONTROLLER-DEAD latch" "$CONTROLLER_DEAD_SEEN" "1"

  # The SECOND genuine shape, alone: do_stop's W4 verdict. Either shape must be
  # sufficient on its own -- they are one event, and a tier that needed both
  # would go silent if a later edit dropped one of them.
  CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""
  W4_LINE='[2026-09-08 12:43:04] ERROR weg2.front: WEG2 STOP W4 Weg2WakeRefused -- unhandled TypeError during the flip of epoch 0 at stage 0'"'"'sleep-kv'"'"': cannot unpack non-iterable CardFree object'
  scan_admission_alarms "$W4_LINE"
  st "do_stop's 'WEG2 STOP W4 Weg2WakeRefused' alone arms the tier" "$CONTROLLER_DEAD_SEEN" "1"
  CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""

  echo "== 7e. FL5 26.09. rename step 1: the renamed front (pdflip) arms tiers 3 and 3b the same way"
  # RENAME_PLAN 8.13: WEG2 -> PDFLIP, Weg2 -> PdFlip in the tree. The tool must
  # read both generations; prose in the new spelling must still arm nothing.
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  scan_admission_alarms '[2026-09-26 12:04:05] this deadman treats a `PDFLIP-FLIP STALL` line as a tier-3 kill signal'
  st "renamed prose naming PDFLIP-FLIP STALL arms NOTHING" "$FLIP_STALL_SEEN" "0"
  scan_admission_alarms '[2026-09-26 12:04:05] ERROR pdflip.front: PDFLIP-FLIP STALL epoch=3 elapsed=41.0 s bound=27.7 s stage_last_known=drain awake=D queue=1'
  st "the renamed 'PDFLIP-FLIP STALL epoch=<digit>' line arms tier 3" "$FLIP_STALL_SEEN" "1"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""
  scan_admission_alarms '[2026-09-26 12:43:04] this deadman treats a `PDFLIP-FLIP CONTROLLER-DEAD` line as a tier-3 kill signal'
  scan_admission_alarms '[2026-09-26 12:43:04] the flip refuses with W4 PdFlipWakeRefused when a leg fails'
  st "renamed prose (PDFLIP-FLIP CONTROLLER-DEAD, bare W4 PdFlipWakeRefused) arms NOTHING" "$CONTROLLER_DEAD_SEEN" "0"
  for _ln in \
    '[2026-09-26 12:43:04] ERROR pdflip.front: PDFLIP-FLIP CONTROLLER-DEAD epoch=0 stage_last_known=sleep-kv age_s=123.1 exc=TypeError -- the controller loop raised while a flip was OPEN' \
    '[2026-09-26 12:43:04] ERROR pdflip.front: PDFLIP STOP W4 PdFlipWakeRefused -- unhandled TypeError during the flip of epoch 0' \
    '[2026-09-26 18:02:11] ERROR pdflip.front: PDFLIP STOP W22 PdFlipHostWatermarkBreached -- W22 PdFlipHostWatermarkBreached current=96.60 watermark=95.90'; do
    CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""
    scan_admission_alarms "$_ln"
    st "renamed shape arms the tier: ${_ln:40:48}" "$CONTROLLER_DEAD_SEEN" "1"
  done
  CONTROLLER_DEAD_SEEN=0; CONTROLLER_DEAD_DETAIL=""

  echo "== 8. #1174: a fatal line NEWER than the newest progress line is NOT alive"
  # The weg1b4 shape, replayed: the boot's whole decode history is ONE print
  # event, the fatal traceback lands 39 s later, and the pre-#1174 watcher still
  # printed "passive-alive via completion".
  EXC_LINE='[2026-08-31 12:23:05 PP0] Scheduler hit an exception: Traceback (most recent call last):'
  SIGQUIT_LINE='[2026-08-31 12:23:40] SIGQUIT received. signum=None, frame=None. It usually means one child failed.'
  KILLTREE_LINE='[2026-08-31 12:24:10] kill_process_tree called: parent_pid=933103, include_parent=True, pid=933103'
  DECODE_LINE2='[2026-08-31 12:22:40 PP0] Decode batch phase=tp, #running-req: 1, #full token: 16443, gen throughput (token/s): 40.1'
  # 12:24:30 -- after ALL THREE fatal shapes below (12:23:05/12:23:40/12:24:10)
  PREFILL_LINE='[2026-08-31 12:24:30 PP0] Prefill batch phase=pp, #new-seq: 2, #new-token: 3824, #cached-token: 4096'
  st_tmp3=$(mktemp); LOG="$st_tmp3"; LAST_OFF=0; LAST_TOK=""; PASSIVE_TOK=""
  printf '%s\n' "$DECODE_LINE" > "$st_tmp3"
  organic_alive && rc=0 || rc=1
  st "decode-only window reads alive" "$rc:$PASSIVE_TOK" "0:16342"
  st "and withholds nothing" "${ALIVE_WITHHELD:-none}" "none"
  # now: real decode PROGRESS (token moves 16342 -> 16443) and THEN a fatal line
  printf '%s\n' "$DECODE_LINE2" >> "$st_tmp3"
  printf '%s\n' "$EXC_LINE" >> "$st_tmp3"
  organic_alive && rc=0 || rc=1
  st "decode progress OLDER than a 'Scheduler hit an exception' is NOT alive" "$rc" "1"
  case "${ALIVE_WITHHELD:-}" in
    *"Scheduler hit an exception"*) st "and the withheld reason names the fatal line" "named" "named" ;;
    *) st "and the withheld reason names the fatal line" "${ALIVE_WITHHELD:-empty}" "named" ;;
  esac
  st "the age gap is reported (25s here)" \
     "$(case "${ALIVE_WITHHELD:-}" in *"is 25s OLDER"*) echo yes ;; *) echo no ;; esac)" "yes"
  # the same rule for the launcher's two fatal shapes
  printf '%s\n' "$SIGQUIT_LINE" >> "$st_tmp3"
  st "'SIGQUIT received' supersedes too" \
     "$(fatal_supersedes_progress && echo superseded || echo clear)" "superseded"
  printf '%s\n' "$KILLTREE_LINE" >> "$st_tmp3"
  st "'kill_process_tree called' supersedes too" \
     "$(fatal_supersedes_progress && echo superseded || echo clear)" "superseded"
  # and the NEGATIVE: forward progress AFTER the fatal line clears it again, so
  # a restarted/recovering group is not permanently scored dead by this gate.
  printf '%s\n' "$PREFILL_LINE" >> "$st_tmp3"
  st "a Prefill batch NEWER than every fatal line clears the gate" \
     "$(fatal_supersedes_progress && echo superseded || echo clear)" "clear"

  echo "== 8b. #1174: EVERY verdict branch carries the anchored counter triple"
  # one genuine wedge alarm + the #1033 prose that names all three markers
  printf '%s\n' "$WEDGE_LINE" >> "$st_tmp3"
  printf '%s\n' "$PROSE_LINE" >> "$st_tmp3"
  for tier in CRASH HANG-OR-LIVELOCK BUSY-STARVED STARVED-ADMISSION FLIP-STALL CONTROLLER-DEAD; do
    v_out=$(verdict "$tier" "self-test")
    case "$v_out" in
      *"wedge_alarms=1 flip_abandoned=0 arm_unfunded=0"*)
        st "$tier verdict carries the triple (prose not counted)" "ok" "ok" ;;
      *) st "$tier verdict carries the triple (prose not counted)" \
            "$v_out" "wedge_alarms=1 flip_abandoned=0 arm_unfunded=0" ;;
    esac
  done
  printf '%s\n' "$ABANDON_LINE" >> "$st_tmp3"
  printf '%s\n' "$UNFUNDED_LINE" >> "$st_tmp3"
  v_out=$(verdict CRASH "self-test")
  case "$v_out" in
    *"wedge_alarms=1 flip_abandoned=1 arm_unfunded=1"*)
      st "genuine abandon/unfunded reach the CRASH triple" "ok" "ok" ;;
    *) st "genuine abandon/unfunded reach the CRASH triple" "$v_out" \
          "wedge_alarms=1 flip_abandoned=1 arm_unfunded=1" ;;
  esac
  rm -f "$st_tmp3"

  echo "== 8c. #1263: a verdict reached BEFORE the first probe, in a fresh shell"
  # THE ONLY SHAPE THAT CAN FAIL ON THE NESTED-DEFINITION DEFECT, and the reason
  # every case above passed while it was live: cases 7/7b/7c call
  # `scan_admission_alarms`, so by the time case 8b drives `verdict` the three
  # helpers exist. ORDER, not correctness, was carrying them. This case spends a
  # WHOLE SEPARATE PROCESS -- `$0` re-executed -- in which no probe has run and
  # `scan_admission_alarms` has never been called, and drives the one verdict
  # branch reachable from the CLI before any probe: tier 1, with a PATTERN no
  # process can match and GRACE_S=0. `verdict` then calls
  # `counter_triple_whole_log` unconditionally (#1174 b).
  #
  # CRASH is deliberately the tier used: it is the pre-probe branch the argument
  # vector can reach, and it is also the one tier that does NOT call
  # dump_live_stacks, so this case never attaches py-spy to anything.
  st_tmp4=$(mktemp); printf 'a log line\n' > "$st_tmp4"
  # positional: LOG PORT PATTERN PROBE_S CHECK_S GRACE_S
  st_pre=$("$0" "$st_tmp4" 1 '__deadman_1263_pattern_no_process_can_match__' \
                120 5 0 2>&1) || true
  case "$st_pre" in
    *"command not found"*|*"not found"*)
      st "no helper is undefined on the pre-probe verdict path" \
         "$st_pre" "a verdict line with no 'command not found'" ;;
    *"DEADMAN[CRASH]"*"counters(whole-log,#995-anchored): wedge_alarms="*)
      st "the pre-probe verdict prints, with its #1174 triple" "ok" "ok" ;;
    *) st "the pre-probe verdict prints, with its #1174 triple" \
          "$st_pre" "DEADMAN[CRASH] ... counters(whole-log,#995-anchored): wedge_alarms=..." ;;
  esac
  # SECOND READING, of the CAUSE rather than the consequence, because a future
  # edit could re-nest one helper without the pre-probe path happening to reach
  # it. Brace depth, not column: the broken form sat at column 0 too -- it was
  # nested, not indented -- so a `grep '^name()'` would have passed on it.
  st_nested=$(awk '
    /^scan_admission_alarms\(\) \{$/ { d = 1; next }
    d > 0 {
      n = gsub(/\{/, "{"); m = gsub(/\}/, "}"); d += n - m
      if (d <= 0) { d = 0; next }
      if ($0 ~ /^[ \t]*[A-Za-z_][A-Za-z0-9_]*\(\) \{/) { sub(/\(.*/, ""); gsub(/[ \t]/, ""); print }
    }' "$0")
  st "no function is defined inside scan_admission_alarms's body" \
     "${st_nested:-none}" "none"
  rm -f "$st_tmp4"

  echo "== 9. #1291 unit: arm_baseline anchors at the log's CURRENT size, not 0"
  # BOOT_weg2sb5f_0909.md: LAST_OFF started at 0 (and stayed at whatever a
  # previous case left it, in-process), so the FIRST passive_alive() call
  # after a fresh arm scanned the WHOLE log and matched a stall line that had
  # latched long before this watcher existed. arm_baseline must make that
  # impossible BY CONSTRUCTION: the baseline is the file's size at the moment
  # of the call, not the accumulated value of a global nobody reset.
  st9_tmp=$(mktemp)
  printf '%s\n' "$FS_LINE" > "$st9_tmp"     # a genuine stall line, entirely PRE-arm
  LOG="$st9_tmp"; LAST_OFF=999; LAST_TOK="stale"; LAST_INO=""
  pre_size=$(stat -c %s "$st9_tmp")
  arm_baseline >/dev/null
  st "LAST_OFF anchors at the file's current size (not 0, not its old stale value)" \
     "$LAST_OFF" "$pre_size"
  st "LAST_TOK is cleared by arm (no stale predecessor survives an arm)" "$LAST_TOK" ""
  st "LAST_INO is captured at arm time" "$([ -n "$LAST_INO" ] && echo set || echo empty)" "set"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  passive_alive || true
  st "a probe right after arm finds NOTHING new (the pre-arm stall line is invisible)" \
     "$FLIP_STALL_SEEN" "0"
  printf '%s\n' "$FS_LINE" >> "$st9_tmp"    # the SAME shape, but written AFTER arm
  passive_alive || true
  st "a probe after fresh bytes are appended DOES see the tier-3 latch arm" \
     "$FLIP_STALL_SEEN" "1"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  rm -f "$st9_tmp"

  echo "== 9b. #1291 end-to-end: an OLD stall line does not fire on arm; a NEW one does"
  st9b_dir=$(mktemp -d)
  st9b_log="$st9b_dir/boot.log"
  st9b_out="$st9b_dir/deadman.out"
  {
    echo "[2026-09-09 05:00:00] The server is fired up and ready to roll!"
    echo "[2026-09-09 05:46:11] ERROR weg2.front: WEG2-FLIP STALL epoch=12 elapsed=34.7 s bound=27.7 s stage_last_known=drain awake=D queue=1 (HISTORICAL, pre-arm)"
  } > "$st9b_log"
  sleep 60 2>/dev/null &
  st9b_kid=$!
  "$0" "$st9b_log" 0 "sleep 60" 2 2 1 > "$st9b_out" 2>&1 &
  st9b_pid=$!
  sleep 1
  if kill -0 "$st9b_pid" 2>/dev/null && ! grep -q 'DEADMAN\[FLIP-STALL\]' "$st9b_out" 2>/dev/null; then
    st "the historical stall line stays silent through the first poll round" "silent" "silent"
  else
    st "the historical stall line stays silent through the first poll round" \
       "$(kill -0 "$st9b_pid" 2>/dev/null && echo fired-or-crashed || echo exited-early)" "silent"
  fi
  printf '%s\n' "[2026-09-09 05:50:00] ERROR weg2.front: WEG2-FLIP STALL epoch=13 elapsed=41.0 s bound=27.7 s stage_last_known=drain awake=D queue=1 (FRESH, post-arm)" >> "$st9b_log"
  st9b_fired=""
  for _ in 1 2 3 4 5 6; do
    sleep 1
    if grep -q 'DEADMAN\[FLIP-STALL\]' "$st9b_out" 2>/dev/null; then st9b_fired=yes; break; fi
  done
  st "a fresh post-arm stall line DOES fire" "${st9b_fired:-no}" "yes"
  kill "$st9b_pid" 2>/dev/null; wait "$st9b_pid" 2>/dev/null
  kill "$st9b_kid" 2>/dev/null; wait "$st9b_kid" 2>/dev/null
  rm -rf "$st9b_dir"

  echo "== 9c. #1291 end-to-end: truncation re-anchors instead of re-scanning stale bytes"
  st9c_dir=$(mktemp -d)
  st9c_log="$st9c_dir/boot.log"
  st9c_out="$st9c_dir/deadman.out"
  {
    echo "[2026-09-09 06:00:00] The server is fired up and ready to roll!"
    echo "[2026-09-09 06:00:01 PP0] Decode batch phase=tp, #running-req: 1, #full token: 200, gen throughput (token/s): 40.0"
  } > "$st9c_log"
  sleep 60 2>/dev/null &
  st9c_kid=$!
  "$0" "$st9c_log" 0 "sleep 60" 2 2 1 > "$st9c_out" 2>&1 &
  st9c_pid=$!
  sleep 1
  # Truncate AND append a heartbeat in the same breath: a bare truncate-to-
  # empty gives the very next round nothing to call "alive", which would race
  # the curl-based HANG-OR-LIVELOCK path (port 0 always hard-fails) to a false
  # verdict before the re-anchor could ever be proven. The heartbeat is what a
  # real rotated-into log looks like anyway -- it keeps writing.
  : > "$st9c_log"
  printf '%s\n' "[2026-09-09 06:00:05 PP0] Decode batch phase=tp, #running-req: 1, #full token: 300, gen throughput (token/s): 40.0" >> "$st9c_log"
  sleep 3
  st "the truncate is NAMED in the watcher's own output" \
     "$(grep -c 'DEADMAN log truncated' "$st9c_out" 2>/dev/null || echo 0)" "1"
  st "the process survived the truncate (heartbeat kept it alive, fails reset)" \
     "$(kill -0 "$st9c_pid" 2>/dev/null && echo alive || echo dead)" "alive"
  printf '%s\n' "[2026-09-09 06:00:10] ERROR weg2.front: WEG2-FLIP STALL epoch=20 elapsed=50.0 s bound=27.7 s stage_last_known=drain awake=D queue=1 (post-truncate)" >> "$st9c_log"
  st9c_fired=""
  for _ in 1 2 3 4 5 6; do
    sleep 1
    if grep -q 'DEADMAN\[FLIP-STALL\]' "$st9c_out" 2>/dev/null; then st9c_fired=yes; break; fi
  done
  st "a stall line written after a truncate still fires (re-anchor works forward)" "${st9c_fired:-no}" "yes"
  kill "$st9c_pid" 2>/dev/null; wait "$st9c_pid" 2>/dev/null
  kill "$st9c_kid" 2>/dev/null; wait "$st9c_kid" 2>/dev/null
  rm -rf "$st9c_dir"

  echo "== 9d. #1291 unit: an inode change (rotation) is named and re-anchors at 0"
  st9d_dir=$(mktemp -d)
  st9d_a="$st9d_dir/a.log"
  st9d_b="$st9d_dir/b.log"
  st9d_msg="$st9d_dir/msg.txt"
  printf '%s\n' "$DECODE_LINE" > "$st9d_a"
  printf '%s\n%s\n' "[2026-09-09 07:00:00] The server is fired up and ready to roll!" "$FS_LINE" > "$st9d_b"
  LOG="$st9d_a"; LAST_OFF=0; LAST_TOK=""; LAST_INO=""
  arm_baseline >/dev/null
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  LOG="$st9d_b"   # simulate a symlink retarget to a DIFFERENT file (new inode)
  passive_alive > "$st9d_msg" 2>&1 || true
  st "an inode change is named in the output" \
     "$(grep -c 'DEADMAN log rotated' "$st9d_msg")" "1"
  st "the rotation re-anchors at 0, so the retargeted file's own stall line is scanned" \
     "$FLIP_STALL_SEEN" "1"
  FLIP_STALL_SEEN=0; FLIP_STALL_DETAIL=""
  rm -rf "$st9d_dir"

  echo
  if [ "$st_fail" -gt 0 ]; then echo "SELFTEST FAILED ($st_fail)"; exit 1; fi
  echo "SELFTEST PASS -- incl. the real boot-20 passive-alive case (#1033 posten c) and the #1291 arm-at-end/rotation cases (9/9b/9c/9d)"
  exit 0
fi

# #1033 posten (d): THE GRACE IS A CEILING, NOT A SCHEDULE.
#
# `sleep "$GRACE_S"` waited the FULL grace unconditionally, so a boot that died
# INSIDE the grace was invisible to its own watcher. Measured 2026-08-31, boot
# 21: READY at 13:00:35Z after 191 s, death at 13:02:14Z -- and the deadman,
# armed 12:57:24Z with the MANDATORY GRACE_S=600, would not have begun judging
# until 13:07:24Z, five minutes after the process was gone. It reported nothing
# and the death was found by hand.
#
# The two requirements are not in conflict once the grace is treated as an
# upper bound: 600 s is mandatory because READY can take ~427 s (barlink JIT),
# but the moment the log says READY the boot no longer needs protecting from a
# premature verdict. So: wait for READY *or* the ceiling, whichever comes
# first. On a boot that never prints READY this is byte-identical to the old
# behaviour -- it waits the whole GRACE_S -- so nothing regresses.
#
# #1291: arm THE MOMENT this process is alive, before the grace wait -- not
# after it. Baselining post-grace would fold anything written DURING grace
# (a boot that stalls before it ever prints READY, say) into "historical",
# permanently invisible to every tier once judging starts. Baselining here
# means only bytes written before THIS PROCESS EXISTED are ever out of
# scope; anything appended from this instant on -- grace included -- is live.
arm_baseline

# The marker is the server's own readiness line; `serving_alive` is NOT used as
# the exit condition, because the process exists long before it can serve.
GRACE_ENDED_BY="CEILING"
if [ "${GRACE_UNTIL_READY:-1}" = "1" ]; then
  grace_wait
  echo "DEADMAN grace ended by $GRACE_ENDED_BY $(date -Is) port=$PORT" \
       "(ceiling was ${GRACE_S}s) -- judging starts now"
else
  sleep "$GRACE_S"  # let the boot reach serving before judging it
fi

fails=0
busy=0
# #1033 posten (a): 10 consecutive 503s at PROBE_S=120 is 20 minutes of a box
# that answers but never decodes. `busy` is ALREADY the zero-organic-progress
# streak by construction -- both a 200 and a passive-alive reset it -- so the
# lower number needs no second counter, only an honest default. 3 x 120 s = 6
# minutes of answered-but-starving, which is past any legitimate flip
# (measured max flip leg ~5.2 s) and well inside a wedge.
BUSY_LIMIT="${BUSY_LIMIT:-3}"
wedge=0
WEDGE_LIMIT="${WEDGE_LIMIT:-3}"
last_probe=0
while true; do
  # Tier 1: crash/exit — no serving process left (pattern excludes the router
  # by construction and this waiter itself via serving_alive).
  if ! serving_alive; then
    sleep 5  # debounce against a restart-in-progress
    serving_alive || verdict CRASH "no process matches pattern"
  fi
  # Tier 2: real-generate probe — the only honest liveness signal.
  now=$(date +%s)
  if [ $(( now - last_probe )) -ge "$PROBE_S" ]; then
    last_probe=$now
    # PASSIVE FIRST: if the box served real work in the window that just
    # passed, do not manufacture a 1-token request to ask whether it is alive.
    # #1033: DIRECT call, never `$( )` -- see the function's own note.
    # #1174 (a): organic_alive = passive_alive AND no fatal line newer than the
    # newest forward-progress line. passive_alive is still called DIRECTLY (one
    # frame down, no `$( )`), so its LAST_OFF/LAST_TOK bookkeeping survives.
    if organic_alive; then alive=1; tok=$PASSIVE_TOK; else alive=0; tok=""; fi
    if [ -n "$ALIVE_WITHHELD" ]; then
      echo "DEADMAN passive-alive WITHHELD $(date -Is) port=$PORT --" \
           "the log proves decode, but not decode SINCE the death: $ALIVE_WITHHELD" \
           "(#1174; weg1b4 declared alive 39s after PP0's fatal traceback)"
    fi
    # #1033 posten (a): the ADMISSION question, asked BEFORE the liveness
    # question and independently of its answer. This tier MUST be able to fire
    # while passive-alive keeps succeeding -- that IS the boot-20 shape, where
    # one resident request decoded happily while 101075 tokens starved behind a
    # flip that armed and abandoned 48 times.
    # #1262 TIER 3: FLIP STALL. Asked before the liveness question and
    # independently of its answer, exactly like the admission tier -- and for
    # the same reason, one step further out: on weg2t2a passive-alive would
    # also have kept succeeding if anything had been decoding, and the front
    # answered 200 throughout regardless. A latched signal, so no consecutive-
    # window count: the front says it once, past its own derived bound.
    # #1264 fix 2b (2): CONTROLLER-DEAD IS ASKED FIRST, ahead of FLIP-STALL.
    # Not a preference -- an ordering by evidence. The death is the CAUSE and
    # lands ~123 s before the stall it produces; the stall line names only the
    # last stage ENTERED, which on weg2t2b was a stage that had already
    # completed. Whoever reads the verdict must be handed the earlier, exact
    # signal, not the later, ambiguous one. `verdict` does not return, so this
    # order is what decides which of the two a reader ever sees.
    if [ "$CONTROLLER_DEAD_SEEN" = "1" ]; then
      verdict CONTROLLER-DEAD \
        "the Weg-2 front's controller loop raised while a flip was OPEN ($CONTROLLER_DEAD_DETAIL). The flip took none of its named exits, so 'state' stays 'flipping' and the loop's own guard skips every later iteration -- the front is alive and doing nothing, permanently. VRAM occupancy is UNDEFINED on both groups: the source's kv_cache (and possibly part of its weights family) is paused and the destination was never resumed. NOTE: tiers 1 and 2 are STRUCTURALLY BLIND here and so is tier 3 for the first ~2 min -- every process is alive, /health answers 200 on all three ports, and the stall timer only speaks after 4x the derived flip bound (boot weg2t2b, 2026-09-08: death at 12:43:04Z, first and only alarm at 12:45:07Z, naming a stage that had completed). Read stage_last_known + age_s on the CONTROLLER-DEAD line, and the exception name; no retry, recovery = teardown + relaunch"
    fi
    if [ "$FLIP_STALL_SEEN" = "1" ]; then
      verdict FLIP-STALL \
        "the front's OWN flip-stall detector fired ($FLIP_STALL_DETAIL). A flip began and did not complete inside a bound derived from this boot's own measured flip cost. NOTE: tier 1 and tier 2 are STRUCTURALLY BLIND here -- the processes are alive and /health_generate answers 200 while the schedulers spin behind the front (boot weg2t2a, 2026-09-08: 7 min, all three ports 200). Look at the scheduler ranks' stacks (dumped below), not at the front"
    fi
    if [ "$WEDGE_IN_CHUNK" = "1" ]; then
      wedge=$(( wedge + 1 ))
      [ "$wedge" -ge "$WEDGE_LIMIT" ] && verdict STARVED-ADMISSION \
        "the server's OWN admission alarm fired in ${wedge} consecutive ${PROBE_S}s windows ($WEDGE_DETAIL). Requests are queued and reaching no first token. NOTE: decode may still be progressing -- 'passive-alive' proves something DECODES, never that ADMISSION works, and a health probe answers 200/503 either way. Look at the scheduler's arm/abandon cycle, not at the detokenizer"
    else
      wedge=0
    fi
    if [ "$alive" = "1" ]; then
      fails=0
      busy=0
      echo "DEADMAN passive-alive via completion $(date -Is) port=$PORT" \
           "organic decode progress in the last ${PROBE_S}s (#full token=$tok," \
           "gen throughput > 0) -- synthetic /health_generate probe SKIPPED (#942)." \
           "This says SOMETHING DECODES; it does NOT say admission works (#1033)."
      sleep "$CHECK_S"
      continue
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 25 \
           "http://127.0.0.1:${PORT}/health_generate" 2>/dev/null)
    if [ "$code" = "200" ]; then
      fails=0
      busy=0
    elif [ "$code" = "503" ]; then
      # Busy answer from a live server: resets the hard-fail streak (the box
      # ANSWERED), counts toward its own slow BUSY-STARVED verdict instead.
      fails=0
      busy=$(( busy + 1 ))
      [ "$busy" -ge "$BUSY_LIMIT" ] && verdict BUSY-STARVED \
        "health_generate answered 503 ${busy}x consecutively with no organic decode progress between probes"
    else
      fails=$(( fails + 1 ))
      [ "$fails" -ge 2 ] && verdict HANG-OR-LIVELOCK "health_generate failed ${fails}x (m=25s each, last code=${code:-none})"
    fi
  fi
  sleep "$CHECK_S"
done
