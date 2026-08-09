# #656 HANDOFF v18 — successor 15

Written 2026-08-09, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this, then `phase_purity.py`'s module docstring (it carries the user
rule and the deadlock argument), then HANDOFF_657 for corpses A–K. Do not
re-walk any corpse.

---

## 1. What this shift changed

**HEAD `2ce40f86a2`** — "[#631] Strict phase purity: no decode in PP, no
prefilled token in TP". Suite `bash scripts/run_631_flip_family.sh` →
**606 passed** (was 583; +23 from the new purity file).

The shift began on the spill ladder and was preempted twice by the
operator: first with defect K (decode running in the PP layout), then with
the user's **strict-purity order**, which fixed the design. Everything
below serves that order.

### New: `--phase-flip-purity {strict|threshold:<n>|off}`

Default **strict**. Two prohibitions, both enforced in
`Scheduler.get_next_batch_to_run`:

- no decode step executes in the PP layout
  (`phase_decode_blocked_here` → the decode branch returns None),
- not a single token is prefilled in the TP layout
  (`phase_prefill_blocked_here` → the prefill batch is never BUILT).

`threshold:<n>` is the escape hatch for decode-in-PP only. It deliberately
does NOT relax prefill-in-TP: that direction is a 4.3x throughput loss with
no latency argument on the other side. `off` restores the old interleaving
for A/B. Invalid values are a loud argument error.

### New: the two policy windows (`phase_policy.py`)

Purity alone deadlocks. The load-triggered rules could not schedule the
alternation, so:

- `pp_window_s` (default **15 s**, `SGLANG_PHASE_POLICY_PP_WINDOW_S`) —
  max continuous time in PP while decode work waits. On expiry PP→TP arms
  REGARDLESS of backlog.
- `tp_decode_floor_s` (default **10 s**,
  `SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S`) — minimum time in TP before a
  backlog-driven TP→PP may arm. Without it the always-present backlog
  would drag the instance out of TP after one min_dwell (3 s) and starve
  decode in the mirror image of defect K.

Both apply only while the other side has work. `min_dwell` was NOT touched:
it gates how SOON a flip may arm, never how LONG a phase may keep the other
side waiting — that distinction is the whole fix.

**Boot-time refusal**: purity enforced + `pp_window_s == 0` raises
(`validate_purity_policy_pair`). Under purity a PP phase that may not
decode and cannot admit prefill (no free state slot) has no other exit.

### `phase_evidence_extract.sh` now JUDGES

It asserts zero decode records in PP, zero prefilled tokens in TP, and that
**both layouts were actually visited** — a starved instance that never left
PP would otherwise pass a purity check trivially. Exits non-zero on
violation; `PHASE_PURITY_ASSERT=0` to report only.

---

## 2. Metal status — READ THIS BEFORE CLAIMING ANYTHING

| Claim | Status |
|---|---|
| Purity gates compile, parse, arm on all 3 ranks | **PROVEN** (log 21:37:10Z, all of PP0/PP1/PP2) |
| Pre-purity build violates purity | **PROVEN**: 581 decode records in PP, 10266 tokens prefilled in TP (`evidence-631/phase_evidence_prepurity_defectK.txt`) — this is also the proof the new gate CAN fail |
| Purity HOLDS under load | **held for 90 s, then the instance wedged on DEFECT M (§4b)** — purity was not the cause, but the run is NOT green |
| Both queues drain in bounded windows | observed working (both directions, floor and window both fired) for 90 s |
| Spill ladder | **NOT IMPLEMENTED** — module written, no rung wired |

### FIRST LIVE SIGNAL on the purity build (boot 21:43:45Z, b55b34ba73)

Measured 21:46:38Z, ~30 s into the green run, on the window log only:

```
  DECODE records
    in TP:      4 records, cuda graph True on 4 (100.0%), mean accept len 2.96

  STRICT PHASE PURITY VERDICT
    ok: no decode record executed in the PP layout
    ok: not a single token prefilled in the TP layout
    => PURITY HELD, both layouts used
```

6 flips inside the first 30 s, both directions. Against the pre-purity
baseline this is an exact inversion:

| | pre-purity | purity build |
|---|---|---|
| decode records in PP | 581 | **0** |
| tokens prefilled in TP | 10266 | **0** |
| CUDA graphs on decode | 0 % | **100 %** |
| mean accept len | 0.00 | **2.96** |

**This is 30 seconds, not 60 minutes.** The standing law of this task is
that a soak green at 4 minutes means nothing (corpse J appeared at minute
5, corpse L at minute 2 of the previous boot). The verdict that counts is
written by the green run at ~22:51Z into
`/spinning/evidence-631/green_20260809T214342Z.verdict.txt`; treat
anything before that as a first signal only.

Pool on this boot: `max_total_num_tokens` **263768** at ctx 262144
(RANK_MIB 22700,11920,11970) — well below the >600k target, and §4 explains
why the spill ladder cannot close that gap without a real runtime grow.

### KNOWN CONSEQUENCE: /health flaps 200 <-> 503 under purity

Expected, not a wedge. The health probe is an ordinary generate: its
prefill runs in a PP window and its decode is DEFERRED to the next TP
window, so a probe issued mid-PP-window can exceed the endpoint's own
timeout and return 503 (and can log "Health check failed. Server couldn't
get a response from detokenizer"). Judge liveness by the soak's advancing
`ok=` counter, not by a single probe. Anything monitoring 30030 with a
strict health gate needs a timeout longer than one PP window (15 s).

Distinguishing this from the real corpse-L wedge: there, health NEVER
recovered and the policy logged the same "holding in tp: pending prefill 1
tok" line every 10 s with `running bs 0` forever. Here the counters move.

### The pre-purity baseline, for comparison

Serving log archived at `evidence-631/serving_prepurity_defectK.log`, soak
at `soak_prepurity_defectK.log` (ok=34 err=0 at T+28, 0 exceptions,
0 deaths), corridor at `corridor_prepurity_defectK.csv`.

---

## 3. Diagnosis of the two priority-0 suspects (both closed)

**(b) "prefill wedged at 302757 tokens" — NOT a wedge.** Prefill drains in
PP at 4200-4500 tok/s; pending fell 302757 → 185072 within a minute of the
plateau ending. The plateau coincided with a decode-dominated stretch: in
that 60 s window the log holds **87 decode batch records against 6 prefill
records**. The PP layout was spending its rounds on decode. No lock, no
new defect from the recent commits — and strict purity removes the
interleaving that produced it.

**(a) starvation — REAL, fixed above.** `phase_policy.py:341` gated PP→TP
on `pending <= N`, which a sustained backlog never satisfies.

**Contamination removed**: a second soak driver from the previous shift
(PID 2483233, started 20:54:27Z, `err=394`) was still injecting 12000-token
prompts every 4 s alongside the current one — double artificial prefill
load, and the reason the backlog looked unconditional. Killed by PID.

---

## 4. THE SPILL LADDER — a hard negative you must not re-derive

`phase_flip_spill.py` is written (rung 1 = draft weights, reusing
`snapshot_and_free` + `bind_arena_views` + `arena_refill`, with a
`SGLANG_PHASE_FLIP_SPILL_VERIFY` falsifier for its immutability
assumption). **It is not wired into the cutover, and wiring it as designed
would NOT raise `max_total_num_tokens`.**

Traced this shift, with citations:

- `/get_server_info.max_total_num_tokens` is the **PP pool**, min-reduced
  across ranks, and it is a **frozen boot snapshot**
  (`scheduler.py:2109` ← `tp_worker.get_worker_info()`; `http_server.py:238`).
- The PP pool is sized at `scheduler.py:1160`, **before**
  `build_phase_flip_tp_stack` (`scheduler.py:1274`) and therefore before
  the draft weights load (`phase_flip_boot.py:564`) and before the draft
  graphs are captured (`phase_flip_boot.py:706`). The binding variable is
  `rest_memory = budget_gb - used_by_me_gb`
  (`model_runner_kv_cache_mixin.py:689`).
- **The PP pool never paid for the draft assets**, so freeing them at
  runtime returns nothing to it — it creates idle card slack only.
- Every grow-path is closed: `post_capture_resize_kv_pool` is clamped to
  never grow (`:2390-2392`) and is disabled here; `runtime_set_backing_tokens`
  is hard-bounded by the VA reservation made at construction
  (`kv_vmm_backing.py:536-540`); `SGLANG_MEASURED_KV_BUDGET` feeds the NEXT
  boot and measures with the draft resident.
- Raising `RANK_MIB` instead runs into the boot guard
  `tp_capacity >= pp_capacity` (`phase_flip_boot.py:636-650`), and the TP
  pool is the one that DOES pay for the draft weights — so the direction
  "free the draft during PP" does not help the constraint that actually
  binds.

**Therefore**: making the spill pay requires reserving PP VA at
`boot_size + spilled_bytes/row_bytes`, calling `runtime_set_backing_tokens`
at the tp→pp seam, propagating capacity the way `vram_dial.py:930-991`
does, re-checking the id-space invariant every flip, and re-broadcasting
the number (the `/get_server_info` copy is a boot snapshot). That is the
real rung 1, and it is substantially more than "apply snapshot_and_free at
the seam". Do not restart from the easy version.

---

## 4a. CORPSE L — purity vs the break-even N (found on the FIRST purity boot)

The first `POLICY=auto` purity boot came up and immediately wedged. Log,
21:39:50Z onward, repeating every 10 s with the server alive and idle:

```
PHASE-POLICY holding in tp: pending prefill 1 tok <= N=7004, running it
in tp (pending prefill 1 tok, running bs 0)
```

A single health-check prompt arrived while the instance sat in TP with
nothing decoding. The policy declined to leave TP because 1 token is far
below the break-even N, and the purity gate refused to prefill it in TP.
Health timed out; the server could not answer a one-token prompt, forever.

**The root is conceptual, not a typo.** N is a BREAK-EVEN: it weighs `n/X`
(prefill in TP) against `C + n/P` (flip first, prefill in PP). That
comparison presupposes the TP option EXISTS. Purity removes it, so "too
small to be worth a flip" silently became "too small to ever run".

**Fix**: `PhasePolicyConfig.prefill_runs_in_tp`, set False by the scheduler
whenever purity forbids TP prefill. The TP→PP threshold then collapses to
0 — ANY pending prefill moves the instance to PP, because PP is the only
place prefill can happen. The decode floor still applies when decode work
exists, so a tiny prefill cannot pre-empt an active decode window.

**Generalisable lesson, and the reason this cost a boot**: purity is not a
local scheduler gate, it INVALIDATES A PREMISE that other components'
arithmetic was built on. Anything deriving a threshold from "what it would
cost to do X in the other layout" is suspect under purity. Grep for other
consumers of `flip_tokens` before adding the next feature.

## 4b. DEFECT M — FATAL, it wedged the instance. Fix this FIRST.

**Status upgraded from "cosmetic log oddity" to "the thing that killed the
green run", 21:47Z. Read this section before anything else.**

Sequence, all in one boot:

1. 21:46:28Z the policy armed a flip on a garbage input:
   `arming pp_to_tp: prefill down to 0 tok (<= N=7004), 10485760 req
   decoding`. 10485760 = 10 x 2^20. True state was IDLE (prefill 0, and
   the sane samples around it read `running bs 0/2/3`).
2. The cutover then handed that same bogus resident set to the draft
   bootstrap. PP1 main thread, py-spy `active+gil` (spinning, NOT
   blocked):

```
committed_slots            (phase_flip_draft_bootstrap.py:190)
arm_draft_bootstrap        (phase_flip_draft_bootstrap.py:411)
arm_draft_bootstrap_all_reachable (phase_flip_draft_bootstrap.py:609)
_cutover                   (phase_flip_runtime.py:1074)
_execute                   (phase_flip_runtime.py:2921)
on_round                   (phase_flip_runtime.py:1977)
```

   `committed_slots` builds one slot tensor PER REQUEST
   (`out.append(req_to_token.new_empty(...))`). Against a claimed 10.5 M
   requests that is millions of allocations: the rank never returns.
3. Rank 0's TCPStore died; ranks 1/2 spun on `sendBytes ... Broken pipe`
   from 21:47:56Z onward. Requests stopped completing at 21:47
   (`ok=3` frozen, `err` climbing to 47), health went to 000.

**So the same wrong number does two things**: it makes the policy arm a
flip that should never have been armed, and it feeds the cutover a
resident set that cannot be iterated. The policy gate is the visible half;
the draft bootstrap is the lethal half.

**The purity build is therefore NOT green.** Purity itself held perfectly
for the 90 s it ran (zero decode in PP, zero prefill in TP, 100 % decode
graphs, accept 2.96) — it was defect M that took the instance down, not
the purity rule. But the two interact: purity leaves requests
resident-but-not-decoding across whole PP windows, which is a state the
resident harvest sees far more often than it used to. That is the most
likely reason this surfaced now and not in the pre-purity 60-min soak.

**Where to look**: `Scheduler.maybe_arm_phase_policy` (~`scheduler.py:7086`)
sums `len(getattr(b, "reqs", []) or [])` over
`harvest_resident_batches` (`phase_flip_resident_carry.py:141`), which
draws from the PP slot array `running_mbs` and from `running_batch`. It
was PP0 in the PP phase, so the PP slot side is the first suspect: a slot
whose `reqs` is not a request list but something tensor-shaped would give
`len()` a dimension. 2^20 smells like a buffer dimension and the x10 like
the 10 GDN resident state slots.

**Do NOT fix by clamping.** A clamp turns a wrong input into a plausible
one and hides the leaking handle — and it would NOT have saved this boot,
because the lethal consumer is `committed_slots`, not the policy. Add a
loud assertion where the resident set is built (a batch may not exceed
`max_running_requests`) so the offending object's type and attribute are
named, then fix the source. `arm_draft_bootstrap` should additionally
refuse an implausible resident set rather than iterate it.

Evidence: `/spinning/evidence-631/serving_defectM_wedge.log` (contains the
full py-spy dumps for all three ranks).

## 4c. Earlier notes on defect M (superseded by 4b)

Live, purity build, 21:46:28Z, the very first `pp_to_tp` arming:

```
PHASE-POLICY arming pp_to_tp: prefill down to 0 tok (<= N=7004),
10485760 req decoding
```

10485760 = 10 x 2^20. Every other sample in the same run is sane
(`running bs 0/2/3`, `1 req decoding`, `2 req decoding`); this is 1
occurrence so far. Not a display bug — the same value is the `running_bs`
the decision was MADE on.

**Why it matters more than a cosmetic log line**: `running_bs > 0` is the
gate on BOTH new fairness rules (the PP window and the TP decode floor)
and on the PP->TP load rule. A spurious large value fabricates decode work
that does not exist, which:
- fires `pp_to_tp` while the instance is actually IDLE (exactly what
  happened here: prefill 0, so the correct verdict was the idle rule and
  the PP resting layout, not a flip),
- would make the PP window expire against phantom waiters,
- would engage the decode floor and hold TP with nothing decoding.

**Where to look** (not yet investigated):
`Scheduler.maybe_arm_phase_policy` computes
`running_bs = sum(len(getattr(b, "reqs", []) or []) for b in
harvest_resident_batches(self))` (`scheduler.py` ~:7086).
`harvest_resident_batches` (`phase_flip_resident_carry.py:141`) draws from
the PP slot array `running_mbs` and from `running_batch`. The occurrence
was in the PP phase on PP0, so the PP slot side is the first suspect: a
slot whose `reqs` attribute is not a request list but something tensor-
shaped would give `len()` a dimension instead of a count. 2^20 smells like
a buffer dimension, and the `x10` like the 10 GDN resident state slots.

**Do NOT "fix" it by clamping.** The value must be explained first: a clamp
would convert a wrong input into a plausible one and hide whichever handle
is leaking into the resident harvest. Add a cheap assertion at the sum
site (running_bs may not exceed `max_running_requests`) and let it raise
with the offending batch's type and attribute — the same
loud-over-plausible discipline the merge_batch guard uses.

## 4d. DEFECT N — flip-time cuMemCreate OOM at ctx 262144 (distinct from M)

Do not conflate this with defect M. Different build, different signature.

Booted `--phase-flip-purity off` at 21:51:41Z on the boot script's
DEFAULTS (non-yarn model, ctx 262144, pool 263768). Healthy 21:54:06Z,
dead 21:55:19Z on a real out-of-memory during the pp_to_tp cutover:

```
_swap -> dst.restore_backing() (phase_flip_runtime.py:1462)
  full_kv_pool.restore_backing() (memory_pool.py:2477)
    _post_capture_owner.finalize -> _back_spans -> commit_range
RuntimeError: cuMemCreate failed: <CUresult.CUDA_ERROR_OUT_OF_MEMORY: 2>

PP1: cuMemCreate: 163577856 bytes refused by the driver; releasing
torch's cached blocks and retrying once. torch reserved 21.68 GiB /
allocated 20.87 GiB
```

The retry-after-empty_cache path fired and still failed. So the TP KV
backing could not be re-committed at the seam: 156 MiB refused with torch
holding ~21 GiB reserved. This is HANDOFF_657 §5c's law biting — sizing
-time headroom is not runtime slack — now at the FLIP rather than at boot.

**MY ERROR, recorded because it cost a boot**: I called this configuration
"the build that survived load". It was not. The config that ran clean
earlier today is `MODEL=...-INT8-W8A8-yarn1.5` at `CTX=393216` (pool
277468); I booted the script's defaults instead and changed both the model
and the context without noticing. **The boot script's defaults are NOT the
proven config.** Always pass MODEL and CTX explicitly, or read them off
the last known-good process with `ps`.

Open question for the next shift, and it matters for the ship config:
whether the OOM is caused by ctx 262144 + pool 263768 specifically, or by
the new fairness windows flipping MORE often than before and so hitting
the seam more often. The second reading is testable cheaply: boot the
proven yarn1.5/393216 config with the windows ON and see whether the seam
OOM reappears. Evidence:
`/spinning/evidence-631/serving_fallback_flipOOM.log`.

## 4e. THE FAIRNESS WINDOWS ARE A REGRESSION — three boots, three deaths

**Read this before re-enabling anything from this shift.**

Three consecutive boots, all carrying my new `pp_window_s=15` /
`tp_decode_floor_s=10`, all dead within minutes:

| boot | config | died | signature |
|---|---|---|---|
| 21:43Z | purity=strict, ctx 262144 | 21:47Z | defect M: spin in `committed_slots` |
| 21:51Z | purity=off, ctx 262144 | 21:55Z | defect N: `cuMemCreate` OOM at the seam |
| 21:56Z | purity=off, **proven** yarn1.5 / ctx 393216 | 22:03Z | `torch.OutOfMemoryError`, 128 MiB on a 3080 with 106 MiB free, **after 12 flips** |

The third one is decisive. It ran the exact configuration that survived
40+ minutes with 0 exceptions earlier today. The ONLY difference was the
windows. It died anyway, with 12 flips in ~4 minutes.

**Mechanism**: the windows raise the FLIP RATE by design (a 15 s PP window
plus a 10 s TP floor is a ~25 s cycle, where the pre-shift build flipped
only when the load rules fired). Every cutover re-commits the KV backing
(`restore_backing` -> `_back_spans` -> `commit_range`) and that seam has a
memory PEAK. HANDOFF_657 §5c already measured the 3080s sitting only
~530-610 MiB above the corridor floor at runtime. More flips per minute =
more visits to that peak, and the rig has no headroom for it. Two of the
three deaths are literally allocation failures at the seam.

**So the honest verdict on this shift's policy change**: it correctly
fixes the STARVATION (defect K) in the decision logic and is pinned by
tests — and it is NOT SHIPPABLE on this rig at these values, because the
flip rate it produces exceeds what the seam can afford. Fixing starvation
by flipping more is the wrong axis while a single cutover costs 1.0-1.7 s
per rank AND spikes memory.

**What the next shift should do with it**, in order:
1. Serving is restored with the windows DISABLED
   (`PHASE_POLICY_PP_WINDOW_S=0`, `PHASE_POLICY_TP_DECODE_FLOOR_S=0`),
   which is behaviourally the pre-shift build. The defaults in
   `phase_policy.py` are still 15/10 — **change the DEFAULTS to 0 or fix
   the seam before anyone boots without those env vars.** That is the
   single most dangerous loose end I am leaving.
2. Make the seam cheap or memory-flat before raising the flip rate again.
   The seam's peak is the real constraint, not the policy.
3. Only then re-tune the windows, with much larger values (minutes, not
   seconds) as the starting point, and measure the corridor minimum
   IN THE PP PHASE specifically.

Evidence: `serving_defectM_wedge.log`, `serving_fallback_flipOOM.log`,
`serving_windows_flipOOM.log`, all under `/spinning/evidence-631/`.

## 5. Exact next steps

0. **§4e FIRST**: the fairness-window defaults (15/10) are still live in
   `phase_policy.py` and they killed three boots. Change them to 0 or fix
   the cutover's memory peak. THEN defect M (§4b), which gates them.
1. **The green-criterion run.** Serving must be up on `2ce40f86a2` with
   `POLICY=auto` (the boot script defaults to `manual` — the first boot
   this shift was discarded for exactly that; always pass `POLICY=auto`).
   Drive it with REAL mixed Qwen-agent traffic through the router, NOT the
   prefill-saturated synthetic (12k tokens every 4 s pins pending above N
   by construction and cannot adjudicate fairness). ≥60 min unmanned, then:
   `bash scripts/phase_evidence_extract.sh` must EXIT 0, and the corridor
   minimum must hold 1024 MiB on every card.
2. **Graph A/Bs** (untouched this shift): NEXTN draft graphs on/off,
   DFLASH x graphs, prefill-graphs-in-PP. Record the numbers even when null.
3. **Spill ladder**: only via §4's real path, or not at all.
4. `PROD_BRINGUP_BENCH.md` §6g still carries the falsified #652
   attribution (HANDOFF_657 §4) — correct it.
5. **PP STAGE IMBALANCE / 5090 UNDER-DRAWN** (user observation, 2026-08-09,
   after the green run — it COMPOSES with it, the green run's PP windows
   are the measurement vehicle). During PP prefill the 5090 draws only
   ~250 W of its 400 W cap. Ranked suspects:
   - (a) stage imbalance: `pp_layer_ratio [32,16,16]` /
     `pp_stage_ratio [2,1,1]` gives the 5090 2x the layers, but it is
     ~2.5-3x faster per layer than a 3080, so stage 0 finishes early and
     idles on pipeline bubbles;
   - (b) `chunked_prefill_size 2048` makes small microbatches, deepening
     the bubble share (#617 is the standing lead on flexible chunks);
   - (c) `pp_async_batch_depth 0` — no microbatch overlap depth.

   Deliverable in order: FIRST the per-stage compute-vs-wait split for the
   PP prefill window (the per-rank CollectiveClock machinery already
   exists — do not build a new instrument). Only if stage 0 is
   wait-dominated, then ONE A/B each for a heavier 5090 layer share
   (36/14/14 or 38/13/13) and a larger prefill chunk. Measure prefill
   tok/s AND per-card power, same-boot A/A noise floor first (this tree
   has a recorded cold-boot decay of 4106 -> 3809 tok/s across six reps of
   ONE build, which reads as an 8 % regression that does not exist). Fold
   the winner into the ship config; record the numbers either way.

## 6. Rules that bit this shift

- The boot script's `POLICY` defaults to `manual`. A boot that looks
  perfect and never flips is this default, not a bug.
- `cd /spinning && nohup bash scripts/...` fails: the script lives in the
  worktree. Use absolute paths in backgrounded commands.
- Three pre-existing policy pins collided with the decode floor because
  they drove a single decision on a freshly-entered phase. The helper now
  takes `aged_s`; a test that means "the server has been in this phase a
  while" must say so rather than inherit a clock.
- `phase_since` is a `None` sentinel, not `0.0`: any caller reaching
  `decide` without `observe_idle` would otherwise read a window that
  expired at t=0 and flip immediately.
