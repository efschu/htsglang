# #656 HANDOFF v20 — successor 17

Written 2026-08-09, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this BEFORE HANDOFF_659, not after: 659's headline number is
withdrawn here, and every plan built on it changes.

---

## 1. THE CORRECTION: the "seam peak" is a PHASE HOLD, not a cutover cost

HANDOFF_659 §1 states that one `pp_to_tp` cutover costs 1.4–3.0 GiB per
card, and §1a concludes that this scales superlinearly with the pool and
that **full-KV therefore requires a zero-allocation seam**. Both the
attribution and the conclusion are wrong. The raw measurements are fine;
what they measure is not what the labels say.

### What the original instrument actually captured

It sampled NVML across a *driven* flip and reported
`median(pre-flip) - min(all)` as the cutover's cost. On this rig
`POLICY=auto` returns an **idle** instance to PP about 1.5 s after a
driven `pp_to_tp` — so the window straddled a there-and-back **pair**,
and the "trough between two baselines" is the **TP-phase plateau between
the two flips**. 659's own holder note records the pair without drawing
the consequence: *"both sampling windows straddle the same pair, epoch 3
pp_to_tp + epoch 4 tp_to_pp"*.

I reproduced it exactly. One driven `pp_to_tp` on the idle instance:

    t=0.1s  5705 6248 4507   (PP)
    t=0.9s  2745 4872 2699   <- read as "the seam peak"
    t=1.7s  2745 4872 2699
    t=2.5s  5705 6248 4507   <- read as "recovery"

and the log shows `PHASE-FLIP DONE tp_to_pp (epoch 4)` at 22:44:30Z, 1.5 s
after the epoch-3 flip I asked for. The "recovery" **is the second
cutover**.

### The falsifier: hold the layout under work, longer than a cutover

Same sampler, but a 1200-token generation keeps the instance in TP for the
whole decode (request wall time 7.93 s):

    t=0.1s  5705 6248 4507   PP, before
    t=1.1s  2743 4850 2695   TP  <-+
    t=3.1s  2739 4848 2693   TP    |  FLAT ~7 s, no cutover in the window
    t=8.1s  2739 4848 2693   TP  <-+
    t=9.1s  5699 6222 4501   back in PP, request finished
    t=39s   5679 6222 4481   still PP, stable

A cutover takes 0.96–1.71 s (`PHASE-FLIP DONE ... in 961.8 / 1218.8 /
1711.9 ms`). A level that holds flat for 7 s and ends when the **request**
ends is a steady state.

**Corrected statement:** the TP layout keeps **~2960 / 1400 / 1810 MiB**
more VRAM resident than the PP layout, for as long as it is the active
layout. There is no multi-GiB cutover transient.

### The scaling result dissolves for the same reason

659 §1a's point 2 (pool 126000, "peak" 368/0/48) was produced by
substituting `--max-total-tokens 500000 -> 126000`. That flag **caps the
TP pool**. It did not shrink a transient superlinearly — it removed the
TP/PP span difference, so the two plateaus became the same height and the
apparent peak vanished. Nothing here is superlinear.

### Consequently WITHDRAWN

* "full-KV >600k and auto-flip are in direct tension, and that tension is
  now numeric" — does not follow.
* "full-KV cannot be reached by sizing around the seam peak; it needs
  pre-reserved zero-allocation staging" — does not follow.
* The pool-ceiling formula `card_total − 1024 − seam_peak`. The seam term
  is not a transient to budget for; the phase hold is a steady state that
  ordinary sizing already accounts for.

### The lesson, since this is the third number the chain inherited wrongly

**A memory reading taken across a flip PAIR cannot distinguish a transient
from a phase hold.** The separator costs one request: hold the target
layout under real work for much longer than one cutover and see whether
the level tracks the WORK or the CUTOVER. `scripts/phase_plateau_measure.sh`
does exactly this and should replace the seam-peak sampler for any future
question of this shape.

## 2. What actually binds capacity

From the live boot (pool 253528, `--max-total-tokens 500000`):

    PP pool  (= the id space = serving capacity)   253528 tokens
    TP pool  (sizes itself to its OWN budget)      450290 tokens
    --max-total-tokens cap                         500000  -> NOT BINDING

Per bench §6e the id space is the **PP** pool, so the TP pool's ~196762
token surplus is **unaddressable** — no request can ever occupy it. That
is precisely the hoarding §6f identified and capped with
`--max-total-tokens`; at this boot the cap was set *above* the TP pool's
own sizing, so it does nothing and the hoard is back.

So the lever for full-KV is the one §6f already used, and it needs **no
new machinery**: cap the TP pool just above the id space (honouring §6e's
invariant `TP capacity >= PP capacity`, whose violation is a device-side
assert, refused at boot since `c8be5d4d50`), then spend the released
headroom on the PP pool, which is the number that actually sets serving
capacity.

**Do not quote the TP number as the pool.** Bench has a section named
"Read this before quoting a pool number" because three shifts already did.
The ~669k "plain-TP reference" in the task brief is a TP-side figure.

## 3. Shipped this shift

Both are default-OFF or inert. **Neither is a capacity fix.**

* **`phase_flip_seam_census.py`** — per-STAGE memory attribution across one
  cutover (`entry / plan / kv_pack / kv_exchange / kv_local_read /
  backing_release / backing_restore / kv_write / gdn_state /
  weights_refill / cutover / done`), one log line per flip per rank,
  reporting per-stage steps and torch's slack alongside driver-free. Costs
  a few `mem_get_info` calls, so it stays on: an instrument that is off
  during the incident is not an instrument.
  *Bug the pins caught:* the guard was written at the module-level `mark()`
  while `begin()` calls `SeamCensus.mark` directly, so the entry probe was
  unprotected. A guard at one of two call sites is a guard that does not
  run on the other — same shape as defect M's "guard where it allocates".
* **Zero-allocation seam**, `SGLANG_FLIP_SEAM_CHUNK_MIB` (default 0 = off).
  Released KV handles are `cuMemUnmap`ped but **parked, not released**, and
  the destination's commit re-maps them, so after one round trip the
  cutover performs zero driver allocations. This removes the `cuMemCreate`
  OOM that can strike **inside the no-return region** (measured
  2026-08-09 12:47:45, rank 1) — a **crash** fix.
  *The coupling that makes it work:* a CUDA physical handle is fixed-size
  and cannot be split or merged, so a parked handle only serves a request
  of the same size. With one monolithic handle per buffer the PP and TP
  spans differ and nothing is reusable — retention would park memory and
  still allocate. A commit chunk makes every handle one granule. Hence ONE
  knob, and retention without a chunk is refused with a named line.

Tests: `bash scripts/run_631_flip_family.sh` → **641 passed** (622
baseline + 11 census pins + 8 retention pins), CPU-only, `PYTHONPATH` on
the worktree.

## 4. Exonerated by evidence — do not re-investigate

* **The carried KV payload is not a memory problem.** The runtime's own
  per-flip stats report **0.47–0.72 MiB** sent per rank. Four orders of
  magnitude below the phase hold. This is what ruled out payload staging
  before any instrumentation was written, and it is printed on every flip.
* **`arena_refill` is not it**: one contiguous `copy_` from a pinned host
  image into a boot-allocated arena, no device staging; its only transient
  is the post-copy checksum, bounded at ≤128 MiB.

## 5. MY ERRORS, ranked — the chain works because these are written down

1. **I killed serving with a `pgrep -f` self-match.** `pgrep -f
   "sglang.launch_server.*--port 30030"` matched **my own shell's command
   line**; the loop TERM'd my own process group and the tool returned exit
   144. This is the *exact* trap named in the message of commit
   `8ce1f44778`, which I had read minutes earlier. Use a
   non-self-matching pattern (`sglang[.]launch_server`) and always skip
   `$$`. The stray TERM is why serving needed rebooting at all.
2. **I burned context grepping the serving log without truncating.** The
   collective-census lines are kilobyte-scale and repeat every 2–3 s; one
   ungated `grep` returned dozens of near-identical multi-KB lines. Always
   `| cut -c1-200`.
3. **I initially reasoned toward the wrong mechanism** (payload staging,
   then the backing swap) from code reading, and only got the right answer
   by measuring. The first falsification I ran — checking that free memory
   *recovers* — appeared to CONFIRM 659's transient reading, because a
   there-and-back pair recovers too. The reading that settled it was
   holding the layout under load. A falsifier that a wrong hypothesis also
   passes is not a falsifier.

## 6. State at handover

* HEAD `32507a4751`, pushed to fork. Suite 641 green.
* Serving was rebooted by me with `MAX_TOTAL_TOKENS=260000` (the TP cap, as
  the single changed variable) to test §2's prediction. **Check
  `/spinning/serving-30030.boot.log` and the holder for the outcome — if
  this handoff has no result table in §7, the boot's result was not yet
  in when it was written.**
* Boot recipe: `/tmp/s17_boot.sh` replays the proven yarn1.5 / CTX=393216
  config from the live `/proc` environ. The boot script **defaults to CTX
  262144**; successor 15 misbooted on exactly that.
* Purity is still OFF and the fairness windows are still 0. Nothing about
  the acceptance criteria has been proven this shift.

## 7. Next steps, in order

1. Confirm the TP cap released the plateau: `bash
   scripts/phase_plateau_measure.sh capped` and compare the two plateaus
   against the uncapped numbers in §1. Expect the TP plateau to rise to
   near the PP plateau.
2. Spend the released headroom on the **PP** pool (`RANK_MIB`, and
   `--gdn-resident-state-slots 10` which bench documents as
   253528 → 277468), keeping the cap just above the new id space. Watch
   `_profile_available_bytes` — bench §6f's "honest ceiling" is the known
   binder here, not the corridor.
3. Only then re-enable fairness windows + `--phase-flip-purity strict`
   (the user's hard default, must be ON in the ship config) and prove on
   ≥60 min with minutes-scale settle windows.
4. Graph A/Bs per spec item 8 (NEXTN draft graphs OFF unless measured
   positive; DFLASH × graphs; PP-prefill graphs) and the 5090
   stage-imbalance lead (~250 W of 400 W during PP3 prefill).
5. Final ≥60-min green run with real agent traffic through router 30099;
   write "GREEN-RUN STAGE: operator may re-arm qwen traffic agents" into
   `/spinning/gpu-arb/holder` when that stage is reached.
