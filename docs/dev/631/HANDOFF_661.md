# #656 HANDOFF v21 — successor 18

Written 2026-08-09/10, tree `/spinning/wt-631-routea`, branch
`feat/route-a-631`.

Read this before HANDOFF_660. 660's headline capacity number is retired
here, and a crash that had been killing this instance for several shifts is
identified and fixed.

---

## 1. MY ERRORS, ranked — read these before my results

**0. I told the operator to arm real agent traffic BEFORE the config had
survived anything longer than 15 minutes.** The program says to write the
`GREEN-RUN STAGE` re-arm line into `/spinning/gpu-arb/holder` when the green
run is armed, and I did — at 23:42:10Z. The green run died **26 seconds
later**, at 23:42:45Z. Had the operator been quick, live agent traffic would
have been pointed at an instance that was already dead. The line is a
*promise to an outside party*, and I issued it on the strength of a 15-minute
rung. **Arm the load first, watch it survive past the failure mode you know
about, and only then write the line.** The ordering costs one poll and it is
the difference between a signal and a false signal.

**1. I stated the cause of that crash before I had checked it.** My first
written reaction was that I had contaminated my own measurement by running
the test suite on the same cards — "self-inflicted", stated as fact. I then
checked, and `scripts/run_631_flip_family.sh:52` pins
`CUDA_VISIBLE_DEVICES=99`: the suite never touches a GPU and was innocent.
The real cause was a scheduler defect that had nothing to do with me. The
check cost one grep; the claim would have sent a successor hunting a
non-existent contamination and, worse, would have left the actual defect
in place wearing my confession as cover. **A plausible self-blame is still
an unverified hypothesis, and it is the most dangerous kind, because
nobody argues with it.**

**2. I ended turns to "wait" for background notifications and stalled.**
An ended turn receives nothing; the operator had to re-poke me. Every wait
must be a bounded in-call poll (`for i in $(seq 1 N); do ... sleep 10; done`
inside one Bash call, or a `run_in_background` command I then *read*), never
a turn boundary. This is the documented agent-idle-after-monitor trap and I
walked into it anyway, having read the note.

**3. I let the first corridor measurement run 15 minutes before questioning
whether the load matched the design point.** It did, but I got there by
luck of the default rather than by checking `max_running_requests` first.
The instrument now encodes it (`route_a_631_bs4_rung.sh`).

---

## 2. THE CRASH, found and fixed — this is the shift's main result

`scheduler.py`, `get_next_batch_to_run`. Commit `1ba907f1b5`.

### What it looked like

All three ranks, 23:42:45Z:

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
    GPU 0 has a total capacity of 19.58 GiB of which 138.38 MiB is free.
      File ".../sampling/sampling_batch_info.py", line 429, in merge_batch
        setattr(self, item, torch.cat([self_val, other_val]))

The three seconds before it contain the diagnosis, in the scheduler's own
words — the flip policy declining to run because *"the resident set is
corrupted (PHASE-FLIP-CARRY running_mbs[0] claims 8388608 resident …)"*,
with the claimed count walking **8388608 → 16777216 → 33554432**. Powers of
two. Something was doubling every round.

### Why

`get_next_batch_to_run` merges an extend `last_batch` into `running_batch`.
The branch immediately above assigns `running_batch = last_batch` when
`running_batch` is empty, which **aliases the two names onto one object**.
Under PP the scheduler then stores that object into *both*
`running_mbs[mb_id]` and — via `mbs[mb_id]` —`last_mbs[mb_id]`
(`scheduler_pp_mixin.py:198` and `:250`). The next visit to that slot
rebinds both names to the same batch and calls `X.merge_batch(X)`, which
`torch.cat`s every per-request field with itself. The batch doubles per
visit; ~23 visits exhaust a card.

**Strict phase purity is what made it reachable.** The merge branch only
runs for an *extend* `last_batch`, and purity confines all prefill to the
PP layout, so the aliased slot is revisited while still carrying an extend
batch. This is why the defect surfaced now and not in the permissive builds.

### The fix, and why skipping is correct rather than a mitigation

An identity guard. When the two names are one object its requests are
**already resident** in `running_batch`, so the merge could only
double-count them — skipping is the semantically right answer, not a
papering-over. The condition is **logged** rather than silent, and that
distinction is the lesson: the flip-policy guard *already detected* this
exact state and correctly refused to arm a flip, but **a detector that only
declines to act cannot stop a doubling**, and the instance died anyway with
its own diagnosis printed in the log. Detection is not containment.

### Evidence

Same boot config, before and after:

| | before | after |
|---|---|---|
| survival | dead 26 s in | 65-min window, 0 deaths |
| requests | — | 0 errors |
| corridor | collapsed to 42 MiB | 1421 / 1396 / 1453 MiB, 0 breaches |

The guard fires **21042 times in 3 minutes, always `bs=1`** — the condition
is pervasive, not a rare race. Suite: **643 passed**.

---

## 3. CAPACITY: 540000 is retired, 460000 is the ship number

`route_a_631_bs4_rung.sh` judges a rung at the **design point** — the 4
concurrent requests `--max-running-requests 4` actually admits — with a
100 ms corridor time series. The previous ladder was climbed on
single-stream readings, and that is the whole difference:

| card | bs=1 (inherited) | bs=4 (design point) | breaches |
|---|---|---|---|
| nvml gpu0 (rank 1, 3080) | 1719 | **355** | 329 |
| nvml gpu1 (rank 0, 5090) | 1686 | **42** | 252 |
| nvml gpu2 (rank 2, 3080) | 1865 | **661** | 465 |

All three cards breach. **A rung is only real when the load that produced
it is the load the server is configured to admit.** Two properties worth
carrying: the allocator does not hand the peak back (the 5090 still read
42 MiB with the load stopped, so a post-run snapshot cannot clear what a
time series condemns), and bs=1 is not a weaker bs=4 but a different
measurement.

**Ship config**, corridor-legal with strict purity and BOTH fairness
windows on:

    MODEL  Qwen3.6-27B-INT8-W8A8-yarn1.5   CTX 393216
    RANK_MIB 31800,17400,17450   MAX_TOTAL_TOKENS 460000
    POLICY=auto  PHASE_FLIP_PURITY=strict
    PHASE_POLICY_PP_WINDOW_S=15  PHASE_POLICY_TP_DECODE_FLOOR_S=10

The corridor edge sits near **480000** (rank 0's slope is 15.0 KiB per
global token and the 5090 has ~390 MiB of margin), so 460000 is "close to
1024 and well filled" per the corridor law, not a timid number.

### The fairness windows were never the defect

HANDOFF_658 §4e condemned them on three boots that died within minutes.
Two of those three deaths were **allocation failures at the seam** on a rig
§657 measured as sitting 530–610 MiB above the floor. The same windows, at
a corridor-legal pool leaving ~1450 MiB, produced 453 flips in 15 minutes
with zero deaths. The windows raise the flip rate; the flip rate was never
affordable because the pool was oversized. **A policy knob was blamed for a
sizing defect** — the third time in this chain that a component measured in
an independently broken configuration was convicted of the breakage.
Restore the configuration before judging the component.

---

## 4. FULL KV (>600k): the target was a misread, and the spill route is
   on the wrong side of the constraint

**Do not implement the spill ladder for capacity.** Two independent
findings, either one sufficient:

1. **669440 was never a serving capacity.** Our own bench calls it "the
   TP-only reference" (`PROD_BRINGUP_BENCH.md:422`) and adds *"treat 669440
   as the ceiling, not more: the model omits activation reserve, graphs,
   NCCL buffers and the continuous corridor minimum"*. It is a modelled
   ceiling for a different topology. HANDOFF_660 §2 flags the same trap.
2. **The spill frees bytes in the wrong phase.** Every asset
   `phase_flip_spill.py` can release — draft weights (~1.9 GiB/rank), draft
   graphs (~0.55 GiB/rank) — is **TP-exclusive**, so it can only be spilled
   while PP is active. The corridor binds in the **TP** phase. The spill
   adds margin to the phase that already has 1.4–3.0 GiB of slack and
   contributes zero to the phase that is short.

Byte model, for the next person who needs to size this: 64 layers,
`full_attention_interval=4` → 16 full-attention layers, 4 KV heads × 256
head_dim at `fp8_e4m3` → **32 KiB per token whole-model**, which under the
TP token split costs **15.0 / 8.5 / 8.5 KiB per GLOBAL token** on ranks
0/1/2. Checked against the measured ladder: 80000 tokens moved the corridor
936/1348/744 MiB where the model predicts 664/1172/664 — right, and
slightly conservative. Going 460000 → 669440 therefore needs **3069 MiB
more on the 5090 alone**, against ~390 MiB of margin.

**Conclusion: >600k with speculation ON, graphs ON and the corridor held is
not reachable on this rig.** That is a hardware statement, not a defeat —
and it should be put to the user as one, since the requirement was written
against a number that turns out to describe a different configuration.

### The one lever that is real, and why it is not a flag flip

`--speculative-draft-placement solo` builds the draft **unsharded on one
rank**; the others get no draft weights, no draft KV pool, no draft CUDA
graphs (`server_args.py:3711`). That frees ~2.5 GiB on two of three ranks
**in the TP phase** — the binding one — and on a no-P2P rig like this the
help text argues it is also a throughput win. Rough sizing: solo on a 3080
moves the ceiling to ~507k, and with the token vector rebalanced toward the
freed ranks the arithmetic reaches the ~669k class.

**It hard-rejects `pp_size > 1`** (`server_args.py:_handle_speculative_draft_placement`,
the DP/PP branch) because divergent scheduler instances would desync the
one-broadcast-per-round contract into a silent NCCL hang. In the flip
topology the draft lives **only in the TP stack**, which is pure TP, so the
refusal is stricter than the hazard — but making it flip-aware is a
distributed change whose failure mode is a silent hang, and it must be
built with the rank-local-before-collective discipline, not by deleting a
guard.

---

## 5. State at handover

* HEAD `1ba907f1b5`, pushed to fork. Suite **643 passed**, exit 0.
* Serving UP on 30030, health 200, pool 460000, purity strict, both
  fairness windows on, POLICY=auto.
* Boot replay: `/tmp/s18_boot.sh` (ship shape; pass `RANK_MIB` and
  `MAX_TOTAL_TOKENS`). It defaults to purity strict and windows 15/10,
  unlike `route_a_631_prod_boot.sh`, whose own defaults are the
  non-proven CTX 262144 config.
* Instruments: `scripts/route_a_631_bs4_rung.sh` (capacity rung at the
  design point), `scripts/green_criterion_631.sh` (the green run).

## 6. Next steps, in order

1. **Finish the green run and judge it.** Evidence must show both layouts
   visited, prefill only in PP, decode only in TP, graphs live, accept-len,
   the pool number, the corridor time-series minimum, and real agent tasks
   completed through router 30099.
2. **Graph A/Bs (spec item 8), all still unmeasured.** NEXTN draft graphs
   (drop if no gain — `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` is the
   lever, and dropping them also returns ~0.55 GiB/rank to the binding TP
   phase, so this is a capacity change as well as a latency one), DFLASH ×
   graphs, and PP-prefill graphs (enable only if measured positive). Plus
   the 5090 stage-imbalance lead: during PP3 prefill it draws ~250 of
   400 W, so A/B a heavier 5090 share in `pp_layer_ratio` and a larger
   prefill chunk, measured as per-stage compute-vs-wait ms/round.
3. **Put the full-KV physics to the user** (§4). The requirement is written
   against a number that describes a different topology.
4. Only then, if the user still wants the last stretch: the flip-aware
   `solo` placement (§4).
