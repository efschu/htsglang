# TICKET #470 — the DSpark solo arm: two boots, one window

Written 2026-08-03 alongside the desk slice of #470. **Everything the desk
slice produced is DESK-WRITTEN AND NEVER EXECUTED** until this ticket's window
runs: no DSpark arm has booted on this rig, on any placement.

Authority for the route: `ANALYSE_463_dspark_formats.md` (§4 placement, §5
ranking, §7 what the code turned into). Rig recipe: `rig-runbook.md` §4.5.4 and
its new "DSpark draft arm (#470) — flags" block.

---

## 0 — Why there are two boots and not one

The solo arm costs VRAM on rank 0 *before* it can earn anything. Rank 0 runs
the 119.4 GiB target under expert offload against a 30 407 MiB budget, so its
budget is filled by a TUNABLE resident expert set, not by a fixed weight load.
Hosting a ~10.5-11 GiB draft means cutting that resident set by roughly 21 %
(≈ 1 040 Q3_K experts, ≈ 25 per layer of rank 0's 119-of-256 shard) — and rank
0 is the clock rank (#439), sitting on a x4 link (`rig-interconnect-p2p`).

**Boot A prices that cut with no draft in the picture at all.** It is the
number that decides whether R1 survives, and it is valid whether or not DSpark
itself works. Boot B then has to beat it.

---

## 1 — Before anything: resolve the cards

Do not assume which NVML index is the 5090; enumeration order shifts between
boots and driver states. Resolve it, then build the flags from the output.

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
```

Take a `/spinning/gpu-arb/` window first. Corridor discipline as standing:
free >= 400 MiB absolute, waste <= 1.5 GiB NET of registered posts; no
measurement on red.

---

## 2 — Boot A: price the residency cut (NO draft)

Target only, exactly the §4.5.4 recipe, with rank 0's resident expert set cut
by ~11 GiB — i.e. rank 0 booted as if the draft were resident, but with the
draft absent. Same boot, A-vs-A floor first, then the cut.

1. Boot the §4.5.4 recipe unchanged. Measure the A-vs-A floor (two identical
   runs, interleaved, fixed clocks) — this is the noise floor for everything
   below, per the benchmark-harness rule.
2. Same boot: reduce rank 0's resident budget by ~11 GiB and measure again.
3. Report **ms/verify round and ms/prefill, per rank**, not tok/s. Rank 0 is
   expected to be the loser; say by how much, and say how much of rank 0's
   round is compute vs wait.

**Deliverable: one number** — the decode cost, in ms/round, of making room for
the head. Everything after this is measured against it.

If the cut cannot be expressed as a budget knob on this build, say so and
price it by the closest available lever rather than skipping the boot: without
Boot A, Boot B's multiplier is unattributable.

---

## 3 — Boot B: the DSpark arm live

Same residency as Boot A step 2, plus the arm from `rig-runbook.md` §4.5.4:

```
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered" \
  --speculative-draft-placement solo \
  --speculative-draft-gpu <NVML index of the 5090> \
  --speculative-moe-runner-backend marlin \
  --speculative-dspark-block-size 5 \
  --speculative-num-draft-tokens 6 \
  --speculative-num-steps 1 --speculative-eagle-topk 1
```

* `SGLANG_DSV4_FP4_DEQUANT` **unset / 0**. At 1 it asserts against a non-auto
  runner backend and inflates the head to 18.6 GiB.
* **Greedy requests only** (`temperature 0`). Solo refuses non-greedy rounds by
  name; that is the v1 limit, not a bug to work around.
* Instrument: `meta_info.spec_accept_length`, `spec_verify_ct`, decode seconds.
  **Not** `spec_ema_accept_len` — it is not the accept length.
* Use **native `/generate`**, not `/v1/chat`.
* Prompts: `/spinning/gpu-battery-results/2026-08-03_447_dspark/prompts.json`.
* Reference band 0.49-0.77 accept (llama.cpp PR #25784, *their* domains, order
  of magnitude only). Below ~0.45 on a comparable mix means the block/Markov
  chaining is wrong, not merely slow.

### 3.1 First-boot checks, in order (cheap before expensive)

1. The log says `Draft-solo placement: rank N HOSTS the unsharded solo draft`
   on the 5090's rank and `... is a draft SHADOW` on the other two.
2. The log says the markov_w2 TP-shard optimization was disabled under solo.
3. `Preparing MXFP4 experts for Marlin backend` appears for the draft's expert
   layers — i.e. `--speculative-moe-runner-backend marlin` actually reached the
   draft build. If it does not, the flag wiring in `draft_worker_common.py` did
   not take and everything after is measuring the wrong kernel.
4. A short greedy generation completes without a hang. A hang here is a missing
   shadow participant in one of the four round collectives (embed all_reduce,
   hidden broadcast, vocab all_gather, round payload) — attach `py-spy` before
   killing anything, and never a broad `pkill`.

### 3.2 Answer inside the same window

`ANALYSE_447` §2.4: **are the CSA/HCA/LID compressor writes idempotent under a
re-run at the same positions?** If they are not, rejected draft tokens corrupt
compressor state and the accept number is meaningless. This is a correctness
question, so it outranks the perf numbers in the same window.

---

## 4 — Boot C (conditional): the EAGLE3 head, same residency

Upstream **sgl-project/sglang #33344** publishes a third-party EAGLE3 draft
head for exactly this checkpoint: `AQ-MedAI/DeepSeek-V4-Flash-0731-eagle3`,
marlin MoE runner, **upstream-reported** accept **2.62-3.20** across six
benchmarks. That is not our measurement and not our prompt mix — but it is an
order of magnitude above the DSpark ladder's 0.49-0.77, and it sits at a
comparable VRAM ask on the same card.

EAGLE was already admitted by the solo refusal, so this arm needs no code from
#470 beyond the marlin flag wiring, which it shares. Run it if Boot A leaves
window time:

```
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path <AQ-MedAI/DeepSeek-V4-Flash-0731-eagle3> \
  --speculative-draft-placement solo \
  --speculative-draft-gpu <NVML index of the 5090> \
  --speculative-moe-runner-backend marlin \
  --speculative-eagle-topk 1
```

Same instrument, same prompts, same residency as Boot A step 2, so the three
boots are directly comparable.

**Decision rule:** Boot A prices the residency cut ONCE and both draft arms pay
it. Whichever of B and C returns more against that same cost wins the slot. If
C dominates B, the fork's next step on this checkpoint is the EAGLE3 head and
DSpark becomes a second-order question — the #470 placement work is not wasted
either way, because the solo mechanism and the marlin wiring are what both arms
ride on.

---

## 5 — GATE

* **If Boot A's residency cut costs more than the best draft arm returns, R1 is
  refuted.** The next step is then **R3** — GPTQ-INT4 requant of the experts
  plus `split` placement, which makes the VRAM gap vanish by construction
  (~3.2 GiB/rank) and gives all three cards a marlin path — **not R2**. R2
  (Q2_K GGUF head) is the same effort class as R3 and strictly less general;
  `ANALYSE_463` §5 says so and nothing in this ticket changes it.
* If Boot A cannot be run at all, do not run Boot B: an unattributed multiplier
  is not a result.

---

## 6 — Rebase gates (upstream, checked 2026-08-03)

* **#33298 (merged upstream)** moves the DSPARK graph-folded sampler into a new
  `dspark_draft_sampler.py` and adds in-graph philox sampling. Our solo path
  deliberately does not fold the sampler, so there is no behavioural conflict,
  but `dspark_draft.py` moves. Re-check `DraftBlockProposer.propose`'s fold
  predicate and the solo embed hoist after that rebase.
* **#33312**: DSpark shared-expert loading is broken on upstream main
  (ServerArgs-burndown fallout). We are not on upstream main, so this is not a
  live exposure — but do not rebase past it without checking that the
  `mtp.*.ffn.shared_experts.*` tensors still land.
* **#33276** (open): adds `stages.*` to the hybrid-DSV4-NVFP4 exclusions so an
  MXFP4-expert DSpark head loads under a NVFP4 target. Not our configuration;
  noted so it is not mistaken for a new format.

---

## 7 — BOOT-PENDING inventory (what this ticket is the only evidence for)

Nothing below has run on a card. All of it is desk-written.

1. The DSpark solo round contract end to end: four collectives per round in the
   host's order, one payload broadcast, shadows committing the host's tokens.
   Hermetically tested with a fake collective (`test_dspark_solo_round.py`,
   including the executed can-fail arm) — **never over real NCCL**.
2. The solo embed hoist (`forward_embed` -> eager `input_embeds`) on the real
   DSV4 DSpark model. Shape-correct on paper; no forward has run.
3. The `_solo_normed_publisher` seam in `_logits_from_x_post_hc`: host and
   shadow entering the same vocab all_gather. No forward has run.
4. `--speculative-moe-runner-backend marlin` actually selecting
   `Mxfp4MarlinMoEMethod` for the draft's routed experts (§3.1 check 3).
5. The non-SM90/SM120 named refusal in `draft_worker_common.py` — it is a
   `torch.cuda` capability read, never exercised on a real sm86 rank.
6. Whether ~11 GiB is the right ask at all. It is arithmetic
   (`ANALYSE_463` §4.4), not a measurement; the "~4 GiB short" figure it
   replaces was itself never persisted anywhere.
