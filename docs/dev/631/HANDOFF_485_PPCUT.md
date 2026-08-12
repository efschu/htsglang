# HANDOFF 485 — the decoupled GDN/attention PP cut

Branch `feat/pp-family-cut-485`, worktree `/spinning/wt-485-ppcut`, base
`03b6fb990d`. Parallel strand next to the #656 flip successor; no GPU window
was available for the whole shift (see §6), so **everything below is
hermetic desk work unless a line says otherwise**.

---

## 1. ERRORS FIRST — what the briefing got wrong, and what I got wrong

### 1a. The briefing's central premise is false: deep-prefill attention is NOT bandwidth-bound

The task was framed as "the deep-prefill attention term is
bandwidth-proportional, so give the 5090 attention layers in proportion to
its memory bandwidth (~9 of 16)". That is wrong, and the arithmetic is not
close.

One full-attention layer, chunk `C` query tokens against KV depth `D`:

```
core FLOPs = 4 * C * D * q_heads * head_dim      (QK^T and A@V)
KV bytes   = D * kv_heads * head_dim * 2 * dtype_bytes
intensity  = 2 * C * q_heads / (kv_heads * dtype_bytes)     <- D CANCELS
```

At the reference geometry (`q=24, kv=4, head_dim=256`, fp8 KV) and `C=2048`
that is **24 576 FLOP/byte**, against measured ridge points of **151**
(5090: 231.97 TFLOPS / 1533.8 GB/s) and **91** (3080). The term is
compute-bound by more than two orders of magnitude, at every depth — the
intensity does not depend on `D` at all, so no amount of context makes it
bandwidth-bound. The crossover is at a chunk of **~13 tokens**, i.e. decode.

Consequence: apportioning attention on the **2.14x** memory-bandwidth spread
instead of the **3.54x** bf16 GEMM spread under-skews the cut. The direction
in the briefing was right; the rate, the magnitude and the stated reason
were all wrong. The solver therefore prices the attention core as a
**roofline**, `max(flops/gemm, bytes/bw)`, and reports which side bound it
(`StageCost.attn_bound_by`) so this cannot be mis-read again. The roofline
is kept rather than simplified to a FLOP count precisely so a decode-shaped
stage flips sides automatically.

### 1b. The briefing's "uniform-layer-count" premise is also false

`derive_pp_layer_split` (`distributed/utils.py:1481`) has been
full-attention-aware since #201 slice 3. It does not cut uniformly. The real
gap is different and narrower: **the ratio is supplied by hand and the
planner never sees it** (`planner/plan.py:73` — `pp_size = 1`, "planner
scope: pure single-node TP"). `--pp-layer-ratio` reaches `get_pp_indices`
through a process env var (`server_args.py:14824` sets
`SGLANG_PP_LAYER_PARTITION`), which is why it bypasses everything.

### 1c. The model is 64 layers, not 48

Briefing said "16 full-attn of 48 total". The checkpoint declares
`num_hidden_layers: 64`, `full_attention_interval: 4` → **16 of 64**. Every
number in the briefing derived from 48 is off.

### 1d. MY OWN ERRORS, both caught by tests

1. I first conflated **KV pool tokens** (memory, `--max-total-tokens`) with
   **request depth** (time). They are different quantities and mixing them
   either prices a 600k arena as one 179k request or times a request as if
   it swept the whole pool. Now separate fields (`kv_pool_tokens` vs
   `kv_depth_tokens`).
2. I asserted the bench log's `T=540000 PP rank0 K = 4.12 GiB` row against
   a `[28,20,16]` split. The test went red: 4.12 GiB is **8** attention
   layers, not 7. The ship config is `[32,16,16]` (attention 8/4/4), stated
   at `PROD_BRINGUP_BENCH.md:1595`. The arena model now reproduces four
   measured rows exactly (K = 4.12 / 2.06 / 3.51 / 2.90 GiB).

### 1e. UNRESOLVED AND HANDED FORWARD: the memory gate is uncalibrated

`RankResources.fixed_overhead_mib` defaults to **0**, which makes the
feasibility verdict a **LOWER BOUND on real occupancy**. It will read
"feasible" where metal is not. I did not invent a constant, because the
residual is not cut-invariant and therefore a single scalar may not even be
the right shape. Measured off the sec-1g at-rest boot (pool 600000, split
28/20/16), `residual = used - model_weights - model_kv`:

| rank | card | used MiB | weights | KV arena | **residual** |
|---|---|---:|---:|---:|---:|
| rank0 | 5090 | 28325 | 9951 | 8203 | **10171** |
| rank1 | 3080 | 18671 | 7108 | 6581 | **4982** |
| rank2 | 3080 | 18331 | 5686 | 5062 | **7582** |

Two identical 3080s differ by 2600 MiB, so the residual is absorbing at
least one cut-dependent term (a co-resident TP weight shard is the leading
suspect — the flip keeps both layouts' weights resident, and the uneven-TP
shares .463/.268/.268 are the right order of magnitude — plus the
asymmetric mamba/GDN state pool, `PROD_BRINGUP_BENCH.md:2105`, 1516/758/758).

**Do not gate a boot on this constraint until it is calibrated.** The
mechanism is tested (`test_fixed_overhead_binds_the_constraint`,
`test_overhead_changes_the_solved_cut`); only the constant is missing.

---

## 2. What is PROVEN — hermetic only, no metal

68 tests, all green, all CPU-only, all runnable in 0.4 s:

```
cd /spinning/wt-485-ppcut && PYTHONPATH=/spinning/wt-485-ppcut/python \
  /spinning/htsglang-gpu/.venv/bin/python \
  test/registered/unit/planner/test_pp_family_cut_485.py        # 42
cd /spinning/wt-485-ppcut && PYTHONPATH=/spinning/wt-485-ppcut/python \
  /spinning/htsglang-gpu/.venv/bin/python \
  test/registered/unit/server_args/test_pp_stage_ratio.py       # 26 (16 pre-existing)
```

`PYTHONPATH` is mandatory — without it the tests silently run against
`/spinning/htsglang-gpu/python` and prove nothing about this branch.

**Can-fail proof.** Passing tests I have not seen fail are not evidence, so
each load-bearing claim was sabotaged and confirmed red:

| sabotage | tests that went red |
|---|---|
| force the attention core to "bandwidth" | `test_prefill_attention_is_compute_bound` |
| replace the solver with the uniform split | 3 of 4 solver tests, incl. brute-force equality |
| make `derive_pp_layer_split` ignore `attn_scores` | 2 of 3 decoupling tests |
| disable the memory gate | both refusal tests |

### 2a. The 4-layer quantization is not a hardware limit

`PROD_BRINGUP_BENCH.md:2430-2448` records "a stage boundary can only fall on
a multiple of 4" and `:2499-2504` concludes "the surplus is UNREACHABLE with
the layer knob… there is no setting that levels this rig from the PP side."

Both are **false**, and the first is already false in the shipped code:
`derive_pp_layer_split([15,10,7], …)` returns `[32,18,14]`, a boundary at
50. The 4-quantization is an artifact of deriving BOTH targets from ONE
score vector:

```
target_full   = round(n_full   * f)
target_layers = round(n_layers * f)   # = round(P * target_full) for period P
```

so the layer target lands at the bottom of the snap window whenever `f` sits
near a multiple of `1/n_full`. Give the attention target its own vector and
the coupling disappears. On this checkpoint **16 distinct layer splits hold
the attention split at `[7,5,4]`** (`[28,20,16]` … `[31,17,16]`) and 16 more
hold it at `[8,4,4]` (`[32,16,16]` … `[35,16,13]`). Moving three linear
layers off the binding card costs **zero** KV bytes.

### 2b. The arena model reproduces metal

`T x 32 KiB x max(PP layer share, TP token share)` — the relation the bench
log validates to the byte — is implemented and pinned against four measured
rows in `test_kv_arena_matches_the_validated_formula`.

---

## 3. What awaits metal

The solver's recommendation and the predicted deltas below are **model
output, not measurements.** Nothing here has run on a card.

Predicted at depth 179000, pool 460000 (the regime where the term binds):

| arm | layers | attention | stage ms | makespan | vs control |
|---|---|---|---|---:|---:|
| **A** control (ship) | 32,16,16 | 8,4,4 | 521 / 922 / 922 | 922 ms | — |
| **B** decoupled, KV-neutral | 35,15,14 | 8,4,4 | 542 / 898 / 874 | 898 ms | **+2.6 %** |
| **C** planner-optimal | 42,11,11 | 10,3,3 | 665 / 668 / 667 | 668 ms | **+27.6 %** |
| **D** falsifier, anti-proportional | 16,24,24 | 4,6,6 | — | 1383 ms | **−50 %** |

Read these honestly:

* **B is the pure decoupling arm** — it changes only linear-layer placement,
  KV is bit-for-bit identical to A. Its +2.6 % is **inside the rig's own
  3.0–3.5 % A-vs-A prefill noise floor**, so B is very likely NOT
  independently measurable. Its value is memory relief at zero KV cost, not
  speed; measure it on the corridor, not on the clock.
* **C is the arm worth the window.** +27.6 % is far outside the floor. It is
  also the risky one: it moves the attention split to 10/3/3, i.e. 62.5 % of
  the KV arena onto the 5090, and §1e means its feasibility is **unverified**.
  Boot it expecting a corridor problem and watch the 5090.
* **D must come out slower.** If it does not, the objective is measuring
  nothing and everything above is void.

### 3a. Exact boot commands

Substitute the running ship boot's own flags; only the two ratio flags change.
Resolve card indices via NVML at run time (rank0 = 5090; nvidia-smi index 0
is a 3080 — do not assume they agree).

```
# arm A -- control, the current ship cut
--pp-size 3 --pp-stage-ratio 15,9,8

# arm B -- decoupled, attention held at the ship split, 3 linear layers moved
--pp-size 3 --pp-stage-ratio 35,15,14 --pp-attn-stage-ratio 8,4,4

# arm C -- planner-optimal
--pp-size 3 --pp-stage-ratio 42,11,11 --pp-attn-stage-ratio 10,3,3

# arm D -- falsifier, must be slower than A
--pp-size 3 --pp-stage-ratio 16,24,24 --pp-attn-stage-ratio 4,6,6
```

Each was verified through the real handler to produce exactly the intended
`SGLANG_PP_LAYER_PARTITION`; read `pp_layer_ratio=[...], pp_stage_ratio=[...]`
out of the boot log to confirm, rather than inferring the split from an
arena size.

### 3b. Acceptance

Same-boot-floor 3-arm (4 with D) A/B at deep prefill, ms/round per rank split
COMPUTE vs WAIT via CollectiveClock, runs ≥ 10 s, warmup discarded,
A-vs-A noise floor established first. Accept C if it wins at depth without
regressing shallow prefill beyond the floor **and** the corridor holds
(seam census, NVML free column, 100 ms sampling, minimum under load — not a
boot snapshot). B is accepted on the corridor axis alone.

---

## 4. What changed in the code

| file | change |
|---|---|
| `python/sglang/srt/planner/pp_cut.py` | **new**, stdlib-only. Roofline stage cost, exact two-pass DP over contiguous cuts (minimize makespan, then maximize the tightest headroom), loud per-rank refusals, and `validate_pp_cut` for priced override checking. |
| `python/sglang/srt/distributed/utils.py` | `derive_pp_layer_split` gains optional `attn_scores`. Absent ⇒ byte-identical to before (pinned by three legacy rows). |
| `python/sglang/srt/server_args.py` | new `--pp-attn-stage-ratio`; refuses without `--pp-stage-ratio`, refuses on a non-hybrid, logs what the coupled derivation *would* have produced. Plus the family-census gate on the explicit `--pp-layer-ratio` path (§4c). |
| `test/registered/unit/planner/test_pp_family_cut_485.py` | **new**, 42 tests. |
| `test/registered/unit/server_args/test_pp_stage_ratio.py` | +10 tests (5 for the new flag, 5 for the gate). |

`ruff` clean on every file I touched; `server_args.py` has the same 356
pre-existing findings as the base commit, i.e. I added none. `codespell`
clean.

### 4a. Deliberate deviation from the briefing

The briefing asked for "an explicit per-stage layer list (which layers, not
just counts)". I kept the cut **contiguous** and did not build
non-contiguous ownership. Reasons, in order:

1. It is unnecessary. A boundary may sit anywhere inside a full-attention
   period, so a contiguous cut already spans the useful decoupled space (16
   splits per attention split, §2a).
2. It is expensive and dangerous. Non-contiguous ownership requires
   replacing the `layer_id - self.start_layer` offset arithmetic across
   `mem_cache/memory_pool.py` (dozens of sites), the `PPMissingLayer`
   padding in `utils/common.py:make_layers`, and the explicit contiguity
   assertions in `managers/phase_flip_runtime.py:870-874` and
   `managers/gdn_flip_mover.py:65-69`.
3. The flip consumers already accept explicit per-stage ordinal tuples
   (`layers/dcp/phase_flip_plan.py:48-83`), so nothing downstream is blocked
   by this choice if it is ever revisited.

### 4c. The explicit `--pp-layer-ratio` path is no longer unexamined

An explicitly spelled-out ratio reached `get_pp_indices` without ever
meeting the hybrid check that the derived path applies, so a list leaving a
stage with **no full-attention layer — an empty KV pool** — was accepted
silently. `--pp-layer-ratio 3,45,16` on the reference checkpoint is exactly
that, and at base commit `03b6fb990d` it boots.

The gate now applied is pure geometry read off the declared layer kinds: no
probe, no measured rates, no calibration. It refuses precisely what
`derive_pp_layer_split` already refuses and nothing more, and it stands down
entirely when the checkpoint is not a hybrid or its layer kinds are
unreadable — it never invents a census. Proven new: the test file run
against the base commit fails `test_zero_full_attention_stage_is_refused`
with "ValueError not raised".

Deliberately **not** included in that gate: any memory or performance
verdict. Both need §1e's calibration; a gate is only worth having if it is
right.

### 4b. Not done

* **The planner does not yet solve the cut at boot.** `solve_pp_cut` is a
  library function with no call site in `server_args`; wiring it needs the
  probe artifact plumbed into PP parse time, and it should not be wired
  before §1e is calibrated. `--pp-attn-stage-ratio` is the manual surface
  that makes the solver's output usable today.
* **`validate_pp_cut`'s memory verdict is not wired.** The structural half
  of the generality gap is closed (§4c); the memory half is built and tested
  but uncalled, because with `fixed_overhead_mib = 0` it can only ever
  under-report occupancy. It would not produce false refusals — a lower
  bound refuses only the genuinely impossible — so it is safe to wire the
  moment §1e lands, and that is the recommended first use of the calibration.

---

## 5. Register

`PROD_BRINGUP_BENCH.md:2444` and `:2499-2504` are overturned; entry added to
`CONTRADICTIONS_REGISTER.md` (C29).

---

## 6. GPU

No window. `/spinning/gpu-arb/holder` was held by `656-successor47` for the
entire shift with a live heartbeat (checked 06:14Z, 06:35Z; serving up on
30030, cards at 17.4 / 26.6 / 17.2 GiB). Per the isolation rules I did not
contend for it. Nothing in this branch has touched a card.

---

## 7. Where I stopped

Hermetic work is complete and green. The next actor should, in order:

1. Calibrate `fixed_overhead_mib` from an at-rest boot (§1e) — this unblocks
   everything else.
2. Run arms A/C/D at a window (§3a). Skip B unless the corridor is the
   question.
3. Only then wire `solve_pp_cut` and `validate_pp_cut` into `server_args`
   (§4b).

Do not merge this into the flip line or `integration/r2` — the operator
sequences that.
