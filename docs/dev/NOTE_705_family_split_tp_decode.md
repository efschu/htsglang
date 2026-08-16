# #705 — family-split TP decode: desk pricing

Desk half of #705, a member of the #485 phase-matrix family applied to the
DECODE phase (as #702 was applied to prefill). Tree `/spinning/wt-602-slot2`.
**No GPU was touched.** Model: Qwen3.8-27B-INT8, 64 layers = 48 GDN
(linear-attention) + 16 full-attention, every layer MoE.

Proposal: place all 48 GDN attention modules WHOLE on rank0 (the 5090) instead
of TP-sharding them, so their per-layer all-reduces vanish. Attention side stays
sharded with a DCP vector at the bandwidth proportion 2.4:1:1.

---

## 0 — Verdict

> **Revision 2 correction (planner-rules pass).** Re-expressing this verdict as
> planner rules (`planner/family_split.py`, per the PLAN_PERF_PIPELINE binding
> directive) surfaced an error below: §2 priced concentration against an **equal
> 1/3 shard**, where the slow 3080s bind. But this fork ships **uneven TP**
> (`--rank-tp-ratio`), so the honest baseline is a **bandwidth-proportional**
> shard, under which every rank finishes together and the family costs
> `bytes / sum(bandwidth)` = 1.73 ms rather than 2.51 ms. Against that faster
> baseline concentration has more to repay and **the gate roughly doubles, from
> 14.3 us to 30.5 us**. The 14.3 us figure below understated the bar; the
> solver derives both and `test_the_baseline_shard_POLICY_moves_the_threshold_twofold`
> pins them. Everything else in this note stands.

**The desk-net is positive if and only if a blocking TP all-reduce costs more
than 30.5 microseconds on this rig (14.3 us against an equal-shard baseline).** Everything else is priced below; the whole
question collapses to that one threshold, which is measurable and is the only
number this desk could not source.

The cost side is much SMALLER than the ticket's back-of-envelope (+0.69 ms, not
~+5 ms) and the capacity objection is world-neutral. But two premises in the
proposal are wrong, and one of them cuts against it.

## 1 — Premise corrections

**(a) "20 GB of GDN weights" is ~4x too high for the part that moves.** The MoE
block is **77.7 % of all weights** and it does NOT move — it stays sharded, on
every layer. What relocates is only the GDN *attention* module:

| family | per layer | total | moves? |
|---|---:|---:|---|
| GDN attention module | 110.5 MiB | 5,304 MiB | **yes** |
| full-attention module | 70.0 MiB | 1,120 MiB | no |
| MoE (residual of the 450.7 census flat) | ~350 MiB | 22,420 MiB | **no** |
| total (census) | 450.7 | 28,845 MiB | |

**(b) "75 % of depth becomes sync-free" is FALSE, and this is the finding that
cuts against the proposal.** Every layer is MoE — `qwen3_next.py:580`,
`is_layer_sparse = True` unconditionally — and the MoE block issues its own
all-reduce whenever `moe_tp_size > 1`
(`layers/moe/fused_moe_triton/layer.py:2047`, `:2062`). So **no layer becomes
sync-free at any depth.** Per decode round under TP=3:

* 64 MoE all-reduces — unchanged by family-split;
* 64 attention-side all-reduces — 48 of these vanish.

That is **48 of 128, i.e. 37.5 % of collectives removed**, not 75 % of depth.

**(c) A correction that cuts FOR the proposal.** The all-reduces removed are the
*blocking* kind: `RowParallelLinear.forward` calls
`tensor_model_parallel_all_reduce` synchronously (`layers/linear.py:2340`). The
surviving MoE all-reduce is **deferred** and joined in the next layer's
`prepare_attn` (`communicator.py:888-891`, #597), i.e. already latency-hidden.
So the 48 removed collectives are worth more per unit than a count comparison
against the 64 survivors suggests.

## 2 — The bandwidth cost, priced

Bandwidths 1.79 TB/s (5090) and 0.76 TB/s (3080), ratio 2.36 — which is where
the 2.4:1:1 DCP vector comes from.

| term | sharded TP=3 | solo on 5090 | delta |
|---|---:|---:|---:|
| GDN attention weights read/round | 1,768 MiB each -> **2.439 ms** (3080-bound) | 5,304 MiB -> 3.107 ms | **+0.668 ms** |
| GDN state read+write /round (bs=1) | 48 MiB each -> 0.066 ms | 144 MiB -> 0.084 ms | +0.018 ms |
| **total** | | | **+0.686 ms/round** |

Note the sharded case is bound by the **3080s**, not the 5090 — which is
precisely why solo-on-5090 costs so little: it trades three slow readers for one
fast one. The ticket's premise correction ("5090 solo 1.79 TB/s loses to 3.3
TB/s aggregate") is right in aggregate but the *binding* term is the slowest
rank, so the realised loss is 0.67 ms rather than the naive aggregate ratio.

**Break-even: 0.686 ms / 48 removed collectives = 14.3 us per all-reduce.**

For scale: the payload is a 10 KB-class tensor (hidden 5120 x 2 B) per
collective, so on this no-P2P rig the collective cost is latency-dominated, not
bandwidth-dominated — 14.3 us is a low bar for a PHB round trip, which is why
this desk expects the gate to pass rather than fail.

## 3 — Capacity ledger

| | rank0 (5090) | each 3080 | world |
|---|---:|---:|---:|
| GDN attention weights | +3,536 MiB | -1,768 MiB | 0 |
| GDN states (12 mamba slots, unsharded vs 1/3) | +1,152 MiB | -576 MiB | 0 |
| **net** | **+4,688 MiB** | **-2,344 MiB** | **0** |

**The world capacity is exactly neutral**, and here that conservation is
*correct* rather than the #702 error: decode really is the TP phase, the pool
really is the sum of per-rank capacities, and the DCP vector really can relieve
a tight rank. #702's mistake was applying this rule to PP prefill, where the
pool is layer-sharded and takes the min.

**The residual tension the ticket asked about.** Capacity says rank0 should
carry a SMALLER token share (it just gave up 4,688 MiB); bandwidth says 2.4:1:1,
i.e. rank0 carries 54.5 %. These pull opposite ways, and under DCP the vector is
free, so the arm must state which it optimises. Recommended: hold 2.4:1:1 and
absorb the 4,688 MiB out of rank0's KV share, because the world pool is
conserved and the 3080s have just been handed 2,344 MiB each to take it up.
Per #115, ranks 1 and 2 become zero-shard for the GDN family — an intermediate
rung, not all-or-nothing, exactly as #324's per-(rank,family) ratios allow.

**Ledger gap, stated:** the census on disk (`/spinning/evidence-665-f1/census-602/`)
is the **PP** layout (rank0: 7 attn + 21 linear, graphs 9,766 MiB). There is no
TP-decode census, so the per-rank *fit* is unverified. The world-neutrality
above is an accounting identity and does not depend on it; the claim that rank0
can absorb +4,688 MiB does.

## 4 — The one A/B arm, conditional

**Do not arm until the break-even is measured.** The desk cannot source the per
all-reduce cost; `CollectiveClock`
(`python/sglang/srt/utils/collective_clock.py`) is the instrument.

*Gate*: measure the blocking TP all-reduce cost at bs=1 decode on the current
build. If it is **> 14.3 us**, arm; if below, the proposal is refused on its own
arithmetic and no boot is needed.

*Arm* (exactly one): control = current TP=3 decode; treatment = GDN family
whole on rank0, ranks 1/2 zero-shard for the GDN family, full-attention and MoE
unchanged, DCP vector held at 2.4:1:1.

*Measure*: ms/round split compute vs wait per rank — the wait column is the
whole point, since the claim is collective elimination. Expect rank1/rank2 wait
to RISE (they idle through the 48 solo GDN modules) while total round time
falls; if total falls but wait rises more than compute drops, the bubble was
moved rather than removed.

*Falsifier*: rank0 compute per round should rise by ~0.69 ms and no more. If it
rises materially further, the solo GDN path is not weight-bandwidth-bound the
way §2 models it and the pricing is void.

*Composition*: this is another #704 ladder rung and composes with the rungs
already there; it is a decode-phase member of the #485 matrix, so it must not be
co-armed with a prefill-cut change (#702) — one phase at a time or neither
result is attributable.

---

## 8 — VERDICT (desk, measured): REFUSE the A/B. Do uneven TP instead.

The gate is no longer a threshold to guess at. The on-rig collective cost is
**measured**: `ar_10kb_us` = **31.0-33.7 us** for the 10 KB bs=1 payload
(`INTEGRATION_R3_VALIDATION`, `__group__` rows; ANALYSE_321 uses 32.4). The 44.92
us figure in the tier tables is **UCX to a remote rig over 40G RoCE** and does
not apply here. Total decode collective cost is ~4.1 ms/round at 128
collectives/token, of which this proposal removes 48.

### The three baselines, and why the answer hinges on which one is honest

| GDN attention family, per decode round | ms |
|---|---:|
| solo on the 5090 | 3.192 |
| **EQUAL 1/3 shard** — what the rig runs today (`rank_tp_ratio=None`) | 2.506 |
| **PROPORTIONAL shard** — uneven TP, `--rank-tp-ratio`, already shipped | **1.726** |

Collective saving from removing 48: **1.488 - 1.618 ms** across the measured
range.

| net per round | vs EQUAL (today) | vs PROPORTIONAL (uneven TP) |
|---|---:|---:|
| at 31.0 us | +0.802 ms | +0.022 ms |
| at 32.4 us | +0.869 ms | +0.090 ms |
| at 33.7 us | +0.932 ms | +0.152 ms |

### The finding

**Uneven TP alone is worth +0.780 ms/round, at zero capacity cost and no
structural change.** It is already shipped and simply not enabled here
(`rank_tp_ratio=None`).

**The family split beyond uneven TP is worth +0.090 ms/round** — about **0.3 %
of a ~30 ms bs=1 round** — in exchange for concentrating a family on one rank,
moving 4,688 MiB of residency, and taking on the #115 zero-shard machinery.

That is not a trade worth a window. **Refused**, and the A/B is not requested.

### Capacity ledger (not the blocker, but it was asked for)

Priced against the measured TP-stack sizing pass (holdback 0.000 % there, so
these are the real KV budgets):

| | rank0 | rank1 | rank2 | world |
|---|---:|---:|---:|---:|
| current KV, MiB | 20,118.6 | 12,566.3 | 12,072.3 | |
| tokens | 643,795 | 402,122 | 386,314 | **1,432,230** |
| after split, MiB | 15,430.4 | 14,910.4 | 14,416.4 | |
| tokens | 493,773 | 477,133 | 461,325 | **1,432,230** |

**World delta: exactly 0 tokens.** The split is capacity-neutral under the TP
sum rule and every budget stays positive, so it *fits* — rank0's share simply
falls from 45.0 % to 34.5 %. Capacity was never the obstacle; the win was.

Note the premise correction from section 1 still stands and is load-bearing
here: only **5,304 MiB** of GDN *attention* relocates, not "20 GB". The MoE
block is 77.7 % of weights and stays sharded on every layer, which is also why
no layer becomes sync-free and only 48 of 128 collectives are removable.

### If it is ever revisited

The order is fixed by the numbers: enable uneven TP first and re-measure. The
family split's remaining value is whatever survives *after* the proportional
shard, and today that is 0.09 ms. It becomes interesting only if the collective
cost rises materially (a slower interconnect, or many more ranks) or if the
GDN attention family grows relative to MoE — neither is true on this rig.
