# NOTE 470 — the residency cut, priced in bytes before the window spends a boot on it

Desk companion to `TICKET_470_dspark_boots.md`. **No GPU number here.**

TICKET_470's gate is: *"If Boot A's residency cut costs more than the best draft
arm returns, R1 is refuted"* (`TICKET_470_dspark_boots.md:161-169`). Boot A
exists to price that cut. This note fixes the **memory** half of the price
exactly, at desk, so Boot A only has to measure the **performance** half — and
so the window knows before it starts whether the cut is small enough to be
plausible or large enough that the arm is already in trouble.

Instrument: `scripts/dev/478_quant_footprint/gguf_footprint.py`
(`--extra-vram-gib`), whose selftest proves it discriminates — a co-resident
head must cut residency and move exactly its own size to the host pool.

## The head

The DSpark draft head is **10.12 GiB**, of which 9.562 GiB is MXFP4 routed
experts (ANALYSE_447 §1.5, `:166-234`). MXFP4 Marlin is **SM90/SM120 only**
(`mxfp4_marlin_moe.py:116-117`), so the head can only sit on the 5090. On this
rig the 5090 is **cuda:0**, which carries **rank 0** — so the cut lands on the
one rank that has the largest expert shard. There is no placement freedom here.

## The cut, in bytes

Active driver UD-IQ3_XXS, TP=3, context 8192, rank0 = 5090:

| | without head | with 10.12 GiB head | delta |
|---|---|---|---|
| rank0 resident experts | 24.29 GiB | 14.17 GiB | **−10.12** |
| rank0 resident fraction | 0.611 | **0.356** | −0.255 |
| rank0 spill to host | 15.48 GiB | 25.60 GiB | +10.12 |
| host cold pool (all ranks) | 36.28 GiB | **46.40 GiB** | +10.12 |
| rank1/rank2 | unchanged | unchanged | 0 |

**The head costs rank0 42 % of its resident experts** (24.29 → 14.17 GiB). It is
paid entirely by rank 0; the two 3080s are untouched.

## Why that is the number that matters

The #394 window measured converged expert hit rates of tp0 **0.7794**, tp1
0.8398, tp2 0.8272 at the residency the recipe funds. Rank 0's hit rate is
already the **lowest of the three** — it holds the biggest shard on the biggest
card, so its resident *fraction* is the least favourable — and this cut removes
42 % of exactly that rank's resident set. Every additional miss is an H2D fetch
of a whole expert over PCIe on a rig with **no P2P and no NVLink** (all PHB).

Two structural consequences the window must not discover the hard way:

1. **The cut is not symmetric and the slowest rank sets the clock.** Per the
   per-barrier rule, rank 0 becoming the pacesetter is the expected failure
   mode, not a surprise. Boot A must therefore report **per-rank** ms/round, not
   only an aggregate — an aggregate hides exactly the asymmetry this cut
   creates.
2. **The comparison must be work-matched** (#482/#523,
   `expert_compute_placement.py:82-83`): both arms read at a common work point,
   enforced by `read_arm.py --against`. A residency cut changes the miss rate,
   which changes streamed bytes per token, which changes how far a run gets in a
   fixed wall time — so two arms sampled at equal wall time are sampled at
   *different* work points and their counters are not comparable.

## What this does not settle

The byte price is exact; the **time** price is not derivable at desk. Whether a
42 % residency cut on one rank costs more than a working draft arm returns
depends on the router's peakedness (`expert_stats.py:93-124` reports
`normalized_entropy` and `top-k share`) — a peaked router loses little when the
cold tail grows, a uniform one loses a great deal. `SGLANG_EXPERT_STATS=1` is
armed in every arm of this window anyway, so Boot A gets this for free.

## Consequence for the window

Boot A stays mandatory and stays first — the ticket's *"If Boot A cannot be run
at all, do not run Boot B: an unattributed multiplier is not a result"*
(`:161-169`) is unaffected by anything here. What changes is that Boot A now has
a **falsifiable prediction** to check itself against rather than an open
question: rank0 resident 24.29 → 14.17 GiB, host cold pool 36.28 → 46.40 GiB. If
the boot's own offload ledger disagrees with those two numbers, the model behind
this note is wrong and its conclusions are void — which is the point of writing
them down before the boot instead of after.

RAM headroom is not a constraint for this arm: 46.40 GiB of cold pool plus
~14 GiB of runtime overhead sits far under the 104.0 GiB `MemTotal`
(see ANALYSE_478 §3 for how that floor is derived).
