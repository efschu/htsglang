# Pricing the full plan's crossings: breaks, transport, and which regime pays

Date: 2026-08-17. Desk only, no boots. Input to `DESIGN_family_fullplan.md`
(Slot-2, `dd75cfc1cf`); its layer map and 31-crossing derivation are taken as
GIVEN and not re-derived here.

**Headline: the amendment inverts the answer. In PREFILL — the plan's actual
home — breaks are negligible and TRANSPORT dominates, and the BAR1 p2p kernel
measurably does NOT pay in that payload band. The capturability argument
belongs to the decode arm, which is the arm Slot-2 says not to adopt.**

## 0. Why this note is priced per PASS, not per token

Slot-2's answer #1: the full plan replaces the flip's **prefill** phase; it is
"decode-unfavourable by construction and prefill-favourable (crossings amortise
over many tokens)". A prefill pass crosses each boundary **once for the whole
chunk**, so `N_breaks = 31 per pass`, not per token. That single change moves
the break term by the chunk size — 512x at the live setting.

Both regimes are priced below, because the decode arm has to stay on record.

## 1. Break count: 31, and no batching floor below it

Given the stride-4 map, `N_breaks` = card switches in execution order, not
crossings-as-payloads. Recomputed from `layer_types`: 15 interior FA layers
contribute 2 switches each (in and back) and the terminal FA at 63 contributes
1 = **31**. That already counts consecutive same-card layers as one segment, so
**31 is simultaneously the worst case and the floor** under this map. Lowering
it means changing the interleave, which is the layer map — Slot-2's domain, not
this note's.

## 2. Break cost: the instrument, and the number that does not exist

`break_cost_clock.py` is **not on this branch**: `ls
python/sglang/srt/utils/break_cost_clock.py` fails, and the catalog entry citing
`utils/break_cost_clock.py:201` lives on a lineage this branch does not carry.
EXISTS-other-lineage.

**Scope of the absence claim, stated rather than implied:** what I searched is
this branch's tree and `git log --all --grep=494`. A broader hunt across battery
artifacts and other lineages was dispatched and had not returned when this note
was written, so the claim here is "no per-break number is reachable from this
branch", NOT "none exists anywhere". If one turns up it replaces the parameter
below; it does not change the thresholds, which is why this note ships without
waiting for it.

The model is therefore parameterised on `break_cost`, and the window measurement
has a decision threshold waiting for it rather than a number to confirm.

## 3. PREFILL — transport dominates, breaks vanish

Per-crossing payload is the whole chunk, not one token: `C x 5120 x 2 B`.

| chunk | per-crossing | 31 crossings |
|---|---|---|
| **512** (live, `--chunked-prefill-size 512`) | **5.0 MiB** | 155 MiB/pass |
| 2048 (observed in the window-8 log) | 20.0 MiB | 620 MiB/pass |

At the measured pairwise link bandwidths (5.10 / 9.06 / 5.77 GB/s, hardware
profile `__group__` rows; ~5.6 GB/s taken as the typical pair), the **29 EXTRA**
crossings over the incumbent's 2 cost:

```
chunk  512:  0.94 ms/crossing -> 29x =  27.2 ms   =  5.7 % of a ~476 ms pass
chunk 2048:  3.74 ms/crossing -> 29x = 108.6 ms   =  5.7 % of a ~1906 ms pass
```

The ratio is chunk-invariant, because payload and pass length scale together.
(The 1906 ms reference is my own #588 read of a 2048-token prefill batch;
the 476 ms figure is that scaled linearly to 512 — **INFERRED**, and the one
number here worth re-measuring.)

Breaks, by contrast, are amortised over the whole chunk:

| break_cost | 31 breaks/pass | share of a ~476 ms pass |
|---|---|---|
| 10 us | 0.31 ms | **0.07 %** |
| 41.5 us | 1.29 ms | **0.27 %** |
| 100 us | 3.10 ms | **0.65 %** |
| 200 us | 6.20 ms | **1.30 %** |

**Verdict for prefill: the break term is negligible at any plausible break cost,
and does not gate the plan.** Even a 200 us break — well above a normal segment
relaunch — costs a fifth of what transport costs. The prefill decision is a
transport-bandwidth decision.

## 4. DECODE — kept on record, and the arm where breaks bite

At bs=1 the payload is Slot-2's exact 10240 B and breaks land **per token**:

```
transport  ~0.212-0.227 ms/token   (31 crossings x ~7.3 us)
breaks     31 x break_cost PER TOKEN
```

Break-even against a ~30 ms round, i.e. the break cost at which crossing
overhead consumes a given budget:

| budget of the round | break-even `break_cost` |
|---|---|
| 1 % (0.30 ms) | **2.8 us** |
| 2 % (0.60 ms) | **12.5 us** |
| 5 % (1.50 ms) | **41.5 us** |
| 10 % (3.00 ms) | **89.9 us** |

**So the decode arm's decision threshold is ~41.5 us at a 5 % budget.** That is
squarely inside the plausible range for a graph-break, which is exactly why the
measurement decides it rather than intuition — it is neither obviously fine nor
obviously fatal. This is the threshold the window measurement should carry.

## 5. Does the BAR1 p2p kernel still pay? — regime-split, and NO for prefill

With `barlink_host.send`/`recv` already present (`barlink_host.py:1100`,
`:1120`), the wire exists today and the BAR1 kernel is an optimisation. Its two
possible values separate cleanly by regime:

**PREFILL — it does not pay, and this is measured, not inferred.** The crossing
payload is 5-20 MiB point-to-point, i.e. a **2-rank** pairing. The measured
2-rank BAR1 ratios are **0.86-0.99x in exactly the 1-8 MiB band**, with the x8
pair losing down to **0.81x** (`FEATURES_VS_UPSTREAM.md:1349`). BAR1's gains are
3-rank collective gains and must not be borrowed here. A kernel that is
measurably at-or-below parity in the plan's own payload band cannot be justified
by transport speed, and the capturability it would buy is worth <=1.30 % of a
pass (section 3).

**DECODE — capturability, and only capturability.** At 10240 B the transport
saving is bounded by the whole transport term, <=0.227 ms/token, while removing
31 breaks/token saves `31 x break_cost`. Those are equal at
**break_cost = 6.8 us**; above that, capturability is worth more than the entire
transport, and at the 41.5 us threshold it is worth ~6x it. So in decode the
kernel's case is essentially all capturability.

**The two combine into an awkward conclusion, stated plainly:** the kernel does
not help the regime the plan lives in, and the regime where it would help is the
one Slot-2 argues against adopting. **The BAR1 p2p kernel is therefore not on
the critical path for the full plan** — it is a decode-arm optimisation whose
arm is not currently wanted.

## 6. What this means for the critical path

Slot-2 names two blockers: the WIRE (mine) and the ADDRESSING (`get_pp_indices`
returns an interval; PP cannot express a non-contiguous map). This note adds a
third fact rather than a third blocker:

* the wire **already exists** for prefill via `barlink_host.send/recv`;
* the BAR1 kernel is **not** what unblocks it;
* so the remaining blocker for the prefill plan is Slot-2's **addressing** work
  plus a transport-bandwidth decision at ~5.7 % of a pass.

**Filed as window items:** (1) a per-break measurement to resolve section 4's
decode threshold, (2) a prefill-pass reference time at the live chunk size to
replace section 3's linear scaling, (3) a point-to-point bandwidth row at 5 MiB,
since section 3 rests on the profile's pairwise link rows rather than on a p2p
measurement at the plan's own payload size.
