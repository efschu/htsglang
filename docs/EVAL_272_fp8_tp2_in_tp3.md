# EVAL — Qwen3.6-27B-FP8 as a dual group with rank reuse (tp2-in-tp3)

Desk evaluation of PRIO-Nachtrag 7 (DESIGN_201 §1415-1427), computed with the
#272 key solver at `29bf7ced20` + a small nesting-bound interface. CPU only,
no GPU, no boot, no webui touch. Scripts:
`/root/.claude/jobs/1481bb40/tmp/eval_fp8_tp2_in_tp3{,_scale,_pairs}.py`.

Configuration under test: a PD lane as uneven-DCP **TP=2** on the 5090 plus
the x8-attached 3080, and a main lane as uneven-DCP **TP=3** over all three
cards, whose shards on the two shared cards are **subsets** of the PD lane's
(`u_r <= v_r`). The third card carries the complement, so the full weights
live in the rig exactly once.

---

## 0. Result in four lines

1. **It fits — but only in a narrow corner**, and the binding post is not the
   weights, it is the **per-lane GDN state pool**. At the recipe's
   `max_running_requests=16` per lane the 5090 is over by 9.0 GiB. At
   `mrr_pd<=2, mrr_main<=4` it closes with 0.8-1.8 GiB to spare.
2. **The tightest card is the 5090, not the x8-3080** — the opposite of the
   briefing's expectation, and for a reason worth knowing (§1.3).
3. **The nesting coupling is cheap: 4.2 % decode, 4.3 % prefill, 0 % on KV
   and sessions.** Multiple directions carry; the PD direction does not
   dominate. What *does* bind is the co-residence bracket, which rules out
   the PD lane's prefill-optimal key for a different reason (§2.3).
4. **The aggregate KV figure the merged solver prints is wrong for this
   case, by 3.3-4.8x**, and that is a defect of `aggregate()` worth fixing
   (§4.1). The corrected joint capacity is ~240-343k tokens, not 1.14M.

---

## 1. Feasibility bracket

### 1.1 Device resolution first — the label is not the card

The briefing names "3080a, the x8-attached one". Resolved from the profile
rather than from the label, because the two orders diverge on this rig:

| cuda_index | pinned H2D / D2H | ordered bandwidth to the 5090 |
|---|---|---|
| 1 | 6.47 / 6.58 GB/s | 4.52 GB/s |
| **2** | **13.4 / 13.16 GB/s** | **6.88 GB/s** |

Both signals agree, and they agree by a factor of ~2 — the signature of x8
against x4. **The x8-attached 3080 is `cuda_index 2`.** The PD lane is
therefore `--rank-gpu-id 0,2` and the complement card is `cuda_index 1`. A
run that took "3080a" to mean index 1 would put the PD lane on the *slow*
card and silently invalidate the whole comparison.

### 1.2 The bracket at the recipe's settings — does not fit

Both lanes at `max_running_requests=16`, reserve 3000/2700/2700, shared
weights counted once:

| GPU | weights (shared, once) | other (both lanes) | process post | claimed | available | headroom |
|---|---|---|---|---|---|---|
| 0 (5090) | 20558 | 17916 | 1536 | 41494 | 29607 | **-11887** |
| 2 (3080 x8) | 7494 | 13315 | 1536 | 22345 | 17780 | **-4565** |
| 1 (3080 x4) | 9777 | 5507 | 0 | 15284 | 17780 | +2496 |

All figures MiB. The refusal names its posts: of the 17916 MiB of "other" on
the 5090, **13036 MiB are two GDN state pools** (main 5368 + PD 7668) and
4608 MiB are two per-rank overhead allowances. The weights are *not* the
problem — sharing already saves 14221 MiB on the 5090 alone, and naive
duplication would be 12.6-16.7 GiB worse still.

### 1.3 Why the 5090 is the tightest card, not the x8-3080

Three effects stack on it and only on it:

* both lanes' capacity-optimal keys **concentrate MLP mass on the largest
  card** (max-min on free bytes gives the big card more weight), so the
  shared weight maximum lands there;
* the GDN state pool follows the unit partition, which follows the budget —
  so the 5090 carries the largest share of **both** state pools;
* it is one of the two shared cards, so it also pays the extra process post.

The x8-3080 carries two lanes as well but a smaller share of each. The
briefing's assumption -- that the tightest card would be the x8-3080 --
does not hold.

### 1.4 Which dimension has to be scaled — and by how much

`max_running_requests` per lane, swept, worst headroom over all three cards:

| mrr_pd \ mrr_main | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **1** | +1759 | +1432 | **+777** | -565 | -3249 |
| **2** | +1379 | +1051 | +397 | -945 | -3629 |
| **4** | +618 | +291 | -364 | -1706 | -4390 |
| **8** | -950 | -1277 | -1932 | -3274 | -5957 |
| **16** | -3998 | -4325 | -4980 | -6322 | -9006 |

MiB; the worst card is GPU 0 in every cell. Against the corridor rule
(>= 400 MiB free), the usable corners are `mrr_pd<=2` with `mrr_main<=4`,
plus `(4, 1)`. `(4, 2)` at 291 MiB is below the floor and should be treated
as red, not amber.

Nothing else moves the wall meaningfully: the weights are already shared, the
reserve is the runbook 4.1 recipe, and dropping the KV floor only buys the
408+272 MiB of minimum pool.

**One reading changes the table.** The post above charges 1536 MiB for a
second engine process on each shared card. If the dual-group runtime (#274)
puts both groups in **one** process, that post disappears and every cell
gains 3072 MiB — `(1,8)`, `(2,8)` and `(4,4)` would then close too. Which of
the two it is, is a property of the runtime and not something this model can
decide; both readings are stated rather than one being assumed.

Since it is a property of the runtime and not of the plan, it is now a
**parameter** rather than this footnote: `coexistence(..., shared_process=True)`
and `aggregate(..., shared_process=True)`. On candidate A it lifts the joint
KV from 240361 to 332883 tokens and the 5090's headroom from 777 to 2313 MiB.

---

## 2. The price of the nesting coupling

Best TP=3 key **with** the nesting box against the free TP=3 solve, at the
feasible corner (`mrr_pd=1, mrr_main=4`), PD key = its capacity optimum
`59,9`:

| main goal | free key | free value | nested key | nested value | **price** |
|---|---|---|---|---|---|
| dec | 30,17,21 | 93.9 tok/s | 69,18,49 | 90.0 tok/s | **-4.15 %** |
| enc | 1,0,0 | 1419 tok/s | 59,4,5 | 1358 tok/s | **-4.30 %** |
| maxkv | 116,5,15 | 769557 tok | 116,5,15 (unchanged) | 769557 tok | **0 %** |
| sessions | 116,5,15 | 5 | 116,5,15 (unchanged) | 5 | **0 %** |

**The coupling is cheap, and the reason is geometric.** The box is
`u_0 <= 118` on the 5090 and `u_1 <= 18` on the x8-3080, against 136 units in
total. Each goal meets a different wall, or none:

* **decode** wants `60,34,42` units. The 5090 ceiling is nowhere near it
  (60 of 118), but the x8-3080 ceiling is: 34 wanted, 18 allowed. Pushing the
  surplus onto the x4-3080 costs the 4.15 %.
* **prefill** wants all 136 units on the 5090 and may take 118. That single
  ceiling costs the 4.3 %.
* **maxkv / sessions** want `116,5,15` — already inside the box on both
  cards, so the coupling is free. They are also flat in the key here anyway
  (the invariance in §2 of the solver's docstring: with budgets fixed, the
  key moves KV between cards rather than creating it).

**Verdict on "several directions or only one": several directions carry.**
A 4 % ceiling on two of four goals and nothing on the other two is not a
regime where the PD direction dominates. The user should optimize both lanes
for their own purpose.

### 2.2 The coupling gets expensive only at the extreme PD key

If the PD lane is given its *prefill* optimum `1,0` (everything on the 5090),
the main lane's decode price rises to **-8.3 %**, because the ceiling on the
x8-3080 becomes zero units. That is the case where one direction really would
dominate — and it is also the case that does not fit (next section).

### 2.3 What actually binds is the bracket, not the nesting

The PD prefill-optimal key `1,0` is feasible *standalone* and predicts 2376
tok/s prefill against 1882 for the capacity-optimal `59,9` (+26 %). But
concentrating all MLP mass on the 5090 pushes the shared weight maximum there
to 22042 MiB, and the bracket then fails by **520 MiB** on that card even at
`mrr 2/2`. So the PD lane cannot be run at its prefill optimum in this
configuration — not because of the nesting, but because of co-residence.
Naming the two separately matters: the first is a design choice, the second
is a wall.

---

## 3. Three concrete key pairs

All at reserve 3000/2700/2700, `--rank-gpu-id` in the resolved order
(PD `0,2`; main `0,2,1`), context target 8192, NEXTN 3/1/4, `fp8_e4m3` KV,
`SGLANG_MAMBA_SSM_DTYPE=bfloat16`.

### Candidate A — balanced (recommended)

| | PD lane (TP=2) | main lane (TP=3) |
|---|---|---|
| key | `--rank-mlp-ratio 59,9` | `--rank-mlp-ratio 69,18,49` |
| units | 118 / 18 | 69 / 18 / 49 |
| mrr | 1 | 4 |
| prefill | 1882 tok/s *(estimate)* | 1156 tok/s *(estimate)* |
| prefill, anchored cell | absent (no TP=2 anchor) | 1098 tok/s *(estimate)* |
| decode | **absent** (no TP=2 anchor); ratio 0.904 vs its own base | 90.0 tok/s *(estimate)* |
| KV alone | 374063 tok | 769557 tok |
| **KV co-resident** | **36715 tok** | **203646 tok** |

Bracket: 5090 +777 MiB, x8-3080 +1119, x4-3080 +5407. Aggregate prefill
**3038 tok/s** (upper bound, see §4.2). Naive duplication of the same pair
misses by 12.6 GiB — the sharing is the whole reason this exists.

### Candidate C — main lane on prefill

Same PD key and corner; main lane `--rank-mlp-ratio 59,4,5` (units
118 / 8 / 10). Main: prefill 1622 tok/s first-principles (anchored cell 1358
tok/s), decode 74.5 tok/s, KV co-resident 306227 tok. Aggregate prefill
**3504 tok/s**. Bracket +777 / +1119 / +8813 MiB.
Buys +40 % main prefill for -17 % main decode against A.

Two prefill numbers appear per lane on purpose and are not interchangeable:
the **anchored cell** scales a measured TP=3 split-probe rate by the
predicted ratio, and the **first-principles** figure (used in the aggregate,
because the PD lane has no anchor at all) is the two-term roofline of §4.4.
They differ by 5-19 % here, which is inside that model's stated band.

### Candidate B — PD lane on prefill — **REJECTED**

PD `1,0`, main `10,0,7`, mrr 2/2. PD prefill 2376 tok/s, main decode 86.1.
**Bracket fails: 5090 over by 520 MiB.** Listed because it is the natural
first instinct and it is the one that does not work; the 520 MiB is small
enough that a one-process runtime (§1.4) would close it, so it is a
wiedervorlage candidate rather than a dead end.

---

## 4. Caveats — what the profile does not support

### 4.1 The aggregate KV figure of `aggregate()` was wrong here — now fixed

`estimate_instance` sizes each lane's capacity **as if that lane were alone
on its cards**, and `aggregate()` then sums those. For lanes that share
cards this double-counts the same free bytes. Corrected by re-solving each
lane against its #260 co-residence budget share:

| candidate | solver's summed KV | corrected joint KV | over-count |
|---|---|---|---|
| A | 1 143 619 | **240 361** | **4.76x** |
| C | 1 143 619 | **342 942** | **3.33x** |

The bracket itself was right (it never used the capacity figure); it was the
KV **cell** of the aggregate that was unsafe for overlapping instance sets.

**Fixed** (`fix/solver-aggregate-coresident`): when any card carries more
than one lane, `aggregate()` now re-sizes every lane against its
co-residence share via `coresident_budgets()` — the same #260 mapping used
by hand above — and the KV cell says so, naming the over-count it removed.
Where the mapping cannot be built (an overflowing card, a lane with no local
footprint) or where a lane's share leaves it unable to fund a KV pool at all,
the cell is `absent` with the reason rather than a number. Disjoint lanes are
untouched and still simply sum. The two candidates of this document are the
regression test; the module now reproduces 240361 and 342942 exactly.

### 4.2 Interference between the lanes is named, not estimated

The 5090 and the x8-3080 are computed by **both** lanes. The model has no
time-multiplex term, so every aggregate throughput above is an **upper bound
under zero interference**. The true value is bounded:
`max(lane) <= real <= sum(lanes)`, i.e. for candidate A between 1882 and
3038 tok/s. Nothing in the profile constrains where in that interval it
lands — that needs a measurement (two lanes booted, both driven), and it is
not guessed here.

Additionally, two engine processes on one physical GPU are gated on this rig
by **NCCL 2.28.9 < 2.30** and by an MPS daemon. That gate is orthogonal to
the arithmetic above and applies only to the two-process reading.

### 4.3 What the nesting box does and does not prove

`u_r <= v_r` is **necessary** for reuse, not sufficient. Containment must
also hold on the axes the MLP vector does not carry — attention, GDN and
vocab shards — and the unit ranges must be laid out **contiguously**, so the
inner shard is an interval of the outer one rather than merely smaller. Both
are properties of the partition layout; the solver models counts. For this
configuration the count condition is satisfied on the other axes too
(TP=2 gives each shared card a larger attention/GDN share than TP=3 does), so
a contiguous layout exists — but it has to be built, and a satisfied box must
not be read as proof that the bytes are shared.

### 4.4 Absent values, and gate-5 discipline

* **PD-lane decode and prefill in tok/s are `absent`.** The split-probe store
  holds only TP=3 rows for this checkpoint, so the TP=2 lane has no absolute
  anchor. Its relative figures are reported instead (decode ratio 0.904,
  prefill ratio 1.230 against its own base plan), and the 1882 / 2376 tok/s
  prefill numbers come from the first-principles model, whose error is the
  documented one-sided ±25 % band (-19 % / +15 % / +11 % on the three
  measured arms; collective-bearing arms predicted too fast). Fixing this is
  one `split_probe` run at `tp_size=2`.
* **Every throughput above is `estimate`.** Nothing in this document is a
  measurement.
* **The per-lane state pool is the model's own sizing**, not a measured
  residency; the measured-registry path was not used because no boot of this
  configuration exists.
* Both lanes were priced with the **same** reserve as runbook 4.1. A dual
  group may well need a larger one; that would move every cell of §1.4 down.

---

## 5. Recommendation

Run **candidate A** if it is run at all: PD `59,9` at `mrr 1`, main
`69,18,49` at `mrr 4`. It is the only corner that is both feasible and above
the VRAM corridor floor with room to spare, the coupling costs 4 %, and the
sharing saves 14 GiB on the 5090.

But the honest headline is the one in §1.4: **at the concurrency this rig's
recipes normally use (`mrr 16`), the dual group does not fit at all**, and
even at the feasible corner the two lanes together hold ~240k KV tokens where
the main lane alone holds 837k. The configuration buys a second, prefill-fast
lane by giving up roughly three quarters of the rig's KV and almost all of
its concurrency. Whether that trade is worth making is a workload question —
it pays for bursty TTFT-critical prefill against a small session count, and
loses badly for anything KV-hungry or concurrent.

The first measurement that would sharpen this materially is a `split_probe`
at `tp_size=2` on the two PD cards (turns four `absent` cells into
`measured`), and the second is a two-lane boot to put a number on §4.2.
