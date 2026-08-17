# BOOT PACKAGE — family cut [35,15,14] for F4-r4

Desk-only. No boot, no GPU, nothing executed. Base `4a16043d1a`
(`train/0817-control`). Every number below is REUSED from the arm that was
solved and filed today (`9c25330131`) or read at file:line — none is recomputed
blind, and the two I re-derived independently are marked as such.

---

## 0. What this arm is, and what it is NOT

**It is a MEMORY-RELIEF cut, not the family split.** The distinction decides how
to judge it, and #732 (`1dfa3af0b9`, landed today) makes it load-bearing:

* the FAMILY SPLIT — GDN compute onto the 5090, full-attention/KV onto the
  3080s — was re-priced against barlink BAR1 and **does not merely fail to pay,
  it costs**: net moves from `+0.022..+0.152 ms` to `-0.034..-0.356 ms`. No
  window ticket. A faster interconnect cannot make collective-REMOVAL more
  valuable.
* **[35,15,14] is the opposite direction** and was filed as "the one arm"
  precisely because the requested direction was worse on both halves at once.
  It raises GDN on the 5090 from 24 to 27 layers and RELIEVES both 3080s. Its
  value is pool headroom on the constrained cards, not collectives removed.

So: **do not judge this arm by collective savings.** #732 already priced that
axis negative. Judge it by per-stage VRAM relief and what the freed bytes buy in
KV pool.

---

## 1. Recipe

### 1.1 The cut vector

`--pp-stage-ratio` (`server_args.py:1473`) carries the layer split;
`--pp-attn-stage-ratio` (`:1493`) is the #485 decoupling that resolves the
attention family separately. Both are consumed by `_handle_pp_stage_ratio`
(`:15726`).

**Why the second flag is not optional here**: `derive_pp_layer_split` returns
`[32,16,16]` with FA `[8,4,4]` from the same scores WITHOUT it. The flag is
worth exactly one FA layer and one total layer off the big card — i.e. it is
what turns `[32,16,16]` into `[35,15,14]`. Booting the cut without it silently
tests a different geometry.

```
--pp-size 3
--pp-stage-ratio 35,15,14
--pp-attn-stage-ratio <the vector that resolves FA to 8,4,4>
```

**VERIFY BEFORE BOOT (desk could not settle it):** `--pp-stage-ratio` is
documented as SCORES, not literal layer counts, and `_handle_pp_stage_ratio`
derives the split from them. Whether `35,15,14` is accepted verbatim as counts
or must be expressed as scores is the one recipe item this desk did not confirm
— resolve it by calling the resolver both ways before the window, exactly as
`9c25330131` reports doing ("calling the real derivation both ways corrected
that" — the same trap already cost one wrong reading of 7,5,4).

### 1.2 #709 activation — and the trap that would void the arm

The shipped-dark lever is `--rank-tp-ratio` with an EXPLICIT vector:

```
--rank-tp-ratio 2,1,1
```

**NOT `auto`.** `--rank-tp-ratio auto` is CAPACITY-first — its own help says it
"does NOT optimize for speed: it maximizes the KV pool and ignores how fast the
cards are". On this rig that is the VRAM ratio ~1.6:1:1, **not** the
bandwidth-proportional ~2.36:1:1 (1.79 vs 0.76 TB/s) that #705's +0.780 ms was
derived from. **An arm booted on `auto` tests a different lever and its result
cannot be compared to the prediction.**

`2,1,1` is the practical vector because `sum(weights)` must divide every sharded
dimension (sum 4 divides 5120). Its ratio is 2.0 against an ideal 2.36 — a
**~15 % shortfall, so it cannot deliver the full +0.780 ms and must not be
judged as though it should**. `acceptance.admissible_ratios()` enumerates what
the checkpoint permits, and the #709 runner REFUSES an arm whose ratio flag was
not recorded verbatim.

### 1.3 Stage-to-card identity — resolve, never assume

Rank order is not card order. Resolve through the NVML canon:
`rank_card_vector()` / `rank_card_uuids()` (`registry/rank_cards.py:268`,
`:291`) and record the mapping in the boot report. The device-order canon exists
because NVML and torch enumeration diverge and both can shift across a driver
state change.

**The mapping matters for this arm specifically**: #690's constraint is fewest
layers on the x4-linked 3080, and the arm's own filing tracks that (the rejected
direction was called out for taking it from 17 to 21). Which physical card is
rank 1 must be READ, not assumed, or the #690 risk below is being reasoned about
for the wrong card.

---

## 2. Expected numbers

### 2.1 FA layers per stage — re-derived independently, and they agree

FA ids are `3,7,…,63` (period 4, 16 layers total).

| cut | stage 0 | stage 1 | stage 2 | total |
| --- | --- | --- | --- | --- |
| **[35,15,14] (arm)** | layers 0-34 → FA 3,7,11,15,19,23,27,31 = **8** | 35-49 → 35,39,43,47 = **4** | 50-63 → 51,55,59,63 = **4** | 16 |
| [31,17,16] (incumbent) | 0-30 → 7 | 31-47 → 5 | 48-63 → 4 | 16 |

I computed these from the id series rather than taking them on report; they
match the filed `[8,4,4]` and `[7,5,4]`.

### 2.2 KV share — the thing that actually moves

KV lives on the FA layers, so the per-stage KV share follows the FA counts:

| stage | today (7/5/4) | arm (8/4/4) | delta |
| --- | --- | --- | --- |
| 0 | 7/16 = 43.8 % | 8/16 = 50.0 % | **+1 FA layer of KV** |
| 1 | 5/16 = 31.2 % | 4/16 = 25.0 % | **−1 FA layer of KV** |
| 2 | 4/16 = 25.0 % | 4/16 = 25.0 % | unchanged |

So the arm moves one FA layer's worth of KV from stage 1 ONTO stage 0, while
also moving weight off both 3080s. Both directions relieve the small cards.

### 2.3 Per-stage VRAM, and where it is tight

Measured per-layer weights from the safetensors index (not formula-derived):
**FA layer 355.1 MiB, GDN layer 476.1 MiB** — GDN is 1.34x heavier, which is
why concentrating GDN concentrates weight.

Filed stage totals for the arm vs incumbent:

| stage | incumbent | arm | delta |
| --- | --- | --- | --- |
| 1 (3080) | 11750 MiB | **10066 MiB** | −1684 |
| 2 (3080) | 10542 MiB | **9590 MiB** | −952 |
| 0 (5090) | rises (GDN 24 → 27 layers) | — | absorbs both |

**Headroom check against capacities**, with the posts this rig's ledger
machinery already prices:

* **3080s (20480 MiB)**: arm stages at 10066 / 9590 MiB of weights leave
  ~10.4 / ~10.9 GiB before KV, graphs and the arming floor. Comfortable, and
  ~1.6 GiB better than today on the worse card.
* **Arming floors are per-rank and NOT uniform**: `basis_arming_floor_mib=(1728,
  1825, 2467)` (`managers/phase_flip_seam_reserve.py:218`). **Rank 2 carries the
  largest floor (2467 MiB)** while receiving the smallest relief (−952). That is
  the stage to watch, not the one with the most layers.
* **Corridor law** reserves ~1024 MiB free per card (a target, not a hard
  limit).
* **5090 (32768 MiB)**: absorbs both reliefs plus one more FA layer of KV. The
  filing's own argument is that this is "spending the headroom that actually
  exists on the big card"; it is the stage with the most slack and the least
  risk.

**Bootability verdict: nothing here says [35,15,14] is unbootable.** The arm
strictly relieves both constrained cards and loads the card with headroom.
I did NOT independently compute the 5090's post-arm total (the filing reports
the deltas, not the absolute), so the one number a window should print first is
stage 0's total against 32768 MiB with graphs and spec posts included.

---

## 3. Probe checklist for the A/B vs [31,17,16]

**Same-boot A-vs-A floor FIRST**, per canon — and for the #709 half, note the
cross-boot rule below.

| # | probe | why / gate |
| --- | --- | --- |
| 1 | **A-vs-A floor, same boot, per arm** | canon. For #709 the arms are unavoidably cross-boot (`--rank-tp-ratio` is a boot flag), so each arm measures its OWN floor and the cross-boot delta must clear the **LARGER** of the two; two arms carrying the same `boot_id` are refused as mislabelled |
| 2 | **prefill tok/s at s=1 and s=8** | the memory arm's headline: more pool on the small cards |
| 3 | **decode ms/round per rank, COMPUTE vs WAIT** | via `utils/collective_clock.py`. **This is the discriminator for #709, not the round time** — see the acceptance note below |
| 4 | **pool tokens per rank** | what the freed 1684/952 MiB actually bought |
| 5 | **flip health**: cutover count, no churn | the cut changes the seam |
| 6 | **#699 quiet** | the admission-wedge detector must not fire; recall it is blind to starvation while `running > 0` |
| 7 | **#706 rows 2-6 on THIS cut** | geometry-neutral keys must hit across the flip on the NEW geometry — that is the point of running them here rather than on the incumbent |

### The acceptance rule for the #709 half — do not use the round

#709 established that the briefed "does the decode round improve" rule **cannot
return the right answer**: predicted gain 0.780 ms against a ~30 ms round is
2.6 %, against a measured rig A-vs-A noise floor of **14.1 %** — the effect is
**5.4x smaller than the floor it would have to clear**. The layered rule that
replaced it:

* **PRIMARY — per-rank WAIT SPREAD (max−min).** Under an equal shard the 5090
  finishes early and waits; under a proportional shard that wait should
  collapse. On the family slice the same delta is 31.1 %, which IS resolvable.
* **SECONDARY — end-to-end round.** Reported, and explicitly NOT the
  discriminator. **A null here is EXPECTED** and is not evidence against the
  lever.
* **GATE — coherence.** The lever is lossless; a changed determined answer voids
  a speed win regardless of the numbers.

**One input the harness refuses to invent**: per-rank COMPUTE/WAIT does not come
over HTTP and is supplied via `--clock-json`. The per-rank line confirmed
in-tree is the PREFILL one (`metrics_reporter.py:91-97`); a decode parser was
deliberately not desk-written against an unconfirmed format. **Producing that
JSON is the window's first five-minute check, before it commits the reboot.**

---

## 4. Risks

1. **Rank 2 is the tight stage, not the big one.** It carries the largest arming
   floor (2467 MiB, `phase_flip_seam_reserve.py:218`) and gets the smallest
   relief (−952 MiB). If any stage runs out of corridor, expect it here.
2. **#690 flip cost and the x4 card.** The measured H2D asymmetry is real
   (rank1 x4 = 4.93 GB/s vs 7.08 / 8.88 on x8), and #690's constraint is fewest
   layers on the x4-linked card. The arm takes stage 1 from 17 to 15 layers,
   which moves in the RIGHT direction — but only if rank 1 is in fact the x4
   3080. **Resolve the identity first (§1.3); this risk is card-specific, not
   rank-specific.**
3. **Seam funding at the new cut.** The seam draw and arming floor were solved
   for the incumbent geometry; the cut moves both the layer boundary and one FA
   layer of KV. Re-read the funding rather than carrying the old solve across.
4. **`auto` would void the #709 half** (§1.2) — the single most likely way to
   spend the window and learn nothing.
5. **The `--pp-stage-ratio` scores-vs-counts ambiguity** (§1.1) — the same trap
   that already produced one wrong reading of `7,5,4` in the arm's own filing.
6. **#732 changed what this arm can claim.** Any report framing it as a
   family-split win is measuring against an objective that was priced NEGATIVE
   today. It is a memory arm.

---

## 5. What this package does not contain

No turnkey script. The two things a script would need — the resolved
scores-vs-counts form of the cut flag, and a confirmed decode-side `--clock-json`
producer — are exactly the two items above that the desk could not settle, and a
script that guessed either would be the kind of confident wrong artifact the
window cannot afford. Both are five-minute checks at the head of the window;
after them the flags in §1 are a complete recipe.
