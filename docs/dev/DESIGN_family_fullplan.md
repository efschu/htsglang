# DESIGN — family full plan: all GDN on the 5090, all FA on the 3080s

Desk-only. No boot, no GPU. Base `4a16043d1a`. Every number below is either
COMPUTED here from the checkpoint config, or CITED to a commit/file:line;
inferences are marked **INFERRED**.

**Checkpoint facts, read from
`models-cache/Qwen3.6-27B-INT8-W8A8-yarn4.0/config.json`:** `hidden_size 5120`,
`num_hidden_layers 64`, `vocab_size 248320`, `full_attention_interval 4`,
`tie_word_embeddings false`, and `layer_types` giving FA at exactly
`[3, 7, 11, …, 63]` — 16 FA, 48 GDN.

---

## 1. The map, and the crossing schedule

### 1.1 Layer → card

| card | layers | count |
| --- | --- | --- |
| **5090** | every `linear_attention` layer — all indices NOT ≡ 3 (mod 4) | **48 GDN** |
| **3080-A** | 8 of `[3,7,…,63]` | 8 FA |
| **3080-B** | the other 8 | 8 FA |

### 1.2 The FA split: **8/8, and the crossing count does not depend on it**

Because FA layers are never adjacent (three GDN layers separate every pair),
**every FA layer is entered from the 5090 and exited back to it**. So which
3080 owns which FA layer changes *which* card a crossing targets, never *how
many* crossings there are. The split is therefore free in schedule terms and
should be decided on capacity — which is equal — hence **8/8**.

The alternative (9 on the x8-linked 3080, 7 on the x4) is available but weak:
#690's measured H2D is 4.93 GB/s on the x4 card against 8.88 on the x8, so a
10 KiB payload costs ~2.0 µs there vs ~1.1 µs — a ~0.9 µs difference against a
measured ~7.3 µs per crossing (§3), i.e. **~12 % of one crossing**. Shifting one
FA layer buys about one crossing-equivalent per token and unbalances the KV
pools, which are the point of the plan. **Recommendation: 8/8; revisit only if
a measurement shows the per-crossing link penalty is much larger than derived.**

### 1.3 The crossing schedule — "31 crossings" VERIFIED

Execution order per token is `A A A X | A A A X | …` (A = 5090, X = an FA card),
16 times.

* **5090 → FA**: once per FA layer = **16**
* **FA → 5090**: once after each FA layer *except the last*, because layer 63
  is the final layer and its output goes to the head, not back into a GDN
  block = **15**

**Total 31 INTER-LAYER crossings per token per forward.** The figure in the #732
material is correct as stated; I re-derived it from the layer map rather than
adopting it, and the "29 extra" phrasing elsewhere is the same schedule counted
against a 2-crossing PP baseline.

**But 31 is not the movement count, and the bullet above is careless.** It says
layer 63's output "goes to the head, not back into a GDN block" — and §2.1 puts
`lm_head` on the 5090, so that output crosses back too. The full-forward
MOVEMENT count is therefore **32**. Both numbers are kept because both are used:
31 is what the crossing schedule emits and what a transport matches sends to
(`DESIGN_pp_layer_set.md` §5); 32 is what a wire trace of a whole forward will
show. Every cost figure below derived from 31 understates the transport term by
one movement, i.e. **~3 %** — below the resolution of the verdicts they support,
so they are left as printed rather than re-derived.

---

## 2. Per-card numbers

Per-layer weights are the MEASURED values from the safetensors index cited by
`9c25330131`: **FA 355.1 MiB, GDN 476.1 MiB**.

### 2.1 The 5090 (32607 MiB NVML) — and #727 is load-bearing

Embed and lm_head are **untied** (`tie_word_embeddings false`), so both must be
placed, and at `248320 × 5120` each they are large:

| | per tensor | both |
| --- | --- | --- |
| bf16 | 2425 MiB | 4850 MiB |
| **int8 (#727)** | **1212 MiB** | **2425 MiB** |

48 GDN layers = **22 852.8 MiB**. With the head tensors on the same card:

Against the card's LIVE NVML total of **32 607 MiB**, not the nominal 32 768 —
the width-canon lesson applies to totals as well as widths, and the 161 MiB
difference moves the mamba-slot ceiling by a whole slot at every graph-pool
size (`DESIGN_pp_layer_set.md` §4):

| vocab dtype | weights total | free | after arming floor 1728 + corridor 1024 |
| --- | --- | --- | --- |
| **int8 (#727)** | 25 278 MiB | 7329 MiB | **4577 MiB** |
| bf16 | 27 703 MiB | 4904 MiB | **2152 MiB** |

(The nominal-total figures this spec first printed were 7490/4738 and
5065/2313.)

**#727 roughly doubles the working headroom on the 5090** (4577 vs 2152 MiB),
and that headroom is what must cover GDN state + graph pool. Both #727
artifacts exist (`3f8996a1a6` requant + read method, `5745a545d6` lm_head), so
this is a placement decision, not a build. **The full plan should be specified
as requiring #727's int8 vocab**; on bf16 it is not obviously fundable.

**INFERRED / OPEN — the number this spec cannot supply.** GDN state for all 48
layers lands on the 5090 (state lives with its layer). I have no measured
per-layer GDN state size, so I cannot say whether 4577 MiB covers state + graph
pool at the target concurrency. **That is the first number a build must
produce**, and it is the plan's real capacity question — not the weights, which
fit comfortably.

### 2.2 The 3080s (20480 MiB each) — this is the payoff

8 FA layers = **2840.8 MiB** of weights per card.

| card | weights | free | after its floor + corridor |
| --- | --- | --- | --- |
| 3080 w/ floor 1825 | 2841 | 17 639 | **14 790 MiB** |
| 3080 w/ floor 2467 | 2841 | 17 639 | **14 148 MiB** |

Compare the contiguous memory arm now booting: its 3080 stages carry
**10 066 / 9590 MiB** of weights, leaving **7565 / 6989 MiB** for KV after the
same floor and corridor.

**RETRACTED: this spec claimed "roughly 5× more room for KV on the small
cards". It does not reproduce.** Against the contiguous arm's own 3080s the
ratio is 14 790 / 7565 = **1.96×** and 14 790 / 6989 = **2.12×**, i.e. **~2×**;
against the 5090's 4577 MiB it is **~3.2×** (the adopted `NOTE_735_arithmetic_check.md` prints 3.1× there because it used the nominal-total 4738 MiB; same ratio, corrected denominator). The 5× has a derivation path — it
is 14 790 / 2841 = **5.21×**, this card's KV room divided by its OWN weights —
but that is a different statement from "more room than the contiguous arm", and
substituting one for the other is the error. Real arithmetic, wrong pair.

**~2× more KV room on the small cards** is the corrected figure, and it is
still the entire point — it is also the only
layout that achieves the KV-away-from-the-5090 half, because (as the boot
package established) *any* contiguous 35-layer block carries 8–9 of the 16 FA
layers and therefore RAISES the 5090's KV share to ~50 %.

### 2.3 Draft / NEXTN

**INFERRED**: the draft head is an attention-family consumer, so it belongs
with FA on a 3080, and its KV must live with the FA KV it verifies against.
Splitting draft KV from target KV across cards would add crossings inside the
verify loop. Not settled here; flagged as a placement decision the build owes.

---

## 3. Interface expectations against Slot-3's seam

| requirement | value | derivation |
| --- | --- | --- |
| payload per crossing | **10 240 B = 10 KiB** at bs=1 | `hidden_size 5120` × 2 B (bf16 activation). Exact, not approximate |
| crossings per token | **31** inter-layer (16 out, 15 back); **32** movements incl. terminal -> head | §1.3 |
| direction split | 16 × 5090→3080, 15 × 3080→5090 | §1.3 |
| cost budget | ~**7.3 µs per crossing**, **~0.227 ms/token** for 31 | from #732's amendment (29 crossings ≈ 0.212 ms); the per-crossing figure is my division, the 0.212 is cited |
| metadata | small per-crossing header | **INFERRED** — not sized here |

**CAPTURE-SAFETY IS THE HARD REQUIREMENT.** These crossings are on the DECODE
path, 31 per token. If the send/recv seam is not capturable inside a CUDA
graph, decode either runs eager — losing far more than 0.227 ms/token — or the
graph must be broken 31 times per token. **The seam must be capture-safe or the
plan does not pay.** This is the single interface property most likely to
decide the outcome and should be stated to Slot-3 as a requirement, not a
preference.

**The 0.81× x8-pair regime** applies to the 15 return crossings only where both
endpoints sit on x8 links; on the x4-linked 3080 it does not. With an 8/8 split
half the return traffic is in each regime. **INFERRED** — I did not re-derive
the 0.81 figure.

---

## 4. Risks, and the two that change the plan's shape

### 4.1 Does the phase flip still apply, or does the full plan retire it?

**Answer: it does NOT retire the flip — it replaces the flip's PREFILL phase.**
The reasoning, stated because the brief says this changes everything downstream:

* the flip exists because PP suits prefill and TP suits decode;
* the full plan is a third layout — neither a contiguous PP range nor TP — and
  it carries **31 crossings per token on the decode path**;
* TP decode carries none. So the full plan is decode-*unfavourable* by
  construction and prefill-favourable (crossings amortise over many tokens).

Therefore the full plan is best understood as **a candidate PP-phase layout
inside the existing flip world**, not as a static replacement for it. Adopting
it as the whole serving layout would pay 31 crossings on every decode token to
buy KV room the decode phase does not need in that form. **INFERRED** — this
follows from the crossing asymmetry, not from a measurement, and a build should
test the decode arm before accepting it.

### 4.2 The REAL second blocker: PP cannot express a non-contiguous map

**This is a build, and it is not small.** At code:

* `get_pp_indices` (`distributed/utils.py:1506`) returns
  `Tuple[start_layer, end_layer]` and computes
  `start = sum(partitions[:pp_rank])` (`:1526-1527`). A stage is an
  **interval**, by construction.
* `SGLANG_PP_LAYER_PARTITION` takes per-stage **counts** that must sum to
  `num_hidden_layers` (`:1515-1525`) — there is no syntax for "stage 0 owns
  {0,1,2,4,5,6,…}".
* The `(modules, start_layer, end_layer)` contract is threaded through **59
  model files** under `models/` that reference `start_layer`.

**The good news, which sizes it down:** `make_layers`
(`utils/common.py:1969-2011`) already builds a **full-length** `ModuleList` with
`PPMissingLayer` placeholders before and after the owned range, and
`PPMissingLayer` is a `torch.nn.Identity` pass-through
(`layers/utils/common.py:109-125`). **The module structure already tolerates
non-owned layers at arbitrary positions** — placeholders interleaved in the
middle are structurally the same object as placeholders at the ends.

So the build is not "rewrite the model tree". It is:

1. an index contract that can carry a SET (or mask) instead of `(start, end)`
   — the change at `get_pp_indices` and `make_layers` is small;
2. the 59 `start_layer` consumers, most of which use it as a base offset and
   would need auditing rather than rewriting — **INFERRED**, I did not read all
   59;
3. **the crossing schedule**, which is the genuinely new part: today PP sends
   once per stage boundary, and this map needs a send at every ownership change
   — 31 per token instead of 2.

**Size: M–L, and it is a SECOND blocker independent of send/recv.** Slot-3's
transport unblocks the *wire*; this unblocks the *addressing*. Both are
required and they are not the same work. Naming it now because the critical
path currently shows only one blocker.

### 4.3 Other risks

* **Capture-safety** (§3) — the decode path's 31 crossings must be graphable.
* **GDN state sizing on the 5090** (§2.1) — the unsupplied number.
* **Corridor pressure on the 5090**, not the 3080s: the plan inverts today's
  tightness. The 3080s become roomy (14+ GiB free) and the 5090 becomes the
  constrained card at 4.7 GiB after floors — with bf16 vocab, 2.3 GiB.
* **#690's fewest-layers-on-x4 constraint becomes moot for weights** (each 3080
  holds only 8 FA layers ≈ 2.8 GiB) but applies to the per-crossing latency
  instead (§1.2).

---

## 5. What this spec does not settle

The GDN state budget on the 5090 (§2.1), the draft/NEXTN placement (§2.3), the
metadata size per crossing (§3), and whether the decode arm survives 31
crossings (§4.1). Each is named where it sits rather than estimated, because
every one of them can move the verdict and none can be settled from a desk.
