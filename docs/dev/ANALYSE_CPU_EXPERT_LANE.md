# ANALYSE — computing cold MoE experts on the LOCAL CPU: feasibility and verdict

Desk analysis, 2026-08-04, branch `fix/spill-composability`. **No build.**
Question, from the user order: instead of streaming a cold expert's weights
host->device, let the CPU compute that expert on the pinned host shard, so only
activations cross the link.

Scope: the **local** CPU (this rig, 98 GB RAM, no swap). `DESIGN_453` prices the
REMOTE CPU lane on rig 2; that is a different vehicle with a network in the
middle and is not re-derived here.

**Verdict up front: PROBE FIRST, and the probe is narrow** — one measured
number decides it, and §2 already produced enough to say which number. Details
in §6.

---

## 0. Where this sits

`ANALYSE_393` asked the question (CPU compute instead of RAM->GPU streaming, to
get around the ~2.8 GB/token transfer wall of the K3 class). The decision then
was: build #394/#439 first (link-proportional shards + compute placement — both
now BUILT), then decide on CPU. Price state carried forward: ~2.00x on the
DSV4F/K3 class, but only 0-28 % residual margin on the local Qwen class after
#394. That asymmetry survives this analysis and sharpens: see §5.

---

## 1. The mechanical cut — where CPU GEMM would dock

The existing path (`layers/moe/expert_offload.py`): only a hot subset of a
FusedMoE layer's local routed experts is resident on GPU; the rest live in a
pinned host pool. Before each `apply()`, misses from `topk_ids` are async
H2D-copied into LRU-evicted slots and `topk_ids` are remapped to slot indices,
so the unmodified grouped-GEMM runs over the small resident buffer.

The CPU lane replaces the **fetch** for a chosen subset of experts:

| | today | CPU lane |
|---|---|---|
| what crosses the link | the expert's WEIGHTS (H2D) | that expert's token ACTIVATIONS (D2H) and its OUTPUT rows (H2D) |
| bytes per event | `3 * h * i * dtype` — fixed, token-count independent | `2 * tokens * h * dtype` — scales with tokens, not with the expert |
| who computes | GPU grouped-GEMM over the resident slot | CPU GEMM on the pinned host shard |

Concrete, for the two local vehicles (shapes read from the shipped configs):

| model | hidden | moe_inter | experts | top-k | one expert, bf16 | activations for 1 token, bf16 |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 | **6.29 MB** | 2 x 4 KB = **8 KB** |
| Qwen3.5-122B-A10B | 3072 | 1024 | 256 | 8 | **18.87 MB** | 2 x 6 KB = **12 KB** |

So in the decode regime the traffic ratio is roughly **800:1 in the CPU lane's
favour**. That is the whole attraction, and it is why the question keeps coming
back. The transfer wall genuinely disappears; what replaces it is a compute
question, which §2 measures.

**Docking point.** #462's breakable route is the right seam and it already
states the discipline: "the graph addresses SLOTS; the eager phase decides
which expert occupies which slot, materialises its bytes there, and publishes
the mapping — all of it BEFORE the replay that reads them." A CPU-computed
expert is the same shape with a different producer: the eager phase writes
graph-addressed buffers, the replay reads them. See §4 for why this is not free.

---

## 2. The price anchor — MEASURED, not cited

Hermetic CPU GEMM on the real expert shapes (torch, CPU only, no card,
16 threads on this box). One "expert" = gate(h->i) + up(h->i) + SiLU-gate +
down(i->h), i.e. `6 * tokens * h * i` FLOPs.

**Qwen3.5-35B-A3B (h=2048, i=512):**

| dtype | 1 token | 8 tokens | 64 tokens |
|---|---|---|---|
| fp32 | 0.268 ms (23.5 GFLOP/s) | 0.338 ms (149 GFLOP/s) | 0.932 ms (432 GFLOP/s) |
| fp16 | 6.107 ms (1.0 GFLOP/s) | 48.3 ms | 383 ms |
| bf16 | 6.343 ms (1.0 GFLOP/s) | 49.9 ms | 401 ms |

**Qwen3.5-122B-A10B (h=3072, i=1024):**

| dtype | 1 token | 8 tokens | 64 tokens |
|---|---|---|---|
| fp32 | 1.005 ms (18.8 GFLOP/s) | 1.011 ms (149 GFLOP/s) | 3.185 ms (379 GFLOP/s) |
| fp16 | 18.2 ms | 144 ms | 1139 ms |
| bf16 | 17.2 ms | 133 ms | 1013 ms |

### Finding 1, and it is the most important thing in this document

**Half precision on this CPU is a 400x cliff.** ~1 GFLOP/s for bf16/fp16
against 23-432 GFLOP/s for fp32 — torch falls back to a scalar path with no
fast kernel. A CPU expert lane that computes in the model's native bf16 is not
slow, it is **impossible**: one expert-token would cost 6 ms.

So the lane's numeric format is not an implementation detail, it is the
feasibility question. Any build must dequantise into fp32 (or use a
llama.cpp-style integer kernel, see §3). This single number would have sunk a
naive implementation after the fact; it costs 20 seconds to learn beforehand.

### Finding 2 — the crossover, and it runs the right way

Against an H2D fetch at an assumed ~10 GB/s effective (this rig is all PHB, no
P2P — see the interconnect record; treat this as the one unmeasured term):

| model | fetch cost | CPU fp32 breaks even at |
|---|---|---|
| Qwen3.5-35B-A3B | 6.29 MB -> ~0.63 ms | **~40 tokens per expert** |
| Qwen3.5-122B-A10B | 18.87 MB -> ~1.89 ms | **>64 tokens per expert** |

Per-expert token counts are tiny in DECODE (batch x top_k / num_experts) and
large in PREFILL (a 2048-token chunk against 256 experts gives ~64). So:

* **decode: CPU compute wins**, by ~2.3x on the 35B shapes and more on the
  122B — which is the same order as `ANALYSE_393`'s ~2.00x for the K3 class,
  now reproduced on local shapes;
* **prefill: the fetch wins**, and #254's expert-major wave order already
  exists to make it win by more.

That the crossover falls between the two regimes is the useful result: this is
a **decode-only lane**, and any build should say so in its flag help rather
than presenting it as a general offload mode.

---

## 3. Formats, and the RAM wall that decides the local vehicle

fp32 is the only fast CPU format measured above — but storing experts as fp32
on the host doubles the pool:

| model | all routed experts, bf16 | as fp32 |
|---|---|---|
| Qwen3.5-35B-A3B | 256 x 40 x 6.29 MB = **64 GB** | **129 GB** |
| Qwen3.5-122B-A10B | 256 x 48 x 18.87 MB = **232 GB** | 464 GB |

Against 98 GB of RAM with no swap, **neither fp32 pool fits**, and the 122B does
not fit even in bf16. So a local CPU lane cannot hold fp32 masters. It must
either:

1. **dequantise per event** from the quantised master (GPTQ Int4 for both local
   vehicles) into a small fp32 tile, paying dequant cost per fetch — cost
   UNMEASURED, and it is the probe in §6; or
2. **use an integer kernel** (llama.cpp-style K-quant GEMM) that computes
   directly on the quantised bytes. This is the format family the fork already
   handles on the GGUF side, so the shapes and block layouts are familiar — but
   there is no such CPU kernel in this tree, and importing one is a
   substantially larger build than (1).

Option 1 is the honest first step, and its cost is exactly what makes or breaks
the 2.3x margin from §2: the margin is 0.63 ms vs 0.268 ms, i.e. **0.36 ms of
headroom per expert-token**. A dequant of 3.1M Int4 params into fp32 has to fit
inside that. Nothing in this tree tells us whether it does.

---

## 4. Honest costs

**Numerics.** CPU fp32 (or dequantised-Int4-to-fp32) against the GPU's
bf16/Int4 grouped GEMM will not be bit-identical. The fork already has a
precedent and a decision for exactly this class: GPTQ/AWQ-marlin offload is
intrinsically ~1e-2 and that was ACCEPTED, while FP8 offload is byte-identical.
A CPU lane belongs in the accepted ~1e-2 class and must be declared as such at
the flag, not discovered in a quality gate. It is therefore a **lossy** feature
under the quality-last rule and should be built after byte-identical wins.

**Scheduler and sync.** The CPU work is synchronous with the layer unless it is
overlapped. The only overlap that pays is CPU and GPU working on **disjoint
expert sets of the SAME layer**: GPU runs the hot experts while the CPU runs
the cold ones, so the layer costs `max(GPU_hot, CPU_cold)` instead of the sum.
Any other arrangement serialises, and 40 layers x 0.268 ms = 10.7 ms/token of
pure serial CPU time would eat the whole win. This is the single design
constraint a build must start from, not add later.

**Graph compatibility — the hard requirement.** "Breaks capturable" is NOT an
acceptable end state. The route to full coverage exists and is #462's, which
this tree already built:

* the graph addresses SLOTS, and the eager phase publishes what is in them
  before replay;
* so the CPU lane becomes an **eager segment between captured segments**, with
  the expert output written into a **graph-addressed result buffer** at a
  **fixed shape** (fixed rows per expert slot, zero-padded), and the captured
  grouped-GEMM masked over those rows;
* the segmentation machinery for exactly this already exists in
  `breakable_offload.py`, and `offload_capture_gate.resolve_offload_graph_mode`
  is the one place the mode is decided.

Start state: a build would begin with the MoE layer's eager boundary already
present in breakable mode, so the starting point is not "no graphs" but
"graphs with an existing break". That is materially better than starting from
scratch, and it is why #462 should be checked as the dock before any other
design is drawn. What is genuinely new is the fixed-shape result buffer and the
mask — both hermetically testable.

---

## 5. Pricing per vehicle, kept separate

**DSV4F-GGUF (K3 class) — transfer-wall dominated.** This is where
`ANALYSE_393`'s ~2.00x came from and where the 800:1 traffic ratio of §1 bites
hardest. But it is a GGUF vehicle, so option 2 in §3 (integer kernel) is the
natural format and option 1 (dequant to fp32) is the awkward one. The upside is
largest and the build is largest.

**Qwen3.5-35B-A3B — post-#394, only 0-28 % residual margin.** The link-
proportional shards and compute placement already collected most of what was
available locally. A decode-only lane with a 0.36 ms/expert-token headroom
that still has to pay dequant is being asked to fit inside a margin that may
already be closed. This vehicle is the WEAK case, and it happens to be the one
the rig runs.

That split is the honest summary: **the vehicle with the margin is the one with
the expensive build, and the vehicle with the cheap build has little margin
left.**

---

## 6. Verdict and probe

**PROBE FIRST. Do not build.** Two of the three terms are now measured
(CPU GEMM rate; the format cliff). The third decides everything and is
unmeasured:

> **How long does dequantising one expert's GPTQ-Int4 weights into an fp32 tile
> take on this CPU, in the shapes of §1?**

If that is well under the 0.36 ms/expert-token headroom, the decode lane is
real and §4's overlap design is worth drawing. If it is comparable to or above
it, the local lane is dead on the Qwen class and the question reduces to the
DSV4F/K3 vehicle with an integer kernel — a much larger decision that should
not be taken on this evidence.

**The probe is hermetic, needs no GPU, and costs minutes**: load one expert's
quantised tensors from the shipped `Qwen3.5-35B-A3B-GPTQ-Int4` checkpoint,
dequantise to fp32 with the tree's existing GPTQ dequant path, and time it
against the §2 GEMM. Report ms per expert alongside the 0.268 ms GEMM and the
~0.63 ms fetch, so all three terms sit in one table.

**Anti-fooling:** measure the dequant on a COLD tile (not one already in L3
from a previous iteration), and report the thread count — §2's numbers are at
16 threads, and a lane competing with the serving process for cores will not
get 16. A rate measured on an idle box is an upper bound, and the honest
comparison is against a box that is also serving.

---

## 7. The crossing question with P6 (asked, and answered)

P6 (`ANALYSE_SPILL_SPEC_LANE.md`) and this lane both want the same scarce
thing: **eager time inside a captured decode step**. P6's spec-in-tick wants
the spill tick to carry a drafter; this lane wants an eager CPU segment at
every MoE layer. Per "everything combinable", the answer is not an exclusion
but a named condition:

* they touch **different layers** (attention/KV vs MoE) and different
  resources (PCIe for the KV tail vs CPU cores for expert GEMM), so there is no
  correctness conflict;
* they **compete for the same latency budget**, and both are justified only if
  the round is not already dominated by the other. So the composition rule is:
  whichever probe runs second must measure with the first one ARMED, or its
  margin is measured against a floor that will not exist in production;
* and both funnel through the same graph discipline (eager phase before replay,
  graph-addressed buffers), so the segmentation work is shared rather than
  duplicated — which is an argument for doing #462's dock properly once.

No exclusion is warranted. The condition is that the second probe must not be
run against a bare baseline.
