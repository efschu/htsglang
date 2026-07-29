# Novelty evaluation: multi-group ("lane") runtime

Status: literature and system sweep, no code, no measurements of our own.
Date: 2026-07-29. Author: agent-lit-sweep. **Not committed.**

Method: 14 web searches plus 9 full-text fetches, ~110 distinct result links
surfaced, 11 systems read at the level of mechanism (not just abstract).
Per-claim search depth is stated in each section. Rule applied throughout:
"not found" means "not found by this sweep", never "does not exist".

## Object under test

One inference process holds several independent engines ("lanes") over one
byte-shared weight set; lanes may run *different* TP geometries (nested
layouts, fine shards being byte-subsets of coarse shards, proven at runtime
via `data_ptr` identity). Geometry and spec-config changes are plan flips on a
pre-captured CUDA-graph ladder, never a re-capture. Above that: load-shape-aware
pairing, priority classes with lend/reclaim primitives (~1-3 ms), and a
configurable per-card weight redundancy budget (0 = nested layouts only, up to
whole model = free role choice). Measured concurrency effect E = 1.28-1.54
card-equivalents vs ~1.0 for any time-multiplexed alternative. Target hardware
primarily multi-GPU without NVLink.

---

## B1 — Byte-shared *nested* TP geometries (down-set property)

Sources sighted: 14.

| Prior work | Year / venue | What it does | Difference to ours |
|---|---|---|---|
| **Flying Serving** ([arXiv 2602.22593](https://arxiv.org/abs/2602.22593)) | 2026, preprint (vLLM-based) | "Model parameters are loaded exactly once per engine and never physically moved thereafter. Mode switching is realized purely by changing which portions of an existing weight tensor are *activated* for computation." Creates per-rank `View`s over the full matrix. | The retained tensor is a **full DP replica per engine**; a TP shard is a subset *of the whole model*. Nesting is therefore the trivial two-level case (full ⊃ slice). No lattice among non-full layouts, no down-set formalization, no runtime identity proof. Requires the model to fit on one GPU ("our system assumes that the model weights and the necessary KV cache can be managed within the memory hierarchy of a single multi-GPU node"). TP groups must be **contiguous** rank segments. |
| **Nitsum** ([arXiv 2605.05467](https://arxiv.org/abs/2605.05467)) | 2026, preprint | "Our solution is to keep one full copy of model weights on each GPU and select the TP-specific shard at execution time using customized kernels." | Same structural shape as Flying Serving: full replica as the top of the lattice, shard selection by kernel. Explicitly prices the alternative ("storing all TP-specific copies of a 13B Llama model requires 45.5 GB … about 56% of an A100/H100's capacity") and accepts full replication. No sub-replica nesting. |
| **AnchorTP** ([arXiv 2511.11617](https://arxiv.org/abs/2511.11617)) | 2025, preprint | Elastic TP with **unequal-width** sharding; "parameters reside as contiguous blocks in a daemon-managed device-memory pool and are exposed via versioned handles that provide a unified address view"; "allowing reuse of surviving shards". | Closest thing to a real byte-level shard algebra without full replication — but it is **sequential** failure recovery: one geometry valid at a time, versioned handles swap, not co-valid layouts. No concurrency, no down-set claim. |
| **Shift Parallelism** ([arXiv 2507.11830](https://arxiv.org/abs/2507.11830), [2509.16495](https://arxiv.org/abs/2509.16495)) | 2025, Snowflake Arctic Inference, in production | TP ↔ Arctic SP per batch; weights stay in their original shards, only the computation pattern around them changes; KV layout invariant across modes. | Weight *placement* is invariant, i.e. it never changes the TP degree of the weights at all — it changes how sequences are split. Not a family of shard layouts over shared bytes. |
| **S-LoRA / Punica** ([arXiv 2311.03285](https://arxiv.org/abs/2311.03285), [2310.18547](https://arxiv.org/abs/2310.18547)) | 2023-24 | One base-model copy shared across thousands of adapters; unified paged pool for adapter weights + KV. | Sharing is across *adapters*, all at the **same** TP geometry. Orthogonal axis. |
| Megatron-FSDP / FSDP2 / GSPMD | 2021-2026 | Sliced views over contiguous flat buffers to avoid copies; nested FSDP units. | Training-side buffer views within *one* sharding plan; "nested FSDP" means module nesting, not co-valid shard geometries. |

**Strongest counter-candidate: Nitsum.** It is the most uncomfortable one
because it independently arrives at "one weight buffer, several TP shard
interpretations, selected at execution time" and states the memory arithmetic
explicitly. Our differentiator is not the idea of shard-views-over-one-buffer;
it is doing it *below* full replication.

**VERDICT: teilweise (Kern praeempted).** Multiple TP shard layouts over one
physical weight buffer: **exists** (two independent 2026 systems). Nesting
among non-full layouts (coarse shard ⊃ fine shard with no rank ever holding the
whole model), stated as a down-set property and verified at runtime by pointer
identity: **not found**.

---

## B2 — Runtime geometry switch by plan flip, no re-sharding / restart / re-capture

Sources sighted: 10.

| Prior work | Year | What it does | Difference to ours |
|---|---|---|---|
| **Flying Serving** | 2026 | Online DP↔TP switching without restarting engine workers; long-context switch latency **15 ms** vs 146-292 s for a static restart. Communicator Pool eagerly initialized; KV Cache Adaptor preserves state across layouts. | Switch is global and atomic across all engines ("These broadcasts atomically configure the active communicator and execution mode across all engines"). CUDA graphs are not discussed at all. |
| **Nitsum** | 2026 | "reload-free TP weight switching"; "we pre-launch alive but inactive execution processes for all candidate TP levels … we complete cuda-graph profiling and `torch.compile` optimization offline for every TP level before serving starts". | This *is* B2 including the no-recapture property, done by pre-launched-process-per-geometry rather than by a graph ladder inside one process. |
| **Shift Parallelism** | 2025, production | Per-batch TP↔SP switch, pre-captured graphs for both modes, shipped as a vLLM plugin. | Same property, narrower axis (TP↔SP, not TP degree). |
| **Moebius** ([arXiv 2606.26607](https://arxiv.org/abs/2606.26607)) | 2026 | "Seamless runtime parallelism switch" for MoE (TP↔EP). | Full text not extractable; at minimum an additional independent instance of the claim. |
| **SpotServe** ([arXiv 2311.15566](https://arxiv.org/abs/2311.15566)) | 2024, ASPLOS | Dynamic reparallelization on preemption, with parameter reuse and migration planning. | Explicitly **does** re-shard/migrate; the named cost we avoid. Predecessor, not competitor. |
| **LoongServe** ([SOSP'24](https://dl.acm.org/doi/10.1145/3694715.3695948)) | 2024, SOSP | Elastic sequence parallelism, degree changed in real time per request/phase. | Elasticity over sequence dimension; weights never re-laid-out, KV migration is the cost centre. |
| ElasticMoE, FlexPipe, DynaTrain, TensorHub | 2025-26 | Zero-copy HBM remapping, inflight pipeline refactoring, online parallelism switching for training. | Adjacent, mostly scaling/elasticity rather than serving-time geometry choice. |

**Strongest counter-candidate: Nitsum** (offline graph capture for *every*
candidate geometry, zero recapture at switch), with Shift Parallelism as the
production-deployed instance.

**VERDICT: existiert.** This is 2025/2026 state of the art in at least three
independent systems, one of them shipped. Nothing here is ours to claim.

---

## B3 — Multiple *independent* engines (own schedulers) concurrently in-process on the same cards, with lend/reclaim

Sources sighted: 13.

| Prior work | Year / venue | What it does | Difference to ours |
|---|---|---|---|
| **MuxServe** ([ICML 2024](https://proceedings.mlr.press/v235/duan24a.html), [arXiv 2404.02015](https://arxiv.org/abs/2404.02015)) | 2024, ICML | Colocates several LLMs; "the second partition stores a single replica of LLM weights that can be shared among prefill and decoding jobs"; splits inference into separate prefill and decoding **jobs**; "the parallel runtime dynamically assigns SMs to each job at runtime rather than statically allocating", at "the granularity of SM based on NVIDIA MPS"; ADBS batch scheduler. | The genuinely closest system: concurrent jobs, shared weight bytes, dynamic (not static) resource split. Differences: colocated jobs of one LLM run **one** parallel geometry (the LLM's assigned mesh); resource sharing is MPS/SM-level across processes, not in-process lanes; the co-scheduled units are prefill vs decode of the same engine, not fully independent engines; no geometry heterogeneity, no ms-scale lend/reclaim primitive between priority classes. |
| **Prism** ([arXiv 2505.04021](https://arxiv.org/abs/2505.04021)) | 2025 | Cross-model GPU memory coordination via a `kvcached` balloon driver; flexible space+time sharing; "multiple low-rate models can be colocated on a single GPU or a model-parallel GPU group". | Different models, no weight sharing between them; elasticity is on the **KV/memory** axis, weights are per-model. |
| **Aegaeon** ([SOSP 2025](https://dl.acm.org/doi/10.1145/3731569.3764815)) | 2025, SOSP | Token-granularity auto-scaling between model instances; up to seven models per GPU; 82% GPU reduction in Alibaba Cloud production. | Explicitly **time**-multiplexing (preemptive scale-down/scale-up at token boundaries) — the ~1.0 baseline we compare against, at industrial polish. |
| **Harli** ([arXiv 2511.11729](https://arxiv.org/abs/2511.11729)) | 2025 | Co-locates PEFT finetuning with decode instances; unified memory allocator for runtime memory reuse; QoS-guaranteed scheduler. | Two *different kinds of work* sharing a card, one of which is training; not multiple serving lanes at different geometries. |
| **Flying Serving** | 2026 | K DP engines, but: "The scheduler runs as a centralized event loop that coordinates K DP engines" and mode transitions are broadcast atomically. | One centralized scheduler, one global mode at a time. Two TP groups cannot serve different requests concurrently. |
| **Nitsum** | 2026 | Per-tier TP levels. | "at runtime, only the process for the active TP receives work, while the others remain hibernated with lightweight keep-alive signals" — geometry-heterogeneous processes exist but are **mutually exclusive**; GPUs are partitioned between tiers. |
| Orion (EuroSys'24), REEF (OSDI'22), Salus, Triton instance groups, MPS/MIG | 2020-2024 | Kernel-level / process-level GPU multiplexing, microsecond preemption, interference-aware submission. | Mechanism layer beneath us; model-agnostic, no weight sharing across geometries, no notion of a serving lane. |
| vLLM DP RFC ([aibrix #1858](https://github.com/vllm-project/aibrix/issues/1858)), sglang DWDP ([#22084](https://github.com/sgl-project/sglang/issues/22084)) | 2025-26 | DP ranks as separate core-engine processes, each owning TP workers; DWDP shards experts and replicates attention. | Upstream state: DP replicas are independent processes with **independent weight copies** at a **uniform** TP size. No shared-byte multi-geometry lane concept proposed. |

**Strongest counter-candidate: MuxServe.** If someone wants to argue we are not
novel, MuxServe is the paper they will cite: concurrent jobs, one weight
replica, dynamic SM split, prefill/decode separation. The honest rebuttal is
narrow — geometry heterogeneity between *concurrently active* lanes, and the
lend/reclaim primitive.

**VERDICT: teilweise — hier liegt der groesste eigene Anteil.** Concurrent
weight-sharing jobs on one GPU: **exists** (MuxServe). Concurrently *active*,
independently scheduled lanes running *different TP geometries* over one shared
weight set: **not found** — the two systems that have geometry heterogeneity
(Flying Serving, Nitsum) both explicitly serialize it.

---

## B4 — Load-shape-aware co-placement at card level

Sources sighted: 9.

| Prior work | Year | What it does | Difference to ours |
|---|---|---|---|
| **Bullet** ([arXiv 2504.19516](https://arxiv.org/abs/2504.19516)) | 2025 | Concurrent execution of the compute-intensive prefill phase and the memory-bound decode phase on the same GPU with real-time performance modelling. **1.26x average, up to 1.55x throughput.** | Same objective function, same card level. Note the reported band is essentially identical to our E = 1.28-1.54. |
| **MuxServe** | 2024 | "leverage the characteristics of prefill and decoding phases to separate and flexibly colocate them to multiplex computation resources". | Same principle, MPS/SM mechanism. |
| **CoLLM** ([arXiv 2604.16400](https://arxiv.org/abs/2604.16400)) | 2026 | Dynamic replica co-orchestration, interference-aware batch coordination. | Cluster + card level, no weight sharing. |
| **DistServe** ([arXiv 2401.09670](https://arxiv.org/abs/2401.09670)), **Splitwise**, Mooncake, TetriInfer, MemServe | 2024 | Separate prefill/decode **pools**, each pool with its own GPUs, its own parallelism strategy, and **its own weight copies**; scaled by replication. | Answer to the mandated question: PD-disaggregation does **not** share weight bytes — it separates roles across pools, each holding a full copy. That distinction is real and worth stating, but it does not make B4 novel. |
| Harli, iGniter, NanoFlow, Sarathi-Serve, DeepSpeed-FastGen (SplitFuse), POD-Attention | 2023-25 | Phase mixing/overlap *inside one* engine (chunked prefill, nano-batches, operation-level pipelining, fused attention for mixed batches). | Intra-engine overlap. Contrasts cleanly with our independent engines, but the *scheduling objective* (pair SM-saturating with bandwidth-bound) is identical and older than us. |

**Strongest counter-candidate: Bullet** — same objective, same level, same
measured magnitude, one year earlier.

**VERDICT: existiert.** Well-populated line. Our contribution is applying a
known objective inside a new substrate, not the objective.

---

## B5 — Discrete capture ladder as a policy action space

Sources sighted: 7.

| Prior work | Year | What it does | Difference to ours |
|---|---|---|---|
| **sglang adaptive speculative decoding** ([docs](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding), roadmap [#23705](https://github.com/sgl-project/sglang/issues/23705), [#21459](https://github.com/sgl-project/sglang/issues/21459)) | 2025-26, **upstream** | Predefined speculative-length tiers (default `[1, 3, 7]`), EMA of accepted length as the control signal, "Each tier has its own pre-captured CUDA graph, so switching between tiers is inexpensive and does not require graph recapture", "runtime switching is a reference swap, not an online graph recapture", "Tier switch happens after the current round completes". | This is B5 verbatim, in the engine we fork. Only the axis set differs: upstream ladders over draft length; ours additionally over algorithm, topk, and geometry. |
| **Nitsum** | 2026 | Offline cuda-graph profiling for every candidate TP level, selected at runtime. | Same pattern applied to the geometry axis. |
| AdaSpec ([arXiv 2503.05096](https://arxiv.org/abs/2503.05096)), Nightjar, AdaServe (EuroSys'26) | 2025-26 | SLO-aware adaptive speculation control policies. | Policy layer above the same mechanism. |

**Strongest counter-candidate: upstream sglang itself.** Consistent with our own
"Adaptive-Draft-Provenance" note that the adaptive controller is upstream, not
fork.

**VERDICT: existiert (upstream).** Claiming the mechanism would be a
provenance error. Only the widened axis set is ours.

---

## B6 — Redundancy budget as a continuous configuration parameter

Sources sighted: 8.

| Prior work | Year | What it does | Difference to ours |
|---|---|---|---|
| **Nitsum** | 2026 | Names the trade exactly ("storing all TP-specific copies of a 13B Llama model requires 45.5 GB … about 56% of an A100/H100's capacity") and resolves it by picking the **fixed** endpoint: one full copy per GPU, justified by "HBM-bandwidth-bound before … memory-capacity-bound". | Aware of the axis, no dial. This is the strongest counter precisely because it proves the trade-off is understood in the field and was deliberately not parameterized. |
| **Flying Serving** | 2026 | Fixed at the full-replica endpoint by construction (DP replica per engine), motivating it as maximizing KV memory relative to naive duplication. | Fixed endpoint, no dial. |
| **sglang DWDP** ([#22084](https://github.com/sgl-project/sglang/issues/22084)) | 2026 | Expert weights sharded within a node, attention weights fully replicated; TP=1 inside each DWDP group. | A single hand-chosen hybrid point on the sharded↔replicated axis, not a configurable budget. |
| MuxServe, S-LoRA, AlpaServe, "Automatic Cross-Replica Sharding" ([arXiv 2004.13336](https://arxiv.org/abs/2004.13336)) | 2020-24 | Weight sharing / replication choices made statically at placement time. | Placement-time discrete choice; no runtime capacity-vs-flexibility dial trading KV against switching freedom. |

**VERDICT: nicht gefunden.** No system in this sweep exposes weight redundancy
as a continuous per-card budget that buys switching freedom at the price of KV
capacity. Caveat: this is a narrow, implementation-flavoured claim; absence from
the literature partly reflects that papers report chosen operating points rather
than the knobs behind them.

---

## Overall verdict (5 sentences)

Three of our six claims are established state of the art and must not be
presented as ours: runtime geometry switching without re-sharding or graph
re-capture (B2) is done independently by Flying Serving, Nitsum and the
production-deployed Arctic Shift Parallelism; load-shape-aware prefill/decode
co-placement on one card (B4) is the explicit objective of Bullet and MuxServe,
with Bullet reporting 1.26-1.55x, a band indistinguishable from our E =
1.28-1.54; and the pre-captured ladder as a policy action space (B5) already
ships upstream in sglang's adaptive speculative decoding. Byte-sharing across TP
geometries (B1) also exists, but only in its degenerate form — both Flying
Serving and Nitsum keep a *full replica* per GPU and select a shard view from
it, so the nesting lattice collapses to "full ⊃ slice" and the model must fit on
one GPU. What survives as honestly new is the intersection of B3 and B6:
concurrently *active*, independently scheduled lanes at *different* TP
geometries over one shared, sub-replicated weight set — the two systems with
geometry heterogeneity both serialize it (Nitsum hibernates all but the active
TP process; Flying Serving switches all engines atomically to one global mode),
and the one system with true concurrent weight-sharing jobs (MuxServe) runs them
at a single geometry via MPS SM-partitioning across processes. The enabling
piece for that intersection is the redundancy budget as a continuous dial (B6),
which no surveyed system exposes — Nitsum demonstrates the field understands the
trade-off and chose to pay the full-replica price rather than parameterize it.
Net: the *mechanisms* are largely prior art, the *combination* is not, and the
value proposition should be framed as "concurrency where the field currently
serializes", with the heterogeneous / no-NVLink target being a differentiator of
setting rather than of mechanism.

## Consequence for external presentation (README / FEATURES)

Do not write "first", "novel", or "unique" against B1, B2, B4 or B5 — instead
name Flying Serving, Nitsum, Arctic Shift Parallelism, Bullet/MuxServe and
upstream sglang adaptive speculation as related work, and state our delta
against them explicitly, since a reviewer will find these within one search and
an unqualified claim would discredit the parts that hold. The defensible
sentence is roughly: "several concurrently active, independently scheduled lanes
over one shared weight set at *differing* TP geometries, with a configurable
per-card weight-redundancy budget that permits operation below full replication"
— always with the qualifier "not found in the surveyed literature", never
"does not exist". Report E = 1.28-1.54 as *comparable to* published
spatial-sharing results rather than as evidence of a superior mechanism, and
carry the provenance note that the adaptive-ladder controller is upstream, not
fork.
