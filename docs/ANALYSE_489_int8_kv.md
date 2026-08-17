# ANALYSE 489 — int8-PTH KV cache dtype as a performance candidate

Status: **survey → evaluation, desk only. No build, no GPU, no load.**
Written per Feature-Analysen-Pflicht as a persistent file.

**Verdict up front: do not pursue on this rig.** It is unbuilt rather than
unflagged (a), and the published depth-inversion lands squarely on the side our
workload lives on (b). The hetero variant that would rescue it — per-rank KV
dtype — is blocked by two independent contracts (d).

---

## (a) Support status in our fork: NOT SUPPORTED

`server_args.py:1008`:

```
choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"]
```

**`int8` is not an accepted `kv_cache_dtype`.** The help text (`:1003-1006`)
lists the same set. A tree-wide search for an int8 KV path returns **zero
hits**: no `torch.int8` in `mem_cache/memory_pool.py`, none in the FlashInfer or
Triton attention backends, and no "int8 kv cache" anywhere in the fork.

(The `int8` at `server_args.py:3954` is `deepep_dispatcher_output_dtype`, a MoE
dispatcher setting, unrelated to KV. It is named here because it is the obvious
false positive for anyone grepping.)

So this is **a feature to build, not a flag to set**: pool storage dtype,
quant/dequant on the read and write paths, and backend kernel support would all
be new. That is the baseline against which every benefit below must be priced.

---

## (b) The depth-inversion risk, priced for THIS rig

The published shape (Whamp): int8 KV gives **+76-81% decode throughput at short
context** but **−72% at 58K**, because the dtype choice selects a *backend
tier* — fp8 keeps FlashInfer while int8 falls back to Triton-only.

**Which backend do we actually run?** From the live boot:

```
attention_backend='flashinfer'
```

FlashInfer on the text path. (`triton_attn` appears in the log only as the
**multimodal** backend — "Multimodal attention backend not set. Use
triton_attn" — which is a different selection and must not be misread as the
text path. The GDN layers take the separate "hybrid linear attention backend".)

So adopting int8 would move the text attention path **off FlashInfer onto
Triton**, which is exactly the trade that produced the −72%.

**Which side of the inversion is our workload on?**

| our regime | value | side |
|---|---|---|
| context length | 327,680 | far beyond the 58K inversion point |
| pool | ~436k tokens | long-context by construction |
| `max_running_requests` | 4 | not the many-short-request regime |
| `chunked_prefill_size` | 512 | long prompts arrive as 640 chunks |

The **gain** regime is short-context decode with high concurrency. The **loss**
regime is deep context. We are configured for deep context and low concurrency:
we sit on the losing side, and not marginally — our operating depth is ~5.6x
past the measured inversion point.

**Conclusion for (b):** the inversion is not a risk to be managed here, it is
the expected outcome. int8 would trade a benefit we cannot collect for a
penalty we would pay on every long request.

---

## (c) sm86 microbench PLAN — ticket spec, not run

Worth specifying because the published numbers are not from this hardware, and
because one rig-specific fact could overturn (b): if FlashInfer's fp8 path is
*already* not being taken on some of our shapes, the backend-tier argument
weakens.

**Note the rig is heterogeneous in SM version**: the two RTX 3080s are **sm_86**
(GA102); the RTX 5090 is **sm_120** (GB202). An "sm86 microbench" therefore
covers **two of three cards only**, and its result must not be generalised to
the 5090. That asymmetry is itself part of the hetero twist in (d).

Ticket spec:

* **Arms:** fp8_e4m3 (today's shipped dtype) vs a simulated int8 store, at
  identical shapes. Since int8 KV does not exist (a), the honest arm is a
  *kernel-level* microbench of the attention kernels at int8 vs fp8 input, not
  an end-to-end server arm — an end-to-end arm cannot be built without the
  feature.
* **Sweep:** context depth 1K / 8K / 32K / 58K / 128K / 327K, batch 1 and 4.
  58K is included specifically to reproduce the published inversion point on our
  silicon rather than assume it transfers.
* **Per card:** run on one 3080 (sm_86) and on the 5090 (sm_120) separately and
  report separately. Never average across SM versions.
* **Measure:** decode tok/s and attention-kernel ms per step, plus which backend
  each arm actually selected (log the selection, do not infer it — that
  inference is what the multimodal-vs-text confusion above would have caused).
* **Noise:** A-vs-A first, per the standing rule. The rig's measured A-vs-A
  spread is **14.1%**; any arm difference below that is not a finding.
* **Cost:** small — kernel-level, no model load. Rides any window; needs a
  gpu-arb claim, no serving restart.
* **Kill condition:** if the 58K point reproduces the published inversion on
  sm_86, stop. (b) is then confirmed on our own silicon and the feature is
  closed.

---

## (d) The hetero twist: PER-RANK KV dtype

The idea: give each rank the dtype its silicon prefers — int8 where it is fast,
fp8 where FlashInfer wants it — since our three cards differ (sm_86 ×2,
sm_120 ×1). Attractive in principle on a heterogeneous rig.

**Is it expressible today?**

*The pool side: yes, nearly.* Per-rank cell width is **already** a per-rank
quantity — the sizing line logs `cell 14336 / 10240 / 8192`, i.e.
`attn_layers_i × bytes_per_token_per_layer`. Nothing in the pool arithmetic
requires the *bytes* factor to be uniform across ranks; it is uniform only
because one `kv_cache_dtype` is passed to every pool.

*The seam: no, and it is pinned at runtime.* `phase_flip_runtime.py:32-35`
states the contract:

> "The receiver derives the expected byte count from **its own** pool's
> per-layer row width — a sender whose row format diverges is a loud
> size/checksum error, which is the runtime pin of the 'PP and TP rows are
> byte-compatible' claim."

A per-rank dtype makes row widths diverge between sender and receiver, so every
seam exchange becomes a size/checksum failure. This is a **hard block**, and a
well-behaved one: it fails loudly rather than corrupting.

*The canonical page contract: no, independently.* The #706/#704b agreed
contract requires the stored form to be **canonical and layout-neutral** — the
same key must name the same bytes regardless of which rank wrote them. A
per-rank dtype makes the bytes rank-dependent, which is precisely the
two-geometry problem that contract exists to prevent. Solving it would require
either canonicalising on store (dtype-converting on every spill, paying the
conversion on the hot path) or per-rank key namespaces (re-introducing #703).

**What it would take:** a dtype-aware row negotiation in the seam protocol
(sender declares its width; receiver converts or refuses), plus canonical-form
conversion at the store boundary. Both are real protocol changes, not
parameters. **Recommendation: not worth opening while (a) and (b) stand** — it
is a large change to enable a dtype that is unbuilt and expected to lose on our
depth.

---

## (e) Quality gate — required before any enable

int8 KV is **lossy**. Under the Quality-Last ordering, lossy changes come after
lossless ones and require measurement *before* enabling, never after.

Minimum gate, stated so it cannot be softened later:

1. **Determinism first** (A-vs-A, byte-identical across runs, CPU-sampled
   inputs) — the same discipline as the #704b LSE gate. Establishes that the
   quantised path is stable before asking whether it is accurate.
2. **Accuracy against the fp8 baseline** on a fixed prompt set, with tolerances
   **fixed before the run**: greedy-decode token-sequence match rate plus a
   stated max divergence position. A tolerance chosen after seeing the numbers
   is not a gate.
3. **Depth-stratified**, because the failure mode is depth-dependent: report at
   1K / 32K / 128K / 327K separately. A quality result averaged over depth would
   hide exactly the regime we operate in.

No enable without all three, and no default-on regardless of result — a lossy
storage dtype is an opt-in.

---

## Recommendation

**Close as evaluated-and-declined for this rig**, unless the (c) microbench
contradicts the published inversion on sm_86.

The chain is: it does not exist in our fork (a); building it would move the
text path off FlashInfer onto Triton (b); our workload sits ~5.6x past the depth
where that trade turns negative (b); the hetero rescue is blocked by the seam
row-width contract and the canonical-page contract (d); and any enable would
still need a three-part lossy-quality gate first (e).

The one cheap thing worth doing is (c) — it is kernel-level, rides any window,
and its kill condition closes the ticket on our own silicon rather than on
someone else's.
