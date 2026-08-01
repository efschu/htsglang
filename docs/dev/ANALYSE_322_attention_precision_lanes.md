# ANALYSE #322 — attention COMPUTE precision as a selectable lane

Survey, design and task placement. **No implementation.** This file is the
feature-analysis file for #322: every decision below is persisted here rather
than in a chat, so a later agent can act on it without re-deriving it.

Scope: the precision of the attention COMPUTE (the QK^T and PV GEMMs inside
the attention kernel), not the KV cache dtype. `--kv-cache-dtype fp8_e4m3`
already exists and is orthogonal: it changes what is STORED, this changes what
is MULTIPLIED. A deployment can run either, both, or neither.

---

## 1. Upstream state (PR-check first)

Searched `sgl-project/sglang` via the API on 2026-08-01.

**SageAttention is already in the tree — but only in the DIFFUSION runtime.**

| what | where | state |
| --- | --- | --- |
| SageAttention v2 backend | `python/sglang/multimodal_gen/runtime/layers/attention/backends/sage_attn.py` | in tree |
| SageAttention 3 (Blackwell) | `.../backends/sage_attn3.py` | in tree, PR #15382 merged |
| AITER Sage MXFP4 (AMD) | `.../backends/aiter_sage.py`, PR #21856 | in tree |
| Sage on MUSA | PR #24752 | merged |
| "use sage as default backend for SM120" | PR #15668 | **open** |
| Selection + graceful fallback | `multimodal_gen/runtime/platforms/cuda.py:133-170` | resolver per backend |

The resolver pattern is worth copying: each backend tries its import and falls
back with a log line naming the exact install command
(`pip install sageattention==2.2.0 --no-build-isolation` for v2; the upstream
Blackwell instructions for v3). A missing kernel degrades to FA / Torch SDPA
rather than failing the boot.

**For the LLM serving path (`python/sglang/srt/`) there is nothing.** `grep`
for `sageattn|SageAttention` across `srt/` returns no hits. The one attempt,
PR **#17679 "SageAttention"** (5 files, +1394 lines), was **closed unmerged
two days after opening**, with a community question — *"why close this pr? is
there any problem with sageattention?"* — left unanswered.

### What that means for us

1. The kernels are a solved dependency; the *serving-path integration* is not.
2. An unexplained abandoned PR is a warning, not a verdict. Before any code,
   someone should ask #17679's author and the maintainer why it closed. That
   is a five-minute question that could save the whole task, and it is the
   first item in the sequencing below.
3. Our delta is therefore not "port SageAttention" — the port exists twice
   over. Our delta is **selectability**: attention precision as a per-lane,
   opt-in choice with a quality gate, which is the "alles auswählbar" line and
   is not what either upstream integration does (diffusion picks one backend
   per platform, globally).

---

## 2. Kernel availability per architecture

Verifiable from the tree and the upstream install instructions:

| arch | card here | what exists | confidence |
| --- | --- | --- | --- |
| sm120 Blackwell | RTX 5090 | SageAttention 3 (FP4 attention), separate install | in-tree backend, merged PR |
| sm86 Ampere | RTX 3080 20GB | SageAttention v2 (`pip sageattention==2.2.0`), INT8 QK | in-tree backend |
| sm75 Turing | 2080 Ti (other rig) | **unverified** — not claimed by either in-tree backend | must be probed |
| gfx90a/gfx942 | — | AITER Sage MXFP4 (AMD path) | merged PR |
| gfx900 Vega | other rig | none; stock Triton already rejects gfx900 | closed |

The sm75 row is deliberately left open rather than guessed. SageAttention's
own support matrix is the authority and it must be read at implementation
time, not asserted here — the fork has been bitten before by assuming a
kernel family covers an arch it does not (the CDNA-line and stock-Triton
findings). **Rig-is-lower-bound**: sm75 having no kernel would not sink the
feature; it would make sm75 a rank that stays full-precision, which the
per-rank design below already expresses.

---

## 3. Integration point: backend registry, planner lane, or both

Two seams exist and they answer different questions.

**The attention backend registry** (`server_args.py`:
`ATTENTION_BACKEND_CHOICES`, `add_attention_backend_choices()`) is where a
new kernel becomes selectable at all. A SageAttention backend for the serving
path is a new entry here plus a backend class next to the flashinfer/triton
ones. This is the mechanism.

**The planner lane** (à la #353's INT8-GEMM lane, `ANALYSE_319_int8_lane.md`)
is where a per-(rank, family) CHOICE gets made from measured rates. #324
already made the GEMM scores per-(rank, family); attention precision is the
same shape of question one axis over.

**Recommendation: the registry first, the planner lane second, and only if
the registry step measures a win.** The planner lane is worthless without a
kernel to select, and the registry step is where the whole risk lives. Do not
build the lane speculatively.

---

## 4. Is mixed precision across ranks in one TP group sound?

This is the interesting cut on a heterogeneous rig — INT8 attention on the
sm86 ranks, full precision (or FP4) on the 5090 — and it splits by which axis
the group is sharded on.

### 4a. Head-sharded (pure TP): sound, but must be measured

Under plain TP each rank owns a disjoint set of attention heads, computes
them independently, and the results meet only at `o_proj`'s all-reduce. Two
ranks running different attention precision produce different rounding on
DIFFERENT heads — structurally the same situation as the heterogeneous GEMM
rates the fork already runs, and no worse in kind. Nothing reads another
rank's attention output before the reduction.

Caveat that makes it measurable rather than free: the per-head error is no
longer uniform across the head axis, so a downstream layer sees a
head-dependent noise floor. Whether that matters is an empirical question and
the gate in §6 is how it gets answered.

### 4b. Token-sharded DCP: NOT sound as a per-rank choice — group-uniform

Under token-sharded DCP every rank computes attention over the tokens it OWNS
and the partial results are merged through an LSE reduction into ONE output
for the SAME head. Mixing precisions there does not distribute error across
independent outputs; it combines a quantized partial softmax with a
full-precision one inside a single result, and the LSE merge is exactly the
step that is sensitive to the relative scale of its inputs. Under UNEVEN DCP
the per-rank token counts differ too, so the quantized contribution is
weighted differently from the full-precision one, by a ratio the user chose
for capacity reasons and never intended as a numerics knob.

**Guard: under token-sharded DCP the attention precision must be
group-uniform.** A per-rank vector combined with DCP is a hard reject at
argument time, not a warning — the same validate-early shape as the other
lane guards.

### 4c. Speculative decoding: accept length is the instrument

Two distinct risks, both named:

* **Draft/verify precision mismatch.** If the draft runs one attention
  precision and the target verify another, the verify rejects tokens the
  draft would have produced under its own numerics. That does not corrupt
  output — the accept rule still only accepts what the target agrees with —
  but it silently costs throughput, which is the whole point of speculation.
  It shows up as a DROP IN ACCEPT LENGTH and nowhere else.
* **Verify-vs-decode asymmetry.** The verify forward is a 2-row batch and the
  decode a single row; that reassociation already flips near-tie positions
  (#139, and the whole #274 line). Adding a quantized attention path widens
  the perturbation, so more positions become flippable.

**Instrument, per the standing rule (#326):** accept length is read from
`meta_info.spec_accept_length` and `spec_verify_ct`, **never** from the
Prometheus `spec_ema_accept_len`, which is not the accept length. Any arm of
the evaluation that quotes an EMA number is void.

**Guard:** draft and target must run the same attention precision unless a
measurement says otherwise. Default the draft to the target's choice.

### 4d. CUDA graphs

Per-rank backend selection means ranks capture different kernels. Graphs are
per-process so this is legal, but the fork has been bitten by asymmetric
capture before (#133's symmetric-worker-capture fix). **Guard:** the capture
shape set must stay identical across ranks even when the kernel differs, and
a lane that cannot be captured is a lane that loses more to eager replay than
quantization buys — see the stop rule.

---

## 5. Where the quality axis sits (Quality-Last)

Attention-compute quantization is **lossy**. Under the standing
priority rule it ranks **behind every byte-identical speed win** — behind the
remaining graph/scheduling/transport work, in the same class as fp8-KV,
spill-quant and eviction, and it must not be started while byte-identical
wins are still on the table.

Consequences that are design constraints, not preferences:

* **Opt-in, never default.** No auto-selection, no "planner picked it for
  you" without the flag. The open upstream PR #15668 proposes exactly the
  opposite for diffusion on sm120 (sage as the DEFAULT); whatever upstream
  decides there, the serving path here does not inherit it.
* **Per-lane, not global.** A deployment must be able to run quantized
  attention on a throughput lane and full precision on a quality lane in the
  same server — that is the "alles auswählbar" requirement and it is the only
  thing that makes a lossy lane acceptable at all.
* **The default path stays byte-identical.** Flag unset must produce the same
  kernels and the same numbers as today, pinned by a test, as with every
  other lane in this fork.

---

## 6. The quality gate

The instrument already exists: **`scripts/dual_group/chain_quality_gate.py`
(#328)**. It was built for the lane-spec chain, and the question here is the
same shape — "did turning this on make the output worse" — so it is reused,
not re-invented.

    band   = max(|ref_a - ref_b|, |cand_a - cand_b|)   # same-boot A-vs-A
    margin = |cand_a - ref_a|
    GREEN <=> margin <= band

Two properties matter here specifically:

* **Graded content, not text identity** (#360/#365). Quantized attention will
  flip near-tie positions; that is expected and is not evidence of damage. A
  gate keyed on token identity would go red on the first flip and tell us
  nothing.
* **The band is measured on the same boot, never pre-registered** (#274). The
  perturbation an INT8 QK introduces is not a number anyone can guess in
  advance — guessing it would decide the verdict by the guess.

**Second instrument, mandatory alongside:** accept length under speculation
(§4c), from `meta_info`. A configuration that passes the content gate but
loses accept length has not paid for itself, and the two numbers must be
reported together.

**Third, for the per-rank arm specifically:** rank-vs-rank self-determinism.
The fork's hetero-determinism work (#50) established that mixed-card groups
can diverge in ways that only show under specific arms; a per-rank precision
vector is a new way to produce exactly that, so the boot-to-boot and
rank-to-rank identity checks belong in the same window.

---

## 7. Sequenced task cuts

Each cut is independently abandonable — that is the point of the ordering.

| # | cut | effort | gate to proceed |
| --- | --- | --- | --- |
| 0 | **Ask why #17679 closed.** One GitHub comment to author + maintainer. | minutes | A "the kernel is wrong for serving" answer stops the line here. |
| 1 | **Kernel probe, no integration.** Micro-benchmark SageAttention v2 (sm86) and v3 (sm120) standalone against the FA path at this model's head_dim/GQA shapes and at both prefill and decode shapes. | S, one card window | A win under 10 % on PREFILL shapes stops the line (see §8). |
| 2 | **Registry backend, group-uniform, opt-in flag.** One `--attention-backend sage*` entry, resolver with graceful fallback (copy the diffusion pattern), default path byte-identical + test. | M | Content gate GREEN and accept length within band on a single-card boot. |
| 3 | **Spec interaction.** Draft/target precision policy, accept-length measurement, the verify-vs-decode widening. | M | Accept length must not regress beyond the measured band. |
| 4 | **Per-(rank, family) vector**, TP-only, with the DCP hard reject from §4b. | M-L | Only after 2 and 3 are green; needs the rank-vs-rank determinism arm. |
| 5 | **Planner lane** (per-rank selection from measured rates, #353 shape). | M | Only if 4 shows the per-rank asymmetry is worth choosing automatically. |

Cuts 0 and 1 are cheap and answer the only question that matters early. Cuts
4-5 are the fork's actual delta and are also the ones most likely never to be
reached.

---

## 8. Stop rule — "not worth it if X"

Abandon the line, and record it in the discard register, if ANY of these
holds:

1. **The win is prefill-only and the workload is decode-bound.** At bs=1
   decode, attention is memory-bound on KV streaming; quantizing the attention
   COMPUTE buys close to nothing there. If cut 1 shows no decode-shape win and
   the target workload is decode-dominated, this feature is a prefill feature
   and should be judged as one — against the other prefill levers already
   measured, most of which are byte-identical and therefore rank ahead of it.
2. **It cannot be CUDA-graph captured.** The stack is graph-first; a lane that
   forces eager replay loses more than quantization gains.
3. **The kernel does not cover the model's head_dim / GQA layout** without a
   pad that eats the win.
4. **Accept length regresses beyond the band** under speculation and no
   draft-side policy recovers it. Speculation is worth more than the
   attention-compute win on this stack's measured numbers.
5. **The content gate goes RED** at the deployment's own context length.
   Quality-last does not mean quality-optional.

The honest prior: this is a **prefill-shaped, opt-in, lossy** feature whose
main value on a heterogeneous rig is the per-rank asymmetry in cut 4, and
whose main risk is that cut 1 shows the win is smaller than the graph and
spec interactions cost. It should not start while byte-identical wins remain.

---

## 9. Recommendation

Do cut 0 and cut 1 only, then re-decide with numbers. Do not schedule cuts
2-5 now: they sit behind the byte-identical work by the standing priority
rule, and the abandoned upstream PR is an unexplained signal that a
five-minute question can resolve before any card time is spent.
