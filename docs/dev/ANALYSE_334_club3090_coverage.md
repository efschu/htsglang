# ANALYSE #334 — club-3090 model coverage

Survey and task placement for "alles auswählbar" across the club-3090 model
set. **No implementation.** Discharges the feature-analysis-file duty for
#334.

---

## 1. Inventory first — what is actually on the box

Two trees, and they are different things.

**`club-3090/models-cache/` — real checkpoints, what can be served today.**
Qwen3.5 (2B, 4B, 9B, 35B-A3B, 122B-A10B), Qwen3.6 (27B dense, 35B-A3B),
Gemma-4 (12B, 26B-A4B, 31B, E4B), Llama-3.1-8B, plus drafters (EAGLE3,
speculator-eagle3, dflash per model). Formats: FP8, INT8-W8A8, AWQ,
GPTQ-Int4, NVFP4, AutoRound-int4, and GGUF (Q3_K_M / Q4 / Q6, several with
MTP preserved).

**`club-3090/models/` — deployment RECIPES, not weights.** Compose files and
READMEs per model × engine (`beellama`, `llama-cpp`, `ik-llama`, `vllm`,
`vllm-omni`), 28-80 KB per directory.

### The finding that reframes the task

Of the five families in the brief, **three have no artifact on this box at
all**:

| family | on disk | note |
| --- | --- | --- |
| Qwen3-Omni-30B-A3B | **recipe only** (`models/qwen3-omni-30b-a3b/vllm-omni/`) | runs today on 2×3090 under **vLLM-Omni**, not sglang |
| Ornith / ik-llama quants | **recipe only** (`models/qwen3.6-{27b,35b-a3b}/ik-llama/`) | compose files for IQ-family quants |
| DiffusionGemma | **absent** | no checkpoint, no recipe |
| Nemotron-Puzzle | **absent** | no checkpoint, no recipe |
| agents-a1 | **absent** | no checkpoint, no recipe, no upstream signal |

So this survey is not "five models to integrate". It is: **two families the
user already runs on another engine and might want on ours, and three that
would first have to be downloaded and justified.** Naming that is the point
of inventorying first.

---

## 2. Coverage matrix

Axes: **Load** (does sglang have the architecture), **unevenTP/DCP** (the
hetero enabler — cuttable per the standing order), **Spec/MTP**, **GGUF**,
**Spill/Hibernate**, **Graphs**.

| model | Load | unevenTP/DCP | Spec/MTP | GGUF | Spill/Hib | Graphs |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5/3.6 dense + A3B/A10B | yes (served) | yes | yes | yes | yes | yes |
| Gemma-4 family | yes (`gemma4_*.py`) | yes | yes (`gemma4_mtp.py`) | yes | yes | yes |
| **Qwen3-Omni Thinker** | **yes** (`qwen3_omni_moe.py`) | inherits A3B | inherits A3B | soft gap | yes | yes |
| **Qwen3-Omni Talker+Code2Wav** | **no** | **HARD** | **HARD** | n/a | n/a | n/a |
| **Nemotron-Puzzle (NAS)** | **yes** (`nemotron_nas.py`) | **HARD** (see 3b) | soft | soft | likely yes | yes |
| Nemotron-H (hybrid mamba) | yes (`nemotron_h*.py`) | GDN-shaped, likely yes | `nemotron_h_mtp.py` | soft | yes | yes |
| **Text-diffusion LM (LLaDA-2)** | **yes** (`llada2.py` + `dllm_config`) | **needs review** (3c) | **HARD** (3c) | soft | soft | partial |
| DiffusionGemma specifically | **no arch** | — | — | — | — | — |
| **ik-llama IQ quants** | **no** (3d) | n/a | n/a | **HARD** | n/a | n/a |
| agents-a1 | unknown | — | — | — | — | — |

Legend: **HARD** = architectural, with the reason named in §3. *soft* =
merely unbuilt.

---

## 3. Per family: upstream state, blockers, smallest cut

### 3a. Qwen3-Omni — the Thinker is already ours; the audio-out is not

**Upstream:** PR **#10911 "model: qwen3-omni (thinker-only)"** is **merged**
— and the title is the whole story. sglang serves the *Thinker*, the 30B-A3B
MoE text LLM. It does not serve the Talker (text → audio codec tokens) or
Code2Wav (codec → waveform).

**Which known gap family:** the Thinker is *not a new model family at all* —
it is the A3B MoE geometry the fork already serves with uneven TP, DCP, spec
and spill. Everything on the text axis is inherited for free.

**Hard blocker for audio-out:** the Talker/Code2Wav stages are a **3-stage
pipeline of separate engines**, stage-parallel by construction (the user's
own recipe puts thinker on GPU0, talker+code2wav on GPU1). They are not
layers a TP group shards; they are *tenants*. So audio-out is not a
model-loading task, it is a **lane/tenant composition** task, and it belongs
to the #333 multimodal-class line, not here.

**Smallest cut:** serve the Thinker on our stack and compare against the
user's vLLM-Omni text path — a text-only A/B that costs nothing new
(**effort S**, it is a boot of an already-supported architecture). Audio-out
is **L** and gated behind #333.

### 3b. Nemotron-Puzzle — supported, and it lands squarely in the #100 gap

**Upstream/local:** `python/sglang/srt/models/nemotron_nas.py` exists (ported
from vLLM). Nemotron-NAS *is* the Puzzle architecture: `config.block_configs
[layer_idx]`, `block_config.attention.no_op`, `block_config.ffn.no_op`,
`ffn_mult` — **per-layer heterogeneous blocks, where some layers have no
attention and some have no FFN.** No upstream PR search hits for
"nemotron puzzle", i.e. nobody is actively working it.

**Hard blocker, and it is exactly the known one:** the fork's uneven-TP unit
partitioning assumes **every layer has the same families**. That is the
#100 `tp_units` family-table lesson in its purest form: a layer with
`attention.no_op` contributes zero attention units, and a family vector
computed over a uniform layer count is wrong for this model in a way that
does not announce itself — it produces a plausible split for a geometry that
does not exist. The same table drives #324's per-(rank, family) GEMM scores
and the #348b cost library, so the error propagates into the planner.

**Smallest cut:** make the family table **per-layer** rather than per-model
(**effort M-L**), which is a fork-wide improvement, not a Nemotron feature —
it is the same change any future heterogeneous-layer model needs. Falsifier
first: a unit test with a synthetic `block_configs` containing `no_op` layers
that shows today's partitioner producing a wrong vector.

**But: no checkpoint on the box.** Do the falsifier test now if the
family-table work is wanted for its own sake; do not download a model to
justify it.

### 3c. Text-diffusion LM — the machinery exists, the semantics fight our axes

**Upstream:** active. `#20615 [SGLang-Diffusion LLM] Add inference support`
(open), `#17316 [DLLM] Optimize batching algorithm efficiency` (open).
Locally: `llada2.py` plus a real serving path — `scheduler.dllm_config`,
`get_new_batch_dllm`. **DiffusionGemma specifically is not present**, but the
class is.

**Hard blocker — spec/MTP is meaningless here.** A diffusion LM does not
extend a causal prefix one token at a time; it iteratively denoises a masked
block. There is no "next token" for a drafter to guess and no accept rule to
apply, so the fork's entire spec line (MTP, EAGLE, DFLASH, the k-ladder,
#328's chain gate) simply does not apply. That is architectural, not unbuilt.

**Needs review, not yet classifiable — DCP.** Token-sharded DCP assumes a
causal ownership rule over a growing prefix. A denoising block is rewritten
in place across iterations, so "who owns token L" is stable but "what is at
token L" is not. Whether the owner rule survives that is a real question and
the honest answer today is *unknown*; it is a design question before it is a
task.

**Smallest cut:** boot `llada2` on our stack with spec explicitly OFF and
measure whether anything in the DCP/spill path even engages (**effort M**).
That is the cheapest way to turn the DCP question from speculation into a
finding.

### 3d. ik-llama IQ quant families — a GGUF type-code gap

**Upstream:** **zero** PRs for `ik_llama` or `IQ4_KS`. These quant types
(IQ4_KS, IQ2_KL and relatives) are **ik_llama.cpp-specific**, not mainline
llama.cpp, so a mainline GGUF reader does not know their type codes at all.

**Which known gap family:** #129's GGUF registry generalization. The gap is
"the registry does not know these type codes and their dequant kernels", not
anything about the model — the same shape #129 already solved once for the
types it does support.

**Hard blocker:** each IQ type needs its own dequant kernel. That is real
kernel work per type, and the fork's own record says these are the expensive
kind (the #109 MMQ out-of-bounds and the uneven-TP GGUF alignment bugs all
lived here).

**Smallest cut:** read the type codes actually used by the user's ik-llama
recipes and report *which* types would be needed (**effort S, desk**) before
anyone writes a kernel. It may be one type, in which case this is M; it may
be five, in which case it is L and probably not worth it.

### 3e. agents-a1 — RESOLVED: operator transcription error, strand closed

Investigated 2026-08-01 at the user's request: the name appears nowhere in
the club-3090 recipes, the model cache, ANALYSE_347, or upstream. Its first
occurrence is the #334 task subject itself — an operator transcription
error when the task list was composed, not a real model family. Strand
closed; nothing to integrate.

Nothing on disk, nothing upstream that matches. **Recommend closing this
strand** unless the user can name the artifact; a survey cannot cover a name.

---

## 4. Recommended order

Per Feature-Doku-Reihenfolge (hetero enablers first, then normal-rig
utility):

1. **Nemotron-Puzzle's per-layer family table** — a HETERO ENABLER that
   happens to be found via Nemotron. It fixes a latent wrongness in the
   uneven-TP partitioner and the planner cost library for *any*
   heterogeneous-layer model. Do the falsifier test even without the
   checkpoint. **M-L.**
2. **Qwen3-Omni Thinker on our stack** — near-free (already-supported A3B
   geometry) and it answers a real user question: is our stack better than
   the vLLM-Omni text path the user runs today. **S.**
3. **ik-llama type-code inventory** — desk-only, decides whether the IQ
   family is an M or an L before any kernel is written. **S.**

Then, only if wanted: the dllm DCP question (design), and Omni audio-out
(behind #333).

---

## 5. Stop rules

* **No downloads to justify work.** Three of the five families have no
  artifact. Do not fetch a checkpoint to make a task real; fetch one when a
  user need is stated.
* **Stop on ik-llama if more than ~two IQ types are needed.** Per-type
  dequant kernels are the fork's historically expensive and bug-dense corner.
* **Stop on Omni audio-out until #333 lands a tenant/stage composition.**
  Modelling a 3-engine pipeline as a TP group is the wrong shape and would
  have to be undone.
* **Do not chase DiffusionGemma as a model.** If the diffusion-LM class is
  wanted, `llada2` is on hand and upstream is actively working the path; a
  second architecture adds nothing until the first one's DCP/spec semantics
  are settled.
* **Every exclusion above is a priority candidate**, per
  alles-mit-allem-kombinierbar. The two genuinely HARD ones —
  spec-on-diffusion-LM and TP-sharding the Talker — are hard for stated
  architectural reasons and are the two that should stay excluded.

---

## 6. Summary

The survey's real content is that the coverage is **much better than the
brief assumed and the gaps are not where the model names suggest**:
Qwen3-Omni's Thinker, Nemotron-Puzzle and a text-diffusion LM are all already
in the model registry. What is missing is not loaders but (a) a per-layer
family table, which is a hetero enabler worth doing on its own merits, (b)
GGUF type codes for a non-mainline quant family, and (c) a tenant composition
for audio-out that belongs to a different line of work.

---

# Appendix (#372) — ik_llama type-code inventory

The §3d cut, executed: **which GGUF quant type codes do the user's ik-llama
recipes actually use**, and what would each cost in our loader. Desk-only, no
kernels, no loader code.

Source: every file under `club-3090/models/*/ik-llama/` (compose YAML plus
their READMEs) for `qwen3.6-27b` and `qwen3.6-35b-a3b`.

## The split that decides the verdict

A raw grep over those recipes returns 121 quant-type mentions across 10
distinct spellings, which reads like "ten types to support". It is not. The
mentions fall on **two different axes**, and only one of them is a weight
dequant question:

| spelling | mentions | axis | what it actually is |
| --- | --- | --- | --- |
| `q4_0` | 57 | **KV cache** | `-ctk`/`-ctv` default in every recipe ("K and V quant type (default: q4_0)") |
| `q8_0` | 23 | **KV cache** | the same flag's alternative ("q8_0 boots...") |
| `q5_0` | 4 | **KV cache** | ditto |
| `IQ4_KS` | 17 | **weights** | ubergarm MTP GGUF, Qwen3.6-27B — "imatrix IQK quant" |
| `IQ4_XS` | 6 | **weights** | byteshape 4.19 bpw GGUF, Qwen3.6-35B-A3B |
| `F16` / `f16` / `bf16` | 12 | **weights** | the `mmproj` vision projector |
| `Q8_K_XL` | 1 | **weights** | mentioned in prose, no recipe pins it |
| `Q4_K_M` | 1 | **weights** | mentioned in prose, no recipe pins it |

**84 of the 121 mentions are KV-cache quantization**, an axis we address with
`--kv-cache-dtype` and which needs no weight dequant path at all. Counting
them as weight types is the mistake this inventory exists to prevent: it would
have turned a one-type verdict into a five-type one, i.e. the exact M-vs-L
question §3d was created to answer.

## Weight types, per type

| type | ik-specific? | our loader today | on disk | verdict |
| --- | --- | --- | --- | --- |
| **IQ4_KS** | **yes** — absent from mainline llama.cpp's ggml sources and from our type list | **NO** | **no artifact** | the only real gap — **M** |
| IQ4_XS | no, mainline | **YES** (already in the supported IQ set) | **no artifact** | nothing to do |
| Q4_K_M | no, mainline | yes | no artifact | nothing to do |
| Q8_K_XL | no, mainline (unsloth naming) | yes | no artifact | nothing to do |
| F16 / BF16 (mmproj) | no | yes | no artifact | nothing to do |
| q4_0 / q8_0 / q5_0 | n/a — **KV cache, not weights** | `--kv-cache-dtype` | n/a | wrong axis |

Our loader's IQ coverage today, for the record: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_NL, IQ4_XS. The mainline IQ
family is complete; `IQ4_KS` is the one outside it.

## The honest "no artifact" rows

**Not one of these checkpoints is on this box.** Both GGUFs the recipes name —
`Qwen3.6-27B-MTP-IQ4_KS.gguf` and `Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf` — are
download URLs in compose files, not files. So:

* no artifact on disk uses **IQ4_KS**;
* no artifact on disk uses **IQ4_XS** (though our loader would already read
  one);
* no artifact on disk uses **Q4_K_M** or **Q8_K_XL** in an ik-llama context.

The GGUFs that ARE on disk (`models-cache/`) are mainline Q3_K_M / Q4 / Q6
builds, all already supported.

## Effort and yield for IQ4_KS

**Effort: M.** One type, one dequant path in the #129 registry pattern. It is
M rather than S because of the #109 corner: the MMQ out-of-bounds bug lived in
exactly this code, and a new block layout has to be exercised against the
uneven-TP shard boundaries where that class of bug appears (a type whose block
size does not divide a per-rank shard width is where the alignment failures
have always shown up).

**Yield: one checkpoint the user has a recipe for and does not currently hold.**
The gain is access to the ubergarm MTP-IQ4_KS build of a model we already
serve in five other formats (FP8, INT8-W8A8, AWQ, GPTQ-Int4, mainline GGUF).
So the yield is not "a new model" — it is "one more quantization point for an
existing model, from an ecosystem whose tooling we do not otherwise consume".

Judged as a pair rather than against a threshold: the effort is a bounded,
single-type kernel task, and the yield is narrow but real if the user wants
that specific build. **Recommendation: do it when the user asks for that
checkpoint, not before** — and download the GGUF first, because the block
layout should be read off the real file rather than from a description.

## What would change this verdict

* A second ik-specific type appearing in a recipe (IQ2_KL, IQ5_KS and
  relatives exist in that ecosystem). Two types is still M; the cost is mostly
  per-type kernel work, so it scales linearly and the "when the user asks"
  rule holds until the list grows.
* The user actually downloading an IQ4_KS build — at that point the yield
  stops being hypothetical and the block layout becomes readable.

---

# CLOSING DETERMINATION (2026-08-17)

Re-checked at code and in `git log --all` on `train/0817-control`
(`4a16043d1a`). **Verdict: #334 closes as DETERMINED-WITH-FILED-RESIDUE. No
named family lacks a scoping.**

## Per-family status

| family | integration path | status | evidence |
| --- | --- | --- | --- |
| **Qwen3-Omni Thinker** | already ours — it is the A3B MoE geometry, not a new family | **DELIVERED** | `models/qwen3_omni_moe.py`; §2 marks Load/unevenTP/DCP/spec/spill/graphs all inherited |
| **Qwen3-Omni Talker + Code2Wav** (audio-out) | a 3-engine tenant composition, not a TP group | **SCOPED, DEFERRED BY STOP RULE** | §5: "Stop on Omni audio-out until #333 lands a tenant/stage composition" — modelling it as a TP group "is the wrong shape and would have to be undone" |
| **Nemotron-Puzzle** (hetero layers) | per-layer family table in the uneven-TP partitioner + planner cost library | **DELIVERED** | `d55deb6cd3` (`Merge feat/per-layer-family-table-371`), an ancestor of this branch; the table lives in `uneven_perf.py` |
| **Ornith / ik-llama IQ quants** | GGUF type-code support | **DETERMINED; one gap named** | `addf27b339` (`Merge docs/393-ik-llama-survey`) — **EXISTS-OTHER-LINEAGE**, not an ancestor of `train/0817-control`. Appendix verdict: **IQ4_KS is the only real gap, effort M**; IQ4_XS/Q4_K_M/Q8_K_XL already load, and `q4_0`/`q8_0`/`q5_0` were the wrong axis entirely (KV-cache dtype, not weights) |
| **DiffusionGemma** | none, deliberately | **SCOPED-OUT WITH A REASON** | §5: "Do not chase DiffusionGemma as a model" — the text-diffusion class is already served by `llada2.py` + `dllm_config`, and a second architecture "adds nothing until the first one's DCP/spec semantics are settled" |
| **agents-a1** | none — no such artifact | **RESOLVED, STRAND CLOSED** | §3e: investigated 2026-08-01; the name's first occurrence is the #334 task subject itself, an operator transcription error |

## The two families the closing brief flagged as possible remainders

Both already carry an analysis, so neither is the honest remainder:

* **agents-a1** is not unscoped — it is *resolved*. A survey cannot cover a
  name that names nothing, and §3e says so with the date it was checked.
* **DiffusionGemma** is not unscoped either — it carries an explicit STOP RULE
  with a reason that still holds: `llada2` is on hand and upstream is actively
  working that path. Declining a model for a stated reason is a scoping, not a
  gap.

## One attribution I could not confirm

The closing brief attributes the Thinker half to a task **#373**. That number
has **no trace anywhere in this tree** — not in `docs/dev/`, not in the
sources, not in `git log --all`. What *is* verifiable is that the Thinker
itself is DELIVERED at code, so the work is not missing; only the ticket
attribution is unconfirmable from here. Recorded rather than silently adopted:
it may live in the operator's register outside this repo, and an audit that
repeats a number it cannot check is how a wrong pointer propagates (the same
class this week's stale-gate sweep found in `layers/dcp/owner.py:127`).

## Residue, with its gating class

1. **IQ4_KS loader** — effort **M**, and gated by §5's own stop rule: *no
   downloads to justify work*. There is no IQ4_KS artifact on this box. The
   gap is real and priced; it should stay unbuilt until a user need names it.
2. **Omni audio-out** — gated behind **#333** (tenant/stage composition). Not
   a model-coverage task at all once you accept that framing.
3. **dllm DCP/spec semantics** — a design question, not an integration one;
   §3c holds it as the thing that must settle before any second
   diffusion-LM architecture is worth considering.

## Rollup

#334 asked which club-3090 model families the fork cannot serve. The survey's
own finding was that the coverage was better than the brief assumed and the
gaps were not where the model names suggested, and this re-check confirms it
held up: of the five families, one was never a model (agents-a1, a
transcription error), one is deliberately declined with a live reason
(DiffusionGemma, because `llada2` already covers the class), one is delivered
as a hetero enabler that outlived its originating model (#371's per-layer
family table, now in `uneven_perf.py` and useful to any heterogeneous-layer
model), one is delivered for free because it turned out to be an
already-served geometry (the Omni Thinker), and one is determined down to a
single named gap (IQ4_KS, M, artifact-gated). What remains is not a coverage
question but three items belonging to other lines of work — a quant type code,
a tenant composition, and a diffusion-LM semantics decision — each with its
gate named. #334 can close.
