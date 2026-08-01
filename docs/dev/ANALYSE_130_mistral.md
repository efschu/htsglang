# ANALYSE #130 — Mistral family support

Deep research, design and task placement. **No implementation.** Discharges
the feature-analysis-file duty for #130.

---

## 1. PR-check first — and it settles most of the task

Upstream sglang's Mistral coverage is **broad and current**. The registry
holds eight Mistral-family model files:

| file | lines | derivation | what it is |
| --- | --- | --- | --- |
| `mistral.py` | 195 | **subclasses `LlamaForCausalLM`** | Mistral dense + the `MistralFormat` (native, non-HF) variant + `Mistral3ForConditionalGeneration` |
| `ministral3.py` | 169 | **subclasses `LlamaForCausalLM`** | Ministral 3 |
| `mistral_large_3.py` | 79 | **subclasses `DeepseekV3ForCausalLM`** | Mistral Large 3 (MLA + MoE, DeepSeek-V3-shaped) |
| `mixtral.py` | 480 | standalone `nn.Module` | Mixtral sparse MoE |
| `mixtral_quant.py` | — | standalone | quantized Mixtral path |
| `pixtral.py` | 1043 | composes the two above | Pixtral vision |
| `mistral_eagle.py` | 208 | `LlamaForCausalLMEagle` | **EAGLE drafter for Mistral** |
| `mistral_large_3_eagle.py` | — | — | **EAGLE drafter for Mistral Large 3** |

Plus the parts that historically cost the most effort in a Mistral
integration, all already present:

* `function_call/mistral_detector.py` — tool-call parsing
* `utils/hf_transformers/mistral_utils.py` — `mistral_common` / tekken
  tokenizer handling
* `multimodal/processors/voxtral.py` + `models/voxtral.py` — Voxtral (audio)

Open/closed PR state (API, 2026-08-01):

* **Magistral: zero PRs.** Magistral is a reasoning finetune of Mistral Small
  and loads through `MistralForCausalLM`; there is nothing to port.
* **Devstral: `#24110 "[codex] Support text-only Devstral serving"` — CLOSED
  unmerged.** Devstral is likewise a Mistral-Small-shaped coding model. Worth
  one question upstream before anyone assumes it needs work (same posture as
  #322's cut 0), but the base architecture is already served.
* **Ministral3: `#26679` open** — an XPU loading fix, i.e. the model is live
  and being maintained.
* Pixtral and Mistral-Large-3 appear across dozens of PRs as ordinary
  first-class citizens (AMD kernels, CI, batching fixes).

**Conclusion of phase 1: "Mistral support" is not a porting task.** The models
load. What is left is the fork-axis question — which of *our* levers apply,
and where a real gap sits.

---

## 2. Inventory and what fits 72 GB

**Nothing Mistral-family is on this box.** No checkpoint, no recipe.

Against 72 GB (5090 32 + 2x 3080 20), realistically servable:

| model | params | verdict on this rig |
| --- | --- | --- |
| Ministral 3B / 8B | 3-8 B | trivially fits, any format |
| Mistral Small 3.x / Magistral Small / Devstral Small | ~24 B | comfortable in FP8, INT8-W8A8, AWQ, GPTQ — the same formats we already run Qwen3.6-27B in |
| Pixtral 12B | 12 B + tower | fits |
| Voxtral | small | fits |
| Mixtral 8x7B | ~47 B | fits at 4-bit |
| Mixtral 8x22B | ~141 B | 4-bit is ~70 GB — marginal, and marginal on a heterogeneous rig means the weak card binds |
| **Mistral Large 3** | DeepSeek-V3 class | **does not fit.** Named as an exclusion with a hard reason: the weights exceed the rig. Not a code gap. |

The interesting band is therefore **the 24B Small family** — it is the exact
size and format profile of the model this fork is tuned on, which is why the
axis analysis below comes out as favourably as it does.

---

## 3. Coverage matrix — fork axes per family member

| model | uneven TP | DCP | spec | GGUF | spill/hib | graphs | gap family |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral Small / Magistral / Devstral | inherited (Llama) | inherited | **EAGLE present** | inherited | inherited | inherited | none — Llama path |
| Ministral 3 | inherited (Llama) | **see §4** | via Llama EAGLE | inherited | inherited | inherited | **#91 sliding-window class** |
| Mixtral 8x7B / 8x22B | MoE expert split | MoE | soft | soft | soft | yes | **#74/#116 MoE-expert class** |
| Pixtral | text part inherited | text part | soft | soft | yes | yes | **#54 vision-adapter class** |
| Voxtral | inherited | inherited | soft | soft | yes | yes | audio processor present |
| Mistral Large 3 | DSV3 path | DSV3 path | **EAGLE present** | soft | yes | yes | does not fit the rig |

The single most useful fact in that table: **Mistral dense is a
`LlamaForCausalLM` subclass in 195 lines.** Llama-3.1-8B is a model this fork
already serves under uneven TP, DCP, spec and spill — so for the whole Small
family those axes are inherited rather than ported. Mistral Large 3 inherits
the same way from `DeepseekV3ForCausalLM`, which is why it is 79 lines.

---

## 4. The one real defect found

`ministral3.py:61-66`:

```python
# sliding window
self.sliding_window = getattr(config, "sliding_window", None)
if self.sliding_window is not None:
    # Update RadixAttention with sliding window if needed
    # currently RadixAttention in sglang handles this mostly via logic in
    # forward/flashinfer
    pass
```

The config value is read and then **nothing is done with it**, on the stated
assumption that the attention backend applies the window. Two possibilities,
and the comment does not distinguish them:

1. the backend really does apply it — then this is dead code and a misleading
   comment; or
2. it does not — then Ministral 3 runs **full attention where the model was
   trained with a window**, which is a correctness bug that only shows past
   the window length, i.e. exactly the failure mode that does not appear in a
   short smoke test.

**This is soft (unbuilt/unverified), not hard**, and it is cheap to settle: a
long-context A/B against a reference implementation, or reading whether the
per-layer `RadixAttention` is constructed with a window. It should be settled
before anyone serves Ministral 3 for long context.

**SETTLED (#378).** The window was NOT applied. Three gates missed at once,
each confirmed by execution: `LlamaAttention` built `RadixAttention` without
a window (taking the -1 "no window" default); `Ministral3ForCausalLM` had no
`get_attention_sliding_window_size`, so `ModelRunner`'s first branch missed;
and `"Ministral3ForCausalLM"` is absent from `is_hybrid_swa_model`'s arch
set, so its second branch missed too. Ministral 3 therefore ran full
attention on a model whose config declares a window — correct up to the
window length, silently wrong past it. Fixed in #378 through the same
plumbing Gemma4 uses (optional window on `LlamaAttention` forwarded to
`RadixAttention`, plus the runner hook), with the hybrid-SWA classification
deliberately left alone: a uniformly windowed model is not a two-pool
hybrid, and the interleaving question needs a checkpoint. The #91 SWA x DCP
interaction remains open.

It also places Ministral 3 in the **#91 Gemma4 sliding-window x DCP** gap
family: under token-sharded DCP a windowed layer's owner rule interacts with
the window boundary, which the fork solved once for the SWA-hybrid pool
(`swa_hybrid_dcp_lane`). Whether Ministral 3's window is global rather than
per-layer-interleaved decides whether that machinery applies as-is.

---

## 5. Performance angle — where our levers already apply

**What makes the Small family fast on our stack, for free:**

* **Attention geometry is GQA and Llama-shaped**, so the uneven-TP head split,
  the per-(rank, family) GEMM scores (#324) and the cost library (#348b) apply
  without a new family entry.
* **An EAGLE drafter exists in-tree** (`mistral_eagle.py`), so the spec line —
  the fork's largest measured lever — is available on day one rather than
  needing a drafter trained or ported.
* **Vocab is the ordinary 32k/131k range**, not a Gemma-class 256k. The vocab
  shard is a decode-time weight stream, so a smaller vocab means the
  `--rank-vocab-ratio` lever matters *less* here than on Gemma — one fewer
  thing to tune, not a gap.

**Where a real gap sits:**

* **Mixtral routing.** The MoE expert split is the #74/#116 class; our expert
  offload and uneven-expert work was built and measured on Qwen3.5/3.6 MoE
  geometry, and Mixtral's 8-expert top-2 routing is a different shape from a
  fine-grained 128-expert MoE. Not hard — the machinery is general — but it is
  the one place where "inherited" would be an overclaim.
* **Pixtral's tower** is the #54 vision-adapter class: our text-only-serving
  toggle and the vision byte accounting already exist, but they were exercised
  on Qwen-VL, not on Pixtral's tower.

**Versus llama.cpp / vLLM:** for the 24B Small family there is no structural
reason we would be slower, and two reasons we should be faster on *this* rig —
uneven TP across mismatched cards, and spec with an existing drafter. Neither
is available in llama.cpp's split, and vLLM's even split leaves the 5090
underused. That is a hypothesis to measure, not a claim.

---

## 6. Recommended order, with effort/yield pairs

No thresholds — each line is effort against yield, judged as a ratio.

1. **Boot Mistral Small 3.x (or Magistral Small) FP8 at TP=3 uneven, with the
   in-tree EAGLE drafter.**
   *Effort:* S — a download and a boot; zero code, since the axes are
   inherited from the Llama path.
   *Yield:* high — it either confirms the whole Small family is free on our
   stack, or it produces the first real gap list. Also the cheapest possible
   test of whether our hetero levers generalize beyond Qwen/Gemma, which is a
   question worth more than Mistral itself.

2. **Settle the Ministral 3 sliding-window question (§4).**
   *Effort:* S — read the backend construction, or one long-context A/B.
   *Yield:* moderate but correctness-shaped: it is a silent-wrong-output risk,
   and silent beats loud in cost every time.

3. **Mixtral 8x7B 4-bit under uneven TP.**
   *Effort:* M — the MoE expert-split path exists but was measured on a
   different routing shape.
   *Yield:* moderate — a second MoE geometry through our expert machinery is
   real generalization evidence, and 8x7B is the largest Mistral MoE that fits
   comfortably.

4. **Pixtral**, only if the user wants vision from this family.
   *Effort:* M. *Yield:* narrow — we already serve a VL family.

---

## 7. Exclusions, each with a named hard reason

* **Mistral Large 3 — does not fit 72 GB.** A DeepSeek-V3-class checkpoint
  exceeds the rig. This is a hardware statement, not a code gap: the model
  file is 79 lines because it inherits everything, so on a bigger rig it is
  already supported. Per rig-is-lower-bound this must **not** be recorded as a
  general limitation.
* **Nothing else is excluded.** Per alles-mit-allem-kombinierbar, every other
  family member is a priority candidate; none of them has an architectural
  blocker, only unmeasured surface.

---

## 8. Summary

"Mistral support" is largely already there, and the survey's value is saying
so precisely rather than opening an integration project. The dense Small
family is a `LlamaForCausalLM` subclass with an EAGLE drafter in the tree, so
our uneven-TP, DCP, spec and spill axes are inherited; the tokenizer and
tool-call work that usually dominates a Mistral port is done; Magistral and
Devstral are finetunes of an architecture that already loads.

What is actually open is one possible correctness defect (Ministral 3's
unused `sliding_window`), one genuine generalization question (Mixtral's
routing shape through our expert machinery), and one hardware exclusion
(Mistral Large 3). The recommended first step is a download and a boot, not a
patch.
