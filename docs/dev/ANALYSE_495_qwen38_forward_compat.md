# ANALYSE #495 — Qwen3.8 forward-compatibility, and what #497 did about it

Research state: 2026-08-03. Adopted into the tree by #497. The research half
(§1-§3) is the #495 desk study, unchanged in substance; §4 is what #497
executed against this fork and is the part that is testable.

**Read §1.1's caveat before quoting anything from §1.** Almost every fact
about Qwen3.8 in this document comes from secondary tech media, not from a
primary source anyone here read. The one primary artifact is a vLLM pull
request, and it is unmerged and failing CI.

---

## 1. Evidence

### 1.1 Established, with source and date

**Qwen3.8-Max exists, a preview is running, and a 27B open-weight checkpoint
is announced for "next week" — but only via secondary sources.**

- 2026-07-19: Alibaba previewed "Qwen3.8-Max": 2.4 T parameters, sparse MoE,
  first multimodal (text/image/video input) Qwen above 1 T.
  ([MarkTechPost](https://www.marktechpost.com/2026/07/19/alibaba-previews-qwen3-8-max-a-2-4-trillion-parameter-multimodal-model-days-after-moonshots-kimi-k3-open-weight-launch/))
- 2026-08-02/03: launch as an API product, with detail values: 1 M-token
  context, max input 991 K tokens (983 K with thinking on), max output 131 K,
  max reasoning budget 262 K.
  ([MarkTechPost](https://www.marktechpost.com/2026/08/03/alibaba-qwen-releases-qwen3-8-max/),
  [remio.ai](https://www.remio.ai/post/qwen-3-8-open-weight-model-announcement-promises-2-4t-parameters-but-proof-comes))
- Qwen3.8-27B is named as an open on-prem checkpoint alongside Max, weights
  promised "next week" as of 2026-08-02/03.
  ([SCMP](https://www.scmp.com/tech/article/3362738/alibabas-ai-model-qwen38-max-made-widely-accessible-ahead-open-weights-release),
  [buildfastwithai](https://www.buildfastwithai.com/blogs/qwen3-8-preview-2-4t-params-open-weights-release))
- **Caveat, self-checked.** The official Qwen blog could not be loaded (the
  fetch returned only the word "Qwen") and the linked vendor post could not be
  read (HTTP 402). Everything above rests on several mutually confirming tech
  outlets, **not** on a primary source read here. The critical article
  (remio.ai) says so explicitly: "Alibaba has supplied something users can
  experience, but not something researchers can fully audit" — no model card,
  no architecture statement, no benchmark methodology published.

**vLLM PR #50068 — the only real code artifact, read directly.**

- [`vllm-project/vllm#50068`](https://github.com/vllm-project/vllm/pull/50068)
  "[Model] Enable Qwen3.8 for AMD Rocm", opened 2026-07-28, still open and
  unmerged as of 2026-08-03, pre-commit CI failing.
- Description, verbatim: *"register the text-only `Qwen3_5ForCausalLM` and
  `Qwen3_5MoeForCausalLM` architectures / advertise hybrid and M-RoPE support
  on the causal implementation / expose Gated DeltaNet Mamba cache dtype,
  shape, and copy metadata so text-only Qwen3.5-compatible checkpoints such as
  Qwen3.8 Max FP8 can initialize through the causal LM path"*.
- The test plan claims a two-node ROCm validation at TP=8/PP=2 of an
  "equivalent patch", i.e. the author appears to have early access to a real
  Qwen3.8-Max FP8 checkpoint.
- Diff: one file, `vllm/model_executor/models/qwen3_5.py`, +40/−0. Adds
  `SupportsMRoPE` to `Qwen3_5ForCausalLMBase` (absent from the *text-only*
  causal path although `rope_parameters.mrope_section` already exists in the
  config) plus three classmethods deriving GDN cache metadata from exactly the
  HF config fields `linear_num_key_heads`, `linear_num_value_heads`,
  `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` —
  the same field names Qwen3.5/3.6 use.
- **Defect in the PR:** it adds `IsHybrid` to the base-class list a second
  time, which is a `TypeError` (duplicate base class) in Python. Consistent
  with the failing pre-commit. Treat the PR as an intent signal, not as
  something to copy.
- **What it means for us:** an engineer with an early-access checkpoint wires
  Qwen3.8 Max through the **existing** `Qwen3_5*` classes, not a new
  `Qwen3_8...` class. That is the strongest available hint of architecture
  continuity.

**Null results, stated as results.**

- `huggingface/transformers`: 0 hits for "qwen3.8"/"qwen38"/"qwen3_8" in
  issues/PRs. No new model class in sight.
- `sgl-project/sglang`: 0 hits. (For contrast, `qwen3_5` has 17 — an actively
  maintained path.)
- `ggml-org/llama.cpp`: 0 hits. `gguf-py/gguf/constants.py` already carries
  `MODEL_ARCH.QWEN35` / `QWEN35MOE` (`"qwen35"`/`"qwen35moe"`) but no
  `QWEN38`.
- QwenLM GitHub org (~50 repos by `updated_at`): newest version-named repo is
  still "Qwen3.6".
- Official `Qwen` HF organisation by `lastModified`: newest entry is
  `Qwen/Qwen3-ASR-*-hf` (2026-07-22). No Qwen3.7 or 3.8 repos. The only
  "Qwen3.8" hit in model search is a third-party namespace — noise.
- ModelScope could not be read (client-rendered page). That is a **tool
  limit**, not a "nothing found" result.

### 1.2 Rumour, marked as such

- An NVIDIA developer-forum thread (2026-07-19) speculates about "35B MoE and
  27B Dense" variants. Pure user speculation, no confirmation.
- No Reddit thread with defensible technical detail (layer count, kv heads,
  GDN variants) was found.

### 1.3 Version nomenclature

"Qwen3.7" exists only as a closed API-only Max tier (announced 2026-05-19/20,
1 M context, never open-weight). So the Max tier numbering (3.6 → 3.7 → 3.8)
and the open-weight numbering (3.5-27B → 3.6-27B → *no open 3.7* → 3.8-27B)
are **not synchronous**. Do not assume every Max version has an open
counterpart of the same number.

---

## 2. Architecture delta

**There is no public `config.json` for Qwen3.8-27B.** Everything about
geometry below is precedent inference from 3.5/3.6, not evidence about 3.8.

### 2.1 Read directly: `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.6-27B` configs

Byte-identical in every relevant field:

| field | value (3.5 AND 3.6) |
|---|---|
| `model_type` (top level) | `qwen3_5` |
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` |
| `text_config.model_type` | `qwen3_5_text` |
| `num_hidden_layers` | 64 |
| `full_attention_interval` | 4 (3x linear, 1x full, repeating) |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_conv_kernel_dim` | 4 |
| `mamba_ssm_dtype` | float32 |
| `mtp_num_hidden_layers` | 1 |
| `mtp_use_dedicated_embeddings` | false |
| `vocab_size` | **248320** |
| `rope_parameters` | `mrope_interleaved: true`, `mrope_section: [11,11,10]`, `rope_theta: 10000000` |
| `hidden_size` / `intermediate_size` | 5120 / 17408 |

**Vocabulary correction.** A "~152 k vocab" figure was in circulation as the
comparison baseline. The configs actually read say `vocab_size: 248320`,
~1.63x larger; 152 k is the older plain Qwen3 text vocab. Correct this wherever
it is quoted, independently of the Qwen3.8 question.

### 2.2 Architecture-string continuity (precedent, not proof)

`model_type`/`architectures` did not change between 3.5 and 3.6, and PR #50068
reuses the existing classes. The **working hypothesis** is therefore that
Qwen3.8 also carries `model_type: qwen3_5` / `qwen3_5_text`. This is inference
from one unmerged PR plus one precedent, not a confirmation.

### 2.3 The M-RoPE gap

`rope_parameters.mrope_section` is already present in Qwen3.5/3.6 configs, but
vLLM's text-only causal path did not declare `SupportsMRoPE`. That is a
pre-existing gap that the Qwen3.8 work made visible, not a 3.8 novelty — which
is why §4.3 checks the same shape here.

### 2.4 llama.cpp / GGUF

Upstream llama.cpp has native `qwen35`/`qwen35moe` MODEL_ARCH enums with SSM
tensor tags. No `qwen38`. Whether the existing `qwen35` converter picks up
Qwen3.8 depends on the `model_type` string — untestable while no checkpoint
exists.

---

## 3. Day-0 checklist

Run these against the first real Qwen3.8-27B checkpoint.

1. `config.json` / `text_config`: is `model_type` still `qwen3_5` /
   `qwen3_5_text` and `architectures` still
   `Qwen3_5ForConditionalGeneration`, or is there a new string? **This decides
   whether the family table matches automatically.** If a NEW architecture
   string appears, note §4.1: resolution does not refuse, it falls back to the
   generic transformers backend — so check the resolved class explicitly
   rather than trusting a successful boot.
2. `full_attention_interval`, `num_hidden_layers`, `layer_types` — geometry
   against 3.5/3.6 (64 layers, interval 4). Covered generically by §4.2.
3. `linear_num_key_heads` / `linear_num_value_heads` / `linear_key_head_dim` /
   `linear_value_head_dim` / `linear_conv_kernel_dim` — GDN geometry, feeds
   the uneven-TP/DCP sibling validation.
4. `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings` — MTP/NEXTN draft
   head format, feeds the spec-head route.
5. `vocab_size` + tokenizer files — still 248320? Affects lm_head sharding and
   GGUF vocab handling.
6. First published GGUF: pull the tensor name list (`gguf-dump`) and compare
   against the `qwen35` arch tensor names in llama.cpp's `constants.py` —
   shows immediately whether the community converter reuses the existing arch.
7. `AutoConfig.from_pretrained(...).architectures` on the real checkpoint.
8. Follow vLLM PR #50068 (or a successor): on merge it shows the real,
   early-access-validated wiring, and whether the duplicate-`IsHybrid` defect
   was fixed.

**Explicitly NOT recommended:** do not pre-build tensor names, arch enums or
family-table entries for a guessed geometry. The evidence base is empty and a
wrong entry is worse than none.

---

## 4. What #497 checked in this fork

All three checks are generic — they hold for any future checkpoint reusing an
existing `model_type`, not just Qwen3.8. Pinned by
`test/registered/unit/model_loader/test_qwen38_forward_compat_497.py`
(21 tests, hermetic, no GPU). **No production code was changed**: all three
came back "already generic", and the tests are ratchets that keep them that
way.

### 4.1 Resolution keys on `model_type`, not on a version string — CONFIRMED

`create_gguf_adapter` (`model_loader/gguf_registry.py:81-88`) reads exactly one
field:

```python
model_type = getattr(hf_config, "model_type", None)
cls = get_gguf_adapter_class(model_type)
```

and `_MODEL_TYPE_TO_GGUF_ARCH` (`model_loader/gguf_qwen35.py:65-70`) is keyed
on `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`. The string
`name_or_path` does not appear in the registry at all. A checkpoint named
anything, carrying a known `model_type`, resolves without a code change.

A sweep of every string literal the `model_loader` and `configs` packages
EVALUATE (docstrings and comments excluded — the family is documented in prose
everywhere) finds no display-form `Qwen3.5`/`Qwen3.6` match on the load path.
The planner does map display names
(`planner/rig_profile_source.py:64-65`), which is legitimate: it labels a
measurement store, it does not decide a load.

**One finding worth acting on at day 0.** An architecture string that is NOT
registered does not refuse. `_normalize_archs` (`models/registry.py:61-78`)
filters unregistered architectures out and, if anything was dropped, appends
`TransformersForCausalLM`:

```python
normalized_arch = list(filter(lambda model: model in self.models, architectures))
if len(normalized_arch) != len(architectures):
    normalized_arch.append("TransformersForCausalLM")
```

So a Qwen3.8 checkpoint declaring a new architecture would come up on the
generic transformers backend rather than saying so — a soft landing, but a
silent one, with none of this fork's features. Day 0: assert the resolved
class, do not infer it from a boot that succeeded. (The named refusal exists —
`_raise_for_unsupported` names the architecture and lists the supported set —
but only fires when even the fallback is absent.)

### 4.2 Hybrid geometry is config-driven — CONFIRMED

`ServerArgs.declared_layer_kinds` (`server_args.py:12930-12968`) probes, in
order, `layer_types` / `layers_block_type`, then `full_attention_interval`,
then defaults to all-attention, reading from the top level or from
`text_config`. Depth comes from `declared_num_hidden_layers` over
`("num_hidden_layers", "n_layer", "num_layers")`. The GDN tuple is read
per field with `getattr` (`server_args.py:10292-10298`), and
`layer_family_census` (`uneven_perf.py:2878+`) derives the per-layer family
census from the config rather than assuming a uniform stack.

Exercised at depth 48 / interval 6 and depth 32 / interval 8 under a nested
`text_config`, plus an explicit `layer_types` list overriding the interval:
all correct with no code change. A ratchet forbids re-introducing a
`num_hidden_layers == <literal>` guard in the Qwen3.5 model or adapter.

### 4.3 The M-RoPE declaration gap — PRESENT HERE TOO, not closed

Both gate predicates, at their sources:

* **Runner side**, `model_executor/model_runner.py:599-604` — reads the
  CONFIG only, with no model-class term:

  ```python
  rope_scaling = getattr(model_config.hf_text_config, "rope_parameters", None) \
      or getattr(model_config.hf_text_config, "rope_scaling", {})
  self.model_is_mrope = (rope_scaling is not None and "mrope_section" in rope_scaling)
  ```

  A text-only Qwen3.5/3.6 config carries `mrope_section`, so this is **True**
  for a text-only checkpoint, and `forward_batch_info.py:876` then builds
  `mrope_positions`.

* **Model side**, `models/qwen3_5.py` — `self.is_mrope_enabled` is set at
  `:1839` and `:1996` only, i.e. on `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration`. The text-only `Qwen3_5ForCausalLM`
  (`:1280`) and `Qwen3_5MoeForCausalLM` (`:1611`) never set it.

* **Consumer**, `model_executor/runner/prefill_cuda_graph_runner.py:521-531`:

  ```python
  if forward_batch.mrope_positions is None:
      return forward_batch.positions
  if getattr(model, "is_mrope_enabled", False):
      return forward_batch.mrope_positions
  language_model = getattr(model, "language_model", None)
  if getattr(language_model, "is_mrope_enabled", False):
      return forward_batch.mrope_positions
  return forward_batch.positions
  ```

So on the text-only path the runner COMPUTES mrope positions and this consumer
discards them, because neither attribute exists. That is the same declaration
gap vLLM #50068 closes on its side.

**Deliberately not fixed here.** Adding the declaration changes which
positions a captured graph replays, on the breakable/piecewise prefill path.
That needs a boot to validate and no GPU was available for this task, so
closing it is a GPU-window decision, not a desk change. The gap is
characterised by tests instead: they assert that exactly two classes declare
the attribute and that both are the `ForConditionalGeneration` ones. When the
fix lands, those tests flip together — that is the intended signal, and the
count in the test plus this section must be updated in the same change.

Day-0 relevance: a Qwen3.8-27B text checkpoint would take exactly this path.
Whether the mismatch changes output depends on whether the plain and mrope
position tensors coincide for pure-text sequences, which is what the boot has
to establish.

---

## Sources

- https://www.marktechpost.com/2026/07/19/alibaba-previews-qwen3-8-max-a-2-4-trillion-parameter-multimodal-model-days-after-moonshots-kimi-k3-open-weight-launch/
- https://www.marktechpost.com/2026/08/03/alibaba-qwen-releases-qwen3-8-max/
- https://www.remio.ai/post/qwen-3-8-open-weight-model-announcement-promises-2-4t-parameters-but-proof-comes
- https://www.scmp.com/tech/article/3362738/alibabas-ai-model-qwen38-max-made-widely-accessible-ahead-open-weights-release
- https://www.buildfastwithai.com/blogs/qwen3-8-preview-2-4t-params-open-weights-release
- https://github.com/vllm-project/vllm/pull/50068
- https://forums.developer.nvidia.com/t/qwen-3-8-is-about-to-launch-open-weight-too/377396
- https://www.yottalabs.ai/post/qwen-3-7-max-release-date-features-open-source-status-and-how-to-access-2026
- https://www.datacamp.com/blog/qwen3-7-max
- https://www.vals.ai/models/alibaba_qwen3.7-max
- https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py
- https://huggingface.co/api/models?author=Qwen
