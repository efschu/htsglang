# DESIGN — non-contiguous PP layer ownership (the addressing blocker)

Built on `feat/pp-layer-set` off the train tip. Hermetic; no boot, no GPU.

`DESIGN_family_fullplan.md` (branch `docs/family-fullplan`) named two
independent blockers for the family full plan: the WIRE (barlink send/recv,
Slot-3) and the ADDRESSING — a pipeline stage has always been an interval, so
"all 48 linear-attention layers here, the 16 interleaved full-attention layers
there" could not be said at all. This is the addressing half.
**§4 below belongs in `DESIGN_family_fullplan.md` §2.1 when the branches
meet** — it answers the capacity question that spec left open.

## 1. The index contract

`SGLANG_PP_LAYER_SET`, per-stage sets separated by `;`, each a comma list of
ranges and singletons:

```
SGLANG_PP_LAYER_SET="0,1,2,4,5,6,…;3,7,11,…;35,39,…"
```

`get_pp_layer_set()` (`distributed/utils.py`) returns this stage's owned ids,
or **`None` when the variable is unset** — which is what keeps the contiguous
path byte-identical. `get_pp_indices` is untouched.

**The validation is the deliverable, not the parser.** Both ways of getting a
partition wrong are silent: a duplicated layer is computed twice and merely
costs time, and a missing layer becomes a `PPMissingLayer` pass-through
(`torch.nn.Identity`), so the model answers with that layer quietly skipped.
Refusals therefore name the exact layers: duplicates with the stages that claim
them, missing layers as a list, out-of-range ids, wrong stage count, backwards
ranges, non-integers.

## 2. Construction

`make_layers` already built a FULL-LENGTH `ModuleList` with placeholders
outside the owned range, which is why this was cheap: interleaved placeholders
are the same object as trailing ones. Under a set it emits a real layer at each
owned index and a placeholder elsewhere, and publishes `owned_layers` for
membership questions. `start_layer`/`end_layer` become FIRST and one-past-LAST
owned — which is what their consumers actually use them for ("am I the first
layer on this stage").

## 3. The consumer audit — one real hazard, one filed question

**FIXED: `num_effective_layers` (`model_executor/model_runner.py`).** It was
`end_layer - start_layer`, i.e. the SPAN, which equals the count only while a
stage is an interval. It feeds `layer_num=` for KV pool allocation at 8+ sites
in `model_runner_kv_cache_mixin.py`. Under the full plan's FA stage — 8 layers
spanning 29 — the pool would be sized for **29 layers instead of 8**, and the
boot would fail on memory it never needed. It now takes `len(owned)` when a set
is configured, reading the SAME parser rather than an attribute the models
would each have to propagate (two derivations is how they disagree).

**FILED, not forced: KV pool INDEXING under non-contiguous ownership.** Sizing
is fixed; addressing is not. The pool is indexed by layer, and a stage owning
`{3,7,11,…}` needs either a compacted index or a sparse map — that is a design
question with a correctness surface, not a mechanical rename. Adjacent prior
art is the #718/#719 pool-rebind work. **Not attempted here**; a wrong choice
would be silently wrong per layer, which is the worst available outcome.

Also noted, not fixed: `hardware_backend/npu/memory_pool_npu.py` iterates
`range(self.start_layer, …)` in six places; vendor-owned and not on this rig's
path. `deepseek_v2.py`, `glm4_moe_lite.py`, `kimi_linear.py`,
`bailing_moe_linear.py` each compute `total_num_layers = end - start`; the same
span-vs-count shape, per model, and each needs the same one-line treatment if
that model is ever placed non-contiguously. **INFERRED** — I did not read all
59 `start_layer` consumers; I audited the two grep shapes that can be wrong
(range loops and span arithmetic) and fixed the one on this rig's path.

## 4. GDN state capacity — ANALYTIC, and it PINCHES

The full-plan spec left this open. Derived from the checkpoint config and the
tree's own shape code (`configs/mamba_utils.py`: `temporal = (num_heads,
head_dim, state_size)`, `conv = (intermediate + 2·n_groups·state_size,
conv_kernel−1)`):

| quantity | value | source |
| --- | --- | --- |
| temporal state | 48 × 128 × 128 = 786 432 el | `linear_num_value_heads/​value_head_dim/​key_head_dim` |
| conv state | (6144 + 2·16·128) × 3 = 30 720 el | `linear_conv_kernel_dim 4` |
| per layer per slot | 817 152 elements | sum |
| **state dtype** | **float32** | `text_config.mamba_ssm_dtype: float32`, explicit in the checkpoint, and the tree's default (`mamba_utils.py:133`) |
| **per layer per slot** | **3.117 MiB** | 817 152 × 4 B |
| **48 layers per slot** | **149.6 MiB** | the full plan concentrates ALL GDN state on the 5090 |

Against the 5090 headroom the spec computed (**4738 MiB** free after GDN
weights, #727 int8 vocab, arming floor 1728 and corridor 1024):

| graph pool | slots (fp32, 48 layers) |
| --- | --- |
| 0 MiB | 31 |
| 1024 MiB | **24** |
| 1536 MiB | **21** |

**And without #727 the plan pinches hard.** On bf16 vocab the free figure is
2313 MiB, giving **~8 slots** at a 1 GiB graph pool. That moves #727 from
"load-bearing" to **required**, now with a number behind the word.

**Headline, stated loudly as asked: at fp32 state — which this checkpoint
explicitly selects — the full plan supports roughly 21–24 concurrent mamba
slots on the 5090.** That is the same order as the rig's existing
`--max-mamba-cache-size` settings, so it does not KILL the plan, but it leaves
little margin, and it is **2× the concentration of today's contiguous layout**,
where the 5090 holds 24 GDN layers rather than 48. Anyone planning higher
concurrency on the full plan should treat this as the binding constraint —
not the weights, which fit comfortably.

**ANALYTIC, pending metal confirmation.** It is arithmetic over the config and
the tree's shape code; it does not include allocator granularity, fragmentation
or any per-slot overhead beyond the state tensors.

## 5. Not built here

The crossing schedule (send at every ownership change). The addressing contract
above is its precondition — a schedule needs to know where a boundary IS — but
the schedule itself wants the send/recv SHAPE that Slot-3 is defining, and
writing it against a guessed interface is how two halves end up disagreeing.
Filed rather than begun.
