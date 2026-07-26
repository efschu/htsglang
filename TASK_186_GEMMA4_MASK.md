# Task #186 -- Gemma-4 custom_mask on prefill: decision record

Branch `fix/gemma4-textonly-mask`, worktree `/spinning/wt-gemma-mask`, based on
`origin/integration/r3-probe` @ `8d87afe941`.

Trigger: `INTEGRATION_R3_VALIDATION.md` "#96 Stage B is RED: Gemma-4 installs a
custom mask on EVERY prefill". This record supersedes that section's root-cause
attribution.

---

## 1. The briefed premise is false. Corrected finding.

The briefing (and the H4 section of `INTEGRATION_R3_VALIDATION.md`) reads the
RED as: `gemma4_mm.py:406` appends a mask slot outside the `mm_inputs` guard,
so *even a pure-text prefill* installs one, and `75625 = 275^2` was that
text-only warmup prefill.

Both halves are wrong.

**(1a) The 275-token prefill was an image request, not a text one.**

`prepare_attn_masks` is not called unconditionally. The callsite guards on
`contains_image_inputs()`:

```
python/sglang/srt/models/gemma4_mm.py:657-665
    if (forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.contains_image_inputs()):
        self.prepare_attn_masks(...)
```

A batch in which no request carries an image never reaches the method at all.
So the 275-token prefill must have contained an image -- and it did. sglang's
own server warmup sends a **base64 PNG** for any model advertising
`has_image_understanding`:

```
python/sglang/srt/entrypoints/http_server.py:2068  is_vlm = model_info["has_image_understanding"] ...
python/sglang/srt/entrypoints/http_server.py:2034  MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGg..."   # 32x32
python/sglang/srt/entrypoints/http_server.py:2091-2115  chat request: [image_url, "Describe the image."]
```

Measured on CPU against `gemma-4-31B-it-int4-AutoRound` (its real processor and
chat template, no GPU):

| request | prompt tokens | image tokens |
|---|---|---|
| warmup PNG + "Describe the image." | **275** | **256** |
| same text, no image | 17 | 0 |

275 exactly. `275^2 = 75625` exactly. The mask that killed #96 Stage B was a
**genuine bidirectional image mask** over 256 soft tokens, not a degenerate
causal one. Route (a) as briefed -- "make text-only mask-free" -- would not
have moved that boot one inch.

**(1b) The append at `:406` is load-bearing, not a stray.**

Moving it inside the `mm_inputs is not None` branch, which is what the briefed
route (a) asks for, is **not a cleanup -- it silently corrupts attention.**

The mask is one flat batch-wide buffer addressed by `mask_indptr`, and the
extend kernel's mask switch is a whole-launch constexpr:

```
python/sglang/kernels/ops/attention/extend_attention.py:675
    USE_CUSTOM_MASK = custom_mask is not None          # per launch, not per request
python/sglang/kernels/ops/attention/extend_attention.py:311
    cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)
python/sglang/kernels/ops/attention/extend_attention.py:500-509
    row stride = (cur_seq_len + window_kv_offset)      # this request's own dims
```

There is no per-request "no mask here" escape. If a text-only request in a
mixed batch contributes no slot, `mask_indptr[i] == mask_indptr[i+1]`, and the
kernel still reads at `mask_indptr[i]` using request *i*'s row stride -- i.e.
it reads the *next* request's mask bytes. Wrong attention, no crash; and an
out-of-bounds read when the skipped request is last in the batch.

This is the [[geteilte-puffer-familie]] shape once more, but inverted from how
H4 read it: the redundant-looking append is the invariant, and the "obvious
fix" is the bug. The invariant is now pinned by
`test_mixed_batch_keeps_a_slot_for_the_text_only_request`.

---

## 2. Mask semantics: what the mask actually encodes

Question the briefing asks to settle before touching anything: does the mask
carry semantics beyond image bidirectionality -- sliding-window interaction in
particular?

**No. It is a pure AND-term on top of the kernel's own masking, and it carries
nothing but image bidirectionality.**

* **Sliding window is applied independently** and is never replaced by the
  custom mask -- separate `if SLIDING_WINDOW_SIZE > 0:` terms `&=`-ed after the
  mask term, in every stage:
  `extend_attention.py:381-387` (prefix), `:524-529` (extend), `:946-963`
  (unified). Dropping the mask cannot lose SWA semantics.
* **The mask replaces only the causal term**, and only in the extend stage:
  `:499-522` -- `if USE_CUSTOM_MASK: ... elif IS_CAUSAL: mask_causual = m >= n`.
* **The prefix stage ignores the mask entirely** in the non-DCP path:
  `SKIP_PREFIX_CUSTOM_MASK` defaults to `True` (`:640`) and no caller overrides
  it (grep: the only two hits in the repo are the definition and its use).
* **The row stride is the GLOBAL prefix length**, consistent with #180's §2.1
  note. For SWA layers `cur_seq_len_prefix` is the *windowed* prefix and
  `window_kv_offset` (`:315`, `:505`) adds back `global_prefix - window_prefix`,
  so stride and column base both land on global coordinates. The model builds
  the mask on global coordinates, so this is coherent -- and it is exactly why
  an owner-sharded prefix breaks it.

**Degeneracy proof.** With no image span written, the mask is
`ones(extend, extend+prefix).tril(diagonal=prefix)`, i.e.
`mask[m, n] = (n <= m + prefix)`.

* Extend stage reads column `prefix + n_ext`: `prefix + n_ext <= m + prefix`
  reduces to `n_ext <= m` -- **bit-identical to the `IS_CAUSAL` branch**.
* Prefix stage reads column `n < prefix`: `n <= m + prefix` is always true --
  all-ones, no effect. (Moot anyway under `SKIP_PREFIX_CUSTOM_MASK`.)
* Tile skipping does not diverge either: `cur_block_m_end` depends on
  `IS_CAUSAL` only, and `SKIP_TILE` (`:532`) can never fire on a causal tile
  inside the diagonal-limited range.

So on a degenerate batch the custom-mask path and the causal path select
exactly the same elements in exactly the same accumulation order. Byte
identity is expected, not merely numerical closeness. Pinned on CPU by
`test_textonly_mask_is_exactly_the_causal_predicate` (mask == kernel predicate)
and `test_sdpa_under_the_textonly_mask_is_byte_identical_to_causal`
(`torch.equal`, float64 SDPA).

---

## 3. Decision

Neither (a) nor (b) as briefed. The premise that separates them does not hold.

**Built: (a'), a batch-level degeneracy skip.** Not "text-only mask-free" --
that is already true (§1a) -- and not per-request (§1b, unsafe). Instead:
*install the mask only if at least one bidirectional bit was actually written
anywhere in the batch.* All-or-nothing at batch granularity is the only
granularity the flat buffer permits, and it is safe by construction.

This is a pure no-op on outputs (§2 degeneracy proof) and removes the buffer +
mask-load path for three real, frequent batches:

| batch | before | after |
|---|---|---|
| multi-turn follow-up, image span wholly in the cached prefix | full causal mask built + installed | none |
| chunked-prefill chunk splitting an image span | full causal mask, image gets causal anyway | none |
| single-token image span | mask that only rewrites the diagonal | none |
| text-only request beside an image one | causal slot, installed | causal slot, **kept** (invariant) |
| image span contained in the extend window | bidirectional mask | unchanged |

The multi-turn row is the common one: every follow-up turn of an image
conversation currently pays an `extend x (extend+prefix)` bool tensor and a
per-tile mask load for a mask that is exactly causal.

Side effect, deliberate but *not* the justification: those batches also stop
tripping the `custom_mask` refusals in `_forward_extend_dcp`
(`triton_backend.py:2307`) and `verify_splitkv_fwd`. That is correct precisely
because there is no non-causal semantics to lose.

**Not built: the real #96 unblock.** See §5. It is (b)-class and larger.

**On the in-place mutation at `:428`.** The briefing asks to set the mask "at
construction time instead of afterwards". That is not reachable: the backend
builds `forward_metadata` *before* the model's forward runs, and the image
spans the mask derives from are known only inside the model. There is no
pre-attention-metadata model hook. What was done instead: both fields are
written together in one place with the structural reason documented inline, the
degeneracy skip removes the write on the batches that never needed it, and the
flat-buffer invariant is stated at the append site and pinned by a test.
Removing the out-of-band write entirely means moving mask construction into the
backend -- a separate refactor, and one that would also have to cover
`gemma3_mm.py:272`, which does the same thing.

**Scope.** `gemma3_mm.py` has the identical structure (`:217` / `:406`) and the
identical degeneracy. It is untouched here on purpose -- Gemma-3 is not on the
DCP lane, the change is not required for #96, and touching it widens the blast
radius of a fix that has not yet had a GPU pass. Registered as a follow-up.

---

## 4. owner.py cross-check (#96 q-head basis vs #180 verify) -- OWED, now clear

Question: does the merged #180 verify path consume
`_plan_aware_dcp_group_q_head_counts` with an expectation that #96's change to
it (full-attention kv basis instead of `max()` over the full and SWA bases)
would break, or vice versa?

**Verdict: NO CONFLICT.** #96's change is a no-op on every configuration #180
can currently reach, and is a *precondition* for the one new configuration the
merge opens.

* Definition: `triton_backend.py:216` (merge-probe) / `:201` (#96).
  Old basis `max(get_total_num_kv_heads(), swa_num_key_value_heads)`; new basis
  the full base only, plus a new `sum(counts) != total_q` hard check.
* One production callsite (`_dcp_group_q_head_counts`, `:1875` / `:1836`) and
  three consumers -- `_replicated_kv_ragged_reindex` (`:1910` / `:1871`),
  `_dcp_gather_q_heads` (`:1942` / `:1903`), `_dcp_merge_q_heads` (`:1963` /
  `:1924`). All three require an exact exhaustive partition:
  `cp_all_gather_heads_uneven` asserts `counts[rank] == local_heads`, merge
  asserts `sum(counts) == out.shape[1]`.
* #180 reaches them (chain target-verify is served through `_forward_extend_dcp`,
  see its own comment at `:2308-2317`) but **touched no head-count code** --
  `git diff b7ffaba1ed~1..cd651ee0dc` over `triton_backend.py` filtered for
  the head-count symbols is empty. Its neighbouring code already assumes the
  full basis (`dcp_full_kv_heads = get_total_num_kv_heads()`, `:669`, used at
  `:1915`), so the old `max()` was an internal inconsistency #96 removes.
* The divergent branch is unreachable on merge-probe anyway: without a plan the
  function early-returns `[local_heads] * dcp_size` (`:250-252`) and never
  evaluates the kv bases; with a plan, `reject_unsupported_dcp_geometry`
  (`:394`) refuses every sliding-window model at boot -- and every in-tree
  model that sets `swa_num_key_value_heads` is a sliding-window model
  (`gemma4_causal.py:311`, `mimo_v2.py:627`, `laguna.py:184`,
  `hf_transformers/config.py:202`).
* Reverse direction: #96's base predates the #180 merge, but the two guard
  edits are distinct lines. `git merge-tree` produces a clean
  `reject_unsupported_dcp_geometry` carrying both conditions.
* No existing test encodes the old basis --
  `test_triton_dcp_head_gather.py` is byte-identical in both trees and its only
  plan-active case is single-base. #96 ships
  `test_swa_dcp_stage_b.py:355 test_a_single_base_model_is_unchanged`.
* Without #96's fix, Gemma-4-31B TP=3 would produce `[8,16,10]` = 34 against 32
  q heads and trip the gather assert (`test_swa_dcp_stage_b.py:310-337`).

**So the #96 merge is not blocked on the head-count basis.** What remains
blocking it: (i) the custom-mask defect, §5 below; (ii) mechanical textual
conflicts in 4 files -- `triton_backend.py` (one import-list hunk),
`layers/dcp/__init__.py` (2 export lists), `layers/dcp/owner.py` (1 adjacent
helper block), `test_triton_dcp_geometry_guard.py` (2, both branches extended
the same guard tests).

**Newly registered seam, mask-related, not head-count-related.** In the merged
tree `init_forward_metadata` drops `custom_mask`/`mask_indptr` unconditionally
under `dcp_size > 1` on the target-verify branch (#180's chain-verify mask
drop). Under #96 a *sliding-window* layer no longer takes the DCP path -- it
falls through to the plain 2-stage kernel and *does* receive
`forward_metadata.custom_mask`. #180's justification for the drop ("the dxd
draft->draft block IS the causal mask") was reasoned about the DCP paged path
only; whether it holds for the window path with `window_kv_offsets` is
unverified in either direction. Settle with a chain-verify
(`--speculative-eagle-topk 1`) run of an SWA-hybrid model on the uneven-DCP
lane, asserting per-layer what a sliding-window layer receives under
`is_target_verify()`.

---

## 5. What still blocks #96 Stage B, and the routes (design only, not built)

Restated correctly: **the DCP Triton extend lane cannot serve an image request
at all, and sglang's own boot warmup sends one.** Every Gemma-4 server boot
therefore dies in warmup on that lane, before any user request. (a') does not
change this -- the warmup's image span is fully contained, so its mask is
genuinely bidirectional.

Routes, in increasing cost:

1. **Make the boot survive.** The warmup image is an artefact of
   `has_image_understanding`, not of the workload. Either skip the image
   warmup when the DCP extend lane is active (text warmup instead of the PNG),
   or boot with `--skip-server-warmup`. Cheapest, and enough to get H6/H7/H8
   moving; but it only defers the problem to the first real image request.
2. **Refuse image requests on the lane, cleanly.** Turn the
   `NotImplementedError` deep in `_forward_extend_dcp` into an explicit
   boot-time / admission-time refusal with a message naming the reason. Honest
   and diagnosable; text-only DCP works, image requests are rejected instead of
   crashing the forward. This is the pragmatic V1 and pairs with route 1.
3. **Teach the DCP extend path an owner-sharded mask.** The mask's row stride
   is the global prefix length; under owner sharding a rank walks a
   *subsequence* of the prefix, so the stride stops describing the rows the
   kernel walks. Requires either re-striding the mask per rank at metadata
   build (rank knows its owner set) or an indirection from kernel column index
   to global column index. This is the only route that makes image + uneven DCP
   actually work, and it is a #76-class task, not a patch.

Recommendation: 1 + 2 together as the #96 unblock, 3 registered separately.
Not decided here -- this is the design surface, the call is the user's.

---

## 6. Tests

CPU, `PYTHONPATH=/spinning/wt-gemma-mask/python`, venv
`/spinning/htsglang-gpu/.venv`.

New: `test/registered/unit/models/test_gemma4_bidirectional_mask.py`,
6 tests, `register_cpu_ci(est_time=12, suite="base-a-test-cpu")`.

**Falsifier check (the test must bite):** run against the *unfixed*
`gemma4_mm.py` -> **3 failed, 3 passed**; the three failures are exactly the
degeneracy-skip assertions (`test_textonly_mask_is_exactly_the_causal_predicate`,
`test_degenerate_batches_install_nothing`,
`test_single_token_image_span_stays_degenerate`). With the fix -> **6 passed**.
The three that pass in both are the invariant/semantics guards (SDPA byte
identity, mixed-batch slot invariant, contained-image non-causality), which is
what they are for.

Regression sweep, related CPU suites (`test_gemma4_geometry.py`,
`test_swa_pool_sizing.py`, `unit/multimodal/`, `test_model_overrides.py`, the
two source ratchets): **89 passed, 1 failed**. The failure is
`test_dllm_forces_flashinfer_with_cuda_graph` (`_dllm_attention_backend`
override missing from `_resolved_overrides`) -- **pre-existing**, reproduced
identically on the untouched base tree `/spinning/wt-merge-probe` @ `8d87afe941`.
Error sets are identical before and after. Nothing to do with Gemma or masks.

Token-count falsifier for §1a: `AutoProcessor` from
`gemma-4-31B-it-int4-AutoRound`, the literal
`MINIMUM_PNG_PICTURE_BASE64` from `http_server.py:2034`, warmup chat template
-> `(1, 275)` tokens, 256 of them `image_token_id=258880`. Text-only same
prompt -> `(1, 17)`.

---

## 7. GPU recipe (owed -- no boots performed, main rig on #190 kernel isolation)

Resolve physical GPU indices via NVML at runtime first (`r3val/gpu_map.py`);
torch and NVML order diverge on this rig.

1. **Gemma-4 text-only, with and without the fix.** Non-DCP Triton lane,
   `--attention-backend triton`, CUDA graphs + spec ON (full-perf, not eager).
   Short prompts only, **< 109 tokens** (byte-gate policy). Expect
   **byte-identical** token streams between the two trees -- §2 argues this is
   exact, not approximate, so any divergence is a real finding, not noise.
   Record prefill latency delta; for text-only batches the fix should be a
   no-op on latency too (the outer `contains_image_inputs()` guard already
   short-circuits), which is itself a useful control.
2. **The batch that actually changes: multi-turn image conversation.** Turn 1
   with an image, then turn 2..N text-only follow-ups against the cached
   prefix. Before the fix turn 2+ builds and installs a causal mask; after, it
   does not. Assert byte-identical outputs across all turns, and measure the
   prefill-latency delta on turns 2+ -- this is where the win is.
3. **MM regression.** Single image prompt, span fully contained: mask still
   installed, output byte-identical to the base tree. Plus a
   chunked-prefill run (`--chunked-prefill-size` small enough to split the
   256-token span) -- before: causal mask installed; after: none; outputs must
   match.
4. **Mixed-batch guard.** Fire a text-only and an image request concurrently so
   they batch together. This is the §1b invariant under live conditions:
   outputs of *both* must match their solo-run outputs. A regression here would
   be the silent-corruption mode.
5. **Then #96 H4 again** -- expected still RED for the reason in §5 (the warmup
   image is genuinely bidirectional). Re-run with route 1 applied
   (`--skip-server-warmup`) to confirm the lane opens for text-only traffic and
   to unblock H6/H7/H8. If it does *not* open, that is new information.
