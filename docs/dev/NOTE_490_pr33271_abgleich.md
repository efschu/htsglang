# NOTE 490 — upstream PR #33271, the 8 commits of 2026-08-03, against our line

Desk-only analysis (`CUDA_VISIBLE_DEVICES` untouched, no GPU, no
`/spinning/gpu-arb` hold, no build). Read against
`origin/integration/r3-probe-next2` at **`08bde23da79e4a7ab799e572b7f37f9813da9e57`**
("Merge #398: native GGUF MXFP4 (ggml type 39) kernels for sgl-kernel"), fetched
2026-08-03; a merge agent is pushing further merges onto the same line, so any
line number below is against that hash.

Upstream side read from the public commit diffs
(`api.github.com/repos/sgl-project/sglang/commits/<sha>`, `Accept:
vnd.github.diff`), PR branch `Hakureirm/sglang`, head `f0f0ce4bc1`. Raw diffs
archived at `/root/.claude/jobs/1481bb40/tmp/diff_<sha10>.diff`.

Catalog sections read: `docs/dev/FEATURE_CATALOG.md` §8 (GGUF stack), §9 (quant
lanes), §15 (model bring-ups / DSV4-Flash indexer), §17 (combination matrix).
Also read: `NOTE_440_c4_indexer_head_fold.md`, `NOTE_449_dsv4_indexer_query_chunk.md`,
`NOTE_473_rebase_gates.md`, `PLAN_417_dsv4_arch_paths.md`,
`ANALYSE_447_llamacpp_dsv4_harvest.md`, `ANALYSE_463_dspark_formats.md`,
`TICKET_470_dspark_boots.md`, `python/sglang/srt/planner/rejected.py`.

---

## D (asked first because it was flagged PRIO) — the raw C4 head fold

**Verdict: NO PRIO. The fold was never adopted here. It is not gated, it is
absent — refused by name, with a registered rejection and a can-fail test.**

The task briefing calls the fold "unsere #440-Adoption". That premise is wrong:
#440 was a *refusal* task, not an adoption task. Verified three ways, at code
level, not from the note's own claim:

* `grep -rniE "folded_paged|q_eff|bhd,bh->bd|TRITON_FOLDED|FOLDED_INDEXER"` over
  `python/` returns **zero hits in any DSV4 file**. The only `q_eff` in the tree
  is an unrelated local in
  `python/sglang/srt/layers/attention/fla/fused_recurrent_linear_replayssm.py:254,294`
  (GDN recurrent kernel).
* The only fold in the tree is `_folded_reference` at
  `test/registered/unit/layers/attention/test_dsv4_indexer_head_fold_440.py:178-189`
  — a test-local reference that exists precisely to *prove the fold wrong*.
* Both production torch paths still carry the per-head ReLU that makes the fold
  invalid: `dsv4/indexer.py:482` (`F.relu(score)` in
  `fp8_paged_mqa_logits_torch_sm120`, the serving path on every card of this rig)
  and `dsv4/indexer.py:178` (`F.relu(scores)` in the reference twin).

The refusal is registered, not merely documented:
`python/sglang/srt/planner/rejected.py:570-607`, key `c4_indexer_head_fold`,
`level=BLOCKED`, verdict "WRONG OPERATOR", with the 0.94 relative divergence and
0.41-0.56 top-k overlap numbers and an explicit reopen condition ("only if the
GPU arm in NOTE_440 shows the ReLU is inert on the real checkpoint, and then
only as a labelled approximation, never as an identity").

The upstream thread's refutation on *trained* weights (top-512 overlap ~0.75,
`frac_neg` ~0.5) therefore does not create an exposure for us — it **closes the
one open question NOTE_440 left**. `NOTE_440:149-172` had stated the two
remaining explanations honestly and made explanation (2) ("the ReLU is nearly
inert on the trained checkpoint") the sole reopen path, with the decision rule
`frac_neg < 0.02` and overlap `>= 0.99` fixed in advance
(`NOTE_440:329-333`). The thread's `frac_neg ~ 0.5` and overlap ~0.75 miss that
rule by a wide margin on both terms. Consequence, and this is the one action D
produces: **the GPU measurement arm in `NOTE_440` §"GPU measurement arm" is now
answerable from the archived third-party data and no longer needs a window.**
Upstream reached the same conclusion in code — commit **`e2bcb585f151`**
("Revert 'Fold the indexer head dimension out before the GEMM on Ampere'")
deletes `_folded_paged_logits` and `_triton_folded_paged_mqa_logits` from their
branch entirely.

**Two-stage / superset-c variant**: absent here as well. `grep -rniE
"superset|two.stage|two_stage"` over `python/sglang/srt/layers/attention/`
returns one unrelated hit (`linear/short_conv_backend.py:22`, ZAYA1 CCA). We
carry no candidate-generation stage of any kind on the indexer path, so the
still-open upstream two-stage idea (miss @c=8 <= 0.008) is a *new feature*
question for us, not a live code path — and it must not be conflated with the
raw fold's BLOCKED entry if it is ever picked up.

---

## A — indexer chunking hardening (`97240add49`, `c0c645a7b0`, `e2d5079a74`, `2e1ef6af3c`)

**Verdict: all four defects are structurally absent here. Upstream is walking
back toward the shape #426+#449 already has. Nothing to adopt.**

Upstream's four commits are one bug walked in four steps: their `_QUERY_CHUNK =
1024` chunking (`af7b3cc60104`) was a *recursive self-call* that (i) mis-sized
the chunk budget, (ii) `torch.cat`'d the sub-outputs, (iii) allocated a fresh
full-size output per sub-call, and (iv) left the pre-chunking full gather in
place. Our #449 chunking is a flat double loop over a single preallocated
output, so none of the four applies.

### A1 — does our chunking budget the fp32 weighted-score product?

**Yes, and more conservatively than upstream's fix.**

* Upstream after `97240add49` (`indexer.py:244-253` upstream): `_per = max(1,
  batch_size * (head_dim * 2 + num_heads * 6))` — bf16 values (`head_dim*2`),
  bf16 scores (`num_heads*2`), fp32 product (`num_heads*4`).
* Ours: `_indexer_logits_step_bytes` at
  `python/sglang/srt/layers/attention/dsv4/indexer.py:277-297`, the term at
  `:296`:
  `per_position = (head_dim + 4) + head_dim * 4 + 4 + num_heads * 4 * 2`.
  The `num_heads * 4 * 2` term is exactly the missing product — the `[rows,
  chunk, heads]` `bmm` result **plus** one elementwise temporary of it (the
  `score * weight_row` result at `:483`) — and is documented as such at
  `:288-290`. The remaining terms are larger than upstream's because our path
  accumulates in **fp32**, not bf16 (`:428` `q_fp8[:,0].to(torch.float32)`,
  `:474` `.to(torch.float32)`), so `head_dim * 4` for the dequantised KV is the
  correct, larger figure for our arithmetic and would be an undercount if
  upstream's formula were ported verbatim.

Do **not** port `97240add49`: its constants are for a bf16 path we do not run.

### A2 — preallocated output, or `torch.cat`?

**Preallocated, once, outside both loops — we never had a `cat`.**

`indexer.py:432`: `logits = q.new_full((batch_size, max_seq_len), float("-inf"))`,
allocated before the query loop; every chunk writes in place at `:491-493`
(`logits[row_start:row_stop, seq_start:seq_start+out_width] = score[:, :out_width]`).
There is no recursion and therefore no per-sub-call output: the peak transient
is one `[rows, chunk_seq, heads]` product, which is exactly what the budget at
`:443-448` sizes.

Upstream's `c0c645a7b0` (cat -> preallocated) and `e2d5079a74` (add an `out=`
parameter so the sub-call stops allocating) are two repair steps toward a shape
we already have; `e2d5079a74`'s `out=` plumbing exists only because their
chunking is recursive. **Upstream is behind us on this point.**

One structural difference worth recording, in upstream's favour on *one* axis:
their chunking is unconditional above `B > 1024` and needs no env, ours is
governed by `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` (default 2048 MiB,
`environ.py`; 0 disables). Ours is the #395-discipline knob and is deliberate
(NOTE_449 §2 "Why MiB and not a row count"); noted so the difference is not
mistaken for a gap.

### A3 — is there a dead full-context gather left in our tree?

**No, not on the serving path. The one remaining full gather is the reference
twin's, is reachable only from tests, and is intentional.**

* Production (`fp8_paged_mqa_logits_torch_sm120`): the only gather is
  `_gather_pages(kvcache_flat, page_ids_rows[:, page_start:page_stop])` at
  `indexer.py:462`, **inside** both loops. Line `:426` is a slice
  (`page_ids = page_table[:, :max_pages]`), not a gather — no memory traffic,
  no materialisation. This is precisely the leftover upstream deleted in
  `2e1ef6af3c` (their `kvcache_gathered = kvcache_flat[page_ids]` plus
  `kv_value_raw` / `kv_scale_raw`), and it never existed in our rewrite.
* `indexer.py:165` (`kvcache_gathered = kvcache_flat[pages_clamped]`) is in the
  reference twin `fp8_paged_mqa_logits_torch`, which is documented
  **NOT reachable from the serving path** at `:139-143` and confirmed by
  `select_paged_mqa_logits_fn`, which returns `fp8_paged_mqa_logits_torch_sm120`
  for `BACKEND_TORCH` unconditionally (`:102-113`). Its only callers are
  `test/registered/kernels/test_sm120_paged_mqa_logits.py:187,245` and
  `test_dsv4_indexer_head_fold_440.py:309`. Single-pass on both axes is what
  makes it usable as the byte oracle (NOTE_449:113-114) — removing its gather
  would destroy the oracle, so this is not a cleanup candidate.

---

## B — non-paged arch guard (`8a930437f1`)

**Verdict: we have had this since #417 Cut 3, in a strictly better form.
Upstream has now converged onto our capability set. Nothing to adopt; one
observation worth a future upstream comment.**

Upstream adds a module-level constant plus one early return:

```python
_DEEP_GEMM_INDEXER_CAPABILITIES = (9, 10)
_has_deep_gemm_indexer = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] in _DEEP_GEMM_INDEXER_CAPABILITIES
)
...
    if not _has_deep_gemm_indexer:
        return False       # inside _can_use_nonpaged_indexer
```

Ours: `python/sglang/srt/layers/attention/dsv4/indexer_arch.py:51`
(`_DEEPGEMM_MAJORS = (9, 10)`) and `:62-71` (`@per_device_gate
deepgemm_indexer_supported(device_id)`), consumed at
`python/sglang/srt/layers/attention/dsv4/indexer.py:731` (`if not
deepgemm_indexer_supported(): return False`) — placed **before** the env clause
at `:743-749`, which is the property `TestNonPagedCouplingIsAlreadyArchGuarded`
pins.

Two substantive differences, both ours:

1. **Capability set.** `NOTE_473:49-64` recorded that upstream's *other* PR
   (#33288) used `capability[0] >= 9`, unbounded above, which routes a 5090
   (major 12) into `deep_gemm.fp8_mqa_logits` and lets it die on the kernel's
   internal assertion. `8a930437f1` changes that to the finite tuple `(9, 10)`
   — i.e. **upstream has now adopted our `_DEEPGEMM_MAJORS` semantics**. The
   `NOTE_473` (a) gap is closed on their side; no port needed on ours.
2. **When and for which device it is asked.** Upstream's flag is a *module
   import-time* global with **no device argument**, so it freezes the capability
   of whatever device is current when `dsv4/indexer.py` is first imported.
   Ours is `@per_device_gate`-cached and asked per `device_id` at call time,
   after `set_device` (#343: one process can hold parts of one model on two
   cards of different architecture). On a heterogeneous rig upstream's constant
   is wrong for at least one rank whenever import precedes `set_device`. This is
   our uneven/hetero delta and is the one thing in the whole 8-commit batch that
   would be worth raising upstream — **not now** (Posting-Politik: #490 is an
   analysis task, the only open channel to that PR is the `|w|` arm under #474).

Upstream's guard also covers `torch.cuda.is_available()`; ours covers the same
ground with `is_cuda()`/`is_hip()` at `indexer.py:724-725`, ahead of the arch
check, plus a non-CUDA passthrough at `indexer_arch.py:69-70`.

---

## C — the DSpark trio

Two of the three are **real, live defects in our tree**, both fixable by
adopting upstream's one-line shape. The third does not apply.

**Status: C1 and C2 are FIXED on `fix/dspark-probe-scale-491` (#491)**, each
with the falsifier arm run in both directions; C3's production half stays
N/A and its test half came along with C2. The rest of this section is the
analysis as written before the fix.

### C1 — `728a27fa81` suffix-anchored `.scale` rename: **WE CARRY THE BUG**

`python/sglang/srt/models/deepseek_v4_dspark.py:905`:

```python
mapped_rest = mapped_rest.replace(".scale", ".weight_scale_inv")
```

byte-identical to upstream's pre-fix line. Executed against packed-checkpoint
names, this is what it produces:

| checkpoint tensor (`mtp.<stage>.` relative) | our mapped name | in `params_dict`? |
|---|---|---|
| `attn.wo_b.scales` (gptq/awq/auto_round) | `self_attn.wo_b.weight_scale_invs` | **no** |
| `ffn.w1.scales` | `mlp.gate_proj.weight_scale_invs` | **no** |
| `ffn.w1.scale_inv` | `mlp.gate_proj.weight_scale_inv_inv` | **no** |
| `attn.wo_b.scale` (fp8, the intended case) | `self_attn.wo_b.weight_scale_inv` | yes |
| `ffn.w2.qweight`, `ffn.w3.qzeros` | unchanged | yes |

An unmatched name does not raise: `load_weights` logs
`"DSpark V4 draft: unexpected weight %r -> %r"` and `continue`s
(`deepseek_v4_dspark.py:842-846`). So a packed DSpark draft loads its `qweight`
tensors, silently drops every scale, and produces an accept rate pinned at zero
with a warning as the only signal — upstream's own description of the failure.
`_assert_confidence_head_loaded` (`:858-872`) does not catch it: it only checks
`confidence_head.*`.

This is **not a hypothetical configuration for us.** `ANALYSE_463_dspark_formats.md:453`
already names this exact line as a prerequisite for option **R3** ("GPTQ-INT4
requant of the experts plus `split` placement... GPTQ name/shape handling in
`deepseek_v4_dspark.py` (whose `.scale -> .weight_scale_inv` rewrite at `:889`
is fp8-specific)"), and R3 is the named next step in `TICKET_470`'s GATE if
Boot A refutes R1. `ANALYSE_463:296` records that GPTQ-INT4 marlin is one of
only two quant routes that run on all three cards of this rig.

This is the same failure **family** as our #113 GGUF draft-MTP namespace bug
(`tp5-emulation-uneven-gguf-bugs`): a name-mapping rule written for one format,
applied unanchored to a second format's namespace, failing by silent drop rather
than by exception. Anchoring at the suffix is the right general rule and the
same rule the target model's loader already uses.

**Recommendation: adopt, small effort, high value.** Upstream's shape is a
static `_remap_mtp_rest` plus `if rest.endswith(".scale"): rest = rest[:-6] +
".weight_scale_inv"`, with a 4-case hermetic CPU test
(`test/registered/unit/models/test_dspark_weight_name_remap.py`, 40 lines, no
GPU, no weights). Taking the extraction into a `@staticmethod` as well is what
makes the test possible without constructing the model. Our fix should go one
step further than upstream's and add a **can-fail arm** plus a loud failure for
the drop case, because "warning only" is what made this silent in the first
place — but the loud-failure question is a design call for the fix task, not
this note. Do **not** bundle it with C2 in one commit; they are different files
and different failure modes.

### C2 — `383a3a6af2` support probe must answer, not raise: **WE CARRY THE BUG**

Our copy lives at a different path than upstream's — theirs is
`python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py`, ours is
`python/sglang/srt/speculative/dspark_components/kernels/dspark_draft_model.py`
— but the function is identical, at `:296-309`:

```python
def _dequant_supported(linear: torch.nn.Module) -> bool:
    """Mirrors the preconditions asserted in _dequant_linear_weight."""
    weight = linear.weight                    # :298  AttributeError on packed
    ...
    return tuple(linear.weight_scale_inv.shape) == expected_scale_shape   # :309
```

Reachability, traced: `CommitKvProj.execute` at `:210` calls
`_fused_commit_kv_proj_supported(...)` on every CUDA forward;
`_fused_commit_kv_proj_supported` at `:312-317` first tries
`_block_quant_stack_applies` (`:283-293`, which short-circuits safely on
`hasattr(quant_method, "block_quant")` before touching `linear.weight`) and then
falls through to `all(_dequant_supported(linear) for linear in wkv_linears)`.
For a marlin/AWQ/GPTQ `ReplicatedLinear` the module exposes `qweight` and has no
`weight`, so line `:298` raises `AttributeError` — inside a *support probe*,
i.e. in the branch whose entire contract is to route unsupported schemes to the
per-linear torch fallback at `:212`/`:238`. Upstream's symptom: scheduler killed
on the first speculative prefill. Same here.

Second, narrower hole on the same function: `:309` reads
`linear.weight_scale_inv` unguarded, so an fp8 linear whose scale is absent
raises instead of answering `False`. Upstream's fix covers both with `getattr(...,
None)`.

Same live relevance as C1: `TICKET_470` §4 and `ANALYSE_463:482` both drive
`--speculative-moe-runner-backend marlin` for the draft, and `TICKET_470` §7
item 4 lists "`--speculative-moe-runner-backend marlin` actually selecting
`Mxfp4MarlinMoEMethod` for the draft's routed experts" as BOOT-PENDING — a
window that would hit this probe. This is a **pre-boot blocker for #470 Boot B/C
on any packed draft**, and it is desk-fixable.

**Recommendation: adopt, trivial effort.** Two `getattr` guards, plus upstream's
hermetic test file adapted to our import path
(`sglang.srt.speculative.dspark_components.kernels.dspark_draft_model`). Their
`_FakeLinear` fixture — a bare module whose attributes exist only if handed in —
is the right shape and needs no GPU. Note the file-path divergence for the
rebase gate below.

### C3 — `f0f0ce4bc1` runner/bcg import cycle: **DOES NOT APPLY**

The cycle is between upstream's `sglang.srt.layers.cp.bcg` and
`model_executor/runner/prefill_cuda_graph_runner.py`. We have no `layers/cp/bcg`
module at all (`find python -path "*cp/bcg*"` empty; our `python/sglang/srt/layers/cp/`
holds `base.py`, `interleave.py`, `utils.py`, `zigzag.py`), and our
`prefill_cuda_graph_runner.py` imports nothing from `layers.cp` — its
breakable-graph edges go to
`model_executor/runner_backend/breakable_cuda_graph_backend.py:75-76` and
`model_executor/runner_backend_utils/breakable_cuda_graph/context.py:87`. Their
`PrefillCPBCGInput` / `execute_prefill_cp_bcg` seam is upstream-only DSA
prefill-CP machinery.

The *test* half of that commit (`test/registered/unit/spec/test_dspark_dequant_probe.py`)
is the C2 test and is worth taking; the production half is not portable.

Note also that upstream's fix comment is written in Chinese, which would violate
`Englisch-Code-Standard` if carried over — irrelevant here since we take none of
that hunk.

---

## E — what this batch changes for the `NOTE_473` rebase gates

Four of the eight commits touch `dsv4/indexer.py`, which is the single most
diverged file between the two trees. Adding to `NOTE_473` §(b):

6. **`dsv4/indexer.py` is now a guaranteed conflict at any rebase past this PR,
   in the same region we own.** Their `fp8_paged_mqa_logits_torch_sm120` after
   `e2d5079a74` has a different signature (`out: Optional[torch.Tensor] = None`),
   a recursive `_QUERY_CHUNK = 1024` self-call, and a bf16 accumulation; ours has
   a flat double loop, an fp32 accumulation, an env-driven MiB budget, and no
   `out=`. Resolution rule, fixed in advance so it is not re-litigated at rebase
   time: **keep ours whole**; the four upstream commits are repairs of a defect
   class ours does not have (A1-A3 above), and taking any of them piecemeal
   — especially the `head_dim * 2` bf16 byte formula — would *undercount* our
   fp32 path's transient and reintroduce the OOM they were fixing.
7. **The upstream arch guard now lands inside `_can_use_nonpaged_indexer`
   itself** (`8a930437f1`), i.e. in the same function our `#417 Cut 3` guard
   already occupies (`indexer.py:731`). Expect a conflict hunk there; resolution
   is to keep our `deepgemm_indexer_supported(device_id)` call and drop their
   `_has_deep_gemm_indexer` module global, which is device-blind (see B2). Do
   **not** take both — two guards with different device semantics in one
   function is exactly the "two decisions that must not be able to disagree"
   trap `indexer_arch.py:98-101` names.
8. **The DSpark fused-KV-projection module has MOVED upstream**, from our
   `python/sglang/srt/speculative/dspark_components/kernels/dspark_draft_model.py`
   to their `python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py`.
   Same family as gate 3 (`flash_mla_sm120.py`): any of our patches to this file
   become a *port*, not a cherry-pick, once we rebase past the move. This
   includes the C2 fix if we take it before rebasing — which we should, since C2
   is a pre-boot blocker and the rebase is not scheduled.
9. `models/deepseek_v4_dspark.py` gains a `_remap_mtp_rest` staticmethod
   (`728a27fa81`). We patch this file; if we adopt C1 with upstream's exact
   method name and signature the rebase is a no-op merge, which is a reason to
   match their shape rather than invent our own.

Nothing in the batch touches `deepseek_v2.py`, `deepseek_nextn.py`,
`kv_cache_configurator.py` or the ServerArgs stack, so gates 1, 2, 4 and 5 of
`NOTE_473` §(b) are unaffected.

---

## Summary table

| item | commit(s) | our state | action |
|---|---|---|---|
| D — raw C4 head fold | (`2ce48ba` + revert `e2bcb58`) | **never adopted**; `rejected.py:570-607` `level=BLOCKED` | none. Close `NOTE_440`'s GPU arm from the archived third-party data — the reopen rule is missed by both terms |
| A1 fp32 product in the budget | `97240add49` | `indexer.py:296` already counts it (`num_heads*4*2`), for an fp32 path | none; do NOT port the bf16 constants |
| A2 preallocated output | `c0c645a7b0`, `e2d5079a74` | `indexer.py:432` + in-place writes at `:491`; never used `cat`, no recursion | none; we are ahead |
| A3 dead full-context gather | `2e1ef6af3c` | absent on the serving path; the twin's gather at `:165` is the test oracle | none |
| B non-paged arch guard | `8a930437f1` | `indexer.py:731` + `indexer_arch.py:51,62-71`, per-device, ahead of the env clause | none; upstream converged onto our `(9,10)`. Their import-time global is device-blind — our delta, do not port |
| C1 `.scale` suffix anchor | `728a27fa81` | **BUG PRESENT**, `deepseek_v4_dspark.py:905` | **adopt.** Small. Blocks `ANALYSE_463` R3 / any packed DSpark draft |
| C2 probe answers, not raises | `383a3a6af2` | **BUG PRESENT**, `dspark_components/kernels/dspark_draft_model.py:298,309` | **adopt.** Trivial. Pre-boot blocker for `TICKET_470` Boot B/C with marlin |
| C3 runner/bcg import cycle | `f0f0ce4bc1` | module does not exist here | production half N/A; take the test half with C2 |
