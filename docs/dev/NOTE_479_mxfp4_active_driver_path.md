# NOTE #479 — what the active DSV4-Flash driver does with its MXFP4 tensors

Desk trace, no GPU window. Everything below is either read out of the tree at
the line cited or executed hermetically under `CUDA_VISIBLE_DEVICES=99`.

## 0 — The premise the ticket was opened on is refuted

#479 was raised as "two MXFP4 tensors at zero MXFP4 kernel coverage, because
#398 is unbuilt". Both halves of that are false on this line:

* **Source**: the kernel set is in the tree —
  `sgl-kernel/csrc/quantization/gguf/` carries the MXFP4 cases in
  `dequantize.cuh`, `mmvq.cuh`, `mmq.cuh`, `moe.cuh`, `moe_vec.cuh`, and the
  python side admits ggml type 39 to all three type sets at
  `layers/quantization/gguf.py:282-285`.
* **Installed wheel**: executed probe, this venv
  (`/spinning/htsglang-gpu/.venv`) —
  `hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")` is **True**, so
  `MXFP4_NATIVE` (`gguf.py:281`) is True at import and the type is native on
  the wheel the rig actually boots.

So the question "what carries these two tensors" has an answer that is neither
a fallback nor an untraced path: **the #398 kernels carry them directly**, and
on a pre-#398 wheel the lossless Q5_0 repack carries them. There is no third
route.

## 1 — The checkpoint fact, read off disk

`DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS`, 4 shards, tensor-type histogram:

```
Q6_K 170, F32 662, Q8_0 321, IQ3_XXS 75, IQ2_XS 50, BF16 43, I32 3,
MXFP4 2, IQ3_S 2
```

The two type-39 tensors, and — the part the ticket did not state — their
siblings:

| layer | ffn_gate_exps | ffn_up_exps | ffn_down_exps |
|---|---|---|---|
| 26 | IQ3_S (21) | IQ3_S (21) | **MXFP4 (39)** |
| 42 | IQ3_XXS (18) | IQ3_XXS (18) | **MXFP4 (39)** |

Each is `[2048, 4096, 256]` = 1 140 850 688 B, 2.125 GiB together. Gate and up
are stacked into one `w13_qweight`, so layers 26 and 42 are the only place in
this checkpoint where `fused_moe_gguf` receives a **mixed type pair**
(`qweight_type` != `qweight_type2`). That is the cell #479 actually had to
check, and nothing tested it before.

## 2 — The load path: no host dequant, no dtype change

`gguf_quant_weights_iterator` is the single door
(`model_loader/weight_utils.py:1355`):

* pass 1, `:1419` — `repacked_gguf_type(tensor.tensor_type, name)` is what the
  consumer records as `qweight_type`;
* pass 2, `:1503-1505` — `repacked_gguf_bytes(source_type, weight[expert_id],
  name)` per expert (the stacked tensor is split first), `:1520` for dense.

With the kernels native, `gguf_mxfp4_repack._type_map()` returns `{}` before
any tensor is read (`gguf_mxfp4_repack.py:113-115`), so both functions are the
identity (`:205-207`, `:224-226`): same numpy view, same `uint8` dtype, 17
bytes per 32 values. **There is no host dequantization of these tensors at any
dtype**, and therefore nothing beyond the block bytes lands in RAM or VRAM —
which answers the open question TICKET_398 §2 parked ("if #479 finds a
load-time dequant, the VRAM delta is larger than 0.625 GiB": it does not, so
the delta is exactly the repack's 0.625 GiB).

On a pre-#398 wheel the same two calls return Q5_0 blocks, 22 bytes per 32
values, still `uint8` — value-exact, `+29.4 %` bytes.

Parameter creation is untouched by the type: `GGUFUninitializedParameter`
(`gguf.py:1433`, `:1459`) holds the raw blocks and the ggml type rides beside
it on `w13_qweight_type` / `w2_qweight_type` (`:1447-1455`, `:1473-1482`).

## 3 — The forward path: which kernel actually runs

`GGUFMoEMethod.apply` (`gguf.py:1500`) passes both types into `fused_moe_gguf`
(`:1519-1527`), which has exactly three exits:

1. `:1017-1021` MMQ MoE — requires **both** types in `MMQ_QUANT_TYPES`. Layer
   26/42's w13 is an imatrix type, which has no MMQ kernel, so this branch is
   **unreachable for these two layers at any batch size**, native wheel or not.
2. `:1093` MMVQ MoE — requires both types in `MMVQ_QUANT_TYPES`. IQ3_S/IQ3_XXS
   are imatrix types (in the set); MXFP4 is in the set iff `MXFP4_NATIVE`.
   **This is the branch both arms take**: native → `ggml_moe_a8_vec(..., 39,
   ...)` for the down side (`:1101-1103`); repack arm → the same call with
   type 6.
3. `:1108-1131` the slow per-expert loop, entered only when a type is in
   neither set. It is not a silent path: `fused_mul_mat_gguf` refuses an
   unknown type by name (`:958-963`, `NotImplementedError`).

The only combination that could put an unrepacked type 39 in front of a
non-native build is `SGLANG_GGUF_MXFP4_NATIVE=0` together with a payload that
was never repacked, and that combination raises out of exit 3. Verdict:
**correct on every reachable arm; loud, never silent, on the unreachable one.**

Expert offload does not change the answer. Its admission door checks **both**
expert tensors, not just w13 (`layers/moe/fused_moe_triton/layer.py:2520-2534`
over `("w13_qweight_type", "w2_qweight_type")`), against
`MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES` (`gguf.py:296`), so a mixed
layer is admitted exactly when both halves have an MMVQ kernel and is declined
whole — the half-staged state `_finish_gguf_moe_offload_staging` exists to
refuse (`layer.py:2550-2556`) is not reachable through the type check.

## 4 — Verdict

Correct-but-priced, not silently wrong. The price on a pre-#398 wheel is
0.625 GiB of extra weight bytes and 29.4 % more read traffic on the two down
projections; on the wheel installed here that price is already zero. No fix is
owed; the path is now pinned instead.

## 5 — What is pinned, and how it was falsified

`test/registered/unit/quantization/test_gguf_mxfp4_dsv4f_moe_479.py`, 8 tests,
hermetic. One reads the shipped GGUF headers (skips when the export is absent)
so the type pairs above are a probe, not a restated claim; the rest drive
`fused_moe_gguf` with kernel recorders and assert the branch, the ggml type
handed to the kernel, and the uint8/17-vs-22-byte load payload.

Three executed can-fail arms (mutation, run, revert):

| mutation | result |
|---|---|
| `:1093` MMVQ predicate drops its `qweight_type2` half | 1 red |
| `:284` type 39 no longer joins `MMVQ_QUANT_TYPES` | 2 red |
| `gguf_mxfp4_repack.py:113` native short-circuit removed | 1 red |

Restored tree: 8 passed.

## 6 — Measurement ticket recipe (not this ticket's work)

The only number #479 could still owe is the A/B between the native arm and the
repack arm, and TICKET_398 §2/§5 already own it — this note supplies the two
inputs that ticket was missing: the delta is **exactly** 0.625 GiB (no
load-time dequant on top), and the affected kernel call is the MoE MMVQ down
projection of layers 26 and 42 only (2 of 43 MoE layers), so a whole-model
ms/verify effect below the A-vs-A floor is the expected outcome, not a null
result. Boot recipe: the standard DSV4-GGUF TP=3 recipe with `--enable-metrics`,
once with `SGLANG_GGUF_MXFP4_NATIVE=0` and once without, A-vs-A floor per arm
first (catalog §10 canon), report ms/verify and ms/prefill per rank.
