# #398 -- native GGUF MXFP4 (ggml type 39): what is left at the desk

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots, no builds,
no window taken.

**Verdict: the desk half is DONE and was done before this task started. The
kernel is built, merged, and installed. Nothing in the plan calls for building
a kernel today. The only genuine desk-fundable remainder was the turnkey
falsifier, and that is what was built here.**

The brief's framing -- "the precondition is NOW fulfilled, determine then build
the desk half" -- reads as though type 39 were still waiting on an enabler
(#442's Marlin adoption, #463's DSpark findings). At code it is not. Those may
be why the ticket was reopened, but they are not what it is blocked on.

## What the tree has TODAY for type 39

`TICKET_398_mxfp4_validation.md` §7 is a continuation note written for exactly
this situation, and it holds up:

| piece | where | state |
|---|---|---|
| `block_mxfp4` struct (E8M0 + 32x fp4-e2m1) | `sgl-kernel/csrc/quantization/gguf/ggml-common.h` | done |
| `dequantize_block_mxfp4` | `.../dequantize.cuh` | done |
| `vec_dot_mxfp4_q8_1`, dense MMVQ | `.../vecdotq.cuh`, `.../mmvq.cuh` | done |
| dense MMQ tile (`MMQ_X/Y_MXFP4 = 4/32`, `NWARPS_MXFP4 = 4`) | `.../mmq.cuh` | done |
| MoE MMVQ / MoE MMQ | `.../moe_vec.cuh`, `.../moe.cuh` | done |
| capability marker `ggml_mxfp4_native` | `.../common_extension.cc` | done |
| python gate `MXFP4_NATIVE` + the three type sets | `layers/quantization/gguf.py` | done |
| repack hand-off (identity on a native wheel) | `model_loader/gguf_mxfp4_repack.py` | done |

The wheel carrying it has been **installed** in the serving venv since
2026-08-03 12:37 (sha `67f03cfa`, recorded in `direct_url.json`), so
`MXFP4_NATIVE` is already True in production and the load-time MXFP4->Q5_0
repack is already a no-op unless a boot sets `SGLANG_GGUF_MXFP4_NATIVE=0`.

So the answer to "which piece is the actual gap -- dequant kernel? MMVQ
variant? loader mapping? per-arch dispatch?" is: **none of them.** There is no
desk gap in the feature. The gap is evidentiary.

### Verified at source, not taken from the ticket

The table above is a documentation claim, so it was checked against the tree
rather than believed. Every piece resolves, introduced by `755cdb1588` and
unmodified since:

- dequant `dequantize.cuh:462`, dispatched `:619-620` (`case 39`); dense MMVQ
  `mmvq.cuh:510`, dot product `vecdotq.cuh:2092`; dense MMQ `mmq.cuh:900`; MoE
  MMVQ `moe_vec.cuh:416`; MoE MMQ `moe.cuh:1450`; block size
  `gguf_kernel.cu:921` (`case 39: return MOE_X_MXFP4`); marker
  `gguf_kernel.cu:119` + `common_extension.cc:479`.
- `block_mxfp4` is `ggml-common.h:207-211` with a
  `static_assert(sizeof(block_mxfp4) == 17)` -- the 17-byte odd stride is
  enforced by the compiler, not by comment.
- **The odd-stride discipline holds.** `qs` is touched in exactly two places
  (`vecdotq.cuh:2104`, `:2146`) and both use the byte-granular `get_int_b1`.
  Grepped `block_mxfp4` against `get_int_b2` / `get_int_from_uint8`: nothing.
  The MoE paths define no loader of their own; they route through the same two
  functions, so there is no second read path to drift. This matters because a
  2-byte-aligned read on a 17-byte stride is one of the ticket's four
  falsifiers (a misaligned-address fault).
- Python: `MXFP4_NATIVE` at `gguf.py:281`, probe at `:276`, lever at `:269`;
  the three type sets gated at `:282-285`, plus a fourth gated mirror
  `_GGML_MOE_MMQ_TYPES` at `:984-986`.

Hermetic test state at HEAD: **86 passed** across the six MXFP4 files plus the
#518 regression test; the only skips are the 14 GPU-gated cases, which skip
cleanly rather than false-passing.

The installed wheel carries the marker (`hasattr` -> True). A direct *call*
raises `NotImplementedError` ("no tensor arguments to this function"), which is
precisely why the probe must test existence and never invoke -- and why #518's
dispatch-key defect does not block MXFP4.

### Catalog corrections made here

Three cited line numbers had drifted, on an entry that already carried a
warning about drift:

| symbol | catalogued | actual |
|---|---|---|
| `SGLANG_GGUF_MXFP4_REPACK` | `environ.py:1897` | **`:1982`** |
| `ggml_mxfp4_native` probe | `gguf.py:272` | **`:276`** |
| `SGLANG_GGUF_MXFP4_NATIVE` first-char test | `gguf.py:265` | **`:269`** |

The catalog's own parenthetical says of the first one: *"Verify a line number
against the tree before quoting it -- this one has drifted twice."* It has now
drifted a third time, and `:1897` today points at a MiniMax-M3 comment. The
note is updated to say so.

Also corrected: the repack module is
`python/sglang/srt/model_loader/gguf_mxfp4_repack.py`. The catalog cites it by
bare filename inside a passage about `layers/quantization/`, which reads as
though it lived there; it does not, and there is no file of that name under
`layers/quantization/`.

## The real remainder, and why it is not a kernel

Fourteen CUDA tests in
`test/registered/unit/quantization/test_gguf_mxfp4_cuda.py` report
`SKIPPED: no CUDA device` off-GPU. They are the entire distance between
"installed" and "proven", and they must pass on **both** arches -- sm86 and
sm120 are separate gates, because sm120 already had an MXFP4 route
(`Mxfp4MarlinMoEMethod`, safetensors) before #398, so an sm120-only green would
not isolate the GGUF path at all.

That is window work. What the desk can do for it is make it turnkey, and that
had not been done: the ticket described the gates in prose and listed four
falsifiers in §6, but there was no script. Grepped `scripts/gpu_battery/` for
`398|mxfp4` -- nothing. The window operator was going to reconstruct the
procedure by hand, and the procedure has three traps in it.

## Built here: `bench/398/run_398_gate_a.py`

Executes ticket sections 0, 1 and 1b end to end, writes `report.json` and a
`TICKET_BLOCK.md` ready to paste into the PENDING blocks. Claims no window,
boots no server. Exit 0 = all green, 1 = a gate failed, **2 = could not run** --
"could not run" is never reported as a pass.

It encodes the three traps, each of which has already bitten this ticket once:

**1. The falsifier's second arm is `True / False / False`, not
`False / False / False`.** The first value is
`hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")` -- a property of which
objects are on disk, i.e. of the WHEEL. `SGLANG_GGUF_MXFP4_NATIVE` is read one
level up, in `gguf.mxfp4_native()`, which returns False before it ever reaches
the hasattr. The lever flips the two DERIVED answers and must leave the marker
alone. A runner that expected all three to flip would go red on a correct build
-- the expectation would be the defect, not the code (#519). The table is
`FALSIFIER_ARMS`, kept as data so it can be read and argued with rather than
buried in an assertion.

**2. `sgl_kernel` must be imported BEFORE the marker is probed.** torch
registers ops when the extension `.so` loads, not when the namespace is
touched. A probe that runs first answers "absent" on a wheel that carries the
op; if anything then defines a fake schema, the real `.so` registers the same
schema twice and the process takes a C++ abort -- not a catchable exception.
This is not hypothetical: it aborted the interpreter under pytest and the gate
looked like it had merely not run.

**3. Physical index != torch ordinal on this rig.** Cards are resolved through
the NVML `IdentityMap` and classified by the compute capability torch actually
reports, never by a hardcoded ordinal or a card name. The self-test exercises a
rig whose CUDA and NVML orders deliberately disagree (5090 = NVML 1 / CUDA 0,
this rig's real shape) and asserts the sm120 gate still lands on the 5090.

### Mock-smoked, and proven able to fail

`--self-test` runs hermetically: 18 checks, 4 of which assert the gates REJECT
bad input. The count is computed, not claimed.

Can-fail proof: reinstating the pre-#519 expectation
(`expected=(False, False, False)` on the lever-off arm) turns the self-test red
on exactly the two relevant checks with exit 1; restored, it is green again.

Pinned by `test/registered/unit/quantization/test_gate_a_runner_398.py`
(5 tests) so the runner cannot rot between now and its window -- including two
that pin the corrected #519 expectation directly, so a future edit cannot
quietly restore the broken one.

## Not built, and why

**No kernel work.** The plan does not call for any: §7 says both stages are
merged on `integration/r3-probe-next2` (merge `08bde23da7`) and explicitly
"do not rebuild them". Building would have been duplicate work against an
installed wheel.

**No rebuild of the #512/#518 bundle.** §7 already determined this is off the
critical path for #398 (#512 needs a >2 GiB per-rank per-layer expert tensor,
out of reach at TP=3; #518 is worked around by the `_ggml_moe_get_block_size`
python mirror) and that its preconditions are not met while the DSV4F boot is
live.

**No window taken.** `/spinning/gpu-arb/holder` reads `706-retry` with a live
heartbeat, so the cards belong to another strand. The runner is written for
whoever holds the next window and deliberately does not claim one for itself --
claiming canon lives in `evidence-qwen38/claim_window.sh` (heartbeat staleness
AND no `launch_server` mid-startup) and duplicating it here would be a second
copy to drift.

## Acceptance criteria for the window

Gate A is green when, on **both** arches:

- 14/14 pass in `test_gguf_mxfp4_cuda.py`, summary line recorded verbatim
  alongside `nvidia-smi --query-gpu=name,uuid` for the card that ran it;
- the dequant tests pass **exactly**, zero tolerance -- the kernel is a table
  lookup and one multiply by a power of two, and `ggml_cuda_e8m0_to_fp32_half`
  is bit-identical to the host reference, so any delta is a bug, not rounding;
- `TestMXFP4MMQ::test_agrees_with_mmvq` holds **relmax < 1e-3** (the meaningful
  cross-check; agreeing with fp32 but not with each other means the MMQ tile's
  scale handling is wrong).

Gate A' is green when the native arm reads `True/True/True` and the lever-off
arm reads `True/False/False`.

Anything short of that is `NOT GREEN`, and the runner says so rather than
summarising optimistically.

## Sequence after the window

Unchanged from §7: Gate A, then Gate B (native vs repack on the two live
IQ3_XXS type-39 tensors, `blk.26`/`blk.42.ffn_down_exps`, worth 0.625 GiB),
then Gates C-E only once A and B are green.
