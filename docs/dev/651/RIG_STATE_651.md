# #651 on the rig: what is actually true, and what is left

Scope: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` on the 3-card CUDA rig
(2x RTX 3080 20 GB + 1x RTX 5090 32 GB), with PP/TP and speculation.

`docs/dev/651/FINAL_651.md` is **not** about this machine. It is 1571 lines
about the gfx1103 APU laptop and mentions the rig twice, both times as a
contrast. Rig state had to be reconstructed from commit messages and the small
companion scripts; this file is that reconstruction, plus two corrections
measured here on 2026-08-24.

---

## 1. The 5090 "19.58 GiB CUDA-visible" ceiling — REFUTED

`5598241c65` recorded rig window 1 as *no verdict*, because
`torch.cuda.mem_get_info` reported the 5090 as **19.58 GiB** CUDA-visible
against the 32,607 MiB `nvidia-smi` shows. It attributed the gap to "the rig's
barlink/BAR1-pinned environment", never root-caused it, and worked around it:
a desk-time capacity gate in `rig_window_probe.sh`, and a switch of the TP=1
discriminator from Q4_K_XL (21.8 GiB) to Q3_K_XL (16.04 GiB). The conclusion
drawn was "Q4_K_XL cannot run TP=1 on the 5090, full stop".

Measured here, bare (no barlink env), each card resolved **by NVML UUID** and
probed in its own process with `CUDA_VISIBLE_DEVICES` set to that UUID:

| idx | card      | NVML total | NVML free | torch total | torch free |
|-----|-----------|-----------:|----------:|------------:|-----------:|
| 0   | RTX 3080  |      20480 |     20054 |       20054 |      19830 |
| 1   | RTX 5090  |      32607 |     32088 |   **32088** |      31582 |
| 2   | RTX 3080  |      20480 |     20054 |       20054 |      19830 |

The 5090 exposes **32,088 MiB** to CUDA. There is no ceiling.

The likely cause of the original reading is arithmetic, not environmental:

    19.58 GiB x 1024 = 20,050 MiB  ~=  20,054 MiB = a 3080's torch-visible total

19.58 GiB is, to within rounding, **exactly what a 3080 reports**. The
overwhelmingly probable reading is that window 1 probed a 3080 while believing
it held the 5090 -- the index-vs-UUID divergence that `docs/dev/651/boot.sh`
warns about in its own device-resolution comment ("torch's enumeration and
NVML's diverge, and NVML order itself can shift across boots/driver states").
An index-addressed probe on this rig hits a 3080 two times in three.

Consequences:
- Q4_K_XL (21.8 GiB) **fits TP=1 on the 5090** with ~9.5 GiB to spare. Stage a
  is unblocked.
- The Q3_K_XL substitution was unnecessary; the Aug-8 CUDA coherence proof
  (`rig_cuda_probe_120135.txt`) is valid but was run on the wrong file for the
  mission's purposes.
- The `model_size + 1.5 GiB < CUDA-visible total` desk gate is still a good
  gate. Its input was wrong, not its logic.

This is the INDIKATOR-GESETZ case in its pure form: the number was never tested
against a known state, so a misaddressed device became a hardware verdict that
redirected two windows.

## 2. The int32 `topk_ids` cast is ROCm-only

`b7a46481c3` adds a defensive `topk_ids.to(torch.int32)` at the `fused_moe_gguf`
boundary. Root-cause commit `ed26aacb8d` states it plainly: the in-tree
`sgl_kernel` AOT op **enforces** int32, "which is why identical python code was
always coherent on CUDA". The defect is in the standalone ROCm binding, which
forwards the tensor unchecked. Nothing about it was ever a CUDA-observed
defect, and the cast is already present on the current tree at
`python/sglang/srt/layers/quantization/gguf.py:1924`.

Do not re-port it, and do not cite it as a rig fix.

## 3. What already ran on this rig

**2026-07-18, commit `dff1ef16c0`** -- the target checkpoint, on this rig, with
speculation and graphs:

- `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- TP=3 uneven, `--rank-gpu-id 0,1,2 --rank-tp-ratio auto`, DCP=3
- flashinfer attention, **CUDA graphs ON**, **NEXTN/MTP speculation**
- coherent temp-0 battery, 15k-token needle HIT, decode ~89-100 tok/s

Three GGUF-MoE-specific defects had to be fixed to reach it, all now upstream
of the current tree: `8290c8a490` (uneven-TP GDN in_proj head coarsening),
`b97e60b25e` (GGUF-MoE MMQ prefill path, two stale sgl_kernel APIs),
`1d689439f6` (OOB expert-id read under uneven-TP expert sharding, fixed
CUDA-graph-capture-safely via `masked_fill`), and `dff1ef16c0` itself (the
NEXTN/MTP draft never loaded its routed experts from blk.40).

So **TP + speculation on this rig is a re-validation, not a bring-up.** The
qualifier is that the 2026-07-20 registry refactor `d68d8075cd` landed after
it; `2408a5c507` re-proved the **weight stream** byte-identical across that
refactor (63,841 tensors, ordered digest `8aaa592250f43568`) but that covers
loading only, not runtime.

## 4. The router-gate fix is already on the current tree

`0155ff2c00` (the BF16 MoE router-gate rename, the defect that makes *this
exact checkpoint's* blk.40 MTP draft router load as uninitialized garbage while
the model stays fluent) is **not** an ancestor of `integ/round7`, but its
content is present as rebase twins from the #647 line:

- `90131244f2` [#647] Cover the MoE shared-expert gate in the dense-GGUF suffix table
- `67b4e0464d` [#647] Keep a dense BF16/F16 router gate on the .weight path
- guard test: `test/registered/unit/model_loader/test_gguf_dense_router_gate_647.py`

Verified by content at `python/sglang/srt/model_loader/gguf_qwen35.py` (the
`mlp.gate` / `shared_expert_gate` suffix entries), not by ancestry. This closes
the "unverified provenance" residual.

## 5. `test_gguf_draft_quantization` fails here for an environmental reason

`test_a_gguf_drafter_is_built_quantized` is RED on this box with
`AssertionError: None != 'gguf'`. It is **not** a regression. The production
fix is present and correct at `python/sglang/srt/configs/model_config.py:698`:

    draft_path = model_path or server_args.speculative_draft_model_path
    if draft_path is not None and check_gguf_file(draft_path):
        quantization = "gguf"

`check_gguf_file` is a FILE test, and the DFLASH drafter it names
(`.../qwen3.6-27b-dflash-gguf/Qwen3.6-27B-DFlash-Q8_0.gguf`) is absent from
this machine, so the branch cannot arm. The sibling class in the same file
guards on `os.path.exists`; this test does not, which is the actual defect --
a checkpoint-dependent assertion presented as pure config plumbing. The other
25 tests in the GGUF cluster pass.

This matters to the mission because NEXTN passes the target `.gguf` as
`--speculative-draft-model-path`, so this is exactly the branch that arms the
drafter's quantization on a real boot.

## 6. Things that would each have cost a GPU window

**PP and speculation cannot share a phase.** `server_args.py:19303`, reached
from `entrypoints/engine.py:889`:

    assert self.speculative_algorithm is None or self.enable_phase_flip

No draft worker exists in a PP phase and the draft constructors take no
pp_rank, so this is enforced by construction. `boot.sh` STAGE=d asked for PP=3
+ NEXTN and would have died on it. Note that construction alone does NOT
refuse -- `ServerArgs(pp_size=3, speculative_algorithm="NEXTN")` builds fine and
only `check_server_args()` rejects it, so a constructor-only test misses this
entirely. Measured: PP=3+NEXTN REFUSED, PP=3 no-spec ACCEPTED, TP=2+NEXTN
ACCEPTED, TP=1+NEXTN ACCEPTED.

**`test/registered/quant/test_gguf.py` is not a desk test.** It launches a real
`sglang::scheduler`, which took 26,464 MiB on the 5090 when run as part of what
looked like a unit sweep. Outside a GPU window that is an unarbitrated claim.
The genuine desk-safe kernel tests are the three under
`test/registered/unit/quantization/`: `test_gguf_moe_stride_width_512.py`,
`test_gguf_moe_expert_ids_sanitize.py`, `test_gguf_capability_floor.py` --
24 passed in 9.4 s with the cards left at ~0 MiB. They cover the
`b97e60b25e` / `1d689439f6` defect families.

**The multimodal wrapper resolves itself, provided nothing else is staged.**
This checkpoint's `config.json` declares `Qwen3_5MoeForConditionalGeneration`,
and GGUF here is text-only (llama.cpp keeps the vision tower in a separate
mmproj). `model_config.py:404-434` detects exactly this, looks for an
`mmproj*.gguf` beside the backbone, finds none, and forces multimodal OFF. That
is not cosmetic: an uninitialized vision tower yields NaN image features on the
server's own VLM warmup, the NaN residue survives in recycled mamba slots, and
every later prompt crossing the mamba chunk grid is corrupted until
`/flush_cache` (#52). Consequence for staging: do NOT drop `mmproj-*.gguf` into
that directory unless vision is actually wanted, because its mere presence
re-enables multimodal.

**`hasattr(torch.ops.sgl_kernel, name)` is only meaningful after
`import sgl_kernel`.** Before the import it answers for an unpopulated
namespace. All eight GGUF ops are present in this venv's build
(`ggml_moe_a8_vec`, `ggml_moe_a8`, `moe_align_block_size`,
`ggml_moe_get_block_size`, `ggml_dequantize`, `ggml_mul_mat_vec_a8`,
`ggml_mul_mat_a8`, `ggml_mxfp4_native`). Two of them cannot be probed by hand:
`ggml_moe_get_block_size` takes no tensor argument and so has no dispatch key
(call the module-level wrapper, not `torch.ops`), and `moe_align_block_size`
takes a bool `pad_sorted_token_ids` in position 7. Use the unit tests above
rather than hand-rolled calls.

## 6b. Stage d is structurally sound (PP layer ownership)

Worth checking before spending a window on it, because the GGUF adapter has no
pipeline awareness of its own: `gguf_qwen35.py` and `gguf_adapter_base.py`
contain no `pp_rank` / `start_layer` handling at all, and the one PP layer
filter in `model_loader/loader.py:1446-1451` belongs to `QuantizedRLModelLoader`
— a different loader that the GGUF path does not use.

The filtering happens model-side instead, which is the right place: the GGUF
stream is fed to the model's own `load_weights`, and `models/qwen3_5.py`
(load_weights at 2011) skips any tensor whose `layer_id` falls outside
`[self.start_layer, self.end_layer)` and gates `lm_head` on
`pp_group.is_last_rank`. The model builds its layers through `make_layers` with
`pp_rank` / `pp_size` and exposes `start_layer` / `end_layer` (1515-1553).

So each PP stage loads only the layers it owns, and stage d does not need a
GGUF-side change.

## 7. Open residuals for the rig

1. **PP=3 GGUF has never been executed.** `boot.sh` STAGE=d was written and its
   own commit says "Nothing in this commit has been run on a GPU". No later
   commit records any stage of it running. This is the genuine unexplored half
   of "PP/TP".
2. **Q4_K_XL has had no on-card serving run since 2026-07-18**, across the
   registry refactor and ~1500 commits. The Aug-8 coherence proof used
   Q3_K_XL, because of the refuted ceiling in section 1.
3. **topk>1 tree drafts on the rig's native sgl_kernel path** are not
   confirmed. k=3 was validated exact on the laptop only, behind
   `SGLANG_ALLOW_TRITON_SPEC_TREE=1`, which the rig should not need.
4. **Uneven-TP `auto` ratio vs real per-card capacity** was never reconciled.
   Section 1 removes the reason to worry about the 5090 specifically, but the
   sizing input remains unaudited.
5. The checkpoint's location on this box is unresolved; `boot.sh`'s
   `MODEL_DIR` does not exist here.

## 8. What replaces `boot.sh`

`docs/dev/651/rig_boot.sh` in this tree. Same STAGE a/b/c/d ladder and the same
NVML-UUID device resolution, with the corrections this file records:

- runs the **current** tree, not `/spinning/wt-gguf-q4-651` (1509 commits
  behind; BOOT NUR NEUESTER STAND forbids booting it),
- **resolves** the checkpoint instead of asserting a path that is not here, and
  refuses before CUDA init if it cannot,
- validates STAGE before anything expensive,
- refuses when NVML resolves fewer cards than TP x PP needs,
- `DRYRUN=1` prints the resolved cmdline and exits, so the whole script above
  the launch is exercisable at desk without a GPU window,
- does not set the barlink BAR1 env, with section 1's reasoning inline.
