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

## 6. Open residuals for the rig

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

## 7. What replaces `boot.sh`

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
