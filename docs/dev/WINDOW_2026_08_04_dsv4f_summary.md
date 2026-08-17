# DSV4F window 2026-08-04 — summary

Artifacts: `/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`
Branch: `feat/dsv4f-window-2026-08-04`

**Power state, every measurement:** 3080s **200 W** (default 320), 5090 **400 W**
(default 575), recorded per arm in `powerstate_*.json`. The user lowered all
targets on 2026-08-03, so **no number here is comparable with a full-power
anchor from an earlier day.**

**Device index spaces:** NVML/PCI order is 3080, **5090**, 3080; CUDA
FASTEST_FIRST order is **5090**, 3080, 3080. All `--rank-*` flags and
`--speculative-draft-gpu` are CUDA-indexed. Recorded per arm in
`device_order.json`.

---

## Results

| arm | outcome |
|---|---|
| **#478 UD-Q3_K_XL** | **REFUSED on capacity.** Two host-OOM kills at 103.9 of 104.0 GiB during weight load. Driver stays UD-IQ3_XXS. |
| **#470 Boot A** | **Measured.** Residency cut 0.485→0.23 on rank 0 frees 10.21 GiB and costs **~1.4 % of decode ms/round**. |
| **#470 Boot B** | **Blocked.** Four first-boot defects; three fixed, the fourth needs a draft-copy contract change. No accept length, no R1 verdict. |
| **#462 eager control** | **Measured.** 131.475 ms/round, floor 0.401 %. |
| **#462 F2 breakable** | **Blocked** in the BCG buffer layer. #494 instrument never exercised. |
| **#390/#394 expert_stats** | Harvested from every arm (`expert_stats_*`). |

### The measurement band, established

Four arms, four A-vs-A floors: **0.33 %, 0.40 %, 0.72 %, 6.443 %**. The last is
an outlier. Two independent boots of the same configuration (`470_a_base` and
`462_eager`) landed **0.09 % apart** (131.353 vs 131.475 ms/round). So DSV4F
bs=1 decode on this rig at this power state reproduces to well under half a
percent, and 131.4 ms/round is the reference figure.

Quality was identical across every arm that ran: determined 5/8, same items,
byte-identical answers — including across the residency cut, which is what
offload numerical neutrality should look like.

---

## The headline finding: #478

UD-Q3_K_XL is **133.5 GiB in memory, not 119.4 GiB on disk**, because 47.8 GiB
of it is MXFP4 with no kernel on this fork and is repacked to Q5_0 at 22/17
(`gguf_mxfp4_repack.py:39-41`). Planning against the disk figure understates the
swap by **60 %**.

VRAM is saturated either way, so the entire +35.8 GiB lands in host RAM. Both
attempts died during load, not at VRAM allocation. Raising the resident fraction
(the direction that *shrinks* the host pool) did not move the peak at all —
because the kill is a **load-time** phenomenon: the 119.4 GiB stream's page
cache, the pinned pool being built, and ~15 GiB of runtime anon are all held at
once.

**Mechanism correction worth carrying forward:** CUDA pinned host memory is
accounted in the cgroup's `file` bucket, not `anon` (the offload ledger showed a
49.66 GiB pinned pool while `anon` sat at 14.6 GiB). `maybe_trim()` computes its
target from `memory.current`, which *includes* the pinned pool, while
`memory.reclaim` can only take clean page cache — so the target is unsatisfiable
at any setting and the trim evicts the loader's own read-ahead as fast as it is
created. The trim already has `pinned_bytes` in the ledger and should compute
`target = pinned + anon + headroom`. **This is a defect independent of #478.**

---

## Instrument defects found (all in the probe layer, none in the runtime)

Six, and five produced output that *looked* like data:

1. `stream_bounded` defaulted to `max_new_tokens=100000`, over the 8192 context → instant 400 → 1 chunk, empty meta, `None` floors.
2. `--max-new-tokens` was never an argparse option → `AttributeError`.
3. That exception was **swallowed** by the per-mode handler into a record indistinguishable from a real one.
4. `mode_determined` treated `_post`'s raw body as parsed JSON → every chat answer empty.
5. The determined probe sent instructions to `/generate`, where a base-completion endpoint **continues** them instead of obeying (scored 1/8; failures were echoed prompts and web boilerplate). **This would have been written up as a quality regression against IQ3_XXS.**
6. `wait_ready` tested curl's **exit code**, which is 0 for 4xx/5xx, so it declared readiness the moment the port bound and answered 503 — two full ~6 minute loads produced no measurement at all.

Root cause of the pattern: the desk phase validated the probes' pure helpers
thoroughly and their **server-facing path not at all**. Fixed by building
`mock_sglang.py`, which reproduces the live SSE shape (including the
over-context 400 and final-chunk-only spec counters) and found the root causes
in seconds instead of 6-minute boots. **A GPU boot is not a test harness.**

Also verified on the way: `spec_verify_ct` exists **only on the final chunk**
(`tokenizer_manager.py:2145-2153`), so the original per-chunk delta could never
fire and ms/verify would silently have been ms/token under a different label.

---

## Runtime fixes committed

* **#470 solo-shadow marlin exemption** — the docstring promised it, the code had no shadow predicate, so the guard refused the very configuration its own error message recommends. Falsifier with two can-fail arms.
* **#470 axis correction** — the first version of that fix compared `gpu_id` to `--speculative-draft-gpu`, passed its hermetic test, and failed identically on hardware. The workers' predicate is `tp_rank == speculative_draft_solo_rank()`. *A hermetic test passing is not evidence a predicate selects the right ranks when the test supplies the same wrong axis the code reads.*
* **#470 draft load format** — the draft inherited the target's `gguf` format and was handed a directory. Named explicitly in the recipe.
* **#462 eager control flags** — `--disable-cuda-graph` is demanded by name; the control is now the proven recipe unmodified.

## Open, with mechanism identified

All three code-side blockers below were closed after the window (#537/#535,
branch `fix/534-535-followups`). **None of them has a card behind it**: what
follows each entry is what the NEXT window has to measure before the item can
be called done.

* **#470 Boot B**: `resident_fraction._from_flag()` falls back to the runtime context, which still holds the *target's* args during the draft build. FIXED — `build_draft_tp_worker` now publishes the draft copy for the build (the restore already existed). Reach is the DFLASH/DSPARK builder only; the EAGLE family passes the target's `ServerArgs` object with no copy at all, so the same defect class is OPEN there. **Next window: Boot B unchanged, ready with `0.23,0.42,0.42` intact, then accept length + ms/verify + the ANALYSE_447 §2.4 idempotence comparison against `idem_reference_470_a_cut.json`.**
* **#462 F2**: one missing `LogitsProcessorOutput` branch in the BCG buffer layer. Deliberately not patched: its fields do not share a leading dimension, so a wrong mapping would not raise — it would yield silently wrong logits. FIXED — an allowlist of the five per-token tensor fields, every other field refused by name, and a refusal when the present fields disagree on their leading dimension. A SECOND defect surfaced in the same layer: the shared buffer was sized from the graph key (the BATCH size) while the body emits `bs * num_tokens_per_bs` rows under a non-ragged verify. **Next window: the F2 arm reruns; the first check after a completed capture is byte-identical greedy output against the `462_eager` control (131.475 ms/round, floor 0.401 %), then the #494 crossing count against the 43/round desk figure.**
* **#478 stream-trim budget model** — see above. FIXED — the trim's target is raised to `anon + pinned(all live ranks) + headroom` when that floor sits above the configured target, with the pinned term published cross-rank because `memory.current` is cgroup-wide. **Next window: rerun the UD-Q3_K_XL attempt as a falsifier repetition — the sawtooth (86.2 → 76.8 → 95.5 → 82.7 → 101.5 → 103.5 → 103.9) must NOT recur, and one floor-above-target warning must appear instead; plus a neutrality boot on the standing IQ3_XXS recipe, whose floor is below its target and whose load time must not move.**
* ~~`RESIDENT_FRACTION_CUT` default of 0.383 should become **~0.23** (measured); at 0.383 rank 0 OOMs mid-build.~~ DONE (`boot_470_dspark.sh:59`).
* The `geom_seq` determined scorer is too strict — it marks `2 4 8 16 32 64 128` wrong for wanting `32 64 128`, understating quality.

## Boot B, the #447 residue written out (added 2026-08-17, #447 §6)

#447's remainder determination found item (a) has NO desk residue left: every
open piece is this one boot. The Boot B entry above names the §2.4 comparison;
these three are the rest of it, previously implicit as "prerequisites" rather
than named gates. Nothing here is new work — it is #447 §1.6's risk list,
carried to the window that can actually settle it.

* **§2.4 compressor rollback — now a CONFIRMATION, not an open question.**
  #447 §6.3 answered the design half at the desk: our compressor state is
  position-addressed (`compressor_v2.py:385`, and the kernel's step 1 is a bare
  `tl.store` with no read of the destination), paged rather than sequential,
  and pooled only at a page-completion boundary (`seq_len % COMPRESS_RATIO ==
  0`). llama.cpp needs snapshot/restore because its state is a recurrent ring
  that a write ADVANCES; ours has no position to rewind. **What the boot must
  confirm** is therefore narrow: no page is ever pooled while it still holds a
  slot belonging to a REJECTED position that is never recomputed. The pooling
  loop is `tl.static_range(128)` with no per-slot `seq_len` mask, so this rests
  on positional overwrite plus completion gating — check it against
  `idem_reference_470_a_cut.json` as already planned.
* **Risk 2 — the draft attention backend is pinned.**
  `dspark_config.py:19-21` hard-pins `DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"`,
  the same backend whose sm86/sm120 routes #417 addresses. #417's own gates are
  desk-only and have never run (no `RESULT_417` exists). If Boot B fails inside
  the draft's attention rather than in placement, this is the first suspect,
  and it makes #417 a dependency of this window rather than a parallel task.
* **Acceptance-ladder floor — measure OURS, do not import theirs.** The
  external 0.6-0.77 accept / 1.4-1.8x decode band is a reference, not a
  baseline. #447 §1.6 requires "an A-vs-A same-boot measurement before any
  delta is reported". Boot B produces the first DSpark accept length and
  ms/verify this fork has ever had; until the A-vs-A floor is recorded in the
  same boot, no delta against the external band may be claimed.

Standing correction for whoever reads the DSpark claims: **no DSpark arm has
ever booted on this rig.** Four attempts on 2026-08-04 died before
`/health_generate`. Boot A's residency-cut number (~1.3-1.4 % of decode
ms/round to free 10.21 GiB) is real and was measured with NO DRAFT PRESENT.
Every DSpark-specific number in the tree is desk-only.
