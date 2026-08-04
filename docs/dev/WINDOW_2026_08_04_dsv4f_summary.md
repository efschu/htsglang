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

* **#470 Boot B**: `resident_fraction._from_flag()` falls back to the runtime context, which still holds the *target's* args during the draft build. Needs the draft args published into the context — affects DFLASH/EAGLE too.
* **#462 F2**: one missing `LogitsProcessorOutput` branch in the BCG buffer layer. Deliberately not patched: its fields do not share a leading dimension, so a wrong mapping would not raise — it would yield silently wrong logits.
* **#478 stream-trim budget model** — see above.
* ~~`RESIDENT_FRACTION_CUT` default of 0.383 should become **~0.23** (measured); at 0.383 rank 0 OOMs mid-build.~~ DONE (`boot_470_dspark.sh:59`).
* The `geom_seq` determined scorer is too strict — it marks `2 4 8 16 32 64 128` wrong for wanting `32 64 128`, understating quality.
