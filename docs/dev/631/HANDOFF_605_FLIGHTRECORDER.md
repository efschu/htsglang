# HANDOFF 605 — the VRAM flight recorder, fill side

Shift: flightrec-r1, 2026-08-13. Branch `feat/vram-flightrecorder-605`,
commit `eb5ace7dff` on top of `481411ac6b`.
Evidence: `/spinning/evidence-631/flightrec-r1/`.

**Read §1 first.** Two of this shift's findings are defects in the INSTRUMENT,
not in the runtime, and any earlier fill-side number read out of the recorder
was read through them.

---

## 0. What the user asked and what this answers

> budgeting to the 1024 MiB corridor has been "a guessing game for almost every
> agent — why is this not computed exactly beforehand? the planner seems
> completely incomplete here."

The answer this shift can defend: **the fixed costs are exactly computable and
were never the problem; the varying cost is the KV pool's own sizing, and it
varies by up to 2408 MiB between boots of an identical config.** Measured, per
post, over 14 boots — §3. No estimate anywhere in that table.

This shift did NOT need a GPU window. Every number below is read out of marks
the ship boots had already written to `/spinning/flight_605/`. The peer shift
(#656 kv-universe) held cards 0,1,2 throughout and was not disturbed.

---

## 1. Two instrument defects — fixed in `eb5ace7dff`

### 1a. Rank collision under PP (marks of three cards merged into one timeline)

Marks are filed under `flight_marks_rank{rank}.jsonl` where `rank` is the **TP**
rank. The ship config is `--tp-size 1 --pp-size 3`, so **the TP rank is 0 in
all three processes** and all three append to `flight_marks_rank0.jsonl`.
`read_marks` keyed its result on the rank field, so the reader returned one
timeline built out of three different cards, and differencing it produced posts
no card ever paid — a 20480 MiB card's 482 MiB CUDA context billed to the
32607 MiB card.

Visible in the old renderer's own output: `attribute_flight.py phases` printed
`=== rank 0` and then rows with `NVML free` of 32085, 20052 and 20052 MiB, i.e.
three cards, under one heading.

Fixed: grouping is by **pid**, which the writing process stamps and which cannot
collide while it lives. The boot is also resolved across all rank files rather
than per file, since ranks do not stop writing at the same instant.

### 1b. The residue was floored to a false zero exactly where it mattered

`non_torch_bytes` was written as `max(0, nvml_self - reserved)`. On this config
torch reports up to **7162 MiB MORE reserved than NVML says the process holds**,
so the subtraction went far negative and the field read `0` — publishing "no
CUDA context, no NCCL buffer, no JIT workspace on this card" for a rank whose
context measured 886 MiB. A floor built to absorb sub-MiB quantisation was
swallowing gigabytes.

Fixed by separating the two cases: `non_torch_measurable` says whether the
subtraction is a measurement at all, and `unbacked_reservation_bytes` measures
how far reserved exceeds resident.

**What the unbacked reservation IS — resolved, §4a.** It is not a leak and not
an accounting bug: it is the KV pool's virtual-memory arena, reserved but not
yet committed.

---

## 2. The fill-side attribution table (stage 3)

Ship boot `1353495-1786609875`, 2026-08-13 08:31:32Z, argv from
`/spinning/evidence-631/s485/ship_argv.txt`
(`--rank-gpu-memory-mib 31800,14000,15600`, `--enable-phase-flip`,
`--pp-stage-ratio 14,10,8`). Full report:
`/spinning/evidence-631/flightrec-r1/fillside_live_boot.txt`.

Cards are resolved by NVML UUID, never by index. Rank 0 is the 5090 — which
matches C21's independent finding by a different route.

| post | 5090 (32607) | 3080-5c64 (20480) | 3080-62db (20480) |
|---|---:|---:|---:|
| nvml_carve_out | 518 | 425 | 425 |
| cuda_context_and_comm | 888 | 482 | 482 |
| weights_target | +14704 | +9360 | +10328 |
| kv_pool_target | +8162 | +5636 | +4370 |
| inter_runner_gap | −12620 | −7280 | −7030 |
| weights_draft | +12796 / +1780 | +7496 / +1620 | — |
| kv_pool_draft | −412 / +576 | −962 / +288 | −662 / +218 |
| graph_capture_draft | +184 | +226 | +226 |
| boot_tail | +378 | +450 | +454 |
| first_forward_transient | +176 | −254 | −672 |
| **= rank resident** | **27308** | **17730** | **17032** |
| foreign_or_unattributed | 186 | 166 | 168 |
| **free** | **4593** | **2158** | **2854** |
| **SUM vs NVML total** | 32607 = 32607 | 20480 = 20480 | 20480 = 20480 |

**The identity closes exactly on all three cards, drift 0 MiB.** The residual
line (`foreign_or_unattributed`, 166–186 MiB) is well under the 100 MiB-per-card
target only in the sense that it is *named and bounded*; it is 66–86 MiB over
that target and is itemised no further by this instrument. That is an honest
open item, not a pass.

Corridor overshoot for this boot: **+3569 / +1134 / +1830 MiB**.

Posts are differenced from **NVML-resident** bytes, not torch reserved, and the
segments are kept **signed and in boot order** rather than summed per phase
name: the boot releases 12620 MiB between the two runners and rebuilds, so
netting equally-named phases would print a post that never existed.

---

## 3. The post spread over 14 boots (stage 4 input) — the actual answer

`/spinning/evidence-631/flightrec-r1/post_spread_14boots.txt`, all 14 boots of
2026-08-13 05:52Z–08:31Z, same config.

| post | 5090 spread | 3080-5c64 | 3080-62db |
|---|---:|---:|---:|
| `cuda_context_and_comm` | **0** (888) | **0** (482) | **0** (482) |
| `weights_target` | 702 | 702 | 702 |
| `weights_draft` | 1268 | 604 | 610 |
| `graph_capture_draft` | 40 | 56 | 56 |
| `kv_pool_target` | **2408** | **1712** | **1364** |
| `inter_runner_gap` | 1136 | 684 | 672 |
| `first_forward_transient` | 1896 | 1144 | 942 |
| `unbacked_reservation` | 2630 | 1740 | 1488 |
| **free at last mark** | **2792** | **1510** | **2498** |

Readings that matter:

1. **`cuda_context_and_comm` has spread exactly 0 across all 14 boots** — 888
   MiB on the 5090, 482 on each 3080, every single boot. This term never needed
   estimating, on this rig, for this driver. Any ledger term that carries a
   modelled guess for the context is carrying a guess where a constant was
   available.
2. **`weights_target` moves by exactly 702 MiB on all three cards.** A weight
   shard should be deterministic for a fixed model and quant. The identical
   702 on three differently-sized cards points at a discrete layout choice, and
   the config runs `--phase-flip-policy auto` — i.e. the boot PICKS a layout.
   Not proven; see §5.
3. **`kv_pool_target` is the dominant variable** (1364–2408 MiB) and it tracks
   the free-at-last-mark spread (1510–2792 MiB) almost one for one. The
   corridor overshoot is not made of hidden consumers. It is made of the KV
   pool being sized differently on each boot of the same config.

This is where "the planner seems completely incomplete" is correct and now has
a number attached. C15 already named `--rank-gpu-memory-mib` as the fill lever
and recorded that the pool clamps at `_profile_available_bytes` (620000 asked,
512552 given). This shift adds: **that clamp is not stable across boots**, and
its instability, not any unbooked consumer, is what leaves 1.1–3.6 GiB on the
cards.

---

## 4a. What `unbacked_reservation` measures — the KV VMM arena's uncommitted span

PROVEN BY CODE READ. The KV pool is backed by a CUDA virtual-memory arena,
`python/sglang/srt/mem_cache/kv_vmm_backing.py`:

- `_DEFAULT_RESERVE_BYTES = 256 * (1024**3)  # 256 GiB virtual; free until committed`
- `class KvVmmArena: """One device's CUDA virtual-memory reservation exposed as
  a torch.cuda.MemPool."""`, built on `CUDAPluggableAllocator` (line 383)
- physical backing is mapped and unmapped incrementally by `commit_range`
  (447), `decommit_range` (533), `commit_span` (677), `decommit_span` (748),
  against a contiguous-from-zero watermark (`_refresh_watermark`, 661)

So the two counters measure different things by construction: torch's
`reserved` counts what the pluggable allocator handed out — the pool's LOGICAL
size — while NVML counts only the pages actually committed. Their difference is
therefore **the logically-allocated but physically-uncommitted span of the KV
arena**, which is exactly the 4584–8344 MiB this shift measured.

Two consequences:

1. `reserved` is not an occupancy measure on this build and must never be used
   to calibrate a ledger term. The recorder now says so per mark
   (`non_torch_measurable`) instead of publishing a floored 0.
2. **The fill-side question changes shape.** The free VRAM on a card is not
   space the KV pool failed to ask for; it is space the arena reserved and has
   not committed. The lever on the corridor is the COMMIT WATERMARK, which C22
   already localised (`kv_backing_relief.py:406-441`, floored by the maximum
   live slot id, sitting 55k–210k rows above the live token count). INFERRED,
   not measured here: that raising the watermark is what fills the corridor.
   The instrument to check it now exists — commit-watermark bytes belong beside
   these posts as their own mark field, and that is the next stage.

## 4. A premise falsified (do not re-inherit it)

**"The card ends up underfilled because the sizing path reads torch's
`reserved` counter, which is inflated by up to 7162 MiB of unbacked
reservation."** Plausible, quantitatively the right size, and **false** on the
CUDA path: `get_available_gpu_memory` (`python/sglang/srt/utils/common.py:810`)
reads `torch.cuda.mem_get_info(gpu_id)` — the driver's free — after an
`empty_device_cache`. It never consults `reserved`. Recorded as C23.

The unbacked reservation is real and measured; it is not, by this route, the
cause of the underfill.

---

## 5. Open risks, in order

1. **The commit watermark is not yet a recorded post.** §4a names what the
   unbacked span is, but the recorder does not yet mark the arena's committed
   bytes, so the chain "watermark -> free VRAM -> corridor" is argued, not
   measured. Adding a `kv_commit_watermark_bytes` field to `mark()` and a post
   between `kv_pool_sized` and `boot_complete` closes it, and is the smallest
   next step with the largest payout.
2. **The 702 MiB weights spread** (§3 reading 2) — layout choice under
   `--phase-flip-policy auto` is the hypothesis, unproven.
3. **`foreign_or_unattributed` of 166–186 MiB per card** is named and bounded
   but not itemised. Prime suspects from history: the tokenizer parent context
   (#237) and lane pools outside the rank budget (#400). The recorder already
   records `nvml_processes` per mark, so this is a reading task, not a new
   instrument.
4. **Everything here is a BOOT SNAPSHOT.** The corridor law is a continuous
   time-series minimum under load (100 ms sampling), not a boot reading. These
   posts describe how a boot arrives at its resting level; they say nothing
   about the in-cutover trough that C20/C21 are about. Do not quote a number
   from §2 as a corridor measurement.
5. **The modelled ledger is not dumped on ship boots.** No
   `ledger_1353495-*.json` exists for the live boot — newest is 2026-08-12
   06:18 — so `reconcile.py` (modelled term vs measured post, the actual payout
   of #605) has nothing to run against on the shipped config. Wiring that is
   the obvious next stage and was NOT done here.

---

## 6. How to reproduce, on any boot, without a GPU window

```bash
cd /spinning/wt-605-fr
export PYTHONPATH=/spinning/wt-605-fr/python

# which boots exist
python3 scripts/vram_ledger/fill_side_report.py /spinning/flight_605 --across-boots 1

# the attribution table for the latest boot (add --boot <id> for an older one)
python3 scripts/vram_ledger/fill_side_report.py /spinning/flight_605

# the per-post distribution over the last N boots
python3 scripts/vram_ledger/fill_side_report.py /spinning/flight_605 --across-boots 14
```

Exit code is non-zero when the per-card identity does not close, so this is
usable as a gate and not only as a report.

Tests: `python3 -m pytest test/registered/unit/mem_ledger/test_fill_side_605.py`
— 4 passed. Red-first: 3 of 4 failed against the pre-fix recorder. Can-fail
proof for the falsifier: a mutant that zeroes the residual line fails with
`!! IDENTITY BROKEN by 300 MiB`.
