# ANALYSE #809 — hybrid residency for the phase-flip weight images

Desk analysis, no boot, no metal run. Written before any implementation
because the prior-art gate falsified two of the three things the task asked
for, and the third rests on a comparison that does not exist yet.

The task as posed: keep a small PINNED share of the flip weight images (the
"hot head, in wave order"), leave the rest file-backed as today, and reload
with prefetch overlap; finance the pinned share from the ~20 GiB of host RAM
freed by #810 and price it as a post in the boot preflight; size it from the
planner, never by hand; treat compression as a named window item only.

## 1. THE OVERLAP IS ALREADY BUILT (#802, 2026-08-22)

`_staged_file_refill` (`python/sglang/srt/model_executor/weights_arena.py:522`)
already reads the file-backed image with bounded `preadv` into a pinned
staging ring and overlaps the next chunk's read with the previous chunk's H2D
DMA: `depth` CUDA streams, `depth` events, an `inflight[]` flag, and a
`events[i].synchronize()` before a buffer is refilled
(`weights_arena.py:582-617`). The ring is #720's `ReadBufferPool`, so its
bytes are charged to the #729 registry before they are allocated
(`weights_arena.py:496-519`).

The depth is a flag: `SGLANG_PHASE_FLIP_REFILL_DEPTH`, default **2**
(`environ.py:287`) — literally a double buffer. Chunk size is
`SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB`, default 32 (`environ.py:286`).

So the #125 double-buffer pattern the task names as the thing to build is
present, shipped, and default-on for the file-backed arm. Building it again
is the #720-verbatim mistake in a new place. What is NOT overlapped is the
PINNED arm: that is one contiguous `dst.copy_(payload)` (`weights_arena.py:1097`)
and needs no overlap, because there is no read to overlap with.

## 2. THERE IS NO "HOT HEAD", BECAUSE NOTHING CONSUMES A PARTIAL ARENA

The task's framing borrows #254's wave order
(`SGLANG_MOE_OFFLOAD_WAVE_ORDER`) as the precedent for "the order is fixed and
known". The order here IS fixed and known -- `ArenaLayout.slots` is built from
`sorted(named)` and the image is copied by strictly increasing byte offset --
but the precedent does not transfer, for a reason that removes the whole
question:

`arena_refill` copies the entire payload and only THEN checksums it
(`weights_arena.py:1093-1099`); on mismatch it either restores the previous
layout or marks the arena undefined. Nothing reads the arena before that
checksum passes. There is no partial publication, no per-wave consumption, no
first-needed subset.

Therefore WHICH bytes are pinned cannot matter -- only HOW MANY. #254's wave
order exists because a MoE forward consumes experts wave by wave and the
order changes how often a spill expert crosses PCIe. A flip consumes the arena
once, whole. The pinned share of a flip image is a QUANTITY, not an ORDER, and
any design that ships a "hot head selection" ships a knob with no effect.

## 3. THE PIN IS WORTH ~2x ON THE BYTES IT COVERS -- FROM TWO SEPARATE CAMPAIGNS

Both numbers exist in the tree, with citations, but they were NOT measured
against each other. Stated as two measurements, not as an A/B:

* PINNED path, `#690`: 4.93 / 7.08 / 8.88 GB/s per rank
  (`phase_flip_boot.py:502-509`, raw table in
  `docs/dev/NOTE_690_gdn_state_spread.md:83-85`), i.e. ~4702 / 6752 / 8468
  MiB/s, mean over 14 flips, 9614.9 MiB per rank, `pp_to_tp`.
* FILE-BACKED staged read path, `#802`: 8304 MiB/s synthetic full-sweep
  O_DIRECT, but **2651 / 2602 / 3751 MiB/s per rank in the real flip on
  metal** (`weights_arena.py:553-571`).

So on metal the pinned path is roughly **1.8-2.6x** the staged path. A pinned
share removes the read entirely for the bytes it covers; it does not remove a
write, because the weight images are built ONCE at boot
(`phase_flip_boot.py:330` -> `image_from_tensors`) and every flip only reads
them. Any argument that a pin also saves a write-back is wrong for these
images.

Order-of-magnitude, explicitly labelled as an ESTIMATE across two campaigns
and not a measurement: ~6.8 GiB pinned per rank (a ~20 GiB budget over three
ranks) moves in ~1.45 s pinned against ~2.6 s staged on the slowest rank, so
~1.2 s off a flip whose measured wall time is 4.998 s -- about a quarter.
Worth having, and NOT worth shipping on this arithmetic alone (see §6).

## 4. TWO OF THE THREE BRIEFING NUMBERS DO NOT EXIST IN THIS TREE

The task cites "refill warm 2850-4263 MiB/s (pp_to_tp)" and "writeback-cold
1763-1844 (tp_to_pp)". Grepped as rates over `python/sglang/srt/` and
`docs/dev/`: **zero hits for 2850, 4263, 1763 and 1844 as a MiB/s, MB/s or
GB/s figure.** The third number, pinned 4930-8880, is real and is
`_PINNED_REF_LO_GBPS` / `_PINNED_REF_HI_GBPS`.

Nearby non-matches, so the next reader does not "find" them by accident:
`1763` occurs in `phase_flip_runtime.py` as a byte budget in an affordability
check, and in `WINDOW_LADDER_0818.md` as an NVML free-memory corridor floor;
`1844` occurs in `docs/dev/631/PROD_BRINGUP_BENCH.md` as a seam-staging free
memory delta in MiB. None is a refill rate.

This does not make the task wrong -- it makes those two figures unsourced.
Any sizing that used them would be a rig-fit against numbers nobody can
reproduce. The numbers that DO exist are in §3 and should be cited instead.

## 5. COMPRESSION: DISCARDED, WITH THE ARITHMETIC

Not a window item. Refused twice over.

By this tree's own prior analysis: `docs/dev/ANALYSE_306_lossless_ratio.md`
ends with "do not build #306 as a codec" -- parallel decompression saturates
at 4.3-4.8 GB/s while serial break-even needs 1.59-1.72x, and the best ratio
achieved on any real asset was 1.211x (FP8 weights). #456's sparse write is
not a codec but a hole-punch, it has one production caller
(`hibernate.py:538`), and `DESIGN_456_sparse_image_write.md` records **0 byte
win on /spinning** because ZFS already folds the same holes -- which is the
filesystem the flip images live on.

And independently, against the CURRENT read rates. With ratio r = 1.145 the
bytes saved are 1 - 1/r = 12.7 %.

* Serial (decompress after read): a win needs
  `R_dec > R_read / (1 - 1/r)` = **7.9 x R_read**, i.e. 64.0 GiB/s at the
  synthetic 8304 MiB/s, or 20.1-28.9 GiB/s at the metal per-rank rates.
* Fully overlapped (best case, decompression pipelined behind the read):
  `R_dec >= r x R_read` = **9.29 GiB/s per rank** synthetically, and
  **10.07 GiB/s aggregate** across the three concurrent ranks on metal.

Against a measured decompression ceiling of 4.3-4.8 GB/s, both bars are out of
reach by a factor of 2-13. Compression buys 12.7 % of the bytes at a cost that
exceeds the read it replaces. Discarded; re-open only if the read path ever
becomes the bottleneck again AND a decompressor faster than the pool exists.

## 6. THE DANGER DIRECTION, AND WHY A REFUSAL IS ALLOWED HERE

The file-backed arm exists to stop the images -- ~68.7 GiB unreclaimable host
RAM -- from OOM-killing a swapless boot. Its refusals are explicit
(`weights_arena.py:704-710`, `:719-725`): it refuses a missing image dir and
refuses a tmpfs dir, "refusing rather than silently allocating a pinned image
the host ledger would then double-count as reclaimable". A pinned share must
not reopen that hole.

The tree has already taken a position on checking image posts, and it is the
OPPOSITE of what the task asks for. `_register_image_post`
(`weights_arena.py:796-822`) uses the bare `register_pinned_post`, i.e.
visibility only, no admission check, and says why
(`weights_arena.py:918-924`): *"Registered, not CHECKED: a new refusal path
here could break a boot that works today."*

That decision stands and must not be reversed. But it does not bind a NEW,
opt-in, default-off pinned share, and for exactly the reason it gives: a
refusal on a share that no current boot requests cannot break a boot that
works today. So the hybrid share -- and only that share -- may be CHECKED
through `check_and_register_pinned_post`, or priced through
`joint_pinned_host_error` the way `ServerArgs._post_hicache_staging_host_ledger`
prices the #810 staging tier. The existing whole-image post stays unchecked.

Corollary for the ledger: the file-backed image is deliberately NOT registered
(`weights_arena.py:696`, `:766`) because the registry sums NON-reclaimable
bytes. A hybrid image must register exactly its pinned head and nothing else,
or the ledger goes wrong in the direction that matters.

## 7. THE ONE HAZARD ANY IMPLEMENTATION MUST HANDLE

`_staged_file_refill` decides per chunk:
`use_direct = at < direct_limit and at % _DIRECT_ALIGN == 0`
(`weights_arena.py:600`, `_DIRECT_ALIGN = 4096` at `:427`).

If a pinned head of arbitrary size is skipped and the file reads resume at
`off = pin_head_bytes`, then every chunk start is `pin_head_bytes + k*chunk`.
Unless `pin_head_bytes` is a multiple of 4096, `at % _DIRECT_ALIGN != 0` for
EVERY chunk, `use_direct` is false throughout, and the whole refill silently
falls back to the buffered fd: **8304 -> 2595 MiB/s, a 3.2x regression**, with
no error and no log line. The pinned head would then cost more than it saves.

So the head must be floored to `_DIRECT_ALIGN`. This is the piece most likely
to be got wrong and it is pure arithmetic, so it belongs in a tested function
rather than inline in the loop.

## 8. WHAT IS LEFT TO BUILD, AND WHAT IT NEEDS FIRST

Left to build: a pinned head of the image, its size a parameter, default 0
(byte-identical to today), priced as its own post, floored to 4096, with
`_staged_file_refill` starting at the head boundary and the head served by a
plain `dst[:head].copy_(pinned_head)`.

NOT built here, deliberately. The share's size is the whole feature, and the
comparison that sets it -- pinned vs staged-O_DIRECT refill on ONE binary,
same load, same bytes -- has never been run. §3 combines two campaigns and
says so. Shipping a size derived from that mixture would be the rig-fit this
project's own planner rules forbid, and shipping the actuator with the size
left as an unmeasured parameter would put a knob in the tree that nobody can
set. A helper with no caller is the class of defect removed from
`planner/hicache_staging.py` in this same branch.

## 9. WINDOW ITEM (metal, needs a load window)

A/B on one binary, same load, same bytes, three arms:
`SGLANG_PHASE_FLIP_REFILL_STAGED=1` (today's default),
the pinned arm (`--phase-flip-image-file-backed` off, if the box can hold it
long enough to measure one flip), and the hybrid at one or two head sizes.
Instrument: `_timed_arena_refill` / `refill_report` already log per-rank
elapsed and rate (`phase_flip_boot.py:512-559`, `:694-749`), so the arms are
comparable without new instrumentation.

That measurement decides whether §8 gets built at all, and at what size. Until
it exists, the honest state of #809 is: overlap done, order irrelevant,
compression refused, quantity unmeasured.
