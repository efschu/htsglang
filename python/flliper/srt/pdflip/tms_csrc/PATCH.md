# Vendored torch_memory_saver 0.0.9.post1 csrc (MIT, fzyzcjy) with FIVE patches

Source: the PyPI sdist `torch_memory_saver-0.0.9.post1.tar.gz` (the version
installed in /spinning/htsglang-gpu/.venv, a binary wheel that ships no csrc).
Only the CUDA preload hook is built from this copy (`hardware_amd_support.h`
is not vendored: `USE_CUDA` only).

## Patch 1 -- #1233, one backup

(core.cpp, `TorchMemorySaver::resume`): after the H2D restore of a
cpu-backed allocation the pinned host image is `cudaFreeHost`-ed and the
pointer nulled.  Upstream keeps it ("currently keep it there to reduce re-alloc
time"), which is DR-1: every group that has slept once holds its whole weights
image in host RAM for the rest of its life, so two groups hold two images at
the flip (boot weg2ls1b2 2026-09-07: host OOM).  `pause()` already handles a
null pointer (it re-allocates), so nothing else changes.

Built by scripts/pdflip/tms/build_tms_preload.sh into a file whose name carries
`torch_memory_saver` (HookUtilModePreload.get_path_binary asserts that on the
LD_PRELOAD value) and the sha256 of these sources; the launcher points
FLLIPER_PDFLIP_TMS_PRELOAD_SO at it and the adapter's configure_subprocess()
honours that variable.  Unset, the stock wheel's hook is preloaded unchanged.

## Patch 2 -- #1235, the shared host granule ring (WEG2_FLIPCOST_SPEC_0907 C1-C7)

New file `host_ring.{h,cpp}`; `core.{h,cpp}` and `entrypoint.cpp` changed.

`cudaMallocHost` / `cudaFreeHost` per allocation is REPLACED (not wrapped) by
`acquire` / `release` against ONE per-card region of `H(c) = max_g image_g(c)`
bytes, shared by both co-located rank processes.  Two groups flipping weights in
opposite directions on the same card no longer need two private pinned images
(61.4 GiB, refused in record 1h) -- they need one region, and `acquire` BLOCKS
on the peer rather than falling back to a second allocator the ledger cannot
price (spec R4/R5).  The copies move from a per-allocation `cudaMemcpy` on the
legacy default stream to `cudaMemcpyAsync` on ONE `backup_stream_` with ONE
`cudaStreamSynchronize` per leg; that stream is created with
`cudaStreamCreate` -- DEFAULT/blocking flags, never `cudaStreamNonBlocking`,
which would silently drop the ordering against PyTorch's default stream (R12).

Two C entrypoints are added: `tms_tag_bytes(tag)` (the per-tag byte instrument,
because the previous one -- an RssShmem delta -- is dead by construction once
the granules are shared pages, R8) and `tms_ring_stats(...)`.

The switch is the presence of `TMS_HOST_RING_DIR` / `_MAP` / `_EPOCH` / `_FORM`,
all four published by `pdflip/launcher.py` and none of them operator-facing; unset,
this file behaves exactly as patch 1 left it.  `_FORM` selects the cross-process
registration form (`MAP_SHARED` file, or `memfd_create` + fd inheritance) --
whether `cudaHostRegister` accepts either is a metal fact, so a ring directory
published without a proven form is refused by name (W33 PdFlipRingFormUnproven)
rather than guessed at.

## Patch 3 -- H95c, the span map (D's seat posts per phase)

`core.{h,cpp}` and `entrypoint.cpp` changed.  An allocation may carry a SPAN
PLAN: granularity-aligned byte ranges that the next `resume` maps, one
physical handle per range, the rest of its VA reserved and empty.  The VA is
never touched, so a captured CUDA graph keeps its addresses.  Two entrypoints:

* `tms_set_spans(ptr, n, lo, hi, now)` -- the plan of the allocation whose BASE
  is `ptr` (`n == 0` = the whole allocation = the stock mapping).  `now` on an
  ACTIVE allocation applies it at once: extents wholly inside the new plan keep
  their pages (and bytes), the others are unmapped, the uncovered ranges get
  fresh pages.  Refuses a cpu-backed allocation (-2: the host backup walks the
  whole size), a malformed plan (-3) and a non-base pointer (-1).
* `tms_alloc_info(ptr, &size, &mapped, &planned, &active)`.

`pause`/`free` release every extent; `resume` maps the plan (a failed map rolls
the whole tag back, like the stock path); `tag_bytes` reports the PHYSICAL
bytes (mapped while ACTIVE, planned while PAUSED).  Without a plan -- every
allocation nobody called `tms_set_spans` on -- each of these is the patch-2
behaviour unchanged.  Used by `pdflip/d_seat_vram.py` (FLLIPER_OPT_PDFLIP_D_SEAT_VRAM):
D's GDN temporal state maps the slots of the phase's n seats, the expert bank
the rows those seats' pages fund.  Desk proof: the unit test builds these
sources against a mock driver (`test_pdflip_d_seat_vram_h95c.py`, section 5).

## Patch 4 -- PAUSE-SUB, the pause's own split (30.09., NF y4h)

`core.{h,cpp}` and `entrypoint.cpp` changed, instrument only.  `pause` pass 3
makes the same cuMemUnmap/cuMemRelease calls in the same order; each pair now
runs on its own clocks (`timed_unmap_release`), and the call records
`(tag, allocations, unmaps, unmap_ms, release_ms, total_ms)`.  New entrypoint
`tms_pause_stats(tag, len, &allocations, &unmaps, &unmap_ms, &release_ms,
&total_ms)` with the `tms_resume_stats` contract (tag written back, 0 = no
pause recorded).  Why: D's `pause_ms` on the 3080 ranks is ~28 ms per ~1 GiB
tag against ~6 ms on the 5090 with `sync_ms=0` on every tag, and P's pause on
the same card costs a third per byte -- the suspect is D's H95c extent count,
which no line carried.  The scheduler prints it as `PDFLIP-PAUSE-SUB`.  Desk
proof against the mock driver: `test_pdflip_pause_overlap_0930.py`.

## Patch 5 -- PAUSE-MAPS, one cuMemUnmap per run of extents (30.09., NF y4i)

`core.{h,cpp}` and `entrypoint.cpp` changed.  With the flag set
(`tms_set_pause_coalesce(1)`, pushed by `pdflip/pause_maps.py` under
`FLLIPER_PDFLIP_ENABLE_PAUSE_COALESCE_UNMAP`, default off) `pause` pass 3
releases a span-mapped allocation's extents sorted by offset, cut into runs
of back-to-back extents: a run of two or more is ONE `cuMemUnmap` over its
whole range (the driver accepts a range covering several adjacent mappings
-- the CUDA samples' multi-device mmap frees its striped range so), then each
handle is `cuMemRelease`d.  A run the driver refuses as one range is unmapped
extent by extent -- the patch-4 walk -- and counted.  Stock allocations and
the flag off: patch 4 call for call.  New entrypoint
`tms_pause_maps_stats(tag, len, &extents, &runs, &fallbacks, &coalesce)`
(contract of `tms_pause_stats`, same sequence).  Why: y4i's 3080 D tags make
31-64 calls for 10-12 allocations (H95c lattice cells, one handle each) at
~0.1-1 ms per call; the lattice must stay (a live shrink keeps only whole
extents), the number of calls need not.  Desk proof against the mock driver:
`test_pdflip_pause_maps_0930.py`.
