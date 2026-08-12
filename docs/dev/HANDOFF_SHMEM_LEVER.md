# HANDOFF: the 75-GB host-shmem lever (#695)

Branch `fix/scheduler-shmem-residency`, worktree `/spinning/wt-shmem-lever`,
based on `281bbfb739`. Desk + hermetic only. **Not merged — the operator
sequences merges.**

---

## 1. ERRORS FIRST — what is NOT proven, and what could still be wrong

Read this section before believing anything in section 3.

### 1.1 The fix has never run on metal. This is the whole open risk.

Serving stopped (2026-08-12, mid-shift, by the validation window) before the
fix existed, so **the before/after was never measured on a real boot**. Every
number in this document about the DEFECT is measured; every number about the
FIX is projected from the rounding arithmetic. Section 5 is the executable
recipe that closes this.

Specifically unproven on metal:

* **`cudaHostRegister` on a ~13 GiB region.** The host KV pool uses this path
  routinely but at pool sizes, not at 13 GiB in one call. If the driver
  refuses to lock that many pages in one go, the code logs a warning and falls
  back to `torch.zeros(pin_memory=True)` — the boot survives and you get the
  old rounding back. **If you see `#695 exact-size pinned host image
  unavailable` in the log, the lever did not fire and this is why.**
* **H2D throughput of the restore after the change.** `cudaHostRegister`
  page-locks the same way `cudaHostAlloc` does, so the DMA property should be
  identical, but "should be" is not a measurement. The flip's restore cost is
  the thing to watch; `phase_flip_spill.py` calls it "most of the restore's
  cost" on the Gen4 x4 rank. **Measure the flip latency before/after.**
* **`is_pinned()` now returns False on these images.** Nothing in the tree
  reads it on them (grepped: `weights_arena.py`, `phase_flip_boot.py`,
  `phase_flip_spill.py` — the only `pin_memory` mentions are the allocations
  themselves), but a future caller that branches on `is_pinned()` would take
  the wrong branch. The memory *is* locked; torch's bookkeeping just does not
  own it.

### 1.2 A correctness risk I accepted deliberately, and you should re-check

`image_from_tensors` promises "alignment-gap bytes are zeroed, so the checksum
is deterministic". The old code got that from `torch.zeros`. The new code gets
it from **MAP_ANONYMOUS's kernel zero-fill guarantee** rather than memsetting
13 GiB per image at boot. That is a real guarantee and there is a unit test for
it (`test_anonymous_mapping_arrives_zeroed`), but it is now load-bearing in a
place where it previously was not. If anyone swaps `HostTensorAllocator` for a
pooling allocator that recycles buffers, **the guarantee dies silently and the
checksum goes non-deterministic across boots.** `_alloc_host_image` pins the
allocator explicitly (`allocator=None` → a fresh `HostTensorAllocator`) to stop
that, but it is a comment-and-argument defence, not a mechanism.

### 1.3 What I did NOT do

* **No madvise / PAGEOUT lever.** The task anticipated the `#408/#644`
  `ConsumedPageDropper` as the tool. It does not apply: that dropper reclaims
  *clean file-backed page cache* behind a GGUF stream, and these images are
  *dirty, page-locked shmem*. `MADV_PAGEOUT` cannot reclaim a page the driver
  has pinned. The repo's own probe note says so (`memtier/probe.py:208-212`).
  The lever here is **not allocating the bytes**, not dropping them after.
* **~58 GiB is genuine and remains.** Two full weight images per rank, held
  for process life because `PhaseFlipStacks.refill` re-reads them on every
  flip. Nothing frees them and nothing can without rebuilding the image per
  flip. This branch recovers the *rounding*, not the design.
* **No refusal path added.** The flip images now *register* a pinned-host post
  but are not *checked* against the budget. Adding the check could refuse a
  boot that works today; I chose visibility over veto. Someone should decide
  whether the check belongs there — see section 6.
* **`--pp-stage-ratio` was not touched.** See 6.2: it is the bigger lever and
  it is a one-flag change.

### 1.4 Measurement caveat

The 75.07 GiB figure comes from cgroup `/.lxc`, the ranks' own cgroup. The
namespace root `/sys/fs/cgroup` reported *nothing useful* — the kernel does not
fully maintain accounting for the root cgroup. The pre-existing
`memtier/profile.py:_read_cgroup_memory` reads the **root**, which happens to
be conservative-but-useless here. The new census module reads the process's own
cgroup from `/proc/self/cgroup`. **`_read_cgroup_memory` still reads the root**
— I left it alone rather than change the behaviour of a function six guards
depend on, but that is an inconsistency someone should resolve.

---

## 2. THE FINDING, measured

Live, read-only, 2026-08-12, PP=3 INT8-W8A8 boot (pids 2045250/1/2,
`sglang::scheduler_PP{0,1,2}`).

```
Pss_Shmem per rank      34.62 + 17.85 + 26.23 GiB   = 75.0 GiB
cgroup /.lxc memory.stat shmem                       = 75.07 GiB  (80,607,027,200 B)
             memory.stat anon                        = 13.6  GiB
             memory.stat file                        = 84.9  GiB   <- shmem lives HERE
             memory.current                          = 99.06 GiB
             memory.events oom_kill                  = 9
/proc/meminfo SwapTotal                              = 0
df /dev/shm                                          = 1.3 GB used
```

`/dev/shm` shows 1.3 GB while the ranks hold 75 GiB, because the mappings are
`MAP_SHARED|MAP_ANONYMOUS` — the kernel renders them `/dev/zero (deleted)` and
they belong to no visible tmpfs.

**Breakdown of the 75 GiB** (from `/proc/<pid>/smaps`, every mapping an exact
power of two):

| rank | mappings | total |
|---|---|---|
| PP0 | 2 × 16384 MiB | 32 GiB |
| PP1 | 2 × 8192 MiB | 16 GiB |
| PP2 | 16384 + 8192 MiB | 24 GiB |
| each rank also | 1 × 512 MiB, 36 × 8 MiB, ~8 × 2 MiB | ~0.8 GiB |
| NCCL `/dev/shm/nccl-*` | ~0.2 GiB | |

---

## 3. ROOT CAUSE — two independent defects, both at file:line

### 3.1 The allocator: power-of-two rounding on a permanent allocation

`weights_arena.py:289` (pre-fix): `torch.zeros(total, dtype=torch.uint8,
pin_memory=pin)`.

Every `pin_memory=True` allocation goes through PyTorch's pinned-host caching
allocator, which rounds up **before** `cudaHostAlloc`. Verified in our own venv,
not from memory:

```
.venv/lib/python3.12/site-packages/torch/include/ATen/core/CachingHostAllocator.h:302
    size_t roundSize = c10::llvm::PowerOf2Ceil(size);
:334
    allocate_host_memory(roundSize, &ptr);
```

Two images per rank (`image_pp` at `phase_flip_boot.py:566`, `image_tp` at
`:627`, both via `snapshot_and_free` → `image_from_tensors`), plus one 512 MiB
draft image (`phase_flip_spill.py:637`). **None is ever freed** —
`PhaseFlipStacks.refill` re-reads them on every flip.

Against the payload figures this repo already records at
`phase_flip_spill.py:851-854`, the rounding reproduces **all six observed
mappings**:

| rank | layout_pp | → | layout_tp | → |
|---|---|---|---|---|
| PP0 | 13482.18 MiB | 16384 | 13163.45 MiB | 16384 |
| PP1 | 8144.00 MiB | 8192 | 7923.95 MiB | 8192 |
| PP2 | 9114.95 MiB | **16384** | 7923.95 MiB | 8192 |

58.35 GiB of payload in 72 GiB of mappings. **13,975 MiB = 13.65 GiB of pure
rounding.** PP2 alone accounts for 7,269 MiB of it: its PP layout clears the
8 GiB line by 923 MiB.

### 3.2 The pricer: shmem counted as reclaimable page cache

`memtier/profile.py:honest_host_memory_bytes` — the #407 declared owner of
"how much host RAM may I have", consulted by every pinned pool through
`pinned_host_budget` — computed `resident = anon + kernel` and deliberately did
not subtract `file`, on the correct grounds that page cache is reclaimable.

**In cgroup v2, shmem is accounted inside `file`, not inside `anon`.** So 75 GiB
of page-locked, swapless, unreclaimable memory was priced as free cache. Every
host-RAM guard in the tree believed it had ~75 GiB more than existed. Nine
cumulative cgroup OOM kills, one presenting as a silent rank death — while every
*GPU-side* ledger correctly said the configuration fitted, because GPU-free was
never the binding constraint.

### 3.3 The ledger: the images registered no post

`pinned_host_budget` exists because "two independently plausible budgets can be
jointly impossible" and sums every pinned pool that registers a
`PinnedHostPost`. HiCache and kv-session-offload register. **The phase-flip
images never did**, so 72 GiB of pinned host RAM sat outside the only registry
that adds host posts up.

---

## 4. WHAT CHANGED

| file | change |
|---|---|
| `python/sglang/srt/memtier/profile.py` | `honest_host_memory_bytes` takes `cgroup_shmem` + `swap_free` and charges the unreclaimable part (`shmem - free_swap`). `_read_cgroup_memory` now returns a 5-tuple and reads `shmem`; new `_swap_headroom_bytes`. Unknown shmem → old answer, never a guess. |
| `python/sglang/srt/model_executor/weights_arena.py` | new `_alloc_host_image` — exact-size `MAP_SHARED\|MAP_ANONYMOUS\|MAP_POPULATE` + `cudaHostRegister` via the in-tree `alloc_with_host_register`, with fallback to the old allocation. Both image sites (`image_from_tensors`, `arena_image`) routed through it. Registers a `PinnedHostPost` per image. |
| `python/sglang/srt/mem_ledger/host_shmem.py` | **new.** Read-only per-rank host-shmem census from `/proc/<pid>/smaps`, classified by owner (`anon-shared`, `shm-nccl`, `shm-named`, `driver`, `file-shared`), in Pss. Reads the process's own cgroup. One grep-able line, **not env-gated**. |
| `python/sglang/srt/managers/scheduler.py` | emits the boot line at the at-rest point, next to the #485 residency census. |
| `scripts/vram_ledger/host_shmem_695.py` | **new, executable.** The metal recipe: `--save`, `--compare`, verdict with exit code. |
| 3 new test files | 32 tests, all hermetic, all CPU. |

### The boot line

```
HOST-SHMEM rank0 host=32.81GiB declared=32.80GiB residual=+0.01GiB \
  anon-shared=32.81GiB/n=48 shm-nccl=0.20GiB/n=30 driver=0.06GiB/n=32 (not host RAM) \
  cgroup_shmem=75.07GiB cgroup_current=99.06GiB cgroup_max=unset swap_free=0.00GiB \
  oom_kills=9 big=[16384MiB:anon-shared 16384MiB:anon-shared 512MiB:anon-shared]
```

`residual >= 1 GiB` escalates it to WARNING with the "if this rank is later
SIGKILLed with no traceback, this is the first line to read" text. Not
env-gated on purpose: an instrument you must request in advance is no use to
the boot that needed it.

---

## 5. THE RECIPE for the next GPU window

`scripts/vram_ledger/host_shmem_695.py` — its module docstring is the full
procedure. Short form, same launch command both halves, only the commit differs:

```bash
# BEFORE — boot on 281bbfb739 (parent), wait for the server, then:
python3 scripts/vram_ledger/host_shmem_695.py --save /tmp/shmem_before.json
curl -s localhost:30030/generate -H 'Content-Type: application/json' \
  -d '{"text":"Count from 1 to 20.","sampling_params":{"temperature":0,"max_new_tokens":64}}' \
  > /tmp/gen_before.json

# AFTER — stop serving, boot on fix/scheduler-shmem-residency, same flags:
python3 scripts/vram_ledger/host_shmem_695.py --save /tmp/shmem_after.json
curl -s ... > /tmp/gen_after.json     # identical request

# VERDICT
python3 scripts/vram_ledger/host_shmem_695.py \
    --compare /tmp/shmem_before.json /tmp/shmem_after.json
diff <(jq -r .text /tmp/gen_before.json) <(jq -r .text /tmp/gen_after.json)
grep 'HOST-SHMEM' <boot log>
grep '#695 exact-size pinned host image unavailable' <boot log>   # must be EMPTY
```

**Acceptance, and note what it deliberately is not.** A fix that frees memory
cannot be validated by a free-memory metric — "more free" is the reading you get
whether or not it worked. The gate is a statement about the **allocation**:

1. No `anon-shared` mapping ≥ 1 GiB is a power of two any more. `--compare`
   exits 1 and names the offenders otherwise. *(This is the load-bearing one.)*
2. Total anon-shared falls ~13.6 GiB across three ranks; largest single drop on
   the rank whose PP layout is 9114.95 MiB.
3. The generation diff is **empty**.
4. One `HOST-SHMEM rank<N>` line per rank, small `residual`.
5. No fallback warning in the log.

The `--compare` gate was proven in both directions at the desk: it passes on
the projected mappings (13.65 GiB drop) and fails loudly, naming all six
mappings, when fed a no-op.

---

## 6. NEXT LEVERS, in bytes-per-risk order

1. **`--pp-stage-ratio`, ~7.1 GiB, one flag, zero code.** PP2's PP layout is
   9114.95 MiB — 923 MiB over the 8 GiB line, and that alone costs 7,269 MiB of
   rounding. Any ratio that pushes it under 8192 MiB reclaims that even
   *without* this branch. Caution at `phase_flip_spill.py:862-873` applies: do
   not re-derive which layout is larger; that assumption already cost three
   ranks on 2026-08-11.
2. **Decide whether the images should be *checked*, not just registered.**
   `check_and_register_pinned_post` would refuse a jointly-impossible
   configuration at boot with both posts named, instead of at the OOM killer's
   convenience. I did not wire it because it can refuse a boot that works
   today. With the pricer fixed (3.2) the numbers it would compare against are
   now honest for the first time, so this is worth revisiting deliberately.
3. **~58 GiB is the design, not a bug.** Reducing it means rebuilding an image
   per flip instead of holding both — a real trade against flip latency. Out of
   scope here; name it before anyone "discovers" it again.
4. **`--phase-flip-spill-depth cache`** instead of `arena` removes 3 × 512 MiB
   and nothing more. Not worth the VRAM rungs it costs.
5. **Resolve the cgroup-path inconsistency** in 1.4.
