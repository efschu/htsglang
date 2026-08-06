# NOTE 616b — the residual index-OOB crash: what the window established

Window: 2026-08-06, agent hunter-4, worktree `/spinning/wt-616b-hunter4`,
branch `fix/overlap-stream-race-616b`, base `07b23d9d5c`.
Artifacts: `/spinning/616b-hunter4/` (per-arm boot logs, verdicts, py-spy wedge
dumps, guard counters, CUDA coredump).

## 1. What the crash actually is (now pinned, not inferred)

A CUDA coredump was captured on a live reproduction (arm E) with
`SGLANG_CUDA_COREDUMP=1`, which costs nothing until a device exception fires.
`cuda-gdb` names the faulting kernel outright:

```
at::native::index_elementwise_kernel<128, 4,
  at::native::gpu_index_kernel<at::native::index_kernel_impl<
    at::native::OpaqueType<4> > ...>>  <<<(1,1,1),(128,1,1)>>>
```

So: an ATen **advanced-index GATHER** (`index_kernel_impl`, not index_put),
element type `OpaqueType<4>` = a **4-byte dtype (int32)**, launched as **one
block of 128 threads**. That is a small int32 gather, which is what
`predict[accept_index]` and its siblings in the verify path are.

**The lane pattern is the real find.** `info cuda threads` on the coredump:

| trapped lanes | count |
|---|---|
| 0,1,2 | 3 |
| 4,5,6,7 | 5 (4..8) |
| 8 | 1 |
| 12,13 | 2 |

Ten trapped lanes: `{0,1,2}, {4,5,6,7}, {8}, {12,13}`. Read as a
`bs x (spec_steps+1)` = 4x4 grid, those are the **first 3, 4, 1 and 2 entries
of each row** — a per-row PREFIX. `accept_index` is initialised to `-1` and
filled left-to-right for the accepted positions only, and `-1` is *in range*
for the ATen bound (`-size <= index < size`), so it never asserts.

Therefore: **the trapped lanes are exactly the ACCEPTED entries, and the `-1`
padding lanes are untouched.** The index tensor is structurally intact — right
shape, right per-request accept lengths (3,4,1,2) — but the values it carries
at the real entries are out of range for the tensor being gathered.

This kills the "the buffer is garbage / uninitialised" reading. Garbage would
not leave the `-1` padding perfectly intact in exactly the right places. What
is wrong is the *magnitude* of otherwise well-formed indices, i.e. a
**size/stride disagreement between the index tensor and the gathered tensor**,
not corruption of the index tensor's bytes.

The earlier reproductions are consistent: 12 lanes at bs=3 (all 12 trapped),
16 lanes at bs=4 with 8 trapped. Width always `bs x (spec_steps+1)`.

**An unresolved tension in this reading, stated rather than glossed.** If the
trapped lanes really are `accept_index` gathering from `predict`, then row 0's
accepted entries should hold the SMALL values 0,1,2 (for a topk=1 chain,
`accept_index[i,j] = i*draft_token_num + j`), and those are in range for any
non-empty `predict` — yet they trapped. So at least one of the following must
also be true, and the values in the dump would say which:

1. the gathered tensor is not `predict` but a much smaller int32 tensor (the
   verify path has several int32 gathers of this shape — the mamba commit
   indexes `accept_index` too, and two of the three observed surfaces were in
   mamba/req-pool code), or
2. the index tensor is not `accept_index` but another tensor that happens to
   carry `-1` in exactly the padding positions, or
3. `accept_index`'s accepted entries do not carry the chain-offset values this
   reasoning assumes.

What the lane pattern establishes independently of which of those holds is the
strong part: the trapped set is *structured*, aligned to a
`bs x (spec_steps+1)` grid and to per-row accept lengths, so whatever tensor
this is, it was written by something that knew the batch's real accept
structure. That is not the signature of a freed-and-reused allocation.

## 2. The surface site is aftermath — now confirmed three times over

A device-side assert destroys the CUDA context, so the exception surfaces at
whatever sync point runs next. Observed surfaces, all different, all unrelated
to the fault:

- `memory_pool.py:1484` (mamba index mapping scatter) — 12:39 wedge
- `overlap_utils.py:542` (`fwd_prepare_d2h_stream.synchronize()`) — 14:06
- `paged.py:298` `torch.unique(free_index // page_size)` reached through
  `mamba_component.drive_eviction` → `unified_radix_cache._evict_to_host` — arm C

Nothing should be read into the surface site. The coredump is the only thing
that named the real kernel, and it did so on the first capture.

## 3. Instrumenting the hot path PERTURBS this bug — measured, not assumed

`srt/debug_utils/index_race_guard.py` was built for this window: sync-free
(counters accumulated on the current stream, read back with the #517 staged
non-blocking D2H + event query), non-fatal (counts instead of asserting), with
a production/consumption `snapshot`/`check_stable` pair whose non-zero
mutation count would be positive proof of a foreign-stream write.

It works — its counters are populated and its own can-fail proof passes — but
**arming it changed the failure mode**:

| arm | guard | outcome | time to failure |
|---|---|---|---|
| A | on, clamp on | wedge → Bar1CollectiveAborted | ~40 s of load |
| B | on, clamp off | wedge → Bar1CollectiveAborted | ~48 s |
| C | **off** | **IndexKernel assert, 12 lanes** | ~311 s |
| D | off (+coredump) | wedge (killed by operator) | ~420 s |
| E | off (+coredump) | **IndexKernel assert, 10 lanes + coredump** | ~233 s |

Guard-on runs never reached the index assert. The guard's counters were clean
(`bad=0`, `mutated=0`) up to the moment the device stopped progressing, so they
report nothing about the fault itself.

The mechanism is not cost. The guard adds on the order of 0.5 % of round time;
what it changes is the *ordering* inside the verify → broadcast → consume
window, which is precisely where this race lives. This is the same lesson
`CUDA_LAUNCH_BLOCKING=1` already taught (15 min clean under CLB vs 3.5 min to
crash without), and it generalises:

> **For this bug class, any instrument placed inside the racing window is a
> treatment, not a measurement.** The only trustworthy instruments are the ones
> that cost nothing until the fault fires — the GPU coredump is one, a py-spy
> of a live wedge is another.

The guard is kept, defaulted OFF and documented as such, because it is the
right tool for the *next* question (is a specific tensor mutated across
streams?) and because its disabled path is a single module-level bool test. It
is not the tool for finding this fault.

## 4. A SECOND, independent failure family in the same load

Three arms (A, B, D) did not assert at all; they **wedged**, and the py-spy
wedge-catch caught all three ranks in the identical place:

```
common_template (flashinfer_backend.py:7429)
init_forward_metadata_out_graph (flashinfer_backend.py:7503)
execute (eagle_draft_cuda_graph_runner.py:647)
draft (eagle_worker_v2.py:1021)
```

`flashinfer_backend.py:7429` is `self.kv_indptr[:, : bs + 1].cpu()` — a
**blocking D2H** inside the draft path. All three ranks host-blocked there
means each is waiting for its own device queue to drain while that queue holds
a BAR1 spin kernel waiting on a peer flag; after the 300e9-cycle deadline the
transport raises `Bar1CollectiveAborted`. Two ranks then abort and the third
dies on the torn-down gloo group ("Connection closed by peer").

This wedge occurs **with the guard off** (arm D), so it is not an artefact of
this window's instrumentation. It is a distinct live failure sharing the same
load, and it is why "the reproducer hits in 3.5-4 min" is only true about half
the time — the two families compete for which kills the run first.

## 4b. The LXC hypothesis: FALSIFIED for the wedge family

The user's hypothesis was that the crashes began when htsglang moved from the
Proxmox host into the LXC container (CT999), i.e. that containerisation is the
axis. Tested directly, in three environments running the SAME worktree:

| environment | wedge reproduced? | where |
|---|---|---|
| CT999 (LXC) | yes, arms A/B/D | `flashinfer_backend.py:7429`, all 3 ranks |
| Docker on the host | yes, ~72 s of load | same line, all 3 ranks |
| **Bare metal on the host** | **yes, ~86 s of load** | same line, all 3 ranks |

The bare-metal arm used the Proxmox host's own kernel and init with **no
container at all** — the only isolation was a private MOUNT namespace, used
solely so the hardcoded `/spinning/...` paths resolve (no pid/net/user
namespace, no cgroup limits, no seccomp/apparmor). It ran CT999's OWN venv
(same torch 2.11.0+cu130, same sgl_kernel 0.4.4, same flashinfer) and this
worktree, so code and wheels were identical to the LXC arms.

**Verdict: the wedge is NOT caused by the LXC container.** It reproduces
outside it, in the same time envelope, with an identical three-rank stack.
Evidence: `/spinning/616b-hunter4/bmrun/baremetal_wedge_dumps.txt` (bare
metal), `hostrun/dockerhost_wedge_dumps.txt` (docker), `armB/`, `armD/` (LXC).

Scope this honestly: what is falsified is the LXC axis for the WEDGE family.
The INDEX-ASSERT family was not observed on bare metal within this window, so
for that family the LXC axis is untested rather than excluded — though the two
families share one load and one code path, and nothing about the assert's
mechanism (§1) is container-related.

Confounds recorded for the bare-metal arm:
- The Proxmox host (Debian 13) has no python3.12; the venv's base interpreter
  is `/bin/python3.12`. CT999's python3.12 binary, stdlib and headers were
  staged under `/spinning` and symlinked into the host's `/usr`. Nothing was
  built and no new venv was created; the binary needs no libpython and only
  libm/libz/libexpat/libc, all present on the host. Remove
  `/usr/bin/python3.12`, `/usr/lib/python3.12`,
  `/usr/include/python3.12` and `/usr/include/x86_64-linux-gnu/python3.12`
  to restore the host.
- Neither the host system CUDA (12.2) nor CT999's (12.9) can target the 5090's
  `compute_120`; both reject it. CT999 in fact builds with the CUDA 13 nvcc
  BUNDLED IN THE VENV (`nvidia/cu13/bin/nvcc`), and the bare-metal arm was
  pointed at that same toolchain, so the barlink extension is the same build.
- `ulimit -n` differs: host default 1024 vs CT999 524288. The bare-metal arm
  was launched with `ulimit -n 524288` to match.
- `/dev/shm` is 63G in both.

## 5. What is NOT the cause (falsified here or earlier)

- Not the two req-pool double-frees (#616 hunter-3): their fail-loud guards
  logged zero refusals across every arm.
- Not rank-divergent captured graphs (#603b hunter-2 census).
- Not "the index buffer is garbage": the `-1` padding survives intact in
  exactly the right lanes (§1).
- Not a barlink partial-broadcast of `accept_index`: the guard's
  pre/post-broadcast counters were equal and clean on every guarded round.

## 6. Named next step (not yet done)

The coredump was taken with the default lightweight flags, which include
`skip_global_memory` — so the kernel is named but the *values* and the gathered
tensor's `sizes[i]` are not in the dump. The single decisive next measurement
is a repeat of arm E with global memory retained
(`CUDA_COREDUMP_GENERATION_FLAGS` without `skip_global_memory`, ~17 GB/rank,
disk permitting): that yields the actual index values and the actual bound,
which turns §1's "size disagreement" from a strongly-evidenced inference into
a named pair of numbers, and with it the specific tensor.

Given §3, that measurement must NOT be paired with any hot-path guard.
