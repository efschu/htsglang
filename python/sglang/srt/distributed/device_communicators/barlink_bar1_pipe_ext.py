# SPDX-License-Identifier: Apache-2.0
"""Pipelined BAR1 collective kernel: ``mesh_pipe``.

Why this file exists
-------------------------
``barlink_bar1_ext.py`` carries the kernels ``mesh`` (two grid-wide barriers)
and ``ring`` (``2(R-1)`` barriers) ported from the measurement probe
``bar1_kollektiv.cu``. In the probe, clean phase boundaries were
**desirable** -- a phase clock was meant to report sending, reducing, and
distributing separately. In the transport this decision no longer serves a
purpose, it only costs: between setting a flag and its arrival, the fabric
sits idle. Right in the 1-8 MiB range that's the difference between a
continuously versus a bursty-loaded link, and that's exactly where the
direct path loses to NCCL (0.97x at 1 MiB, 0.86x at 4 MiB, two cards).

This kernel splits the payload into ``K`` chunks and lets sending, reducing,
and adopting overlap. There is **no more grid-wide lockstep per phase**.

Why a separate file instead of a third kernel in ``barlink_bar1_ext.py``:
another effort is working on that file in parallel. The primitives
(``readV4``, ``writeV4``, ``readFlag``, ``chunkBounds``, ``addV4``)
are therefore **duplicated verbatim** here instead of shared. That is
deliberate, and it is a debt: once both efforts are merged, the shared part
belongs in ONE header. Until then: whoever changes a primitive here must
change it there too -- they must stay bit-identical, because both kernels
can write into the same slots (not simultaneously, but in consecutive
rounds).

The sliding window (adopted from NCCL, Apache-2.0)
---------------------------------------------------
The flag mechanism is **not** the probe's, and it isn't the naive "one flag
per (chunk, sender)" scheme either. It is NCCL's sliding window from
``src/device/prims_simple.h`` (NCCL, Apache-2.0, NVIDIA):

* ``NCCL_STEPS 8`` (``src/include/device.h:26``) -- fixed ring depth.
* ``connStepPtr`` / ``connStepCache`` (``prims_simple.h:40-41``) -- ONE
  monotonically increasing counter per connection, plus a **local cache**.
* ``waitPeer`` (``prims_simple.h:101-171``), wait condition on line 107:
  ``while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice)``
* ``postPeer`` (``:165-171``) -- publish with
  ``st_relaxed_sys_global``.

What's adopted is the **principle**, not the source: per directed
connection and per ring (RS, AG) there are two absolute counters,

``tail``  written by the producer  = "this many chunks I have deposited",
``head``  written by the consumer  = "this many chunks I have read".

Both are written by the producer into the **reader's** memory (a posted
PCIe write, cheap) and read there **locally**. A read from a foreign BAR
would be a round trip; that's the point where NCCL's assumption (NVLink,
coherent) doesn't hold for us, and where the direction is therefore not
arbitrary. ``connStepCache`` corresponds here to ``sTailC``/``sHeadC`` in
shared memory: as long as the cached value suffices, no re-read happens at
all.

The counters are **absolute over the transport's lifetime**, not reset per
round. A counter reset per round would have a gap right at the round
boundary: when rank A finishes its round ``n``, rank B may still be reading
in round ``n``, and a window comparison against a round-local counter would
have missed that. The counter therefore lives in ``stepDev`` and grows
by ``K`` per call.

What is NOT adopted, and why
-----------------------------------
* **Direct mode** (``prims_simple.h:39`` ``directBuff``, ``:135-148``).
  When P2P is available, NCCL bypasses the FIFO and writes directly into
  the target buffer. For us that fails not at the kernel but at bootstrap:
  only what was exported as a dma-buf and bound via ``mmap`` +
  ``cudaHostRegister`` during setup is reachable. The result tensor changes
  every call; binding it per call would mean ioctl + mmap + registration on
  the hot path. The assessment is in the report -- in short: what would be
  saved is one local VRAM pass, and against the PCIe bottleneck that's two
  orders of magnitude too cheap to explain the 0.86x. What this kernel does
  save, by contrast: the distribute step does **not read the result back
  from VRAM**, but writes it straight from the register it was just formed
  in (``reduceAndScatter``). ``bar1_mesh_kernel`` turns that into two
  passes (``reduceNPhase`` + ``scatterPhase``).
* **LL / LL128** (``prims_ll.h:111-119``, ``:154``). The flag embedded in
  the data word assumes you never observe "flag new, data stale". Over
  NVLink that holds, over PCIe into a write-combining aperture it is
  **unverified**, and on this rig it is established that the L2 of the
  receiving card is not coherent with incoming PCIe writes
  (``BEFUND_L2_NICHT_KOHAERENT.md``). Not built without evidence.

Direct mode and graph replay
-----------------------------------------
Direct mode (``SGLANG_BARLINK_BAR1_PIPE_DIRECT``) makes the compute kernel not
place the result into a slot that the receiver then copies out of, but
write it directly into that receiver's RESULT BUFFER. This buffer lives in
the exported region -- it is a ring slot, and the result tensor that
``all_reduce`` returns is a ``from_blob`` right over it. Whoever picks the
slot therefore determines the address of a tensor the caller ends up
holding; the slot can therefore NOT be diced up inside the kernel, but must
be fixed host-side before the kernel runs.

That is precisely what graph capturability hinged on. A freely running ring
index gets baked in at capture time, and multiple captures then run over
the same slots -- two graphs end up sharing one BAR1 slot and, on
alternating replay, deliver each other's numbers.

The fix is two pieces, and they belong together:

1. **Ownership instead of rotation** (:func:`result_slot_split`). The ring
   is split statically: the first ``eager_slots`` slots keep rotating for
   eager calls, everything above that is a pool from which each captured
   call site takes ONE slot and never returns it. That way no second
   capture can get the same slot.
2. **Release handshake** (flag family 4, ``resultReady``). A reserved slot is
   overwritten on EVERY replay; the spacing that the eager ring's two slots
   provide on their own is therefore gone. Every rank publishes its
   **generation** on entering a direct call -- "my result slot is free" --
   and anyone about to write into a foreign result slot waits for it. The
   generation counter lives in LOCAL VRAM and is advanced by the KERNEL, not
   the host: during a replay no host code runs, and a host-side counter
   would sit frozen.

Both are off as long as ``SGLANG_BARLINK_BAR1_PIPE_DIRECT_GRAPH`` is not set.
The kernel then follows the measured path byte for byte and never touches
flag family 4.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from sglang.srt.distributed.device_communicators import (
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
)
from sglang.srt.distributed.device_communicators.barlink_liveness import (
    bounded_barrier,
    bounded_device_sync,
    check_peers,
)

logger = logging.getLogger(__name__)

_ext = None

#: Largest rank group for which the argument arrays have room. Equal to
#: ``barlink_bar1_ext.MAX_RANGE`` -- the two kernels share the transport's peer
#: tables and must have the same limit.
MAX_RANGE = 8

#: Upper bound on the chunk count. Just a loop bound; the slot geometry
#: depends on ``T``, not on ``K``.
MAX_CHUNKS = 1024

#: Smallest useful depth. ``T = 1`` is NOT merely slow but deadlocked: at
#: ``T = 1``, sending chunk ``c`` and consuming it fall into the same loop
#: iteration, and the wait condition would stand before the publication it
#: waits for. See the derivation in :func:`pipe_plan`.
MIN_DEPTH = 2

#: Largest depth. From NCCL: ``NCCL_STEPS 8``.
MAX_DEPTH = 8


# ===========================================================================
# Geometry -- everything the transport needs to know about this kernel.
#
# Deliberately HERE and not in barlink_bar1.py: the kernel and its memory
# layout belong together, and barlink_bar1.py is meant only to plug in the
# results.
# ===========================================================================


def pipe_flags_extra(world: int) -> int:
    """Extra flag bytes for ``mesh_pipe``: ``5 * R * 256``.

    Five families of one 256-byte line per rank each:

    ==== ================= ============================================
    Fam. Name              written by / read by
    ==== ================= ============================================
    0    ``tailRS``        producer of the RS chunk / its receiver
    1    ``tailAG``        producer of the AG chunk / its receiver
    2    ``headRS``        consumer of the RS chunk / its producer
    3    ``headAG``        consumer of the AG chunk / its producer
    4    ``resultReady``   every rank on entering a direct call /
                           anyone writing into its result buffer
    ==== ================= ============================================

    Family 4 is the release handshake of the graph-capturable direct mode
    (:data:`RESULT_EAGER_SLOTS`). It sits BEHIND the four original lines, so
    every existing line offset stays byte-for-byte what it was; without
    direct mode it is never written and never read.

    **Independent of K and T.** That is the real payoff of the sliding
    window versus "one flag per (chunk, sender)": there the requirement
    would have been ``2 * K * R`` lines, and the flag region would have
    shifted with every size change.

    256 bytes per line, as everywhere in this transport: no false sharing
    between senders, none between families.
    """
    return 5 * world * 256


#: Family index of ``resultReady`` in the pipe flag region. Kept as a
#: constant because kernel and host side need the same number, and a second
#: version would be exactly the place where sender and receiver end up
#: pointing at different lines.
RESULT_READY_FAMILY = 4


def pipe_fbase(world: int, with_a2a: bool = True) -> int:
    """Offset of the pipe lines within the flag region.

    **Behind** mesh, ring, and a2a. That way all the measured topologies
    stay byte-for-byte where they were; the pipe only attaches at the end.
    The computation does NOT appear a second time in the kernel -- it is
    passed in as an argument, because a second version would be exactly the
    place where sender and receiver end up pointing at different lines.
    """
    return (2 + 2 * (world - 1) + (1 if with_a2a else 0)) * world * 256


def pipe_slot_default(world: int, chunk_target_bytes: int) -> int:
    """The slot size that the pipelined path actually needs.

    A slot carries ONE piece of ONE chunk, i.e. ``chunk / R``. How big a
    chunk gets is decided by ``pipe_chunk_bytes`` (the target size from
    which :func:`pipe_plan` derives its ``K``) -- not the transport's
    largest payload. The requirement therefore depends on the TARGET SIZE
    and is independent of ``max_bytes``, and it is exactly this independence
    that makes the geometry computable: the pipe region is an absolute byte
    count and no longer a fraction of the window.

    Rounded up to 16 bytes, because the access width is 128 bits: a slot
    whose size wasn't a multiple of 16 would force a misaligned start for
    the next one.

    **What used to be here, and why it's gone.** The pipe region used to be
    a full slot set like mesh, ring, and a2a, i.e. slot size
    ``chunk_max / T`` -- derived from the largest payload instead of from
    the chunk size a slot actually has to carry. At R=3, T=4, and this
    rig's 96 MiB window, that was 2047 KiB per slot against a 342 KiB
    requirement: a 32 MiB region for 5.4 MiB, and the difference was missing
    from the all_reduce slot. That was the measured 7.5% loss in the pipe
    arm of the lever benchmark for #293 (slot 8188 -> 6140 KiB, tipping
    point 2456 -> 1842 tokens, i.e. below the 2048 operating point).

    **Why a too-small slot isn't dangerous.**
    :func:`pipe_plan` searches its ``K`` ascending and checks each one with
    :func:`largest_chunk4` against the slot; if it finds none, the path
    declines via ``handles()``, and the kernel checks the same condition
    once more with ``TORCH_CHECK``. A tight slot costs coverage, never
    correctness. At the top end it covers ``k_max * chunk_target_bytes`` --
    at 64 chunks of 1 MiB that's 64 MiB, far more than this rig's window
    ever carries as payload.
    """
    if world < 2:
        return 0
    piece = -(-int(chunk_target_bytes) // int(world))
    return ((piece + 15) // 16) * 16


def pipe_range_bytes(world: int, depth: int, slot: int) -> int:
    """Bytes the pipe region occupies in the receive region.

    ``2 * T * (R-1)`` slots, rounded up to a page -- so that the region
    starting BEHIND the pipe region (the result ring) again starts on a
    page boundary. The start of the pipe region already sits on a page
    boundary, so every slot start is 16-byte-aligned as long as ``slot``
    is.
    """
    raw = 2 * int(depth) * (int(world) - 1) * int(slot)
    return ((raw + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


#: Page size. Deliberately its own constant and not an import from
#: ``barlink_bar1`` -- that module imports this one, and an import in both
#: directions at module level would be a cycle.
PAGE_SIZE = 4096

#: How many result buffers the ring holds. Two is the minimum with which
#: round ``n`` doesn't write into the buffer the caller from round ``n-1``
#: is still holding.
RESULT_RING_DEFAULT = 2

#: How many ring slots the eager path retains when the graph-capturable
#: direct mode is turned on -- the DEFAULT, not the fixed number anymore.
#:
#: Two is the minimum below which round ``n`` would write into the buffer
#: the caller from round ``n-1`` is still holding; the same derivation that
#: makes :data:`RESULT_RING_DEFAULT` two. But two is NOT a statement about how
#: many results any particular caller holds alive simultaneously -- and
#: that is exactly what tripped up the graph-capturable direct mode in the
#: lever benchmark for #293: the standard run's capture warmup holds more
#: than two, and ``SGLANG_BARLINK_BAR1_PIPE_RESULT_RING`` didn't help because
#: a larger ring only ever handed out GRAPH slots. The number is therefore
#: a parameter (``SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER``) and no longer a
#: constant. The default stays two: how many the standard run really needs
#: has not been measured, and a guessed larger number would not be a better
#: value, only a more expensive one.
RESULT_EAGER_SLOTS = 2


def result_slot_split(ring: int, graph_safe: bool,
                   eager_slots: int = RESULT_EAGER_SLOTS) -> tuple[int, int]:
    """``(eager_slots, graph_slots)`` for a ring of size ``ring``.

    Pure arithmetic, so it can be checked without a card. It is the result
    ring's ownership policy, and it lives here instead of scattered through
    the transport, because a slot whose owner two places disagree about
    leads to swapped results without a crash.

    **Without** ``graph_safe`` (the default), the WHOLE ring belongs to the
    eager path and rotates as before -- byte for byte the measured
    behavior.

    **With** ``graph_safe``, the ring is split statically:

    * the first ``eager_slots`` slots keep rotating for eager calls,
    * every slot above that is a pool from which one graph capture per
      call site takes ONE slot and never returns it.

    ``eager_slots`` is a parameter and not a constant, because it
    describes a property of the CALLER -- how many results it holds alive
    simultaneously -- not one of the transport. Pinned at
    :data:`RESULT_EAGER_SLOTS`, every larger ring handed out exclusively
    graph slots, while the capture WARMUP, which runs eager, failed on
    exactly the eager count.

    **Static and not growing along with it**, and that is the point: had
    the pool drawn from the upper end of the eager region while a capture
    was running, it could have grabbed a slot whose eager tensor the caller
    was still holding. The boundary is therefore fixed before the first
    call runs.

    If the ring isn't big enough for both, there are **zero** graph slots --
    not, say, one eager slot fewer. The caller reports this loudly and runs
    the ``direct=0`` path; an eager ring below two slots would be exactly
    the bug that :data:`RESULT_RING_DEFAULT` prevents.
    """
    ring = max(0, int(ring))
    eager = max(0, int(eager_slots))
    if not graph_safe:
        return ring, 0
    if ring <= eager:
        return ring, 0
    return eager, ring - eager


def result_eager_slot(previous: int, eager_slots: int) -> int:
    """The next eager ring slot after ``previous``.

    Factored out so the monotonicity and modulo guarantee can be checked
    without a card. ``previous = -1`` is the initial state.
    """
    if eager_slots <= 0:
        raise ValueError("eager ring has no slots")
    return (int(previous) + 1) % int(eager_slots)


def result_eager_free_slot(previous: int, eager_slots: int,
                           busy) -> Optional[int]:
    """The next FREE eager slot after ``previous``, or ``None``.

    ``busy[i]`` says whether the tensor last handed out from slot ``i``
    is still alive. Scanned round-robin starting at ``previous + 1``, so
    that the order among free slots stays the old one step for step -- the
    first candidate is exactly the one :func:`result_eager_slot` returns.

    **Why this searches at all.** Previously only the one successor was
    checked, and a hard abort followed if it was occupied. But the
    successor is the longest-unused one, i.e. the most likely candidate --
    not the only one. If the caller happens to be holding that one while
    not holding the others, aborting is simply wrong.

    ``None`` means "every slot is still being held". Then the call runs
    ``direct=0``, reported, with the same justification as an exhausted
    graph pool: ``direct=0`` is the measured control path, it costs the
    saved VRAM pass, not correctness. What must NOT happen is writing into
    a buffer that's still held -- and that is exactly what doesn't happen
    here.
    """
    if eager_slots <= 0:
        raise ValueError("eager ring has no slots")
    L = int(eager_slots)
    for step in range(L):
        i = (int(previous) + 1 + step) % L
        if not busy[i]:
            return i
    return None


def result_eager_slack(slot: int, counter: int, last_used, eager_slots: int) -> int:
    """How many direct calls have happened since ``slot`` was last used.

    The kernel waits, via ``resultSlack``, for the peer to have entered
    generation ``TARGET - resultSlack + 1`` -- a LARGER slack is the WEAKER
    condition. It must therefore be a lower bound on the actual reuse
    distance, never an upper bound.

    Under strict rotation over ``L`` slots the distance is exactly ``L``,
    and that's the value this returns -- step for step the measured one.
    As soon as :func:`result_eager_free_slot` skips a slot, the distance is
    smaller, and then the slack must be smaller too. Capped at ``L``,
    because a larger value only weakens the condition and gains nothing;
    at least ``1``, because ``0`` would disable the handshake entirely.

    ``last_used[i] is None`` means "never used yet" -- then there is nothing
    to overwrite and ``L`` is admissible.
    """
    L = max(1, int(eager_slots))
    previous = last_used[int(slot)]
    if previous is None:
        return L
    return max(1, min(L, int(counter) - int(previous)))


def result_graph_slot(assigned: int, eager_slots: int,
                    graph_slots: int) -> Optional[int]:
    """The ``assigned``-th graph slot, or ``None`` if the pool is empty.

    Graph slots sit BEHIND the eager slots and are handed out ascending.
    A slot once handed out is never handed out again: which captured graph
    is still alive cannot be determined from here, and a slot handed out
    twice would be exactly the bug the graph-capturable mode is meant to
    eliminate -- two captures sharing one BAR1 slot and, on alternating
    replay, delivering each other's numbers.

    ``None`` means: this call runs ``direct=0``. That is not a silent
    fallback -- the caller reports it -- and it is correct, because
    ``direct=0`` is the same measured control path.
    """
    if int(assigned) >= int(graph_slots):
        return None
    return int(eager_slots) + int(assigned)


def result_stride_bytes(max_bytes: int) -> int:
    """Distance between two result buffers in the ring.

    Rounded up to a page. That way every buffer starts on a page boundary,
    i.e. is 16-byte-aligned -- the condition every write into the WC
    aperture depends on -- and a buffer never shares a page with its
    neighbor.
    """
    return ((max_bytes + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


def result_ring_bytes(max_bytes: int, ring: int) -> int:
    """What the result ring costs in the BAR window: ``L * stride``.

    This is the computation that must be checked against the ACTUALLY
    mapped length, not against the raw size from sysfs. It isn't small: at
    ``R = 2`` with a2a and pipe turned on, ``6 (R-1) = 6`` slots of
    ``chunk_max`` each stand against ``L * R * chunk_max`` of result
    buffers -- at ``L = 2`` that's 10 units instead of 6, and the largest
    payload shrinks accordingly to 6/10. Direct mode isn't free; it trades
    BAR1 window for a saved VRAM pass and for eliminating the copy-out.
    """
    return max(0, int(ring)) * result_stride_bytes(max_bytes)


def pipe_window_requirement(world: int, depth: int, slot: int) -> int:
    """The **computed** requirement: ``2 * T * (R-1)`` slots.

    Not "one slot set, that'll do". This number is checked in ``handles()``
    against the ACTUALLY mapped length, and it is smaller than or equal to
    :func:`pipe_range_bytes` -- the difference is the waste from rounding
    up to a page. It is also exactly the region the kernel touches: the AG
    ring starts at ``T(R-1)*slot``, and the highest address within it is
    ``ringBytes + (T-1)*(R-1)*slot + (R-1)*slot``.
    """
    return 2 * int(depth) * (int(world) - 1) * int(slot)


def _chunk_bounds(n: int, j: int, parts: int) -> tuple[int, int]:
    """The same split as ``chunkBounds`` in the kernel: remainder up front.

    This lives here ONLY for planning and the byte-level check. The seam
    check in the kernel computes with ``chunkBounds`` itself -- a seam
    checked on both sides with the same wrong formula would never surface.
    """
    base = n // parts
    rest = n - base * parts
    length = base + (1 if j < rest else 0)
    offset = j * base + (j if j < rest else rest)
    return offset, length


def largest_chunk4(n4: int, world: int, k: int) -> int:
    """Largest piece (in 128-bit packets) that ever has to fit in a slot.

    Computed over ALL (chunk, rank) pairs, not via
    ``ceil(ceil(n4/K)/R)``. The closed form does happen to be correct here,
    but it is a second version of the same split, and that is exactly the
    error class this transport deliberately avoids in several places.
    """
    largest = 0
    for c in range(k):
        _, clen = _chunk_bounds(n4, c, k)
        for z in range(world):
            _, plen = _chunk_bounds(clen, z, world)
            if plen > largest:
                largest = plen
    return largest


def pipe_plan(nbytes: int, world: int, slot: int, depth: int,
              k_wanted: int, chunk_target_bytes: int,
              k_max: int = 64) -> Optional[int]:
    """The chunk count ``K`` for this payload, or ``None``.

    ``None`` means: this size isn't carried by the pipelined path -- the
    caller declines via ``handles()`` instead of silently computing
    something smaller.

    **Rank-uniform.** Every input is the same group-wide (``nbytes`` and
    ``world`` inherently, ``slot`` from setup, the rest from rank-uniform
    environment variables). Two ranks must never answer differently here.

    Two conditions, both hard:

    1. ``K >= T``. At ``K < T`` there would be slot classes that are never
       occupied, and the kernel enforces this anyway
       (``TORCH_CHECK(K >= TT)``). Additionally ``T >= 2`` is enforced (see
       :data:`MIN_DEPTH`).
    2. The largest piece must fit in a slot. Checked with
       :func:`largest_chunk4`, i.e. with the split itself.

    The default for ``K`` (``k_wanted <= 0``) is derived from
    ``chunk_target_bytes``. It is a **starting point for the benchmark
    series**, not a measurement result: the justification is that in
    ``MESSUNG_ALLES_IM_SELBEN_LAUF.md`` (three ranks, rig 1), ``mesh`` takes
    330.30 us at 1 MiB and the empty-round overhead is 25.74 us; a 256 KiB
    chunk therefore costs about 82 us of transfer against about 26 us of
    round trip, so the round trip stays hidden behind the traffic. On two
    cards and on the x8 port this is unmeasured.
    """
    if nbytes % 16 != 0:
        return None
    n4 = nbytes // 16
    if depth < MIN_DEPTH or depth > MAX_DEPTH:
        return None
    slot = int(slot)
    if slot <= 0 or slot % 16:
        return None
    upper_limit = min(k_max, MAX_CHUNKS, max(1, n4 // world))
    if upper_limit < depth:
        # Fewer chunks possible than the depth requires: too little payload
        # per rank. No lowering the depth -- it determines the slot
        # geometry, and that has been fixed since setup.
        return None

    if k_wanted > 0:
        candidates = [k_wanted]
    else:
        dst = max(1, chunk_target_bytes)
        k0 = max(depth, min(upper_limit, max(1, nbytes // dst)))
        # Try ascending: if the largest piece doesn't fit in a slot, MORE
        # splitting helps, never less.
        candidates = list(range(k0, upper_limit + 1))
    for k in candidates:
        if k < depth or k > upper_limit:
            continue
        if largest_chunk4(n4, world, k) * 16 <= slot:
            return int(k)
    return None


# ===========================================================================
# Kernel source
# ===========================================================================

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define BARLINK_PIPE_MAX_RANKS 8
#define BARLINK_PIPE_MAX_DEPTH 8

// Rate limit of the host abort probe inside the wait macros. Same value and
// same reasoning as BARLINK_BAR1_HOST_MASK in barlink_bar1_ext.py: the word sits
// in pinned, device-mapped host memory, so one read per 1024 spin iterations
// stays far below the flag loads the loop already issues, and a wait that is
// satisfied leaves through its own break before the probe is ever evaluated.
#define BARLINK_BAR1_HOST_MASK 1023u

#define K_1BLK   0
#define K_GRID 1
#define LA_CV    0
#define LA_MMIO  2

using u64 = unsigned long long;

// ===========================================================================
// Primitives -- duplicated VERBATIM from barlink_bar1_ext.py. See the module
// docstring: the duplication is deliberate (concurrent file ownership) and
// is a debt. Whoever changes something here must change it there too.
// ===========================================================================

__device__ __forceinline__ uint4 readV4(const void *p)
{
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void writeV4(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template<int LA>
__device__ __forceinline__ u64 readFlag(const u64 *p)
{
    u64 v;
    if (LA == LA_MMIO) {
        asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];"
                     : "=l"(v) : "l"(p) : "memory");
    } else {
        asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    }
    return v;
}

__device__ __forceinline__ void writeU64(void *p, u64 v)
{
    asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned int addF(unsigned int a, unsigned int b)
{
    return __float_as_uint(__uint_as_float(a) + __uint_as_float(b));
}

__device__ __forceinline__ unsigned int addH2(unsigned int a, unsigned int b)
{
    __half2 x = *(const __half2 *)&a;
    __half2 y = *(const __half2 *)&b;
    __half2 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
}

__device__ __forceinline__ unsigned int addBF2(unsigned int a, unsigned int b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 x = *(const __nv_bfloat162 *)&a;
    __nv_bfloat162 y = *(const __nv_bfloat162 *)&b;
    __nv_bfloat162 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
#else
    const __nv_bfloat16 *x = (const __nv_bfloat16 *)&a;
    const __nv_bfloat16 *y = (const __nv_bfloat16 *)&b;
    __nv_bfloat16 r[2];
    r[0] = __float2bfloat16(__bfloat162float(x[0]) + __bfloat162float(y[0]));
    r[1] = __float2bfloat16(__bfloat162float(x[1]) + __bfloat162float(y[1]));
    return *(const unsigned int *)r;
#endif
}

template<typename T>
__device__ __forceinline__ uint4 addV4(uint4 a, uint4 b);

template<>
__device__ __forceinline__ uint4 addV4<float>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addF(a.x, b.x); r.y = addF(a.y, b.y);
    r.z = addF(a.z, b.z); r.w = addF(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__half>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addH2(a.x, b.x); r.y = addH2(a.y, b.y);
    r.z = addH2(a.z, b.z); r.w = addH2(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__nv_bfloat16>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addBF2(a.x, b.x); r.y = addBF2(a.y, b.y);
    r.z = addBF2(a.z, b.z); r.w = addBF2(a.w, b.w);
    return r;
}

// Chunk j out of n units over `parts`; remainder onto the LEADING chunks.
// Bit-identical to barlink_bar1_ext.py::chunkBounds -- the pipe splits
// twice with the same rule (first into K chunks, then into R pieces).
__device__ __host__ __forceinline__ void chunkBounds(int n, int j, int parts,
                                                      int *off, int *len)
{
    const int base = n / parts;
    const int rest  = n - base * parts;
    *len = base + (j < rest ? 1 : 0);
    *off = j * base + (j < rest ? j : rest);
}

template<int GRID>
__device__ __forceinline__ void barrier(void)
{
    if (GRID == K_GRID) cg::this_grid().sync();
    else                  __syncthreads();
}

// ===========================================================================
// Inverse of chunkBounds: for a flat index j in [0, n), the piece z and
// the offset within it. Closed form, no scan.
//
// It allows the reduce-scatter send to run over ALL targets in ONE flat
// loop -- different warps then write to different cards at the same time,
// instead of target by target. Same technique as in the a2a kernel, there
// via a prefix scan in shared memory; here it works in closed form because
// the split is uniform.
//
// base == 0 (n < parts) is NOT the special case it appears to be: then
// rest == n and bound == n, so the else branch is unreachable. A division
// by 0 cannot occur.
// ===========================================================================
__device__ __forceinline__ void pieceOf(int n, int parts, int j,
                                          int *z, int *loc)
{
    const int base = n / parts;
    const int rest  = n - base * parts;
    const int bound = rest * (base + 1);
    if (j < bound) {
        const int q = j / (base + 1);
        *z = q; *loc = j - q * (base + 1);
    } else {
        const int d = j - bound;
        const int q = d / base;               // base > 0, s.o.
        *z = rest + q; *loc = d - q * base;
    }
}

// ===========================================================================
// Arguments
// ===========================================================================
struct PipeArgs {
    const uint4 *in;
    uint4       *out;
    u64         *roundDev;
    u64         *stepDev;   // absolute chunk counter, survives the call
    // Generation counter of direct mode. LOCAL VRAM, not in the window: a
    // local counter is coherent with the own reads, and only the peer's
    // VIEW of it needs the flag protocol. It lives on the DEVICE because it
    // must keep counting on every graph replay -- a host-side counter gets
    // baked in at capture time and sits frozen during replay.
    u64         *resultGenDev;
    unsigned int *ctlStatus;
    unsigned int *abortDev;
    // Host-set abort word (pinned, device-mapped). nullptr = absent, and then
    // the wait macros keep only their cycle deadline. Set to 1 by the peer
    // watchdog when a peer process is provably gone.
    const unsigned int *abortHost;
    int          n4;
    int          R;
    int          rank;
    int          K;            // chunk count of this call
    int          TT;           // ring depth = slot classes per phase
    int          PP;           // schedule lead, 2 <= PP <= TT
    int          ack;     // 1 = publish head (default)
    long long    slot4;     // slot size in 128-bit packets
    long long    class4;      // distance between two slot classes = (R-1)*slot4
    u64          capCycles;

    int          direct;       // 1 = allgather writes into the result buffer
    // Release handshake of direct mode. 0 = off (the measured legacy
    // behavior, flag family 4 is never touched). > 0 = on, and the number
    // is the DISTANCE IN GENERATIONS after which a result slot is reused:
    // 1 for a permanently reserved graph slot, otherwise the number of
    // eager ring slots.
    int          resultSlack;

    // Payload slots, class 0. Class t sits at + t*class4.
    uint4       *sendRS[BARLINK_PIPE_MAX_RANKS];
    uint4       *sendAG[BARLINK_PIPE_MAX_RANKS];
    const uint4 *recvRS[BARLINK_PIPE_MAX_RANKS];
    const uint4 *recvAG[BARLINK_PIPE_MAX_RANKS];
    // Direct mode: peer z's result buffer for THIS round. It lives in the
    // same exported region as the slots, so it's reachable without any
    // registration on the hot path. The offset within the buffer is the
    // same as in the own `out` -- the allgather doesn't copy anything
    // around, it writes straight to the final destination.
    uint4       *resultTo[BARLINK_PIPE_MAX_RANKS];

    // Counters. [0] = RS ring, [1] = AG ring.
    u64         *tailTo [2][BARLINK_PIPE_MAX_RANKS];  // me -> receiver z
    const u64   *tailFrom[2][BARLINK_PIPE_MAX_RANKS];  // local, from sender s
    u64         *headTo [2][BARLINK_PIPE_MAX_RANKS];  // me -> producer s
    const u64   *headFrom[2][BARLINK_PIPE_MAX_RANKS];  // local, from receiver z

    // Release of the result slot (flag family 4). Same direction as
    // tail/head: written at the peer, read locally.
    u64         *resultReadyTo [BARLINK_PIPE_MAX_RANKS]; // me -> peer z
    const u64   *resultReadyFrom[BARLINK_PIPE_MAX_RANKS]; // local, from peer z
};

// ===========================================================================
// The kernel
// ===========================================================================
template<typename T, int LA, int GRID>
__global__ void bar1_mesh_pipe_kernel(PipeArgs A)
{
    const int tid = (GRID == K_GRID)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GRID)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool isFirst = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rank, K = A.K, TT = A.TT, PP = A.PP;
    const u64  round  = *(const volatile u64 *)A.roundDev + 1ull;
    const u64  base  = *(const volatile u64 *)A.stepDev;
    // The handshake only runs when both come together: direct mode AND a
    // slack value. Both are group-uniform, so either all ranks enter the
    // handshake or none do -- a rank that skipped it while the others wait
    // for its flag would be a hang, not a wrong result.
    const bool resultHandshake = (A.direct != 0) && (A.resultSlack > 0);
    const u64  resultGen  = resultHandshake
                             ? (*(const volatile u64 *)A.resultGenDev + 1ull)
                             : 0ull;

    // Everything indexed with a RUNNING index goes into shared memory. A
    // dynamically indexed field of the argument struct forces nvcc to put
    // the whole parameter block into local memory per thread -- the same
    // reason as in the a2a kernel.
    __shared__ uint4       *sSendRS[BARLINK_PIPE_MAX_RANKS];
    __shared__ uint4       *sSendAG[BARLINK_PIPE_MAX_RANKS];
    __shared__ const uint4 *sRecvRS[BARLINK_PIPE_MAX_RANKS];
    __shared__ const uint4 *sRecvAG[BARLINK_PIPE_MAX_RANKS];
    __shared__ uint4       *sResultTo [BARLINK_PIPE_MAX_RANKS];
    // Counter pointers and their local caches (NCCL: connStepCache). Only
    // the first thread touches them; they still live in shared memory,
    // because otherwise 4*8 u64 would pile up as registers per thread.
    __shared__ u64         *sTailTo [2][BARLINK_PIPE_MAX_RANKS];
    __shared__ const u64   *sTailFrom[2][BARLINK_PIPE_MAX_RANKS];
    __shared__ u64         *sHeadTo [2][BARLINK_PIPE_MAX_RANKS];
    __shared__ const u64   *sHeadFrom[2][BARLINK_PIPE_MAX_RANKS];
    __shared__ u64          sTailC[2][BARLINK_PIPE_MAX_RANKS];
    __shared__ u64          sHeadC[2][BARLINK_PIPE_MAX_RANKS];
    __shared__ u64         *sReadyTo [BARLINK_PIPE_MAX_RANKS];
    __shared__ const u64   *sReadyFrom[BARLINK_PIPE_MAX_RANKS];
    __shared__ u64          sReadyC  [BARLINK_PIPE_MAX_RANKS];
    __shared__ int          abortS;

    if (threadIdx.x == 0) {
        for (int i = 0; i < R; ++i) {
            sSendRS[i] = A.sendRS[i]; sSendAG[i] = A.sendAG[i];
            sRecvRS[i] = A.recvRS[i]; sRecvAG[i] = A.recvAG[i];
            sResultTo[i]  = A.resultTo[i];
            for (int ph = 0; ph < 2; ++ph) {
                sTailTo [ph][i] = A.tailTo [ph][i];
                sTailFrom[ph][i] = A.tailFrom[ph][i];
                sHeadTo [ph][i] = A.headTo [ph][i];
                sHeadFrom[ph][i] = A.headFrom[ph][i];
                // The cache's initial value is BASE, not 0: what happened
                // before this call is already reflected in stepDev. A
                // cache starting at 0 wouldn't be wrong, but it would force
                // a re-read per connection in the first loop iteration.
                sTailC[ph][i] = base;
                sHeadC[ph][i] = base;
            }
            sReadyTo [i] = A.resultReadyTo [i];
            sReadyFrom[i] = A.resultReadyFrom[i];
            // Initial value 0, and NOT `resultGen - 1`: the optimistic initial
            // value would be exactly the assumption the handshake is
            // supposed to verify. It costs ONE local read per peer and per
            // call -- the line sits in the own VRAM, not behind PCIe.
            sReadyC  [i] = 0ull;
        }
        abortS = 0;
    }
    __syncthreads();

    if (GRID == K_GRID && isFirst) {
        *(volatile unsigned int *)A.abortDev = 0u;
        __threadfence();
    }

    // -- Publish the release, as early as possible --------------------------
    //
    // "I have entered generation g." Because this kernel sits on the same
    // stream as everything that read generation g-1's result, entering it
    // simultaneously means: my result slot is free. The peer may write into
    // it from now on.
    //
    // Before the barrier and before the loop, so the PCIe round-trip time
    // (about 3 us, BEFUND_L2_UMGEHBAR.md) disappears behind the
    // reduce-scatter phase. The wait only happens once the first direct
    // write is due -- i.e. PP-1 loop iterations later.
    if (resultHandshake && isFirst) {
        for (int z = 0; z < R; ++z)
            if (z != r) writeU64(sReadyTo[z], resultGen);
        __threadfence_system();
    }
    barrier<GRID>();

    // -- Wait conditions ------------------------------------------------------
    //
    // Both FIRST read the cache and only touch the line if it isn't
    // sufficient. Over PCIe, every re-read would be a round trip -- without
    // the cache, the window would cost more than it gains (NCCL
    // prims_simple.h:40-41, 107).
    //
    // `dst` is always an ABSOLUTE step (base + c + 1).

#define PIPE_WAIT_DATA(PH, TARGET)                                             \
    do {                                                                       \
        const u64 _z = (TARGET);                                                 \
        long long _t0 = clock64();                                             \
        unsigned int _probe = 0u;                                              \
        for (;;) {                                                             \
            bool _all = true;                                                 \
            for (int _s = 0; _s < R; ++_s) {                                   \
                if (_s == r) continue;                                         \
                if (sTailC[PH][_s] < _z) {                                     \
                    sTailC[PH][_s] = readFlag<LA>(sTailFrom[PH][_s]);        \
                    if (sTailC[PH][_s] < _z) { _all = false; break; }         \
                }                                                              \
            }                                                                  \
            if (_all) break;                                                  \
            if ((u64)(clock64() - _t0) > A.capCycles) { abortS = 1; break; } \
            if (A.abortHost != nullptr &&                                    \
                ((++_probe & BARLINK_BAR1_HOST_MASK) == 0u) &&                  \
                *(const volatile unsigned int *)A.abortHost != 0u)           \
                { abortS = 1; break; }                                       \
        }                                                                      \
    } while (0)

    // Window: receiver z has read at least (TARGET - TT) chunks. Unsigned
    // comparison in the form `cache + TT < dst` -- exactly like NCCL, so
    // that dst < TT does not underflow.
#define PIPE_WAIT_WINDOW(PH, TARGET)                                           \
    do {                                                                       \
        if (A.ack) {                                                      \
            const u64 _z = (TARGET);                                             \
            long long _t0 = clock64();                                         \
            unsigned int _probe = 0u;                                          \
            for (;;) {                                                         \
                bool _all = true;                                             \
                for (int _q = 0; _q < R; ++_q) {                               \
                    if (_q == r) continue;                                     \
                    if (sHeadC[PH][_q] + (u64)TT < _z) {                       \
                        sHeadC[PH][_q] = readFlag<LA>(sHeadFrom[PH][_q]);    \
                        if (sHeadC[PH][_q] + (u64)TT < _z) { _all = false; break; } \
                    }                                                          \
                }                                                              \
                if (_all) break;                                              \
                if ((u64)(clock64() - _t0) > A.capCycles) { abortS = 1; break; } \
                if (A.abortHost != nullptr &&                                \
                    ((++_probe & BARLINK_BAR1_HOST_MASK) == 0u) &&              \
                    *(const volatile unsigned int *)A.abortHost != 0u)       \
                    { abortS = 1; break; }                                   \
            }                                                                  \
        }                                                                      \
    } while (0)

    // Peer z has released its result slot for generation TARGET.
    //
    // The slot is reused every ``resultSlack`` generations; whoever writes
    // into generation TARGET therefore overwrites the contents of generation
    // ``TARGET - resultSlack``. It is released once z has entered generation
    // ``TARGET - resultSlack + 1``. The same unsigned-safe form as
    // PIPE_WAIT_WINDOW: the slack sits on the LEFT side, so that
    // TARGET < resultSlack does not underflow.
    //
    // Why this condition is needed at all, and the AG acknowledgement isn't
    // enough: at ``resultSlack >= 2`` it follows from the AG window (the
    // predecessor is two calls back, and even ONE call of distance is
    // already enforced by tail/head). For a permanently reserved graph
    // slot, ``resultSlack == 1``: the same slot is overwritten on EVERY
    // replay, and then there is no distance left that the AG window could
    // enforce.
#define PIPE_WAIT_RESULT_FREE(TARGET)                                               \
    do {                                                                       \
        const u64 _z = (TARGET);                                                 \
        const u64 _sl = (u64)(A.resultSlack - 1);                                 \
        long long _t0 = clock64();                                             \
        unsigned int _probe = 0u;                                              \
        for (;;) {                                                             \
            bool _all = true;                                                 \
            for (int _q = 0; _q < R; ++_q) {                                   \
                if (_q == r) continue;                                         \
                if (sReadyC[_q] + _sl < _z) {                                 \
                    sReadyC[_q] = readFlag<LA>(sReadyFrom[_q]);            \
                    if (sReadyC[_q] + _sl < _z) { _all = false; break; }     \
                }                                                              \
            }                                                                  \
            if (_all) break;                                                  \
            if ((u64)(clock64() - _t0) > A.capCycles) { abortS = 1; break; } \
            if (A.abortHost != nullptr &&                                    \
                ((++_probe & BARLINK_BAR1_HOST_MASK) == 0u) &&                  \
                *(const volatile unsigned int *)A.abortHost != 0u)           \
                { abortS = 1; break; }                                       \
        }                                                                      \
    } while (0)

    // Once per call, not per loop iteration: the condition depends on the
    // generation, and that doesn't change within a call. Only thread 0
    // evaluates it, so an ordinary local variable is enough.
    bool resultFreeSeen = !resultHandshake;

    // -- Loop -----------------------------------------------------------------
    //
    // Three stages, offset by PP-1 and PP loop iterations respectively:
    //
    //   cs = i        reduce-scatter piece of chunk cs to all peers
    //   cr = i-(PP-1) reduce chunk cr AND distribute the result immediately
    //   cg = i-PP     adopt allgather pieces of chunk cg
    //
    // So a rank pushes out chunk i while reducing chunk i-PP+1 and
    // collecting chunk i-PP. There is NO grid-wide lockstep per phase left
    // between the stages anymore; the two barriers per loop iteration only
    // separate "the first thread has waited" from "everyone works", and
    // "everyone is done" from "the first thread publishes".
    //
    // WHY LEAD AND RING DEPTH ARE SEPARATE
    // -----------------------------------------
    // They used to be the same (PP == TT), and that was a design mistake.
    // The window condition for sending chunk i requires the receiver to
    // have consumed chunk i-TT; for it, that consumption happens in loop
    // iteration (i-TT)+(PP-1). So the receiver is allowed to lag by
    //
    //     TT - PP + 1  loop iterations
    //
    // before the sender blocks. With PP == TT that is exactly ONE
    // iteration -- the ring would be TT slots deep, but the schedule would
    // consume them again immediately, and two cards of unequal speed would
    // effectively run in lockstep. Exactly the temporal decoupling that
    // gives the host path its one genuine strength (the sender dumps into
    // RAM and is done) would be thrown away by that.
    //
    // With PP = 2 (the minimum that pipelines at all) and TT = 4, that's
    // three iterations of offset. This is NCCL's design: eight steps in the
    // ring (NCCL_STEPS, src/include/device.h:26) against two steps per
    // slice (ALLREDUCE_SLICESTEPS, src/include/collectives.h:19) -- ring
    // depth and lead are different numbers there too.
    for (int i = 0; i < K + PP; ++i) {
        const int cs = i;
        const int cr = i - (PP - 1);
        const int cg = i - PP;
        // SLOT CLASS FROM THE ABSOLUTE STEP, not from this call's chunk
        // index. With `c % TT`, the class <-> slot mapping would shift at
        // every round boundary where K isn't a multiple of TT -- the
        // window condition "head + TT >= step" would then be talking about
        // a different slot than the one being written to. Example TT=2,
        // K=3: round n places steps 1,2,3 into classes 0,1,0; round n+1
        // places step 4 into class 0, while step 3 still sits there -- but
        // the condition would only have required step 2. With the absolute
        // step, the class is `(step-1) mod TT` and the condition always
        // hits the correct predecessor.
        const int ts = (int)((base + (u64)cs) % (u64)TT);
        const int tr = (cr >= 0) ? (int)((base + (u64)cr) % (u64)TT) : 0;
        const int tg = (cg >= 0) ? (int)((base + (u64)cg) % (u64)TT) : 0;

        // (a) first thread only: wait
        if (isFirst) {
            if (cs < K)             PIPE_WAIT_WINDOW(0, base + (u64)cs + 1ull);
            if (!abortS && cr >= 0 && cr < K) {
                                    PIPE_WAIT_DATA  (0, base + (u64)cr + 1ull);
                if (!abortS)      PIPE_WAIT_WINDOW(1, base + (u64)cr + 1ull);
                // Only here, right before this call's first direct write --
                // the reduce-scatter iterations before it have already
                // carried the peers' flags across the wire.
                if (!abortS && !resultFreeSeen) {
                    PIPE_WAIT_RESULT_FREE(resultGen);
                    resultFreeSeen = true;
                }
            }
            if (!abortS && cg >= 0) PIPE_WAIT_DATA(1, base + (u64)cg + 1ull);
            if (abortS && GRID == K_GRID) {
                *(volatile unsigned int *)A.abortDev = 1u;
                __threadfence();
            }
        }
        barrier<GRID>();
        {
            const int ab = (GRID == K_1BLK)
                               ? abortS
                               : (int)*(volatile unsigned int *)A.abortDev;
            if (ab) {
                // ALL blocks return together -- a single block that no
                // longer reaches a grid.sync() hangs the rest.
                if (isFirst) {
                    *A.ctlStatus = 1u;
                    *(volatile u64 *)A.roundDev   = round;
                    *(volatile u64 *)A.stepDev = base + (u64)K;
                    // ALSO on the abort path. The generation counter must
                    // stay group-uniform; a rank that left it standing on
                    // abort while another kept advancing it would wait, on
                    // the next call, for a generation that never comes.
                    // Same reasoning as for roundDev and stepDev right
                    // above.
                    if (resultHandshake) *(volatile u64 *)A.resultGenDev = resultGen;
                    __threadfence_system();
                }
                return;
            }
        }
        // WITHOUT THIS FENCE, THE REDUCTION READS STALE LINES.
        //
        // `readV4` is `ld.global.cv`, and `ld.global.cv` alone does NOT see
        // incoming PCIe writes on this rig: the receiving card's L2 is not
        // coherent with them (BEFUND_L2_NICHT_KOHAERENT.md;
        // BEFUND_L2_UMGEHBAR.md, line (1): 46.5 million read attempts,
        // never visible). It only becomes visible with
        // `__threadfence_system()` BEFORE it -- line (6) of the same
        // benchmark, SASS `MEMBAR.SC.SYS` + `CCTL.IVALL`, six read
        // attempts. The barrier alone is not enough: `__syncthreads()` and
        // `grid.sync()` are device-wide, not system-wide.
        //
        // From ALL threads, not just the first: the first thread has seen
        // the flag, but every thread reads its own slots right after.
        // `bar1_mesh_kernel` has the same fence at the same spot
        // (barlink_bar1_ext.py, behind every abort check).
        __threadfence_system();

        // (b) all threads: this loop iteration's work.
        //
        // The three stages sit side by side without a barrier. They don't
        // get in each other's way:
        //   - cs writes into FOREIGN RS slots,
        //   - cr reads OWN RS slots and writes out + FOREIGN AG slots,
        //   - cg reads OWN AG slots and writes out.
        // cr and cg both write out, but into different chunks (cr = cg+1).
        // cs and cg sit in the same slot class ((i) mod TT == (i-TT) mod
        // TT), but in different rings (RS versus AG) and on different
        // cards (foreign versus own).
        if (cs < K) {
            int coff, clen;
            chunkBounds(n4, cs, K, &coff, &clen);
            const long long kv = (long long)ts * A.class4;
            for (int j = tid; j < clen; j += nth) {
                int z, loc;
                pieceOf(clen, R, j, &z, &loc);
                if (z == r) continue;      // own piece stays in `in`
                writeV4(sSendRS[z] + kv + loc, A.in[coff + j]);
            }
        }
        if (cr >= 0 && cr < K) {
            int coff, clen, poff, plen;
            chunkBounds(n4, cr, K, &coff, &clen);
            chunkBounds(clen, r, R, &poff, &plen);
            const long long kv = (long long)tr * A.class4;
            const uint4 *q = A.in  + coff + poff;
            uint4       *o = A.out + coff + poff;
            for (int j = tid; j < plen; j += nth) {
                uint4 s = q[j];
                for (int p = 0; p < R; ++p) {
                    if (p == r) continue;
                    s = addV4<T>(s, readV4(sRecvRS[p] + kv + j));
                }
                // Form the result ONCE, then write it from the register
                // both into the own output buffer and to all peers. The
                // probe kernel reads `out` a second time for this
                // (reduceNPhase, then scatterPhase).
                // In direct mode, `out` lives in the exported region, and
                // peers write into that SAME region via PCIe. An ordinary
                // store would leave a dirty L2 line behind; a line is 128
                // bytes, but a piece boundary is only 16-byte-granular, so
                // the line at the edge of my piece also contains bytes the
                // neighbor writes via PCIe. If it gets written back later,
                // it overwrites their bytes -- wrong numbers without a
                // crash. `st.wt` leaves no dirty line behind. Outside direct
                // mode, `out` is an ordinary torch buffer that no peer
                // touches; there the normal store remains.
                if (A.direct) writeV4(o + j, s);
                else          o[j] = s;
                if (A.direct) {
                    // DIRECT MODE: straight to the final destination in the
                    // receiver's result buffer, not into a slot. That
                    // eliminates the receiver's read-out-and-copy pass
                    // ENTIRELY -- not halfway. Same offset as in the own
                    // `out`, hence 16-byte-aligned, because the buffer
                    // starts on a page boundary.
                    const int dst = coff + poff + j;
                    for (int z = 0; z < R; ++z) {
                        if (z == r) continue;
                        writeV4(sResultTo[z] + dst, s);
                    }
                } else {
                    for (int z = 0; z < R; ++z) {
                        if (z == r) continue;
                        writeV4(sSendAG[z] + kv + j, s);
                    }
                }
            }
        }
        if (cg >= 0 && !A.direct) {
            int coff, clen;
            chunkBounds(n4, cg, K, &coff, &clen);
            const long long kv = (long long)tg * A.class4;
            for (int j = tid; j < clen; j += nth) {
                int s, loc;
                pieceOf(clen, R, j, &s, &loc);
                if (s == r) continue;      // own piece is already in out
                A.out[coff + j] = readV4(sRecvAG[s] + kv + loc);
            }
        }
        // In direct mode there is NOTHING to do here: the peers' pieces are
        // already at their final destination. The wait condition on tailAG
        // in phase (a) still remains, though -- it's what ensures the
        // kernel doesn't end before the peers' last packets have arrived.
        // Without it the call would return a half-filled result tensor.
        __threadfence_system();
        barrier<GRID>();

        // (c) first thread only: publish.
        //
        // Order: payload first (above, before the barrier and the fence),
        // then the counter. Payload and counter go to the SAME target
        // card, so per-requester/completer-pair PCIe ordering holds --
        // the same assumption mesh/ring/a2a already rely on.
        if (isFirst) {
            if (cs < K) {
                const u64 m = base + (u64)cs + 1ull;
                for (int z = 0; z < R; ++z)
                    if (z != r) writeU64(sTailTo[0][z], m);
            }
            if (cr >= 0 && cr < K) {
                const u64 m = base + (u64)cr + 1ull;
                for (int z = 0; z < R; ++z) {
                    if (z == r) continue;
                    writeU64(sTailTo[1][z], m);          // AG data is in place
                    if (A.ack) writeU64(sHeadTo[0][z], m);  // RS read
                }
            }
            if (cg >= 0 && A.ack) {
                const u64 m = base + (u64)cg + 1ull;
                for (int z = 0; z < R; ++z)
                    if (z != r) writeU64(sHeadTo[1][z], m);      // AG read
            }
            __threadfence_system();
        }
    }

    barrier<GRID>();
    if (isFirst) {
        *(volatile u64 *)A.roundDev   = round;
        *(volatile u64 *)A.stepDev = base + (u64)K;
        if (resultHandshake) *(volatile u64 *)A.resultGenDev = resultGen;
        __threadfence_system();
    }
#undef PIPE_WAIT_DATA
#undef PIPE_WAIT_WINDOW
#undef PIPE_WAIT_RESULT_FREE
}

// ===========================================================================
// Host side
// ===========================================================================

static int gridSize(const void *fn, int threads, int n4)
{
    int perSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, fn, threads, 0)
            != cudaSuccess || perSM < 1)
        return 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess || sms < 1)
        return 1;
    int g = perSM * sms;
    int needed = (n4 + threads - 1) / threads;
    if (needed < 1) needed = 1;
    if (needed < g) g = needed;
    if (g < 1) g = 1;
    return g;
}

template<typename T>
static void startPipe(int kernel_variant, int la, PipeArgs &A, int threads,
                       int grid_n4, cudaStream_t stream)
{
#define BARLINK_PIPE_LAUNCH(LA)                                                  \
    do {                                                                       \
        if (kernel_variant == K_GRID) {                                                \
            const void *fn =                                                   \
                (const void *)bar1_mesh_pipe_kernel<T, LA, K_GRID>;          \
            int g = gridSize(fn, threads, grid_n4);                       \
            dim3 gd((unsigned)g), bd((unsigned)threads);                       \
            void *args[1] = { &A };                                            \
            cudaError_t e = cudaLaunchCooperativeKernel(fn, gd, bd, args, 0,   \
                                                        stream);                \
            TORCH_CHECK(e == cudaSuccess,                                      \
                        "barlink-bar1 pipe: cudaLaunchCooperativeKernel -> ",    \
                        cudaGetErrorString(e));                                \
            return;                                                            \
        }                                                                      \
        bar1_mesh_pipe_kernel<T, LA, K_1BLK><<<1, threads, 0, stream>>>(A);     \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "barlink-bar1 pipe: kernel launch failed");                  \
        return;                                                                \
    } while (0)

    if (la == LA_MMIO) BARLINK_PIPE_LAUNCH(LA_MMIO);
    else               BARLINK_PIPE_LAUNCH(LA_CV);
#undef BARLINK_PIPE_LAUNCH
}

// A tensor OVER the exported receive region, without a copy.
//
// This is the pivot point of direct mode: `all_reduce` operates
// out-of-place and returns a fresh tensor -- so WE decide where it lives.
// Instead of letting torch pick a buffer the neighbors don't know about,
// the result tensor points into a ring buffer exported at setup time. The
// caller gets a perfectly ordinary tensor; it just happens to live in
// memory the neighboring cards may write into directly. No registration on
// the hot path, no copying.
//
// `at::from_blob` WITHOUT a deleter: the memory belongs to the transport
// and is freed by `close()`. A deleter would be the bug here -- torch
// would try to hand a cuMemMap address back to the caching allocator.
at::Tensor bar1_result_tensor(int64_t ptr, at::Tensor like)
{
    TORCH_CHECK(ptr != 0, "barlink-bar1 pipe: result buffer is a null pointer");
    TORCH_CHECK((ptr % 16) == 0,
                "barlink-bar1 pipe: result buffer ", ptr,
                " is not 16-byte-aligned -- every write into the WC "
                "aperture is 128 bits wide");
    return at::from_blob((void *)(uintptr_t)ptr, like.sizes(),
                         like.options());
}

void bar1_mesh_pipe(at::Tensor inp, at::Tensor out,
                    int64_t rank, int64_t world,
                    std::vector<int64_t> peer_payload,
                    std::vector<int64_t> peer_flag,
                    std::vector<int64_t> peer_result,
                    int64_t own_payload, int64_t own_flag,
                    int64_t slot, int64_t off_pipe, int64_t fbase_pipe,
                    int64_t k_chunks, int64_t depth, int64_t lead,
                    int64_t ack, int64_t direct, int64_t result_slack,
                    at::Tensor round_dev, at::Tensor step_dev,
                    at::Tensor result_gen_dev, at::Tensor ctl_dev,
                    int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                    int64_t load_shape, int64_t abort_host)
{
    const int R = (int)world, r = (int)rank;
    const int K = (int)k_chunks, TT = (int)depth, PP = (int)lead;
    TORCH_CHECK(R >= 2 && R <= BARLINK_PIPE_MAX_RANKS,
                "barlink-bar1 pipe: world ", R, " outside 2..",
                BARLINK_PIPE_MAX_RANKS);
    TORCH_CHECK(r >= 0 && r < R, "barlink-bar1 pipe: rank out of range");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "barlink-bar1 pipe: only contiguous tensors");
    TORCH_CHECK(inp.numel() == out.numel() &&
                inp.scalar_type() == out.scalar_type(),
                "barlink-bar1 pipe: in and out do not match");
    TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                "barlink-bar1 pipe: in and out must not be the same -- the "
                "reduction still reads 'in' while 'out' is already set");
    TORCH_CHECK((int64_t)peer_payload.size() == world &&
                (int64_t)peer_flag.size() == world &&
                (int64_t)peer_result.size() == world,
                "barlink-bar1 pipe: peer table has the wrong length");
    // PP >= 2: at PP == 1, sending chunk c and consuming it would fall into
    // the same loop iteration, and the wait condition would stand BEFORE
    // the publication it waits for -- a self-deadlock that no timeout cap
    // turns into an error, only into a slow one.
    // PP <= TT: the ring must be at least as deep as the lead, otherwise
    // the schedule overtakes the slots.
    TORCH_CHECK(TT >= 2 && TT <= BARLINK_PIPE_MAX_DEPTH,
                "barlink-bar1 pipe: ring depth ", TT, " outside 2..",
                BARLINK_PIPE_MAX_DEPTH);
    TORCH_CHECK(PP >= 2 && PP <= TT,
                "barlink-bar1 pipe: lead ", PP, " outside 2..", TT,
                " -- it must not exceed the ring depth");
    TORCH_CHECK(K >= TT, "barlink-bar1 pipe: K=", K, " < T=", TT,
                " -- fewer chunks than ring depth");
    TORCH_CHECK(slot > 0 && (slot % 16) == 0,
                "barlink-bar1 pipe: slot size ", slot,
                " is not a positive multiple of 16");
    TORCH_CHECK((off_pipe % 16) == 0,
                "barlink-bar1 pipe: off_pipe ", off_pipe,
                " is not a multiple of 16 -- every write into the WC "
                "aperture is 128 bits wide and must be aligned");

    const size_t nbytes = (size_t)inp.numel() * (size_t)inp.element_size();
    TORCH_CHECK(nbytes % 16 == 0,
                "barlink-bar1 pipe: payload ", nbytes,
                " is not a multiple of 16 bytes");
    TORCH_CHECK(((uintptr_t)inp.data_ptr() % 16) == 0 &&
                ((uintptr_t)out.data_ptr() % 16) == 0,
                "barlink-bar1 pipe: buffer not aligned to 16 bytes");
    const int n4 = (int)(nbytes / 16);
    TORCH_CHECK(n4 >= R, "barlink-bar1 pipe: fewer than one packet per rank");

    // Seam check. Computed with chunkBounds ITSELF, over all (chunk, rank)
    // pairs -- not with a second, closed-form version.
    int maxPiece = 0, maxChunk = 0;
    for (int c = 0; c < K; ++c) {
        int coff, clen;
        chunkBounds(n4, c, K, &coff, &clen);
        if (clen > maxChunk) maxChunk = clen;
        for (int z = 0; z < R; ++z) {
            int poff, plen;
            chunkBounds(clen, z, R, &poff, &plen);
            if (plen > maxPiece) maxPiece = plen;
        }
    }
    TORCH_CHECK((int64_t)maxPiece * 16 <= slot,
                "barlink-bar1 pipe: largest piece ", (int64_t)maxPiece * 16,
                " bytes does not fit in the pipe slot of ", slot,
                " bytes (K=", K, ", T=", TT, ", R=", R,
                "). The caller should have asked handles().");

    PipeArgs A;
    std::memset(&A, 0, sizeof(A));
    A.in           = (const uint4 *)inp.data_ptr();
    A.out          = (uint4 *)out.data_ptr();
    A.roundDev     = (u64 *)round_dev.data_ptr();
    A.stepDev   = (u64 *)step_dev.data_ptr();
    A.ctlStatus    = (unsigned int *)ctl_dev.data_ptr();
    A.abortDev   = ((unsigned int *)ctl_dev.data_ptr()) + 1;
    // 0 means the host could not map the abort word; the wait macros then see
    // nullptr and skip the probe entirely.
    A.abortHost  = (const unsigned int *)(uintptr_t)abort_host;
    A.n4           = n4;
    A.R            = R;
    A.rank         = r;
    A.K            = K;
    A.TT           = TT;
    A.PP           = PP;
    A.ack     = (int)ack;
    A.direct       = (int)direct;
    A.resultSlack     = (int)result_slack;
    A.capCycles = (u64)cap_cycles;
    A.slot4     = (long long)(slot / 16);
    A.class4      = (long long)(R - 1) * A.slot4;

    // The AG ring sits behind the RS ring; each ring holds T*(R-1) slots.
    // Within a ring the order is (class, position), so the class is
    // reached with a single multiple of class4.
    const long long ringBytes = (long long)TT * (R - 1) * slot;
    for (int z = 0; z < R; ++z) {
        if (z == r) continue;
        // My position in receiver z's ascending peer list. NOT (r < z) --
        // the same derivation as in mesh and a2a.
        const long long p = (long long)(r - (r > z ? 1 : 0));
        char *zb = (char *)(uintptr_t)peer_payload[z] + off_pipe;
        A.sendRS[z] = (uint4 *)(zb + p * slot);
        A.sendAG[z] = (uint4 *)(zb + ringBytes + p * slot);
        if (direct) {
            TORCH_CHECK(peer_result[z] != 0,
                        "barlink-bar1 pipe: direct mode without a result "
                        "buffer from rank ", z);
            TORCH_CHECK((peer_result[z] % 16) == 0,
                        "barlink-bar1 pipe: result buffer of rank ", z,
                        " is not 16-byte-aligned");
            A.resultTo[z] = (uint4 *)(uintptr_t)peer_result[z];
        }
    }
    if (direct) {
        // The own result buffer MUST be `out` -- otherwise the peer would
        // write into a place the caller never gets to see. This is the
        // seam between Python and the kernel, and it is checked, not
        // assumed.
        TORCH_CHECK((uintptr_t)out.data_ptr() == (uintptr_t)peer_result[r],
                    "barlink-bar1 pipe: in direct mode `out` must be the own "
                    "result buffer (out=", (int64_t)(uintptr_t)out.data_ptr(),
                    ", result=", peer_result[r], ")");
    }
    {
        char *mb = (char *)(uintptr_t)own_payload + off_pipe;
        for (int s = 0; s < R; ++s) {
            if (s == r) continue;
            const long long p = (long long)(s - (s > r ? 1 : 0));
            A.recvRS[s] = (const uint4 *)(mb + p * slot);
            A.recvAG[s] = (const uint4 *)(mb + ringBytes + p * slot);
        }
    }

    // Counter lines: four families of R lines at 256 bytes each.
    //   0 tailRS  1 tailAG  2 headRS  3 headAG
    // I write tail into MY line at the RECEIVER and read THEIR line
    // locally; head exactly the other way around. Both are posted writes
    // and local reads -- a read from a foreign BAR would be a round trip.
    for (int ph = 0; ph < 2; ++ph) {
        for (int q = 0; q < R; ++q) {
            if (q == r) continue;
            char *pf = (char *)(uintptr_t)peer_flag[q] + fbase_pipe;
            char *ef = (char *)(uintptr_t)own_flag   + fbase_pipe;
            A.tailTo [ph][q] = (u64 *)(pf + (size_t)(ph * R + r) * 256u);
            A.tailFrom[ph][q] = (const u64 *)(ef + (size_t)(ph * R + q) * 256u);
            A.headTo [ph][q] = (u64 *)(pf + (size_t)((2 + ph) * R + r) * 256u);
            A.headFrom[ph][q] = (const u64 *)(ef + (size_t)((2 + ph) * R + q) * 256u);
        }
    }

    // Family 4: the release handshake. Only set up if it actually runs --
    // without it the pointers stay null and the kernel never touches the
    // family, so the path is byte-for-byte the measured one.
    TORCH_CHECK(result_slack >= 0,
                "barlink-bar1 pipe: result_slack ", result_slack, " is negative");
    if (direct && result_slack > 0) {
        TORCH_CHECK(result_gen_dev.scalar_type() == at::kLong &&
                    result_gen_dev.numel() >= 1,
                    "barlink-bar1 pipe: result_gen_dev must be an int64 tensor "
                    "with at least one element");
        A.resultGenDev = (u64 *)result_gen_dev.data_ptr();
        for (int q = 0; q < R; ++q) {
            if (q == r) continue;
            char *pf = (char *)(uintptr_t)peer_flag[q] + fbase_pipe;
            char *ef = (char *)(uintptr_t)own_flag   + fbase_pipe;
            A.resultReadyTo [q] = (u64 *)(pf + (size_t)(4 * R + r) * 256u);
            A.resultReadyFrom[q] = (const u64 *)(ef + (size_t)(4 * R + q) * 256u);
        }
    } else {
        // Without direct mode, a slack value is meaningless, and a
        // meaningless value that's silently ignored is exactly the spot
        // where someone later assumes an effect that isn't there.
        TORCH_CHECK(result_slack == 0 || direct,
                    "barlink-bar1 pipe: result_slack ", result_slack,
                    " without direct mode -- the handshake protects a "
                    "result slot that doesn't exist in this call");
        A.resultSlack = 0;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // The grid is sized after the LARGEST CHUNK, not the whole payload:
    // more blocks than one chunk's worth of work only wait at
    // grid.sync().
    switch (inp.scalar_type()) {
        case at::kFloat:
            startPipe<float>((int)kernel_variant, (int)load_shape, A, (int)threads,
                              maxChunk, stream);
            break;
        case at::kHalf:
            startPipe<__half>((int)kernel_variant, (int)load_shape, A, (int)threads,
                               maxChunk, stream);
            break;
        case at::kBFloat16:
            startPipe<__nv_bfloat16>((int)kernel_variant, (int)load_shape, A, (int)threads,
                                      maxChunk, stream);
            break;
        default:
            TORCH_CHECK(false, "barlink-bar1 pipe: data type ", inp.scalar_type(),
                        " is not supported (float32/float16/bfloat16)");
    }
}
"""

_CPP_SRC = """
at::Tensor bar1_result_tensor(int64_t ptr, at::Tensor like);

void bar1_mesh_pipe(at::Tensor inp, at::Tensor out,
                    int64_t rank, int64_t world,
                    std::vector<int64_t> peer_payload,
                    std::vector<int64_t> peer_flag,
                    std::vector<int64_t> peer_result,
                    int64_t own_payload, int64_t own_flag,
                    int64_t slot, int64_t off_pipe, int64_t fbase_pipe,
                    int64_t k_chunks, int64_t depth, int64_t lead,
                    int64_t ack, int64_t direct, int64_t result_slack,
                    at::Tensor round_dev, at::Tensor step_dev,
                    at::Tensor result_gen_dev, at::Tensor ctl_dev,
                    int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                    int64_t load_shape, int64_t abort_host);
"""


# ===========================================================================
# Compiling
# ===========================================================================


def load_pipe_ext(cpu_group, ptxas_verbose: bool = False):
    """The CUDA extension with ``bar1_mesh_pipe``.

    Own name, own build directory, own cache key: torch keys its build
    directory only by the name, and a .so under a foreign name could be
    handed to a rank whose card cannot run it. The mechanism and cache
    hygiene come from ``barlink_device``, just as for ``barlink_bar1_ext``.
    """
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load_inline

    from sglang.srt.distributed.device_communicators.barlink_device import (
        _build_flags,
        _ext_cache_guarded,
        _local_vendor,
        _resolve_build_arches,
    )

    vendor = _local_vendor()
    if vendor != "cuda":
        raise RuntimeError(
            f"barlink-BAR1-PIPE: the direct path is NVIDIA-specific (BAR1 "
            f"aperture, PTX inline asm such as ld.mmio.relaxed.sys). This "
            f"rank reports {vendor!r}."
        )
    by_vendor = _resolve_build_arches(cpu_group)
    arches = by_vendor.get(vendor, [])
    flags = list(_build_flags(vendor, arches))
    if ptxas_verbose or os.environ.get("SGLANG_BARLINK_BAR1_PTXAS_V", "0") == "1":
        flags += ["-Xptxas", "-v"]
    name = "barlink_bar1_pipe_ext_" + vendor
    if arches:
        name += "_" + "_".join(a.replace(".", "") for a in arches)
    t0 = time.time()
    with _ext_cache_guarded(name) as build_dir:
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["bar1_mesh_pipe", "bar1_result_tensor"],
            extra_cuda_cflags=flags or None,
            verbose=bool(ptxas_verbose),
            build_directory=str(build_dir) if build_dir is not None else None,
        )
    logger.info(
        "barlink-BAR1-PIPE: extension %r built for arches %s in %.1f s.",
        name, ",".join(arches) or "<torch default>", time.time() - t0,
    )
    return _ext


# ===========================================================================
# Byte-level proof
# ===========================================================================


def byte_proof_pipe(transport, rounds: int = 0) -> bool:
    """The proof for ``mesh_pipe``: multiple rounds, uneven splitting.

    A proof that only checks ONE round misses exactly the most dangerous
    bug -- slot reuse only strikes once a slot class comes around a second
    time. Hence:

    * **multiple consecutive rounds** (default ``2T + 3``, at least enough
      that every class is hit more than once), each with DIFFERENT
      numbers -- a stale slot would otherwise not surface, because the old
      contents would happen to be correct;
    * **uneven chunk counts**: payloads whose packet count is divisible by
      neither ``K`` nor ``R``, so ``chunkBounds`` puts the remainder onto
      the leading chunks and the pieces have different lengths;
    * **remainder chunks**: the smallest permitted payload, and sizes just
      above a power of two.

    On the question "chunk not a multiple of 16": for ``all_reduce`` that
    **cannot** occur, and that is proven, not assumed. The unit of the
    split is the 128-bit packet (``n4 = nbytes/16``, ``nbytes % 16 == 0``
    is enforced at the seam), so every chunk and piece boundary is a
    multiple of 16 bytes. The proof checks this claim explicitly instead
    of trusting it.

    The comparison is **bit-exact**. The inputs are small integers, exactly
    representable in float32/float16/bfloat16, and the kernel sums in a
    fixed order (own contribution, then peers ascending). A tolerance-based
    comparison would hide exactly the error class this is about: a slot
    that carries a contribution too early or too late shifts the result by
    a whole summand.
    """
    import torch
    import torch.distributed as dist

    if not transport.pipe_on:
        return False
    depth = int(transport.pipe_t)
    if rounds <= 0:
        rounds = 2 * depth + 3

    world = transport.world
    rank = transport.rank
    device = transport.device
    slot = int(transport.pipe_slot)
    max_bytes = int(transport.max_bytes)

    # Sizes: deliberately awkward. 16*R*T is the smallest at which every
    # chunk of every rank gets at least one packet; the rest are chosen so
    # that n4 divides evenly by neither K nor R.
    candidates = [
        16 * world * depth,
        16 * (world * depth + 1),
        16 * 1021,                    # a prime number of packets
        (1 << 20) + 16 * 3,
        (4 << 20) + 16 * 7,
    ]
    sizes = []
    for n in candidates:
        if n < transport.min_bytes or n > max_bytes:
            continue
        if n % 16 or n // 16 < world:
            continue
        sizes.append(n)
    if not sizes:
        logger.warning(
            "barlink-BAR1-PIPE: no sample size fits between %d and %d "
            "bytes -- the proof cannot say anything, so it declines.",
            transport.min_bytes, max_bytes,
        )
        return False

    all_good = True
    for nbytes in sizes:
        n4 = nbytes // 16
        k = transport._pipe_k(nbytes)
        if k is None:
            logger.info(
                "barlink-BAR1-PIPE: %d bytes is not carried by the pipelined "
                "path (no K fits) -- skipped, not failed.", nbytes,
            )
            continue

        # Claim 1, proven rather than trusted: no chunk or piece boundary
        # falls on an address that isn't a multiple of 16, and no piece
        # exceeds the slot.
        for c in range(k):
            coff, clen = _chunk_bounds(n4, c, k)
            if (coff * 16) % 16 or (clen * 16) % 16:
                raise AssertionError("chunk boundary not 16-byte-aligned")
            for z in range(world):
                poff, plen = _chunk_bounds(clen, z, world)
                if ((coff + poff) * 16) % 16:
                    raise AssertionError("piece boundary not aligned")
                if plen * 16 > slot:
                    raise AssertionError(
                        f"piece {plen * 16} > slot {slot}"
                    )

        n = nbytes // 4
        for round in range(rounds):
            # Different numbers every round, and different per rank -- a
            # slot carrying the previous round's contents would thereby
            # surface.
            wave = torch.arange(n, dtype=torch.float32, device=device)
            own = ((wave % 7.0) + 1.0) * float(rank + 1) + float(round * 13)
            expected = torch.zeros(n, dtype=torch.float32, device=device)
            for q in range(world):
                expected += ((wave % 7.0) + 1.0) * float(q + 1) + float(round * 13)
            table = getattr(transport, "_peer_table", None)
            bounded_barrier(
                transport.cpu_group,
                f"bar1 pipe proof: before round {round + 1}/{rounds}",
                table=table,
            )
            actual = transport._pipe_all_reduce(own, k)
            bounded_device_sync(
                f"bar1 pipe proof: round {round + 1}/{rounds}",
                device=device,
                table=table,
            )
            # A tripped kernel is otherwise silent -- the comparison below
            # would report a data bug where the cause was an abort.
            transport.raise_if_aborted(f"pipe proof round {round + 1}")
            bad = int((actual != expected).sum().item())
            if bad:
                first_bad = int((actual != expected).nonzero()[0].item())
                logger.warning(
                    "barlink-BAR1-PIPE: byte-level proof FAILED at %d bytes, "
                    "K=%d, T=%d, round %d/%d: %d of %d values wrong, first "
                    "at index %d (got %r, expected %r).",
                    nbytes, k, depth, round + 1, rounds, bad, n, first_bad,
                    float(actual[first_bad]), float(expected[first_bad]),
                )
                all_good = False
                break
        else:
            logger.info(
                "barlink-BAR1-PIPE: byte-level proof passed: %d bytes (n4=%d, "
                "K=%d, T=%d), %d rounds, 0 of %d values wrong.",
                nbytes, n4, k, depth, rounds, n,
            )
    # ONE answer group-wide -- otherwise `handles` would answer
    # rank-dependently, and one rank would enter the collective while
    # another bailed out. A broadcast from rank 0 would NOT do that: the
    # proof has failed as soon as ANY rank has seen a wrong value.
    carrier: list = [None] * world
    # torch runs the object collectives inline; there is no Work to bound, so
    # the one-shot check names an already dead peer instead of entering the
    # 7200 s gloo wait for it.
    check_peers(
        "bar1 pipe proof: verdict exchange",
        getattr(transport, "_peer_table", None),
    )
    dist.all_gather_object(carrier, bool(all_good), group=transport.cpu_group)
    return all(bool(x) for x in carrier)
