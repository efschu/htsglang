# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#732 follow-up: the BAR1 point-to-point seam -- the algebra, not the kernel.

WHY THIS EXISTS. barlink BAR1 carries only collectives: the three kernels are
``bar1_mesh_kernel``, ``bar1_ring_kernel`` and ``bar1_a2a_kernel``, and there is
no ``send``/``recv`` on the transport. #732 established that this API absence --
NOT cost -- is the only remaining ground for the PP family-placement
foreclosure. Its cost argument was withdrawn: 29 extra crossings at the measured
7.30 us host ping-pong (``benchmark/bench_host_transport.py:12``) is ~0.7 % of a
30 ms decode round.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT. A real BAR1 send/recv
needs a new CUDA kernel (or a host spin) and TWO cards with dmabuf peer mapping
to exercise. That half is **window-gated** and is not faked here. What is
provable at the desk is the part a kernel cannot be allowed to get wrong: which
slot a directed pair owns, how many flag lines it costs, how an oversized
payload is chunked, and which asks are refused. That is this module, and a
future kernel is expected to consume it rather than re-derive it.

Nothing here is wired into the live geometry. ``geometry()`` and
``flags_requirement()`` are untouched, so every existing layout stays
byte-for-byte; :func:`p2p_layout` computes what a p2p region WOULD occupy given
a base geometry, and returns ``off_p2p = -1`` when it would occupy nothing.

TWO DISCIPLINES INHERITED, both load-bearing:

* From ``barlink_host.send``/``recv`` (``barlink_host.py:1100``, ``:1120``) --
  the shape a point-to-point transport has here: a per-PAIR address slot, a
  flags address, a per-peer sequence counter, a bounded timeout, and a named
  refusal when p2p is disabled. Keeping that shape means a future BAR1
  implementation is a transport swap rather than a second protocol.
* From ``geometry()`` -- append new regions at the END and use ``-1`` for
  "does not exist", so existing offsets never move; and from
  ``flags_requirement()`` -- one 256-byte line per (topology, step, sender), so
  senders never false-share a cache line.

KNOWN REGIME, recorded because a caller will otherwise meet it as a surprise:
BAR1 is not a uniform win. On the fast x8 PAIR the transport LOSES between 1 and
8 MiB, down to **0.81x** versus NCCL, with 2-rank ratios 1.11/1.13/0.97/0.86/0.99
recorded as unexplained (``FEATURES_VS_UPSTREAM.md:1349``). The measured 3-rank
gains (1.13/1.34/1.15/1.04/1.30x at 20 KiB..16 MiB) must NOT be reused for a
2-rank pairing. The PP-crossing payload class this seam targets is ~10 KiB
one-way, which is below that weak band -- but a caller that grows the payload
into 1..8 MiB on a 2-rank group should expect a loss, not a gain.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

#: One flag line per directed pair. 256 bytes is the module's own line size
#: (``flags_requirement``: "one 256-byte line per (topology, step, sender): no
#: false sharing between senders"), and a p2p pair is exactly one sender.
P2P_LINE_BYTES = 256


class P2pUnavailable(RuntimeError):
    """A point-to-point ask that cannot be served, with the reason.

    Deliberately its own type rather than ``Bar1Unavailable``: a caller that
    falls back on a collective refusal should not silently swallow a p2p one,
    because the fallbacks differ (a collective can go to NCCL; a p2p crossing
    on this transport has nowhere to go yet).
    """


def p2p_slot_index(src: int, dst: int, world: int) -> int:
    """The slot a DIRECTED pair owns, dense in ``[0, R*(R-1))``.

    Directed, not symmetric: ``a->b`` and ``b->a`` are in flight at the same
    time on a pipeline that runs forward and backward passes, and sharing a
    slot between them is a silent overwrite rather than a detectable error.
    """
    if not (0 <= src < world) or not (0 <= dst < world):
        raise P2pUnavailable(
            f"p2p slot ({src}->{dst}) is outside world {world}; a rank index "
            "that does not exist is a caller bug, not a small world"
        )
    if src == dst:
        raise P2pUnavailable(
            f"p2p slot ({src}->{dst}): a rank cannot send to itself over BAR1. "
            "The local copy is a device-to-device memcpy and needs no window."
        )
    # Dense packing: each source owns (R-1) consecutive slots, in destination
    # order with the self-slot skipped.
    offset = dst if dst < src else dst - 1
    return src * (world - 1) + offset


def p2p_flags_extra(world: int) -> int:
    """Flag bytes a p2p region adds: one line per directed pair."""
    if world < 2:
        return 0
    return world * (world - 1) * P2P_LINE_BYTES


def p2p_region_bytes(world: int, slot_bytes: int) -> int:
    """Window bytes a p2p region occupies."""
    if world < 2 or slot_bytes <= 0:
        return 0
    return world * (world - 1) * int(slot_bytes)


def p2p_layout(
    base: Mapping, world: int, slot_bytes: int
) -> Dict[str, object]:
    """Where a p2p region would sit behind an existing geometry.

    APPEND-ONLY, like every region before it: the returned dict carries every
    key of ``base`` unchanged and adds ``off_p2p`` / ``p2p_slot_bytes`` /
    ``p2p_region_bytes``. ``off_p2p = -1`` means the region does not exist --
    never ``0``, which is the mesh region's own offset.
    """
    out: Dict[str, object] = dict(base)
    region = p2p_region_bytes(world, slot_bytes)
    if region <= 0:
        out["off_p2p"] = -1
        out["p2p_slot_bytes"] = 0
        out["p2p_region_bytes"] = 0
        return out
    # Behind everything the base geometry already accounts for.
    out["off_p2p"] = int(base["region_bytes"])
    out["p2p_slot_bytes"] = int(slot_bytes)
    out["p2p_region_bytes"] = region
    out["region_bytes"] = int(base["region_bytes"]) + region
    return out


def p2p_plan(nbytes: int, slot_bytes: int) -> List[Tuple[int, int]]:
    """``[(offset, length), ...]`` chunks of one payload into one slot.

    Chunked by the CALLER, mirroring ``put()``'s rule: "The caller must chunk;
    automatic re-mapping on the hot path is excluded -- it is exactly the
    expensive part."
    """
    if slot_bytes <= 0:
        raise P2pUnavailable(
            "p2p plan needs a positive slot size; a zero slot cannot carry a "
            "payload and silently planning nothing would look like success"
        )
    n = int(nbytes)
    if n <= 0:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        out.append((start, min(slot_bytes, n - start)))
        start += slot_bytes
    return out


def check_p2p_payload(
    nbytes: int, slot_bytes: int, world: int, window_bytes: int
) -> None:
    """Refuse an ask that cannot be served, naming the arithmetic.

    Returns ``None`` when the ask fits. The refusals carry their numbers for
    the same reason ``put()``'s does: a bare "does not fit" leaves the caller
    unable to tell whether to chunk, shrink the world, or grow the window.
    """
    if world < 2:
        raise P2pUnavailable(
            f"p2p needs at least two ranks, got world={world}"
        )
    region = p2p_region_bytes(world, slot_bytes)
    if region > int(window_bytes):
        raise P2pUnavailable(
            f"p2p region needs {region} bytes ({world}x{world - 1} directed "
            f"slots of {slot_bytes}) but the mapped window is {window_bytes} "
            "bytes. Shrink the slot, shrink the world, or map a larger window; "
            "re-mapping on the hot path is excluded."
        )
    if int(nbytes) > region:
        raise P2pUnavailable(
            f"p2p payload of {nbytes} bytes exceeds the whole p2p region of "
            f"{region} bytes within a {window_bytes}-byte window"
        )


def capture_safety() -> Dict[str, object]:
    """Whether a BAR1 p2p crossing can live inside a CUDA-graph capture.

    Stated rather than assumed, because the PP crossing sits in the decode
    path and a wrong answer here is a capture failure at boot, not a slow path.

    * SEND is capturable: ``put()`` is a ``memcpy_async`` on the caller's
      stream (``barlink_bar1.py:2562``), which records into a graph.
    * RECV is NOT capturable today: completion has to be waited on, and BAR1
      exposes no device-side wait -- its three kernels are all collectives and
      none of them is a bare flag spin. A host-side spin inside a capture
      raises ``cudaErrorStreamCaptureUnsupported``, which is the same class of
      failure ``_enforce_cpu_transport_needs_eager`` already refuses at startup
      for host-staged transports.

    Consequence: until a device-side p2p wait kernel exists, a PP crossing over
    this seam must run on the BREAKABLE route and be priced with #494's clock,
    not assumed free inside a captured decode.
    """
    return {
        "send_capturable": True,
        "recv_capturable": False,
        "breakable_required": True,
        "reason": (
            "BAR1 exposes no device-side wait: its kernels (bar1_mesh_kernel, "
            "bar1_ring_kernel, bar1_a2a_kernel) are collectives, and a "
            "host-side flag spin inside a capture raises "
            "cudaErrorStreamCaptureUnsupported. A device-side p2p wait kernel "
            "would make the recv half capturable."
        ),
    }
