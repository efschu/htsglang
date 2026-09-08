"""#1243 step 1 -- EXPERIMENT-ONLY emulation of a "precision tail" KV cache.

WHAT THIS IS FOR, and what it is NOT
------------------------------------
This module exists so ONE question can be answered with numbers before any
feature is built: does today's ``--kv-cache-dtype fp8_e4m3`` KV cache lose
measurable quality against a bf16 KV cache, and does keeping only the youngest
N tokens of each sequence in 16 bit recover it?

It is NOT a serving knob and must never become one.  There is no server_args
flag, no CLI option, and no config field.  It is reached only through the
environment, only together with an acknowledgement token, and it REFUSES loudly
(``KvTailProbeRefused``) on every configuration it cannot emulate faithfully.
The standing law it obeys is "an experiment vector is never shipped"
(memory ``vram-config-gesetze-2026-08-15``, rule 2).

WHY STORAGE-SIDE ROUNDING AND NOT READ-SIDE
-------------------------------------------
The honest place to emulate "old tokens are fp8, young tokens are bf16" is at
the attention READ.  On this deployment that site is
``flashinfer_backend.py:5747`` --

    o, lse = decode_wrapper.forward_return_lse(
        q_full, self.token_to_kv_pool.get_kv_buffer(layer.layer_id), ...)

-- i.e. the WHOLE K/V buffer is handed to FlashInfer, which indexes it per
token inside a CUDA kernel using the page table.  There is no Python between
"pick token p" and "read its K/V".  Per-token-age rounding at read time is
therefore NOT patchable without editing a CUDA kernel, which is out of scope
for a probe.

The closest honest alternative, and the one implemented here, is STORAGE-SIDE
rounding at the moment a token LEAVES the tail window: the pool is bf16, and a
token's K/V is fp8-round-tripped IN PLACE exactly once, when its position falls
below ``seq_len - N``.

That is mathematically identical to read-side rounding, for two reasons:

  1. Once rounded, the stored value is already exactly fp8-representable, so
     every later read returns the same value read-side rounding would have
     produced.  (bf16 has 7 mantissa bits and fp8_e4m3fn has 3, and the whole
     e4m3 exponent range sits inside bf16's, so ``bf16(fp8(x)) == fp8(x)``
     exactly -- the round trip is idempotent, asserted by the hermetic test.)
  2. A token that has once become older than N never becomes young again, so
     "rounded from now on" and "rounded at every read from now on" describe the
     same value sequence.

THE ONE RESIDUAL, AND HOW THE PLAN DRIVES IT TO ZERO
----------------------------------------------------
Eviction runs at the TOP of ``set_kv_buffer``, i.e. before the current call's
rows are physically written.  The current call's own rows are therefore never
evicted in the same call -- they are evicted at the next one.  So for a write
of ``n`` rows, rows at positions ``[L-n, L-N)`` are read at bf16 during exactly
one forward pass and are demoted at the next.

  * decode: n = 1 (or num_draft_tokens), always <= N, residual = 0.
  * prefill: n = chunked-prefill-size C, residual = max(0, C - N) rows per
    chunk.

``KVTAIL_PROBE_PLAN_0908.md`` therefore runs EVERY arm with
``--chunked-prefill-size 1024`` <= min(N) = 1024, which makes the residual
EXACTLY ZERO on every arm.  The residual is reported by
``probe_stats()["residual_rows_max"]`` so a run can never quietly carry one.

SINGLE-SEQUENCE ONLY -- and why that is not a limitation here
-------------------------------------------------------------
"Youngest N of each sequence" needs a position for every physical slot.  Under
uneven DCP the physical slot is a COMPACTED, owner-rule-derived index
(``flashinfer_backend.py:2598`` ``loc = cache_loc // self.dcp_size`` for the
even rule, and a weighted variant for ``--uneven-dcp-weighted``), so
``req_to_token[req, pos]`` is NOT the pool slot on this rank and a position
lookup would be wrong.

Instead this module reads the slots the write actually touched (``loc`` plus
``dcp_kv_mask``, exactly this rank's owned subset) and pairs row i of a write
with logical position ``seq_len - n + i``.  That pairing is exact for ONE
sequence in flight and meaningless for several, so the probe REFUSES a batch
with more than one request.  The probe protocol is one request at a time
against an idle server anyway -- inherited from the #855 KLD capture
(``scripts/int8_368/kld_capture_855.py`` at 477fb3dc7b), whose discipline was
"one request at a time against an idle server, each point cached_tokens=0
verified".

Environment contract
--------------------
``SGLANG_KV_TAIL_PROBE_N``    ``<int>`` | ``all`` | ``0``.  Absent => inert.
``SGLANG_KV_TAIL_PROBE_ACK``  must equal ``ACK_TOKEN`` below, or the probe
                              refuses at first use.

  N = 0     -> every token is fp8 (emulates today's fp8 body).
  N = all   -> no token is ever rounded (emulates full bf16; the identity).
  N = k     -> tokens older than the youngest k positions are fp8.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ENV_N = "SGLANG_KV_TAIL_PROBE_N"
ENV_ACK = "SGLANG_KV_TAIL_PROBE_ACK"

#: Deliberately unpleasant to type into a launch command by accident.  Anyone
#: who sets this has read the module docstring and knows it is not a feature.
ACK_TOKEN = "i-am-an-experiment-not-a-feature-1243"

#: fp8_e4m3fn is the storage dtype of ``--kv-cache-dtype fp8_e4m3``; the round
#: trip must use the SAME dtype the real path stores into, or the emulation is
#: not an emulation.  See ``MHATokenToKVPool.set_kv_buffer``:
#:     cache_k = cache_k.to(self.dtype)   # self.dtype is float8_e4m3fn
FP8_DTYPE = torch.float8_e4m3fn

#: Only these pool dtypes can carry the emulation: the pool must be WIDER than
#: fp8, otherwise there is nothing to round down from.
SUPPORTED_POOL_DTYPES = (torch.bfloat16, torch.float16)


class KvTailProbeRefused(RuntimeError):
    """The probe cannot emulate this configuration faithfully, so it stops.

    Every raise site names the configuration and why it cannot be emulated.
    The probe never degrades silently into "roughly right" -- a quality number
    taken from a silently-degraded emulation would be worse than no number.
    """


def _parse_n(raw: str) -> Optional[int]:
    """``"all"`` -> ``None`` (never round), otherwise a non-negative int."""
    s = raw.strip().lower()
    if s in ("all", "inf", "infinite"):
        return None
    try:
        n = int(s)
    except ValueError as exc:
        raise KvTailProbeRefused(
            f"{ENV_N}={raw!r} is neither an integer nor 'all'"
        ) from exc
    if n < 0:
        raise KvTailProbeRefused(f"{ENV_N}={raw!r} must be >= 0 (or 'all')")
    return n


class _Config:
    """Parsed once at import.  ``active`` False => this module does nothing."""

    __slots__ = ("active", "raw", "tail_n", "acked")

    def __init__(self, environ: Optional[Dict[str, str]] = None):
        env = os.environ if environ is None else environ
        raw = env.get(ENV_N)
        self.raw = raw
        self.active = raw is not None
        self.acked = env.get(ENV_ACK) == ACK_TOKEN
        self.tail_n: Optional[int] = _parse_n(raw) if self.active else 0

    def check_ack(self) -> None:
        if not self.acked:
            raise KvTailProbeRefused(
                f"{ENV_N} is set but {ENV_ACK} != {ACK_TOKEN!r}.  This switch is "
                "an EXPERIMENT-ONLY emulation for #1243 step 1, not a serving "
                "knob; it changes what attention reads and must never be set on "
                "a serving boot.  Set the acknowledgement token explicitly, or "
                "unset both."
            )


_CFG = _Config()

#: Hot-path guard.  False on every normal boot => the two call sites in
#: ``RadixAttention.forward`` and ``MHATokenToKVPool.set_kv_buffer`` are a
#: single module-global bool test and nothing else runs.  Behaviourally
#: byte-identical to the unpatched tree; not literally the same bytecode.
ACTIVE = _CFG.active


class _LayerRing:
    """Chronological record of the rows this rank wrote for ONE layer.

    Each entry is ``(first_pos, loc, mask)``:
      ``first_pos``  logical position of row 0 of that write
      ``loc``        physical pool slots, one per row of the write (int, 1-D)
      ``mask``       DCP owner mask (bool, 1-D) or None when every row was
                     written (non-DCP path)
    """

    __slots__ = ("records", "evicted_upto", "seq_len", "rounded_rows")

    def __init__(self) -> None:
        self.records: Deque[Tuple[int, torch.Tensor, Optional[torch.Tensor]]] = deque()
        self.evicted_upto: int = 0
        self.seq_len: int = 0
        self.rounded_rows: int = 0

    def reset(self) -> None:
        self.records.clear()
        self.evicted_upto = 0
        self.seq_len = 0


class _State:
    """Per-process probe state.  One ring per (pool identity, layer)."""

    __slots__ = ("rings", "ctx_seq_len", "ctx_scale", "ctx_layer_id", "residual_max",
                 "banner_done", "writes", "reset_count")

    def __init__(self) -> None:
        self.rings: Dict[Tuple[int, int], _LayerRing] = {}
        self.ctx_seq_len: Optional[int] = None
        self.ctx_scale: Optional[float] = None
        self.ctx_layer_id: Optional[int] = None
        self.residual_max: int = 0
        self.banner_done: bool = False
        self.writes: int = 0
        self.reset_count: int = 0

    def reset_all(self) -> None:
        for ring in self.rings.values():
            ring.reset()
        self.reset_count += 1


_STATE = _State()


def reset_for_test() -> None:
    """Test-only: drop all rings and re-read the environment."""
    global _CFG, ACTIVE, _STATE
    _CFG = _Config()
    ACTIVE = _CFG.active
    _STATE = _State()


def configure_for_test(tail_n: Optional[int], acked: bool = True) -> None:
    """Test-only: arm the probe without touching ``os.environ``."""
    global ACTIVE
    _CFG.active = True
    _CFG.acked = acked
    _CFG.tail_n = tail_n
    _CFG.raw = "all" if tail_n is None else str(tail_n)
    ACTIVE = True
    _STATE.rings.clear()
    _STATE.ctx_seq_len = None
    _STATE.residual_max = 0
    _STATE.writes = 0


def probe_stats() -> Dict[str, object]:
    """Everything a run must report about the emulation itself.

    ``residual_rows_max`` is the largest number of rows that were read at bf16
    in the pass that wrote them while the tail rule said fp8 (see the module
    docstring).  A run with ``--chunked-prefill-size <= N`` must report 0; any
    other value has to be quoted beside the quality numbers.
    """
    return {
        "active": ACTIVE,
        "tail_n": "all" if _CFG.tail_n is None else _CFG.tail_n,
        "raw": _CFG.raw,
        "writes": _STATE.writes,
        "rings": len(_STATE.rings),
        "sequence_resets": _STATE.reset_count,
        "rounded_rows_total": sum(r.rounded_rows for r in _STATE.rings.values()),
        "residual_rows_max": _STATE.residual_max,
    }


def fp8_round_trip_(
    buf: torch.Tensor, slots: torch.Tensor, scale: Optional[float]
) -> None:
    """fp8_e4m3 round-trip rows ``slots`` of ``buf``, in place.

    THIS EXPRESSION IS THE EMULATION.  It must stay identical to the real fp8
    storage path in ``MHATokenToKVPool.set_kv_buffer``, which is:

        if cache_k.dtype != self.dtype:          # self.dtype = float8_e4m3fn
            if k_scale is not None:
                cache_k.div_(k_scale)
            cache_k = cache_k.to(self.dtype)

    ... after which FlashInfer multiplies by ``k_scale`` again on read.  So the
    value attention sees in the real path is ``fp8(x / s) * s``, and that is
    what this reproduces.  ``test_kv_tail_probe_1243.py`` asserts the two
    bitwise on random blocks AND greps the real function's source for those
    lines, so a change upstream turns the test red instead of silently making
    the emulation a fiction.
    """
    if slots.numel() == 0:
        return
    rows = buf[slots]
    if scale is not None and scale != 1.0:
        rows = rows / scale
    rows = rows.to(FP8_DTYPE).to(buf.dtype)
    if scale is not None and scale != 1.0:
        rows = rows * scale
    buf[slots] = rows


def _resolve_mha_pool(pool):
    """Unwrap to the object that actually owns ``k_buffer`` / ``v_buffer``."""
    inner = getattr(pool, "full_kv_pool", None)
    return inner if inner is not None else pool


def begin_attention(layer, forward_batch) -> None:
    """Call site 1: top of ``RadixAttention.forward``'s backend dispatch.

    Publishes the single-sequence context the write hook needs (sequence
    length, kv scale) and enforces every refusal that depends on the batch.
    """
    if not ACTIVE:
        return
    _CFG.check_ack()

    seq_lens = getattr(forward_batch, "seq_lens", None)
    if seq_lens is None:
        raise KvTailProbeRefused(
            "forward batch carries no seq_lens; the probe cannot place tokens "
            "on a position axis without it"
        )
    n_req = int(seq_lens.numel())
    if n_req != 1:
        raise KvTailProbeRefused(
            f"batch carries {n_req} requests; the #1243 probe emulation is only "
            "exact for ONE sequence in flight (physical slots under uneven DCP "
            "are owner-compacted, so a per-request position axis does not "
            "exist).  Drive the probe one request at a time against an idle "
            "server, as the #855 KLD capture did."
        )

    k_scale = getattr(layer, "k_scale_float", None)
    v_scale = getattr(layer, "v_scale_float", None)
    for name, s in (("k_scale", k_scale), ("v_scale", v_scale)):
        if s is not None and float(s) != 1.0:
            raise KvTailProbeRefused(
                f"{name}={s} != 1.0.  The emulation is bit-exact against the "
                "real fp8 path only at scale 1.0 (at another scale the extra "
                "bf16 store of x*s costs one rounding the real path does not "
                "take).  This checkpoint carries no kv scales, so a non-unit "
                "scale means the assumption behind the probe no longer holds."
            )

    seq_len = int(seq_lens[0].item())
    prev = _STATE.ctx_seq_len
    # A shorter (or equal) sequence than the previous pass means a NEW request
    # took the slot: positions restart at 0 and every ring is stale.
    if prev is not None and seq_len <= prev:
        _STATE.reset_all()
    _STATE.ctx_seq_len = seq_len
    _STATE.ctx_scale = None if k_scale is None else float(k_scale)
    _STATE.ctx_layer_id = getattr(layer, "layer_id", None)

    if not _STATE.banner_done:
        _STATE.banner_done = True
        logger.warning(
            "#1243 KV-TAIL PROBE ACTIVE -- EXPERIMENT ONLY, NOT A SERVING MODE. "
            "%s=%s: tokens older than the youngest %s positions are "
            "fp8_e4m3-round-tripped in the bf16 KV pool. This changes what "
            "attention reads. Never set %s on a serving boot.",
            ENV_N,
            _CFG.raw,
            "ALL (identity, nothing rounded)" if _CFG.tail_n is None else _CFG.tail_n,
            ENV_N,
        )


def after_write(pool, layer_id: int, loc, dcp_kv_mask=None) -> None:
    """Call site 2: top of ``MHATokenToKVPool.set_kv_buffer``, before the write.

    Appends this write's rows to the layer's ring and rounds every row that has
    just fallen out of the tail window.  Rows of THIS call are never rounded
    here -- they are not written yet; they leave the window at a later call.
    See the residual note in the module docstring.
    """
    if not ACTIVE:
        return
    _CFG.check_ack()

    seq_len = _STATE.ctx_seq_len
    if seq_len is None:
        raise KvTailProbeRefused(
            "a KV write reached the probe without an attention context; the "
            "RadixAttention.forward hook did not run for this layer, so the "
            "position axis is unknown and the emulation would be a guess"
        )

    if not isinstance(loc, torch.Tensor):
        loc = torch.as_tensor(loc)
    n_rows = int(loc.numel())
    if n_rows == 0:
        return

    inner = _resolve_mha_pool(pool)
    dtype = getattr(inner, "dtype", None)
    if dtype not in SUPPORTED_POOL_DTYPES:
        raise KvTailProbeRefused(
            f"KV pool dtype is {dtype}; the probe emulates a 16-bit pool with an "
            "fp8 body, so the arm must be booted with a 16-bit "
            "--kv-cache-dtype (bf16).  Booting an fp8 pool AND the probe would "
            "double-quantize."
        )

    key = (id(inner), int(layer_id))
    ring = _STATE.rings.get(key)
    if ring is None:
        ring = _LayerRing()
        _STATE.rings[key] = ring

    first_pos = seq_len - n_rows
    if first_pos < 0:
        raise KvTailProbeRefused(
            f"write of {n_rows} rows against seq_len={seq_len}: the write is "
            "longer than the sequence, so row->position pairing is impossible"
        )

    tail_n = _CFG.tail_n
    if tail_n is None:
        # N = all: the identity arm.  Record nothing, round nothing.
        _STATE.writes += 1
        ring.seq_len = seq_len
        return

    mask = dcp_kv_mask
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    ring.records.append((first_pos, loc, mask))
    ring.seq_len = seq_len
    _STATE.writes += 1

    boundary = seq_len - tail_n
    # Rows of the CURRENT write that the rule already calls old but that are not
    # written yet: they are demoted one call later.  Report the largest such
    # count so a run can never hide it.
    residual = max(0, min(boundary, seq_len) - first_pos)
    if residual > _STATE.residual_max:
        _STATE.residual_max = residual
    # Never evict into the current record.
    boundary = min(boundary, first_pos)
    if boundary <= ring.evicted_upto:
        return

    k_buf = inner._get_key_buffer(layer_id)
    v_buf = inner._get_value_buffer(layer_id)
    scale = _STATE.ctx_scale

    while ring.records:
        rec_first, rec_loc, rec_mask = ring.records[0]
        rec_len = int(rec_loc.numel())
        rec_end = rec_first + rec_len
        take = min(rec_end, boundary) - max(rec_first, ring.evicted_upto)
        if take <= 0:
            break
        lo = max(rec_first, ring.evicted_upto) - rec_first
        hi = lo + take
        slots = rec_loc[lo:hi]
        if rec_mask is not None:
            slots = slots[rec_mask[lo:hi]]
        slots = slots.to(dtype=torch.long)
        fp8_round_trip_(k_buf, slots, scale)
        fp8_round_trip_(v_buf, slots, scale)
        ring.rounded_rows += int(slots.numel())
        ring.evicted_upto = max(rec_first, ring.evicted_upto) + take
        if ring.evicted_upto >= rec_end:
            ring.records.popleft()
        else:
            break
