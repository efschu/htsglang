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

THE 2026-09-09 NO-OP, ITS ROOT, AND THE TWO GUARDS THAT NOW MAKE IT LOUD
-----------------------------------------------------------------------
Boot ``weg2kvtail1`` ran four arms and every one came back BIT-IDENTICAL at all
53,047 scored positions, ``N0`` included -- an fp8 round trip of every KV byte
cannot do that.  The banner printed; the rounding never happened.

ROOT, one line: ``begin_attention`` runs ONCE PER LAYER, and every layer of one
forward pass carries the SAME ``forward_batch.seq_lens``.  The stale-sequence
test was ``seq_len <= prev``, which is therefore TRUE at every layer boundary
inside a single pass -- so ``reset_all()`` wiped every ring 60-odd times per
forward, no ring ever held the PREVIOUS chunk's record, the eviction loop found
``take <= 0`` on every call, and ``rounded_rows_total`` stayed 0 for the whole
boot.  This is the lifecycle-table class: a DELETER (``reset_all``) sat between
the WRITER (``after_write`` appending a record) and the READER (the eviction
loop), separated by the LAYER event.

The fix is ``<`` instead of ``<=`` plus a per-ring continuity guard, and it is
deliberately not trusted on its own.  Two refusals make a repeat impossible to
mistake for a null result:

  * COVERAGE (``after_write``): after the eviction loop the ring must have
    advanced ``evicted_upto`` all the way to the boundary the tail rule
    demands.  The 2026-09-09 defect fails this on the FIRST write of the second
    chunk.
  * NO-OP (``begin_attention``, once per prefill batch): if the rule was DUE --
    the sequence has grown past ``tail_n`` by at least one full write -- and not
    one row has been round-tripped, the boot dies instead of producing a
    perfect, meaningless A/A.

Plus one log line per prefill batch (``KV-TAIL-PROBE roundtrips=...``) so a run
carries its own engagement evidence and no future reader has to infer it from a
banner.
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

#: Printed for ``site=`` before any KV write has reached the probe.  It says
#: "nothing arrived", never a plausible-looking class name -- a guessed site is
#: exactly the fiction that cost boot weg2kvtail1.
SITE_UNKNOWN = "NONE-REACHED-THE-PROBE"

#: The one-line engagement report, one per prefill batch.  Kept as a constant
#: so the runner and the tests grep the SAME literal the boot emits.
REPORT_PREFIX = "KV-TAIL-PROBE"

#: The boot-level no-op refusal.  W51 was picked by ENUMERATING the used set
#: with the census in ``test/registered/unit/weg2/test_weg2_wcode_uniqueness_1263.py``
#: (37 codes assigned, highest W50; 51 is the first free number above it, and
#: it is not one of the {15, 18, 23, 39} that file pins as free).
WCODE_NOOP = "W51 Weg2KvTailProbeNoOp"


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
                 "banner_done", "writes", "reset_count", "tokens_seen",
                 "rounded_total", "roundtrips", "site", "pool_dtype",
                 "max_write_rows", "max_seq_len")

    def __init__(self) -> None:
        self.rings: Dict[Tuple[int, int], _LayerRing] = {}
        self.ctx_seq_len: Optional[int] = None
        self.ctx_scale: Optional[float] = None
        self.ctx_layer_id: Optional[int] = None
        self.residual_max: int = 0
        self.banner_done: bool = False
        self.writes: int = 0
        self.reset_count: int = 0
        #: Rows OFFERED to the probe: one row per token per layer, so this is
        #: tokens x layers and never a token count.  Named, not inferred.
        self.tokens_seen: int = 0
        #: Rows actually fp8-round-tripped.  Under DCP this is only the subset
        #: THIS rank owns, which is why it is reported beside ``tokens_seen``
        #: and never as a fraction of it.
        self.rounded_total: int = 0
        #: ``fp8_round_trip_`` invocations that touched at least one row; k and
        #: v are counted separately, so a demoted block scores 2.
        self.roundtrips: int = 0
        #: The pool class the bytes ACTUALLY took, recorded from the live call
        #: rather than assumed from a docstring -- the exact question the
        #: 2026-09-09 no-op had to be rooted by reading code.
        self.site: Optional[str] = None
        self.pool_dtype: Optional[str] = None
        self.max_write_rows: int = 0
        self.max_seq_len: int = 0

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
    _STATE.tokens_seen = 0
    _STATE.rounded_total = 0
    _STATE.roundtrips = 0
    _STATE.site = None
    _STATE.pool_dtype = None
    _STATE.max_write_rows = 0
    _STATE.max_seq_len = 0


def probe_stats() -> Dict[str, object]:
    """Everything a run must report about the emulation itself.

    ``residual_rows_max`` is the largest number of rows that were read at bf16
    in the pass that wrote them while the tail rule said fp8 (see the module
    docstring).  A run with ``--chunked-prefill-size <= N`` must report 0; any
    other value has to be quoted beside the quality numbers.

    DENOMINATORS, because two of these counters look like token counts and are
    not: ``tokens_seen`` is rows OFFERED (tokens x layers), ``tokens_rounded``
    is rows DEMOTED and under DCP covers only this rank's owned subset.  They
    are never divided by one another here.
    """
    return {
        "active": ACTIVE,
        "tail_n": "all" if _CFG.tail_n is None else _CFG.tail_n,
        "raw": _CFG.raw,
        "writes": _STATE.writes,
        "rings": len(_STATE.rings),
        "sequence_resets": _STATE.reset_count,
        "rounded_rows_total": _STATE.rounded_total,
        "residual_rows_max": _STATE.residual_max,
        "roundtrips": _STATE.roundtrips,
        "tokens_seen": _STATE.tokens_seen,
        "site": _STATE.site or SITE_UNKNOWN,
        "pool_dtype": _STATE.pool_dtype or "unknown",
        "max_write_rows": _STATE.max_write_rows,
        "max_seq_len": _STATE.max_seq_len,
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


def _is_extend_batch(forward_batch) -> bool:
    """True for a prefill/extend pass.  Unknown shape counts as extend.

    An unknown batch shape REPORTS rather than hides: a missing
    ``forward_mode`` must not be able to silence the engagement line, because
    silence is what the 2026-09-09 no-op looked like.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    is_extend = getattr(mode, "is_extend", None)
    return True if is_extend is None else bool(is_extend())


def _report_pass(prev_seq_len: Optional[int]) -> None:
    """One engagement line per prefill batch, plus the boot-level no-op refusal.

    DENOMINATOR AND TIMING, both stated because both can be misread: the
    counters are CUMULATIVE over the process, and the line is emitted at the
    START of a pass, so it covers everything up to and including the PREVIOUS
    pass.  ``tokens_seen`` is rows offered (tokens x layers); ``tokens_rounded``
    is rows demoted, which under DCP is only this rank's owned subset.

    The refusal fires only once the rule was DUE -- the sequence has grown past
    ``tail_n`` by at least one full write, evaluated on the PREVIOUS pass's
    ``seq_len`` so that the writes which would do the rounding have already
    had their chance.  The identity arm (``N=all``) rounds nothing by design
    and is excluded by name, not by threshold.
    """
    logger.warning(
        "%s roundtrips=%d tokens_seen=%d tokens_rounded=%d layers=%d site=%s "
        "dtype=%s",
        REPORT_PREFIX,
        _STATE.roundtrips,
        _STATE.tokens_seen,
        _STATE.rounded_total,
        len(_STATE.rings),
        _STATE.site or SITE_UNKNOWN,
        _STATE.pool_dtype or "unknown",
    )
    tail_n = _CFG.tail_n
    if tail_n is None:
        return
    if _STATE.rounded_total > 0 or _STATE.tokens_seen == 0:
        return
    if prev_seq_len is None or prev_seq_len <= tail_n + _STATE.max_write_rows:
        return
    raise KvTailProbeRefused(
        f"{WCODE_NOOP}: {REPORT_PREFIX} REFUSED: 0 round-trips after "
        f"{_STATE.tokens_seen} tokens (rows offered across {len(_STATE.rings)} "
        f"layer rings; tail_n={tail_n}, largest write {_STATE.max_write_rows} "
        f"rows, previous pass seq_len={prev_seq_len}, so the tail rule was DUE "
        f"and demanded demotions).  Writes reached "
        f"{_STATE.site or SITE_UNKNOWN} at dtype "
        f"{_STATE.pool_dtype or 'unknown'}.  This is the boot weg2kvtail1 "
        "failure: the banner prints, nothing is rounded, and every arm comes "
        "back bit-identical -- a PERFECT A/A that means nothing.  The arm dies "
        "here instead."
    )


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
    # A STRICTLY shorter sequence than the previous pass means a NEW request
    # took the slot: positions restart at 0 and every ring is stale.
    #
    # STRICTLY -- and that single character is the whole 2026-09-09 no-op.
    # This hook runs ONCE PER LAYER and every layer of one forward pass carries
    # the SAME seq_lens, so the old ``<=`` wiped every ring at every layer
    # boundary; no ring survived to hold the previous chunk, the eviction loop
    # always found take<=0, and four arms came back bit-identical at 53,047
    # positions with the ACTIVE banner printed.  An EQUAL seq_len is the normal
    # within-pass case here, not a new request.  A new request that is LONGER
    # than the last is caught per-ring in ``after_write`` instead, because the
    # global seq_len cannot see it.
    if prev is not None and seq_len < prev:
        _STATE.reset_all()
    new_pass = prev is None or seq_len != prev
    _STATE.ctx_seq_len = seq_len
    if seq_len > _STATE.max_seq_len:
        _STATE.max_seq_len = seq_len
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

    # One engagement line per prefill batch (and the no-op refusal).  Emitted
    # on the FIRST layer of a new pass, because that is the only point at which
    # a per-layer hook can see a batch boundary at all.
    if new_pass and _is_extend_batch(forward_batch):
        _report_pass(prev)


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

    # CONTINUITY, per ring.  For one sequence the writes of a layer are
    # contiguous: chunk c must start exactly where chunk c-1 ended.  A gap or a
    # rewind means a DIFFERENT request took the slot -- including the case the
    # global seq_len test structurally cannot see, a new request LONGER than
    # the last one, whose stale ring would otherwise round slots that now
    # belong to someone else.
    if not ring.records and ring.seq_len == 0:
        # An EMPTY ring and a write that does not start at position 0.  An
        # empty ring is only legitimate at the start of a sequence, and the
        # probe's position axis assumes a sequence starts at 0 (the arms run
        # --disable-radix-cache and verify cached_tokens == 0 per passage).
        # So this is either a prefix-cached start the emulation cannot place,
        # or history that was WIPED between the write and the read -- which is
        # exactly the 2026-09-09 no-op.  Refusing here is what makes the two
        # indistinguishable states both loud instead of both silent.
        if first_pos != 0:
            raise KvTailProbeRefused(
                f"{WCODE_NOOP}: {REPORT_PREFIX} REFUSED: layer {layer_id} holds "
                f"no write history, yet this write starts at position "
                f"{first_pos} (seq_len={seq_len}, {n_rows} rows).  Either the "
                "sequence did not start at 0 for this rank (prefix cache -- "
                "run with --disable-radix-cache and cached_tokens == 0), or "
                "the ring was wiped between the write and the eviction read, "
                "which is the boot weg2kvtail1 no-op: nothing gets demoted and "
                "every arm comes back bit-identical."
            )
        ring.evicted_upto = 0
    elif first_pos != ring.seq_len:
        ring.reset()
        ring.evicted_upto = first_pos
        _STATE.reset_count += 1

    _STATE.tokens_seen += n_rows
    if n_rows > _STATE.max_write_rows:
        _STATE.max_write_rows = n_rows
    _STATE.pool_dtype = str(dtype)
    _STATE.site = (
        f"{type(pool).__name__}.set_kv_buffer"
        if inner is pool
        else f"{type(pool).__name__}->{type(inner).__name__}.set_kv_buffer"
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
        n_slots = int(slots.numel())
        ring.rounded_rows += n_slots
        _STATE.rounded_total += n_slots
        if n_slots:
            _STATE.roundtrips += 2  # k and v are two round trips, counted apart
        ring.evicted_upto = max(rec_first, ring.evicted_upto) + take
        if ring.evicted_upto >= rec_end:
            ring.records.popleft()
        else:
            break

    # COVERAGE INVARIANT -- the guard that would have killed boot weg2kvtail1
    # on its second chunk instead of after four arms.  The records of one ring
    # are contiguous from its eviction cursor, so the loop above can only stop
    # short if history was LOST between the write and the read.  Positions are
    # compared, never owned rows: a DCP mask changes how many slots are
    # demoted, never how far the cursor may advance.
    if ring.evicted_upto < boundary:
        raise KvTailProbeRefused(
            f"{WCODE_NOOP}: {REPORT_PREFIX} REFUSED: layer {layer_id} demoted "
            f"only up to position {ring.evicted_upto} of the {boundary} the "
            f"tail rule demands (tail_n={tail_n}, seq_len={seq_len}, this "
            f"write {n_rows} rows at {first_pos}, {len(ring.records)} records "
            "held).  The ring lost the history it needs, so tokens older than "
            "the tail are still 16-bit and every arm would come back "
            "bit-identical."
        )
