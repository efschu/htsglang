# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#622/#649: online cross-rank lockstep sentinel — first-divergence attribution.

WHY THIS EXISTS
---------------
The wedge family aborts tens of thousands of graph replays after whatever
diverged (replay #71753 in one specimen); the abort site is not the injury
site. The collective census (#583) compares COUNTS and has reported them
byte-identical across ranks in wedged specimens — counts cannot see a
POSITIONAL divergence (rank A ran op X where rank B ran op Y, same totals),
nor which graph each rank SELECTED per step. The capture census proved the
captured sequences identical across ranks; what was never compared is the
per-step selection. This module compares the actual per-rank event stream,
positionally, online, and names the first mismatch as `rank R diverged at
seq S: expected X, got Y` — converting "wedged somewhere in replay" into a
file and line.

WHAT IT RECORDS (one append per event, hot-path cost ~1 dict lookup)
--------------------------------------------------------------------
- every host-path collective dispatched through a GroupCoordinator (hooked
  at the collective census bump, so coverage is identical to the census and
  transport-agnostic: barlink AND NCCL arms record the same stream);
- every CUDA-graph replay with its full selection key (kind, ShapeKey,
  segment index) — replays re-execute a fixed captured sequence, so replay
  divergence enters exactly through WHICH graph each rank picks per step;
- capture-time dispatches, tagged separately ("c"): capture order is
  rank-identical by the capture census's proof, but a divergence during a
  RE-capture would still be caught.

Payload sizes are deliberately EXCLUDED from the position hash: under uneven
TP (rank-tp-ratio) per-rank shard bytes differ legitimately.

HOW IT COMPARES (sidecar thread, bounded lag, bounded wait)
-----------------------------------------------------------
A dedicated gloo process group (created collectively at model-parallel init;
NEVER the census's cpu_group — two threads on one gloo group corrupt op
ordering). Every interval the sidecar all-gathers (my_seq, chain@min) and
compares the rolling FNV chain at the group-min position: equal chains prove
the entire prefixes equal; a mismatch triggers a windowed entry exchange
that names the first divergent seq, dumps every rank's surrounding ring
entries to SGLANG_SENTINEL_DUMP_DIR, and logs a loud `SENTINEL DIVERGENCE`
line. The sidecar never launches an unbounded collective — the gloo group
carries a timeout, and a sync loss is logged and ends the sidecar instead of
adding a second wedge on top of the first.

A stalled rank (no divergence, position frozen while peers advance) is
reported as `SENTINEL STALL` with both positions.

CAN-FAIL PROOF (mandatory before any green is believed)
-------------------------------------------------------
`SGLANG_SENTINEL_FAULT=rank:seq:mode` (mode `skip` or `dup`) makes exactly
one rank's ring drop or double exactly one event — indistinguishable, from
the comparison's view, from that rank skipping/adding one collective. The
sentinel must then name exactly that seq and op. No green without this
firing red first.

ENV
---
SGLANG_LOCKSTEP_SENTINEL=1     arm (default off; zero cost when off)
SGLANG_SENTINEL_INTERVAL_S     compare cadence, default 0.5
SGLANG_SENTINEL_RING           ring length, default 65536
SGLANG_SENTINEL_DUMP_DIR       divergence dumps, default /tmp
SGLANG_SENTINEL_FAULT          fault injector, "rank:seq:mode"
"""

from __future__ import annotations

import logging
import os
import threading
import time
import zlib
from array import array
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MASK64 = (1 << 64) - 1
_FNV_PRIME = 0x100000001B3

ENV_ENABLE = "SGLANG_LOCKSTEP_SENTINEL"
ENV_INTERVAL = "SGLANG_SENTINEL_INTERVAL_S"
ENV_RING = "SGLANG_SENTINEL_RING"
ENV_DUMP_DIR = "SGLANG_SENTINEL_DUMP_DIR"
ENV_FAULT = "SGLANG_SENTINEL_FAULT"

#: How long the group-min position may sit still, while a peer is ahead,
#: before a STALL line is emitted (seconds).
STALL_AFTER_S = 5.0
#: Entries dumped around a divergence, per rank.
DUMP_CONTEXT = 400
#: Cap on entries exchanged per localization window.
WINDOW_CAP = 8192


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") not in ("0", "", "false", "False")


def _parse_fault(rank: int) -> Tuple[int, str]:
    """Return (seq, mode) armed for THIS rank, or (-1, "")."""
    raw = os.environ.get(ENV_FAULT, "")
    if not raw:
        return -1, ""
    try:
        r, s, mode = raw.split(":")
        if int(r) != rank or mode not in ("skip", "dup"):
            return -1, ""
        return int(s), mode
    except ValueError:
        logger.warning("lockstep sentinel: unparseable %s=%r", ENV_FAULT, raw)
        return -1, ""


class LockstepSentinel:
    """Per-process event ring + positional cross-rank comparison."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        group: Any,  # torch.distributed ProcessGroup (gloo), or None for tests
        ring_len: int = 65536,
        interval_s: float = 0.5,
        dump_dir: str = "/tmp",
        start_thread: bool = True,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.ring_len = ring_len
        self.interval_s = interval_s
        self.dump_dir = dump_dir
        # ring: entry tuples + rolling chain, index = seq % ring_len
        self._tags: List[Any] = [None] * ring_len
        self._chains = array("Q", [0]) * ring_len
        self._seq = 0  # number of recorded events; next event gets this seq
        self._chain = 0
        # memoized deterministic tag hash (Python's hash() is per-process
        # randomized for strings and useless across ranks)
        self._tag_ints: dict = {}
        # capture-state probe, resolved lazily so tests need no torch/cuda
        self._capturing = None
        self.diverged = False
        #: (div_seq, culprit_ranks, per_rank_tags) of the named divergence.
        self.last_divergence: Optional[Tuple[int, List[int], List[Any]]] = None
        self.sync_lost = False
        self._verified = -1  # highest seq proven equal across ranks
        self._fault_seq, self._fault_mode = _parse_fault(rank)
        self._fault_fired = False
        self._last_min = -1
        self._last_min_t = time.monotonic()
        self._last_stall_log = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._fault_seq >= 0:
            logger.warning(
                "lockstep sentinel rank %d: FAULT INJECTOR ARMED: %s at seq %d",
                rank,
                self._fault_mode,
                self._fault_seq,
            )
        if start_thread:
            self._thread = threading.Thread(
                target=self._run, name="lockstep-sentinel", daemon=True
            )
            self._thread.start()

    # -- hot path -------------------------------------------------------

    def _tag_int(self, tag: tuple) -> int:
        v = self._tag_ints.get(tag)
        if v is None:
            v = zlib.crc32(repr(tag).encode())
            self._tag_ints[tag] = v
        return v

    def _record(self, tag: tuple) -> None:
        s = self._seq
        if s == self._fault_seq and not self._fault_fired:
            self._fault_fired = True
            if self._fault_mode == "skip":
                logger.error(
                    "lockstep sentinel rank %d: FAULT INJECTED: skipped %r at seq %d",
                    self.rank,
                    tag,
                    s,
                )
                return
            # dup: fall through, then record a second time
        c = ((self._chain * _FNV_PRIME) ^ self._tag_int(tag)) & _MASK64
        i = s % self.ring_len
        self._tags[i] = tag
        self._chains[i] = c
        self._chain = c
        self._seq = s + 1
        if self._fault_fired and self._fault_mode == "dup" and s == self._fault_seq:
            logger.error(
                "lockstep sentinel rank %d: FAULT INJECTED: duplicated %r at seq %d",
                self.rank,
                tag,
                s,
            )
            self._fault_mode = ""  # never again
            self._record(tag)

    def _is_capturing(self) -> bool:
        if self._capturing is None:
            try:
                import torch

                self._capturing = torch.cuda.is_current_stream_capturing
            except Exception:  # noqa: BLE001 - tests without cuda
                self._capturing = lambda: False
        try:
            return bool(self._capturing())
        except Exception:  # noqa: BLE001 - never break a collective for this
            return False

    def note_host(self, family: str) -> None:
        """One host-path collective dispatch (census-coverage-identical)."""
        if self._is_capturing():
            self._record(("c", family))
        else:
            self._record(("h", family))

    def note_replay(self, kind: str, key: Any, index: int) -> None:
        """One graph replay with its full selection key."""
        size = getattr(key, "size", None)
        stream_idx = getattr(key, "stream_idx", None)
        variant = getattr(key, "variant_label", None)
        if size is None and key is not None:
            size = str(key)  # unknown key type: stringify, deterministic
        self._record(("r", kind, size, stream_idx, variant, index))

    # -- sidecar --------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def _gather(self, obj: Any) -> Optional[List[Any]]:
        import torch.distributed as dist

        out: List[Any] = [None] * self.world_size
        try:
            dist.all_gather_object(out, obj, group=self.group)
        except Exception as exc:  # noqa: BLE001 - bounded wait, never a wedge
            if not self.sync_lost:
                self.sync_lost = True
                logger.error(
                    "lockstep sentinel rank %d: SENTINEL SYNC LOST (%s) — "
                    "sidecar ends; a peer is gone or the gloo timeout fired",
                    self.rank,
                    exc,
                )
            return None
        return out

    def _entry_window(self, start: int, stop_: int) -> Tuple[int, List[Any]]:
        """Entries [start, stop_) as (effective_start, tags); ring-bounded."""
        lo = max(start, self._seq - self.ring_len + 8)  # margin vs writer
        lo = max(lo, 0)
        return lo, [self._tags[i % self.ring_len] for i in range(lo, stop_)]

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            if self.diverged or self.sync_lost:
                continue
            self.compare_once()

    def compare_once(self) -> bool:
        """One comparison round. Returns True if a divergence was named."""
        seqs = self._gather(self._seq)
        if seqs is None:
            return False
        m = min(seqs)
        if m <= 0:
            return False
        # stall detection: min frozen while a peer advances
        now = time.monotonic()
        if m != self._last_min:
            self._last_min = m
            self._last_min_t = now
        elif max(seqs) > m and now - self._last_min_t > STALL_AFTER_S:
            if now - self._last_stall_log > 10.0:
                self._last_stall_log = now
                slow = seqs.index(m)
                logger.error(
                    "SENTINEL STALL: rank %d frozen at seq %d for %.1fs while "
                    "leader is at seq %d (all: %s)",
                    slow,
                    m,
                    now - self._last_min_t,
                    max(seqs),
                    seqs,
                )
        # chain compare at position m-1 (chain AFTER the first m events)
        if self._seq - m >= self.ring_len - 8:
            return False  # peer too far behind to compare from our ring
        my_chain = self._chains[(m - 1) % self.ring_len]
        chains = self._gather((m, my_chain))
        if chains is None:
            return False
        if any(c[0] != m for c in chains):
            return False  # a rank advanced between rounds; next round realigns
        vals = [c[1] for c in chains]
        if all(v == vals[0] for v in vals):
            self._verified = m - 1
            return False
        return self._localize(m)

    def _localize(self, m: int) -> bool:
        """Chains differ at m: exchange entries, name the first mismatch."""
        start = self._verified + 1
        if m - start > WINDOW_CAP:
            start = m - WINDOW_CAP
        lo, mine = self._entry_window(start, m)
        got = self._gather((lo, mine))
        if got is None:
            return False
        eff = max(g[0] for g in got)
        rows = [g[1][eff - g[0] :] for g in got]
        n = min(len(r) for r in rows)
        div_seq = -1
        div_tags: List[Any] = []
        for k in range(n):
            tags_k = [r[k] for r in rows]
            if any(t != tags_k[0] for t in tags_k):
                div_seq = eff + k
                div_tags = tags_k
                break
        if div_seq < 0:
            # chains differ but visible windows agree: divergence is older
            # than every surviving ring entry
            logger.error(
                "SENTINEL DIVERGENCE (position > ring): chains differ at seq %d "
                "but entries [%d, %d) agree on all ranks — divergence predates "
                "the ring; increase %s",
                m,
                eff,
                eff + n,
                ENV_RING,
            )
            self.diverged = True
            return True
        # majority = expected
        counts: dict = {}
        for t in div_tags:
            counts[repr(t)] = counts.get(repr(t), 0) + 1
        expected = max(counts, key=counts.get)
        culprits = [r for r, t in enumerate(div_tags) if repr(t) != expected]
        logger.error(
            "SENTINEL DIVERGENCE: first mismatch at seq %d: rank(s) %s ran %s, "
            "expected %s (majority of %d ranks); per-rank: %s",
            div_seq,
            culprits,
            [repr(div_tags[r]) for r in culprits],
            expected,
            self.world_size,
            [repr(t) for t in div_tags],
        )
        self._dump(div_seq, eff, rows)
        self.last_divergence = (div_seq, culprits, div_tags)
        self.diverged = True
        return True

    def _dump(self, div_seq: int, eff: int, rows: List[List[Any]]) -> None:
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            path = os.path.join(
                self.dump_dir, f"sentinel_divergence_{ts}_rank{self.rank}.log"
            )
            with open(path, "w") as f:
                f.write(
                    f"SENTINEL DIVERGENCE at seq {div_seq} "
                    f"(window starts at seq {eff})\n"
                )
                a = max(0, div_seq - eff - DUMP_CONTEXT)
                b = min(min(len(r) for r in rows), div_seq - eff + DUMP_CONTEXT)
                for k in range(a, b):
                    marker = "  <-- FIRST DIVERGENCE" if eff + k == div_seq else ""
                    f.write(
                        f"seq {eff + k}: "
                        + " | ".join(repr(r[k]) for r in rows)
                        + marker
                        + "\n"
                    )
            logger.error("SENTINEL ring dump written: %s", path)
        except Exception:  # noqa: BLE001 - evidence writing must not raise
            logger.exception("lockstep sentinel: ring dump failed")


# -- module-level singleton (production wiring) -----------------------------

_SENTINEL: Optional[LockstepSentinel] = None


def install(rank: int, world_size: int, group: Any) -> None:
    """Arm the process-wide sentinel. Called from model-parallel init."""
    global _SENTINEL
    if _SENTINEL is not None:
        return
    _SENTINEL = LockstepSentinel(
        rank=rank,
        world_size=world_size,
        group=group,
        ring_len=int(os.environ.get(ENV_RING, "65536")),
        interval_s=float(os.environ.get(ENV_INTERVAL, "0.5")),
        dump_dir=os.environ.get(ENV_DUMP_DIR, "/tmp"),
    )
    logger.info(
        "lockstep sentinel ARMED: rank %d/%d, interval %.2fs, ring %d",
        rank,
        world_size,
        _SENTINEL.interval_s,
        _SENTINEL.ring_len,
    )


def armed() -> bool:
    return _SENTINEL is not None


def note_host(family: str) -> None:
    s = _SENTINEL
    if s is not None:
        s.note_host(family)


def note_replay(kind: str, key: Any, index: int) -> None:
    s = _SENTINEL
    if s is not None:
        s.note_replay(kind, key, index)


def note_decision(kind: str, *fields: Any) -> None:
    """One batch-membership decision (drop/admit, with reason).

    Under pure TP these decisions MUST be rank-uniform — every rank runs the
    same batch — so they belong in the position chain: a rank that finishes,
    aborts, retracts or admits a request its peers do not has ALREADY
    diverged, steps before the graph-selection mismatch makes it visible in
    the replay stream. Recording the decision names the diverging code path
    and its reason instead of its downstream symptom (proven need: specimen
    20260808T051733Z diverged at a bs 5-vs-6 full-graph selection with the
    entire preceding host-op stream still in lockstep).
    """
    s = _SENTINEL
    if s is not None:
        s._record(("d", kind) + fields)
