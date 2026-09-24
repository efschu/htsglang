# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H28 (Task #53): the all-reduce census of one decode round, by class.

WHAT THE ROUND LINE CANNOT SAY TODAY. Boot x138 (24.09.), TP0, stationary::

    gpu-ms 31.3 (compute 17.0, wait 14.7)
    (wait by family: spec_verify:tp.all_reduce 7.6/96x min0.012, ...)

96 all-reduces of 20480 B (4 rows x 2560 x bf16) per round, 79 us mean, a
12-us minimum. Under Form A those 96 are TWO classes that the family name
folds into one (eager trace of the same boot, ``site=``):

* 48x the MoE-INPUT CARRIER, ``form_a_worker_forward.publish_moe_input`` on
  the host / ``receive_moe_input`` on a worker -- an all-reduce whose worker
  addends are zeros, i.e. a broadcast of the host's MoE input;
* 48x the MoE COMBINE, ``qwen2_moe.py`` after the routed experts -- a real
  three-way sum of expert partials.

There is no attention/o_proj/LSE all-reduce in a Form A decode round: the
host holds the dense chain whole (``form_a_dense_is_unsharded``), so
``LinearBase.reduce`` and the attention output reduce return early.

The two classes answer different questions. On the host, the carrier's
span is the TRANSPORT (the workers have been waiting in it since their
previous combine), while the combine's span is transport PLUS the time the
slowest worker's MoE (router, pool fetch, experts) runs past the host's own.
A mean over both says neither. This module does two things, both OFF unless
``SGLANG_WEG2_AR_ROUND_CENSUS`` is set:

1. ``carrier_scope()`` names the carrier's family ``tp.moe_carrier`` via the
   collective clock's ``label_scope`` -- in eager spans AND in the graph
   event-record nodes (the family is bound at capture), so the round line
   splits ``spec_verify:tp.moe_carrier`` from ``spec_verify:tp.all_reduce``.
2. ``RoundCensus`` folds the round families and every ``EVERY`` rounds
   prints ONE line per rank::

       BARLINK-ROUND-CENSUS rank=0 rounds=50 ar=96.0 bytes=20480 mode=oneshot
       us_mean=79.2 us_min=12.0 floor_ms=1.15 skew_ms=6.45
       carrier=48.0x/14.1us/min9.0 combine=48.0x/144.2us/min12.0

   ``floor_ms`` = per-round count x per-class minimum, the transport a
   perfectly synchronous round would still pay; ``skew_ms`` = the rest,
   waiting for a peer. A transport lever can win at most ``floor_ms``
   minus its own floor; ``skew_ms`` is not a transport property at all.

``bytes`` and ``mode`` come from the CAPTURE: ``note_captured`` records
each all-reduce size a captured graph bakes in together with the algorithm
the transport chose for it (``oneshot`` / ``mesh`` / ``ring`` ...). Replay
executes no Python, so the census never touches the hot path; its whole
cost is one dict fold per round on the scheduler thread.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CARRIER_FAMILY",
    "COMBINE_FAMILY",
    "census_on",
    "carrier_scope",
    "note_captured",
    "captured_classes",
    "RoundCensus",
    "on_decode_round",
]

#: The carrier's family name. Deliberately NOT a suffix of ``tp.all_reduce``:
#: a reader that sums ``tp.all_reduce`` must see that it now misses the
#: carrier, not silently still include half of it.
CARRIER_FAMILY = "tp.moe_carrier"
#: What the combine keeps being called -- the dispatch site's own family.
COMBINE_FAMILY = "tp.all_reduce"
#: Family suffixes (after the phase prefix) that count as all-reduces here.
_AR_FAMILIES = (CARRIER_FAMILY, COMBINE_FAMILY)
_PHASE_SEP = ":"


def census_on() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_AR_ROUND_CENSUS.get())


def _every() -> int:
    from sglang.srt.environ import envs

    return max(1, int(envs.SGLANG_WEG2_AR_ROUND_CENSUS_EVERY.get()))


def carrier_scope(clock=None):
    """Label the MoE-input carrier's collective ``tp.moe_carrier``.

    A no-op context when the census is off -- the carrier then records under
    the dispatch site's ``tp.all_reduce`` exactly as before. The scope changes
    only the NAME of the span; the collective itself is untouched.
    """
    if not census_on():
        return nullcontext()
    if clock is None:
        from sglang.srt.utils.collective_clock import collective_clock

        clock = collective_clock()
    return clock.label_scope(CARRIER_FAMILY)


# ---------------------------------------------------------------------------
# Capture-time classes: which sizes a captured graph bakes in, and how.
# ---------------------------------------------------------------------------
_CAPTURED: Dict[Tuple[str, int, str], int] = {}


def note_captured(op: str, nbytes: int, transport) -> None:
    """Record one collective laid into a CUDA graph. Capture-time only.

    ``transport`` is whatever ``BarlinkCommunicator._select`` returned; its
    ``algorithm_for`` (bar1) names the kernel topology, otherwise the
    transport's class name is the mode.
    """
    mode = "host-staged"
    if transport is not None:
        algo = getattr(transport, "algorithm_for", None)
        if callable(algo) and op == "all_reduce":
            try:
                mode = str(algo(int(nbytes)))
            except Exception:  # noqa: BLE001 - a census never breaks a capture
                mode = type(transport).__name__
        else:
            mode = type(transport).__name__
    key = (str(op), int(nbytes), mode)
    _CAPTURED[key] = _CAPTURED.get(key, 0) + 1


def captured_classes(op: str = "all_reduce") -> List[Tuple[int, str, int]]:
    """``(nbytes, mode, count)`` of ``op``, most frequent first."""
    rows = [(b, m, n) for (o, b, m), n in _CAPTURED.items() if o == op]
    rows.sort(key=lambda r: (-r[2], -r[0]))
    return rows


def _reset_captured() -> None:  # tests
    _CAPTURED.clear()


# ---------------------------------------------------------------------------
# The round fold.
# ---------------------------------------------------------------------------
def _split(name: str) -> str:
    return name.rsplit(_PHASE_SEP, 1)[-1]


class RoundCensus:
    """Folds round families; emits one line every ``every`` rounds."""

    def __init__(self, rank: int, every: Optional[int] = None,
                 emit=None, classes=None) -> None:
        self.rank = int(rank)
        self.every = int(every) if every is not None else _every()
        self._emit = emit or logger.info
        self._classes = classes or captured_classes
        self.lines: List[str] = []
        self._reset()

    def _reset(self) -> None:
        self.rounds = 0
        # family suffix -> [total_ms, count, min_ms]
        self.acc: Dict[str, List[float]] = {}

    def on_round(self, families: Mapping[str, Sequence[float]]) -> Optional[str]:
        """``families``: name -> (total_ms, count[, min_ms]), one round.

        A round with no all-reduce family is not counted -- it is not a round
        of the form this census describes (e.g. a draft-only bracket).
        """
        seen = False
        for name, stat in families.items():
            fam = _split(name)
            if fam not in _AR_FAMILIES:
                continue
            total = float(stat[0])
            count = float(stat[1])
            mn = float(stat[2]) if len(stat) > 2 and stat[2] else 0.0
            if count <= 0:
                continue
            seen = True
            slot = self.acc.get(fam)
            if slot is None:
                self.acc[fam] = [total, count, mn]
            else:
                slot[0] += total
                slot[1] += count
                if mn > 0.0 and (slot[2] <= 0.0 or mn < slot[2]):
                    slot[2] = mn
        if not seen:
            return None
        self.rounds += 1
        if self.rounds < self.every:
            return None
        line = self.format_line()
        self.lines.append(line)
        self._emit(line)
        self._reset()
        return line

    def format_line(self) -> str:
        n = max(1, self.rounds)
        total_ms = sum(v[0] for v in self.acc.values())
        count = sum(v[1] for v in self.acc.values())
        mins = [v[2] for v in self.acc.values() if v[2] > 0.0]
        us_mean = 1000.0 * total_ms / count if count else 0.0
        us_min = 1000.0 * min(mins) if mins else 0.0
        # Per-round transport floor: each class's count times ITS minimum.
        floor_ms = sum(v[1] * v[2] for v in self.acc.values()) / n
        per_round_ms = total_ms / n
        classes = self._classes("all_reduce")
        if classes:
            nbytes, mode, _ = classes[0]
            modes = sorted({m for _, m, _ in classes})
            mode_s = mode if len(modes) == 1 else "+".join(modes)
        else:
            nbytes, mode_s = 0, "unknown"
        parts = [
            "BARLINK-ROUND-CENSUS rank=%d rounds=%d ar=%.1f bytes=%d mode=%s"
            % (self.rank, self.rounds, count / n, nbytes, mode_s),
            "us_mean=%.1f us_min=%.1f ar_ms=%.2f floor_ms=%.2f skew_ms=%.2f"
            % (us_mean, us_min, per_round_ms, floor_ms,
               max(per_round_ms - floor_ms, 0.0)),
        ]
        for label, fam in (("carrier", CARRIER_FAMILY), ("combine", COMBINE_FAMILY)):
            v = self.acc.get(fam)
            if v is None or v[1] <= 0:
                continue
            parts.append(
                "%s=%.1fx/%.1fus/min%.1f"
                % (label, v[1] / n, 1000.0 * v[0] / v[1], 1000.0 * v[2])
            )
        if CARRIER_FAMILY not in self.acc:
            # Without the carrier label the combine family holds both classes;
            # say so instead of letting 'combine=' be read as the combine.
            parts.append("split=off")
        return " ".join(parts)


_CENSUS: Dict[int, RoundCensus] = {}


def on_decode_round(rank: int, families: Mapping[str, Sequence[float]]) -> Optional[str]:
    """Scheduler-thread hook, one call per emitted decode round. Off: no-op."""
    if not census_on():
        return None
    census = _CENSUS.get(int(rank))
    if census is None:
        census = RoundCensus(rank)
        _CENSUS[int(rank)] = census
    return census.on_round(families)


def _reset_rounds() -> None:  # tests
    _CENSUS.clear()
