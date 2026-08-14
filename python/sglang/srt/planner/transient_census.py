# SPDX-License-Identifier: Apache-2.0
"""Per-load-state transient measurement for the #485 cut gate.

WHY THIS EXISTS (law 27, and law 31 which is law 27 repeated).

The cut gate has to answer "can this rank hold this stage AND still run".
The second half of that is a TRANSIENT: the working set a rank draws below
its at-rest level once traffic arrives. Three shifts fed the gate a single
scalar for it, and the scalar was measured at a prefill trigger. Measured on
the same rank of the same rig:

    deep-prefill A/B load   ->   956 MiB drawn below at-rest
    22-minute mixed soak    ->  1989 MiB (planner cut) / 3148 MiB (ship)

So "the transient" is not one number, and the load state that produced a
number is part of the number. A gate fed the 956 admitted a cut that the
1989 refuses, and metal broke the corridor on it -- twice, on two cuts.

WHAT THIS MODULE DOES. It records, per rank, the driver-visible free-memory
MINIMUM reached in each load state the rank actually served, labelled by that
load state, and writes it next to the residency census. The gate then funds
the WORST state, because the worst state is one the deployment will serve.
Nothing here is fitted; every number is a measurement with its load state
attached.

WHY THE DRIVER'S FREE COLUMN, SAMPLED AFTER THE BATCH. The corridor law is
written on the NVML free column, and that column tracks the caching
allocator's RESERVED high-water, not its live bytes -- reserved memory is not
returned to the driver between iterations. So a per-batch sample after the
forward observes the same quantity the corridor sampler observes from
outside, and does not need to catch the instant of peak liveness.

OFF BY DEFAULT AND READ-ONLY. Gated on SGLANG_TRANSIENT_CENSUS. When off,
`note()` returns on a boolean and the boot is byte-identical (law 30: an
instrument that must not perturb what it measures has to be default-off AND
read-only -- this one allocates nothing on the device and writes no state
that any later boot reads back).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_MIB = 1024.0 * 1024.0

#: Sample one batch in this many. A driver query is tens of microseconds and
#: the quantity being tracked is a high-water mark, which a stride cannot
#: hide for long -- it only delays the observation by a few batches.
_DEFAULT_STRIDE = 8

#: Rewrite the artifact at most this often, so a 20-minute window leaves a
#: usable file even if the process is killed without ever exiting cleanly --
#: which is exactly what happened to the boot that motivated this module.
_WRITE_INTERVAL_S = 30.0

__all__ = [
    "census_enabled",
    "begin",
    "note",
    "snapshot",
    "write_now",
]


def census_enabled() -> bool:
    """True when the operator asked for the transient census on this boot."""
    try:
        from sglang.srt import environ as _environ

        return bool(_environ.envs.SGLANG_TRANSIENT_CENSUS.get())
    except Exception:
        return False


def _census_dir() -> Optional[str]:
    try:
        from sglang.srt import environ as _environ

        return _environ.envs.SGLANG_RESIDENCY_CENSUS_DIR.get()
    except Exception:
        return None


def _stride() -> int:
    try:
        from sglang.srt import environ as _environ

        return max(1, int(_environ.envs.SGLANG_TRANSIENT_CENSUS_STRIDE.get()))
    except Exception:
        return _DEFAULT_STRIDE


class TransientCensus:
    """Per-load-state minima of the driver's free column, on one rank."""

    def __init__(self, pp_rank: int, gpu_name: str, baseline_free_bytes: int):
        self.pp_rank = int(pp_rank)
        self.gpu_name = gpu_name
        #: Free bytes at rest, i.e. after capture and before any request. The
        #: draw is measured against THIS, not against the card's total, so it
        #: is a property of the load and not of the cut.
        self.baseline_free_bytes = int(baseline_free_bytes)
        #: Highest free level seen while serving, RECORDED BUT NOT USED AS
        #: THE REFERENCE -- see the note on ``draw_mib``. Kept because on a
        #: phase-flip boot it is several GiB above the post-capture reading
        #: and a reader needs to know that before calling anything "at rest".
        self.max_free_bytes = int(baseline_free_bytes)
        self.min_free_bytes: Dict[str, int] = {}
        self.samples: Dict[str, int] = {}
        self._seen = 0
        self._last_write = 0.0
        self._lock = threading.Lock()

    def note(self, load_state: str, free_bytes: int) -> None:
        with self._lock:
            if free_bytes > self.max_free_bytes:
                self.max_free_bytes = int(free_bytes)
            self.samples[load_state] = self.samples.get(load_state, 0) + 1
            prev = self.min_free_bytes.get(load_state)
            if prev is None or free_bytes < prev:
                self.min_free_bytes[load_state] = int(free_bytes)

    def draw_mib(self) -> Dict[str, float]:
        """Per load state: the draw BELOW THE POST-CAPTURE LEVEL.

        WHY THE POST-CAPTURE LEVEL AND NOT THE HIGHEST FREE OBSERVED. I tried
        the latter mid-shift, on the reasoning that the boot-time backing swap
        releases the non-resident layout after capture so free at rest ends up
        GiB higher. That reasoning is right about the memory and wrong about
        the reference: the gate's OTHER terms -- in particular
        ``fixed_overhead_mib``, calibrated as ``nvml_used - params - pools``
        -- are all measured AT THE POST-CAPTURE POINT, so the residual already
        counts the layout that is later released. Referencing the transient to
        the later, higher level would charge that same layout twice, on the
        rank where the constraint binds. A transient is a difference, and a
        difference is only meaningful against the baseline the rest of the
        model uses.

        The consequence, stated plainly: under a running phase flip there is
        no single "at rest" -- free on rank0 of the reference rig oscillates
        by ~6.5 GiB between flip phases. That is why the corridor law is
        judged on the observed MINIMUM by a separate instrument, and why the
        raw minima are written into the payload here: a reader who wants a
        different reference can compute it, and does not have to trust this
        one.
        """
        at_rest = self.baseline_free_bytes
        return {
            state: max(0.0, (at_rest - low) / _MIB)
            for state, low in self.min_free_bytes.items()
        }

    def min_free_mib(self) -> Dict[str, float]:
        """Per load state, the raw minimum. Recorded so a reader can
        re-reference the draws to a baseline measured some other way."""
        return {s: low / _MIB for s, low in self.min_free_bytes.items()}

    def worst(self) -> Optional[str]:
        draws = self.draw_mib()
        if not draws:
            return None
        return max(draws, key=lambda k: draws[k])

    def payload(self) -> dict:
        draws = self.draw_mib()
        worst = self.worst()
        return {
            "pp_rank": self.pp_rank,
            "gpu_name": self.gpu_name,
            "baseline_free_mib": self.baseline_free_bytes / _MIB,
            "max_free_observed_mib": self.max_free_bytes / _MIB,
            "min_free_mib_by_load_state": self.min_free_mib(),
            "transient_mib_by_load_state": draws,
            "samples_by_load_state": dict(self.samples),
            "worst_load_state": worst,
            "worst_transient_mib": draws.get(worst) if worst else None,
        }

    @staticmethod
    def _staging_path(path: str) -> str:
        """Where this PROCESS stages its census before publishing it.

        MUST be unique per process. ``pp_rank`` is not enough to make it so:
        under pure tensor parallelism every rank carries ``pp_rank=0``, so all
        of them derive the same output path -- and, before this, the same
        staging path. Three schedulers then opened one
        ``transient_pp0.json.tmp`` with mode ``"w"`` at overlapping times.

        Write-tmp-then-rename makes the PUBLISH atomic. It does not make the
        STAGING exclusive, and a shared staging name defeats the whole idiom:
        measured hermetically, 23 of 600 concurrent flushes were LOST because
        a peer's ``os.replace`` renamed the file away mid-write, and on metal
        the published file was left unparsable (#363 window 2026-08-14).
        """
        return f"{path}.{os.getpid()}.tmp"

    def write(self, out_dir: str) -> Optional[str]:
        if not self.min_free_bytes:
            return None
        tmp = None
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"transient_pp{self.pp_rank}.json")
            tmp = self._staging_path(path)
            with open(tmp, "w") as fh:
                json.dump(self.payload(), fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return path
        except OSError as exc:
            logger.warning("transient census could not be written: %s", exc)
            # A staging file that never got published would otherwise sit in
            # the census directory looking like data to anyone globbing it.
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return None


_CENSUS: Optional[TransientCensus] = None

#: Module-level fast path. The scheduler tests this once per batch on EVERY
#: boot, so the default path must cost a global lookup and a boolean test and
#: nothing else -- not an enum attribute read, not a function call into a
#: disabled instrument.
ARMED = False


def begin(pp_rank: int, gpu_name: str, baseline_free_bytes: int) -> None:
    """Arm the census with the at-rest baseline this rank starts from."""
    global _CENSUS, ARMED
    if not census_enabled():
        return
    _CENSUS = TransientCensus(pp_rank, gpu_name, baseline_free_bytes)
    ARMED = True
    logger.info(
        "TRANSIENT CENSUS armed on pp_rank %d (%s): at-rest free %.1f MiB. "
        "Per-load-state minima will be written to %s",
        pp_rank,
        gpu_name,
        baseline_free_bytes / _MIB,
        _census_dir() or "<no SGLANG_RESIDENCY_CENSUS_DIR set>",
    )


def note(load_state: str) -> None:
    """Record one sample for this load state. Cheap, strided, best-effort."""
    census = _CENSUS
    if census is None:
        return
    census._seen += 1
    if census._seen % _stride():
        return
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return
    census.note(load_state, int(free_bytes))

    now = time.monotonic()
    if now - census._last_write >= _WRITE_INTERVAL_S:
        census._last_write = now
        out_dir = _census_dir()
        if out_dir:
            census.write(out_dir)


def snapshot() -> Optional[dict]:
    return _CENSUS.payload() if _CENSUS is not None else None


def write_now() -> Optional[str]:
    out_dir = _census_dir()
    if _CENSUS is None or not out_dir:
        return None
    return _CENSUS.write(out_dir)
