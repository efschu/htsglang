# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#603b: what each rank BAKED into its CUDA graphs, compared across ranks.

THE GAP THIS FILLS
------------------
Two instruments already watch this crash family and both are blind in the
same place.

* The collective census (#583) counts host-side collective calls per family
  and diffs the counts across ranks. A captured collective is called ONCE, at
  capture; every subsequent replay executes it with no host code at all. So
  the census sees each in-graph collective exactly once in a whole boot, and
  a replay -- where the wedge happens -- is invisible to it. It also counts
  by FAMILY, so it cannot see a per-rank difference in payload size or in
  which kernel variant was recorded.
* The launch record (``barlink_launch_dump``) samples the transport's
  ``_last_op``/``_unchecked_launches``. ``_unchecked_launches`` deliberately
  does not advance under capture, so at wedge time those fields describe the
  last HOST-path collective, not the in-graph one the ranks are stuck in.

Both therefore exonerate the host path, which is exactly what the 2026-08-06
evidence shows them doing, and neither can say anything about the sequence
that a replay actually runs.

That sequence is fixed at CAPTURE. A replayed graph is a fixed list of kernel
launches: if two ranks recorded different lists, every replay re-runs the same
mismatch, and the BAR1 spin kernels wait on peer flags that the peer's graph
never sets. The wedge is then not an event -- it is a property of the graphs,
established once, at boot, seconds after startup and minutes before the first
crash.

So this module records the list, at capture, and diffs it across ranks. No
wedge is needed to read the result: one boot with graph capture answers the
question.

WHAT IS RECORDED, AND WHY EACH FIELD
------------------------------------
Per collective, in order, within one captured graph:

``op``        the collective the CALLER asked for (``all_reduce``,
              ``all_gather``, ``broadcast``, ...). A different op at the same
              position is a sequence divergence.
``nbytes``    the payload. Equal counts with unequal sizes still deadlock:
              the BAR1 planners (``ar_plan``/``ag_plan``/``bc_plan``) turn
              bytes into a ROUND COUNT, and two ranks with different round
              counts stop pairing up after the first shared round. The census
              cannot see this -- it is one call either way.
``variant``   1 = cooperative multi-block launch, 0 = single block, as chosen
              by ``BarlinkBar1Transport._kernel``. Recorded because that
              choice is made from ``moved`` bytes, and on at least one path
              (``barlink_all_to_all_single``) ``moved`` EXCLUDES the rank's
              own block -- an expression that is only rank-uniform while the
              blocks are equal. A per-rank variant split is invisible to every
              other instrument in the tree.
``callsite``  the first non-barlink frames, as ``file:line``. Rank-uniform by
              construction (same source, same lines), so it is safe to
              compare, and it is what turns "position 41 differs" into a line
              of code to go and read.

COST
----
Zero on replay: nothing here runs on the replay path, because nothing here
runs outside a capture. At capture it is one tuple append and a short frame
walk per recorded collective, a few thousand times, once per boot. The
cross-rank comparison is one small ``all_gather_object`` on the gloo group,
ONCE per boot, riding the scheduler cadence the #583 census already uses.

ARMED BY DEFAULT, for the reason #583 and the #605 marks are: an instrument
that has to be switched on before it can explain a crash explains no crashes.
``SGLANG_BARLINK_CAPTURE_CENSUS=0`` disables it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureCensus",
    "capture_census",
    "capture_census_enabled",
    "segment",
    "note",
    "format_local_capture_census",
]

#: Kill switch. Default ON -- see the module docstring.
ENV_ENABLE = "SGLANG_BARLINK_CAPTURE_CENSUS"

#: Where the per-rank dump is written. One file per rank per boot.
ENV_DIR = "SGLANG_BARLINK_CAPTURE_CENSUS_DIR"

#: Default dump location: beside the launch sampler's files, so a wedge
#: investigation finds the capture record and the launch record together.
DEFAULT_DIR = "/spinning/wedge-catch-603b"

#: How many callsite frames to keep. Three is enough to separate "the layer
#: loop" from "which seam inside it" without turning the record into a stack
#: dump that no diff can be read out of.
CALLSITE_FRAMES = 3

#: Substrings identifying frames INSIDE the transport. Those frames are the
#: instrument's own plumbing and say nothing about who asked for the
#: collective, so they are skipped when the callsite is resolved.
_INTERNAL_MARKERS = ("barlink_bar1.py", "barlink_capture_census.py", "barlink.py")


def capture_census_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1") not in ("0", "false", "False")


def _callsite() -> str:
    """``file:line`` of the first frames outside the transport.

    Walks frames directly instead of ``traceback.extract_stack``: the latter
    reads source lines off disk to fill in the text, which this never uses.
    Never raises -- a diagnostic must not be the reason a capture fails.
    """
    try:
        out: List[str] = []
        frame = sys._getframe(1)
        while frame is not None and len(out) < CALLSITE_FRAMES:
            name = frame.f_code.co_filename
            if not any(marker in name for marker in _INTERNAL_MARKERS):
                out.append(f"{os.path.basename(name)}:{frame.f_lineno}")
            frame = frame.f_back
        return "<".join(out) if out else "?"
    except Exception:  # noqa: BLE001
        return "?"


class CaptureCensus:
    """Per-rank record of the collectives baked into each captured graph."""

    def __init__(self) -> None:
        #: segment key -> ordered records. A plain dict: capture is
        #: single-threaded, and the lock below only guards the rare reader.
        self._segments: Dict[str, List[Tuple[str, int, int, str]]] = {}
        #: Stack of open segment keys. A stack, not a scalar, because a
        #: backend may nest a capture inside another context; the innermost
        #: open segment owns the record.
        self._open: List[str] = []
        self._lock = threading.Lock()
        #: Collectives recorded while a capture was running but no segment
        #: was open. Non-zero means a capture path exists that this module
        #: does not segment -- reported, never silently dropped, because an
        #: unsegmented graph is exactly the graph a diff would miss.
        self.unsegmented = 0
        self.compared = False

    # -- recording ---------------------------------------------------------

    def open_segment(self, key: str) -> None:
        """Begin recording ONE captured graph.

        Every call is a distinct graph, so a repeated key gets an occurrence
        suffix rather than appending into the previous one. Without this, the
        decode runner and the EAGLE draft runner -- which both capture through
        the same backend and can produce the same ShapeKey -- would merge two
        different graphs into one segment, and the diff would report a
        position inside a sequence that no single graph ever runs.

        The suffix is rank-uniform for the same reason the keys are: it counts
        capture calls, and the capture order is the property being verified.
        A rank that captured a different NUMBER of graphs shifts its suffixes
        and the segment key sets stop matching -- which is itself the
        divergence, reported rather than hidden.
        """
        if key in self._segments:
            n = 2
            while f"{key}#{n}" in self._segments:
                n += 1
            key = f"{key}#{n}"
        self._segments[key] = []
        self._open.append(key)

    def close_segment(self) -> None:
        if self._open:
            self._open.pop()

    def note(self, op: str, nbytes: int, variant: Optional[int]) -> None:
        """Record one collective. Called only while a capture is running."""
        try:
            key = self._open[-1] if self._open else None
            if key is None:
                self.unsegmented += 1
                key = "<unsegmented>"
                self._segments.setdefault(key, [])
            self._segments[key].append(
                (str(op), int(nbytes), -1 if variant is None else int(variant),
                 _callsite())
            )
        except Exception:  # noqa: BLE001 - instrument must not raise
            pass

    # -- reading -----------------------------------------------------------

    def segments(self) -> Dict[str, List[Tuple[str, int, int, str]]]:
        with self._lock:
            return {k: list(v) for k, v in self._segments.items()}

    @staticmethod
    def render(record: Tuple[str, int, int, str]) -> str:
        op, nbytes, variant, site = record
        return f"{op}|{nbytes}|v{variant}|{site}"

    def digests(self) -> Dict[str, Tuple[int, str]]:
        """Per segment: how many collectives, and a digest of the sequence.

        The digest, not the sequence, is what crosses the wire in the common
        case: on a healthy boot the answer is "equal" and shipping a few
        thousand tuples per rank to establish that would be a boot-time cost
        paid forever for a diagnostic. The full records follow only for the
        segments that actually disagree.
        """
        out: Dict[str, Tuple[int, str]] = {}
        for key, records in self.segments().items():
            body = "\n".join(self.render(r) for r in records)
            out[key] = (len(records), hashlib.sha1(body.encode()).hexdigest())
        return out

    def details(self, keys: Sequence[str]) -> Dict[str, List[str]]:
        segs = self.segments()
        return {k: [self.render(r) for r in segs.get(k, [])] for k in keys}

    # -- the cross-rank diff ----------------------------------------------

    def compare_across_ranks(self, group, world_size: int, rank: int) -> bool:
        """Diff the captured sequences across ranks. ``True`` when they agree.

        Runs on the GLOO cpu group, never on the device transport -- the
        device path is the thing under suspicion, and an instrument that
        deadlocks with its subject explains nothing.

        RANK-UNIFORMITY OF THE INSTRUMENT ITSELF. Both phases are entered by
        every rank or by none: phase 1 is unconditional, and the decision to
        enter phase 2 is computed from the phase-1 result, which is identical
        on every rank because it is what the all_gather produced. A diagnostic
        that took a rank-local branch around a collective would introduce the
        exact defect it is looking for.

        WARN-NEVER-RAISE, like the #583 census: it reports, it does not
        repair, and it must never be why a healthy boot fails.
        """
        try:
            import torch.distributed as dist

            if group is None or world_size <= 1:
                return True
            local = self.digests()
            gathered: List[Optional[Dict[str, Tuple[int, str]]]] = [None] * world_size
            dist.all_gather_object(gathered, local, group=group)
            self.compared = True

            keys = sorted({k for g in gathered if g for k in g})
            bad = [
                k for k in keys
                if len({(g or {}).get(k, (-1, "missing")) for g in gathered}) > 1
            ]
            if not bad:
                logger.info(
                    "barlink capture census (rank %d): the captured collective "
                    "sequences AGREE across all %d ranks -- %d graph segment(s), "
                    "%d collectives recorded on this rank%s. A replay therefore "
                    "runs the same op/size/variant list on every rank.",
                    rank,
                    world_size,
                    len(keys),
                    sum(n for n, _ in local.values()),
                    "" if not self.unsegmented
                    else f" ({self.unsegmented} of them outside any segment)",
                )
                return True

            # Phase 2: only the disagreeing segments, in full, so the log names
            # the first differing POSITION and its callsite rather than just
            # asserting that something differs.
            detail: List[Optional[Dict[str, List[str]]]] = [None] * world_size
            dist.all_gather_object(detail, self.details(bad), group=group)
            self._log_mismatch(bad, gathered, detail, rank, world_size)
            return False
        except Exception as exc:  # noqa: BLE001 - instrument must not raise
            logger.warning(
                "barlink capture census: cross-rank comparison unavailable "
                "(%s: %s); the per-rank dump is unaffected",
                type(exc).__name__,
                exc,
            )
            return True

    @staticmethod
    def _log_mismatch(
        bad: Sequence[str],
        gathered: Sequence[Optional[Dict[str, Tuple[int, str]]]],
        detail: Sequence[Optional[Dict[str, List[str]]]],
        rank: int,
        world_size: int,
    ) -> None:
        for key in bad:
            counts = [(g or {}).get(key, (-1, "missing"))[0] for g in gathered]
            rows = [(d or {}).get(key, []) for d in detail]
            first = None
            for i in range(max((len(r) for r in rows), default=0)):
                column = {r[i] if i < len(r) else "<end>" for r in rows}
                if len(column) > 1:
                    first = i
                    break
            logger.error(
                "BARLINK CAPTURE CENSUS DIVERGENCE (rank %d): graph segment %s "
                "was captured DIFFERENTLY on different ranks, so every replay "
                "of it re-runs the same mismatch and the BAR1 spin kernels wait "
                "on peer flags no peer will set. Collectives per rank: %s. "
                "First differing position: %s. %s",
                rank,
                key,
                counts,
                "none (the sequences differ only in length)" if first is None
                else str(first),
                "" if first is None else "; ".join(
                    f"rank{r}={rows[r][first] if first < len(rows[r]) else '<end>'}"
                    for r in range(world_size)
                ),
            )

    # -- the per-rank dump -------------------------------------------------

    def dump_to_file(self, rank: int, directory: Optional[str] = None) -> Optional[str]:
        """Write this rank's full record. Never raises; returns the path."""
        try:
            directory = directory or os.environ.get(ENV_DIR, DEFAULT_DIR)
            os.makedirs(directory, exist_ok=True)
            # #622/#649: scope the record by boot, because the un-scoped name
            # silently destroyed the evidence it exists to preserve.
            #
            # On 2026-08-07 the ordered per-segment collective list was the one
            # datum that would have separated "the replay stopped at a segment
            # boundary" from "a transport was frozen behind another" for the
            # 16:08 wedge. By the time it was read, three later boots had each
            # rewritten capture_census_rank*.txt at this fixed path -- and the
            # last of them ran with barlink disabled, so every file said "0
            # collectives". The forensic question was unanswerable not because
            # the instrument failed but because it overwrote itself.
            #
            # The VRAM flight recorder already solved exactly this by scoping
            # its marks to a boot id (SGLANG_VRAM_FLIGHT_BOOT_ID, "so the
            # appended file stays honest across boots"). Same fix here: prefer
            # the boot commit the launcher publishes, fall back to the pid,
            # which is still unique per boot.
            #
            # The stable un-suffixed name is ALSO written, as a copy, so
            # existing tooling and the log line that points at it keep working
            # and always show the current boot.
            boot_id = (
                os.environ.get("SGLANG_VRAM_FLIGHT_BOOT_ID")
                or os.environ.get("SGLANG_BOOT_COMMIT")
                or str(os.getpid())
            )
            # Defend the path component: a boot id is used unvalidated here and
            # comes from the environment.
            boot_id = "".join(c for c in boot_id if c.isalnum() or c in "-_")[:40]
            path = os.path.join(directory, f"capture_census_rank{rank}.txt")
            boot_path = os.path.join(
                directory, f"capture_census_rank{rank}.{boot_id}.txt"
            )
            segs = self.segments()
            with open(path, "w") as fh:
                fh.write(
                    f"# barlink capture census, rank {rank}: the collectives "
                    f"baked into each captured CUDA graph, in order.\n"
                    f"# fields: op|nbytes|variant|callsite  "
                    f"(variant 1=cooperative grid, 0=1blk, -1=not applicable)\n"
                    f"# unsegmented records: {self.unsegmented}\n"
                )
                for key in sorted(segs):
                    fh.write(f"\n[{key}] {len(segs[key])} collectives\n")
                    for i, record in enumerate(segs[key]):
                        fh.write(f"{i:5d} {self.render(record)}\n")
            # The boot-scoped copy is what survives the next boot. Written
            # after the primary file and guarded separately: if this fails,
            # the caller still gets the stable path rather than nothing.
            try:
                shutil.copyfile(path, boot_path)
            except OSError:
                pass
            return path
        except Exception:  # noqa: BLE001
            return None


#: Process-wide instance. One per scheduler process, which is one per rank.
_CAPTURE_CENSUS = CaptureCensus()


def capture_census() -> CaptureCensus:
    return _CAPTURE_CENSUS


@contextmanager
def segment(key: str):
    """Mark the recorded pass of ONE captured graph.

    Wraps the graph-recording call in the cuda-graph backends. The key must be
    rank-uniform (it is built from the ShapeKey, which has no rank-local
    component); a rank-local key would make every segment look like a
    divergence and hide the real one.
    """
    if not capture_census_enabled():
        yield
        return
    _CAPTURE_CENSUS.open_segment(key)
    try:
        yield
    finally:
        _CAPTURE_CENSUS.close_segment()


def note(op: str, nbytes: int, variant: Optional[int] = None) -> None:
    """Record one collective being baked into the graph currently captured."""
    if not capture_census_enabled():
        return
    _CAPTURE_CENSUS.note(op, nbytes, variant)


def format_local_capture_census(rank: int) -> str:
    """This rank's per-segment summary, for the abort path. Takes NO collective."""
    digests = _CAPTURE_CENSUS.digests()
    if not digests:
        return (
            f"barlink capture census (rank {rank}): no collectives were "
            f"recorded under graph capture."
        )
    body = ", ".join(
        f"{key} {n}x {digest[:12]}" for key, (n, digest) in sorted(digests.items())
    )
    return (
        f"barlink capture census (rank {rank}, per captured graph): {body}. "
        f"Compare against the peers' -- the segment whose count or digest "
        f"differs is the graph whose replay cannot pair up."
    )
