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

"""Per-rank collective-decision recorder and sequence comparator (task #431).

THE BUG FAMILY THIS EXISTS FOR
------------------------------
"A rank-local condition is evaluated BEFORE a group collective, so two ranks
take different collective sequences." This tree has now diagnosed that shape
four times (#94, #194, #312, #431). Every time, the observable was a hang and
every time the diagnosis cost a GPU window, because nothing in the process
records what each rank actually decided.

The transport layer makes the shape structurally easy to hit. Verbatim from
``barlink_bar1.BarlinkBar1Transport.handles``:

    Every condition is **rank-uniform**: it depends only on group-wide
    reconciled sizes [...]. Two ranks must never answer differently here --
    one would run into the collective and the other would not, and the
    result would be a hang instead of an error.

Every *condition* is indeed group-reconciled. The *argument* is not:
``BarlinkCommunicator._select`` is called with ``input_.numel() *
input_.element_size()``, i.e. with THIS rank's own byte count
(``barlink.py`` all_reduce / all_gather / reduce_scatter / broadcast call
sites). For ops whose inputs are contractually equal-shaped that is the same
number on every rank; for anything uneven it is not, and nothing checks which
of the two is in front of it. ``barlink_bar1.barlink_all_gather`` even builds
the plan vector as ``ag_plan([shard] * self.world, ...)`` -- the group vector
faked from the local shard -- while ``ag_plan`` itself was written to take a
genuine per-rank vector and warns that a rank counting differently "would not
be an error but a hang".

WHAT THIS MODULE PROVIDES
-------------------------
1. :class:`DecisionRecorder` -- a bounded, allocation-cheap ring buffer of the
   decisions one rank took, in order. Recording is OFF unless
   ``SGLANG_BARLINK_RECORD_DECISIONS=1``; when off, the hot path costs one
   module-global boolean test.
2. :func:`first_divergence` -- a PURE comparator over ``{rank: [decision]}``
   that names the first index at which the ranks stop agreeing, and on which
   field. Pure means it is testable without GPUs, without torch.distributed
   and without a model: hand it recorded sequences from anywhere.
3. :func:`bar1_fp8_uneven_dcp_notice` -- the named notice for the one feature
   combination whose first boot is slow enough to be mistaken for a hang
   (#431 / #424 / #431-recheck evidence), kept as a pure predicate for the
   same reason. It was a hard refusal until the 2026-08-02 re-check measured
   the arm completing; it is a warning now, with the refusal still reachable
   on request.

The recorder is deliberately transport-agnostic: it records ``(op, nbytes,
path, rounds)``, which is the full input to the dispatch decision and the
full output of it. Any future collective seam can feed it.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

#: How many decisions one rank keeps. A per-layer DCP schedule issues four
#: collectives per attention layer, so a 60-layer model produces ~240 per
#: forward -- a few forwards of history is what a post-mortem needs, and it
#: is bounded so a long run cannot grow it.
DEFAULT_CAPACITY = 4096

ENV_RECORD = "SGLANG_BARLINK_RECORD_DECISIONS"
ENV_DUMP_DIR = "SGLANG_BARLINK_RECORD_DUMP_DIR"

#: Pre-#431-recheck name. Its "1" meaning ("admit the arm") is now the
#: default, so the variable is redundant in that direction -- but it is NOT a
#: no-op: an explicit "0" still means what it always meant, "do not admit
#: this arm", and now restores the hard refusal. See
#: :func:`bar1_fp8_uneven_dcp_notice` for the full compatibility argument.
ENV_ALLOW_FP8_UNEVEN_DCP_BAR1 = "SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1"

#: The explicit force-off. "1" restores the pre-#431-recheck hard refusal for
#: operators who want a boot-time error instead of a slow first boot; "0"
#: states the default in the affirmative. Wins over the legacy variable.
ENV_REFUSE_FP8_UNEVEN_DCP_BAR1 = "SGLANG_BARLINK_REFUSE_FP8_UNEVEN_DCP_BAR1"

_FALSE_STRINGS = ("0", "false", "no", "off", "")

#: File name a rank's dump lands in. The RANK, not the pid: the comparison is
#: per rank and a pid says nothing about which rank wedged.
DUMP_NAME = "barlink_decisions.{group}.rank{rank}.jsonl"


class CollectiveSequenceDivergence(RuntimeError):
    """Ranks did not take the same sequence of collective decisions."""


@dataclass(frozen=True)
class CollectiveDecision:
    """One dispatch decision, in issue order.

    ``op``      the collective's name as the seam knows it.
    ``nbytes``  the value the decision was actually taken on -- the point of
                recording it is that this is the rank-local quantity.
    ``path``    the chosen transport's name, or ``"gloo"`` for the
                host-staged fallback. This is the field whose divergence is
                the hang.
    ``rounds``  how many kernel launches the transport will make. A round
                count that differs across ranks desynchronises bar1's shared
                device round counter just as thoroughly as a different path.
    """

    op: str
    nbytes: int
    path: str
    rounds: int = 0

    def key(self) -> tuple:
        return (self.op, self.nbytes, self.path, self.rounds)


class DecisionRecorder:
    """Bounded, in-order log of one rank's dispatch decisions."""

    def __init__(
        self,
        group: str = "?",
        capacity: int = DEFAULT_CAPACITY,
        dump_path: Optional[str] = None,
    ):
        self.group = group
        self.capacity = int(capacity)
        self._log: deque = deque(maxlen=self.capacity)
        #: Total decisions ever seen, not just the retained ones. Without it,
        #: two snapshots taken after different amounts of history would be
        #: compared as if they started at the same collective.
        self.total = 0
        self.dump_path = dump_path
        self._dump_fh = None

    def record(self, op: str, nbytes: int, path: str, rounds: int = 0) -> None:
        self.total += 1
        decision = CollectiveDecision(op, int(nbytes), path, int(rounds))
        self._log.append(decision)
        if self.dump_path is not None:
            self._append_to_dump(decision)

    def _append_to_dump(self, decision: "CollectiveDecision") -> None:
        """One flushed line per decision -- deliberately unbuffered.

        A wedged rank never gets to flush a buffer, and the decisions that
        matter are exactly the last few before the wedge. One write syscall
        per collective is the price of having them; this whole path is off
        unless a diagnostic run asked for it.
        """
        if self._dump_fh is None:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            self._dump_fh = open(self.dump_path, "a", buffering=1)
        self._dump_fh.write(
            json.dumps(
                {
                    "i": self.total - 1,
                    "op": decision.op,
                    "nbytes": decision.nbytes,
                    "path": decision.path,
                    "rounds": decision.rounds,
                }
            )
            + "\n"
        )
        self._dump_fh.flush()

    def snapshot(self) -> List[CollectiveDecision]:
        return list(self._log)

    def clear(self) -> None:
        self._log.clear()
        self.total = 0

    def __len__(self) -> int:
        return len(self._log)


# --------------------------------------------------------------------------
# Process-global registry. One recorder per communicator group name, because
# a process holds several ('tp', 'dcp', ...) and their sequences are
# independent -- comparing them merged would report a divergence that is only
# an interleaving.
# --------------------------------------------------------------------------
_RECORDERS: Dict[str, DecisionRecorder] = {}

#: Read ONCE at import. The hot path must not do an environment lookup per
#: collective, and a flag that can flip mid-run would produce snapshots that
#: start at different collectives on different ranks -- which is exactly the
#: thing this module exists to distinguish from a real divergence.
_RECORDING = os.environ.get(ENV_RECORD, "0") not in ("0", "", "false", "False")


def recording_enabled() -> bool:
    return _RECORDING


def set_recording_for_test(enabled: bool) -> bool:
    """Flip recording from a test. Returns the previous value.

    Named for what it is. Production code reads the module global directly.
    """
    global _RECORDING
    previous = _RECORDING
    _RECORDING = bool(enabled)
    return previous


def _dump_path_for(group: str) -> Optional[str]:
    """Where this rank's dump goes, or ``None`` when dumping is off.

    The rank is read from the environment the launcher already sets, not from
    a process group: the recorder must work before -- and after -- any group
    is usable, which is the situation a wedged run is in.
    """
    directory = os.environ.get(ENV_DUMP_DIR, "")
    if not directory:
        return None
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or str(os.getpid())
    safe_group = str(group).replace("/", "_").replace(":", "-")
    return os.path.join(directory, DUMP_NAME.format(group=safe_group, rank=rank))


def recorder_for(group: str) -> DecisionRecorder:
    rec = _RECORDERS.get(group)
    if rec is None:
        rec = DecisionRecorder(group=group, dump_path=_dump_path_for(group))
        _RECORDERS[group] = rec
    return rec


def record_decision(
    group: str, op: str, nbytes: int, path: str, rounds: int = 0
) -> None:
    """The seam's entry point. A no-op unless recording is on."""
    if not _RECORDING:
        return
    recorder_for(group).record(op, nbytes, path, rounds)


def snapshots() -> Dict[str, List[CollectiveDecision]]:
    return {g: r.snapshot() for g, r in _RECORDERS.items()}


def clear_all() -> None:
    for rec in _RECORDERS.values():
        rec.clear()
    _RECORDERS.clear()


# --------------------------------------------------------------------------
# The comparator. Pure -- this is what makes the falsifier hermetic.
# --------------------------------------------------------------------------
def first_divergence(
    sequences: Mapping[int, Sequence[CollectiveDecision]],
) -> Optional[str]:
    """Describe the first point at which the ranks stop agreeing, or ``None``.

    Compared positionally, because that is what the transports assume: bar1
    sequences every collective on ONE shared device round counter and waits
    on flag EQUALITY, so decision *i* on rank A must be decision *i* on rank
    B. A set-based comparison would pass on two ranks that ran the same
    collectives in a different order -- which hangs just as hard.

    The returned string names the index, the ranks, and the differing field,
    because "the sequences differ" is not a diagnosis.
    """
    if len(sequences) < 2:
        return None
    ranks = sorted(sequences)
    base_rank = ranks[0]
    base = list(sequences[base_rank])
    for other_rank in ranks[1:]:
        other = list(sequences[other_rank])
        limit = min(len(base), len(other))
        for i in range(limit):
            a, b = base[i], other[i]
            if a.key() == b.key():
                continue
            fields = [
                name
                for name, x, y in (
                    ("op", a.op, b.op),
                    ("nbytes", a.nbytes, b.nbytes),
                    ("path", a.path, b.path),
                    ("rounds", a.rounds, b.rounds),
                )
                if x != y
            ]
            return (
                f"collective #{i}: rank {base_rank} took "
                f"{a.op}/{a.nbytes}B -> {a.path} ({a.rounds} rounds) while "
                f"rank {other_rank} took {b.op}/{b.nbytes}B -> {b.path} "
                f"({b.rounds} rounds); differing field(s): "
                f"{', '.join(fields)}"
            )
        if len(base) != len(other):
            return (
                f"collective count: rank {base_rank} issued {len(base)} "
                f"decisions, rank {other_rank} issued {len(other)}; the "
                f"first {limit} agree, so the shorter rank returned early "
                f"from a branch the longer one did not take"
            )
    return None


def assert_sequences_uniform(
    sequences: Mapping[int, Sequence[CollectiveDecision]],
    context: str = "",
) -> None:
    """Raise :class:`CollectiveSequenceDivergence` on the first mismatch."""
    detail = first_divergence(sequences)
    if detail is None:
        return
    where = f" [{context}]" if context else ""
    raise CollectiveSequenceDivergence(
        f"barlink collective sequences diverge across ranks{where}. {detail}. "
        "This is the rank-local-condition-before-a-group-collective family "
        "(#94, #194, #312, #431): under bar1 every collective is sequenced on "
        "ONE shared device round counter with an equality wait, so a single "
        "divergence is absorbing -- no later collective can ever match again."
    )


def sequences_from_gathered(
    gathered: Iterable[Sequence[CollectiveDecision]],
) -> Dict[int, List[CollectiveDecision]]:
    """``all_gather_object`` output -> the mapping the comparator takes."""
    return {rank: list(seq) for rank, seq in enumerate(gathered)}


def load_dump_dir(
    directory: str, group: Optional[str] = None
) -> Dict[int, List[CollectiveDecision]]:
    """Read back the per-rank dumps a wedged run left behind.

    The post-mortem half of :data:`ENV_DUMP_DIR`: the ranks cannot compare
    their sequences themselves once they are wedged, so they write and
    something outside the process reads. Returns the mapping
    :func:`first_divergence` takes.
    """
    import glob
    import re

    pattern = DUMP_NAME.format(group=group or "*", rank="*")
    out: Dict[int, List[CollectiveDecision]] = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        match = re.search(r"rank(\d+)\.jsonl$", path)
        if match is None:
            continue
        decisions: List[CollectiveDecision] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    # A truncated last line is expected: the rank was killed
                    # mid-write. Everything before it still counts.
                    continue
                decisions.append(
                    CollectiveDecision(
                        row["op"],
                        int(row["nbytes"]),
                        row["path"],
                        int(row.get("rounds", 0)),
                    )
                )
        out[int(match.group(1))] = decisions
    return out


# --------------------------------------------------------------------------
# The named notice (#431; a refusal until the 2026-08-02 re-check, a warning
# since)
# --------------------------------------------------------------------------
def _is_fp8_weight_quant(quantization: Optional[str]) -> bool:
    """True for the fp8 WEIGHT quantizations, not for an fp8 KV cache.

    The distinction is the whole point of the refusal below: the #424 battery
    ran ``--kv-cache-dtype fp8_e4m3`` on BOTH checkpoints and only the fp8
    *checkpoint* wedged, so the KV dtype is not the discriminator and must not
    be part of the predicate.
    """
    if not quantization:
        return False
    q = str(quantization).lower()
    if "fp8" not in q:
        return False
    # 'w8a8_int8', 'compressed-tensors' etc. never match; 'modelopt_fp8',
    # 'fbgemm_fp8', 'fp8' do.
    return True


#: Where the measurement that turned the refusal into a warning lives.
RECHECK_EVIDENCE = "/spinning/gpu-battery-results/2026-08-02_431_recheck/RESULTS.md"
#: The second, independent proof: #435's coupling window carried BOTH FP8
#: layouts through real serving load over BAR1 with an uneven weighted DCP
#: token vector -- nine ACHIEVED=bar1 lines per arm, full probe sets, and no
#: Bar1CollectiveAborted / PeerLost / CollectiveTimeout in any log. The
#: re-check above proves the boot completes; this one proves it serves.
LOAD_EVIDENCE = "/spinning/gpu-battery-results/2026-08-02_435_coupling_fp8bar1/"


@dataclass(frozen=True)
class Bar1Fp8DcpNotice:
    """What to do about the #431 combination, and what to say about it.

    ``refuse``  raise instead of booting. False on the default path.
    ``message`` the full text for the log line or the exception.
    ``source``  what decided ``refuse`` -- "default", or the variable that
                overrode it. Carried separately so a caller can act on the
                decision without parsing the prose.
    """

    refuse: bool
    message: str
    source: str = "default"


def _forced_refusal_from_env() -> tuple[Optional[bool], str]:
    """``(force_refusal, what decided it)`` from the environment.

    ``(None, "default")`` when neither variable is set.

    THE COMPATIBILITY ARGUMENT, in one place because it is the only
    surprising part of this module. Before the 2026-08-02 re-check the
    combination was refused and :data:`ENV_ALLOW_FP8_UNEVEN_DCP_BAR1` was the
    escape hatch: "1" admitted the arm, unset or "0" refused it. The re-check
    measured the arm completing, so admitting it is now the default. Rather
    than let the old variable rot into a no-op, its two values keep their
    literal meanings and one of them now does the work:

    * ``ALLOW=1`` -- "admit this arm". Still honoured, now redundant; the
      notice says so instead of pretending the variable was consulted.
    * ``ALLOW=0`` -- "do not admit this arm". Still honoured, and it is
      exactly the old behaviour: a hard refusal at boot. A launch script that
      pinned ``ALLOW=0`` therefore behaves identically before and after this
      change.

    :data:`ENV_REFUSE_FP8_UNEVEN_DCP_BAR1` is the same switch under a name
    that reads correctly now that refusing is the non-default direction, and
    it wins when both are set -- the explicit name beats the legacy one.
    """
    raw = os.environ.get(ENV_REFUSE_FP8_UNEVEN_DCP_BAR1)
    if raw is not None:
        forced = raw.strip().lower() not in _FALSE_STRINGS
        return forced, f"{ENV_REFUSE_FP8_UNEVEN_DCP_BAR1}={raw!r}"
    raw = os.environ.get(ENV_ALLOW_FP8_UNEVEN_DCP_BAR1)
    if raw is not None:
        allowed = raw.strip().lower() not in _FALSE_STRINGS
        return (not allowed), f"legacy {ENV_ALLOW_FP8_UNEVEN_DCP_BAR1}={raw!r}"
    return None, "default"


def _slow_boot_warning(source: str) -> str:
    legacy_note = ""
    if source.startswith("legacy "):
        legacy_note = (
            f" Note: {source} is set. Admitting this combination is the "
            f"default since the re-check, so that variable no longer changes "
            f"anything in this direction; set "
            f"{ENV_REFUSE_FP8_UNEVEN_DCP_BAR1}=1 if what you wanted was the "
            f"old hard refusal."
        )
    return (
        "barlink BAR1 transport + uneven WEIGHTED DCP + an fp8-quantized "
        "checkpoint: this combination has a SLOW FIRST BOOT. It is not a "
        "hang. On a cold JIT kernel cache the first CUDA-graph capture batch "
        "spends roughly 190 s per rank inside the JIT cold-build window "
        "(measured 184-197 s on three ranks concurrently), during which "
        "deadline-bearing BAR1 collectives run on the 40x extended device "
        "deadline and nothing appears to progress. The full capture pass took "
        "4 min 58 s and the server reached READY 6 min 5 s after launch, then "
        "served 176/176 requests with no aborted collective. Warm boots run "
        "at normal speed. Do not kill the process during those windows -- "
        "grep the server log for 'JIT cold-build window' to see them open and "
        "close. This was a hard refusal until the arm was re-measured on the "
        "tip carrying the BAR1 deadline fix (the launch sites now go through "
        "resolve_timeout_cycles) and the loud abort (Bar1CollectiveAborted "
        "via barlink_abort_gate); under the raw ~30 s cap the same windows "
        "tripped the deadline about six times over and produced the crawl "
        "#424 recorded as a wedge. A second window carried both fp8 layouts "
        "through real serving load over the same transport, also clean. "
        "Evidence: "
        f"{RECHECK_EVIDENCE} (capture), {LOAD_EVIDENCE} (load) and "
        "docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md."
        f"{legacy_note} To refuse this combination at boot instead of warning, "
        f"set {ENV_REFUSE_FP8_UNEVEN_DCP_BAR1}=1."
    )


def _forced_refusal_message(source: str) -> str:
    return (
        "barlink BAR1 transport + uneven WEIGHTED DCP + an fp8-quantized "
        f"checkpoint is refused because {source} asked for it. This is no "
        "longer the default: the arm was re-measured on the tip carrying the "
        "BAR1 deadline fix and completed at the stock-NCCL twin's speed "
        "(capture 12/12 in 4 min 58 s, READY after 6 min 5 s, 176/176 "
        f"requests, no Bar1CollectiveAborted -- {RECHECK_EVIDENCE}). What "
        "remains is a slow FIRST boot: roughly 190 s per rank in the initial "
        "JIT cold-build window on a cold kernel cache. Unset "
        f"{ENV_REFUSE_FP8_UNEVEN_DCP_BAR1} (and "
        f"{ENV_ALLOW_FP8_UNEVEN_DCP_BAR1}, whose '0' means the same thing) to "
        "get the warning instead. Alternatives that avoid the slow window "
        "altogether: run over stock NCCL by unsetting the SGLANG_BARLINK* "
        "block, or use the INT8-W8A8 checkpoint, which the #424 battery ran "
        "over BAR1 with identical DCP settings without any of this."
    )


def bar1_fp8_uneven_dcp_notice(
    *,
    barlink_enabled: bool,
    transport: Optional[str],
    uneven_weighted_dcp: bool,
    quantization: Optional[str],
    force_refusal: Optional[bool] = None,
) -> Optional[Bar1Fp8DcpNotice]:
    """Notice for the #431 combination, or ``None`` if it is not in play.

    Scope, stated narrowly on purpose and unchanged: the #424 battery ran the
    INT8-W8A8 checkpoint over bar1 with uneven weighted DCP through both
    layouts and it completed -- that arm is the fork's recommended INT8
    operating point (+18.6 %/+21.9 % prefill). Nothing here touches any arm
    that was measured to work.

    HISTORY, because the direction of this predicate has flipped once. #424
    saw the fp8 arm wedge and this function refused it. The 2026-08-02 repro
    window then showed the ranks' collective sequences byte-identical -- not
    a dispatch divergence, a BAR1-internal crawl at ~30-40 s per collective,
    which is the raw ``SGLANG_BARLINK_BAR1_CAP_CYCLES`` deadline -- and
    exposed two defects: the BAR1 launch sites bypassed
    ``resolve_timeout_cycles``, so the documented 40x JIT cold-build
    extension never reached them, and a tripped kernel wrote a status word no
    production path read. Both were fixed. The re-check on the fixed tip
    (:data:`RECHECK_EVIDENCE`) then ran the same arm to completion at the
    stock-NCCL twin's speed. So the crawl WAS the raw cap firing repeatedly
    inside three ~190 s capture-warmup windows, outcome 1 of the three the
    analysis listed, and the refusal became this warning.

    What is still unmeasured, and why the notice is loud rather than absent:
    the re-check ran with a warm ``extcache_docker``, so the BAR1 extension's
    own build cost 0.1 s. A genuinely cold first boot adds that build on top
    of the capture windows.

    ``force_refusal`` restores the hard error. ``None`` reads the
    environment; see :func:`_forced_refusal_from_env` for the two variables
    and the backward-compatibility argument.
    """
    if not barlink_enabled:
        return None
    if (transport or "").lower() != "bar1":
        return None
    if not uneven_weighted_dcp:
        return None
    if not _is_fp8_weight_quant(quantization):
        return None

    if force_refusal is None:
        forced, source = _forced_refusal_from_env()
    else:
        forced, source = force_refusal, "force_refusal argument"
    if forced:
        return Bar1Fp8DcpNotice(
            refuse=True, message=_forced_refusal_message(source), source=source
        )
    return Bar1Fp8DcpNotice(
        refuse=False, message=_slow_boot_warning(source), source=source
    )
