"""#232 -- the tipping point of ONE MLP split candidate, measured on this rig.

The planner can predict what moving dense-MLP mass toward the compute-strong
rank buys in prefill and costs in decode. #264 measured two candidates by
hand and the prediction was out by a factor of 2.2, so the dashboard shows
what was measured, per candidate, and says "not measured" for the rest. This
module is the machinery behind that: one click boots one server with one
candidate, runs the #264 sequence against it, tears it down, and appends one
row to the store the Benchmark tab reads.

Three quantities per candidate, because a split is a trade and one number
cannot express a trade:

* what it COSTS in decode -- ms per verify round and decode tok/s,
* what it BUYS in prefill -- cold-prefill tok/s at ~20k tokens,
* what it SPENDS in KV -- ``max_total_num_tokens``.

Design notes that are load-bearing rather than stylistic:

* **One candidate per click.** The ladder is not swept. A sweep is hours of
  GPU time behind a control that looks like the others, and the reader
  usually wants one comparison, not eight.
* **The measurement runs in its own interpreter.** The dashboard process
  serves other panels while this runs and must not inherit the child's
  lifetime; the parent only reads the file the child wrote.
* **The GPU lock is held by the process that touches the GPU** -- the child,
  not the dashboard -- taken with an atomic ``mkdir`` and released in a
  ``finally``. A stale lock whose owner is gone is reclaimed and said so.
* **Reserves are derived, not pinned.** #265 found that a concentrated
  candidate spends the slack the balanced plan left on rank 0, so the same
  ``--rank-auto-reserve-mib`` that boots ``2,1,1`` OOMs ``6,1,1`` in its
  first real prefill. The bump comes from the checkpoint's own GDN prefill
  scratch (``ServerArgs.gdn_prefill_scratch_mib``) at the candidate's share,
  and a boot that still dies of OOM is retried once at the reserve the
  failure itself asked for. What is never done is hardcoding "4500".
* **An unbootable candidate is a result, not an error.** It is recorded with
  its reason and rendered as such; a candidate that cannot boot at a
  sensible reserve is exactly the kind of thing the reader came for.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid as _uuid
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

__all__ = [
    "SPLIT_PROBE_VERSION",
    "LADDER",
    "SplitProbeResult",
    "SplitProbeStore",
    "SplitProbeRejected",
    "SplitProbeJob",
    "SplitProbeJobStore",
    "JOBS",
    "default_store_path",
    "gpu_lock",
    "GpuLockBusy",
    "reserve_for_candidate",
    "run_split_probe",
    "tipping_point_table",
    "import_264_rows",
]

#: Bumped when a stored row's meaning changes. Rows of another version are
#: ignored rather than reinterpreted -- see SplitProbeStore.load.
SPLIT_PROBE_VERSION = 1

CACHE_DIR = os.path.expanduser("~/.cache/sglang")

#: Ran here, on this rig, through run_split_probe.
MEASURED = "measured"
#: Measured here by hand before the endpoint existed, transcribed verbatim.
IMPORTED = "imported"
PROVENANCE = (MEASURED, IMPORTED)

#: The candidates the auto-performance optimizer enumerates on this rig class,
#: in ascending concentration, with "auto" (let the optimizer choose) first.
#: A ladder entry is a ROW in the table whether or not it was ever measured;
#: an unmeasured one carries a button, never a number.
LADDER: Tuple[str, ...] = (
    "auto",
    "2,1,1",
    "4,1,1",
    "5,1,1",
    "6,1,1",
    "8,1,1",
    "10,1,2",
    "16,1,2",
)

#: The candidate every delta is taken against: what the optimizer picks when
#: it is left alone. A table of concentrated candidates with no reference
#: point says nothing about whether concentrating is worth it.
BASELINE_CANDIDATE = "auto"

#: Runbook 4.1 reserve for the 27B-FP8 TP=3 class, in CUDA device order
#: (device 0 = the 5090). The request may override it; the concentration bump
#: below is applied on top of whichever is in force.
DEFAULT_RESERVE_MIB: Tuple[int, ...] = (3000, 2700, 2700)

#: A boot that OOMs anyway is retried once with the concentrated rank's
#: reserve raised by this fraction of the reserve already planned there. The
#: GDN scratch bump above is exact but small -- it is the only term of the
#: concentration demand with a closed form -- so the retry carries the rest.
RETRY_RESERVE_FRACTION = 0.5

DEFAULT_PREFILL_TOKENS = 20000
DEFAULT_DECODE_SECONDS = 25
#: Decode tokens per second assumed when converting the decode WINDOW into a
#: fixed token count. Both arms then do the same decode work, so tok/s is
#: comparable; the window only has to land in the 20-30 s band.
DECODE_TOKENS_PER_SECOND = 60

#: The #264 decode prompt, verbatim. It is part of the measurement: decode
#: throughput follows the output CONTENT on this rig, so two candidates are
#: only comparable when they were asked the same thing.
DECODE_PROMPT = (
    "Write a detailed technical explanation of how tensor parallelism works "
    "in large language model inference. Cover the sharding of attention "
    "heads and MLP matrices, the collective operations required, and why "
    "heterogeneous GPUs complicate the split. Be thorough and precise."
)

#: Characters of the decode output kept in the row, so a reader can see that
#: the server produced language rather than a repetition loop.
OUTPUT_HEAD_CHARS = 300

BOOT_TIMEOUT_S = 900.0
TEARDOWN_GRACE_S = 30.0
#: A boot plus a cold 20k prefill plus a 25 s decode is 6-8 minutes; the
#: OOM retry can double it. The job timeout has to clear both.
JOB_TIMEOUT_S = 2100.0
JOB_TTL_S = 24 * 3600.0

PENDING = "pending"
RUNNING = "running"
OK = "ok"
ERROR = "error"


class SplitProbeRejected(ValueError):
    """A row that does not carry what it claims to carry."""


class GpuLockBusy(RuntimeError):
    """Another split probe (or its remains) owns the cards."""


# ---------------------------------------------------------------------------
# paths


def default_store_path() -> str:
    return os.path.join(CACHE_DIR, "split_probe.jsonl")


def default_lock_path() -> str:
    return os.path.join(CACHE_DIR, "split_probe.gpulock")


# ---------------------------------------------------------------------------
# the GPU lock
#
# mkdir is the atomic primitive: it either creates the directory or fails with
# EEXIST, with no window between the check and the create. The owner file is
# written after the fact and is advisory -- it exists so a human (and the
# stale-reclaim path) can see WHO holds the lock, not to decide who does.


class _GpuLock:
    def __init__(self, path: str, label: str = ""):
        self.path = path
        self.label = label
        self.reclaimed_from: Optional[int] = None
        self._held = False

    def _owner(self) -> dict:
        try:
            with open(os.path.join(self.path, "owner.json")) as f:
                return json.load(f)
        except Exception:
            return {}

    def _owner_alive(self) -> bool:
        pid = self._owner().get("pid")
        if not isinstance(pid, int):
            # No readable owner: treat as alive, because guessing wrong here
            # means two processes measuring the same cards at once.
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self) -> "_GpuLock":
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            os.mkdir(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if self._owner_alive():
                owner = self._owner()
                raise GpuLockBusy(
                    "the cards are held by another split probe "
                    f"(pid {owner.get('pid', '?')}, {owner.get('label', 'unlabelled')}, "
                    f"since {time.strftime('%H:%M:%S', time.localtime(owner.get('at', 0)))}). "
                    "Nothing was started; the running one is the measurement."
                )
            self.reclaimed_from = self._owner().get("pid")
            logger.warning(
                "reclaiming the split-probe GPU lock at %s: its owner (pid %s) is gone",
                self.path,
                self.reclaimed_from,
            )
        self._held = True
        try:
            with open(os.path.join(self.path, "owner.json"), "w") as f:
                json.dump(
                    {"pid": os.getpid(), "label": self.label, "at": time.time()}, f
                )
        except OSError:  # pragma: no cover - advisory only
            pass
        return self

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            os.unlink(os.path.join(self.path, "owner.json"))
        except OSError:
            pass
        try:
            os.rmdir(self.path)
        except OSError:  # pragma: no cover - defensive
            logger.warning("could not remove the split-probe lock at %s", self.path)

    def __enter__(self) -> "_GpuLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def gpu_lock(path: Optional[str] = None, label: str = "") -> _GpuLock:
    """The cards, held for the length of ONE candidate's boot.

    Used as a context manager so the release is in a ``finally`` the caller
    cannot forget -- a probe that dies holding the lock would block every
    later one, and the next probe would have no way to tell that from a
    measurement in progress.
    """
    return _GpuLock(path or default_lock_path(), label=label)


def busy_cards() -> Dict[str, str]:
    """``{gpu_uuid: pids}`` for cards with a live compute process.

    A card in this map belongs to someone else. We never fight for it and
    never broad-kill: the probe refuses to start and says which card.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except Exception:
        return {}
    busy: Dict[str, str] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        busy.setdefault(parts[0], "")
        busy[parts[0]] = (busy[parts[0]] + " " + parts[1]).strip()
    return busy


def clock_context() -> dict:
    """The clock state the row was measured under.

    Two rows taken at different clocks are not comparable, and the difference
    is invisible in the numbers themselves. Recording it costs one
    ``nvidia-smi`` call and makes a surprising delta checkable afterwards.
    """
    ctx: dict = {"cards": []}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,clocks.max.sm,clocks.sm,"
                "power.limit,persistence_mode",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except Exception as e:
        ctx["error"] = str(e)
        return ctx
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        ctx["cards"].append(
            {
                "index": parts[0],
                "name": parts[1],
                "sm_clock_max_mhz": parts[2],
                "sm_clock_mhz": parts[3],
                "power_limit_w": parts[4],
                "persistence": parts[5],
            }
        )
    return ctx


# ---------------------------------------------------------------------------
# the reserve for a candidate


def _vector_of(candidate: str, tp_size: int) -> Optional[List[int]]:
    if candidate == BASELINE_CANDIDATE:
        return None
    try:
        vec = [int(x) for x in candidate.split(",")]
    except ValueError:
        raise SplitProbeRejected(
            f"candidate {candidate!r} is neither {BASELINE_CANDIDATE!r} nor a "
            "comma-separated MLP vector"
        )
    if len(vec) != tp_size or min(vec) < 1:
        raise SplitProbeRejected(
            f"candidate {candidate!r} needs {tp_size} positive entries, one per rank"
        )
    return vec


def gdn_scratch_mib(model_path: str, tp_size: int, share: float) -> Optional[float]:
    """The checkpoint's GDN prefill scratch at one rank's head share, or None.

    Isolated behind its own function so the caller can degrade to the plain
    reserve with a recorded note when the checkpoint has no GDN layers or its
    config cannot be read here.
    """
    try:
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path=model_path,
            tp_size=tp_size,
            trust_remote_code=True,
        )
        return args.gdn_prefill_scratch_mib(share)
    except Exception as e:  # pragma: no cover - depends on the checkpoint
        logger.info("GDN prefill scratch unavailable for %s: %s", model_path, e)
        return None


def reserve_for_candidate(
    candidate: str,
    tp_size: int,
    base_reserve: Sequence[int] = DEFAULT_RESERVE_MIB,
    model_path: str = "",
    scratch_fn=gdn_scratch_mib,
) -> Tuple[List[int], str]:
    """``(reserve_mib_per_rank, note)`` for one candidate.

    #265: the KV pool follows the token vector scaled by the TIGHTEST rank, so
    concentrating the dense MLP onto one rank moves the tight rank onto that
    card and spends the slack the balanced plan was leaving there. The reserve
    that boots the balanced candidate therefore does not boot the concentrated
    one. The part of that demand which is computable from the checkpoint
    rather than guessed is the GDN prefill scratch, which grows with the
    rank's share: the bump is exactly that growth, per rank.

    This is a first attempt at the reserve, not a guarantee. The scratch is
    the smallest of the terms concentration moves, so a candidate can still
    OOM at the bumped reserve -- run_split_probe then retries once at the
    reserve the failure asked for, and the row records both.
    """
    reserve = [int(x) for x in base_reserve][:tp_size]
    while len(reserve) < tp_size:
        reserve.append(reserve[-1] if reserve else 2700)
    vec = _vector_of(candidate, tp_size)
    if vec is None:
        return reserve, "runbook reserve, unbumped: the optimizer's own choice"
    if not model_path:
        return reserve, "runbook reserve, unbumped: no model path to size the scratch"

    total = float(sum(vec))
    base_share = 1.0 / tp_size
    base_scratch = scratch_fn(model_path, tp_size, base_share)
    if base_scratch is None:
        return (
            reserve,
            "runbook reserve, unbumped: this checkpoint has no GDN layers to size",
        )
    bumped: List[int] = []
    parts: List[str] = []
    for r in range(tp_size):
        share = vec[r] / total
        if share <= base_share:
            bumped.append(reserve[r])
            continue
        scratch = scratch_fn(model_path, tp_size, share)
        if scratch is None:  # pragma: no cover - defensive
            bumped.append(reserve[r])
            continue
        delta = int(round(max(0.0, scratch - base_scratch)))
        bumped.append(reserve[r] + delta)
        if delta:
            parts.append(f"rank {r} +{delta} MiB at share {share:.2f}")
    if len(bumped) < tp_size:  # pragma: no cover - defensive
        bumped = reserve
    note = (
        "GDN prefill scratch bump (#265): " + "; ".join(parts)
        if parts
        else "runbook reserve, unbumped: the scratch does not grow at this share"
    )
    return bumped, note


# ---------------------------------------------------------------------------
# the result row


@dataclasses.dataclass
class SplitProbeResult:
    """One candidate, measured once. Every number carries its own denominator."""

    candidate: str
    model_path: str = ""
    tp_size: int = 3
    provenance: str = MEASURED
    version: int = SPLIT_PROBE_VERSION
    timestamp: Optional[float] = None

    #: What the server actually materialised, which for "auto" is what the
    #: optimizer chose and for a pinned vector is that vector.
    chosen_vector: str = ""
    reserve_mib: List[int] = dataclasses.field(default_factory=list)
    #: Set when the first boot OOMed and the reserve was raised for a retry.
    reserve_retried_from: Optional[List[int]] = None
    reserve_note: str = ""

    #: None with a reason in ``unbootable``: a candidate that cannot boot at a
    #: sensible reserve is a finding, not a hole in the table.
    unbootable: str = ""

    max_total_num_tokens: Optional[int] = None

    prefill_tokens: Optional[int] = None
    prefill_cached_tokens: Optional[int] = None
    prefill_wall_s: Optional[float] = None
    prefill_tok_s: Optional[float] = None

    decode_tokens: Optional[int] = None
    decode_wall_s: Optional[float] = None
    decode_tok_s: Optional[float] = None
    accept_length: Optional[float] = None
    verify_ct: Optional[int] = None
    ms_per_verify: Optional[float] = None

    #: Per rank, from the #252 prefill line: ``{rank, chunks, gpu_ms,
    #: compute_ms, wait_ms}``. Read the SPREAD of wait across ranks -- the
    #: rank with the largest shard shows the smallest wait.
    rank_compute_wait: List[dict] = dataclasses.field(default_factory=list)

    output_head: str = ""
    output_verdict: str = ""
    clock_context: dict = dataclasses.field(default_factory=dict)
    source: str = ""
    boot_log: str = ""

    def has_numbers(self) -> bool:
        return self.decode_tok_s is not None and self.prefill_tok_s is not None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SplitProbeResult":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def judge_output(text: str) -> str:
    """A one-line verdict on the decode output.

    Throughput on this rig follows the output CONTENT, so a row whose decode
    tok/s came from a repetition loop is not comparable with one that came
    from prose. The judgement is coarse on purpose: it separates "language"
    from "degenerate", and anything finer would be a quality suite.
    """
    body = (text or "").strip()
    if not body:
        return "no output: the decode produced nothing"
    words = body.split()
    if len(words) < 20:
        return f"short output ({len(words)} words): too little to judge"
    distinct = len(set(w.lower() for w in words))
    ratio = distinct / len(words)
    if ratio < 0.15:
        return (
            f"degenerate: {distinct} distinct of {len(words)} words "
            f"({ratio:.0%}) -- this looks like a repetition loop, not prose"
        )
    if ratio < 0.25:
        return (
            f"repetitive: {distinct} distinct of {len(words)} words ({ratio:.0%}) "
            "-- treat the throughput as content-bound"
        )
    return f"coherent prose: {distinct} distinct of {len(words)} words ({ratio:.0%})"


# ---------------------------------------------------------------------------
# the store


class SplitProbeStore:
    """Append-only rows, newest wins per candidate.

    Deliberately NOT ``planner.results_store``: a ResultEntry is a full
    benchmark record keyed by model/quant/hardware and has nowhere to put a
    per-rank compute/wait split, an ms-per-verify figure or an unbootable
    verdict, and its ingest guard rejects a row that is not one. This store
    keeps the same discipline -- a row that claims to be measured must carry
    numbers, the load path re-runs the guard so a hand-edited file cannot
    smuggle one in -- on the schema this study actually has.
    """

    def __init__(self, entries: Optional[Sequence[SplitProbeResult]] = None):
        self._entries: List[SplitProbeResult] = list(entries or [])

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[SplitProbeResult]:
        return list(self._entries)

    def check(self, entry: SplitProbeResult) -> None:
        if entry.provenance not in PROVENANCE:
            raise SplitProbeRejected(
                f"provenance {entry.provenance!r} is not one of {PROVENANCE}"
            )
        if not entry.candidate:
            raise SplitProbeRejected("a row without a candidate names nothing")
        if entry.unbootable:
            return
        if not entry.has_numbers():
            raise SplitProbeRejected(
                f"{entry.candidate}: a bootable row must carry both a prefill and "
                "a decode figure -- a half-measured candidate is not a tipping point"
            )

    def ingest(self, entry: SplitProbeResult) -> None:
        self.check(entry)
        if entry.timestamp is None:
            entry.timestamp = time.time()
        self._entries.append(entry)

    def append_to_file(self, path: str, entry: SplitProbeResult) -> None:
        self.ingest(entry)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry.to_json()) + "\n")

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            for e in self._entries:
                f.write(json.dumps(e.to_json()) + "\n")
        os.replace(tmp, path)

    def latest(self) -> Dict[str, SplitProbeResult]:
        """The newest row per candidate. A re-measurement supersedes."""
        best: Dict[str, SplitProbeResult] = {}
        for e in self._entries:
            prev = best.get(e.candidate)
            if prev is None or (e.timestamp or 0) >= (prev.timestamp or 0):
                best[e.candidate] = e
        return best

    @classmethod
    def load(
        cls, path: Optional[str] = None, strict: bool = False
    ) -> "SplitProbeStore":
        path = path or default_store_path()
        store = cls()
        if not os.path.exists(path):
            return store
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    if strict:
                        raise
                    continue
                if int(d.get("version") or 0) != SPLIT_PROBE_VERSION:
                    logger.info(
                        "ignoring a split-probe row of version %s (this build reads %d)",
                        d.get("version"),
                        SPLIT_PROBE_VERSION,
                    )
                    continue
                entry = SplitProbeResult.from_json(d)
                try:
                    store.ingest(entry)
                except SplitProbeRejected:
                    if strict:
                        raise
        return store


# ---------------------------------------------------------------------------
# the #264 rows, transcribed
#
# Measured by hand on 2026-07-28 before this endpoint existed, at TP=3 on
# 5090+2x3080 with Qwen3.6-27B-FP8, NEXTN 3/1/4, fp8_e4m3 KV, ctx 32768.
# They are the two rows that make the table a comparison on the first load
# instead of an empty frame with buttons.

_ROWS_264: Tuple[dict, ...] = (
    {
        "candidate": "auto",
        "chosen_vector": "2,1,1",
        "source": "#264-A 2026-07-28",
        "reserve_mib": [3000, 2700, 2700],
        "max_total_num_tokens": 502528,
        "prefill_tokens": 20000,
        "prefill_cached_tokens": 0,
        "prefill_wall_s": 17.398,
        "prefill_tok_s": 1149.6,
        "decode_tokens": 1500,
        "decode_wall_s": 16.006,
        "decode_tok_s": 93.71,
        "accept_length": 3.054989816700611,
        "verify_ct": 491,
        "ms_per_verify": 32.599,
        "rank_compute_wait": [
            {
                "rank": 0,
                "chunks": 9,
                "gpu_ms": 1730.5,
                "compute_ms": 175.4,
                "wait_ms": 1555.1,
            },
            {
                "rank": 1,
                "chunks": 9,
                "gpu_ms": 1728.9,
                "compute_ms": 540.9,
                "wait_ms": 1188.0,
            },
            {
                "rank": 2,
                "chunks": 9,
                "gpu_ms": 1729.1,
                "compute_ms": 518.7,
                "wait_ms": 1210.4,
            },
        ],
    },
    {
        "candidate": "6,1,1",
        "chosen_vector": "6,1,1",
        "source": "#264-B 2026-07-28",
        "reserve_mib": [4500, 2700, 2700],
        "reserve_retried_from": [3000, 2700, 2700],
        "reserve_note": (
            "raised by hand after the first boot OOMed in its first real prefill"
        ),
        "max_total_num_tokens": 261632,
        "prefill_tokens": 20000,
        "prefill_cached_tokens": 0,
        "prefill_wall_s": 16.076,
        "prefill_tok_s": 1244.1,
        "decode_tokens": 1500,
        "decode_wall_s": 18.542,
        "decode_tok_s": 80.90,
        "accept_length": 3.0181086519114686,
        "verify_ct": 497,
        "ms_per_verify": 37.307,
        "rank_compute_wait": [
            {
                "rank": 0,
                "chunks": 9,
                "gpu_ms": 1597.6,
                "compute_ms": 220.6,
                "wait_ms": 1377.0,
            },
            {
                "rank": 1,
                "chunks": 9,
                "gpu_ms": 1597.5,
                "compute_ms": 406.8,
                "wait_ms": 1190.7,
            },
            {
                "rank": 2,
                "chunks": 9,
                "gpu_ms": 1597.6,
                "compute_ms": 380.3,
                "wait_ms": 1217.3,
            },
        ],
    },
)

#: 2026-07-28 00:30 local, the wall clock of the #264 boots.
_TS_264 = 1785191400.0


def import_264_rows(
    path: Optional[str] = None, model_path: str = ""
) -> List[SplitProbeResult]:
    """Seed the store with the two rows #264 measured by hand.

    Idempotent by ``source``: re-running never duplicates a row. The rows are
    tagged ``imported`` and carry their own date, so a reader can see they
    predate the endpoint rather than mistaking them for something this build
    produced.
    """
    path = path or default_store_path()
    store = SplitProbeStore.load(path)
    have = {e.source for e in store.entries()}
    added: List[SplitProbeResult] = []
    for i, row in enumerate(_ROWS_264):
        if row["source"] in have:
            continue
        entry = SplitProbeResult.from_json(dict(row))
        entry.provenance = IMPORTED
        entry.model_path = model_path
        entry.tp_size = 3
        entry.timestamp = _TS_264 + i
        entry.output_verdict = "not recorded: the run predates the output check"
        store.append_to_file(path, entry)
        added.append(entry)
    return added


# ---------------------------------------------------------------------------
# the table


def _pct(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base in (None, 0):
        return None
    return round((new - base) / base * 100.0, 1)


def tipping_point_table(
    path: Optional[str] = None,
    ladder: Sequence[str] = LADDER,
    store: Optional[SplitProbeStore] = None,
) -> dict:
    """Every ladder candidate as a row: measured with deltas, or not measured.

    The deltas are against the baseline candidate, and only exist when the
    baseline itself was measured -- a percentage against nothing is a number
    with no meaning, and leaving it out is the honest rendering.
    """
    store = store if store is not None else SplitProbeStore.load(path)
    latest = store.latest()
    base = latest.get(BASELINE_CANDIDATE)
    order = list(ladder) + [c for c in latest if c not in ladder]

    rows: List[dict] = []
    for cand in order:
        e = latest.get(cand)
        if e is None:
            rows.append(
                {
                    "candidate": cand,
                    "measured": False,
                    "is_baseline": cand == BASELINE_CANDIDATE,
                    "missing_reason": (
                        "not measured on this rig. One boot with this split "
                        "produces the row."
                    ),
                }
            )
            continue
        row = {
            "candidate": cand,
            "measured": True,
            "is_baseline": cand == BASELINE_CANDIDATE,
            "provenance": e.provenance,
            "source": e.source,
            "timestamp": e.timestamp,
            "chosen_vector": e.chosen_vector,
            "reserve_mib": e.reserve_mib,
            "reserve_note": e.reserve_note,
            "reserve_retried_from": e.reserve_retried_from,
            "unbootable": e.unbootable,
            "prefill_tok_s": e.prefill_tok_s,
            "decode_tok_s": e.decode_tok_s,
            "ms_per_verify": e.ms_per_verify,
            "accept_length": (
                round(e.accept_length, 3) if e.accept_length is not None else None
            ),
            "max_total_num_tokens": e.max_total_num_tokens,
            "rank_compute_wait": e.rank_compute_wait,
            "output_verdict": e.output_verdict,
            "output_head": e.output_head,
            "clock_context": e.clock_context,
        }
        if base is not None and cand != BASELINE_CANDIDATE and not e.unbootable:
            row["delta"] = {
                "prefill_pct": _pct(e.prefill_tok_s, base.prefill_tok_s),
                "decode_pct": _pct(e.decode_tok_s, base.decode_tok_s),
                "ms_per_verify_pct": _pct(e.ms_per_verify, base.ms_per_verify),
                "max_kv_pct": _pct(
                    float(e.max_total_num_tokens or 0) or None,
                    float(base.max_total_num_tokens or 0) or None,
                ),
                "against": BASELINE_CANDIDATE,
            }
        rows.append(row)

    measured = sum(1 for r in rows if r.get("measured"))
    if base is None:
        summary = (
            f"{measured} of {len(rows)} candidates measured; no baseline yet, so "
            f"no deltas -- measure {BASELINE_CANDIDATE!r} first"
        )
    else:
        summary = (
            f"{measured} of {len(rows)} candidates measured; deltas against "
            f"{BASELINE_CANDIDATE!r} (chosen: {base.chosen_vector or 'n/a'})"
        )
    return {
        "rows": rows,
        "baseline": BASELINE_CANDIDATE,
        "measured_count": measured,
        "ladder": list(ladder),
        "summary": summary,
        "cost_note": (
            "About 6-8 minutes of exclusive GPU time per candidate: one boot, "
            "one cold 20k prefill, one 25 s decode, teardown. A candidate that "
            "needs its reserve raised takes two boots."
        ),
    }


# ---------------------------------------------------------------------------
# the measurement

_PREFILL_LINE = re.compile(
    r"\bTP(\d+)\].*Prefill rank batch, #new-token: (\d+), #cached-token: (\d+), "
    r"#chunks: (\d+), gpu-ms: ([0-9.]+) \(compute ([0-9.]+), wait ([0-9.]+)\)"
)
_CHOSEN = re.compile(r"CHOSEN MLP vector: ([0-9,]+)")
_READY = re.compile(r"The server is fired up and ready to roll")
_OOM = re.compile(r"(CUDA out of memory|torch\.OutOfMemoryError|out of memory)", re.I)
_MAXTOK = re.compile(r"max_total_num_tokens=(\d+)")


def parse_rank_compute_wait(text: str) -> List[dict]:
    """Mean per-rank compute/wait over the steady full-size prefill chunks.

    Only ``#chunks: 1`` lines are used: a line that folded several forwards
    together carries their sum, and the partial last chunk of a prefill is a
    different amount of work from the full ones. The split is absent for
    graph-replayed forwards by design, so decode rounds never appear here.
    """
    per_rank: Dict[int, List[Tuple[float, float, float]]] = {}
    sizes: Dict[int, int] = {}
    for m in _PREFILL_LINE.finditer(text):
        rank, new_tok, _cached, chunks, gpu_ms, compute, wait = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            float(m.group(5)),
            float(m.group(6)),
            float(m.group(7)),
        )
        if chunks != 1:
            continue
        sizes[new_tok] = sizes.get(new_tok, 0) + 1
        per_rank.setdefault(rank, []).append((new_tok, gpu_ms, compute, wait))
    if not per_rank:
        return []
    # The steady chunk size is the one that occurs most often; anything else
    # is a warmup forward or a prefill tail.
    steady = max(sizes.items(), key=lambda kv: kv[1])[0]
    out: List[dict] = []
    for rank in sorted(per_rank):
        rows = [r for r in per_rank[rank] if r[0] == steady]
        if not rows:
            continue
        n = len(rows)
        out.append(
            {
                "rank": rank,
                "chunks": n,
                "new_token": steady,
                "gpu_ms": round(sum(r[1] for r in rows) / n, 1),
                "compute_ms": round(sum(r[2] for r in rows) / n, 1),
                "wait_ms": round(sum(r[3] for r in rows) / n, 1),
            }
        )
    return out


def _http_get(url: str, timeout: float = 30.0) -> dict:
    with urlopen(url, timeout=timeout) as r:  # noqa: S310 - loopback only
        return json.loads(r.read().decode())


def _http_post(url: str, body: dict, timeout: float = 900.0) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback only
        return json.loads(r.read().decode())


def launch_command(
    model_path: str,
    candidate: str,
    reserve_mib: Sequence[int],
    port: int,
    tp_size: int = 3,
    context_length: int = 32768,
    python: str = "",
) -> List[str]:
    """The runbook 4.1 recipe, with the candidate pinned.

    ``--enable-metrics`` is not optional on this rig, and
    ``--enable-metrics-for-all-schedulers`` is what makes the per-rank figures
    exist at all; the device timer is set in the environment alongside it.
    """
    cmd = [
        python or sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp-size",
        str(tp_size),
        "--rank-gpu-id",
        ",".join(str(i) for i in range(tp_size)),
        "--rank-auto-reserve-mib",
        ",".join(str(int(x)) for x in reserve_mib),
        "--rank-tp-ratio",
        "auto-performance",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        str(context_length),
        "--trust-remote-code",
        "--max-running-requests",
        "16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-metrics",
        "--enable-metrics-for-all-schedulers",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    vec = _vector_of(candidate, tp_size)
    if vec is not None:
        cmd += ["--rank-mlp-ratio", ",".join(str(v) for v in vec)]
    return cmd


def launch_env(worktree: str = "", base: Optional[dict] = None) -> dict:
    env = dict(os.environ if base is None else base)
    env["SGLANG_UNEVEN_DCP"] = "1"
    env["SGLANG_UNEVEN_DCP_WEIGHTED"] = "1"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"
    # The pinned vector goes on the command line; an inherited env override
    # would silently win over it and we would measure the wrong candidate.
    for stale in (
        "SGLANG_UNEVEN_MLP_VECTOR",
        "SGLANG_UNEVEN_TOKEN_VECTOR",
        "SGLANG_UNEVEN_MOE_VECTOR",
    ):
        env.pop(stale, None)
    if worktree:
        env["PYTHONPATH"] = os.path.join(worktree, "python")
    return env


class _Server:
    """One booted server, torn down by the process group that started it.

    Only PIDs we spawned are ever signalled. The orphan check afterwards is a
    report, not a licence to kill someone else's process.
    """

    def __init__(self, cmd: Sequence[str], env: dict, log_path: str, port: int):
        self.cmd = list(cmd)
        self.env = env
        self.log_path = log_path
        self.port = port
        self.proc: Optional[subprocess.Popen] = None

    def log_text(self) -> str:
        try:
            with open(self.log_path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float = BOOT_TIMEOUT_S) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                text = self.log_text()
                if _OOM.search(text):
                    raise _BootOOM(text[-4000:])
                raise RuntimeError(
                    f"the server exited with rc={self.proc.returncode} before it was "
                    f"ready:\n{text[-3000:]}"
                )
            if _READY.search(self.log_text()):
                return
            time.sleep(3.0)
        raise RuntimeError(f"the server was not ready within {timeout:.0f} s")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=TEARDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=TEARDOWN_GRACE_S)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        try:
            self._log.close()
        except Exception:  # pragma: no cover
            pass
        self.proc = None


class _BootOOM(RuntimeError):
    """The boot or its first real prefill ran out of VRAM."""


def _orphan_note(before: Dict[str, str], busy_fn=busy_cards) -> str:
    """What is still holding VRAM that was not holding it before we started."""
    after = busy_fn()
    new = {k: v for k, v in after.items() if k not in before}
    if not new:
        return ""
    return "VRAM still held after teardown on " + ", ".join(
        f"{uuid} (pid {pids})" for uuid, pids in sorted(new.items())
    )


def measure_server(
    port: int,
    prefill_tokens: int = DEFAULT_PREFILL_TOKENS,
    decode_seconds: int = DEFAULT_DECODE_SECONDS,
    seed: int = 232,
    get=_http_get,
    post=_http_post,
) -> dict:
    """The #264 sequence against a booted server. Touches no GPU directly."""
    base = f"http://127.0.0.1:{port}"
    out: dict = {}

    info = get(f"{base}/get_server_info")
    out["max_total_num_tokens"] = info.get("max_total_num_tokens")

    # Random ids, so no prefix-cache hit is possible: cached_tokens == 0 in the
    # answer is the evidence that a prefill was actually computed.
    rng = random.Random(seed + port)
    ids = [rng.randrange(1000, 100000) for _ in range(prefill_tokens)]
    t0 = time.perf_counter()
    r = post(
        f"{base}/generate",
        {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        },
    )
    wall = time.perf_counter() - t0
    meta = r.get("meta_info", {})
    prompt_tokens = meta.get("prompt_tokens") or prefill_tokens
    out["prefill_tokens"] = prompt_tokens
    out["prefill_cached_tokens"] = meta.get("cached_tokens")
    out["prefill_wall_s"] = round(wall, 3)
    out["prefill_tok_s"] = round(prompt_tokens / wall, 1) if wall > 0 else None

    # A short warm request first, so the decode window does not carry
    # first-touch costs that belong to neither candidate.
    post(
        f"{base}/generate",
        {
            "text": "Hello.",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
        },
    )

    max_new = int(decode_seconds * DECODE_TOKENS_PER_SECOND)
    t0 = time.perf_counter()
    r = post(
        f"{base}/generate",
        {
            "text": DECODE_PROMPT,
            "sampling_params": {
                "max_new_tokens": max_new,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        },
    )
    wall = time.perf_counter() - t0
    meta = r.get("meta_info", {})
    completion = meta.get("completion_tokens") or 0
    verify_ct = meta.get("spec_verify_ct")
    out["decode_tokens"] = completion
    out["decode_wall_s"] = round(wall, 3)
    out["decode_tok_s"] = round(completion / wall, 2) if wall > 0 else None
    out["accept_length"] = meta.get("spec_accept_length")
    out["verify_ct"] = verify_ct
    out["ms_per_verify"] = round(wall * 1000.0 / verify_ct, 3) if verify_ct else None
    text = r.get("text") or ""
    out["output_head"] = text[:OUTPUT_HEAD_CHARS]
    out["output_verdict"] = judge_output(text)
    return out


def run_split_probe(
    model_path: str,
    candidate: str = BASELINE_CANDIDATE,
    tp_size: int = 3,
    reserve_mib: Optional[Sequence[int]] = None,
    port: int = 8899,
    worktree: str = "",
    context_length: int = 32768,
    prefill_tokens: int = DEFAULT_PREFILL_TOKENS,
    decode_seconds: int = DEFAULT_DECODE_SECONDS,
    store_path: Optional[str] = None,
    lock_path: Optional[str] = None,
    log_dir: str = "/tmp",
    progress=None,
    server_factory=_Server,
    measure=measure_server,
    busy_fn=busy_cards,
) -> SplitProbeResult:
    """Boot one candidate, measure it, tear it down, append the row.

    ``progress(done, total, label)`` is called between steps. The GPU lock is
    taken HERE, in the process that boots the server, and released in a
    ``finally`` -- the dashboard never holds it, so a dashboard restart cannot
    strand it.
    """
    store_path = store_path or default_store_path()
    steps = ["lock", "boot", "prefill+decode", "teardown"]

    def say(i: int, label: str) -> None:
        if progress:
            progress(i, len(steps), label)

    say(0, "taking the cards")
    lock = gpu_lock(lock_path, label=f"split_probe {candidate}")
    lock.acquire()
    try:
        held = busy_fn()
        if held:
            raise GpuLockBusy(
                "cards already carry someone else's compute process: "
                + ", ".join(f"{k} (pid {v})" for k, v in sorted(held.items()))
                + ". Nothing was started."
            )
        before = held

        base_reserve = reserve_mib or DEFAULT_RESERVE_MIB
        reserve, note = reserve_for_candidate(
            candidate, tp_size, base_reserve, model_path
        )
        result = SplitProbeResult(
            candidate=candidate,
            model_path=model_path,
            tp_size=tp_size,
            provenance=MEASURED,
            reserve_mib=list(reserve),
            reserve_note=note,
            source=f"#232 {time.strftime('%Y-%m-%d %H:%M')}",
            clock_context=clock_context(),
        )

        attempt = 0
        while True:
            attempt += 1
            log_path = os.path.join(
                log_dir, f"split_probe.{candidate.replace(',', '_')}.{attempt}.log"
            )
            result.boot_log = log_path
            cmd = launch_command(
                model_path,
                candidate,
                reserve,
                port,
                tp_size=tp_size,
                context_length=context_length,
            )
            server = server_factory(cmd, launch_env(worktree), log_path, port)
            say(1, f"booting {candidate} at reserve {','.join(map(str, reserve))}")
            try:
                server.start()
                server.wait_ready()
                say(2, "cold prefill, then the decode window")
                measured = measure(
                    port,
                    prefill_tokens=prefill_tokens,
                    decode_seconds=decode_seconds,
                )
            except _BootOOM as e:
                server.stop()
                if attempt >= 2:
                    result.unbootable = (
                        f"OOM at reserve {','.join(map(str, reserve))} MiB after a "
                        f"raise. This candidate does not fit on this rig at this "
                        f"context. {str(e)[-400:]}"
                    )
                    break
                # Concentration spends the slack on the rank that gained the
                # mass, so that is the rank whose headroom is raised. The step
                # is half of what the plan already reserves there: a figure
                # already in play rather than an invented constant, and large
                # enough that the retry is one boot rather than a search.
                idx = _concentrated_rank(candidate, tp_size)
                result.reserve_retried_from = list(result.reserve_mib)
                reserve = list(reserve)
                reserve[idx] += int(round(base_reserve[idx] * RETRY_RESERVE_FRACTION))
                result.reserve_mib = list(reserve)
                result.reserve_note = (
                    f"{note}; first boot OOMed, retried with rank {idx} raised "
                    f"to {reserve[idx]} MiB"
                )
                continue
            except Exception:
                server.stop()
                raise
            else:
                for k, v in measured.items():
                    setattr(result, k, v)
                text = server.log_text()
                result.rank_compute_wait = parse_rank_compute_wait(text)
                m = _CHOSEN.search(text)
                result.chosen_vector = m.group(1) if m else candidate
                if result.max_total_num_tokens is None:
                    m = _MAXTOK.search(text)
                    if m:
                        result.max_total_num_tokens = int(m.group(1))
                say(3, "tearing the server down")
                server.stop()
                break

        orphan = _orphan_note(before, busy_fn)
        if orphan:
            result.reserve_note = (result.reserve_note + "; " + orphan).strip("; ")
            logger.warning("%s", orphan)

        SplitProbeStore().append_to_file(store_path, result)
        say(4, "done")
        return result
    finally:
        lock.release()


def _concentrated_rank(candidate: str, tp_size: int) -> int:
    vec = _vector_of(candidate, tp_size)
    if vec is None:
        return 0
    return max(range(len(vec)), key=lambda i: vec[i])


# ---------------------------------------------------------------------------
# the job


@dataclasses.dataclass
class SplitProbeJob:
    job_id: str
    candidate: str = BASELINE_CANDIDATE
    state: str = PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    step: str = ""
    error: Optional[str] = None
    remedy: Optional[str] = None
    result: Optional[SplitProbeResult] = None

    def to_json(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        return {
            "job_id": self.job_id,
            "candidate": self.candidate,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": elapsed,
            "step": self.step,
            "error": self.error,
            "remedy": self.remedy,
            "result": self.result.to_json() if self.result else None,
        }


def _run_probe_subprocess(request: dict) -> SplitProbeResult:
    """Run the probe in its own interpreter and read back the row it wrote.

    The dashboard must not own the server's lifetime: a page reload or a
    restart of the dashboard would otherwise take a booted 27B model with it,
    or leave one behind. The child writes the row into the store; the parent
    reads the store's last line for that candidate.
    """
    store_path = request.get("store_path") or default_store_path()
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.planner.split_probe",
        "--run",
        "--model-path",
        str(request["model_path"]),
        "--candidate",
        str(request.get("candidate") or BASELINE_CANDIDATE),
        "--tp-size",
        str(int(request.get("tp_size") or 3)),
        "--port",
        str(int(request.get("port") or 8899)),
        "--store",
        store_path,
    ]
    if request.get("reserve_mib"):
        cmd += ["--reserve-mib", ",".join(str(int(x)) for x in request["reserve_mib"])]
    if request.get("worktree"):
        cmd += ["--worktree", str(request["worktree"])]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "split probe failed (rc=%d): %s"
            % (proc.returncode, (proc.stderr or proc.stdout or "")[-1500:])
        )
    cand = request.get("candidate") or BASELINE_CANDIDATE
    entry = SplitProbeStore.load(store_path).latest().get(cand)
    if entry is None:
        raise RuntimeError(f"the split probe wrote no row for {cand!r} to {store_path}")
    return entry


class SplitProbeJobStore:
    """One measurement at a time; a second request joins the first.

    Two probes at once would each measure the other's interference, so the
    store is single-flight by construction rather than by a warning in the UI.
    """

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._jobs: Dict[str, SplitProbeJob] = {}
        #: Overridable for tests: run inline instead of on a thread.
        self.synchronous = False
        #: Overridable for tests: what performs the run.
        self.runner = _run_probe_subprocess

    def jobs(self) -> List[SplitProbeJob]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> Optional[SplitProbeJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Optional[SplitProbeJob]:
        for j in self.jobs():
            if j.state in (PENDING, RUNNING):
                return j
        return None

    def latest(self) -> Optional[SplitProbeJob]:
        jobs = sorted(self.jobs(), key=lambda j: j.started_at or 0)
        return jobs[-1] if jobs else None

    def start(self, request: dict) -> SplitProbeJob:
        running = self.active()
        if running is not None:
            return running
        job = SplitProbeJob(
            job_id=_uuid.uuid4().hex[:12],
            candidate=str(request.get("candidate") or BASELINE_CANDIDATE),
            state=RUNNING,
            started_at=time.time(),
            step="starting",
        )
        with self._lock:
            self._expire_locked(time.time())
            self._jobs[job.job_id] = job
        if self.synchronous:
            self._run(job, request)
        else:
            import threading

            threading.Thread(target=self._run, args=(job, request), daemon=True).start()
        return job

    def _run(self, job: SplitProbeJob, request: dict) -> None:
        try:
            job.step = "booting and measuring"
            job.result = self.runner(request)
            job.state = OK
            job.step = "done"
        except GpuLockBusy as e:
            job.state = ERROR
            job.error = str(e)
            job.remedy = (
                "Wait for the running measurement, or free the cards. Never "
                "kill a process you did not start."
            )
        except Exception as e:
            job.state = ERROR
            job.error = f"{type(e).__name__}: {e}"
            job.remedy = (
                "Check the boot log named in the error, then run the same "
                "candidate from the shell with "
                "`python -m sglang.srt.planner.split_probe --run` to see the "
                "boot output directly."
            )
        finally:
            job.finished_at = time.time()

    def _expire_locked(self, now: float) -> None:
        for jid in [
            j.job_id
            for j in self._jobs.values()
            if j.finished_at is not None and now - j.finished_at > JOB_TTL_S
        ]:
            del self._jobs[jid]


JOBS = SplitProbeJobStore()


# ---------------------------------------------------------------------------
# CLI


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sglang.srt.planner.split_probe")
    p.add_argument("--run", action="store_true", help="measure one candidate")
    p.add_argument("--model-path", default="")
    p.add_argument(
        "--candidate",
        default=BASELINE_CANDIDATE,
        help="'auto' or an MLP vector such as 6,1,1",
    )
    p.add_argument("--tp-size", type=int, default=3)
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--reserve-mib", default="")
    p.add_argument("--worktree", default="")
    p.add_argument("--store", default="")
    p.add_argument("--import-264", action="store_true", help="seed the #264 rows")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    store_path = args.store or default_store_path()

    if args.import_264:
        added = import_264_rows(store_path, model_path=args.model_path)
        print(f"imported {len(added)} row(s) into {store_path}")
        return 0

    if not args.run:
        table = tipping_point_table(store_path)
        if args.json:
            print(json.dumps(table, indent=1))
            return 0
        print(table["summary"])
        for row in table["rows"]:
            if not row.get("measured"):
                print(f"  {row['candidate']:<8} not measured")
                continue
            d = row.get("delta") or {}
            print(
                f"  {row['candidate']:<8} prefill {row['prefill_tok_s']} tok/s"
                f"  decode {row['decode_tok_s']} tok/s"
                f"  ms/verify {row['ms_per_verify']}"
                f"  maxKV {row['max_total_num_tokens']}"
                + (
                    f"  [prefill {d.get('prefill_pct')}%, decode {d.get('decode_pct')}%,"
                    f" KV {d.get('max_kv_pct')}%]"
                    if d
                    else ""
                )
            )
        return 0

    if not args.model_path:
        p.error("--run needs --model-path")
    reserve = (
        [int(x) for x in args.reserve_mib.split(",")] if args.reserve_mib else None
    )

    def progress(done, total, label):
        print(f"[{done}/{total}] {label}", flush=True)

    result = run_split_probe(
        args.model_path,
        candidate=args.candidate,
        tp_size=args.tp_size,
        reserve_mib=reserve,
        port=args.port,
        worktree=args.worktree,
        store_path=store_path,
        progress=progress,
    )
    print(json.dumps(result.to_json(), indent=1))
    return 0 if not result.unbootable else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
