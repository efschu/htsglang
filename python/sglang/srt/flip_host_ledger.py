# SPDX-License-Identifier: Apache-2.0
"""``[flip-host-ledger]`` -- what THIS process really holds of the host mark.

WHY A SECOND HOST NUMBER. ``[offload-kv-regain]``
(``model_runner_kv_cache_mixin.py:1125-1138``) already prints the page-locked
expert pool per rank -- 22.73/5.30/3.85 GiB on fn7s, 20.62/7.36/10.88 on
fnFA19. What it does NOT print is the other half of the mark: the anonymous
runtime footprint. The design's §5.2 had to ESTIMATE that half (~27.5 GiB per
three-process group, from ``cgroup_current`` minus pinned) and its risk R1
says so in as many words: *"the non-pinned host post is only ESTIMATED ... if
cgroup_current contains page cache, the six-process form is far more relaxed
than §5.2 says"*. This ledger is the measurement that retires R1.

THE DOUBLE-COUNT THIS MODULE EXISTS TO AVOID. Without the shared cold tier,
``pinned_exact_empty`` maps ``MAP_PRIVATE|MAP_ANONYMOUS``
(``expert_offload.py:2581,2614``) -- so the pinned expert pool IS anonymous
memory and appears in ``RssAnon`` in full. Adding ``pinned`` to ``RssAnon``
would count 22.73 GiB twice on stage 0 alone. With the shared cold tier the
pool is a ``MAP_SHARED`` tmpfs mapping (``shared_pinned.py:35-39``) and
appears in ``RssShmem`` instead, where it must be counted ONCE for the whole
rig rather than once per reader. The two cases are therefore NOT a detail of
presentation; they are two different arithmetics, and
:func:`host_post_from_rss` is where the difference lives.

Three fields per process, never a single total:

    ``[flip-host-ledger] layout=<P|D> rank=<n> pid=<n> pinned=<GiB>
      anon=<GiB> shm=<GiB> shared=<true|false> mark=<GiB>``

The line is designed to be grep-able out of a boot log and fed back through
:func:`parse_host_ledger_line`, so the six lines of a flip boot aggregate into
one :func:`sglang.srt.flip_nextflash_plan.solve_host_pool` call and one W114
verdict -- instead of an agent adding six numbers by hand from a log.

Memory ``HOST-SCHWELLE-WEICH``: the 88 GiB mark is SOFT. Exceeding it is a
refusal that must be deviated from with a number and a runtime latch, never
silently. W114 is that latch.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.flip_nextflash_plan import (
    HOST_MARK_GIB,
    HostPoolLedger,
    HostPoolPost,
    Weg2FlipHostPoolDoubled,
    solve_host_pool,
)

__all__ = [
    "LEDGER_PREFIX",
    "ProcessHostPost",
    "host_post_from_rss",
    "read_rss_fields",
    "format_host_ledger_line",
    "parse_host_ledger_line",
    "ledger_from_lines",
    "emit_host_ledger_line",
]

LEDGER_PREFIX = "[flip-host-ledger]"

_GIB = 1024.0**3


@dataclass(frozen=True)
class ProcessHostPost:
    """One process's host footprint, decomposed rather than summed."""

    layout: str
    rank: int
    pid: int
    pinned_bytes: int
    anon_bytes: int
    shm_bytes: int
    shared: bool
    mark_gib: float = HOST_MARK_GIB

    @property
    def pinned_gib(self) -> float:
        return self.pinned_bytes / _GIB

    @property
    def anon_gib(self) -> float:
        return self.anon_bytes / _GIB

    @property
    def shm_gib(self) -> float:
        return self.shm_bytes / _GIB

    def as_pool_post(self) -> HostPoolPost:
        """The shape :func:`solve_host_pool` consumes.

        ``shm_bytes`` is deliberately NOT folded in here: a shared segment is
        the same bytes in every reader, so summing it per process is the very
        double-count this design exists to remove. It is carried separately
        and added once by :func:`ledger_from_lines`.
        """
        return HostPoolPost(
            layout=self.layout,
            rank=self.rank,
            pinned_gib=self.pinned_gib,
            anon_gib=self.anon_gib,
        )


def read_rss_fields(path: str = "/proc/self/status") -> dict:
    """``RssAnon`` / ``RssShmem`` / ``RssFile`` / ``VmLck`` in BYTES.

    Read from ``/proc/<pid>/status`` because it separates the three residency
    kinds, which ``cgroup_current`` does not -- and that separation is exactly
    what risk R1 of the design asks for. Missing fields come back absent
    rather than zero: a kernel that does not report ``RssAnon`` must not read
    as a process with no anonymous memory.
    """
    out: dict = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                key = key.strip()
                if key not in ("RssAnon", "RssShmem", "RssFile", "VmLck", "VmRSS"):
                    continue
                parts = rest.split()
                if not parts:
                    continue
                try:
                    kib = int(parts[0])
                except ValueError:
                    continue
                out[key] = kib * 1024
    except OSError:
        return {}
    return out


def host_post_from_rss(
    layout: str,
    rank: int,
    pinned_bytes: int,
    shared: bool,
    rss: Mapping[str, int],
    pid: Optional[int] = None,
    mark_gib: float = HOST_MARK_GIB,
) -> ProcessHostPost:
    """Decompose one process's residency WITHOUT double-counting the pool.

    PRIVATE pool (``shared=False``): ``pinned_exact_empty`` maps
    ``MAP_PRIVATE|MAP_ANONYMOUS``, so the pool is already inside ``RssAnon``.
    The anonymous post is therefore ``RssAnon - pinned``, and the shm post is
    whatever shared memory the process holds for other reasons.

    SHARED pool (``shared=True``): the pool is a ``MAP_SHARED`` tmpfs mapping
    and lands in ``RssShmem``. ``RssAnon`` is then the honest anonymous post
    on its own, and the pool is reported as ``shm`` so the aggregator can
    count it ONCE across readers instead of once per reader.

    A negative anonymous post is clamped to zero rather than reported: it
    means the pinned figure and the RSS reading came from different moments,
    not that the process has negative memory. The clamp is visible because
    the pinned figure is printed beside it.
    """
    anon = int(rss.get("RssAnon", 0))
    shm = int(rss.get("RssShmem", 0))
    pinned = int(pinned_bytes)
    if shared:
        pool_shm = min(pinned, shm) if shm else pinned
        return ProcessHostPost(
            layout=layout,
            rank=int(rank),
            pid=int(os.getpid() if pid is None else pid),
            pinned_bytes=pinned,
            anon_bytes=max(0, anon),
            shm_bytes=pool_shm,
            shared=True,
            mark_gib=mark_gib,
        )
    return ProcessHostPost(
        layout=layout,
        rank=int(rank),
        pid=int(os.getpid() if pid is None else pid),
        pinned_bytes=pinned,
        anon_bytes=max(0, anon - pinned),
        shm_bytes=max(0, shm),
        shared=False,
        mark_gib=mark_gib,
    )


def format_host_ledger_line(post: ProcessHostPost) -> str:
    """The one line a boot log carries per process."""
    return (
        f"{LEDGER_PREFIX} layout={post.layout} rank={post.rank} pid={post.pid} "
        f"pinned={post.pinned_gib:.2f} anon={post.anon_gib:.2f} "
        f"shm={post.shm_gib:.2f} shared={'true' if post.shared else 'false'} "
        f"mark={post.mark_gib:.1f}"
    )


_FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_host_ledger_line(line: str) -> ProcessHostPost:
    """The inverse of :func:`format_host_ledger_line`.

    Tolerates a log prefix (timestamp, level, rank banner) before the marker,
    because that is how the line arrives in a real boot log. Refuses a line
    that lacks a field rather than defaulting it: a ledger with a missing
    post is not a ledger with a zero post.
    """
    idx = line.find(LEDGER_PREFIX)
    if idx < 0:
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- not a ledger line (no "
            f"{LEDGER_PREFIX!r}): {line.strip()[:120]!r}"
        )
    fields = dict(_FIELD_RE.findall(line[idx + len(LEDGER_PREFIX) :]))
    required = ("layout", "rank", "pid", "pinned", "anon", "shm", "shared", "mark")
    missing = [k for k in required if k not in fields]
    if missing:
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- ledger line is missing {missing}; "
            f"a missing post is not a zero post: {line.strip()[:160]!r}"
        )
    return ProcessHostPost(
        layout=fields["layout"],
        rank=int(fields["rank"]),
        pid=int(fields["pid"]),
        pinned_bytes=int(round(float(fields["pinned"]) * _GIB)),
        anon_bytes=int(round(float(fields["anon"]) * _GIB)),
        shm_bytes=int(round(float(fields["shm"]) * _GIB)),
        shared=fields["shared"].lower() == "true",
        mark_gib=float(fields["mark"]),
    )


def ledger_from_lines(
    lines: Iterable[str],
    mark_gib: float = HOST_MARK_GIB,
) -> HostPoolLedger:
    """Aggregate the ledger lines of a whole boot into ONE W114 verdict.

    The shared segment is added ONCE -- as the maximum ``shm`` reported for a
    given rank slot across all readers, not the sum over readers. Summing it
    would reintroduce the exact double-count the shared pool removes, and the
    number would look like a failure of sharing rather than of counting.

    A boot whose lines disagree about ``shared`` is refused: half a rig on the
    shared pool and half on private copies is not a state whose total means
    anything.
    """
    posts = [parse_host_ledger_line(ln) for ln in lines if LEDGER_PREFIX in ln]
    if not posts:
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- no {LEDGER_PREFIX} lines found; "
            f"an unmeasured host is not a host under the mark"
        )
    modes = {p.shared for p in posts}
    if len(modes) > 1:
        shared_pids = sorted(p.pid for p in posts if p.shared)
        private_pids = sorted(p.pid for p in posts if not p.shared)
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- the boot is HALF shared: pids "
            f"{shared_pids} read the shared cold tier, pids {private_pids} hold "
            f"private copies. Either every process attaches to the segment or "
            f"none does; a mixed rig double-counts exactly the processes that "
            f"did not attach, and its total is not interpretable. Check that "
            f"SGLANG_MOE_COLD_TIER_SHM and SGLANG_MOE_COLD_TIER_INSTANCE reach "
            f"BOTH groups (flip_cold_tier_share.build_cold_tier_group_env)."
        )

    shared = posts[0].shared
    ledger = solve_host_pool(
        [p.as_pool_post() for p in posts], shared=shared, mark_gib=mark_gib
    )
    if not shared:
        return ledger

    # The shared segment, once per rank slot.
    per_rank: dict = {}
    for p in posts:
        per_rank[p.rank] = max(per_rank.get(p.rank, 0.0), p.shm_gib)
    shm_once = sum(per_rank.values())
    total = ledger.pinned_gib + ledger.anon_gib + shm_once
    if total > mark_gib:
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- {len(posts)} live process(es) hold "
            f"{ledger.pinned_gib:.2f} GiB page-locked + {ledger.anon_gib:.2f} GiB "
            f"anonymous + {shm_once:.2f} GiB shared segment (counted ONCE per "
            f"rank slot) = {total:.2f} GiB against a host mark of {mark_gib:.1f} "
            f"GiB (over by {total - mark_gib:.2f} GiB). The pool is ALREADY "
            f"shared, so the mark cannot be met by sharing more -- the remaining "
            f"levers are fewer resident-equivalent experts per layout or a "
            f"smaller anonymous footprint (design §5.2, lever (a): one process "
            f"per rank slot, slice 7)."
        )
    return ledger


def emit_host_ledger_line(
    logger,
    layout: str,
    rank: int,
    pinned_bytes: int,
    shared: bool,
    mark_gib: float = HOST_MARK_GIB,
    rss_path: str = "/proc/self/status",
) -> Optional[ProcessHostPost]:
    """Log this process's ledger line. Never raises -- an instrument that
    kills the boot it measures is not an instrument.

    The W114 verdict is deliberately NOT taken here: one process cannot see
    the other five, and a per-process refusal would fire on the first rank of
    a form that fits. The verdict belongs to whoever reads the six lines back
    (:func:`ledger_from_lines`), which is the launcher or the desk.
    """
    try:
        rss = read_rss_fields(rss_path)
        post = host_post_from_rss(
            layout=layout,
            rank=rank,
            pinned_bytes=pinned_bytes,
            shared=shared,
            rss=rss,
            mark_gib=mark_gib,
        )
        logger.info("%s", format_host_ledger_line(post))
        return post
    except Exception:  # noqa: BLE001 -- see docstring
        return None
