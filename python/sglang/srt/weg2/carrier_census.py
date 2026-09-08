"""#1246: where the front's ``--carrier-max-tokens`` comes from.

THE BOUND IS READ, NOT RE-DERIVED.  Group D's KV carrier host pool prints the
prefetch budget it will actually enforce, once per rank, unconditionally, at
``init_hicache``::

    #915 PREFETCH LIMIT now=27466 (fraction=0.9 x host size 30518) role=staging
      pool_id=131401976285280 phase=pp generation=0 site=init_hicache

That line is the census's sole source.  Every term the front needs is on it and
named: the budget, the fraction it came from, the pool size, the role that chose
the fraction, the pool identity, the binding phase and the emitting site
(``prefetch_budget.log_prefetch_limit``, :file:`prefetch_budget.py:71-99`).

WHY IT NAMES THE KV CARRIER AND NOTHING ELSE.  The line is formed from
``cache_controller.mem_pool_host`` through ``prefetch_budget.host_pool_anchor``
(:file:`prefetch_budget.py:52-66`), which is documented and implemented as *the
KV host pool a controller is bound to*, a pool GROUP unwrapped to its anchor
entry.  The mamba/GDN state pool has no cache controller and emits no such line.
Measured on two boots of the same tip family: exactly THREE lines, one per TP
rank, in ``boot_weg2_weg2rg3_5b015ad139_0908_041053.D.log`` and in
``boot_weg2_weg2rg5_15a46a611a_0908_050519.D.log``.

WHAT THIS REPLACED, and why the old source could not be repaired in place
(boot weg2rg5, :file:`/spinning/gpu-arb/weg2/BOOT_weg2rg5_0908.md`, "THE
FINDING").  The launcher used to scan D's log for the upstream WARNING

    HiCache host KV pool (N tokens) is smaller than the device pool (M tokens)

and take ``int(0.9 * min(N))``.  Three independent defects, and the source
change closes all three at once:

1. **The population was never named.**  That warning is emitted by every host
   pool that is smaller than its device pool -- including the mamba/GDN state
   pool, which printed the words "KV pool" about 19 mamba slots
   (:file:`memory_pool_host.py` ``MambaPoolHost``) and about any state pool
   reaching the shared base emitter (:file:`pool_host/base.py`).  On rg5 the
   M=600 ledger arm left the mamba host pool (19 slots) below its device pool
   (20), so ``min`` was 19 and the bound was **17**: the front took
   CARRIER-EXCEEDS on *every* request, group P executed zero prefill passes, and
   the boot could not answer the question it was booted for -- at exit 0,
   because only a bound of ZERO counted as "route disabled".  rg3, one arm up at
   M=1200, saw six 30518-token lines and no mamba line at all, and got 27466.
   The whole difference was 0.80 GiB of host RAM.  (The two emitters are made
   honest in the same commit -- instrument-text law, Klasse A -- but the census
   no longer depends on either of them.)
2. **``0.9 * size`` was a second bookkeeping** of
   ``HiCacheController.prefetch_capacity_fraction``.  The front's own docstring
   already cites the runtime limit as the thing this bound exists to stay under
   (``#915 PREFETCH REFUSED``, :file:`front.py:432`).  Reading the enforced
   number removes the copy instead of repairing it.
3. **The warning is CONDITIONAL** -- it fires only when the host pool is at or
   below the device pool.  A KV host pool larger than its device pool emits
   nothing at all, and the resulting parse miss read as "route disabled"
   rather than "not measured".

This module is a pure function over a log path: it computes a verdict and never
raises and never launches anything.  The refusal (W45
``Weg2CarrierCensusRefused``) is the launcher's, raised as ``Weg2LaunchRefused``
in the same shape as W7/W9/W10, so the existing refusal handling covers it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: What a reader should grep for in group D's log to see the census's input.
SOURCE_MARKER = "#915 PREFETCH LIMIT now=<n> (fraction=<f> x host size <s>) role=<r> ... site=init_hicache"

#: The component the bound describes, named once so the log line cannot drift
#: from the docstring above.
COMPONENT = "group D KV carrier host pool (cache_controller.mem_pool_host via prefetch_budget.host_pool_anchor)"

#: The emitting site the census admits.  ``log_prefetch_limit`` also runs after
#: every cutover rebind; the census is taken at launch, before any traffic and
#: before any flip, so ``init_hicache`` IS the population and saying so keeps a
#: later rebind line from silently joining the sample (denominator law).
CENSUS_SITE = "init_hicache"

#: TRAP-SAFE by shape, not by token (#995).  ``#915 PREFETCH LIMIT`` also occurs
#: in the fork's own PROSE -- :file:`scheduler_pp_mixin.py:2932` tells the
#: reader to "read the #915 PREFETCH LIMIT line".  Anchoring on the full field
#: shape (``now=`` immediately followed by the parenthesised fraction term)
#: cannot match that sentence.  The rank comes from the log prefix that the
#: server writes on every line: ``[<ts> TP<k>]``.
LIMIT_RE = re.compile(
    r"\bTP(?P<rank>\d+)\]\s*"
    r"#915 PREFETCH LIMIT now=(?P<now>\d+) "
    r"\(fraction=(?P<fraction>[0-9.]+) x host size (?P<host_size>\d+)\) "
    r"role=(?P<role>\S+) pool_id=(?P<pool_id>\d+) "
    r"phase=(?P<phase>\S+) generation=(?P<generation>\d+) site=(?P<site>\S+)"
)


@dataclass(frozen=True)
class CarrierCensus:
    """One census over one group log.  ``verdict`` is the whole answer.

    ``verdict`` is one of:

    ``ok``
        every expected rank reported, they agree, and the agreed bound clears
        the floor.  ``bound`` is what the front gets.
    ``missing``
        the KV carrier's line is absent for at least one expected rank while the
        route is expected.  NOT a bound of 0 -- a bound of 0 means the launcher
        deliberately switched the route off, and a parse miss must never be able
        to imitate that (boot weg2rg5's exit 0).
    ``disagree``
        the ranks reported different budgets, fractions, host sizes or roles.
        The front routes by ONE number for a group of three lockstep ranks; if
        they disagree, no single number is that group's bound.
    ``below_floor``
        the ranks agree and the agreed bound cannot carry the route.
    """

    bound: int
    floor: int
    per_rank: Dict[int, int]
    role: str
    fraction: float
    host_size: int
    site: str
    expected_ranks: int
    lines: Tuple[str, ...]
    verdict: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"


def tp_size_of(argv: Sequence[str], default: int = 0) -> int:
    """The rank count the census must see, taken from the argv the launcher
    itself built for the group -- never a constant here.

    ``--tp-size N`` and ``--tp-size=N`` are both accepted because both forms are
    legal on the command line the launcher hands to ``sglang.launch_server``.
    """
    items = list(argv)
    for i, a in enumerate(items):
        if a == "--tp-size" and i + 1 < len(items):
            try:
                return int(items[i + 1])
            except ValueError:
                return default
        if a.startswith("--tp-size="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return default
    return default


def route_floor() -> Tuple[int, str]:
    """The smallest carrier bound at which the round trip can still happen.

    DERIVED FROM WHAT THE ROUTE NEEDS, not chosen.  The front tiles the
    prompt-length axis with two branches that both bypass the carrier:

    * CARRIER-EXCEEDS (:file:`front.py:561`) sends every prompt estimated above
      ``carrier_max_tokens`` to ONE prefill on D -- no leg 1, no store read;
    * SHORT (:file:`front.py:279`, :file:`front.py:578`) serves every prompt
      whose uncached remainder is ``<= front.CHUNK_TOKENS`` straight on D --
      again no leg 1 and therefore no store read.

    A prompt can only take the round trip if it is above ``CHUNK_TOKENS`` AND at
    or below ``carrier_max_tokens``.  That interval is empty unless
    ``carrier_max_tokens > CHUNK_TOKENS``.  So the floor is exactly
    ``front.CHUNK_TOKENS`` and the test is strict: a bound at or below it is an
    off switch wearing a number, which is what a bound of 17 was on boot
    weg2rg5.

    The value is READ from the module that owns it, the same rule
    ``launcher._front_leg_form`` follows -- a copy here would be a second
    bookkeeping of the front's own constant and would go stale the moment the
    chunk grant changes.

    Returns ``(floor, provenance)``; the provenance string goes on the log line
    so the number is never printed without its derivation.
    """
    from sglang.srt.weg2 import front

    floor = int(front.CHUNK_TOKENS)
    why = (
        f"front.CHUNK_TOKENS={floor} (front.py:67) -- the round-trip interval is "
        f"CHUNK_TOKENS < prompt <= carrier_max, empty unless carrier_max > {floor}: "
        f"SHORT serves at or below it on D (front.py:279/:578) and CARRIER-EXCEEDS "
        f"serves above carrier_max on D (front.py:561)"
    )
    return floor, why


def parse_limit_lines(
    log_path: str, *, site: str = CENSUS_SITE
) -> List[Tuple[int, Dict[str, str], str]]:
    """Every ``#915 PREFETCH LIMIT`` line of ``log_path`` emitted at ``site``.

    Returns ``(rank, fields, raw_line)`` in file order.  A missing file yields an
    empty list -- the caller turns that into the ``missing`` verdict, never into
    a bound.
    """
    out: List[Tuple[int, Dict[str, str], str]] = []
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if "#915 PREFETCH LIMIT" not in line:
                    continue
                m = LIMIT_RE.search(line)
                if not m:
                    continue
                d = m.groupdict()
                if site and d["site"] != site:
                    continue
                out.append((int(d["rank"]), d, line.rstrip("\n")))
    except OSError:
        return []
    return out


def census(
    log_path: str,
    *,
    expected_ranks: int,
    floor: int,
    site: str = CENSUS_SITE,
) -> CarrierCensus:
    """Take the carrier census over one group log.  Pure; never raises."""
    rows = parse_limit_lines(log_path, site=site)

    per_rank: Dict[int, int] = {}
    rank_rows: Dict[int, List[Dict[str, str]]] = {}
    lines: List[str] = []
    for rank, d, raw in rows:
        rank_rows.setdefault(rank, []).append(d)
        per_rank.setdefault(rank, int(d["now"]))
        lines.append(raw)

    def _mk(verdict: str, detail: str, bound: int = 0, first: Optional[Dict[str, str]] = None) -> CarrierCensus:
        return CarrierCensus(
            bound=bound,
            floor=int(floor),
            per_rank=dict(sorted(per_rank.items())),
            role=(first or {}).get("role", "?"),
            fraction=float((first or {}).get("fraction", 0.0) or 0.0),
            host_size=int((first or {}).get("host_size", 0) or 0),
            site=site,
            expected_ranks=int(expected_ranks),
            lines=tuple(lines),
            verdict=verdict,
            detail=detail,
        )

    if expected_ranks <= 0:
        return _mk(
            "missing",
            f"the group's argv named no --tp-size, so the census has no rank population to check "
            f"(expected_ranks={expected_ranks})",
        )

    if not rows:
        return _mk(
            "missing",
            f"no '#915 PREFETCH LIMIT ... site={site}' line in {log_path}: the KV carrier host pool "
            f"never reported the budget it enforces, so this boot's carrier bound was NOT measured "
            f"(that is not the same fact as a route that was switched off)",
        )

    first = rank_rows[sorted(rank_rows)[0]][0]

    missing_ranks = [r for r in range(expected_ranks) if r not in per_rank]
    if missing_ranks:
        return _mk(
            "missing",
            f"only ranks {sorted(per_rank)} reported at site={site}, expected {expected_ranks} "
            f"(missing TP {missing_ranks}); a bound taken from a partial rank population is not "
            f"the group's bound",
            first=first,
        )

    extra_ranks = [r for r in sorted(per_rank) if r >= expected_ranks]
    if extra_ranks:
        return _mk(
            "disagree",
            f"ranks {extra_ranks} reported beyond the expected TP population of {expected_ranks}: "
            f"the log holds more carrier pools than the group has ranks",
            first=first,
        )

    # Agreement over every term the bound is made of, not just the bound: two
    # ranks can print the same budget from different pools or fractions, and
    # that is a divergence the front's single number would hide.
    for axis in ("now", "fraction", "host_size", "role"):
        seen = {}
        for rank in sorted(rank_rows):
            for d in rank_rows[rank]:
                seen.setdefault(d[axis], []).append(rank)
        if len(seen) > 1:
            return _mk(
                "disagree",
                f"the {expected_ranks} TP ranks do not agree on '{axis}': "
                + "; ".join(f"{v!r} on TP {sorted(set(rs))}" for v, rs in sorted(seen.items()))
                + " -- the front routes the whole group by ONE bound, so a divergent group has none",
                first=first,
            )

    bound = int(first["now"])
    if bound <= int(floor):
        return _mk(
            "below_floor",
            f"the agreed bound {bound} is at or below the floor {floor}: no prompt can be both above "
            f"the front's SHORT grant and within this bound, so the carrier round trip is unreachable "
            f"for every prompt length (boot weg2rg5 measured exactly this at bound 17: zero round "
            f"trips, zero prefill passes on group P, exit 0)",
            bound=bound,
            first=first,
        )

    return _mk(
        "ok",
        f"{expected_ranks} TP ranks agree on {bound} tokens (fraction {first['fraction']} x host size "
        f"{first['host_size']}, role {first['role']}), which clears the floor {floor}",
        bound=bound,
        first=first,
    )
