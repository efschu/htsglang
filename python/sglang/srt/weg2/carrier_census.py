"""#1246: where the front's ``--carrier-max-tokens`` comes from.

THE BOUND IS READ, NOT RE-DERIVED.  Group D's KV carrier host pool prints the
prefetch budget it will actually enforce, once per rank, unconditionally, at
``init_hicache``::

    #915 PREFETCH LIMIT now=27466 (fraction=0.9 x host size 30518) role=staging
      pool_id=131401976285280 phase=pp generation=0 site=init_hicache

That line is the census's sole source.  Every term the front needs is on it and
named: the budget, the fraction it came from, the pool size, the role that chose
the fraction, the pool identity, the binding phase and the emitting site
(``prefetch_budget.log_prefetch_limit``).  ``site`` is the field that says WHEN
the line was emitted, and the census admits only the two launch-time cache-init
sites -- see :data:`CENSUS_SITES`, which names all three call sites and why the
third is excluded.

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
SOURCE_MARKER = ("#915 PREFETCH LIMIT now=<n> (fraction=<f> x host size <s>) role=<r> ... "
                 "site=<init_hicache|hiradix_init>")

#: The component the bound describes, named once so the log line cannot drift
#: from the docstring above.
COMPONENT = "group D KV carrier host pool (cache_controller.mem_pool_host via prefetch_budget.host_pool_anchor)"

#: The emitting sites the census admits: the two LAUNCH-TIME cache-init sites.
#: ``log_prefetch_limit`` has THREE call sites and they are not interchangeable
#: -- ``init_hicache`` (:file:`unified_radix_cache.py:1049`), ``hiradix_init``
#: (:file:`hiradix_cache.py:216`) and ``rebind_for_cutover``
#: (:file:`hicache_phase_binding.py:971`).  The first two are the constructors of
#: the two cache classes, exactly one of which any group D builds; the third runs
#: after every flip.  The census is taken at launch, before any traffic and
#: before any flip, so the two init sites ARE the population, and naming them
#: keeps a later rebind line from silently joining the sample (denominator law).
#: Admitting only ``init_hicache`` would turn a group D built on ``HiRadixCache``
#: into a W45 ``missing`` refusal while its carrier was perfectly healthy.
CENSUS_SITES: Tuple[str, ...] = ("init_hicache", "hiradix_init")

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
        route is expected.  NOT a bound of 0.  0 is not an off switch and never
        was: the front's two carrier guards read ``self.carrier_max_tokens > 0``
        (:file:`front.py:561` CARRIER-EXCEEDS and :file:`front.py:624` the
        post-leg-1 correction), so 0 removes the BYPASS and sends every prompt
        above the SHORT grant through the leg-1/leg-2 round trip with NO bound on
        what the store is asked to carry -- the ``#915 PREFETCH REFUSED`` / W16
        shape of boot weg2ls4b2 (84,027 tokens against a 30,518-token host pool).
        The front's own help says so in one line (:file:`front.py:1156`, "0 = no
        CARRIER-EXCEEDS route").  The launcher therefore never ships 0: it is
        below the floor and is refused like any other bound that cannot carry the
        route, whether the census produced it or the operator typed it.  A parse
        miss must be able to imitate neither (boot weg2rg5's exit 0).
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

    def terms(self) -> str:
        """The measured terms, or the words that say they were NOT measured.

        A census that found no source line has ``role="?"``, ``fraction=0.0``
        and ``host_size=0`` because a frozen dataclass needs values, and
        ``fraction=0.0`` on a log line reads as a MEASURED zero rather than as
        "there was nothing to measure".  The distinction is the whole point of
        the ``missing`` verdict, so the line that carries the fields says which
        of the two it is (indicator law: a field is a finding only once it has
        been checked that it measured anything at all).
        """
        if not self.lines:
            return "role/fraction/host_size=<not measured: no source line in this log>"
        return f"role={self.role} fraction={self.fraction} host_size={self.host_size}"


def tp_size_of(argv: Sequence[str], default: int = 0) -> int:
    """The rank count the census must see, taken from the argv the launcher
    itself built for the group -- never a constant here.

    ``--tp-size N`` and ``--tp-size=N`` are both accepted because both forms are
    legal on the command line the launcher hands to ``sglang.launch_server``.

    THE LAST OCCURRENCE WINS, because that is what the SERVER does.  The
    launcher writes ``--tp-size 3`` into group D's argv and then appends the
    operator's ``--extra-d`` tail, so ``--extra-d "--tp-size 6"`` is the rank
    count the group actually runs.  A census that took the FIRST occurrence
    would check a rank population the group does not have and refuse a healthy
    carrier as ``missing`` (or, worse, accept a partial one).
    """
    items = list(argv)
    found = default
    for i, a in enumerate(items):
        if a == "--tp-size" and i + 1 < len(items):
            try:
                found = int(items[i + 1])
            except ValueError:
                found = default
        elif a.startswith("--tp-size="):
            try:
                found = int(a.split("=", 1)[1])
            except ValueError:
                found = default
    return found


def _front_line(anchor: str, default: int) -> int:
    """The line of :file:`front.py` that currently holds ``anchor``.

    RESOLVED, NEVER TRANSCRIBED (instrument-text law, the same Klasse A this
    module exists to close).  A hand-typed ``front.py:561`` in a provenance
    string is an instrument whose text stops describing the code the first time
    a line is inserted above it, and no test can see the drift because the test
    can only pin the string.  Resolving the citation at the moment it is printed
    means the number is either right or the anchor is gone -- and the anchors
    are the branch texts themselves, so an anchor that disappears means the
    branch it names disappeared, which is a change this module must not survive
    quietly.  ``default`` is what the line was when this was written; it is used
    only if the anchor cannot be found at all.
    """
    try:
        import inspect

        from sglang.srt.weg2 import front

        lines, _ = inspect.getsourcelines(front)
        for n, line in enumerate(lines, start=1):
            if anchor in line:
                return n
    except Exception:  # noqa: BLE001 - a citation may never break a launch
        pass
    return default


def route_floor() -> Tuple[int, str]:
    """The largest carrier bound at which the round trip is still unreachable.

    DERIVED FROM WHAT THE ROUTE NEEDS, and derived IN THE UNIT THE BOUND IS
    COMPARED IN.  The front tiles the prompt-length axis with two branches that
    both bypass the carrier, and they do NOT price the same prompt in the same
    unit:

    * SHORT (:file:`front.py:578`) compares ``remainder``, which
      :func:`front.price_remainder` computes with ``CHARS_PER_TOKEN`` (3.0);
    * CARRIER-EXCEEDS (:file:`front.py:561`) compares ``carrier_est``, computed
      with ``CARRIER_CHARS_PER_TOKEN`` (2.4) -- deliberately a lower divisor, so
      that route cannot UNDER-estimate (the constant's own comment, #1233).

    ``carrier_max_tokens`` is compared against ``carrier_est``, so the floor has
    to be expressed in ``carrier_est`` tokens.  ``front.CHUNK_TOKENS`` is not:
    it is a ``remainder`` number, 1.25x smaller, and taking it as the floor
    passed every bound in (4096, 5120] as usable when in fact NO prompt of ANY
    length can round-trip under one -- the same silent capability loss as boot
    weg2rg5's 17, one order of magnitude up.

    Both prices are non-decreasing in the prompt's character length, so the set
    of prompts SHORT will not serve is a suffix ``[L*, inf)`` of that axis, and
    the cheapest carrier estimate over that suffix is the one at ``L*``.  A
    bound ``B`` therefore admits a round trip for at least one prompt length iff
    ``B >= carrier_est(L*)``, and the floor -- the largest ``B`` that admits
    none -- is ``carrier_est(L*) - 1``.  ``L*`` is found by bisection over
    :func:`front.price_remainder` itself, with an EMPTY span store: a cached
    prefix only lowers ``remainder`` and so only makes the round trip harder to
    reach, which makes the empty store the most permissive cache state and the
    honest one to derive a floor from.

    SCOPE (the floor is a conservative refusal, never a licence).  SHORT is four
    conjuncts, not one: ``awake == "D" and admit_d and state == "serving" and
    remainder <= CHUNK_TOKENS``.  While group D is asleep a sub-chunk prompt
    falls through to BATCH and DOES take the round trip, so at a bound below
    this floor the round-trip set is empty only in D's serving steady state --
    which is the state a boot spends its time in and the state group P's prefill
    work has to come from.  The direction of the approximation is therefore a
    false REFUSAL and never a bound that is shipped when it should not be.

    Every number is READ from the module that owns it, the same rule
    ``launcher._front_leg_form`` follows; a copy here would be a second
    bookkeeping of the front's own constants and would go stale the moment the
    chunk grant or either divisor changes.

    Returns ``(floor, provenance)``; the provenance string goes on the log line
    so the number is never printed without its derivation.
    """
    from sglang.srt.weg2 import front

    chunk = int(front.CHUNK_TOKENS)
    spans = front.SpanLRU()

    def _not_short(n_chars: int) -> bool:
        remainder, _est, _known = front.price_remainder("x" * n_chars, spans)
        return remainder > chunk

    # Bisect for L*, the shortest prompt SHORT will not serve.  The seed only
    # has to bracket it; price_remainder is the authority on where it is.
    hi = 16
    while not _not_short(hi):
        hi *= 2
        if hi > (chunk + 4) * 64:  # unreachable for any sane divisor; fail loud
            raise RuntimeError(
                "#1246 route_floor: no prompt length up to %d chars exceeds "
                "front.CHUNK_TOKENS=%d under front.price_remainder" % (hi, chunk)
            )
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if _not_short(mid):
            hi = mid
        else:
            lo = mid + 1
    l_star = lo

    # The CARRIER-EXCEEDS price of that same prompt, by the front's own
    # expression at front.py:560 (`int(len(text) / CARRIER_CHARS_PER_TOKEN) + 1`
    # for a prompt whose exact token count is not yet known -- and at launch
    # time none is).
    carrier_est_at_l_star = int(l_star / front.CARRIER_CHARS_PER_TOKEN) + 1
    floor = carrier_est_at_l_star - 1

    ln_chunk = _front_line("CHUNK_TOKENS = ", 67)
    ln_carrier = _front_line("and carrier_est > self.carrier_max_tokens:", 561)
    ln_short = _front_line("and remainder <= CHUNK_TOKENS:", 578)
    ln_est = _front_line("carrier_est = exact if exact else", 560)
    why = (
        f"floor={floor} tokens of CARRIER-EXCEEDS price, NOT front.CHUNK_TOKENS={chunk}: the two "
        f"bypass branches price the same prompt in different units -- SHORT (front.py:{ln_short}) "
        f"compares front.price_remainder's estimate at CHARS_PER_TOKEN={front.CHARS_PER_TOKEN}, "
        f"CARRIER-EXCEEDS (front.py:{ln_carrier}) compares carrier_est at "
        f"CARRIER_CHARS_PER_TOKEN={front.CARRIER_CHARS_PER_TOKEN} (front.py:{ln_est}), and "
        f"carrier_max_tokens is compared against the latter. The shortest prompt SHORT will not "
        f"serve is {l_star} chars (front.CHUNK_TOKENS={chunk}, front.py:{ln_chunk}, bisected over "
        f"front.price_remainder with an empty span store = the most permissive cache state), which "
        f"CARRIER-EXCEEDS prices at {carrier_est_at_l_star} tokens; so every bound <= {floor} "
        f"bypasses the leg-1/leg-2 round trip for EVERY prompt length. Scope: SHORT is four "
        f"conjuncts (awake=D, admit_d, state=serving, remainder<=CHUNK_TOKENS), so while D sleeps a "
        f"sub-chunk prompt does queue to BATCH -- this floor is a conservative refusal in D's "
        f"serving steady state, never a licence"
    )
    return floor, why


def parse_limit_lines(
    log_path: str, *, sites: Sequence[str] = CENSUS_SITES
) -> List[Tuple[int, Dict[str, str], str]]:
    """Every ``#915 PREFETCH LIMIT`` line of ``log_path`` emitted at ``sites``.

    Returns ``(rank, fields, raw_line)`` in file order.  A missing file yields an
    empty list -- the caller turns that into the ``missing`` verdict, never into
    a bound.
    """
    admitted = tuple(sites or ())
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
                if admitted and d["site"] not in admitted:
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
    sites: Sequence[str] = CENSUS_SITES,
) -> CarrierCensus:
    """Take the carrier census over one group log.  Pure; never raises."""
    rows = parse_limit_lines(log_path, sites=sites)
    site_names = ",".join(sites) or "<any>"

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
            site=site_names,
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
            f"no '#915 PREFETCH LIMIT ... site in ({site_names})' line in {log_path}: the KV carrier host pool "
            f"never reported the budget it enforces, so this boot's carrier bound was NOT measured "
            f"(that is not the same fact as a route that was switched off)",
        )

    first = rank_rows[sorted(rank_rows)[0]][0]

    missing_ranks = [r for r in range(expected_ranks) if r not in per_rank]
    if missing_ranks:
        return _mk(
            "missing",
            f"only ranks {sorted(per_rank)} reported at site in ({site_names}), expected {expected_ranks} "
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
