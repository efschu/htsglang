# SPDX-License-Identifier: Apache-2.0
"""WIRING for the D-vector menu: the boot declaration, the backlog sensor, the
one-verdict carrier and the install seam.

``planner/d_vector_menu.py`` is the pure arithmetic -- a function from
(backlog, census, calibration) to a verdict. It installs nothing and it reads
nothing. THIS module is the half that touches the running system, and it is
kept separate for one reason: the pure half can be pinned on a desk without a
GPU, and the impure half is then a thin, enumerable set of seams rather than
arithmetic hidden inside a scheduler.

THE FOUR SEAMS, each named at its call site:

  1. DECLARATION (boot). :func:`declare_menu_from_env` builds the CEILING SET
     once, before the pools exist, from a file the operator supplies. The
     discipline is ``KvReshardRuntime.allowed_vectors``
     (``managers/kv_reshard.py:517``): a vector that was not declared before
     the pools were built may never be installed, because the pools were
     pre-sized for the declared set and an undeclared vector points the
     weighted owner rule at a split no allocation backs. A boot that declares
     nothing gets NO menu and the pre-order behaviour bit-for-bit.

  2. SENSOR (every round). :func:`flip_backlog_from_scheduler` reads the
     backlog with DEPTHS, not only a count. The flip's own arming condition
     (``phase_flip_runtime._arming_condition_persists``) is documented as
     "Deliberately coarse: any queued request is enough", and that coarseness
     is exactly what makes a count-only signal unable to tell one 200k-token
     request from two hundred 1k ones -- the two ends of the menu.

  3. CARRIER (the flip). PP0 decides, and the choice rides the EXISTING
     one-decider channel (``PhaseFlipDecision``). No new collective: a new
     collective on the flip path is recorded fatal in the #969 design.
     :meth:`DVectorMenuRuntime.adopt` is what every rank below PP0 runs, and
     it is where the two failure classes are separated:

       * a rank whose MENU differs from PP0's -> CRASH/STOP. Two ranks with
         different ceiling sets have different pools, and the only honest
         response is to stop (``raenge-nie-uneins-crash-stop``).
       * a name PP0 chose that this rank never declared -> REFUSAL BY NAME,
         quoting the name and the declared set, in the shape
         ``KvReshardRuntime.arm`` already uses for the same mistake.

  4. INSTALL (the cutover). :func:`effective_token_vector` is read at
     ``phase_flip_runtime.py`` step 2, where ``set_cp_token_ratios`` +
     ``refresh_all_owner_bounds`` run. That call site carried the comment "the
     vector is boot-constant"; with a menu declared it no longer is, and the
     comment now names what made it change.

WHAT IS DELIBERATELY *NOT* CHANGED HERE
---------------------------------------
``PhaseFlipDecision.vector`` keeps its meaning exactly: it is the PREVIOUS
effective vector, checked for identity by every follower, and a mismatch stays
group-fatal. The menu choice travels in NEW fields beside it. This matters:
the identity check is the backstop that catches ranks drifting apart, and
re-purposing it into "the vector I am about to install" would have deleted the
backstop in the same commit that gave the vector a reason to move.

THE GRAPH GATE, WHICH IS THE WHOLE PRICE OF THE MENU
----------------------------------------------------
A pure TOKEN-KEY change is NOT CUDA-graph-neutral on this tree, and this was
measured out of the code rather than assumed either way (2026-09-20). The
weighted owner rule's READ side is graph-safe -- ``kv_indices`` are rebuilt
host-side before every replay, which is what ``kv_reshard.py:46-50`` correctly
claims. Its WRITE side is not: ``dcp_weighted_write_slots``
(``layers/dcp/owner.py:429-436``) takes ``cp_S``/``cp_lo``/``cp_hi``/
``cp_ratio`` as PYTHON-INT SCALAR OPERANDS, and the call sites
(``flashinfer_backend.py:2594``, ``triton_backend.py:2392``,
``qwen_sparse_attn_backend.py:457``) are inside ``forward_decode``, which is
inside the captured body (``decode_cuda_graph_runner.py:1893`` ->
``runner_backend/full_cuda_graph_backend.py:163-170``). A scalar read during
capture is baked by value; ``ShapeKey`` (``shape_key.py:22-35``) has no layout
axis, so the stale graph is silently REUSED. The consequence is the exact
failure ``owner.py:423-427`` exists to prevent: the replay WRITES token L to
the old compact row while the rebuilt metadata READS it from the new one --
silently wrong output, not a crash.

So :meth:`DVectorMenuRuntime.adopt` REFUSES BY NAME any switch into a vector
whose graph set is not already resident. That refusal is not a limitation
bolted on; it is what turns ``second_graph_set_trade`` from an optimisation
into the load-bearing question of the whole feature, and it is why the boot
declaration carries ``resident_graph_sets``.

The WEIGHT ratio is not installed from here at all -- that also needs the
arena refill, which this seam does not run.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Optional, Sequence, Tuple

from sglang.srt.planner.d_vector_menu import (
    DBacklog,
    DVector,
    DVectorMenuError,
    MenuCalibration,
    MenuVerdict,
    RankCensus,
    build_menu,
    select_d_vector,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "[D-VECTOR-MENU]"

#: Boot flag. Absent -> no menu, and every seam below is a no-op that returns
#: the boot-constant vector. Present and unreadable -> the boot DIES: a
#: declared menu that silently failed to load would run the pre-order layout
#: under a log line claiming a menu, which is the failure mode the #578
#: unmeasured rule exists to forbid.
MENU_ENV = "SGLANG_D_VECTOR_MENU"

#: Attribute the declaration is parked on, on the PhaseFlipRuntime.
MENU_ATTR = "d_vector_menu"

#: The boot log line every CUDA-graph capture emits, on every rank
#: (``model_executor/model_runner.py``: "Capture {name} ... end. elapsed=..").
#: Quoted rather than re-invented so a format change is a failed parse instead
#: of a silently wrong number.
CAPTURE_END_RE = re.compile(
    r"Capture\s+(?P<name>.+?)\s+(?:CUDA graph|cuda graph|\S+)\s+end\.\s+"
    r"elapsed=(?P<elapsed>[0-9]+(?:\.[0-9]+)?)\s*s",
)

#: Rank tag as the boot log prints it, e.g. ``[2026-09-08 16:30:17 PP0]``.
RANK_TAG_RE = re.compile(r"\[(?:[^\]]*\s)?(?P<rank>(?:PP|TP|DP)\d+)\]")

__all__ = [
    "LOG_PREFIX",
    "MENU_ENV",
    "MENU_ATTR",
    "DVectorMenuRuntime",
    "declare_menu_from_env",
    "declare_menu_from_payload",
    "flip_backlog_from_scheduler",
    "effective_token_vector",
    "menu_of",
    "graph_capture_seconds_from_boot_log",
]


# ---------------------------------------------------------------------------
# 1. DECLARATION
# ---------------------------------------------------------------------------


class DVectorMenuRuntime:
    """The boot-declared ceiling set, plus who may install what.

    One instance per rank. Every rank builds it from the SAME declaration
    file and the SAME census, so :attr:`menu_fingerprint` is equal across the
    group by construction -- and when it is not, that is the one thing this
    class treats as unsurvivable.
    """

    def __init__(
        self,
        *,
        rank: int,
        menu: Sequence[DVector],
        census: RankCensus,
        calibration: MenuCalibration,
        current: str,
        preference: Optional[Dict[str, str]] = None,
        resident_graph_sets: Sequence[str] = (),
        resident_arena_layouts: Sequence[str] = (),
        source: str = "<unknown>",
    ) -> None:
        if not menu:
            raise DVectorMenuError(
                f"{LOG_PREFIX} declared an EMPTY ceiling set. An empty menu is "
                f"not 'keep the boot vector' -- it is a declaration that named "
                f"nothing, and accepting it would let the install seam fall "
                f"back to the boot vector under a log line claiming a menu."
            )
        self._rank = int(rank)
        self._menu: Tuple[DVector, ...] = tuple(menu)
        self._by_name: Dict[str, DVector] = {v.name: v for v in self._menu}
        if len(self._by_name) != len(self._menu):
            raise DVectorMenuError(
                f"{LOG_PREFIX} two menu entries share a name: "
                f"{[v.name for v in self._menu]}"
            )
        self._census = census
        self._calib = calibration
        self._preference = dict(preference or {})
        self._source = str(source)

        if current not in self._by_name:
            raise DVectorMenuError(
                f"{LOG_PREFIX} the boot names {current!r} as the resident "
                f"vector, but it is not in the declared ceiling set "
                f"{sorted(self._by_name)}. The resident vector is the one the "
                f"pools were actually built for, so a boot that cannot find "
                f"it in its own menu has two different ideas of what is in "
                f"VRAM."
            )
        self._current = str(current)
        #: The boot-resident weight ratio. Only entries carrying THIS ratio
        #: can be installed by a pure token-key flip; see :meth:`adopt`.
        self._resident_weight_ratio = self._by_name[self._current].weight_ratio
        #: Names whose CUDA-graph set is captured and resident. The current
        #: vector is always among them (its graphs are the boot's). Anything
        #: else had to be paid for by a declared second_graph_set_trade, which
        #: is the only thing that makes a menu switch installable at all --
        #: see the module docstring's graph gate.
        self._resident_graph_sets = frozenset(
            {str(current), *(str(x) for x in resident_graph_sets)}
        )
        #: Names whose WEIGHT COLUMN RANGE this rank can serve without a new
        #: arena build. The menu's entries differ on BOTH axes by construction
        #: -- ``derive_vector`` reads the token key off the weight vector, so
        #: two entries with one weight ratio would be one entry -- which means
        #: a menu switch is never weight-free, and pretending otherwise would
        #: make the token key the only thing installed while the shards stayed
        #: where they were.
        #:
        #: Declared exactly like the graph sets, and for the same reason: the
        #: seam cannot inspect VRAM, so what it can install is what the boot
        #: says it prepared. Default = the resident layout only, i.e. a boot
        #: that prepares nothing gets a refusal rather than a half-install.
        self._resident_arena_layouts = frozenset(
            {str(current), *(str(x) for x in resident_arena_layouts)}
        )
        unknown = sorted(
            (self._resident_graph_sets | self._resident_arena_layouts)
            - set(self._by_name)
        )
        if unknown:
            raise DVectorMenuError(
                f"{LOG_PREFIX} the boot declares graph sets / arena layouts "
                f"resident for {unknown}, which are not menu entries "
                f"{sorted(self._by_name)}. A captured set nobody can select "
                f"is VRAM spent on nothing, and the mismatch more likely means "
                f"the declaration and the capture disagree about the names."
            )
        self._flips_on_current = 0
        #: Set by PP0 at publish time, consumed at adoption. Never read by a
        #: follower -- a follower's choice is PP0's, delivered.
        self._proposed: Optional[MenuVerdict] = None

        self._fingerprint = _menu_fingerprint(self._menu)
        logger.info(
            "%s rank %d declared ceiling set %s from %s; resident %r, "
            "menu fingerprint %s. Pools must be sized for the ELEMENTWISE MAX "
            "of the declared caps %s -- a vector outside this set is refused "
            "by name at adoption, never installed. INSTALLABLE TODAY: %s "
            "(graph sets %s, arena layouts %s); anything else holds.",
            LOG_PREFIX,
            self._rank,
            [v.name for v in self._menu],
            self._source,
            self._current,
            self._fingerprint,
            list(self.ceiling_kv_cap_rows),
            list(self.installable),
            list(self.resident_graph_sets),
            list(self.resident_arena_layouts),
        )

    # -- what was declared --------------------------------------------------
    @property
    def rank(self) -> int:
        return self._rank

    @property
    def menu(self) -> Tuple[DVector, ...]:
        return self._menu

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_name))

    @property
    def allowed_token_vectors(self) -> Tuple[Tuple[int, ...], ...]:
        """The ceiling set in the shape ``KvReshardRuntime.allowed_vectors``
        publishes it, so the two disciplines read alike at a glance."""
        return tuple(v.token_ratio for v in self._menu)

    @property
    def ceiling_kv_cap_rows(self) -> Tuple[int, ...]:
        """Per-rank rows the pools must carry to back EVERY declared vector.

        The elementwise MAX, not the max entry's vector: two vectors can each
        be the larger one on a different rank, and sizing to either alone
        leaves the other pointing the owner rule at rows that were never
        allocated.
        """
        n = self._census.n_ranks
        return tuple(max(v.kv_cap_rows[r] for v in self._menu) for r in range(n))

    @property
    def menu_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def current(self) -> str:
        return self._current

    @property
    def current_vector(self) -> DVector:
        return self._by_name[self._current]

    @property
    def flips_on_current(self) -> int:
        return self._flips_on_current

    @property
    def resident_graph_sets(self) -> Tuple[str, ...]:
        return tuple(sorted(self._resident_graph_sets))

    @property
    def resident_arena_layouts(self) -> Tuple[str, ...]:
        return tuple(sorted(self._resident_arena_layouts))

    @property
    def installable(self) -> Tuple[str, ...]:
        """What this rank could actually install today.

        The intersection, stated once: an entry needs BOTH a resident graph
        set (or its replay writes under stale owner bounds) and a resident
        arena layout (or its weights are not in VRAM). A menu whose
        ``installable`` is a single name is a menu that will only ever hold,
        and a boot should be able to see that from the log rather than from a
        refusal three flips later.
        """
        return tuple(sorted(self._resident_graph_sets & self._resident_arena_layouts))

    @property
    def census(self) -> RankCensus:
        return self._census

    def describe(self) -> str:
        return (
            f"{LOG_PREFIX} rank {self._rank}: resident {self._current!r}, "
            f"declared {list(self.names)}, fingerprint {self._fingerprint}, "
            f"{self._flips_on_current} flips on the current vector"
        )

    # -- refusal by name ----------------------------------------------------
    def require_declared(self, name: str) -> DVector:
        """The one refusal this module exists to make possible.

        Shaped after ``KvReshardRuntime.arm``'s refusal for the same mistake
        (``kv_reshard.py``: "is not in the declared ceiling set ...; the pool
        has no reserved rows for it"), because it IS the same mistake: an
        owner rule pointed at a split with no allocation behind it.
        """
        entry = self._by_name.get(str(name))
        if entry is None:
            raise DVectorMenuError(
                f"{LOG_PREFIX} REFUSED: {name!r} is not in the declared "
                f"ceiling set {sorted(self._by_name)} on rank {self._rank}. "
                f"The pools were pre-sized for the declared set only, so "
                f"installing it would split the token space under a vector "
                f"that has no rows reserved for it -- an out-of-bounds slot "
                f"id, not a slow path. Declare it in {MENU_ENV} and reboot."
            )
        return entry

    # -- 3. the carrier ------------------------------------------------------
    def propose(
        self,
        backlog: DBacklog,
        *,
        resident_kv_tokens: int = 0,
        gain_pct: Optional[Dict[str, float]] = None,
    ) -> MenuVerdict:
        """PP0 ONLY. Choose the D vector for the flip that is about to run."""
        verdict = select_d_vector(
            backlog,
            self._menu,
            self._census,
            self._calib,
            current=self._current,
            preference=self._preference or None,
            gain_pct=gain_pct,
            resident_kv_tokens=int(resident_kv_tokens),
            flips_on_current=self._flips_on_current,
            # The BOOT's declaration, not a caller's opinion: whether a set is
            # resident is a fact about what was captured, and letting a caller
            # override it would let the price model claim a free switch the
            # install seam then refuses.
            resident_graph_sets=self.resident_graph_sets,
        )
        self._proposed = verdict
        return verdict

    def proposal(self) -> Optional[MenuVerdict]:
        return self._proposed

    def clear_proposal(self) -> None:
        self._proposed = None

    def adopt(
        self,
        *,
        name: str,
        token_ratio: Sequence[int],
        menu_fp: str,
        told_by: str = "PP0",
    ) -> DVector:
        """Install PP0's choice on THIS rank. The two failure classes, apart.

        Returns the adopted entry; the caller installs
        ``entry.token_ratio`` and nothing else.
        """
        # (a) DIFFERENT MENU -> CRASH/STOP. Not a refusal: a refusal implies a
        # safe state to refuse INTO, and there is none here. Two ranks with
        # different ceiling sets built different pools at boot, so they have
        # already been running on divergent premises and every later number
        # from either is suspect.
        if str(menu_fp) != self._fingerprint:
            raise DVectorMenuError(
                f"{LOG_PREFIX} RANKS DISAGREE ABOUT THE MENU ITSELF: "
                f"{told_by} carries fingerprint {menu_fp!r}, rank "
                f"{self._rank} holds {self._fingerprint!r} over "
                f"{sorted(self._by_name)}. The menu is a pure function of the "
                f"declaration file, the census and the calibration, so a "
                f"divergence means those inputs differ per rank -- the pools "
                f"were then sized differently too. The group STOPS here, "
                f"before any rank moves a byte under the wrong layout."
            )
        # (b) UNDECLARED NAME -> refusal by name (see require_declared).
        entry = self.require_declared(name)

        told = tuple(int(x) for x in token_ratio)
        if told != entry.token_ratio:
            raise DVectorMenuError(
                f"{LOG_PREFIX} RANKS DISAGREE ABOUT THE VECTOR BEHIND A NAME: "
                f"{told_by} sends {name!r} = {list(told)}, rank {self._rank} "
                f"derives {list(entry.token_ratio)} for the same name under "
                f"the same menu fingerprint. A name that means two token "
                f"splits is the silent out-of-bounds slot id the fingerprint "
                f"was supposed to exclude, so the premise is false and the "
                f"group STOPS."
            )
        switched = entry.name != self._current

        # (c) THE ARENA GATE. A menu entry moves BOTH axes -- the weight
        # column range and the token key -- because the key is derived from
        # the vector. Installing the key alone would leave the owner rule
        # splitting rows under a column split the weights do not have. The
        # boot says which layouts it prepared an arena image for; anything
        # else is a refusal, not a best effort.
        if switched and entry.name not in self._resident_arena_layouts:
            raise DVectorMenuError(
                f"{LOG_PREFIX} REFUSED: {name!r} carries weight ratio "
                f"{list(entry.weight_ratio)} against the resident "
                f"{list(self._resident_weight_ratio)}, and this boot declared "
                f"an arena layout only for "
                f"{sorted(self._resident_arena_layouts)}. The token key is "
                f"derived from the weight vector, so a menu switch is never "
                f"token-only; installing the key without the matching shards "
                f"would point the owner rule at a column split that is not in "
                f"VRAM. Prepare the layout image for {name!r} at boot, or keep "
                f"the menu at one entry."
            )

        # (d) THE GRAPH GATE. A token-key change is NOT graph-neutral on this
        # tree; the module docstring carries the file:line chain. Without a
        # resident graph set for the destination, the replayed decode would
        # keep WRITING under the old owner bounds while the host-rebuilt
        # metadata READS under the new ones -- wrong tokens, no exception, and
        # only on the graph-covered fast path, so an eager smoke test would
        # show the switch working. That is the worst shape available here, so
        # this refuses rather than installs.
        if switched and entry.name not in self._resident_graph_sets:
            raise DVectorMenuError(
                f"{LOG_PREFIX} REFUSED: {name!r} has no RESIDENT CUDA-GRAPH "
                f"SET on rank {self._rank} (resident: "
                f"{sorted(self._resident_graph_sets)}). Its token key "
                f"{list(entry.token_ratio)} differs from the resident "
                f"{list(self._by_name[self._current].token_ratio)}, and the "
                f"weighted owner rule's WRITE side reads cp_S/cp_lo/cp_hi/"
                f"cp_ratio as python scalars INSIDE the captured decode body "
                f"(layers/dcp/owner.py:429-436 from "
                f"flashinfer_backend.py:2594 / triton_backend.py:2392 / "
                f"qwen_sparse_attn_backend.py:457, captured at "
                f"decode_cuda_graph_runner.py:1893). Those scalars are baked "
                f"by value and ShapeKey carries no layout axis, so the stale "
                f"graph would be reused SILENTLY: the replay writes token L to "
                f"the old compact row while the rebuilt kv_indices read it "
                f"from the new one. Capture {name!r}'s set at boot (the "
                f"second_graph_set_trade, priced in GESAMTPOOL tokens) or "
                f"keep the menu at one entry."
            )
        if switched:
            logger.warning(
                "%s rank %d INSTALLS %r: token key %s -> %s, GESAMTPOOL %d "
                "(told by %s, menu %s). The vector is no longer "
                "boot-constant; this is the flip that moved it.",
                LOG_PREFIX,
                self._rank,
                entry.name,
                list(self._by_name[self._current].token_ratio),
                list(entry.token_ratio),
                entry.pool_tokens,
                told_by,
                self._fingerprint,
            )
            self._current = entry.name
            self._flips_on_current = 0
        else:
            self._flips_on_current += 1
        self._proposed = None
        return entry


def _menu_fingerprint(menu: Sequence[DVector]) -> str:
    """What every rank must agree on BEFORE a choice is even made.

    The whole ceiling set, by name and by the three vectors that define a
    layout -- not the prose, and not the census that produced them: two ranks
    may legitimately carry different per-rank bytes in their census (they are
    different cards) while deriving the same set of layouts. What may never
    differ is the SET.
    """
    from sglang.srt.planner.d_vector_menu import _digest

    return _digest(
        [
            {
                "name": v.name,
                "weight": list(v.weight_ratio),
                "token": list(v.token_ratio),
                "cap": list(v.kv_cap_rows),
                "pool": v.pool_tokens,
            }
            for v in sorted(menu, key=lambda e: e.name)
        ]
    )


def declare_menu_from_payload(
    payload: dict, *, rank: int, source: str = "<payload>"
) -> DVectorMenuRuntime:
    """Build the ceiling set from a plain dict. Pure; no env, no files.

    The payload carries the MEASURED terms and the names; every derived
    quantity (token key, caps, GESAMTPOOL, prices) comes out of
    ``derive_vector``. Nothing in it is a hand-pinned pool figure -- the one
    thing the Planner-Alleinzustaendigkeit rule forbids.
    """
    try:
        census = RankCensus(**_tuple_ify(payload["census"]))
        calib = MenuCalibration(**dict(payload.get("calibration", {})))
        positions = {
            str(k): tuple(int(x) for x in v)
            for k, v in dict(payload["entries"]).items()
        }
    except KeyError as e:
        raise DVectorMenuError(
            f"{LOG_PREFIX} the declaration at {source} is missing {e}. It "
            f"needs 'census', 'entries' and 'current'; 'calibration' and "
            f"'preference' are optional. A declaration that half-parses must "
            f"not become a half-menu."
        ) from e
    except TypeError as e:
        raise DVectorMenuError(
            f"{LOG_PREFIX} the declaration at {source} does not fit the "
            f"census/calibration shape: {e}"
        ) from e

    menu = build_menu(positions, census)
    return DVectorMenuRuntime(
        rank=rank,
        menu=menu,
        census=census,
        calibration=calib,
        current=str(payload["current"]),
        preference=dict(payload.get("preference", {})),
        resident_graph_sets=tuple(payload.get("resident_graph_sets", ())),
        resident_arena_layouts=tuple(payload.get("resident_arena_layouts", ())),
        source=source,
    )


def _tuple_ify(census_payload: dict) -> dict:
    """JSON has no tuples; RankCensus is frozen and compares by value."""
    out = dict(census_payload)
    for key, value in list(out.items()):
        if isinstance(value, list):
            out[key] = tuple(value)
    return out


def declare_menu_from_env(
    *, rank: int, env: Optional[Dict[str, str]] = None
) -> Optional[DVectorMenuRuntime]:
    """Boot seam. ``None`` when no menu is declared -- the pre-order path.

    A PRESENT-BUT-BROKEN declaration RAISES. The alternative (log and carry
    on) is the exact shape of the defect class this fork keeps paying for: a
    feature flag that is set, a feature that is off, and a log that says
    neither.
    """
    src = (env if env is not None else os.environ).get(MENU_ENV)
    if not src:
        return None
    try:
        with open(src, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as e:
        raise DVectorMenuError(
            f"{LOG_PREFIX} {MENU_ENV}={src!r} is set but unreadable: {e}. A "
            f"declared menu that failed to load must not boot as no menu."
        ) from e
    except ValueError as e:
        raise DVectorMenuError(
            f"{LOG_PREFIX} {MENU_ENV}={src!r} is not valid JSON: {e}"
        ) from e
    return declare_menu_from_payload(payload, rank=rank, source=src)


def menu_of(holder) -> Optional[DVectorMenuRuntime]:
    """The declared menu, or ``None``.

    PARKED ON THE SCHEDULER, not on the PhaseFlipRuntime, and that is not
    arbitrary: ``scheduler.phase_flip_runtime`` is read in a dozen places in
    this tree but ASSIGNED in none of the production ones (only test doubles
    set it), so a seam that reached the menu through it would be a seam that
    is always ``None`` at metal. The scheduler is reachable from both ends
    that need the menu -- the cutover closure takes it as an argument, and the
    runtime keeps it as ``_census_scheduler``.

    Accepts either, so a caller does not have to know which it holds.
    """
    if holder is None:
        return None
    menu = getattr(holder, MENU_ATTR, None)
    if menu is not None:
        return menu
    return getattr(getattr(holder, "_census_scheduler", None), MENU_ATTR, None)


# ---------------------------------------------------------------------------
# 2. SENSOR -- the backlog WITH DEPTHS
# ---------------------------------------------------------------------------


def flip_backlog_from_scheduler(scheduler) -> Optional[DBacklog]:
    """What the D phase is about to serve, counted AND measured.

    THE TERMS ARE THE SCHEDULER'S OWN. ``managers/scheduler.py`` already
    reduces the waiting queue to exactly ``len(self.waiting_queue)`` /
    ``sum(len(req.origin_input_ids) ...)`` / ``max(len(...))`` for the regime
    observer, and the running batch to ``sum(req.seqlen for req in
    running_batch.reqs)``. Re-deriving them here with a different expression
    would give the menu and the observer two different backlogs for one round.

    ADMITTED WORK COUNTS AS BACKLOG, and that is not a liberty -- it is the
    lesson ``_arming_condition_persists`` already records in prose: "The load
    that most wants the other layout is the load that is already in the
    machine." A count that drops to zero the moment work is admitted cannot
    see the deep-single-request case at all, which is one whole end of the
    menu.

    So ``queued_reqs`` here is waiting + running, and ``depths`` carries one
    length per counted request (prompt length for a waiting one, ``seqlen``
    for a running one). ``held_tokens`` stays 0 BECAUSE the running tokens are
    already in ``depths``; carrying them twice would inflate
    ``demand_tokens`` by the resident working set and make every flip look
    like it needs a bigger pool than it does.

    Returns ``None`` -- never a fabricated empty backlog -- when the scheduler
    cannot be read. A missing sensor must not read as "no load"; the caller
    holds the current vector instead.
    """
    if scheduler is None:
        return None
    try:
        depths = []
        for name in ("waiting_queue", "grammar_queue"):
            q = getattr(scheduler, name, None) or ()
            for req in q:
                ids = getattr(req, "origin_input_ids", None)
                depths.append(int(len(ids)) if ids is not None else 0)
        seen = set()
        for name in ("running_batch", "cur_batch"):
            batch = getattr(scheduler, name, None)
            reqs = getattr(batch, "reqs", None) if batch is not None else None
            for req in reqs or ():
                # running_batch and cur_batch overlap by construction at some
                # points of the loop; the same request must be one entry.
                key = id(req)
                if key in seen:
                    continue
                seen.add(key)
                depths.append(int(getattr(req, "seqlen", 0) or 0))
    except Exception as e:  # noqa: BLE001 - a sensor must not raise into a flip
        logger.warning(
            "%s backlog sensor unreadable (%s); the menu holds the current "
            "vector rather than choosing on a guess.",
            LOG_PREFIX,
            e,
        )
        return None

    depths_t = tuple(max(0, d) for d in depths)
    return DBacklog(
        queued_reqs=len(depths_t),
        queued_prompt_tokens=int(sum(depths_t)),
        max_queued_prompt_tokens=int(max(depths_t)) if depths_t else 0,
        held_tokens=0,
        depths=depths_t,
    )


# ---------------------------------------------------------------------------
# 4. INSTALL
# ---------------------------------------------------------------------------


def effective_token_vector(holder, boot_vector: Sequence[int]) -> Tuple[int, ...]:
    """The token key the cutover installs.

    Equal to ``boot_vector`` bit-for-bit whenever no menu is declared, which
    is what keeps the pre-order boot unchanged. With a menu, it is the entry
    THIS rank adopted -- which is PP0's choice, because adoption is the only
    writer and a follower adopts only what it was told.
    """
    menu = menu_of(holder)
    if menu is None:
        return tuple(int(x) for x in boot_vector)
    return tuple(int(x) for x in menu.current_vector.token_ratio)


# ---------------------------------------------------------------------------
# 5. THE CAPTURE MEASUREMENT (#578: measured, or named unmeasured)
# ---------------------------------------------------------------------------


def graph_capture_seconds_from_boot_log(path: str) -> Dict[str, Dict[str, float]]:
    """Per-rank CUDA-graph capture seconds, read from a boot log.

    ``RankCensus.graph_capture_s`` is the dominant term of
    :func:`~sglang.srt.planner.d_vector_menu.switch_price` -- it decides
    whether a menu is worth having at all -- so it may not be a modelled
    number. The runtime already prints it per rank and per capture set:

        ``Capture target decode CUDA graph end. elapsed=2.18 s, mem usage=..``

    Returns ``{rank_tag: {capture_name: seconds, "__total__": seconds}}``.
    The TOTAL per rank is the sum over capture sets, because a switch that
    invalidates the shapes invalidates all of them; the per-set breakdown is
    kept so a reader can see WHICH set dominates instead of trusting one sum.

    An empty result means the log never reached capture -- which is an absence
    of measurement, not a measurement of zero. The caller must then leave
    ``graph_capture_s`` at 0.0 and say "unmeasured", never quote a default.
    """
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = CAPTURE_END_RE.search(line)
            if m is None:
                continue
            tag = RANK_TAG_RE.search(line)
            rank = tag.group("rank") if tag else "?"
            name = m.group("name").strip()
            secs = float(m.group("elapsed"))
            bucket = out.setdefault(rank, {})
            # A rank that captures the same set twice (a rebuild) is a real
            # event, not a duplicate to drop: both captures were paid for.
            bucket[name] = bucket.get(name, 0.0) + secs
            bucket["__total__"] = bucket.get("__total__", 0.0) + secs
    return out
