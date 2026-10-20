"""HW-GENERIC 1002 Stage 2 / HW-P1a 1003: the pdflip group topology as a
function of the INVENTORY (its card count), instead of a fixed
``PDFLIP_CARD_COUNT = 3``.

User order 03.10. ~19:30Z (verbatim): "unsere software muss mit beliebiger
anzahl an karten und sm86 sm89 und sm120 laufen. 1,2,3,4,5,6 ... karten".

Release topology (proven on metal for N = 3 only): group P = TP1 x PP<N>
over every card, group D = TP<N> x PP1 over every card, one rank per card,
rank i on the i-th card of ``order_cards`` (biggest first); Form A (the D
attention host on ordinal 0, experts on the workers) comes from the profile
argv (``--rank-role host,worker,...``), not from here.

What this module decides:

* the argv sizes and the rank map the launcher prints -- byte-identical for
  N = 3 ("--pp-size 3", "--tp-size 3", "--rank-gpu-id 0,1,2");
* whether the launch's card count is servable (:func:`plan_topology`, wired
  into ``launcher.main`` by :func:`launcher.topology_check_line`): N in
  :data:`PROVEN_CARD_COUNTS` passes exactly as before; any other N is
  refused BY NAME with the CONCRETE list of what still assumes three ranks
  for THIS launch (:func:`blockers`), instead of a blanket HW-COUNT.

Each blocker is a PROBE of the live code where one exists (the exchange
region's ``N_CARDS``, the PP-cut floor tuple, the format's cut pin, the
profile's positional vectors, the L1.5 ordinal keys): when that path is made
N-capable the blocker disappears by itself, and the simulation harness
(``tools/hw_sim.py``) shows the progress. A blocker that is code shape, not a
constant (the dual front's ``range(3)``, the BAR1 window constants without a
feasibility check), is declared with its file and removed by the change that
fixes it. ``METAL-UNPROVEN`` stays for every N outside
:data:`PROVEN_CARD_COUNTS` until a metal boot adds N there.

PURE at import: stdlib only. The probes import their live constants lazily
(stdlib-only modules of this package).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

#: Card counts a release boot has been proven on (metal).
PROVEN_CARD_COUNTS: Tuple[int, ...] = (3,)
#: barlink_bar1_ext.py BARLINK_BAR1_MAX_RANKS (#define 8): the BAR1
#: collectives cannot address more ranks than this.
MAX_CARDS_BAR1 = 8
#: A flip needs two groups that share the cards; one card cannot hold a PP
#: or TP group of more than one rank (and PP1 == TP1 makes the flip moot).
MIN_CARDS = 2

#: Refusal codes (message heads).
CODE_TOPOLOGY = "HW-TOPOLOGY"   # N outside [MIN_CARDS, MAX_CARDS_BAR1]: no flip topology at all
CODE_COUNT = "HW-COUNT"         # N inside the range, not yet N-capable / proven


@dataclass(frozen=True)
class Blocker:
    """One concrete thing that keeps an N-card launch from running."""

    code: str
    where: str
    what: str

    def text(self) -> str:
        return f"[{self.code}] {self.what} ({self.where})"


@dataclass(frozen=True)
class TopologyContext:
    """What of THIS launch the blockers depend on (launcher: from ``ns`` and
    the environment, :func:`launcher.topology_context`; simulation: from the
    model table). Empty = the generic, most conservative view."""

    #: launcher profile row ("qwen27b" / "nextflash"); "" = unknown
    profile: str = ""
    #: the weight format of the profile row ("int8", "fp8", "nvfp4", "gguf", "int4-mixed")
    weight_format: str = ""
    #: --dual-layout / --dual-share
    dual: bool = False
    #: --pdflip-weight-source ("ring" | "exchange"); "" = unknown (exchange assumed)
    weight_source: str = ""
    #: FLLIPER_PDFLIP_L15 on, and the FLLIPER_PDFLIP_L15_MIB value
    l15: bool = False
    l15_mib: str = ""
    #: positional per-card vectors this launch carries: name -> entry count
    vectors: Mapping[str, int] = field(default_factory=dict)
    #: HW-P1c: the LIVE inventory (calibration-class labels in card order), the
    #: inventory the profile's vectors were written for, and the one its
    #: measured records were taken on; () = unknown (then nothing is derivable)
    live_inventory: Tuple[str, ...] = ()
    vector_inventory: Tuple[str, ...] = ()
    record_inventory: Tuple[str, ...] = ()
    #: BAR1 total (MiB) of each live card in card order, None = not reported
    bar1_mib: Tuple[Optional[int], ...] = ()


@dataclass(frozen=True)
class Topology:
    n_cards: int
    p_tp: int
    p_pp: int
    d_tp: int
    d_pp: int
    #: Form A attention host ordinal (the biggest card, ordinal 0 of order_cards)
    host_ordinal: int
    proven: bool
    blockers: Tuple[str, ...]

    @property
    def rank_gpu_id(self) -> Tuple[int, ...]:
        return tuple(range(self.n_cards))


class TopologyRefused(RuntimeError):
    """No pdflip topology for this card count (named, with its blockers)."""

    def __init__(self, msg: str, blockers: Sequence[Blocker] = ()):
        super().__init__(msg)
        self.blockers: Tuple[Blocker, ...] = tuple(blockers)


# ---------------------------------------------------------------------------
# the blocker probes (one per 3-shaped path; HW-KOPPLUNG-AUDIT-1002 /
# PLAN_HWGEN_N_KARTEN_1003 section 2a/2d/2e)


def _form_a(profile: str) -> bool:
    """The profile's D attention layout is Form A (host + workers; a registry
    row field, not a profile NAME)."""
    from flliper.srt.pdflip import form as _form

    row = _form.profile_row(profile) if profile else None
    return bool(row is not None and getattr(row, "d_layout", "") == "qsa_forma")


def _b_single(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n >= MIN_CARDS:
        return []
    out = [Blocker("SINGLE-MODE", "pdflip/topology.py MIN_CARDS=2; launcher --d-only (H87)",
                   "one card: P = PP1 and D = TP1 are the same layout, a flip is moot; the "
                   "single-group mode (--d-only TP1 behind the front on :30030, model fit check "
                   "before load) does not exist yet (plan P4)"),
           Blocker("BARLINK-R1", "barlink_bar1_ext.py R >= 2",
                   "the BAR1 collectives need at least two ranks; TP1 has no collective and no "
                   "transport 'none' is wired")]
    if ctx.dual:
        out.append(Blocker("DUAL-1", "launcher resolve_dual_layout",
                           "dual on one card = two engines with the same weights under MPS, i.e. "
                           "chunked prefill in one engine with two contexts; not a mode"))
    if _form_a(ctx.profile):
        out.append(Blocker("FORM-A-1", "managers/rank_role.py Form A >= 2 ranks",
                           "NF Form A needs a host and at least one worker; a one-card NF needs "
                           "expert offload from the host store (plan P4)"))
    return out


def _b_max(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n <= MAX_CARDS_BAR1:
        return []
    return [Blocker("BAR1-MAX-RANKS", f"barlink_bar1_ext.py BARLINK_BAR1_MAX_RANKS {MAX_CARDS_BAR1}",
                    f"{n} cards: the BAR1 collectives address at most {MAX_CARDS_BAR1} ranks; "
                    "beyond that only cells (several P/D pairs behind one front, plan P6)")]


def _b_xchg_region(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if str(ctx.weight_source or "") == "ring":
        return []   # the ring arm creates no exchange region
    from flliper.srt.pdflip import weight_exchange_region as _wxr

    # HW-P1c: the region's geometry is a function of N (geometry / configure);
    # a blocker only where it cannot be laid out for this count.
    problems = _wxr.layout_problems(n)
    if not problems:
        return []
    return [Blocker("XCHG-REGION", "pdflip/weight_exchange_region.py geometry(n)",
                    f"the host exchange region cannot be laid out for {n} cards: "
                    + "; ".join(problems))]


def _b_bar1_windows(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    from flliper.srt.pdflip import bar1_windows as _bw

    # HW-P1c (K3): the windows are derived from N and the MEASURED BAR1 of the
    # cards (pdflip/bar1_windows.py); a blocker only where they cannot fit.
    pl = _bw.plan(n, ctx.bar1_mib, dual=ctx.dual)
    if pl.ok:
        return []
    return [Blocker("BAR1-WINDOW", "pdflip/bar1_windows.py plan(); launcher P/D --barlink-bar1-window-mib",
                    pl.why)]


def _b_dual_front(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.dual:
        return []
    # HW-P1c: the dual KV loan reads one stage file per card (front.py reads
    # max(3, len(dual_kv_ledgers)) of them; a missing file is skipped), so it is
    # correct for every N >= 2. Nothing blocks here; the dual's own vectors and
    # records are judged by their probes.
    return []


def _b_pp_cut_floor(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    from flliper.srt.pdflip import DEFAULT_PP_ORDERED_CUT, pp_ordered_cut_for

    # HW-P1c: the ordered cut (user order 2026-09-09) is a floor for exactly its
    # own stage count; any other P group has NO floor by rule
    # (launcher.resolve_pool_floor n_stages) and the solver ranks the unfloored
    # makespan. A blocker only if the rule is missing or contradicts the tuple.
    got = pp_ordered_cut_for(n)
    if got is None and n != len(DEFAULT_PP_ORDERED_CUT):
        return []
    if got is not None and len(got) == n:
        return []
    return [Blocker("PP-CUT-FLOOR", "pdflip/__init__.py pp_ordered_cut_for",
                    f"the 27B pool floor of the PP cut has no rule for P = PP{n}")]


def _b_pp_cut_pin(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.profile or not ctx.weight_format:
        return []
    from flliper.srt.pdflip import form as _form

    row = _form.profile_row(ctx.profile)
    wf = None if row is None else row.formats.get(ctx.weight_format)
    pin = getattr(wf, "p_cut_pin", None)
    if not pin:
        return []
    bad = [p for p in pin if len(p) != n]
    if not bad:
        return []
    # HW-P1c: a pin holds only for its own stage count. For another N it is
    # DROPPED (inventory_view FLAG_POLICY cut-pin) and the planner's cut solver
    # (planner/pp_cut.py: layers x card rates x budgets) derives the cut. A
    # blocker only where the drop is not wired for the pinned flag.
    from flliper.srt.pdflip import inventory_view as _iv

    if all(_iv.FLAG_POLICY.get(f) == _iv.CUT_PIN for f in ("--pp-stage-ratio", "--pp-attn-stage-ratio")):
        return []
    return [Blocker("PP-CUT-PIN",
                    f"pdflip/form.py {ctx.profile}.formats[{ctx.weight_format!r}].p_cut_pin",
                    "the format's P cut is pinned as "
                    + " / ".join(",".join(str(x) for x in p) for p in pin)
                    + f" ({len(bad[0])} stages); P = PP{n} needs its own cut (solver or pin)")]


def _b_vectors(n: int, ctx: TopologyContext) -> List[Blocker]:
    from flliper.srt.pdflip import inventory_view as _iv

    live, cal = tuple(ctx.live_inventory), tuple(ctx.vector_inventory)
    bad = sorted((k, int(v)) for k, v in (ctx.vectors or {}).items() if int(v) != n
                 and not (live and cal and _iv.vector_derivable(k, int(v), cal, live)))
    if not bad:
        return []
    return [Blocker("PROFILE-VECTORS", "profile argv/env positional vectors",
                    f"per-card vectors written for another inventory that cannot be derived for "
                    f"this one (derivation needs every live card to have a measured twin of its "
                    f"class and a policy for the vector): "
                    + ", ".join(f"{k} ({v} entries)" for k, v in bad)
                    + f"; this launch has {n} cards")]


def _b_l15(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.l15:
        return []
    from flliper.srt.pdflip import inventory_view as _iv
    try:
        from flliper.srt.pdflip import l15_plan as _l15
    except ImportError:  # NF tree: no l15_plan module; L15 is off by default there
        return []

    keys: List[int] = []
    try:
        mode, posts = _l15.parse_l15_mib(ctx.l15_mib)
        keys = sorted(posts) if mode == "override" else []
    except ValueError:
        keys = []   # identity keys: resolved (and refused by name) against the cards later
    beyond = [k for k in keys if k >= n]
    if mode_is_auto(ctx.l15_mib) or not keys:
        return []   # auto / identity keys: no ordinal of the 3-card rig is named
    # HW-P1c: ordinal posts are derived by class from the inventory they were
    # measured on (inventory_view.derive_l15_override); a blocker only where
    # that is impossible (a card without a measured twin, no inventory known).
    cal, live = tuple(ctx.record_inventory), tuple(ctx.live_inventory)
    if cal and live and _iv.l15_derivable(ctx.l15_mib, cal, live):
        return []
    what = ("the L1.5 posts are per-ordinal measurements of the 3-card rig (CAP0 proven on "
            "exactly one rank, K9) and are not derivable for this inventory (every live card "
            "needs a measured twin of its class holding a post)")
    if beyond:
        what += "; " + ", ".join(f"c{k}" for k in beyond) + f" names no card of {n}"
    return [Blocker("L15-POSTS", "FLLIPER_PDFLIP_L15_MIB " + (repr(ctx.l15_mib) if ctx.l15_mib else "auto"),
                    what)]


def mode_is_auto(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in ("", "auto")


def _b_records(n: int, ctx: TopologyContext) -> List[Blocker]:
    if not ctx.profile:
        return []
    from flliper.srt.pdflip import inventory_view as _iv
    from flliper.srt.pdflip import profile_records as _pr

    try:
        inv = _pr.inventory_of(str(ctx.profile))
        recs = _pr.records(str(ctx.profile))
    except Exception:  # noqa: BLE001 - a record file problem is named elsewhere (inventory check)
        return []
    width = len(inv) if inv is not None else 3
    if width == n and (not ctx.live_inventory or tuple(ctx.live_inventory) == tuple(inv or ())):
        return []
    pos = [(r.name, r.value) for r in recs if _pr.is_positional(r.value, width)]
    if not pos:
        return []
    # HW-P1c: positional records are DERIVED for a live subset of the cards
    # they were measured on (pdflip/inventory_view.py); a blocker names the
    # records that are not.
    if inv is not None and ctx.live_inventory:
        bad = list(_iv.assess_records(pos, tuple(inv), tuple(ctx.live_inventory)).underivable)
    else:
        bad = [p[0] for p in pos]
    if not bad:
        return []
    names = sorted(bad)
    return [Blocker("RECORDS-NVEC", f"pdflip/profile_records_data/{ctx.profile}.json",
                    f"{len(names)} measured records are {width}-vectors of one inventory that cannot "
                    f"be derived for this one ({', '.join(names[:4])}{', ...' if len(names) > 4 else ''}); "
                    f"{n} cards of this inventory need a calibration boot that writes their records "
                    "(plan K1/P3a)")]


def _b_metal(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n in PROVEN_CARD_COUNTS:
        return []
    return [Blocker("METAL-UNPROVEN", "pdflip/topology.py PROVEN_CARD_COUNTS",
                    f"no release boot on {n} cards yet (proven: {list(PROVEN_CARD_COUNTS)})")]


#: In report order: count limits, data paths, records, proof.
PROBES: Tuple[Callable[[int, TopologyContext], List[Blocker]], ...] = (
    _b_single, _b_max, _b_xchg_region, _b_bar1_windows, _b_dual_front, _b_pp_cut_floor,
    _b_pp_cut_pin, _b_vectors, _b_l15, _b_records, _b_metal,
)


def blockers(n_cards: int, ctx: Optional[TopologyContext] = None) -> Tuple[Blocker, ...]:
    """The concrete blockers of an ``n_cards`` launch in ``ctx``; empty for
    an N in :data:`PROVEN_CARD_COUNTS` (the proven layout is not re-judged:
    the reference boot passes exactly as before)."""
    n = int(n_cards)
    if n in PROVEN_CARD_COUNTS:
        return ()
    ctx = ctx or TopologyContext()
    out: List[Blocker] = []
    for probe in PROBES:
        out.extend(probe(n, ctx))
    return tuple(out)


#: Back-compat name (HW-GENERIC 1002 S2): the generic blocker texts of an
#: N != 3 launch without a context.
N_NOT_3_BLOCKERS: Tuple[str, ...] = tuple(b.text() for b in (
    _b_bar1_windows(2, TopologyContext()) + _b_metal(2, TopologyContext())))


def release_topology(n_cards: int) -> Topology:
    """The release layout for ``n_cards`` cards (P = PP<N>, D = TP<N>, one
    rank per card, host on ordinal 0) with its proof status. Raises
    :class:`TopologyRefused` outside [:data:`MIN_CARDS`, :data:`MAX_CARDS_BAR1`]."""
    n = int(n_cards)
    if n < MIN_CARDS or n > MAX_CARDS_BAR1:
        raise TopologyRefused(
            f"{CODE_TOPOLOGY}: {n} card(s): the pdflip flip needs {MIN_CARDS}..{MAX_CARDS_BAR1} cards "
            f"(two groups over the same cards; BAR1 collectives address at most {MAX_CARDS_BAR1} ranks)")
    proven = n in PROVEN_CARD_COUNTS
    return Topology(n_cards=n, p_tp=1, p_pp=n, d_tp=n, d_pp=1, host_ordinal=0,
                    proven=proven, blockers=() if proven else N_NOT_3_BLOCKERS)


def plan_topology(n_cards: int, ctx: Optional[TopologyContext] = None) -> Topology:
    """The topology of an ``n_cards`` launch, or :class:`TopologyRefused`
    BY NAME with the concrete :func:`blockers` of this launch:
    ``HW-TOPOLOGY`` for N outside [:data:`MIN_CARDS`, :data:`MAX_CARDS_BAR1`],
    ``HW-COUNT`` for an N inside it that is not N-capable / proven yet."""
    n = int(n_cards)
    bl = blockers(n, ctx)
    if n < MIN_CARDS or n > MAX_CARDS_BAR1:
        raise TopologyRefused(
            f"{CODE_TOPOLOGY}: {n} card(s): no pdflip flip topology (needs {MIN_CARDS}.."
            f"{MAX_CARDS_BAR1} cards). Blockers: " + "; ".join(b.text() for b in bl), bl)
    t = release_topology(n)
    if bl:
        raise TopologyRefused(
            f"{CODE_COUNT}: {n} cards would be P = TP1 x PP{t.p_pp}, D = TP{t.d_tp} x PP1 (host "
            f"ordinal {t.host_ordinal}); not yet runnable, {len(bl)} blocker(s): "
            + "; ".join(b.text() for b in bl), bl)
    return t


def topology_line(t: Topology) -> str:
    """The launcher's HW-TOPOLOGY line of a planned topology."""
    return (f"HW-TOPOLOGY N={t.n_cards}: P = TP{t.p_tp} x PP{t.p_pp}, D = TP{t.d_tp} x PP{t.d_pp}, "
            f"rank-gpu-id {rank_gpu_id_csv(t.n_cards)}, host ordinal {t.host_ordinal} -- "
            + ("proven on metal" if t.proven else "UNPROVEN") + f" (N in {list(PROVEN_CARD_COUNTS)})")


def rank_gpu_id_csv(n_cards: int) -> str:
    """``--rank-gpu-id`` of the release topology ("0,1,2" for N = 3)."""
    return ",".join(str(i) for i in release_topology(n_cards).rank_gpu_id)
