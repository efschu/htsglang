"""HW-GENERIC 1002 Stage 2 / HW-P1a 1003: the weg2 group topology as a
function of the INVENTORY (its card count), instead of a fixed
``WEG2_CARD_COUNT = 3``.

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
    #: --weg2-weight-source ("ring" | "exchange"); "" = unknown (exchange assumed)
    weight_source: str = ""
    #: SGLANG_WEG2_L15 on, and the SGLANG_WEG2_L15_MIB value
    l15: bool = False
    l15_mib: str = ""
    #: positional per-card vectors this launch carries: name -> entry count
    vectors: Mapping[str, int] = field(default_factory=dict)


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
    """No weg2 topology for this card count (named, with its blockers)."""

    def __init__(self, msg: str, blockers: Sequence[Blocker] = ()):
        super().__init__(msg)
        self.blockers: Tuple[Blocker, ...] = tuple(blockers)


# ---------------------------------------------------------------------------
# the blocker probes (one per 3-shaped path; HW-KOPPLUNG-AUDIT-1002 /
# PLAN_HWGEN_N_KARTEN_1003 section 2a/2d/2e)


def _b_single(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n >= MIN_CARDS:
        return []
    out = [Blocker("SINGLE-MODE", "weg2/topology.py MIN_CARDS=2; launcher --d-only (H87)",
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
    if str(ctx.profile) == "nextflash":
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
    from sglang.srt.weg2 import weight_exchange_region as _wxr

    if int(_wxr.N_CARDS) == n and len(_wxr.CROSS_PAIRS) == n * (n - 1):
        return []
    return [Blocker("XCHG-REGION",
                    f"weg2/weight_exchange_region.py N_CARDS={_wxr.N_CARDS}, "
                    f"{len(_wxr.CROSS_PAIRS)} fixed CROSS_PAIRS",
                    f"the host exchange region is laid out for {_wxr.N_CARDS} cards; "
                    f"{n} cards need {n * (n - 1)} directed pairs "
                    "(--weg2-weight-source exchange; the launcher sizes it n(n-1), the module "
                    f"lays out {len(_wxr.CROSS_PAIRS)})")]


def _b_bar1_windows(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    where = "launcher P_BARLINK_BAR1_WINDOW_MIB '24,PP_0=96', D windows 16+32+40"
    if ctx.dual:
        where += ", DUAL_P_BARLINK_BAR1_WINDOW_MIB '16,PP_0=64'"
    return [Blocker("BAR1-WINDOW", where,
                    f"the BAR1 group windows are constants sized for 2 peers per card on a "
                    f"256 MiB BAR with PP0 on a big BAR; {n} cards = {n - 1} peer(s) per card, "
                    "no feasibility check against the measured bar1_total_mib yet (plan P1c/K3)")]


def _b_dual_front(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.dual:
        return []
    return [Blocker("DUAL-FRONT-STAGES", "weg2/front.py dual KV loan: for r in range(3)",
                    f"the dual KV loan reads P's stage files 0..2; P has {n} stages "
                    "(plan P7, behind the dual gate only)")]


def _b_pp_cut_floor(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if str(ctx.profile) != "qwen27b":
        return []
    from sglang.srt.weg2 import DEFAULT_PP_ORDERED_CUT

    if len(DEFAULT_PP_ORDERED_CUT) == n:
        return []
    return [Blocker("PP-CUT-FLOOR",
                    "weg2/__init__.py DEFAULT_PP_ORDERED_CUT="
                    + ",".join(str(x) for x in DEFAULT_PP_ORDERED_CUT),
                    f"the 27B pool floor of the PP cut is a {len(DEFAULT_PP_ORDERED_CUT)}-stage "
                    f"tuple; P = PP{n} has no floor")]


def _b_pp_cut_pin(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.profile or not ctx.weight_format:
        return []
    from sglang.srt.weg2 import form as _form

    row = _form.profile_row(ctx.profile)
    wf = None if row is None else row.formats.get(ctx.weight_format)
    pin = getattr(wf, "p_cut_pin", None)
    if not pin:
        return []
    bad = [p for p in pin if len(p) != n]
    if not bad:
        return []
    return [Blocker("PP-CUT-PIN",
                    f"weg2/form.py {ctx.profile}.formats[{ctx.weight_format!r}].p_cut_pin",
                    "the format's P cut is pinned as "
                    + " / ".join(",".join(str(x) for x in p) for p in pin)
                    + f" ({len(bad[0])} stages); P = PP{n} needs its own cut (solver or pin)")]


def _b_vectors(n: int, ctx: TopologyContext) -> List[Blocker]:
    bad = sorted((k, int(v)) for k, v in (ctx.vectors or {}).items() if int(v) != n)
    if not bad:
        return []
    return [Blocker("PROFILE-VECTORS", "profile argv/env positional vectors",
                    f"per-card vectors written for another card count: "
                    + ", ".join(f"{k} ({v} entries)" for k, v in bad)
                    + f"; this launch has {n} cards")]


def _b_l15(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n < MIN_CARDS:
        return []   # no flip groups on one card (SINGLE-MODE names it)
    if not ctx.l15:
        return []
    from sglang.srt.weg2 import l15_plan as _l15

    keys: List[int] = []
    try:
        mode, posts = _l15.parse_l15_mib(ctx.l15_mib)
        keys = sorted(posts) if mode == "override" else []
    except ValueError:
        keys = []   # identity keys: resolved (and refused by name) against the cards later
    beyond = [k for k in keys if k >= n]
    what = ("the L1.5 posts are per-ordinal measurements of the 3-card rig (CAP0 proven on "
            "exactly one rank, K9)")
    if beyond:
        what += "; " + ", ".join(f"c{k}" for k in beyond) + f" names no card of {n}"
    return [Blocker("L15-POSTS", "SGLANG_WEG2_L15_MIB " + (repr(ctx.l15_mib) if ctx.l15_mib else "auto"),
                    what)]


def _b_records(n: int, ctx: TopologyContext) -> List[Blocker]:
    if not ctx.profile:
        return []
    from sglang.srt.weg2 import profile_records as _pr

    try:
        inv = _pr.inventory_of(str(ctx.profile))
        recs = _pr.records(str(ctx.profile))
    except Exception:  # noqa: BLE001 - a record file problem is named elsewhere (inventory check)
        return []
    width = len(inv) if inv is not None else 3
    if width == n:
        return []
    names = sorted({r.name for r in recs if _pr.is_positional(r.value, width)})
    if not names:
        return []
    return [Blocker("RECORDS-NVEC", f"weg2/profile_records_data/{ctx.profile}.json",
                    f"{len(names)} measured records are {width}-vectors of one inventory "
                    f"({', '.join(names[:4])}{', ...' if len(names) > 4 else ''}); {n} cards need "
                    "a calibration boot that writes N-vectors (plan K1/P2)")]


def _b_metal(n: int, ctx: TopologyContext) -> List[Blocker]:
    if n in PROVEN_CARD_COUNTS:
        return []
    return [Blocker("METAL-UNPROVEN", "weg2/topology.py PROVEN_CARD_COUNTS",
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
            f"{CODE_TOPOLOGY}: {n} card(s): the weg2 flip needs {MIN_CARDS}..{MAX_CARDS_BAR1} cards "
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
            f"{CODE_TOPOLOGY}: {n} card(s): no weg2 flip topology (needs {MIN_CARDS}.."
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
