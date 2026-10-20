"""EZ-S1 (plan nf-entscheidungszentrale-plan-1019 B1, step S1; user order 19.10. "nichts schaetzen, exakte Zahlen"):
the numbers a decision stands on -- one place, every number with its SOURCE and its AGE.

THE RULES, nailed in ``test_decision_numbers_1019.py``
------------------------------------------------------
* No estimate has a way into a field of a class. ``n_est`` (chars/3) is DISPLAY ONLY.
* **u for the class** (plan A1, replaces the first draft):
  1. ``u_d`` when exact and fresh (not older than ``fresh_ms``);
  2. else it is probed again (the arrival's ``_d_probe_result`` retries until its deadline; a QUEUED member is
     probed again by SET-REPROBE, S63r) -- both happen BEFORE this module is asked;
  3. else ``u_p`` -- but ONLY DOWNWARDS: ``u_p < halbX`` => kurz. That is safe because D reads the same store AND
     has its own radix, so ``u_d <= u_p``; the precondition is that ``n_prompt`` comes from D's ids (a text prompt's
     front ids ARE D's; media have no D ids at the front);
  4. ``u_p >= halbX`` or no number at all => ``MEASURE_ON_D``. NEVER half / big / p_only on ``u_p`` alone: an upper
     bound is no exact number for u_D and classifies systematically too high (a flip under X, SOLL R7; the very
     defect behind the user order -- rc16 pdflip-140-474: L3 view 18-24k, P computed 378-3590).
* UNKNOWN is never kurz and never gross. MEASURE_ON_D = D measures and refuses what is not kurz, with its exact
  extent (Stufe 1: the front hands D the measuring barrier ``halbX - 1``; see :func:`measure_limit`).
* S and class come from ``u_own`` (plan A4); ``s_p`` (the share of S_P) is carried and reported, it decides nothing.
* KV / seat readings are fresh or ABSENT (plan A10): a reading older than ``fresh_ms`` counts as not there, never as
  a number; ``kv_free@src/age_ms`` goes into the record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------------------------
# sources (closed set -- the checker of S4 reads them)
# ---------------------------------------------------------------------------------------------------------------

SRC_D_RADIX = "d_radix"
SRC_D_HOST = "d_host"
SRC_L2 = "l2"
SRC_L3 = "l3"
SRC_P_STORE = "p_store"
SRC_FRONT_IDS = "front_ids"
SRC_D_LEDGER = "d_ledger"
SRC_D_BEACON = "d_beacon"
SRC_D_EXTENT = "d_extent"     # D's refusal named its realised uncached extent (measured at D's admission)
SRC_D_PROBE = "d_probe"       # D's match probe (radix + host + store on D's ids)
SRC_NONE = "none"
SOURCES = (SRC_D_RADIX, SRC_D_HOST, SRC_L2, SRC_L3, SRC_P_STORE, SRC_FRONT_IDS, SRC_D_LEDGER, SRC_D_BEACON,
           SRC_D_EXTENT, SRC_D_PROBE, SRC_NONE)

#: class names (the same strings as phase_policy.CLASS_*; kept here as plain strings so the module stays a leaf)
CLASS_KURZ = "kurz"
CLASS_HALB = "halb"
CLASS_GROSS = "gross"
CLASS_P_ONLY = "p_only"
CLASS_UNKNOWN = "unknown"

KIND_EXACT = "exact"          # u_d exact and fresh: the class stands on it
KIND_P_UPPER = "p_upper"      # u_p < halbX: kurz, the SAFE direction (an upper bound of u_d)
KIND_MEASURE = "measure"      # no exact number: D measures, nothing is classified


@dataclass(frozen=True)
class Exact:
    value: int
    src: str
    t_mono: float

    def age_ms(self, now: float) -> float:
        return max(0.0, (float(now) - float(self.t_mono)) * 1000.0)


@dataclass(frozen=True)
class RequestNumbers:
    rid: str
    n_prompt: Optional[Exact] = None
    covered_d: Optional[Exact] = None
    u_d: Optional[Exact] = None
    u_p: Optional[Exact] = None
    u_own: Optional[Exact] = None
    s_p: Optional[int] = None
    depth: int = 0
    x_own: int = 0
    image_end: int = 0
    image_covered: Optional[bool] = None
    image_in_tail: Optional[bool] = None
    tower_on_d: Optional[bool] = None
    n_est: int = 0
    unpriced_why: Optional[str] = None


@dataclass(frozen=True)
class GroupNumbers:
    t_mono: float = 0.0
    epoch: int = 0
    awake: str = ""
    state: str = ""
    d_decoding: bool = False
    seats_free: Optional[Exact] = None
    kv_free: Optional[Exact] = None
    d_prefill_running_rest: Optional[Exact] = None
    d_granted_unstarted: int = 0
    flip_phase_ms: Optional[float] = None


@dataclass(frozen=True)
class ClassBasis:
    """What the class of one arrival may stand on."""
    kind: str                   # KIND_*
    u: Optional[int]            # the number the class stands on (None: MEASURE)
    src: str
    klass: Optional[str]        # CLASS_KURZ for exact-kurz / p_upper; the exact class for KIND_EXACT; None: MEASURE
    why: str
    age_ms: int = 0


def classify(u: int, x_tokens: int, half_x: int) -> str:
    """The class of ONE request on an EXACT u (the arithmetic of ``phase_policy.classify_arrival`` without p_only)."""
    if int(u) >= int(x_tokens):
        return CLASS_GROSS
    if int(u) >= int(half_x):
        return CLASS_HALB
    return CLASS_KURZ


def class_basis(*, u_d: Optional[Exact], u_p: Optional[Exact], n_from_d_ids: bool, x_tokens: int, half_x: int,
                now: float, fresh_ms: int = 500) -> ClassBasis:
    """The plan-A1 ladder (module docstring). Pure: no I/O, no clock but ``now``.

    ``u_d`` older than ``fresh_ms`` is NOT used (the caller had to probe again before asking; a stale number never
    classifies). ``u_p`` is used for ONE thing: ``u_p < half_x`` => kurz, and only when ``n_from_d_ids``."""
    if u_d is not None:
        age = u_d.age_ms(now)
        if age <= float(fresh_ms):
            return ClassBasis(KIND_EXACT, int(u_d.value), u_d.src, classify(u_d.value, x_tokens, half_x),
                              "u_d exact, age_ms=%d" % int(age), int(age))
    if u_p is not None and n_from_d_ids and int(u_p.value) < int(half_x):
        return ClassBasis(KIND_P_UPPER, int(u_p.value), u_p.src, CLASS_KURZ,
                          "u_p=%d < halfX=%d: kurz in the safe direction (u_d <= u_p, D reads the same store plus "
                          "its radix)" % (int(u_p.value), int(half_x)), int(u_p.age_ms(now)))
    why = ("no exact fresh u_d" if u_d is None else "u_d older than fresh_ms=%d" % int(fresh_ms))
    if u_p is None:
        why += ", no u_p"
    elif not n_from_d_ids:
        why += ", u_p without D ids (media): no upper bound"
    else:
        why += ", u_p=%d >= halfX=%d is an upper bound, no number for the class" % (int(u_p.value), int(half_x))
    return ClassBasis(KIND_MEASURE, None, SRC_NONE, None, why + ": D measures")


def measured_class(u: int, x_tokens: int, half_x: int) -> str:
    """The class of a MEASURE request after D reported its realised extent ``u`` (:data:`SRC_D_EXTENT`)."""
    return classify(u, x_tokens, half_x)


def measure_limit(half_x: int) -> int:
    """The refusal limit D must apply to a MEASURE request: ``halbX - 1`` -- D admits ``u <= halbX - 1`` (kurz, R2:
    right on D) and refuses every ``u >= halbX`` with its exact extent (``scheduler._pdflip_x_refuses`` refuses when
    ``uncached > limit``)."""
    return max(0, int(half_x) - 1)


def measure_front_x(half_x: int, page_size: Optional[int], static_x: int = 0) -> int:
    """The ``x-pdflip-front-x`` header value for MEASURE_ON_D: ``front_x + tolerance(page) == halbX - 1`` so that
    ``front_x.limit_for`` yields exactly :func:`measure_limit` (D adds ONE page of tolerance to the header X).
    Never below 1 (a zero/negative header X reads as 'absent' at D -- then the static gate stands)."""
    try:
        tol = max(1, int(page_size or 1))
    except (TypeError, ValueError):
        tol = 1
    return max(1, measure_limit(half_x) - tol)


# ---------------------------------------------------------------------------------------------------------------
# KV / seat freshness (plan A10)
# ---------------------------------------------------------------------------------------------------------------

def reading_age_ms(t_reading: float, now: float) -> int:
    return int(round(max(0.0, (float(now) - float(t_reading)) * 1000.0)))


def kv_fresh(cached: Optional[Tuple[float, Any]], now: float, fresh_ms: int) -> Tuple[Optional[Any], Optional[int]]:
    """``(reading, age_ms)`` of a cached ``(t_reading, reading)`` pair, or ``(None, age_ms|None)`` when there is none
    or it is older than ``fresh_ms``: an older reading counts as NOT PRESENT, never as a number (plan A10).
    Both ``t_reading`` and ``now`` are the same wall clock (``time.time()``), as the front keeps them."""
    if cached is None:
        return None, None
    t, reading = cached
    age = reading_age_ms(t, now)
    if reading is None or age > int(fresh_ms):
        return None, age
    return reading, age


def kv_refresh_due(cached: Optional[Tuple[float, Any]], now: float, refresh_ms: int, d_awake: bool,
                   fetch_pending: bool) -> bool:
    """Is a background ``/server_info`` read of D due? Only while D is awake (a sleeping D has no KV worth reading),
    never a second one while one is in flight, and only when the last reading is older than ``refresh_ms``
    (0 = the refresh is off)."""
    if refresh_ms <= 0 or not d_awake or fetch_pending:
        return False
    if cached is None:
        return True
    return reading_age_ms(cached[0], now) >= int(refresh_ms)


# ---------------------------------------------------------------------------------------------------------------
# NumberSource: the facade over the sub-sources (they are encapsulated, not rewritten)
# ---------------------------------------------------------------------------------------------------------------

def exact_of(value: Optional[int], src: str, t_mono: Optional[float] = None) -> Optional[Exact]:
    if value is None or int(value) < 0:
        return None
    return Exact(int(value), src, time.monotonic() if t_mono is None else float(t_mono))


def commit_if_current(front: Any, snap: GroupNumbers, commit: Any) -> bool:
    """The commit of a decision taken on ``snap`` (plan B1: all numbers of ONE decision come from ONE snapshot; between
    the snapshot and the commit there is no ``await``, so the atom is one asyncio loop turn). The epoch check is the net
    for the caller that did await: a phase change since the snapshot (the group flipped) makes the decision one on an
    old phase -- nothing is committed, the caller decides again on a new snapshot. Returns whether ``commit()`` ran."""
    if int(getattr(front, "epoch", 0)) != int(snap.epoch):
        return False
    commit()
    return True


class NumberSource:
    """The one place a decision gets its numbers from. S1 encapsulates the existing sub-sources of ``Front``
    (``_x_exact_price`` = D probe + store probe + front count, ``_set_reprobe_*`` = SET-REPROBE, the ARRIVAL-SEAT
    ledger and KV reading, ``shared_prefix_discount``); it adds no second measurement of its own."""

    def __init__(self, front: Any):
        self.front = front

    def arrival_numbers(self, rid: str, xx: Any, *, x_own: int, depth: int, now: Optional[float] = None) -> RequestNumbers:
        """The ``RequestNumbers`` of one arrival from ``Front._x_exact_price``'s answer ``xx`` (``None``: the
        pricing did not run). ``n_est`` is carried for display only."""
        t = time.monotonic() if now is None else now
        if xx is None:
            return RequestNumbers(rid=rid, depth=int(depth), x_own=int(x_own), unpriced_why="no_exact_price")
        unpriced = bool(getattr(xx, "unpriced", False))
        n_src = str(getattr(xx, "n_src", SRC_FRONT_IDS))
        n_exact = (not unpriced) or n_src == SRC_FRONT_IDS
        n_prompt = exact_of(int(xx.n), SRC_FRONT_IDS if unpriced else SRC_D_PROBE, t) if n_exact else None
        u_p = exact_of(getattr(xx, "uncached_p", -1) if unpriced else None, SRC_P_STORE, t)
        u_d = None if unpriced else exact_of(int(xx.pending), SRC_D_PROBE if getattr(xx, "probe", None) is not None
                                             else str(getattr(xx, "src", SRC_NONE)) or SRC_NONE, t)
        return RequestNumbers(
            rid=rid, n_prompt=n_prompt, u_d=u_d, u_p=u_p, u_own=u_d, depth=int(depth), x_own=int(x_own),
            image_end=int(getattr(xx, "image_end", 0) or 0),
            image_in_tail=bool(getattr(xx, "image_in_tail", False)),
            n_est=int(getattr(xx, "n", 0) or 0) if not n_exact else 0,
            unpriced_why="d_probe_unpriced" if unpriced else None)

    def group_numbers(self, *, now_wall: Optional[float] = None) -> GroupNumbers:
        """The group's numbers at ONE instant (no await between the reads): seats, decode state, the D-KV reading
        (fresh or absent, A10), the R6 rest."""
        f = self.front
        now_w = time.time() if now_wall is None else now_wall
        now_m = time.monotonic()
        try:
            seats = exact_of(f._set_seats_free(), SRC_D_LEDGER, now_m)
        except Exception:  # noqa: BLE001
            seats = None
        kv = None
        try:
            from flliper.srt.environ import envs
            fresh_ms = int(envs.FLLIPER_PDFLIP_KV_FRESH_MS.get())
            cached = (f.__dict__.get("_asr_state") or {}).get("kv")
            reading, age = kv_fresh(cached, now_w, fresh_ms)
            if reading is not None:
                from flliper.srt.pdflip import arrival_seat_rule as _asr
                free = _asr.kv_free(reading)
                if free is not None:
                    kv = Exact(int(free), SRC_D_LEDGER, now_m - (age or 0) / 1000.0)
        except Exception:  # noqa: BLE001
            kv = None
        try:
            decoding = bool(f._dc_decoding())
        except Exception:  # noqa: BLE001
            decoding = False
        return GroupNumbers(t_mono=now_m, epoch=int(getattr(f, "epoch", 0)), awake=str(getattr(f, "awake", "")),
                            state=str(getattr(f, "state", "")), d_decoding=decoding, seats_free=seats, kv_free=kv)
