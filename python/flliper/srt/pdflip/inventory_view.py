"""HW-P1c 1003 (PROFILE-VECTORS / RECORDS-NVEC / L15-POSTS / PP-CUT-PIN): the
profile's POSITIONAL measured values for the LIVE inventory, derived instead of
refused -- where the live cards are a subset of the cards the values were
measured on.

User order 03.10. ~19:30Z: "unsere software muss mit beliebiger anzahl an
karten ... laufen"; this round: "vorhandene 3-Vektoren gelten nur, wenn
Inventar und Record-Karten uebereinstimmen (sonst ableiten statt verweigern)".

THE RULE (one, stated once):

* **Subset.** A live inventory is derivable from a calibrated one when every
  live card has its OWN measured twin of the same calibration class
  (``card_identity.class_label``): the live class multiset is contained in the
  calibrated one. 5090 + 3080 and 3080 + 3080 are subsets of
  [RTX5090, RTX3080, RTX3080]; 2x5090, 3x3080, 4090 or any 4+ card inventory
  are not -- they stay HW-UNCALIBRATED (a calibration boot writes their
  records; plan K1/P3a).
* **Per record / per flag, ONE policy** (:data:`RECORD_POLICY`,
  :data:`FLAG_POLICY`), because "which entry belongs to which live card"
  depends on what the number IS:

  - ``class-max``  a per-card booking of the card's class (rest, overshoot,
    user reserve, cost per layer): each live card takes the MAXIMUM over the
    calibrated cards of its class -- the conservative direction (books more,
    prices slower); never an average, never another class's number;
  - ``role``       a per-PP-stage figure that depends on the stage's role (the
    first stage holds the embedding, the last the head, any other neither):
    first, last, and the middle one repeated;
  - ``uniform``    all entries equal: repeated;
  - ``consumer``   the consumer already indexes by card class
    (``DC_MEASURED_D_XCHG_MIB``: NVML-order triple read through
    ``DC_MEASURED_D_BY_CLASS``): nothing to derive;
  - ``cut-gated``  / ``advisory``  the record holds only for its own P cut
    (``profile_records.select`` refuses it otherwise) or is an advisory line
    (``d_reshard``): nothing to derive;
  - ``cut-pin``    a pinned P cut (``--pp-stage-ratio`` ...): holds only for its
    own stage count; for another N it is DROPPED and the planner's cut solver
    (``planner/pp_cut.py``) derives the cut from layers and card rates;
  - no policy     NOT derivable: the value needs a calibration boot of this
    inventory (HW-UNCALIBRATED names it).

Every derived value carries its provenance in the launcher's ``HW-DERIVE``
line (``derived``, never ``measured``).

PURE: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

CLASS_MAX = "class-max"
CLASS_MIN = "class-min"
ROLE = "role"
UNIFORM = "uniform"
CONSUMER = "consumer"
CUT_GATED = "cut-gated"
ADVISORY = "advisory"
CUT_PIN = "cut-pin"

#: Record name -> policy (qwen27b.json; the NF records are NOT listed: NF is
#: tested on all three cards only, a subset of it stays HW-UNCALIBRATED).
RECORD_POLICY: Dict[str, str] = {
    # NVML-order triple, read by calibration class through DC_MEASURED_D_BY_CLASS
    "DC_MEASURED_D_XCHG_MIB": CONSUMER,
    "P_OVERSHOOT_MIB": CLASS_MAX,
    "D_OVERSHOOT_MIB": CLASS_MAX,
    "D_AWAKE_REST_BOOKED_MIB": CLASS_MAX,
    "D_TORCH_CAP_OTHER_MIB": CLASS_MAX,
    "D_AWAKE_REST_CAPPED_MIB": CLASS_MAX,
    "D_DUAL_ALLOC_CACHE_BOOK_MIB": CLASS_MAX,
    "D_EXTEND_CAP_PER_ROW_MIB": UNIFORM,
    "P_PP_STAGE_FIXED_MIB": ROLE,
    "P_PP_STAGE_FIXED_YARN2_DELTA_MIB": ROLE,
    "MEASURED_MS_PER_LAYER": CLASS_MAX,
    "P_CHUNK_BUILTIN": CUT_GATED,
    "D_SPEED_DECODE": ADVISORY,
    "D_SPEED_PREFILL": ADVISORY,
}
#: positional launcher flags / --extra-*,--env-* tokens -> policy
FLAG_POLICY: Dict[str, str] = {
    "--user-reserve-mib": CLASS_MAX,
    "FLLIPER_PDFLIP_EXTEND_TRIM_MIB": CLASS_MAX,
    "--pp-stage-ratio": CUT_PIN,
    "--pp-attn-stage-ratio": CUT_PIN,
}
#: the launcher flags (dest) that carry a record's vector as a "a,b,c" string
RECORD_FLAGS: Dict[str, str] = {
    "pp_cut_measured_ms_per_layer": "MEASURED_MS_PER_LAYER",
    "pp_cut_stage_fixed_mib": "P_PP_STAGE_FIXED_MIB",
}

Vec = Union[List, Tuple]


def class_counts(inv: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in inv:
        out[c] = out.get(c, 0) + 1
    return out


def is_subset(live: Sequence[str], calibrated: Sequence[str]) -> bool:
    """Every live card has its own measured twin: the live class multiset is
    contained in the calibrated one. An unclassified card ('-') never is."""
    have = class_counts(calibrated)
    for c, n in class_counts(live).items():
        if c in ("", "-", None) or have.get(c, 0) < n:
            return False
    return True


def _as_list(value) -> Optional[list]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [x.strip() for x in value.split(",")]
    return None


def _num(x):
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return float(x)
    return x


def _render(like, vals: list):
    """The derived vector in the SAME shape as the source (string / tuple / list);
    entries keep their source spelling."""
    if isinstance(like, str):
        return ",".join(str(v) for v in vals)
    return tuple(vals) if isinstance(like, tuple) else list(vals)


def derive_vector(policy: str, value, calibrated: Sequence[str], live: Sequence[str]):
    """The vector for ``live`` from ``value`` (one entry per ``calibrated``
    card) under ``policy``; ``None`` when it cannot be derived. ``consumer`` /
    ``cut-gated`` / ``advisory`` return the value unchanged (nothing to do);
    ``cut-pin`` returns ``()`` (dropped)."""
    if policy in (CONSUMER, CUT_GATED, ADVISORY):
        return value
    if policy == CUT_PIN:
        return ()
    vec = _as_list(value)
    if vec is None or len(vec) != len(calibrated):
        return None
    n = len(live)
    if policy == ROLE:
        if n < 2 or len(vec) < 2:
            return None
        mid = vec[1] if len(vec) > 2 else vec[-1]
        return _render(value, [vec[0]] + [mid] * (n - 2) + [vec[-1]])
    if policy == UNIFORM:
        if len({str(x) for x in vec}) != 1:
            return None
        return _render(value, [vec[0]] * n)
    if policy in (CLASS_MAX, CLASS_MIN):
        if not is_subset(live, calibrated):
            return None
        pick = max if policy == CLASS_MAX else min
        out = []
        for c in live:
            same = [vec[i] for i, k in enumerate(calibrated) if k == c and vec[i] is not None]
            if not same:
                return None
            out.append(pick(same, key=_num))
        return _render(value, out)
    return None


@dataclass(frozen=True)
class Assessment:
    """Which positional values of a profile are derivable for a live inventory."""

    subset: bool
    derivable: Tuple[str, ...]
    exempt: Tuple[str, ...]
    underivable: Tuple[str, ...]


def assess_records(records: Sequence[Tuple[str, object]], calibrated: Sequence[str],
                   live: Sequence[str]) -> Assessment:
    """``records``: ``(name, value)`` of the positional records. A record is
    derivable under its policy when the live inventory is a subset (for the
    stage-role / uniform / exempt kinds the subset condition is not needed:
    they do not depend on the card class)."""
    sub = is_subset(live, calibrated)
    ok: List[str] = []
    ex: List[str] = []
    bad: List[str] = []
    for name, value in records:
        pol = RECORD_POLICY.get(name)
        if pol is None:
            bad.append(name)
        elif pol in (CONSUMER, CUT_GATED, ADVISORY):
            ex.append(name)
        elif derive_vector(pol, value, calibrated, live) is None:
            bad.append(name)
        else:
            ok.append(name)
    return Assessment(sub, tuple(sorted(ok)), tuple(sorted(ex)), tuple(sorted(bad)))


def vector_derivable(flag: str, count: int, calibrated: Sequence[str],
                     live: Sequence[str]) -> bool:
    """Can the profile vector ``flag`` (``count`` entries, written for
    ``calibrated``) be derived for ``live``? Cut pins always (dropped, solved)."""
    pol = FLAG_POLICY.get(flag)
    if pol is None:
        return False
    if pol == CUT_PIN:
        return True
    if int(count) != len(calibrated) or not calibrated:
        return False
    return is_subset(live, calibrated) if pol in (CLASS_MAX, CLASS_MIN) else True


# ---------------------------------------------------------------------------
# the active view: profile_constant reads it (pdflip/form.py)

_ACTIVE: Optional[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = None


def set_active(profile: str, calibrated: Sequence[str], live: Sequence[str]) -> None:
    """Install the view of this launch (``live`` differs from ``calibrated``)."""
    global _ACTIVE
    _ACTIVE = (str(profile), tuple(calibrated), tuple(live))


def clear_active() -> None:
    global _ACTIVE
    _ACTIVE = None


def active() -> Optional[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
    return _ACTIVE


def apply_active(name: str, profile: str, value):
    """``value`` of the profile constant ``name``, derived for the active live
    inventory when a view is installed for ``profile`` and the record is a
    positional one with a derivation; else ``value`` unchanged."""
    if _ACTIVE is None or _ACTIVE[0] != str(profile):
        return value
    _, calibrated, live = _ACTIVE
    if tuple(calibrated) == tuple(live):
        return value
    pol = RECORD_POLICY.get(name)
    if pol is None:
        return value
    got = derive_vector(pol, value, calibrated, live)
    return value if got is None else got


# ---------------------------------------------------------------------------
# L15: the ordinal override


def derive_l15_override(value: Optional[str], calibrated: Sequence[str],
                        live: Sequence[str]) -> Tuple[Optional[str], str]:
    """``FLLIPER_PDFLIP_L15_MIB`` ``c<i>=<mib>,...`` (ordinals of ``calibrated``)
    for ``live``: each live ordinal takes the MINIMUM post over the calibrated
    cards of its class (a card the value does not name holds 0; a hold larger
    than the room P leaves is the one direction that is wrong, so the smallest
    measured hold of the class). A live ordinal whose class holds 0 anywhere is
    left unnamed (held 0). Returns ``(new value, note)``; unchanged text when
    ``live`` is the calibrated inventory, an identity-keyed or auto value; the
    original with an empty note when not derivable."""
    text = str(value or "").strip()
    if not text or text.lower() == "auto" or tuple(live) == tuple(calibrated):
        return value, ""
    pairs = []
    for part in text.split(","):
        k, sep, v = part.strip().partition("=")
        if not sep or not k.startswith("c") or not k[1:].isdigit() or not v.strip().isdigit():
            return value, ""   # identity keys / malformed: resolved (or refused) by the launcher
        pairs.append((int(k[1:]), int(v)))
    named = dict(pairs)
    if not is_subset(live, calibrated) or any(i >= len(calibrated) for i in named):
        return value, ""
    out = []
    for j, c in enumerate(live):
        posts = [named.get(i, 0) for i, k in enumerate(calibrated) if k == c]
        mib = min(posts) if posts else 0
        if mib > 0:
            out.append(f"c{j}={mib}")
    new = ",".join(out)
    return new, (f"L15 posts derived by class (smallest measured hold of the class) from '{text}' on "
                 f"[{', '.join(calibrated)}] for [{', '.join(live)}]: '{new}'")


def l15_derivable(value: Optional[str], calibrated: Sequence[str], live: Sequence[str]) -> bool:
    """True when every ordinal key of ``value`` can be placed on ``live``
    (derivation succeeds) -- the L15-POSTS probe."""
    new, note = derive_l15_override(value, calibrated, live)
    return bool(note) and bool(new)
