"""The power limit as a calibration field (27B line, release table row 27, 26.09.).

User 24.09. ~18:55Z (verbatim): "die geschwindigkeiten der karten ist noch im
powerlimit bei 400 und 230 deswegen muss der schnitt aufjedenfall anpassbar
sein, da ich spaeter das powerlimit ggf. erhoehen werde". Measured the same
minute and again 26.09. (nvidia-smi power.limit): 5090 400 W (max 600),
3080 230 W (max 320) each. The user also said the finding "more layers on the
5090 are slower under NVFP4" held only UNDER that limit.

WHAT THIS MODULE DOES. Every compute figure a cut or a chunk plan is computed
from (#PGAP gpu_fwd, the P-chunk points, the layer-type model, the D speed
constants) was measured under SOME limit. From row 27 on:

  * every rank prints ONE ``POWER-LIMIT`` line at boot (:func:`boot_line`,
    NVML ``nvmlDeviceGetEnforcedPowerLimit`` + SM clock max); logindex parses
    it into its ``counters`` table (family ``power_limit``);
  * a model JSON carries ``power_limit_w`` per card -- the limit it was
    CALIBRATED under -- and the consumer compares it with the limit the cards
    run under NOW (:func:`check`). More than :data:`TOLERANCE` (5 %) apart =
    a loud warning; a JSON without the field = a warning, never a refusal;
  * optionally (an explicit flag, default off) the COMPUTE terms are rescaled
    by :func:`compute_factor`. THIS IS A MODEL, NOT A MEASUREMENT: every line
    that reports a scaled figure says so (:data:`MODEL_LABEL`). The only
    honest answer to a changed limit is a new measurement.

THE SCALING LAW (stated, not measured on this rig):
  ``T_now = T_cal * (P_cal / P_now) ** alpha``, applied only to compute terms.
  ``linear``: alpha = 1 (time inversely proportional to the power limit --
  the pessimistic/optimistic bound, whichever way the limit moved);
  ``power``: alpha = :data:`DEFAULT_EXPONENT` (1/3) unless overridden: under
  a power cap the SM clock follows the DVFS curve, dynamic power ~ f * V^2
  with V roughly ~ f near the top of the curve, so f ~ P^(1/3), and a
  compute-bound kernel's time ~ 1/f. Memory bandwidth is NOT scaled (the
  memory clock does not follow the board power cap in the same way), nor is
  host launch time (eager floors).

WHICH TERMS ARE COMPUTE. P chunk model (``p_chunk_policy.StageModel``): the
per-width base points and the attention coefficients at widths >=
:data:`COMPUTE_BOUND_MIN_TOKENS`; below that width a forward is weight-
bandwidth bound (the 16-token tiny bucket, launcher ``--p-prefill-graph-tiny``)
and stays unscaled; ``eager_floor_ms`` is host launch and stays unscaled.
Layer-type model (``p_stage_model.CardCost``): ``gemm_ms``, ``attn``,
``first_fixed_ms``, ``last_fixed_ms`` at the same widths; eager floors not.
D speed constants (``d_reshard.DCalib``): NOT scaled at all -- decode is
bandwidth bound (``e_gbs``); the D lines only NAME their limit.

PURE: stdlib plus ``p_chunk_policy`` (itself stdlib-only). NVML is reached
only through :func:`read_current` (lazy import of ``registry.nvml``), which
tests replace.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The limits in force on this rig since 24.09. (user order; nvidia-smi
#: power.limit read 24.09. 18:55Z and 26.09. at row 27). Keyed by card class
#: (:func:`card_class`). This is the calibration limit of every model fitted
#: on this rig before row 27 wrote the limit into the boot log: boots between
#: 24.09. and row 27 did not RECORD their limit; the two readings bracket them
#: and no change was ordered in between.
RIG_POWER_LIMIT_W_0924: Dict[str, float] = {"RTX5090": 400.0, "RTX3080": 230.0}
RIG_POWER_LIMIT_PROVENANCE = (
    "nvidia-smi power.limit 24.09. 18:55Z and 26.09. (row 27): 5090 400 W, 3080 230 W; "
    "boots in between did not record the limit")

#: Relative deviation above which a model counts as calibrated under a
#: FOREIGN limit (release table row 27).
TOLERANCE = 0.05

SCALE_OFF = "off"
SCALE_LINEAR = "linear"
SCALE_POWER = "power"
SCALE_LAWS = (SCALE_OFF, SCALE_LINEAR, SCALE_POWER)
#: alpha of the ``power`` law (f ~ P^(1/3), see the module docstring).
DEFAULT_EXPONENT = 1.0 / 3.0

#: Widths below this are weight-bandwidth bound and never scaled.
COMPUTE_BOUND_MIN_TOKENS = 256

LOG_TAG = "POWER-LIMIT"
MODEL_LABEL = "MODELL (Skalierungsgesetz, keine Messung)"
INSTRUMENT = "nvmlDeviceGetEnforcedPowerLimit"


class PowerLimitError(ValueError):
    """An invalid scaling law, exponent or power_limit_w field."""


# ---------------------------------------------------------------------------
# identity and the boot line


def card_class(name: str) -> str:
    """'NVIDIA GeForce RTX 5090' -> 'RTX5090' (the key the model JSONs use)."""
    s = str(name or "")
    m = re.search(r"(RTX|GTX|TITAN|A|H|L|B)\s*(\d{3,4}\w*)", s)
    if m:
        return (m.group(1) + m.group(2)).replace(" ", "")
    return re.sub(r"\s+", "", s.replace("NVIDIA", "").replace("GeForce", "")) or "unknown"


def _num(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def boot_line(fields: Mapping[str, object], info=None, reason: str = "") -> str:
    """The ONE ``POWER-LIMIT`` line a rank (or the launcher, per card) prints.

    ``fields``: the where-part (``tp_rank``, ``pp_rank``, ``gpu_id`` ...),
    printed first as ``k=v``. ``info``: a ``registry.nvml.PowerInfo`` or None
    (then ``reason`` says why: the line is printed anyway so a boot without an
    NVML answer is visible as such, not as a boot that predates row 27).
    Every numeric field is ``key=<number>`` so logindex harvests it; an
    unanswered query prints ``NA`` (never 0)."""
    where = " ".join(f"{k}={v}" for k, v in fields.items())
    if info is None:
        return f"{LOG_TAG} rank {where} power_limit_w=NA unavailable reason={reason or 'unknown'}".rstrip()
    return (f"{LOG_TAG} rank {where} nvml_index={int(info.index)} card={card_class(info.name)} "
            f"power_limit_w={_num(info.power_limit_w)} "
            f"power_limit_default_w={_num(info.power_limit_default_w)} "
            f"power_limit_max_w={_num(info.power_limit_max_w)} "
            f"sm_clock_max_mhz={_num(info.sm_clock_max_mhz)} "
            f"uuid={info.uuid} instrument={INSTRUMENT}")


_LINE_RE = re.compile(r"POWER-LIMIT rank (.*)$")
_KV_RE = re.compile(r"\b([a-z_]+)=(\S+)")
_PREFIX_RE = re.compile(r"^\[[^\]]*? ((?:PP|TP|DP)\d+)\]")


def parse_boot_lines(lines: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """``{rank label: fields}`` from ``POWER-LIMIT rank`` lines (last wins).

    The label is the log prefix tag (``PP0``/``TP1``) when present, else
    ``tp<tp_rank>pp<pp_rank>`` from the fields.
    Numeric fields come back as float, ``NA`` as None."""
    out: Dict[str, Dict[str, object]] = {}
    for ln in lines:
        if "POWER-LIMIT rank " not in ln:
            continue
        m = _LINE_RE.search(ln)
        if not m:
            continue
        f: Dict[str, object] = {}
        for k, v in _KV_RE.findall(m.group(1)):
            if v == "NA":
                f[k] = None
                continue
            try:
                f[k] = float(v)
            except ValueError:
                f[k] = v
        pm = _PREFIX_RE.search(ln)
        if pm:
            label = pm.group(1)
        else:
            tp, pp = f.get("tp_rank"), f.get("pp_rank")
            label = (f"tp{int(tp) if isinstance(tp, float) else '?'}"
                     f"pp{int(pp) if isinstance(pp, float) else '?'}")
        out[str(label)] = f
    return out


# ---------------------------------------------------------------------------
# the check and the law


@dataclasses.dataclass(frozen=True)
class Verdict:
    label: str
    calibrated_w: Optional[float]
    current_w: Optional[float]
    #: ``(current - calibrated) / calibrated``; None when either side is unknown.
    deviation: Optional[float]
    #: The compute-time factor applied (1.0 = unscaled).
    factor: float

    @property
    def mismatch(self) -> bool:
        return self.deviation is not None and abs(self.deviation) > TOLERANCE


def exponent_of(law: str, exponent: Optional[float] = None) -> float:
    if law not in SCALE_LAWS:
        raise PowerLimitError(f"power scale law {law!r}: one of {list(SCALE_LAWS)}")
    if law == SCALE_OFF:
        return 0.0
    if law == SCALE_LINEAR:
        return 1.0
    a = DEFAULT_EXPONENT if exponent is None else float(exponent)
    if not (0.0 < a <= 2.0):
        raise PowerLimitError(f"power scale exponent {a}: expected 0 < alpha <= 2")
    return a


def compute_factor(calibrated_w: Optional[float], current_w: Optional[float], law: str,
                   exponent: Optional[float] = None) -> float:
    """``(P_cal / P_now) ** alpha`` -- the factor on a compute TIME. 1.0 under
    ``off`` or when either limit is unknown. A MODEL (module docstring)."""
    a = exponent_of(law, exponent)
    if a == 0.0 or not calibrated_w or not current_w:
        return 1.0
    if float(calibrated_w) <= 0 or float(current_w) <= 0:
        raise PowerLimitError(f"power limits must be > 0 W, got {calibrated_w} / {current_w}")
    return (float(calibrated_w) / float(current_w)) ** a


def check(labels: Sequence[str], calibrated: Sequence[Optional[float]],
          current: Sequence[Optional[float]], law: str = SCALE_OFF,
          exponent: Optional[float] = None) -> List[Verdict]:
    """One :class:`Verdict` per label. The factor is != 1 ONLY on a mismatch
    (> :data:`TOLERANCE`) under an explicit law: inside the tolerance the
    model is used as calibrated."""
    exponent_of(law, exponent)
    out = []
    for i, lab in enumerate(labels):
        cal = calibrated[i] if i < len(calibrated) else None
        cur = current[i] if i < len(current) else None
        dev = None
        if cal and cur:
            dev = (float(cur) - float(cal)) / float(cal)
        fac = 1.0
        if dev is not None and abs(dev) > TOLERANCE:
            fac = compute_factor(cal, cur, law, exponent)
        out.append(Verdict(str(lab), None if cal is None else float(cal),
                           None if cur is None else float(cur), dev, fac))
    return out


def verdict_lines(what: str, verdicts: Sequence[Verdict], law: str = SCALE_OFF,
                  exponent: Optional[float] = None, source: str = "") -> List[str]:
    """The lines a consumer prints for its check: one per label, loud on a
    mismatch, plus one summary line naming the law when it scaled."""
    out = []
    src = f" source={source}" if source else ""
    for v in verdicts:
        if v.calibrated_w is None:
            out.append(f"{LOG_TAG} WARNING {what} {v.label}: model carries no power_limit_w (old JSON) -- "
                       f"calibration limit UNKNOWN, running {_num(v.current_w)} W; the cut/plan cannot be "
                       f"checked against the running limit (used as is, not refused){src}")
        elif v.current_w is None:
            out.append(f"{LOG_TAG} WARNING {what} {v.label}: running limit UNKNOWN (NVML did not answer), "
                       f"model calibrated at {_num(v.calibrated_w)} W -- used as calibrated{src}")
        elif v.mismatch:
            out.append(f"{LOG_TAG} MISMATCH !!! {what} {v.label}: model calibrated at "
                       f"{_num(v.calibrated_w)} W, card runs {_num(v.current_w)} W "
                       f"({100.0 * v.deviation:+.1f} % > {100.0 * TOLERANCE:.0f} %) -- every compute figure of "
                       f"this model was measured under a FOREIGN limit; re-measure before trusting a cut"
                       + (f"; compute terms x{v.factor:.4f} ({MODEL_LABEL})" if v.factor != 1.0
                          else "; NOT rescaled (flag off)") + src)
        else:
            out.append(f"{LOG_TAG} ok {what} {v.label}: calibrated {_num(v.calibrated_w)} W, runs "
                       f"{_num(v.current_w)} W ({100.0 * (v.deviation or 0.0):+.1f} %){src}")
    if any(v.factor != 1.0 for v in verdicts):
        out.append(f"{LOG_TAG} SCALE {what}: law={law} alpha={exponent_of(law, exponent):.4f} "
                   f"T_now = T_cal * (P_cal/P_now)^alpha on compute terms only (widths >= "
                   f"{COMPUTE_BOUND_MIN_TOKENS}; bandwidth and host-launch terms unscaled) -- {MODEL_LABEL}")
    return out


# ---------------------------------------------------------------------------
# applying the factor to the P chunk model


def _scale_pts(pts, f: float):
    return tuple((int(w), (float(v) * f if int(w) >= COMPUTE_BOUND_MIN_TOKENS else float(v)))
                 for w, v in pts)


def scale_stage_model(stage, factor: float):
    """A ``p_chunk_policy.StageModel`` with its compute terms x ``factor``
    (base and attention points at widths >= :data:`COMPUTE_BOUND_MIN_TOKENS`,
    the flat attention coefficient); ``eager_floor_ms`` (host) unchanged."""
    if factor == 1.0:
        return stage
    return dataclasses.replace(
        stage,
        points=_scale_pts(stage.points, factor),
        attn_ms_per_tok_1k=float(stage.attn_ms_per_tok_1k) * factor,
        eager_points=_scale_pts(stage.eager_points, factor),
        attn_points=_scale_pts(stage.attn_points, factor),
        eager_attn_points=_scale_pts(stage.eager_attn_points, factor),
    )


# ---------------------------------------------------------------------------
# the calibration field of a model JSON


def stage_limits_from_json(data: Mapping, n_stages: int,
                           stage_cards: Optional[Sequence[str]] = None) -> Optional[List[Optional[float]]]:
    """The ``power_limit_w`` field of a ``--p-chunk-model`` JSON as a per-stage
    list, or None when the JSON has no such field (old JSON: warn, never
    refuse). Accepted shapes: a list (one value per stage); a dict keyed by
    ``PP<r>``; a dict keyed by card class together with the JSON's
    ``power_limit_cards`` (or ``stage_cards``) list or ``stage_cards``."""
    raw = data.get("power_limit_w") if isinstance(data, Mapping) else None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals = [None if v is None else float(v) for v in raw]
        if len(vals) != int(n_stages):
            raise PowerLimitError(f"power_limit_w has {len(vals)} values, the model has {n_stages} stages")
        return vals
    if isinstance(raw, Mapping):
        if all(str(k).startswith("PP") for k in raw):
            return [None if raw.get(f"PP{r}") is None else float(raw[f"PP{r}"]) for r in range(int(n_stages))]
        cards = list(data.get("power_limit_cards") or data.get("stage_cards") or stage_cards or [])
        if len(cards) != int(n_stages):
            raise PowerLimitError("power_limit_w keyed by card needs power_limit_cards/stage_cards per stage")
        return [None if raw.get(c) is None else float(raw[c]) for c in cards]
    raise PowerLimitError(f"power_limit_w: list or dict expected, got {type(raw).__name__}")


def uuid_for_ordinal(ordinal: int) -> str:
    """NVML UUID of the card behind in-process CUDA ordinal ``ordinal``.

    The weg2 launcher masks every group with ``CUDA_VISIBLE_DEVICES`` =
    the ordered cards' UUIDs, so the mask itself answers (no CUDA context,
    no index read as an NVML index -- #392/#589). Any other mask shape goes
    through ``registry.nvml.current_device_uuid`` (the #331 identity map)."""
    import os

    from sglang.srt.registry import nvml as _nvml

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    toks = [t.strip() for t in raw.split(",") if t.strip()] if raw is not None else []
    if toks and all(t.startswith(("GPU-", "MIG-")) for t in toks) and 0 <= int(ordinal) < len(toks):
        return toks[int(ordinal)]
    return _nvml.current_device_uuid()


def rank_boot_line(tp_rank: int, pp_rank: int, gpu_id: int) -> str:
    """The rank's ``POWER-LIMIT`` line (model_runner, once per rank after the
    device is set). Never raises: an NVML failure prints the line with
    ``power_limit_w=NA`` and the reason."""
    fields = {"tp_rank": int(tp_rank), "pp_rank": int(pp_rank), "gpu_id": int(gpu_id)}
    try:
        from sglang.srt.registry import nvml as _nvml

        return boot_line(fields, _nvml.power_info_for_uuid(uuid_for_ordinal(gpu_id)))
    except Exception as exc:  # noqa: BLE001 - an instrument never kills a rank
        return boot_line(fields, None, f"{type(exc).__name__}:{str(exc)[:120]}".replace(" ", "_"))


def read_current() -> Dict[str, Tuple[float, str]]:
    """``{uuid: (power_limit_w, card class)}`` of every card from NVML. The
    only I/O here; tests replace it. Raises when NVML cannot answer."""
    from sglang.srt.registry import nvml as _nvml

    return {p.uuid: (p.power_limit_w, card_class(p.name)) for p in _nvml.power_snapshot()}


__all__ = [
    "RIG_POWER_LIMIT_W_0924", "RIG_POWER_LIMIT_PROVENANCE", "TOLERANCE", "SCALE_OFF", "SCALE_LINEAR",
    "SCALE_POWER", "SCALE_LAWS", "DEFAULT_EXPONENT", "COMPUTE_BOUND_MIN_TOKENS", "LOG_TAG",
    "MODEL_LABEL", "INSTRUMENT", "PowerLimitError", "Verdict", "card_class", "boot_line",
    "parse_boot_lines", "exponent_of", "compute_factor", "check", "verdict_lines",
    "scale_stage_model", "stage_limits_from_json", "read_current",
]
