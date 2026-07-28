"""One limiting factor, one measurement — and one composed suggestion.

WHY THE FACTORS ARE SEPARATE
----------------------------
"Is this rig slow?" is not answerable. "Which of these six things is holding
this rig back?" is, but only if each of the six is measured on its own. A
combined benchmark that moves by 8 % says nothing about which term moved; six
short measurements, read side by side, say exactly that. So this module
exposes the limits as SEPARATE tiles, each with its own measurement, its own
provenance and its own re-measure action.

The measurements themselves are not new. Every one of them already exists as
a study elsewhere in the tree -- the card probe, the power calibration, the
engine's phase-labelled forward timer, the HiCache accumulator, the MLP-split
crossover, the state/KV balance point. What was missing was a surface that
reads them one at a time and says, per factor, whether a number exists at all.

NOTHING IS INVENTED
-------------------
A factor with no study on disk reports ``provenance = "absent"`` and names the
study that would produce it. It never reports a zero, a nameplate figure, or a
number carried over from another rig. That is the whole reason this file
exists next to the surfaces it feeds: a placeholder is indistinguishable from
a measurement once it has been rendered.

Three provenances, and they are not interchangeable:

``measured``
    A study ran ON THIS RIG and its result is on disk (or was just read off a
    running engine). Carries the timestamp it was taken at.
``estimate``
    Planner arithmetic over measured inputs -- the capacity model, the
    state/KV balance. Correct as arithmetic, still not a measurement.
``absent``
    No study has been run. Carries the reason and the action, never a value.

THE SUGGESTION
--------------
:func:`suggest_scenario` composes what is already computed -- the lever
profiles, the card probe basis, the state/KV balance -- into ONE concrete
proposal: which working point, which flags, which expected figures, and where
each figure came from. It computes no new number and it boots nothing. The
selection rules are conservative by construction: when the evidence for a
directed split is missing, the answer is the baseline, stated as such, rather
than a guess dressed up as a recommendation.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "MEASURED",
    "ESTIMATE",
    "ABSENT",
    "Remeasure",
    "FactorSpec",
    "FACTORS",
    "FACTOR_KEYS",
    "PREFILL_RATIO_THRESHOLD",
    "DECODE_RATIO_THRESHOLD",
    "read_factors",
    "suggest_scenario",
]


#: A study ran on this rig and its result is on disk.
MEASURED = "measured"
#: Planner arithmetic over measured inputs. Not a measurement.
ESTIMATE = "estimate"
#: No study has been run. Never accompanied by a value.
ABSENT = "absent"


@dataclasses.dataclass(frozen=True)
class Remeasure:
    """How to run THIS factor's measurement again, expressed as data.

    The dashboard renders a control from it and ``curl`` runs the same thing;
    neither side holds a list of endpoints of its own, so the two cannot
    drift. ``kind`` decides the shape:

    ``job``
        POST ``path``, then poll ``status_path`` until the job leaves
        ``running``. The request returns immediately -- a measurement that
        held an HTTP request open for its duration would time the browser out
        and block every other panel.
    ``call``
        One POST that answers directly. Used only where the work is short and
        bounded.
    ``poll``
        Nothing to start. The figure is a delta across two reads, so "measure
        again" means "read again" and the first read only opens the window.
    ``command``
        No endpoint exists; the harness is a command line. The surface shows
        the command rather than a button that would do nothing -- a dead
        control is worse than an honest instruction.
    ``derived``
        Not measured at all: recomputed from the current plan whenever the
        configuration changes. There is nothing to trigger.
    """

    kind: str
    label: str
    path: str = ""
    status_path: str = ""
    body: Dict[str, Any] = dataclasses.field(default_factory=dict)
    command: str = ""
    #: What running it costs, in the reader's terms. Empty when it costs
    #: nothing (a read, a recomputation).
    cost: str = ""

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        # Filled per reading, because whether an action can run at all is a
        # fact about the current state, not about the factor. A control that
        # cannot work is disabled WITH its reason rather than left to fail.
        d.setdefault("ready", True)
        d.setdefault("blocked_reason", "")
        return d


@dataclasses.dataclass(frozen=True)
class FactorSpec:
    """One limiting factor: what it limits, and how it is measured."""

    key: str
    label: str
    #: The question this factor answers on its own. Written as the limit it
    #: sets, not as the instrument that reads it.
    question: str
    #: Key into ``planner/tooltips.py``. The gives/costs sentence lives there
    #: with every other control's, so it is written once.
    tooltip_key: str
    remeasure: Remeasure


#: Ordered from the hardware outward: what the cards can do, what the link
#: between them can do, what the running engine actually spends, and what the
#: configuration then costs. A reader working down the list moves from "the
#: rig cannot go faster than this" to "this configuration chose this".
FACTORS: Tuple[FactorSpec, ...] = (
    FactorSpec(
        key="card_rates",
        label="card rates",
        question=(
            "What each card can actually do: streaming bandwidth, the "
            "decode-shaped weight read, and the bf16 / fp8 GEMM rate. This is "
            "the ceiling every per-rank figure is measured against."
        ),
        tooltip_key="factor.card_rates",
        remeasure=Remeasure(
            kind="job",
            label="run the card probe",
            path="/api/card_probe",
            status_path="/api/card_probe/status",
            body={"node_id": "local"},
            cost=(
                "about 30 s of GPU time, allocating on every card while it "
                "runs; a live server loses that memory for the duration"
            ),
        ),
    ),
    FactorSpec(
        key="pair_link",
        label="pair links",
        question=(
            "The narrowest ordered card-to-card link. Every collective waits "
            "on the slow direction of the slowest pair, so this is the floor "
            "under ms per verify round, not a side figure."
        ),
        tooltip_key="factor.pair_link",
        remeasure=Remeasure(
            kind="job",
            label="run the card probe",
            path="/api/card_probe",
            status_path="/api/card_probe/status",
            body={"node_id": "local"},
            cost="the same probe run as the card rates; one run fills both",
        ),
    ),
    FactorSpec(
        key="round_time",
        label="ms per round",
        question=(
            "How long a round actually takes on the running server: the "
            "verify round, the decode round, and prefill per 1000 prompt "
            "tokens. tok/s hides which phase moved; these do not."
        ),
        tooltip_key="factor.round_time",
        remeasure=Remeasure(
            kind="poll",
            label="read the engine again",
            path="/api/bench_factors",
            cost="",
        ),
    ),
    FactorSpec(
        key="rank_balance",
        label="rank balance",
        question=(
            "Per rank, how much of a round is compute and how much is waiting "
            "on the collective. The waiting half is what a different split "
            "could recover; the pacing rank is the one everybody waits on."
        ),
        tooltip_key="factor.rank_balance",
        remeasure=Remeasure(
            kind="poll",
            label="read the engine again",
            path="/api/bench_factors",
            cost="",
        ),
    ),
    FactorSpec(
        key="power",
        label="power ceiling",
        question=(
            "Per card, the idle draw and the draw under a bandwidth-bound and "
            "a compute-bound load. It sets what a J-per-token figure may be "
            "compared against, and which card runs into its limit first."
        ),
        tooltip_key="factor.power",
        remeasure=Remeasure(
            kind="call",
            label="calibrate card power",
            path="/api/measure_power",
            cost=(
                "short micro-benchmarks per card; cards busy with another "
                "process are skipped rather than contended"
            ),
        ),
    ),
    FactorSpec(
        key="prefix_cache",
        label="prefix cache",
        question=(
            "Prefill tokens the host-side cache tiers served instead of "
            "recomputing. It is the only lever here that removes work rather "
            "than moving it."
        ),
        tooltip_key="factor.prefix_cache",
        remeasure=Remeasure(
            kind="call",
            label="record from the running server",
            path="/api/hicache_saved",
            body={},
            cost="one scrape of the target's /metrics; nothing is booted",
        ),
    ),
    FactorSpec(
        key="mlp_split",
        label="MLP split crossover",
        question=(
            "What moving dense-MLP mass toward the compute-strong rank buys "
            "in prefill and costs in decode. It is the only measured source "
            "for the move between two splits."
        ),
        tooltip_key="factor.mlp_split",
        remeasure=Remeasure(
            kind="command",
            label="run the crossover sweep",
            command="python -m sglang.planner --crossover --run",
            cost=(
                "a boot per split vector under load; the longest study on " "this list"
            ),
        ),
    ),
    FactorSpec(
        key="concurrency_balance",
        label="state / KV balance",
        question=(
            "Where the recurrent state pool stops paying for itself: the "
            "concurrency at which this rig carries the most sessions at a "
            "given context."
        ),
        tooltip_key="factor.concurrency_balance",
        remeasure=Remeasure(
            kind="derived",
            label="recomputed with the plan",
            cost="",
        ),
    ),
)

FACTOR_KEYS: Tuple[str, ...] = tuple(f.key for f in FACTORS)


# ---------------------------------------------------------------------------
# Reading a factor
# ---------------------------------------------------------------------------


def _val(key: str, label: str, value, unit: str = "", note: str = "") -> dict:
    return {"key": key, "label": label, "value": value, "unit": unit, "note": note}


def _absent(reason: str, notes: Optional[List[str]] = None) -> dict:
    """The shape a factor with no study takes. Deliberately carries no
    ``values`` list at all rather than an empty one that could be rendered as
    a row of dashes and read as "measured, and it was nothing"."""
    return {
        "available": False,
        "provenance": ABSENT,
        "source": "",
        "measured_at": None,
        "age_s": None,
        "missing_reason": reason,
        "notes": list(notes or []),
        "values": [],
    }


def _present(
    provenance: str,
    source: str,
    values: List[dict],
    *,
    measured_at: Optional[float] = None,
    notes: Optional[List[str]] = None,
) -> dict:
    now = time.time()
    return {
        "available": True,
        "provenance": provenance,
        "source": source,
        "measured_at": measured_at,
        "age_s": (None if not measured_at else max(0.0, now - float(measured_at))),
        "missing_reason": "",
        "notes": list(notes or []),
        "values": values,
    }


def _card_probe(path: Optional[str] = None):
    """The cached probe, or None. Never triggers a measurement."""
    try:
        from sglang.srt.rigmon.card_probe import load_card_probe

        return load_card_probe(path)
    except Exception:  # pragma: no cover - a missing probe is the normal state
        return None


_NO_PROBE = (
    "No card probe is cached for the cards visible now, so the rates are "
    "unknown rather than estimated. Everything ranking cards falls back to "
    "nameplate specs until this study runs."
)


def _read_card_rates(ctx: dict) -> dict:
    profile = ctx.get("card_probe")
    if profile is None:
        return _absent(_NO_PROBE)
    values: List[dict] = []
    for c in profile.cards:
        parts = []
        if c.membw_gbs:
            parts.append(f"{c.membw_gbs:.0f} GB/s stream")
        if c.membw_gemv_gbs:
            parts.append(f"{c.membw_gemv_gbs:.0f} GB/s decode read")
        if c.gemm_bf16_tflops:
            parts.append(f"{c.gemm_bf16_tflops:.0f} TFLOP/s bf16")
        if c.gemm_fp8_tflops:
            parts.append(f"{c.gemm_fp8_tflops:.0f} TFLOP/s fp8")
        elif c.fp8_note:
            parts.append("no fp8 rate")
        values.append(
            _val(
                f"card{c.cuda_index}",
                f"{c.name} #{c.cuda_index}",
                " / ".join(parts) if parts else None,
                note=c.fp8_note or "",
            )
        )
    return _present(
        MEASURED,
        f"card probe of {len(profile.cards)} cards, driver {profile.driver or '?'}",
        values,
        measured_at=profile.created or None,
        notes=profile.rate_caveats() + _stale_note(profile),
    )


def _stale_note(profile) -> List[str]:
    try:
        if profile.is_stale():
            age = (profile.age_s() or 0.0) / 3600.0
            return [
                f"This probe is {age:.0f} h old. Clocks and thermal state "
                "drift; re-probe before deriving a configuration from it."
            ]
    except Exception:  # pragma: no cover - defensive
        pass
    return []


def _read_pair_link(ctx: dict) -> dict:
    profile = ctx.get("card_probe")
    if profile is None:
        return _absent(_NO_PROBE)
    pairs = [p for p in profile.pairs if p.bandwidth_gbs]
    if not pairs:
        return _absent(
            "The cached probe measured no pair bandwidth, so the collective "
            "floor is unknown. A single-card rig has no pair to measure; on a "
            "multi-card rig re-run the probe."
        )
    slowest = min(pairs, key=lambda p: p.bandwidth_gbs)
    by_uuid = {c.uuid: c for c in profile.cards}

    def _nm(u):
        c = by_uuid.get(u)
        return f"{c.name} #{c.cuda_index}" if c else u[:12]

    values = [
        _val(
            "narrowest",
            "narrowest ordered link",
            round(float(slowest.bandwidth_gbs), 2),
            "GB/s",
            note=f"{_nm(slowest.src_uuid)} -> {_nm(slowest.dst_uuid)}",
        ),
        _val("pairs", "ordered pairs measured", len(pairs), ""),
        _val(
            "transport",
            "transport",
            ", ".join(profile.transports) or "unknown",
            "",
        ),
    ]
    if slowest.latency_us:
        values.append(
            _val(
                "latency",
                "latency on that link",
                round(float(slowest.latency_us), 1),
                "us",
            )
        )
    return _present(
        MEASURED,
        "ordered pair matrix from the card probe",
        values,
        measured_at=profile.created or None,
        notes=profile.rate_caveats(),
    )


# ---------------------------------------------------------------------------
# The engine window: two factors, one scrape
# ---------------------------------------------------------------------------

#: Last engine scrape per endpoint. The round times and the per-rank split are
#: both deltas across a window, and the window is the gap between two calls to
#: this module -- sampling twice with a sleep in between would block the
#: request for the length of the window while the caller is already polling.
#: Kept module-side and never persisted, exactly like ``webui._LEAD_PREV``.
_ENGINE_PREV: Dict[str, Tuple[float, Any]] = {}

_FIRST_SAMPLE = (
    "First read of this endpoint: the round times are a delta across a "
    "window, so they appear on the next read. Nothing is missing."
)

_NO_TIMER = (
    "The engine exported no phase-labelled forward time in this window. Boot "
    "the server with SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 and "
    "--enable-metrics-for-all-schedulers for the per-rank split. The figures "
    "are absent, not zero."
)


def _engine_window(endpoint: str) -> dict:
    """One scrape, differenced against the previous one. Feeds both the round
    time and the rank balance factor, so a reader who wants both pays for one
    scrape rather than two windows that do not line up."""
    from sglang.srt.rigmon.rates import phase_seconds, round_time
    from sglang.srt.rigmon.sources import EngineScraper

    if not endpoint:
        return {"ok": False, "reason": "no endpoint given"}
    scraper = EngineScraper(endpoint)
    now = time.time()
    try:
        sample = scraper.scrape()
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "reason": f"scrape of {endpoint} failed: {e}"}
    if not sample.up:
        return {"ok": False, "reason": sample.reason or "engine not reachable"}

    prev = _ENGINE_PREV.get(endpoint)
    _ENGINE_PREV[endpoint] = (now, sample)
    if prev is None:
        return {"ok": False, "reason": _FIRST_SAMPLE, "first_sample": True}

    prev_t, prev_sample = prev
    dt = max(now - prev_t, 1e-6)
    delta = phase_seconds(sample.per_rank_phase, prev_sample.per_rank_phase)
    rt = round_time(sample.metrics, prev_sample.metrics, delta, dt)
    return {
        "ok": True,
        "endpoint": endpoint,
        "window_s": round(dt, 2),
        "sample": sample,
        "phase_delta": delta,
        "round_time": rt,
        "at": now,
    }


_ROUND_LABELS = (
    ("ms_per_verify_round", "ms per verify round", "ms"),
    ("ms_per_decode_round", "ms per decode round", "ms"),
    ("ms_per_1k_prefill_tokens", "ms per 1k prefill tokens", "ms"),
    ("ms_per_draft_pass", "ms per draft pass", "ms"),
    ("accept_length", "accepted tokens per round", ""),
)


def _read_round_time(ctx: dict) -> dict:
    win = ctx.get("engine") or {}
    if not win.get("ok"):
        return _absent(win.get("reason") or "no running server to read")
    rt = win["round_time"]
    values = []
    for key, label, unit in _ROUND_LABELS:
        v = getattr(rt, key, None)
        if v is not None:
            values.append(_val(key, label, round(float(v), 2), unit))
    if not values:
        return _absent(_NO_TIMER if not rt.phase_seconds else _NO_TIMER)
    notes = [f"window {win['window_s']} s"]
    if rt.rounds_source and rt.rounds_source != "unavailable":
        notes.append("rounds counted: " + rt.rounds_source)
    return _present(
        MEASURED,
        f"engine forward timer at {win['endpoint']}",
        values,
        measured_at=win["at"],
        notes=notes,
    )


def _read_rank_balance(ctx: dict) -> dict:
    win = ctx.get("engine") or {}
    if not win.get("ok"):
        return _absent(win.get("reason") or "no running server to read")
    if not win.get("phase_delta"):
        return _absent(_NO_TIMER)
    from sglang.srt.rigmon.rates import rank_shares
    from sglang.srt.rigmon.sources import GpuSampler

    sample = win["sample"]
    info = sample.info or {}
    rank_gpu = None
    if isinstance(info.get("rank_gpu_id"), list):
        rank_gpu = info["rank_gpu_id"]
    elif info.get("tp_size"):
        rank_gpu = list(range(int(info["tp_size"])))
    try:
        cards = GpuSampler(counters_every=0).sample()
    except Exception:  # pragma: no cover - defensive
        cards = []
    if not cards:
        return _absent(
            "No device sampler on this host (no NVML, no nvidia-smi), so the "
            "per-rank split cannot be attached to a card. The group round "
            "time above is unaffected."
        )
    view = rank_shares(
        cards,
        None,
        rank_gpu,
        single_rank_export=(len(sample.per_rank or {}) == 1),
        phase_delta=win["phase_delta"],
        round_times=win["round_time"],
    )
    values = []
    for s in view.ranks:
        if s.compute_ms_per_round is None:
            continue
        wait = s.collective_wait_ms_per_round
        txt = f"{s.compute_ms_per_round:.2f} ms compute"
        if wait is not None:
            txt += f" / {wait:.2f} ms wait"
        values.append(
            _val(
                f"rank{s.rank if s.rank is not None else s.gpu_index}",
                f"rank {s.rank if s.rank is not None else '?'} "
                f"({s.name} #{s.gpu_index})",
                txt,
                note=s.work_source,
            )
        )
    if not values:
        return _absent(_NO_TIMER)
    notes = [c.text if hasattr(c, "text") else str(c) for c in view.caveats]
    if view.pacer_basis:
        notes.insert(0, "pacing rank: " + str(view.pacer_basis))
    return _present(
        MEASURED,
        f"engine forward timer at {win['endpoint']}, per rank",
        values,
        measured_at=win["at"],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# The persisted studies
# ---------------------------------------------------------------------------


def _read_power(ctx: dict) -> dict:
    try:
        from sglang.srt.planner import power_calibration as pc

        profile = pc.load_power_profile(
            ctx.get("power_profile_path") or pc.DEFAULT_POWER_PROFILE_PATH
        )
    except Exception as e:  # pragma: no cover - defensive
        return _absent(f"the power profile could not be read: {e}")
    if not profile:
        return _absent(
            "No per-card power calibration on this rig, so there is no idle "
            "floor and no active anchor. Every energy figure stays a "
            "TDP-shaped estimate until this study runs."
        )
    values = []
    newest = None
    for card in profile.values():
        parts = []
        if card.p_idle_w:
            parts.append(f"{card.p_idle_w:.0f} W idle")
        if card.p_membw_w:
            parts.append(f"{card.p_membw_w:.0f} W bandwidth-bound")
        if card.p_gemm_w:
            parts.append(f"{card.p_gemm_w:.0f} W compute-bound")
        values.append(_val(card.uuid, f"{card.name} ({card.arch})", " / ".join(parts)))
        if card.measured_at and (newest is None or card.measured_at > newest):
            newest = card.measured_at
    return _present(
        MEASURED,
        "per-card power calibration",
        values,
        # The stored stamp is a human string, not an epoch; it is shown as the
        # source line rather than converted into an age that would be wrong.
        notes=[f"measured {newest}"] if newest else [],
    )


def _read_prefix_cache(ctx: dict) -> dict:
    try:
        from sglang.srt.planner.hicache_savings import (
            DEFAULT_HICACHE_STORE,
            HiCacheSavingsStore,
        )

        store = HiCacheSavingsStore.load(
            ctx.get("hicache_store") or DEFAULT_HICACHE_STORE
        )
    except Exception as e:  # pragma: no cover - defensive
        return _absent(f"the prefix-cache accumulator could not be read: {e}")
    if not len(store):
        return _absent(
            "Nothing recorded yet: the prefix-cache accumulator is empty. "
            "Point it at a running server once (POST /api/hicache_saved with "
            "the model and the target) and it grows from the server's own "
            "counter from then on."
        )
    total = store.grand_total_recovered_tokens()
    values = [
        _val("recovered", "prefill tokens served from cache", round(total), "tokens")
    ]
    band = store.grand_total_saved_kwh_band()
    if band:
        values.append(
            _val(
                "kwh",
                "energy not spent",
                f"{band[0]:.3f} - {band[1]:.3f}",
                "kWh",
                note="band, from the measured J-per-prefill-token range",
            )
        )
    else:
        values.append(
            _val(
                "kwh",
                "energy not spent",
                None,
                "kWh",
                note=(
                    "no measured J-per-prefill-token band for this model yet, "
                    "so the tokens are counted but not priced"
                ),
            )
        )
    values.append(_val("records", "records", len(store), ""))
    return _present(
        MEASURED,
        "HiCache recovered-token accumulator",
        values,
        notes=[
            "Counts the host and storage tiers only: a device-side radix hit "
            "never left the GPU and costs nothing to re-serve."
        ],
    )


def _hicache_body(ctx: dict) -> dict:
    """What ``POST /api/hicache_saved`` needs, or why it cannot run.

    Recording a delta needs a model to key it under and a server to scrape.
    Both come from the surface the reader is already looking at, so the action
    carries them rather than making the browser know which factor needs what.
    """
    model, endpoint = ctx.get("model") or "", ctx.get("endpoint") or ""
    if not model or not endpoint:
        missing = []
        if not model:
            missing.append("a model to key the record under")
        if not endpoint:
            missing.append("a running server to scrape")
        return {"_blocked": "Recording needs " + " and ".join(missing) + "."}
    return {"model": model, "target": endpoint}


def _read_mlp_split(ctx: dict) -> dict:
    finding = ctx.get("crossover")
    if finding is None:
        return _absent(
            "The MLP-split crossover sweep has not been run on this rig, so "
            "the move between two splits has no measured term. The planner "
            "still ranks splits, on arithmetic rather than on a measurement."
        )
    try:
        from sglang.srt.planner import crossover as crossovermod

        measured_here = finding.provenance == crossovermod.MEASURED_HERE
        rows = [r for r in finding.break_even_table() if r.get("proposable")]
    except Exception as e:  # pragma: no cover - defensive
        return _absent(f"the crossover finding could not be read: {e}")
    if not measured_here:
        return _absent(
            "A crossover finding exists but was not measured on this rig "
            f"(provenance: {finding.provenance}). It is not used as a number "
            "here -- another rig's cards do not price these."
        )
    if not rows:
        return _absent(
            "The crossover sweep ran but produced no proposable vector on "
            "this rig: every split it tried was inside the noise floor."
        )
    best = max(rows, key=lambda r: r.get("prefill_gain_pct") or 0.0)
    return _present(
        MEASURED,
        "MLP-split crossover sweep",
        [
            _val("vector", "best proposable vector", str(best.get("vector")), ""),
            _val(
                "prefill",
                "prefill",
                round(float(best["prefill_gain_pct"]), 1),
                "%",
                note="against the base split",
            ),
            _val(
                "decode",
                "decode",
                round(float(best["decode_cost_pct"]), 1),
                "%",
                note="against the base split",
            ),
        ],
        measured_at=getattr(finding, "created", None),
    )


def _read_concurrency_balance(ctx: dict) -> dict:
    plan = ctx.get("plan") or {}
    if not plan:
        return _absent(
            "No plan to balance: pick a model and cards on the Runner tab and "
            "this recomputes with them. It is arithmetic, not a study."
        )
    bal = plan.get("mrr_balance")
    if not bal:
        return _absent(
            "This model has no recurrent state pool to trade against the KV "
            "cache, so there is no balance point. That is a property of the "
            "model, not a missing measurement."
        )
    points = bal.get("points") or []
    if not points:
        return _absent(
            "The balance could not be solved for any target context on this "
            "configuration."
        )
    values = []
    for p in points:
        values.append(
            _val(
                f"ctx{p['target_context_tokens']}",
                f"at {int(p['target_context_tokens']):,} tokens".replace(",", " "),
                f"{p['sessions']} sessions at "
                f"--max-running-requests {p['recommended_max_running_requests']}",
                note=f"bound by {p.get('binding', '?')}",
            )
        )
    if bal.get("break_even_context_tokens"):
        values.append(
            _val(
                "break_even",
                "one session's state weighs its own KV at",
                round(float(bal["break_even_context_tokens"])),
                "tokens",
            )
        )
    return _present(
        ESTIMATE,
        "planner arithmetic: the state/KV balance point for this plan",
        values,
        notes=[bal.get("note") or ""]
        + ([bal["predictor_clamp_note"]] if bal.get("predictor_clamp_note") else []),
    )


#: Factors whose action needs values only the current state can supply. The
#: body is filled HERE rather than in the browser: which factor needs which
#: field is a fact about the measurement, and the page should not have to
#: know it.
_REMEASURE_BODY = {"prefix_cache": _hicache_body}


_READERS = {
    "card_rates": _read_card_rates,
    "pair_link": _read_pair_link,
    "round_time": _read_round_time,
    "rank_balance": _read_rank_balance,
    "power": _read_power,
    "prefix_cache": _read_prefix_cache,
    "mlp_split": _read_mlp_split,
    "concurrency_balance": _read_concurrency_balance,
}


def read_factors(
    payload: Optional[dict] = None, *, plan: Optional[dict] = None
) -> dict:
    """Every limiting factor, read ONE AT A TIME, with its provenance.

    Reads only. Nothing here starts a measurement, boots a server or allocates
    on a card: a page that measured as a side effect of being drawn would make
    every refresh a surprise. The at-most-one network call is a bounded scrape
    of ``endpoint``'s ``/metrics``, which is what the round-time factor IS.

    ``plan`` is the answer ``/api/plan`` already produced for the same body;
    passing it keeps the payload-parsing machinery in one place instead of
    being repeated here.
    """
    payload = payload or {}
    endpoint = _norm_endpoint(payload.get("endpoint") or "")
    ctx = {
        "card_probe": _card_probe(payload.get("card_probe_path")),
        "crossover": _crossover(),
        "plan": plan,
        "endpoint": endpoint,
        "model": (payload.get("bench_model") or payload.get("model") or "").strip(),
        "engine": (
            _engine_window(endpoint)
            if endpoint
            else {
                "ok": False,
                "reason": (
                    "No endpoint given, so nothing was read off a running server. "
                    "The round times and the per-rank split exist only while a "
                    "server is serving."
                ),
            }
        ),
        "power_profile_path": payload.get("power_profile_path"),
        "hicache_store": payload.get("hicache_store"),
    }
    out = []
    for spec in FACTORS:
        try:
            reading = _READERS[spec.key](ctx)
        except Exception as e:  # pragma: no cover - defensive
            reading = _absent(f"this factor could not be read: {e}")
        remeasure = spec.remeasure.to_json()
        builder = _REMEASURE_BODY.get(spec.key)
        if builder is not None:
            extra = dict(builder(ctx))
            blocked = extra.pop("_blocked", "")
            if blocked:
                remeasure["ready"] = False
                remeasure["blocked_reason"] = blocked
            else:
                remeasure["body"] = {**remeasure["body"], **extra}
        out.append(
            {
                "key": spec.key,
                "label": spec.label,
                "question": spec.question,
                "tooltip_key": spec.tooltip_key,
                "remeasure": remeasure,
                **reading,
            }
        )
    measured = sum(1 for f in out if f["provenance"] == MEASURED)
    return {
        "ok": True,
        "endpoint": endpoint,
        "factors": out,
        "counts": {
            MEASURED: measured,
            ESTIMATE: sum(1 for f in out if f["provenance"] == ESTIMATE),
            ABSENT: sum(1 for f in out if f["provenance"] == ABSENT),
        },
        # Said in one line so a reader does not have to count tiles to learn
        # how much of the picture is actually measured.
        "summary": (f"{measured} of {len(out)} limiting factors measured on this rig"),
    }


def _norm_endpoint(v: str) -> str:
    v = (v or "").strip()
    if not v:
        return ""
    return v if v.startswith("http") else "http://" + v


def _crossover():
    try:
        from sglang.srt.planner.crossover import load_finding

        return load_finding()
    except Exception:  # pragma: no cover - absent is the normal state
        return None


# ---------------------------------------------------------------------------
# The suggestion
# ---------------------------------------------------------------------------

#: Prompt-to-output ratio above which the prefill working point is worth its
#: context. Deliberately high: the prefill lever is the weaker-calibrated of
#: the two terms in the cost model, so it has to clear a wide margin before it
#: is proposed rather than merely offered.
PREFILL_RATIO_THRESHOLD = 8.0

#: Ratio at or below which output dominates and the decode working point is
#: the one worth asking for.
DECODE_RATIO_THRESHOLD = 1.5


def _step(statement: str, provenance: str, basis: str = "") -> dict:
    return {"statement": statement, "provenance": provenance, "basis": basis}


def _row(profiles: List[dict], key: str) -> Optional[dict]:
    for p in profiles:
        if p.get("key") == key and p.get("resolved"):
            return p
    return None


def _metric(row: Optional[dict], key: str) -> Optional[dict]:
    for m in (row or {}).get("metrics") or []:
        if m.get("key") == key:
            return m
    return None


def _gains(row: Optional[dict], key: str) -> bool:
    """Whether this profile's metric ``key`` is a GAIN against the baseline.

    A metric that is unavailable, or equal to the baseline, is not a gain. The
    distinction matters: a stop that resolves back to the baseline must not be
    proposed as an improvement over it.
    """
    m = _metric(row, key)
    if not m or not m.get("available"):
        return False
    d = m.get("vs_baseline") or {}
    return d.get("direction") == "gain"


def suggest_scenario(
    profiles: dict,
    *,
    plan: Optional[dict] = None,
    prompt_to_output_ratio: Optional[float] = None,
    min_context_tokens: Optional[int] = None,
) -> dict:
    """One concrete proposal for this rig and this model, from what is known.

    ``profiles`` is the answer :func:`lever_profiles.build_profiles` already
    produced for the same body -- this composes it, it does not recompute it,
    so the suggestion and the working-point table can never disagree about the
    same configuration.

    The rules are stated in the answer, not hidden in it: every step lands in
    ``reasoning`` with the provenance it rests on. Where the evidence for a
    directed split is missing the answer is the BASELINE, said plainly. A
    recommendation is only worth more than the reference configuration when
    something measured says so.

    Nothing is applied and nothing is booted. The caller gets flags and a
    profile key; pressing "apply" writes them into the ordinary configuration
    state, which is a separate, explicit act.
    """
    if not profiles or not profiles.get("ok"):
        return {
            "ok": False,
            "reasons": list(
                (profiles or {}).get("reasons") or ["no plan to suggest for"]
            ),
        }

    rows = [p for p in profiles.get("profiles") or [] if p.get("resolved")]
    if not rows:
        return {"ok": False, "reasons": ["no working point resolved on this rig"]}

    baseline_key = profiles.get("baseline") or "balanced"
    baseline = _row(rows, baseline_key)
    if baseline is None:
        return {"ok": False, "reasons": ["the baseline working point did not resolve"]}

    basis = profiles.get("basis") or {}
    scores = basis.get("speed_scores")
    probe = basis.get("card_probe")
    reasoning: List[dict] = []

    # --- step 1: what the ranking rests on ---------------------------------
    if probe:
        reasoning.append(
            _step(
                f"Cards ranked on a measured probe of {probe.get('cards')} cards "
                f"and {probe.get('ordered_pairs')} ordered pairs.",
                MEASURED,
                "card probe",
            )
        )
    else:
        reasoning.append(
            _step(
                "No card probe is cached, so the cards are ranked on nameplate "
                "specs. Run POST /api/card_probe before spending context on a "
                "speed working point.",
                ABSENT,
                "card probe",
            )
        )
    reasoning.append(
        _step(
            "Narrowest link "
            f"{basis.get('min_link_gbs')} GB/s ({basis.get('min_link_source')}).",
            MEASURED if probe else ESTIMATE,
            "pair matrix",
        )
    )

    # --- step 2: pick -------------------------------------------------------
    ratio = None
    if prompt_to_output_ratio is not None:
        try:
            ratio = float(prompt_to_output_ratio)
        except (TypeError, ValueError):
            ratio = None

    chosen_key = baseline_key
    measured_speed = bool(scores and scores.get("measured"))
    distinct = int(profiles.get("distinct_working_points") or 1)

    if not measured_speed:
        reasoning.append(
            _step(
                "The speed objectives are ranked on nameplate specs rather "
                "than a probe of these cards, so no directed split is "
                "proposed: the baseline is the working point whose cost is "
                "known to be zero.",
                ABSENT,
                "card probe",
            )
        )
    elif distinct <= 1:
        reasoning.append(
            _step(
                "Every stop resolves to the same configuration on this rig, "
                "so there is nothing to choose between. That is the answer, "
                "not a gap.",
                ESTIMATE,
                "planner arithmetic",
            )
        )
    elif ratio is None:
        reasoning.append(
            _step(
                "No prompt-to-output ratio was given. Without a workload "
                "shape neither speed direction can be justified, so the "
                "baseline stands and the alternatives stay one click away.",
                ESTIMATE,
                "planner arithmetic",
            )
        )
    elif ratio >= PREFILL_RATIO_THRESHOLD:
        cand = _row(rows, "max_prefill")
        if cand and not cand.get("is_base_split") and _gains(cand, "prefill_tok_s"):
            chosen_key = "max_prefill"
            reasoning.append(
                _step(
                    f"Prompt-to-output ratio {ratio:g} is at or above "
                    f"{PREFILL_RATIO_THRESHOLD:g}, and the prefill working "
                    "point predicts a gain over the baseline on this split.",
                    ESTIMATE,
                    "cost model move, roofline level",
                )
            )
        else:
            reasoning.append(
                _step(
                    f"Prompt-to-output ratio {ratio:g} argues for prefill, but "
                    "the prefill working point resolves back to the base split "
                    "or predicts no gain here, so it would cost context for "
                    "nothing.",
                    ESTIMATE,
                    "cost model move",
                )
            )
    elif ratio <= DECODE_RATIO_THRESHOLD:
        cand = _row(rows, "max_decode")
        if cand and not cand.get("is_base_split") and _gains(cand, "decode_tok_s"):
            chosen_key = "max_decode"
            reasoning.append(
                _step(
                    f"Prompt-to-output ratio {ratio:g} is at or below "
                    f"{DECODE_RATIO_THRESHOLD:g}, and the decode working point "
                    "predicts a gain over the baseline.",
                    ESTIMATE,
                    "cost model move, roofline level",
                )
            )
        else:
            reasoning.append(
                _step(
                    "Output dominates, but the decode working point resolves "
                    "back to the base split or predicts no gain: the measured "
                    "decode optimum is flat across representable splits here.",
                    ESTIMATE,
                    "cost model move",
                )
            )
    else:
        reasoning.append(
            _step(
                f"Prompt-to-output ratio {ratio:g} sits between the two "
                "directions, so neither is worth the context it spends.",
                ESTIMATE,
                "planner arithmetic",
            )
        )

    # --- step 3: the context floor, when one was named ----------------------
    if min_context_tokens:
        floor = int(min_context_tokens)
        cur = _metric(_row(rows, chosen_key), "kv_tokens")
        if cur and cur.get("available") and float(cur["value"]) < floor:
            alt = _row(rows, "max_context")
            alt_m = _metric(alt, "kv_tokens")
            if alt_m and alt_m.get("available") and float(alt_m["value"]) >= floor:
                chosen_key = "max_context"
                reasoning.append(
                    _step(
                        f"The chosen point predicts less than the required "
                        f"{floor:,} tokens".replace(",", " ")
                        + ", so the max-context working point is taken "
                        "instead: a configuration that cannot hold the "
                        "context is not faster, it is unusable.",
                        ESTIMATE,
                        "planner capacity arithmetic",
                    )
                )
            else:
                reasoning.append(
                    _step(
                        f"No working point predicts the required "
                        f"{floor:,} tokens".replace(",", " ")
                        + " on these cards. The suggestion stands, but the "
                        "context requirement is not met by any of them.",
                        ESTIMATE,
                        "planner capacity arithmetic",
                    )
                )

    chosen = _row(rows, chosen_key) or baseline

    # --- step 4: the flags and the expected figures -------------------------
    flags = list(chosen.get("flag_delta") or []) + list(chosen.get("lever_flags") or [])
    settings: Dict[str, Any] = {}
    if not chosen.get("is_base_split") and chosen.get("mlp_vector"):
        settings["rank_mlp_ratio"] = ",".join(str(int(v)) for v in chosen["mlp_vector"])
    if chosen.get("tune") and chosen["tune"] != "both":
        settings["rank_perf_tune"] = chosen["tune"]

    expected = []
    for m in chosen.get("metrics") or []:
        expected.append(
            {
                "key": m.get("key"),
                "label": m.get("label"),
                "unit": m.get("unit"),
                "available": bool(m.get("available")),
                "value": m.get("value"),
                "reason": m.get("reason", ""),
                "vs_baseline": m.get("vs_baseline"),
                # Every expected figure names where it came from, in the same
                # words the working-point table uses. A number on a
                # recommendation without its basis is the failure mode this
                # whole surface exists to avoid.
                "basis": m.get("basis"),
                "provenance": ESTIMATE,
            }
        )

    # --- step 5: the concurrency suggestion, kept separate ------------------
    concurrency = _concurrency_suggestion(plan)
    if concurrency.get("available"):
        reasoning.append(
            _step(
                "The state/KV balance point suggests "
                f"--max-running-requests {concurrency['recommended']} at "
                f"{concurrency['target_context_tokens']} tokens per session. "
                "It is offered separately because it is a workload choice, "
                "not part of the split.",
                ESTIMATE,
                "state/KV balance arithmetic",
            )
        )

    return {
        "ok": True,
        "model": profiles.get("model"),
        "profile": chosen_key,
        "profile_label": chosen.get("label"),
        "is_baseline": chosen_key == baseline_key,
        "selection_reason": chosen.get("selection_reason", ""),
        "axis_note": chosen.get("axis_note", ""),
        "flags": flags,
        "launch_flags": list(chosen.get("launch_flags") or []),
        "apply": {
            # The apply path is the EXISTING one: the profile key drives the
            # same control the working-point slider drives, so applying a
            # suggestion and moving the slider produce one state, not two.
            "lever_profile": chosen_key,
            "settings": settings,
            "max_running_requests": concurrency.get("recommended"),
        },
        "expected": expected,
        "baseline": {
            "key": baseline_key,
            "label": baseline.get("label"),
            "metrics": baseline.get("metrics") or [],
        },
        "concurrency": concurrency,
        "reasoning": reasoning,
        "gains": list(chosen.get("gains") or []),
        "costs": list(chosen.get("costs") or []),
        "evidence": list(chosen.get("evidence") or []),
        "caveats": list(profiles.get("caveats") or []),
        # Said in the answer rather than assumed by the reader: pressing the
        # button that produced this changed nothing on the machine.
        "boots_nothing": True,
        "note": (
            "A proposal, not an action. Nothing was started, no server was "
            "booted and no flag was written; applying it fills the ordinary "
            "configuration fields, which is a separate click."
        ),
    }


def _concurrency_suggestion(plan: Optional[dict]) -> dict:
    """The state/KV balance point at the plan's own target, or why there is
    none. Kept apart from the split: it answers a different question and a
    reader must be able to take one without the other."""
    bal = (plan or {}).get("mrr_balance")
    if not bal:
        return {
            "available": False,
            "reason": (
                "No state/KV balance for this plan: either no model is picked "
                "or this model has no recurrent state pool to trade against "
                "the KV cache."
            ),
        }
    points = bal.get("points") or []
    if not points:
        return {
            "available": False,
            "reason": "the balance solved for no target context",
        }
    # The middle rung is the one the working-point table is priced at, so the
    # two surfaces recommend a concurrency for the same session length.
    point = points[len(points) // 2]
    return {
        "available": True,
        "recommended": int(point["recommended_max_running_requests"]),
        "target_context_tokens": int(point["target_context_tokens"]),
        "sessions": int(point["sessions"]),
        "binding": point.get("binding"),
        "current": bal.get("current_max_running_requests"),
        "provenance": ESTIMATE,
        "basis": "planner arithmetic over the plan's own budgets",
        "note": bal.get("note") or "",
    }
