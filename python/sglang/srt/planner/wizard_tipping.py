# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tipping points: where a number stops recommending one thing and starts
recommending another -- and where that number came from.

WHAT A TIPPING POINT IS HERE
----------------------------
Not "a number the wizard shows". A tipping point is a threshold with a SIDE:
below it one configuration wins, above it another does. The wizard leans on
several of them -- the prompt:output ratio at which concentrating MLP units
starts paying, the decode-knee guard that refuses exactly that concentration
when the weakest rank's bandwidth binds, the satellite prefill rate at which
moving prefill off the serving cards starts buying TTFT rather than only
undisturbedness, the link rate at which the wire stops being a rounding
error. Each of them decides a recommendation, so each of them owes the reader
two sentences: what it tips, and where it came from.

THE THREE ORIGINS, AND THE BUTTON THAT FOLLOWS FROM THEM
--------------------------------------------------------
The vocabulary is ``bench_factors``' and there is no fourth word.

``measured``
    a study booted on this rig produced it. Nothing to offer.
``estimate``
    arithmetic over measured inputs, or a figure carried from a study on
    other hardware. The arithmetic is stated and the study is named, and the
    reader is offered the measurement that would replace it.
``absent``
    nobody measured it and nothing can be derived. It carries no value, and
    it carries the study that would produce one.

The last two get a "measure it now" action, because an absent number that
cannot be filled in is a dead end rather than a finding. This module builds
NO measurement machinery: the actions point at endpoints that already exist
(``/api/split_probe`` from #232, ``/api/commsuite/run`` from #271), and the
preview says what running one costs before it is run -- how long, which
cards, and what it interrupts. A reader must be able to decline for a reason.

WHY THE PREVIEW IS PART OF THE ACTION
-------------------------------------
The split probe takes every card in the set exclusively for several minutes.
On a rig where something is already serving, that is not a button one presses
to see what happens. So the action carries the card list with whoever holds
each card right now, and is reported BLOCKED with the holder named rather
than offered and then refused.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED

__all__ = [
    "TippingSources",
    "MeasureAction",
    "SPLIT_PROBE_ACTION",
    "COMM_SUITE_LINK_ACTION",
    "build_tipping_points",
    "coverage",
]


# ---------------------------------------------------------------------------
# The measurement actions -- endpoints that already exist
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MeasureAction:
    """One existing measurement, expressed as data.

    Shaped after ``bench_factors.Remeasure`` on purpose: the dashboard
    already knows how to render a job with a status path, and a second shape
    for the same idea would be a second thing to keep in step. What is added
    here is the PREVIEW -- this action takes cards away from whatever is
    using them, and the reader is told that before pressing it.
    """

    key: str
    label: str
    #: ``job`` -> POST ``path``, poll ``status_path``. ``command`` -> no
    #: endpoint exists; the harness is a command line and the surface shows
    #: it rather than a button that would do nothing. ``remote`` -> the
    #: measurement has to run on another machine.
    kind: str
    path: str = ""
    status_path: str = ""
    what: str = ""
    #: Typical wall time, as a range. A single number would be read as a
    #: promise.
    minutes: Sequence[int] = ()
    #: What it takes away while it runs.
    occupies: str = ""
    interruption: str = ""
    command: str = ""

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["minutes"] = list(self.minutes)
        return d


#: The split probe (#232): one boot, one cold prefill, one decode window,
#: teardown, for ONE split candidate. The only action here that can turn an
#: estimate about this rig's splits into a measurement.
SPLIT_PROBE_ACTION = MeasureAction(
    key="split_probe",
    label="measure this split now",
    kind="job",
    path="/api/split_probe",
    status_path="/api/split_probe/status",
    what=(
        "boots the checkpoint at this split, runs one cold 20 000-token "
        "prefill and one 25 s decode window, reads the per-rank compute/wait "
        "split off the boot log, and tears the server down again"
    ),
    minutes=(6, 8),
    occupies="every card in the tensor-parallel set, exclusively",
    interruption=(
        "a server serving on these cards has to be stopped first: the probe "
        "takes the card lock and refuses to start while another process "
        "holds a card, naming the holder rather than fighting for it"
    ),
)

#: The comm suite's cross-rig arm (#271). The link rate between MACHINES is
#: the one number an intra-rig probe may never stand in for.
COMM_SUITE_LINK_ACTION = MeasureAction(
    key="comm_suite_cross_rig",
    label="measure the link now",
    kind="job",
    path="/api/commsuite/run",
    status_path="/api/commsuite/status",
    what=(
        "runs the comm suite's cross-rig arm against the paired host and "
        "records the wire rate with its noise floor"
    ),
    minutes=(1, 1),
    occupies="the network path to the paired rig; no card is taken",
    interruption=(
        "nothing local is interrupted. The arm needs a runner that can reach "
        "the fast line -- from a container that cannot, it reports absent "
        "rather than substituting a loopback number for a wire"
    ),
)

#: The full crossover study. No endpoint exists for it, and inventing a
#: button that does nothing would be worse than printing the command.
CROSSOVER_STUDY_ACTION = MeasureAction(
    key="mlp_crossover_study",
    label="run the crossover study",
    kind="command",
    what=(
        "the interleaved multi-boot study that produces the per-candidate "
        "break-even table and the envelope, rather than one point on it"
    ),
    minutes=(55, 270),
    occupies="every card in the set, for the whole run",
    interruption="the rig serves nothing while it runs",
    command="see planner/crossover.py STUDY_TIERS for the two tiers",
)

#: A rate that belongs to a machine this rig cannot measure.
REMOTE_RATE_ACTION = MeasureAction(
    key="remote_prefill_rate",
    label="measure it on that host",
    kind="remote",
    what=(
        "the satellite's prefill rate is a property of the satellite. It "
        "comes from a probe or a boot ON that host; nothing measurable here "
        "produces it"
    ),
    occupies="the remote host's cards",
    interruption="nothing here is interrupted",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TippingSources:
    """Everything already read from disk, handed in.

    Nothing in this module opens a file, calls NVML or starts a subprocess:
    the caller (``webui``) does the reading, so the logic is testable without
    a rig and a test cannot accidentally measure something.
    """

    model_path: str = ""
    tp_size: int = 0
    #: The ticked card set, in the wizard's own shape.
    gpus: List[dict] = dataclasses.field(default_factory=list)
    #: ``split_probe.tipping_point_table()``, or None when it could not be
    #: read. The ladder rows are the measured half of the split question.
    split_table: Optional[dict] = None
    #: ``crossover.describe_evidence(crossover.load_finding())``, or None.
    crossover: Optional[dict] = None
    #: The break-even rows of that finding, already rendered to JSON.
    crossover_points: List[dict] = dataclasses.field(default_factory=list)
    #: ``plan["advantage"]``, which carries the modelled decode-knee guard.
    advantage: Optional[dict] = None
    #: The base plan's prefill rate, for the satellite threshold.
    prefill_tok_s: Optional[float] = None
    satellite_prefill_tok_s: Optional[float] = None
    #: Serving-card prefill rate under load, as a fraction of the idle rate.
    loaded_fraction: float = 0.0
    #: What the family matrix is priced at.
    context_tokens: int = 8192
    #: Bytes a handover moves per token, and the length-independent block.
    kv_bytes_per_token: Optional[float] = None
    state_bytes: Optional[float] = None
    #: The cross-rig link, when one is known, and where it came from.
    link_gbs: Optional[float] = None
    link_provenance: str = ABSENT
    link_source: str = ""
    #: Fixed cost of a PD handover, in seconds.
    handshake_s: float = 0.0
    #: ``{gpu_uuid: pids}`` for cards with a live compute process.
    busy_cards: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: The study every carried-over anchor is attributed to.
    anchor_study: str = ""
    #: Whether a remote host is paired at all.
    have_remote: bool = False
    #: Fraction of the answer above which the wire stops being negligible.
    link_share_threshold: float = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell(
    value: Optional[float], provenance: str, basis: str, unit: str = "",
    study: Optional[str] = None,
) -> dict:
    """Same invariant as ``wizard.cell``: an absent cell carries no value.

    Written out here rather than imported so this module does not depend on
    the family matrix; the two are checked against each other by a test.
    """
    if provenance == ABSENT:
        value = None
    return {
        "value": value,
        "available": value is not None,
        "provenance": provenance,
        "basis": basis,
        "unit": unit,
        "study": study,
    }


def _preview(src: TippingSources, action: MeasureAction) -> dict:
    """What running ``action`` costs, in cards and in interruption.

    The card list is the ticked set with each card's current holder, so a
    reader sees "this will stop what is running on card 1" before pressing
    rather than after.
    """
    cards: List[dict] = []
    blocked: List[str] = []
    if action.kind == "job" and action.key == "split_probe":
        for g in src.gpus:
            uuid = str(g.get("uuid") or "")
            pids = src.busy_cards.get(uuid, "") if uuid else ""
            cards.append(
                {
                    "index": g.get("index"),
                    "name": g.get("name"),
                    "uuid": uuid,
                    # None, not False: without a UUID the card cannot be
                    # matched against the running-process list, and "we did
                    # not look" must not render as "it is free".
                    "busy": bool(pids) if uuid else None,
                    "held_by_pids": pids,
                }
            )
            if pids:
                blocked.append(
                    f"{g.get('name') or 'card'} #{g.get('index')} is held by "
                    f"pid(s) {pids}"
                )
    return {
        "duration_minutes": list(action.minutes),
        "duration_text": (
            f"about {action.minutes[0]}-{action.minutes[-1]} minutes"
            if len(action.minutes) > 1 and action.minutes[0] != action.minutes[-1]
            else (f"about {action.minutes[0]} minute(s)" if action.minutes else "")
        ),
        "cards": cards,
        "occupies": action.occupies,
        "interruption": action.interruption,
        "blocked": bool(blocked),
        "blocked_reason": "; ".join(blocked),
    }


def _measure(
    src: TippingSources,
    action: Optional[MeasureAction],
    *,
    provenance: str,
    body: Optional[dict] = None,
    why: str = "",
    unavailable_reason: str = "",
) -> dict:
    """The "measure it now" block for one tipping point.

    Offered whenever the number is not a measurement -- that is the whole
    rule. A measured tipping point still carries the action, marked
    ``offered: false``, so re-measuring a stale row is one click and not a
    hunt for the page that owns the study.
    """
    if action is None:
        return {
            "available": False,
            "offered": False,
            "reason": unavailable_reason
            or "no study on this rig produces this number",
        }
    return {
        "available": not unavailable_reason,
        "offered": provenance != MEASURED and not unavailable_reason,
        "why": why
        or (
            "this number is not a measurement of this rig, so the study that "
            "would replace it is offered here"
            if provenance != MEASURED
            else "already measured here; re-run to refresh a stale row"
        ),
        "reason": unavailable_reason,
        "action": action.to_json(),
        "body": dict(body or {}),
        "preview": _preview(src, action),
    }


def _split_body(src: TippingSources, candidate: str) -> dict:
    body: Dict[str, Any] = {"model_path": src.model_path, "mlp_vector": candidate}
    if src.tp_size:
        body["tp_size"] = int(src.tp_size)
    return body


def _point(
    key: str,
    label: str,
    question: str,
    tips: str,
    value: dict,
    origin: dict,
    side: dict,
    measure: dict,
    extra: Optional[dict] = None,
) -> dict:
    row = {
        "key": key,
        "label": label,
        # What this threshold decides. Written as the decision, not as the
        # instrument -- a reader wants to know what changes, not what ran.
        "question": question,
        "tips": tips,
        "value": value,
        "origin": origin,
        "side": side,
        "measure": measure,
    }
    row.update(extra or {})
    return row


def _origin(provenance: str, source: str, detail: str, measured_at=None) -> dict:
    return {
        "provenance": provenance,
        "source": source,
        "detail": detail,
        "measured_at": measured_at,
    }


def _side(text: str, available: bool = True, which: str = "") -> dict:
    return {"available": available, "which": which, "text": text}


# ---------------------------------------------------------------------------
# The tipping points
# ---------------------------------------------------------------------------


def _best_candidate(src: TippingSources) -> str:
    """Which split the "measure it now" button would measure.

    The first ladder candidate that has no row yet, so pressing the button
    twice measures two different things. When the ladder is fully measured
    the baseline is re-measured, which is the honest default for a stale
    table.
    """
    table = src.split_table or {}
    ladder = list(table.get("ladder") or [])
    have = {
        str(r.get("candidate"))
        for r in (table.get("rows") or [])
        if r.get("measured")
    }
    for cand in ladder:
        if cand not in have:
            return str(cand)
    return str(table.get("baseline") or "auto")


def _mlp_split_point(src: TippingSources) -> dict:
    """The prompt:output ratio at which an MLP concentration starts paying.

    Concentrating dense-MLP units on the compute-strong rank buys prefill and
    costs decode, so the break-even is a ratio and not a rate. The finding is
    a property of the RIG (card mix, interconnect, per-rank quantisation
    path), which is why a number measured elsewhere may be shown for shape
    and may never select a vector.
    """
    ev = src.crossover or {}
    state = str(ev.get("state") or "unmeasured")
    points = [p for p in src.crossover_points if p.get("break_even_ratio")]
    best = min(points, key=lambda p: float(p["break_even_ratio"])) if points else None
    tips = (
        "below the ratio the base split wins and the wizard proposes it; "
        "above it the concentrated split wins and the wizard proposes that "
        "one instead. It is the switch between two whole recommendations, "
        "not a tuning detail"
    )
    question = (
        "At what mix of prompt tokens to output tokens does concentrating "
        "MLP units on the strong card start to pay?"
    )
    if state == "measured_here" and best:
        value = _cell(
            float(best["break_even_ratio"]),
            MEASURED,
            "the break-even prompt:output ratio of the best paying candidate "
            f"({best.get('label') or best.get('vector')}) in the crossover "
            "study booted on this rig",
            unit="prompt tokens per output token",
        )
        origin = _origin(
            MEASURED,
            "planner/crossover.py finding for this rig",
            "measured here; cache bypass proven and inside the staleness "
            "limit, so it may select a vector",
            ev.get("measured_at"),
        )
        side = _side(
            "Your workload's ratio decides which side you are on. The wizard "
            "applies the finding only when the form states a ratio.",
            which="",
        )
    elif points:
        value = _cell(
            float(best["break_even_ratio"]) if best else None,
            ESTIMATE,
            "carried from a finding that is not this rig's: "
            + str(ev.get("reference") or "another rig")
            + ". Shown for size and shape -- the crossover depends on the "
            "card mix, the interconnect and the per-rank quantisation path, "
            "so it may not select a vector here",
            unit="prompt tokens per output token",
        )
        origin = _origin(
            ESTIMATE,
            "planner/crossover.py reference finding",
            "; ".join(ev.get("caveats") or []) or "not measured on this rig",
        )
        side = _side(
            "Which side this rig is on is unknown until the study runs here.",
            available=False,
        )
    else:
        value = _cell(
            None,
            ABSENT,
            "no crossover finding exists for this rig and no reference "
            "finding matches this configuration. The break-even ratio is "
            "unknown rather than assumed to be zero",
            unit="prompt tokens per output token",
        )
        origin = _origin(ABSENT, "planner/crossover.py", "no finding on disk")
        side = _side("unknown", available=False)

    cand = _best_candidate(src)
    return _point(
        "mlp_split_crossover",
        "MLP concentration break-even",
        question,
        tips,
        value,
        origin,
        side,
        _measure(
            src,
            SPLIT_PROBE_ACTION,
            provenance=value["provenance"],
            body=_split_body(src, cand),
            why=(
                "one split probe measures ONE point of this curve -- the "
                f"candidate {cand!r} -- which is enough to say whether the "
                "concentration pays at this rig's operating point. The full "
                "break-even table needs the crossover study"
            ),
            unavailable_reason=(
                "" if src.model_path else "no checkpoint picked: the probe "
                "boots a real model, not a plan"
            ),
        ),
        {
            "candidates": _ladder_rows(src),
            "thorough": CROSSOVER_STUDY_ACTION.to_json(),
        },
    )


def _ladder_rows(src: TippingSources) -> List[dict]:
    """The split ladder, one row per candidate, each with its own action.

    A measured row states its deltas against the baseline; an unmeasured one
    states that it is unmeasured and offers the boot that would fill it. The
    deltas are the table's own -- recomputing them here would be a second
    arithmetic over the same rows.
    """
    table = src.split_table or {}
    rows: List[dict] = []
    for r in table.get("rows") or []:
        cand = str(r.get("candidate"))
        measured = bool(r.get("measured"))
        rows.append(
            {
                "candidate": cand,
                "is_baseline": bool(r.get("is_baseline")),
                "provenance": MEASURED if measured else ABSENT,
                "basis": (
                    f"split_probe boot, {r.get('provenance') or 'measured'}, "
                    f"{r.get('source') or 'this rig'}"
                    if measured
                    else str(
                        r.get("missing_reason")
                        or "not measured on this rig; one boot produces the row"
                    )
                ),
                "unbootable": r.get("unbootable"),
                "decode_tok_s": r.get("decode_tok_s"),
                "prefill_tok_s": r.get("prefill_tok_s"),
                "ms_per_verify": r.get("ms_per_verify"),
                "delta": r.get("delta"),
                "measure": _measure(
                    src,
                    SPLIT_PROBE_ACTION,
                    provenance=MEASURED if measured else ABSENT,
                    body=_split_body(src, cand),
                    unavailable_reason=(
                        "" if src.model_path else "no checkpoint picked"
                    ),
                ),
            }
        )
    return rows


def _decode_knee_point(src: TippingSources) -> dict:
    """The guard that refuses a concentration the prefill side would want.

    Decode is bandwidth bound and gated by the SLOWEST rank, so moving units
    onto the strong card can leave the weak ranks unable to keep up. The
    optimizer models this and refuses past the knee. Modelled is not
    measured, and the honest reading of a modelled guard is that it says
    which side it believes we are on, not by how much.
    """
    adv = src.advantage or {}
    ok = adv.get("decode_knee_ok")
    question = (
        "Does concentrating MLP units past this point cost more decode on "
        "the weakest rank than it buys in prefill?"
    )
    tips = (
        "on the safe side the optimizer will propose a concentrated split; "
        "past the knee it refuses that split outright, whatever the prefill "
        "gain looks like"
    )
    if ok is None:
        value = _cell(
            None,
            ABSENT,
            "the guard needs measured per-card memory-bandwidth scores and "
            "there are none on disk, so the knee is not computable -- not "
            "'safe'",
        )
        origin = _origin(
            ABSENT,
            "planner/advantage.py decode-knee guard",
            "no measured membw scores: run the card probe first, then a "
            "split probe to observe the decode side directly",
        )
        side = _side("unknown", available=False)
    else:
        value = _cell(
            1.0 if ok else 0.0,
            ESTIMATE,
            "modelled by the optimizer's decode-knee guard from the measured "
            "per-card bandwidth scores. A guard verdict, not a measured "
            "decode rate: it says which side of the knee this split is on",
            unit="1 = clear, 0 = past the knee",
        )
        origin = _origin(
            ESTIMATE,
            "planner/advantage.py decode-knee guard",
            "modelled over measured card rates",
        )
        side = _side(
            "clear of the knee at this split"
            if ok
            else "past the knee: the optimizer refuses this concentration",
            which="clear" if ok else "past",
        )
    cand = _best_candidate(src)
    return _point(
        "decode_knee",
        "decode knee",
        question,
        tips,
        value,
        origin,
        side,
        _measure(
            src,
            SPLIT_PROBE_ACTION,
            provenance=value["provenance"],
            body=_split_body(src, cand),
            why=(
                "the guard is modelled. A split probe boots the split and "
                "reads the decode rate and the per-rank compute/wait split "
                "off the running server, which is the observation the model "
                "stands in for"
            ),
            unavailable_reason=(
                "" if src.model_path else "no checkpoint picked"
            ),
        ),
    )


def _satellite_point(src: TippingSources) -> dict:
    """The satellite prefill rate at which moving prefill off starts to pay.

    A satellite arm always buys undisturbedness. Whether it buys TTFT depends
    on whether its own prefill, plus the wire, plus the handover, beats the
    serving cards' prefill UNDER LOAD -- which is the rate that matters,
    because an idle serving card is not the case anybody moves prefill away
    from. The threshold follows from that comparison and nothing else.
    """
    question = (
        "How fast would the satellite's prefill have to be before moving "
        "prefill off the serving cards buys time to first token, and not "
        "only undisturbedness?"
    )
    tips = (
        "below the threshold the satellite costs TTFT and buys only "
        "undisturbedness; above it the satellite wins both, and the wizard "
        "stops presenting the arm as a trade"
    )
    ctx = max(int(src.context_tokens), 1)
    loaded_rate = (
        float(src.prefill_tok_s) * float(src.loaded_fraction)
        if src.prefill_tok_s and src.loaded_fraction
        else None
    )
    overhead = float(src.handshake_s or 0.0)
    if src.kv_bytes_per_token and src.link_gbs:
        overhead += (
            float(src.kv_bytes_per_token) * ctx + float(src.state_bytes or 0.0)
        ) / (float(src.link_gbs) * 1e9)
    budget = (ctx / loaded_rate - overhead) if loaded_rate else None
    if budget is not None and budget > 0:
        threshold = ctx / budget
        value = _cell(
            threshold,
            ESTIMATE,
            "the serving cards' loaded prefill time at this context, minus "
            "the handover and the transport, is the satellite's whole time "
            "budget; the threshold is the context divided by what is left. "
            "The loaded rate is the plan's idle prefill rate times the "
            f"measured loaded fraction {src.loaded_fraction:.3f}",
            unit="tok/s",
            study=src.anchor_study or None,
        )
        origin = _origin(
            ESTIMATE,
            "arithmetic over the plan's prefill rate and " + (
                src.anchor_study or "the satellite study's anchors"
            ),
            "the inputs are measured; the division is not a measurement",
        )
        if src.satellite_prefill_tok_s:
            wins = float(src.satellite_prefill_tok_s) >= threshold
            side = _side(
                (
                    f"the satellite's {src.satellite_prefill_tok_s:.0f} tok/s "
                    + ("clears" if wins else "does not reach")
                    + f" the {threshold:.0f} tok/s threshold at "
                    f"{ctx:,} tokens"
                ),
                which="above" if wins else "below",
            )
        else:
            side = _side(
                "no satellite prefill rate is known, so which side the arm "
                "is on cannot be stated",
                available=False,
            )
    else:
        value = _cell(
            None,
            ABSENT,
            (
                "no prefill rate for the serving cards, so there is nothing "
                "for the satellite to be compared against"
                if not loaded_rate
                else "the handover and the transport alone already exceed "
                "the serving cards' loaded prefill time at this context: no "
                "satellite rate can win here, however fast the card is"
            ),
            unit="tok/s",
        )
        origin = _origin(
            ABSENT,
            "the plan's prefill rate",
            "absent input, so the threshold is not computed",
        )
        side = _side(
            "below, at any satellite rate"
            if loaded_rate
            else "unknown",
            available=bool(loaded_rate),
            which="below" if loaded_rate else "",
        )
    return _point(
        "satellite_break_even",
        "satellite break-even prefill rate",
        question,
        tips,
        value,
        origin,
        side,
        _measure(
            src,
            REMOTE_RATE_ACTION,
            provenance=value["provenance"],
            why=(
                "the threshold is arithmetic and needs no measurement; what "
                "is missing is the satellite's own prefill rate, and that is "
                "a property of the other machine"
            ),
            unavailable_reason=(
                ""
                if src.have_remote
                else "no remote host is paired, so there is no machine to "
                "measure"
            ),
        ),
    )


def _link_share_point(src: TippingSources) -> dict:
    """The link rate below which the wire stops being a rounding error.

    In the measured satellite arm the wire was 1.8 % of the answer, which is
    why "a faster line" was not the fix. The same payload over a slow line is
    a different verdict, and the threshold is where the transport share
    crosses the stated fraction of the answer.
    """
    share = float(src.link_share_threshold)
    question = (
        f"Below what line rate does the handover stop being negligible "
        f"(more than {share * 100:.0f} % of the answer)?"
    )
    tips = (
        "above the threshold the line is a rounding error and a faster one "
        "changes nothing; below it the wire is the term to attack, and the "
        "wizard's advice flips from 'get a stronger prefill card' to 'get a "
        "faster line'"
    )
    ctx = max(int(src.context_tokens), 1)
    answer_s = None
    if src.satellite_prefill_tok_s:
        answer_s = ctx / float(src.satellite_prefill_tok_s) + float(
            src.handshake_s or 0.0
        )
    payload = (
        float(src.kv_bytes_per_token) * ctx + float(src.state_bytes or 0.0)
        if src.kv_bytes_per_token
        else None
    )
    if answer_s and payload:
        # transport = payload / rate; transport / (answer + transport) = share
        threshold_gbs = payload * (1.0 - share) / (share * answer_s) / 1e9
        value = _cell(
            threshold_gbs,
            ESTIMATE,
            "the handover payload at this context (KV bytes per token from "
            "the plan's balance report plus the length-independent state "
            "block) set against the rest of the answer, solved for the rate "
            f"at which transport reaches {share * 100:.0f} % of it",
            unit="GB/s",
        )
        origin = _origin(
            ESTIMATE,
            "model geometry from the plan + the satellite's prefill rate",
            "arithmetic over measured inputs",
        )
        if src.link_gbs:
            above = float(src.link_gbs) >= threshold_gbs
            side = _side(
                f"the known line at {float(src.link_gbs):.2f} GB/s is "
                + ("above" if above else "below")
                + f" the {threshold_gbs:.2f} GB/s threshold",
                which="above" if above else "below",
            )
        else:
            side = _side(
                "no cross-rig link rate is known, so which side the line is "
                "on cannot be stated",
                available=False,
            )
    else:
        value = _cell(
            None,
            ABSENT,
            "needs the handover payload (KV bytes per token from the plan) "
            "and the length of the rest of the answer (the satellite's "
            "prefill rate). Without both, the share the wire carries is not "
            "computable",
            unit="GB/s",
        )
        origin = _origin(ABSENT, "the plan's balance report", "absent input")
        side = _side("unknown", available=False)
    return _point(
        "link_share_break_even",
        "link rate at which the wire starts to matter",
        question,
        tips,
        value,
        origin,
        side,
        _measure(
            src,
            COMM_SUITE_LINK_ACTION,
            provenance=src.link_provenance,
            body={"arms": ["cross_rig"]},
            why=(
                "the threshold is arithmetic; what decides the verdict is "
                "the actual line rate, and the cross-rig arm measures it. An "
                "intra-rig pair figure is never substituted here -- it prices "
                "a path the handover does not cross"
            ),
            unavailable_reason=(
                ""
                if src.have_remote
                else "no remote host is paired, so there is no wire to measure"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_tipping_points(src: TippingSources) -> dict:
    """Every threshold the wizard leans on, with its origin and its button."""
    points = [
        _mlp_split_point(src),
        _decode_knee_point(src),
        _satellite_point(src),
        _link_share_point(src),
    ]
    return {
        "points": points,
        "coverage": coverage(points),
        "vocabulary": [MEASURED, ESTIMATE, ABSENT],
        "note": (
            "A tipping point is a threshold with a side: one configuration "
            "wins below it and another above it. Each row says what it tips "
            "and where the number came from; anything that is not a "
            "measurement of this rig carries the study that would replace it."
        ),
    }


def coverage(points: Sequence[dict]) -> dict:
    """How many of the thresholds are measured, derived, or missing.

    Counted over the tipping points AND their ladder rows, because a table of
    candidates in which one row is measured is not a measured table.
    """
    counts = {MEASURED: 0, ESTIMATE: 0, ABSENT: 0}
    offered = 0
    for p in points:
        prov = str((p.get("value") or {}).get("provenance") or ABSENT)
        counts[prov] = counts.get(prov, 0) + 1
        if (p.get("measure") or {}).get("offered"):
            offered += 1
        for row in p.get("candidates") or []:
            prov = str(row.get("provenance") or ABSENT)
            counts[prov] = counts.get(prov, 0) + 1
            if (row.get("measure") or {}).get("offered"):
                offered += 1
    total = sum(counts.values())
    return {
        "measured": counts[MEASURED],
        "estimate": counts[ESTIMATE],
        "absent": counts[ABSENT],
        "total": total,
        "measurable_now": offered,
        "summary": (
            f"{counts[MEASURED]} measured, {counts[ESTIMATE]} derived, "
            f"{counts[ABSENT]} not measured; {offered} of them have a study "
            "that can be started from here"
        ),
    }
