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
"""Where the wizard's link and satellite rates come from.

THE PROBLEM THIS SOLVES
-----------------------
The v1 family matrix priced the satellite and PD rows off module constants:
the 40G line's 1930 MB/s, the 2080 Ti's 2385 tok/s, the 10 850 / 25 400
loaded-prefill ratio, the 0.136 s handover. Every one of them is a real
measurement -- but of one afternoon, one pair of machines, one 2B checkpoint.
Written as constants they read like properties of the method, and a reader on
different hardware has no way to tell which numbers describe their rig and
which describe ours.

So the numbers keep their place as REFERENCE ANCHORS and stop being the first
source. Each rate is resolved down a fixed ladder, and every answer says which
rung it came from:

1. what the form states (the reader knows their own line);
2. what this rig measured -- the card probe's ordered pair matrix on disk,
   the comm suite's arms in the shared artifact schema;
3. the reference anchor, labelled ``estimate`` and named as a figure from
   other hardware;
4. absent, with the study that would produce it.

ONE RULE THAT IS NOT NEGOTIABLE
-------------------------------
An intra-rig pair figure never fills a cross-rig cell. The card-to-card
matrix prices a path a network handover does not cross, and substituting one
for the other would make the wizard's most consequential recommendation --
whether to move prefill to another machine -- rest on a PCIe number. When no
cross-rig rate exists the cell is absent and names the arm that measures it.
The rule is asserted by a test, not only by this paragraph.

READ ONLY
---------
Nothing here opens a file or a socket. The caller hands in what it already
read; a source that measured on read would make "look at the numbers" an
action with side effects.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED

__all__ = [
    "LinkSources",
    "RATE_KEYS",
    "read_rates",
    "pair_matrix_rows",
    "coverage",
]

#: The rates the family matrix prices its network and PD rows with.
RATE_KEYS = (
    "intra_rig_narrowest_gbs",
    "cross_rig_gbs",
    "satellite_prefill_tok_s",
    "loaded_prefill_fraction",
    "pd_handshake_s",
    "decode_spike_ms",
    "decode_spike_offloaded_ms",
)

#: Measurement ids in the shared artifact (``htsglang-rig-artifact/v1``) that
#: carry a cross-rig wire rate. The comm suite emits ``comm/<arm>/<cell>`` and
#: a ``/rate`` sibling in Gbit/s for cells that have one.
_CROSS_RIG_PREFIX = "comm/cross_rig/"


@dataclasses.dataclass
class LinkSources:
    """Everything already read, handed in.

    ``artifact_measurements`` are rows in the shared artifact's measurement
    shape (``id``, ``unit``, ``value``, ``taken_at``, ``context``) -- what
    ``comm_suite.to_sections`` produces. Passing rows rather than a job keeps
    this module independent of whether the run is still in memory or came
    back from the shared digest.
    """

    #: ``card_probe.CardProbeProfile.pairs``, as JSON rows.
    card_probe_pairs: List[dict] = dataclasses.field(default_factory=list)
    #: ``uuid -> display name``, for naming the narrowest pair.
    card_names: Dict[str, str] = dataclasses.field(default_factory=dict)
    card_probe_created: Optional[float] = None
    card_probe_transports: List[str] = dataclasses.field(default_factory=list)
    artifact_measurements: List[dict] = dataclasses.field(default_factory=list)
    #: What the form states, and where the reader says it came from.
    form_link_gbs: Optional[float] = None
    form_link_source: str = ""
    form_satellite_tok_s: Optional[float] = None
    #: Paired remote hosts, by address. Empty means no cross-rig anything.
    remote_targets: List[str] = dataclasses.field(default_factory=list)
    #: The reference anchors, handed in so there is exactly one definition of
    #: them in the tree (``wizard.ANCHORS``).
    anchors: Dict[str, Any] = dataclasses.field(default_factory=dict)
    anchor_study: str = ""

    @property
    def have_remote(self) -> bool:
        return bool(self.remote_targets)


# ---------------------------------------------------------------------------
# One reading
# ---------------------------------------------------------------------------


def _reading(
    key: str,
    label: str,
    unit: str,
    value: Optional[float],
    provenance: str,
    source: str,
    basis: str,
    *,
    study: str = "",
    measured_at: Optional[float] = None,
    way: str = "",
    considered: Optional[List[dict]] = None,
) -> dict:
    """One rate with the rung it came from.

    ``considered`` is the ladder that was walked, with what each rung said.
    It is the difference between "absent" and "absent, and here is what was
    looked at" -- the second is checkable and the first is an assertion.
    """
    if provenance == ABSENT:
        value = None
    return {
        "key": key,
        "label": label,
        "unit": unit,
        "value": value,
        "available": value is not None,
        "provenance": provenance,
        "source": source,
        "basis": basis,
        "study": study,
        "measured_at": measured_at,
        "way": way,
        "considered": list(considered or []),
    }


def _rung(name: str, verdict: str) -> dict:
    return {"rung": name, "verdict": verdict}


# ---------------------------------------------------------------------------
# The intra-rig matrix
# ---------------------------------------------------------------------------


def pair_matrix_rows(src: LinkSources) -> List[dict]:
    """The ordered card-to-card matrix, named and labelled.

    Ordered because a route can be asymmetric: a card on a narrow slot
    uploads and downloads differently. Every row carries the transport it was
    actually measured over, so a host-staging matrix cannot be read as a
    degraded p2p one.
    """
    out: List[dict] = []
    for p in src.card_probe_pairs:
        if p.get("bandwidth_gbs") is None:
            continue
        s, d = str(p.get("src_uuid") or ""), str(p.get("dst_uuid") or "")
        out.append(
            {
                "src": src.card_names.get(s, s[:12]),
                "dst": src.card_names.get(d, d[:12]),
                "bandwidth_gbs": float(p["bandwidth_gbs"]),
                "latency_us": p.get("latency_us"),
                "transport": p.get("transport") or "unknown",
                "peer_access": bool(p.get("peer_access")),
                "provenance": MEASURED,
                "scope": "intra-rig",
            }
        )
    return sorted(out, key=lambda r: r["bandwidth_gbs"])


def _intra_rig(src: LinkSources) -> dict:
    rows = pair_matrix_rows(src)
    considered = [
        _rung(
            "card probe pair matrix",
            f"{len(rows)} ordered pair(s) with a bandwidth"
            if rows
            else "no cached probe, or it measured no pair bandwidth",
        )
    ]
    if not rows:
        return _reading(
            "intra_rig_narrowest_gbs",
            "narrowest ordered card-to-card link",
            "GB/s",
            None,
            ABSENT,
            "",
            "no card probe on disk for the cards visible now, so the "
            "collective floor is unknown rather than estimated",
            way="run the card probe (/api/card_probe); a single-card rig has "
            "no pair to measure",
            considered=considered,
        )
    slow = rows[0]
    return _reading(
        "intra_rig_narrowest_gbs",
        "narrowest ordered card-to-card link",
        "GB/s",
        slow["bandwidth_gbs"],
        MEASURED,
        "card probe, ordered pair matrix",
        f"{slow['src']} -> {slow['dst']} over {slow['transport']}; every "
        "collective waits on the slow direction of the slowest pair",
        measured_at=src.card_probe_created,
        considered=considered,
    )


# ---------------------------------------------------------------------------
# The cross-rig wire
# ---------------------------------------------------------------------------


def _artifact_cross_rig(src: LinkSources) -> Optional[dict]:
    """The comm suite's cross-rig arm, if it produced a rate.

    The suite emits a ``/rate`` row in Gbit/s beside each timing cell. GB/s
    is what the rest of the wizard prices in, so the conversion happens here,
    once, and is stated in the basis rather than left for the reader.
    """
    best: Optional[dict] = None
    for m in src.artifact_measurements:
        mid = str(m.get("id") or "")
        if not mid.startswith(_CROSS_RIG_PREFIX):
            continue
        if str(m.get("unit") or "") != "Gbit/s" or m.get("value") is None:
            continue
        if best is None or float(m["value"]) > float(best["value"]):
            best = m
    if best is None:
        return None
    return {
        "gbs": float(best["value"]) / 8.0,
        "gbit_s": float(best["value"]),
        "id": best.get("id"),
        "taken_at": best.get("taken_at"),
        "label": best.get("label"),
    }


def _cross_rig(src: LinkSources) -> dict:
    considered: List[dict] = []
    if src.form_link_gbs:
        considered.append(_rung("the form", "a rate was supplied"))
        return _reading(
            "cross_rig_gbs",
            "link between the machines",
            "GB/s",
            float(src.form_link_gbs),
            MEASURED if src.form_link_source else ESTIMATE,
            src.form_link_source or "supplied on the form",
            "supplied by the reader"
            + (f" ({src.form_link_source})" if src.form_link_source else "")
            + ". A stated rate is taken at face value; it is the reader's rig",
            considered=considered,
        )
    considered.append(_rung("the form", "no rate supplied"))

    art = _artifact_cross_rig(src)
    considered.append(
        _rung(
            "comm suite cross-rig arm (htsglang-rig-artifact/v1)",
            f"{art['label']} at {art['gbit_s']:.1f} Gbit/s"
            if art
            else "no cross-rig row in the artifact",
        )
    )
    if art:
        return _reading(
            "cross_rig_gbs",
            "link between the machines",
            "GB/s",
            art["gbs"],
            MEASURED,
            f"comm suite, {art['id']}",
            f"{art['gbit_s']:.1f} Gbit/s measured by the cross-rig arm, "
            "converted to GB/s (divide by 8) because the rest of this page "
            "prices payloads in bytes",
            measured_at=art.get("taken_at"),
            considered=considered,
        )

    # Deliberately NOT falling through to the intra-rig matrix.
    considered.append(
        _rung(
            "card probe pair matrix",
            "not consulted: it prices card-to-card paths inside one machine, "
            "which a network handover does not cross",
        )
    )
    anchors = src.anchors or {}
    ref = anchors.get("link_gbs_40g")
    study = src.anchor_study or "the satellite study"
    if ref:
        considered.append(
            _rung(
                "reference anchor",
                f"{float(ref):.2f} GB/s from {study}, which is a fact about "
                "that cabling",
            )
        )
    return _reading(
        "cross_rig_gbs",
        "link between the machines",
        "GB/s",
        None,
        ABSENT,
        "",
        "nothing on this rig has measured a wire to another machine. The "
        "reference line from "
        + (src.anchor_study or "the satellite study")
        + (
            f" ran at {float(ref):.2f} GB/s, which is a fact about that "
            "cabling and not about yours"
            if ref
            else ""
        ),
        way=(
            "run the comm suite's cross-rig arm from a runner that can reach "
            "the fast line (/api/commsuite/run with arms=[cross_rig]), or "
            "state the rate on the form"
            if src.have_remote
            else "pair a remote host first; nothing about a machine that has "
            "not answered can be measured"
        ),
        considered=considered,
    )


# ---------------------------------------------------------------------------
# The satellite's own rate, and the anchors the arithmetic runs over
# ---------------------------------------------------------------------------


def _satellite(src: LinkSources) -> dict:
    considered: List[dict] = []
    if src.form_satellite_tok_s:
        considered.append(_rung("the form", "a rate was supplied"))
        return _reading(
            "satellite_prefill_tok_s",
            "satellite prefill rate",
            "tok/s",
            float(src.form_satellite_tok_s),
            MEASURED,
            "supplied on the form",
            "the reader's own figure for the satellite card. It is the term "
            "that decides this family, so it is never guessed",
            considered=considered,
        )
    considered.append(_rung("the form", "no rate supplied"))
    considered.append(
        _rung(
            "this rig's studies",
            "not applicable: the satellite's prefill rate is a property of "
            "the other machine, and nothing measurable here produces it",
        )
    )
    anchors = src.anchors or {}
    ref = anchors.get("satellite_prefill_tok_s")
    if ref:
        considered.append(
            _rung(
                "reference anchor",
                f"{float(ref):.0f} tok/s, measured on the study's own "
                "satellite card and checkpoint",
            )
        )
    return _reading(
        "satellite_prefill_tok_s",
        "satellite prefill rate",
        "tok/s",
        None,
        ABSENT,
        "",
        "not known for the machine you would use. In "
        + (src.anchor_study or "the satellite study")
        + " this term was 93.5 % of the whole answer, so stating a TTFT "
        "without it would be stating the one number that decides the family "
        "from a card nobody named",
        way=(
            "probe or boot the satellite host and read its prefill rate, or "
            "state it on the form"
        ),
        considered=considered,
    )


def _anchor_reading(
    src: LinkSources,
    key: str,
    anchor_key: str,
    label: str,
    unit: str,
    what: str,
) -> dict:
    """A figure the arithmetic needs that only the reference study measured.

    Carried as an ``estimate`` for this rig rather than as a measurement:
    the value IS measured, but on other hardware and another checkpoint, and
    the label has to say which of the two a reader is looking at.
    """
    anchors = src.anchors or {}
    v = anchors.get(anchor_key)
    considered = [
        _rung("this rig's studies", "no local study measures this"),
        _rung(
            "reference anchor",
            f"{v} from {src.anchor_study or 'the satellite study'}"
            if v is not None
            else "not present in the anchor set",
        ),
    ]
    if v is None:
        return _reading(
            key, label, unit, None, ABSENT, "", what,
            way="measure the pair on your own hardware",
            considered=considered,
        )
    return _reading(
        key,
        label,
        unit,
        float(v),
        ESTIMATE,
        src.anchor_study or "the satellite study",
        what
        + ". Measured, but on that study's hardware and checkpoint -- carried "
        "here because the arithmetic needs it, and labelled so it is not read "
        "as a figure about your cards",
        study=src.anchor_study,
        way="re-measure the same pair on your rig to replace it",
        considered=considered,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def read_rates(src: LinkSources) -> dict:
    """Every rate the network and PD families price with, plus its origin."""
    rates: Dict[str, dict] = {
        "intra_rig_narrowest_gbs": _intra_rig(src),
        "cross_rig_gbs": _cross_rig(src),
        "satellite_prefill_tok_s": _satellite(src),
        "loaded_prefill_fraction": _anchor_reading(
            src,
            "loaded_prefill_fraction",
            "loaded_prefill_rate_fraction",
            "prefill rate under load, as a fraction of idle",
            "",
            "the serving cards' prefill rate with decode streams in flight, "
            "divided by their idle rate. It is what makes a TTFT pair a pair",
        ),
        "pd_handshake_s": _anchor_reading(
            src,
            "pd_handshake_s",
            "pd_handshake_s",
            "PD handover",
            "s",
            "bootstrap handshake, scheduling and the first decode step of a "
            "handover -- the fixed cost every PD family adds",
        ),
        "decode_spike_ms": _anchor_reading(
            src,
            "decode_spike_ms",
            "decode_spike_ms",
            "worst inter-token time with prefill on the decode cards",
            "ms",
            "the decode spike a long prompt causes when prefill shares the "
            "serving cards",
        ),
        "decode_spike_offloaded_ms": _anchor_reading(
            src,
            "decode_spike_offloaded_ms",
            "decode_spike_offloaded_ms",
            "worst inter-token time with prefill moved off",
            "ms",
            "the same quantity once prefill no longer runs on the decode "
            "cards -- the undisturbedness the network families buy",
        ),
    }
    return {
        "rates": rates,
        "pairs": pair_matrix_rows(src),
        "remotes": list(src.remote_targets),
        "coverage": coverage(rates),
        "rules": [
            "An intra-rig pair figure never fills a cross-rig cell: the card "
            "matrix prices a path the handover does not cross.",
            "A figure carried from another rig is an estimate here, however "
            "carefully it was measured there.",
            "An absent rate names the study that would produce it; it is "
            "never replaced by the nearest available number.",
        ],
    }


def coverage(rates: Dict[str, dict]) -> dict:
    counts = {MEASURED: 0, ESTIMATE: 0, ABSENT: 0}
    for r in rates.values():
        counts[str(r.get("provenance"))] = (
            counts.get(str(r.get("provenance")), 0) + 1
        )
    return {
        "measured": counts[MEASURED],
        "estimate": counts[ESTIMATE],
        "absent": counts[ABSENT],
        "total": len(rates),
        "summary": (
            f"{counts[MEASURED]} of {len(rates)} rates measured on this rig, "
            f"{counts[ESTIMATE]} carried from the reference study, "
            f"{counts[ABSENT]} absent"
        ),
    }
