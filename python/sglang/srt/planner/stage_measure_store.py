# SPDX-License-Identifier: Apache-2.0
"""The per-stage measurement canon: gain, band and instrumented flip cost.

WHAT WAS MISSING, STATED EXACTLY (#584 second half, the gate on #363).

`#584`'s first half put measured CARD RATES on disk, UUID-keyed
(`planner/card_rate_pass.py`). That cleared the planner feed's own blocker --
`PlannerFeedUnavailable('no card probe on disk')` -- and the solver now
produces stage candidates on this rig. It did not produce a single FLIP
TARGET, and the reason is named by the code that refuses them:

    RegimeError: stage table refused (#578): the planner solved 1 stage(s) --
    solved-enc -- but they carry no measurement. Each needs
    measured_gain_pct, measured_band_pct and flip_cost_s ... The solver cannot
    predict any of the three.

Those three numbers are MEASUREMENTS. A solver predicts a layout; it cannot
predict what that layout does to ms/round on this rig, how much of that
difference is the rig's own noise, or how many seconds the flip into it costs.
This module is where the three live once somebody has taken them.

WHY A STORE AND NOT A FLAG. Three reasons, in the order they bite:

1. A measurement is expensive (two boots and a flip, >= 10 s of load each) and
   a flag would have to be re-typed on every subsequent boot, which is how a
   number that was measured once becomes a number nobody can attribute.
2. The numbers are RIG-SPECIFIC and CHECKPOINT-SPECIFIC. `#584` measured what
   happens when they are not: rates borrowed from an earlier shift were
   -12.2 % and -22.5 % off after a power-target change, and the admission they
   produced rested on cards that no longer existed in that state. A store can
   carry the identity a flag cannot.
3. The refusal has to stay reachable. A missing record must produce a NAMED
   refusal ("this stage has never been measured, here is the pass that
   measures it"), never a default, and never a zero that reads as a measured
   zero.

KEYED BY RIG UUID SET, THE SAME IDENTITY `#584` USES. `card_rate_pass` keys
cards by NVML UUID because two RTX 3080s in one box share `props.name` and do
not measure identically. A stage measurement is a property of the whole GROUP
-- it is a ms/round difference across a TP set -- so its key is the SORTED SET
of the participating cards' UUIDs, plus the checkpoint. A record measured on
another rig is not read; it is REFUSED BY NAME, so the operator sees "measured
on a different card set" rather than a silently borrowed number.

WHAT IS DELIBERATELY NOT HERE. No measuring. This module is the record, its
identity, its validation and its file. Taking the measurement is
`planner/stage_measure_pass.py`, which reads the controller's own traces; the
separation is the same one `card_rate_pass` keeps between the probe and the
library it projects onto.

Pure: json, dataclasses and os. NVML is reached lazily and only to derive the
rig key, so every test in this file's suite runs on a machine with no driver.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ARTIFACT_VERSION",
    "MIN_MEASURE_SECONDS",
    "STAGE_MEASUREMENT_BASENAME",
    "StageMeasurement",
    "StageMeasurementError",
    "StageMeasurementLibrary",
    "rig_key_from_uuids",
    "rig_key",
    "stage_measure_path",
]

#: Beside `card_library.json`, in the directory `rigmon/card_probe.py` owns.
#: A reader who finds the card rates should find the stage measurements taken
#: on those cards, without being told where to look.
STAGE_MEASUREMENT_BASENAME = "stage_measurements.json"

ARTIFACT_VERSION = 1

#: THE MEASUREMENT-RUN FLOOR, in seconds of covered device time per arm.
#: House canon (`ms/round als Messlatte`): a measurement run is >= 10 s. Below
#: that the mean is a handful of rounds and the "band" is whichever round
#: happened to be slow. Enforced by `StageMeasurement.refusals`, not advised
#: in a comment, because every previous version of this rule was advised in a
#: comment.
MIN_MEASURE_SECONDS = 10.0


class StageMeasurementError(RuntimeError):
    """A malformed record, an unreadable store, or a borrowed identity."""


def stage_measure_path(path: Optional[str] = None) -> str:
    """The canonical location of the stage-measurement store.

    Resolution order, deliberately the same shape as
    `card_rate_pass.card_library_path`:

      1. an explicit argument (tests, and an operator naming an artifact),
      2. ``SGLANG_STAGE_MEASUREMENTS``,
      3. ``stage_measurements.json`` in the directory that holds the card
         library.

    Returns a path and promises nothing about its existence. A missing file is
    a measurement that has not been taken; the caller REFUSES rather than
    defaulting, which is the whole point of the slice.
    """
    if path:
        return str(path)
    env = None
    try:
        from sglang.srt import environ as _environ

        env = _environ.envs.SGLANG_STAGE_MEASUREMENTS.get()
    except Exception:
        env = os.environ.get("SGLANG_STAGE_MEASUREMENTS") or None
    if env:
        return str(env)
    from sglang.srt.planner.card_rate_pass import card_library_path

    return os.path.join(
        os.path.dirname(card_library_path()), STAGE_MEASUREMENT_BASENAME
    )


def rig_key_from_uuids(uuids: Sequence[str]) -> str:
    """``<n>:<uuid>,<uuid>,...`` over the SORTED uuid set.

    Sorted, so the key does not depend on which rank enumerated first; the
    count is carried in front so a two-card subset of a three-card rig cannot
    collide with anything by prefix. Empty is refused: a key that means "some
    cards" would let every record match every rig.
    """
    clean = [str(u).strip() for u in uuids if str(u).strip()]
    if not clean:
        raise StageMeasurementError(
            "a rig key needs at least one card UUID. An empty key would match "
            "every rig, which is exactly the borrowed-measurement failure this "
            "key exists to prevent (#584 section 3)."
        )
    if len(set(clean)) != len(clean):
        raise StageMeasurementError(
            f"duplicate card UUIDs in the rig key: {clean}. Two ranks on one "
            f"physical card are two ranks, but they are ONE card in the "
            f"identity, and a duplicate here means the caller passed per-rank "
            f"entries where per-card ones were wanted."
        )
    return f"{len(clean)}:" + ",".join(sorted(clean))


def rig_key(uuids: Optional[Sequence[str]] = None) -> str:
    """This rig's key, from NVML when no explicit uuid list is given.

    NVML, never torch's enumeration order: `registry/nvml.py`'s ``IdentityMap``
    is the house's single source of card identity for exactly the reason this
    key exists (#331, and the device-order trap in the catalogue's section 11).
    """
    if uuids is not None:
        return rig_key_from_uuids(uuids)
    try:
        from sglang.srt.registry.nvml import identity_map

        cards = identity_map().cards
    except Exception as exc:  # noqa: BLE001
        raise StageMeasurementError(
            f"cannot derive the rig key: NVML identity is unavailable ({exc!r}). "
            f"A stage measurement is keyed by the card set it was taken on; "
            f"without the identity the record can be neither written nor "
            f"matched, and guessing one would be the borrowed-rates defect."
        ) from exc
    return rig_key_from_uuids([c.uuid for c in cards])


@dataclasses.dataclass(frozen=True)
class StageMeasurement:
    """One stage's measured gain, band and flip cost, with its provenance.

    THE THREE NUMBERS AND WHAT EACH ONE IS.

    * ``gain_pct`` -- the ms/round difference between this stage and the
      REFERENCE stage, measured on this stage's own phase, positive when this
      stage is faster. Same quantity `regime_classifier.Stage.measured_gain_pct`
      names, in the same sign convention.
    * ``band_pct`` -- what that difference has to beat to be a difference. It
      is the LARGER of the same-boot A-vs-A floor (the reference arm against
      its own repeat) and the drift the measurement run itself showed. Taking
      the larger is the conservative direction, and the drift term is in there
      because #459 measured a monotone 13.0 % drift being reported as a 3.0 %
      noise floor: a run that walks is not a run that is quiet.
    * ``flip_cost_s`` -- the INSTRUMENTED duration of the move into this stage,
      in seconds, taken from the actuator's own report (#297 reshard
      ``total_ms``, #330 commit, or the #631 seam census) and never estimated.

    Everything else on this record is provenance, and it is on the record
    rather than in a note because a measurement whose conditions cannot be
    read is a number, not a measurement.
    """

    stage: str
    regime: str
    reference: str
    rig_key: str
    model_key: str
    gain_pct: float
    band_pct: float
    flip_cost_s: float
    #: The two halves of the band, kept separately so a reader can see which
    #: one bound.
    avs_a_floor_pct: float = 0.0
    drift_pct: float = 0.0
    #: Covered device time per arm, seconds. The >= 10 s canon is checked
    #: against BOTH: an arm measured for 40 s against one measured for 3 s is
    #: a 3 s measurement with a long control attached.
    covered_s_reference: float = 0.0
    covered_s_stage: float = 0.0
    #: Boundaries (consensus rounds) each arm contributed.
    boundaries_reference: int = 0
    boundaries_stage: int = 0
    #: Instrumented flips this cost came from. `flip_cost_s` is their MAXIMUM;
    #: the mean is carried for the reader, never used for the decision.
    flip_samples: int = 0
    flip_cost_mean_s: float = 0.0
    #: Free text naming the boots, the window and the operator. An
    #: unattributed record is refused, the same rule `EntryGate` applies to
    #: gate evidence: an unattributed pass is a claim, not evidence.
    source: str = ""
    taken_at: str = ""

    def __post_init__(self) -> None:
        if not str(self.stage).strip():
            raise StageMeasurementError("a stage measurement needs a stage name")
        if not str(self.reference).strip():
            raise StageMeasurementError(
                f"{self.stage!r}: a gain is a gain OVER something; the "
                f"reference stage has to be named or the number cannot be "
                f"compared with any other stage's."
            )
        if self.band_pct < 0.0 or self.flip_cost_s < 0.0:
            raise StageMeasurementError(
                f"{self.stage!r}: band_pct and flip_cost_s must be >= 0, got "
                f"{self.band_pct} / {self.flip_cost_s}. A negative band is not "
                f"a tighter measurement."
            )

    # -- the honest refusal path ---------------------------------------------
    @property
    def refusals(self) -> List[str]:
        """Why this record may NOT be used, one line each. Empty = usable.

        Checked here rather than at the point of use so that a record written
        by a bad pass is refused by every consumer identically, and so the
        `--show` listing can print the reason next to the row.
        """
        out: List[str] = []
        if not str(self.source).strip():
            out.append(
                "no source: an unattributed measurement is a claim, not "
                "evidence (the EntryGate rule, applied to numbers)"
            )
        if not str(self.rig_key).strip():
            out.append("no rig key: the record cannot be matched to a card set")
        if self.flip_samples < 1:
            out.append(
                "flip_cost_s was not instrumented (0 flips sampled): the cost "
                "of the move is the one term a controller must not assume, "
                "and an unpriced term must not read as a free one"
            )
        for label, covered in (
            ("reference", self.covered_s_reference),
            ("stage", self.covered_s_stage),
        ):
            if covered < MIN_MEASURE_SECONDS:
                out.append(
                    f"the {label} arm covers {covered:.1f} s of device time, "
                    f"below the {MIN_MEASURE_SECONDS:.0f} s measurement-run "
                    f"floor; below it the mean is a few rounds and the band is "
                    f"whichever round was slow"
                )
        if abs(self.gain_pct) <= self.band_pct:
            out.append(
                f"gain {self.gain_pct:+.2f} % does not clear its own band of "
                f"{self.band_pct:.2f} % (#360): a difference inside the band "
                f"is not a difference, however it is labelled"
            )
        return out

    @property
    def usable(self) -> bool:
        return not self.refusals

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data) -> "StageMeasurement":
        if not isinstance(data, dict):
            raise StageMeasurementError(
                f"a stage measurement must be a JSON object, got "
                f"{type(data).__name__}"
            )
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - fields)
        if unknown:
            raise StageMeasurementError(
                f"unknown field(s) {unknown} in a stage measurement record. "
                f"Refused rather than ignored: a record written by a newer "
                f"pass may mean something different by the fields this reader "
                f"does understand."
            )
        missing = sorted(
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.name not in data
        )
        if missing:
            raise StageMeasurementError(
                f"stage measurement is missing required field(s) {missing}"
            )
        return cls(**data)  # type: ignore[arg-type]

    def describe(self) -> str:
        state = "USABLE" if self.usable else f"REFUSED ({len(self.refusals)})"
        return (
            f"{self.stage} [{self.regime}] vs {self.reference}: "
            f"gain {self.gain_pct:+.2f} % band {self.band_pct:.2f} % "
            f"flip {self.flip_cost_s:.2f} s "
            f"({self.flip_samples} flip sample(s), "
            f"{self.covered_s_stage:.0f}/{self.covered_s_reference:.0f} s covered) "
            f"-- {state}"
        )


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class StageMeasurementLibrary:
    """The on-disk set of stage measurements, keyed by (rig, checkpoint, stage).

    LOOKUP NEVER FALLS BACK. `lookup` returns ``(record, reason)`` and the
    reason is filled on every miss: no record, a record from another card set,
    a record for another checkpoint, or a record whose own validation refuses
    it. Every one of those is a different remedy, and a consumer that receives
    ``None`` with no reason writes "not measured" over all four.
    """

    def __init__(self, records: Iterable[StageMeasurement] = ()):
        self._records: Dict[Tuple[str, str, str], StageMeasurement] = {}
        for rec in records:
            self.add(rec)

    # -- contents ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[StageMeasurement]:
        return sorted(
            self._records.values(), key=lambda r: (r.rig_key, r.model_key, r.stage)
        )

    def add(self, record: StageMeasurement) -> None:
        key = (record.rig_key, record.model_key, record.stage)
        self._records[key] = record

    # -- the seam the table consumes -----------------------------------------
    def lookup(
        self, stage: str, *, rig: str, model: str
    ) -> Tuple[Optional[StageMeasurement], str]:
        """``(record, reason)``. ``record`` is non-None only when usable."""
        exact = self._records.get((rig, model, stage))
        if exact is not None:
            refusals = exact.refusals
            if refusals:
                return None, (
                    f"stage {stage!r} HAS a measurement and it is refused: "
                    + "; ".join(refusals)
                )
            return exact, "measured"
        same_name = [r for r in self._records.values() if r.stage == stage]
        if not same_name:
            return None, (
                f"stage {stage!r} has never been measured on any rig. Take the "
                f"measurement with `python -m sglang.srt.planner."
                f"stage_measure_pass` (gain vs the incumbent, band from a "
                f"same-boot A-vs-A pair, flip cost from an instrumented flip); "
                f"until then it is a solved candidate, not a flip target."
            )
        other_rigs = sorted({r.rig_key for r in same_name if r.rig_key != rig})
        if other_rigs:
            return None, (
                f"stage {stage!r} was measured on a DIFFERENT card set "
                f"({', '.join(other_rigs)}); this rig is {rig}. A ms/round "
                f"measurement does not transfer between card sets -- #584 "
                f"measured borrowed rates being 12-22 % wrong after a power "
                f"target changed on the SAME cards. Re-measure here."
            )
        other_models = sorted({r.model_key for r in same_name if r.model_key != model})
        return None, (
            f"stage {stage!r} was measured on checkpoint(s) "
            f"{', '.join(other_models) or '<none>'}; this boot serves {model!r}. "
            f"A stage's gain is a property of the weights it moves, so the "
            f"record does not apply."
        )

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": ARTIFACT_VERSION,
            "written_at": _now_iso(),
            "measurements": [r.to_json() for r in self.records],
        }

    def save(self, path: Optional[str] = None) -> str:
        """Write the store atomically, staging through a PER-PROCESS name.

        The staging name carries the pid because the shared-staging idiom is
        exactly what corrupted the transient census on 2026-08-14: under pure
        TP every rank derives the same output path, three processes opened one
        ``.tmp`` with mode ``"w"``, and 23 of 600 concurrent flushes were lost.
        Write-tmp-then-rename makes the PUBLISH atomic; it never made the
        STAGING exclusive.
        """
        target = stage_measure_path(path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{target}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(self.to_json(), fh, indent=2, sort_keys=True)
            os.replace(tmp, target)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise StageMeasurementError(
                f"could not write the stage measurement store {target!r}: {exc}"
            ) from exc
        return target

    @classmethod
    def load(cls, path: Optional[str] = None) -> "StageMeasurementLibrary":
        """Read the store. A MISSING file is an empty library; a CORRUPT one
        raises.

        The asymmetry is the same one `regime_stages.load_gate_evidence` makes,
        and for the same reason: "not measured yet" and "measured and
        unreadable" have different remedies, and collapsing them lets the
        second read as the first.
        """
        target = stage_measure_path(path)
        if not os.path.exists(target):
            return cls()
        try:
            with open(target) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StageMeasurementError(
                f"the stage measurement store {target!r} exists and could not "
                f"be read as JSON ({exc}). Refused rather than treated as "
                f"absent: an unreadable store would read as 'never measured' "
                f"and silently re-open the refusal path."
            ) from exc
        if not isinstance(data, dict) or "measurements" not in data:
            raise StageMeasurementError(
                f"{target!r} is not a stage measurement store (no "
                f"'measurements' key)."
            )
        return cls(StageMeasurement.from_json(row) for row in data["measurements"])
