# SPDX-License-Identifier: Apache-2.0
"""#584 -- the card-rate MEASUREMENT PASS: put measured rates on disk, keyed by card.

WHY THIS MODULE EXISTS.

``--pp-solve-cut`` prices each pipeline stage from two measured numbers per
card -- a GEMM rate and a memory-bandwidth rate. ``server_args._pp_cut_card_rates``
looked them up in a :class:`CardLibrary`, and refused, correctly and loudly,
when a card carried none:

    --pp-solve-cut: the profile for 'NVIDIA GeForce RTX 5090' carries no
    measured gemm/bandwidth rate, so stage 0 cannot be priced.

That refusal fired on **every card, on every rig, on every boot**, because of a
gap that is easy to miss when reading either half on its own:

  * not one of the 16 ``SEED_CARDS`` carries ``gemm_tflops`` / ``membw_gbs``.
    They are curated nameplate entries; the measured fields are documented as
    filled "ONLY from a submission that carried a cached probe".
  * ``CardLibrary`` HAS working ``save(path)`` / ``load(path)``, and **nothing
    in the tree ever called either one**, on any branch.
  * the reason nothing called them is that neither takes a default and no code
    anywhere computed a path. The class had persistence with no LOCATION, so
    there was no file for ``load`` to read and no place for ``save`` to put
    one. A store nobody can name is a store nobody can fill.

So the gate was not merely uncalibrated, it was *uncalibratable*: the refusal
had no reachable remedy. #485's window found this by executing the path
(WINDOW_VERDICT_485_R12.md section 2, defect 3) and correctly declined to patch
around it -- inventing rates into ``SEED_CARDS`` would be exactly the "unpriced
term reads as free memory" failure the flag's own help text refuses.

WHAT THIS PASS DOES, AND WHAT IT DELIBERATELY DOES NOT.

It does **not** measure anything new. ``rigmon/card_probe.py`` (#213) already
measures precisely these two quantities on device, using
``uneven_perf._bench_gemm_tflops`` and ``_bench_membw_rates`` -- the same
kernels the boot path calibrates against, deliberately not a second opinion.
It already caches its result UUID-keyed under ``~/.cache/sglang``. Eighteen
planner sites consume it.

``_pp_cut_card_rates`` was the one consumer that did not: it constructed a
fresh seed-only ``CardLibrary()`` instead. The missing piece was never a probe.
It was a BRIDGE from the UUID-keyed measurement to the name-keyed catalog the
solver reads, plus the location that catalog was missing.

IDENTITY: UUID IS THE KEY, NAME IS ONLY A LOOKUP.

Rates are stored per UUID because a name is not an identity. This rig carries
**two RTX 3080s**, indistinguishable by ``props.name``, and a bare device index
is worse still: ``--rank-gpu-id 0,1,2`` puts stage 0 on NVML index 1 here,
because ``CUDA_VISIBLE_DEVICES`` is set by UUID and torch's enumeration order
is not NVML's. That is the documented device-order trap, and it is why
``registry/nvml.py``'s ``IdentityMap`` is the canonical bridge and is used here
to attach the PCI BDF alongside the UUID.

The solver, however, asks by NAME -- that is what the residency census records
per rank. Projecting many UUIDs onto one name therefore has to combine them,
and the combination is **the slowest measured instance wins**:

    gemm_tflops(name) = min over UUIDs of that name
    membw_gbs(name)   = min over UUIDs of that name

This is not caution for its own sake. A pipeline's makespan is set by its
PACER -- the slowest stage -- so pricing a name by its fastest instance
under-predicts the makespan of whichever stage lands on the slower card, and an
under-predicted makespan is how a cut gets admitted that should not have been.
The per-UUID numbers are kept in the artifact so the spread stays visible
rather than being averaged away.

THROTTLING IS RECORDED, NOT SILENTLY DROPPED.

A card measured while throttled reports a low rate. That does not make the
memory verdict unsafe -- a slow card yields a LONGER predicted makespan, so the
error is conservative on the axis the corridor law governs -- but it does move
which cut is chosen, so it is recorded as a caveat on the artifact and printed
by ``show``. #149's tagging convention: never a substitute number, always the
state it was measured in.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CARD_LIBRARY_BASENAME",
    "MeasuredCardRate",
    "CardRatePassReport",
    "card_library_path",
    "rates_by_uuid",
    "rates_by_name",
    "project_onto_library",
    "run_card_rate_pass",
    "load_measured_library",
]

#: The name of the store under the same cache directory ``card_probe`` uses.
#: Kept beside the probe on purpose: the library is a PROJECTION of the probe,
#: and a reader who finds one should find the other.
CARD_LIBRARY_BASENAME = "card_library.json"

#: Written into the artifact so a reader can tell which pass produced it.
ARTIFACT_VERSION = 1


def card_library_path(path: Optional[str] = None) -> str:
    """The canonical location of the measured card library.

    THE DEFECT THIS FUNCTION FIXES. ``CardLibrary.save``/``load`` require an
    explicit path and nothing computed one, so the persistence API was
    unreachable in practice. Resolution order:

      1. an explicit argument (tests, and an operator pointing at an artifact),
      2. ``SGLANG_CARD_LIBRARY``,
      3. ``~/.cache/sglang/card_library.json`` -- the directory
         ``rigmon/card_probe.py`` already owns.

    Returns a path; makes no promise that it exists. A missing file is a
    measurement that has not been taken, and the caller must refuse rather than
    default -- which is the whole point of #584.
    """
    if path:
        return str(path)
    try:
        from sglang.srt import environ as _environ

        env = _environ.envs.SGLANG_CARD_LIBRARY.get()
    except Exception:
        env = os.environ.get("SGLANG_CARD_LIBRARY") or None
    if env:
        return str(env)
    from sglang.srt.rigmon.card_probe import CACHE_DIR

    return os.path.join(CACHE_DIR, CARD_LIBRARY_BASENAME)


@dataclasses.dataclass(frozen=True)
class MeasuredCardRate:
    """One physical card's measured rates, keyed by the only stable identity.

    ``uuid`` is the key. ``pci_bus_id`` is carried as a second, human-checkable
    identity (it is what ``nvidia-smi`` prints and what survives a driver
    reload). ``name`` is a LOOKUP field, never an identity -- two cards of the
    same model share it.
    """

    uuid: str
    name: str
    gemm_tflops: Optional[float] = None
    membw_gbs: Optional[float] = None
    total_mib: Optional[int] = None
    pci_bus_id: Optional[str] = None
    nvml_index: Optional[int] = None
    throttled: bool = False

    @property
    def complete(self) -> bool:
        """Both rates present and positive. A zero is not a measurement."""
        return bool(self.gemm_tflops) and bool(self.membw_gbs)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CardRatePassReport:
    """What the pass did, in enough detail to argue with."""

    path: str
    rates: List[MeasuredCardRate] = dataclasses.field(default_factory=list)
    names_written: List[str] = dataclasses.field(default_factory=list)
    caveats: List[str] = dataclasses.field(default_factory=list)
    probe_source: str = ""
    wrote: bool = False

    @property
    def complete_rates(self) -> List[MeasuredCardRate]:
        return [r for r in self.rates if r.complete]

    def to_json(self) -> dict:
        return {
            "version": ARTIFACT_VERSION,
            "path": self.path,
            "probe_source": self.probe_source,
            "wrote": self.wrote,
            "names_written": list(self.names_written),
            "caveats": list(self.caveats),
            "rates_by_uuid": {r.uuid: r.to_json() for r in self.rates},
        }

    def format_text(self) -> str:
        lines = [f"card-rate pass -> {self.path}", f"  probe: {self.probe_source}"]
        for r in self.rates:
            bdf = f" {r.pci_bus_id}" if r.pci_bus_id else ""
            state = " THROTTLED" if r.throttled else ""
            gemm = f"{r.gemm_tflops:.2f}" if r.gemm_tflops else "-"
            bw = f"{r.membw_gbs:.1f}" if r.membw_gbs else "-"
            lines.append(
                f"  {r.uuid}{bdf}  {r.name}: "
                f"gemm {gemm} TFLOPS  membw {bw} GB/s{state}"
            )
        for c in self.caveats:
            lines.append(f"  CAVEAT: {c}")
        lines.append(
            f"  wrote {len(self.names_written)} name(s): "
            f"{', '.join(self.names_written) or '-'}"
        )
        return "\n".join(lines)


def _identity_by_uuid() -> Dict[str, Any]:
    """UUID -> CardIdentity from NVML, or {} when NVML is unavailable.

    Best-effort by design: the UUID key comes from the probe, so a rig without
    a usable NVML still produces a correctly keyed artifact -- it just carries
    no PCI BDF. Identity is never invented from an index here or anywhere else.
    """
    try:
        from sglang.srt.registry.nvml import identity_map

        return {c.uuid: c for c in identity_map().cards}
    except Exception:
        return {}


def rates_by_uuid(profile=None) -> Dict[str, MeasuredCardRate]:
    """Project the #213 card probe onto ``{uuid: MeasuredCardRate}``.

    Reuses ``card_probe.measured_card_rates`` rather than reading the probe's
    dataclasses directly, so this stays on the probe's own published contract.
    Returns ``{}`` when no probe is cached -- the signal to REFUSE, never to
    substitute a nameplate peak.
    """
    from sglang.srt.rigmon.card_probe import measured_card_rates

    measured = measured_card_rates(profile)
    if not measured:
        return {}

    ident = _identity_by_uuid()
    # The probe records each card's own VRAM total. Preferred over NVML here
    # because it travels WITH the measurement: an artifact carried to another
    # host still says how big the card it was measured on actually was.
    probe_cards = {}
    try:
        probe_cards = dict(getattr(profile, "by_uuid", {}) or {})
    except Exception:
        probe_cards = {}

    out: Dict[str, MeasuredCardRate] = {}
    for uuid, entry in measured.items():
        card = ident.get(uuid)
        probed = probe_cards.get(uuid)
        total = getattr(probed, "total_mib", None) or getattr(card, "total_mib", None)
        out[uuid] = MeasuredCardRate(
            uuid=uuid,
            name=str(entry.get("name") or (card.name if card else "")),
            gemm_tflops=entry.get("gemm_bf16_tflops"),
            membw_gbs=entry.get("membw_gbs"),
            total_mib=int(total) if total else None,
            pci_bus_id=getattr(card, "pci_bus_id", None),
            nvml_index=getattr(card, "nvml_index", None),
            throttled=bool(entry.get("throttled")),
        )
    return out


def rates_by_name(
    rates: Dict[str, MeasuredCardRate]
) -> Dict[str, MeasuredCardRate]:
    """Collapse per-UUID rates onto the name the solver looks up by.

    THE SLOWEST INSTANCE WINS. Two cards of one model do not measure
    identically, and the solver prices a STAGE by the card's name. A pipeline's
    makespan is set by its pacer, so taking the faster of two same-named cards
    would under-predict the makespan of the stage that lands on the slower one
    -- and an under-predicted makespan admits cuts that should have been
    refused. Taking the slower one can only over-predict, which refuses cuts
    that might have fit. That asymmetry is deliberate and is the same direction
    every other refusal in this gate points.

    Cards with an incomplete measurement are skipped entirely rather than
    contributing a None that would silently win a ``min``.
    """
    out: Dict[str, MeasuredCardRate] = {}
    for rate in rates.values():
        if not rate.complete:
            continue
        cur = out.get(rate.name)
        if cur is None:
            out[rate.name] = rate
            continue
        totals = [t for t in (cur.total_mib, rate.total_mib) if t]
        out[rate.name] = dataclasses.replace(
            cur,
            gemm_tflops=min(cur.gemm_tflops, rate.gemm_tflops),
            membw_gbs=min(cur.membw_gbs, rate.membw_gbs),
            # A capacity is a budget: crediting a name with the larger of two
            # same-named cards would over-fund whichever rank lands on the
            # smaller one.
            total_mib=min(totals) if totals else None,
            throttled=cur.throttled or rate.throttled,
            # The identity fields now describe a COMBINATION, not one card.
            # Blanked rather than left pointing at whichever card was seen
            # first, which would read as though the combined rate were that
            # card's measurement.
            uuid="",
            pci_bus_id=None,
            nvml_index=None,
        )
    return out


def project_onto_library(
    rates: Dict[str, MeasuredCardRate],
    library=None,
    total_mib_by_name: Optional[Dict[str, int]] = None,
):
    """Write measured rates into a :class:`CardLibrary`, returning (library, names).

    A name already in the library keeps its curated capacity and nameplate
    fields; only the measured rates are set, and they OVERWRITE -- a fresh
    measurement supersedes a stale one, which ``add(overwrite=False)`` would
    not do (it fills only absent fields).

    A name NOT in the library is added, which needs a VRAM total. It comes from
    NVML via the caller (``total_mib_by_name``); a card whose total is unknown
    is skipped rather than added with a fabricated capacity.
    """
    from sglang.srt.planner.card_library import CardLibrary, CardSpec

    lib = library if library is not None else CardLibrary()
    written: List[str] = []
    caveats: List[str] = []
    for name, rate in sorted(rates_by_name(rates).items()):
        measured_total = rate.total_mib or (total_mib_by_name or {}).get(name)
        if lib.has(name):
            cur = lib.get(name)
            # CAPACITY COLLISION, and it is not hypothetical. The driver
            # reports this rig's 20 GB cards as "NVIDIA GeForce RTX 3080",
            # which canonicalises to the same key as the seed entry for the
            # 10 GB RTX 3080 -- the seed set holds "RTX 3080" and
            # "RTX 3080 20GB" as deliberately distinct entries, but no name
            # the driver emits can select between them. Left alone, this rig's
            # measured rates would be written onto a profile claiming 10240
            # MiB. The measurement is the authority on the card it measured,
            # so the capacity is corrected to what was actually seen.
            total = int(measured_total) if measured_total else cur.total_mib
            if measured_total and cur.total_mib and int(measured_total) != cur.total_mib:
                caveats.append(
                    f"{name}: catalog capacity {cur.total_mib} MiB does not "
                    f"match the {int(measured_total)} MiB actually measured. "
                    f"The card name cannot distinguish these variants, so the "
                    f"MEASURED capacity wins and the profile now describes the "
                    f"card in this rig."
                )
            lib.add(
                dataclasses.replace(
                    cur,
                    total_mib=total,
                    gemm_tflops=float(rate.gemm_tflops),
                    membw_gbs=float(rate.membw_gbs),
                ),
                overwrite=True,
            )
            written.append(name)
            continue
        if not measured_total:
            caveats.append(
                f"{name}: not in the catalog and no measured VRAM total, so it "
                f"is skipped rather than added with a fabricated capacity."
            )
            continue
        lib.add(
            CardSpec(
                name=name,
                total_mib=int(measured_total),
                gemm_tflops=float(rate.gemm_tflops),
                membw_gbs=float(rate.membw_gbs),
            ),
            overwrite=True,
        )
        written.append(name)
    return lib, written, caveats


def _totals_by_name(rates: Dict[str, MeasuredCardRate]) -> Dict[str, int]:
    """VRAM totals from NVML, keyed by name, taking the SMALLEST per name.

    Same conservative direction as the rates: a capacity is a budget, and
    crediting a name with the larger of two same-named cards would over-fund
    whichever rank lands on the smaller one.
    """
    ident = _identity_by_uuid()
    out: Dict[str, int] = {}
    for rate in rates.values():
        card = ident.get(rate.uuid)
        total = rate.total_mib or getattr(card, "total_mib", None)
        if not total:
            continue
        prev = out.get(rate.name)
        out[rate.name] = int(total) if prev is None else min(prev, int(total))
    return out


def run_card_rate_pass(
    path: Optional[str] = None,
    profile=None,
    run_probe: bool = False,
    save: bool = True,
) -> CardRatePassReport:
    """THE PASS. Measure (or reuse a measurement), project, persist.

    ``run_probe=True`` runs the #213 probe on device -- roughly 25-30 s for
    three cards. Otherwise the cached probe is used, and its ABSENCE is
    reported as a refusal rather than filled in.
    """
    target = card_library_path(path)
    report = CardRatePassReport(path=target)

    if profile is None and run_probe:
        from sglang.srt.rigmon.card_probe import run_card_probe

        profile = run_card_probe()
        report.probe_source = "measured on device (#213 card probe, this pass)"
    elif profile is None:
        from sglang.srt.rigmon.card_probe import load_card_probe

        profile = load_card_probe()
        report.probe_source = (
            "cached #213 card probe" if profile is not None else "NO PROBE CACHED"
        )
    else:
        report.probe_source = "supplied profile"

    rates = rates_by_uuid(profile)
    report.rates = sorted(rates.values(), key=lambda r: (r.name, r.uuid))

    if not rates:
        report.caveats.append(
            "no card probe on disk, so no rate could be written. Run the pass "
            "with --run (or `python -m sglang.srt.rigmon.card_probe --run`) on "
            "the rig whose cards are being priced. A rate must be MEASURED; "
            "there is no nameplate fallback here by design."
        )
        return report

    incomplete = [r for r in report.rates if not r.complete]
    for r in incomplete:
        report.caveats.append(
            f"{r.name} ({r.uuid}) measured no complete rate pair "
            f"(gemm={r.gemm_tflops}, membw={r.membw_gbs}); it is NOT written, "
            f"so the gate will still refuse to price a stage on this card."
        )
    for r in report.rates:
        if r.throttled:
            report.caveats.append(
                f"{r.name} ({r.uuid}) was THROTTLED while measured, so its rate "
                f"is a floor rather than the card's capability. The memory "
                f"verdict stays conservative -- a slow card predicts a longer "
                f"makespan -- but the CHOICE of cut may differ from an "
                f"unthrottled rig. Re-run when the card is cool to remove this."
            )

    lib, written, projection_caveats = project_onto_library(
        rates, total_mib_by_name=_totals_by_name(rates)
    )
    report.names_written = written
    report.caveats.extend(projection_caveats)

    if save and written:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        lib.save(target)
        _write_sidecar(target, report, rates)
        report.wrote = True
    return report


def _write_sidecar(
    target: str, report: CardRatePassReport, rates: Dict[str, MeasuredCardRate]
) -> str:
    """Persist the UUID-keyed truth beside the name-keyed library.

    ``CardLibrary``'s own format is name-keyed, so saving it alone would DROP
    the per-card identity and the per-card spread between two same-named cards
    -- the exact information the device-order trap makes load-bearing. The
    sidecar keeps it, and it is what a later reader should audit; the library
    is the projection the solver consumes.
    """
    side = target + ".by-uuid.json"
    payload = report.to_json()
    payload["written_at"] = time.time()
    tmp = side + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.replace(tmp, side)
    return side


def load_measured_library(path: Optional[str] = None):
    """The library the solver should read, or None when no pass has been run.

    None means REFUSE. It must never be turned into a seed-only
    ``CardLibrary()``: that is precisely the substitution that made the gate
    refuse every card on every rig while looking like it had a catalog.
    """
    target = card_library_path(path)
    if not os.path.isfile(target):
        return None
    from sglang.srt.planner.card_library import CardLibrary

    try:
        return CardLibrary.load(target)
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sglang.srt.planner.card_rate_pass")
    ap.add_argument("--run", action="store_true",
                    help="measure on device now (#213 probe, ~25-30 s for 3 cards)")
    ap.add_argument("--path", default=None, help="library path (default: cache dir)")
    ap.add_argument("--show", action="store_true",
                    help="report what is on disk without measuring or writing")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.show:
        target = card_library_path(args.path)
        lib = load_measured_library(target)
        if lib is None:
            print(f"NO MEASURED LIBRARY at {target}. "
                  f"--pp-solve-cut will refuse every card until the pass runs.")
            return 1
        rated = [n for n in lib.names()
                 if lib.get(n).gemm_tflops and lib.get(n).membw_gbs]
        print(f"{target}: {len(rated)} of {len(lib.names())} profiles carry "
              f"measured rates")
        for n in rated:
            s = lib.get(n)
            print(f"  {n}: gemm {s.gemm_tflops:.2f} TFLOPS  "
                  f"membw {s.membw_gbs:.1f} GB/s")
        return 0 if rated else 1

    report = run_card_rate_pass(path=args.path, run_probe=args.run)
    print(json.dumps(report.to_json(), indent=1, sort_keys=True)
          if args.json else report.format_text())
    return 0 if report.wrote else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
