#!/usr/bin/env python3
"""s01 check -- did the P2P re-probe actually measure, and is the result
consumable by the code that will have to consume it?

Three artifacts, three different questions:

capability_matrix.json  (rows under "directed_pairs", the producer's key)
  * one row per ORDERED pair (peer writes into dst are what the dst's BAR
    window constrains, so the matrix is directed and both directions matter),
  * can_access_peer decided for every pair -- None means the probe did not
    run, which is not the same as False,
  * the dst BAR1 classification is filled for every row. It is what makes a
    directed row readable at all: "windowed" and "full" are different
    hardware situations, and a row that does not say which one it describes
    cannot be compared against the other direction,
  * for every pair where peer access IS possible: the EFFECTIVE aperture
    fields are filled. This is the point of the whole step. The nominal
    256-MiB BAR1 figure is an upper bound, not a usability promise, and every
    downstream consumer is written to ignore it. A matrix with p2p=True and a
    null effective aperture would silently degrade those consumers to
    placeholders while looking like a successful run.

d2d_bench.json
  * a ladder per directed pair and mode,
  * the 255/256/257 MiB bracket present IN EVERY ladder -- the knee at the
    window boundary is the measurement, and a ladder that steps over it
    cannot show one,
  * the pressure arms (#278 methodology: bidir, dual-window) ran and every
    leg carries a verdict.

nccl_transport.json
  * every pair reached a verdict; a timeout or a crashed rank is a FAIL, not
    a datum.

Field names here are taken from the artifacts the producers in
scripts/p2p_readiness/ actually write, not from what a reader would expect
them to write. A check that asserts an imagined schema fails on every healthy
run and passes on none, which is worse than no check at all.

And then the real gate: the artifacts are loaded with
htccl_path_rates.load_p2p_capability_matrix / load_p2p_d2d_bench -- the SAME
loaders #279 will use. A file that parses in a bespoke check but yields zero
profiles in the consumer is a file that has not helped anyone.

NOT judged: whether P2P engaged. "No P2P anywhere" is a legitimate, fully
recorded outcome. Filling verdict_diff.md is the reader's job.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    add_repo_to_path,
    load_json,
    require_envelope,
    require_number,
    run_check,
)

STEP = "s01_p2p_reprobe"
MIB = 1024 * 1024
BRACKET_MIB = (255, 256, 257)

# The key capability_matrix.py writes its ordered rows under. Named here once,
# spelled the way the producer spells it: the matrix is DIRECTED, and the key
# says so. (scripts/p2p_readiness/capability_matrix.py, "directed_pairs")
CAPABILITY_ROWS_KEY = "directed_pairs"
# classify_bar() emits exactly these two plus "unknown" when the nominal BAR1
# size could not be resolved at all -- and "unknown" is a gap, not a class.
BAR_CLASSES = ("windowed", "full")


def check(step_dir: str) -> None:
    results = os.path.join(step_dir, "results")
    if not os.path.isdir(results):
        raise CheckStop(
            f"kein results-Verzeichnis ({results}) -- run_all.sh ist nicht gelaufen"
        )

    cap = load_json(
        os.path.join(results, "capability_matrix.json"), "capability_matrix.json"
    )
    d2d = load_json(os.path.join(results, "d2d_bench.json"), "d2d_bench.json")
    nccl = load_json(
        os.path.join(results, "nccl_transport.json"), "nccl_transport.json"
    )

    require_envelope(cap, "capability_matrix", "capability_matrix.json")
    require_envelope(d2d, "d2d_bench", "d2d_bench.json")
    require_envelope(nccl, "nccl_transport", "nccl_transport.json")

    any_p2p = _check_capability(cap)
    _check_d2d(d2d)
    _check_nccl(nccl)
    _check_loadable(cap, d2d, any_p2p)


def _check_capability(cap: dict) -> bool:
    pairs = cap.get(CAPABILITY_ROWS_KEY)
    if not isinstance(pairs, list) or not pairs:
        raise CheckFail(
            f"capability_matrix.json hat keine {CAPABILITY_ROWS_KEY} "
            f"(Top-Level-Schluessel: {sorted(cap)})"
        )

    directed = set()
    p2p_pairs = []
    for i, row in enumerate(pairs):
        where = f"capability_matrix {CAPABILITY_ROWS_KEY}[{i}]"
        src, dst = row.get("src_pci"), row.get("dst_pci")
        if not src or not dst:
            raise CheckFail(f"{where}: src_pci/dst_pci fehlt")
        directed.add((src, dst))
        if row.get("can_access_peer") is None:
            raise CheckFail(
                f"capability_matrix {src}->{dst}: can_access_peer ist None -- "
                "die Sonde ist nicht gelaufen (None ist nicht False)"
            )
        # The dst BAR classification travels with the DIRECTED row because the
        # constraint is the target's window, not the source's.
        klass = row.get("dst_bar1_classification")
        if klass not in BAR_CLASSES:
            raise CheckFail(
                f"capability_matrix {src}->{dst}: dst_bar1_classification ist "
                f"{klass!r}, erwartet eine von {BAR_CLASSES} -- ohne sie ist die "
                "Zeile nicht gegen die Gegenrichtung lesbar"
            )
        nominal = row.get("dst_bar1_nominal_bytes")
        if not isinstance(nominal, int) or isinstance(nominal, bool) or nominal <= 0:
            raise CheckFail(
                f"capability_matrix {src}->{dst}: dst_bar1_nominal_bytes ist "
                f"{nominal!r} -- die Klassifikation {klass!r} haengt an dieser Zahl"
            )
        if not isinstance(row.get("probe_errors"), list):
            raise CheckFail(
                f"capability_matrix {src}->{dst}: probe_errors fehlt oder ist keine "
                "Liste -- Sondenfehler sind Messergebnisse und muessen dastehen"
            )
        if row.get("can_access_peer"):
            p2p_pairs.append(row)

    # Directed means both orderings: an asymmetric rig (full-BAR 5090 vs
    # windowed 3080) is exactly the case where one direction is not the other.
    for src, dst in list(directed):
        if (dst, src) not in directed:
            raise CheckFail(
                f"capability_matrix: Paar {dst}->{src} fehlt, die Matrix ist nicht "
                "gerichtet vollstaendig"
            )

    # The per-row classification is a copy of the target card's; if the two
    # disagree the survey and the matrix were built from different states.
    by_pci = {
        d.get("pci_bus_id"): d
        for d in (cap.get("devices") or [])
        if isinstance(d, dict)
    }
    for row in pairs:
        dev = by_pci.get(row["dst_pci"])
        if dev is None:
            raise CheckFail(
                f"capability_matrix: dst {row['dst_pci']} kommt in devices[] nicht "
                "vor -- Matrix und Kartenerhebung passen nicht zusammen"
            )
        if dev.get("bar1_classification") != row["dst_bar1_classification"]:
            raise CheckFail(
                f"capability_matrix {row['src_pci']}->{row['dst_pci']}: Zeile sagt "
                f"{row['dst_bar1_classification']!r}, devices[] sagt "
                f"{dev.get('bar1_classification')!r}"
            )

    for row in p2p_pairs:
        src, dst = row["src_pci"], row["dst_pci"]
        for field in (
            "effective_max_single_copy_bytes",
            "effective_max_region_chunked_bytes",
        ):
            value = row.get(field)
            if value is None:
                raise CheckFail(
                    f"capability_matrix {src}->{dst}: {field} ist None, obwohl "
                    "can_access_peer True -- die effektive Apertur wurde nicht gemessen"
                )
            if not isinstance(value, int) or value < 0:
                raise CheckFail(
                    f"capability_matrix {src}->{dst}: {field} ist {value!r}"
                )
        if row["effective_max_single_copy_bytes"] == 0 and not row.get("probe_errors"):
            raise CheckFail(
                f"capability_matrix {src}->{dst}: effektive Apertur 0 ohne einen "
                "einzigen probe_error -- das ist kein Messergebnis, sondern eine Luecke"
            )

    return bool(p2p_pairs)


def _check_d2d(d2d: dict) -> None:
    pairs = d2d.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise CheckFail("d2d_bench.json hat keine pairs")
    modes = set()
    for i, row in enumerate(pairs):
        if not row.get("src_pci") or not row.get("dst_pci"):
            raise CheckFail(f"d2d_bench pairs[{i}]: src_pci/dst_pci fehlt")
        mode = row.get("mode")
        if mode not in ("direct", "staged"):
            raise CheckFail(f"d2d_bench pairs[{i}]: mode ist {mode!r}")
        modes.add(mode)
        label = f"d2d_bench {row['src_pci']}->{row['dst_pci']} ({mode})"
        points = row.get("points") or []
        if len(points) < 4:
            raise CheckFail(f"{label}: nur {len(points)} Leiterpunkte")
        bracket_seen = set()
        for j, pt in enumerate(points):
            size = pt.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise CheckFail(f"{label}: points[{j}].size_bytes ist {size!r}")
            if size % MIB == 0 and size // MIB in BRACKET_MIB:
                bracket_seen.add(size // MIB)
            status = pt.get("status")
            if status not in ("ok", "failed"):
                raise CheckFail(f"{label}: points[{j}].status ist {status!r}")
            # A failed point is a RESULT (above the effective aperture); it must
            # say why. An ok point must carry the number it was measured for.
            if status == "failed":
                if not pt.get("error"):
                    raise CheckFail(
                        f"{label}: points[{j}] ist failed ohne error -- ein Punkt "
                        "ohne Grund ist keine Messung"
                    )
            else:
                require_number(pt.get("median_s"), f"{label}: points[{j}].median_s")
        # Per ladder, not globally: one pair carrying the bracket says nothing
        # about the pair that stepped over it.
        if not set(BRACKET_MIB) <= bracket_seen:
            raise CheckFail(
                f"{label}: die 255/256/257-MiB-Klammer fehlt (gefunden: "
                f"{sorted(bracket_seen)}) -- der Knick an der Fenstergrenze ist "
                "die Messung"
            )
    if "direct" not in modes or "staged" not in modes:
        raise CheckFail(
            f"d2d_bench: nur Modi {sorted(modes)} -- direkt gegen Host-Staging ist "
            "die Frage, ein Modus allein beantwortet sie nicht"
        )
    _check_d2d_arms(d2d)


def _check_d2d_arms(d2d: dict) -> None:
    """The #278 pressure arms. A ladder measured one copy at a time cannot see
    what two simultaneous copies do to the same window, which is the entire
    reason the arms exist -- so a run without them is a run that answered the
    easier question."""
    arms = d2d.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CheckFail(
            "d2d_bench.json hat keine arms -- die Druckarme (bidir/dual-window) "
            "sind die einzige Messung unter gleichzeitiger Aperturlast"
        )
    for i, arm in enumerate(arms):
        kind = arm.get("kind")
        if not kind:
            raise CheckFail(f"d2d_bench arms[{i}]: kind fehlt")
        legs = arm.get("legs")
        if not isinstance(legs, list) or len(legs) < 2:
            raise CheckFail(
                f"d2d_bench arms[{i}] ({kind}): {len(legs or [])} Leg(s) -- ein Arm "
                "mit weniger als zwei gleichzeitigen Kopien ist kein Druckarm"
            )
        for j, leg in enumerate(legs):
            status = leg.get("status")
            if status not in ("ok", "failed"):
                raise CheckFail(
                    f"d2d_bench arms[{i}] ({kind}) legs[{j}]: status ist {status!r}"
                )
            if status == "ok":
                require_number(
                    leg.get("median_s"),
                    f"d2d_bench arms[{i}] ({kind}) legs[{j}]: median_s",
                )


def _check_nccl(nccl: dict) -> None:
    pairs = nccl.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise CheckFail("nccl_transport.json hat keine pairs")
    for row in pairs:
        status = row.get("status")
        if status != "ok":
            raise CheckFail(
                f"nccl_transport {row.get('pci_pair')}: status {status!r} -- ein "
                f"Timeout oder Absturz ist kein Transportbefund; "
                f"{_nccl_cause(row)}"
            )
        if not row.get("transport_summary"):
            raise CheckFail(
                f"nccl_transport {row.get('pci_pair')}: leeres transport_summary, "
                "der NCCL_DEBUG-Grep hat nichts gefunden"
            )


# Lines in the NCCL/torch log that name a cause rather than a consequence.
_NCCL_CAUSE_MARKERS = (
    "NCCL WARN",
    "ncclInvalidUsage",
    "ncclUnhandledCudaError",
    "ncclSystemError",
    "DistBackendError",
    "Error",
)


def _nccl_cause(row: dict) -> str:
    """One line out of log_tail that names WHY the pair failed.

    The bugfixer gets a cause, not "go read a 4000-character tail". The tail
    itself never enters anyone's context.
    """
    tail = row.get("log_tail") or ""
    for marker in _NCCL_CAUSE_MARKERS:
        for line in tail.splitlines():
            if marker in line:
                return f"Ursache laut log_tail: {line.strip()[:180]}"
    return "log_tail nennt keine Ursache"


def _check_loadable(cap: dict, d2d: dict, any_p2p: bool) -> None:
    add_repo_to_path()
    try:
        from sglang.srt.distributed.device_communicators.htccl_path_rates import (
            load_p2p_capability_matrix,
            load_p2p_d2d_bench,
        )
    except Exception as exc:  # an unimportable consumer is an env problem
        raise CheckStop(f"htccl_path_rates nicht importierbar: {exc}") from exc

    cap_res = load_p2p_capability_matrix(cap)
    if cap_res.errors:
        raise CheckFail(
            f"capability_matrix vom #279-Lader abgelehnt: {cap_res.errors[0]}"
        )
    # Apertures only exist where peer access does. A rig where NCCL picks no
    # P2P for any pair is a fully recorded outcome -- demanding an aperture
    # there would turn the honest answer into a failure.
    if any_p2p and not cap_res.apertures:
        raise CheckFail(
            "capability_matrix ergibt beim #279-Lader NULL Aperturen, obwohl "
            "Peer-Zugriff gemessen wurde -- nur nominale Zeilen werden "
            "uebersprungen, damit bleibt der Dispatcher auf Platzhaltern"
        )

    d2d_res = load_p2p_d2d_bench(d2d)
    if d2d_res.errors:
        raise CheckFail(f"d2d_bench vom #279-Lader abgelehnt: {d2d_res.errors[0]}")
    if not d2d_res.profiles:
        raise CheckFail("d2d_bench ergibt beim #279-Lader NULL Profile")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
