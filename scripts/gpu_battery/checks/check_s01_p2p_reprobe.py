#!/usr/bin/env python3
"""s01 check -- did the P2P re-probe actually measure, and is the result
consumable by the code that will have to consume it?

Three artifacts, three different questions:

capability_matrix.json
  * one row per ORDERED pair (peer writes into dst are what the dst's BAR
    window constrains, so the matrix is directed and both directions matter),
  * can_access_peer decided for every pair -- None means the probe did not
    run, which is not the same as False,
  * for every pair where peer access IS possible: the EFFECTIVE aperture
    fields are filled. This is the point of the whole step. The nominal
    256-MiB BAR1 figure is an upper bound, not a usability promise, and every
    downstream consumer is written to ignore it. A matrix with p2p=True and a
    null effective aperture would silently degrade those consumers to
    placeholders while looking like a successful run.

d2d_bench.json
  * a ladder per directed pair and mode,
  * the 255/256/257 MiB bracket present -- the knee at the window boundary is
    the measurement, and a ladder that steps over it cannot show one.

nccl_transport.json
  * every pair reached a verdict; a timeout row is a FAIL, not a datum.

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
    run_check,
)

STEP = "s01_p2p_reprobe"
MIB = 1024 * 1024
BRACKET_MIB = (255, 256, 257)


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
    pairs = cap.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise CheckFail("capability_matrix.json hat keine pairs")

    directed = set()
    p2p_pairs = []
    for i, row in enumerate(pairs):
        src, dst = row.get("src_pci"), row.get("dst_pci")
        if not src or not dst:
            raise CheckFail(f"capability_matrix pairs[{i}]: src_pci/dst_pci fehlt")
        directed.add((src, dst))
        if row.get("can_access_peer") is None:
            raise CheckFail(
                f"capability_matrix {src}->{dst}: can_access_peer ist None -- "
                "die Sonde ist nicht gelaufen (None ist nicht False)"
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
    bracket_seen = set()
    for i, row in enumerate(pairs):
        if not row.get("src_pci") or not row.get("dst_pci"):
            raise CheckFail(f"d2d_bench pairs[{i}]: src_pci/dst_pci fehlt")
        mode = row.get("mode")
        if mode not in ("direct", "staged"):
            raise CheckFail(f"d2d_bench pairs[{i}]: mode ist {mode!r}")
        modes.add(mode)
        points = row.get("points") or []
        if len(points) < 4:
            raise CheckFail(
                f"d2d_bench {row['src_pci']}->{row['dst_pci']} ({mode}): nur "
                f"{len(points)} Leiterpunkte"
            )
        for pt in points:
            size = pt.get("size_bytes")
            if isinstance(size, int) and size % MIB == 0 and size // MIB in BRACKET_MIB:
                bracket_seen.add(size // MIB)
    if "direct" not in modes or "staged" not in modes:
        raise CheckFail(
            f"d2d_bench: nur Modi {sorted(modes)} -- direkt gegen Host-Staging ist "
            "die Frage, ein Modus allein beantwortet sie nicht"
        )
    if not set(BRACKET_MIB) <= bracket_seen:
        raise CheckFail(
            f"d2d_bench: die 255/256/257-MiB-Klammer fehlt (gefunden: "
            f"{sorted(bracket_seen)}) -- der Knick an der Fenstergrenze ist die Messung"
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
                "Timeout oder Absturz ist kein Transportbefund"
            )
        if not row.get("transport_summary"):
            raise CheckFail(
                f"nccl_transport {row.get('pci_pair')}: leeres transport_summary, "
                "der NCCL_DEBUG-Grep hat nichts gefunden"
            )


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
