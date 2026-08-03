#!/usr/bin/env python3
"""#306 step 1 -- aggregate ratio_probe.py output into the verdict tables.

Emits markdown to stdout. Every number printed here traces to a line in
``results.jsonl``, which traces to a real file slice recorded in
``samples.json``.

Link rates are the MEASURED rows of the canonical tier table in
`docs/dev/DESIGN_407_memory_tier_registry.md` §1.1, cited inline; none is
invented. A cell whose rate has no measurement is reported ABSENT, per
`DESIGN_407_memtier_registry.md` §3.2 ("a row that did not succeed yields no
value", "a number is never re-labelled").

Decision rule: per (asset class, link) the winning method is the one that
maximises the SERIAL speedup, not the one with the best ratio -- a method that
compresses harder but decompresses slower can lose to a weaker, faster one.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# --- measured link rates, with provenance ---------------------------------
# (bytes/s, tier id, citation). Ordered slowest first: the slowest link is the
# most favourable case for compression, so a DEAD verdict there is DEAD
# everywhere.
LINKS: list[tuple[str, float, str]] = [
    (
        "T3 local NVMe / disk image, cold read",
        1.8e9,
        "`ANALYSE_389_nvme_expert_tier.md` §(b), `iflag=direct`, reproduced 3x; "
        "tier table `DESIGN_407_memory_tier_registry.md:135`",
    ),
    (
        "T4 remote rig-2 over 40G, NCCL-over-sockets",
        2.07e9,
        "`NOTE_453_remote_expert_lane.md:9-10` / `INTEGRATION_R3_VALIDATION.md:5053`; "
        "tier table `DESIGN_407_memory_tier_registry.md:137`",
    ),
    (
        "T4 remote rig-2 over 40G, staged RDMA 1 MiB",
        2.83e9,
        "tier table `DESIGN_407_memory_tier_registry.md:137`",
    ),
    (
        "T2 host RAM -> card, PCIe H2D pinned, gen4 x4",
        6.4e9,
        "`ANALYSE_393_ik_llama.md:301-304`; tier table "
        "`DESIGN_407_memory_tier_registry.md:136`",
    ),
    (
        "T2 host RAM -> card, PCIe H2D pinned, gen4 x8",
        13.0e9,
        "`ANALYSE_393_ik_llama.md:301-304`; tier table "
        "`DESIGN_407_memory_tier_registry.md:136`",
    ),
]

DEAD_RATIO = 1.08  # kill criterion, stated before the run


def med(xs):
    return st.median(xs) if xs else float("nan")


def required_ratio(link_bps: float, dec_bps: float) -> float:
    """Smallest ratio that makes a SERIAL decompress-after-transfer win.

    From  C/L + S/D < S/L  with  C = S/r:
        1/(rL) + 1/D < 1/L   <=>   r > D / (D - L)
    Impossible at any ratio when D <= L: decompression alone is already
    slower than sending the payload uncompressed.
    """
    if dec_bps <= link_bps:
        return float("inf")
    return dec_bps / (dec_bps - link_bps)


def speedup_serial(ratio: float, link_bps: float, dec_bps: float) -> float:
    return 1.0 / (1.0 / ratio + link_bps / dec_bps)


def speedup_pipelined(ratio: float, link_bps: float, dec_bps: float) -> float:
    return min(ratio, dec_bps / link_bps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/spinning/wt-306-ratio/.probe-data")
    args = ap.parse_args()
    data = Path(args.data)
    rows = [json.loads(ln) for ln in (data / "results.jsonl").read_text().splitlines() if ln.strip()]
    manifest = json.loads((data / "samples.json").read_text())

    by = defaultdict(list)
    for r in rows:
        by[(r["asset_class"], r["layout"], r["codec"])].append(r)
    classes = sorted({r["asset_class"] for r in rows})

    # ---- A. full ratio matrix -------------------------------------------
    print("### A. Ratio matrix -- median over 8 samples per class\n")
    print("Ratio = uncompressed / compressed; > 1 means the codec found something.\n")
    combos = sorted({(r["layout"], r["codec"]) for r in rows})
    layouts = sorted({ly for ly, _ in combos})
    codecs = sorted({cd for _, cd in combos})
    print("| asset class | layout | " + " | ".join(codecs) + " |")
    print("|---" * (len(codecs) + 2) + "|")
    for c in classes:
        for ly in layouts:
            cells = []
            any_cell = False
            for cd in codecs:
                v = by.get((c, ly, cd))
                if v:
                    any_cell = True
                    cells.append(f"{med([x['ratio'] for x in v]):.4f}")
                else:
                    cells.append("--")
            if any_cell:
                print(f"| `{c}` | {ly} | " + " | ".join(cells) + " |")

    # ---- B. decompress rates --------------------------------------------
    print("\n### B. Decompress rate matrix -- median MB/s (1e6 B/s), "
          "inverse permutation included\n")
    print("| asset class | layout | " + " | ".join(codecs) + " |")
    print("|---" * (len(codecs) + 2) + "|")
    for c in classes:
        for ly in layouts:
            cells, any_cell = [], False
            for cd in codecs:
                v = by.get((c, ly, cd))
                if v:
                    any_cell = True
                    cells.append(f"{med([x['decomp_mbs'] for x in v]):.0f}")
                else:
                    cells.append("--")
            if any_cell:
                print(f"| `{c}` | {ly} | " + " | ".join(cells) + " |")

    # ---- C. best ratio per class ----------------------------------------
    print("\n### C. Best achievable ratio per asset class (any method)\n")
    print(
        "| asset class | n | best method | ratio median | ratio min-max | "
        "decompress MB/s | compress MB/s | kill criterion (< 1.08) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    best_ratio: dict[str, dict] = {}
    for c in classes:
        cands = [
            (med([x["ratio"] for x in v]), ly, cd, v)
            for (cc, ly, cd), v in by.items()
            if cc == c
        ]
        cands.sort(reverse=True, key=lambda t: t[0])
        rmed, ly, cd, v = cands[0]
        ratios = [x["ratio"] for x in v]
        dec = med([x["decomp_mbs"] for x in v])
        comp = med([x["comp_mbs"] for x in v if x["comp_mbs"]]) if any(
            x["comp_mbs"] for x in v
        ) else float("nan")
        best_ratio[c] = {"ratio": rmed, "layout": ly, "codec": cd, "dec_bps": dec * 1e6}
        print(
            f"| `{c}` | {len(v)} | {ly}/{cd} | **{rmed:.4f}** | "
            f"{min(ratios):.4f}-{max(ratios):.4f} | {dec:.0f} | {comp:.0f} | "
            f"{'**DEAD**' if rmed < DEAD_RATIO else 'alive'} |"
        )

    # ---- D. per-link verdicts, method chosen to maximise the speedup ----
    print("\n### D. Cell verdicts -- serial speedup per (asset class, link)\n")
    print(
        "Each cell: the SERIAL speedup of the method that maximises it for that "
        "link (pipelined bound in brackets). The no-compression baseline is "
        "1.000x by definition, so > 1.000x is a win and < 1.000x means storing "
        "the asset RAW is strictly faster. The method is re-chosen per link, so "
        "a cell is the best this probe can do there, not the best-ratio method "
        "forced onto a link it does not suit.\n"
    )
    header = " | ".join(f"{n.split(',')[0]} {b / 1e9:.2f} GB/s" for n, b, _ in LINKS)
    print(f"| asset class | best ratio (any method) | {header} | verdict |")
    print("|---" * (len(LINKS) + 3) + "|")
    for c in classes:
        cells = []
        any_win = False
        for lname, lbps, _cite in LINKS:
            best = None
            for (cc, ly, cd), v in by.items():
                if cc != c:
                    continue
                r = med([x["ratio"] for x in v])
                d = med([x["decomp_mbs"] for x in v]) * 1e6
                s = speedup_serial(r, lbps, d)
                if best is None or s > best[0]:
                    best = (s, r, d, ly, cd)
            s, r, d, ly, cd = best
            sp = speedup_pipelined(r, lbps, d)
            any_win = any_win or s > 1.0
            cells.append(f"{s:.3f}x [{sp:.3f}x] {ly}/{cd}")
        dead = best_ratio[c]["ratio"] < DEAD_RATIO
        verdict = "**DEAD**" if dead and not any_win else ("WIN" if any_win else "no win")
        print(
            f"| `{c}` | {best_ratio[c]['ratio']:.4f} | " + " | ".join(cells) + f" | {verdict} |"
        )

    print("\n#### D.1 Required ratio `r_min = D/(D-L)` at the fastest decompress arm\n")
    print(
        "`r_min` is the smallest ratio that could make a serial win, given the "
        "decompress rate actually measured. Compare it against the best ratio "
        "column above.\n"
    )
    print("| asset class | fastest decompress arm | D (MB/s) | " + " | ".join(
        f"r_min @ {b / 1e9:.2f} GB/s" for _, b, _ in LINKS
    ) + " |")
    print("|---" * (len(LINKS) + 3) + "|")
    for c in classes:
        fastest = max(
            ((med([x["decomp_mbs"] for x in v]) * 1e6, ly, cd) for (cc, ly, cd), v in by.items() if cc == c),
        )
        d, ly, cd = fastest
        rs = []
        for _n, lbps, _c in LINKS:
            rm = required_ratio(lbps, d)
            rs.append("impossible" if rm == float("inf") else f"{rm:.3f}")
        print(f"| `{c}` | {ly}/{cd} | {d / 1e6:.0f} | " + " | ".join(rs) + " |")

    # ---- E. MT / chunked arms -------------------------------------------
    print("\n### E. Multi-thread and chunked-frame arms (raw layout)\n")
    print(
        "| asset class | zstd-3 1T comp MB/s | zstd-3 16T comp MB/s | "
        "zstd-19 1T comp MB/s | zstd-19 16T comp MB/s | "
        "zstd-3 1T decomp MB/s | zstd-3 4 MiB frames x8 decomp MB/s | frame-chunk ratio |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for c in classes:
        def g(codec, field):
            v = by.get((c, "raw", codec), [])
            xs = [x[field] for x in v if x.get(field) is not None]
            return f"{med(xs):.0f}" if xs else "--"

        print(
            f"| `{c}` | {g('zstd-3', 'comp_mbs')} | {g('zstd-3-mt16', 'comp_mbs')} | "
            f"{g('zstd-19', 'comp_mbs')} | {g('zstd-19-mt16', 'comp_mbs')} | "
            f"{g('zstd-3', 'decomp_mbs')} | {g('zstd-3-chunk4M-x8', 'decomp_mbs')} | "
            f"{med([x['ratio'] for x in by.get((c, 'raw', 'zstd-3-chunk4M-x8'), [])]):.4f} |"
        )

    # ---- F. provenance ---------------------------------------------------
    print("\n### F. Sample provenance\n")
    prov = defaultdict(list)
    for s in manifest["samples"]:
        prov[s["asset_class"]].append(s)
    print(
        "| asset class | n | bytes/sample | source file(s) | example tensor | "
        "format | block bytes |"
    )
    print("|---|---|---|---|---|---|---|")
    for c, ss in sorted(prov.items()):
        srcs = sorted({Path(s["source_path"]).name for s in ss})
        src = f"`{srcs[0]}`" if len(srcs) == 1 else f"{len(srcs)} files, e.g. `{srcs[0]}`"
        from blocks import LAYOUTS  # local import: report is usable without it

        bl = LAYOUTS.get(ss[0].get("block_layout") or "")
        print(
            f"| `{c}` | {len(ss)} | {ss[0]['n_bytes']} | {src} | "
            f"`{ss[0]['tensor']}` | {ss[0]['ggml_type']} | "
            f"{bl.block_bytes if bl else 'n/a (flat 1-byte elements)'} |"
        )

    # ---- G. link citations ----------------------------------------------
    print("\n### G. Link-rate provenance (all MEASURED, none invented)\n")
    print("| link | rate | source |")
    print("|---|---|---|")
    for lname, lbps, cite in LINKS:
        print(f"| {lname} | {lbps / 1e9:.2f} GB/s | {cite} |")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
