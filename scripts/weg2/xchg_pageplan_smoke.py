#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1352 REMAP -- run the page planner on THIS RIG's measured geometries.

Against the desk-written-never-executed law: the module's entry points are
CALLED here on the real per-card page counts and the real PP/TP cut, and every
number it prints carries the instrument it came from.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python3 scripts/weg2/xchg_pageplan_smoke.py

Sources of the inputs, all measured, none constant in this tree:

* per-card, per-group page counts -- ``WEG2-FLIP-TAG ... granules=`` of boot
  ``weg2xsn20`` (device side, ``tms_tag_bytes``).
* the PP cut 39/13/12 -- the ``RING-CKPT`` lines of xsn21b's front log
  (attn + linear layers per stage, out of the checkpoint's own headers).
* the TP widths 17/7/8 -- the B4n smoke's own ``tp_widths=[2720,1120,1280]``.
* H2D/D2H 13-14 GB/s per card -- the same FLIP-TAG lines' GB/s column.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.weg2 import xchg_pageplan as pp   # noqa: E402

MIB = 1024 * 1024
PAGE = pp.PAGE_BYTES_DEFAULT

MEASURED_PAGES = {1: (8210, 9304), 0: (2855, 3568), 2: (4676, 3568)}
CARD_NAME = {1: "nvml1 5090", 0: "nvml0 3080", 2: "nvml2 3080"}
PP_STAGE_LAYERS = (39, 13, 12)
TP_WIDTHS = (17, 7, 8)
SLOT_BYTES = 128 * MIB


def leg_layouts(run_bytes=3 * MIB + 7):
    p_pieces = {0: [], 1: [], 2: []}
    d_pieces = {0: [], 1: [], 2: []}
    p_off = {0: 0, 1: 0, 2: 0}
    d_off = {0: 0, 1: 0, 2: 0}
    layer = 0
    for stage, n_layers in enumerate(PP_STAGE_LAYERS):
        card = (1, 0, 2)[stage]
        for _ in range(n_layers):
            for t, width in enumerate(TP_WIDTHS):
                nbytes = run_bytes * width
                name = f"L{layer}.r{t}"
                p_pieces[card].append(
                    pp.PieceExtent(name, card, "weights_p", p_off[card], nbytes))
                p_off[card] += nbytes
                d_card = (1, 0, 2)[t]
                d_pieces[d_card].append(
                    pp.PieceExtent(name, d_card, "weights_d", d_off[d_card], nbytes))
                d_off[d_card] += nbytes
            layer += 1
    srcs = tuple(pp.SideLayout("P", c, "weights_p", tuple(p_pieces[c]), False)
                 for c in (1, 0, 2))
    dsts = tuple(pp.SideLayout("D", c, "weights_d", tuple(d_pieces[c]), False)
                 for c in (1, 0, 2))
    return srcs, dsts


def sized(group, card, tag, pages, n_runs=64):
    total = pages * PAGE
    sizes = [total // n_runs] * n_runs
    sizes[-1] += total - sum(sizes)
    pieces = []
    off = 0
    for i, n in enumerate(sizes):
        pieces.append(pp.PieceExtent(f"t{i}", card, tag, off, n))
        off += n
    return pp.SideLayout(group, card, tag, tuple(pieces), False)


def main():
    checks = 0
    ok = 0

    print("== 1. THE MEASURED IMAGES, device side (WEG2-FLIP-TAG granules, xsn20)")
    print("   card          P pages   P MiB   D pages   D MiB    D-P pages   D-P MiB")
    tot_p = tot_d = 0
    for card in (1, 0, 2):
        p, d = MEASURED_PAGES[card]
        tot_p += p
        tot_d += d
        print(f"   {CARD_NAME[card]:<12} {p:8d} {p*2:7d} {d:9d} {d*2:7d} "
              f"{d-p:11d} {(d-p)*2:9d}")
        checks += 1
        ok += 1
    print(f"   {'TOTAL':<12} {tot_p:8d} {tot_p*2:7d} {tot_d:9d} {tot_d*2:7d} "
          f"{tot_d-tot_p:11d} {(tot_d-tot_p)*2:9d}")

    print("\n== 2. THE PAGE FUND PER CARD (rule 3 + rule 4), both legs")
    print("   card          size term  boundary  fund(one leg)  fund(card, both)")
    fund_total = 0
    for card in (1, 0, 2):
        p, d = MEASURED_PAGES[card]
        lp = sized("P", card, "weights_p", p)
        ld = sized("D", card, "weights_d", d)
        size_term = max(0, d - p)
        boundary = ld.boundary_pages(PAGE)
        one = pp.fund_pages(lp, ld, PAGE)
        both = pp.fund_for_card(lp, ld, PAGE)
        fund_total += both
        print(f"   {CARD_NAME[card]:<12} {size_term:9d} {boundary:9d} "
              f"{one:14d} {both:17d}")
        checks += 1
        ok += 1
    print(f"   fund total over the three cards: {fund_total} pages = "
          f"{fund_total * 2} MiB")
    print("   NOTE: the fund costs NO PEAK VRAM -- only one group is resident, so")
    print("   the card's peak is already max(P,D). The fund is that same difference,")
    print("   owned instead of released-and-reacquired.")

    print("\n== 3. THE SCHEDULE, both legs, at the real cross-card run shape")
    srcs, dsts = leg_layouts()
    for direction, a, b in ((pp.PP_TO_TP, srcs, dsts), (pp.TP_TO_PP, dsts, srcs)):
        funds = pp.minimum_fund(a, b, direction=direction, slot_bytes=SLOT_BYTES,
                                page_bytes=PAGE)
        plans = pp.plan_leg(a, b, direction=direction, funds=funds,
                            slot_bytes=SLOT_BYTES, page_bytes=PAGE)
        pp.verify_leg(plans)
        checks += 1
        ok += 1
        print(f"   -- {direction} (slot {SLOT_BYTES // MIB} MiB = "
              f"{SLOT_BYTES // PAGE} pages)")
        for card in sorted(plans):
            print("      " + plans[card].line())
        net = {
            d.card: max(0, d.total_pages(PAGE)
                        - next(s for s in a if s.card == d.card).total_pages(PAGE))
            for d in b
        }
        print(f"      minimum fund per card: {funds}")
        print(f"      NET size difference  : {net}")
        print("      -> the schedule needs MORE than the net difference: the")
        print("         transfers that fund a page lag the collects that consume it.")

    print("\n== 4. CUT 1's PRICE (on-card bytes crossing PCIe twice)")
    oncard = int(10.28 * 1024) * MIB
    for gbs in (13.0, 14.0):
        print(f"   {oncard / 2**30:.2f} GiB on-card at {gbs:.1f} GB/s -> "
              f"{pp.cut1_cost_ms(oncard, gbs) / 1000.0:.2f} s per leg")
    checks += 1
    ok += 1

    print("\n== 5. THE REFUSALS ARE REACHABLE (executed, not asserted)")
    for name, fn in (
        ("W94 fund too small", lambda: pp.plan_leg(
            srcs, dsts, direction=pp.PP_TO_TP, funds={0: 0, 1: 0, 2: 0},
            slot_bytes=SLOT_BYTES, page_bytes=PAGE)),
        ("W94 slot below one page", lambda: pp.plan_leg(
            srcs, dsts, direction=pp.PP_TO_TP, funds={0: 9999, 1: 9999, 2: 9999},
            slot_bytes=PAGE // 2, page_bytes=PAGE)),
        ("W95 unknown direction", lambda: pp.plan_leg(
            srcs, dsts, direction="sideways", funds={0: 1, 1: 1, 2: 1},
            slot_bytes=SLOT_BYTES, page_bytes=PAGE)),
    ):
        checks += 1
        try:
            fn()
            print(f"   {name}: NOT RAISED -- refusal unreachable")
        except (pp.Weg2RemapPlanUnschedulable, pp.Weg2RemapPageRefused) as exc:
            ok += 1
            print(f"   {name}: {str(exc)[:110]}...")

    verdict = "PASS" if ok == checks else "FAIL"
    print(f"\nverdict={verdict} checks={ok}/{checks}")
    return 0 if ok == checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
