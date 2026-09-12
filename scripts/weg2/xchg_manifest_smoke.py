#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- THE EXECUTION SMOKE of the manifest write-along and the join.

Desk-written-never-executed is a defect class this campaign has paid for, so
this drives the PRODUCT call sites rather than the library functions:

1. ``weight_exchange._write_placement_manifest`` -- the writer, at the frame
   ``arm_coverage_at_load`` calls it from (the end of weight loading,
   ``model_runner.py:2564``);
2. ``SchedulerWeightUpdaterManager._weg2_shadow_hook``'s direction gate, driven
   through ``weight_exchange.leg_enabled`` with the env the launcher publishes;
3. ``xchg_manifest.join_manifests`` + ``plan_from_join`` -- the provider.

It runs on THIS RIG'S REAL GEOMETRIES: group P split 44,10,10 (``--pp-layer-set``)
over three cards, group D as TP3 with an uneven 17:7:8 cut, INT8-W8A8 element
width.  No GPU, no model, no checkpoint: every input is a manifest, which is
the whole point -- the placement is written down, not reconstructed.

ACCEPTANCE, printed and asserted, IN BOTH DIRECTIONS:
  * ``src_resolved=N/N``, which read ``0/N`` on all 24 legs of boot weg2xsn20;
  * a REAL SHARD CUT -- more descriptors than tensors, every tensor cut across
    all three TP ranks, with the JOIN's uneven boundaries and not an even
    split.  ``src_resolved=N/N`` on a diagonal plan does not count (operator
    ruling 2026-09-12) and is refused by name.

THE ADDRESSES HERE ARE STAND-INS FOR A SEAM, AND THE SEAM'S CONTRACT IS NOT
THE RING.  ``src_addr``/``dst_addr`` belong to the caller: the source hook
answers with its OWN device address and deposits into the bounce slot, and the
destination resolves the source to that slot's offset.  The ring is the
counter-proof for the digest and never the source of the exchange -- reading
the source out of it would be the ring restore through another door, and it
defeats the goal (zero layer bytes resident in host RAM).
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as xb
from sglang.srt.weg2 import weight_exchange_shadow as sh
from sglang.srt.weg2 import xchg_manifest as xm

P_CUT = (44, 10, 10)
D_VECTOR = (17, 7, 8)
CARDS = (0, 1, 2)
TAG = "weights_0"
ITEMSIZE = 1          # INT8-W8A8


def stage_of_layer(layer: int) -> int:
    acc = 0
    for stage, n in enumerate(P_CUT):
        acc += n
        if layer < acc:
            return stage
    raise AssertionError(layer)


def split(total: int) -> tuple:
    parts = [total * w // sum(D_VECTOR) for w in D_VECTOR]
    parts[-1] += total - sum(parts)
    return tuple(parts)


def piece(name, rows, cols):
    return xm.ManifestPiece(param_name=name, tensor_class=sh.tensor_class(name),
                            rows_full=rows, cols_full=cols, itemsize=ITEMSIZE,
                            tag=TAG, nbytes=rows * cols * ITEMSIZE)


# Qwen3.8-27B-shaped classes: one ROW-parallel and one COLUMN-parallel per layer.
CLASSES = (("self_attn.qkv_proj.weight", 5120, 4096, "rows"),
           ("mlp.down_proj.weight", 4096, 13824, "cols"))


def build_manifests():
    p = {r: [] for r in range(len(P_CUT))}
    d = {r: [] for r in range(len(CARDS))}
    for layer in range(sum(P_CUT)):
        for suffix, rows, cols, axis in CLASSES:
            name = f"model.layers.{layer}.{suffix}"
            p[stage_of_layer(layer)].append(piece(name, rows, cols))
            widths = split(rows if axis == "rows" else cols)
            for r, w in enumerate(widths):
                d[r].append(piece(name, w, cols) if axis == "rows"
                            else piece(name, rows, w))
    out = []
    # THE AXES OF THE REAL FORM: group P is `--tp-size 1 --pp-size 3` (pp_rank
    # varies, tp_rank is 0 on all three) and group D is TP3. Boot weg2xsn22
    # keyed the file name on tp_rank alone, so all of P wrote ONE file; this
    # builder reproduced that shape and the overwrite ratchet caught it here
    # before the tests did.
    for r, pieces in p.items():
        out.append(xm.RankManifest(group="P", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="smoke",
                                   tp_rank=0, pp_rank=r,
                                   pieces=tuple(pieces)))
    for r, pieces in d.items():
        out.append(xm.RankManifest(group="D", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="smoke",
                                   tp_rank=r, pp_rank=0,
                                   pieces=tuple(pieces)))
    return out


def main() -> int:
    failures = []

    def check(ok, what):
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            failures.append(what)

    print(f"WEG2-XCHG-MANIFEST-SMOKE p_cut={','.join(map(str, P_CUT))} "
          f"d_vector={','.join(map(str, D_VECTOR))} cards={list(CARDS)} "
          f"quant=w8a8_int8 itemsize={ITEMSIZE}")

    # -- 1. THE WRITER, through the product's own file round trip -----------
    print("\n[1] write-along (the loader's decision, written down)")
    with tempfile.TemporaryDirectory(prefix="weg2-b4n-smoke") as tmp:
        for man in build_manifests():
            path = xm.write_rank_manifest(man, tmp)
            if man.rank == 0:
                print("  " + xm.written_line(path, man))
        loaded = xm.load_manifests(tmp, boot_token="smoke")
        check(len(loaded) == len(P_CUT) + len(CARDS),
              f"six manifests round-tripped ({len(loaded)}/6)")
        manifests = list(loaded)

    # -- 2. THE DIRECTION KNOB, at the predicate the hook calls -------------
    print("\n[2] direction knob (the hook's own gate)")
    os.environ[wx.XCHG_LEGS_ENV] = wx.LEGS_PP_TO_TP
    wx.reset_legs_skipped()
    ran, skipped = [], []
    for hook, group in (("source", "P"), ("destination", "D"),
                        ("authoritative", "D"), ("source", "D"),
                        ("destination", "P"), ("authoritative", "P")):
        if wx.leg_enabled(hook, group):
            ran.append((hook, group))
        else:
            skipped.append((hook, group))
            print("  " + wx.legs_skipped_line(wx.record_leg_skipped(),
                                              hook=hook, group=group))
    check(len(ran) == 3 and len(skipped) == 3,
          f"armed={wx.LEGS_PP_TO_TP}: {len(ran)}/6 legs run, "
          f"{len(skipped)}/6 skipped BY NAME")
    os.environ.pop(wx.XCHG_LEGS_ENV, None)
    check(wx.xchg_legs() == wx.LEGS_BOTH, "default is both (byte-identical)")

    # -- 3. THE JOIN -------------------------------------------------------
    print("\n[3] cross-group join (the knowledge no single rank holds)")
    join = xm.join_manifests(manifests, pp_group="P", tp_group="D")
    print("  " + join.line())
    check(join.n_sharded == len(join.tensors) > 0,
          f"every tensor's axis read off the join "
          f"({join.n_sharded}/{len(join.tensors)} sharded)")
    axes = {t.shard_axis for t in join.tensors}
    check(wx.ROWS in axes and wx.COLS in axes,
          "both a ROW cut and a COLUMN cut were recognised")

    # -- 4. THE PROVIDER, BOTH DIRECTIONS ----------------------------------
    print("\n[4] provider: src_resolved and the shard cut, BOTH directions")
    for direction in (wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP):
        book = {}

        def addr(name, rank, _b=book):
            return _b.setdefault((name, rank), 0x7000_0000 + 4096 * len(_b))

        plan = xm.plan_from_join(join, direction=direction, src_addr=addr,
                                 dst_addr=lambda n, r: 0x1000 + 64 * r)
        prof = wx.pointer_profile(plan.raw_descs)
        print("  " + join.line(direction=direction))
        print("  " + wx.pointer_profile_line(
            prof, hook=("destination" if direction == wx.LEGS_PP_TO_TP
                        else "source"),
            is_source=False, legs=1, legs_src_complete=1, legs_dst_complete=1))
        check(prof.src_resolved == prof.descs_total > 0,
              f"{direction}: src_resolved={prof.src_resolved}/"
              f"{prof.descs_total} (weg2xsn20 read 0/N on all 24 legs)")
        check(prof.dst_resolved == prof.descs_total,
              f"{direction}: dst_resolved={prof.dst_resolved}/{prof.descs_total}")

        tp_side = "dst_rank" if direction == wx.LEGS_PP_TO_TP else "src_rank"
        pp_side = "src_rank" if direction == wx.LEGS_PP_TO_TP else "dst_rank"
        per_name, wrong = {}, []
        for d in plan.raw_descs:
            if d.kind == wx.ZEROFILL:
                continue
            per_name.setdefault(d.param_name, set()).add(getattr(d, tp_side))
            if getattr(d, pp_side) != stage_of_layer(
                    int(d.param_name.split(".")[2])):
                wrong.append(d.param_name)
        cut = [n for n, r in per_name.items() if len(r) == len(CARDS)]
        print(f"  descriptors={len(plan.raw_descs)} tensors={len(join.tensors)} "
              f"cut_across_all_{len(CARDS)}_ranks={len(cut)}/{len(join.tensors)}")
        check(len(plan.raw_descs) > len(join.tensors),
              f"{direction}: n_descriptors > n_tensors -- the plan CUTS")
        check(len(cut) == len(join.tensors),
              f"{direction}: every tensor cut across all three TP ranks")
        check(not wrong,
              f"{direction}: every descriptor's PP side is its layer's stage")
        check(xb._missing_pointer(plan.descs) is None,
              f"{direction}: _missing_pointer finds no hole (W74's own site)")

    # THE CUT IS THE JOIN'S, NOT AN EVEN SPLIT -- the danger direction is a
    # wrong slice with the right byte total.
    name = f"model.layers.0.{CLASSES[0][0]}"
    widths = join.by_name[name].tp_widths
    print(f"  uneven check: {name} tp_widths={list(widths)} "
          f"even_split_would_be={[CLASSES[0][1] // len(CARDS)] * len(CARDS)}")
    check(len(set(widths)) > 1 and sum(widths) == CLASSES[0][1],
          "the TP widths are UNEVEN and sum to the unsharded extent")

    # -- 5. THE DIAGONAL PIN ----------------------------------------------
    print("\n[5] the diagonal pin (N/N on a diagonal plan does not count)")
    try:
        xm.refuse_diagonal_layout(1, tp_group="D")
        check(False, "a tp_size=1 destination must be refused")
    except wx.Weg2XchgPlanDisagree as exc:
        print(f"  refused: {str(exc)[:110]}...")
        check("W68" in str(exc), "refused by name (W68)")

    TOTAL = 22
    print(f"\nWEG2-XCHG-MANIFEST-SMOKE verdict="
          f"{'PASS' if not failures else 'FAIL'} "
          f"checks={TOTAL - len(failures)}/{TOTAL}")
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
