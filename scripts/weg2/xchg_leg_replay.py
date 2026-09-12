#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- THE DESK REPLAY OF A WHOLE EXCHANGE LEG, six real processes.

WHY THIS EXISTS (operator rule 2026-09-12, after weg2xsn24): no further
fix-boot cycle for walls BELOW the join. Four boots each found one layer --
the rank collision (xsn22), the padded vocabulary cut (xsn23), then the source
addresses and the slot handshake (xsn24) -- and each cost a window to learn
one fact that a desk run could have produced in minutes.

WHAT IT IS. Six OS processes with the real rank roles (3 P + 3 D), the REAL
semaphore names in /dev/shm (``<epoch>-<r>-<r>-<n>-{empty,full}`` and the
diagonal's per-card twelve), the real host-slot layout, and the REAL manifests
of a boot as input. The tensor bytes are synthetic and DETERMINISTIC from
(name, rank), which is what makes the seam digest a real verdict: a byte that
took the wrong path produces the wrong digest, every time, with no GPU.

WHAT IT IS NOT. No CUDA: the device ops are the transport's own fake, which
moves real bytes through mmap and is cross-process by construction. So this
proves the PLAN, the PHASES, the HANDSHAKE and the BYTE ROUTING. It does not
prove cudaMemcpy2DAsync or VRAM residency -- those stay metal questions.

    scripts/weg2/xchg_leg_replay.py --evidence /spinning/evidence-665-f1/weg2xsn24_0912
    scripts/weg2/xchg_leg_replay.py --self-test        # synthetic manifests, no evidence tree
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "test", "registered", "unit", "weg2"))

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

# WARM THE PADDED-CUT IMPORT IN THE PARENT, BEFORE ANY FORK.
#
# `xchg_manifest._pad_vocab_size` imports `layers.vocab_parallel_embedding`
# lazily, and that import is only REACHED when a tensor is padded -- so the
# synthetic self-test never took it and the real manifests always did. Taken
# for the first time inside a forked child of a process that already holds
# torch, it SEGFAULTS the child: six "Segfault encountered" and
# `ranks_reported=0/6 errors=0`, an absence that reads like nothing happened.
#
# In production this does not arise (the scheduler has long imported it), so
# the lazy import stays where it is and the REPLAY warms it instead -- the
# harness adapts to the product, not the other way round.
xm.vocab_pad_unit()
xm._pad_vocab_size(1)

NONCE = f"replay{os.getpid()}"
SLOT_BYTES = 4 << 20          # the layout, scaled; the RATIO is what matters
#: Must stay under FakeDeviceOps' FAKE_DEV_BYTES (8 MiB per rank).
FAKE_DEV_BUDGET = 5 << 20
DEPTH = 1


def seed_bytes(name: str, nbytes: int) -> bytes:
    """Deterministic content for one tensor, from its NAME alone.

    From the name and not from the rank: a shard of ``lm_head.weight`` on D
    rank 1 must contain exactly the rows of P's ``lm_head.weight`` that the
    plan assigns to it, so both ends derive the same stream and the digest can
    tell a correct route from a plausible one.
    """
    out = bytearray()
    h = hashlib.blake2b(name.encode(), digest_size=64)
    while len(out) < nbytes:
        out += h.digest()
        h = hashlib.blake2b(h.digest(), digest_size=64)
    return bytes(out[:nbytes])


def load_manifests(evidence: str):
    mans = []
    for f in sorted(os.listdir(evidence)):
        if not (f.startswith("phase_manifest_") and f.endswith(".json")):
            continue
        with open(os.path.join(evidence, f), encoding="utf-8") as fh:
            mans.append(xm.RankManifest.from_json(json.load(fh), path=f))
    return mans


def synthetic_manifests():
    """A six-rank set shaped like the real one, for --self-test."""
    cards = tuple(range(xr.N_CARDS))
    classes = (("self_attn.qkv_proj.weight", 512, 256),
               ("mlp.down_proj.weight", 256, 384))
    cut = (2, 1, 1)
    out = []

    def piece(n, r, c):
        return xm.ManifestPiece(param_name=n, tensor_class=n.rsplit(".", 2)[-2],
                                rows_full=r, cols_full=c, itemsize=1,
                                tag="weights_0", nbytes=r * c)

    def stage(layer):
        acc = 0
        for st, k in enumerate(cut):
            acc += k
            if layer < acc:
                return st
        raise AssertionError

    for pp in range(3):
        ps = [piece(f"model.layers.{l}.{sfx}", r, c)
              for l in range(sum(cut)) for sfx, r, c in classes
              if stage(l) == pp]
        out.append(xm.RankManifest(group="P", rank=pp, card=cards[pp],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=0, pp_rank=pp, pieces=tuple(ps)))
    for t in range(3):
        ps = []
        for l in range(sum(cut)):
            for sfx, r, c in classes:
                w = [r // 3] * 3
                w[-1] += r - sum(w)
                ps.append(piece(f"model.layers.{l}.{sfx}", w[t], c))
        out.append(xm.RankManifest(group="D", rank=t, card=cards[t],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=t, pp_rank=0, pieces=tuple(ps)))
    return out


def narrow(mans, col_div: int):
    """Scale COLUMNS for materialisation -- never rows, never the padding.

    THE BYTE BUDGET, and it is deliberately the one axis that is not under
    test. What the replay grades is the ROW mapping: which rows of the PP-side
    tensor land on which TP rank, and what happens to the padded tail. Columns
    are payload width. Scaling them uniformly on BOTH sides leaves the join's
    axis decision, the shard vector and ``pad_units`` bit-for-bit identical
    while turning a 2.54 GiB ``lm_head.weight`` into something six processes
    can hold.

    ONLY WHERE THE COLUMN COUNTS ALREADY AGREE, so a COLUMN cut is never
    touched -- scaling one there would break the sum the join checks, i.e. it
    would change the arithmetic instead of the budget.
    """
    if int(col_div) <= 1:
        return mans
    cols = {}
    for m in mans:
        for p in m.pieces:
            cols.setdefault(p.param_name, set()).add(int(p.cols_full))
    scalable = {n for n, c in cols.items() if len(c) == 1}
    out = []
    for m in mans:
        ps = []
        for p in m.pieces:
            if p.param_name in scalable:
                c = max(1, int(p.cols_full) // int(col_div))
                ps.append(xm.ManifestPiece(
                    param_name=p.param_name, tensor_class=p.tensor_class,
                    rows_full=p.rows_full, cols_full=c, itemsize=p.itemsize,
                    tag=p.tag, nbytes=p.rows_full * c * p.itemsize))
            else:
                ps.append(p)
        out.append(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(ps)))
    return out


def rank_proc(group, rank, evidence, root, q, self_test, max_tensors,
              col_div):
    """ONE rank: build the plan, take its half of every leg, report."""
    try:
        from test_weg2_xchg_transport_1273 import FakeDeviceOps

        mans = synthetic_manifests() if self_test else load_manifests(evidence)
        mans = narrow(mans, col_div)
        # THE JOIN SEES EVERY TENSOR -- its refusals must be the real ones.
        join = xm.join_manifests(mans, pp_group="P", tp_group="D")
        # THE REPLAY MATERIALISES A SUBSET, and that is a byte budget, not a
        # simplification of the arithmetic: each selected tensor keeps its REAL
        # extents, its real shard vector and its real padding, because those
        # are what the routing is made of. Materialising all of them would mean
        # synthesising the whole 27.5 GiB image per rank.
        #
        # The selection is DETERMINISTIC and always carries the padded
        # vocabulary tensors, which are the ones with a ZEROFILL tail and
        # therefore the ones a naive router gets wrong.
        # THE BYTE BUDGET IS A REFUSAL, NOT A HOPE. `FakeDeviceOps` maps
        # FAKE_DEV_BYTES (8 MiB) per rank; allocating past it used to walk off
        # the mapping and SEGFAULT the child, which reports as
        # `ranks_reported=0/6 errors=0` -- an absence that looks like nothing
        # happened. Tensors that do not fit are dropped BY NAME and counted.
        # THE BUDGET MUST BE THE WORST RANK'S ACTUAL ALLOCATION, not the
        # tensor's full size: a P stage holds WHOLE tensors of its own layers
        # while a D rank holds a shard of EVERY tensor, so the two sides fill
        # up on different sets. Budgeting on the full size let D past the gate
        # and then `raw_malloc` asserted `fake device out of memory` mid-leg --
        # after the P side had already deposited, which cascaded into
        # `slot still full` on all three P ranks. A budget that is not the
        # quantity actually allocated is not a budget.
        #
        # SELECTED IDENTICALLY ON ALL SIX RANKS (same sort, same predicate),
        # because a per-rank selection would make the six plans differ, which
        # is the one thing the whole slice exists to prevent.
        budget = FAKE_DEV_BUDGET
        fitted, dropped = [], 0
        worst = 0

        def rank_cost(t):
            whole = max((t.rows_full - t.pad_units) * t.cols_full * t.itemsize, 1)
            shard = max(max(t.tp_widths) * t.cols_full * t.itemsize, 1)
            return max(whole, shard)

        for t in sorted(join.tensors, key=rank_cost):
            nb = rank_cost(t)
            if worst + nb > budget:
                dropped += 1
                continue
            worst += nb
            fitted.append(t)
        used = worst
        if dropped:
            join = xm.ManifestJoin(
                pp_group=join.pp_group, tp_group=join.tp_group,
                cards=join.cards,
                tensors=tuple(sorted(fitted, key=lambda t: t.param_name)),
                unsourced=(), pp_ranks=join.pp_ranks, tp_ranks=join.tp_ranks)
            q.put(("budget", group, rank, dropped, used, len(fitted)))
        if max_tensors and len(join.tensors) > max_tensors:
            padded = [t for t in join.tensors if t.pad_units]
            rest = [t for t in join.tensors if not t.pad_units]
            rest.sort(key=lambda t: t.param_name)
            step = max(1, len(rest) // max(1, max_tensors - len(padded)))
            keep = padded + rest[::step][:max_tensors - len(padded)]
            join = xm.ManifestJoin(
                pp_group=join.pp_group, tp_group=join.tp_group,
                cards=join.cards,
                tensors=tuple(sorted(keep, key=lambda t: t.param_name)),
                unsourced=(), pp_ranks=join.pp_ranks, tp_ranks=join.tp_ranks)

        ops = FakeDeviceOps(root, rank=(rank if group == "D" else 3 + rank))
        sems = tp.SemSet(NONCE)
        slots = wb.BounceSlots(NONCE, shm_root=root, create=True)

        # This rank's own tensors, with DETERMINISTIC content. The PP side
        # holds whole tensors of its stage; the TP side holds its shard.
        own, dst_of = {}, {}
        for t in join.tensors:
            if group == "P":
                if t.pp_stage != rank:
                    continue
                nb = (t.rows_full - t.pad_units) * t.cols_full * t.itemsize
            else:
                nb = t.tp_widths[rank] * t.cols_full * t.itemsize
            ptr = ops.raw_malloc(0, max(nb, 1))
            own[t.param_name] = (ptr, nb)
            if group == "P":
                ctypes.memmove(ops.real(ptr), seed_bytes(t.param_name, nb), nb)
            else:
                ctypes.memset(ops.real(ptr), 0, nb)
            dst_of[t.param_name] = ptr

        def addr(name, r):
            got = own.get(str(name))
            return None if got is None or int(r) != int(rank) else got[0]

        # PP -> TP: P deposits, D collects.
        hook = "source" if group == "P" else "destination"
        direction = wx.leg_direction(hook, group)
        plan = xm.plan_from_join(
            join, direction=direction,
            src_addr=(addr if group == "P" else None),
            dst_addr=(addr if group == "D" else None))
        side = "src_rank" if group == "P" else "dst_rank"
        mine = [d for d in plan.descs if int(getattr(d, side, -1)) == rank]
        prof = wx.pointer_profile(mine)
        q.put(("profile", group, rank, prof.src_resolved, prof.dst_resolved,
               prof.descs_total))

        phase = wb.PHASE_DEPOSIT if group == "P" else wb.PHASE_COLLECT
        legs = 0
        for pair, grp in wb.group_descs_by_pair(mine).items():
            rv = (wb.CrossSlotRendezvous(sems, slots, pair=pair)
                  if pair is not None else
                  wb.CrossSlotRendezvous(sems, slots, card=rank))
            lane = f"p{pair}" if pair is not None else f"c{rank}"
            # THE SHMEM EMITTER (operator request, xsn24's +2.207 GiB at the
            # first flip was unattributed because no line names shmem per
            # region). Every host slot this leg maps says its bytes and its
            # region path, so an attribution has an address instead of a total.
            q.put(("slot", group, rank,
                   f"lane={lane} slot_bytes={SLOT_BYTES} depth={DEPTH} "
                   f"bytes={SLOT_BYTES * DEPTH} "
                   f"region={wb.bounce_path(NONCE, root, lane)}", 0, 0))
            wb.run_bounce_leg(grp, ops, NONCE, slot_bytes=SLOT_BYTES,
                              depth=DEPTH, mode=wx.INJECT_AUTHORITATIVE,
                              phase=phase, rendezvous=rv, shm_root=root,
                              device=0, lane=lane)
            legs += 1
        q.put(("legs", group, rank, legs, 0, 0))

        # THE VERDICT, on the collecting side only: did the bytes that landed
        # equal the bytes the source held for exactly this rank's slice?
        if group == "D":
            good = bad = 0
            bad_names = []
            for t in join.tensors:
                ptr, nb = own[t.param_name]
                if not nb:
                    continue
                # THE EXPECTATION IS AXIS-AWARE, and the first draft was not:
                # it computed a contiguous prefix for every class, which is
                # right for a ROW cut and wrong for a COLUMN cut -- where this
                # rank holds a STRIDED sub-block of every row. Four of eight
                # tensors "mismatched" against an expectation that described a
                # different tensor. A verdict that cannot express the layout it
                # grades is an instrument fault, not a finding.
                rows_content = t.rows_full - t.pad_units
                whole = seed_bytes(t.param_name,
                                   rows_content * t.cols_full * t.itemsize)
                if t.shard_axis == wx.COLS:
                    w = t.tp_widths[rank]
                    off = sum(t.tp_widths[:rank]) * t.itemsize
                    pitch = t.cols_full * t.itemsize
                    run = w * t.itemsize
                    want = b"".join(whole[i * pitch + off: i * pitch + off + run]
                                    for i in range(rows_content))
                else:
                    start = sum(t.tp_widths[:rank]) * t.cols_full * t.itemsize
                    want = whole[start:start + nb]
                got = ctypes.string_at(ops.real(ptr), len(want))
                if got == want:
                    good += 1
                else:
                    bad += 1
                    if len(bad_names) < 3:
                        off = next((i for i in range(min(len(got), len(want)))
                                    if got[i] != want[i]), -1)
                        zeros = got.count(0)
                        bad_names.append(
                            f"{t.param_name} axis={t.shard_axis} nb={nb} "
                            f"first_diff={off} zero_bytes={zeros}/{len(got)} "
                            f"stage={t.pp_stage} w={list(t.tp_widths)}")
            q.put(("digest", group, rank, good, bad, len(join.tensors)))
            if bad_names:
                q.put(("badnames", group, rank, ",".join(bad_names), 0, 0))
        q.put(("done", group, rank, 0, 0, 0))
    except BaseException as exc:  # noqa: BLE001
        q.put(("error", group, rank, f"{type(exc).__name__}: {exc}",
               traceback.format_exc()[-900:], 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="/spinning/evidence-665-f1/weg2xsn24_0912")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--col-div", type=int, default=1,
                    help="divide every tensor's COLUMN count by N for "
                         "materialisation, uniformly on both sides and only "
                         "where the column count already agrees (i.e. never on "
                         "a column cut). The ROW vector and the padding are "
                         "untouched, so the routing under test is unchanged; "
                         "this is a byte budget, not a change of arithmetic")
    ap.add_argument("--max-tensors", type=int, default=24,
                    help="materialise at most N tensors (0 = all); "
                         "the JOIN always sees every one")
    ns = ap.parse_args()

    if not ns.self_test and not os.path.isdir(ns.evidence):
        print(f"WEG2-XCHG-LEG-REPLAY REFUSED: no evidence dir {ns.evidence}")
        return 2

    root = f"/dev/shm/weg2-xchg-{NONCE}"
    os.makedirs(root, exist_ok=True)
    xr.create_semaphores(NONCE)
    print(f"WEG2-XCHG-LEG-REPLAY nonce={NONCE} slot_bytes={SLOT_BYTES} "
          f"depth={DEPTH} max_tensors={ns.max_tensors} col_div={ns.col_div} "
          f"source={'synthetic' if ns.self_test else ns.evidence}")

    q = mp.Queue()
    procs = [mp.Process(target=rank_proc,
                        args=(g, r, ns.evidence, root, q, ns.self_test, ns.max_tensors,
                              ns.col_div))
             for g in ("P", "D") for r in range(xr.N_CARDS)]
    for p in procs:
        p.start()
    seen, errors = [], []
    alive = len(procs)
    while alive:
        try:
            kind, g, r, a, b, c = q.get(timeout=300)
        except Exception:
            break
        if kind == "error":
            errors.append((g, r, a, b))
            alive -= 1
        elif kind == "done":
            alive -= 1
        else:
            seen.append((kind, g, r, a, b, c))
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    for kind, g, r, a, b, c in sorted(seen):
        if kind == "profile":
            print(f"  WEG2-XCHG-POINTER-PROFILE group={g} rank={r} "
                  f"src_resolved={a}/{c} dst_resolved={b}/{c}")
        elif kind == "legs":
            print(f"  WEG2-XCHG-LEGS group={g} rank={r} legs={a}")
        elif kind == "slot":
            print(f"  WEG2-XCHG-HOST-SLOT group={g} rank={r} {a}")
        elif kind == "budget":
            print(f"    BUDGET group={g} rank={r} dropped={a} tensors "
                  f"(over {FAKE_DEV_BUDGET} B), materialised={c} using {b} B")
        elif kind == "badnames":
            print(f"    first mismatches group={g} rank={r}: {a}")
        elif kind == "digest":
            v = "MATCH" if b == 0 and a else ("MISMATCH" if b else "NO-COMPARE")
            print(f"  WEG2-XCHG-SEAM-DIGEST group={g} rank={r} "
                  f"tensors_ok={a} tensors_bad={b} of={c} verdict={v}")
    for g, r, msg, tb in errors:
        print(f"  ERROR group={g} rank={r}: {msg}")
        print("    " + tb.replace("\n", "\n    ")[-700:])

    digests = [s for s in seen if s[0] == "digest"]
    ok = (not errors and digests
          and all(b == 0 and a for _k, _g, _r, a, b, _c in digests))
    print(f"WEG2-XCHG-LEG-REPLAY verdict={'MATCH' if ok else 'FAIL'} "
          f"ranks_reported={len({(s[1], s[2]) for s in seen})}/6 "
          f"errors={len(errors)}")

    xr.unlink_semaphores(NONCE)
    try:
        wb.BounceSlots(NONCE, shm_root=root).unlink()
    except BaseException:  # noqa: BLE001
        pass
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
