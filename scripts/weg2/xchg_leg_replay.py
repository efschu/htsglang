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
FAKE_DEV_BUDGET = 2 << 20
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


BOOT_TOKEN = "replay"


def rank_extents(t, group: str, rank: int):
    """(rows, cols) THIS rank holds of one joined tensor -- AXIS-AWARE.

    The first version took ``tp_widths[rank]`` as the ROW count for every
    class, which is right for a row cut and wrong for a column cut. The
    product's own materialisation check caught it immediately and by name:
    ``A_log: the manifest records (1, 30) ... and the materialised tensor
    holds (30, 48)``. That guard is doing exactly what it was wired for -- the
    manifest is what every other reader plans from -- so the fixture was the
    thing that had drifted.
    """
    if group == "P":
        rows = t.rows_full - (t.pad_units if t.shard_axis == wx.ROWS else 0)
        cols = t.cols_full - (t.pad_units if t.shard_axis == wx.COLS else 0)
        return max(rows, 1), max(cols, 1)
    if t.shard_axis == wx.COLS:
        return max(t.rows_full, 1), max(t.tp_widths[rank], 1)
    if t.shard_axis == wx.ROWS:
        return max(t.tp_widths[rank], 1), max(t.cols_full, 1)
    return max(t.rows_full, 1), max(t.cols_full, 1)


def budget_manifests(mans, max_tensors: int):
    """Drop tensors that do not fit, BY NAME, identically on all six ranks.

    THE COST IS COMPUTED FROM THE JOIN, not estimated from the manifests, and
    that correction is the second one this budget needed. A per-manifest
    estimate under-counted a D rank by 8x (`allocation 16032228 B over budget
    2097152 B at visual.blocks.0.attn.qkv_proj.bias`) because what a rank
    actually allocates is a function of the JOIN's extents -- the padded total,
    the per-rank width -- and not of any single manifest row. Budgeting on a
    quantity that is not the one allocated is the same defect twice.

    SELECTED IN THE PARENT so the six children cannot disagree about the
    population: six plans that differ is the one thing this slice prevents.
    """
    join = xm.join_manifests(mans, pp_group="P", tp_group="D")

    def rank_costs(t):
        """(worst P stage cost, worst D rank cost) for one tensor."""
        pr, pc = rank_extents(t, "P", 0)
        whole = pr * pc * t.itemsize
        shard = max(
            rank_extents(t, "D", r)[0] * rank_extents(t, "D", r)[1] * t.itemsize
            for r in range(len(t.tp_widths)))
        return max(whole, 1), max(shard, 1)

    # STRATIFIED BY SOURCE RANK, and this is the defect the train seat's call
    # exposed. Picking globally cheapest-first put the whole subset on ONE P
    # stage: the DEFAULT form read
    #   `the joined plan has 72 descriptors and none with src_rank=2`
    # on P ranks 1 and 2 -- `ranks_reported=4/6` -- while my own call with
    # `--col-div 512` saw all three stages and read MATCH. Two invocations of
    # one script, and the SELECTION was the whole difference. A filter that can
    # starve a rank grades a lane it never exercised.
    #
    # Round-robin over the stages keeps every source rank represented, which is
    # what makes a six-rank replay a six-rank replay.
    by_stage = {}
    for t in join.tensors:
        by_stage.setdefault(int(t.pp_stage), []).append(t)
    for st in by_stage:
        by_stage[st].sort(key=lambda t: sum(rank_costs(t)))
    keep, p_used, d_used, dropped = set(), {}, 0, 0
    order = sorted(by_stage)
    idx = {st: 0 for st in order}
    while True:
        progressed = False
        for st in order:
            if idx[st] >= len(by_stage[st]):
                continue
            t = by_stage[st][idx[st]]
            idx[st] += 1
            progressed = True
            w, sh = rank_costs(t)
            # BOTH SIDES BOUNDED, SEPARATELY: a P stage holds whole tensors of
            # its OWN layers (cap per stage) while a D rank holds a shard of
            # EVERY tensor (cap global).
            if (p_used.get(st, 0) + w > FAKE_DEV_BUDGET
                    or d_used + sh > FAKE_DEV_BUDGET):
                dropped += 1
                continue
            p_used[st] = p_used.get(st, 0) + w
            d_used += sh
            keep.add(t.param_name)
            if max_tensors and len(keep) >= max_tensors:
                progressed = False
                break
        if not progressed:
            break
    p_used_max = max(p_used.values()) if p_used else 0

    # COVERAGE PER RANK AND PER DIRECTION, because those are the two things a
    # leg needs to exist at all. Under `pp_to_tp` the SOURCE ranks are the P
    # stages; under `tp_to_pp` they are the D ranks, which hold a shard of
    # every kept tensor.
    kept_t = [t for t in join.tensors if t.param_name in keep]
    stages_full = sorted({int(t.pp_stage) for t in join.tensors})
    stages_kept = sorted({int(t.pp_stage) for t in kept_t})
    d_full = sorted(range(len(join.cards)))
    d_kept = sorted({r for t in kept_t for r in range(len(t.tp_widths))
                     if t.tp_widths[r] > 0})
    if stages_kept != stages_full or d_kept != d_full:
        raise SystemExit(
            f"WEG2-XCHG-LEG-REPLAY REFUSED reason=subset-starves-a-rank "
            f"subset covers pp_to_tp src ranks {set(stages_kept)} of "
            f"{set(stages_full)} and tp_to_pp src ranks {set(d_kept)} of "
            f"{set(d_full)}; kept={len(keep)} of {len(join.tensors)} "
            f"cap={FAKE_DEV_BUDGET} max_tensors={max_tensors} -- a source rank "
            f"with no descriptor cannot deposit, so the run would grade four "
            f"ranks of six and call it a result. THIS IS THE SELECTION'S "
            f"DEFECT, NOT THE PLAN'S: the full join covers {set(stages_full)}. "
            f"Raise --max-tensors or lower --col-div.")
    print(f"  BUDGET kept={len(keep)} dropped={dropped} of {len(join.tensors)} "
          f"pp_to_tp_src_ranks={stages_kept}/{stages_full} "
          f"tp_to_pp_src_ranks={d_kept}/{d_full} p_bytes_max={p_used_max} "
          f"d_bytes={d_used} cap={FAKE_DEV_BUDGET} -- the dropped tensors are "
          f"NOT byte-routed here; their byte path stays a metal question")
    out = []
    for m in mans:
        out.append(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(p for p in m.pieces if p.param_name in keep)))
    return out


class _FakeParam:
    """A tensor as the product path reads one: address AND geometry.

    ``data_ptr()`` is what the address books ask; ``shape``/``stride()``/
    ``element_size()`` are what `refuse_on_materialisation_drift` reads through
    `StorageGeom.of`. A double with only the pointer made every rank refuse
    with ``derivation-failed: '_FakeParam' object has no attribute 'shape'`` --
    which is the materialisation check doing its job on a fixture that had not
    materialised anything.
    """

    def __init__(self, ptr: int, rows: int, cols: int, itemsize: int):
        self._ptr = int(ptr)
        self.shape = (int(rows), int(cols))
        self._stride = (int(cols), 1)
        self._itemsize = int(itemsize)

    def data_ptr(self) -> int:
        return self._ptr

    def stride(self):
        return self._stride

    def element_size(self) -> int:
        return self._itemsize

    def dim(self) -> int:
        return 2


class _FakeModel:
    def __init__(self, params):
        self._p = list(params)

    def named_parameters(self):
        return list(self._p)


class _FakeRunner:
    def __init__(self, model):
        self.model = model


class _FakeWorker:
    def __init__(self, model):
        self.model_runner = _FakeRunner(model)


class _FakeDraftWorker:
    """The shape ``_get_draft_model_runner`` resolves."""

    def __init__(self, model):
        self.draft_model_runner = _FakeRunner(model)


class _Stub:
    """Borrows the REAL unbound methods -- nothing is re-implemented.

    ``draft_params=None`` is the CAN-FAIL ARM: it reproduces the single-runner
    address book weg2xsn24 shipped, where ``fc.weight`` had no home and the
    leg refused with W74 at ``dst_resolved=894/904``.
    """

    def __init__(self, main_params, draft_params, group="P", rank=0):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        self._group, self._rank = str(group), int(rank)
        self.tp_worker = _FakeWorker(_FakeModel(main_params))
        self.draft_worker = (None if draft_params is None
                             else _FakeDraftWorker(_FakeModel(draft_params)))
        cls = wu.SchedulerWeightUpdaterManager
        for name in ("_weg2_rank_param_table", "_weg2_join_src_addr",
                     "_weg2_join_dst_addr", "_weg2_shadow_plan",
                     "_weg2_xchg_bounce_leg"):
            setattr(type(self), name, getattr(cls, name))

    # #1358: the adapter reads its own identity for the host-slot lines.
    def _weg2_group_name(self):
        return self._group

    def _weg2_rank(self):
        return self._rank


def rank_proc(group, rank, evidence, root, q, self_test, max_tensors,
              col_div, single_runner, manifest_dir):
    """ONE rank, THROUGH THE PRODUCT ADAPTER.

    THE ENTRY POINT IS THE BOOT'S, and that is the whole point of this
    revision. The first version called ``wb.run_bounce_leg`` with an address
    book the replay built itself, so it graded the library and left the two
    things weg2xsn24 actually died on -- the adapter's two-runner address
    table and its lazy import -- UNTESTED at the desk. Now the stub borrows the
    real unbound methods (`_weg2_shadow_plan`, `_weg2_rank_param_table`, the
    two address books, `_weg2_xchg_bounce_leg`) and the bytes travel the path a
    flip leg travels.
    """
    try:
        from test_weg2_xchg_transport_1273 import FakeDeviceOps

        os.environ[xm.DIR_ENV] = manifest_dir
        os.environ[xr.ENV_REGION_BOOT] = BOOT_TOKEN
        os.environ["SGLANG_WEG2_WEIGHT_SOURCE"] = "exchange"
        os.environ["SGLANG_WEG2_GROUP"] = group

        mans = xm.load_manifests(manifest_dir, boot_token=BOOT_TOKEN)
        join = xm.join_manifests(mans, pp_group="P", tp_group="D")

        ops = FakeDeviceOps(root, rank=(rank if group == "D" else 3 + rank))

        # THE TWO RUNNERS, as the real rank has them. The draft head's tensors
        # (`fc.weight` and friends) live ONLY in the draft runner -- which is
        # exactly why a single-runner address book resolved 894 of 904 and
        # refused the leg.
        mine = [m for m in mans if m.group == group and m.rank == rank]
        # WHICH RUNNER A TENSOR BELONGS TO IS THE FILE IT CAME FROM, not the
        # tag on the piece. The can-fail arm deliberately re-tags the draft
        # PIECES into the main region (so the region cut cannot separate them,
        # which is the pre-fix state) while the FILE keeps its own region_tag
        # -- so this is the reading that still places them in the draft runner,
        # where a single-runner address book cannot reach them.
        draft_names = {p.param_name for m in mine
                       if str(m.region_tag) == "weights_draft"
                       for p in m.pieces}
        # In the can-fail arm the draft pieces were re-tagged into the
        # main region, so `draft_names` is empty and every piece lands in
        # the MAIN double -- except the ones the single-runner book drops.

        main_params, draft_params, own = [], [], {}
        allocated = 0
        for t in join.tensors:
            if group == "P" and t.pp_stage != rank:
                continue
            rows, cols = rank_extents(t, group, rank)
            nb = max(rows * cols * t.itemsize, 1)
            allocated += nb
            if allocated > FAKE_DEV_BUDGET:
                # THE CHILD'S OWN CEILING, named rather than asserted deep in
                # the fake device. The parent budgets on a per-tensor cost; if
                # that estimate is ever short, this says so with the number
                # instead of dying as `fake device out of memory` mid-leg.
                q.put(("error", group, rank,
                       f"allocation {allocated} B over budget "
                       f"{FAKE_DEV_BUDGET} B at {t.param_name} -- the "
                       f"parent's per-tensor cost under-counted this rank",
                       "", 0))
                q.put(("done", group, rank, 0, 0, 0))
                return
            ptr = ops.raw_malloc(0, nb)
            own[t.param_name] = (ptr, nb)
            if group == "P":
                ctypes.memmove(ops.real(ptr), seed_bytes(t.param_name, nb), nb)
            else:
                ctypes.memset(ops.real(ptr), 0, nb)
            # A TENSOR WHOSE data_ptr() IS THE FAKE DEVICE ADDRESS: the address
            # books call `.data_ptr()`, so the double must answer it.
            param = _FakeParam(ptr, rows, cols, t.itemsize)
            (draft_params if t.param_name in draft_names
             else main_params).append((t.param_name, param))

        stub = _Stub(main_params, None if single_runner else draft_params,
                     group=group, rank=rank)

        hook = "source" if group == "P" else "destination"
        plan, reason = stub._weg2_shadow_plan(
            hook, group, rank, agreed=None, require_agreement=False)
        if plan is None:
            q.put(("error", group, rank, f"no plan: {reason}", "", 0))
            q.put(("done", group, rank, 0, 0, 0))
            return
        prof = wx.pointer_profile(plan.descs)
        q.put(("profile", group, rank, prof.src_resolved, prof.dst_resolved,
               prof.descs_total))
        q.put(("alloc", group, rank, allocated, len(own), 0))

        sems = tp.SemSet(NONCE)
        # #1358: the PRODUCT's own host-slot lines, forwarded verbatim, so the
        # replay reads the same form a boot does instead of a line of its own.
        def _capture(line: str) -> None:
            text = str(line)
            if text.startswith(wb.HOST_SLOT_MARKER):
                q.put(("hostslot", group, rank, text, 0, 0))

        stub._weg2_xchg_bounce_leg(
            descs=list(plan.descs), ops=ops, boot_nonce=NONCE,
            slot_bytes=SLOT_BYTES, depth=DEPTH,
            mode=wx.INJECT_AUTHORITATIVE, shm_root=root, device=0,
            hook=hook, region=None, sems=sems)
        q.put(("hostslotleg", group, rank,
               wb.host_slot_leg_line(group=group, rank=rank,
                                     leg=f"{NONCE}/{hook}"), 0, 0))
        q.put(("entry", group, rank, "adapter", 0, 0))

        if group == "D":
            good = bad = 0
            bad_names = []
            for t in join.tensors:
                ptr, nb = own[t.param_name]
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
                        bad_names.append(t.param_name)
            q.put(("digest", group, rank, good, bad, len(join.tensors)))
            if bad_names:
                q.put(("badnames", group, rank, ",".join(bad_names), 0, 0))
        q.put(("done", group, rank, 0, 0, 0))
    except BaseException as exc:  # noqa: BLE001
        q.put(("error", group, rank, f"{type(exc).__name__}: {exc}",
               traceback.format_exc()[-900:], 0))
        q.put(("done", group, rank, 0, 0, 0))


def _run(ns) -> int:

    if not ns.self_test and not os.path.isdir(ns.evidence):
        print(f"WEG2-XCHG-LEG-REPLAY REFUSED: no evidence dir {ns.evidence}")
        return 2

    root = f"/dev/shm/weg2-xchg-{NONCE}"
    os.makedirs(root, exist_ok=True)
    # CREATED AFTER THE BUDGET, and that ordering is a leak I caused:
    # `budget_manifests` can `SystemExit` on a starved subset, which skips
    # every line below it -- so a REFUSAL left 36 named semaphores in
    # /dev/shm. A refusal that leaks is a refusal that costs the next run.
    # The teardown below is in a `finally` for the same reason.
    print(f"WEG2-XCHG-LEG-REPLAY nonce={NONCE} slot_bytes={SLOT_BYTES} "
          f"depth={DEPTH} max_tensors={ns.max_tensors} col_div={ns.col_div} "
          f"source={'synthetic' if ns.self_test else ns.evidence}")

    q = mp.Queue()
    # THE PARENT WRITES THE MANIFESTS ONCE, narrowed and budgeted, under ONE
    # boot token -- the children then read them exactly as a rank reads its
    # boot's, through `manifests_for_boot`, which is the product's own entry.
    mdir = os.path.join(root, "manifests")
    os.makedirs(mdir, exist_ok=True)
    src_mans = (synthetic_manifests() if ns.self_test
                else load_manifests(ns.evidence))
    src_mans = narrow(src_mans, ns.col_div)
    src_mans = budget_manifests(src_mans, ns.max_tensors)
    # THE CAN-FAIL ARM MODELS THE PRE-FIX PRODUCT, and it needs no production
    # knob to do it. Before the region cut a leg carried every runner's pieces
    # in ONE region, so the cut could not separate them and the single-runner
    # address book had no home for the draft head's. Re-tagging the draft
    # pieces into the main region reproduces exactly that state -- which is
    # what weg2xsn25 measured as `dst_resolved=893/904` + W74 on fc.weight,
    # and (per the operator's causality question) the W68 that follows it.
    for m in src_mans:
        pieces = m.pieces
        region = m.region_tag
        if ns.single_runner and str(m.region_tag) == "weights_draft":
            pieces = tuple(
                xm.ManifestPiece(
                    param_name=p_.param_name, tensor_class=p_.tensor_class,
                    rows_full=p_.rows_full, cols_full=p_.cols_full,
                    itemsize=p_.itemsize, tag="weights", nbytes=p_.nbytes)
                for p_ in m.pieces)
            # THE FILE NAME KEEPS ITS OWN region_tag: the region cut reads the
            # PIECE's tag, while the name's third axis exists so two runners of
            # one rank cannot collide (weg2xsn20/22). Changing both made the
            # overwrite ratchet fire -- correctly.
        xm.write_rank_manifest(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=region,
            boot_token=BOOT_TOKEN, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=pieces), mdir)
    print(f"  manifests={len(src_mans)} entry={ns.entry} "
          f"address_book="
          f"{'single-runner (CAN-FAIL ARM)' if ns.single_runner else 'both runners'}")

    # CREATED ONLY ONCE THE SUBSET IS ACCEPTED. `budget_manifests` can
    # `SystemExit` on a starved subset, and creating the handshake before that
    # point leaked 36 named semaphores per refusal -- a refusal that leaks is a
    # refusal that costs the next run.
    xr.create_semaphores(NONCE)
    procs = [mp.Process(target=rank_proc,
                        args=(g, r, ns.evidence, root, q, ns.self_test,
                              ns.max_tensors, ns.col_div, ns.single_runner,
                              mdir))
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
        elif kind == "alloc":
            print(f"  WEG2-XCHG-ALLOC group={g} rank={r} bytes={a} tensors={b}")
        elif kind in ("hostslot", "hostslotleg"):
            print(f"  {a}")
        elif kind == "entry":
            print(f"  WEG2-XCHG-ENTRY group={g} rank={r} entry={a}")
        elif kind == "slot":
            print(f"  WEG2-XCHG-HOST-SLOT group={g} rank={r} {a}")
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

    return 0 if ok else 1


def _teardown(root: str) -> None:
    """Remove EVERYTHING this run made, on every exit path.

    IN A `finally` BECAUSE A REFUSAL IS AN EXIT PATH TOO: `budget_manifests`
    raises `SystemExit` on a starved subset, and with the teardown inline each
    refusal left its root directory (and, before the ordering fix, 36 named
    semaphores) behind. A refusal that leaks is a refusal that costs the next
    run -- the same class as a test that does not sweep its own shm.
    """
    import shutil

    try:
        xr.unlink_semaphores(NONCE)
    except BaseException:  # noqa: BLE001
        pass
    try:
        wb.BounceSlots(NONCE, shm_root=root).unlink()
    except BaseException:  # noqa: BLE001
        pass
    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="/spinning/evidence-665-f1/weg2xsn24_0912")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--single-runner", action="store_true",
                    help="CAN-FAIL ARM: build the address book from the MAIN "
                         "runner only, as weg2xsn24 shipped it. The draft "
                         "head then has no address and the leg must refuse "
                         "with W74 on fc.weight (dst_resolved=894/904)")
    ap.add_argument("--entry", choices=("adapter", "library"),
                    default="adapter",
                    help="which entry point the bytes travel; the boot's is "
                         "the adapter")
    ap.add_argument("--col-div", type=int, default=1,
                    help="divide every tensor's COLUMN count by N for "
                         "materialisation, uniformly on both sides and only "
                         "where the column count already agrees (i.e. never on "
                         "a column cut). The ROW vector and the padding are "
                         "untouched, so the routing under test is unchanged; "
                         "this is a byte budget, not a change of arithmetic")
    ap.add_argument("--max-tensors", type=int, default=0,
                    help="materialise at most N tensors (0 = all that fit the byte "
                         "budget, which IS the default form the xsn25 "
                         "acceptance cites); "
                         "the JOIN always sees every one")
    ns = ap.parse_args()
    root = f"/dev/shm/weg2-xchg-{NONCE}"
    try:
        return _run(ns)
    except SystemExit as exc:
        # A NAMED REFUSAL, printed as one line and exited 2 -- not a traceback,
        # and not a silent 0.
        print(str(exc))
        return 2
    finally:
        _teardown(root)


if __name__ == "__main__":
    sys.exit(main())
