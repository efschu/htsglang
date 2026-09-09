# SPDX-License-Identifier: Apache-2.0
"""#1273: emit every S1/S2/S3/S7 acceptance line ONCE, from ONE merged tree.

WHY THIS EXISTS AND WHY IT IS NOT A TEST FILE.  Each slice's own suite asserts
its own line, in its own worktree, against its own half of
``weight_exchange.py``.  None of them can answer the integrator's question:
does the MERGED tree still emit all six?  An add/add union of two modules can
compile, pass both suites file-by-file, and still have lost a format string --
so the merge needs one process that imports the merged modules and prints the
lines the spec accepts, side by side, on stdout.

It lives under ``scripts/weg2`` rather than ``test/registered`` on purpose: it
is EVIDENCE, produced on demand for a merge, not a registered regression, and
the tier-2 gate must not silently inherit a six-process fork test.

RUN IT (hermetically -- no GPU, no checkpoint, no boot):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=python python3 -m pytest \\
        scripts/weg2/xchg_acceptance_driver.py -q -s

Every fixture is imported from the slices' own suites, so this driver states no
number of its own: if a slice changes its fixture, this changes with it, and it
cannot drift into agreeing with itself.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTDIR = os.path.join(ROOT, "test", "registered", "unit", "weg2")
sys.path.insert(0, os.path.join(ROOT, "python"))


def _load(mod_name: str):
    path = os.path.join(TESTDIR, mod_name + ".py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_S1 = _load("test_weg2_xchg_plan_1273")
_S2 = _load("test_weg2_xchg_cover_1273")
_S3 = _load("test_weg2_xchg_region_1273")
_S7 = _load("test_weg2_xchg_instruments_1273")
_S4 = _load("test_weg2_xchg_transport_1273")

EMITTED: list[str] = []


def _say(slice_name: str, line: str) -> None:
    EMITTED.append(line)
    print(f"ACCEPTANCE[{slice_name}] {line}", flush=True)


def test_s1_plan_line() -> None:
    """``WEG2-XCHG-PLAN`` on the three-wave sb4 schedule (spec 6/S1)."""
    from sglang.srt.managers.weg2_memory_saver import (
        chunk_tag_cards,
        weights_family_tags,
    )
    from sglang.srt.weg2.weight_exchange import build_plan, derive_waves, emit_plan_line

    cards = chunk_tag_cards(
        _S1.SB4_STAGE_LAYERS, _S1.SB4_LAYERS_PER_CHUNK, _S1.SB4_CHUNKS, _S1.CARDS
    )
    tags = weights_family_tags(_S1.SB4_CHUNKS)
    waves = derive_waves(tags, cards, _S1.CARDS)
    inventory = []
    for tag in tags:
        stage = min(cards.get(tag, _S1.CARDS))
        inventory.append(
            _S1._qkvz_geom().replace(
                name=f"model.{tag}.linear_attn.in_proj_qkvz.weight",
                tag=tag,
                stage=stage,
            )
        )
        inventory.append(
            _S1._mlp_down_geom().replace(
                name=f"model.{tag}.mlp.down_proj.weight", tag=tag, stage=stage
            )
        )
    plan = build_plan(
        inventory, _S1._p_layout(), _S1._d_layout(_S1.MLP_RATIO), waves=waves
    )
    line = emit_plan_line(plan)
    _say("S1", line)
    assert line.startswith("WEG2-XCHG-PLAN ")
    for token in (
        "dir=P2D",
        "waves=3",
        "descs=",
        "coalesced=",
        "min_piece_mib=",
        "bytes_gib=",
        "oncard_gib=",
        "cross_gib=",
        "zerofill_mib=",
        "hist=",
        "plan_id=",
    ):
        assert token in line, f"{token!r} missing from {line!r}"


def test_s2_cover_and_resident_lines() -> None:
    """``WEG2-XCHG-COVER`` and ``WEG2-XCHG-RESIDENT`` (spec 6/S2, 4.1)."""
    from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
    from sglang.srt.weg2 import weight_exchange as wx

    model = _S2._Model()
    log = _S2._CaptureLog()
    wx.register_plan_provider(lambda m: _S2._planned_bytes(m))
    try:
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            vote = wx.arm_coverage_at_load(
                model,
                rank=1,
                tag_bytes=lambda tag: int(30 * _S2.MIB),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                log=log,
            )
    finally:
        wx.register_plan_provider(None)
    assert vote.ok, vote.reason
    cover = [l for l in log.lines if l.startswith("WEG2-XCHG-COVER ")]
    resident = [l for l in log.lines if l.startswith("WEG2-XCHG-RESIDENT ")]
    assert cover and resident, log.lines
    for line in cover:
        _say("S2", line)
        for token in (
            "rank=",
            "tag=",
            "planned_mib=",
            "buffers_mib=",
            "tms_mib=",
            "slack_mib=",
            "uncovered=",
        ):
            assert token in line, f"{token!r} missing from {line!r}"
    _say("S2", resident[0])
    for token in ("tag=", "mib=", "in_family=", "rank=", "mode="):
        assert token in resident[0], resident[0]

    # and the draft tag, which is the line the spec quotes: in_family=no
    log2 = _S2._CaptureLog()
    from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS_DRAFT

    wx.register_plan_provider(lambda m: _S2._planned_bytes(m))
    try:
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            wx.arm_coverage_at_load(
                model,
                rank=2,
                tag_bytes=lambda tag: int(1311 * _S2.MIB),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                log=log2,
            )
    finally:
        wx.register_plan_provider(None)
    draft = [l for l in log2.lines if l.startswith("WEG2-XCHG-RESIDENT ")][0]
    _say("S2", draft)
    assert "in_family=no" in draft, draft


def test_s3_region_and_gate_lines(tmp_path) -> None:
    """``WEG2-XCHG-REGION`` and ``WEG2-XCHG-GATE`` from six real processes."""
    from sglang.srt.weg2 import weight_exchange_region as xr

    region = _S3._make(tmp_path)
    _say("S3", xr.region_line(region, sems=24))
    ctx = mp.get_context("fork")
    procs = []
    try:
        outs = [str(tmp_path / f"rank{row}.log") for row in range(xr.N_RANKS)]
        finish = ctx.Barrier(xr.N_RANKS)
        for row in range(xr.N_RANKS):
            p = ctx.Process(
                target=_S3._rank_child, args=(region.path, row, outs[row], 3, finish)
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(180)
        assert [p.exitcode for p in procs] == [0] * xr.N_RANKS, [
            p.exitcode for p in procs
        ]
        gates, regions = [], []
        for out in outs:
            with open(out) as fh:
                lines = fh.read().splitlines()
            gates += [l for l in lines if l.startswith("WEG2-XCHG-GATE ")]
            regions += [l for l in lines if l.startswith("WEG2-XCHG-REGION ")]
        assert regions, "registered=6/6 region line never re-emitted"
        _say("S3", regions[0])
        assert "registered=6/6" in regions[0]
        for line in gates[:3]:
            _say("S3", line)
        assert len(gates) == 3 * xr.N_RANKS, len(gates)
        for line in gates:
            for token in ("epoch=", "wave=", "joined=6/6", "ok=6/6", "skew_ms="):
                assert token in line, f"{token!r} missing from {line!r}"
            assert "(denominator: the six rows carrying this epoch)" in line
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        region.close()


def test_s7_flip_tag_and_armed_lines() -> None:
    """``WEG2-FLIP-TAG ... allocations= map_ms= copy_ms=`` and ``WEG2-XCHG-ARMED``."""
    from sglang.srt.weg2 import launcher

    lines, _adapter = _S7._drive_emitter(
        {"weights_0": {"allocations": 271, "map_ms": 111.1, "copy_ms": 222.2}},
        {"weights_0": [3.0 * 1024 * 1024 * 1024, 1000.0]},
    )
    flip = [l for l in lines if "WEG2-FLIP-TAG" in l]
    assert flip, lines
    _say("S7", flip[0])
    for token in ("allocations=", "map_ms=", "copy_ms="):
        assert token in flip[0], flip[0]

    out = []
    with _S7._census_file(_S7._sb4_census()) as path:
        res = launcher.prepare_weight_exchange(
            _S7._cards(), out.append, "exchange", path, "b17.4", 0
        )
    assert res is not None
    checks = [l for l in out if l.startswith("WEG2-XCHG-CHECK")]
    armed = [l for l in out if l.startswith("WEG2-XCHG-ARMED")]
    assert len(armed) == 1 and len(checks) == 6, out
    _say("S7", checks[0])
    _say("S7", armed[0])
    for token in (
        "epoch=",
        "waves=3",
        "peak_mib=",
        "free_mib=",
        "wave1_ok=6/6",
        "floor_mib=",
        "region_mib=385",
        "ring_H_mib=0",
        "wired=",
    ):
        assert token in armed[0], f"{token!r} missing from {armed[0]!r}"


def test_s4_pair_and_oncard_lines(tmp_path) -> None:
    """``WEG2-XCHG-PAIR`` and ``WEG2-XCHG-ONCARD`` from a real six-rank run.

    Not from a hand-built stats object: the numbers come from S4's own
    hermetic double, so a format string that drifted from what the transport
    actually produces cannot pass here.
    """
    from sglang.srt.weg2 import weight_exchange_region as xr
    from sglang.srt.weg2 import weight_exchange_transport as tp

    boot = _S4._fresh_boot()
    root = os.path.join(str(tmp_path), "dev")
    os.makedirs(root, exist_ok=True)
    region = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    path = region.path
    region.close()
    xr.create_semaphores(boot)
    ctx = mp.get_context("fork")
    ready = ctx.Barrier(xr.N_RANKS)
    procs, outs = [], []
    try:
        for group in ("P", "D"):
            for rank in range(xr.N_CARDS):
                out = os.path.join(str(tmp_path), f"a-{group}{rank}.txt")
                outs.append(out)
                proc = ctx.Process(
                    target=_S4._rank_child,
                    args=(root, path, boot, group, rank, out, ready))
                proc.start()
                procs.append(proc)
        for proc in procs:
            proc.join(150)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        xr.unlink_semaphores(boot)

    lines = []
    for out in outs:
        with open(out) as fh:
            verdict = eval(fh.read())  # noqa: S307 -- our own repr
        assert verdict["error"] == "", verdict["error"]
        assert verdict["mismatch"] == [], verdict["mismatch"]
        lines.extend(verdict["lines"])

    pairs = [l for l in lines if l.startswith(tp.PAIR_LINE_PREFIX)]
    oncard = [l for l in lines if l.startswith(tp.ONCARD_LINE_PREFIX)]
    waves = _S4.DOUBLE_WAVES
    assert len(pairs) == waves * 2 * xr.N_PAIRS, len(pairs)
    assert len(oncard) == waves * 2 * xr.N_CARDS, len(oncard)
    _say("S4", pairs[0])
    _say("S4", oncard[0])
    for token in ("src=", "dst=", "bytes_mib=", "pieces=", "strided_mib=",
                  "ms=", "gbs=", "slot_waits=", "slot_wait_ms="):
        assert token in pairs[0], f"{token!r} missing from {pairs[0]!r}"
    for token in ("card=", "mode=", "bytes_mib=", "batches=", "hop_ms="):
        assert token in oncard[0], f"{token!r} missing from {oncard[0]!r}"
    # The count is of BATCHES and says so: the lane has exactly two hops by
    # construction, so a `hops=` printing ~329 on the real flip would be a
    # field whose name states what the number is not.
    assert "hops=" not in oncard[0], oncard[0]

    degraded = []
    tp.arm_oncard_lane(card_uuid="GPU-probe", probe=lambda: (False, "smoke"),
                       log=degraded.append)
    _say("S4", degraded[0])
    assert tp.ONCARD_UNAVAILABLE_MARKER in degraded[0]
    # The degrade line must name the target it really uses -- a per-card bounce
    # file, NOT the staging region the spec prose names (CROSS_PAIRS has no
    # diagonal to borrow), and the host term that goes with it.
    assert "bounce=oncard-<card>.bin" in degraded[0], degraded[0]
    assert f"host_add_mib={tp.ONCARD_HOST_DEGRADE_MIB}" in degraded[0]


def tp_marker() -> str:
    from sglang.srt.weg2 import weight_exchange_transport as tp

    return tp.ONCARD_UNAVAILABLE_MARKER


def test_all_six_prefixes_were_emitted() -> None:
    """The integrator's actual question: all of them, from one merged tree."""
    want = (
        "WEG2-XCHG-PLAN",
        "WEG2-XCHG-COVER",
        "WEG2-XCHG-RESIDENT",
        "WEG2-XCHG-REGION",
        "WEG2-XCHG-GATE",
        "WEG2-FLIP-TAG",
        "WEG2-XCHG-ARMED",
        "WEG2-XCHG-PAIR",
        "WEG2-XCHG-ONCARD",
        tp_marker(),
    )
    missing = [p for p in want if not any(p in line for line in EMITTED)]
    assert not missing, (
        f"acceptance prefixes never emitted by the merged tree: {missing}; "
        f"note this check only holds when the whole file runs in ONE process "
        f"(no xdist), because EMITTED is per-process"
    )
