# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1273 slice S3 -- the shared region, Gate 0 and the wave gate.  No CUDA.

WEG2_REUSE_SPEC_0908 section 6 / S3, red-first.  Every test here failed at the
parent ``3ea18deb95``, where ``sglang.srt.weg2.weight_exchange_region`` does
not exist.

THE STOP-LOSS THIS FILE ENFORCES, in the spec's own words: *"any gate that can
pass while a rank is absent is not a gate; the denominator must be printed and
must be the rows carrying THIS epoch."*  FIVE tests are built to fail on that
one sentence rather than on the happy path, and four of them exist because the
first cut of this module passed 22/22 while the gate could still close with a
rank absent:

* :func:`test_gate_names_the_non_joiner` -- five rows advance, one does not.
* :func:`test_gate_denominator_is_the_rows_carrying_this_epoch` -- six rows all
  carry a gate_seq high enough, but one of them was written by the PREVIOUS
  flip.  A count that forgets the epoch filter reads 6/6 and lets a rank that
  has not woken up this flip be counted as a joiner.  That is the
  ``credit_epoch`` failure (boot weg2rg2's three terminal counters, one with
  ``{"epoch": 12, ... "leg_complete": true}``) one mechanism over.
* :func:`test_a_second_flip_on_one_region_inherits_no_join` -- the same failure
  from the OTHER side: a region that lives for the boot, whose stamp is not
  re-made per flip, hands flip 2 a full set of flip 1's rows.
* :func:`test_a_joined_row_whose_writer_is_dead_is_not_a_join` and
  :func:`test_six_rows_written_by_one_process_are_not_six_ranks` -- six rows
  are not six ranks.
* :func:`test_a_torn_row_is_not_a_join` -- a row whose seal does not match its
  payload is a half-written row, not a vote.

The hermetic double is the spec's: six ``multiprocessing`` children under
``CUDA_VISIBLE_DEVICES=""``, a region in ``tmp_path``, and ``memcpy`` in place
of every CUDA call.
"""

from __future__ import annotations

import ast
import ctypes
import inspect
import multiprocessing as mp
import os
import signal
import struct
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.weg2 import weight_exchange_region as xr

#: The BOOT nonce names the region, its 24 semaphores and its registered
#: flags; the FLIP token stamps the gate rows, the matrix rows and the slots.
BOOT = "1788900000"
EPOCH = f"{BOOT}.7"
NEXT_EPOCH = f"{BOOT}.8"
OTHER_BOOT = "1788000000"
STALE_EPOCH = f"{OTHER_BOOT}.11"


def _make(tmp_path, epoch: str = EPOCH, boot: str = BOOT) -> xr.XchgRegion:
    region = xr.XchgRegion.create(boot, path=str(tmp_path / "xchg.bin"))
    region.begin_flip(epoch)
    return region


def _open(path: str, epoch: str = EPOCH, boot: str = BOOT) -> xr.XchgRegion:
    region = xr.XchgRegion.open(path, expect_boot=boot)
    region.begin_flip(epoch)
    return region


def _byte_matrix() -> list:
    """A deterministic, self-consistent 6x6 send matrix.

    Cell ``[a][b]`` is what rank ``a`` sends rank ``b``; the diagonal and the
    on-card pair are zero because that traffic takes the IPC lane, never a
    staging slot.
    """
    m = [[0] * xr.N_RANKS for _ in range(xr.N_RANKS)]
    for a in range(xr.N_RANKS):
        for b in range(xr.N_RANKS):
            if a == b or (a % xr.N_CARDS) == (b % xr.N_CARDS):
                continue
            m[a][b] = (a + 1) * 1_000_000 + (b + 1) * 4096
    return m


def _publish_all(region: xr.XchgRegion, matrix, plan_hash: int = 0xABCDEF) -> None:
    for r in range(xr.N_RANKS):
        region.write_matrix_row(
            r, matrix[r], [matrix[a][r] for a in range(xr.N_RANKS)], plan_hash
        )


NO_CENSUS = "S1/S2 not wired in this fixture"


def _fake_proc(tmp_path, pids) -> str:
    """A ``/proc`` double: the pids named here are alive, everything else is not."""
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    for pid in pids:
        (root / str(pid)).mkdir(exist_ok=True)
    return str(root)


# --------------------------------------------------------------------------
# children of the hermetic double.  Module level so any start method works.
# --------------------------------------------------------------------------


def _gate_child(path: str, row: int, waves, hold_s: float) -> None:
    region = _open(path)
    try:
        for wave in waves:
            region.write_gate_row(row, wave, True)
        time.sleep(hold_s)
    finally:
        region.close()


def _write_and_die_child(path: str, row: int, wave: int) -> None:
    """Write a gate row, then exit CLEANLY.  The row outlives the rank."""
    region = _open(path)
    region.write_gate_row(row, wave, True)
    region.close()


def _producer_child(path: str, pair: int, slot: int, nbytes: int, publish: bool) -> None:
    region = _open(path)
    region.begin_fill(pair, slot, seq=1)
    if publish:
        off = region.data_offset(pair, slot)
        region._mm[off: off + nbytes] = b"\xa5" * nbytes  # memcpy in place of CUDA
        region.publish(pair, slot, nbytes, checksum=0xA5A5)
    while True:  # the parent SIGKILLs us; a clean exit would prove nothing
        time.sleep(0.05)


def _register_child(path: str, row: int, start) -> None:
    """Registration is BOOT-scoped: no ``begin_flip`` here, deliberately."""
    region = xr.XchgRegion.open(path, expect_boot=BOOT)
    try:
        start.wait(30)
        region.mark_registered(row)
    finally:
        region.close()


def _rank_child(path: str, row: int, out_path: str, waves: int, finish) -> None:
    """One rank of the six-rank double: Gate 0, then ``waves`` wave gates.

    ``finish`` holds every child alive until all six are through their last
    gate.  Without it the first child to finish would exit while a slower peer
    was still polling the same gate, and that peer would -- correctly -- refuse
    its row as ``joined_but_dead``.  In the real form a rank does not exit
    after a flip; it goes back to serving.
    """
    lines = []
    region = _open(path)
    try:
        matrix = _byte_matrix()
        region.bind(row)
        xr.gate0_publish(
            region, row, matrix[row],
            [matrix[a][row] for a in range(xr.N_RANKS)], 0xABCDEF,
        )
        total = sum(matrix[row])
        xr.gate0_check(region, row, front_plan_hash=0xABCDEF, budget_s=30.0,
                       tag_totals={"weights_0": total},
                       tms_tag_bytes={"weights_0": total})
        region.mark_registered(row, log=lines.append)
        for wave in range(waves):
            pair = row % xr.N_PAIRS
            slot = wave % xr.SLOTS_PER_PAIR
            off = region.data_offset(pair, slot)
            region._mm[off: off + 4096] = bytes([row + 1]) * 4096
            result = xr.wave_gate(region, row, wave, True, budget_s=30.0,
                                  log=lines.append)
            assert result["joined"] == xr.N_RANKS
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines))
        finish.wait(120)
    finally:
        region.close()


def _two_flip_child(path: str, row: int, start, done) -> None:
    """Join every wave of flip 1, hold, then leave -- the rows stay behind."""
    region = _open(path)
    try:
        start.wait(30)
        for wave in range(3):
            region.write_gate_row(row, wave, True)
        done.wait(120)
    finally:
        region.close()


# --------------------------------------------------------------------------
# THE GATE
# --------------------------------------------------------------------------


def test_gate_names_the_non_joiner(tmp_path):
    """Five rows advance, one does not.  Assert on the STRING (spec S3)."""
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    straggler_row = xr.rank_row("D", 1)  # row 4
    procs = {}
    lines = []
    try:
        for row in range(1, xr.N_RANKS):
            waves = (0,) if row == straggler_row else (0, 1)
            p = ctx.Process(target=_gate_child, args=(region.path, row, waves, 30.0))
            p.start()
            procs[row] = p

        first = xr.wave_gate(region, 0, 0, True, budget_s=20.0, log=lines.append)
        assert first["joined"] == xr.N_RANKS, "wave 0 must close: all six rows wrote it"
        assert first["ok"] == xr.N_RANKS

        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 1, True, budget_s=1.0, log=lines.append)
        msg = str(excinfo.value)

        assert "W69 Weg2XchgGateTimeout" in msg
        assert f"joined={xr.N_RANKS - 1}/{xr.N_RANKS}" in msg, msg
        assert "group=D" in msg and "rank=1" in msg and f"row={straggler_row}" in msg, msg
        assert f"pid={procs[straggler_row].pid}" in msg, msg
        assert "wave_seen=0" in msg and "wave_wanted=1" in msg, msg
        assert "alive_in_proc=yes" in msg, msg
        assert "denominator: the six rows carrying epoch_hash=" in msg, msg
        # and it named ONLY the straggler
        for row, proc in procs.items():
            if row != straggler_row:
                assert f"pid={proc.pid}" not in msg, msg
    finally:
        for proc in procs.values():
            proc.kill()
            proc.join(10)
        region.close()


def test_gate_denominator_is_the_rows_carrying_this_epoch(tmp_path):
    """A row from the PREVIOUS flip is not a join, however high its gate_seq.

    This is the stop-loss made falsifiable: drop the epoch filter and the gate
    passes with rank D2 asleep, because last flip left every row at wave 2.
    """
    region = _make(tmp_path)
    lines = []
    try:
        stale_row = xr.rank_row("D", 2)  # row 5
        real = region.epoch_hash
        region.epoch_hash = xr.epoch_hash(STALE_EPOCH)
        region.write_gate_row(stale_row, 99, True, pid=4242)
        region.epoch_hash = real

        for row in range(1, stale_row):
            region.write_gate_row(row, 1, True)

        assert region.read_gate_row(stale_row).gate_seq > 1, (
            "the fixture must give the stale row a gate_seq that WOULD pass an "
            "unfiltered count, or it proves nothing"
        )
        assert region.read_gate_row(stale_row).sealed, (
            "the stale row must be a VALID row of the previous flip; if the seal "
            "rejected it the epoch filter would never be exercised"
        )
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 1, True, budget_s=0.4, log=lines.append)
        msg = str(excinfo.value)
        assert f"joined={xr.N_RANKS - 1}/{xr.N_RANKS}" in msg, msg
        assert f"row={stale_row}" in msg and "pid=4242" in msg, msg
        assert "denominator: the six rows carrying epoch_hash=" in msg, msg
    finally:
        region.close()


def test_a_second_flip_on_one_region_inherits_no_join(tmp_path):
    """ONE region, TWO flips: flip 2's wave 0 may not close on flip 1's rows.

    The region is created once per BOOT (385 MiB ftruncate + cudaHostRegister +
    24 sem_open belong nowhere near a 1.5 s transport), so after flip 1 all six
    rows sit at ``gate_seq=3``, sealed, alive.  If the flip stamp were the
    region's identity rather than ``begin_flip``'s, flip 2's very first poll
    would count 6/6 with five ranks still asleep and move bytes on a gate that
    nobody joined.  That is verbatim the stop-loss.
    """
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    procs = []
    lines = []
    try:
        start = ctx.Barrier(xr.N_RANKS)
        done = ctx.Barrier(xr.N_RANKS)
        for row in range(1, xr.N_RANKS):
            p = ctx.Process(target=_two_flip_child, args=(region.path, row, start, done))
            p.start()
            procs.append(p)
        start.wait(30)
        for wave in range(3):
            got = xr.wave_gate(region, 0, wave, True, budget_s=30.0, log=lines.append)
            assert got["joined"] == xr.N_RANKS
        done.wait(120)
        for p in procs:
            p.join(30)
        assert [p.exitcode for p in procs] == [0] * (xr.N_RANKS - 1)

        # flip 1 left six sealed, this-region rows at gate_seq=3.
        for row in range(xr.N_RANKS):
            assert region.read_gate_row(row).gate_seq == 3

        region.begin_flip(NEXT_EPOCH)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=0.4, log=lines.append)
        msg = str(excinfo.value)
        assert "joined=1/6" in msg, (
            "only THIS rank has joined flip 2; the other five rows are flip 1's. "
            f"got: {msg}"
        )
        assert "non_joiners" in msg, msg
        assert f"epoch={NEXT_EPOCH}" in msg, msg
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(10)
        region.close()


def test_begin_flip_refuses_a_foreign_boot_and_a_flip_that_does_not_advance(tmp_path):
    region = _make(tmp_path)
    try:
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            region.begin_flip(f"{OTHER_BOOT}.9")
        assert "is not this region's" in str(excinfo.value)

        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            region.begin_flip(EPOCH)  # the flip this view already ran
        assert "does not advance" in str(excinfo.value)

        assert region.begin_flip(NEXT_EPOCH) == xr.epoch_hash(NEXT_EPOCH)
        assert region.epoch == NEXT_EPOCH and region.flip_index == 8
    finally:
        region.close()


def test_the_flip_stamp_is_required_before_any_row_or_slot_is_written(tmp_path):
    """A region with no flip bound has the BOOT's identity, which is not a flip."""
    region = xr.XchgRegion.create(BOOT, path=str(tmp_path / "xchg.bin"))
    try:
        for call in (
            lambda: region.write_gate_row(0, 0, True),
            lambda: region.write_matrix_row(0, [0] * 6, [0] * 6, 1),
            lambda: region.begin_fill(0, 0, seq=1),
            lambda: region.publish(0, 0, 4),
            lambda: region.claim_produced(0, 0),
            lambda: xr.wave_gate(region, 0, 0, True, budget_s=0.1, log=print),
            lambda: xr.gate0_check(region, 0, budget_s=0.1,
                                   census_unavailable_reason=NO_CENSUS),
        ):
            with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
                call()
            assert "no flip is bound" in str(excinfo.value), str(excinfo.value)
        # registration is boot-scoped and must NOT need a flip
        assert region.mark_registered(0) == 1
    finally:
        region.close()


def test_a_joined_row_whose_writer_is_dead_is_not_a_join(tmp_path):
    """Six rows are not six ranks: a phantom join must be named, not counted.

    A rank that writes wave k's row and then dies would otherwise close every
    later gate 6/6 without it -- and after gate 1 the source has unmapped, so
    the flip would be committed to the W73 roll-forward by a gate that
    reported PASS.
    """
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    lines = []
    try:
        p = ctx.Process(target=_write_and_die_child, args=(region.path, 1, 0))
        p.start()
        p.join(30)
        assert p.exitcode == 0
        assert not os.path.isdir(f"/proc/{p.pid}")
        row1 = region.read_gate_row(1)
        assert row1.sealed and row1.gate_seq == 1 and row1.pid == p.pid, (
            "the fixture must leave a row that a naive count WOULD accept"
        )

        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=5.0, log=lines.append)
        msg = str(excinfo.value)
        assert "joined_but_dead" in msg, msg
        assert f"pid={p.pid}" in msg and "alive_in_proc=no" in msg, msg
        assert "row=1" in msg, msg
        # it did not sit out the budget waiting for a corpse
        assert float(msg.split("waited_s=")[1].split()[0]) < 4.0, msg
    finally:
        region.close()


def test_six_rows_written_by_one_process_are_not_six_ranks(tmp_path):
    region = _make(tmp_path)
    lines = []
    try:
        for row in range(1, xr.N_RANKS):
            region.write_gate_row(row, 0, True)  # all carry THIS pid
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=5.0, log=lines.append)
        msg = str(excinfo.value)
        assert "six_rows_are_not_six_ranks" in msg, msg
        assert "1 distinct pids" in msg, msg
        assert not lines, "a gate that refuses may not print its acceptance line"
    finally:
        region.close()


def test_a_torn_row_is_not_a_join(tmp_path):
    """The seal, not the field order, is what makes a row published.

    ``struct.pack_into`` writes ascending, so without a seal a reader can pair
    this flip's ``gate_seq`` with the previous store's ``ok`` -- a joined-and-
    failed rank that never existed, aborting a healthy flip.
    """
    region = _make(tmp_path)
    lines = []
    try:
        region.write_gate_row(3, 0, True, pid=os.getpid())
        off = xr.GATE_OFF + 3 * xr.GATE_ROW_BYTES
        raw = bytes(region._mm[off: off + xr.GATE_ROW_BYTES])
        payload, seal = raw[: xr.GATE_SEAL_OFF], struct.unpack_from("<Q", raw, xr.GATE_SEAL_OFF)[0]
        assert seal == xr._seal(payload) and seal & xr.SEAL_MARK, (
            "the seal is written last and marks itself, so a zeroed word can "
            "never validate"
        )
        assert region.read_gate_row(3).sealed

        # tear it: one payload byte moves, the seal does not
        region._mm[off] = (region._mm[off] + 1) & 0xFF
        torn = region.read_gate_row(3)
        assert not torn.sealed
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=0.3, log=lines.append)
        msg = str(excinfo.value)
        assert "row=3" in msg and "sealed=no" in msg, msg

        # the matrix row carries the same seal
        _publish_all(region, _byte_matrix())
        moff = xr.MATRIX_OFF + 2 * xr.MATRIX_ROW_BYTES
        region._mm[moff] = (region._mm[moff] + 1) & 0xFF
        assert not region.read_matrix_row(2).sealed
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.gate0_check(region, 0, budget_s=0.3, census_unavailable_reason=NO_CENSUS)
        assert "row=2" in str(excinfo.value) and "sealed=no" in str(excinfo.value)
    finally:
        region.close()


def test_wave_gate_refuses_an_ok_false_vote(tmp_path):
    """Spec section 1.3 step 17: "Any ok=False or expiry -> W69"."""
    region = _make(tmp_path)
    lines = []
    proc_root = _fake_proc(tmp_path, [os.getpid()] + [9000 + r for r in range(1, 6)])
    try:
        for row in range(1, xr.N_RANKS):
            region.write_gate_row(row, 0, row != 3, pid=9000 + row)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=5.0, log=lines.append,
                         proc_root=proc_root)
        msg = str(excinfo.value)
        assert "voted_not_ok" in msg and "row=3" in msg and "pid=9003" in msg, msg
        assert "ok=5/6" in msg, msg
    finally:
        region.close()


def test_a_refusal_votes_false_so_the_peers_do_not_wait_the_budget(tmp_path):
    """Spec section 3.6: a rank that cannot go on VOTES, it does not go quiet.

    Every W68/W69 in this module leaves by exception.  If the refusing rank
    does not also publish ``ok=False``, its five peers learn nothing until
    their own gate reaches ``WEG2_GROUP_FENCE_BUDGET_S`` = 120 s -- which is
    the fast named propagation section 3.2 sells against ``monitored_barrier``.
    """
    region = _make(tmp_path)
    try:
        region.bind(row=2, wave=1)
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            region.publish(0, 0, 4096)  # never claimed by this flip
        msg = str(excinfo.value)
        assert "failure voted: row=2 wave=1 ok=False" in msg, msg
        voted = region.read_gate_row(2)
        assert voted.sealed and voted.ok is False and voted.gate_seq == 2
        assert voted.epoch_hash == region.epoch_hash
        assert voted.state == xr.GATE_FAILED

        # and an UNBOUND view says so instead of leaving the gap silent
        other = _open(region.path)
        try:
            with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
                other.publish(1, 0, 4096)
            assert "NO failure vote published" in str(excinfo.value)
        finally:
            other.close()
    finally:
        region.close()


def test_gate_deadline_is_the_fence_budget():
    """The module READS ``WEG2_GROUP_FENCE_BUDGET_S``; it owns no literal."""
    from sglang.srt.managers.scheduler_components.weight_updater import (
        WEG2_GROUP_FENCE_BUDGET_S,
    )

    assert xr.fence_budget_s() == float(WEG2_GROUP_FENCE_BUDGET_S)

    tree = ast.parse(inspect.getsource(xr))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "WEG2_GROUP_FENCE_BUDGET_S" in (names | imported), (
        "the budget must come from weight_updater's constant, not from a "
        "number typed here -- spec section 3.2: 'No second timeout constant.'"
    )
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg, default in zip(
            args.args[len(args.args) - len(args.defaults):], args.defaults
        ):
            _assert_no_numeric_budget_default(node.name, arg.arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                _assert_no_numeric_budget_default(node.name, arg.arg, default)


def _assert_no_numeric_budget_default(func: str, arg: str, default) -> None:
    if arg != "budget_s":
        return
    assert isinstance(default, ast.Constant) and default.value is None, (
        f"{func}(budget_s=...) defaults to a literal; it must default to None "
        f"and resolve through fence_budget_s()"
    )


def test_the_gate_acceptance_line_cannot_be_silently_omitted():
    """``log`` is required: the line IS S3's acceptance criterion."""
    sig = inspect.signature(xr.wave_gate)
    param = sig.parameters["log"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty, (
        "a default of None lets a caller keep the WEG2-XCHG-GATE line out of "
        "the boot log while still passing this slice's tests"
    )


# --------------------------------------------------------------------------
# GATE 0
# --------------------------------------------------------------------------


def test_matrix_refuses_asymmetric_plan(tmp_path):
    """``send[0][1] != recv[1][0]`` -> W68 on EVERY reader, before any transfer."""
    region = _make(tmp_path)
    try:
        matrix = _byte_matrix()
        _publish_all(region, matrix)
        clean = xr.gate0_check(region, 0, front_plan_hash=0xABCDEF, budget_s=5.0,
                               census_unavailable_reason=NO_CENSUS)
        assert clean["cells"] == 36
        assert clean["tags_checked"] == f"skipped({NO_CENSUS})"
        assert clean["verdicts_ok"] == xr.N_RANKS

        row0 = region.read_matrix_row(0)
        broken = list(row0.send)
        broken[1] += 4096
        region.write_matrix_row(0, broken, row0.recv, row0.plan_hash)

        for reader in range(xr.N_RANKS):
            with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
                xr.gate0_check(region, reader, front_plan_hash=0xABCDEF, budget_s=5.0,
                               census_unavailable_reason=NO_CENSUS)
            msg = str(excinfo.value)
            assert "W68 Weg2XchgPlanDisagree" in msg, msg
            assert "send[0][1]=" in msg and "recv[1][0]=" in msg, msg
            assert "delta=4096" in msg, msg
            assert "no byte has moved" in msg, msg

        for pair in range(xr.N_PAIRS):
            for slot in range(xr.SLOTS_PER_PAIR):
                rec = region.read_slot(pair, slot)
                assert rec.state == xr.SLOT_FREE and rec.bytes_filled == 0, (
                    "Gate 0 runs before any resume and before any pause: it may "
                    "not have touched a slot"
                )
    finally:
        region.close()


def test_a_rank_local_gate0_refusal_stops_every_reader(tmp_path):
    """The tag census is checked per rank; the REFUSAL must still be global.

    A rank whose per-tag total disagrees with ``tms_tag_bytes`` raising alone
    would leave the other five to proceed and discover it only when wave 1's
    gate expired at the full 120 s budget -- after they had moved wave-1 bytes.
    Spec section 3.3: *"Any mismatch -> W68 on every reader."*
    """
    region = _make(tmp_path)
    try:
        matrix = _byte_matrix()
        _publish_all(region, matrix)
        bad_row = 4
        with pytest.raises(xr.Weg2XchgPlanDisagree) as own:
            xr.gate0_check(region, bad_row, budget_s=5.0,
                           tag_totals={"weights_0": 200},
                           tms_tag_bytes={"weights_0": 120})
        assert "tag=weights_0" in str(own.value) and "delta=80" in str(own.value)

        assert region.read_matrix_row(bad_row).local_ok is False, (
            "the verdict must be IN the shared row, or the peers cannot see it"
        )
        for reader in range(xr.N_RANKS):
            if reader == bad_row:
                continue
            with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
                xr.gate0_check(region, reader, budget_s=5.0,
                               tag_totals={"weights_0": 100},
                               tms_tag_bytes={"weights_0": 120})
            msg = str(excinfo.value)
            assert f"row={bad_row}" in msg and "local_ok=0" in msg, msg
            assert "verdicts=5/6" in msg, msg
            assert "group=D" in msg and "rank=1" in msg, msg
    finally:
        region.close()


def test_gate0_refuses_a_census_that_is_absent_without_a_reason(tmp_path):
    """Gate 0's only independent oracle may not default to off.

    The 36-cell and plan-hash checks compare the ranks against each other, so a
    common-mode derivation error passes both by construction.  An S6 that
    forgot ``tms_tag_bytes`` must not get a half-armed gate that prints a pass.
    """
    region = _make(tmp_path)
    try:
        _publish_all(region, _byte_matrix())
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, budget_s=5.0)
        msg = str(excinfo.value)
        assert "no census_unavailable_reason was declared" in msg, msg
        assert "only independent oracle" in msg, msg

        with pytest.raises(xr.Weg2XchgPlanDisagree):
            xr.gate0_check(region, 0, budget_s=5.0, tag_totals={"weights_0": 1})

        declared = xr.gate0_check(region, 0, budget_s=5.0,
                                  census_unavailable_reason="S0 probe only")
        assert declared["tags_checked"] == "skipped(S0 probe only)", (
            "a declared omission prints its reason; it never prints as agreement"
        )
    finally:
        region.close()


def test_gate0_tag_totals_are_the_pinned_interface_to_s1_s2(tmp_path):
    """The per-tag half of Gate 0: planned vs ``tms_tag_bytes``, or ABSENT."""
    region = _make(tmp_path)
    try:
        _publish_all(region, _byte_matrix())
        ok = xr.gate0_check(region, 0, tag_totals={"weights_0": 100},
                            tms_tag_bytes={"weights_0": 120}, budget_s=5.0)
        assert ok["tags_checked"] == 1

        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, tag_totals={"weights_0": 200},
                           tms_tag_bytes={"weights_0": 120}, budget_s=5.0)
        assert "tag=weights_0" in str(excinfo.value)
        region.write_matrix_verdict(0, True)

        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, tag_totals={"weights_9": 1},
                           tms_tag_bytes={}, budget_s=5.0)
        assert "tms_tag_bytes=ABSENT" in str(excinfo.value)

        with pytest.raises(ValueError):
            xr.gate0_check(region, xr.N_RANKS, budget_s=5.0,
                           census_unavailable_reason=NO_CENSUS)
    finally:
        region.close()


def test_gate0_refuses_a_plan_hash_that_is_not_the_fronts(tmp_path):
    region = _make(tmp_path)
    try:
        _publish_all(region, _byte_matrix(), plan_hash=0x1111)
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, front_plan_hash=0x2222, budget_s=5.0,
                           census_unavailable_reason=NO_CENSUS)
        assert "plan_hash=0x1111" in str(excinfo.value)
        assert region.read_matrix_row(0).local_ok is False, (
            "the front-hash check is rank-local too, so its verdict is published"
        )
    finally:
        region.close()


def test_gate0_refuses_a_reader_that_never_published_its_own_row(tmp_path):
    region = _make(tmp_path)
    try:
        matrix = _byte_matrix()
        for r in range(1, xr.N_RANKS):
            region.write_matrix_row(
                r, matrix[r], [matrix[a][r] for a in range(xr.N_RANKS)], 0xABCDEF)
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, budget_s=0.3, census_unavailable_reason=NO_CENSUS)
        assert "has not published its own matrix row" in str(excinfo.value)
    finally:
        region.close()


# --------------------------------------------------------------------------
# SLOTS
# --------------------------------------------------------------------------


def test_producer_death_after_publish_does_not_lose_the_slot(tmp_path):
    """Published bytes survive their producer; a death mid-FILL is W69."""
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    nbytes = 8192
    try:
        producer = ctx.Process(target=_producer_child,
                               args=(region.path, 0, 0, nbytes, True))
        producer.start()
        deadline = time.monotonic() + 20.0
        while region.read_slot(0, 0).state != xr.SLOT_PRODUCED:
            assert time.monotonic() < deadline, "producer never published"
            time.sleep(0.01)
        assert region.read_slot(0, 0).producer_pid == producer.pid

        os.kill(producer.pid, signal.SIGKILL)
        producer.join(10)
        assert not os.path.isdir(f"/proc/{producer.pid}")

        drained = region.claim_produced(0, 0)
        assert drained is not None, (
            "a producer that published and then died has already delivered; "
            "the bytes are in shared memory and belong to the consumer"
        )
        assert drained.bytes_filled == nbytes
        assert drained.state == xr.SLOT_DRAINING
        assert drained.consumer_pid == os.getpid()
        off = region.data_offset(0, 0)
        assert bytes(region._mm[off: off + nbytes]) == b"\xa5" * nbytes

        filler = ctx.Process(target=_producer_child,
                             args=(region.path, 1, 0, nbytes, False))
        filler.start()
        deadline = time.monotonic() + 20.0
        while region.read_slot(1, 0).state != xr.SLOT_FILLING:
            assert time.monotonic() < deadline, "producer never claimed the slot"
            time.sleep(0.01)
        os.kill(filler.pid, signal.SIGKILL)
        filler.join(10)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            region.claim_produced(1, 0)
        msg = str(excinfo.value)
        assert "W69 Weg2XchgGateTimeout" in msg and "state=FILLING" in msg, msg
        assert f"producer_pid={filler.pid}" in msg and "alive_in_proc=no" in msg, msg
    finally:
        region.close()


def test_stale_epoch_slot_is_not_adopted(tmp_path):
    """``host_ring.cpp:285-291``'s rule, ported to slots."""
    region = _make(tmp_path)
    try:
        real = region.epoch_hash
        region.epoch_hash = xr.epoch_hash(STALE_EPOCH)
        region.begin_fill(3, 1, seq=7, producer_pid=os.getpid())
        region.publish(3, 1, 4096, checksum=0xDEAD)
        region.epoch_hash = real

        assert region.claim_produced(3, 1) is None, (
            "a slot carrying another epoch is never read as this flip's funding"
        )
        rec = region.read_slot(3, 1)
        assert rec.state == xr.SLOT_FREE and rec.bytes_filled == 0
        assert rec.epoch_hash == 0 and rec.producer_pid == 0
    finally:
        region.close()


def test_publish_refuses_a_slot_this_flip_never_claimed(tmp_path):
    region = _make(tmp_path)
    try:
        with pytest.raises(xr.Weg2XchgPlanDisagree):
            region.publish(2, 0, 4096)
        region.begin_fill(2, 0, seq=1)
        with pytest.raises(xr.Weg2XchgPlanDisagree):
            region.publish(2, 0, xr.SLOT_BYTES + 1)
    finally:
        region.close()


# --------------------------------------------------------------------------
# THE REGION ITSELF
# --------------------------------------------------------------------------


def test_region_layout_is_disjoint_and_the_line_prints_its_denominator(tmp_path):
    region = _make(tmp_path)
    try:
        assert xr.SLOTS_OFF + xr.N_SLOTS * xr.SLOT_RECORD_BYTES <= xr.GATE_OFF
        assert xr.GATE_OFF + xr.N_RANKS * xr.GATE_ROW_BYTES <= xr.MATRIX_OFF
        assert xr.MATRIX_OFF + xr.N_RANKS * xr.MATRIX_ROW_BYTES <= xr.DIR_OFF
        assert xr.DIR_OFF < xr.DATA_OFF
        assert xr.REGION_BYTES == xr.DATA_OFF + xr.N_PAIRS * xr.SLOTS_PER_PAIR * xr.SLOT_BYTES
        assert os.path.getsize(region.path) == xr.REGION_BYTES
        assert region.data_offset(xr.N_PAIRS - 1, xr.SLOTS_PER_PAIR - 1) + xr.SLOT_BYTES \
            == xr.REGION_BYTES
        assert xr.GATE_SEAL_OFF + 8 == xr.GATE_ROW_BYTES
        assert xr.MATRIX_SEAL_OFF + 8 == xr.MATRIX_ROW_BYTES

        hdr = region.header()
        assert hdr["magic"] == xr.XCHG_MAGIC and hdr["n_ranks"] == 6
        assert hdr["slot_bytes"] == 32 * xr.MIB
        assert hdr["boot_hash"] == xr.epoch_hash(BOOT)

        line = xr.region_line(region, sems=24)
        assert line.startswith("WEG2-XCHG-REGION ")
        assert f"epoch={BOOT}" in line and "slots=6x2x32MiB" in line
        assert f"bytes={xr.REGION_BYTES}" in line and "sems=24" in line
        assert "scope=boot" in line
        assert "registered=0/6" in line, (
            "at create time NO rank has cudaHostRegister'ed the region; a line "
            "that printed 6/6 here would be an unarmed gate reading as a "
            "passed one"
        )
        lines = []
        for row in range(xr.N_RANKS):
            region.mark_registered(row, log=lines.append)
        assert len(lines) == 1 and "registered=6/6" in lines[0], (
            "S3's acceptance is registered=6/6; if only the launch-time line "
            "existed it could never be true, because no rank has registered yet"
        )
        region.mark_registered(0)  # idempotent: a byte, not a counter
        assert "registered=6/6" in xr.region_line(region, sems=24)
    finally:
        region.close()


def test_mark_registered_writes_only_that_ranks_own_byte(tmp_path):
    """A per-rank byte, not a bitmap -- a MEASURED lost update, not a style.

    The first implementation was ``bits |= 1 << row`` on one shared u64.  Six
    processes read-modify-writing one word drop each other's bits, and the
    six-rank double below reported ``registered=5/6`` on the remote desk
    (2026-09-08) before this was structural.
    """
    region = _make(tmp_path)
    try:
        region.mark_registered(2)
        raw = bytes(region._mm[xr.REGISTERED_OFF: xr.REGISTERED_OFF + 8])
        assert raw == b"\x00\x00\x01\x00\x00\x00\x00\x00", raw
        assert region.registered_rows() == [2]
        region.mark_registered(2)
        assert region.registered_count() == 1, "idempotent: a byte, not a counter"
        with pytest.raises(ValueError):
            region.mark_registered(xr.N_RANKS)
        assert "registered_bytes" in region.header(), (
            "the header key may not call six per-rank bytes a bitmap"
        )
    finally:
        region.close()


def test_six_processes_registering_at_once_are_all_counted(tmp_path):
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    procs = []
    try:
        start = ctx.Barrier(xr.N_RANKS)
        for row in range(xr.N_RANKS):
            p = ctx.Process(target=_register_child, args=(region.path, row, start))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(60)
        assert [p.exitcode for p in procs] == [0] * xr.N_RANKS
        assert region.registered_rows() == list(range(xr.N_RANKS))
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(10)
        region.close()


def test_open_names_the_boot_it_expects_and_refuses_every_other(tmp_path):
    """The anti-adoption guard is REQUIRED, never a default a caller forgets."""
    region = _make(tmp_path)
    region.close()
    path = str(tmp_path / "xchg.bin")

    assert inspect.signature(xr.XchgRegion.open).parameters["expect_boot"].default \
        is inspect.Parameter.empty, "a default makes the guard opt-in"
    with pytest.raises(TypeError):
        xr.XchgRegion.open(path)

    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        xr.XchgRegion.open(path, expect_boot=OTHER_BOOT)
    assert "never adopted as this one's" in str(excinfo.value)
    reopened = xr.XchgRegion.open(path, expect_boot=BOOT)
    assert reopened.boot_nonce == BOOT and reopened.epoch == "" and reopened.epoch_hash == 0
    reopened.close()


def test_open_refuses_a_region_whose_geometry_is_not_this_builds(tmp_path):
    """The header's geometry is checked, not merely printed.

    ``region_line`` prints the FILE's ``slot_bytes`` while every offset in the
    module strides its own constant.  A peer built from another tree with the
    same version would otherwise be adopted silently and addressed with the
    wrong stride.
    """
    region = _make(tmp_path)
    path = region.path
    region.close()
    with open(path, "r+b") as fh:
        fh.seek(4 * 8)  # header word 4: slot_bytes
        fh.write(struct.pack("<Q", 64 * xr.MIB))
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        xr.XchgRegion.open(path, expect_boot=BOOT)
    msg = str(excinfo.value)
    assert "header slot_bytes=" in msg and "not the geometry" in msg, msg

    with open(path, "r+b") as fh:
        fh.seek(4 * 8)
        fh.write(struct.pack("<Q", xr.SLOT_BYTES))
        fh.seek(2 * 8)  # header word 2: the boot hash
        fh.write(struct.pack("<Q", xr.epoch_hash(OTHER_BOOT)))
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        xr.XchgRegion.open(path, expect_boot=BOOT)
    assert "the string and the hash the creator wrote disagree" in str(excinfo.value)


def test_create_refuses_to_re_zero_an_existing_region(tmp_path):
    """A second create would wipe a LIVE boot's gates and report success."""
    region = _make(tmp_path)
    try:
        region.write_gate_row(1, 0, True)
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.XchgRegion.create(BOOT, path=region.path)
        assert "already exists" in str(excinfo.value)
        assert region.read_gate_row(1).gate_seq == 1, "and it did not touch the rows"
    finally:
        region.close()


def test_epoch_hash_is_stable_across_processes_and_never_zero():
    assert xr.epoch_hash(EPOCH) == xr.epoch_hash(EPOCH)
    assert xr.epoch_hash(EPOCH) != xr.epoch_hash(STALE_EPOCH)
    assert xr.epoch_hash(EPOCH) != xr.epoch_hash(BOOT), (
        "the flip stamp and the region's own identity must never collide"
    )
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        assert pool.apply(xr.epoch_hash, (EPOCH,)) == xr.epoch_hash(EPOCH), (
            "PYTHONHASHSEED randomises hash() per process; six ranks are six "
            "processes and every epoch filter in this module depends on them "
            "agreeing"
        )
    assert xr.split_flip_epoch(EPOCH) == (BOOT, 7)
    with pytest.raises(ValueError):
        xr.split_flip_epoch("no-flip-index")


# --------------------------------------------------------------------------
# THE 24 NAMED SEMAPHORES
# --------------------------------------------------------------------------


def test_semaphores_are_created_and_unlinked():
    boot = f"1788900000sem{os.getpid()}"
    names = []
    try:
        names = xr.create_semaphores(boot)
        assert len(names) == 24, "6 directed pairs x 2 slots x {empty, full}"
        assert len(set(names)) == 24
        for name in names:
            assert os.path.exists(f"/dev/shm/sem.{name.lstrip('/')}"), name
        assert xr.unlink_semaphores(boot) == 24
        for name in names:
            assert not os.path.exists(f"/dev/shm/sem.{name.lstrip('/')}"), name
        assert xr.unlink_semaphores(boot) == 0, "unlink is idempotent"
    finally:
        if names:
            xr.unlink_semaphores(boot)


def test_the_boot_nonce_is_in_the_semaphore_name(tmp_path):
    """The nonce in the NAME is the only thing keeping two boots apart.

    Every other property of the 24 names -- count, uniqueness, creation,
    idempotent unlink -- holds identically with the nonce deleted from the
    format string, so nothing else in this file would notice.
    """
    name = xr.sem_name(BOOT, 0, 0, "empty")
    assert name == f"/{xr.REGION_PREFIX}{BOOT}-0-1-0-empty", name
    assert BOOT in name
    mine, theirs = set(xr.all_sem_names(BOOT)), set(xr.all_sem_names(OTHER_BOOT))
    assert len(mine) == len(theirs) == 24
    assert mine.isdisjoint(theirs), (
        "two boots' handshakes must not alias; without the nonce every name "
        "collides and a dead boot's waiters meet a live boot's producers"
    )
    for n in mine:
        assert BOOT in n
    with pytest.raises(ValueError):
        xr.sem_name(BOOT, 0, 0, "neither")


def test_create_semaphores_never_adopts_a_stale_count():
    """``sem_open(O_CREAT)`` IGNORES ``value`` for a name that already exists.

    A surviving ``empty`` at 0 would make the boot's first producer block to
    the 120 s fence budget instead of the launch refusing, and nothing would
    say the initial values were never applied.  So: unlink first, then
    ``O_EXCL``.
    """
    boot = f"1788900000stale{os.getpid()}"
    lib = xr._libc()
    lib.sem_getvalue.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.sem_wait.argtypes = [ctypes.c_void_p]
    name = xr.sem_name(boot, 0, 0, "empty").encode("ascii")

    def value_of() -> int:
        handle = lib.sem_open(name, 0)
        assert handle not in (None, 0, xr._SEM_FAILED), ctypes.get_errno()
        out = ctypes.c_int(-1)
        lib.sem_getvalue(ctypes.c_void_p(handle), ctypes.byref(out))
        lib.sem_close(ctypes.c_void_p(handle))
        return out.value

    try:
        xr.create_semaphores(boot)
        assert value_of() == 1, "empty starts at 1"
        handle = lib.sem_open(name, 0)
        assert lib.sem_wait(ctypes.c_void_p(handle)) == 0
        lib.sem_close(ctypes.c_void_p(handle))
        assert value_of() == 0, "the fixture must leave a DRAINED semaphore"

        xr.create_semaphores(boot)
        assert value_of() == 1, (
            "a re-created handshake must carry its initial values, not the "
            "previous attempt's counts"
        )
    finally:
        xr.unlink_semaphores(boot)


def test_rank_row_round_trips_and_refuses_nonsense():
    seen = set()
    for group in ("P", "D"):
        for rank in range(xr.N_CARDS):
            row = xr.rank_row(group, rank)
            seen.add(row)
            assert xr.row_group_rank(row) == (group, rank)
    assert seen == set(range(xr.N_RANKS))
    with pytest.raises(ValueError):
        xr.rank_row("X", 0)
    with pytest.raises(ValueError):
        xr.rank_row("P", 3)
    with pytest.raises(ValueError):
        xr.pair_id(1, 1)


# --------------------------------------------------------------------------
# THE HERMETIC DOUBLE -- six children, memcpy in place of every CUDA call
# --------------------------------------------------------------------------


def test_six_ranks_close_three_waves(tmp_path):
    region = _make(tmp_path)
    ctx = mp.get_context("fork")
    procs = []
    try:
        outs = [str(tmp_path / f"rank{row}.log") for row in range(xr.N_RANKS)]
        finish = ctx.Barrier(xr.N_RANKS)
        for row in range(xr.N_RANKS):
            p = ctx.Process(target=_rank_child,
                            args=(region.path, row, outs[row], 3, finish))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(120)
        assert [p.exitcode for p in procs] == [0] * xr.N_RANKS, (
            f"exit codes {[p.exitcode for p in procs]}"
        )
        seen_region_line = 0
        for row, out in enumerate(outs):
            with open(out) as fh:
                lines = fh.read().splitlines()
            gates = [ln for ln in lines if ln.startswith("WEG2-XCHG-GATE ")]
            regions = [ln for ln in lines if ln.startswith("WEG2-XCHG-REGION ")]
            seen_region_line += len(regions)
            for line in regions:
                assert "registered=6/6" in line, line
            assert len(gates) == 3, f"rank {row}: {lines}"
            for wave, line in enumerate(gates):
                assert f"epoch={EPOCH}" in line
                assert f"wave={wave}" in line and "joined=6/6" in line and "ok=6/6" in line
                assert "skew_ms=" in line
                assert "(denominator: the six rows carrying this epoch)" in line
        assert seen_region_line >= 1, (
            "the rank whose byte completes the six re-emits the region line, or "
            "S3's registered=6/6 acceptance can never be observed at all"
        )
        assert region.registered_count() == xr.N_RANKS
        for row in range(xr.N_RANKS):
            assert region.read_gate_row(row).gate_seq == 3
            assert region.read_gate_row(row).epoch_hash == region.epoch_hash
            assert region.read_gate_row(row).sealed
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(10)
        region.close()


# --------------------------------------------------------------------------
# THE LAUNCHER HALF
# --------------------------------------------------------------------------


def test_launcher_owns_both_xchg_shm_families(tmp_path, monkeypatch):
    """The region AND ``sem.<region>``; a foreign name is never touched."""
    from sglang.srt.weg2 import launcher

    assert xr.REGION_PREFIX in launcher.SHM_OWN_PREFIXES
    assert f"sem.{xr.REGION_PREFIX}" in launcher.SHM_OWN_PREFIXES

    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / f"{xr.REGION_PREFIX}{BOOT}").mkdir()
    (shm / f"{xr.REGION_PREFIX}{BOOT}" / "xchg.bin").write_bytes(b"\x00" * 64)
    (shm / f"sem.{xr.REGION_PREFIX}{BOOT}-0-1-0-empty").write_bytes(b"\x00" * 32)
    (shm / "sem.mp-someoneelse").write_bytes(b"\x00" * 32)

    class _NoPids:
        stdout = ""

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _NoPids())
    lines = []
    result = launcher.shm_residue_sweep(
        lines.append, "t", "s", True, shm_dir=str(shm), proc_root=str(tmp_path / "noproc")
    )
    swept = set(result["swept"])
    assert f"{xr.REGION_PREFIX}{BOOT}" in swept
    assert f"sem.{xr.REGION_PREFIX}{BOOT}-0-1-0-empty" in swept
    assert "sem.mp-someoneelse" not in swept, "a foreign name is never ours"
    text = "\n".join(lines)
    assert "WEG2-XCHG-SEM DRY-RUN" in text and "0-1-0-empty" in text
    assert "mp-someoneelse" not in text


def test_the_residue_sweep_sem_unlinks_before_it_moves(tmp_path, monkeypatch):
    """THE NON-DRY PATH, END TO END -- the one that runs at a real boot.

    Previously ``sweep_xchg_semaphores`` was a second call in ``main()`` AFTER
    ``shm_residue_sweep``.  By then the residue sweep had ``shutil.move``d
    every ``sem.weg2-xchg-*`` file into the archive, so the sem sweep re-listed
    /dev/shm, found nothing, and logged ``residue: none`` on every boot -- and
    the kernel semaphore objects stayed, which is what ``sem_unlink`` exists
    for.  Both of the tests that looked like coverage ran only ``dry=True``,
    the one combination in which the interference cannot appear.
    """
    from sglang.srt.weg2 import launcher

    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / f"{xr.REGION_PREFIX}{BOOT}").mkdir()
    (shm / f"{xr.REGION_PREFIX}{BOOT}" / "xchg.bin").write_bytes(b"\x00" * 64)
    sem_file = shm / f"sem.{xr.REGION_PREFIX}{BOOT}-0-1-0-empty"
    sem_file.write_bytes(b"\x00" * 32)
    foreign = shm / "sem.mp-someoneelse"
    foreign.write_bytes(b"\x00" * 32)

    class _NoPids:
        stdout = ""

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _NoPids())

    unlinked = []

    def fake_unlink(posix_name: str) -> int:
        # the real sem_unlink removes the backing file too; the double must
        # behave the same or this test would not see the interference
        unlinked.append(posix_name)
        os.unlink(str(shm / f"sem.{posix_name.lstrip('/')}"))
        return 0

    archive_root = tmp_path / "archive"
    lines = []
    result = launcher.shm_residue_sweep(
        lines.append, "t", "s", False, shm_dir=str(shm),
        proc_root=str(tmp_path / "noproc"), archive_root=str(archive_root),
        sem_unlink=fake_unlink,
    )
    assert unlinked == [f"/{xr.REGION_PREFIX}{BOOT}-0-1-0-empty"], (
        f"the sem_unlink never ran: {unlinked!r}; log was {lines!r}"
    )
    assert result["sems_unlinked"] == unlinked
    assert not sem_file.exists()
    archived = [p.name for p in archive_root.rglob("*")]
    assert not any(n.startswith(f"sem.{xr.REGION_PREFIX}") for n in archived), (
        f"a named semaphore was archived instead of unlinked: {archived}"
    )
    assert f"{xr.REGION_PREFIX}{BOOT}" in result["swept"], "the region is still swept"
    assert foreign.exists(), "a foreign name is never ours"
    assert any("WEG2-XCHG-SEM residue swept: 1/1" in ln for ln in lines), lines


def test_the_semaphore_sweep_lives_inside_the_residue_sweep():
    """Ordering, structural: after the refusals, before the move, once.

    ``shm_residue_sweep`` is the only call that refuses the boot while another
    ``launch_server`` lives, so unlinking must come after that refusal;
    ``shutil.move`` renames the backing file out from under the name, so it
    must come before that.  A caller outside the function can satisfy only one
    of the two.
    """
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.shm_residue_sweep)
    assert "sweep_xchg_semaphores(" in src
    assert src.index("LIVE HOLDER") < src.index("sweep_xchg_semaphores("), (
        "the live-holder refusal must still come first"
    )
    assert src.index("sweep_xchg_semaphores(") < src.index("shutil.move("), (
        "a moved file cannot be sem_unlink'ed by name"
    )
    assert "sweep_xchg_semaphores(" not in inspect.getsource(launcher.main), (
        "a second call after the sweep can never find anything -- that was the "
        "defect, and the ordering test that pinned it pinned the breakage"
    )


def test_prepare_and_teardown_round_trip_the_region_and_the_sems(tmp_path):
    boot = f"1788900000prep{os.getpid()}"
    shm_root = str(tmp_path)
    lines = []
    try:
        out = xr.prepare_region(boot, shm_root=shm_root, log=lines.append)
        assert out["sems"] == 24
        assert out["env"][xr.ENV_REGION_PATH] == xr.region_path(boot, shm_root)
        assert out["env"][xr.ENV_REGION_BOOT] == boot
        assert os.path.getsize(out["path"]) == xr.REGION_BYTES
        assert lines and lines[0].startswith("WEG2-XCHG-REGION ")
        assert "sems=24" in lines[0], "the count is read back from the header"
        region = xr.XchgRegion.open(out["path"], expect_boot=boot)
        region.close()
    finally:
        removed = xr.teardown_region(boot, shm_root=shm_root, log=lines.append)
    assert removed == {"sems": 24, "region": 1}
    assert not os.path.exists(xr.region_path(boot, shm_root))
    assert any("WEG2-XCHG-TEARDOWN" in line for line in lines)


def test_prepare_leaves_no_region_behind_when_the_handshake_fails(tmp_path, monkeypatch):
    """385 MiB with no semaphores would make the NEXT launch refuse by O_EXCL."""
    boot = f"1788900000fail{os.getpid()}"
    shm_root = str(tmp_path)

    def boom(_boot):
        raise OSError("no space for a semaphore")

    monkeypatch.setattr(xr, "create_semaphores", boom)
    with pytest.raises(OSError):
        xr.prepare_region(boot, shm_root=shm_root)
    assert not os.path.exists(xr.region_path(boot, shm_root))


def test_the_two_wcodes_are_the_ones_the_spec_allocated():
    """W68/W69, one code one name -- the census test pins the rest."""
    src = inspect.getsource(xr)
    assert "W68 Weg2XchgPlanDisagree" in src and "W69 Weg2XchgGateTimeout" in src
    for taken in ("W50 ", "W49 ", "W35 "):
        assert f"{taken}Weg2Xchg" not in src
    assert "W70" in src, (
        "W70 Weg2XchgShortPiece is S4's, and the widening of W68 past spec "
        "section 7's row for it is stated where it happens"
    )
