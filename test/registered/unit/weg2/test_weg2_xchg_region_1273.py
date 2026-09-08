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
must be the rows carrying THIS epoch."*  Two tests are built to fail on that
one sentence rather than on the happy path:

* :func:`test_gate_names_the_non_joiner` -- five rows advance, one does not.
* :func:`test_gate_denominator_is_the_rows_carrying_this_epoch` -- six rows all
  carry a gate_seq high enough, but one of them was written by the PREVIOUS
  flip.  A count that forgets the epoch filter reads 6/6 and lets a rank that
  has not woken up this flip be counted as a joiner.  That is the
  ``credit_epoch`` failure (boot weg2rg2's three terminal counters, one with
  ``{"epoch": 12, ... "leg_complete": true}``) one mechanism over.

The hermetic double is the spec's: six ``multiprocessing`` children under
``CUDA_VISIBLE_DEVICES=""``, a region in ``tmp_path``, and ``memcpy`` in place
of every CUDA call.
"""

from __future__ import annotations

import ast
import inspect
import multiprocessing as mp
import os
import signal
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.weg2 import weight_exchange_region as xr

EPOCH = "1788900000.7"
STALE_EPOCH = "1788000000.11"


def _make(tmp_path, epoch: str = EPOCH) -> xr.XchgRegion:
    return xr.XchgRegion.create(epoch, path=str(tmp_path / "xchg.bin"))


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


# --------------------------------------------------------------------------
# children of the hermetic double.  Module level so any start method works.
# --------------------------------------------------------------------------


def _gate_child(path: str, row: int, waves, hold_s: float) -> None:
    region = xr.XchgRegion.open(path)
    try:
        for wave in waves:
            region.write_gate_row(row, wave, True)
        time.sleep(hold_s)
    finally:
        region.close()


def _producer_child(path: str, pair: int, slot: int, nbytes: int, publish: bool) -> None:
    region = xr.XchgRegion.open(path)
    region.begin_fill(pair, slot, seq=1)
    if publish:
        off = region.data_offset(pair, slot)
        region._mm[off: off + nbytes] = b"\xa5" * nbytes  # memcpy in place of CUDA
        region.publish(pair, slot, nbytes, checksum=0xA5A5)
    while True:  # the parent SIGKILLs us; a clean exit would prove nothing
        time.sleep(0.05)


def _register_child(path: str, row: int, start) -> None:
    region = xr.XchgRegion.open(path)
    try:
        start.wait(30)
        region.mark_registered(row)
    finally:
        region.close()


def _rank_child(path: str, row: int, out_path: str, waves: int) -> None:
    """One rank of the six-rank double: Gate 0, then ``waves`` wave gates."""
    lines = []
    region = xr.XchgRegion.open(path)
    try:
        matrix = _byte_matrix()
        xr.gate0_publish(
            region, row, matrix[row],
            [matrix[a][row] for a in range(xr.N_RANKS)], 0xABCDEF,
        )
        xr.gate0_check(region, row, front_plan_hash=0xABCDEF, budget_s=30.0)
        region.mark_registered(row)
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
    try:
        for row in range(1, xr.N_RANKS):
            waves = (0,) if row == straggler_row else (0, 1)
            p = ctx.Process(target=_gate_child, args=(region.path, row, waves, 30.0))
            p.start()
            procs[row] = p

        first = xr.wave_gate(region, 0, 0, True, budget_s=20.0)
        assert first["joined"] == xr.N_RANKS, "wave 0 must close: all six rows wrote it"
        assert first["ok"] == xr.N_RANKS

        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 1, True, budget_s=1.0)
        msg = str(excinfo.value)

        assert "W53 Weg2XchgGateTimeout" in msg
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
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 1, True, budget_s=0.4)
        msg = str(excinfo.value)
        assert f"joined={xr.N_RANKS - 1}/{xr.N_RANKS}" in msg, msg
        assert f"row={stale_row}" in msg and "pid=4242" in msg, msg
        assert "denominator: the six rows carrying epoch_hash=" in msg, msg
    finally:
        region.close()


def test_wave_gate_refuses_an_ok_false_vote(tmp_path):
    """Spec section 1.3 step 17: "Any ok=False or expiry -> W53"."""
    region = _make(tmp_path)
    try:
        for row in range(1, xr.N_RANKS):
            region.write_gate_row(row, 0, row != 3, pid=9000 + row)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            xr.wave_gate(region, 0, 0, True, budget_s=5.0)
        msg = str(excinfo.value)
        assert "voted_not_ok" in msg and "row=3" in msg and "pid=9003" in msg, msg
        assert "ok=5/6" in msg, msg
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


# --------------------------------------------------------------------------
# GATE 0
# --------------------------------------------------------------------------


def test_matrix_refuses_asymmetric_plan(tmp_path):
    """``send[0][1] != recv[1][0]`` -> W52 on EVERY reader, before any transfer."""
    region = _make(tmp_path)
    try:
        matrix = _byte_matrix()
        _publish_all(region, matrix)
        clean = xr.gate0_check(region, 0, front_plan_hash=0xABCDEF, budget_s=5.0)
        assert clean["cells"] == 36 and clean["tags_checked"] == "skipped"

        row0 = region.read_matrix_row(0)
        broken = list(row0.send)
        broken[1] += 4096
        region.write_matrix_row(0, broken, row0.recv, row0.plan_hash)

        for reader in range(xr.N_RANKS):
            with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
                xr.gate0_check(region, reader, front_plan_hash=0xABCDEF, budget_s=5.0)
            msg = str(excinfo.value)
            assert "W52 Weg2XchgPlanDisagree" in msg, msg
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

        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, tag_totals={"weights_9": 1},
                           tms_tag_bytes={}, budget_s=5.0)
        assert "tms_tag_bytes=ABSENT" in str(excinfo.value)

        skipped = xr.gate0_check(region, 0, budget_s=5.0)
        assert skipped["tags_checked"] == "skipped", (
            "an unsupplied census must print as skipped, never as agreement"
        )
    finally:
        region.close()


def test_gate0_refuses_a_plan_hash_that_is_not_the_fronts(tmp_path):
    region = _make(tmp_path)
    try:
        _publish_all(region, _byte_matrix(), plan_hash=0x1111)
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            xr.gate0_check(region, 0, front_plan_hash=0x2222, budget_s=5.0)
        assert "plan_hash=0x1111" in str(excinfo.value)
    finally:
        region.close()


# --------------------------------------------------------------------------
# SLOTS
# --------------------------------------------------------------------------


def test_producer_death_after_publish_does_not_lose_the_slot(tmp_path):
    """Published bytes survive their producer; a death mid-FILL is W53."""
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
        assert "W53 Weg2XchgGateTimeout" in msg and "state=FILLING" in msg, msg
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

        hdr = region.header()
        assert hdr["magic"] == xr.XCHG_MAGIC and hdr["n_ranks"] == 6
        assert hdr["slot_bytes"] == 32 * xr.MIB

        line = xr.region_line(region, sems=24)
        assert line.startswith("WEG2-XCHG-REGION ")
        assert f"epoch={EPOCH}" in line and "slots=6x2x32MiB" in line
        assert f"bytes={xr.REGION_BYTES}" in line and "sems=24" in line
        assert "registered=0/6" in line, (
            "at create time NO rank has cudaHostRegister'ed the region; a line "
            "that printed 6/6 here would be an unarmed gate reading as a "
            "passed one"
        )
        for row in range(xr.N_RANKS):
            region.mark_registered(row)
        region.mark_registered(0)  # idempotent: a bitmap, not a counter
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


def test_open_refuses_a_previous_boots_region(tmp_path):
    region = _make(tmp_path)
    region.close()
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        xr.XchgRegion.open(str(tmp_path / "xchg.bin"), expect_epoch="1788900000.8")
    assert "never adopted as this one's" in str(excinfo.value)
    reopened = xr.XchgRegion.open(str(tmp_path / "xchg.bin"), expect_epoch=EPOCH)
    assert reopened.epoch == EPOCH
    reopened.close()


def test_epoch_hash_is_stable_across_processes_and_never_zero():
    assert xr.epoch_hash(EPOCH) == xr.epoch_hash(EPOCH)
    assert xr.epoch_hash(EPOCH) != xr.epoch_hash(STALE_EPOCH)
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        assert pool.apply(xr.epoch_hash, (EPOCH,)) == xr.epoch_hash(EPOCH), (
            "PYTHONHASHSEED randomises hash() per process; six ranks are six "
            "processes and every epoch filter in this module depends on them "
            "agreeing"
        )


def test_semaphores_are_created_and_unlinked():
    epoch = f"1788900000.sem{os.getpid()}"
    names = []
    try:
        names = xr.create_semaphores(epoch)
        assert len(names) == 24, "6 directed pairs x 2 slots x {empty, full}"
        assert len(set(names)) == 24
        for name in names:
            assert os.path.exists(f"/dev/shm/sem.{name.lstrip('/')}"), name
        assert xr.unlink_semaphores(epoch) == 24
        for name in names:
            assert not os.path.exists(f"/dev/shm/sem.{name.lstrip('/')}"), name
        assert xr.unlink_semaphores(epoch) == 0, "unlink is idempotent"
    finally:
        if names:
            xr.unlink_semaphores(epoch)


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
        for row in range(xr.N_RANKS):
            p = ctx.Process(target=_rank_child, args=(region.path, row, outs[row], 3))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(120)
        assert [p.exitcode for p in procs] == [0] * xr.N_RANKS, (
            f"exit codes {[p.exitcode for p in procs]}"
        )
        for row, out in enumerate(outs):
            with open(out) as fh:
                lines = fh.read().splitlines()
            assert len(lines) == 3, f"rank {row}: {lines}"
            for wave, line in enumerate(lines):
                assert line.startswith("WEG2-XCHG-GATE ")
                assert f"wave={wave}" in line and "joined=6/6" in line and "ok=6/6" in line
                assert "skew_ms=" in line
                assert "(denominator: the six rows carrying this epoch)" in line
        assert region.registered_count() == xr.N_RANKS
        for row in range(xr.N_RANKS):
            assert region.read_gate_row(row).gate_seq == 3
            assert region.read_gate_row(row).epoch_hash == region.epoch_hash
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
    (shm / f"{xr.REGION_PREFIX}{EPOCH}").mkdir()
    (shm / f"{xr.REGION_PREFIX}{EPOCH}" / "xchg.bin").write_bytes(b"\x00" * 64)
    (shm / f"sem.{xr.REGION_PREFIX}{EPOCH}-0-1-0-empty").write_bytes(b"\x00" * 32)
    (shm / "sem.mp-someoneelse").write_bytes(b"\x00" * 32)

    class _NoPids:
        stdout = ""

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _NoPids())
    lines = []
    result = launcher.shm_residue_sweep(
        lines.append, "t", "s", True, shm_dir=str(shm), proc_root=str(tmp_path / "noproc")
    )
    swept = set(result["swept"])
    assert f"{xr.REGION_PREFIX}{EPOCH}" in swept
    assert f"sem.{xr.REGION_PREFIX}{EPOCH}-0-1-0-empty" in swept
    assert "sem.mp-someoneelse" not in swept, "a foreign name is never ours"

    lines = []
    launcher.sweep_xchg_semaphores(lines.append, shm_dir=str(shm), dry=True)
    text = "\n".join(lines)
    assert "WEG2-XCHG-SEM DRY-RUN" in text and "0-1-0-empty" in text
    assert "mp-someoneelse" not in text


def test_semaphore_sweep_runs_after_the_residue_sweep(tmp_path):
    """Ordering, pinned in the source: the live-boot refusal comes first.

    ``shm_residue_sweep`` is the only call that refuses the boot while another
    ``launch_server`` lives.  Unlinking a LIVE boot's 24 semaphores would be a
    strictly worse outcome than leaving a dead boot's behind, so the order is
    an invariant, not a preference.
    """
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    residue = src.index("state.shm_sweep = shm_residue_sweep(")
    sems = src.index("sweep_xchg_semaphores(log, dry=dry)")
    assert residue < sems, "sweep_xchg_semaphores must not precede shm_residue_sweep"


def test_prepare_and_teardown_round_trip_the_region_and_the_sems(tmp_path):
    epoch = f"1788900000.prep{os.getpid()}"
    shm_root = str(tmp_path)
    lines = []
    try:
        out = xr.prepare_region(epoch, shm_root=shm_root, log=lines.append)
        assert out["sems"] == 24
        assert out["env"][xr.ENV_REGION_PATH] == xr.region_path(epoch, shm_root)
        assert out["env"][xr.ENV_REGION_EPOCH] == epoch
        assert os.path.getsize(out["path"]) == xr.REGION_BYTES
        assert lines and lines[0].startswith("WEG2-XCHG-REGION ")
        region = xr.XchgRegion.open(out["path"], expect_epoch=epoch)
        region.close()
    finally:
        removed = xr.teardown_region(epoch, shm_root=shm_root, log=lines.append)
    assert removed == {"sems": 24, "region": 1}
    assert not os.path.exists(xr.region_path(epoch, shm_root))
    assert any("WEG2-XCHG-TEARDOWN" in line for line in lines)


def test_the_two_wcodes_are_the_ones_the_spec_allocated():
    """W52/W53, one code one name -- the census test pins the rest."""
    src = inspect.getsource(xr)
    assert "W52 Weg2XchgPlanDisagree" in src and "W53 Weg2XchgGateTimeout" in src
    for taken in ("W50 ", "W49 ", "W35 "):
        assert f"{taken}Weg2Xchg" not in src
