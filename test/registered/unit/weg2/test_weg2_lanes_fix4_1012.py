# SPDX-License-Identifier: Apache-2.0
"""PRIORITY LANES, fix round 4 (NF metal 5, image int23lanes4 @ 8bfd25d842, boot 06:36:51Z, done/lanes-metall5-1012.md).

The finding (MAJOR, reproduced twice, probe 2b lage (i)): on group P no ``WEG2-PARK (lane)`` ever appears.  The lane-0 continuation
(weg2-2-8, 98 000 tokens, chunked) IS held at the chunk boundary on all three stages (``PR LANE-FLOOR`` eff=at=stamped+1, no chunk
after fwd_ct 9, #1004 = 0), but it gives its rows back only when the front aborts it (``LANE-P-TAKE``, ``WEG2-PP-CHUNKED-ABORT``).

ROOT (found in the code, consistent with the P log; no log line prints the counter): ``Scheduler._get_new_batch_prefill_raw`` counted
``self.chunked_req.inflight_middle_chunks += 1`` for EVERY non-empty batch while a chunked request exists -- also for a continuation
the lane floor holds in place, which is NOT in the batch (``add_chunked_req`` parks it before ``can_run_list``).  In the very pass
that starts the hold, the higher lane (weg2-3-9, 16 000 tokens) is admitted, the batch is non-empty, and the held continuation is
counted for a chunk that was never launched.  Nothing ever takes that count back (the result processor decrements the requests of
``batch.reqs`` only), so ``process_pending_weg2_park`` -- which waits for ``inflight_middle_chunks == 0`` ("a launched chunk still
writes its rows") -- returns at every step.  The 17 lane tests of L3/fix1-3 all built the continuation with
``inflight_middle_chunks = 0`` and never ran the batch accounting.

The same leak has a worse second face (silent so far, because the front took the request away first): a held continuation whose floor
falls BEFORE the rows went back resumes in place, and its FINAL chunk (no increment, ``chunked_req`` is None for the last chunk) is
classified as a MIDDLE chunk by ``process_batch_result_prefill`` (``inflight_middle_chunks <= 0`` is the "prefill finished" test): the
first token is never produced.

What is pinned here: (1) the real accounting statement of the scheduler (taken out of the real method by AST, executed on a stand-in);
(2) the metal-5 sequence on three PP stages of the form without a row carrier, through the real stamp/absorb functions, the real
``PrefillAdder.add_chunked_req`` and the real ``Scheduler.process_pending_weg2_park``: every stage gives its rows back in the SAME
pass, the batches stay identical on all stages; (3) the resume in place; (4) R1: nothing changes for a pressure park or a continuation
that takes a seat; (5) the ``PARK-WAIT`` marker; (6) the phase line of the D lane park (minor 902 ms).
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_lanes_l3_1008 import (  # noqa: E402
    CHUNK, N_TOK, SPAN, _LanesOn, _adder, _continuation_on, _park_stage, _req, _rpc,
)

import pytest  # noqa: E402

from test_weg2_lanes_l2_1008 import _Sched, _park_lane, lanes_on  # noqa: E402,F401  (the fixture + stand-ins of the L2 file)
from test_weg2_lanes_l2_1008 import _req as _l2_req  # noqa: E402

from sglang.srt.managers import scheduler as sch  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402
from sglang.srt.weg2 import lanes_p  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as pr  # noqa: E402


# ---------------------------------------------------------------------------
# the REAL accounting statement of the batch builder
# ---------------------------------------------------------------------------

def _accounting_statement():
    """The statement of ``Scheduler._get_new_batch_prefill_raw`` that counts ``self.chunked_req.inflight_middle_chunks`` for the batch
    it just built, cut out by AST (the 4000-line pass cannot be driven in a unit test) and compiled as ``frag(self, can_run_list)``."""
    src = textwrap.dedent(inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw))
    tree = ast.parse(src)
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AugAssign) and ast.unparse(stmt.target) == "self.chunked_req.inflight_middle_chunks":
                best = node
                break
        if best is not None:
            break
    assert best is not None, "the accounting statement is gone from _get_new_batch_prefill_raw"
    code = "def frag(self, can_run_list):\n" + textwrap.indent(ast.unparse(best), "    ") + "\n"
    ns = dict(vars(sch))
    exec(compile(code, "<accounting-statement>", "exec"), ns)
    return ns["frag"]


def _account(sched, can_run_list):
    _accounting_statement()(sched, can_run_list)


def _held_continuation(inflight=0, *, lane_parked=True, pool_parked=True):
    req = _req("R", 0, prefix=SPAN)
    req.inflight_middle_chunks = inflight
    req.weg2_lane_parked = lane_parked
    req.weg2_pool_parked = pool_parked
    return req


class AccountingStatementTest(unittest.TestCase):
    """The statement itself, on the real code of the scheduler."""

    def test_a_held_continuation_that_is_not_in_the_batch_is_not_counted(self):
        """RED on 8bfd25d842: the held lane-0 continuation got +1 in the pass that admitted the higher lane."""
        held = _held_continuation(inflight=1)
        spur1 = _req("S", 1)
        _account(SimpleNamespace(chunked_req=held), [spur1])
        self.assertEqual(held.inflight_middle_chunks, 1, "no chunk of it was launched: nothing to count")

    def test_a_continuation_that_takes_a_chunk_is_counted_as_ever(self):
        req = _held_continuation(inflight=0, lane_parked=False, pool_parked=False)
        _account(SimpleNamespace(chunked_req=req), [req])
        self.assertEqual(req.inflight_middle_chunks, 1)

    def test_a_lane_flagged_continuation_that_the_schedule_names_is_counted(self):
        """The row-carrier form: the forwarded schedule names the continuation, it is in the batch (flag may be stale)."""
        req = _held_continuation(inflight=0)
        _account(SimpleNamespace(chunked_req=req), [_req("S", 1), req])
        self.assertEqual(req.inflight_middle_chunks, 1)

    def test_r1_a_pressure_park_is_counted_exactly_as_before(self):
        """The #679 pressure park is not a lane park: its accounting is not touched (R1, default path)."""
        req = _held_continuation(inflight=0, lane_parked=False, pool_parked=True)
        _account(SimpleNamespace(chunked_req=req), [_req("S", 0)])
        self.assertEqual(req.inflight_middle_chunks, 1)

    def test_r1_a_request_without_any_lane_attribute_is_counted_as_before(self):
        class _Plain:
            inflight_middle_chunks = 0
        plain = _Plain()
        _account(SimpleNamespace(chunked_req=plain), [_req("S", 0)])
        self.assertEqual(plain.inflight_middle_chunks, 1)

    def test_no_continuation_nothing_to_count(self):
        _account(SimpleNamespace(chunked_req=None), [_req("S", 0)])  # must not raise


# ---------------------------------------------------------------------------
# the metal-5 sequence on three PP stages (form without a row carrier: every stage plans for itself)
# ---------------------------------------------------------------------------

POOL = 3  # pp_loop_size: the result of the batch planned in pass m is processed in pass m + 3


class _Stage:
    """One PP stage: the real stamp functions, the real adder for its continuation, the real accounting statement, the real head-of-step
    park.  Batches launched in pass m return in pass m + POOL (``_process_result`` = the two branches of ``process_batch_result_prefill``
    that decide between "prefill finished" and "middle chunk")."""

    def __init__(self, rank, n_chunks_left):
        self.s = _park_stage(rank)
        self.s.forward_ct = 0
        self.s.mbs = [None] * POOL
        self.s._pp_admission_incoming_effective = None
        self.s.waiting_queue = []
        req = _continuation_on(self.s)
        req.prefix_indices = list(range(CHUNK))  # one chunk done: five middle chunks and the final one are still to run
        req.reset_for_retract = MagicMock(side_effect=lambda: setattr(req, "inflight_middle_chunks", 0))
        self.req = req
        self.first_token = []           # (pass, rid) of a request whose prefill finished
        self.batches = {}               # pass -> tuple of rids
        self.returns = {}               # pass -> list of reqs of the batch that returns then
        self.park_pass = None
        self.released = 0

    def chunk_done(self, req, end):
        req.prefix_indices = list(range(end))  # cache_unfinished_req: the chunk's rows are stashed in the tree

    def head(self, m):
        """get_next_batch_to_run, first lines: the result of the batch from pass m - POOL lands, then the head-of-step park."""
        for r in self.returns.pop(m, []):
            if r.inflight_middle_chunks <= 0:
                self.first_token.append((m, r.rid))          # process_batch_result_prefill: prefill finished
            else:
                r.inflight_middle_chunks -= 1                # a middle chunk
        before = self.s.chunked_req
        sch.Scheduler.process_pending_weg2_park(self.s)
        if before is not None and self.s.chunked_req is None and self.park_pass is None:
            self.park_pass = m

    def plan(self, m, arrivals):
        s = self.s
        adder = _adder()
        lanes_p.configure_adder(s, adder)
        req = s.chunked_req
        if req is not None:
            with lanes_p.chunk_scope(adder, req):
                s.chunked_req = adder.add_chunked_req(req)
        floor = lanes_p.floor_for_pass(s) if lanes_p.skips_by_lane(s) else 0
        for r in list(s.waiting_queue):
            if floor and lanes_p.holds(lanes_p.req_lane(r), floor):
                continue
            s.waiting_queue.remove(r)
            adder.can_run_list.append(r)
            break
        for r in arrivals:
            if floor and lanes_p.holds(lanes_p.req_lane(r), floor):
                s.waiting_queue.append(r)
            else:
                adder.can_run_list.append(r)
        can_run = list(adder.can_run_list)
        if not can_run:
            self.batches[m] = ()
            s.mbs[m % POOL] = None
            return ()
        for r in can_run:
            if r is req and r.extend_range.end > r.extend_range.start:
                self.chunk_done(r, r.extend_range.end)
        _account(s, can_run)                                 # the REAL statement
        s.forward_ct += 1
        s.mbs[m % POOL] = object()
        self.returns[m + POOL] = can_run
        self.batches[m] = tuple(r.rid for r in can_run)
        return self.batches[m]


def _run(*, passes, rpcs, arrivals, chunks_before=0):
    """Lockstep on the pass index (a stage's pass m plans the batch PP0 planned in pass m): PP0's pass top (stamp), the followers absorb in
    the same pass, then head-of-step park, plan, accounting.  ``rpcs`` = {pass: (floor, epoch)}; ``arrivals`` = {pass: [Req factory]}."""
    stages = [_Stage(i, chunks_before) for i in range(3)]
    for m in range(passes):
        stamp = lanes_p.pp0_pass(stages[0].s)
        wire = pr.stamp_lane_floor([], stamp)
        for st in stages[1:]:
            sent_on = list(wire)
            _rest, got = pr.absorb_lane_floor(wire)
            lanes_p.apply_follower(st.s, got)
            wire = sent_on
        for st in stages:
            st.head(m)
        for st in stages:
            st.plan(m, [mk() for mk in arrivals.get(m, [])])
        if m in rpcs:
            lanes_p.on_rpc(stages[0].s, _rpc(*rpcs[m]))
    return stages


def _spur(rid="S1"):
    r = _req(rid, 1, n_tokens=16000)  # a higher lane that fits ONE chunk (the metal-5 weg2-3-9: 16 000 tokens)
    r.inflight_middle_chunks = 0
    return r


def _spur1():
    return _spur("S1")


def _spur1b():
    return _spur("S1b")


class _SchedPatch(unittest.TestCase):
    def setUp(self):
        self.released = []
        self.orig = sch.release_kv_cache
        sch.release_kv_cache = lambda req, tree, is_insert=True: self.released.append((req.rid, is_insert))

    def tearDown(self):
        sch.release_kv_cache = self.orig


class MetalFiveTest(_SchedPatch):
    """probe 2b lage (i): the floor rises while the lane-0 continuation has chunks in flight, the higher lane arrives in the same pass."""

    def test_every_stage_gives_its_rows_back_in_the_same_pass_once_the_launched_chunk_drained(self):
        """RED on 8bfd25d842: the held continuation kept a leaked count, no stage ever parked (genuine `WEG2-PARK (lane)` = 0)."""
        with _LanesOn():
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm:
                stages = _run(passes=14, rpcs={3: (1, 1)}, arrivals={5: [_spur1]})
            parks = [st.park_pass for st in stages]
            self.assertTrue(all(p is not None for p in parks), f"no park on {parks}")
            self.assertEqual(len(set(parks)), 1, f"the stages parked in different passes: {parks}")
            self.assertEqual(self.released, [("R", True)] * 3, "each stage returns the rows INSERTED into the tree")
            lines = [l for l in cm.output if "WEG2-PARK (lane) n=1" in l]
            self.assertEqual(len(lines), 3, cm.output)
            for st in stages:
                self.assertIsNone(st.s.chunked_req)
                self.assertEqual(st.s.queued, [("R", True)], "back at the head of the queue of its lane")
                self.assertEqual(st.req.weg2_parked_span, st.req.prefix_indices.__len__())
                self.assertEqual(st.req.inflight_middle_chunks, 0)

    def test_the_batches_are_identical_on_every_stage_through_the_whole_sequence(self):
        """Ränge nie uneins: the higher lane runs on all stages in the same batch, the lower lane on none after the floor."""
        with _LanesOn():
            stages = _run(passes=14, rpcs={3: (1, 1)}, arrivals={5: [_spur1]})
            rows = [tuple(st.batches[m] for st in stages) for m in range(14)]
            for m, row in enumerate(rows):
                self.assertEqual(len(set(row)), 1, f"pass {m}: the stages planned different batches {row}")
            flat = [b for b in stages[0].batches.values() if b]
            self.assertIn(("S1",), flat, "the higher lane was admitted")
            after = [b for m, b in stages[0].batches.items() if m > 5]
            self.assertTrue(all("R" not in b for b in after), "the held lane gets no chunk after the floor")

    def test_the_rows_go_back_while_the_higher_lane_is_still_in_the_pipeline(self):
        """The reason the park matters (P short of pool, a higher lane that needs its own chunked slot): the lower lane's rows are
        returned within the pipeline depth after the hold, they do not wait for the front to abort the request."""
        with _LanesOn():
            stages = _run(passes=14, rpcs={3: (1, 1)}, arrivals={5: [_spur1]})
            hold_pass = min(m for m, b in stages[0].batches.items() if b == ("S1",))
            self.assertLessEqual(stages[0].park_pass - hold_pass, POOL + 1)

    def test_canfail_the_leaked_count_would_keep_every_stage_from_parking(self):
        """CAN-FAIL twin: the accounting as it was (every non-empty batch counts the chunked request) -> no park."""
        with _LanesOn():
            orig = lanes_p.held_without_seat
            lanes_p.held_without_seat = lambda req, can_run_list: False
            try:
                stages = _run(passes=14, rpcs={3: (1, 1)}, arrivals={5: [_spur1]})
            finally:
                lanes_p.held_without_seat = orig
            self.assertEqual([st.park_pass for st in stages], [None, None, None])
            self.assertEqual(self.released, [])


class ResumeInPlaceTest(_SchedPatch):
    """The floor falls before the rows went back (a SHORT higher lane): the continuation resumes in place and must FINISH its prefill."""

    def _fall(self):
        # the floor rises at 3 (held from the plan of pass 5), the higher lane arrives at 5; its RPC to fall reaches PP0 in pass 5 too and
        # is effective one batch after its stamp -- the second small arrival in pass 6 is that batch -- so the floor is gone at the
        # head of pass 7, BEFORE the launched chunk of pass 4 drained and the rows could go back: the park is void, the continuation
        # resumes in place and runs its final chunk
        return _run(passes=30, rpcs={3: (1, 1), 5: (0, 2)}, arrivals={5: [_spur1], 6: [_spur1b]})

    def test_the_final_chunk_is_a_final_chunk(self):
        """RED on 8bfd25d842: the leaked +1 turns the last chunk's result into a 'middle chunk', the first token never comes."""
        with _LanesOn():
            stages = self._fall()
            for st in stages:
                self.assertIsNone(st.park_pass, "the floor fell first: no rows went back on stage %d" % st.s.ps.pp_rank)
                done = [rid for _m, rid in st.first_token if rid == "R"]
                self.assertEqual(done, ["R"], f"stage {st.s.ps.pp_rank}: the lane-0 prefill never finished {st.first_token}")
                self.assertEqual(st.req.inflight_middle_chunks, 0)
            self.assertEqual(self.released, [])

    def test_the_resume_batches_are_identical_on_every_stage(self):
        with _LanesOn():
            stages = self._fall()
            for m in range(30):
                self.assertEqual(len({st.batches[m] for st in stages}), 1, f"pass {m}")


class ParkWaitMarkerTest(_SchedPatch):
    def test_one_line_names_the_count_that_holds_the_rows_back(self):
        with _LanesOn():
            stages = [_park_stage(0)]
            s = stages[0]
            req = _continuation_on(s)
            req.inflight_middle_chunks = 2
            req.weg2_pool_parked = True
            req.weg2_lane_parked = True
            st = lanes_p.state_of(s, create=True)
            st.offer(1, 1)
            st.promote()
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm:
                for _ in range(5):
                    sch.Scheduler.process_pending_weg2_park(s)
            lines = [l for l in cm.output if lanes_p.MARK_PARK_WAIT in l]
            self.assertEqual(len(lines), 1, "once per request and floor epoch, not once per step")
            self.assertIn("rid=R", lines[0])
            self.assertIn("inflight=2", lines[0])
            self.assertIn("floor=1 epoch=1", lines[0])
            self.assertEqual(self.released, [])
            req.inflight_middle_chunks = 0
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm2:
                sch.Scheduler.process_pending_weg2_park(s)
            self.assertTrue(any("WEG2-PARK (lane) n=1" in l for l in cm2.output))
            self.assertEqual(self.released, [("R", True)])

    def test_a_pressure_park_waiting_for_its_chunk_says_nothing_new(self):
        """R1: the marker belongs to the lane park only."""
        with _LanesOn():
            s = _park_stage(0)
            req = _continuation_on(s)
            req.inflight_middle_chunks = 1
            req.weg2_pool_parked = True
            s.waiting_queue = [object()]
            logger = logging.getLogger("sglang.srt.weg2.lanes_p")
            with self.assertNoLogs(logger, level="INFO"):
                sch.Scheduler.process_pending_weg2_park(s)


class SourcePinTest(unittest.TestCase):
    def test_the_batch_builder_asks_the_lane_predicate_before_it_counts(self):
        src = inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw)
        i = src.index("inflight_middle_chunks += 1")
        blk = src[max(0, i - 400): i + 60]
        self.assertIn("_lanes_p.held_without_seat(self.chunked_req, can_run_list)", blk)

    def test_the_predicate_is_the_lane_flag_and_the_batch_membership(self):
        held = _held_continuation()
        self.assertTrue(lanes_p.held_without_seat(held, []))
        self.assertTrue(lanes_p.held_without_seat(held, [_req("S", 1)]))
        self.assertFalse(lanes_p.held_without_seat(held, [held]), "in the batch: it took a seat")
        self.assertFalse(lanes_p.held_without_seat(_held_continuation(lane_parked=False), []), "not a lane park")
        self.assertFalse(lanes_p.held_without_seat(object(), []))
        flag_true_only = _held_continuation()
        flag_true_only.weg2_lane_parked = "yes"  # the flag is `is True`, as everywhere else in lanes_p
        self.assertFalse(lanes_p.held_without_seat(flag_true_only, []))


# ---------------------------------------------------------------------------
# the minor: the phases of the D lane park (probe 5: LANE-PREEMPT floor=1 902 ms, park body 815 ms, no phase in the log)
# ---------------------------------------------------------------------------

@pytest.fixture
def group_d(monkeypatch):
    """What the autouse fixture of the L2 file does for its tests (group D, the park env at its defaults); a fixture of THIS file, so
    the group-P tests above keep their own environment."""
    from sglang.srt.weg2 import d_seats as ds

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")


def test_the_lane_park_names_its_phases_in_one_line_after_the_park_line(group_d, lanes_on, caplog):
    a = _l2_req(1, 0)
    s = _Sched(running=[a])
    with caplog.at_level("INFO"):
        out = _park_lane(s, [a.rid])
    assert out.parked == [a.rid]
    msgs = [r.getMessage() for r in caplog.records]
    park_lines = [i for i, m in enumerate(msgs) if m.startswith(LN.MARK_D_PARK_PARK)]
    phase_lines = [i for i, m in enumerate(msgs) if m.startswith(LN.MARK_D_PARK_PHASES)]
    assert len(park_lines) == 1 and len(phase_lines) == 1, "the park(lane) line is unchanged, the phase line is one more"
    assert park_lines[0] < phase_lines[0]
    line = msgs[phase_lines[0]]
    assert " n=1 " in line
    for name in ("land", "select", "draft", "end", "retract", "rest"):
        assert f"{name}:" in line, line
    assert LN.MARK_D_PARK_PHASES != LN.MARK_D_PARK_PARK and not LN.MARK_D_PARK_PHASES.startswith(LN.MARK_D_PARK_PARK), (
        "a count of the park(lane) marker must not pick the phase line up")


def test_r1_the_flip_park_is_untouched_the_phase_line_lives_in_the_lane_park_only():
    src = open(rt.__file__).read()
    i, j = src.index("def park_rids("), src.index("def lane_floor(")
    assert src.count(LN.MARK_D_PARK_PHASES.split()[0]) >= 0  # (marker text is a lanes.py constant)
    assert src.count("lanes.MARK_D_PARK_PHASES") == 1 and i < src.index("lanes.MARK_D_PARK_PHASES") < j


if __name__ == "__main__":
    unittest.main()
