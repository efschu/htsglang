# SPDX-License-Identifier: Apache-2.0
"""27B-PP-SKIPOUT-1024 (SCHEDULER-SOLL v2 R11): the chunked slot agrees across the PP stages.

THE SPECIMEN (27B int4ar, P = PP2 on 2x3080, image rc35i4ar3 = fe572f3224,
/spinning/docker-acceptance/27b/evidence/boot_weg2_dkr27bint4arbar1int4ar10240630_fe572f3224_1024_063046.P.log):

    06:36:00 PP0  chunked prefill PARKED: the pool can fund 0 tokens       (pdflip-3-7 at 25157 of 40067, #679 in place)
    06:36:02 PP0  PDFLIP-PARK n=1 rid=pdflip-3-7 span=25157 of 40067        (PP0's slot is EMPTY now)
    06:36:02 PP0  #969N ADMIT slot=1 fwd_ct=26 extend=4096 rids=[pdflip-3-8] (minted as PP0's continuation)
    06:36:04 PP1  #992 CHUNKED CONTINUATION NOT NAMED rid=pdflip-3-7        (PP1's slot still holds it)
    06:36:04 PP1  [#967] SECOND CONTINUATION REFUSED rid=pdflip-3-8 site=_add_scheduled_req
    06:36:05 PP1  #969N ADMIT slot=1 fwd_ct=26 extend=4096 rids=[pdflip-3-8] (#995: runs unannounced)
    06:36:08 PP1  AssertionError: PP skip output comm invariant violated: req pdflip-3-8 has inflight_middle_chunks=0 ...

HARNESS. A fake PP group (one object per stage, its OWN Req objects): PP0 decides with the REAL ``PrefillAdder.add_one_req``,
the followers execute PP0's forwarded decision (extent + carried last-chunk verdict) through the REAL ``_add_scheduled_req``.
The scheduler's pieces are the REAL code: the displacement and the drain (``managers/pp_continuation_seat.py``, called by
scheduler.py exactly as here -- pinned by ``SourcePinTest``), the accounting statement cut out of
``Scheduler._get_new_batch_prefill_raw`` by AST (as test_pdflip_lanes_fix4_1012 does), ``anchor_tails.contains_last_prefill_chunk``,
``_pp_can_skip_output_comm`` and ``SchedulerBatchResultProcessor._validate_pp_skip_output_comm`` -- the assert that killed the boot.
PP0's PDFLIP-PARK is the scheduler's (a #679 in-place park in a pass with nothing else admissible, the rows back at the head of a
later step once the launched chunk drained); results of pass k are processed at pass k + LAP.

DANGER DIRECTION: PP stages uneins -> death (the assert) or hang (one stage skips the output exchange the other waits for). Every
test checks the per-pass skip verdict of all stages for equality, not only the end state.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import anchor_tails as at  # noqa: E402

try:  # the module of the fix; absent on the parent, where the specimen test must fail with the BOOT's assertion, not an ImportError
    from flliper.srt.managers import pp_continuation_seat as seat  # noqa: E402
except ImportError:  # pragma: no cover - parent tree only
    seat = None
from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.managers import scheduler as sch  # noqa: E402
from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from flliper.srt.managers.scheduler_pp_mixin import _pp_can_skip_output_comm  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult  # noqa: E402
from flliper.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from flliper.srt.utils.common import Range  # noqa: E402

CHUNK = 4096
LAP = 2
C_LEN, C_START = 14000, 0  # the continuation PP0 parks (pdflip-3-7)
X_LEN, X_PREFIX = 11000, 2048  # the request PP0 mints next (pdflip-3-8: a cached head, then chunks)
Y_LEN = 1500  # a short whole request (one last chunk)


@pytest.fixture(autouse=True)
def _skip_comm_on(monkeypatch):
    """The boot ran with the pure-chunked output-comm skip armed (the validator is a no-op without it)."""
    monkeypatch.setenv("FLLIPER_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM", "1")
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


# ----------------------------------------------------------------------------------------------------------- stand-ins
def _tree_cache():
    tc = MagicMock()
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.evictable_size.return_value = 0
    tc.disable = False
    tc.supports_mamba.return_value = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator():
    al = MagicMock()
    for n in ("full_available_size", "swa_available_size", "available_size"):
        getattr(al, n).return_value = 10**7
    return al


def _running_batch():
    b = MagicMock()
    b.reqs = []
    b.batch_size.return_value = 0
    return b


def _req(rid, n, prefix=0):
    r = MagicMock(spec=Req)
    r.rid = rid
    r.priority = 0
    r.prefix_indices = list(range(prefix))
    r.full_untruncated_fill_ids = list(range(n))
    r.origin_input_ids = list(range(n))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=1, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = None
    r.mamba_pool_idx = None
    r.req_pool_idx = 7
    r.inflight_middle_chunks = 0
    r.extend_range = None
    r.to_finish = None
    r.finished_reason = None
    r.is_retracted = False
    r.finished.side_effect = lambda: r.finished_reason is not None
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(side_effect=lambda s, e: setattr(r, "extend_range", Range(s, e)))
    r.truncate_prefix_to = MagicMock(side_effect=lambda n: setattr(r, "prefix_indices", r.prefix_indices[:n]))
    return r


def _accounting_statement():
    """The REAL statement of ``Scheduler._get_new_batch_prefill_raw`` that counts the chunked request's middle chunk."""
    src = textwrap.dedent(inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw))
    best = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and any(
            isinstance(s, ast.AugAssign) and ast.unparse(s.target) == "self.chunked_req.inflight_middle_chunks"
            for s in node.body
        ):
            best = node
            break
    assert best is not None, "the accounting statement is gone from _get_new_batch_prefill_raw"
    ns = dict(vars(sch))
    exec(compile("def frag(self, can_run_list):\n" + textwrap.indent(ast.unparse(best), "    ") + "\n", "<acct>", "exec"), ns)
    return ns["frag"]


class Stage:
    """One P stage: the scheduler's prefill coordination around the real adder (attributes as the real Scheduler names them)."""

    def __init__(self, pp_rank, pp_size, specs):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
        self.reqs = OrderedDict((rid, _req(rid, n, p)) for rid, n, p in specs)
        self.waiting = []
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.outputs = {}
        self.batches = {}
        self.hold = set()  # PP0: rids its admission skips this pass (the TWIN-DEFER / store read of the specimen)
        self.park_in_place = set()  # PP0: passes in which the pool cannot fund the continuation's next chunk (#679)
        self.displaces = 0
        self.violations = []  # strict=False: the validator's AssertionErrors, recorded instead of raised
        self.strict = True
        self._acct = _accounting_statement()

    # the PDFLIP-PARK give-back (scheduler: `_pdflip_give_back_continuation`): rows to the tree, request to the head of the queue.
    # The prefix is kept here: the re-admission's prefix match finds the inserted span again.
    def _give_back(self, req, *, clear_slot):
        req.extend_range = None
        req.pdflip_pool_parked = False
        if clear_slot:
            self.chunked_req = None
        self.waiting.insert(0, req)
        return True

    def head_of_step(self):
        # `process_pending_pdflip_park` -> the displaced drain first, then the slot's own pool park
        if seat is not None:
            seat.drain_displaced(self, lambda r: self._give_back(r, clear_slot=False))
        c = self.chunked_req
        if c is not None and getattr(c, "pdflip_pool_parked", False) is True and c.inflight_middle_chunks <= 0:
            self._give_back(c, clear_slot=True)

    def run_pass(self, k, decision=None):
        self.head_of_step()
        c = self.chunked_req
        if c is not None and c.extend_range is not None and c.extend_range.end > len(c.prefix_indices):
            c.prefix_indices = list(range(c.extend_range.end))  # stash_chunked_request
        follower = decision is not None
        adder = sp.PrefillAdder(
            page_size=1,
            tree_cache=_tree_cache(),
            token_to_kv_pool_allocator=_allocator(),
            running_batch=_running_batch(),
            new_token_ratio=1.0,
            rem_input_tokens=10**7,
            rem_chunk_tokens=CHUNK,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
            scheduled_extents=({rid: (p, e) for rid, (p, e, _l) in decision.items()} if follower else None),
            scheduled_last_chunk=({rid: last for rid, (_p, _e, last) in decision.items()} if follower else None),
        )
        incoming = {rid: p for rid, (p, _e, _l) in decision.items()} if follower else None
        if c is not None:
            if not follower and k in self.park_in_place:
                c.pdflip_pool_parked = True  # #679: no chunk this pass, kept in place; nothing else is admissible either
            elif not sch._pp_chunked_continuation_not_named(self, incoming, c):
                self.chunked_req = adder.add_chunked_req(c)
        adder.chunked_req_outstanding = self.chunked_req is not None
        if follower:
            by_rid = {r.rid: r for r in self.waiting}
            for rid in decision:
                if any(x.rid == rid for x in adder.can_run_list):
                    continue  # the carried continuation, already admitted by add_chunked_req (also on its final chunk)
                assert rid in by_rid, f"PP{self.ps.pp_rank} cannot reach named rid {rid} (#791 membership)"
                adder.add_one_req(by_rid[rid], truncation_align_size=None)
        elif k not in self.park_in_place:
            for r in list(self.waiting):
                if r.rid in self.hold:
                    continue
                if adder.add_one_req(r, truncation_align_size=None).name != "CONTINUE":
                    break
        can = list(adder.can_run_list)
        if follower:
            assert sorted(x.rid for x in can) == sorted(decision), (self.ps.pp_rank, [x.rid for x in can], list(decision))
            can.sort(key=lambda x: list(decision).index(x.rid))
        # scheduler.py, right before `assert self.chunked_req is None`
        if adder.new_chunked_req is not None:
            if getattr(adder, "displaced_resident_continuation", False):
                seat.displace_resident(self, adder.new_chunked_req, can_run_list=can)
                self.displaces += 1
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req
        if not can:
            return None, OrderedDict()
        self._acct(self, can)
        if seat is not None:  # review S4, as the scheduler calls it right after the tail adoption
            seat.assert_no_unannounced_middle_chunk(self, can, self.chunked_req, (), adder._told_last_chunk)
        self.waiting = [r for r in self.waiting if not any(r is x for x in can)]
        batch = SimpleNamespace(
            reqs=can,
            forward_mode=ForwardMode.EXTEND,
            return_logprob=False,
            contains_last_prefill_chunk=at.contains_last_prefill_chunk(can, self.chunked_req, ()),
            chunked_req=self.chunked_req,
        )
        batch.skip = bool(_pp_can_skip_output_comm(batch))
        self.batches[k] = batch
        out = OrderedDict()
        for x in can:
            s, e = x.extend_range.start, x.extend_range.end
            out[x.rid] = (s, e - s, e >= len(x.full_untruncated_fill_ids))
        return batch, out

    def process(self, batch):
        """``process_batch_result_prefill``: the validator that killed the boot, then the inflight branch per request."""
        try:
            SchedulerBatchResultProcessor._validate_pp_skip_output_comm(batch, SimpleNamespace(skipped_output_comm=batch.skip))
        except AssertionError as e:
            if self.strict:
                raise
            self.violations.append(str(e))
            return
        for r in batch.reqs:
            if r.finished() and r.inflight_middle_chunks <= 0:
                continue
            if r.inflight_middle_chunks <= 0:
                self.outputs[r.rid] = self.outputs.get(r.rid, 0) + 1
                r.finished_reason = "LENGTH"
            else:
                r.inflight_middle_chunks -= 1


def run_group(*, pp=2, passes=16, park_at=2, x_at=3, c_at=8, second="X", strict=True, drain_end=True, lap=LAP):
    """C runs from pass 0; PP0 parks it in place at ``park_at`` (X and C held), gives it back at the next head of step once the
    launched chunk drained, admits ``second`` (X: a chunked continuation, Y: a short whole request) from ``x_at`` and lets C
    back in from ``c_at``. Returns the stages and the per-pass skip verdicts."""
    specs = [("C", C_LEN, C_START), ("X", X_LEN, X_PREFIX), ("Y", Y_LEN, 0)]
    stages = [Stage(i, pp, specs) for i in range(pp)]
    for s in stages:
        s.waiting = [s.reqs["C"]]
        s.strict = strict
    pp0 = stages[0]
    pp0.park_in_place = {park_at}
    pending, verdicts = [], []
    for k in range(passes):
        while pending and pending[0][0] <= k:
            _due, s, b = pending.pop(0)
            s.process(b)
        if k == 1:  # the second request arrives on every stage (it is in every waiting queue from here on)
            for s in stages:
                s.waiting.append(s.reqs[second])
        pp0.hold = set()
        if k < x_at:
            pp0.hold.add(second)
        if park_at <= k < c_at:
            pp0.hold.add("C")
        b0, decision = pp0.run_pass(k)
        row = [None if b0 is None else b0.skip]
        if b0 is not None:
            pending.append((k + lap, pp0, b0))
        for s in stages[1:]:
            b, _ = s.run_pass(k, decision)
            row.append(None if b is None else b.skip)
            if b is not None:
                pending.append((k + lap, s, b))
        verdicts.append(row)
    while pending and drain_end:
        _due, s, b = pending.pop(0)
        s.process(b)
    return stages, verdicts


def _assert_uniform(stages, verdicts):
    for k, row in enumerate(verdicts):
        assert len(set(row)) == 1, f"pass {k}: the stages disagree on the output exchange {row} (one skips, one waits: hang)"
    for s in stages:
        assert s.chunked_req is None, (s.ps.pp_rank, s.chunked_req and s.chunked_req.rid)
        assert not seat.displaced(s), (s.ps.pp_rank, [r.rid for r in seat.displaced(s)])
        for r in s.reqs.values():
            assert r.inflight_middle_chunks == 0, (s.ps.pp_rank, r.rid, r.inflight_middle_chunks)


# ---------------------------------------------------------------------------------------------------- the specimen
class TestSpecimen:
    @pytest.mark.parametrize("lap", [1, 2])
    def test_pp2_the_follower_follows_pp0s_park_and_nothing_trips(self, lap):
        """RED on 5b20670b47 / fe572f3224. lap=1 is the boot's order (PP1 processed fwd 26's result before it built the next
        pass) and fails with the boot's exact AssertionError on PP1 (rid X = pdflip-3-8); lap=2 (the result one pass later)
        fails one pass earlier at the membership: X ran unannounced, it is in no queue and no slot when PP0 names its next chunk."""
        stages, verdicts = run_group(pp=2, lap=lap)
        _assert_uniform(stages, verdicts)
        for s in stages:
            assert s.outputs == {"X": 1, "C": 1}, (s.ps.pp_rank, s.outputs)
        assert stages[0].displaces == 0, "PP0 decides: it never displaces"
        assert stages[1].displaces == 1, "PP1 displaced the stale continuation exactly once"

    def test_pp3_every_follower_follows(self):
        stages, verdicts = run_group(pp=3)
        _assert_uniform(stages, verdicts)
        assert [s.displaces for s in stages] == [0, 1, 1]
        for s in stages:
            assert s.outputs == {"X": 1, "C": 1}, (s.ps.pp_rank, s.outputs)

    def test_the_minted_request_is_the_followers_continuation_in_its_first_pass(self):
        stages, _ = run_group(pp=2, passes=4, drain_end=False)  # stop right after X's first chunk, its result still owed
        pp0, pp1 = stages
        assert pp0.chunked_req is pp0.reqs["X"]
        assert pp1.chunked_req is pp1.reqs["X"], "the named continuation holds the slot on the follower as on PP0"
        b0, b1 = pp0.batches[3], pp1.batches[3]
        assert [r.rid for r in b0.reqs] == [r.rid for r in b1.reqs] == ["X"]
        assert b0.contains_last_prefill_chunk is b1.contains_last_prefill_chunk is False
        assert pp1.reqs["X"].inflight_middle_chunks == 1, "its middle chunk is counted (the boot's 0)"
        assert pp1.reqs["C"].inflight_middle_chunks == 0, "the displaced continuation collected no count"
        assert [r.rid for r in seat.displaced(pp1)] in (["C"], []), "displaced (its rows go back once drained)"

    def test_the_displaced_continuation_is_back_in_the_queue_before_pp0_names_it(self):
        stages, _ = run_group(pp=2, passes=8)  # up to (not into) C's re-admission
        pp1 = stages[1]
        assert not seat.displaced(pp1)
        assert pp1.reqs["C"] in pp1.waiting
        assert pp1.reqs["C"].extend_range is None


# --------------------------------------------------------------------------- the hang direction without a displacement
class TestShortRequestBesideAStaleContinuation:
    """PP0 parks C and admits a SHORT whole request Y: no continuation is minted, the follower keeps C (#992) -- it must agree on the
    output exchange anyway (the seat, not the slot) and must not leak a count onto C (its final chunk would read as a middle one)."""

    def test_pp2_uniform_and_both_finish_once_everywhere(self):
        stages, verdicts = run_group(pp=2, second="Y")
        _assert_uniform(stages, verdicts)
        for s in stages:
            assert s.outputs == {"Y": 1, "C": 1}, (s.ps.pp_rank, s.outputs)
        assert [s.displaces for s in stages] == [0, 0]

    def test_canfail_the_slot_question_makes_the_follower_skip_what_pp0_waits_for(self, monkeypatch):
        stock = lambda can, chunked, tails: chunked is None or len(can) != 1  # noqa: E731 (the pre-fix expression)
        monkeypatch.setattr(at, "contains_last_prefill_chunk", stock)
        stages, verdicts = run_group(pp=2, second="Y", strict=False)
        mixed = [(k, row) for k, row in enumerate(verdicts) if len(set(row)) > 1]
        assert mixed == [(3, [False, True])], verdicts  # PP0 receives Y's output, PP1 skips sending it: the hang
        assert stages[0].violations == []
        assert len(stages[1].violations) == 1 and "req Y has inflight_middle_chunks=0" in stages[1].violations[0]

    def test_canfail_the_leaked_count_starves_the_followers_final_chunk(self, monkeypatch):
        monkeypatch.setattr(seat, "counts_middle_chunk", lambda req, can: req is not None)  # the pre-fix accounting
        stages, _verdicts = run_group(pp=2, second="Y")
        assert stages[0].outputs.get("C") == 1
        assert stages[1].outputs.get("C") is None, "PP1 read C's final chunk as a middle chunk: no first token there"


# ------------------------------------------------------------------------------------------------ the scheduler halves
def _ns_stage(chunked):
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=2), chunked_req=chunked, _pending_chunked_abort_req=None)
    return s


class TestDisplaceAndDrain:
    def test_displace_empties_the_slot_and_keeps_the_resident(self, caplog):
        c, x = _req("C", 10), _req("X", 10)
        s = _ns_stage(c)
        with caplog.at_level("INFO", logger=seat.__name__):
            assert seat.displace_resident(s, x, can_run_list=[x]) is c
        assert s.chunked_req is None and seat.displaced(s) == [c]
        assert any("PP-SLOT DISPLACE n=1 pp_rank=1 resident=C" in m for m in caplog.messages)

    def test_displace_without_a_resident_is_loud(self):
        x = _req("X", 10)
        for chunked in (None, x):
            with pytest.raises(RuntimeError, match="DISPLACE WITHOUT A RESIDENT"):
                seat.displace_resident(_ns_stage(chunked), x, can_run_list=[x])

    def test_drain_waits_for_the_launched_chunk(self):
        c = _req("C", 10)
        c.inflight_middle_chunks = 1
        s = _ns_stage(None)
        seat.displaced(s).append(c)
        back = []
        assert seat.drain_displaced(s, lambda r: back.append(r)) == 0
        assert back == [] and seat.displaced(s) == [c]
        c.inflight_middle_chunks = 0
        assert seat.drain_displaced(s, lambda r: back.append(r)) == 1
        assert back == [c] and seat.displaced(s) == []

    def test_drain_leaves_an_abort_to_the_abort_path_and_drops_a_finished_one(self):
        a, f, g = _req("A", 10), _req("F", 10), _req("G", 10)
        f.finished_reason = "ABORT"
        s = _ns_stage(None)
        s._pending_chunked_abort_req = a
        seat.displaced(s).extend([a, f, g])
        back = []
        seat.drain_displaced(s, lambda r: back.append(r))
        assert back == [g] and seat.displaced(s) == [a]

    def test_a_failed_give_back_keeps_it(self):
        c = _req("C", 10)
        s = _ns_stage(None)
        seat.displaced(s).append(c)
        assert seat.drain_displaced(s, lambda r: False) == 0
        assert seat.displaced(s) == [c]

    def test_the_real_scheduler_give_back_inserts_and_requeues_without_touching_the_slot(self, monkeypatch):
        released = []
        monkeypatch.setattr(sch, "release_kv_cache", lambda req, tree, is_insert=True: released.append((req.rid, is_insert)))
        c, x = _req("C", 10, prefix=6), _req("X", 10)
        c.reset_for_retract = MagicMock()
        s = _ns_stage(x)
        s.tree_cache = object()
        s.queued = []
        s._add_request_to_queue = lambda req, is_retracted=False: s.queued.append((req.rid, is_retracted))
        seat.displaced(s).append(c)
        assert sch.Scheduler._pp_drain_displaced_continuations(s) == 1
        assert released == [("C", True)] and s.queued == [("C", True)]
        assert s.chunked_req is x, "the slot belongs to the named continuation"
        assert c.pdflip_parked_span == 6
        c.reset_for_retract.assert_called_once()

    def test_the_head_of_step_park_runs_the_drain_first(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sch.Scheduler, "_pp_drain_displaced_continuations", lambda self: calls.append("drain") or 0)
        sch.Scheduler.process_pending_pdflip_park(_ns_stage(None))
        assert calls == ["drain"]


class TestDisplacedAbort:
    """A displaced continuation's chunked abort (#791C) is applied by `process_pending_chunked_abort` although it is no longer the
    slot's -- and the slot's new continuation is left alone."""

    def test_applied_and_the_new_continuation_untouched(self, monkeypatch):
        from flliper.srt.pdflip import p_row_authority

        sent = []
        monkeypatch.setattr(sch, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
        monkeypatch.setattr(sch, "release_kv_cache", lambda *a, **k: None)
        monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
        c = SimpleNamespace(
            rid="C", req_pool_idx=None, kv_committed_freed=True, to_finish=1, finished=lambda: False,
            time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None)),
        )
        x = SimpleNamespace(rid="X")
        fake = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=1, pp_size=2), _pending_chunked_abort_req=c, chunked_req=x,
            _pending_chunked_abort_delay=0, _pp_scheduled_extents=lambda: {"X": (0, 4096)},
            disaggregation_mode=None, enable_hicache_storage=False,
            tree_cache=SimpleNamespace(supports_mamba=lambda: False),
            ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=lambda obj, r: sent.append(obj.rid))),
        )
        seat.displaced(fake).append(c)
        sch.Scheduler.process_pending_chunked_abort(fake)
        assert sent == ["C"] and c.aborted_why == "Aborted"
        assert fake.chunked_req is x
        assert fake._pending_chunked_abort_req is None
        assert seat.displaced(fake) == []


# ------------------------------------------------------------------------------------------------------- source pins
class TestSourcePin:
    def test_the_scheduler_displaces_right_before_its_assert(self):
        src = textwrap.dedent(inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw))
        i = src.index("_pp_seat.displace_resident(self, adder.new_chunked_req, can_run_list=can_run_list)")
        j = src.index("assert self.chunked_req is None", i)
        assert "displaced_resident_continuation" in src[i - 200: i]
        assert j - i < 200

    def test_the_adder_mints_instead_of_announcing_nothing(self):
        src = inspect.getsource(sp.PrefillAdder._add_scheduled_req)
        assert 'note_second_continuation_refused(req, "_add_scheduled_req")' not in src
        assert 'self._mint_chunked(req, "_add_scheduled_req/displace")' in src


# ======================================================================================== review conditions (af40b9561f)
from flliper.srt.managers.io_struct import AbortReq  # noqa: E402


class _AbortStage:
    """A follower stage as ``_abort_request_now`` / ``process_pending_pdflip_park`` read it (the fix5 stand-in shape)."""

    def __init__(self, pp_rank=1, pp_size=2):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1)
        self.enable_hicache_storage = False
        self.waiting_queue = []
        self.chunked_req = None
        self.anchor_tails = ()
        self.ipc_channels = MagicMock()
        self.disaggregation_mode = None
        self.grammar_manager = MagicMock()
        self.running_mbs = []
        self.mbs = []
        self.kv_session_offload = None
        self._pdflip_intake_watch = None
        self._pending_chunked_abort_req = None
        self.tree_cache = SimpleNamespace(supports_mamba=lambda: False, release_aborted_request=lambda rid: None)
        self.queued = []

    def _pdflip_abort_dormant_hold(self, recv_req):
        pass

    def _pdflip_d_park_abort(self, recv_req):
        pass

    def _add_request_to_queue(self, req, is_retracted=False):
        self.queued.append((req.rid, is_retracted))
        self.waiting_queue.append(req)


def _live(rid, n=40, prefix=25):
    r = _req(rid, n, prefix=prefix)
    r.kv_committed_freed = False
    r.reset_for_retract = MagicMock()
    r.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None), wait_queue_entry_time=0)
    return r


@pytest.fixture
def _released(monkeypatch):
    log = []
    monkeypatch.setattr(sch, "release_kv_cache", lambda req, tree, is_insert=True: log.append((req.rid, is_insert)))
    monkeypatch.setattr(sch, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    return log


def _displaced_stage():
    st = _AbortStage()
    c, x = _live("pdflip-3-7"), _live("pdflip-3-8")
    st.chunked_req = c
    seat.displace_resident(st, x, can_run_list=[x])
    st.chunked_req = x  # the scheduler adopts the minted continuation
    return st, c, x


def _sent_aborts(st):
    return [call.args[0].rid for call in st.ipc_channels.send_to_tokenizer.send_output.call_args_list
            if isinstance(call.args[0], AbortReq)]


class TestA1AbortAfterDisplacement:
    """MUSS A1: an AbortReq that reaches a follower AFTER the displacement goes through the REAL ``_abort_request_now``; the
    continuation is torn down at the next head of step -- not queued as a zombie PP0 never names."""

    def test_the_real_abort_route_reaches_the_displaced_continuation(self, _released):
        st, c, x = _displaced_stage()
        sch.Scheduler._abort_request_now(st, AbortReq(rid="pdflip-3-7"))
        assert st.waiting_queue == [] and st.chunked_req is x, "the abort touched nothing else"
        sch.Scheduler.process_pending_pdflip_park(st)  # the head of the next step: the displaced drain
        assert st.queued == [], "no zombie in the follower's waiting queue"
        assert seat.displaced(st) == []
        assert _released == [("pdflip-3-7", False)], "rows released WITHOUT a tree insert (an abort, not a park)"
        assert _sent_aborts(st) == ["pdflip-3-7"] and c.aborted_why == "Aborted"
        assert st.chunked_req is x

    def test_abort_all_reaches_it_too(self, _released):
        st, c, _x = _displaced_stage()
        sch.Scheduler._abort_request_now(st, AbortReq(abort_all=True))
        sch.Scheduler.process_pending_pdflip_park(st)
        assert st.queued == [] and seat.displaced(st) == []
        assert "pdflip-3-7" in _sent_aborts(st)

    def test_the_abort_waits_for_a_launched_chunk(self, _released):
        st, c, _x = _displaced_stage()
        c.inflight_middle_chunks = 1
        sch.Scheduler._abort_request_now(st, AbortReq(rid="pdflip-3-7"))
        sch.Scheduler.process_pending_pdflip_park(st)
        assert seat.displaced(st) == [c] and _released == [] and st.queued == []
        c.inflight_middle_chunks = 0
        sch.Scheduler.process_pending_pdflip_park(st)
        assert seat.displaced(st) == [] and _released == [("pdflip-3-7", False)] and st.queued == []

    def test_an_abort_for_another_rid_leaves_it_to_the_give_back(self, _released):
        st, c, _x = _displaced_stage()
        sch.Scheduler._abort_request_now(st, AbortReq(rid="pdflip-9-9"))
        sch.Scheduler.process_pending_pdflip_park(st)
        assert st.queued == [("pdflip-3-7", True)] and _released == [("pdflip-3-7", True)]


class TestA2SeatedResident:
    """MUSS A2: the resident runs a chunk in this very batch (#996 fallback / H42 tail corners): stop by name, never displace."""

    def test_a_seated_resident_is_a_named_stop(self):
        c, x = _req("C", 10), _req("X", 10)
        s = _ns_stage(c)
        with pytest.raises(RuntimeError, match="DISPLACE OF A SEATED RESIDENT: rid=C .* mints rid=X"):
            seat.displace_resident(s, x, can_run_list=[c, x])
        assert s.chunked_req is c and seat.displaced(s) == [], "nothing moved"

    def test_the_scheduler_passes_its_batch(self):
        src = inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw)
        assert "displace_resident(self, adder.new_chunked_req, can_run_list=can_run_list)" in src


class TestA3RestoredToTheSlot:
    """MUSS A3: an own-void restore (``chunked_req = chunked_before``) puts the displaced continuation back into the slot while it
    is still listed -- the drain must leave it alone (no give-back, no reset: the stash reads its extend_range) and unlist it."""

    def test_restore_then_head_of_step_leaves_it_in_the_slot(self, _released, caplog):
        st, c, _x = _displaced_stage()
        c.extend_range = Range(22015, 25157)
        st.chunked_req = c  # the void restore of this pass
        with caplog.at_level("WARNING", logger=seat.__name__):
            sch.Scheduler.process_pending_pdflip_park(st)
        assert st.chunked_req is c and seat.displaced(st) == []
        assert _released == [] and st.queued == []
        c.reset_for_retract.assert_not_called()
        assert c.extend_range == Range(22015, 25157), "the head-of-step stash still reads it (instr19)"
        assert any("PP-SLOT DISPLACE UNDONE rid=pdflip-3-7" in m for m in caplog.messages)

    def test_an_abort_recorded_while_displaced_moves_to_the_slot(self, _released):
        st, c, _x = _displaced_stage()
        sch.Scheduler._abort_request_now(st, AbortReq(rid="pdflip-3-7"))
        st.chunked_req = c
        sch.Scheduler._pp_drain_displaced_continuations(st)
        assert st._pending_chunked_abort_req is c and st._pending_chunked_abort_delay == 0
        assert seat.displaced(st) == [] and _released == []


class TestS2ResidencyAuthorities:
    def test_live_reqs_and_the_retirement_see_the_list(self):
        from flliper.srt.managers import phase_flip_runtime as pfr

        st, c, x = _displaced_stage()
        live = pfr._live_reqs(st)
        assert any(r is c for r in live) and any(r is x for r in live)
        assert pfr.consume_retracted_from_live_universe(st, [c]) >= 1
        assert seat.displaced(st) == []

    def test_the_device_is_not_idle_while_a_displaced_continuation_holds_rows(self):
        st, _c, _x = _displaced_stage()
        st.chunked_req = None
        assert sch._pdflip_device_idle(st) is False
        seat.displaced(st).clear()
        assert sch._pdflip_device_idle(st) is True


class TestS3NoSilentLoss:
    def test_a_live_request_without_rows_is_queued_loudly(self, _released, caplog):
        st, c, _x = _displaced_stage()
        c.req_pool_idx = None
        with caplog.at_level("WARNING", logger=seat.__name__):
            sch.Scheduler._pp_drain_displaced_continuations(st)
        assert st.queued == [("pdflip-3-7", True)] and _released == [] and seat.displaced(st) == []
        assert any("PP-SLOT DISPLACED WITHOUT ROWS rid=pdflip-3-7" in m for m in caplog.messages)


class TestS4UnannouncedMiddleChunk:
    def _sched(self, pp_size=2):
        return SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=pp_size))

    def test_an_untracked_not_last_chunk_is_a_named_stop(self):
        x = _req("X", 100)
        x.extend_range = Range(0, 40)
        with pytest.raises(RuntimeError, match="UNANNOUNCED MIDDLE CHUNK rid=X"):
            seat.assert_no_unannounced_middle_chunk(self._sched(), [x], None, (), lambda r: None)
        with pytest.raises(RuntimeError, match="UNANNOUNCED MIDDLE CHUNK rid=X"):
            seat.assert_no_unannounced_middle_chunk(self._sched(), [x], None, (), lambda r: False)

    def test_tracked_last_or_non_pp_is_quiet(self):
        x, t, y = _req("X", 100), _req("T", 100), _req("Y", 100)
        x.extend_range, t.extend_range, y.extend_range = Range(0, 40), Range(0, 99), Range(60, 100)
        ok = seat.assert_no_unannounced_middle_chunk
        ok(self._sched(), [x], x, (), lambda r: None)  # the chunked request
        ok(self._sched(), [t], None, (t,), lambda r: None)  # an anchor tail
        ok(self._sched(), [y], None, (), lambda r: None)  # a last chunk by the local fill
        ok(self._sched(), [x], None, (), lambda r: True)  # #996: the carried verdict says last (follower fill longer)
        ok(self._sched(pp_size=1), [x], None, (), lambda r: None)  # no PP: no skip, no hang direction

    def test_the_batch_builder_calls_it(self):
        src = inspect.getsource(sch.Scheduler._get_new_batch_prefill_raw)
        i = src.index("_pp_seat.assert_no_unannounced_middle_chunk(")
        assert src.index("_anchor_tails.log_pass(") < i < src.index("_wk_t2 = time.perf_counter()")
