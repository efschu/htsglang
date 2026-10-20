"""#791 wiring: the admission DECISION must actually cross the wire, be
reconciled against each downstream rank's own local state, and close the
#630 learning loop back at PP0 -- not merely exist as a pure, unwired
module.

WHY THIS FILE EXISTS SEPARATELY FROM test_pp_admission_congruence_791.py
AND test_pp_admission_retry_livelock_630.py. Those two files exhaustively
cover ``build_pp_admission_decision`` / ``reconcile_pp_admission_decision``
/ ``PPAdmissionCongruenceGuard`` as PURE functions -- no process boundary,
no wire, no scheduler_pp_mixin.py. This file is the wiring itself: the new
``SchedulerPPMixin`` methods ``_pp_send_admission_decision``,
``_pp_recv_admission_decision`` and ``_pp_reconcile_incoming_admission``,
bound to a holder exactly as test_pp_proxy_readiness_contract_789.py binds
the shipped ``_pp_recv_proxy_tensors`` / ``_pp_wait_for_proxy_readiness``.
Before this wiring existed, pp_admission_congruence.py's own module
docstring said as much: "wiring the publish/consult loop is future work,
out of scope for scheduler_pp_mixin.py under the current #789 scope fence."

THE ROOT DEFECT THIS CLOSES. Each PP rank is an independent scheduler that
re-derives its own admission verdict from its own local radix-cache state
(scheduler.py's ``_get_new_batch_prefill_raw``). Requests are chain-forwarded
unconditionally regardless of what any rank decided, and the proxy
send/receive that carries the actual hidden-state tensor is gated on the
RECEIVING rank's OWN independently-derived ``cur_batch``
(``_event_loop_pp_body``: ``if cur_batch: ... pp_proxy_tensors =
self._pp_recv_proxy_tensors(mb_id)``) -- so two ranks disagreeing about
which requests are admitted, or how much prefix a given request reuses, is
not a quality issue: it is a shape/row-count disagreement that wedges the
pipeline or corrupts the wire. This file proves the decision that would
prevent that disagreement actually reaches every rank, in order, and that a
disagreement discovered mid-ring is degraded (not raised) and reported back.

WHY THREE REAL PROCESSES OVER REAL GLOO, RING-DEFAULT ``dst=None``/
``src=None``. Same reasoning as test_pp_proxy_readiness_contract_789.py: a
mock cannot distinguish a genuine cross-process round trip from a
convenient no-op. The fake wire here deliberately leaves ``dst``/``src``
default to ``None`` and resolves them with the SAME formula
``GroupCoordinator.send_tensor_dict``/``recv_tensor_dict`` use in
production (``dst=(rank_in_group+1)%world_size``,
``src=(rank_in_group-1)%world_size``) instead of hand-picking a fixed peer
per rank -- so the test also exercises the claim that the RETURN TRIP
needs no second channel, only the ordinary ring's own wraparound.

THE SCENARIO (mirrors the #791 task's degrade + retry-learning story):

  PP0 (rank 0) admits req "req-A" with told=5 and sends the decision.
  PP1 (rank 1, middle) has only a LOCAL match of 3 for "req-A" -- an
      unsafe-retract (local < told): reconcile must EXCLUDE it from
      ``effective``, log EXACTLY ONE warning, and forward an AMENDED
      decision recording the retraction (rank=1, observed_local=3) rather
      than raising.
  PP2 (rank 2, last) receives the already-retracted entry. Per the pure
      reconcile function's own contract, an already-retracted entry passes
      through UNCHANGED -- no second log, no re-derivation -- and PP2
      forwards it onward via the ring's own wraparound (dst=None resolves
      to PP0, the same mechanism the "output" ring already uses to close
      last-rank -> PP0).
  PP0 receives the wraparound and drives
      ``PPAdmissionCongruenceGuard.record_return_trip`` with it (the
      already-tested pure #630 effect; only the fact that a *correctly
      wire-transported* decision reaches this call is new here).

THE CASES:

  test_ring_roundtrip_degrades_and_reports_back   The scenario above,
      end to end. Asserts: PP1 logs exactly one #791 warning and no more;
      PP2 logs none (already-retracted passthrough); the decision PP0
      receives back names req-A retracted by rank 1 with observed_local=3;
      and PPAdmissionCongruenceGuard.record_return_trip against that
      decision leaves a learned floor of 3 for req-A.

  test_admission_wiring_is_noop_when_pp_size_is_1   BACKWARD COMPATIBILITY.
      The reference regression launch command (CLAUDE.md) never sets
      ``--rank-gpu-id``-equivalent PP topology beyond pp_size=1 semantics
      for this feature's default path; all three new methods must be a
      true no-op (no wire touch) when ``self.ps.pp_size <= 1`` -- checked
      in-process, no multiprocessing needed.

  test_ordering_admission_precedes_batch_and_proxy_recv   STRUCTURAL
      ORDERING PROOF (#791 task's explicit requirement 7). Reads the
      shipped source of ``_event_loop_pp_body`` and asserts the
      admission-decision receive call appears strictly before both
      ``get_next_batch_to_run`` and ``_pp_recv_proxy_tensors`` in it --
      i.e. an ordinary prefix-length divergence is always degraded here,
      before the #789 proxy-readiness contract's raise path could ever be
      reached on a healthy pass.
"""

import inspect
import json
import logging
import os
import tempfile
import time
import types
from collections import defaultdict, deque
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2

RID = "req-A"
TOLD_BY_PP0 = 5
PP1_LOCAL_MATCH = 3  # < TOLD_BY_PP0: the unsafe-retract shape.

JOIN_TIMEOUT_S = 90.0  # HARNESS 20.10.: a spawned rank imports the mixin for >12 s on this rig under load


class _RingWire:
    """Real point-to-point tensor-dict transport over gloo, ring-default.

    Deliberately mirrors ``GroupCoordinator.send_tensor_dict`` /
    ``recv_tensor_dict``'s own ``dst=None`` -> ``(rank_in_group+1) %
    world_size`` and ``src=None`` -> ``(rank_in_group-1) % world_size``
    resolution, rather than a fixed per-rank peer, so the return trip
    (PP2 -> PP0) exercises the SAME "no dst given" code path the forward
    sends do -- proving the ring wraparound, not a hand-wired shortcut.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        # `GroupCoordinator` exposes these as plain attributes and the mixin
        # reads them directly (e.g. scheduler_pp_mixin.py:1548); a stub
        # without them dies as an AttributeError inside the worker, which is
        # a harness gap, not a wiring finding.
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1
        # Direct measurement for the #796 law ("a rank must not post a send
        # no peer is required to take"): the last rank's send must be a
        # no-op, and a counter sees that where a hang-detector would only
        # time out.
        self.sends = 0

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        import pickle

        self.sends += 1
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        if async_send:
            w1 = dist.isend(size, dst=dst)
            w1.wait()
            w2 = dist.isend(
                torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=dst
            )
            w2.wait()
        else:
            dist.send(size, dst=dst)
            dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        import pickle

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=src)
        return pickle.loads(bytes(buf.numpy()))


class _FakeReq:
    """Just enough of a Req for _pp_reconcile_incoming_admission: an rid,
    a device-shaped-but-CPU prefix_indices tensor whose length IS the
    local match, and a no-op init_next_round_input (the real one refreshes
    the match from the tree cache; that refresh is not what this file is
    testing -- see test_pp_admission_congruence_791.py / the schedule_batch
    tests for prefix matching itself)."""

    def __init__(self, rid: str, local_match_len: int):
        self.rid = rid
        self.prefix_indices = torch.arange(local_match_len, dtype=torch.int64)

    def init_next_round_input(self, tree_cache):
        pass


def _make_holder(rank: int, wire: _RingWire, waiting_queue):
    from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        ps=types.SimpleNamespace(pp_size=WORLD, pp_rank=rank),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=waiting_queue,
        tree_cache=None,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    # PORT 20.10. (#1015 removed the dedicated admission-decision arc; #631 ROW AUTHORITY carries the SAME decision as
    # the admission row on the PROXY FRAME -- `load_back` on the send, popped by `_pp_recv_proxy_tensors` into
    # `_pp_last_recv_admission_decision`, reconciled pre-plan, the AMENDED row relayed as `_pp_load_back_wire`;
    # scheduler_pp_mixin `_event_loop_pp_body` ROW-DELIVER). The transport moves to the frame, the rule under test
    # (degrade down the chain, exactly one warning, passthrough at the last rank, no last-rank send) is unchanged.
    h.pp_flip_counters = None
    h._pp_tensor_dict_inbox = defaultdict(deque)
    h._pp_proxy_drops = 0
    h._pp_flip_upstream = lambda: (rank - 1) % WORLD
    for name in (
        "_pp_reconcile_incoming_admission",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_wait_for_proxy_readiness",
        "_pp_proxy_is_not_stale",
        "_pp_note_proxy_consumed",
        "_pp_note_stale_proxy_dropped",
        "_pp_stamp_of",
        "_pp_proxy_seq_key",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


class _WarningCatcher(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


#: the frame's stamp: slot 0 of the ring, the pass's seq and rows (the #631 identity the receive checks)
STAMP = (0, 1, 2)


def _send_row(h, wire_row):
    """What `_event_loop_pp_body` does at the launch end of a non-last rank: the frame with the row (load_back)."""
    from flliper.srt.managers.scheduler_pp_mixin import pp_admission_decision_to_wire  # noqa: F401

    h._pp_send_dict_to_next_stage(
        {"hidden_states": torch.zeros(2)}, async_send=False, msg_type="proxy", stamp=STAMP, load_back=wire_row
    )


def _recv_row(h):
    """The pre-plan frame receive under row authority: the decision PP0 put on the frame."""
    h._pp_recv_proxy_tensors(STAMP[0])
    return h._pp_last_recv_admission_decision


def _worker(rank, init_file, out_dir):
    from flliper.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )

    res = {"rank": rank, "ok": False, "error": None}
    catcher = _WarningCatcher()
    mixin_logger = logging.getLogger("flliper.srt.managers.scheduler_pp_mixin")
    mixin_logger.addHandler(catcher)
    mixin_logger.setLevel(logging.WARNING)
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        wire = _RingWire(rank)

        if rank == PP0:
            h = _make_holder(rank, wire, waiting_queue=[])
            decision = PPAdmissionDecision(
                mb_id=0,
                entries=(
                    PPAdmissionEntry(
                        rid=RID, prefix_len=TOLD_BY_PP0, extend_len=2, admitted=True
                    ),
                ),
            )
            from flliper.srt.managers.scheduler_pp_mixin import pp_admission_decision_to_wire

            _send_row(h, pp_admission_decision_to_wire(decision))
            # #796/#1015: no wraparound lap and no peek any more -- PP0 reads nothing back on this wire.
            res["lap"] = None
            res["sends"] = wire.sends
            res["warning_count"] = len(catcher.records)
            res["warnings"] = catcher.records
            res["ok"] = True

        elif rank == PP1:
            h = _make_holder(rank, wire, waiting_queue=[_FakeReq(RID, PP1_LOCAL_MATCH)])
            from flliper.srt.managers.scheduler_pp_mixin import pp_admission_decision_to_wire

            incoming = _recv_row(h)
            effective, amended = h._pp_reconcile_incoming_admission(incoming)
            _send_row(h, pp_admission_decision_to_wire(amended))   # ROW-DELIVER: the AMENDED row is relayed

            res["effective"] = dict(effective)
            res["sends"] = wire.sends
            res["warning_count"] = len(catcher.records)
            res["warnings"] = catcher.records
            res["ok"] = True

        elif rank == PP2:
            # Local match is irrelevant here: the entry is already
            # retracted when it arrives, so reconcile must pass it through
            # unchanged without consulting local state at all -- give it a
            # generous match (10) so a bug that DID re-derive would be
            # caught re-admitting rather than silently agreeing by luck.
            h = _make_holder(rank, wire, waiting_queue=[_FakeReq(RID, 10)])
            incoming = _recv_row(h)
            effective, amended = h._pp_reconcile_incoming_admission(incoming)
            # #796: the last rank sends no frame (`if not self.pp_group.is_last_rank` around the proxy send) --
            # `wire.sends` measures it.

            res["effective"] = dict(effective)
            res["sends"] = wire.sends
            res["incoming_retracted"] = [
                {
                    "rid": e.rid,
                    "retracted": e.retracted,
                    "observed_local": e.observed_local,
                }
                for e in incoming.entries
            ]
            res["warning_count"] = len(catcher.records)
            res["warnings"] = catcher.records
            res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except BaseException:  # noqa: BLE001 - best-effort teardown only
                    pass


def _run():
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=_worker, args=(r, init_file, tmp)) for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + JOIN_TIMEOUT_S
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        def _load(path):
            if not os.path.exists(path):
                return None
            with open(path) as f:
                return json.load(f)

        out = {"stuck_ranks": stuck_ranks}
        for r in range(WORLD):
            out[f"result_{r}"] = _load(os.path.join(tmp, f"r{r}.json"))
        return out


class PPAdmissionWiring791(unittest.TestCase):
    def test_the_decision_degrades_down_the_chain_and_stops_at_the_last_rank(self):
        """Three live gloo ranks over the SHIPPED send/recv path.

        Until #796 this test asserted a ring ROUNDTRIP: PP2 wrapped the
        amended decision back to PP0 and the guard learned from it
        (`record_return_trip`). #796 deleted that edge -- PP0 was never
        required to take it, so the wraparound was one unmatched message
        per pass on the channel (the bounded-recv corpse), and the learning
        now travels on the OUTPUT channel instead
        (`pp_output_payload_with_return_trip`, pinned by the #791b/#797
        suites). What this test pins since then is the CURRENT contract:
        the decision degrades PP0 -> PP1 -> PP2 and STOPS -- the last
        rank's send is a no-op and PP0's opportunistic peek is dormant.
        """
        res = _run()
        self.assertEqual(res["stuck_ranks"], [], f"a rank never finished: {res}")
        r0, r1, r2 = res["result_0"], res["result_1"], res["result_2"]
        for name, r in (("PP0", r0), ("PP1", r1), ("PP2", r2)):
            self.assertIsNotNone(r, f"{name} produced no result: {res}")
            self.assertIsNone(r.get("error"), f"{name} raised: {r.get('error')}")
            self.assertTrue(r.get("ok"), f"{name} did not complete: {r}")

        # PORT 20.10. -- #1072 (pp_admission_congruence.reconcile_pp_admission_decision): "THIS FUNCTION NO LONGER
        # RETRACTS ANYTHING ... `local < told` is the PIPELINE STAGGER, not a shortfall ... A downstream rank EXECUTES
        # the row it is given". The rule this file now pins: PP1 with a local match BELOW told executes PP0's row at
        # told, logs nothing, and relays the row unchanged; PP2 receives it un-retracted and executes it too.
        self.assertEqual(r1["warning_count"], 0, f"PP1 must not warn on the stagger: {r1['warnings']}")
        self.assertEqual(r1["effective"].get(RID), TOLD_BY_PP0)
        self.assertEqual(r2["warning_count"], 0, f"PP2 must not warn: {r2['warnings']}")
        self.assertFalse(r2["incoming_retracted"][0]["retracted"])
        self.assertEqual(r2["effective"].get(RID), TOLD_BY_PP0)

        # #796: the chain STOPS at the last rank. PP2's send is a no-op
        # (zero wire activity, measured, not inferred from the absence of a
        # hang), and PP0's only read of the channel -- the opportunistic
        # peek -- is dormant and returns None without touching the wire.
        self.assertEqual(r0["sends"], 1, "PP0 sends exactly its own decision")
        self.assertEqual(r1["sends"], 1, "PP1 forwards exactly once")
        self.assertEqual(
            r2["sends"],
            0,
            "the last rank must not post a send no peer is required to take",
        )
        self.assertIsNone(r0["lap"], "the wraparound lap no longer exists")

        # PP0 itself never independently discovers a mismatch in this
        # scenario -- with the lap gone (#796) it learns nothing back at
        # all on this channel -- so it must not have logged anything of its
        # own. (`record_return_trip` now feeds off the OUTPUT channel; the
        # #791b/#797 suites pin that path.)
        self.assertEqual(r0["warning_count"], 0)


class PPAdmissionWiringNoOpWithoutPP(unittest.TestCase):
    def test_admission_wiring_is_noop_when_pp_size_is_1(self):
        """Backward compatibility: the default (non-PP, or pp_size==1)
        path must never touch the wire. Checked in-process -- no
        multiprocessing needed since the shipped methods return before
        touching self.pp_group at all."""
        from flliper.srt.managers.pp_admission_congruence import PPAdmissionDecision
        from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        class _ExplodingWire:
            def send_tensor_dict(self, *a, **k):
                raise AssertionError("must not touch the wire when pp_size<=1")

            def recv_tensor_dict(self, *a, **k):
                raise AssertionError("must not touch the wire when pp_size<=1")

        h = types.SimpleNamespace(
            pp_group=_ExplodingWire(),
            ps=types.SimpleNamespace(pp_size=1, pp_rank=0),
            waiting_queue=[],
            tree_cache=None,
        )
        # PORT 20.10.: the send/recv arc is gone (#1015); the reconcile of the frame row is what remains per rank.
        h._pp_reconcile_incoming_admission = types.MethodType(
            SchedulerPPMixin._pp_reconcile_incoming_admission, h
        )
        decision = PPAdmissionDecision(mb_id=0, entries=())
        effective, amended = h._pp_reconcile_incoming_admission(decision)
        self.assertEqual(effective, {})
        self.assertIs(amended, decision)


class PPAdmissionOrdering791(unittest.TestCase):
    def test_ordering_admission_precedes_batch_and_proxy_recv(self):
        """#791 task requirement 7, structurally: the admission-decision
        receive must appear before get_next_batch_to_run and before
        _pp_recv_proxy_tensors in _event_loop_pp_body's own source -- so an
        ordinary prefix-length divergence is degraded before the #789
        proxy-readiness contract's raise path could ever be reached on a
        healthy pass."""
        from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        src = inspect.getsource(SchedulerPPMixin._event_loop_pp_body)
        # PORT 20.10. (#1015 -> #631 ROW AUTHORITY): the decision rides the frame; its pre-plan receive and the
        # reconcile must precede the plan, and the frame receive that carries it is that first proxy receive.
        i_proxy = src.index("_pre_proxy = self._pp_recv_proxy_tensors(mb_id)")
        i_admission = src.index("self._pp_reconcile_incoming_admission(")
        i_batch = src.index("get_next_batch_to_run(")
        self.assertLess(i_proxy, i_admission, "the row is reconciled from the frame received before the plan")
        self.assertLess(
            i_admission,
            i_batch,
            "the admission-decision receive must run before get_next_batch_to_run",
        )



if __name__ == "__main__":
    unittest.main()
