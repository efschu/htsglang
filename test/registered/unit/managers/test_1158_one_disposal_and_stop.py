"""#1158 -- ONE health-check disposal at the origin; a ballot digest mismatch
is a group STOP; the ORDER arm's silent degradation is a STOP.

THE SPECIMEN (boot weg1b3 @ 6980c75eac, log 5585-5587 / 8284 / 29349 / 38834).
A /health_generate probe reached all three ranks at 23:54:18. PP0 was busy and
DROPPED it in its dispatch loop; PP1/PP2 took the #631 row-authority branch of
the same loop and ENQUEUED it. From that second PP0 queue=6 vs PP1/PP2 queue=7,
+1 through five seams, '#969C READMIT-PREFETCH rid=HEALTH_C' on the followers
only. The #791b ballot saw the disagreement on every TP pass (18 mismatch lines,
cadence 1..32, 0 'restored') and FELL BACK to the rank-local verdict each time,
until the rank-local verdicts split at 23:59:54 and PP0/PP2 formed a batch PP1
never joined.

THREE CLAIMS, driven and structural:

 (a) the disposal is taken ONCE, on the request origin, BEFORE the TP
     broadcast: the broadcast payload excludes the probe when the origin is
     busy and includes it when idle (a 2-rank fake of recv_requests over a
     fake broadcast wire); no rank-conditional disposal survives in
     process_input_requests;
 (b) unpack_prefetch_ballot RAISES on min != max with both digests in the
     text, returns the ballot on agreement, and the scheduler no longer
     carries the void/fallback/streak wiring;
 (c) _apply_uniform_head_order re-raises a failed group order as the named
     STOP instead of forming rank-locally.

MUTANTS (run by hand, each red): (a) keep the probe when busy / drop it after
the broadcast; (b) `return None` instead of raise, or raise only once a streak
counter exceeds 1; (c) restore `source = SOURCE_RANK_LOCAL` in the except.
"""

import ast
import inspect
import pathlib
import textwrap
import types
import unittest
from unittest import mock

import zmq

from sglang.srt.managers import prefetch_ballot, scheduler as scheduler_mod
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import request_receiver as rr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HEALTH_RID = "HEALTH_CHECK_1158deadbeef"
WORK_RID = "c4e85437work"


# --------------------------------------------------------------------------
# (a) the 2-rank fake of recv_requests
# --------------------------------------------------------------------------


class _Wire:
    """The TP broadcast, as a list: the source appends, the peers read."""

    def __init__(self):
        self.sent = []

    def broadcast(self, data, rank, dist_group=None, src=0, **_kw):
        if rank == src:
            self.sent.append(list(data) if data is not None else data)
            return data
        return list(self.sent[-1])


class _Sock:
    def __init__(self, items=()):
        self.items = list(items)

    def pop(self):
        if not self.items:
            raise zmq.ZMQError()
        return self.items.pop(0)


def _sock_recv(sock, _flags=None):
    return sock.pop()


def _req(rid, ipc="ipc://probe"):
    return types.SimpleNamespace(rid=rid, http_worker_ipc=ipc)


def _receiver(rank, intake, gate, returned):
    ps = types.SimpleNamespace(
        pp_rank=0,
        pp_size=1,
        tp_size=2,
        tp_rank=rank,
        attn_tp_rank=rank,
        attn_tp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
    )
    server_args = types.SimpleNamespace(
        enable_dp_attention=False,
        enable_phase_flip=False,
        language_only=False,
        encoder_transfer_backend=None,
    )
    return rr.SchedulerRequestReceiver(
        recv_from_tokenizer=_Sock(intake if rank == 0 else ()),
        recv_from_rpc=_Sock(),
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=ps,
        tp_group=types.SimpleNamespace(rank=rank, ranks=[0, 1]),
        tp_cpu_group=object(),
        attn_tp_group=None,
        attn_tp_cpu_group=None,
        attn_cp_group=None,
        attn_cp_cpu_group=None,
        world_group=None,
        server_args=server_args,
        model_config=types.SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *a, **k: None,
        get_last_forward_mode=lambda: None,
        health_check_gate=gate if rank == 0 else None,
        return_health_check_ipc=returned.append if rank == 0 else None,
    )


def _two_rank_recv(intake, idle, queue_len=6, running=1):
    """Run recv_requests on the origin and on one TP peer over a fake wire.

    Returns (origin_list, peer_list, broadcast_payload, returned_ipcs).
    """
    wire = _Wire()
    returned = []
    gate = lambda: (idle, queue_len, running)  # noqa: E731 - a one-line stub
    origin = _receiver(0, intake, gate, returned)
    peer = _receiver(1, intake, gate, returned)
    # unwrap_shm_features reads the GLOBAL server args (a multimodal concern,
    # not this test's): stubbed so the fake stays hermetic.
    with mock.patch.object(rr, "sock_recv", _sock_recv), mock.patch.object(
        rr, "broadcast_pyobj", wire.broadcast
    ), mock.patch.object(rr, "unwrap_shm_features", lambda _r: None):
        got0 = origin.recv_requests()
        got1 = peer.recv_requests()
    return got0, got1, wire.sent, returned


def _rids(reqs):
    return [r.rid for r in reqs]


class OneDisposalAtTheOrigin(CustomTestCase):
    def test_busy_origin_drops_the_probe_before_the_broadcast(self):
        """THE WEG1B3 SHAPE INVERTED: the peer never sees a probe the origin
        dropped, because the drop happens before the payload is put on the
        wire -- and the probe's ipc is answered from the origin."""
        got0, got1, sent, returned = _two_rank_recv(
            [_req(WORK_RID), _req(HEALTH_RID)], idle=False
        )
        self.assertEqual(len(sent), 1, "exactly one broadcast")
        self.assertEqual(_rids(sent[0]), [WORK_RID], "the payload carries no probe")
        self.assertEqual(_rids(got0), [WORK_RID])
        self.assertEqual(_rids(got1), [WORK_RID], "peer == origin: replicated")
        self.assertEqual(returned, ["ipc://probe"], "the dropped probe is answered")

    def test_idle_origin_keeps_the_probe_and_the_peer_gets_it_too(self):
        got0, got1, sent, returned = _two_rank_recv([_req(HEALTH_RID)], idle=True)
        self.assertEqual(_rids(sent[0]), [HEALTH_RID])
        self.assertEqual(_rids(got0), [HEALTH_RID])
        self.assertEqual(_rids(got1), [HEALTH_RID], "a kept probe is enqueued on every rank")
        self.assertEqual(returned, [])

    def test_a_second_probe_behind_a_kept_one_is_busy_by_construction(self):
        got0, _got1, _sent, returned = _two_rank_recv(
            [_req(HEALTH_RID, "ipc://a"), _req(HEALTH_RID + "2", "ipc://b")], idle=True
        )
        self.assertEqual(_rids(got0), [HEALTH_RID])
        self.assertEqual(returned, ["ipc://b"])

    def test_the_drop_line_is_the_boot_4_proof_line(self):
        with self.assertLogs(rr.logger, level="INFO") as cm:
            _two_rank_recv([_req(HEALTH_RID)], idle=False, queue_len=6, running=1)
        lines = [m for m in cm.output if "#1158 HEALTH-CHECK dropped at origin" in m]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(f"before broadcast rid={HEALTH_RID} busy queue=6 running=1", lines[0])

    def test_a_receiver_without_a_gate_relays_untouched(self):
        """No scheduler behind the receiver: nothing is disposed here."""
        returned = []
        origin = _receiver(0, [_req(HEALTH_RID)], None, returned)
        wire = _Wire()
        with mock.patch.object(rr, "sock_recv", _sock_recv), mock.patch.object(
            rr, "broadcast_pyobj", wire.broadcast
        ), mock.patch.object(rr, "unwrap_shm_features", lambda _r: None):
            got = origin.recv_requests()
        self.assertEqual(_rids(got), [HEALTH_RID])
        self.assertEqual(returned, [])


class NoRankConditionalDisposalSurvives(CustomTestCase):
    """STRUCTURAL: the matched check the operator named."""

    FAMILY = [
        "managers/scheduler.py",
        "managers/scheduler_pp_mixin.py",
    ]

    @staticmethod
    def _srt():
        return pathlib.Path(scheduler_mod.__file__).resolve().parents[1]

    def _predicate_sites(self):
        files = [self._srt() / f for f in self.FAMILY] + sorted(
            (self._srt() / "managers" / "scheduler_components").glob("*.py")
        )
        sites = []
        for f in files:
            tree = ast.parse(f.read_text(), str(f))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                calls = [
                    n
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Call)
                    and (
                        (isinstance(n.func, ast.Name) and n.func.id == "is_health_check_generate_req")
                        or (
                            isinstance(n.func, ast.Attribute)
                            and n.func.attr == "is_health_check_generate_req"
                        )
                    )
                ]
                if not calls:
                    continue
                src = ast.get_source_segment(f.read_text(), fn) or ""
                disposes = (
                    "return_health_check_ipc" in src
                    or "http_worker_ipc" in src
                    or "waiting_queue = [" in src
                )
                sites.append((f.name, fn.name, disposes))
        return sites

    def test_exactly_one_disposal_site_and_it_is_the_origin(self):
        sites = self._predicate_sites()
        disposing = [(f, fn) for f, fn, d in sites if d]
        self.assertEqual(
            disposing,
            [("request_receiver.py", "_dispose_health_checks_at_origin")],
            f"all predicate sites: {sites}",
        )

    def test_process_input_requests_has_no_pp_rank_and_no_health_gate(self):
        src = inspect.getsource(Scheduler.process_input_requests)
        tree = ast.parse(textwrap.dedent(src))
        toks = [
            n.lineno
            for n in ast.walk(tree)
            if (isinstance(n, ast.Attribute) and n.attr == "pp_rank")
            or (isinstance(n, ast.Name) and n.id == "pp_rank")
        ]
        self.assertEqual(toks, [], "process_input_requests branches on pp_rank again")
        self.assertNotIn("is_health_check_generate_req(", src)
        self.assertNotIn("return_health_check_ipcs.append", src)

    def test_the_disposal_precedes_the_broadcast_in_recv_requests(self):
        src = inspect.getsource(rr.SchedulerRequestReceiver.recv_requests)
        self.assertLess(
            src.index("_dispose_health_checks_at_origin("),
            src.index("_broadcast_reqs_across_ranks("),
        )

    def test_the_scheduler_wires_the_gate_on_every_boot(self):
        src = inspect.getsource(Scheduler.init_request_receiver)
        self.assertIn("health_check_gate=self._health_check_gate", src)
        self.assertIn("return_health_check_ipc=lambda ipc: self.return_health_check_ipcs.append(", src)


# --------------------------------------------------------------------------
# (b) a digest mismatch is a STOP
# --------------------------------------------------------------------------


def _reduced(min_digest, max_digest, verdicts=()):
    slots = prefetch_ballot.PREFETCH_BALLOT_SLOTS
    body = [1 if v else 0 for v in verdicts] + [1] * (slots - len(verdicts))
    return [min_digest, -max_digest] + body


class BallotDigestMismatchIsAStop(CustomTestCase):
    RIDS = ["5e58e0b6fd32", "25bd2696b3e7"]

    def test_min_neq_max_raises_with_both_digests(self):
        own = prefetch_ballot.prefetch_ballot_digest(self.RIDS)
        with self.assertRaises(RuntimeError) as cm:
            prefetch_ballot.unpack_prefetch_ballot(
                _reduced(837080468, 1265531867), self.RIDS, rank=1, queue_len=7
            )
        msg = str(cm.exception)
        self.assertIn("#791b PREFETCH-BALLOT DIGEST MISMATCH STOP", msg)
        self.assertIn("rank=1", msg)
        self.assertIn(f"digest={own}", msg)
        self.assertIn("group_min=837080468", msg)
        self.assertIn("group_max=1265531867", msg)
        self.assertIn("queue_len=7", msg)
        self.assertIn("head=[5e58e0b6,25bd2696]", msg)
        self.assertIsInstance(cm.exception, prefetch_ballot.PrefetchBallotDigestMismatch)

    def test_the_first_diverged_pass_raises_no_streak_no_cadence(self):
        """The mutant the operator named: raise only once a counter reaches
        a value. Every call here is a fresh first pass and must raise."""
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                prefetch_ballot.unpack_prefetch_ballot(_reduced(1, 2), self.RIDS)

    def test_agreement_returns_the_group_verdict(self):
        d = prefetch_ballot.prefetch_ballot_digest(self.RIDS)
        ballot = prefetch_ballot.unpack_prefetch_ballot(
            _reduced(d, d, [False, True]), self.RIDS
        )
        self.assertEqual(ballot, {self.RIDS[0]: False, self.RIDS[1]: True})

    def test_a_wrong_width_is_none_for_the_caller_to_stop_on(self):
        self.assertIsNone(prefetch_ballot.unpack_prefetch_ballot([1, -1], self.RIDS))

    def test_no_ballot_taken_is_still_the_local_verdict(self):
        """None is reserved for the single-rank / PP-loop callers."""
        for local in (True, False):
            self.assertEqual(
                prefetch_ballot.prefetch_done_under_ballot(local, "x", None), local
            )


class TheSchedulerCarriesNoFallback(CustomTestCase):
    """STRUCTURAL over the shipped TP-loop reduce."""

    def _src(self):
        return inspect.getsource(Scheduler._update_uniform_pool_budget)

    def test_the_void_fallback_and_the_streak_are_gone(self):
        src = self._src()
        for gone in (
            "Ballot void for",
            "falls back to the rank-local",
            "_prefetch_ballot_mismatch_streak",
            "_prefetch_ballot_mismatch_total",
            "agreement restored",
        ):
            self.assertNotIn(gone, src, gone)

    def test_the_tp_loop_asserts_a_ballot_after_the_unpack(self):
        src = self._src()
        i = src.index("unpack_prefetch_ballot(")
        tail = src[i:]
        self.assertIn("if self._uniform_prefetch_ballot is None:", tail)
        self.assertIn("raise RuntimeError(", tail[: tail.index("_uniform_min_avail")])
        self.assertIn("rank=int(self.ps.tp_rank)", tail)

    def test_the_stop_reaches_the_process_death_path(self):
        """No new collective, no new handler: run_scheduler_process's
        except -> SIGQUIT is the group stop, as #1153 documented."""
        src = inspect.getsource(scheduler_mod.run_scheduler_process)
        self.assertIn("except Exception:", src)
        self.assertIn("parent_process.send_signal(signal.SIGQUIT)", src)
        self.assertTrue(
            issubclass(prefetch_ballot.PrefetchBallotDigestMismatch, RuntimeError)
        )


# --------------------------------------------------------------------------
# (c) the ORDER arm's silent degradation is a STOP
# --------------------------------------------------------------------------


class HeadOrderApplyFailureIsAStop(CustomTestCase):
    def _self(self, enabled=True):
        notes = []
        return (
            types.SimpleNamespace(
                waiting_queue=[],
                _tp_head_enforcer_gate=lambda: types.SimpleNamespace(enabled=enabled),
                _note_tp_head_degradation=lambda arm, d: notes.append((arm, d)),
            ),
            notes,
        )

    def test_a_failed_group_order_raises_the_named_stop(self):
        fake, notes = self._self()
        head = types.SimpleNamespace(canonical=["a"], group_match_lens=[0], digest_agreed=True)

        def boom(*_a, **_k):
            raise KeyError("a")

        with mock.patch.object(scheduler_mod.tp_head_congruence, "head_decision", boom):
            with self.assertRaises(RuntimeError) as cm:
                Scheduler._apply_uniform_head_order(fake, head)
        self.assertIn("#823 HEAD-ORDER APPLY STOP", str(cm.exception))
        self.assertEqual(notes, [], "no degradation note: the pass did not continue")

    def test_a_disabled_gate_still_forms_rank_locally_and_says_so(self):
        fake, notes = self._self(enabled=False)
        Scheduler._apply_uniform_head_order(fake, None)
        self.assertEqual(len(notes), 1)
        self.assertFalse(notes[0][1], "gate off: rank-local is not a defect")

    def test_no_silent_rank_local_source_in_the_except(self):
        src = inspect.getsource(Scheduler._apply_uniform_head_order)
        handler = src[src.index("except Exception as exc"):]
        self.assertNotIn("SOURCE_RANK_LOCAL", handler)
        self.assertIn("raise RuntimeError(", handler)


if __name__ == "__main__":
    unittest.main()
