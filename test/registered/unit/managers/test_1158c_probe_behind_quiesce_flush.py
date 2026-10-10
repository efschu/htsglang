"""#1158c -- a health probe that arrives AFTER the front's quiesce /flush_cache and
before the ReleaseMemoryOccupation that follows it is disposed at the origin.

THE SPECIMEN (boot pdflip_dkr27bggufrabar1fs10100058_a8e569fd83_1010_005819, D.log
01:07:52-53, first idle flip D->P, overlap scheduler). The D event loop started
late (01:07:52); the front's quiesce /flush_cache (sent 01:07:29) and the group
deadman's first direct /health_generate (sent ~01:07:32, ``last_probe=0``) waited
in the socket and came out in ONE intake, flush first. The flush answered idle
(200), the probe passed the #1158 gate (idle) and was enqueued on every rank and
launched as the next batch. The Release arrived 0.7 s later, in the next pass,
while that batch was still in flight: ``last_batch`` + ``overlap_result_queue``
-> the sleep drain's ``other_terms=2`` -> W120 on 3/3 -> W29 -> W17, no serving.
#1158b did not apply: the probe was not in the Release's intake.

The quiesce verdict holds only while nothing is admitted after it; the origin is
the one place that sees both the flush and the probe. A probe behind a flush and
before the next Release (or the next real request) is answered alive at once and
never reaches the broadcast -- every rank gets the same list.

CLAIMS
 (a) [Flush, probe] in one intake: the probe never reaches the wire, it is
     answered at once; the gate is not asked;
 (b) the window spans intakes and ends at the Release: a probe in a later intake
     before the Release is disposed, a probe after the Release follows the gate
     (W25 path on the dormant group, unchanged);
 (c) a real request ends the window (the front serves the group again): a probe
     after it follows the gate;
 (d) an unrelated control request does not end the window.
The negative branch "probe AHEAD of the flush keeps the gate path" is pinned by
test_1158b (test_flush_in_the_list_does_not_drop_a_probe).
"""

import types
import unittest
from unittest import mock

import msgspec
import zmq

from flliper.srt.managers.io_struct import (
    FlushCacheReqInput,
    ReleaseMemoryOccupationReqInput,
    TokenizedGenerateReqInput,
)
from flliper.srt.managers.scheduler_components import request_receiver as rr
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HEALTH_RID = "HEALTH_CHECK_1158cdeadbeef"


class _Wire:
    def __init__(self):
        self.sent = []

    def broadcast(self, data, rank, dist_group=None, src=0, **_kw):
        if rank == src:
            self.sent.append(list(data) if data is not None else data)
            return data
        return list(self.sent[-1])


class _Sock:
    def __init__(self):
        self.items = []

    def pop(self):
        if not self.items:
            raise zmq.ZMQError()
        return self.items.pop(0)


def _sock_recv(sock, _flags=None):
    return sock.pop()


def _gen(rid, ipc):
    required = [f.name for f in msgspec.structs.fields(TokenizedGenerateReqInput) if f.required]
    req = TokenizedGenerateReqInput(**{n: None for n in required}, rid=rid)
    req.http_worker_ipc = ipc
    return req


def _probe(ipc="ipc://probe"):
    return _gen(HEALTH_RID, ipc)


def _release():
    return ReleaseMemoryOccupationReqInput(tags=["kv_cache"], epoch="e1")


def _kind(r):
    if isinstance(r, ReleaseMemoryOccupationReqInput):
        return "RELEASE"
    if isinstance(r, FlushCacheReqInput):
        return "FLUSH"
    return getattr(r, "rid", type(r).__name__)


class _Group:
    """The origin and one TP peer, persistent across intake passes."""

    def __init__(self, idle=True):
        self.wire = _Wire()
        self.deque_ipcs, self.now_ipcs, self.gate_calls = [], [], []
        self.tok, self.rpc = _Sock(), _Sock()

        def gate():
            self.gate_calls.append(1)
            return (idle, 0, 0)

        self.origin = self._receiver(0, gate)
        self.peer = self._receiver(1, gate)

    def _receiver(self, rank, gate):
        ps = types.SimpleNamespace(
            pp_rank=0, pp_size=1, tp_size=2, tp_rank=rank, attn_tp_rank=rank,
            attn_tp_size=2, attn_cp_rank=0, attn_cp_size=1, attn_dp_rank=0,
        )
        server_args = types.SimpleNamespace(
            enable_dp_attention=False, enable_phase_flip=False,
            language_only=False, encoder_transfer_backend=None,
        )
        return rr.SchedulerRequestReceiver(
            recv_from_tokenizer=self.tok if rank == 0 else _Sock(),
            recv_from_rpc=self.rpc if rank == 0 else _Sock(),
            recv_skipper=None, input_blocker=None, mm_receiver=None, ps=ps,
            tp_group=types.SimpleNamespace(rank=rank, ranks=[0, 1]),
            tp_cpu_group=object(), attn_tp_group=None, attn_tp_cpu_group=None,
            attn_cp_group=None, attn_cp_cpu_group=None, world_group=None,
            server_args=server_args,
            model_config=types.SimpleNamespace(is_multimodal=False),
            max_recv_per_poll=-1, stream_output=lambda *a, **k: None,
            get_last_forward_mode=lambda: None,
            health_check_gate=gate if rank == 0 else None,
            return_health_check_ipc=self.deque_ipcs.append if rank == 0 else None,
            answer_health_check_now=self.now_ipcs.append if rank == 0 else None,
        )

    def intake(self, tok=(), rpc=()):
        """One recv pass on both ranks; returns (origin list, peer list) as kinds."""
        self.tok.items = list(tok)
        self.rpc.items = list(rpc)
        with mock.patch.object(rr, "sock_recv", _sock_recv), mock.patch.object(
            rr, "broadcast_pyobj", self.wire.broadcast
        ), mock.patch.object(rr, "unwrap_shm_features", lambda _r: None):
            got0 = self.origin.recv_requests()
            got1 = self.peer.recv_requests()
        return [_kind(r) for r in got0], [_kind(r) for r in got1]


class ProbeBehindQuiesceFlushIsDisposedAtTheOrigin(CustomTestCase):
    def test_flush_then_probe_same_intake_the_specimen(self):
        """THE SPECIMEN: [Flush, probe] on the tokenizer socket, one pass, idle
        gate. RED on 23c85f1d20 (probe kept -> enqueued on every rank), green with
        the fix."""
        g = _Group(idle=True)
        with self.assertLogs(rr.logger, level="INFO") as cm:
            got0, got1 = g.intake(tok=[FlushCacheReqInput(), _probe()])
        self.assertEqual(got0, ["FLUSH"], "no probe behind the quiesce flush")
        self.assertEqual(got1, ["FLUSH"], "peer == origin: replicated")
        self.assertEqual([_kind(r) for r in g.wire.sent[-1]], ["FLUSH"])
        self.assertEqual(g.now_ipcs, ["ipc://probe"], "answered at once, exactly once")
        self.assertEqual(g.deque_ipcs, [])
        self.assertEqual(g.gate_calls, [], "the verdict is the quiesce's, not the gate's")
        self.assertTrue(
            any("#1158c HEALTH-CHECK dropped at origin" in m and HEALTH_RID in m
                for m in cm.output),
            cm.output,
        )

    def test_window_spans_intakes_and_ends_at_the_release(self):
        g = _Group(idle=True)
        g.intake(tok=[FlushCacheReqInput()])
        got0, _ = g.intake(tok=[_probe("ipc://a")])
        self.assertEqual(got0, [], "a later intake before the Release: disposed")
        self.assertEqual(g.now_ipcs, ["ipc://a"])
        g.intake(rpc=[_release()])
        got0, got1 = g.intake(tok=[_probe("ipc://b")])
        self.assertEqual(got0, [HEALTH_RID], "after the Release: the gate path (W25)")
        self.assertEqual(got1, [HEALTH_RID])
        self.assertEqual(g.now_ipcs, ["ipc://a"])
        self.assertEqual(len(g.gate_calls), 1)

    def test_a_real_request_ends_the_window(self):
        g = _Group(idle=True)
        g.intake(tok=[FlushCacheReqInput()])
        got0, _ = g.intake(tok=[_gen("c4e85437work", "ipc://w")])
        self.assertEqual(got0, ["c4e85437work"], "a real request is never dropped")
        got0, _ = g.intake(tok=[_probe()])
        self.assertEqual(got0, [HEALTH_RID], "the front serves again: gate path")
        self.assertEqual(g.now_ipcs, [])
        self.assertEqual(len(g.gate_calls), 1)

    def test_an_unrelated_control_request_does_not_end_the_window(self):
        g = _Group(idle=True)
        g.intake(tok=[FlushCacheReqInput()])
        g.intake(tok=[types.SimpleNamespace(rid=None, http_worker_ipc=None)])
        got0, _ = g.intake(tok=[_probe()])
        self.assertEqual(got0, [], "still between quiesce and release: disposed")
        self.assertEqual(g.now_ipcs, ["ipc://probe"])


if __name__ == "__main__":
    unittest.main()
