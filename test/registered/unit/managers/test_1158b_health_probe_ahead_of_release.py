"""#1158b -- a health probe that rides the SAME intake as a ReleaseMemoryOccupation,
ahead of it, is disposed at the origin (boot race, W120 at the first idle flip).

THE SPECIMEN (boot pdflip_dkr27browauthoritybar1fs10060558_09757b0a44_1006_055847,
D.log 06:02:01). Flush 01.208, front's Release RPC 01.322, the Release reaches the
scheduler at 01.647 -- in the SAME intake pass as a /health_generate probe
(HEALTH_CHECK_6ed9e716...). The origin gate read "idle" (the sleep does not exist
yet), the probe was enqueued on every rank, the release leg's drain read
waiting_queue=[probe] as a non-HiCache blocker -> `other_terms=1` -> W120 on all
ranks -> W29 -> boot without serving. Fifteen earlier boots had the same window and
landed on a harmless side of it.

CLAIMS
 (a) [probe, Release]: the probe never reaches the broadcast; it is answered at once
     (HealthCheckOutput path), the Release and every other request survive in order;
 (b) nothing else changes: a probe in a list WITHOUT a release follows the #1158
     gate verdict (kept when idle, dropped when busy); a probe BEHIND a release
     keeps the old path (W25 on the dormant group); a real request is never
     dropped, whatever its position;
 (c) without the immediate-answer hook the drop answers through the deque
     (return_health_check_ipc) -- the prober is never left without any answer;
 (d) the scheduler wires the immediate answer (structural).

MUTANTS (run by hand, each red): never drop ahead of a release; drop every probe
when a release is anywhere in the list including behind it; drop every probe
regardless of a release; drop non-probe requests ahead of the release; drop
without answering.
"""

import pathlib
import types
import unittest
from unittest import mock

import zmq

from flliper.srt.managers import scheduler as scheduler_mod
from flliper.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from flliper.srt.managers.scheduler_components import request_receiver as rr
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HEALTH_RID = "HEALTH_CHECK_1158bdeadbeef"
WORK_RID = "c4e85437work"


class _Wire:
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


def _release():
    return ReleaseMemoryOccupationReqInput(tags=["kv_cache"], epoch="e1")


def _receiver(rank, tok, rpc, gate, deque_ipcs, now_ipcs, with_now):
    ps = types.SimpleNamespace(
        pp_rank=0, pp_size=1, tp_size=2, tp_rank=rank, attn_tp_rank=rank,
        attn_tp_size=2, attn_cp_rank=0, attn_cp_size=1, attn_dp_rank=0,
    )
    server_args = types.SimpleNamespace(
        enable_dp_attention=False, enable_phase_flip=False,
        language_only=False, encoder_transfer_backend=None,
    )
    return rr.SchedulerRequestReceiver(
        recv_from_tokenizer=_Sock(tok if rank == 0 else ()),
        recv_from_rpc=_Sock(rpc if rank == 0 else ()),
        recv_skipper=None, input_blocker=None, mm_receiver=None, ps=ps,
        tp_group=types.SimpleNamespace(rank=rank, ranks=[0, 1]),
        tp_cpu_group=object(), attn_tp_group=None, attn_tp_cpu_group=None,
        attn_cp_group=None, attn_cp_cpu_group=None, world_group=None,
        server_args=server_args, model_config=types.SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1, stream_output=lambda *a, **k: None,
        get_last_forward_mode=lambda: None,
        health_check_gate=gate if rank == 0 else None,
        return_health_check_ipc=deque_ipcs.append if rank == 0 else None,
        answer_health_check_now=(now_ipcs.append if (rank == 0 and with_now) else None),
    )


def _recv(tok, rpc=(), idle=True, with_now=True):
    """recv_requests on the origin and one TP peer over a fake wire.
    Returns (origin_list, peer_list, wire_payloads, deque_ipcs, now_ipcs, gate_calls)."""
    wire = _Wire()
    deque_ipcs, now_ipcs, gate_calls = [], [], []

    def gate():
        gate_calls.append(1)
        return (idle, 0, 0)

    origin = _receiver(0, tok, rpc, gate, deque_ipcs, now_ipcs, with_now)
    peer = _receiver(1, tok, rpc, gate, deque_ipcs, now_ipcs, with_now)
    with mock.patch.object(rr, "sock_recv", _sock_recv), mock.patch.object(
        rr, "broadcast_pyobj", wire.broadcast
    ), mock.patch.object(rr, "unwrap_shm_features", lambda _r: None):
        got0 = origin.recv_requests()
        got1 = peer.recv_requests()
    return got0, got1, wire.sent, deque_ipcs, now_ipcs, gate_calls


def _kinds(reqs):
    return [
        "RELEASE" if isinstance(r, ReleaseMemoryOccupationReqInput) else r.rid
        for r in reqs
    ]


class ProbeAheadOfReleaseIsDisposedAtTheOrigin(CustomTestCase):
    def test_probe_then_release_same_intake_the_specimen(self):
        """THE SPECIMEN: idle gate, [probe] on the tokenizer socket, [Release] on
        the rpc socket, one pass. RED on HEAD (probe kept), green with the fix."""
        got0, got1, sent, deq, now, gate_calls = _recv(
            [_req(HEALTH_RID)], [_release()], idle=True
        )
        self.assertEqual(len(sent), 1, "exactly one broadcast")
        self.assertEqual(_kinds(sent[0]), ["RELEASE"], "no probe on the wire")
        self.assertEqual(_kinds(got0), ["RELEASE"])
        self.assertEqual(_kinds(got1), ["RELEASE"], "peer == origin: replicated")
        self.assertEqual(now, ["ipc://probe"], "answered at once, exactly once")
        self.assertEqual(deq, [], "not parked behind the next batch result")
        self.assertEqual(gate_calls, [], "the verdict is the release's, not the gate's")

    def test_the_drop_line_names_the_race(self):
        with self.assertLogs(rr.logger, level="INFO") as cm:
            _recv([_req(HEALTH_RID)], [_release()])
        lines = [m for m in cm.output if "#1158b HEALTH-CHECK dropped at origin" in m]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(f"rid={HEALTH_RID}", lines[0])
        self.assertIn("ahead of ReleaseMemoryOccupation", lines[0])

    def test_two_probes_ahead_both_answered_none_kept(self):
        got0, _g1, _s, _d, now, _c = _recv(
            [_req(HEALTH_RID, "ipc://a"), _req(HEALTH_RID + "2", "ipc://b")],
            [_release()],
        )
        self.assertEqual(_kinds(got0), ["RELEASE"])
        self.assertEqual(now, ["ipc://a", "ipc://b"])

    def test_real_requests_are_never_dropped_and_keep_their_order(self):
        """work, probe, work2 on the tokenizer socket + Release on rpc: only the
        probe goes."""
        got0, got1, _s, _d, now, _c = _recv(
            [_req(WORK_RID), _req(HEALTH_RID), _req(WORK_RID + "2")], [_release()]
        )
        self.assertEqual(_kinds(got0), [WORK_RID, WORK_RID + "2", "RELEASE"])
        self.assertEqual(_kinds(got1), [WORK_RID, WORK_RID + "2", "RELEASE"])
        self.assertEqual(now, ["ipc://probe"])

    def test_only_the_health_prefix_is_a_probe(self):
        """A request whose rid merely contains the word is a real request."""
        got0, _g1, _s, _d, now, _c = _recv(
            [_req("user_HEALTH_CHECK_x"), _req("health_check_lower")], [_release()]
        )
        self.assertEqual(
            _kinds(got0), ["user_HEALTH_CHECK_x", "health_check_lower", "RELEASE"]
        )
        self.assertEqual(now, [])

    def test_without_the_now_hook_the_answer_goes_through_the_deque(self):
        got0, _g1, _s, deq, now, _c = _recv(
            [_req(HEALTH_RID)], [_release()], with_now=False
        )
        self.assertEqual(_kinds(got0), ["RELEASE"])
        self.assertEqual(deq, ["ipc://probe"])
        self.assertEqual(now, [])


class NothingElseChanges(CustomTestCase):
    def test_probe_without_a_release_follows_the_gate_idle_kept(self):
        got0, got1, _s, deq, now, gate_calls = _recv([_req(HEALTH_RID)], idle=True)
        self.assertEqual(_kinds(got0), [HEALTH_RID])
        self.assertEqual(_kinds(got1), [HEALTH_RID])
        self.assertEqual((deq, now), ([], []))
        self.assertEqual(len(gate_calls), 1)

    def test_probe_without_a_release_follows_the_gate_busy_dropped_via_deque(self):
        got0, _g1, _s, deq, now, _c = _recv(
            [_req(WORK_RID), _req(HEALTH_RID)], idle=False
        )
        self.assertEqual(_kinds(got0), [WORK_RID])
        self.assertEqual(deq, ["ipc://probe"], "the #1158 busy path is unchanged")
        self.assertEqual(now, [])

    def test_probe_behind_the_release_keeps_the_old_path(self):
        """List order [Release, probe] (rpc cannot precede tokenizer in the real
        intake, but the PP chain relays any order): the probe runs on a dormant
        group and takes the W25 path, as before. Idle gate -> kept."""
        d, n = [], []
        origin = _receiver(0, [], [], lambda: (True, 0, 0), d, n, True)
        with mock.patch.object(rr, "unwrap_shm_features", lambda _r: None):
            out = origin._dispose_health_checks_at_origin(
                [_release(), _req(HEALTH_RID)]
            )
        self.assertEqual(_kinds(out), ["RELEASE", HEALTH_RID])
        self.assertEqual((d, n), ([], []))

    def test_a_list_with_only_a_release_is_untouched(self):
        got0, _g1, _s, deq, now, gate_calls = _recv([_req(WORK_RID)], [_release()])
        self.assertEqual(_kinds(got0), [WORK_RID, "RELEASE"])
        self.assertEqual((deq, now, gate_calls), ([], [], []))

    def test_flush_in_the_list_does_not_drop_a_probe(self):
        """Deliberate scope: only the release kills the boot (a probe queued
        behind a flush makes that flush answer 400, the front re-polls)."""
        from flliper.srt.managers.io_struct import FlushCacheReqInput

        got0, _g1, _s, deq, now, _c = _recv(
            [_req(HEALTH_RID)], [FlushCacheReqInput()], idle=True
        )
        self.assertEqual(_kinds(got0)[0], HEALTH_RID)
        self.assertEqual(now, [])


class SchedulerWiresTheImmediateAnswer(CustomTestCase):
    def test_scheduler_passes_answer_health_check_now_sending_a_health_check_output(self):
        src = pathlib.Path(scheduler_mod.__file__).read_text()
        i = src.index("answer_health_check_now=")
        window = src[i : i + 300]
        self.assertIn("send_to_tokenizer.send_output", window)
        self.assertIn("HealthCheckOutput(http_worker_ipc=ipc)", window)


if __name__ == "__main__":
    unittest.main()
