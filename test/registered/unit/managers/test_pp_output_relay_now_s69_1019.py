"""S69: a middle PP rank forwards a received output in the pass that received it (OUTPUT-RELAY-NOW).

THE SPECIMEN (rc16 bf9ca46f45, boot ..._1019_121629, P log). Outputs travel 2 -> 0 -> 1 -> 2. PP0 took
pdflip-30-41's output off the wire at 12:27:51.702 (``#1463 PASS-TAIL pp_rank=0 ... commit_ms=1``), but on the
contiguous ring a middle rank forwards ``pp_outputs`` -- what it received LAST pass -- inside the NEXT pass's
output commit, i.e. after that pass's launch. PP0's next pass launched pdflip-31-42's 1542-token chunk
(``launch=1787``), so PP1 got the output at 12:27:53.611 (``#1463 PASS-TAIL pp_rank=1 ... commit_ms=4819``):
1909 ms of relay lag. PP1 sat in that blocking output receive with the proxy of pdflip-31-42's first chunk
already on its wire (PP0 ``#631 PROXY-SEND t31`` at 12:27:51) and started that chunk 1.9 s late; the whole
epoch tail moved with it (PP0 ``#1466 PASS-STALL ... commit_ms=3887`` at 12:27:57, the monitor's COMMIT-STALL-P).

THE FIX. ``_pp_relay_output_now`` posts the forward at the end of the receiving pass -- after this pass's proxy
send, before the chain flush -- so the wire ORDER is unchanged (proxy(k-1), output(k), proxy(k), ...) and only
the time moves. The join of that send is filed one lap later (``_pp_post_send(lap_offset=1)``), so it is joined
at exactly the lap today's forward of the same message is joined (#1015e: the join is the cycle).

WHY THREE REAL PROCESSES OVER REAL GLOO (precedent: test_pp_chain_flush_deadlock_788). The relay delay is a
timing fact on a real wire. The SHIPPED ``_pp_commit_send_output_work_and_preprocess_output_tensors``,
``_pp_send_recv_and_preprocess_output_tensors``, ``_pp_send_output_to_next_stage``, ``_pp_commit_comm_work``,
``_pp_post_send``, ``_pp_drain_due_sends``, ``_pp_join_comm_work`` and ``_pp_relay_output_now`` are bound onto
a holder; only the tensor-dict transport is a thin gloo stand-in (one int64 frame per message, demultiplexed by
kind through an inbox as ``_pp_recv_typed_dict`` does). The pass driver follows ``_event_loop_pp_body``'s
order; the placement of the relay inside the real loop is pinned separately below.
"""
import collections
import contextlib
import inspect
import json
import multiprocessing as mp
import os
import tempfile
import time
import types
import unittest

import torch
import torch.distributed as dist

WORLD = 3
LAST = WORLD - 1
KIND = {"proxy": 1, "output": 2}
KIND_NAME = {v: k for k, v in KIND.items()}

#: The specimen, scaled: PP0 receives A's output in the commit of the pass that launches C, and the pass after
#: that launches D for 0.8 s -- the stand-in for PP0's blocking 1787 ms launch of pdflip-31-42's second chunk.
SCHED = {0: 0, 1: 1, 2: 2, 3: 3}  # pass -> request (A=0, B=1, C=2, D=3); slot = pass % 3
LAUNCH_S = {
    0: {0: 0.05, 1: 0.30, 2: 0.30, 3: 0.80},
    1: {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05},
    2: {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05},
}
N_PASSES = 9
JOIN_TIMEOUT_S = 60.0


class _Wire:
    """Real gloo point-to-point, one int64 frame [kind, req, tok] per message, inbox demux by kind."""

    def __init__(self, rank):
        self.rank = rank
        self.prev = (rank - 1) % WORLD
        self.next = (rank + 1) % WORLD
        self.inbox = collections.defaultdict(collections.deque)
        self.log = []  # kinds/reqs in the order they came OFF the wire from prev
        self.sent = collections.Counter()

    def send(self, kind, req, tok):
        from flliper.srt.distributed.parallel_state import P2PWork

        t = torch.tensor([KIND[kind], int(req), int(tok)], dtype=torch.long)
        w = dist.isend(t, dst=self.next)
        self.sent[kind] += 1
        return [P2PWork(work=w, payload=t)]

    def recv(self, kind):
        while not self.inbox[kind]:
            buf = torch.zeros(3, dtype=torch.long)
            dist.recv(buf, src=self.prev)
            k, req, tok = (int(x) for x in buf.tolist())
            self.log.append([KIND_NAME[k], req])
            self.inbox[KIND_NAME[k]].append((req, tok))
        return self.inbox[kind].popleft()


class _Ev:
    def record(self, *a, **k):
        return None

    def synchronize(self):
        return None


class _Stream:
    def wait_event(self, *a, **k):
        return None

    def wait_stream(self, *a, **k):
        return None


def _batch(req):
    fm = types.SimpleNamespace(is_prebuilt=lambda: False)
    return types.SimpleNamespace(
        req=req, reqs=[req], forward_mode=fm, contains_last_prefill_chunk=True, return_logprob=False
    )


def _make_holder(rank, wire):
    """Bind the SHIPPED PP mixin methods to a minimal holder (#757/#788 style)."""
    from flliper.srt.managers import scheduler_pp_mixin as m
    from flliper.srt.model_executor.forward_batch_info import PPProxyTensors

    ps = types.SimpleNamespace(
        pp_rank=rank, pp_size=WORLD, tp_size=1, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0,
        attn_cp_size=1, attn_tp_size=1,
    )
    h = types.SimpleNamespace(
        ps=ps,
        pp_group=types.SimpleNamespace(is_first_rank=(rank == 0), is_last_rank=(rank == LAST)),
        server_args=types.SimpleNamespace(pp_async_batch_depth=0),
        mbs=[None] * WORLD,
        mb_metadata=[None] * WORLD,
        last_rank_comm_queue=collections.deque(),
        pp_outputs=None,
        send_output_work=[],
        send_proxy_work=[],
        send_req_work=[],
        _pp_gapped_wire=False,
        _pp_send_lap=0,
        copy_stream_ctx=contextlib.nullcontext(),
        copy_stream=_Stream(),
        schedule_stream=None,
        device_module=types.SimpleNamespace(Event=_Ev, current_stream=lambda: _Stream()),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
    )
    for name in (
        "_pp_commit_send_output_work_and_preprocess_output_tensors",
        "_pp_send_recv_and_preprocess_output_tensors",
        "_pp_send_output_to_next_stage",
        "_pp_commit_comm_work",
        "_pp_post_send",
        "_pp_drain_due_sends",
        "_pp_send_join_lag",
        "_pp_join_comm_work",
        "_pp_relay_output_now",
    ):
        fn = getattr(m.SchedulerPPMixin, name, None)
        if fn is not None:  # absent on the base: the driver then runs the base's ring
            setattr(h, name, types.MethodType(fn, h))

    def _send_dict(tensor_dict, async_send=True, msg_type="default", **kw):
        return wire.send(msg_type, int(tensor_dict["req"]), int(tensor_dict["tok"]))

    def _recv_output(mb_id=-1, soft=False):
        req, tok = wire.recv("output")
        return {"req": torch.tensor([req]), "tok": torch.tensor([tok])}

    def _prep(target, md, outs):
        return types.SimpleNamespace(req=int(outs["req"][0]), next_token_ids=int(outs["tok"][0]))

    h._pp_send_dict_to_next_stage = _send_dict
    h._pp_recv_dict_from_prev_stage = _recv_output
    h._pp_prep_batch_result = _prep
    h._PPProxyTensors = PPProxyTensors
    return h


def _worker(rank, init_file, out_dir, relay_env):
    os.environ["FLLIPER_PP_OUTPUT_RELAY_NOW"] = relay_env
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    res = {"rank": rank, "ok": False}
    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        wire = _Wire(rank)
        h = _make_holder(rank, wire)
        got = []
        for p in range(N_PASSES):
            mb = p % WORLD
            nxt = (mb + 1) % WORLD
            h._pp_send_lap += 1  # _event_loop_pp_body: one lap tick per pass
            req = SCHED.get(p)
            h.mbs[mb] = _batch(req) if req is not None else None
            if req is not None:
                if rank != 0:
                    preq, _ = wire.recv("proxy")
                    assert preq == req, (rank, p, preq, req)
                h._pp_commit_comm_work(h.send_proxy_work)
                time.sleep(LAUNCH_S[rank][req])  # the blocking launch
                if rank == LAST:
                    h.last_rank_comm_queue.append(
                        (_Ev(), h._PPProxyTensors({"req": torch.tensor([req]), "tok": torch.tensor([1000 + req])}))
                    )
            outs, result, _ev = h._pp_commit_send_output_work_and_preprocess_output_tensors(mb, nxt)
            if h.mbs[nxt] is not None and result is not None:
                got.append([result.req, time.time(), result.next_token_ids])  # host clock, shared by the three ranks
            if req is not None and rank != LAST:
                h.send_proxy_work = wire.send("proxy", req, 0)
            relay = getattr(h, "_pp_relay_output_now", None)
            if relay is not None and relay(outs):
                h.pp_outputs = None
            else:
                h.pp_outputs = outs
        # teardown: join everything this rank still holds (the real loop would join it laps later)
        h._pp_commit_comm_work(h.send_proxy_work)
        h._pp_commit_comm_work(h.send_output_work)
        h._pp_send_lap += 100
        h._pp_drain_due_sends()
        res.update(ok=True, got=got, log=wire.log, sent=dict(wire.sent), leftover={k: len(v) for k, v in wire.inbox.items()})
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        with open(os.path.join(out_dir, f"result_r{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001
                pass


def _run(relay_env):
    # fork, not spawn: the children share the parent's imported mixin pages (rig rule: one 2 GiB scope for the
    # whole run; three spawned interpreters re-importing the scheduler need ~2.6 GiB)
    from flliper.srt.managers import scheduler_pp_mixin  # noqa: F401  (imported before the fork)

    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [ctx.Process(target=_worker, args=(r, init_file, tmp, relay_env)) for r in range(WORLD)]
        for p in procs:
            p.start()
        deadline = time.time() + JOIN_TIMEOUT_S
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck = [r for r, p in enumerate(procs) if p.is_alive()]
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        out = {"stuck": stuck}
        for r in range(WORLD):
            path = os.path.join(tmp, f"result_r{r}.json")
            out[r] = json.load(open(path)) if os.path.exists(path) else None
        return out


def _relay_ms(out, req, src, dst):
    t = {r: {g[0]: g[1] for g in out[r]["got"]} for r in (src, dst)}
    return (t[dst][req] - t[src][req]) * 1000.0


_RUNS = {}


def _cached(env):
    if env not in _RUNS:
        _RUNS[env] = _run(env)
    return _RUNS[env]


class OutputRelayNowRing(unittest.TestCase):
    """Real ring, three processes. ``1`` = the default (relay now), ``0`` = the old one-pass relay."""

    def _ok(self, out):
        self.assertEqual(out["stuck"], [], out)
        for r in range(WORLD):
            self.assertTrue(out[r] and out[r]["ok"], out[r])

    def test_every_output_reaches_every_rank_once_in_order(self):
        for env in ("1", "0"):
            out = _cached(env)
            self._ok(out)
            for r in range(WORLD):
                self.assertEqual([g[0] for g in out[r]["got"]], [0, 1, 2, 3], (env, r))
                self.assertEqual([g[2] for g in out[r]["got"]], [1000, 1001, 1002, 1003], (env, r))
                self.assertEqual(out[r]["leftover"].get("output", 0), 0, (env, r))
            # no duplicate, no drop: every rank sends exactly one output per request
            for r in range(WORLD):
                self.assertEqual(out[r]["sent"].get("output", 0), 4, (env, r, out[r]["sent"]))

    def test_relay_follows_the_receipt_not_the_next_launch(self):
        """RED on rc16: PP1 got A's output only after PP0's NEXT launch (0.8 s), like 12:27:51.702 -> 53.611."""
        out = _cached("1")
        self._ok(out)
        lag = _relay_ms(out, 0, 0, 1)
        self.assertLess(lag, 300.0, f"PP0->PP1 relay of A took {lag:.0f} ms (the next launch is 800 ms)")

    def test_the_old_form_reproduces_the_specimen(self):
        """The switch-off form is the shipped rc16 ring: the relay waits for PP0's next launch."""
        out = _cached("0")
        self._ok(out)
        self.assertGreater(_relay_ms(out, 0, 0, 1), 600.0)

    def test_wire_order_is_unchanged(self):
        """Only the TIME of the forward moves: every rank reads the same kind sequence off its wire in both forms."""
        on, off = _cached("1"), _cached("0")
        self._ok(on)
        self._ok(off)
        for r in range(WORLD):
            self.assertEqual(on[r]["log"], off[r]["log"], r)


class OutputRelayNowUnits(unittest.TestCase):
    def setUp(self):
        from flliper.srt.managers import scheduler_pp_mixin as m

        self.m = m
        self._env = os.environ.get("FLLIPER_PP_OUTPUT_RELAY_NOW")
        os.environ.pop("FLLIPER_PP_OUTPUT_RELAY_NOW", None)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("FLLIPER_PP_OUTPUT_RELAY_NOW", None)
        else:
            os.environ["FLLIPER_PP_OUTPUT_RELAY_NOW"] = self._env

    def _holder(self, rank=0, depth=0, gapped=False):
        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=rank, pp_size=WORLD),
            pp_group=types.SimpleNamespace(is_last_rank=(rank == LAST)),
            server_args=types.SimpleNamespace(pp_async_batch_depth=depth),
            _pp_gapped_wire=gapped,
            send_output_work=[],
            posted=[],
        )
        h._pp_send_dict_to_next_stage = lambda d, async_send=True, msg_type="default", **k: (
            h.posted.append((msg_type, dict(d))) or [("work", msg_type)]
        )
        h._pp_relay_output_now = types.MethodType(self.m.SchedulerPPMixin._pp_relay_output_now, h)
        return h

    def _outs(self):
        from flliper.srt.model_executor.forward_batch_info import PPProxyTensors

        return PPProxyTensors({"next_token_ids": torch.tensor([7])})

    def test_default_is_on(self):
        from flliper.srt.environ import envs

        self.assertTrue(envs.FLLIPER_PP_OUTPUT_RELAY_NOW.get())

    def test_middle_and_first_rank_relay_once_as_output(self):
        for rank in (0, 1):
            h = self._holder(rank)
            self.assertTrue(h._pp_relay_output_now(self._outs()))
            self.assertEqual([p[0] for p in h.posted], ["output"])
            self.assertEqual(h.send_output_work, [("work", "output")])
            self.assertTrue(h._pp_relay_posted)

    def test_nothing_received_posts_nothing_but_owns_the_pass(self):
        h = self._holder(1)
        self.assertTrue(h._pp_relay_output_now(None))
        self.assertEqual(h.posted, [])
        self.assertFalse(getattr(h, "_pp_relay_posted", False))

    def test_last_rank_never_relays(self):
        h = self._holder(LAST)
        self.assertFalse(h._pp_relay_output_now(self._outs()))
        self.assertEqual(h.posted, [])

    def test_gapped_wire_keeps_its_own_lag_removal(self):
        h = self._holder(1, gapped=True)
        self.assertFalse(h._pp_relay_output_now(self._outs()))
        self.assertEqual(h.posted, [])

    def test_async_batch_depth_keeps_the_old_relay(self):
        h = self._holder(1, depth=1)
        self.assertFalse(h._pp_relay_output_now(self._outs()))

    def test_xpu_keeps_the_parity_ordered_exchange(self):
        orig = self.m.is_xpu
        self.m.is_xpu = lambda: True
        try:
            h = self._holder(1)
            self.assertFalse(h._pp_relay_output_now(self._outs()))
        finally:
            self.m.is_xpu = orig

    def test_switch_off_is_the_old_ring(self):
        os.environ["FLLIPER_PP_OUTPUT_RELAY_NOW"] = "0"
        h = self._holder(1)
        self.assertFalse(h._pp_relay_output_now(self._outs()))
        self.assertEqual(h.posted, [])

    def _join_lap_of_relay(self, relay_on):
        """Drive the SHIPPED commit + post/drain on PP0 for one received output and return the lap at which
        the forward of that output is JOINED. The join of a relayed send is the #1015e cycle point: it may
        not move earlier than the old ring's join of the same message."""
        from flliper.srt.model_executor.forward_batch_info import PPProxyTensors

        m = self.m
        os.environ["FLLIPER_PP_OUTPUT_RELAY_NOW"] = "1" if relay_on else "0"
        joined = []
        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=0, pp_size=WORLD),
            pp_group=types.SimpleNamespace(is_last_rank=False),
            server_args=types.SimpleNamespace(pp_async_batch_depth=0),
            _pp_gapped_wire=False, send_output_work=[], pp_outputs=None, _pp_send_lap=0,
            mbs=[None] * WORLD, mb_metadata=[None] * WORLD, last_rank_comm_queue=collections.deque(),
        )
        for name in ("_pp_commit_send_output_work_and_preprocess_output_tensors", "_pp_commit_comm_work",
                     "_pp_post_send", "_pp_drain_due_sends", "_pp_send_join_lag", "_pp_relay_output_now"):
            fn = getattr(m.SchedulerPPMixin, name, None)
            if fn is not None:
                setattr(h, name, types.MethodType(fn, h))
        h._pp_join_comm_work = lambda work: (joined.extend((h._pp_send_lap, w) for w in work), work.clear())
        received = {"n": 0}

        def _exchange(nf, nm, mbs, md, q, pp_outputs):
            work = []
            if pp_outputs:  # the old relay: forward what was received LAST pass
                work = [("fwd", int(pp_outputs["next_token_ids"][0]))]
            outs = None
            if received["n"] == 0:  # pass 1 receives the one output of this test
                received["n"] = 1
                outs = PPProxyTensors({"next_token_ids": torch.tensor([42])})
            return outs, None, None, work

        h._pp_send_recv_and_preprocess_output_tensors = _exchange
        h._pp_send_dict_to_next_stage = lambda d, async_send=True, msg_type="default", **k: [
            ("fwd", int(d["next_token_ids"][0]))
        ]
        for _ in range(12):
            h._pp_send_lap += 1
            # a busy pipeline posts a proxy every pass; the due-send drain rides on those posts
            h._pp_commit_comm_work([("proxy", h._pp_send_lap)])
            outs, _, _ = h._pp_commit_send_output_work_and_preprocess_output_tensors(0, 1)
            relay = getattr(h, "_pp_relay_output_now", None)
            if relay is not None and relay(outs):
                h.pp_outputs = None
            else:
                h.pp_outputs = outs
        laps = [lap for lap, w in joined if w == ("fwd", 42)]
        self.assertEqual(len(laps), 1, joined)
        return laps[0]

    def test_relayed_send_is_joined_at_the_old_lap(self):
        self.assertEqual(self._join_lap_of_relay(True), self._join_lap_of_relay(False))

    def test_the_commit_that_files_the_relay_clears_the_mark(self):
        """Only the ONE early-posted relay is filed late; the next commit's sends are filed at their own lap."""
        m = self.m
        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=1, pp_size=WORLD), _pp_gapped_wire=False, send_output_work=[],
            pp_outputs=None, _pp_send_lap=10, mbs=[None] * WORLD, mb_metadata=[None] * WORLD,
            last_rank_comm_queue=collections.deque(), _pp_relay_posted=True,
        )
        for name in ("_pp_commit_send_output_work_and_preprocess_output_tensors", "_pp_commit_comm_work",
                     "_pp_post_send", "_pp_drain_due_sends", "_pp_send_join_lag"):
            setattr(h, name, types.MethodType(getattr(m.SchedulerPPMixin, name), h))
        h._pp_join_comm_work = lambda work: work.clear()
        posts = iter([[("relay", 1)], [("later", 2)]])
        h._pp_send_recv_and_preprocess_output_tensors = lambda *a: (None, None, None, [])
        h.send_output_work = next(posts)
        h._pp_commit_send_output_work_and_preprocess_output_tensors(0, 1)
        self.assertFalse(h._pp_relay_posted)
        self.assertEqual(h._pp_pending_sends, {11: [[("relay", 1)]]})
        h._pp_send_lap = 11
        h.send_output_work = next(posts)
        h._pp_commit_send_output_work_and_preprocess_output_tensors(0, 1)
        self.assertEqual(h._pp_pending_sends[11], [[("relay", 1)], [("later", 2)]])


class OutputRelayNowWiring(unittest.TestCase):
    """The real loop body: the relay sits after this pass's proxy send and before the chain flush."""

    def setUp(self):
        from flliper.srt.managers import scheduler_pp_mixin as m

        self.src = inspect.getsource(m.SchedulerPPMixin._event_loop_pp_body)
        self.m = m

    def test_relay_after_proxy_send_before_chain_flush(self):
        i_proxy = self.src.index('"send_proxy_dict_to_next_stage"')
        i_relay = self.src.index("self._pp_relay_output_now(next_pp_outputs)")
        i_flush = self.src.index("self._pp_commit_pending_req_work()")
        self.assertLess(i_proxy, i_relay)
        self.assertLess(i_relay, i_flush)

    def test_pp_outputs_is_kept_only_when_the_relay_did_not_take_it(self):
        i_relay = self.src.index("self._pp_relay_output_now(next_pp_outputs)")
        tail = self.src[i_relay:i_relay + 400]
        self.assertIn("self.pp_outputs = None", tail)
        self.assertIn("self.pp_outputs = next_pp_outputs", tail)
        self.assertEqual(self.src.count("self.pp_outputs = next_pp_outputs"), 1)

    def test_cutover_reset_clears_the_relay_mark(self):
        src = inspect.getsource(self.m.SchedulerPPMixin.init_pp_loop_state)
        self.assertIn("self._pp_relay_posted = False", src)


if __name__ == "__main__":
    unittest.main()
