"""--p-attn-head-split V2 (27B, 26.09., release table row 22, NVFP4 long):
PP0 (5090) owner hands HALF kv groups to both downstream 3080 stages.

Pinned without a GPU (design: /spinning/gpu-arb/docs/P_MICROBATCH_AH2.md sec. 4,
module weg2/attn_head_split.py, docstring "V2"):
  * SPLIT GROUP == FULL GROUP -- owner pieces (whole groups, a partial group
    = the second flashinfer call) + two helpers' halves of one group, through
    the fp8 mirror, give torch.equal the unsplit attention; so does the
    owner's self-computed fallback;
  * NO DEADLOCK -- a simulated three-stage pipeline (blocking proxy frames,
    blocking barlink with its TWO slots per pair, side streams as FIFO worker
    threads) completes bit-equal while PP1/PP2's main threads are blocked on
    PP0's frame or busy with the previous chunk; the NAIVE variant (helper
    work served by the stage's main loop between its own forwards) stalls --
    the danger direction, pinned;
  * DEADLINE -- a late helper makes the owner compute the range itself
    (bit-equal), end the split for that request, drain the owed outputs, and
    the next request splits again; nothing is left on any wire;
  * THE SERVER HELPER decides nothing and crash-stops by name on any
    announcement it cannot continue;
  * THE POST -- the mirror is booked on BOTH 3080 stages (256 MiB at 128k),
    the owner's shapes/messages on PP0, and the planner charges exactly that;
  * A/B per request, env and launcher round trips; V1 env bytes unchanged.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import unittest
from collections import defaultdict, deque

import torch

from sglang.srt.weg2 import attn_head_split as AH
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn
T_OUT = 20.0  # anything blocking longer than this in a test is a deadlock


# ---------------------------------------------------------------------------
# fakes: blocking barlink (two slots per pair), side streams, mini stages
# ---------------------------------------------------------------------------


class Deadlock(RuntimeError):
    pass


class BlockingNet:
    def __init__(self, slots=2, timeout=T_OUT):
        self.cv = threading.Condition()
        self.box = defaultdict(deque)  # (src, dst) -> deque[(seq_start, tensor)]
        self.pub = defaultdict(int)
        self.slots, self.timeout = slots, timeout

    def idle(self):
        with self.cv:
            return all(not q for q in self.box.values())


class BlockingWire:
    """barlink p2p as the split sees it: a send blocks while both slots of
    the pair are occupied, a recv blocks until a message is there (both
    model a spin kernel on a stream), published/peek are host reads."""

    def __init__(self, net, rank, p2p_bytes=1 << 30):
        self.net, self.rank, self.p2p_bytes = net, rank, p2p_bytes

    def pieces(self, nbytes):
        return max(1, -(-int(nbytes) // self.p2p_bytes))

    def send(self, t, dst):
        n = self.net
        with n.cv:
            q = n.box[(self.rank, dst)]
            if not n.cv.wait_for(lambda: len(q) < n.slots, timeout=n.timeout):
                raise Deadlock(f"send PP{self.rank}->PP{dst}: both barlink slots stayed full")
            q.append((n.pub[(self.rank, dst)] + 1, t.detach().clone().reshape(-1)))
            n.pub[(self.rank, dst)] += self.pieces(t.numel() * t.element_size())
            n.cv.notify_all()

    def recv(self, t, src):
        n = self.net
        with n.cv:
            q = n.box[(src, self.rank)]
            if not n.cv.wait_for(lambda: len(q) > 0, timeout=n.timeout):
                raise Deadlock(f"recv PP{src}->PP{self.rank}: nothing arrived")
            t.view(-1).copy_(q.popleft()[1])
            n.cv.notify_all()

    def published(self, src):
        with self.net.cv:
            return self.net.pub[(src, self.rank)]

    def peek_header(self, src, seq):
        with self.net.cv:
            for s0, t in self.net.box[(src, self.rank)]:
                if s0 == seq:
                    return [int(x) for x in t[: AH.HDR_BF16].view(torch.int64)]
        raise AssertionError(f"peek PP{src}->PP{self.rank} seq {seq}: no such message")


class TEvent:
    def __init__(self):
        self.e = threading.Event()

    def query(self):
        return self.e.is_set()

    def synchronize(self):
        if not self.e.wait(T_OUT):
            raise Deadlock("event never completed")


class ThreadExec:
    """A side stream: FIFO, runs on its own thread, a blocking recv inside
    blocks the stream and never the host. After a failure the work items are
    skipped but events still complete (the host sees ``failure``)."""

    def __init__(self, name):
        self.q = queue.Queue()
        self.failure = None
        self.t = threading.Thread(target=self._run, name=name, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            kind, fn = item
            if kind == "ev":
                fn()
                continue
            if self.failure:
                continue
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                self.failure = f"{type(exc).__name__}: {exc}"

    def submit(self, fn):
        self.q.put(("fn", fn))

    def record(self):
        ev = TEvent()
        self.q.put(("ev", ev.e.set))
        return ev

    def main_record(self):
        return AH._DoneEvent()

    def side_waits(self, ev):
        self.submit(ev.synchronize)

    def main_waits(self, ev):
        ev.synchronize()
        if self.failure:
            raise RuntimeError(f"side stream failed: {self.failure}")

    def keep(self, t):
        return None

    def synchronize(self):
        self.record().synchronize()

    def close(self):
        self.q.put(None)


def v2_cfg(spec="0:1:2:h4-5,0:2:2:h6-7", deadline_ms=5000.0, ab="", cap=512, min_w=32):
    # mini geometry: 8 q heads over 2 kv groups (gqa 4); group 1 split 2+2
    return AH.AHConfig(AH.parse_spec(spec), min_w=min_w, cap_tokens=cap, max_w=64,
                       head_dim=16, gqa=4, num_kv_heads=2, deadline_ms=deadline_ms, ab=ab)


class Stage:
    """Attention layers of one P stage: q/k/v/o projections, an fp8 pool
    written like set_kv_buffer, reference paged attention over it."""

    def __init__(self, cfg, ids, hidden=48, seed=0, busy_s=0.0):
        g = torch.Generator().manual_seed(seed)
        self.cfg, self.ids, self.hidden, self.busy_s = cfg, ids, hidden, busy_s
        H, Hk, D = cfg.num_kv_heads * cfg.gqa, cfg.num_kv_heads, cfg.head_dim
        self.H, self.Hk, self.D, self.sm = H, Hk, D, D ** -0.5
        mk = lambda *s: (torch.randn(*s, generator=g) * 0.2).to(torch.bfloat16)
        self.w = {lid: (mk(hidden, H * D), mk(hidden, Hk * D), mk(hidden, Hk * D), mk(H * D, hidden))
                  for lid in ids}
        self.pool = {lid: (torch.zeros(cfg.cap_tokens, Hk, D, dtype=FP8),
                           torch.zeros(cfg.cap_tokens, Hk, D, dtype=FP8)) for lid in ids}

    def own_fn(self, lid, p):
        def fn(q_sub, k3, v3, g0, g1, store=True):
            if store:
                pk, pv = self.pool[lid]
                pk[p:p + k3.shape[0]] = k3.to(FP8)
                pv[p:p + v3.shape[0]] = v3.to(FP8)
            pk, pv = self.pool[lid]
            return AH.reference_attention(q_sub, pk[:, g0:g1], pv[:, g0:g1], p, self.sm).reshape(q_sub.shape[0], -1)
        return fn

    def forward(self, x, p, row=None, side=None):
        if self.busy_s:
            time.sleep(self.busy_s)
        for lid in self.ids:
            wq, wk, wv, wo = self.w[lid]
            # input norm (as the model's): keeps K/V far inside fp8's range --
            # an overflow would give NaN, and NaN != NaN makes equality vacuous
            xf = x.float()
            xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
            q, k, v = xn @ wq, xn @ wk, xn @ wv
            w = x.shape[0]
            if row is not None and side is not None and lid in side.layers:
                a = side.attention(row, lid, q, k, v, self.own_fn(lid, p))
            else:
                a = self.own_fn(lid, p)(q.view(w, self.H, self.D), k.view(w, self.Hk, self.D),
                                        v.view(w, self.Hk, self.D), 0, self.Hk)
            x = x + a @ wo
        return x


STAGE_IDS = ((3, 7, 11), (15,), (19,))


def make_stages(cfg, busy=(0.0, 0.0, 0.0)):
    return [Stage(cfg, ids, seed=10 + i, busy_s=busy[i]) for i, ids in enumerate(STAGE_IDS)]


def make_inputs(requests, hidden=48, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {rid: torch.randn(sum(ws), hidden, generator=g).to(torch.bfloat16) for rid, ws in requests}


def reference(cfg, requests, xs):
    st = make_stages(cfg)
    out = {}
    out_finite = True
    for rid, ws in requests:
        p, ys = 0, []
        for w in ws:
            y = xs[rid][p:p + w]
            for s in st:
                y = s.forward(y, p)
            ys.append(y)
            p += w
        out[rid] = torch.cat(ys)
        out_finite = out_finite and bool(torch.isfinite(out[rid].float()).all())
    assert out_finite, "the mini model overflowed (NaN/inf): bit-equality would be vacuous"
    return out


def helper_attend(delay=None):
    def attend(owner, info, q_d, mk, mv, p, w):
        if delay is not None:
            delay(info.layer_id, p)
        return AH.reference_attention(q_d, mk, mv, p, info.sm_scale, info.k_scale, info.v_scale)
    return attend


class Pipeline:
    """PP0 (owner) -> PP1 -> PP2, each stage a thread with blocking frames.
    ``mode``: 'server' = V2 (ServerHelper with its own thread + side stream),
    'naive' = the helper served by the stage's main loop between forwards."""

    def __init__(self, cfg, requests, *, mode="server", busy=(0.0, 0.0, 0.0), delay=None,
                 frame_timeout=T_OUT / 2):
        self.cfg, self.requests, self.mode = cfg, requests, mode
        self.frame_timeout = frame_timeout
        self.net = BlockingNet()
        self.wires = {r: BlockingWire(self.net, r) for r in range(3)}
        self.stages = make_stages(cfg, busy)
        self.rule = AH.SplitRule(cfg.min_w, cfg.cap_tokens, cfg.ab)
        d0 = cfg.of_owner_all(0)
        self.layers = AH.owner_layers(STAGE_IDS[0], d0[0].n_layers)
        self.owner_exec = ThreadExec("owner-side")
        self.side = AH.OwnerSide(cfg, d0, self.wires[0], "cpu", exec_=self.owner_exec,
                                 layers=self.layers, on_veto=self.rule.end_request)
        self.servers, self.execs = {}, [self.owner_exec]
        for d in d0:
            eng = AH.HelperEngine(cfg, "cpu", FP8, helper_attend(delay))
            eng.allocate(0, self.layers, cfg.heads_of(d).n_kv)
            infos = {(0, lid): AH.LayerInfo(lid, self.stages[0].sm, 0.0, None, None) for lid in self.layers}
            ex = ThreadExec(f"h{d.helper}-side") if mode == "server" else AH.InlineExec()
            if mode == "server":
                self.execs.append(ex)
            self.servers[d.helper] = AH.ServerHelper(cfg, self.wires[d.helper], eng, infos,
                                                     {0: self.layers}, exec_=ex)
        self.rows, self.errors, self.out = [], [], defaultdict(list)

    def _guard(self, fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                self.errors.append(exc)
        return run

    def _pp0(self, q_out):
        xs = self.xs
        for rid, ws in self.requests:
            p = 0
            for w in ws:
                row = self.rule.decide(True, [rid], [p], [w])
                self.rows.append(row)
                q_out.put((rid, p, self.stages[0].forward(xs[rid][p:p + w], p, row, self.side)))
                p += w
        q_out.put(None)

    def _mid(self, s, q_in, q_out):
        srv = self.servers.get(s) if self.mode == "naive" else None
        while True:
            try:
                item = q_in.get(timeout=self.frame_timeout)  # the PP proxy recv
            except queue.Empty:
                raise Deadlock(f"PP{s} never got the next proxy frame")
            if item is None:
                q_out.put(None)
                return
            rid, p, y = item
            q_out.put((rid, p, self.stages[s].forward(y, p)))
            if srv is not None:  # NAIVE: serve helper work between own forwards
                while srv.poll_once():
                    pass

    def run(self):
        self.xs = make_inputs(self.requests)
        q01, q12, q2o = queue.Queue(), queue.Queue(), queue.Queue()
        if self.mode == "server":
            for srv in self.servers.values():
                srv.start()
        ths = [threading.Thread(target=self._guard(lambda: self._pp0(q01)), daemon=True),
               threading.Thread(target=self._guard(lambda: self._mid(1, q01, q12)), daemon=True),
               threading.Thread(target=self._guard(lambda: self._mid(2, q12, q2o)), daemon=True)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(T_OUT * 2)
        hung = [t for t in ths if t.is_alive()]
        if not self.errors and not hung:
            while True:
                item = q2o.get_nowait()
                if item is None:
                    break
                self.out[item[0]].append(item[2])
            for ex in self.execs:
                ex.synchronize()
        for srv in self.servers.values():
            srv.stop()
            if srv.failure:
                self.errors.append(RuntimeError(srv.failure))
        for ex in self.execs:
            if ex.failure:
                self.errors.append(RuntimeError(ex.failure))
            ex.close()
        return hung

    def result(self):
        return {rid: torch.cat(ys) for rid, ys in self.out.items()}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestGeometry(unittest.TestCase):
    def test_own_pieces(self):
        R = AH.HeadRange
        # NVFP4 minimal form: heads 18-20 -> PP1, 21-23 -> PP2: owner keeps groups 0-2 in ONE call
        self.assertEqual(AH.own_pieces([R(18, 21, 3, 4), R(21, 24, 3, 4)], 24, 6), (R(0, 18, 0, 3),))
        # only the second half away: owner keeps 18-20 of the shared group -> second call
        self.assertEqual(AH.own_pieces([R(21, 24, 3, 4)], 24, 6), (R(0, 18, 0, 3), R(18, 21, 3, 4)))
        # a middle group away: two whole-group runs
        self.assertEqual(AH.own_pieces([R(6, 12, 1, 2)], 24, 6), (R(0, 6, 0, 1), R(12, 24, 2, 4)))
        # a middle half group: partial piece at both ends of its group
        self.assertEqual(AH.own_pieces([R(8, 10, 1, 2)], 24, 6),
                         (R(0, 6, 0, 1), R(6, 8, 1, 2), R(10, 12, 1, 2), R(12, 24, 2, 4)))
        # V1 suffix: the old own_counts
        self.assertEqual(AH.own_pieces([R(18, 24, 3, 4)], 24, 6), (R(0, 18, 0, 3),))

    def test_v2_spec_and_refusals(self):
        V = lambda s, **kw: AH.validate(AH.parse_spec(s), pp_size=3, num_kv_heads=4, gqa=6, **kw)
        V("0:1:4:h18-20,0:2:4:h21-23")
        V("0:2:4:h21-23")
        V("0:1:4:1")
        V("0:1:4:h18-20,0:2:4:h21-23", owner_attn_layers={0: 12, 1: 2, 2: 2})
        for bad, word in (("0:1:4:h18-20,0:2:3:h21-23", "one n_layers per owner"),
                          ("1:0:1:h18-20,1:2:1:h21-23", "mixes upstream and downstream"),
                          ("0:1:4:h16-20,0:2:4:h21-23", "inside ONE group"),
                          ("0:1:4:h18-20,0:2:4:h20-20", "overlap"),
                          ("0:1:1:h0-17,0:2:1:h18-23", "keep at least one"),
                          ("0:1:4:h18-20,2:1:1:h21-23", "upstream AND a downstream"),
                          ("0:1:4:h18-20,0:1:4:h18-20", "twice")):
            with self.assertRaises(AH.AHSpecError, msg=bad) as cm:
                V(bad)
            self.assertIn(word, str(cm.exception))
        with self.assertRaises(AH.AHSpecError):
            V("0:1:13:h18-20,0:2:13:h21-23", owner_attn_layers={0: 12})


class TestSplitGroupEqualsFullGroup(unittest.TestCase):
    """Owner piece(s) + helper halves (through the fp8 mirror) == the full
    attention of all heads, bit for bit (CPU reference)."""

    def _full_and_split(self, spec, p=40, w=24, scale=None):
        cfg = v2_cfg(spec)
        H, Hk, D = 8, 2, 16
        g = torch.Generator().manual_seed(5)
        q = torch.randn(w, H, D, generator=g).to(torch.bfloat16)
        k_all = torch.randn(p + w, Hk, D, generator=g).to(torch.bfloat16)
        v_all = torch.randn(p + w, Hk, D, generator=g).to(torch.bfloat16)
        pk = AH.quantize_like_pool(k_all, scale, FP8)
        pv = AH.quantize_like_pool(v_all, scale, FP8)
        sm = D ** -0.5
        full = AH.reference_attention(q, pk, pv, p, sm, scale, scale)
        dels = cfg.of_owner_all(0)
        ranges = [cfg.heads_of(d) for d in dels]
        parts = []
        for pc in AH.own_pieces(ranges, H, 4):
            parts.append((pc.q0, AH.reference_attention(q[:, pc.q0:pc.q1].contiguous(), pk[:, pc.g0:pc.g1],
                                                        pv[:, pc.g0:pc.g1], p, sm, scale, scale)))
        for d, hr in zip(dels, ranges):
            eng = AH.HelperEngine(cfg, "cpu", FP8, helper_attend())
            eng.allocate(0, (3,), hr.n_kv)
            mk, mv = eng.mirror[(0, 3)]
            mk[:p].copy_(pk[:p, hr.g0:hr.g1])  # earlier chunks already mirrored
            mv[:p].copy_(pv[:p, hr.g0:hr.g1])
            row = AH.ChunkRow("R", 1, p, w, 1)
            hdr = torch.tensor(AH.header_values(row, 3, hr), dtype=torch.int64)
            pay = AH.pack_payload(hdr, q[:, hr.q0:hr.q1], k_all[p:, hr.g0:hr.g1], v_all[p:, hr.g0:hr.g1])
            out = eng.step(0, AH.LayerInfo(3, sm, 0.0, scale, scale), pay, row, hr)
            self.assertTrue(torch.equal(out[:AH.HDR_BF16].view(torch.int64), hdr))
            parts.append((hr.q0, out[AH.HDR_BF16:].view(w, hr.n_q, D)))
            # the owner's fallback over its own complete pool is the same bytes
            fb = AH.reference_attention(q[:, hr.q0:hr.q1].contiguous(), pk[:, hr.g0:hr.g1], pv[:, hr.g0:hr.g1],
                                        p, sm, scale, scale)
            self.assertTrue(torch.equal(fb, parts[-1][1]))
        parts.sort(key=lambda t: t[0])
        split = torch.cat([t for _, t in parts], dim=1)
        self.assertTrue(torch.equal(full, split), msg=f"max diff {(full.float() - split.float()).abs().max()}")

    def test_half_group_to_two_helpers(self):
        self._full_and_split("0:1:2:h4-5,0:2:2:h6-7")

    def test_half_group_owner_keeps_the_other_half(self):
        self._full_and_split("0:2:2:h6-7")  # owner: group 0 + heads 4-5 of group 1 (second call)

    def test_with_kv_scale(self):
        self._full_and_split("0:1:2:h4-5,0:2:2:h6-7", scale=0.5)


class TestPipelineNoDeadlock(unittest.TestCase):
    REQ = [("A", [64, 64, 48]), ("B", [40, 64])]

    def test_v2_pipeline_completes_bit_equal(self):
        cfg = v2_cfg()
        # PP1/PP2 slower than PP0: their main threads are busy with chunk i-1
        # or blocked on PP0's frame of chunk i while PP0 waits for their O
        pl = Pipeline(cfg, self.REQ, busy=(0.0, 0.05, 0.08))
        hung = pl.run()
        self.assertEqual(hung, [], "pipeline hung")
        self.assertEqual(pl.errors, [])
        ref = reference(cfg, self.REQ, make_inputs(self.REQ))
        got = pl.result()
        for rid in ref:
            self.assertTrue(torch.equal(ref[rid], got[rid]), msg=rid)
        self.assertTrue(all(r is not None for r in pl.rows))
        self.assertEqual(pl.side.stats["fallback"], 0)
        self.assertEqual(pl.side.stats["layers"], 5 * 2)
        for srv in pl.servers.values():  # one wake-up per chunk, the whole chunk enqueued
            self.assertEqual((srv.stats["chunks"], srv.stats["layers"]), (5, 10))
        self.assertTrue(pl.net.idle(), "a message was left on a wire")

    def test_owner_keeps_half_group_second_call(self):
        cfg = v2_cfg("0:2:2:h6-7")
        pl = Pipeline(cfg, self.REQ[:1], busy=(0.0, 0.02, 0.02))
        self.assertEqual(pl.run(), [])
        self.assertEqual(pl.errors, [])
        ref = reference(cfg, self.REQ[:1], make_inputs(self.REQ[:1]))
        self.assertTrue(torch.equal(ref["A"], pl.result()["A"]))
        self.assertEqual(len(pl.side.pieces), 2)

    def test_naive_helper_between_forwards_stalls(self):
        """The danger direction: served by PP1's main loop, PP0 waits for O of
        chunk 0 while PP1 waits for PP0's frame of chunk 0. Without a deadline
        that is a hang (bounded here by the named owner stall)."""
        cfg = v2_cfg(deadline_ms=0.0)
        old = AH.HELPER_STALL_S
        AH.HELPER_STALL_S = 0.5
        try:
            pl = Pipeline(cfg, self.REQ[:1], mode="naive", frame_timeout=2.0)
            pl.run()
        finally:
            AH.HELPER_STALL_S = old
        self.assertTrue(any("owner stall" in str(e) for e in pl.errors), pl.errors)

    def test_naive_with_deadline_degrades_but_never_hangs(self):
        cfg = v2_cfg(deadline_ms=50.0)
        pl = Pipeline(cfg, self.REQ, mode="naive")
        self.assertEqual(pl.run(), [])
        self.assertEqual(pl.errors, [])
        ref = reference(cfg, self.REQ, make_inputs(self.REQ))
        for rid, y in pl.result().items():
            self.assertTrue(torch.equal(ref[rid], y), msg=rid)
        self.assertGreater(pl.side.stats["fallback"], 0)
        # A's chunk 0 always falls back (PP1/PP2 have no frame yet) and ends A's
        # split; B may or may not, depending on when PP1/PP2 poll between frames
        self.assertIn(pl.side.stats["vetoes"], (1, 2))
        self.assertEqual([r is not None for r in pl.rows][:3], [True, False, False])
        self.assertTrue(pl.net.idle())


class TestDeadlineFallback(unittest.TestCase):
    def test_late_helper_owner_computes_ends_request_next_splits(self):
        cfg = v2_cfg(deadline_ms=250.0)
        slow = {"on": True}

        def delay(layer_id, p):
            if slow["on"] and layer_id == 7:  # first delegated layer of request A's chunk 0
                slow["on"] = False
                time.sleep(1.5)

        req = [("A", [64, 64, 48]), ("B", [64, 64])]
        pl = Pipeline(cfg, req, busy=(0.0, 0.01, 0.01), delay=delay)
        self.assertEqual(pl.run(), [])
        self.assertEqual(pl.errors, [])
        ref = reference(cfg, req, make_inputs(req))
        got = pl.result()
        for rid in ref:
            self.assertTrue(torch.equal(ref[rid], got[rid]), msg=rid)
        # A: chunk 0 announced and fell back, chunks 1-2 unsplit; B: split throughout
        self.assertEqual([r is not None for r in pl.rows], [True, False, False, True, True])
        self.assertEqual(pl.side.stats["vetoes"], 1)
        self.assertGreaterEqual(pl.side.stats["fallback"], 1)
        self.assertGreaterEqual(pl.side.stats["late_drained"], 1)
        for srv in pl.servers.values():
            self.assertEqual(srv.stats["chunks"], 3)
        self.assertTrue(pl.net.idle(), "an owed output was never drained")
        for h in (1, 2):
            self.assertEqual(pl.side.late[h], [])
            self.assertEqual(pl.wires[0].published(h), pl.side.need[h])


class TestServerHelperCrashStops(unittest.TestCase):
    def _server(self, cfg=None):
        cfg = cfg or v2_cfg()
        net = BlockingNet(timeout=1.0)
        ow, hw = BlockingWire(net, 0), BlockingWire(net, 1)
        d = cfg.delegation(0, 1)
        eng = AH.HelperEngine(cfg, "cpu", FP8, helper_attend())
        eng.allocate(0, (7, 11), cfg.heads_of(d).n_kv)
        infos = {(0, lid): AH.LayerInfo(lid, 0.25, 0.0, None, None) for lid in (7, 11)}
        srv = AH.ServerHelper(cfg, hw, eng, infos, {0: (7, 11)})
        return cfg, ow, srv, cfg.heads_of(d)

    def _announce(self, ow, hr, row, layer=7):
        w = row.w
        hdr = torch.tensor(AH.header_values(row, layer, hr), dtype=torch.int64)
        z = torch.zeros(w, hr.n_q, 16, dtype=torch.bfloat16)
        zk = torch.zeros(w, hr.n_kv, 16, dtype=torch.bfloat16)
        ow.send(AH.pack_payload(hdr, z, zk, zk), 1)

    def test_continuation_without_start(self):
        cfg, ow, srv, hr = self._server()
        self._announce(ow, hr, AH.ChunkRow("R", 9, 64, 32, 1))
        with self.assertRaises(RuntimeError) as cm:
            srv.poll_once()
        self.assertIn("RANKS DISAGREE", str(cm.exception))
        self.assertIn("never started", str(cm.exception))

    def test_wrong_layer_and_range(self):
        cfg, ow, srv, hr = self._server()
        self._announce(ow, hr, AH.ChunkRow("R", 9, 0, 32, 0), layer=11)
        with self.assertRaises(RuntimeError) as cm:
            srv.poll_once()
        self.assertIn("announcement is layer 7", str(cm.exception))
        cfg, ow, srv, _ = self._server()
        self._announce(ow, AH.HeadRange(6, 8, 1, 2), AH.ChunkRow("R", 9, 0, 32, 0))
        with self.assertRaises(RuntimeError) as cm:
            srv.poll_once()
        self.assertIn("head range", str(cm.exception))

    def test_gap_in_the_request(self):
        cfg, ow, srv, hr = self._server()
        self._announce(ow, hr, AH.ChunkRow("R", 9, 0, 32, 0))
        self._announce(ow, hr, AH.ChunkRow("R", 9, 0, 32, 0), layer=11)
        self.assertEqual(srv.poll_once(), 1)  # serves chunk 0 (both layers)
        for _ in range(2):
            ow.recv(torch.empty(AH.message_bytes(hr.n_q, hr.n_kv, 32, 16)[1] // 2, dtype=torch.bfloat16), 1)
        self._announce(ow, hr, AH.ChunkRow("R", 9, 48, 32, 1))  # p should be 32
        with self.assertRaises(RuntimeError) as cm:
            srv.poll_once()
        self.assertIn("expected 9/32/1", str(cm.exception))

    def test_bad_header_inside_an_announced_chunk(self):
        cfg, ow, srv, hr = self._server()
        self._announce(ow, hr, AH.ChunkRow("R", 9, 0, 32, 0))
        self._announce(ow, hr, AH.ChunkRow("R", 9, 0, 32, 5), layer=11)
        with self.assertRaises(RuntimeError) as cm:
            srv.poll_once()
        self.assertIn("inside an announced chunk", str(cm.exception))

    def test_server_refuses_upstream_delegations(self):
        cfg = AH.AHConfig(AH.parse_spec("1:0:1:1"), 32, 512, 64, 16, 4, 2)
        with self.assertRaises(AH.AHSpecError):
            AH.ServerHelper(cfg, BlockingWire(BlockingNet(), 0), None, {}, {1: (3,)})

    def test_kv_release_forgets_the_request(self):
        cfg, ow, srv, hr = self._server()
        srv.req[0] = (9, 32, 0)
        srv.drop_all()
        self.assertIsNone(srv.req[0])
        rt = AH.AHRuntime(cfg, pp_rank=1, pp_size=3)
        self.assertEqual(rt.role, "server")
        self.assertEqual(AH.AHRuntime(cfg, pp_rank=0, pp_size=3).role, "owner")


class TestABRule(unittest.TestCase):
    def test_every_second_request_is_base(self):
        r = AH.SplitRule(32, 1 << 20, ab="alt")
        arms = []
        for rid in "ABCD":
            row = r.decide(True, [rid], [0], [64])
            arms.append((r.last_start[1], row is not None))
            nxt = r.decide(True, [rid], [64], [64])
            self.assertEqual(nxt is not None, row is not None)
        self.assertEqual(arms, [("split", True), ("base", False), ("split", True), ("base", False)])
        r.reset()
        self.assertEqual(r.n_req, 4)  # the A/B counter survives KV releases

    def test_end_request_stops_the_rest(self):
        r = AH.SplitRule(32, 1 << 20)
        self.assertIsNotNone(r.decide(True, ["A"], [0], [64]))
        r.end_request()
        self.assertIsNone(r.decide(True, ["A"], [64], [64]))
        self.assertIsNotNone(r.decide(True, ["B"], [0], [64]))


class TestPostOnBoth3080(unittest.TestCase):
    SPEC = "0:1:4:h18-20,0:2:4:h21-23"

    def test_mirror_on_both_helpers(self):
        for cap, mirror in ((131072, 256.0), (262144, 512.0)):
            cfg = AH.AHConfig(AH.parse_spec(self.SPEC), 1024, cap, 2048)
            pay, out = AH.message_bytes(3, 1, 2048, 256)
            for s in (1, 2):
                self.assertAlmostEqual(AH.stage_post_mib(cfg, s),
                                       mirror + AH.HELPER_FLOAT_WS_MIB + AH.INT_WS_MIB + (pay + out) / 2**20)
            # owner: one own shape (18 q / 3 kv) + one fallback shape (3 / 1); both messages
            # + one owed late output per helper; q copies of all 24 heads
            owner = AH.INT_WS_MIB * 2 + (2 * (pay + out) + 2 * out + 2048 * 24 * 256 * 2) / 2**20
            self.assertAlmostEqual(AH.stage_post_mib(cfg, 0), owner)

    def test_the_planner_charges_both_3080(self):
        from sglang.srt.planner.pp_cut import PhasePoolModel, _stage_free_after_residency

        cfg = AH.AHConfig(AH.parse_spec(self.SPEC), 1024, 131072, 2048)
        vec = AH.stage_post_vector(cfg, 3)
        self.assertGreater(vec[1], 256.0)
        self.assertEqual(vec[1], vec[2])
        mk = lambda **kw: PhasePoolModel(free_mib=(28000.0, 16000.0, 16000.0), weight_mib_per_layer=400.0,
                                         kv_mib_per_token_per_attn_layer=2048 / 2**20,
                                         arming_floor_mib=(1229.0, 1229.0, 1229.0), **kw)
        counts, attn = (49, 8, 7), (12, 2, 2)
        off = _stage_free_after_residency(counts, attn, mk())
        on = _stage_free_after_residency(counts, attn, mk(attn_head_split_mib=vec))
        for a, b, c in zip(off, on, vec):
            self.assertAlmostEqual(a - b, c)

    def test_p2p_fits_the_largest_message(self):
        cfg = AH.AHConfig(AH.parse_spec(self.SPEC), 1024, 131072, 2048)
        self.assertGreaterEqual(cfg.p2p_bytes(), max(AH.message_bytes(3, 1, 2048, 256)))


class TestEnvAndLauncher(unittest.TestCase):
    def test_env_round_trip_and_v1_bytes(self):
        v1 = AH.AHConfig(AH.parse_spec("2:0:2:1,1:0:1:1"), 1024, 131072, 2048)
        raw = json.loads(v1.to_env())
        self.assertNotIn("deadline_ms", raw)
        self.assertNotIn("ab", raw)
        v2 = AH.AHConfig(AH.parse_spec("0:1:4:h18-20,0:2:4:h21-23"), 1024, 131072, 2048,
                         deadline_ms=12.5, ab="alt")
        self.assertEqual(AH.AHConfig.from_env(v2.to_env()), v2)
        with self.assertRaises(AH.AHSpecError):
            AH.AHConfig.from_env(json.dumps({**raw, "ab": "every"}))

    def _ns(self, *extra):
        from sglang.srt.weg2 import launcher as L

        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump({"text_config": {"num_attention_heads": 24, "num_key_value_heads": 4,
                                       "head_dim": 256, "hidden_size": 5120}}, f)
        return L, L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", tmp] + list(extra))

    def test_launcher_v2(self):
        L, ns = self._ns("--p-attn-head-split", "0:1:4:h18-20,0:2:4:h21-23", "--max-kv-per-request", "131072",
                         "--p-attn-head-split-ab", "alt", "--p-attn-head-split-deadline-ms", "15")
        cfg = L.p_attn_head_split_cfg(ns, 3, 2048, ns.model)
        self.assertEqual((cfg.ab, cfg.deadline_ms, cfg.cap_tokens), ("alt", 15.0, 131072))
        self.assertEqual(AH.config_from_env(L.p_attn_head_split_env(cfg)), cfg)
        line = L.p_attn_head_split_line(cfg, 3)
        self.assertIn("deadline 15.0 ms, ab alt", line)
        self.assertEqual(L._p_ah_post_vector(ns, 3, 2048, ns.model), AH.stage_post_vector(cfg, 3))

    def test_launcher_defaults_match_the_module(self):
        L, ns = self._ns()
        self.assertEqual(ns.p_attn_head_split_deadline_ms, AH.DEADLINE_MS_DEFAULT)
        self.assertEqual(ns.p_attn_head_split_ab, "off")
        self.assertIsNone(L.p_attn_head_split_cfg(ns, 3, 2048, ns.model))
        L, ns = self._ns("--p-attn-head-split", "2:0:2:1", "--p-attn-head-split-deadline-ms", "5")
        with self.assertRaises(SystemExit) as cm:
            L.p_attn_head_split_cfg(ns, 3, 2048, ns.model)
        self.assertIn("downstream", str(cm.exception.code))


if __name__ == "__main__":
    unittest.main()
