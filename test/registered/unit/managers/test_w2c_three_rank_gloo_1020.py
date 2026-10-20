"""W2C C2/C3 on THREE gloo processes (review A1, done/nf-review-p-pp3-overlap-1020.md).

S95b form: spawn 3 ranks, gloo, real isend/recv, the real mixin methods bound to holders. What runs for real:
* the request chain: ``SchedulerPPMixin._pp_send_pyobj_to_next_stage`` (point_to_point_pyobj, async isend, counter bump
  after the post), ``PpChainReceiver`` on the world cpu group, real ``PhaseFlipCounters`` files shared by the ranks,
  the counter gate ``_pp_row_chain_pending``;
* the frame probe ``_pp_proxy_frame_pending`` (with the C3 defer inside) on the follower's proxy inbox;
* PP0's pass top ``pdflip_store_told.pp0_publish`` (V1) and the chain send hook ``pp0_after_chain_send``;
* the frame extras ``proxy_extra`` (PP0 announce / PP1 relay) and the receiver pop ``note_frame``.
Stand-ins (named): the forward is a sleep; S82's park is a poll on the ack/output counters with the receive budget;
frames, acks and outputs ride their own gloo groups behind their own counters (no cross-wire order, as #631 row14).

The PP0 pass order mirrors ``_event_loop_pp_body``: pass s (slot 0) sends chain k-1 at its top, launches the older set,
sends its frame and then receives slot 1's output (empty -> no receive); pass s+1 (slot 1) is the announced pass:
pp0_publish (no park), chain k (the younger told + an abort of a rid of row s, K14-shaped), slot 2's receive (empty);
pass s+2 (slot 2) parks (S82 stand-in) because slot 0's output is owed, admits, launches, then receives slot 0.

Cases: (1) green: chain k absorbed before the older row on PP1 and PP2, the abort of row s's rid absorbed before that
row and the row still runs as stamped, the younger set admitted on ACKS before slot 0's output, no frame computed with
the key; (2) another slot in flight: no announcement, the row runs first, nothing hangs; (3) PP0 never sends chain k:
a NAMED stop on every rank (PP1 '#789 W2C ANNOUNCED-CHAIN TIMEOUT', PP0 and PP2 their readiness budgets) -- no hang;
(4) fault injection for V2: PP1 forwards chain k only after its row (late hop) -- with the relay PP2 still absorbs k
before its row.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

EPOCH = 3
SLOT = 0
F = {0: 0.6, 1: 0.6, 2: 0.4}
BUDGET_S = 3.0


class _Told:
    def __init__(self, rid):
        self.rid = rid


class _Abort:  # K14-shaped: PP0's verdict "a frame I sent names this rid" rides the request
    def __init__(self, rid, framed):
        self.rid, self.k14_framed = rid, framed


def _send_obj(obj, dst, group):
    buf = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
    size = torch.tensor([buf.numel()], dtype=torch.long)
    return [dist.isend(size, dst, group=group), dist.isend(buf, dst, group=group)]


def _recv_obj(src, group):
    size = torch.zeros(1, dtype=torch.long)
    dist.recv(size, src, group=group)
    buf = torch.zeros(int(size.item()), dtype=torch.uint8)
    dist.recv(buf, src, group=group)
    return pickle.loads(bytes(buf.numpy()))


class _Side:
    """One counter-gated side wire (frames / acks / outputs) on its own gloo group."""

    def __init__(self, counters, chan, group):
        self.c, self.chan, self.g, self.got, self.works = counters, chan, group, {}, []

    def send(self, obj, dst):
        self.works += _send_obj(obj, dst, self.g)
        self.c.bump_sent(self.chan)

    def poll(self, src):
        if self.c.sent(self.chan, src) > self.got.get(src, 0):
            self.got[src] = self.got.get(src, 0) + 1
            return _recv_obj(src, self.g)
        return None


def _holder(rank, counters, receiver=None):
    from flliper.srt.managers import scheduler_pp_mixin as ppm

    cls = ppm.SchedulerPPMixin
    ps = types.SimpleNamespace(pp_rank=rank, pp_size=3, tp_size=1, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0,
                               attn_cp_size=1, attn_tp_size=1)
    h = types.SimpleNamespace(ps=ps, pp_flip_counters=counters, pp_chain_receiver=receiver,
                              world_group=types.SimpleNamespace(cpu_group=dist.group.WORLD), pp_group=object(),
                              mbs=[None, None, None], chunked_req=None, running_batch=None, waiting_queue=[],
                              _pp_row_chain_owed=False, _pdflip_told_paced_on=True, pdflip_dormant=False,
                              _pdflip_store_told_armed=False, _pdflip_store_held={}, _pdflip_store_told={})
    h._pp_flip_epoch = lambda: EPOCH
    for name in ("_pp_send_pyobj_to_next_stage", "_pp_flip_bump_sent", "_pp_row_chain_pending", "_pp_flip_upstream",
                 "_pp_flip_ring", "_pp_proxy_frame_pending"):
        setattr(h, name, types.MethodType(getattr(cls, name), h))
    return h


def _wait_all(sends, works):
    for wk in sends:
        for w in wk or ():
            getattr(w, "work", w).wait()
    for w in works:
        w.wait()


def _frame(extra, rows, seq):
    return dict(extra or {}, __stamp__=(SLOT, seq, 2048, EPOCH, seq, ("older", 0, 2048)), rows=rows)


def _rank_main(rank, port, shm, case, q):
    try:
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
        os.environ["FLLIPER_PP_PROXY_READINESS_BUDGET_S"] = str(BUDGET_S)
        dist.init_process_group("gloo", rank=rank, world_size=3)
        g_frm, g_ack, g_out = (dist.new_group([0, 1, 2], backend="gloo") for _ in range(3))
        from flliper.srt.managers import pdflip_store_told as told
        from flliper.srt.managers import scheduler_pp_mixin as ppm
        from flliper.srt.managers.phase_flip_counters import PhaseFlipCounters
        from flliper.srt.managers.pp_chain_receiver import PpChainReceiver
        from flliper.srt.pdflip import pp_chain_announce as w2c

        ctr = PhaseFlipCounters(n_ranks=3, rank=rank, directory=shm, instance="w2c")
        frm, ack, out = _Side(ctr, "frm", g_frm), _Side(ctr, "ack", g_ack), _Side(ctr, "out", g_out)
        res = {"rank": rank}
        t0 = time.monotonic()
        now = lambda: time.monotonic() - t0  # noqa: E731
        dist.barrier()
        if rank == 0:
            _pp0(res, ctr, frm, ack, out, told, w2c, case, now)
        else:
            rcv = PpChainReceiver(group=dist.group.WORLD, src=rank - 1, dst=rank,
                                  on_consumed=lambda _n: ctr.bump_consumed("req"), stall_timeout_s=0)
            h = _holder(rank, ctr, rcv)
            inbox = []
            ppm.typed_inbox = lambda group: {(0, "proxy"): inbox}
            ppm.resolve_src = lambda group, x: 0
            _follower(res, h, rcv, inbox, frm, ack, out, w2c, case, now)
        q.put(res)
    except Exception as exc:  # noqa: BLE001
        q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        q.close()
        q.join_thread()  # flush the result before the hard exit
        os._exit(0)  # a stopped peer may leave a gloo receive posted; the result is already reported


def _pp0(res, ctr, frm, ack, out, told, w2c, case, now):
    h = _holder(0, ctr)
    told.armed = lambda s: True
    told.w2c_publishable = lambda s: ["young"]
    told._twin.tick = lambda s: None
    told._minifwd_hold = lambda s, r: None
    pending = {"items": []}
    told._pp0_publish_paced = lambda s, r: list(r) + pending.pop("items", [])
    acks = {}
    cur = {"pass": "warm"}
    res["park_calls"] = []

    def _s82_park(s, r):                                   # stand-in: first of {acks 1+2, output}, receive budget
        # Parks where the S82 rule parks: the pass whose end receives an owed output (pass s+2 for slot 0). In the
        # announced pass k it would park too IF its receive could block -- the hazard V1 removes; a park call there
        # waits exactly as S82 would (the mutant "park in pass k").
        res["park_calls"].append(cur["pass"])
        if cur["pass"] not in ("k", "s+2"):
            return
        t_end = time.monotonic() + BUDGET_S
        while time.monotonic() < t_end:
            for src in (1, 2):
                a = ack.poll(src)
                if a is not None:
                    acks[src] = a
            if acks.get(1) == "young" and acks.get(2) == "young":
                res["park"] = ("acks", now())
                return
            o = out.poll(2)
            if o is not None:
                res["park"], res["output"] = ("output", now()), (o, now())
                return
            time.sleep(0.002)
        raise RuntimeError("#789 PROXY READINESS TIMEOUT (S82 park + receive budget) on PP0")

    told._commit_park = _s82_park
    sends = []

    def _chain(wire):
        time.sleep(0.05)                                   # PP0's pass-top work before the send
        sends.append(h._pp_send_pyobj_to_next_stage(list(wire), async_send=True))
        w2c.pp0_after_chain_send(h)

    _chain([])                                             # an earlier pass: the counters are not at 0
    # pass s (slot 0): chain k-1, launch the older set, its frame, then slot 1's receive (empty)
    cur["pass"] = "s"
    _chain(told.pp0_publish(h, []))
    h.mbs[SLOT] = "older"
    if case == "other_in_flight":
        h.mbs[2] = "other"
    time.sleep(F[0])
    extra = w2c.proxy_extra(h, SLOT, None)
    res["announced"] = None if not extra else extra.get(w2c.KEY)
    frm.send(_frame(extra, "older", 1), 1)
    assert h.mbs[1] is None                                # slot 1 owes nothing: the receive does not block
    # pass s+1 (slot 1): the announced pass -- V1, no wait before chain k
    if case == "pp0_silent":
        t_end = time.monotonic() + BUDGET_S + 2.0         # PP0 stops posting chains: its own receive is budgeted
        while time.monotonic() < t_end and out.poll(2) is None:
            time.sleep(0.01)
        raise RuntimeError("#789 PROXY READINESS TIMEOUT on PP0 (slot 0's output never posted)")
    pending["items"] = [_Told("young"), _Abort("older", framed=True)]
    cur["pass"] = "k"
    _chain(told.pp0_publish(h, []))
    res["chain_k"] = ctr.local_sent("req")
    # pass s+2 (slot 2): slot 0's output is owed -> the park, then admit + launch the younger set
    cur["pass"] = "s+2"
    _chain(told.pp0_publish(h, []))
    res["admit"] = now()
    time.sleep(F[0])
    if "output" not in res:
        t_end = time.monotonic() + BUDGET_S + 2.0
        while time.monotonic() < t_end:
            o = out.poll(2)
            if o is not None:
                res["output"] = (o, now())
                break
            time.sleep(0.002)
        else:
            raise RuntimeError("#789 PROXY READINESS TIMEOUT on PP0 (slot 0's output never posted)")
    _chain(["stop"])
    _wait_all(sends, frm.works + ack.works + out.works)


def _follower(res, h, rcv, inbox, frm, ack, out, w2c, case, now):
    rank = h.ps.pp_rank
    late_hop = []                                          # case late_hop: PP1 forwards chain k after its row
    done_row, stop = False, False
    sends = []
    t_end = time.monotonic() + 30.0
    t_frame = time.monotonic() + 2 * BUDGET_S + 2.0       # stand-in for this rank's own readiness budget
    while time.monotonic() < t_end and not (done_row and stop):
        if not done_row and time.monotonic() > t_frame:
            raise RuntimeError(f"#789 PROXY READINESS TIMEOUT on PP{rank} (the frame for slot {SLOT} never came)")
        if h._pp_row_chain_pending():
            msg = rcv.recv()
            if rank == 1:
                if case == "late_hop" and any(isinstance(m, _Told) for m in msg) and not done_row:
                    late_hop.append(msg)
                else:
                    sends.append(h._pp_send_pyobj_to_next_stage(msg, async_send=True))
            for m in msg:
                if isinstance(m, _Told):
                    res["told_t"] = now()
                    ack.send(m.rid, 0)
                elif isinstance(m, _Abort):
                    res["abort_t"] = (now(), m.rid, m.k14_framed)
                elif m == "stop":
                    stop = True
            continue
        f = frm.poll(rank - 1)
        if f is not None:
            inbox.append(f)
        if inbox and not done_row and h._pp_proxy_frame_pending(SLOT):
            fr = inbox.pop(0)
            res["frame_key"] = fr.get(w2c.KEY)
            w2c.note_frame(h, SLOT, fr)
            res["key_left"] = w2c.KEY in fr
            res["row"] = (now(), fr["rows"])
            time.sleep(F[rank])
            done_row = True
            if rank == 1:
                for msg in late_hop:
                    sends.append(h._pp_send_pyobj_to_next_stage(msg, async_send=True))
                frm.send(_frame(w2c.proxy_extra(h, SLOT, None), fr["rows"], 1), 2)
            else:
                out.send("older-output", 0)
            continue
        time.sleep(0.002)
    _wait_all(sends, frm.works + ack.works + out.works)


def _run(case):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29900 + (os.getpid() % 1500)
    with tempfile.TemporaryDirectory() as shm:
        procs = [ctx.Process(target=_rank_main, args=(r, port, shm, case, q)) for r in range(3)]
        for p in procs:
            p.start()
        out = {}
        for _ in range(3):
            r = q.get(timeout=60)
            out[r["rank"]] = r
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
    return out


def test_green_chain_k_before_the_older_row_and_admit_on_acks():
    o = _run("green")
    assert all("error" not in o[r] for r in o), o
    assert o[0]["announced"] is not None
    for r in (1, 2):
        assert o[r]["frame_key"] is not None, (r, "the frame carried no announcement (PP2: no relay)")
        assert o[r]["told_t"] < o[r]["row"][0], (r, "chain k absorbed AFTER the older row")
        t_abort, rid, framed = o[r]["abort_t"]
        assert t_abort < o[r]["row"][0] and rid == "older" and framed     # the K14 abort before the row ...
        assert o[r]["row"][1] == "older"                                   # ... and the row still runs as stamped
        assert o[r]["key_left"] is False
    assert o[0]["park"][0] == "acks", o[0]
    assert o[0]["park_calls"] == ["s", "s+2"], o[0]["park_calls"]       # V1: no park call in the announced pass
    assert o[0]["admit"] < o[0]["output"][1] - 0.5, o[0]                  # the younger set beside the older row


def test_other_slot_in_flight_announces_nothing_and_runs_serial():
    o = _run("other_in_flight")
    assert all("error" not in o[r] for r in o), o
    assert o[0]["announced"] is None and o[1]["frame_key"] is None
    assert o[1]["row"][0] < o[1]["told_t"]


def test_pp0_silent_after_announcing_is_a_named_stop_on_every_rank():
    o = _run("pp0_silent")
    assert "#789 W2C ANNOUNCED-CHAIN TIMEOUT" in o[1].get("error", ""), o[1]
    assert "#789" in o[0].get("error", ""), o[0]
    assert "#789" in o[2].get("error", "") and "row" not in o[2], o[2]


def test_late_chain_hop_on_pp1_is_covered_by_the_relay():
    o = _run("late_hop")
    assert all("error" not in o[r] for r in o), o
    assert o[2]["frame_key"] is not None
    assert o[2]["told_t"] < o[2]["row"][0], o[2]
