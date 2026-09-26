"""PF (26.09.): the group told=0 fallback of the paced told
(SGLANG_WEG2_TOLD_GROUP_FALLBACK, weg2_told_fallback).

Danger directions pinned here:
  * a follower slower than the pacing window kills the group (WAIT EXCEEDED)
    or stops its stage in the admission busy-wait -- with the switch on, PP0
    must switch the rid to told=0 for EVERY rank and nobody may wait;
  * ranks disagree (one rank admits told, another 0, or at another pass);
  * a released read leaks its rows / reader references, or a record stays
    behind (``op_refs`` / ``ongoing`` of the ledger tree double);
  * a follower decides on its own;
  * the switch off changes a single byte of the wire, a plan or a log line
    (golden digest taken from the pre-PF head 2bddf0417b).
The last test runs the real ack stream on real gloo in three CPU processes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import socket
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402

#: run_digest over every scenario with the switch OFF, computed on the
#: pre-PF head 2bddf0417b (desk/27b-unified-0926) -- unset and "0" alike.
GOLDEN_OFF = "ec4419848dae7cca41d26e57b02633477827306b111d2d58b08237bbe6d73dfc"

SLOW = "cccc-slow"
PROMPTS = {"aaaa-told": 100_000, "bbbb-fresh": 0, SLOW: 4096}


def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("SGLANG_WEG2_TOLD") or k == "SGLANG_WEG2_P_TWIN_DEFER":
            monkeypatch.delenv(k, raising=False)


def _reads(pp0=0.1, pp1=0.1, pp2=0.1, rid=SLOW):
    base = {"aaaa-told": 0.2, "bbbb-fresh": 0.0, SLOW: 0.1}
    out = {}
    for r, v in ((0, pp0), (1, pp1), (2, pp2)):
        d = dict(base)
        d[rid] = v
        out[r] = d
    return out


def _fb_ring(monkeypatch, read_s, prompts=PROMPTS, on=True, **env):
    _clean_env(monkeypatch)
    if on:
        monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return R.Ring(m, monkeypatch, prompts, read_s)


def _kinds(ring):
    return [
        (type(o).__name__, o.told, getattr(o, "paced", None), getattr(o, fb.WIRE_ACK, None), getattr(o, fb.WIRE_FALLBACK, None))
        for o in ring.wire_objs()
    ]


def _clean_ledgers(ring, rid):
    for s in ring.stages:
        t = s.tree_cache
        assert t.op_refs == 0, (s.ps.pp_rank, t.op_refs)
        assert rid not in t.ongoing
        assert rid not in t.completed and rid not in t.loaded


# ---------------------------------------------------------------------------
# ring: the fallback itself
# ---------------------------------------------------------------------------


def test_slow_follower_gets_told_zero_on_every_rank_nobody_waits(monkeypatch):
    """PP2's read never ends: at the Frist PP0 sends Admit(0, fallback); all
    three admit at the same PP0 pass with cap 0 and credit 0, no stage
    busy-waits, PP2's read is released through the abort path, PP0's own
    record too, and no row / reader reference stays behind."""
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6))
    ring.arrive(SLOW)
    ring.run(120)  # 6 s
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}
    plans = ring.plans(SLOW)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2]
    assert plans[0][0][2] == 0  # the prefix cap: told 0 on every rank
    assert [a[3] for s in ring.stages for a in s.admitted] == [0, 0, 0]
    assert _kinds(ring) == [
        ("Weg2StoreTold", 4096, True, 1, None),
        ("Weg2StoreAdmit", 0, None, None, 1),
    ]
    # the Frist: window (1.25 x PP0's 0.1 s read) + 2 s grace
    pp0_pass = plans[0][0][0]
    assert 2.0 <= pp0_pass * R.DT <= 2.6, pp0_pass
    assert SLOW in ring.stages[2].tree_cache.released  # the slow read was cut
    assert SLOW in ring.stages[0].tree_cache.released  # PP0's record dropped
    _clean_ledgers(ring, SLOW)
    assert ring.stages[0]._pf_fallback_n == 1
    assert fb._pp0_open_map(ring.stages[0]) == {}
    assert m._pacing(ring.stages[0]) == {}


def test_same_slow_follower_without_the_switch_dies_by_name(monkeypatch):
    """Characterisation (switch off): the same geometry is the group death
    this switch exists for -- WAIT EXCEEDED on the slow follower."""
    monkeypatch.setattr(m, "WAIT_CAP_S", 1.0)
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6), on=False)
    ring.arrive(SLOW)
    with pytest.raises(m.Weg2StoreToldMismatch, match="WAIT EXCEEDED"):
        ring.run(120)


def test_fast_followers_admit_told_on_their_acks_before_the_window(monkeypatch):
    """Every follower's read reproduced told: Admit(told) as soon as the acks
    are in -- earlier than the pacing window, which only estimated them."""
    ring = _fb_ring(monkeypatch, _reads(pp0=0.8, pp1=0.1, pp2=0.1, rid="aaaa-told"))
    ring.arrive("aaaa-told")
    ring.run(80)
    plans = ring.plans("aaaa-told")
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2]
    assert plans[0][0][2] == 100_000
    assert _kinds(ring) == [
        ("Weg2StoreTold", 100_000, True, 1, None),
        ("Weg2StoreAdmit", 100_000, None, None, None),
    ]
    told_k = next(k for k in sorted(ring.wire) if ring.wire[k])
    admit_k = max(k for k in ring.wire if ring.wire[k])
    window = m.pace_window_s(0.8, 100_000)  # 1.0 s
    assert (admit_k - told_k) * R.DT < window
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}
    assert ring.stages[0]._pf_admit_acks_n == 1
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert all(not s.tree_cache.released for s in ring.stages)


def test_short_read_on_a_follower_is_told_zero_at_once(monkeypatch):
    """PP2 cannot read at all (declined registration): its ack says own=0,
    PP0 answers told=0 right away (no Frist wait) -- where the switch off
    dies in STORE-TOLD MISMATCH."""
    rs = _reads()
    rs[2][SLOW] = None  # declined:store_absent
    ring = _fb_ring(monkeypatch, rs)
    ring.arrive(SLOW)
    ring.run(60)
    plans = ring.plans(SLOW)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1
    assert plans[0][0][2] == 0
    assert plans[0][0][0] * R.DT < 1.0  # far before the 2.1 s Frist
    assert ring.wire_objs()[-1].told == 0
    assert getattr(ring.wire_objs()[-1], fb.WIRE_FALLBACK) == 1
    _clean_ledgers(ring, SLOW)
    off = _fb_ring(monkeypatch, rs, on=False)
    off.arrive(SLOW)
    with pytest.raises(m.Weg2StoreToldMismatch, match="MISMATCH"):
        off.run(60)


def test_absolute_twin_told_fallback_drops_the_head_on_every_rank(monkeypatch):
    """TK absolute told (head + span): the fallback must also take the
    follower's twin mark -- else its admission adds the head to 0."""
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6), SGLANG_WEG2_TOLD_ABSOLUTE="1")
    for s in ring.stages:
        r = R.req(SLOW)
        r._prefetch_registered_prefix_len = 64
        s.waiting_queue.append(r)
        m.intake(s, r, lambda g: None)
    ring.run(120)
    assert _kinds(ring)[0][1] == 64 + 4096
    plans = ring.plans(SLOW)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1
    assert plans[0][0][2] == 0
    _clean_ledgers(ring, SLOW)
    # and the fast absolute case admits head + span on every rank
    ring2 = _fb_ring(monkeypatch, _reads(), SGLANG_WEG2_TOLD_ABSOLUTE="1")
    for s in ring2.stages:
        r = R.req(SLOW)
        r._prefetch_registered_prefix_len = 64
        s.waiting_queue.append(r)
        m.intake(s, r, lambda g: None)
    ring2.run(60)
    p2 = ring2.plans(SLOW)
    assert p2[0] == p2[1] == p2[2] and p2[0][0][2] == 64 + 4096


def test_request_behind_the_slow_one_is_never_held(monkeypatch):
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6))
    ring.arrive(SLOW)
    ring.run(2)
    ring.arrive("bbbb-fresh")
    ring.run(120)
    b = ring.plans("bbbb-fresh")
    s = ring.plans(SLOW)
    assert b[0] == b[1] == b[2] and s[0] == s[1] == s[2]
    assert b[0][0][0] < s[0][0][0]


def test_abort_inside_the_frist_publishes_nothing_and_forgets(monkeypatch):
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6))
    ring.arrive(SLOW)
    ring.run(8)
    assert SLOW in fb._pp0_open_map(ring.stages[0])
    for s in ring.stages:
        s.waiting_queue.clear()
    ring.run(80)
    assert [k[0] for k in _kinds(ring)] == ["Weg2StoreTold"]
    assert fb._pp0_open_map(ring.stages[0]) == {}
    assert all(not s.admitted for s in ring.stages)


def test_parked_rid_keeps_its_verdict_until_released(monkeypatch):
    """TK path 4 on PP0 (dormant hold): the paced entry is not dropped while
    parked; the verdict follows once the rid is queued again."""
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6))
    ring.arrive(SLOW)
    ring.run(4)
    pp0 = ring.stages[0]
    r0 = pp0.waiting_queue.pop(0)
    pp0.weg2_dormant_hold = [r0]
    ring.run(60)  # past the Frist while parked on PP0
    assert SLOW in m._pacing(pp0)
    pp0.weg2_dormant_hold = []
    pp0.waiting_queue.append(r0)
    ring.run(20)
    plans = ring.plans(SLOW)
    assert all(len(p) == 1 for p in plans) and plans[0][0][2] == plans[1][0][2] == plans[2][0][2] == 0
    assert plans[0] == plans[1] == plans[2]


def test_frist_values(monkeypatch):
    _clean_env(monkeypatch)
    assert fb.frist_s(0.5) == pytest.approx(2.5)
    assert fb.frist_s(10.0) == pytest.approx(12.0)  # window + 2 capped at 12
    assert fb.frist_s(50.0) == pytest.approx(12.0)
    # intake + 16 s bounds it: PP0's own read took 8 s
    assert fb.deadline(100.0, 8.0, 10.0) == pytest.approx(108.0)
    # never before the publication itself
    assert fb.deadline(100.0, 30.0, 1.0) == pytest.approx(100.0)
    monkeypatch.setenv(fb.ENV_GRACE_S, "0.5")
    monkeypatch.setenv(fb.ENV_CAP_S, "3")
    monkeypatch.setenv(fb.ENV_TOTAL_S, "4")
    assert fb.frist_s(1.0) == pytest.approx(1.5)
    assert fb.frist_s(9.0) == pytest.approx(3.0)
    assert fb.deadline(10.0, 2.0, 9.0) == pytest.approx(12.0)
    monkeypatch.setenv(fb.ENV_CAP_S, "junk")
    assert fb.frist_s(20.0) == pytest.approx(fb.CAP_S_DEFAULT)


# ---------------------------------------------------------------------------
# switch discipline
# ---------------------------------------------------------------------------


def test_switch_off_is_byte_identical_to_the_pre_pf_head(monkeypatch):
    assert R.run_digest(m, monkeypatch, {}) == GOLDEN_OFF
    assert R.run_digest(m, monkeypatch, {fb.ENV_FALLBACK: "0"}) == GOLDEN_OFF


def test_switch_off_builds_no_fallback_state_and_no_channel(monkeypatch):
    _clean_env(monkeypatch)
    ring = R.Ring(m, monkeypatch, PROMPTS, _reads(), channel=False)

    def _boom(*a, **k):
        raise AssertionError("ack channel built with the switch off")

    monkeypatch.setattr(fb.GlooAckChannel, "for_scheduler", classmethod(_boom))
    ring.arrive(SLOW)
    ring.arrive("aaaa-told")
    ring.run(80)
    for s in ring.stages:
        assert s._weg2_told_fallback_on is False
        for attr in ("_weg2_fb_open", "_weg2_fb_follower", "_weg2_fb_channel"):
            assert not hasattr(s, attr), (s.ps.pp_rank, attr)
    for o in ring.wire_objs():
        assert fb.WIRE_ACK not in vars(o) and fb.WIRE_FALLBACK not in vars(o)


def _wire_and_plans(ring):
    return (
        [pickle.dumps(ring.wire[k], protocol=4) for k in sorted(ring.wire)],
        [s.admitted for s in ring.stages],
    )


def test_switch_without_paced_changes_nothing(monkeypatch):
    """The fallback needs the paced form: on the single-phase form the
    switch alone leaves wire and plans untouched."""
    runs = []
    for on in (False, True):
        _clean_env(monkeypatch)
        if on:
            monkeypatch.setenv(fb.ENV_FALLBACK, "1")
        ring = R.Ring(m, monkeypatch, PROMPTS, _reads(), paced=False, channel=False)
        ring.arrive("aaaa-told")
        ring.arrive(SLOW)
        ring.run(60)
        assert ring.stages[0]._weg2_told_fallback_on is False
        runs.append(_wire_and_plans(ring))
    assert runs[0] == runs[1]


def test_followers_follow_the_wire_not_their_env(monkeypatch):
    """PP0 paced WITHOUT the fallback, the followers' env has it on: no ack
    is ever sent, no follower builds fallback state -- a follower never
    takes part on its own."""
    ring = _fb_ring(monkeypatch, _reads(), on=True)
    ring.stages[0]._weg2_told_fallback_on = False  # PP0's own resolution
    ring.arrive(SLOW)
    ring.run(60)
    for s in ring.stages[1:]:
        assert not hasattr(s, "_weg2_fb_follower")
        assert s._weg2_fb_channel.sent == 0
    plans = ring.plans(SLOW)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 4096


def test_follower_never_falls_back_alone(monkeypatch):
    """A follower with a slow read and acks armed does not admit at 0 (nor
    at all) until PP0's Admit arrives -- here PP0's Admit is withheld."""
    ring = _fb_ring(monkeypatch, _reads(pp2=10**6))
    ring.arrive(SLOW)
    monkeypatch.setattr(fb, "pp0_decide", lambda *a, **k: None)
    ring.run(200)
    assert all(not s.admitted for s in ring.stages)
    assert SLOW not in ring.stages[2].tree_cache.released


# ---------------------------------------------------------------------------
# ObjectRecvFrame.poll: quiet and non-blocking (no gloo needed for this half)
# ---------------------------------------------------------------------------


def test_frame_poll_reads_a_flag_and_says_nothing(monkeypatch, caplog):
    from sglang.srt.distributed import pp_object_recv as por

    class _W:
        def __init__(self):
            self.go = __import__("threading").Event()

        def wait(self):
            self.go.wait()

    works = []

    def _irecv(t, src, group, tag):
        w = _W()
        works.append((w, t))
        return w

    monkeypatch.setattr(por.dist, "irecv", _irecv)
    frame = por.ObjectRecvFrame(None, 1, fb.WEG2_TOLD_ACK_TAG, "t", "pp_rank=0")
    caplog.set_level("DEBUG")
    t0 = time.perf_counter()
    for _ in range(200):
        assert frame.poll() is False
    assert time.perf_counter() - t0 < 0.5  # 200 polls, no join budget spent
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]
    payload = pickle.dumps(fb.Weg2ToldReadAck(1, 0, [("x", 7)]))
    works[0][1][0] = len(payload)
    works[0][0].go.set()
    deadline = time.perf_counter() + 2
    while not frame.poll() and time.perf_counter() < deadline:
        if len(works) == 2:
            works[1][1][:] = __import__("torch").frombuffer(bytearray(payload), dtype=__import__("torch").uint8)
            works[1][0].go.set()
    obj = frame.take()
    assert obj == fb.Weg2ToldReadAck(1, 0, [("x", 7)])
    assert frame.state == "idle"


def test_channel_errors_never_reach_the_pass(monkeypatch, caplog):
    """A send whose payload post raises after its size went out would
    misframe PP0's stream: the follower's channel goes DEAD (drops every
    later ack, PP0's Frist decides) and nothing raises into the pass; a
    receive error on PP0 is counted, not raised."""
    import torch.distributed as dist

    calls = []

    class _Done:
        def wait(self):
            return None

    def _isend(t, dst, group=None, tag=0):
        calls.append(tag)
        if len(calls) == 2:
            raise RuntimeError("pair closed")
        return _Done()

    monkeypatch.setattr(dist, "isend", _isend)
    ch = fb.GlooAckChannel(None, 2, [0, 1, 2])
    assert ch.send_nowait(fb.Weg2ToldReadAck(2, 0, [("r", 1)])) is True
    assert ch.dead and ch.errors == 1 and calls == [fb.WEG2_TOLD_ACK_TAG] * 2
    assert ch.send_nowait(fb.Weg2ToldReadAck(2, 1, [("s", 1)])) is True
    assert len(calls) == 2  # nothing more on a misframed stream

    def _irecv(*a, **k):
        raise RuntimeError("no group")

    monkeypatch.setattr(dist, "irecv", _irecv)
    pp0 = fb.GlooAckChannel(None, 0, [0, 1, 2])
    assert pp0.harvest() == [] and pp0.errors == 2


# ---------------------------------------------------------------------------
# hermetic: three CPU processes, real gloo, the real ack stream
# ---------------------------------------------------------------------------


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


GLOO_DT = 0.02


def _gloo_rank(rank, port, fallback_on, passes, q):
    try:
        import torch.distributed as dist

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        for k in list(os.environ):
            if k.startswith("SGLANG_WEG2_TOLD"):
                os.environ.pop(k)
        os.environ[m.ENV_PACED] = "1"
        os.environ[fb.ENV_FALLBACK] = "1" if fallback_on else "0"
        os.environ[fb.ENV_GRACE_S] = "0.5"
        os.environ[fb.ENV_CAP_S] = "1.0"
        dist.init_process_group(
            backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=3
        )
        m.WAIT_CAP_S = 1.0
        clock = time.monotonic
        slept = [0.0]
        real_sleep = time.sleep

        def _sleep(sec):
            slept[0] += sec
            real_sleep(sec)

        m.time = SimpleNamespace(monotonic=time.monotonic, sleep=_sleep)
        reads = {"ffff-fast": 0.1, SLOW: 30.0 if rank == 2 else 0.1}
        st = R.Stage(rank, clock, {"ffff-fast": 4096, SLOW: 4096}, reads)
        st.world_group = SimpleNamespace(cpu_group=None)
        st.pp_group = SimpleNamespace(ranks=[0, 1, 2])
        assert m.armed(st)
        for rid in (SLOW, "ffff-fast"):
            r = R.req(rid)
            st.waiting_queue.append(r)
            m.intake(st, r, lambda g: None)
        history = {}
        wire_kinds = []
        publish_ms = []
        errors = []
        for k in range(passes):
            if rank == 0:
                t0 = time.perf_counter()
                wire = m.pp0_publish(st, [])
                publish_ms.append((time.perf_counter() - t0) * 1000.0)
                wire_kinds += [
                    (type(o).__name__, o.rid, o.told, getattr(o, fb.WIRE_FALLBACK, None)) for o in wire
                ]
                box = [pickle.dumps(wire)]
            else:
                box = [None]
            dist.broadcast_object_list(box, src=0)
            history[k] = pickle.loads(box[0])
            if rank > 0 and k - rank >= 0:
                m.follower_absorb(st, list(history[k - rank]))
            for r in list(st.waiting_queue):
                try:
                    credit = m.admission(st, r, lambda kind, rid: None)
                except m.Weg2StoreToldMismatch as exc:
                    errors.append(str(exc)[:60])
                    st.waiting_queue.remove(r)
                    continue
                if credit is None:
                    continue
                st.waiting_queue.remove(r)
                st.admitted.append((k - rank, r.rid, getattr(r, "_weg2_prefix_cap", None), credit))
            real_sleep(GLOO_DT)
        t = st.tree_cache
        q.put((rank, {
            "admitted": st.admitted,
            "op_refs": t.op_refs,
            "ongoing": sorted(t.ongoing),
            "released": list(t.released),
            "slept": slept[0],
            "wire": wire_kinds,
            "publish_ms_max": max(publish_ms) if publish_ms else 0.0,
            "errors": errors,
            "acks_sent": getattr(st, "_pf_ack_sent_n", 0),
            "has_fb_state": any(hasattr(st, a) for a in ("_weg2_fb_open", "_weg2_fb_follower", "_weg2_fb_channel")),
        }))
        # no teardown: PP0's standing receive stays posted by design; a
        # destroy_process_group against it is not what is under test. The
        # queue's feeder thread must flush before the hard exit.
        q.close()
        q.join_thread()
        os._exit(0)
    except Exception:  # noqa: BLE001
        import traceback

        q.put((rank, {"exception": traceback.format_exc()[-1500:]}))
        q.close()
        q.join_thread()
        os._exit(1)


def _run_gloo(fallback_on, passes=150, timeout=180.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_gloo_rank, args=(r, port, fallback_on, passes, q)) for r in range(3)]
    for p in procs:
        p.start()
    out = {}
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and len(out) < 3:
        try:
            rank, res = q.get(timeout=1.0)
        except Exception:  # noqa: BLE001
            if not any(p.is_alive() for p in procs):
                break
            continue
        out[rank] = res
    for p in procs:
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    return out


def test_gloo_three_ranks_slow_follower_told_zero_everywhere():
    out = _run_gloo(fallback_on=True)
    assert sorted(out) == [0, 1, 2], out
    for r in range(3):
        assert "exception" not in out[r], out[r]
    by = {r: {a[1]: a for a in out[r]["admitted"]} for r in range(3)}
    for rid in (SLOW, "ffff-fast"):
        plans = [by[r].get(rid) for r in range(3)]
        assert all(p is not None for p in plans), (rid, plans)
        assert plans[0][0] == plans[1][0] == plans[2][0], (rid, plans)  # same PP0 pass
        assert plans[0][2] == plans[1][2] == plans[2][2], (rid, plans)  # same cap
    assert by[0][SLOW][2] == 0 and by[0][SLOW][3] == 0  # told 0 fallback
    assert by[0]["ffff-fast"][2] == 4096  # admitted on the gloo acks
    kinds = {(w[0], w[1]): w for w in out[0]["wire"]}
    assert kinds[("Weg2StoreAdmit", SLOW)][2:] == (0, 1)
    assert kinds[("Weg2StoreAdmit", "ffff-fast")][2:] == (4096, None)
    for r in range(3):
        assert out[r]["op_refs"] == 0 and out[r]["ongoing"] == [], out[r]
        assert out[r]["slept"] == 0.0, out[r]  # nobody waited in admission
        assert out[r]["errors"] == []
    assert SLOW in out[2]["released"] and SLOW in out[0]["released"]
    assert out[1]["acks_sent"] >= 1 and out[2]["acks_sent"] >= 1
    # the harvest never blocks PP0's pass (a joined receive would cost the
    # whole idle wait per pass)
    assert out[0]["publish_ms_max"] < 250.0, out[0]["publish_ms_max"]


def test_gloo_three_ranks_switch_off_is_the_group_death():
    """Same geometry, switch off: the slow follower's admission ends in the
    named WAIT EXCEEDED (the group death), and no rank builds fallback
    state -- nothing rides the ack stream."""
    out = _run_gloo(fallback_on=False, passes=120)
    assert sorted(out) == [0, 1, 2], out
    for r in range(3):
        assert "exception" not in out[r], out[r]
        assert out[r]["has_fb_state"] is False
    assert any("WAIT EXCEEDED" in e for e in out[2]["errors"]), out[2]
