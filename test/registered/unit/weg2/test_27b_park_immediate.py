"""27B PARK (user decision 26.09. ~19:00Z, memory d2p-sofort-flippen-und-x-exakt-0926):
D->P waits for nothing. A queued request whose pending tokens exceed X while D
decodes parks D's running decodes AT ONCE (the H91b flip park, reused as it is)
and the front flips to P; the parked requests resume first after the flip back,
together, like continuous batching.

What is new for the 27B and pinned here:

* registry -- ``ModelProfile.d_park_immediate`` (qwen27b off until measured,
  nextflash off), switch ``SGLANG_WEG2_D_PARK_IMMEDIATE`` (explicit wins);
* D -- ``d_seats.d_flip_park_active``: the FLIP park only (park_running, late
  hold, sleep hold, awake re-queue, admission barrier). The pressure park, the
  youngest-first retraction order and the MTP draft carry stay on
  ``d_park_active`` (standard form): off for 27B, byte-identical;
* front -- the immediate trigger (``phase_policy.needs_p`` over the queue,
  behind min-dwell and the fairness floor) fires the existing
  ``_wait_bound_park`` with reason ``immediate-over-x``;
* numerics -- park -> flip -> flip back: both requests emit exactly the
  tokens of the run without a park (deterministic test double, incl. a
  DFlash2-shaped draft whose window reads HOLES after the resume: the target
  verifies, so the tokens are equal and only the acceptance drops);
* ranks -- every D rank parks the same requests in the same order and
  re-queues on the same pass (replicated inputs, group-MIN clock verdict);
* switch off -- the 27B front sends no park and flips only after D's decode
  ended (today's behaviour); the D park refuses; admission is the stock loop.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMM = "SGLANG_WEG2_D_PARK_IMMEDIATE"
SWITCHES = (IMM, "SGLANG_WEG2_STANDARD_FORM", "SGLANG_WEG2_D_PARK",
            "SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV", "SGLANG_WEG2_X_IDLE_REGRANT")


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in SWITCHES + (FM.FORM_ENV, "SGLANG_WEG2_GROUP", ds.RESUME_MARGIN_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def d27(clean):
    """Group D of a qwen27b boot with the immediate park switched on."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    clean.setenv("SGLANG_WEG2_GROUP", "D")
    clean.setenv(IMM, "1")
    return clean


# ------------------------------------------------------------------ registry
def test_the_registry_rows_are_off_until_measured():
    assert FM.PROFILES["qwen27b"].d_park_immediate is False
    assert FM.PROFILES["nextflash"].d_park_immediate is False
    for prof in ("qwen27b", "nextflash"):
        assert FM.PROFILE_SWITCH_DEFAULTS[prof][IMM] is False


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, False), ("nextflash", None, False), (None, None, False),
    ("qwen27b", "1", True), ("qwen27b", "on", True), ("qwen27b", "0", False), (None, "1", True)])
def test_the_switch_follows_the_profile_and_an_explicit_value_wins(clean, profile, explicit, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    if explicit is not None:
        clean.setenv(IMM, explicit)
    on, src = FM.d_park_immediate_state()
    assert on is want
    assert src.startswith("env ") if explicit is not None else not src.startswith("env ")
    if explicit not in ("on",):  # EnvBool reads only true/1/yes/y; front + D use the form reader
        assert envs.SGLANG_WEG2_D_PARK_IMMEDIATE.get() is want


@pytest.mark.parametrize("profile,imm,park,group,flip,full", [
    ("qwen27b", None, None, "D", False, False),     # 27B today: no park at all
    ("qwen27b", "1", None, "D", True, False),       # 27B park: flip park ONLY
    ("qwen27b", "1", None, "P", False, False),      # group P never parks
    ("qwen27b", "1", "0", "D", False, False),       # explicit D_PARK=0 = no park at all
    ("qwen27b", "1", "1", "D", True, True),         # explicit D_PARK=1 = the full H91b park
    ("nextflash", None, None, "D", True, True),     # NF standard form unchanged
    ("nextflash", "0", None, "D", True, True),      # the NF park does not hang on this switch
])
def test_the_flip_park_gate_matrix(clean, profile, imm, park, group, flip, full):
    env = {FM.FORM_ENV: _form_env(profile), "SGLANG_WEG2_GROUP": group}
    if imm is not None:
        env[IMM] = imm
    if park is not None:
        env[ds.PARK_ENV] = park
    assert ds.d_flip_park_active(env) is flip
    assert ds.d_park_active(env) is full


# ------------------------------------------------------------------ D doubles
def _req(rid, seq, n_in=10, n_out=3):
    return types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * n_in, output_ids=[0] * n_out,
        is_fast_lane=False, spill_class=None,
    )


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.retract_calls = []
        self.on_retract = None

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        self.retract_calls.append((offload_kv, retain))
        out, self.reqs = self.reqs, []
        if self.on_retract is not None:
            self.on_retract(out, retain)
        return out


class _Sched:
    def __init__(self, running=(), waiting=()):
        self.running_batch = _Batch(running)
        self.waiting_queue = list(waiting)
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace()
        self.weg2_dormant = False
        self.noted = []
        self.sent = []
        self.enable_hicache_storage = False
        self.draft_worker = None
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o)))
        self.min_flags = None  # a group MIN over the ranks, injected by the rank test

    def _969ad_note_retract(self, req, site):
        self.noted.append((req.rid, site))

    def _add_request_to_queue(self, req, is_retracted=False):
        if self.weg2_dormant:
            hold = getattr(self, "weg2_dormant_hold", None)
            if hold is None:
                hold = self.weg2_dormant_hold = []
            hold.append(req)
        else:
            self.waiting_queue.append(req)

    def _weg2_group_min_flags(self, flags):
        if self.min_flags is not None:
            return self.min_flags(flags)
        return [1 if f else 0 for f in flags]

    def uniform_min_avail(self):
        return 0


def _park(s, epoch=7, reason="immediate-over-x"):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason=reason),
                           late_hold_armed=True)


def test_27b_park_running_parks_retaining_oldest_first(d27):
    old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
    s = _Sched(running=[young, old], waiting=[new])
    out = _park(s)
    assert out.success and out.parked == ["old", "young"] and out.held == ["new"]
    assert out.late_hold is True
    assert s.running_batch.retract_calls == [(False, True)]  # retain, no D2H copy
    assert all(getattr(r, FORCE_HOST_WRITE_THROUGH_ATTR) for r in (old, young))
    assert ds.park_site(old) == ds.SITE_FLIP
    # the sleep holds them first, the wake releases them first
    s.weg2_dormant = True
    s.weg2_dormant_hold = [_req("arrived-while-dormant", 9)]
    assert rt.hold_parked(s, hold_armed=True) == 3
    assert [r.rid for r in s.weg2_dormant_hold] == ["old", "young", "new", "arrived-while-dormant"]


def test_27b_park_is_flip_only_no_pressure_park_no_youngest_order(d27):
    """Under the immediate park alone a decode-pressure retraction stays the
    27B's stock retain-and-requeue: no pressure mark, no reorder."""
    young, other = _req("young", 2), _req("other", 7)
    s = _Sched(running=[_req("old", 1)], waiting=[other, young])
    assert rt.note_retracted(s, [young]) == 0
    assert ds.park_site(young) is None and [r.rid for r in s.waiting_queue] == ["other", "young"]
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    sa = types.SimpleNamespace(retraction_policy="fcfs", schedule_low_priority_values_first=False)
    old = types.SimpleNamespace(rid="old", kv_arrival_seq=1, output_ids=[0], origin_input_ids=[0] * 5,
                                is_fast_lane=False, spill_class=None, priority=None)
    yng = types.SimpleNamespace(rid="young", kv_arrival_seq=2, output_ids=[0] * 50,
                                origin_input_ids=[0] * 5, is_fast_lane=False, spill_class=None,
                                priority=None)
    order = ScheduleBatch._get_decode_retraction_order([yng, old], sa, allow_policy_sort=True)
    assert [yng, old][order[-1]] is old  # the stock key (fewest outputs), not the H91b youngest


def test_27b_admission_is_the_stock_loop_until_something_is_flip_parked(d27):
    s = _Sched(running=[], waiting=[_req("a", 1), _req("b", 2)])
    q = s.waiting_queue
    assert rt.admission(s, s.running_batch) is None
    assert s.waiting_queue is q  # not even re-listed
    parked = _req("p", 0)
    ds.mark_parked(parked, ds.SITE_FLIP)
    s.waiting_queue = [q[0], parked, q[1]]
    gate = rt.admission(s, s.running_batch)
    assert gate is not None and gate.barrier
    assert [r.rid for r in s.waiting_queue] == ["p", "a", "b"]
    assert gate.skip(parked) is None and gate.skip(q[0]) == "weg2_d_park_first"


@pytest.mark.parametrize("imm", [None, "0"])
def test_switch_off_27b_d_is_unchanged(clean, imm):
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    clean.setenv("SGLANG_WEG2_GROUP", "D")
    if imm is not None:
        clean.setenv(IMM, imm)
    s = _Sched(running=[_req("a", 1)], waiting=[_req("b", 2)])
    out = _park(s)
    assert not out.success and out.parked == [] and s.running_batch.retract_calls == []
    q = s.waiting_queue
    assert rt.admission(s, s.running_batch) is None and s.waiting_queue is q
    assert rt.hold_late_arrival(s, _req("late", 5)) is False


# ------------------------------------------------ ranks never disagree
def test_every_rank_parks_the_same_requests_in_the_same_order(d27):
    """The park is ONE broadcast control request; every input it reads is
    replicated (running batch, queue, kv_arrival_seq). Rank-local clocks
    differ -- the awake re-queue is decided by a group MIN, so all ranks
    re-queue on the same pass."""
    ranks = []
    for r in range(3):
        reqs = [_req("r2", 2), _req("r0", 0), _req("r1", 1)]
        s = _Sched(running=reqs[:2], waiting=reqs[2:])
        ranks.append(s)
    outs = [_park(s) for s in ranks]
    assert len({(tuple(o.parked), tuple(o.held), o.late_hold) for o in outs}) == 1
    assert [[r.rid for r in s.weg2_d_parked] for s in ranks] == [["r0", "r2", "r1"]] * 3

    # rank 1's monotonic clock says "due", ranks 0/2 not yet -> nobody moves
    def group_min(local_by_rank):
        def _mn(flags):
            return [min(local_by_rank)]
        return _mn

    for i, s in enumerate(ranks):
        for req in s.weg2_d_parked:
            setattr(req, ds.SINCE_ATTR, 0.0 if i == 1 else 1e18)
    locals_ = [0, 1, 0]
    for s in ranks:
        s.min_flags = group_min(locals_)
    assert [rt.park_tick(s) for s in ranks] == [0, 0, 0]
    locals_[:] = [1, 1, 1]
    assert [rt.park_tick(s) for s in ranks] == [3, 3, 3]
    assert len({tuple(r.rid for r in s.waiting_queue) for s in ranks}) == 1


# ------------------------------------------------ numerics: park == no park
V = 97          # vocabulary of the toy target
WINDOW = 8      # DFlash-shaped draft window (rows of the prefix it attends)
BLOCK = 3       # draft tokens per round


def _target_next(ctx):
    """Deterministic 'target': the next token is a hash of the WHOLE context
    (so any lost, duplicated or reordered token changes every later one)."""
    h = hashlib.blake2b(bytes(t % 256 for t in ctx) + len(ctx).to_bytes(4, "little"),
                        digest_size=2).digest()
    return int.from_bytes(h, "little") % V


class _ToyD:
    """A D group of the test double: requests decode speculatively (a
    DFlash-shaped draft over a window of per-slot draft rows, greedy verify
    by the target), a device tree keeps retained spans, a host store is the
    only thing that survives D's sleep. The park and its bookkeeping are the
    REAL d_park_runtime functions on a scheduler double."""

    def __init__(self):
        self.device = {}   # rid -> committed context (retained span on the device)
        self.host = {}     # rid -> committed context (write-through copy)
        self.draft_rows = {}  # rid -> {pos: token} the drafter wrote (the window pool)
        self.accepted = 0
        self.proposed = 0

    def admit(self, req):
        """Resume/admission: prefix match on the device tree, else the host
        (load-back); the DFlash window rows of a prefix that came back through
        the host are HOLES (P/park carry no draft rows)."""
        ctx = self.device.get(req.rid) or self.host.get(req.rid)
        if ctx is not None:
            assert ctx == list(req.origin_input_ids) + list(req.output_ids), "resume lost its position"
            if req.rid not in self.device:
                self.draft_rows[req.rid] = {}  # holes: nothing wrote them here
        else:
            self.draft_rows[req.rid] = {}
        self.device[req.rid] = list(req.origin_input_ids) + list(req.output_ids)

    def _draft(self, req, ctx):
        rows = self.draft_rows.setdefault(req.rid, {})
        win = [rows.get(i, 0) for i in range(max(0, len(ctx) - WINDOW), len(ctx))]  # hole = 0
        guess, out = list(ctx), []
        for k in range(BLOCK):
            # a drafter that is RIGHT where its window rows are real
            if all(w != 0 for w in win) and len(win) == min(WINDOW, len(ctx)):
                t = _target_next(guess)
            else:
                t = (sum(win) + k + 1) % V  # garbage over holes
            out.append(t)
            guess.append(t)
        return out

    def step(self, req, max_new):
        ctx = list(req.origin_input_ids) + list(req.output_ids)
        props = self._draft(req, ctx)
        rows = self.draft_rows[req.rid]
        emitted = []
        for t in props:  # greedy verify: the target decides every token
            want = _target_next(ctx + emitted)
            self.proposed += 1
            if t != want:
                emitted.append(want)
                break
            self.accepted += 1
            emitted.append(t)
        else:
            emitted.append(_target_next(ctx + emitted))
        emitted = emitted[: max(0, max_new - len(req.output_ids))]
        for i, t in enumerate(emitted):
            rows[len(ctx) + i] = 1 + t  # the drafter writes the rows of what it verified
            req.output_ids.append(t)
        self.device[req.rid] = list(req.origin_input_ids) + list(req.output_ids)

    def sleep(self):
        """D's sleep: the tree is flushed, the draft pool re-backed -- only
        the host copies survive."""
        self.device.clear()
        self.draft_rows.clear()


def _run(park_after):
    """Two requests decode together (bs2); after ``park_after`` rounds the
    front parks them (immediate park), D sleeps through a P phase, wakes,
    and the two resume together. ``park_after=None``: no park."""
    toy = _ToyD()
    a = _req("A", 1, n_in=0, n_out=0)
    b = _req("B", 2, n_in=0, n_out=0)
    a.origin_input_ids = [5, 17, 33, 2]
    b.origin_input_ids = [9, 9, 41, 7, 60, 1]
    target = {"A": 40, "B": 33}
    s = _Sched(running=[a, b])

    def _retract(out, retain):
        assert retain
        for r in out:
            assert getattr(r, FORCE_HOST_WRITE_THROUGH_ATTR)
            toy.host[r.rid] = list(toy.device[r.rid])  # forced write-through

    s.running_batch.on_retract = _retract
    for r in (a, b):
        toy.admit(r)
    rounds = 0
    while any(len(r.output_ids) < target[r.rid] for r in (a, b)):
        if park_after is not None and rounds == park_after:
            out = _park(s)
            assert out.success and out.parked == ["A", "B"]
            s.weg2_dormant = True
            toy.sleep()
            assert rt.hold_parked(s, hold_armed=True) == 2
            # ... the P phase prefills the long request here ...
            s.weg2_dormant = False
            released, s.weg2_dormant_hold = s.weg2_dormant_hold, []
            for r in released:
                s._add_request_to_queue(r, is_retracted=True)
            gate = rt.admission(s, s.running_batch)
            assert gate is not None and gate.barrier  # nobody overtakes the parked pair
            resumed = [r for r in s.waiting_queue if gate.skip(r) is None]
            assert [r.rid for r in resumed] == ["A", "B"]  # together, oldest first
            s.waiting_queue = [r for r in s.waiting_queue if r not in resumed]
            for r in resumed:
                toy.admit(r)
            s.running_batch.reqs = resumed
            park_after = None
        for r in list(s.running_batch.reqs):
            if len(r.output_ids) < target[r.rid]:
                toy.step(r, target[r.rid])
        rounds += 1
    return {"A": list(a.output_ids), "B": list(b.output_ids)}, toy


@pytest.mark.parametrize("park_after", [1, 3, 6])
def test_park_flip_flipback_both_requests_emit_the_tokens_of_the_unparked_run(d27, park_after):
    ref, toy_ref = _run(None)
    got, toy = _run(park_after)
    assert got == ref
    # the DFlash-shaped window read holes after the resume: same tokens, less acceptance
    assert toy.accepted / toy.proposed <= toy_ref.accepted / toy_ref.proposed


# ------------------------------------------------ front: the trigger (pure)
def _p(**kw):
    base = dict(rid="r", est_uncached=0, skip_leg1=False, leg1_done=False, p_only=False, x_requeues=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_needs_p_is_exactly_over_x_or_p_only():
    X = 12288
    assert not pp.needs_p(X, X)                         # AT X: D may prefill it (law 4)
    assert pp.needs_p(X + 1, X)
    assert pp.needs_p(10, X, p_only=True)               # image under transient: P by rule
    assert pp.needs_p(10, X, x_requeues=1)              # D refused it as over X (W31)
    assert not pp.needs_p(10 * X, X, leg1_done=True)    # already prefilled: waits for D
    assert not pp.needs_p(10 * X, X, skip_leg1=True)    # CARRIER-EXCEEDS: D prefills once
    q = [_p(rid="short", est_uncached=100), _p(rid="band", est_uncached=X), _p(rid="long", est_uncached=X + 5)]
    assert pp.immediate_park_trigger(q, X).rid == "long"
    assert pp.immediate_park_trigger(q[:2], X) is None
    assert pp.park_body(3, 0.0, reason=pp.PARK_REASON_IMMEDIATE) == {"epoch": 3, "reason": "immediate-over-x"}
    assert pp.park_body(3, 60.0) == {"epoch": 3, "reason": "wait-bound-60s"}  # H91c unchanged


def test_the_dwell_gate_keeps_the_fairness_floor_and_k7():
    assert not pp.immediate_park_dwell_ok(1.9, 0.0, 2000.0)
    assert pp.immediate_park_dwell_ok(2.0, 0.0, 2000.0)
    assert not pp.immediate_park_dwell_ok(2.5, 2700.0, 2000.0)
    assert pp.immediate_park_dwell_ok(2.7, 2700.0, 2000.0)


# ------------------------------------------------ front: end to end (aiohttp doubles)
def _h91c():
    spec = importlib.util.spec_from_file_location(
        "_park27_h91c", os.path.join(HERE, "test_weg2_phase_policy_h91c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_27b_front_parks_at_once_and_resumes_first(clean):
    """qwen27b form (standard form off), immediate park on: r1 (> X) arrives
    while D decodes r0 -> park after the 2 s floor, NOT after r0 ends; P
    prefills r1; D wakes and resumes r0 with r1; r0 is never re-posted."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    H = _h91c()

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2, d_park_immediate=True) as h:
            assert h.front.standard_form is False and h.front.d_wait_bound_s == 0
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running, 20)
            rid0 = next(iter(h.d.running))
            epoch_d = h.front.epoch
            t_arr = asyncio.get_event_loop().time()
            t1 = h.post("r1")
            assert await H._until(lambda: "gen:r1" in h.p.timeline, 20)
            took = asyncio.get_event_loop().time() - t_arr
            assert h.d.park_bodies == [{"epoch": epoch_d, "reason": "immediate-over-x"}]
            # B4 (review RV): the FIRST release in D's timeline may belong to an
            # EARLIER flip -- the harness front starts awake=P with idle_layout D,
            # so an idle P->D can land before r0 is posted and r0 (LONG) then pays a
            # D->P flip of its own (release #1) before D ever decodes it. What must
            # hold is the order around r0's decode: no D sleep between D starting
            # r0 and the park, and D's next sleep after the park.
            tl = h.d.timeline
            i_gen = H._first(tl, "gen:r0")
            i_park = H._first(tl, "rpc:weg2/park_running")
            rel = [i for i, e in enumerate(tl) if e == "rpc:release_memory_occupation"]
            assert i_gen < i_park, tl
            assert not [i for i in rel if i_gen < i < i_park], tl   # never slept under r0 unparked
            assert [i for i in rel if i > i_park], tl                # the park's own sleep follows
            assert not t0.done() and h.p.gen_marks.count("r0") == 1
            assert h.front.counters["park_immediate_fired"] == 1
            assert h.front.counters["wait_bound_fired"] == 0
            assert took < 10.0, took  # floor 2 s + flip; r0 still decodes (held)
            assert await H._until(lambda: h.front.awake == "D" and not h.front._d_parked, 20)
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert "handoff_n" not in kv[-1] and "parked_n" not in kv[-1]  # 27B wake body unchanged
            assert h.front.counters["d_parked_resumed"] == 1
            assert rid0 in h.front.groups["D"].outstanding
            h.d.release_all()
            (s0, _), (s1, _) = await asyncio.wait_for(asyncio.gather(t0, t1), 20)
            assert (s0, s1) == (200, 200)
            assert h.d.gen_marks.count("r0") == 1

    asyncio.run(body())


def test_27b_front_switch_off_waits_for_d_as_today(clean):
    """Same scenario, switch off (the qwen27b row): no park RPC, no flip while
    r0 decodes; the flip comes after r0 ended (fairness disabled here)."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    H = _h91c()

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2) as h:
            assert h.front.d_park_immediate is False
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running, 20)
            t1 = h.post("r1")
            await asyncio.sleep(3.0)  # past the immediate park's floor
            assert h.d.park_bodies == [] and h.front.awake == "D"
            assert "rpc:release_memory_occupation" not in h.d.timeline
            assert h.front.counters.get("park_immediate_fired", 0) == 0
            h.d.release("r0")
            (s0, _) = await asyncio.wait_for(t0, 20)
            assert await H._until(lambda: "gen:r1" in h.p.timeline, 20)
            h.d.release_all()
            (s1, _) = await asyncio.wait_for(t1, 20)
            assert (s0, s1) == (200, 200)

    asyncio.run(body())


def test_main_resolves_the_flag_from_the_switch(clean):
    import sys

    from sglang.srt.weg2 import front as front_mod

    src = open(front_mod.__file__).read()
    assert '"--d-park-immediate", choices=("on", "off"), default=None' in src
    assert "_imm, _imm_src = d_park_immediate_state()" in src
    assert 'd_park_immediate=args.d_park_immediate == "on")' in src
    assert sys.modules.get("sglang.srt.weg2.front") is front_mod


# ------------------------------------------------ DFlash2 draft state at the park
def _draft_worker(mapped: bool):
    pool = types.SimpleNamespace(weg2_slot_mapper=object() if mapped else None)
    runner = types.SimpleNamespace(token_to_kv_pool=pool)
    return types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner))


@pytest.mark.parametrize("mapped,want", [(True, False), (False, True)])
def test_the_park_rearms_a_dflash_window_resume_like_a_fresh_hand_off(d27, mapped, want):
    """DFlash window pool (slot-mapped): the resumed request's draft admission
    is evaluated again (COLD_ARMED cleared -> WEG2 DRAFT-COLD names the
    restored prefix, like any hand-off from P). An MTP pool (target-slot
    indexed, NF H91d carry) keeps the mark -- NF unchanged."""
    from sglang.srt.managers.phase_flip_draft_bootstrap import COLD_ARMED_ATTR

    a, b = _req("a", 1), _req("b", 2)
    for r in (a, b):
        setattr(r, COLD_ARMED_ATTR, True)
    s = _Sched(running=[a, b])
    s.draft_worker = _draft_worker(mapped)
    out = _park(s)
    assert out.success
    assert getattr(a, COLD_ARMED_ATTR) is want and getattr(b, COLD_ARMED_ATTR) is want
    assert rt.rearm_window_draft_cold(_Sched(), [a]) == 0  # no drafter: nothing to do


def test_a_rearmed_dflash_resume_is_judged_cold_only_when_it_came_back_through_the_host(d27, monkeypatch):
    """The admission then treats the resume exactly like a hand-off: a prefix
    restored through the host tier is cold by name (the window reads holes),
    a device hit (an awake re-queue, the span never left the card) stays warm."""
    from sglang.srt.managers import phase_flip_draft_bootstrap as B

    monkeypatch.setattr(B, "_draft_tier_off_by_switch", lambda: True)
    req = types.SimpleNamespace(rid="a", prefix_indices=list(range(100)), host_hit_length=100)
    monkeypatch.setattr(B, "prefix_len", lambda r: 100)
    assert "restored target-only" in (B.draft_cold_reason(None, req, tier_armed=False) or "")
    req.host_hit_length = 0
    assert B.draft_cold_reason(None, req, tier_armed=False) is None
