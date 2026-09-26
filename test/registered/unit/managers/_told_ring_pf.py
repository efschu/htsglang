"""Shared PP3 ring double for the PF told-fallback tests (not collected).

A rank's pass ``k`` plans PP0's pass ``k - pp_rank`` (the carrierless plan
lag); PP0 publishes at the top of its pass, the followers absorb PP0's wire
one pass per hop later, every rank runs its admission over its own queue in
every pass. The tree double keeps a LEDGER of the rows (and with them the
arena reader references) an in-flight store read holds: a read that is
neither completed into the host tree nor released through the abort path
leaves ``op_refs > 0`` -- the #718-stray / release-table row 30 shape.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from types import SimpleNamespace

PAGE = 64
DT = 0.05  # one pass of wall time


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class LedgerTree:
    """Store reads that terminate at a wall-clock time, with a row ledger."""

    def __init__(self, clock):
        self.clock = clock
        self.ongoing = {}  # rid -> (done_at, tokens, rows)
        # the real tree's record (#1175), which PP0's told clamp rewrites
        self.completed = self._prefetch_completed_tokens = {}
        self.loaded = {}
        self.op_refs = 0  # rows held by in-flight reads (arena reader refs)
        self.released = []  # rids released through release_aborted_request
        self.is_eagle = False

    # -- the calls the told module makes ---------------------------------
    def start_read(self, rid, tokens, duration):
        rows = -(-int(tokens) // PAGE)
        self.ongoing[rid] = (self.clock() + duration, int(tokens), rows)
        self.op_refs += rows

    def check_prefetch_progress(self, rid):
        op = self.ongoing.get(rid)
        if op is None:
            return True
        done_at, tokens, rows = op
        if self.clock() < done_at:
            return False
        del self.ongoing[rid]
        self.op_refs -= rows  # adopted by the host tree: the tree owns them
        self.completed[rid] = tokens
        self.loaded[rid] = tokens
        return True

    def completed_prefetch_tokens(self, rid):
        return self.completed.get(rid)

    @property
    def prefetch_loaded_tokens_by_reqid(self):
        return self.loaded

    def pop_prefetch_loaded_tokens(self, rid):
        return self.loaded.pop(rid, 0)

    # -- the upstream abort path (unified_radix_cache.release_aborted_request)
    def release_aborted_request(self, rid):
        self.released.append(rid)
        self.loaded.pop(rid, None)
        self.completed.pop(rid, None)
        op = self.ongoing.pop(rid, None)
        if op is not None:
            self.op_refs -= op[2]  # rows back through the release queue

    def prefetch_progress_is_collective_free(self):
        return True


class Stage:
    def __init__(self, pp_rank, clock, prompts, read_s, pp_size=3):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1)
        self.enable_hicache_storage = True
        self.pp_flip_counters = None
        self.tree_cache = LedgerTree(clock)
        self.waiting_queue = []
        self.prompts = prompts  # rid -> store-hit tokens
        self.read_s = read_s  # rid -> this rank's read duration
        self.admitted = []  # (planned PP0 pass, rid, told cap, credit)

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        hit = self.prompts.get(req.rid, 0)
        if limit_tokens is not None:
            hit = min(hit, int(limit_tokens))
        dur = self.read_s.get(req.rid, 0.0)
        if hit <= 0 or dur is None:
            return "declined:store_absent"
        self.tree_cache.start_read(req.rid, hit, dur)
        return "issued"


def req(rid):
    return SimpleNamespace(rid=rid, prefetch_deferred=None)


class MemChannel:
    """In-memory stand-in for the gloo ack stream: what a follower sends in
    its pass k is harvested by PP0 at the top of its pass k+1."""

    def __init__(self, box):
        self.box = box
        self.sent = 0

    def send_nowait(self, ack):
        self.box.append(ack)
        self.sent += 1
        return True

    def pump(self):
        return None

    def harvest(self):
        out = list(self.box)
        self.box.clear()
        return out


class Ring:
    def __init__(self, m, monkeypatch, prompts, read_s_by_rank, paced=True, channel=True):
        self.m = m
        self.clock = Clock()
        monkeypatch.setattr(m, "_clock", self.clock)
        if paced:
            monkeypatch.setenv(m.ENV_PACED, "1")
        else:
            monkeypatch.delenv(m.ENV_PACED, raising=False)
        self.stages = [Stage(r, self.clock, prompts, read_s_by_rank[r]) for r in range(3)]
        self.box = []
        for s in self.stages:
            if channel:
                s._weg2_fb_channel = MemChannel(self.box)
            assert m.armed(s)
        self.wire = {}
        self.pass_n = 0
        self.sleeps = {0: 0.0, 1: 0.0, 2: 0.0}
        self._in = None

        def _sleep(sec):
            self.sleeps[self._in] += sec
            self.clock.t += sec

        monkeypatch.setattr(m.time, "sleep", _sleep)

    def arrive(self, rid, ranks=(0, 1, 2)):
        for s in self.stages:
            if s.ps.pp_rank not in ranks:
                continue
            r = req(rid)
            s.waiting_queue.append(r)
            self.m.intake(s, r, lambda gate: None)

    def step(self):
        m = self.m
        k = self.pass_n
        self.wire[k] = m.pp0_publish(self.stages[0], [])
        for s in self.stages[1:]:
            src = k - s.ps.pp_rank
            if src >= 0:
                m.follower_absorb(s, list(self.wire.get(src, [])))
        for s in self.stages:
            self._in = s.ps.pp_rank
            for r in list(s.waiting_queue):
                credit = m.admission(s, r, lambda kind, rid: None)
                if credit is None:
                    continue
                s.waiting_queue.remove(r)
                s.admitted.append(
                    (k - s.ps.pp_rank, r.rid, getattr(r, "_weg2_prefix_cap", None), credit)
                )
        self.pass_n += 1
        self.clock.t += DT

    def run(self, passes):
        for _ in range(passes):
            self.step()

    def plans(self, rid):
        return [[a[:3] for a in s.admitted if a[1] == rid] for s in self.stages]

    def wire_objs(self):
        return [o for k in sorted(self.wire) for o in self.wire[k]]


# ---------------------------------------------------------------------------
# golden digest of the switch-OFF behaviour (wire bytes, plans, logs)
# ---------------------------------------------------------------------------

PROMPTS = {"aaaa-told": 100_000, "bbbb-fresh": 0, "cccc-slow": 4096}


def _rs(pp0, fol, slow=None):
    out = {}
    for r in range(3):
        d = {"aaaa-told": pp0 if r == 0 else fol, "bbbb-fresh": 0.0, "cccc-slow": 0.1}
        if slow is not None and r == 2:
            d["cccc-slow"] = slow
        out[r] = d
    return out


def _scenarios():
    """(name, env, paced, read_s, script) -- script: list of (op, arg)."""
    return [
        ("paced-two", {}, True, _rs(0.5, 0.8), [("arrive", "aaaa-told"), ("run", 2), ("arrive", "bbbb-fresh"), ("run", 60)]),
        ("paced-residual", {"SGLANG_WEG2_TOLD_PACE_CAP_S": "0.5"}, True, _rs(0.1, 2.0), [("arrive", "aaaa-told"), ("run", 80)]),
        ("single-phase", {}, False, _rs(0.5, 0.8), [("arrive", "aaaa-told"), ("run", 2), ("arrive", "bbbb-fresh"), ("run", 60)]),
        ("paced-abort", {}, True, _rs(0.2, 0.2), [("arrive", "aaaa-told"), ("run", 8), ("clear", None), ("run", 40)]),
        ("paced-absolute", {"SGLANG_WEG2_TOLD_ABSOLUTE": "1"}, True, _rs(0.3, 0.4), [("arrive", "aaaa-told"), ("arrive", "cccc-slow"), ("run", 60)]),
        ("paced-slow-follower", {"SGLANG_WEG2_TOLD_PACE_CAP_S": "0.5"}, True, _rs(0.1, 0.1, slow=3.0), [("arrive", "cccc-slow"), ("run", 120)]),
    ]


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        if record.name.startswith("sglang.srt.managers.weg2") or record.name.startswith("sglang.srt.weg2"):
            self.lines.append((record.levelno, record.getMessage()))


def run_digest(m, monkeypatch, extra_env):
    """sha256 over every scenario's wire pickles, plans, credits, busy-wait
    time and the told module's log lines, under ``extra_env``."""
    h = hashlib.sha256()
    cap = _Cap()
    lg = logging.getLogger("sglang.srt")
    old_level = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    try:
        for name, env, paced, read_s, script in _scenarios():
            with monkeypatch.context() as mp:
                for k in list(__import__("os").environ):
                    if k.startswith("SGLANG_WEG2_TOLD") or k == "SGLANG_WEG2_P_TWIN_DEFER":
                        mp.delenv(k, raising=False)
                for k, v in {**env, **extra_env}.items():
                    mp.setenv(k, v)
                cap.lines.clear()
                ring = Ring(m, mp, PROMPTS, read_s, paced=paced, channel=False)
                for op, arg in script:
                    if op == "arrive":
                        ring.arrive(arg)
                    elif op == "run":
                        ring.run(arg)
                    elif op == "clear":
                        for s in ring.stages:
                            s.waiting_queue.clear()
                h.update(name.encode())
                for k in sorted(ring.wire):
                    h.update(pickle.dumps(ring.wire[k], protocol=4))
                for s in ring.stages:
                    h.update(repr(s.admitted).encode())
                    h.update(repr(sorted(s.tree_cache.ongoing)).encode())
                h.update(repr(sorted(ring.sleeps.items())).encode())
                for lv, msg in cap.lines:
                    h.update(f"{lv}:{msg}".encode())
    finally:
        lg.removeHandler(cap)
        lg.setLevel(old_level)
    return h.hexdigest()
