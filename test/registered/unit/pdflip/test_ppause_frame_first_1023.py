"""PPAUSE-FRAME-FIRST + PPAUSE-LOAD-FENCE (23.10.2026), 27B rc35i4ar a45112428e, P = PP2 over 2x3080.

(B) Metal, boot ..._1023_144422 (container restart 14:43:59Z):
  14:48:32 PP0 '#969N ADMIT slot=0 fwd_ct=6 bs=1 extend=2008 rids=[pdflip-1-3]'   (launched; proxy after PP0's commit)
  14:48:32-53 PP0 'S82 COMMIT-PARK n=1 outcome=read rid=pdflip-1-2 waited_ms=20480.2'  (a real store read, l3fill 20.8 s)
  14:48:53 PP1 'PP-CHAIN-RECV #1460 CHAIN-RECV blocked 20660 ms', then '#969N ADMIT slot=0 fwd_ct=6 extend=2008'
PP1 had looked at its gate before the frame of fwd 6 was posted ('gate=False proxy=False mbs=0') and took the
INDEFINITE chain park; the frame lay on its wire ~20 s. Pinned (RED on 48b7e73135 -- the method does not exist,
the gate parks at once):
  1. two-stage fake with the metal timeline: the follower resumes when the FRAME is posted (1.2 s), not when PP0's
     next chain comes (21.7 s, the old park);
  2. SOLL R11 / danger direction (stages drift apart): a posted chain is ALWAYS read before a frame -- also when
     the chain is posted between the follower's chain read and its frame read; no signal channel -> the old park;
  3. MUTANTS of the real source (chain check after the frame check; no chain re-check after a frame) are caught;
  4. the gate's idle arm runs the wait and a frame-first answer takes the existing proxy arm (recv_reqs = []);
  5. the window lapses into the old indefinite park; window 0 = the old park at once.
(2) PPAUSE-LOAD-FENCE instrument (PP1 'PDFLIP-START-LOADING tokens=29184 ... kv_issue_ms=2190' vs PP0 12 ms):
  6. the arena guard's index reads are timed (kv.guard), the controller records fence events and harvests them by
     query() only (never synchronize), switchable by FLLIPER_PDFLIP_LOAD_FENCE_PROBE (default on).
"""

import inspect
import textwrap
import time
from types import MethodType, SimpleNamespace

import pytest

from flliper.srt.environ import envs
from flliper.srt.managers import scheduler_pp_mixin as mixin
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

WAIT = mixin.SchedulerPPMixin._pp_follower_idle_wait


class _TwoStage:
    """PP0 -> PP1 over the two counters PP1 can poll: PP0 posts the frame of the forward it launched at
    ``frame_t`` and its next chain at ``chain_t`` (seconds on a fake clock that the follower's sleeps advance).
    ``chain_none`` = no signal channel (``_pp_row_chain_pending`` answers None)."""

    def __init__(self, frame_t=None, chain_t=None, chain_none=False, chain_reads=None):
        self.t = 0.0
        self.frame_t, self.chain_t, self.chain_none = frame_t, chain_t, chain_none
        self.chain_reads = list(chain_reads) if chain_reads is not None else None
        self.ps = SimpleNamespace(pp_rank=1)
        self.calls = []

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt

    def _pp_row_chain_pending(self):
        self.calls.append("chain")
        if self.chain_none:
            return None
        if self.chain_reads is not None:
            return self.chain_reads.pop(0) if self.chain_reads else True
        return self.chain_t is not None and self.t >= self.chain_t

    def _pp_row_any_proxy_signal(self):
        self.calls.append("proxy")
        return self.frame_t is not None and self.t >= self.frame_t

    def old_park_resumes_at(self):
        """The old indefinite chain park: the follower moves when PP0's next chain is posted."""
        return self.chain_t


def _run(stage, fn=WAIT):
    return fn(stage, clock=stage.clock, sleep=stage.sleep), stage.t


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    monkeypatch.setenv("FLLIPER_PP_FOLLOWER_FRAME_WAIT_S", "30")
    yield


# 1. the metal timeline -------------------------------------------------------------------------------------------

def test_metal_144832_the_frame_is_run_when_it_arrives_not_after_pp0s_park():
    # fwd 6 launched at 0; its frame follows PP0's commit (+1.2 s); PP0 parks 20.5 s on pdflip-1-2's read
    st = _TwoStage(frame_t=1.2, chain_t=21.7)
    how, at = _run(st)
    assert how == "proxy"
    assert at == pytest.approx(1.2, abs=0.01)
    assert st.old_park_resumes_at() - at > 20.0          # the old park: 'CHAIN-RECV blocked 20660 ms'


def test_the_metal_default_window_is_on():
    assert float(envs.FLLIPER_PP_FOLLOWER_FRAME_WAIT_S.default) == 30.0


# 2. R11 / danger direction: the chain is never overtaken -----------------------------------------------------------

def test_a_posted_chain_goes_first_even_with_the_frame_there():
    how, at = _run(_TwoStage(frame_t=1.0, chain_t=1.0))
    assert how == "chain" and at == pytest.approx(1.0, abs=0.01)


def test_a_chain_posted_between_the_two_reads_still_goes_first():
    # poll k: chain read False (not yet posted) -> PP0 posts chain s and frame s -> frame read True;
    # the re-read finds the chain: it is taken before the frame
    st = _TwoStage(frame_t=0.0, chain_reads=[False, True])
    how, _ = _run(st)
    assert how == "chain" and st.calls == ["chain", "proxy", "chain"]


def test_no_signal_channel_is_the_old_park_never_a_frame():
    how, at = _run(_TwoStage(frame_t=0.0, chain_none=True))
    assert how == "park" and at == 0.0


def test_the_window_lapses_into_the_old_park(monkeypatch):
    monkeypatch.setenv("FLLIPER_PP_FOLLOWER_FRAME_WAIT_S", "2")
    how, at = _run(_TwoStage())                         # nothing comes: a truly idle follower
    assert how == "park" and 2.0 <= at < 2.01


def test_window_zero_is_the_old_park_at_once(monkeypatch):
    monkeypatch.setenv("FLLIPER_PP_FOLLOWER_FRAME_WAIT_S", "0")
    st = _TwoStage(frame_t=0.0)
    assert _run(st) == ("park", 0.0) and st.calls == []


# 2b. Review-Auflage A1: only tp0 has a chain receiver -> tp / cp / dp-attention = the old park at once -------------

def test_tp_size_2_is_the_old_park_at_once_even_with_a_frame_posted():
    st = _TwoStage(frame_t=0.0)
    st.ps = SimpleNamespace(pp_rank=1, tp_size=2, attn_cp_size=1)
    assert _run(st) == ("park", 0.0) and st.calls == []


def test_attn_cp_size_2_is_the_old_park_at_once():
    st = _TwoStage(frame_t=0.0)
    st.ps = SimpleNamespace(pp_rank=1, tp_size=1, attn_cp_size=2)
    assert _run(st) == ("park", 0.0) and st.calls == []


def test_dp_attention_is_the_old_park_at_once():
    st = _TwoStage(frame_t=0.0)
    st.ps = SimpleNamespace(pp_rank=1, tp_size=1, attn_cp_size=1)
    st.enable_dp_attention = True
    assert _run(st) == ("park", 0.0) and st.calls == []


def test_tp1_cp1_still_takes_the_frame():
    st = _TwoStage(frame_t=0.0)
    st.ps = SimpleNamespace(pp_rank=1, tp_size=1, attn_cp_size=1)
    assert _run(st)[0] == "proxy"


def test_mutant_without_the_tp_guard_is_caught():
    src = textwrap.dedent(inspect.getsource(WAIT))
    a = src.index('    ps = getattr(self, "ps", None)\n    if (')
    b = src.index("    bound = float(")
    m = _mutant2((src[a:b], '    ps = getattr(self, "ps", None)\n'))
    st = _TwoStage(frame_t=0.0)
    st.ps = SimpleNamespace(pp_rank=1, tp_size=2, attn_cp_size=2)
    assert _run(st, fn=m)[0] == "proxy"                   # the mutant hangs a tp group on the proxy arm
    assert _run(st)[0] == "park"                          # the real code does not


def test_the_window_expiry_names_itself(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PP_FOLLOWER_FRAME_WAIT_S", "2")
    with caplog.at_level("INFO"):
        _run(_TwoStage())
    assert any("PPAUSE FRAME-WAIT-EXPIRED pp_rank=1" in r.getMessage() for r in caplog.records)


# 3. mutants of the real source --------------------------------------------------------------------------------------

def _mutant(old, new):
    src = textwrap.dedent(inspect.getsource(WAIT))
    assert src.count(old) == 1, old
    ns = dict(vars(mixin))
    ns.setdefault("Scheduler", object)                    # a TYPE_CHECKING-only name in the annotation
    exec(compile(src.replace(old, new), "<mutant>", "exec"), ns)
    return ns["_pp_follower_idle_wait"]


_SWAP = ("        chain = self._pp_row_chain_pending()\n"
         "        if chain is None:\n"
         "            return \"park\"\n"
         "        if chain:\n"
         "            return \"chain\"\n"
         "        if self._pp_row_any_proxy_signal():\n",
         "        if self._pp_row_any_proxy_signal():\n")
_NO_REREAD = ("            if self._pp_row_chain_pending():\n"
              "                return \"chain\"\n", "")


def _mutant2(*pairs):
    src = textwrap.dedent(inspect.getsource(WAIT))
    for old, new in pairs:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(mixin))
    ns.setdefault("Scheduler", object)
    exec(compile(src, "<mutant>", "exec"), ns)
    return ns["_pp_follower_idle_wait"]


def test_mutant_frame_checked_before_chain_is_held_by_the_reread():
    """Defence in depth: with the frame asked first, the re-read after a frame still puts the chain ahead."""
    how, _ = _run(_TwoStage(frame_t=1.0, chain_t=1.0), fn=_mutant2(_SWAP))
    assert how == "chain"


def test_mutant_frame_first_without_the_reread_is_caught():
    m = _mutant2(_SWAP, _NO_REREAD)
    how, _ = _run(_TwoStage(frame_t=1.0, chain_t=1.0), fn=m)
    assert how == "proxy"                                 # the mutant runs the frame over a posted chain
    how, _ = _run(_TwoStage(frame_t=1.0, chain_t=1.0))
    assert how == "chain"                                 # ... the real code does not


def test_mutant_without_the_chain_reread_is_caught():
    m = _mutant(
        "            if self._pp_row_chain_pending():\n"
        "                return \"chain\"\n",
        "")
    how, _ = _run(_TwoStage(frame_t=0.0, chain_reads=[False, True]), fn=m)
    assert how == "proxy"
    how, _ = _run(_TwoStage(frame_t=0.0, chain_reads=[False, True]))
    assert how == "chain"


# 4. the gate wiring -------------------------------------------------------------------------------------------------

def test_the_idle_arm_runs_the_wait_and_a_frame_takes_the_proxy_arm():
    src = inspect.getsource(mixin.SchedulerPPMixin._event_loop_pp_body)
    i = src.index('if self._pp_follower_idle_wait() == "proxy":')
    arm = src[i:i + 600]
    assert arm.index("recv_reqs = []") < arm.index("else:") < arm.index("self.request_receiver.recv_requests()")
    # the wait sits in the idle (elif) arm, after the chain-gate's own blocking receive
    assert src.index("if _chain_gate is None or _chain_gate:") < i


def test_a_frame_first_line_names_the_wait(caplog):
    st = _TwoStage(frame_t=1.2, chain_t=21.7)
    with caplog.at_level("INFO"):
        _run(st)
    assert any("PPAUSE FRAME-FIRST pp_rank=1 waited_ms=1200" in r.getMessage() for r in caplog.records)


# 6. PPAUSE-LOAD-FENCE -----------------------------------------------------------------------------------------------

class _SlowIdx:
    """A device index tensor stand-in: reading its min/max on the host waits (the D2H sync behind the fence)."""

    def __init__(self, wait_s):
        self.wait_s = wait_s

    def numel(self):
        return 4

    def _v(self, v):
        idx = self

        class _I:
            def __int__(self_inner):
                time.sleep(idx.wait_s)
                return v
        return _I()

    def min(self):
        return self._v(0)

    def max(self):
        return self._v(3)


def test_the_arena_guard_times_its_index_reads(monkeypatch):
    import torch

    from flliper.srt.mem_cache.pool_host import arena_pool as ap

    cls = next(c for c in vars(ap).values() if isinstance(c, type) and "_arena_load_guard" in vars(c))
    stub = SimpleNamespace(arena_slots=16, _pinned=None)
    dev = SimpleNamespace(k_buffer=[torch.zeros(8, 1)])
    cls._arena_load_guard(stub, dev, torch.tensor([1, 2]), _SlowIdx(0.03), 1, nrows=2, nmiss=0)
    assert stub._pdflip_guard_ms >= 55.0                  # two blocking reads of 30 ms each
    monkeypatch.setenv("FLLIPER_PDFLIP_LOAD_FENCE_PROBE", "0")
    stub2 = SimpleNamespace(arena_slots=16, _pinned=None)
    cls._arena_load_guard(stub2, dev, torch.tensor([1, 2]), _SlowIdx(0.0), 1, nrows=2, nmiss=0)
    assert not hasattr(stub2, "_pdflip_guard_ms")


class _Ev:
    def __init__(self, done, ms=0.0):
        self.done, self.ms, self.synced = done, ms, False

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.ms

    def synchronize(self):  # never called by the harvest
        self.synced = True


def test_the_fence_harvest_reads_completed_events_only_and_never_syncs(caplog):
    from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hc

    done0, done1 = _Ev(True), _Ev(True, ms=2140.0)
    open0, open1 = _Ev(True), _Ev(False)
    ctrl = SimpleNamespace(_pdflip_load_fence_pending=[(done0, done1, 29184, 2190.0, 2185.0),
                                                       (open0, open1, 4096, 10.0, 1.0)])
    with caplog.at_level("INFO"):
        hc._load_fence_harvest(ctrl)
    msgs = [r.getMessage() for r in caplog.records if "PDFLIP-LOAD-FENCE" in r.getMessage()]
    assert len(msgs) == 1 and "tokens=29184 fence_wait_ms=2140.0 l0_host_ms=2190.0 guard_host_ms=2185.0" in msgs[0]
    assert len(ctrl._pdflip_load_fence_pending) == 1      # the open one waits for the next load
    assert not any(e.synced for e in (done0, done1, open0, open1))


def test_the_probe_is_wired_and_on_by_default():
    from flliper.srt.mem_cache import memory_pool_host as mph
    from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hc

    assert envs.FLLIPER_PDFLIP_LOAD_FENCE_PROBE.default is True
    sl = inspect.getsource(hc.HybridCacheController.start_loading) if hasattr(hc, "HybridCacheController") else \
        inspect.getsource(hc)
    assert "_load_fence_event(self.load_stream)" in sl and "_load_fence_harvest(self)" in sl
    assert sl.index("_lf_e0 = _load_fence_event") < sl.index("self.load_stream.wait_stream(fence_stream)") \
        < sl.index("_lf_e1 = _load_fence_event")
    assert '_acc["kv.guard"]' in inspect.getsource(mph)
