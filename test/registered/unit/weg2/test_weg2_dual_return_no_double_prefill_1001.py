# SPDX-License-Identifier: Apache-2.0
"""D PRIORITY (iv): P's return after a stop or a sleep computes only what the
store does not hold -- no double prefill ("Lesen statt Rechnen", user order
30.09. 07:25Z: "gibt den context frei und das erarbeitete in den L2 zur
späteren weiterverwendung").

The path under test is the shipped one end to end at the protocol level:
  front: D presses -> stage 1 (P-PAUSE) -> the paused leg 1 is requeued at the
  head (front._dual_requeue_paused) -> [stage 2: P sleeps, wakes after a seat
  ended] -> stage resume -> the head goes back to P as leg 1;
  P PP0: #1400 intake registers the store read -> the read terminates with the
  k finished chunks -> pp0_publish tells the completed prefix -> admission
  returns it as the prefix P adopts (#1419 caps the radix match at it), so P
  prefills N - told tokens.

DANGER DIRECTION: P recomputes the k chunks it finished before the pause.
MUTANT: the requeued leg's intake without the store read (the #1400
registration line removed from weg2_store_told.intake, exec'd from the real
source) -> told 0 -> P computes N -> red.
"""
from __future__ import annotations

import collections
import importlib.util
import inspect
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import weg2_store_told as m
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import front as FR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_told1400", os.path.join(_HERE, "..", "managers", "test_weg2_store_told_1400.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

N = 41188              # the metal dual1m prompt
CHUNK = 4096
K = 7                  # chunks P finished before the pause (all in L2 by the per-chunk write-through)
STORE = K * CHUNK


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", "SGLANG_WEG2_DUAL_SHARE"):
        monkeypatch.delenv(k, raising=False)


def _front_requeue(rid):
    """stage 1 -> the paused leg comes back aborted -> requeued at the head."""
    f = types.SimpleNamespace(queue=collections.deque(), counters=collections.Counter())
    f._dual_requeue_paused = types.MethodType(FR.Front._dual_requeue_paused, f)
    p = types.SimpleNamespace(rid=rid, dual_pause=True, leg1_done=True, dual_paused_n=0)
    f._dual_requeue_paused(p)
    return f, p


def _p_leg1(rid, intake=None):
    """The head's leg 1 on P PP0: intake -> the store read terminates with the
    k finished chunks -> publish -> admission. Returns the tokens P prefills."""
    s = T._Sched(0)
    assert m.armed(s), "#1400 armed on P (carrierless PP, tp 1, storage)"
    r = T._req(rid)
    s.waiting_queue.append(r)
    (intake or m.intake)(s, r, lambda *_: None)
    if s.registered:
        s.tree_cache.register(rid, loaded=STORE, flips=0, completed=STORE)
    m.pp0_publish(s, [])
    told = m.admission(s, r, lambda *_: None)
    return N - int(told or 0)


def _return(sleep: bool, intake=None) -> int:
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    base = dict(p_grant_bytes=64, d_air_bytes=32, weights_bytes=1000, host_ok=True)
    act, _ = st.tick(pressure=9, p_committed=500, free_min=0, seats_done=0, **base)
    assert act == "stop"
    f, p = _front_requeue("weg2-0-1")
    assert f.queue[0] is p and p.dual_paused_n == 1 and not p.leg1_done
    if sleep:
        act, _ = st.tick(pressure=9, p_committed=0, free_min=0, seats_done=0, **base)
        assert act == "sleep"
        act, _ = st.tick(pressure=0, p_committed=0, free_min=2000, seats_done=1, **base)
        assert act == "wake"
    else:
        act, _ = st.tick(pressure=0, p_committed=0, free_min=2000, seats_done=0, **base)
        assert act == "resume"
    assert st.p_state == "serving"
    head = f.queue.popleft()                                  # the pump's next leg 1
    return _p_leg1(head.rid, intake=intake)


@pytest.mark.parametrize("sleep", [False, True], ids=["stop", "sleep"])
def test_p_returns_and_computes_only_n_minus_the_store(sleep):
    computed = _return(sleep)
    assert computed == N - STORE, "P prefills what the store does not hold, not the whole prompt"
    assert computed != N


def _intake_without_store_read():
    src = textwrap.dedent(inspect.getsource(m.intake))
    line = "verdict = scheduler._prefetch_kvcache(req)"
    assert src.count(line) == 1, "the #1400 registration moved -- re-aim the mutant"
    ns = dict(vars(m))
    exec(compile(src.replace(line, 'verdict = "declined:mutant"'), m.__file__, "exec"), ns)
    return ns["intake"]


@pytest.mark.parametrize("sleep", [False, True], ids=["stop", "sleep"])
def test_the_requeue_without_store_intake_mutant_is_red(sleep):
    with pytest.raises(AssertionError):
        computed = _return(sleep, intake=_intake_without_store_read())
        assert computed == N - STORE
