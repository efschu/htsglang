# SPDX-License-Identifier: Apache-2.0
"""#1480 LEND-RESUME-GATE: the dual front resumes P 'from=lend' only when EVERY P
stage file says lent==0.

Metal B9c (boot ...fs10041650_ed0803afa6, 17:02:26-17:03:06Z): Q-660 stage 1-lend --
PP0 (5090, the card with margin 0) returned its allocator cache (1.85 GiB) to the card
pool, D grew into it, PP0's reclaim was REFUSED (302 MiB free < 1.85 GiB; PP1/PP2 got
theirs back), and the front sent `resume from=lend` 200 ms later although
dual_d_priority.PressureStages resumes from 'reclaiming' only at p_lent <= 0. Cause:
the front read the stage files under the tag 'weg2' (its own env carries neither
SGLANG_WEG2_DUAL_KV_TAG nor SGLANG_WEG2_TAG -- the ranks' env does, launcher.py:14297),
found none, p_lent was always 0. P then ran without cache on a card with margin 0 and
died with an OOM 40 s later.

The tests set NO env tag on the front (that is the B9c condition the older tests hid by
exporting SGLANG_WEG2_DUAL_KV_TAG): only ``front.tag`` knows the boot.

DANGER DIRECTION: P resumes while a rank's loan is still with D. Default OFF = the old
reading byte for byte (the B9c reading, asserted as such).
"""
from __future__ import annotations

import inspect
import logging
import os
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import dual_p_kv_stage as PK
from sglang.srt.weg2 import front as FR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
TAG = "boot-1480-fs10041650"
LENDS = (1851785216, 1392508928, 1283457024)         # B9c P.log:52094-52096 (PP0 / PP1 / PP2)
STEP_B = 64 << 20


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # the FRONT's env: no tag at all (B9c)
    for k in ("SGLANG_WEG2_DUAL_KV_TAG", "SGLANG_WEG2_TAG", "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    tmp = tempfile.mkdtemp(prefix="wkvs1480")
    monkeypatch.setattr(PK, "stage_file",
                        lambda tag, r, root="": os.path.join(tmp, "%s-pp%d.json" % (tag, int(r))))
    return tmp


class _Cards:
    """Three cards, three P ranks: the real awake_lend / awake_reclaim, the real stage files."""

    def __init__(self, monkeypatch):
        self.paths, self.d, self.p, self.actors, self.phys = [], [], [], [], [0, 0, 0]
        self.sched = []
        monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        for r in range(3):
            path = os.path.join(tempfile.mkdtemp(prefix="wkvc1480"), "card")
            d = K.CardKvLedger(path, "D")
            d.contribute(2 * GiB, committed=0)
            p = K.CardKvLedger(path, "P")
            p.contribute(0)
            actor = types.SimpleNamespace(ledger=p, mapped_tokens=0, step=4096, top=196608,
                                          table=lambda: [0, STEP_B], weights_bytes=0)
            self.paths.append(path)
            self.d.append(d)
            self.p.append(p)
            self.actors.append(actor)
            self.sched.append(types.SimpleNamespace(tp_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(pp_rank=r, **{PK.ACTOR_ATTR: actor}))))
        # the ranks publish under THEIR tag (rank env = launcher.py:14297)
        monkeypatch.setattr(PK, "_republish_stage", lambda s, a: PK.publish_stage(
            a, TAG, s.tp_worker.model_runner.pp_rank))
        for r in range(3):
            PK.publish_stage(self.actors[r], TAG, r)
        monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT")
        monkeypatch.delenv("SGLANG_WEG2_GROUP")

    def rank(self, monkeypatch, r, fn, *a):
        monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        try:
            return fn(self.sched[r], *a)
        finally:
            monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT")
            monkeypatch.delenv("SGLANG_WEG2_GROUP")

    def lend_all(self, monkeypatch):
        for r in range(3):
            self.phys[r] = 0
            got = self.rank(monkeypatch, r, lambda s, r=r: PK.awake_lend(
                s, "t", phys=lambda r=r: self.phys[r],
                empty_cache=lambda r=r: self.phys.__setitem__(r, LENDS[r])))
            assert got == LENDS[r]

    def front(self, p_state="reclaiming"):
        f = types.SimpleNamespace(dual_kv_ledgers=list(self.paths), tag=TAG,
                                  DUAL_LEND_GATE_LOG_S=FR.Front.DUAL_LEND_GATE_LOG_S)
        st = DP.PressureStages(sleep_capable=False, reclaim_after=10)
        st.p_state = p_state
        f._dual_stages_obj = st
        for name in ("_dual_p_stage_reading", "_dual_lend_gate_lent"):
            setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
        return f, st

    def tick(self, f, st):
        rd = f._dual_p_stage_reading()
        rd.pop("d_short", None)
        return st.tick(pressure=0, seats_done=0, host_ok=True, **rd)


def _b9c(monkeypatch):
    """B9c 17:02:26-28: all three lend, D grows into PP0's loan, PP1/PP2 reclaim, PP0 is refused."""
    c = _Cards(monkeypatch)
    c.lend_all(monkeypatch)
    free0 = K.peek(c.paths[0]).free
    assert c.d[0].request(free0)[0] == free0              # D takes everything on card 0
    assert c.rank(monkeypatch, 0, PK.awake_reclaim, "calm") == 0, "PP0 refused (D holds the loan)"
    assert c.rank(monkeypatch, 1, PK.awake_reclaim, "calm") == LENDS[1]
    assert c.rank(monkeypatch, 2, PK.awake_reclaim, "calm") == LENDS[2]
    return c, free0


# -- default OFF: the B9c reading, byte for byte ------------------------------------

def test_default_is_off():
    assert envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.get() is False


def test_off_reproduces_b9c_resume_after_a_refused_reclaim(monkeypatch):
    c, _ = _b9c(monkeypatch)
    f, st = c.front()
    rd = f._dual_p_stage_reading()
    assert rd["p_lent"] == 0, "B9c: the front reads the stage files under 'weg2' and finds none"
    action, line = c.tick(f, st)
    assert action == "resume" and "from=lend" in line, "B9c 17:02:28.251: resume although PP0's loan is with D"


def test_off_never_calls_the_gate(monkeypatch):
    c, _ = _b9c(monkeypatch)
    f, st = c.front()
    f._dual_lend_gate_lent = lambda *_: pytest.fail("the gate ran with the switch off")
    assert f._dual_p_stage_reading()["p_lent"] == 0


# -- ON ------------------------------------------------------------------------------

def test_on_refused_reclaim_holds_the_resume_until_pp0_has_its_loan_back(monkeypatch):
    c, free0 = _b9c(monkeypatch)
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        rd = f._dual_p_stage_reading()
        assert rd["p_lent"] == LENDS[0], "only PP0 still lends; PP1/PP2 are back"
        assert c.tick(f, st) == (None, None) and st.p_state == "reclaiming", "no resume while the reclaim is open"
        # D leaves the loan; PP0 reclaims; every file at 0 -> resume
        c.d[0].release(free0)
        assert c.rank(monkeypatch, 0, PK.awake_reclaim, "retry") == LENDS[0]
        action, line = c.tick(f, st)
        assert action == "resume" and "from=lend" in line and st.p_state == "serving"


def test_on_clean_reclaim_resumes_as_before(monkeypatch):
    c = _Cards(monkeypatch)
    c.lend_all(monkeypatch)
    for r in range(3):
        assert c.rank(monkeypatch, r, PK.awake_reclaim, "calm") == LENDS[r]
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        assert f._dual_p_stage_reading()["p_lent"] == 0
        action, line = c.tick(f, st)
        assert action == "resume" and "from=lend" in line


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_on_any_single_rank_holding_its_loan_holds_the_resume(monkeypatch, rank):
    """Rank agreement: the verdict is the sum over EVERY rank's own file, no rank is special."""
    c = _Cards(monkeypatch)
    c.lend_all(monkeypatch)
    for r in range(3):
        if r != rank:
            assert c.rank(monkeypatch, r, PK.awake_reclaim, "calm") == LENDS[r]
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        assert f._dual_p_stage_reading()["p_lent"] == LENDS[rank]
        assert c.tick(f, st) == (None, None) and st.p_state == "reclaiming"


def test_on_gate_never_lowers_the_old_reading(monkeypatch):
    """The env tag, when the front has one (tests / other launchers), still counts."""
    c, _ = _b9c(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_DUAL_KV_TAG", TAG)
    f, st = c.front()
    legacy = f._dual_p_stage_reading()["p_lent"]
    assert legacy == LENDS[0]
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        assert f._dual_p_stage_reading()["p_lent"] == legacy


def test_on_blind_front_does_not_deadlock_p(monkeypatch):
    """No readable stage file under any tag: no verdict to hold on -- P is not held for ever."""
    c, _ = _b9c(monkeypatch)
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        f.tag = "some-other-boot"
        assert f._dual_p_stage_reading()["p_lent"] == 0


# -- the marker --------------------------------------------------------------------------

def test_marker_is_key_value_and_rate_limited(monkeypatch, caplog):
    c, free0 = _b9c(monkeypatch)
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        with caplog.at_level(logging.WARNING, logger=FR.logger.name):
            for _ in range(5):
                f._dual_p_stage_reading()
            held = [r.getMessage() for r in caplog.records if "#1480 LEND-RESUME-GATE" in r.getMessage()]
            assert len(held) == 1, "5 ticks inside the log interval -> one line"
            assert "verdict=HOLD" in held[0] and "p_lent=%d" % LENDS[0] in held[0]
            assert "per_rank=[%d, 0, 0]" % LENDS[0] in held[0] and "files=3/3" in held[0] and "tag=%s" % TAG in held[0]
            f._dual_lend_gate_next = time.time() - 1
            f._dual_p_stage_reading()
            assert len([r for r in caplog.records if "verdict=HOLD" in r.getMessage()]) == 2
            c.d[0].release(free0)
            c.rank(monkeypatch, 0, PK.awake_reclaim, "retry")
            f._dual_p_stage_reading()
            f._dual_p_stage_reading()
            rel = [r.getMessage() for r in caplog.records if "verdict=RELEASE" in r.getMessage()]
            assert len(rel) == 1 and "p_lent=0" in rel[0]


# -- mutants: each guard has a red form ------------------------------------------------------

def test_mutant_gate_ignored_goes_red(monkeypatch):
    """Without the gate's override of p_lent the B9c resume comes back."""
    c, _ = _b9c(monkeypatch)
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        f._dual_lend_gate_lent = lambda legacy: legacy                   # mutant: gate computes nothing
        with pytest.raises(AssertionError):
            assert c.tick(f, st) == (None, None)


def test_mutant_front_tag_not_read_goes_red(monkeypatch):
    c, _ = _b9c(monkeypatch)
    with envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.override(True):
        f, st = c.front()
        f.tag = ""                                                       # mutant: front tag dropped
        with pytest.raises(AssertionError):
            assert c.tick(f, st) == (None, None)


def test_gate_is_front_dual_ladder_only():
    """The helper is reached only from the stage reading behind the switch (flip / NF / INT8 never)."""
    src = inspect.getsource(FR.Front._dual_p_stage_reading)
    assert "if envs.SGLANG_WEG2_DUAL_LEND_RESUME_GATE.get():" in src
    assert src.count("_dual_lend_gate_lent") == 1
    import pathlib
    text = pathlib.Path(FR.__file__).read_text()
    assert text.count("_dual_lend_gate_lent(") == 2        # the one call + the def
