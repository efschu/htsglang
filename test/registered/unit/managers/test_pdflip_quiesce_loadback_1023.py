# SPDX-License-Identifier: Apache-2.0
"""QUIESCE-LOADBACK-1023 / QUIESCE-BUSY-1023 (NF rc36p ..._1023_215911, D epoch 4).

The death, in the log's own order (D.log / front.log of that boot):

  22:29:25,686 front  PARK-RUNNING parked=3 rids=[pdflip-4-207,-218,-219] still_running=[]
  22:29:25     D x3   Cache not flushed ... #running-req: 0, not-idle because: hicache_load_back(1)
                      (1602 refusals, the ack of the REVERTED admission of pdflip-4-218)
  22:29:55     D x3   PDFLIP-D-PARK requeue (awake): ['pdflip-4-207']   (30 s awake re-queue)
  22:29:56     D x3   not-idle because: running_batch, last_batch, overlap_result_queue (2091)
  22:30:55,701 front  PDFLIP-RPC ISSUED leg=sleep group=D path=/release_memory_occupation
  22:30:55     D x3   W120 PdFlipSleepDrainRefused ... other_terms=3 -> W29 -> RANK-DEATH

Two defects, two tests each side:

* D: the flush's idle vote never polled the finished load-back ack (``loading_check``
  only runs in a planned prefill pass; a parked D plans none). The vote now drains
  finished acks first, on trees whose ``loading_check`` is rank-local.
* front: at the quiesce deadline ``witness_verdict(1, False)`` is None ("agree": both
  say busy) and the flip went on into the sleep legs. Both-busy now aborts the flip.

Hermetic: no GPU, no torch device.
"""

import os
import re
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import pdflip_quiesce_loadback as ql

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SCHED = os.path.join(ROOT, "python", "flliper", "srt", "managers", "scheduler.py")
FRONT = os.path.join(ROOT, "python", "flliper", "srt", "pdflip", "front.py")
URC = os.path.join(ROOT, "python", "flliper", "srt", "mem_cache", "unified_radix_cache.py")


class _RankLocalTree:
    """The shape of UnifiedRadixCache's load-back bookkeeping (rank-local check)."""

    LOADING_CHECK_RANK_LOCAL = True

    def __init__(self, finished=True):
        self.ongoing_load_back = {7: "node-of-pdflip-4-218"}
        self.finished = finished
        self.calls = 0

    def loading_check(self):
        self.calls += 1
        if self.finished:
            self.ongoing_load_back.pop(7, None)


class _CollectiveTree(_RankLocalTree):
    """An upstream-style tree: loading_check MIN-reduces over the group."""

    LOADING_CHECK_RANK_LOCAL = False


def _sched(tree, hicache=True):
    return types.SimpleNamespace(enable_hierarchical_cache=hicache, tree_cache=tree)


def _end(fn: str) -> int:
    ends = [i for i in (fn.find("\n    def ", 10), fn.find("\n    async def ", 10)) if i > 0]
    return min(ends) if ends else len(fn)


def _blockers(sched):
    # the clause idle_blockers()/is_fully_idle() read for load-backs
    n = len(sched.tree_cache.ongoing_load_back)
    return [f"hicache_load_back({n})"] if n else []


# -- D: the vote drains the finished ack -------------------------------------


def test_repro_parked_d_votes_busy_forever_without_the_drain():
    # the 215911 state: nothing running, nothing queued, one finished ack
    s = _sched(_RankLocalTree(finished=True))
    for _ in range(1602):  # every refused flush of the 31 s
        assert _blockers(s) == ["hicache_load_back(1)"]
    assert s.tree_cache.calls == 0  # no planned pass ever polled it


def test_the_vote_drains_a_finished_ack_and_the_group_is_idle():
    s = _sched(_RankLocalTree(finished=True))
    assert ql.drain_finished_loadbacks(s, env={}) == 1
    assert _blockers(s) == []


def test_an_unfinished_copy_still_votes_busy():
    s = _sched(_RankLocalTree(finished=False))
    assert ql.drain_finished_loadbacks(s, env={}) == 0
    assert _blockers(s) == ["hicache_load_back(1)"]
    assert s.tree_cache.calls == 1  # polled, nothing waited on


def test_a_collective_loading_check_is_never_called_from_the_vote():
    s = _sched(_CollectiveTree(finished=True))
    assert ql.drain_finished_loadbacks(s, env={}) == 0
    assert s.tree_cache.calls == 0


def test_switch_off_restores_the_old_vote():
    s = _sched(_RankLocalTree(finished=True))
    assert ql.drain_finished_loadbacks(s, env={ql.ENV: "0"}) == 0
    assert s.tree_cache.calls == 0
    assert ql.enabled({}) is True  # default ON


def test_no_hicache_no_poll():
    s = _sched(_RankLocalTree(finished=True), hicache=False)
    assert ql.drain_finished_loadbacks(s, env={}) == 0
    assert s.tree_cache.calls == 0


def test_a_raising_check_never_breaks_the_flush():
    t = _RankLocalTree()

    def boom():
        raise RuntimeError("x")

    t.loading_check = boom
    assert ql.drain_finished_loadbacks(_sched(t), env={}) == 0


def test_unified_radix_cache_declares_its_check_rank_local():
    src = open(URC).read()
    assert re.search(r"^\s+LOADING_CHECK_RANK_LOCAL = True$", src, re.M)
    body = src[src.index("    def loading_check(self) -> None:"):]
    body = body[: _end(body)]
    # the claim the attribute makes: no reduction, only this rank's own acks (#737)
    assert "_count_ready_acks(" in body
    assert "all_reduce" not in body


def test_flush_cache_drains_before_its_group_vote():
    src = open(SCHED).read()
    fn = src[src.index("    def flush_cache("):]
    fn = fn[: _end(fn)]
    i_drain = fn.index("_pdflip_quiesce_loadback.drain_finished_loadbacks(self)")
    i_vote = fn.index("self.group_idle_verdict(")
    assert i_drain < i_vote


# -- front: both witnesses busy at the deadline aborts the flip --------------


def _front_funcs():
    src = open(FRONT).read()
    ns = {"List": list, "Optional": __import__("typing").Optional, "os": os}
    m = re.search(r'^QUIESCE_BUSY_ABORT_ENV = "([A-Z_]+)"$', src, re.M)
    ns["QUIESCE_BUSY_ABORT_ENV"] = m.group(1)
    for name in ("witness_verdict", "quiesce_busy_abort_enabled", "quiesce_busy_both", "quiesce_busy_abort"):
        i = src.index(f"def {name}(")
        j = src.index("\n\n\n", i)
        exec(src[i:j], ns)  # noqa: S102 -- the module's own pure functions, not the 25k-line import
    return ns["witness_verdict"], ns["quiesce_busy_abort"]


def _front_ns():
    witness_verdict, abort = _front_funcs()
    return abort.__globals__


def test_repro_both_busy_reads_as_agreement():
    witness_verdict, _ = _front_funcs()
    # 22:30:55: front ledger ['pdflip-4-207'], group not idle -> no W3, the legs ran
    assert witness_verdict(1, False) is None


def test_both_busy_aborts_and_nothing_else_does():
    witness_verdict, abort = _front_funcs()
    assert abort(1, False) is True
    assert abort(0, True) is False   # the normal flip
    assert abort(0, False) is False  # W3 "front drained, rank NOT idle" stays W3
    assert abort(1, True) is False   # W3 / park-lapse race stays as it was
    # every state that continues to the legs is idle afterwards
    for out in (0, 1, 3):
        for idle in (False, True):
            if witness_verdict(out, idle) is None and not abort(out, idle):
                assert idle


def test_flip_checks_the_abort_before_any_sleep_rpc():
    src = open(FRONT).read()
    # the Front's flip (front.py carries more than one ``async def flip``): the one with the quiesce
    q = src.index("idle, msg = await self.quiesce(S)")
    fn = src[src.rindex("    async def flip(self, src: str, dst: str) -> None:", 0, q):]
    fn = fn[: _end(fn)]
    i_q = fn.index("idle, msg = await self.quiesce(S)")
    i_abort = fn.index("if quiesce_busy_abort(len(_wl), idle):")
    i_early = fn.index("if _early is not None:\n            # EARLY-FLIP: drain + quiesce ran")
    assert i_q < i_abort < i_early
    block = fn[i_abort:i_early]
    assert "Front._early_flip_abort(" in block and "return" in block
    assert "self.admit_d = True" in block
    assert "release_memory_occupation" not in block


# -- 27B port 1024: the abort carries its own env gate (SCHNITT5 review RISK) --


def test_abort_gate_is_named_and_default_on():
    ns = _front_ns()
    assert ns["QUIESCE_BUSY_ABORT_ENV"] == "FLLIPER_PDFLIP_QUIESCE_BUSY_ABORT"
    assert ns["quiesce_busy_abort_enabled"]({}) is True
    assert ns["quiesce_busy_abort"](1, False, env={}) is True


@pytest.mark.parametrize("off", ["0", "false", "no", "off", "OFF"])
def test_abort_gate_off_restores_the_old_both_busy_reading(off):
    ns = _front_ns()
    env = {"FLLIPER_PDFLIP_QUIESCE_BUSY_ABORT": off}
    assert ns["quiesce_busy_abort_enabled"](env) is False
    assert ns["quiesce_busy_abort"](1, False, env=env) is False
    # the gate-free predicate still names the W120 form for the gated-off warning
    assert ns["quiesce_busy_both"](1, False) is True


def test_abort_gate_never_turns_a_non_busy_state_into_an_abort():
    ns = _front_ns()
    for env in ({}, {"FLLIPER_PDFLIP_QUIESCE_BUSY_ABORT": "1"}, {"FLLIPER_PDFLIP_QUIESCE_BUSY_ABORT": "0"}):
        for out, idle in ((0, True), (0, False), (1, True), (3, True)):
            assert ns["quiesce_busy_abort"](out, idle, env=env) is False


def test_flip_names_a_gated_off_both_busy_before_the_abort_check():
    src = open(FRONT).read()
    q = src.index("idle, msg = await self.quiesce(S)")
    fn = src[src.rindex("    async def flip(self, src: str, dst: str) -> None:", 0, q):]
    fn = fn[: _end(fn)]
    i_off = fn.index("if not quiesce_busy_abort_enabled() and quiesce_busy_both(len(_wl), idle):")
    i_abort = fn.index("if quiesce_busy_abort(len(_wl), idle):")
    assert i_off < i_abort
    assert 'self.counters["quiesce_busy_abort_gated_off"] += 1' in fn[i_off:i_abort]
    assert "return" not in fn[i_off:i_abort]  # gated off = the old path runs on
