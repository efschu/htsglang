"""fnFL2x81 (23.09.): the mid-legs kv resume is a GROUP verdict.

Bug regression (xsn377, 18.09.): the per-rank decision differed (TP1 6.5 GB
funded at the first tag, TP0 12.3 GB never), the preload then moved ONE
rank's prefixes and the first extend died on PrefixLensRankDivergence; the
hook was switched off by default. Now every rank votes after every tag, a
rank whose pool is not outstanding votes True, and the group resumes only
when every vote is True -- the tightest rank rules, the same tag on every
rank. Hermetic: pure decisions, no process group.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import wake_kv as wk  # noqa: E402

MIB = 1 << 20


def test_a_rank_without_an_outstanding_pool_votes_true_whatever_its_card_says():
    """Deletion leaves the silent path: a worker whose pool is already back
    would otherwise veto the host rank forever (kv_bytes 0 -> kv_mid_ok False)."""
    assert wk.kv_mid_vote(outstanding=False, free_bytes=0, floor_bytes=0, kv_bytes=0,
                          remaining_bytes=10 * MIB) is True
    assert wk.kv_mid_vote(outstanding=True, free_bytes=0, floor_bytes=0, kv_bytes=0,
                          remaining_bytes=10 * MIB) is False


def test_the_vote_of_an_outstanding_pool_is_the_fit_after_the_remaining_tags():
    # x80 TP0 after tag 6 of 10: ~9,5 GB free, floor 767, kv 4286, 4 tags a 0,9 GB
    free, floor, kv, rest = 9500 * MIB, 767 * MIB, 4286 * MIB, 3600 * MIB
    assert wk.kv_mid_vote(outstanding=True, free_bytes=free, floor_bytes=floor,
                          kv_bytes=kv, remaining_bytes=rest) is True
    # after tag 2: 6,3 GB free, 8 tags left -> wait
    assert wk.kv_mid_vote(outstanding=True, free_bytes=6300 * MIB, floor_bytes=floor,
                          kv_bytes=kv, remaining_bytes=7200 * MIB) is False


def test_the_group_verdict_is_the_tightest_ranks():
    """xsn377's shape: TP1/TP2 funded, TP0 not -> the group waits; only a
    unanimous round resumes. An empty gather resumes nothing."""
    assert wk.kv_mid_uniform([True, True, True]) is True
    assert wk.kv_mid_uniform([False, True, True]) is False
    assert wk.kv_mid_uniform([True, False, True]) is False
    assert wk.kv_mid_uniform([]) is False


def test_the_hook_is_on_by_default_and_the_env_still_switches_it_off():
    assert wk.kv_mid_on({}) is True
    assert wk.kv_mid_on({wk.KV_MID_ENV: "0"}) is False
    assert wk.KV_MID_ENV == "SGLANG_WEG2_WAKE_KV_MID"


def test_the_resume_loop_gathers_the_votes_before_it_decides():
    """Bookkeeping: the wake handler must ask the group (``_weg2_group_votes``)
    and hand the gather to ``kv_mid_uniform``; a per-rank verdict is xsn377."""
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(wu.__file__).read()
    assert "_votes = self._weg2_group_votes(bool(_vote))" in src
    assert "_wk.kv_mid_uniform(_votes)" in src
    assert "_weg2_kv_mid_settled = bool(_mid_ok)" in src
