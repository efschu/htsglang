# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-H (02.10.2026): stale and orphaned #243 hand-off marks.

(1) STALE VIEW. P's reset census (ARENA-REF-HOLDERS at=reset, inside every
    sleep flush) printed ``handoff_kept=90691/90691 handoff_rids=3`` from the
    first reset of N3u (1002_072908, 07:33:18) to the end of the boot: 90691 =
    weg2-0-1 (45394) + weg2-0-4 (45279) + weg2-1-5 (18), all ``#243
    HANDOFF-PENDING CONSUMED`` on D at 07:33:21-22. N4p (1002_095319): 95986 =
    weg2-0-1 + weg2-0-2 + weg2-1-3, all consumed, same freeze. The census read
    the pool's cached keep and never re-read it (P's FULL arena never filled,
    so no claim refreshed it) -- and looked every stale key up per reset.
(2) ORPHAN. weg2-0-2 (N3u): WEG2-CLIENT-GONE abort-p 07:32:46.174, the
    rid-end drop found no mark, P published and marked the hand-off at
    07:32:47 -- D kept 117304 keys for the 900 s bound (every D reset census,
    a 1.1 GB PARK-DEMOTE L3 copy). A terminal drop now leaves a tombstone; a
    later mark is refused by name. Write-then-check against
    tombstone-then-unlink: no interleaving (and no rank order -- one shared
    file) leaves the orphan.

Hermetic, CPU. Each test named red_* is red on 120796f0af.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.weg2 import handoff as ho  # noqa: E402
from sglang.srt.weg2 import handoff_pending as hp  # noqa: E402

CHAIN = ["k0", "k1", "k2"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("SGLANG_WEG2_HANDOFF", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    (tmp_path / "arena").mkdir()
    return types.SimpleNamespace(mp=monkeypatch, tmp=tmp_path)


def _pool():
    p = types.SimpleNamespace(arena=None)
    p._stems = lambda chain: [f"{k}.sfx" for k in chain]
    return p


def _pending(rid):
    return os.path.exists(os.path.join(hp._sub(hp.PENDING), rid))


def _p_publishes(rid, chain=CHAIN):
    assert ho.write(rid, list(range(len(chain) + 1)), chain)
    return hp.mark(rid, len(chain), 1)


def test_red_a_mark_after_the_rid_end_is_refused(env):
    """N3u weg2-0-2: the client went away during P's leg 1; the front's rid-end
    drop ran first, P's mark came after it."""
    assert hp.drop("weg2-0-2", "disconnect") is False      # nothing to drop yet
    assert _p_publishes("weg2-0-2") is False
    assert not _pending("weg2-0-2")
    assert hp.status("weg2-0-2")["state"] == "none"


def test_a_mark_before_the_rid_end_is_dropped_as_before(env):
    assert _p_publishes("weg2-1-1") is True and _pending("weg2-1-1")
    assert hp.drop("weg2-1-1", "served") is True
    assert not _pending("weg2-1-1")


def test_a_reroute_is_not_a_rid_end(env):
    """front.py reroute_fresh: the rid goes back to P and is marked again."""
    assert _p_publishes("weg2-2-2") is True
    hp.drop("weg2-2-2", "reroute_fresh")
    assert not _pending("weg2-2-2")
    assert _p_publishes("weg2-2-2") is True and _pending("weg2-2-2")


def test_red_a_park_after_the_rid_end_is_refused(env):
    env.mp.setenv("SGLANG_WEG2_GROUP", "D")
    hp.drop("weg2-3-3", "abort:RuntimeError")
    assert hp.mark_park("weg2-3-3", CHAIN, 1) is False
    assert not os.path.exists(os.path.join(hp._sub(hp.PARK), "weg2-3-3"))


def test_the_check_after_write_closes_the_race(env, monkeypatch):
    """The drop lands between the mark's write and its check: the mark goes."""
    real = hp._write_atomic

    def write_then_drop(path, rec):
        ok = real(path, rec)
        if os.path.dirname(path) == hp._sub(hp.PENDING):
            hp._tombstone("weg2-4-4")   # the front's rid end, right after the write
        return ok

    monkeypatch.setattr(hp, "_write_atomic", write_then_drop)
    assert _p_publishes("weg2-4-4") is False and not _pending("weg2-4-4")


def test_tombstones_older_than_the_bound_are_pruned(env, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_HANDOFF_PENDING_EXPIRE_S", "60")
    hp.drop("weg2-old", "served")
    old = hp._ended_path("weg2-old")
    os.utime(old, (1, 1))
    monkeypatch.setattr(hp, "_DROPS", [hp.ENDED_PRUNE_EVERY - 1])
    hp.drop("weg2-new", "served")
    assert not os.path.exists(old) and os.path.exists(hp._ended_path("weg2-new"))


def test_red_the_reset_census_does_not_print_consumed_hand_offs(env):
    """N3u P FULL: 90691/90691 rids=3 at every reset after all three were consumed."""
    pool = _pool()
    for rid in ("weg2-0-1", "weg2-0-4", "weg2-1-5"):
        assert _p_publishes(rid)
    assert len(hp.keep_for(pool)) == 9                     # the first reset's view
    assert "handoff_kept=0/9 handoff_rids=3" in hp.census(pool, refresh=False)
    for rid in ("weg2-0-1", "weg2-0-4", "weg2-1-5"):
        hp.consume(rid, "fetch")                           # D took them
    line = hp.census(pool, refresh=False)
    assert "handoff_kept=0/0 handoff_rids=0" in line and "unread_marks=0" in line


def test_red_the_reset_census_never_hashes_a_new_hand_off(env, monkeypatch):
    """A new 45k hand-off costs ~76 ms in keep_for -- not on the flush path."""
    pool = _pool()
    assert _p_publishes("weg2-6-10")
    monkeypatch.setattr(hp, "keep_for", lambda _p: (_ for _ in ()).throw(AssertionError("hashed at reset")))
    line = hp.census(pool, refresh=False)
    assert "handoff_kept=0/0 handoff_rids=0" in line and "unread_marks=1" in line


def test_the_census_thread_re_reads_a_changed_mark_directory(env):
    pool = _pool()
    assert _p_publishes("weg2-7-1")
    hp.keep_for(pool)
    assert _p_publishes("weg2-7-2")
    st = os.stat(hp._sub(hp.PENDING))                       # coarse-mtime filesystems: a new stamp
    os.utime(hp._sub(hp.PENDING), ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    line = hp.census(pool)                                  # refresh=True
    assert "handoff_kept=0/6 handoff_rids=2" in line and "unread_marks=0" in line


def test_the_reset_passes_refresh_false():
    import inspect

    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache._weg2_log_holder_census)
    assert 'refresh=where != "reset"' in src
