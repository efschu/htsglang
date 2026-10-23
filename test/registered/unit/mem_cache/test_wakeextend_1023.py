"""WAKEEXT-1023 (desk/27b-wakeextend-1023): the two open items of the 27B AutoRound flip-time report.

(1) S109-A5-N1 ``L3WB-PUBLISH-RACE`` 13867x on P PP0 / 4992x on D TP0 (boot 1023_140032) with NO L3 eviction in the
    boot: the re-read after the publish missed because the SIBLING group's write-behind had completed the file (this
    pass: status 2, ``adopt_foreign``) and not yet listed it in #1459 (its commit follows its whole C chunk). The
    void then wrote -1 into the shared record. Fix: add-then-verify -- a stem whose file is on disk is listed and
    kept; a stem whose file is gone is struck and voided as before (the 20.10. Q1 race stays covered).
(2) The first D extend after a wake: TP0-only CPU contention; the wake's async store rescan (journal apply, owner
    rank only) is deferred until the first forwards are launched (``pdflip.wake_cpu_defer``), and the first-forward
    instrument names every thread's CPU over the forward window.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


# ------------------------------------------------------------------------------------------------ (1) A5-N1 heal
def _boot_capped(tmp_path, monkeypatch):
    """The real backend of the 20.10. re-review (C arenas, L3Index, LRUFileEvictor with a cap -> ``owns`` and
    ``adopt_foreign`` are live)."""
    import test_rereview_s109_a5a8_1020 as RV

    # one uninterrupted pass: the 25-ms slice cuts a pass short on a loaded desk (CPUQuota) -- a timing flake, not
    # the behaviour under test
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_SLICE_MS", "0")
    return RV._boot(tmp_path, monkeypatch, cap=1 << 40)


def _sibling_completed_unlisted(be, RR, h):
    """The sibling group's write-behind completed ``h``'s file (rename done) and has not yet run its commit: the
    file is on disk, #1459 does not list it, this owner does not own it, and the page is ``new`` here."""
    from flliper.srt.mem_cache import l3_secured as L3S

    kv = be._arenas[RR.KV_TOTAL]
    victim = be._get_suffixed_key(h)
    (slot, _st), = kv.find_slots([victim])
    assert os.path.exists(be._existing_path(victim))
    be._l3idx.remove([victim])
    ev = be._evictor
    with ev._lock:
        sz = ev._lru.pop(victim)
        ev._total_bytes -= sz
    assert not ev.owns(victim)
    kv._l3wb_sec[0][slot] = -1
    L3S.arrays(kv)[0][slot] = -1
    return kv, victim, slot


@_needs_gcc
def test_a5n1_index_lag_of_the_sibling_is_healed_not_voided(tmp_path, monkeypatch):
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    monkeypatch.delenv("FLLIPER_PDFLIP_L3WB_PUBLISH_HEAL", raising=False)
    RR, be = _boot_capped(tmp_path, monkeypatch)
    hs = RR._hashes("s")
    RR._prefill(be, hs)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES
    kv, victim, slot = _sibling_completed_unlisted(be, RR, hs[0])
    v0 = int(getattr(HiCacheFile, "_a5n1_voided", 0))
    h0 = int(getattr(HiCacheFile, "_a5n1_healed", 0))
    r = be.l3_write_behind_pass()
    assert r.get("foreign_done", 0) == 1, r                     # status 2, not this owner's: adopt_foreign
    assert int(getattr(HiCacheFile, "_a5n1_voided", 0)) == v0, "no void: the file IS on disk"
    assert int(getattr(HiCacheFile, "_a5n1_healed", 0)) == h0 + 1
    assert be._l3idx.has([victim]) == [True], "listed now (the sibling's commit would list it too)"
    assert L3S.secured(kv, [slot], be).tolist() == [True], "the shared record keeps the copy"
    assert int(kv._l3wb_sec[0][slot]) >= 0
    assert be.l3_write_behind_pass()["written"] == 0, "nothing to copy again"


@_needs_gcc
def test_a5n1_switch_off_reproduces_the_false_void(tmp_path, monkeypatch):
    """The 20.10. form (FLLIPER_PDFLIP_L3WB_PUBLISH_HEAL=0) on the same shape: voided although nothing was evicted --
    the 13867 lines of P PP0."""
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    monkeypatch.setenv("FLLIPER_PDFLIP_L3WB_PUBLISH_HEAL", "0")
    RR, be = _boot_capped(tmp_path, monkeypatch)
    hs = RR._hashes("s")
    RR._prefill(be, hs)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES
    kv, victim, slot = _sibling_completed_unlisted(be, RR, hs[0])
    v0 = int(getattr(HiCacheFile, "_a5n1_voided", 0))
    be.l3_write_behind_pass()
    assert int(getattr(HiCacheFile, "_a5n1_voided", 0)) == v0 + 1
    assert os.path.exists(be._existing_path(victim)), "the file was never gone"
    assert L3S.secured(kv, [slot], be).tolist() == [False]


@_needs_gcc
def test_a5n1_a_file_the_lru_took_is_still_voided_and_struck(tmp_path, monkeypatch):
    """The heal never keeps a page whose file is gone: unlisted AND not on disk -> struck from #1459, voided."""
    from flliper.srt.mem_cache import l3_secured as L3S
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    monkeypatch.delenv("FLLIPER_PDFLIP_L3WB_PUBLISH_HEAL", raising=False)
    RR, be = _boot_capped(tmp_path, monkeypatch)
    hs = RR._hashes("s")
    RR._prefill(be, hs)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES
    kv = be._arenas[RR.KV_TOTAL]
    victim = be._get_suffixed_key(hs[1])
    (slot, _), = kv.find_slots([victim])
    v0 = int(getattr(HiCacheFile, "_a5n1_voided", 0))
    os.remove(be._existing_path(victim))
    be._l3idx.remove([victim])
    n = be._l3wb_publish(kv, np.asarray([slot]), np.asarray(kv.slot_gens_np([slot])), [victim], kv._l3wb_sec[0])
    assert n == 1 and int(getattr(HiCacheFile, "_a5n1_voided", 0)) == v0 + 1
    assert be._l3idx.has([victim]) == [False], "the heal's add is struck again (file gone)"
    assert L3S.secured(kv, [slot], be).tolist() == [False]


def test_a5n1_lines_are_bounded_per_window_and_counters_stay_exact(monkeypatch, caplog):
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    be = object.__new__(HiCacheFile)
    monkeypatch.setattr(HiCacheFile, "_a5n1_logwin", {}, raising=False)
    t = [1000.0]
    monkeypatch.setattr("flliper.srt.mem_cache.hicache_storage.time.monotonic", lambda: t[0])
    oks = [be._l3wb_publish_log_ok("race") for _ in range(50)]
    assert sum(oks) == HiCacheFile._A5N1_LOG_PER_WINDOW
    t[0] += HiCacheFile._A5N1_LOG_WINDOW_S + 0.1
    with caplog.at_level("INFO"):
        assert be._l3wb_publish_log_ok("race")
    assert any("suppressed=%d" % (50 - HiCacheFile._A5N1_LOG_PER_WINDOW) in r.getMessage() for r in caplog.records)
    assert be._l3wb_publish_log_ok("heal"), "kinds have their own window"


def test_a5n1_publish_reads_back_after_the_publish_and_heals_before_the_void():
    import inspect

    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    src = inspect.getsource(HiCacheFile._l3wb_publish)
    assert src.index("_l3s.publish(arena, slots, gens)") < src.index("idx.has(") \
        < src.index("_l3wb_publish_heal(") < src.index("base_gen[_b] = -1")
    hsrc = inspect.getsource(HiCacheFile._l3wb_publish_heal)
    assert hsrc.index("idx.add(miss)") < hsrc.index("self._stat_stems(miss)"), "add THEN verify (linearizable)"


# ------------------------------------------------------------------------------------------------ (2) wake deferral
@pytest.fixture
def wcd(monkeypatch):
    from flliper.srt.pdflip import wake_cpu_defer as W

    for k in (W.ENV, W.ENV_MAX_MS, W.ENV_EXTENDS):
        monkeypatch.delenv(k, raising=False)
    W._reset_for_tests()
    yield W
    W._reset_for_tests()


def test_wake_defer_is_on_by_default_and_released_after_two_extend_launches(wcd):
    assert wcd.arm(has_work=True) and wcd.armed()
    wcd.note_pass("EXTEND", 3)
    assert wcd.armed()
    wcd.note_pass("EXTEND", 0)
    assert not wcd.armed() and wcd.state()["why"] == "extends=2"


def test_wake_defer_counts_only_prefill_launch_names(wcd):
    wcd.arm(has_work=True)
    wcd.note_pass("TARGET_VERIFY", 1)
    wcd.note_pass("IDLE", 1)
    assert wcd.armed() and wcd.state()["extends"] == 0
    wcd.note_pass("SPLIT_PREFILL", 1)
    wcd.note_pass("MIXED", 1)
    assert not wcd.armed()


def test_wake_defer_released_by_a_decode_launch_with_nothing_left_to_extend(wcd):
    wcd.arm(has_work=True)
    wcd.note_pass("DECODE", 2)
    assert wcd.armed(), "requests still queued for their extend"
    wcd.note_pass("DECODE", 0)
    assert not wcd.armed() and wcd.state()["why"].startswith("no_extend_left")


def test_wake_defer_not_armed_without_work_or_switched_off(wcd, monkeypatch):
    assert not wcd.arm(has_work=False) and not wcd.armed()
    monkeypatch.setenv(wcd.ENV, "0")
    assert not wcd.arm(has_work=True) and not wcd.armed()
    assert wcd.wait("x") == 0.0


def test_wake_defer_wait_is_bounded_and_released_by_a_launch(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "60")
    wcd.arm(has_work=True)
    t0 = time.monotonic()
    held = wcd.wait("bound")
    assert 0.04 <= held < 1.0 and time.monotonic() - t0 < 1.0
    assert wcd.state()["why"] == "bound"
    monkeypatch.setenv(wcd.ENV_MAX_MS, "5000")
    wcd.arm(has_work=True)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("h", wcd.wait("rescan")))
    th.start()
    time.sleep(0.05)
    assert th.is_alive()
    wcd.note_pass("EXTEND", 0)
    wcd.note_pass("EXTEND", 0)
    th.join(2.0)
    assert not th.is_alive() and out["h"] < 2.0


def _updater_stub():
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as M

    calls = []
    st = types.SimpleNamespace(pdflip_store_rescan_thread=None)
    st._pdflip_rescan_store_index_sync = lambda: calls.append(time.monotonic())
    for name in ("_pdflip_rescan_store_index", "_pdflip_join_store_rescan"):
        setattr(st, name, types.MethodType(getattr(M, name), st))
    return st, calls


def test_the_async_rescan_waits_for_the_first_launches(wcd, monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_RESCAN_ASYNC", raising=False)
    st, calls = _updater_stub()
    st._pdflip_wake_defer_work = True
    st._pdflip_rescan_store_index()
    assert wcd.armed()
    time.sleep(0.1)
    assert calls == [], "the journal apply waits until the wake's first forwards are launched"
    t_rel = time.monotonic()
    wcd.note_pass("EXTEND", 0)
    wcd.note_pass("EXTEND", 0)
    st.pdflip_store_rescan_thread.join(2.0)
    assert len(calls) == 1 and calls[0] >= t_rel


def test_a_wake_without_work_starts_the_rescan_at_once(wcd, monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_RESCAN_ASYNC", raising=False)
    st, calls = _updater_stub()
    st._pdflip_wake_defer_work = False
    st._pdflip_rescan_store_index()
    st.pdflip_store_rescan_thread.join(2.0)
    assert len(calls) == 1 and not wcd.armed()


def test_the_wake_rescan_join_of_the_previous_thread_does_not_undo_the_new_arm(wcd, monkeypatch):
    """Review 1023: a previous rescan thread still registered (no release-leg join in between) is joined by the
    next wake's rescan start; that join releases -- the arm must come after it."""
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_RESCAN_ASYNC", raising=False)
    st, calls = _updater_stub()
    st._pdflip_wake_defer_work = False
    st._pdflip_rescan_store_index()            # wake 1: no work, runs at once, thread stays registered
    st.pdflip_store_rescan_thread.join(2.0)
    st._pdflip_wake_defer_work = True
    st._pdflip_rescan_store_index()            # wake 2: joins wake 1's thread, then arms
    assert wcd.armed(), "the join's release ran before this wake's arm"
    time.sleep(0.05)
    assert len(calls) == 1
    wcd.release("test")
    st.pdflip_store_rescan_thread.join(2.0)
    assert len(calls) == 2


def test_the_release_leg_join_never_waits_for_the_deferral(wcd, monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_RESCAN_ASYNC", raising=False)
    monkeypatch.setenv(wcd.ENV_MAX_MS, "30000")
    st, calls = _updater_stub()
    st._pdflip_wake_defer_work = True
    st._pdflip_rescan_store_index()
    assert wcd.armed()
    t0 = time.monotonic()
    st._pdflip_join_store_rescan("release")
    assert time.monotonic() - t0 < 2.0 and len(calls) == 1
    assert st.pdflip_store_rescan_thread is None


def test_the_wake_arms_the_deferral_before_the_rescan_and_every_launch_loop_notes():
    import inspect

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers import scheduler_pp_mixin as PPM
    from flliper.srt.managers.scheduler_components import weight_updater as WU

    src = inspect.getsource(WU)
    i_work = src.index("self._pdflip_wake_defer_work = bool(_rw_work)")
    assert i_work < src.index("self._pdflip_rescan_store_index()\n", i_work)
    rsrc = inspect.getsource(WU.SchedulerWeightUpdaterManager._pdflip_rescan_store_index)
    assert rsrc.index("self._pdflip_join_store_rescan(") < rsrc.index("_wcd.arm(") < rsrc.index("th.start()"), (
        "armed after the join: the join's release must not undo this wake's arm")
    assert "target=_pdflip_rescan_deferred, args=(self,)" in src
    assert src.index("_wcd.wait(\"store_rescan\")") < src.index("updater._pdflip_rescan_store_index_sync()")
    ssrc = inspect.getsource(S)
    assert ssrc.count("_pdflip_wake_defer_note(self, batch)") == 2, "overlap and normal loop"
    assert "_wcd.note_pass(" in inspect.getsource(PPM)


# ------------------------------------------------------------------------------------------------ (2) instrument
def test_thread_cpu_snapshot_reads_proc_and_names_python_threads():
    from flliper.srt.pdflip import first_fwd_timing as F

    if not os.path.isdir("/proc/self/task"):
        pytest.skip("no procfs")
    stop = threading.Event()

    def _burn():
        while not stop.is_set():
            sum(range(2000))

    th = threading.Thread(target=_burn, name="l3wb-test-burner", daemon=True)
    th.start()
    try:
        a = F.thread_cpu_snapshot()
        time.sleep(0.15)
        b = F.thread_cpu_snapshot()
    finally:
        stop.set()
        th.join(1.0)
    me = threading.get_native_id()
    assert me in a and a[me][0] == threading.current_thread().name
    line = F.thread_cpu_delta(a, b, 0.15, own_tid=me)
    assert line.startswith("cpu[wall_ms=150 ") and "l3wb-test-burner:" in line, line


def test_thread_cpu_delta_from_a_fake_task_dir(tmp_path):
    from flliper.srt.pdflip import first_fwd_timing as F

    def _task(tid, comm, ns):
        d = tmp_path / str(tid)
        d.mkdir(exist_ok=True)
        (d / "schedstat").write_text("%d 0 0\n" % ns)
        (d / "comm").write_text(comm + "\n")

    _task(11, "pt_main", 1_000_000)
    _task(12, "rescan", 5_000_000)
    a = F.thread_cpu_snapshot(str(tmp_path))
    _task(11, "pt_main", 41_000_000)
    _task(12, "rescan", 305_000_000)
    _task(13, "late", 7_000_000)                      # started inside the window: counts from 0
    b = F.thread_cpu_snapshot(str(tmp_path))
    line = F.thread_cpu_delta(a, b, 0.5, own_tid=11)
    assert line == "cpu[wall_ms=500 sched_ms=40 others_ms=307 top=native:rescan:300,native:late:7]", line


def test_first_forward_record_carries_the_thread_cpu(monkeypatch):
    from flliper.srt.pdflip import first_fwd_timing as F

    class _Ev:
        def record(self, *a, **k):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(F.S, "event_factory", _Ev)
    monkeypatch.setattr(F.S, "available", lambda: True)
    monkeypatch.setenv(F.GIL_ENV, "off")
    F.S.pending, F.S.cur, F.S.left = [], None, 0
    F.arm("test")
    F.on_set_consumer(-1)
    assert "cpu0" in F.S.cur
    for lid in range(4):
        F.timed_wait(F.NO_WAIT, lid)
    rec = F.S.cur
    F.S.left = 0
    F.on_set_consumer(-1)
    assert rec.get("cpu", "").startswith("cpu[wall_ms=")
    line = F.summarize(rec)
    assert line is not None and " cpu[wall_ms=" in line
    F.S.pending, F.S.cur = [], None


# ------------------------------------------------------------------------------------------------ write path (Planer)
def _pio_or_skip():
    from flliper.srt.mem_cache.storage.file.pageio import load

    pio = load()
    if pio is None or not pio.has_touch_paths:
        pytest.skip("pageio helper not buildable here")
    return pio


@_needs_gcc
def test_pageio_touch_paths_bumps_mtime_in_one_call(tmp_path):
    pio = _pio_or_skip()
    files = []
    for i in range(5):
        f = tmp_path / ("p%d.bin" % i)
        f.write_bytes(b"x")
        os.utime(f, (1_000_000, 1_000_000))
        files.append(str(f))
    st = pio.touch_paths(files + [str(tmp_path / "gone.bin")])
    assert st == [0, 0, 0, 0, 0, 1]
    assert all(os.stat(f).st_mtime > 1_000_000 + 1e6 for f in files)


class _CountingEvictor:
    enabled = True

    def __init__(self):
        self.calls = []

    def touch(self, stem, path, mtime=True):
        self.calls.append((stem, path, mtime))


def _bare_backend(tmp_path, flat=False):
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    be = object.__new__(HiCacheFile)
    be.file_path = str(tmp_path)
    be._known_shards = set()
    be._legacy_flat = bool(flat)
    be._evictor = _CountingEvictor()
    be.n_stat = 0
    orig = HiCacheFile._stat_stem

    def _counted(self_, stem):
        self_.n_stat += 1
        return orig(self_, stem)

    be._stat_stem = types.MethodType(_counted, be)
    return be


@_needs_gcc
def test_write_path_touch_of_present_pages_needs_no_python_stat(tmp_path, monkeypatch):
    pio = _pio_or_skip()
    monkeypatch.delenv("FLLIPER_HICACHE_L3_WRITE_NOSTAT", raising=False)
    be = _bare_backend(tmp_path)
    stems = ["%064x_sfx" % i for i in range(64)]
    for st in stems:
        p = be._sharded_path(st)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"y")
        os.utime(p, (1_000_000, 1_000_000))
    calls = []
    monkeypatch.setattr(type(pio), "touch_paths", lambda self_, ps: calls.append(list(ps)) or [0] * len(ps))
    be._touch_present(stems)
    assert be.n_stat == 0, "no Python stat per page (sharded-only store)"
    assert len(calls) == 1 and calls[0] == [be._sharded_path(st) for st in stems], "ONE C utime call"
    assert be._evictor.calls == [(st, be._sharded_path(st), False) for st in stems], "LRU + journal as before"


def test_write_path_with_a_legacy_flat_layout_takes_the_batched_stat_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("FLLIPER_HICACHE_L3_WRITE_NOSTAT", raising=False)
    be = _bare_backend(tmp_path, flat=True)
    pm = {"a_sfx": "/x/a.bin"}
    paths = be._known_paths(["a_sfx", "b_sfx"], pm)
    assert paths[0] == "/x/a.bin" and be.n_stat == 1, "a stem the map does not name falls back to one stat"


def test_write_path_switch_off_is_the_per_page_stat(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_L3_WRITE_NOSTAT", "0")
    be = _bare_backend(tmp_path)
    be._touch_present(["a_sfx", "b_sfx"])
    assert be.n_stat == 2
    assert [c[2] for c in be._evictor.calls] == [True, True]


def test_batch_set_each_touches_from_the_batched_stat_and_the_durchlauf_resolves_without_stat():
    import inspect

    from flliper.srt.managers import cache_controller as CC
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    src = inspect.getsource(HiCacheFile._batch_set_each)
    assert "self._evictor.touch(suffixed, self._existing_path(suffixed))" not in src
    i_arm = src.index("_FILL_TL.paths = {} if write_nostat_on() else None")
    assert i_arm < src.index("present = self._stat_stems(") < src.index("self._touch_present(")
    assert "_kv_dl_paths(be, [stems[j] for j in pos])" in inspect.getsource(CC._kv_dl_resolve)

    class _Be:
        def __init__(self):
            self.n = 0

        def _existing_path(self, st):
            self.n += 1
            return "/old/" + st

    b = _Be()
    assert CC._kv_dl_paths(b, ["a", "b"]) == ["/old/a", "/old/b"] and b.n == 2, "a double keeps the old path"
