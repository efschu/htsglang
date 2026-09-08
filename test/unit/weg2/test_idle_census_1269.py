"""#1269: malloc arenas bounded, and the per-rank idle census line.

Boot weg2sb4 grew ~19.0 MiB/min of host anon across its six ranks while
completely idle (P PP0 +5.4, PP1 +5.4, PP2 +0.0, each D rank +2.8), in
64 MiB-aligned glibc malloc arenas -- not the host ring (dShmem was 0.0 on
every process), not CUDA pinned memory, not the store.  Neither the round
rate nor the growth appeared in the log; both had to be reconstructed from
outside, and the processes were gone to a host OOM minutes later.

These tests pin the launcher half (the env the ranks are actually given) and
the instrument half (the line the next boot's acceptance reads).
"""

import os

from sglang.srt.weg2.idle_census import (
    WEG2_TRIM_MIN_GROWTH_MIB,
    IdleCensus,
    _read_status,
)


# ------------------------------------------------------------ launcher env
def _build_env(**kw):
    from sglang.srt.weg2.launcher import build_env

    kw.setdefault("tree", "/spinning/wt-weg2-anon")
    kw.setdefault("venv", "/spinning/htsglang-gpu/.venv")
    kw.setdefault("cvd", "0,1,2")
    kw.setdefault("store_dir", "/tmp/store-1269")
    kw.setdefault("debug_hold", False)
    kw.setdefault("tag", "t1269")
    return build_env(**kw)


def test_launcher_publishes_the_idle_family_to_both_groups():
    """THE #1269 LAUNCHER FIX.  Red before it: neither key was ever set."""
    for group in ("P", "D"):
        env = _build_env(group=group)
        assert env["MALLOC_ARENA_MAX"] == "4", group
        assert env["SGLANG_IDLE_BLOCKING_POLL"] == "1", group


def test_arena_max_is_a_ceiling_the_operator_can_override(monkeypatch):
    """setdefault, not assignment: an operator experiment must still win."""
    monkeypatch.setenv("MALLOC_ARENA_MAX", "1")
    monkeypatch.setenv("SGLANG_IDLE_BLOCKING_POLL", "0")
    env = _build_env(group="P")
    assert env["MALLOC_ARENA_MAX"] == "1"
    assert env["SGLANG_IDLE_BLOCKING_POLL"] == "0"


def test_the_group_discriminator_still_arms_the_serve_side():
    """The launcher env and the serve-side arming are belt and braces; both
    must be present, because either alone would have fixed weg2sb4."""
    env = _build_env(group="P")
    assert env["SGLANG_WEG2_GROUP"] == "P"


# ------------------------------------------------------------- the instrument
def test_read_status_uses_proc_and_reports_anon_and_threads():
    st = _read_status()
    assert "RssAnon" in st and st["RssAnon"] > 0
    assert "Threads" in st and st["Threads"] >= 1


def test_first_call_only_arms_the_window():
    c = IdleCensus(group="P", rank=0, period_s=0.0)
    assert c.maybe_emit() is None  # nothing to compare against yet


def test_line_carries_the_three_acceptance_fields():
    c = IdleCensus(group="D", rank=2, period_s=0.0)
    c.maybe_emit()
    for _ in range(7):
        c.tick()
    line = c.maybe_emit()
    assert line is not None
    for field in ("rounds_per_s=", "rss_anon_mib=", "d_anon_mib_per_min="):
        assert field in line, line
    assert "group=D" in line and "rank=2" in line
    assert "threads=" in line and "arena_max=" in line


def test_emission_is_bounded_by_the_period():
    c = IdleCensus(group="P", rank=0, period_s=3600.0)
    c.maybe_emit()
    for _ in range(1000):
        c.tick()
    assert c.maybe_emit() is None, "must not emit again inside the period"


def test_reset_restarts_the_window_so_a_loaded_lap_is_not_averaged_in():
    c = IdleCensus(group="P", rank=0, period_s=0.0)
    c.maybe_emit()
    c.reset()
    assert c._last_t is None
    assert c.maybe_emit() is None  # re-arms rather than reporting a bogus rate


def test_rounds_per_s_reflects_the_ticks():
    c = IdleCensus(group="P", rank=1, period_s=0.0)
    c.maybe_emit()
    for _ in range(50):
        c.tick()
    line = c.maybe_emit()
    rate = float(line.split("rounds_per_s=")[1].split()[0])
    assert rate > 0.0


def test_trim_is_not_run_on_a_rank_that_is_not_growing():
    """malloc_trim walks the arena free lists; a flat rank must not pay it."""
    c = IdleCensus(group="P", rank=0, period_s=0.0)
    c.maybe_emit()
    c._anon_at_trim = (_read_status().get("RssAnon") or 0) + int(
        WEG2_TRIM_MIN_GROWTH_MIB * 1024 * 4
    )
    c.tick()
    c.maybe_emit()
    assert c._trimmed == 0


def test_the_census_never_raises_without_proc(monkeypatch):
    monkeypatch.setattr("sglang.srt.weg2.idle_census._read_status", dict)
    c = IdleCensus(group="P", rank=0, period_s=0.0)
    c.maybe_emit()
    c.tick()
    line = c.maybe_emit()
    assert line is not None and "rss_anon_mib=0.0" in line
