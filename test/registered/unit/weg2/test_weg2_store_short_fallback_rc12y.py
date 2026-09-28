"""rc12y (28.09., D-Log weg2-16-42): the store-short deferral must be bounded.

THE LIVELOCK (ce041f0ad0, boot dkrnfh91dprsabar1dauer09272323, D, TP0):
weg2-16-42 (46850 tokens, the follow-up turn of weg2-13-41, routed SHORT->D on
the front's d_leg2_cached credit) read the store every pass and every read
terminated 192 short -- ``#1324 STORE READ INCOMPLETE delivered=46656
deliverable=46848``. 692 cycles in 6 minutes, each

    #1068 PREFETCH DEFERRED ... reason=store_prefix_short attempt=1
    #1068 PREFETCH LANDED ... after_passes=1 verdict=issued

i.e. the drain raised a FRESH mark, the retry re-issued the read, the
re-issue ("issued") cleared the mark as LANDED, the read terminated short
again. The progress witness only runs when a mark survives into a second
observation, so it never ran; the X gate held the request (X-DEFER
bound_s=inf) until the front's deadman said BUSY-STARVED. No writer was
filling the tail: D's own previous turn had left it out of the store
(#1469 RETAIN ... cache_len=None value=False) and P was asleep.

Pinned here on the metal numbers (hermetic, CPU): the fresh-mark cycle takes
at most SGLANG_WEG2_STORE_SHORT_MAX_CYCLES DEFERRED lines per rid, then
``PREFETCH-DEFER-FALLBACK`` releases the request with its matched prefix (the
remainder 194 <= X=12288, D extends it) and the short record raises no new
mark; a read that still grows is waited for; a sleeping D does not count; a
remainder over X takes the named W88. The NF profile runs with the xsn437
store-short tail OFF, so every case here runs with it off too.
"""

import logging
import types

import pytest

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome

#: (N, delivered, deliverable, X) as measured on rc12y, rid weg2-16-42
RC12Y = (46850, 46656, 46848, 12288)
#: the #1324 form (weg2sn6s): the remainder 55,885 lies over X=11,101
SN6S = (109132, 53247, 109131, 11101)
MAX_CYCLES_ENV = "SGLANG_WEG2_STORE_SHORT_MAX_CYCLES"
BOUND = 4    # the default bound: at most 4 DEFERRED lines per rid


@pytest.fixture(autouse=True)
def _nf_profile(monkeypatch):
    # NF: ModelProfile.store_short_tail is off -- the xsn437 recompute never
    # fires there, so the bound must not depend on it.
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(MAX_CYCLES_ENV, raising=False)


def _sched(n_tokens, delivered, deliverable, x):
    s = types.SimpleNamespace()
    s.tree_cache = types.SimpleNamespace(
        prefetch_loaded_tokens_by_reqid={
            "weg2-16-42": PrefetchOutcome(delivered, matched=0, deliverable=deliverable,
                                          synced=delivered)},
        ongoing_prefetch={},
        prefetch_timeout_base=1.0,
        prefetch_timeout_per_page=0.01,
        page_size=1,
        cache_controller=types.SimpleNamespace(host_role="staging"),
        release_aborted_request=lambda rid: None,
    )
    s.server_args = types.SimpleNamespace(tp_prefill_max_tokens=x)
    s.enable_hicache_storage = True
    s.enable_hierarchical_cache = False
    s.weg2_dormant = False
    s.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=lambda *a, **k: None))
    for name in (
        "_weg2_note_store_shortfall",
        "_apply_prefetch_deferral",
        "_apply_group_shortfall_deferral",
        "_weg2_store_read_is_pending",
        "_weg2_note_prefetch_progress",
        "_weg2_prefetch_progress_terms",
        "_weg2_prefetch_stall_passes",
        "_weg2_windowed_store_read_active",
        "_weg2_store_load_terminal",
        "_prefetch_deferral_refusal_reason",
        "_prefetch_capacity_limit_or_none",
        "_clear_prefetch_deferral_fields",
    ):
        setattr(s, name, getattr(sched_mod.Scheduler, name).__get__(s))
    r = types.SimpleNamespace(
        rid="weg2-16-42", prefetch_deferred=None, _prefetch_span_tokens=n_tokens,
        prefix_indices=None, host_hit_length=0,
        full_untruncated_fill_ids=list(range(n_tokens)),
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    s.waiting_queue = [r]
    return s, r


def _cycle(s, r):
    """One metal cycle: the drain sees the short record, the retry re-issues
    the read (verdict 'issued'). Returns the drain's outcome."""
    out = s._weg2_note_store_shortfall(r)
    if out == "deferred":
        s._apply_prefetch_deferral(r, "issued", site="retry")
    return out


def _deferred_lines(caplog):
    return sum("#1068 PREFETCH DEFERRED" in rec.getMessage() for rec in caplog.records)


def test_the_fresh_mark_cycle_is_bounded_and_falls_back(caplog):
    n, delivered, deliverable, x = RC12Y
    s, r = _sched(n, delivered, deliverable, x)
    outs = []
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        for _ in range(50):                      # the metal ran 692 of these
            outs.append(_cycle(s, r))
            if outs[-1] == "expired":
                break
    assert outs[-1] == "expired", f"50 cycles still {outs[-5:]} -- the rc12y livelock"
    assert _deferred_lines(caplog) <= BOUND, "#1068 DEFERRED per rid at most N"
    fb = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("PREFETCH-DEFER-FALLBACK")]
    assert len(fb) == 1 and "rid=weg2-16-42" in fb[0] and f"tail={n - delivered}" in fb[0]
    assert "reason=no_writer_progress" in fb[0]
    # released: in the queue, no mark, the X gate no longer sees a pending read
    assert r in s.waiting_queue and r.prefetch_deferred is None
    assert not s._weg2_store_read_is_pending(r)
    # the short record stays -- it must not raise a fresh mark every pass
    for _ in range(5):
        assert s._weg2_note_store_shortfall(r) is None
    assert not s._weg2_store_read_is_pending(r)
    assert "W88" not in "".join(fb)


def test_a_read_that_still_grows_is_waited_for():
    n, delivered, deliverable, x = RC12Y
    s, r = _sched(n, delivered - 4096, deliverable, x)
    recs = s.tree_cache.prefetch_loaded_tokens_by_reqid
    for k in range(3 * BOUND):
        got = delivered - 4096 + 64 * k          # the write-through delivers more each read
        recs["weg2-16-42"] = PrefetchOutcome(got, matched=0, deliverable=deliverable, synced=got)
        assert _cycle(s, r) == "deferred", f"cycle {k}: a growing read is a wait, not a fallback"


def test_a_sleeping_d_does_not_count(caplog):
    n, delivered, deliverable, x = RC12Y
    s, r = _sched(n, delivered, deliverable, x)
    s.weg2_dormant = True
    for _ in range(3 * BOUND):
        assert _cycle(s, r) == "deferred"          # P may still publish (xsn344)
    s.weg2_dormant = False
    # the first read set the best; from the wake on the bound counts
    outs = [_cycle(s, r) for _ in range(BOUND)]
    assert outs == ["deferred"] * (BOUND - 1) + ["expired"]


def test_over_x_the_bound_is_the_named_w88(caplog):
    n, delivered, deliverable, x = SN6S
    s, r = _sched(n, delivered, deliverable, x)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        outs = [_cycle(s, r) for _ in range(BOUND + 1)]
    assert outs[-1] == "failed" and "W88" in caplog.text
    assert "reason=over_x" in caplog.text
    assert r not in s.waiting_queue, "never a prefill over X"


def test_the_bound_is_env_overridable(monkeypatch):
    monkeypatch.setenv(MAX_CYCLES_ENV, "2")
    n, delivered, deliverable, x = RC12Y
    s, r = _sched(n, delivered, deliverable, x)
    assert [_cycle(s, r) for _ in range(3)] == ["deferred", "deferred", "expired"]
