"""W88 CYCLE (29.09., NF boot dkrnfh91dprsavisnoadoptstcutvsyncbar1dauer09290232,
e7c0200ccf, D TP0 03:21:06): two mid-stream requests answered 503 by W88 on
the FIRST drain of a new read cycle, although their stores held the prefix.

The metal lines:

    03:20:16 #1068 PREFETCH DEFERRED rid=pdflip-122 ... span=96129      (cycle n)
    03:20:17 #1068 PREFETCH DEFER RELEASED ... after_passes=11         (witness 96000)
    03:20:56 PDFLIP-D-PARK park_running ... parked=[... pdflip-122-144 ...] (cycle n+1)
    03:21:06 HiCache prefetch INCOMPLETE req=pdflip-122-144 ... deliverable=97024 shortfall=67328
    03:21:06 W88 ... rid=pdflip-122-144 arm=store_prefix_short span=97025 site=drain
             no_progress_passes=10 ... witness=(0, 0, 0, 0, 0, 0, 0, 96000)
    03:21:06 W88 ... rid=pdflip-126-148 ... no_progress_passes=1 ... witness=(..., 7872)
    03:21:14 PREFETCH-DEFER-FALLBACK rid=pdflip-122-144 delivered=29696 tail=67339
             X=12288 cycles=5 bound=4 reason=over_x -- ... named W88

(1) The park opens a new read cycle and cleared only the #1324 stamp; the
progress witness of the previous cycle (best 96000, 9 passes, a wall clock
49 s old) survived, so the fresh read (29696) was "no progress" and the 30 s
wall bound had long run out: terminal on one pass, 0.6 s after the read began.
(2) Over X the store-short arm answered W88 (503) although a readable prefix
existed and RESUME-VIA-P / the W50 re-route could let P read it and prefill
only the rest.

Hermetic, CPU; the NF profile runs with the xsn437 tail off.
"""
from __future__ import annotations

import logging
import os
import time
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as rt  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402

RID = "pdflip-122-144"
#: metal: context 97025 tokens, the new cycle's read delivered 29696 of 97024
N, DELIVERED, DELIVERABLE, X = 97025, 29696, 97024, 12288
#: the previous cycle's witness, as the W88 line printed it
STALE_TERMS = (0, 0, 0, 0, 0, 0, 0, 96000)


@pytest.fixture(autouse=True)
def _nf_d(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_SHORT_MAX_CYCLES", raising=False)


class _Batch:
    def __init__(self, reqs):
        self.reqs, self.batch_is_full = list(reqs), True

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        out, self.reqs = self.reqs, []
        return out


def _sched():
    s = types.SimpleNamespace()
    s.tree_cache = types.SimpleNamespace(
        prefetch_loaded_tokens_by_reqid={},
        ongoing_prefetch={},
        prefetch_timeout_base=1.0,
        prefetch_timeout_per_page=0.01,
        page_size=1,
        cache_controller=types.SimpleNamespace(host_role="staging"),
        release_aborted_request=lambda rid: None,
    )
    s.server_args = types.SimpleNamespace(tp_prefill_max_tokens=X)
    s.enable_hicache_storage = True
    s.enable_hierarchical_cache = False
    s.pdflip_dormant = False
    s.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=lambda *a, **k: None))
    # what park_running touches
    s.waiting_queue, s.last_batch, s.enable_overlap = [], None, False
    s.result_queue, s.chunked_req, s.anchor_tails = deque(), None, []
    s._969ad_note_retract = lambda req, site: None
    for name in (
        "_pdflip_note_store_shortfall",
        "_apply_prefetch_deferral",
        "_apply_group_shortfall_deferral",
        "_pdflip_store_read_is_pending",
        "_pdflip_note_prefetch_progress",
        "_pdflip_prefetch_progress_terms",
        "_pdflip_prefetch_stall_passes",
        "_pdflip_windowed_store_read_active",
        "_pdflip_store_load_terminal",
        "_prefetch_deferral_refusal_reason",
        "_prefetch_capacity_limit_or_none",
        "_clear_prefetch_deferral_fields",
    ):
        setattr(s, name, getattr(sched_mod.Scheduler, name).__get__(s))
    return s


def _req_after_previous_cycle():
    """The request as the previous cycle's read left it (cycle n, 03:20:16)."""
    r = types.SimpleNamespace(
        rid=RID, prefetch_deferred=None, _prefetch_span_tokens=N,
        prefix_indices=None, host_hit_length=0,
        full_untruncated_fill_ids=list(range(N)),
        origin_input_ids=[1] * (N - 10), output_ids=[2] * 10,
        kv_arrival_seq=1, is_fast_lane=False, spill_class=None, stream=True,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    r._pdflip_store_delivered = 96000
    r._pdflip_best_delivered = 96000
    r._pdflip_progress_terms = STALE_TERMS
    r._pdflip_no_progress_passes = 9
    r._pdflip_no_progress_t0 = time.perf_counter() - 49.0
    r._pdflip_store_short_cycle_best = 96000
    r._pdflip_store_short_cycles = 1
    return r


def _park(s, r):
    from flliper.srt.managers.io_struct import PdFlipParkRunningReqInput

    s.running_batch = _Batch([r])
    out = rt.park_running(s, PdFlipParkRunningReqInput(epoch=132, reason="immediate-over-x"))
    assert out.success and RID in out.parked
    # the wake releases it to the queue (#1471 settle / park_tick)
    s.waiting_queue = [r]


def _short_read(s):
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = PrefetchOutcome(
        DELIVERED, matched=0, deliverable=DELIVERABLE, synced=DELIVERED)


def test_a_new_read_cycle_is_not_judged_by_the_previous_cycles_witness(caplog):
    """Metal 03:21:06: the first drains of the new cycle answered W88 on the
    previous cycle's witness (passes 10, clock 49 s). A new cycle starts at 0."""
    s, r = _sched(), _req_after_previous_cycle()
    _park(s, r)
    _short_read(s)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        first = s._pdflip_note_store_shortfall(r)     # fresh mark
        second = s._pdflip_note_store_shortfall(r)    # the mark survives: witness
    assert first == "deferred"
    assert second == "deferred", f"{second}: judged on the previous cycle's witness"
    assert "W88 PdFlipStoreLoadNotProgressing" not in caplog.text
    assert r in s.waiting_queue


def test_a_standstill_over_x_is_rerouted_not_a_503(caplog):
    """The store-short arm over X: the read stood still (wall bound run out
    within THIS cycle). The request falls back by name to the delivered depth
    -- released to the X gate, which re-routes it through P -- never a 503."""
    s, r = _sched(), _req_after_previous_cycle()
    _park(s, r)
    _short_read(s)
    assert s._pdflip_note_store_shortfall(r) == "deferred"
    assert s._pdflip_note_store_shortfall(r) == "deferred"
    r._pdflip_no_progress_t0 -= 31.0                  # this cycle's read stood 31 s
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        out = s._pdflip_note_store_shortfall(r)
    assert out == "expired", f"{out}: over X the store-short arm must re-route"
    assert "W88 PdFlipStoreLoadNotProgressing" not in caplog.text
    line = [m.getMessage() for m in caplog.records if m.getMessage().startswith("W88-REROUTE")]
    assert len(line) == 1 and f"rid={RID}" in line[0] and "cause=standstill" in line[0]
    assert f"delivered={DELIVERED}" in line[0] and f"remainder={N - DELIVERED}" in line[0]
    # released: in the queue, no mark, the short record raises no fresh one
    assert r in s.waiting_queue and r.prefetch_deferred is None
    assert s._pdflip_note_store_shortfall(r) is None
    assert not s._pdflip_store_read_is_pending(r)


def test_the_host_pool_arm_keeps_its_named_w88(caplog):
    """Negative branch: the re-route is the STORE-SHORT arm's (a prefix P can
    fill). A read our own staging pool cut (``host_pool_shortfall``) that
    stands still keeps the named W88 -- judged on THIS cycle's witness (on the
    base the previous cycle's best 96000 survived the park and read the
    standstill as progress, the mirror image of the metal case)."""
    s, r = _sched(), _req_after_previous_cycle()
    _park(s, r)
    r.prefetch_deferred = sched_mod._DEFER_REASON_SHORTFALL
    r.prefetch_defer_attempts = 1
    r._pdflip_progress_terms = s._pdflip_prefetch_progress_terms(r)
    r._pdflip_no_progress_passes = 0
    r._pdflip_no_progress_t0 = time.perf_counter() - 31.0
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        out = s._apply_prefetch_deferral(r, sched_mod._VERDICT_TRUNCATED_GROUP, site="drain")
    assert out == "failed" and "W88 PdFlipStoreLoadNotProgressing" in caplog.text
    assert "W88-REROUTE" not in caplog.text
