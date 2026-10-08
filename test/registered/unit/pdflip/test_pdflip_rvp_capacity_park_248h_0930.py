"""#248h (30.09., NF y4b D 03:58:57 - 03:59:24, rid pdflip-32-72): a streamed
request whose store read ended short on D's OWN capacity -- the store held
the context P had written (``#1324c DELIVERABLE-CAP ... deliverable=95104``,
P served it with ``cached_tokens=95104``) -- was refused by the X gate after
the W88 re-route three times: two RESUME-VIA-P legs (p_ms 7712 / 7460, flips
included) that could add nothing, then ``RESUME-VIA-P attempt`` exhausted,
``W50 PdFlipTpPrefillExceeded`` in-band and the front's ``LEG2-TERMINAL-NAMED
... re-route impossible``: the client's stream ended with an error.

Now such a refusal parks the request on D (no P leg, no attempt spent, no
W50) and the park re-queues it for a re-read as soon as the arena holds it;
bounded by ``FLLIPER_PDFLIP_RVP_CAPACITY_PARK_S``, past which the old path
applies. A short read the store itself cannot fill (P's write not landed)
keeps the old path."""

import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.pdflip import d_park_runtime, park_l3
from flliper.srt.pdflip import resume_via_p as rvp

S = sched_mod.Scheduler
X = 12288
TOTAL = 95137


@pytest.fixture(autouse=True)
def _group_d(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv("FLLIPER_PDFLIP_RESUME_VIA_P", raising=False)
    yield


def _req(attempts=0, delivered=39168, deliverable=95104, total=TOTAL):
    r = types.SimpleNamespace(
        rid="pdflip-32-72", stream=True, multimodal_inputs=None,
        origin_input_ids=[1] * total, output_ids=[], full_untruncated_fill_ids=[1] * total,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    # the W88 over-X re-route's state (y4b 03:58:57 / 03:59:11 / 03:59:24)
    r._pdflip_store_short_fallback = True
    r._pdflip_store_delivered = delivered
    r._pdflip_store_deliverable = deliverable
    setattr(r, rvp.ATTEMPTS_ATTR, attempts)
    return r


def _sched():
    h = types.SimpleNamespace()
    h.sent = []
    h.server_args = types.SimpleNamespace(tp_prefill_max_tokens=X)
    h.ps = types.SimpleNamespace(tp_size=1, tp_rank=0)
    h.tree_cache = None
    h.enable_hicache_storage = False
    h.enable_hierarchical_cache = False
    h.waiting_queue = []
    h.pdflip_d_parked = []
    h.pdflip_dormant = False
    h._pdflip_d_park_slept = False
    h.page_size = 64
    h.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=lambda out, req: h.sent.append(out)))
    h.pdflip_uncached_extent = lambda req, head_inputs=None: TOTAL
    h._add_request_to_queue = lambda req, is_retracted=False: h.waiting_queue.append(req)
    for n in ("_pdflip_answer_x_refusals", "_pdflip_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    return h


def _needs_p(tmp_path):
    d = tmp_path / rvp.SUBDIR
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_y4b_third_refusal_parks_instead_of_the_client_w50(tmp_path):
    h = _sched()
    req = _req(attempts=2)                     # both P legs spent (base: named end)
    h.waiting_queue = [req]
    h._pdflip_answer_x_refusals([req])
    assert h.sent == []                        # base: AbortReq W50 -> client error
    assert h.pdflip_d_parked == [req] and h.waiting_queue == []
    assert _needs_p(tmp_path) == []
    assert getattr(req, rvp.ATTEMPTS_ATTR) == 2


def test_y4b_first_refusal_needs_no_p_leg(tmp_path):
    """P already holds the context (it served it with cached_tokens=95104):
    the base's P leg cost a flip round trip for nothing."""
    h = _sched()
    req = _req(attempts=0)
    h._pdflip_answer_x_refusals([req])
    assert _needs_p(tmp_path) == []            # base: pdflip-32-72.json
    assert getattr(req, rvp.ATTEMPTS_ATTR) == 0
    assert h.pdflip_d_parked == [req]


def test_a_store_that_lacks_the_context_keeps_the_p_leg(tmp_path):
    h = _sched()
    req = _req(attempts=0, delivered=39168, deliverable=40000)   # P's write not landed
    h._pdflip_answer_x_refusals([req])
    assert _needs_p(tmp_path) == ["pdflip-32-72.json"]
    assert getattr(req, rvp.ATTEMPTS_ATTR) == 1


def test_the_bound_lapses_into_the_named_end(tmp_path):
    h = _sched()
    req = _req(attempts=2)
    setattr(req, rvp.CAPPARK_SINCE_ATTR, time.monotonic() - 10_000)
    h._pdflip_answer_x_refusals([req])
    assert len(h.sent) == 1                    # the old W50, by name


def test_the_switch_off_keeps_the_old_path(tmp_path):
    h = _sched()
    req = _req(attempts=2)
    with envs.FLLIPER_PDFLIP_RVP_CAPACITY_PARK_S.override(0.0):
        h._pdflip_answer_x_refusals([req])
    assert len(h.sent) == 1


def test_the_park_requeues_the_capacity_park_when_the_arena_has_room():
    h = _sched()
    other = types.SimpleNamespace(rid="p-leg", origin_input_ids=[1] * 64, output_ids=[])
    req = _req(attempts=2)
    h._pdflip_answer_x_refusals([req])
    h.pdflip_d_parked.insert(0, other)
    from flliper.srt.pdflip import d_seats

    d_seats.mark_parked(other, d_seats.SITE_FLIP, epoch=None, now=time.monotonic())
    assert d_park_runtime.park_tick(h) == 0                  # < 2 s since the park
    setattr(req, rvp.CAPPARK_AT_ATTR, time.monotonic() - 3.0)
    assert d_park_runtime.park_tick(h) == 1
    assert h.waiting_queue == [req] and h.pdflip_d_parked == [other]
    # the arena is held by this wake's released reads: no re-read yet
    h.waiting_queue, h.pdflip_d_parked = [], [other, req]
    setattr(req, rvp.CAPPARK_AT_ATTR, time.monotonic() - 3.0)
    held = types.SimpleNamespace(rid="held", origin_input_ids=[1] * (6000 * 64), output_ids=[])
    setattr(held, park_l3.PAGES_ATTR, 6000)
    setattr(held, park_l3.WAKE_ATTR, None)
    h.waiting_queue = [held]
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(
        page_size=64, mem_pool_host=types.SimpleNamespace(arena=types.SimpleNamespace(slots=6485))))
    assert d_park_runtime.park_tick(h) == 0                  # 6000 + 1487 > 6485
    h.waiting_queue = []
    assert d_park_runtime.park_tick(h) == 1


def test_an_admit_ends_the_capacity_park():
    req = _req()
    setattr(req, rvp.CAPPARK_SINCE_ATTR, 1.0)
    setattr(req, rvp.CAPPARK_AT_ATTR, 1.0)
    rvp.clear_capacity_park(req)
    assert getattr(req, rvp.CAPPARK_SINCE_ATTR) is None and getattr(req, rvp.CAPPARK_AT_ATTR) is None
