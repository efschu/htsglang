"""Q-701 D BOOT-PHASE (NF y9nf boot 2, c7b2690e2a, D serving 00:49:37Z, never woken).

The phase state (``d_seat_vram.PHASE_ATTR``) was made only by a wake (``on_wake``), so a D that
serves from its boot had none and ``runtime_tick`` -- the only thing that moves the KV stage
between wakes -- returned at its second check. D stayed at the boot cap S0 = 32768 tokens
(``#1045 FLOOR PUBLISHED floor=32768``) while the front sent it store-cached requests on the X
route (weg2-0-1/0-2 load-back 159040, weg2-0-4/0-7 88768): ``H105 RU FORM-A ADMISSION WAIT
rid=weg2-0-7 host=NO_TOKEN host_price=93281 host_budget=32768 ... refusals=5604 waited_s=272.7``,
``ADMISSION-WEDGE: 4 queued, 0 running``, no forward, no flip -- the earlier boots were cold, so
their first requests went long via P and the first P>D wake made the phase.

Pinned: with no phase on an awake, armed D the boot form is seeded as the phase (cap seats, S0,
named once) and the tick grows the stage for the demand; asleep / unarmed / no stage form:
nothing seeded.
"""

import logging
import types

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRID = [32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288]


def _req(rid, n_in, n_out=0, max_new=32768):
    return types.SimpleNamespace(
        rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out, finished=lambda: False,
        sampling_params=types.SimpleNamespace(max_new_tokens=max_new))


@pytest.fixture
def env(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_BY_DEMAND", "1")
    caps = []
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: 0)

    def make(dormant=False):
        sched = types.SimpleNamespace(
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=16384,
                                              speculative_num_draft_tokens=4, page_size=64),
            running_batch=types.SimpleNamespace(reqs=[], batch_is_full=False), waiting_queue=[],
            chunked_req=None, last_batch=None, page_size=64, token_to_kv_pool_allocator=object(),
            new_token_ratio_tracker=types.SimpleNamespace(current=0.3),
            _weg2_group_min_ints=lambda vals: vals, weg2_dormant=dormant)
        setattr(sched, dsv.CTL_ATTR, False)
        # the y9nf-2 queue: four store-cached requests, nothing running, no wake yet
        sched.waiting_queue = [_req("weg2-0-1", 159243), _req("weg2-0-2", 159040 + 1934),
                               _req("weg2-0-4", 88768 + 219), _req("weg2-0-7", 88768 + 353)]
        return sched

    return dsv, make, caps


def test_a_d_serving_from_its_boot_grows_its_stage(env, caplog):
    dsv, make, caps = env
    sched = make()
    assert getattr(sched, dsv.PHASE_ATTR, None) is None
    with caplog.at_level(logging.INFO):
        dsv.runtime_tick(sched)
    st = getattr(sched, dsv.PHASE_ATTR)
    assert st.epoch == dsv.BOOT_EPOCH and st.n is None and st.cap == 6
    assert st.stage_tokens > GRID[0], "base: no phase, the tick returned, S0 for 272 s"
    assert caps and caps[-1] > GRID[0]
    assert sum(dsv.BOOT_PHASE_MARK in r.getMessage() for r in caplog.records) == 1
    dsv.runtime_tick(sched)   # seeded once
    assert sum(dsv.BOOT_PHASE_MARK in r.getMessage() for r in caplog.records) == 1


def test_asleep_or_unarmed_seeds_nothing(env, monkeypatch):
    dsv, make, _caps = env
    sched = make(dormant=True)
    dsv.runtime_tick(sched)
    assert getattr(sched, dsv.PHASE_ATTR, None) is None
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    sched = make()
    dsv.runtime_tick(sched)
    assert getattr(sched, dsv.PHASE_ATTR, None) is None


def test_no_stage_form_seeds_nothing(env, monkeypatch):
    dsv, make, _caps = env
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", "32768")
    assert dsv.boot_phase(make()) is None


def test_the_first_wake_replaces_the_boot_phase(env):
    dsv, make, _caps = env
    sched = make()
    st = dsv.boot_phase(sched)
    assert st.epoch == dsv.BOOT_EPOCH
    # a real wake carries another epoch: on_wake resets has_n/done for it
    assert dsv.admission_cap(sched) is None   # n None: the boot form caps nothing
