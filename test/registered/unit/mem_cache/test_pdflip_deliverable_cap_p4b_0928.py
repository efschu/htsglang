"""P4b-cap (28.09.): a store read that delivered everything that EXISTS is not short.

Metal (NF rc12z17-s0, 103bfdf29a, D log boot ...09281111): three follow-up
turns routed ``short`` straight to D read the store at the wake and were
stamped ``#1324 STORE READ INCOMPLETE``:

    rid        prompt  common prefix  delivered  deliverable  shortfall
    pdflip-2-17  41021   40569          40512      40960        448
    pdflip-4-36  47005   45270          45248      46976        1728
    pdflip-5-37  47167   47001          46976      47104        128

The "missing" pages lie past the prefix the turn shares with the previous one
-- new prompt tokens nobody ever computed, and no writer existed (no hand-off
chain, no hand-off record, no tail part; P never saw the requests). The
deliverable was the page floor of the whole key, so each read sat the #1471
settle hold (20 s), re-read every 2 s (#1456) and showed on the dashboard as
"L3 unvollst. Lesungen". With no writer the deliverable is the deepest point
the store knows for the key sequence (the probe's hit, group MIN) or what the
read delivered: the three reads are complete, named by '#1324c DELIVERABLE-CAP'.

The vote is conservative: a read registered while D was dormant (its probe saw
a store P was still writing), a leg in between, a hand-off chain or record, a
tail part (P's publish threads), no shared hand-off directory, group P -- each
keeps the old deliverable.
"""
import importlib.util
import json
import logging
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import cache_controller as cc_mod
from flliper.srt.mem_cache import l3_write_behind as wb
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE = 64  # NF page size
#: (rid, prompt, common prefix, delivered, deliverable before the cap)
METAL = [
    ("pdflip-2-17", 41021, 40569, 40512, 40960),
    ("pdflip-4-36", 47005, 45270, 45248, 46976),
    ("pdflip-5-37", 47167, 47001, 46976, 47104),
]
LOGGER = "flliper.srt.mem_cache.unified_radix_cache"


@pytest.fixture
def d_awake(tmp_path, monkeypatch):
    """Group D, a shared hand-off directory, the scheduler awake after one leg."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv("FLLIPER_PDFLIP_HANDOFF", raising=False)
    (tmp_path / "handoff").mkdir()
    wb._reset_for_tests()
    sched = _Sched()
    wb.leg_enter("resume_memory_occupation", sched)
    wb.leg_exit("resume_memory_occupation", True)
    saved = dict(cc_mod.PDFLIP_HANDOFF_PAGE_KEYS)
    cc_mod.PDFLIP_HANDOFF_PAGE_KEYS.clear()
    assert wb.awake_epoch() == 1
    yield types.SimpleNamespace(dir=tmp_path / "handoff", sched=sched)
    cc_mod.PDFLIP_HANDOFF_PAGE_KEYS.clear()
    cc_mod.PDFLIP_HANDOFF_PAGE_KEYS.update(saved)
    wb._reset_for_tests()


class _Sched:
    """weakref-able like the real Scheduler (SimpleNamespace is not)."""

    pdflip_dormant = False


def _op():
    """A read registered now (the stamp the registration site writes)."""
    return types.SimpleNamespace(_pdflip_awake_epoch=wb.awake_epoch())


# ---------------------------------------------------------------- the cap itself


@pytest.mark.parametrize("rid,prompt,common,delivered,asked", METAL)
def test_metal_form_is_complete_with_no_writer(caplog, rid, prompt, common, delivered, asked):
    assert asked == (prompt // PAGE) * PAGE  # the old deliverable: floor of the whole key
    assert delivered == (common // PAGE) * PAGE  # the store held the common prefix
    with caplog.at_level(logging.INFO, logger=LOGGER):
        capped = urc._pdflip_cap_deliverable(rid, asked, True, delivered, delivered, PAGE)
    assert capped == delivered
    out = PrefetchOutcome(delivered, hit_tokens=delivered, probed=True, matched=0,
                          deliverable=capped, synced=delivered)
    assert not out.is_incomplete
    line = [r.getMessage() for r in caplog.records if "#1324c DELIVERABLE-CAP" in r.getMessage()]
    assert len(line) == 1
    assert f"rid={rid} asked={asked} known={delivered} " in line[0]
    assert f"-> deliverable={delivered} " in line[0]


@pytest.mark.parametrize("rid,prompt,common,delivered,asked", METAL)
def test_metal_form_with_a_writer_stays_short(caplog, rid, prompt, common, delivered, asked):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        kept = urc._pdflip_cap_deliverable(rid, asked, False, delivered, delivered, PAGE)
    assert kept == asked
    out = PrefetchOutcome(delivered, deliverable=kept, synced=delivered)
    assert out.is_incomplete
    assert not [r for r in caplog.records if "DELIVERABLE-CAP" in r.getMessage()]


def test_cap_takes_the_deeper_of_store_hit_and_delivered():
    """The store knew 40960 (probe) but the read kept 40512 (e.g. #257's anchor
    cut): the read is still short against what exists."""
    assert urc._pdflip_cap_deliverable("r", 41024, True, 40960, 40512, PAGE) == 40960
    out = PrefetchOutcome(40512, deliverable=40960, synced=40512)
    assert out.is_incomplete


def test_cap_never_raises_the_deliverable():
    assert urc._pdflip_cap_deliverable("r", 4096, True, 8192, 8192, PAGE) == 4096


# ---------------------------------------------------------------- the rank's vote


def test_vote_no_writer_on_the_metal_form(d_awake):
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 1


def test_vote_group_p_keeps_the_old_deliverable(d_awake, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 0


def test_vote_read_registered_while_dormant(d_awake):
    """Registered in the flip / D's sleep: its probe saw a store P still wrote."""
    d_awake.sched.pdflip_dormant = True
    op = _op()
    assert op._pdflip_awake_epoch is None
    d_awake.sched.pdflip_dormant = False
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", op) == 0


def test_vote_a_leg_between_registration_and_termination(d_awake):
    op = _op()
    wb.leg_enter("release_memory_occupation", d_awake.sched)
    wb.leg_exit("release_memory_occupation", True)
    wb.leg_enter("resume_memory_occupation", d_awake.sched)
    wb.leg_exit("resume_memory_occupation", True)
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", op) == 0


def test_vote_dormant_at_termination(d_awake):
    op = _op()
    d_awake.sched.pdflip_dormant = True
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", op) == 0


def test_vote_no_scheduler_known(d_awake):
    op = _op()
    wb._reset_for_tests()
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", op) == 0


def test_vote_hand_off_chain_registered(d_awake):
    cc_mod.PDFLIP_HANDOFF_PAGE_KEYS["pdflip-2-17"] = ["k0", "k1"]
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 0


def test_vote_hand_off_record_present(d_awake):
    (d_awake.dir / "pdflip-2-17.json").write_text(json.dumps({"input_ids": [1], "page_keys": ["k"]}))
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 0


def test_vote_tail_part_under_write(d_awake):
    (d_awake.dir / "pdflip-2-17.tail.0.json.tmp").write_text("{}")
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 0


def test_vote_no_hand_off_directory(d_awake, monkeypatch):
    monkeypatch.delenv("FLLIPER_HICACHE_ARENA_DIR")
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 0


def test_vote_other_rid_does_not_count(d_awake):
    cc_mod.PDFLIP_HANDOFF_PAGE_KEYS["pdflip-9-99"] = ["k0"]
    (d_awake.dir / "pdflip-9-99.json").write_text("{}")
    assert urc._pdflip_read_no_writer_local("pdflip-2-17", _op()) == 1


# ---------------------------------------------------------------- the real reap path


def _harness():
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "_t1157_harness", os.path.join(here, "test_1157_reaper_prices_requested_span.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _short_read(h, *, stamp=True):
    """The metal shape at the harness scale (page_size 1): 16 tokens asked,
    the store holds 12 of them, the read delivers the 12."""
    cache, operation = h._reap_scenario(probed=False)
    held = h.REAP_TOKENS - 4
    operation.hash_value = [f"h{i}" for i in range(held)]
    operation.probed_hit_tokens = held
    operation.increment(held)
    operation._pdflip_awake_epoch = wb.awake_epoch() if stamp else None
    return cache, held


def test_real_termination_metal_form_is_not_incomplete(d_awake, caplog):
    h = _harness()
    cache, held = _short_read(h)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert cache.check_prefetch_progress(h.REAP_REQ)
    out = cache.prefetch_loaded_tokens_by_reqid[h.REAP_REQ]
    assert out.synced == held
    assert out.deliverable == held
    assert not out.is_incomplete
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert f"#1324c DELIVERABLE-CAP rid={h.REAP_REQ} asked={h.REAP_TOKENS} known={held} " in msgs
    assert "HiCache prefetch INCOMPLETE" not in msgs
    assert f"deliverable={held} shortfall=0" in msgs


def test_real_termination_read_registered_dormant_stays_incomplete(d_awake):
    h = _harness()
    cache, held = _short_read(h, stamp=False)
    cache.check_prefetch_progress(h.REAP_REQ)
    out = cache.prefetch_loaded_tokens_by_reqid[h.REAP_REQ]
    assert out.deliverable == h.REAP_TOKENS
    assert out.is_incomplete


def test_real_termination_peer_sees_a_writer_holds_the_group(d_awake):
    """The slot rides the packed MIN: one rank voting 0 keeps the old deliverable."""
    h = _harness()
    cache, held = _short_read(h)
    cache.tp_world_size = 2
    seen = {}

    def _peer_min(packed, op, label=""):
        if label != "check_prefetch_progress":
            return
        seen["local"] = int(packed[urc._REAP_SLOT_NO_WRITER].item())
        packed[urc._REAP_SLOT_NO_WRITER] = 0

    cache._all_reduce_attn_groups = _peer_min
    cache.check_prefetch_progress(h.REAP_REQ)
    assert seen["local"] == 1
    out = cache.prefetch_loaded_tokens_by_reqid[h.REAP_REQ]
    assert out.deliverable == h.REAP_TOKENS
    assert out.is_incomplete
