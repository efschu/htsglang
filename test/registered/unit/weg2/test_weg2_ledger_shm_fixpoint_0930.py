"""LEDGER-FIXPOINT (30.09., NF y4m W87 12:29Z / 12:32Z): the census shmem claim
is the measured rest OUTSIDE the store and the arena plus THIS arm's own
store and arena -- not the record's all-time total.

Metal (launcher.log of both y4m attempts, record
/spinning/docker-acceptance/nf/evidence/host_census_record.json, 493
samples): the record's peak of the SUM ``shm_total_max`` = 60.14 GiB at
08:34:06Z, with a 43.96 GiB store and a 6.54 GiB arena at that instant. y4m
planned a 38.97 GiB store. The old cap credited a store only when it was
BIGGER than the instant's, so the 4.99 GiB difference was charged as
``unposted`` shmem: 11.20 GiB instead of 6.21 -> run peak 90.03 GiB, + the
1.50 GiB cushion floor = 91.53 > 90.44 hard bound -> W87/W20. Replayed from
the arm line: origin 0.79 + heaps 17.23 + small posts 0.81 + arena 6.50 +
l3 0.11 + store 38.97 + non-rank anon 8.88 + census shm posts 2.50 +
unposted 11.20 + flip ratchet 2.99 = 90.03 GiB (the logged value).
"""

import json

import pytest

from sglang.srt.weg2 import host_census as hc
from sglang.srt.weg2 import host_ledger as hl

#: the y4m record entry (the fields the ledger reads), as on disk
RECORD = {
    "samples": 493, "last_at": "2026-09-30T12:04:49Z",
    "unattributed_shm_gib": 52.49, "shm_total_max_gib": 60.14,
    "shm_total_store_gib": 43.96, "shm_total_arena_gib": 6.54,
    "shm_total_source": "live /proc smaps(_rollup) + memory.stat 2026-09-30T08:34:06Z",
    "roles_anon_gib": {"front": 1.16, "server_main": 2.36, "launcher": 0.73, "detokenizer": 2.18,
                       "inductor_compile_worker": 2.28, "ple_pread_worker": 0.14,
                       "mp_resource_tracker": 0.02, "other": 0.0, "rank": 19.61},
    "shm_classes_gib": {"anon_shared": 6.88, "arena_booked": 6.55, "arena_handoff": 0.23,
                        "arena_sidecar": 0.32, "l3idx": 0.14, "other_tmpfs": 0.97,
                        "seq_ring": 1.95, "store": 45.05, "xchg": 0.0},
}


def _terms(store_gib):
    """y4m's ARM S=1 M=600 posts, the store varied."""
    c = hc.ledger_terms(RECORD)
    return {
        "anchors_gib": 0.40, "rings_gib": 0.22, "overhead_gib": 0.02,
        "draft_host_p_gib": 119.2 / 1024, "draft_host_d_gib": 59.6 / 1024, "d_draft_host_gib": 0.0,
        "arena_gib": 6.50, "cold_tier_shm_gib": store_gib, "l3_index_gib": 0.11,
        "seq_ring_gib": c["seq_ring_gib"], "arena_sidecar_gib": c["arena_sidecar_gib"],
        "arena_handoff_gib": c["arena_handoff_gib"],
    }, c


def _claim(store_gib):
    t, c = _terms(store_gib)
    posts = hl.census_shm_posts(t, c)
    priced = sum(t[k] for k in ("cold_tier_shm_gib", "arena_gib", "l3_index_gib", "seq_ring_gib",
                                "arena_sidecar_gib", "arena_handoff_gib", "anchors_gib", "rings_gib",
                                "overhead_gib", "draft_host_p_gib", "draft_host_d_gib"))
    return posts["unposted_shm_gib"], priced + posts["arena_census_excess_gib"] + posts["unposted_shm_gib"]


def test_y4m_is_charged_the_measured_rest_not_the_old_store():
    unposted, total = _claim(38.97)
    # rest at the instant: 60.14 - 43.96 - 6.54 = 9.64, minus the posts priced
    # inside it (l3 0.11, seq 1.95, sidecar 0.32, hand-off 0.23, small 0.81)
    assert unposted == pytest.approx(6.21, abs=0.02)      # before: 11.20
    assert total == pytest.approx(55.16, abs=0.02)        # before: 60.14


def test_the_same_checkpoint_and_form_give_the_same_charge_whatever_the_store():
    """The fixpoint: the pathless remainder does not move with the store."""
    got = {s: _claim(s)[0] for s in (38.97, 42.49, 43.96, 45.05)}
    assert max(got.values()) - min(got.values()) < 1e-6, got


def test_a_bigger_store_is_charged_in_full():
    """The funding direction only for a SMALLER store: a bigger one still
    raises the claim by exactly its own difference (never below the old form)."""
    _u1, t1 = _claim(43.96)
    _u2, t2 = _claim(45.05)
    assert t2 - t1 == pytest.approx(45.05 - 43.96, abs=1e-6)
    assert t1 == pytest.approx(60.14 + (6.50 - 6.54) + 0.05, abs=0.01)  # the instant, own arena + excess


def test_y4m_run_peak_replay_funds_under_the_cushion_floor():
    unposted_old, unposted_new = 11.20, _claim(38.97)[0]
    rest_of_peak = 90.03 - unposted_old                    # every other term of the logged peak
    peak = rest_of_peak + unposted_new
    assert peak == pytest.approx(85.04, abs=0.05)
    assert peak + 1.50 <= 90.44                            # cushion floor vs hard bound


def test_the_record_keeps_the_rest_of_one_instant(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    key = "digest|form"
    for tot, store, arena in ((60.14, 43.96, 6.54), (56.00, 38.97, 6.50), (58.0, 45.05, 2.0)):
        hc.merge_into_record(path, key, {
            "cg_shmem_gib": tot, "shm_classes_gib": {"store": store, "arena_booked": arena},
            "roles_anon_gib": {}, "unattributed_shm_gib": 0.0, "source": "t", "at": str(tot)})
    ent = json.load(open(path))[key]
    assert ent["shm_total_max_gib"] == pytest.approx(60.14)
    # rests 9.64, 10.53, 10.95 -> the max of the DIFFERENCE, not of the total
    assert ent["shm_rest_max_gib"] == pytest.approx(10.95)
    assert hc.ledger_terms(ent)["shm_rest_max_gib"] == pytest.approx(10.95)
    assert hl.shm_rest_gib(hc.ledger_terms(ent)) == pytest.approx(10.95)


def test_an_old_record_derives_the_rest_from_its_peak_instant():
    c = hc.ledger_terms(RECORD)
    assert c["shm_rest_max_gib"] is None
    assert hl.shm_rest_gib(c) == pytest.approx(60.14 - 43.96 - 6.54)
    assert hl.shm_rest_gib({"shm_total_max_gib": None}) is None
