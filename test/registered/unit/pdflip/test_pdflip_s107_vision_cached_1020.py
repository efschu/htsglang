"""S107 (NF 20.10.): a request whose image and context are cached is classed by its exact u -- no p_only by state, no flip
under X. SOLL R7 (no flip under X; a request D cannot take under X is a defect of D), R9 (p_only only for an image in the
UNCACHED tail), user decision 19.10. ("Bild-Token im gecachten Präfix brauchen keinen Tower", S66) and user order 20.10.
~13:35Z ("Ist das Bild samt Kontext gecacht, wird nichts geladen und nicht geflippt").

THE ROOT (live, boots of 20.10.): D's PW-R (``pdflip/d_wall_head.py``) answers a head its device cannot HOLD while the
backup wall stands with W50 and the DEVICE ROWS as the extent. The front took the rows as u:
  * held/midstream (RESUME-VIA-P, text branch): ``p_only=True`` by state, ``est_uncached = d_extent`` -> rc22 _094747
    pdflip-2-21 09:54:29Z: arrival u=76 (D tree 56064 of 56140), D-WALL-HEAD need_rows=56140 -> SET-VERDICT p_only ->
    PARK_NO_SEAT flip, P computed 76 (monitor R7-VERSTOSS "p_only per Zustand OHNE Bild" + KURZ-NACH-P-R5).
  * fresh (X-REQUEUE): rc19 _074525 pdflip-22-78 08:04:07Z: 5 images all cached (credit 85248 >= image_end 24639, front
    "image-cached -- route short stands: no tower"), u=246; D-WALL-HEAD need_rows=85494 -> "gross 85494" -> flip, P
    computed 246, D refused again after P -> W53 to the client (monitor VISION-FLIP).
Both now class on the exact number; the D record names a wall refusal (``REASON_WALL``)."""
from __future__ import annotations

import asyncio
import os
import sys
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

X = 6400


def _price(n, covered, image_in_tail=False):
    return dmp.ExactPrice(uncached=n - covered, uncached_p=n - covered, covered=covered, tree_tokens=covered,
                          store_tokens=covered, image_in_tail=image_in_tail, src="d_tree")


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    return tmp_path


def _front():
    from test_ez_s1_review_1019 import _fresh
    f = _fresh()
    f.tag = "dkrtest"
    f.tp_prefill_max_tokens = X
    return f


def _take(f):
    async def body():
        return f._rvp_take()
    return asyncio.run(body())


# ---------------------------------------------------------------------------------------------------- midstream (RESUME-VIA-P)
def test_live_2_21_a_wall_refusal_of_a_text_leg_classes_by_the_exact_u_and_makes_no_flip(d_env):
    """rc22 pdflip-2-21: u=76 at the arrival, D-WALL-HEAD need_rows=56140. Before: p_only, u=56140, PARK_NO_SEAT flip."""
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-2-21"] = _price(56140, 56064)
    rvp.write_request("pdflip-2-21", list(range(56140)), 56140, X, rvp.REASON_WALL)
    assert _take(f) == 1
    p = f.queue[-1]
    assert p.resume_via_p and p.p_only is False and p.est_uncached == 76
    assert pp.classify_arrival(p.est_uncached, X, 0.5, False) == pp.CLASS_KURZ
    # no member of the set, no P trigger, no flip verdict (R7)
    st = f._set_st()
    assert F.Front._set_cands(f, [p], st) == []
    assert pp.immediate_park_trigger([p], X) is None
    with F.envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(time.time()) != "P"
    # never handed to D's line: D holds the stream and re-offers it itself
    p.d_eligible = True
    assert p.d_eligible is False


def test_a_text_leg_over_x_still_needs_p(d_env):
    """D's X gate refused a real extent over X (rc19 pdflip-14-63: 15077 > 12288): gross by its exact u, P-bound as ever."""
    f = _front()
    rvp.write_request("pdflip-14-63", list(range(20000)), 15077, X, "x_refusal_midstream")
    assert _take(f) == 1
    p = f.queue[-1]
    assert p.p_only is False and p.est_uncached == 15077
    assert pp.immediate_park_trigger([p], X) is p


def test_a_wall_record_without_any_price_is_unpriced_never_the_rows(d_env, caplog):
    """NF-VISION-CACHED (20.10., rc28 pdflip-4-226): before, no probe price -> the rows (the whole context) were the class
    (u_src=d_rows) and R4 flipped. Now: no D price and no store depth -> UNPRICED (u=0: no set member, no flip reason,
    R7), named, and a store re-probe prices it (test_nf_vision_cached_flip_1020)."""
    f = _front()
    rvp.write_request("pdflip-9-9", list(range(30000)), 30000, X, rvp.REASON_WALL)
    with caplog.at_level("INFO"):
        assert _take(f) == 1
    assert f.queue[-1].est_uncached == 0 and f.queue[-1].p_only is False
    assert f.counters["rvp_text_unpriced"] == 1
    assert any("RESUME-VIA-P CLASS rid=pdflip-9-9" in m and "u_src=unpriced" in m for m in caplog.messages)
    assert not any("u_src=d_rows" in m for m in caplog.messages)


def test_streamed_output_past_the_probed_prompt_counts_as_uncached(d_env):
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-3-3"] = _price(50000, 49900)   # u=100 at the arrival
    rvp.write_request("pdflip-3-3", list(range(50300)), 50300, X, rvp.REASON_WALL)   # +300 streamed output tokens
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 400


def test_the_vision_leg_stays_p_only_r9(d_env):
    """REASON_VISION (an image in D's uncached tail, D has no tower): P-only, R9 unchanged."""
    f = _front()
    f._rvp_state()
    f._rvp_bodies = {"pdflip-238-3262": ("/v1/messages", {"messages": []})}
    rvp.write_request("pdflip-238-3262", list(range(100)), 50, X, rvp.REASON_VISION)
    assert _take(f) == 1
    assert f.queue[-1].p_only is True


# ---------------------------------------------------------------------------------------------------- fresh (X-REQUEUE)
WALL_BODY = (b'{"error": "W50 PdFlipTpPrefillExceeded: this group may prefill at most 12288 uncached tokens itself; this '
             b'request\'s extent after prefix matching is 85494. Refused by name D-WALL-HEAD: the extent is the device '
             b'rows this request needs; the host arena takes no backup"}')


def test_live_22_78_a_fresh_wall_refusal_of_a_cached_image_request_is_priced_at_u():
    """rc19 pdflip-22-78: 5 images cached, u=246; the wall's 85494 rows are no uncached extent."""
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-22-78"] = _price(85494, 85248)
    assert F._d_refusal_extent(WALL_BODY) == 85494
    u = F.Front._wall_extent_to_u(f, "pdflip-22-78", WALL_BODY, F._d_refusal_extent(WALL_BODY))
    assert u == 246
    assert pp.classify_arrival(u, 12288, 0.5, False) == pp.CLASS_KURZ


def test_an_x_refusal_body_keeps_ds_extent():
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["r"] = _price(20000, 19000)
    body = b"W50 PdFlipTpPrefillExceeded: this request's extent after prefix matching is 15077."
    assert F.Front._wall_extent_to_u(f, "r", body, 15077) == 15077
    assert F.Front._wall_extent_to_u(f, "r", WALL_BODY, None) is None
    assert F.Front._wall_extent_to_u(f, "unknown", WALL_BODY, 85494) == 0  # NF-VISION-CACHED: no number -> unpriced


def test_the_requeue_path_calls_the_wall_conversion_before_any_use_of_the_extent():
    import inspect
    src = inspect.getsource(F.Front._requeue_after_x_refusal)
    i_parse = src.index("d_extent = _d_refusal_extent(body)")
    i_conv = src.index("Front._wall_extent_to_u(self, rid, body, d_extent,")
    i_use = src.index("handback_empty = (")
    assert i_parse < i_conv < i_use


# ---------------------------------------------------------------------------------------------------- D side, three ranks
def _wall_req(rid="pdflip-2-21", prompt=56140, out=3):
    r = types.SimpleNamespace(rid=rid, stream=True, origin_input_ids=list(range(prompt)),
                              output_ids=list(range(10 ** 6, 10 ** 6 + out)), kv_arrival_seq=1,
                              is_fast_lane=False, spill_class=None)
    r.full_untruncated_fill_ids = r.origin_input_ids + r.output_ids
    r.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **kw: None))
    return r


def _sched(rank, queue):
    sent = []
    return types.SimpleNamespace(
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288), waiting_queue=list(queue),
        tree_cache=types.SimpleNamespace(release_aborted_request=lambda rid: None),
        enable_hicache_storage=True, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: sent.append(o))),
        ps=types.SimpleNamespace(tp_rank=rank), pdflip_d_parked=[],
        pdflip_uncached_extent=lambda req, head=None: 76), sent


@pytest.mark.parametrize("wall", [True, False])
def test_three_ranks_park_alike_and_rank0_names_the_wall(d_env, monkeypatch, wall):
    """R11 / rank agreement: the reason is a replicated term (``wall_rows`` is the group verdict); every rank parks the
    same request, only TP0 writes the record, and the record says whether the extent is device rows."""
    from flliper.srt.managers import scheduler as S

    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    monkeypatch.setattr(S, "release_admission_acquired_mamba_slot", lambda req, tc, site: False)
    parked = []
    for rank in (0, 1, 2):
        req = _wall_req()
        ns, sent = _sched(rank, [req])
        S.Scheduler._pdflip_answer_x_refusals(ns, [req], None, wall_rows=({id(req): 56143} if wall else None))
        assert ns.waiting_queue == [] and sent == [] and [q.rid for q in ns.pdflip_d_parked] == ["pdflip-2-21"]
        parked.append(len(ns.pdflip_d_parked))
        got = rvp.take_requests(rvp.needs_p_dir())
        if rank == 0:
            assert len(got) == 1
            assert got[0]["reason"] == (rvp.REASON_WALL if wall else "x_refusal_midstream")
            assert got[0]["d_extent"] == (56143 if wall else 76)
        else:
            assert got == [], "rank > 0 writes nothing"
    assert parked == [1, 1, 1]


# ---------------------------------------------------------------------------------------------------- the image's place decides (R9)
def _ans(n, device, spans):
    import numpy as np
    return dmp.ProbeAnswer(ids=np.arange(n, dtype=np.int32), device_tokens=device, host_tokens=0, key_match_depth=device,
                           mm_spans=tuple(spans))


def test_mandatory_rc19_22_78_all_images_in_the_cached_prefix_is_text_no_p_only():
    """rc19 _074525 pdflip-22-78: 5 images ending at 24639, credit 85248 of 85494: no tower, kurz u=246 (S66, user 19.10.)."""
    xp = dmp.exact_price(_ans(85494, 0, [(1000, 5000), (6000, 10000), (11000, 15000), (16000, 20000), (21000, 24639)]),
                         85248)
    assert xp.image_in_tail is False and dmp.class_uncached(xp) == 246
    assert pp.classify_arrival(dmp.class_uncached(xp), 12288, 0.5, False) == pp.CLASS_KURZ


def test_mandatory_rc26_20_269_a_new_image_in_the_uncached_tail_is_r9():
    """rc26 _130752 pdflip-20-269 (13:45:10Z): the predecessor pdflip-20-265 (84848 tokens) had NO image; this turn brings
    one ending at 85859 of 85869 (P log ``PROBE-WAIT ... image_end=85859``, ``P-KNOWN-IMAGE verdict=miss``) past D's 84800:
    an image in the UNCACHED tail -- p_only (R9) and priced at P's number (1901), NOT the S107 class. Whether D may run it
    with a briefly loaded tower is the user decision of 19.10. (R9 addendum, from step S3b); until then R9 stands."""
    xp = dmp.exact_price(_ans(85869, 84800, [(84980, 85859)]), 83968)
    assert xp.image_in_tail is True and dmp.class_uncached(xp) == 1901
    assert pp.classify_arrival(1901, 5223, 0.5, True) == pp.CLASS_P_ONLY


def test_mandatory_rc26_22_271_and_22_272_the_new_image_and_its_sibling_are_r9_at_the_decision():
    """rc26 pdflip-22-271 (13:45:33Z, 2 images, the newest ending at 86846 of 86856, covered 85824, ``P-KNOWN-IMAGE
    verdict=miss``): a new image in the uncached tail -> R9. pdflip-22-272 (13:45:35Z) extends 22-271 by 154 tokens
    (common 86852) and carries the SAME image: at its decision 22-271 was not prefilled yet, so the image lay in its
    uncached tail too (exact at the decision, user 19.10.) -> R9; P then computed 162 because 22-271 filled P's tree in
    the same batch (the sibling class, ``prefix-shared=1028: P counts once``). Not the S107 class either."""
    a = dmp.exact_price(_ans(86856, 85824, [(80000, 81000), (85900, 86846)]), 85824)
    b = dmp.exact_price(_ans(87010, 85824, [(80000, 81000), (85900, 86846)]), 85824)
    assert a.image_in_tail and b.image_in_tail
    assert (dmp.class_uncached(a), dmp.class_uncached(b)) == (1032, 1186)
    # once 22-271's prefill is in the store, the same request is text: the image in the cached prefix (S66)
    after = dmp.exact_price(_ans(87010, 86848, [(80000, 81000), (85900, 86846)]), 86848)
    assert after.image_in_tail is False and dmp.class_uncached(after) == 162
