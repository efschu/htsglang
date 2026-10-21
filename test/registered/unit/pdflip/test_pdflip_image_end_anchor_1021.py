"""IMAGE-END TRACK (21.10.2026): a request that shares a prompt through the first page boundary past its last
image finds a recurrent anchor there -- the image stays inside its covered prefix, no second D->P flip.

Metal (NF rc30, boot ..._1d822bd150_1021_063252, session a4c8b097ac):
  fr 09:06:08 'D-PROBE rid=pdflip-127-1253 n=63879 ... store=63680 covered=63680 uncached_d=199 image_in_tail=1
              images=2' -> flip to P
  P  09:06:15 'W102 PdFlipVisionStage PROBE-WAIT rid=pdflip-127-1253 promise=63680 ... image_end=63714'
  P  09:06:16 '#969 EXTENT ... (pdflip-127-1253, 63680, 63879, 63680, 199)'          (ONE step)
  P  09:06:18 'END-ANCHOR ... rid=pdflip-127-1253 tokens=63879 anchor=63872'          (the only track)
  fr 09:06:20 'STORE-PRESENCE src=p_flush ... anchors=[pdflip-127-1253:63872]'      (P DID write back)
  fr 09:07:07 'D-PROBE rid=pdflip-130-1258 n=63880 device=63680 store=63680 covered=63680 uncached_d=200
              image_in_tail=1 images=2' -> a SECOND flip to P
  P  09:07:13 'PDFLIP-CLAIM-SHORT rid=pdflip-130-1258 pp=0 asked=998 kv=996 claimed=995 lost=1 short=3 by=mamba
              tokens=63680'
The sibling shares the KV through page 996 (63744) and diverges inside page 997; the recurrent state existed only at
63680 and 63872: claim 63680 < image_end 63714 -> image in the uncached tail (R9) -> flip, 64 stored tokens
recomputed. With an anchor at ceil_page(63714) = 63744 the claim is 63744 >= 63714: priced as text, kurz, D.

Pinned (hermetic, CPU): A. geometry (ceil_page of the step's last image end, from the pad ids); B. the rc30 step
plans the image track at 63744 and turn/close do not (they round onto the step start); C. the repro price: with
the anchor the sibling is kurz on D, without it P-only; D. PP3 rank agreement; E. the REAL UnifiedRadixCache: the
sibling resumes at the image end with the state the track wrote; F. wiring (radix cache consumes it at both
inserts, no own switch, no rank-local state).
"""

import inspect
from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from flliper.srt.pdflip import d_match_probe as dmp
from flliper.srt.pdflip import phase_policy as pp
from flliper.srt.pdflip import turn_anchor as ta
from flliper.srt.pdflip import twin_anchor as twa
from flliper.test.ci.ci_register import register_cpu_ci

from test_pdflip_close_fork_anchor_1020 import (  # noqa: E402  (the shared real-tree harness)
    CLOSE_PENDING, GEN, GEN_PENDING, IM, NL, TURN_PENDING, _Alloc, _claim, _fixture, _ids, _plan,
    _req, _sreq, _state, _write_kv, group_p,  # noqa: F401  (pytest fixture)
)

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

IMAGE_PENDING = "_pdflip_image_pending"   # turn_anchor.IMAGE_PENDING_ATTR, by name
PAD = 1_000_000                           # MM_PAD_SHIFT_VALUE

# rc30 pdflip-127-1253 / 130-1258 in tokens (page 64, FLA chunk 64)
R_PREFIX, R_N, R_PAGE = 63680, 63879, 64
R_MAIN = 63872                            # END-ANCHOR anchor=63872
R_IMAGE_END = 63714                       # PROBE-WAIT image_end=63714
R_IMAGE_ROWS = (1444, 1444)               # EMB-CACHE HIT items=2 rows=2888
R_LAST_MSG = 63718                        # the last message opens inside the step's first page
R_ANCHOR = 63744                          # ceil_page(63714) = page 996 of the sibling's KV (kv=996)
R_SIB_N = 63880
R_SIB_DIVERGE = 63750                     # inside page 997 (asked=998 kv=996)


def _rc30_ids(n=R_N, last_msg=R_LAST_MSG):
    ids = _ids(n, last_msg)
    e = R_IMAGE_END
    b = e - R_IMAGE_ROWS[1]
    a = b - 2 - R_IMAGE_ROWS[0]
    ids[a:b - 2] = [PAD + 11] * R_IMAGE_ROWS[0]
    ids[b:e] = [PAD + 22] * R_IMAGE_ROWS[1]
    return ids


R_SRC = _rc30_ids()


@pytest.fixture(autouse=True)
def _ledger(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    twa._reset_for_test()
    yield
    twa._reset_for_test()


# -- A. geometry -------------------------------------------------------------------
def test_the_image_end_is_read_off_the_pad_ids():
    assert ta.image_end_in(array("q", R_SRC), R_PREFIX, R_N) == R_IMAGE_END
    assert ta.image_end_in(R_SRC, R_PREFIX, R_N) == R_IMAGE_END              # list form too
    assert ta.image_end_in(array("q", R_SRC), R_IMAGE_END, R_N) is None       # no image in [e, N)
    assert ta.image_end_pos(array("q", R_SRC), R_PREFIX, R_N, R_PAGE) == R_ANCHOR
    aligned = list(R_SRC)
    aligned[R_IMAGE_END:R_ANCHOR] = [PAD + 22] * (R_ANCHOR - R_IMAGE_END)   # an image ending ON a boundary
    assert ta.image_end_pos(aligned, R_PREFIX, R_N, R_PAGE) == R_ANCHOR
    assert ta.image_end_in(None, 0, 10) is None


# -- B. the planner ----------------------------------------------------------------
def test_the_rc30_step_plans_the_image_track_and_no_turn_or_close_track():
    alloc = _Alloc([7, 8, 9])
    req = _sreq(R_SRC, rid="pdflip-127-1253")
    desc = _plan(req, alloc, R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)
    assert desc is not None and desc.targets == [R_ANCHOR]
    # the premise of the gap: turn and close positions round down onto the step start
    assert ta.anchor_pos(R_SRC, IM, R_PAGE) == R_PREFIX
    assert getattr(req, TURN_PENDING, None) is None and getattr(req, CLOSE_PENDING, None) is None
    t, slot, d, s0, s1 = getattr(req, IMAGE_PENDING)
    assert (t, d, s0, s1) == (R_ANCHOR, desc, R_PREFIX, R_N)
    assert ta.has_pending(req)
    assert [(k, int(p[0])) for k, p in ta.pop_all_pending(req)] == [("image", R_ANCHOR)]
    assert not ta.has_pending(req)


@pytest.mark.parametrize("why,kw", [
    ("image ends in the END anchor's page", dict(main=R_IMAGE_END + 10)),
    ("protected prefix already past it", dict(cpl=R_ANCHOR)),
    ("turn anchor unarmed", dict(tok=None)),
    ("off the FLA grid", dict(chunk=96)),
    ("decode step", dict(decoding=True)),
])
def test_no_image_track(why, kw):
    alloc = _Alloc([7])
    req = _sreq(R_SRC, cpl=kw.get("cpl", 0))
    if kw.get("decoding"):
        req.output_ids = [5]
    _plan(req, alloc, R_PREFIX, R_N, kw.get("main", R_MAIN), tok=kw.get("tok", IM),
          chunk=kw.get("chunk", 64), page=R_PAGE)
    assert getattr(req, IMAGE_PENDING, None) is None, why


def test_a_text_step_plans_no_image_track():
    ids = _ids(R_N, R_LAST_MSG)
    req = _sreq(ids)
    _plan(req, _Alloc([7]), R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)
    assert getattr(req, IMAGE_PENDING, None) is None


def test_no_slot_no_plan_and_a_stale_plan_is_freed(monkeypatch):
    req = _sreq(R_SRC)
    _plan(req, _Alloc([]), R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)
    assert getattr(req, IMAGE_PENDING, None) is None
    alloc = _Alloc([4, 5])
    _plan(req, alloc, R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)
    first = int(getattr(req, IMAGE_PENDING)[1])
    _plan(req, alloc, R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)     # unconsumed: freed, re-drawn
    assert first in alloc.freed


def test_a_position_another_track_writes_is_one_slot():
    alloc = _Alloc([3, 4, 5])
    req = _sreq(R_SRC)
    _plan(req, alloc, R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE, fork=R_ANCHOR)
    assert int(getattr(req, "_pdflip_fork_pending")[0]) == R_ANCHOR
    assert getattr(req, IMAGE_PENDING, None) is None


# -- C. the repro price -------------------------------------------------------------
def _sib_ans(device):
    return dmp.ProbeAnswer(ids=np.arange(R_SIB_N, dtype=np.int32), device_tokens=device, host_tokens=0,
                           key_match_depth=device, mm_spans=((R_IMAGE_END - 1444, R_IMAGE_END),
                                                             (R_IMAGE_END - 2890, R_IMAGE_END - 1446)))


def test_repro_130_1258_without_the_anchor_is_p_only_and_with_it_kurz_on_d():
    X, HALF = 6153, 3076                                        # SET-VERDICT X=6153 halfX=3076
    # the metal: the claim stops at 63680 (lost=1 by=mamba)
    before = dmp.exact_price(_sib_ans(0), 63680)
    assert (before.covered, before.uncached, before.image_in_tail) == (63680, 200, True)
    assert pp.classify_arrival(dmp.class_uncached(before), X, 0.5, True) == pp.CLASS_P_ONLY
    # with the image-end anchor the store's anchored depth reaches page 996 = the KV the sibling shares
    claim = (R_SIB_DIVERGE - 1) // R_PAGE * R_PAGE
    assert claim == R_ANCHOR
    after = dmp.exact_price(_sib_ans(0), claim)
    assert (after.covered, after.uncached, after.image_in_tail) == (R_ANCHOR, R_SIB_N - R_ANCHOR, False)
    assert after.uncached < HALF
    assert pp.classify_arrival(dmp.class_uncached(after), X, 0.5, False) == pp.CLASS_KURZ


# -- D. PP rank agreement -----------------------------------------------------------
def test_three_pp_stages_plan_the_same_position():
    plans = []
    for free in ([3, 4], [17, 18], [40, 41]):
        req = _sreq(R_SRC)
        desc = _plan(req, _Alloc(free), R_PREFIX, R_N, R_MAIN, chunk=64, page=R_PAGE)
        plans.append((desc.targets, getattr(req, IMAGE_PENDING)[0], getattr(req, IMAGE_PENDING)[3:]))
    assert plans[0] == plans[1] == plans[2] == ([R_ANCHOR], R_ANCHOR, (R_PREFIX, R_N))


# -- E. the real tree (page 1) -------------------------------------------------------
N = 120
TURN = 100
IMG = (60, 80)                       # image placeholders [60, 80)
S_IMAGE, S_END, S_OTHER = 3.25, 7.5, 1.5


def _tree_src():
    ids = _ids(N, TURN)
    ids[IMG[0]:IMG[1]] = [PAD + 7] * (IMG[1] - IMG[0])
    return ids


SRC = _tree_src()
SIB = SRC[:IMG[1] + 5] + [600_000 + i for i in range(30)]     # shares the image, diverges after it


def _one_forward(fx, image=True):
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    orig = ta._note_image_step
    if not image:
        ta._note_image_step = lambda batch, desc, *a, **k: desc
    try:
        desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                            track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    finally:
        ta._note_image_step = orig
    for attr in (TURN_PENDING, GEN_PENDING, CLOSE_PENDING):
        if getattr(req, attr, None) is not None:
            _state(fx, getattr(req, attr)[1], S_OTHER)
    if image:
        assert getattr(req, IMAGE_PENDING)[0] == IMG[1]
        _state(fx, getattr(req, IMAGE_PENDING)[1], S_IMAGE)
    if desc is not None:
        desc.done.add("gdn")
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_sibling_resumes_at_the_image_end(group_p, bigram):
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, SIB) == (IMG[1], pytest.approx(S_IMAGE))
    fx0 = _fixture(bigram)
    _one_forward(fx0, image=False)
    assert _claim(fx0, SIB)[0] < IMG[1], "the gap itself, rc30: the claim falls behind the image"


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_end_anchor_is_unchanged(group_p, bigram):
    later = SRC + [5, 6, 7]
    fx = _fixture(bigram)
    _one_forward(fx)
    fx0 = _fixture(bigram)
    _one_forward(fx0, image=False)
    assert _claim(fx, later) == _claim(fx0, later) == (N - (1 if bigram else 0), pytest.approx(S_END))


# -- F. wiring -----------------------------------------------------------------------
def test_the_radix_cache_asks_for_an_image_plan_at_both_inserts():
    from flliper.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache)
    assert src.count("_IMAGE_PENDING_ATTR") == 2
    assert urc._IMAGE_PENDING_ATTR == ta.IMAGE_PENDING_ATTR == IMAGE_PENDING


def test_no_own_switch_and_no_rank_local_state():
    src = inspect.getsource(ta._note_image_step) + inspect.getsource(ta.image_end_pos) + \
        inspect.getsource(ta.image_end_in)
    assert "envs." not in src and "environ" not in src
    for name in ("tp_rank", "pp_rank", "get_rank", "time.", "random"):
        assert name not in src
    assert "_note_image_step" in inspect.getsource(ta.note_step)
