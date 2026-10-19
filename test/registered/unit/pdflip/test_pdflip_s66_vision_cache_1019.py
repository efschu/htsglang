# SPDX-License-Identifier: Apache-2.0
"""S66 (NF, Vision auf P): no tower and no image processor for an image the
prefix P reads already holds.

rc15 P log, boot ..._1019_092806, pdflip-12-340 (images end at token 20556):
  10:08:06  PDFLIP-MM-TOKENIZE process_ms=8505 total_ms=8710   (P's processor)
  10:08:06  PDFLIP-CLAIM-SHORT pp=0 claimed=660 tokens=42240    (PP0's store probe)
  10:08:06  #257 PROBE-HOLD pages=660
  10:08:09  W102 PdFlipVisionStage run=4 ... load 251, encode 1675   (the tower)
  10:08:09  #1416d TOLD-PROBE told=42240 reused=prefetch_hit pages=660
P admitted 42240 > 20556: both the processor and the tower were avoidable.
Root: ``stage_skip_verdict`` waits for an outstanding told only when the LOCAL
match covers the image -- 0 on a P that reads from the store -- and staged.

Half 1 (PROBE-WAIT): PP0's own store probe (or its read's completed prefix)
holds the verdict for the told. Half 2 (P-KNOWN-IMAGE): P's tokenizer builds a
request whose images are all known without pixels; PP0 skips it on the told or
fetches the pixels from P's tokenizer before it stages -- never admits it
without rows.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import io
import logging
import os
import pickle
import types
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import p_known_image as K  # noqa: E402
from flliper.srt.pdflip import vision_rank_runner as V  # noqa: E402
from flliper.srt.pdflip import vision_verdict as VV  # noqa: E402

RID = "pdflip-12-340"
IMAGE_END = 20556          # one past the last image position (front: image_end=20556)
PAGE = 64


class _Item:
    """A MultimodalDataItem stand-in: offsets, hash, pixels or not."""

    def __init__(self, offsets, feature=None, hash=None, emb=None, grid=None):
        self.offsets = offsets
        self.feature = feature
        self.hash = hash
        self.precomputed_embeddings = emb
        n = sum(e - a + 1 for a, e in offsets)
        self.model_specific_data = {"image_grid_thw": torch.tensor([grid or [1, 2, 2 * n]])}

    def is_image(self):
        return True


def _req(rid=RID, featureless=False, prefix=0, host=0, ends=(5000, IMAGE_END - 1)):
    items = []
    start = 100
    for i, end in enumerate(ends):
        items.append(_Item([(start, end)], feature=None if featureless else torch.zeros(2),
                           hash=1000 + i if featureless else None))
        start = end + 10
    return SimpleNamespace(rid=rid, multimodal_inputs=SimpleNamespace(mm_items=items),
                           prefix_indices=[0] * prefix, host_hit_length=host,
                           _prefetch_registered_prefix_len=0)


class _Tree:
    def __init__(self, completed=None, ongoing=()):
        self._completed = completed or {}
        self._ongoing = set(ongoing)

    def completed_prefetch_tokens(self, rid):
        return self._completed.get(rid)

    def prefetch_is_ongoing(self, rid):
        return rid in self._ongoing


class _Ipc:
    def __init__(self):
        self.sent = []

    def send_output(self, obj, recv=None):
        self.sent.append(obj)


def _sched(told=None, held=(), probe=None, completed=None, ongoing=(), armed=True, wq=()):
    cc = SimpleNamespace(page_size=PAGE)
    if probe is not None:
        cc._pdflip_probe_hit_pages = dict(probe)
    s = SimpleNamespace(
        _pdflip_store_told_armed=armed,
        _pdflip_store_told=dict(told or {}),
        _pdflip_store_held={r: object() for r in held},
        tree_cache=_Tree(completed, ongoing),
        cache_controller=cc,
        ipc_channels=SimpleNamespace(send_to_tokenizer=_Ipc()),
        waiting_queue=list(wq),
        _pdflip_vision_origin_aborts=[],
        _pdflip_vision_refused=set(),
    )
    return s


# ---- Half 1: PROBE-WAIT ---------------------------------------------------------------

def test_pdflip_12_340_waits_for_the_told_instead_of_staging(caplog):
    """THE case: told outstanding, local 0, PP0's probe answered 660 pages = 42240."""
    caplog.set_level(logging.INFO)
    r = _req()
    s = _sched(held=[RID], probe={RID: 660}, ongoing=[RID])
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 42240, "probe")
    stage, wait = V.skip_cached(s, [r])
    assert (stage, wait) == ([], [r]), "no tower ahead of the told"
    assert any(m.startswith(f"W102 PdFlipVisionStage PROBE-WAIT rid={RID} promise=42240 src=probe "
                            f"image_end={IMAGE_END}") for m in caplog.messages)
    # next pass: the told is published (the read completed the probe's span)
    s._pdflip_store_told[RID] = 42240
    stage, wait = V.skip_cached(s, [r])
    assert (stage, wait) == ([], []) and r._pdflip_vision_skip
    assert V.pending_items(r) == [] and V.unstaged_items(r) == []


def test_dormant_held_read_finished_during_the_flip_waits_on_its_completed_prefix():
    """pdflip-14-492 (092806, 10:26:43-52): read done during the flip, told only at
    the wake pass after the vision pass -- the completed prefix (61952) decides."""
    r = _req(rid="pdflip-14-492", ends=(21757,))
    s = _sched(held=["pdflip-14-492"], completed={"pdflip-14-492": 61952})
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 61952, "read")


def test_rc16_paced_form_waits_between_read_ahead_and_admit():
    """rc16 boot 121629, P PP0, pdflip-2-9: PROBE-HOLD pages=2684 (12:20:41), tower at
    12:20:44 (load 231, encode 1749), then TOLD-PROBE / PACED-TOLD told=171776 and the
    Admit; admitted 171776, images end ~21k. Between read-ahead and Admit the rid is in
    ``pacing`` (not ``held``) and ``_pdflip_store_told`` is still empty."""
    r = _req(rid="pdflip-2-9", ends=(21757,))
    s = _sched(held=[])
    s._pdflip_told_pacing = {"pdflip-2-9": SimpleNamespace(told=171776, absolute=True)}
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 171776, "paced")
    s._pdflip_told_pacing = {}
    s._pdflip_store_told["pdflip-2-9"] = 171776  # the Admit
    assert V.stage_skip_verdict(s, r) == (V.SKIP, 171776, "told")


def test_paced_fallback_admit_zero_stages():
    """S66b A1: a paced told short of the image waits for the Admit (an S64 raise may still lift it); the PF
    fallback Admit 0 then stages."""
    r = _req(rid="pdflip-2-9", ends=(21757,))
    s = _sched(held=[])
    s._pdflip_told_pacing = {"pdflip-2-9": SimpleNamespace(told=8192, absolute=True)}
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 8192, "paced-raise")
    s._pdflip_told_pacing = {}
    s._pdflip_store_told["pdflip-2-9"] = 0  # PF fallback Admit
    assert V.stage_skip_verdict(s, r) == (V.STAGE, 0, "told")


def test_review_s66b_paced_raise_admit_above_image_end_skips_without_fetch(monkeypatch):
    """Review S66b A1 (reviewer test /tmp/rv66b/test_rv66b_paced_raise_1019.py as template): paced told 72128 <
    image_end 72215, the S64 raise (default on) admits 72256 >= image_end -> SKIP, no pixel fetch, no tower."""
    from flliper.srt.managers import pdflip_told_raise as R

    assert R.raise_on({}) is True
    r = _req(rid="rv-1", featureless=True, ends=(72214,))
    s = _pass_sched(r, held=[])
    s._pdflip_told_pacing = {"rv-1": SimpleNamespace(told=72128, absolute=True)}
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower before the Admit")):
        parked = V.vision_rank_pass(s)
        assert [x for _, x in parked] == [r] and s.ipc_channels.send_to_tokenizer.sent == []
        V.vision_unpark(s, parked)
        s._pdflip_told_pacing = {}
        s._pdflip_store_told["rv-1"] = 72256  # Admit with raise
        V.vision_rank_pass(s)
    assert r._pdflip_vision_skip and s.ipc_channels.send_to_tokenizer.sent == []


def test_paced_short_stages_at_once_only_when_no_raise_can_come(monkeypatch):
    from flliper.srt.managers import pdflip_told_raise as R

    r = _req(rid="rv-2", ends=(72214,))
    s = _sched(held=[])
    s._pdflip_told_pacing = {"rv-2": SimpleNamespace(told=72128, absolute=True)}
    monkeypatch.setenv(R.ENV_RAISE, "0")
    assert V.stage_skip_verdict(s, r)[0] == V.STAGE, "switch off: the Admit can only lower it"
    monkeypatch.setenv(R.ENV_RAISE, "1")
    setattr(s, R._ATTR, {"rv-2": SimpleNamespace(target=72192, state="failed")})
    assert V.stage_skip_verdict(s, r)[0] == V.STAGE, "the rid's one raise failed"
    setattr(s, R._ATTR, {"rv-2": SimpleNamespace(target=72192, state="issued")})
    assert V.stage_skip_verdict(s, r)[0] == V.STAGE, "the raise targets a depth short of the image"
    setattr(s, R._ATTR, {"rv-2": SimpleNamespace(target=72256, state="issued")})
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 72128, "paced-raise")


def test_a_probe_not_answered_yet_waits_while_the_read_is_registered():
    r = _req()
    s = _sched(held=[RID], ongoing=[RID])
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 0, "probe-pending")


def test_a_probe_short_of_the_images_still_waits_for_the_told():
    """S66b (user order 19.10.): while a told is outstanding neither stage nor fetch runs -- the told decides."""
    r = _req()
    s = _sched(held=[RID], probe={RID: 300}, ongoing=[RID])  # 19200 < 20556
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 19200, "told-pending")
    assert V.skip_cached(s, [r]) == ([], [r])
    s._pdflip_store_told[RID] = 19200
    assert V.skip_cached(s, [r]) == ([r], []), "the told is short: the tower runs"


def test_rc17_pdflip_1_13_deferred_twin_known_image_waits_for_the_told_no_fetch_no_tower(caplog):
    """rc17 boot 171636, P PP0, pdflip-1-13 (image_end 72215), in this order:
    17:21:34 P-KNOWN-IMAGE hit (no pixels) -> 17:21:35 W102k FETCH covered=0 src=local -> 17:21:37 FETCH-ANSWER
    -> 17:21:38 ATTACHED -> 17:21:40 tower -> 17:21:52 #TW TWIN-REGISTER twin=True, TOLD told=81664 -> P computed 61.
    The twin was held by #1400 with no store read (sibling in flight): no probe, no open read."""
    caplog.set_level(logging.INFO)
    rid = "pdflip-1-13"
    r = _req(rid=rid, featureless=True, ends=(72214,))
    s = _pass_sched(r, held=[rid])
    s._tw_state = SimpleNamespace(waits={rid: 0.0})
    with mock.patch("flliper.srt.pdflip.p_twin_defer.is_deferred", lambda sch, x: x in sch._tw_state.waits), \
            mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower ran before the told")):
        parked = V.vision_rank_pass(s)
        assert [x for _, x in parked] == [r]
        assert s.ipc_channels.send_to_tokenizer.sent == [], "no pixel fetch while the told is outstanding"
        assert VV.take_verdicts(s) == []
        assert any(f"PROBE-WAIT rid={rid} promise=0 src=twin-deferred" in m for m in caplog.messages)
        V.vision_unpark(s, parked)
        # the sibling finished: the twin registers at head 81664 and its told is published
        s._tw_state.waits.pop(rid)
        s._pdflip_store_held.pop(rid)
        s._pdflip_store_told[rid] = 81664
        V.vision_rank_pass(s)
    assert r._pdflip_vision_skip and s.ipc_channels.send_to_tokenizer.sent == []
    assert [v.rid for v in VV.take_verdicts(s)] == [rid]


def test_registration_head_counts_into_the_promise():
    r = _req()
    r._prefetch_registered_prefix_len = 8576
    s = _sched(held=[RID], probe={RID: 190})  # 8576 + 12160 = 20736 >= 20556
    assert V.stage_skip_verdict(s, r) == (V.WAIT, 20736, "probe")


def test_no_told_coming_means_no_wait():
    r = _req()
    s = _sched(held=[], probe={RID: 660})
    assert V.stage_skip_verdict(s, r)[0] == V.STAGE


def test_never_skips_on_the_probe_only_on_the_told():
    """Danger direction 'skip without read KV': the probe promised 42240, the read
    ended short (told 19200 < image_end) -> the tower runs."""
    r = _req()
    s = _sched(told={RID: 19200}, held=[RID], probe={RID: 660})
    assert V.stage_skip_verdict(s, r) == (V.STAGE, 19200, "told")
    assert V.skip_cached(s, [r]) == ([r], []) and not getattr(r, "_pdflip_vision_skip", False)


def test_switch_off_restores_the_old_verdict(monkeypatch):
    monkeypatch.setenv(V.VISION_PROBE_WAIT_ENV, "0")
    r = _req()
    s = _sched(held=[RID], probe={RID: 660})
    assert V.stage_skip_verdict(s, r)[0] == V.STAGE


def test_a_refuted_skip_stages_next_instead_of_re_skipping():
    r = _req(prefix=8000)
    r._pdflip_vision_skip = True
    assert V.skip_still_covered(None, r) is False
    s = _sched(told={RID: 42240})
    assert V.stage_skip_verdict(s, r) == (V.STAGE, 0, "refuted")


# ---- the pass: PP0 holds, no verdict on the wire, followers hold too ------------------

def _pass_sched(r, **kw):
    s = _sched(wq=[r], **kw)
    s.pdflip_dormant = False
    s._pdflip_vision_inflight = None
    s._pdflip_vision_runs = 0
    s._pdflip_vision_arm_refusal = ""
    s.ps = SimpleNamespace(pp_size=3, pp_rank=0)
    VV.arm(s)
    return s


def test_vision_pass_holds_a_probe_wait_out_of_this_pass_and_puts_no_verdict_on_the_wire():
    r = _req()
    s = _pass_sched(r, held=[RID], probe={RID: 660})
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower ran")):
        parked = V.vision_rank_pass(s)
    assert [x for _, x in parked] == [r] and s.waiting_queue == []
    assert VV.take_verdicts(s) == [], "followers must not release what PP0 still decides"
    V.vision_unpark(s, parked)
    s._pdflip_store_told[RID] = 42240
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower ran")):
        parked = V.vision_rank_pass(s)
    assert r._pdflip_vision_skip
    assert [v.rid for v in VV.take_verdicts(s)] == [RID], "the skip goes out as PP0's verdict"


def test_a_follower_never_runs_the_verdict_and_holds_until_pp0s_release():
    r = _req(featureless=True)
    f = SimpleNamespace(waiting_queue=[r], pdflip_dormant=False, ps=SimpleNamespace(pp_size=3, pp_rank=1))
    VV.arm(f)
    with mock.patch.object(V, "stage_skip_verdict", side_effect=AssertionError("follower decided")):
        parked = VV.follower_pass(f)
    assert [x for _, x in parked] == [r]
    VV.absorb(f, VV.PdFlipVisionVerdict(rid=RID))
    f.waiting_queue = [r]
    assert VV.follower_pass(f) == []


def test_wiring_only_pp0_runs_the_vision_pass():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S)
    assert 'if getattr(self, "_pdflip_vision_rank_stage", False):' in src
    assert "_vision_parked = vision_rank_pass(self)" in src
    assert "_vision_parked = follower_pass(self)" in src
    assert inspect.signature(V.arm_rank_stage)  # PP0-only arming (pp_rank != 0 -> False)
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, tp_size=1))
    assert V.arm_rank_stage(s, env={"FLLIPER_PDFLIP_VISION": "transient"}) is False


# ---- Half 2: P-KNOWN-IMAGE, scheduler side --------------------------------------------

def test_featureless_items_are_pending_and_hold_the_verdict():
    r = _req(featureless=True)
    assert len(K.featureless_items(r)) == 2
    assert V.unstaged_items(r) == [] and len(V.pending_items(r)) == 2


def test_known_images_inside_the_told_skip_without_any_pixels():
    r = _req(featureless=True)
    s = _sched(told={RID: 42240})
    assert V.skip_cached(s, [r]) == ([], []) and r._pdflip_vision_skip
    assert s.ipc_channels.send_to_tokenizer.sent == [], "no fetch for a covered image"
    assert K.featureless_items(r) == []


def test_known_images_outside_the_told_fetch_pixels_and_are_held(caplog):
    caplog.set_level(logging.INFO)
    r = _req(featureless=True)
    s = _pass_sched(r, told={RID: 19200})
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("staged without pixels")):
        parked = V.vision_rank_pass(s)
    sent = s.ipc_channels.send_to_tokenizer.sent
    assert [type(x).__name__ for x in sent] == ["PdFlipMmFeatureFetchReq"] and sent[0].rid == RID
    assert [x for _, x in parked] == [r], "held on PP0"
    assert VV.take_verdicts(s) == [], "no verdict, the followers hold it too"
    V.vision_unpark(s, parked)
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("staged without pixels")):
        V.vision_rank_pass(s)
    assert len(s.ipc_channels.send_to_tokenizer.sent) == 1, "one fetch per request"
    assert any(m.startswith("W102k PdFlipKnownImage FETCH rid=pdflip-12-340") for m in caplog.messages)


def test_mutant_queue_verdicts_on_unstaged_items_would_release_a_featureless_request():
    """Rank-agreement mutant: deciding the gate by pixels alone sends PP0's ok for a
    request PP0 still holds -- the followers would admit it on their own."""
    r = _req(featureless=True)
    s = _pass_sched(r, told={RID: 19200})
    VV.queue_verdicts(s, [r], V.unstaged_items, set())
    assert [v.rid for v in VV.take_verdicts(s)] == [RID]  # the mutant's split
    s2 = _pass_sched(_req(featureless=True), told={RID: 19200})
    VV.queue_verdicts(s2, s2.waiting_queue, V.pending_items, set())
    assert VV.take_verdicts(s2) == []  # what the code does


def _answer(r, hashes=None, ok=True, n=None):
    from flliper.srt.managers.io_struct import PdFlipMmFeaturesOutput

    items = K.featureless_items(r)
    feats = [torch.full((3,), float(i + 1)) for i in range(n if n is not None else len(items))]
    hashes = hashes if hashes is not None else [int(it.hash) for it in items]
    return PdFlipMmFeaturesOutput(rid=r.rid, ok=ok, detail="" if ok else "boom",
                            payload=pickle.dumps({"features": feats, "hashes": hashes, "offsets": []}))


def test_answer_with_equal_hashes_attaches_and_the_next_pass_stages():
    r = _req(featureless=True)
    s = _pass_sched(r, told={RID: 19200})
    V.skip_cached(s, [r])
    out = K.consume_at_origin(s, ["text-req", _answer(r)])
    assert out == ["text-req"], "the answer never rides the chain"
    assert K.featureless_items(r) == [] and len(V.unstaged_items(r)) == 2
    assert V.skip_cached(s, [r]) == ([r], []), "staged like any image"


def test_answer_with_another_hash_is_refused_never_encoded():
    r = _req(featureless=True)
    s = _sched(told={RID: 19200}, wq=[r])
    V.skip_cached(s, [r])
    assert K.apply_features(s, _answer(r, hashes=[1, 2])) == "refused"
    assert all(it.feature is None for it in r.multimodal_inputs.mm_items)
    assert s._pdflip_vision_origin_aborts and s._pdflip_vision_origin_aborts[0][1].startswith("W102k")
    assert RID in s._pdflip_vision_refused


def test_failed_or_misshaped_answers_are_refused():
    r = _req(featureless=True)
    s = _sched(told={RID: 19200}, wq=[r])
    assert K.apply_features(s, _answer(r, ok=False)) == "refused"
    r2 = _req(rid="pdflip-1-2", featureless=True)
    s2 = _sched(told={"pdflip-1-2": 19200}, wq=[r2])
    assert K.apply_features(s2, _answer(r2, n=1, hashes=[1000])) == "refused"


def test_answer_after_the_told_covered_it_is_dropped_unneeded():
    r = _req(featureless=True)
    s = _sched(told={RID: 19200}, wq=[r])
    V.skip_cached(s, [r])
    s._pdflip_store_told[RID] = 42240
    r._pdflip_vision_skip = True
    assert K.apply_features(s, _answer(r)) == "unneeded" and not s._pdflip_vision_origin_aborts


def test_fetch_without_answer_is_refused_after_the_deadline():
    r = _req(featureless=True)
    s = _sched(told={RID: 19200}, wq=[r])
    assert K.request_features(s, r, now=100.0) == "sent"
    assert K.request_features(s, r, now=100.0 + K.FETCH_DEADLINE_S - 1) == "waiting"
    assert K.request_features(s, r, now=100.0 + K.FETCH_DEADLINE_S + 1) == "refused"
    assert s._pdflip_vision_origin_aborts[0][0] == RID


def test_mutant_admitting_a_featureless_request_on_an_uncovered_told_is_caught():
    """Danger direction 'no output without rows': with told < image_end a known
    image must never leave PP0's pass as staged / skipped."""
    r = _req(featureless=True)
    s = _sched(told={RID: 19200})
    stage, wait = V.skip_cached(s, [r])
    assert stage == [] and wait == [r] and not getattr(r, "_pdflip_vision_skip", False)
    assert V.pending_items(r), "still owed: the gate keeps it on every rank"


def test_wiring_origin_hook_and_tokenizer_handler():
    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers import tokenizer_manager as T
    from flliper.srt.managers.scheduler_components import request_receiver as R

    assert "origin_consume_hook=(" in inspect.getsource(S)
    assert "self.origin_consume_hook(recv_reqs)" in inspect.getsource(R)
    src = inspect.getsource(T)
    assert "(PdFlipMmFeatureFetchReq, self._handle_pdflip_mm_feature_fetch)" in src
    assert "_item_cache = self.d_mm_item_cache or self.p_known_image_cache" in src


def test_the_two_wire_objects_round_trip_msgpack():
    from flliper.srt.managers import io_struct as IO

    for obj in (IO.PdFlipMmFeatureFetchReq(rid=RID),
                IO.PdFlipMmFeaturesOutput(rid=RID, ok=True, detail="x", payload=b"\x00\x01")):
        back = IO.msgpack_decode(IO.msgpack_encode(obj))
        assert type(back) is type(obj) and back.rid == RID
    assert back.payload == b"\x00\x01" and back.ok


# ---- Half 2: P-KNOWN-IMAGE, tokenizer side --------------------------------------------

class _Proc:
    def __init__(self, feats):
        self.feats = feats
        self.calls = 0

    async def process_mm_data_async(self, **kw):
        self.calls += 1
        return SimpleNamespace(mm_items=[_Item([(1, 2)], feature=f) for f in self.feats])


def test_fetch_features_answers_pixels_and_their_content_hashes():
    from flliper.srt.managers.mm_utils import hash_feature

    feats = [torch.arange(6, dtype=torch.float32), torch.ones(4)]
    obj = SimpleNamespace(image_data=["data:image/png;base64,xx"], audio_data=None, text="t")
    tm = SimpleNamespace(rid_to_state={RID: SimpleNamespace(obj=obj)}, mm_processor=_Proc(feats),
                         max_req_input_len=10)
    msg = asyncio.run(K.fetch_features(tm, RID))
    assert msg.ok and msg.rid == RID
    body = pickle.loads(msg.payload)
    assert body["hashes"] == [hash_feature(f) for f in feats]
    assert all(torch.equal(a, b) for a, b in zip(body["features"], feats))


def test_fetch_features_without_a_held_request_is_a_named_no():
    tm = SimpleNamespace(rid_to_state={}, mm_processor=_Proc([]), max_req_input_len=10)
    msg = asyncio.run(K.fetch_features(tm, RID))
    assert not msg.ok and "no request" in msg.detail and tm.mm_processor.calls == 0


def test_arming_only_on_a_transient_vision_group_p():
    sa = SimpleNamespace(tp_size=1, language_only=False, skip_tokenizer_init=False,
                         keep_mm_feature_on_device=False)
    mc = SimpleNamespace(is_multimodal=True)
    env_p = {"FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_VISION": "transient"}
    assert K.build_p_known_image_cache(server_args=sa, model_config=mc, mm_processor=object(),
                                       env={"FLLIPER_PDFLIP_GROUP": "D"}) is None
    assert K.armed_reason(server_args=sa, model_config=mc, mm_processor=object(),
                          env=env_p) == "no Qwen-VL processor"
    assert "not a transient" in K.armed_reason(server_args=sa, model_config=mc, mm_processor=object(),
                                               env={"FLLIPER_PDFLIP_GROUP": "P"})
    assert K.armed_reason(server_args=sa, model_config=mc, mm_processor=object(),
                          env=dict(env_p, **{K.ENV: "0"})) == f"{K.ENV}=0"


# ---- real Qwen processor of the NF checkpoint: hit without pixels, fetch equal ---------

CKPT = os.environ.get("PDFLIP_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "preprocessor_config.json")) and \
    os.path.isfile(os.path.join(CKPT, "tokenizer.json"))


def _png(color, size) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


PROMPT = ("<|im_start|>user\nZwei Bilder:\n<|vision_start|><|image_pad|><|vision_end|> und "
          "<|vision_start|><|image_pad|><|vision_end|> ?<|im_end|>\n<|im_start|>assistant\n")


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class RealProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flliper.srt.managers.multimodal_processor import get_mm_processor, import_processors
        from flliper.srt.runtime_context import get_context
        from flliper.srt.utils.hf_transformers.config import get_config
        from flliper.srt.utils.hf_transformers.processor import get_processor

        cls._override = get_context().override_server_args(base_gpu_id=0, mm_process_config={})
        sa = cls._override.install()
        import_processors("flliper.srt.multimodal.processors")
        cls.mp = get_mm_processor(get_config(CKPT, trust_remote_code=True), sa,
                                  get_processor(CKPT, trust_remote_code=True), "default")
        cls.tok = cls.mp._tokenizer

    @classmethod
    def tearDownClass(cls):
        cls._override.restore()

    def test_p_hit_leaves_without_pixels_and_the_fetch_brings_the_same_image(self):
        from flliper.srt.managers.io_struct import GenerateReqInput
        from flliper.srt.managers.schedule_batch import MultimodalInputs

        imgs = [_png((200, 10, 10), (64, 48)), _png((10, 200, 10), (40, 72))]
        sa = SimpleNamespace(tp_size=1, language_only=False, skip_tokenizer_init=False,
                             keep_mm_feature_on_device=False)
        env = {"FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_VISION": "transient"}
        cache = K.build_p_known_image_cache(server_args=sa, model_config=SimpleNamespace(is_multimodal=True),
                                            mm_processor=self.mp, env=env)
        self.assertIsNotNone(cache)

        def via(rid):
            obj = GenerateReqInput(text=PROMPT, image_data=list(imgs), rid=rid)
            return obj, asyncio.run(cache.process(obj=obj, prompt=PROMPT,
                                                  compact_ids=self.tok(PROMPT)["input_ids"],
                                                  max_req_input_len=10 ** 6))

        _o1, first = via("pdflip-1-1")
        ref_feats = [it.feature.clone() for it in first.mm_items]
        with mock.patch.object(self.mp, "process_mm_data_async",
                               side_effect=AssertionError("P's processor ran on a known image")):
            obj2, second = via("pdflip-1-2")
        self.assertTrue(all(it.feature is None for it in second.mm_items))
        mm = MultimodalInputs.from_processor_output(copy.deepcopy(second))
        req = SimpleNamespace(rid="pdflip-1-2", multimodal_inputs=mm, prefix_indices=[], host_hit_length=0)
        self.assertEqual(len(K.featureless_items(req)), 2)
        # PP0: told short of the images -> fetch; P's tokenizer answers from the held request
        s = _sched(told={"pdflip-1-2": 0}, wq=[req])
        self.assertEqual(V.skip_cached(s, [req]), ([], [req]))
        tm = SimpleNamespace(rid_to_state={"pdflip-1-2": SimpleNamespace(obj=obj2)}, mm_processor=self.mp,
                             max_req_input_len=10 ** 6)
        msg = asyncio.run(K.fetch_features(tm, "pdflip-1-2"))
        self.assertTrue(msg.ok, msg.detail)
        self.assertEqual(K.apply_features(s, msg), "attached")
        self.assertEqual(len(V.unstaged_items(req)), 2)
        for it, ref in zip(req.multimodal_inputs.mm_items, ref_feats):
            self.assertTrue(torch.equal(torch.as_tensor(it.feature), ref))


# ---- front: an UNPRICED image request is D's admission's call, not P-only by rule ------

def test_front_unpriced_image_request_is_not_p_only_by_rule():
    """rc16 boot 121629: every D probe failed (KeyError '<rid>~probe'), 65 image requests went P-only by rule
    (``route short -> long (P)``) and flipped P; P then read 171776 of 172072 (pdflip-2-9). An unpriced request
    is routed to D's admission (S56r A1), which measures the image exactly (W123 -> P-only requeue)."""
    from flliper.srt.pdflip import front as F

    src = inspect.getsource(F.Front)
    i_unpriced = src.index("route %s stands UNPRICED")
    i_rule = src.index("request is prefilled on P by rule; its embeddings are")
    assert i_unpriced < i_rule, "the unpriced branch must be decided before the P-only rule"
    assert "_w102_unpriced_d = True" in src
    assert ("p_only=(_verdict == VERDICT_STAGE and not _mm_cached and not _w102_unpriced_d)\n"
            "                    or _probe_p_only,") in src
    # an image MEASURED in the uncached tail stays P-only (R9 unchanged)
    assert "_probe_p_only = bool(_xx is not None and getattr(_xx, \"image_in_tail\", False))" in src
    assert F.vision_unpriced_to_d_on({}) is True
    assert F.vision_unpriced_to_d_on({F.VISION_UNPRICED_TO_D_ENV: "0"}) is False


# ---- S66c EMB-CACHE: an image P encoded once is not fetched / encoded again --------------

def _rows_for(req, width=8, fill=1.0):
    return [torch.full((sum(b - a + 1 for a, b in it.offsets), width), fill + i)
            for i, it in enumerate(req.multimodal_inputs.mm_items)]


def _store_rows(req, width=8):
    """A previous turn's stage: rows attached to items with the same content hashes, then kept."""
    prev = _req(rid="prev", featureless=True, ends=[it.offsets[0][1] for it in req.multimodal_inputs.mm_items])
    for it, rows in zip(prev.multimodal_inputs.mm_items, _rows_for(prev, width)):
        it.precomputed_embeddings = rows
    return V.emb_cache_store(prev.multimodal_inputs.mm_items)


@pytest.fixture(autouse=False)
def fresh_emb_cache(monkeypatch):
    monkeypatch.setattr(V, "_EMB_CACHE", None)
    monkeypatch.setenv(V.VISION_EMB_CACHE_ENV, "64")
    yield


def test_rc17_pdflip_6_245_told_short_by_31_positions_uses_cached_rows_no_fetch_no_tower(fresh_emb_cache, caplog):
    """rc17 boot 171636, P PP0, pdflip-6-245 (also 8-254): P-KNOWN-IMAGE hit, TOLD told=96768, W102k FETCH
    covered=0 image_end=96799 -> pixels 2963 ms + tower 0.6 s for 210 new tokens. The two images' rows had been
    encoded by this rank one turn earlier."""
    caplog.set_level(logging.INFO)
    rid = "pdflip-6-245"
    r = _req(rid=rid, featureless=True, ends=(95900, 96798))  # image_end 96799
    assert _store_rows(r) == 2
    s = _pass_sched(r, told={rid: 96768})
    s.model_config = SimpleNamespace(hf_config=SimpleNamespace(
        vision_config=SimpleNamespace(out_hidden_size=8, deepstack_visual_indexes=[])))
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower ran for cached rows")):
        parked = V.vision_rank_pass(s)
    assert s.ipc_channels.send_to_tokenizer.sent == [], "no pixel fetch"
    assert [x for _, x in parked] == [r], "held this pass only by the H125f gate, like every follower"
    assert [v.rid for v in VV.take_verdicts(s)] == [rid], "PP0's verdict goes out at once"
    for it, want in zip(r.multimodal_inputs.mm_items, _rows_for(r)):
        assert torch.equal(it.precomputed_embeddings, want) and it.feature is None
    assert V.pending_items(r) == []
    assert any(m.startswith(f"W102 PdFlipVisionStage EMB-CACHE HIT rid={rid} items=2") for m in caplog.messages)


def test_emb_cache_is_all_or_nothing_and_checks_count_and_width(fresh_emb_cache):
    r = _req(rid="r1", featureless=True, ends=(5000, 20555))
    _store_rows(_req(rid="r0", featureless=True, ends=(5000,)))  # only the first image known
    s = _sched(told={"r1": 0})
    assert V.emb_cache_attach(s, r) is False and all(
        it.precomputed_embeddings is None for it in r.multimodal_inputs.mm_items), "partial attach never"
    # a row count that does not match the placeholders is never attached
    r2 = _req(rid="r2", featureless=True, ends=(5000,))
    bad = _req(rid="bad", featureless=True, ends=(5001,))
    bad.multimodal_inputs.mm_items[0].precomputed_embeddings = _rows_for(bad)[0]
    V._EMB_CACHE = None  # fresh cache (the fixture restores it)
    V.emb_cache_store(bad.multimodal_inputs.mm_items)  # same hash 1000, one row more
    assert V.emb_cache_attach(s, r2) is False
    # width against the tower
    r3 = _req(rid="r3", featureless=True, ends=(5000,))
    V._EMB_CACHE = None
    V.emb_cache_store(_set_rows(_req(rid="ok", featureless=True, ends=(5000,))))
    assert V.emb_cache_attach(_sched(), _req(rid="r3ok", featureless=True, ends=(5000,))), "same rows, no width set"
    s.model_config = SimpleNamespace(hf_config=SimpleNamespace(
        vision_config=SimpleNamespace(out_hidden_size=16, deepstack_visual_indexes=[])))
    assert V.emb_cache_attach(s, r3) is False


def _set_rows(req, width=8):
    for it, rows in zip(req.multimodal_inputs.mm_items, _rows_for(req, width)):
        it.precomputed_embeddings = rows
    return req.multimodal_inputs.mm_items


def test_emb_cache_rows_are_copies_and_off_switch(fresh_emb_cache, monkeypatch):
    src = _req(rid="s", featureless=True, ends=(5000,))
    items = _set_rows(src)
    V.emb_cache_store(items)
    items[0].precomputed_embeddings.add_(100.0)  # the request mutates its own rows
    r = _req(rid="r", featureless=True, ends=(5000,))
    assert V.emb_cache_attach(_sched(), r)
    assert torch.equal(r.multimodal_inputs.mm_items[0].precomputed_embeddings, _rows_for(r)[0])
    r.multimodal_inputs.mm_items[0].precomputed_embeddings.mul_(-1.0)  # a request mutates the rows it got
    r5 = _req(rid="r5", featureless=True, ends=(5000,))
    assert V.emb_cache_attach(_sched(), r5)
    assert torch.equal(r5.multimodal_inputs.mm_items[0].precomputed_embeddings, _rows_for(r5)[0])
    monkeypatch.setenv(V.VISION_EMB_CACHE_ENV, "0")
    r4 = _req(rid="r4", featureless=True, ends=(5000,))
    assert V.emb_cache_attach(_sched(), r4) is False


def test_emb_cache_miss_still_fetches_and_the_stage_keeps_the_rows(fresh_emb_cache):
    r = _req(rid="m", featureless=True, ends=(72214,))
    s = _sched(told={"m": 0}, wq=[r])
    assert V.skip_cached(s, [r]) == ([], [r]) and len(s.ipc_channels.send_to_tokenizer.sent) == 1
    src = inspect.getsource(V)
    assert src.count("emb_cache_store(items)") == 1 and src.count("emb_cache_store(st.items)") == 1



# ---- S66c-r B1: the grid is part of an entry's identity ----------------------------------

def test_review_s66c_transposed_blank_page_is_a_miss_not_the_other_grids_rows(monkeypatch, caplog):
    """Reviewer test /tmp/rv66b/test_rv66c_grid_collision_1019.py, expectation turned: a blank A4 page portrait
    (grid 1x84x60) and landscape (1x60x84) hash alike (Qwen2VLImageProcessor: 1634794136353122902) with 1260 rows
    each; the tower's pos_embed is grid-interpolated -> other rows. Landscape must MISS (then FETCH / tower)."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(V, "_EMB_CACHE", None)
    H = 1634794136353122902

    def item(grid):
        return SimpleNamespace(offsets=[(10, 10 + 1260 - 1)], feature=torch.zeros(2), hash=H,
                               precomputed_embeddings=None, image_grid_thw=torch.tensor([grid]),
                               is_image=lambda: True)

    sched = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(
        vision_config=SimpleNamespace(out_hidden_size=8, deepstack_visual_indexes=[]))))
    portrait = item([1, 84, 60])
    portrait.precomputed_embeddings = torch.arange(1260 * 8, dtype=torch.float32).reshape(1260, 8)
    assert V.emb_cache_store([portrait]) == 1
    landscape = item([1, 60, 84])
    req = SimpleNamespace(rid="rv-quer", multimodal_inputs=SimpleNamespace(mm_items=[landscape]))
    assert V.emb_cache_attach(sched, req) is False
    assert landscape.precomputed_embeddings is None and landscape.feature is not None
    assert any("EMB-CACHE GRID-MISS rid=rv-quer" in m for m in caplog.messages), "the collision is named"
    # the same grid still hits; both grids live side by side once both were staged
    again = item([1, 84, 60])
    assert V.emb_cache_attach(sched, SimpleNamespace(rid="rv-hoch", multimodal_inputs=SimpleNamespace(
        mm_items=[again])))
    landscape.precomputed_embeddings = torch.ones(1260, 8)
    assert V.emb_cache_store([landscape]) == 1 and len(V._EMB_CACHE) == 2


def test_an_item_without_a_grid_is_never_stored_nor_served(monkeypatch):
    monkeypatch.setattr(V, "_EMB_CACHE", None)
    it = SimpleNamespace(offsets=[(0, 3)], feature=None, hash=7, precomputed_embeddings=torch.ones(4, 8),
                         is_image=lambda: True)
    assert V.emb_cache_store([it]) == 0


def test_h3_a_throwing_attach_falls_back_untouched(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(V, "_EMB_CACHE", None)
    src = _req(rid="s", featureless=True, ends=(5000,))
    V.emb_cache_store(_set_rows(src))
    r = _req(rid="h3", featureless=True, ends=(5000,))
    s = _sched(told={"h3": 0}, wq=[r])
    def half_then_raise(items, rows, expected_width):
        items[0].precomputed_embeddings, items[0].feature = rows[0], None  # half done, then a throw
        raise ValueError("boom")

    with mock.patch.object(V, "attach_precomputed_embeddings", side_effect=half_then_raise):
        assert V.emb_cache_attach(s, r) is False
        assert V.skip_cached(s, [r]) == ([], [r]) and len(s.ipc_channels.send_to_tokenizer.sent) == 1, \
            "named fall back to FETCH"
    assert r.multimodal_inputs.mm_items[0].precomputed_embeddings is None
    assert any("EMB-CACHE REFUSED rid=h3" in m for m in caplog.messages)
    # review S66c-r B2: a PIXEL item -- after the throw its pixels are back and it has no rows (it stages)
    px = _req(rid="h3px", ends=(5000,))
    px.multimodal_inputs.mm_items[0].hash = 1000
    pixels = px.multimodal_inputs.mm_items[0].feature
    s2 = _sched(told={"h3px": 0}, wq=[px])
    with mock.patch.object(V, "attach_precomputed_embeddings", side_effect=half_then_raise):
        assert V.emb_cache_attach(s2, px) is False
    it = px.multimodal_inputs.mm_items[0]
    assert it.feature is pixels and it.precomputed_embeddings is None
    assert V.unstaged_items(px) == [it], "the tower stages it after the fall back"


def test_h1_rc17_pdflip_6_245_behaviour_second_turn_sends_no_fetch(monkeypatch):
    """H1 (review S66c): a behavioural repro without the cache API in the fixture -- turn 1 is a staged pass (the
    fake stage attaches rows like run_rank_stage and keeps them where the tree keeps them), turn 2 is the rc17
    order: told 96768 < image_end 96799. Base code (no EMB-CACHE): turn 2 sends a pixel FETCH."""
    monkeypatch.setenv("FLLIPER_PDFLIP_VISION_EMB_CACHE_MB", "64")
    monkeypatch.setattr(V, "_EMB_CACHE", None, raising=False)

    def fake_stage(scheduler, reqs, **kw):
        from flliper.srt.planner.vision_stage_load import attach_precomputed_embeddings

        items = [it for r in reqs for it in V.unstaged_items(r)]
        attach_precomputed_embeddings(items, [torch.full((sum(b - a + 1 for a, b in it.offsets), 8), 3.0)
                                              for it in items], expected_width=8)
        getattr(V, "emb_cache_store", lambda _i: 0)(items)  # what run_rank_stage does on the tree under test
        return V.StageOutcome()

    t1 = _req(rid="pdflip-5-240", ends=(95900, 96798))  # turn 1: pixels, same images (hashes set below)
    for i, it in enumerate(t1.multimodal_inputs.mm_items):
        it.hash = 1000 + i
    s1 = _pass_sched(t1, told={"pdflip-5-240": 0})
    s1.server_args = SimpleNamespace(model_path="/nonexistent")
    s1.model_config = SimpleNamespace(hf_config=SimpleNamespace(
        vision_config=SimpleNamespace(out_hidden_size=8, deepstack_visual_indexes=[])))
    with mock.patch.object(V, "run_rank_stage", side_effect=fake_stage), mock.patch.object(V, "_rank_device",
                                                                                         lambda: torch.device("cpu")):
        V.vision_rank_pass(s1)
    assert all(it.precomputed_embeddings is not None for it in t1.multimodal_inputs.mm_items), "turn 1 staged"
    t2 = _req(rid="pdflip-6-245", featureless=True, ends=(95900, 96798))  # known images, hashes 1000/1001
    s2 = _pass_sched(t2, told={"pdflip-6-245": 96768})
    with mock.patch.object(V, "run_rank_stage", side_effect=AssertionError("tower ran")):
        V.vision_rank_pass(s2)
    assert s2.ipc_channels.send_to_tokenizer.sent == [], "turn 2 fetched pixels for rows encoded in turn 1"
