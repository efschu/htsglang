"""The D side of the transient vision form: the admission guard (slice V3a).

Hermetic, CPU. Pinned:
  * the guard arms on group D with multimodal tokenization and no tower only;
  * the verdict: a text request and an image inside the covered prefix are
    admitted, an image reaching into the extent is refused, and a request the
    group has no match for is deferred (never admitted on a local number);
  * the named refusal says which tokens and why;
  * the scheduler prices it with the group's match, defers without one,
    answers a refusal on the W31 path, and sits between W31 and the adder.
"""

import types

import pytest

from flliper.srt.pdflip import vision_d_guard as g


def _mc(multimodal=True, lmo=True):
    return types.SimpleNamespace(
        is_multimodal=multimodal,
        hf_config=types.SimpleNamespace(language_model_only=lmo))


@pytest.mark.parametrize("group,mm,lmo,armed", [
    ("D", True, True, True),
    ("d", True, True, True),
    ("P", True, True, False),     # P has the rank stage
    ("", True, True, False),
    ("D", False, True, False),    # --no-enable-multimodal: no mm_items ever
    ("D", True, False, False),    # resident tower: D could encode itself
])
def test_the_guard_arms_on_a_towerless_multimodal_d_only(group, mm, lmo, armed):
    assert g.d_guard_armed(_mc(mm, lmo), env={"FLLIPER_PDFLIP_GROUP": group}) is armed


def _req(*spans):
    items = [types.SimpleNamespace(offsets=[s]) for s in spans]
    return types.SimpleNamespace(rid="r", multimodal_inputs=types.SimpleNamespace(mm_items=items))


def test_text_is_always_admitted():
    text = types.SimpleNamespace(rid="t", multimodal_inputs=None)
    assert g.verdict(text, None) == g.ADMIT
    assert g.verdict(text, 0) == g.ADMIT


def test_an_image_inside_the_covered_prefix_is_admitted_and_one_token_short_is_refused():
    req = _req((10, 1033), (1100, 2123))
    assert g.verdict(req, 2124) == g.ADMIT      # both images read from the store
    assert g.verdict(req, 2123) == g.REFUSE     # the last placeholder is in the extent
    assert g.verdict(req, 500) == g.REFUSE


def test_no_group_match_defers_instead_of_guessing():
    assert g.verdict(_req((10, 20)), None) == g.DEFER


def test_the_refusal_names_the_uncovered_span_and_the_reason():
    msg = g.refusal_message(_req((10, 1033), (1100, 2123)), covered=1500)
    assert msg.startswith(g.W_NOT_IN_PREFIX)
    assert "1100..2123" in msg and "1500 tokens" in msg and "_require_visual" in msg


# ------------------------------------------------------ the scheduler seam --


def _img_req(rid="r", fill=3000, span=(10, 1033)):
    items = [types.SimpleNamespace(offsets=[span])]
    return types.SimpleNamespace(
        rid=rid, full_untruncated_fill_ids=list(range(fill)),
        multimodal_inputs=types.SimpleNamespace(mm_items=items),
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)))


def test_the_d_verdict_is_priced_with_the_groups_match(monkeypatch):
    from flliper.srt.managers import tp_head_congruence as thc
    from flliper.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=3),
                              prefix_indices=None)
    h._pdflip_vision_d_covered = lambda r, hi: Scheduler._pdflip_vision_d_covered(h, r, hi)
    req = _img_req()
    req.prefix_indices, req.host_hit_length = [0] * 5000, 0   # a local number, never used
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: 2100)
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.ADMIT   # 1033 < 2100
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: 10)
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.REFUSE  # image in the extent
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: None)
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.DEFER   # TP3, no group opinion
    h.ps.tp_size = 1                                          # single rank: its own match
    req.prefix_indices, req.host_hit_length = [0] * 1000, 34
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.ADMIT   # 1034 > 1033
    req.host_hit_length = 33
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.REFUSE
    text = types.SimpleNamespace(rid="t", multimodal_inputs=None)
    assert Scheduler._pdflip_vision_d_verdict(h, text, None) == g.ADMIT


def test_a_refused_request_leaves_the_queue_and_is_answered_by_name():
    from flliper.srt.managers.scheduler import Scheduler

    sent = []
    req, other = _img_req("r1"), _img_req("r2")
    h = types.SimpleNamespace(
        waiting_queue=[req, other], tree_cache=None,
        enable_hicache_storage=False, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, r: sent.append(out))),
        _pdflip_vision_d_covered=lambda r, hi: 10,
    )
    req.output_ids = [5]  # already streamed: the P reroute cannot carry it (see below)
    Scheduler._pdflip_answer_vision_d_refusals(h, [req], None)
    assert h.waiting_queue == [other]
    assert sent[0].rid == "r1"
    assert sent[0].finished_reason["message"].startswith(g.W_NOT_IN_PREFIX)
    assert int(sent[0].finished_reason["status_code"]) == 503


def test_the_guard_sits_after_w31_and_before_the_adder_and_is_answered_after_the_loop():
    import inspect

    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
    w31 = src.index("self._pdflip_x_refuses(req, _head_inputs)")
    guard = src.index("self._pdflip_vision_d_verdict(\n                    req, _head_inputs, batch_empty=")
    adder = src.index("res = adder.add_one_req(")
    assert w31 < guard < adder
    assert src.index("self._pdflip_answer_x_refusals(") < src.index(
        "self._pdflip_answer_vision_d_refusals(_v_refused, _head_inputs)")
    init = inspect.getsource(Scheduler.init_request_receiver)
    assert 'os.environ.get("FLLIPER_PDFLIP_GROUP", "").strip().upper() == "D"' in init


def test_a_deferral_is_bounded_and_refused_by_name_after_the_bound():
    defers = {}
    for _ in range(g.DEFER_BOUND_PASSES):
        assert g.bounded(g.DEFER, "r", defers) == g.DEFER
    assert g.bounded(g.DEFER, "r", defers) == g.REFUSE
    assert g.bounded(g.ADMIT, "r", defers) == g.ADMIT and "r" not in defers  # cleared
    msg = g.refusal_message(_req((10, 20)), None)
    assert msg.startswith(g.W_NOT_IN_PREFIX) and "no covered prefix" in msg


def test_the_scheduler_counts_deferrals_per_rid(monkeypatch):
    from flliper.srt.managers import tp_head_congruence as thc
    from flliper.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=3))
    h._pdflip_vision_d_covered = lambda r, hi: Scheduler._pdflip_vision_d_covered(h, r, hi)
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: None)
    req = _img_req("slow")
    verdicts = [Scheduler._pdflip_vision_d_verdict(h, req, object())
                for _ in range(g.DEFER_BOUND_PASSES + 1)]
    assert verdicts[:-1] == [g.DEFER] * g.DEFER_BOUND_PASSES and verdicts[-1] == g.REFUSE


# ------------------------------------------------------------------------------
# 27B rc12z7b (1961f756ad) D 07:47:28, rid pdflip-11-10: "HiCache prefetch success
# ... matched=69 loaded=9466 ... deliverable=9535", PHASE-PURITY "host_hit=9466
# matched=69 materialized=9535", then "W123 ... covered=9466" for an image
# spanning 6290..9528 of a 9537-token prompt. The first 69 tokens were a device
# node without a recurrent state (the sibling pdflip-10-9 had been loaded and had
# released its host rows); the match counted device 0 + host 9466.
# ------------------------------------------------------------------------------


def _vote_req(anchor=9535, device=0, host=9466, fill=9537, span=(6290, 9528)):
    r = _img_req(rid="pdflip-11-10", fill=fill, span=span)
    r.prefix_indices = [0] * device
    r.host_hit_length = host
    r.num_matched_prefix_tokens = device + host
    r.state_anchor_depth = anchor
    r._compute_max_prefix_len = lambda n: max(n - 1, 0)
    return r


def _voter(req, monkeypatch):
    """A Scheduler double whose head vote runs the shipped method; the match
    itself is the metal's (match_prefix_for_req stamps nothing new here)."""
    from flliper.srt.managers import schedule_policy as sp
    from flliper.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(waiting_queue=[req], tree_cache=types.SimpleNamespace(match_prefix=None))
    monkeypatch.setattr(sp, "match_prefix_for_req", lambda tree, r, include_req=True: None)
    return Scheduler._local_head_prefix_matches(h)


def test_the_head_vote_is_the_materializable_prefix_not_device_plus_host(monkeypatch):
    req = _vote_req()
    canonical, matches = _voter(req, monkeypatch)
    assert canonical == ["pdflip-11-10"]
    assert matches["pdflip-11-10"] == 9535, "device 0 + host 9466 misses the 69-token device head"


def test_the_metal_image_is_admitted_with_the_groups_vote(monkeypatch):
    from flliper.srt.managers import tp_head_congruence as thc
    from flliper.srt.managers.scheduler import Scheduler

    req = _vote_req()
    _canonical, matches = _voter(req, monkeypatch)
    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=3))
    h._pdflip_vision_d_covered = lambda r, hi: Scheduler._pdflip_vision_d_covered(h, r, hi)
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: matches[rid])
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.ADMIT   # 9528 < 9535
    # the old vote (9466) is the metal's W123
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: 9466)
    assert Scheduler._pdflip_vision_d_verdict(h, req, object()) == g.REFUSE


def test_the_anchor_vote_is_capped_and_never_lowers_the_match(monkeypatch):
    req = _vote_req(anchor=9537)                      # capped at fill-1
    assert _voter(req, monkeypatch)[1]["pdflip-11-10"] == 9536
    req = _vote_req(anchor=None)                      # pure-KV tree: unchanged
    assert _voter(req, monkeypatch)[1]["pdflip-11-10"] == 9466
    req = _vote_req(anchor=4096, device=5000, host=0)  # anchor below the match: unchanged
    assert _voter(req, monkeypatch)[1]["pdflip-11-10"] == 5000


def test_head_vote_log_line_survives_an_empty_tensor_prefix():
    """27B rc12z9 D 08:23:24: the #823 HEAD-VOTE ANCHOR line evaluated
    ``prefix_indices or ()`` -- an empty tensor's truth value raises."""
    import torch
    from flliper.srt.managers import scheduler as S

    class _R:
        rid = "pdflip-0-2"
        num_matched_prefix_tokens = 22551
        state_anchor_depth = 38911
        prefix_indices = torch.empty(0, dtype=torch.int64)
        host_hit_length = 22551
        full_untruncated_fill_ids = list(range(38913))

        def _compute_max_prefix_len(self, n):
            return n - 1

    assert S._head_vote_len(_R()) == 38911
    assert S._len_or_zero(torch.empty(0)) == 0
    assert S._len_or_zero(None) == 0
    assert S._len_or_zero([1, 2]) == 2


# ------------------------------------------------------------------------------
# (B) safety net, NF rc12z10 08:40:07Z rid pdflip-2-12: image 6290..9528 of 9537
# tokens, P "END-ANCHOR anchor=0 ... ok=False", D "#928 REFUSING resume ...
# NONE-ON-THIS-PATH", covered=0 -> W123 to the client. A request that has sent
# NO byte yet is answered with the W50 refusal the front re-routes through P
# (X-REQUEUE: the ORIGINAL request, image included, P has the tower); W123 is
# named in it. The front's own bound (second refusal after a P leg -> W35/W53)
# ends it by name. An already streamed request keeps W123 (the RESUME-VIA-P leg
# carries input_ids only, P could not encode the image from them).
# ------------------------------------------------------------------------------


def _metal_refused():
    from flliper.srt.managers.scheduler import Scheduler

    sent = []
    req = _img_req("pdflip-2-12", fill=9537, span=(6290, 9528))
    req.output_ids = []
    h = types.SimpleNamespace(
        waiting_queue=[req], tree_cache=None,
        enable_hicache_storage=False, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, r: sent.append(out))),
        _pdflip_vision_d_covered=lambda r, hi: 0,
    )
    return Scheduler, h, req, sent


def test_an_unstreamed_vision_refusal_is_rerouted_through_p_not_w123():
    from flliper.srt.pdflip import front as F

    Scheduler, h, req, sent = _metal_refused()
    Scheduler._pdflip_answer_vision_d_refusals(h, [req], None)
    msg = sent[0].finished_reason["message"]
    assert F.x_refusal_marker_in(msg), "the front must see its X-REQUEUE marker"
    assert F._d_refusal_extent(msg.encode()) == 9537, "the extent D would have to prefill"
    assert g.W_NOT_IN_PREFIX in msg, "W123 stays named inside the reroute"
    assert int(sent[0].finished_reason["status_code"]) == 503
    assert h.waiting_queue == []


def test_a_streamed_vision_refusal_keeps_w123():
    Scheduler, h, req, sent = _metal_refused()
    req.output_ids = [1, 2, 3]
    Scheduler._pdflip_answer_vision_d_refusals(h, [req], None)
    assert sent[0].finished_reason["message"].startswith(g.W_NOT_IN_PREFIX)


def test_reroute_switch_off_is_the_old_w123(monkeypatch):
    monkeypatch.setenv(g.REROUTE_ENV, "0")
    Scheduler, h, req, sent = _metal_refused()
    Scheduler._pdflip_answer_vision_d_refusals(h, [req], None)
    assert sent[0].finished_reason["message"].startswith(g.W_NOT_IN_PREFIX)
