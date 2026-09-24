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

from sglang.srt.weg2 import vision_d_guard as g


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
    assert g.d_guard_armed(_mc(mm, lmo), env={"SGLANG_WEG2_GROUP": group}) is armed


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
    from sglang.srt.managers import tp_head_congruence as thc
    from sglang.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=3),
                              prefix_indices=None)
    h._weg2_vision_d_covered = lambda r, hi: Scheduler._weg2_vision_d_covered(h, r, hi)
    req = _img_req()
    req.prefix_indices, req.host_hit_length = [0] * 5000, 0   # a local number, never used
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: 2100)
    assert Scheduler._weg2_vision_d_verdict(h, req, object()) == g.ADMIT   # 1033 < 2100
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: 10)
    assert Scheduler._weg2_vision_d_verdict(h, req, object()) == g.REFUSE  # image in the extent
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: None)
    assert Scheduler._weg2_vision_d_verdict(h, req, object()) == g.DEFER   # TP3, no group opinion
    h.ps.tp_size = 1                                          # single rank: its own match
    req.prefix_indices, req.host_hit_length = [0] * 1000, 34
    assert Scheduler._weg2_vision_d_verdict(h, req, object()) == g.ADMIT   # 1034 > 1033
    req.host_hit_length = 33
    assert Scheduler._weg2_vision_d_verdict(h, req, object()) == g.REFUSE
    text = types.SimpleNamespace(rid="t", multimodal_inputs=None)
    assert Scheduler._weg2_vision_d_verdict(h, text, None) == g.ADMIT


def test_a_refused_request_leaves_the_queue_and_is_answered_by_name():
    from sglang.srt.managers.scheduler import Scheduler

    sent = []
    req, other = _img_req("r1"), _img_req("r2")
    h = types.SimpleNamespace(
        waiting_queue=[req, other], tree_cache=None,
        enable_hicache_storage=False, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, r: sent.append(out))),
        _weg2_vision_d_covered=lambda r, hi: 10,
    )
    Scheduler._weg2_answer_vision_d_refusals(h, [req], None)
    assert h.waiting_queue == [other]
    assert sent[0].rid == "r1"
    assert sent[0].finished_reason["message"].startswith(g.W_NOT_IN_PREFIX)
    assert int(sent[0].finished_reason["status_code"]) == 503


def test_the_guard_sits_after_w31_and_before_the_adder_and_is_answered_after_the_loop():
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
    w31 = src.index("self._weg2_x_refuses(req, _head_inputs)")
    guard = src.index("self._weg2_vision_d_verdict(req, _head_inputs)")
    adder = src.index("res = adder.add_one_req(")
    assert w31 < guard < adder
    assert src.index("self._weg2_answer_x_refusals(") < src.index(
        "self._weg2_answer_vision_d_refusals(_v_refused, _head_inputs)")
    init = inspect.getsource(Scheduler.init_request_receiver)
    assert 'os.environ.get("SGLANG_WEG2_GROUP", "").strip().upper() == "D"' in init


def test_a_deferral_is_bounded_and_refused_by_name_after_the_bound():
    defers = {}
    for _ in range(g.DEFER_BOUND_PASSES):
        assert g.bounded(g.DEFER, "r", defers) == g.DEFER
    assert g.bounded(g.DEFER, "r", defers) == g.REFUSE
    assert g.bounded(g.ADMIT, "r", defers) == g.ADMIT and "r" not in defers  # cleared
    msg = g.refusal_message(_req((10, 20)), None)
    assert msg.startswith(g.W_NOT_IN_PREFIX) and "no covered prefix" in msg


def test_the_scheduler_counts_deferrals_per_rid(monkeypatch):
    from sglang.srt.managers import tp_head_congruence as thc
    from sglang.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=3))
    h._weg2_vision_d_covered = lambda r, hi: Scheduler._weg2_vision_d_covered(h, r, hi)
    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: None)
    req = _img_req("slow")
    verdicts = [Scheduler._weg2_vision_d_verdict(h, req, object())
                for _ in range(g.DEFER_BOUND_PASSES + 1)]
    assert verdicts[:-1] == [g.DEFER] * g.DEFER_BOUND_PASSES and verdicts[-1] == g.REFUSE
