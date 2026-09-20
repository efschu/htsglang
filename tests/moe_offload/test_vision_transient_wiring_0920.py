# SPDX-License-Identifier: Apache-2.0
"""Task #58 slices 4, 5 and the two riders -- the wiring around the stage.

HERMETIC: no CUDA, no GPU, no network, no server.  Four things are pinned
here, and each is a place where a silent default would be the failure:

* **the launcher's model argv per vision form** (slice 4).  `off` and
  `transient` differ by exactly which switch turns the tower off, and only one
  of the two leaves the tokenizer's image path alive.
* **the front's verdict table** (slice 5).  Enumerated at a desk rather than
  inferred from a running front -- including the VIDEO hole, which existed
  because nothing counted videos.
* **the loader veto** (`weight_name_needed`), which stops a text-only rank
  reading 0.858 GiB of tower it will drop anyway.
* **the `[vram-idle]` emitter**, which is the placement input the stage
  planner has been fed a stand-in for.
"""

import logging
import types

import pytest

from sglang.srt.models import qwen3_vl as qv
from sglang.srt.model_executor import vram_family_census as vfc
from sglang.srt.weg2 import front as fr
from sglang.srt.weg2 import launcher as lz


# --------------------------------------------------- slice 4: the launcher --


def test_the_three_vision_forms_exist_and_transient_is_one_of_them():
    assert lz.VISION_CHOICES == ("off", "resident", "transient")
    assert lz.VISION_TRANSIENT == "transient"


def test_off_and_transient_turn_the_tower_off_by_DIFFERENT_switches():
    """The distinction this whole slice turns on.

    `--no-enable-multimodal` also switches off the TOKENIZER's image path
    (`model_config.py:573 is_multimodal`), so no mm_items are ever built and
    there is nothing for a stage to encode.  `language_model_only` is honoured
    by the same predicate (`qwen3_vl.py:1230`) but leaves the tokenizer alone.
    """
    assert lz.vision_model_flags("off") == ["--no-enable-multimodal"]
    assert lz.vision_model_flags("transient") == [
        "--json-model-override-args", '{"language_model_only": true}'
    ]
    assert "--no-enable-multimodal" not in lz.vision_model_flags("transient")


def test_resident_passes_nothing_so_the_default_path_is_untouched():
    assert lz.vision_model_flags("resident") == []


def test_an_unknown_vision_form_refuses_instead_of_defaulting():
    with pytest.raises(ValueError) as e:
        lz.vision_model_flags("maybe")
    assert "plausible wrong text" in str(e.value)


def test_the_override_payload_is_the_predicate_the_model_reads():
    """The constant and the model's test must not drift: what the launcher
    writes has to be the key `vision_tower_forced_off` reads."""
    import json

    payload = json.loads(lz.VISION_TRANSIENT_OVERRIDE)
    assert payload == {"language_model_only": True}
    cfg = types.SimpleNamespace(**payload)
    assert qv.vision_tower_forced_off(cfg) is True


# ------------------------------------------------------- slice 5: the front --


def test_the_video_hole_is_closed():
    """Before #58 `_image_parts` counted three IMAGE type names and nothing
    else, so a video part scored 0, routed as text, and reached
    `_require_visual` on a tower-less rank -- past the refusal that was
    supposed to be structural."""
    body = {"messages": [{"content": [
        {"type": "video_url", "video_url": {"url": "x"}},
        {"type": "text", "text": "what happens here"},
    ]}]}
    assert fr._image_parts(body) == 0
    assert fr._video_parts(body) == 1


@pytest.mark.parametrize("t", ["video_url", "video", "input_video"])
def test_every_video_type_name_is_counted(t):
    assert fr._video_parts({"messages": [{"content": [{"type": t}]}]}) == 1


def test_a_malformed_body_is_not_a_video():
    assert fr._video_parts(None) == 0
    assert fr._video_parts({"messages": "nope"}) == 0
    assert fr._video_parts({"messages": [{"content": [42, None]}]}) == 0


def test_a_prompt_that_merely_says_video_is_text():
    """The #995 prose trap, one layer up -- structural counting, not a string
    search."""
    body = {"messages": [{"content": [{"type": "text", "text": "video video video"}]}]}
    assert fr._video_parts(body) == 0
    assert fr.vision_verdict(0, fr._video_parts(body), "off")[0] == fr.VERDICT_ROUTE


@pytest.mark.parametrize("mode", ["off", "resident", "transient"])
def test_text_routes_in_every_mode(mode):
    assert fr.vision_verdict(0, 0, mode) == (fr.VERDICT_ROUTE, "")


@pytest.mark.parametrize("mode", ["off", "resident", "transient"])
def test_video_refuses_in_every_mode_including_transient(mode):
    """Refused by name, not discovered on the card: a clip's encoder cost is
    quadratic in the patch rows and its frame count is unmeasured here."""
    verdict, why = fr.vision_verdict(0, 1, mode)
    assert verdict == fr.VERDICT_REFUSE_VIDEO
    assert "still images only" in why


def test_video_beats_image_when_a_request_carries_both():
    assert fr.vision_verdict(3, 1, "transient")[0] == fr.VERDICT_REFUSE_VIDEO


def test_an_image_refuses_under_off_and_the_message_names_both_ways_out():
    verdict, why = fr.vision_verdict(2, 0, "off")
    assert verdict == fr.VERDICT_REFUSE_IMAGE
    assert "TEXT-ONLY" in why
    assert "--weg2-vision transient" in why
    assert "--weg2-vision resident" in why


def test_an_image_routes_under_resident():
    assert fr.vision_verdict(1, 0, "resident") == (fr.VERDICT_ROUTE, "")


def test_an_image_stages_under_transient():
    verdict, why = fr.vision_verdict(1, 0, "transient")
    assert verdict == fr.VERDICT_STAGE
    assert "taking it down again" in why


def test_an_unknown_mode_refuses_and_never_falls_through_to_route():
    """`route` is the one verdict that can return wrong text, so it must
    never be what an unrecognised mode gets by accident."""
    verdict, why = fr.vision_verdict(1, 0, "")
    assert verdict == fr.VERDICT_REFUSE_MODE
    assert verdict != fr.VERDICT_ROUTE
    assert "unknown vision mode" in why
    assert fr.vision_verdict(1, 0, "Transient")[0] == fr.VERDICT_REFUSE_MODE


def test_the_front_argparse_puts_no_choices_on_the_mode():
    """argparse `choices=` would kill the front PROCESS on a bad value.  A
    front that will not start is a worse failure than one that refuses image
    requests, so the check lives in `vision_verdict` instead."""
    import inspect

    src = inspect.getsource(fr)
    assert '"--vision", default=VISION_MODE_OFF' in src
    assert '"--vision", choices' not in src


def test_every_non_route_verdict_has_a_W_CODE_in_the_handler():
    """The danger: a verdict added later with no entry in the handler's dict
    raises KeyError INSIDE the request handler -- a 500 where a named 501
    belongs, i.e. exactly the silent shape this whole path exists to avoid.

    Checked against the source because the dict is a local inside an async
    handler; a structural read is the honest instrument here and it is named
    as one.
    """
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    refusals = {
        fr.VERDICT_REFUSE_IMAGE, fr.VERDICT_REFUSE_VIDEO, fr.VERDICT_REFUSE_MODE,
    }
    # the verdict set is exactly the refusals plus route plus stage, and the
    # last two are NOT refusals -- so the dict must cover the refusals and
    # nothing else can reach it
    assert set(fr.__dict__[n] for n in dir(fr) if n.startswith("VERDICT_")) == (
        refusals | {fr.VERDICT_ROUTE, fr.VERDICT_STAGE}
    )
    for v in refusals:
        assert f"VERDICT_{v.upper().replace('-', '_')}:" in src
    assert "W101 Weg2VisionRefused" in src
    assert "W103 Weg2VideoRefused" in src
    assert "W104 Weg2VisionModeUnknown" in src


def test_the_transient_mode_now_ROUTES_instead_of_refusing():
    """SUPERSEDES `test_the_transient_mode_refuses_LOUDLY_until_the_group_side_
    is_wired`, which pinned the round-2 state: `transient` answered 501 with
    "the group-side runtime is not wired yet".

    It is wired now (`weg2/vision_stage_service.py`, called from
    `base_processor.process_and_combine_mm_data`), so the verdict STAGE falls
    through to routing and W102 is an INFO line on the way to P, not a
    refusal. The old assertion is kept here as its negation so the two states
    can never both be true, and so a reader of the history sees the change
    rather than a test that quietly vanished.
    """
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    assert "not wired yet" not in src
    assert "W102 Weg2VisionStage" in src
    assert "routing to P" in src
    # STAGE is no longer in the REFUSAL dict -- it does still appear later,
    # in the force-to-P branch, which is the opposite of a refusal
    dict_block = src[src.index("_code = {"): src.index("}[_verdict]")]
    assert "VERDICT_STAGE" not in dict_block
    assert 'route = "long"' in src


def test_the_front_carries_the_mode_it_was_told():
    f = fr.Front.__init__
    import inspect

    assert inspect.signature(f).parameters["vision"].default == fr.VISION_MODE_OFF


def test_the_launcher_tells_the_front_its_mode():
    """The front cannot infer the mode from the groups it talks to -- their
    tower state is in THEIR argv, not in any response the front sees."""
    import inspect

    src = inspect.getsource(lz.front_argv_for)
    assert '"--vision"' in src
    assert 'weg2_vision' in src


# -------------------------------------------- the loader veto (5.1 GiB/boot) --


class _FakeVL:
    """Only the two attributes `weight_name_needed` touches."""

    weight_name_needed = qv.Qwen3VLForConditionalGeneration.weight_name_needed

    def __init__(self, visual):
        self.visual = visual


def test_a_tower_less_rank_vetoes_the_tower_before_it_is_READ():
    m = _FakeVL(visual=None)
    assert m.weight_name_needed("model.visual.blocks.0.attn.qkv.weight") is False
    assert m.weight_name_needed("visual.merger.linear_fc2.weight") is False
    assert m.weight_name_needed("model.layers.0.self_attn.q_proj.weight") is True
    assert m.weight_name_needed("lm_head.weight") is True


def test_a_rank_WITH_a_tower_vetoes_nothing_at_all():
    """The load path of every boot that builds a tower must be byte-for-byte
    what it was."""
    m = _FakeVL(visual=object())
    for n in ("model.visual.blocks.0.attn.qkv.weight", "model.layers.0.w",
              "lm_head.weight", "visual.merger.linear_fc2.bias"):
        assert m.weight_name_needed(n) is True


def test_the_veto_and_the_load_time_skip_are_ONE_test():
    """`skip_vision_weight` stays where it is -- it is the same test one layer
    later, and it must keep working for a loader that never consults
    `weight_name_needed`.  Drift between the two is the defect this pins."""
    for visual in (None, object()):
        m = _FakeVL(visual=visual)
        for n in ("model.visual.x", "visual.y", "model.layers.0.w"):
            assert m.weight_name_needed(n) is not qv.skip_vision_weight(n, visual)


def test_the_loader_looks_the_method_up_by_name():
    """The wiring is `getattr(model, "weight_name_needed", None)` at
    loader.py:763 -- there is no registration, so the method existing IS the
    wiring."""
    from sglang.srt.model_loader import loader as ld
    import inspect

    src = inspect.getsource(ld.DefaultModelLoader._get_all_weights)
    assert 'getattr(model, "weight_name_needed", None)' in src
    assert hasattr(qv.Qwen3VLForConditionalGeneration, "weight_name_needed")
    # and Qwen3_5, the serving arch, inherits it
    from sglang.srt.models import qwen3_5 as q5

    assert q5.Qwen3_5ForConditionalGeneration.weight_name_needed is (
        qv.Qwen3VLForConditionalGeneration.weight_name_needed
    )


def test_what_the_veto_saves_is_the_measured_tower():
    """333 tensors / 921_460_192 bytes per rank, six ranks per Weg-2 boot."""
    assert 6 * 921_460_192 / 2**30 == pytest.approx(5.149, abs=0.001)


# ------------------------------------------------- the [vram-idle] emitter --


class _FakeCuda:
    def __init__(self, free, total, alloc=0.0, reserved=0.0, raise_on_info=False):
        self._f, self._t, self._a, self._r = free, total, alloc, reserved
        self._raise = raise_on_info

    def mem_get_info(self):
        if self._raise:
            raise RuntimeError("no cuda")
        return self._f, self._t

    def memory_allocated(self):
        return self._a

    def memory_reserved(self):
        return self._r


def test_the_idle_emitter_prints_free_and_total_and_returns_the_free_gib(caplog):
    cuda = _FakeCuda(int(2.09 * 2**30), int(19.58 * 2**30),
                     int(13.1 * 2**30), int(16.9 * 2**30))
    with caplog.at_level(logging.INFO, logger=vfc.__name__):
        got = vfc.log_vram_idle(object(), "after pools", cuda=cuda)
    assert got == pytest.approx(2.09, abs=0.001)
    text = caplog.text
    assert "[vram-idle] after pools" in text
    assert "card free 2.090 of 19.580 GiB" in text
    assert "free_idle" in text


def test_an_unreadable_card_is_None_and_NOT_zero(caplog):
    """A card that could not be read is not a full card.  Returning 0.0 would
    place the stage against a lie."""
    with caplog.at_level(logging.INFO, logger=vfc.__name__):
        got = vfc.log_vram_idle(object(), "idle", cuda=_FakeCuda(0, 0, raise_on_info=True))
    assert got is None
    assert "[vram-idle]" not in caplog.text


def test_the_where_label_is_carried_so_two_samples_are_never_confused(caplog):
    cuda = _FakeCuda(int(1 * 2**30), int(20 * 2**30))
    with caplog.at_level(logging.INFO, logger=vfc.__name__):
        vfc.log_vram_idle(object(), "before vision stage", cuda=cuda)
    assert "[vram-idle] before vision stage" in caplog.text


def test_the_census_emits_it_at_the_one_honest_moment():
    """Pools built, nothing forwarding yet -- and NOT at the other census
    points, where a forward may already have drawn its transient."""
    import inspect

    src = inspect.getsource(vfc)
    assert 'log_vram_idle(model, "after pools")' in src
    # exactly one CALL (the `def` line matches the same prefix, hence `("`)
    assert src.count('log_vram_idle(model, "') == 1


def test_the_idle_reading_is_a_different_instrument_from_the_peak_one():
    """Both exist; neither replaces the other.  `[vram-peak]` is for pools
    that coexist with the prefill, `[vram-idle]` for a stage that does not."""
    assert hasattr(vfc, "log_vram_idle")
    assert hasattr(vfc, "maybe_log_vram_peak")
    import inspect

    assert "free_idle" in inspect.getdoc(vfc.log_vram_idle) or "free_idle" in (
        inspect.getsource(vfc.log_vram_idle)
    )


def test_argv_p_hands_the_vision_form_to_the_model_argv():
    """xsn403 (20.09.): argv_p accepted ``vision`` and dropped it, so P was
    launched text-only (``--no-enable-multimodal``) and the transient stage
    refused to arm (W111). The P argv must carry the transient override and
    NOT the multimodal switch-off; ``off`` keeps the old argv; D stays
    text-only regardless."""
    from sglang.srt.weg2 import launcher as lz

    def p_argv(vision):
        return lz.argv_p("py", "/models/Qwen3.8-27B-INT8-gdncov", [1, 1, 1], 4, 512,
                         "store", [], vision=vision)

    transient = p_argv("transient")
    assert "--no-enable-multimodal" not in transient
    i = transient.index("--json-model-override-args")
    assert transient[i + 1] == lz.VISION_TRANSIENT_OVERRIDE
    off = p_argv("off")
    assert "--no-enable-multimodal" in off
    assert "--json-model-override-args" not in off


def test_after_pools_census_reaches_the_idle_reading(monkeypatch):
    """xsn405: ``log_vram_family_census(..., "after pools")`` referenced a
    ``runner`` the function never had; the NameError was swallowed by the
    caller's census guard after the census line had printed, so no P log
    ever carried ``[vram-idle] after pools`` (acceptance (b), design §6)."""
    import torch

    from sglang.srt.model_executor import vram_family_census as vc

    seen = []
    monkeypatch.setattr(vc, "log_vram_idle", lambda runner, where, **kw: seen.append(where))
    monkeypatch.setattr(vc.torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    model = torch.nn.Linear(4, 4)
    vc.log_vram_family_census(model, "pp0tp0", "after pools")
    assert seen == ["after pools"]
    seen.clear()
    vc.log_vram_family_census(model, "pp0tp0", "after load")
    assert seen == []
