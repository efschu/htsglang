"""The D side of the transient vision form: the launcher (slice V3b).

Hermetic. Pinned: under transient D gets language_model_only instead of
--no-enable-multimodal (mm_items, pad ids, mrope delta), the default is
unchanged, the vision env stays P-only, main passes the form to both argv_d
calls, and an EXTRA override that would drop language_model_only is refused.
"""

import pytest


def test_the_launcher_gives_d_the_vision_form_and_keeps_the_env_p_only():
    from sglang.srt.weg2 import launcher as lz

    def d_argv(vision):
        return lz.argv_d("py", "/m", [1, 1, 1], 1, 1, lz.RING_FORM_SENTINEL_STORE_CFG, [],
                         vision=vision)

    transient = d_argv(lz.VISION_TRANSIENT)
    assert "--no-enable-multimodal" not in transient
    i = transient.index("--json-model-override-args")
    assert transient[i + 1] == lz.VISION_TRANSIENT_OVERRIDE
    assert "--no-enable-multimodal" in d_argv(lz.VISION_OFF)   # the default is unchanged
    assert "--no-enable-multimodal" in lz.argv_d(
        "py", "/m", [1, 1, 1], 1, 1, lz.RING_FORM_SENTINEL_STORE_CFG, [])
    env = lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group="D",
                       vision=lz.VISION_TRANSIENT)
    assert lz.VISION_STAGE_ENV not in env  # D arms no stage


def test_main_passes_the_form_to_both_d_argv_calls():
    import inspect

    from sglang.srt.weg2 import launcher as lz

    src = inspect.getsource(lz)
    calls = [line for line in src.splitlines() if "spec_d = GroupSpec(\"D\"" in line]
    assert len(calls) == 2
    assert all("vision=ns.weg2_vision" in line for line in calls)


def test_an_extra_override_without_language_model_only_is_refused_not_lost():
    from sglang.srt.weg2 import launcher as lz

    def d_argv(extra):
        return lz.argv_d("py", "/m", [1, 1, 1], 1, 1, lz.RING_FORM_SENTINEL_STORE_CFG,
                         extra, vision=lz.VISION_TRANSIENT)

    with pytest.raises(lz.Weg2LaunchRefused, match="W111 Weg2VisionArmRefused"):
        d_argv(["--json-model-override-args", '{"rope_theta": 1}'])
    argv = d_argv(["--json-model-override-args", '{"language_model_only": true, "rope_theta": 1}'])
    last = len(argv) - 1 - argv[::-1].index("--json-model-override-args")
    assert '"language_model_only": true' in argv[last + 1]  # the one argparse keeps
    # outside the transient form EXTRA stays the operator's business
    lz.argv_d("py", "/m", [1, 1, 1], 1, 1, lz.RING_FORM_SENTINEL_STORE_CFG,
              ["--json-model-override-args", '{"rope_theta": 1}'])
