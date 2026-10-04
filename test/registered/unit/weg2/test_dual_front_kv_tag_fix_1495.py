"""#1495: the front env gets the boot tag only with the switch, only in the dual layout, never over an operator value."""
from types import SimpleNamespace

from sglang.srt.weg2.launcher import dual_front_kv_tag_env as F


def _ns(dual=True, tag="dkr27bX"):
    return SimpleNamespace(dual_layout=dual, tag=tag)


def test_default_off_is_empty():
    assert F(_ns(), {}) == {}
    assert F(_ns(), {"SGLANG_WEG2_DUAL_FRONT_KV_TAG_FIX": "0"}) == {}


def test_on_exports_the_tag_in_dual_only():
    on = {"SGLANG_WEG2_DUAL_FRONT_KV_TAG_FIX": "1"}
    assert F(_ns(), on) == {"SGLANG_WEG2_DUAL_KV_TAG": "dkr27bX"}
    assert F(_ns(dual=False), on) == {}
    assert F(_ns(tag=""), on) == {}


def test_operator_value_is_never_overwritten():
    assert F(_ns(), {"SGLANG_WEG2_DUAL_FRONT_KV_TAG_FIX": "1", "SGLANG_WEG2_DUAL_KV_TAG": "op"}) == {}


def test_wired_after_the_p_sleep_front_env_and_registered():
    import inspect
    from sglang.srt.weg2 import launcher as L
    src = inspect.getsource(L)
    assert src.index("fenv.update(dual_p_sleep_front_env(") < src.index("fenv.update(dual_front_kv_tag_env(ns))")
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_DUAL_FRONT_KV_TAG_FIX.get() is False
