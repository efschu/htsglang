"""#1495: the front env gets the boot tag (default on since 22.10.), only in the dual layout, never over an operator value."""
from types import SimpleNamespace

from flliper.srt.pdflip.launcher import dual_front_kv_tag_env as F


def _launcher_text() -> str:
    """The launcher source as one text (launcher.py + launcher_*.py) with the split functions expanded back to the original statement
    order: the launcher was split behaviour-neutrally (refactor 28.10.), see ``_launcher_source.expanded_launcher_text``."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_launcher_text()


def _ns(dual=True, tag="dkr27bX"):
    return SimpleNamespace(dual_layout=dual, tag=tag)


def test_default_on_exports_the_tag_and_0_switches_it_off():
    """Default ON since 22.10. (user order 14.10.; 27B dual 21.10.: the blind front resumed P past a refused
    reclaim). =0 keeps the old front env."""
    assert F(_ns(), {}) == {"FLLIPER_PDFLIP_DUAL_KV_TAG": "dkr27bX"}
    assert F(_ns(), {"FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX": "0"}) == {}


def test_on_exports_the_tag_in_dual_only():
    on = {"FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX": "1"}
    assert F(_ns(), on) == {"FLLIPER_PDFLIP_DUAL_KV_TAG": "dkr27bX"}
    assert F(_ns(dual=False), on) == {}
    assert F(_ns(tag=""), on) == {}


def test_operator_value_is_never_overwritten():
    assert F(_ns(), {"FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX": "1", "FLLIPER_PDFLIP_DUAL_KV_TAG": "op"}) == {}


def test_wired_after_the_p_sleep_front_env_and_registered():
    import inspect
    from flliper.srt.pdflip import launcher as L
    src = _launcher_text()
    assert src.index("fenv.update(dual_p_sleep_front_env(") < src.index("fenv.update(dual_front_kv_tag_env(ns))")
    from flliper.srt.environ import envs
    assert envs.FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX.get() is True
