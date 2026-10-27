"""Front side of vision on D (user order 27.10.: "vision natürlich in D und in P, für die flip layouts,
vision nur auf P in dual layout").

The front stops forcing an image request to P (route override + ``Pending.p_only``) exactly when the
D group arms its stage (``pdflip.vision_d_runner.d_stage_wanted``). The group reads the launcher's published
``FLLIPER_PDFLIP_VISION_D`` (``launcher.build_env`` for group D), the front reads ``front.vision_on_d``; both come from the
ONE decision ``launcher.vision_on_d_of`` (review 27.10. B1: the group used to look for FLLIPER_PDFLIP_VISION, which the
launcher pops for D, so a front that said "on D" faced a group that never armed). A front that says "on D" while the group is not
armed routes an image to a D that refuses it; the reverse leaves the 5-6 s flip.
"""

import types

import pytest

from flliper.srt.pdflip import front as wf
from flliper.srt.pdflip import vision_d_runner as vdr


@pytest.mark.parametrize("vision,dual,extra,want", [
    ("transient", False, {}, True),
    ("transient", True, {}, False),                                   # dual: vision on P only
    ("off", False, {}, False),
    ("resident", False, {}, False),
    ("transient", False, {"FLLIPER_PDFLIP_VISION_D": "0"}, False),       # the kill switch
    ("transient", False, {"FLLIPER_PDFLIP_VISION_D": "1"}, True),
])
def test_front_launcher_and_group_agree_on_when_d_encodes(monkeypatch, vision, dual, extra, want):
    """One decision, three readers, driven through the REAL launcher: ``build_env`` publishes the D flag for group D only,
    ``d_stage_wanted`` arms from it, the front's ``vision_on_d`` answers the same."""
    from flliper.srt.pdflip import launcher as lz

    monkeypatch.delenv("FLLIPER_PDFLIP_VISION_D", raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    ns = types.SimpleNamespace(pdflip_vision=vision, dual_layout=dual, dual_share=False)
    decision = lz.vision_on_d_of(ns)
    assert decision is want and wf.vision_on_d(vision, dual, extra) is want
    denv = lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group="D", vision=vision, vision_on_d=decision)
    penv = lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group="P", vision=vision, vision_on_d=decision)
    assert (denv.get(lz.VISION_D_ENV) == "1") is want and lz.VISION_D_ENV not in penv
    cfg = types.SimpleNamespace(vision_config=object())
    assert vdr.d_stage_wanted(cfg, {"FLLIPER_PDFLIP_GROUP": "D", **denv}) is want
    assert vdr.d_stage_wanted(cfg, {"FLLIPER_PDFLIP_GROUP": "P", **penv}) is False


def test_dual_share_alone_is_the_dual_layout_for_the_launchers_decision(monkeypatch):
    """Review M6: ``--dual-share`` implies the dual layout (resolve_dual_layout), and the decision must not wait for that
    resolution to have run -- either flag alone keeps vision on P."""
    from flliper.srt.pdflip import launcher as lz

    monkeypatch.delenv("FLLIPER_PDFLIP_VISION_D", raising=False)
    for flag in ("dual_layout", "dual_share"):
        ns = types.SimpleNamespace(pdflip_vision="transient", dual_layout=False, dual_share=False)
        setattr(ns, flag, True)
        assert lz.vision_on_d_of(ns) is False


def test_a_flip_layout_image_request_is_not_forced_to_p(monkeypatch, caplog):
    """The routing itself, through the real ``handle_generate``: by default (flip layout, transient)
    the W102 override "route -> long (P)" does not fire and the request is not queued as p_only -- a
    short image request goes straight to D (the fake front has no HTTP session, so the D leg fails
    with the stub's AttributeError; that failure IS the proof that the D leg was attempted)."""
    import logging

    from test_pdflip_vision_native_generate_h125b import PNG, _front, _gen, _send

    monkeypatch.delenv("FLLIPER_PDFLIP_VISION_D", raising=False)
    f = _front("transient")
    assert f.vision_on_d is True
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _resp, queued = _send(f, _gen(image_data=PNG))
    assert queued is None or queued.p_only is False
    assert not any("-> long (P)" in r.getMessage() for r in caplog.records)
