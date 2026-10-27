"""Front side of vision on D (user order 27.10.: "vision natürlich in D und in P, für die flip layouts,
vision nur auf P in dual layout").

The front stops forcing an image request to P (route override + ``Pending.p_only``) exactly when the
D group arms its stage (``pdflip.vision_d_runner.d_stage_wanted``): both read the same three facts
(transient, not the dual layout, ``FLLIPER_PDFLIP_VISION_D``). A front that says "on D" while the group
is not armed would route an image to a D that refuses it; the reverse leaves the 5-6 s flip.
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
def test_front_and_group_agree_on_when_d_encodes(vision, dual, extra, want):
    assert wf.vision_on_d(vision, dual, extra) is want
    genv = {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION": vision, **extra}
    if dual:
        genv["FLLIPER_PDFLIP_DUAL_LAYOUT"] = "1"
    cfg = types.SimpleNamespace(vision_config=object())
    assert vdr.d_stage_wanted(cfg, genv) is want


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
