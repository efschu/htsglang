# SPDX-License-Identifier: Apache-2.0
"""Default-on audit 1300 (04.10.2026, user order "alle Schalter, die default an sein sollten ... ANSCHALTEN"):
SGLANG_WEG2_VISION_FLIP_URGENT (front.py) was OFF in the code but ON in every metal boot of the 27B line
(profile 27b.env `_form ... 1`; group env of the INT8 boot dkr27browauthoritybar1fs10040532 and of the NVFP4
dual boot ...10040710; NF has it on through its registry). It acts only on a request that ONLY P can serve (an
image under --weg2-vision transient), so a text-only front stays byte-identical.

DANGER DIRECTIONS guarded here: an explicit "0"/"off" must still switch the latch off; a text-only front must
not log the vision fields (H125).
"""
from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = "SGLANG_WEG2_VISION_FLIP_URGENT"


# ------------------------------------------------------------------ vision latch
def test_vision_flip_urgent_switch_default_on():
    assert front_mod.vision_flip_urgent({}) is True
    assert front_mod.vision_flip_urgent({ENV: ""}) is True, "blank = the default (on), like every switch of its family"


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "OFF"])
def test_vision_flip_urgent_explicit_off_still_disables(raw):
    assert front_mod.vision_flip_urgent({ENV: raw}) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
def test_vision_flip_urgent_explicit_on(raw):
    assert front_mod.vision_flip_urgent({ENV: raw}) is True


def test_default_reaches_a_transient_front_and_leaves_a_text_only_front_alone(monkeypatch, caplog):
    import test_weg2_vision_flip_economics_0924 as V

    monkeypatch.delenv(ENV, raising=False)
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f_img = V._front(vision="transient")
    assert f_img.vision_flip_urgent is True
    assert any("VISION-FLIP-URGENT on" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f_txt = V._front(vision="off")
        f_txt._flip_economics_ok(fairness_fired=False)
    assert f_txt.vision_flip_urgent is False, "default-on has no object on a text-only front (H125: byte-identical)"
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("VISION-FLIP-URGENT" in m for m in msgs)
    monkeypatch.setenv(ENV, "0")
    assert V._front(vision="transient").vision_flip_urgent is False
