"""110 switch defaults (NF line): switches proven at the NF metal are default ON in code.

Hermetic: no GPU, no server. Each test pins the code default with the variable
unset (the nf-int4.env profile line it replaces is redundant) and the explicit off.
"""

import os
from unittest import mock

from flliper.srt.environ import envs
from flliper.srt.managers import pdflip_store_told as told
from flliper.srt.pdflip import arrival_seat_rule as asr
from flliper.srt.pdflip import front as front_mod

ASR = "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE"


def _unset(*names):
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in names}, clear=True)


def test_arrival_seat_rule_is_default_on_in_code():
    """NF metal since 29.09. ~19:40Z (DP-WAIT p90 3-6.5 s); no profile line needed."""
    with _unset(ASR):
        assert envs.FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE.get() is True
        assert asr.enabled() is True
        assert asr.enabled({}) is True          # the explicit-env readers (d_seats resume margin)
        assert asr.age_plan_enabled({}) is True  # the age plan needs the rule, so it follows


def test_arrival_seat_rule_explicit_off_still_disables():
    with mock.patch.dict(os.environ, {ASR: "0"}):
        assert asr.enabled() is False
    for off in ("0", "false", "no", "off"):
        assert asr.enabled({ASR: off}) is False
    assert asr.age_plan_enabled({ASR: "0"}) is False
    assert asr.enabled({ASR: "1"}) is True


def test_follower_early_read_empty_value_is_the_default_on():
    assert told.follower_early_read_on({}) is True
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: ""}) is True
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: "0"}) is False
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: "off"}) is False


def test_leg1_early_empty_value_is_the_default_on():
    assert front_mod.leg1_early_on({}) is True
    assert front_mod.leg1_early_on({front_mod.LEG1_EARLY_ENV: ""}) is True
    assert front_mod.leg1_early_on({front_mod.LEG1_EARLY_ENV: "0"}) is False
