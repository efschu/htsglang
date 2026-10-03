"""110 switch defaults (27B line): switches proven at the metal are default ON in code.

Hermetic: no GPU, no server. Each test pins the code default with the variable
unset (the 27B profile line it replaces is redundant) and the explicit off.
"""

import os
from unittest import mock

from sglang.srt.managers import weg2_store_told as told
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import weight_exchange_bounce as bx


def _unset(*names):
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in names}, clear=True)


def test_seq_sync_batch_defaults_are_the_27b_proven_256_128():
    """27B xsn123: 256 MiB / 128 units (was 64 / 32, set only in the 27B profile)."""
    with _unset(bx.SEQ_SYNC_BATCH_MIB_ENV, bx.SEQ_SYNC_BATCH_UNITS_ENV):
        assert bx.seq_sync_batch() == (256 << 20, 128)


def test_seq_sync_batch_explicit_values_and_garbage():
    with mock.patch.dict(os.environ, {bx.SEQ_SYNC_BATCH_MIB_ENV: "64", bx.SEQ_SYNC_BATCH_UNITS_ENV: "32"}):
        assert bx.seq_sync_batch() == (64 << 20, 32)
    with mock.patch.dict(os.environ, {bx.SEQ_SYNC_BATCH_MIB_ENV: "x", bx.SEQ_SYNC_BATCH_UNITS_ENV: "y"}):
        assert bx.seq_sync_batch() == (256 << 20, 128)


def test_follower_early_read_empty_value_is_the_default_on():
    assert told.follower_early_read_on({}) is True
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: ""}) is True
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: "0"}) is False
    assert told.follower_early_read_on({told.ENV_FOLLOWER_EARLY_READ: "off"}) is False


def test_leg1_early_empty_value_is_the_default_on():
    assert front_mod.leg1_early_on({}) is True
    assert front_mod.leg1_early_on({front_mod.LEG1_EARLY_ENV: ""}) is True
    assert front_mod.leg1_early_on({front_mod.LEG1_EARLY_ENV: "0"}) is False
