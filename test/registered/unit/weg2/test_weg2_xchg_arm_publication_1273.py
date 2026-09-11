# SPDX-License-Identifier: Apache-2.0
"""#1273 S6 step 7 -- THE EXCHANGE ARM MUST REACH THE RANKS.

The blocker this closes, named in the HELD S6I order: `prepare_shadow_env`
returned `{}` for every arm but `shadow` (`launcher.py:3385`), so under
`--weg2-weight-source exchange` NO region was armed and NOTHING was published
-- not the on-card arm, not the published slot, and not the two variables step
6 added after that guard. Three consequences, each a silent wrong answer:

* a rank read `inject_mode()` = `shadow` BY DEFAULT rather than by
  publication, so the mode was right by luck and would have stayed `shadow`
  even with `--weg2-xchg-inject authoritative` on the command line;
* `read_published_terms()` answered `None`, so the injector refused W4;
* no region meant no Gate 0 and no transport at all.

The function is RENAMED `prepare_xchg_env`, because it now arms two arms and a
name that says `shadow` would be the lie a future reader trusts (one reader,
one name).

RED ON 093e80ed1d: the name does not exist and `exchange` publishes nothing.
"""

from __future__ import annotations

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as lz  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402


def _terms():
    return xb.bounce_terms(
        bytes_per_direction=24_000_000, n_layers=8,
        widest_layer_bytes=4_000_000, pairs=xr.N_CARDS, depth=2,
        slot_bytes=128 * xb.MIB,
    )


def _env(arm, **kw):
    """Publish for one arm, with a throwaway region under the test's own root."""
    return lz.prepare_xchg_env(lambda *_a: None, kw.pop("boot", "b1"), arm,
                               **kw)


# ===========================================================================
# (i) UNDER `exchange`, EVERYTHING THE RANKS NEED IS PUBLISHED.
# ===========================================================================


class TheExchangeArmPublishes:
    """Namespace only; the collected tests are the functions below."""


def test_the_exchange_arm_publishes_region_arm_slot_mode_and_terms(tmp_path):
    env = _env(lz.WEIGHT_SOURCE_EXCHANGE, dry=True,
               oncard_mode=tp.ONCARD_MODE_HOST, oncard_slot_mib=128,
               bounce_terms=_terms(),
               inject_mode=wx.INJECT_AUTHORITATIVE)
    assert env, "the exchange arm published nothing -- the step 7 blocker"
    # THE ARM ITSELF, and this is the second half of the blocker: the line
    # used to publish the literal "shadow", so an `exchange` boot would have
    # run as `shadow` in all six ranks with nothing in its log to say so.
    assert env["SGLANG_WEG2_WEIGHT_SOURCE"] == lz.WEIGHT_SOURCE_EXCHANGE
    # the on-card arm and its slot, the one value both sides size from
    assert env[tp.ENV_ONCARD_MODE] == tp.ONCARD_MODE_HOST
    assert env[tp.ENV_ONCARD_SLOT_MIB] == "128"
    # step 6's two: the mode and the priced term
    assert env[wx.INJECT_ENV] == wx.INJECT_AUTHORITATIVE
    assert xb.ENV_BOUNCE_TERMS in env
    assert xb.read_published_terms(env[xb.ENV_BOUNCE_TERMS]) == _terms()


def test_the_shadow_arm_publishes_the_same_keys(tmp_path):
    """The two arms differ by AUTHORITY, not by what the ranks are told.

    That was already the function's own docstring ("the shadow needs exactly
    what the exchange needs, minus the authority"); the guard contradicted it.
    """
    keys_x = set(_env(lz.WEIGHT_SOURCE_EXCHANGE, dry=True,
                      bounce_terms=_terms()))
    keys_s = set(_env(sh.WEIGHT_SOURCE_SHADOW, dry=True,
                      bounce_terms=_terms()))
    assert keys_x == keys_s


# ===========================================================================
# (ii) UNDER `ring`, NOTHING -- byte-identical.
# ===========================================================================


class TheRingArmPublishesNothing:
    """Namespace only; the collected tests are the functions below."""


def test_the_ring_arm_publishes_nothing_at_all():
    """The empty dict is what keeps the default boot byte-identical: no region
    file, no semaphores, no environment.  Widening the guard to admit
    `exchange` may not widen it to admit `ring`.
    """
    assert _env(lz.WEIGHT_SOURCE_DEFAULT, dry=True) == {}
    assert _env(lz.WEIGHT_SOURCE_DEFAULT, dry=True,
                oncard_mode=tp.ONCARD_MODE_HOST, oncard_slot_mib=128,
                bounce_terms=_terms(),
                inject_mode=wx.INJECT_AUTHORITATIVE) == {}, \
        "ring published something when every other argument was armed"


def test_an_unknown_arm_publishes_nothing():
    """Fail closed on a string nobody recognises, never arm on a typo."""
    assert _env("exchagne", dry=True, bounce_terms=_terms()) == {}


# ===========================================================================
# (iii) THE MODE IS PROVEN FROM ARGV, NEVER FROM A DEFAULT.
# ===========================================================================


class TheModeComesFromTheFlag:
    """Namespace only; the collected tests are the functions below."""


def test_authoritative_on_the_command_line_reaches_the_ranks(tmp_path):
    """THE DEFECT THE BLOCKER HID.

    With nothing published, a rank fell back to `inject_mode()`'s default,
    which is `shadow` -- so `--weg2-xchg-inject authoritative` would have been
    silently ignored and the boot's mode was unprovable from its own argv.
    """
    env = _env(lz.WEIGHT_SOURCE_EXCHANGE, dry=True, bounce_terms=_terms(),
               inject_mode=wx.INJECT_AUTHORITATIVE)
    assert env[wx.INJECT_ENV] == wx.INJECT_AUTHORITATIVE
    # And a rank reading that environment agrees.
    old = os.environ.get(wx.INJECT_ENV)
    try:
        os.environ[wx.INJECT_ENV] = env[wx.INJECT_ENV]
        assert wx.inject_authoritative() is True
    finally:
        if old is None:
            os.environ.pop(wx.INJECT_ENV, None)
        else:
            os.environ[wx.INJECT_ENV] = old


def test_the_published_mode_is_the_flag_and_not_the_reader(tmp_path):
    """The publisher may not ask `inject_mode()` -- that reads the LAUNCHER's
    own environment, which does not have the flag in it.

    This is S6 fix E's defect in a new place: the launcher's process was
    started without the flag, so any value read from its own environment
    prices/publishes the default while the ranks were told something else.
    """
    import inspect

    src = inspect.getsource(lz.prepare_xchg_env)
    # Asserted on the ASSIGNMENT, not on the absence of a token: the docstring
    # names `inject_mode()` in order to explain why it is not called, and a
    # text-absence guard reads that epitaph as a live member (e25a88c2a2).
    assert "env[weight_exchange.INJECT_ENV] = str(inject_mode)" in src, src


def test_the_name_no_longer_says_shadow():
    """One reader, one name: the function arms BOTH arms now."""
    assert hasattr(lz, "prepare_xchg_env")
    assert not hasattr(lz, "prepare_shadow_env"), \
        "two names for one publisher is how two readers begin"


# ===========================================================================
# EXECUTION SMOKE: the seam driven by the PUBLISHED environment.
#
# Every other test of the seam sets the env vars by hand, which proves the
# seam reads them but not that the launcher WRITES the ones it reads. This one
# closes that loop: the env comes out of `prepare_xchg_env`, is applied
# verbatim, and the seam is driven through the product method -- so a name the
# publisher spells differently than the reader fails HERE rather than on the
# metal, which is the whole class of defect step 7 was.
# ===========================================================================


class TheSeamRunsOnThePublishedEnvironment:
    """Namespace only; the collected tests are the functions below."""


def test_the_seam_reads_the_mode_and_terms_the_launcher_published(tmp_path,
                                                                  monkeypatch):
    from .test_weg2_xchg_bounce_execution_smoke_1273 import (
        DEPTH, SLOT_BYTES, _all_descs, _manager, _seed_source,
    )
    from .test_weg2_xchg_transport_1273 import FakeDeviceOps

    published = _env(lz.WEIGHT_SOURCE_EXCHANGE, dry=True,
                     oncard_mode=tp.ONCARD_MODE_HOST, oncard_slot_mib=128,
                     bounce_terms=_terms(),
                     inject_mode=wx.INJECT_SHADOW)
    # APPLIED VERBATIM, as a rank process would receive it.
    for key, value in published.items():
        monkeypatch.setenv(key, str(value))

    # The reader agrees with the publisher, on all three of step 6/7's names.
    assert wx.exchange_armed() is True
    assert wx.inject_mode() == wx.INJECT_SHADOW
    assert xb.read_published_terms() == _terms()

    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    # NO `mode=` ARGUMENT: the leg must take it from the published env through
    # `wx.inject_mode()`, which is the path the product uses.
    result = _manager()._weg2_xchg_bounce_leg(
        descs=_all_descs(), ops=ops,
        # The region names are absent under `dry` (no region is created), so
        # the leg gets the nonce directly -- the publication under test here
        # is the MODE and the TERMS, which are added unconditionally.
        boot_nonce="b1",
        slot_bytes=SLOT_BYTES, depth=DEPTH, shm_root=str(tmp_path),
    )
    assert result.inject is not None, \
        "the leg did not take shadow mode from the published environment"
    assert result.inject.mode == wx.INJECT_SHADOW
    assert result.inject.pieces > 0


def test_the_published_slot_is_the_one_the_rank_sizes_from(tmp_path,
                                                           monkeypatch):
    """The one-value identity, end to end through the publication.

    `tp.ONCARD_SLOT_BYTES_MAX` resolves `ENV_ONCARD_SLOT_MIB` AT IMPORT, so a
    rank's per-card staging budget is fixed by what the launcher published
    before the rank started -- which is why the publisher and the reader must
    spell the same name, and why this asserts the arithmetic rather than the
    string.
    """
    published = _env(lz.WEIGHT_SOURCE_EXCHANGE, dry=True,
                     oncard_mode=tp.ONCARD_MODE_HOST, oncard_slot_mib=128,
                     bounce_terms=_terms())
    assert published[tp.ENV_ONCARD_SLOT_MIB] == "128"
    terms = xb.read_published_terms(published[xb.ENV_BOUNCE_TERMS])
    assert terms.slot_bytes == 128 * xb.MIB
    assert terms.staging_per_card == xb.staging_bytes_per_card(128 * xb.MIB)
