# SPDX-License-Identifier: Apache-2.0
"""#1336 -- the card ORDER has ONE producer, and it is the launcher.

THE MEASUREMENT THAT FORCED THIS INTERFACE (boot weg2xsn11, `1b84c4c69f`,
record `BOOT_weg2xsn11_0911.md`).  #1335 read the map from
``CUDA_VISIBLE_DEVICES`` on the premise -- `weight_updater.py:1494-1498`,
verbatim -- that *"the launcher hands EVERY rank the same
CUDA_VISIBLE_DEVICES -- all three card uuids, one string, both groups"*.  That
sentence is TRUE of the process the launcher SPAWNS and FALSE of the process
that RUNS THE LEG.  Read from ``/proc/<pid>/environ`` of the live ranks:

    launch_server (P)        CVD = GPU-31d7ef41,GPU-5c648f96,GPU-62dbbae1
    sglang::sched (P rank 0) CVD = GPU-31d7ef41
    sglang::sched (P rank 1) CVD = GPU-5c648f96
    sglang::sched (P rank 2) CVD = GPU-62dbbae1      (D identical)

Every scheduler process -- the one that runs the flip leg and the shadow hook
-- has ``CUDA_VISIBLE_DEVICES`` narrowed to its OWN SINGLE CARD, which is also
this repo's documented preferred isolation.  So the producer read one entry and
refused ``reason=count`` on all 39 agreed legs: correct refusal, wrong source.

**AND NVML IS NOT THE ANSWER, WHICH IS WHY THE NO-FALLBACK RULE IS A TEST AND
NOT A COMMENT.**  Also measured on that boot: the launcher's card order is
``GPU-31d7ef41 (5090), GPU-5c648f96 (3080), GPU-62dbbae1 (3080)`` while NVML
enumerates ``0=GPU-5c648f96, 1=GPU-31d7ef41, 2=GPU-62dbbae1``.  The two orders
DIFFER.  A producer that "helpfully" fell back to NVML or to CUDA enumeration
would return the right uuids under the WRONG ORDINALS -- a wrong ``PairStats``
label, and worse a wrong identity check that refuses correct ranks.  A refusal
is strictly better than a plausible map, so there is no fallback at all and a
mutant that adds one must go red.

OPERATOR RULING (2026-09-11, plan S6-BOUNCE): the launcher publishes
``SGLANG_WEG2_XCHG_CARD_UUIDS`` -- comma-separated GPU UUIDs in LAUNCHER CARD
ORDER, index == the card ordinal the region and pair tables use -- beside the
other xchg variables, on the shadow and exchange arms (the ring publishes
nothing), inherited by the re-exec'd scheduler children.  One producer, no
region handshake, no second producer of card order.
"""

from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

THREE = ("GPU-aaa", "GPU-bbb", "GPU-ccc")


@pytest.fixture()
def no_env(monkeypatch):
    """Neither variable set, so every test states its own input."""
    monkeypatch.delenv(xr.ENV_CARD_UUIDS, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return monkeypatch


# ===========================================================================
# (1) THE NAME, fixed by the ruling, spelled ONCE.
# ===========================================================================


def test_the_variable_is_the_ruled_name_and_is_a_published_constant():
    """Red at `8dc93297f7`: `xr.ENV_CARD_UUIDS` does not exist.

    A constant and not a literal, because the launcher publishes what the
    region reads and a retyped string is how two ends drift (the same reason
    the transport publishes `ENV_ONCARD_MODE` rather than spelling it twice).
    """
    assert xr.ENV_CARD_UUIDS == "SGLANG_WEG2_XCHG_CARD_UUIDS"
    src = inspect.getsource(xr.uuid_of_card)
    assert "SGLANG_WEG2_XCHG_CARD_UUIDS" not in src, \
        "the producer reads the CONSTANT, it does not retype the name"
    assert "ENV_CARD_UUIDS" in src


def test_the_producer_reads_that_variable_in_launcher_card_order(no_env):
    no_env.setenv(xr.ENV_CARD_UUIDS, ",".join(THREE))
    got = xr.uuid_of_card()
    assert got == THREE, got
    # INDEX == CARD ORDINAL is the whole contract; the launcher's order is not
    # NVML's, so this is the only thing that makes CROSS_PAIRS addressable.
    assert got[1] == "GPU-bbb"
    assert len(got) == xr.N_CARDS
    no_env.setenv(xr.ENV_CARD_UUIDS, " GPU-aaa , GPU-bbb ,GPU-ccc ")
    assert xr.uuid_of_card() == THREE


# ===========================================================================
# (2) NO FALLBACK. Ever. This is the danger direction of this slice.
# ===========================================================================


def test_the_producer_never_falls_back_to_cuda_visible_devices(no_env):
    """CVD holding a perfectly good three-uuid string must NOT satisfy it.

    This is the #1335 source, and on the process that matters it holds ONE
    card -- but even when it holds three (the parent's value) it is the wrong
    authority, because its order is not guaranteed to be the card order the
    pair tables use. The variable is the contract; nothing else is.
    """
    no_env.setenv("CUDA_VISIBLE_DEVICES", ",".join(THREE))
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    assert "reason=unset" in str(exc.value), str(exc.value)


def test_the_producer_never_reaches_for_nvml_or_torch(no_env):
    """A plausible map is WORSE than a refusal -- measured, not asserted.

    The launcher's order (5090 first on this rig) and NVML's (3080 first)
    differ, so an NVML fallback returns the right uuids under the wrong
    ordinals: a silently wrong label and a wrong identity check. Pinned two
    ways -- the source may not mention either module, and the call may not
    import them.
    """
    # CALL SHAPES, not mentions: the docstring EXPLAINS the absence of a
    # fallback and naming NVML there is the documentation doing its job. A
    # test that forbade the word would forbid the explanation -- it did, on
    # its first run, and that is a finding about the test.
    src = inspect.getsource(xr.uuid_of_card)
    for forbidden in ("nvml.", "nvml_registry", "from sglang.srt.registry",
                      "torch.cuda", "device_count", "current_device("):
        assert forbidden not in src, \
            f"the producer must not call {forbidden!r}: a fallback map is wrong-ordinal"
    import sys
    before = {m for m in sys.modules if "nvml" in m.lower()}
    no_env.setenv(xr.ENV_CARD_UUIDS, ",".join(THREE))
    xr.uuid_of_card()
    assert {m for m in sys.modules if "nvml" in m.lower()} == before, \
        "resolving the map imported an NVML module"


def test_the_whole_region_module_has_no_card_enumeration_fallback():
    """The rule is about the MODULE, not one function: a helper that enumerates
    devices and a producer that calls it is the same defect one hop away."""
    src = inspect.getsource(xr)
    assert "nvmlDeviceGetHandleByIndex" not in src
    assert "torch.cuda.device_count" not in src
    assert "current_device_uuid" not in src


# ===========================================================================
# (3) THE FOUR REFUSALS the ruling names, each BY NAME and with its reason.
# ===========================================================================


@pytest.mark.parametrize("value,why", [
    (None, "unset"),
    ("", "unset"),
    ("   ", "unset"),
    ("GPU-aaa,GPU-bbb", "count"),
    ("GPU-aaa,GPU-bbb,GPU-ccc,GPU-ddd", "count"),
    ("GPU-aaa,,GPU-ccc", "blank"),
    ("GPU-aaa, ,GPU-ccc", "blank"),
    ("GPU-aaa,GPU-bbb,GPU-aaa", "duplicate"),
    ("GPU-aaa,GPU-aaa,GPU-aaa", "duplicate"),
])
def test_every_unusable_value_is_refused_by_name_with_its_reason(no_env, value, why):
    if value is not None:
        no_env.setenv(xr.ENV_CARD_UUIDS, value)
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    text = str(exc.value)
    assert text.startswith("W23 Weg2XchgCardUuidMapUnusable"), text
    assert f"reason={why}" in text, text


def test_duplicate_is_its_own_reason_and_says_why_it_cannot_be_tolerated(no_env):
    """Two ordinals naming ONE card is not a cosmetic problem.

    `CROSS_PAIRS` addresses six DIRECTED cross-card pairs by ordinal. If two
    ordinals hold the same uuid, a "cross" pair is really on-card: the staging
    lane would be asked to move bytes from a card to itself while the diagonal
    carrier -- the thing that exists for exactly that -- sits unused. The
    refusal names the repeated uuid.
    """
    no_env.setenv(xr.ENV_CARD_UUIDS, "GPU-aaa,GPU-bbb,GPU-aaa")
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    text = str(exc.value)
    assert "reason=duplicate" in text and "GPU-aaa" in text, text


def test_the_count_must_equal_N_CARDS_exactly_never_at_least(no_env):
    """Both directions, because only one of them is the obvious one.

    Too FEW is the boot's own failure. Too MANY is the quieter defect: a
    producer that truncated a four-entry value would run happily on a rig
    whose card table it does not describe.
    """
    for n in (1, 2, 4, 5):
        no_env.setenv(xr.ENV_CARD_UUIDS,
                      ",".join(f"GPU-{i}" for i in range(n)))
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
            xr.uuid_of_card()
        text = str(exc.value)
        assert "reason=count" in text, text
        assert f"got={n}" in text and f"want={xr.N_CARDS}" in text, text


def test_the_refusal_is_countable_by_the_wcode_census(no_env):
    """The lesson of weg2xsn11: a code that never reaches the log is
    unfalsifiable. Every reason carries `W23` in its first token."""
    from sglang.srt.weg2 import weight_exchange_shadow as sh
    for value in ("", "GPU-a,GPU-b", "GPU-a,,GPU-c", "GPU-a,GPU-b,GPU-a"):
        no_env.setenv(xr.ENV_CARD_UUIDS, value)
        try:
            xr.uuid_of_card()
        except xr.Weg2XchgCardUuidMapUnusable as exc:
            note = sh.exc_note(exc)
        assert "W23" in note, note
        assert "@ weight_exchange_region.py:" in note, note


# ===========================================================================
# (4) INJECTION stays, because the doubles have no launcher.
# ===========================================================================


def test_an_injected_string_bypasses_the_environment_but_not_the_rules(no_env):
    assert xr.uuid_of_card(env=",".join(THREE)) == THREE
    for bad, why in (("", "unset"), ("a,b", "count"), ("a,a,a", "duplicate")):
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
            xr.uuid_of_card(env=bad)
        assert f"reason={why}" in str(exc.value)
