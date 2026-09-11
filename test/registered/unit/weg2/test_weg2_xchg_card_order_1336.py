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

OPERATOR RULING, FIRST FORM, then REVISED -- and the revision is the point.
The first ruling had the launcher publish a NEW variable
``SGLANG_WEG2_XCHG_CARD_UUIDS``.  Its premise ("no publisher exists") is
FALSIFIED BY MEASURED CODE: a canonical mover of this exact payload already
exists and already runs on our boots.

    RANK_CARD_UUIDS_ENV = "SGLANG_RANK_CARD_UUIDS"     registry/rank_cards.py:82
    publish_rank_card_uuids(server_args)               entrypoints/engine.py:671
      -- in the PARENT, BEFORE the spawn loop, whose own comment states our
         inheritance premise verbatim: "the channel is the environment, and a
         spawned scheduler inherits it only if it is set by now"
    resolve_rank_card_vector -> uuids[rank] = by_cuda_ordinal(ordinals[rank])
      -- source "launcher placement (gpu_id_for_rank -> #331 IdentityMap)",
         i.e. LAUNCHER PLACEMENT ORDER and explicitly not NVML enumeration
    readers rank_card_vector / rank_card_uuids, with their own length check
    weg2/launcher.py already passes ``--rank-gpu-id 0,1,2``, so the
      publisher's CUDA-side gate is satisfied on every weg2 boot

MEASURED on XSN11 (both runs, BOTH groups, from the boot logs):

    rank->card vector (launcher placement (gpu_id_for_rank -> #331 IdentityMap)):
      rank0=GPU-31d7ef41 [0A:00.0]  rank1=GPU-5c648f96 [05:00.0]  rank2=GPU-62dbbae1

byte-identical to the parent's ``CUDA_VISIBLE_DEVICES`` order, the SAME vector
on P and D, and ``no CUDA context`` 0.  So a second publisher would be the
one-job-one-mover violation the ruling invoked, entered from the producer side.

**REVISED RULING (OPTION B): the region CONSUMES the existing vector through
``rank_cards``' own reader and nothing else.  No launcher change; seat 4's
publisher is parked as the fallback.**  What stays from the first form: the
W23-family refusals and the NO-FALLBACK rule, because they are load-bearing
and independent of the source.

AND ONE THING THE REVISION ADDS, because two different quantities coincide
here by accident of form: ``rank_cards`` checks **WORLD_SIZE**, this module's
invariant is **N_CARDS**.  They are equal on the Weg-2 form only because each
group is three ranks and *rank n of either group runs on cards[n]*.  A future
arm (a pipeline stage, a MoE-TP subgroup, six ranks in one world) separates
them, and a vector of the wrong length would be silently re-indexed as a card
table.  The coincidence is therefore ASSERTED, not relied on.
"""

from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import rank_cards as RC  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

THREE = ("GPU-aaa", "GPU-bbb", "GPU-ccc")


@pytest.fixture()
def no_env(monkeypatch):
    """Nothing set, so every test states its own input."""
    monkeypatch.delenv(RC.RANK_CARD_UUIDS_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return monkeypatch


# ===========================================================================
# (1) THE NAME, fixed by the ruling, spelled ONCE.
# ===========================================================================


def test_the_producer_reads_that_variable_in_launcher_card_order(no_env):
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, ",".join(THREE))
    got = xr.uuid_of_card()
    assert got == THREE, got
    # INDEX == CARD ORDINAL is the whole contract; the launcher's order is not
    # NVML's, so this is the only thing that makes CROSS_PAIRS addressable.
    assert got[1] == "GPU-bbb"
    assert len(got) == xr.N_CARDS
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, " GPU-aaa , GPU-bbb ,GPU-ccc ")
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
    # `from sglang.srt.registry import rank_cards` is the OWNER import and is
    # the whole point of Option B; what stays forbidden is the DEVICE
    # ENUMERATION side of that same package.
    for forbidden in ("nvml.", "nvml_registry", "import nvml", "identity_map",
                      "torch.cuda", "device_count", "current_device("):
        assert forbidden not in src, \
            f"the producer must not call {forbidden!r}: a fallback map is wrong-ordinal"
    import sys
    before = {m for m in sys.modules if "nvml" in m.lower()}
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, ",".join(THREE))
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
    # rank_cards' own parser DROPS empty fields, so a blank entry reaches
    # this module as a SHORT vector and is refused `count`. Reclassified on
    # purpose: re-parsing the raw string here to say `blank` instead would be
    # a second bookkeeping of the same string (see the dedicated test below).
    ("GPU-aaa,,GPU-ccc", "count"),
    ("GPU-aaa, ,GPU-ccc", "count"),
    ("GPU-aaa,GPU-bbb,GPU-aaa", "duplicate"),
    ("GPU-aaa,GPU-aaa,GPU-aaa", "duplicate"),
])
def test_every_unusable_value_is_refused_by_name_with_its_reason(no_env, value, why):
    if value is not None:
        no_env.setenv(RC.RANK_CARD_UUIDS_ENV, value)
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
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, "GPU-aaa,GPU-bbb,GPU-aaa")
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
        no_env.setenv(RC.RANK_CARD_UUIDS_ENV,
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
        no_env.setenv(RC.RANK_CARD_UUIDS_ENV, value)
        try:
            xr.uuid_of_card()
        except xr.Weg2XchgCardUuidMapUnusable as exc:
            note = sh.exc_note(exc)
        assert "W23" in note, note
        assert "@ weight_exchange_region.py:" in note, note


# ===========================================================================
# (4) THE COINCIDENCE, ASSERTED. rank_cards checks WORLD_SIZE; this module's
#     invariant is N_CARDS. They are equal here by accident of form.
# ===========================================================================


def test_a_vector_whose_length_is_not_N_CARDS_is_refused_by_name(no_env):
    """THE RULING'S OWN PIN. Red at `f4b9c8cd91` in the six-rank direction.

    `SGLANG_RANK_CARD_UUIDS` is one uuid per WORLD rank; this module indexes
    by CARD ORDINAL. On the Weg-2 form both are 3 because each group is three
    ranks and *rank n of either group runs on cards[n]* -- an accident of this
    form, not a law. A world of six (one group, six ranks; a pipeline stage; a
    MoE-TP subgroup) publishes a six-entry vector, and re-indexing that as a
    card table would attribute one rank's card to another SILENTLY.
    """
    six = [f"GPU-{i}" for i in range(6)]
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, ",".join(six))
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    text = str(exc.value)
    assert "reason=count" in text, text
    assert "got=6" in text and f"want={xr.N_CARDS}" in text, text
    # The refusal must SAY that two different quantities were compared, or the
    # next reader will "fix" it by slicing the vector to three.
    assert "world" in text.lower(), text


def test_the_producer_does_not_slice_or_stretch_a_wrong_length_vector(no_env):
    """The danger direction of the check above: a helpful truncation."""
    for n in (1, 2, 4, 6):
        no_env.setenv(RC.RANK_CARD_UUIDS_ENV,
                      ",".join(f"GPU-{i}" for i in range(n)))
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable):
            xr.uuid_of_card()


def test_the_owners_length_check_is_still_there_and_we_do_not_depend_on_it():
    """Belt and braces, deliberately: `rank_cards` refuses a wrong-length
    vector when asked with a `world_size`, and this module refuses it again by
    its own name. Two checks of one fact is right here and is NOT a second
    bookkeeping: the quantities differ (world ranks vs cards), so each owner
    checks the quantity it owns."""
    v = RC.rank_card_vector.__doc__ or ""
    assert "world_size" in v
    assert "N_CARDS" in inspect.getsource(xr.uuid_of_card)


def test_a_blank_entry_reaches_us_as_a_short_vector_and_that_is_stated(no_env):
    """NAMED BEHAVIOUR, not a silent reclassification.

    `rank_cards._parse_env_vector` drops empty fields, so "a,,c" arrives as
    two entries. This module therefore refuses it `count`, not `blank`. The
    alternative -- re-parsing the raw string here to distinguish them -- would
    put a second parser of the same string in a second module, which is the
    defect Option B exists to avoid. The docstring says so, and this test is
    what keeps the statement true.
    """
    no_env.setenv(RC.RANK_CARD_UUIDS_ENV, "GPU-aaa,,GPU-ccc")
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    assert "reason=count" in str(exc.value)
    assert "drops" in (xr.uuid_of_card.__doc__ or "").lower(), \
        "the docstring must state that the owner's parser drops blank fields"


def test_an_absent_vector_carries_the_owners_own_reason(no_env):
    """A refusal that swallows the producer's reason makes the next reader
    rediscover it. `rank_cards` names why it has nothing; we forward it."""
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card()
    text = str(exc.value)
    assert "reason=unset" in text, text
    assert RC.RANK_CARD_UUIDS_ENV in text, text
