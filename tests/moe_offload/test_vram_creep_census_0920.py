# SPDX-License-Identifier: Apache-2.0
"""#1491: name the post that takes the card between one wake and the next.

THE QUESTION THAT HAD NO INSTRUMENT. Boots weg2xsn406 and weg2xsn408 both
died at their SECOND flip because the kv_cache resume no longer fit:

    D TP1  wake 1 (17:55:41)  WEG2-WAKE-KV-FIRST LATE free=8387 MiB need=6904 MiB
    D TP1  wake 2 (17:57:21)  WEG2-WAKE-KV-FIRST LATE free=5974 MiB need=6904 MiB

2413 MiB gone between two wakes of the same rank. This rank's own dormant
residue accounts for a tenth of it -- `WEG2-SLEEP-RESIDUE` reads
untagged_live 224 -> 437 MiB and nvml_proc_used 1598 -> 1830 MiB -- and
nothing in any log said where the rest went, because `WEG2-DC-BREAKDOWN`
(a) runs only at the SLEEP (`stage=release`) and (b) decomposes only THIS
pid. The co-resident P group prefills on the same card between D's wakes and
appeared in no reading at all.

So two things are added and tested here: the CARD terms (total, free, and the
residual "held by someone who is not me"), and the wake-to-wake DELTA of every
post, with the biggest taker named.

Hermetic: every function under test is pure. The measured MiB figures below
are read off the two boot logs verbatim.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers.weg2_memory_saver import (
    DC_CREEP_POSTS,
    dc_breakdown,
    dc_creep,
    forget_dc,
    format_dc_breakdown,
    format_dc_creep,
    remember_dc,
)

MiB = 1 << 20


def _tp1(free_mib, nvml_mib, reserved_mib, allocated_mib):
    """D TP1 of xsn408 at one wake, from its own log lines."""
    return dc_breakdown(
        nvml_proc_bytes=nvml_mib * MiB,
        torch_reserved=reserved_mib * MiB,
        torch_allocated=allocated_mib * MiB,
        tag_bytes={"kv_cache": 6904 * MiB, "cuda_graph": 488 * MiB},
        offload_tags=("kv_cache", "cuda_graph"),
        card_total_bytes=20000 * MiB,
        card_free_bytes=free_mib * MiB,
    )


WAKE1 = dict(free_mib=8387, nvml_mib=1598, reserved_mib=17458, allocated_mib=16662)
WAKE2 = dict(free_mib=5974, nvml_mib=1830, reserved_mib=17676, allocated_mib=16875)


@pytest.fixture(autouse=True)
def _clean_store():
    forget_dc()
    yield
    forget_dc()


# --- the card terms that were missing ---------------------------------------


def test_the_card_residual_is_total_minus_free_minus_mine():
    rec = _tp1(**WAKE2)
    assert rec["card_total_mib"] == 20000
    assert rec["card_free_mib"] == 5974
    assert rec["card_other_procs_mib"] == 20000 - 5974 - 1830


def test_a_card_reading_that_could_not_be_taken_is_none_not_zero():
    """NULL-NUR-BEI-ERREICHTEM-EMITTER. A zero here would read as 'nobody else
    is on this card', which is the claim that cost two boots."""
    rec = dc_breakdown(
        nvml_proc_bytes=1830 * MiB,
        torch_reserved=17676 * MiB,
        torch_allocated=16875 * MiB,
        tag_bytes={"kv_cache": 6904 * MiB},
        offload_tags=("kv_cache",),
    )
    assert rec["card_total_mib"] is None
    assert rec["card_free_mib"] is None
    assert rec["card_other_procs_mib"] is None


def test_the_residual_needs_all_three_readings():
    rec = dc_breakdown(
        nvml_proc_bytes=None,
        torch_reserved=17676 * MiB,
        torch_allocated=16875 * MiB,
        tag_bytes={},
        offload_tags=(),
        card_total_bytes=20000 * MiB,
        card_free_bytes=5974 * MiB,
    )
    assert rec["card_other_procs_mib"] is None


def test_the_residual_never_goes_negative():
    """NVML per-process sums can exceed total-free by rounding across pids; a
    negative 'someone else holds' is nonsense, so it clamps at zero."""
    rec = dc_breakdown(
        nvml_proc_bytes=9000 * MiB,
        torch_reserved=0,
        torch_allocated=0,
        tag_bytes={},
        offload_tags=(),
        card_total_bytes=20000 * MiB,
        card_free_bytes=15000 * MiB,
    )
    assert rec["card_other_procs_mib"] == 0


def test_the_breakdown_line_carries_the_card_terms():
    line = format_dc_breakdown(_tp1(**WAKE2), stage="wake-pre-kv epoch=7")
    assert "card free 5974 of 20000 MiB total" in line
    assert "other processes hold 12196 MiB (residual total-free-mine)" in line


def test_the_breakdown_line_says_na_when_the_card_was_unreadable():
    rec = dc_breakdown(nvml_proc_bytes=None, torch_reserved=None,
                       torch_allocated=None, tag_bytes={}, offload_tags=())
    line = format_dc_breakdown(rec, stage="wake-pre-kv epoch=1")
    assert "card free n/a of n/a MiB total" in line
    assert "other processes hold n/a MiB" in line


# --- the wake-to-wake delta -------------------------------------------------


def test_the_measured_xsn408_pair_names_the_co_resident_group():
    """The whole point, on the measured numbers: nine tenths of the loss is
    NOT this rank's."""
    delta = dc_creep(_tp1(**WAKE1), _tp1(**WAKE2))
    assert delta["card_free_mib"] == -2413
    assert delta["nvml_proc_mib"] == 232, "this rank's own growth, from SLEEP-RESIDUE"
    assert delta["card_other_procs_mib"] == 2181
    line = format_dc_creep(delta, stage="wake-pre-kv epoch=7")
    assert "card_free -2413 MiB" in line
    assert "BIGGEST TAKER card_other_procs +2181 MiB" in line


def test_this_ranks_own_creep_is_the_allocator_tail_not_a_tag():
    delta = dc_creep(_tp1(**WAKE1), _tp1(**WAKE2))
    assert delta["tms_paused_mib"] == 0 and delta["tms_resident_mib"] == 0
    assert delta["torch_untagged_mib"] == 218


def test_card_free_is_never_nominated_as_a_taker():
    """It is the headline, and growth in it is good news."""
    prev = {p: 0 for p in DC_CREEP_POSTS}
    cur = dict(prev, card_free_mib=5000, torch_untagged_mib=7)
    line = format_dc_creep(dc_creep(prev, cur), stage="s")
    assert "BIGGEST TAKER torch_untagged +7 MiB" in line
    assert "BIGGEST TAKER card_free" not in line


def test_no_post_grew_is_said_plainly():
    prev = {p: 100 for p in DC_CREEP_POSTS}
    cur = {p: 100 for p in DC_CREEP_POSTS}
    assert "no post grew" in format_dc_creep(dc_creep(prev, cur), stage="s")


def test_a_post_absent_from_either_record_is_skipped_not_zeroed():
    """A zero delta built from two absences is the exact shape that made this
    creep invisible for two boots."""
    prev = {"card_free_mib": 8387, "card_other_procs_mib": None}
    cur = {"card_free_mib": 5974, "card_other_procs_mib": 12196}
    delta = dc_creep(prev, cur)
    assert delta == {"card_free_mib": -2413}
    line = format_dc_creep(delta, stage="s")
    assert "card_other_procs=" not in line, "an absent post must not appear as a delta"
    assert "BIGGEST TAKER" not in line


def test_a_first_wake_has_nothing_to_compare_and_says_so():
    assert dc_creep(None, _tp1(**WAKE1)) is None
    line = format_dc_creep(None, stage="wake-pre-kv epoch=1")
    assert "n/a" in line and "an absence, not a zero" in line


def test_two_records_with_no_common_post_are_none():
    assert dc_creep({"card_free_mib": 1}, {"nvml_proc_mib": 1}) is None


# --- the per-rank history store ---------------------------------------------


def test_remember_returns_the_previous_record_and_stores_the_new_one():
    a, b = _tp1(**WAKE1), _tp1(**WAKE2)
    assert remember_dc("wake-pre-kv", a) is None
    prev = remember_dc("wake-pre-kv", b)
    assert prev is not None and prev["card_free_mib"] == 8387


def test_the_store_is_keyed_so_two_stages_do_not_cross_talk():
    remember_dc("wake-pre-kv", _tp1(**WAKE1))
    assert remember_dc("release", _tp1(**WAKE2)) is None


def test_a_stored_record_is_a_copy():
    rec = _tp1(**WAKE1)
    remember_dc("wake-pre-kv", rec)
    rec["card_free_mib"] = 999
    assert remember_dc("wake-pre-kv", _tp1(**WAKE2))["card_free_mib"] == 8387


def test_remembering_none_keeps_the_last_good_record():
    """A wake whose instrument failed must not erase the comparison point."""
    remember_dc("wake-pre-kv", _tp1(**WAKE1))
    assert remember_dc("wake-pre-kv", None)["card_free_mib"] == 8387
    assert remember_dc("wake-pre-kv", None)["card_free_mib"] == 8387


def test_forget_one_key_leaves_the_others():
    remember_dc("a", _tp1(**WAKE1))
    remember_dc("b", _tp1(**WAKE2))
    forget_dc("a")
    assert remember_dc("a", None) is None
    assert remember_dc("b", None)["card_free_mib"] == 5974
