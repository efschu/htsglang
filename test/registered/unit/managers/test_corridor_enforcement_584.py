# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#584: the CAN-FAIL CORPUS, as configurations this rig actually booted.

A gate that has never been observed to refuse is indistinguishable from a gate
that is never reached, and this tree has shipped several of those. So the
corpus is not invented inputs: every member below is a configuration that ran
on this box on 2026-08-15, with the numbers its own 100 ms NVML sampler
recorded, and each states which way the gate must fail on it.

THE FAMILY THESE MEMBERS BELONG TO is silent falseness: something that could
not answer returned the answer that looks like success. Members 3 and 4 are
that shape directly, which is why they are here beside the two sizing ones --
the gate's job is not only "is this configuration lawful" but "was it
CHECKED", and the second question is the one that kept getting skipped.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers import corridor_enforcement as ce
from sglang.srt.managers import corridor_guard as cg

# The band in force, from the one declaration.
FLOOR, CENTRE, CEILING = 819, 1024, 1229


def _m(**free):
    return ce.predictions_from_measurement(free)


def test_the_band_constants_match_the_one_declaration():
    """This file quotes the band; if it ever drifts from corridor_guard the
    corpus would be judging against a private copy."""
    assert cg.corridor_band_mib() == (FLOOR, CENTRE, CEILING)


# ---------------------------------------------------------------------------
# MEMBER 1 -- the shipped loose vector. ~12.8 GiB idle.
# ---------------------------------------------------------------------------


def test_member1_the_shipped_loose_vector_is_refused_for_being_under_filled():
    """`--rank-gpu-memory-mib 31800,14000,15600`, measured at rest
    3545/5210/4325 MiB. This is what the box shipped with, and nothing ever
    refused it: ~12.8 GiB of VRAM buying no tokens, held so a flip seam could
    be funded out of idle memory."""
    verdict, cards = ce.evaluate(
        _m(gpu0=3545, gpu1=5210, gpu2=4325),
        floor_mib=FLOOR,
        ceiling_mib=CEILING,
    )
    assert verdict is ce.Verdict.ABOVE_CEILING
    assert all(c.verdict is ce.Verdict.ABOVE_CEILING for c in cards)
    with pytest.raises(ce.CorridorConfigRefused, match="ABOVE_CEILING"):
        ce.enforce(
            _m(gpu0=3545, gpu1=5210, gpu2=4325),
            floor_mib=FLOOR,
            ceiling_mib=CEILING,
        )


# ---------------------------------------------------------------------------
# MEMBER 2 -- over-compliant on the binding card, still idle on the others.
# ---------------------------------------------------------------------------


def test_member2_one_card_in_band_does_not_make_a_configuration_lawful():
    """`31800,15800,17700`, measured at rest 1223/3010/1751 MiB.

    gpu0 is inside the band. The group verdict is still a refusal, because the
    law is stated PER CARD and a fleet is only as lawful as the card that
    breaks it. Averaging these three would report ~1995 MiB and look almost
    reasonable, which is exactly how five breaches on one card hid behind two
    comfortable ones in the #656 acceptance."""
    verdict, cards = ce.evaluate(
        _m(gpu0=1223, gpu1=3010, gpu2=1751), floor_mib=FLOOR, ceiling_mib=CEILING
    )
    by = {c.card: c.verdict for c in cards}
    assert by["gpu0"] is ce.Verdict.IN_BAND
    assert by["gpu1"] is ce.Verdict.ABOVE_CEILING
    assert by["gpu2"] is ce.Verdict.ABOVE_CEILING
    assert verdict is ce.Verdict.ABOVE_CEILING


# ---------------------------------------------------------------------------
# MEMBER 3 -- the cold-boot seam-record fluke. A number that is not reproducible.
# ---------------------------------------------------------------------------


def test_member3_an_unreproducible_sizing_may_not_be_certified():
    """The same command line, minutes apart, sized the KV pool at 620000 and
    then 406600 tokens -- because `phase_flip_seam_reserve` writes a per-config
    record and a boot with a NEW budget vector finds it COLD and sizes with no
    seam term at all. The first boot of a vector therefore looks tight and
    lawful and the second does not, from identical inputs.

    A prediction from such a source is not MEASURED, whatever it was read
    from: it describes one draw of a two-valued process. The gate must treat
    it as unchecked rather than certify whichever draw it happened to see."""
    cold = ce.CardPrediction(
        "gpu0",
        1139,
        ce.Provenance.UNKNOWN,
        "cold seam record: the same vector sized 620000 then 406600 tokens, so "
        "this resting position is one draw of a two-valued process",
    )
    verdict, _ = ce.evaluate([cold], floor_mib=FLOOR, ceiling_mib=CEILING)
    assert verdict is ce.Verdict.UNVERIFIABLE
    with pytest.raises(ce.CorridorConfigRefused, match="NOT CHECKED"):
        ce.enforce([cold], floor_mib=FLOOR, ceiling_mib=CEILING)


# ---------------------------------------------------------------------------
# MEMBER 4 -- the dead audit. Green because blind.
# ---------------------------------------------------------------------------


def test_member4_a_law_nothing_measured_is_not_a_law_that_held():
    """`corridor_trace.start()` returned None unless an env var was set, so on
    a default boot the only continuous corridor instrument never armed. Across
    two boots an external sampler recorded 57 and 15 breaches (minima 895 and
    935 MiB) while the serving logs contained ZERO breach lines and
    `corridor_shortfall_bytes` stayed 0.

    The in-process verdict was "no breach reported". This pins that such a
    verdict is UNVERIFIABLE and refuses -- if the gate passed on it, the gate
    would be the next member of this corpus rather than its keeper."""
    blind = ce.CardPrediction(
        "gpu0",
        None,
        ce.Provenance.UNKNOWN,
        "no corridor sampler armed on this boot, so nothing observed the law",
    )
    verdict, cards = ce.evaluate([blind], floor_mib=FLOOR, ceiling_mib=CEILING)
    assert verdict is ce.Verdict.UNVERIFIABLE
    assert "nothing observed the law" in cards[0].describe(FLOOR, CEILING)
    with pytest.raises(ce.CorridorConfigRefused):
        ce.enforce([blind], floor_mib=FLOOR, ceiling_mib=CEILING)


# ---------------------------------------------------------------------------
# MEMBER 5 -- the ACCEPTED configuration. Still refused in one direction.
# ---------------------------------------------------------------------------


def test_member5_the_accepted_config_is_still_above_the_ceiling_at_rest():
    """`31800,19000,19000` @ 8ca27d49c7: at rest 1763/2572/2891 MiB.

    This is the configuration this window accepted, and it passes everything
    it was accepted on -- pool 518949, zero samples below the 819 floor under
    load, tp_to_pp reachable with the seam funded from KV. It is in the corpus
    anyway, because at REST all three cards sit above the ceiling and the gate
    must say so about the boot its own author signed off.

    The lever is not the memory vector: under PP the KV mass per card follows
    the FULL-ATTENTION count per stage, which `--pp-stage-ratio 14,10,8`
    resolves to [7,5,4] of 16, so one shared token count against an uneven
    per-card cost lets the tightest stage stop the pool while the others keep
    slack. Closing it is a LAYER CUT (`planner/pp_cut.solve_pp_cut`)."""
    verdict, cards = ce.evaluate(
        _m(gpu0=1763, gpu1=2572, gpu2=2891), floor_mib=FLOOR, ceiling_mib=CEILING
    )
    assert verdict is ce.Verdict.ABOVE_CEILING
    assert all(c.verdict is ce.Verdict.ABOVE_CEILING for c in cards)


def test_member5_under_load_the_binding_card_does_come_into_band():
    """The same boot's measured minima under a 12-request load:
    905/2272/1401. gpu0 enters the band; nothing goes below the floor. So the
    refusal above is about the RESTING position, which is the right thing for
    a boot-time gate to judge -- and the distinction is why the gate is fed
    at-rest numbers rather than load minima."""
    verdict, cards = ce.evaluate(
        _m(gpu0=905, gpu1=2272, gpu2=1401), floor_mib=FLOOR, ceiling_mib=CEILING
    )
    by = {c.card: c.verdict for c in cards}
    assert by["gpu0"] is ce.Verdict.IN_BAND
    assert verdict is ce.Verdict.ABOVE_CEILING  # cards 1 and 2 still idle
    assert not any(c.verdict is ce.Verdict.BELOW_FLOOR for c in cards)


# ---------------------------------------------------------------------------
# The gate's own properties.
# ---------------------------------------------------------------------------


def test_a_lawful_configuration_passes():
    verdict, _ = ce.evaluate(
        _m(gpu0=1024, gpu1=900, gpu2=1200), floor_mib=FLOOR, ceiling_mib=CEILING
    )
    assert verdict is ce.Verdict.IN_BAND
    assert ce.enforce(
        _m(gpu0=1024, gpu1=900, gpu2=1200), floor_mib=FLOOR, ceiling_mib=CEILING
    )[0].is_pass


def test_below_the_floor_outranks_above_the_ceiling():
    """A breach and an over-fill in one fleet reports the BREACH: it is the
    half of the law that costs correctness rather than throughput."""
    verdict, _ = ce.evaluate(
        _m(gpu0=700, gpu1=5000), floor_mib=FLOOR, ceiling_mib=CEILING
    )
    assert verdict is ce.Verdict.BELOW_FLOOR


def test_unverifiable_outranks_every_sizing_verdict():
    """An unchecked card is worse news than a badly sized one, because a
    badly sized one at least reports a number somebody measured."""
    verdict, _ = ce.evaluate(
        [
            ce.CardPrediction("gpu0", 700, ce.Provenance.MEASURED),
            ce.CardPrediction("gpu1", None, ce.Provenance.UNKNOWN, "nothing solved it"),
        ],
        floor_mib=FLOOR,
        ceiling_mib=CEILING,
    )
    assert verdict is ce.Verdict.UNVERIFIABLE


def test_an_empty_prediction_set_is_never_a_pass():
    """The case a refactor produces by accident."""
    verdict, cards = ce.evaluate([], floor_mib=FLOOR, ceiling_mib=CEILING)
    assert verdict is ce.Verdict.UNVERIFIABLE
    assert cards == []


def test_declared_slack_widens_the_ceiling_and_never_the_floor():
    over = _m(gpu0=1500)
    assert (
        ce.evaluate(over, floor_mib=FLOOR, ceiling_mib=CEILING, slack_mib=400)[0]
        is ce.Verdict.IN_BAND
    )
    under = _m(gpu0=700)
    assert (
        ce.evaluate(under, floor_mib=FLOOR, ceiling_mib=CEILING, slack_mib=400)[0]
        is ce.Verdict.BELOW_FLOOR
    ), "slack must not be usable to excuse a breach"


def test_non_strict_reports_the_same_verdict_without_raising():
    verdict, _ = ce.enforce(
        _m(gpu0=3545), floor_mib=FLOOR, ceiling_mib=CEILING, strict=False
    )
    assert verdict is ce.Verdict.ABOVE_CEILING


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def test_a_verdict_is_only_as_strong_as_its_weakest_input():
    assert (
        ce.weakest([ce.Provenance.MEASURED, ce.Provenance.DEFAULTED])
        is ce.Provenance.DEFAULTED
    )
    assert (
        ce.weakest([ce.Provenance.MEASURED, ce.Provenance.UNKNOWN])
        is ce.Provenance.UNKNOWN
    )
    assert ce.weakest([]) is ce.Provenance.UNKNOWN, "no inputs is not measured"


def test_the_planner_says_it_cannot_solve_a_pipeline_budget_and_names_the_home():
    """The documented gap, pinned so it cannot be quietly assumed away.

    If somebody later teaches a solver this topology, this test fails and its
    replacement should assert a SOLVED provenance instead -- which is the
    correct way for this pin to die."""

    class _Args:
        pp_size = 3
        tp_size = 1

    preds = ce.predict_from_planner(_Args(), ["gpu0", "gpu1", "gpu2"])
    assert [p.provenance for p in preds] == [ce.Provenance.UNKNOWN] * 3
    assert all(p.predicted_free_mib is None for p in preds)
    assert "solve_pp_cut" in preds[0].detail
    assert "pp_size=3" in preds[0].detail
    verdict, _ = ce.evaluate(preds, floor_mib=FLOOR, ceiling_mib=CEILING)
    assert verdict is ce.Verdict.UNVERIFIABLE


def test_a_measured_reading_is_the_strongest_provenance_this_rig_can_make():
    preds = ce.predictions_from_measurement({"gpu0": 1100}, detail="100 ms NVML free")
    assert preds[0].provenance is ce.Provenance.MEASURED
    assert preds[0].predicted_free_mib == 1100


# ---------------------------------------------------------------------------
# MEMBER 6 -- a number that LOOKS solved and is not.
#
# The phase policy's break-even ladder is derived from three measured knobs.
# An operator set them, the derivation read the module CONSTANTS instead, and
# the boot logged a ladder that looked solved: the measured 5.918 s seam
# produced [7004, 19430, 21589, 22669, 23316] where the solve gives
# [7878, 11817, 13130, 13786, 14180]. Rung 0 had not moved at all, because it
# was still the constant's number.
#
# Nothing was wrong with the solver or the inputs. The number had a provenance
# nobody checked -- which is this corpus's whole subject.
# ---------------------------------------------------------------------------

LADDER_FROM_DEFAULTS = [7004, 10506, 11674, 12257, 12608]
LADDER_FROM_INPUTS = [7878, 11817, 13130, 13786, 14180]


def test_member6_a_knob_that_never_reached_the_number_is_not_SOLVED():
    """The bug as it shipped: knobs set, ladder still the constants'."""
    prov = ce.derived_provenance(
        inputs_supplied=True,
        value_from_inputs=LADDER_FROM_DEFAULTS,  # what the boot logged
        value_from_defaults=LADDER_FROM_DEFAULTS,
    )
    assert prov is ce.Provenance.DEFAULTED


def test_member6_the_fixed_derivation_is_SOLVED():
    prov = ce.derived_provenance(
        inputs_supplied=True,
        value_from_inputs=LADDER_FROM_INPUTS,
        value_from_defaults=LADDER_FROM_DEFAULTS,
    )
    assert prov is ce.Provenance.SOLVED


def test_member6_no_inputs_is_honestly_defaulted():
    assert (
        ce.derived_provenance(
            inputs_supplied=False,
            value_from_inputs=LADDER_FROM_DEFAULTS,
            value_from_defaults=LADDER_FROM_DEFAULTS,
        )
        is ce.Provenance.DEFAULTED
    )


def test_member6_a_defaulted_ladder_cannot_certify_a_boot():
    """Wired to the gate: a DEFAULTED input drags the verdict down, so a boot
    cannot be certified on a number whose knobs never reached it."""
    prov = ce.derived_provenance(
        inputs_supplied=True,
        value_from_inputs=LADDER_FROM_DEFAULTS,
        value_from_defaults=LADDER_FROM_DEFAULTS,
    )
    assert ce.weakest([ce.Provenance.MEASURED, prov]) is ce.Provenance.DEFAULTED


def test_member6_under_claiming_is_the_safe_direction():
    """Inputs that happen to reproduce the default exactly report DEFAULTED.
    A gate that refuses a correctly-configured boot is cheaper than one that
    certifies a silently-unwired knob."""
    assert (
        ce.derived_provenance(
            inputs_supplied=True, value_from_inputs=42, value_from_defaults=42
        )
        is ce.Provenance.DEFAULTED
    )
