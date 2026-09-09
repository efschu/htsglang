# SPDX-License-Identifier: Apache-2.0
"""#1257 -- the corridor law as a hard constraint on group D's budget solve.

RED-FIRST ANCHOR, and the reason this file exists rather than a docstring: at
the parent commit ``80de2d31d1`` group D's budget line subtracts a per-card
EXPECTATION (``CORRIDOR_MIB = 1024 + 404`` at ``launcher.py:109``, plus one
cross-boot ``D_OVERSHOOT_MIB``) and NOTHING ever compares that expectation with
what the front's corridor sampler measures.  Boot weg2sb5f is the whole case in
three numbers: the 5090 was budgeted as if it would leave 1428 + 489 MiB, and
under load it left **474**.  550 MiB below a law nobody was checking.

EVERY FIXTURE NUMBER BELOW IS A LOG LINE, named at its constant.  Nothing here
is chosen, estimated or rounded, and the two derived terms are asserted against
the two independent measurements they must reproduce:

* ``predicted_free`` at the sample's own budgets must equal the sampler's
  measured free-under-load column, per card (474 / 1095 / 1103);
* the predicted world pool must equal the pool group D actually served
  (674,080), and the re-solved vector must equal the one the runtime actually
  installed (``[17, 7, 8]``).

An instrument that could not reproduce those five numbers would be a model, not
a measurement, and its SATISFIED would be worth nothing.

THE FIVE MUTANTS are all on the danger direction -- the two ways this pass
could quietly do harm:

  (a) LOWER THE BINDER.  A cut on the rank that achieves ``min_r(P_r // v_r)``
      shrinks the served context for every rank at once.  M1/M1b pin both
      halves: the refusal fires on the binding card, and does NOT fire on a
      non-binding one (a blanket refusal would pass half of this by accident).
  (b) SAY SATISFIED ON A BREACHING CARD.  Three separate terms can produce that
      lie and each is mutated out in turn: the driver carve-out (M2 -- reading
      the free column as ``total - used``, the form the corridor rule forbids),
      the unbudgeted awake residue (M3 -- the brief's bare formula), and the
      load transient (M4).  M5 pins the missing-input path to byte-identical
      budgets, because a silent default there is the same lie with no line at
      all.

M6 AND M7 were added by the refuter pass (2026-09-09) and are the same danger
direction reached by two SILENT routes rather than a wrong number:

  M6. The re-solved world pool cannot be priced (the runtime's own
      ``partition_units`` unimportable in this process).  The first draft then
      fell back to the FIXED-vector check alone and APPLIED the cut -- a
      constraint satisfied on paper, on the one input that is derived rather
      than read.  It must refuse instead, and M6 pins both halves: APPLIED
      while the pool is priceable, REFUSED-UNPRICED-POOL when it is not.
  M7. The sample's rows are paired with this boot's cards BY UUID, but the
      token vector is positional.  If NVML enumeration order shifted between
      the sample's boot and this one, the pool and the binder come out
      plausible and belong to the wrong cards.  M7 shifts it and requires an
      UNPRICED refusal.

W-CODE: this pass owns W54, W55 and W56 (#1257c renumbered W53 -> W55: #1291 claimed W53 on the base between this pass's enumeration and its merge).  It shipped as W52 for one revision, which
#1290 already held at the base commit; the collision census could not see that
claim because #1290 writes the code in two whitespace-free forms.  Both the
label and the census are fixed in the same commit -- see
``test_weg2_wcode_uniqueness_1263.py``, which now reads all three forms.
"""

import ast
import inspect
import json
import os
import tempfile
import textwrap
import unittest
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import corridor_budget as cb
from sglang.test.test_utils import CustomTestCase

# --------------------------------------------------------------------------
# FIXTURE -- boot weg2sb5f 4f762260ba, 2026-09-09.  Each block names its line.
# --------------------------------------------------------------------------

#: front.log:99-101 (WEG2-LAUNCH budget D) and :96-98 (WEG2-DC group=P:
#: measured dormant image + "driver-reserved" carve-out + NVML total).
SB5F_UUID_5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SB5F_UUID_3080_A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SB5F_UUID_3080_B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

#: corridor_weg2sb5f_idle.json / corridor_weg2sb5f_load.json, free_min_mib,
#: n=600 samples each over 60 s at 100 ms.  The 5090 breaches idle ALREADY
#: (-446 to the law) and load costs it another 104 MiB; the 3080s sit +71..+87
#: over the law and load costs them 6 and 8.
SB5F_FREE_IDLE = {SB5F_UUID_5090: 578, SB5F_UUID_3080_A: 1101, SB5F_UUID_3080_B: 1111}
SB5F_FREE_LOAD = {SB5F_UUID_5090: 474, SB5F_UUID_3080_A: 1095, SB5F_UUID_3080_B: 1103}

#: D.log:360-362 "KV pool sizing: available_bytes=... -> max_total_num_tokens",
#: and D.log:367 for the vector the runtime installed from them.
SB5F_PROFILED = {SB5F_UUID_5090: 358120, SB5F_UUID_3080_A: 153848, SB5F_UUID_3080_B: 172832}
SB5F_VECTOR = [17, 7, 8]
SB5F_CELL = 32768
#: D.log:380-382 "EFFECTIVE max_total_num_tokens 674080".
SB5F_WORLD_POOL = 674080

SB5F_CARDS = (
    dict(card_uuid=SB5F_UUID_5090, name="NVIDIA GeForce RTX 5090", world_rank=0,
         nvml_total_mib=32607, budget_mib=29352, dormant_other_mib=1334,
         driver_reserved_mib=518),
    dict(card_uuid=SB5F_UUID_3080_A, name="NVIDIA GeForce RTX 3080", world_rank=1,
         nvml_total_mib=20480, budget_mib=18136, dormant_other_mib=910,
         driver_reserved_mib=425),
    dict(card_uuid=SB5F_UUID_3080_B, name="NVIDIA GeForce RTX 3080", world_rank=2,
         nvml_total_mib=20480, budget_mib=18120, dormant_other_mib=930,
         driver_reserved_mib=425),
)


class _Card:
    """The three attributes ``solve_corridor_budgets`` reads off a launcher
    ``Card`` (``launcher.py:951-962``).  Duck-typed on purpose: this file must
    not need NVML, torch or the launcher's import graph to run."""

    def __init__(self, nvml_index, uuid, name, total_mib, reserved_mib):
        self.nvml_index = nvml_index
        self.uuid = uuid
        self.name = name
        self.total_mib = total_mib
        self.reserved_mib = reserved_mib


def sb5f_sample(**over) -> cb.CorridorSample:
    cards = tuple(
        cb.SampleCard(
            free_idle_mib=SB5F_FREE_IDLE[c["card_uuid"]],
            free_load_mib=SB5F_FREE_LOAD[c["card_uuid"]],
            profiled_tokens=SB5F_PROFILED[c["card_uuid"]],
            **c,
        )
        for c in SB5F_CARDS
    )
    kw = dict(
        provenance="boot weg2sb5f 4f762260ba (test fixture)",
        cell_size=SB5F_CELL,
        token_vector=tuple(SB5F_VECTOR),
        cards=cards,
        budget_return_ratio=None,
        budget_return_provenance="",
    )
    kw.update(over)
    return cb.CorridorSample(**kw)


def sb5f_boot():
    """This boot's live inputs: the cards, the budgets ``budgets_from_dc``
    produced, and the dormant image it produced them from."""
    cards = [
        _Card(1, SB5F_UUID_5090, "NVIDIA GeForce RTX 5090", 32607, 518),
        _Card(0, SB5F_UUID_3080_A, "NVIDIA GeForce RTX 3080", 20480, 425),
        _Card(2, SB5F_UUID_3080_B, "NVIDIA GeForce RTX 3080", 20480, 425),
    ]
    budgets = [29352, 18136, 18120]
    dormant = {SB5F_UUID_5090: 1334, SB5F_UUID_3080_A: 910, SB5F_UUID_3080_B: 930}
    return cards, budgets, dormant


def line_for(solve, uuid):
    """THE VERDICT LINE for one card.

    #1257c: a card may now emit a SECOND line -- the ``INSTALLED`` provenance
    line that names what the cut cost -- so "exactly one line per card" became
    "exactly one VERDICT line per card". The count is still asserted, because
    two verdict lines for one card is still the mispairing this assertion was
    written to catch.
    """
    hits = [
        ln for ln in solve.lines
        if f"card={uuid} " in ln and " verdict=" in ln
    ]
    assert len(hits) == 1, (
        f"expected exactly one verdict line for {uuid}, got {len(hits)}"
    )
    return hits[0]


def installed_line_for(solve, uuid):
    """The ``INSTALLED`` provenance line for one card, or ``None`` (#1257c)."""
    hits = [
        ln for ln in solve.lines
        if f"card={uuid} INSTALLED " in ln
    ]
    return hits[0] if hits else None


def field(line, key):
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    raise AssertionError(f"{key}= missing from {line!r}")


# --------------------------------------------------------------------------


class TheInstrumentReproducesTheMetal(CustomTestCase):
    """Before any verdict: does the arithmetic reproduce what was measured?"""

    def test_predicted_free_equals_the_sampler_free_under_load(self):
        """THE CALIBRATION. At the sample's OWN budgets the prediction must be
        the sampler's measured load column, per card -- 474 / 1095 / 1103.
        This is what separates a measurement from a model."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        for card, b in zip(cards, budgets):
            sc = s.by_uuid[card.uuid]
            self.assertEqual(
                cb._predicted_free_mib(card.total_mib, b, dormant[card.uuid],
                                       card.reserved_mib, sc),
                SB5F_FREE_LOAD[card.uuid],
                f"{card.name} {card.uuid}: prediction diverges from the sampler",
            )

    def test_the_two_derived_terms_are_the_measured_ones(self):
        s = sb5f_sample()
        self.assertEqual(s.by_uuid[SB5F_UUID_5090].load_transient_mib, 104)
        self.assertEqual(s.by_uuid[SB5F_UUID_3080_A].load_transient_mib, 6)
        self.assertEqual(s.by_uuid[SB5F_UUID_3080_B].load_transient_mib, 8)
        # 32607 - 29352 - 1334 - 518 = 1403 booked, 578 measured: 825 MiB of
        # D's awake consumption that the budget line does not book.
        self.assertEqual(s.by_uuid[SB5F_UUID_5090].awake_residue_mib, 825)
        # Negative on the 3080s: the min-reduced pool leaves rows unallocated
        # there, so they spend LESS than their budget.
        self.assertEqual(s.by_uuid[SB5F_UUID_3080_A].awake_residue_mib, -92)
        self.assertEqual(s.by_uuid[SB5F_UUID_3080_B].awake_residue_mib, -106)

    def test_one_mib_of_budget_is_thirty_two_world_bytes_worth_of_tokens(self):
        self.assertEqual(sb5f_sample().tokens_per_mib, 32)

    def test_predicted_world_pool_equals_the_pool_D_actually_served(self):
        """D.log:380 EFFECTIVE max_total_num_tokens 674080, reproduced from the
        three profiled capacities and the installed vector."""
        sb5f_sample()
        caps = [SB5F_PROFILED[c["card_uuid"]] for c in SB5F_CARDS]
        pool, binder = cb._world_pool(caps, SB5F_VECTOR)
        self.assertEqual(pool, SB5F_WORLD_POOL)
        self.assertEqual(binder, 0, "on sb5f the 5090 is the binding rank")

    def test_the_resolved_vector_is_the_one_the_runtime_installed(self):
        """The launcher ships NO vector, so the runtime installs
        partition_units(64, [P_r]) gcd-reduced. On sb5f that was [17, 7, 8]
        (D.log:367) -- reproduced through the runtime's OWN helper, never a
        local reimplementation."""
        try:
            from sglang.srt.distributed.utils import partition_units
        except Exception as exc:  # pragma: no cover - torch-less desk
            self.skipTest(f"partition_units not importable: {exc}")
        import math as _m
        caps = [SB5F_PROFILED[c["card_uuid"]] for c in SB5F_CARDS]
        units = list(partition_units(64, caps))
        g = _m.gcd(*units)
        self.assertEqual([u // g for u in units], SB5F_VECTOR)
        # (pool, why): the reason half is load-bearing -- an unpriceable pool
        # must REFUSE a cut rather than fall back to the fixed-vector check
        # (M6), so the helper has to say why it could not price it.
        pool, why = cb._resolved_world_pool(caps)
        self.assertEqual(pool, SB5F_WORLD_POOL)
        self.assertIsNone(why, "a priceable pool carries no refusal reason")


def _law_floors(cards, mib=1024, group="D"):
    """An ACTUATING floor at the stated law, for every card.

    #1257c. The subject of this file is the CUT ARITHMETIC, and that is
    unchanged: it still grades a predicted free against a floor and still
    refuses where the world pool would shrink. What #1257c changed is WHERE
    the floor comes from and whether an UNPRICED one may actuate at all -- so
    these cases hand the pass a floor somebody priced, exactly as a boot with
    a measured transient would, and the new gate gets its own class below.
    """
    from sglang.srt.managers.corridor_guard import CorridorFloor

    return {
        c.uuid: CorridorFloor(
            card_uuid=c.uuid,
            group=group,
            transient_mib=int(mib),
            reserve_mib=0,
            source="MEASURED-" + group,
            provenance="test fixture: the stated law, priced",
        )
        for c in cards
    }


def _solve(cards, budgets, dormant, sample, why=None, group="D", floors=None):
    return cb.solve_corridor_budgets(
        cards, budgets, dormant, sample, why,
        floors=_law_floors(cards, group=group) if floors is None else floors,
        group=group,
    )


class TheCorridorIsCheckedAtAll(CustomTestCase):
    """RED-FIRST: at 80de2d31d1 nothing compares the budget line against the
    sampler, so none of these lines exists and none of these verdicts is
    reachable."""

    def test_sb5f_breaches_on_the_5090_and_holds_on_both_3080s(self):
        cards, budgets, dormant = sb5f_boot()
        solve = _solve(cards, budgets, dormant, sb5f_sample())
        self.assertIsNone(solve.unpriced_reason)
        self.assertEqual(field(line_for(solve, SB5F_UUID_5090), "verdict"),
                         "REFUSED-WOULD-BIND")
        self.assertEqual(field(line_for(solve, SB5F_UUID_3080_A), "verdict"),
                         "SATISFIED")
        self.assertEqual(field(line_for(solve, SB5F_UUID_3080_B), "verdict"),
                         "SATISFIED")

    def test_the_line_is_grepable_and_carries_every_required_field(self):
        cards, budgets, dormant = sb5f_boot()
        solve = _solve(cards, budgets, dormant, sb5f_sample())
        line = line_for(solve, SB5F_UUID_5090)
        self.assertTrue(line.startswith("WEG2-BUDGET corridor-constrained card="))
        self.assertEqual(field(line, "budget_mib"), "29352->29352")
        self.assertEqual(field(line, "predicted_free_mib"), "474")
        self.assertEqual(field(line, "law"), "1024")
        self.assertEqual(field(line, "world_pool"), f"{SB5F_WORLD_POOL}->{SB5F_WORLD_POOL}")
        self.assertEqual(field(line, "binder"), "0")
        # one line per card, once -- plus exactly one group summary
        self.assertEqual(len(solve.lines), len(cards) + 1)
        self.assertIn("SOURCE=boot weg2sb5f", solve.lines[-1])

    def test_the_refusal_names_the_numbers_it_refuses(self):
        cards, budgets, dormant = sb5f_boot()
        solve = _solve(cards, budgets, dormant, sb5f_sample())
        line = line_for(solve, SB5F_UUID_5090)
        self.assertIn(cb.WOULD_BIND_NAME, line)
        self.assertEqual(field(line, "shortfall_mib"), "550")
        self.assertEqual(field(line, "cut_mib"), "552")
        self.assertEqual(field(line, "would_be_free_mib"), "1026")
        self.assertIn("640832", line, "the world pool the cut would cost")

    def test_the_law_boundary_is_inclusive(self):
        """1024 exactly is SATISFIED; 1023 is not. The band is 819-1229 and the
        verdict constant is 1024, so >= is the rule."""
        cards, budgets, dormant = sb5f_boot()
        for delta, want in ((0, "SATISFIED"), (1, "REFUSED-WOULD-BIND")):
            s = sb5f_sample()
            sc = s.by_uuid[SB5F_UUID_5090]
            # At the sample's own budgets the prediction reduces exactly to the
            # sampler's load column, so move BOTH measured columns together and
            # keep the 104 MiB transient intact.
            tuned = replace(sc, free_load_mib=1024 - delta,
                            free_idle_mib=1024 - delta + sc.load_transient_mib)
            s2 = replace(s, cards=(tuned,) + s.cards[1:])
            solve = _solve(cards, budgets, dormant, s2)
            self.assertEqual(field(line_for(solve, SB5F_UUID_5090), "verdict"), want,
                             f"predicted_free {1024 - delta} vs law 1024")


class MutantsOnTheDangerDirection(CustomTestCase):

    def test_mutant_1_lowering_the_binder_shrinks_the_world_pool(self):
        """M1 (a): the harm the guard exists to stop, priced. Removing the
        guard would ship the 552 MiB cut and cost 33,248 world tokens."""
        s = sb5f_sample()
        caps0 = cb._capacities([29352, 18136, 18120], list(s.cards), s.tokens_per_mib)
        caps1 = cb._capacities([29352 - 552, 18136, 18120], list(s.cards), s.tokens_per_mib)
        p0, b0 = cb._world_pool(caps0, SB5F_VECTOR)
        p1, b1 = cb._world_pool(caps1, SB5F_VECTOR)
        self.assertEqual((p0, b0), (674080, 0))
        self.assertEqual((p1, b1), (640832, 0))
        self.assertLess(p1, p0)
        # ...and the shipped solver did NOT take it.
        cards, budgets, dormant = sb5f_boot()
        solve = _solve(cards, budgets, dormant, s)
        self.assertEqual(list(solve.budgets), budgets, "budgets must be byte-identical")
        self.assertEqual(solve.world_pool_before, solve.world_pool_after)

    def test_mutant_1b_a_non_binding_card_is_cut_not_refused(self):
        """M1 (b): the other half. A blanket refusal would pass M1 by accident,
        so give the breaching card so much spare capacity that it cannot bind,
        and require the cut to be APPLIED. Same fixture, one term moved: the
        5090's profiled capacity is raised until its unit is no longer the min."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        big = replace(s.by_uuid[SB5F_UUID_5090], profiled_tokens=600000)
        s2 = replace(s, cards=(big,) + s.cards[1:])
        solve = _solve(cards, budgets, dormant, s2)
        line = line_for(solve, SB5F_UUID_5090)
        self.assertEqual(field(line, "verdict"), "APPLIED")
        self.assertEqual(field(line, "budget_mib"), "29352->28800")
        self.assertEqual(field(line, "predicted_free_mib"), "1026")
        self.assertEqual(solve.world_pool_before, solve.world_pool_after,
                         "a cut is only taken where the pool does not move")
        self.assertEqual(list(solve.budgets), [28800, 18136, 18120])

    def test_mutant_2_free_column_read_as_total_minus_used(self):
        """M2: read the free column as ``total - used`` -- the solver dropping
        the driver carve-out from its OWN terms while the sample stays as
        measured. +518 MiB of fiction on the 5090: 474 reads as 992.

        Sharpened by the first remote run (2026-09-09, cachyllama, junit): the
        naive mutation (carve zeroed on BOTH sides) is SELF-COMPENSATING --
        ``awake_residue`` is derived from the same sample, absorbs the 518 and
        the prediction stays 474. That is a calibration property worth its own
        assertion, and it means the dangerous mutant is precisely the
        asymmetric one: a live boot whose solver reads total-used against a
        sample that was measured honestly."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        sc = s.by_uuid[SB5F_UUID_5090]
        honest_free = cb._predicted_free_mib(32607, 29352, 1334, 518, sc)
        self.assertEqual(honest_free, 474)
        # THE MUTANT: solver-side carve dropped, sample untouched.
        mutant_free = cb._predicted_free_mib(32607, 29352, 1334, 0, sc)
        self.assertEqual(mutant_free, honest_free + 518)
        # THE CALIBRATION PROPERTY (measured on the first remote run): zeroing
        # the carve on both sides cancels through the derived residue, so a
        # consistently carve-blind instrument still reproduces the sampler at
        # the sample's own budget -- the error would only open up as budgets
        # move. Pinned so nobody reads it as the mutant above being harmless.
        blind = replace(sc, driver_reserved_mib=0)
        self.assertEqual(blind.awake_residue_mib, sc.awake_residue_mib + 518)
        self.assertEqual(cb._predicted_free_mib(32607, 29352, 1334, 0, blind), 474)
        # The shipped solve carries the measured carve and stays at 474.
        solve = _solve(cards, budgets, dormant, s)
        self.assertEqual(field(line_for(solve, SB5F_UUID_5090), "predicted_free_mib"),
                         "474")
        self.assertIn("carve=518", line_for(solve, SB5F_UUID_5090))

    def test_mutant_3_the_bare_formula_says_satisfied_on_a_breaching_card(self):
        """M3, the load-bearing one. The brief's formula without the measured
        awake residue predicts 1299 MiB free on a card the sampler measured at
        474 -- SATISFIED on paper, 550 MiB below the law on metal. That is the
        indicator law's exact failure mode and the reason the term exists."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        sc = s.by_uuid[SB5F_UUID_5090]
        bare = replace(sc, free_idle_mib=32607 - 29352 - 1334 - 518)  # residue -> 0
        self.assertEqual(bare.awake_residue_mib, 0)
        self.assertEqual(
            cb._predicted_free_mib(32607, 29352, 1334, 518, bare) + bare.load_transient_mib,
            1403,
        )
        s_mut = replace(s, cards=(replace(bare, free_load_mib=bare.free_idle_mib - 104),)
                        + s.cards[1:])
        mut = _solve(cards, budgets, dormant, s_mut)
        self.assertEqual(field(line_for(mut, SB5F_UUID_5090), "verdict"), "SATISFIED",
                         "premise: without the residue term the card reads clean")
        self.assertEqual(field(line_for(mut, SB5F_UUID_5090), "predicted_free_mib"), "1299")
        honest = _solve(cards, budgets, dormant, s)
        self.assertEqual(field(line_for(honest, SB5F_UUID_5090), "verdict"),
                         "REFUSED-WOULD-BIND")
        self.assertIn("awake_residue=825", line_for(honest, SB5F_UUID_5090))

    def test_mutant_4_dropping_the_load_transient_overstates_every_card(self):
        """M4: price the corridor at IDLE. The law is about the free column
        under the awake group's LOAD, and load costs the 5090 104 MiB and the
        3080s 6-8. Without the term the 5090 reads 578, still a breach -- but
        the 3080 margins inflate and the instrument stops reproducing the
        sampler, which is what makes any later SATISFIED unreadable."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        idle_only = replace(s, cards=tuple(replace(c, free_load_mib=c.free_idle_mib)
                                           for c in s.cards))
        for c in idle_only.cards:
            self.assertEqual(c.load_transient_mib, 0, "premise: transient mutated out")
        mut = _solve(cards, budgets, dormant, idle_only)
        self.assertEqual(field(line_for(mut, SB5F_UUID_5090), "predicted_free_mib"), "578")
        self.assertEqual(field(line_for(mut, SB5F_UUID_3080_A), "predicted_free_mib"), "1101")
        honest = _solve(cards, budgets, dormant, s)
        for uuid in (SB5F_UUID_5090, SB5F_UUID_3080_A, SB5F_UUID_3080_B):
            self.assertEqual(
                int(field(line_for(honest, uuid), "predicted_free_mib")),
                SB5F_FREE_LOAD[uuid],
                "only the honest instrument reproduces the sampler",
            )

    def test_mutant_6_an_unpriceable_resolved_pool_refuses_the_cut(self):
        """M6: the silent-degradation route to the danger direction.

        ``_resolved_world_pool`` returns None when the runtime's own
        ``partition_units`` is not importable in this process. The first draft
        then evaluated ``shrinks_resolved`` as False and applied the cut under
        the FIXED-vector check alone -- but the runtime RE-SOLVES the vector at
        boot, and under a re-solved vector every rank is near-binding, so a cut
        the fixed check calls free can still shrink the pool actually served.

        GREEN-THEN-RED in one test, on the same fixture: with the pool
        priceable the non-binding breacher is APPLIED (that is M1b), and with
        the pool unpriceable the SAME input must refuse and leave the budget
        byte-identical."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        big = replace(s.by_uuid[SB5F_UUID_5090], profiled_tokens=600000)
        s2 = replace(s, cards=(big,) + s.cards[1:])

        green = _solve(cards, budgets, dormant, s2)
        self.assertEqual(field(line_for(green, SB5F_UUID_5090), "verdict"),
                         "APPLIED", "premise: this cut IS taken when priceable")
        self.assertEqual(list(green.budgets), [28800, 18136, 18120])

        real = cb._resolved_world_pool
        cb._resolved_world_pool = lambda caps: (
            None, "simulated: the runtime's vector solver is not importable"
        )
        try:
            red = _solve(cards, budgets, dormant, s2)
        finally:
            cb._resolved_world_pool = real
        line = line_for(red, SB5F_UUID_5090)
        self.assertEqual(field(line, "verdict"), "REFUSED-UNPRICED-POOL")
        self.assertEqual(field(line, "budget_mib"), "29352->29352")
        self.assertIn(cb.UNPRICED_NAME, line)
        self.assertIn("not importable", line, "the refusal names WHY")
        self.assertEqual(list(red.budgets), budgets,
                         "an unpriceable danger direction leaves every budget "
                         "byte-identical -- it does not fall back to the "
                         "fixed-vector check")

    def test_mutant_7_a_shifted_card_order_is_unpriced_not_mispaired(self):
        """M7: the sample's rows pair with this boot's cards by uuid, but the
        token vector is POSITIONAL. NVML enumeration order can shift between
        boots on this rig, and a shifted pairing does not fail loudly -- it
        prices a perfectly plausible pool and binder for the wrong cards.

        The mutation is the shift itself: the same three measured rows, with
        the world_rank column of two of them swapped, as a boot that
        re-enumerated would produce. The pass must refuse, not re-sort: if the
        two boots disagree about which card is rank 0, the residue and
        transient measured on the other boot are not this boot's either."""
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        shifted = replace(
            s,
            cards=(
                replace(s.cards[0], world_rank=2),
                s.cards[1],
                replace(s.cards[2], world_rank=0),
            ),
        )
        solve = _solve(cards, budgets, dormant, shifted)
        self.assertEqual(list(solve.budgets), budgets,
                         "the budget must stand byte-identical")
        self.assertEqual(len(solve.lines), 1)
        self.assertIn(cb.UNPRICED_NAME, solve.lines[0])
        self.assertIn("world_rank", solve.lines[0])
        self.assertIn("ordinal 0", solve.lines[0], "the line names the ordinal")
        self.assertIsNotNone(solve.unpriced_reason)
        # ...and the honest ordering still prices, so this is the SHIFT being
        # detected and not the check refusing everything.
        ok = _solve(cards, budgets, dormant, s)
        self.assertIsNone(ok.unpriced_reason)

    def test_mutant_5_missing_inputs_must_not_produce_a_default_budget(self):
        """M5: the silent-default shape. No sample, an unreadable one, an
        incomplete one and one that does not cover this boot's cards must ALL
        leave the budgets byte-identical and say W54 -- never quietly ship a
        number nobody measured, and never let the absence of a corridor line
        read as a satisfied corridor."""
        cards, budgets, dormant = sb5f_boot()
        cases = ["no sample supplied at all"]
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "nope.json")
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as fh:
                fh.write("{not json")
            short = os.path.join(d, "short.json")
            with open(short, "w") as fh:
                # genuinely incomplete: the card row has no profiled_tokens,
                # so the world-pool side of the constraint cannot be priced
                card = dict(SB5F_CARDS[0], free_idle_mib=578, free_load_mib=474)
                json.dump({"cell_size": 32768, "token_vector": [1],
                           "cards": [card]}, fh)
            for path in (missing, bad, short):
                sample, why = cb.load_sample(path)
                self.assertIsNone(sample, f"{path} must not load")
                self.assertTrue(why)
                cases.append(why)

        for why in cases:
            sample = None
            solve = _solve(cards, budgets, dormant, sample, why)
            self.assertEqual(list(solve.budgets), budgets,
                             "the budget must stand byte-identical")
            self.assertEqual(len(solve.lines), 1)
            self.assertIn(cb.UNPRICED_NAME, solve.lines[0])
            self.assertIn("REFUSED TO PRICE", solve.lines[0])
            self.assertIsNotNone(solve.unpriced_reason)

    def test_mutant_5b_a_sample_that_misses_a_card_is_unpriced_not_partial(self):
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        partial = replace(s, cards=s.cards[:2], token_vector=(17, 7))
        solve = _solve(cards, budgets, dormant, partial)
        self.assertEqual(list(solve.budgets), budgets)
        self.assertIn(cb.UNPRICED_NAME, solve.lines[0])
        self.assertIn("no row for card", solve.lines[0])


class TheReturnRatioIsNamedNotAssumedSilently(CustomTestCase):
    """#631 (htsglang aeac561711) measured -1200 MiB of budget returning +382
    MiB of free on the Weg-1 form. 1.0 is therefore an optimistic bound and the
    line has to say so, or a cut computed from it reads as sufficient."""

    def test_the_assumption_is_printed_with_its_counter_evidence(self):
        cards, budgets, dormant = sb5f_boot()
        solve = _solve(cards, budgets, dormant, sb5f_sample())
        self.assertIn("return_ratio=1.0:ASSUMED", solve.lines[-1])
        self.assertIn("0.318", solve.lines[-1])

    def test_a_measured_ratio_is_used_and_labelled_measured(self):
        cards, budgets, dormant = sb5f_boot()
        s = replace(sb5f_sample(), budget_return_ratio=0.5,
                    budget_return_provenance="boot weg2xx two-point pair")
        solve = _solve(cards, budgets, dormant, s)
        self.assertIn("return_ratio=0.5:MEASURED", solve.lines[-1])
        # 550 MiB of free needed at a 0.5 return = 1100 MiB of budget.
        self.assertEqual(field(line_for(solve, SB5F_UUID_5090), "cut_mib"), "1104")

    def test_a_ratio_outside_zero_to_one_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            body = {
                "cell_size": 32768, "token_vector": SB5F_VECTOR,
                "budget_return_ratio": 1.4,
                "cards": [dict(c, free_idle_mib=SB5F_FREE_IDLE[c["card_uuid"]],
                               free_load_mib=SB5F_FREE_LOAD[c["card_uuid"]],
                               profiled_tokens=SB5F_PROFILED[c["card_uuid"]])
                          for c in SB5F_CARDS],
            }
            with open(p, "w") as fh:
                json.dump(body, fh)
            sample, why = cb.load_sample(p)
        self.assertIsNone(sample)
        self.assertIn("budget_return_ratio", why)


class TheSampleOnDiskLoads(CustomTestCase):
    """The shipped default path is evidence-tree-bound (it is a measurement of
    THIS rig, not repo content), so this test is skipped where the tree is not
    mounted rather than faked."""

    def test_the_default_sample_reproduces_the_fixture(self):
        if not os.path.exists(cb.DEFAULT_SAMPLE_PATH):
            self.skipTest(f"{cb.DEFAULT_SAMPLE_PATH} not present (evidence tree)")
        s, why = cb.load_sample(cb.DEFAULT_SAMPLE_PATH)
        self.assertIsNone(why)
        self.assertEqual(list(s.token_vector), SB5F_VECTOR)
        self.assertEqual(s.cell_size, SB5F_CELL)
        for c in s.cards:
            self.assertEqual(c.free_load_mib, SB5F_FREE_LOAD[c.card_uuid])
            self.assertEqual(c.profiled_tokens, SB5F_PROFILED[c.card_uuid])
        caps = [c.profiled_tokens for c in s.cards]
        self.assertEqual(cb._world_pool(caps, list(s.token_vector))[0], SB5F_WORLD_POOL)


class TheLauncherWiresItOnDOnly(CustomTestCase):
    """The constraint is the LAW for the Weg-2 form, so group D's solve is
    called with it on -- and group P's is not (P sleeps while D serves; the
    corridor is measured under the AWAKE group's load and P has no such
    sample). Read off the source so this cannot drift into a docstring."""

    @staticmethod
    def _calls():
        from sglang.srt.weg2 import launcher
        tree = ast.parse(inspect.getsource(launcher))
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "budgets_from_dc":
                kw = {k.arg: k for k in node.keywords}
                label = None
                if len(node.args) >= 4 and isinstance(node.args[3], ast.Constant):
                    label = node.args[3].value
                on = "corridor_constrain" in kw and getattr(
                    kw["corridor_constrain"].value, "value", False) is True
                out.append((label, on))
        return out

    def test_every_d_call_site_constrains_and_no_p_call_site_does(self):
        calls = self._calls()
        self.assertGreaterEqual(len(calls), 3, "P, D(dry), D")
        for label, on in calls:
            self.assertIsNotNone(label, "every call names its group")
            if label.startswith("D"):
                self.assertTrue(on, f"group {label} must be corridor-constrained")
            else:
                self.assertFalse(on, f"group {label} must not be")

    def test_the_constraint_is_not_a_boolean_knob_on_the_cli(self):
        """Only the INPUT is selectable. A ``--no-corridor-constraint`` style
        flag would make the law optional, which it is not."""
        from sglang.srt.weg2 import launcher
        src = inspect.getsource(launcher)
        for forbidden in ("--no-corridor", "--disable-corridor", "--corridor-constraint"):
            self.assertNotIn(forbidden, src)
        self.assertIn("--corridor-budget-sample", src)


class TheBudgetProducerStaysSingular(CustomTestCase):
    """UPSTREAM-MINIMAL: one writer of a budget number. The pass may only
    lower or leave alone, and it lives inside ``budgets_from_dc`` rather than
    beside it."""

    def test_the_pass_never_raises_a_budget(self):
        cards, budgets, dormant = sb5f_boot()
        for scale in (0.8, 1.0, 1.2):
            b = [int(x * scale) // 8 * 8 for x in budgets]
            solve = _solve(cards, b, dormant, sb5f_sample())
            for before, after in zip(b, solve.budgets):
                self.assertLessEqual(after, before)

    def test_cuts_keep_the_eight_mib_alignment(self):
        cards, budgets, dormant = sb5f_boot()
        s = sb5f_sample()
        big = replace(s.by_uuid[SB5F_UUID_5090], profiled_tokens=600000)
        solve = _solve(cards, budgets, dormant,
                                          replace(s, cards=(big,) + s.cards[1:]))
        for b in solve.budgets:
            self.assertEqual(b % 8, 0)

    def test_the_pass_holds_no_private_copy_of_the_law(self):
        """STRENGTHENED by #1257c: not one assignment -- ZERO.

        This used to allow exactly one ``CORRIDOR_LAW_MIB = 1024`` here and
        pinned its value. That was already a private copy of a number
        ``managers.corridor_guard`` declares and whose own comment forbids
        repeating; #1257c removed it in favour of an import, so the assignment
        count is now 0 and the identity is asserted instead.
        """
        from sglang.srt.managers import corridor_guard as cg

        src = inspect.getsource(cb)
        tree = ast.parse(textwrap.dedent(src))
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "CORRIDOR_LAW_MIB" for t in n.targets)]
        self.assertEqual(assigns, [], "no private copy of the law may return")
        self.assertIs(cb.CORRIDOR_LAW_MIB, cg.CORRIDOR_LAW_MIB)
        self.assertEqual(cb.CORRIDOR_LAW_MIB, 1024)

    def test_an_unpriced_floor_is_a_verdict_and_never_a_cut(self):
        """#1257c, the user decision, on THIS file's own sb5f fixture.

        The 5090 predicts 474 MiB free -- 345 below the band floor the
        fallback grades against. With no measured transient and no user
        reserve, nothing priced that floor, so the pass says so by name and
        every budget stands byte-identical. This is the case the whole rest of
        the file's fixtures previously took for granted.
        """
        cards, budgets, dormant = sb5f_boot()
        solve = cb.solve_corridor_budgets(
            cards, budgets, dormant, sb5f_sample(), group="D"
        )
        self.assertEqual(list(solve.budgets), list(budgets))
        self.assertFalse(solve.changed)
        text = "\n".join(solve.lines)
        self.assertIn(cb.UNMEASURED_FLOOR_NAME, text)
        self.assertIn("REFUSED-UNMEASURED-FLOOR", text)
        self.assertNotIn(" INSTALLED ", text)


if __name__ == "__main__":
    unittest.main()
