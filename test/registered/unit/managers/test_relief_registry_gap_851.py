"""#851/#813: a relief post the guard ladder cannot spend must not read as "nothing".

W22 (boot_w22_0824_0656.log) refused every tp_to_pp seam with::

    corridor gate refused the seam staging: want 3248 MiB, free 2420 -> 2420
    MiB, reclaimed 0 MiB from [nothing]

39 times. "[nothing]" is not true. ``FundingAuthority`` had declared a third
post -- ``kv-slack``, the KV pool's own unoccupied backing -- and the guard
ladder simply cannot see it. ``funding_authority.py`` says so itself, in a
comment written before this window:

    THE POST THE GUARD LADDER CANNOT SEE. It is declared here precisely
    because it is absent there: the rung pays before the gate, so its bytes
    never enter the ladder's provider list and a refusal has never once been
    able to name it.

THE ASYMMETRY, counted:
  * ``authority_from_seam_snapshot`` declares THREE posts: allocator-cache,
    draft-weights, kv-slack.
  * ``phase_flip_spill`` calls ``guard.register`` exactly TWICE, for
    allocator-cache and draft-weights. Two, in the whole tree.
So when the allocator cache is dry -- which is the normal state at a seam,
because the seam is what drained it -- the ladder's ``used`` list is empty and
the refusal reports "nothing" while a declared post sits unreachable beside it.

WHY THIS IS A DEFECT AND NOT A WORDING NIT. The refusal line is the ONLY
artifact a later reader gets. "reclaimed 0 from [nothing]" says "this rig had
no memory to give", which sends the next investigator to capacity planning.
The truth is "this rig had a post it could not reach", which sends them to the
registry. W22 cost a window partly to that distinction: the same log shows the
registry DELIVERING elsewhere (``reclaimed 1210 MiB from [allocator-cache]``
x12, 1124 x7, 1398 x4), which reads as a healthy registry and hid the gap.

ONE REMEDY IS FORBIDDEN, and this file originally asserted it. Registering
kv-slack with the ladder is WRONG: ``phase_flip_runtime.py:8290-8294`` states
"No KV provider is registered with the guard at all, BY DESIGN (the cap is a
group decision and the ladder is rank-local)". A rank-local ladder must not
spend a group-decided cap, and it would spend it a SECOND time after the rung
already paid. The corrected red assertion below therefore pins the property
rather than a mechanism: a refusal must not describe its sources as "nothing"
while a declared post holds credit. That is satisfied by naming the funder in
the refusal, and NOT satisfiable by making the ladder spend it.

Hermetic: pure registry inspection, no scheduler, no CUDA.
"""

import ast
import inspect
import unittest

from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.funding_authority import MIB, authority_from_seam_snapshot


def _declared_post_names():
    """Post names the funding authority declares, from the shipped builder."""
    auth = authority_from_seam_snapshot(
        rank=0,
        allocator_cache_bytes=1 << 20,
        draft_available_bytes=1 << 20,
        kv_slack_rows=1024,
        row_bytes=4096,
        kv_granule_rows=256,
    )
    return {p.name for p in auth._posts}


def _ladder_registered_names():
    """Names passed to ``guard.register`` in the shipped registration site.

    Read out of the source with ``ast`` rather than restated, so this cannot
    drift from the code it measures -- the same pin style the #850 marker table
    uses. A literal list here would keep passing after someone adds a third
    provider, which is exactly the event this file exists to notice.
    """
    tree = ast.parse(inspect.getsource(phase_flip_spill))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "register"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


class TestTheRegistryAsymmetry(unittest.TestCase):
    def test_the_authority_declares_kv_slack(self):
        declared = _declared_post_names()
        self.assertIn("kv-slack", declared)
        self.assertEqual(
            declared, {"allocator-cache", "draft-weights", "kv-slack"}
        )

    def test_the_pin_actually_finds_the_registrations(self):
        """CAN-FAIL PROOF. An empty scrape would make the next test vacuous."""
        registered = _ladder_registered_names()
        self.assertIn("allocator-cache", registered)
        self.assertIn("draft-weights", registered)

    def test_the_ladder_registers_fewer_posts_than_are_declared(self):
        """CHARACTERISATION of today: passes NOW, and names the gap by size."""
        declared = _declared_post_names()
        registered = _ladder_registered_names()
        self.assertTrue(declared - registered)
        self.assertEqual(declared - registered, {"kv-slack"})

    def test_kv_slack_must_NOT_be_registered_with_the_ladder(self):
        """THE ASSERTION THIS FILE ORIGINALLY GOT BACKWARDS -- corrected, and
        kept as a guard so nobody "fixes" #813 the dangerous way.

        The first version of this file asserted ``declared - registered ==
        set()``, i.e. that the ladder must register every declared post. That
        is WRONG, and the tree says so at
        ``phase_flip_runtime.py:8290-8294``:

            No KV provider is registered with the guard at all, BY DESIGN (the
            cap is a group decision and the ladder is rank-local), so the
            rung's bytes arrive as `kv_freed` BEFORE the probe and can never
            appear in that list.

        Registering kv-slack with the ladder would let a RANK-LOCAL ladder
        spend a GROUP-DECIDED cap, and would spend it a second time after the
        rung already paid it. The gap is real; that remedy is not. #813 closes
        by making the refusal NAME what it could not reach -- which is what
        the plan's F4 says ("a refusal prices ALL declared posts") and what
        the corrected red test below pins.
        """
        self.assertNotIn("kv-slack", _ladder_registered_names())

    def test_a_refusal_names_the_funder_it_cannot_spend(self):
        """GREEN SINCE F4. Was the red acceptance assertion for it.

        The defect is not that the ladder cannot spend kv-slack -- it must
        not. The defect is that the refusal REPORTS "[nothing]" while a
        declared post holds credit, which reads as "this rig had no memory to
        give" and sends the next reader to capacity planning instead of to the
        registry. W22 printed that 39 times while kv-slack held 2776 MiB.

        The property, and it is fix-shape-independent: a guard whose ladder
        delivered nothing, but which can see a declared post holding credit,
        must not describe its sources as "nothing".
        """
        from sglang.srt.managers import corridor_guard as cg

        # W22's numbers: the gate wanted 3248 MiB against 2420 MiB free, with
        # an empty ladder (no provider registered), while kv-slack held credit.
        guard = cg.CorridorGuard(
            0,
            floor_mib=1255,
            probe=lambda: 2420 * MIB,
            law_floor_mib=1024,
        )
        guard.declare_offledger_funder(
            lambda want: (("kv-slack", 2560 * MIB, ""),)
        )
        res = guard.ensure_headroom(3248 * MIB, reason="seam staging tp_to_pp")
        self.assertFalse(res.ok)
        # The funder must be NAMED, with its figure.
        self.assertIn("kv-slack", res.detail)
        self.assertIn("2560", res.detail)
        # "[nothing]" LEGITIMATELY REMAINS, and this is deliberate rather than
        # a weakened assertion. That token is the LADDER's record of what it
        # actually spent, and it is true: the ladder spent nothing. The defect
        # was never the word, it was that the word stood ALONE and therefore
        # read as "this rig had no memory to give". So the property pinned
        # here is that the two facts appear TOGETHER -- a future change that
        # drops the suffix and leaves the bare list fails this, which is the
        # regression that matters.
        self.assertIn("[nothing]", res.detail)
        self.assertLess(
            res.detail.index("[nothing]"),
            res.detail.index("kv-slack"),
            "the ladder record must still come first; the funder clause explains it",
        )

    def test_an_unavailable_post_still_carries_a_reason(self):
        """The other half: a post that is reachable but EMPTY must say why.

        This one already holds, and it is the model the red test above is
        asking the unreachable post to be held to: "torch cache already
        returned" is a fact a reader can act on; "nothing" is not.
        """
        auth = authority_from_seam_snapshot(
            rank=0,
            allocator_cache_bytes=0,
            draft_available_bytes=0,
            kv_slack_rows=0,
            row_bytes=4096,
            kv_granule_rows=256,
        )
        reasons = {p.name: p.unavailable_reason for p in auth._posts}
        self.assertTrue(reasons["allocator-cache"])
        self.assertTrue(reasons["kv-slack"])


if __name__ == "__main__":
    unittest.main()
