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

FIX-SHAPE-INDEPENDENT. The red assertion below is satisfied by registering
kv-slack with the ladder, by having the rung report through the ladder, or by
making the refusal name declared-but-unreachable posts. It does not prescribe
which. It only forbids a refusal that says "nothing" while a post is declared.

Hermetic: pure registry inspection, no scheduler, no CUDA.
"""

import ast
import inspect
import unittest

from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.funding_authority import authority_from_seam_snapshot


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

    @unittest.expectedFailure
    def test_every_declared_post_is_reachable_by_the_ladder(self):
        """RED TODAY, BY DESIGN. UNEXPECTED SUCCESS when #851 closes the gap.

        A post the authority declares but the ladder cannot spend is a post
        that can never appear in a refusal's provider list -- so the refusal
        says "nothing" and blames the rig. Either the ladder learns to reach
        it, or the refusal learns to name it; this assertion does not care
        which, only that "[nothing]" stops being printed while a post exists.
        """
        declared = _declared_post_names()
        registered = _ladder_registered_names()
        self.assertEqual(
            declared - registered,
            set(),
            "declared but unreachable posts: a refusal will report [nothing] "
            "while these sit beside it",
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
