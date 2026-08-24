"""#856: RETRACT BEFORE RESET, and the reverse order must still crash.

THE LAW. Once the flip carries no KV, the new phase's device pool holds no
valid rows while the radix tree still maps prefixes to row ids. The tree must
therefore be dropped so a lookup MISSES and falls through to the host tier.

That action is already known to be fatal in the wrong order. #825 tried it and
took the instance down on all three ranks (2026-08-23), recorded verbatim in
`phase_flip_runtime`:

    cache_finished_req -> dec_lock_ref
    -> full_component.py:239  `if cur.id in skip_lock_node_ids`
    AttributeError: 'NoneType' object has no attribute 'id'

    "PARKED IS NOT UNREFERENCED. The cutover carries RESIDENT requests
     across, and each holds a `last_node` with a lock ref. `reset()` rebuilds
     the root, orphaning those nodes, so the parent walk in `dec_lock_ref` no
     longer terminates at the live root and runs off the top into None."

THIS FILE PINS THE ORDER, NOT THE OUTCOME. A test that only asserted "the
tree ended up empty" would pass against the fatal ordering. So the fatal
ordering is REPRODUCED here, hermetically, against a faithful model of the
real walk -- `dec_lock_ref`'s loop is literally
`while node != self.root_node: ... node = node.parent`
(mem_cache/hi_mamba_radix_cache.py:1610) -- and the correct ordering is shown
to survive it.

The model is deliberately minimal: nodes with a `parent`, a tree whose
`reset()` installs a NEW root object, and a release that walks parents until
it reaches the CURRENT root. That is the whole mechanism of the crash; every
other detail of the radix cache is irrelevant to it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    SeamOrderError,
    release_residents_for_cutover,
)
from sglang.test.test_utils import CustomTestCase


class _Node:
    __slots__ = ("id", "parent", "lock_ref")

    def __init__(self, node_id, parent=None):
        self.id = node_id
        self.parent = parent
        self.lock_ref = 0


class _Tree:
    """A faithful model of the two lines that matter."""

    def __init__(self):
        self.root_node = _Node("root")
        self.resets = 0

    def reset(self):
        # THE CRASH'S MECHANISM: a NEW root object. Every node still held by a
        # parked request now has a parent chain that never reaches it.
        self.root_node = _Node("root")
        self.resets += 1

    def dec_lock_ref(self, node):
        # mem_cache/hi_mamba_radix_cache.py:1610, reduced to its walk.
        while node != self.root_node:
            node.lock_ref -= 1
            node = node.parent  # runs off the top into None when orphaned
            if node is None:
                raise AttributeError("'NoneType' object has no attribute 'id'")


class _Req:
    def __init__(self, last_node):
        self.last_node = last_node
        self.released = False


def _world(n_resident=3):
    tree = _Tree()
    reqs = []
    for i in range(n_resident):
        node = _Node(f"n{i}", parent=tree.root_node)
        node.lock_ref = 1
        reqs.append(_Req(node))
    return tree, reqs


def _retract(tree):
    def _do(reqs):
        for r in reqs:
            tree.dec_lock_ref(r.last_node)
            r.released = True
        return list(reqs)

    return _do


class TestTheFatalOrderIsStillFatal(CustomTestCase):
    """The can-fail direction, and the reason this file exists.

    If this ever stops raising, the model has drifted from the crash it
    represents and every other assertion here is worthless.
    """

    def test_reset_before_retract_reproduces_the_825_crash(self):
        tree, reqs = _world()
        tree.reset()  # the #825 ordering
        with self.assertRaises(AttributeError) as caught:
            _retract(tree)(reqs)
        self.assertIn("NoneType", str(caught.exception))

    def test_the_orphaned_walk_is_what_raises_not_the_reset(self):
        tree, _reqs = _world()
        tree.reset()  # resetting alone is harmless
        self.assertEqual(tree.resets, 1)


class TestTheLawfulOrderSurvives(CustomTestCase):
    def test_retract_then_reset_releases_every_lock_and_drops_the_tree(self):
        tree, reqs = _world()
        out = release_residents_for_cutover(
            reqs, retract=_retract(tree), reset_tree=tree.reset
        )
        self.assertEqual(len(out), 3)
        self.assertTrue(all(r.released for r in reqs))
        self.assertTrue(all(r.last_node.lock_ref == 0 for r in reqs))
        self.assertEqual(tree.resets, 1)

    def test_the_helper_actually_runs_them_in_that_order(self):
        # Pinning the ORDER, not just that both happened: a helper that reset
        # first and retracted second would satisfy the assertions above on a
        # tree model that tolerated it.
        seen = []
        release_residents_for_cutover(
            [],
            retract=lambda reqs: seen.append("retract") or [],
            reset_tree=lambda: seen.append("reset"),
        )
        self.assertEqual(seen, ["retract", "reset"])

    def test_no_residents_still_resets(self):
        # An idle instance still needs the tree dropped: its prefixes name
        # rows that the new phase's pool does not hold either.
        tree, _ = _world(0)
        release_residents_for_cutover([], retract=_retract(tree), reset_tree=tree.reset)
        self.assertEqual(tree.resets, 1)


class TestNeitherStepMayBeSkipped(CustomTestCase):
    """Silently skipping one is worse than refusing."""

    def test_a_missing_reset_is_refused(self):
        tree, reqs = _world()
        with self.assertRaises(SeamOrderError):
            release_residents_for_cutover(reqs, retract=_retract(tree), reset_tree=None)

    def test_a_missing_retraction_is_refused(self):
        tree, reqs = _world()
        with self.assertRaises(SeamOrderError):
            release_residents_for_cutover(reqs, retract=None, reset_tree=tree.reset)

    def test_the_refusal_names_both_failure_modes(self):
        tree, _ = _world()
        with self.assertRaises(SeamOrderError) as caught:
            release_residents_for_cutover([], retract=None, reset_tree=tree.reset)
        msg = str(caught.exception)
        self.assertIn("orphans locked nodes", msg)
        self.assertIn("hold no KV", msg)


class TestTheSchedulerBindingRefusesRatherThanDegrades(CustomTestCase):
    """`build_cutover_release` must not hand back a half-usable pair."""

    def test_no_tree_cache_is_refused(self):
        import types

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        self.assertIsNone(build_cutover_release(types.SimpleNamespace()))

    def test_an_unresettable_tree_is_refused(self):
        # A ChunkCache has no reset(); a flip that cannot drop its tree would
        # enter the next phase naming rows that hold no KV.
        import types

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        sched = types.SimpleNamespace(tree_cache=types.SimpleNamespace())
        self.assertIsNone(build_cutover_release(sched))

    def test_a_resettable_tree_yields_both_halves(self):
        import types

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        tree = types.SimpleNamespace(reset=lambda: None)
        built = build_cutover_release(types.SimpleNamespace(tree_cache=tree))
        self.assertIsNotNone(built)
        retract, reset_tree = built
        self.assertTrue(callable(retract))
        # #856 W27-retry: the drop is NO LONGER the bare `tree.reset`. A bare
        # reset is a bookkeeping reset that orphans the tree's rows -- it
        # leaked 152 rows per cycle on metal. The binding must hand back the
        # row-RETURNING drop instead.
        self.assertTrue(callable(reset_tree))
        self.assertIsNot(reset_tree, tree.reset)

    def test_retracting_nothing_needs_no_scheduler_state(self):
        # The empty case must not reach into pools that an idle instance may
        # not have wired -- an idle flip still has to drop its tree.
        import types

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        tree = types.SimpleNamespace(reset=lambda: None)
        retract, _ = build_cutover_release(types.SimpleNamespace(tree_cache=tree))
        self.assertEqual(retract([]), [])

    def test_the_fence_already_paid_so_the_seam_does_not_copy_again(self):
        # `offload_kv=False` is the deliberate choice: that flag exists for
        # decode-disaggregation to copy retracted KV device->host, and the
        # fence has already persisted these prefixes. Copying again would pay
        # twice at the one instant the instance is blocked.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        src = inspect.getsource(build_cutover_release)
        self.assertIn("offload_kv=False", src)


if __name__ == "__main__":
    unittest.main()
