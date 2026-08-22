"""#773 for #796: make the pool census say WHO owns the unaccounted rows.

`PhaseFlipRuntime._pool_census` derives `unaccounted` as

    set(range(1, size + 1)) - free - cached

from exactly ONE allocator and ONE tree, both read off the scheduler, with
`cached` collecting only BASE-component DEVICE values. Two consequences the
census line cannot distinguish:

* a row owned by a DIFFERENT pool object is unaccounted by definition, not by
  defect -- and a second owner is known to be possible on the flip-TP stack,
  where `is_draft_pool_worker` and `is_draft_worker` diverge;
* the sum `free + cached + unaccounted == size` is a TAUTOLOGY, since the
  three sets partition the id space by construction. It corroborates nothing,
  and it was briefly cited as if it did.

The probe answers the first by naming every pool object it can reach, and
reports the SHAPE of the unaccounted set so the second is not mistaken for
evidence. The census prints `sorted(leaked)[:12]`; twelve consecutive ids at
the minimum say nothing about the other ~94000, which is how a "contiguous
block" reading survived until it was checked.

These tests drive the shipped helpers directly. No GPU, no flip, no scheduler.
"""

import logging
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)


def _probe(scheduler):
    """Bind the two real methods onto a bare object."""
    stub = SimpleNamespace(_census_scheduler=scheduler)
    stub._owner_ident = PhaseFlipRuntime._owner_ident
    stub._census_owner_probe = PhaseFlipRuntime._census_owner_probe.__get__(
        stub, SimpleNamespace
    )
    return stub


def _run(scheduler, leaked, alloc=None, tree=None):
    stub = _probe(scheduler)
    with unittest.mock.patch.object(
        logging.getLogger("sglang.srt.managers.phase_flip_runtime"), "warning"
    ) as warn:
        stub._census_owner_probe("at-arm", "pp_to_tp", alloc, tree, leaked)
    assert warn.called, "the probe must always emit"
    args = warn.call_args[0]
    return args[0] % tuple(args[1:])


class TestTheShapeOfTheUnaccountedSet(CustomTestCase):
    """`runs`/`longest_run` are what make the contiguity question answerable."""

    def test_one_contiguous_block_reports_a_single_run(self):
        text = _run(SimpleNamespace(phase_flip_stacks=None), set(range(1000, 1500)))
        self.assertIn("n=500", text)
        self.assertIn("min=1000", text)
        self.assertIn("max=1499", text)
        self.assertIn("runs=1", text)
        self.assertIn("longest_run=500", text)

    def test_scattered_rows_report_many_runs(self):
        """THE DISCRIMINATOR. Identical n and min to a block; different shape.

        A set that reads as `1000, 1001, ...` in its first twelve ids can
        still be scattered everywhere else, which is exactly the reading that
        had to be withdrawn.
        """
        scattered = set(range(1000, 1012)) | set(range(2000, 3000, 2))
        text = _run(SimpleNamespace(phase_flip_stacks=None), scattered)
        self.assertIn("min=1000", text)
        self.assertNotIn("runs=1 ", text)
        self.assertIn("longest_run=12", text)

    def test_an_empty_set_is_said_plainly(self):
        text = _run(SimpleNamespace(phase_flip_stacks=None), set())
        self.assertIn("UNACCOUNTED: n=0", text)


class TestItNamesEveryOwnerItCanReach(CustomTestCase):
    def test_the_census_pair_is_always_named(self):
        alloc = SimpleNamespace(size=448698)
        tree = SimpleNamespace()
        text = _run(SimpleNamespace(phase_flip_stacks=None), set(), alloc, tree)
        self.assertIn("CENSUS:", text)
        self.assertIn(f"alloc_id={id(alloc)}", text)
        self.assertIn("alloc_size=448698", text)
        self.assertIn(f"tree_id={id(tree)}", text)

    def test_a_second_owner_shows_a_DIFFERENT_id(self):
        """The whole point: second-owner and leak stop looking alike."""
        census_alloc, census_tree = SimpleNamespace(size=10), SimpleNamespace()
        other_alloc, other_tree = SimpleNamespace(size=10), SimpleNamespace()
        runner = SimpleNamespace(
            token_to_kv_pool_allocator=other_alloc,
            tree_cache=other_tree,
            token_to_kv_pool=SimpleNamespace(),
            is_phase_flip_tp_stack=True,
            is_draft_pool_worker=False,
        )
        scheduler = SimpleNamespace(
            phase_flip_stacks=SimpleNamespace(
                tp_worker=SimpleNamespace(model_runner=runner), draft_worker=None
            )
        )
        text = _run(scheduler, set(), census_alloc, census_tree)
        self.assertIn(f"alloc_id={id(census_alloc)}", text)
        self.assertIn(f"alloc_id={id(other_alloc)}", text)
        self.assertNotEqual(id(census_alloc), id(other_alloc))
        self.assertIn("flip_tp=True", text)

    def test_absent_stacks_are_stated_not_silent(self):
        text = _run(SimpleNamespace(phase_flip_stacks=None), set())
        self.assertIn("STACKS: None at census time", text)

    def test_a_missing_worker_is_stated_not_silent(self):
        scheduler = SimpleNamespace(
            phase_flip_stacks=SimpleNamespace(tp_worker=None, draft_worker=None)
        )
        text = _run(scheduler, set())
        self.assertIn("TP: worker None", text)
        self.assertIn("DRAFT: worker None", text)


class TestItCanNeverBreakTheFlip(CustomTestCase):
    """A census watches; it must not be able to affect what it watches."""

    def test_a_hostile_scheduler_is_swallowed_and_reported(self):
        class Boom:
            @property
            def phase_flip_stacks(self):
                raise RuntimeError("exploding handle")

        text = _run(Boom(), {1, 2, 3})
        self.assertIn("pool owner probe", text)
        self.assertIn("exploding handle", text)


if __name__ == "__main__":
    unittest.main()


class TestTheCensusActuallyCallsTheProbe(CustomTestCase):
    """The WIRING, not the helper.

    Every test above drives `_census_owner_probe` directly, so removing the
    call from `_pool_census` leaves them all green -- verified by mutation.
    That is the third time in this task that a direct-helper test failed to
    bind its own call site, so it gets its own class.
    """

    def _scheduler(self, size=32):
        import torch

        alloc = SimpleNamespace(
            size=size,
            free_pages=torch.arange(1, 11, dtype=torch.int64),
            release_pages=torch.tensor([], dtype=torch.int64),
            available_size=lambda: 10,
            _kvcache=SimpleNamespace(),
        )
        tree = SimpleNamespace(
            all_values_flatten=lambda: torch.arange(11, 21, dtype=torch.int64)
        )
        return SimpleNamespace(
            token_to_kv_pool_allocator=alloc,
            tree_cache=tree,
            running_mbs=[],
            phase_flip_stacks=None,
            tp_worker=None,
        )

    def test_pool_census_emits_the_owner_line(self):
        scheduler = self._scheduler()
        stub = SimpleNamespace(_census_scheduler=scheduler)
        for name in ("_owner_ident", "_owner_pool_of"):
            setattr(stub, name, getattr(PhaseFlipRuntime, name))
        for name in ("_census_owner_probe", "_pool_census"):
            setattr(
                stub,
                name,
                getattr(PhaseFlipRuntime, name).__get__(stub, SimpleNamespace),
            )

        with unittest.mock.patch.object(
            logging.getLogger("sglang.srt.managers.phase_flip_runtime"), "warning"
        ) as warn:
            stub._pool_census("at-arm", "pp_to_tp")

        rendered = [c[0][0] % tuple(c[0][1:]) for c in warn.call_args_list]
        self.assertTrue(
            any("POOL CENSUS" in t for t in rendered),
            "the census itself must still emit",
        )
        self.assertTrue(
            any("POOL OWNERS" in t for t in rendered),
            "the census must REACH the owner probe -- this is the assertion "
            "that dies when the call is removed",
        )
        owners = next(t for t in rendered if "POOL OWNERS" in t)
        self.assertIn("CENSUS:", owners)
        self.assertIn("UNACCOUNTED:", owners)
