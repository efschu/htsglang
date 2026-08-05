"""#583: the census tick must address the REAL Scheduler attribute surface.

WHAT WENT WRONG
---------------
``Scheduler._census_tick`` was written against ``self.tp_rank``. That
attribute does not exist: the Scheduler keeps its parallel identity on the
``ParallelState`` wrapper, ``self.ps.tp_rank``. So in production EVERY tick
raised ``AttributeError`` and was swallowed by the tick's own
warn-never-raise guard. The census counted nothing, compared nothing, and
would have dumped nothing at abort time -- for 2661 ticks in one boot.

The unit tests did not catch it because they exercised
``CollectiveCensus`` directly and stubbed the scheduler surface. A stub
that grants the attribute proves only that the stub has it. That is the
desk-written-never-executed trap in its purest form: the instrument built
to make silence falsifiable was itself silently dead.

This file closes the gap two ways:

  * STRUCTURAL -- parse the real ``scheduler.py`` and assert the methods
    that read the parallel identity go through ``self.ps``, and that
    ``self.tp_rank`` / ``self.tp_size`` appear nowhere in them. This cannot
    be satisfied by a stub, because it reads the shipped source.
  * EXECUTION -- run the real ``Scheduler._census_tick`` bound to an object
    carrying ONLY the attribute surface the real class actually has, and
    assert it completes without taking its skip path.

Hermetic: no CUDA, no process group, no model; the Scheduler is never
constructed (its constructor wants a GPU), only its unbound methods are.
"""

import ast
import inspect
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed import collective_census as cc  # noqa: E402
from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCHEDULER_PY = inspect.getsourcefile(Scheduler)

#: Methods that read the rank/world identity and must go through `self.ps`.
IDENTITY_READERS = ("_census_tick", "uniform_min_avail")


def _self_attr_paths(method_name: str):
    """Every ``self.X`` / ``self.X.Y`` path a Scheduler method references."""
    src = inspect.getsource(getattr(Scheduler, method_name))
    tree = ast.parse(ast.unparse(ast.parse(src.strip())))
    one, two = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # self.X
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            one.add(node.attr)
        # self.X.Y
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        ):
            two.add((node.value.attr, node.attr))
    return one, two


class CensusAttributeSurfaceTest(unittest.TestCase):
    # -- the fact that was got wrong, pinned directly ----------------------

    def test_the_scheduler_has_no_tp_rank_of_its_own(self):
        """THE FALSIFIER. If this ever becomes false the guard below is
        pointless -- and if it is true, `self.tp_rank` is a bug, which is
        exactly what shipped."""
        self.assertFalse(hasattr(Scheduler, "tp_rank"))
        self.assertFalse(hasattr(Scheduler, "tp_size"))

    def test_the_parallel_state_is_where_the_identity_lives(self):
        for field in ("tp_rank", "tp_size"):
            self.assertIn(field, ParallelState.__annotations__)

    # -- STRUCTURAL: reads the shipped source, so no stub can satisfy it ---

    def test_identity_readers_go_through_ps_and_never_bare_self(self):
        for name in IDENTITY_READERS:
            one, two = _self_attr_paths(name)
            self.assertNotIn(
                "tp_rank",
                one,
                msg=f"Scheduler.{name} reads self.tp_rank, which does not "
                f"exist -- use self.ps.tp_rank",
            )
            self.assertNotIn(
                "tp_size",
                one,
                msg=f"Scheduler.{name} reads self.tp_size, which does not "
                f"exist -- use self.ps.tp_size",
            )

    def test_the_census_tick_actually_reads_ps_tp_rank(self):
        """Positive half: the guard above passes trivially if the method
        stopped reading the rank at all."""
        _, two = _self_attr_paths("_census_tick")
        self.assertIn(("ps", "tp_rank"), two)

    def test_uniform_min_avail_actually_reads_ps_tp_size(self):
        _, two = _self_attr_paths("uniform_min_avail")
        self.assertIn(("ps", "tp_size"), two)

    def test_every_ps_field_the_readers_use_exists_on_the_real_class(self):
        """Generalises the two pins above: any future `self.ps.X` in these
        methods must be a real ParallelState field."""
        for name in IDENTITY_READERS:
            _, two = _self_attr_paths(name)
            for holder, field in two:
                if holder != "ps":
                    continue
                self.assertIn(
                    field,
                    ParallelState.__annotations__,
                    msg=f"Scheduler.{name} reads self.ps.{field}, which is "
                    f"not a ParallelState field",
                )

    # -- EXECUTION: the real method against the real attribute shape -------

    def test_the_real_tick_runs_against_the_real_surface_without_skipping(self):
        """Binds the REAL `_census_tick` to an object carrying only what the
        real Scheduler has. Had this existed, the shipped bug would have
        failed here instead of in production."""
        census = cc.CollectiveCensus()
        census.bump("tp.all_reduce")
        obj = mock.Mock()
        # Built from the class's OWN annotations, so adding a field to
        # ParallelState cannot silently turn this test into a stub again.
        obj.ps = ParallelState(
            **{f: (0 if f != "tp_size" else 1) for f in ParallelState.__annotations__}
        )
        obj.tp_cpu_group = None
        with (
            mock.patch.object(cc, "_CENSUS", census),
            mock.patch.multiple(
                "sglang.srt.managers.scheduler",
                _CENSUS=census,
                _CENSUS_ON=True,
                _CENSUS_INTERVAL=1,
                _CENSUS_HEARTBEAT=0,
            ),
        ):
            Scheduler._census_tick(obj)
        # The skip path is what production took; it must not be taken here.
        self.assertFalse(
            census._skip_warned,
            msg="the census tick took its skip path against the real "
            "attribute surface -- it is dead again",
        )
        # And a completed tick is what licenses the arming line.
        self.assertTrue(census._armed_announced)

    # -- the skip warning must not spam --------------------------------

    def test_the_skip_warning_is_emitted_once_not_per_tick(self):
        census = cc.CollectiveCensus()
        with mock.patch.object(cc.logger, "error") as err:
            for _ in range(500):
                census.warn_skipped_once(AttributeError("boom"))
        self.assertEqual(err.call_count, 1)
        self.assertIn("NOT RUNNING", err.call_args[0][0])

    def test_a_broken_tick_never_announces_itself_as_armed(self):
        """The arming line certifies the instrument. It must follow the work,
        never precede it, or it certifies a corpse."""
        census = cc.CollectiveCensus()
        obj = mock.Mock()
        del obj.ps  # the exact production failure: no parallel state
        with mock.patch.multiple(
            "sglang.srt.managers.scheduler",
            _CENSUS=census,
            _CENSUS_ON=True,
            _CENSUS_INTERVAL=1,
            _CENSUS_HEARTBEAT=0,
        ):
            Scheduler._census_tick(obj)
        self.assertFalse(census._armed_announced)
        self.assertTrue(census._skip_warned)


if __name__ == "__main__":
    unittest.main()
