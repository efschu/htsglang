"""y6o (01.10.): the host-planned eager forward republishes its pool tables
once per model forward, not after every MoE layer.

y6m slot 2, py-spy on D TP0 during an extend: ``check_pool_error`` 18.8 % of
all samples (382 of the 526 under ``_run_eager_host_plan``), ``sync_tables``
34 more -- ``sync_pool_from_host`` reads the device after EVERY MoE layer
(``int(tables.error[0])``, ``take_report``, ``.cpu()`` of the tables), so each
layer drains the stream before the next layer's attention is launched.

Pinned (hermetic, real CPU pool tables, the real ``sync_pool_from_host`` /
``_finish_pool_sync``; the device reads counted on ``check_pool_error`` and
``sync_tables``):
* inside ``eager_pool_sync_scope`` no layer reads the device before the scope
  closes; at the close every layer is republished once, in layer order;
* a sticky device error in layer k is raised at the close, naming layer k,
  before the forward's output leaves the scope;
* outside a scope (any other caller) and with the switch off the republish
  runs at once, per layer, as before;
* the same layer twice in one scope is published before it is queued again;
* a forward that fails inside the scope still publishes, and the first error
  is the one that propagates;
* ``Qwen4ExpModel.forward`` runs its layer loop inside the scope.
"""

import contextlib
import inspect
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as epd  # noqa: E402

E, R, ROWS, STAGING = 16, 4, 12, 2

# the base (no scope, no switch) runs the same tests: there every layer reads
# the device at once, which is the red this file pins
_scope = getattr(eo, "eager_pool_sync_scope", contextlib.nullcontext)
_QUEUE = getattr(eo, "_EAGER_SYNC", types.SimpleNamespace(pending=[], depth=0))


def _switch(on):
    field = getattr(envs, "FLLIPER_OPT_MOE_POOL_DEFER_EAGER_SYNC", None)
    return field.override(on) if field is not None else contextlib.nullcontext()


def _tables():
    hot = {e: e for e in range(R)}
    host = [-1] * R + list(range(E - R))
    return epd.allocate_pool_tables("cpu", E, ROWS, R, STAGING, hot, host)


class _Cache:
    """A pool layer after its eager waves: real tables, real sync methods."""

    def __init__(self, lid):
        self.layer = types.SimpleNamespace(layer_id=lid)
        self._pool_ready = True
        self._pool_tables = _tables()
        self._eager_lru_used = {}
        self._scratch_holds = {R: 5}

    sync_pool_from_host = eo.MoEExpertOffloadCache.sync_pool_from_host
    _finish_pool_sync = getattr(eo.MoEExpertOffloadCache, "_finish_pool_sync", None)


class _Reads:
    """Count the device reads of the republish, in order, per layer."""

    def __init__(self, test):
        self.log = []
        real_check, real_sync = epd.check_pool_error, epd.sync_tables

        def check(tables, where=""):
            self.log.append(("check", where))
            return real_check(tables, where)

        def sync(tables, holds, **kw):
            self.log.append(("sync", id(tables)))
            return real_sync(tables, holds, **kw)

        for name, fn in (("check_pool_error", check), ("sync_tables", sync)):
            p = mock.patch.object(epd, name, fn)
            p.start()
            test.addCleanup(p.stop)

    def checks(self):
        return [w for k, w in self.log if k == "check"]


class EagerPoolSyncDeferTest(unittest.TestCase):
    def setUp(self):
        self._env = _switch(True)
        self._env.__enter__()
        self.addCleanup(self._env.__exit__, None, None, None)
        _QUEUE.pending.clear()
        _QUEUE.depth = 0

    def test_no_device_read_inside_the_scope_then_once_per_layer_in_order(self):
        reads = _Reads(self)
        layers = [_Cache(i) for i in range(4)]
        with _scope():
            for c in layers:
                c.sync_pool_from_host()
            self.assertEqual(reads.log, [], "a layer read the device inside the forward")
        self.assertEqual(reads.checks(), [f"layer {i} sync" for i in range(4)])
        self.assertEqual(
            [t for k, t in reads.log if k == "sync"],
            [id(c._pool_tables) for c in layers],
        )
        self.assertEqual(_QUEUE.pending, [])

    def test_error_in_layer_k_raises_at_the_close_naming_layer_k(self):
        reads = _Reads(self)
        layers = [_Cache(i) for i in range(5)]
        layers[3]._pool_tables.error.fill_(1)
        with self.assertRaisesRegex(RuntimeError, r"sticky device error set at layer 3 sync"):
            with _scope():
                for c in layers:
                    c.sync_pool_from_host()
                self.assertEqual(reads.log, [])
        # layers 0..2 were published, the error stopped the forward at 3
        self.assertEqual(reads.checks(), [f"layer {i} sync" for i in range(4)])
        self.assertEqual(_QUEUE.pending, [])
        self.assertEqual(_QUEUE.depth, 0)

    def test_outside_a_scope_the_republish_runs_at_once(self):
        reads = _Reads(self)
        c = _Cache(7)
        c.sync_pool_from_host()
        self.assertEqual(reads.checks(), ["layer 7 sync"])
        c._pool_tables.error.fill_(1)
        with self.assertRaisesRegex(RuntimeError, "layer 7 sync"):
            c.sync_pool_from_host()

    def test_switch_off_is_per_layer_as_before(self):
        reads = _Reads(self)
        with _switch(False):
            with _scope():
                _Cache(0).sync_pool_from_host()
                self.assertEqual(reads.checks(), ["layer 0 sync"])
                _Cache(1).sync_pool_from_host()
                self.assertEqual(reads.checks(), ["layer 0 sync", "layer 1 sync"])

    def test_same_layer_twice_is_published_before_it_is_queued_again(self):
        reads = _Reads(self)
        a, b = _Cache(0), _Cache(1)
        with _scope():
            a.sync_pool_from_host()
            b.sync_pool_from_host()
            self.assertEqual(reads.log, [])
            a.sync_pool_from_host()
            self.assertEqual(reads.checks(), ["layer 0 sync", "layer 1 sync"])
        self.assertEqual(reads.checks(), ["layer 0 sync", "layer 1 sync", "layer 0 sync"])

    def test_holds_are_taken_when_queued_not_when_published(self):
        seen = []
        real_sync = epd.sync_tables

        def sync(tables, holds, **kw):
            seen.append(dict(holds))
            return real_sync(tables, holds, **kw)

        with mock.patch.object(epd, "sync_tables", sync):
            c = _Cache(0)
            with _scope():
                c.sync_pool_from_host()
                c._scratch_holds.clear()  # the next begin_eager_pool would
        self.assertEqual(seen, [{R: 5}])

    def test_failed_forward_still_publishes_and_keeps_its_own_error(self):
        reads = _Reads(self)
        layers = [_Cache(0), _Cache(1)]
        layers[1]._pool_tables.error.fill_(1)
        with self.assertRaisesRegex(ValueError, "forward broke"):
            with _scope():
                for c in layers:
                    c.sync_pool_from_host()
                raise ValueError("forward broke")
        self.assertEqual(reads.checks(), ["layer 0 sync", "layer 1 sync"])
        self.assertEqual(_QUEUE.depth, 0)

    def test_nested_scope_publishes_only_at_the_outermost_close(self):
        reads = _Reads(self)
        with _scope():
            with _scope():
                _Cache(0).sync_pool_from_host()
            self.assertEqual(reads.log, [])
        self.assertEqual(reads.checks(), ["layer 0 sync"])

    def test_model_forward_runs_its_layer_loop_inside_the_scope(self):
        from flliper.srt.models import qwen4_exp

        src = inspect.getsource(qwen4_exp.Qwen4ExpModel.forward)
        scope = src.index("with eager_pool_sync_scope():")
        loop = src.index("for i in range(self.start_layer, self.end_layer):")
        self.assertLess(scope, loop)
        self.assertLess(loop, src.index("_hap.pass_end()"))


if __name__ == "__main__":
    unittest.main()
