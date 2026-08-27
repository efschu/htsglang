"""#943: a prefetch operation's binding generation is WRITE-ONCE.

WHY THIS GUARD EXISTS, from the bisection that produced it. Window
window-943-bisect-0827 bisected the "anchor regression" over the 16 first-parent
merges of ``f1a3391b19..dd0e3bc224``, one boot per step, the same
``probe_2k.py`` workload each time. Two columns flip at ONE commit:

    merge  pin           hits  resumes  coherence  verdict
      2    f55843e3d2      1      6       1/7      PASS
      9    d90900a1bc      2      9       1/7      PASS
     13    dcf43f7dcc      2      9       1/7      PASS
     14    4e855cc80a      2      9       1/7      PASS
     15    22a0c290c3      0      0       7/7      FAIL
     16    a48d04b962      0      0       7/7      FAIL

Merge 15 is ``#937`` -- "a prefetch must not publish a span whose host tier was
replaced under it". Every boot that publishes stale spans has live anchors AND
returns garbage; every boot that refuses them has dead anchors AND returns
correct output. The refusal is the garbage fix and the lost anchors are its
price, so #937 is not a regression to undo.

THE DANGER DIRECTION, WHICH IS WHY THIS IS A GUARD AND NOT A COMMENT.
``write_back_stamp_is_current(operation.binding_generation)`` is the only thing
between a stale span and the model. Anyone tasked with winning those cache hits
back can make the symptom vanish by re-stamping the operation instead of
re-fetching under the new binding: the hits return, the log goes quiet, and the
2j garbage comes back byte for byte -- same span, same replaced pool, now with a
stamp that passes the check. That is the cheapest wrong fix available here, and
it is indistinguishable from the right one by every counter the tree has.

So a re-issue must MINT a new operation -- new stamp, new host slots, a fresh
fetch from the content-keyed store -- and never revive this one. The rewrite
raises instead of being merely discouraged.

IDEMPOTENT WRITES STAY LEGAL. The constructor assigns through this setter, and
an equal re-assignment changes nothing; only a CHANGE to an already-stamped
operation is refused. A ``None`` stamp (the constructor's except path) may still
be replaced -- it is refused by ``write_back_stamp_is_current`` anyway, so
guarding it would block a legitimate late stamp to protect nothing.
"""

import unittest

import torch

from sglang.srt.managers.cache_controller import PrefetchOperation, StorageOperation
from sglang.srt.mem_cache.hicache_phase_binding import (
    current_generation,
    write_back_stamp_is_current,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~1s: constructs a handful of plain Python objects. No pool, no accelerator.
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _rewrite_error():
    """Resolved at CALL time, never imported at module scope.

    On the unguarded tree a module-level import of `StaleStampRewrite` fails
    collection, and an ImportError proves a SYMBOL is missing -- not that the
    stamp is rewritable, which is the property under test. This module's own
    family already paid for that distinction (#938: "The new reader is resolved
    at CALL time so that red is behavioural rather than an ImportError").
    """
    from sglang.srt.managers import cache_controller

    err = getattr(cache_controller, "StaleStampRewrite", None)
    if err is None:
        raise AssertionError(
            "cache_controller carries no StaleStampRewrite: the binding stamp "
            "is not write-once, so a re-issue can revive a stale span by "
            "re-stamping the operation instead of fetching again (#943)"
        )
    return err


def _op():
    return StorageOperation(torch.zeros(1, dtype=torch.int64), [1, 2, 3], "h")


class TestTheStampIsWriteOnce(CustomTestCase):
    def test_the_constructor_stamps_the_current_generation(self):
        self.assertEqual(_op().binding_generation, current_generation())

    def test_rewriting_the_stamp_raises(self):
        """THE DANGER DIRECTION. This is the assertion that goes red the day
        someone 'fixes' the lost cache hits by re-stamping."""
        op = _op()
        with self.assertRaises(_rewrite_error()):
            op.binding_generation = op.binding_generation + 1

    def test_the_refusal_names_both_generations_and_the_reason(self):
        op = _op()
        with self.assertRaises(_rewrite_error()) as caught:
            op.binding_generation = 4242
        msg = str(caught.exception)
        self.assertIn(str(op.binding_generation), msg)
        self.assertIn("4242", msg)
        self.assertIn("re-issue", msg)

    def test_an_equal_write_is_allowed(self):
        """The constructor writes through this setter, and a no-op assignment
        must not be mistaken for a rewrite."""
        op = _op()
        op.binding_generation = op.binding_generation
        self.assertEqual(op.binding_generation, current_generation())

    def test_a_none_stamp_may_still_be_filled_in(self):
        """`StorageOperation.__init__` records None when the generation cannot
        be read. That op is refused by the publish check anyway, so blocking a
        later stamp would protect nothing and break a legitimate path."""
        op = _op()
        op._binding_generation = None
        op.binding_generation = 7
        self.assertEqual(op.binding_generation, 7)

    def test_the_prefetch_subclass_inherits_the_guard(self):
        """`PrefetchOperation` is the class the #937 check actually reads."""
        op = PrefetchOperation(torch.zeros(1, dtype=torch.int64), [1, 2, 3], "h")
        with self.assertRaises(_rewrite_error()):
            op.binding_generation = (op.binding_generation or 0) + 1


class TestWhyTheRewriteWouldBeCatastrophic(CustomTestCase):
    """The guard is only as motivated as the thing it prevents, so the
    consequence is pinned beside it rather than asserted in prose."""

    def test_a_stale_stamp_is_not_publishable_and_a_rewrite_would_make_it_so(self):
        stale = current_generation() - 1
        self.assertFalse(write_back_stamp_is_current(stale))
        # The rewrite's ENTIRE effect: the same span, now passing the gate.
        self.assertTrue(write_back_stamp_is_current(current_generation()))

    def test_an_unstamped_operation_is_never_publishable(self):
        self.assertFalse(write_back_stamp_is_current(None))


if __name__ == "__main__":
    unittest.main()
