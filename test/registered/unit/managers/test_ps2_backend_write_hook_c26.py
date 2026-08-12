"""#656 / register C26: PS2 may not be admitted onto a backend that cannot
divert its sentinels.

THE BUG, MEASURED, NOT INFERRED. PS2 ("born-spilled deep" prefill-spill) hands
the extend an ``out_cache_loc`` full of HOST SENTINELS -- ``spill_extend_alloc``
returns ``make_sentinels(...)`` -- and exactly ONE thing in the tree diverts
that tensor away from the KV write: ``_dcp_write_scatter``'s
``_sess_prefill_owner_write`` branch, which is reachable only from the
token-sharded DCP lane (``forward_extend`` enters it under ``if
self.uneven_dcp``).

On plain TP the backend still BUILDS the prefill-spill state -- ``_sess_mode``
is "plain", ``_sess_prefill_spill`` is computed, the staging carve is even
reserved -- and nothing complains. Then ``forward_extend`` falls through to
the stock ``set_kv_buffer`` and the sentinels go into ``store_kvcache``:

    jit_kernel/csrc/elementwise/kvcache.cuh:112
      Assertion `index >= 0 && index < size_limit` failed

With ``host_base=4097`` against a 4096-row allocator, a request with
``boundary=2620 L=3012`` writes indices 6717..7108 against a ``size_limit`` of
~4097: all 392 rows out of bounds, on layer 0, on every rank. The instance
dies. This is what blocked #659's end-to-end proof -- the park round trip
completed and then the instance was gone before any parked session could
finish.

The admission gate could not have known: it was never told which backend it
was admitting onto. It is told now, and these are the pins for that.

BOTH ARMS, ALWAYS. A gate that declines everything would pass a
"declines on plain TP" test perfectly, so every decline pin here has a sibling
that admits through the same function with the same other inputs.
"""

import unittest

from sglang.srt.managers.kv_session_offload import (
    prefill_spill_deep_gate,
    prefill_spill_deep_reject_reason,
)


class TheGateRefusesABackendWithNoWriteHook(unittest.TestCase):
    def test_plain_tp_is_declined_even_with_nothing_else_wrong(self):
        self.assertFalse(
            prefill_spill_deep_gate(True, spec_active=False, backend_write_hook=False)
        )

    def test_a_backend_with_the_hook_is_admitted_on_the_same_inputs(self):
        # SIBLING ARM. Identical call, one flag flipped: this is what proves
        # the pin above is about the hook and not about the gate refusing
        # everything.
        self.assertTrue(
            prefill_spill_deep_gate(True, spec_active=False, backend_write_hook=True)
        )

    def test_the_default_preserves_the_pre_c26_behaviour(self):
        # The pure-function pins that predate C26 call this without the new
        # argument. They must keep meaning what they meant.
        self.assertTrue(prefill_spill_deep_gate(True, spec_active=False))
        self.assertFalse(prefill_spill_deep_gate(False, spec_active=False))

    def test_the_reason_names_the_mechanism_not_just_the_verdict(self):
        reason = prefill_spill_deep_reject_reason(
            False, False, False, False, backend_write_hook=False
        )
        self.assertIsNotNone(reason)
        # A reason a successor cannot act on is a reason that gets re-debugged.
        self.assertIn("plain-TP", reason)
        self.assertIn("store_kvcache", reason)
        self.assertIn("C26", reason)

    def test_no_reason_when_the_hook_is_present_and_spec_is_quiet(self):
        # SIBLING ARM on the reason channel.
        self.assertIsNone(
            prefill_spill_deep_reject_reason(
                False, False, False, False, backend_write_hook=True
            )
        )

    def test_the_hook_condition_outranks_the_speculation_conditions(self):
        # Order matters for the OPERATOR, not for the verdict: when both a
        # spec blocker and the missing hook apply, the reported reason must be
        # the one that would kill the instance, not the one that would merely
        # decline it.
        reason = prefill_spill_deep_reject_reason(
            True, True, False, False, backend_write_hook=False
        )
        self.assertIn("plain-TP", reason)
        self.assertNotIn("spec-in-tick", reason)
        # And with the hook present the spec reason comes through unchanged --
        # the new condition must not have swallowed the old ones.
        spec_reason = prefill_spill_deep_reject_reason(
            True, True, False, False, backend_write_hook=True
        )
        self.assertIn("spec-in-tick", spec_reason)

    def test_prefill_spill_off_still_dominates_everything(self):
        # The feature switch is still the outermost condition; a missing hook
        # must not turn "the feature is off" into a different answer.
        self.assertFalse(
            prefill_spill_deep_gate(False, spec_active=False, backend_write_hook=True)
        )
        self.assertFalse(
            prefill_spill_deep_gate(False, spec_active=False, backend_write_hook=False)
        )


class TheAllocatorRefusesToBuildSentinelsItCannotDivert(unittest.TestCase):
    """Belt and braces: if the gate is ever bypassed, fail in Python.

    The point is not redundancy for its own sake. The two failures are not
    equivalent: a Python ``RuntimeError`` names the cause and unwinds, while
    the device-side assert it replaces poisons the CUDA context and takes the
    whole instance down with a traceback that points at an ``all_reduce``
    three call frames away from the actual mistake.
    """

    def _manager(self, mode):
        from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

        mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
        mgr.mode = mode
        mgr.prefill_spill_extend_ready = lambda _batch: True
        return mgr

    def _batch(self):
        class _Req:
            rid = "fast-3"
            req_pool_idx = 0

        class _Cpu:
            def __init__(self, v):
                self._v = v

            def __getitem__(self, _i):
                return self

            def item(self):
                return self._v

        class _Batch:
            reqs = [_Req()]
            prefix_lens = [2620]
            seq_lens_cpu = _Cpu(3012)
            extend_num_tokens = 392

        return _Batch()

    def test_plain_mode_raises_before_a_single_sentinel_is_built(self):
        mgr = self._manager("plain")
        with self.assertRaises(RuntimeError) as ctx:
            mgr.spill_extend_alloc(self._batch())
        msg = str(ctx.exception)
        self.assertIn("plain-TP", msg)
        self.assertIn("C26", msg)
        self.assertIn("fast-3", msg)  # the request, so the log is actionable

    def test_a_dcp_mode_gets_past_the_guard(self):
        # SIBLING ARM. "even" must NOT raise the C26 error -- it will fail
        # later for want of the real fixtures this stub does not provide, and
        # that is the point: the guard is specific to the missing hook and
        # does not stand in the way of the lane that has one.
        mgr = self._manager("even")
        try:
            mgr.spill_extend_alloc(self._batch())
        except RuntimeError as err:
            self.assertNotIn("C26", str(err))
        except Exception:
            pass  # any other failure is this stub's incompleteness, not the guard


if __name__ == "__main__":
    unittest.main()
