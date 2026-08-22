"""#796: `build_pp_admission_decision` must read a prefix length from a
TENSOR without ever putting it in boolean context.

THE SPECIMEN (boot instr5, 2026-08-21, /spinning/evidence-665-f1/
boot_instr5.log:6126-6155). PP0 aborted on its first real prefill:

    File "pp_admission_congruence.py", line 352, in build_pp_admission_decision
      raw_prefix_len = len(getattr(req, "prefix_indices", None) or [])
    RuntimeError: Boolean value of Tensor with no values is ambiguous

`x or []` evaluates `bool(x)`. `req.prefix_indices` is a tensor of KV-pool
slot pointers, and torch refuses that question at both ends of the range
this code actually sees: an EMPTY tensor (a request with no cached prefix --
the common case, and the one that crashed the boot) raises "no values is
ambiguous", and a tensor with several matched pages raises the "more than
one element is ambiguous" variant. ONLY a single-element prefix would have
passed through silently. The line was therefore broken for very nearly every
request that could ever reach it.

WHY IT SURVIVED THIS LONG, which is the part worth recording. It is on the
PP0-only decision-building path, reached only once a request is actually
being admitted -- and until the #796 send-handle fix landed, the ring wedged
at idle before any request was ever admitted. Six boots died upstream of
this line. It is a desk-written expression that no boot had executed.

WHAT THIS FILE PINS. All four shapes the attribute is observed in -- empty
tensor, multi-element tensor, absent/None, and a plain list -- reach a
correct integer length, and the entry that goes on the wire carries it. The
empty and multi-element cases are the two that raised; a test that only
covered `None` and a one-element tensor would have passed against the
defect.
"""

import types
import unittest

import torch

from sglang.srt.managers.pp_admission_congruence import build_pp_admission_decision


def _req(rid, prefix_indices, extend_input_len=7):
    """A stand-in carrying only what `build_pp_admission_decision` reads:
    `rid`, `prefix_indices`, and the request's own extend length."""
    r = types.SimpleNamespace(rid=rid, extend_input_len=extend_input_len)
    if prefix_indices is not _ABSENT:
        r.prefix_indices = prefix_indices
    return r


class _Absent:
    pass


_ABSENT = _Absent()


class PPAdmissionPrefixIndicesTensor796(unittest.TestCase):
    def _prefix_len_for(self, prefix_indices):
        decision = build_pp_admission_decision(
            0, [_req("r0", prefix_indices)], pp_size=3
        )
        self.assertEqual(len(decision.entries), 1)
        return decision.entries[0].prefix_len

    def test_empty_tensor_is_length_zero(self):
        """The exact specimen: a request with no cached prefix. This raised
        'Boolean value of Tensor with no values is ambiguous' and killed the
        boot."""
        self.assertEqual(self._prefix_len_for(torch.empty(0, dtype=torch.int64)), 0)

    def test_multi_element_tensor_is_its_row_count(self):
        """The other raising shape: 'more than one element is ambiguous'.
        A real matched prefix is many slot pointers, never one."""
        self.assertEqual(
            self._prefix_len_for(torch.arange(128, dtype=torch.int64)), 128
        )

    def test_single_element_tensor_still_works(self):
        """The only shape the defective spelling handled -- pinned so a
        future 'simplification' back to a truthiness test does not look
        harmless on the sample it is tried against."""
        self.assertEqual(self._prefix_len_for(torch.tensor([5], dtype=torch.int64)), 1)

    def test_absent_and_none_are_length_zero(self):
        self.assertEqual(self._prefix_len_for(None), 0)
        self.assertEqual(self._prefix_len_for(_ABSENT), 0)

    def test_a_plain_list_still_works(self):
        """Stand-ins and the older tests in this feature pass lists."""
        self.assertEqual(self._prefix_len_for([1, 2, 3]), 3)

    def test_the_extend_length_absorbs_nothing_from_the_prefix_read(self):
        """The prefix read must not disturb the other half of the entry:
        prefix_len + extend_len is the quantity the crossing shapes depend
        on (scheduler_pp_mixin.py's shape invariant)."""
        decision = build_pp_admission_decision(
            0,
            [_req("r0", torch.arange(64, dtype=torch.int64), extend_input_len=9)],
            pp_size=3,
        )
        entry = decision.entries[0]
        self.assertEqual(entry.prefix_len, 64)
        self.assertEqual(entry.extend_len, 9)


class PPAdmissionTracePrefixIndicesTensor796(unittest.TestCase):
    """The SECOND instance of the same defect, in the instrument itself.

    `Scheduler._trace_pp_admission_verdict` spelled the same read as
    `str(len(getattr(r, "prefix_indices", []) or []))`. That branch runs only
    when the verdict is ADMIT, so the failure mode was silent and precisely
    inverted from useful: every DECLINE (idle) pass logged cleanly, and every
    ADMIT pass -- the only ones that could show the ranks agreeing or
    diverging on a real request -- threw into the instrument's own
    except-and-swallow and printed nothing but "RuntimeError". Boot instr6
    burned a whole GPU window learning exactly that.
    """

    def _trace_holder(self, lines):
        from sglang.srt.managers.scheduler import Scheduler

        h = types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(
                available_size=lambda: 161378
            ),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: 0),
            waiting_queue=[],
            running_batch=None,
            chunked_req=None,
        )
        h._trace_pp_admission_verdict = types.MethodType(
            Scheduler._trace_pp_admission_verdict, h
        )
        return h

    def _admit_batch(self, prefix_indices):
        req = types.SimpleNamespace(rid="r0", prefix_indices=prefix_indices)
        return types.SimpleNamespace(reqs=[req])

    def _capture(self, prefix_indices):
        import logging

        records = []

        class _Grab(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("sglang.srt.managers.scheduler")
        handler = _Grab()
        logger.addHandler(handler)
        prior = logger.level
        logger.setLevel(logging.INFO)
        try:
            h = self._trace_holder(records)
            h._trace_pp_admission_verdict(self._admit_batch(prefix_indices))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior)
        return records

    def _assert_traced_admit(self, prefix_indices, expected_prefix_len):
        records = self._capture(prefix_indices)
        rendered = [r.getMessage() for r in records]
        unavailable = [m for m in rendered if "trace unavailable" in m]
        self.assertEqual(
            unavailable,
            [],
            f"the instrument swallowed its own failure instead of tracing: "
            f"{unavailable}",
        )
        admits = [m for m in rendered if "verdict=ADMIT" in m]
        self.assertEqual(len(admits), 1, f"expected one ADMIT line: {rendered}")
        self.assertIn(f"prefix_lens={expected_prefix_len}", admits[0])

    def test_admit_with_an_empty_prefix_tensor_is_traced(self):
        """The specimen: a first request with no cached prefix."""
        self._assert_traced_admit(torch.empty(0, dtype=torch.int64), 0)

    def test_admit_with_a_multi_element_prefix_tensor_is_traced(self):
        self._assert_traced_admit(torch.arange(96, dtype=torch.int64), 96)

    def test_admit_with_no_prefix_attribute_is_traced(self):
        req = types.SimpleNamespace(rid="r0")
        records = []
        import logging

        class _Grab(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("sglang.srt.managers.scheduler")
        handler = _Grab()
        logger.addHandler(handler)
        prior = logger.level
        logger.setLevel(logging.INFO)
        try:
            h = self._trace_holder(records)
            h._trace_pp_admission_verdict(types.SimpleNamespace(reqs=[req]))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior)
        rendered = [r.getMessage() for r in records]
        self.assertEqual([m for m in rendered if "trace unavailable" in m], [])
        self.assertTrue(any("prefix_lens=0" in m for m in rendered), rendered)


if __name__ == "__main__":
    unittest.main()
