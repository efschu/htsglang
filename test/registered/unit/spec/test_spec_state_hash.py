"""Unit tests for the persistent-state hash walker (spec_state_hash).

The walker is the locator for the #50 deterministic cross-request state
evolution: it must (a) reach tensors nested in objects/containers/modules,
(b) produce hashes that change iff bytes change, (c) survive cycles, and
(d) skip weights (nn.Parameter) while keeping buffers.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.debug_utils import spec_state_hash
from sglang.srt.debug_utils.spec_state_hash import (
    collect_state_entries,
    hash_tensor,
    maybe_dump_on_request_finish,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Traverse into this test module's classes plus torch.nn containers.
_PREFIXES = (__name__, "torch.nn.", "sglang.")


class _Node:
    """Object the walker should traverse into (module prefix allowlisted)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _entry_map(entries):
    out = {}
    for line in entries:
        # "path=... kind=..." -> key: path value: rest
        path = line.split(" ", 1)[0][len("path=") :]
        out[path] = line
    return out


class TestHashTensor(CustomTestCase):
    def test_deterministic_and_change_sensitive(self):
        a = torch.arange(1000, dtype=torch.float32)
        b = torch.arange(1000, dtype=torch.float32)
        self.assertEqual(hash_tensor(a), hash_tensor(b))
        b[500] += 1
        self.assertNotEqual(hash_tensor(a), hash_tensor(b))

    def test_non_numpy_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.bool, torch.uint8):
            t = torch.zeros(64, dtype=dtype)
            h0 = hash_tensor(t)
            self.assertNotIn("ERR", h0)
            if dtype is torch.bool:
                t[3] = True
            else:
                t[3] = 1
            self.assertNotEqual(h0, hash_tensor(t))

    def test_noncontiguous_view_matches_contiguous_copy(self):
        base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        col = base[:, 3]
        self.assertEqual(hash_tensor(col), hash_tensor(col.contiguous().clone()))

    def test_sampled_fingerprint_detects_change(self):
        t = torch.zeros(1 << 12, dtype=torch.float32)  # 16 KiB
        h0 = hash_tensor(t, max_bytes=1 << 10)  # force sampling
        self.assertTrue(h0.startswith("~"))
        t[0] = 5.0
        self.assertNotEqual(h0, hash_tensor(t, max_bytes=1 << 10))

    def test_empty(self):
        self.assertEqual(hash_tensor(torch.empty(0)), "empty")


class TestCollectStateEntries(CustomTestCase):
    def test_nested_tensor_scalar_and_skip_rules(self):
        inner = _Node(
            buf=torch.ones(4),
            counter=7,
            ratio=0.5,
            flag=True,
            name="skipped-string",
            fn=lambda x: x,
        )
        root = _Node(
            inner=inner,
            lst=[torch.zeros(2), 3],
            dct={"k": torch.full((2,), 2.0)},
        )
        entries, n_tensors, _ = collect_state_entries(
            {"root": root}, traverse_prefixes=_PREFIXES
        )
        paths = _entry_map(entries)
        self.assertIn("root.inner.buf", paths)
        self.assertIn("root.inner.counter", paths)
        self.assertIn("value=7", paths["root.inner.counter"])
        self.assertIn("root.inner.ratio", paths)
        self.assertIn("root.inner.flag", paths)
        self.assertIn("value=True", paths["root.inner.flag"])
        self.assertIn("root.lst[0]", paths)
        self.assertIn("root.lst[1]", paths)  # scalar in list
        self.assertIn("root.dct['k']", paths)
        self.assertNotIn("root.inner.name", paths)  # strings skipped
        self.assertNotIn("root.inner.fn", paths)  # routines skipped
        self.assertEqual(n_tensors, 3)

    def test_mutation_changes_exactly_one_hash(self):
        t1, t2 = torch.zeros(8), torch.zeros(8)
        root = _Node(a=t1, b=t2)
        before = _entry_map(
            collect_state_entries({"r": root}, traverse_prefixes=_PREFIXES)[0]
        )
        t2[0] = 1.0
        after = _entry_map(
            collect_state_entries({"r": root}, traverse_prefixes=_PREFIXES)[0]
        )
        self.assertEqual(before["r.a"], after["r.a"])
        self.assertNotEqual(before["r.b"], after["r.b"])

    def test_cycle_safe_and_dedup(self):
        shared = torch.ones(3)
        a = _Node(t=shared)
        b = _Node(t=shared, back=a)
        a.fwd = b
        entries, n_tensors, _ = collect_state_entries(
            {"a": a}, traverse_prefixes=_PREFIXES
        )
        # Shared tensor emitted once; traversal terminates.
        self.assertEqual(n_tensors, 1)

    def test_parameters_skipped_buffers_kept(self):
        mod = torch.nn.Linear(4, 4)
        mod.register_buffer("state_buf", torch.zeros(4))
        root = _Node(model=mod)
        entries, n_tensors, n_params = collect_state_entries(
            {"r": root}, traverse_prefixes=_PREFIXES
        )
        paths = _entry_map(entries)
        self.assertIn("r.model._buffers['state_buf']", paths)
        self.assertEqual(n_params, 2)  # weight + bias skipped
        self.assertEqual(n_tensors, 1)

    def test_untraversable_foreign_objects_are_leaves(self):
        class Foreign:  # simulate non-sglang module by prefix filter
            pass

        f = Foreign()
        f.t = torch.ones(2)
        root = _Node(foreign=f)
        entries, n_tensors, _ = collect_state_entries(
            {"r": root}, traverse_prefixes=("nonexistent.prefix.",)
        )
        # root itself is not traversable under this prefix either
        self.assertEqual(n_tensors, 0)

    def test_flashinfer_like_wrappers_are_traversed_by_default(self):
        # Round-8 coverage hole: attention wrapper objects (module prefix
        # "flashinfer...") hold persistent plan/workspace tensors and must be
        # reached with the DEFAULT prefixes.
        class FakeWrapper:
            pass

        FakeWrapper.__module__ = "flashinfer.prefill"
        w = FakeWrapper()
        w._int_workspace_buffer = torch.zeros(8, dtype=torch.int32)

        class FakeBackend:
            pass

        FakeBackend.__module__ = "sglang.srt.test_fake"
        root = FakeBackend()
        root.wrapper = w
        entries, n_tensors, _ = collect_state_entries({"r": root})
        paths = _entry_map(entries)
        self.assertIn("r.wrapper._int_workspace_buffer", paths)
        self.assertEqual(n_tensors, 1)


class TestRequestFinishHook(CustomTestCase):
    def test_dump_emitted_only_on_finish_and_counts(self):
        spec_state_hash._finished_request_count = 0
        sched = SimpleNamespace(
            tp_rank=0,
            draft_worker=None,
            tp_worker=_Node(runner=_Node(buf=torch.zeros(2))),
        )
        running = SimpleNamespace(
            reqs=[SimpleNamespace(finished=lambda: False)]
        )
        finished = SimpleNamespace(
            reqs=[SimpleNamespace(finished=lambda: True)]
        )
        logger_name = "sglang.srt.debug_utils.spec_state_hash"
        # Unfinished batch: no dump lines.
        with self.assertLogs(logger_name, level="INFO") as cm:
            maybe_dump_on_request_finish(sched, running)
            maybe_dump_on_request_finish(sched, finished)
        text = "\n".join(cm.output)
        self.assertIn("SPEC_STATE_HASH BEGIN tag=req_end_1 rank=0", text)
        self.assertEqual(text.count("BEGIN"), 1)
        with self.assertLogs(logger_name, level="INFO") as cm2:
            maybe_dump_on_request_finish(sched, finished)
        self.assertIn("tag=req_end_2", "\n".join(cm2.output))


if __name__ == "__main__":
    unittest.main()
