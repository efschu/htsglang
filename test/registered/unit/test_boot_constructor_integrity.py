"""Structural guards for constructors that no test ever executes.

Background: 38220757cc. A method definition written at class-body indentation
in the MIDDLE of `KVSessionOffloadManager.__init__` truncated that constructor
from 63 statements to 38 -- silently. The whole CPU suite stayed green, because
every unit test for that class builds it with `object.__new__` and injects the
handful of attributes it needs (a real `__init__` there wants a scheduler, KV
pools and a GPU). That is a legitimate way to test those methods, and it means
the constructor itself is never exercised by any test.

`test_no_unreachable_after_return.py` catches the usual shape of that accident
repo-wide (the host function's tail ends up behind the inserted function's
`return`). It cannot catch the variant where the inserted method has no
top-level `return` before the tail: then the stolen statements simply become
part of the new method and run at the wrong time, with no unreachable code
anywhere. This file is the second layer for exactly that case.

Scope: constructors that are (a) never executed by any test, (b) reached only
behind an opt-in flag, so a routine boot does not exercise them either, and
(c) load-bearing at runtime. Constructors that run on EVERY boot (Scheduler,
ModelRunner, TpModelWorker, GroupCoordinator) are deliberately NOT listed: they
are equally untested, but any smoke boot fails loudly on a truncated one, so a
guard buys little. See AUDIT notes in the commit message.

Two of the classes below (`DFlashWorkerV2`, `CrossAlgoWorker`) define
`__getattr__` delegating to the wrapped worker. A missing attribute there does
not raise AttributeError at all -- it silently resolves to the OTHER worker's
value. Those are the quietest failures in the set.

The check is deliberately structural and import-free: the source file is
parsed with `ast`, so this stays a cheap CPU test and needs neither CUDA nor
the heavy import chain of the speculative workers.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRT = _REPO_ROOT / "python" / "sglang" / "srt"

# (source file, class, attributes that MUST still be assigned inside __init__).
# The attributes are picked from the second half of each constructor -- the
# part that a `def` inserted mid-body would swallow. Truncation drops them and
# the assertion below names the class.
#
# If a rename makes one of these fail: confirm the constructor is intact, then
# update the name here. Do not delete an entry to make the test pass.
_LOAD_BEARING_INIT_ATTRS = (
    (
        "speculative/cross_algo_worker.py",
        "CrossAlgoWorker",
        # __getattr__ delegates to self._primary -> a missing attribute here is
        # silently answered by the primary sub-worker instead of raising.
        (
            "_secondary_args",
            "_primary",
            "_secondary",
            "_active_name",
            "_bandit",
            "_boundary_pending",
        ),
    ),
    (
        "speculative/dflash_worker_v2.py",
        "DFlashWorkerV2",
        # __getattr__ delegates to self.target_worker -- same silent fallback.
        (
            "_use_fused_kv_materialize",
            "_accept_len_buf",
            "_solo_hs_buf",
            "_audit_round_events",
            "_audit_timing_flush_every",
        ),
    ),
    (
        "layers/moe/expert_offload.py",
        "MoEExpertOffloadCache",
        (
            "_hot_counts",
            "_spill_pool_index",
            "_graph_mode",
            "_capturable_ready",
            "_cap_pool_dev",
            "_cap_view_holders",
        ),
    ),
    (
        "speculative/eagle_worker_v2.py",
        "EagleDraftWorker",
        (
            "draft_worker",
            "draft_runner",
            "dsa_extend_topk_buf",
            "draft_tp_context",
            "tree_mask_mode",
            "plan_stream",
            "plan_stream_ctx",
        ),
    ),
    (
        "speculative/eagle_worker_v2.py",
        "EAGLEWorkerV2",
        (
            "_draft_worker",
            "adaptive_controller",
            "num_new_pages_per_topk",
            "extend_lens",
            "plan_stream",
            "plan_stream_ctx",
        ),
    ),
    (
        "speculative/frozen_kv_mtp_worker_v2.py",
        "FrozenKVMTPWorkerV2",
        (
            "_draft_worker",
            "adaptive_controller",
            "num_new_pages_per_topk",
            "extend_lens",
            "plan_stream",
            "plan_stream_ctx",
        ),
    ),
    (
        "speculative/multi_layer_eagle_worker_v2.py",
        "MultiLayerEagleWorkerV2",
        (
            "_draft_worker",
            "adaptive_controller",
            "num_new_pages_per_topk",
            "extend_lens",
            "plan_stream",
            "plan_stream_ctx",
        ),
    ),
    (
        "speculative/frozen_kv_mtp_worker_v2.py",
        "FrozenKVMTPDraftWorker",
        (
            "hot_token_id",
            "kv_context",
            "draft_tp_context",
            "draft_attn_backend",
            "cuda_graph_runner",
            "draft_extend_attn_backend",
            "cuda_graph_runner_for_draft_extend",
        ),
    ),
    (
        "distributed/device_communicators/barlink.py",
        "BarlinkCommunicator",
        (
            # `device_transport` / `shm_transport` were replaced by the single
            # pluggable seam `transport` (shm | device | ucx). The pin follows
            # the seam, not the old names.
            "transport",
            "_stream",
            "_host_bufs",
            "_host_buf_bytes",
            # NOTE: `_out_pool` was deliberately REMOVED, not lost. It cached
            # one output tensor per (shape, dtype) and so made two same-shape
            # all_reduce results the same tensor -- the second call clobbered
            # the first while the model still held it, corrupting the forward
            # on every non-device transport. all_reduce is documented and
            # dispatched as out-of-place; it now allocates per call, like the
            # device transport always did.
        ),
    ),
    (
        "distributed/device_communicators/barlink_device.py",
        "BarlinkDeviceTransport",
        (
            "_slot_addrs",
            "_seq_dev",
            "_owner_weights",
            "_scratch",
            "_stream2",
            "_pipe_chunk_bytes",
        ),
    ),
    (
        "distributed/device_communicators/barlink_shm.py",
        "BarlinkShmTransport",
        ("_counters", "_slots", "_pinned", "_seq", "_slot_tensors"),
    ),
)


class TestBootConstructorIntegrity(CustomTestCase):
    def test_load_bearing_constructors_are_not_truncated(self):
        failures = []
        for rel, class_name, attrs in _LOAD_BEARING_INIT_ATTRS:
            path = _SRT / rel
            self.assertTrue(path.exists(), f"{rel} no longer exists")
            assigned, error = _init_assigned_attrs(path, class_name)
            if error:
                failures.append(f"{rel}::{class_name}: {error}")
                continue
            missing = [a for a in attrs if a not in assigned]
            if missing:
                failures.append(
                    f"{rel}::{class_name}.__init__ no longer assigns "
                    f"{missing} (it still assigns {len(assigned)} attributes)"
                )

        self.assertFalse(
            failures,
            msg=(
                "Constructor(s) below lost load-bearing assignments. No test "
                "executes these __init__ bodies, so nothing else will tell "
                "you. The usual cause is a `def` written at class-body "
                "indentation inside the constructor, which ends it early and "
                "turns the rest into another method's body:\n  "
                + "\n  ".join(failures)
            ),
        )


def _init_assigned_attrs(path: pathlib.Path, class_name: str):
    """Return (set of self.<attr> assigned in <class_name>.__init__, error)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return set(), f"cannot parse: {exc}"

    cls = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if cls is None:
        return set(), "class not found"

    init = next(
        (
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ),
        None,
    )
    if init is None:
        return set(), "no __init__"

    assigned = {
        node.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    return assigned, None


if __name__ == "__main__":
    unittest.main()
