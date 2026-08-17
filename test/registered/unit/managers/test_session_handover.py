"""Hermetic proof of the live session handover (#261, second half).

No GPU, no server, no collective: the scheduler, tree and storage backend
are injected fakes, so what is proved here is the CONTRACT -- the
session-scoped five-phase state machine, the manifest gates, and the
rollback rule.

The two falsifiers the task demands are both here, and both sides of each
gate are exercised (a gate that cannot fail proves nothing):

* planted GDN-state omission: a hybrid-GDN export whose mamba blob never
  reached the store MUST fail the completeness gate loudly AND roll the
  park back; the identical export with the blob present passes.
* rollback discipline: abort is legal before commit and refused after it;
  a parked prefix refuses extension requests while active and releases on
  commit/abort.
"""

import json
import queue
import types
import unittest

import torch

from sglang.srt.managers.io_struct import SessionHandoverReqInput
from sglang.srt.managers.session_handover import (
    STATE_EXPORTED,
    STATE_PARKED,
    HandoverLedger,
    SessionHandoverError,
    SessionHandoverRuntime,
    build_manifest,
    handover_id_for,
    prefixes_conflict,
    validate_manifest_completeness,
    verify_import,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_IDENTITY = "id-test-0001"


def _manifest(
    kv_keys=("h0", "h1", "h2"),
    mamba_key="h2.mamba",
    hybrid_gdn=True,
    draft_keys=(),
    identity=_IDENTITY,
    token_ids=(1, 2, 3),
):
    return build_manifest(
        handover_id=handover_id_for(token_ids, identity),
        model_identity_hash=identity,
        token_ids=token_ids,
        kv_keys=list(kv_keys),
        mamba_key=mamba_key,
        hybrid_gdn=hybrid_gdn,
        draft_keys=list(draft_keys),
        tp_size=1,
        tp_rank=0,
        dcp_owner_mode=False,
        page_size=1,
    )


class TestPrefixConflict(CustomTestCase):
    def test_truth_table(self):
        self.assertTrue(prefixes_conflict([1, 2, 3], [1, 2, 3]))
        self.assertTrue(prefixes_conflict([1, 2], [1, 2, 3]))  # extension
        self.assertTrue(prefixes_conflict([1, 2, 3, 4], [1, 2, 3]))
        self.assertFalse(prefixes_conflict([1, 2, 3], [1, 9]))
        self.assertFalse(prefixes_conflict([], [1, 2]))


class TestLedger(CustomTestCase):
    def test_full_lifecycle_and_rollback_rule(self):
        ledger = HandoverLedger()
        rec = ledger.park([1, 2, 3], _IDENTITY)
        self.assertEqual(rec.state, STATE_PARKED)
        self.assertIsNotNone(ledger.conflicts([1, 2, 3, 4]))

        # Commit requires EXPORTED, not merely PARKED.
        with self.assertRaises(SessionHandoverError):
            ledger.commit(rec.handover_id)

        ledger.mark_exported(rec.handover_id, {"kv_keys": ["h0"]})
        self.assertEqual(rec.state, STATE_EXPORTED)
        self.assertIsNotNone(ledger.conflicts([1, 2, 3]))  # still parked

        ledger.commit(rec.handover_id)
        self.assertIsNone(ledger.conflicts([1, 2, 3]))  # released

        # THE rollback rule: no abort after commit.
        with self.assertRaises(SessionHandoverError) as ctx:
            ledger.abort(rec.handover_id)
        self.assertIn("no rollback after commit", str(ctx.exception))

    def test_abort_before_commit_releases(self):
        ledger = HandoverLedger()
        rec = ledger.park([5, 6], _IDENTITY)
        ledger.abort(rec.handover_id)
        self.assertIsNone(ledger.conflicts([5, 6, 7]))
        with self.assertRaises(SessionHandoverError):
            ledger.abort(rec.handover_id)  # double abort refused

    def test_conflicting_park_refused(self):
        ledger = HandoverLedger()
        ledger.park([1, 2, 3], _IDENTITY)
        with self.assertRaises(SessionHandoverError):
            ledger.park([1, 2], _IDENTITY)
        ledger.park([9, 9], _IDENTITY)  # unrelated prefix is fine

    def test_unknown_id_refused(self):
        ledger = HandoverLedger()
        with self.assertRaises(SessionHandoverError):
            ledger.commit("nope")


class TestCompletenessGate(CustomTestCase):
    def test_complete_manifest_passes(self):
        store = {"h0", "h1", "h2", "h2.mamba"}
        checked = validate_manifest_completeness(_manifest(), store.__contains__)
        self.assertEqual(checked["kv_keys"], 3)
        self.assertEqual(checked["mamba"], 1)

    def test_planted_gdn_omission_fails_loudly(self):
        # THE #212 falsifier: KV complete, GDN blob missing from the store.
        store = {"h0", "h1", "h2"}
        with self.assertRaises(SessionHandoverError) as ctx:
            validate_manifest_completeness(_manifest(), store.__contains__)
        msg = str(ctx.exception)
        self.assertIn("h2.mamba", msg)
        self.assertIn("#212", msg)

    def test_hybrid_without_mamba_key_fails(self):
        store = {"h0", "h1", "h2"}
        with self.assertRaises(SessionHandoverError) as ctx:
            validate_manifest_completeness(
                _manifest(mamba_key=None), store.__contains__
            )
        self.assertIn("#212", str(ctx.exception))

    def test_non_hybrid_needs_no_mamba(self):
        store = {"h0", "h1", "h2"}
        validate_manifest_completeness(
            _manifest(mamba_key=None, hybrid_gdn=False), store.__contains__
        )

    def test_missing_kv_page_fails(self):
        store = {"h0", "h2", "h2.mamba"}
        with self.assertRaises(SessionHandoverError) as ctx:
            validate_manifest_completeness(_manifest(), store.__contains__)
        self.assertIn("h1", str(ctx.exception))


class TestVerifyImport(CustomTestCase):
    def test_pass(self):
        store = {"h0", "h1", "h2", "h2.mamba"}
        ok, msg = verify_import(_manifest(), store.__contains__, _IDENTITY)
        self.assertTrue(ok, msg)

    def test_identity_mismatch_refused(self):
        store = {"h0", "h1", "h2", "h2.mamba"}
        ok, msg = verify_import(_manifest(), store.__contains__, "other-model")
        self.assertFalse(ok)
        self.assertIn("identity mismatch", msg)

    def test_missing_blob_refused(self):
        store = {"h0", "h1", "h2"}
        ok, msg = verify_import(_manifest(), store.__contains__, _IDENTITY)
        self.assertFalse(ok)
        self.assertIn("h2.mamba", msg)

    def test_version_mismatch_refused(self):
        m = _manifest()
        m["version"] = 99
        ok, msg = verify_import(m, lambda k: True, _IDENTITY)
        self.assertFalse(ok)
        self.assertIn("version", msg)


# ---------------------------------------------------------------------------
# Runtime adapter with injected fakes
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self):
        self.keys = set()

    def exists(self, key):
        return key in self.keys


class _FakeNode:
    _next_id = 0

    def __init__(self, tokens, parent, backuped=False):
        _FakeNode._next_id += 1
        self.id = _FakeNode._next_id
        self.key = list(tokens)
        self.parent = parent
        self.backuped = backuped
        self.evicted = False
        self.hash_value = [f"h{t}" for t in tokens]


class _FakeTree:
    """Storage-enabled radix tree fake. write_backup/write_backup_storage
    place keys into the backend synchronously; `withhold_mamba` plants the
    GDN omission for the runtime-level falsifier."""

    def __init__(self, token_ids, hybrid=True, withhold_mamba=False):
        self.enable_storage = True
        self.root_node = _FakeNode([], None)
        mid = len(token_ids) // 2 or 1
        n1 = _FakeNode(token_ids[:mid], self.root_node)
        n2 = _FakeNode(token_ids[mid:], n1)
        self.leaf = n2 if n2.key else n1
        self.chain = [n for n in (n1, n2) if n.key]
        self._match = MatchResult(
            device_indices=torch.arange(len(token_ids)),
            last_device_node=self.leaf,
            last_host_node=self.leaf,
            best_match_node=self.leaf,
        )
        self.ongoing_write_through = {}
        self.ongoing_backup = {}
        backend = _FakeBackend()
        self.cache_controller = types.SimpleNamespace(
            storage_backend=backend,
            backup_queue=queue.Queue(),
            ack_backup_queue=queue.Queue(),
            has_draft=False,
            storage_config=types.SimpleNamespace(dcp_owner_mode=False),
        )
        self._hybrid = hybrid
        self._withhold_mamba = withhold_mamba

    # -- runtime-facing API --
    def supports_mamba(self):
        # The BasePrefixCache capability every real tree implements, and the
        # ONE thing the runtime may key the #212 gate on. An earlier version
        # of this fake grew a ``mamba_archive_transfers`` attribute instead,
        # which exists only on HiMambaRadixCache -- so the fake declared a
        # hybrid tree in a way the UnifiedRadixCache the server actually
        # builds never does, and the gate passed here while silently doing
        # nothing on a real boot.
        return self._hybrid

    def match_prefix(self, params):
        return self._match

    def write_backup(self, node):
        node.backuped = True
        self.write_backup_storage(node)

    def write_backup_storage(self, node):
        backend = self.cache_controller.storage_backend
        backend.keys.update(node.hash_value)
        if self._hybrid and node is self.leaf and not self._withhold_mamba:
            backend.keys.add(f"{node.hash_value[-1]}.mamba")

    def writing_check(self, write_back=False):
        pass

    def drain_storage_control_queues(self):
        pass


def _fake_scheduler(tree):
    args = types.SimpleNamespace(
        tp_size=1, pp_size=1, page_size=1, hicache_storage_backend="file"
    )
    return types.SimpleNamespace(
        server_args=args,
        tree_cache=tree,
        running_batch=types.SimpleNamespace(reqs=[]),
        waiting_queue=[],
        chunked_req=None,
    )


def _runtime(tree):
    rt = SessionHandoverRuntime(_fake_scheduler(tree))
    rt._identity_hash = _IDENTITY  # bypass server_args-derived hash
    return rt


_TOKENS = [11, 12, 13, 14]


class TestRuntimeExport(CustomTestCase):
    def test_happy_path_export_commit(self):
        rt = _runtime(_FakeTree(_TOKENS))
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertTrue(out.success, out.message)
        manifest = json.loads(out.manifest_json)
        self.assertEqual(manifest["kv_keys"], [f"h{t}" for t in _TOKENS])
        self.assertEqual(manifest["mamba_key"], f"h{_TOKENS[-1]}.mamba")
        self.assertTrue(manifest["hybrid_gdn"])

        # QUIESCE holds: the prefix and any extension are refused.
        self.assertIsNotNone(rt.parked_conflict(_TOKENS))
        self.assertIsNotNone(rt.parked_conflict(_TOKENS + [99]))
        self.assertIsNone(rt.parked_conflict([7, 7, 7]))

        # RESUME: commit releases the park; abort afterwards is refused.
        hid = manifest["handover_id"]
        self.assertTrue(
            rt.handle(SessionHandoverReqInput(action="commit", handover_id=hid)).success
        )
        self.assertIsNone(rt.parked_conflict(_TOKENS))
        out = rt.handle(SessionHandoverReqInput(action="abort", handover_id=hid))
        self.assertFalse(out.success)
        self.assertIn("no rollback after commit", out.message)

    def test_abort_rolls_back(self):
        rt = _runtime(_FakeTree(_TOKENS))
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        hid = json.loads(out.manifest_json)["handover_id"]
        self.assertTrue(
            rt.handle(SessionHandoverReqInput(action="abort", handover_id=hid)).success
        )
        self.assertIsNone(rt.parked_conflict(_TOKENS))

    def test_planted_gdn_omission_fails_and_unparks(self):
        # THE runtime-level #212 falsifier: hybrid tree whose mamba blob
        # never reaches the store. Export must fail loudly AND roll back.
        rt = _runtime(_FakeTree(_TOKENS, withhold_mamba=True))
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertFalse(out.success)
        self.assertIn("#212", out.message)
        self.assertIsNone(rt.parked_conflict(_TOKENS))  # rollback happened

    def test_non_hybrid_export_passes_without_mamba(self):
        rt = _runtime(_FakeTree(_TOKENS, hybrid=False))
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertTrue(out.success, out.message)
        manifest = json.loads(out.manifest_json)
        self.assertIsNone(manifest["mamba_key"])
        self.assertFalse(manifest["hybrid_gdn"])

    def test_in_flight_request_refuses_quiesce(self):
        tree = _FakeTree(_TOKENS)
        rt = _runtime(tree)
        req = types.SimpleNamespace(
            rid="r-1", origin_input_ids=list(_TOKENS), output_ids=[42]
        )
        rt.scheduler.running_batch.reqs.append(req)
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertFalse(out.success)
        self.assertIn("in-flight", out.message)
        self.assertIsNone(rt.parked_conflict(_TOKENS))  # nothing left parked

    def test_partial_residency_refused(self):
        tree = _FakeTree(_TOKENS)
        tree._match = MatchResult(
            device_indices=torch.arange(len(_TOKENS) - 1),
            last_device_node=tree.leaf,
            last_host_node=tree.leaf,
            best_match_node=tree.leaf,
        )
        rt = _runtime(tree)
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertFalse(out.success)
        self.assertIn("resident", out.message)

    def test_tp_gt_1_source_refused(self):
        rt = _runtime(_FakeTree(_TOKENS))
        rt.scheduler.server_args.tp_size = 2
        out = rt.handle(SessionHandoverReqInput(action="export", token_ids=_TOKENS))
        self.assertFalse(out.success)
        self.assertIn("TP=1", out.message)

    def test_unknown_action_refused(self):
        rt = _runtime(_FakeTree(_TOKENS))
        out = rt.handle(SessionHandoverReqInput(action="teleport"))
        self.assertFalse(out.success)
        self.assertIn("unknown action", out.message)


class TestRuntimeVerifyImport(CustomTestCase):
    def test_verify_import_pass_and_fail(self):
        tree = _FakeTree(_TOKENS)
        rt = _runtime(tree)
        manifest = _manifest(
            kv_keys=[f"h{t}" for t in _TOKENS],
            mamba_key=f"h{_TOKENS[-1]}.mamba",
            token_ids=_TOKENS,
        )
        out = rt.handle(
            SessionHandoverReqInput(
                action="verify_import", manifest_json=json.dumps(manifest)
            )
        )
        self.assertFalse(out.success)  # store is empty on the destination
        tree.cache_controller.storage_backend.keys.update(
            manifest["kv_keys"] + [manifest["mamba_key"]]
        )
        out = rt.handle(
            SessionHandoverReqInput(
                action="verify_import", manifest_json=json.dumps(manifest)
            )
        )
        self.assertTrue(out.success, out.message)


if __name__ == "__main__":
    unittest.main()
