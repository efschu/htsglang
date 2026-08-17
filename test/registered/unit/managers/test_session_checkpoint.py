"""Hermetic proof of server-side conversation checkpoints (#410).

No GPU, no server, no collective: the scheduler, radix tree, storage backend
and memory-tier registry are injected fakes and fixtures, so what is proved
here is the CONTRACT -- manifest round-trip, the restore gates, the branch
accounting and the session splice.

Every gate is exercised in BOTH directions; a gate that cannot fail proves
nothing. The can-fail halves, by group:

* MANIFEST ROUND-TRIP: a checkpoint manifest survives JSON and is still a
  valid #261 manifest -- ``verify_import`` reads it without knowing #410
  exists. There is one serialization in this fork, and this test is what
  keeps that true.
* GDN GATE (#212): a hybrid-GDN export whose mamba blob never reached the
  store MUST be refused; the identical export with the blob present passes.
  The hybrid flag comes from ``supports_mamba()`` and nothing else.
* REFUSAL MATRIX: geometry (tp_size, page_size, dcp_owner_mode), model
  identity (#241: model/dtype/quantization/kv-cache dtype), a missing blob,
  and an absent tier -- each refused by name, each with the passing twin.
* BRANCH ACCOUNTING: ``copied`` is structurally zero, ``shared`` is what the
  tree still holds, ``prefetch`` is the honest remainder, and the sum is the
  checkpoint. A tree that has evicted half the prefix reports half.
* SESSION SPLICE: a rewind moves the continuation point and is consumed
  once; a branch never mutates the parent's token arrays.

    python -m pytest test/registered/unit/managers/test_session_checkpoint.py -v
"""

import json
import queue
import types
import unittest

import torch

from sglang.srt.managers.io_struct import SessionCheckpointReqInput
from sglang.srt.managers.session_checkpoint import (
    CHECKPOINT_ENVELOPE_VERSION,
    BranchAccounting,
    CheckpointLedger,
    CheckpointRecord,
    SessionCheckpointError,
    SessionCheckpointRuntime,
    account_branch,
    build_checkpoint_manifest,
    checkpoint_id_for,
    geometry_of,
    verify_geometry,
    verify_restore,
)
from sglang.srt.managers.session_handover import build_manifest, verify_import
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.memtier.registry import TierRegistry
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
)
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = 1024**3
_IDENTITY = "id-checkpoint-01"
_TOKENS = [11, 12, 13, 14]
_GEOMETRY = {"tp_size": 1, "page_size": 1, "dcp_owner_mode": False}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_manifest(
    kv_keys=("h11", "h12", "h13", "h14"),
    mamba_key="h14.mamba",
    hybrid_gdn=True,
    identity=_IDENTITY,
    token_ids=tuple(_TOKENS),
    tp_size=1,
    page_size=1,
    dcp_owner_mode=False,
):
    return build_manifest(
        handover_id=checkpoint_id_for(token_ids, identity),
        model_identity_hash=identity,
        token_ids=list(token_ids),
        kv_keys=list(kv_keys),
        mamba_key=mamba_key,
        hybrid_gdn=hybrid_gdn,
        draft_keys=[],
        tp_size=tp_size,
        tp_rank=0,
        dcp_owner_mode=dcp_owner_mode,
        page_size=page_size,
    )


def _checkpoint_manifest(**kwargs):
    base = _base_manifest(**kwargs)
    return build_checkpoint_manifest(
        checkpoint_id=base["handover_id"],
        session_id="sess-a",
        base_manifest=base,
        tier_id="host:unit",
        tier_provenance="measured",
        tier_alternatives=("fs:unit:/fast",),
    )


def _exists_all(manifest):
    present = set(manifest["kv_keys"])
    if manifest.get("mamba_key"):
        present.add(manifest["mamba_key"])
    return lambda key: key in present


def _tier(tier_id, kind, *, volatility, total=100 * GIB, bandwidth=40.0):
    return TierDescriptor(
        id=tier_id,
        kind=kind,
        host="unit",
        capacity=TierCapacity(
            total=Rate.measured(float(total), "unit fixture", unit="bytes"),
            floor=Rate.measured(0.0, "unit fixture", unit="bytes"),
        ),
        volatility=volatility,
        caps=TierCaps(
            latency_us=Rate.absent("not measured", unit="us"),
            bandwidth_gbs=Rate.measured(bandwidth, "unit fixture", unit="GB/s"),
            aperture_bytes=Rate.absent("not an aperture tier", unit="bytes"),
            ledger_key=tier_id.split(":")[0],
        ),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="unit"),
        profile_id="unit",
    )


_HOST_TIER = _tier("host:unit", TierKind.HOST, volatility=Volatility.EXPENSIVE_OK)


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
        self.lock_ref = 0
        self.hash_value = [f"h{t}" for t in tokens]


class _FakeTree:
    """Storage-enabled radix tree fake with the lock-reference API a branch
    uses. ``resident`` shrinks the prefix the tree still holds, which is how
    the prefetch half of the branch accounting is falsified."""

    def __init__(self, token_ids, hybrid=True, withhold_mamba=False, resident=None):
        self.enable_storage = True
        self.root_node = _FakeNode([], None)
        mid = len(token_ids) // 2 or 1
        n1 = _FakeNode(token_ids[:mid], self.root_node)
        n2 = _FakeNode(token_ids[mid:], n1)
        self.leaf = n2 if n2.key else n1
        self.chain = [n for n in (n1, n2) if n.key]
        self._resident = len(token_ids) if resident is None else resident
        self._full_match = MatchResult(
            device_indices=torch.arange(len(token_ids)),
            last_device_node=self.leaf,
            last_host_node=self.leaf,
            best_match_node=self.leaf,
        )
        self._partial_match = MatchResult(
            device_indices=torch.arange(self._resident),
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
        self.token_to_kv_pool_allocator = None

    def supports_mamba(self):
        return self._hybrid

    def match_prefix(self, params):
        return self._partial_match

    def evict_to(self, resident):
        """Shrink the resident prefix AFTER a checkpoint was taken -- which is
        the only order this can happen in: the export refuses a partial
        prefix, and the eviction that makes a branch prefetch comes later."""
        self._resident = resident
        self._partial_match = MatchResult(
            device_indices=torch.arange(resident),
            last_device_node=self.leaf,
            last_host_node=self.leaf,
            best_match_node=self.leaf,
        )

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

    def inc_lock_ref(self, node):
        node.lock_ref += 1

    def dec_lock_ref(self, node, params=None):
        node.lock_ref -= 1


class _FakeReq:
    def __init__(self, rid, origin_input_ids, output_ids=(), session=None):
        self.rid = rid
        self.origin_input_ids = list(origin_input_ids)
        self.origin_input_ids_unpadded = list(origin_input_ids)
        self.output_ids = list(output_ids)
        self.session = session


class _FakeSessionNode:
    def __init__(self, req):
        self.req = req


class _FakeSession:
    def __init__(self, session_id, req):
        self.session_id = session_id
        self.req_nodes = {req.rid: _FakeSessionNode(req)}
        self.pending_rewind_offset = None
        self.capacity_of_str_len = 0
        self.streaming = True
        self.timeout = None
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids)
        self.committed_fill_len = len(req.origin_input_ids)


class _FakeSessionController:
    def __init__(self, sessions):
        self.sessions = sessions
        self.branches = []
        self.rewinds = []

    def get(self, session_id):
        return self.sessions.get(session_id)

    def branch_from(self, *, parent_session_id, checkpoint_tokens, new_session_id=None):
        parent = self.sessions[parent_session_id]
        sid = new_session_id or f"{parent_session_id}-branch"
        child = _FakeSession(sid, parent.req_nodes[next(iter(parent.req_nodes))].req)
        child.pending_rewind_offset = len(checkpoint_tokens)
        self.sessions[sid] = child
        self.branches.append((parent_session_id, sid, len(checkpoint_tokens)))
        return sid

    def rewind_to(self, session_id, checkpoint_tokens):
        session = self.sessions[session_id]
        session.pending_rewind_offset = len(checkpoint_tokens)
        self.rewinds.append((session_id, len(checkpoint_tokens)))
        return len(checkpoint_tokens)


def _fake_scheduler(tree, sessions=None, running=()):
    args = types.SimpleNamespace(
        tp_size=1,
        pp_size=1,
        page_size=1,
        hicache_storage_backend="file",
        hicache_storage_file_path=None,
        # The ServerArgs default. page_head is refused (#441a); see
        # TestRuntimeCheckpoint.test_the_page_head_host_layout_is_refused.
        hicache_mem_layout="page_first",
        enable_session_checkpoints=True,
        session_checkpoint_vram_max_age_s=60.0,
        session_checkpoint_host_max_age_s=900.0,
    )
    return types.SimpleNamespace(
        server_args=args,
        tree_cache=tree,
        running_batch=types.SimpleNamespace(reqs=list(running)),
        waiting_queue=[],
        chunked_req=None,
        session_controller=_FakeSessionController(sessions or {}),
    )


def _runtime(tree, sessions=None, running=(), tiers=(_HOST_TIER,)):
    scheduler = _fake_scheduler(tree, sessions=sessions, running=running)
    rt = SessionCheckpointRuntime(scheduler)
    rt._identity_hash = _IDENTITY  # bypass the server_args-derived hash
    rt._registry = TierRegistry(list(tiers), profile_id="unit", local_host="unit")
    return rt


def _session_fixture(session_id="sess-a", tokens=_TOKENS):
    req = _FakeReq("rid-0", tokens[:-1], tokens[-1:])
    session = _FakeSession(session_id, req)
    req.session = session
    return {session_id: session}, req


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


class TestManifestRoundTrip(CustomTestCase):
    def test_a_checkpoint_manifest_survives_json(self):
        manifest = _checkpoint_manifest()
        restored = json.loads(json.dumps(manifest))
        self.assertEqual(restored, manifest)
        envelope = restored["checkpoint"]
        self.assertEqual(envelope["envelope_version"], CHECKPOINT_ENVELOPE_VERSION)
        self.assertEqual(envelope["checkpoint_id"], manifest["handover_id"])
        self.assertEqual(envelope["tier_id"], "host:unit")
        self.assertEqual(envelope["tier_provenance"], "measured")

    def test_the_envelope_is_additive_and_a_261_reader_still_accepts_it(self):
        """One serialization, not two: ``verify_import`` knows nothing about
        #410 and must still validate a checkpoint manifest."""
        manifest = _checkpoint_manifest()
        ok, message = verify_import(manifest, _exists_all(manifest), _IDENTITY)
        self.assertTrue(ok, message)
        # ...and every #261 key is byte-identical to what build_manifest wrote.
        base = _base_manifest()
        for key in base:
            if key == "created_unix":
                continue
            self.assertEqual(manifest[key], base[key], key)

    def test_checkpoint_ids_are_content_addressed(self):
        first = checkpoint_id_for(_TOKENS, _IDENTITY)
        self.assertEqual(first, checkpoint_id_for(list(_TOKENS), _IDENTITY))
        # can-fail: a different prefix, or a different model, is a different id.
        self.assertNotEqual(first, checkpoint_id_for(_TOKENS + [15], _IDENTITY))
        self.assertNotEqual(first, checkpoint_id_for(_TOKENS, "other-identity"))


# ---------------------------------------------------------------------------
# Restore gates: geometry, identity, blobs
# ---------------------------------------------------------------------------


class TestGeometryGate(CustomTestCase):
    def test_same_geometry_passes(self):
        manifest = _checkpoint_manifest()
        ok, message = verify_geometry(manifest, _GEOMETRY)
        self.assertTrue(ok, message)
        self.assertEqual(geometry_of(manifest), _GEOMETRY)

    def test_a_different_tp_size_is_refused_by_name(self):
        manifest = _checkpoint_manifest(tp_size=2)
        ok, message = verify_geometry(manifest, _GEOMETRY)
        self.assertFalse(ok)
        self.assertIn("tp_size: checkpoint 2 vs this server 1", message)
        self.assertIn("umsharder", message)
        self.assertIn("Refusing to convert silently", message)

    def test_a_different_page_size_is_refused_by_name(self):
        ok, message = verify_geometry(_checkpoint_manifest(page_size=16), _GEOMETRY)
        self.assertFalse(ok)
        self.assertIn("page_size: checkpoint 16 vs this server 1", message)

    def test_a_different_dcp_owner_mode_is_refused_by_name(self):
        ok, message = verify_geometry(
            _checkpoint_manifest(dcp_owner_mode=True), _GEOMETRY
        )
        self.assertFalse(ok)
        self.assertIn("dcp_owner_mode", message)

    def test_every_mismatched_field_is_named_at_once(self):
        ok, message = verify_geometry(
            _checkpoint_manifest(tp_size=4, page_size=32), _GEOMETRY
        )
        self.assertFalse(ok)
        self.assertIn("page_size", message)
        self.assertIn("tp_size", message)


class TestRefusalMatrix(CustomTestCase):
    """Identity, geometry and blob presence, each in both directions."""

    def test_the_happy_path_passes_all_gates(self):
        manifest = _checkpoint_manifest()
        ok, message = verify_restore(
            manifest, _exists_all(manifest), _IDENTITY, _GEOMETRY
        )
        self.assertTrue(ok, message)
        self.assertIn("GDN state present", message)
        self.assertIn("geometry matches", message)

    def test_a_model_identity_mismatch_is_refused_before_geometry(self):
        manifest = _checkpoint_manifest()
        ok, message = verify_restore(
            manifest, _exists_all(manifest), "a-different-model", _GEOMETRY
        )
        self.assertFalse(ok)
        self.assertIn("model identity mismatch", message)

    def test_a_kv_dtype_change_reads_as_an_identity_mismatch(self):
        """#241: the identity hash covers model, revision, dtype, quantization
        and kv-cache dtype, so a --kv-cache-dtype change is a clean refusal
        rather than a silent read of pages in another byte format."""
        manifest = _checkpoint_manifest(identity="hash-under-fp8-kv")
        ok, message = verify_restore(
            manifest, _exists_all(manifest), "hash-under-bf16-kv", _GEOMETRY
        )
        self.assertFalse(ok)
        self.assertIn("never model/dtype/kv-format", message)

    def test_a_missing_kv_blob_is_refused(self):
        manifest = _checkpoint_manifest()
        present = _exists_all(manifest)
        ok, message = verify_restore(
            manifest, lambda k: present(k) and k != "h12", _IDENTITY, _GEOMETRY
        )
        self.assertFalse(ok)
        self.assertIn("absent from this rank's store", message)

    def test_a_missing_gdn_blob_is_refused_212(self):
        manifest = _checkpoint_manifest()
        present = _exists_all(manifest)
        ok, message = verify_restore(
            manifest,
            lambda k: present(k) and not k.endswith(".mamba"),
            _IDENTITY,
            _GEOMETRY,
        )
        self.assertFalse(ok)
        self.assertIn("h14.mamba", message)

    def test_a_non_hybrid_model_needs_no_gdn_blob(self):
        """can-fail twin of the #212 gate: with supports_mamba() False the
        same manifest without a mamba key passes."""
        manifest = _checkpoint_manifest(hybrid_gdn=False, mamba_key=None)
        ok, message = verify_restore(
            manifest, _exists_all(manifest), _IDENTITY, _GEOMETRY
        )
        self.assertTrue(ok, message)
        self.assertNotIn("GDN state present", message)


# ---------------------------------------------------------------------------
# Branch accounting
# ---------------------------------------------------------------------------


class TestBranchAccounting(CustomTestCase):
    def test_a_fully_resident_checkpoint_shares_everything_and_copies_nothing(self):
        accounting = account_branch(total_tokens=4, matched_tokens=4)
        self.assertEqual(accounting.copied, 0)
        self.assertEqual(accounting.shared, 4)
        self.assertEqual(accounting.prefetch, 0)
        self.assertEqual(accounting.shared_fraction, 1.0)

    def test_an_evicted_prefix_is_reported_as_prefetch_not_as_a_copy(self):
        accounting = account_branch(total_tokens=4, matched_tokens=1)
        self.assertEqual(accounting.copied, 0)
        self.assertEqual(accounting.shared, 1)
        self.assertEqual(accounting.prefetch, 3)
        self.assertEqual(accounting.shared + accounting.prefetch, accounting.total)

    def test_copied_is_structurally_zero(self):
        for matched in range(5):
            self.assertEqual(account_branch(4, min(matched, 4)).copied, 0)

    def test_an_impossible_match_is_refused_rather_than_reported_negative(self):
        with self.assertRaises(SessionCheckpointError) as ctx:
            account_branch(total_tokens=2, matched_tokens=5)
        self.assertIn("cannot be longer than", str(ctx.exception))

    def test_the_accounting_serialises(self):
        payload = BranchAccounting(total=4, shared=3, prefetch=1).to_json()
        self.assertEqual(payload["copied_pages"], 0)
        self.assertEqual(payload["shared_pages"], 3)
        self.assertEqual(payload["prefetch_pages"], 1)
        self.assertEqual(payload["shared_fraction"], 0.75)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def _record(checkpoint_id="cp-1", session_id="sess-a"):
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        session_id=session_id,
        token_ids=list(_TOKENS),
        manifest=_checkpoint_manifest(),
    )


class TestLedger(CustomTestCase):
    def test_overlapping_prefixes_are_the_normal_case_not_a_conflict(self):
        """#261's ledger refuses a second active record on an overlapping
        prefix, which is right for a transfer of ownership and wrong for a
        checkpoint: every checkpoint of one conversation is a prefix of the
        next."""
        ledger = CheckpointLedger()
        ledger.add(_record("cp-1"))
        ledger.add(_record("cp-2"))
        self.assertEqual(len(ledger.list_for("sess-a")), 2)

    def test_adding_the_same_content_addressed_id_is_idempotent(self):
        ledger = CheckpointLedger()
        first = ledger.add(_record("cp-1"))
        first.label = "kept"
        second = ledger.add(_record("cp-1"))
        self.assertIs(first, second)
        self.assertEqual(second.label, "kept")

    def test_an_unknown_id_names_the_known_ones(self):
        ledger = CheckpointLedger()
        ledger.add(_record("cp-1"))
        with self.assertRaises(SessionCheckpointError) as ctx:
            ledger.get("cp-nope")
        self.assertIn("cp-1", str(ctx.exception))

    def test_many_sessions_may_derive_from_one_checkpoint(self):
        ledger = CheckpointLedger()
        ledger.add(_record("cp-1"))
        ledger.derive("branch-a", "cp-1")
        ledger.derive("branch-b", "cp-1")
        self.assertEqual(ledger.get("cp-1").derived_sessions, ["branch-a", "branch-b"])
        self.assertEqual(ledger.pinned_checkpoint_ids(), ("cp-1",))

    def test_a_second_derivation_supersedes_the_first(self):
        ledger = CheckpointLedger()
        ledger.add(_record("cp-1"))
        ledger.add(_record("cp-2"))
        ledger.derive("sess-a", "cp-1")
        ledger.derive("sess-a", "cp-2")
        self.assertEqual(ledger.get("cp-1").derived_sessions, [])
        self.assertEqual(ledger.get("cp-2").derived_sessions, ["sess-a"])
        self.assertEqual(ledger.derivation_of("sess-a"), "cp-2")

    def test_dropping_a_derived_checkpoint_is_refused(self):
        ledger = CheckpointLedger()
        ledger.add(_record("cp-1"))
        ledger.derive("branch-a", "cp-1")
        with self.assertRaises(SessionCheckpointError) as ctx:
            ledger.drop("cp-1")
        self.assertIn("branch-a", str(ctx.exception))
        # can-fail twin: release the derivation and the drop succeeds.
        ledger.release("branch-a")
        self.assertEqual(ledger.drop("cp-1").checkpoint_id, "cp-1")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class TestRuntimeCheckpoint(CustomTestCase):
    def test_happy_path_checkpoint(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertTrue(out.success, out.message)
        manifest = json.loads(out.manifest_json)
        self.assertEqual(manifest["kv_keys"], [f"h{t}" for t in _TOKENS])
        self.assertEqual(manifest["mamba_key"], f"h{_TOKENS[-1]}.mamba")
        self.assertTrue(manifest["hybrid_gdn"])
        self.assertEqual(manifest["checkpoint"]["session_id"], "sess-a")
        self.assertEqual(manifest["checkpoint"]["tier_id"], "host:unit")
        self.assertEqual(out.checkpoint_id, manifest["handover_id"])

    def test_the_gdn_blob_is_mandatory_for_a_hybrid_model_212(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS, withhold_mamba=True), sessions=sessions)
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("#212", out.message)
        self.assertIn("mamba", out.message.lower())
        self.assertEqual(rt.ledger.list_for("sess-a"), [])

    def test_the_same_export_passes_once_the_gdn_blob_is_there(self):
        """can-fail twin of the test above: one flag apart on the same fake."""
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS, withhold_mamba=False), sessions=sessions)
        self.assertTrue(
            rt.handle(
                SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
            ).success
        )

    def test_a_non_hybrid_model_needs_no_gdn_blob(self):
        sessions, _ = _session_fixture()
        rt = _runtime(
            _FakeTree(_TOKENS, hybrid=False, withhold_mamba=True), sessions=sessions
        )
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertTrue(out.success, out.message)
        self.assertIsNone(json.loads(out.manifest_json)["mamba_key"])

    def test_an_in_flight_turn_refuses_the_checkpoint(self):
        sessions, req = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions, running=[req])
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("in-flight request (rid-0)", out.message)

    def test_another_sessions_in_flight_turn_does_not_block_this_one(self):
        """can-fail twin: the same running request, a different session."""
        sessions, _ = _session_fixture()
        other_sessions, other_req = _session_fixture("sess-b")
        sessions.update(other_sessions)
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions, running=[other_req])
        self.assertTrue(
            rt.handle(
                SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
            ).success
        )

    def test_an_unknown_session_is_refused(self):
        rt = _runtime(_FakeTree(_TOKENS), sessions={})
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="nope")
        )
        self.assertFalse(out.success)
        self.assertIn("unknown session", out.message)

    def test_the_feature_flag_gates_every_action(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        rt.scheduler.server_args.enable_session_checkpoints = False
        for action in ("checkpoint", "branch", "rewind"):
            out = rt.handle(
                SessionCheckpointReqInput(
                    action=action, session_id="sess-a", checkpoint_id="cp-1"
                )
            )
            self.assertFalse(out.success, action)
            self.assertIn("--enable-session-checkpoints", out.message)

    def test_tp_greater_than_one_is_refused_by_name(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        rt.scheduler.server_args.tp_size = 2
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("TP=1", out.message)
        self.assertIn("#261", out.message)

    def test_a_page_size_above_one_is_refused_by_name(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        rt.scheduler.server_args.page_size = 16
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("page_size == 1", out.message)

    def test_the_page_head_host_layout_is_refused_by_name(self):
        """#441a: page_head's host write-back takes the lf_ph route, which
        segfaults. Refused at argument time too; repeated here because a
        server can be reconfigured after boot.

        can-fail twin: the layouts that reach the working
        transfer_kv_all_layer_direct_lf_pf / layer-first routes still pass.
        """
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        rt.scheduler.server_args.hicache_mem_layout = "page_head"
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("page_head", out.message)
        self.assertIn("#441a", out.message)
        for layout in ("page_first", "page_first_direct", "layer_first"):
            sessions, _ = _session_fixture()
            ok = _runtime(_FakeTree(_TOKENS), sessions=sessions)
            ok.scheduler.server_args.hicache_mem_layout = layout
            got = ok.handle(
                SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
            )
            self.assertTrue(got.success, f"{layout}: {got.message}")

    def test_an_absent_tier_is_a_named_refusal_not_a_guessed_target(self):
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions, tiers=())
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("no memory tier can hold this checkpoint", out.message)
        self.assertIn("#407", out.message)

    def test_a_durable_checkpoint_refuses_a_ram_only_rig(self):
        """can-fail twin: the same rig serves a non-durable checkpoint."""
        sessions, _ = _session_fixture()
        rt = _runtime(_FakeTree(_TOKENS), sessions=sessions)
        durable = rt.handle(
            SessionCheckpointReqInput(
                action="checkpoint", session_id="sess-a", durable=True
            )
        )
        self.assertFalse(durable.success)
        self.assertIn("persistence_required", durable.message)
        loose = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertTrue(loose.success, loose.message)


class TestRuntimeBranchAndRewind(CustomTestCase):
    def _checkpointed(self, tree=None, sessions=None):
        if sessions is None:
            sessions, _ = _session_fixture()
        rt = _runtime(tree or _FakeTree(_TOKENS), sessions=sessions)
        out = rt.handle(
            SessionCheckpointReqInput(action="checkpoint", session_id="sess-a")
        )
        self.assertTrue(out.success, out.message)
        return rt, out.checkpoint_id

    def test_branch_opens_a_new_session_and_shares_every_page(self):
        rt, cp = self._checkpointed()
        out = rt.handle(
            SessionCheckpointReqInput(
                action="branch",
                session_id="sess-a",
                checkpoint_id=cp,
                new_session_id="sess-branch",
            )
        )
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.session_id, "sess-branch")
        accounting = out.info["accounting"]
        self.assertEqual(accounting["copied_pages"], 0)
        self.assertEqual(accounting["shared_pages"], len(_TOKENS))
        self.assertEqual(accounting["prefetch_pages"], 0)
        # The shared chain is pinned so eviction cannot take it out from
        # under a branch that has not run its first turn.
        self.assertEqual(rt.scheduler.tree_cache.leaf.lock_ref, 1)
        self.assertEqual(rt.ledger.derivation_of("sess-branch"), cp)

    def test_branch_of_a_partly_evicted_checkpoint_reports_the_prefetch(self):
        """can-fail twin of the accounting: the same branch on a tree that
        holds only one of the four pages."""
        rt, cp = self._checkpointed()
        rt.scheduler.tree_cache.evict_to(1)
        out = rt.handle(
            SessionCheckpointReqInput(
                action="branch", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertTrue(out.success, out.message)
        accounting = out.info["accounting"]
        self.assertEqual(accounting["shared_pages"], 1)
        self.assertEqual(accounting["prefetch_pages"], 3)
        self.assertEqual(accounting["copied_pages"], 0)

    def test_branching_sets_the_childs_splice_point_and_not_the_parents(self):
        rt, cp = self._checkpointed()
        rt.handle(
            SessionCheckpointReqInput(
                action="branch",
                session_id="sess-a",
                checkpoint_id=cp,
                new_session_id="sess-branch",
            )
        )
        controller = rt.scheduler.session_controller
        self.assertEqual(
            controller.sessions["sess-branch"].pending_rewind_offset, len(_TOKENS)
        )
        self.assertIsNone(controller.sessions["sess-a"].pending_rewind_offset)

    def test_two_branches_of_one_checkpoint_pin_it_once(self):
        rt, cp = self._checkpointed()
        for name in ("b1", "b2"):
            self.assertTrue(
                rt.handle(
                    SessionCheckpointReqInput(
                        action="branch",
                        session_id="sess-a",
                        checkpoint_id=cp,
                        new_session_id=name,
                    )
                ).success
            )
        self.assertEqual(rt.scheduler.tree_cache.leaf.lock_ref, 1)
        self.assertEqual(len(rt.ledger.get(cp).derived_sessions), 2)

    def test_rewind_moves_the_continuation_point(self):
        rt, cp = self._checkpointed()
        out = rt.handle(
            SessionCheckpointReqInput(
                action="rewind", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertTrue(out.success, out.message)
        self.assertIn("no re-prefill", out.message)
        controller = rt.scheduler.session_controller
        self.assertEqual(controller.rewinds, [("sess-a", len(_TOKENS))])
        self.assertEqual(
            controller.sessions["sess-a"].pending_rewind_offset, len(_TOKENS)
        )

    def test_rewind_is_refused_while_a_turn_is_in_flight(self):
        sessions, req = _session_fixture()
        rt, cp = self._checkpointed(sessions=sessions)
        rt.scheduler.running_batch.reqs.append(req)
        out = rt.handle(
            SessionCheckpointReqInput(
                action="rewind", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertFalse(out.success)
        self.assertIn("rewind refused", out.message)

    def test_a_cross_geometry_checkpoint_is_refused_on_restore(self):
        rt, cp = self._checkpointed()
        # The server was reconfigured after the checkpoint was written -- the
        # case a PERSISTENT tier makes reachable across boots.
        rt.scheduler.server_args.page_size = 1
        rt.ledger.get(cp).manifest["source"]["tp_size"] = 2
        out = rt.handle(
            SessionCheckpointReqInput(
                action="branch", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertFalse(out.success)
        self.assertIn("geometry does not match", out.message)
        self.assertIn("umsharder", out.message)

    def test_an_unknown_checkpoint_id_is_refused(self):
        rt, _ = self._checkpointed()
        out = rt.handle(
            SessionCheckpointReqInput(
                action="branch", session_id="sess-a", checkpoint_id="cp-nope"
            )
        )
        self.assertFalse(out.success)
        self.assertIn("unknown checkpoint id", out.message)

    def test_list_and_drop(self):
        rt, cp = self._checkpointed()
        listed = rt.handle(
            SessionCheckpointReqInput(action="list", session_id="sess-a")
        )
        self.assertTrue(listed.success)
        self.assertEqual(len(listed.info["checkpoints"]), 1)
        self.assertEqual(listed.info["checkpoints"][0]["checkpoint_id"], cp)

        rt.handle(
            SessionCheckpointReqInput(
                action="branch",
                session_id="sess-a",
                checkpoint_id=cp,
                new_session_id="b1",
            )
        )
        blocked = rt.handle(
            SessionCheckpointReqInput(
                action="drop", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertFalse(blocked.success)
        self.assertIn("derived session", blocked.message)

        rt.ledger.release("b1")
        dropped = rt.handle(
            SessionCheckpointReqInput(
                action="drop", session_id="sess-a", checkpoint_id=cp
            )
        )
        self.assertTrue(dropped.success, dropped.message)
        # Dropping releases the lock reference the branch took.
        self.assertEqual(rt.scheduler.tree_cache.leaf.lock_ref, 0)

    def test_an_unknown_action_names_the_known_ones(self):
        rt, _ = self._checkpointed()
        out = rt.handle(
            SessionCheckpointReqInput(action="teleport", session_id="sess-a")
        )
        self.assertFalse(out.success)
        self.assertIn("checkpoint, branch, rewind, list, drop", out.message)


if __name__ == "__main__":
    unittest.main()
