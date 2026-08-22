"""#411 Cut 3: an imported bundle seeds a session, or nothing happens at all.

THE ENTRY IS THE EXISTING ONE, deliberately. ``import_bundle_and_seed``
composes the same lifecycle branch/rewind use --
``_preconditions`` -> gate -> ``controller.branch_from`` -> ledger -- rather
than a parallel path. A second seeding path would be a second place for the
mid-generation rule, the TP=1 limit and the page_head refusal to be got wrong.

THE THREE PROPERTIES, in the order their failure would hurt:

  1. **Nothing partial.** Every refusal precedes ``branch_from``, the only
     step that creates a session. So a failure anywhere leaves no session to
     roll back -- not "a rollback that works", but no state to roll back.
     The mutation moves seeding earlier and these pins go red.
  2. **The C1 direction gets a standing falsifier.** A hybrid bundle whose
     mamba blob is missing must refuse AT THE GATE and never seed a truncated
     prefix. That is the failure Cut 1 would have shipped: a bundle that
     imports cleanly and replays a wrong session.
  3. **Mid-generation target refused.** #410's rule, checked before any store
     IO, because a refusal that has already spent IO is a worse refusal.

Hermetic: a runtime shell with fakes for scheduler, tree and controller. No
server, no CUDA, no store.
"""

import os
import tempfile
import unittest

from sglang.srt.managers.session_checkpoint import (
    SessionCheckpointError,
    SessionCheckpointRuntime,
)
from sglang.srt.managers.session_handover import MANIFEST_VERSION
from sglang.srt.managers.session_portable import export_bundle

IDENTITY = "modelhash-abc"
GEOMETRY = {"tp_size": 1, "page_size": 1, "dcp_owner_mode": False}
BLOBS = {"kv0": b"\x00" * 16, "mamba": b"\xab" * 8}


def _manifest(**over):
    m = {
        "version": MANIFEST_VERSION,
        "handover_id": "h1",
        "model_identity_hash": IDENTITY,
        "source": dict(GEOMETRY),
        "token_ids": [7, 8, 9],
        "kv_keys": ["kv0"],
        "mamba_key": "mamba",
        "hybrid_gdn": True,
        "draft_keys": [],
        "checkpoint": {"envelope_version": 1, "session_id": "s-src"},
    }
    m.update(over)
    return m


class _Controller:
    def __init__(self):
        self.calls = []

    def branch_from(self, *, parent_session_id, checkpoint_tokens, new_session_id):
        self.calls.append((parent_session_id, list(checkpoint_tokens), new_session_id))
        return new_session_id or "s-new"


class _Ledger:
    def __init__(self):
        self.derived = []

    def derive(self, session_id, checkpoint_id):
        self.derived.append((session_id, checkpoint_id))


class _Runtime(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.controller = _Controller()
        self.store = {}

        rt = SessionCheckpointRuntime.__new__(SessionCheckpointRuntime)
        rt.scheduler = type(
            "S",
            (),
            {
                "session_controller": self.controller,
                "running_batch": None,
                "waiting_queue": [],
                "chunked_req": None,
            },
        )()
        rt.ledger = _Ledger()
        rt._preconditions = lambda: "TREE"
        rt._identity = lambda: IDENTITY
        rt._local_geometry = lambda: dict(GEOMETRY)
        rt._in_flight = None
        rt._in_flight_conflict = lambda sid: rt._in_flight
        self.rt = rt

        # backend_exists_fn(tree) is patched to read our fake store, so
        # verify_restore judges what the import actually materialised.
        import sglang.srt.managers.session_checkpoint as m

        self._orig_exists = m.backend_exists_fn
        m.backend_exists_fn = lambda tree: lambda key: key in self.store
        self.addCleanup(setattr, m, "backend_exists_fn", self._orig_exists)

    def _bundle(self, manifest=None, blobs=None, name="b.tar"):
        blobs = BLOBS if blobs is None else blobs
        path = os.path.join(self.dir, name)
        export_bundle(manifest or _manifest(), lambda k: blobs.get(k), path)
        return path

    def _seed(self, path, **kw):
        return self.rt.import_bundle_and_seed(
            path, store_put_fn=lambda k, v: self.store.__setitem__(k, v), **kw
        )


class TestTheHappyPathSeedsThroughTheExistingEntry(_Runtime):
    def test_it_seeds_and_returns_the_session(self):
        seeded, detail = self._seed(self._bundle(), new_session_id="s-imported")
        self.assertEqual(seeded, "s-imported")
        self.assertIn("import verified", detail)

    def test_it_uses_the_branch_entry_with_the_manifest_tokens(self):
        self._seed(self._bundle())
        self.assertEqual(len(self.controller.calls), 1)
        parent, tokens, _new = self.controller.calls[0]
        self.assertEqual(parent, "s-src")
        self.assertEqual(tokens, [7, 8, 9])

    def test_every_blob_reaches_the_store(self):
        self._seed(self._bundle())
        self.assertEqual(set(self.store), set(BLOBS))


class TestNothingPartial(_Runtime):
    """Every refusal precedes branch_from, the only step that creates a
    session. There is nothing to roll back because nothing was seeded."""

    def test_an_identity_mismatch_seeds_nothing(self):
        self.rt._identity = lambda: "some-other-model"
        with self.assertRaises(Exception):
            self._seed(self._bundle())
        self.assertEqual(self.controller.calls, [], "a session was created anyway")

    def test_a_geometry_mismatch_seeds_nothing(self):
        self.rt._local_geometry = lambda: {**GEOMETRY, "tp_size": 4}
        with self.assertRaises(Exception):
            self._seed(self._bundle())
        self.assertEqual(self.controller.calls, [])

    def test_a_store_that_did_not_take_the_blobs_seeds_nothing(self):
        """The second verification exists for exactly this: the bundle was
        complete, and this rank still cannot read what it was handed."""
        with self.assertRaises(SessionCheckpointError) as ctx:
            self.rt.import_bundle_and_seed(self._bundle(), store_put_fn=None)
        self.assertIn("did not verify against this rank's store", str(ctx.exception))
        self.assertEqual(self.controller.calls, [])

    def test_a_missing_controller_seeds_nothing(self):
        self.rt.scheduler.session_controller = None
        with self.assertRaises(SessionCheckpointError):
            self._seed(self._bundle())


class TestTheC1FalsifierStands(_Runtime):
    """A hybrid bundle without its recurrent state must refuse at the gate.

    This is the failure Cut 1 would have shipped: KV pages travelled, the
    mamba blob did not, and the destination silently re-prefilled a wrong
    session. It is a standing falsifier, not a one-off regression pin.
    """

    def test_a_hybrid_bundle_missing_its_mamba_blob_refuses(self):
        # export refuses to build one, so the falsifier constructs the bad
        # bundle by declaring the blob and not shipping it.
        manifest = _manifest(mamba_key="mamba_absent")
        with self.assertRaises(Exception):
            export_bundle(
                manifest,
                lambda k: BLOBS.get(k),
                os.path.join(self.dir, "nogdn.tar"),
            )

    def test_a_hybrid_manifest_without_a_key_never_seeds(self):
        path = self._bundle(_manifest(mamba_key=None), {"kv0": BLOBS["kv0"]})
        with self.assertRaises(Exception):
            self._seed(path)
        self.assertEqual(self.controller.calls, [], "a truncated prefix was seeded")


class TestMidGenerationIsRefused(_Runtime):
    """#410's rule, checked before any store IO."""

    def test_an_in_flight_session_is_refused_by_name(self):
        self.rt._in_flight = "rid-42"
        with self.assertRaises(SessionCheckpointError) as ctx:
            self._seed(self._bundle())
        msg = str(ctx.exception)
        self.assertIn("in flight", msg)
        self.assertIn("rid-42", msg)

    def test_the_refusal_happens_before_any_store_write(self):
        """A refusal that has already spent store IO is a worse refusal."""
        self.rt._in_flight = "rid-42"
        with self.assertRaises(SessionCheckpointError):
            self._seed(self._bundle())
        self.assertEqual(self.store, {}, "store IO was spent before refusing")
        self.assertEqual(self.controller.calls, [])


class TestIntegrityHarvestedFromTheWip(_Runtime):
    """Per-payload digests, harvested from the paused #411 WIP (0dc48c92d8).

    Content-addressed keys do not give this for free: the key names what the
    bytes SHOULD be, and nothing re-checks that they are.
    """

    def test_a_corrupted_payload_refuses_and_seeds_nothing(self):
        import io
        import json
        import tarfile

        src = self._bundle(name="src.tar")
        with tarfile.open(src, "r") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        members["pages/kv0"] = b"\x99" * 16  # same length, wrong bytes
        path = os.path.join(self.dir, "corrupt.tar")
        with tarfile.open(path, "w") as tar:
            for name, blob in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))

        with self.assertRaises(Exception) as ctx:
            self._seed(path)
        self.assertIn("digest", str(ctx.exception))
        self.assertEqual(self.controller.calls, [])
        del json


if __name__ == "__main__":
    unittest.main()
