"""#411: a session checkpoint as a portable file, gated before any bytes move.

CUT 2 REWROTE THIS FILE, and the reason is the prior-art gate. Cut 1 built
against DESIGN_410's PROSE: it read ``page_hashes`` and hand-rolled its own
version and identity checks. The code says otherwise, in one place, and the
code is the authority:

  * the manifest ``session_handover.build_manifest`` actually writes carries
    ``kv_keys``, ``mamba_key`` and ``draft_keys`` -- not ``page_hashes``;
  * ``session_handover.verify_import`` already checks version, identity,
    blob presence AND the #212 hybrid-GDN clause;
  * ``session_checkpoint.verify_geometry`` already owns the geometry axis,
    which ``verify_import`` deliberately omits.

So this module now DELEGATES both gates and only supplies the container. The
correction is not cosmetic: exporting a hybrid session without its
``mamba_key`` yields a bundle that imports cleanly and replays a WRONG
session, because a missing recurrent state truncates the prefix match at the
destination and silently re-prefills. That is #212 verbatim.

WHAT EACH PIN PREVENTS:

  * **Round-trip byte-identical** -- #261's proof extended to the file.
  * **Every referenced blob travels** -- kv pages, the GDN blob, draft blobs.
    A bundle missing the mamba blob is the #212 failure.
  * **Gate before extraction, on NAMES** -- an incomplete or incompatible
    bundle is refused before a single payload is read, so partial-seed-then-
    fail is not reachable: nothing is returned to seed from.
  * **No holes at export** -- #410's evicted-page rule applies at export too.

Hermetic: tars in a tmpdir, injected page reader. No server, no CUDA.
"""

import io
import json
import os
import tarfile
import tempfile
import unittest

from sglang.srt.managers.session_handover import MANIFEST_VERSION
from sglang.srt.managers.session_portable import (
    IDENTITY_LAYOUT_GAP,
    MANIFEST_MEMBER,
    PAGE_PREFIX,
    PortableSessionError,
    check_compatibility,
    export_bundle,
    import_bundle,
    read_manifest,
    referenced_blobs,
)

IDENTITY = "modelhash-abc"
GEOMETRY = {"tp_size": 3, "page_size": 64, "dcp_owner_mode": False}
BLOBS = {"kv0": b"\x00" * 32, "kv1": b"\x01\x02\x03", "mamba": b"\xff" * 7}


def _manifest(**over):
    m = {
        "version": MANIFEST_VERSION,
        "handover_id": "h1",
        "model_identity_hash": IDENTITY,
        "source": dict(GEOMETRY),
        "token_ids": [1, 2, 3],
        "kv_keys": ["kv0", "kv1"],
        "mamba_key": "mamba",
        "hybrid_gdn": True,
        "draft_keys": [],
    }
    for key, value in over.items():
        if key == "geometry":
            m["source"] = dict(value)
        else:
            m[key] = value
    return m


class _Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _export(self, manifest=None, blobs=None, name="s.tar"):
        blobs = BLOBS if blobs is None else blobs
        path = os.path.join(self.dir, name)
        export_bundle(manifest or _manifest(), lambda k: blobs.get(k), path)
        return path

    def _import(self, path, **over):
        kw = dict(local_identity=IDENTITY, local_geometry=dict(GEOMETRY))
        kw.update(over)
        return import_bundle(path, **kw)


class TestEveryReferencedBlobTravels(_Store):
    """The Cut 1 defect: only kv pages travelled."""

    def test_the_gdn_blob_is_referenced(self):
        self.assertIn("mamba", referenced_blobs(_manifest()))

    def test_draft_blobs_are_referenced(self):
        self.assertIn("d0", referenced_blobs(_manifest(draft_keys=["d0"])))

    def test_the_gdn_blob_is_actually_exported(self):
        """#212: a hybrid bundle without its recurrent state imports cleanly
        and replays a WRONG session."""
        _m, blobs = self._import(self._export())
        self.assertIn("mamba", blobs)
        self.assertEqual(blobs["mamba"], BLOBS["mamba"])

    def test_a_design_prose_manifest_is_still_honoured(self):
        """``page_hashes`` was the design's name; a manifest written to it
        must not silently export nothing."""
        m = {"page_hashes": ["kv0"], "kv_keys": []}
        self.assertEqual(referenced_blobs(m), ["kv0"])

    def test_duplicates_are_not_exported_twice(self):
        m = _manifest(kv_keys=["kv0", "kv0"], mamba_key="kv0")
        self.assertEqual(referenced_blobs(m), ["kv0"])


class TestRoundTripIsExact(_Store):
    def test_blobs_come_back_byte_identical(self):
        _m, blobs = self._import(self._export())
        self.assertEqual(blobs, BLOBS)

    def test_the_manifest_comes_back_unchanged(self):
        m = _manifest()
        got, _blobs = self._import(self._export(m))
        self.assertEqual(got, m)

    def test_an_empty_blob_survives(self):
        blobs = {"kv0": b"", "kv1": b"x", "mamba": b"m"}
        _m, got = self._import(self._export(blobs=blobs, name="e.tar"))
        self.assertEqual(got["kv0"], b"")


class TestGateBeforeExtraction(_Store):
    def test_the_manifest_is_the_first_member(self):
        with tarfile.open(self._export(), "r") as tar:
            self.assertEqual(tar.getnames()[0], MANIFEST_MEMBER)

    def test_read_manifest_extracts_no_payload(self):
        self.assertEqual(read_manifest(self._export())["model_identity_hash"], IDENTITY)

    def test_a_non_bundle_is_refused_by_name(self):
        path = os.path.join(self.dir, "junk.tar")
        with tarfile.open(path, "w") as tar:
            info = tarfile.TarInfo("something_else")
            info.size = 0
            tar.addfile(info)
        with self.assertRaises(PortableSessionError) as ctx:
            read_manifest(path)
        self.assertIn("not a session bundle", str(ctx.exception))


class TestPartialSeedIsUnreachable(_Store):
    """A bundle whose manifest claims a blob the tar lacks must be refused
    with NOTHING returned -- so no caller can have begun seeding."""

    def _doctor(self, name, mutate):
        src = self._export(name="src.tar")
        with tarfile.open(src, "r") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        manifest = json.loads(members[MANIFEST_MEMBER].decode())
        mutate(manifest, members)
        members[MANIFEST_MEMBER] = json.dumps(manifest, sort_keys=True).encode()
        path = os.path.join(self.dir, name)
        with tarfile.open(path, "w") as tar:
            for member_name, blob in members.items():
                info = tarfile.TarInfo(member_name)
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))
        return path

    def test_a_missing_kv_page_is_refused_and_returns_nothing(self):
        def _mutate(manifest, members):
            manifest["kv_keys"] = ["kv0", "kv_absent"]

        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._doctor("miss_kv.tar", _mutate))
        self.assertIn("absent from this rank's store", str(ctx.exception))

    def test_a_missing_gdn_blob_is_refused(self):
        """#212 read-side: the store route would truncate the prefix match
        and silently re-prefill a recurrent state."""

        def _mutate(manifest, members):
            members.pop(f"{PAGE_PREFIX}mamba", None)

        with self.assertRaises(PortableSessionError):
            self._import(self._doctor("miss_gdn.tar", _mutate))

    def test_a_hybrid_manifest_without_a_mamba_key_is_refused(self):
        def _mutate(manifest, members):
            manifest["mamba_key"] = None

        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._doctor("no_key.tar", _mutate))
        self.assertIn("hybrid-GDN manifest without a mamba key", str(ctx.exception))


class TestEveryAxisRefusesByName(_Store):
    def test_wrong_model_identity(self):
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(), local_identity="different-model")
        self.assertIn("model identity mismatch", str(ctx.exception))

    def test_an_identity_refusal_carries_the_layout_gap(self):
        """#726: the hash covers the dtype NAME, not the layout within it."""
        ok, detail = check_compatibility(
            _manifest(), local_identity="other", local_geometry=dict(GEOMETRY)
        )
        self.assertFalse(ok)
        self.assertIn("byte layout within it", detail)
        self.assertIn("group size", IDENTITY_LAYOUT_GAP)

    def test_wrong_geometry_names_the_offline_converter(self):
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(), local_geometry={**GEOMETRY, "tp_size": 2})
        msg = str(ctx.exception)
        self.assertIn("geometry does not match", msg)
        self.assertIn("hicache_migrate", msg)

    def test_unknown_manifest_version(self):
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(_manifest(version=9999)))
        self.assertIn("refusing to guess the schema", str(ctx.exception))

    def test_a_malformed_manifest_is_refused_not_judged_leniently(self):
        ok, detail = check_compatibility(
            {"version": MANIFEST_VERSION},
            local_identity=IDENTITY,
            local_geometry=dict(GEOMETRY),
        )
        self.assertFalse(ok)
        self.assertIn("malformed manifest", detail)


class TestNoHolesAtExport(_Store):
    def test_an_unretrievable_blob_refuses_the_export(self):
        path = os.path.join(self.dir, "holed.tar")
        with self.assertRaises(PortableSessionError) as ctx:
            export_bundle(_manifest(), lambda k: None if k == "kv1" else BLOBS[k], path)
        self.assertIn("not retrievable", str(ctx.exception))

    def test_a_refused_export_leaves_no_file(self):
        path = os.path.join(self.dir, "holed2.tar")
        with self.assertRaises(PortableSessionError):
            export_bundle(_manifest(), lambda k: None, path)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".part"))


class TestTheGatesAreDelegatedNotReimplemented(unittest.TestCase):
    """Source pin: a second authority for a correctness rule is the failure
    this codebase keeps paying for."""

    def test_check_compatibility_calls_both_existing_gates(self):
        import inspect

        from sglang.srt.managers import session_portable as m

        src = inspect.getsource(m.check_compatibility)
        self.assertIn(
            "verify_restore",
            src,
            "the composition of verify_import + verify_geometry is itself an "
            "existing function; re-composing it here would be a second "
            "authority one level up",
        )


if __name__ == "__main__":
    unittest.main()
