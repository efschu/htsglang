"""#411: a session checkpoint as a portable file, gated before any bytes move.

#410 versioned its manifest "for #411" so this would be a CONVERTER rather
than a silent drop. These pins hold the two things that makes true: a
round-trip that preserves content exactly, and a gate that refuses BY NAME on
every axis rather than converting.

WHAT EACH PIN PREVENTS, which is the only reason any of them are here:

  * **Round-trip byte-identical.** #261 proved the handover byte-identical;
    this extends that proof to the FILE boundary. A container that reordered
    or re-encoded payloads would break replay somewhere far away.
  * **Gate before extraction.** The manifest is the first tar member, so an
    incompatible bundle is refused after ONE member read. Refusing after
    unpacking is a worse refusal, and a gate that ran after seeding could
    leave a half-seeded session.
  * **No silent conversion.** Version, identity and geometry each refuse by
    name. Geometry delegates to ``verify_geometry``, which already owns the
    rule and already names the offline umsharder -- re-deriving it here would
    be a second authority for a correctness rule.
  * **No holes.** An export that omitted an evicted page would produce a
    bundle that imports cleanly and decodes WRONG on a machine that cannot
    tell. #410's evicted-page rule applies at export too.

Hermetic: tars in a tmpdir, injected page reader. No server, no CUDA.
"""

import json
import os
import tarfile
import tempfile
import unittest

from sglang.srt.managers.session_checkpoint import CHECKPOINT_ENVELOPE_VERSION
from sglang.srt.managers.session_portable import (
    IDENTITY_LAYOUT_GAP,
    MANIFEST_MEMBER,
    PortableSessionError,
    check_compatibility,
    export_bundle,
    import_bundle,
    read_manifest,
)

IDENTITY = "modelhash-abc"
GEOMETRY = {"tp_size": 3, "page_size": 64, "dcp_owner_mode": False}
PAGES = {"h0": b"\x00" * 32, "h1": b"\x01\x02\x03", "h2": b"\xff" * 7}


def _manifest(**over):
    m = {
        "model_identity": IDENTITY,
        "page_hashes": ["h0", "h1", "h2"],
        "source": dict(GEOMETRY),
        "checkpoint": {
            "envelope_version": CHECKPOINT_ENVELOPE_VERSION,
            "checkpoint_id": "ck1",
            "session_id": "s1",
        },
    }
    for key, value in over.items():
        if key == "geometry":
            m["source"] = dict(value)
        elif key == "envelope_version":
            m["checkpoint"]["envelope_version"] = value
        else:
            m[key] = value
    return m


class _Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _export(self, manifest=None, pages=None, name="s.tar"):
        pages = PAGES if pages is None else pages
        path = os.path.join(self.dir, name)
        export_bundle(
            manifest or _manifest(), lambda h: pages.get(h), path
        )
        return path

    def _import(self, path, **over):
        kw = dict(local_identity=IDENTITY, local_geometry=dict(GEOMETRY))
        kw.update(over)
        return import_bundle(path, **kw)


class TestRoundTripIsExact(_Store):
    def test_pages_come_back_byte_identical(self):
        """The #261 byte-identical proof, extended to the file boundary."""
        _manifest_out, pages = self._import(self._export())
        self.assertEqual(pages, PAGES)

    def test_the_manifest_comes_back_unchanged(self):
        m = _manifest()
        got, _pages = self._import(self._export(m))
        self.assertEqual(got, m)

    def test_an_empty_page_survives(self):
        """A zero-length payload is a real page, not an absent one."""
        pages = {"h0": b""}
        path = self._export(_manifest(page_hashes=["h0"]), pages, name="e.tar")
        _m, got = self._import(path)
        self.assertEqual(got, {"h0": b""})


class TestTheGateRunsBeforeExtraction(_Store):
    def test_the_manifest_is_the_first_member(self):
        """Load-bearing: it is what lets a reader gate on one member."""
        with tarfile.open(self._export(), "r") as tar:
            self.assertEqual(tar.getnames()[0], MANIFEST_MEMBER)

    def test_read_manifest_extracts_no_payload(self):
        got = read_manifest(self._export())
        self.assertEqual(got["model_identity"], IDENTITY)

    def test_a_non_bundle_is_refused_by_name(self):
        path = os.path.join(self.dir, "junk.tar")
        with tarfile.open(path, "w") as tar:
            info = tarfile.TarInfo("something_else")
            info.size = 0
            tar.addfile(info)
        with self.assertRaises(PortableSessionError) as ctx:
            read_manifest(path)
        self.assertIn("not a session bundle", str(ctx.exception))


class TestEveryAxisRefusesByName(_Store):
    def test_wrong_model_identity(self):
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(), local_identity="different-model")
        msg = str(ctx.exception)
        self.assertIn("model identity mismatch", msg)
        self.assertIn("different-model", msg)

    def test_wrong_kv_dtype_is_caught_as_an_identity_mismatch(self):
        """kv-cache dtype lives INSIDE the identity hash, so a differing
        dtype changes the hash and lands here rather than needing its own
        check -- which is why the gate has three axes and not five."""
        bundle = self._export(_manifest(model_identity="hash-with-fp8"))
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(bundle)
        self.assertIn("model identity mismatch", str(ctx.exception))

    def test_wrong_geometry(self):
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(), local_geometry={**GEOMETRY, "tp_size": 2})
        msg = str(ctx.exception)
        self.assertIn("geometry does not match", msg)
        self.assertIn("tp_size", msg)

    def test_a_geometry_refusal_names_the_offline_converter(self):
        """Never a silent conversion: the refusal must say what to run."""
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(self._export(), local_geometry={**GEOMETRY, "page_size": 16})
        self.assertIn("hicache_migrate", str(ctx.exception))

    def test_unknown_envelope_version(self):
        bundle = self._export(_manifest(envelope_version=9999))
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(bundle)
        msg = str(ctx.exception)
        self.assertIn("unknown checkpoint envelope version", msg)
        self.assertIn("9999", msg)

    def test_version_is_checked_before_identity(self):
        """A bundle from an unknown version may not even carry the fields the
        later checks read, so an identity complaint about it would mislead."""
        m = _manifest(envelope_version=9999, model_identity="whatever")
        ok, detail = check_compatibility(
            m, local_identity=IDENTITY, local_geometry=dict(GEOMETRY)
        )
        self.assertFalse(ok)
        self.assertIn("version", detail)
        self.assertNotIn("model identity mismatch", detail)


class TestTheGateNamesWhatItDoesNotCover(unittest.TestCase):
    """#726: the identity hash covers the dtype NAME, not the layout within
    it. A gate that implied full coverage would be worse than one that says
    where it stops."""

    def test_an_identity_refusal_carries_the_layout_gap_note(self):
        ok, detail = check_compatibility(
            _manifest(),
            local_identity="other",
            local_geometry=dict(GEOMETRY),
        )
        self.assertFalse(ok)
        self.assertIn("byte layout within it", detail)

    def test_the_gap_is_stated_as_a_module_constant(self):
        self.assertIn("group size", IDENTITY_LAYOUT_GAP)


class TestNoHoles(_Store):
    def test_an_evicted_page_refuses_the_export(self):
        """#410's evicted-page rule, at export. An omitted page yields a
        bundle that imports cleanly and decodes WRONG."""
        path = os.path.join(self.dir, "holed.tar")
        with self.assertRaises(PortableSessionError) as ctx:
            export_bundle(
                _manifest(), lambda h: None if h == "h1" else PAGES[h], path
            )
        self.assertIn("not retrievable", str(ctx.exception))

    def test_a_refused_export_leaves_no_file(self):
        path = os.path.join(self.dir, "holed2.tar")
        with self.assertRaises(PortableSessionError):
            export_bundle(_manifest(), lambda h: None, path)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".part"))

    def test_a_bundle_missing_a_referenced_page_refuses_at_import(self):
        """Completeness BEFORE seeding, mirroring #410's own rule."""
        # Build a bundle whose manifest claims a page the tar does not hold.
        # It cannot be produced by export_bundle -- that refuses (the pin
        # above) -- so it is doctored here, which is exactly the shape a
        # truncated transfer or a hand-edited bundle would arrive in.
        path = os.path.join(self.dir, "doctored.tar")
        short = os.path.join(self.dir, "short.tar")
        export_bundle(_manifest(page_hashes=["h0"]), lambda h: PAGES[h], short)
        with tarfile.open(short, "r") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        doctored = json.loads(members[MANIFEST_MEMBER].decode())
        doctored["page_hashes"] = ["h0", "h_missing"]
        members[MANIFEST_MEMBER] = json.dumps(doctored, sort_keys=True).encode()
        with tarfile.open(path, "w") as tar:
            for name, blob in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                import io as _io

                tar.addfile(info, _io.BytesIO(blob))
        with self.assertRaises(PortableSessionError) as ctx:
            self._import(path)
        self.assertIn("missing 1 of 2 pages", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
