"""1539 06b: the in-tree stand-ins for the empty models-cache directories and the
frozen host-ledger evidence are what they say they are.

Two things can go silently wrong with a fixture and both end in a test that
passes while checking nothing: the fixture is edited until a red test turns
green, or the overlay that serves it stops serving (the tests then fail loudly
-- or, where they were guarded by ``skipUnless(os.path.exists(...))``, skip).
Pinned here:

1. every fixture file still has the sha256 and size PROVENANCE.json recorded;
2. each model directory's safetensors headers still add up, to the byte, to
   the shard bytes of the real directory (tensor bytes from the headers + the
   8-byte length words + the JSON headers == ``ondisk_safetensors_bytes``);
3. the overlay serves reads below an EMPTY models-cache directory, never below
   one that has its own ``config.json``, never serves writes, and puts every
   patched function back.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import struct
import tempfile
import unittest
from unittest import mock

import model_dir_fixtures_1539 as FX

_DT = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
       "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
_EVIDENCE = os.path.join(FX.HERE, "fixtures", "host_ledger_evidence_1539")


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _header_sum(directory):
    tensor_bytes = header_bytes = tensors = 0
    for shard in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        with open(shard, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
            assert not fh.read(1), f"{shard} carries payload, a fixture is header-only"
        header_bytes += 8 + n
        for name, meta in header.items():
            if name != "__metadata__":
                tensors += 1
                tensor_bytes += math.prod(meta["shape"]) * _DT[meta["dtype"]]
    return tensors, tensor_bytes, header_bytes


class ModelDirFixturesAreFaithful(unittest.TestCase):
    def test_every_recorded_file_is_unchanged(self):
        prov = FX.provenance()
        self.assertEqual(set(prov), {FX.VOCABEMBED, FX.DFLASH_W8, FX.NF_MINACHIST})
        for name, rec in prov.items():
            for fname, meta in rec["files"].items():
                path = os.path.join(FX.ROOT, name, fname) if name != FX.NF_MINACHIST else os.path.join(FX.ROOT, fname)
                self.assertEqual(_sha(path), meta["sha256"], f"{name}/{fname} was edited after extraction")
                self.assertEqual(os.path.getsize(path), meta["bytes"], f"{name}/{fname}")

    def test_the_headers_add_up_to_the_real_shards_to_the_byte(self):
        prov = FX.provenance()
        for name in (FX.VOCABEMBED, FX.DFLASH_W8, FX.NF_MINACHIST):
            tensors, tensor_bytes, header_bytes = _header_sum(FX.path(name))
            rec = prov[name]
            self.assertEqual(tensors, rec["tensors"], name)
            self.assertEqual(tensor_bytes, rec["tensor_bytes_from_headers"], name)
            self.assertEqual(header_bytes, rec["header_bytes"], name)
            self.assertEqual(tensor_bytes + header_bytes, rec["ondisk_safetensors_bytes"], name)

    def test_the_nf_config_is_the_snapshots(self):
        self.assertEqual(
            _sha(os.path.join(FX.path(FX.NF_MINACHIST), "config.json")),
            "b058d1b1cde7b76b342b0838ac02e1d1175f5d4aa5a6b0471d6f43a91bcbfaba",
        )


class FrozenHostLedgerEvidence(unittest.TestCase):
    def test_every_recorded_file_is_unchanged(self):
        for sub in ("nf_0929", "27b_z30y_0929"):
            with open(os.path.join(_EVIDENCE, sub, "PROVENANCE.json"), encoding="utf-8") as fh:
                prov = json.load(fh)
            for fname, rec in prov.items():
                self.assertEqual(_sha(os.path.join(_EVIDENCE, sub, fname)), rec["fixture_sha256"],
                                 f"{sub}/{fname} was edited after it was frozen")

    def test_the_measured_record_holds_only_the_boot_the_tests_bind_to(self):
        for sub, tag in (
            ("nf_0929", "dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30y2bar1dauer09291559"),
            ("27b_z30y_0929", "dkr27browauthorityz30ybar1fs09291331"),
        ):
            with open(os.path.join(_EVIDENCE, sub, "weg2_measured_record.json"), encoding="utf-8") as fh:
                samples = json.load(fh)["samples"]
            self.assertTrue(samples)
            self.assertEqual({s["boot_tag"] for s in samples}, {tag})

    def test_the_27b_census_says_it_is_not_a_copy(self):
        # the one place a number was not copied but rebuilt from documented values;
        # the file must keep saying so (an honest label is the only guard it has).
        with open(os.path.join(_EVIDENCE, "27b_z30y_0929", "PROVENANCE.json"), encoding="utf-8") as fh:
            prov = json.load(fh)
        self.assertIn("NOT A COPY", prov["census_terms_0929_1337.json"]["source"])


class OverlayServesOnlyWhatIsEmpty(unittest.TestCase):
    def test_reads_below_an_empty_directory_are_served(self):
        lost = FX.CACHE + FX.VOCABEMBED
        with FX.overlay():
            self.assertTrue(os.path.isdir(lost))
            self.assertTrue(os.path.isfile(lost + "/config.json"))
            with open(lost + "/config.json", encoding="utf-8") as fh:
                self.assertIn("architectures", json.load(fh))
            shards = glob.glob(lost + "/*.safetensors")
            self.assertEqual(len(shards), 18)
            self.assertTrue(all(s.startswith(lost + "/") for s in shards), "results keep the models-cache spelling")
            self.assertEqual(len(os.listdir(lost)) >= 20, True)

    def test_a_directory_with_its_own_config_is_never_overlaid(self):
        with tempfile.TemporaryDirectory() as cache:
            leaf = os.path.join(cache, FX.VOCABEMBED)
            os.makedirs(leaf)
            with open(os.path.join(leaf, "config.json"), "w") as fh:
                fh.write('{"real": true}')
            with mock.patch.object(FX, "CACHE", cache + "/"):
                with FX.overlay():
                    with open(os.path.join(leaf, "config.json")) as fh:
                        self.assertEqual(json.load(fh), {"real": True})
                    self.assertEqual(glob.glob(leaf + "/*.safetensors"), [])

    def test_writes_and_other_paths_are_untouched_and_everything_is_restored(self):
        import builtins
        import io

        before = (builtins.open, io.open, glob.glob, os.listdir, os.stat, os.path.exists,
                  os.path.isdir, os.path.isfile, os.path.getsize)
        # a PRIVATE cache with an empty leaf (never the real models-cache: a write test must not touch it)
        with tempfile.TemporaryDirectory() as cache:
            leaf = os.path.join(cache, FX.VOCABEMBED)
            os.makedirs(leaf)
            with mock.patch.object(FX, "CACHE", cache + "/"), FX.overlay():
                self.assertTrue(os.path.exists(__file__))
                self.assertFalse(os.path.exists(cache + "/no-such-model/config.json"))
                self.assertTrue(os.path.isfile(leaf + "/config.json"), "the empty leaf IS overlaid")
                with open(leaf + "/new-file.txt", "w") as fh:  # a write goes where it was aimed
                    fh.write("x")
            self.assertTrue(os.path.isfile(leaf + "/new-file.txt"))
            self.assertFalse(os.path.exists(os.path.join(FX.path(FX.VOCABEMBED), "new-file.txt")))
        after = (builtins.open, io.open, glob.glob, os.listdir, os.stat, os.path.exists,
                 os.path.isdir, os.path.isfile, os.path.getsize)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
