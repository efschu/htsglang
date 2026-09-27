"""weg2 start-line wizard: the host line and its boot-free dry run."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import weg2line  # noqa: E402

REAL_HA = "/spinning/gpu-arb/docker/host_acceptance.sh"
IMAGES = ("htsglang:cu130-weg2-rc12k27-27b-nf\t292cc6411ada\t2026-09-27 11:29:20 +0200 CEST\n"
          "htsglang:cu130-weg2-rc12g-27b-nf-flat\t840cefacf0fa\t2026-09-27 08:15:44 +0200 CEST\n"
          "wyoming-whisper:latest\tabc\t2026-09-26\n")

MINI_HA = """#!/bin/bash
set -euo pipefail
S=/x
CTX=${CTX:?CTX}
cap_for() { echo 76g; }
mkdir -p "$ACC"
say(){ echo x | tee -a "$LOG"; }
die(){ say "ABBRUCH: $*"; exit 1; }
for _kv in $NCCL_EXTRA_ENV; do :; done
[[ $RUN_LABEL =~ ^[a-z0-9]*$ ]] || die "RUN_LABEL"
trap cleanup EXIT
ct999_state() {
  echo running
}
ct999_mem_mib() { echo 1; }
house_args() {
  echo --memory 76g
}
ct999_tmpfs_note() {
  return 0
}
house_check() {
  return 0
}
"""


class ParseTests(unittest.TestCase):
    def test_images_and_ctx_mapping(self):
        imgs = weg2line.parse_images(IMAGES)
        self.assertEqual([i["release"] for i in imgs], ["rc12k27", "rc12g"])
        self.assertTrue(imgs[1]["flat"])
        ctxs = [{"ctx": "/c/delta-rc12k27", "release": "rc12k27", "cuda": "cu130"},
                {"ctx": "/c/delta-rc12g", "release": "rc12g", "cuda": "cu130"}]
        self.assertEqual(weg2line.ctx_for_image(imgs[1], ctxs)["ctx"], "/c/delta-rc12g")

    def test_profile_fields_read_without_sourcing(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
            fh.write("PROFILE_NAME=nf-h91\nPROFILE_LINE=nf\nPROFILE_STATUS=experimentell   # H91: bis ...\n"
                     "PROFILE_OWNER=\"NF-Sitz (x)\"\n")
        f = weg2line.read_profile_fields(fh.name)
        os.unlink(fh.name)
        self.assertEqual((f["PROFILE_LINE"], f["PROFILE_STATUS"], f["PROFILE_OWNER"]), ("nf", "experimentell", "NF-Sitz (x)"))


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        os.makedirs(os.path.join(d, "profiles"))
        ctx = os.path.join(d, "ctx", "delta-rc12k27-27b9417507cd2-from-rc12g")
        os.makedirs(os.path.join(ctx, "tools", "profiles"))
        with open(os.path.join(ctx, "BUILD_INFO.json"), "w") as fh:
            json.dump({"layout": "duo-delta", "release": "rc12k27", "cuda": "cu130",
                       "lines": {"27b": {"revision": "9417507cd2aa"}, "nf": {"revision": "9417507cd2aa"}}}, fh)
        with open(os.path.join(ctx, "tools", "profiles", "27b-release-draft.env"), "w") as fh:
            fh.write("PROFILE_NAME=27b-release-draft\nPROFILE_STATUS=experimentell\n")
        with open(os.path.join(d, "profiles", "27b-release-draft.env"), "w") as fh:
            fh.write("PROFILE_NAME=27b-release-draft\nPROFILE_STATUS=experimentell\n# newer\n")
        with open(os.path.join(d, "profiles", "nf-h91.env"), "w") as fh:
            fh.write("PROFILE_NAME=nf-h91\nPROFILE_LINE=nf\nPROFILE_STATUS=experimentell\nPROFILE_NCCL_STATUS=unproven\n")
        self.w = weg2line.Weg2Lines(["false"], ["27b-release-draft", "nf-h91"], docker_dir=d,
                                    ctx_glob=os.path.join(d, "ctx", "*", "BUILD_INFO.json"))
        self.w._images = (1e18, weg2line.parse_images(IMAGES), None)   # no ssh in tests
        self.ctx = ctx

    def tearDown(self):
        self.tmp.cleanup()

    def test_27b_line_from_image_profile(self):
        b = self.w.build("27b-release-draft", "htsglang:cu130-weg2-rc12k27-27b-nf")
        self.assertIn("CTX=%s IMAGE=htsglang:cu130-weg2-rc12k27-27b-nf LINE=27b PROFILE=27b-release-draft "
                      "PROFILE_MOUNT=0 ALLOW_EXPERIMENTAL=1 HOUSE_GUARD=memlimit GPUQ_ID=<fenster-id> bash " % self.ctx,
                      b["command"])
        self.assertTrue(b["command"].endswith("/spinning/gpu-arb/docker/host_acceptance.sh serve bar1"))
        self.assertTrue(b["notes"][0].startswith("ZUERST ein gpuq-Fenster buchen"))
        self.assertTrue(any("weicht" in n for n in b["notes"]))   # image copy differs from host file

    def test_nf_profile_not_in_image_is_mounted_and_nccl_warned(self):
        b = self.w.build("nf-h91", "htsglang:cu130-weg2-rc12k27-27b-nf", transport="nccl")
        self.assertIn("LINE=nf PROFILE=nf-h91 PROFILE_MOUNT=1 ALLOW_EXPERIMENTAL=1", b["command"])
        self.assertTrue(b["command"].endswith("serve nccl"))
        self.assertTrue(any("unproven" in n for n in b["notes"]))

    def test_inputs_outside_the_offer_are_refused(self):
        for kw in ({"profile": "27b-int8-x16"}, {"image": "htsglang:nope"}, {"transport": "tcp"},
                   {"house_guard": "none"}, {"profile": "27b-release-draft; rm -rf /"}):
            args = dict(profile="27b-release-draft", image="htsglang:cu130-weg2-rc12k27-27b-nf",
                        transport="bar1", house_guard="memlimit")
            args.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                self.w.build(**args)


class DryScriptTests(unittest.TestCase):
    def test_mini_script_keeps_prelude_checks_and_functions_only(self):
        s = weg2line.build_dry_script(MINI_HA)
        self.assertIn("CTX=${CTX:?CTX}", s)
        self.assertIn('[[ $RUN_LABEL =~ ^[a-z0-9]*$ ]]', s)
        self.assertIn("house_args() {", s)
        self.assertNotIn('mkdir -p "$ACC"', s)
        self.assertNotIn("trap cleanup EXIT", s)
        self.assertNotIn("tee -a", s)

    def test_missing_marker_refuses(self):
        with self.assertRaises(ValueError):
            weg2line.build_dry_script(MINI_HA.replace('mkdir -p "$ACC"\n', ""))

    def test_side_effect_moved_into_prelude_refuses(self):
        with self.assertRaises(ValueError):
            weg2line.build_dry_script(MINI_HA.replace("S=/x\n", "S=/x\ntrap cleanup EXIT\n"))

    @unittest.skipUnless(os.path.isfile(REAL_HA), "host_acceptance.sh not on this machine")
    def test_real_script_never_carries_the_container_killing_trap(self):
        with open(REAL_HA) as fh:
            s = weg2line.build_dry_script(fh.read())
        # the EXIT trap of host_acceptance.sh stops and removes every htsglang-acc-<line>-* container
        for bad in ("trap cleanup EXIT", "docker run", "docker stop", "docker rm", 'mkdir -p "$ACC"', "holder_on"):
            self.assertNotIn(bad, s, bad)
        self.assertIn("DRY GRUEN", s)


if __name__ == "__main__":
    unittest.main()
