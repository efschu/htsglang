"""HW-GENERIC 1003 (desk item 030-hwgen, 27B line y8p bd2e3bc22d): the per-profile
regression gate.

The plan fingerprint golden (fixtures/hw_generic_1002/
rig_plan_fingerprint_base_bd2e3bc22d.json) was written on the tree WITHOUT
HW-GENERIC. This file pins, for every launcher profile file that is a release
candidate of the NF line and of the 27B line (docker/profiles_release/*.env
plus 27b-row-authority-cut43, 27b-nvfp4-dual1i, nf-int4-h6, nf-int4 from
docker/profiles/):

  * the profile's PROFILE_ARGS still parse with the launcher's parser and
    name one of the two profile rows the fingerprint covers (nextflash /
    qwen27b), so the row section of the golden IS that profile's plan;
  * that row section of the current tree equals the golden byte for byte;
  * the HW-INVENTORY check on the reference rig MATCHES for the profile's
    own argv (positional vectors included), i.e. the port opens no refusal on
    the reference rig for any of them.

Skips by name where the docker profile directory is not on this box. GPU-free
and NVML-free.
"""

import glob
import json
import os
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "hw_generic_1002", "rig_plan_fingerprint_base_bd2e3bc22d.json")
DOCKER = os.environ.get("HW_GENERIC_DOCKER_DIR", "/spinning/gpu-arb/docker")
EXTRA = ("27b-row-authority-cut43", "27b-nvfp4-dual1i", "27b-nvfp4-dual1m-psleep", "nf-int4-h6", "nf-int4")


def profile_files():
    files = sorted(glob.glob(os.path.join(DOCKER, "profiles_release", "*.env")))
    files += [os.path.join(DOCKER, "profiles", n + ".env") for n in EXTRA]
    return [f for f in files if os.path.isfile(f)]


def profile_argv(path):
    out = subprocess.run(
        ["bash", "-c", 'set +e; source "$1" >/dev/null 2>&1; printf "%s\\0" "${PROFILE_ARGS[@]}"', "x", path],
        capture_output=True, text=True, env={**os.environ, "HOME": "/root"}, check=False).stdout
    return [a for a in out.split("\0") if a]


def rig_ordered():
    sys.path.insert(0, HERE)
    try:
        import hw_generic_rig_plan_fingerprint_1002 as FP
    finally:
        sys.path.remove(HERE)
    return FP, L.order_cards(FP.rig_cards(L))


@unittest.skipUnless(profile_files(), "docker profile directory not on this box")
class EveryReleaseProfileKeepsItsReferenceRigPlan(unittest.TestCase):
    def test_golden_row_section_and_inventory_match_for_every_profile(self):
        FP, cards = rig_ordered()
        with open(GOLDEN) as fh:
            golden = json.load(fh)
        now = json.loads(json.dumps(FP.fingerprint(), sort_keys=True, default=str))
        seen = {}
        for path in profile_files():
            name = os.path.basename(path)
            with self.subTest(profile=name):
                argv = profile_argv(path)
                self.assertTrue(argv, f"{name}: PROFILE_ARGS empty")
                ns, _unknown = L.build_parser().parse_known_args(["--tree", "/t", "--tag", "t"] + argv)
                self.assertIn(ns.profile, ("nextflash", "qwen27b"), f"{name}: row {ns.profile!r}")
                self.assertEqual(now[ns.profile], golden[ns.profile],
                                 f"{name}: row {ns.profile} plan differs from the pre-HW-GENERIC golden")
                line = L.inventory_check_line(ns, cards)
                self.assertIn("MATCH", line, f"{name}: {line[:200]}")
                self.assertNotIn("UNCALIBRATED", line)
                seen[ns.profile] = seen.get(ns.profile, 0) + 1
        self.assertEqual(sorted(seen), ["nextflash", "qwen27b"], "both profile rows must be exercised")

    def test_release_dir_covers_the_named_profiles(self):
        names = {os.path.basename(f) for f in profile_files()}
        for want in ("27b-row-authority-cut43.env", "27b-nvfp4-dual1i.env", "nf-int4-h6.env", "nf-int4.env"):
            self.assertIn(want, names)


if __name__ == "__main__":
    unittest.main()
