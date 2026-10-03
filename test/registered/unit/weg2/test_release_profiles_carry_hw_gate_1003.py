"""HW-GENERIC 1003 (desk item 820): every release profile carries the count /
inventory gate.

test_hw_generic_profile_gate_1003.py pins the launcher plan per profile
(PROFILE_ARGS); it cannot see PROFILE_CARD_COUNT / PROFILE_INVENTORY, which are
entrypoint variables and never reach PROFILE_ARGS. A profile that fell back to
the bare PROFILE_GPUS card-name gate would keep that test green. This file
closes the gap, hardware-generically: it asserts THAT a count and inventory
gate exists for every docker/profiles_release/*.env (resolved through its
`source` chain), not WHICH cards it names.

Per profile:
  * PROFILE_CARD_COUNT is set and a positive integer;
  * PROFILE_INVENTORY is set, non-empty, one comma separated entry per card
    (the entry count equals PROFILE_CARD_COUNT), no empty entry;
  * PROFILE_GPUS may remain (tree compatibility) but never stands alone.

Skips by name where the docker profile directory is not on this box. GPU-free
and NVML-free.
"""

import glob
import os
import subprocess
import unittest

DOCKER = os.environ.get("HW_GENERIC_DOCKER_DIR", "/spinning/gpu-arb/docker")
RELEASE_DIR = os.path.join(DOCKER, "profiles_release")
_PROBE = (
    'set +e; source "$1" >/dev/null 2>&1; '
    'printf "COUNT_SET=%s\\n" "${PROFILE_CARD_COUNT+1}"; '
    'printf "COUNT=%s\\n" "${PROFILE_CARD_COUNT-}"; '
    'printf "INV_SET=%s\\n" "${PROFILE_INVENTORY+1}"; '
    'printf "INV=%s\\n" "${PROFILE_INVENTORY-}"; '
    'printf "GPUS_SET=%s\\n" "${PROFILE_GPUS+1}"'
)


def release_profiles():
    return sorted(glob.glob(os.path.join(RELEASE_DIR, "*.env")))


def gate_vars(path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROFILE_")}
    env["HOME"] = "/root"
    out = subprocess.run(["bash", "-c", _PROBE, "x", path], capture_output=True, text=True,
                         env=env, check=False).stdout
    kv = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        kv[k] = v
    return kv


@unittest.skipUnless(release_profiles(), "docker profiles_release directory not on this box")
class EveryReleaseProfileCarriesTheCountInventoryGate(unittest.TestCase):
    def test_count_and_inventory_gate_present_for_every_release_profile(self):
        for path in release_profiles():
            name = os.path.basename(path)
            with self.subTest(profile=name):
                v = gate_vars(path)
                self.assertEqual(v.get("COUNT_SET"), "1", f"{name}: PROFILE_CARD_COUNT not set (name-only gate?)")
                self.assertEqual(v.get("INV_SET"), "1", f"{name}: PROFILE_INVENTORY not set (name-only gate?)")
                count = v.get("COUNT", "")
                self.assertTrue(count.isdigit() and int(count) > 0,
                                f"{name}: PROFILE_CARD_COUNT={count!r} is not a positive integer")
                entries = v.get("INV", "").split(",")
                self.assertTrue(all(e.strip() for e in entries),
                                f"{name}: PROFILE_INVENTORY={v.get('INV')!r} is empty or has an empty entry")
                self.assertEqual(len(entries), int(count),
                                f"{name}: PROFILE_INVENTORY has {len(entries)} entries, PROFILE_CARD_COUNT={count}")

    def test_release_dir_has_the_gate_checked_profiles(self):
        names = {os.path.basename(p) for p in release_profiles()}
        for want in ("27b.env", "27b-nvfp4-dual.env", "nf.env", "nf-int4.env"):
            self.assertIn(want, names)


if __name__ == "__main__":
    unittest.main()
