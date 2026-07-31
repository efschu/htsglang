# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""AUDIT #331: every persisted GPU reference keys on UUID, not on an index.

The whole suite runs on a fabricated rig whose CUDA and NVML orders
deliberately disagree, because that is the real shape of the reference
hardware and the shape under which every defect in this family appeared:

    NVML 0  RTX 3080 (A)   CUDA 1   0000:01:00.0
    NVML 1  RTX 5090       CUDA 0   0000:2d:00.0
    NVML 2  RTX 3080 (B)   CUDA 2   0000:41:00.0

A test that used a rig where the two orders agree would pass with the bug
still in place, which is why none of these do.

No driver is touched: ``identity_map`` takes both the NVML device list and
the CUDA-ordinal bridge as arguments precisely so this file can run on a desk
host with ``CUDA_VISIBLE_DEVICES=99``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.registry import nvml
from sglang.srt.registry.nvml import (
    DeviceInfo,
    DeviceNotFoundError,
    identity_map,
)

GIB = 1024**3

UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

BDF_3080_A = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"
BDF_3080_B = "00000000:41:00.0"


def rig_devices() -> list[DeviceInfo]:
    """NVML's view: PCI bus order, so the 5090 sits at index 1."""
    return [
        DeviceInfo(0, UUID_3080_A, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_A),
        DeviceInfo(1, UUID_5090, "NVIDIA GeForce RTX 5090", 32 * GIB, BDF_5090),
        DeviceInfo(2, UUID_3080_B, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_B),
    ]


def rig_cuda_bridge() -> dict[str, int]:
    """torch's view: FASTEST_FIRST, so the 5090 is ordinal 0."""
    return {BDF_5090: 0, BDF_3080_A: 1, BDF_3080_B: 2}


def rig_map():
    return identity_map(rig_devices(), rig_cuda_bridge())


def shuffled_devices() -> list[DeviceInfo]:
    """The same three cards after a reboot re-enumerated them.

    The 5090 moved from NVML 1 to NVML 0 (it was reseated into the first
    slot); the two 3080s moved down. Same physical cards, same UUIDs, every
    index different.
    """
    return [
        DeviceInfo(0, UUID_5090, "NVIDIA GeForce RTX 5090", 32 * GIB, BDF_5090),
        DeviceInfo(1, UUID_3080_A, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_A),
        DeviceInfo(2, UUID_3080_B, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_B),
    ]


def shuffled_map():
    # After the shuffle CUDA agrees with NVML on the 5090 but not on the rest.
    return identity_map(
        shuffled_devices(), {BDF_5090: 0, BDF_3080_B: 1, BDF_3080_A: 2}
    )


def map_without_5090():
    return identity_map(
        [
            DeviceInfo(0, UUID_3080_A, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_A),
            DeviceInfo(1, UUID_3080_B, "NVIDIA GeForce RTX 3080", 20 * GIB, BDF_3080_B),
        ],
        {BDF_3080_A: 0, BDF_3080_B: 1},
    )


# ===========================================================================
# The resolver itself
# ===========================================================================
class IdentityMapTest(unittest.TestCase):
    def test_the_two_enumerations_disagree_and_both_resolve(self):
        imap = rig_map()
        self.assertEqual(imap.by_nvml_index(1).uuid, UUID_5090)
        self.assertEqual(imap.by_cuda_ordinal(0).uuid, UUID_5090)
        # The point of the whole audit in one assertion: index 1 means two
        # different cards depending on which enumeration you meant.
        self.assertNotEqual(
            imap.by_nvml_index(1).uuid, imap.by_cuda_ordinal(1).uuid
        )
        self.assertEqual(imap.by_cuda_ordinal(1).uuid, UUID_3080_A)

    def test_uuid_is_the_key_in_both_directions(self):
        imap = rig_map()
        self.assertEqual(imap.nvml_index_of(UUID_5090), 1)
        self.assertEqual(imap.cuda_ordinal_of(UUID_5090), 0)
        self.assertEqual(imap.require(UUID_3080_B).name, "NVIDIA GeForce RTX 3080")
        self.assertEqual(len(imap), 3)
        self.assertEqual(
            set(imap.uuids), {UUID_3080_A, UUID_5090, UUID_3080_B}
        )

    def test_bdf_lookup_ignores_domain_padding_and_case(self):
        imap = rig_map()
        self.assertEqual(imap.by_pci_bus_id("0000:2d:00.0").uuid, UUID_5090)
        self.assertEqual(imap.by_pci_bus_id("00000000:2D:00.0").uuid, UUID_5090)

    def test_a_card_that_is_gone_is_a_named_error_not_a_neighbour(self):
        imap = map_without_5090()
        with self.assertRaises(DeviceNotFoundError) as caught:
            imap.require(UUID_5090)
        message = str(caught.exception)
        self.assertIn(UUID_5090, message)
        self.assertIn("not present on this host", message)
        # And it did NOT quietly hand back whatever now sits at that index.
        self.assertIsNone(imap.get(UUID_5090))

    def test_a_card_masked_from_torch_says_so(self):
        # NVML sees three cards, CUDA_VISIBLE_DEVICES shows torch only one.
        imap = identity_map(rig_devices(), {BDF_5090: 0})
        self.assertEqual(imap.cuda_ordinal_of(UUID_5090), 0)
        with self.assertRaises(DeviceNotFoundError) as caught:
            imap.cuda_ordinal_of(UUID_3080_A)
        self.assertIn("not visible to torch", str(caught.exception))

    def test_legacy_migration_states_which_enumeration_it_assumes(self):
        imap = rig_map()
        self.assertEqual(
            imap.adopt_legacy_indices([0, 1], order="nvml"),
            [UUID_3080_A, UUID_5090],
        )
        self.assertEqual(
            imap.adopt_legacy_indices([0, 1], order="cuda"),
            [UUID_5090, UUID_3080_A],
        )

    def test_legacy_migration_of_an_absent_index_raises(self):
        imap = rig_map()
        with self.assertRaises(DeviceNotFoundError) as caught:
            imap.adopt_legacy_indices([0, 7], order="nvml")
        self.assertIn("nvml index 7", str(caught.exception))
        with self.assertRaises(ValueError):
            imap.adopt_legacy_indices([0], order="pci")

    def test_building_the_map_does_not_create_a_cuda_context(self):
        """The default map is NVML-only, because a context is not free.

        ``torch.cuda.get_device_properties`` goes through ``_lazy_init``,
        which costs a few hundred MiB on every visible card. The boot-path
        presence checks run in the launcher before it forks, so the CUDA side
        must be opt-in rather than a side effect of asking who a uuid is.
        """
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.is_initialized.return_value = False
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(nvml._cuda_ordinals_by_bus(), {})
            fake_torch.cuda.get_device_properties.assert_not_called()
            # ... and with the licence, it does look.
            fake_torch.cuda.device_count.return_value = 0
            nvml._cuda_ordinals_by_bus(allow_cuda_init=True)
            fake_torch.cuda.device_count.assert_called()

    def test_describe_and_json_carry_all_four_names(self):
        card = rig_map().require(UUID_5090)
        text = card.describe()
        for token in (UUID_5090, "nvml=1", "cuda=0", BDF_5090):
            self.assertIn(token, text)
        self.assertEqual(
            card.to_json(),
            {
                "uuid": UUID_5090,
                "nvml_index": 1,
                "cuda_ordinal": 0,
                "pci_bus_id": BDF_5090,
                "name": "NVIDIA GeForce RTX 5090",
                "total_mib": 32 * 1024,
            },
        )


# ===========================================================================
# Cross-session arbitration files (/spinning/gpu-arb)
# ===========================================================================
class ArbIdentityTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.workbench.arb import ArbDirectory

        self.root = Path(tempfile.mkdtemp(prefix="arb-331-"))
        self.addCleanup(self._cleanup)
        self.now = 1_000_000.0
        self.ArbDirectory = ArbDirectory

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def _arb(self, imap, **kw):
        return self.ArbDirectory(
            self.root,
            occupancy=lambda idx: {i: 0 for i in idx},
            clock=lambda: self.now,
            identity=lambda: imap,
            **kw,
        )

    def test_a_claim_writes_uuids_beside_the_indices(self):
        claim = self._arb(rig_map()).claim([1], "measure the 5090")
        text = self.root.joinpath("holder").read_text()
        self.assertIn("cards=1", text)
        self.assertIn(f"card_uuids={UUID_5090}", text)
        self.assertEqual(claim.uuids, (UUID_5090,))
        claim.release()

    def test_a_holder_survives_an_enumeration_shuffle_on_the_right_cards(self):
        # Written yesterday, when the 5090 was NVML 1.
        self._arb(rig_map()).claim([1], "long run")
        # Read today, after the reboot that made the 5090 NVML 0.
        body = self._arb(shuffled_map()).snapshot()
        self.assertEqual(body["holder"]["cards"], "1")  # what the file says
        self.assertEqual(body["holder"]["resolved_card_uuids"], [UUID_5090])
        # ... and what it MEANS: card 0 today, not card 1.
        self.assertEqual(body["holder"]["resolved_cards"], [0])

    def test_a_stale_holder_is_checked_against_the_cards_it_really_names(self):
        # Stale holder from the old enumeration: it names NVML 1, which was
        # the 5090. Today the 5090 is NVML 0, and NVML 0 is the card that is
        # busy. Resolving through the uuid must find the busy card and refuse
        # to reap; an index-keyed read would have looked at NVML 1, found it
        # empty, and reaped a live holder.
        self.root.joinpath("holder").write_text(
            f"session=treiber  cards=1  card_uuids={UUID_5090}  "
            "purpose=old  since=2026-01-01T00:00:00Z\n"
        )
        os.utime(self.root / "holder", (self.now - 99999, self.now - 99999))
        arb = self.ArbDirectory(
            self.root,
            occupancy=lambda idx: {i: (8 * GIB if i == 0 else 0) for i in idx},
            clock=lambda: self.now,
            identity=shuffled_map,
        )
        from sglang.srt.workbench.arb import ArbRefused

        with self.assertRaises(ArbRefused) as caught:
            arb.claim([2], "something else")
        self.assertIn("still busy", str(caught.exception))
        self.assertTrue(self.root.joinpath("holder").is_file())

    def test_a_legacy_index_only_holder_is_migrated_and_says_so(self):
        self.root.joinpath("holder").write_text(
            "session=treiber  cards=1  purpose=old  since=2026-01-01T00:00:00Z\n"
        )
        body = self._arb(rig_map()).snapshot()
        self.assertEqual(body["holder"]["resolved_card_uuids"], [UUID_5090])
        self.assertIn("legacy index-only record", body["holder"]["identity_note"])

    def test_a_holder_naming_a_departed_card_is_named_not_rebound(self):
        self.root.joinpath("holder").write_text(
            f"session=treiber  cards=1  card_uuids={UUID_5090}  "
            "purpose=old  since=2026-01-01T00:00:00Z\n"
        )
        body = self._arb(map_without_5090()).snapshot()
        self.assertEqual(body["holder"]["resolved_card_uuids"], [UUID_5090])
        # No card was substituted for the missing one.
        self.assertEqual(body["holder"]["resolved_cards"], [])
        self.assertIn("are not present on this host", body["holder"]["identity_note"])

    def test_a_free_window_follows_the_card_not_the_index(self):
        from sglang.srt.workbench.arb import ArbRefused

        # The other session published a window on the 5090 by uuid.
        self.root.joinpath("free-until").write_text(
            f"2999-01-01T00:00:00Z  cards=1  card_uuids={UUID_5090}  by=treiber\n"
        )
        arb = self._arb(shuffled_map())
        # Today the 5090 is index 0. Claiming it must be refused ...
        with self.assertRaises(ArbRefused) as caught:
            arb.claim([0], "would collide")
        self.assertIn("free window is published", str(caught.exception))
        # ... and claiming index 1, which is now a 3080, must be allowed.
        arb.claim([1], "fine").release()

    def test_a_missing_identity_map_passes_indices_through(self):
        # A desk host without NVML must still be able to read the directory.
        body = self.ArbDirectory(
            self.root, clock=lambda: self.now, identity=lambda: None
        ).snapshot()
        self.assertTrue(body["usable"])


# ===========================================================================
# Saved planner profiles (~/.cache/sglang/planner_profiles.json)
# ===========================================================================
class ProfileStoreIdentityTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.planner.flags import CARD_UUIDS_KEY, Profile, ProfileStore

        self.tmp = tempfile.mkdtemp(prefix="profiles-331-")
        self.addCleanup(self._cleanup)
        self.path = os.path.join(self.tmp, "planner_profiles.json")
        self.Profile = Profile
        self.ProfileStore = ProfileStore
        self.key = CARD_UUIDS_KEY

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _profile(self, ordinals):
        return self.Profile(
            name="tp3-uneven", kind="uneven-max-perf",
            settings={"rank_gpu_id": list(ordinals), "tp_size": len(ordinals)},
        )

    def test_saving_stamps_the_card_identities(self):
        store = self.ProfileStore(self.path, identity=rig_map)
        # CUDA ordinals 0,1,2 == 5090, 3080-A, 3080-B on this rig.
        store.save(self._profile([0, 1, 2]))
        raw = json.loads(Path(self.path).read_text())["profiles"][0]
        self.assertEqual(
            raw[self.key]["rank_gpu_id"], [UUID_5090, UUID_3080_A, UUID_3080_B]
        )
        self.assertEqual(raw["settings"]["rank_gpu_id"], [0, 1, 2])

    def test_loading_after_a_shuffle_rewrites_the_ordinals(self):
        self.ProfileStore(self.path, identity=rig_map).save(self._profile([0, 1, 2]))
        # Reboot: 5090 -> CUDA 0, 3080-B -> CUDA 1, 3080-A -> CUDA 2.
        loaded = self.ProfileStore(self.path, identity=shuffled_map).load("tp3-uneven")
        self.assertEqual(loaded.settings["rank_gpu_id"], [0, 2, 1])
        # The rewrite is persisted, not recomputed on every read.
        on_disk = json.loads(Path(self.path).read_text())["profiles"][0]
        self.assertEqual(on_disk["settings"]["rank_gpu_id"], [0, 2, 1])
        self.assertEqual(
            on_disk[self.key]["rank_gpu_id"], [UUID_5090, UUID_3080_A, UUID_3080_B]
        )

    def test_load_all_migrates_too(self):
        self.ProfileStore(self.path, identity=rig_map).save(self._profile([0, 1, 2]))
        profiles = self.ProfileStore(self.path, identity=shuffled_map).load_all()
        self.assertEqual(profiles[0].settings["rank_gpu_id"], [0, 2, 1])

    def test_a_legacy_profile_loads_unchanged_and_warns(self):
        Path(self.path).write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": "old",
                            "kind": "custom",
                            "settings": {"rank_gpu_id": [0, 1]},
                            "info": [],
                            "env": {},
                        }
                    ]
                }
            )
        )
        store = self.ProfileStore(self.path, identity=shuffled_map)
        with self.assertLogs("sglang.srt.planner.flags", level="WARNING") as logs:
            loaded = store.load("old")
        self.assertEqual(loaded.settings["rank_gpu_id"], [0, 1])
        self.assertIn("predates card identity stamping", "\n".join(logs.output))

    def test_a_departed_card_is_a_named_error(self):
        self.ProfileStore(self.path, identity=rig_map).save(self._profile([0, 1, 2]))
        store = self.ProfileStore(self.path, identity=map_without_5090)
        with self.assertRaises(DeviceNotFoundError) as caught:
            store.load("tp3-uneven")
        self.assertIn(UUID_5090, str(caught.exception))

    def test_a_profile_without_cards_is_untouched(self):
        store = self.ProfileStore(self.path, identity=rig_map)
        store.save(self.Profile(name="plain", kind="single", settings={"tp_size": 1}))
        raw = json.loads(Path(self.path).read_text())["profiles"][0]
        self.assertNotIn(self.key, raw)
        self.assertEqual(store.load("plain").settings, {"tp_size": 1})


# ===========================================================================
# Measured KV-budget registry (~/.cache/sglang/kv_budget-*.json)
# ===========================================================================
class MeasuredRegistryIdentityTest(unittest.TestCase):
    def _check(self, components, imap):
        from sglang.srt import uneven_perf

        # list_devices, not identity_map: the boot-path checks are forbidden
        # from creating a CUDA context, so they never build the full map.
        with mock.patch.object(nvml, "is_available", return_value=True), \
                mock.patch.object(nvml, "list_devices", return_value=list(imap.cards)):
            return uneven_perf.measured_registry_cards_still_present(components)

    def test_a_registry_measured_on_the_present_cards_is_kept(self):
        comps = [{"card_uuid": UUID_5090}, {"card_uuid": UUID_3080_A}]
        self.assertTrue(self._check(comps, shuffled_map()))

    def test_a_registry_measured_on_a_departed_card_is_discarded(self):
        comps = [{"card_uuid": UUID_5090}, {"card_uuid": UUID_3080_A}]
        self.assertFalse(self._check(comps, map_without_5090()))

    def test_a_pre_331_registry_is_kept_with_a_warning(self):
        from sglang.srt import uneven_perf

        comps = [{"device_total_bytes": 1}, {"device_total_bytes": 2}]
        with self.assertLogs("sglang.srt.uneven_perf", level="WARNING") as logs:
            kept = uneven_perf.measured_registry_cards_still_present(comps)
        self.assertTrue(kept)
        self.assertIn("predates card-identity stamping", "\n".join(logs.output))

    def test_an_unreachable_driver_does_not_lose_the_registry(self):
        from sglang.srt import uneven_perf

        comps = [{"card_uuid": UUID_5090}]
        with mock.patch.object(nvml, "is_available", return_value=False):
            self.assertTrue(uneven_perf.measured_registry_cards_still_present(comps))


# ===========================================================================
# Hibernate images (#89)
# ===========================================================================
class HibernateIdentityTest(unittest.TestCase):
    def _manifest(self, uuids):
        return {"ranks": {str(i): {"nvml_uuid": u} for i, u in enumerate(uuids)}}

    def _check(self, manifest, imap):
        from sglang.srt.model_loader import hibernate

        with mock.patch.object(nvml, "is_available", return_value=True), \
                mock.patch.object(nvml, "list_devices", return_value=list(imap.cards)):
            return hibernate._manifest_cards_present(manifest)

    def test_an_image_parked_on_present_cards_still_matches(self):
        self.assertTrue(
            self._check(self._manifest([UUID_5090, UUID_3080_A]), shuffled_map())
        )

    def test_an_image_parked_on_a_departed_card_falls_back_to_cold_load(self):
        from sglang.srt.model_loader import hibernate

        with mock.patch.object(nvml, "is_available", return_value=True), \
                mock.patch.object(
                    nvml, "list_devices",
                    return_value=list(map_without_5090().cards)), \
                self.assertLogs("sglang.srt.model_loader.hibernate", level="WARNING") as logs:
            present = hibernate._manifest_cards_present(
                self._manifest([UUID_5090, UUID_3080_A])
            )
        self.assertFalse(present)
        self.assertIn(UUID_5090, "\n".join(logs.output))

    def test_a_manifest_without_rank_uuids_is_not_rejected_here(self):
        self.assertTrue(self._check({"ranks": {"0": {}}}, rig_map()))


# ===========================================================================
# Card-window locks (/tmp/gpu-card-N.lock)
# ===========================================================================
class CardWindowIdentityTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.planner import comm_suite

        self.comm_suite = comm_suite
        self.tmp = tempfile.mkdtemp(prefix="locks-331-")
        self.addCleanup(self._cleanup)
        self.fmt = os.path.join(self.tmp, "gpu-card-{}.lock")
        patcher = mock.patch.object(comm_suite, "LOCK_DIR_FMT", self.fmt)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("QUIET_LOCK_DIR", "LEGACY_LOCK_DIR"):
            p = mock.patch.object(comm_suite, name, os.path.join(self.tmp, name))
            p.start()
            self.addCleanup(p.stop)

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_lock_info_records_the_physical_card(self):
        with mock.patch.object(self.comm_suite, "_nvidia_smi_cards", return_value=[]):
            window = self.comm_suite._CardWindow([1], identity=rig_map)
            self.assertTrue(window.acquire(), window.reason)
        info = Path(self.fmt.format(1), "info").read_text()
        self.assertIn(f"uuid={UUID_5090}", info)
        self.assertIn(f"pci_bus_id={BDF_5090}", info)
        self.assertIn("nvml_index=1", info)
        window.release()
        self.assertFalse(os.path.isdir(self.fmt.format(1)))

    def test_a_lock_that_outlived_a_reenumeration_is_reported_as_such(self):
        # A lock taken when NVML 1 was the 5090 ...
        os.mkdir(self.fmt.format(1))
        Path(self.fmt.format(1), "info").write_text(
            f"owner=gpu_battery\nnvml_index=1\nuuid={UUID_5090}\n"
        )
        # ... read after the shuffle, where NVML 1 is a 3080.
        window = self.comm_suite._CardWindow([1], identity=shuffled_map)
        self.assertFalse(window.acquire())
        self.assertIn("outlived a re-enumeration", window.reason)
        self.assertIn(UUID_5090, window.reason)

    def test_a_matching_lock_names_the_card_plainly(self):
        os.mkdir(self.fmt.format(1))
        Path(self.fmt.format(1), "info").write_text(
            f"owner=gpu_battery\nuuid={UUID_5090}\n"
        )
        window = self.comm_suite._CardWindow([1], identity=rig_map)
        self.assertFalse(window.acquire())
        self.assertIn("RTX 5090", window.reason)
        self.assertNotIn("outlived", window.reason)

    def test_a_pre_331_lock_says_it_cannot_be_verified(self):
        os.mkdir(self.fmt.format(1))
        Path(self.fmt.format(1), "info").write_text("owner=gpu_battery\n")
        window = self.comm_suite._CardWindow([1], identity=rig_map)
        self.assertFalse(window.acquire())
        self.assertIn("records no card uuid", window.reason)


# ===========================================================================
# Workbench grants: the subprocess is pinned by uuid, not by index
# ===========================================================================
class WorkGrantIdentityTest(unittest.TestCase):
    def test_visible_devices_uses_uuids(self):
        from sglang.srt.workbench.tenant import WorkGrant

        grant = WorkGrant(
            card_uuids=(UUID_5090, UUID_3080_A),
            card_indices=(1, 0),
            per_card_bytes=GIB,
            artifact_root=Path("/tmp"),
        )
        self.assertEqual(grant.visible_devices, f"{UUID_5090},{UUID_3080_A}")

    def test_an_unpinned_grant_still_falls_back_to_indices(self):
        from sglang.srt.workbench.tenant import WorkGrant

        grant = WorkGrant(
            card_uuids=(),
            card_indices=(0, 2),
            per_card_bytes=GIB,
            artifact_root=Path("/tmp"),
        )
        self.assertEqual(grant.visible_devices, "0,2")


if __name__ == "__main__":
    unittest.main()
