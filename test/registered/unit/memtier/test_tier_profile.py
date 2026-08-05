"""The rig profile: measured numbers as data, and the honesty rules (#407).

The bundled profile describes ONE machine. Three properties are pinned here
because losing any of them turns a rig measurement into a hardware truth:

1. **Every number names its source.** A rate with an empty ``source`` is
   refused at load -- including an absent one, where the source IS the record
   ("nobody ran this probe").

2. **The three honest absences stay absent.** DESIGN_407 §0 corrected three
   figures a reader would otherwise assume were measured: the peer-VRAM
   "1-3 us posted write" class is an assumption, NVMe latency does not exist
   in the tree in any unit, and host DRAM bandwidth is derived from an
   *assumed* DDR4-3200 peak. They ship as ABSENT, ABSENT and ESTIMATE. A
   future diff that "filled in" any of them turns this red -- which is the
   entire point of the file.

3. **An overlay merges by id.** A second rig states only what differs, and
   tier matching is by name, not by position: a positional merge would
   silently retarget every entry after a dropped one.

Hermetic: the bundled JSON, plus tiny overlays written to a temp directory.
``collect_local_facts`` is exercised for real -- ``/proc/meminfo`` and
``statvfs`` need no driver -- so the live path is not desk-written code.

    python -m pytest test/registered/unit/memtier/test_tier_profile.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.memtier.profile import (
    BUNDLED_PROFILE_PATH,
    CardFact,
    FilesystemFact,
    LocalFacts,
    ProfileError,
    apply_local_facts,
    bind_device_tiers,
    bundled_profile,
    collect_local_facts,
    honest_host_memory_bytes,
    host_memory_bytes_for_pinning,
    load_profile,
    profile_from_json,
)
from sglang.srt.memtier.registry import TierRegistry
from sglang.srt.memtier.tiers import TierKind, Volatility
from sglang.srt.planner.cost_model import Provenance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _document():
    return json.loads(BUNDLED_PROFILE_PATH.read_text())


class BundledProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = bundled_profile()

    def test_every_rate_in_the_bundled_profile_names_its_source(self):
        """The rule the loader enforces, asserted against the shipped file so
        a hand-edited entry cannot slip through review."""
        for tier in self.profile.tiers:
            for name, rate in (
                ("total", tier.capacity.total),
                ("floor", tier.capacity.floor),
                ("latency", tier.caps.latency_us),
                ("bandwidth", tier.caps.bandwidth_gbs),
                ("aperture", tier.caps.aperture_bytes),
            ):
                self.assertTrue(
                    rate.source.strip(), msg=f"{tier.id}.{name} has no source"
                )

    def test_the_three_honest_absences(self):
        """DESIGN_407 §0's honesty corrections, pinned one by one."""
        host = self.profile.tier("host:rig-1")
        self.assertIs(host.caps.bandwidth_gbs.provenance, Provenance.ESTIMATE)
        self.assertIn("DDR4-3200", host.caps.bandwidth_gbs.source)

        nvme = self.profile.tier("fs:rig-1:/spinning")
        self.assertTrue(nvme.caps.latency_us.is_absent)
        self.assertIn("fio", nvme.caps.latency_us.source)

        card = self.profile.device_models["NVIDIA GeForce RTX 5090"]
        self.assertTrue(card.caps.latency_us.is_absent)
        self.assertIn("ASSUMPTION", card.properties["peer_latency_claim"])

    def test_the_measured_numbers_are_the_ones_the_design_recorded(self):
        """External-source literals: each came off a named probe in the design
        doc's §1.1 table. Deleting these leaves nothing guarding a typo in the
        one file the whole registry reads its numbers from."""
        self.assertEqual(
            self.profile.tier("fs:rig-1:/spinning").caps.bandwidth_gbs.value, 1.8
        )
        rig2 = self.profile.tier("host:rig-2")
        self.assertEqual(rig2.caps.bandwidth_gbs.value, 2.83)
        self.assertEqual(rig2.caps.latency_us.value, 1.47)
        self.assertEqual(
            self.profile.device_models[
                "NVIDIA GeForce RTX 5090"
            ].caps.bandwidth_gbs.value,
            1558.0,
        )
        self.assertEqual(
            self.profile.device_models[
                "NVIDIA GeForce RTX 3080"
            ].caps.bandwidth_gbs.value,
            723.0,
        )

    def test_the_effective_aperture_is_absent_on_every_card(self):
        """§3.3: effective, never nominal. The nominal window lives in
        properties precisely so it cannot be read as the effective one."""
        for model in self.profile.device_models.values():
            self.assertTrue(model.caps.aperture_bytes.is_absent)
            self.assertIn("bar1_nominal_bytes", model.properties)

    def test_the_tmpfs_tier_is_declared_non_persistent(self):
        """The record that makes #89's silent hole checkable: persistence is a
        declared class, not a property of a path."""
        shm = self.profile.tier("fs:rig-1:/dev/shm")
        self.assertIs(shm.volatility, Volatility.EXPENSIVE_OK)
        self.assertIn("no", shm.properties["persistent_across_reboot"])
        self.assertIs(
            self.profile.tier("fs:rig-1:/spinning").volatility, Volatility.PERSISTENT
        )

    def test_unreachable_tiers_are_declared_with_a_blocking_verdict(self):
        """X3: a rig where cross-rig GPU-to-GPU works needs a data change, not
        a code change -- which only holds if the tier is in the file at all."""
        remote = self.profile.tier("vram:unenumerated@rig-2")
        self.assertEqual(remote.health.verdict, "block")
        self.assertFalse(remote.is_bound)
        self.assertIn("osCheckGpuBarsOverlapAddrRange", remote.health.reason)

    def test_the_caveat_travels_with_the_profile(self):
        registry = TierRegistry.from_profile(self.profile)
        self.assertIn("rig profile, not a hardware truth", registry.to_json()["caveat"])


class ProfileLoaderTest(unittest.TestCase):
    def test_a_rate_without_a_source_is_refused(self):
        """Can-fail proof: strip the source from one rate in a valid document
        and the load fails naming that exact field."""
        document = _document()
        document["tiers"][0]["caps"]["bandwidth_gbs"]["source"] = "  "
        with self.assertRaises(ProfileError) as ctx:
            profile_from_json(document)
        self.assertIn("bandwidth_gbs", str(ctx.exception))
        self.assertIn("source", str(ctx.exception))

    def test_an_absent_rate_carrying_a_value_is_refused(self):
        document = _document()
        document["tiers"][0]["caps"]["latency_us"]["value"] = 1.0
        with self.assertRaises(ProfileError):
            profile_from_json(document)

    def test_a_measured_rate_without_a_value_is_refused(self):
        document = _document()
        document["tiers"][0]["caps"]["bandwidth_gbs"]["value"] = None
        with self.assertRaises(ProfileError) as ctx:
            profile_from_json(document)
        self.assertIn('"absent"', str(ctx.exception))

    def test_a_tier_without_a_ledger_key_is_refused(self):
        """A reservation must post to a named bucket; a tier that cannot name
        one cannot be reserved against, so it is refused at load."""
        document = _document()
        document["tiers"][0]["caps"].pop("ledger_key")
        with self.assertRaises(ProfileError) as ctx:
            profile_from_json(document)
        self.assertIn("ledger_key", str(ctx.exception))

    def test_duplicate_tier_ids_are_refused(self):
        document = _document()
        document["tiers"].append(dict(document["tiers"][0]))
        with self.assertRaises(ProfileError) as ctx:
            profile_from_json(document)
        self.assertIn("duplicate", str(ctx.exception))

    def test_an_unsupported_schema_version_is_refused(self):
        document = _document()
        document["schema_version"] = 99
        with self.assertRaises(ProfileError):
            profile_from_json(document)

    def test_an_overlay_merges_by_id_and_may_add_a_tier(self):
        """A second rig states only what differs. Matching by id rather than
        by position is what keeps an overlay that omits one entry from
        retargeting every later one."""
        overlay = {
            "schema_version": 1,
            "profile_id": "other-rig",
            "host": "rig-1",
            "tiers": [
                {
                    "id": "fs:rig-1:/spinning",
                    "caps": {
                        "bandwidth_gbs": {
                            "value": 6.4,
                            "provenance": "measured",
                            "source": "overlay: gen4 NVMe on the other rig",
                            "unit": "GB/s",
                        }
                    },
                },
                {
                    "id": "blob:mooncake:cluster-a",
                    "kind": "blob",
                    "host": "rig-1",
                    "volatility": "persistent",
                    "admits": [],
                    "capacity": {
                        "total": {
                            "value": None,
                            "provenance": "absent",
                            "source": "overlay: the cluster does not publish a size",
                            "unit": "bytes",
                        },
                        "floor": {
                            "value": 0,
                            "provenance": "measured",
                            "source": "overlay: no local floor",
                            "unit": "bytes",
                        },
                    },
                    "caps": {
                        "latency_us": {
                            "value": None,
                            "provenance": "absent",
                            "source": "overlay: unmeasured",
                            "unit": "us",
                        },
                        "bandwidth_gbs": {
                            "value": None,
                            "provenance": "absent",
                            "source": "overlay: unmeasured",
                            "unit": "GB/s",
                        },
                        "aperture_bytes": {
                            "value": None,
                            "provenance": "absent",
                            "source": "overlay: not aperture gated",
                            "unit": "bytes",
                        },
                        "ledger_key": "mooncake",
                    },
                    "health": {"reachable": True, "verdict": "ok"},
                    "transport": {"name": "rpc"},
                    "properties": {"pointer_io": "yes"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.json"
            path.write_text(json.dumps(overlay))
            profile = load_profile(path, over_bundled=True)
        self.assertEqual(profile.profile_id, "other-rig")
        self.assertEqual(
            profile.tier("fs:rig-1:/spinning").caps.bandwidth_gbs.value, 6.4
        )
        # untouched fields survive the merge
        self.assertTrue(profile.tier("fs:rig-1:/spinning").caps.latency_us.is_absent)
        self.assertEqual(
            profile.tier("fs:rig-1:/spinning").properties["fs_type"], "zfs"
        )
        # the bundled tiers are still there, in order, plus the new one
        self.assertEqual(profile.tiers[0].id, "host:rig-1")
        self.assertEqual(profile.tiers[-1].id, "blob:mooncake:cluster-a")
        self.assertIs(profile.tiers[-1].kind, TierKind.BLOB)

    def test_an_overlay_rate_still_needs_its_own_source(self):
        overlay = {
            "schema_version": 1,
            "profile_id": "other-rig",
            "host": "rig-1",
            "tiers": [
                {
                    "id": "fs:rig-1:/spinning",
                    "caps": {
                        "bandwidth_gbs": {
                            "value": 6.4,
                            "provenance": "measured",
                            "source": "",
                            "unit": "GB/s",
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.json"
            path.write_text(json.dumps(overlay))
            with self.assertRaises(ProfileError):
                load_profile(path, over_bundled=True)

    def test_a_missing_profile_says_so(self):
        with self.assertRaises(ProfileError):
            load_profile(Path("/nonexistent/memtier.json"))


class DeviceBindingTest(unittest.TestCase):
    def test_a_known_model_binds_its_measured_caps_and_the_live_capacity(self):
        profile = bundled_profile()
        cards = (
            CardFact(
                uuid="GPU-1111",
                model="NVIDIA GeForce RTX 5090",
                total_bytes=34_000_000_000,
            ),
        )
        tier = bind_device_tiers(profile, cards)[0]
        self.assertEqual(tier.id, "vram:GPU-1111")
        self.assertEqual(tier.caps.bandwidth_gbs.value, 1558.0)
        self.assertEqual(tier.capacity.total.value, 34_000_000_000.0)
        self.assertIn("GPU-1111", tier.capacity.total.source)
        self.assertEqual(tier.capacity.corridor, 400 * 1024 * 1024)

    def test_an_unknown_model_gets_absent_caps_rather_than_a_roofline(self):
        """#348b D4, one layer down: a card nobody profiled must not read as an
        extremely slow but usable tier. Can-fail proof: substituting any
        default bandwidth here makes the tier selectable against a floor."""
        profile = bundled_profile()
        cards = (
            CardFact(uuid="GPU-9999", model="NVIDIA H100", total_bytes=80_000_000_000),
        )
        tier = bind_device_tiers(profile, cards)[0]
        self.assertTrue(tier.caps.bandwidth_gbs.is_absent)
        self.assertIn("no measured record", tier.caps.bandwidth_gbs.source)
        self.assertIn("model_template", tier.properties)
        self.assertEqual(tier.capacity.total.value, 80_000_000_000.0)


class LocalFactsTest(unittest.TestCase):
    def test_statvfs_facts_set_the_floor_from_what_is_already_in_use(self):
        profile = bundled_profile()
        registry = TierRegistry.from_profile(
            profile,
            LocalFacts(
                filesystems=(
                    FilesystemFact(
                        mount="/spinning",
                        total_bytes=2_000_000_000_000,
                        available_bytes=700_000_000_000,
                    ),
                )
            ),
        )
        tier = registry.get("fs:rig-1:/spinning")
        self.assertEqual(tier.capacity.total.value, 2_000_000_000_000.0)
        self.assertEqual(tier.capacity.floor.value, 1_300_000_000_000.0)
        self.assertEqual(tier.capacity.headroom().require("h"), 700_000_000_000.0)

    def test_host_facts_turn_an_absent_floor_into_a_measured_one(self):
        """The declared host floor is absent on purpose (#400's blind spot at
        this tier); a live reading is what makes the tier reservable at all."""
        profile = bundled_profile()
        declared = profile.tier("host:rig-1")
        self.assertTrue(declared.capacity.floor.is_absent)
        registry = TierRegistry.from_profile(
            profile,
            LocalFacts(host_total_bytes=100, host_available_bytes=40),
        )
        tier = registry.get("host:rig-1")
        self.assertEqual(tier.capacity.floor.value, 60.0)
        self.assertEqual(tier.capacity.headroom().require("h"), 40.0)

    def test_a_tier_with_no_matching_fact_is_left_alone(self):
        """A fact that could not be collected leaves the measured-earlier
        number in place rather than overwriting it with a zero."""
        profile = bundled_profile()
        before = profile.tier("fs:rig-1:/spinning")
        after = apply_local_facts([before], LocalFacts())[0]
        self.assertEqual(after.capacity.total.value, before.capacity.total.value)

    def test_collect_local_facts_reads_this_machine(self):
        """Executes the live gatherer -- no driver, no card, no network. The
        guard is against the path being written and never run."""
        with tempfile.TemporaryDirectory() as tmp:
            facts = collect_local_facts(mounts=(tmp, "/nonexistent-mount"))
        self.assertGreater(facts.host_total_bytes, 0)
        self.assertGreater(facts.host_available_bytes, 0)
        self.assertEqual(len(facts.filesystems), 1)
        self.assertGreater(facts.filesystems[0].total_bytes, 0)
        self.assertEqual(facts.cards, ())


GIB = 1024**3


class TestHonestHostMemory(unittest.TestCase):
    """#551: the host-memory number a PINNED pool is sized against must not
    come from a source that can state an impossibility.

    Inside a container ``/proc/meminfo`` is synthesised by lxcfs and does not
    describe what this process may have. Two lies, both seen on this rig:
    ``MemAvailable`` can EXCEED ``MemTotal`` -- an arithmetic impossibility, so
    a guard comparing a request against it compares against a number that
    denotes nothing -- and with ``memory.max`` unlimited it reports the HOST's
    figures on a box other containers are also spending. ``/sys/fs/cgroup`` is
    the honest source. That matters here and not merely in principle, because
    an over-commit of PINNED memory is the OOM killer choosing a victim, not a
    swap.

    Every input is passed in, so none of this asserts anything about the
    machine the test runs on (directive #434).

    CAN-FAIL: drop the clamp in ``honest_host_memory_bytes`` and
    ``test_the_impossible_meminfo_reading_is_clamped`` goes red; prefer
    ``/proc/meminfo`` over a finite ``memory.max`` and
    ``test_a_finite_cgroup_limit_wins`` goes red.
    """

    def test_the_impossible_meminfo_reading_is_clamped(self):
        """MemAvailable > MemTotal, no cgroup limit: the observed lxcfs bug."""
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=125 * GIB,  # impossible on a real machine
            cgroup_max=None,
            cgroup_anon=None,
            cgroup_kernel=None,
        )
        self.assertEqual(total, 120 * GIB)
        self.assertEqual(available, 120 * GIB)
        self.assertLessEqual(available, total)

    def test_a_finite_cgroup_limit_wins_over_meminfo(self):
        """The ceiling this process group is killed at beats the host's."""
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=118 * GIB,
            cgroup_max=32 * GIB,
            cgroup_anon=2 * GIB,
            cgroup_kernel=1 * GIB,
        )
        self.assertEqual(total, 32 * GIB)
        # page cache is reclaimable and deliberately NOT subtracted
        self.assertEqual(available, 29 * GIB)

    def test_a_limit_above_the_machine_is_capped_by_the_machine(self):
        total, _ = honest_host_memory_bytes(
            meminfo_total=64 * GIB,
            meminfo_available=60 * GIB,
            cgroup_max=1024 * GIB,
            cgroup_anon=0,
            cgroup_kernel=0,
        )
        self.assertEqual(total, 64 * GIB)

    def test_resident_accounting_also_bounds_the_unlimited_case(self):
        """No ceiling, but the cgroup knows what it already spent."""
        _total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=118 * GIB,
            cgroup_max=None,
            cgroup_anon=100 * GIB,
            cgroup_kernel=2 * GIB,
        )
        self.assertEqual(available, 18 * GIB)

    def test_an_unestablishable_number_stays_none(self):
        """A caller that cannot get a number must be told so, not handed a
        guess -- the guard then skips rather than refusing a good boot."""
        self.assertEqual(
            honest_host_memory_bytes(None, None, None, None, None), (None, None)
        )
        # a ceiling with no usage accounting says nothing about what is free
        self.assertEqual(
            honest_host_memory_bytes(None, None, 32 * GIB, None, None),
            (32 * GIB, None),
        )

    def test_available_is_never_negative(self):
        _total, available = honest_host_memory_bytes(
            meminfo_total=8 * GIB,
            meminfo_available=8 * GIB,
            cgroup_max=8 * GIB,
            cgroup_anon=9 * GIB,  # over the ceiling: accounting skew
            cgroup_kernel=1 * GIB,
        )
        self.assertEqual(available, 0)

    def test_the_live_path_runs_on_this_machine(self):
        """Executes the real reader -- the guard against desk-written code."""
        total, available = host_memory_bytes_for_pinning()
        self.assertIsNotNone(total)
        self.assertGreater(total, 0)
        if available is not None:
            self.assertLessEqual(available, total)


if __name__ == "__main__":
    unittest.main()
