"""No rig constants anywhere: the generality falsifiers (#407 / directive #434).

This file exists because of a defect the package shipped and its 82 tests did
not see. Cut 1's ``TierRegistry.from_profile()`` defaulted to the bundled
rig-1 profile, so a machine nobody had ever measured received one particular
development box's host RAM at an estimated 38 GB/s, its ZFS pool at 1.8 GB/s
and its 40G peer -- and every existing test passed, because every existing test
either built its tiers by hand or ran on the rig the profile describes.

The three synthetic rigs below are the fix for that blind spot. None of them
resembles the development box, and each is chosen for a different way a
registry can smuggle a local assumption:

``NVLINK_ISLAND``
    Four cards with a fast interconnect and a small host. If anything in the
    package believes peer VRAM is exotic and host RAM is plentiful, this rig
    finds it.

``EIGHT_EQUAL``
    Eight identical cards -- the shape this fork is *least* like, since its
    whole delta is heterogeneous TP. A model signature over eight identical
    cards is one entry with a count, and any code that walked cards
    positionally reads wrong here.

``INT8_SMALL_MIX``
    Five small cards of three different models and a large host. The
    many-weak-cards inversion of the development box: here host RAM is the
    plentiful tier and VRAM is scarce, so any ordering that assumes VRAM is
    always the biggest number comes out backwards.

Three groups of falsifier:

1. **The bootstrap group** -- an unmatched machine gets its own memories and
   nobody else's, with absent costs.
2. **The leak group** -- perturbing a profile changes the selection. A test
   that passes whatever the profile says is not testing selection.
3. **The grep guard** -- the package's executable source, with docstrings and
   comments stripped by AST (the #421 detector-B2 technique), contains none of
   the development rig's measured numbers or names. Prose may explain the rig;
   code may not encode it.

    python -m pytest test/registered/unit/memtier/test_tier_generality.py -v
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.memtier import bootstrap, fingerprint as fp_mod
from sglang.srt.memtier.bootstrap import BOOTSTRAP_PROFILE_ID, bootstrap_tiers
from sglang.srt.memtier.fingerprint import MatchScope, fingerprint_from_facts
from sglang.srt.memtier.profile import (
    BUNDLED_PROFILE_PATH,
    CardFact,
    FilesystemFact,
    LocalFacts,
)
from sglang.srt.memtier.profile_store import select_profile
from sglang.srt.memtier.registry import TierRegistry
from sglang.srt.memtier.tiers import TierKind, Volatility
from sglang.srt.planner.cost_model import Provenance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3
PACKAGE_DIR = Path(bootstrap.__file__).parent


def rig(models, *, host_gib, mounts=()):
    return LocalFacts(
        cards=tuple(
            CardFact(
                uuid=f"GPU-{i:08x}-0000-0000-0000-{i:012x}",
                model=model,
                total_bytes=int(gib * GIB),
                bdf=f"0000:{0x10 + i:02x}:00.0",
            )
            for i, (model, gib) in enumerate(models)
        ),
        host_total_bytes=int(host_gib * GIB),
        host_available_bytes=int(host_gib * GIB * 0.8),
        filesystems=tuple(
            FilesystemFact(mount=m, total_bytes=t, available_bytes=a)
            for m, t, a in mounts
        ),
    )


NVLINK_ISLAND = rig(
    [("NVIDIA A100-SXM4-80GB", 80)] * 4,
    host_gib=48,
    mounts=(("/nvme0", 2 * 10**12, 1 * 10**12),),
)
EIGHT_EQUAL = rig(
    [("NVIDIA H100 80GB HBM3", 80)] * 8,
    host_gib=2048,
    mounts=(("/mnt/data", 60 * 10**12, 55 * 10**12),),
)
INT8_SMALL_MIX = rig(
    [
        ("NVIDIA GeForce RTX 4060 Ti", 16),
        ("NVIDIA GeForce RTX 4060 Ti", 16),
        ("NVIDIA GeForce GTX 1080 Ti", 11),
        ("Intel Arc A770", 16),
        ("Intel Arc A770", 16),
    ],
    host_gib=512,
    mounts=(("/spill", 1 * 10**12, 9 * 10**11),),
)

ALL_RIGS = {
    "nvlink_island": NVLINK_ISLAND,
    "eight_equal": EIGHT_EQUAL,
    "int8_small_mix": INT8_SMALL_MIX,
}


class TestBootstrapOnUnknownHardware(unittest.TestCase):
    """An unmatched machine gets its own memories, and nobody else's."""

    def registry_for(self, facts, name):
        registry, selection = TierRegistry.for_machine(
            facts,
            selection=select_profile(
                fingerprint_from_facts(facts), paths=[BUNDLED_PROFILE_PATH]
            ),
            fs_types={f.mount: "ext4" for f in facts.filesystems},
            host=name,
        )
        return registry, selection

    def test_the_bundled_profile_never_applies_to_a_foreign_rig(self):
        for name, facts in ALL_RIGS.items():
            with self.subTest(rig=name):
                registry, selection = self.registry_for(facts, name)
                self.assertIsNone(selection.profile)
                self.assertIs(selection.scope, MatchScope.NONE)
                self.assertEqual(registry.profile_id, BOOTSTRAP_PROFILE_ID)

    def test_no_tier_belongs_to_a_machine_that_was_not_enumerated(self):
        for name, facts in ALL_RIGS.items():
            with self.subTest(rig=name):
                registry, _ = self.registry_for(facts, name)
                for tier in registry.tiers():
                    self.assertEqual(tier.host, name, tier.id)
                declared_mounts = {f.mount for f in facts.filesystems}
                for tier in registry.of_kind(TierKind.FILESYSTEM):
                    self.assertIn(tier.parsed.mount, declared_mounts)

    def test_every_card_becomes_exactly_one_device_tier(self):
        for name, facts in ALL_RIGS.items():
            with self.subTest(rig=name):
                registry, _ = self.registry_for(facts, name)
                device_ids = {
                    t.parsed.card_key for t in registry.of_kind(TierKind.DEVICE)
                }
                self.assertEqual(device_ids, {c.uuid for c in facts.cards})

    def test_sizes_are_measured_and_costs_are_absent(self):
        for name, facts in ALL_RIGS.items():
            with self.subTest(rig=name):
                registry, _ = self.registry_for(facts, name)
                for tier in registry.tiers():
                    self.assertIs(
                        tier.capacity.total.provenance, Provenance.MEASURED, tier.id
                    )
                    for cap in (
                        tier.caps.bandwidth_gbs,
                        tier.caps.latency_us,
                        tier.caps.aperture_bytes,
                    ):
                        self.assertTrue(cap.is_absent, f"{tier.id}: {cap.source}")

    def test_every_absence_names_the_probe_that_would_fill_it(self):
        registry, _ = self.registry_for(EIGHT_EQUAL, "eight_equal")
        for tier in registry.tiers():
            self.assertIn("probe", tier.caps.bandwidth_gbs.source.lower(), tier.id)

    def test_a_tmpfs_mount_is_never_persistent(self):
        """#89's silent-correctness hole, made checkable from the mount type."""
        facts = rig(
            [("Some Card", 8)],
            host_gib=32,
            mounts=(("/dev/shm", 16 * 10**9, 16 * 10**9),),
        )
        tiers = bootstrap_tiers(facts, host="unit", fs_types={"/dev/shm": "tmpfs"})
        shm = next(t for t in tiers if t.kind is TierKind.FILESYSTEM)
        self.assertIs(shm.volatility, Volatility.EXPENSIVE_OK)
        self.assertEqual(shm.properties["persistent_across_reboot"], "no")
        # can-fail: the same mount on a real filesystem IS persistent.
        on_disk = bootstrap_tiers(facts, host="unit", fs_types={"/dev/shm": "xfs"})
        self.assertIs(
            next(t for t in on_disk if t.kind is TierKind.FILESYSTEM).volatility,
            Volatility.PERSISTENT,
        )

    def test_an_unreadable_mount_type_yields_unknown_and_not_a_default(self):
        facts = rig([("Some Card", 8)], host_gib=32, mounts=(("/x", 10**9, 10**9),))
        tiers = bootstrap_tiers(facts, host="unit", fs_types={})
        fs = next(t for t in tiers if t.kind is TierKind.FILESYSTEM)
        self.assertEqual(fs.properties["flock"], "unknown")
        self.assertEqual(fs.properties["persistent_across_reboot"], "unknown")

    def test_a_network_filesystem_reports_no_flock(self):
        self.assertEqual(bootstrap.filesystem_kind_properties("nfs4")["flock"], "no")
        self.assertEqual(bootstrap.filesystem_kind_properties("xfs")["flock"], "yes")

    def test_from_profile_no_longer_has_a_rig_default(self):
        """The #421 F6 half that was quiet: an argument-less registry."""
        with self.assertRaises(TypeError):
            TierRegistry.from_profile()


class TestLeakFalsifier(unittest.TestCase):
    """Perturb the profile, and the selection must move."""

    def synthetic_profile(self, facts, **extra):
        doc = {
            "schema_version": 1,
            "profile_id": extra.pop("profile_id", "synthetic"),
            "host": extra.pop("host", "synthetic"),
            "hardware": fp_mod.hardware_block(fingerprint_from_facts(facts)),
            "device_models": {},
            "tiers": [],
        }
        doc.update(extra)
        return doc

    def select(self, document, facts):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(document))
            return select_profile(fingerprint_from_facts(facts), paths=[path])

    def test_perturbing_one_card_uuid_flips_exact_to_model(self):
        document = self.synthetic_profile(NVLINK_ISLAND)
        self.assertIs(self.select(document, NVLINK_ISLAND).scope, MatchScope.EXACT)
        document["hardware"]["cards"][0]["uuid"] = (
            "GPU-ffffffff-dead-beef-0000-000000000000"
        )
        self.assertIs(self.select(document, NVLINK_ISLAND).scope, MatchScope.MODEL)

    def test_perturbing_a_card_model_flips_model_to_none(self):
        document = self.synthetic_profile(EIGHT_EQUAL)
        self.assertIs(self.select(document, EIGHT_EQUAL).scope, MatchScope.EXACT)
        document["hardware"]["cards"] = []
        document["hardware"]["models"] = ["8x H200 141GB HBM3e:141GiB"]
        self.assertIs(self.select(document, EIGHT_EQUAL).scope, MatchScope.NONE)

    def test_a_model_scoped_profile_contributes_no_host_tier(self):
        """The load-bearing licence rule, with the value it protects."""
        document = self.synthetic_profile(
            INT8_SMALL_MIX,
            profile_id="twin",
            host="theirs",
            tiers=[
                {
                    "id": "host:theirs",
                    "kind": "host",
                    "host": "theirs",
                    "volatility": "expensive_ok",
                    "capacity": {
                        "total": {
                            "value": 1.0,
                            "provenance": "measured",
                            "source": "theirs",
                        },
                        "floor": {
                            "value": 0.0,
                            "provenance": "measured",
                            "source": "theirs",
                        },
                    },
                    "caps": {
                        "latency_us": {
                            "value": None,
                            "provenance": "absent",
                            "source": "x",
                        },
                        "bandwidth_gbs": {
                            "value": 999.0,
                            "provenance": "measured",
                            "source": "measured on THEIR box",
                        },
                        "aperture_bytes": {
                            "value": None,
                            "provenance": "absent",
                            "source": "x",
                        },
                        "ledger_key": "host_ram",
                    },
                    "health": {"reachable": True},
                }
            ],
        )
        # Same models, different serials -> MODEL scope.
        twin_facts = rig(
            [
                ("NVIDIA GeForce RTX 4060 Ti", 16),
                ("NVIDIA GeForce RTX 4060 Ti", 16),
                ("NVIDIA GeForce GTX 1080 Ti", 11),
                ("Intel Arc A770", 16),
                ("Intel Arc A770", 16),
            ],
            host_gib=16,
        )
        twin_facts = LocalFacts(
            cards=tuple(
                CardFact(
                    uuid=f"GPU-{i:08x}-1111-1111-1111-{i:012x}",
                    model=c.model,
                    total_bytes=c.total_bytes,
                    bdf=c.bdf,
                )
                for i, c in enumerate(twin_facts.cards)
            ),
            host_total_bytes=twin_facts.host_total_bytes,
            host_available_bytes=twin_facts.host_available_bytes,
        )
        selection = self.select(document, twin_facts)
        self.assertIs(selection.scope, MatchScope.MODEL)
        self.assertEqual(selection.profile.tiers, ())
        # can-fail: on the machine it WAS measured on, the tier does arrive.
        self.assertEqual(len(self.select(document, INT8_SMALL_MIX).profile.tiers), 1)

    def test_the_selection_record_names_what_it_passed_over(self):
        document = self.synthetic_profile(NVLINK_ISLAND, profile_id="not-yours")
        selection = self.select(document, EIGHT_EQUAL)
        self.assertIsNone(selection.profile)
        self.assertIn("not-yours", selection.render())


class TestNoRigConstantsInCode(unittest.TestCase):
    """The grep guard, on executable source only.

    Docstrings and comments in this package *do* discuss the development rig,
    on purpose: an explanation of why a number is absent is worth more than the
    absence. What must never appear is a rig figure in code -- a literal, a
    default, a table entry. The two are separated the way #421's detector B2
    separates them: unparse the AST with docstrings removed, which drops
    comments as a side effect of unparsing, and search what is left.
    """

    #: Measured figures and names from the development rig's own profile.
    #: If one of these appears in executable source, the package has grown a
    #: constant about one machine.
    FORBIDDEN_NUMBERS = (
        1558.0,
        723.0,
        1533.5,
        718.2,  # card membw figures
        38.0,
        2.83,
        1.47,
        4.99,
        0.83,
        1.8,  # host / wire / disk figures
        34190917632,
        21474836480,  # card totals
        105763569664,
        729000000000,  # host RAM, pool free space
        419430400,
        100663296,
        268435456,  # corridor, BAR1 contiguous/nominal
    )
    FORBIDDEN_TEXT = (
        "rig-1",
        "rig-2",
        "CT999",
        "/spinning",
        "RTX 5090",
        "RTX 3080",
        "2080 Ti",
        "htsglang-rig-1",
    )

    def executable_sources(self):
        for path in sorted(PACKAGE_DIR.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    body = getattr(node, "body", None)
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        node.body = body[1:] or [ast.Pass()]
            yield path, ast.unparse(ast.fix_missing_locations(tree))

    def test_no_measured_rig_number_appears_in_executable_source(self):
        offences = []
        for path, source in self.executable_sources():
            for number in self.FORBIDDEN_NUMBERS:
                if str(number) in source:
                    offences.append(f"{path.name}: {number}")
        self.assertEqual(offences, [], f"rig constants in code: {offences}")

    def test_no_rig_name_appears_in_executable_source(self):
        offences = []
        for path, source in self.executable_sources():
            for text in self.FORBIDDEN_TEXT:
                if text in source:
                    offences.append(f"{path.name}: {text!r}")
        self.assertEqual(offences, [], f"rig names in code: {offences}")

    def test_the_detector_can_fail(self):
        """A guard that cannot fire is not a guard.

        Two halves: the stripper must keep executable strings (so a real
        offence is visible), and it must drop docstrings (so prose about the
        rig is not an offence).
        """
        module = ast.parse(
            '"""A docstring naming rig-1 and 1558.0."""\n'
            "X = 'rig-1'\n"
            "# a comment naming 1558.0\n"
        )
        for node in ast.walk(module):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        stripped = ast.unparse(module)
        self.assertIn("rig-1", stripped)  # the assignment survives
        self.assertNotIn("1558.0", stripped)  # docstring and comment do not

    def test_nothing_calls_the_bundled_profile(self):
        """rig1.json may say anything; nothing may LOAD it unconditionally.

        Re-exporting the name is fine -- ``__init__`` does. Calling it is the
        defect: that call was the pre-#434 default of
        ``TierRegistry.from_profile`` and the whole reason a foreign machine
        could receive one particular box's numbers. Checked as an AST call
        rather than a substring so the export survives and the call does not.
        """
        callers = []
        for path in sorted(PACKAGE_DIR.glob("*.py")):
            if path.name == "profile.py":  # defines it
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", None) or getattr(
                        node.func, "attr", None
                    )
                    if name == "bundled_profile":
                        callers.append(f"{path.name}:{node.lineno}")
        self.assertEqual(callers, [], f"bundled_profile() is called at {callers}")

    def test_no_module_names_the_bundled_file(self):
        for path, source in self.executable_sources():
            if path.name == "profile.py":  # BUNDLED_PROFILE_PATH is defined there
                continue
            self.assertNotIn("rig1.json", source, path.name)

    def test_the_bundled_profile_can_only_match_by_model(self):
        """It ships with no card rows, so it cannot license a host or a disk."""
        document = json.loads(BUNDLED_PROFILE_PATH.read_text())
        self.assertEqual(document["hardware"]["cards"], [])
        self.assertTrue(document["hardware"]["models"])


if __name__ == "__main__":
    unittest.main()
