"""AP0 reference harness of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 3 row AP0, R6, R9).

``weg2/propose_oracle.py`` is the ORACLE stage of ``propose()``: the launcher's real ``main()`` with ``--dry-run`` on a
replayed NVML inventory, hermetic (private SHM dir, pinned quiet-host meminfo/cgroup, scrubbed environment).  This file
proves the harness, and pins the REFERENCE RIG (RTX 5090 + 2x RTX 3080, ``nvml_devices_1378.json``) for the release
profiles as GOLDEN dumps: ``propose()`` of AP-C must reproduce these plans with 0 diff lines (plan 1.3 A1).

Parts

* ``TestReplaySynthesis``   (a) replay rows from a real ``flliper.hardware/1`` document and from card-catalog rows; the
                            launcher's own reader (``registry.nvml._replay_devices``) accepts them.
* ``TestProfileLaunchInput`` (d) the three release profiles as argv + environment: bash arrays read whole (count checked by
                            an independent ``bash``), the ``source`` chain of the dual profile, ``_form`` environment,
                            runtime placeholders, image paths.
* ``TestDumpTools``         (c)(e) normalisation, live-box mask (``plan_diff.py:6-17``), diff, plan-dump parser.
* ``TestDryRunGolden``      (b)(d) THE reference test: dump == golden with 0 diff lines for 27B Flip and 27B Dual on the
                            reference rig (and with the replay synthesised from a hardware profile, and with another
                            scratch dir: the golden must not depend on the run); environment hygiene; ``--force`` plumbing;
                            NF in the abl form (see below).

The profiles are SNAPSHOTS under ``fixtures/planer_1006/profiles`` (copies of ``/spinning/gpu-arb/docker/profiles_release/
27b-base.env``, ``27b-nvfp4-dual.env`` + its ``27b-nvfp4.pchunk.json``, and ``docker/profiles/nf-int4-h6-abl.env``; sha256 in
``PROVENANCE`` below): a golden of a profile that moves under it would be a test that moves with the rig.  The goldens are
box-bound like every dry-run golden of this directory (they carry the census/evidence files and the sibling checkpoint
headers of this box); the tests SKIP, with the reason, where those inputs are absent.

NF (abl form, R9): ``launch_nf-int4-h6-abl.json`` (the argv and environment of the abl profile) is pinned here.  The
DRY-RUN golden ``plan_nf_abl_n3.txt`` is NOT in the tree: on the box that produced this file the four NF checkpoint dirs
(``Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp`` and the MTP draft ``...albucino-abl-wxp``) are EMPTY mount
points, and the launcher refuses the plan at W128 (draft header unreadable) -- a refusal of the missing data, not a plan.
``test_nf_abl_dump_equals_golden`` runs as soon as the checkpoints and the golden exist; generate it on the box that has
them with::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=python python3 -m sglang.srt.weg2.propose_oracle golden \\
        --profile test/registered/unit/weg2/fixtures/planer_1006/profiles/nf-int4-h6-abl.env \\
        --replay test/registered/unit/weg2/fixtures/xchg_launch_replay_0911/nvml_devices_1378.json \\
        --out test/registered/unit/weg2/fixtures/planer_1006/golden/plan_nf_abl_n3.txt

REGENERATE the 27B goldens after a change that moves the plan ON PURPOSE (the same command with ``27b-base.env`` /
``27b-nvfp4-dual.env`` and ``plan_27b_flip_n3.txt`` / ``plan_27b_dual_n3.txt``; the launch JSONs with the ``launch``
sub-command).  Measured 2026-10-06 on tree 173161c595: run-to-run diff 0 lines (two runs, two scratch dirs); against
``deskq/work/hw1004/plan_dump.py`` on the same profile and the same inputs (PROFILE_ARGS only, no exported env) 0 value
differences, only the path tokens this harness normalises.

GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.registry import nvml as nvml_registry
    from sglang.srt.weg2 import launcher, refusals
    from sglang.srt.weg2 import propose_oracle as O
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
GOLDEN = os.path.join(FIX, "golden")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"

#: sha256 of the snapshot files (the live files they were copied from, 2026-10-06)
PROVENANCE = {
    "27b-base.env": "7f48188299747a86c76b7eb591b57de07cd4897cd65a058fa9859295ca6026b9",
    "27b-nvfp4-dual.env": "b758c03aa60632872b51756a7c466b6ae2f2c5adc5183635ae1015e7b0065005",
    "27b-nvfp4.pchunk.json": "a56d1c7a4fb93206e6251540db80dc36656355daed95a3faf1e19de5fc99237c",
    "nf-int4-h6-abl.env": "12f9a824b3d9fc66e065ae6856ca91f2c48be43225c55e302ddfab4fd5cb8092",
}
PROFILE_FILES = {"flip": "27b-base", "dual": "27b-nvfp4-dual", "nf": "nf-int4-h6-abl"}


def _p(name: str) -> str:
    return os.path.join(PROFILES, name + ".env")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _have(*paths: str) -> bool:
    return all(os.path.exists(p) for p in paths)


_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
_NEEDS_27B_FLIP = unittest.skipUnless(
    _have(_CENSUS_27B, MC + "Qwen3.8-27B-INT8-gdncov/config.json", MC + "Qwen3.8-27B-DFlash2/config.json"),
    "27B INT8 sibling checkpoints / census not on this box",
)
_NEEDS_27B_DUAL = unittest.skipUnless(
    _have(_CENSUS_27B, MC + "Qwen3.8-27B-NVFP4-RadixArk/config.json", MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal/config.json"),
    "27B NVFP4 checkpoints / census not on this box",
)


def _nf_checkpoint_present() -> bool:
    """Target config.json AND a non-empty draft dir (the launcher reads both headers; empty mount points here)."""
    d = MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp"
    dd = MC + "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp"
    return os.path.isfile(os.path.join(d, "config.json")) and os.path.isdir(dd) and bool(os.listdir(dd))


def _load_catalog():
    path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("planer_ap0_kartenplan_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hardware_profile_of(rows, *, with_bar1=False):
    """A REAL ``flliper.hardware/1`` document (``rigmon.hardware_profile.build``) over NVML-shaped cards derived from the
    replay rows; BAR1 / PCIe left unread (None) unless ``with_bar1``."""
    from sglang.srt.rigmon import hardware_profile as HP

    cards = [{
        "nvml_index": r["index"], "uuid": r["uuid"], "name": r["name"], "total_mib": r["total_bytes"] >> 20,
        "cc": [r["cc_major"], r["cc_minor"]], "pci_bus_id": r["pci_bus_id"],
        "bar1_total_mib": 256 if with_bar1 else None, "pcie_max_gen": 4 if with_bar1 else None,
        "pcie_max_width": 16 if with_bar1 else None, "pcie_cur_gen": None, "pcie_cur_width": None,
        "mem_bus_width_bits": None, "mem_clock_max_mhz": None, "sm_clock_max_mhz": None,
        "power_limit_w": None, "power_default_w": None,
    } for r in rows]
    return HP.build(cache_dir=tempfile.mkdtemp(prefix="ap0-hwprof-"), nvml=(cards, "595.58.03", []), now=1.0)


# ---------------------------------------------------------------------------
# (a) replay synthesis
# ---------------------------------------------------------------------------

class TestReplaySynthesis(unittest.TestCase):
    FIELDS = ("index", "uuid", "name", "total_bytes", "pci_bus_id", "reserved_bytes", "cc_major", "cc_minor")

    def test_hardware_profile_reproduces_the_reference_fixture_rows(self):
        ref = O.read_replay(REPLAY_REF)
        doc = _hardware_profile_of(ref)
        self.assertEqual(doc["schema"], "flliper.hardware/1")
        # the profile is in PLANNER order (5090 first), the replay in NVML index order: the launcher orders cards itself
        self.assertEqual([c["nvml_index"] for c in doc["cards"]], [1, 0, 2])
        got = O.replay_from_hardware_profile(doc)
        self.assertEqual([r["index"] for r in got], [0, 1, 2])
        for g, r in zip(got, ref):
            for k in self.FIELDS:
                self.assertEqual(g[k], r[k], k)
        # a property the profile does not hold is LEFT OUT (unknown), never filled
        for g in got:
            for k in ("bar1_total_bytes", "pcie_max_gen", "mem_bus_width_bits", "mem_clock_max_mhz"):
                self.assertNotIn(k, g)

    def test_constants_are_the_launchers(self):
        self.assertEqual(O.ENV_NVML_REPLAY, nvml_registry.ENV_NVML_REPLAY)
        self.assertEqual(O.ENV_FORCED_BOOT, refusals.ENV_FORCED_BOOT)
        self.assertEqual(set(O._IDENTITY_FIELDS), set(nvml_registry.IDENTITY_FIELDS))

    def test_hardware_profile_property_values_are_carried(self):
        doc = _hardware_profile_of(O.read_replay(REPLAY_REF), with_bar1=True)
        got = O.replay_from_hardware_profile(doc)
        self.assertTrue(all(g["bar1_total_bytes"] == 256 << 20 and g["pcie_max_gen"] == 4 and g["pcie_max_width"] == 16
                            for g in got))

    def test_hardware_profile_refusals(self):
        with self.assertRaises(ValueError):
            O.replay_from_hardware_profile({"schema": "flliper.server/1", "cards": [{}]})
        with self.assertRaises(ValueError):
            O.replay_from_hardware_profile({"schema": "flliper.hardware/1", "cards": []})
        doc = _hardware_profile_of(O.read_replay(REPLAY_REF))
        doc["cards"][0]["vram_total_mib"] = {"v": None, "src": "nicht gemessen"}
        with self.assertRaises(ValueError):
            O.replay_from_hardware_profile(doc)

    def test_catalog_cards_match_the_rig_fixture_and_hw_sim(self):
        cat = _load_catalog()
        if cat is None:
            self.skipTest("dashboard catalog not in this tree")
        from sglang.srt.weg2 import hw_sim

        ref = {r["name"]: r for r in O.read_replay(REPLAY_REF)}
        entries = [cat.card("rtx5090-32"), cat.card("rtx3080-20"), cat.card("rtx3080-20")]
        rows = O.replay_from_catalog(entries)
        self.assertEqual([r["index"] for r in rows], [0, 1, 2])
        self.assertEqual(len({r["uuid"] for r in rows}), 3)
        for r, e in zip(rows, entries):
            self.assertEqual(r["name"], e["nvml_name"])
            self.assertEqual(r["total_bytes"], ref[r["name"]]["total_bytes"])       # catalog usable_mib = the NVML record
            self.assertEqual((r["cc_major"], r["cc_minor"]), (ref[r["name"]]["cc_major"], ref[r["name"]]["cc_minor"]))
        # memory clock from the nameplate bandwidth == hw_sim's catalog value (+-1 MHz: NVML reports 9501, 14001)
        self.assertLessEqual(abs(rows[1]["mem_clock_max_mhz"] - hw_sim.CATALOG["3080-20G"].mem_clock_mhz), 1)
        self.assertLessEqual(abs(rows[0]["mem_clock_max_mhz"] - hw_sim.CATALOG["5090"].mem_clock_mhz), 1)
        self.assertEqual(rows[0]["mem_bus_width_bits"], 512)
        # no BAR1 in the catalog: left out unless the caller says
        self.assertNotIn("bar1_total_bytes", rows[0])
        self.assertEqual(O.replay_from_catalog(entries, bar1_mib=256)[0]["bar1_total_bytes"], 256 << 20)
        # stable: the same cards give the same UUIDs (the golden depends on it)
        self.assertEqual(O.replay_from_catalog(entries), rows)

    def test_catalog_foreign_card(self):
        cat = _load_catalog()
        if cat is None:
            self.skipTest("dashboard catalog not in this tree")
        rows = O.replay_from_catalog([cat.card("rtx3090-24")] * 4)
        self.assertEqual({r["total_bytes"] for r in rows}, {24576 << 20})
        self.assertEqual(len({r["uuid"] for r in rows}), 4)
        self.assertEqual(len({r["pci_bus_id"] for r in rows}), 4)

    def test_the_launcher_reader_accepts_what_is_written(self):
        cat = _load_catalog()
        if cat is None:
            self.skipTest("dashboard catalog not in this tree")
        rows = O.replay_from_catalog([cat.card("rtx5090-32"), cat.card("rtx3090-24")], bar1_mib=256)
        with tempfile.TemporaryDirectory() as td:
            path = O.write_replay(rows, os.path.join(td, "r.json"))
            old = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
            os.environ[nvml_registry.ENV_NVML_REPLAY] = path
            try:
                devs = nvml_registry.list_devices()
            finally:
                if old is None:
                    os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
                else:
                    os.environ[nvml_registry.ENV_NVML_REPLAY] = old
        self.assertEqual([d.name for d in devs], [r["name"] for r in rows])
        self.assertEqual([d.total_bytes for d in devs], [r["total_bytes"] for r in rows])
        self.assertEqual([(d.cc_major, d.cc_minor) for d in devs], [(12, 0), (8, 6)])
        self.assertEqual(devs[0].bar1_total_bytes, 256 << 20)


# ---------------------------------------------------------------------------
# (d) the profile reader
# ---------------------------------------------------------------------------

def _bash_argv_count(path: str) -> int:
    """PROFILE_ARGS length as bash itself counts it (independent of ``profile_json``'s NUL protocol)."""
    out = subprocess.run(
        ["bash", "-c", 'set +e; export HTSGLANG_INSTRUMENTS=0; source "$1" >/dev/null 2>&1; echo "${#PROFILE_ARGS[@]}"', "x", path],
        capture_output=True, text=True, check=True, cwd=os.path.dirname(path)).stdout
    return int(out.strip())


class TestProfileLaunchInput(unittest.TestCase):
    def test_snapshots_are_the_recorded_files(self):
        import hashlib

        for name, want in PROVENANCE.items():
            with open(os.path.join(PROFILES, name), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), want, name)

    def test_launch_input_equals_golden_json(self):
        for key, name in PROFILE_FILES.items():
            li = O.profile_launch_input(_p(name), asset_dirs=())
            want = json.loads(_read(os.path.join(GOLDEN, "launch_%s.json" % name)))
            self.assertEqual(O.launch_input_doc(li), want, name)

    def test_bash_arrays_are_read_whole(self):
        for name in PROFILE_FILES.values():
            li = O.profile_launch_input(_p(name), asset_dirs=())
            self.assertEqual(len(li.argv), _bash_argv_count(_p(name)), name)
        # the quoted --extra-p value of the NF abl profile is ONE token with its embedded quotes
        nf = O.profile_launch_input(_p("nf-int4-h6-abl"), asset_dirs=()).argv
        extra_p = nf[nf.index("--extra-p") + 1]
        self.assertIn('--json-model-override-args "{\\"language_model_only\\":true}"', extra_p)
        self.assertNotIn("\n", extra_p)

    def test_dual_profile_evaluates_its_source_chain_and_form_env(self):
        dual = O.profile_launch_input(_p("27b-nvfp4-dual"), asset_dirs=())
        base = O.profile_launch_input(_p("27b-base"), asset_dirs=())
        self.assertEqual(dual.vars["PROFILE_NAME"], "27b-nvfp4-dual")
        self.assertIn("--dual-share", dual.argv)
        self.assertNotIn("--dual-share", base.argv)
        # the MPS opt-in is a _form env the LAUNCHER PROCESS reads (profile header): it must reach the dry run
        self.assertEqual(dual.env["SGLANG_WEG2_DUAL_MPS_OPT_IN"], "1")
        self.assertNotIn("SGLANG_WEG2_DUAL_MPS_OPT_IN", base.env)
        self.assertEqual(dual.model, MC + "Qwen3.8-27B-NVFP4-RadixArk")

    def test_runtime_placeholders_become_tag_evidence_dir_and_arb(self):
        a = O.profile_launch_input(_p("nf-int4-h6-abl"), tag="tagA", asset_dirs=())
        b = O.profile_launch_input(_p("nf-int4-h6-abl"), tag="tagB", asset_dirs=())
        self.assertEqual(a.env["SGLANG_MOE_COLD_TIER_INSTANCE"], "tagA")
        self.assertEqual(b.env["SGLANG_MOE_COLD_TIER_INSTANCE"], "tagB")
        self.assertNotIn("@@", json.dumps(O.launch_input_doc(a)))

    def test_instruments_switch_reaches_the_argv(self):
        off = O.profile_launch_input(_p("nf-int4-h6-abl"), instruments="0", asset_dirs=())
        on = O.profile_launch_input(_p("nf-int4-h6-abl"), instruments="1", asset_dirs=())
        self.assertNotEqual(off.argv, on.argv)          # NF puts NF_ENV_*_INSTR into --env-p/-d
        self.assertEqual(off.env["HTSGLANG_INSTRUMENTS"], "0")
        self.assertEqual(on.env["HTSGLANG_INSTRUMENTS"], "1")

    def test_image_paths_resolve_to_host_files_or_are_listed(self):
        li = O.profile_launch_input(_p("nf-int4-h6-abl"), asset_dirs=())
        self.assertTrue(any("/opt/htsglang/profiles/" in a for a in li.argv))     # untouched without asset dirs
        if not _have(_CENSUS_27B, "/spinning/gpu-arb/weg2/corridor_budget_sample_nextflash_0921.json"):
            self.skipTest("rig asset dirs not on this box")
        li = O.profile_launch_input(_p("nf-int4-h6-abl"))
        self.assertEqual(li.unresolved_paths, [])
        self.assertFalse(any("/opt/htsglang/profiles/" in a for a in li.argv))
        # a path nothing resolves stays and is LISTED
        with tempfile.TemporaryDirectory() as td:
            li = O.profile_launch_input(_p("nf-int4-h6-abl"), asset_dirs=(td,))
        self.assertTrue(li.unresolved_paths)
        self.assertTrue(all("/opt/htsglang/profiles/" in p for p in li.unresolved_paths))

    def test_resolve_image_path_units(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "x.json"), "w").close()
            self.assertEqual(O.resolve_image_path("/opt/htsglang/profiles/nf/x.json", (td,)),
                             (os.path.join(td, "x.json"), True))
            self.assertEqual(O.resolve_image_path("--f=/opt/htsglang/profiles/27b/x.json", (td,)),
                             ("--f=" + os.path.join(td, "x.json"), True))
            self.assertEqual(O.resolve_image_path("/opt/htsglang/profiles/nf/nope.json", (td,)),
                             ("/opt/htsglang/profiles/nf/nope.json", False))
            self.assertEqual(O.resolve_image_path("/plain/path", (td,)), ("/plain/path", True))

    def test_model_name_farm(self):
        with tempfile.TemporaryDirectory() as td:
            sib = os.path.join(td, "sibling")
            os.makedirs(sib)
            for n in ("config.json", "model.safetensors"):
                open(os.path.join(sib, n), "w").write("{}")
            empty = os.path.join(td, "models", "Registry-Name")
            os.makedirs(empty)
            farm = O.ensure_model_dir(empty, siblings=(sib,), farm_root=os.path.join(td, "farm"))
            self.assertEqual(os.path.basename(farm), "Registry-Name")                 # the identity is the NAME
            self.assertTrue(os.path.islink(os.path.join(farm, "config.json")))
            self.assertEqual(O.ensure_model_dir(empty, siblings=(sib,), farm_root=os.path.join(td, "farm")), farm)  # idempotent
            self.assertEqual(O.ensure_model_dir(sib), sib)                           # a real dir stays
            with self.assertRaises(FileNotFoundError):                               # nothing is invented
                O.ensure_model_dir(empty, siblings=(), farm_root=os.path.join(td, "farm2"))


# ---------------------------------------------------------------------------
# (c)(e) dump tools
# ---------------------------------------------------------------------------

class TestDumpTools(unittest.TestCase):
    def test_normalise_replaces_only_per_run_values(self):
        raw = ("[2026-10-06T08:00:01Z] WEG2-LAUNCH === WEG2 BOOT tag=plangate tree=/w/t @ abcdef0123 "
               "(DIRTY: ?? a.py\n?? b/) stamp=1006_080001 dry=True\n"
               "x epoch=1791275135.5 BOOT_TOKEN=plangate:123:456 dir=/scr/store/l3 shm=/scr/shm_EMPTY host=/scr/host/m "
               "models=/tmp/farm/M replay=/scr/nvml_replay.json budget 26064 MiB\n")
        out = O.normalise_dump(raw, tree="/w/t", host="/scr/host", shm="/scr/shm_EMPTY", store="/scr/store",
                               scratch="/scr", farm_root="/tmp/farm", replay="/scr/nvml_replay.json")
        self.assertIn("[TS]", out)
        self.assertIn("tree=<TREE> @ <SHA> (<TREE-STATE>) stamp=<STAMP>", out)
        self.assertIn("epoch=<EPOCH>", out)
        self.assertIn("BOOT_TOKEN=<TOKEN>", out)
        for tok in ("<STORE>/l3", "<SHM>", "<HOST>/m", "<MODELS>/M", "<NVML-REPLAY>"):
            self.assertIn(tok, out)
        self.assertIn("budget 26064 MiB", out)                      # a value is never touched
        self.assertNotIn("/scr", out)
        self.assertNotIn("DIRTY", out)

    def test_clean_tree_normalises_like_a_dirty_one(self):
        a = O.normalise_dump("tag=plangate tree=/w @ 0123456789 (clean) stamp=1_2", tree="/w")
        b = O.normalise_dump("tag=plangate tree=/w @ 0123456789 (DIRTY: ?? x) stamp=1_2", tree="/w")
        self.assertEqual(a.replace("(<TREE-STATE>)", ""), b.replace("(<TREE-STATE>)", ""))

    def test_mask_live_box_masks_exactly_the_listed_readings(self):
        line = ("LIVE nonreclaim=4.67 raw_current=13.10 file_reclaimable=8.00 GiB | c_max=5.00 GiB size=5.03 GiB | "
                "memavail=113.33 GiB anon=4.67 GiB shmem=0.00 GiB | free=264.1 GiB | which has 247.5 GiB free of | "
                "against 264.1 GB free on | foreign_load_now=0.00 GiB | /dev/shm/weg2-xchg-1791275135/x | "
                "WEG2-DRY-RUN-BOX-STATE at=2026-10-06T08:25:35Z | budget 26064 MiB total 32607")
        masked, counts = O.mask_live_box(line)
        self.assertEqual({k for k, v in counts.items() if v}, {n for n, _ in O.LIVE_BOX_RULES})
        self.assertIn("budget 26064 MiB total 32607", masked)
        self.assertIn("c_max=5.00 GiB", masked)                    # the ARC limit stays, only the live size goes
        self.assertNotIn("113.33", masked)

    def test_diff_lines_zero_on_equal_and_nonzero_on_one_changed_value(self):
        g = _read(os.path.join(GOLDEN, "plan_27b_dual_n3.txt"))
        self.assertEqual(O.diff_lines(g, g), [])
        # MUTANT: one budget digit
        self.assertIn("26064 MiB", g)
        bad = g.replace("26064 MiB", "26065 MiB", 1)
        self.assertGreaterEqual(len(O.diff_lines(g, bad)), 3)
        # a live-box reading moving is NOT a diff
        live = g.replace("<box-state-numbers>", "memavail=1.00 GiB anon=2.00 GiB shmem=3.00 GiB")
        self.assertEqual(O.diff_lines(g, live), [])

    def test_parse_plan_dump_of_the_goldens(self):
        for name, cut in (("plan_27b_flip_n3", "40,13,11"), ("plan_27b_dual_n3", "45,10,9")):
            d = O.parse_plan_dump(_read(os.path.join(GOLDEN, name + ".txt")))
            self.assertEqual(d["header"]["rc"], "0")
            self.assertIsNone(d["header"]["exc_type"])
            self.assertEqual(len(d["lines"]) + 1, _read(os.path.join(GOLDEN, name + ".txt")).count("\n"))   # no line dropped
            # the solved cut the launcher printed == the cut it put into group P's argv (two independent places)
            self.assertEqual(d["pp_cut"]["layers"], cut)
            self.assertEqual(d["group_flags"]["P"]["--pp-stage-ratio"], cut)
            self.assertEqual(d["group_flags"]["P"]["--pp-size"], "3")
            self.assertEqual(d["group_flags"]["D"]["--tp-size"], "3")
            self.assertTrue(d["group_argv"]["P"] and d["group_argv"]["D"] and d["group_argv"]["front"])
            # per-rank budgets: 5090 first (card_identity order), the two 3080 behind
            pb = [b for b in d["budgets"] if b["pass"] == "P"]
            self.assertEqual([b["ordinal"] for b in pb], [0, 1, 2])
            self.assertEqual([b["nvml_idx"] for b in pb], [1, 0, 2])
            self.assertEqual(pb[0]["card"], "NVIDIA GeForce RTX 5090")
            if name == "plan_27b_flip_n3":      # the Flip's P argv carries the solved budgets (the Dual's are profile-pinned)
                self.assertEqual(",".join(str(b["mib"]) for b in pb), d["group_flags"]["P"]["--rank-gpu-memory-mib"])
            self.assertEqual(d["forced"], [])
        dual = O.parse_plan_dump(_read(os.path.join(GOLDEN, "plan_27b_dual_n3.txt")))
        self.assertIn("W64", dual["w_codes"])                                  # the dual-only code is in the index

    def test_parse_refusal_header_and_forced_lines(self):
        text = ("# argv_n=7 rc=None exc=Weg2LaunchRefused: W19 Weg2Foo: x\n"
                "[TS] WEG2-LAUNCH FORCED-PAST HW-COUNT 2 cards would be P = TP1\n"
                "[TS] WEG2-LAUNCH PP-CUT SHIPPED: layers=49,15 attn=12,4 chosen_pool=225914 pool_floor=1\n")
        d = O.parse_plan_dump(text)
        self.assertEqual(d["header"]["exc_type"], "Weg2LaunchRefused")
        self.assertEqual(d["forced"], [{"code": "HW-COUNT", "text": "2 cards would be P = TP1"}])
        self.assertEqual(d["pp_cut"]["layers"], "49,15")
        self.assertEqual(d["pp_cut"]["chosen_pool"], "225914")


# ---------------------------------------------------------------------------
# (b) the reference test: dump == golden
# ---------------------------------------------------------------------------

def _dump_of(name: str, devices, **kw):
    return O.run_profile(_p(name), devices, tree=TREE, **kw)


def _assert_zero_diff(tc: unittest.TestCase, golden_name: str, run) -> None:
    res = run.result
    tc.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:300]))
    tc.assertEqual(res.rc, 0)
    d = O.diff_lines(_read(os.path.join(GOLDEN, golden_name)), res.dump())
    tc.assertEqual(d, [], "plan diff vs %s: %d lines\n%s" % (golden_name, len(d), "\n".join(x[:240] for x in d[:12])))


class TestDryRunGolden(unittest.TestCase):
    def setUp(self):
        self.env_before = dict(os.environ)

    def tearDown(self):
        # the oracle must leave NOTHING behind: neither env nor the refusals state
        self.assertEqual(dict(os.environ), self.env_before)
        self.assertFalse(refusals.forced_boot())

    @_NEEDS_27B_FLIP
    def test_27b_flip_dump_equals_golden(self):
        run = _dump_of("27b-base", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)
        self.assertEqual(len(run.notes), 3)           # model farm, draft farm, census-foreign: nothing silent
        self.assertEqual(run.result.forced, [])

    @_NEEDS_27B_FLIP
    def test_27b_flip_golden_does_not_depend_on_the_run(self):
        """Another scratch dir AND the replay synthesised from a real hardware profile: the same plan."""
        rows = O.replay_from_hardware_profile(_hardware_profile_of(O.read_replay(REPLAY_REF)))
        with tempfile.TemporaryDirectory(prefix="ap0-other-scratch-") as td:
            run = _dump_of("27b-base", rows, scratch=td)
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)

    @_NEEDS_27B_DUAL
    def test_27b_dual_dump_equals_golden(self):
        run = _dump_of("27b-nvfp4-dual", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_27b_dual_n3.txt", run)
        self.assertEqual(run.notes, [])               # nothing substituted: the NVFP4 checkpoints are on the box
        # the dual-only form really ran (MPS opt-in env reached the launcher process; two groups on the same cards)
        p = O.parse_plan_dump(run.result.text)
        self.assertTrue(any(k.startswith("WEG2-DUAL-SHARE") for k in p["kinds"]), sorted(p["kinds"]))
        self.assertNotIn("SGLANG_WEG2_DUAL_MPS_OPT_IN", os.environ)

    @unittest.skipUnless(
        _nf_checkpoint_present() and os.path.exists(os.path.join(GOLDEN, "plan_nf_abl_n3.txt")),
        "NF abl checkpoints are empty mount points on this box and plan_nf_abl_n3.txt was never generated "
        "(see the module docstring): the NF dry-run golden needs the box that has them",
    )
    def test_nf_abl_dump_equals_golden(self):
        run = _dump_of("nf-int4-h6-abl", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_nf_abl_n3.txt", run)

    def test_foreign_environment_does_not_leak_into_the_run(self):
        """A SGLANG_* / HTSGLANG_* variable of the surrounding process is not seen by the launcher, the profile's own env is,
        and everything is restored afterwards (two cards: refused fast, which is all this needs)."""
        from unittest import mock

        two = [dict(r, index=i) for i, r in enumerate(d for d in O.read_replay(REPLAY_REF) if d["index"] in (1, 2))]
        seen = {}
        real_main = launcher.main

        def spy(argv):
            seen.update({k: v for k, v in os.environ.items() if k.startswith(("SGLANG_", "HTSGLANG_", "CUDA_"))})
            return real_main(argv)

        with mock.patch.dict(os.environ, {"SGLANG_WEG2_AP0_TEST_JUNK": "1", "HTSGLANG_AP0_JUNK": "1"}):
            before = dict(os.environ)
            with mock.patch.object(launcher, "main", spy):
                O.run_profile(_p("27b-nvfp4-dual"), two, tree=TREE)
            self.assertEqual(dict(os.environ), before)
        self.assertNotIn("SGLANG_WEG2_AP0_TEST_JUNK", seen)
        self.assertNotIn("HTSGLANG_AP0_JUNK", seen)
        self.assertEqual(seen["SGLANG_WEG2_DUAL_MPS_OPT_IN"], "1")          # the profile's _form env arrived
        self.assertEqual(seen["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(seen["SGLANG_NVML_REPLAY_JSON"].endswith("nvml_replay.json"))

    @_NEEDS_27B_FLIP
    def test_force_plumbing_on_two_cards(self):
        """N=2 is refused by name (HW-COUNT); --force goes past it and the oracle reports it, the next refusal is the result."""
        two = [dict(r, index=i) for i, r in enumerate(d for d in O.read_replay(REPLAY_REF) if d["index"] in (1, 2))]
        plain = _dump_of("27b-base", two)
        self.assertEqual(plain.result.exc_type, "Weg2LaunchRefused")
        self.assertIn("HW-COUNT", plain.result.exc_msg)
        self.assertEqual(plain.result.forced, [])
        forced = _dump_of("27b-base", two, force=True)
        self.assertIn("HW-COUNT", [f["code"] for f in forced.result.forced])
        self.assertIn("--force", forced.result.argv)
        self.assertNotIn("--force", plain.result.argv)
        self.assertIsNotNone(forced.result.exc_type)          # a later, non-forcebar or value refusal stands: a RESULT
        self.assertNotIn("HW-COUNT", forced.result.exc_msg.split(":")[0])


if __name__ == "__main__":
    unittest.main()
