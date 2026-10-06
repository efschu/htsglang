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
box-bound like every dry-run golden of this directory (they carry the census/evidence files of this box); the tests SKIP,
with the reason, where those inputs are absent.  The checkpoint headers are NOT box-bound: every golden of a profile whose
checkpoint dir is an empty mount point here (27B Flip model+draft, NF model+draft) is made from the header snapshots of the
RELEASE checkpoints (below), never from a sibling checkpoint -- measured 2026-10-06 the siblings differ (index total_size
30819147232 vs 29548245472 B for the 27B model, draft 3848817896 vs 2172742656 B).  Each golden has a
``<golden>.provenance.json`` (profile sha256, census, checkpoint config/index sha256); ``test_golden_provenance`` ties the
three together, ``test_profile_snapshots_vs_live`` REPORTS (skip with the reason) when the live release profile has moved
under its snapshot.  The 27B Dual golden is of the profile with the P/D-stage block (``--dual-priority dynamic``,
``--dual-share-actuators green,duty``, ``--dual-green-ladder on`` and the ``_form`` GREEN_TABLE/STARVE_* values); the dry run
passes ``refuse_dual_priority`` (rc=0, nothing forced).

NF (abl form, R9): ``launch_nf-int4-h6-abl.json`` (the argv and environment of the abl profile) and the DRY-RUN golden
``plan_nf_abl_n3.txt`` are pinned.  The NF checkpoint dirs (``Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp`` and
the MTP draft ``...albucino-abl-wxp``) are empty mount points on the dev LXC; the real files live on the Proxmox host
(``/spinning/subvol-999-disk-0/spinning/llm_stuff/club-3090/models-cache/``).  What the launcher reads of a checkpoint is its
headers and ``stat`` sizes only, so ``fixtures/planer_1006/checkpoints/<registry name>/`` holds the HEADER SNAPSHOTS
(:func:`propose_oracle.snapshot_checkpoint`: safetensors header bytes, config/index verbatim, sizes; gzip above 64 KiB, ~4 MB
for 164 GB of weights) taken read-only over ``ssh proxmox`` (no weight byte read; sha256 of every header, config and index in
``golden/plan_nf_abl_n3.provenance.json`` and in the snapshot manifests) and :func:`propose_oracle.materialize_checkpoint`
rebuilds exactly those, so the golden runs on ANY box (``TestCheckpointSnapshot`` proves the stub is the checkpoint as far as
the census can tell).  The dry run with the snapshots is a full plan (rc=0, 376 lines, D parks a 1587 MiB solo draft priced
from the draft snapshot's headers at W128) -- no refusal asks for weight bytes.  REGENERATE on a box that has the files
(or after a header change on the host)::

    PYTHONPATH=python python3 -m sglang.srt.weg2.propose_oracle snapshot \\
        --model-dir <models-cache>/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp \\
        --out test/registered/unit/weg2/fixtures/planer_1006/checkpoints/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp
    (the same for ...MTP-INT4-g32-albucino-abl-wxp)
    CUDA_VISIBLE_DEVICES= PYTHONPATH=python python3 -m sglang.srt.weg2.propose_oracle golden \\
        --profile test/registered/unit/weg2/fixtures/planer_1006/profiles/nf-int4-h6-abl.env \\
        --replay test/registered/unit/weg2/fixtures/xchg_launch_replay_0911/nvml_devices_1378.json \\
        --checkpoint-snapshot test/registered/unit/weg2/fixtures/planer_1006/checkpoints/<model snapshot> \\
        --checkpoint-snapshot test/registered/unit/weg2/fixtures/planer_1006/checkpoints/<draft snapshot> \\
        --out test/registered/unit/weg2/fixtures/planer_1006/golden/plan_nf_abl_n3.txt

``test_nf_abl_dump_equals_golden`` never skips: a missing snapshot or golden FAILS it.  The model/draft dir is stood in for
everywhere the argv names it, including inside the quoted ``--extra-p``/``--extra-d`` values
(``--speculative-draft-model-path``).

LIVE-BOX readings: besides the ``plan_diff.py:6-17`` list, W65 names every ``boot_*.D.log`` of the live evidence dir
(834 names on 2026-10-06); that enumeration is masked (``LIVE_BOX_RULES``) and ``test_golden_does_not_move_when_the_
evidence_dir_grows`` proves a new log does not turn the reference red.  The files the PROFILE names (census, PP-CUT stage
model, rank-dump log of the depth axis) are inputs of the golden and are read from the live evidence dir as named.

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
    "27b-nvfp4-dual.env": "1cc8890ccece7f9bd1097c30072ec6d87031bf4a8d5e9afb03646992ab39a24e",
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
_NEEDS_27B_FLIP = unittest.skipUnless(_have(_CENSUS_27B), "27B census not on this box")
#: the RELEASE checkpoints of the 27B Flip profile (empty mount points on the dev box; their header snapshots are committed)
_FLIP_NAMES = ("Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued")
_NEEDS_27B_DUAL = unittest.skipUnless(
    _have(_CENSUS_27B, MC + "Qwen3.8-27B-NVFP4-RadixArk/config.json", MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal/config.json"),
    "27B NVFP4 checkpoints / census not on this box",
)


_NF_NAMES = ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp",
             "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp")
CKPT_SNAPSHOTS = os.path.join(FIX, "checkpoints")


def _snapshots() -> dict:
    """``{registry dir name: header snapshot dir}`` of the snapshots committed under ``fixtures/planer_1006/checkpoints``."""
    out = {}
    if os.path.isdir(CKPT_SNAPSHOTS):
        for n in sorted(os.listdir(CKPT_SNAPSHOTS)):
            if os.path.isfile(os.path.join(CKPT_SNAPSHOTS, n, "manifest.json")):
                out[O.read_snapshot_manifest(os.path.join(CKPT_SNAPSHOTS, n))["name"]] = os.path.join(CKPT_SNAPSHOTS, n)
    return out


def _flip_snapshots_present() -> bool:
    """Both release checkpoints of 27b-base readable: committed header snapshots (the siblings are NOT a stand-in)."""
    snaps = _snapshots()
    return all(n in snaps for n in _FLIP_NAMES)


def _nf_checkpoint_present() -> bool:
    """Both NF checkpoints readable: REAL (target config.json AND a non-empty draft dir; empty mount points on the box
    that produced this file) or a committed header snapshot of each."""
    snaps = _snapshots()
    real = (os.path.isfile(os.path.join(MC + _NF_NAMES[0], "config.json")) and os.path.isdir(MC + _NF_NAMES[1])
            and bool(os.listdir(MC + _NF_NAMES[1])))
    return real or all(n in snaps for n in _NF_NAMES)


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
                "WEG2-DRY-RUN-BOX-STATE at=2026-10-06T08:25:35Z | budget 26064 MiB total 32607 | "
                "1295 Budget-Loesungen in 0.2 s\n"
                "  W65 Weg2MeasuredAnchor: no anchor, heuristic path stands: no boot_*.D.log of this form: "
                "boot_weg2_a_1_0926_1.D.log: different model (/m/a); boot_weg2_b_2_0926_2.D.log: different model (none)")
        masked, counts = O.mask_live_box(line)
        self.assertEqual({k for k, v in counts.items() if v}, {n for n, _ in O.LIVE_BOX_RULES})
        self.assertIn("budget 26064 MiB total 32607", masked)
        self.assertIn("1295 Budget-Loesungen in <solver-wall-time> s", masked)   # the count stays, the clock goes
        self.assertNotIn("in 0.2 s", masked)
        self.assertIn("c_max=5.00 GiB", masked)                    # the ARC limit stays, only the live size goes
        self.assertNotIn("113.33", masked)
        self.assertNotIn("boot_weg2_a_1", masked)                  # the evidence-dir enumeration goes ...
        self.assertIn("no anchor, heuristic path stands: no boot_*.D.log of this form: <w65-d-log-enumeration>", masked)

    def test_w65_mask_hides_the_enumeration_not_the_verdict(self):
        """Another boot logging a D log changes the list, not the plan; an anchor FOUND (other message) is a plan change."""
        head = "  W65 Weg2MeasuredAnchor: no anchor, heuristic path stands: no boot_*.D.log of this form: "
        two = head + "boot_a.D.log: different model (/m/x); boot_b.D.log: different model (none)\n"
        three = head + "boot_new.D.log: different model (/m/y); boot_a.D.log: different model (/m/x); boot_b.D.log: x\n"
        self.assertEqual(O.diff_lines(two, three), [])
        found = "  W65 Weg2MeasuredAnchor: anchored on boot_new.D.log (posts 1,2,3)\n"
        self.assertGreater(len(O.diff_lines(two, found)), 0)
        # the verdict words themselves are NOT masked: "no anchor" -> "anchor" differs
        self.assertGreater(len(O.diff_lines(two, two.replace("no anchor, heuristic path stands", "anchor stands"))), 0)

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


class TestCheckpointSnapshot(unittest.TestCase):
    """``snapshot_checkpoint`` / ``materialize_checkpoint``: the headers-only stand-in for a checkpoint that is not on the box."""

    _REAL = MC + "Qwen3.8-27B-NVFP4-RadixArk"          # 3 shards of ~10 GB, index, big tokenizer files: every file kind

    @unittest.skipUnless(os.path.isfile(_REAL + "/config.json"), "NVFP4 RadixArk checkpoint not on this box")
    def test_the_stub_is_the_checkpoint_as_far_as_the_census_can_tell(self):
        import struct
        from sglang.srt.weg2 import checkpoint_census as CC

        with tempfile.TemporaryDirectory(prefix="ap0-snap-") as td:
            snap = O.snapshot_checkpoint(self._REAL, os.path.join(td, "snap"))
            stub = O.materialize_checkpoint(snap, os.path.join(td, "farm"))
            self.assertEqual(os.path.basename(stub), "Qwen3.8-27B-NVFP4-RadixArk")      # the NAME is the identity
            names = sorted(f for f in os.listdir(self._REAL) if os.path.isfile(os.path.join(self._REAL, f)))
            self.assertEqual(sorted(os.listdir(stub)), names)
            for fn in names:
                real, st = os.path.join(self._REAL, fn), os.path.join(stub, fn)
                self.assertEqual(os.path.getsize(st), os.path.getsize(real), fn)       # stat sizes: equal
                if fn.endswith(".safetensors"):
                    with open(real, "rb") as a, open(st, "rb") as b:
                        n = struct.unpack("<Q", a.read(8))[0]
                        a.seek(0)
                        self.assertEqual(a.read(8 + n), b.read(8 + n), fn)             # header bytes: identical
                    self.assertLess(os.stat(st).st_blocks * 512, 1 << 24, "stub must be sparse: " + fn)
                elif os.path.getsize(real) <= O.SNAPSHOT_COPY_MAX:
                    self.assertEqual(_read(real), _read(st), fn)                       # small files: verbatim
            kw = dict(exclude_prefixes=("mtp.",), exclude_segments=())
            a, b = CC.layer_census_from_headers(self._REAL, **kw), CC.layer_census_from_headers(stub, **kw)
            self.assertEqual((a.layer_bytes, a.unlayered_bytes, a.total_bytes), (b.layer_bytes, b.unlayered_bytes, b.total_bytes))
            self.assertGreater(a.total_bytes, 1 << 30)
            # the snapshot itself is small: headers + small files, not weights
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _d, fs in os.walk(snap) for f in fs)
            self.assertLess(size, 8 << 20)

    @unittest.skipUnless(os.path.isfile(_REAL + "/config.json"), "NVFP4 RadixArk checkpoint not on this box")
    def test_run_profile_prefers_the_checkpoints_own_snapshot_over_a_sibling(self):
        with tempfile.TemporaryDirectory(prefix="ap0-snap-") as td:
            snap = O.snapshot_checkpoint(self._REAL, os.path.join(td, "snap"))
            empty = os.path.join(td, "empty_mount", "Qwen3.8-27B-NVFP4-RadixArk")
            os.makedirs(empty)
            farm = os.path.join(td, "farm")
            got = O.ensure_model_dir(empty, siblings=[MC + "Qwen3.8-27B-INT8-gdncov"], farm_root=farm, snapshot=snap)
            self.assertEqual(got, os.path.join(farm, "Qwen3.8-27B-NVFP4-RadixArk"))
            self.assertFalse(os.path.islink(os.path.join(got, "config.json")))          # a stub, not the sibling's symlink
            self.assertEqual(_read(os.path.join(got, "config.json")), _read(self._REAL + "/config.json"))
            # a REAL dir (config.json present) is used as it is, snapshot or not
            self.assertEqual(O.ensure_model_dir(self._REAL, snapshot=snap, farm_root=farm), self._REAL)
            # a snapshot of ANOTHER checkpoint is refused by name, never used
            with self.assertRaises(ValueError):
                O.ensure_model_dir(os.path.join(td, "empty_mount", "Other-Model"), snapshot=snap, farm_root=farm)

    def test_an_empty_mount_point_cannot_be_snapshotted(self):
        with tempfile.TemporaryDirectory(prefix="ap0-snap-") as td:
            with self.assertRaises(FileNotFoundError):
                O.snapshot_checkpoint(td, os.path.join(td, "out"))

    def test_committed_snapshots_are_well_formed(self):
        """Whatever snapshot is committed rebuilds, and its headers fit their recorded sizes (nothing truncated)."""
        for name, d in _snapshots().items():
            m = O.read_snapshot_manifest(d)
            self.assertEqual(m["name"], name)
            with tempfile.TemporaryDirectory(prefix="ap0-snap-") as td:
                stub = O.materialize_checkpoint(d, td)
                for e in m["files"]:
                    self.assertEqual(os.path.getsize(os.path.join(stub, e["name"])), e["size"])


# ---------------------------------------------------------------------------
# (b) the reference test: dump == golden
# ---------------------------------------------------------------------------

def _dump_of(name: str, devices, **kw):
    kw.setdefault("snapshots", _snapshots())
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
        self.assertTrue(_flip_snapshots_present(), "27B Flip release-checkpoint header snapshots missing under fixtures/planer_1006/checkpoints")
        run = _dump_of("27b-base", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)
        self.assertEqual(len(run.notes), 3)           # model stub, draft stub, census-foreign: nothing silent
        self.assertEqual(sum("header snapshot stub" in n for n in run.notes), 2)    # both from the release headers, NOT a name farm/sibling
        self.assertFalse(any("name farm" in n or "sibling" in n for n in run.notes), run.notes)
        self.assertEqual(run.result.forced, [])
        # the plan is priced on the RELEASE model, not the sibling: the launcher recognises it as the reference model
        self.assertTrue(any("WEG2-HOST-REFERENCE-MODEL" in ln and "APPLY to this boot" in ln for ln in run.result.text.splitlines()))

    @_NEEDS_27B_FLIP
    def test_27b_flip_golden_does_not_depend_on_the_run(self):
        """Another scratch dir AND the replay synthesised from a real hardware profile: the same plan."""
        rows = O.replay_from_hardware_profile(_hardware_profile_of(O.read_replay(REPLAY_REF)))
        with tempfile.TemporaryDirectory(prefix="ap0-other-scratch-") as td:
            run = _dump_of("27b-base", rows, scratch=td)
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)

    @_NEEDS_27B_FLIP
    def test_a_forced_two_card_run_leaves_no_state_for_the_next_plan(self):
        """``launcher.main`` fills process caches (``corridor_guard._RIG_FP_CACHE``) and sets the inventory view / exchange
        geometry.  Measured 2026-10-06: after a ``--force`` run on two cards the next THREE-card run of the same process was
        refused (W40) instead of planned.  The oracle restores the module state: the plan after the forced run is the golden."""
        two = [dict(r, index=i) for i, r in enumerate(d for d in O.read_replay(REPLAY_REF) if d["index"] in (1, 2))]
        forced = _dump_of("27b-base", two, force=True)
        self.assertIsNotNone(forced.result.exc_type)
        from sglang.srt.managers import corridor_guard
        self.assertEqual(len(corridor_guard._RIG_FP_CACHE), 0)      # the cache the forced run filled is back to its entry state
        run = _dump_of("27b-base", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)

    @_NEEDS_27B_FLIP
    def test_golden_does_not_move_when_the_evidence_dir_grows(self):
        """The launcher names every ``boot_*.D.log`` of the evidence dir in W65 (``find_dual_d_measurement``): a boot of the
        rig adds one.  An OVERLAY of the live dir (every entry linked) plus one NEWER, unrelated D log: the plan is the
        golden (0 diff), while the RAW W65 line really did change -- the mask is what holds it, nothing else moved."""
        live = launcher.EVIDENCE_DIR
        if not os.path.isdir(live):
            self.skipTest("no evidence dir on this box")
        name = "boot_weg2_ap0overlayfake_0000000000_1006_000000.D.log"
        with tempfile.TemporaryDirectory(prefix="ap0-evidence-overlay-") as td:
            for n in os.listdir(live):
                os.symlink(os.path.join(live, n), os.path.join(td, n))
            with open(os.path.join(td, name), "w") as fh:
                fh.write("not a boot log: the oracle must only list it\n")
            run = O.run_profile(_p("27b-base"), O.read_replay(REPLAY_REF), tree=TREE, evidence_dir=td)
        self.assertIn(name, run.result.raw)                       # the overlay really was the evidence dir of the run
        self.assertNotIn(td, run.result.text)                     # ... and the text is normalised back to the live path
        _assert_zero_diff(self, "plan_27b_flip_n3.txt", run)

    @_NEEDS_27B_DUAL
    def test_27b_dual_dump_equals_golden(self):
        run = _dump_of("27b-nvfp4-dual", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_27b_dual_n3.txt", run)
        self.assertEqual(run.notes, [])               # nothing substituted: the NVFP4 checkpoints are on the box
        self.assertEqual(run.result.forced, [])       # refuse_dual_priority let the P/D-stage block through
        for flag, val in (("--dual-priority", "dynamic"), ("--dual-share-actuators", "green,duty"), ("--dual-green-ladder", "on")):
            i = run.argv.index(flag)
            self.assertEqual(run.argv[i + 1], val, flag)
        text = run.result.text
        self.assertIn("SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE=1:1:1;2:2:2;1000000000:3:3", text)
        self.assertIn("SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG=1", text)
        self.assertIn("SGLANG_WEG2_DUAL_GRANT_RETRY_MS=20", text)
        # the dual-only form really ran (MPS opt-in env reached the launcher process; two groups on the same cards)
        p = O.parse_plan_dump(run.result.text)
        self.assertTrue(any(k.startswith("WEG2-DUAL-SHARE") for k in p["kinds"]), sorted(p["kinds"]))
        self.assertNotIn("SGLANG_WEG2_DUAL_MPS_OPT_IN", os.environ)

    def test_nf_abl_dump_equals_golden(self):
        self.assertTrue(_nf_checkpoint_present(), "NF abl header snapshots missing under fixtures/planer_1006/checkpoints")
        self.assertTrue(os.path.isfile(os.path.join(GOLDEN, "plan_nf_abl_n3.txt")))
        run = _dump_of("nf-int4-h6-abl", O.read_replay(REPLAY_REF))
        _assert_zero_diff(self, "plan_nf_abl_n3.txt", run)
        self.assertEqual(len(run.notes), 3)           # model stub, draft stub, census-foreign: nothing silent
        self.assertEqual(run.result.forced, [])
        self.assertTrue(any("d_draft_host=1587 MiB" in ln for ln in run.result.text.splitlines()))   # W128 priced, not refused

    _GOLDEN_PROVENANCE = (("plan_27b_flip_n3", "27b-base"), ("plan_27b_dual_n3", "27b-nvfp4-dual"), ("plan_nf_abl_n3", "nf-int4-h6-abl"))

    def test_golden_provenance(self):
        """Each golden names the profile and the checkpoint files it was made from, by sha256 (provenance json, committed
        profile snapshot and the ``PROVENANCE`` table agree; header snapshots reproduce the recorded config/index)."""
        import hashlib

        def sha(path):
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()

        snaps = _snapshots()
        for golden, prof in self._GOLDEN_PROVENANCE:
            side = json.loads(_read(os.path.join(GOLDEN, golden + ".provenance.json")))
            self.assertEqual(side["golden"], golden + ".txt")
            self.assertTrue(os.path.isfile(os.path.join(GOLDEN, side["golden"])), golden)
            self.assertEqual(side["profile"]["sha256"], PROVENANCE[prof + ".env"], golden)
            self.assertEqual(sha(_p(prof)), side["profile"]["sha256"], golden)
            for inp in side.get("profile_inputs", {}).values():
                if not inp["file"].startswith("/"):                         # committed snapshot of a file
                    self.assertEqual(sha(os.path.join(FIX, inp["file"])), inp["sha256"], golden)
            for name, rec in side.get("checkpoints", {}).items():
                self.assertIn(name, snaps)
                with tempfile.TemporaryDirectory(prefix="ap0-prov-") as td:
                    stub = O.materialize_checkpoint(snaps[name], td)
                    for fn, want in rec["files_sha256_on_host"].items():
                        self.assertEqual(sha(os.path.join(stub, fn)), want, "%s/%s" % (name, fn))

    def test_flip_golden_is_not_made_from_the_sibling_checkpoints(self):
        """The release model/draft are the snapshots, and they are NOT the sibling checkpoints of the dev box (sizes measured
        2026-10-06 on the Proxmox host): index total_size 29548245472 B (release) vs 30819147232 B (gdncov sibling)."""
        snaps = _snapshots()
        for n in _FLIP_NAMES:
            self.assertIn(n, snaps)
        with tempfile.TemporaryDirectory(prefix="ap0-flipsnap-") as td:
            stub = O.materialize_checkpoint(snaps[_FLIP_NAMES[0]], td)
            idx = json.loads(_read(os.path.join(stub, "model.safetensors.index.json")))
            self.assertEqual(idx["metadata"]["total_size"], 29548245472)
            dstub = O.materialize_checkpoint(snaps[_FLIP_NAMES[1]], td)
            self.assertEqual(os.path.getsize(os.path.join(dstub, "model.safetensors")), 2172742656)

    def test_profile_snapshots_vs_live(self):
        """REPORT, never silently follow: when the live release profile has moved under its committed snapshot the golden is
        of an old form (AP0 round 2 found the Dual snapshot 3 additions behind).  Skips with the names, passes when equal."""
        import hashlib

        live_dir = {"27b-base": "/spinning/gpu-arb/docker/profiles_release", "27b-nvfp4-dual": "/spinning/gpu-arb/docker/profiles_release",
                    "nf-int4-h6-abl": "/spinning/gpu-arb/docker/profiles"}
        moved = []
        for prof, d in live_dir.items():
            live = os.path.join(d, prof + ".env")
            if os.path.isfile(live):
                with open(live, "rb") as fh:
                    if hashlib.sha256(fh.read()).hexdigest() != PROVENANCE[prof + ".env"]:
                        moved.append(live)
        if moved:
            self.skipTest("LIVE PROFILE MOVED under its snapshot (re-snapshot, regenerate golden + launch json, update PROVENANCE): "
                          + ", ".join(moved))

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
