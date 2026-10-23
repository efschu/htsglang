"""planer-n2fix 1023: the start profile of the planner page carries what the page shows, for every card count and every card mix.

The live planner (``/opt/rigdash/current`` = 85a95da48b) answered "27B NVFP4, Flip, RTX 5090 + RTX 3080 20 GB" (deskq/done/27b-nvfp4-flip-planer-antwort-1023.json)
with three findings; each one is pinned here on REAL launcher dry runs (the oracle child of this checkout, ``ZielPlaner.plan`` as the page calls it):

* B1  ``startprofil.vars`` kept the 3-card release profile (PROFILE_CARD_COUNT=3, PROFILE_GPUS "5090=1;3080=2", PROFILE_INVENTORY of 3 cards): only args and meta
      were converted.  Cause: ``ProfilEditor._startprofil`` applied the proposal's argv values only; the card variables are no argv value and nothing derived
      them.  Now ``ProfilEditor.card_vars`` derives all three from the proposal's card list (launcher order) for every N and card mix.
* B2  the fields named ``--rank-tp-ratio`` 58,25 and per-phase ``--rank-gpu-memory-mib``, the start profile carried neither.  Cause: those are LAUNCHER outputs of
      the dry run (``verdict.plan.group_flags``), and the start profile was built from the stage-A proposal only.  Now all three are RECORDED in the start
      profile (``meta.planner``) and never pinned: a word in ``--extra-*`` would beat the launcher's own derivation at boot (memory argv-doppelung), and W100
      refuses any pinned budget above the live one -- user rule ``planner-alleinzustaendigkeit-vram``.  The fields say so.
* B3  ``--hicache-mamba-host-mib`` 100: the launcher constant (``launcher.common_flags``, #1430), not derived; on the 27B it buys 2 slots per rank against the
      runtime floor of max running + 1 (every 27B boot logs ``MAMBA-ANCHOR-FLOOR RAISED``).  Now stage A (``propose._mamba_host_section``) names
      ceil((max running + 1) x one mamba slot of the whole model) per group when the launcher's value buys fewer (27B: P 225 / D 524 MiB).

Cases: 5090 + 3080 20 GB (n=2, mixed, the hardware profile of the rig with its measured card library), 2 / 3 / 7 x RTX 3090 (catalog cards).  For each: the vars
match the card list, every positional value of the start profile has n entries, the fields equal the start profile's ``meta.planner`` (recorded), and a SECOND
launcher dry run of the start profile itself ships the same layout (weight split, mamba pools) and has no entrypoint HW-COUNT.  B4 (image paths), B7 (a subset of
the rig on its real cards; a git-archive stage resolves the record identity) and the BAR1 rows are pinned below as well.
"""

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan as K  # noqa: E402
from rigdash import planer_ziel as Z  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import profile_oracle as ORA  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO_PY = os.path.join(REPO_ROOT, "python")
sys.path.insert(0, REPO_PY)
RELEASE_DIR = os.path.join(REPO_ROOT, "docker", "flliper", "profiles_release")
CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
FIX = os.path.join(HERE, "fixtures", "planer_n2fix_1023")
#: the rig's card library as measured 14.08. (RTX 5090 and RTX 3080 20GB rows "measured", the rest "seed"): the D objective decode-bs1 of the 27B profile
#: prices its weight vector from it (W61 without it); an INPUT of the test, never ~/.cache
CARD_LIBRARY = os.path.join(FIX, "card_library_rig_0814.json")
REPLAY = os.path.join(REPO_ROOT, "test", "registered", "unit", "pdflip", "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
NVFP4 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"
HAS_MODEL = os.path.exists(os.path.join(NVFP4, "config.json"))
HAS_LAUNCHER = os.path.isfile(os.path.join(REPO_PY, "flliper", "srt", "pdflip", "launcher.py"))
IMAGE_PCHUNK = "/opt/htsglang/profiles/27b-nvfp4.pchunk.json"

MIX = "mix"
RIG3 = "rig3"
CASES = {MIX: [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 1}],
         RIG3: [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}],
         2: [{"card": "rtx3090-24", "count": 2}], 3: [{"card": "rtx3090-24", "count": 3}], 7: [{"card": "rtx3090-24", "count": 7}]}
#: the rig's hardware profile of 23.10. (GET /api/hwprofil): NVML 0 = RTX 5090 (ReBAR, BAR1 32768 MiB), 1 / 2 = RTX 3080 20 GB (BAR1 256 MiB; NVML 2
#: carries a display seat, BAR1 free 189); bar1_free_mib from nvidia-smi the same morning.  MIX picks 2 of its 3 cards (B7: real UUIDs, not catalog cards)
RIG_HW = os.path.join(FIX, "hwprofil_rig_1023.json")


def _rig_hw():
    with open(RIG_HW, encoding="utf-8") as fh:
        return json.load(fh)


def _vars(doc):
    return {v["name"]: str(v.get("value", "")) for v in doc.get("vars") or [] if "value" in v}


def _extra(doc, g):
    """``{flag: last value}`` of the ``--extra-g`` string of a start profile document."""
    out = {}
    for a in doc.get("args") or []:
        if a.get("flag") == "--extra-%s" % g.lower():
            for m in re.finditer(r"(?<![\w-])(--[\w-]+)(?:\s+|=)(\S+)", " ".join(a.get("values") or [])):
                out[m.group(1)] = m.group(2)
    return out


def _flag(doc, name):
    for a in doc.get("args") or []:
        if a.get("flag") == name:
            return " ".join(a.get("values") or [])
    return None


def _fields(out):
    return {z["key"]: z for g in out["felder"] for z in g["zeilen"]}


# ------------------------------------------------------------------------------------------------------------------------------- pure
class CardVars(unittest.TestCase):
    def test_mixed_list_counts_each_name_in_order_of_first_appearance(self):
        cards = [{"name": "NVIDIA GeForce RTX 5090", "class": "RTX5090"}, {"name": "NVIDIA GeForce RTX 3080", "class": "RTX3080"}]
        self.assertEqual(P.ProfilEditor.card_vars(cards), {"PROFILE_CARD_COUNT": "2", "PROFILE_GPUS": "NVIDIA GeForce RTX 5090=1;NVIDIA GeForce RTX 3080=1",
                                                           "PROFILE_INVENTORY": "RTX5090,RTX3080"})

    def test_identical_cards_and_the_rig_format(self):
        rig = [{"name": "NVIDIA GeForce RTX 5090", "class": "RTX5090"}] + [{"name": "NVIDIA GeForce RTX 3080", "class": "RTX3080"}] * 2
        self.assertEqual(P.ProfilEditor.card_vars(rig), {"PROFILE_CARD_COUNT": "3", "PROFILE_GPUS": "NVIDIA GeForce RTX 5090=1;NVIDIA GeForce RTX 3080=2",
                                                         "PROFILE_INVENTORY": "RTX5090,RTX3080,RTX3080"})          # = the release profile 27b-nvfp4.env
        seven = [{"name": "NVIDIA GeForce RTX 3090", "class": "RTX3090/24576MiB/sm86"}] * 7
        v = P.ProfilEditor.card_vars(seven)
        self.assertEqual((v["PROFILE_CARD_COUNT"], v["PROFILE_GPUS"]), ("7", "NVIDIA GeForce RTX 3090=7"))
        self.assertEqual(v["PROFILE_INVENTORY"].split(","), ["RTX3090/24576MiB/sm86"] * 7)


@unittest.skipUnless(HAS_LAUNCHER, "no pdflip launcher in this tree")
class MambaRule(unittest.TestCase):
    def test_the_staging_law_is_the_runtimes(self):
        from flliper.srt.pdflip import propose_rules as R
        try:
            from flliper.srt.mem_cache.memory_pool_host import mamba_anchor_slot_floor_staging as law
        except Exception as exc:  # pragma: no cover - torch-less build
            self.skipTest("memory_pool_host not importable: %r" % exc)
        for m in (0, 1, 2, 6, 8, 32):
            self.assertEqual(R.mamba_staging_slots(m), law(m), m)

    def test_the_launcher_default_is_the_launcher_constant(self):
        from flliper.srt.pdflip import propose_rules as R
        with open(os.path.join(REPO_PY, "flliper", "srt", "pdflip", "launcher.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('"--hicache-mamba-host-mib", str(int(os.environ.get("FLLIPER_PDFLIP_MAMBA_ANCHOR_MIB", "%d")' % R.LAUNCHER_MAMBA_ANCHOR_MIB, src)

    def test_mib_buys_the_floor_at_the_largest_slot(self):
        from flliper.srt.pdflip import propose_rules as R
        self.assertEqual(R.mamba_host_mib(6, 74.85), 524)            # 7 slots x 74.85 MiB (27B: one slot of the whole model), rounded up
        self.assertEqual(R.mamba_host_mib(2, 74.85), 225)
        self.assertEqual((R.mamba_host_mib(0, 74.85), R.mamba_host_mib(6, 0.0)), (0, 0))
        # the rebuke's evidence: the 27B boot 1022_015348 measured 37.41 MiB per slot on D-TP0 and 48.32 on P-PP0 -- both <= one slot of the whole model,
        # so the named value buys at least the floor on every rank (7 and 3)
        self.assertGreaterEqual(524 // 37.41, 7)
        self.assertGreaterEqual(225 // 48.32, 3)


class PlanValues(unittest.TestCase):
    """``ProfilEditor._plan_values`` on a canned verdict: what is pinned, what is only recorded."""

    VD = {"plan": {"group_flags": {"P": {"--rank-gpu-memory-mib": "26064,15336"}, "D": {"--rank-gpu-memory-mib": "27784,17528", "--rank-tp-ratio": "58,25"}}}}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="n2fix_")
        self.ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=RELEASE_DIR, user_dir=os.path.join(self.tmp, "usr"), tree=REPO_PY,
                                 catalog_file=CATALOG)
        self.base = self.ed._import_release("27b-nvfp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _prop(self, same):
        cards = [{"name": "NVIDIA GeForce RTX 5090", "class": "RTX5090"}, {"name": "NVIDIA GeForce RTX 3080", "class": "RTX3080"}]
        return {"cards": cards if not same else cards + [cards[1]], "inventory": {"same_as_profile": same}}

    def test_foreign_cards_get_their_card_vars_and_record_the_launcher_values(self):
        pins, records, notes = self.ed._plan_values(self.base, self._prop(False), self.VD, "flip")
        self.assertEqual(pins, {"var:PROFILE_CARD_COUNT": "2", "var:PROFILE_GPUS": "NVIDIA GeForce RTX 5090=1;NVIDIA GeForce RTX 3080=1",
                                "var:PROFILE_INVENTORY": "RTX5090,RTX3080"})
        # B2 (n2fix 1023 review of the first draft): the split is RECORDED, never pinned -- a word in --extra-d beats the launcher's own derivation
        self.assertEqual(records, {"extra:D:--rank-tp-ratio": "58,25", "extra:P:--rank-gpu-memory-mib": "26064,15336",
                                   "extra:D:--rank-gpu-memory-mib": "27784,17528"})
        self.assertTrue(any("W100" in n and "D --rank-tp-ratio 58,25" in n for n in notes), notes)

    def test_the_profiles_own_cards_are_the_profile(self):
        pins, records, _ = self.ed._plan_values(self.base, self._prop(True), self.VD, "flip")
        self.assertEqual(pins, {})                                   # A1: vars equal, the profile's own objective yields the split
        self.assertEqual(set(records), {"extra:D:--rank-tp-ratio", "extra:P:--rank-gpu-memory-mib", "extra:D:--rank-gpu-memory-mib"})

    def test_single_card_has_no_pdflip_values(self):
        self.assertEqual(self.ed._plan_values(None, self._prop(False), self.VD, "single"), ({}, {}, []))


class HwPick(unittest.TestCase):
    """B7: the chosen cards may be a SUBSET of the rig's NVML cards; they then run with their real identity (UUID, BAR1, PCIe)."""

    def _entries(self, *names):
        cat = {"5090": ("NVIDIA GeForce RTX 5090", 32607), "3080": ("NVIDIA GeForce RTX 3080", 20480), "3090": ("NVIDIA GeForce RTX 3090", 24576)}
        return [{"entry": {"nvml_name": cat[n][0], "usable_mib": cat[n][1]}} for n in names]

    def test_two_of_three_pick_the_lowest_nvml_index_and_renumber(self):
        hw = _rig_hw()
        got = P.ProfilEditor._hw_cards_pick(hw, self._entries("5090", "3080"))
        self.assertEqual([c["uuid"] for c in got["cards"]], [hw["cards"][0]["uuid"], hw["cards"][1]["uuid"]])
        self.assertEqual([c["nvml_index"] for c in got["cards"]], [0, 1])
        self.assertEqual(got["subset_of"], {"cards": 3, "nvml_index": [0, 1]})
        self.assertEqual(got["cards"][0]["bar1_total_mib"]["v"], 32768)          # the real BAR1, not the datasheet's 256
        self.assertEqual(hw["cards"][1]["nvml_index"], 1)                         # the input profile is not modified

    def test_the_two_3080_are_renumbered_like_a_container_sees_them(self):
        hw = _rig_hw()
        got = P.ProfilEditor._hw_cards_pick(hw, self._entries("3080", "3080"))
        self.assertEqual([c["uuid"] for c in got["cards"]], [hw["cards"][1]["uuid"], hw["cards"][2]["uuid"]])
        self.assertEqual([c["nvml_index"] for c in got["cards"]], [0, 1])
        self.assertEqual(got["subset_of"]["nvml_index"], [1, 2])

    def test_all_cards_is_the_profile_and_a_foreign_card_is_none(self):
        hw = _rig_hw()
        got = P.ProfilEditor._hw_cards_pick(hw, self._entries("3080", "5090", "3080"))
        self.assertNotIn("subset_of", got)
        self.assertEqual(len(got["cards"]), 3)
        self.assertIsNone(P.ProfilEditor._hw_cards_pick(hw, self._entries("5090", "3090")))
        self.assertIsNone(P.ProfilEditor._hw_cards_pick(hw, self._entries("5090", "5090")))           # more of a name than the rig has
        self.assertIsNone(P.ProfilEditor._hw_cards_pick({"cards": []}, self._entries("5090")))
        self.assertTrue(P.ProfilEditor._hw_cards_match(hw, self._entries("5090")))
        self.assertFalse(P.ProfilEditor._hw_cards_match(hw, self._entries("3090")))

    def test_the_pair_and_index_parts_follow_the_renumbering(self):
        hw = _rig_hw()
        got = P.ProfilEditor._hw_cards_pick(hw, self._entries("3080", "3080"))
        pairs = {(lk["src"], lk["dst"]) for lk in got["links"]}
        self.assertEqual(pairs, {(0, 1), (1, 0)})                                  # old 1->2 / 2->1, nothing that touched the 5090
        self.assertEqual(len(got["links"]), len([lk for lk in hw["links"] if {lk["src"], lk["dst"]} == {1, 2}]))
        self.assertEqual(sorted(got["unmeasured"]), ["0", "1"])
        self.assertNotIn("inventory_sig", got)
        self.assertIn("inventory_sig", hw)                                          # the input is untouched

    def test_the_replay_takes_the_picked_cards(self):
        from flliper.srt.pdflip import propose_oracle as O
        got = P.ProfilEditor._hw_cards_pick(_rig_hw(), self._entries("5090", "3080"))
        rows = O.replay_from_hardware_profile(got)
        self.assertEqual([r["index"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["bar1_total_bytes"] >> 20, 32768)


@unittest.skipUnless(HAS_LAUNCHER, "no pdflip launcher in this tree")
class StageLineIdentity(unittest.TestCase):
    """B7: a git-archive stage (REV + SOURCE_REPO, rigdash stage_profil_modules.sh) resolves the launcher's ``line`` term."""

    def setUp(self):
        import subprocess
        self.git_dir = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True,
                                      text=True).stdout.strip()
        self.head = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if not self.git_dir or not self.head:
            self.skipTest("no git checkout")
        self.tmp = tempfile.mkdtemp(prefix="n2fix_stage_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage(self, rev=None, src=None):
        if rev is not None:
            with open(os.path.join(self.tmp, "REV"), "w") as fh:
                fh.write(rev + "\n")
        if src is not None:
            with open(os.path.join(self.tmp, "SOURCE_REPO"), "w") as fh:
                fh.write(src + "\n")

    def test_stage_with_rev_and_source_reads_head_and_ancestry(self):
        from flliper.srt.pdflip import form as F
        self._stage(self.head[:10], self.git_dir)
        self.assertEqual(F._repo_head(self.tmp), self.head)
        self.assertTrue(F.is_line_ancestor(self.tmp, self.head[:10]))
        self.assertTrue(F.is_line_ancestor(self.tmp, "85a95da48b"))             # the A2 deploy is an ancestor of this branch
        self.assertFalse(F.is_line_ancestor(self.tmp, "f6de5974db"))            # a 27B-line commit is not

    def test_the_stamp_wins_over_an_enclosing_work_tree(self):
        from flliper.srt.pdflip import form as F
        inner = tempfile.mkdtemp(prefix="n2fix_stage_in_repo_", dir=os.path.join(REPO_ROOT, "test"))     # a stage dir INSIDE a git work tree
        try:
            with open(os.path.join(inner, "REV"), "w") as fh:
                fh.write("85a95da48b\n")
            with open(os.path.join(inner, "SOURCE_REPO"), "w") as fh:
                fh.write(self.git_dir + "\n")
            self.assertTrue(F._repo_head(inner).startswith("85a95da48b"))               # not the enclosing checkout's HEAD
        finally:
            shutil.rmtree(inner, ignore_errors=True)

    def test_a_stage_without_its_source_stays_unresolvable_by_name(self):
        from flliper.srt.pdflip import form as F
        self._stage(self.head[:10], None)
        self.assertIsNone(F._repo_head(self.tmp))
        ident = F.CalibrationIdentity(model="m", evidence_dir=self.tmp, form=None, fields=("checkpoint", "line"), repo=self.tmp, line_heads=(),
                                      power_limits=())
        line = ident.head_unresolvable_line()
        self.assertIn("RECORD-IDENTITY UNRESOLVABLE", line)
        self.assertIn("REV + SOURCE_REPO", line)

    def test_the_stage_script_writes_the_stamp_and_the_launcher_reads_it(self):
        """Runs the real stage script into a scratch root (git archive of HEAD, ~80 MiB): REV + SOURCE_REPO (the COMMON git dir) are written,
        and the stage's own form module resolves the line from them."""
        import subprocess
        r = subprocess.run(["bash", os.path.join(HERE, "..", "deploy", "stage_profil_modules.sh"), "--apply", "--root", self.tmp, "HEAD"],
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        cur = os.path.join(self.tmp, "profil", "current")
        with open(os.path.join(cur, "SOURCE_REPO")) as fh:
            self.assertEqual(fh.read().strip(), self.git_dir)
        with open(os.path.join(cur, "REV")) as fh:
            rev = fh.read().strip()
        self.assertTrue(self.head.startswith(rev))
        self.assertFalse(os.path.exists(os.path.join(cur, ".git")))                  # an archive, no checkout
        from flliper.srt.pdflip import form as F
        self.assertEqual(F._repo_head(os.path.realpath(cur)), self.head)


class ImagePaths(unittest.TestCase):
    """B4: a data file named next to the profile gets the image path (/opt/htsglang/profiles/), whatever directory it was sourced from."""

    def test_relocator_maps_logical_and_resolved_directory(self):
        from flliper.srt.pdflip import profile_json as PJ
        tmp = tempfile.mkdtemp(prefix="n2fix_reloc_")
        try:
            real = os.path.join(tmp, "real")
            os.makedirs(real)
            os.symlink(real, os.path.join(tmp, "current"))
            env = os.path.join(tmp, "current", "x.env")
            fn = PJ._image_relocator(env, PJ.IMAGE_PROFILES_DIR)
            self.assertEqual(fn(os.path.join(tmp, "current", "a.json")), "/opt/htsglang/profiles/a.json")
            self.assertEqual(fn(os.path.join(real, "a.json")), "/opt/htsglang/profiles/a.json")
            self.assertEqual(fn("--p-chunk-model=" + os.path.join(real, "a.json")), "--p-chunk-model=/opt/htsglang/profiles/a.json")
            self.assertEqual(fn("/spinning/llm_stuff/x/config.json"), "/spinning/llm_stuff/x/config.json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_relocator_is_anchored(self):
        from flliper.srt.pdflip import profile_json as PJ
        fn = PJ._image_relocator("/a/rel/x.env", PJ.IMAGE_PROFILES_DIR)
        self.assertEqual(fn("/a/rel/f.json"), "/opt/htsglang/profiles/f.json")
        self.assertEqual(fn("--m=/a/rel/f.json,/a/rel/g.json"), "--m=/opt/htsglang/profiles/f.json,/opt/htsglang/profiles/g.json")
        self.assertEqual(fn("/b/a/rel/f.json"), "/b/a/rel/f.json")                   # the directory inside another path is not touched
        self.assertEqual(fn("/a/rel2/f.json"), "/a/rel2/f.json")

    @unittest.skipUnless(HAS_LAUNCHER, "no pdflip launcher in this tree")
    def test_the_oracle_resolves_the_image_path_in_its_own_tree(self):
        from flliper.srt.pdflip import propose_oracle as O
        self.assertEqual(O.tree_release_dir(), RELEASE_DIR)
        tmp = tempfile.mkdtemp(prefix="n2fix_li_")
        try:
            env = os.path.join(tmp, "x.env")
            with open(env, "w") as fh:
                fh.write('PROFILE_NAME=x\nPROFILE_ARGS=(--p-chunk-model %s)\n' % IMAGE_PCHUNK)
            li = O.profile_launch_input(env)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(li.argv, ["--p-chunk-model", os.path.join(RELEASE_DIR, "27b-nvfp4.pchunk.json")])
        self.assertEqual(li.unresolved_paths, [])

    def test_the_release_import_names_the_image_path(self):
        tmp = tempfile.mkdtemp(prefix="n2fix_imp_")
        try:
            ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=RELEASE_DIR, user_dir=os.path.join(tmp, "usr"), tree=REPO_PY,
                                catalog_file=CATALOG)
            doc = ed._import_release("27b-nvfp4")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(_vars(doc)["PROFILE_P_CHUNK_MODEL_NVFP4"], IMAGE_PCHUNK)
        self.assertEqual(_flag(doc, "--p-chunk-model"), IMAGE_PCHUNK)
        self.assertNotIn(RELEASE_DIR, json.dumps(doc))


# ------------------------------------------------------------------------------------------------------------------------------- real dry runs
@unittest.skipUnless(HAS_LAUNCHER and HAS_MODEL, "needs this tree's launcher and the 27B NVFP4 checkpoint headers on this box")
class RealPlans(unittest.TestCase):
    """``ZielPlaner.plan`` with the REAL oracle child (one launcher dry run per case, serial), then a second dry run of each start profile."""

    @classmethod
    def setUpClass(cls):
        cls._env = os.environ.get("FLLIPER_CARD_LIBRARY")
        os.environ["FLLIPER_CARD_LIBRARY"] = CARD_LIBRARY             # the child inherits it
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        cls.tmp = tempfile.mkdtemp(prefix="n2fix_real_")
        cls.oracle = ORA.OracleService(REPO_PY, python=sys.executable)
        cls.out, cls.dry, cls.ed = {}, {}, {}
        for case, cards in CASES.items():
            hw = _rig_hw() if case in (MIX, RIG3) else None
            ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=RELEASE_DIR, user_dir=os.path.join(cls.tmp, "usr%s" % case), tree=REPO_PY,
                                catalog_file=CATALOG, oracle=cls.oracle, hardware=(lambda h=hw: {"ok": True, "profile": h}) if hw else (lambda: {"ok": False}))
            ed.check_path = lambda p, fld: p
            zp = Z.ZielPlaner(ed)
            out = zp.plan({"model": "qwen38-27b-nvfp4", "layout": "flip", "cards": cards})
            cls.out[case], cls.ed[case] = out, ed
            if out.get("status") == "ok":
                req = [{"card": e["id"], "pcie": {"gen": e["pcie_native"]["gen"], "lanes": e["pcie_native"]["lanes"]}} for e in zp._cards(cards)]
                cls.dry[case] = ed.dry_run(copy.deepcopy(out["intern"]["startprofil"]["doc"]), req)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        if cls._env is None:
            os.environ.pop("FLLIPER_CARD_LIBRARY", None)
        else:
            os.environ["FLLIPER_CARD_LIBRARY"] = cls._env

    def _case(self, case):
        out = self.out[case]
        self.assertEqual(out.get("status"), "ok", out.get("hinweise"))
        return out, out["intern"]["startprofil"]["doc"], _fields(out)

    def test_b1_vars_come_from_the_card_list(self):
        for case in CASES:
            with self.subTest(case=case):
                out, doc, f = self._case(case)
                n = out["n"]
                v = _vars(doc)
                self.assertEqual(v["PROFILE_CARD_COUNT"], str(n))
                gpus = dict(x.split("=") for x in v["PROFILE_GPUS"].split(";"))
                self.assertEqual(sum(int(c) for c in gpus.values()), n)
                want = ({"NVIDIA GeForce RTX 5090": "1", "NVIDIA GeForce RTX 3080": "1"} if case == MIX else
                        {"NVIDIA GeForce RTX 5090": "1", "NVIDIA GeForce RTX 3080": "2"} if case == RIG3 else {"NVIDIA GeForce RTX 3090": str(n)})
                self.assertEqual(gpus, want)
                inv = v["PROFILE_INVENTORY"].split(",")
                self.assertEqual(len(inv), n)
                self.assertEqual(inv[0], "RTX5090" if case in (MIX, RIG3) else "RTX3090/24576MiB/sm86")
                # the entrypoint's count gate on the start profile itself: no HW-COUNT from PROFILE_CARD_COUNT (the old start profile said 3 for 2 cards)
                self.assertFalse([r for r in self.dry[case]["rejections"] if r["code"] == "HW-COUNT" and "PROFILE_CARD_COUNT" in str(r.get("source"))],
                                 self.dry[case]["rejections"])

    def test_b2_the_split_and_the_budgets_are_recorded_not_pinned(self):
        for case in CASES:
            with self.subTest(case=case):
                out, doc, f = self._case(case)
                n = out["n"]
                shown = f["d_split"]["je_karte"] or [f["d_split"]["wert"]]
                self.assertNotIn("--rank-tp-ratio", _extra(doc, "D"))          # never a pin that beats the launcher's derivation
                self.assertEqual(doc["meta"]["planner"]["extra:D:--rank-tp-ratio"], ",".join(shown))
                self.assertIn("does not pin it", f["d_split"]["warum"])
                if case == MIX:
                    self.assertEqual(shown, ["58", "25"])                 # decode-bs1 = the measured rate ratio 5090 : 3080 (as the live page showed)
                planner = doc["meta"]["planner"]
                for key, g in (("p_mem", "P"), ("d_mem", "D")):
                    self.assertEqual(planner["extra:%s:--rank-gpu-memory-mib" % g], ",".join(f[key]["je_karte"]))
                    self.assertEqual(len(f[key]["je_karte"]), n)
                    self.assertNotIn("--rank-gpu-memory-mib", _extra(doc, g))      # recorded, never pinned (W100 vs the live budget)
                    self.assertIn("does not pin it", f[key]["warum"])
                # the second dry run of the start profile (no pin) derives the same split again
                gf = self.dry[case]["oracle"]["plan"]["group_flags"]
                self.assertEqual(gf["D"]["--rank-tp-ratio"], ",".join(shown))

    def test_b3_mamba_host_is_derived_from_the_floor(self):
        for case in CASES:
            with self.subTest(case=case):
                out, doc, f = self._case(case)
                p, d = _extra(doc, "P").get("--hicache-mamba-host-mib"), _extra(doc, "D").get("--hicache-mamba-host-mib")
                if case == RIG3:                                          # the profile's own inventory: A1 carries the profile (open point, see propose)
                    self.assertEqual((p, d), (None, None))
                    continue
                self.assertEqual((p, d), ("225", "524"))                 # 27B: (2 + 1) / (6 + 1) slots x 74.85 MiB, both above the launcher's 100
                self.assertEqual(f["mamba_host"]["wert"], "P 225 MiB, D 524 MiB")
                gf = self.dry[case]["oracle"]["plan"]["group_flags"]
                self.assertEqual((gf["P"]["--hicache-mamba-host-mib"], gf["D"]["--hicache-mamba-host-mib"]), ("225", "524"))
                vals = {w["label"]: w for w in self.ed[case].propose({"basis": {"kind": "release", "name": "27b-nvfp4"}, "form": "flip",
                                                                      "inventory": CASES_INV[case](self.ed[case])})["values"]}
                self.assertEqual(vals["--extra-d --hicache-mamba-host-mib"]["value"], "524")
                self.assertIn("max running", vals["--extra-d --hicache-mamba-host-mib"]["source"])

    def test_every_positional_value_has_n_entries(self):
        for case in CASES:
            with self.subTest(case=case):
                out, doc, f = self._case(case)
                n = out["n"]
                for flag in ("--user-reserve-mib", "--pp-stage-ratio", "--pp-attn-stage-ratio"):
                    val = _flag(doc, flag)
                    if val:
                        self.assertEqual(len(val.split(",")), n, (flag, val))
                tp = _extra(doc, "D").get("--rank-tp-ratio")
                if tp and tp != "auto":
                    self.assertEqual(len(tp.split(",")), n)
                self.assertEqual(len(f["p_layers"]["je_karte"]), n)


    def test_b4_data_files_next_to_the_profile_carry_their_image_path(self):
        for case in CASES:
            with self.subTest(case=case):
                out, doc, f = self._case(case)
                v = _vars(doc)
                self.assertEqual(v["PROFILE_P_CHUNK_MODEL_NVFP4"], IMAGE_PCHUNK)
                req = next(x for x in doc["vars"] if x["name"] == "PROFILE_REQUIRED_PATHS")["values"]
                self.assertIn(IMAGE_PCHUNK, req)
                self.assertEqual(_flag(doc, "--p-chunk-model"), IMAGE_PCHUNK)
                self.assertNotIn("/opt/rigdash", json.dumps(doc))             # no host path of the staged planner tree in a start profile
                self.assertNotIn(RELEASE_DIR, json.dumps(doc))
                # the dry run of the start profile itself resolves the image path (oracle asset dirs) and runs
                self.assertIn(self.dry[case]["oracle"]["verdict"]["outcome"] if "verdict" in self.dry[case]["oracle"] else self.dry[case]["oracle"].get("outcome"),
                              ("geht", "ok", "ok_with_force"), self.dry[case]["oracle"].get("notes"))

    def test_b7_a_subset_of_the_rig_runs_on_its_real_cards(self):
        out, doc, f = self._case(MIX)
        self.assertEqual(out["hw"]["quelle"], "gemessen")
        self.assertTrue(any("real UUIDs" in h and "2 of the rig's 3 cards (NVML 0,1)" in h for h in out["hinweise"]), out["hinweise"])
        self.assertFalse(any("Synthetic cards" in h for h in out["hinweise"]), out["hinweise"])
        # the budgets are priced off the measured dormant residue of these cards, not the xsn14 constant of a synthetic card (live page 23.10.:
        # D 26216,15696 / P 25304,14488 against the boot's 28016,17176 / 26288,15456)
        d = [int(x) for x in f["d_mem"]["je_karte"]]
        p = [int(x) for x in f["p_mem"]["je_karte"]]
        self.assertGreater(d[0], 26216 + 1000, d)
        self.assertGreater(d[1], 15696 + 1000, d)
        self.assertGreater(p[0], 25304 + 500, p)
        self.assertGreater(p[1], 14488 + 500, p)

    def test_bar1_per_card_and_the_windows_the_launcher_derives(self):
        out, doc, f = self._case(MIX)
        self.assertEqual(f["bar1"]["je_karte"], ["32768 (free 32751)", "256 (free 255)"])
        self.assertEqual(f["bar1_windows"]["wert"], "P 24,PP_0=96; D 16,TP_0=32,DCP_0=40")      # the 3080 decides: 208 + 4 x 3.5 <= 255 - 32
        for g in ("P", "D"):
            self.assertNotIn("--barlink-bar1-window-mib", _extra(doc, g))                 # never pinned: the launcher derives them at boot
        out, doc, f = self._case(RIG3)
        self.assertEqual(f["bar1"]["je_karte"], ["32768 (free 32751)", "256 (free 255)", "256 (free 189)"])
        self.assertEqual(f["bar1_windows"]["wert"], "P 16,PP_0=66; D 11,TP_0=22,DCP_0=27")      # (157 - 4 x 3.5) / 208 = 0.69
        self.assertNotEqual(f["bar1_windows"]["wert"], "P 24,PP_0=96; D 16,TP_0=32,DCP_0=40")   # NVML 2 has only 189 free: scaled before the start


#: the propose inventory of each case as the page sends it (``ZielPlaner._propose``)
CASES_INV = {c: (lambda ed, c=c: [{"card": e["id"], "pcie": {"gen": e["pcie_native"]["gen"], "lanes": e["pcie_native"]["lanes"]}} for e in Z.ZielPlaner(ed)._cards(CASES[c])])
             for c in CASES}


if __name__ == "__main__":
    unittest.main()
