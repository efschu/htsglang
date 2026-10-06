"""AP-E of the profile planner: the DUAL form of ``propose()`` (plan PLAN-PROFIL-PLANER-1006 section 3 row AP-E, R3, 4c).

``weg2/propose_dual.py`` (what is Dual-specific) behind ``propose(form="dual")`` of ``weg2/propose.py``.  Dual = group P (PP<N>) and
group D (TP<N>) awake together on the SAME cards; the 27B NVFP4 tree.  GPU-free, NVML-free, Docker-free.

* ``TestPureRules``        the arithmetic that needs no model on the box: level grid, role mapping, cut enumeration, shares, byte-model
                           physics, drift guards (the measured table copy equals ``d_reshard``, the calibration point equals the ledger bytes
                           of the boot it was read from, the env names the planner shows are the ones the Dual code reads).
* ``TestReferenceDual``    A1: the reference rig (5090 + 2x 3080), profile ``27b-nvfp4-dual.env`` (AP0 snapshot, sha 1cc8890c): the proposal IS
                           the profile (argv and env byte-equal, mode ``profil``), its launcher dry run equals the AP0 golden with 0 diff lines;
                           the KV obligation (262144) is JUDGED against those values, not forced onto them: not met, and the verdict says why,
                           per card, with the numbers of ``done/1959-dual-p-262k.md`` (K0 pool 3178 < need 5720 MiB).
* ``TestRuleMode``         ``force_rules`` / ``kv_tokens`` / ``dual_cut``: the shift rule against the 27B seat's own 262k computation
                           (``done/dual-schnitt-262k-1006.md``), the pool search, the pin, the flags of the obligation.
* ``TestOtherInventories`` N=2 (5090 + 3080) and foreign classes: no crash, vectors of N entries, the Dual-Passung and the obligation verdicts
                           present and labelled "Planer-Rechnung, nicht hw_fit", the launcher dry run ends in a NAMED verdict (never an
                           exception that is not a refusal).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.weg2 import dual_layout_plan as DL
    from sglang.srt.weg2 import hw_fit, launcher
    from sglang.srt.weg2 import model_profile as MP
    from sglang.srt.weg2 import propose as P
    from sglang.srt.weg2 import propose_dual as PD
    from sglang.srt.weg2 import propose_oracle as O
    from sglang.srt.weg2 import propose_rules as R
    from sglang.srt.weg2 import propose_verdict as V
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
GOLDEN = os.path.join(FIX, "golden")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
NVFP4 = MC + "Qwen3.8-27B-NVFP4-RadixArk"
DRAFT = MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
#: the Dual golden is made from the REAL NVFP4 checkpoints and the census of the rig box (AP0): the tests that need them skip, with the reason,
#: where they are absent
_NEEDS_DUAL = unittest.skipUnless(
    all(os.path.exists(x) for x in (_CENSUS_27B, NVFP4 + "/config.json", DRAFT + "/config.json")),
    "27B NVFP4 checkpoints / census not on this box (the Dual golden is box-bound, like AP0's)")
MEASURED_RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}

#: tolerance (MiB per card) of the planer's shift rule against the 27B seat's own 262k computation.  The seat's coefficients (90 MiB/layer on the
#: 5090, 205 on a 3080) are "GERECHNET from the dual1e revision terms" (``profiles/27b-nvfp4-dual1h.env`` header), not measured; the planer's follow
#: the INSTALLED D vector (58,25,25: ~116 / ~192 MiB per layer, fix round 1), which the measured D weights confirm (12236 vs 12308 MiB on card 0).
#: Card 0 moves 14 layers between the two cuts: 14 x 26 MiB = ~364 MiB of difference that is the seat's estimate, not a planer error.
TOL_SEAT = 400

_STATE = {}


def setUpModule():
    os.environ["SGLANG_CARD_LIBRARY"] = os.path.join(tempfile.gettempdir(), "ape-no-card-library.json")
    if os.path.exists(NVFP4 + "/config.json") and os.path.exists(DRAFT + "/config.json"):
        mp = MP.estimate_or_state(NVFP4)
        assert mp["ok"], mp
        _STATE["modell"] = mp["profile"]
        _STATE["draft"] = MP.estimate_draft(DRAFT)


def tearDownModule():
    os.environ.pop("SGLANG_CARD_LIBRARY", None)


#: sha256 of the two Dual profile snapshots: the AP0 reference (14:01Z, the golden's profile) and the live release profile as it was at 15:21Z
#: (moved by the 27B seat: cut 31,17,16 / attn 7,5,4, P budgets 6610,5050,5200, P pool level 266240, no 131072 cap, ``--env-d ...D_WANT_LOCKED=1``)
PROVENANCE = {"27b-nvfp4-dual.env": "1cc8890ccece7f9bd1097c30072ec6d87031bf4a8d5e9afb03646992ab39a24e",
              "27b-nvfp4-dual-live1521.env": "ce347c790b8e1aa466ada84c1adfa6765601de982eef0cf8f3b68382fb0e3977"}


def _dual_profile(name: str = "27b-nvfp4-dual"):
    return O.profile_launch_input(os.path.join(PROFILES, name + ".env"))


def _ref_rows():
    return O.read_replay(REPLAY_REF)


def _two_cards():
    rows = [r for r in _ref_rows() if r["index"] in (1, 0)]
    return [dict(r, index=i) for i, r in enumerate(rows)]


def _catalog():
    path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py")
    spec = importlib.util.spec_from_file_location("planer_ape_kartenplan_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {e["id"]: e for e in mod.CATALOG}


def _inventory(name: str):
    """``(hardware for propose, replay rows for the oracle)``."""
    if name == "ref3":
        rows = _ref_rows()
        return rows, rows
    if name == "n2":
        rows = _two_cards()
        return rows, rows
    cat = _catalog()
    ids = {"n3_3090": ["rtx3090-24"] * 3, "n4_3090": ["rtx3090-24"] * 4, "n4_mixed": ["rtx5090-32"] * 2 + ["rtx3080-20"] * 2,
           "n2_5090_3090": ["rtx5090-32", "rtx3090-24"]}[name]
    ents = [cat[i] for i in ids]
    return ents, O.replay_from_catalog(ents)


def _propose(inv: str = "ref3", form: str = "dual", profile: str = "27b-nvfp4-dual", **ziele):
    hw, _ = _inventory(inv)
    return P.propose(hw, _STATE["modell"], form, ziele, basis=_dual_profile(profile), draft=_STATE["draft"], rates=MEASURED_RATES)


def _dry(v, rows, *, force=False):
    b = _dual_profile()
    li = O.LaunchInput(v["argv"], v["env"], b.vars, [], "propose:" + v["basis"], b.instruments)
    return O.run_profile("", rows, tree=TREE, force=force, launch_input=li)


def _by_key(v):
    return {w["key"]: w for w in v["werte"]}


# ---------------------------------------------------------------------------

class TestPureRules(unittest.TestCase):
    def _fp(self, n_layers=16):
        fams = tuple("attn" if i % 4 == 3 else "gdn" for i in range(n_layers))
        return hw_fit.FitProfile(
            profile="syn", weight_format="nvfp4", derived_from="synthetic", n_layers=n_layers, layer_families=fams,
            layer_dense_mib=tuple(250.0 for _ in fams), layer_expert_mib=tuple(0.0 for _ in fams), n_experts=0, embed_mib=2000.0,
            lm_head_mib=600.0, draft_mib=1000.0, visual_mib=800.0, disk_mib=0.0,
            kv_bytes_per_token_per_attn_layer={"fp8_e4m3": 2176.0, "bf16": 4096.0}, mamba_mib_per_slot_per_linear_layer=3.0)

    def test_level_grid(self):
        self.assertEqual(PD.level_tokens(262144), 266240)          # 262144 + 1 page, grid 4096 (done/dual-schnitt-262k-1006.md section 4)
        self.assertEqual(PD.level_tokens(131072), 135168)
        self.assertEqual(PD.level_tokens(4095), 4096)
        self.assertEqual(PD.level_tokens(4096), 8192)              # the page pushes a full grid step over the line
        self.assertEqual(PD.level_tokens(196608 - 1), 196608)

    def test_role_index_reads_a_reference_vector_like_hw_fit(self):
        self.assertEqual([PD.role_index(k, 3, 3) for k in range(3)], [0, 1, 2])
        self.assertEqual([PD.role_index(k, 2, 3) for k in range(2)], [0, 2])
        self.assertEqual([PD.role_index(k, 4, 3) for k in range(4)], [0, 1, 1, 2])
        self.assertEqual([PD.role_index(k, 5, 2) for k in range(5)], [0, 1, 1, 1, 1])
        for n in (2, 3, 4, 5):                                       # the same entry as hw_fit.role_value picks
            vec = [10.0, 20.0, 30.0]
            self.assertEqual([vec[PD.role_index(k, n, 3)] for k in range(n)], [hw_fit.role_value(vec, k, n) for k in range(n)])

    def test_compositions_cover_every_cut(self):
        cuts = list(PD.compositions(64, 3))
        self.assertEqual(len(cuts), 1953)                           # C(63, 2)
        self.assertTrue(all(sum(c) == 64 and min(c) >= 1 and len(c) == 3 for c in cuts))
        self.assertIn((45, 10, 9), cuts)
        self.assertEqual(list(PD.compositions(3, 3)), [(1, 1, 1)])
        self.assertEqual(list(PD.compositions(2, 3)), [])

    def test_budget_rounding_is_the_27b_seats_style(self):
        self.assertEqual(PD.round_budget(6607.0), 6610)
        self.assertEqual(PD.round_budget(6604.9), 6600)
        self.assertEqual(PD.round_budget(5050.0), 5050)

    def test_shares_sum_to_one_and_follow_the_installed_vector(self):
        sh, why = PD.d_shares("nvfp4", [32607, 20480, 20480], True)
        self.assertAlmostEqual(sum(sh), 1.0)
        self.assertAlmostEqual(sh[0], 58 / 108.0)                              # RC9_BASE (58,25,25): the vector D is INSTALLED with
        self.assertAlmostEqual(sh[1], 25 / 108.0)
        self.assertIn("installierter D-Vektor", why)
        self.assertNotIn("Preset 'dec' (Host", why)
        sh, why = PD.d_shares("nvfp4", [32607, 20480, 20480], False)           # no reshard: capacity-first (planer assumption)
        self.assertAlmostEqual(sum(sh), 1.0)
        self.assertAlmostEqual(sh[0], 32607 / 73567.0)
        self.assertIn("proportional", why)
        sh, _ = PD.d_shares("int4", [30000, 20000], True)                      # RC9_BASE is a three-rank vector: other N capacity-first
        self.assertAlmostEqual(sh[1], 0.4)

    def test_installed_vector_equals_d_reshard_base(self):
        try:
            from sglang.srt.weg2 import d_reshard
        except Exception as exc:  # pragma: no cover - the distributed stack is not importable here
            self.skipTest("d_reshard not importable: %s" % exc)
        self.assertEqual(tuple(d_reshard.RC9_BASE), PD.INSTALLED_D_BASE)

    def test_calibration_point_is_the_ledger_bytes_of_its_boot(self):
        """``POOL_REF`` pool MiB = the ``LEDGER-PHYS ... budget=`` bytes of boot b9p (done/1959-dual-p-262k.md section A and
        done/dual-schnitt-262k-1006.md section 2: K0 3 332 374 528 B, K1 6 408 896 512 B, K2 6 123 683 840 B)."""
        self.assertEqual([x * PD.MIB for x in PD.POOL_REF["pool_mib"]], [3332374528, 6408896512, 6123683840])
        self.assertEqual(sum(PD.POOL_REF["cut"]), PD.POOL_REF["n_layers"])

    def test_the_env_names_the_planner_shows_are_the_ones_the_dual_code_reads(self):
        root = os.path.join(TREE, "python", "sglang", "srt")
        green = open(os.path.join(root, "weg2", "dual_green.py"), encoding="utf-8").read()
        share = open(os.path.join(root, "weg2", "dual_share.py"), encoding="utf-8").read()
        env = open(os.path.join(root, "environ.py"), encoding="utf-8").read()
        prefix = re.search(r'ENV_PREFIX = "([A-Z0-9_]+)"', share).group(1)
        self.assertEqual(prefix, "SGLANG_WEG2_DUAL_SHARE_")
        self.assertIn('"TABLE"', green)                                  # GreenConfig.from_env reads <prefix>GREEN_ + TABLE
        self.assertIn('"GREEN_"', green)
        self.assertIn("STARVE_AGE_S", share)
        self.assertIn("STARVE_MAX_RUNG", share)
        self.assertIn("SGLANG_WEG2_DUAL_GRANT_RETRY_MS", env)

    def test_stage_terms_are_physically_consistent(self):
        fp = self._fp()
        m = PD.model_bytes(fp)
        cut = (8, 5, 3)
        sh = [0.6, 0.25, 0.15]
        t = PD.stage_terms(m, cut, sh, slots=8, slot_mib=1.5, cell_b=2048.0, boot_tokens=1000, vision_in_p=False)
        lay = DL.DualLayout(d_card=(0, 1, 2), p_card=(0, 1, 2), p_cut=cut, mixer=tuple(sh), mlp=tuple(sh), vocab=tuple(sh), draft=tuple(sh),
                            draft_stage=None, vision_in_p=False, d_embed_replicated=True)
        rows = DL.plan(m, lay, {i: 1 << 50 for i in range(3)}, {i: 0 for i in range(3)})
        self.assertEqual(DL.check_physics(rows, m, lay), [])
        self.assertEqual([round(x["priv"] * PD.MIB) for x in t], [r.pp_only for r in rows])
        self.assertEqual([x["fa"] for x in t], [2, 1, 1])                         # layers 3, 7 | 11 | 15
        self.assertEqual([x["gdn"] for x in t], [6, 4, 2])
        self.assertAlmostEqual(t[0]["mam"], 6 * 1.5 * 8)
        self.assertAlmostEqual(t[1]["kv"], 1 * 2048.0 * 1000 / PD.MIB)
        # D's shard of the draft is its share of the draft bytes
        self.assertAlmostEqual(t[2]["d_draft"], 1000.0 * 0.15)


@_NEEDS_DUAL
class TestReferenceDual(unittest.TestCase):
    """A1: the reference rig, the Dual release profile."""

    def test_the_proposal_for_the_profiles_own_inventory_is_the_profile(self):
        li = _dual_profile()
        v = _propose("ref3")
        self.assertEqual(v["form"], "dual")
        self.assertTrue(v["inventory"]["gleich_wie_profil"])
        self.assertEqual(v["argv"], list(li.argv))                       # byte-equal argv: nothing re-serialised
        self.assertEqual(v["env"], dict(li.env))
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])           # the BAR1 window spec "16,PP_0=64" is no per-card vector
        self.assertEqual(v["blocker"], [])
        d = v["dual"]
        self.assertEqual(d["schema"], PD.SCHEMA)
        self.assertEqual(d["modus"], "profil")
        self.assertTrue(d["regeln"]["uebernommen"])
        self.assertEqual(d["regeln"]["schnitt"], [45, 10, 9])
        self.assertEqual(d["regeln"]["attn"], [11, 2, 3])
        self.assertEqual(d["regeln"]["budgets"], [8740, 3000, 3500])
        for w in v["werte"]:
            if w["policy"] in ("class", "cut", "cut_attn", "advisory", "dual_pflicht", "dual_knob", "dual_env"):
                self.assertFalse(w["geaendert"], w["key"])
                if w["key"] != "--pp-solve-pool-floor":                    # the floor is no profile value: its origin is the obligation
                    self.assertTrue(w["herkunft"].startswith("Profil "), w)
        json.dumps(v)                                                     # the worker returns it over a pipe

    def test_the_dual_values_of_the_profile_are_recorded_with_their_origin(self):
        w = _by_key(_propose("ref3"))
        for key in ("--extra-p --rank-gpu-memory-mib", "--pp-stage-ratio", "--pp-attn-stage-ratio", "--dual-p-kv-max-tokens",
                    "--dual-d-kv-max-tokens", "--dual-p-overhead-mib", "--max-kv-per-request", "--dual-priority", "--dual-share-actuators",
                    "--dual-green-ladder", "env SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE", "env SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S",
                    "env SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG", "env SGLANG_WEG2_DUAL_GRANT_RETRY_MS", "--pp-solve-pool-floor"):
            self.assertIn(key, w, key)
            self.assertTrue(w[key]["herkunft"] and w[key]["grund"], key)
        self.assertEqual(w["--extra-p --rank-gpu-memory-mib"]["wert"], "8740,3000,3500")
        self.assertEqual(w["--dual-p-overhead-mib"]["wert"], "2500")
        self.assertEqual(w["env SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE"]["wert"], "1:1:1;2:2:2;1000000000:3:3")
        self.assertEqual(w["env SGLANG_WEG2_DUAL_GRANT_RETRY_MS"]["wert"], "20")
        # the pool floor is NOT typed: the launcher derives cap + chunk (and a typed 262144 would lower it)
        self.assertFalse(w["--pp-solve-pool-floor"]["in_argv"])
        self.assertIn("132096 = 131072 + 1024", w["--pp-solve-pool-floor"]["grund"])
        self.assertIn("263168", w["--pp-solve-pool-floor"]["grund"])

    def test_the_green_table_the_planner_carries_is_the_one_the_code_parses(self):
        from sglang.srt.weg2 import dual_green

        v = _propose("ref3")
        cfg = dual_green.GreenConfig.from_env(v["env"])
        self.assertEqual(tuple(cfg.table), ((1, 1, 1), (2, 2, 2), (1000000000, 3, 3)))

    def test_the_obligation_is_judged_not_forced(self):
        d = _propose("ref3")["dual"]
        p = d["pflicht"]
        self.assertEqual((p["kv_tokens"], p["level_tokens"], p["top"], p["cap"]), (262144, 266240, 196608, 131072))
        self.assertEqual((p["pool_floor_in_kraft"], p["pool_floor_pflicht"]), (132096, 263168))
        self.assertIs(p["erfuellt"], False)
        self.assertTrue(p["pool_geeicht"])
        text = " | ".join(p["grund"])
        self.assertIn("--dual-p-kv-max-tokens 196608 < Level 266240", text)
        self.assertIn("--max-kv-per-request 131072 < 262144", text)
        # the numbers of done/1959-dual-p-262k.md: K0 pool 3178 MiB < need 5720 MiB (11 FA layers x 2048 B x 266240), short 2542
        k0 = d["karten"][0]
        self.assertEqual((k0["pool_mib"], k0["bedarf_mib"], k0["pool_rest_mib"]), (3178.0, 5720.0, -2542.0))
        self.assertIn("Karte 0 (RTX5090): Pool 3178 MiB < Bedarf 5720 MiB (Fehlbetrag 2542 MiB", text)
        for k in (1, 2):                                              # K1 / K2 have room (1959: "nur K0 scheitert")
            self.assertGreater(d["karten"][k]["pool_rest_mib"], 0)
        self.assertEqual([c["bedarf_mib"] for c in d["karten"]], [5720.0, 1040.0, 1560.0])

    def test_the_dual_passung_is_a_planner_calculation_and_says_so(self):
        d = _propose("ref3")["dual"]
        self.assertEqual(d["etikett"], "Dual-Passung: Planer-Rechnung, nicht hw_fit")
        self.assertEqual(d["passung"]["stufe"], "ja")
        self.assertIn("Dual-Passung: Planer-Rechnung, nicht hw_fit", d["verdikte"][0]["text"])
        for r in d["karten"]:
            # the coupling: P budget + overhead + D rest + D weights (draft inside) + D mamba <= card, exactly as the rows state it
            self.assertAlmostEqual(r["summe_mib"], r["p_budget_mib"] + r["overhead_mib"] + r["d_ruhe_mib"] + r["d_gewichte_mib"] + r["d_mamba_mib"], delta=0.2)
            self.assertAlmostEqual(r["rest_mib"], r["karte_mib"] - r["summe_mib"], delta=0.2)
            self.assertGreater(r["d_draft_mib"], 0)
        # D is sized from P's plan: the card's D budget of the launcher ("D sized from P's PLAN") is total - P - overhead - awake rest
        self.assertEqual([r["d_budget_mib"] for r in d["karten"]], [18176, 12901, 12413])
        self.assertEqual(d["draft"]["p"], "keiner (--draft-kv-on-p off)")

    def test_d_weights_on_card_0_meet_the_measured_dual_boot(self):
        """Fix round 1, finding 1: D's weights of the Passung are the shard of the INSTALLED vector (RC9_BASE 58,25,25), measured at boot a3t5js:
        D TP0 weights 12.020 GiB = 12308 MiB (``dual_w64.py`` docstring; the family model of the launcher prices 13362).  The old line priced
        card 0 at 10523 MiB (capacity-proportional 0.443 share, 1785 MiB too low)."""
        self.assertEqual(PD.D_WEIGHTS_K0_MEASURED_MIB, 12308)
        for kw in ({}, {"force_rules": True}):
            d = _propose("ref3", **kw)["dual"]
            k0 = d["karten"][0]
            self.assertLessEqual(abs(k0["d_gewichte_mib"] - 12308), 123, k0)               # model vs measurement: within 1 %
            self.assertGreater(k0["d_gewichte_mib"], 10523 + 1000, k0)                      # the old, too-low value is gone
            # D's Mamba pool follows the same vector: card 0 holds the 58/108 share of it (measured 1.388 GiB = 1421 MiB, +-10 %)
            self.assertLessEqual(abs(k0["d_mamba_mib"] - 1421), 142, k0)

    def test_p_private_and_d_weights_of_a_card_are_one_vector(self):
        """Fix round 1, finding 1: P-private weights (what P holds beyond D's shard) and D's weights come from the SAME D vector on every card:
        the sum is a real state of D.  Recomputed here from the installed vector with the planer's own byte model."""
        v = _propose("ref3")
        d = v["dual"]
        fp = R.fit_profile_from_model(_STATE["modell"], _STATE["draft"])
        m = PD.model_bytes(fp)
        tot = float(sum(PD.INSTALLED_D_BASE))
        sh = [x / tot for x in PD.INSTALLED_D_BASE]
        cut = d["regeln"]["schnitt"]
        argv = list(_dual_profile().argv)
        vision = argv[argv.index("--weg2-vision") + 1] if "--weg2-vision" in argv else ""
        t = PD.stage_terms(m, cut, sh, slots=8, slot_mib=1.0, cell_b=1.0, boot_tokens=1, vision_in_p=vision != "transient")
        for k, r in enumerate(d["karten"]):
            self.assertAlmostEqual(r["d_gewichte_mib"], t[k]["d_total"], delta=0.2)
            self.assertAlmostEqual(r["p_privat_mib"], t[k]["priv"], delta=0.2)
        self.assertTrue(any("EIN Vektor" in a for a in d["annahmen"]), d["annahmen"])
        self.assertFalse(any("0.72" in a for a in d["annahmen"]), d["annahmen"])

    def test_hw_fit_is_told_the_form_is_dual_and_its_verdict_never_blocks_it(self):
        v = _propose("ref3")
        self.assertTrue(any("NOT modelled" in m for m in v["fit"]["marks"]), v["fit"]["marks"])      # hw_fit.py: "Dual ... NOT modelled"

    def test_proposal_dry_run_equals_golden(self):
        """The launcher dry run of the proposal for the profile's own inventory is the AP0 Dual golden: 0 diff lines, nothing forced."""
        v = _propose("ref3")
        run = _dry(v, _ref_rows())
        res = run.result
        self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:300]))
        self.assertEqual((res.rc, res.forced), (0, []))
        with open(os.path.join(GOLDEN, "plan_27b_dual_n3.txt"), encoding="utf-8") as fh:
            want = fh.read()
        d = O.diff_lines(want, res.dump())
        self.assertEqual(d, [], "plan diff vs plan_27b_dual_n3.txt: %d lines\n%s" % (len(d), "\n".join(x[:240] for x in d[:12])))
        self.assertTrue(any(k.startswith("WEG2-DUAL-SHARE") for k in O.parse_plan_dump(res.text)["kinds"]))

    def test_flip_and_tp_proposals_carry_no_dual_section(self):
        for form in ("flip", "tp"):
            v = _propose("ref3", form)
            self.assertIsNone(v["dual"], form)
            self.assertFalse(any(w["policy"].startswith("dual") for w in v["werte"]), form)


@_NEEDS_DUAL
class TestLiveProfile(unittest.TestCase):
    """The live release profile of 15:21Z (the 27B seat moved it to the 262k cut): carried byte for byte; the obligation it was built for is
    judged MET by the planner's own calibrated pool, within the tolerance the planner states."""

    LIVE = "27b-nvfp4-dual-live1521"

    def test_snapshots_are_what_their_provenance_says(self):
        import hashlib

        for name, sha in PROVENANCE.items():
            with open(os.path.join(PROFILES, name), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), sha, name)

    def test_the_live_profile_is_carried_and_meets_the_obligation(self):
        li = _dual_profile(self.LIVE)
        v = _propose("ref3", profile=self.LIVE)
        self.assertEqual(v["argv"], list(li.argv))
        self.assertEqual(v["env"], dict(li.env))
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])
        self.assertEqual(v["blocker"], [])
        d = v["dual"]
        self.assertEqual(d["modus"], "profil")
        self.assertEqual((d["regeln"]["schnitt"], d["regeln"]["attn"], d["regeln"]["budgets"]), ([31, 17, 16], [7, 5, 4], [6610, 5050, 5200]))
        p = d["pflicht"]
        self.assertEqual((p["cap"], p["top"], p["level_tokens"]), (262144, 266240, 266240))
        self.assertEqual((p["pool_floor_in_kraft"], p["pool_floor_pflicht"]), (263168, 263168))
        self.assertIs(p["erfuellt"], True, p["grund"])
        self.assertEqual([r["bedarf_mib"] for r in d["karten"]], [3640.0, 2600.0, 2080.0])       # the seat's need column for 31,17,16
        for r, want in zip(d["karten"], (4563, 4627, 4330)):                                    # the seat's pool column; tolerance: see TOL_SEAT
            self.assertLessEqual(abs(r["pool_mib"] - want), TOL_SEAT, (r["pool_mib"], want))
        self.assertEqual(d["passung"]["stufe"], "ja")
        # the profile's budgets ARE the calibration of the rule: at its own cut the rule returns them
        w = _propose("ref3", profile=self.LIVE, dual_cut=[31, 17, 16])["dual"]
        self.assertEqual(w["regeln"]["budgets"], [6610, 5050, 5200])

    def test_the_rule_search_on_the_live_profile_keeps_the_obligation_and_does_not_slow_p_down(self):
        """The search starts from the live profile's own calibration; the cut it picks is at least as fast as the profile's (it may be faster:
        the live cut 31,17,16 was chosen for margin under D load, which the search does not model)."""
        d = _propose("ref3", profile=self.LIVE, force_rules=True)["dual"]
        self.assertIs(d["pflicht"]["erfuellt"], True)
        rate = (203.42, 50.97, 50.97)
        mk = lambda c: max(c[k] / rate[k] for k in range(3))                                     # noqa: E731
        self.assertLessEqual(mk(d["regeln"]["schnitt"]), mk((31, 17, 16)) + 1e-9)

    def test_the_moved_profile_on_this_box_meets_the_launchers_w64(self):
        """The dry run of the moved profile: on a box whose evidence dir holds no measured dual-D log the launcher's own W64 model refuses
        it (the seat's note: 'Ohne gemessenes D-Log refused das MODELL-W64 jeden Schnitt, der den 3080 mehr P-Budget gibt').  What matters here:
        the oracle returns that as a NAMED launcher verdict, the planner does not hide it."""
        v = _propose("ref3", profile=self.LIVE)
        b = _dual_profile(self.LIVE)
        li = O.LaunchInput(v["argv"], v["env"], b.vars, [], "propose:" + v["basis"], b.instruments)
        res = O.run_profile("", _ref_rows(), tree=TREE, force=False, launch_input=li).result
        if res.exc_type is not None:
            self.assertEqual(res.exc_type, "Weg2LaunchRefused", res.exc_msg[:300])
            self.assertIn("W64", res.exc_msg[:200])
        else:
            self.assertEqual(res.rc, 0)


@_NEEDS_DUAL
class TestRuleMode(unittest.TestCase):
    """Mode ``regel``: the Dual values are derived (shift rule + pool search) instead of carried."""

    def test_force_rules_derives_a_cut_that_carries_the_obligation(self):
        v = _propose("ref3", force_rules=True)
        d = v["dual"]
        self.assertEqual(d["modus"], "regel")
        self.assertFalse(d["regeln"]["uebernommen"])
        self.assertIs(d["pflicht"]["erfuellt"], True, d["pflicht"]["grund"])
        w = _by_key(v)
        self.assertEqual(w["--max-kv-per-request"]["wert"], "262144")
        self.assertEqual(w["--dual-p-kv-max-tokens"]["wert"], "266240")
        self.assertEqual((d["pflicht"]["cap"], d["pflicht"]["top"]), (262144, 266240))
        self.assertEqual(d["pflicht"]["pool_floor_in_kraft"], 263168)                 # cap + chunk, derived by the launcher
        self.assertNotIn("--pp-solve-pool-floor", v["argv"])                          # never typed (it would lower the floor)
        self.assertEqual(sum(d["regeln"]["schnitt"]), 64)
        self.assertEqual(d["regeln"]["attn"], R.attn_counts(self._families(), d["regeln"]["schnitt"]))
        for r in d["karten"]:
            self.assertGreaterEqual(r["pool_rest_mib"], 0.0, r)                          # the search kept every card above the level
            self.assertTrue(r["ok"], r)
        # the cut is the FASTEST that fits: the search ran, the note says so, the values are labelled a Planer-Rechnung
        self.assertIn("Schnitt gesucht", d["regeln"]["suche"])
        for k in ("--pp-stage-ratio", "--pp-attn-stage-ratio", "--extra-p --rank-gpu-memory-mib"):
            self.assertEqual(w[k]["zustand"], "unbelegt", k)
            self.assertTrue(w[k]["geaendert"], k)
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])
        # the profile's cut/budget are shown as the OLD value (read from the profile, not from a half-derived state)
        self.assertEqual(w["--pp-stage-ratio"]["alt"], "45,10,9")
        self.assertEqual(w["--extra-p --rank-gpu-memory-mib"]["alt"], "8740,3000,3500")
        json.dumps(v)

    def _families(self):
        return R.fit_profile_from_model(_STATE["modell"], _STATE["draft"]).layer_families

    def test_the_search_is_exhaustive_and_minimal(self):
        """No cut that is faster (smaller slowest stage by GEMM rate) clears the obligation on every card."""
        v = _propose("ref3", force_rules=True)
        d = v["dual"]
        rate = [c["tflops"] for c in v["cards"]]
        chosen = d["regeln"]["schnitt"]
        mk = max(chosen[k] / rate[k] for k in range(3))
        fp = R.fit_profile_from_model(_STATE["modell"], _STATE["draft"])
        m = PD.model_bytes(fp)
        sh, _ = PD.d_shares("nvfp4", [c["total_mib"] for c in v["cards"]], True)
        cell = 2048.0
        lvl = PD.level_tokens(262144)
        ref = PD.stage_terms(m, PD.POOL_REF["cut"], sh, slots=8, slot_mib=1.5588, cell_b=cell, boot_tokens=95771, vision_in_p=False)
        faster = 0
        for cand in PD.compositions(64, 3):
            if max(cand[k] / rate[k] for k in range(3)) >= mk - 1e-9:
                continue
            t = PD.stage_terms(m, cand, sh, slots=8, slot_mib=1.5588, cell_b=cell, boot_tokens=95771, vision_in_p=False)
            pool = [PD.POOL_REF["pool_mib"][k] + (ref[k]["priv"] + ref[k]["mam"]) - (t[k]["priv"] + t[k]["mam"]) for k in range(3)]
            need = [t[k]["fa"] * cell * lvl / PD.MIB for k in range(3)]
            if min(pool[k] - need[k] for k in range(3)) >= 0:
                faster += 1
        self.assertEqual(faster, 0, "a faster cut %s of the search space also clears the obligation" % chosen)

    def test_the_shift_rule_against_the_27b_seats_own_262k_computation(self):
        """``done/dual-schnitt-262k-1006.md`` section 3/4 (dual1h method, measured coefficients 90 / 205 MiB per layer): cut 31,17,16 / 7,5,4 ->
        P budgets 6610,5050,5200 and pool 4563/4627/4330 MiB.  The planner's coefficients come from the model's layer sizes and D's installed vector,
        not from fitted per-class numbers: it must agree within TOL_SEAT (400 MiB per card, see there) and never claim more."""
        v = _propose("ref3", dual_cut=[31, 17, 16])
        d = v["dual"]
        self.assertEqual(d["regeln"]["schnitt"], [31, 17, 16])
        self.assertEqual(d["regeln"]["attn"], [7, 5, 4])                                  # the natural FA count of 31,17,16 (the seat's W40 note)
        for got, want in zip(d["regeln"]["budgets"], (6610, 5050, 5200)):
            self.assertLessEqual(abs(got - want), TOL_SEAT, (d["regeln"]["budgets"], want))
        for r, want in zip(d["karten"], (4563, 4627, 4330)):
            self.assertLessEqual(abs(r["pool_mib"] - want), TOL_SEAT, (r["pool_mib"], want))
        # the 262k obligation holds at this cut on all three cards (the seat's margin +923/+2027/+2250 at D -> 0)
        self.assertIs(d["pflicht"]["erfuellt"], True)
        self.assertEqual([r["bedarf_mib"] for r in d["karten"]], [3640.0, 2600.0, 2080.0])
        self.assertIn("dual_cut", _by_key(v)["--pp-stage-ratio"]["herkunft"])
        self.assertEqual(_by_key(v)["--pp-stage-ratio"]["zustand"], "vorgeschlagen")

    def test_the_old_cut_of_the_profile_does_not_carry_the_obligation(self):
        """The profile's own cut 45,10,9 fails on K0 (done/1959): a pin on it keeps the seat's finding."""
        d = _propose("ref3", dual_cut=[45, 10, 9])["dual"]
        self.assertIs(d["pflicht"]["erfuellt"], False)
        self.assertEqual(d["karten"][0]["pool_rest_mib"], -2542.0)
        self.assertEqual(d["regeln"]["budgets"][0], 8740)                              # at the profile's own cut the shift is zero
        self.assertEqual(d["regeln"]["budgets"], [8740, 3000, 3500])

    def test_an_explicit_kv_obligation_switches_to_the_rule(self):
        v = _propose("ref3", kv_tokens=131072)
        d = v["dual"]
        self.assertEqual(d["modus"], "regel")
        self.assertEqual((d["pflicht"]["level_tokens"], d["pflicht"]["top"], d["pflicht"]["cap"]), (135168, 135168, 131072))
        self.assertEqual(d["pflicht"]["pool_floor_pflicht"], 131072 + 1024)

    def test_a_bad_pin_is_refused(self):
        for bad in ([10, 10], [30, 30, 5], [0, 32, 32], "a,b,c"):
            with self.assertRaises(P.ProposeError, msg=repr(bad)):
                _propose("ref3", dual_cut=bad)

    def test_a_basis_without_dual_flags_gets_the_flags_of_the_form(self):
        li = O.profile_launch_input(os.path.join(PROFILES, "27b-base.env"))
        hw, _ = _inventory("ref3")
        v = P.propose(hw, _STATE["modell"], "dual", {}, basis=li, draft=_STATE["draft"], rates=MEASURED_RATES)
        self.assertIn("--dual-share", v["argv"])
        self.assertEqual(v["argv"][v["argv"].index("--dual-unified-kv") + 1], "on")
        self.assertTrue(any("kein Dual-Profil" in h for h in v["hinweise"]), v["hinweise"])
        self.assertEqual(v["dual"]["modus"], "regel")

    def test_the_w64_note_names_what_decides_it(self):
        v = _propose("ref3", force_rules=True)
        self.assertTrue(any("W64" in h and "dual_w64" in h for h in v["hinweise"]), v["hinweise"])


@_NEEDS_DUAL
class TestOtherInventories(unittest.TestCase):
    """A2-style: other inventories.  The proposal is a Planer-Rechnung; the launcher dry run says what it makes of it."""

    def _check(self, inv: str, *, n: int):
        hw, rows = _inventory(inv)
        v = P.propose(hw, _STATE["modell"], "dual", {}, basis=_dual_profile(), draft=_STATE["draft"], rates=MEASURED_RATES)
        self.assertEqual(v["n"], n)
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])                   # every positional vector of the proposal has N entries
        d = v["dual"]
        self.assertEqual(d["modus"], "regel")
        self.assertEqual(len(d["regeln"]["schnitt"]), n)
        self.assertEqual(len(d["regeln"]["budgets"]), n)
        self.assertEqual(sum(d["regeln"]["schnitt"]), 64)
        self.assertEqual(len(d["karten"]), n)
        self.assertEqual([x["code"] for x in d["verdikte"]], ["DUAL-PASSUNG", "DUAL-PFLICHT"])
        self.assertTrue(all("Planer-Rechnung, nicht hw_fit" in x["text"] for x in d["verdikte"]))
        json.dumps(v)
        run = _dry(v, rows, force=True)
        res = run.result
        # a result, never a crash: either the dry run ran through or it ended in a launcher REFUSAL (named, with its W-code)
        if res.exc_type is not None:
            self.assertIn(res.exc_type, ("Weg2LaunchRefused", "Weg2XchgResidencyUnarmable", "Weg2PPCutRefused", "Weg2TpOperatingPointInfeasible"),
                          "%s: %s" % (res.exc_type, res.exc_msg[:300]))
        return v, run

    def test_n2_5090_plus_3080_dual(self):
        v, run = self._check("n2", n=2)
        d = v["dual"]
        # the 5090 and the 3080 have twin classes in the profile: the budgets carry its calibration residual (labelled unbelegt)
        self.assertEqual(len(d["regeln"]["budgets"]), 2)
        self.assertIs(d["pflicht"]["pool_geeicht"], False)                        # the pool calibration is the 3-card reference boot's
        self.assertIsNone(d["pflicht"]["erfuellt"])                                 # flags are met, the pool is not computable: "nicht gerechnet"
        self.assertTrue(any("nicht gerechnet" in x["text"] for x in d["verdikte"] if x["code"] == "DUAL-PFLICHT"))
        # the launcher cannot price the profile's 3-entry alloc-cache record on two cards: a NAMED refusal (record of another inventory)
        self.assertEqual(run.result.exc_type, "Weg2LaunchRefused")
        self.assertIn("D_DUAL_ALLOC_CACHE_BOOK_MIB has 3 entries for 2 cards", run.result.exc_msg)

    def test_n2_goes_through_the_whole_pipeline_with_a_verdict_document(self):
        """The dashboard worker's job (``propose_verdict.run_propose``): proposal + oracle + verdicts, as one JSON document."""
        res = V.run_propose({"basis": {"env_path": os.path.join(PROFILES, "27b-nvfp4-dual.env")}, "inventar": {"devices": _two_cards()},
                             "form": "dual", "ziele": {}, "snapshots": {}}, tree=TREE)
        self.assertTrue(res["ok"], res)
        json.dumps(res)
        verd = res["verdikt"]
        self.assertNotEqual(verd["ausgang"], "orakel_fehler", verd.get("verdikte"))
        self.assertNotEqual(verd["ausgang"], "absturz")
        codes = [x["code"] for x in verd["verdikte"]]
        self.assertIn("DUAL-PASSUNG", codes)
        self.assertIn("DUAL-PFLICHT", codes)
        self.assertIn("FIT", codes)
        fit = next(x for x in verd["verdikte"] if x["code"] == "FIT")
        self.assertIn("Dual nicht modelliert", fit["text"])
        self.assertEqual(fit["force_state"], "hinweis")                            # hw_fit does not block a form it does not model
        pas = next(x for x in verd["verdikte"] if x["code"] == "DUAL-PASSUNG")
        self.assertIn("Planer-Rechnung, nicht hw_fit", pas["text"])
        self.assertEqual(pas["titel"], "Dual-Passung: Planer-Rechnung, nicht hw_fit")
        self.assertTrue(any(x["ebene"] == "lauf" for x in verd["verdikte"]))      # the launcher's own refusal is a verdict too
        self.assertEqual(set(res["je_wert"]), {w["key"] for w in res["vorschlag"]["werte"]})

    def test_foreign_classes_run_to_a_named_launcher_verdict(self):
        for inv, n in (("n3_3090", 3), ("n4_3090", 4), ("n4_mixed", 4), ("n2_5090_3090", 2)):
            with self.subTest(inv=inv):
                v, run = self._check(inv, n=n)
                for r in v["dual"]["karten"]:
                    self.assertIn("p_budget_mib", r)
                # a card class without a twin in the profile has no calibration residual: its budget is a model value, said so
                self.assertTrue(any("kein Eichrest" in u or "Eichrest" in u for u in v["unbelegt"]) or inv == "n4_mixed", v["unbelegt"])

    def test_the_launcher_refusal_of_foreign_cards_is_not_hidden_by_the_planer(self):
        v, run = self._check("n3_3090", n=3)
        self.assertEqual(run.result.exc_type, "Weg2LaunchRefused")
        self.assertIn("W19", run.result.exc_msg)                                  # an uncalibrated third class: AP1's port, not the planner's


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
