"""VISION-WEIGHTS AP4 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 section 8, AP4): the planner side, the launcher verdicts and the host ledger post
of ``--pdflip-vision-place weights``.  GPU-free, NVML-free, Docker-free.

* ``TestPlanArithmetic``    ``pdflip/vision_victim_plan.py``: the item 'Vision transient' (victim bytes >= tower bytes, 878.8 MiB on the 27B, resident 0),
                            the host item (27B = the victim bytes, NF = 0), the verdict W105b (code + numbers, never a lock), the victim kind per form.
* ``TestProposal``          ``propose()`` (flip / dual) carries the ``vision`` section ONLY with the place ``weights``; the default profile is byte-equal
                            (argv, env, values); ``propose_verdict.build_verdict`` turns the section into the W105b / VISION-VICTIM verdict (hint, not a lock).
* ``TestBars``              ``profile_couplings.phase_bars`` (schema ``flliper.bar/1``): sub-item in the weights segment + host row, NOT added to the sum;
                            the default bar has not one new key.
* ``TestLauncherVerdicts``  each launcher refusal red -> green: ``kvtail|auto`` told + ``--dual-layout``; ``vision_async_refusal`` (port from the NF line);
                            the host-RAM warning term; the default dual / flip dry runs print nothing new.  Since 2026-10-09 the 27B release
                            profile snapshots name ``weights`` themselves (``27b-base.env``): "default" = that line taken out (``_without_place``).
* ``TestHostLedgerPost``    ``host_ledger.charge_terms(vision_victim_host_gib=)``: run-moment term only, 0 changes no number, warning only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from flliper.srt.planner import profile_couplings as PC
    from flliper.srt.pdflip import host_ledger, launcher
    from flliper.srt.pdflip import model_profile as MP
    from flliper.srt.pdflip import propose as P
    from flliper.srt.pdflip import propose_oracle as O
    from flliper.srt.pdflip import propose_verdict as PV
    from flliper.srt.pdflip import vision_victim_plan as VV
except Exception as exc:  # pragma: no cover - no pdflip launcher in this build
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
NVFP4 = MC + "Qwen3.8-27B-NVFP4-RadixArk"
DRAFT = MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
_NEEDS_27B = unittest.skipUnless(os.path.exists(NVFP4 + "/config.json") and os.path.exists(DRAFT + "/config.json"),
                                 "27B NVFP4 checkpoints not on this box")
MIB = 1 << 20
RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}
_STATE = {}


def _ref_rows():
    return O.read_replay(REPLAY_REF)


def _profile(name="27b-nvfp4-dual"):
    return O.profile_launch_input(os.path.join(PROFILES, name + ".env"))


def _without_place(argv):
    """``argv`` with every ``--pdflip-vision-place <value>`` pair removed: the launcher DEFAULT (place unset).  Since 2026-10-09 the 27B release
    profiles (and their snapshots here) name ``weights`` themselves (user decision; ``27b-base.env``, the Dual inherits it by ``source``)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--pdflip-vision-place":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _with_place(base, place):
    """The profile with its own place removed and ``--pdflip-vision-place <place>`` appended (None = the place left unset)."""
    argv = _without_place(base.argv) + ([] if place is None else ["--pdflip-vision-place", place])
    return O.LaunchInput(argv, base.env, base.vars, [], "t", base.instruments)


def setUpModule():
    os.environ["FLLIPER_CARD_LIBRARY"] = os.path.join(tempfile.gettempdir(), "ap4-no-card-library.json")
    if os.path.exists(NVFP4 + "/config.json"):
        mp = MP.estimate_or_state(NVFP4)
        assert mp["ok"], mp
        _STATE["modell"] = mp["profile"]
    if os.path.exists(DRAFT + "/config.json"):
        _STATE["draft"] = MP.estimate_draft(DRAFT)


def tearDownModule():
    os.environ.pop("FLLIPER_CARD_LIBRARY", None)


# ---------------------------------------------------------------------------
# the arithmetic of the module
# ---------------------------------------------------------------------------
class TestPlanArithmetic(unittest.TestCase):
    def sec(self, **kw):
        a = dict(form="dual", vision="transient", place="weights", is_moe=False, dual=True, tower_bytes=VV.TOWER_BYTES_27B, tower_src="t",
                 available_mib=4060.0, available_src="a")
        a.update(kw)
        return VV.section(**a)

    def test_the_27b_tower_is_878_8_mib(self):
        """921460192 B = the sum of the 333 tower tensors of the 27B checkpoints' safetensors headers (measured 2026-10-09), 878.8 MiB: the number of
        the AP2 commit c685baaf8f and of the plan section 0."""
        self.assertEqual(VV.TOWER_BYTES_27B, 921460192)
        self.assertEqual(round(VV.TOWER_BYTES_27B / MIB, 1), VV.TOWER_MIB_27B)
        self.assertEqual(VV.TOWER_MIB_27B, 878.8)

    @unittest.skipUnless(os.path.exists(NVFP4 + "/model.safetensors.index.json"), "27B NVFP4 checkpoint not on this box")
    def test_the_constant_is_the_header_of_the_checkpoint_and_the_model_profile(self):
        b, src = VV.tower_bytes_from_checkpoint(NVFP4)
        self.assertEqual(b, VV.TOWER_BYTES_27B, src)
        self.assertIn("333 tensors", src)
        self.assertEqual(_STATE["modell"]["weights"]["visual_bytes"]["v"], VV.TOWER_BYTES_27B)

    def test_unreadable_checkpoint_is_said_not_guessed(self):
        b, why = VV.tower_bytes_from_checkpoint("/nonexistent-ap4-dir")
        self.assertIsNone(b)
        self.assertIn("unread", why)

    def test_inactive_unless_transient_and_weights(self):
        for vision, place in (("transient", "auto"), ("transient", ""), ("transient", None), ("off", "weights"), ("resident", "weights"), (None, None)):
            s = VV.section(form="flip", vision=vision, place=place, is_moe=False, dual=False, tower_bytes=1.0, tower_src="", available_mib=1.0, available_src="")
            self.assertEqual(set(s), {"schema", "aktiv", "vision", "place"}, (vision, place))
            self.assertFalse(s["aktiv"])

    def test_the_item_is_a_check_item_not_a_cost(self):
        s = self.sec()
        t = s["vision_transient"]
        self.assertEqual((t["mib"], t["resident_mib"], t["in_summe"], t["transient"]), (878.8, 0.0, False, True))
        self.assertEqual(s["resident_mib"], 0.0)
        self.assertEqual(s["opferart"], "pp_only")

    def test_dual_expects_the_two_merger_linears_row_split_flip_and_nf_nothing(self):
        d = self.sec(dual=True)
        self.assertEqual(d["split_tensors_erwartet"], 2)
        self.assertIn("unverified on the metal", d["split_tensors_quelle"])
        self.assertNotIn("split_tensors_erwartet", self.sec(form="flip", dual=False))
        self.assertNotIn("split_tensors_erwartet", self.sec(form="flip", is_moe=True, dual=False))

    def test_host_item_27b_is_the_victim_bytes_nf_is_zero(self):
        d = self.sec(dual=True)
        self.assertEqual((d["host_mib"], d["host_posten"]["mib"], d["host_posten"]["nach_rueckholung_mib"]), (878.8, 878.8, 0.0))
        f = self.sec(form="flip", dual=False)
        self.assertEqual((f["opferart"], f["host_mib"]), ("dense", 878.8))
        nf = self.sec(form="flip", is_moe=True, dual=False, tower_bytes=897862112)
        self.assertEqual((nf["opferart"], nf["host_mib"], nf["turm_mib"]), ("experts", 0.0, 856.3))

    def test_tp_form_has_no_vision_stage(self):
        s = self.sec(form="tp", dual=False)
        self.assertFalse(s["aktiv"])
        self.assertIn("--d-only", s["note"])

    def test_verdict_w105b_carries_code_and_numbers(self):
        no = self.sec(available_mib=500.0)["verdict"]
        self.assertEqual((no["code"], no["stage"], no["missing_mib"]), ("W105b", "nein", 378.8))
        self.assertIn("W105b PdFlipVisionVictimShort", no["text"])
        for n in ("500.0", "878.8", "378.8"):
            self.assertIn(n, no["text"])
        ja = self.sec(available_mib=4060.0)["verdict"]
        self.assertEqual((ja["stage"], ja["rest_mib"]), ("ja", 3181.2))
        edge = self.sec(available_mib=878.8)["verdict"]
        self.assertEqual(edge["stage"], "ja")                                  # victim bytes == tower bytes is enough (>=)
        unk = self.sec(available_mib=None)["verdict"]
        self.assertEqual(unk["stage"], "ungeprueft")
        self.assertIsNone(self.sec(tower_bytes=None)["vision_transient"]["mib"])
        self.assertEqual(self.sec(tower_bytes=None)["verdict"]["stage"], "ungeprueft")

    def test_codes_are_the_runtimes(self):
        from flliper.srt.pdflip import vision_victim as vv

        self.assertEqual((VV.W_SHORT, VV.W_NOT_RESTORED, VV.W_PLAN_REFUSED), (vv.W_VICTIM_SHORT, vv.W_VICTIM_NOT_RESTORED, vv.W_VICTIM_PLAN_REFUSED))
        self.assertEqual(VV.PLACE_ENV, launcher.VISION_PLACE_ENV)
        self.assertEqual(VV.PLACE_WEIGHTS, "weights")
        self.assertIn(VV.PLACE_WEIGHTS, launcher.VISION_PLACES)

    def test_victim_kind_follows_the_runtime(self):
        from flliper.srt.pdflip import vision_victim_27b as v27

        self.assertEqual(VV.victim_kind(is_moe=False, dual=False), v27.KIND_DENSE)
        self.assertEqual(VV.victim_kind(is_moe=False, dual=True), v27.KIND_PP_ONLY)
        self.assertEqual(VV.victim_kind(is_moe=True, dual=False), "experts")

    def test_available_dense_uses_the_mlp_bytes_of_stage_0(self):
        m = {"weights": {"per_role_bytes": {"v": {"mlp": 64 * 100 * MIB}, "src": "Index"}}, "arch": {"n_layers": {"v": 64}}}
        mib, src = VV.available_dense_mib(m, 40)
        self.assertAlmostEqual(mib, 4000.0, places=3)
        self.assertIn("approximation", src)
        self.assertIsNone(VV.available_dense_mib({"weights": {}}, 40)[0])
        self.assertIsNone(VV.available_dense_mib(m, None)[0])

    def test_available_experts(self):
        mib, _ = VV.available_experts_mib([100.0] * 10, 4, 0.5)
        self.assertEqual(mib, 200.0)
        self.assertIsNone(VV.available_experts_mib([], 4, 0.5)[0])


# ---------------------------------------------------------------------------
# propose() and the verdicts
# ---------------------------------------------------------------------------
@_NEEDS_27B
class TestProposal(unittest.TestCase):
    def prop(self, place, **goals):
        return P.propose(_ref_rows(), _STATE["modell"], "dual", goals, basis=_with_place(_profile(), place), draft=_STATE["draft"], rates=RATES)

    def test_default_profile_is_byte_equal_and_carries_no_new_item(self):
        a, b = self.prop(None), self.prop("auto")
        self.assertEqual(a["vision"], {"schema": VV.SCHEMA, "aktiv": False, "vision": "transient", "place": ""})
        self.assertFalse(b["vision"]["aktiv"])
        a_c = {k: v for k, v in a.items() if k not in ("vision", "argv")}
        b_c = {k: v for k, v in b.items() if k not in ("vision", "argv")}
        self.assertEqual(a_c, b_c)                               # the place 'auto' changes nothing in the proposal (values, fit, dual, hints)
        self.assertEqual(a["env"], b["env"])
        self.assertEqual(self.prop(None)["argv"], a["argv"])
        self.assertNotIn("Vision on displaced weights", " ".join(a["dual"]["annahmen"]))

    def test_weights_adds_the_section_and_nothing_else_to_the_values(self):
        a, w = self.prop(None), self.prop("weights")
        self.assertEqual(w["argv"], a["argv"] + ["--pdflip-vision-place", "weights"])      # the flag is carried unchanged, the planner decides nothing
        self.assertEqual([x for x in w["values"] if x["key"] != "--pdflip-vision-place"], [x for x in a["values"] if x["key"] != "--pdflip-vision-place"])
        v = w["vision"]
        self.assertTrue(v["aktiv"])
        self.assertEqual((v["opferart"], v["turm_mib"], v["resident_mib"], v["host_mib"]), ("pp_only", 878.8, 0.0, 878.8))
        self.assertEqual(v["vision_transient"]["mib"], 878.8)
        self.assertEqual(v["verdict"]["stage"], "ja")
        self.assertEqual(v["split_tensors_erwartet"], 2)
        k0 = w["dual"]["karten"][0]
        self.assertEqual(v["verfuegbar_mib"], k0["p_privat_mib"])            # the pp_only of the stage 0 card of the Dual plan
        # the sum of the weights does not grow: P private weights / budgets / Dual fit are the ones without the item
        self.assertEqual(w["dual"]["karten"], a["dual"]["karten"])
        self.assertEqual(w["dual"]["fit"], a["dual"]["fit"])
        self.assertTrue(any("Vision on displaced weights" in x for x in w["dual"]["annahmen"]))

    def test_a_cut_with_a_tiny_stage_0_gives_the_w105b_verdict_with_numbers(self):
        w = self.prop("weights", dual_cut=[2, 31, 31])
        v = w["vision"]["verdict"]
        self.assertEqual((v["code"], v["stage"]), ("W105b", "nein"))
        self.assertLess(v["verfuegbar_mib"], 878.8)
        self.assertAlmostEqual(v["missing_mib"], round(878.8 - v["verfuegbar_mib"], 1), places=1)
        self.assertTrue(any("W105b" in h for h in w["notes"]))
        vd = PV.build_verdict(3, O_clean(), None, proposal=w)
        by = {x["code"]: x for x in vd["verdikte"]}
        self.assertIn("W105b", by)
        self.assertEqual(by["W105b"]["force_state"], PV.HINT)                 # a verdict with code and number, NEVER a lock
        self.assertEqual(by["W105b"]["stage"], "nein")
        self.assertEqual(by["W105b"]["missing_mib"], v["missing_mib"])
        self.assertIsNone(by["W105b"]["forcebar"])
        self.assertEqual(vd["outcome"], "geht")                              # the launcher run was clean: the verdict does not turn it into a refusal
        self.assertFalse([x for x in vd["verdikte"] if x["level"] in ("run", "crash")])

    def test_ok_case_is_a_vision_victim_hint_not_a_w105b(self):
        w = self.prop("weights")
        vd = PV.build_verdict(3, O_clean(), None, proposal=w)
        codes = [x["code"] for x in vd["verdikte"]]
        self.assertIn("VISION-VICTIM", codes)
        self.assertNotIn("W105b", codes)
        self.assertEqual([x for x in vd["verdikte"] if x["code"] == "VISION-VICTIM"][0]["force_state"], PV.GOES)

    def test_default_proposal_has_no_vision_verdict(self):
        vd = PV.build_verdict(3, O_clean(), None, proposal=self.prop(None))
        self.assertFalse({"W105b", "VISION-VICTIM"} & {x["code"] for x in vd["verdikte"]})

    def test_the_section_is_json(self):
        json.dumps(self.prop("weights")["vision"])


def O_clean():
    return types.SimpleNamespace(rc=0, exc_type=None, exc_msg="", exc_where="", exc_mro=(), forced=[], text="", raw="", argv=[], dump=lambda: "")


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------
class TestBars(unittest.TestCase):
    FX = os.path.join(HERE, "fixtures", "profil_s3_1003")
    RIG3 = [("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)]
    NF = {"--pp-stage-ratio": "29,11,8", "--pp-attn-stage-ratio": "7,3,2", "--pp-cut-expert-device-fraction": "0.330,0.701,0.652",
          "--pp-cut-expert-lru-rows": "32,32,32", "--max-kv-per-request": "262144", "--pdflip-vision": "transient"}

    def bars(self, place, model="nextflash_int4mixed", args=None, form="flip"):
        m = MP.estimate(os.path.join(self.FX, model))
        a = dict(args or self.NF)
        if place:
            a["--pdflip-vision-place"] = place
        return PC.phase_bars(PC.synthetic_hardware(self.RIG3), m, a, {"P": {}, "D": {}}, {}, form=form)["phases"]["P"]["bars"]

    def test_default_bar_has_not_one_new_key(self):
        for place in (None, "auto", "kvtail"):
            b = self.bars(place)[0]
            self.assertNotIn("host_zeilen", b)
            self.assertNotIn("unterposten_ohne_ziel", b)
            self.assertTrue(all("unterposten" not in s for s in b["segments"]))

    def test_weights_place_gives_the_subitem_and_the_host_row_and_changes_no_sum(self):
        d, w = self.bars(None)[0], self.bars("weights")[0]
        self.assertEqual({k: v for k, v in w.items() if k != "host_zeilen"}, {k: v for k, v in d.items()} | {"segments": w["segments"]})   # same bar data
        self.assertEqual(w["posts_mib"], d["posts_mib"])
        self.assertEqual([(s["name"], s["mib"]) for s in w["segments"]], [(s["name"], s["mib"]) for s in d["segments"]])
        seg = next(s for s in w["segments"] if s["name"] == "weights")
        (sub,) = seg["unterposten"]
        self.assertEqual((sub["name"], sub["label"], sub["mib"], sub["resident_mib"], sub["in_summe"]), ("vision_transient", "Vision transient", 856.3, 0.0, False))
        self.assertEqual(sub["opferart"], "experts")
        self.assertEqual(sub["verdict"]["stage"], "ja")
        (host,) = w["host_zeilen"]
        self.assertEqual((host["name"], host["mib"], host["nach_rueckholung_mib"]), ("vision_victim_host", 0.0, 0.0))     # NF: no extra host RAM
        # the later stages carry nothing (the tower sits on PP0)
        for b in self.bars("weights")[1:]:
            self.assertNotIn("host_zeilen", b)
        json.dumps(w)

    def test_tp_form_and_single_card_have_no_vision_item(self):
        res = PC.phase_bars(PC.synthetic_hardware(self.RIG3), MP.estimate(os.path.join(self.FX, "nextflash_int4mixed")),
                            dict(self.NF, **{"--pdflip-vision-place": "weights"}), {"P": {}, "D": {}}, {}, form="d_only")
        self.assertEqual(list(res["phases"]), ["D"])
        for b in res["phases"]["D"]["bars"]:
            self.assertNotIn("host_zeilen", b)

    def test_vision_off_or_resident_is_no_item(self):
        for v in ("off", "resident"):
            a = dict(self.NF, **{"--pdflip-vision": v, "--pdflip-vision-place": "weights"})
            self.assertNotIn("host_zeilen", self.bars("weights", args=a)[0])


# ---------------------------------------------------------------------------
# launcher verdicts: each refusal red -> green
# ---------------------------------------------------------------------------
class TestLauncherVerdicts(unittest.TestCase):
    def ns(self, **kw):
        a = dict(dual_layout=True, pdflip_vision="transient", pdflip_vision_place="kvtail", env_p="", env_d="")
        a.update(kw)
        return types.SimpleNamespace(**a)

    def test_kvtail_and_auto_told_with_dual_layout_are_refused_by_name(self):
        for place in ("kvtail", "auto"):
            line = launcher.vision_place_dual_refusal(self.ns(pdflip_vision_place=place), ["--dual-layout", "--pdflip-vision-place", place])
            self.assertTrue(line.startswith("W111 PdFlipVisionArmRefused"), place)
            self.assertIn(f"--pdflip-vision-place {place}", line)
            self.assertIn("--dual-layout", line)
            self.assertIn("weights", line)
        eq = launcher.vision_place_dual_refusal(self.ns(), ["--pdflip-vision-place=kvtail"])
        self.assertTrue(eq and "W111" in eq)                                        # the argparse spelling counts too

    def test_everything_else_passes(self):
        told = ["--pdflip-vision-place", "kvtail"]
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision_place="weights"), ["--pdflip-vision-place", "weights"]))
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision_place="free"), ["--pdflip-vision-place", "free"]))
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(dual_layout=False), told))            # flip form: kvtail is the proven stage
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision="off"), told))
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision="resident"), told))
        # the DEFAULT (place not told) keeps the old path byte for byte: the release dual profile names no place
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision_place="auto"), ["--dual-layout"]))
        self.assertIsNone(launcher.vision_place_dual_refusal(self.ns(pdflip_vision_place="kvtail"), []))
        self.assertIsNone(launcher.vision_place_dual_refusal(types.SimpleNamespace(), ["--pdflip-vision-place", "kvtail"]))

    def test_the_refusal_is_wired_after_the_teardown_branch(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertIn("vision_place_dual_refusal(ns,", src)
        self.assertIn("raise SystemExit(_vis_place_dual)", src)
        self.assertLess(src.index("return teardown(ns.teardown)"), src.index("vision_place_dual_refusal(ns,"))
        self.assertLess(src.index("resolve_dual_layout(ns)"), src.index("vision_place_dual_refusal(ns,"))      # --dual-share implies --dual-layout first

    def test_the_three_places_still_are_the_launchers_choices(self):
        self.assertEqual(launcher.VISION_PLACES_BROKEN_IN_DUAL, ("kvtail", "auto"))
        self.assertTrue(set(launcher.VISION_PLACES_BROKEN_IN_DUAL) <= set(launcher.VISION_PLACES))

    # --- the VISION-SYNC LAW, ported from the NF line (c651892375) -------------------------------------------------------
    def test_vision_async_is_refused_at_launch(self):
        cases = [("os", "FLLIPER_PDFLIP_VISION_ASYNC", "1", True), ("os", "FLLIPER_PDFLIP_P_ROW_VISION_ASYNC", "1", True),
                 ("p", "FLLIPER_PDFLIP_VISION_ASYNC", "true", True), ("d", "FLLIPER_PDFLIP_P_ROW_VISION_ASYNC", "on", True),
                 ("os", "FLLIPER_PDFLIP_VISION_ASYNC", "0", False), ("p", "FLLIPER_PDFLIP_VISION_ASYNC", "0", False), (None, None, None, False)]
        for where, name, val, refused in cases:
            env, env_p, env_d = {}, "", ""
            if where == "os":
                env[name] = val
            elif where == "p":
                env_p = f"{name}={val}"
            elif where == "d":
                env_d = f"{name}={val}"
            line = launcher.vision_async_refusal(types.SimpleNamespace(env_p=env_p, env_d=env_d), environ=env)
            if refused:
                self.assertTrue(line.startswith("PDFLIP VISION-ASYNC refused") and name in line and "VISION-SYNC LAW" in line, (where, name, val))
            else:
                self.assertIsNone(line, (where, name, val))

    def test_vision_async_refusal_is_wired_into_main(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertIn("vision_async_refusal(ns)", src)
        self.assertIn("raise SystemExit(_vis_async)", src)

    def test_the_refusal_text_does_not_send_the_27b_operator_to_the_async_default(self):
        """Review 2: on the 27B line an UNSET FLLIPER_PDFLIP_VISION_ASYNC is the async stage (vision_rank_runner.vision_async_on, default ON), so the text
        must say 'set it 0', never 'unset it'."""
        from flliper.srt.pdflip import vision_rank_runner as vrr

        self.assertTrue(vrr.vision_async_on({}))                                    # the premise: unset = async on the 27B line
        line = launcher.vision_async_refusal(types.SimpleNamespace(env_p="", env_d=""), environ={"FLLIPER_PDFLIP_VISION_ASYNC": "1"})
        self.assertIn("Set it 0", line)
        self.assertNotIn("Unset it", line)
        self.assertNotIn("the code default is the synchronous", line)
        self.assertIn("UNSET FLLIPER_PDFLIP_VISION_ASYNC is the async stage", line)

    def test_the_unset_switch_is_the_old_default_path(self):
        """The 27B runner's code default is the async stage (vision_rank_runner.vision_async_on); the refusal fires on an EXPLICIT on only."""
        self.assertIsNone(launcher.vision_async_refusal(types.SimpleNamespace(env_p="", env_d=""), environ={}))

    # --- the host term ------------------------------------------------------------------------------------------------------
    def host_ns(self, **kw):
        a = dict(pdflip_vision="transient", pdflip_vision_place="weights", dual_layout=False, model=NVFP4,
                 pdflip_boot_form=types.SimpleNamespace(arch="dense"))
        a.update(kw)
        return types.SimpleNamespace(**a)

    def test_host_term_is_zero_without_the_weights_place(self):
        for place in ("auto", "kvtail", "free"):
            mib, prov = launcher.vision_victim_host_term(self.host_ns(pdflip_vision_place=place))
            self.assertEqual(mib, 0.0)
            self.assertTrue(prov.startswith("none (--pdflip-vision-place"))
        self.assertEqual(launcher.vision_victim_host_term(self.host_ns(pdflip_vision="off"))[0], 0.0)

    def test_host_term_nf_is_zero_experts_have_their_copy_in_the_store(self):
        mib, prov = launcher.vision_victim_host_term(self.host_ns(pdflip_boot_form=types.SimpleNamespace(arch="moe")))
        self.assertEqual(mib, 0.0)
        self.assertIn("expert store", prov)

    @unittest.skipUnless(os.path.exists(NVFP4 + "/model.safetensors.index.json"), "27B NVFP4 checkpoint not on this box")
    def test_host_term_27b_is_the_tower_bytes_dense_and_pp_only(self):
        for dual, kind in ((False, "dense"), (True, "pp_only")):
            mib, prov = launcher.vision_victim_host_term(self.host_ns(dual_layout=dual))
            self.assertAlmostEqual(mib, VV.TOWER_BYTES_27B / MIB, places=6)
            self.assertIn(f"victim={kind}", prov)
            self.assertIn("transient", prov)

    def test_unreadable_checkpoint_is_unmeasured_zero_never_a_refusal(self):
        mib, prov = launcher.vision_victim_host_term(self.host_ns(model="/nonexistent-ap4-dir"))
        self.assertEqual(mib, 0.0)
        self.assertTrue(prov.startswith("UNMEASURED"))

    def test_the_host_post_reaches_the_ledger_call(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertIn("vision_victim_host_gib=vision_victim_host_mib / 1024.0", src)
        self.assertIn("vision_victim_host_gib", inspect.signature(launcher.choose_host_ledger).parameters)
        self.assertEqual(inspect.signature(launcher.choose_host_ledger).parameters["vision_victim_host_gib"].default, 0.0)


# ---------------------------------------------------------------------------
# launcher dry runs (oracle): the default path prints nothing new, the new place prints the host warning
# ---------------------------------------------------------------------------
def _dry_possible():
    cen = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
    return (os.path.exists(cen) and os.path.exists(NVFP4 + "/config.json") and os.path.exists(DRAFT + "/config.json")
            and O.launcher_line() == O.LINE_27B)


@unittest.skipUnless(_dry_possible(), "27B launcher line + NVFP4 checkpoints + census needed (box-bound, like the AP0 goldens)")
class TestDryRuns(unittest.TestCase):
    #: the release line of the 27B base profile (user decision 2026-10-09); removed, the chain is the launcher's default path again
    PLACE_LINE = "PROFILE_ARGS+=(--pdflip-vision-place weights)\n"

    def run_dual(self, *extra, profiles=PROFILES):
        return O.run_profile(os.path.join(profiles, "27b-nvfp4-dual.env"), _ref_rows(), tree=str(launcher.__file__).split("/python/")[0],
                             extra_args=list(extra))

    def run_dual_default(self):
        """The Dual chain with the release line taken out of its base (copies in a temp dir): the place UNSET."""
        with tempfile.TemporaryDirectory(prefix="ap4-default-place-") as td:
            for n in ("27b-base.env", "27b-nvfp4-dual.env", "27b-nvfp4.pchunk.json"):
                with open(os.path.join(PROFILES, n), encoding="utf-8") as fh:
                    text = fh.read()
                if n == "27b-base.env":
                    self.assertEqual(text.count(self.PLACE_LINE), 1, "the release line of 27b-base.env moved")
                    text = text.replace(self.PLACE_LINE, "")
                with open(os.path.join(td, n), "w", encoding="utf-8") as fh:
                    fh.write(text)
            run = self.run_dual(profiles=td)
        self.assertNotIn("--pdflip-vision-place", run.argv)
        return run

    def test_dual_with_kvtail_told_is_refused_by_the_launcher_itself(self):
        r = self.run_dual("--pdflip-vision-place", "kvtail").result
        self.assertIsNotNone(r.exc_type)
        self.assertIn("W111 PdFlipVisionArmRefused", r.exc_msg)
        self.assertIn("--pdflip-vision-place kvtail", r.exc_msg)

    def test_dual_with_weights_runs_and_names_the_host_price(self):
        """The release Dual profile itself (no extra flag): it names ``weights`` once, via its base (user decision 2026-10-09)."""
        run = self.run_dual()
        self.assertEqual(run.argv.count("--pdflip-vision-place"), 1)
        self.assertEqual(run.argv[run.argv.index("--pdflip-vision-place") + 1], "weights")
        r = run.result
        self.assertIsNone(r.exc_type, "%s: %s" % (r.exc_type, r.exc_msg[:300]))
        self.assertEqual(r.rc, 0)
        host = [x for x in r.text.splitlines() if "PDFLIP-HOST vision_victim_host=" in x]
        self.assertEqual(len(host), 1, host)
        self.assertIn("879 MiB", host[0])
        self.assertIn("victim=pp_only", host[0])
        warn = [x for x in r.text.splitlines() if "VISION-VICTIM host-RAM WARNING" in x]
        self.assertEqual(len(warn), 1)
        self.assertIn("never a refusal", warn[0])
        self.assertEqual(r.forced, [])

    def test_nf_abl_with_weights_names_zero_host_and_warns_of_nothing(self):
        """NF: the victim rows have their copy in the expert store -> the HOST line says 0 MiB, and a warning about 0 MiB is not printed."""
        ck = os.path.join(FIX, "checkpoints")
        snaps = {}
        for n in sorted(os.listdir(ck)):
            if os.path.isfile(os.path.join(ck, n, "manifest.json")):
                snaps[O.read_snapshot_manifest(os.path.join(ck, n))["name"]] = os.path.join(ck, n)
        r = O.run_profile(os.path.join(PROFILES, "nf-int4-h6-abl.env"), _ref_rows(), tree=str(launcher.__file__).split("/python/")[0],
                          extra_args=["--pdflip-vision-place", "weights"], snapshots=snaps).result
        self.assertIsNone(r.exc_type, "%s: %s" % (r.exc_type, r.exc_msg[:300]))
        host = [x for x in r.text.splitlines() if "PDFLIP-HOST vision_victim_host=" in x]
        self.assertEqual(len(host), 1, host)
        self.assertIn("vision_victim_host=0 MiB", host[0])
        self.assertIn("expert store", host[0])
        self.assertFalse([x for x in r.text.splitlines() if "host-RAM WARNING" in x])

    def test_dual_default_prints_nothing_of_it(self):
        r = self.run_dual_default().result
        self.assertIsNone(r.exc_type, r.exc_msg[:200])
        self.assertNotIn("vision_victim", r.text)
        self.assertNotIn("VISION-VICTIM", r.text)


# ---------------------------------------------------------------------------
# host ledger
# ---------------------------------------------------------------------------
class TestHostLedgerPost(unittest.TestCase):
    def images(self):
        return host_ledger.ImageTerms(*[0.0] * 0) if False else None

    def terms(self, **kw):
        # the pure term builder needs only the image terms object; build the smallest one the ledger accepts
        im = host_ledger.ImageTerms.__new__(host_ledger.ImageTerms)
        for f in getattr(host_ledger.ImageTerms, "__slots__", ()) or getattr(host_ledger.ImageTerms, "__dataclass_fields__", {}):
            setattr(im, f, 0.0)
        return host_ledger.charge_terms(10, 150, 3, im, **kw)

    def test_default_key_is_present_and_zero(self):
        t = self.terms()
        self.assertEqual(t["vision_victim_host_gib"], 0.0)

    def test_the_post_is_in_no_sum(self):
        """Fix round 1: a warning post (host-RAM rule 06.10.) is named and printed, but it is in NEITHER sum -- it cannot move a fundability number."""
        base, withv = self.terms(), self.terms(vision_victim_host_gib=0.86)
        self.assertEqual(withv["vision_victim_host_gib"], 0.86)                                              # the post is named
        self.assertEqual(host_ledger._boot_charges_gib(withv), host_ledger._boot_charges_gib(base))          # launch moment / both-moment sum
        self.assertEqual(host_ledger._run_moment_charges_gib(withv), host_ledger._run_moment_charges_gib(base))   # run moment

    def test_a_huge_post_moves_no_arm_and_refuses_nothing(self):
        """Fix round 1 (review 1): price/choose with an absurd post (500 GiB) give the same run peak, leftover, arm and verdict lines as 0.0."""
        from unittest import mock

        GIB, MIB = int(host_ledger.GIB), 1 << 20
        kw = dict(ring_bytes=4096 * MIB, ring_span1_bytes=4096 * MIB, cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=84 * GIB)
        a0 = host_ledger.price(int(125.70 * GIB), int(101.96 * GIB), 1, 600, **kw)
        a1 = host_ledger.price(int(125.70 * GIB), int(101.96 * GIB), 1, 600, **kw, vision_victim_host_gib=500.0)
        self.assertEqual(a1.terms["vision_victim_host_gib"], 500.0)
        self.assertEqual(a1.predicted_run_peak_gib(), a0.predicted_run_peak_gib())
        self.assertEqual(a1.run_leftover_gib, a0.run_leftover_gib)
        live = {"current_gib": 1.09, "nonreclaim_gib": 0.79, "file_reclaimable_gib": 0.3, "source": "test", "max_gib": None}
        ck = dict(ring_bytes=4096 * MIB, ring_span1_bytes=4096 * MIB, ring_provenance="test", cg_current_bytes=int(1.09 * GIB),
                  cg_ceiling_bytes=84 * GIB, cg_ceiling_source="cgroup memory.max", cg_oom_kill=0)
        res = []
        for v in (0.0, 500.0):
            with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=live):
                arm, _h, lines = host_ledger.choose(int(125.70 * GIB), int(101.96 * GIB), **ck, vision_victim_host_gib=v)   # no W20/W21 raised
            res.append((arm.s_gb, arm.m_mib, [ln for ln in lines if "vision_victim_host" not in ln]))
        self.assertEqual(res[0], res[1])

    def test_a_negative_value_is_clamped(self):
        self.assertEqual(self.terms(vision_victim_host_gib=-1.0)["vision_victim_host_gib"], 0.0)

    def test_the_arm_line_names_the_post_only_when_it_is_charged(self):
        arm = types.SimpleNamespace(terms=dict(self.terms(), flip_ratchet_charged_gib=0.0, flip_ratchet_source="x", host_ring_gib=0.0))
        self.assertNotIn("vision_victim_host", host_ledger.arm_terms_line(arm))
        arm.terms["vision_victim_host_gib"] = 0.86
        self.assertIn("vision_victim_host=0.86(run)", host_ledger.arm_terms_line(arm))

    def test_price_and_choose_take_the_post_as_a_trailing_keyword(self):
        import inspect

        for fn in (host_ledger.charge_terms, host_ledger.price, host_ledger.choose):
            ps = list(inspect.signature(fn).parameters.values())
            self.assertEqual(ps[-1].name, "vision_victim_host_gib", fn.__name__)         # LAST: no positional caller shifts
            self.assertEqual(ps[-1].default, 0.0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
