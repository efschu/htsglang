"""WEG2-FORM (24.09.): the boot's form axes -- ONE resolver, ONE line, ONE env,
ONE accessor -- and the calibration identity it keys.

User order (verbatim): "da muessen schalter rein, in den code, dense, moe,
vollstaendig im vram, mit offload - oder sowas aehnliches oder mehr oder
weniger (ergruende/begruende). sonst knallts doch an jeder stelle".
Inventory: FORM_AXES_INVENTORY.md (H1 #114 transient, H2 #1444 D residue,
H3 P-cut/depth logs, M1 host-ledger record -- all NF measurements a 27B boot
read because "newest" was the only key).

Pinned here:
  * both real forms resolve to the axes their arms mean (27B-DFLASH, NF-MTP);
  * without a --form-* flag the resolver is a PURE READER of ``ns`` -- the NF
    argv cannot move through it;
  * every contradiction is refused by name (W140), never guessed;
  * --form-draft / --form-p-draft drive the legacy flags only when those were
    not given, and refuse when they were and disagree;
  * build_env publishes SGLANG_WEG2_FORM from the resolved form and POPS an
    inherited one; the #114 transient reaches group P only for the
    checkpoint it was measured on;
  * the calibration sources (measured record, P logs) accept only samples of
    this boot's checkpoint;
  * MoE-only gates print ``SKIPPED (form ...)`` for a dense form and nothing
    new for a MoE form.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher

#: --dflash-produce-on-p (0ae5d7dead/853c448693) lives on the 27B line only; the NF boot tree
#: cherry-picks this file without it (NF operator 24.09.), so the two tests that pass the flag
#: are conditional on the parser literal, not on a branch name.
_HAS_DFLASH_PRODUCE_FLAG = '"--dflash-produce-on-p"' in open(launcher.__file__).read()
_needs_produce_flag = unittest.skipUnless(
    _HAS_DFLASH_PRODUCE_FLAG, "--dflash-produce-on-p not in this tree's launcher (27B line only)"
)

NF_MODEL_NAME = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
Q27_MODEL_NAME = "Qwen3.8-27B-INT8-gdncov-vocabembed"


def _mk_model(root: str, name: str, moe: bool) -> str:
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    cfg = {"architectures": ["X"], "model_type": "x", "text_config": {"model_type": "x_text"}}
    if moe:
        cfg["text_config"].update({"num_experts": 512, "num_experts_per_tok": 10})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return path


def _nf_words(model: str):
    extra_p = ('--max-total-tokens 262144 --page-size 64 --max-running-requests 1 '
               '--rank-moe-resident-fraction 0.35,0.6,0.6 --context-length 262144')
    extra_d = ('--max-total-tokens 262144 --page-size 64 --rank-role host,worker,worker '
               '--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 '
               '--rank-moe-resident-fraction 0.006,0.564,0.467 --speculative-algorithm NEXTN '
               '--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4')
    return [
        "--tree", "/tmp", "--tag", "fnFL2t", "--profile", "nextflash", "--model", model,
        "--flip-weights", "family", "--weg2-weight-source", "exchange",
        "--env-p", "SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.35,0.6,0.6;"
                   "SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/fnFL2",
        "--env-d", "SGLANG_MOE_SCRATCH_SLOTS=44,48,48;SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/fnFL2",
        "--draft-kv-on-p", "on", "--weg2-vision", "off",
        "--pp-cut-expert-device-fraction", "0.35,0.6,0.6", "--pp-cut-expert-lru-rows", "32,32,32",
        "--extra-p", extra_p, "--extra-d", extra_d,
    ]


def _q27_words(model: str):
    return [
        "--tree", "/tmp", "--tag", "weg2xsnT", "--model", model,
        "--weg2-weight-source", "exchange", "--weg2-vision", "transient",
        "--spec-form", "DFLASH", "--d-tp-objective", "maxkv", "--p-bs", "1",
        "--extra-p=--max-running-requests=2",
    ]


def _resolve(words):
    ns = launcher.build_parser().parse_args(words)
    form = F.resolve_form(ns, words, parse_group_env=launcher.parse_group_env,
                          shlex_split=shlex.split)
    return ns, form


class _Models(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        self.dense = _mk_model(self.root, Q27_MODEL_NAME, moe=False)
        self.moe = _mk_model(self.root, NF_MODEL_NAME, moe=True)

    def tearDown(self):
        self._td.cleanup()


class TestResolveRealForms(_Models):
    def test_27b_dflash_arm(self):
        ns, form = _resolve(_q27_words(self.dense))
        self.assertEqual(
            form.axes(),
            {"arch": "dense", "experts": "none", "draft": "dflash", "p_draft": "cold",
             "kv": "paged_dcp", "flip": "family", "vision": "transient"})
        self.assertEqual(form.profile, "qwen27b")
        self.assertEqual(form.model, Q27_MODEL_NAME)

    def test_nf_mtp_arm(self):
        ns, form = _resolve(_nf_words(self.moe))
        self.assertEqual(
            form.axes(),
            {"arch": "moe", "experts": "offload", "draft": "mtp", "p_draft": "compute",
             "kv": "qsa_forma", "flip": "family", "vision": "off"})
        self.assertEqual(form.model, NF_MODEL_NAME)

    def test_resolver_is_a_pure_reader_without_form_flags(self):
        """NF byte-identity: no --form-* flag -> ``ns`` is not touched, so every
        argv/env the launcher builds from it is the one it built before."""
        for words in (_nf_words(self.moe), _q27_words(self.dense)):
            ns = launcher.build_parser().parse_args(words)
            before = copy.deepcopy(vars(ns))
            F.resolve_form(ns, words, parse_group_env=launcher.parse_group_env,
                           shlex_split=shlex.split)
            self.assertEqual(vars(ns), before)

    def test_unreadable_checkpoint_keeps_profile_expectation(self):
        ns, form = _resolve(_q27_words(os.path.join(self.root, "missing")))
        self.assertEqual(form.arch, "dense")
        self.assertIn("unreadable", dict(form.sources)["arch"])

    def test_hicache_disabled_means_no_p_draft(self):
        ns, form = _resolve(_q27_words(self.dense) + ["--weg2-disable-hicache"])
        self.assertEqual(form.p_draft, "none")

    @_needs_produce_flag
    def test_produce_on_is_compute(self):
        ns, form = _resolve(_q27_words(self.dense) + ["--dflash-produce-on-p", "on"])
        self.assertEqual(form.p_draft, "compute")

    def test_moe_without_store_is_resident(self):
        ns, form = _resolve([
            "--tree", "/tmp", "--tag", "t", "--profile", "nextflash", "--model", self.moe,
            "--extra-d", "--rank-role host,worker,worker --rank-moe-ratio 1,1,1 "
                         "--rank-moe-resident-fraction 1.0,1.0,1.0"])
        self.assertEqual(form.experts, "resident")


class TestContradictions(_Models):
    def _refused(self, words, needle):
        with self.assertRaises(F.Weg2FormContradiction) as cm:
            _resolve(words)
        self.assertIn("W140 Weg2FormContradiction", str(cm.exception))
        self.assertIn(needle, str(cm.exception))

    def test_nextflash_profile_on_dense_checkpoint(self):
        words = [w if w != self.moe else self.dense for w in _nf_words(self.moe)]
        self._refused(words, "DENSE")

    def test_qwen27b_profile_on_moe_checkpoint(self):
        self._refused(_q27_words(self.moe), "axis arch")

    def test_expert_flags_on_dense(self):
        self._refused(_q27_words(self.dense) + ["--extra-d", "--rank-moe-ratio 1,1,1"],
                      "--rank-moe-ratio (extra_d)")

    def test_form_a_under_qwen27b(self):
        self._refused(_q27_words(self.dense) + ["--extra-d", "--rank-role host,worker,worker"],
                      "axis kv")

    def test_dflash_under_nextflash(self):
        self._refused(_nf_words(self.moe) + ["--spec-form", "DFLASH"], "axis draft")

    def test_form_draft_none(self):
        self._refused(_q27_words(self.dense) + ["--form-draft", "none"], "--form-draft none")

    def test_form_draft_against_explicit_spec_form(self):
        self._refused(_q27_words(self.dense) + ["--form-draft", "mtp"], "EXPLICIT --spec-form")

    def test_cold_needs_dflash(self):
        self._refused(["--tree", "/tmp", "--tag", "t", "--model", self.dense,
                       "--form-p-draft", "cold"], "needs --spec-form DFLASH")

    @_needs_produce_flag
    def test_form_p_draft_against_explicit_produce(self):
        self._refused(_q27_words(self.dense) + ["--dflash-produce-on-p", "off",
                                                "--form-p-draft", "compute"],
                      "EXPLICIT --dflash-produce-on-p")

    def test_stated_arch_against_checkpoint(self):
        self._refused(_q27_words(self.dense) + ["--form-arch", "moe"], "axis arch")

    def test_refusal_is_a_launcher_refusal(self):
        self.assertIn(F.Weg2FormContradiction, launcher.REFUSALS)


class TestStatedAxesDrive(_Models):
    def test_form_draft_drives_spec_form(self):
        words = ["--tree", "/tmp", "--tag", "t", "--model", self.dense, "--form-draft", "dflash"]
        ns, form = _resolve(words)
        self.assertEqual(ns.spec_form, "DFLASH")
        self.assertEqual(form.draft, "dflash")
        self.assertEqual(form.p_draft, "cold")  # the DFLASH standard form

    def test_form_p_draft_compute_drives_produce(self):
        ns, form = _resolve(_q27_words(self.dense) + ["--form-p-draft", "compute"])
        self.assertEqual(ns.dflash_produce_on_p, "on")
        self.assertEqual(form.p_draft, "compute")

    def test_form_p_draft_none_drives_draft_kv_on_p(self):
        ns, form = _resolve(_q27_words(self.dense) + ["--form-p-draft", "none"])
        self.assertEqual(ns.draft_kv_on_p, "off")
        self.assertEqual(form.p_draft, "none")

    def test_stated_kv_overrides_profile_expectation(self):
        words = [w for w in _nf_words(self.moe)]
        i = words.index("--extra-d")
        words[i + 1] = words[i + 1].replace("--rank-role host,worker,worker ", "")
        with self.assertRaises(F.Weg2FormContradiction):
            _resolve(words)
        ns, form = _resolve(words + ["--form-kv", "paged_dcp"])
        self.assertEqual(form.kv, "paged_dcp")


class TestEnvAndAccessor(_Models):
    def test_round_trip(self):
        ns, form = _resolve(_q27_words(self.dense))
        v = form.env_value()
        self.assertNotIn(";", v)
        self.assertNotIn(" ", v)
        back = F.parse_form(v)
        self.assertEqual(back, form)
        self.assertEqual(F.current_form({F.FORM_ENV: v}), form)
        self.assertIsNone(F.current_form({}))
        self.assertIsNone(F.parse_form("arch=dense"))
        self.assertIsNone(F.parse_form(v.replace("arch=dense", "arch=hybrid")))

    def test_line_names_every_axis_and_the_model(self):
        ns, form = _resolve(_q27_words(self.dense))
        line = form.line()
        self.assertTrue(line.startswith("WEG2-FORM arch=dense experts=none draft=dflash "
                                        "p_draft=cold kv=paged_dcp flip=family vision=transient"))
        self.assertIn("model=" + Q27_MODEL_NAME, line)

    def test_profiles_agree_with_launcher(self):
        self.assertEqual(set(launcher.PROFILES), set(F.PROFILE_EXPECT))

    def test_build_env_publishes_and_pops(self):
        ns, form = _resolve(_q27_words(self.dense))
        with mock.patch.dict(os.environ, {F.FORM_ENV: "stale"}, clear=False):
            os.environ.pop(launcher.P_PREFILL_TRANSIENT_ENV, None)
            env_none = launcher.build_env("/t", "/v", "0,1,2", "/s", False, "t", group="P")
            self.assertNotIn(F.FORM_ENV, env_none)
            # desk caller: #114 exactly as before
            self.assertIn(launcher.P_PREFILL_TRANSIENT_ENV, env_none)
            env_27 = launcher.build_env("/t", "/v", "0,1,2", "/s", False, "t", group="P",
                                        boot_form=form)
            self.assertEqual(env_27[F.FORM_ENV], form.env_value())
            self.assertNotIn(launcher.P_PREFILL_TRANSIENT_ENV, env_27)
            env_d = launcher.build_env("/t", "/v", "0,1,2", "/s", False, "t", group="D",
                                       boot_form=form)
            self.assertEqual(env_d[F.FORM_ENV], form.env_value())
            self.assertNotIn(launcher.P_PREFILL_TRANSIENT_ENV, env_d)

    def test_build_env_nf_keeps_transient(self):
        ns, form = _resolve(_nf_words(self.moe))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(launcher.P_PREFILL_TRANSIENT_ENV, None)
            env = launcher.build_env("/t", "/v", "0,1,2", "/s", False, "t", group="P",
                                     boot_form=form)
        want = ",".join("%.0f" % v for v in launcher.p_prefill_transient_vector_mib(
            launcher.P_CHUNKED_PREFILL_TOKENS))
        self.assertEqual(env[launcher.P_PREFILL_TRANSIENT_ENV], want)

    def test_operator_transient_still_wins_on_27b(self):
        ns, form = _resolve(_q27_words(self.dense))
        with mock.patch.dict(os.environ, {launcher.P_PREFILL_TRANSIENT_ENV: "1,2,3"}):
            env = launcher.build_env("/t", "/v", "0,1,2", "/s", False, "t", group="P",
                                     boot_form=form)
        self.assertEqual(env[launcher.P_PREFILL_TRANSIENT_ENV], "1,2,3")

    def test_transient_provenance(self):
        ns, q27 = _resolve(_q27_words(self.dense))
        ok, why = launcher.p_prefill_transient_for(q27)
        self.assertFalse(ok)
        self.assertIn("NOT published", why)
        ns, nf = _resolve(_nf_words(self.moe))
        self.assertTrue(launcher.p_prefill_transient_for(nf)[0])
        self.assertTrue(launcher.p_prefill_transient_for(None)[0])

    def test_env_knobs_carry_the_form(self):
        ns, form = _resolve(_q27_words(self.dense))
        ns.weg2_boot_form = form
        self.assertIs(launcher._env_knobs(ns)["boot_form"], form)
        ns2 = launcher.build_parser().parse_args(_q27_words(self.dense))
        self.assertIsNone(launcher._env_knobs(ns2)["boot_form"])


class TestGateDeclarations(_Models):
    def test_dense_skips_every_moe_gate(self):
        ns, form = _resolve(_q27_words(self.dense))
        for name in ("#106 WEG2-STORE-GEOMETRY", "#107 WEG2-EXPERT-MAP", "#134 WEG2-EXPERT-BAND",
                     "#140 PP-CUT FRACTION-SOLVE", "#145 D-RANK FRACTION-SOLVE", "H14 WAKE-CREDIT"):
            line = F.gate_skip_line(name, form)
            self.assertIsNotNone(line, name)
            self.assertIn("SKIPPED (form arch=dense", line)

    def test_moe_and_no_form_skip_nothing(self):
        ns, form = _resolve(_nf_words(self.moe))
        for name in F.FORM_GATES:
            self.assertIsNone(F.gate_skip_line(name, form))
            self.assertIsNone(F.gate_skip_line(name, None))

    def test_publish_expert_map_skips_for_dense(self):
        ns, form = _resolve(_q27_words(self.dense))
        ns.weg2_boot_form = form
        lines = []
        self.assertEqual(launcher.publish_expert_map(ns, self.dense, self.root, lines.append), "")
        self.assertEqual(len(lines), 1)
        self.assertIn("#107 WEG2-EXPERT-MAP SKIPPED (form arch=dense", lines[0])


class TestFirstUse(_Models):
    def test_dense_names_the_nf_line_switches(self):
        ns, form = _resolve(_q27_words(self.dense))
        line = F.first_use_line(form, {})
        self.assertIn("FIRST-USE", line)
        for k in F.NF_LINE_DEFAULT_ON:
            self.assertIn(k, line)
        # a switch the arm turned off is not a first use
        line2 = F.first_use_line(form, {"SGLANG_WEG2_SLEEP_RELEASE_LMEM": "0"})
        self.assertNotIn("SGLANG_WEG2_SLEEP_RELEASE_LMEM", line2)

    def test_moe_and_no_form_print_nothing(self):
        ns, form = _resolve(_nf_words(self.moe))
        self.assertIsNone(F.first_use_line(form, {}))
        self.assertIsNone(F.first_use_line(None, {}))

    def test_switches_exist_with_their_defaults(self):
        from sglang.srt.environ import envs
        for k, (default, _where, _off) in F.NF_LINE_DEFAULT_ON.items():
            got = getattr(envs, k).default
            self.assertEqual(str(int(got) if isinstance(got, bool) else got), default, k)


def _front_log(ev, tag, tip, stamp, model):
    path = os.path.join(ev, f"boot_weg2_{tag}_{tip}_{stamp}.front.log")
    with open(path, "w") as f:
        f.write(f"[t] WEG2-LAUNCH === WEG2 BOOT tag={tag}\n")
        f.write("[t] WEG2-LAUNCH WEG2-USER-RESERVE PROVENANCE: argv[0] -> ordinal=0\n")
        f.write(f"[t] WEG2-LAUNCH group P argv: /v/bin/python -m sglang.launch_server "
                f"--model-path {model} --speculative-draft-model-path /x/draft --tp-size 1\n")
    return path


def _p_log(ev, tag, tip, stamp, prefill=True):
    path = os.path.join(ev, f"boot_weg2_{tag}_{tip}_{stamp}.P.log")
    with open(path, "w") as f:
        if prefill:
            for _ in range(3):
                f.write("[2026-09-24 09:57:14 PP0] Prefill batch, #new-seq: 1, "
                        "#new-token: 4096, #cached-token: 0, token usage: 0.01\n")
    return path


class TestCalibrationIdentity(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ev = self._td.name
        _front_log(self.ev, "q27old", "aaaaaaaaaa", "0920_211224", "/m/" + Q27_MODEL_NAME)
        _front_log(self.ev, "nfnew", "bbbbbbbbbb", "0924_094612", "/m/" + NF_MODEL_NAME)
        self.p_old = _p_log(self.ev, "q27old", "aaaaaaaaaa", "0920_211224")
        self.p_new = _p_log(self.ev, "nfnew", "bbbbbbbbbb", "0924_094612")
        now = time.time()
        os.utime(self.p_old, (now - 100, now - 100))
        os.utime(self.p_new, (now, now))

    def tearDown(self):
        self._td.cleanup()

    def test_tag_and_log_model(self):
        self.assertEqual(F.boot_tag_model("q27old", self.ev), Q27_MODEL_NAME)
        self.assertEqual(F.boot_tag_model("nfnew", self.ev), NF_MODEL_NAME)
        self.assertIsNone(F.boot_tag_model("nosuch", self.ev))
        self.assertEqual(F.group_log_model(self.p_new), NF_MODEL_NAME)

    def test_prefill_census_log_takes_this_models_log(self):
        self.assertEqual(launcher.newest_prefill_census_log(self.ev), self.p_new)
        acc = F.same_model_log("/any/" + Q27_MODEL_NAME)
        self.assertEqual(launcher.newest_prefill_census_log(self.ev, accept=acc), self.p_old)
        acc_none = F.same_model_log("/any/OtherModel")
        self.assertIsNone(launcher.newest_prefill_census_log(self.ev, accept=acc_none))

    def test_calib_log_accept_of(self):
        ns = argparse.Namespace(model="/m/" + Q27_MODEL_NAME)
        self.assertIsNone(launcher.calib_log_accept_of(ns))
        ns.weg2_boot_form = object()
        self.assertTrue(launcher.calib_log_accept_of(ns)(self.p_old))
        self.assertFalse(launcher.calib_log_accept_of(ns)(self.p_new))

    def test_measured_record_accept(self):
        rec = os.path.join(self.ev, "rec.json")
        with open(rec, "w") as f:
            json.dump({"samples": [
                {"group": "D", "boot_tag": "q27old", "at": "2026-09-20T21:14:45Z",
                 "rss_shmem_gib": 1.0, "vram_residue_mib": {"u": 1496}},
                {"group": "D", "boot_tag": "nfnew", "at": "2026-09-24T09:51:52Z",
                 "rss_shmem_gib": 1.0, "vram_residue_mib": {"u": 768}},
                {"group": "D", "boot_tag": "gone", "at": "2026-09-25T00:00:00Z",
                 "rss_shmem_gib": 1.0},
            ]}, f)
        self.assertEqual(host_ledger.read_measured_record(rec)["D"]["boot_tag"], "gone")
        acc = F.same_model_sample("/x/" + Q27_MODEL_NAME, self.ev)
        self.assertEqual(host_ledger.read_measured_record(rec, accept=acc)["D"]["boot_tag"], "q27old")
        acc_nf = F.same_model_sample("/x/" + NF_MODEL_NAME, self.ev)
        self.assertEqual(host_ledger.read_measured_record(rec, accept=acc_nf)["D"]["boot_tag"], "nfnew")

    def test_measured_record_accept_keys_the_residue_form(self):
        """Same checkpoint, other draft: not this form's residue. A pre-form
        boot is judged by the drafter its argv names; a boot with a WEG2-FORM
        line by every residue axis."""
        with open(os.path.join(self.ev, "boot_weg2_q27nextn_cccccccccc_0921_000000.front.log"), "w") as f:
            f.write(f"[t] WEG2-LAUNCH group P argv: /v/bin/python -m sglang.launch_server "
                    f"--model-path /m/{Q27_MODEL_NAME} --speculative-algorithm NEXTN\n")
        dflash = F.Weg2Form(arch="dense", experts="none", draft="dflash", p_draft="cold",
                            kv="paged_dcp", flip="family", vision="transient",
                            profile="qwen27b", model=Q27_MODEL_NAME)
        with open(os.path.join(self.ev, "boot_weg2_q27formed_dddddddddd_0922_000000.front.log"), "w") as f:
            f.write("[t] WEG2-LAUNCH tree: /t @ dddddddddd\n")
            f.write("[t] WEG2-LAUNCH " + dflash.line() + "\n")
        mtp_line = F.Weg2Form(**{**dflash.axes(), "draft": "mtp"}, profile="qwen27b",
                              model=Q27_MODEL_NAME).line()
        with open(os.path.join(self.ev, "boot_weg2_q27formmtp_eeeeeeeeee_0923_000000.front.log"), "w") as f:
            f.write("[t] WEG2-LAUNCH " + mtp_line + "\n")
        acc = F.same_model_sample("/x/" + Q27_MODEL_NAME, self.ev, dflash)
        self.assertTrue(acc({"boot_tag": "q27old"}))        # pre-form, no drafter named
        self.assertFalse(acc({"boot_tag": "q27nextn"}))     # pre-form, NEXTN
        self.assertTrue(acc({"boot_tag": "q27formed"}))     # same form line
        self.assertFalse(acc({"boot_tag": "q27formmtp"}))   # form line, other draft
        self.assertFalse(acc({"boot_tag": "nfnew"}))        # other checkpoint
        self.assertEqual(F.boot_tag_identity("q27formed", self.ev).form, dflash)


if __name__ == "__main__":
    unittest.main()
