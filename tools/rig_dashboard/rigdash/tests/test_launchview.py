"""Startflags + ENV oben im Dashboard (Nutzer 29.09. 11:50Z): "ich will oben im dashboard die
kompletten startflags des jeweiligen boots auch inkl. aller ENV die dafür gesetzt werden.
collapsable. weil es ziemlich viel ist. gestalte es übersichtlich".

Quelle nur state.json. Die Fixtures sind gekürzte Kopien der zwei state.json vom 29.09.:
  launch_state_27b.json  27bbf-boot-20260929T115200Z-ac03 (P argv 100 / env 145, D 104 / 138)
  launch_state_nf.json   nfh91dprsavisadoptstcutvsyncodx2bswre2cutz30x2dauer-boot-20260929T113026Z-d4b1
Gepinnt: die Zählwerte stimmen mit dem argv/env der Datei, Geheimnis-Werte sind maskiert (auch
wenn der Launcher sie durchließe), der P<->D-Vergleich teilt jedes Flag in genau eine Klasse.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import launchview as lv  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load(m):
    with open(os.path.join(FIX, "launch_state_%s.json" % m)) as fh:
        return json.load(fh)


def _flags_in(argv):
    """Unabhängig gezählt: Tokens mit '--' nach dem Kommando."""
    return sum(1 for a in argv if a.startswith("--"))


class CountsFromBothStateFiles(unittest.TestCase):
    def test_counts_match_argv_and_env_of_27b_and_nf(self):
        for m, name in (("27b", "27B"), ("nf", "NF")):
            st = _load(m)
            v = lv.model_view(name, st)
            self.assertEqual(v["boot_id"], st["boot_id"])
            self.assertEqual((v["image"], v["rev"], v["profile"], v["tag"]),
                             (st["image"], st["rev"], st["profile"], st["tag"]))
            for g in ("P", "D"):
                launch = st["groups"][g]["launch"]
                L = v["layers"][g]
                self.assertEqual(L["n_flags"], _flags_in(launch["argv"]), (m, g))
                self.assertEqual(sum(len(x["rows"]) for x in L["flags"]), _flags_in(launch["argv"]), (m, g))
                # every env key lands in exactly one of gesetzt / Basis
                self.assertEqual(L["n_env"] + L["n_env_base"], len(launch["env"]), (m, g))
                self.assertEqual(v["counts"][g], {"flags": L["n_flags"], "env": L["n_env"]})
            self.assertEqual(v["missing"], ["container", "front"])   # vor 814c6657bb / Host-Teil

    def test_27b_numbers_as_the_operator_quoted_them(self):
        st = _load("27b")
        self.assertEqual((len(st["groups"]["P"]["launch"]["argv"]), len(st["groups"]["P"]["launch"]["env"])), (100, 145))
        v = lv.model_view("27B", st)
        self.assertEqual(v["counts"]["P"], {"flags": 56, "env": 143})   # 2 Image-Werte (CUDA_HOME/VERSION) -> Basis
        self.assertEqual([k for k, _ in v["layers"]["P"]["env_base"]], ["CUDA_HOME", "CUDA_VERSION"])

    def test_flags_grouped_by_prefix_json_indented(self):
        v = lv.model_view("NF", _load("nf"))
        groups = [g["group"] for g in v["layers"]["D"]["flags"]]
        self.assertEqual(groups, [g for g in [n for n, _ in lv.FLAG_GROUPS] + [lv.FLAG_REST] if g in groups])
        for g in v["layers"]["D"]["flags"]:
            for f in g["rows"]:
                self.assertEqual(lv.flag_group(f["flag"]), g["group"])
        js = [f for g in v["layers"]["P"]["flags"] for f in g["rows"] if f["flag"] == "--json-model-override-args"]
        self.assertTrue(js and js[0]["json"].startswith("{\n"))


class Masking(unittest.TestCase):
    def test_secret_values_masked_paths_kept_counts_unchanged(self):
        st = copy.deepcopy(_load("nf"))
        p = st["groups"]["P"]["launch"]
        p["argv"] += ["--hf-token", "hf_SECRETVALUE", "--api-key=sk-live-xyz", "--admin-key-file", "/run/x.adminkey"]
        p["env"].update({"HF_TOKEN": "hf_abc", "GITHUB_PAT": "ghp_x", "DB_PASSWORD": "pw", "WEG2_SECRET_SALT": "s"})
        p["env_base"] = {"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-1"}
        st["launch"] = {"container": {"image": "htsglang:x", "image_id": "sha256:1", "rev": "abc", "profile": "p",
                                      "run": {"argv": ["docker", "run", "-e", "HF_TOKEN=hf_run", "-e", "SGLANG_X=1"],
                                              "env": {"HF_TOKEN": "hf_run", "SGLANG_X": "1"},
                                              "mounts": [{"src": "/l3/nf", "dst": "/var/lib/x", "mode": "rw"}]}}}
        v = lv.model_view("NF", st)
        dump = json.dumps(v)
        for secret in ("hf_SECRETVALUE", "sk-live-xyz", "hf_abc", "ghp_x", "\"pw\"", "sk-1", "hf_run", "\"s\""):
            self.assertNotIn(secret, dump, secret)
        rows = {f["flag"]: f["value"] for g in v["layers"]["P"]["flags"] for f in g["rows"]}
        self.assertEqual(rows["--hf-token"], lv.MASK)
        self.assertEqual(rows["--api-key"], lv.MASK)
        self.assertEqual(rows["--admin-key-file"], "/run/x.adminkey")        # Pfad ok, Inhalt nie
        d27 = {f["flag"]: f["value"] for g in lv.model_view("27B", _load("27b"))["layers"]["P"]["flags"] for f in g["rows"]}
        self.assertNotEqual(d27["--max-total-tokens"], lv.MASK)                 # TOKENS ist kein Geheimnis
        self.assertTrue(d27["--max-total-tokens"].isdigit())
        env = dict(v["layers"]["P"]["env_set"])
        self.assertEqual((env["HF_TOKEN"], env["GITHUB_PAT"], env["DB_PASSWORD"]), (lv.MASK,) * 3)
        base = dict(v["layers"]["P"]["env_base"])
        self.assertEqual((base["PATH"], base["OPENAI_API_KEY"]), ("/usr/bin", lv.MASK))
        c = v["layers"]["container"]
        self.assertIn("HF_TOKEN=***", c["argv_raw"])
        self.assertEqual(dict(c["env_set"])["HF_TOKEN"], lv.MASK)

    def test_not_secret_token_counts_and_path(self):
        for name in ("--max-total-tokens", "--speculative-num-draft-tokens", "PATH", "SGLANG_HICACHE_KEY_SCHEME",
                     "--tp-prefill-max-tokens", "CUDA_HOME"):
            self.assertFalse(lv.is_secret_name(name), name)
        for name in ("HF_TOKEN", "--hf-token", "--admin-api-key", "GITHUB_PAT", "DB_PASSWORD", "X_SECRET_Y", "OPENAI_API_KEY"):
            self.assertTrue(lv.is_secret_name(name), name)


class DiffPD(unittest.TestCase):
    def test_every_flag_in_exactly_one_class(self):
        for m in ("27b", "nf"):
            st = _load(m)
            v = lv.model_view(m, st)
            pmap = lv._flag_map(lv.parse_argv(st["groups"]["P"]["launch"]["argv"])[1])
            dmap = lv._flag_map(lv.parse_argv(st["groups"]["D"]["launch"]["argv"])[1])
            fd = v["pd"]["flags"]
            classes = [k for k, *_ in fd["same"]] + [k for k, *_ in fd["differ"]] \
                + [k for k, _ in fd["only_a"]] + [k for k, _ in fd["only_b"]]
            self.assertEqual(sorted(classes), sorted(set(pmap) | set(dmap)))
            self.assertEqual(len(classes), len(set(classes)))
            for k, a, b in fd["differ"]:
                self.assertNotEqual(a, b)
                self.assertEqual((pmap[k], dmap[k]), (a, b))
            for k, val in fd["same"]:
                self.assertEqual(pmap[k], dmap[k])
            self.assertTrue(all(k not in dmap for k, _ in fd["only_a"]))
            self.assertTrue(all(k not in pmap for k, _ in fd["only_b"]))

    def test_known_differences_of_the_27b_boot(self):
        fd = lv.model_view("27B", _load("27b"))["pd"]["flags"]
        differ = {k: (a, b) for k, a, b in fd["differ"]}
        self.assertEqual(differ["--chunked-prefill-size"], ("2048", "4096"))
        self.assertEqual(differ["--hicache-size"], ("1", "4"))

    def test_small_synthetic_diff(self):
        d = lv.diff({"--a": "1", "--b": "2", "--p": "x"}, {"--a": "1", "--b": "3", "--d": "y"})
        self.assertEqual(d, {"same": [["--a", "1"]], "differ": [["--b", "2", "3"]],
                             "only_a": [["--p", "x"]], "only_b": [["--d", "y"]]})


class SnapshotAndVersion(unittest.TestCase):
    def test_snapshot_reads_current_and_ver_changes_only_with_content(self):
        with tempfile.TemporaryDirectory() as root:
            roots = []
            for m in ("27b", "nf"):
                d = os.path.join(root, m, "b1")
                os.makedirs(d)
                with open(os.path.join(d, "state.json"), "w") as fh:
                    json.dump(_load(m), fh)
                os.symlink(d, os.path.join(root, m, "current"))
                roots.append((m.upper(), os.path.join(root, m)))
            a = lv.snapshot(tuple(roots))
            b = lv.snapshot(tuple(roots))
            self.assertEqual(a["ver"], b["ver"])
            self.assertEqual([x["model"] for x in a["models"]], ["27B", "NF"])
            st = _load("nf")
            st["seq"] = 999                                   # a heartbeat alone: same picture
            with open(os.path.join(root, "nf", "b1", "state.json"), "w") as fh:
                json.dump(st, fh)
            self.assertEqual(lv.snapshot(tuple(roots))["ver"], a["ver"])
            st["groups"]["D"]["launch"]["argv"].append("--new-flag")
            with open(os.path.join(root, "nf", "b1", "state.json"), "w") as fh:
                json.dump(st, fh)
            self.assertNotEqual(lv.snapshot(tuple(roots))["ver"], a["ver"])

    def test_front_layer_with_env_base(self):
        st = copy.deepcopy(_load("27b"))
        st["launch"] = {"front": {"argv": ["python", "-m", "sglang.srt.weg2.front", "--port", "30030"],
                                  "env": {"SGLANG_WEG2_FRONT_SPAN_INFLIGHT": "1"}, "env_base": {"PATH": "/bin"}}}
        v = lv.model_view("27B", st)
        f = v["layers"]["front"]
        self.assertEqual((f["n_flags"], f["n_env"], f["n_env_base"]), (1, 1, 1))
        self.assertEqual(v["missing"], ["container"])
        self.assertIn("--port 30030", f["argv_raw"])


if __name__ == "__main__":
    unittest.main()
