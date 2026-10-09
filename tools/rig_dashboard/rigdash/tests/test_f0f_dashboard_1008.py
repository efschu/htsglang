"""F0-F (rename 08.10.): the dashboard in the renamed tree reads both generations and installs under the new names.

* metrics (F0-M: the names are must-keep, the sampler pushes the old prefix): every PromQL the dashboard sends and every Grafana
  panel reads BOTH the old prefix and the interim ``pdflip`` prefix (``vmpush.dual_promql``), whichever the query text uses.
* deploy: ``install.sh`` picks the state dir (/var/lib/flliper, fallback /var/lib/rigdash with the pre-rename history),
  the unit is written for /var/lib/flliper, install_510.sh no longer pins pre-rename commits.
* container names of either product name; issue-text templates in the new product name.
* residue: no old env prefix / package path in the dashboard package outside the locked boot recordings.

The old subsystem name is taken from ``names.STEM_TOKENS`` (split there: the rename tool must not rewrite it here).
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../rigdash
DASH = os.path.dirname(PKG)                       # .../tools/rig_dashboard
sys.path.insert(0, DASH)

from rigdash import hwprofil as H  # noqa: E402
from rigdash import names as N  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import server as S  # noqa: E402
from rigdash import vmpush as V  # noqa: E402

OLD, NEW = N.STEM_TOKENS
DEPLOY = os.path.join(PKG, "deploy")


class DualPromql(unittest.TestCase):
    def test_bare_metric_reads_both_prefixes(self):
        self.assertEqual(V.dual_promql(NEW + "_front_queue"), '{__name__=~"(%s|%s)_front_queue"}' % (OLD, NEW))

    def test_old_prefix_reads_both_too(self):
        """F0-M: the names are must-keep, a query in the old spelling gets the same dual form (the same series set)."""
        self.assertEqual(V.dual_promql(OLD + "_front_queue"), V.dual_promql(NEW + "_front_queue"))
        q = 'max by (model) (%s_rank_kv_usage_ratio{model=~"$model"}) * 100' % OLD
        self.assertEqual(V.dual_promql(q), 'max by (model) ({__name__=~"(%s|%s)_rank_kv_usage_ratio",model=~"$model"}) * 100' % (OLD, NEW))

    def test_labels_are_kept_and_selectors_inside_functions_work(self):
        q = V.dual_promql('sum by (model) (increase(%s_front_ttft_ms_sum{model="NF",boot=~"b.*"}[5m]))' % NEW)
        self.assertEqual(q, 'sum by (model) (increase({__name__=~"(%s|%s)_front_ttft_ms_sum",model="NF",boot=~"b.*"}[5m]))' % (OLD, NEW))

    def test_range_vector_selector_of_raw(self):
        self.assertEqual(V.dual_promql('%s_rank_x{boot="b"}[43200s]' % NEW), '{__name__=~"(%s|%s)_rank_x",boot="b"}[43200s]' % (OLD, NEW))

    def test_other_metrics_and_idempotence(self):
        for q in ("nvidia_smi_power_draw_watts", "sum(node_cpu_seconds_total)", 'up{job="x"}'):
            self.assertEqual(V.dual_promql(q), q)
        q = 'max by (model) (%s_flip_user_view_ms{part="total"}) / 1000' % NEW
        once = V.dual_promql(q)
        self.assertEqual(V.dual_promql(once), once)

    def test_a_name_that_only_ends_in_the_prefix_is_not_touched(self):
        self.assertEqual(V.dual_promql("my_" + NEW + "_x"), "my_" + NEW + "_x")

    def test_every_vm_client_call_sends_the_dual_form(self):
        seen = []

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(url, timeout=0):
            seen.append(url)
            return R(json.dumps({"status": "success", "data": {"result": []}}).encode())

        c = V.VmClient("http://x")
        with mock.patch("urllib.request.urlopen", fake):
            c.query(NEW + "_front_queue")
            c.query_range("sum(%s_front_queue)" % NEW, 0, 10, 5)
            c.query_range_by("sum by (model) (%s_front_queue)" % NEW, 0, 10, 5, "model")
            c.raw(NEW + "_rank_x", 60)
        self.assertEqual(len(seen), 4)
        from urllib.parse import unquote_plus
        for u in seen:
            self.assertIn("(%s|%s)_" % (OLD, NEW), unquote_plus(u))
            self.assertNotRegex(unquote_plus(u), r"(?<![A-Za-z0-9_])%s_[a-z]" % NEW)


class Grafana(unittest.TestCase):
    def _exprs(self, doc):
        return [t["expr"] for p in doc["panels"] for t in p.get("targets", [])]

    def test_shipped_dashboard_reads_both_generations(self):
        with open(os.path.join(DEPLOY, "grafana", "dashboards", "rig-verlauf.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        ex = self._exprs(doc)
        fam = [e for e in ex if "__name__" in e]
        self.assertGreaterEqual(len(fam), 10)
        for e in ex:
            self.assertNotRegex(e, r"(?<![A-Za-z0-9_])%s_[a-z]" % NEW, e)        # no bare renamed metric left
        for e in fam:
            self.assertIn("(%s|%s)_" % (OLD, NEW), e)

    def test_generator_writes_the_dual_form_too(self):
        r = subprocess.run([sys.executable, os.path.join(DEPLOY, "grafana", "make_dashboard.py")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        ex = self._exprs(json.loads(r.stdout))
        self.assertTrue(ex)
        for e in ex:
            self.assertNotRegex(e, r"(?<![A-Za-z0-9_])%s_[a-z]" % NEW, e)


class StateDir(unittest.TestCase):
    """install.sh ``pick_state_dir`` (the block between BEGIN/END state-dir), run in bash against a fake /var/lib."""

    def _pick(self, varlib, **env):
        script = os.path.join(DEPLOY, "install.sh")
        src = open(script, encoding="utf-8").read()
        m = re.search(r"# BEGIN state-dir\n(.*?)# END state-dir", src, re.S)
        self.assertTrue(m, "state-dir block missing in install.sh")
        e = dict(os.environ, RIGDASH_VARLIB=varlib)
        e.pop("RIGDASH_STATE_DIR", None)
        e.update(env)
        r = subprocess.run(["bash", "-c", m.group(1) + "\npick_state_dir"], capture_output=True, text=True, env=e, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def _touch(self, base, *parts):
        p = os.path.join(base, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()

    def test_fresh_machine_uses_the_new_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._pick(d), d + "/flliper")

    def test_pre_rename_history_keeps_its_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "rigdash", "history.sqlite")
            self._touch(d, "flliper", "hardware.json")          # hardware.json alone is no dashboard history
            self.assertEqual(self._pick(d), d + "/rigdash")

    def test_history_in_both_goes_to_the_new_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "rigdash", "history.sqlite")
            self._touch(d, "flliper", "history.sqlite")
            self.assertEqual(self._pick(d), d + "/flliper")

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "rigdash", "history.sqlite")
            self.assertEqual(self._pick(d, RIGDASH_STATE_DIR="/srv/state"), "/srv/state")

    def test_the_unit_is_written_for_the_new_directory_and_the_substitution_is_consistent(self):
        unit = open(os.path.join(DEPLOY, "rig-dashboard.service"), encoding="utf-8").read()
        self.assertIn("--state-dir /var/lib/flliper", unit)
        self.assertIn("StateDirectory=flliper\n", unit)
        self.assertIn("--port 8890", unit)                       # the port does not change
        inst = open(os.path.join(DEPLOY, "install.sh"), encoding="utf-8").read()
        sed = re.search(r'sed -e "s#--state-dir /var/lib/flliper#--state-dir \$state_dir#" \\\n\s+-e "(s#.*?)" \\\n', inst)
        self.assertTrue(sed, "install.sh substitutes --state-dir and StateDirectory=")
        r = subprocess.run(["bash", "-c", 'state_dir=/var/lib/rigdash; sed -e "s#--state-dir /var/lib/flliper#--state-dir $state_dir#" '
                                         '-e "s#^StateDirectory=flliper\\$#StateDirectory=${state_dir#/var/lib/}#" "$1"', "_",
                            os.path.join(DEPLOY, "rig-dashboard.service")], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--state-dir /var/lib/rigdash ", r.stdout)
        self.assertIn("StateDirectory=rigdash\n", r.stdout)
        live = [l for l in r.stdout.splitlines() if l.startswith(("ExecStart=", "StateDirectory="))]
        self.assertEqual(len(live), 2)
        self.assertNotIn("/var/lib/flliper", "\n".join(live))

    def test_scripts_parse(self):
        for f in ("install.sh", "install_510.sh", "stage_hwprofil.sh", "stage_profil_modules.sh"):
            r = subprocess.run(["bash", "-n", os.path.join(DEPLOY, f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f + ": " + r.stderr)

    def test_install_510_pins_no_pre_rename_commit(self):
        src = open(os.path.join(DEPLOY, "install_510.sh"), encoding="utf-8").read()
        for var in ("GATE_REV", "PROFIL_REV", "MODELLPROFIL_REV"):
            self.assertRegex(src, r"%s=\$\{KARTENPLAN_%s:-\$sha\}" % (var, var))
        # every path it stages lies in the renamed package
        self.assertNotIn("python/%s/srt" % ("sg" "lang"), src)


class Names(unittest.TestCase):
    def test_container_token_of_either_product_name(self):
        self.assertEqual(S._container_token("htsglang-acc-nf-h91"), "nfh91")
        self.assertEqual(S._container_token("flliper-acc-nf-h91"), "nfh91")
        self.assertEqual(S._container_token("flliper-mine"), "mine")

    def test_docker_example_names_the_product(self):
        txt = "\n".join(P.docker_run_example("p", {"show_line": False}))
        self.assertIn("--name flliper-mine", txt)
        self.assertIn("ghcr.io/efschu/flliper:<tag> serve", txt)
        self.assertNotIn("htsglang", txt)


class Versions(unittest.TestCase):
    """The revision an issue text names comes from the image ENV under either product prefix."""

    def test_either_prefix_is_read_and_named(self):
        for key in ("HTSGLANG_REVISION_NF", "FLLIPER_REVISION_NF"):
            f = H.version_facts({}, {}, environ={"STAND": "nf", key: "a" * 40})
            self.assertEqual((f["tree_rev"], f["tree_rev_src"]), ("a" * 40, "Image-ENV " + key))
        f = H.version_facts({}, {}, environ={"FLLIPER_REVISION": "b" * 40})
        self.assertEqual(f["tree_rev"], "b" * 40)

    def test_the_stand_specific_name_wins_and_a_non_sha_is_no_evidence(self):
        f = H.version_facts({}, {}, environ={"STAND": "27b", "FLLIPER_REVISION_27B": "c" * 40, "HTSGLANG_REVISION": "d" * 40})
        self.assertEqual(f["tree_rev"], "c" * 40)
        f = H.version_facts({}, {}, environ={"FLLIPER_REVISION": "unknown"})
        self.assertIsNone(f["tree_rev"])


class Routes(unittest.TestCase):
    def test_start_line_page_answers_under_the_new_and_the_old_name(self):
        self.assertEqual(S.LINE_PAGES[:2], ("/pdflip", "/pdflip.html"))
        self.assertIn("/%s" % OLD, S.LINE_PAGES)
        self.assertIn("/%s.html" % OLD, S.LINE_PAGES)

    def test_messages_name_the_new_route(self):
        src = open(os.path.join(PKG, "server.py"), encoding="utf-8").read()
        self.assertNotIn("8890/%s" % OLD, src)
        self.assertIn("8890/pdflip", src)


class Residue(unittest.TestCase):
    """No old env prefix / package path in the dashboard package. The boot recordings (kartenplan_data) are locked by
    rename_rigdash (their plan_id is a hash over the content) and are outside the check."""

    def test_no_old_prefix_or_path_left(self):
        rx = re.compile(r"%s_%s_|%s\.srt|%s/srt" % ("SG" "LANG", "WE" "G2", "sg" "lang", "sg" "lang"))
        bad = {}
        for base in (PKG, os.path.join(DASH, "kartenplan_build")):
            for d, dirs, files in os.walk(base):
                dirs[:] = [x for x in dirs if x not in ("__pycache__", "kartenplan_data")]
                for f in files:
                    if os.path.splitext(f)[1] not in (".py", ".sh", ".js", ".html", ".json", ".md", ".txt", ".service", ""):
                        continue
                    p = os.path.join(d, f)
                    try:
                        txt = open(p, encoding="utf-8").read()
                    except (OSError, UnicodeDecodeError):
                        continue
                    n = len(rx.findall(txt))
                    if n:
                        bad[os.path.relpath(p, DASH)] = n
        self.assertEqual(bad, {})


if __name__ == "__main__":
    unittest.main()
