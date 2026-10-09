"""F0-I: the rest-inventory tool (tools/release/rest_inventory_1008.py) gives every old-name hit exactly one verdict.

The old words are assembled from fragments here, so that this file is itself free of the names it tests (the inventory of the tree
must not list its own test).  Pure text in, verdicts out: no tree scan, no GPU.
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
_spec = importlib.util.spec_from_file_location("rest_inventory_1008", os.path.join(ROOT, "tools", "release", "rest_inventory_1008.py"))
RI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RI)

SG = "sg" + "lang"
SGU = "SG" + "LANG"
W2 = "we" + "g2"
W2U = "WE" + "G2"
HT = "hts" + "glang"
HTU = "HTS" + "GLANG"
CTX = {"collision_ok": {}}


def verdicts(path, text, ctx=CTX):
    return [(h.word, kind, rid) for h, kind, rid, _why in RI.classify_file(path, text, ctx)]


class Rules(unittest.TestCase):
    def test_unexplained_old_import_is_a_residue(self):
        v = verdicts("python/flliper/srt/x.py", "import %s.srt.utils\n" % SG)
        self.assertEqual(v, [(SG, "residue", "unexplained")])

    def test_unexplained_old_env_and_subsystem_are_residues(self):
        v = verdicts("python/flliper/srt/x.py", 'os.environ.get("%s_FOO")\nx = "%s_foobar"\n' % (SGU, W2))
        self.assertEqual([k for _w, k, _r in v], ["residue", "residue"])

    def test_boot_tag_is_kept_by_the_engine_span(self):
        v = verdicts("python/flliper/srt/x.py", "# measured on boot %sxsn246\n" % W2)
        self.assertEqual(v, [(W2, "keep", "kit-span:boot-tag")])

    def test_attribution_and_url_are_kept(self):
        t = "# Copyright 2023-2024 SGLang Team\n# see https://github.com/sgl-project/%s/pull/1\n" % SG
        v = verdicts("python/flliper/srt/x.py", t)
        self.assertTrue(v and all(k == "keep" for _w, k, _r in v), v)

    def test_whole_trees_and_cxx_are_kept(self):
        self.assertEqual(verdicts("docs/x.md", "%s\n" % SG)[0][1:], ("keep", "tree:docs/"))
        self.assertEqual(verdicts("sgl-kernel/a.py", "%s\n" % SG)[0][1:], ("keep", "tree:sgl-kernel/"))
        self.assertEqual(verdicts("python/flliper/csrc/a.cu", "namespace %s {}\n" % SG)[0][1:], ("keep", "cxx-namespace"))

    def test_product_env_and_host_path_are_kept(self):
        v = verdicts("docker/x.sh", "export %s_PROFILE=1\ncd /spinning/%s/x\n" % (HTU, HT))
        self.assertEqual([(k, r) for _w, k, r in v], [("keep", "HTSGLANG_env"), ("keep", "host-path-htsglang")])

    def test_product_layer_names_are_findings_not_keeps(self):
        v = verdicts("python/flliper/srt/x.py", 'H = "x-%s-id"\nU = "%s-serving@.service"\n' % (HT, HT))
        self.assertEqual([(k, r) for _w, k, r in v], [("finding", "htsglang-api-namespace"), ("finding", "htsglang-units-config")])

    def test_dual_line_is_kept_only_with_the_new_spelling_on_the_line(self):
        v = verdicts("python/flliper/srt/x.py", 'N = ("%s", "flliper")\nM = "%s"\n' % (SG, SG))
        self.assertEqual([(k, r) for _w, k, r in v], [("keep", "dual-line"), ("residue", "unexplained")])

    def test_collision_ok_file_keeps_the_old_word_but_not_the_product_name(self):
        ctx = {"collision_ok": {"python/flliper/srt/y.py": frozenset({"flliper"})}}
        v = verdicts("python/flliper/srt/y.py", "a = '%s'\nb = '%s-pip'\n" % (SG, HT), ctx)
        self.assertEqual([(k, r) for _w, k, r in v][0], ("keep", "collision_ok-file"))
        self.assertEqual([(k, r) for _w, k, r in v][1][0], "finding")

    def test_outside_the_kit_scope_is_a_finding(self):
        v = verdicts("HANDOVER_760.md", "the %s fork\n" % SG)
        self.assertEqual([(k, r) for _w, k, r in v], [("finding", "outside-kit-scope")])

    def test_summary_counts_every_hit_once(self):
        res = RI.classify_file("python/flliper/srt/x.py", "import %s\nboot %sdk5\n" % (SG, W2), CTX)
        s = RI.summarize(res)
        self.assertEqual(s["total"], 2)
        self.assertEqual(sum(r["total"] for r in s["rules"].values()), 2)


if __name__ == "__main__":
    unittest.main()
