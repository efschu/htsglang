"""F0-B (rename R5): the dashboard's readers accept the OLD and the RENAMED spelling of every name they match on.

The rename turns the subsystem token into ``PDFLIP`` (markers ``PDFLIP-FLIP``, logger ``pdflip.front``, schemas
``pdflip.state/1``, route ``/pdflip/state``), the package into ``flliper`` and the env prefix into ``FLLIPER_``.  Evidence
written before it keeps the old spelling for good (boot logs, ``state.json``, container names), and a dashboard that is
started after the rename reads both generations side by side.  Each test here feeds one reader the same input in the old
and in the renamed spelling and requires the same, NON-EMPTY answer: a test that only passes the old half, or an empty
answer, proves nothing.

The old excerpts are verbatim log lines of the existing suite; the excerpts are templates (:func:`T`) expanded into the
old and the new spelling; the template machinery is written without the old token in one piece (the mechanical rename would
otherwise turn the fixtures into the new spelling and every pair into a no-op).
"""

import http.server
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import features, grouplog, history, ipcfields, ipcstate, launchview, live, names as N  # noqa: E402
from rigdash import parse, redact, sources, weg2line  # noqa: E402

OLD_U, NEW_U = "WE" "G2", "PDFLIP"
OLD_L, NEW_L = "we" "g2", "pdflip"
OLD_C, NEW_C = "We" "g2", "PdFlip"


def T(template: str, new: bool = False) -> str:
    """A log excerpt template (``@U@`` marker token, ``@l@`` logger/rid token, ``@C@`` class-name token) in the old or the
    renamed spelling -- the same text a renamed tree writes.  Written without the old token in one piece (the mechanical
    rename would otherwise turn the fixtures into the new spelling and the pair into a no-op)."""
    u, l, c = (NEW_U, NEW_L, NEW_C) if new else (OLD_U, OLD_L, OLD_C)
    return template.replace("@U@", u).replace("@l@", l).replace("@C@", c)


# --- verbatim lines of the existing suite (test_rigdash.py), as templates -------------------------------------------------------
FLIP_BEGIN = "[2026-09-27 09:20:42,596] INFO @l@.front: @U@-FLIP begin epoch=17 sleep=P wake=D outstanding=0 queue=2"
FLIP_DONE = ("[2026-09-27 09:20:44,544] INFO @l@.front: @U@-FLIP done epoch=18 slept=P woke=D drain+quiesce=177 ms "
             "sleep=1563 ms (kv RPC + the P leg of the gathered pair) wake=1744 ms (the D leg + kv RPC) interleave=1642 "
             "ms overlap=1540 ms critical_path=wake/D rank=0 card=GPU-31d7 ms=1520 flip_total=1948 ms weights_tags=17")
CORRIDOR = ("[2026-09-27 09:20:35,254] INFO @l@.front: @U@-CORRIDOR phase=P(awake) epoch=17 "
            "instrument=nvml_v2_free,allocatable band=858-1314MiB")
HEALTH = "[2026-09-27 09:24:07,713] WARNING @l@.front: @U@-HEALTH group=P http_ok=False process_alive=False streak=4"
ROUTE = "[2026-09-27 09:20:30,001] INFO @l@.front: @U@-ROUTE decision (awake=P, epoch=17) queue=2"
SEATS = ("[2026-09-27 09:21:00 TP0] @U@ D-PHASE-SEATS (H95) epoch=3 handoff_n=6 parked_n=4 -> n=6 of cap 6 (CLAMPED to cap)")
POST_WAKE = ("[2026-09-27 09:21:02 TP0] @U@-POST-WAKE-PASS n=0 mode=decode schedule_ms=3 run_ms=41 prepare_ms=7")
SERVED = ("[2026-09-27 09:05:39,208] INFO @l@.front: @U@-SERVED group=D leg=2 rid=@l@-0-6 stream=1 status=200 "
          "prompt_tokens=100")
BOOT = ("[2026-09-27T08:59:39Z] @U@-LAUNCH === @U@ BOOT tag=dkrnfh91bar1dauer09270859 tree=/opt/htsglang/src-nf "
        "@ 8f0bf40c2f (clean) stamp=0927_085939 dry=False")
FORM = ("[2026-09-27T08:59:39Z] @U@-LAUNCH @U@-FORM arch=moe experts=offload draft=mtp model=Qwen3.8-Flash-Next "
        "(sources: arch <- checkpoint)")
PROSE_W27 = ("[2026-09-27T14:22:10Z] @U@-LAUNCH W27 PP WIDTH guard armed: a divergence is refused with "
             "'@C@PpWidthDivergence: ...'")
GROUP_ENV = "[2026-09-27T09:00:01Z] @U@-LAUNCH @U@-GROUP-ENV D {'ADMIN_API_KEY': 'abcdef0123456789abcdef'}"


class TestHelper(unittest.TestCase):
    def test_marker_variants_both_directions(self):
        self.assertEqual(N.marker_variants(OLD_U + "-FLIP begin"), (OLD_U + "-FLIP begin", NEW_U + "-FLIP begin"))
        self.assertEqual(N.marker_variants(NEW_U + "-FLIP begin"), (OLD_U + "-FLIP begin", NEW_U + "-FLIP begin"))
        self.assertEqual(N.marker_variants("PP-CUT ACTIVATION"), ("PP-CUT ACTIVATION",))
        self.assertEqual(N.marker_variants("/%s/state" % OLD_L), ("/%s/state" % OLD_L, "/%s/state" % NEW_L))
        self.assertEqual(N.marker_variants("boot_%s_%%s_*.D.log" % NEW_L),
                         ("boot_%s_%%s_*.D.log" % OLD_L, "boot_%s_%%s_*.D.log" % NEW_L))
        self.assertEqual(N.marker_variants("--%s-xchg-census-map" % OLD_L),
                         ("--%s-xchg-census-map" % OLD_L, "--%s-xchg-census-map" % NEW_L))
        self.assertEqual(N.marker_variants("cu130-%s-rc1-27b-nf" % OLD_L)[1], "cu130-%s-rc1-27b-nf" % NEW_L)

    def test_class_names_follow_the_rename_tools_flip_rule(self):
        # the rename tool's rule: <old>Flip<X> -> PdFlip<X> (never PdFlipFlip<X>), <old><X> -> PdFlip<X>
        self.assertEqual(N.marker_variants("We" "g2" "FlipRankDisagree")[1], "PdFlipRankDisagree")
        self.assertEqual(N.marker_variants("We" "g2" "WakeRefused")[1], "PdFlipWakeRefused")
        rx = N.tolerant_compile("We" "g2" "FlipRankDisagree")
        self.assertTrue(rx.search("x PdFlipRankDisagree y") and rx.search("x We" "g2" "FlipRankDisagree y"))

    def test_env_names_both_directions(self):
        sg = "SG" "LANG_"
        self.assertEqual(N.env_variants(sg + "WE" "G2_FORM")[:2], (sg + "WE" "G2_FORM", "FLLIPER_PDFLIP_FORM"))
        self.assertEqual(N.env_variants("FLLIPER_PDFLIP_FORM")[1], sg + "WE" "G2_FORM")
        self.assertIn("HTS" "GLANG_TRANSPORT", N.env_variants("FLLIPER_TRANSPORT"))
        self.assertEqual(N.env_variants("HTS" "GLANG_TRANSPORT"), ("HTS" "GLANG_TRANSPORT", "FLLIPER_TRANSPORT"))
        self.assertEqual(N.env_variants("CUDA_VISIBLE_DEVICES"), ("CUDA_VISIBLE_DEVICES",))
        self.assertEqual(N.env_variants(sg + "OPT_" + "WE" "G2_X")[1], "FLLIPER_OPT_PDFLIP_X")
        self.assertEqual(N.env_get({"FLLIPER_TRANSPORT": "nccl"}, "HTS" "GLANG_TRANSPORT"), "nccl")
        self.assertEqual(N.env_get({"HTS" "GLANG_TRANSPORT": "bar1"}, "FLLIPER_TRANSPORT"), "bar1")
        self.assertIsNone(N.env_get({}, "HTS" "GLANG_TRANSPORT"))
        self.assertEqual(N.env_get({"HTS" "GLANG_TRANSPORT": "a", "FLLIPER_TRANSPORT": "b"}, "FLLIPER_TRANSPORT"), "b")

    def test_state_paths_and_filters(self):
        self.assertEqual(N.state_path_variants("/var/lib/hts" "glang/evidence"),
                         ("/var/lib/hts" "glang/evidence", "/var/lib/flliper/evidence"))
        self.assertEqual(N.state_path_variants("/var/lib/flliper/evidence")[0], "/var/lib/hts" "glang/evidence")
        self.assertEqual(N.state_path_variants("/elsewhere"), ("/elsewhere",))
        self.assertEqual(N.docker_name_filters(), "--filter name=hts" "glang --filter name=flliper")
        self.assertTrue(N.schema_ok("pdflip.state/1", "we" "g2.state/1") and N.schema_ok("we" "g2.state/1", "we" "g2.state/1"))
        self.assertFalse(N.schema_ok("other.state/1", "we" "g2.state/1") or N.schema_ok(None, "we" "g2.state/1"))

    def test_helper_spells_no_old_token_in_one_piece(self):
        """The hermetic half of the rename check below: the rename rewrites exactly these tokens."""
        src = open(N.__file__).read()
        self.assertFalse(re.search(OLD_L + "|" + OLD_U + "|W" "eg2", src))
        self.assertFalse(re.search("sg" "lang|SG" "LANG|SG" "Lang|Sg" "lang|hts" "glang|HTS" "GLANG", src))

    def test_helper_is_a_fixed_point_of_the_rename(self):
        """The mechanical rename (all rule sets on) leaves names.py byte-identical, so its both-spellings answer is the same
        before and after the rename (the kit must be on this box)."""
        p = "/spinning/flliper/tools/rename_to_flliper.py"
        if not os.path.exists(p):
            self.skipTest("rename tool not on this box")
        spec = importlib.util.spec_from_file_location("_rename_to_flliper", p)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        src = open(N.__file__).read()
        out, rep, _skip = tool.rewrite_all(src, True, True, {})
        self.assertEqual(out, src, rep)


class TestLogLineParsers(unittest.TestCase):
    def _pair(self, line):
        old, new = parse.parse_line(T(line)), parse.parse_line(T(line, True))
        self.assertIsNotNone(old, line)
        self.assertEqual(new, old)
        return old

    def test_front_and_launcher_lines(self):
        self.assertEqual(self._pair(FLIP_BEGIN)["kind"], "flip_begin")
        self.assertEqual(self._pair(FLIP_DONE)["total_ms"], 1948.0)
        self.assertEqual(self._pair(CORRIDOR)["kind"], "phase")
        self.assertEqual(self._pair(HEALTH)["kind"], "health")
        self.assertEqual(self._pair(ROUTE)["kind"], "route")
        self.assertEqual(self._pair(SEATS)["kind"], "d_seats")
        self.assertEqual(self._pair(POST_WAKE)["kind"], "post_wake0")

    def test_field_regexes(self):
        for rx, line in ((parse.F_SERVED, SERVED), (parse.F_BOOT, BOOT), (parse.F_FORM, FORM), (parse.F_FORM_MODEL, FORM)):
            old, new = rx.search(T(line)), rx.search(T(line, True))
            self.assertTrue(old and new, line)
            # the rid of a front line carries the subsystem token as its prefix (``<token>-0-6``): compare it spelled the old way
            self.assertEqual([g and g.replace(NEW_L, OLD_L) for g in new.groups()], list(old.groups()))

    def test_stop_match_excludes_launcher_prose_in_both_spellings(self):
        self.assertFalse(parse.stop_match(T(PROSE_W27)))
        self.assertFalse(parse.stop_match(T(PROSE_W27, True)))
        real = "[2026-09-27T14:22:10Z] Traceback (most recent call last):"
        self.assertTrue(parse.stop_match(real))

    def test_launch_lines_and_group_env_drop(self):
        old = live.launch_lines([T(x) for x in (BOOT, FORM, PROSE_W27, GROUP_ENV)])
        new = live.launch_lines([T(x, True) for x in (BOOT, FORM, PROSE_W27, GROUP_ENV)])
        self.assertEqual(len(old), 2)          # BOOT, FORM; the GROUP-ENV line is dropped whole, the prose has no key
        self.assertEqual(len(new), len(old))
        self.assertTrue(old[0].startswith("=== " + OLD_U + " BOOT tag=") and new[0].startswith("=== " + NEW_U + " BOOT tag="))
        self.assertEqual([x.replace(NEW_U, OLD_U) for x in new], old)

    def test_group_env_line_never_reaches_a_page(self):
        secret = "abcdef0123456789abcdef"
        for new in (False, True):
            line = T(GROUP_ENV, new)
            self.assertIsNone(redact.clean(line), line)
            marker = T("@U@-GROUP-ENV", new)
            self.assertNotIn(marker, redact.guard('{"t": "%s D {}"}' % marker))
            self.assertNotIn(secret, redact.clean(T("@U@-LAUNCH note token=%s" % secret, new)) or "")


class TestImages(unittest.TestCase):
    def test_old_renamed_and_fl_images(self):
        text = ("hts" "glang:cu130-" + OLD_L + "-rc12z30-27b-nf-flat\tid1\tt\n"
                "flliper:cu130-" + NEW_L + "-rc12z30-27b-nf\tid2\tt\n"
                "flliper:0.1.0-rc1-cu130\tid3\tt\n"
                "flliper:cu130-2c5fd4a2b7\tid4\tt\n"
                "flliper:0.1.0-dry0930-cu130\tid5\tt\n"
                "wyoming-whisper:latest\tid6\tt\n")
        got = weg2line.parse_images(text)
        self.assertEqual([(i["cuda"], i["release"], i["flat"]) for i in got],
                         [("cu130", "rc12z30", True), ("cu130", "rc12z30", False), ("cu130", "0.1.0-rc1", True),
                          ("cu130", "2c5fd4a2b7", True), ("cu130", "0.1.0-dry0930", True)])


class TestGroupLogs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ev = os.path.join(self.tmp.name, "evidence")
        os.makedirs(self.ev)
        self.sdir = os.path.join(self.tmp.name, "state", "boot-1")
        os.makedirs(self.sdir)

    def _touch(self, name, text=""):
        p = os.path.join(self.ev, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_log_stems_of_both_generations(self):
        old = self._touch("boot_%s_tagA_x.D.log" % OLD_L)
        new = self._touch("boot_%s_tagB_x.D.log" % NEW_L)
        fold = self._touch("boot_%s_tagA_x.front.log" % OLD_L)
        fnew = self._touch("boot_%s_tagB_x.front.log" % NEW_L)
        self.assertEqual(grouplog.d_log_path({"dir": self.sdir, "tag": "tagA"}), old)
        self.assertEqual(grouplog.d_log_path({"dir": self.sdir, "tag": "tagB"}), new)
        self.assertEqual(grouplog.front_log_path({"dir": self.sdir, "tag": "tagA"}), fold)
        self.assertEqual(grouplog.front_log_path({"dir": self.sdir, "tag": "tagB"}), fnew)
        self.assertIsNone(grouplog.d_log_path({"dir": self.sdir, "tag": "tagC"}))

    def test_manifest_env_of_a_renamed_boot(self):
        p = self._touch("anystem.D.log")
        for key in ("SG" "LANG_WEIGHT_LOADER_SHARED_CACHE_MANIFEST", "FLLIPER_WEIGHT_LOADER_SHARED_CACHE_MANIFEST"):
            ipc = {"dir": self.sdir, "launch": {"D": {"env": {key: os.path.join(self.ev, "anystem.shared_cache")}}}}
            self.assertEqual(grouplog.d_log_path(ipc), p, key)

    def test_front_arrivals_read_both_session_markers(self):
        lines = ["[2026-10-02 18:25:01,100] INFO %s.front: %s SESSION rid=r-%d x=1\n" % (OLD_L, OLD_U, 1),
                 "[2026-10-02 18:25:02,200] INFO %s.front: %s SESSION rid=r-%d x=1\n" % (NEW_L, NEW_U, 2),
                 "[2026-10-02 18:25:03,300] INFO %s.front: no session here rid=r-3\n" % NEW_L]
        p = self._touch("s.front.log", "".join(lines))
        first = grouplog.FrontArrivals(p).poll()
        self.assertEqual(sorted(first), ["r-1", "r-2"])


def _state(schema, env=None):
    return {"schema": schema, "seq": 1, "boot_id": "b-1", "kind": "boot", "tag": "tagX", "line": "nf",
            "lifecycle": {"state": "serving", "since_ts": 1.0}, "front": {"awake": "P"},
            "groups": {"P": {"state": "ready", "launch": {"argv": ["python", "-m", "x"], "env": env or {}}}}}


class TestIpc(unittest.TestCase):
    def test_state_json_of_both_generations(self):
        for schema, env, want in ((OLD_L + ".state/1", {"HTS" "GLANG_TRANSPORT": "bar1"}, "bar1"),
                                  (NEW_L + ".state/1", {"FLLIPER_TRANSPORT": "nccl"}, "nccl")):
            with tempfile.TemporaryDirectory() as root:
                d = os.path.join(root, "b-1")
                os.makedirs(d)
                with open(os.path.join(d, "state.json"), "w") as fh:
                    json.dump(_state(schema, env), fh)
                ev = ({"schema": schema.replace(".state/", ".event/"), "seq": 1, "boot_id": "b-1", "type": "flip_user_time",
                       "ts": 5.0, "data": {"dir": "D>P"}})
                with open(os.path.join(d, "events.jsonl"), "w") as fh:
                    fh.write(json.dumps(ev) + "\n")
                s = ipcstate.IpcStates(roots=(root,))
                s.poll()
                v = s.for_tag("tagX")
                self.assertIsNotNone(v, schema)
                self.assertEqual(v["transport"], want, schema)
                self.assertEqual(len(v["flip_user_time"]), 1, schema)

    def test_unknown_schema_stays_refused(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "b-1")
            os.makedirs(d)
            with open(os.path.join(d, "state.json"), "w") as fh:
                json.dump(_state("other.state/1"), fh)
            s = ipcstate.IpcStates(roots=(root,))
            s.poll()
            self.assertIsNone(s.for_tag("tagX"))

    def test_rankstats_of_both_generations(self):
        with tempfile.TemporaryDirectory() as root:
            for i, schema in enumerate((OLD_L + ".rankstats/1", NEW_L + ".rankstats/1", "foreign.rankstats/1")):
                with open(os.path.join(root, "D.tp%dpp0.rankstats" % i), "w") as fh:
                    json.dump({"schema": schema, "tokens": {"prefill_total": 1}}, fh)
            got = ipcfields.read_rank_files([root])
            self.assertEqual(sorted(got["rankstats"]), ["D.tp0pp0", "D.tp1pp0"])


class TestSources(unittest.TestCase):
    def test_container_volume_of_both_names(self):
        for dest in ("/var/lib/hts" "glang/evidence", "/var/lib/flliper/evidence"):
            ins = [{"Name": "/c", "Mounts": [{"Destination": dest, "Source": "/hostpfx/ev"}]}]
            self.assertEqual(sources.container_log_dirs(ins, "/hostpfx"), {"c": "/ev"}, dest)
        self.assertEqual(sources.container_log_dirs([{"Name": "/c", "Mounts": [{"Destination": "/var/lib/other"}]}], ""), {})

    def test_docker_listings_name_both_products(self):
        self.assertIn("--filter name=hts" "glang --filter name=flliper", history.HistoryDB.HOST_CMD
                      if hasattr(history.HistoryDB, "HOST_CMD") else history.Recorder.HOST_CMD)

    def _serve(self, routes):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):    # noqa: N802
                if self.path in routes:
                    body = json.dumps(routes[self.path]).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return "http://127.0.0.1:%d" % srv.server_address[1]

    def test_front_state_route_of_either_generation(self):
        sources._FRONT_ROUTE.clear()
        old_front = self._serve({"/%s/state" % OLD_L: {"tag": "t1"}})
        new_front = self._serve({"/%s/state" % NEW_L: {"tag": "t2"}})
        both = self._serve({"/%s/state" % OLD_L: {"tag": "t3"}, "/%s/state" % NEW_L: {"tag": "t3"}})
        self.assertEqual(sources.front_state(old_front), {"tag": "t1"})
        self.assertEqual(sources.front_state(new_front), {"tag": "t2"})
        self.assertEqual(sources.front_state(both), {"tag": "t3"})
        self.assertEqual(sources._FRONT_ROUTE[old_front], "/%s/state" % OLD_L)      # the answering spelling is remembered
        self.assertEqual(sources._FRONT_ROUTE[new_front], "/%s/state" % NEW_L)
        dead = self._serve({})
        with self.assertRaises(Exception):
            sources.front_state(dead)


class TestSwitchesAndFlags(unittest.TestCase):
    def _launch(self, env, argv):
        return {"groups": {"D": {"launch": {"env": env, "argv": argv}}, "P": {"launch": {"env": env, "argv": argv}}}}

    def test_env_switch_of_both_generations(self):
        sw = {"name": "SG" "LANG_WE" "G2_FORM_X", "art": "env", "gruppe": "D", "default": "aus", "an_value": "1"}
        for env in ({"SG" "LANG_WE" "G2_FORM_X": "1"}, {"FLLIPER_PDFLIP_FORM_X": "1"}):
            r = features.switch_state(sw, self._launch(env, []), None)
            self.assertEqual((r["state"], r["value"]), ("an", "D=1"), env)
        r = features.switch_state(sw, self._launch({}, []), None)
        self.assertEqual(r["state"], "aus")

    def test_flag_switch_of_both_generations(self):
        sw = {"name": "--" + OLD_L + "-x-y", "art": "flag", "gruppe": "D", "default": "aus"}
        for argv in (["--" + OLD_L + "-x-y"], ["--" + NEW_L + "-x-y", "3"]):
            r = features.switch_state(sw, self._launch({}, argv), None)
            self.assertEqual(r["state"], "an", argv)

    def test_profile_text_lookup_of_both_generations(self):
        sw = {"name": "SG" "LANG_WE" "G2_FORM_X", "art": "env", "gruppe": "front", "default": "aus", "an_value": "1"}
        for text in ('export SG' 'LANG_WE' 'G2_FORM_X=1', 'export FLLIPER_PDFLIP_FORM_X=1'):
            r = features.switch_state(sw, None, text)
            self.assertEqual((r["state"], r["value"]), ("an", "1"), text)

    def test_flag_groups(self):
        for flag in ("--" + OLD_L + "-a", "--" + NEW_L + "-a"):
            self.assertEqual(launchview.flag_group(flag), launchview.FLAG_GROUPS[0][0], flag)
        self.assertEqual(launchview.flag_group("--p-x"), "--p-*")


class TestRedactKnownNames(unittest.TestCase):
    def test_builtin_names_both_spellings_derived_without_tree(self):
        """Fix-Runde 1, Befund 1: the built-in set holds old AND new spelling of every refusal class, derived through names (no contiguous old literal)."""
        old, new = N.CAMEL_TOKENS
        for suffix in ("TpOperatingPointInfeasible", "XchgResidencyUnarmable", "XchgSemaphoreNotRearmed", "DualCompactBreach"):
            self.assertIn(old + suffix, redact._KNOWN_IDENT_BUILTIN)
            self.assertIn(new + suffix, redact._KNOWN_IDENT_BUILTIN)
        self.assertIn(old + "FlipPeerLegAborted", redact._KNOWN_IDENT_BUILTIN)
        self.assertIn(new + "PeerLegAborted", redact._KNOWN_IDENT_BUILTIN)
        self.assertEqual(len(redact._KNOWN_IDENT_BUILTIN), 10)

    def test_renamed_tree_class_names_stay_readable(self):
        with tempfile.TemporaryDirectory() as tree:
            sub = os.path.join(tree, "flliper", "srt", NEW_L)
            os.makedirs(sub)
            with open(os.path.join(sub, "launcher.py"), "w") as fh:
                fh.write("class PdFlipTpOperatingPointInfeasible(Exception):\n    pass\nraise PdFlipOtherThing('x')\n")
            old_cache, old_env = redact._known_cache, os.environ.get("HWPROFIL_TREE")
            try:
                redact._known_cache = None
                os.environ["HWPROFIL_TREE"] = tree
                self.assertIn("PdFlipOtherThing", redact.known_idents())
                self.assertIn("PdFlipTpOperatingPointInfeasible", redact.known_idents())
                self.assertIn("We" "g2" "TpOperatingPointInfeasible", redact.known_idents())
            finally:
                redact._known_cache = old_cache
                if old_env is None:
                    os.environ.pop("HWPROFIL_TREE", None)
                else:
                    os.environ["HWPROFIL_TREE"] = old_env


if __name__ == "__main__":
    unittest.main()
