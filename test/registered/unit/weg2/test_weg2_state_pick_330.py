"""ITEM 330: the DRY check '#791C ABORT-MID-CHUNK plan from 27B state.json' must not hang on a stale
``state/current`` that points at a start the host guard refused (refused_preflight, groups {}).

weg2/tools/state_pick.py chooses the state dir: current if it carries groups.P, else the newest
kind=boot state that does; the DRY line names the chosen dir.  Fake state dirs only (written through
the real state_file writer, so ``current`` moves exactly like on the host) -- no GPU, no server, no
launcher process.
"""

import os
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.weg2 import state_file as sf
from sglang.srt.weg2.tools import state_pick as sp

TMP_BASE = "/root/.claude/jobs/aef87d47/tmp"
NEW_SCRIPT = "/spinning/gpu-arb/docker/acc_cu130_rc12z30y_27b.sh.new_330"
BENCH = "/spinning/gpu-arb/devtools/nf_cycle_bench.py"
P_ARGV = ["python3", "-m", "sglang.launch_server", "--host", "127.0.0.1", "--port", "30031",
          "--chunked-prefill-size", "16384"]


def _tmp():
    return tempfile.TemporaryDirectory(dir=TMP_BASE if os.path.isdir(TMP_BASE) else None)


def _groups_with_p():
    return {"P": {"launch": {"argv": P_ARGV}, "state": "ready"}, "D": {"launch": {"argv": ["x"]}, "state": "ready"}}


def _boot(root, bid, *, groups=None, serving=False, kind="boot"):
    """init repoints `current` at the new id (like the host); the launcher then writes groups."""
    d = sf.init(root, bid, kind, {})
    if groups:
        sf.transition(d, "loading", writer="host")
        sf.transition(d, None, fields={"groups": groups}, writer="launcher")
    if serving:
        sf.transition(d, "serving", writer="host")
    return d


def _refuse(d):
    sf.transition(d, "refused_preflight", writer="host",
                  cause=sf.make_cause("PRE_CT999_BOOT", "preflight", "CT999 boot", rc=sf.RC_REFUSED_PREFLIGHT))


def _incident(root):
    """03.10. 08:24Z: a serving boot (older), then a refused_preflight start (newest, and `current`)."""
    old = _boot(root, "27bbf-boot-20261003T074633Z-c2bc", groups=_groups_with_p(), serving=True)
    new = _boot(root, "27bbf-boot-20261003T082237Z-779c")
    _refuse(new)
    return old, new


class TestPick(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmp()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _current(self):
        return os.path.basename(os.path.realpath(os.path.join(self.root, "current")))

    def test_refused_newest_is_skipped_and_serving_dir_named(self):
        old, new = _incident(self.root)
        self.assertEqual(self._current(), os.path.basename(new))
        d, note = sp.pick(self.root, ("P",))
        self.assertEqual(d, old)
        self.assertIn("27bbf-boot-20261003T074633Z-c2bc", note)
        self.assertIn("uebersprungen", note)
        self.assertIn("refused_preflight", note)
        self.assertEqual(self._current(), os.path.basename(new), "the pointer itself is not touched")

    def test_current_with_groups_wins_even_if_an_older_one_has_groups_too(self):
        _boot(self.root, "27bbf-boot-20261003T062350Z-d18f", groups=_groups_with_p())
        cur = _boot(self.root, "27bbf-boot-20261003T071803Z-ba1f", groups=_groups_with_p())
        d, note = sp.pick(self.root, ("P",))
        self.assertEqual(d, cur)
        self.assertNotIn("uebersprungen", note)

    def test_newest_by_boot_id_stamp_not_by_listing_order(self):
        _boot(self.root, "27bbf-boot-20261003T062350Z-d18f", groups=_groups_with_p())
        newer = _boot(self.root, "27bbf-boot-20261003T071803Z-ba1f", groups=_groups_with_p())
        _boot(self.root, "27bbf-boot-20261002T195107Z-3444", groups=_groups_with_p())   # created last, oldest stamp
        bad = _boot(self.root, "27bbf-boot-20261003T082237Z-779c")                       # current, no groups
        self.assertEqual(self._current(), os.path.basename(bad))
        d, _ = sp.pick(self.root, ("P",))
        self.assertEqual(d, newer)

    def test_d2_run_is_not_a_candidate_and_nothing_found_is_none(self):
        _boot(self.root, "27bbf-d2-20261003T082739Z-e4a9", groups=_groups_with_p(), kind="d2")
        _boot(self.root, "27bbf-boot-20261003T082237Z-779c")
        d, note = sp.pick(self.root, ("P",))
        self.assertIsNone(d)
        self.assertIn("kein State mit groups", note)

    def test_need_group_must_be_non_empty(self):
        only_d = _boot(self.root, "27bbf-boot-20261003T071803Z-ba1f", groups={"D": {"launch": {"argv": ["x"]}}})
        self.assertEqual(sp.pick(self.root, ())[0], only_d)
        self.assertIsNone(sp.pick(self.root, ("P",))[0])

    def test_unreadable_state_and_dangling_current_are_skipped(self):
        good = _boot(self.root, "27bbf-boot-20261003T071803Z-ba1f", groups=_groups_with_p())
        junk = os.path.join(self.root, "27bbf-boot-20261003T090000Z-0000")
        os.makedirs(junk)
        with open(os.path.join(junk, "state.json"), "w") as fh:
            fh.write("{not json")
        self.assertEqual(sp.pick(self.root, ("P",))[0], good)
        os.unlink(os.path.join(self.root, "current"))
        os.symlink("does-not-exist", os.path.join(self.root, "current"))
        self.assertEqual(sp.pick(self.root, ("P",))[0], good)

    def test_cli_prints_path_then_note(self):
        old, _ = _incident(self.root)
        r = subprocess.run([sys.executable, sp.__file__, "--root", self.root, "--need", "P"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(lines[0], os.path.join(old, "state.json"))
        self.assertIn("27bbf-boot-20261003T074633Z-c2bc", lines[1])

    def test_cli_exit_1_and_empty_path_when_nothing_has_groups(self):
        _boot(self.root, "27bbf-boot-20261003T082237Z-779c")
        r = subprocess.run([sys.executable, sp.__file__, "--root", self.root, "--need", "P"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout.splitlines()[0], "")


@unittest.skipUnless(os.path.exists(NEW_SCRIPT) and os.path.exists(BENCH),
                     "host DRY script / bench not on this machine")
class TestDryLineOfTheNewScript(unittest.TestCase):
    """Runs the exact lines of the .new_330 DRY check (extracted from the script, not re-implemented)."""

    def _snippet(self):
        lines = open(NEW_SCRIPT).read().splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.lstrip().startswith('_sp=$(python3 "$D/state_pick.py"'))
        j = next(k for k in range(i, len(lines)) if "ABORT-MID-CHUNK (#791C) Plan aus 27B-state.json" in lines[k])
        return "\n".join(lines[i:j + 1])

    def _run(self, acc, with_picker=True):
        d_dir = os.path.join(acc, "docker")
        os.makedirs(d_dir, exist_ok=True)
        if with_picker:
            os.symlink(sp.__file__, os.path.join(d_dir, "state_pick.py"))
        prog = ('chk(){ local ok=$1; shift; if [ "$ok" = 0 ]; then echo "DRY OK   $*"; else echo "DRY FEHL $*"; fi; }\n'
                f"D={d_dir}; ACC={acc}; PORT=1; MODEL_NAME=m; _cl={os.path.dirname(BENCH)}; _cb={BENCH}\n"
                + self._snippet())
        r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, timeout=120)
        return r.stdout

    def test_refused_newest_serving_older_passes_and_names_serving_dir(self):
        with _tmp() as acc:
            root = os.path.join(acc, "state")
            os.makedirs(root)
            _incident(root)
            out = self._run(acc)
            self.assertIn("DRY OK   ABORT-MID-CHUNK (#791C)", out)
            self.assertIn("chunk=16384", out)
            self.assertIn("state=27bbf-boot-20261003T074633Z-c2bc", out)
            self.assertNotIn("p=None", out)

    def test_control_original_line_fails_on_the_same_dirs(self):
        """The fixture IS the incident: reading `current` directly (the original line) gives p=None / no chunk."""
        with _tmp() as acc:
            root = os.path.join(acc, "state")
            os.makedirs(root)
            _incident(root)
            r = subprocess.run([sys.executable, BENCH, "--abort-mid-chunk", "--dry-run", "--front", "http://127.0.0.1:1",
                                "--model", "m", "--tokenizer", "none",
                                "--state-json", os.path.join(root, "current", "state.json")],
                               capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONPATH": os.path.dirname(BENCH)})
            self.assertIn("p=None", r.stdout)
            self.assertNotIn("chunk=16384", r.stdout)

    def test_missing_picker_falls_back_to_current_and_says_so(self):
        with _tmp() as acc:
            root = os.path.join(acc, "state")
            os.makedirs(root)
            _incident(root)
            out = self._run(acc, with_picker=False)
            self.assertIn("DRY FEHL", out)
            self.assertIn("Rueckfall current", out)


if __name__ == "__main__":
    unittest.main()
