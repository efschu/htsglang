"""NF #162 (23.10.): every scheduler rank arms the CUDA coredump itself and the CUDA driver names, on the rank's own
log line, whether a GPU exception there writes a dump -- where, and with which flags.

Background: the import-time injection (debug_utils/cuda_coredump.py) runs once in the top-level process and prints
"Injecting CUDA coredump env vars" only there, so the rc32 boot log showed it for launch_server alone; the ranks'
/proc/<pid>/environ cannot show it (setproctitle overwrites that block, desk/nf-rc32-cudacore-1022). The rc31b death
(D-TP1 illegal memory access, deskq/done/nf-rc31b-ima-1022.md) needs the dump from exactly such a rank.

These tests run the REAL libcuda without a GPU: under CUDA_VISIBLE_DEVICES="" cuInit returns CUDA_ERROR_NO_DEVICE
(100) but still latches the coredump environment, and cuCoredumpGetAttributeGlobal reports what the driver took
(measured 23.10. on driver 13020: enable=1, file expanded to <dir>/cuda_coredump_<host>.<pid>.<t>, flags 0x3f).
"""
import ast
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PY = os.path.join(ROOT, "python")
sys.path.insert(0, PY)

LIGHT = "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory"


def _libcuda_ok():
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        return hasattr(lib, "cuCoredumpGetAttributeGlobal") and hasattr(lib, "cuCoredumpSetAttributeGlobal")
    except OSError:
        return False


def _clean_env(extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CUDA_COREDUMP") and k not in (
               "CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_ENABLE_LIGHTWEIGHT_COREDUMP",
               "CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION", "FLLIPER_CUDA_COREDUMP", "SGLANG_CUDA_COREDUMP",
               "FLLIPER_CUDA_COREDUMP_DIR", "SGLANG_CUDA_COREDUMP_DIR", "_FLLIPER_CUDA_COREDUMP_INJECTED",
               "FLLIPER_CUDA_COREDUMP_SKIP_ABORT")}
    env.update({"PYTHONPATH": PY, "CUDA_VISIBLE_DEVICES": ""})
    env.update(extra)
    return env


# The production chain for real. Parent = the launch_server group process: the launcher env (FLLIPER_CUDA_COREDUMP=1,
# DIR, the sentinel inherited from the launcher, and the live NF profile's --env-d override of the CUDA_* variables
# WITHOUT skip_abort). It imports flliper.srt.environ, then spawns the scheduler like Engine does (mp spawn). The child
# does what a rank does in that order: its unpickling import of flliper (the driver latches the coredump env here),
# setproctitle, the RANK-ENV line, CUDA up, the DRIVER line.
_RANK_PROBE = r"""
import json, logging, multiprocessing as mp, os, sys

def rank(q):
    from flliper.srt.debug_utils import cuda_coredump as cc   # package init -> srt.platforms -> driver latch
    import setproctitle
    setproctitle.setproctitle("flliper::scheduler_TP1")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    log = logging.getLogger("probe")
    env = cc.log_rank_env("D TP1 PP0", log)
    drv = cc.driver_report("D TP1 PP0", log)
    sys.stdout.flush()
    q.put({"env": env, "drv": drv, "pid": os.getpid()})

if __name__ == "__main__":
    import flliper.srt.environ  # noqa: F401
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=rank, args=(q,))
    p.start()
    res = q.get(timeout=240)
    p.join(60)
    res["parent_flags"] = os.environ.get("CUDA_COREDUMP_GENERATION_FLAGS")
    print("RESULT " + json.dumps(res))
"""


def _run_rank(extra):
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "rank_probe.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_RANK_PROBE)
        out = subprocess.run([sys.executable, script], env=_clean_env(extra), capture_output=True,
                             text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.splitlines()
    res = json.loads([l for l in lines if l.startswith("RESULT ")][-1][len("RESULT "):])
    res["log"] = [l for l in lines if "CUDA-COREDUMP" in l]
    return res


def _launcher_env(dump_dir, **more):
    env = {"FLLIPER_CUDA_COREDUMP": "1", "FLLIPER_CUDA_COREDUMP_DIR": dump_dir,
           "_FLLIPER_CUDA_COREDUMP_INJECTED": "1",  # the launcher already injected; the group process skips that block
           "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1", "CUDA_COREDUMP_SHOW_PROGRESS": "1",
           "CUDA_COREDUMP_GENERATION_FLAGS": LIGHT,   # profile _cudacore override, applied last by build_env
           "CUDA_COREDUMP_FILE": dump_dir + "/cuda_coredump_%h.%p.%t"}
    env.update(more)
    return env


@unittest.skipUnless(_libcuda_ok(), "libcuda.so.1 with the coredump attribute API not loadable")
class TheRankIsSpawnedArmedWithSkipAbortAndTheDriverSaysSo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dump_dir = os.path.join(cls.tmp.name, "evidence", "cudacore")
        cls.res = _run_rank(_launcher_env(cls.dump_dir))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_group_process_puts_skip_abort_into_the_spawn_env(self):
        self.assertEqual(self.res["parent_flags"], LIGHT + ",skip_abort")

    def test_the_rank_logs_one_rank_env_line_from_libc_getenv(self):
        line = [l for l in self.res["log"] if l.startswith("INFO CUDA-COREDUMP RANK-ENV rank=D TP1 PP0 ")]
        self.assertEqual(len(line), 1, self.res["log"])
        self.assertIn("CUDA_COREDUMP_GENERATION_FLAGS=" + LIGHT + ",skip_abort ", line[0])
        self.assertIn("writable=1", line[0])

    def test_the_driver_itself_says_enable_skip_abort_and_the_expanded_file(self):
        d = self.res["drv"]
        self.assertTrue(d["ok"] and d["skip_abort_ok"], d)
        self.assertTrue(d["enable"])
        self.assertEqual(d["source"], "env")
        self.assertEqual(d["flags"], 0x3F)  # lightweight 0x2f + skip_abort 0x10
        self.assertTrue(d["file"].startswith(self.dump_dir + "/cuda_coredump_"), d["file"])
        self.assertIn(".%d." % self.res["pid"], d["file"])
        line = [l for l in self.res["log"] if l.startswith("INFO CUDA-COREDUMP DRIVER rank=D TP1 PP0 ")]
        self.assertEqual(len(line), 1, self.res["log"])
        self.assertIn("returns the error to the rank (skip_abort)", line[0])


@unittest.skipUnless(_libcuda_ok(), "libcuda.so.1 with the coredump attribute API not loadable")
class NegativeControlTheSpawnEnvIsWhatTheDriverKeeps(unittest.TestCase):
    """Switch FLLIPER_CUDA_COREDUMP_SKIP_ABORT off: the group process leaves the profile's flags alone, the rank is
    spawned without skip_abort, its own import cannot change what the driver latched, the API set is refused, and the
    DRIVER line says so as a WARNING -- the rank-side witness does not paper over it."""

    def test_without_skip_abort_in_the_spawn_env_the_driver_keeps_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = os.path.join(tmp, "cudacore")
            res = _run_rank(_launcher_env(dump_dir, FLLIPER_CUDA_COREDUMP_SKIP_ABORT="0"))
        self.assertEqual(res["drv"]["flags"], 0x2F, res["drv"])
        self.assertTrue(res["drv"]["enable"])
        self.assertTrue(any("calls abort()" in l for l in res["log"]), res["log"])


class SwitchedOffTheRankDoesNothing(unittest.TestCase):
    def test_off_means_no_record_no_env_no_driver_call(self):
        code = (
            "import json, os\n"
            "from flliper.srt.debug_utils import cuda_coredump as cc\n"
            "def boom():\n    raise AssertionError('driver touched while off')\n"
            "r = cc.rank_env_record('D TP0 PP0'); d = cc.driver_report('D TP0 PP0', driver_factory=boom)\n"
            "print(json.dumps({'r': r, 'd': d, 'env': [k for k in os.environ if k.startswith('CUDA_COREDUMP') "
            "or k == 'CUDA_ENABLE_COREDUMP_ON_EXCEPTION']}))\n")
        out = subprocess.run([sys.executable, "-c", code], env=_clean_env({}), capture_output=True, text=True,
                             timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(res, {"r": None, "d": None, "env": []})


class _FakeDriver:
    def __init__(self, get_rc=0, enable=b"\x01", flags=0x3F, file=b"/x/cuda_coredump_h.1.2\0", raise_on=None):
        self.get_rc, self.enable, self.flags, self.file, self.raise_on = get_rc, enable, flags, file, raise_on
        self.sets = []

    def init(self):
        if self.raise_on == "init":
            raise RuntimeError("no libcuda")
        return 0

    def get(self, attrib, scope):
        if self.get_rc:
            return self.get_rc, b""
        return 0, {1: self.enable, 7: self.flags.to_bytes(4, "little"), 5: self.file}[attrib]

    def set(self, attrib, raw, scope):
        self.sets.append((attrib, raw, scope))
        if attrib == 1:
            self.enable = raw
        elif attrib == 7:
            self.flags = int.from_bytes(raw, "little")
        elif attrib == 5:
            self.file = raw
        return 0


class TheDriverReportNeverStopsARank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FLLIPER_CUDA_COREDUMP"] = "1"
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["FLLIPER_CUDA_COREDUMP_DIR"] = cls.tmp.name
        from flliper.srt.debug_utils import cuda_coredump as cc
        cls.cc = cc

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("FLLIPER_CUDA_COREDUMP", None)
        os.environ.pop("FLLIPER_CUDA_COREDUMP_DIR", None)
        cls.tmp.cleanup()

    def _log(self):
        log = logging.getLogger("t_nf_cudacore_1023")
        return log

    def test_an_unreadable_driver_is_a_named_error_not_an_exception(self):
        with self.assertLogs("t_nf_cudacore_1023", level="ERROR") as cm:
            rec = self.cc.driver_report("D TP1 PP0", self._log(), driver_factory=lambda: _FakeDriver(get_rc=999))
        self.assertEqual(rec["rc"], 999)
        self.assertIn("UNREADABLE", cm.output[0])

    def test_a_missing_libcuda_is_a_named_error_not_an_exception(self):
        with self.assertLogs("t_nf_cudacore_1023", level="ERROR") as cm:
            rec = self.cc.driver_report("D TP1 PP0", self._log(), driver_factory=lambda: _FakeDriver(raise_on="init"))
        self.assertIn("error", rec)
        self.assertIn("raised RuntimeError", cm.output[0])

    def test_a_driver_without_skip_abort_is_set_and_read_back(self):
        fake = _FakeDriver(flags=0x2F, file=(self.tmp.name + "/cuda_coredump_h.1.2\0").encode())
        rec = self.cc.driver_report("D TP1 PP0", self._log(), driver_factory=lambda: fake)
        self.assertTrue(rec["flags"] & 0x10 and rec["ok"], rec)
        self.assertTrue(rec["source"].startswith("api("))
        self.assertTrue(any(a == 7 for a, _, _ in fake.sets))

    def test_a_dump_directory_that_cannot_exist_is_not_armed(self):
        blocker = os.path.join(self.tmp.name, "a_file")
        open(blocker, "w").close()
        saved = dict(self.cc._IMPORT_ENV or {})
        self.cc._IMPORT_ENV = {"CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1", "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                               "CUDA_COREDUMP_GENERATION_FLAGS": LIGHT + ",skip_abort",
                               "CUDA_COREDUMP_FILE": os.path.join(blocker, "sub", "cuda_coredump_%h.%p.%t")}
        try:
            with self.assertLogs("t_nf_cudacore_1023", level="ERROR") as cm:
                rec = self.cc.log_rank_env("D TP2 PP0", self._log())
        finally:
            self.cc._IMPORT_ENV = saved or None
        self.assertFalse(rec["writable"])
        self.assertIn("CUDA-COREDUMP RANK-ENV NOT ARMED rank=D TP2 PP0", cm.output[0])

    def test_flag_names_round_trip(self):
        self.assertEqual(self.cc.decode_flags(0x3F), "skip_nonrelocated_elf_images,skip_global_memory,"
                         "skip_shared_memory,skip_local_memory,skip_abort,skip_constbank_memory")
        self.assertEqual(self.cc.encode_flags(LIGHT + ",skip_abort"), 0x3F)
        self.assertEqual(self.cc.decode_flags(0), "default")


class TheSkipAbortSwitch(unittest.TestCase):
    def test_on_by_default_and_off_leaves_the_flags_as_given(self):
        code = (
            "from flliper.srt.debug_utils import cuda_coredump as cc\n"
            "print(cc.generation_flags()); print(cc.generation_flags('a,skip_abort'))\n")
        on = subprocess.run([sys.executable, "-c", code], env=_clean_env({}), capture_output=True, text=True,
                            timeout=300)
        off = subprocess.run([sys.executable, "-c", code], env=_clean_env({"FLLIPER_CUDA_COREDUMP_SKIP_ABORT": "0"}),
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(on.returncode, 0, on.stderr[-2000:])
        self.assertEqual(on.stdout.split(), [LIGHT + ",skip_abort", "a,skip_abort"])
        self.assertEqual(off.stdout.split(), [LIGHT, "a,skip_abort"])

    def test_the_import_ensures_the_env_even_behind_the_sentinel(self):
        code = ("import json, os\nimport flliper.srt.environ\nfrom flliper.srt.debug_utils import cuda_coredump as cc\n"
                "print(json.dumps({'ensured': cc._ENSURED, 'flags': os.environ.get('CUDA_COREDUMP_GENERATION_FLAGS'), "
                "'file': os.environ.get('CUDA_COREDUMP_FILE'), 'import_env_flags': cc._IMPORT_ENV["
                "'CUDA_COREDUMP_GENERATION_FLAGS']}))\n")
        with tempfile.TemporaryDirectory() as tmp:
            env = _launcher_env(tmp)
            env.pop("CUDA_COREDUMP_FILE")
            out = subprocess.run([sys.executable, "-c", code], env=_clean_env(env), capture_output=True, text=True,
                                 timeout=300)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            res = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(res["ensured"], {"filled": ["CUDA_COREDUMP_FILE"], "skip_abort": "added"})
            self.assertEqual(res["flags"], LIGHT + ",skip_abort")
            self.assertEqual(res["file"], tmp + "/cuda_coredump_%h.%p.%t")
            self.assertEqual(res["import_env_flags"], LIGHT)  # the witness keeps what the process was spawned with


class TheSchedulerEntryLogsBothWitnesses(unittest.TestCase):
    """Source pin of run_scheduler_process: the RANK-ENV line after configure_scheduler_process (logger configured),
    the DRIVER line after the Scheduler is constructed (CUDA up) and before the init info goes to the parent."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(PY, "flliper", "srt", "managers", "scheduler.py")
        tree = ast.parse(open(path, encoding="utf-8").read())
        cls.fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_scheduler_process")

    def _call_line(self, attr):
        lines = [node.lineno for node in ast.walk(self.fn) if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == attr)
            or (isinstance(node.func, ast.Name) and node.func.id == attr))]
        return min(lines) if lines else None

    def test_the_rank_env_line_follows_the_logger_setup(self):
        line = self._call_line("log_rank_env")
        self.assertIsNotNone(line)
        self.assertLess(self._call_line("configure_scheduler_process"), line)

    def test_the_driver_report_follows_the_scheduler_and_precedes_the_init_info(self):
        rep = self._call_line("driver_report")
        self.assertIsNotNone(rep)
        self.assertLess(self._call_line("Scheduler"), rep)
        self.assertLess(rep, self._call_line("send"))


class TheCrashPathWaitsForAGrowingDump(unittest.TestCase):
    """rc31b: SIGQUIT 23:58:39, kill_process_tree 23:58:46. The launch_server crash path waits while a GPU exception
    dump of this boot is still growing, then names it -- the kill must not cut it."""

    @classmethod
    def setUpClass(cls):
        os.environ["FLLIPER_CUDA_COREDUMP"] = "1"
        from flliper.srt.debug_utils import cuda_coredump as cc
        cls.cc = cc

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("FLLIPER_CUDA_COREDUMP", None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = os.environ.get("CUDA_COREDUMP_FILE")
        os.environ["CUDA_COREDUMP_FILE"] = self.tmp.name + "/cuda_coredump_%h.%p.%t"
        self.now = [1000.0]

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("CUDA_COREDUMP_FILE", None)
        else:
            os.environ["CUDA_COREDUMP_FILE"] = self.saved
        self.tmp.cleanup()

    def _file(self, name, size, mtime):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as fh:
            fh.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def test_it_waits_until_the_dump_stops_growing_and_names_it(self):
        path = self._file("cuda_coredump_CT999.1418.1792", 10, 1000.0)
        grows = [3]

        def sleep(dt):
            self.now[0] += dt
            if grows[0] > 0:  # the rank is still writing: size and mtime move
                grows[0] -= 1
                with open(path, "ab") as fh:
                    fh.write(b"y" * 100)
                os.utime(path, (self.now[0], self.now[0]))

        with self.assertLogs("t_cc_wait", level="ERROR") as cm:
            files = self.cc.wait_for_exception_dumps(900.0, logging.getLogger("t_cc_wait"), quiet_s=2.0,
                                                     poll_s=0.5, clock=lambda: self.now[0], sleep=sleep)
        self.assertEqual([(p, s) for p, s, _ in files], [(path, 310)])
        self.assertGreaterEqual(self.now[0] - 1000.0, 1.5 + 2.0)  # three growth polls, then 2 s quiet
        self.assertIn("CUDA-COREDUMP FILE at crash: " + path + " size=310", cm.output[0])

    def test_the_cap_bounds_the_wait(self):
        path = self._file("cuda_coredump_CT999.1418.1792", 10, 1000.0)

        def sleep(dt):
            self.now[0] += dt
            os.utime(path, (self.now[0], self.now[0]))  # never stops growing

        with self.assertLogs("t_cc_wait", level="ERROR") as cm:
            self.cc.wait_for_exception_dumps(900.0, logging.getLogger("t_cc_wait"), cap_s=5.0, poll_s=1.0,
                                             clock=lambda: self.now[0], sleep=sleep)
        self.assertLessEqual(self.now[0] - 1000.0, 5.0 + 1.0)
        self.assertIn("CAP reached", cm.output[0])

    def test_an_old_dump_of_an_earlier_boot_is_not_this_boots(self):
        self._file("cuda_coredump_CT999.77.1", 10, 500.0)
        with self.assertLogs("t_cc_wait", level="ERROR") as cm:
            files = self.cc.wait_for_exception_dumps(900.0, logging.getLogger("t_cc_wait"),
                                                     clock=lambda: self.now[0], sleep=lambda dt: None)
        self.assertEqual(files, [])
        self.assertIn("none in " + self.tmp.name, cm.output[0])

    def test_the_tokenizer_manager_calls_it_between_the_settle_and_the_diagnostics(self):
        src = open(os.path.join(PY, "flliper", "srt", "managers", "tokenizer_manager.py"), encoding="utf-8").read()
        i_sleep = src.index("time.sleep(5)")
        i_wait = src.index("_cudacore.wait_for_exception_dumps(")
        i_procs = src.index("scheduler_procs = collect_scheduler_processes()")
        self.assertLess(i_sleep, i_wait)
        self.assertLess(i_wait, i_procs)


if __name__ == "__main__":
    unittest.main()
