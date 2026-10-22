"""rc32 (22.10.): the NF boot profile switches the upstream CUDA coredump on (FLLIPER_CUDA_COREDUMP=1,
FLLIPER_CUDA_COREDUMP_DIR=<evidence>/cudacore), because the rc31b rank death (illegal memory access on D-TP1,
deskq/done/nf-rc31b-ima-1022.md) has no proven root and the next one must name the faulting kernel.

The instrument is only armed if the CUDA_* variables reach every rank process BEFORE its CUDA init. These tests
pin the chain that makes that true without a GPU:

1. importing ``flliper.srt.environ`` (the launcher does it at module level) injects CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1,
   the lightweight skip_* generation flags and the dump file pattern into ``os.environ`` -- and it does so without
   initialising CUDA;
2. a child spawned with ``dict(os.environ)`` (launcher build_env / front env) inherits them, and the launcher's
   ``canonical_env`` keeps them;
3. the old profile spelling ``SGLANG_CUDA_COREDUMP(_DIR)`` (the rig's source profiles) converts to the FLLIPER names
   via tools/release/profconv.py;
4. switched off (the code default), nothing is injected.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PY = os.path.join(ROOT, "python")

_PROBE = r"""
import json, os, subprocess, sys
import flliper.srt.environ  # noqa: F401  (injects at import when FLLIPER_CUDA_COREDUMP=1)
torch_cuda_init = None
if "torch" in sys.modules:
    import torch
    torch_cuda_init = bool(torch.cuda.is_initialized())
from flliper.srt.name_compat import canonical_env
child = subprocess.run([sys.executable, "-c",
    "import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('CUDA_')}))"],
    env=dict(os.environ), capture_output=True, text=True, check=True).stdout.strip()
canon = canonical_env(dict(os.environ))
print(json.dumps({
    "parent": {k: v for k, v in os.environ.items() if k.startswith("CUDA_")},
    "child": json.loads(child),
    "canon": {k: v for k, v in canon.items() if k.startswith("CUDA_")},
    "torch_cuda_init": torch_cuda_init,
}))
"""


def _run(extra_env):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CUDA_COREDUMP") and k not in (
               "CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_ENABLE_LIGHTWEIGHT_COREDUMP",
               "FLLIPER_CUDA_COREDUMP", "SGLANG_CUDA_COREDUMP", "FLLIPER_CUDA_COREDUMP_DIR",
               "SGLANG_CUDA_COREDUMP_DIR", "_FLLIPER_CUDA_COREDUMP_INJECTED")}
    env["PYTHONPATH"] = PY
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(extra_env)
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class TheProfileSwitchArmsEveryRank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dump_dir = os.path.join(cls.tmp.name, "evidence", "cudacore")
        cls.res = _run({"FLLIPER_CUDA_COREDUMP": "1", "FLLIPER_CUDA_COREDUMP_DIR": cls.dump_dir})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_import_injects_the_coredump_variables(self):
        p = self.res["parent"]
        self.assertEqual(p.get("CUDA_ENABLE_COREDUMP_ON_EXCEPTION"), "1")
        self.assertEqual(p.get("CUDA_COREDUMP_GENERATION_FLAGS"),
                         "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,"
                         "skip_local_memory,skip_constbank_memory")
        self.assertEqual(p.get("CUDA_COREDUMP_FILE"), self.dump_dir + "/cuda_coredump_%h.%p.%t")

    def test_the_dump_directory_exists_on_the_evidence_path(self):
        self.assertTrue(os.path.isdir(self.dump_dir))

    def test_the_injection_does_not_initialise_cuda(self):
        # None = torch not even imported by environ; False = imported but no CUDA context. Both are "before CUDA init".
        self.assertIn(self.res["torch_cuda_init"], (None, False))

    def test_a_child_with_the_launcher_env_inherits_them(self):
        for key in ("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_COREDUMP_GENERATION_FLAGS", "CUDA_COREDUMP_FILE"):
            self.assertEqual(self.res["child"].get(key), self.res["parent"][key], key)
            self.assertEqual(self.res["canon"].get(key), self.res["parent"][key], key)


class TheOldProfileSpellingConverts(unittest.TestCase):
    def test_sglang_names_become_the_flliper_names(self):
        sys.path.insert(0, os.path.join(ROOT, "tools", "release"))
        try:
            import profconv as P
        finally:
            sys.path.pop(0)
        imap = P.R._load_imap(os.path.join(P.KIT, "data", "merged_0928.json"))
        src = ('profile_form_env() {\n  _form SGLANG_CUDA_COREDUMP 1\n'
               '  _form SGLANG_CUDA_COREDUMP_DIR "${SGLANG_WEG2_EVIDENCE_DIR:-/var/lib/htsglang/evidence}/cudacore"\n}\n')
        new = P.convert(src, imap)
        self.assertIn("_form FLLIPER_CUDA_COREDUMP 1\n", new)
        self.assertIn('_form FLLIPER_CUDA_COREDUMP_DIR "${FLLIPER_PDFLIP_EVIDENCE_DIR:-/var/lib/htsglang/evidence}/cudacore"', new)
        self.assertEqual(P.old_name_lines(new), [])


class TheLauncherImportsEnvironAtModuleLevel(unittest.TestCase):
    def test_environ_is_a_top_level_import_of_the_launcher(self):
        path = os.path.join(PY, "flliper", "srt", "pdflip", "launcher.py")
        tree = ast.parse(open(path, encoding="utf-8").read())
        top = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "flliper.srt.environ"]
        self.assertTrue(top, "launcher.py must import flliper.srt.environ at module level (injection before any group spawn)")


class SwitchedOffNothingIsInjected(unittest.TestCase):
    def test_the_code_default_leaves_the_environment_alone(self):
        res = _run({})
        self.assertNotIn("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", res["parent"])
        self.assertNotIn("CUDA_COREDUMP_FILE", res["parent"])


if __name__ == "__main__":
    unittest.main()
