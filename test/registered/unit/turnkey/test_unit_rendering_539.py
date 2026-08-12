# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The #539 units must follow [stack].repo, not a path frozen into the file.

WHY, measured on this rig 2026-08-12: all five units hardcoded
``/spinning/htsglang-gpu`` for PYTHONPATH and for the interpreter, regardless
of what ``[stack].repo`` said. That checkout predates the turnkey merge, so
every unit died with ``No module named sglang.srt.turnkey`` and the serving
unit died on the dependency. ``[stack].repo`` READS as the single source of
truth and is not one, and the divergence is invisible until the import error.

THE FALSIFIER is ``test_a_foreign_repo_is_actually_written``: render against a
config whose repo is somewhere else entirely and assert the WRITTEN unit names
that repo, that the interpreter and WorkingDirectory follow it, and that
``/spinning/htsglang-gpu`` is gone from every directive.

Hermetic: rendering writes into a temp dir. Nothing under /etc is read for a
decision or written, and no unit is installed, enabled or started.
"""

import importlib.util
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_TOOL = _ROOT / "scripts" / "turnkey_539_render_units.py"
_SRC = _ROOT / "deploy" / "turnkey"

UNITS = ("htsglang.target", "htsglang-preflight.service",
         "htsglang-planner.service", "htsglang-serving@.service",
         "htsglang-watchdog@.service")

#: Directive lines whose paths must follow the config. Comments are exempt:
#: they cite real incidents on real paths and are not what systemd executes.
_PATH_DIRECTIVES = ("Environment", "ExecStart", "ExecStartPre", "ExecStop",
                    "WorkingDirectory", "Documentation", "EnvironmentFile")


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "turnkey_539_render_units", str(_TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RU = _load_tool()

FOREIGN = """
[stack]
name = "foreign"
repo = "/opt/elsewhere/htsglang"
venv = "/opt/elsewhere/venv"
log_dir = "/var/log/elsewhere"

[[cards]]
uuid = "GPU-11111111-1111-1111-1111-111111111111"

[serving.ship]
port = 30030
cards = [0]
boot_log = "/var/log/elsewhere/ship.boot.log"
argv = ["/opt/elsewhere/venv/bin/python", "-m", "sglang.launch_server"]
"""


def _directive_lines(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or "=" not in s:
            continue
        key = s.split("=", 1)[0]
        if key in _PATH_DIRECTIVES:
            yield s


class TestUnitsCarryPlaceholders(CustomTestCase):
    """A unit that still holds a literal path cannot follow the config."""

    def test_no_directive_names_a_literal_stack_path(self):
        offenders = []
        for u in UNITS:
            text = (_SRC / u).read_text()
            for line in _directive_lines(text):
                if "/spinning/" in line or "/var/log/htsglang" in line:
                    offenders.append(f"{u}: {line}")
        self.assertFalse(
            offenders,
            "these directives freeze a path that [stack].repo is supposed to "
            "decide:\n  " + "\n  ".join(offenders))

    def test_every_placeholder_used_is_one_the_renderer_knows(self):
        known = set(RU.PLACEHOLDERS)
        for u in UNITS:
            for ph in re.findall(r"@@[A-Z_]+@@", (_SRC / u).read_text()):
                self.assertIn(ph, known, f"{u} uses unknown {ph}")


class TestRendering(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conf = os.path.join(self.tmp.name, "stack.toml")
        with open(self.conf, "w") as fh:
            fh.write(FOREIGN)
        self.dst = os.path.join(self.tmp.name, "units")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_foreign_repo_is_actually_written(self):
        """THE FALSIFIER. Render against a repo that is not the default and
        read back what landed on disk."""
        RU.render_tree(str(_SRC), self.dst, RU.substitutions_from_config(
            self.conf))
        for u in UNITS:
            text = pathlib.Path(self.dst, u).read_text()
            for line in _directive_lines(text):
                self.assertNotIn("/spinning/htsglang-gpu", line,
                                 f"{u} still names the default checkout")
                self.assertNotIn("@@", line, f"{u} left a placeholder: {line}")

    def test_pythonpath_follows_the_configured_repo(self):
        RU.render_tree(str(_SRC), self.dst, RU.substitutions_from_config(
            self.conf))
        text = pathlib.Path(self.dst, "htsglang-serving@.service").read_text()
        self.assertIn("Environment=PYTHONPATH=/opt/elsewhere/htsglang/python",
                      text)

    def test_the_interpreter_follows_the_configured_venv(self):
        RU.render_tree(str(_SRC), self.dst, RU.substitutions_from_config(
            self.conf))
        for u in ("htsglang-serving@.service", "htsglang-watchdog@.service",
                  "htsglang-preflight.service", "htsglang-planner.service"):
            text = pathlib.Path(self.dst, u).read_text()
            execs = [l for l in text.splitlines()
                     if l.startswith("ExecStart=")]
            self.assertTrue(execs, f"{u} has no ExecStart")
            self.assertTrue(
                execs[0].startswith("ExecStart=/opt/elsewhere/venv/bin/python"),
                f"{u}: {execs[0]}")

    def test_working_directory_follows_the_configured_repo(self):
        RU.render_tree(str(_SRC), self.dst, RU.substitutions_from_config(
            self.conf))
        text = pathlib.Path(self.dst, "htsglang-planner.service").read_text()
        self.assertIn("WorkingDirectory=/opt/elsewhere/htsglang", text)

    def test_the_log_dir_follows_the_config(self):
        RU.render_tree(str(_SRC), self.dst, RU.substitutions_from_config(
            self.conf))
        text = pathlib.Path(self.dst, "htsglang-preflight.service").read_text()
        self.assertIn("/var/log/elsewhere", text)

    def test_rendering_the_rig_config_reproduces_the_shipped_paths(self):
        """The rig's own config must render to what the units said before
        this change: a refactor that also moves the rig is two changes."""
        subs = RU.substitutions_from_config(str(_SRC / "stack.rig3.toml"))
        RU.render_tree(str(_SRC), self.dst, subs)
        text = pathlib.Path(self.dst, "htsglang-serving@.service").read_text()
        self.assertIn("Environment=PYTHONPATH=/spinning/htsglang-gpu/python",
                      text)
        self.assertIn(
            "ExecStart=/spinning/htsglang-gpu/.venv/bin/python "
            "-m sglang.srt.turnkey --config /etc/htsglang/stack.toml boot %i",
            text)

    def test_an_unresolved_placeholder_is_refused_not_written(self):
        """A renderer that writes @@REPO@@ into /etc has produced a unit that
        fails at start with a path nobody can grep for."""
        with self.assertRaises(RU.RenderError) as cm:
            RU.render_text("ExecStart=@@NOPE@@/bin/python", {"REPO": "/x"},
                           source="u.service")
        self.assertIn("@@NOPE@@", str(cm.exception))
        self.assertIn("u.service", str(cm.exception))

    def test_a_config_without_repo_is_refused(self):
        bad = os.path.join(self.tmp.name, "bad.toml")
        with open(bad, "w") as fh:
            fh.write('[stack]\nname = "x"\n')
        with self.assertRaises(Exception):
            RU.substitutions_from_config(bad)


class TestTheInstallerWritesRenderedUnits(CustomTestCase):
    """The falsifier through the shell wiring, not only the Python helper.

    Runs the real installer with --apply into a TEMP unit dir. Nothing under
    /etc is read for a decision or written, no unit is enabled or started, and
    the installer skips daemon-reload because the unit dir is not systemd's.
    """

    def test_the_installer_writes_the_configured_repo_into_etc_style_units(
            self):
        with tempfile.TemporaryDirectory() as tmp:
            conf_dir = os.path.join(tmp, "conf")
            unit_dir = os.path.join(tmp, "units")
            os.makedirs(conf_dir)
            os.makedirs(unit_dir)
            cfg = os.path.join(tmp, "foreign.toml")
            with open(cfg, "w") as fh:
                fh.write(FOREIGN)
            env = dict(os.environ)
            env.update({"CONFIG": cfg, "UNIT_DIR": unit_dir,
                        "CONF_DIR": conf_dir, "LOG_DIR": os.path.join(tmp,
                                                                     "log")})
            p = subprocess.run(
                ["/bin/bash",
                 str(_ROOT / "scripts" / "turnkey_539_install.sh"), "--apply"],
                env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            for u in UNITS:
                text = pathlib.Path(unit_dir, u).read_text()
                for line in _directive_lines(text):
                    self.assertNotIn("/spinning/htsglang-gpu", line,
                                     f"{u} was installed with the default "
                                     f"checkout: {line}")
                    self.assertNotIn("@@", line)
            serving = pathlib.Path(unit_dir,
                                   "htsglang-serving@.service").read_text()
            self.assertIn("PYTHONPATH=/opt/elsewhere/htsglang/python", serving)
            self.assertIn("ExecStart=/opt/elsewhere/venv/bin/python", serving)
            self.assertIn(f"--config {conf_dir}/stack.toml", serving)


class TestInstallerUsesTheRenderer(CustomTestCase):
    def test_the_installer_no_longer_copies_units_byte_for_byte(self):
        text = (_ROOT / "scripts" / "turnkey_539_install.sh").read_text()
        self.assertIn("turnkey_539_render_units.py", text)

    def test_the_installer_verifies_the_rendered_units(self):
        """systemd-analyze verify on a file full of @@REPO@@ proves nothing
        about what gets installed."""
        text = (_ROOT / "scripts" / "turnkey_539_install.sh").read_text()
        m = re.search(r"systemd-analyze verify \"([^\"]+)\"", text)
        self.assertIsNotNone(m, "no systemd-analyze verify call found")
        self.assertNotIn("$SRC", m.group(1),
                         "verify must run on the RENDERED unit, not the "
                         "placeholder source")


if __name__ == "__main__":
    unittest.main()
