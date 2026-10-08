"""RIG-RAM GUARD (02.10.): y7e died (W3, 09:45Z) because uncapped agent pytest
runs in the agent LXC pushed the host to 2.2 GiB MemAvailable during an NF boot.
While a rig boot is live, a pytest outside a <= 4 GiB cgroup refuses by name."""

import json
import unittest

from flliper.test import rig_ram_guard as g

NOW = 1790934321.0


def _fs(state=None, cgroup="0::/system.slice/claude.service\n", maxes=None):
    files = {}
    if state is not None:
        files["/nf/current/state.json"] = json.dumps(state)
    files["/proc/self/cgroup"] = cgroup
    for k, v in (maxes or {}).items():
        files["/sys/fs/cgroup" + k + "/memory.max"] = v
    return files.get


def _boot(state="serving", age=5.0):
    return {"boot_id": "nfint4dauer-boot-20261002T092630Z-ab53",
            "lifecycle": {"state": state},
            "heartbeat": {"front": {"ts": NOW - age}}}


def _v(read, env=None):
    return g.verdict(env=env or {}, read=read, now=NOW, roots=("/nf",))


class TheGuard(unittest.TestCase):
    def test_uncapped_during_a_live_boot_is_refused(self):
        why = _v(_fs(_boot(), maxes={"/system.slice/claude.service": "max"}))
        self.assertIn("RIG-RAM GUARD", why)
        self.assertIn("nfint4dauer-boot-20261002T092630Z-ab53", why)

    def test_incg_cap_runs(self):
        read = _fs(_boot(), cgroup="0::/agent-tests-1\n", maxes={"/agent-tests-1": str(4 << 30)})
        self.assertIsNone(_v(read))

    def test_a_too_large_cap_is_not_a_cap(self):
        read = _fs(_boot(), cgroup="0::/agent-tests\n", maxes={"/agent-tests": str(16 << 30)})
        self.assertIsNotNone(_v(read))

    def test_a_parent_cap_counts(self):
        read = _fs(_boot(), cgroup="0::/agent-tests/sub\n",
                   maxes={"/agent-tests/sub": "max", "/agent-tests": str(4 << 30)})
        self.assertIsNone(_v(read))

    def test_no_live_boot_runs(self):
        self.assertIsNone(_v(_fs(None)))
        self.assertIsNone(_v(_fs(_boot(state="stopped_clean"))))
        self.assertIsNone(_v(_fs(_boot(age=g.HEARTBEAT_S + 60))))

    def test_explicit_override(self):
        self.assertIsNone(_v(_fs(_boot()), env={"RIG_RAM_GUARD": "off"}))

    def test_wired_into_the_tree_conftest(self):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[3] / "conftest.py").read_text()
        self.assertIn("rig_ram_guard.py", src)
        self.assertIn("pytest.exit(why, returncode=3)", src)


if __name__ == "__main__":
    unittest.main()
