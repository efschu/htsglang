"""MAMBA-ARENA-FLOOR 1016: the ONE shared mamba arena gets at least the sum of both groups' floors
(device_slots + max_running + 1 each); a lower slot count (profile env, launcher default) is raised in both
group envs before the ledger prices it.

THE DEFECT (live, boot 1016_124240). The NF profiles set FLLIPER_HICACHE_ARENA_MAMBA_SLOTS=32 (``_form`` and both
NF_ENV_*_FORM strings), over the launcher default 140 and over the ledger's price. The arena every rank of P and
D shares held 32 canonical blobs (58834944 B each): ARENA-REF-CENSUS complete=32 of 32, and D lost anchors
``at=evict why=mamba_claim backuped=False`` (n up to 9 per line). The floors: P 32 + 6 + 1 = 39, D 38 + 6 + 1 = 45,
together 84; 32 is under D's floor alone.

    CUDA_VISIBLE_DEVICES='' python -m pytest test/registered/unit/pdflip/test_pdflip_mamba_arena_floor_1016.py -q
"""

import ast
import inspect
import types
import unittest
from unittest import mock

from flliper.srt.pdflip import launcher as L

#: the live NF profile's group argv (nf-int4-h6-abl-xc-lanes.env, --extra-p / --extra-d), shortened
EXTRA_P = "--page-size 64 --max-running-requests 6 --hicache-size 2 --max-mamba-cache-size 32"
EXTRA_D = "--page-size 64 --max-running-requests 6 --hicache-size 4"
ENV_LIVE = "FLLIPER_HICACHE_ARENA_GIB=4.75;FLLIPER_HICACHE_ARENA_MAMBA_SLOTS=32;FLLIPER_X=1"


def _main_text() -> str:
    """``launcher.main`` as it read before the split into ``_main_stepNN_*`` functions: every step call replaced by the step body, in call order
    (behaviour-neutral refactor, 28.10.; see ``_launcher_source.expanded_main_source``)."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_main_source()


def _ns(env_p=ENV_LIVE, env_d=ENV_LIVE, extra_p=EXTRA_P, extra_d=EXTRA_D, **kw):
    return types.SimpleNamespace(env_p=env_p, env_d=env_d, extra_p=extra_p, extra_d=extra_d, **kw)


def _envs(ns):
    return [L.parse_group_env(ns.env_d), L.parse_group_env(ns.env_p)]


class TheLiveCase(unittest.TestCase):
    def test_32_is_raised_to_84_in_both_group_envs_and_the_ledger_dicts(self):
        """RED before the fix: nothing touched the profile's 32."""
        ns = _ns()
        envs = _envs(ns)
        line = L.mamba_arena_slots_into_groups(ns, envs)
        self.assertIn("PDFLIP-MAMBA-ARENA-FLOOR RAISED", line)
        self.assertIn("32 -> 84", line)
        for spec in (ns.env_p, ns.env_d):
            self.assertEqual(L.parse_group_env(spec)["FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"], "84")
            self.assertEqual(L.parse_group_env(spec)["FLLIPER_X"], "1")  # every other entry kept
        self.assertEqual([e["FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"] for e in envs], ["84", "84"])
        self.assertIn("never refuses", line)

    def test_the_line_names_both_groups_terms(self):
        line = L.mamba_arena_slots_into_groups(_ns(), _envs(_ns()))
        self.assertIn("P: device 32 [--max-mamba-cache-size] + running 6 + 1 = 39", line)
        self.assertIn("D: device 38 [demand upper bound ceil(6 x 5 x 1.25)] + running 6 + 1 = 45", line)

    def test_the_d_upper_bound_is_the_measured_demand_pool(self):
        """D 1016_103428: target_concurrency=6 ratio=5 safety=1.25 -> max_mamba_cache_size=38."""
        self.assertEqual(L.mamba_arena_group_floor(EXTRA_D, None), (45, 38, 6, "demand upper bound ceil(6 x 5 x 1.25)"))


class TheOtherCases(unittest.TestCase):
    def test_a_count_over_the_floor_is_kept(self):
        env = "FLLIPER_HICACHE_ARENA_MAMBA_SLOTS=140"
        ns = _ns(env_p=env, env_d=env)
        line = L.mamba_arena_slots_into_groups(ns, _envs(ns))
        self.assertTrue(line.startswith("PDFLIP-MAMBA-ARENA-FLOOR ok"), line)
        self.assertEqual((ns.env_p, ns.env_d), (env, env))

    def test_nobody_names_a_count_so_the_floor_is_the_count(self):
        """No profile value: the arena gets the computed floor, not the fixed launcher default 140."""
        ns = _ns(env_p="", env_d="")
        envs = _envs(ns)
        with mock.patch.dict(L.os.environ, {}, clear=False):
            L.os.environ.pop("FLLIPER_HICACHE_ARENA_MAMBA_SLOTS", None)
            line = L.mamba_arena_slots_into_groups(ns, envs)
        self.assertTrue(line.startswith("PDFLIP-MAMBA-ARENA-FLOOR COMPUTED"), line)
        self.assertEqual(L.parse_group_env(ns.env_p)["FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"], "84")
        self.assertEqual([e["FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"] for e in envs], ["84", "84"])

    def test_d_only_counts_d_alone(self):
        ns = _ns(d_only=True)
        line = L.mamba_arena_slots_into_groups(ns, _envs(ns))
        self.assertIn("32 -> 45", line)
        self.assertNotIn("P:", line)

    def test_unknown_concurrency_is_named_and_changes_nothing(self):
        ns = _ns(extra_p="--page-size 64", extra_d="--page-size 64")
        line = L.mamba_arena_slots_into_groups(ns, _envs(ns))
        self.assertTrue(line.startswith("PDFLIP-MAMBA-ARENA-FLOOR skipped"), line)
        self.assertEqual(ns.env_p, ENV_LIVE)

    def test_the_last_flag_value_counts_like_argparse(self):
        self.assertEqual(L.mamba_arena_group_floor("--max-running-requests 4 --max-running-requests 6", None)[2], 6)


class TheLauncherRunsItBeforeTheLedger(unittest.TestCase):
    def test_main_raises_before_the_ledger_prices(self):
        src = _main_text()
        i = src.index("log(mamba_arena_slots_into_groups(ns, _ledger_group_envs))")
        self.assertLess(src.index("_ledger_group_envs = [parse_group_env("), i)
        self.assertLess(i, src.index("def _price_host_ledger():"))
        ast.parse(inspect.getsource(L.mamba_arena_slots_into_groups))


if __name__ == "__main__":
    unittest.main()
