# SPDX-License-Identifier: Apache-2.0
"""#236 -- the operator's per-card reserve (``--user-reserve-mib``) reaches the RANKS.

THE DEFECT (found by the #233 second review, pre-existing since the 27B base):
``corridor_guard.USER_RESERVE_ENV`` had ONE setter, the front's ``fenv`` in
``launcher.main``. ``build_env`` published it to no group, and every rank-side
reader of the floor (``vram_dial``, ``weight_updater``,
``corridor_floor_for_current_device``) reads the reserve only from that env.
So with ``--user-reserve-mib`` the launcher priced ``floor = transient +
reserve`` and every rank priced ``floor = transient + 0`` -- the OOM direction.

THE FIX under test: ``build_env`` hands the mapping to every group (P and D,
host and worker of any form inherit the group env), keyed by card UUID, and a
rank logs where it read it from.

HERMETIC: no NVML, no torch, no GPU, no boot. The rank's card is injected the
way ``registry.nvml.current_device_uuid`` would answer it.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
import unittest
from dataclasses import dataclass
from unittest import mock

from flliper.srt.managers import corridor_guard as cg
from flliper.srt.pdflip import corridor_budget as cb
from flliper.srt.pdflip import launcher
from flliper.srt.registry import nvml as registry_nvml

C5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
C3080A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
C3080B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
#: The standing user form: 5090 1800 / 3080s 1400 (launcher.parse_user_reserve).
RESERVE = {C5090: 1800, C3080A: 1400, C3080B: 1400}


@dataclass
class _Card:
    uuid: str
    name: str
    nvml_index: int
    total_mib: int
    reserved_mib: int = 0


CARDS = [
    _Card(C5090, "5090", 1, 32607),
    _Card(C3080A, "3080a", 0, 20480),
    _Card(C3080B, "3080b", 2, 20480),
]


def _env(group, reserve=None):
    return launcher.build_env(
        tree="/tmp/t",
        venv="/tmp/v",
        cvd="0",
        store_dir="/tmp/s",
        debug_hold=False,
        tag="probe",
        group=group,
        user_reserve_by_card=reserve,
    )


def _check_group_env_carries_the_reserve():
    """The assertion the mutant must break (shared by test and mutant test)."""
    for group in ("P", "D"):
        env = _env(group, RESERVE)
        raw = env.get(cg.USER_RESERVE_ENV)
        assert raw, f"group {group} env carries no {cg.USER_RESERVE_ENV}"
        assert {k: int(v) for k, v in json.loads(raw).items()} == RESERVE, (
            f"group {group} got {raw!r}, expected the per-card UUID mapping {RESERVE}"
        )


class GroupEnvCarriesTheReserve(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(cg.USER_RESERVE_ENV, None)

    def test_t1_group_env_contains_the_reserve_per_card_uuid_for_p_and_d(self):
        _check_group_env_carries_the_reserve()

    def test_t1b_the_keys_are_card_uuids_not_indices(self):
        raw = _env("D", RESERVE)[cg.USER_RESERVE_ENV]
        for key in json.loads(raw):
            self.assertTrue(key.startswith("GPU-"), f"index-shaped key {key!r}")

    def test_t2_group_and_front_share_one_serialiser(self):
        self.assertEqual(
            _env("P", RESERVE)[cg.USER_RESERVE_ENV],
            launcher.user_reserve_env_value(RESERVE),
        )
        src = inspect.getsource(launcher.main)
        self.assertIn("user_reserve_env_value(user_reserve_by_card)", src)

    def test_t3_every_build_env_call_site_in_main_passes_the_reserve(self):
        src = inspect.getsource(launcher)
        calls = [ln for ln in src.splitlines() if "= build_env(" in ln]
        self.assertGreaterEqual(len(calls), 3)
        for ln in calls:
            self.assertIn(
                "user_reserve_by_card=user_reserve_by_card",
                ln,
                f"a launch site does not hand the reserve to its group: {ln.strip()[:120]}",
            )

    def test_t4_without_a_stated_reserve_nothing_is_published(self):
        baseline = {g: _env(g) for g in ("P", "D", "")}
        for group, env in baseline.items():
            self.assertNotIn(cg.USER_RESERVE_ENV, env)
        # parse_user_reserve("0", cards) -> {uuid: 0}: "not stated", must not
        # become "reserve 0 stated" on a rank (it would beat the argv scalar).
        zeros = {c.uuid: 0 for c in CARDS}
        for group in ("P", "D", ""):
            self.assertEqual(_env(group, zeros), baseline[group])
            self.assertEqual(_env(group, {}), baseline[group])

    def test_t4b_a_value_inherited_from_a_shell_is_scrubbed(self):
        os.environ[cg.USER_RESERVE_ENV] = '{"GPU-leftover": 99999}'
        self.assertNotIn(cg.USER_RESERVE_ENV, _env("D"))
        self.assertEqual(
            json.loads(_env("D", RESERVE)[cg.USER_RESERVE_ENV]),
            RESERVE,
        )

    def test_t4c_the_parser_form_without_the_flag_publishes_nothing(self):
        stated = launcher.parse_user_reserve("0", CARDS)
        self.assertEqual(launcher.user_reserve_rank_env(stated), {})
        stated = launcher.parse_user_reserve("1800,1400,1400", CARDS)
        self.assertEqual(
            json.loads(launcher.user_reserve_rank_env(stated)[cg.USER_RESERVE_ENV]),
            {CARDS[0].uuid: 1800, CARDS[1].uuid: 1400, CARDS[2].uuid: 1400},
        )

    def test_t5_mutant_group_without_reserve_is_red(self):
        """Mutant: build_env stops publishing the reserve to the group."""
        _check_group_env_carries_the_reserve()  # green on the real code
        with mock.patch.object(launcher, "user_reserve_rank_env", lambda _d: {}):
            with self.assertRaises(AssertionError):
                _check_group_env_carries_the_reserve()


class RankFloorEqualsLauncherFloor(unittest.TestCase):
    """ONE truth: the floor a rank derives from ITS env equals the floor the
    launcher priced for that card with the parsed reserve."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {cg.FLOOR_DIGEST_FILE_ENV: os.path.join(self._tmp.name, "absent.json")},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (cg.LAW_ENV, cg.USER_RESERVE_ENV):
            os.environ.pop(k, None)
        cg._RESERVE_READ_NOTED.clear()
        self.addCleanup(cg._RESERVE_READ_NOTED.clear)

    def _rank_floor(self, uuid, group, rank_env, fallback=0):
        with mock.patch.dict(os.environ, rank_env), mock.patch.object(
            registry_nvml, "current_device_uuid", lambda: uuid
        ):
            return cg.corridor_floor_for_current_device(
                group=group, fallback_reserve_mib=fallback
            )

    def test_t6_rank_floor_equals_launcher_floor_with_a_stated_reserve(self):
        for group in ("P", "D"):
            launcher_floors = cb.floors_for_cards(CARDS, group, RESERVE)
            rank_env = {
                cg.USER_RESERVE_ENV: _env(group, RESERVE)[cg.USER_RESERVE_ENV]
            }
            for c in CARDS:
                rank = self._rank_floor(c.uuid, group, rank_env)
                self.assertEqual(
                    int(rank.mib), int(launcher_floors[c.uuid].mib), (group, c.uuid)
                )
                self.assertEqual(
                    int(rank.reserve_mib), RESERVE[c.uuid], (group, c.uuid)
                )
                self.assertEqual(
                    int(rank.reserve_mib),
                    int(launcher_floors[c.uuid].reserve_mib),
                    (group, c.uuid),
                )

    def test_t6b_the_reserve_is_per_card_not_rig_uniform(self):
        env = {cg.USER_RESERVE_ENV: _env("D", RESERVE)[cg.USER_RESERVE_ENV]}
        r5090 = self._rank_floor(C5090, "D", env)
        r3080 = self._rank_floor(C3080A, "D", env)
        self.assertEqual(int(r5090.mib) - int(r3080.mib), 1800 - 1400)

    def test_t7_mutant_rank_without_the_env_prices_transient_plus_zero(self):
        """The pre-fix state: same card, same law, NO env -> the floor is lower
        by exactly the reserve (the OOM direction). The fixed path is not."""
        with_env = self._rank_floor(
            C5090, "D", {cg.USER_RESERVE_ENV: _env("D", RESERVE)[cg.USER_RESERVE_ENV]}
        )
        without_env = self._rank_floor(C5090, "D", {})
        self.assertEqual(int(with_env.mib) - int(without_env.mib), 1800)
        launcher_floor = cb.floors_for_cards(CARDS, "D", RESERVE)[C5090]
        self.assertEqual(int(with_env.mib), int(launcher_floor.mib))
        self.assertNotEqual(int(without_env.mib), int(launcher_floor.mib))

    def test_t8_without_a_stated_reserve_the_rank_floor_is_unchanged(self):
        """No --user-reserve-mib: the rank env has no mapping, the rank prices
        transient + its argv-scalar fallback exactly as before #236."""
        for c in CARDS:
            env = _env("D", {x.uuid: 0 for x in CARDS})
            self.assertNotIn(cg.USER_RESERVE_ENV, env)
            base = self._rank_floor(c.uuid, "D", {})
            self.assertEqual(int(base.reserve_mib), 0)
            scalar = self._rank_floor(c.uuid, "D", {}, fallback=512)
            self.assertEqual(int(scalar.reserve_mib), 512)


class TheRankSaysWhereItReadTheReserve(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(cg.USER_RESERVE_ENV, None)
        cg._RESERVE_READ_NOTED.clear()
        self.addCleanup(cg._RESERVE_READ_NOTED.clear)

    def test_t9_a_named_line_with_source_and_value(self):
        os.environ[cg.USER_RESERVE_ENV] = json.dumps(RESERVE)
        with self.assertLogs(cg.logger, level=logging.INFO) as cm:
            self.assertEqual(cg.user_reserve_by_card(), RESERVE)
            cg.user_reserve_by_card()  # second call: no second line
        lines = [m for m in cm.output if "PDFLIP-USER-RESERVE-READ" in m]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn(f"source=env:{cg.USER_RESERVE_ENV}", lines[0])
        self.assertIn(str(RESERVE[C5090]), lines[0])
        self.assertIn(C5090, lines[0])

    def test_t9b_an_absent_variable_is_named_too(self):
        with self.assertLogs(cg.logger, level=logging.INFO) as cm:
            self.assertEqual(cg.user_reserve_by_card(), {})
        lines = [m for m in cm.output if "PDFLIP-USER-RESERVE-READ" in m]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("source=none", lines[0])

    def test_t9c_the_card_line_names_uuid_reserve_and_source(self):
        os.environ[cg.USER_RESERVE_ENV] = json.dumps(RESERVE)
        os.environ[cg.FLOOR_DIGEST_FILE_ENV] = "/nonexistent/d.json"
        with mock.patch.object(registry_nvml, "current_device_uuid", lambda: C3080A):
            with self.assertLogs(cg.logger, level=logging.INFO) as cm:
                cg.corridor_floor_for_current_device(group="D")
        card_lines = [m for m in cm.output if f"card={C3080A}" in m]
        self.assertEqual(len(card_lines), 1, cm.output)
        self.assertIn("reserve_mib=1400", card_lines[0])
        self.assertIn("source=env-by-card", card_lines[0])


if __name__ == "__main__":
    unittest.main()
