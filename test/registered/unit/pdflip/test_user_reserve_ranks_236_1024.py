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

from types import SimpleNamespace

from flliper.srt.managers import corridor_guard as cg
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import corridor_budget as cb
from flliper.srt.pdflip import launcher
from flliper.srt.registry import nvml as registry_nvml

C5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
C3080A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
C3080B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
#: The standing user form: 5090 1800 / 3080s 1400 (launcher.parse_user_reserve).
RESERVE = {C5090: 1800, C3080A: 1400, C3080B: 1400}


def _launcher_text() -> str:
    """The launcher source as one text (launcher.py + launcher_*.py) with the split functions expanded back to the original statement
    order: the launcher was split behaviour-neutrally (refactor 28.10.), see ``_launcher_source.expanded_launcher_text``."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_launcher_text()


def _main_text() -> str:
    """``launcher.main`` as it read before the split into ``_main_stepNN_*`` functions: every step call replaced by the step body, in call order
    (behaviour-neutral refactor, 28.10.; see ``_launcher_source.expanded_main_source``)."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_main_source()


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
        src = _main_text()
        self.assertIn("user_reserve_env_value(user_reserve_by_card)", src)

    def test_t3_every_build_env_call_site_in_main_passes_the_reserve(self):
        src = _launcher_text()
        calls = [ln for ln in src.splitlines() if "= build_env(" in ln]
        self.assertGreaterEqual(len(calls), 3)
        for ln in calls:
            self.assertIn(
                "**rank_reserve_kwargs(",
                ln,
                f"a launch site does not hand the reserve to its group: {ln.strip()[:120]}",
            )
            self.assertIn('group="P"' if 'group="P"' in ln else 'group="D"', ln)
            self.assertIn('rank_reserve_kwargs("P"' if 'group="P"' in ln else 'rank_reserve_kwargs("D"', ln)

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
        with mock.patch.object(launcher, "user_reserve_rank_env", lambda *_a, **_k: {}):
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


class TheRankGetsWhatTheLauncherBooked(unittest.TestCase):
    """#236 review MUSS-1: per group and card the rank gets the EFFECTIVE reserve the
    launcher booked -- D's booked-rest cards 0 (stated value only named), P the
    operator's value. Driven through ``budgets_from_dc``'s booked path, not through
    ``floors_for_cards`` directly."""

    PROFILE = pdflip_form.PROFILE_QWEN27B

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {cg.FLOOR_DIGEST_FILE_ENV: "/nonexistent/flliper-236-floor-digest.json"},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (cg.LAW_ENV, cg.USER_RESERVE_ENV, cg.USER_RESERVE_MASKED_ENV):
            os.environ.pop(k, None)
        cg._RESERVE_READ_NOTED.clear()
        self.addCleanup(cg._RESERVE_READ_NOTED.clear)
        self.cards = launcher.order_cards([
            launcher.Card(nvml_index=0, uuid=C3080A, name="3080", total_mib=20480, reserved_mib=425),
            launcher.Card(nvml_index=1, uuid=C5090, name="5090", total_mib=32607, reserved_mib=518),
            launcher.Card(nvml_index=2, uuid=C3080B, name="3080", total_mib=20480, reserved_mib=425),
        ])
        self.ns = SimpleNamespace(profile=self.PROFILE, env_d="", env_p="")

    def _launcher_pass(self, group):
        """Run the REAL ``budgets_from_dc`` pass of ``group``; return the reserve it
        priced the floors with and the floors it derived."""
        seen = {}
        real = cb.floors_for_cards

        def spy(cards, grp, reserve=None):
            floors = real(cards, grp, reserve)
            seen["reserve"], seen["floors"] = reserve, floors
            return floors

        kw = {}
        if group == "D":
            kw = dict(launcher.booked_rest_kwargs(self.cards, self.PROFILE, "D"))
            self.assertIsNotNone(kw.get("booked_rest_mib"), "the 27B profile must book D's rest")
        with mock.patch.object(cb, "floors_for_cards", spy):
            launcher.budgets_from_dc(
                self.cards, {c.uuid: 1000 for c in self.cards}, [].append, group,
                user_reserve_by_card=dict(RESERVE), **kw)
        return seen

    def _rank_floor(self, uuid, group, env):
        rank_env = {k: v for k, v in env.items() if k in (cg.USER_RESERVE_ENV, cg.USER_RESERVE_MASKED_ENV)}
        with mock.patch.dict(os.environ, rank_env), mock.patch.object(
            registry_nvml, "current_device_uuid", lambda: uuid
        ):
            return cg.corridor_floor_for_current_device(group=group)

    def _env(self, group, **extra):
        return launcher.build_env(
            tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s", debug_hold=False,
            tag="probe", group=group,
            **launcher.rank_reserve_kwargs(group, self.ns, self.cards, RESERVE), **extra)

    def _check_d_rank_equals_launcher(self):
        seen = self._launcher_pass("D")
        env = self._env("D")
        for c in self.cards:
            rank = self._rank_floor(c.uuid, "D", env)
            lf = seen["floors"][c.uuid]
            assert int(rank.mib) == int(lf.mib), (c.uuid, int(rank.mib), int(lf.mib))
            assert int(rank.reserve_mib) == int(lf.reserve_mib) == 0, (c.uuid, rank.reserve_mib, lf.reserve_mib)

    def test_t10_d_booked_cards_rank_floor_equals_launcher_floor_equals_transient_plus_zero(self):
        seen = self._launcher_pass("D")
        self.assertTrue(all(v == 0 for v in seen["reserve"].values()), seen["reserve"])
        env = self._env("D")
        self.assertEqual(
            {k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_ENV]).items()},
            {c.uuid: 0 for c in self.cards},
        )
        self.assertEqual(
            {k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_MASKED_ENV]).items()}, RESERVE
        )
        self._check_d_rank_equals_launcher()

    def test_t11_p_unbooked_rank_floor_equals_launcher_floor_equals_transient_plus_reserve(self):
        seen = self._launcher_pass("P")
        self.assertEqual({k: int(v) for k, v in seen["reserve"].items()}, RESERVE)
        env = self._env("P")
        self.assertNotIn(cg.USER_RESERVE_MASKED_ENV, env)
        self.assertEqual({k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_ENV]).items()}, RESERVE)
        for c in self.cards:
            rank = self._rank_floor(c.uuid, "P", env)
            lf = seen["floors"][c.uuid]
            self.assertEqual(int(rank.mib), int(lf.mib))
            self.assertEqual(int(rank.reserve_mib), RESERVE[c.uuid])
            self.assertEqual(int(rank.mib), int(rank.transient_mib) + RESERVE[c.uuid])

    def test_t12_the_launcher_and_the_rank_env_use_one_function(self):
        import inspect

        self.assertIn("effective_reserve_by_card(cards, user_reserve_by_card, booked)", inspect.getsource(launcher.budgets_from_dc))
        self.assertIn("effective_reserve_by_card(cards, user_reserve_by_card, booked_rest_mib)", inspect.getsource(launcher.user_reserve_rank_env))

    def test_t13_mutant_d_gets_the_users_reserve_is_red(self):
        """Mutant: D's ranks are handed the stated reserve although the launcher booked 0."""
        self._check_d_rank_equals_launcher()  # green on the real code

        real = launcher.rank_reserve_kwargs

        def mutant(group, ns, cards, urc, log=None):
            kw = real(group, ns, cards, urc, log)
            kw["reserve_booked_rest_mib"] = None  # D "unbooked": publishes the stated value
            return kw

        with mock.patch.object(launcher, "rank_reserve_kwargs", mutant):
            with self.assertRaises(AssertionError):
                self._check_d_rank_equals_launcher()

    def test_t14_all_masked_still_publishes_an_explicit_zero_and_names_the_mask(self):
        cg._RESERVE_READ_NOTED.clear()
        env = self._env("D")
        with mock.patch.dict(
            os.environ,
            {k: env[k] for k in (cg.USER_RESERVE_ENV, cg.USER_RESERVE_MASKED_ENV)},
        ), mock.patch.object(registry_nvml, "current_device_uuid", lambda: C5090):
            with self.assertLogs(cg.logger, level=logging.INFO) as cm:
                cg.corridor_floor_for_current_device(group="D")
        text = "\n".join(cm.output)
        self.assertIn("source=masked-by-booked-rest stated_mib=1800", text)
        self.assertIn("reserve_mib=0", text)
        self.assertIn("masked_by_booked_rest=", text)

    def test_t15_a_launcher_log_line_names_stated_effective_and_mask(self):
        lines = []
        launcher.rank_reserve_kwargs("D", self.ns, self.cards, RESERVE, lines.append)
        joined = "\n".join(lines)
        self.assertIn("PDFLIP-USER-RESERVE-RANK-ENV group=D", joined)
        self.assertIn("stated=1800 effective=0 (masked: booked awake rest", joined)
        lines.clear()
        launcher.rank_reserve_kwargs("P", self.ns, self.cards, RESERVE, lines.append)
        self.assertIn("stated=1800 effective=1800", "\n".join(lines))
        self.assertNotIn("masked", "\n".join(lines))

    def test_t16_an_operator_env_value_is_respected_and_named(self):
        operator = '{"GPU-operator": 7}'
        ns = SimpleNamespace(profile=self.PROFILE, env_d=f"{cg.USER_RESERVE_ENV}={operator}", env_p="")
        lines = []
        kw = launcher.rank_reserve_kwargs("D", ns, self.cards, RESERVE, lines.append)
        env = launcher.build_env(
            tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s", debug_hold=False,
            tag="probe", group="D", group_env_extra={cg.USER_RESERVE_ENV: operator}, **kw)
        self.assertEqual(env[cg.USER_RESERVE_ENV], operator)
        self.assertNotIn(cg.USER_RESERVE_MASKED_ENV, env)
        self.assertTrue(any("--env-d states" in ln and "STANDS" in ln for ln in lines), lines)

    def _spy_pass(self, run):
        """Run ``run()`` (a launcher pass) and return what it priced the floors with."""
        seen = {}
        real = cb.floors_for_cards

        def spy(cards, grp, reserve=None):
            floors = real(cards, grp, reserve)
            seen["reserve"], seen["floors"] = reserve, floors
            return floors

        with mock.patch.object(cb, "floors_for_cards", spy):
            run()
        return seen

    def _assert_rank_equals_launcher(self, seen, ns, group="D"):
        kw = launcher.rank_reserve_kwargs(group, ns, self.cards, RESERVE)
        env = launcher.build_env(
            tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s", debug_hold=False,
            tag="probe", group=group, **kw)
        for c in self.cards:
            rank = self._rank_floor(c.uuid, group, env)
            lf = seen["floors"][c.uuid]
            self.assertEqual(int(rank.mib), int(lf.mib), c.uuid)
            self.assertEqual(int(rank.reserve_mib), int(lf.reserve_mib), c.uuid)
        return env

    def test_t20_d_only_launcher_charges_the_reserve_and_so_do_the_ranks(self):
        """--d-only: ``d_only_solve`` prices D WITHOUT a booked rest (floor + reserve on
        every card) -- the ranks must carry the reserve too (nachpruefung BUG 1)."""
        ns = SimpleNamespace(profile=self.PROFILE, env_d="", env_p="", d_only=True,
                             corridor_budget_sample=None)
        with mock.patch.object(launcher, "log_d_rank_vram_solve", lambda *a, **k: None):
            seen = self._spy_pass(lambda: launcher.d_only_solve(
                ns, self.cards, {c.uuid: 1000 for c in self.cards}, [].append,
                user_reserve_by_card=dict(RESERVE), p_split=None, chunk_layers=0))
        self.assertEqual({k: int(v) for k, v in seen["reserve"].items()}, RESERVE)
        env = self._assert_rank_equals_launcher(seen, ns)
        self.assertEqual(
            {k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_ENV]).items()}, RESERVE)
        self.assertNotIn(cg.USER_RESERVE_MASKED_ENV, env)
        # the same ns WITHOUT --d-only is the booked path: masked to 0
        ns2 = SimpleNamespace(profile=self.PROFILE, env_d="", env_p="", d_only=False)
        env2 = launcher.build_env(
            tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s", debug_hold=False,
            tag="probe", group="D", **launcher.rank_reserve_kwargs("D", ns2, self.cards, RESERVE))
        self.assertTrue(all(int(v) == 0 for v in json.loads(env2[cg.USER_RESERVE_ENV]).values()))

    def test_t21_mixed_booked_vector_with_an_unbooked_card(self):
        mixed = [None, 3000, 2000]  # card 0 (ordinal 0) unbooked, the others booked
        ns = SimpleNamespace(profile=self.PROFILE, env_d="", env_p="")
        fake = {"booked_rest_mib": mixed, "booked_rest_provenance": "test"}
        with mock.patch.object(launcher, "booked_rest_kwargs", lambda *a, **k: dict(fake)):
            seen = self._spy_pass(lambda: launcher.budgets_from_dc(
                self.cards, {c.uuid: 1000 for c in self.cards}, [].append, "D",
                user_reserve_by_card=dict(RESERVE), **launcher.booked_rest_kwargs()))
            env = self._assert_rank_equals_launcher(seen, ns)
        unbooked = self.cards[0].uuid
        eff = {k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_ENV]).items()}
        self.assertEqual(eff[unbooked], RESERVE[unbooked])
        self.assertTrue(all(eff[c.uuid] == 0 for c in self.cards[1:]))
        masked = {k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_MASKED_ENV]).items()}
        self.assertEqual(set(masked), {c.uuid for c in self.cards[1:]})

    def test_t22_capped_branch_uses_the_same_vector_on_both_sides(self):
        ns = SimpleNamespace(
            profile=self.PROFILE, env_p="",
            env_d=f"{launcher.TORCH_CACHE_CAP_ENV}=1")
        self.assertTrue(launcher.torch_cache_cap_armed(ns))
        kw = dict(launcher.booked_rest_kwargs(self.cards, self.PROFILE, "D", capped=True))
        self.assertIsNotNone(kw.get("booked_rest_mib"))
        seen = self._spy_pass(lambda: launcher.budgets_from_dc(
            self.cards, {c.uuid: 1000 for c in self.cards}, [].append, "D",
            user_reserve_by_card=dict(RESERVE), **kw))
        self.assertTrue(all(int(v) == 0 for v in seen["reserve"].values()))
        env = self._assert_rank_equals_launcher(seen, ns)
        self.assertEqual({k: int(v) for k, v in json.loads(env[cg.USER_RESERVE_MASKED_ENV]).items()}, RESERVE)

    def test_t23_the_dial_reads_the_reserve_through_the_one_reader(self):
        import inspect

        from flliper.srt.managers import vram_dial

        src = inspect.getsource(vram_dial)
        self.assertIn("reserve_for_card(card_uuid, 0)", src)
        self.assertNotIn("user_reserve_by_card().get(card_uuid", src)

    def test_t17_effective_reserve_function_masks_only_booked_cards(self):
        eff, masked = launcher.effective_reserve_by_card(
            self.cards, RESERVE, [None, 5, None])
        self.assertEqual(eff[self.cards[1].uuid], 0)
        self.assertEqual(masked, {self.cards[1].uuid: RESERVE[self.cards[1].uuid]})
        self.assertEqual(eff[self.cards[0].uuid], RESERVE[self.cards[0].uuid])
        with self.assertRaises(launcher.PdFlipLaunchRefused):
            launcher.effective_reserve_by_card(self.cards, RESERVE, [None])


class OneRankSideReserveChoice(unittest.TestCase):
    """#236 review SOLL-2: weight_updater and the floor door resolve one card the same way."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (cg.USER_RESERVE_ENV, cg.USER_RESERVE_MASKED_ENV):
            os.environ.pop(k, None)

    def test_t18_mapping_first_scalar_only_without_an_entry(self):
        self.assertEqual(cg.reserve_for_card(C5090, 512), (512, "fallback-scalar"))
        self.assertEqual(cg.reserve_for_card(C5090, 0), (0, "none"))
        os.environ[cg.USER_RESERVE_ENV] = json.dumps({C5090: 1800})
        self.assertEqual(cg.reserve_for_card(C5090, 512), (1800, "env-by-card"))
        self.assertEqual(cg.reserve_for_card(C3080A, 512), (512, "fallback-scalar"))
        os.environ[cg.USER_RESERVE_ENV] = json.dumps({C5090: 0})
        os.environ[cg.USER_RESERVE_MASKED_ENV] = json.dumps({C5090: 1800})
        self.assertEqual(cg.reserve_for_card(C5090, 512), (0, "masked-by-booked-rest"))

    def test_t19_the_credit_floor_reader_uses_that_one_function(self):
        import pathlib

        import flliper.srt.managers.scheduler_components.weight_updater as wu

        src = pathlib.Path(wu.__file__).read_text()
        self.assertIn("reserve_for_card(", src)
        self.assertNotIn("user_reserve_by_card().get(uuid_key)", src)


if __name__ == "__main__":
    unittest.main()
