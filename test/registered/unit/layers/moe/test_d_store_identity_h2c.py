"""NF-Bootzeit H2c: a store sentinel vouches only for the boot that wrote it.

WHY (ENTSCHEIDUNGEN-0927, 28.09.): D-Store-Adopt (H2) reads expert rows out of
the shared store instead of the checkpoint, on the word of the sentinels
``L<n>-<attr>.bin.r<k>.written.json``. A sentinel said only "these slots are
written" -- not with WHICH checkpoint and WHICH slot map. A store that outlives
a boot (host tmpfs, a changed FR / --rank-moe-ratio, another checkpoint) would
have handed D foreign bytes as its own experts. That is why the profile ran
noadopt ("Expertenstore-Identitaet ist VORBEDINGUNG").

THE FIX: the launcher computes one identity from the checkpoint fingerprint
and the expert-map bytes and gives it to BOTH groups; ``mark_rows_written``
stamps it; ``rows_written`` and ``store_adopt.snapshot_rows`` ignore every
sentinel that does not carry it; ``store_adopt.active()`` refuses without it.

RED on ebeccfd190 (no identity anywhere), GREEN with the fix.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from flliper.srt.layers.moe import expert_store as es

try:
    from flliper.srt.layers.moe import store_adopt as sa
except ImportError:  # pragma: no cover
    sa = None

IDENT_ENV = "FLLIPER_MOE_EXPERT_STORE_IDENTITY"


def _ckpt(root, config=b'{"num_experts": 512}', shard_size=10):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "wb") as fh:
        fh.write(config)
    with open(os.path.join(root, "model.safetensors.index.json"), "wb") as fh:
        fh.write(b'{"weight_map": {}}')
    with open(os.path.join(root, "model-00001.safetensors"), "wb") as fh:
        fh.write(b"\0" * shard_size)
    return root


class TestComputeIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="h2c_")
        self.model = _ckpt(os.path.join(self.tmp, "model"))
        self.map = os.path.join(self.tmp, "karte.json")
        with open(self.map, "w") as fh:
            json.dump({"version": 2, "slots": 324}, fh)

    def test_stable_for_the_same_checkpoint_and_map(self):
        self.assertTrue(hasattr(es, "compute_identity"), "no compute_identity (base)")
        a = es.compute_identity(self.model, self.map)
        self.assertTrue(a)
        self.assertEqual(a, es.compute_identity(self.model, self.map))

    def test_changes_with_the_map(self):
        self.assertTrue(hasattr(es, "compute_identity"), "no compute_identity (base)")
        a = es.compute_identity(self.model, self.map)
        with open(self.map, "w") as fh:
            json.dump({"version": 2, "slots": 332}, fh)  # other FR -> other slot map
        self.assertNotEqual(a, es.compute_identity(self.model, self.map))

    def test_changes_with_the_checkpoint(self):
        self.assertTrue(hasattr(es, "compute_identity"), "no compute_identity (base)")
        a = es.compute_identity(self.model, self.map)
        other = _ckpt(os.path.join(self.tmp, "other"), shard_size=11)
        self.assertNotEqual(a, es.compute_identity(other, self.map))
        other2 = _ckpt(os.path.join(self.tmp, "other2"), config=b'{"num_experts": 256}')
        self.assertNotEqual(a, es.compute_identity(other2, self.map))

    def test_no_map_no_identity(self):
        self.assertTrue(hasattr(es, "compute_identity"), "no compute_identity (base)")
        self.assertEqual(es.compute_identity(self.model, ""), "")
        self.assertEqual(es.compute_identity(self.model, os.path.join(self.tmp, "nope.json")), "")


class TestSentinelStamp(unittest.TestCase):
    def setUp(self):
        self.store = tempfile.mkdtemp(prefix="h2c_store_")

    def _foreign(self, rows, identity=None):
        path = es.store_path(self.store, "L0", "w13_weight_packed") + ".r1.written.json"
        data = {"rank": 1, "rows": rows}
        if identity is not None:
            data["identity"] = identity
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_writer_stamps_the_identity(self):
        with mock.patch.dict(os.environ, {IDENT_ENV: "boot-A"}):
            path = es.mark_rows_written(self.store, "L0", "w13_weight_packed", 0, [3, 1])
        with open(path) as fh:
            self.assertEqual(json.load(fh).get("identity"), "boot-A")

    def test_reader_ignores_foreign_and_unstamped_sentinels(self):
        with mock.patch.dict(os.environ, {IDENT_ENV: "boot-A"}):
            es.mark_rows_written(self.store, "L0", "w13_weight_packed", 0, [0, 1])
            self._foreign([2, 3], identity="boot-B")
            got = es.rows_written(self.store, "L0", "w13_weight_packed", 4)
        self.assertEqual(sorted(got), [0, 1], "a foreign boot's rows were believed")
        with mock.patch.dict(os.environ, {IDENT_ENV: "boot-A"}):
            self._foreign([2, 3], identity=None)  # a pre-H2c sentinel
            got = es.rows_written(self.store, "L0", "w13_weight_packed", 4)
        self.assertEqual(sorted(got), [0, 1], "an unstamped sentinel was believed")

    def test_without_identity_the_old_behaviour_holds(self):
        env = {k: v for k, v in os.environ.items() if k != IDENT_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            es.mark_rows_written(self.store, "L0", "w13_weight_packed", 0, [0, 1])
            self._foreign([2, 3], identity=None)
            got = es.rows_written(self.store, "L0", "w13_weight_packed", 4)
        self.assertEqual(sorted(got), [0, 1, 2, 3])


KARTE = {
    "version": 2, "total": 12, "slots": 10, "p_layer_stage": [0],
    "phases": {
        "P": {"resident": [[0, 6]], "common": [[0, 6]], "slot_of": {}},
        "D": {"resident": [[0], [6, 7]], "prefix_by_stage": [[[0]], [[6]]],
              "extra_by_stage": [[[]], [[7]]], "slot_of": {}},
    },
}
_SLOT = {g: i for i, g in enumerate(g for g in range(12) if g not in (0, 6))}
for _ph in ("P", "D"):
    KARTE["phases"][_ph]["slot_of"] = {str(g): s for g, s in _SLOT.items()}


def _layer():
    import types

    import torch

    return types.SimpleNamespace(
        layer_id=0, num_local_experts=7, num_experts=12,
        _expert_shard_generic=True, _gguf_expert_range=(6, 12),
        moe_tp_rank=1, _moe_store_adopt_ok=True,
        w13_weight_packed=torch.zeros(7, 2), w2_weight_packed=torch.zeros(7, 3))


class TestAdoptNeedsIdentity(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(sa)
        sa.reset_for_tests()
        self.store = tempfile.mkdtemp(prefix="h2c_adopt_")
        self._map = mock.patch("flliper.srt.layers.moe.expert_store.expert_map",
                               return_value=KARTE)
        self._map.start()

    def tearDown(self):
        self._map.stop()

    def _publish(self, identity):
        for attr in ("w13_weight_packed", "w2_weight_packed"):
            open(os.path.join(self.store, f"L0-{attr}.bin"), "wb").close()
            data = {"rank": 0, "rows": sorted(_SLOT.values())}
            if identity is not None:
                data["identity"] = identity
            with open(os.path.join(self.store, f"L0-{attr}.bin.r0.written.json"), "w") as fh:
                json.dump(data, fh)

    def _env(self, identity):
        env = {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_MOE_EXPERT_STORE_DIR": self.store}
        if identity is not None:
            env[IDENT_ENV] = identity
        base = {k: v for k, v in os.environ.items() if k != IDENT_ENV}
        base.update(env)
        return mock.patch.dict(os.environ, base, clear=True)

    def test_inactive_without_identity(self):
        self._publish(identity=None)
        with self._env(identity=None):
            self.assertFalse(sa.active(), "adopt ran without any store identity")
            self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset())

    def test_foreign_store_vetoes_nothing(self):
        self._publish(identity="boot-OLD")  # the store survived another boot
        with self._env(identity="boot-NEW"):
            self.assertTrue(sa.active())
            self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset(),
                             "D would have adopted another boot's expert bytes")

    def test_own_store_is_adopted(self):
        self._publish(identity="boot-NEW")
        with self._env(identity="boot-NEW"):
            self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset({8, 9, 10, 11}))


class TestLauncherPublishesIdentity(unittest.TestCase):
    def _build(self, **kw):
        from flliper.srt.pdflip import launcher

        return launcher.build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                                  debug_hold=False, tag="probe", **kw)

    def test_both_groups_get_it_with_the_map(self):
        for group in ("P", "D"):
            env = self._build(group=group, expert_map_path="/tmp/karte.json",
                              expert_store_identity="abc123")
            self.assertEqual(env.get(IDENT_ENV), "abc123")

    def test_no_map_pops_an_inherited_value(self):
        with mock.patch.dict(os.environ, {IDENT_ENV: "from-the-shell"}):
            env = self._build(group="D")
        self.assertNotIn(IDENT_ENV, env)

    def test_publish_store_identity_logs_the_marker(self):
        from flliper.srt.pdflip import launcher

        tmp = tempfile.mkdtemp(prefix="h2c_launch_")
        model = _ckpt(os.path.join(tmp, "m"))
        emap = os.path.join(tmp, "k.json")
        with open(emap, "w") as fh:
            fh.write("{}")
        lines = []
        ident = launcher.publish_store_identity(model, emap, lines.append)
        self.assertEqual(ident, es.compute_identity(model, emap))
        self.assertTrue(any("H2c STORE-IDENTITY id=" + ident in l for l in lines))
        self.assertEqual(launcher.publish_store_identity(model, "", lines.append), "")


if __name__ == "__main__":
    unittest.main()
