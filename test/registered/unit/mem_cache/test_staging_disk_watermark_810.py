"""#810: a staging host tier may not sit in front of an UNBOUNDED file tier.

THE FINDING. ``LRUFileEvictor`` is default-off: ``_eviction_configured`` is
``max_size_bytes > 0 or min_free_bytes > 0``, and with neither set ``reserve()``
admits every write forever. That is a defensible default while the host tier is
an L2 retention cache -- the disk tier is then a second copy of something that
is also in RAM.

Under ``--hicache-host-role staging`` it stops being defensible. The pinned host
tier becomes a small write-through buffer and THIS store becomes the retention
tier, i.e. the only copy. An unbounded only-copy grows until the filesystem is
full, at which point ``HiCacheFile.set`` rolls its reservation back and returns
``False`` -- which the ack path treats as a partial backup, i.e. as a cache miss.
The capacity loss is silent, and it is permanent.

THE TRADE THIS PINS: refusal, not auto-arming. Auto-arming needs a NUMBER and
there is no honest source for one. ``max_size`` and ``min_free_space`` would
both have to be invented, and the single derivation that does exist,
``_clamp_max_size_to_fs``, clamps a cap that was already supplied -- it is not
itself a bound, since the filesystem's own capacity permits the store to
consume all of it. An invented default would read to the next operator as a
considered budget. So the launch is refused and the number stays the
operator's, which is also the only party that knows what else shares the
filesystem.

Both directions of the gate are covered, plus the retention default, plus the
precedence: the per-backend ``extra_config`` overrides the env vars in the
arming check exactly as it does in ``_load_config``, because the two readings
are the SAME reading -- the check runs inside the evictor, on the config the
evictor resolved.
"""

import os
import tempfile
import unittest
from unittest import mock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - registration is a CI-time marker

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _evictor(tmpdir, *, require_watermark=False, extra_config=None):
    return LRUFileEvictor(
        tmpdir,
        "_test_suffix",
        tp_rank=0,
        is_mla_model=False,
        extra_config=extra_config,
        require_watermark=require_watermark,
    )


class StagingRequiresABoundedFileTierTest(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        # The env fallbacks are a real config source for this check, so they
        # are cleared rather than assumed absent.
        self._env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "",
                "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE": "",
            },
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_staging_refuses_an_unarmed_file_tier(self):
        """THE FALSIFIER. Without this the retention tier is unbounded exactly
        when the host tier in front of it has been made small."""
        with self.assertRaises(ValueError) as ctx:
            _evictor(self.tmpdir, require_watermark=True)
        message = str(ctx.exception)
        self.assertIn("max_size", message)
        self.assertIn("min_free_space", message)

    def test_the_other_direction_a_max_size_arms_it(self):
        evictor = _evictor(
            self.tmpdir, require_watermark=True, extra_config={"max_size": "1G"}
        )
        self.assertTrue(evictor._eviction_configured)

    def test_the_other_direction_a_min_free_space_arms_it(self):
        """Either knob arms the evictor -- the check must not demand the one
        the operator did not pick."""
        evictor = _evictor(
            self.tmpdir,
            require_watermark=True,
            extra_config={"min_free_space": "1G"},
        )
        self.assertTrue(evictor._eviction_configured)

    def test_an_env_var_arms_it_too(self):
        """The env vars are the documented second source. A check that only
        read ``extra_config`` would refuse a correctly configured launch."""
        with mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "1G"}
        ):
            evictor = _evictor(self.tmpdir, require_watermark=True)
        self.assertTrue(evictor._eviction_configured)

    def test_an_extra_config_zero_overrides_an_env_var_and_is_refused(self):
        """Precedence, and why the check lives inside the evictor: per-backend
        config wins over the env var in ``_load_config``, so an explicit ``0``
        DISARMS a knob the environment had set. A parse-time check that merely
        OR-ed the two sources would pass this launch and leave it unbounded."""
        with mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "1G"}
        ):
            with self.assertRaises(ValueError):
                _evictor(
                    self.tmpdir,
                    require_watermark=True,
                    extra_config={"max_size": "0"},
                )

    def test_retention_keeps_the_unbounded_default(self):
        """The backward-compatibility pin. Default role, no watermark: the
        evictor is inert and ``reserve`` admits, exactly as before."""
        evictor = _evictor(self.tmpdir, require_watermark=False)
        self.assertFalse(evictor._eviction_configured)
        self.assertTrue(evictor.reserve("some-key", 1 << 30, key="some-key"))

    def test_a_non_owner_rank_is_judged_on_the_config_not_on_its_bookkeeping(self):
        """Under MLA only rank 0 keeps the LRU index, so ``_eviction_enabled``
        is False on the other ranks even when a cap IS configured. Refusing on
        that would refuse an armed directory on every rank but one."""
        evictor = LRUFileEvictor(
            self.tmpdir,
            "_test_suffix",
            tp_rank=1,
            is_mla_model=True,
            extra_config={"max_size": "1G"},
            require_watermark=True,
        )
        self.assertTrue(evictor._eviction_configured)
        self.assertFalse(evictor._eviction_enabled)


class TheBackendPassesTheRoleTest(CustomTestCase):
    """The production edge: ``HiCacheFile`` is what builds the evictor."""

    @staticmethod
    def _config(host_role, tmpdir):
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

        return HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=None,
            host_role=host_role,
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self._env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "",
                "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE": "",
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": "",
            },
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_the_default_role_builds_an_unbounded_backend_as_before(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        store = HiCacheFile(self._config("retention", self.tmpdir), self.tmpdir)
        self.assertFalse(store._evictor._eviction_configured)

    def test_a_staging_role_refuses_to_build_an_unbounded_backend(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with self.assertRaises(ValueError):
            HiCacheFile(self._config("staging", self.tmpdir), self.tmpdir)

    def test_the_config_field_defaults_to_retention(self):
        """Every other construction site of ``HiCacheStorageConfig`` is
        unchanged because the field is defaulted."""
        self.assertEqual(self._config("retention", self.tmpdir).host_role, "retention")
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

        self.assertEqual(
            HiCacheStorageConfig.__dataclass_fields__["host_role"].default,
            "retention",
        )


class TheControllerCarriesTheRoleTest(CustomTestCase):
    """The other production edge: ``ServerArgs`` -> ``HiCacheStorageConfig``.

    Without this hop the field is always its default and the refusal above can
    never fire on a real boot -- the feature would be a unit test with no
    production path, which is the shape this project's rules exist to catch.
    """

    def _generate(self, server_args):
        import types

        from sglang.srt.managers import cache_controller as cc_mod

        controller = types.SimpleNamespace(
            mem_pool_device=object(),
            mem_pool_host=types.SimpleNamespace(layout="layer_first"),
            enable_storage_metrics=False,
            get_attn_cp_rank_and_size=lambda: (0, 1),
            _dcp_owner_ctx=lambda: None,
        )
        parallel = types.SimpleNamespace(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_tp_rank=0, attn_tp_size=1
        )
        with mock.patch.object(
            cc_mod, "get_parallel", lambda: parallel
        ), mock.patch.object(
            cc_mod, "is_dp_attention_enabled", lambda: False
        ), mock.patch.object(
            cc_mod, "get_server_args", lambda: server_args
        ), mock.patch.object(
            cc_mod, "canonical_identity_hash_for", lambda *a, **k: None
        ):
            return cc_mod.HiCacheController._generate_storage_config(
                controller, model_name=None, storage_backend_extra_config=None
            )

    def test_a_staging_launch_reaches_the_backend_as_staging(self):
        import types

        config = self._generate(
            types.SimpleNamespace(
                hicache_host_role="staging", phase_flip_canonical_kv_page=False
            )
        )
        self.assertEqual(
            config.host_role,
            "staging",
            "the role never left ServerArgs: the backend cannot refuse an "
            "unbounded retention tier because it never learns it is one",
        )

    def test_the_default_launch_reaches_the_backend_as_retention(self):
        import types

        config = self._generate(
            types.SimpleNamespace(
                hicache_host_role="retention", phase_flip_canonical_kv_page=False
            )
        )
        self.assertEqual(config.host_role, "retention")

    def test_a_bare_controller_without_server_args_stays_retention(self):
        """``_generate_storage_config`` documents a fallback for bare-controller
        unit tests, where ``get_server_args`` raises. That path must not become
        a staging claim."""
        config = self._generate(None)
        self.assertEqual(config.host_role, "retention")


if __name__ == "__main__":
    unittest.main()
