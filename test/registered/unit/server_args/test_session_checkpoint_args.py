"""Argument-time fail-fast for #410 session checkpoints.

The runtime repeats every one of these as a named refusal on the control
route -- a server can be reconfigured after boot -- but failing at argument
time means a misconfigured launch never reaches the point where a user
believes their conversation was checkpointed and finds out later that it was
not.

Both directions of each gate, and the DEFAULT: with the flag unset none of
this validation runs and no other argument changes meaning.

    python -m pytest test/registered/unit/server_args/test_session_checkpoint_args.py -v
"""

import inspect
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__ (the
    repo-wide convention for argument tests), so the #410 handler is invoked
    explicitly -- which is also what pins it as a handler of its own rather
    than a block buried in an unrelated one."""
    args = ServerArgs(model_path="dummy", **kwargs)
    args._handle_session_checkpoints()
    return args


def _enabled(**kwargs):
    base = dict(
        enable_session_checkpoints=True,
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
    )
    base.update(kwargs)
    return _args(**base)


class TestSessionCheckpointArgs(CustomTestCase):
    def test_off_by_default(self):
        args = _args()
        self.assertFalse(args.enable_session_checkpoints)
        self.assertEqual(args.session_checkpoint_vram_max_age_s, 60.0)
        self.assertEqual(args.session_checkpoint_host_max_age_s, 900.0)

    def test_the_valid_combination_constructs(self):
        args = _enabled()
        self.assertTrue(args.enable_session_checkpoints)

    def test_hierarchical_cache_is_required(self):
        with self.assertRaises(ValueError) as ctx:
            _args(enable_session_checkpoints=True, enable_hierarchical_cache=False)
        self.assertIn("--enable-hierarchical-cache", str(ctx.exception))

    def test_only_the_file_storage_backend_is_accepted(self):
        with self.assertRaises(ValueError) as ctx:
            _enabled(hicache_storage_backend="mooncake")
        self.assertIn("--hicache-storage-backend file", str(ctx.exception))
        self.assertIn("#261", str(ctx.exception))

    def test_the_page_head_layout_is_refused_by_name(self):
        """#441a: page_head's host write-back is the lf_ph route, which
        segfaults, and a checkpoint writes a whole session through it."""
        with self.assertRaises(ValueError) as ctx:
            _enabled(hicache_mem_layout="page_head")
        message = str(ctx.exception)
        self.assertIn("page_head", message)
        self.assertIn("#441a", message)
        self.assertIn("transfer_kv_all_layer_lf_ph", message)
        # can-fail twin: the layouts that reach the working routes construct.
        for layout in ("page_first", "page_first_direct", "layer_first"):
            self.assertTrue(
                _enabled(hicache_mem_layout=layout).enable_session_checkpoints,
                layout,
            )

    def test_a_negative_age_threshold_is_refused(self):
        for name in (
            "session_checkpoint_vram_max_age_s",
            "session_checkpoint_host_max_age_s",
        ):
            with self.assertRaises(ValueError) as ctx:
                _enabled(**{name: -1.0})
            self.assertIn(name.replace("_", "-"), str(ctx.exception))

    def test_an_inverted_ladder_is_refused_with_both_numbers(self):
        with self.assertRaises(ValueError) as ctx:
            _enabled(
                session_checkpoint_vram_max_age_s=100.0,
                session_checkpoint_host_max_age_s=10.0,
            )
        message = str(ctx.exception)
        self.assertIn("100.0", message)
        self.assertIn("10.0", message)
        self.assertIn("VRAM -> RAM -> disk", message)
        # can-fail twin: equal thresholds are legal (no VRAM window at all).
        self.assertTrue(
            _enabled(
                session_checkpoint_vram_max_age_s=10.0,
                session_checkpoint_host_max_age_s=10.0,
            ).enable_session_checkpoints
        )

    def test_the_handler_is_wired_into_post_init(self):
        """The gates above are only worth anything if a real launch runs
        them. `model_path='dummy'` returns from `__post_init__` early, so no
        constructor-level test can see the call site -- this pin reads it
        directly. It fails if the handler is ever defined but not called."""
        source = inspect.getsource(ServerArgs.__post_init__)
        self.assertIn("self._handle_session_checkpoints()", source)

    def test_none_of_this_fires_while_the_feature_is_off(self):
        """The pin that keeps the default path untouched: the same arguments
        that are refused above are accepted when the flag is unset."""
        args = _args(
            enable_session_checkpoints=False,
            hicache_storage_backend="mooncake",
            session_checkpoint_vram_max_age_s=100.0,
            session_checkpoint_host_max_age_s=10.0,
        )
        self.assertFalse(args.enable_session_checkpoints)


if __name__ == "__main__":
    unittest.main()
