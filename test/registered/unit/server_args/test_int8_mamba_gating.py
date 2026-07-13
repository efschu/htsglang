"""CPU unit tests for the int8-mamba-checkpoint compatibility gating.

--enable-int8-mamba-checkpoint moves radix-cached mamba states into a
separate int8 checkpoint pool (own slot namespace, qdata+scale layout).
The hierarchical cache (HiMambaRadixCache/MambaPoolHost) and custom
radix-cache backends transfer/free node states against the ACTIVE
MambaPool, so the combinations must be rejected up front with the
technical reason, not fail deep inside pool assembly.
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_args(**kwargs):
    # model_path='dummy' short-circuits __post_init__ so the handler can
    # be exercised in isolation.
    return ServerArgs(model_path="dummy", **kwargs)


class Int8MambaGatingTest(CustomTestCase):
    def test_flag_off_is_noop(self):
        args = make_args(enable_hierarchical_cache=True)
        args._handle_int8_mamba_checkpoint()  # must not raise

    def test_int8_alone_is_ok(self):
        args = make_args(enable_int8_mamba_checkpoint=True)
        args._handle_int8_mamba_checkpoint()  # must not raise

    def test_rejects_hierarchical_cache_with_reason(self):
        args = make_args(
            enable_int8_mamba_checkpoint=True,
            enable_hierarchical_cache=True,
        )
        with self.assertRaises(ValueError) as ctx:
            args._handle_int8_mamba_checkpoint()
        msg = str(ctx.exception)
        # The message must carry the actual technical grounds, not just
        # "unsupported".
        self.assertIn("int8 checkpoint pool", msg)
        self.assertIn("slot namespace", msg)
        self.assertIn("ACTIVE MambaPool", msg)
        self.assertIn("re-quantize", msg)

    def test_rejects_custom_radix_backend(self):
        args = make_args(
            enable_int8_mamba_checkpoint=True,
            radix_cache_backend="some_backend",
        )
        with self.assertRaises(ValueError) as ctx:
            args._handle_int8_mamba_checkpoint()
        self.assertIn("not int8-aware", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
