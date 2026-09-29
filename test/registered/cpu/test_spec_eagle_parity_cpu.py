import unittest

from flliper.srt.environ import envs
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.kits.spec_server_kits import SpecParityKit
from flliper.test.server_fixtures.spec_eagle_fixture import Eagle3Base

# Estimated: 2 sequential 8B server launches + one 4-prompt greedy method
# (CUDA sibling: 360); tune from CI TIMINGS once it has run there.
register_cpu_ci(est_time=480, suite="base-b-test-cpu")


class TestEagle3ParityCPU(SpecParityKit, Eagle3Base):
    """EAGLE3 spec (intel_amx) greedy output == non-spec reference."""

    attention_backend = "intel_amx"
    disable_overlap = True
    mem_fraction_static = 0.3
    env_overrides = ((envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
