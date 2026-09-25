# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The pinned-host OS reserve is an env knob, and a container can lower it.

THE SPECIMEN, NF host acceptance 2026-09-25 (container, cap 82g, NF-RC2
b89592806a). On the first store hand-back P registered its read buffers and
the runtime backstop refused, four times:

    Pinned host RAM over-committed: 0.54 GB requested across 4 pool(s)
    [ArenaMHAHostPool, Mamba anchor, qsa_indexer, read buffers] does not fit in
    8.73 GB available minus a 10.74 GB OS reserve = 0.00 GB usable

-> W53 Weg2StoreHandbackFailed -> HTTP 413 on the 148k needle. Inside the
cgroup ``available`` is the cap minus nonreclaim (82 - 73.9), and the 10 GiB
"OS reserve" duplicates what the cap already guarantees. Natively (CT999, 118
GiB limit) the same boot never came near the reserve.

WHAT MUST HOLD: unset, the reserve is the historical 10 GiB byte for byte, so
a native boot is unchanged; set, the module constant -- and with it the default
of both admission functions -- follows it, so the container can admit the 0.54
GB the specimen refused.
"""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

GB = 10**9
GIB = 1024**3
#: The specimen's own numbers, as the P log printed them.
AVAILABLE_GB = 8.73
DEMAND_GB = 0.54
TOTAL_GB = 82.0
ENV = "SGLANG_PINNED_HOST_RESERVE_GIB"

#: Runs in a FRESH interpreter: the reserve is bound at import (module constant
#: and the default argument of both admission functions), so only a new process
#: shows what a container launched with the env actually gets -- and a reload
#: inside this process would swap the registry under every other test module.
_PROBE = f"""
import json
from unittest import mock
from sglang.srt.mem_cache import pinned_host_budget as m
post = m.PinnedHostPost("read buffers", "--hicache-size", int({DEMAND_GB} * {GB}))
err = m.joint_pinned_host_error([post], int({TOTAL_GB} * {GB}), int({AVAILABLE_GB} * {GB}))
m.clear_registered_posts()
admitted = True
with mock.patch.object(m, "pinned_host_memory_bytes",
                       return_value=(int({TOTAL_GB} * {GB}), int({AVAILABLE_GB} * {GB}))):
    try:
        m.check_and_register_pinned_post("read buffers", "--hicache-size", int({DEMAND_GB} * {GB}))
    except ValueError:
        admitted = False
print(json.dumps({{"reserve": m.PINNED_HOST_RESERVE_BYTES, "err": err, "admitted": admitted}}))
"""


def _probe(value):
    env = {k: v for k, v in os.environ.items() if k != ENV}
    if value is not None:
        env[ENV] = value
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


class TheReserveIsAnEnvKnob(unittest.TestCase):
    def test_unset_is_the_historical_ten_gib_and_refuses_the_specimen(self):
        # The can-fail: without the knob the container refuses exactly as it did
        # on metal, so the admission below proves the knob, not the numbers.
        got = _probe(None)
        self.assertEqual(got["reserve"], 10 * GIB)
        self.assertIsNotNone(got["err"])
        self.assertIn("0.00 GB usable", got["err"])
        self.assertFalse(got["admitted"])

    def test_a_lowered_reserve_admits_the_specimen_through_the_default_argument(self):
        # No reserve is passed at either call: the DEFAULT argument must carry
        # the env value, or every call site relying on it still refuses.
        got = _probe("2")
        self.assertEqual(got["reserve"], 2 * GIB)
        self.assertIsNone(got["err"])
        self.assertTrue(got["admitted"])


if __name__ == "__main__":
    unittest.main()
