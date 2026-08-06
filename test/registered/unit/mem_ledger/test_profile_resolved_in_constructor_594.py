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
"""#594: the activation profile must be resolved in the LEDGER CONSTRUCTOR.

#596 found this defect and fixed it on one caller
(``ledger_full_demand_per_gpu``). The gated path
(``_vram_ledger_non_kv_per_gpu``, i.e. every ``--enable-vram-ledger`` boot)
calls ``_build_card_ledgers`` directly and so routed straight around that fix,
which is why the payout still could not boot: the ledger asked the footprint
cache for ``a77d53df9f2e`` (chunked_prefill_size 0, decode_max_bs 0) while the
ranks had written their measurements under ``ff1fa555fe7a`` (2048, 24). Both
digests were produced on the same rig, in the same window, on 2026-08-06.

These tests EXECUTE ``_build_card_ledgers`` rather than reading it. A
source-level assertion would have passed for the shipped defect too -- the
resolving call WAS present in the file, just at a caller this path never went
through. That is the #605 lesson and it is the reason for the seam below.
"""

import types
import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GPU_MEM = 20480


class _LauncherStub:
    """A ServerArgs at the moment ``_handle_uneven_tp`` prices the reserve.

    Both profile fields are unset, which is the real state at that point: they
    are filled by ``_handle_gpu_memory_settings``, which runs later.
    """

    def __init__(self):
        self.chunked_prefill_size = None
        self.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=None, bs=None)
        )
        self.tp_size = 3
        self.device = "cuda"
        self.rank_gpu_id = [0]
        self.rank_user_reserve_mib = 1024
        #: What the profile fields looked like when the inputs were built.
        self.profile_at_inputs = None

    def _build_card_ledgers(self):
        return ServerArgs._build_card_ledgers(self)

    def user_reserve_mib_per_gpu(self, rank_gpu_id):
        return {0: 1024}

    # -- the seam: the real resolver, idempotent, fills only unset values ---
    def _apply_gpu_mem_capacity_defaults(self, gpu_mem):
        if self.chunked_prefill_size is None:
            self.chunked_prefill_size = 2048
        if self.cuda_graph_config.decode.max_bs is None:
            self.cuda_graph_config.decode.max_bs = 24

    def _widen_decode_capture_to_session_ceiling(self, decode_cfg):
        self.widen_calls = getattr(self, "widen_calls", 0) + 1


def _run(stub):
    """Execute the real constructor with its collaborators stubbed.

    Everything faked here is I/O (NVML, the checkpoint config, the calibration
    cache). The ordering under test is the constructor's own.
    """
    card = types.SimpleNamespace(
        uuid="GPU-aaaa", name="NVIDIA GeForce RTX 3080", total_mib=GPU_MEM,
        reserved_mib=425,
    )

    def fake_from_server_args(server_args, **kw):
        # The moment the profile key would be built. Capture what it would be
        # built FROM.
        server_args.profile_at_inputs = (
            server_args.chunked_prefill_size,
            server_args.cuda_graph_config.decode.max_bs,
        )
        return types.SimpleNamespace()

    with mock.patch(
        "sglang.srt.server_args.get_device_memory_capacity", return_value=GPU_MEM
    ), mock.patch(
        "sglang.srt.server_args._resolve_rank_gpu_cards", return_value={0: card}
    ), mock.patch(
        "sglang.srt.mem_ledger.engine.DemandInputs.from_server_args",
        side_effect=fake_from_server_args,
    ), mock.patch(
        "sglang.srt.mem_ledger.engine.build_card_ledgers", return_value=[]
    ), mock.patch(
        "sglang.srt.mem_ledger.calibration.load_calibration", return_value=None
    ):
        return stub._build_card_ledgers()


class TestProfileIsResolvedBeforeTheInputsAreBuilt(unittest.TestCase):
    def test_constructor_resolves_the_profile_itself(self):
        """The gated path does no resolving of its own, so this must."""
        stub = _LauncherStub()
        _run(stub)
        self.assertEqual(
            stub.profile_at_inputs,
            (2048, 24),
            "the ledger built its inputs from an unresolved profile, so the "
            "footprint cache is keyed on a digest nothing was cached under",
        )

    def test_session_ceiling_widening_also_runs(self):
        """max_bs must be the FINAL one, not the tier default (#596)."""
        stub = _LauncherStub()
        _run(stub)
        self.assertEqual(getattr(stub, "widen_calls", 0), 1)

    def test_already_resolved_values_are_left_alone(self):
        """Idempotent: the path #596 already fixed stays byte-identical."""
        stub = _LauncherStub()
        stub.chunked_prefill_size = 8192
        stub.cuda_graph_config.decode.max_bs = 64
        _run(stub)
        self.assertEqual(stub.profile_at_inputs, (8192, 64))


if __name__ == "__main__":
    unittest.main()
