# SPDX-License-Identifier: Apache-2.0
"""#785 desk pre-stage: the meta TP layout, on three gloo ranks, off-GPU.

WHAT THIS GATES AND WHAT IT DELIBERATELY DOES NOT. This reproduces the PURE
LAYOUT ARITHMETIC -- given the real checkpoint config and the flip vector, does
a meta-device TP model lay out to the same total the runtime later measures?
It runs three real processes because the answer is rank-dependent: a shard plan
is only exercised by a group that actually has that many ranks, and a
single-process stand-in would silently take the even-split path.

It is NOT the acceptance gate. The deciding gate for the derivation is the BOOT
INSTRUMENT, which computes the tail inside a real boot and checks it against
the layout totals that same boot logs when it builds the TP stack. This test
cannot replace it, for a reason worth stating: the compressed-tensors linear
scheme is chosen from ``torch.cuda.get_device_capability()`` at CONSTRUCTION
time (compressed_tensors.py:530), so the layout is in principle
capability-dependent -- and this rig is mixed, rank 0 on sm120 and ranks 1/2 on
sm86. A desk run has to pin one capability. It reproduces rank 0 anyway, which
says the scheme choice does not move this layout; that is a measurement, not a
guarantee.

REFERENCE, from boot_735_default791b.log:952/962/975 (the boot's own
"TP stack built: vector [32, 16, 16], arena ... (pp ... / tp ...)" lines):

    layout_tp = {0: 15925.80, 1: 8573.78, 2: 8573.78} MiB
    layout_pp = {0: 16007.47, 1: 8008.96, 2: 10789.22} MiB

Ranks 1 and 2 are byte-identical, as a vector of 32,16,16 requires. That
equality is a free correctness check: a derivation that gets the total right
but the sharding wrong is unlikely to make those two agree.
"""

import json
import os
import socket
import subprocess
import sys

import pytest

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-vocabint8-embed"
VECTOR = [32, 16, 16]

#: MiB, from the boot log named in the module docstring.
REFERENCE_TP_MIB = {0: 15925.80, 1: 8573.78, 2: 8573.78}

#: A layout is a budget input, so the bar is 1 MiB against a ~16 GiB number.
TOLERANCE_MIB = 1.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Worker. This module runs itself as the rank body; see _run_ranks.
# ---------------------------------------------------------------------------


def _worker(rank: int, world: int, port: str) -> None:
    import torch

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.utils import scoped_tp_partition_ratios
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.managers.arena_tail_probe import plan_meta_layout
    from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
        set_default_torch_dtype,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    import sglang.srt.server_args as SA

    # DESK FIXTURES, and none of the three touches layout arithmetic:
    #  - is_cuda() gates FLA/mamba SUPPORT in argument validation
    #  - get_device_capability selects the compressed-tensors scheme; pinned to
    #    sm86 so an off-GPU run is reproducible (see the module docstring)
    #  - pynccl is a communicator; off-GPU there is no device to bind it to,
    #    and no arena layout consults one
    SA.is_cuda = lambda: True
    torch.cuda.get_device_capability = lambda *a, **k: (8, 6)
    ps.should_build_pynccl = lambda *a, **k: False

    server_args = ServerArgs(
        model_path=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=world,
        pp_size=1,
        page_size=1,
        disable_overlap_schedule=True,
        kv_cache_dtype="fp8_e4m3",
        context_length=262144,
        mamba_checkpoint_interval=8192,
        max_mamba_cache_size=24,
        device="cuda",
    )
    get_context().set_server_args(server_args)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=world,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    model_config = ModelConfig(
        model_path=MODEL, trust_remote_code=True, dtype="bfloat16"
    )
    initialize_dp_attention(server_args=server_args, model_config=model_config)

    load_config = LoadConfig()
    quant_config = _get_quantization_config(model_config, load_config)
    with scoped_tp_partition_ratios(VECTOR):
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(model_config, load_config, quant_config)

    layout = plan_meta_layout(checkpoint_param_dict(model))
    print(
        "DESKLAYOUT "
        + json.dumps({"rank": rank, "total_mib": layout.total_bytes / 1048576.0}),
        flush=True,
    )


def _run_ranks(world: int = 3):
    port = str(_free_port())
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = port
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, str(rank), str(world), port],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(world)
    ]
    totals = {}
    for rank, proc in enumerate(procs):
        try:
            out, err = proc.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"rank {rank} did not finish within 900 s")
        assert proc.returncode == 0, (
            f"rank {rank} exited {proc.returncode}:\n{err[-4000:]}"
        )
        for line in out.splitlines():
            if line.startswith("DESKLAYOUT "):
                rec = json.loads(line[len("DESKLAYOUT ") :])
                totals[rec["rank"]] = rec["total_mib"]
    return totals


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------


needs_checkpoint = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason=f"checkpoint not present at {MODEL}"
)


@pytest.fixture(scope="module")
def totals():
    """One 3-process run shared by the assertions below."""
    return _run_ranks()


@needs_checkpoint
def test_the_meta_layout_reproduces_every_measured_tp_total(totals):
    assert set(totals) == set(REFERENCE_TP_MIB), f"missing ranks: {sorted(totals)}"
    for rank, reference in REFERENCE_TP_MIB.items():
        derived = totals[rank]
        assert abs(derived - reference) <= TOLERANCE_MIB, (
            f"rank {rank}: derived {derived:.2f} MiB against the boot's own "
            f"{reference:.2f} MiB. A layout is subtracted from a memory "
            f"budget, so a difference here is a real difference in the pool."
        )


@needs_checkpoint
def test_the_two_equal_shares_lay_out_identically(totals):
    """Vector 32,16,16 gives ranks 1 and 2 the same share.

    Checks the SHARDING rather than the total: a derivation that reproduced
    the aggregate while splitting it wrongly would break this equality.
    """
    assert totals[1] == totals[2]
    assert totals[0] > totals[1]
