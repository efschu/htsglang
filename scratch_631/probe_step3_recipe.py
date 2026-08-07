"""Step 3: parse the LITERAL recipe command lines, exactly as written.

Validates the argv strings that go into the boot recipe, not a kwargs
paraphrase of them, so a typo in the recipe is a failure here. No GPUs:
CUDA_VISIBLE_DEVICES=99. The two harness stubs are the same as
probe_step1_composition.py -- accelerator presence and card identity, both
unavailable on a cardless box, neither a gate under test.
"""
import os
import shlex

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-2B"


def install_stubs():
    import sglang.srt.server_args as sa_mod

    sa_mod.is_cuda = lambda: True
    # get_device() probes real hardware; on the rig it returns 'cuda'. The
    # recipe deliberately does NOT pass --device, matching the rig boots.
    sa_mod.get_device = lambda *a, **k: "cuda"

    def _fake_cards(gpu_ids):
        spec = {
            0: ("NVIDIA GeForce RTX 5090", 32768, 1),
            1: ("NVIDIA GeForce RTX 3080", 20480, 0),
            2: ("NVIDIA GeForce RTX 3080", 20480, 2),
        }
        out = {}
        for o in sorted({int(g) for g in gpu_ids}):
            name, total, nvml_idx = spec[o]
            out[o] = sa_mod._RankGpuCard(
                cuda_ordinal=o,
                nvml_index=nvml_idx,
                uuid=f"GPU-fake-{o}",
                pci_bus_id=f"0000:0{o}:00.0",
                name=name,
                total_mib=total,
                free_mib=total - 1024,
                reserved_mib=424,
            )
        return out

    sa_mod._resolve_rank_gpu_cards = _fake_cards
    return sa_mod


# --------------------------------------------------------------------------
# ARM 1 -- runnable TODAY on this rig. Disjoint topology: no shared card, so
# neither the NCCL >= 2.30 co-location threshold nor the MPS daemon applies.
# PREFILL is a genuine PP group (pp=2) on the two 3080s; DECODE is tp=1 on
# the 5090. Speculation is absent on both arms by construction (#631a).
# --------------------------------------------------------------------------

PREFILL_ARM1 = f"""
--model-path {MODEL}
--served-model-name pd631-arm1
--disaggregation-mode prefill
--disaggregation-transfer-backend mooncake
--disaggregation-bootstrap-port 8998
--pp-size 2
--tp-size 1
--base-gpu-id 1
--page-size 1
--kv-cache-dtype fp8_e4m3
--context-length 16384
--max-running-requests 8
--mem-fraction-static 0.60
--trust-remote-code
--enable-metrics
--host 127.0.0.1
--port 31241
"""

DECODE_ARM1 = f"""
--model-path {MODEL}
--served-model-name pd631-arm1
--disaggregation-mode decode
--disaggregation-transfer-backend mooncake
--tp-size 1
--base-gpu-id 0
--page-size 1
--kv-cache-dtype fp8_e4m3
--context-length 16384
--max-running-requests 8
--mem-fraction-static 0.60
--trust-remote-code
--enable-metrics
--host 127.0.0.1
--port 31243
"""

# --------------------------------------------------------------------------
# ARM 2 -- the LITERAL #631 ask: PP prefill group (pp=3) + TP decode group
# (tp=3, uneven weighted DCP), both resident on the same three cards.
# Requires --disaggregation-topology colocated-process, which this rig
# refuses (no MPS daemon; NCCL 2.28.9/2.29.7 < 2.30). Parsed here to prove
# the ARGUMENTS are well-formed, so that the only thing standing between this
# recipe and a boot is the environment, not the flags.
# --------------------------------------------------------------------------

PREFILL_ARM2 = """
--model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
--served-model-name pd631-arm2
--disaggregation-mode prefill
--disaggregation-transfer-backend mooncake
--disaggregation-bootstrap-port 8998
--pp-size 3
--tp-size 1
--rank-gpu-id 0,1,2
--rank-gpu-memory-mib 13000,8000,8000
--page-size 1
--kv-cache-dtype fp8_e4m3
--context-length 16384
--max-running-requests 8
--trust-remote-code
--enable-metrics
--host 127.0.0.1
--port 31251
"""

DECODE_ARM2 = """
--model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
--served-model-name pd631-arm2
--disaggregation-mode decode
--disaggregation-transfer-backend mooncake
--tp-size 3
--rank-tp-ratio 3,2,2
--rank-kv-ratio 3,2,2
--rank-gpu-id 0,1,2
--rank-gpu-memory-mib 17000,10000,10000
--page-size 1
--kv-cache-dtype fp8_e4m3
--context-length 16384
--max-running-requests 8
--trust-remote-code
--enable-metrics
--host 127.0.0.1
--port 31253
"""

CASES = [
    ("ARM1 prefill  (pp=2, disjoint, 3080 x2)", PREFILL_ARM1, {}),
    ("ARM1 decode   (tp=1, disjoint, 5090)", DECODE_ARM1, {}),
    ("ARM2 prefill  (pp=3, colocated)", PREFILL_ARM2, {}),
    (
        "ARM2 decode   (tp=3, uneven weighted DCP)",
        DECODE_ARM2,
        {"SGLANG_UNEVEN_DCP": "1", "SGLANG_UNEVEN_DCP_WEIGHTED": "1"},
    ),
]


def main():
    install_stubs()
    from sglang.srt.server_args import prepare_server_args

    print("\n==== STEP 3: LITERAL RECIPE ARGV PARSE ====")
    for name, block, env in CASES:
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        argv = shlex.split(block)
        try:
            sa = prepare_server_args(argv)
            detail = (
                f"tp={sa.tp_size} pp={sa.pp_size} dcp={sa.dcp_size} "
                f"layout={sa.draft_kv_layout} spec={sa.speculative_algorithm} "
                f"page={sa.page_size} kvdtype={sa.kv_cache_dtype}"
            )
            print(f"{name:<45} | PARSED   | {detail}")
        except SystemExit as e:
            print(f"{name:<45} | ARGPARSE-EXIT({e.code})")
        except Exception as e:  # noqa: BLE001
            head = str(e).strip().splitlines()
            print(
                f"{name:<45} | REFUSED[{type(e).__name__}] | "
                f"{(head[0] if head else '')[:170]}"
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("==== END ====")


if __name__ == "__main__":
    main()
