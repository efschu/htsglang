"""Step 1 probe: does --draft-kv-layout dcp compose with disaggregation_mode?

Executes ServerArgs construction for the reachable configuration matrix and
records, per cell, which gate (if any) refuses and with what message. No GPUs:
run with CUDA_VISIBLE_DEVICES=99.
"""
import os

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"

BASE = dict(
    model_path=MODEL,
    tp_size=3,
    rank_tp_ratio=[3, 2, 2],
    rank_kv_ratio=[3, 2, 2],
    rank_gpu_id=[0, 1, 2],
    rank_gpu_memory_mib=[28000, 17000, 17000],
    trust_remote_code=True,
    context_length=8192,
    kv_cache_dtype="fp8_e4m3",
    page_size=1,
    disaggregation_transfer_backend="mooncake",
    device="cuda",
)

SPEC = dict(
    speculative_algorithm="NEXTN",
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
)


def build(**over):
    import sglang.srt.server_args as sa_mod
    from sglang.srt.server_args import ServerArgs

    # HARNESS STUB, not a behaviour change: this box has no accelerator, so
    # is_cuda() is False and _handle_dcp_validation would take the
    # "non-HIP platform" else-branch that the rig never takes. Patching it
    # makes the parse pipeline follow the rig's CUDA branch. dcp_size is
    # deliberately NOT passed: production lets SGLANG_UNEVEN_DCP auto-set it
    # in _handle_uneven_tp (:5957), which runs AFTER _handle_dcp_validation
    # (:5922).
    sa_mod.is_cuda = lambda: True

    # HARNESS STUB #2: fabricate the rig's card identity. NVML/CUDA are
    # unavailable at CUDA_VISIBLE_DEVICES=99, and _resolve_rank_gpu_cards
    # refuses to guess (correctly, #392). This supplies the three real cards
    # -- one RTX 5090 32768 MiB at CUDA ordinal 0, two RTX 3080 20480 MiB --
    # so the gate ordering under test is reached. It fabricates HARDWARE
    # IDENTITY only; no gate predicate is patched.
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

    kw = dict(BASE)
    kw.update(over)
    return ServerArgs(**kw)


def run_cell(name, env, over):
    old = {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        sa = build(**over)
        return (name, "ACCEPTED", f"draft_kv_layout={sa.draft_kv_layout} "
                f"spec={sa.speculative_algorithm} dcp={sa.dcp_size}")
    except Exception as e:  # noqa: BLE001
        head = str(e).strip().splitlines()
        head = head[0] if head else type(e).__name__
        return (name, f"REFUSED[{type(e).__name__}]", head[:300])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


WEIGHTED = {"SGLANG_UNEVEN_DCP_WEIGHTED": "1", "SGLANG_UNEVEN_DCP": "1"}
AUTODIS = dict(WEIGHTED, SGLANG_PD_AUTO_DISABLE_SPEC="1")

CELLS = [
    # (name, env, overrides)
    ("A monolithic + spec + layout=replicated", WEIGHTED,
     dict(SPEC, draft_kv_layout="replicated")),
    ("B monolithic + spec + layout=dcp", WEIGHTED,
     dict(SPEC, draft_kv_layout="dcp")),
    ("C prefill-arm + spec + layout=dcp", WEIGHTED,
     dict(SPEC, draft_kv_layout="dcp", disaggregation_mode="prefill")),
    ("D decode-arm  + spec + layout=dcp", WEIGHTED,
     dict(SPEC, draft_kv_layout="dcp", disaggregation_mode="decode")),
    ("E decode-arm  + spec + layout=replicated", WEIGHTED,
     dict(SPEC, draft_kv_layout="replicated", disaggregation_mode="decode")),
    ("F decode-arm  + spec + layout=dcp + AUTO_DISABLE", AUTODIS,
     dict(SPEC, draft_kv_layout="dcp", disaggregation_mode="decode")),
    ("G decode-arm  + NO spec + layout=dcp", WEIGHTED,
     dict(draft_kv_layout="dcp", disaggregation_mode="decode")),
    ("H decode-arm  + NO spec + layout=replicated", WEIGHTED,
     dict(draft_kv_layout="replicated", disaggregation_mode="decode")),
]

if __name__ == "__main__":
    rows = [run_cell(*c) for c in CELLS]
    width = max(len(r[0]) for r in rows)
    print("\n==== STEP 1 COMPOSITION MATRIX ====")
    for name, verdict, detail in rows:
        print(f"{name:<{width}} | {verdict:<22} | {detail}")
    print("==== END ====")
