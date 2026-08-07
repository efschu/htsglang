"""Step 1b: what the gate order does the moment #631a's refusal is lifted.

#631a raises at server_args.py:5916 (_handle_pd_disaggregation), which is
BEFORE the #108 gate (:6113), the #636 contract (:6114) and the #642 gate
(:6123). So today none of those three is reachable on a PD arm that asks for
speculation -- probe_step1_composition.py proves that. This probe removes only
that one refusal and re-runs the same cells, to establish:

  * whether the downstream gates are correctly ordered after the lift,
  * that #642 is not merely dead code but becomes live and REFUSES the
    replicated layout on a token-sharded decode arm (its can-fail arm),
  * whether the PP prefill arm is covered by #642 at all.

LIFT SIMULATION, stated precisely: ``_handle_pd_disaggregation`` is wrapped so
that ``speculative_algorithm`` is hidden (set to None) for the duration of the
hook and restored afterwards. That suppresses the #631a refusal and nothing
else in the resolution pipeline. CAVEAT: it also suppresses the other
spec-dependent checks INSIDE that hook -- notably
``--disaggregation-decode-enable-radix-cache`` + spec. No cell below enables
the decode radix cache, so that check has nothing to say here; a real lift
must keep it.
"""
import os

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"

BASE = dict(
    model_path=MODEL,
    trust_remote_code=True,
    context_length=8192,
    kv_cache_dtype="fp8_e4m3",
    page_size=1,
    disaggregation_transfer_backend="mooncake",
    device="cuda",
)

TP_DECODE = dict(
    tp_size=3,
    rank_tp_ratio=[3, 2, 2],
    rank_kv_ratio=[3, 2, 2],
    rank_gpu_id=[0, 1, 2],
    rank_gpu_memory_mib=[28000, 17000, 17000],
)

PP_PREFILL = dict(
    tp_size=1,
    pp_size=3,
    rank_gpu_id=[0, 1, 2],
    rank_gpu_memory_mib=[28000, 17000, 17000],
)

SPEC = dict(
    speculative_algorithm="NEXTN",
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
)


def install_stubs():
    import sglang.srt.server_args as sa_mod

    sa_mod.is_cuda = lambda: True

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


def lift_631a(sa_mod):
    ServerArgs = sa_mod.ServerArgs
    orig = ServerArgs._handle_pd_disaggregation

    def lifted(self):
        saved = self.speculative_algorithm
        self.speculative_algorithm = None
        try:
            orig(self)
        finally:
            self.speculative_algorithm = saved

    ServerArgs._handle_pd_disaggregation = lifted


def run_cell(sa_mod, name, env, over):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        kw = dict(BASE)
        kw.update(over)
        sa = sa_mod.ServerArgs(**kw)
        return (name, "ACCEPTED", f"layout={sa.draft_kv_layout} "
                f"spec={sa.speculative_algorithm} dcp={sa.dcp_size} "
                f"tp={sa.tp_size} pp={sa.pp_size}")
    except Exception as e:  # noqa: BLE001
        first = str(e).strip().splitlines()
        return (name, f"REFUSED[{type(e).__name__}]",
                (first[0] if first else type(e).__name__)[:240])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


W = {"SGLANG_UNEVEN_DCP": "1", "SGLANG_UNEVEN_DCP_WEIGHTED": "1"}

CELLS = [
    ("D' decode tp3/dcp3 + spec + layout=dcp        [expect ACCEPT]", W,
     dict(TP_DECODE, **SPEC, draft_kv_layout="dcp", disaggregation_mode="decode")),
    ("E' decode tp3/dcp3 + spec + layout=replicated [expect #642 REFUSE]", W,
     dict(TP_DECODE, **SPEC, draft_kv_layout="replicated",
          disaggregation_mode="decode")),
    ("P1 prefill pp3/tp1 + spec + layout=replicated [#642 coverage?]", W,
     dict(PP_PREFILL, **SPEC, draft_kv_layout="replicated",
          disaggregation_mode="prefill")),
    ("P2 prefill pp3/tp1 + spec + layout=dcp        [#108 coverage?]", W,
     dict(PP_PREFILL, **SPEC, draft_kv_layout="dcp",
          disaggregation_mode="prefill")),
    ("P3 prefill pp3/tp1 + NO spec                  [arm-1 baseline]", W,
     dict(PP_PREFILL, draft_kv_layout="replicated",
          disaggregation_mode="prefill")),
]

if __name__ == "__main__":
    sa_mod = install_stubs()
    lift_631a(sa_mod)
    rows = [run_cell(sa_mod, *c) for c in CELLS]
    width = max(len(r[0]) for r in rows)
    print("\n==== STEP 1b: GATE ORDER WITH #631a LIFTED ====")
    for name, verdict, detail in rows:
        print(f"{name:<{width}} | {verdict:<20} | {detail}")
    print("==== END ====")
