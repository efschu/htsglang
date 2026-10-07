import importlib, sys, os, time
mods = ["flliper", "flliper.srt.environ", "flliper.srt.server_args", "flliper.srt.pdflip.form",
        "flliper.srt.pdflip.launcher", "flliper.srt.pdflip.front", "flliper.srt.managers.scheduler",
        "flliper.srt.managers.phase_flip_runtime", "flliper.srt.entrypoints.http_server",
        "flliper.srt.layers.moe.expert_offload", "flliper.srt.layers.moe.slot_ledger",
        "flliper.srt.layers.moe.expert_map", "flliper.srt.planner.expert_residency",
        "flliper.srt.flip_nextflash_plan", "flliper.srt.flip_cold_tier_share",
        "flliper.srt.distributed.device_communicators.barlink_env_guard", "flliper.srt.name_compat", "flliper.srt.compat_shims", "flliper.srt.pdflip.host_ledger", "flliper.srt.pdflip.ring_table", "flliper.srt.pdflip.tools.vram_hires_report", "flliper.launch_server"]
ok = True
for m in mods:
    t = time.time()
    try:
        importlib.import_module(m); print(f"ok   {m} ({time.time()-t:.1f}s)")
    except Exception as e:
        ok = False; print(f"FAIL {m}: {type(e).__name__}: {e}")
old = sorted(k for k in sys.modules if k == "sglang" or k.startswith("sglang.") or ".weg2" in k)
print("sglang*/weg2 modules loaded:", len(old), old[:5])
from flliper.srt.environ import envs
names = [n for n in dir(envs) if n.startswith("FLLIPER_")]
print("envs FLLIPER_*:", len(names), "FLLIPER_PDFLIP_*:", sum(n.startswith("FLLIPER_PDFLIP_") for n in names),
      "SGLANG_*:", sum(n.startswith("SGLANG_") for n in dir(envs)), "WEG2 in names:", sum("WEG2" in n for n in dir(envs)))
import flliper.srt.flip_nextflash_plan as P
print("PdFlipKvRelayInfeasible exported:", "PdFlipKvRelayInfeasible" in P.__all__ and hasattr(P, "PdFlipKvRelayInfeasible"))
import flliper.srt.flip_nextflash_plan as P2; from flliper.srt.flip_nextflash_plan import *  # noqa: star import checks __all__
print("star-import over __all__: ok")
print("IMPORTS", "PASS" if ok and not old else "FAIL")
import flliper.srt.name_compat as nc, flliper.srt.pdflip.host_ledger as hl, flliper.srt.pdflip.launcher as L
print("name_compat CANONICAL_SIDE:", nc.CANONICAL_SIDE, "MEASURED_RECORD_NAME:", hl.MEASURED_RECORD_NAME)
print("shm families old+new:", [p for p in L.SHM_OWN_PREFIXES if "seq" in p])
print("HOME cache flliper created by import:", os.path.lexists(os.path.expanduser("~/.cache/flliper")))
