import importlib, json, os, sys
pkg = sys.argv[1]; sub = "weg2" if pkg == "sglang" else "pdflip"
L = importlib.import_module(f"{pkg}.srt.{sub}.launcher")
_orig = L.build_env
PFX = ("SGLANG_", "FLLIPER_", "NCCL_", "TMS_", "WEG2_", "PDFLIP_")
def _wrap(*a, **k):
    env = _orig(*a, **k)
    items = sorted((kk, vv) for kk, vv in env.items() if str(kk).startswith(PFX))
    sys.stderr.write("ENVDUMP %s %s\n" % (k.get("group", "?"), json.dumps(items)))
    return env
L.build_env = _wrap
sys.exit(L.main(sys.argv[2:]))
