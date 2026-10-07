"""Startflags + ENV je Modell (Nutzer 29.09. 11:50Z: "ich will oben im dashboard die kompletten
startflags des jeweiligen boots auch inkl. aller ENV die dafür gesetzt werden. collapsable").

Quelle nur IPC: state.json des laufenden bzw. letzten Boots je Modell (``current`` im
State-Root, sonst der neueste ``kind=boot``):

  groups.{P,D}.launch  = {argv, env, env_base}   Launcher (env_base ab 814c6657bb)
  launch.front         = {argv, env, env_base}   Launcher, ab 814c6657bb
  launch.container     = {image, image_id, rev, profile, run: {argv, env, mounts, ...}}   Host (27B)

Hier wird nur geformt, nichts gelesen außer diesen Dateien: Flags nach Präfix gruppiert,
ENV in "gesetzt" und "Basis/Image" geteilt, Geheimnis-Werte maskiert, P↔D verglichen.
``ver`` ist ein Digest des Gezeigten -- die Seite zeichnet nur neu, wenn er sich ändert.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from typing import Dict, List, Optional, Tuple

from .features import last_boot

MODELS = (("27B", "/spinning/docker-acceptance/27b/state"), ("NF", "/spinning/docker-acceptance/nf/state"))
LAYERS = ("container", "front", "P", "D")
#: Präfix-Gruppen der Flags in dieser Reihenfolge; was keiner passt, steht unter "übrige"
FLAG_GROUPS = (("--weg2-*", "--weg2-"), ("--p-*", "--p-"), ("--d-*", "--d-"),
               ("--hicache*", "--hicache"), ("--speculative*", "--speculative"))
FLAG_REST = "übrige"
#: vom Image mitgebracht, auch wenn der Launcher sie unter env führt
IMAGE_BASE_KEYS = ("CUDA_HOME", "CUDA_VERSION", "CUDA_PATH", "CUDA_ROOT", "PATH", "LD_LIBRARY_PATH",
                   "PYTHONPATH", "HOME", "HOSTNAME", "LANG", "LC_ALL", "TERM", "SHLVL", "PWD", "OLDPWD",
                   "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA",
                   "NV_CUDA_CUDART_VERSION", "VIRTUAL_ENV")
MASK = "***"
#: Wortstücke, die überall ein Geheimnis anzeigen; KEY/TOKEN nur als letztes Stück
#: (HF_TOKEN, ADMIN_API_KEY) -- sonst wären --max-total-tokens oder SGLANG_..._KEY_SCHEME maskiert
_SECRET_PARTS = ("SECRET", "SECRETS", "PASS", "PASSWD", "PASSWORD", "PAT", "CREDENTIAL", "CREDENTIALS",
                 "APIKEY", "BEARER")
_SECRET_LAST = ("TOKEN", "KEY")


def is_secret_name(name: str) -> bool:
    """TOKEN|KEY|SECRET|PASS|PAT im Namen, als Wortstück (PAT nicht in PATH, TOKEN nicht in TOKENS)."""
    parts = [p for p in (name or "").upper().lstrip("-").replace("-", "_").split("_") if p]
    return bool(parts) and (any(p in _SECRET_PARTS for p in parts) or parts[-1] in _SECRET_LAST)


def _is_path_like(name: str, value: str) -> bool:
    """Ein Pfad zu einer Schlüsseldatei darf stehen, ihr Inhalt nie (Nutzer: adminkey-Pfad ok)."""
    u = (name or "").upper()
    return (u.endswith(("FILE", "PATH", "DIR")) or u.endswith(("_FILE", "_PATH", "_DIR"))) \
        and str(value).startswith(("/", "./"))


def mask(name: str, value) -> str:
    v = "" if value is None else str(value)
    if v in ("", MASK, "<redacted>") or not is_secret_name(name) or _is_path_like(name, v):
        return v
    return MASK


def parse_argv(argv: List[str]) -> Tuple[List[str], List[dict]]:
    """(Kommando, Flags). Ein Flag ist ein Token mit ``--``; ``--k=v`` und ``--k v [v ...]``;
    ohne Wert ist es ein Schalter (value None). Reihenfolge bleibt."""
    argv = [str(a) for a in (argv or [])]
    i, cmd, flags = 0, [], []
    while i < len(argv) and not argv[i].startswith("--"):
        cmd.append(argv[i])
        i += 1
    while i < len(argv):
        tok = argv[i]
        i += 1
        if not tok.startswith("--"):
            # a stray positional after the flags: keep it visible as its own row
            flags.append({"flag": tok, "value": None, "positional": True})
            continue
        name, eq, val = tok.partition("=")
        vals = [val] if eq else []
        while not eq and i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        value = None if not vals else (vals[0] if len(vals) == 1 else " ".join(vals))
        flags.append({"flag": name, "value": value})
    for f in flags:
        f["value"] = None if f["value"] is None else mask(f["flag"], f["value"])
        f["json"] = _pretty_json(f["value"])
        f["group"] = flag_group(f["flag"])
    return cmd, flags


def _pretty_json(v: Optional[str]) -> Optional[str]:
    if not v or v[0] not in "{[":
        return None
    try:
        return json.dumps(json.loads(v), indent=2, sort_keys=True, ensure_ascii=False)
    except ValueError:
        return None


def flag_group(flag: str) -> str:
    for name, pre in FLAG_GROUPS:
        if flag.startswith(pre):
            return name
    return FLAG_REST


def _grouped(flags: List[dict]) -> List[dict]:
    order = [n for n, _ in FLAG_GROUPS] + [FLAG_REST]
    out = []
    for g in order:
        rows = [f for f in flags if f["group"] == g]
        if rows:
            out.append({"group": g, "rows": rows})
    return out


def _env_split(env: Dict[str, str], env_base: Dict[str, str]) -> Tuple[List[list], List[list]]:
    """(gesetzt, Basis/Image), beide sortiert, Werte maskiert."""
    setv, base = [], []
    for k, v in sorted((env or {}).items()):
        (base if k in IMAGE_BASE_KEYS else setv).append([k, mask(k, v)])
    for k, v in sorted((env_base or {}).items()):
        base.append([k, mask(k, v)])
    base.sort()
    return setv, base


def _flag_map(flags: List[dict]) -> Dict[str, str]:
    """Flag -> Wert für den Vergleich; ein Flag mehrfach = Werte mit " | " verbunden."""
    m: Dict[str, str] = {}
    for f in flags:
        if f.get("positional"):
            continue
        v = "(Schalter)" if f["value"] is None else f["value"]
        m[f["flag"]] = v if f["flag"] not in m else m[f["flag"]] + " | " + v
    return m


def diff(a: Dict[str, str], b: Dict[str, str]) -> dict:
    """P↔D: gleich (einmal), verschieden, nur links, nur rechts -- je nach Schlüssel sortiert."""
    ks = sorted(set(a) | set(b))
    return {
        "same": [[k, a[k]] for k in ks if k in a and k in b and a[k] == b[k]],
        "differ": [[k, a[k], b[k]] for k in ks if k in a and k in b and a[k] != b[k]],
        "only_a": [[k, a[k]] for k in ks if k in a and k not in b],
        "only_b": [[k, b[k]] for k in ks if k in b and k not in a],
    }


def layer_view(launch: Optional[dict]) -> Optional[dict]:
    """Eine Ebene (Front/P/D) aus {argv, env, env_base}."""
    if not launch:
        return None
    cmd, flags = parse_argv(launch.get("argv") or [])
    setv, base = _env_split(launch.get("env") or {}, launch.get("env_base") or {})
    argv_shown = cmd + [t for f in flags for t in ([f["flag"]] if f["value"] is None
                                                     else [f["flag"], f["value"]])]
    return {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "flags": _grouped(flags),
        "n_flags": sum(1 for f in flags if not f.get("positional")),
        "env_set": setv,
        "env_base": base,
        "n_env": len(setv),
        "n_env_base": len(base),
        "argv_raw": " ".join(shlex.quote(t) for t in argv_shown),
        "env_raw": "\n".join("%s=%s" % (k, shlex.quote(v)) for k, v in setv),
        "has_env_base": "env_base" in launch,
        "_flagmap": _flag_map(flags),
        "_envmap": dict(setv),
    }


def container_view(c: Optional[dict]) -> Optional[dict]:
    """launch.container (Host): Kopf-Zeilen, docker-run-Env, Mounts/Geräte, Rohzeile."""
    if not c:
        return None
    run = c.get("run") or {}
    meta = [[k, c.get(k)] for k in ("image", "image_id", "rev", "line", "profile", "profile_path") if c.get(k)]
    for k in ("name", "gpus", "shm_size", "ipc", "network", "pid"):
        if run.get(k) not in (None, ""):
            meta.append(["run." + k, run.get(k)])
    for k in ("devices", "env_files", "cap_add", "security_opt", "extra"):
        if run.get(k):
            meta.append(["run." + k, ", ".join(str(x) for x in run[k])])
    for k, v in sorted((run.get("ulimits") or {}).items()):
        meta.append(["run.ulimit." + k, v])
    for k, v in sorted((run.get("cgroup") or {}).items()):
        meta.append(["run.cgroup." + k, v])
    mounts = [[m.get("src"), m.get("dst"), m.get("mode") or "", m.get("type") or ""] for m in (run.get("mounts") or [])
              if isinstance(m, dict)]
    setv = [[k, mask(k, v)] for k, v in sorted((run.get("env") or {}).items())]
    argv = [str(a) for a in (run.get("argv") or [])]
    # docker's own flags are single-dash too (-e K=V): mask the value after -e/--env by its key
    shown, i = [], 0
    while i < len(argv):
        t = argv[i]
        shown.append(t)
        if t in ("-e", "--env") and i + 1 < len(argv):
            k, eq, v = argv[i + 1].partition("=")
            shown.append(k + eq + (mask(k, v) if eq else ""))
            i += 1
        i += 1
    return {
        "meta": [[k, str(v)] for k, v in meta],
        "mounts": mounts,
        "env_set": setv,
        "env_base": [],
        "n_env": len(setv),
        "n_env_base": 0,
        "n_flags": len(meta) + len(mounts),
        "argv_raw": " ".join(shlex.quote(t) for t in shown),
        "env_raw": "\n".join("%s=%s" % (k, shlex.quote(v)) for k, v in setv),
    }


def model_view(model: str, st: Optional[dict]) -> dict:
    if not st:
        return {"model": model, "boot_id": None}
    groups = st.get("groups") or {}
    launch = st.get("launch") or {}
    layers = {
        "container": container_view(launch.get("container")),
        "front": layer_view(launch.get("front")),
        "P": layer_view((groups.get("P") or {}).get("launch")),
        "D": layer_view((groups.get("D") or {}).get("launch")),
    }
    pd = None
    if layers["P"] and layers["D"]:
        pd = {"flags": diff(layers["P"]["_flagmap"], layers["D"]["_flagmap"]),
              "env": diff(layers["P"]["_envmap"], layers["D"]["_envmap"])}
    for v in layers.values():
        if v:
            v.pop("_flagmap", None)
            v.pop("_envmap", None)
    lc = st.get("lifecycle") or {}
    out = {
        "model": model,
        "boot_id": st.get("boot_id"),
        "tag": st.get("tag"),
        "image": st.get("image"),
        "image_id": st.get("image_id"),
        "rev": st.get("rev"),
        "profile": st.get("profile"),
        "container": st.get("container"),
        "lifecycle": lc.get("state"),
        "lifecycle_since": lc.get("since_ts"),
        "layers": layers,
        "missing": [k for k in LAYERS if not layers[k]],
        "counts": {k: {"flags": v["n_flags"], "env": v["n_env"]} for k, v in layers.items() if v},
        "pd": pd,
    }
    out["ver"] = hashlib.sha1(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return out


def snapshot(models=MODELS) -> dict:
    views = [model_view(m, last_boot(root)) for m, root in models]
    return {"src": "state.json", "models": views, "ver": "-".join(v.get("ver") or "0" for v in views)}
