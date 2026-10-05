"""Kartenplaner (Item 510): Planer-Aufzeichnungen ("Records") aus echten Boots BAUEN (Schreibtisch-Werkzeug).

Dieses Paket liegt bewusst AUSSERHALB von rigdash/: der Dienst liest keine Logs (Regel DASHBOARD-AUS-IPC, test_no_new_log_parsers).
Nur dieses Werkzeug liest einmalig die Boot-Logs und schreibt strukturierte JSON-Records; der Dienst liest nur die JSON-Dateien.


Ein Record hält fest, was UNSER Planer für ein Profil auf dem Referenz-Rig entschieden hat und was
der Boot daraus gemacht hat, mit Herkunft je Zahl:

  * ``vram_plan``      die vram_plan.json des Boots (Planer-Ausgabe, plan_id wird beim Laden nachgerechnet)
  * ``budget_lines``   die Zeilen "budget <G> group=<G> ordinal=<i> ..." aus dem Front-Log des Launchers
  * ``launch``         argv/env der Gruppen aus state.json (Boots ab 29.09.) bzw. argv aus dem Rang-Log-Kopf
  * ``rank_posts``     je Gruppe und Rang die Posten aus den Rang-Logs (Gewichte, Draft, Mamba, KV, Graphen)
  * ``flag_docs``      je Flag/Env die Erklärung aus dem Code (argparse-help) und dem Profil (Kommentar), mit Quelle
  * ``weights``        Dateigröße der Checkpoints (gemessen: stat)

Dieses Modul liest nur Dateien (state.json, vram_plan.json, Logs, Profildateien); es startet nichts.
``build`` läuft auf dem Schreibtisch und legt JSON unter ``kartenplan_data/`` ab, das mit dem Code
ausgeliefert wird.  ``load`` prüft die plan_id gegen den Inhalt (Veränderung fällt auf).
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

SCHEMA = "kartenplan.record/1"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rigdash", "kartenplan_data")
DOCKER_DIR = "/spinning/gpu-arb/docker"

RE_BUDGET = re.compile(r"budget (?P<label>\S+) group=(?P<group>\S+) ordinal=(?P<ord>\d+) nvml_idx=(?P<nvml>\d+) "
                       r"(?P<name>.+?): (?P<mib>\d+) MiB = (?P<terms>.*)$")
RE_RANK = re.compile(r"^\[[\d\- :T]+(?:Z)? (?P<kind>TP|PP)(?P<idx>\d+)\] (?P<msg>.*)$")
RE_WEIGHT = re.compile(r"Load weight end\. elapsed=[\d.]+ s, type=(?P<type>\w+), .*?mem usage=(?P<gb>[\d.]+) GB")
RE_KVALLOC = re.compile(r"KV Cache is allocated\. dtype: (?P<dtype>\S+), #tokens: (?P<tok>\d+), "
                        r"K size: (?P<k>[\d.]+) GB, V size: (?P<v>[\d.]+) GB")
RE_MAMBA = re.compile(r"Mamba Cache is allocated\. max_mamba_cache_size: (?P<n>\d+), conv_state size: (?P<conv>[\d.]+)GB, "
                      r"ssm_state size: (?P<ssm>[\d.]+)GB(?: intermediate_ssm_state_cache size: (?P<issm>[\d.]+)GB)?"
                      r"(?: intermediate_conv_window_cache size: (?P<iconv>[\d.]+)GB)?")
RE_KVPOSTS = re.compile(r"KV budget posts \(GiB\): (?P<posts>.*?) \| rest=(?P<rest>[\d.]+) \| measured free=(?P<free>[\d.]+)")
RE_GRAPH = re.compile(r"Capture (?P<what>.*?) CUDA graph end\. elapsed=[\d.]+ s, mem usage=(?P<gb>[\d.]+) GB")
RE_POOLEND = re.compile(r"Memory pool end\. avail mem=(?P<gb>[\d.]+) GB")
RE_INITDIST = re.compile(r"Init torch distributed ends\. elapsed=[\d.]+ s, mem usage=(?P<gb>[\d.]+) GB")
RE_PFGRAPH = re.compile(r"PREFILL-GRAPH captured .*?capture_mib=(?P<mib>[\d.]+)")
GIB = 1024.0


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_id_ok(plan: dict) -> bool:
    """Wie weg2/vram_plan.compute_plan_id: sha256 über den Plan ohne plan_id."""
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    return plan.get("plan_id") == "sha256:" + hashlib.sha256(canonical(body).encode()).hexdigest()


def _gib_to_mib(x: str) -> int:
    return int(round(float(x) * GIB))


# --------------------------------------------------------------------------- Front-Log
def parse_budget_lines(path: str) -> List[dict]:
    out = []
    try:
        with open(path, errors="replace") as fh:
            for n, ln in enumerate(fh, 1):
                if " budget " not in ln:
                    continue
                m = RE_BUDGET.search(ln.rstrip("\n"))
                if m:
                    d = m.groupdict()
                    out.append({"seq": len(out), "log_line": n, "label": d["label"], "group": d["group"],
                                "ordinal": int(d["ord"]), "nvml": int(d["nvml"]), "name": d["name"],
                                "budget_mib": int(d["mib"]), "terms": d["terms"][:900], "parsed": parse_terms(d["terms"])})
    except OSError:
        pass
    return out


RE_FLOOR = re.compile(r"floor (\d+) source=([A-Za-z0-9_\-]+)")
RE_CORR = re.compile(r"corridor (\d+)")
RE_AWAKE_REST = re.compile(r"awake_rest (\d+) \(D_AWAKE_REST_MIB")
RE_BOOKED = re.compile(r"awake_rest_booked (\d+) \(RECORD (?P<prov>[^;]*);")
RE_CARVE = re.compile(r"driver_carve (\d+)")
RE_DORMANT = re.compile(r"dormant_other (\d+)")
RE_GROWTH = re.compile(r"served_dormant_growth (\d+)")
RE_OVER = re.compile(r"measured_awake_overshoot (\d+)")
RE_L15 = re.compile(r"l15 (\d+) \(L1\.5 post\)")
RE_RESERVE = re.compile(r"reserve=(\d+)")
RE_BOOT_PROV = re.compile(r"\(boot ([^)]*)\)")


def parse_terms(t: str) -> dict:
    """Die Terme einer Budgetzeile als Zahlen (der Dienst liest nur diese Felder, nie den Text)."""
    def num(rx):
        m = rx.search(t)
        return int(m.group(1)) if m else None
    fl = RE_FLOOR.search(t)
    bk = RE_BOOKED.search(t)
    bp = RE_BOOT_PROV.search(t)
    return {"carve": num(RE_CARVE), "dormant": num(RE_DORMANT), "growth": num(RE_GROWTH), "over": num(RE_OVER),
            "corridor": num(RE_CORR), "floor": int(fl.group(1)) if fl else None, "floor_source": fl.group(2) if fl else None,
            "reserve": num(RE_RESERVE), "awake_rest": num(RE_AWAKE_REST), "booked": int(bk.group(1)) if bk else None,
            "booked_prov": bk.group("prov") if bk else None, "l15": num(RE_L15), "over_prov": ("boot " + bp.group(1)) if bp else None}


def final_budgets(lines: List[dict]) -> Dict[str, List[Optional[int]]]:
    """Je Gruppe die Budgets der LETZTEN Rechnung (das, womit gebootet wurde), nach Ordinal."""
    res: Dict[str, Dict[int, int]] = {}
    for ln in lines:
        res.setdefault(ln["group"], {})[ln["ordinal"]] = ln["budget_mib"]
    return {g: [v.get(i) for i in range(max(v) + 1)] for g, v in res.items()}


# --------------------------------------------------------------------------- Rang-Logs
def parse_rank_log(path: str) -> Dict[str, dict]:
    """Posten je Rang aus einem Rang-Log (D: TPk, P: PPk).  Erste Treffer je Rang (Boot-Aufbau)."""
    ranks: Dict[str, dict] = {}

    def rk(kind, idx):
        return ranks.setdefault("%s%s" % (kind.lower(), idx), {"kv_pools": [], "graphs": [], "weights": []})

    try:
        fh = open(path, errors="replace")
    except OSError:
        return ranks
    with fh:
        for ln in fh:
            if ln.startswith("argv: "):
                ranks.setdefault("_argv", {})["text"] = ln[6:].rstrip("\n")
                continue
            m = RE_RANK.match(ln)
            if not m:
                continue
            r = rk(m.group("kind"), m.group("idx"))
            msg = m.group("msg")
            if "Load weight end" in msg:
                w = RE_WEIGHT.search(msg)
                if w:
                    r["weights"].append({"type": w.group("type"), "mib": _gib_to_mib(w.group("gb"))})
            elif "KV Cache is allocated" in msg:
                k = RE_KVALLOC.search(msg)
                if k:
                    r["kv_pools"].append({"dtype": k.group("dtype"), "tokens": int(k.group("tok")),
                                          "k_mib": _gib_to_mib(k.group("k")), "v_mib": _gib_to_mib(k.group("v"))})
            elif "Mamba Cache is allocated" in msg and "mamba" not in r:
                k = RE_MAMBA.search(msg)
                if k:
                    parts = {x: float(k.group(x) or 0) for x in ("conv", "ssm", "issm", "iconv")}
                    r["mamba"] = {"slots": int(k.group("n")), "mib": _gib_to_mib(str(sum(parts.values()))),
                                  "conv_mib": _gib_to_mib(str(parts["conv"])), "ssm_mib": _gib_to_mib(str(parts["ssm"])),
                                  "intermediate_mib": _gib_to_mib(str(parts["issm"] + parts["iconv"]))}
            elif "KV budget posts (GiB)" in msg and "kv_posts" not in r:
                k = RE_KVPOSTS.search(msg)
                if k:
                    posts = {}
                    for part in k.group("posts").split(", "):
                        if "=" in part:
                            name, val = part.rsplit("=", 1)
                            try:
                                posts[name.strip()] = _gib_to_mib(val)
                            except ValueError:
                                pass
                    r["kv_posts"] = {"posts_mib": posts, "rest_mib": _gib_to_mib(k.group("rest")),
                                     "measured_free_mib": _gib_to_mib(k.group("free"))}
            elif "CUDA graph end" in msg:
                g = RE_GRAPH.search(msg)
                if g:
                    r["graphs"].append({"what": g.group("what"), "mib": _gib_to_mib(g.group("gb"))})
            elif "PREFILL-GRAPH captured" in msg and "prefill_graph_mib" not in r:
                g = RE_PFGRAPH.search(msg)
                if g:
                    r["prefill_graph_mib"] = int(float(g.group("mib")))
            elif "Memory pool end" in msg and "avail_after_pool_mib" not in r:
                g = RE_POOLEND.search(msg)
                if g:
                    r["avail_after_pool_mib"] = _gib_to_mib(g.group("gb"))
            elif "Init torch distributed ends" in msg and "init_dist_mib" not in r:
                g = RE_INITDIST.search(msg)
                if g:
                    r["init_dist_mib"] = _gib_to_mib(g.group("gb"))
    return ranks


def argv_of_log(ranks: Dict[str, dict]) -> List[str]:
    t = (ranks.get("_argv") or {}).get("text")
    if not t:
        return []
    try:
        return shlex.split(t)
    except ValueError:
        return t.split()


# --------------------------------------------------------------------------- Flag-Erklärungen
def _str_of(node) -> Optional[str]:
    """Konstante Zeichenkette oder f-String (Platzhalter bleiben als {...} stehen)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{...}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, r = _str_of(node.left), _str_of(node.right)
        return (l or "") + (r or "") if (l is not None or r is not None) else None
    return None


def argparse_help(source: str) -> Dict[str, str]:
    """Erklärungen je Flag: ``add_argument("--flag", ..., help=...)`` und die ServerArgs-Felder
    ``name: A[type, Arg(help=...)]`` per ast (Zeichenketten-Konkatenation und f-Strings werden aufgelöst)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")]
            help_ = None
            for kw in node.keywords:
                if kw.arg == "help":
                    help_ = _str_of(kw.value)
            if names and help_:
                for n in names:
                    out.setdefault(n, " ".join(help_.split()))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            for sub in ast.walk(node.annotation):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "Arg":
                    for kw in sub.keywords:
                        if kw.arg == "help":
                            h = _str_of(kw.value)
                            if h:
                                out.setdefault("--" + node.target.id.replace("_", "-"), " ".join(h.split()))
    return out


def sourced_env_files(profile_names: List[str]) -> List[str]:
    """Profildateien + alles, was sie per ``source`` einbinden (profiles/ und profiles_release/)."""
    seen: List[str] = []
    todo = list(profile_names)
    dirs = [os.path.join(DOCKER_DIR, "profiles"), os.path.join(DOCKER_DIR, "profiles_release")]
    while todo:
        n = todo.pop(0)
        for d in dirs:
            p = os.path.join(d, n if n.endswith(".env") else n + ".env")
            if os.path.isfile(p) and p not in seen:
                seen.append(p)
                try:
                    txt = open(p, errors="replace").read()
                except OSError:
                    continue
                for m in re.finditer(r'source\s+"[^"]*?/([A-Za-z0-9_.\-]+\.env)"', txt):
                    todo.append(m.group(1))
    return seen


def comment_above(files: List[str], needle: str, max_lines: int = 6) -> Optional[dict]:
    pat = re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(needle) + r"(?![A-Za-z0-9_\-])")
    for p in files:
        try:
            lines = open(p, errors="replace").read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("#") or not pat.search(ln):
                continue
            block = []
            j = i - 1
            while j >= 0 and lines[j].lstrip().startswith("#") and len(block) < max_lines:
                block.append(lines[j].lstrip("# ").rstrip())
                j -= 1
            tail = ln.split("#", 1)[1].strip() if "#" in ln else ""
            text = " ".join(reversed(block)).strip()
            if tail:
                text = (text + " | " + tail).strip(" |")
            if text:
                return {"text": text[:700], "source": "%s:%d" % (os.path.basename(p), i + 1)}
    return None


def harvest_flag_docs(flags: List[str], envs: List[str], code_help: Dict[str, str], env_files: List[str]) -> Dict[str, dict]:
    docs: Dict[str, dict] = {}
    for f in flags:
        d = {}
        if f in code_help:
            d["code_help"] = code_help[f][:700]
        c = comment_above(env_files, f)
        if c:
            d["profile_comment"] = c
        docs[f] = d
    for e in envs:
        c = comment_above(env_files, e)
        if c:
            docs[e] = {"profile_comment": c}
    return docs


def git_show(rev: str, path: str) -> str:
    try:
        return subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------- Aufbau eines Records
def dir_bytes(path: str) -> Optional[int]:
    if os.path.isfile(path):
        return os.path.getsize(path)
    if os.path.isdir(path):
        tot = 0
        for p in glob.glob(os.path.join(path, "*.safetensors")):
            tot += os.path.getsize(p)
        return tot or None
    return None


SECRET_FLAGS = ("--admin-api-key", "--admin-key-file")


def strip_secrets(argv: List[str]) -> List[str]:
    """Admin-Schlüssel und Schlüsseldatei gehören in keine Seite (redact.py); sie werden schon im Record entfernt."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        name = a.split("=", 1)[0]
        if name in SECRET_FLAGS:
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    return out


def flags_of(argv: List[str]) -> List[str]:
    return [a.split("=", 1)[0] for a in argv if a.startswith("--")]


def build_record(spec: dict) -> dict:
    rec: dict = {"schema": SCHEMA, "id": spec["id"], "profile": spec["profile_id"], "line": spec["line"],
                 "built_from": {}}
    ev = spec["evidence_root"]
    tag = spec["tag"]
    front = glob.glob(os.path.join(ev, "boot_weg2_%s_*.front.log" % tag))
    dlog = glob.glob(os.path.join(ev, "boot_weg2_%s_*.D.log" % tag))
    plog = glob.glob(os.path.join(ev, "boot_weg2_%s_*.P.log" % tag))
    rec["built_from"]["evidence"] = {"front": front[:1], "D": dlog[:1], "P": plog[:1]}
    rec["boot"] = {"tag": tag, "rev": spec["rev"], "image": spec.get("image"), "boot_profile": spec["boot_profile"],
                   "state_dir": spec.get("state_dir"), "ipc_era": bool(spec.get("state_dir"))}
    blines = parse_budget_lines(front[0]) if front else []
    rec["budget_lines"] = blines
    rec["budgets_final"] = final_budgets(blines)
    ranks = {"D": parse_rank_log(dlog[0]) if dlog else {}, "P": parse_rank_log(plog[0]) if plog else {}}
    rec["argv_from_log"] = {g: argv_of_log(r) for g, r in ranks.items()}
    rec["rank_posts"] = {g: {k: v for k, v in r.items() if not k.startswith("_")} for g, r in ranks.items()}
    argv = {"D": rec["argv_from_log"]["D"], "P": rec["argv_from_log"]["P"]}
    env = {"D": {}, "P": {}}
    front_cmd: List[str] = []
    sd = spec.get("state_dir")
    if sd:
        st = json.load(open(os.path.join(sd, "state.json")))
        rec["boot"].update({"boot_id": st.get("boot_id"), "image": st.get("image"), "lifecycle": (st.get("lifecycle") or {}).get("state"),
                            "form": {g: (st["groups"][g].get("form") or {}) for g in ("P", "D") if g in st.get("groups", {})}})
        for g in ("P", "D"):
            gl = (st.get("groups", {}).get(g) or {}).get("launch") or {}
            if gl.get("argv"):
                argv[g] = list(gl["argv"])
            env[g] = dict(gl.get("env") or {})
        front_cmd = list(((st.get("launch") or {}).get("front") or {}).get("argv") or [])
        vp = os.path.join(sd, "vram_plan.json")
        if os.path.exists(vp):
            plan = json.load(open(vp))
            rec["vram_plan"] = plan
            rec["vram_plan_ok"] = plan_id_ok(plan)
        # Rang-Zustand (Sitze, KV-Token)
        rs = {}
        for jf in glob.glob(os.path.join(sd, "rankstate", "*", "*.json")):
            try:
                j = json.load(open(jf))
            except (OSError, ValueError):
                continue
            rs["%s.tp%dpp%d" % (j.get("group"), j.get("tp_rank", 0), j.get("pp_rank", 0))] = {
                k: j.get(k) for k in ("role", "seats", "kv", "pp_size", "tp_size", "page_size")}
        rec["rank_state"] = rs
    argv = {g: strip_secrets(a) for g, a in argv.items()}
    front_cmd = strip_secrets(front_cmd)
    rec["argv_from_log"] = {g: strip_secrets(a) for g, a in rec["argv_from_log"].items()}
    rec["launch"] = {"argv": argv, "env": env, "front_argv": front_cmd,
                     "redacted": "Admin-Schlüssel und -Schlüsseldatei (%s) werden vom Launcher je Boot erzeugt und nie gezeigt." % ", ".join(SECRET_FLAGS)}
    bj = glob.glob(os.path.join(ev, "docker_%s" % tag, "boot_%s.json" % tag))
    if bj:
        try:
            b = json.load(open(bj[0]))
            rec["boot_json"] = {k: b.get(k) for k in ("tag", "tip", "stamp", "budgets", "dc_measured_p", "dc_expect_d")}
            rec["boot_json"]["cards"] = [{k: c.get(k) for k in ("nvml_index", "uuid", "name", "total_mib", "reserved_mib")}
                                         for c in b.get("cards") or []]
            rec["built_from"]["boot_json"] = bj[0]
        except (OSError, ValueError):
            pass
    # Erklärungen aus Code (Revision des Boots) und Profil
    code_help: Dict[str, str] = {}
    for rel in ("python/sglang/srt/weg2/launcher.py", "python/sglang/srt/server_args.py", "python/sglang/srt/weg2/front.py"):
        code_help.update({k: v for k, v in argparse_help(git_show(spec["rev"], rel)).items() if k not in code_help})
    all_flags = sorted({f for g in argv.values() for f in flags_of(g)} | set(flags_of(front_cmd)))
    all_envs = sorted({k for g in env.values() for k in g})
    files = sourced_env_files([spec["boot_profile"], spec.get("release_profile") or spec["boot_profile"]])
    rec["flag_docs"] = harvest_flag_docs(all_flags, all_envs, code_help, files)
    rec["flag_docs_sources"] = {"code": "argparse help am Boot-Rev %s" % spec["rev"], "profile_files": [os.path.basename(f) for f in files]}
    # Gewichte
    weights = []
    for g in ("D",):
        a = argv[g]
        for key in ("--model-path", "--model", "--speculative-draft-model-path"):
            if key in a:
                p = a[a.index(key) + 1]
                b = dir_bytes(p)
                weights.append({"flag": key, "path": p, "bytes": b, "src": "stat, gemessen" if b else "Pfad nicht lesbar"})
    rec["weights"] = weights
    return rec


def save(rec: dict, data_dir: str = DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    p = os.path.join(data_dir, rec["id"] + ".json")
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    return p


def load(record_id: str, data_dir: str = DATA_DIR) -> Optional[dict]:
    p = os.path.join(data_dir, record_id + ".json")
    try:
        with open(p) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if rec.get("schema") != SCHEMA:
        raise ValueError("Record %s: Schema %r unbekannt" % (record_id, rec.get("schema")))
    if rec.get("vram_plan") and not plan_id_ok(rec["vram_plan"]):
        rec["vram_plan_ok"] = False
    return rec


#: die Aufzeichnungen, die der Kartenplaner kennt (Boots mit Beleg; Quellen in der Item-Datei 510)
SPECS = [
    {"id": "27b-int8", "profile_id": "27b-int8", "line": "27b", "tag": "dkr27browauthoritycut43bar1fs10030906",
     "boot_profile": "27b-row-authority-cut43", "release_profile": "27b", "rev": "55c95a89c7",
     "state_dir": "/spinning/docker-acceptance/27b/state/27bbf-boot-20261003T090641Z-30ca",
     "evidence_root": "/spinning/docker-acceptance/27b/evidence"},
    {"id": "nf-int4-abl", "profile_id": "nf-int4-abl", "line": "nf", "tag": "dkrnfint4h6ablbar1dauer10030924",
     "boot_profile": "nf-int4-h6-abl", "release_profile": "nf-int4", "rev": "044316dd1a",
     "state_dir": "/spinning/docker-acceptance/nf/state/nfint4h6abldauer-boot-20261003T092427Z-bb0c",
     "evidence_root": "/spinning/docker-acceptance/nf/evidence"},
    {"id": "27b-nvfp4-dual", "profile_id": "27b-nvfp4-dual", "line": "27b", "tag": "dkr27bnvfp4dual1mpsleepbar1fs10030828",
     "boot_profile": "27b-nvfp4-dual1m-psleep", "release_profile": "27b-nvfp4-dual", "rev": "bd2e3bc22d",
     "state_dir": "/spinning/docker-acceptance/27b/state/27bbf-boot-20261003T082805Z-c7ae",
     "evidence_root": "/spinning/docker-acceptance/27b/evidence"},
    {"id": "27b-fp8", "profile_id": "27b-fp8", "line": "27b", "tag": "dkr27bfp8bar1final09260250",
     "boot_profile": "27b-fp8", "release_profile": "27b-fp8", "rev": "103712cdb2", "state_dir": None,
     "evidence_root": "/spinning/docker-acceptance/27b/evidence"},
    {"id": "27b-gguf-iq4xs", "profile_id": "27b-gguf-iq4xs", "line": "27b", "tag": "dkr27bggufbar1final09260238",
     "boot_profile": "27b-gguf", "release_profile": "27b-gguf", "rev": "103712cdb2", "state_dir": None,
     "evidence_root": "/spinning/docker-acceptance/27b/evidence"},
]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Kartenplaner-Records aus echten Boots bauen (Schreibtisch, liest nur)")
    ap.add_argument("ids", nargs="*", help="Record-IDs (leer = alle)")
    ap.add_argument("--out", default=DATA_DIR)
    ns = ap.parse_args(argv)
    for spec in SPECS:
        if ns.ids and spec["id"] not in ns.ids:
            continue
        rec = build_record(spec)
        print("%s -> %s (%d Budgetzeilen, plan %s, %d Flags erklärt)" % (
            spec["id"], save(rec, ns.out), len(rec["budget_lines"]),
            "ok" if rec.get("vram_plan_ok") else ("fehlt" if "vram_plan" not in rec else "ID PASST NICHT"),
            sum(1 for v in rec["flag_docs"].values() if v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
