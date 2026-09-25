#!/usr/bin/env python3
"""JIT-Vorbau fuer das htsglang-Release-Image -- ENTWURF (27B-Sitz R, 24.09.2026).

NICHT GELAUFEN. Laeuft im `docker build` (keine GPU, kein libcuda.so.1) und
erzeugt die JIT-Artefakte, die ein Boot sonst mit nvcc baut:

1. FlashInfer-Module je Arch-Verzeichnis, in der REIHENFOLGE DES SERVERS.
   flashinfer legt sein Cache-Verzeichnis beim IMPORT fest (flashinfer/jit/env.py
   `_get_workspace_dir_name()`, aus CompilationContext: FLASHINFER_CUDA_ARCH_LIST,
   sonst die sichtbaren Geraete), nimmt die -gencode-Flags aber beim Schreiben
   jeder build.ninja aus einem FRISCHEN CompilationContext (jit/cpp_ext.py). Der
   sglang-Server setzt FLASHINFER_CUDA_ARCH_LIST erst NACH dem Import
   (model_runner.py:2494 -> utils/common.py:1547 set_cuda_arch(): "12.0a" auf der
   5090, "8.6" auf den 3080). Folge am Rig: Verzeichnis 0.6.14/120f, Flags
   compute_120a (Beleg: cached_ops/*/build.ninja). Ein Vorbau, der die Reihenfolge
   nicht nachbildet, schreibt compute_120f -> ninja baut beim Boot alles neu
   (fi_jit_cache_check.py, Kopf). Darum je Verzeichnis ein eigener Kindprozess:
   Variable VOR dem Import auf den Verzeichnis-Wert, NACH dem Import auf den
   Server-Wert.

2. barlink-Erweiterungen unter den Laufzeit-Namen. Zur Laufzeit ist die
   Arch-Liste die Union der Gruppe (barlink_device.py:681-712) -> auf diesem Rig
   ["8.6", "12.0"] -> Namen barlink_device_ext_cuda_86_120,
   barlink_bar1_ext_cuda_86_120 (+ barlink_bar1_dmabuf_ext, C++ gegen die
   NV-Header). Hier wird _resolve_build_arches durch genau diese Union ersetzt
   und der modul-eigene Ladepfad gerufen (gleiche Quellen, gleiche Flags, gleiches
   Build-Verzeichnis unter TORCH_EXTENSIONS_DIR).

3. Der weg2-TMS-Preload-Hook ueber das Skript des Baums
   (scripts/weg2/tms/build_tms_preload.sh, Name = sha256 der Quellen, Default-
   Ausgabe /spinning/gpu-arb/weg2/tms -- dieselbe, die der Launcher beim Boot
   wiederverwendet, launcher.py:7105-7114).

Was NICHT hier gebaut wird:
- sglang jit_kernel (tvm-ffi, u.a. die Marlin-Kerne fuer FP8/NVFP4/INT4 auf sm_86):
  dieser Cache ist INHALTSADRESSIERT (Verzeichnisname = Modul + Hash aus Arch,
  Wrappern und Flags, OHNE Include-Pfade; Wiederverwendung nur nach passender
  Provenienz source_hash/build_hash/vendor/arch, jit_kernel/utils.py:196-345, 600-700).
  prepare_context.sh legt deshalb den Rig-Cache ~/.cache/tvm-ffi ins Image; was zum
  Baum passt, wird geladen, der Rest liegt ungenutzt.
- Triton (~/.triton): Saat per Volume aus dem Rig (Host-Skript), nicht im Image.
- FlashInfer-Module, deren URI hier nicht abgebildet ist: im Bericht "skipped",
  nie verschwiegen; sie bauen beim ersten Boot ins Cache-Volume.

Aufruf (im Dockerfile):
  python prebuild_jit.py --tree /opt/htsglang/src --venv /opt/htsglang/venv \
      --archs 8.6,12.0 --flashinfer-manifest flashinfer_modules.txt \
      --report /opt/htsglang/JIT_PREBUILD.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

#: Cache-Verzeichnis -> (Wert VOR dem Import, Wert NACH dem Import = set_cuda_arch()).
#: 12.0 wird von flashinfer zu (12, "0f") normalisiert -> "120f"; set_cuda_arch()
#: haengt fuer major >= 9 ein "a" an -> "12.0a". 8.6 bleibt 8.6.
#: ACHTUNG, am Rig gemessen (build.ninja, 24.09.): im Verzeichnis 120f stehen ZWEI
#: Flag-Sorten. Module aus einem FRISCHEN CompilationContext (norm, sampling, topk,
#: batch_prefill/decode, ...) tragen compute_120a; Module aus dem beim IMPORT angelegten
#: `current_compilation_context` (flashinfer/jit/core.py:138: gemm_sm120) und solche mit
#: festen sm120f-Flags (fp4_quantization_120f) tragen compute_120f. Der Vorbau erzeugt
#: beides von selbst richtig (Import mit "12.0", danach "12.0a"); geprueft wird je Modul
#: gegen die gencode-Menge, die prepare_context.sh aus der Rig-build.ninja liest.
FI_DIRS = {
    "120f": ("12.0", "12.0a"),
    "86": ("8.6", "8.6"),
}
ARCH_TO_DIRKEY = {"12.0": "120f", "8.6": "86"}

_DT = r"([a-z0-9]+)"
_PREFILL_RE = re.compile(
    rf"^batch_prefill_with_kv_cache_dtype_q_{_DT}_dtype_kv_{_DT}_dtype_o_{_DT}"
    rf"_dtype_idx_{_DT}_head_dim_qk_(\d+)_head_dim_vo_(\d+)_posenc_(\d+)"
    r"_use_swa_(True|False)_use_logits_cap_(True|False)_f16qk_(True|False)$"
)
_DECODE_RE = re.compile(
    rf"^batch_decode_with_kv_cache_dtype_q_{_DT}_dtype_kv_{_DT}_dtype_o_{_DT}"
    rf"_dtype_idx_{_DT}_head_dim_qk_(\d+)_head_dim_vo_(\d+)_posenc_(\d+)"
    r"_use_swa_(True|False)_use_logits_cap_(True|False)$"
)
_XQA_RE = re.compile(
    rf"^xqa_input_{_DT}_kv_cache_{_DT}_output_{_DT}_page_size_(\d+)_head_dim_(\d+)"
    r"_head_group_ratio_(\d+)_use_sliding_window_(True|False)_use_spec_dec_(True|False)"
    r"_spec_q_seq_len_(\d+)$"
)
#: URIs ohne Parameter -> (Modul, Generator). Alles andere: skipped (benannt).
#: gemm_sm120 / fp4_quantization_120f: FP8- und NVFP4-GEMM-Pfade auf der 5090
#: (Formate 27B-FP8, 27B-NVFP4, NF-NVFP4); sparse_mla_sm120 steht ebenfalls im Rig-Cache.
_FIXED = {
    "cascade": ("flashinfer.jit.cascade", "gen_cascade_module"),
    "norm": ("flashinfer.jit.norm", "gen_norm_module"),
    "sampling": ("flashinfer.jit.sampling", "gen_sampling_module"),
    "topk": ("flashinfer.jit.topk", "gen_topk_module"),
    "page": ("flashinfer.jit.page", "gen_page_module"),
    "quantization": ("flashinfer.jit.quantization", "gen_quantization_module"),
    "gemm_sm120": ("flashinfer.jit.gemm.core", "gen_gemm_sm120_module"),
    "fp4_quantization_120f": ("flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm120f_module"),
    "sparse_mla_sm120": ("flashinfer.jit.mla", "gen_sparse_mla_sm120_module"),
}


def _dtypes():
    import torch

    return {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
        "i32": torch.int32,
    }


def _spec_for(uri: str):
    """JitSpec fuer eine URI aus dem Rig-Cache, oder None (= skipped)."""
    import importlib

    from flashinfer.jit.attention.modules import (
        gen_batch_decode_module,
        gen_batch_prefill_module,
    )

    d = _dtypes()
    m = _PREFILL_RE.match(uri)
    if m:
        q, kv, o, idx, dqk, dvo, pe, swa, cap, f16 = m.groups()
        return gen_batch_prefill_module(
            "fa2", d[q], d[kv], d[o], d[idx], int(dqk), int(dvo), int(pe),
            swa == "True", cap == "True", f16 == "True",
        )
    m = _DECODE_RE.match(uri)
    if m:
        q, kv, o, idx, dqk, dvo, pe, swa, cap = m.groups()
        return gen_batch_decode_module(
            d[q], d[kv], d[o], d[idx], int(dqk), int(dvo), int(pe),
            swa == "True", cap == "True",
        )
    m = _XQA_RE.match(uri)
    if m:
        from flashinfer.jit.xqa import gen_xqa_module

        i, kv, o, ps, hd, g, swa, _spec, q = m.groups()
        return gen_xqa_module(d[i], d[kv], int(ps), int(hd), int(g), swa == "True",
                              d[o], q_seq_len=int(q))
    if uri in _FIXED:
        mod, fn = _FIXED[uri]
        return getattr(importlib.import_module(mod), fn)()
    return None


_GENCODE_RE = re.compile(r"-gencode=arch=compute_[0-9a-z]+,code=sm_[0-9a-z]+")


def fi_child(dirkey: str, entries: list) -> int:
    """EIN Arch-Verzeichnis, eigener Prozess (das Verzeichnis steht ab Import fest).

    ``entries`` = [(uri, erwartete gencode-Menge oder None)] aus dem Manifest.
    """
    before, after = FI_DIRS[dirkey]
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = before
    import flashinfer  # noqa: F401  -- legt FLASHINFER_WORKSPACE_DIR fest
    from flashinfer.jit import env as jit_env

    got = jit_env.FLASHINFER_WORKSPACE_DIR.name
    if got != dirkey:
        print(json.dumps({"dirkey": dirkey, "fatal": f"workspace dir {got!r} != {dirkey!r}"}))
        return 2
    rows = []
    for uri, want in entries:
        row = {"dirkey": dirkey, "uri": uri}
        t0 = time.time()
        # Flag-Kontext JE MODUL wie am Rig: ein Modul, das der Rig mit compute_120f gebaut hat, entstand im
        # Import-Kontext (FLASHINFER_CUDA_ARCH_LIST wie beim Import, "12.0" -> 120f); alle anderen im Server-Kontext
        # nach sglang set_cuda_arch() ("12.0a" -> 120a). Gleiche Flags = gleiche build.ninja = kein Neubau zur
        # Laufzeit. Geprueft 25.09. per write_ninja an allen 52 Manifest-Eintraegen (ohne Kompilat).
        wants_import_ctx = bool(want) and any("compute_120f" in g for g in want)
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = before if wants_import_ctx else after
        row["flag_context"] = "import" if wants_import_ctx else "server"
        try:
            spec = _spec_for(uri)
            if spec is None:
                row["status"] = "skipped (kein Generator-Mapping; baut beim ersten Boot)"
            elif spec.name != uri:
                row["status"] = (f"skipped (Mapping liefert {spec.name!r} statt der Rig-URI; "
                                 f"baut beim ersten Boot)")
            else:
                spec.build(verbose=False)
                got = sorted(set(_GENCODE_RE.findall(spec.ninja_path.read_text())))
                row["gencode"] = got
                if want is None:
                    row["status"] = "built (keine Rig-Erwartung im Manifest)"
                elif got == want:
                    row["status"] = "built"
                else:
                    # gebaut, aber mit anderen Flags als am Rig: die Laufzeit baut es dann neu. Kein Abbruchgrund.
                    row["status"] = f"built (WARN gencode {got} != Rig {want})"
                row["so"] = str(spec.jit_library_path) if spec.jit_library_path.exists() else None
        except Exception as exc:  # noqa: BLE001 -- jeder Fehler wird benannt berichtet
            row["status"] = f"error: {type(exc).__name__}: {exc}"[:400]
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
    print(json.dumps({"dirkey": dirkey, "workspace": str(jit_env.FLASHINFER_WORKSPACE_DIR), "rows": rows}))
    return 0


def barlink_child(archs: list) -> int:
    from sglang.srt.distributed.device_communicators import barlink_bar1_ext as b1
    from sglang.srt.distributed.device_communicators import barlink_device as bd

    union = sorted(archs, key=lambda a: tuple(int(x) for x in a.split(".")))
    # Laufzeit: all_gather_object ueber die Gruppe; hier dieselbe Union ohne Gruppe.
    bd._resolve_build_arches = lambda cpu_group: {"cuda": list(union)}
    rows = []
    for name, call in (
        ("barlink_device_ext", lambda: bd._load_ext(None)),
        ("barlink_bar1_ext", lambda: b1.load_collective_ext(None)),
        ("barlink_bar1_dmabuf_ext", lambda: b1.load_dmabuf_ext()),
    ):
        t0 = time.time()
        try:
            ext = call()
            if ext is None:
                # WITH_NV_HEADERS=0 (Default): gewollt, kein Fehler -- der Entrypoint
                # verweigert dann bar1 mit Grund bzw. faehrt den NCCL-Pfad.
                status = f"declined: {b1.dmabuf_reason()}"
            else:
                status = f"built: {getattr(ext, '__file__', '?')}"
        except Exception as exc:  # noqa: BLE001
            status = f"error: {type(exc).__name__}: {exc}"[:400]
        rows.append({"ext": name, "archs": union, "status": status, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({"barlink": rows}))
    return 0


def cpu_ext_child() -> int:
    """CPU-Erweiterungen, die die Linie zur Laufzeit per torch.utils.cpp_extension baut. Am Rig belegt
    (25.09., Boot weg2rc2f8 P/D-Log): hicache_hash_cpp_avx2 (HiCache-Hash, #1409) -- 109 s Einzelschritte
    laut .ninja_log, sonst beim ersten Container-Boot. Der ISA-Name kommt aus der Bau-CPU (AVX2 auf dem
    5950X des Hosts = Laufzeit-CPU); ein Image auf einer CPU ohne AVX2 baut zur Laufzeit hicache_hash_cpp_baseline."""
    rows = []
    t0 = time.time()
    try:
        from sglang.srt.mem_cache.cpp_utils import native_hash as nh
        mod = nh._load_via_torch()
        status = f"built: {getattr(mod, '__file__', '?')}"
    except Exception as exc:  # noqa: BLE001
        status = f"error: {type(exc).__name__}: {exc}"[:400]
    rows.append({"ext": "hicache_hash_cpp", "status": status, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({"cpu_ext": rows}))
    return 0


def _read_manifest(path: str) -> dict:
    """Zeilen `dirkey uri [gencode;gencode;...]` (3. Feld: die gencode-Menge der
    Rig-build.ninja, von prepare_context.sh eingetragen) -> {dirkey: [(uri, want)]}.

    Die gencode-Tokens werden per Regex aus dem Feld gezogen, NICHT am Komma getrennt: jedes Token
    enthaelt selbst ein Komma (-gencode=arch=compute_86,code=sm_86). Der Host-Bau vom 25.09. (02:04Z)
    hat an genau diesem Fehler alle 21 Module von 86 gebaut und dann als "error" gemeldet."""
    out: dict = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            dirkey, uri = parts[0], parts[1]
            if uri == "tmp":
                continue
            want = sorted(set(_GENCODE_RE.findall(parts[2]))) if len(parts) > 2 else None
            want = want or None
            out.setdefault(dirkey, []).append((uri, want))
    return out


def _run_child(argv: list) -> dict:
    r = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv,
                       capture_output=True, text=True)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    try:
        payload = json.loads(last[-1]) if last else {}
    except json.JSONDecodeError:
        payload = {}
    payload["rc"] = r.returncode
    if r.returncode != 0 or not last:
        payload["stderr_tail"] = r.stderr[-2000:]
    return payload


SECTIONS = ("flashinfer", "barlink", "cpu_ext", "tms")


def report_problems(report: dict) -> list:
    """Alle Fehler im GANZEN Bericht (auch aus einem frueheren Teil-Lauf) plus fehlende Abschnitte, je Zeile benannt."""
    probs = []
    for d in report.get("flashinfer", []) or []:
        if d.get("rc") or d.get("fatal"):
            probs.append(f"flashinfer {d.get('dirkey')}: Kindprozess rc={d.get('rc')} fatal={d.get('fatal')}")
        probs += [f"flashinfer {d.get('dirkey')} {r.get('uri', '?')[:90]}: {str(r.get('status'))[:200]}"
                  for r in d.get("rows", []) if str(r.get("status", "")).startswith("error")]
    for key in ("barlink", "cpu_ext"):
        sec = report.get(key)
        if sec is None:
            continue
        if sec.get("rc"):
            probs.append(f"{key}: Kindprozess rc={sec.get('rc')}")
        probs += [f"{key} {r.get('ext')}: {str(r.get('status'))[:200]}"
                  for r in sec.get(key, []) if str(r.get("status", "")).startswith("error")]
    t = report.get("tms")
    if t is not None and t.get("rc"):
        probs.append(f"tms: rc={t.get('rc')} {str(t.get('stderr_tail', ''))[-200:]}")
    probs += [f"fehlt: Abschnitt {k} (noch nicht gelaufen)" for k in SECTIONS if k not in report]
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--venv")
    ap.add_argument("--archs", default="8.6,12.0")
    ap.add_argument("--flashinfer-manifest")
    ap.add_argument("--report", default="/opt/htsglang/JIT_PREBUILD.json")
    ap.add_argument("--only", default="flashinfer,barlink,cpu_ext,tms")
    ap.add_argument("--strict", action="store_true",
                    help="jeder Modul-Fehler bricht den Bau ab (Exit 1). Ohne: Fehler laut im Log und im Bericht "
                         "(verdict INCOMPLETE), Exit 0 -- das Modul baut dann beim ersten Boot per JIT")
    ap.add_argument("--continue-report", action="store_true",
                    help="einen vorhandenen --report fortfuehren: Abschnitte dieses Laufs ersetzen, die uebrigen behalten, "
                         "verdict ueber den ganzen Bericht (Schichtfolge: FlashInfer vor src/, der Rest danach)")
    ap.add_argument("--fi-child")          # intern
    ap.add_argument("--barlink-child")     # intern
    ap.add_argument("--cpu-ext-child", action="store_true")   # intern
    ns = ap.parse_args()

    if ns.fi_child:
        entries = _read_manifest(ns.flashinfer_manifest).get(ns.fi_child, [])
        return fi_child(ns.fi_child, entries)
    if ns.barlink_child:
        return barlink_child(ns.barlink_child.split(","))
    if ns.cpu_ext_child:
        return cpu_ext_child()

    # Die Laufzeit setzt keine dieser Variablen; der Vorbau darf sie auch nicht erben.
    for var in ("FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST", "CC",
                "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_VERBOSE"):
        os.environ.pop(var, None)

    archs = [a.strip() for a in ns.archs.split(",") if a.strip()]
    only = set(ns.only.split(","))
    unknown = only - set(SECTIONS)
    if unknown:
        print(f"[prebuild] FATAL: unbekannte Abschnitte {sorted(unknown)} (erlaubt: {', '.join(SECTIONS)})", flush=True)
        return 2
    if "tms" in only and not ns.tree:
        print("[prebuild] FATAL: tms braucht --tree", flush=True)
        return 2
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict = {}
    if ns.continue_report and os.path.exists(ns.report):
        with open(ns.report) as fh:
            report = json.load(fh)
        for k in only:
            report.pop(k, None)
    report.setdefault("archs", archs)
    report.setdefault("started", now)
    report.setdefault("stages", []).append({"only": sorted(only), "started": now})
    rc = 0

    if "flashinfer" in only:
        manifest = _read_manifest(ns.flashinfer_manifest)
        report["flashinfer"] = []
        for arch in archs:
            dirkey = ARCH_TO_DIRKEY[arch]
            res = _run_child(["--fi-child", dirkey, "--flashinfer-manifest", ns.flashinfer_manifest])
            report["flashinfer"].append(res)
            n_err = sum(1 for r in res.get("rows", []) if str(r.get("status", "")).startswith("error"))
            if res.get("rc") or res.get("fatal") or n_err:
                rc = 1
            for r in res.get("rows", []):
                if str(r.get("status", "")).startswith("error"):
                    print(f"[prebuild]   FEHLER {dirkey} {r.get('uri', '?')[:90]}: {str(r['status'])[:300]}", flush=True)
            if res.get("fatal") or (res.get("rc") and res.get("stderr_tail")):
                print(f"[prebuild]   FEHLER {dirkey} Kindprozess rc={res.get('rc')} fatal={res.get('fatal')} "
                      f"stderr: {str(res.get('stderr_tail', ''))[-600:]}", flush=True)
            print(f"[prebuild] flashinfer {dirkey}: "
                  f"{sum(1 for r in res.get('rows', []) if r.get('status') == 'built')} built, "
                  f"{sum(1 for r in res.get('rows', []) if str(r.get('status','')).startswith('skipped'))} skipped, "
                  f"{n_err} errors (of {len(manifest.get(dirkey, []))})", flush=True)

    if "barlink" in only:
        res = _run_child(["--barlink-child", ",".join(archs)])
        report["barlink"] = res
        if res.get("rc") or any(str(r.get("status", "")).startswith("error") for r in res.get("barlink", [])):
            rc = 1
        for r in res.get("barlink", []):
            print(f"[prebuild] {r['ext']}: {r['status'][:160]}", flush=True)

    if "cpu_ext" in only:
        res = _run_child(["--cpu-ext-child"])
        report["cpu_ext"] = res
        if res.get("rc") or any(str(r.get("status", "")).startswith("error") for r in res.get("cpu_ext", [])):
            rc = 1
        for r in res.get("cpu_ext", []):
            print(f"[prebuild] {r['ext']}: {r['status'][:160]}", flush=True)

    if "tms" in only:
        script = os.path.join(ns.tree, "scripts", "weg2", "tms", "build_tms_preload.sh")
        r = subprocess.run(["bash", script, "--venv", ns.venv], capture_output=True, text=True)
        report["tms"] = {"rc": r.returncode, "so": r.stdout.strip().splitlines()[-1:] or None,
                         "stderr_tail": r.stderr[-800:]}
        if r.returncode != 0:
            rc = 1
        print(f"[prebuild] tms preload rc={r.returncode} {r.stdout.strip()[-160:]}", flush=True)

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["stages"][-1]["finished"] = report["finished"]
    problems = report_problems(report)
    report["problems"] = problems
    report["verdict"] = "OK" if not problems else "INCOMPLETE"
    with open(ns.report, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[prebuild] report {ns.report}: {report['verdict']} ({len(problems)} offene Punkte)", flush=True)
    for pr in problems:
        print(f"[prebuild]   offen: {pr}", flush=True)
    # rc bewertet nur die Abschnitte DIESES Laufs; fehlende Abschnitte eines Teil-Laufs sind kein Fehler.
    if rc and not ns.strict:
        print("[prebuild] WARN: Vorbau UNVOLLSTAENDIG (Fehler oben) -- nicht fatal (ohne --strict); die fehlenden Module "
              "baut der Server beim ersten Boot per JIT. Beleg: JIT_PREBUILD.json verdict, `version`.", flush=True)
        return 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
