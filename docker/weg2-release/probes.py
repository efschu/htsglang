#!/usr/bin/env python3
"""Abnahme-Proben gegen die weg2-Front eines Containers -- ENTWURF (27B-Sitz R, 24.09.2026).

NICHT GELAUFEN. Nur Standardbibliothek (laeuft auf dem Proxmox-Host mit python3 3.12).
Messformen wie die Rig-Arme, damit Container und nativer Boot vergleichbar sind:

  gen      kurze Frage, erwartet ein Wort ("Paris"); vor und zwischen Flips
  needle   Fuelltext mit Nadel "Der Geheimcode lautet 4711-QX." an Position P (0..1), Frage nach
           dem Code -> MATCH / IN-REASONING-ONLY / MISS (wie arm_xsn435.sh:889)
  decode   gestreamt, ttft_s = Senden -> erstes Text-/Reasoning-Delta; decode_s = erstes -> letztes
           Delta; tok_s = (completion_tokens - 1) / decode_s (wie nf_decode_probe.py); Kontext
           @depth: Python-Quelltext aus einer EINGEFRORENEN Datei (--code-file, z. B. die
           launcher.py des Image-Baums, NF OP-14) bzw. englische Prosa; thinking mit enable_thinking
  flip     POST /weg2/flip (ein POST = beide Richtungen, front.py), dann /weg2/state bis serving;
           Dauer und Epoche
  state    /weg2/state

Ausgabe: je Probe eine JSON-Zeile (fuer das Abnahme-Protokoll).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

FILLER = ("The river trade of the northern towns grew for three centuries, and "
          "every harbour kept its own ledgers of grain, timber and wool. ")
NEEDLE = "Der Geheimcode lautet 4711-QX. "


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float = 20) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base: str, model: str, content: str, max_tokens: int, thinking: bool = False) -> dict:
    return _post(f"{base}/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }, timeout=1800)


def p_gen(a) -> dict:
    t0 = time.time()
    r = chat(a.base, a.model, "Was ist die Hauptstadt von Frankreich? Antworte mit einem Wort.", 32)
    c = (r["choices"][0]["message"].get("content") or "")
    return {"probe": "gen", "ok": "paris" in c.lower(), "answer": c[:80], "wall_s": round(time.time() - t0, 2)}


def p_needle(a) -> dict:
    n = a.sentences
    k = max(0, min(n - 1, int(n * a.position)))
    text = "".join(NEEDLE if i == k else FILLER for i in range(n))
    t0 = time.time()
    r = chat(a.base, a.model, text + "\n\nWie lautet der Geheimcode? Antworte nur mit dem Code.", 64)
    msg = r["choices"][0]["message"]
    c, rs = msg.get("content") or "", msg.get("reasoning_content") or ""
    verdict = "MATCH" if "4711-QX" in c else ("IN-REASONING-ONLY" if "4711-QX" in rs else "MISS")
    u = r.get("usage") or {}
    return {"probe": "needle", "verdict": verdict, "sentences": n, "position": a.position,
            "prompt_tokens": u.get("prompt_tokens"), "wall_s": round(time.time() - t0, 2)}


def _context(kind: str, depth: int, code_file: str) -> str:
    chars = 4 * max(0, depth)
    if kind == "code":
        src = open(code_file, encoding="utf-8", errors="replace").read()
        while len(src) < chars:
            src += src
        return src[:chars]
    return (FILLER * (chars // len(FILLER) + 1))[:chars]


def p_decode(a) -> dict:
    ctx = _context(a.kind, a.depth, a.code_file)
    ask = ("Erklaere den folgenden Code ausfuehrlich." if a.kind == "code"
           else "Setze diesen Text ausfuehrlich fort.")
    body = {"model": a.model, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": ctx + "\n\n" + ask}],
            "max_tokens": a.max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": a.kind == "thinking"}}
    req = urllib.request.Request(f"{a.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now
    ct = (usage or {}).get("completion_tokens")
    dec = (last - first) if (first and last and last > first) else None
    return {"probe": "decode", "kind": a.kind, "depth": a.depth,
            "ttft_s": round(first - t0, 3) if first else None,
            "decode_s": round(dec, 3) if dec else None, "completion_tokens": ct,
            "tok_s": round((ct - 1) / dec, 2) if (ct and dec) else None}


def p_flip(a) -> dict:
    s0 = _get(f"{a.base}/weg2/state")
    t0 = time.time()
    _post(f"{a.base}/weg2/flip", {}, timeout=600)
    seen, st = False, {}
    for _ in range(1200):
        st = _get(f"{a.base}/weg2/state")
        if st.get("state") != "serving":
            seen = True
        elif seen or st.get("epoch") != s0.get("epoch"):
            break
        time.sleep(0.5)
    return {"probe": "flip", "epoch_before": s0.get("epoch"), "epoch_after": st.get("epoch"),
            "state": st.get("state"), "wall_s": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["gen", "needle", "decode", "flip", "state"])
    ap.add_argument("--base", default="http://127.0.0.1:31030")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--sentences", type=int, default=5500)     # ~100k Token bei ~18 Tok/Satz
    ap.add_argument("--position", type=float, default=0.10)
    ap.add_argument("--kind", default="code", choices=["code", "prosa", "thinking"])
    ap.add_argument("--depth", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--code-file", default="")
    a = ap.parse_args()
    if a.probe == "state":
        print(json.dumps(_get(f"{a.base}/weg2/state")))
        return 0
    out = {"gen": p_gen, "needle": p_needle, "decode": p_decode, "flip": p_flip}[a.probe](a)
    out["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
