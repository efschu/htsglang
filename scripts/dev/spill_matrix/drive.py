#!/usr/bin/env python3
"""Request driver and signal reader for the spill-night matrix.

Stdlib only, so it runs under any interpreter on the rig without touching the
serving venv's package set.

The design point worth stating: a cell PASSES when a NAMED observable appears,
never when a boot merely fails to crash. So this tool separates two jobs and
reports them separately -- it DRIVES load, and it READS the server's own log
for literal signal strings. It never infers that something happened because
something else did.

Subcommands
    ready    <port> [timeout_s]         poll /health, bounded
    chat     <port> <prompt> [max_tok]  one greedy completion, prints the text
    load     <port> <n> <seconds>       n concurrent greedy streams, time-boxed
    probe    <port> <n> <seconds> <out.json>
                                        load + record every stream's output so
                                        two arms can be compared token-exactly
    signals  <log> <cell>               read the named signals for one cell
    compare  <a.json> <b.json>          exact per-prompt output comparison
"""

import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request

# The literal log strings that prove a mechanism fired. Every entry cites the
# source line it was read from, so a renamed message is a test failure and not
# a silent always-green.
SIGNALS = {
    # --- HOT / kvso -------------------------------------------------------
    "H1": [("armed", r"kv-session-offload \(S4\) armed: mode=")],          # :2435
    "H2": [("spill", r"kv-session-offload SPILL\(partial\): rid=")],       # :3747
    # H3/H4 signals CORRECTED after the K1 run. The first choices were wrong in
    # a way worth recording, because both would have reported a working feature
    # as broken:
    #   * "tick build: rid=" (:3895) is an ERROR path ("has no output token
    #     yet"), so a healthy tick never emits it;
    #   * "wave-back THRESHOLD armed" (:2454) only fires when
    #     --kv-session-offload-wave-back-min-free-tokens is NON-default, so at
    #     the default 0 its absence means nothing at all.
    # The real per-tick observable is the restore-gate trace at :4465, which
    # needs SGLANG_KVSO_TICK_TRACE=1 and reports the host tail draining
    # (boundary advancing towards L) once every 16 iterations.
    "H3": [("restoregate", r"kv-session-offload restore-gate: iter=\d+ L=\d+ boundary=\d+")],
    "H4": [("restore", r"restored to device")],                           # :5276
    "H5": [("spill", r"kv-session-offload SPILL\(partial\): rid=.*arrival_seq=")],
    "H6": [("draftkv", r"kv-session-offload: draft-KV bundle armed")],     # :2175
    "H7": [("restore", r"restored to device")],
    "H8": [("specintick", r"kv-session-offload spec-in-tick: reserved \d+ draft-read")],  # :2520
    "H9": [("specspill", r"kv-session-offload spec-in-tick: rid=.* spill batch armed with")],  # :3867
    "H11": [("budget", r"kv-session-offload SPILL BUDGET \(#236\) armed:"),        # :2409
            ("demote", r"kv-session-offload BUDGET: DEMOTING rid=")],              # :3054
    "H12": [("cadence", r"kv-session-offload: SELF-CALIBRATING spill-tick cadence armed")],  # :2464
    "H14": [("prefillspill", r"kv-session-offload prefill-spill \(born-spilled\) ENABLED")],  # :2201
    # Refusals we EXPECT to see, i.e. a PASS is the refusal appearing.
    "H15": [("refusal", r"--enable-kv-session-offload is its own host tier")],
}

HDR = {"Content-Type": "application/json"}


def _post(port, path, payload, timeout=120):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers=HDR,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def cmd_ready(port, timeout=600):
    """Poll the real condition (a 200 from /health), never a fixed sleep."""
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as r:
                if r.status == 200:
                    print(f"ready after {time.time() - deadline + float(timeout):.0f}s")
                    return 0
        except Exception:
            pass
        time.sleep(2)
    print(f"NOT ready within {timeout}s", file=sys.stderr)
    return 1


def _greedy(port, prompt, max_tokens):
    """One deterministic completion. temperature=0 so two arms are comparable."""
    out = _post(
        port,
        "/v1/chat/completions",
        {
            "model": "Qwen3.6-27B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return out["choices"][0]["message"]["content"]


def cmd_chat(port, prompt, max_tokens=64):
    print(_greedy(port, prompt, int(max_tokens)))
    return 0


def _prompts(n):
    """Distinct, long prompts.

    Distinct so sessions do not share a radix prefix -- a shared prefix would be
    served from the radix cache and hide the very pressure we are building.

    LENGTH MATTERS AND IS THE POINT. K1 measured this the hard way: 4 streams of
    ~570 KV tokens against an 8192-token pool produced no pressure and therefore
    no spill, and a missing spill then looks exactly like a broken spill. Each
    prompt is padded to roughly SPILL_PROMPT_WORDS words of DISTINCT filler
    (distinct per stream, so the pad cannot be prefix-shared either), so that
    streams x (prompt + generation) genuinely exceeds --max-total-tokens.
    """
    import os as _os

    pad_words = int(_os.environ.get("SPILL_PROMPT_WORDS", "900"))
    base = (
        "Write a detailed technical explanation, at least 400 words, about "
        "topic number {i}: {t}. Be specific and structured."
    )
    topics = [
        "cache coherence protocols in multi-socket systems",
        "the design of log-structured merge trees",
        "error correction in DDR5 memory subsystems",
        "scheduling policies in modern GPU drivers",
        "the mathematics of rotary positional embeddings",
        "network congestion control algorithms",
        "filesystem journaling and crash consistency",
        "branch prediction in out-of-order cores",
    ]
    out = []
    for i in range(n):
        # Per-stream distinct filler: the stream index is woven into every
        # sentence so no two prompts share a usable radix prefix.
        pad = " ".join(
            f"context-{i}-token-{j} notes on subsystem {j % 17} of stream {i};"
            for j in range(pad_words)
        )
        out.append(base.format(i=i, t=topics[i % len(topics)])
                   + "\n\nBackground material to take into account:\n" + pad)
    return out


def cmd_load(port, n, seconds, out=None):
    """n concurrent greedy streams, hard time box."""
    n, seconds = int(n), float(seconds)
    prompts = _prompts(n)
    results, t0 = {}, time.time()

    import os as _os
    max_tok = int(_os.environ.get("SPILL_MAX_TOKENS", "512"))
    # SPILL_MIXED=1 makes stream 0 far longer than the rest. This is what a
    # RESTORE needs: the short streams finish, KV pressure falls, and a still
    # living spilled session can wave back. With uniform lengths every session
    # simply host-ticks to completion and no restore is ever required -- which
    # is correct behaviour, but leaves the restore path unobserved.
    mixed = _os.environ.get("SPILL_MIXED", "0") == "1"
    long_mult = int(_os.environ.get("SPILL_LONG_MULT", "8"))

    def _budget(i):
        return max_tok * long_mult if (mixed and i == 0) else max_tok

    def one(i):
        try:
            return i, _greedy(port, prompts[i], _budget(i)), None
        except Exception as e:                      # a failed stream is data
            return i, None, repr(e)

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one, i) for i in range(n)]
        for f in cf.as_completed(futs, timeout=seconds + 120):
            i, text, err = f.result()
            results[i] = {"prompt": prompts[i], "text": text, "error": err}

    dt = time.time() - t0
    ok = sum(1 for v in results.values() if v["text"] is not None)
    print(f"streams={n} ok={ok} failed={n - ok} wall={dt:.1f}s")
    if out:
        with open(out, "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
        print(f"wrote {out}")
    return 0 if ok == n else 1


def cmd_signals(log, cell):
    """Read the named signals for one cell out of the server's own log.

    Prints a count per signal. Absence is reported as absence -- this function
    never upgrades a missing signal into a pass.
    """
    pats = SIGNALS.get(cell)
    if not pats:
        print(f"cell {cell}: no signal defined -- NOT-EXAMINED")
        return 3
    try:
        with open(log, "r", errors="replace") as fh:
            body = fh.read()
    except OSError as e:
        print(f"cell {cell}: log unreadable ({e}) -- NOT-EXAMINED")
        return 3
    allhit = True
    for name, pat in pats:
        hits = len(re.findall(pat, body))
        print(f"cell {cell}: {name:12s} hits={hits:<5d} /{pat}/")
        if hits == 0:
            allhit = False
    print(f"cell {cell}: {'ALL SIGNALS PRESENT' if allhit else 'SIGNAL MISSING'}")
    return 0 if allhit else 1


def cmd_compare(a, b):
    """Exact per-prompt comparison of two arms.

    Used for the host-vs-device probe: arm A has a KV pool large enough that no
    spill occurs, arm B has the same concurrency but a pool small enough that it
    does. Same batch composition, one difference. A divergence here is about the
    spill and not about batching.
    """
    A, B = (json.load(open(p)) for p in (a, b))
    keys = sorted(set(A) | set(B))
    same = diff = missing = 0
    for k in keys:
        ta = (A.get(k) or {}).get("text")
        tb = (B.get(k) or {}).get("text")
        if ta is None or tb is None:
            missing += 1
            print(f"stream {k}: MISSING in one arm")
        elif ta == tb:
            same += 1
        else:
            diff += 1
            # Show where they part, not the whole texts.
            i = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]),
                     min(len(ta), len(tb)))
            print(f"stream {k}: DIVERGES at char {i}")
            print(f"    A: ...{ta[max(0, i - 40):i + 40]!r}")
            print(f"    B: ...{tb[max(0, i - 40):i + 40]!r}")
    print(f"identical={same} diverged={diff} missing={missing}")
    return 0 if diff == 0 and missing == 0 else 1


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    c, rest = argv[1], argv[2:]
    table = {
        "ready": cmd_ready,
        "chat": cmd_chat,
        "load": cmd_load,
        "probe": lambda port, n, s, out: cmd_load(port, n, s, out),
        "signals": cmd_signals,
        "compare": cmd_compare,
    }
    if c not in table:
        print(__doc__)
        return 2
    return table[c](*rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
