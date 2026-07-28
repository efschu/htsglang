#!/usr/bin/env python3
"""Prefill satellite (#212): drive one cold prefill onto a second machine.

The satellite is a second, weaker box that owns the *prefill* of a request
while the main rig keeps decoding. This script is the wire between them: it
performs the compatibility preflight the servers do not perform themselves,
routes one request through the pair, and measures the two things that decide
whether the detour paid -- time to first token, and what the main rig's
running decodes lost while it happened.

What actually carries the state
-------------------------------
For a hybrid GDN model (Qwen3.5/3.6: 8 full-attention layers, 24
gated-delta-net layers) there is exactly one implemented cross-machine path,
and it is PD disaggregation:

    request --> prefill arm (satellite)          computes KV + GDN state
            --> mooncake/nixl transfer           KV rows *and* the mamba slot
            --> decode arm (main rig)            resumes without recompute

The HiCache L3 store is NOT an alternative here, and the reason is worth
stating because it is the first thing one reaches for. A store round trip
carries KV pages; the GDN recurrent state is a separate pool. On a prefix
match ``MambaRadixCache._match_post_processor`` truncates the match to the
deepest node that owns a mamba checkpoint (``value = value[:best_value_len]``,
mem_cache/mamba_radix_cache.py). A KV-only import therefore matches zero
tokens and the decode side recomputes the whole prompt -- the store route
looks like it works right up to the point where it silently does nothing.
For a dense (non-hybrid) model the store route is viable; for this model
family it is not.

What the servers do NOT check, and this script does
---------------------------------------------------
The PD bootstrap handshake compares ``page_size`` and ``kv_cache_dtype``
(disaggregation/common/conn.py) and nothing else. Two arms holding different
weights connect happily and produce fluent nonsense, because the decode arm
never sees the prefill arm's model identity. ``--preflight`` compares model
path, served name, dtype, quantization, context length, TP geometry and
attention/KV geometry across the pair and refuses to run on a mismatch.

Usage
-----
    # 1. compatibility gate only
    python scripts/satellite/prefill_offload.py preflight \
        --prefill http://<satellite>:31212 --decode http://127.0.0.1:31213

    # 2. one request through the pair (b), or straight at one server (a)
    python scripts/satellite/prefill_offload.py probe \
        --prefill http://<satellite>:31212 --decode http://127.0.0.1:31213 \
        --bootstrap-port 8998 --prompt-tokens 8192

    python scripts/satellite/prefill_offload.py probe \
        --local http://127.0.0.1:31213 --prompt-tokens 8192

    # 3. the measurement: cold prefill while N decodes are running
    python scripts/satellite/prefill_offload.py measure \
        --prefill http://<satellite>:31212 --decode http://127.0.0.1:31213 \
        --bootstrap-port 8998 --load 3 --prompt-tokens 8192 --window 30

``measure`` is the honest form of the comparison: the same load generator,
the same prompt, once against a monolithic server and once against the pair.
Report both or neither -- a satellite TTFT without the load it was supposed
to dodge says nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Fields that must agree across the two arms. These are the ones the bootstrap
# handshake never looks at; a mismatch here is a wrong-weights run that
# produces output instead of an error.
#
# ``model_path`` is deliberately NOT in this list and is compared separately by
# checkpoint name (_ckpt_name). Two boxes legitimately mount the same weights
# at different paths, and a literal string compare would reject every real
# cross-machine pair. The name comparison keeps the check that matters -- two
# different checkpoints -- without inventing a filesystem requirement the
# transport does not have. (The HiCache store, by contrast, DOES compare the
# normalized path literally: hicache_storage.py compute_model_identity_hash.
# That is why a store-based handover needs a shared mount and this one does
# not.)
IDENTITY_FIELDS = (
    "served_model_name",
    "dtype",
    "quantization",
)
# Fields the servers do assert on, repeated here so the failure names itself
# before a 30 s bootstrap timeout does.
TRANSPORT_FIELDS = (
    "page_size",
    "kv_cache_dtype",
)
# Geometry that does not have to match but changes what the numbers mean.
INFORMATIVE_FIELDS = (
    "tp_size",
    "dp_size",
    "context_length",
    "max_running_requests",
    "disaggregation_mode",
    "disaggregation_transfer_backend",
    "speculative_algorithm",
    "attention_backend",
)


def _http(url: str, payload: Optional[dict] = None, timeout: float = 60.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def server_info(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    """``/get_server_info`` flattened to the fields this script compares."""
    info = _http(f"{url.rstrip('/')}/get_server_info", timeout=timeout)
    if not isinstance(info, dict):
        raise RuntimeError(f"{url}: /get_server_info returned {type(info).__name__}")
    # The endpoint has carried the ServerArgs under a nested key in some
    # versions and flat in others; accept both rather than pin one.
    flat = dict(info)
    for nest in ("server_args", "internal_states"):
        sub = info.get(nest)
        if isinstance(sub, dict):
            for k, v in sub.items():
                flat.setdefault(k, v)
        elif isinstance(sub, list) and sub and isinstance(sub[0], dict):
            for k, v in sub[0].items():
                flat.setdefault(k, v)
    return flat


def _ckpt_name(path: str) -> str:
    """Checkpoint identity from a path: last component, case- and
    separator-insensitive. ``/opt/models/foo-4b`` and
    ``/data/cache/Foo-4B`` are the same checkpoint at two mounts."""
    if not path:
        return ""
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    return "".join(c for c in tail.lower() if c.isalnum())


def preflight(prefill_url: str, decode_url: str) -> Tuple[bool, List[str]]:
    """Compare the two arms. Returns (ok, lines) -- lines are always printed."""
    lines: List[str] = []
    ok = True
    try:
        p = server_info(prefill_url)
        d = server_info(decode_url)
    except Exception as e:  # noqa: BLE001 - the message is the result
        return False, [f"FAIL  could not read /get_server_info: {e}"]

    def cmp(field: str, hard: bool, note: str = "") -> None:
        nonlocal ok
        pv, dv = p.get(field, "<absent>"), d.get(field, "<absent>")
        same = pv == dv
        if same:
            lines.append(f"ok    {field}: {pv}")
            return
        if hard:
            ok = False
            lines.append(f"FAIL  {field}: prefill={pv!r} decode={dv!r} {note}")
        else:
            lines.append(f"note  {field}: prefill={pv!r} decode={dv!r} {note}")

    lines.append("-- identity (NOT checked by the PD handshake) --")
    pp, dp = p.get("model_path", ""), d.get("model_path", "")
    if _ckpt_name(pp) == _ckpt_name(dp):
        if pp == dp:
            lines.append(f"ok    model_path: {pp}")
        else:
            lines.append(
                f"ok    model_path: same checkpoint name at different mounts "
                f"(prefill={pp!r} decode={dp!r})"
            )
    else:
        ok = False
        lines.append(
            f"FAIL  model_path: prefill={pp!r} decode={dp!r} -- different "
            "checkpoints would decode as fluent nonsense, and nothing in the "
            "bootstrap handshake would say so"
        )
    for f in IDENTITY_FIELDS:
        cmp(f, hard=True, note="different weights would decode as fluent nonsense")
    lines.append("-- transport (asserted by the bootstrap handshake) --")
    for f in TRANSPORT_FIELDS:
        cmp(f, hard=True, note="the arms will refuse to pair")
    lines.append("-- informative --")
    for f in INFORMATIVE_FIELDS:
        cmp(f, hard=False)

    if p.get("disaggregation_mode") != "prefill":
        ok = False
        lines.append(
            f"FAIL  prefill arm is in disaggregation_mode="
            f"{p.get('disaggregation_mode')!r}, expected 'prefill'"
        )
    if d.get("disaggregation_mode") != "decode":
        ok = False
        lines.append(
            f"FAIL  decode arm is in disaggregation_mode="
            f"{d.get('disaggregation_mode')!r}, expected 'decode'"
        )
    if p.get("speculative_algorithm") or d.get("speculative_algorithm"):
        lines.append(
            "note  speculative decoding is force-disabled in both PD arms on "
            "this fork; a monolithic baseline must run without it too, or the "
            "comparison measures the draft model instead of the satellite"
        )
    return ok, lines


_SENTENCES = [
    "The survey crew reached the lower gallery shortly after the tide turned, and the readings they took there disagreed with every earlier pass.",
    "A trestle of that span settles unevenly when the ground beneath one footing drains faster than the ground beneath the other.",
    "The foreman kept two notebooks: one for the numbers the office expected, and one for the numbers the site actually produced.",
    "Nothing in the original drawings accounted for the seam of brackish water that runs under the eastern quarry.",
    "By the third season the mortise joints on the upper deck had opened wide enough to admit a knife blade.",
    "The inspector argued that the settlement was within tolerance; the crew argued that tolerance had been written by people who had never stood on the deck.",
    "Iron cools at a rate the kiln logs never recorded, which is why the early castings cracked and the later ones did not.",
    "Every measurement taken before the scaffold came down has to be treated as provisional, and most of them were not.",
    "The cistern was built to hold four months of water and has never once been filled past the second course of stone.",
    "A drawing that disagrees with the building is not evidence about the building; it is evidence about the drawing.",
    "The windlass on the north jetty was replaced twice, each time with a heavier unit, and each time the mounting plate failed first.",
    "Granite from the upper bed weathers differently than granite from the lower bed, though the two are indistinguishable when freshly cut.",
    "They stopped recording the wind readings when it became clear that the anemometer had been mounted in the lee of the gantry.",
    "The revised schedule assumed the escarpment could be cut in a single season, which no one who had worked that face believed.",
    "What finally settled the argument was not a calculation but a winter, and the winter was not kind to either position.",
    "The lintel above the main opening carries a load the specification assigns to a beam that was never installed.",
    "Half the crew read the plinth survey as proof of subsidence and half read it as proof of a bad benchmark.",
    "A ratchet that slips once under load will slip again, and the second time it will be carrying more than the first.",
    "The vellum copies held up better than the paper ones, which is the only reason the earliest figures survive at all.",
    "Nobody disputes the totals; the dispute is entirely about which column they belong in.",
]


def make_prompt(n_tokens: int, seed: int) -> str:
    """A prompt of roughly ``n_tokens`` tokens that no cache has seen.

    Two requirements pull against each other here.

    It must be COLD: a prompt repeated across runs is compressible by the
    radix cache and would make a cold prefill look warm. So the sentence
    order is drawn fresh from the seed, and the text opens with a nonce.

    It must also be CONTINUABLE: random word soup is uncached but degenerate,
    and a greedy continuation of it is a wall of one repeated token. That
    output cannot be judged, so a measurement built on it cannot be judged
    either. Coherent sentences in a shuffled order satisfy both -- unique as a
    token sequence, ordinary as language.
    """
    rng = random.Random(seed)
    head = f"Field notes, survey reference {rng.randrange(10**9):09d}.\n\n"
    out = [head]
    approx = len(head) // 4
    while approx < n_tokens:
        s = rng.choice(_SENTENCES)
        out.append(s + " ")
        approx += len(s) // 4 + 1
    out.append(
        "\n\nSummarize the disagreement described in these notes in three sentences.\n\n"
    )
    return "".join(out)


class _Result(dict):
    pass


def one_request(
    url: str,
    prompt: str,
    max_new_tokens: int,
    bootstrap: Optional[dict] = None,
    timeout: float = 600.0,
) -> _Result:
    """One non-streaming /generate. Returns timings + the meta_info fields."""
    payload: Dict[str, Any] = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
    }
    if bootstrap:
        payload.update(bootstrap)
    t0 = time.perf_counter()
    out = _http(f"{url.rstrip('/')}/generate", payload, timeout=timeout)
    e2e = time.perf_counter() - t0
    if isinstance(out, list):
        out = out[0]
    meta = (out or {}).get("meta_info") or {}
    return _Result(
        e2e_s=e2e,
        text=(out or {}).get("text", ""),
        prompt_tokens=meta.get("prompt_tokens"),
        completion_tokens=meta.get("completion_tokens"),
        cached_tokens=meta.get("cached_tokens"),
        cached_tokens_details=meta.get("cached_tokens_details"),
        e2e_latency=meta.get("e2e_latency"),
        finish_reason=(meta.get("finish_reason") or {}).get("type"),
    )


def ttft_request(
    url: str,
    prompt: str,
    max_new_tokens: int,
    bootstrap: Optional[dict] = None,
    timeout: float = 600.0,
) -> _Result:
    """Streaming /generate, so the first token's arrival is observed directly.

    TTFT is the whole point of the satellite; deriving it from a non-streaming
    total would fold the decode of every subsequent token into it.
    """
    payload: Dict[str, Any] = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        "stream": True,
    }
    if bootstrap:
        payload.update(bootstrap)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    last = {}
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if ttft is None and obj.get("text"):
                ttft = time.perf_counter() - t0
            text = obj.get("text", text)
            last = obj
    e2e = time.perf_counter() - t0
    meta = (last or {}).get("meta_info") or {}
    return _Result(
        ttft_s=ttft,
        e2e_s=e2e,
        text=text,
        prompt_tokens=meta.get("prompt_tokens"),
        completion_tokens=meta.get("completion_tokens"),
        cached_tokens=meta.get("cached_tokens"),
        cached_tokens_details=meta.get("cached_tokens_details"),
        finish_reason=(meta.get("finish_reason") or {}).get("type"),
    )


class LoadGenerator:
    """N concurrent natural decodes against a target, for the whole window.

    Each stream reports its own per-token time, because the number that says
    whether the main rig was disturbed is the running decodes' inter-token
    latency, not an aggregate throughput.
    """

    def __init__(
        self,
        url: str,
        n: int,
        prompt_tokens: int,
        max_new_tokens: int,
        bootstrap_factory=None,
        seed: int = 0,
        prefill_url: Optional[str] = None,
    ):
        self.url = url
        self.n = n
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.bootstrap_factory = bootstrap_factory
        self.seed = seed
        # In PD mode every request needs BOTH copies. Sending only the decode
        # copy leaves it waiting for a prefill that is never issued: the
        # request sits in the prealloc queue and the load generator silently
        # produces nothing at all rather than producing load.
        self.prefill_url = prefill_url
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.samples: List[dict] = []
        self._lock = threading.Lock()

    def _worker(self, idx: int) -> None:
        round_no = 0
        while not self._stop.is_set():
            prompt = make_prompt(
                self.prompt_tokens, seed=self.seed + idx * 1000 + round_no
            )
            bs = self.bootstrap_factory() if self.bootstrap_factory else None
            try:
                if self.prefill_url and bs:
                    r = _pair_request(
                        self.prefill_url, self.url, bs, prompt, self.max_new_tokens
                    )
                else:
                    r = ttft_request(
                        self.url, prompt, self.max_new_tokens, bs, timeout=300
                    )
            except Exception as e:  # noqa: BLE001 - a failed stream is a datum
                with self._lock:
                    self.samples.append({"worker": idx, "error": repr(e)[:200]})
                time.sleep(1.0)
                round_no += 1
                continue
            n_out = r.get("completion_tokens") or 0
            decode_s = (r["e2e_s"] - (r["ttft_s"] or 0.0)) if n_out > 1 else None
            with self._lock:
                self.samples.append(
                    {
                        "worker": idx,
                        "ttft_s": r["ttft_s"],
                        "e2e_s": r["e2e_s"],
                        "completion_tokens": n_out,
                        "ms_per_token": (decode_s * 1000.0 / (n_out - 1))
                        if decode_s and n_out > 1
                        else None,
                    }
                )
            round_no += 1

    def start(self) -> None:
        for i in range(self.n):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=30)

    def summary(self) -> dict:
        with self._lock:
            good = [s for s in self.samples if s.get("ms_per_token")]
            errs = [s for s in self.samples if "error" in s]
        vals = sorted(s["ms_per_token"] for s in good)
        return {
            "streams": self.n,
            "completed": len(good),
            "errors": len(errs),
            "first_error": errs[0]["error"] if errs else None,
            "ms_per_token_median": round(vals[len(vals) // 2], 2) if vals else None,
            "ms_per_token_min": round(vals[0], 2) if vals else None,
            "ms_per_token_max": round(vals[-1], 2) if vals else None,
        }


def _mint_bootstrap(host: str, port: int):
    def factory() -> dict:
        return {
            "bootstrap_host": host,
            "bootstrap_port": port,
            "bootstrap_room": random.randint(0, 2**63 - 1),
        }

    return factory


def _pair_request(
    prefill_url: str,
    decode_url: str,
    bootstrap: dict,
    prompt: str,
    max_new_tokens: int,
) -> _Result:
    """One request through the pair: both arms, same room, decode streams.

    The prefill copy is fired first and left to run; the fork's prefill arm
    caps itself at one output token regardless of what is asked, so the reply
    is only interesting for its error status.
    """
    err: List[str] = []

    def fire_prefill() -> None:
        try:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                **bootstrap,
            }
            _http(f"{prefill_url.rstrip('/')}/generate", payload, timeout=600)
        except Exception as e:  # noqa: BLE001
            err.append(repr(e)[:300])

    th = threading.Thread(target=fire_prefill, daemon=True)
    th.start()
    r = ttft_request(decode_url, prompt, max_new_tokens, bootstrap, timeout=600)
    th.join(timeout=60)
    if err:
        r["prefill_error"] = err[0]
    return r


def cmd_preflight(args) -> int:
    ok, lines = preflight(args.prefill, args.decode)
    for line in lines:
        print(line)
    print("PREFLIGHT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_probe(args) -> int:
    prompt = make_prompt(args.prompt_tokens, seed=args.seed)
    if args.local:
        r = ttft_request(args.local, prompt, args.max_new_tokens)
        label = "local"
    else:
        bs = _mint_bootstrap(
            args.bootstrap_host or _host_of(args.prefill), args.bootstrap_port
        )()
        r = _pair_request(args.prefill, args.decode, bs, prompt, args.max_new_tokens)
        label = "satellite"
    out = {"path": label, **{k: v for k, v in r.items() if k != "text"}}
    print(json.dumps(out, indent=2))
    print("--- first 300 chars of output ---")
    print(r["text"][:300])
    return 0


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).hostname or "127.0.0.1"


def cmd_measure(args) -> int:
    """Cold prefill under sustained decode load, once per path.

    Sequence per path: start the load, let it reach steady state, fire ONE
    cold prefill, keep the load running to the end of the window, stop.
    """
    results: Dict[str, Any] = {"prompt_tokens_target": args.prompt_tokens}

    def run_path(
        label: str, target_url: str, bootstrap_factory, cold_fn, prefill_url=None
    ) -> dict:
        load = LoadGenerator(
            target_url,
            args.load,
            args.load_prompt_tokens,
            args.load_max_new_tokens,
            bootstrap_factory,
            seed=args.seed + 7777,
            prefill_url=prefill_url,
        )
        load.start()
        time.sleep(args.warmup)
        before = load.summary()
        cold = cold_fn()
        time.sleep(args.window)
        after = load.summary()
        load.stop()
        return {
            "cold": {k: v for k, v in cold.items() if k != "text"},
            "cold_text_head": cold["text"][:300],
            "load_before_cold": before,
            "load_full_window": after,
        }

    if args.local:
        prompt = make_prompt(args.prompt_tokens, seed=args.seed)
        results["a_local"] = run_path(
            "local",
            args.local,
            None,
            lambda: ttft_request(args.local, prompt, args.max_new_tokens),
        )
    if args.prefill and args.decode:
        prompt = make_prompt(args.prompt_tokens, seed=args.seed)
        host = args.bootstrap_host or _host_of(args.prefill)
        factory = _mint_bootstrap(host, args.bootstrap_port)
        results["b_satellite"] = run_path(
            "satellite",
            args.decode,
            factory,
            lambda: _pair_request(
                args.prefill, args.decode, factory(), prompt, args.max_new_tokens
            ),
            prefill_url=args.prefill,
        )
    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument(
            "--prefill", help="satellite (disaggregation-mode prefill) base URL"
        )
        p.add_argument(
            "--decode", help="main rig (disaggregation-mode decode) base URL"
        )
        p.add_argument("--local", help="monolithic server base URL, for path (a)")
        p.add_argument("--bootstrap-port", type=int, default=8998)
        p.add_argument(
            "--bootstrap-host",
            default=None,
            help="defaults to the host part of --prefill",
        )
        p.add_argument("--seed", type=int, default=1212)

    p = sub.add_parser("preflight", help="compare the two arms and refuse mismatches")
    p.add_argument("--prefill", required=True)
    p.add_argument("--decode", required=True)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("probe", help="one request, one path")
    common(p)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("measure", help="cold prefill under load, per path")
    common(p)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--load", type=int, default=3, help="concurrent decode streams")
    p.add_argument("--load-prompt-tokens", type=int, default=256)
    p.add_argument("--load-max-new-tokens", type=int, default=256)
    p.add_argument(
        "--warmup", type=float, default=8.0, help="s before the cold prefill"
    )
    p.add_argument(
        "--window", type=float, default=20.0, help="s after the cold prefill"
    )
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_measure)

    args = ap.parse_args(argv)
    if args.cmd in ("probe", "measure"):
        if not args.local and not (args.prefill and args.decode):
            ap.error("give --local, or both --prefill and --decode")
    try:
        return args.func(args)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:400]!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
