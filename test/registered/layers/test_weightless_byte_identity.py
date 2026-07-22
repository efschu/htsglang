"""Server-level byte-identity regression for the weightless-KV fast lane
(Variant C, Option-B B1) -- the #12 determinism-CI seed.

Crystallizes the two logit comparisons that classified B1 GREEN into a
reusable guard. It boots the REAL weightless server (TP=DCP=3, head on the
5090, weightless workers on the two 3080s) and a plain TP=1-solo baseline on
the same GGUF, then asserts two invariants:

  (1) EXTEND path is BIT-IDENTICAL. Feed prompt + a fixed teacher-forced
      prefix as ONE prefill and request the next-token top-k logprobs. The
      weightless head's DCP-sharded attention + LSE-merge must reproduce the
      TP=1-solo logits to 0.0 EXACTLY (proves weightless geometry / owner-write
      / merge are correct, not merely close).

  (2) DECODE trajectory diverges ONLY as benign decode-kernel fp-order. Run
      free greedy generation on both, walk the two per-step top-k logprob
      streams in lockstep while their argmax matches, and require:
        - step-0 (the first token, computed from the prompt PREFILL = extend
          path) Δ == 0 exactly;
        - every pre-divergence step's Δ on the argmax-contender token stays
          within the model's intrinsic decode fp-order band (DECODE_FP_BAND);
        - the FIRST argmax flip, if any, lands on a near-degenerate baseline
          distribution (top-2 margin <= FLIP_TIE_EPS) -- a genuine coin-flip
          tipped by fp-order, NOT a confident disagreement.

  A REAL geometry bug would break (1) (extend Δ != 0) or make (2)'s
  pre-divergence Δ grow large / flip at a confidently-separated token.

Measured on Qwen3.6-27B-Q3_K_M (2026-07-19): extend Δ = 0.0 exact; decode
pre-divergence contender Δ ~2e-3..6e-2; first flip at step 4 where baseline is
a perfect 0.000-margin tie (198 vs 271, both ln 0.5). The model's OWN
decode-vs-extend Δ (baseline alone) is 1.7e-2..3.3e-1, i.e. the observed
weightless decode Δ is the same order as intrinsic decode-kernel fp-order.

REQUIRES 3 visible GPUs (one >=28 GB for the head + two >=19 GB) AND the GGUF.
Run only inside an explicitly granted GPU window:
    .venv/bin/python -m pytest -s \
        test/registered/layers/test_weightless_byte_identity.py
  or directly:
    .venv/bin/python test/registered/layers/test_weightless_byte_identity.py

Env knobs: WL_MODEL_PATH (REQUIRED -- the test skips without it),
WL_TOKENIZER_PATH (default: the model's directory), HTSGLANG_TEST_VENV
(python used to boot the servers; default: this interpreter),
HTSGLANG_TEST_REPO_PY (PYTHONPATH for the servers; default: the ``python/``
dir of the checkout containing this test).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

# --------------------------------------------------------------------------
# Config (overridable by env)
# --------------------------------------------------------------------------
REPO_PY = os.environ.get(
    "HTSGLANG_TEST_REPO_PY",
    os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "python")),
)
VENV_PY = os.environ.get("HTSGLANG_TEST_VENV", sys.executable)
MODEL_PATH = os.environ.get("WL_MODEL_PATH", "")
TOKENIZER_PATH = os.environ.get(
    "WL_TOKENIZER_PATH", os.path.dirname(MODEL_PATH) if MODEL_PATH else "")
WL_PORT = int(os.environ.get("WL_PORT", "31800"))
BASE_PORT = int(os.environ.get("WL_BASE_PORT", "31801"))

PROMPT = "The capital of France is"
# Fixed teacher-forced prefix (" Paris.\n\n<think>\n") appended for the extend
# test -- token ids for the Qwen3.6 tokenizer.
PREFIX_GEN_IDS = [11751, 13, 271, 248068, 198]
N_DECODE = 48
TOPK = 20
BOOT_TIMEOUT_S = 240

# Tolerances.
EXTEND_EXACT_EPS = 1e-9      # extend path must be bit-identical
DECODE_FP_BAND = 0.5         # per-step contender Δ stays within intrinsic band
FLIP_TIE_EPS = 0.25          # baseline top-2 margin at the flip = near-tie


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------
def _http_json(port, path, payload=None, timeout=180):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    hdr = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _health(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def _boot(args, env, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [VENV_PY, "-m", "sglang.launch_server", *args],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,  # own process group -> clean own-PID teardown
    )
    return proc, log


def _wait_ready(proc, port, log_path):
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early (rc={proc.returncode}); see {log_path}"
            )
        if _health(port):
            return
        time.sleep(2)
    raise RuntimeError(f"server not ready within {BOOT_TIMEOUT_S}s; see {log_path}")


def _teardown(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def _weightless_args(chunked_size="-1", radix=True, graph=False, port=WL_PORT):
    args = [
        "--model-path", MODEL_PATH,
        "--tokenizer-path", TOKENIZER_PATH,
        "--tp-size", "3", "--dcp-size", "3",
        "--weightless-kv-fastlane", "--weightless-kv-head-rank", "0",
        "--rank-gpu-id", "0,1,2",
        "--rank-gpu-memory-mib", "28000,19000,19000",
        "--context-length", "4096",
        "--attention-backend", "flashinfer",
        "--dtype", "bfloat16",
        "--chunked-prefill-size", str(chunked_size),
        "--trust-remote-code",
        "--port", str(port),
    ]
    if not graph:
        args.append("--disable-cuda-graph")
    if not radix:
        args.append("--disable-radix-cache")
    return args


def _baseline_args(chunked_size="-1", radix=True, graph=False, port=BASE_PORT):
    args = [
        "--model-path", MODEL_PATH,
        "--tokenizer-path", TOKENIZER_PATH,
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--context-length", "4096",
        "--attention-backend", "flashinfer",
        "--dtype", "bfloat16",
        "--chunked-prefill-size", str(chunked_size),
        "--trust-remote-code",
        "--port", str(port),
    ]
    if not graph:
        args.append("--disable-cuda-graph")
    if not radix:
        args.append("--disable-radix-cache")
    return args


# --------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------
_TOKENIZER = None


def _prompt_ids():
    # Encode the prompt with the model's own tokenizer (from TOKENIZER_PATH) so
    # the ids match the served model exactly. Cached across the two servers.
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, trust_remote_code=True
        )
    return _TOKENIZER.encode(PROMPT)


def _capture_extend(port):
    """Feed prompt + fixed prefix as one prefill; return {tok: logprob} top-k
    for the next token."""
    input_ids = list(_prompt_ids()) + PREFIX_GEN_IDS
    d = _http_json(
        port,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": TOPK,
        },
    )
    top = d["meta_info"]["output_top_logprobs"][0]
    return {e[1]: e[0] for e in top}, d.get("output_ids", [None])[0]


def _capture_freegen(port):
    d = _http_json(
        port,
        "/generate",
        {
            "text": PROMPT,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": N_DECODE},
            "return_logprob": True,
            "top_logprobs_num": TOPK,
        },
    )
    return d.get("output_ids"), d["meta_info"]["output_top_logprobs"]


# --------------------------------------------------------------------------
# Long-prompt helpers (#131 chunked prefill: a prompt long enough to span
# several chunks at --chunked-prefill-size 256, so the has_prefix=True worker
# branch and the cross-chunk owner-write -> owned-prefix read are exercised).
# --------------------------------------------------------------------------
_LONG_IDS = None


def _long_prompt_ids():
    """A deterministic >=600-token prompt (>= 3 chunks at chunk size 256).
    Encoded with the served model's own tokenizer so ids match exactly."""
    global _LONG_IDS
    if _LONG_IDS is not None:
        return _LONG_IDS
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    sentences = [
        f"Fact {i}: the number {i * 7 + 3} is followed by {i * 7 + 4}, "
        f"and city {i} lies on river {i % 13}."
        for i in range(90)
    ]
    text = (
        "You will memorize the following numbered facts and then answer.\n"
        + "\n".join(sentences)
        + "\nQuestion: what number follows 24? Answer:"
    )
    ids = tok.encode(text)
    assert len(ids) >= 600, f"long prompt too short ({len(ids)} tok)"
    _LONG_IDS = ids
    return ids


def _capture_extend_ids(port, ids):
    """Prefill an explicit id sequence; return ({tok: logprob} top-k, next_tok)
    for the next token. Used for the long-prompt chunked-vs-single-shot gate."""
    d = _http_json(
        port,
        "/generate",
        {
            "input_ids": list(ids),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": TOPK,
        },
    )
    top = d["meta_info"]["output_top_logprobs"][0]
    return {e[1]: e[0] for e in top}, d.get("output_ids", [None])[0]


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------
def _argmax(topmap):
    return max(topmap, key=lambda t: topmap[t])


def _common_delta(wmap, bmap):
    common = set(wmap) & set(bmap)
    if not common:
        return None
    return max(abs(wmap[t] - bmap[t]) for t in common)


def _top2_margin(topmap):
    vals = sorted(topmap.values(), reverse=True)
    return (vals[0] - vals[1]) if len(vals) > 1 else float("inf")


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------
def _require_env():
    try:
        import torch
    except Exception:
        pytest.skip("torch unavailable")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        pytest.skip("need >= 3 visible GPUs")
    if not MODEL_PATH:
        pytest.skip("WL_MODEL_PATH not set (path to the dense GGUF)")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model GGUF not found: {MODEL_PATH}")


def test_weightless_byte_identity():
    _require_env()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_PY + ":" + env.get("PYTHONPATH", "")
    env["SGLANG_UNEVEN_DCP"] = "1"

    wl_proc = base_proc = None
    try:
        # --- weightless server: capture extend top-k + free-gen trajectory ---
        wl_proc, _ = _boot(_weightless_args(), env, "/tmp/wl_bi_weightless.log")
        _wait_ready(wl_proc, WL_PORT, "/tmp/wl_bi_weightless.log")
        wl_extend, wl_ext_tok = _capture_extend(WL_PORT)
        wl_ids, wl_top = _capture_freegen(WL_PORT)
        _teardown(wl_proc)
        wl_proc = None

        # --- baseline TP=1-solo: same captures ---
        base_env = dict(env)
        base_env["CUDA_VISIBLE_DEVICES"] = "0"  # fastest-first -> the 5090
        base_proc, _ = _boot(_baseline_args(), base_env, "/tmp/wl_bi_baseline.log")
        _wait_ready(base_proc, BASE_PORT, "/tmp/wl_bi_baseline.log")
        base_extend, base_ext_tok = _capture_extend(BASE_PORT)
        base_ids, base_top = _capture_freegen(BASE_PORT)
        _teardown(base_proc)
        base_proc = None

        # ---------- (1) EXTEND path bit-identical ----------
        extend_delta = _common_delta(wl_extend, base_extend)
        print(f"\n[extend] next tok wl={wl_ext_tok} base={base_ext_tok} "
              f"max|Δ|={extend_delta:.3e}")
        assert wl_ext_tok == base_ext_tok, (
            f"extend next-token differs: wl={wl_ext_tok} base={base_ext_tok}"
        )
        assert extend_delta is not None and extend_delta <= EXTEND_EXACT_EPS, (
            f"extend path NOT bit-identical: max|Δ|={extend_delta:.3e} "
            f"(> {EXTEND_EXACT_EPS}); weightless geometry/merge regression"
        )

        # ---------- (2) DECODE trajectory: benign fp-order only ----------
        n = min(len(wl_top), len(base_top))
        first_flip = None
        print("[decode] step  w_tok  b_tok  contenderΔ  match")
        for s in range(n):
            wmap = {e[1]: e[0] for e in wl_top[s]}
            bmap = {e[1]: e[0] for e in base_top[s]}
            wa, ba = _argmax(wmap), _argmax(bmap)
            # Δ on the argmax-contender tokens (union of the two argmaxes) --
            # the meaningful measure; the raw top-k max is dominated by amplified
            # tail tokens (logprob -14..-15) and is intentionally NOT asserted on.
            contenders = {wa, ba}
            cd = max(
                abs(wmap[t] - bmap[t])
                for t in contenders
                if t in wmap and t in bmap
            )
            print(f"         {s:<5}{wa:<7}{ba:<7}{cd:.3e}   {'OK' if wa==ba else 'FLIP'}")
            if s == 0:
                assert cd <= EXTEND_EXACT_EPS, (
                    f"decode step 0 (prompt-prefill extend path) NOT exact: "
                    f"Δ={cd:.3e}"
                )
            assert cd <= DECODE_FP_BAND, (
                f"decode step {s} contender Δ={cd:.3e} exceeds fp-order band "
                f"{DECODE_FP_BAND}: systematic divergence, not decode fp-order"
            )
            if wa != ba:
                first_flip = s
                break

        if first_flip is None:
            print("[decode] no argmax divergence across lockstep steps")
        else:
            s = first_flip
            bmap = {e[1]: e[0] for e in base_top[s]}
            base_margin = _top2_margin(bmap)
            print(f"[decode] first flip @ step {s}; baseline top-2 margin="
                  f"{base_margin:.4f}")
            assert base_margin <= FLIP_TIE_EPS, (
                f"first argmax flip at step {s} is a CONFIDENT baseline "
                f"disagreement (margin={base_margin:.4f} > {FLIP_TIE_EPS}), "
                f"not a coin-flip near-tie -> real divergence, not fp-order"
            )
    finally:
        _teardown(wl_proc)
        _teardown(base_proc)


CHUNK_SIZE = int(os.environ.get("WL_CHUNK_SIZE", "256"))


def _boot_capture_long(args, port, log, cap_extend=True, cap_freegen=False,
                       n_selfdet=0):
    """Boot one server, run the requested captures on the LONG prompt, tear it
    down, and return the results. Never keeps two weightless servers up at once
    (they both need the 5090 head)."""
    proc = None
    try:
        proc, _ = _boot(args, _ENV, log)
        _wait_ready(proc, port, log)
        out = {}
        if cap_extend:
            out["extend"] = _capture_extend_ids(port, _long_prompt_ids())
        if cap_freegen:
            d = _http_json(
                port,
                "/generate",
                {
                    "input_ids": list(_long_prompt_ids()),
                    "sampling_params": {
                        "temperature": 0.0, "max_new_tokens": N_DECODE
                    },
                    "return_logprob": True,
                    "top_logprobs_num": TOPK,
                },
            )
            out["freegen"] = (
                d.get("output_ids"), d["meta_info"]["output_top_logprobs"]
            )
        if n_selfdet:
            outs = []
            for _ in range(n_selfdet):
                d = _http_json(
                    port,
                    "/generate",
                    {
                        "input_ids": list(_long_prompt_ids()),
                        "sampling_params": {
                            "temperature": 0.0, "max_new_tokens": N_DECODE
                        },
                    },
                )
                outs.append(tuple(d.get("output_ids") or []))
            out["selfdet"] = outs
        return out
    finally:
        _teardown(proc)


_ENV = None


def test_weightless_chunked_prefill():
    """#131: CHUNKED prefill on the weightless lane -- correctness gate.

    IMPORTANT byte-identity classification (validated empirically 2026-07-20,
    see the four-way falsifier below). The weightless lane has TWO distinct
    prefill classes with DIFFERENT byte-identity bars, and the difference is
    STRUCTURAL, not a bug:

      * SINGLE-SHOT prefill (fresh prompt, EMPTY prefix): the head holds all
        q-heads and the freshly-projected current-chunk k/v, so its attention
        is computed HEAD-LOCALLY with NO cross-rank reduction. It is therefore
        MACHINE-ZERO bit-identical to TP=1-solo. Measured: Δ == 0.000 on every
        top-k token.

      * CHUNKED prefill (chunk >= 2, i.e. a NON-empty committed prefix): the
        prefix KV lives TOKEN-SHARDED across the weightless workers. The head
        owns only its shard, so it q-broadcasts, does a paged read over its
        OWNED prefix slots, and LSE-MERGES the per-rank partials across the DCP
        group (cp_lse_ag_out_ar_mha_uneven). That logsumexp merge fp-
        REASSOCIATES exactly like the DECODE path -- so the chunked prefix read
        is DECODE-CLASS, NOT machine-zero. Asking for Δ==0 here is asking the
        sharded LSE-merge to bit-match a single-rank softmax, which is
        impossible in floating point (the same reason decode is not exact).

    Hence the gates below:
      (1) SINGLE-SHOT wl vs solo  -> MACHINE-ZERO (Δ==0). The head-local invariant.
      (2) CHUNKED   wl vs solo    -> DECODE-CLASS: argmax next-token MATCHES and
          the contender Δ stays within the intrinsic DCP fp-order band
          (DECODE_FP_BAND) -- a real cross-chunk owner-write/prefix-read
          CORRUPTION bug would flip the argmax or diverge confidently, which
          this catches. Plus the full decode trajectory within band (first flip
          on a near-tie) and self-determinism 5/5.

    NOTE on chunked-vs-single-shot: these are NOT bit-identical even on stock
    TP=1-solo, because chunked uses the PAGED prefix-read kernel while single-
    shot uses ONE ragged pass -- a paged-vs-ragged reassociation independent of
    the lane (measured solo intrinsic top-token Δ ~8e-4). So we do NOT assert
    chunked==single-shot; we anchor chunked to solo-CHUNKED (same kernels, only
    the DCP shard differs) for the lane-correctness signal. Radix/prefix caching
    is exercised ON and OFF (WL_DISABLE_RADIX=1); CUDA-graph via WL_GRAPH=1.
    """
    _require_env()
    global _ENV
    _ENV = dict(os.environ)
    _ENV["PYTHONPATH"] = REPO_PY + ":" + _ENV.get("PYTHONPATH", "")
    _ENV["SGLANG_UNEVEN_DCP"] = "1"

    radix = os.environ.get("WL_DISABLE_RADIX") != "1"
    graph = os.environ.get("WL_GRAPH") == "1"
    tag = f"radix={'ON' if radix else 'OFF'} graph={'ON' if graph else 'OFF'}"
    print(f"\n=== #131 chunked-prefill byte-identity [{tag}], "
          f"chunk={CHUNK_SIZE} ===")
    n_prompt = len(_long_prompt_ids())
    print(f"long prompt = {n_prompt} tok -> "
          f"~{-(-n_prompt // CHUNK_SIZE)} chunks at size {CHUNK_SIZE}")

    # (A) weightless CHUNKED: extend + decode + self-determinism.
    chunked = _boot_capture_long(
        _weightless_args(chunked_size=CHUNK_SIZE, radix=radix, graph=graph,
                         port=WL_PORT),
        WL_PORT, "/tmp/wl_chunked.log",
        cap_extend=True, cap_freegen=True, n_selfdet=5,
    )
    # (B) weightless SINGLE-SHOT (same lane, one extend forward).
    single = _boot_capture_long(
        _weightless_args(chunked_size=-1, radix=radix, graph=graph,
                         port=WL_PORT),
        WL_PORT, "/tmp/wl_single.log", cap_extend=True,
    )
    # (C) TP=1-solo CHUNKED ground truth (same paged kernels, no DCP shard):
    #     extend + decode in one boot. (D) TP=1-solo SINGLE-SHOT: the machine-
    #     zero anchor for the head-local single-shot invariant.
    solo_env = dict(_ENV)
    solo_env["CUDA_VISIBLE_DEVICES"] = "0"
    _saved_env = _ENV
    globals()["_ENV"] = solo_env
    try:
        solo = _boot_capture_long(
            _baseline_args(chunked_size=CHUNK_SIZE, radix=radix, graph=graph,
                           port=BASE_PORT),
            BASE_PORT, "/tmp/wl_solo.log", cap_extend=True, cap_freegen=True,
        )
        solo_sg = _boot_capture_long(
            _baseline_args(chunked_size=-1, radix=radix, graph=graph,
                           port=BASE_PORT),
            BASE_PORT, "/tmp/wl_solo_single.log", cap_extend=True,
        )
    finally:
        globals()["_ENV"] = _saved_env

    ch_map, ch_tok = chunked["extend"]
    sg_map, sg_tok = single["extend"]
    so_map, so_tok = solo["extend"]
    sos_map, sos_tok = solo_sg["extend"]

    # ---------- (1) SINGLE-SHOT wl vs solo: MACHINE-ZERO (head-local) --------
    # A fresh prompt has an empty prefix -> the head computes attention locally
    # over all its q-heads with no cross-rank LSE-merge, so it is bit-identical
    # to TP=1-solo (no fp reassociation to introduce). This is the invariant the
    # short-prompt test already guards; here on the long prompt.
    d_ss = _common_delta(sg_map, sos_map)
    print(f"[single ] wl next={sg_tok} solo next={sos_tok} max|Δ|={d_ss:.3e}")
    assert sg_tok == sos_tok, (
        f"single-shot next-token differs: wl={sg_tok} solo={sos_tok}"
    )
    assert d_ss is not None and d_ss <= EXTEND_EXACT_EPS, (
        f"SINGLE-SHOT wl vs solo NOT machine-zero: max|Δ|={d_ss:.3e} "
        f"(> {EXTEND_EXACT_EPS}) -- head-local single-shot invariant broken"
    )

    # ---------- (2) CHUNKED wl vs solo: DECODE-CLASS (sharded prefix read) ---
    # The chunk>=2 prefix read is a DCP-sharded LSE-merge (same as decode), so
    # it is NOT machine-zero; the correct bar is argmax-preserved + contender
    # within the intrinsic DCP fp-order band. A cross-chunk owner-write /
    # owned-prefix-read CORRUPTION bug would flip the argmax or diverge
    # confidently -- this catches it. (max|Δ| over ALL top-k is tail-token
    # softmax amplification and is intentionally NOT asserted, matching the
    # decode-trajectory convention below.)
    ch_contender = max(
        abs(ch_map[t] - so_map[t])
        for t in {ch_tok, so_tok} if t in ch_map and t in so_map
    )
    print(f"[chunked] wl next={ch_tok} solo next={so_tok} "
          f"contenderΔ={ch_contender:.3e} max|Δ|={_common_delta(ch_map, so_map):.3e}")
    assert ch_tok == so_tok, (
        f"CHUNKED wl vs solo next-token differs: wl={ch_tok} solo={so_tok} "
        f"-- argmax flip => real cross-chunk prefix-read bug"
    )
    assert ch_contender <= DECODE_FP_BAND, (
        f"CHUNKED wl vs solo contender Δ={ch_contender:.3e} exceeds DCP fp-band "
        f"{DECODE_FP_BAND} -- systematic divergence, not sharded-merge fp-order"
    )
    # Informational only (NOT asserted machine-zero): chunked-vs-single-shot is
    # a paged-vs-ragged kernel reassociation present without the lane.
    print(f"[info   ] chunked-vs-single-shot max|Δ|="
          f"{_common_delta(ch_map, sg_map):.3e} (kernel reassoc, not asserted)")

    # ---------- (3) self-determinism 5/5 on the chunked lane ----------------
    sd = chunked["selfdet"]
    assert len(set(sd)) == 1, f"chunked lane not self-deterministic: {set(sd)}"
    print(f"[selfdet] chunked 5/5 identical output_ids: OK")

    # ---------- (4) decode trajectory: benign fp-order only (chunked vs solo) -
    ch_ids, ch_top = chunked["freegen"]
    so_ids, so_top = solo["freegen"]

    n = min(len(ch_top), len(so_top))
    first_flip = None
    print("[decode] step  c_tok  s_tok  contenderΔ  match")
    for s in range(n):
        wmap = {e[1]: e[0] for e in ch_top[s]}
        bmap = {e[1]: e[0] for e in so_top[s]}
        wa, ba = _argmax(wmap), _argmax(bmap)
        cd = max(
            abs(wmap[t] - bmap[t]) for t in {wa, ba} if t in wmap and t in bmap
        )
        print(f"         {s:<5}{wa:<7}{ba:<7}{cd:.3e}   "
              f"{'OK' if wa == ba else 'FLIP'}")
        # NOTE: step 0 here is the chunked-prefill pre-decode logit, which is
        # DECODE-CLASS (sharded prefix read), so it is within-band, NOT exact
        # (unlike the single-shot step-0 in the sibling test).
        assert cd <= DECODE_FP_BAND, (
            f"decode step {s} contender Δ={cd:.3e} exceeds fp-order band"
        )
        if wa != ba:
            first_flip = s
            break
    if first_flip is None:
        print("[decode] no argmax divergence across lockstep steps")
    else:
        bmap = {e[1]: e[0] for e in so_top[first_flip]}
        m = _top2_margin(bmap)
        print(f"[decode] first flip @ step {first_flip}; solo top-2 margin={m:.4f}")
        assert m <= FLIP_TIE_EPS, (
            f"first flip @ step {first_flip} is a confident disagreement "
            f"(margin={m:.4f} > {FLIP_TIE_EPS}), not fp-order"
        )
    print(f"#131 CHUNKED-PREFILL CORRECTNESS GREEN [{tag}]")


if __name__ == "__main__":
    test_weightless_byte_identity()
    print("\nWEIGHTLESS BYTE-IDENTITY REGRESSION GREEN")
    test_weightless_chunked_prefill()
    print("\n#131 CHUNKED-PREFILL REGRESSION GREEN")
