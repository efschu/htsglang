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
    /spinning/htsglang-gpu/.venv/bin/python -m pytest -s \
        test/registered/layers/test_weightless_byte_identity.py
  or directly:
    /spinning/htsglang-gpu/.venv/bin/python \
        test/registered/layers/test_weightless_byte_identity.py

Override the model via env: WL_MODEL_PATH, WL_TOKENIZER_PATH.
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
REPO_PY = "/spinning/wt-weightless-kv/python"
VENV_PY = "/spinning/htsglang-gpu/.venv/bin/python"
MODEL_PATH = os.environ.get(
    "WL_MODEL_PATH",
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.6-27B-MTP-Q3_K_M-GGUF/Qwen3.6-27B-Q3_K_M.gguf",
)
TOKENIZER_PATH = os.environ.get(
    "WL_TOKENIZER_PATH",
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
)
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


def _weightless_args():
    return [
        "--model-path", MODEL_PATH,
        "--tokenizer-path", TOKENIZER_PATH,
        "--tp-size", "3", "--dcp-size", "3",
        "--weightless-kv-fastlane", "--weightless-kv-head-rank", "0",
        "--rank-gpu-id", "0,1,2",
        "--rank-gpu-memory-mib", "28000,19000,19000",
        "--context-length", "4096",
        "--attention-backend", "flashinfer",
        "--dtype", "bfloat16",
        "--chunked-prefill-size", "-1",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--port", str(WL_PORT),
    ]


def _baseline_args():
    return [
        "--model-path", MODEL_PATH,
        "--tokenizer-path", TOKENIZER_PATH,
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--context-length", "4096",
        "--attention-backend", "flashinfer",
        "--dtype", "bfloat16",
        "--chunked-prefill-size", "-1",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--port", str(BASE_PORT),
    ]


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


if __name__ == "__main__":
    test_weightless_byte_identity()
    print("\nWEIGHTLESS BYTE-IDENTITY REGRESSION GREEN")
