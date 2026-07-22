# SPDX-License-Identifier: Apache-2.0
"""The GPU runner for the #124 harness.

:func:`run_case_config` (formerly the one deliberate stub) is now
implemented against the integration/r2 GPU window on this box; the seam
contract below is kept verbatim as its specification. The logits surface is
the fork's ``--determinism-logits-dump-dir`` debug dump (ServerArgs-threaded,
full-vocab, dtype-as-served -- see ModelRunner._determinism_dump_logits),
chosen over HTTP top-k logprobs so the machine-zero rows stay dtype-strict
and full-vocab.

========================  THE SEAM (r2 GPU wiring)  ========================
``run_case_config(case, which)`` must, per boot:

1.  BOOT: launch ``python -m sglang.launch_server`` (or Engine API) with the
    row's server args (``case.test_config`` / ``reference_config``) INCLUDING
    the pinned ``random_seed`` -- never let sglang randomize it -- and the
    row's env baked into the MAIN process environment before launch. Do NOT
    rely on worker-side env toggles: sglang scrubs custom env for scheduler
    TP workers, making them silent no-ops (memory ``full-perf-testen``).
    Model path comes from a ``model_role -> local path`` map resolved at the
    GPU window (dense_lane / moe_fp8 / moe_marlin vehicles).

2.  SERIALIZE: this is a shared box. One boot at a time per GPU set; track
    and kill ONLY your own PIDs on teardown -- never a broad pkill (memory
    ``shared-box-agent-koordination``). Co-located gates (test + rerun) are
    sequential fresh boots with the identical command line, or same-boot
    repeated requests where the gate definition allows.

3.  DRIVE: send the fixed gate prompt with ``temperature=0``,
    ``max_new_tokens=case.num_decode_tokens``, requesting per-step logprobs
    (``return_logprob`` + a large ``top_logprobs_num``). Full-vocab rows are
    preferred; a top-k slice is acceptable ONLY if k is large enough that
    every flip candidate is inside the slice -- margin classification
    degrades to "flip outside top-k = CORRUPTION by default", which is
    conservative but can misclassify a deep near-tie. If the HTTP surface
    cannot return raw logits, add a debug dump env in the fork (server-side
    per-step logits row capture) rather than weakening the class assertions.
    For the prefill row (``num_decode_tokens == 1``) capture the
    last-position prefill logits. For ``chunked_prefill_with_prefix``, first
    warm the radix prefix with a prefix-sharing request, then run the gate
    request. For graph rows, keep the full-perf discipline: graphs ON is the
    test arm, ``--disable-cuda-graph`` ONLY as the designated reference arm.

4.  COLLECT: build ``Trajectory(token_ids, logits, seed=case.seed,
    label=f"{case.case_id}:{which}")`` with logits rows in decode-step order
    and dtype preserved as served (machine-zero rows are dtype-strict).

5.  TEARDOWN: shut down the server (own PID only), verify port free before
    the next boot.

Then the gate is simply::

    ref  = run_case_config(case, "reference")
    test = run_case_config(case, "test")
    rerun = run_case_config(case, "test-rerun") if case.needs_rerun else None
    verdict = evaluate_case(case, ref=ref, test=test, rerun=rerun)

===========================================================================
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest
import torch

from .classes import check_class
from .matrix import TEST_MATRIX, CaseSpec, validate_matrix
from .trajectory import Trajectory, Verdict

__all__ = ["gpu_available", "requires_gpu", "run_case_config", "evaluate_case", "run_matrix"]


def gpu_available() -> bool:
    return torch.cuda.is_available()


#: pytest marker for anything that needs the GPU runner wiring.
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA + the r2 GPU runner wiring (server boots)",
)


# ===========================================================================
# Host bindings (this box: 1x RTX 5090 = cuda:0 + 2x RTX 3080 = cuda:1/2).
# --rank-gpu-id is CUDA (fastest-first) order, NOT NVML order (memory
# ``tp5-emulation-uneven-gguf-bugs``); _preflight_device_order() verifies the
# 5090 really is cuda:0 in a throwaway subprocess before any boot.
# ===========================================================================

# Local model-zoo root (machine-specific): HTSGLANG_TEST_MODEL_DIR must point
# at the populated cache; without it the per-row boots fail with a clear
# missing-path message instead of guessing.
_MODELS_ROOT = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "/models")

#: model_role -> boot bindings. ``None`` = no usable vehicle on this box (the
#: runner fails that row's boot with a clear message instead of guessing).
MODEL_ROLES: Dict[str, Optional[Dict[str, Any]]] = {
    "dense_lane": {
        "model_path": f"{_MODELS_ROOT}/Qwen3.6-27B-MTP-Q3_K_M-GGUF/Qwen3.6-27B-Q3_K_M.gguf",
        "tokenizer_path": f"{_MODELS_ROOT}/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
        "extra_args": [
            "--load-format", "gguf",
            "--quantization", "gguf",
            "--attention-backend", "flashinfer",
            "--dtype", "float16",
            "--context-length", "16384",
        ],
    },
    # A real 256-expert MoE that fits the 5090 at TP=1 (20 GB GPTQ-Int4 ->
    # marlin MoE kernels). The briefing's Qwen3.6-27B-* "MoE" vehicles are
    # dense qwen3_5 checkpoints (no experts) and cannot exercise the offload
    # rows.
    "moe_marlin": {
        "model_path": f"{_MODELS_ROOT}/Qwen3.5-35B-A3B-GPTQ-Int4",
        "tokenizer_path": None,
        "extra_args": [
            # The GPTQ checkpoint carries float16 scales; the config default
            # (bfloat16) trips moe_wna16_marlin_gemm's dtype assert.
            "--dtype", "float16",
            "--context-length", "16384",
        ],
    },
    # No FP8 MoE vehicle fits this box at TP=1: Qwen3.6-35B-A3B-FP8 is 31 GB
    # (5090 = 32 GB total; the no-offload REFERENCE arm cannot boot), and
    # Qwen3.6-27B-FP8 is a dense model. Report, don't guess.
    "moe_fp8": None,
}

#: tp_size -> hardware placement (validated lane budgets from the #136a/#136b
#: boots: head 5090 26000 MiB, workers 3080 17000 MiB).
_GPU_ARGS: Dict[int, List[str]] = {
    1: ["--rank-gpu-id", "0", "--rank-gpu-memory-mib", "26000"],
    3: ["--rank-gpu-id", "0,1,2", "--rank-gpu-memory-mib", "26000,17000,17000"],
}

_PORT = int(os.environ.get("SGLANG_DET_PORT", "30124"))
_WORKDIR = Path(os.environ.get("SGLANG_DET_WORKDIR", "/tmp/det124"))
_HEALTH_TIMEOUT_S = int(os.environ.get("SGLANG_DET_HEALTH_TIMEOUT_S", "2400"))
_GPU_WAIT_S = int(os.environ.get("SGLANG_DET_GPU_WAIT_S", "1800"))
_GEN_TIMEOUT_S = int(os.environ.get("SGLANG_DET_GEN_TIMEOUT_S", "1800"))

# Fixed gate prompts (plain constants -> identical across every arm/boot).
#
# PROMPT-MARGIN DISCIPLINE (measured at the GPU window): the lane's
# fp-reassociation envelope vs. solo is O(0.3-3) absolute on fp32 logits, so
# an open-ended natural-language prompt WILL pass through greedy near-ties
# with top-2 margins below the delta scale somewhere in a 96-step window
# (measured: min margins 0.062/0.016/0.125 on the first-draft prompts; one
# genuine near-tie fork at ref-margin 0.047 actually flipped). A 0-flip
# gate over such a trajectory is luck, not a contract. The gate prompts are
# therefore LOW-ENTROPY tasks (deterministic enumeration / verbatim copy)
# whose greedy margins sit far above the fp band over the whole window --
# which turns "0 flips" into a consequence of in-band deltas and makes any
# flip a real out-of-band signal. Runner asserts nothing about margins; the
# measured min-margin per row is recorded in the matrix notes.
_DEFAULT_PROMPT = (
    "Count carefully from one to two hundred in English words, lowercase, "
    "separated by commas, without stopping: one, two, three, four, five,"
)
_PREFIX_SENTENCE = (
    "The archive room kept meticulous records of every experiment: dates, "
    "seeds, kernel versions, and the exact ordering of collective operations. "
)
#: chunked_prefill_with_prefix: warm this prefix first, then gate on
#: prefix+suffix. The suffix is a verbatim-copy task (induction-head
#: attractor -> confident margins).
_CHUUNK_SUFFIX = (
    "\n\nRepeat the first sentence above exactly, word for word, and keep "
    "repeating it without stopping: The archive room kept"
)
_CHUNK_PREFIX = _PREFIX_SENTENCE * 40
#: streaming_host_spill: long prompt so each rank's owned KV shard exceeds the
#: 2048-slot device cap (seq/dcp > 2048 -> genuine host streaming).
_SPILL_PROMPT = _PREFIX_SENTENCE * 480 + _CHUUNK_SUFFIX
_SPILL_MIN_PROMPT_TOKENS = 7000


# ===========================================================================
# Server lifecycle (shared box discipline: ONE boot at a time, own PIDs only)
# ===========================================================================


@dataclass
class _Server:
    key: str
    proc: subprocess.Popen
    port: int
    dump_dir: Path
    log_path: Path
    env_marker: Dict[str, str] = field(default_factory=dict)


_LIVE: Optional[_Server] = None
_BOOT_COUNTER = 0
_DEVICE_ORDER_OK = False


def _repo_python_root() -> str:
    # tests/determinism/determinism_harness/runner.py -> repo root / python
    return str(Path(__file__).resolve().parents[3] / "python")


def _child_env(extra: Mapping[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _repo_python_root()
    nvrtc = Path(sys.executable).resolve().parents[1] / (
        "lib/python%d.%d/site-packages/nvidia/cu13/lib" % sys.version_info[:2]
    )
    if nvrtc.is_dir():
        env["LD_LIBRARY_PATH"] = f"{nvrtc}:{env.get('LD_LIBRARY_PATH', '')}"
    env.update(extra)
    return env


def _preflight_device_order() -> None:
    """Assert cuda:0 is the RTX 5090 (the head / solo GPU) in a THROWAWAY
    subprocess -- never initialize CUDA in the harness process itself (a
    lingering context would squat VRAM next to the server boots)."""
    global _DEVICE_ORDER_OK
    if _DEVICE_ORDER_OK:
        return
    out = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('|'.join(torch.cuda.get_device_name(i) "
         "for i in range(torch.cuda.device_count())))"],
        capture_output=True, text=True, timeout=120, env=_child_env({}),
    )
    names = (out.stdout or "").strip().split("|")
    if not names or "5090" not in names[0]:
        raise RuntimeError(
            f"device-order preflight failed: expected the RTX 5090 at cuda:0 "
            f"(CUDA fastest-first order), got {names!r} "
            f"(stderr: {out.stderr[-500:]}). The _GPU_ARGS bindings assume "
            f"5090=cuda:0; refusing to boot with a wrong mapping."
        )
    _DEVICE_ORDER_OK = True


def _own_pids() -> set:
    pids = set()
    if _LIVE is not None and _LIVE.proc.poll() is None:
        try:
            out = subprocess.run(
                ["ps", "-o", "pid=", "--ppid", str(_LIVE.proc.pid)],
                capture_output=True, text=True, timeout=30,
            ).stdout
            pids = {int(p) for p in out.split()}
        except Exception:
            pass
        pids.add(_LIVE.proc.pid)
    return pids


def _gpu_compute_pids() -> List[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _wait_gpu_window() -> None:
    """Shared-box rule: never boot while a foreign process holds the GPUs.
    Bounded poll (anti-idle discipline), then a hard error -- we never kill
    anything that is not ours."""
    deadline = time.time() + _GPU_WAIT_S
    while True:
        foreign = [p for p in _gpu_compute_pids() if p not in _own_pids()]
        if not foreign:
            return
        if time.time() > deadline:
            raise RuntimeError(
                f"GPU window not free after {_GPU_WAIT_S}s: foreign compute "
                f"pids {foreign} hold the GPUs. Refusing to boot (shared-box "
                f"rule: serialize boots, never kill foreign processes)."
            )
        time.sleep(15)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _http(path: str, payload: Optional[dict] = None, timeout: float = 60.0) -> Any:
    url = f"http://127.0.0.1:{_PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def teardown_server() -> None:
    """Kill OUR server process group (never a broad pkill), wait for the port
    and the GPUs to drain."""
    global _LIVE
    if _LIVE is None:
        return
    srv, _LIVE = _LIVE, None
    if srv.proc.poll() is None:
        try:
            os.killpg(os.getpgid(srv.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.time() + 60
        while srv.proc.poll() is None and time.time() < deadline:
            time.sleep(1)
        if srv.proc.poll() is None:
            try:
                os.killpg(os.getpgid(srv.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            srv.proc.wait(timeout=30)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _port_free(srv.port) and srv.proc.pid not in _gpu_compute_pids():
            break
        time.sleep(2)
    time.sleep(3)  # VRAM settle


atexit.register(teardown_server)


def _boot_key(case: CaseSpec, which: str) -> Tuple[str, str]:
    cfg, env = _arm_config(case, which)
    key = json.dumps(
        {"role": case.model_role, "cfg": dict(sorted(cfg.items())),
         "env": dict(sorted(env.items()))},
        sort_keys=True,
    )
    return key, ""


def _arm_config(case: CaseSpec, which: str) -> Tuple[Mapping[str, Any], Mapping[str, str]]:
    if which == "reference":
        return case.reference_config, case.reference_env
    if which in ("test", "test-rerun"):
        return case.test_config, case.test_env
    raise ValueError(f"unknown arm {which!r}")


def _config_to_cli(cfg: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in sorted(cfg.items()):
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args += [flag, str(v)]
    return args


def _ensure_server(case: CaseSpec, which: str) -> _Server:
    global _LIVE, _BOOT_COUNTER
    key, _ = _boot_key(case, which)
    if _LIVE is not None and _LIVE.key == key and _LIVE.proc.poll() is None:
        return _LIVE
    role = MODEL_ROLES.get(case.model_role)
    if role is None:
        raise RuntimeError(
            f"row {case.case_id!r}: no usable model vehicle for role "
            f"{case.model_role!r} on this box (see MODEL_ROLES in runner.py). "
            f"Reported, not guessed."
        )
    teardown_server()
    _preflight_device_order()

    cfg, env_extra = _arm_config(case, which)
    tp = int(cfg.get("tp_size", 1))
    if tp not in _GPU_ARGS:
        raise RuntimeError(f"no GPU placement binding for tp_size={tp}")

    _wait_gpu_window()
    if not _port_free(_PORT):
        raise RuntimeError(f"port {_PORT} still occupied before boot")

    _BOOT_COUNTER += 1
    # Unique per-boot directory (pid + epoch): the dump-row watermark is
    # name-based, so a REUSED dump dir from an earlier runner process would
    # let a fresh server silently overwrite same-named stale step files and
    # hide the gate rows from collection (observed live on run 1).
    boot_dir = _WORKDIR / f"boot_{_BOOT_COUNTER:03d}_{os.getpid()}_{int(time.time())}"
    dump_dir = boot_dir / "dump"
    dump_dir.mkdir(parents=True, exist_ok=False)
    log_path = boot_dir / "server.log"

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", role["model_path"],
    ]
    if role["tokenizer_path"]:
        cmd += ["--tokenizer-path", role["tokenizer_path"]]
    cmd += role["extra_args"]
    cmd += _GPU_ARGS[tp]
    cmd += _config_to_cli(cfg)
    cmd += [
        "--determinism-logits-dump-dir", str(dump_dir),
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", str(_PORT),
    ]
    # Row env goes into the MAIN process environment before launch (worker-
    # side env is scrubbed for TP scheduler workers -- memory full-perf-testen).
    env = _child_env(env_extra)

    log_f = open(log_path, "w")
    log_f.write(f"# cmd: {' '.join(cmd)}\n# env_extra: {dict(env_extra)}\n\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    _LIVE = _Server(key=key, proc=proc, port=_PORT, dump_dir=dump_dir,
                    log_path=log_path, env_marker=dict(env_extra))

    deadline = time.time() + _HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text()[-3000:]
            _LIVE = None
            raise RuntimeError(
                f"server boot for {case.case_id}:{which} exited rc="
                f"{proc.returncode} before healthy. Log tail:\n{tail}"
            )
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{_PORT}/health", timeout=5
            ).read()
            break
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    else:
        tail = log_path.read_text()[-3000:]
        teardown_server()
        raise RuntimeError(
            f"server for {case.case_id}:{which} not healthy after "
            f"{_HEALTH_TIMEOUT_S}s. Log tail:\n{tail}"
        )

    # If this arm claims MoE offload, verify it actually engaged (a silently
    # ignored env would make offload-vs-no-offload compare base-vs-base --
    # the exact silent-no-op failure mode memory full-perf-testen warns about).
    frac = env_extra.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION")
    if frac is not None and float(frac) < 1.0:
        if "expert-offload installed" not in log_path.read_text():
            teardown_server()
            raise RuntimeError(
                f"{case.case_id}:{which}: SGLANG_MOE_RESIDENT_EXPERT_FRACTION="
                f"{frac} was set but the boot log never reported 'expert-"
                f"offload installed' -- the env did not reach the model "
                f"runner; refusing to compare base-vs-base."
            )
    return _LIVE


# ===========================================================================
# Drive + collect
# ===========================================================================


def _case_prompts(case: CaseSpec) -> Tuple[Optional[str], str, Optional[int]]:
    """(warm_prompt, gate_prompt, min_prompt_tokens) per row protocol."""
    if case.case_id == "chunked_prefill_with_prefix":
        return _CHUNK_PREFIX, _CHUNK_PREFIX + _CHUUNK_SUFFIX, None
    if case.case_id == "streaming_host_spill":
        return None, _SPILL_PROMPT, _SPILL_MIN_PROMPT_TOKENS
    return None, _DEFAULT_PROMPT, None


def _dump_rows(dump_dir: Path) -> List[Path]:
    return sorted(dump_dir.glob("rank0_step*.pt"))


def _collect_new_rows(
    dump_dir: Path, watermark: set, num: int, label: str
) -> List[dict]:
    """Collect the gate request's ``num`` logits rows from the dump.

    Alignment rule: the gate trajectory starts at the LAST ``extend`` row
    dumped after ``watermark`` (leading extra ``extend`` rows are
    intermediate chunked-prefill sample calls of a long prompt) and spans the
    ``num - 1`` decode rows that follow it. Rows after that are ignored: the
    overlap scheduler runs one speculative decode forward past the length
    limit, which dumps a trailing phantom row that belongs to no emitted
    token. Bounded wait for the writer, then strict structural assertions.
    """
    deadline = time.time() + 60
    rows: List[dict] = []
    files: List[Path] = []
    while time.time() < deadline:
        files = [f for f in _dump_rows(dump_dir) if f.name not in watermark]
        loaded = [
            torch.load(f, map_location="cpu", weights_only=True) for f in files
        ]
        extend_idx = [i for i, r in enumerate(loaded) if r["mode"] == "extend"]
        if extend_idx and len(loaded) - extend_idx[-1] >= num:
            rows = loaded[extend_idx[-1] : extend_idx[-1] + num]
            break
        time.sleep(1.0)
    if not rows:
        raise RuntimeError(
            f"{label}: could not align {num} gate rows in {dump_dir} "
            f"({len(files)} new rows) -- dump surface misalignment"
        )
    for i, r in enumerate(rows[1:], start=1):
        if r["mode"] != "decode":
            raise RuntimeError(
                f"{label}: row {i} is {r['mode']!r}, expected 'decode'"
            )
    return rows


def run_case_config(case: CaseSpec, which: str) -> Trajectory:
    """Boot (or reuse) the server for one arm of a matrix row, drive the gate
    request, and collect the trajectory from the server-side logits dump.
    ``which`` is ``"reference"``, ``"test"`` or ``"test-rerun"``.
    """
    label = f"{case.case_id}:{which}"
    srv = _ensure_server(case, which)
    T = case.num_decode_tokens

    _http("/flush_cache", payload={}, timeout=120)
    warm, gate_prompt, min_prompt_tokens = _case_prompts(case)
    if warm is not None:
        _http("/generate", {
            "text": warm,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1,
                                "ignore_eos": True},
        }, timeout=_GEN_TIMEOUT_S)
        time.sleep(2.0)

    watermark = {f.name for f in _dump_rows(srv.dump_dir)}
    resp = _http("/generate", {
        "text": gate_prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": T,
                            "ignore_eos": True},
        "return_logprob": True,
    }, timeout=_GEN_TIMEOUT_S)
    meta = resp.get("meta_info", {})
    out_lp = meta.get("output_token_logprobs") or []
    tokens = [int(t[1]) for t in out_lp]
    if len(tokens) != T:
        raise RuntimeError(
            f"{label}: expected {T} output tokens, got {len(tokens)} "
            f"(finish: {meta.get('finish_reason')})"
        )
    if min_prompt_tokens is not None:
        ptok = int(meta.get("prompt_tokens", 0))
        if ptok < min_prompt_tokens:
            raise RuntimeError(
                f"{label}: prompt tokenized to {ptok} tokens < required "
                f"{min_prompt_tokens} (host-spill row must actually stream)"
            )

    rows = _collect_new_rows(srv.dump_dir, watermark, T, label)
    logits = torch.stack([r["logits"] for r in rows])
    argmax = [int(t) for t in logits.argmax(dim=-1)]
    if argmax != tokens:
        first = next(i for i, (a, b) in enumerate(zip(argmax, tokens)) if a != b)
        raise RuntimeError(
            f"{label}: dumped-logits argmax disagrees with the emitted greedy "
            f"tokens at step {first} (argmax={argmax[first]} "
            f"emitted={tokens[first]}) -- capture misalignment, not a verdict"
        )
    return Trajectory(token_ids=tokens, logits=logits, seed=case.seed, label=label)


def evaluate_case(
    case: CaseSpec,
    *,
    ref: Trajectory,
    test: Trajectory,
    rerun: Optional[Trajectory] = None,
) -> Verdict:
    """Fully-implemented pure dispatch: apply the row's expected class to the
    captured trajectories. This is the function the GPU runner calls; it is
    also exercised end-to-end on synthetic trajectories by the CPU suite.
    """
    return check_class(
        case.expected_class,
        ref=ref,
        test=test,
        rerun=rerun,
        band=case.band,
        near_tie_margin=case.near_tie_margin,
    )


def run_matrix(
    case_ids: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> "list[tuple[CaseSpec, object]]":
    """Run matrix rows end-to-end (GPU window entry point).

    Validates the matrix first (fail fast before any boot). Arms are grouped
    by boot configuration so identical server configs share one boot (shared
    box: every boot is serialized and expensive); trajectories are captured
    per arm, then every row is evaluated. A row whose boot/capture fails is
    recorded as an error entry instead of aborting the whole matrix
    (diagnose-first: the error text carries the boot log tail).
    """
    if not gpu_available():
        raise RuntimeError("run_matrix needs CUDA; CPU-side use evaluate_case directly")
    validate_matrix()
    cases = [
        c for c in TEST_MATRIX if case_ids is None or c.case_id in case_ids
    ]
    arms: List[Tuple[CaseSpec, str]] = []
    for case in cases:
        arms.append((case, "reference"))
        arms.append((case, "test"))
        if case.needs_rerun:
            arms.append((case, "test-rerun"))

    # Group by boot key, preserving first-appearance order of keys. A rerun
    # arm shares its test arm's key, so it lands in the same boot.
    key_order: List[str] = []
    grouped: Dict[str, List[Tuple[CaseSpec, str]]] = {}
    for case, which in arms:
        key, _ = _boot_key(case, which)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append((case, which))

    trajs: Dict[Tuple[str, str], Trajectory] = {}
    errors: Dict[str, str] = {}
    try:
        for key in key_order:
            for case, which in grouped[key]:
                if case.case_id in errors:
                    continue
                try:
                    t = run_case_config(case, which)
                    trajs[(case.case_id, which)] = t
                    if save_dir:
                        out = Path(save_dir)
                        out.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"token_ids": list(t.token_ids), "logits": t.logits,
                             "seed": t.seed, "label": t.label},
                            out / f"{case.case_id}__{which}.pt",
                        )
                except Exception as e:  # noqa: BLE001 -- recorded, not hidden
                    errors[case.case_id] = f"{which}: {e}"
                    teardown_server()
    finally:
        teardown_server()

    results: List[Tuple[CaseSpec, object]] = []
    for case in cases:
        if case.case_id in errors:
            results.append((case, RuntimeError(errors[case.case_id])))
            continue
        ref = trajs[(case.case_id, "reference")]
        test = trajs[(case.case_id, "test")]
        rerun = trajs.get((case.case_id, "test-rerun"))
        results.append((case, evaluate_case(case, ref=ref, test=test, rerun=rerun)))
    return results


def load_saved_trajectory(save_dir: str, case_id: str, which: str) -> Trajectory:
    d = torch.load(
        Path(save_dir) / f"{case_id}__{which}.pt",
        map_location="cpu", weights_only=True,
    )
    return Trajectory(
        token_ids=list(d["token_ids"]), logits=d["logits"],
        seed=int(d["seed"]), label=str(d["label"]),
    )


def evaluate_saved(
    save_dir: str, case_ids: Optional[List[str]] = None
) -> "list[tuple[CaseSpec, object]]":
    """Offline re-evaluation: apply the CURRENT matrix (e.g. after band
    calibration) to trajectories captured earlier by :func:`run_matrix`.
    No GPU, no boots -- pure evaluate_case over the saved arms."""
    validate_matrix()
    results: List[Tuple[CaseSpec, object]] = []
    for case in TEST_MATRIX:
        if case_ids is not None and case.case_id not in case_ids:
            continue
        try:
            ref = load_saved_trajectory(save_dir, case.case_id, "reference")
            test = load_saved_trajectory(save_dir, case.case_id, "test")
            rerun = (
                load_saved_trajectory(save_dir, case.case_id, "test-rerun")
                if case.needs_rerun else None
            )
        except FileNotFoundError as e:
            results.append((case, RuntimeError(f"missing saved trajectory: {e}")))
            continue
        results.append((case, evaluate_case(case, ref=ref, test=test, rerun=rerun)))
    return results


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="#124 determinism harness GPU runner (boots servers -- "
        "shared box: make sure the GPU window is yours)"
    )
    p.add_argument("--cases", nargs="*", default=None,
                   help="subset of case_ids (default: full TEST_MATRIX)")
    p.add_argument("--save-dir", default=str(_WORKDIR / "trajectories"),
                   help="where to save captured trajectories (.pt per arm)")
    p.add_argument("--evaluate-saved", action="store_true",
                   help="no boots: re-evaluate saved trajectories from "
                        "--save-dir against the current matrix (band "
                        "calibration loop)")
    args = p.parse_args(argv)

    if args.evaluate_saved:
        results = evaluate_saved(args.save_dir, case_ids=args.cases)
    else:
        results = run_matrix(case_ids=args.cases, save_dir=args.save_dir)
    n_fail = 0
    for case, verdict in results:
        if isinstance(verdict, Exception):
            n_fail += 1
            print(f"[ERROR] {case.case_id}: {verdict}")
        else:
            if not verdict.ok:
                n_fail += 1
            print(verdict.summary())
    print(f"\n{len(results) - n_fail}/{len(results)} rows green")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(_main())
