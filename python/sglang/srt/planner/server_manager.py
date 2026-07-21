# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Local-model manager + sglang server supervisor (dashboard control-plane).

This is the standalone CORE of the dashboard "server manager" feature. The web
UI (``webui.py``) and the docker entrypoint are wired up in a LATER phase; this
module is only the control-plane logic + its unit tests. It does three things:

  1. ``discover_models()`` -- enumerate every local model under the model roots
     (HF-format dirs, HF-hub snapshot dirs, and GGUF dirs with one or more quant
     variants), reading the quant scheme CONFIG-AUTHORITATIVELY (the checkpoint's
     own ``config.json`` ``quantization_config`` / the GGUF header ``file_type``),
     never trusting the directory name.

  2. ``SglangSupervisor`` -- a long-lived supervisor that owns ONE managed sglang
     child process. It can ``start`` / ``stop`` / ``restart`` / ``status`` the
     server so the dashboard can swap models WITHOUT the dashboard (or, later, the
     whole container) exiting. The supervisor process stays alive across a
     restart; only the sglang child is replaced.

  3. Model download (gated on write access) -- ``model_root_writable`` /
     ``available_downloads`` / ``download_model`` let the UI pull a model in a
     chosen quantization into the mounted model root, but ONLY when that root is
     writable (a mounted volume is frequently read-only). GGUF downloads fetch
     only the chosen quant file, not the whole repo.

Reuse discipline (do NOT reimplement -- imported/factored from the proven paths):
  * ``energy.MeasurementConfig.launch_command()`` builds the sglang argv (model
    path, tp-size, rank-gpu-id, rank-tp-ratio, rank-gpu-memory-mib, kv-cache
    dtype, spec flags). ``LaunchSettings`` maps onto it.
  * ``energy._venv_cuda_lib_dirs()`` supplies the LD_LIBRARY_PATH fix so the
    re-exec'd rank-gpu-id / spec-draft worker pythons find the venv's bundled
    libcudart at interpreter startup.
  * ``start_new_session=True`` gives the child its OWN process group, exactly as
    ``EnergyHarness.boot()`` does, so teardown is scoped to that group and never
    touches a foreign PID.
  * GGUF header reading reuses ``uneven_perf._read_gguf_metadata``.

Teardown safety is the #1 correctness property. ``stop()`` signals the child's
process GROUP ONLY (``os.killpg(pgid, ...)`` with the pgid captured at start),
escalates SIGTERM -> SIGKILL, then waits for NVML VRAM to actually free on the
involved GPUs (the "orphaned-VRAM" guard). It NEVER broad-pkills and NEVER
signals a PID the supervisor did not spawn -- the box is shared with other live
servers.

------------------------------------------------------------------------------
CONTAINER PID-1 REQUIREMENT (for the LATER docker/webui phase -- DOCUMENTED, not
implemented here):

For the docker deployment the container ENTRYPOINT must become this supervisor /
dashboard process (PID 1), which then spawns sglang as a managed CHILD. Today
``docker/htsglang.*`` launches ``sglang.launch_server`` DIRECTLY as PID 1, so a
model restart would terminate the container. The docker phase must change the
entrypoint to run the dashboard (long-lived, PID 1) and let it own the sglang
lifecycle through this supervisor. When the dashboard is PID 1 it must also reap
zombies (the child is re-parented to it on exit); ``subprocess.Popen.poll()`` /
``wait()`` already reaps the direct child, which is sufficient for the single
managed instance here. This module deliberately does NOT implement the
entrypoint change -- it only flags it.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --- reuse the proven launcher / GGUF-reader plumbing ----------------------
from sglang.srt.planner.energy import MeasurementConfig, _venv_cuda_lib_dirs
from sglang.srt.uneven_perf import _read_gguf_metadata

# ---------------------------------------------------------------------------
# Default model roots scanned by discover_models().
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ROOTS: Tuple[str, ...] = (
    "/spinning/llm_stuff/club-3090/models-cache",
    "~/.cache/huggingface/hub",
)

#: llama.cpp ``LLAMA_FTYPE`` enum -> human quant tag (config-authoritative GGUF
#: quant, read from the GGUF header ``general.file_type``, NOT the file name).
#: Only the common subset; unknown values fall back to ``ftype<n>``.
_GGUF_FILE_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}

#: GGUF sidecars that live beside the text checkpoint but are NOT selectable
#: model quant variants (vision encoder, draft/MTP head, imatrix calibration).
_GGUF_SIDECAR_RE = re.compile(r"(?:^mmproj|^mtp[-_]|imatrix)", re.IGNORECASE)

#: Quant tag embedded in a GGUF file name (fallback / download-variant label
#: when the header is not read). Handles stock (``Q4_K_M``) and Unsloth
#: dynamic (``UD-Q4_K_XL``, ``IQ4_XS``) naming.
_GGUF_NAME_QUANT_RE = re.compile(
    r"(UD[-_])?(IQ\d[A-Z_]*\d*[A-Z]*|Q\d(?:_\d|_K(?:_[SML]|_XL)?)?|BF16|F16|F32)",
    re.IGNORECASE,
)


# ===========================================================================
# Model discovery.
# ===========================================================================
@dataclasses.dataclass
class GgufVariant:
    """One selectable GGUF quant file inside a model dir."""

    quant: str            # config-authoritative header quant (or name-parsed)
    filename: str         # basename of the .gguf file
    path: str             # absolute path to the .gguf file
    size_bytes: int = 0


@dataclasses.dataclass
class DiscoveredModel:
    """One local model the dashboard can serve."""

    name: str                       # display name (dir/repo basename)
    path: str                       # absolute dir (hf/gguf dir) or snapshot dir
    format: str                     # "hf" | "gguf"
    quant_method: str               # config-authoritative scheme label
    gguf_variants: List[GgufVariant] = dataclasses.field(default_factory=list)
    vision: bool = False
    size_bytes: int = 0
    error: Optional[str] = None     # set (not raised) when the entry is broken


def _gguf_quant_from_name(filename: str) -> str:
    m = _GGUF_NAME_QUANT_RE.search(os.path.basename(filename))
    if not m:
        return "gguf"
    tag = m.group(0).upper().replace("UD-", "").replace("UD_", "")
    return tag


def _gguf_quant_from_header(path: str) -> Optional[str]:
    """Config-authoritative GGUF quant: read ``general.file_type`` from the
    header (dependency-free reader reused from uneven_perf). Returns None when
    the header can't be read (caller falls back to the name)."""
    try:
        scalars, _arrays, _tensors = _read_gguf_metadata(path)
    except Exception:
        return None
    ft = scalars.get("general.file_type")
    if ft is None:
        return None
    try:
        return _GGUF_FILE_TYPE_NAMES.get(int(ft), f"ftype{int(ft)}")
    except Exception:
        return None


def _selectable_gguf_files(dir_path: str) -> List[str]:
    out = []
    try:
        names = os.listdir(dir_path)
    except OSError:
        return out
    for n in sorted(names):
        if n.lower().endswith(".gguf") and not _GGUF_SIDECAR_RE.search(n):
            out.append(os.path.join(dir_path, n))
    return out


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _hf_quant_method(cfg: dict) -> str:
    """Config-AUTHORITATIVE HF quant scheme label -- reads the same
    ``quantization_config.quant_method`` field the planner's cost model
    (``uneven_perf._config_quant_bpp`` / ``roofline._infer_compute``) treats as
    authoritative, honoring the ``text_config`` nesting used by VL checkpoints.
    Never inferred from the directory name. Returns "bf16" for an un-quantized
    checkpoint."""
    qc = cfg.get("quantization_config")
    if not qc:
        text = cfg.get("text_config")
        if isinstance(text, dict):
            qc = text.get("quantization_config")
    if not qc:
        return "bf16"
    method = str(qc.get("quant_method") or "").lower()
    if method:
        return method
    fmt = str(qc.get("format") or qc.get("fmt") or "").lower()
    if "fp8" in fmt or "float8" in fmt:
        return "fp8"
    return "quantized"


def _config_is_vision(cfg: dict) -> bool:
    """Vision-capable iff the checkpoint actually declares a vision encoder
    (``vision_config`` / ``visual_config`` / ``vision_tower``, possibly nested
    under ``text_config``) or its architecture name is explicitly a VL/vision
    model. NOT inferred from the broad ``*ForConditionalGeneration`` suffix,
    which text-only Gemma/Qwen checkpoints also carry."""
    keys = ("vision_config", "visual_config", "vision_tower")
    if any(cfg.get(k) for k in keys):
        return True
    if isinstance(cfg.get("architectures"), list):
        if any("vl" in str(a).lower() or "vision" in str(a).lower()
               for a in cfg["architectures"]):
            return True
    return False


def _dir_size_bytes(dir_path: str, exts: Optional[Tuple[str, ...]] = None) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(dir_path):
            for f in files:
                if exts and not f.lower().endswith(exts):
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _resolve_hf_hub_snapshot(entry_dir: str) -> Optional[str]:
    """``models--org--name`` HF-hub cache dir -> its current snapshot dir."""
    snaps = os.path.join(entry_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    candidates = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None
    # Newest snapshot (mtime) -- the ref the hub last resolved.
    return max(candidates, key=lambda c: os.path.getmtime(c))


def _classify_model_dir(name: str, path: str) -> Optional[DiscoveredModel]:
    """Turn a single directory into a DiscoveredModel, or None if it is not a
    model dir. Never raises -- a broken model is returned with ``.error`` set."""
    try:
        gguf_files = _selectable_gguf_files(path)
        cfg_path = os.path.join(path, "config.json")
        has_cfg = os.path.isfile(cfg_path)

        if gguf_files:
            variants: List[GgufVariant] = []
            for gf in gguf_files:
                q = _gguf_quant_from_header(gf) or _gguf_quant_from_name(gf)
                try:
                    sz = os.path.getsize(gf)
                except OSError:
                    sz = 0
                variants.append(GgufVariant(
                    quant=q, filename=os.path.basename(gf), path=gf, size_bytes=sz))
            # A GGUF dir may still ship a config.json (tokenizer/vision meta);
            # use it for vision detection only -- the quant is header-derived.
            vision = False
            cfg = _read_json(cfg_path) if has_cfg else None
            if cfg:
                vision = _config_is_vision(cfg)
            # vision sidecar present -> multimodal GGUF.
            if any(_GGUF_SIDECAR_RE.match(n) and n.lower().startswith("mmproj")
                   for n in os.listdir(path)):
                vision = True
            quant_label = variants[0].quant if len(variants) == 1 else "gguf"
            return DiscoveredModel(
                name=name, path=path, format="gguf", quant_method=quant_label,
                gguf_variants=variants, vision=vision,
                size_bytes=sum(v.size_bytes for v in variants))

        if has_cfg:
            cfg = _read_json(cfg_path)
            if cfg is None:
                return DiscoveredModel(
                    name=name, path=path, format="hf", quant_method="unknown",
                    error=f"unreadable config.json in {path}")
            return DiscoveredModel(
                name=name, path=path, format="hf",
                quant_method=_hf_quant_method(cfg),
                vision=_config_is_vision(cfg),
                size_bytes=_dir_size_bytes(path, (".safetensors", ".bin")))
    except Exception as e:  # never throw on one bad model
        return DiscoveredModel(
            name=name, path=path, format="hf", quant_method="unknown",
            error=f"{type(e).__name__}: {e}")
    return None


def discover_models(
    roots: Optional[Sequence[str]] = None,
    extra_roots: Optional[Sequence[str]] = None,
    max_depth: int = 1,
) -> List[DiscoveredModel]:
    """Scan the default model roots (plus any user-supplied extra roots) and
    return every local model found. Config-authoritative quant detection; robust
    to missing/partial dirs (a broken model is TAGGED via ``.error``, never
    thrown). HF-hub ``models--org--name`` dirs are resolved to their snapshot.
    ``max_depth`` allows descending one level into org-style subdirs
    (``unsloth/<model>``)."""
    root_list: List[str] = []
    for r in (list(roots) if roots is not None else list(DEFAULT_MODEL_ROOTS)):
        root_list.append(os.path.expanduser(r))
    for r in (extra_roots or []):
        root_list.append(os.path.expanduser(r))

    found: List[DiscoveredModel] = []
    seen_paths: set = set()

    def scan(dir_path: str, depth: int):
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return
        for name in entries:
            entry = os.path.join(dir_path, name)
            if not os.path.isdir(entry):
                continue
            real = os.path.realpath(entry)

            # HF-hub cache layout: models--org--name -> snapshot dir.
            if name.startswith("models--"):
                snap = _resolve_hf_hub_snapshot(entry)
                if snap and os.path.realpath(snap) not in seen_paths:
                    disp = name[len("models--"):].replace("--", "/")
                    m = _classify_model_dir(disp, snap)
                    if m is not None:
                        seen_paths.add(os.path.realpath(snap))
                        found.append(m)
                continue

            if real in seen_paths:
                continue
            m = _classify_model_dir(name, entry)
            if m is not None:
                seen_paths.add(real)
                found.append(m)
            elif depth < max_depth:
                # Not a model dir itself -> maybe an org dir of sub-models.
                scan(entry, depth + 1)

    for root in root_list:
        if os.path.isdir(root):
            scan(root, 0)
    return found


# ===========================================================================
# Launch settings -> sglang argv (reuses energy.MeasurementConfig builder).
# ===========================================================================
@dataclasses.dataclass
class LaunchSettings:
    """The knobs the dashboard sets for one server boot. ``validate()`` fails
    fast on invalid mappings; ``launch_command()`` produces the sglang argv by
    reusing ``energy.MeasurementConfig.launch_command()`` for the overlapping
    core and appending the UI-only flags."""

    model_path: str
    format: str = "hf"                       # "hf" | "gguf"
    gguf_variant: Optional[str] = None       # chosen .gguf file (path/basename)
    served_model_name: str = "model"
    tp_size: int = 1
    rank_gpu_id: Optional[List[int]] = None
    rank_tp_ratio: Optional[List[int]] = None
    rank_gpu_memory_mib: Optional[List[int]] = None
    kv_cache_dtype: str = "auto"
    mem_fraction_static: Optional[float] = None
    context_length: int = 8192
    max_running_requests: int = 16
    max_num_seqs: Optional[int] = None
    # spec: "off" | "mtp" | "adaptive"
    spec_mode: str = "off"
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    chat_template: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    vision: bool = False
    host: str = "127.0.0.1"
    port: int = 30000
    trust_remote_code: bool = True
    extra_flags: List[str] = dataclasses.field(default_factory=list)
    python_exe: str = sys.executable

    # -- validation ------------------------------------------------------
    def validate(self) -> "LaunchSettings":
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1 (got {self.tp_size})")
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"invalid port {self.port}")
        if self.rank_gpu_id is not None and len(self.rank_gpu_id) != self.tp_size:
            raise ValueError(
                f"--rank-gpu-id length {len(self.rank_gpu_id)} != tp_size "
                f"{self.tp_size}")
        if (self.rank_tp_ratio is not None
                and len(self.rank_tp_ratio) != self.tp_size):
            raise ValueError(
                f"--rank-tp-ratio length {len(self.rank_tp_ratio)} != tp_size "
                f"{self.tp_size}")
        if (self.rank_gpu_memory_mib is not None
                and len(self.rank_gpu_memory_mib) != self.tp_size):
            raise ValueError(
                f"--rank-gpu-memory-mib length {len(self.rank_gpu_memory_mib)} "
                f"!= tp_size {self.tp_size}")
        if self.spec_mode not in ("off", "mtp", "adaptive"):
            raise ValueError(f"unknown spec_mode {self.spec_mode!r}")
        if self.format not in ("hf", "gguf"):
            raise ValueError(f"unknown format {self.format!r}")
        return self

    def resolved_model_path(self) -> str:
        """The concrete ``--model-path`` value: for a GGUF model the chosen
        variant .gguf FILE, otherwise the model dir."""
        if self.format == "gguf" and self.gguf_variant:
            if os.path.isabs(self.gguf_variant) or os.path.sep in self.gguf_variant:
                return self.gguf_variant
            return os.path.join(self.model_path, self.gguf_variant)
        return self.model_path

    def _to_measurement_config(self) -> MeasurementConfig:
        """Map onto the proven argv builder. Spec + the UI-only flags that
        MeasurementConfig doesn't model are appended via ``extra_flags``."""
        extra: List[str] = []
        if self.mem_fraction_static is not None:
            extra += ["--mem-fraction-static", str(self.mem_fraction_static)]
        if self.max_num_seqs is not None:
            extra += ["--max-running-requests", str(self.max_num_seqs)]
        if self.chat_template:
            extra += ["--chat-template", self.chat_template]
        if self.tool_call_parser:
            extra += ["--tool-call-parser", self.tool_call_parser]
        if self.reasoning_parser:
            extra += ["--reasoning-parser", self.reasoning_parser]
        extra += list(self.extra_flags)

        spec_algo = None
        spec_steps = self.speculative_num_steps
        spec_adaptive = False
        if self.spec_mode == "mtp":
            spec_algo = "NEXTN"
        elif self.spec_mode == "adaptive":
            spec_algo = "NEXTN"
            spec_adaptive = True

        return MeasurementConfig(
            model_path=self.resolved_model_path(),
            served_model_name=self.served_model_name,
            tp_size=self.tp_size,
            rank_gpu_id=self.rank_gpu_id,
            rank_tp_ratio=self.rank_tp_ratio,
            rank_gpu_memory_mib=self.rank_gpu_memory_mib,
            kv_cache_dtype=self.kv_cache_dtype,
            context_length=self.context_length,
            max_running_requests=self.max_running_requests,
            host=self.host,
            port=self.port,
            trust_remote_code=self.trust_remote_code,
            speculative_algorithm=spec_algo,
            speculative_num_steps=spec_steps,
            speculative_eagle_topk=self.speculative_eagle_topk,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            speculative_adaptive=spec_adaptive,
            disable_cuda_graph=False,
            python_exe=self.python_exe,
            extra_flags=extra,
        )

    def launch_command(self) -> List[str]:
        self.validate()
        return self._to_measurement_config().launch_command()


# ===========================================================================
# The supervisor.
# ===========================================================================
_STATE_STOPPED = "stopped"
_STATE_BOOTING = "booting"
_STATE_READY = "ready"
_STATE_ERROR = "error"


class SupervisorBusyError(RuntimeError):
    """Raised when a restart is requested while a live job is in-flight."""


class SglangSupervisor:
    """Owns ONE managed sglang child (the box is VRAM-bound; a restart REPLACES
    the running model). The supervisor process itself is long-lived and stays up
    across restarts -- that is the whole point: the dashboard can swap models
    without exiting.

    Teardown is scoped to the child's process GROUP (captured at start), never a
    broad pkill and never a foreign PID. NVML is injectable (``nvml=``) so the
    VRAM-free guard is unit-testable on a GPU-less box.
    """

    def __init__(
        self,
        log_dir: str = "/tmp",
        nvml=None,
        readiness_path: str = "/get_model_info",
    ):
        self.log_dir = log_dir
        self._nvml = nvml  # injectable pynvml-like; None -> lazy import on use
        self._readiness_path = readiness_path

        self.proc: Optional[subprocess.Popen] = None
        self.settings: Optional[LaunchSettings] = None
        self.state: str = _STATE_STOPPED
        self._pgid: Optional[int] = None
        self._start_time: Optional[float] = None
        self._boot_deadline: Optional[float] = None
        self._log_path: Optional[str] = None
        self._log_file = None
        self._busy: bool = False
        self._error_detail: Optional[str] = None
        self._baseline_free_mib: Optional[Dict[int, int]] = None

    # -- busy guard ------------------------------------------------------
    @property
    def is_busy(self) -> bool:
        return self._busy

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)

    # -- process state ---------------------------------------------------
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _refresh_state(self) -> None:
        """Detect an out-from-under-us child death and surface it as error
        (never wedge)."""
        if self.proc is None:
            return
        if self.proc.poll() is not None and self.state in (
            _STATE_BOOTING, _STATE_READY,
        ):
            self.state = _STATE_ERROR
            self._error_detail = (
                f"child exited unexpectedly (rc={self.proc.returncode})")

    # -- env (LD_LIBRARY_PATH fix reused from energy) --------------------
    def _build_env(self, settings: LaunchSettings) -> Dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "/spinning/wt-integration-r2/python")
        ld = _venv_cuda_lib_dirs(settings.python_exe)
        if ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                ld + ([existing] if existing else []))
        return env

    # -- NVML helpers (injectable) --------------------------------------
    def _nvml_mod(self):
        if self._nvml is not None:
            return self._nvml
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None
        return self._nvml

    def _involved_gpu_indices(self, settings: Optional[LaunchSettings]) -> List[int]:
        if settings is not None and settings.rank_gpu_id:
            return sorted(set(settings.rank_gpu_id))
        return []

    def _free_mib(self, indices: Sequence[int]) -> Dict[int, int]:
        nvml = self._nvml_mod()
        out: Dict[int, int] = {}
        if nvml is None:
            return out
        try:
            n = nvml.nvmlDeviceGetCount()
            idxs = list(indices) if indices else list(range(n))
            for i in idxs:
                if 0 <= i < n:
                    h = nvml.nvmlDeviceGetHandleByIndex(i)
                    out[i] = int(nvml.nvmlDeviceGetMemoryInfo(h).free / 2 ** 20)
        except Exception:
            return out
        return out

    # -- lifecycle -------------------------------------------------------
    def start(
        self,
        settings: LaunchSettings,
        argv: Optional[List[str]] = None,
        wait_ready: bool = True,
        ready_timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
    ) -> dict:
        """Boot the managed server in its own process group. ``argv`` overrides
        the built command (used by tests with a FAKE child -- NEVER a real
        sglang boot in the unit tests). ``wait_ready`` polls
        ``/get_model_info`` until ready or timeout."""
        if self.is_running():
            raise RuntimeError("a server is already running; stop() first")
        settings.validate()
        self.settings = settings
        cmd = argv if argv is not None else settings.launch_command()
        env = self._build_env(settings)

        # Capture pre-boot free VRAM on the involved GPUs so stop() can verify
        # the child's allocation actually came back (orphaned-VRAM guard).
        self._baseline_free_mib = self._free_mib(
            self._involved_gpu_indices(settings)) or None

        self._log_path = os.path.join(
            self.log_dir, f"sglang_boot_{settings.port}.log")
        self._log_file = open(self._log_path, "w")
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=self._log_file, stderr=subprocess.STDOUT,
            start_new_session=True,  # OWN process group -> scoped teardown
        )
        # Capture the pgid NOW so teardown targets the group even if the child
        # later dies and its pid is reused.
        self._pgid = os.getpgid(self.proc.pid)
        self._start_time = time.time()
        self.state = _STATE_BOOTING
        self._error_detail = None
        # Deadline for the PASSIVE (poll-driven) readiness path below, so a boot
        # that never comes up eventually flips to error instead of BOOTING
        # forever when wait_ready=False.
        self._boot_deadline = time.time() + ready_timeout_s

        if wait_ready:
            self._wait_ready(ready_timeout_s, poll_interval_s)
        return self.status()

    def _poll_ready_if_booting(self) -> None:
        """Passive readiness probe for the poll-driven UI. ``start(wait_ready=
        False)`` returns immediately in BOOTING so the dashboard stays
        responsive; each ``status()`` poll then cheaply probes the server and
        flips BOOTING -> READY once it answers. Without this the state would
        never leave BOOTING even though the server is up and serving."""
        if self.state != _STATE_BOOTING or not self.is_running() or self.settings is None:
            return
        try:
            with urllib.request.urlopen(
                self._base_url() + self._readiness_path, timeout=1
            ) as r:
                if r.status == 200:
                    self.state = _STATE_READY
                    return
        except Exception:
            pass
        if self._boot_deadline is not None and time.time() > self._boot_deadline:
            self.state = _STATE_ERROR
            self._error_detail = (
                "server did not become ready before the boot deadline "
                f"({self._readiness_path} never returned 200); see log tail")

    def _base_url(self) -> str:
        s = self.settings
        return f"http://{s.host}:{s.port}"

    def _wait_ready(self, timeout_s: float, poll_interval_s: float) -> None:
        deadline = time.time() + timeout_s
        url = self._base_url() + self._readiness_path
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.state = _STATE_ERROR
                self._error_detail = (
                    f"server exited during boot (rc={self.proc.returncode})")
                raise RuntimeError(
                    f"{self._error_detail}\nlog tail:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        self.state = _STATE_READY
                        return
            except Exception:
                pass
            time.sleep(poll_interval_s)
        self.state = _STATE_ERROR
        self._error_detail = f"server not ready after {timeout_s:.0f}s"
        raise TimeoutError(
            f"{self._error_detail}\nlog tail:\n{self.log_tail()}\n"
            f"{self._diagnose_hang()}")

    def _diagnose_hang(self) -> str:
        """``pgrep -g <pgid>`` scoped process listing (same discipline as
        ``EnergyHarness._diagnose_hang``) -- only the child's own group, never
        a system-wide scan."""
        if self._pgid is None:
            return "(no pgid)"
        try:
            pids = subprocess.check_output(
                ["pgrep", "-g", str(self._pgid)], text=True).split()
            return f"process group {self._pgid} pids: {' '.join(pids)}"
        except Exception:
            return f"process group {self._pgid}: (no live pids)"

    def stop(
        self,
        grace_s: float = 20.0,
        wait_vram: bool = True,
        vram_timeout_s: float = 30.0,
    ) -> dict:
        """Tear the managed server down, scoped to its process GROUP only.

        SIGTERM the group, wait ``grace_s``, then SIGKILL the group if still
        alive. Then (``wait_vram``) poll NVML free-mem on the involved GPUs until
        it recovers toward the pre-boot baseline, reporting if it did not (the
        orphaned-VRAM guard). NEVER a broad pkill, NEVER a PID we didn't spawn.
        """
        report: Dict[str, object] = {"vram_recovered": None}
        if self.proc is None or self._pgid is None:
            self.state = _STATE_STOPPED
            return report

        if self.proc.poll() is None:
            self._killpg(signal.SIGTERM)
            self._wait_exit(grace_s)
            if self.proc.poll() is None:
                self._killpg(signal.SIGKILL)
                self._wait_exit(10.0)
            report["forced_sigkill"] = self.proc.poll() is None

        # Confirm the whole group is gone (no lingering worker holds VRAM).
        report["group_gone"] = not self._pgid_has_live_pids()

        if wait_vram:
            report["vram_recovered"] = self._wait_vram_free(vram_timeout_s)

        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self.state = _STATE_STOPPED
        return report

    def _killpg(self, sig: int) -> None:
        """Signal the child's process GROUP only. ProcessLookupError (group
        already gone) is benign."""
        try:
            os.killpg(self._pgid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    def _pgid_has_live_pids(self) -> bool:
        try:
            os.killpg(self._pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Group exists but is owned by another uid -> not ours; treat as
            # gone for OUR bookkeeping (we never signal a foreign group again).
            return False
        except Exception:
            return False

    def _wait_exit(self, timeout_s: float) -> None:
        end = time.time() + timeout_s
        while time.time() < end and self.proc.poll() is None:
            time.sleep(0.2)

    def _wait_vram_free(self, timeout_s: float) -> Optional[bool]:
        """Poll NVML free-mem on the involved GPUs until it recovers to (within
        3% of) the captured pre-boot baseline, or timeout. Returns True/False,
        or None when NVML/baseline is unavailable (can't judge)."""
        indices = self._involved_gpu_indices(self.settings)
        if not indices or not self._baseline_free_mib:
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            cur = self._free_mib(indices)
            if not cur:
                return None
            if all(cur.get(i, 0) >= 0.97 * self._baseline_free_mib.get(i, 0)
                   for i in indices):
                return True
            time.sleep(0.5)
        return False

    def restart(self, new_settings: LaunchSettings, **start_kwargs) -> dict:
        """Guarded restart: refuse while a live job is in-flight (``is_busy``),
        else stop() the current child and start() the new one. The supervisor
        process stays alive across this."""
        if self.is_busy:
            raise SupervisorBusyError(
                "refusing restart: a live job is in-flight (is_busy). Clear the "
                "busy flag when the job completes, then retry.")
        self.stop(wait_vram=start_kwargs.pop("wait_vram_on_stop", True))
        return self.start(new_settings, **start_kwargs)

    # -- introspection ---------------------------------------------------
    def log_tail(self, n: int = 60) -> str:
        if not self._log_path:
            return "(no log)"
        try:
            with open(self._log_path) as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return "(no log)"

    def status(self) -> dict:
        self._refresh_state()
        self._poll_ready_if_booting()
        pid = self.proc.pid if self.proc is not None else None
        uptime = (time.time() - self._start_time
                  if self._start_time is not None and self.is_running() else None)
        s = self.settings
        return {
            "state": self.state,
            "pid": pid,
            "pgid": self._pgid,
            "busy": self._busy,
            "model_path": s.resolved_model_path() if s else None,
            "served_model_name": s.served_model_name if s else None,
            "port": s.port if s else None,
            "uptime_s": uptime,
            "error": self._error_detail,
            "log_tail": self.log_tail(20),
        }


# ===========================================================================
# Model download (gated on write access).
# ===========================================================================
@dataclasses.dataclass
class DownloadInfo:
    """What ``available_downloads`` returns for a HF repo."""

    repo_id: str
    files: List[str]
    is_gguf: bool
    gguf_variants: List[GgufVariant] = dataclasses.field(default_factory=list)


def model_root_writable(root: str) -> bool:
    """True only if ``root`` exists AND we can actually create a file in it.

    A mounted volume can report ``os.access(..., W_OK) == True`` and still fail
    the real write (read-only remount, overlay quirks), so we probe with an
    actual temp file. The UI disables the download button + annotates
    "mount read-only -- remount rw to enable" when this is False. Granting write
    access (mounting rw) is the user's decision; the code never forces it.
    """
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return False
    if not os.access(root, os.W_OK):
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".wtest_"):
            pass
        return True
    except OSError:
        return False


def _gguf_download_variants(files: Sequence[str]) -> List[GgufVariant]:
    out: List[GgufVariant] = []
    for f in files:
        base = os.path.basename(f)
        if not base.lower().endswith(".gguf"):
            continue
        if _GGUF_SIDECAR_RE.search(base):
            continue
        out.append(GgufVariant(
            quant=_gguf_quant_from_name(base), filename=f, path=f))
    return out


def available_downloads(repo_id: str, hf_api=None) -> DownloadInfo:
    """List a HF repo's files and, for a multi-quant GGUF repo, the selectable
    quant variants (for the UI dropdown). Uses
    ``huggingface_hub.HfApi().list_repo_files`` (no file bytes fetched).
    ``hf_api`` is injectable for tests."""
    if hf_api is None:
        from huggingface_hub import HfApi  # verified importable (sglang dep)

        hf_api = HfApi()
    files = list(hf_api.list_repo_files(repo_id))
    gguf_variants = _gguf_download_variants(files)
    return DownloadInfo(
        repo_id=repo_id, files=files,
        is_gguf=bool(gguf_variants), gguf_variants=gguf_variants)


def _quant_to_filename(quant: str, gguf_variants: Sequence[GgufVariant]) -> str:
    """Map a chosen quant tag (``Q4_K_M``) onto the exact repo file to fetch --
    the "allow pattern" for a GGUF download is precisely this one file, so the
    whole repo is NOT pulled."""
    q = quant.upper()
    exact = [v for v in gguf_variants if v.quant.upper() == q]
    if exact:
        return exact[0].filename
    loose = [v for v in gguf_variants if q in v.filename.upper()]
    if loose:
        return loose[0].filename
    raise ValueError(
        f"quant {quant!r} not among available GGUF variants: "
        f"{[v.quant for v in gguf_variants]}")


def download_model(
    repo_id: str,
    quant: Optional[str] = None,
    root: Optional[str] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    hf_api=None,
    hf_hub_download=None,
    snapshot_download=None,
) -> str:
    """Download ``repo_id`` into the mounted model ``root`` -- ONLY when the root
    is writable (raises ``PermissionError`` otherwise; the UI shows the
    disabled/annotated state instead).

    For a GGUF ``quant`` the ONLY chosen quant file is fetched via
    ``hf_hub_download`` (allow-pattern == that single file) -- the whole repo is
    never pulled. For a full HF-format (quantized) repo the snapshot is fetched
    via ``snapshot_download``. Progress is streamed to ``progress_cb``.

    Downloading pulls from an EXTERNAL service (Hugging Face) on the user's
    explicit action; the UI must confirm (with a size estimate) BEFORE calling
    this. The injectable ``hf_hub_download`` / ``snapshot_download`` /
    ``hf_api`` keep the unit test network-free. Returns the local path.
    """
    if root is None:
        raise ValueError("root (mounted model dir) is required for download")
    root = os.path.expanduser(root)
    if not model_root_writable(root):
        raise PermissionError(
            f"model root {root!r} is not writable -- the mount is read-only. "
            f"Remount it rw to enable downloads; the dashboard will not force "
            f"write access.")

    if progress_cb is not None:
        progress_cb({"stage": "start", "repo_id": repo_id, "quant": quant})

    if quant is not None:
        # GGUF single-quant download: resolve quant -> the one file, fetch only
        # it. allow_patterns is exactly this file (no whole-repo pull).
        info = available_downloads(repo_id, hf_api=hf_api)
        if not info.gguf_variants:
            raise ValueError(
                f"quant {quant!r} requested but {repo_id} exposes no GGUF "
                f"quant files")
        filename = _quant_to_filename(quant, info.gguf_variants)
        if hf_hub_download is None:
            from huggingface_hub import hf_hub_download as hf_hub_download
        local = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=root)
        if progress_cb is not None:
            progress_cb({"stage": "done", "path": local, "filename": filename})
        return local

    # Full HF-format snapshot.
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as snapshot_download
    local = snapshot_download(repo_id=repo_id, local_dir=root)
    if progress_cb is not None:
        progress_cb({"stage": "done", "path": local})
    return local
