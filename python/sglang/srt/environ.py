import functools
import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str):
        super().__init__(default)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class EnvFloatVector(EnvField):
    """A float that may also be given per rank as a comma-list.

    ``"0.45"`` parses to the float ``0.45`` -- indistinguishable from
    :class:`EnvFloat`, so every existing reader and the default path are
    byte-identical. ``"0.485,0.42,0.42"`` parses to a tuple, one entry per
    tensor-parallel rank.

    ``get()`` deliberately REFUSES to answer once a vector is set. The value
    has a dozen readers, and a reader that has not been taught about ranks
    would otherwise compare a tuple against a float and either raise something
    obscure or, worse, silently size a buffer from the wrong number. Failing
    here names the sanctioned accessor instead. Use :meth:`get_vector` (or the
    helpers in ``sglang.srt.layers.moe.resident_fraction``) to read it.
    """

    def parse(self, value: str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{self.name} is empty")
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                raise ValueError(f'"{p}" is not a valid float value in {self.name}')
        return out[0] if len(out) == 1 else tuple(out)

    def get(self):
        value = super().get()
        if isinstance(value, tuple):
            raise RuntimeError(
                f"{self.name} is set per rank ({','.join(str(v) for v in value)}), "
                f"so there is no single value to return. Read it through "
                f"sglang.srt.layers.moe.resident_fraction: "
                f"resident_fraction_for_rank(rank) for anything that sizes or "
                f"books memory, offload_active() for a plain is-offload-on check."
            )
        return value

    def get_vector(self) -> tuple:
        """The sanctioned reader: always a tuple, length 1 when scalar."""
        value = super().get()
        return value if isinstance(value, tuple) else (value,)


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # Raise on bare server_args field assignments after resolution; mutation
    # must go through ServerArgs.override() (enabled by the test harness).
    SGLANG_STRICT_CONFIG_MUTATION = EnvBool(False)

    # Downgrade the draft-model unloaded-parameter check (#290/#318) from a
    # hard error to a log line. An unloaded drafter proposes noise, so this is
    # a debugging escape hatch, not a supported configuration.
    SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS = EnvBool(False)

    # #695: allocate the permanent phase-flip host weight images at their exact
    # size (MAP_ANONYMOUS + cudaHostRegister) instead of through torch's pinned
    # caching allocator, which rounds every request up to a power of two and
    # held 13.65 GiB of pure rounding for the life of the process.
    # Set to 0 to restore the pre-#695 allocation. That is the comparand arm of
    # the flip-latency A/B (MERGE-R5 §6) and the opt-out if the exact-size path
    # ever has to be taken off by default without reverting the commit. It does
    # NOT disable the host-post registration or the shmem pricing from the same
    # commit -- those are correct under either allocator.
    SGLANG_PHASE_FLIP_EXACT_PIN = EnvBool(True)
    # Flip host images as FILE-BACKED shared mappings instead of page-locked
    # RAM: the pages become reclaimable page cache (written back to disk under
    # pressure, refaulted at the next flip) and the ~tens-of-GiB image post
    # leaves the pinned-host ledger. Opt-in: the flip's H2D refill then runs
    # pageable instead of DMA, plus a disk refault when the box was actually
    # under pressure (#89's hibernate restore, 8-14 s for a full weight set on
    # the same pool, is the cold-read anchor; #690 measured the pinned DMA
    # refill at 9,614.9 MiB/rank). Requires SGLANG_PHASE_FLIP_IMAGE_DIR on a
    # persistent filesystem; refuses (never silently pins) otherwise.
    SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED = EnvBool(False)
    SGLANG_PHASE_FLIP_IMAGE_DIR = EnvStr("")
    # #802: refill a FILE-BACKED image by READING the file into a pinned
    # staging ring, instead of copying straight off the mapping and taking one
    # synchronous major fault per 4 KiB page. Measured on this rig 2026-08-22
    # for the 16 699 408 904-byte PP0 image: the mapping path costs 4 077 045
    # faults and 12 572 ms (1266 MiB/s) on a pool that writes at ~3500 MiB/s.
    # Advisory hints do NOT fix it here and this arm deliberately does not use
    # them -- on this OpenZFS pool MADV_WILLNEED populates nothing (12 564 ms,
    # 4 077 052 faults, mincore residency 0.0 after the call) and per-chunk
    # MADV_SEQUENTIAL is a 15.6x regression (196 200 ms).
    # Only affects the file-backed arm; the default pinned image path never
    # reaches it. Set to 0 for the comparand arm of the A/B on one binary.
    SGLANG_PHASE_FLIP_REFILL_STAGED = EnvBool(True)
    # Staging chunk and ring depth. The ring is allocated ONCE and charged to
    # the pinned-host registry (#720's ReadBufferPool), so the whole new host
    # post is CHUNK_MIB x DEPTH per rank -- bounded, unlike the image itself.
    # 32 MiB x 2 measured fastest of the sweep at 1 918 ms / 8 304 MiB/s, a
    # 7.50x improvement on the 14 377 ms mapping baseline. Buffered reads of
    # the same shape reach only 2 242 MiB/s because they pay a second pass
    # into the ARC, so the read path prefers O_DIRECT and falls back to
    # buffered only when the filesystem refuses it.
    SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB = EnvInt(32)
    SGLANG_PHASE_FLIP_REFILL_DEPTH = EnvInt(2)

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)

    # HTTP server
    # Decompress request bodies tagged with `x-body-compressed`.
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    # Override parsed request fields from headers.
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    # #540: what the Anthropic front sends downstream for output_config.effort
    # == "xhigh". Default "xhigh" = pass the client's value through unchanged,
    # which is what the Qwen3.8 family's chat template accepts ('xhigh',
    # 'medium', 'low' -- anything else raises). Set to "max" to restore the
    # pre-fix collapse for a deployment whose template names its top tier
    # "max" instead; the collapse is then logged by name.
    SGLANG_ANTHROPIC_XHIGH_EFFORT = EnvStr("xhigh")
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # IPC
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_ENABLE_CP_V2 = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_NVTX"
    )
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPERATIONS_ENABLE_PROFILE"
    )
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FAST_KERNEL = EnvBool(True)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    SGLANG_DSPARK_FAST_SAMPLING = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    # Disable lazy compaction in the unified memory pool allocator and
    # fall back to the per-free eager compaction. Used for production
    # A/B and quick rollback. Default False (lazy compaction on).
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    # Sort the multi-ended allocator's free list after a merge (perf A/B knob).
    SGLANG_SORT_FREE_LIST_AFTER_MERGE = EnvBool(False)
    # Periodically log lazy-compaction stats per sub-pool (observability only).
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)

    # HND KV layout folds (page, head) into one paged index for per-kv-head sparse
    # page tables (DP attn); paged backends like trtllm_mha consume it directly.
    SGLANG_USE_HND_KVCACHE = EnvBool(False)

    # size the KV pool after CUDA-graph capture
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)

    # #552 env twin of --kv-session-offload-resume-under-spec: let a spilled
    # session under speculative decoding wave back and rejoin the LIVE spec
    # decode batch instead of finishing on host. Either source turning it on
    # is enough (the flag ORs them), so a boot-matrix arm can arm it without
    # rewriting its command line. The bare `KVSO_RESUME` spelling predates the
    # flag and stays as a deprecated alias so existing arms and tickets keep
    # working. Default OFF is a named decision -- see the flag's help text.
    SGLANG_KVSO_RESUME = EnvBoolWithAlias(False, deprecated_name="KVSO_RESUME")

    # #330 --enable-vram-dial: physical commit chunk of the VMM-backed KV
    # pool in MiB. Smaller chunks = finer dial-down release granularity but
    # more driver handles (boot maps pool_bytes / chunk handles per rank).
    SGLANG_VRAM_DIAL_CHUNK_MIB = EnvInt(16)

    # Measured KV-budget correction (two-boot convergence): after load +
    # capture each rank measures its ACTUAL leftover GPU memory and persists
    # `leftover - safety` per rank (config-fingerprinted cache); the next
    # boot adds that correction to the heuristic KV budget, replacing the
    # blind mem-fraction slack with a measured remainder. Fixed-point: once
    # leftover ~= safety the correction stops moving.
    SGLANG_MEASURED_KV_BUDGET = EnvBool(False)
    # Scalar MiB or a comma list with one value per TP rank (roles differ:
    # the draft-solo host carries prompt-length-scaled serving transients).
    SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB = EnvStr("400")
    # #188: how many MiB of this rank's device share may be used by things
    # outside its own allocator reservation (CUDA context, NCCL buffers)
    # before the leftover measurement is reported as contaminated by a
    # FOREIGN consumer -- e.g. a server from the previous boot that never
    # exited, which silently shrinks the persisted correction. Raise this on
    # a device that legitimately hosts a co-resident non-sglang consumer.
    SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB = EnvInt(1024)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)
    # Scheduler: force lazy extra_buffer prealloc to fail at decode boundaries
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    # --mamba-checkpoint-interval: how many of the deepest on-grid mamba
    # checkpoints per radix path evict_mamba keeps live (best effort; a
    # second eviction pass ignores the window when the pool must yield).
    SGLANG_MAMBA_CKPT_WINDOW = EnvInt(2)
    # --mamba-checkpoint-interval: resume only at the DEEPEST interval
    # boundary of the full-KV match (else recompute from 0) instead of the
    # deepest surviving on-grid checkpoint.
    SGLANG_MAMBA_CKPT_STRICT_RESUME = EnvBool(False)
    # Per-request mamba checkpoint diagnostics: log match length, resume
    # length, checkpoint node/slot and cache-insert positions so a
    # nondeterministic resume (or a checkpoint at a wrong position) can be
    # attributed from server logs.
    SGLANG_MAMBA_CKPT_DEBUG = EnvBool(False)
    # #581 mamba pin trace: emit one line per rank every N scheduler ticks
    # with the transfer-queue depths, the outstanding write-through and
    # load-back pin counts, the protected/evictable sizes, and the
    # inc/dec_lock_ref traffic per call site since the previous line. 0 = off
    # (default; the traced path is not entered at all).
    SGLANG_MAMBA_PIN_TRACE = EnvInt(0)
    # #743 slot instrument: SUSTAINED lines per second for MAMBA-SLOT EVICT /
    # TRUNCATED. Successful mamba slot eviction and mamba-caused prefix
    # truncation were both silent, so slot pressure destroying reusable prefix
    # could not be read from a boot log. Per-event while pressure is
    # occasional; a SUPPRESSED rollup carrying the totals takes over above
    # this rate. The bucket's CAPACITY is decoupled (8) so a burst inside one
    # scheduler step is still reported in full. 0 turns the instrument off.
    SGLANG_MAMBA_SLOT_LOG_RATE = EnvFloat(2.0)
    # Zero the attention KV data buffers on /flush_cache (default ON, set 0
    # to opt out): a flushed server must match a fresh boot bit-for-bit even
    # if some kernel folds residual bytes beyond the valid region into its
    # result. Idle-time only, cost irrelevant.
    SGLANG_FLUSH_ZERO_KV = EnvBool(True)
    # Debug lever: after /flush_cache's empty_cache, claim + zero + release
    # the free device memory so allocator-recycled pages read as zeros like
    # the first-touch pages of a fresh boot. Discriminates kernels that are
    # sensitive to residual bytes in uninitialized activation scratch.
    SGLANG_FLUSH_SCRUB_FREE_MEMORY = EnvBool(False)
    # Debug lever: fill pool DATA buffers (mamba states/intermediates/rings,
    # MHA KV) with NaN at boot instead of zeros. Any kernel that reads pool
    # bytes never written for the current request then surfaces as NaN
    # output immediately, instead of a silent traffic-dependent divergence.
    SGLANG_POISON_POOL_DATA = EnvBool(False)
    # Debug lever (#50 campaign): after every finished request, walk the
    # process-persistent objects of the target/draft workers (model runners,
    # attn backends, cuda-graph runners, spec workers) and log a sha256 per
    # reachable tensor plus every plain int/float/bool attribute. Diffing the
    # dumps of two identical requests pinpoints exactly which persistent
    # state a request mutates (deterministic cross-request state evolution).
    SGLANG_SPEC_STATE_HASH = EnvBool(False)
    # 0 = hash every tensor fully. >0 = tensors above this many MiB are
    # fingerprinted from a strided sample instead (faster, still detects
    # virtually any realistic mutation).
    SGLANG_SPEC_STATE_HASH_MAX_MB = EnvInt(0)
    # Falsifier for stale-tail reads of persistent input buffers, eager AND
    # graph replay: (a) the draft-extend graph runner fills the padded tail
    # rows of its replayed input buffers with loud junk (token id 100 /
    # hidden 1024.0) instead of the neutral zero reset; (b) the
    # CudaGraphBufferRegistry poisons every slot-buffer element beyond the
    # current iteration's raw region (floats NaN, uint8 0xFF, ints 100)
    # before the semantic pad reset and head copy. If outputs differ from a
    # clean boot, some kernel reads beyond the active region (a KEEP_PAD /
    # FOREACH_COPY "tail is never read" claim is violated and the junk
    # localizes it); bit-identical output exonerates the stale-tail class
    # for the whole registry. Diagnostic boots only.
    SGLANG_POISON_GRAPH_PAD = EnvBool(False)
    # Reset probe (#50 campaign, round 9): comma-separated families of
    # process-persistent state to hard-reset after every finished request.
    #   "flashinfer" — zero every tensor held by flashinfer/sgl_kernel
    #                  wrapper objects (plan/workspace/kv_lens buffers);
    #                  the next plan() must rebuild everything it consumes
    #                  from the current batch alone.
    #   "registry"   — zero every CudaGraphBufferRegistry slot buffer.
    # If the request-ordinal-dependent output sequence flattens under a
    # family, that family carries the cross-request state; if the sequence
    # continues unchanged, the family is exonerated wholesale. Diagnostic
    # boots only.
    SGLANG_SPEC_RESET_PROBE = EnvStr("")
    # Bisection filter for the "flashinfer" reset-probe family: comma-
    # separated fnmatch globs on wrapper ATTRIBUTE names; only matching
    # tensors are zeroed (empty = all). Needed because in non-graph mode the
    # wrappers' _qo_indptr_buf / _paged_kv_*_buf are REFERENCES to
    # sglang-owned buffer slices (zeroing them perturbs more than wrapper
    # state — kv_last_page_len is init-ones and never refilled). The
    # wrapper-OWNED persistents are: _float_workspace_buffer,
    # _int_workspace_buffer, _pin_memory_int_workspace_buffer,
    # _kv_lens_buffer.
    SGLANG_SPEC_RESET_PROBE_FILTER = EnvStr("")
    # KL tests: skip the cache-hit count assertion (e.g. when alloc failure reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # Physical KV-page checks: committed<=allocated + no page alias.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)

    # #788: seconds a confirmed ADMISSION-WEDGE (invariant_checker.py) must
    # persist, on top of the report threshold, before the watchdog fires ONE
    # forced-admission recovery attempt for that episode. See
    # ADMISSION_WEDGE_RECOVERY_SECONDS in invariant_checker.py for the
    # default's derivation and rationale.
    SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS = EnvFloat(-1)

    # #788: per-rank admission-verdict trace. OFF by default -- it exists to
    # convert a MECHANISM proof into a captured value on one instrumented
    # boot, not to run permanently. Under PP every rank re-derives the
    # admission verdict locally, and a rank that declines forwards the request
    # but can never send the proxy its downstream blocks on. This trace prints
    # each rank's verdict and the host-side inputs behind it so a divergence
    # is visible in one grep instead of a py-spy hunt. It logs ONLY host-side
    # integers: no device tensor may reach a log argument here (see #790,
    # where exactly that stringification synced inside logging.emit and wedged
    # the scheduler for 25 minutes).
    SGLANG_PP_ADMISSION_TRACE = EnvBool(False)

    # Load snapshot backend
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
    # #273: how many times in a row a request may be the sole survivor of
    # retract_decode and still not fit before it is failed instead of
    # re-queued again. Ordinary extreme pressure (e.g. the #236/#242
    # kv-session-offload spill budget running out) resolves within a couple
    # of scheduler iterations; a request still solo-OOMing past this many
    # retries is structurally too large for the pool, not merely contended.
    SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES = EnvInt(8)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    # #631a. Restores the pre-#631a behaviour: PD arms launched with a
    # speculative algorithm auto-DISABLE it and warn, instead of refusing.
    # The default is now the refusal, because the auto-disable is silent in
    # the only way that matters -- a decode arm asked for NEXTN comes up
    # without it and merely serves slower, which no smoke test catches. This
    # escape hatch exists for shared launch configs that pass one flagset to
    # both a PD and a non-PD server (the original design ruling's reason).
    SGLANG_PD_AUTO_DISABLE_SPEC = EnvBool(False)

    # Scheduler: others:
    # #547 idle blocking poll. Turns the scheduler's (and the DP controller's)
    # true-idle busy spin into a blocking zmq poll with a stepped-up timeout,
    # without requiring the `--sleep-on-idle` server arg. Off by default: the
    # CPU win is proven hermetically, but the loaded-path A/B and the idle
    # wattage still need a card window (see IdleSleeper for the ladder and the
    # exact "loaded => unchanged" condition).
    SGLANG_IDLE_BLOCKING_POLL = EnvBool(False)
    # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # #616 index-race guard (srt/debug_utils/index_race_guard.py). Sync-free,
    # non-fatal bounds + stability instrumentation for the index tensors of the
    # overlap / speculative-decode path. Default off; when off the guard costs a
    # single module-level bool test per call site.
    SGLANG_INDEX_RACE_GUARD = EnvBool(False)
    # Clamp offending values back into range instead of letting the kernel
    # assert, so a run SURVIVES the first bad batch and keeps reporting.
    # Diagnostic only -- output is not trustworthy on a round that reports a hit.
    SGLANG_INDEX_RACE_GUARD_CLAMP = EnvBool(False)
    # Poll the guard counters every N scheduler iterations.
    SGLANG_INDEX_RACE_GUARD_POLL = EnvInt(1)
    # Directory for the guard's durable per-rank counter dump. A rank that HANGS
    # never reaches an exception handler and never logs again, so a log line is
    # not a record for that failure mode -- a file is.
    SGLANG_INDEX_RACE_GUARD_DIR = EnvStr("")
    # Force the overlap scheduler's WAR barrier onto its CONSERVATIVE form
    # (full wait_stream on the forward stream) instead of the fast-path
    # read-done event. #616 bisection arm: if the crash disappears with this
    # set, the fast-path event is published before the forward's last read of
    # the shared pool.
    SGLANG_WAR_BARRIER_FASTPATH = EnvBool(True)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    # PP: log the stage-boundary traffic every N crossings (0 = off). Counts
    # bytes and wall time at the two chokepoints every crossing passes through,
    # which is the only way to put a number on a boundary that spans two hosts.
    SGLANG_PP_BOUNDARY_STATS = EnvInt(0)
    # #201 slice 3: cache the pickled tensor-dict METADATA at the pipeline
    # stage boundary. At bs=1 the gloo-pickled metadata costs MORE than the
    # hidden-state payload itself (measured slice 2: 249 us vs 142 us
    # one-way), and the shapes are static per batch geometry -- so a repeat
    # crossing sends a 16-byte reference instead of size+pickle. Mirrored
    # sender/receiver caches stay in lockstep over the FIFO p2p channel.
    # Off by default (byte-identical wire protocol unless set).
    SGLANG_PP_SHAPE_CACHE = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)

    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Uneven TP: per-family weight vectors ("a,b,c", one positive integer
    # per TP rank). Take precedence over --rank-mlp-ratio /
    # --rank-moe-ratio when both are set. Emitted by the KV-pool
    # self-calibration as a restart hint. MLP = dense-MLP/shared-expert
    # family, MOE = fused expert-weight family.
    SGLANG_UNEVEN_MLP_VECTOR = EnvStr(None)
    SGLANG_UNEVEN_MOE_VECTOR = EnvStr(None)
    # Ratio-weighted vocab sharding vector ("a,b,c", one positive integer
    # per rank) for VocabParallelEmbedding/ParallelLMHead; overrides
    # --rank-vocab-ratio when both are set. Unlike MLP/MOE this family
    # NEVER falls back to the base --rank-tp-ratio plan -- without a
    # vector, vocab sharding stays even (the classic layout).
    SGLANG_UNEVEN_VOCAB_VECTOR = EnvStr(None)
    # Uneven DCP token-axis split vector ("a,b,c", one positive integer per
    # DCP rank). Overrides the budget-estimate vector resolve_cp_token_ratios
    # would otherwise derive. Emitted by the KV-pool self-calibration as a
    # restart hint (measured optimal from the actual per-rank profiled token
    # capacity); feeding it back on the next boot converges the per-rank KV
    # pools to the profiled optimum. Model-type-agnostic (keys off measured
    # capacity, which is dtype-independent).
    SGLANG_UNEVEN_TOKEN_VECTOR = EnvStr(None)
    # #797: "pin" (the vector is an assertion) or "seed" (an estimate the
    # measured optimum may supersede in-process). Unset reads as "pin", so a
    # process that never sets it behaves exactly as before.
    SGLANG_UNEVEN_TOKEN_VECTOR_ROLE = EnvStr(None)
    # #797: where the token vector came from -- the investigation, task id or
    # tool that produced it ("#602", "planner", "measured"). An ACTIVE vector
    # whose provenance names a RETRACTED investigation is refused at boot
    # (planner/retracted.py). Unset falls back to matching the vector's VALUE
    # against the values each retraction recorded, which is what catches a
    # retracted vector nobody declared a lineage for.
    SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE = EnvStr(None)
    # Log one per-rank residency census line once everything permanent is
    # resident (planner/residency_census.py). Read-only instrument for
    # calibrating the #485 cut gate against exclusively-owned, measured bytes
    # instead of a fit; unset, the boot is byte-identical.
    SGLANG_RESIDENCY_CENSUS = EnvBool(False)
    # Directory the census also writes itself to as JSON, one file per rank.
    # A later boot points --pp-solve-cut at it to solve the layer cut against
    # these measured bytes. Requires SGLANG_RESIDENCY_CENSUS.
    SGLANG_RESIDENCY_CENSUS_DIR = EnvStr(None)
    # Override the location of the MEASURED card-rate library (#584) that
    # --pp-solve-cut prices its stages from. Unset, the pass and the solver
    # agree on ~/.cache/sglang/card_library.json, beside the #213 card probe
    # the rates are projected from. Set it to point a boot at an artifact
    # measured elsewhere -- the rates are keyed by card UUID, so an artifact
    # from another rig is visibly not this rig's.
    SGLANG_CARD_LIBRARY = EnvStr(None)
    # Override the location of the PER-STAGE measurement canon (#584 second
    # half) that the #363 stage table promotes solved candidates from: the
    # measured gain over the reference stage, the band that gain was taken
    # against, and the instrumented flip cost. Unset, the store sits beside
    # the card library above, and its EXISTENCE is the gate -- no file means
    # the pre-#584 path, where an unmeasured candidate refuses the table by
    # name. Records are keyed by the sorted card-UUID set plus the checkpoint,
    # so a record from another rig or another model is refused rather than
    # borrowed. Written by planner/stage_measure_pass.py.
    SGLANG_STAGE_MEASUREMENTS = EnvStr(None)
    # Record, per rank, the driver-visible free-memory MINIMUM reached in each
    # load state the rank actually serves (planner/transient_census.py), and
    # write it beside the residency census. The #485 cut gate funds the WORST
    # measured state, because a transient measured under one load state does
    # not transfer to another -- a scalar measured at a prefill trigger
    # admitted cuts that broke the corridor under a mixed soak, twice. Uses
    # SGLANG_RESIDENCY_CENSUS_DIR for its output. Unset: byte-identical.
    SGLANG_TRANSIENT_CENSUS = EnvBool(False)
    # Sample one batch in this many for the transient census above.
    SGLANG_TRANSIENT_CENSUS_STRIDE = EnvInt(8)
    # Force a fresh stage-0 hardware micro-probe for --rank-tp-ratio
    # auto-performance, ignoring the cached profile under ~/.cache/sglang.
    SGLANG_PERF_REPROBE = EnvBool(False)
    # Wall-clock cap (seconds) on the WHOLE stage-0 probe subprocess.
    SGLANG_PERF_PROBE_TIMEOUT_S = EnvFloat(600.0)
    # Wall-clock cap (seconds) on the NETWORK phase of the stage-0 probe (the
    # pairwise NCCL link matrix). The phase joins a process group, so it is
    # the one part of the probe that can wait on something other than this
    # rig's own hardware; without a cap it inherits torch's 600 s default
    # process-group timeout and charges it to every boot. On expiry the probe
    # keeps the per-card measurements, stores the reason next to the empty
    # link table, and returns.
    SGLANG_PERF_PROBE_LINK_TIMEOUT_S = EnvFloat(45.0)
    # Skip the link matrix entirely (per-card measurements only).
    SGLANG_PERF_PROBE_SKIP_LINKS = EnvBool(False)
    # Refit seam for the parse-time cost model (uneven_perf.PerfCalibration).
    # The stage-0 probe MEASURES per-card GEMM/membw/GEMV rates on every
    # machine; the four scalars below are the model's FITTED/ASSUMED
    # constants, fitted on the reference rig only. On other hardware they are
    # a hypothesis — refit them there (recipe in the PerfCalibration
    # docstring) and set the result here instead of editing code. Unset
    # (None) keeps the shipped reference-rig values.
    SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP = EnvFloat(None)
    SGLANG_PERF_DECODE_PEAK_COMPRESSION_EXP = EnvFloat(None)
    SGLANG_PERF_DECODE_NONWEIGHT_FRACTION = EnvFloat(None)
    SGLANG_PERF_PREFILL_INVARIANT_FRACTION = EnvFloat(None)
    # Override seam for #330's absolutely-free VRAM corridor (MiB per card)
    # as the PLANNER prices it in the fundability gate. Not a measurement and
    # not rig-fitted: it is the policy "a boot must leave this much
    # unallocated on every card". The single definition is
    # registry.ledger.DEFAULT_CORRIDOR_BYTES (400 MiB, #330), which the ledger
    # daemon exposes as --corridor-mib; unset (None) reads that one.
    SGLANG_PLANNER_CORRIDOR_MIB = EnvInt(None)

    # --- barlink: vendor-neutral host-staged collectives (task #117) ---------
    # Route this group's TP collectives over barlink instead of NCCL. Needed
    # when a TP group spans GPUs with no common device collective library
    # (mixed NVIDIA + AMD); also forceable on a homogeneous group, where it
    # exercises the identical code path (on P2P-less consumer cards NCCL
    # already stages through the host, so the data movement is the same).
    # OFF by default -- with this unset the dispatch is byte-identical to
    # stock sglang.
    #
    # RANK-UNIFORMITY: every SGLANG_BARLINK* variable below MUST be set to the
    # same value on every rank of the group. Divergence does not produce a
    # wrong answer, it deadlocks -- the transports agree on a per-chunk flag
    # protocol, and a rank that took a different branch never publishes the
    # flag its peers spin on.
    SGLANG_BARLINK = EnvBool(False)
    # Data plane: "device" (GPU-driven DMA + spin kernels, CUDA-graph
    # capturable), "host" (GPU-driven zero-copy over ONE pinned, portable
    # host segment -- two kernels per op, no host sync, also capturable),
    # "shm" (CPU-orchestrated pinned staging), "gloo" (TCP, also multi-node)
    # or "ucx" (RDMA, multi-node; same host-staged semantics as gloo). The
    # CPU transports synchronize with the host and therefore require
    # --disable-cuda-graph.
    SGLANG_BARLINK_TRANSPORT = EnvStr("device")
    # #732 per-peer transport override, for A/B against the default per-link
    # policy (NCCL on fast edges where BAR1 is measured to lose, BAR1 on x4
    # edges where it wins). Comma separated, keys are RANKS -- never CUDA
    # ordinals: "all=nccl_sendrecv", "0>1=bar1_p2p", or both, with an explicit
    # pair beating "all". Unset -> the default policy, which on a tree without
    # a BAR1 p2p kernel degrades every BAR1 edge to NCCL and says so loudly.
    # Forcing bar1_p2p while that kernel is absent REFUSES rather than
    # degrading: a silent fallback would answer a different A/B than the one
    # asked. See barlink_peer_transport.py.
    SGLANG_BARLINK_PEER_MAP = EnvStr(None)
    # #279 path dispatcher (skeleton): size/load-aware path choice with
    # saturation overflow. Default off; even when on, decisions fall back to
    # the status-quo #240 class choice until measured rate tables are loaded
    # (placeholder neutrality), so enabling it is byte-identical today.
    SGLANG_BARLINK_PATH_DISPATCHER = EnvBool(False)
    # Per-rank shared-memory slot size (MiB) for payload staging.
    SGLANG_BARLINK_SLOT_MIB = EnvInt(64)
    # Chunk size (MiB) of the gloo data-plane pipeline.
    SGLANG_BARLINK_CHUNK_MIB = EnvInt(8)
    # Chunk size (MiB) of the device transport's dual-stream pipeline.
    # Unset -> calibrated at startup (a collective sweep; see the
    # rank-uniformity note above -- set it on all ranks or on none).
    SGLANG_BARLINK_PIPE_CHUNK_MIB = EnvStr(None)
    # Upcast half dtypes to fp32 for the gloo-plane reduction, to match
    # NCCL numerics.
    SGLANG_BARLINK_FP32_REDUCE = EnvBool(True)
    # RS+AG chunk-ownership weights for world >= 3 ("a,b,c", one positive
    # integer per rank). Unset -> measured from per-rank slot DMA bandwidth
    # at startup, so a slow PCIe link owns fewer reduce-scatter chunks.
    SGLANG_BARLINK_RSAG_SHARES = EnvStr(None)
    # --- host transport (pinned portable host memory, GPU-driven) ----------
    # Per-rank staging slot (MiB). Unset -> inherit SGLANG_BARLINK_SLOT_MIB, so
    # there is one knob for the common case and a second only when the host
    # transport should differ. The segment holds TWO slots per rank (the
    # double buffering that removes a third kernel from every collective),
    # so it costs 2 x world x this.
    SGLANG_BARLINK_HOST_SLOT_MIB = EnvStr(None)
    # Per-ordered-pair send/recv buffer (MiB), also double-buffered. 0
    # disables point-to-point, and the transport then DECLINES send/recv in
    # handles() rather than discovering the missing buffer later.
    SGLANG_BARLINK_HOST_P2P_MIB = EnvInt(4)
    # Grid width of the host transport's two data kernels. Its payloads are
    # latency-bound; more blocks buy nothing below ~1 MiB and cost tail
    # latency. Rank-uniform like every knob in this block.
    SGLANG_BARLINK_HOST_BLOCKS = EnvInt(32)
    # --- ucx transport (RDMA data plane) -----------------------------------
    # Which libucp to load. Unset -> the system "libucp.so.0". Point this at a
    # side-by-side install to satisfy the transport's version-parity check
    # when the hosts ship different UCX releases, e.g.
    # "/opt/ucx116/lib/libucp.so.0"; libucs/libuct are pre-loaded from the
    # same directory, so LD_LIBRARY_PATH is not additionally needed.
    # Mixed releases are REJECTED at rendezvous -- UCX's UCP wire address
    # format is not compatible across them and fails as "invalid bandwidth
    # 0.00".
    SGLANG_BARLINK_UCX_LIB = EnvStr(None)
    # Largest single UCX transfer (MiB). Chunks of one collective step are
    # posted and progressed together, so this caps per-request footprint
    # without costing extra round trips.
    SGLANG_BARLINK_UCX_CHUNK_MIB = EnvInt(4)
    # all_reduce payload (KiB) at or above which the one-step flat exchange
    # gives way to a ring. Below it latency beats bandwidth; above it the
    # flat exchange's (W-1) payloads per direction dominate. Measured
    # crossover on a cross-rig world-4 group is ~22 KiB (task #244), so a
    # speculative verify all-reduce sits on the ring side and a bs=1 decode
    # all-reduce on the flat side.
    SGLANG_BARLINK_UCX_RING_KIB = EnvInt(24)
    # Deprecated MiB spelling of the same threshold; still honoured, and it
    # wins when both are set.
    SGLANG_BARLINK_UCX_RING_MIB = EnvInt(None)
    # Same switch for all_gather (KiB); 0 disables the ring entirely. This
    # ring saves no bytes -- the flat exchange already moves the (W-1) * n
    # every rank must receive, in one round trip -- but it saves the single
    # UCX worker per rank from progressing 2(W-1) simultaneous requests.
    # Measured crossover cross-rig at world 4 is ~32 KiB (task #263), so a
    # bs=1 decode gather stays flat and a 4-token verify gather rings.
    SGLANG_BARLINK_UCX_AG_RING_KIB = EnvInt(32)
    # Largest host-side pass in elements that stays on the calling thread;
    # above it torch dispatches a CPU->CPU copy_/add_ through at::parallel_for.
    # Co-located TP ranks enter their host passes together, so the OpenMP
    # region's join lands on a descheduled thread and the 128 -> 256 KiB step
    # cost milliseconds (task #263). 0 restores the unchunked passes.
    SGLANG_BARLINK_UCX_GRAIN_ELEMS = EnvInt(32768)
    # Seconds before a pending UCX request is declared stuck. Guards against
    # a silent hang when a peer dies or the ranks disagree about the
    # collective sequence.
    SGLANG_BARLINK_UCX_TIMEOUT_S = EnvInt(300)
    # Overlap the MLP all-reduce with the layer boundary: issue it
    # asynchronously at down_proj, complete it in the next layer's
    # prepare_attn (rides the fuse_mlp_allreduce seam). Requires the ucx
    # transport; rank-uniform like every other flag in this block.
    SGLANG_BARLINK_UCX_OVERLAP = EnvBool(False)
    # Token-slice pipelining of the TP all-reduce (task #588). Splits a
    # row-parallel layer's token axis so slice i's transfer occupies the wire
    # during slice i+1's GEMM. Eager prefill only; the saving is bounded by
    # the layer's own GEMM time, never by the transfer term. Off by default:
    # when unset, RowParallelLinear.forward reads this bool and nothing else
    # changes. RANK-UNIFORM, like every flag in this block -- the slice count
    # is part of the collective sequence, so ranks that disagree deadlock.
    SGLANG_TP_AR_PIPELINE = EnvBool(False)
    # Upper bound on the slice count. Caps the K*latency term that grows with
    # K and bounds the extra launch traffic per layer.
    SGLANG_TP_AR_PIPELINE_MAX_SLICES = EnvInt(8)
    # Below this token count a forward stays unsliced. Keeps decode (and any
    # short extend) on the untouched path, where there is no transfer to hide
    # anything behind.
    SGLANG_TP_AR_PIPELINE_MIN_TOKENS = EnvInt(256)
    # Force a fixed slice count instead of deriving it from the measured cost
    # model. 0 = derive. For A/B arms that need K held constant, not for
    # production tuning.
    SGLANG_TP_AR_PIPELINE_SLICES = EnvInt(0)
    # Deferred join (task #597). Issues a layer's all-reduce on the comm
    # stream at the site that already owned it and joins at the first
    # consumer, so the transfer runs under everything in between. Independent
    # of SGLANG_TP_AR_PIPELINE: that one hides a collective under the
    # producing GEMM, this one under the issue-to-join window. Window 8
    # showed the production model's dominant all-reduce is the MoE layer's
    # own reduce, which the in-call hook never sees. RANK-UNIFORM.
    SGLANG_TP_AR_PIPELINE_DEFERRED = EnvBool(False)
    # Minimum token count for the deferred issue. Below it the collective is
    # too small for the handle bookkeeping to pay for itself.
    SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS = EnvInt(256)
    # Number of independent UCX contexts/workers per rank for the collective
    # plane (task #266). 2 splits the flat exchange's peers over the two
    # workers by the symmetric (rank + peer) % ways rule, so no rank has all
    # 2(W-1) requests of a decode collective on one progress engine. Measured
    # cross-rig at world 4: -7.6 % all_reduce and -8.1 % all_gather at the
    # 20 KiB bs=1 decode size, neutral (within noise) at every ring size.
    # Default 1 -- the transport also runs single-host over loopback/shm,
    # where a second context has no peers to spread. RANK-UNIFORM and more
    # strictly so than most: a rank that disagrees posts where nobody is
    # listening, which hangs rather than returning a wrong answer. Checked at
    # rendezvous before any endpoint exists.
    SGLANG_BARLINK_UCX_WORKERS = EnvInt(1)
    # Additionally run the RING half each way round, one direction per worker
    # (needs ..._WORKERS >= 2). Measured negative on this link -- a ring step
    # is two requests in lock step, so there is no concurrency for a second
    # worker to expose, and halving the bytes per hop buys nothing where the
    # bytes were never the cost (task #244). +17 % on an 80 KiB all_reduce.
    # Kept as the A/B control and for links where the bytes DO dominate.
    SGLANG_BARLINK_UCX_RING_BIDIR = EnvBool(False)
    # --- peer liveness for the collective family (task #312) ---------------
    # A rank that dies leaves its peers spinning: the gloo cpu_group every
    # barlink handshake runs on is built with a hardcoded 7200 s timeout, and
    # the BAR1 spin kernels carry only a rank-local cycle deadline whose
    # expiry writes a status word nothing reads. These four knobs bound both
    # sides. 0 restores the previous, unbounded behaviour exactly.
    #
    # Rank-uniform like the rest of this block, and more strictly so than
    # most: the deadline decides WHEN a rank gives up, and ranks that give up
    # minutes apart turn one clean group failure into a cascade.
    SGLANG_BARLINK_PEER_LIVENESS = EnvBool(True)
    # Seconds a host-side wait may make no progress before it gives up. Scaled
    # by SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT while the cold-build window is
    # open, so a first boot on an empty kernel cache does not trip it.
    SGLANG_BARLINK_PEER_TIMEOUT_S = EnvFloat(120.0)
    # How often a stalled wait, and the watchdog thread, may ask whether the
    # peer processes still exist. One kill(pid, 0) per peer, ~1 us.
    SGLANG_BARLINK_PEER_PROBE_S = EnvFloat(1.0)
    # Whether the watchdog thread runs. It is what ends a DEVICE-side spin:
    # no host code runs inside a captured graph replay, so somebody outside
    # the collective has to write the abort word the kernels poll.
    SGLANG_BARLINK_PEER_WATCHDOG = EnvBool(True)

    # --- per-message-class link selection (task #240) ----------------------
    # Env spelling of --collective-net-small / --collective-net-bulk. Set
    # either directly or let server-args resolution export it; the flag and
    # the variable carry the same value, and a value already present in the
    # environment is never overwritten.
    #
    # NOT rank-uniform, unlike the SGLANG_BARLINK* block above: the value is a
    # local device NAME, and the two ends of a link are normally called
    # different things (rocep4s0f1 on one host, rocep1s0f1 on the other).
    # What must match is the wire, not the string.
    #
    # SMALL pins the barlink UCX collective context (small AND large TP
    # collectives -- they share one context, see barlink_ucx.py), BULK reaches
    # the transfers that have a transport of their own: PD-KV / HiCache, by
    # seeding --disaggregation-ib-device when that is unset. On a host with
    # one line both are pointless; the payoff is a host with two, where a
    # FEC-free link wins on small-message latency while a wider one wins on
    # bulk bandwidth.
    SGLANG_COLLECTIVE_NET_SMALL = EnvStr(None)
    SGLANG_COLLECTIVE_NET_BULK = EnvStr(None)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)

    # Base port hint for ephemeral sockets (ZMQ, SHM broadcaster, etc.).
    # When set, get_open_port() and shm_broadcast search upwards from this
    # value instead of asking the OS for a random port.  Useful to keep all
    # SGLang ports in a predictable range behind a firewall.
    SGLANG_PORT = EnvInt(None)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Native web search (Exa). EXA_API_KEY is the vendor BYOK credential
    # (kept as-is, not renamed to SGLANG_*); the SGLANG_EXA_* knobs tune the
    # request defaults for the built-in GPT-OSS web_search tool.
    EXA_API_KEY = EnvStr(None)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)

    # Hi-Cache
    # Deadline (seconds) for the per-step HiCache control collectives. The gloo
    # cpu_group they run on defaults to a two-hour timeout, so a rank whose peer
    # died of OOM would otherwise sit in all_reduce for hours; on expiry the
    # surviving rank raises HiCacheCollectiveTimeoutError instead. <= 0 restores
    # the unbounded blocking wait.
    SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S = EnvFloat(600.0)
    # #410: the pin budget, 0 = unbounded. Read once when the store builds
    # its PinLedger; a checkpoint that would cross it is refused by name.
    SGLANG_HICACHE_PIN_BUDGET_BYTES = EnvInt(0)
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    # Enable client-side metadata caching to optimize filesystem checks (e.g. for Lustre/NFS/FUSE)
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    # Positive cache TTL for filesystem metadata lookups (-1 disables positive expiration)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    # #706: age at which an orphaned canonical partial page/blob (.part706 and
    # its .slots706 marker) is reaped at attach. Must stay comfortably longer
    # than the time all writers of one page need, or a live partial is reaped
    # from under a stage that is still filling it.
    SGLANG_HICACHE_CANONICAL_PARTIAL_TTL_S = EnvFloat(3600.0)
    # #720: size of the reusable, budget-REGISTERED read-buffer ring per pool.
    # 0 (default) keeps today's per-read fresh pinned allocation, which the
    # joint budget cannot see. A positive value declares capacity x page bytes
    # to the registry at first use, so the read path's pinned footprint becomes
    # a number the budget can refuse.
    SGLANG_HICACHE_READ_BUFFERS = EnvInt(0)
    # #558: free-space floor, in bytes, below which the #706 canonical write
    # protocol refuses rather than risking ENOSPC in the middle of a
    # multi-writer page assembly. 0 (default) keeps today's behaviour, where
    # the only protection is the LRU evictor's watermark -- which is disabled
    # unless --hicache-storage-backend-extra-config sets a cap or a min-free.
    SGLANG_HICACHE_CANONICAL_MIN_FREE_BYTES = EnvInt(0)
    # #410 slice 2: ceiling on bytes pinned by conversation checkpoints. Pinned
    # bytes are bytes eviction can never reclaim, so a checkpoint whose pins
    # would cross this is REFUSED with the numbers rather than quietly turning
    # the cache into a pin museum. 0 = no ceiling.
    SGLANG_HICACHE_PIN_BUDGET_BYTES = EnvInt(0)
    # #703: cap on OUTSTANDING eviction-time demotions to the disk tier, and
    # the on/off switch (0 = off, today's behaviour). Eviction runs under
    # memory pressure, so demotion enqueues onto the existing backup queue and
    # DROPS beyond this cap rather than queueing without limit -- a dropped
    # demotion is a later miss, never corruption.
    SGLANG_HICACHE_DEMOTE_ON_EVICT = EnvInt(0)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB = EnvInt(64)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)

    # MoRI KV Transfer
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    USE_ROCM_AITER_ROPE_BACKEND = EnvStr("0")
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("tilelang")

    # MPS (Apple Silicon)
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # Quantize x to int8 in the dispatch operator
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False)  # This argument is deprecated
    SGLANG_NPU_FUSED_MOE_MODE = EnvInt(1)

    # MTHREADS & MUSA
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    # Opt-in BIT-DETERMINISM for fp8 linears on sm80..sm88 (#192, from #190).
    #
    # WHAT IS BROKEN. On sm80..88 an fp8 checkpoint has exactly one GEMM
    # available -- Marlin (can_auto_enable_marlin_fp8: 80 <= sm < 89) -- and
    # gptq_marlin_gemm is NOT run-to-run reproducible there. Measured on an RTX
    # 3080 at the real 27B shape (N=8704, K=5120), repeating one
    # apply_fp8_marlin_linear call on bit-identical inputs: 0/1200 mismatching
    # iterations for M <= 109, then 1/1200 at M=128, 4/1200 at M=256, 12/1200 at
    # M=512, worst per-element |delta| ~1e-1. The cause is the accumulation order
    # across K-slices inside the kernel (falsified: not the atomic-add path, not
    # use_fp32_reduce, not a stale workspace) -- a CUDA-kernel defect, not a
    # config flip. The same shape on sm120 is 0/2000; sm89+ is unaffected.
    #
    # WHAT THIS FLAG DOES. On sm80..88 ONLY, it forces the fp8 Marlin path off
    # for the dense linears that have a paired fallback, which routes them to the
    # dequant W8A16 lane (fused dequant-GEMV for small-batch decode,
    # materialise + F.linear for prefill). sm89+/sm90/sm120 are untouched -- the
    # native / flashinfer fp8 paths there are already clean, and this flag is not
    # a global "no fp8".
    #
    # WHAT IT COSTS. A large decode penalty, and how large depends on how much
    # of the model sits on sm8x ranks:
    #   * #179/#189 anchor, Qwen3.6-27B block-fp8, TP=3 (5090 + 2x 3080), the
    #     5090 carrying its share on the native path: 91.5 tok/s Marlin against
    #     27.6 (uncached dequant) to 37.3 (fused GEMV) on the fallback -- roughly
    #     a factor 2.5, and that is the OPTIMISTIC end;
    #   * this flag's own A/B, same checkpoint but TP=2 across the two 3080s so
    #     every layer is on sm8x: 36.4 -> 5.8 tok/s on a short single request.
    # Prefill is only mildly affected (-8% at the #179 anchor) -- the fallback
    # expands the weight once per FORWARD, so batch-1 decode pays the whole
    # thing for one token while prefill amortises it over the prompt.
    # This is deliberately opt-in: pay it when byte-reproducibility is the
    # product (CI byte gates, bisecting a numerical regression, debugging), not
    # in normal serving.
    #
    # COVERAGE GAPS, on purpose. fp8 MoE experts, FBGEMM fp8 and the
    # multimodal_gen runtime keep Marlin on sm8x because they have no fallback
    # there -- switching them off would leave no fp8 GEMM at all. Each logs a
    # warning when this flag is set. Under mixed-arch TP (5090 + 3080s) the flag
    # is rank-local by construction: only the sm8x ranks change lane, so ranks
    # stop agreeing numerically -- that is the #50 broadcast family's problem,
    # not this flag's.
    SGLANG_DETERMINISTIC_FP8_GEMM = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    # MoE expert-offload (M-B/M-C, feat/moe-expert-offload). Fraction of each
    # layer's local routed experts kept resident on GPU (1.0 = no offload =
    # default, byte-identical). <1.0 activates the pinned-host pool + LRU
    # H2D-fetch cache so the resident-fraction-vs-tok/s curve can be swept on a
    # model that otherwise fully fits (A3B-FP8). Cold experts are FETCHED to
    # GPU and computed on GPU (NOT CPU-computed — this AMD box has no AMX).
    # May be a single float (uniform, the original behaviour) or a comma-list
    # with one entry per TP rank. A vector exists because the fraction is the
    # GPU-resident / host-pinned split WITHIN a rank's own expert shard, and on
    # a heterogeneous group the right split differs per card: on this rig a
    # 5090 rank had ~4.0 GiB of VRAM idle while both 3080 ranks were 32 MiB
    # short of fitting their scratch region. Lowering the fraction only on the
    # small cards frees exactly the VRAM that binds, and raising it on the big
    # card pays the resulting host-pinned-pool growth back. See
    # --rank-moe-resident-fraction and docs/dev/PLAN_MOE_RESIDENT_FRACTION_PER_RANK.md.
    SGLANG_MOE_RESIDENT_EXPERT_FRACTION = EnvFloatVector(1.0)
    # When set to a path, log per-layer routed expert IDs (topk_ids) to that
    # file for offline routing-locality / cache-hit-rate simulation (M-C).
    SGLANG_MOE_OFFLOAD_TRACE = EnvStr("")
    # Stage-1 hot-expert residency (offload-speed bundle). When True (and offload
    # is active, fraction<1.0), the resident GPU set is FROZEN to the R most-
    # frequently-routed experts per layer -- observed over the first
    # SGLANG_MOE_HOT_CALIB_STEPS forwards (a deterministic calibration pass), then
    # never changed. Cuts H2D spill traffic (real MoE routing is heavily skewed;
    # the default static [0,R) set captures only ~uniform R/E). BYTE-IDENTICAL to
    # the static-residency path at the same fraction: residency only changes WHICH
    # experts are physically resident vs fetched, not the per-token math (same
    # GEMM, same buffer size R+C, per-expert token sets unchanged). Frozen-after-
    # calibration => self-deterministic. Default False = static [0,R) (unchanged).
    SGLANG_MOE_HOT_RESIDENCY = EnvBool(False)
    # #302a Stage-2 heat migration: keep re-ranking the resident set against a
    # DECAYED window of live router traffic instead of freezing it once. Stage-1
    # above improves the choice but keeps the one-shot shape; this reacts to a
    # workload that drifts after calibration. Swaps are EQUAL-COUNT pairs, so
    # residency size -- and every VRAM figure derived from it -- is unchanged.
    # Eager offload path only (refused under SGLANG_MOE_OFFLOAD_CUDA_GRAPH: a
    # captured gather's LUTs pin the layout). Default False = unchanged.
    SGLANG_MOE_HEAT_MIGRATION = EnvBool(False)
    # Forwards between two re-rank decisions, per layer. Small = re-ranks on
    # noise and pays PCIe for it; large = tracks a drifting workload slowly.
    SGLANG_MOE_HEAT_PERIOD = EnvInt(512)
    # #516 longer-horizon miss budget for the heat re-rank. 0.0 = OFF and the
    # OFF path is byte-identical. When > 0, a window whose miss rate is at or
    # below this is left alone instead of re-ranked, so a swap is spent only
    # where the miss rate says it is needed. Simulation on the recorded #302a
    # series favours 0.04; nothing here has run on metal.
    SGLANG_MOE_HEAT_MISS_BUDGET = EnvFloat(0.0)
    # Decay multiplied into every expert's count at each round boundary.
    # 1.0 = whole-run heat, 0.0 = only the last period counts.
    SGLANG_MOE_HEAT_DECAY = EnvFloat(0.5)
    # A candidate must be (1+x) times hotter than the victim it would displace.
    # This is the anti-thrash term: without it a one-activation difference
    # swaps back and forth every round.
    SGLANG_MOE_HEAT_HYSTERESIS = EnvFloat(0.25)
    # Absolute companion to the margin above, in observed activations. A purely
    # relative margin is scale-free and churns on sampling noise down in the
    # tail of the routing distribution, where "40 % hotter" is three
    # activations. A swap costs two expert-row transfers; both conditions must
    # hold before it is taken.
    SGLANG_MOE_HEAT_MIN_GAIN = EnvFloat(8.0)
    # Upper bound on swaps per layer per round; the burst is swaps x expert
    # bytes and lands between two forwards.
    SGLANG_MOE_HEAT_MAX_SWAPS = EnvInt(4)
    # Minimum activations observed in a window before it is allowed to re-rank.
    SGLANG_MOE_HEAT_MIN_OBS = EnvInt(32)
    # #286 offload register (DESIGN_201 Nachtrag-13 Erg. 7/7b/7c): enable the
    # generic VRAM item register's ADAPTERS (registration + size/access
    # bookkeeping at the item creation sites: capture rungs, drafter heads,
    # lane workspaces, input-buffer pools). Default False = the adapters are
    # no-ops and the default path stays byte-identical. CPU phase: bookkeeping
    # only, no movement.
    SGLANG_OFFLOAD_REGISTER = EnvBool(False)
    # Number of offload forwards to observe (accumulating per-expert routing
    # counts) before the hot-set is computed, physically installed, and FROZEN.
    # Default 1: freeze right after the first forward (a prefill sees ~all prompt
    # tokens => rich frequency signal in one shot). The freeze happens at the TOP
    # of the triggering forward, so that forward's own output already uses the
    # frozen hot-set (no intra-run residency drift => self-det holds).
    SGLANG_MOE_HOT_CALIB_STEPS = EnvInt(1)
    # Stage-3 CUDA-graph-compatible offload. When True (opt-in), the decode MoE
    # offload uses the on-device index math (prepare_capturable) + a captured
    # gather instead of the per-layer topk_ids.tolist()+Python-planning path, so
    # decode can be CUDA-graph captured (removing the launch-overhead that
    # dominates single-token decode). Requires a residency layout frozen BEFORE
    # capture (static [0,R) or SGLANG_MOE_HOTSET_FILE); live hot-calibration is
    # rejected on this path. Default False = eager run_waves (unchanged).
    #
    # #452: REFUTED on hardware and refused by name at boot
    # (moe/offload_capture_gate.refuse_capturable_offload_decode). Setting this
    # to True aborts the launch unless the override below is also set.
    SGLANG_MOE_OFFLOAD_CUDA_GRAPH = EnvBool(False)
    # Development override past the #452 refusal, for a card window that wants
    # to localise B2 or measure a candidate fix. Not a performance option: the
    # measured operating point is 6.60x slower than the eager offload path and
    # decodes different text.
    SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE = EnvBool(False)
    # Path to a per-layer frozen hot-set file (produced offline from the M-C
    # routing trace). Enables hot-residency under CUDA-graph capture by freezing
    # the resident set from the file before capture, instead of live calibration.
    SGLANG_MOE_HOTSET_FILE = EnvStr("")
    # Max decode batch size eligible for the captured offload path. Buckets with
    # bs*top_k > scratch (would need >1 wave) fall back to eager. 0 = no cap.
    SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS = EnvInt(0)
    # #254: how a prefill forward that overflows the scratch region is split.
    #   "token"  (default) -- waves are disjoint TOKEN subsets; every wave
    #     re-fetches the spill experts its tokens need, so a spill expert is
    #     streamed once PER WAVE (~62x per 2048-token chunk at C=16).
    #   "expert" -- waves are disjoint SPILL-EXPERT groups; each spill expert is
    #     streamed EXACTLY ONCE per chunk (H2D volume drops by the wave count).
    #     Byte-identical to "token": each wave computes the per-(token, k-slot)
    #     contributions into a fixed [T, top_k, H] buffer indexed by the k-slot
    #     (which the routing fixes, independent of the wave split), and the top-k
    #     reduction runs once at the end over the full buffer in k order -- the
    #     same reduction, over the same values, as the unsplit path. Costs one
    #     transient T*top_k*H buffer per layer (freed at the end of the forward).
    # Decode (single-wave) is unaffected by this flag.
    SGLANG_MOE_OFFLOAD_WAVE_ORDER = EnvStr("token")
    # #119: hand the weight VRAM freed by the expert offload to the KV pool.
    # Default ON, but STRICTLY scoped to the offload lane -- every effect is
    # additionally gated on SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0, so with
    # offload off (the default) this flag changes nothing and the sizing path
    # stays byte-identical. On the offload lane it (a) enforces that the offload
    # is installed BEFORE the KV pool is profiled -- otherwise the profiler
    # measures the pre-offload footprint and the reclaim is lost, which was the
    # #77 "known limitation"; (b) synchronizes the release across the TP group
    # so every rank's freed blocks are back with the driver before ANY rank
    # takes its free-memory reading; and (c) logs the reclaimed bytes. Set to
    # False to fall back to the unsynchronized, unaccounted behaviour.
    SGLANG_MOE_OFFLOAD_KV_REGAIN = EnvBool(True)
    # #390: router-distribution and VRAM residency hit-rate counters in the
    # expert-offload path. Opt-in; off by default and the counters are then not
    # even constructed, so the offload path costs one `is not None` test. When
    # on, every eager offload forward folds its already-materialized routing
    # decision (topk_ids.tolist(), the sync run_waves pays anyway) into a
    # per-layer expert-activation histogram plus hit/miss against the resident
    # set. No extra device sync and nothing in the kernel. Captured decode steps
    # (SGLANG_MOE_OFFLOAD_CUDA_GRAPH) are not counted -- counting them would
    # require the host sync that path exists to avoid; the dump flags this.
    SGLANG_EXPERT_STATS = EnvBool(False)
    # #391c: fill the GGUF MoE offload's two tiers FROM THE WEIGHT STREAM.
    # Without this the interception point is process_weights_after_loading,
    # which the loader runs only after the complete load_weights pass -- so
    # every owned expert first accumulates in host anon memory and the residency
    # plan arrives too late to prevent the peak it exists to prevent (boot
    # attempt 5 of #391: rank 0 OOM-killed at 90.7 GiB of anon on a swapless
    # 98.5 GiB box). With it on, each expert is routed into its tier as it
    # leaves the stream and the peak is pinned tier + one layer's expert set.
    # Only ever consulted on a layer the offload already covers
    # (SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0 and a ggml type with a MoE
    # kernel), so a default GGUF boot is byte-identical either way. Set to 0 to
    # fall back to the accumulate-then-materialize path -- a debugging switch,
    # not an operating mode.
    SGLANG_MOE_GGUF_STREAM_STAGING = EnvBool(True)
    # #391c: log the streaming stager's own byte accounting at every layer
    # boundary (cumulative pinned/resident/streamed bytes plus the in-flight
    # peak), so a boot's external RAM monitor can be cross-checked against what
    # the code thinks it is holding. Off by default; pure logging.
    SGLANG_MOE_STAGING_TRACE = EnvBool(False)
    # #396(a): materialize COLD experts on FIRST TOUCH instead of reading every
    # one of them into the pinned host tier at load time. The tier is still
    # allocated at load (so every byte figure and capacity check is unchanged);
    # only the reads move to the first router hit for that expert, behind the
    # same ``pool[row]`` accessor the #125 prefetch and the #394 cold shard
    # already use -- neither consumer gains a branch. Requires a door that can
    # hand ``stage_experts_into_tiers`` a per-expert ``ExpertFileRef``; a door
    # that cannot stages eagerly regardless of this flag, so turning it on can
    # never make a boot read LESS than it can prove it is able to re-read.
    # Off by default: a lazy tier trades a shorter boot for a first-token stall
    # per cold expert and for a hard dependency on the checkpoint staying in
    # place, and neither is a default anybody should get without asking.
    SGLANG_EXPERT_LAZY_STAGING = EnvBool(False)
    # Dump prefix; each rank writes "<prefix>.<rank_tag>.json". Default /tmp.
    SGLANG_EXPERT_STATS_PATH = EnvStr("")
    # Additionally dump every N seconds (0 = only on exit / SIGUSR2).
    SGLANG_EXPERT_STATS_INTERVAL_SEC = EnvFloat(0.0)
    # #407 cut 2: rank -> physical card UUID vector, one per WORLD rank in
    # world_rank order, published by the launcher so no worker needs a
    # collective to learn the group's placement (#394's link-proportional
    # cold-expert shards are the first consumer). Normally written by
    # _launch_subprocesses; set it by hand for a launch that does not go
    # through it, and a hand-set value is never overwritten.
    SGLANG_RANK_CARD_UUIDS = EnvStr("")
    # Licence for the LAUNCHER to create a CUDA context purely to resolve that
    # vector. Off by default: the context costs a few hundred MiB on every
    # visible card, in the process that is about to spawn workers onto them.
    # Unnecessary with --rank-gpu-id, whose validation resolves the cards
    # already.
    SGLANG_RANK_CARD_PROBE_CUDA = EnvBool(False)
    # #394: weakest provenance the cold-expert host split may be weighted by --
    # "measured" (the rigmon card probe's timed H2D, or an operator-supplied
    # SGLANG_MOE_HOST_SHARD_RATIO vector) or "estimate" (the NVML PCIe
    # width x generation nameplate derivation). "absent" is not selectable in
    # either setting; an unknown link yields an equal split, which is exactly
    # today's assignment.
    SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE = EnvStr("estimate")
    # #394: allow cold-expert delegation on a layer whose ranks hold DISJOINT
    # expert ranges (the #82 GGUF expert-dim shard). Off, because there it is
    # unsound: a delegated expert is not relocated to a peer, it is absent, and
    # the first token routed to it fails. Measured 2026-08-02 on V4-Flash TP=3 --
    # all 43 layers staged, then every rank died on the first forward. The flag
    # exists to develop the missing reachability mechanism (a shared-memory host
    # pool, or replicated experts with an EP dispatch) against a real boot. It
    # is not a performance option.
    SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE = EnvBool(False)
    # #394 slice 2: put this rank's cold expert pool in a NAMED SHARED segment
    # (/dev/shm) instead of a private pinned allocation, and publish a manifest
    # so peers can DMA a delegated expert's row out of it. This is the
    # reachability mechanism the refusal above names as missing: with it on, a
    # delegated cold expert is relocated rather than absent. Off by default --
    # the tier costs a tmpfs-visible allocation and the shm size cap is not
    # restart-persistent, so it is an explicit operator decision.
    SGLANG_MOE_COLD_TIER_SHM = EnvBool(False)
    # Bounded wait for a peer's cold-tier manifest at the FIRST fetch, which is
    # long after every rank has loaded. Not a barrier: it expires with a named
    # error rather than hanging the group.
    SGLANG_MOE_COLD_TIER_MANIFEST_TIMEOUT_S = EnvFloat(30.0)
    # #394 slice 2, graph seam: capture a decode graph over a layer whose cold
    # rows live in a peer's segment. BOOT-PENDING -- the UVA device pointer for
    # a peer mapping needs cudaHostGetDevicePointer on the registered range and
    # has not been exercised on hardware, so the capturable installer refuses
    # by default rather than capturing a graph over an address it has not
    # verified. Graphs pin ADDRESSES, not contents, so the seam is sound in
    # principle; this flag exists to prove it in a card window.
    SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE = EnvBool(False)
    # #394 slice 3: which policy produced the installed --rank-moe-ratio, so a
    # #390 dump can tell a SOLVED vector from one an operator typed. Written by
    # the launcher when it resolves "--rank-moe-ratio link"; read only by the
    # expert-stats dump. An A/B arm that cannot identify itself is an A/B whose
    # null result was never tested.
    SGLANG_MOE_COMPUTE_POLICY = EnvStr("")
    # #394 slice 3 / #439: the "moe" family vector the solve held the RESIDENT
    # expert mass against, i.e. the plan the boot would have run without
    # "--rank-moe-ratio link". Written by the launcher next to the policy label
    # above; read by the residency sizing in every worker so a rank that GAINS
    # experts does not also gain resident VRAM. Absent = today's sizing.
    SGLANG_MOE_COMPUTE_BASE_PLAN = EnvStr("")
    # #394 slice 3: per-rank cold-traffic coefficients ("a,b,c", mean 1),
    # measured from a PRIOR boot's #390 dump with
    # cold_traffic_coefficients_from_measurement. Without them the solve uses
    # the first-order model (a cold expert is fetched, a resident one is not),
    # which the reference-rig battery shows has a per-rank residual. There is
    # deliberately no automatic dump -> launch path: a coefficient measured on
    # one recipe is not a property of the rig.
    SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS = EnvStr("")
    # Weightless-KV streaming block-decode graphs (#136a): max decode capture
    # bucket. Each bucket carries a full ladder block-wrapper pool (~8 MB int
    # workspace per block), and the host-spill graph path only supports bs=1;
    # larger decode batches fall back to the eager block loop.
    SGLANG_WL_GRAPH_MAX_BS = EnvInt(1)
    # Weightless-KV streaming H2D prefetch / double-buffer (#136b): carve TWO
    # block-sized staging regions and, inside the captured block-decode graph,
    # issue each block's host-spill H2D copy on a side stream so it overlaps
    # the previous block's attention (PCIe transfer hidden behind compute).
    # Rank-local only -- no collective is added or reordered. Set 0 to restore
    # the single-buffer serial-copy #136a behavior (A/B knob).
    SGLANG_WL_H2D_PREFETCH = EnvBool(True)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # #50 root fix: zero every flashinfer float workspace when a request
    # finishes. The fa2 split-KV kernels read workspace regions the current
    # forward did not write; on a fresh boot those read as first-touch zeros
    # (the contract the kernels were validated against), afterwards as the
    # previous request's partials — making outputs a function of the request
    # ordinal (greedy near-tie flips; degenerate fixed point under cuda
    # graphs). One ~384 MiB memset per finished request (~0.5 ms). Set 0 to
    # restore the old (nondeterministic-across-requests) behavior.
    SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST = EnvBool(True)
    # Enable NVFP4 per-token activation scaling path for FlashInfer TRT-LLM MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)
    # Chunk size for the rebalance expert-weight P2P exchange; set
    # >= num_physical_experts to submit a single batch_isend_irecv.
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # DeepSeek MHA Optimization
    # Deprecated (#395): a flat token count does not scale with per-rank head
    # count/head dim under (uneven) TP. Use --attn-scratch-budget-mib
    # (ServerArgs), which is a MiB scratch budget converted to a per-rank
    # token threshold at attention-layer init
    # (attention_forward_methods/forward_mha.py). Still honored verbatim,
    # with a deprecation warning, when explicitly set; mutually exclusive
    # with --attn-scratch-budget-mib.
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)  # This argument is deprecated
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    # Force dynamic DeepEP Waterfill with runtime EP all-reduce instead of the
    # default static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)

    # NIXL-EP
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)

    # DSA Backend (canonical names; fall back to SGLANG_NSA_* with deprecation warning)
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_FUSE_TOPK"
    )
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # Flash Attention
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)

    # Kernels
    # Force every sglang.kernels BaseFusedOp onto one backend (a KernelBackend
    # value, e.g. "torch" / "torch_compile" / "triton" / "cuda_aot"); unset =
    # auto-select by priority. "torch" flips all fused ops to their pure-torch
    # reference implementations for numerical-bug bisection.
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)
    # #656 T1: reserve the context ceiling's cos/sin rows instead of
    # materializing them. Default OFF -- the eager cache is the shipped
    # behaviour and this changes when memory is spent, which the KV sizer
    # reads. See layers/rotary_embedding/lazy_cos_sin_cache.py.
    SGLANG_ROPE_LAZY_CACHE = EnvBool(False)
    SGLANG_ROPE_LAZY_CHUNK_ROWS = EnvInt(65536)
    SGLANG_ROPE_LAZY_MIN_ROWS = EnvInt(262144)
    # Costs a device sync per batch. For proving the fill actually precedes
    # every position that is read, never for serving.
    SGLANG_ROPE_LAZY_VERIFY = EnvBool(False)

    # Overlap Spec V2
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Debug/stress: on every swap of an adaptive graph-memory state, all-gather
    # (swap ordinal, target steps) over the TP CPU group and assert equality.
    # Turns a rank-divergent swap (a #50-class bug) into an immediate failure
    # instead of silent corruption. Costs one gloo collective per swap (~0.1/s).
    SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC = EnvBool(False)
    # TEST-ONLY: force an adaptive runtime-state swap every N verify
    # completions, cycling through the candidate steps (rank-deterministic:
    # driven by the verify-call counter, identical on all ranks). Overrides the
    # EMA decision; use to stress the offload swap path, never in production.
    SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL = EnvInt(0)
    # Adaptive graph-memory offload: minimum free VRAM (MiB) that must remain
    # AFTER mapping the largest candidate state, enforced at boot
    # (finalize_boot). Covers eager-forward transient allocations (mamba
    # chunked-prefill recompute etc.) that run while a state is mapped.
    # Measured on the T102 rig: 148 MiB post-map free OOM'd at KV-full deep
    # prefill, 1367 MiB survived; 512 is the enforced floor between them.
    SGLANG_ADAPTIVE_SERVING_MARGIN_MIB = EnvInt(512)
    # Stage-2 graph-memory offload fallback: back the per-state CUDA-graph
    # capture pools up to host RAM on pause and restore the exact bytes on
    # resume, instead of relying on replay's rewrite-before-read property
    # over undefined resume content. Costs ~capture-pool-size of host RAM per
    # state and a PCIe round-trip per swap.
    SGLANG_ADAPTIVE_CAPTURE_CPU_BACKUP = EnvBool(False)
    # Kill-switch for the draft-extend cuda graph. Draft extend then always runs
    # eager. Escape hatch for setups where the capture's memory pool costs more
    # than the graph saves (e.g. DeepEP MoE workspace captured at full dispatch
    # capacity).
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)
    # Use the split-KV (flash-decode) kernel for EAGLE target-verify on the
    # Triton backend (ROCm). Only active at speculative topk == 1; falls back to
    # extend_attention_fwd for unsupported cases or when set false (e.g. for
    # debugging). Correctness is unaffected; this only changes performance.
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    # In-kernel slot-id bound check for the masked KV writers (#355). Unlike
    # SGLANG_ENABLE_ASYNC_ASSERT above this is DEFAULT ON and costs no extra
    # kernel launch: the compare runs in-register against a by-value bound the
    # writer already knows, the same mechanism store_cache uses since #352. Set
    # to 1 only to prove the check is what a slowdown is caused by -- with it
    # off, an out-of-range slot id corrupts KV silently again.
    SGLANG_DISABLE_KV_MASKED_BOUND_CHECK = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # Let the GPU-passive tokenizer process preprocess multimodal data on a
    # worker's card again (nvJPEG decode, fast-image-processor resize/normalize,
    # pinned video frames). Off by default since #403: the context it opens is
    # invisible to every per-rank memory budget and the tensors are copied back
    # to the host before they leave the process anyway. See
    # multimodal/processors/base_processor.mm_frontend_gpu_enabled.
    SGLANG_MM_FRONTEND_GPU_PREPROCESS = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)

    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Unified Radix Tree
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)

    # CUDA Graph
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
    # Guards CUDA graph executable dedup via cudaGraphExecUpdate.
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging.
    # Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1)

    # HTTP Server
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # Crash diagnostics
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # Encoder gRPC
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")

    # Native gRPC server. SGLANG_GRPC_PORT is the env fallback for the
    # --grpc-port CLI flag; setting either enables the native server alongside
    # HTTP. The worker-threads knob stays env-only (internal tuning, no CLI
    # surface).
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(False)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer (Kimi tiktoken: cache all_special_tokens / all_special_ids; the ITL can differ by +10x under high batch size).
    SGLANG_PATCH_TOKENIZER = EnvBool(True)

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    # ZBAL, zero buffer accelerate library, currently worked only in npu
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ====================================================================
    # DeepSeek V4
    SGLANG_OPT_DPSK_V4_RADIX = EnvBool(True)
    SGLANG_OPT_USE_OLD_COMPRESSOR = EnvBool(False)
    SGLANG_OPT_USE_TRITON_SWA_PREPARE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS_TRITON = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL = EnvBool(True)
    SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION = EnvBool(True)
    SGLANG_FIX_MTP_HC_HIDDEN = EnvBool(False)
    # ====================================================================

    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    # Default reasoning_effort for dsv4 chat encoder when request doesn't set it.
    # Accepts "", "max", "high" (empty string means unset); other values filtered to None.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")

    # CUDA kernels
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_DSV4_MHC_PREWARM = EnvBool(True)
    SGLANG_OPT_USE_TRITON_FUSED_MHC = EnvBool(True)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(False)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    # Per-rank local query rows (after DP-attention sharding when enabled),
    # not request ISL.
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    # Deprecated: DSV4 compressor V2 is always used.
    SGLANG_OPT_USE_COMPRESSOR_V2 = EnvBool(True)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    # Sequence-axis chunk (in KV positions) of the torch paged-MQA-logits
    # implementation. Bounds its peak intermediate at O(batch x chunk x heads)
    # instead of O(batch x context x heads); see #426 / upstream #33246. Must
    # be a multiple of the 64-position page; 0 disables chunking (one pass over
    # the whole sequence, the pre-#426 shape).
    SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK = EnvInt(8192)
    # Query-axis chunk of the same implementation, expressed as a per-rank MiB
    # budget for the transient working set of ONE (query chunk x KV chunk)
    # step -- not a row count, because the bytes a query row costs scale with
    # the head count and with the KV chunk width, so a flat row count is not
    # comparable across geometries (same reasoning as --attn-scratch-budget-mib,
    # #395). Bounds the per-query-token duplication of the KV gather described
    # in ANALYSE_447 section 2.3 L1. 0 disables it (one pass over the whole
    # query axis, the pre-#449 shape). See #449.
    #
    # THE DEFAULT MUST BIND (#493). #449 shipped 2048 MiB and NOTE_449 section 5
    # names it for what it was: "a ceiling picked at desk, not a tuned value".
    # It is above the peak it was meant to bound on the geometry this fork
    # actually serves, so the cap was inert. On the DeepSeek-V4-Flash C4 indexer
    # (index_n_heads=64, index_head_dim=128, heads replicated) one query row
    # costs `chunk_seq * 1160` bytes, so at --chunked-prefill-size 256:
    #   SEQ_CHUNK 2048 -> 2.27 MiB/row -> 580 MiB for 256 rows; 2048 MiB permits
    #                     903 rows, i.e. it never binds;
    #   SEQ_CHUNK 8192 (this file's default) -> 9.06 MiB/row -> 2320 MiB for 256
    #                     rows; 2048 MiB permits 225 rows, i.e. it trims 12 %.
    # Window 3 of 2026-08-03 measured what that costs: both 3080 ranks fell from
    # 873 MiB free to 271 MiB during the deep DSV4F prefill, a 602 MiB excursion
    # (a LOWER bound -- the sampler ran at 1 Hz against a sub-second transient),
    # breaching the 400 MiB corridor floor on 214 samples. The model for that
    # run is 588 MiB: 580 MiB of loop step (256 rows at SEQ_CHUNK 2048) plus
    # 8 MiB of returned logits at the C4 indexer span of 8196 -- the indexer
    # runs on the compress_ratio-4 span, not on the 32768-token prompt. Raising
    # --rank-auto-reserve-mib by 500 MiB did not move that floor: the reserve
    # forms the rank BUDGET and cannot cap a runtime allocation. Only this knob
    # caps it.
    # 256 MiB is chosen as the largest power-of-two budget that still binds on
    # the reference geometry above at both SEQ_CHUNK settings, and it leaves the
    # corridor intact on the same run: 112 rows x 2.27 MiB = 254 MiB of step
    # plus 8 MiB of logits, i.e. 873 - 262 = 611 MiB free at peak. The
    # regrouping it forces is exact -- no reduction crosses a chunk boundary --
    # so this buys corridor without giving up any KV capacity.
    SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = EnvInt(256)
    SGLANG_TOPK_TRANSFORM_512_TORCH = EnvBool(False)
    # Validate the non-negative-seq_len precondition of the DSV4 top-k
    # wrappers (v1 and v2) before the launch. The check costs a device-to-host
    # sync per call, so it is off on the serving path and meant for bring-up of
    # a new producer of `seq_lens` (DP-idle companion rows, padded MTP rows).
    # See #427 F2 and the docstrings in `sglang.jit_kernel.dsv4.topk`.
    SGLANG_DSV4_CHECK_TOPK_SEQ_LENS = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)

    # SWA radix cache
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)

    # Unified radix cache
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(False)

    # DeepGemm Mega MoE
    SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)

    # When set, the mega-MoE x slot is packed E2M1 (FP4) instead of FP8 E4M3.
    # Halves symm-buffer footprint and unlocks the MXF4 mainloop downstream.
    # Setting this also exports DG_USE_FP4_ACTS=1 so DeepGEMM's symm-buffer
    # sizing + fp8_fp4_mega_moe pick up the FP4 layout.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS = EnvBool(False)
    # Switches the L1+L2 mainloops from kind::mxf8f6f4 (K=32 with-padding) to
    # kind::mxf4 (K=64 dense) inside fp8_fp4_mega_moe. No effect unless
    # SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is also set; DeepGEMM asserts
    # this combination on the host side.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND = EnvBool(False)
    SGLANG_OPT_FIX_MEGA_MOE_MEMORY = EnvBool(False)

    # TopK
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK = EnvBool(True)
    # Opt-in: route DeepSeek-V3 grouped topk through the unified Triton router
    # instead of the flashinfer/AOT grouped kernels. Off by default (flashinfer is
    # the tuned production path); the Triton path is bit-exact on DeepSeek-V3.2 e2e
    # and benchmarks at parity, so this is a consolidation escape hatch, not a perf flip.
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # Reroutes the generic fp8 per-token-group quant (every model, not just MiniMax)
    # to the V1 JIT kernel. Off by default; V1 is byte-identical to V2.
    SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT = EnvBool(False)
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)

    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)

    # Fused JIT store (minimax_store_kv_index) of main+index K/V instead of separate
    # set_*_buffer copies; falls back when main/index dtypes differ or non-CUDA.
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(True)

    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)

    # GEMM / kernel fusion
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_USE_JIT_EP_ACTIVATION = EnvBool(True)
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_SWIGLU_CLAMP_FUSION = EnvBool(True)

    # Cache / overlap
    SGLANG_OPT_USE_FUSED_STORE_CACHE = EnvBool(True)
    SGLANG_OPT_USE_JIT_NORM = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # CUDA graph
    SGLANG_PREP_IN_CUDA_GRAPH = EnvBool(True)

    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)

    # Distributed
    SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER = EnvBool(True)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)
    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # Elastic EP Backup Port
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # Sglang Cache Dir
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(False)

    # Plugin system
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")

    # GGUF loader
    # #391: repack GGUF MXFP4 (ggml type 39) tensors to Q5_0 while reading the
    # weight stream. The MXFP4 lattice is a subset of Q5_0's, so the conversion
    # is value-exact (every element dequantizes to the same fp32 number) and it
    # turns a type no GGUF kernel dispatches on into one every GGUF kernel
    # dispatches on. It costs 22/17 = 1.294x the bytes of the repacked tensors,
    # in host RAM and in VRAM, and that inflation is logged once at load time.
    # Set to False to refuse MXFP4 loudly instead, which is what this build did
    # before the repack existed. There is no silent middle ground.
    SGLANG_GGUF_MXFP4_REPACK = EnvBool(True)

    # #391: release the page cache behind the GGUF weight stream. gguf-py's
    # reader maps every part with np.memmap, so reading the stream faults the
    # whole checkpoint into the page cache and never gives it back -- ~48+ GiB
    # of clean file pages competing with the loader's own anonymous memory on a
    # swapless box (boot attempt 8 died exactly there: memory.current pinned at
    # 98.3 of 98.5 GiB while the kernel traded file pages for anon one for one).
    # With this on, each consumed region is madvise(MADV_DONTNEED)'d and then
    # posix_fadvise(POSIX_FADV_DONTNEED)'d as the stream advances. Read-only
    # shared mapping of an unmodified file: an advised range re-faults the same
    # bytes, so the streamed bytes are identical either way. Set to False to
    # restore the pre-#391 accumulation.
    SGLANG_GGUF_STREAM_DROP_CACHE = EnvBool(True)
    # Synchronous cgroup reclaim during the GGUF stream, in GiB of
    # memory.current. 0 (default) = off, behaviour byte-identical to before.
    # The dropper only releases page cache BEHIND the consumer while the
    # kernel reads AHEAD of it, and on a swapless box that gap is the whole
    # budget (#391). An external sampler chasing it on a wall-clock interval
    # can be outrun -- window 3 saw memory.current move 88 -> 102 GiB inside
    # one 15 s window. Reclaiming here instead ties the trim RATE to consumer
    # PROGRESS, which is what generates the pressure in the first place.
    SGLANG_GGUF_STREAM_TRIM_SOFT_GIB = EnvFloat(0.0)
    #: Reclaim down to about here once the soft watermark is crossed.
    SGLANG_GGUF_STREAM_TRIM_TARGET_GIB = EnvFloat(0.0)
    # Slack above the UNRECLAIMABLE floor (#537). The trim's target is raised
    # to `anon + pinned host pool + this` whenever that sits above the
    # configured target, because cgroup reclaim cannot take either term --
    # CUDA pinned host memory is filed under `file`, not `anon`, so
    # memory.current hides it (49.66 GiB of pool against anon 14.6 GiB,
    # measured 2026-08-04). This term buys the loader's own read-ahead room
    # inside that budget. 0.0 (default) = no slack, i.e. the trim is allowed to
    # drive page cache down to the floor exactly as it did before #537;
    # calibrating it needs a load-time page-cache measurement, not a desk
    # number, so it ships INERT and pinned as such in
    # test_bounding_default_value_pins.py.
    SGLANG_GGUF_STREAM_TRIM_HEADROOM_GIB = EnvFloat(0.0)

    # ===================================================================
    # KV-Canary / Token-Oracle (testing-only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)


envs = Envs()
EnvField._allow_set_name = False


def _print_deprecated_env(old_name: str, new_name: Optional[str] = None):
    if old_name in os.environ:
        if new_name is None:
            warnings.warn(f"Environment variable {old_name} has been deprecated.")
        else:
            warnings.warn(
                f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
            )
            os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_GC_LOG", "SGLANG_LOG_GC")
    _print_deprecated_env(
        "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH", "SGLANG_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
    )
    _print_deprecated_env("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")
    _print_deprecated_env("SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN")
    _print_deprecated_env("SGLANG_ENABLE_THINKING", "SGLANG_DEFAULT_THINKING")
    _print_deprecated_env("SGLANG_REASONING_EFFORT", "SGLANG_DSV4_REASONING_EFFORT")
    _print_deprecated_env(
        "SGLANG_USE_JIT_ALL_REDUCE", "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()
_warn_deprecated_env_to_cli_flag(
    "SGLANG_ENABLE_GRPC",
    "Please use '--grpc-port' to enable the native gRPC server.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_DFLASH_PREFILL_REFILL_TARGET",
    "DFlash now auto-enables the min-free-slots delay; unset this env. To "
    "override the threshold, use '--min-free-slots-delay'.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError("assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
