# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``Class2DiffusionAdapter``: a ``multimodal_gen`` diffusion tenant (#333 §5).

M1 shipped this adapter as an estimator that refused to launch. M3 promotes it
to a launching adapter: it boots the vendored SGLang-Diffusion server as a
child process, pins it to its cards, drives it along the residency ladder using
the server's own ``/release_memory_occupation`` and ``/resume_memory_occupation``
endpoints, and reports what it holds through NVML like every other tenant.

What M3 does NOT do is rewrite the diffusion forward pass. The adapter wraps the
upstream server (§5.1's decision: adapt, do not reimplement) and touches exactly
the seam §5.4 allows. The one distributed fact the adapter contributes is the
per-rank capacity-weight vector for the uneven sequence-parallel split
(``build_shard_plan`` in ``multimodal_gen``'s ``sp_shard_utils``): it is computed
from the SAME per-card GEMM rates the K1 uneven-TP planner uses and handed to the
child through ``SGLANG_SP_CAPACITY_WEIGHTS``. The plan geometry is built and
CPU-proven; wiring the plan into every collective and attention-meta path in the
diffusion runtime is the M4 seam (linear.py / vocab_parallel_embedding.py /
all_gather_v), so uneven SP is gated behind ``launch.enable_uneven_sp`` and the
default single-card / equal-split path is the one M3 validates.

Ladder (§5.3), the best-served of the three classes because the upstream server
already implements host offload:

===========  ==========================================================
``HOT``      process up, DiT / encoders / VAE resident, graphs captured
``WARM_HOST`` weights pushed to host RAM via /release_memory_occupation;
             promotion back is /resume_memory_occupation, no reload
``COLD``     no process. Weights are on disk in their original form, so a
             cold promotion is a normal load -- no hibernate parking needed
===========  ==========================================================

``WARM_GPU`` (drop the breakable-CUDA-graph pool and VAE tiling buffers while
keeping DiT weights on the device) is a real rung §5.3 names but the upstream
server exposes no endpoint for it today, so it is refused rather than faked.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from sglang.srt.registry.adapter import (
    AdapterContext,
    AdapterError,
    EstimateError,
    Health,
    register_adapter,
)
from sglang.srt.registry.adapters.process import (
    ChildProcess,
    ProcessTenantError,
    http_json,
    http_ok,
    wait_for,
)
from sglang.srt.registry.ledger import MIB
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
)

logger = logging.getLogger(__name__)

ADAPTER_NAME = "class2_diffusion"

#: A cold diffusion load plus a graph-capture warmup is minutes; the timeout is
#: generous for the same reason the Class-1 one is (§4.3): tearing down a server
#: that was about to be ready is the worse failure.
DEFAULT_BOOT_TIMEOUT_S = 900.0
DEFAULT_READY_POLL_S = 1.0
DEFAULT_PORT = 30000

#: The child env var the uneven sequence-parallel split reads
#: (``multimodal_gen.runtime.distributed.sp_shard_utils``). Set only when the
#: tenant spans more than one card AND the spec opts into uneven SP.
CAPACITY_WEIGHTS_ENV = "SGLANG_SP_CAPACITY_WEIGHTS"

#: §5.2 memory posts. Every one must be declared: the registry does not model a
#: diffusion activation peak, and guessing one is how a card goes over the
#: corridor at the first full-resolution request instead of at plan time.
REQUIRED_POSTS = (
    "weights_bytes",
    "activation_peak_bytes",
    "latent_bytes",
    "text_encoder_bytes",
    "vae_bytes",
    "ctx_overhead_bytes",
)

#: Posts that are replicated on every SP rank rather than split across them.
#: Sequence parallelism shards the sequence/activations, not the weights: every
#: rank holds the full DiT, encoders and VAE. Only the activation peak scales
#: with a rank's sequence share.
_REPLICATED_POSTS = (
    "weights_bytes",
    "text_encoder_bytes",
    "vae_bytes",
    "ctx_overhead_bytes",
)


class Class2DiffusionAdapter:
    """One ``multimodal_gen`` server process, driven along the ladder."""

    klass = int(EngineClass.DIFFUSION)

    def __init__(self, spec: EngineSpec, context: AdapterContext) -> None:
        self.spec = spec
        self.context = context
        self.launch: Mapping[str, Any] = dict(spec.launch)
        self._state = ResidencyState.COLD
        self._process: ChildProcess | None = None
        self._cards: tuple[str, ...] = ()
        self._last_error: str | None = None
        posts = self.launch.get("posts_mib")
        if not posts:
            raise EstimateError(
                f"engine {spec.engine_id!r}: a Class-2 spec must declare "
                f"launch.posts_mib with the §5.2 posts {list(REQUIRED_POSTS)}; the "
                "registry does not model diffusion and will not invent a peak"
            )
        missing = [p for p in REQUIRED_POSTS if p not in posts]
        if missing:
            raise EstimateError(
                f"engine {spec.engine_id!r}: launch.posts_mib is missing {missing}. "
                "Declare 0 for a post this configuration genuinely does not have, so "
                "the zero is a statement rather than an omission."
            )
        self._posts = {str(k): int(v) * MIB for k, v in posts.items()}
        # model_path is required to BOOT, not to estimate. Registration and
        # planning (§7.4 "validate without booting") must work from posts alone,
        # so the check lives in _boot, not here.

    # -- configuration -----------------------------------------------------

    @property
    def enable_uneven_sp(self) -> bool:
        return bool(self.launch.get("enable_uneven_sp", False))

    def _placed_cards(self) -> tuple[str, ...]:
        return self._cards or tuple(self.spec.placement)

    # -- planning ----------------------------------------------------------

    def estimate(self, spec: EngineSpec, cards: tuple[str, ...]) -> ResourceProfile:
        if len(cards) == 1:
            return self._single_card_profile(cards[0])
        if not self.enable_uneven_sp:
            raise EstimateError(
                f"engine {spec.engine_id!r}: multi-card diffusion needs "
                "launch.enable_uneven_sp. The uneven sequence-parallel plan is built "
                "and CPU-proven (build_shard_plan), but wiring it through every "
                "collective and attention-meta path in the diffusion runtime is the "
                "M4 seam (§5.4); until then a multi-card tenant is opt-in."
            )
        return self._multi_card_profile(cards)

    def _single_card_profile(self, card: str) -> ResourceProfile:
        peak = sum(self._posts.values())
        # Steady = peak minus the activation transient: between denoise steps the
        # tenant holds weights, latents and encoders, not the peak. The
        # reservation is still the peak (§3.8); the difference is declared slack a
        # #330 report names, not a byte the ledger may hand to a neighbour.
        steady = peak - self._posts.get("activation_peak_bytes", 0)
        return ResourceProfile(
            posts={card: dict(self._posts)},
            peak_bytes={card: peak},
            steady_bytes={card: steady},
            notes=("single-card diffusion tenant (§5.2 posts).",),
        )

    def _multi_card_profile(self, cards: tuple[str, ...]) -> ResourceProfile:
        # Sequence parallelism replicates the weights on every rank and splits
        # only the activation peak. The per-card activation share follows the
        # capacity weights; a heavier card owns a longer sequence and thus a
        # larger peak. Reserving each card its own share is exact rather than
        # conservative -- the corridor is a per-card property (§3.3).
        weights = self._capacity_weights(cards)
        total_w = sum(weights)
        replicated = sum(self._posts.get(p, 0) for p in _REPLICATED_POSTS)
        activation = self._posts.get("activation_peak_bytes", 0)
        latent = self._posts.get("latent_bytes", 0)
        posts: dict[str, dict[str, int]] = {}
        peak: dict[str, int] = {}
        steady: dict[str, int] = {}
        for card, w in zip(cards, weights):
            share = w / total_w
            card_activation = int(round(activation * share))
            card_latent = int(round(latent * share))
            card_posts = {p: self._posts.get(p, 0) for p in _REPLICATED_POSTS}
            card_posts["activation_peak_bytes"] = card_activation
            card_posts["latent_bytes"] = card_latent
            posts[card] = card_posts
            peak[card] = replicated + card_activation + card_latent
            steady[card] = replicated + card_latent
        return ResourceProfile(
            posts=posts,
            peak_bytes=peak,
            steady_bytes=steady,
            notes=(
                "uneven sequence-parallel diffusion tenant: weights replicated per "
                f"rank, activation split by capacity weights {tuple(round(w, 3) for w in weights)}.",
            ),
        )

    def _capacity_weights(self, cards: Sequence[str]) -> tuple[float, ...]:
        """Per-card capacity weights from the measured GEMM rates.

        Reuses the K1 uneven-TP rate source through the shared cost library
        (#348b): the boot-fingerprinted hardware profile, resolved per card by
        ``uneven_perf.rank_gemm_scores``. The diffusion DiT is a dense bf16/fp16
        transformer, so ``FORMAT_DENSE_BF16`` is the right format key -- it is a
        real lane-table entry, so the dense GEMM probe is selected on purpose
        and not as a fallback. A faster card earns a proportionally longer slice
        of the sequence.

        Until #348b this function read ``load_measured_registry`` instead. That
        is the measured KV-BUDGET registry: gated behind
        ``SGLANG_MEASURED_KV_BUDGET``, keyed by ``components``/``mlp_vector``
        rather than by card UUID, and typed for a ``ServerArgs`` this caller
        does not have. It could never return a per-card ``gemm_tflops``, so the
        measured branch raised on every call and uneven SP was reachable only
        by declaring ``launch.capacity_weights`` by hand -- while the docstring
        claimed the K1 rates were in use. No test covered it.
        """
        declared = self.launch.get("capacity_weights")
        if declared:
            if len(declared) != len(cards):
                raise EstimateError(
                    f"engine {self.spec.engine_id!r}: launch.capacity_weights has "
                    f"{len(declared)} entries but {len(cards)} cards are placed"
                )
            return tuple(float(w) for w in declared)
        try:
            from sglang.srt.planner import cost_model  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise EstimateError(
                f"engine {self.spec.engine_id!r}: uneven SP needs the measured GEMM "
                f"profile, which could not be loaded ({exc}). Run the rig probe, or "
                "declare launch.capacity_weights explicitly."
            ) from None
        rates = cost_model.compute_rates_for_cards(
            cards, fmt=cost_model.FORMAT_DENSE_BF16
        )
        absences = rates.absences()
        if absences:
            raise EstimateError(
                f"engine {self.spec.engine_id!r}: no measured GEMM rate for "
                f"{len(absences)} of {len(cards)} placed cards. "
                + " ".join(absences[:2])
                + " Declare launch.capacity_weights explicitly, or probe the rig "
                "first."
            )
        return tuple(rates.values())

    def bind(self, cards: tuple[str, ...]) -> None:
        self._cards = tuple(cards)

    # -- residency ---------------------------------------------------------

    def state(self) -> ResidencyState:
        return self._state

    def pids(self) -> tuple[int, ...]:
        return self._process.child_pids() if self._process is not None else ()

    def promote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_GPU:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 2 exposes no WARM_GPU endpoint "
                "today (§5.3 names the rung; the upstream server has no route to drop "
                "just the BCG pool and VAE tiling). The ladder here is "
                "HOT / WARM_HOST / COLD."
            )
        if target == self._state:
            return
        if target == ResidencyState.COLD:
            raise AdapterError("COLD is a demotion, not a promotion")
        if self._state == ResidencyState.COLD:
            self._boot()
            self._state = ResidencyState.HOT
            if target == ResidencyState.WARM_HOST:
                self._release()
                self._state = ResidencyState.WARM_HOST
            return
        if self._state == ResidencyState.WARM_HOST and target == ResidencyState.HOT:
            self._resume()
            self._state = ResidencyState.HOT
            return
        raise AdapterError(
            f"engine {self.spec.engine_id!r}: no promotion path "
            f"{self._state.value} -> {target.value}"
        )

    def demote(self, target: ResidencyState) -> None:
        if target == ResidencyState.WARM_GPU:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: Class 2 has no WARM_GPU rung here "
                "(§5.3); the ladder is HOT / WARM_HOST / COLD"
            )
        if target == self._state:
            return
        if target == ResidencyState.WARM_HOST:
            if self._state != ResidencyState.HOT:
                raise AdapterError(
                    f"engine {self.spec.engine_id!r}: cannot demote "
                    f"{self._state.value} -> WARM_HOST"
                )
            self._release()
            self._state = ResidencyState.WARM_HOST
            return
        if target == ResidencyState.COLD:
            if self._process is not None:
                self._process.stop()
                self._process = None
            self._state = ResidencyState.COLD
            return
        raise AdapterError(f"unsupported demotion target {target.value}")

    # -- process control ---------------------------------------------------

    @property
    def port(self) -> int:
        return int(self.launch.get("port", DEFAULT_PORT))

    @property
    def base_url(self) -> str:
        host = self.launch.get("host", "127.0.0.1")
        return f"http://{host}:{self.port}"

    def visible_devices(self) -> str:
        return ",".join(self._placed_cards())

    def num_gpus(self) -> int:
        return len(self._placed_cards())

    def build_argv(self) -> list[str]:
        """The launch command for the vendored SGLang-Diffusion server.

        Explicit ``-m`` invocation of the diffusion CLI rather than the shared
        ``sglang serve`` dispatcher, so the command is reproducible by hand and
        cannot be misrouted to the LLM path by model-type autodetection.
        """
        launch = self.launch
        argv = [
            sys.executable,
            "-m",
            "sglang.multimodal_gen.runtime.entrypoints.cli.main",
            "serve",
            "--model-path",
            str(launch["model_path"]),
            "--host",
            str(launch.get("host", "127.0.0.1")),
            "--port",
            str(self.port),
            "--num-gpus",
            str(self.num_gpus()),
        ]
        if self.num_gpus() > 1:
            # Pure sequence parallelism across the placed cards.
            argv += ["--ulysses-degree", str(self.num_gpus())]
        extra = launch.get("extra_args") or []
        if isinstance(extra, str):
            extra = shlex.split(extra)
        argv += [str(a) for a in extra]
        return argv

    def _child_env(self) -> dict[str, str]:
        # One physical GPU set per process, named by UUID: inside the tenant
        # cuda:0..N-1 map onto exactly the placed cards, in placement order, and
        # no logical-to-physical table can go wrong (§ Device identity).
        env = {"CUDA_VISIBLE_DEVICES": self.visible_devices()}
        if self.num_gpus() > 1 and self.enable_uneven_sp:
            weights = self._capacity_weights(self._placed_cards())
            # The child's build_shard_plan reads this and hands the faster card a
            # proportionally longer sequence slice. Absent, the split stays equal.
            env[CAPACITY_WEIGHTS_ENV] = ",".join(f"{w:.6g}" for w in weights)
            logger.info(
                "registry: engine %s uneven SP capacity weights %s",
                self.spec.engine_id,
                env[CAPACITY_WEIGHTS_ENV],
            )
        env.update(self.launch.get("env") or {})
        return env

    def _boot(self) -> None:
        log_path = None
        if self.context.work_dir:
            log_path = os.path.join(
                self.context.work_dir, f"{self.spec.engine_id}.server.log"
            )
        # §3.6: the diffusion warmup captures breakable CUDA graphs, which is a
        # capture on this card. Take the arbiter's per-card exclusive lock around
        # the boot so co-located tenants are quiesced during capture, exactly as
        # the capture-lock rule requires. Best effort: an arbiter that supplied no
        # lock (a standalone boot) simply proceeds.
        lock = self.context.card_exclusive_lock
        acquired = None
        if lock is not None:
            try:
                acquired = lock(self._placed_cards())
                if hasattr(acquired, "__enter__"):
                    acquired.__enter__()
            except Exception as exc:  # noqa: BLE001 - lock is advisory here
                logger.debug("registry: capture lock unavailable: %s", exc)
                acquired = None
        try:
            self._start_and_wait(log_path)
        finally:
            if acquired is not None and hasattr(acquired, "__exit__"):
                acquired.__exit__(None, None, None)

    def _start_and_wait(self, log_path: str | None) -> None:
        if not self.launch.get("model_path"):
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: launch.model_path is required to boot "
                "the diffusion server"
            )
        self._process = ChildProcess(
            argv=self.build_argv(), env=self._child_env(), log_path=log_path
        )
        self._process.start()
        timeout = float(self.launch.get("boot_timeout_s", DEFAULT_BOOT_TIMEOUT_S))
        try:
            wait_for(
                lambda: http_ok(f"{self.base_url}/health"),
                timeout_s=timeout,
                poll_s=DEFAULT_READY_POLL_S,
                on_dead=lambda: (
                    None
                    if self._process is None or self._process.running
                    else f"exit code {self._process.returncode}"
                    + (f", log {log_path}" if log_path else "")
                ),
                what=f"engine {self.spec.engine_id!r} to answer /health",
            )
        except ProcessTenantError as exc:
            self._last_error = str(exc)
            self._process.stop()
            self._process = None
            raise AdapterError(str(exc)) from None
        self._last_error = None

    def _release(self) -> None:
        self._post_memory("release_memory_occupation")

    def _resume(self) -> None:
        self._post_memory("resume_memory_occupation")

    def _post_memory(self, endpoint: str) -> None:
        if self._process is None:
            raise AdapterError(
                f"engine {self.spec.engine_id!r} has no process to {endpoint}"
            )
        try:
            http_json(
                f"{self.base_url}/{endpoint}",
                method="POST",
                timeout=float(self.launch.get("memory_op_timeout_s", 300.0)),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an adapter error
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: {endpoint} failed: {exc}"
            ) from None

    # -- observability -----------------------------------------------------

    def measured(self) -> Mapping[str, int]:
        pids = set(self.pids())
        if not pids or not self._cards:
            return {}
        from sglang.srt.registry.nvml import process_bytes_on_uuid  # noqa: PLC0415

        out: dict[str, int] = {}
        for card in self._cards:
            try:
                per_pid = process_bytes_on_uuid(card)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.debug("registry: measured() on %s failed: %s", card, exc)
                continue
            out[card] = sum(b for pid, b in per_pid.items() if pid in pids)
        return out

    def health(self) -> Health:
        if self._state == ResidencyState.COLD:
            return Health(ok=True, detail="cold, no process")
        if self._process is None or not self._process.running:
            return Health(ok=False, detail=self._last_error or "process is gone")
        if self._state == ResidencyState.WARM_HOST:
            # A host-offloaded server still answers /health; the process is up.
            return Health(ok=http_ok(f"{self.base_url}/health"), detail="warm host")
        if http_ok(f"{self.base_url}/health"):
            return Health(ok=True, detail=f"serving on {self.base_url}")
        return Health(ok=False, detail=f"no /health answer from {self.base_url}")

    def image_probe(
        self, prompt: str = "a red cube on a table", size: str = "512x512"
    ) -> int:
        """Generate one image; return the number of images returned.

        Used by the acceptance harness, not by policy. Routes to the same
        ``/v1/images/generations`` endpoint #335-M0's serving surface targets.
        """
        started = time.monotonic()
        result = http_json(
            f"{self.base_url}/v1/images/generations",
            method="POST",
            payload={
                "model": self.launch.get("served_model_name", "diffusion"),
                "prompt": prompt,
                "size": size,
                "n": 1,
            },
            timeout=600.0,
        )
        logger.info(
            "registry: image probe on %s took %.0f ms",
            self.spec.engine_id,
            (time.monotonic() - started) * 1000.0,
        )
        if isinstance(result, dict) and result.get("data"):
            return len(result["data"])
        raise AdapterError(f"unexpected image response: {result!r}")


register_adapter(ADAPTER_NAME, Class2DiffusionAdapter)
