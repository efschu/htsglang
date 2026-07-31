#!/usr/bin/env python3
"""#333-M1 card-window harness: two Class-1 engines, one card, one hot switch.

What this proves, in one window:

1. Two engines are registered against the real ledger. Neither boots.
2. The derived M (§7.2) says only one of them fits hot on this card -- a
   statement in bytes, not a configured count.
3. Engine A is promoted, serves, and is measured.
4. Engine B is requested. The arbiter evicts A (hibernating it to disk, #89),
   boots B, and B serves. The promotion cost is measured, not assumed.
5. Engine A is requested again. It is now restored from its own hibernate
   manifest rather than from the checkpoint, and the two numbers are printed
   side by side. That difference is the point of the ladder.
6. A third engine that cannot fit is rejected with the projected wait and the
   eviction that would have been needed.

An NVML sampler runs independently of the registry for the whole window and
records the minimum free bytes per card, because the corridor is a statement
about the driver's free memory and the ledger cannot verify itself.

    python scripts/registry/m1_card_window.py --out /tmp/m1-window
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.registry.arbiter import (  # noqa: E402
    EngineRegistry,
    PromotionRejected,
    RegistrationRejected,
    card_totals_from_nvml,
    free_bytes_from_nvml,
)
from sglang.srt.registry.ledger import MIB, ReservationStore  # noqa: E402
from sglang.srt.registry.nvml import list_devices, memory_info_for_uuid  # noqa: E402
from sglang.srt.registry.spec import (  # noqa: E402
    EngineClass,
    EngineSpec,
    ResidencyState,
)

MODEL_ROOT = Path("/spinning/llm_stuff/club-3090/models-cache")


class NvmlSampler(threading.Thread):
    """Free bytes per card, sampled outside the registry's own accounting."""

    def __init__(self, cards, period_s=1.0):
        super().__init__(daemon=True)
        self.cards = list(cards)
        self.period_s = period_s
        self.samples: list[dict] = []
        # Not ``_stop``: threading.Thread already owns that name internally
        # and shadowing it breaks join().
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self.period_s):
            row = {"ts": time.time()}
            for card in self.cards:
                try:
                    row[card] = memory_info_for_uuid(card).free_bytes
                except Exception:  # noqa: BLE001 - a lost sample is not a failure
                    row[card] = None
            self.samples.append(row)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    def minimum_free(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for card in self.cards:
            values = [s[card] for s in self.samples if s.get(card) is not None]
            out[card] = min(values) if values else None
        return out


def engine_spec(engine_id, gguf, tokenizer_dir, card, port, budget_mib, hibernate_dir):
    return EngineSpec(
        engine_id=engine_id,
        klass=EngineClass.AUTOREGRESSIVE,
        adapter="class1_srt",
        placement=(card,),
        launch={
            "model_path": str(gguf),
            "port": port,
            "tp_size": 1,
            "rank_gpu_memory_mib": budget_mib,
            "hibernate_dir": str(hibernate_dir),
            "boot_timeout_s": 600.0,
            "extra_args": [
                "--tokenizer-path",
                str(tokenizer_dir),
                "--context-length",
                "4096",
                "--max-running-requests",
                "2",
                "--trust-remote-code",
                "--attention-backend",
                "flashinfer",
            ],
        },
    )


def transition(registry, engine_id, target, log):
    started = time.monotonic()
    registry.ensure_state(engine_id, target)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    log.append(
        {
            "engine_id": engine_id,
            "target": target.value,
            "wall_ms": round(elapsed_ms, 1),
            "registry_promotion_cost_ms": registry.instance(
                engine_id
            ).promotion_cost_ms,
        }
    )
    print(f"  {engine_id} -> {target.value}: {elapsed_ms / 1000.0:.1f} s", flush=True)
    return elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/m1-window")
    parser.add_argument("--budget-mib", type=int, default=20_000)
    parser.add_argument("--port-a", type=int, default=31_501)
    parser.add_argument("--port-b", type=int, default=31_502)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.time(), "transitions": []}

    devices = list_devices()
    for device in devices:
        print(device.describe(), flush=True)
    report["cards"] = [
        {"index": d.index, "uuid": d.uuid, "name": d.name, "total_mib": d.total_mib}
        for d in devices
    ]
    # The biggest card, resolved at runtime. Never a hard-coded index: NVML
    # enumeration order shifts between boots and driver states.
    target_card = max(devices, key=lambda d: d.total_bytes)
    print(f"\ntarget card: {target_card.describe()}\n", flush=True)
    report["target_card"] = target_card.uuid

    totals = card_totals_from_nvml()
    store = ReservationStore(out / "ledger")
    registry = EngineRegistry(
        store=store,
        card_totals=totals,
        work_dir=str(out),
        free_bytes_resolver=free_bytes_from_nvml,
    )

    sampler = NvmlSampler([d.uuid for d in devices])
    sampler.start()

    # Two different GGUF checkpoints, both small enough that a full window
    # fits inside the card booking, both #89-eligible (hibernate is
    # GGUF-scoped upstream, which is why the pair is GGUF and not the
    # safetensors Qwen3.5-2B / Llama-8B).
    specs = {
        "qwen35-9b": engine_spec(
            "qwen35-9b",
            MODEL_ROOT / "Qwen3.5-9B-GGUF" / "Qwen3.5-9B-Q4_K_M.gguf",
            MODEL_ROOT / "Qwen3.5-9B-GGUF",
            target_card.uuid,
            args.port_a,
            args.budget_mib,
            out / "hibernate-qwen35-9b",
        ),
        "qwen36-27b": engine_spec(
            "qwen36-27b",
            MODEL_ROOT / "Qwen3.6-27B-MTP-Q3_K_M-GGUF" / "Qwen3.6-27B-Q3_K_M.gguf",
            MODEL_ROOT / "Qwen3.6-27B-MTP-Q3_K_M-GGUF",
            target_card.uuid,
            args.port_b,
            args.budget_mib,
            out / "hibernate-qwen36-27b",
        ),
    }

    try:
        print("== 1. register (no boot) ==", flush=True)
        for engine_id, spec in specs.items():
            plan = registry.register(spec)
            print(
                f"  {engine_id}: reserved {plan.profile.total_peak_bytes // MIB} MiB, "
                f"fits now={plan.fits}",
                flush=True,
            )
        report["registration"] = {e: registry.instance(e).to_json() for e in specs}
        assert all(
            registry.instance(e).state is ResidencyState.COLD for e in specs
        ), "registration booted something"

        print("\n== 2. derived M ==", flush=True)
        capacity = registry.hot_capacity()
        print(json.dumps(capacity.to_json(), indent=2), flush=True)
        report["hot_capacity"] = capacity.to_json()

        print("\n== 3. promote engine A, serve ==", flush=True)
        transition(registry, "qwen35-9b", ResidencyState.HOT, report["transitions"])
        text_a = registry.adapter("qwen35-9b").generate_probe(
            "The capital of France is"
        )
        print(f"  engine A says: {text_a!r}", flush=True)
        registry.refresh_measured()
        report["ledger_after_a"] = [c.to_json() for c in registry.cards()]
        print(json.dumps(report["ledger_after_a"], indent=2), flush=True)

        print(
            "\n== 4. informative rejection of an engine that cannot fit ==", flush=True
        )
        rejections = {}
        too_big = EngineSpec(
            engine_id="oversized",
            klass=EngineClass.AUTOREGRESSIVE,
            adapter="class1_srt",
            placement=(target_card.uuid,),
            launch={
                "model_path": str(specs["qwen36-27b"].launch["model_path"]),
                "port": 31_599,
                "rank_gpu_memory_mib": target_card.total_mib + 1024,
            },
        )
        try:
            registry.register(too_big)
        except RegistrationRejected as exc:
            print(f"  registration: {exc}", flush=True)
            rejections["registration"] = str(exc)

        # And one that is merely too big *right now*: registrable, but its
        # promotion is refused with the wait and the eviction.
        contender = EngineSpec(
            engine_id="contender",
            klass=EngineClass.AUTOREGRESSIVE,
            adapter="class1_srt",
            placement=(target_card.uuid,),
            launch={
                "model_path": str(specs["qwen36-27b"].launch["model_path"]),
                "port": 31_598,
                "rank_gpu_memory_mib": args.budget_mib,
            },
        )
        registry.register(contender)
        try:
            registry.ensure_state(
                "contender", ResidencyState.HOT, max_promotion_wait_ms=1.0
            )
        except PromotionRejected as exc:
            print(f"  promotion: {exc}", flush=True)
            rejections["promotion"] = exc.to_json()
        registry.deregister("contender")
        report["rejections"] = rejections

        print(
            "\n== 5. hot switch: engine B in, engine A out (hibernated) ==", flush=True
        )
        transition(registry, "qwen36-27b", ResidencyState.HOT, report["transitions"])
        assert registry.instance("qwen35-9b").state is ResidencyState.COLD
        text_b = registry.adapter("qwen36-27b").generate_probe(
            "The capital of France is"
        )
        print(f"  engine B says: {text_b!r}", flush=True)
        registry.refresh_measured()
        report["ledger_after_b"] = [c.to_json() for c in registry.cards()]
        report["hibernate_manifest_a"] = sorted(
            p.name for p in (out / "hibernate-qwen35-9b").glob("*")
        )

        print("\n== 6. switch back: engine A restored from its manifest ==", flush=True)
        resume_ms = transition(
            registry, "qwen35-9b", ResidencyState.HOT, report["transitions"]
        )
        text_a2 = registry.adapter("qwen35-9b").generate_probe(
            "The capital of France is"
        )
        print(f"  engine A says: {text_a2!r}", flush=True)
        report["probe_texts"] = {"a_cold": text_a, "b": text_b, "a_resumed": text_a2}

        cold_ms = report["transitions"][0]["wall_ms"]
        report["hot_switch"] = {
            "engine_a_cold_boot_ms": cold_ms,
            "engine_a_hibernate_restore_ms": round(resume_ms, 1),
            "speedup": round(cold_ms / resume_ms, 2) if resume_ms else None,
            "note": (
                "the restore also pays the eviction of engine B, so it is an upper "
                "bound on the resume itself"
            ),
        }
        print(json.dumps(report["hot_switch"], indent=2), flush=True)

        print("\n== 7. teardown ==", flush=True)
        registry.ensure_state("qwen35-9b", ResidencyState.COLD)
        report["ledger_final"] = [c.to_json() for c in registry.cards()]
    finally:
        registry.shutdown()
        sampler.stop()
        report["nvml_min_free"] = sampler.minimum_free()
        report["nvml_samples"] = len(sampler.samples)
        report["corridor_ok"] = all(
            v is None or v >= 400 * MIB for v in report["nvml_min_free"].values()
        )
        report["finished"] = time.time()
        report["duration_s"] = round(report["finished"] - report["started"], 1)
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport: {out / 'report.json'}", flush=True)
        print(f"duration: {report['duration_s']:.0f} s", flush=True)
        print(
            "min free per card (MiB): "
            + json.dumps(
                {
                    k: (v // MIB if v is not None else None)
                    for k, v in report["nvml_min_free"].items()
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
