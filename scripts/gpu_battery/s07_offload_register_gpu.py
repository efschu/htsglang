#!/usr/bin/env python3
"""Offload register on real silicon: CudaDeviceOps, real item sizes, retrieval
latency per class.

The #286 GPU restlist, items 1 and 4. Everything else about the register was
built and tested CPU-hermetically behind an injectable device layer; this is
the one thing a fake cannot answer -- whether the three real movement routes
move real bytes, and what getting them back costs.

THREE ROUTES, because the register has three and a validation of one is not a
validation of the register:

  tensor   pinned host pool + async H2D behind compute (the
           MoEExpertOffloadCache._fetch pattern, wait_stream in both
           directions). Carries lane_workspaces, kv_shadow, experts.
  tag      #93 tag pools / VMM through the memory saver. The route for
           va_stable items -- graph rungs and GDN state sets are addressed by
           kernels and graphs, so their virtual addresses have to survive the
           park.
  suspend  #89 suspend, for cold_lane.

WHY THE LATENCY NUMBERS MATTER MORE THAN THE PASS: every auto/ram default in
the register carries a measurement obligation ("Messpflicht vor jedem
auto/ram-Default"). Until a class has a measured retrieval latency, its
latency term for the #279 dispatcher is a guess, and a guessed latency term
silently decides placements. This step produces those numbers; it does not
decide the defaults.

The park is timed too, but the retrieval is the number that matters: a park
happens when there is slack, a wave-in happens when something is waiting.

WHY THE PROBE ASKS FOR THE PARK EXPLICITLY: the register parks on DEMAND. With
the ``auto`` policy every park is gated on the 13e/#279 saturation sensor, and
without pressure ``park()`` answers ``OffloadRefused`` -- correctly, that is
the design (the 2026-07-30 run refused six of six classes in 8 s and measured
nothing). The register's documented granular knob is the per-class policy
(``--lane-offload-class-policy``, syntax ``class=mode[:fraction]``), so this
probe sets every class it measures to ``ram`` through that same parser and
records the resolved policy map in the artifact. The gate is verified in the
same run by a NEGATIVE CONTROL: one item left at ``auto`` with no sensor
attached, whose park must still be refused.

The hysteresis windows are zeroed for the same reason: they are turn-boundary
policy (5 s by default) and a loop that measures a RATE deliberately runs
inside that window. Both are constructor arguments of the register -- this
configures the policy half, it does not reach around it.

Usage:
    python s07_offload_register_gpu.py --out <dir>/offload_register_gpu.json
    python s07_offload_register_gpu.py --dry-run [--out <path>]

``--dry-run`` runs the WHOLE probe -- policies, register, backend, all three
routes, the negative control -- against FakeDeviceOps without a card. It is
the plan phase of the step and doubles as the probe's own smoke test; the
artifact it writes names FakeDeviceOps, which the check refuses as a
validation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

MIB = 1024 * 1024

# One size for every route, chosen to be big enough that the copy dominates the
# call overhead and small enough that three of them fit anywhere on this rig.
ITEM_BYTES = 256 * MIB
CYCLES = 5

# (class, route, time-constant tier, va_stable) -- the register's classes as
# this step exercises them.
MEASURED = (
    ("lane_workspaces", "tensor", "phase", False),
    ("kv_shadow", "tensor", "turn", False),
    ("experts", "tensor", "wave", False),
    ("graph_rungs", "tag", "turn", True),
    ("gdn_state_sets", "tag", "turn", True),
    ("cold_lane", "suspend", "turn", False),
)

# The explicit park request, in the exact syntax of the register's own
# per-class knob (--lane-offload-class-policy). 'ram' = park at full depth,
# no sensor gate -- see the module docstring.
MEASUREMENT_CLASS_POLICY = ",".join(f"{cls}=ram" for cls, _, _, _ in MEASURED)

# Guard against an exec loop when the probe re-execs itself with the memory
# saver's preload hook in LD_PRELOAD.
REEXEC_GUARD_ENV = "SGLANG_S07_TMS_REEXEC"


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def repo_python_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )


# --------------------------------------------------------------- preload hook
def preload_binary_path() -> str:
    """The torch_memory_saver preload hook, asked for through the library's
    own public utility rather than by guessing a filename."""
    from torch_memory_saver import configure_subprocess

    with configure_subprocess():
        return os.environ["LD_PRELOAD"]


def cuda_runtime_lib_dir(maps_path: str = "/proc/self/maps"):
    """Directory of the libcudart the preload hook links against.

    The hook is built per CUDA major and NEEDs ``libcudart.so.<major>``, which
    torch ships inside site-packages -- somewhere the dynamic linker does not
    look on its own. Without that directory on LD_LIBRARY_PATH an LD_PRELOAD
    re-exec dies before Python starts ("libcudart.so.13: cannot open shared
    object file"). Read from the LIVE process map instead of guessed from a
    wheel layout: whatever torch actually loaded is what the hook needs, and
    this venv maps a cu12 and a cu13 runtime at once.
    """
    import torch

    major = str(getattr(torch.version, "cuda", "") or "").split(".")[0]
    if not major:
        return None
    needle = f"libcudart.so.{major}"
    with open(maps_path) as f:
        for line in f:
            path = line.rstrip("\n").split(" ", 5)[-1].strip()
            if path.endswith(needle):
                return os.path.dirname(path)
    return None


def maybe_reexec_for_memory_saver(
    environ,
    argv,
    execv,
    path_fn=preload_binary_path,
    libdir_fn=cuda_runtime_lib_dir,
):
    """Put the memory saver's preload hook into LD_PRELOAD and re-exec.

    The #93 tag and #89 suspend routes exist only when torch_memory_saver's
    hook is preloaded into the process -- and LD_PRELOAD is read by the
    dynamic linker at process start, so no in-process setting can repair it
    afterwards (the 2026-07-30 run: both routes died in the first region()
    with 'observes invalid LD_PRELOAD'). Re-exec is the one thing that can,
    and it keeps the pid, so the step's py-spy/pid bookkeeping still points at
    this process.

    Returns a provenance string for the artifact; the injectable ``execv`` /
    ``path_fn`` / ``libdir_fn`` are what makes this testable without a card.
    """
    if environ.get(REEXEC_GUARD_ENV) == "1":
        return "reexec"
    preloaded = environ.get("LD_PRELOAD", "")
    if "torch_memory_saver" in preloaded:
        return "inherited"
    try:
        path = str(path_fn())
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    try:
        libdir = libdir_fn()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    if not libdir:
        # Preloading a hook whose libcudart cannot be resolved does not fail
        # the probe, it fails the INTERPRETER (rc 127, no artifact at all).
        # Reporting the two routes as untested is the honest answer.
        return "unavailable: libcudart des Hooks nicht auffindbar"
    environ["LD_PRELOAD"] = f"{path}:{preloaded}" if preloaded else path
    lib_path = environ.get("LD_LIBRARY_PATH", "")
    if libdir not in lib_path.split(":"):
        environ["LD_LIBRARY_PATH"] = f"{libdir}:{lib_path}" if lib_path else libdir
    environ[REEXEC_GUARD_ENV] = "1"
    execv(sys.executable, [sys.executable, *argv])
    return "reexec"  # not reached under a real execv


# --------------------------------------------------------------- environments
class _FakeTensor:
    """Byte-count stand-in for --dry-run. resolve_size_bytes reads
    numel()/element_size() and FakeDeviceOps never dereferences the payload,
    so the dry run needs neither torch nor a card."""

    def __init__(self, nbytes: int, element_size: int = 2):
        self._element_size = element_size
        self._numel = nbytes // element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _DryRunEnv:
    """FakeDeviceOps + fake tensors: the whole probe, no card."""

    memory_saver = "fake"
    preload = "not needed (dry-run)"

    def __init__(self):
        from sglang.srt.model_executor.offload_movement import FakeDeviceOps

        self.device_ops = FakeDeviceOps()
        self.device_index = 0
        self.device_info = {
            "cuda_index": None,
            "name": "dry-run (keine Karte)",
            "pci_bus_id": None,
            "total_mib": None,
        }
        self.has_saver = True

    def alloc(self, nbytes: int, tag=None):
        return _FakeTensor(nbytes)

    def release(self, obj) -> None:
        pass


class _CudaEnv:
    """The real device layer on the biggest card of the rig."""

    preload = "unknown"

    def __init__(self):
        import torch

        from sglang.srt.model_executor.offload_movement import CudaDeviceOps

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda nicht verfuegbar")

        # The card is resolved, never assumed. The biggest one carries the
        # test because it is the only one with room for three parked items at
        # once.
        big = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.get_device_properties(i).total_memory,
        )
        torch.cuda.set_device(big)
        props = torch.cuda.get_device_properties(big)
        self.device_index = big
        self._device = torch.device(f"cuda:{big}")
        self.device_info = {
            "cuda_index": big,
            "name": props.name,
            "pci_bus_id": "%08x:%02x:%02x.0"
            % (
                getattr(props, "pci_domain_id", 0),
                getattr(props, "pci_bus_id", 0),
                getattr(props, "pci_device_id", 0),
            ),
            "total_mib": props.total_memory // MIB,
        }

        # The memory saver is what makes the tag and suspend routes real.
        # Without it those routes are not "fine", they are untested -- the
        # check turns that into a STOP rather than a green run with two thirds
        # of the register unexercised. Proven here with one tiny region rather
        # than assumed: an adapter constructs fine without the preload hook and
        # only fails at the first region().
        self._saver = None
        try:
            from sglang.srt.utils.torch_memory_saver_adapter import (
                TorchMemorySaverAdapter,
            )

            saver = TorchMemorySaverAdapter.create(enable=True)
            with saver.region(tag="battery_saver_smoke", enable_cpu_backup=True):
                smoke = torch.empty(1024, dtype=torch.float16, device=self._device)
            del smoke
            torch.cuda.empty_cache()
            self._saver = saver
            self.memory_saver = "real"
        except Exception as exc:
            self.memory_saver = f"unavailable: {type(exc).__name__}: {exc}"

        self.device_ops = CudaDeviceOps(memory_saver_adapter=self._saver)
        self.has_saver = self._saver is not None

    def alloc(self, nbytes: int, tag=None):
        torch = self._torch
        nelem = nbytes // 2  # float16
        if tag is None:
            return torch.empty(nelem, dtype=torch.float16, device=self._device)
        with self._saver.region(tag=tag, enable_cpu_backup=True):
            tensor = torch.empty(nelem, dtype=torch.float16, device=self._device)
        return tensor

    def release(self, obj) -> None:
        # The caller drops its own reference; this only returns the freed
        # blocks to the driver so the next 256-MiB item does not stack on top
        # of the last one (VRAM corridor).
        del obj
        self._torch.cuda.empty_cache()


# ---------------------------------------------------------------- the measure
def negative_control(device_ops) -> dict:
    """The falsifier for the measurement's explicit policy: one item left at
    ``auto`` with NO saturation sensor attached must still be refused.

    Without this, a register that silently started parking everything -- the
    exact regression the auto gate exists to prevent -- would produce the same
    green rows as a correct one. No payload is bound on purpose: the policy
    gate has to refuse BEFORE the movement backend is ever consulted.
    """
    from sglang.srt.model_executor.offload_movement import RealMovementBackend
    from sglang.srt.model_executor.offload_register import (
        OffloadRefused,
        OffloadRegister,
        resolve_class_policies,
    )

    policies = resolve_class_policies("auto")
    register = OffloadRegister(
        policies=policies,
        backend=RealMovementBackend(
            device_ops=device_ops,
            target_order=("host_ram",),
            class_policies=policies,
        ),
        hysteresis_window_s=0.0,
        phase_hysteresis_window_s=0.0,
    )
    item_id = "battery:negative_control"
    register.register(
        item_id=item_id,
        offload_class="lane_workspaces",
        size_bytes=1,
        restore_cost_ms=0.0,
        hot=lambda: False,
        time_constant_tier="phase",
    )
    result = {"policy": "auto", "sensor": "none", "refused": False, "error": None}
    try:
        register.park(item_id)
    except OffloadRefused as exc:
        result["refused"] = True
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # a park that failed for any OTHER reason
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_probe(env, item_bytes: int, cycles: int) -> dict:
    """Register, bind and cycle one item per class, per route."""
    from sglang.srt.model_executor.offload_movement import (
        RealMovementBackend,
        SuspendPayload,
        TagPayload,
        TensorPayload,
    )
    from sglang.srt.model_executor.offload_register import (
        OffloadRegister,
        resolve_class_policies,
    )
    from sglang.srt.model_executor.offload_sizes import resolve_size_bytes

    payload = {
        "kind": "offload_register_gpu",
        "schema_version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        # Filled from the object that is actually built further down. A literal
        # here would make the check that guards against a FakeDeviceOps fallback
        # assert its own constant.
        "device_ops": type(env.device_ops).__name__,
        "device": env.device_info,
        "item_bytes": item_bytes,
        "cycles": cycles,
        "memory_saver": env.memory_saver,
        "memory_saver_preload": env.preload,
        "routes": {},
        "rows": [],
        "stats": {},
    }

    policies = resolve_class_policies("auto", MEASUREMENT_CLASS_POLICY)
    # Provenance: the artifact says which policy admitted these parks, so a
    # reader never has to guess whether the run measured the movement or the
    # gate.
    payload["class_policies"] = {
        klass: {"mode": policy.mode, "fraction": policy.fraction}
        for klass, policy in sorted(policies.items())
    }
    backend = RealMovementBackend(
        device_ops=env.device_ops,
        target_order=("host_ram",),
        class_policies=policies,
    )
    register = OffloadRegister(
        policies=policies,
        backend=backend,
        hysteresis_window_s=0.0,
        phase_hysteresis_window_s=0.0,
    )

    # The item's restore cost is the MEASURED wave-in, not a constant: this is
    # the number retrieval_latency_ms / latency_term_ms hand to the #279
    # dispatcher, and a registered 0.0 would hand it a guess dressed as a
    # measurement.
    measured_restore_ms: dict = {}

    def cycle(item_id, row):
        """park -> settle -> wave_in, timed, with the state sequence recorded.

        The states are recorded rather than assumed: a park that silently
        no-ops and a park that moves 256 MiB both return None, and only the
        state machine tells them apart.
        """
        park_ms, wave_ms, states = [], [], []
        for i in range(cycles):
            states.append(backend.state_of(item_id))
            t0 = time.perf_counter()
            register.park(item_id)
            backend.settle(item_id)
            park_ms.append((time.perf_counter() - t0) * 1e3)
            states.append(backend.state_of(item_id))

            if i == cycles - 1:
                # Read while the item IS parked -- a resident item contributes
                # 0.0 by definition, so this is the only honest moment for it.
                row["latency_term_ms"] = register.latency_term_ms(row["offload_class"])

            t0 = time.perf_counter()
            register.wave_in(item_id)
            backend.settle(item_id)
            wave_ms.append((time.perf_counter() - t0) * 1e3)
            states.append(backend.state_of(item_id))
            measured_restore_ms[item_id] = percentile(wave_ms, 0.50)

        wave_p50 = percentile(wave_ms, 0.50)
        row["park_ms_p50"] = round(percentile(park_ms, 0.50), 3)
        row["park_ms_p99"] = round(percentile(park_ms, 0.99), 3)
        row["wave_in_ms_p50"] = round(wave_p50, 3)
        row["wave_in_ms_p99"] = round(percentile(wave_ms, 0.99), 3)
        row["iters"] = len(wave_ms)
        # From the UNROUNDED p50: a fast route must not report "no rate"
        # because its p50 rounded to zero.
        row["wave_in_gb_per_s"] = (
            round(row["size_bytes"] / (wave_p50 / 1e3) / 1e9, 3)
            if wave_p50 > 0
            else None
        )
        row["state_sequence"] = states[:9]
        row["status"] = "ok"

    route_status = {}
    for offload_class, route, tier, va_stable in MEASURED:
        if route != "tensor" and not env.has_saver:
            # Two of three routes untested is a STOP for the check, not a
            # green run on one third of the register.
            route_status[route] = "unavailable"
            continue
        item_id = f"battery:{offload_class}"
        tag = f"battery_{offload_class}" if route != "tensor" else None
        row = {"offload_class": offload_class, "route": route, "item_id": item_id}
        if tag is not None:
            row["tag"] = tag
        obj = None
        try:
            obj = env.alloc(item_bytes, tag=tag)
            resolved = resolve_size_bytes(obj)
            row["size_bytes"] = resolved
            row["size_source_matches"] = resolved == obj.numel() * obj.element_size()
            register.register(
                item_id=item_id,
                offload_class=offload_class,
                size_bytes=resolved,
                restore_cost_ms=lambda iid=item_id: measured_restore_ms.get(iid, 0.0),
                hot=lambda: False,
                va_stable_required=va_stable,
                time_constant_tier=tier,
            )
            if route == "tensor":
                movement = TensorPayload(tensors=(obj,))
            elif route == "tag":
                movement = TagPayload(tag=tag, cpu_backup=True)
            else:
                movement = SuspendPayload(tags=(tag,))
            backend.bind(item_id, movement, source_device=env.device_index)
            cycle(item_id, row)
            route_status.setdefault(route, "ok")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-2000:]
            route_status[route] = "error"
        finally:
            if obj is not None:
                # The tag/suspend payloads carry only a tag, so their item is
                # really freed here; a TensorPayload keeps its tensor alive
                # through the binding, by design -- that is the item.
                env.release(obj)
                obj = None
        payload["rows"].append(row)

    payload["routes"] = route_status

    stats = getattr(backend, "stats", None)
    if stats is not None:
        payload["stats"] = {
            "parks": getattr(stats, "parks", None),
            "wave_ins": getattr(stats, "wave_ins", None),
            "park_failures": getattr(stats, "park_failures", None),
            "wave_in_failures": getattr(stats, "wave_in_failures", None),
            "peer_degradations": getattr(stats, "peer_degradations", None),
            "chunked_transfers": getattr(stats, "chunked_transfers", None),
            "bytes_by_target": dict(getattr(stats, "bytes_by_target", {}) or {}),
        }

    # The latency term is what the #279 dispatcher will actually read.
    # Recording it here closes the loop: the number the cost model sees comes
    # from this measurement and not from a constant.
    payload["latency_term_ms"] = {
        row["offload_class"]: row["latency_term_ms"]
        for row in payload["rows"]
        if row.get("status") == "ok" and "latency_term_ms" in row
    }
    payload["negative_control"] = negative_control(env.device_ops)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        help="Artefaktpfad; ausserhalb von --dry-run erforderlich",
    )
    ap.add_argument("--cycles", type=int, default=CYCLES)
    ap.add_argument("--item-mib", type=int, default=ITEM_BYTES // MIB)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="ganzer Ablauf gegen FakeDeviceOps, ohne Karte (Planphase)",
    )
    args = ap.parse_args(argv)
    if not args.dry_run and not args.out:
        ap.error("--out wird ausserhalb von --dry-run benoetigt")

    sys.path.insert(0, repo_python_path())

    if args.dry_run:
        print(
            "Plan: eine Karte (die groesste, zur Laufzeit per PCI aufgeloest), "
            "CudaDeviceOps + RealMovementBackend; je Route (tensor|tag|suspend) "
            f"ein Posten von {args.item_mib} MiB, {args.cycles} park/wave_in-Zyklen, "
            "p50/p99 je Klasse; echte Groessen ueber resolve_size_bytes; "
            f"expliziter Park ueber die Klassen-Policy '{MEASUREMENT_CLASS_POLICY}' "
            "plus Negativkontrolle (auto ohne Druck muss verweigern). "
            "Dieser Lauf fuehrt genau das gegen FakeDeviceOps aus -- ohne Karte."
        )
        env = _DryRunEnv()
    else:
        preload = maybe_reexec_for_memory_saver(os.environ, sys.argv, os.execv)
        try:
            env = _CudaEnv()
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        env.preload = preload

    print(f"memory saver: {env.memory_saver} (LD_PRELOAD: {env.preload})")
    payload = run_probe(env, item_bytes=args.item_mib * MIB, cycles=args.cycles)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    print(
        f"{'Klasse':<18} {'Route':<8} {'MiB':>6} {'park p50':>9} {'wave p50':>9} {'GB/s':>7}"
    )
    for row in payload["rows"]:
        print(
            f"{row['offload_class']:<18} {row['route']:<8} "
            f"{(row.get('size_bytes') or 0) // MIB:>6} "
            f"{str(row.get('park_ms_p50')):>9} {str(row.get('wave_in_ms_p50')):>9} "
            f"{str(row.get('wave_in_gb_per_s')):>7}  {row.get('status')}"
        )
    control = payload["negative_control"]
    print(
        f"Negativkontrolle (auto ohne Druck): "
        f"{'verweigert' if control['refused'] else 'NICHT verweigert'} "
        f"-- {control['error']}"
    )
    if args.out:
        print(f"geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
