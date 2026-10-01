# SPDX-License-Identifier: Apache-2.0
"""Dual-model co-boot: the host-side namespace of two stacks (pure, stdlib only).

Design (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8.2): each model runs in its
own container, as every stack does today (``host_acceptance_v2.sh`` run_args
pass no ``--network/--ipc/--pid host``). That gives each stack its own netns,
pid namespace, ``/dev/shm`` and memory cgroup, so every in-stack name (ports
30030-30032, the shm families, the teardown pgrep, barlink fd sockets) is
already private. What the two containers still share lives on the HOST:

* the published port (``-p 127.0.0.1:<port>:30030``);
* the container name -- cleanup traps stop containers by name PREFIX, and a
  prefix that also matches the other stack's name stops a foreign boot;
* the bind-mounted directories (gpu-arb holder, state, evidence, store);
* the host RAM both memory caps are carved from. Two awake caps never fit
  (27B peak 89.6 GiB, NF 88.4 GiB, host 125.7 GiB), so the budget that has to
  fit is ONE awake cap plus every other stack's ASLEEP cap; the arbiter moves
  the caps at a model flip.

:func:`check_coboot` returns every such collision by name; an empty list is
the only green. :func:`require_coboot` raises with all of them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

GIB = 1 << 30

#: host ports other rig services own; a stack must never publish on them
#: (30099 lifeline router, 30097 cachy router, 8082 upstream, 8770 gpuq,
#: 8779 idle-clock daemon, 8890 rig dashboard, 6071/6075 devindex/logindex)
RESERVED_HOST_PORTS = frozenset({30099, 30097, 8082, 8770, 8779, 8890, 6071, 6075})

DIR_FIELDS = ("arb_dir", "state_dir", "evidence_dir", "store_dir")

#: published host port per model line; 27B keeps the router's 30030
DEFAULT_PORTS = {"27b": 30030, "nf": 30040}
#: default memory caps (bytes): awake covers the measured peak, asleep is the
#: tiering target of section 8.6
DEFAULT_MEM = {"27b": (92 * GIB, 24 * GIB), "nf": (92 * GIB, 24 * GIB)}


class CobootRefused(ValueError):
    """A co-boot spec that would collide on the host, every collision named."""


@dataclass(frozen=True)
class StackSpec:
    model: str
    container_name: str
    cleanup_prefix: str
    host_port: int
    arb_dir: str
    state_dir: str
    evidence_dir: str
    store_dir: str
    mem_awake_bytes: int
    mem_asleep_bytes: int

    def __post_init__(self) -> None:
        if int(self.host_port) in RESERVED_HOST_PORTS:
            raise CobootRefused(
                f"{self.model}: host_port {self.host_port} belongs to another rig service "
                f"({sorted(RESERVED_HOST_PORTS)})")
        if not self.container_name or not self.cleanup_prefix:
            raise CobootRefused(f"{self.model}: container_name and cleanup_prefix must be set")


def default_specs(*, acc_root: str, tag: str) -> Tuple[StackSpec, StackSpec]:
    """The two stacks with disjoint host names: 27B on 30030, NF on 30040."""
    out = []
    for model in ("27b", "nf"):
        root = os.path.join(acc_root, model)
        awake, asleep = DEFAULT_MEM[model]
        out.append(StackSpec(
            model=model,
            container_name=f"htsglang-dual-{model}-{tag}",
            cleanup_prefix=f"htsglang-dual-{model}-",
            host_port=DEFAULT_PORTS[model],
            arb_dir=os.path.join(root, "arb"),
            state_dir=os.path.join(root, "state"),
            evidence_dir=os.path.join(root, "evidence"),
            store_dir=os.path.join(root, "store"),
            mem_awake_bytes=awake,
            mem_asleep_bytes=asleep,
        ))
    return out[0], out[1]


def _nested(a: str, b: str) -> bool:
    a = os.path.normpath(a)
    b = os.path.normpath(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def check_coboot(specs: Sequence[StackSpec], *, host_mem_bytes: int,
                 host_reserve_bytes: int) -> List[str]:
    errs: List[str] = []
    for s in specs:
        if not s.container_name.startswith(s.cleanup_prefix):
            errs.append(f"{s.model}: cleanup_prefix {s.cleanup_prefix!r} does not match its own "
                        f"container_name {s.container_name!r} -- its cleanup would miss it")
        if s.mem_asleep_bytes > s.mem_awake_bytes:
            errs.append(f"{s.model}: mem_asleep {s.mem_asleep_bytes / GIB:.1f} GiB above "
                        f"mem_awake {s.mem_awake_bytes / GIB:.1f} GiB")
    for i, a in enumerate(specs):
        for b in specs[i + 1:]:
            pair = f"{a.model}/{b.model}"
            if a.model == b.model:
                errs.append(f"{pair}: two stacks of the same model -- one model, one stack")
            if a.host_port == b.host_port:
                errs.append(f"{pair}: same host_port {a.host_port}")
            if a.container_name == b.container_name:
                errs.append(f"{pair}: same container_name {a.container_name!r}")
            for x, y in ((a, b), (b, a)):
                if y.container_name.startswith(x.cleanup_prefix):
                    errs.append(f"{pair}: {x.model} cleanup_prefix {x.cleanup_prefix!r} matches "
                                f"{y.model} container {y.container_name!r} -- a cleanup would stop a foreign boot")
            for fa in DIR_FIELDS:
                for fb in DIR_FIELDS:
                    pa, pb = getattr(a, fa), getattr(b, fb)
                    if _nested(pa, pb):
                        errs.append(f"{pair}: {a.model} {fa} {pa!r} and {b.model} {fb} {pb!r} "
                                    f"are the same or nested")
    budget = int(host_mem_bytes) - int(host_reserve_bytes)
    for s in specs:
        need = s.mem_awake_bytes + sum(o.mem_asleep_bytes for o in specs if o is not s)
        if need > budget:
            others = " + ".join(f"{o.model} asleep {o.mem_asleep_bytes / GIB:.1f}"
                                for o in specs if o is not s)
            errs.append(f"mem: {s.model} awake {s.mem_awake_bytes / GIB:.1f} + {others} = "
                        f"{need / GIB:.1f} GiB > host {host_mem_bytes / GIB:.1f} - reserve "
                        f"{host_reserve_bytes / GIB:.1f} = {budget / GIB:.1f} GiB")
    return errs


def require_coboot(specs: Sequence[StackSpec], *, host_mem_bytes: int,
                   host_reserve_bytes: int) -> None:
    errs = check_coboot(specs, host_mem_bytes=host_mem_bytes,
                        host_reserve_bytes=host_reserve_bytes)
    if errs:
        raise CobootRefused("co-boot refused:\n  " + "\n  ".join(errs))


__all__ = [
    "CobootRefused", "StackSpec", "check_coboot", "default_specs", "require_coboot",
    "RESERVED_HOST_PORTS",
]
