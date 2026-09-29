# Copyright 2023-2026 SGLang Team
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
"""#622/#649: where each piece of forward work is placed, as data.

WHAT THIS MODULE IS FOR
-----------------------
The #622/#649 production family is not a transport fault. The transport is
healthy; five specimens on 2026-08-07 all show three ranks parked on the
IDENTICAL host line, never a divergence, with the GPUs at 100 % utilisation --
that is, spin kernels still resident on the stream. The host lines differ
across specimens (``dcp/owner.py:566`` twice, the draft
``kv_indptr[...].cpu()`` twice, a host-path abort once), so the defect is not
any one of those call sites. What every specimen shares is the SHAPE:

    a blocking host sync, issued in the out-of-graph metadata-prep phase,
    is ordered behind a barlink collective kernel from the previous step.

CUDA streams are in-order. ``cudaStreamSynchronize`` on a stream waits for
everything enqueued on it, and a ``cudaStreamWaitEvent`` node counts. So any
sync issued on the compute stream inherits the wait of every collective that
stream is ordered after. A collective that stalls for D therefore blocks the
host thread of every rank for D -- and a host thread that is blocked cannot
enqueue, cannot service the abort gate, and cannot be the rank that unwedges
its peers. The stall is amplified into a whole-cluster hang.

Enumerating and removing the individual syncs does not close this. It has
already been tried and measured: #623 removed the ``.item()`` at
``owner.py:548`` by threading ``total_tokens`` through every call site, and
the 15:55 specimen wedges at ``owner.py:566``, five lines later in the same
function, on the boolean-mask indexing that has no host-derivable form. The
wedge relocated rather than disappeared, exactly as NOTE_622 section 3
predicted it would. Every future ``.cpu()`` added anywhere on the forward path
re-opens the family.

So the property has to be enforced structurally, and to be enforced it first
has to be DECIDABLE. That is what this module exists for. It states the
placement of forward work -- which stream role, and which cross-stream
ordering edges that implies -- as inspectable data rather than as a scatter of
``torch.cuda.stream(...)`` context managers spread across the attention
backends, the graph runners and the transport. A test can then take a policy,
build the ordering graph a production step would produce under it, and decide
whether the property holds, without a GPU and without booting a model.

WHAT THIS MODULE IS NOT
-----------------------
It is not a scheduler and it allocates nothing. It answers one question --
"where does this op go, and what must be ordered around it" -- and the callers
that own real ``torch.cuda.Stream`` objects act on the answer. Keeping it free
of torch is deliberate: the decision is testable on a machine with no CUDA
device at all, which is where the #622 falsifier runs.

THE THREE POLICIES
------------------
``LEGACY`` is what the tree does today: everything on the compute stream. It is
kept, named, and tested precisely because the falsifier needs something that
demonstrably FAILS the property. A guard test that cannot fail proves nothing,
and this family has already produced one instrument that looked green for the
wrong reason.

``COLLECTIVE_STREAM`` is the fix as first proposed: give the collectives their
own stream. It is retained because it is a real and useful hardening -- it
keeps incidental device-wide syncs and allocator syncs off the collective path
-- but the falsifier shows it does NOT on its own satisfy the property, and
the reason is worth stating here rather than in a commit message. A collective
whose result is consumed by the model must be joined back onto the compute
stream, and that join is itself a node on the compute stream. A later sync on
the compute stream waits for the join, hence for the collective. Forking
without also isolating the sync side moves the kernel and keeps the hazard.

``ISOLATED_PREP`` is the placement that does satisfy it. The out-of-graph
metadata-prep phase -- the triton index-building kernels and the blocking
readback that follows them -- is placed on its own stream, which is ordered
after the host-driven input copies and after nothing else. The prep sync then
waits for the prep kernels and for no collective, on any policy, from any call
site, including call sites not written yet. That last clause is the whole
point: the property becomes a property of the seam rather than of an
enumeration.

THE OBLIGATION ``ISOLATED_PREP`` CARRIES
----------------------------------------
It is sound only if the prep kernels' inputs are host-written -- the H2D copies
of ``req_pool_indices``, ``seq_lens``, ``positions`` and the request-to-token
table -- and not produced by the previous step's device compute. If any prep
input is device-produced downstream of a collective, the prep stream has to be
ordered after that producer and the isolation is void for that input.

This obligation is encoded, not left in prose: ``HOST_INPUT`` is placed on the
prep stream under this policy, and a prep input that is genuinely
device-produced must be declared ``DEVICE_INPUT`` instead, which is placed on
the compute stream and joined into prep -- whereupon the falsifier's property
check fails and says so. Adding a device-produced prep input therefore breaks
a test rather than production.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping


class StreamRole(enum.Enum):
    """A stream identified by what it is for, not by a device handle.

    Roles, not indices: the point of the exercise is that two pieces of work
    are separated because they must not be ordered against each other, and a
    numeric stream index does not carry that reason.
    """

    COMPUTE = "compute"
    COLLECTIVE = "collective"
    PREP = "prep"


class OpKind(enum.Enum):
    """The kinds of forward work whose placement is in question.

    Deliberately coarse. A finer taxonomy would invite exactly the per-call-site
    reasoning this module exists to replace.
    """

    #: Host-driven H2D copies of batch metadata. Host-written by construction:
    #: the scheduler produces these values on the CPU.
    HOST_INPUT = "host_input"

    #: A prep input that IS produced by device compute in the previous step.
    #: Declaring one is what makes the prep stream's isolation obligation
    #: visible; see the module docstring.
    DEVICE_INPUT = "device_input"

    #: Index-building kernels in the out-of-graph metadata-prep phase, e.g.
    #: ``generate_draft_decode_kv_indices`` and the DCP owner-rule masking.
    PREP_KERNEL = "prep_kernel"

    #: The blocking host readback that ends the prep phase. Every #622/#649
    #: specimen is parked on one of these.
    PREP_SYNC = "prep_sync"

    #: Model compute inside a captured graph.
    GRAPH_COMPUTE = "graph_compute"

    #: A barlink collective. The op that can stall.
    COLLECTIVE = "collective"


@dataclass(frozen=True)
class Placement:
    """Where an op runs and what is ordered around it.

    ``waits_on`` are the roles whose current tail this op must be ordered
    after (a ``cudaStreamWaitEvent`` before the op). ``joined_by`` are the
    roles that must be ordered after this op (a ``cudaStreamWaitEvent`` on
    those streams after it). Both are recorded rather than implied, because
    the join is the edge that decides the property and an implied edge is one
    nobody checks.
    """

    role: StreamRole
    waits_on: tuple[StreamRole, ...] = ()
    joined_by: tuple[StreamRole, ...] = ()


@dataclass(frozen=True)
class StreamPolicy:
    """A complete placement decision for every op kind."""

    name: str
    placements: Mapping[OpKind, Placement] = field(default_factory=dict)

    def place(self, op: OpKind) -> Placement:
        try:
            return self.placements[op]
        except KeyError:
            raise KeyError(
                f"stream policy {self.name!r} has no placement for {op.value!r}. "
                f"A new op kind must be placed deliberately -- defaulting it to "
                f"the compute stream is how the #622 family was built."
            ) from None

    def roles_in_use(self) -> frozenset[StreamRole]:
        roles: set[StreamRole] = set()
        for placement in self.placements.values():
            roles.add(placement.role)
            roles.update(placement.waits_on)
            roles.update(placement.joined_by)
        return frozenset(roles)


_C = StreamRole.COMPUTE
_X = StreamRole.COLLECTIVE
_P = StreamRole.PREP


#: What the tree does today. Everything shares one in-order stream, so the
#: prep sync is ordered behind every collective that preceded it. Kept as a
#: named policy so the falsifier has a configuration that must fail.
LEGACY = StreamPolicy(
    name="legacy",
    placements={
        OpKind.HOST_INPUT: Placement(role=_C),
        OpKind.DEVICE_INPUT: Placement(role=_C),
        OpKind.PREP_KERNEL: Placement(role=_C),
        OpKind.PREP_SYNC: Placement(role=_C),
        OpKind.GRAPH_COMPUTE: Placement(role=_C),
        OpKind.COLLECTIVE: Placement(role=_C),
    },
)


#: Collectives forked onto their own stream and joined back. A real hardening,
#: and NOT sufficient on its own: the join is a compute-stream node, so a later
#: compute-stream sync still waits for the collective. The falsifier asserts
#: this failure explicitly so that the limitation cannot be forgotten and
#: re-proposed.
COLLECTIVE_STREAM = StreamPolicy(
    name="collective-stream",
    placements={
        OpKind.HOST_INPUT: Placement(role=_C),
        OpKind.DEVICE_INPUT: Placement(role=_C),
        OpKind.PREP_KERNEL: Placement(role=_C),
        OpKind.PREP_SYNC: Placement(role=_C),
        OpKind.GRAPH_COMPUTE: Placement(role=_C),
        OpKind.COLLECTIVE: Placement(
            role=_X, waits_on=(_C,), joined_by=(_C,)
        ),
    },
)


#: Collectives forked as above, AND the prep phase moved onto a stream ordered
#: after the host-driven input copies and nothing else. This is the placement
#: that satisfies the property. ``DEVICE_INPUT`` stays on the compute stream
#: and is joined into prep, so declaring one deliberately breaks the property
#: check rather than silently voiding the isolation.
ISOLATED_PREP = StreamPolicy(
    name="isolated-prep",
    placements={
        OpKind.HOST_INPUT: Placement(role=_P),
        OpKind.DEVICE_INPUT: Placement(role=_C, joined_by=(_P,)),
        OpKind.PREP_KERNEL: Placement(role=_P),
        OpKind.PREP_SYNC: Placement(role=_P, joined_by=(_C,)),
        OpKind.GRAPH_COMPUTE: Placement(role=_C),
        OpKind.COLLECTIVE: Placement(
            role=_X, waits_on=(_C,), joined_by=(_C,)
        ),
    },
)


POLICIES: Mapping[str, StreamPolicy] = {
    p.name: p for p in (LEGACY, COLLECTIVE_STREAM, ISOLATED_PREP)
}


#: The policy in force. LEGACY until a GPU window validates a replacement:
#: this module and its falsifier were produced on a host with no CUDA device,
#: so nothing here has been shown to preserve numerics or performance on the
#: rig. Changing this default is a GPU-window decision, not a desk one.
ACTIVE = LEGACY


def active_policy() -> StreamPolicy:
    return ACTIVE
