# Task #111 — NCCL cross-instance KV transport for PD (L0 tier)

Status: **seam + contract built and hermetically tested; the wire path is the
GPU slice and has not been written blind.** Nothing here is registered in
`TransferBackend` yet — adding the enum member would advertise a working
backend, and the two-instance ticket below is what earns that.

## What exists already (and is therefore not rebuilt)

The PD transport seam is not new. `disaggregation/base/conn.py` already
defines `BaseKVManager` / `BaseKVSender` / `BaseKVReceiver` /
`BaseKVBootstrapServer`, `KVArgs` (including `state_types` with
`StateType.MAMBA`), and `KVPoll`; `disaggregation/common/conn.py` implements
the bootstrap, rank mapping, heartbeat and registration scaffolding that
mooncake/mori/nixl/ascend all subclass; `get_kv_class` in
`disaggregation/utils.py` is the backend switch.

So #111 is **not** "invent a transport interface". It is: add an NCCL member,
and put the one operation a future fastpath would replace behind a narrower
seam of its own.

## The transport contract

### The link seam (`nccl/link.py`) — #110

A `KvLink` does exactly four things:

| method | contract |
|---|---|
| `setup(session_id, is_sender, peer)` | establish the peer connection; bounded, raises rather than blocking |
| `register(regions)` | make every region addressable, **all or nothing** |
| `transfer(blocks, message_class, timeout_s)` | move a block list and complete under a bound; returns bytes moved |
| `close()` | release; idempotent |

Everything else — who the peer is, whether the payload is legal, which route
carries it, what the pages mean — is decided **above** the link so a new member
cannot redecide policy. A P2P/NVLink/BAR1 fastpath is then a new `KvLink` plus
one registry entry, not a second copy of the sender/receiver/handshake stack.

Members today: `NcclLink` (wire path pending, see below) and `LoopbackLink` (a
real member whose peer is local memory — this is what makes the byte gate a
test rather than a mock assertion).

### The compatibility handshake (`nccl/contract.py`) — #241

`TransportIdentity` carries **model identity + KV dtype + geometry** and is
built on both sides from local facts only; the bootstrap exchange carries the
remote copy and `assert_compatible` does a field-by-field diff with a named
mismatch. The model hash reuses
`mem_cache.hicache_storage.compute_model_identity_hash` — the same function the
storage key uses — because two answers to "is this the same model and byte
format" is how they drift.

Compared: `model_identity_hash`, `kv_dtype`, `page_size`, `total_kv_head_num`,
`head_dim`, `state_types`.
Deliberately **not** compared: `tp_size` / `pp_size` / `dcp_size`. PD already
supports differing prefill/decode TP — `KVArgs` carries `state_dim_per_tensor`
and `state_dim_offsets` precisely so the sender can re-slice — so demanding
equality would refuse working configurations. They ride along for diagnostics.

### Route policy — #212

`resolve_route` refuses `store` for a hybrid GDN model, by name, with the
reason. The paid-for finding: a store round trip carries KV pages only, the
recurrent state lives in a separate pool, and
`MambaRadixCache._match_post_processor` truncates any prefix match to the
deepest node owning a mamba checkpoint — so a KV-only import matches **zero
tokens** and the decode side recomputes the whole prompt while appearing to
work. Direct is what PD already does correctly: `setup_state_kv_args` appends
a `StateType.MAMBA` component and the state moves with the KV.

`store` also refuses when any state component is present, hybrid or not.
Dense models keep the store route unchanged.

### Message classes — #240 / #244 / #263

`MessageClass` is `KV_BULK` / `STATE` / `AUX_SMALL`, and `net_for_class` maps
them onto the **existing** `--collective-net-small` / `--collective-net-bulk`
vocabulary rather than inventing a second one for the same physical decision.
`STATE` is bulk on purpose: the mamba slot moves in the same plan as the KV, so
pinning it to the management net would split one payload across two wires.
Unpinned returns `None` — "transport chooses", the unchanged default.

### Registration — #221

`_validate_regions` is shared by every link member and raises on the first bad
region, so a caller can never observe a link that accepted some regions and
rejected others. This mirrors `CommonKVManager._batch_register_checked`, which
is where the mooncake fix for the ignored `batch_register` status lives; the
NCCL member routes through the same discipline even though NCCL has no MR to
pin, so the two links cannot be told apart by which errors they raise.

### Bounded waits and fixed membership — #259 / #312 / #94

Every link wait is bounded (`DEFAULT_LINK_TIMEOUT_S`, 120 s). The NCCL member
is documented to form its group from the **bootstrap-exchanged fixed peer
universe** and to wait through the #312 helpers
(`bounded_collective` / `bounded_barrier` in
`distributed.device_communicators.barlink_liveness`) rather than re-deriving
them — a dead peer must end the wait with a named error, never a spin. No rank
asks the group who is present: that would be a membership decision taken by a
collective, which is the #94 family and the reason #259 pinned a fixed pool
universe.

### Block planner

`plan_blocks` turns a row mapping into a coalesced block list, splitting on a
gap in **either** the source or the destination (coalescing on the source alone
would write the wrong rows for an owner-rule scatter). A src/dst length
mismatch is refused, not truncated — the mooncake path already recorded that
silent truncation misaligns rows and corrupts KV.

## Tests

`test/registered/unit/disagg_nccl/`, 53 hermetic CPU tests:

* `test_kv_link.py` — region/block validation, all-or-nothing registration on
  every member, the **byte-identity gate** (whole-buffer, scattered rows,
  untouched-rows-not-disturbed, bounds refusals), multi-component KV+mamba
  transfer, the `NcclLink` contract, and the registry.
* `test_transport_contract.py` — handshake compatibility and every refusal,
  the differing-TP allowance, route policy, message classes, the block planner,
  and a planner-feeds-link end-to-end byte check.

Falsifier-checked: a 1-byte destination misalignment in the loopback copy turns
4 byte-gate tests red; removing the hybrid-GDN store refusal turns the route
test red.

## What the two-instance GPU ticket must validate

One prefill instance + one decode instance, hybrid GDN model (Qwen3.6-27B class
— the model that makes the #212 finding bite), plus one dense control.

1. **The wire path exists and moves bytes.** `NcclLink.transfer` is currently
   `NotImplementedError` by design. The ticket lands it and proves a KV payload
   arrives. Until this passes, nothing below is meaningful.
2. **Byte identity against the mooncake path on the same payload.** Run the
   same request through mooncake and through NCCL and compare the decode-side
   KV bytes, not the output text. Text identity is not the instrument (the
   house standard, #360); bytes are, and the hermetic gate already fixes the
   contract this must reproduce end to end.
3. **The GDN slot actually arrives.** The direct-route claim is that
   `StateType.MAMBA` moves with the KV. Prove it the way #212 did: a decode
   instance that receives state should NOT recompute the prompt. Watch the
   prefix-match length on the decode side — a KV-only transfer matches zero
   tokens and looks fine.
4. **Partial registration fails loudly.** Inject a failing region and confirm
   the boot dies naming the region, rather than serving with one component
   unregistered (#221's failure mode).
5. **Peer death ends the wait.** Kill the prefill instance mid-transfer; the
   decode side must surface a named error within the bound, never spin (#312).
   Run it in both directions.
6. **Handshake refusal on a real mismatch.** Boot the two sides with different
   `--kv-cache-dtype` and confirm the refusal fires before the first transfer.
7. **Per-class net pinning takes.** With `--collective-net-small` and
   `--collective-net-bulk` on different interfaces, confirm the bulk KV rides
   the fast wire — the #212 measurement (105 MB/s LAN vs ~1930 MB/s RoCE) is
   the reason this is worth checking rather than assuming.

Not in scope for the first window: a throughput claim against mooncake. Correct
and loud first; the comparison is its own ticket once the path is proven.

## Honest remainder

1. `NcclLink.transfer` — the wire path. Everything else in this task feeds it.
2. Not wired into `TransferBackend` / `get_kv_class`; the enum member lands
   with the proof.
3. The group factory is an injected callable, so the actual TCPStore rendezvous
   between two instances is unwritten — it belongs with (1).
4. No `--disaggregation-kv-link` server flag yet; adding one before the backend
   works would be a knob that selects a `NotImplementedError`.
