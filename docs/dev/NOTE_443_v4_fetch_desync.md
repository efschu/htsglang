# #443 — the V4 GGUF expert fetch, desynchronised

BENCH_394 §3 records the constraint plainly: `--disable-cuda-graph` is required
on DeepSeek-V4-Flash because "the MoE expert offload's default fetch path takes
a device->host sync per forward, which a captured graph cannot contain". That
sync is the ranked-#2 cause of the 2.6x decode gap against the club-3090
reference.

This is a PORTING item, and the catalog said so before the work started (§3:
"CUDA-graph-compatible path EXISTS ... The DeepSeek-V4 GGUF path still uses a
`tolist()`-syncing variant: that is a PORTING item, not a wall"). The work was
therefore to find what actually kept the V4 path off the existing #122
mechanism, not to build a second one.

## Sync inventory — the V4 GGUF expert path

Swept with the #440 interception technique (probe `torch.Tensor.item`,
`tolist`, `cpu`, `numpy`, `nonzero`, `__bool__`, `__int__`, `__float__`,
`__index__` and assert the call list) plus a read of every call site between
`FusedMoE.run_moe_core` and the H2D copy.

| site | what it is | verdict |
|---|---|---|
| `expert_offload.py:2988` `ids_list = topk_ids.tolist()` in `run_waves` | the sync. Feeds `plan_token_waves` / `plan_expert_waves`, the #390 router stats and the hot-set calibration | REPLACED under capture by `prepare_capturable_remap` (device cumsum-rank). The eager path keeps it — prefill and over-bucket decode still run `run_waves`, and neither is captured |
| `expert_offload.py:2798` `torch.unique(...).tolist()` in `prepare` | the same read in the single-wave entry point | not on the captured path; `prepare` is unused by the graph mode, which calls `prepare_capturable` |
| `expert_offload.py:2717` `for expert_id, slot in fetch_plan` in `_fetch` | the H2D copy loop, driven by a HOST list | REPLACED under capture by `_issue_fetch_capturable`: one `index_select` per expert-tensor attr, source = UVA view of the pinned pool, index = the device-resident `src_row` |
| `expert_offload.py:2718` `expert_id in self._remote_ids` | #394 delegation branch, a host read of a Python set | cannot be ported as-is. The capturable path refuses a cold tier by default; behind the development switch the condition is COUNTED on device and raised at the replay boundary (below) |
| `layer.py:2116` `topk_ids.detach().to("cpu").tolist()` | #390 routing trace | already structurally gated: `if self._moe_offload_trace_path and not get_is_capture_mode()`. Pinned by test |
| `layer.py:2164` capture-admission check | compared `topk_ids.numel()` against the scratch size | reads SHAPES, never contents — no sync, but wrong. Tightened, below |
| `layer.py:1240` `param.weight_type = loaded_weight.item()` | GGUF load-time | not a forward-path call |
| `cold_tier_fetch.py` | resolver + `peer_row_tensor` | no `.item()`/`.tolist()`/`.cpu()` anywhere; host-side layout arithmetic only |
| `prepare_capturable_remap` internals | `int(resident_slot_lut.shape[0])`, `int(resident_count)` | `torch.Size` entries and Python ints — free. Confirmed by the interception test finding zero calls |

So the mechanism was already sync-free. Two things kept it from serving V4.

## What actually blocked the port

**1. The capture-admission bound was loose, and loose in exactly the direction
the #82 GGUF shard produces.** `layer.py` demanded `topk_ids.numel()` scratch
slots. But `prepare_capturable_remap` assigns one slot per DISTINCT routed
spill expert, and a step cannot route more distinct SPILL experts than this
rank's cold set holds. Under the #82 expert-dim shard `forward_impl` has
already collapsed every FOREIGN expert id onto the resident zero-pad expert
before the offload sees it, so the surviving ids are drawn from this rank's
local table alone and `E - R` is the real ceiling. The loose bound refused
captures that are provably safe. `worst_case_unique_spill(routed_slots,
num_local_experts, resident_count) = min(routed_slots, E - R)` is the tight
one; it reads shapes, never contents, so it is legal on the capture path.

For decode buckets the routed-slots half is what binds in practice — see the
sizing note in `scripts/dev/443_graph_proof/README.md`. The cold-set half
matters for the admission decision's correctness, not usually for its value.

**2. The #394 seam was a switch into undefined behaviour.** Under a shared cold
tier a routed expert can be delegated to a peer's segment; this rank's pool has
no row, so the frozen `spill_pool_row_lut` holds `-1`. `-1` is not a diagnosis,
it is an out-of-bounds index — and the eager path's cover for it
(`expert_id in self._remote_ids`) is precisely the host read a capture cannot
contain. The refusal in `refuse_capturable_cold_tier` named only the UVA
pointer, which sent the reader looking for a hardware problem when the nearer
gap is that there is no peer source in the captured gather at all.

Fixed the #431 way, not by adding a branch: the remap CLAMPS the index and
increments a device-resident int32 counter, and `moe/offload_capture_gate.py`
reads that counter at the CUDA-graph replay boundary — host code, no capture in
progress — and turns it into `OffloadCaptureBreach(layer_id, breaches, where)`.
The module is separate for the same reason `barlink_abort_gate` is: the
boundary lives in `model_executor` and must not grow an import of
`expert_offload`. Verified: importing `full_cuda_graph_backend` leaves
`layers.moe.expert_offload` out of `sys.modules` (pinned by test).

The default path is unchanged statement for statement — `breach_counter` is
`None` on every launch without `SGLANG_MOE_COLD_TIER_SHM`, and the registry the
boundary consults is empty, so a replay costs one truth test on a list.

## Composition

**#394 s2/s3.** The seam stays a seam. Capture over peer-owned cold rows is
still refused by default, now naming both gaps; `SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`
opens a development window in which a routed delegated expert is a named
exception rather than a silent read of local row 0. The remaining hardware
question is unchanged and honestly unverified: whether `is_pinned()` /
`torch.as_tensor` reproduce the UVA aliasing for a `cudaHostRegister`'d `mmap`
the way they do for a torch `pin_memory()` allocation. `device_view_of_pinned`
already refuses to build a view that copied, so the failure mode is a refusal,
not a wrong read — but only for the local pool, which is why the peer path is
not wired.

**#439 link-mode ranges.** Nothing here touches the plan. The capturable LUTs
are built from `self.planner.resident_slot` and `self._spill_pool_index`, i.e.
from whatever residency the launcher-resolved, rank-uniform plan installed;
`build_capturable_luts` snapshots it after any `freeze_from_source()`
rearrange. A `--rank-moe-ratio link` boot changes which experts a rank owns,
not how the captured gather finds them.

## Honest state

Desk-provable and proven (`tests/moe_offload/test_capture_desync_port.py`,
63 tests, ~16 s, `CUDA_VISIBLE_DEVICES=99`):

- no host read survives the ported step, armed seam included; the can-fail runs
  the eager path under the same probes and catches its `tolist`;
- the captured gather lands byte-identical scratch rows vs the eager `_fetch`,
  and every remapped id addresses the row of the expert it routed to;
- the tightened bound is reachable and not exceedable over 200 adversarial
  draws;
- the layer's two host-only branches are decided by capture state;
- the delegated-expert breach is counted, clamped, raised by name, and reset.

Executed can-fail arms, each reverted: loosen the bound (5 fail), drop the
clamp (4 fail, with a real `IndexError`), test the breach on the host inside
the step (1 fail, catching a `__bool__`), off-by-one the slot rank (48 fail
across this file and `test_capturable_planner.py`).

BOOT-PENDING, and not claimed: that a real V4 decode graph captures, that
replay is correct, that the UVA view survives capture, and that any of it is
faster. Arm spec in `scripts/dev/443_graph_proof/`; the A-vs-A floor is two
eager boots on the same day, not BENCH_394's figure from another one.
