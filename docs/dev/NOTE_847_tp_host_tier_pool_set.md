# POSTEN #847-b — the TP phase host tier is built with ONE pool, and the model has two

Status: **FILED, NOT BUILT.** The guard that makes its absence safe is in
(`check_pool_coverage`, `hicache_phase_binding.py`); this note is the enabler
that would let the device tier stay ARMED across a cutover instead of disarmed.

## What is there now

`phase_flip_boot.py:2019-2032` builds the TP phase's host view as

```python
tp_host = HostPoolGroup([
    build_pool_entry(name=PoolName.KV, host_pool=kv_host, device_pool=inner_pool, ...)
])
```

— a single `PoolName.KV` entry. On this rig the boot-time tier is
`pools=KV + MAMBA` (`hybrid_pool_assembler.build_hybrid_mamba_stack`), so the
cutover installs a tier that cannot describe half the model's state.

The commit that wrote those lines states the rule it was following
(`phase_flip_boot.py:1969-1977`): *"BUILT WITH THE ASSEMBLER'S OWN NAMED
PRIMITIVES ... the assembler has five call sites and this must not become a
sixth hand-rolled one."* It then reuses `build_kv_host_pool` and
`build_pool_entry` but hand-rolls the GROUP — and the group is exactly where the
pool set lives. The one-mover rule was applied one level too shallow.

## Consequence, measured

W38-B (`boot_w38b_0825_1722.log`): rebind to 'tp' at 17:25:53, first
host-backed mamba resume at 17:25:59, `AttributeError: 'NoneType' object has no
attribute 'cpu'` on all three PP ranks in the same second. The silent-skip twin
(`HostPoolGroup.load_to_device_per_layer`) would have moved the KV and left the
recurrent state behind while the tree reported the prefix resident.

## Why it is not built here

The MAMBA pools are phase-invariant — they live in `req_to_token_pool`, which
the flip does not rebind — so carrying the entry across looks like a two-line
change. It is not:

* `build_pool_entry` wraps the layer mapping in `_make_layer_mapper`, which
  returns `None` for any `layer_id` outside `[0, transfer_layer_num)`.
* The TP view reports **16** layers; the boot-time groups report **32 / 18 / 14**
  (per-PP-stage). The boot-time `mamba_layer_mapping` keys are PP-stage-local
  layer ids.

Copying the boot-time MAMBA entry into the TP group therefore yields a mapper
that silently returns `None` for every mamba layer whose stage-local id is >= 16
— which is the *same silent-skip failure*, moved one layer down and harder to
see. The mamba layer mapping has to be **rebuilt for the TP view**, from the TP
device pool's own layer identity, and that is a real piece of work with its own
acceptance.

## Acceptance when it is built

1. `check_pool_coverage` admits the rebind (the tier covers every bound pool).
2. A mamba load-back after a `pp->tp` cutover restores state, measured — not
   "no crash": the discriminator is a cached-prefix resume that is byte-identical
   to the cache-miss answer, which is the `#767` harness's own criterion.
3. Layer coverage asserted directly: for every mamba layer in the TP view the
   mapper returns a non-`None` local id. A mapper that returns `None` everywhere
   passes every crash test and restores nothing.

Until then the correct state is the one the code already documents: the #718
device tier stays disarmed for the TP phase, every read-through misses, and a
miss is recomputed. That is a throughput cost, not a correctness one.
