# Finding — text-only GGUF checkpoints still allocate the full vision tower

Found while bringing up Qwen3.6-35B-A3B-UD-Q4_K_XL (#651), but **not specific to
that checkpoint or that ticket**. Filed separately so it does not vanish into
the bring-up.

Status: **[DESK-PROVEN by code reading + arithmetic]**, never measured on a GPU.

## The waste

`python/sglang/srt/models/qwen3_vl.py:1242` constructs the vision tower
unconditionally:

```python
self.visual = Qwen3VLMoeVisionModel(
    config.vision_config,
    quant_config=None,          # dense bf16
    ...
)
```

There is no gate on whether the checkpoint actually carries vision weights, and
no gate on whether multimodal ended up enabled.

For a GGUF checkpoint this is always wasted, because GGUF backbones are
text-only in this fork: llama.cpp keeps the vision tower in a separate
`mmproj*.gguf`, and `model_config.py` **force-disables multimodal** whenever a
GGUF has no mmproj beside it (the #52 NaN-contamination guard). So the module is
built, occupies VRAM for the life of the process, and can never be reached — no
image request is accepted, and the automatic VLM image warmup is steered to the
text path.

## The cost

For the Qwen3.6-35B-A3B config (`vision_config`: depth 27, hidden 1152,
intermediate 4304, out_hidden 2048, 2304 position embeddings):

```
per block: qkv 1152x3456 + proj 1152x1152 + fc1 1152x4304 + fc2 4304x1152
           = 15.22M params
x 27 blocks + patch_embed + pos_embed + merger
           = ~429M params
           = 818 MiB at bf16, PER RANK
```

Under TP/PP that is 818 MiB **on every rank**, since the tower is not sharded.

On a rig whose standing VRAM rule is a 1024 MiB free corridor per card, an
unreachable 818 MiB per rank is roughly a whole card's reserve. On a laptop it
can be the difference between fits and does-not-fit.

## Suggested fix (not implemented)

Skip constructing `self.visual` when the model will not serve images — the
condition already exists and is computed earlier, so this is a gate, not new
logic. Candidate signals, in order of directness:

1. multimodal was force-disabled for this model (`model_config.py` already
   decides this and knows why);
2. the checkpoint is GGUF with `mmproj_path is None` (the adapter already
   resolves this and reports it);
3. `config.vision_config` present but no vision weights in the checkpoint.

Care is needed on two points, which is why this is filed rather than fixed:

- `Qwen3VLForConditionalGeneration` is a **shared** multimodal path used by
  non-GGUF checkpoints that genuinely need the tower. The gate must not change
  their behaviour.
- Anything that inspects `self.visual` unconditionally (weight loading,
  `named_parameters` sweeps, expert-location config, hibernate manifests) needs
  to tolerate its absence — a `PPMissingLayer`-style placeholder is probably
  safer than deleting the attribute.

## Verification owed before fixing

- Measure the actual allocation on-card (818 MiB is arithmetic from the config,
  not an observation).
- Confirm no non-GGUF multimodal regression.
- Confirm the GGUF **+ mmproj** path (where the tower IS loaded, via
  `Qwen35GGUFAdapter.vision_weights_iterator`) is untouched.
