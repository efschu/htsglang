#!/usr/bin/env python3
"""Build a YaRN overlay model directory for #543.

Why an overlay directory and not a flag:

* ``--json-model-override-args`` cannot reach the rope block. The checkpoint is
  transformers-v5 style, so the block lives in ``text_config.rope_parameters``
  (aliased onto ``rope_scaling`` by ``configs/qwen3_5.py``). The merge is
  ``config.update(model_override_args)`` in ``utils/hf_transformers/config.py``,
  a flat setattr loop: a top-level ``rope_scaling`` never reaches the text
  config, and a nested ``{"text_config": {...}}`` REPLACES the
  ``Qwen3_5TextConfig`` object with a plain dict, after which
  ``model_runner.hybrid_gdn_config``'s isinstance check fails and the model is
  no longer treated as a GDN hybrid (no mamba pool at all).
* ``--decrypted-config-file`` looks like the supported route but was measured
  to have NO effect on this tree: booting the sibling file leaves
  ``rope_type=default`` and the derived context at 262144.

So the overlay is a directory of symlinks to the real checkpoint with a single
patched ``config.json``. Weights are not copied.

Two rules encoded here:

* keep ``mrope_section`` / ``mrope_interleaved``. The rope factory routes to
  ``YaRNScalingMRotaryEmbedding`` only when ``mrope_section`` is present, while
  ``model_runner.model_is_mrope`` is derived from the config either way, so
  dropping the keys desynchronises the two.
* omit ``original_max_position_embeddings``. ``get_context_length`` forces the
  scaling factor to 1 when that key is present, which collapses the derived
  context back to 262144 and makes ``--context-length`` raise. Omitting it lets
  the factory default ``original_max_position`` to ``max_position_embeddings``
  (262144), which is the value we want anyway -- the rope math is identical,
  only the context gate differs.
"""

import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--factor", type=float, required=True)
args = parser.parse_args()

src = pathlib.Path(args.model_dir).resolve()
dst = pathlib.Path(args.out_dir)
dst.mkdir(parents=True, exist_ok=True)

for entry in src.iterdir():
    if entry.name == "config.json":
        continue
    link = dst / entry.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(entry)

config = json.loads((src / "config.json").read_text())
text_config = config["text_config"]
rope = dict(text_config["rope_parameters"])
rope.pop("original_max_position_embeddings", None)
rope.update({"rope_type": "yarn", "factor": args.factor})
text_config["rope_parameters"] = rope
(dst / "config.json").write_text(json.dumps(config, indent=1))

native = text_config["max_position_embeddings"]
print(f"overlay: {dst}")
print(f"rope_parameters: {rope}")
print(f"derived context = {native} x {args.factor} = {int(native * args.factor)}")
