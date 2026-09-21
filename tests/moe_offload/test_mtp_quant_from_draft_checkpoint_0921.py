"""#66 (21.09.): the DRAFT checkpoint decides the MTP's quantization.

``_mtp_quant_config`` read the TARGET's safetensors index to learn whether
the draft ships dense.  With ``--speculative-draft-model-path`` the mtp
weights come from a different checkpoint entirely, and the two disagree in
both directions -- measured on this rig:

    Minachist (target): 1565 'mtp.' keys, ALL plain .weight, vocab PACKED
    albucino (draft):   4637 'mtp.' keys, 1536x weight_packed/_scale/_shape
                        (INT4 g32), vocab DENSE

So the target's bf16 mtp stump made the code call an INT4 draft "dense" and
build the whole module in bf16 (5.21 GB on TP0, fnFL2v65).
"""

import json
import types

import pytest

from sglang.srt.models import qwen3_5_mtp as m


def _ckpt(tmp_path, name, mtp_keys):
    d = tmp_path / name
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "a.safetensors" for k in mtp_keys}})
    )
    return str(d)


DENSE = [f"mtp.layers.0.{n}.weight" for n in ("q_proj", "k_proj")]
INT4 = [
    f"mtp.layers.0.q_proj.{s}" for s in ("weight_packed", "weight_scale", "weight_shape")
]


def test_the_index_reader_itself(tmp_path):
    assert m.mtp_index_is_dense(_ckpt(tmp_path, "dense", DENSE)) is True
    assert m.mtp_index_is_dense(_ckpt(tmp_path, "int4", INT4)) is False
    assert m.mtp_index_is_dense(None) is None
    assert m.mtp_index_is_dense(str(tmp_path / "nope")) is None


class _CT:
    def get_name(self):
        return "compressed-tensors"


@pytest.mark.parametrize(
    "target,draft,kept",
    [
        # the fnFL2 pair: bf16 stump in the target, INT4 in the draft
        (DENSE, INT4, True),
        # no separate draft: the target's own index still answers
        (DENSE, None, False),
        (INT4, None, True),
        # a dense draft beside a quantized target stays dense
        (INT4, DENSE, False),
    ],
)
def test_the_draft_path_wins(tmp_path, monkeypatch, target, draft, kept):
    t = _ckpt(tmp_path, f"t{len(tmp_path.name)}{target[0][-4:]}{kept}", target)
    d = _ckpt(tmp_path, f"d{len(tmp_path.name)}{kept}", draft) if draft else None
    monkeypatch.setattr(
        m,
        "get_server_args",
        lambda: types.SimpleNamespace(
            model_path=t,
            speculative_draft_model_path=d,
            speculative_draft_model_quantization=None,
        ),
    )
    monkeypatch.setattr(m, "is_npu", lambda: False)
    cfg = _CT()
    got = m._mtp_quant_config(cfg)
    assert (got is cfg) is kept, (target[0], draft and draft[0], got)
