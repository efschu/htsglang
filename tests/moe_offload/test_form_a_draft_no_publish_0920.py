"""fnFA11 (20.09. 13:40Z): the solo MTP draft on the Form A host must not
publish its MoE input to the workers -- they run no draft forward, so the
carrier all-reduce has no second participant (Bar1CollectiveAborted)."""

import inspect
import types

from sglang.srt.models import qwen4_exp as q


def test_layers_remember_is_nextn():
    src = inspect.getsource(q)
    assert src.count("self.is_nextn = bool(is_nextn)") == 2


def test_publish_is_skipped_on_the_draft_layer(monkeypatch):
    monkeypatch.setattr(q, "form_a_dense_is_unsharded", lambda: True)
    calls = []
    monkeypatch.setattr(q, "publish_moe_input", lambda h: calls.append("pub") or h)
    src = inspect.getsource(q.Qwen4ExpLayerExtensionMixin._run_qwen4_exp_mlp)
    assert 'if form_a_dense_is_unsharded() and not getattr(self, "is_nextn", False):' in src
    # the predicate pair, evaluated the way the layer evaluates it
    for is_nextn, want in ((True, []), (False, ["pub"])):
        calls.clear()
        layer = types.SimpleNamespace(is_nextn=is_nextn)
        if q.form_a_dense_is_unsharded() and not getattr(layer, "is_nextn", False):
            q.publish_moe_input(object())
        assert calls == want
