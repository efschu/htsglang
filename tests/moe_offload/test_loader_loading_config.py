"""fn8r3 20.09.: the safetensors prefetch branch of
DefaultModelLoader._get_weights_iterator read a name (``model_config``) that
does not exist in that generator's scope -- NameError at the first boot with
--weight-loader-prefetch-checkpoints. The config is parked on the loader by
_get_all_weights for the primary read and cleared afterwards."""
from types import SimpleNamespace

from sglang.srt.model_loader.loader import DefaultModelLoader


def _loader():
    ldr = DefaultModelLoader.__new__(DefaultModelLoader)
    ldr._weight_name_filter = None
    ldr._loading_model_config = None
    return ldr


def _cfg():
    return SimpleNamespace(
        model_path="/nonexistent", revision=None, is_draft_model=True
    )


def test_primary_read_sees_the_loading_config(monkeypatch):
    ldr = _loader()
    seen = []

    def fake_iter(self, source):
        seen.append(getattr(self, "_loading_model_config", None))
        yield ("w", None)

    monkeypatch.setattr(DefaultModelLoader, "_get_weights_iterator", fake_iter)
    monkeypatch.setattr(
        DefaultModelLoader.Source, "init_new", staticmethod(lambda cfg, m: object())
    )
    cfg = _cfg()
    model = SimpleNamespace(secondary_weights=())
    out = list(ldr._get_all_weights(cfg, model))
    assert out == [("w", None)]
    assert seen == [cfg]
    # cleared after the primary read: a later secondary/draft read must not
    # inherit the target's config
    assert ldr._loading_model_config is None
