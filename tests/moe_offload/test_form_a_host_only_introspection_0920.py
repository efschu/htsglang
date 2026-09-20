"""fnFA3 (20.09.): the loader's post-load pass does
``getattr(module, "quant_method", None)`` over every module; on a worker the
host-only placeholder must answer like a module without that attribute, while
direct use keeps refusing by name."""

import pytest

from sglang.srt.form_a_construction import FormAHostOnlyModuleUsed, HostOnlyModule


def test_introspection_with_a_default_sees_no_attribute():
    m = HostOnlyModule("embed_tokens", "model.language_model.embed_tokens")
    assert getattr(m, "quant_method", None) is None
    assert not hasattr(m, "weight")


def test_direct_use_still_refuses_by_name():
    m = HostOnlyModule("embed_tokens", "model.language_model.embed_tokens")
    with pytest.raises(FormAHostOnlyModuleUsed, match="embed_tokens"):
        m.forward()
    with pytest.raises(FormAHostOnlyModuleUsed):
        _ = m.weight
    assert issubclass(FormAHostOnlyModuleUsed, AttributeError)
