"""Unit tests: fast-lane `lane` field on the OpenAI-compatible endpoints.

Protocol parsing (extra_body {"lane": "fast"} -> request model) and
pass-through into GenerateReqInput.lane for both /v1/completions and
/v1/chat/completions. CPU-only; no server.

Run: python -m pytest test/registered/unit/test_openai_lane_passthrough.py -q
"""

from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


def test_completion_request_lane_parses():
    r = CompletionRequest(model="m", prompt="hi", lane="fast")
    assert r.lane == "fast"
    assert CompletionRequest(model="m", prompt="hi").lane is None


def test_chat_request_lane_parses():
    msgs = [{"role": "user", "content": "hi"}]
    r = ChatCompletionRequest(model="m", messages=msgs, lane="fast")
    assert r.lane == "fast"
    assert ChatCompletionRequest(model="m", messages=msgs).lane is None


def test_lane_rejects_other_values():
    with pytest.raises(ValueError):
        CompletionRequest(model="m", prompt="hi", lane="heavy")
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], lane="slow"
        )


def _stub_serving(cls):
    """Bare serving object with the raw-request helpers stubbed out."""
    s = object.__new__(cls)
    s.template_manager = SimpleNamespace(
        completion_template_name=None,
        chat_template_name=None,
        jinja_template_content_format=None,
    )
    s.extract_custom_labels = lambda raw: None
    s.extract_routed_dp_rank_from_header = lambda raw, body: body
    s.extract_routing_key = lambda raw: None
    s._resolve_lora_path = lambda model, lora_path: None
    s._compute_extra_key = lambda req: None
    return s


def test_completions_passthrough_to_generate_req_input():
    from sglang.srt.entrypoints.openai.serving_completions import (
        OpenAIServingCompletion,
    )

    s = _stub_serving(OpenAIServingCompletion)
    req = CompletionRequest(model="m", prompt="hello", lane="fast")
    adapted, _ = s._convert_to_internal_request(req, raw_request=None)
    assert adapted.lane == "fast"

    req2 = CompletionRequest(model="m", prompt="hello")
    adapted2, _ = s._convert_to_internal_request(req2, raw_request=None)
    assert adapted2.lane is None


def test_generate_req_input_tokenized_carries_lane():
    # The scheduler consumes TokenizedGenerateReqInput.lane; make sure the
    # native conversion keeps carrying the field (io_struct contract).
    from sglang.srt.managers.io_struct import (
        GenerateReqInput,
        TokenizedGenerateReqInput,
    )

    g = GenerateReqInput(text="hi", lane="fast")
    assert g.lane == "fast"
    # msgspec Struct: field names in __struct_fields__
    assert "lane" in getattr(TokenizedGenerateReqInput, "__struct_fields__", ())


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            assert et is not None and issubclass(et, self.exc), "expected raise"
            return True

    pytest = SimpleNamespace(raises=_Raises)  # noqa: F811
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
    sys.exit(0)
